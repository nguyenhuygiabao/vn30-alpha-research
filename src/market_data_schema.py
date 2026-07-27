from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
)

ADJUSTED_OHLC_COLUMNS = (
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
)

CANONICAL_MARKET_COLUMNS = {
    "ticker",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
}

FIINQUANT_PRICE_SCALE = 1_000.0

MARKET_REQUIRED_COLUMNS = {
    "date",
    "ticker",
    *RAW_PRICE_COLUMNS,
    "adjusted_close",
    "volume",
    "value_traded",
}

FOREIGN_FLOW_REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "foreign_buy_value",
    "foreign_sell_value",
}

FOREIGN_FLOW_VOLUME_COLUMNS = {
    "foreign_buy_volume",
    "foreign_sell_volume",
}

CORPORATE_ACTION_REQUIRED_COLUMNS = {
    "ticker",
    "announcement_date",
    "effective_date",
    "action_type",
    "cash_amount_vnd",
    "share_ratio",
}

CORPORATE_ACTION_TYPES = {
    "cash_dividend",
    "stock_dividend",
    "stock_split",
    "reverse_split",
    "rights_issue",
}


def _standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    standardized = data.copy()
    standardized.columns = [
        str(column).strip().lower()
        for column in standardized.columns
    ]

    duplicated = standardized.columns[
        standardized.columns.duplicated()
    ].tolist()

    if duplicated:
        raise ValueError(
            "Data contain duplicate normalized columns: "
            f"{sorted(set(duplicated))}"
        )

    return standardized


def _require_columns(
    data: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )


def _normalize_ticker(data: pd.DataFrame) -> None:
    data["ticker"] = (
        data["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    invalid = (
        data["ticker"].isna()
        | data["ticker"].isin({"", "NAN", "NONE"})
    )

    if invalid.any():
        raise ValueError("Ticker values cannot be empty")


def _normalize_keys(
    data: pd.DataFrame,
    dataset_name: str,
) -> None:
    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    ).dt.normalize()
    _normalize_ticker(data)

    if data.duplicated(
        ["date", "ticker"],
        keep=False,
    ).any():
        raise ValueError(
            f"{dataset_name} contains duplicate ticker-date rows"
        )


def _convert_numeric(
    data: pd.DataFrame,
    columns: list[str] | set[str],
    dataset_name: str,
    *,
    allow_missing: set[str] | None = None,
) -> None:
    allowed = allow_missing or set()

    for column in sorted(set(columns).intersection(data.columns)):
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )
        missing = data[column].isna()

        if missing.any() and column not in allowed:
            raise ValueError(
                f"{dataset_name} contains missing {column}"
            )

        nonmissing = data.loc[~missing, column]

        if not np.isfinite(nonmissing).all():
            raise ValueError(
                f"{dataset_name} contains non-finite {column}"
            )


def _validate_nonnegative(
    data: pd.DataFrame,
    columns: set[str],
    dataset_name: str,
) -> None:
    negative_columns = [
        column
        for column in sorted(columns.intersection(data.columns))
        if data[column].lt(0).any()
    ]

    if negative_columns:
        raise ValueError(
            f"{dataset_name} contains negative values in: "
            f"{negative_columns}"
        )


def _validate_identity(
    data: pd.DataFrame,
    buy_column: str,
    sell_column: str,
    net_column: str,
    dataset_name: str,
) -> None:
    expected = data[buy_column] - data[sell_column]

    if not np.allclose(
        data[net_column],
        expected,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise ValueError(
            f"{net_column} is inconsistent: expected "
            f"{buy_column} - {sell_column} in {dataset_name}"
        )


def _load_canonical_market_csv(
    file_path: str | Path,
    dataset_name: str,
    *,
    allow_missing: set[str] | None = None,
) -> pd.DataFrame:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {path}"
        )

    canonical = _standardize_columns(pd.read_csv(path))
    _require_columns(
        canonical,
        CANONICAL_MARKET_COLUMNS,
        dataset_name,
    )

    normalized = canonical[
        sorted(CANONICAL_MARKET_COLUMNS)
    ].rename(
        columns={"timestamp": "date"}
    )
    _normalize_keys(normalized, dataset_name)

    numeric_columns = CANONICAL_MARKET_COLUMNS.difference(
        {"ticker", "timestamp"}
    )
    _convert_numeric(
        normalized,
        numeric_columns,
        dataset_name,
        allow_missing=allow_missing,
    )
    _validate_nonnegative(
        normalized,
        {"volume", "value"},
        dataset_name,
    )

    return normalized.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def load_canonical_market_history(
    raw_file_path: str | Path,
    adjusted_file_path: str | Path,
) -> pd.DataFrame:
    """
    Combine FiinQuant raw OHLC with adjusted close.

    Raw prices are used wherever available. FLC/ROS archive rows retain
    adjusted close and volume while unavailable raw OHLC and traded value
    remain missing. No value is inferred from adjusted prices.
    """
    raw = _load_canonical_market_csv(
        raw_file_path,
        "Canonical raw market data",
    )
    adjusted = _load_canonical_market_csv(
        adjusted_file_path,
        "Canonical adjusted market data",
        allow_missing={"value"},
    )

    key_columns = ["date", "ticker"]
    raw_only = raw[key_columns].merge(
        adjusted[key_columns],
        on=key_columns,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    raw_only = raw_only.loc[
        raw_only["_merge"].ne("both")
    ]

    if not raw_only.empty:
        raise ValueError(
            "Canonical raw market rows lack adjusted counterparts:\n"
            + raw_only.head(10).to_string(index=False)
        )

    value_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
    ]
    combined = adjusted.merge(
        raw[key_columns + value_columns],
        on=key_columns,
        how="left",
        suffixes=("_adjusted", "_raw"),
        indicator=True,
        validate="one_to_one",
    )
    has_raw = combined["_merge"].eq("both")
    missing_adjusted_value = combined[
        "value_adjusted"
    ].isna()

    missing_value_with_raw = (
        has_raw & missing_adjusted_value
    )

    if missing_value_with_raw.any():
        examples = combined.loc[
            missing_value_with_raw,
            key_columns,
        ].head(10)
        raise ValueError(
            "Canonical adjusted traded value is missing "
            "despite available raw market data:\n"
            + examples.to_string(index=False)
        )

    for column in ("volume", "value"):
        comparable = (
            has_raw
            & combined[f"{column}_raw"].notna()
            & combined[f"{column}_adjusted"].notna()
        )
        matching = np.isclose(
            combined.loc[
                comparable,
                f"{column}_raw",
            ],
            combined.loc[
                comparable,
                f"{column}_adjusted",
            ],
            rtol=0.0,
            atol=1e-6,
        )

        if not matching.all():
            bad_indices = combined.index[
                comparable
            ][~matching]
            examples = combined.loc[
                bad_indices,
                key_columns
                + [
                    f"{column}_raw",
                    f"{column}_adjusted",
                ],
            ].head(10)
            raise ValueError(
                "Canonical raw and adjusted market data "
                f"disagree on {column}:\n"
                + examples.to_string(index=False)
            )

    market = combined[key_columns].copy()

    for column in RAW_PRICE_COLUMNS:
        market[column] = (
            combined[f"{column}_raw"]
            / FIINQUANT_PRICE_SCALE
        )

    market["adjusted_close"] = (
        combined["close_adjusted"]
        / FIINQUANT_PRICE_SCALE
    )
    market["volume"] = combined["volume_adjusted"]
    market["value_traded"] = (
        combined["value_adjusted"]
    )

    market = validate_market_data_frame(
        market,
        allow_missing_raw_prices=True,
        allow_missing_value_traded=True,
        allow_raw_price_exceptions=True,
    )

    fallback = market[
        list(RAW_PRICE_COLUMNS)
    ].isna().all(axis=1)
    missing_value = market["value_traded"].isna()

    market.attrs.update(
        {
            "raw_price_fallback_rows": int(fallback.sum()),
            "raw_price_fallback_tickers": sorted(
                market.loc[
                    fallback,
                    "ticker",
                ].unique().tolist()
            ),
            "missing_value_traded_rows": int(
                missing_value.sum()
            ),
            "missing_value_traded_tickers": sorted(
                market.loc[
                    missing_value,
                    "ticker",
                ].unique().tolist()
            ),
            "raw_market_rows": len(raw),
            "adjusted_market_rows": len(adjusted),
            "price_unit": "thousand_vnd",
        }
    )

    return market


def validate_market_data_frame(
    data: pd.DataFrame,
    *,
    require_adjusted_ohlc: bool = False,
    allow_missing_raw_prices: bool = False,
    allow_missing_value_traded: bool = False,
    allow_raw_price_exceptions: bool = False,
) -> pd.DataFrame:
    validated = _standardize_columns(data)

    _require_columns(
        validated,
        MARKET_REQUIRED_COLUMNS,
        "Market data",
    )

    optional_adjusted = {
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
    }
    present_adjusted = optional_adjusted.intersection(
        validated.columns
    )

    if present_adjusted and present_adjusted != optional_adjusted:
        raise ValueError(
            "Adjusted OHLC must contain adjusted_open, "
            "adjusted_high, and adjusted_low together"
        )

    if (
        require_adjusted_ohlc
        and present_adjusted != optional_adjusted
    ):
        raise ValueError(
            "Complete adjusted OHLC prices are required"
        )

    _normalize_keys(validated, "Market data")

    numeric_columns = [
        *RAW_PRICE_COLUMNS,
        "adjusted_close",
        "volume",
        "value_traded",
        *sorted(present_adjusted),
    ]
    allowed_missing: set[str] = set()

    if allow_missing_raw_prices:
        allowed_missing.update(RAW_PRICE_COLUMNS)

    if allow_missing_value_traded:
        allowed_missing.add("value_traded")

    _convert_numeric(
        validated,
        numeric_columns,
        "Market data",
        allow_missing=allowed_missing,
    )

    raw_missing = validated[
        list(RAW_PRICE_COLUMNS)
    ].isna()

    if allow_missing_raw_prices:
        partial_raw = (
            raw_missing.any(axis=1)
            & ~raw_missing.all(axis=1)
        )

        if partial_raw.any():
            raise ValueError(
                "Raw OHLC must be either complete or entirely "
                "missing for each market row"
            )
    elif raw_missing.any().any():
        raise ValueError(
            "Market data contains missing raw OHLC"
        )

    raw_available = ~raw_missing.any(axis=1)
    adjusted_price_columns = [
        "adjusted_close",
        *sorted(present_adjusted),
    ]

    if validated[
        adjusted_price_columns
    ].le(0).any().any():
        raise ValueError(
            "Adjusted market prices must be positive"
        )

    if (
        not allow_raw_price_exceptions
        and validated.loc[
            raw_available,
            list(RAW_PRICE_COLUMNS),
        ].le(0).any().any()
    ):
        raise ValueError(
            "Raw market prices must be positive"
        )

    if validated["volume"].lt(0).any():
        raise ValueError(
            "Volume and traded value cannot be negative"
        )

    if validated["value_traded"].lt(0).any():
        raise ValueError(
            "Volume and traded value cannot be negative"
        )

    if not allow_raw_price_exceptions:
        raw_range_invalid = raw_available & (
            validated["high"].lt(validated["low"])
            | validated["open"].lt(validated["low"])
            | validated["open"].gt(validated["high"])
            | validated["close"].lt(validated["low"])
            | validated["close"].gt(validated["high"])
        )

        if raw_range_invalid.any():
            raise ValueError(
                "Raw OHLC prices contain an invalid daily range"
            )

    if present_adjusted:
        adjusted_range_invalid = (
            validated["adjusted_high"].lt(
                validated["adjusted_low"]
            )
            | validated["adjusted_open"].lt(
                validated["adjusted_low"]
            )
            | validated["adjusted_open"].gt(
                validated["adjusted_high"]
            )
            | validated["adjusted_close"].lt(
                validated["adjusted_low"]
            )
            | validated["adjusted_close"].gt(
                validated["adjusted_high"]
            )
        )

        if adjusted_range_invalid.any():
            raise ValueError(
                "Adjusted OHLC prices contain an invalid daily range"
            )

    return validated.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def build_execution_market_view(
    data: pd.DataFrame,
) -> pd.DataFrame:
    validated = validate_market_data_frame(data)

    return validated[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "value_traded",
        ]
    ].copy()


def validate_foreign_flow_frame(
    data: pd.DataFrame,
) -> pd.DataFrame:
    validated = _standardize_columns(data)

    _require_columns(
        validated,
        FOREIGN_FLOW_REQUIRED_COLUMNS,
        "Foreign-flow data",
    )
    _normalize_keys(validated, "Foreign-flow data")

    present_volume_columns = (
        FOREIGN_FLOW_VOLUME_COLUMNS.intersection(
            validated.columns
        )
    )

    if (
        present_volume_columns
        and present_volume_columns
        != FOREIGN_FLOW_VOLUME_COLUMNS
    ):
        raise ValueError(
            "Foreign volume data must contain both "
            "foreign_buy_volume and foreign_sell_volume"
        )

    numeric_columns = [
        "foreign_buy_value",
        "foreign_sell_value",
        *sorted(present_volume_columns),
    ]
    _convert_numeric(
        validated,
        numeric_columns,
        "Foreign-flow data",
    )
    _validate_nonnegative(
        validated,
        set(numeric_columns),
        "Foreign-flow data",
    )

    derived_net_value = (
        validated["foreign_buy_value"]
        - validated["foreign_sell_value"]
    )

    if "foreign_net_value" in validated.columns:
        _convert_numeric(
            validated,
            ["foreign_net_value"],
            "Foreign-flow data",
        )
        _validate_identity(
            validated,
            "foreign_buy_value",
            "foreign_sell_value",
            "foreign_net_value",
            "Foreign-flow data",
        )

    validated["foreign_net_value"] = (
        derived_net_value
    )

    if present_volume_columns:
        derived_net_volume = (
            validated["foreign_buy_volume"]
            - validated["foreign_sell_volume"]
        )

        if "foreign_net_volume" in validated.columns:
            _convert_numeric(
                validated,
                ["foreign_net_volume"],
                "Foreign-flow data",
            )
            _validate_identity(
                validated,
                "foreign_buy_volume",
                "foreign_sell_volume",
                "foreign_net_volume",
                "Foreign-flow data",
            )

        validated["foreign_net_volume"] = (
            derived_net_volume
        )
    elif "foreign_net_volume" in validated.columns:
        raise ValueError(
            "foreign_net_volume requires foreign buy "
            "and sell volume"
        )

    return validated.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def validate_corporate_actions_frame(
    data: pd.DataFrame,
) -> pd.DataFrame:
    validated = _standardize_columns(data)

    _require_columns(
        validated,
        CORPORATE_ACTION_REQUIRED_COLUMNS,
        "Corporate-action data",
    )
    _normalize_ticker(validated)

    for column in (
        "announcement_date",
        "effective_date",
    ):
        validated[column] = pd.to_datetime(
            validated[column],
            errors="raise",
        ).dt.normalize()

    validated["action_type"] = (
        validated["action_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    unknown_types = sorted(
        set(validated["action_type"])
        .difference(CORPORATE_ACTION_TYPES)
    )

    if unknown_types:
        raise ValueError(
            f"Unknown corporate-action types: {unknown_types}"
        )

    _convert_numeric(
        validated,
        [
            "cash_amount_vnd",
            "share_ratio",
        ],
        "Corporate-action data",
    )

    if validated[
        ["cash_amount_vnd", "share_ratio"]
    ].lt(0).any().any():
        raise ValueError(
            "Corporate-action values cannot be negative"
        )

    if validated["announcement_date"].gt(
        validated["effective_date"]
    ).any():
        raise ValueError(
            "Corporate action cannot be announced "
            "after its effective date"
        )

    cash_dividends = validated[
        "action_type"
    ].eq("cash_dividend")

    if validated.loc[
        cash_dividends,
        "cash_amount_vnd",
    ].le(0).any():
        raise ValueError(
            "Cash dividends require a positive cash amount"
        )

    share_actions = validated[
        "action_type"
    ].isin(
        {
            "stock_dividend",
            "stock_split",
            "reverse_split",
            "rights_issue",
        }
    )

    if validated.loc[
        share_actions,
        "share_ratio",
    ].le(0).any():
        raise ValueError(
            "Share actions require a positive share ratio"
        )

    if validated.duplicated(
        [
            "ticker",
            "effective_date",
            "action_type",
        ],
        keep=False,
    ).any():
        raise ValueError(
            "Corporate-action data contains duplicate events"
        )

    return validated.sort_values(
        [
            "ticker",
            "effective_date",
            "action_type",
        ]
    ).reset_index(drop=True)
