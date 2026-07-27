from __future__ import annotations

import pandas as pd

from src.market_data_schema import (
    validate_foreign_flow_frame,
    validate_market_data_frame,
)


FIINQUANT_PRICE_DIVISOR = 1_000.0

RAW_MARKET_COLUMNS = {
    "ticker",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
}

ADJUSTED_MARKET_COLUMNS = {
    "ticker",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
}

FOREIGN_FLOW_COLUMNS = {
    "ticker",
    "timestamp",
    "fb",
    "fs",
}


def _standardize_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    standardized = data.copy()

    standardized.columns = [
        str(column).strip().lower()
        for column in standardized.columns
    ]

    return standardized


def _require_columns(
    data: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    missing = sorted(
        required.difference(data.columns)
    )

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )


def _normalize_keys(
    data: pd.DataFrame,
) -> pd.DataFrame:
    normalized = data.copy()

    normalized["date"] = pd.to_datetime(
        normalized["timestamp"],
        errors="raise",
    ).dt.normalize()

    normalized["ticker"] = (
        normalized["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return normalized


def normalize_fiinquant_market_data(
    raw_data: pd.DataFrame,
    adjusted_data: pd.DataFrame,
    *,
    price_divisor: float = FIINQUANT_PRICE_DIVISOR,
) -> pd.DataFrame:
    if price_divisor <= 0:
        raise ValueError(
            "Price divisor must be positive"
        )

    raw = _normalize_keys(
        _standardize_columns(raw_data)
    )

    adjusted = _normalize_keys(
        _standardize_columns(adjusted_data)
    )

    _require_columns(
        raw,
        RAW_MARKET_COLUMNS,
        "Raw FiinQuant market data",
    )

    _require_columns(
        adjusted,
        ADJUSTED_MARKET_COLUMNS,
        "Adjusted FiinQuant market data",
    )

    for name, frame in (
        ("Raw", raw),
        ("Adjusted", adjusted),
    ):
        if frame.duplicated(
            ["date", "ticker"],
            keep=False,
        ).any():
            raise ValueError(
                f"{name} FiinQuant data contains "
                "duplicate ticker-date rows"
            )

    raw_market = raw[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "value",
        ]
    ].rename(
        columns={
            "value": "value_traded",
        }
    )

    adjusted_market = adjusted[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
        ]
    ].rename(
        columns={
            "open": "adjusted_open",
            "high": "adjusted_high",
            "low": "adjusted_low",
            "close": "adjusted_close",
        }
    )

    combined = raw_market.merge(
        adjusted_market,
        on=["date", "ticker"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = combined["_merge"].ne("both")

    if unmatched.any():
        examples = combined.loc[
            unmatched,
            ["date", "ticker", "_merge"],
        ].head(5)

        raise ValueError(
            "Raw and adjusted FiinQuant rows do not "
            f"match:\n{examples.to_string(index=False)}"
        )

    combined = combined.drop(
        columns=["_merge"]
    )

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    ]

    for column in price_columns:
        combined[column] = (
            pd.to_numeric(
                combined[column],
                errors="raise",
            )
            / price_divisor
        )

    combined["volume"] = pd.to_numeric(
        combined["volume"],
        errors="raise",
    )

    combined["value_traded"] = pd.to_numeric(
        combined["value_traded"],
        errors="raise",
    )

    return validate_market_data_frame(
        combined,
        require_adjusted_ohlc=True,
    )


def normalize_fiinquant_foreign_flow(
    data: pd.DataFrame,
) -> pd.DataFrame:
    flow = _normalize_keys(
        _standardize_columns(data)
    )

    _require_columns(
        flow,
        FOREIGN_FLOW_COLUMNS,
        "FiinQuant foreign-flow data",
    )

    if flow.duplicated(
        ["date", "ticker"],
        keep=False,
    ).any():
        raise ValueError(
            "FiinQuant foreign-flow data contains "
            "duplicate ticker-date rows"
        )

    output_columns = [
        "date",
        "ticker",
        "fb",
        "fs",
    ]

    if "fn" in flow.columns:
        output_columns.append("fn")

    normalized = flow[
        output_columns
    ].rename(
        columns={
            "fb": "foreign_buy_value",
            "fs": "foreign_sell_value",
            "fn": "foreign_net_value",
        }
    )

    return validate_foreign_flow_frame(
        normalized
    )
