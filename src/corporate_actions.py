from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_data_schema import (
    validate_market_data_frame,
)


REQUIRED_FACTOR_COLUMNS = {
    "date",
    "ticker",
    "price_adjustment_factor",
}


def validate_adjustment_factors_frame(
    data: pd.DataFrame,
) -> pd.DataFrame:
    factors = data.copy()

    factors.columns = [
        str(column).strip().lower()
        for column in factors.columns
    ]

    missing = sorted(
        REQUIRED_FACTOR_COLUMNS.difference(
            factors.columns
        )
    )

    if missing:
        raise ValueError(
            f"Adjustment factors are missing columns: {missing}"
        )

    factors["date"] = pd.to_datetime(
        factors["date"],
        errors="raise",
    ).dt.normalize()

    factors["ticker"] = (
        factors["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_tickers = factors["ticker"].isin(
        {"", "NAN", "NONE"}
    )

    if invalid_tickers.any():
        raise ValueError(
            "Adjustment-factor tickers cannot be empty"
        )

    if factors.duplicated(
        ["date", "ticker"],
        keep=False,
    ).any():
        raise ValueError(
            "Adjustment factors contain duplicate "
            "ticker-date rows"
        )

    factors["price_adjustment_factor"] = (
        pd.to_numeric(
            factors["price_adjustment_factor"],
            errors="raise",
        )
    )

    if (
        ~np.isfinite(
            factors["price_adjustment_factor"]
        )
    ).any():
        raise ValueError(
            "Price-adjustment factors must be finite"
        )

    if factors[
        "price_adjustment_factor"
    ].le(0).any():
        raise ValueError(
            "Price-adjustment factors must be positive"
        )

    if "volume_adjustment_factor" in factors.columns:
        factors["volume_adjustment_factor"] = (
            pd.to_numeric(
                factors["volume_adjustment_factor"],
                errors="raise",
            )
        )

        if (
            ~np.isfinite(
                factors["volume_adjustment_factor"]
            )
        ).any():
            raise ValueError(
                "Volume-adjustment factors must be finite"
            )

        if factors[
            "volume_adjustment_factor"
        ].le(0).any():
            raise ValueError(
                "Volume-adjustment factors must be positive"
            )

    if "source" not in factors.columns:
        factors["source"] = "unspecified"

    factors["source"] = (
        factors["source"]
        .astype(str)
        .str.strip()
    )

    return factors.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def apply_adjustment_factors(
    market_data: pd.DataFrame,
    adjustment_factors: pd.DataFrame,
    *,
    require_full_coverage: bool = True,
) -> pd.DataFrame:
    market = validate_market_data_frame(
        market_data
    )

    factors = validate_adjustment_factors_frame(
        adjustment_factors
    )

    factor_columns = [
        "date",
        "ticker",
        "price_adjustment_factor",
        "source",
    ]

    if "volume_adjustment_factor" in factors.columns:
        factor_columns.append(
            "volume_adjustment_factor"
        )

    adjusted = market.merge(
        factors[factor_columns],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    missing_factor = adjusted[
        "price_adjustment_factor"
    ].isna()

    if require_full_coverage and missing_factor.any():
        examples = (
            adjusted.loc[
                missing_factor,
                ["date", "ticker"],
            ]
            .head(5)
            .to_dict(orient="records")
        )

        raise ValueError(
            "Adjustment factors do not cover every "
            f"market row. Examples: {examples}"
        )

    adjusted["price_adjustment_factor"] = (
        adjusted["price_adjustment_factor"]
        .fillna(1.0)
    )

    adjusted["source"] = (
        adjusted["source"]
        .fillna("unadjusted")
    )

    for raw_column in (
        "open",
        "high",
        "low",
        "close",
    ):
        adjusted_column = (
            f"adjusted_{raw_column}"
        )

        adjusted[adjusted_column] = (
            adjusted[raw_column]
            * adjusted["price_adjustment_factor"]
        )

    if "volume_adjustment_factor" in adjusted.columns:
        adjusted["volume_adjustment_factor"] = (
            adjusted["volume_adjustment_factor"]
            .fillna(1.0)
        )

        adjusted["adjusted_volume"] = (
            adjusted["volume"]
            * adjusted["volume_adjustment_factor"]
        )

    # Revalidate the completed adjusted OHLC structure.
    adjusted = validate_market_data_frame(
        adjusted,
        require_adjusted_ohlc=True,
    )

    return adjusted


def build_adjustment_audit(
    adjusted_market_data: pd.DataFrame,
    *,
    jump_threshold: float = 0.10,
) -> pd.DataFrame:
    data = validate_market_data_frame(
        adjusted_market_data,
        require_adjusted_ohlc=True,
    )

    audit = data[
        [
            "date",
            "ticker",
            "close",
            "adjusted_close",
        ]
    ].copy()

    audit["raw_return_1d"] = (
        audit.groupby("ticker")["close"]
        .pct_change(fill_method=None)
    )

    audit["adjusted_return_1d"] = (
        audit.groupby("ticker")["adjusted_close"]
        .pct_change(fill_method=None)
    )

    audit["raw_jump"] = (
        audit["raw_return_1d"].abs()
        > jump_threshold
    )

    audit["adjusted_jump"] = (
        audit["adjusted_return_1d"].abs()
        > jump_threshold
    )

    audit["corporate_action_jump_removed"] = (
        audit["raw_jump"]
        & ~audit["adjusted_jump"]
    )

    return audit
