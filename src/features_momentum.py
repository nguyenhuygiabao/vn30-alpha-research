from __future__ import annotations
import pandas as pd

MOMENTUM_HORIZONS: tuple[int, ...] = (1,3,5,10,20,60)
REVERSAL_WINDOW: int = 20

def add_momentum_features (df: pd.DataFrame) -> pd.DataFrame: 
    result = df.sort_values(["ticker", "date"]).copy()

    grouped_adjusted_close = result.groupby("ticker")["adjusted_close"]

    for horizon in MOMENTUM_HORIZONS:
        previous_adjusted_close = grouped_adjusted_close.shift(horizon)

        result[f"return_{horizon}d"] = (
            result["adjusted_close"]/previous_adjusted_close -1)

    rolling_high_20d = (
        result
        .groupby("ticker")["adjusted_close"]
        .transform(
            lambda series: series.rolling(
                window = REVERSAL_WINDOW,
                min_periods= REVERSAL_WINDOW,
            ).max()
        )
    )

    rolling_low_20d = (
        result
        .groupby("ticker")["adjusted_close"]
        .transform(
            lambda series: series.rolling(
                window = REVERSAL_WINDOW,
                min_periods= REVERSAL_WINDOW,
            ).min()
        )
    )

    result["distance_from_20d_high"] = (
        result["adjusted_close"]/rolling_high_20d -1
    )

    result["distance_from_20d_low"] = (
        result["adjusted_close"]/rolling_low_20d -1
    )

    result["negative_shock_1d"] = (
        result["return_1d"].clip(upper = 0)
    )

    result["positive_shock_1d"] = (
        result["return_1d"].clip(lower = 0)
    )


    return result
