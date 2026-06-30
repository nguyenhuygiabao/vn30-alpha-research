from __future__ import annotations

import pandas as pd


ROLLING_LIQUIDITY_WINDOW: int = 20
AMIHUD_WINSOR_LOWER_QUANTILE: float = 0.01
AMIHUD_WINSOR_UPPER_QUANTILE: float = 0.99


def calculate_rolling_z_score(
    values: pd.Series,
    window: int = ROLLING_LIQUIDITY_WINDOW,
) -> pd.Series:
    rolling_mean = values.rolling(
        window=window,
        min_periods=window,
    ).mean()

    rolling_std = values.rolling(
        window=window,
        min_periods=window,
    ).std()

    return (values - rolling_mean) / rolling_std


def winsorize_series(
    values: pd.Series,
    lower_quantile: float = AMIHUD_WINSOR_LOWER_QUANTILE,
    upper_quantile: float = AMIHUD_WINSOR_UPPER_QUANTILE,
) -> pd.Series:
    lower_bound = values.quantile(lower_quantile)
    upper_bound = values.quantile(upper_quantile)

    return values.clip(
        lower=lower_bound,
        upper=upper_bound,
    )


def add_liquidity_features(
    data: pd.DataFrame,
    window: int = ROLLING_LIQUIDITY_WINDOW,
    abnormal_volume_z_threshold: float = 2.0,
) -> pd.DataFrame:
    working = data.copy()

    working = working.sort_values(
        [
            "ticker",
            "date",
        ]
    )

    working["volume_z_20"] = working.groupby("ticker")["volume"].transform(
        lambda values: calculate_rolling_z_score(
            values=values,
            window=window,
        )
    )

    working["value_traded_z_20"] = working.groupby("ticker")[
        "value_traded"
    ].transform(
        lambda values: calculate_rolling_z_score(
            values=values,
            window=window,
        )
    )

    working["volume_change_5d"] = working.groupby("ticker")[
        "volume"
    ].pct_change(
        periods=5,
    )

    if "shares_outstanding" in working.columns:
        working["turnover"] = working["volume"] / working["shares_outstanding"]

        working["turnover_z_20"] = working.groupby("ticker")[
            "turnover"
        ].transform(
            lambda values: calculate_rolling_z_score(
                values=values,
                window=window,
            )
        )
    else:
        working["turnover_z_20"] = pd.NA

    working["amihud_illiquidity_raw"] = (
        working["return_1d"].abs() / working["value_traded"]
    )

    working["amihud_illiquidity"] = working.groupby("date")[
        "amihud_illiquidity_raw"
    ].transform(winsorize_series)

    working["abnormal_volume_flag"] = (
        working["volume_z_20"].abs() >= abnormal_volume_z_threshold
    )

    return working.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(drop=True)


def rank_stocks_by_liquidity(
    data: pd.DataFrame,
    date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    working = data.copy()

    if date is None:
        date = working["date"].max()

    latest_rows = working[
        working["date"] == date
    ].copy()

    latest_rows["value_traded_rank"] = latest_rows["value_traded"].rank(
        ascending=False,
        method="min",
    )

    latest_rows["volume_rank"] = latest_rows["volume"].rank(
        ascending=False,
        method="min",
    )

    latest_rows["amihud_rank"] = latest_rows["amihud_illiquidity"].rank(
        ascending=True,
        method="min",
    )

    latest_rows["liquidity_rank_score"] = (
        latest_rows["value_traded_rank"]
        + latest_rows["volume_rank"]
        + latest_rows["amihud_rank"]
    )

    latest_rows["liquidity_rank"] = latest_rows["liquidity_rank_score"].rank(
        ascending=True,
        method="min",
    )

    return latest_rows.sort_values(
        [
            "liquidity_rank",
            "ticker",
        ]
    ).reset_index(drop=True)
