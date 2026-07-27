from __future__ import annotations

import numpy as np
import pandas as pd

from src.features_liquidity import add_liquidity_features


def liquidity_frame(
    volumes: list[float],
    values: list[float] | None = None,
) -> pd.DataFrame:
    row_count = len(volumes)
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=row_count,
                freq="D",
            ),
            "ticker": ["HPG"] * row_count,
            "volume": volumes,
            "value_traded": (
                values
                if values is not None
                else [100_000.0] * row_count
            ),
            "return_1d": [0.01] * row_count,
        }
    )


def test_zero_lagged_volume_does_not_create_infinity() -> None:
    features = add_liquidity_features(
        liquidity_frame(
            [0.0, 10.0, 10.0, 10.0, 10.0, 20.0]
        )
    )

    assert pd.isna(features.loc[5, "volume_change_5d"])
    assert not np.isinf(
        features["volume_change_5d"].dropna()
    ).any()


def test_zero_traded_value_does_not_create_infinity() -> None:
    features = add_liquidity_features(
        liquidity_frame(
            [10.0],
            values=[0.0],
        )
    )

    assert pd.isna(
        features.loc[0, "amihud_illiquidity_raw"]
    )
    assert pd.isna(
        features.loc[0, "amihud_illiquidity"]
    )
