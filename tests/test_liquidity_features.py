from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_amihud_falls_back_to_estimate_when_value_traded_missing() -> None:
    data = liquidity_frame(
        [10.0],
        values=[float("nan")],
    )
    data["value_traded_estimated"] = [5_000.0]

    features = add_liquidity_features(data)

    assert features.loc[0, "amihud_illiquidity_raw"] == pytest.approx(
        0.01 / 5_000.0
    )


def test_amihud_prefers_real_value_over_estimate() -> None:
    data = liquidity_frame(
        [10.0],
        values=[100_000.0],
    )
    data["value_traded_estimated"] = [5_000.0]

    features = add_liquidity_features(data)

    assert features.loc[0, "amihud_illiquidity_raw"] == pytest.approx(
        0.01 / 100_000.0
    )


def test_value_traded_z_20_falls_back_to_estimate() -> None:
    row_count = 25
    data = liquidity_frame(
        [10.0] * row_count,
        values=[float("nan")] * row_count,
    )
    data["value_traded_estimated"] = [
        100_000.0 + index for index in range(row_count)
    ]

    features = add_liquidity_features(data)

    assert not features["value_traded_z_20"].isna().all()
