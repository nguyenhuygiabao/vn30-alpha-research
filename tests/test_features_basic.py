from __future__ import annotations

import pandas as pd

from src.features_basic import build_basic_features


def basic_frame(
    *,
    value_traded: list[float | None],
    value_traded_estimated: list[float | None] | None = None,
) -> pd.DataFrame:
    row_count = len(value_traded)
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=row_count,
                freq="D",
            ),
            "ticker": ["FLC"] * row_count,
            "adjusted_close": [10.0 + index for index in range(row_count)],
            "volume": [1_000.0] * row_count,
            "value_traded": pd.array(value_traded, dtype="float64"),
        }
    )

    if value_traded_estimated is not None:
        frame["value_traded_estimated"] = value_traded_estimated
        frame["value_traded_is_estimated"] = frame["value_traded"].isna() & frame[
            "value_traded_estimated"
        ].notna()

    return frame


def test_average_daily_value_falls_back_to_estimate_when_value_traded_missing() -> None:
    data = basic_frame(
        value_traded=[None] * 20,
        value_traded_estimated=[50_000.0] * 20,
    )

    features = build_basic_features(data)

    assert features["average_daily_value_20d"].iloc[-1] == 50_000.0


def test_average_daily_value_prefers_real_value_over_estimate() -> None:
    data = basic_frame(
        value_traded=[100_000.0] * 20,
        value_traded_estimated=[50_000.0] * 20,
    )

    features = build_basic_features(data)

    assert features["average_daily_value_20d"].iloc[-1] == 100_000.0


def test_average_daily_value_stays_missing_without_estimate_column() -> None:
    data = basic_frame(value_traded=[None] * 20)

    features = build_basic_features(data)

    assert features["average_daily_value_20d"].isna().all()
