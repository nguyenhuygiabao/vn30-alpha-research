from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features_foreign_flow import (
    ENHANCED_FB_FS_FEATURES,
    FOREIGN_FLOW_FEATURE_COLUMNS,
    add_foreign_flow_features,
)


def build_frames(
    *,
    buy_values: list[float],
    sell_values: list[float],
    closes: list[float] | None = None,
    value_traded: float = 100_000.0,
    ticker: str = "HPG",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_count = len(buy_values)
    dates = pd.bdate_range("2026-01-01", periods=row_count)

    if closes is None:
        closes = [25.0] * row_count

    market_rows = []
    flow_rows = []

    for date, close, buy_value, sell_value in zip(
        dates,
        closes,
        buy_values,
        sell_values,
    ):
        market_rows.append(
            {
                "date": date,
                "ticker": ticker,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "adjusted_close": close,
                "volume": 1_000.0,
                "value_traded": value_traded,
            }
        )
        flow_rows.append(
            {
                "date": date,
                "ticker": ticker,
                "foreign_buy_value": buy_value,
                "foreign_sell_value": sell_value,
            }
        )

    return pd.DataFrame(market_rows), pd.DataFrame(flow_rows)


def test_all_enhanced_features_are_produced() -> None:
    market, flow = build_frames(
        buy_values=[100.0] * 30,
        sell_values=[50.0] * 30,
    )

    features = add_foreign_flow_features(market, flow)

    for feature in ENHANCED_FB_FS_FEATURES:
        assert feature in features.columns, feature


def test_enhanced_features_are_in_the_declared_column_contract() -> None:
    for feature in ENHANCED_FB_FS_FEATURES:
        assert feature in FOREIGN_FLOW_FEATURE_COLUMNS


def test_imbalance_matches_hand_computed_ratio() -> None:
    # Constant buy=300, sell=100 over the window:
    #   net  = (300 - 100) * 5 = 1000
    #   gross = (300 + 100) * 5 = 2000
    #   imbalance = 1000 / 2000 = 0.5
    market, flow = build_frames(
        buy_values=[300.0] * 10,
        sell_values=[100.0] * 10,
    )

    features = add_foreign_flow_features(market, flow)
    imbalance = features["foreign_flow_imbalance_5d"].dropna()

    assert not imbalance.empty
    assert imbalance.eq(0.5).all()


def test_imbalance_is_bounded_and_signed_correctly() -> None:
    # Pure one-sided buying must saturate at +1, pure selling at -1.
    buy_only_market, buy_only_flow = build_frames(
        buy_values=[500.0] * 10,
        sell_values=[0.0] * 10,
    )
    sell_only_market, sell_only_flow = build_frames(
        buy_values=[0.0] * 10,
        sell_values=[500.0] * 10,
    )

    buy_only = add_foreign_flow_features(
        buy_only_market, buy_only_flow
    )["foreign_flow_imbalance_10d"].dropna()
    sell_only = add_foreign_flow_features(
        sell_only_market, sell_only_flow
    )["foreign_flow_imbalance_10d"].dropna()

    assert buy_only.eq(1.0).all()
    assert sell_only.eq(-1.0).all()


def test_imbalance_treats_no_foreign_activity_as_balanced_not_missing() -> None:
    # A window with genuinely zero foreign activity is balanced (0.0),
    # matching the existing 1d convention -- not a missing observation.
    market, flow = build_frames(
        buy_values=[0.0] * 10,
        sell_values=[0.0] * 10,
    )

    features = add_foreign_flow_features(market, flow)
    imbalance = features["foreign_flow_imbalance_3d"].dropna()

    assert not imbalance.empty
    assert imbalance.eq(0.0).all()


def test_imbalance_is_independent_of_traded_value() -> None:
    # The whole point of normalizing by gross flow rather than by traded
    # value: the contrast survives when vendor traded value is missing.
    market, flow = build_frames(
        buy_values=[300.0] * 10,
        sell_values=[100.0] * 10,
    )
    market["value_traded"] = np.nan

    features = add_foreign_flow_features(market, flow)
    imbalance = features["foreign_flow_imbalance_5d"].dropna()

    assert not imbalance.empty
    assert imbalance.eq(0.5).all()
    # The traded-value-normalized share must still be missing, confirming
    # the two constructs are genuinely independent.
    assert features["foreign_net_value_share_5d"].isna().all()


def test_return_correlation_detects_contrarian_flow() -> None:
    # Foreign buying concentrated on down days and selling on up days is
    # contrarian: imbalance should move opposite to return.
    row_count = 40
    closes = []
    buy_values = []
    sell_values = []
    price = 100.0

    for index in range(row_count):
        going_down = index % 2 == 0
        price = price * (0.98 if going_down else 1.02)
        closes.append(price)
        buy_values.append(400.0 if going_down else 100.0)
        sell_values.append(100.0 if going_down else 400.0)

    market, flow = build_frames(
        buy_values=buy_values,
        sell_values=sell_values,
        closes=closes,
    )

    features = add_foreign_flow_features(market, flow)
    correlation = features[
        "foreign_flow_return_corr_20d"
    ].dropna()

    assert not correlation.empty
    assert correlation.lt(-0.9).all()


def test_return_correlation_detects_momentum_flow() -> None:
    row_count = 40
    closes = []
    buy_values = []
    sell_values = []
    price = 100.0

    for index in range(row_count):
        going_down = index % 2 == 0
        price = price * (0.98 if going_down else 1.02)
        closes.append(price)
        # Buy on the way up instead of on the way down.
        buy_values.append(100.0 if going_down else 400.0)
        sell_values.append(400.0 if going_down else 100.0)

    market, flow = build_frames(
        buy_values=buy_values,
        sell_values=sell_values,
        closes=closes,
    )

    features = add_foreign_flow_features(market, flow)
    correlation = features[
        "foreign_flow_return_corr_20d"
    ].dropna()

    assert not correlation.empty
    assert correlation.gt(0.9).all()


def test_return_correlation_is_missing_not_infinite_when_flat() -> None:
    # A perfectly flat price series has zero return variance, so the
    # correlation is undefined. It must be missing, never infinite --
    # the combined-feature validator rejects infinities outright.
    market, flow = build_frames(
        buy_values=[300.0] * 40,
        sell_values=[100.0] * 40,
        closes=[25.0] * 40,
    )

    features = add_foreign_flow_features(market, flow)
    correlation = features["foreign_flow_return_corr_20d"]

    assert correlation.isna().all()
    assert not np.isinf(
        correlation.to_numpy(dtype="float64", na_value=np.nan)
    ).any()


def test_downday_share_matches_hand_computed_fraction() -> None:
    # 20-day window, alternating down/up days, foreign buying only on the
    # down days: every unit of buying lands on a down day, so the share
    # must be exactly 1.0.
    row_count = 40
    closes = []
    buy_values = []
    price = 100.0

    for index in range(row_count):
        going_down = index % 2 == 0
        price = price * (0.98 if going_down else 1.02)
        closes.append(price)
        buy_values.append(500.0 if going_down else 0.0)

    market, flow = build_frames(
        buy_values=buy_values,
        sell_values=[10.0] * row_count,
        closes=closes,
    )

    features = add_foreign_flow_features(market, flow)
    share = features["foreign_buy_downday_share_20d"].dropna()

    assert not share.empty
    assert share.eq(1.0).all()


def test_downday_share_is_zero_when_buying_only_on_up_days() -> None:
    row_count = 40
    closes = []
    buy_values = []
    price = 100.0

    for index in range(row_count):
        going_down = index % 2 == 0
        price = price * (0.98 if going_down else 1.02)
        closes.append(price)
        buy_values.append(0.0 if going_down else 500.0)

    market, flow = build_frames(
        buy_values=buy_values,
        sell_values=[10.0] * row_count,
        closes=closes,
    )

    features = add_foreign_flow_features(market, flow)
    share = features["foreign_buy_downday_share_20d"].dropna()

    assert not share.empty
    assert share.eq(0.0).all()


def test_downday_share_is_missing_when_no_foreign_buying() -> None:
    market, flow = build_frames(
        buy_values=[0.0] * 40,
        sell_values=[100.0] * 40,
        closes=[100.0 - index for index in range(40)],
    )

    features = add_foreign_flow_features(market, flow)

    assert features["foreign_buy_downday_share_20d"].isna().all()


def test_enhanced_features_never_produce_infinities() -> None:
    # Adversarial mix: zero flow, zero traded value, and a flat price run.
    row_count = 45
    buy_values = [0.0 if index % 3 == 0 else 250.0 for index in range(row_count)]
    sell_values = [0.0 if index % 3 == 0 else 250.0 for index in range(row_count)]
    closes = [
        25.0 if index < 20 else 25.0 + index for index in range(row_count)
    ]

    market, flow = build_frames(
        buy_values=buy_values,
        sell_values=sell_values,
        closes=closes,
    )
    market.loc[market.index[:5], "value_traded"] = 0.0

    features = add_foreign_flow_features(market, flow)

    for feature in ENHANCED_FB_FS_FEATURES:
        values = features[feature].to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
        assert not np.isinf(values).any(), feature


def test_enhanced_features_use_only_backward_looking_data() -> None:
    # Appending future rows must not change any already-computed value.
    # This is the leakage guard: a forward-looking window would shift
    # earlier rows when later data arrives.
    row_count = 40
    rng = np.random.RandomState(3)
    closes = list(100.0 + np.cumsum(rng.normal(size=row_count + 10)))
    buy_values = list(rng.uniform(100.0, 500.0, size=row_count + 10))
    sell_values = list(rng.uniform(100.0, 500.0, size=row_count + 10))

    short_market, short_flow = build_frames(
        buy_values=buy_values[:row_count],
        sell_values=sell_values[:row_count],
        closes=closes[:row_count],
    )
    long_market, long_flow = build_frames(
        buy_values=buy_values,
        sell_values=sell_values,
        closes=closes,
    )

    short_features = add_foreign_flow_features(
        short_market, short_flow
    )
    long_features = add_foreign_flow_features(
        long_market, long_flow
    )
    truncated = long_features.iloc[:row_count]

    for feature in ENHANCED_FB_FS_FEATURES:
        pd.testing.assert_series_equal(
            short_features[feature],
            truncated[feature],
            check_names=False,
        )


def test_rolling_windows_respect_minimum_periods() -> None:
    # A 20-day feature must not be defined before 20 observations exist.
    market, flow = build_frames(
        buy_values=[300.0] * 25,
        sell_values=[100.0] * 25,
        closes=[100.0 + index for index in range(25)],
    )

    features = add_foreign_flow_features(market, flow).sort_values(
        "date"
    )

    # The 20-day window also needs a prior close to form the first return,
    # so the first defined observation cannot appear before row 20.
    assert features["foreign_flow_imbalance_20d"].iloc[:19].isna().all()
    assert features["foreign_buy_downday_share_20d"].iloc[:19].isna().all()
    assert features["foreign_flow_return_corr_20d"].iloc[:19].isna().all()


def test_features_are_computed_independently_per_ticker() -> None:
    row_count = 30
    frames = []
    flow_frames = []

    for ticker, buy_value, sell_value in (
        ("AAA", 400.0, 100.0),
        ("BBB", 100.0, 400.0),
    ):
        market, flow = build_frames(
            buy_values=[buy_value] * row_count,
            sell_values=[sell_value] * row_count,
            ticker=ticker,
        )
        frames.append(market)
        flow_frames.append(flow)

    features = add_foreign_flow_features(
        pd.concat(frames, ignore_index=True),
        pd.concat(flow_frames, ignore_index=True),
    )

    aaa = features.loc[features["ticker"].eq("AAA")]
    bbb = features.loc[features["ticker"].eq("BBB")]

    assert aaa["foreign_flow_imbalance_5d"].dropna().eq(0.6).all()
    assert bbb["foreign_flow_imbalance_5d"].dropna().eq(-0.6).all()
