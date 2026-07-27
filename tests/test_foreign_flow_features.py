from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features_foreign_flow import (
    ALL_FOREIGN_FLOW_LEARNING_FEATURES,
    FB_FS_FOREIGN_FLOW_FEATURES,
    FROZEN_FOREIGN_FLOW_FEATURES,
    add_foreign_flow_features,
    load_canonical_foreign_flow_csv,
)


def market_frame() -> pd.DataFrame:
    rows = []

    for ticker, base in (
        ("HPG", 25.0),
        ("FPT", 100.0),
    ):
        for offset, date in enumerate(
            pd.date_range(
                "2026-07-01",
                periods=5,
                freq="D",
            )
        ):
            close = base + offset

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000,
                    "value_traded": 100_000,
                }
            )

    return pd.DataFrame(rows)


def foreign_flow_frame() -> pd.DataFrame:
    rows = []
    hpg_net_values = [
        10_000,
        20_000,
        30_000,
        -10_000,
        40_000,
    ]
    fpt_net_values = [
        -10_000,
        -5_000,
        5_000,
        10_000,
        20_000,
    ]

    for ticker, values in (
        ("HPG", hpg_net_values),
        ("FPT", fpt_net_values),
    ):
        for date, net_value in zip(
            pd.date_range(
                "2026-07-01",
                periods=5,
                freq="D",
            ),
            values,
        ):
            buy_value = 50_000 + max(net_value, 0)
            sell_value = 50_000 + max(-net_value, 0)

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "foreign_buy_volume": 600,
                    "foreign_sell_volume": 400,
                    "foreign_buy_value": buy_value,
                    "foreign_sell_value": sell_value,
                }
            )

    return pd.DataFrame(rows)


def test_canonical_loader_maps_vendor_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign_flow_history.csv"
    pd.DataFrame(
        {
            "ticker": [" hpg ", "FPT"],
            "timestamp": [
                "2026-07-01",
                "2026-07-01",
            ],
            "fb": [0, 60_000],
            "fs": [0, 50_000],
            "fn": [0, 10_000],
        }
    ).to_csv(path, index=False)

    loaded = load_canonical_foreign_flow_csv(path)
    zero_row = loaded.loc[
        loaded["ticker"].eq("HPG")
    ].iloc[0]

    assert set(loaded.columns) == {
        "date",
        "ticker",
        "foreign_buy_value",
        "foreign_sell_value",
        "foreign_net_value",
    }
    assert loaded["date"].dtype == "datetime64[ns]"
    assert zero_row["foreign_buy_value"] == 0
    assert zero_row["foreign_sell_value"] == 0
    assert zero_row["foreign_net_value"] == 0


def test_canonical_loader_rejects_bad_net_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign_flow_history.csv"
    pd.DataFrame(
        {
            "ticker": ["HPG"],
            "timestamp": ["2026-07-01"],
            "fb": [60_000],
            "fs": [50_000],
            "fn": [999],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="foreign_net_value is inconsistent",
    ):
        load_canonical_foreign_flow_csv(path)


def test_canonical_loader_rejects_duplicate_normalized_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign_flow_history.csv"
    pd.DataFrame(
        [["HPG", "2026-07-01", 60_000, 50_000, 10_000, 60_000]],
        columns=["ticker", "timestamp", "fb", "fs", "fn", " FB "],
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="duplicate normalized columns",
    ):
        load_canonical_foreign_flow_csv(path)


def test_foreign_flow_features_preserve_market_rows() -> None:
    features = add_foreign_flow_features(
        market_frame(),
        foreign_flow_frame(),
    )

    assert len(features) == 10
    assert features[
        ["date", "ticker"]
    ].duplicated().sum() == 0


def test_one_day_foreign_flow_ratios_are_correct() -> None:
    features = add_foreign_flow_features(
        market_frame(),
        foreign_flow_frame(),
    )
    row = features.loc[
        features["date"].eq("2026-07-01")
        & features["ticker"].eq("HPG")
    ].iloc[0]

    assert row["foreign_net_value"] == 10_000
    assert row["foreign_net_value_share_1d"] == 0.10
    assert row["foreign_gross_value_share_1d"] == 1.10
    assert row["foreign_buy_value_share_1d"] == 0.60
    assert row["foreign_sell_value_share_1d"] == 0.50
    assert row["foreign_net_volume_share_1d"] == 0.20


def test_three_day_rolling_flow_uses_only_past_and_current_rows() -> None:
    features = add_foreign_flow_features(
        market_frame(),
        foreign_flow_frame(),
    )
    row = features.loc[
        features["date"].eq("2026-07-03")
        & features["ticker"].eq("HPG")
    ].iloc[0]

    assert row["foreign_net_value_3d"] == 60_000
    assert row["foreign_net_value_share_3d"] == 0.20
    assert row["foreign_buy_value_share_3d"] == 0.70
    assert row["foreign_sell_value_share_3d"] == 0.50


def test_future_flow_does_not_change_earlier_features() -> None:
    original_flow = foreign_flow_frame()
    original = add_foreign_flow_features(
        market_frame(),
        original_flow,
    )
    changed_flow = original_flow.copy()
    changed_flow.loc[
        changed_flow["date"].eq("2026-07-05"),
        "foreign_buy_value",
    ] = 9_999_999
    changed = add_foreign_flow_features(
        market_frame(),
        changed_flow,
    )

    comparison_columns = [
        "foreign_net_value_share_3d",
        "foreign_buy_value_share_3d",
        "foreign_sell_value_share_3d",
    ]
    earlier_original = original.loc[
        original["date"].lt("2026-07-05"),
        comparison_columns,
    ].reset_index(drop=True)
    earlier_changed = changed.loc[
        changed["date"].lt("2026-07-05"),
        comparison_columns,
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        earlier_original,
        earlier_changed,
    )


def test_missing_flow_is_flagged_not_converted_to_zero() -> None:
    flow = foreign_flow_frame()
    flow = flow.loc[
        ~(
            flow["date"].eq("2026-07-01")
            & flow["ticker"].eq("HPG")
        )
    ]
    features = add_foreign_flow_features(
        market_frame(),
        flow,
    )
    row = features.loc[
        features["date"].eq("2026-07-01")
        & features["ticker"].eq("HPG")
    ].iloc[0]

    assert row["foreign_flow_available"] == 0
    assert pd.isna(row["foreign_buy_value"])
    assert pd.isna(row["foreign_sell_value"])
    assert pd.isna(row["foreign_net_value"])
    assert pd.isna(row["foreign_buy_value_share_1d"])
    assert pd.isna(row["foreign_sell_value_share_1d"])
    assert pd.isna(row["foreign_net_value_share_1d"])


def test_genuine_zero_flow_is_available_and_preserved() -> None:
    flow = foreign_flow_frame()
    zero_row = (
        flow["date"].eq("2026-07-01")
        & flow["ticker"].eq("HPG")
    )
    flow.loc[
        zero_row,
        [
            "foreign_buy_value",
            "foreign_sell_value",
        ],
    ] = 0
    features = add_foreign_flow_features(
        market_frame(),
        flow,
    )
    row = features.loc[
        features["date"].eq("2026-07-01")
        & features["ticker"].eq("HPG")
    ].iloc[0]

    assert row["foreign_flow_available"] == 1
    assert row["foreign_buy_value"] == 0
    assert row["foreign_sell_value"] == 0
    assert row["foreign_net_value"] == 0
    assert row["foreign_gross_value"] == 0
    assert row["foreign_buy_value_share_1d"] == 0
    assert row["foreign_sell_value_share_1d"] == 0
    assert row["foreign_net_value_share_1d"] == 0


def test_missing_turnover_is_not_inferred_from_adjusted_price() -> None:
    market = market_frame().iloc[[0]].copy()
    market.loc[
        :,
        ["open", "high", "low", "close"],
    ] = float("nan")
    market["value_traded"] = float("nan")
    flow = foreign_flow_frame().loc[
        lambda data: (
            data["date"].eq(market.iloc[0]["date"])
            & data["ticker"].eq(market.iloc[0]["ticker"])
        )
    ]

    features = add_foreign_flow_features(
        market,
        flow,
    )
    row = features.iloc[0]

    assert row["foreign_flow_available"] == 1
    assert pd.isna(row["value_traded"])
    assert pd.isna(row["foreign_buy_value_share_1d"])
    assert pd.isna(row["foreign_sell_value_share_1d"])
    assert pd.isna(row["foreign_net_value_share_1d"])


def test_missing_flow_breaks_complete_rolling_window() -> None:
    flow = foreign_flow_frame()
    flow = flow.loc[
        ~(
            flow["date"].eq("2026-07-02")
            & flow["ticker"].eq("HPG")
        )
    ]
    features = add_foreign_flow_features(
        market_frame(),
        flow,
    )
    row = features.loc[
        features["date"].eq("2026-07-03")
        & features["ticker"].eq("HPG")
    ].iloc[0]

    assert pd.isna(
        row["foreign_net_value_share_3d"]
    )
    assert pd.isna(
        row["foreign_buy_value_share_3d"]
    )
    assert pd.isna(
        row["foreign_sell_value_share_3d"]
    )


def test_cross_sectional_ranks_reward_larger_values() -> None:
    features = add_foreign_flow_features(
        market_frame(),
        foreign_flow_frame(),
    )
    day = features.loc[
        features["date"].eq("2026-07-01")
    ].set_index("ticker")

    assert (
        day.loc["HPG", "foreign_net_value_rank_pct"]
        > day.loc["FPT", "foreign_net_value_rank_pct"]
    )
    assert (
        day.loc["HPG", "foreign_buy_value_rank_pct"]
        > day.loc["FPT", "foreign_buy_value_rank_pct"]
    )
    assert (
        day.loc["HPG", "foreign_sell_value_rank_pct"]
        < day.loc["FPT", "foreign_sell_value_rank_pct"]
    )


def test_five_day_buying_persistence_is_calculated() -> None:
    features = add_foreign_flow_features(
        market_frame(),
        foreign_flow_frame(),
    )
    row = features.loc[
        features["date"].eq("2026-07-05")
        & features["ticker"].eq("HPG")
    ].iloc[0]

    assert row["foreign_net_buy_days_5d"] == 4
    assert (
        row["foreign_net_buy_persistence_5d"]
        == 0.8
    )


def test_value_only_flow_builds_value_features() -> None:
    flow = foreign_flow_frame().drop(
        columns=[
            "foreign_buy_volume",
            "foreign_sell_volume",
        ]
    )
    features = add_foreign_flow_features(
        market_frame(),
        flow,
    )

    assert "foreign_net_value_share_1d" in features
    assert "foreign_buy_value_share_1d" in features
    assert "foreign_sell_value_share_1d" in features
    assert (
        "foreign_net_volume_share_1d"
        not in features
    )


def test_learning_feature_groups_are_explicit_and_disjoint() -> None:
    assert len(FROZEN_FOREIGN_FLOW_FEATURES) == 9
    assert len(FB_FS_FOREIGN_FLOW_FEATURES) == 14
    assert (
        set(FROZEN_FOREIGN_FLOW_FEATURES)
        .isdisjoint(FB_FS_FOREIGN_FLOW_FEATURES)
    )
    assert ALL_FOREIGN_FLOW_LEARNING_FEATURES == (
        FROZEN_FOREIGN_FLOW_FEATURES
        + FB_FS_FOREIGN_FLOW_FEATURES
    )

    features = add_foreign_flow_features(
        market_frame(),
        foreign_flow_frame(),
    )
    assert set(
        ALL_FOREIGN_FLOW_LEARNING_FEATURES
    ).issubset(features.columns)
