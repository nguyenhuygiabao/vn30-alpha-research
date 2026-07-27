from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features_foreign_flow import (
    ALL_FOREIGN_FLOW_LEARNING_FEATURES,
    FB_FS_FOREIGN_FLOW_FEATURES,
    FROZEN_FOREIGN_FLOW_FEATURES,
)
from src.foreign_flow_modeling import (
    FEATURE_SET_ALL_FOREIGN,
    FEATURE_SET_FB_FS,
    FEATURE_SET_FROZEN_FOREIGN,
    FEATURE_SET_MARKET_BASELINE,
    build_canonical_modeling_dataset,
    build_common_complete_case_sample,
    build_daily_foreign_flow_metrics,
    build_expanding_walk_forward_windows,
    build_foreign_flow_feature_sets,
    run_foreign_flow_walk_forward,
    summarize_foreign_flow_metrics,
    summarize_paired_foreign_flow_differences,
)


def comparison_frame(
    *,
    date_count: int = 36,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2020-01-01", periods=date_count)

    for date_index, market_date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            cross_section = ticker_index - 1
            row: dict[str, object] = {
                "date": market_date,
                "ticker": ticker,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.1,
                "adjusted_close": 10.1,
                "volume": 1_000.0,
                "value_traded": 10_000.0,
                "market_signal": (
                    cross_section
                    + 0.01 * date_index
                ),
                "foreign_flow_available": 1,
                "foreign_buy_value": 600.0,
                "forward_relative_return_10d": (
                    0.01 * cross_section
                    + 0.0001 * date_index
                ),
            }

            for feature_index, feature in enumerate(
                ALL_FOREIGN_FLOW_LEARNING_FEATURES,
                start=1,
            ):
                row[feature] = (
                    0.1 * cross_section
                    + 0.001 * feature_index
                    + 0.0001 * date_index
                )

            rows.append(row)

    return pd.DataFrame(rows)


def test_feature_sets_exclude_raw_foreign_columns_and_target() -> None:
    modeling = comparison_frame(date_count=2)
    feature_sets = build_foreign_flow_feature_sets(modeling)

    assert tuple(feature_sets) == (
        FEATURE_SET_MARKET_BASELINE,
        FEATURE_SET_FROZEN_FOREIGN,
        FEATURE_SET_FB_FS,
        FEATURE_SET_ALL_FOREIGN,
    )
    assert "market_signal" in feature_sets[
        FEATURE_SET_MARKET_BASELINE
    ]
    assert "foreign_buy_value" not in feature_sets[
        FEATURE_SET_ALL_FOREIGN
    ]
    assert "forward_relative_return_10d" not in feature_sets[
        FEATURE_SET_MARKET_BASELINE
    ]
    assert set(FROZEN_FOREIGN_FLOW_FEATURES).issubset(
        feature_sets[FEATURE_SET_FROZEN_FOREIGN]
    )
    assert set(FB_FS_FOREIGN_FLOW_FEATURES).issubset(
        feature_sets[FEATURE_SET_FB_FS]
    )
    assert (
        set(feature_sets[FEATURE_SET_ALL_FOREIGN])
        .intersection(ALL_FOREIGN_FLOW_LEARNING_FEATURES)
        == set(ALL_FOREIGN_FLOW_LEARNING_FEATURES)
    )


def test_common_sample_excludes_missing_foreign_flow_without_zero_fill() -> None:
    modeling = comparison_frame(
        date_count=2,
        tickers=("AAA", "BBB"),
    )
    missing_index = modeling.index[
        modeling["date"].eq(modeling["date"].max())
        & modeling["ticker"].eq("BBB")
    ][0]
    missing_feature = FB_FS_FOREIGN_FLOW_FEATURES[0]
    modeling.loc[missing_index, missing_feature] = np.nan
    feature_sets = build_foreign_flow_feature_sets(modeling)

    sample, complete_dates, audit = (
        build_common_complete_case_sample(
            modeling,
            feature_sets,
            horizon_days=10,
        )
    )

    assert len(sample) == 3
    assert missing_index not in sample.index
    assert len(complete_dates) == 1
    assert complete_dates[0] == modeling["date"].min()
    assert audit["excluded_rows"] == 1
    assert not sample[missing_feature].isna().any()
    assert (
        sample.loc[
            sample["date"].eq(modeling["date"].max()),
            "ticker",
        ].tolist()
        == ["AAA"]
    )


def test_expanding_windows_keep_full_purge_before_test() -> None:
    dates = pd.bdate_range("2021-01-01", periods=24)
    windows = build_expanding_walk_forward_windows(
        dates,
        initial_training_dates=6,
        purge_dates=3,
        test_block_dates=4,
        evaluation_start=dates[9],
        evaluation_end=dates[20],
    )

    assert windows

    for window in windows:
        assert len(window.purge_dates) == 3
        assert window.train_dates.max() < window.purge_dates.min()
        assert window.purge_dates.max() < window.test_dates.min()
        assert window.test_dates.min() >= dates[9]
        assert window.test_dates.max() <= dates[20]

    assert len(windows[1].train_dates) > len(
        windows[0].train_dates
    )


def test_walk_forward_comparison_has_matched_coverage() -> None:
    modeling = comparison_frame()
    feature_sets = build_foreign_flow_feature_sets(modeling)
    sample, complete_dates, _ = (
        build_common_complete_case_sample(
            modeling,
            feature_sets,
            horizon_days=10,
        )
    )
    windows = build_expanding_walk_forward_windows(
        pd.DatetimeIndex(
            modeling["date"].drop_duplicates()
        ),
        initial_training_dates=12,
        purge_dates=10,
        test_block_dates=7,
    )
    predictions, importance = run_foreign_flow_walk_forward(
        common_sample=sample,
        complete_signal_dates=complete_dates,
        feature_sets=feature_sets,
        windows=windows,
        horizon_days=10,
        model_factory=lambda: GradientBoostingRegressor(
            n_estimators=5,
            max_depth=1,
            random_state=42,
        ),
    )

    assert set(predictions["feature_set"]) == set(
        feature_sets
    )
    assert not predictions[
        ["predicted_return", "actual_return"]
    ].isna().any().any()
    coverage = predictions.groupby(
        ["date", "ticker"]
    )["feature_set"].nunique()
    assert coverage.eq(len(feature_sets)).all()
    assert not importance.empty

    daily = build_daily_foreign_flow_metrics(
        predictions,
        top_n=2,
    )
    summary = summarize_foreign_flow_metrics(daily)
    paired = summarize_paired_foreign_flow_differences(
        daily,
        bootstrap_samples=50,
        block_size=2,
    )

    assert set(summary["feature_set"]) == set(
        feature_sets
    )
    assert len(paired) == 4
    assert paired["paired_dates"].gt(1).all()


def test_canonical_dataset_uses_point_in_time_ten_day_labels() -> None:
    dates = pd.bdate_range("2022-01-03", periods=15)
    market_rows = []
    feature_rows = []

    for ticker, daily_growth in (
        ("AAA", 0.01),
        ("BBB", 0.0),
    ):
        for offset, market_date in enumerate(dates):
            adjusted_close = 10.0 * (
                (1.0 + daily_growth) ** offset
            )
            market_rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "adjusted_close": adjusted_close,
                }
            )
            feature_rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "market_signal": float(offset),
                }
            )

    membership = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "effective_from": [dates[0], dates[0]],
            "effective_to": ["", ""],
        }
    )
    modeling = build_canonical_modeling_dataset(
        features=pd.DataFrame(feature_rows),
        market_data=pd.DataFrame(market_rows),
        membership=membership,
        horizon_days=10,
    )

    valid = modeling[
        "forward_relative_return_10d"
    ].notna()
    assert len(modeling) == 30
    assert int(valid.sum()) == 10
    aaa_first = modeling.loc[
        modeling["date"].eq(dates[0])
        & modeling["ticker"].eq("AAA")
    ].iloc[0]
    assert aaa_first[
        "forward_relative_return_10d"
    ] > 0
