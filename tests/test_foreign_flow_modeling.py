from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

import scripts.run_foreign_flow_experiment as experiment
from src.feature_pipeline import build_combined_features
from src.features_foreign_flow import (
    ALL_FOREIGN_FLOW_LEARNING_FEATURES,
    ENHANCED_FB_FS_FEATURES,
    FB_FS_FOREIGN_FLOW_FEATURES,
    FOREIGN_FLOW_FEATURE_COLUMNS,
    FROZEN_FOREIGN_FLOW_FEATURES,
)
from src.foreign_flow_modeling import (
    FEATURE_SET_ALL_FOREIGN,
    FEATURE_SET_ENHANCED_FB_FS,
    FEATURE_SET_FB_FS,
    FEATURE_SET_FROZEN_FOREIGN,
    FEATURE_SET_MARKET_BASELINE,
    FROZEN_MARKET_BASELINE_FEATURES,
    assert_market_baseline_parity,
    build_canonical_modeling_dataset,
    build_common_calendar_forward_labels,
    build_daily_foreign_flow_metrics,
    build_foreign_flow_feature_sets,
    build_modeling_sample,
    build_paper_horizon_signal_schedule,
    build_production_market_baseline_reference,
    run_foreign_flow_paper_horizon,
    summarize_foreign_flow_metrics,
    summarize_paired_foreign_flow_differences,
    validate_foreign_flow_feature_sets,
)
from src.modeling_utils import get_model_feature_columns


def comparison_frame(
    *,
    date_count: int = 40,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2020-01-01", periods=date_count)

    for date_index, market_date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            cross_section = float(ticker_index - 1)
            row: dict[str, object] = {
                "date": market_date,
                "ticker": ticker,
                "forward_relative_return_10d": (
                    0.01 * cross_section
                    + 0.0001 * date_index
                ),
                "foreign_flow_available": 1,
                "future_unapproved_feature": 999.0,
                "forward_return_10d": 999.0,
            }

            for feature_index, feature in enumerate(
                FROZEN_MARKET_BASELINE_FEATURES,
                start=1,
            ):
                row[feature] = (
                    0.05 * cross_section
                    + 0.001 * feature_index
                    + 0.0002 * date_index
                )

            for feature_index, feature in enumerate(
                FOREIGN_FLOW_FEATURE_COLUMNS,
                start=1,
            ):
                row[feature] = (
                    0.1 * cross_section
                    + 0.001 * feature_index
                    + 0.0001 * date_index
                )

            rows.append(row)

    return pd.DataFrame(rows)


def test_feature_sets_freeze_exact_51_feature_baseline() -> None:
    modeling = comparison_frame(date_count=2)
    feature_sets = build_foreign_flow_feature_sets(modeling)

    assert tuple(feature_sets) == (
        FEATURE_SET_MARKET_BASELINE,
        FEATURE_SET_FROZEN_FOREIGN,
        FEATURE_SET_FB_FS,
        FEATURE_SET_ALL_FOREIGN,
        FEATURE_SET_ENHANCED_FB_FS,
    )
    assert feature_sets[FEATURE_SET_MARKET_BASELINE] == (
        FROZEN_MARKET_BASELINE_FEATURES
    )
    # The four original sets keep their exact published sizes, so the
    # already-committed comparison stays reproducible after the fifth set
    # was appended.
    assert len(
        feature_sets[FEATURE_SET_MARKET_BASELINE]
    ) == 51
    assert len(
        feature_sets[FEATURE_SET_FROZEN_FOREIGN]
    ) == 60
    assert len(feature_sets[FEATURE_SET_FB_FS]) == 64
    assert len(feature_sets[FEATURE_SET_ALL_FOREIGN]) == 73
    assert len(
        feature_sets[FEATURE_SET_ENHANCED_FB_FS]
    ) == 66
    assert "future_unapproved_feature" not in feature_sets[
        FEATURE_SET_MARKET_BASELINE
    ]
    assert "forward_return_10d" not in feature_sets[
        FEATURE_SET_MARKET_BASELINE
    ]
    assert set(FROZEN_FOREIGN_FLOW_FEATURES).issubset(
        feature_sets[FEATURE_SET_FROZEN_FOREIGN]
    )
    assert set(FB_FS_FOREIGN_FLOW_FEATURES).issubset(
        feature_sets[FEATURE_SET_FB_FS]
    )
    assert set(ENHANCED_FB_FS_FEATURES).issubset(
        feature_sets[FEATURE_SET_ENHANCED_FB_FS]
    )
    # The enhanced set must build on the frozen foreign set, not replace it.
    assert set(FROZEN_FOREIGN_FLOW_FEATURES).issubset(
        feature_sets[FEATURE_SET_ENHANCED_FB_FS]
    )


def test_frozen_baseline_matches_canonical_feature_schema() -> None:
    market_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2025-01-01", periods=70)

    for ticker_index, ticker in enumerate(("AAA", "BBB")):
        for date_index, market_date in enumerate(dates):
            close = (
                20.0
                + 2.0 * ticker_index
                + 0.01 * date_index
            )
            volume = (
                100_000
                + 1_000 * ticker_index
                + 10 * date_index
            )
            buy_value = (
                1_000_000
                + 1_000 * date_index
                + 10_000 * ticker_index
            )
            sell_value = (
                900_000
                + 900 * date_index
                + 8_000 * ticker_index
            )
            market_rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adjusted_close": close,
                    "volume": volume,
                    "value_traded": volume * close,
                }
            )
            flow_rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "foreign_buy_value": buy_value,
                    "foreign_sell_value": sell_value,
                    "foreign_net_value": (
                        buy_value - sell_value
                    ),
                }
            )

    features = build_combined_features(
        pd.DataFrame(market_rows),
        pd.DataFrame(flow_rows),
    )
    discovered_market = tuple(
        column
        for column in get_model_feature_columns(features)
        if not column.startswith("foreign_")
    )

    assert discovered_market == FROZEN_MARKET_BASELINE_FEATURES


def test_feature_set_validation_rejects_contract_drift() -> None:
    modeling = comparison_frame(date_count=2)
    feature_sets = build_foreign_flow_feature_sets(modeling)
    altered = dict(feature_sets)
    altered[FEATURE_SET_MARKET_BASELINE] = (
        altered[FEATURE_SET_MARKET_BASELINE]
        + ("future_unapproved_feature",)
    )

    with pytest.raises(
        ValueError,
        match="explicit frozen feature definition",
    ):
        validate_foreign_flow_feature_sets(
            modeling,
            altered,
        )


def test_modeling_sample_preserves_missing_features() -> None:
    modeling = comparison_frame(
        date_count=2,
        tickers=("AAA", "BBB"),
    )
    feature_sets = build_foreign_flow_feature_sets(modeling)
    second_date = modeling["date"].max()
    missing_market_index = modeling.index[
        modeling["date"].eq(second_date)
        & modeling["ticker"].eq("AAA")
    ][0]
    missing_foreign_index = modeling.index[
        modeling["date"].eq(second_date)
        & modeling["ticker"].eq("BBB")
    ][0]
    modeling.loc[
        missing_market_index,
        FROZEN_MARKET_BASELINE_FEATURES[0],
    ] = np.nan
    modeling.loc[
        missing_foreign_index,
        FB_FS_FOREIGN_FLOW_FEATURES[0],
    ] = np.nan

    sample, eligible_dates, audit = build_modeling_sample(
        modeling,
        feature_sets,
        horizon_days=10,
    )

    assert len(sample) == len(modeling)
    assert len(eligible_dates) == 2
    assert audit["feature_missing_rows_excluded"] == 0
    assert audit["excluded_target_rows"] == 0
    assert pd.isna(
        sample.loc[
            sample["ticker"].eq("AAA")
            & sample["date"].eq(second_date),
            FROZEN_MARKET_BASELINE_FEATURES[0],
        ].iloc[0]
    )
    assert (
        audit["missing_rows_by_feature_set"][
            FEATURE_SET_ALL_FOREIGN
        ]
        == 2
    )


def test_modeling_sample_excludes_targets_not_features() -> None:
    modeling = comparison_frame(
        date_count=2,
        tickers=("AAA", "BBB"),
    )
    feature_sets = build_foreign_flow_feature_sets(modeling)
    second_date = modeling["date"].max()
    modeling.loc[
        modeling["date"].eq(second_date)
        & modeling["ticker"].eq("BBB"),
        "forward_relative_return_10d",
    ] = np.nan

    sample, eligible_dates, audit = build_modeling_sample(
        modeling,
        feature_sets,
        horizon_days=10,
    )

    assert len(sample) == len(modeling) - 1
    assert eligible_dates.tolist() == [
        modeling["date"].min()
    ]
    assert audit["excluded_target_rows"] == 1


def test_common_calendar_label_does_not_jump_suspension() -> None:
    dates = pd.bdate_range("2022-01-03", periods=13)
    rows: list[dict[str, object]] = []

    for ticker, growth in (
        ("AAA", 0.01),
        ("BBB", 0.00),
    ):
        for date_index, market_date in enumerate(dates):
            if ticker == "AAA" and market_date == dates[10]:
                continue

            rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "adjusted_close": (
                        10.0 * (1.0 + growth) ** date_index
                    ),
                }
            )

    labels = build_common_calendar_forward_labels(
        pd.DataFrame(rows)
    )
    aaa_first = labels.loc[
        labels["date"].eq(dates[0])
        & labels["ticker"].eq("AAA")
    ].iloc[0]
    bbb_first = labels.loc[
        labels["date"].eq(dates[0])
        & labels["ticker"].eq("BBB")
    ].iloc[0]

    assert pd.isna(aaa_first["forward_return_10d"])
    assert bbb_first["forward_return_10d"] == pytest.approx(
        0.0
    )


def test_canonical_dataset_uses_point_in_time_common_calendar_labels() -> None:
    dates = pd.bdate_range("2022-01-03", periods=13)
    market_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for ticker, daily_growth in (
        ("AAA", 0.01),
        ("BBB", 0.0),
        ("CCC", -0.005),
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
            "ticker": ["AAA", "BBB", "CCC"],
            "effective_from": [
                dates[0],
                dates[0],
                dates[1],
            ],
            "effective_to": ["", "", ""],
        }
    )
    modeling = build_canonical_modeling_dataset(
        features=pd.DataFrame(feature_rows),
        market_data=pd.DataFrame(market_rows),
        membership=membership,
        horizon_days=10,
    )
    first_date = modeling.loc[
        modeling["date"].eq(dates[0])
    ]
    aaa = first_date.loc[
        first_date["ticker"].eq("AAA")
    ].iloc[0]
    ccc = first_date.loc[
        first_date["ticker"].eq("CCC")
    ].iloc[0]

    assert aaa["forward_relative_return_10d"] > 0
    assert pd.isna(ccc["forward_relative_return_10d"])


def test_signal_schedule_matches_paper_horizon_contract() -> None:
    dates = pd.bdate_range("2021-01-01", periods=40)
    signals = build_paper_horizon_signal_schedule(
        dates,
        horizon_days=10,
        minimum_training_dates=8,
        signal_step_days=10,
        evaluation_end=dates[30],
    )

    assert [signal.signal_date for signal in signals] == [
        dates[17],
        dates[27],
    ]
    assert [
        signal.training_cutoff
        for signal in signals
    ] == [
        dates[7],
        dates[17],
    ]


def test_walk_forward_keeps_rows_and_matches_feature_coverage() -> None:
    modeling = comparison_frame()
    modeling.loc[
        modeling["date"].eq(modeling["date"].iloc[5])
        & modeling["ticker"].eq("AAA"),
        FROZEN_MARKET_BASELINE_FEATURES[0],
    ] = np.nan
    modeling.loc[
        modeling["date"].eq(modeling["date"].iloc[17])
        & modeling["ticker"].eq("BBB"),
        FB_FS_FOREIGN_FLOW_FEATURES[0],
    ] = np.nan
    feature_sets = build_foreign_flow_feature_sets(modeling)
    sample, eligible_dates, audit = build_modeling_sample(
        modeling,
        feature_sets,
        horizon_days=10,
    )
    signals = build_paper_horizon_signal_schedule(
        pd.DatetimeIndex(
            modeling["date"].drop_duplicates()
        ),
        horizon_days=10,
        minimum_training_dates=8,
        signal_step_days=10,
    )
    predictions, importance = run_foreign_flow_paper_horizon(
        modeling_sample=sample,
        eligible_signal_dates=eligible_dates,
        feature_sets=feature_sets,
        signals=signals,
        horizon_days=10,
        model_factory=lambda: GradientBoostingRegressor(
            n_estimators=5,
            max_depth=1,
            random_state=42,
        ),
    )

    assert audit["feature_missing_rows_excluded"] == 0
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
    assert set(
        predictions.groupby("feature_set")[
            "feature_count"
        ].first()
    ) == {51, 60, 64, 66, 73}
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
    # Four original comparisons plus the two enhanced-set comparisons.
    assert len(paired) == 6
    assert paired["paired_dates"].gt(1).all()


def test_market_baseline_reproduces_production_paper_history() -> None:
    modeling = comparison_frame(date_count=32)
    feature_sets = build_foreign_flow_feature_sets(modeling)
    sample, eligible_dates, _ = build_modeling_sample(
        modeling,
        feature_sets,
        horizon_days=10,
    )
    signals = build_paper_horizon_signal_schedule(
        pd.DatetimeIndex(
            modeling["date"].drop_duplicates()
        ),
        horizon_days=10,
        minimum_training_dates=8,
        signal_step_days=10,
    )
    baseline_predictions, _ = (
        run_foreign_flow_paper_horizon(
            modeling_sample=sample,
            eligible_signal_dates=eligible_dates,
            feature_sets={
                FEATURE_SET_MARKET_BASELINE: feature_sets[
                    FEATURE_SET_MARKET_BASELINE
                ]
            },
            signals=signals,
            horizon_days=10,
        )
    )
    production_reference = (
        build_production_market_baseline_reference(
            modeling,
            horizon_days=10,
            minimum_training_dates=8,
            signal_step_days=10,
        )
    )
    audit = assert_market_baseline_parity(
        baseline_predictions,
        production_reference,
    )

    assert audit["status"] == "passed"
    assert audit["signal_dates"] == 2
    assert audit["maximum_prediction_difference"] == 0.0


def test_paper_horizon_progress_reports_runnable_signals() -> None:
    modeling = comparison_frame(date_count=32)
    feature_sets = build_foreign_flow_feature_sets(modeling)
    sample, eligible_dates, _ = build_modeling_sample(
        modeling,
        feature_sets,
        horizon_days=10,
    )
    signals = build_paper_horizon_signal_schedule(
        pd.DatetimeIndex(
            modeling["date"].drop_duplicates()
        ),
        horizon_days=10,
        minimum_training_dates=8,
        signal_step_days=10,
    )
    reported: list[tuple[int, int, pd.Timestamp]] = []

    run_foreign_flow_paper_horizon(
        modeling_sample=sample,
        eligible_signal_dates=eligible_dates,
        feature_sets={
            FEATURE_SET_MARKET_BASELINE: feature_sets[
                FEATURE_SET_MARKET_BASELINE
            ]
        },
        signals=signals,
        horizon_days=10,
        model_factory=lambda: GradientBoostingRegressor(
            n_estimators=2,
            max_depth=1,
            random_state=42,
        ),
        max_workers=2,
        progress_callback=lambda completed, total, signal: (
            reported.append(
                (
                    completed,
                    total,
                    signal.signal_date,
                )
            )
        ),
    )

    assert reported == [
        (1, 2, signals[0].signal_date),
        (2, 2, signals[1].signal_date),
    ]


def test_rebuild_script_writes_guarded_development_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    dates = pd.bdate_range("2020-01-01", periods=42)
    features = comparison_frame(date_count=42).drop(
        columns=[
            "forward_relative_return_10d",
            "forward_return_10d",
        ]
    )
    market_rows: list[dict[str, object]] = []

    for ticker, daily_growth in (
        ("AAA", 0.01),
        ("BBB", 0.0),
        ("CCC", -0.005),
    ):
        for date_index, market_date in enumerate(dates):
            market_rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "adjusted_close": (
                        10.0
                        * (1.0 + daily_growth) ** date_index
                    ),
                }
            )

    market = pd.DataFrame(market_rows)
    membership = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "effective_from": [dates[0]] * 3,
            "effective_to": ["", "", ""],
        }
    )
    features_path = tmp_path / "features.parquet"
    output_dir = tmp_path / "reports"
    predictions_dir = tmp_path / "data"
    features.to_parquet(features_path, index=False)

    monkeypatch.setattr(
        experiment,
        "load_canonical_market_history",
        lambda raw, adjusted: market.copy(),
    )
    monkeypatch.setattr(
        experiment,
        "load_verified_membership_history",
        lambda history_dir: membership.copy(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_foreign_flow_experiment.py",
            "--minimum-training-dates",
            "8",
            "--features-path",
            str(features_path),
            "--research-end",
            dates[-1].date().isoformat(),
            "--top-n",
            "2",
            "--bootstrap-samples",
            "20",
            "--output-dir",
            str(output_dir),
            "--predictions-dir",
            str(predictions_dir),
        ],
    )

    experiment.main()

    manifest = json.loads(
        (
            output_dir
            / "foreign_flow_rebuild_development_manifest.json"
        ).read_text(encoding="utf-8")
    )
    summary = pd.read_csv(
        output_dir
        / "foreign_flow_rebuild_development_summary.csv"
    )
    predictions = pd.read_parquet(
        predictions_dir
        / "foreign_flow_rebuild_development_predictions.parquet"
    )

    assert manifest["analysis_class"] == (
        "historical_development_only"
    )
    assert manifest["baseline_parity"]["status"] == "passed"
    assert not manifest["complete_case_filtering_used"]
    assert set(summary["feature_set"]) == {
        FEATURE_SET_MARKET_BASELINE,
        FEATURE_SET_FROZEN_FOREIGN,
        FEATURE_SET_FB_FS,
        FEATURE_SET_ALL_FOREIGN,
        FEATURE_SET_ENHANCED_FB_FS,
    }
    assert predictions["date"].max() <= dates[-1]
