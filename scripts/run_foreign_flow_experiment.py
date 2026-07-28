from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_pipeline import (
    ADJUSTED_MARKET_PATH,
    COMBINED_FEATURES_PATH,
    MEMBERSHIP_HISTORY_DIR,
    RAW_MARKET_PATH,
    load_verified_membership_history,
)
from src.foreign_flow_modeling import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_MINIMUM_TRAINING_DATES,
    DEFAULT_RESEARCH_END,
    DEFAULT_SIGNAL_STEP_DAYS,
    DEFAULT_TOP_N,
    FEATURE_SET_MARKET_BASELINE,
    FOREIGN_FLOW_FEATURE_SET_ORDER,
    FROZEN_MARKET_BASELINE_FEATURES,
    FROZEN_MODEL_NAME,
    FUTURE_HOLDOUT_START,
    assert_market_baseline_parity,
    build_canonical_modeling_dataset,
    build_daily_foreign_flow_metrics,
    build_foreign_flow_feature_sets,
    build_modeling_sample,
    build_paper_horizon_signal_schedule,
    build_production_market_baseline_reference,
    run_foreign_flow_paper_horizon,
    summarize_foreign_feature_importance,
    summarize_foreign_flow_metrics,
    summarize_paired_foreign_flow_differences,
    target_column_for_horizon,
)
from src.market_data_schema import (
    load_canonical_market_history,
)


OUTPUT_STEM = "foreign_flow_rebuild_development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the historical FB/FS comparison on the "
            "production paper-horizon frame."
        )
    )
    parser.add_argument(
        "--features-path",
        default=COMBINED_FEATURES_PATH,
    )
    parser.add_argument(
        "--raw-market-path",
        default=RAW_MARKET_PATH,
    )
    parser.add_argument(
        "--adjusted-market-path",
        default=ADJUSTED_MARKET_PATH,
    )
    parser.add_argument(
        "--membership-dir",
        default=MEMBERSHIP_HISTORY_DIR,
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=DEFAULT_HORIZON_DAYS,
    )
    parser.add_argument(
        "--minimum-training-dates",
        type=int,
        default=DEFAULT_MINIMUM_TRAINING_DATES,
    )
    parser.add_argument(
        "--signal-step-days",
        type=int,
        default=DEFAULT_SIGNAL_STEP_DAYS,
    )
    parser.add_argument("--evaluation-start")
    parser.add_argument(
        "--research-end",
        default=DEFAULT_RESEARCH_END.date().isoformat(),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tables/foreign_flow",
    )
    parser.add_argument(
        "--predictions-dir",
        default="data/processed",
    )
    return parser.parse_args()


def _atomic_csv(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        data.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_parquet(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        data.to_parquet(
            temporary_path,
            index=False,
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(
    payload: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        json.dump(
            payload,
            temporary,
            indent=2,
            sort_keys=True,
        )
        temporary.write("\n")
        temporary_path = Path(temporary.name)

    try:
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _feature_contract_hash() -> str:
    payload = "\n".join(
        FROZEN_MARKET_BASELINE_FEATURES
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_fixed_protocol(args: argparse.Namespace) -> pd.Timestamp:
    if args.horizon_days != DEFAULT_HORIZON_DAYS:
        raise ValueError(
            "The rebuild is frozen to a 10-trading-day target"
        )

    if args.signal_step_days != args.horizon_days:
        raise ValueError(
            "Signal spacing must equal the forecast horizon"
        )

    if args.minimum_training_dates <= 0:
        raise ValueError(
            "Minimum training dates must be positive"
        )

    if args.top_n <= 0:
        raise ValueError("top-n must be positive")

    if args.bootstrap_samples <= 0:
        raise ValueError(
            "Bootstrap sample count must be positive"
        )

    research_end = pd.Timestamp(
        args.research_end
    ).normalize()

    if research_end >= FUTURE_HOLDOUT_START:
        raise ValueError(
            "Historical development cannot enter the future "
            f"holdout beginning {FUTURE_HOLDOUT_START.date()}"
        )

    return research_end


def _restrict_to_research_end(
    data: pd.DataFrame,
    research_end: pd.Timestamp,
) -> pd.DataFrame:
    restricted = data.copy()
    restricted["date"] = pd.to_datetime(
        restricted["date"],
        errors="raise",
    ).dt.normalize()
    return restricted.loc[
        restricted["date"].le(research_end)
    ].copy()


def _validate_final_predictions(
    predictions: pd.DataFrame,
    feature_sets: dict[str, tuple[str, ...]],
    research_end: pd.Timestamp,
) -> None:
    if set(predictions["feature_set"]) != set(
        FOREIGN_FLOW_FEATURE_SET_ORDER
    ):
        raise ValueError(
            "Final predictions lack a frozen feature set"
        )

    if predictions.duplicated(
        ["date", "ticker", "feature_set"]
    ).any():
        raise ValueError(
            "Final predictions contain duplicate keys"
        )

    if predictions[
        ["predicted_return", "actual_return"]
    ].isna().any().any():
        raise ValueError(
            "Final predictions contain missing values"
        )

    if pd.to_datetime(predictions["date"]).max() > research_end:
        raise ValueError(
            "Final predictions crossed the research cutoff"
        )

    coverage = predictions.groupby(
        ["date", "ticker"]
    )["feature_set"].nunique()

    if not coverage.eq(len(feature_sets)).all():
        raise ValueError(
            "Feature sets have unmatched prediction coverage"
        )

    observed_counts = {
        name: [
            int(value)
            for value in values
        ]
        for name, values in (
            predictions.groupby("feature_set")[
                "feature_count"
            ]
            .unique()
            .items()
        )
    }
    expected_counts = {
        name: [len(columns)]
        for name, columns in feature_sets.items()
    }

    if observed_counts != expected_counts:
        raise ValueError(
            "Prediction feature counts differ from the "
            f"frozen contract: {observed_counts}"
        )


def main() -> None:
    args = parse_args()
    research_end = _validate_fixed_protocol(args)
    target_column = target_column_for_horizon(
        args.horizon_days
    )

    print("FOREIGN-FLOW MODEL REBUILD")
    print("=" * 80)
    print("Analysis class: historical development")
    print("Research cutoff:", research_end.date())
    print(
        "Future holdout begins:",
        FUTURE_HOLDOUT_START.date(),
    )
    print("Forecast horizon:", args.horizon_days)
    print("Signal spacing:", args.signal_step_days)
    print("Target:", target_column)
    print("Model:", FROZEN_MODEL_NAME)
    print(
        "Market baseline features:",
        len(FROZEN_MARKET_BASELINE_FEATURES),
    )
    print(
        "Ticker rule: all verified historical members; "
        "point-in-time membership only"
    )
    print(
        "Missing-data rule: preserve NaN values and use "
        "the production tree-model imputer"
    )

    features = pd.read_parquet(args.features_path)
    market_data = load_canonical_market_history(
        args.raw_market_path,
        args.adjusted_market_path,
    )
    membership = load_verified_membership_history(
        args.membership_dir
    )
    features = _restrict_to_research_end(
        features,
        research_end,
    )
    market_data = _restrict_to_research_end(
        market_data,
        research_end,
    )
    modeling = build_canonical_modeling_dataset(
        features=features,
        market_data=market_data,
        membership=membership,
        horizon_days=args.horizon_days,
    )
    feature_sets = build_foreign_flow_feature_sets(
        modeling
    )
    modeling_sample, eligible_dates, sample_audit = (
        build_modeling_sample(
            modeling,
            feature_sets,
            horizon_days=args.horizon_days,
        )
    )
    all_dates = pd.DatetimeIndex(
        modeling["date"].drop_duplicates()
    ).sort_values()
    signals = build_paper_horizon_signal_schedule(
        all_dates,
        horizon_days=args.horizon_days,
        minimum_training_dates=(
            args.minimum_training_dates
        ),
        signal_step_days=args.signal_step_days,
        evaluation_start=args.evaluation_start,
        evaluation_end=research_end,
    )

    print("\nBASELINE PARITY GATE")
    print("-" * 80)
    eligible_signal_set = set(eligible_dates)
    parity_signal = next(
        (
            signal
            for signal in signals
            if signal.signal_date in eligible_signal_set
        ),
        None,
    )

    if parity_signal is None:
        raise ValueError(
            "No eligible signal is available for baseline parity"
        )

    print(
        "Bounded parity signal:",
        parity_signal.signal_date.date(),
    )
    baseline_feature_sets = {
        FEATURE_SET_MARKET_BASELINE: feature_sets[
            FEATURE_SET_MARKET_BASELINE
        ]
    }
    baseline_predictions, _ = (
        run_foreign_flow_paper_horizon(
            modeling_sample=modeling_sample,
            eligible_signal_dates=eligible_dates,
            feature_sets=baseline_feature_sets,
            signals=[parity_signal],
            horizon_days=args.horizon_days,
        )
    )
    production_reference = (
        build_production_market_baseline_reference(
            modeling,
            horizon_days=args.horizon_days,
            minimum_training_dates=(
                args.minimum_training_dates
            ),
            signal_step_days=args.signal_step_days,
            evaluation_start=parity_signal.signal_date,
            evaluation_end=parity_signal.signal_date,
        )
    )
    parity_audit = assert_market_baseline_parity(
        baseline_predictions,
        production_reference,
    )
    print("Baseline parity:", parity_audit["status"])
    print("Parity rows:", parity_audit["rows"])
    print(
        "Maximum prediction difference:",
        parity_audit["maximum_prediction_difference"],
    )

    print("\nFULL MATCHED FEATURE-SET COMPARISON")
    print("-" * 80)
    comparison_workers = min(4, len(feature_sets))
    print("Parallel feature-set workers:", comparison_workers)

    def report_progress(
        completed: int,
        total: int,
        signal,
    ) -> None:
        if (
            completed == 1
            or completed % 10 == 0
            or completed == total
        ):
            print(
                f"Completed signals: {completed}/{total} "
                f"through {signal.signal_date.date()}",
                flush=True,
            )

    predictions, importance = (
        run_foreign_flow_paper_horizon(
            modeling_sample=modeling_sample,
            eligible_signal_dates=eligible_dates,
            feature_sets=feature_sets,
            signals=signals,
            horizon_days=args.horizon_days,
            max_workers=comparison_workers,
            progress_callback=report_progress,
        )
    )
    _validate_final_predictions(
        predictions,
        feature_sets,
        research_end,
    )
    daily_metrics = build_daily_foreign_flow_metrics(
        predictions,
        top_n=args.top_n,
    )
    summary = summarize_foreign_flow_metrics(
        daily_metrics
    )
    paired = summarize_paired_foreign_flow_differences(
        daily_metrics,
        bootstrap_samples=args.bootstrap_samples,
    )
    foreign_importance = (
        summarize_foreign_feature_importance(
            importance
        )
    )

    output_dir = Path(args.output_dir)
    predictions_dir = Path(args.predictions_dir)
    output_paths = {
        "summary": output_dir / f"{OUTPUT_STEM}_summary.csv",
        "paired": output_dir / f"{OUTPUT_STEM}_paired.csv",
        "daily": output_dir / f"{OUTPUT_STEM}_daily.csv",
        "foreign_importance": (
            output_dir
            / f"{OUTPUT_STEM}_foreign_importance.csv"
        ),
        "manifest": output_dir / f"{OUTPUT_STEM}_manifest.json",
        "predictions": (
            predictions_dir
            / f"{OUTPUT_STEM}_predictions.parquet"
        ),
    }
    evaluation_dates = pd.DatetimeIndex(
        predictions["date"].drop_duplicates()
    ).sort_values()
    manifest: dict[str, object] = {
        "protocol_version": 2,
        "analysis_class": "historical_development_only",
        "model": FROZEN_MODEL_NAME,
        "horizon_days": args.horizon_days,
        "target_column": target_column,
        "minimum_training_dates": (
            args.minimum_training_dates
        ),
        "signal_step_days": args.signal_step_days,
        "top_n": args.top_n,
        "research_end": research_end.date().isoformat(),
        "future_holdout_start": (
            FUTURE_HOLDOUT_START.date().isoformat()
        ),
        "evaluation_first_date": (
            evaluation_dates.min().date().isoformat()
        ),
        "evaluation_last_date": (
            evaluation_dates.max().date().isoformat()
        ),
        "evaluated_signal_dates": int(
            len(evaluation_dates)
        ),
        "historical_ticker_count": int(
            modeling["ticker"].nunique()
        ),
        "feature_sets": {
            name: list(columns)
            for name, columns in feature_sets.items()
        },
        "market_baseline_feature_count": len(
            FROZEN_MARKET_BASELINE_FEATURES
        ),
        "market_baseline_contract_sha256": (
            _feature_contract_hash()
        ),
        "sample_audit": sample_audit,
        "baseline_parity": parity_audit,
        "missing_feature_values_preserved": True,
        "complete_case_filtering_used": False,
        "common_exchange_calendar_labels": True,
        "point_in_time_membership": True,
        "production_tree_imputer_used": True,
        "production_configuration_changed": False,
        "paper_orders_changed": False,
        "real_orders_changed": False,
        "promotion_decision": "not_evaluated",
    }

    _atomic_csv(summary, output_paths["summary"])
    _atomic_csv(paired, output_paths["paired"])
    _atomic_csv(daily_metrics, output_paths["daily"])
    _atomic_csv(
        foreign_importance,
        output_paths["foreign_importance"],
    )
    _atomic_parquet(
        predictions,
        output_paths["predictions"],
    )
    _atomic_json(
        manifest,
        output_paths["manifest"],
    )

    print("\nREBUILD RESULTS")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("\nPAIRED RANK-IC DIFFERENCES")
    print(paired.to_string(index=False))
    print("\nSample audit:")
    print(json.dumps(sample_audit, indent=2))
    print("\nOutputs:")

    for output_path in output_paths.values():
        print(output_path)

    print(
        "\nNo production configuration, paper orders, "
        "or real orders were changed."
    )
    print(
        "Do not promote FB/FS until the rebuilt results "
        "and confidence intervals are reviewed."
    )


if __name__ == "__main__":
    main()
