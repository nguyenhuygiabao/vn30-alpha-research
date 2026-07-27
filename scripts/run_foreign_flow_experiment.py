from __future__ import annotations

import argparse
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
    DEFAULT_DEVELOPMENT_END,
    DEFAULT_HOLDOUT_START,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_INITIAL_TRAINING_DATES,
    DEFAULT_PURGE_DATES,
    DEFAULT_TEST_BLOCK_DATES,
    DEFAULT_TOP_N,
    FEATURE_SET_ALL_FOREIGN,
    FEATURE_SET_FROZEN_FOREIGN,
    FEATURE_SET_MARKET_BASELINE,
    FOREIGN_FLOW_FEATURE_SET_ORDER,
    build_canonical_modeling_dataset,
    build_common_complete_case_sample,
    build_daily_foreign_flow_metrics,
    build_expanding_walk_forward_windows,
    build_foreign_flow_feature_sets,
    run_foreign_flow_walk_forward,
    summarize_foreign_feature_importance,
    summarize_foreign_flow_metrics,
    summarize_paired_foreign_flow_differences,
    target_column_for_horizon,
    validate_foreign_flow_feature_sets,
)
from src.market_data_schema import (
    load_canonical_market_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-safe, fixed-parameter comparison of "
            "market, frozen foreign-flow, and fb/fs feature sets."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("development", "holdout"),
        default="development",
    )
    parser.add_argument(
        "--candidate",
        choices=FOREIGN_FLOW_FEATURE_SET_ORDER,
        help=(
            "Frozen development candidate. Required only for "
            "the holdout phase."
        ),
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
        "--initial-training-dates",
        type=int,
        default=DEFAULT_INITIAL_TRAINING_DATES,
    )
    parser.add_argument(
        "--purge-dates",
        type=int,
        default=DEFAULT_PURGE_DATES,
    )
    parser.add_argument(
        "--test-block-dates",
        type=int,
        default=DEFAULT_TEST_BLOCK_DATES,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
    )
    parser.add_argument(
        "--development-end",
        default=DEFAULT_DEVELOPMENT_END.date().isoformat(),
    )
    parser.add_argument(
        "--holdout-start",
        default=DEFAULT_HOLDOUT_START.date().isoformat(),
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tables",
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
        data.to_csv(
            temporary_path,
            index=False,
        )
        os.replace(
            temporary_path,
            output_path,
        )
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
        os.replace(
            temporary_path,
            output_path,
        )
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
        os.replace(
            temporary_path,
            output_path,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _phase_feature_sets(
    all_feature_sets: dict[str, tuple[str, ...]],
    *,
    phase: str,
    candidate: str | None,
) -> dict[str, tuple[str, ...]]:
    if phase == "development":
        if candidate is not None:
            raise ValueError(
                "--candidate is not used during development"
            )

        return all_feature_sets

    if candidate is None:
        raise ValueError(
            "--candidate is required for the holdout phase"
        )

    if candidate == FEATURE_SET_MARKET_BASELINE:
        raise ValueError(
            "The holdout candidate must add foreign-flow features"
        )

    selected_names = list(
        dict.fromkeys(
            [
                FEATURE_SET_MARKET_BASELINE,
                FEATURE_SET_FROZEN_FOREIGN,
                candidate,
            ]
        )
    )

    return {
        name: all_feature_sets[name]
        for name in selected_names
    }


def _load_development_feature_sets(
    manifest_path: Path,
    modeling: pd.DataFrame,
    *,
    horizon_days: int,
    initial_training_dates: int,
    purge_dates: int,
    test_block_dates: int,
) -> dict[str, tuple[str, ...]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Development manifest not found. Run and review the "
            f"development phase first: {manifest_path}"
        )

    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    expected_settings = {
        "phase": "development",
        "horizon_days": horizon_days,
        "initial_training_dates": initial_training_dates,
        "purge_dates": purge_dates,
        "test_block_dates": test_block_dates,
    }
    mismatches = {
        name: {
            "expected": expected,
            "received": payload.get(name),
        }
        for name, expected in expected_settings.items()
        if payload.get(name) != expected
    }

    if mismatches:
        raise ValueError(
            "Development manifest settings do not match the "
            f"holdout request: {mismatches}"
        )

    frozen_feature_sets = payload.get("feature_sets")

    if not isinstance(frozen_feature_sets, dict):
        raise ValueError(
            "Development manifest lacks frozen feature sets"
        )

    return validate_foreign_flow_feature_sets(
        modeling,
        frozen_feature_sets,
    )


def main() -> None:
    args = parse_args()
    target_column = target_column_for_horizon(
        args.horizon_days
    )

    if args.purge_dates < args.horizon_days:
        raise ValueError(
            "Purge dates cannot be shorter than the forecast horizon"
        )

    features_path = Path(args.features_path)
    raw_market_path = Path(args.raw_market_path)
    adjusted_market_path = Path(
        args.adjusted_market_path
    )
    membership_dir = Path(args.membership_dir)

    print("FOREIGN-FLOW MODEL EXPERIMENT")
    print("=" * 80)
    print("Phase:", args.phase)
    print("Forecast horizon:", args.horizon_days)
    print("Target:", target_column)
    print("Model: frozen gradient_boosting")
    print(
        "Missing-data rule: complete cases only; "
        "foreign flow is never filled with zero"
    )

    features = pd.read_parquet(features_path)
    market_data = load_canonical_market_history(
        raw_market_path,
        adjusted_market_path,
    )
    membership = load_verified_membership_history(
        membership_dir
    )
    modeling = build_canonical_modeling_dataset(
        features=features,
        market_data=market_data,
        membership=membership,
        horizon_days=args.horizon_days,
    )
    historical_dates = pd.DatetimeIndex(
        modeling.loc[
            modeling[target_column].notna(),
            "date",
        ].drop_duplicates()
    ).sort_values()

    if (
        len(historical_dates)
        < args.initial_training_dates
    ):
        raise ValueError(
            "Historical data do not cover the initial "
            "training period"
        )

    output_dir = Path(args.output_dir)
    development_manifest_path = (
        output_dir
        / "foreign_flow_development_manifest.json"
    )

    if args.phase == "development":
        initial_feature_cutoff = historical_dates[
            args.initial_training_dates - 1
        ]
        feature_definition_rows = modeling.loc[
            modeling["date"].le(initial_feature_cutoff)
            & modeling[target_column].notna()
        ]
        all_feature_sets = (
            build_foreign_flow_feature_sets(
                feature_definition_rows
            )
        )
    else:
        all_feature_sets = (
            _load_development_feature_sets(
                development_manifest_path,
                modeling,
                horizon_days=args.horizon_days,
                initial_training_dates=(
                    args.initial_training_dates
                ),
                purge_dates=args.purge_dates,
                test_block_dates=args.test_block_dates,
            )
        )

    common_sample, complete_signal_dates, sample_audit = (
        build_common_complete_case_sample(
            modeling,
            all_feature_sets,
            horizon_days=args.horizon_days,
        )
    )
    active_feature_sets = _phase_feature_sets(
        all_feature_sets,
        phase=args.phase,
        candidate=args.candidate,
    )
    if args.phase == "development":
        evaluation_start = None
        evaluation_end = args.development_end
    else:
        evaluation_start = args.holdout_start
        evaluation_end = None

    windows = build_expanding_walk_forward_windows(
        historical_dates,
        initial_training_dates=args.initial_training_dates,
        purge_dates=args.purge_dates,
        test_block_dates=args.test_block_dates,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )
    predictions, importance = run_foreign_flow_walk_forward(
        common_sample=common_sample,
        complete_signal_dates=complete_signal_dates,
        feature_sets=active_feature_sets,
        windows=windows,
        horizon_days=args.horizon_days,
    )
    daily_metrics = build_daily_foreign_flow_metrics(
        predictions,
        top_n=args.top_n,
    )
    summary = summarize_foreign_flow_metrics(
        daily_metrics
    )

    if args.phase == "development":
        paired = summarize_paired_foreign_flow_differences(
            daily_metrics
        )
    else:
        holdout_comparisons = [
            (
                feature_set_name,
                FEATURE_SET_MARKET_BASELINE,
            )
            for feature_set_name in active_feature_sets
            if feature_set_name != FEATURE_SET_MARKET_BASELINE
        ]
        paired = summarize_paired_foreign_flow_differences(
            daily_metrics,
            comparisons=holdout_comparisons,
        )

    foreign_importance = (
        summarize_foreign_feature_importance(
            importance
        )
    )
    predictions_dir = Path(args.predictions_dir)
    stem = f"foreign_flow_{args.phase}"
    output_paths = {
        "summary": output_dir / f"{stem}_summary.csv",
        "paired": output_dir / f"{stem}_paired.csv",
        "daily": output_dir / f"{stem}_daily.csv",
        "foreign_importance": (
            output_dir / f"{stem}_foreign_importance.csv"
        ),
        "manifest": output_dir / f"{stem}_manifest.json",
        "predictions": (
            predictions_dir / f"{stem}_predictions.parquet"
        ),
    }
    manifest = {
        "phase": args.phase,
        "candidate": args.candidate,
        "model": "gradient_boosting",
        "horizon_days": args.horizon_days,
        "target_column": target_column,
        "initial_training_dates": args.initial_training_dates,
        "purge_dates": args.purge_dates,
        "test_block_dates": args.test_block_dates,
        "top_n": args.top_n,
        "development_end": args.development_end,
        "holdout_start": args.holdout_start,
        "evaluation_first_date": (
            pd.Timestamp(predictions["date"].min())
            .date()
            .isoformat()
        ),
        "evaluation_last_date": (
            pd.Timestamp(predictions["date"].max())
            .date()
            .isoformat()
        ),
        "evaluated_dates": int(
            predictions["date"].nunique()
        ),
        "prediction_rows": int(len(predictions)),
        "walk_forward_windows": len(windows),
        "sample_audit": sample_audit,
        "feature_counts": {
            name: len(columns)
            for name, columns in active_feature_sets.items()
        },
        "feature_sets": {
            name: list(columns)
            for name, columns in all_feature_sets.items()
        },
        "source_paths": {
            "features": str(features_path),
            "raw_market": str(raw_market_path),
            "adjusted_market": str(adjusted_market_path),
            "membership": str(membership_dir),
        },
        "production_configuration_changed": False,
        "paper_orders_changed": False,
        "real_orders_changed": False,
    }

    _atomic_csv(summary, output_paths["summary"])
    _atomic_csv(paired, output_paths["paired"])
    _atomic_csv(daily_metrics, output_paths["daily"])
    _atomic_csv(
        foreign_importance,
        output_paths["foreign_importance"],
    )
    _atomic_json(
        manifest,
        output_paths["manifest"],
    )
    _atomic_parquet(
        predictions,
        output_paths["predictions"],
    )

    print("\nCOMMON-SAMPLE AUDIT")
    print("=" * 80)
    for name, value in sample_audit.items():
        print(f"{name}: {value}")
    print(
        "Feature counts:",
        {
            name: len(columns)
            for name, columns in active_feature_sets.items()
        },
    )
    print("Walk-forward windows:", len(windows))
    print(
        "Evaluated dates:",
        predictions["date"].nunique(),
    )
    print("Prediction rows:", len(predictions))

    print("\nFEATURE-SET SUMMARY")
    print("=" * 80)
    print(summary.round(6).to_string(index=False))
    print("\nPAIRED RANK-IC DIFFERENCES")
    print("=" * 80)
    print(paired.round(6).to_string(index=False))

    if not foreign_importance.empty:
        print("\nTOP FOREIGN-FLOW FEATURES")
        print("=" * 80)
        print(
            foreign_importance.groupby(
                "feature_set",
                sort=False,
            )
            .head(5)
            .round(6)
            .to_string(index=False)
        )

    print("\nOUTPUTS")
    print("=" * 80)
    for name, path in output_paths.items():
        print(f"{name}: {path}")

    if args.phase == "development":
        development_leader = summary.iloc[0][
            "feature_set"
        ]
        print(
            "\nDevelopment leader:",
            development_leader,
        )
        print(
            "Holdout data were not evaluated. "
            "Do not promote a feature set yet."
        )
    else:
        print(
            "\nHoldout evaluated for the frozen candidate. "
            "No production configuration was changed."
        )


if __name__ == "__main__":
    main()
