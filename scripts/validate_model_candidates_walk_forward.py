from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_validation import build_fixed_model_policies
from src.regime_policy_backtest import (
    HISTORICAL_TREE_PREDICTION_HORIZON_DAYS,
    build_non_overlapping_policy_returns,
    build_paired_overlay_returns,
    summarize_paired_overlay_stability,
)
from src.walk_forward import (
    select_training_candidate,
    summarize_walk_forward_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a fixed model on training history and validate it on holdout dates."
    )
    parser.add_argument("--predictions-path", default="data/processed/tree_model_predictions.parquet")
    parser.add_argument("--market-data-path", default="data/raw/vnstock/vn30_ohlcv.csv")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--holdout-start", default="2024-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = pd.read_csv(args.market_data_path)
    histories = {
        name: build_non_overlapping_policy_returns(
            predictions,
            market_data,
            policy=policy,
            top_n=args.top_n,
        )
        for name, policy in build_fixed_model_policies().items()
    }
    summary = summarize_walk_forward_candidates(histories, args.holdout_start)
    selected = select_training_candidate(summary)
    holdout_start = pd.Timestamp(args.holdout_start)
    paired = build_paired_overlay_returns(
        histories["gradient_boosting"].loc[
            histories["gradient_boosting"]["date"] >= holdout_start
        ],
        histories[selected].loc[histories[selected]["date"] >= holdout_start],
    )
    paired_summary = summarize_paired_overlay_stability(paired)

    print("\nWALK-FORWARD FIXED-MODEL VALIDATION")
    print("=" * 80)
    print(f"Holdout starts: {args.holdout_start}")
    print(
        "Historical prediction horizon: "
        f"{HISTORICAL_TREE_PREDICTION_HORIZON_DAYS} trading days"
    )
    print(f"Candidate selected from training only: {selected}")
    print("\nCandidate summaries:")
    print(summary.round(6).to_string(index=False))
    print("\nSelected-vs-gradient-boosting holdout paired check:")
    print(paired_summary.round(6).to_string(index=False))
    print("\nDiagnostic only. No configuration, targets, paper orders, or real orders changed.")
    print("This does not validate the separate 10-day daily paper-scoring horizon.")


if __name__ == "__main__":
    main()
