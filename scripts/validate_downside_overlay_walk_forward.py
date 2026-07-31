from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regime_policy_backtest import (
    HISTORICAL_TREE_PREDICTION_HORIZON_DAYS,
    build_market_drawdown_overlay,
    build_non_overlapping_policy_returns,
    build_paired_overlay_returns,
    summarize_paired_overlay_stability,
)
from src.walk_forward import (
    select_training_candidate,
    summarize_walk_forward_candidates,
)
from src.research_market_data import (
    add_market_data_arguments,
    load_research_market_history_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a fixed downside overlay on training data and test later dates."
    )
    parser.add_argument("--predictions-path", default="data/processed/tree_model_predictions.parquet")
    add_market_data_arguments(parser)
    parser.add_argument("--model", default="rank_ensemble")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--holdout-start", default="2024-01-01")
    return parser.parse_args()


def fixed_model_policy(model_name: str) -> dict[str, str]:
    return {regime: model_name for regime in ("trend_up", "trend_down", "high_volatility")}


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = load_research_market_history_from_args(args)
    policy = fixed_model_policy(args.model)
    overlays = {
        "baseline": None,
        "drawdown_10pct_to_50pct": build_market_drawdown_overlay(
            market_data, trigger_drawdown=-0.10, reduced_exposure=0.50
        ),
        "drawdown_15pct_to_50pct": build_market_drawdown_overlay(
            market_data, trigger_drawdown=-0.15, reduced_exposure=0.50
        ),
    }
    histories = {
        name: build_non_overlapping_policy_returns(
            predictions,
            market_data,
            policy=policy,
            top_n=args.top_n,
            target_exposure_by_date=(
                None if overlay is None else overlay.set_index("date")["target_exposure"]
            ),
        )
        for name, overlay in overlays.items()
    }
    summary = summarize_walk_forward_candidates(histories, args.holdout_start)
    selected = select_training_candidate(summary)
    holdout_start = pd.Timestamp(args.holdout_start)
    paired = build_paired_overlay_returns(
        histories["baseline"].loc[histories["baseline"]["date"] >= holdout_start],
        histories[selected].loc[histories[selected]["date"] >= holdout_start],
    )
    paired_summary = summarize_paired_overlay_stability(paired)

    print("\nWALK-FORWARD DOWNSIDE-OVERLAY VALIDATION")
    print("=" * 80)
    print(f"Base model: {args.model}")
    print(
        "Historical prediction horizon: "
        f"{HISTORICAL_TREE_PREDICTION_HORIZON_DAYS} trading days"
    )
    print(f"Holdout starts: {args.holdout_start}")
    print(f"Candidate selected from training only: {selected}")
    print("\nCandidate summaries:")
    print(summary.round(6).to_string(index=False))
    print("\nSelected-vs-baseline holdout paired check:")
    print(paired_summary.round(6).to_string(index=False))
    print("\nDiagnostic only. No configuration, targets, paper orders, or real orders changed.")
    print("This does not validate the separate 10-day daily paper-scoring horizon.")


if __name__ == "__main__":
    main()
