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
    build_non_overlapping_policy_returns,
    summarize_non_overlapping_policy_returns,
)


POLICIES = {
    "gradient_boosting": {regime: "gradient_boosting" for regime in ("trend_up", "trend_down", "high_volatility")},
    "random_forest": {regime: "random_forest" for regime in ("trend_up", "trend_down", "high_volatility")},
    "rank_ensemble": {regime: "rank_ensemble" for regime in ("trend_up", "trend_down", "high_volatility")},
    "rolling_rank_ensemble": {
        regime: "rolling_rank_ensemble"
        for regime in ("trend_up", "trend_down", "high_volatility")
    },
    "regime_policy": {
        "trend_up": "random_forest",
        "high_volatility": "random_forest",
        "trend_down": "cash",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare non-overlapping, cost-aware regime-policy diagnostics."
    )
    parser.add_argument("--predictions-path", default="data/processed/tree_model_predictions.parquet")
    parser.add_argument("--market-data-path", default="data/raw/vnstock/vn30_ohlcv.csv")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument(
        "--holding-period-days",
        type=int,
        default=HISTORICAL_TREE_PREDICTION_HORIZON_DAYS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = pd.read_csv(args.market_data_path)
    summaries = []
    for policy_name, policy in POLICIES.items():
        history = build_non_overlapping_policy_returns(
            predictions,
            market_data,
            policy=policy,
            top_n=args.top_n,
            holding_period_days=args.holding_period_days,
        )
        summary = summarize_non_overlapping_policy_returns(history)
        summary.insert(0, "policy_name", policy_name)
        summaries.append(summary)

    print("\nNON-OVERLAPPING COST-AWARE POLICY COMPARISON")
    print("=" * 80)
    print(pd.concat(summaries, ignore_index=True).round(6).to_string(index=False))
    print(
        "\nEach rebalance is spaced by the historical tree prediction "
        f"horizon: {args.holding_period_days} trading days."
    )
    print("The schedule is T+2 settlement-compatible; this is not an order-fill replay.")
    print("This does not validate the separate 10-day daily paper-scoring horizon.")
    print("Diagnostic only. No configuration, targets, paper orders, or real orders changed.")


if __name__ == "__main__":
    main()
