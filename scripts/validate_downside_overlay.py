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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a fixed drawdown overlay is stable through time."
    )
    parser.add_argument("--predictions-path", default="data/processed/tree_model_predictions.parquet")
    parser.add_argument("--market-data-path", default="data/raw/vnstock/vn30_ohlcv.csv")
    parser.add_argument("--model", default="rank_ensemble")
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def fixed_model_policy(model_name: str) -> dict[str, str]:
    return {regime: model_name for regime in ("trend_up", "trend_down", "high_volatility")}


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = pd.read_csv(args.market_data_path)
    policy = fixed_model_policy(args.model)
    baseline = build_non_overlapping_policy_returns(
        predictions, market_data, policy=policy, top_n=args.top_n
    )
    overlay = build_market_drawdown_overlay(
        market_data, trigger_drawdown=-0.10, reduced_exposure=0.50
    )
    guarded = build_non_overlapping_policy_returns(
        predictions,
        market_data,
        policy=policy,
        top_n=args.top_n,
        target_exposure_by_date=overlay.set_index("date")["target_exposure"],
    )
    paired = build_paired_overlay_returns(baseline, guarded)
    stability = summarize_paired_overlay_stability(paired)
    yearly = paired.assign(year=pd.to_datetime(paired["date"]).dt.year).groupby("year").agg(
        rebalance_dates=("date", "count"),
        average_after_cost_difference=("after_cost_return_difference", "mean"),
    )

    print("\nDRAWDOWN-OVERLAY PAIRED STABILITY CHECK")
    print("=" * 80)
    print(f"Base model: {args.model}")
    print(
        "Historical prediction horizon: "
        f"{HISTORICAL_TREE_PREDICTION_HORIZON_DAYS} trading days"
    )
    print("Overlay: reduce exposure from 97% to 50% below a 10% market drawdown")
    print(stability.round(6).to_string(index=False))
    print("\nCalendar-year paired differences:")
    print(yearly.round(6).to_string())
    print("\nDiagnostic only. No configuration, targets, paper orders, or real orders changed.")
    print("This does not validate the separate 10-day daily paper-scoring horizon.")


if __name__ == "__main__":
    main()
