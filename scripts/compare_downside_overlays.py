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
    summarize_non_overlapping_policy_returns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare observable drawdown overlays without changing the model."
    )
    parser.add_argument("--predictions-path", default="data/processed/tree_model_predictions.parquet")
    parser.add_argument("--market-data-path", default="data/raw/vnstock/vn30_ohlcv.csv")
    parser.add_argument("--model", default="gradient_boosting")
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def fixed_model_policy(model_name: str) -> dict[str, str]:
    return {regime: model_name for regime in ("trend_up", "trend_down", "high_volatility")}


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = pd.read_csv(args.market_data_path)
    policy = fixed_model_policy(args.model)
    overlay_specs = {
        "baseline": None,
        "drawdown_10pct_to_50pct": build_market_drawdown_overlay(
            market_data, trigger_drawdown=-0.10, reduced_exposure=0.50
        ),
        "drawdown_15pct_to_50pct": build_market_drawdown_overlay(
            market_data, trigger_drawdown=-0.15, reduced_exposure=0.50
        ),
    }
    summaries = []
    for overlay_name, overlay in overlay_specs.items():
        history = build_non_overlapping_policy_returns(
            predictions,
            market_data,
            policy=policy,
            top_n=args.top_n,
            target_exposure_by_date=(
                None if overlay is None else overlay.set_index("date")["target_exposure"]
            ),
        )
        summary = summarize_non_overlapping_policy_returns(history)
        summary.insert(0, "overlay_name", overlay_name)
        summaries.append(summary)

    print("\nDOWNSIDE-OVERLAY COST-AWARE COMPARISON")
    print("=" * 80)
    print(f"Base model: {args.model}")
    print(
        "Historical prediction horizon: "
        f"{HISTORICAL_TREE_PREDICTION_HORIZON_DAYS} trading days"
    )
    print(pd.concat(summaries, ignore_index=True).round(6).to_string(index=False))
    print("\nOverlays use only the equal-weight market drawdown available on each date.")
    print("This does not validate the separate 10-day daily paper-scoring horizon.")
    print("Diagnostic only. No model setting, target, paper order, or real order changed.")


if __name__ == "__main__":
    main()
