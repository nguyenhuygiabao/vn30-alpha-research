from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_regime_policy import POLICIES
from src.paper_horizon_history import PAPER_HORIZON_PREDICTIONS_PATH
from src.paper_horizon_evaluation import (
    build_paper_policy_histories,
    prepare_paper_horizon_inputs,
)
from src.regime_policy_backtest import summarize_non_overlapping_policy_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the aligned ten-day paper-horizon prediction history."
    )
    parser.add_argument("--predictions-path", default=PAPER_HORIZON_PREDICTIONS_PATH)
    parser.add_argument("--market-data-path", default="data/raw/vnstock/vn30_ohlcv.csv")
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = pd.read_csv(args.market_data_path)
    predictions, market_data, benchmark_returns = prepare_paper_horizon_inputs(
        predictions, market_data
    )
    histories = build_paper_policy_histories(
        predictions, market_data, benchmark_returns, POLICIES, top_n=args.top_n
    )

    summaries = []
    for policy_name, history in histories.items():
        summary = summarize_non_overlapping_policy_returns(history)
        summary.insert(0, "policy_name", policy_name)
        summaries.append(summary)

    print("\nALIGNED TEN-DAY PAPER-HORIZON COST-AWARE COMPARISON")
    print("=" * 80)
    print(pd.concat(summaries, ignore_index=True).round(6).to_string(index=False))
    print("\nPrediction dates were already spaced by 10 trading days.")
    print("No second schedule subsampling was applied.")
    print("The schedule is T+2-compatible; this is not an order-fill replay.")
    print("Diagnostic only. No configuration, targets, paper orders, or real orders changed.")


if __name__ == "__main__":
    main()
