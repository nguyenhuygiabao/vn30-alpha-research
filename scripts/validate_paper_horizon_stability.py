from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_regime_policy import POLICIES
from src.paper_horizon_evaluation import (
    build_paper_policy_histories,
    prepare_paper_horizon_inputs,
    summarize_holdout_active_stability,
    summarize_paper_policy_periods,
    summarize_paper_policy_years,
)
from src.paper_horizon_history import PAPER_HORIZON_PREDICTIONS_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate aligned paper-horizon policies across time."
    )
    parser.add_argument("--predictions-path", default=PAPER_HORIZON_PREDICTIONS_PATH)
    parser.add_argument("--market-data-path", default="data/raw/vnstock/vn30_ohlcv.csv")
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = pd.read_csv(args.market_data_path)
    predictions, market_data, benchmark = prepare_paper_horizon_inputs(
        predictions, market_data
    )
    histories = build_paper_policy_histories(
        predictions, market_data, benchmark, POLICIES, top_n=args.top_n
    )
    periods = summarize_paper_policy_periods(histories, args.holdout_start)
    stability = summarize_holdout_active_stability(histories, args.holdout_start)
    years = summarize_paper_policy_years(histories)

    print("\nALIGNED PAPER-HORIZON STABILITY VALIDATION")
    print("=" * 80)
    print(f"Raw market history starts: {market_data['date'].min().date()}")
    print(f"Out-of-sample predictions start: {predictions['date'].min().date()}")
    print(f"Holdout starts: {args.holdout_start}")
    pre_covid_available = market_data["date"].min() < pd.Timestamp("2020-01-01")
    print(f"Pre-COVID history available: {'yes' if pre_covid_available else 'no'}")
    print("\nTraining and holdout summaries:")
    print(periods.round(6).to_string(index=False))
    print("\nHoldout active-return block-bootstrap checks:")
    print(stability.round(6).to_string(index=False))
    print("\nCalendar-year results:")
    print(years.round(6).to_string(index=False))
    print("\nCurrent-constituent historical testing may contain survivorship bias.")
    print("Diagnostic only. No model, target, paper order, or real order changed.")


if __name__ == "__main__":
    main()
