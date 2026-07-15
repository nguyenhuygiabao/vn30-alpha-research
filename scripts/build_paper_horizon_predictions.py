from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_ohlcv_csv
from src.paper_horizon_history import (
    build_historical_paper_predictions,
    save_historical_paper_predictions,
)
from src.paper_trading.scoring import build_daily_modeling_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe historical predictions aligned to paper scoring."
    )
    parser.add_argument("--raw-data-path", default="data/raw/vnstock/vn30_ohlcv.csv")
    parser.add_argument("--output-path", default="data/processed/paper_horizon_predictions.parquet")
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--minimum-training-dates", type=int, default=252)
    parser.add_argument("--signal-step-days", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = str(Path(args.output_path).with_suffix(".partial.csv"))
    print("BUILDING PAPER-HORIZON HISTORICAL PREDICTIONS", flush=True)
    print("=" * 80, flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print("An interrupted run can be resumed with the same command.", flush=True)
    market_data = load_ohlcv_csv(args.raw_data_path)
    modeling_dataset = build_daily_modeling_dataset(
        market_data,
        horizon_days=args.horizon_days,
    )
    predictions = build_historical_paper_predictions(
        modeling_dataset,
        horizon_days=args.horizon_days,
        minimum_training_dates=args.minimum_training_dates,
        signal_step_days=args.signal_step_days,
        checkpoint_path=checkpoint_path,
        progress_callback=lambda completed, total, date: print(
            f"Progress: {completed}/{total} signal dates through {date.date()}",
            flush=True,
        ),
    )
    output_path = save_historical_paper_predictions(predictions, args.output_path)
    Path(checkpoint_path).unlink(missing_ok=True)

    print("\nPAPER-HORIZON HISTORICAL PREDICTIONS COMPLETED")
    print("=" * 80)
    print(f"Forecast horizon: {args.horizon_days} trading days")
    print(f"Signal spacing: {args.signal_step_days} trading days")
    print(f"Minimum training history: {args.minimum_training_dates} dates")
    print(f"Prediction dates: {predictions['date'].nunique()}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Models: {sorted(predictions['model_name'].unique())}")
    print(f"Output path: {output_path}")
    print("No configuration, targets, paper orders, or real orders changed.")


if __name__ == "__main__":
    main()
