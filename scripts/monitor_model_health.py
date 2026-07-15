from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_health import build_model_health_history, summarize_latest_model_health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report rolling out-of-sample health for model candidates."
    )
    parser.add_argument("--predictions-path", default="data/processed/tree_model_predictions.parquet")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--rolling-window", type=int, default=126)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    history = build_model_health_history(
        predictions,
        top_n=args.top_n,
        rolling_window=args.rolling_window,
    )
    summary = summarize_latest_model_health(history)

    print("\nMODEL HEALTH DIAGNOSTIC")
    print("=" * 80)
    print(f"Rolling out-of-sample window: {args.rolling_window} signal dates")
    print(summary.round(6).to_string(index=False))
    print("\nDiagnostic only. It does not select or change the daily model.")


if __name__ == "__main__":
    main()
