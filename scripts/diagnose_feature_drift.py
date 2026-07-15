from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_drift import build_feature_drift_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recent model-feature distributions with earlier training history."
    )
    parser.add_argument("--features-path", default="data/processed/features_combined.parquet")
    parser.add_argument("--reference-window", type=int, default=756)
    parser.add_argument("--recent-window", type=int, default=126)
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features_path)
    report = build_feature_drift_report(
        features,
        reference_window_dates=args.reference_window,
        recent_window_dates=args.recent_window,
    )
    print("\nFEATURE-DISTRIBUTION DRIFT DIAGNOSTIC")
    print("=" * 80)
    print(
        f"Reference window: {args.reference_window} dates; "
        f"recent window: {args.recent_window} dates"
    )
    print(report.head(args.top).round(6).to_string(index=False))
    print("\nDiagnostic only. It does not retrain or change the daily model.")


if __name__ == "__main__":
    main()
