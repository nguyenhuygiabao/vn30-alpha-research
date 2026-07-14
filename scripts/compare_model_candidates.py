from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_candidates import (
    summarize_model_candidates,
    summarize_paired_candidate_stability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare out-of-sample tree-model candidates and rank ensemble."
    )
    parser.add_argument(
        "--predictions-path",
        default="data/processed/tree_model_predictions.parquet",
    )
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--rolling-window", type=int, default=126)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    summary = summarize_model_candidates(predictions, top_n=args.top_n)
    stability = summarize_paired_candidate_stability(
        predictions,
        top_n=args.top_n,
        rolling_window=args.rolling_window,
    )
    print("\nMODEL CANDIDATE OUT-OF-SAMPLE COMPARISON")
    print("=" * 80)
    print(summary.round(6).to_string(index=False))
    print("\nRANK-ENSEMBLE PAIRED STABILITY CHECK")
    print("-" * 80)
    print(stability.round(6).to_string(index=False))
    print("\nComparison only. No model configuration or orders were changed.")


if __name__ == "__main__":
    main()
