from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_candidates import (
    DEFAULT_REGIME_POLICY,
    evaluate_regime_policy,
    summarize_regime_policy,
)
from src.research_market_data import (
    add_market_data_arguments,
    load_research_market_history_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed regime-aware daily model-selection hypothesis."
    )
    parser.add_argument(
        "--predictions-path",
        default="data/processed/tree_model_predictions.parquet",
    )
    add_market_data_arguments(parser)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    market_data = load_research_market_history_from_args(args)
    history = evaluate_regime_policy(
        predictions,
        market_data,
        top_n=args.top_n,
    )
    summary = summarize_regime_policy(history)

    print("\nREGIME-AWARE MODEL POLICY DIAGNOSTIC")
    print("=" * 80)
    print("Fixed policy:")
    for regime, model_name in DEFAULT_REGIME_POLICY.items():
        print(f"- {regime}: {model_name}")
    print()
    print(summary.round(6).to_string(index=False))
    print()
    print("Selection counts:")
    print(history["selected_model"].value_counts().rename_axis("selected_model").to_string())
    print()
    print("Diagnostic only. It does not change daily scoring, targets, or orders.")


if __name__ == "__main__":
    main()
