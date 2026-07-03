from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENERATED_OUTPUTS = [
    ROOT / "data" / "processed" / "ablation_tree_predictions.parquet",
    ROOT / "data" / "processed" / "backtest_returns.parquet",
    ROOT / "data" / "processed" / "baseline_returns.parquet",
    ROOT / "data" / "processed" / "classification_model_predictions.parquet",
    ROOT / "data" / "processed" / "features_combined.parquet",
    ROOT / "data" / "processed" / "features_momentum.parquet",
    ROOT / "data" / "processed" / "horizon_tree_predictions.parquet",
    ROOT / "data" / "processed" / "labels.parquet",
    ROOT / "data" / "processed" / "linear_model_predictions.parquet",
    ROOT / "data" / "processed" / "optimized_weights.parquet",
    ROOT / "data" / "processed" / "risk_model.parquet",
    ROOT / "data" / "processed" / "tree_feature_importance.parquet",
    ROOT / "data" / "processed" / "tree_model_predictions.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean generated local parquet outputs before rerunning the VN30 pipeline.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete files. Without this flag, the script only previews what would be removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    existing_files = [path for path in GENERATED_OUTPUTS if path.exists()]
    missing_files = [path for path in GENERATED_OUTPUTS if not path.exists()]

    print()
    print("VN30 GENERATED OUTPUT CLEANER")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print()
    print("Scope:")
    print("- Deletes generated data/processed parquet outputs only.")
    print("- Does not delete raw data.")
    print("- Does not delete reports, tables, or figures.")
    print()

    print(f"Generated output files listed: {len(GENERATED_OUTPUTS)}")
    print(f"Existing files found: {len(existing_files)}")
    print(f"Missing files: {len(missing_files)}")
    print()

    if existing_files:
        print("Files selected for deletion:")
        for path in existing_files:
            print(f"- {path.relative_to(ROOT)}")
    else:
        print("No generated output files found.")

    print()

    if not args.confirm:
        print("Dry run only. No files were deleted.")
        print("To delete these files, rerun with:")
        print("  py .\\scripts\\clean_generated_outputs.py --confirm")
        print()
        return

    for path in existing_files:
        path.unlink()

    print(f"Deleted files: {len(existing_files)}")
    print()


if __name__ == "__main__":
    main()