"""Measure how the walk-forward training window size affects prediction quality.

`build_walk_forward_date_windows` counts `train_size` in trading dates, not in
blocks of dates. The production models in `src/tree_models.py` and
`src/linear_models.py` default to `train_size=3`, so every window fits on roughly
90 rows (3 dates x ~30 active constituents) against 99 features.

The foreign-flow experiment, which trains on an expanding window of at least 252
dates, reports Rank IC around 0.06 where the production pipeline reports about
0.02. This script quantifies how much of that gap is attributable to the training
window alone, holding the model, features, labels, and purge gap fixed.

It is a diagnostic. It changes no default, writes no production artefact, and
does not touch the published tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import load_baseline_dataset
from src.metrics import calculate_spearman_ic_by_date
from src.modeling_utils import get_model_feature_columns
from src.tree_models import (
    create_tree_models,
    predict_tree_models_for_windows,
)
from src.walk_forward_split import get_sorted_dates

MODEL_NAME: str = "gradient_boosting"
DEFAULT_TRAIN_SIZES: tuple[int, ...] = (3, 21, 63, 126, 252)
OUTPUT_PATH: str = "reports/tables/training_window_diagnostic.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare walk-forward training window sizes on prediction quality."
        ),
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_SIZES),
        help="Training window sizes in trading dates.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help=(
            "Stride between windows. The production pipeline uses 1; a larger "
            "stride subsamples windows so the sweep finishes in minutes."
        ),
    )
    parser.add_argument("--validation-size", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--purge-size", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output", default=OUTPUT_PATH)
    return parser.parse_args()


def build_windows_for_shared_test_positions(
    dates: pd.DatetimeIndex,
    test_positions: list[int],
    train_size: int,
    validation_size: int,
    test_size: int,
    purge_size: int,
) -> list[dict[str, pd.DatetimeIndex]]:
    """Build walk-forward windows that all end on the same test dates.

    `build_walk_forward_date_windows` strides forward from each configuration's
    own first valid position. Because that start position depends on
    `train_size`, two configurations sampled with a stride greater than one land
    on disjoint test dates, which makes their scores incomparable.

    Here the test positions are fixed up front and each window is constructed
    backwards from them, so every training-window size is scored on an identical
    set of dates. The layout -- train, purge, validation, purge, test -- matches
    `build_walk_forward_date_windows` exactly.
    """
    windows: list[dict[str, pd.DatetimeIndex]] = []

    for test_start in test_positions:
        validation_end = test_start - purge_size
        validation_start = validation_end - validation_size
        train_end = validation_start - purge_size
        train_start = train_end - train_size

        if train_start < 0:
            continue

        windows.append(
            {
                "train_dates": dates[train_start:train_end],
                "validation_dates": dates[validation_start:validation_end],
                "test_dates": dates[test_start : test_start + test_size],
            }
        )

    return windows


def summarize(
    predictions: pd.DataFrame,
    top_n: int,
) -> dict[str, float]:
    ic_by_date = calculate_spearman_ic_by_date(predictions)

    hit_rates = []
    top_returns = []
    for _, date_rows in predictions.groupby("date"):
        selected = date_rows.sort_values(
            ["predicted_return", "ticker"],
            ascending=[False, True],
        ).head(top_n)
        hit_rates.append((selected["actual_return"] > 0).mean())
        top_returns.append(selected["actual_return"].mean())

    return {
        "average_rank_ic": float(ic_by_date["rank_ic"].mean()),
        "average_top_n_hit_rate": float(pd.Series(hit_rates).mean()),
        "average_top_n_actual_return": float(pd.Series(top_returns).mean()),
    }


def main() -> None:
    args = parse_args()

    _, historical_rows, _ = load_baseline_dataset()
    dates = get_sorted_dates(historical_rows)
    feature_columns = get_model_feature_columns(historical_rows)

    print("TRAINING WINDOW DIAGNOSTIC")
    print("=" * 80)
    print("Model:", MODEL_NAME)
    print("Features:", len(feature_columns))
    print("Historical rows:", len(historical_rows))
    print("Historical dates:", len(dates))
    print("Step size:", args.step_size)
    print("Purge size:", args.purge_size)
    print("Train sizes:", args.train_sizes)
    print()

    # Every configuration is scored on exactly these test dates. The first one
    # must be reachable by the largest training window, so the comparison is
    # limited by the most data-hungry configuration rather than each one
    # starting wherever it can.
    earliest_test_position = (
        max(args.train_sizes)
        + args.purge_size
        + args.validation_size
        + args.purge_size
    )
    test_positions = list(
        range(
            earliest_test_position,
            len(dates) - args.test_size + 1,
            args.step_size,
        )
    )

    if not test_positions:
        raise ValueError("Not enough dates for the requested training sizes")

    print("Shared test positions:", len(test_positions))
    print(
        "Evaluation span:",
        dates[test_positions[0]].date(),
        "to",
        dates[test_positions[-1]].date(),
    )
    print()

    runs: dict[int, pd.DataFrame] = {}
    window_counts: dict[int, int] = {}
    train_row_counts: dict[int, int] = {}

    for train_size in args.train_sizes:
        windows = build_windows_for_shared_test_positions(
            dates=dates,
            test_positions=test_positions,
            train_size=train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            purge_size=args.purge_size,
        )

        if not windows:
            print(f"train_size={train_size}: no windows, skipping")
            continue

        sample_train_dates = windows[0]["train_dates"]
        train_rows = int(
            historical_rows["date"].isin(sample_train_dates).sum()
        )

        started = time.perf_counter()
        predictions, _ = predict_tree_models_for_windows(
            models={MODEL_NAME: create_tree_models()[MODEL_NAME]},
            historical_rows=historical_rows,
            windows=windows,
            feature_columns=feature_columns,
        )
        elapsed = time.perf_counter() - started

        runs[train_size] = predictions
        window_counts[train_size] = len(windows)
        train_row_counts[train_size] = train_rows

        print(
            f"train_size={train_size:>4}  windows={len(windows):>5}  "
            f"train_rows={train_rows:>6}  fitted in {elapsed:>6.1f}s",
            flush=True,
        )

    if not runs:
        raise ValueError("No configuration produced windows")

    # A larger training window pushes the first test date later, so each
    # configuration would otherwise be scored over a different period. Restrict
    # every configuration to the dates all of them evaluated, so the comparison
    # reflects the training window and not the evaluation period.
    shared_dates = set.intersection(
        *(set(predictions["date"].unique()) for predictions in runs.values())
    )

    if not shared_dates:
        raise ValueError("Configurations share no evaluation dates")

    summary_rows = []
    for train_size, predictions in runs.items():
        aligned = predictions[predictions["date"].isin(shared_dates)]
        metrics = summarize(aligned, top_n=args.top_n)
        summary_rows.append(
            {
                "train_size_dates": train_size,
                "approximate_train_rows": train_row_counts[train_size],
                "windows": window_counts[train_size],
                "evaluated_dates": aligned["date"].nunique(),
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("train_size_dates")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print()
    print("COMPARISON ON A COMMON EVALUATION WINDOW")
    print("-" * 80)
    print(f"Shared evaluation dates: {len(shared_dates)}")
    print()
    print(summary.round(6).to_string(index=False))
    print()
    print("Output:", output_path)
    print(
        "Diagnostic only. No production default, published table, or model "
        "configuration was changed."
    )


if __name__ == "__main__":
    main()
