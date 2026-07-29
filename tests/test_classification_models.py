from __future__ import annotations

import numpy as np
import pandas as pd

from src.classification_models import (
    TOP_QUINTILE_LABEL_COLUMN,
    add_top_quintile_label,
    predict_classification_windows,
    predict_one_classification_window,
)
from src.walk_forward_split import TARGET_COLUMN, build_walk_forward_date_windows


FEATURE_COLUMNS = ["feat1", "feat2"]
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def synthetic_labeled_rows(date_count: int = 20) -> pd.DataFrame:
    rng = np.random.RandomState(13)
    dates = pd.date_range("2026-01-01", periods=date_count, freq="D")

    rows = []
    for date in dates:
        for ticker in TICKERS:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "feat1": rng.normal(),
                    "feat2": rng.normal(),
                    TARGET_COLUMN: rng.normal(),
                }
            )

    return add_top_quintile_label(pd.DataFrame(rows))


def small_windows(historical_rows: pd.DataFrame) -> list[dict[str, pd.DatetimeIndex]]:
    dates = pd.DatetimeIndex(
        historical_rows["date"].drop_duplicates().sort_values()
    )

    return build_walk_forward_date_windows(
        dates=dates,
        train_size=5,
        validation_size=1,
        test_size=1,
        purge_size=1,
        step_size=1,
    )


def test_parallel_windows_match_sequential_single_window_calls() -> None:
    historical_rows = synthetic_labeled_rows()
    windows = small_windows(historical_rows)
    assert len(windows) >= 5, "test needs several windows to be meaningful"

    parallel_predictions = predict_classification_windows(
        historical_rows=historical_rows,
        windows=windows,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=2,
    )

    expected_frames = [
        predict_one_classification_window(
            historical_rows=historical_rows,
            window=window,
            feature_columns=FEATURE_COLUMNS,
        )
        for window in windows
    ]
    expected_predictions = pd.concat(expected_frames, ignore_index=True)

    pd.testing.assert_frame_equal(
        parallel_predictions.reset_index(drop=True),
        expected_predictions.reset_index(drop=True),
    )


def test_parallel_result_independent_of_n_jobs() -> None:
    historical_rows = synthetic_labeled_rows()
    windows = small_windows(historical_rows)

    single_job = predict_classification_windows(
        historical_rows=historical_rows,
        windows=windows,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=1,
    )
    multi_job = predict_classification_windows(
        historical_rows=historical_rows,
        windows=windows,
        feature_columns=FEATURE_COLUMNS,
        n_jobs=-1,
    )

    pd.testing.assert_frame_equal(single_job, multi_job)


def test_empty_windows_return_empty_frame() -> None:
    historical_rows = synthetic_labeled_rows()

    predictions = predict_classification_windows(
        historical_rows=historical_rows,
        windows=[],
        feature_columns=FEATURE_COLUMNS,
    )

    assert predictions.empty
    assert TOP_QUINTILE_LABEL_COLUMN not in predictions.columns
