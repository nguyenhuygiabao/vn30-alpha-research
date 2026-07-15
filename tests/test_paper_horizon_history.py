from __future__ import annotations

import numpy as np
import pandas as pd

from src.paper_horizon_history import build_historical_paper_predictions
from src.walk_forward_split import TARGET_COLUMN


def modeling_dataset() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2025-01-01", periods=20)
    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "feature_one": float(date_index + ticker_index),
                    "feature_two": float((date_index + 1) * (ticker_index + 1)),
                    TARGET_COLUMN: (
                        np.nan
                        if date_index >= len(dates) - 2
                        else float(ticker_index - 1) / 100.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_paper_horizon_history_enforces_cutoff_and_signal_spacing() -> None:
    predictions = build_historical_paper_predictions(
        modeling_dataset(),
        horizon_days=2,
        minimum_training_dates=6,
        signal_step_days=2,
    )

    assert set(predictions["model_name"]) == {
        "gradient_boosting",
        "random_forest",
    }
    assert (predictions["horizon_days"] == 2).all()
    assert (
        (predictions["date"] - predictions["training_cutoff"]).dt.days > 0
    ).all()
    signal_dates = pd.DatetimeIndex(predictions["date"].drop_duplicates()).sort_values()
    assert len(signal_dates) >= 2
    all_dates = pd.DatetimeIndex(modeling_dataset()["date"].drop_duplicates()).sort_values()
    signal_positions = [all_dates.get_loc(date) for date in signal_dates]
    assert np.diff(signal_positions).tolist() == [2] * (len(signal_positions) - 1)
    assert predictions["actual_return"].notna().all()


def test_paper_horizon_history_resumes_from_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "paper_history.partial.csv"
    first = build_historical_paper_predictions(
        modeling_dataset(),
        horizon_days=2,
        minimum_training_dates=6,
        signal_step_days=2,
        checkpoint_path=str(checkpoint_path),
    )
    resumed = build_historical_paper_predictions(
        modeling_dataset(),
        horizon_days=2,
        minimum_training_dates=6,
        signal_step_days=2,
        checkpoint_path=str(checkpoint_path),
    )

    assert checkpoint_path.exists()
    pd.testing.assert_frame_equal(first, resumed)
