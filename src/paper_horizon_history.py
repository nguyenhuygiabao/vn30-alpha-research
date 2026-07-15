from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.modeling_utils import get_model_feature_columns
from src.tree_models import create_tree_models, fit_predict_tree_model
from src.walk_forward_split import KEY_COLUMNS, TARGET_COLUMN


PAPER_HORIZON_PREDICTIONS_PATH = "data/processed/paper_horizon_predictions.parquet"


def build_historical_paper_predictions(
    modeling_dataset: pd.DataFrame,
    horizon_days: int = 10,
    minimum_training_dates: int = 252,
    signal_step_days: int = 10,
    checkpoint_path: str | None = None,
    progress_callback: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> pd.DataFrame:
    """Build expanding-window predictions aligned to a paper forecast horizon."""
    if horizon_days <= 0 or minimum_training_dates <= 0 or signal_step_days <= 0:
        raise ValueError("Horizon, training dates, and signal step must be positive")
    required = {"date", "ticker", TARGET_COLUMN}
    missing = sorted(required.difference(modeling_dataset.columns))
    if missing:
        raise ValueError(f"Modeling dataset is missing columns: {missing}")

    working = modeling_dataset.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.strip().str.upper()
    if working.duplicated(KEY_COLUMNS).any():
        raise ValueError("Modeling dataset contains duplicate ticker-date keys")
    dates = pd.DatetimeIndex(working["date"].drop_duplicates()).sort_values()
    first_signal_position = minimum_training_dates - 1 + horizon_days
    if first_signal_position >= len(dates):
        raise ValueError("Not enough dates for the requested expanding-window history")

    models = create_tree_models()
    random_forest = models.get("random_forest")
    if random_forest is not None:
        random_forest.set_params(n_jobs=-1)

    signal_positions = list(range(
        first_signal_position,
        len(dates),
        signal_step_days,
    ))
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    frames: list[pd.DataFrame] = []
    completed_keys: set[tuple[pd.Timestamp, str]] = set()
    if checkpoint is not None and checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
        existing["training_cutoff"] = pd.to_datetime(
            existing["training_cutoff"]
        ).dt.normalize()
        if not existing.empty:
            if not (existing["horizon_days"] == horizon_days).all():
                raise ValueError("Checkpoint forecast horizon does not match")
            frames.append(existing)
            completed_keys = set(
                existing[["date", "model_name"]].itertuples(index=False, name=None)
            )

    for completed_count, signal_position in enumerate(signal_positions, start=1):
        signal_date = dates[signal_position]
        training_cutoff = dates[signal_position - horizon_days]
        training = working.loc[
            (working["date"] <= training_cutoff)
            & working[TARGET_COLUMN].notna()
        ].copy()
        signal_rows = working.loc[working["date"] == signal_date].copy()
        if signal_rows.empty or signal_rows[TARGET_COLUMN].isna().any():
            continue
        training_dates = training["date"].nunique()
        if training_dates < minimum_training_dates:
            continue
        feature_columns = get_model_feature_columns(training)
        if not feature_columns:
            raise ValueError("No usable model features are available")

        for model_name, model in models.items():
            if (signal_date, model_name) in completed_keys:
                continue
            predicted, _ = fit_predict_tree_model(
                model=model,
                x_train=training[feature_columns],
                y_train=training[TARGET_COLUMN],
                x_test=signal_rows[feature_columns],
            )
            frame = signal_rows[KEY_COLUMNS + [TARGET_COLUMN]].copy()
            frame["model_name"] = model_name
            frame["predicted_return"] = np.asarray(predicted, dtype=float)
            frame["actual_return"] = frame.pop(TARGET_COLUMN)
            frame["horizon_days"] = horizon_days
            frame["training_cutoff"] = training_cutoff
            frame["training_date_count"] = training_dates
            frames.append(frame)
            completed_keys.add((signal_date, model_name))
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(frames, ignore_index=True).to_csv(
                    checkpoint,
                    index=False,
                )
        if progress_callback is not None:
            progress_callback(completed_count, len(signal_positions), signal_date)

    if not frames:
        raise ValueError("No complete historical paper-horizon signals were produced")
    predictions = pd.concat(frames, ignore_index=True)
    return predictions[
        [
            "date",
            "ticker",
            "model_name",
            "predicted_return",
            "actual_return",
            "horizon_days",
            "training_cutoff",
            "training_date_count",
        ]
    ].sort_values(["date", "model_name", "ticker"]).reset_index(drop=True)


def save_historical_paper_predictions(
    predictions: pd.DataFrame,
    output_path: str = PAPER_HORIZON_PREDICTIONS_PATH,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    return str(path)
