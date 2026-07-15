from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_candidates import build_daily_candidate_metrics


def build_model_health_history(
    predictions: pd.DataFrame,
    top_n: int = 8,
    rolling_window: int = 126,
) -> pd.DataFrame:
    """Build rolling out-of-sample prediction-health diagnostics by candidate."""
    if rolling_window <= 1:
        raise ValueError("rolling_window must be greater than one")

    daily = build_daily_candidate_metrics(predictions, top_n=top_n)
    rows: list[pd.DataFrame] = []
    for model_name, model_daily in daily.groupby("model_name"):
        ordered = model_daily.sort_values("date").copy()
        ordered["rolling_rank_ic"] = ordered["rank_ic"].rolling(
            rolling_window,
            min_periods=rolling_window,
        ).mean()
        ordered["rolling_top_n_actual_return"] = ordered[
            "top_n_actual_return"
        ].rolling(
            rolling_window,
            min_periods=rolling_window,
        ).mean()
        ordered["rolling_rank_ic_volatility"] = ordered["rank_ic"].rolling(
            rolling_window,
            min_periods=rolling_window,
        ).std()
        rows.append(ordered)

    return pd.concat(rows, ignore_index=True).sort_values(
        ["date", "model_name"]
    ).reset_index(drop=True)


def summarize_latest_model_health(
    health_history: pd.DataFrame,
    minimum_rank_ic: float = 0.005,
) -> pd.DataFrame:
    """Summarize latest available rolling health without selecting a live model."""
    required = {
        "date",
        "model_name",
        "rolling_rank_ic",
        "rolling_top_n_actual_return",
        "rolling_rank_ic_volatility",
    }
    missing = sorted(required.difference(health_history.columns))
    if missing:
        raise ValueError(f"Health history is missing columns: {missing}")
    if minimum_rank_ic < 0.0:
        raise ValueError("minimum_rank_ic cannot be negative")

    rows: list[dict[str, object]] = []
    for model_name, history in health_history.groupby("model_name"):
        available = history.dropna(subset=["rolling_rank_ic"])
        if available.empty:
            status = "insufficient_history"
            latest = None
        else:
            latest = available.sort_values("date").iloc[-1]
            if latest["rolling_rank_ic"] <= 0.0:
                status = "degraded"
            elif latest["rolling_rank_ic"] < minimum_rank_ic:
                status = "weak"
            else:
                status = "positive"
        rows.append(
            {
                "model_name": model_name,
                "health_status": status,
                "latest_health_date": None if latest is None else latest["date"],
                "latest_rolling_rank_ic": (
                    np.nan if latest is None else latest["rolling_rank_ic"]
                ),
                "latest_rolling_top_n_actual_return": (
                    np.nan
                    if latest is None
                    else latest["rolling_top_n_actual_return"]
                ),
                "latest_rolling_rank_ic_volatility": (
                    np.nan
                    if latest is None
                    else latest["rolling_rank_ic_volatility"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
