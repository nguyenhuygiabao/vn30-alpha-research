from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling_utils import get_model_feature_columns


def population_stability_index(
    reference: pd.Series,
    recent: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI using bins fixed from the earlier reference distribution."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    reference_values = (
        pd.to_numeric(reference, errors="coerce").dropna().astype(float)
    )
    recent_values = (
        pd.to_numeric(recent, errors="coerce").dropna().astype(float)
    )
    if len(reference_values) < bins or recent_values.empty:
        return float("nan")

    edges = np.unique(np.quantile(reference_values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        center = float(reference_values.iloc[0])
        tolerance = max(abs(center) * 1e-6, 1e-6)
        edges = np.array([-np.inf, center - tolerance, center + tolerance, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    recent_counts, _ = np.histogram(recent_values, bins=edges)
    epsilon = 1e-6
    reference_share = (reference_counts + epsilon) / (reference_counts.sum() + epsilon * len(reference_counts))
    recent_share = (recent_counts + epsilon) / (recent_counts.sum() + epsilon * len(recent_counts))
    return float(np.sum((recent_share - reference_share) * np.log(recent_share / reference_share)))


def build_feature_drift_report(
    features: pd.DataFrame,
    reference_window_dates: int = 756,
    recent_window_dates: int = 126,
    bins: int = 10,
) -> pd.DataFrame:
    """Compare the latest feature distribution with a preceding training window."""
    if reference_window_dates <= 0 or recent_window_dates <= 0:
        raise ValueError("Feature-drift windows must be positive")
    if "date" not in features.columns:
        raise ValueError("Features are missing date")

    working = features.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise")
    dates = pd.DatetimeIndex(working["date"].drop_duplicates()).sort_values()
    required_dates = reference_window_dates + recent_window_dates
    if len(dates) < required_dates:
        raise ValueError(
            f"Only {len(dates)} feature dates are available; need {required_dates}"
        )
    reference_dates = dates[-required_dates:-recent_window_dates]
    recent_dates = dates[-recent_window_dates:]
    reference_rows = working.loc[working["date"].isin(reference_dates)]
    recent_rows = working.loc[working["date"].isin(recent_dates)]
    rows: list[dict[str, object]] = []

    for feature in get_model_feature_columns(working):
        reference = pd.to_numeric(
            reference_rows[feature], errors="coerce"
        ).astype(float)
        recent = pd.to_numeric(
            recent_rows[feature], errors="coerce"
        ).astype(float)
        if reference.notna().sum() < bins or recent.notna().sum() == 0:
            continue
        psi = population_stability_index(reference, recent, bins=bins)
        if psi >= 0.25:
            status = "significant_shift"
        elif psi >= 0.10:
            status = "moderate_shift"
        else:
            status = "stable"
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "drift_status": status,
                "reference_mean": reference.mean(),
                "recent_mean": recent.mean(),
                "reference_missing_rate": reference.isna().mean(),
                "recent_missing_rate": recent.isna().mean(),
            }
        )

    if not rows:
        raise ValueError("No usable numeric model features were available for drift")
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
