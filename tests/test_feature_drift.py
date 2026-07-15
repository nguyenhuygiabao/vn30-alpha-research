from __future__ import annotations

import pandas as pd

from src.feature_drift import build_feature_drift_report, population_stability_index


def feature_rows() -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.bdate_range("2025-01-01", periods=40)):
        for ticker in ("AAA", "BBB"):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "stable_feature": float(date_index % 5),
                    "shifted_feature": (
                        0.0
                        if date_index < 20
                        else 1.0
                        if date_index < 30
                        else 10.0
                    ),
                    "boolean_feature": date_index >= 30,
                }
            )
    return pd.DataFrame(rows)


def test_population_stability_index_detects_distribution_shift() -> None:
    psi = population_stability_index(
        pd.Series([0.0] * 20 + [1.0] * 20),
        pd.Series([10.0] * 40),
        bins=4,
    )
    assert psi > 0.25


def test_population_stability_index_detects_shift_from_constant_reference() -> None:
    psi = population_stability_index(
        pd.Series([0.0] * 40),
        pd.Series([10.0] * 40),
        bins=4,
    )
    assert psi > 0.25


def test_feature_drift_report_labels_recent_shift() -> None:
    report = build_feature_drift_report(
        feature_rows(),
        reference_window_dates=20,
        recent_window_dates=10,
        bins=4,
    )
    shifted = report.loc[report["feature"] == "shifted_feature"].iloc[0]

    assert shifted["drift_status"] == "significant_shift"
    assert shifted["recent_mean"] > shifted["reference_mean"]


def test_feature_drift_report_supports_boolean_model_features() -> None:
    report = build_feature_drift_report(
        feature_rows(),
        reference_window_dates=20,
        recent_window_dates=10,
        bins=4,
    )
    boolean = report.loc[report["feature"] == "boolean_feature"].iloc[0]

    assert boolean["drift_status"] == "significant_shift"
    assert boolean["recent_mean"] == 1.0
