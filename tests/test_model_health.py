from __future__ import annotations

from src.model_health import build_model_health_history, summarize_latest_model_health

from tests.test_model_candidates import rolling_predictions


def test_model_health_reports_latest_rolling_out_of_sample_metrics() -> None:
    history = build_model_health_history(
        rolling_predictions(),
        top_n=2,
        rolling_window=3,
    )
    summary = summarize_latest_model_health(history, minimum_rank_ic=0.005)

    assert set(summary["model_name"]) == {
        "gradient_boosting",
        "random_forest",
        "rank_ensemble",
        "rolling_rank_ensemble",
    }
    assert summary["latest_rolling_rank_ic"].notna().sum() >= 2
    assert set(summary["health_status"]).issubset(
        {"positive", "weak", "degraded", "insufficient_history"}
    )
