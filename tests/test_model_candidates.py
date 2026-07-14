from __future__ import annotations

import pandas as pd

from src.model_candidates import (
    RANK_ENSEMBLE_MODEL_NAME,
    build_rank_ensemble_history,
    summarize_model_candidates,
)


def predictions() -> pd.DataFrame:
    rows = []
    for market_date in pd.to_datetime(["2026-07-01", "2026-07-02"]):
        for model_name, scores in {
            "gradient_boosting": [0.3, 0.2, 0.1],
            "random_forest": [0.2, 0.3, 0.1],
        }.items():
            for ticker, score, actual in zip(
                ["AAA", "BBB", "CCC"], scores, [0.04, 0.02, -0.01]
            ):
                rows.append(
                    {
                        "date": market_date,
                        "ticker": ticker,
                        "model_name": model_name,
                        "predicted_return": score,
                        "actual_return": actual,
                    }
                )
    return pd.DataFrame(rows)


def test_rank_ensemble_requires_matched_member_coverage() -> None:
    frame = predictions().iloc[:-1]
    try:
        build_rank_ensemble_history(frame)
    except ValueError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("Expected incomplete member coverage to be rejected")


def test_rank_ensemble_and_candidate_summary_are_out_of_sample_only() -> None:
    ensemble = build_rank_ensemble_history(predictions())
    summary = summarize_model_candidates(predictions(), top_n=2)

    assert len(ensemble) == 6
    assert set(ensemble["model_name"]) == {RANK_ENSEMBLE_MODEL_NAME}
    assert set(summary["model_name"]) == {
        "gradient_boosting",
        "random_forest",
        RANK_ENSEMBLE_MODEL_NAME,
    }
    assert (summary["evaluated_dates"] == 2).all()
