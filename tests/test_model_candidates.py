from __future__ import annotations

import pandas as pd

from src.model_candidates import (
    RANK_ENSEMBLE_MODEL_NAME,
    build_rank_ensemble_history,
    build_historical_market_regimes,
    summarize_candidates_by_market_regime,
    summarize_model_candidates,
    summarize_paired_candidate_stability,
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


def test_paired_stability_reports_matched_dates_and_confidence_interval() -> None:
    stability = summarize_paired_candidate_stability(
        predictions(),
        top_n=2,
        rolling_window=2,
    )

    assert set(stability["baseline_model"]) == {
        "gradient_boosting",
        "random_forest",
    }
    assert (stability["paired_dates"] == 2).all()
    assert (stability["bootstrap_95pct_lower"] <= stability["bootstrap_95pct_upper"]).all()


def market_data() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2025-01-01", periods=160)
    for ticker, multiplier in [("AAA", 1.0), ("BBB", 1.1), ("CCC", 0.9)]:
        close = 100.0
        for index, market_date in enumerate(dates):
            close *= 1.01 if index % 9 else 0.98
            rows.append(
                {
                    "date": market_date,
                    "ticker": ticker,
                    "adjusted_close": close * multiplier,
                }
            )
    return pd.DataFrame(rows)


def test_market_regimes_are_historical_and_candidate_summary_is_joined() -> None:
    regimes = build_historical_market_regimes(
        market_data(),
        return_window=5,
        volatility_baseline_window=10,
    )
    repeated_predictions = pd.concat([predictions()] * 80, ignore_index=True)
    repeated_predictions["date"] = pd.bdate_range(
        "2025-03-03", periods=len(repeated_predictions) // 6
    ).repeat(6).to_numpy()
    summary = summarize_candidates_by_market_regime(
        repeated_predictions,
        market_data(),
        top_n=2,
        return_window=5,
        volatility_baseline_window=10,
    )

    assert not regimes.empty
    assert set(regimes["market_regime"]).issubset(
        {"trend_up", "trend_down", "high_volatility"}
    )
    assert set(summary["model_name"]) == {
        "gradient_boosting",
        "random_forest",
        RANK_ENSEMBLE_MODEL_NAME,
    }
    assert (summary["evaluated_dates"] > 0).all()
