from __future__ import annotations

import pandas as pd

from src.model_candidates import (
    DEFAULT_REGIME_POLICY,
    RANK_ENSEMBLE_MODEL_NAME,
    ROLLING_RANK_ENSEMBLE_MODEL_NAME,
    build_rank_ensemble_history,
    build_rolling_rank_ensemble_history,
    build_historical_market_regimes,
    summarize_candidates_by_market_regime,
    evaluate_regime_policy,
    summarize_regime_policy,
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


def test_rank_ensemble_tolerates_only_round_trip_return_precision() -> None:
    rounded = predictions()
    random_forest = rounded["model_name"] == "random_forest"
    rounded.loc[random_forest, "actual_return"] += 1e-14
    ensemble = build_rank_ensemble_history(rounded)
    assert not ensemble.empty

    inconsistent = predictions()
    inconsistent.loc[random_forest, "actual_return"] += 1e-4
    try:
        build_rank_ensemble_history(inconsistent)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("Materially inconsistent realized returns were accepted")


def test_rank_ensemble_and_candidate_summary_are_out_of_sample_only() -> None:
    ensemble = build_rank_ensemble_history(predictions())
    summary = summarize_model_candidates(predictions(), top_n=2)

    assert len(ensemble) == 6
    assert set(ensemble["model_name"]) == {RANK_ENSEMBLE_MODEL_NAME}
    assert set(summary["model_name"]) == {
        "gradient_boosting",
        "random_forest",
        RANK_ENSEMBLE_MODEL_NAME,
        ROLLING_RANK_ENSEMBLE_MODEL_NAME,
    }
    assert (summary["evaluated_dates"] == 2).all()


def rolling_predictions() -> pd.DataFrame:
    rows = []
    for market_date in pd.bdate_range("2025-01-01", periods=8):
        for model_name, scores in {
            "gradient_boosting": [0.3, 0.2, 0.1],
            "random_forest": [0.2, 0.1, 0.3],
        }.items():
            for ticker, score, actual in zip(
                ["AAA", "BBB", "CCC"], scores, [0.03, 0.01, -0.02]
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


def test_rolling_rank_ensemble_uses_only_lagged_member_performance() -> None:
    frame = rolling_predictions()
    rolling = build_rolling_rank_ensemble_history(
        frame,
        performance_window=3,
        label_availability_lag=1,
    )
    altered = frame.copy()
    last_date = altered["date"].max()
    altered.loc[altered["date"] == last_date, "actual_return"] *= -1.0
    altered_rolling = build_rolling_rank_ensemble_history(
        altered,
        performance_window=3,
        label_availability_lag=1,
    )

    latest = rolling.loc[rolling["date"] == last_date]
    latest_altered = altered_rolling.loc[altered_rolling["date"] == last_date]
    assert (latest["weight_gradient_boosting"] > latest["weight_random_forest"]).all()
    assert latest["predicted_return"].tolist() == latest_altered["predicted_return"].tolist()


def test_paired_stability_reports_matched_dates_and_confidence_interval() -> None:
    stability = summarize_paired_candidate_stability(
        predictions(),
        top_n=2,
        rolling_window=2,
    )

    assert set(stability["baseline_model"]) == {
        "gradient_boosting",
        "random_forest",
        ROLLING_RANK_ENSEMBLE_MODEL_NAME,
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
        ROLLING_RANK_ENSEMBLE_MODEL_NAME,
    }
    assert (summary["evaluated_dates"] > 0).all()


def test_regime_policy_can_hold_cash_in_downtrends() -> None:
    data = market_data()
    data = data.sort_values(["ticker", "date"]).copy()
    data["adjusted_close"] = data.groupby("ticker").cumcount().map(
        lambda index: 100.0 * 0.99**index
    )
    prediction_dates = pd.bdate_range("2025-06-02", periods=40)
    rows = []
    for market_date in prediction_dates:
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
    history = evaluate_regime_policy(
        pd.DataFrame(rows),
        data,
        top_n=2,
        return_window=5,
        volatility_baseline_window=10,
    )
    summary = summarize_regime_policy(history)

    assert DEFAULT_REGIME_POLICY["trend_down"] == "cash"
    assert (history["selected_model"] == "cash").any()
    assert (history.loc[history["selected_model"] == "cash", "top_n_actual_return"] == 0.0).all()
    assert summary.loc[0, "cash_dates"] > 0
