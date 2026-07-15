from __future__ import annotations

import pandas as pd

from src.regime_policy_backtest import (
    HISTORICAL_TREE_PREDICTION_HORIZON_DAYS,
    build_historical_market_regimes,
    build_market_drawdown_overlay,
    build_non_overlapping_policy_returns,
    build_paired_overlay_returns,
    summarize_paired_overlay_stability,
    summarize_non_overlapping_policy_returns,
)


def market_data() -> pd.DataFrame:
    rows = []
    for ticker in ("AAA", "BBB", "CCC"):
        close = 100.0
        for date in pd.bdate_range("2025-01-01", periods=220):
            close *= 1.005 if date.day % 7 else 0.99
            rows.append({"date": date, "ticker": ticker, "adjusted_close": close})
    return pd.DataFrame(rows)


def predictions(periods: int = 30) -> pd.DataFrame:
    rows = []
    for date in pd.bdate_range("2025-07-01", periods=periods):
        for model_name, scores in {
            "gradient_boosting": [0.3, 0.2, 0.1],
            "random_forest": [0.2, 0.3, 0.1],
        }.items():
            for ticker, score, actual in zip(
                ("AAA", "BBB", "CCC"), scores, (0.04, 0.02, -0.01)
            ):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "model_name": model_name,
                        "predicted_return": score,
                        "actual_return": actual,
                    }
                )
    return pd.DataFrame(rows)


def test_non_overlapping_policy_backtest_caps_turnover_and_compounds() -> None:
    policy = {
        "trend_up": "random_forest",
        "trend_down": "cash",
        "high_volatility": "random_forest",
    }
    history = build_non_overlapping_policy_returns(
        predictions(),
        market_data(),
        policy=policy,
        top_n=2,
        holding_period_days=10,
        max_turnover=0.25,
    )
    summary = summarize_non_overlapping_policy_returns(history)

    assert len(history) >= 1
    assert history["portfolio_turnover"].max() <= 0.25
    assert history["settlement_compatible"].all()
    assert summary.loc[0, "rebalance_dates"] == len(history)


def test_historical_tree_prediction_horizon_is_explicit() -> None:
    assert HISTORICAL_TREE_PREDICTION_HORIZON_DAYS == 5


def test_non_overlapping_policy_accepts_rolling_rank_ensemble() -> None:
    policy = {
        "trend_up": "rolling_rank_ensemble",
        "trend_down": "rolling_rank_ensemble",
        "high_volatility": "rolling_rank_ensemble",
    }
    history = build_non_overlapping_policy_returns(
        predictions(),
        market_data(),
        policy=policy,
        top_n=2,
        holding_period_days=10,
    )

    assert not history.empty
    assert set(history["selected_model"]) == {"rolling_rank_ensemble"}


def test_pre_spaced_signals_are_not_subsampled_twice() -> None:
    source = predictions(periods=80)
    selected_dates = pd.DatetimeIndex(source["date"].drop_duplicates()).sort_values()[::10]
    pre_spaced = source.loc[source["date"].isin(selected_dates)].copy()
    available_regime_dates = set(build_historical_market_regimes(market_data())["date"])
    expected_dates = [date for date in selected_dates if date in available_regime_dates]
    history = build_non_overlapping_policy_returns(
        pre_spaced,
        market_data(),
        top_n=2,
        holding_period_days=10,
        signals_are_pre_spaced=True,
        rolling_performance_window=2,
        rolling_label_availability_lag=0,
    )

    assert len(expected_dates) > 1
    assert history["date"].tolist() == expected_dates


def test_pre_spaced_signals_reject_overlapping_dates() -> None:
    source = predictions()
    dense_dates = pd.DatetimeIndex(source["date"].drop_duplicates()).sort_values()[:2]
    dense = source.loc[source["date"].isin(dense_dates)].copy()

    try:
        build_non_overlapping_policy_returns(
            dense,
            market_data(),
            top_n=2,
            holding_period_days=10,
            signals_are_pre_spaced=True,
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Overlapping pre-spaced signals were accepted")


def test_policy_backtest_can_use_absolute_returns_separately_from_model_labels() -> None:
    frame = predictions()
    frame["actual_return"] = -0.50
    frame["portfolio_return"] = 0.10
    history = build_non_overlapping_policy_returns(
        frame,
        market_data(),
        top_n=2,
        holding_period_days=10,
        realized_return_column="portfolio_return",
    )

    assert (history["before_cost_return"] > 0.0).all()


def test_policy_summary_separates_portfolio_benchmark_and_active_returns() -> None:
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=3),
            "after_cost_return": [0.02, -0.01, 0.03],
            "benchmark_return": [0.01, 0.00, 0.01],
            "portfolio_turnover": [0.1, 0.1, 0.1],
            "target_exposure": [0.97, 0.97, 0.97],
            "forced_exit_weight": [0.0, 0.0, 0.0],
            "settlement_compatible": [True, True, True],
        }
    )
    summary = summarize_non_overlapping_policy_returns(history)

    assert "final_cumulative_benchmark_return" in summary
    assert "final_cumulative_active_return" in summary
    assert "max_active_drawdown" in summary


def test_policy_summary_measures_drawdown_from_initial_capital() -> None:
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=2),
            "after_cost_return": [-0.20, 0.10],
            "portfolio_turnover": [0.1, 0.1],
            "target_exposure": [0.97, 0.97],
            "forced_exit_weight": [0.0, 0.0],
            "settlement_compatible": [True, True],
        }
    )
    summary = summarize_non_overlapping_policy_returns(history)

    assert abs(summary.loc[0, "max_after_cost_drawdown"] + 0.20) < 1e-12


def test_non_overlapping_policy_skips_incomplete_realized_return_dates() -> None:
    incomplete = predictions()
    incomplete.loc[incomplete.index[0], "actual_return"] = float("nan")
    history = build_non_overlapping_policy_returns(
        incomplete,
        market_data(),
        top_n=2,
        holding_period_days=10,
    )

    assert not history.empty
    assert history["after_cost_return"].notna().all()


def test_non_overlapping_policy_forces_exit_when_held_ticker_disappears() -> None:
    complete = predictions()
    baseline = build_non_overlapping_policy_returns(
        complete,
        market_data(),
        policy={
            "trend_up": "gradient_boosting",
            "trend_down": "gradient_boosting",
            "high_volatility": "gradient_boosting",
        },
        top_n=2,
        holding_period_days=10,
    )
    assert len(baseline) >= 2

    second_date = baseline.iloc[1]["date"]
    missing_ticker = "AAA"
    reduced = complete.loc[
        ~((complete["date"] == second_date) & (complete["ticker"] == missing_ticker))
    ].copy()
    history = build_non_overlapping_policy_returns(
        reduced,
        market_data(),
        policy={
            "trend_up": "gradient_boosting",
            "trend_down": "gradient_boosting",
            "high_volatility": "gradient_boosting",
        },
        top_n=2,
        holding_period_days=10,
    )

    exit_row = history.loc[history["date"] == second_date].iloc[0]
    assert exit_row["forced_exit_weight"] > 0.0
    assert pd.notna(exit_row["after_cost_return"])


def test_drawdown_overlay_reduces_exposure_without_using_future_prices() -> None:
    falling = market_data().copy()
    falling["adjusted_close"] = falling.groupby("ticker").cumcount().map(
        lambda index: 100.0 * 0.99**index
    )
    overlay = build_market_drawdown_overlay(
        falling,
        trigger_drawdown=-0.10,
        reduced_exposure=0.50,
    )
    history = build_non_overlapping_policy_returns(
        predictions(),
        falling,
        top_n=2,
        holding_period_days=10,
        target_exposure_by_date=overlay.set_index("date")["target_exposure"],
    )

    assert overlay["risk_off"].any()
    assert (overlay.loc[overlay["risk_off"], "target_exposure"] == 0.50).all()
    assert history["target_exposure"].min() == 0.50


def test_paired_overlay_stability_uses_matched_rebalance_dates() -> None:
    baseline = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=8),
            "after_cost_return": [0.001] * 8,
        }
    )
    guarded = baseline.copy()
    guarded["after_cost_return"] = [0.002, 0.001, 0.003, 0.001, 0.002, 0.002, 0.001, 0.003]
    paired = build_paired_overlay_returns(baseline, guarded)
    summary = summarize_paired_overlay_stability(
        paired,
        block_size=2,
        bootstrap_samples=100,
    )

    assert len(paired) == 8
    assert (paired["after_cost_return_difference"] >= 0.0).all()
    assert summary.loc[0, "positive_rebalance_dates"] == 5
    assert summary.loc[0, "bootstrap_95pct_lower"] >= 0.0
