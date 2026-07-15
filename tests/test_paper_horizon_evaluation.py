from __future__ import annotations

import pandas as pd

from src.paper_horizon_evaluation import (
    summarize_holdout_active_stability,
    summarize_paper_policy_periods,
    summarize_paper_policy_years,
)


def histories() -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(
        ["2023-11-01", "2023-12-01", "2024-01-02", "2024-02-01"]
    )
    return {
        "candidate": pd.DataFrame(
            {
                "date": dates,
                "after_cost_return": [0.02, 0.01, 0.03, -0.01],
                "benchmark_return": [0.01, 0.01, 0.01, 0.00],
                "portfolio_turnover": [0.1] * 4,
                "target_exposure": [0.97] * 4,
                "forced_exit_weight": [0.0] * 4,
                "settlement_compatible": [True] * 4,
            }
        )
    }


def test_paper_horizon_period_and_year_summaries_do_not_cross_boundaries() -> None:
    periods = summarize_paper_policy_periods(histories(), "2024-01-01")
    years = summarize_paper_policy_years(histories())

    assert set(periods["period"]) == {"training", "holdout"}
    assert (periods["rebalance_dates"] == 2).all()
    assert set(years["year"]) == {2023, 2024}
    assert (years["rebalance_dates"] == 2).all()


def test_paper_horizon_holdout_stability_uses_only_holdout_dates() -> None:
    stability = summarize_holdout_active_stability(
        histories(),
        "2024-01-01",
        block_size=1,
        bootstrap_samples=100,
    )

    assert stability.loc[0, "holdout_dates"] == 2
    assert stability.loc[0, "positive_active_dates"] == 1
