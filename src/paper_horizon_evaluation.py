from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.labels import build_forward_labels
from src.regime_policy_backtest import (
    build_non_overlapping_policy_returns,
    paired_block_bootstrap_mean_difference,
    summarize_non_overlapping_policy_returns,
)


def prepare_paper_horizon_inputs(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    horizon_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    prepared_predictions = predictions.copy()
    prepared_market = market_data.copy()
    prepared_predictions["date"] = pd.to_datetime(
        prepared_predictions["date"]
    ).dt.normalize()
    prepared_market["date"] = pd.to_datetime(prepared_market["date"]).dt.normalize()
    horizons = prepared_predictions["horizon_days"].dropna().unique()
    if len(horizons) != 1 or int(horizons[0]) != horizon_days:
        raise ValueError(
            f"Paper-horizon predictions must use one {horizon_days}-day horizon"
        )

    absolute_returns = build_forward_labels(prepared_market)[
        ["date", "ticker", f"forward_return_{horizon_days}d"]
    ].rename(columns={f"forward_return_{horizon_days}d": "portfolio_return"})
    prepared_predictions = prepared_predictions.merge(
        absolute_returns,
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )
    benchmark_returns = absolute_returns.groupby("date")["portfolio_return"].mean()
    return prepared_predictions, prepared_market, benchmark_returns


def build_paper_policy_histories(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    benchmark_returns: pd.Series,
    policies: Mapping[str, Mapping[str, str]],
    top_n: int = 8,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for policy_name, policy in policies.items():
        history = build_non_overlapping_policy_returns(
            predictions,
            market_data,
            policy=policy,
            top_n=top_n,
            holding_period_days=10,
            signals_are_pre_spaced=True,
            rolling_performance_window=13,
            rolling_label_availability_lag=0,
            realized_return_column="portfolio_return",
        )
        history["benchmark_return"] = history["date"].map(benchmark_returns)
        if history["benchmark_return"].isna().any():
            raise ValueError("Equal-weight benchmark is incomplete on backtest dates")
        histories[policy_name] = history
    return histories


def summarize_paper_policy_periods(
    histories: Mapping[str, pd.DataFrame],
    holdout_start: str | pd.Timestamp = "2024-01-01",
) -> pd.DataFrame:
    cutoff = pd.Timestamp(holdout_start)
    rows = []
    for policy_name, history in histories.items():
        for period, mask in {
            "training": history["date"] < cutoff,
            "holdout": history["date"] >= cutoff,
        }.items():
            subset = history.loc[mask].copy()
            if subset.empty:
                continue
            summary = summarize_non_overlapping_policy_returns(subset)
            summary.insert(0, "period", period)
            summary.insert(0, "policy_name", policy_name)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def summarize_holdout_active_stability(
    histories: Mapping[str, pd.DataFrame],
    holdout_start: str | pd.Timestamp = "2024-01-01",
    block_size: int = 3,
    bootstrap_samples: int = 2_000,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(holdout_start)
    rows = []
    for policy_name, history in histories.items():
        holdout = history.loc[history["date"] >= cutoff]
        active_returns = holdout["after_cost_return"] - holdout["benchmark_return"]
        mean, lower, upper = paired_block_bootstrap_mean_difference(
            active_returns,
            block_size=block_size,
            bootstrap_samples=bootstrap_samples,
        )
        rows.append(
            {
                "policy_name": policy_name,
                "holdout_dates": len(holdout),
                "mean_active_return": mean,
                "bootstrap_95pct_lower": lower,
                "bootstrap_95pct_upper": upper,
                "positive_active_dates": int((active_returns > 0.0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_active_return", ascending=False
    ).reset_index(drop=True)


def summarize_paper_policy_years(
    histories: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for policy_name, history in histories.items():
        working = history.copy()
        working["year"] = pd.to_datetime(working["date"]).dt.year
        for year, yearly in working.groupby("year"):
            summary = summarize_non_overlapping_policy_returns(yearly).iloc[0]
            rows.append(
                {
                    "policy_name": policy_name,
                    "year": int(year),
                    "rebalance_dates": int(summary["rebalance_dates"]),
                    "cumulative_portfolio_return": summary[
                        "final_cumulative_after_cost_return"
                    ],
                    "cumulative_benchmark_return": summary[
                        "final_cumulative_benchmark_return"
                    ],
                    "cumulative_active_return": summary[
                        "final_cumulative_active_return"
                    ],
                    "max_portfolio_drawdown": summary["max_after_cost_drawdown"],
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "policy_name"]).reset_index(drop=True)
