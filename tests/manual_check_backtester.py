from __future__ import annotations

import pandas as pd

from src.backtester import (
    BACKTEST_RETURNS_PATH,
    OPTIMIZED_WEIGHTS_PATH,
    calculate_performance_summary,
)


TOLERANCE: float = 1e-8


REQUIRED_COLUMNS: list[str] = [
    "optimization_mode",
    "execution_mode",
    "date",
    "before_cost_return",
    "portfolio_turnover",
    "selected_count",
    "total_weight",
    "low_liquidity_weight",
    "commission_cost",
    "slippage_cost",
    "liquidity_penalty",
    "total_cost",
    "after_cost_return",
    "cumulative_before_cost_active_return",
    "cumulative_after_cost_active_return",
]


def main() -> None:
    backtest_returns = pd.read_parquet(BACKTEST_RETURNS_PATH)
    optimized_weights = pd.read_parquet(OPTIMIZED_WEIGHTS_PATH)

    backtest_returns["date"] = pd.to_datetime(backtest_returns["date"])
    optimized_weights["date"] = pd.to_datetime(optimized_weights["date"])

    required_columns_present = all(
        column in backtest_returns.columns
        for column in REQUIRED_COLUMNS
    )

    one_row_per_date_per_mode = not backtest_returns.duplicated(
    [
        "optimization_mode",
        "execution_mode",
        "date",
    ]
    ).any()

    execution_modes = sorted(backtest_returns["execution_mode"].unique().tolist())
    optimization_modes = sorted(
    backtest_returns["optimization_mode"].unique().tolist()
    )

    expected_execution_modes_present = execution_modes == [
        "normal",
        "price_limit_aware",
    ]

    expected_optimization_modes_present = optimization_modes == [
    "herding_aware",
    "normal",
    ]

    dates_match_weights = (
    backtest_returns.groupby(
        [
            "optimization_mode",
            "execution_mode",
        ]
    )["date"].nunique()
    == optimized_weights["date"].nunique()
    ).all()

    total_weight_respected = (
        backtest_returns["total_weight"].max()
        <= 1.0 + TOLERANCE
    )

    turnover_respected = (
        backtest_returns["portfolio_turnover"].max()
        <= 0.40 + TOLERANCE
    )

    costs_non_negative = (
        backtest_returns[
            [
                "commission_cost",
                "slippage_cost",
                "liquidity_penalty",
                "total_cost",
            ]
        ]
        .min()
        .min()
        >= 0
    )

    total_cost_matches_components = (
        (
            backtest_returns["commission_cost"]
            + backtest_returns["slippage_cost"]
            + backtest_returns["liquidity_penalty"]
            - backtest_returns["total_cost"]
        )
        .abs()
        .max()
        <= TOLERANCE
    )

    after_cost_matches_formula = (
        (
            backtest_returns["before_cost_return"]
            - backtest_returns["total_cost"]
            - backtest_returns["after_cost_return"]
        )
        .abs()
        .max()
        <= TOLERANCE
    )

    average_after_cost_below_before_cost = (
        backtest_returns["after_cost_return"].mean()
        < backtest_returns["before_cost_return"].mean()
    )

    low_liquidity_weight_valid = (
        backtest_returns["low_liquidity_weight"].min() >= 0
    ) and (
        backtest_returns["low_liquidity_weight"].max()
        <= backtest_returns["total_weight"].max() + TOLERANCE
    )

    old_compounded_columns_absent = not any(
        column in backtest_returns.columns
        for column in [
            "cumulative_before_cost_return",
            "cumulative_after_cost_return",
        ]
    )

    mode_summary = backtest_returns.groupby(
    [
        "optimization_mode",
        "execution_mode",
    ]
    ).agg(
        average_after_cost_return=("after_cost_return", "mean"),
        final_after_cost_active_return=(
            "cumulative_after_cost_active_return",
            "last",
            ),
    )

    performance_summary = calculate_performance_summary(backtest_returns)

    performance_summary_has_four_rows = len(performance_summary) == 4

    performance_summary_columns_present = all(
        column in performance_summary.columns
        for column in [
        "optimization_mode",
        "execution_mode",
        "diagnostic_sharpe",
        "max_active_drawdown",
        "average_turnover",
        "maximum_turnover",
        ]
    )

    performance_summary_values_valid = (
        performance_summary["diagnostic_sharpe"].notna().all()
        and performance_summary["max_active_drawdown"].le(0).all()
        and performance_summary["average_turnover"].ge(0).all()
        and performance_summary["maximum_turnover"].le(0.40 + TOLERANCE).all()
    )

    normal_optimizer_final_after_cost = mode_summary.loc[
    ("normal", "normal"),
    "final_after_cost_active_return",
    ]

    herding_aware_final_after_cost = mode_summary.loc[
        ("herding_aware", "normal"),
        "final_after_cost_active_return",
    ]

    price_limit_final_after_cost = mode_summary.loc[
        ("normal", "price_limit_aware"),
        "final_after_cost_active_return",
    ]

    optimization_modes_differ = (
        abs(normal_optimizer_final_after_cost - herding_aware_final_after_cost)
        > TOLERANCE
    )

    execution_modes_differ = (
        abs(normal_optimizer_final_after_cost - price_limit_final_after_cost)
        > TOLERANCE
    )

    print("\nExecution mode summary:")
    print(mode_summary)
    print("\nPerformance summary:")
    print(performance_summary.to_string(index=False))
    print("Backtest rows:", len(backtest_returns))
    print("Date count:", backtest_returns["date"].nunique())
    print("Average before-cost return:", backtest_returns["before_cost_return"].mean())
    print("Average after-cost return:", backtest_returns["after_cost_return"].mean())
    print("Average total cost:", backtest_returns["total_cost"].mean())
    print("Maximum turnover:", backtest_returns["portfolio_turnover"].max())
    print("Maximum total weight:", backtest_returns["total_weight"].max())
    print("Maximum low-liquidity weight:", backtest_returns["low_liquidity_weight"].max())
    print("Optimization modes:", optimization_modes)
    print("\nRequired columns present:", required_columns_present)
    print("Execution modes:", execution_modes)

    print("One row per date per mode:", one_row_per_date_per_mode)
    print("Expected execution modes present:", expected_execution_modes_present)
    print("Expected optimization modes present:", expected_optimization_modes_present)
    print("Dates match optimized weights:", dates_match_weights)
    print("Total weight respected:", total_weight_respected)
    print("Turnover respected:", turnover_respected)
    print("Costs non-negative:", costs_non_negative)
    print("Total cost matches components:", total_cost_matches_components)
    print("After-cost return matches formula:", after_cost_matches_formula)
    print("Average after-cost below before-cost:", average_after_cost_below_before_cost)
    print("Low-liquidity weight valid:", low_liquidity_weight_valid)
    print("Old compounded columns absent:", old_compounded_columns_absent)
    print("Execution modes differ:", execution_modes_differ)
    print("Optimization modes differ:", optimization_modes_differ)
    print("Performance summary has four rows:", performance_summary_has_four_rows)
    print("Performance summary columns present:", performance_summary_columns_present)
    print("Performance summary values valid:", performance_summary_values_valid)


if __name__ == "__main__":
    main()
