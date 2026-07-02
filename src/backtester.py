from __future__ import annotations

from pathlib import Path

import pandas as pd


OPTIMIZED_WEIGHTS_PATH: str = "data/processed/optimized_weights.parquet"
FEATURES_PATH: str = "data/processed/features_combined.parquet"
BACKTEST_RETURNS_PATH: str = "data/processed/backtest_returns.parquet"

TREE_MODEL_PREDICTIONS_PATH: str = "data/processed/tree_model_predictions.parquet"
MODEL_NAME: str = "gradient_boosting"

COMMISSION_RATE: float = 0.001
SLIPPAGE_RATE: float = 0.001
LIQUIDITY_PENALTY_RATE: float = 0.001
LOW_LIQUIDITY_QUANTILE: float = 0.20
MAX_EXECUTION_TURNOVER: float = 0.40
MIN_EXECUTION_WEIGHT: float = 1e-8
EXECUTION_TURNOVER_BUFFER: float = 1e-6

KEY_COLUMNS: list[str] = [
    "date",
    "ticker",
]


def load_optimized_weights(
    path: str = OPTIMIZED_WEIGHTS_PATH,
) -> pd.DataFrame:
    weights = pd.read_parquet(path)

    weights["date"] = pd.to_datetime(weights["date"])

    return weights


def load_liquidity_features(
    path: str = FEATURES_PATH,
) -> pd.DataFrame:
    features = pd.read_parquet(path)

    features["date"] = pd.to_datetime(features["date"])

    required_columns = [
        "date",
        "ticker",
        "average_daily_value_20d",
        "close_at_ceiling_today",
        "close_at_floor_today",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing liquidity feature columns: {missing_columns}")

    liquidity = features[required_columns].copy()

    daily_low_liquidity_cutoff = (
        liquidity.groupby("date")["average_daily_value_20d"]
        .quantile(LOW_LIQUIDITY_QUANTILE)
        .rename("low_liquidity_cutoff")
        .reset_index()
    )

    liquidity = liquidity.merge(
        daily_low_liquidity_cutoff,
        on="date",
        how="left",
    )

    liquidity["is_low_liquidity"] = (
        liquidity["average_daily_value_20d"]
        <= liquidity["low_liquidity_cutoff"]
    )

    return liquidity


def attach_liquidity_features(
    weights: pd.DataFrame,
    liquidity: pd.DataFrame,
) -> pd.DataFrame:
    return weights.merge(
        liquidity,
        on=KEY_COLUMNS,
        how="left",
    )

def load_execution_reference(
    prediction_path: str = TREE_MODEL_PREDICTIONS_PATH,
) -> pd.DataFrame:
    predictions = pd.read_parquet(prediction_path)
    predictions["date"] = pd.to_datetime(predictions["date"])

    predictions = predictions[predictions["model_name"] == MODEL_NAME].copy()

    execution_reference = predictions[
        [
            "date",
            "ticker",
            "predicted_return",
            "actual_return",
        ]
    ].copy()

    liquidity = load_liquidity_features()

    execution_reference = execution_reference.merge(
        liquidity,
        on=KEY_COLUMNS,
        how="left",
    )

    return execution_reference

def apply_price_limit_execution_rules(
    weights: pd.DataFrame,
    execution_reference: pd.DataFrame,
) -> pd.DataFrame:
    adjusted_frames = []

    for optimization_mode, mode_weights in weights.groupby("optimization_mode"):
        previous_weights = pd.Series(dtype=float)

        for date, date_weights in mode_weights.groupby("date"):
            working = date_weights.copy()
            current_weights = working.set_index("ticker")["weight"]

            all_tickers = previous_weights.index.union(current_weights.index)

            previous_aligned = previous_weights.reindex(
                all_tickers,
                fill_value=0.0,
            )

            current_aligned = current_weights.reindex(
                all_tickers,
                fill_value=0.0,
            )

            execution_rows = pd.DataFrame(
                {
                    "ticker": all_tickers,
                    "previous_weight": previous_aligned.values,
                    "target_weight": current_aligned.values,
                }
            )

            execution_rows["date"] = date
            execution_rows["optimization_mode"] = optimization_mode

            date_execution_reference = execution_reference[
                execution_reference["date"] == date
            ]

            execution_rows = execution_rows.merge(
                date_execution_reference,
                on=[
                    "date",
                    "ticker",
                ],
                how="left",
            )

            execution_rows["close_at_ceiling_today"] = execution_rows[
                "close_at_ceiling_today"
            ].eq(True)

            execution_rows["close_at_floor_today"] = execution_rows[
                "close_at_floor_today"
            ].eq(True)

            execution_rows["buy_blocked_by_ceiling"] = (
                execution_rows["target_weight"]
                > execution_rows["previous_weight"]
            ) & execution_rows["close_at_ceiling_today"]

            execution_rows["sell_blocked_by_floor"] = (
                execution_rows["target_weight"]
                < execution_rows["previous_weight"]
            ) & execution_rows["close_at_floor_today"]

            execution_rows["execution_adjusted_weight"] = execution_rows[
                "target_weight"
            ]

            blocked_trade = (
                execution_rows["buy_blocked_by_ceiling"]
                | execution_rows["sell_blocked_by_floor"]
            )

            execution_rows.loc[
                blocked_trade,
                "execution_adjusted_weight",
            ] = execution_rows.loc[
                blocked_trade,
                "previous_weight",
            ]

            total_weight = execution_rows["execution_adjusted_weight"].sum()

            if total_weight > 1.0:
                buy_increase = (
                    execution_rows["execution_adjusted_weight"]
                    - execution_rows["previous_weight"]
                ).clip(lower=0.0)

                total_buy_increase = buy_increase.sum()

                excess_weight = total_weight - 1.0

                if total_buy_increase > 0:
                    buy_reduction_scale = max(
                        1.0 - excess_weight / total_buy_increase,
                        0.0,
                    )

                    execution_rows["execution_adjusted_weight"] = (
                        execution_rows["previous_weight"]
                        + (
                            execution_rows["execution_adjusted_weight"]
                            - execution_rows["previous_weight"]
                        ).clip(lower=0.0) * buy_reduction_scale
                        + (
                            execution_rows["execution_adjusted_weight"]
                            - execution_rows["previous_weight"]
                        ).clip(upper=0.0)
                    )

            actual_turnover = 0.5 * (
                execution_rows["execution_adjusted_weight"]
                - execution_rows["previous_weight"]
            ).abs().sum()

            if actual_turnover > MAX_EXECUTION_TURNOVER:
                scale = max(
                    (MAX_EXECUTION_TURNOVER - EXECUTION_TURNOVER_BUFFER)
                    / actual_turnover,
                    0.0,
                )

                execution_rows["execution_adjusted_weight"] = (
                    execution_rows["previous_weight"]
                    + scale
                    * (
                        execution_rows["execution_adjusted_weight"]
                        - execution_rows["previous_weight"]
                    )
                )

                actual_turnover = 0.5 * (
                    execution_rows["execution_adjusted_weight"]
                    - execution_rows["previous_weight"]
                ).abs().sum()

            execution_rows["weight"] = execution_rows["execution_adjusted_weight"]
            execution_rows["portfolio_turnover"] = actual_turnover

            execution_rows = execution_rows[
                execution_rows["execution_adjusted_weight"] > MIN_EXECUTION_WEIGHT
            ]

            adjusted_frames.append(execution_rows)

            previous_weights = execution_rows.set_index("ticker")[
                "execution_adjusted_weight"
            ]

    return pd.concat(
        adjusted_frames,
        ignore_index=True,
    ).sort_values(
        [
            "optimization_mode",
            "date",
            "ticker",
        ]
    ).reset_index(drop=True)

def calculate_backtest_returns(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    working = weights.copy()

    working["is_low_liquidity"] = working["is_low_liquidity"].eq(True)

    working["weighted_actual_return"] = (
        working["weight"] * working["actual_return"]
    )

    working["low_liquidity_weight_component"] = working["weight"].where(
        working["is_low_liquidity"],
        0.0,
    )

    daily_returns = working.groupby(
        [
            "optimization_mode",
            "date",
        ]
    ).agg(
        before_cost_return=("weighted_actual_return", "sum"),
        portfolio_turnover=("portfolio_turnover", "max"),
        selected_count=("ticker", "count"),
        total_weight=("weight", "sum"),
        low_liquidity_weight=("low_liquidity_weight_component", "sum"),
    )

    daily_returns["commission_cost"] = (
        daily_returns["portfolio_turnover"] * COMMISSION_RATE
    )

    daily_returns["slippage_cost"] = (
        daily_returns["portfolio_turnover"] * SLIPPAGE_RATE
    )

    daily_returns["liquidity_penalty"] = (
        daily_returns["portfolio_turnover"]
        * daily_returns["low_liquidity_weight"]
        * LIQUIDITY_PENALTY_RATE
    )

    daily_returns["total_cost"] = (
        daily_returns["commission_cost"]
        + daily_returns["slippage_cost"]
        + daily_returns["liquidity_penalty"]
    )

    daily_returns["after_cost_return"] = (
        daily_returns["before_cost_return"]
        - daily_returns["total_cost"]
    )

    daily_returns["cumulative_before_cost_active_return"] = (
        daily_returns.groupby(level="optimization_mode")[
            "before_cost_return"
        ].cumsum()
    )

    daily_returns["cumulative_after_cost_active_return"] = (
        daily_returns.groupby(level="optimization_mode")[
            "after_cost_return"
        ].cumsum()
    )

    return daily_returns.reset_index()

def calculate_performance_summary(
    backtest_returns: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    for keys, group in backtest_returns.groupby(
        [
            "optimization_mode",
            "execution_mode",
        ]
    ):
        optimization_mode, execution_mode = keys

        ordered = group.sort_values("date").copy()

        cumulative_return = ordered["cumulative_after_cost_active_return"]
        running_peak = cumulative_return.cummax()
        drawdown = cumulative_return - running_peak

        average_after_cost_return = ordered["after_cost_return"].mean()
        return_volatility = ordered["after_cost_return"].std()

        if return_volatility > 0:
            diagnostic_sharpe = average_after_cost_return / return_volatility
        else:
            diagnostic_sharpe = float("nan")

        summary_rows.append(
            {
                "optimization_mode": optimization_mode,
                "execution_mode": execution_mode,
                "average_after_cost_return": average_after_cost_return,
                "return_volatility": return_volatility,
                "diagnostic_sharpe": diagnostic_sharpe,
                "max_active_drawdown": drawdown.min(),
                "average_turnover": ordered["portfolio_turnover"].mean(),
                "maximum_turnover": ordered["portfolio_turnover"].max(),
                "final_cumulative_after_cost_active_return": (
                    cumulative_return.iloc[-1]
                ),
            }
        )

    return pd.DataFrame(summary_rows)

def save_backtest_returns(
    backtest_returns: pd.DataFrame,
    output_path: str = BACKTEST_RETURNS_PATH,
) -> str:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backtest_returns.to_parquet(
        path,
        index=False,
    )

    return str(path)


def main() -> None:
    weights = load_optimized_weights()
    liquidity = load_liquidity_features()
    weights_with_liquidity = attach_liquidity_features(weights, liquidity)
    execution_reference = load_execution_reference()

    print("Execution reference rows:", len(execution_reference))
    print("Execution reference dates:", execution_reference["date"].nunique())
    print("Missing execution reference actual returns:", execution_reference["actual_return"].isna().sum())
    print("Missing execution reference ceiling flags:", execution_reference["close_at_ceiling_today"].isna().sum())
    print("Missing execution reference floor flags:", execution_reference["close_at_floor_today"].isna().sum())
    print("Missing close-at-ceiling flags:", weights_with_liquidity["close_at_ceiling_today"].isna().sum())
    print("Missing close-at-floor flags:", weights_with_liquidity["close_at_floor_today"].isna().sum())
    print("Held rows closing at ceiling:", weights_with_liquidity["close_at_ceiling_today"].eq(True).sum())
    print("Held rows closing at floor:", weights_with_liquidity["close_at_floor_today"].eq(True).sum())
    normal_backtest_returns = calculate_backtest_returns(weights_with_liquidity)
    normal_backtest_returns["execution_mode"] = "normal"

    execution_weights = apply_price_limit_execution_rules(
        weights=weights_with_liquidity,
        execution_reference=execution_reference,
    )

    price_limit_backtest_returns = calculate_backtest_returns(execution_weights)
    price_limit_backtest_returns["execution_mode"] = "price_limit_aware"

    backtest_returns = pd.concat(
        [
            normal_backtest_returns,
            price_limit_backtest_returns,
        ],
        ignore_index=True,
    )

    output_path = save_backtest_returns(backtest_returns)

    print("Backtest returns path:", output_path)
    comparison_summary = backtest_returns.groupby(
    [
        "optimization_mode",
        "execution_mode",
    ]
).agg(
    date_count=("date", "nunique"),
    average_before_cost_return=("before_cost_return", "mean"),
    average_after_cost_return=("after_cost_return", "mean"),
    average_total_cost=("total_cost", "mean"),
    maximum_turnover=("portfolio_turnover", "max"),
    maximum_total_weight=("total_weight", "max"),
    final_cumulative_after_cost_active_return=(
        "cumulative_after_cost_active_return",
        "last",
    ),
)
    print("\nOptimization and execution mode comparison:")
    print(comparison_summary)
    performance_summary = calculate_performance_summary(backtest_returns)

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
    print("Final cumulative before-cost active return:", backtest_returns["cumulative_before_cost_active_return"].iloc[-1])
    print("Final cumulative after-cost active return:", backtest_returns["cumulative_after_cost_active_return"].iloc[-1])
    print("Execution-adjusted rows:", len(execution_weights))
    print("Blocked buy rows:", execution_weights["buy_blocked_by_ceiling"].sum())
    print("Blocked sell rows:", execution_weights["sell_blocked_by_floor"].sum())

    print("\nSample:")
    print(backtest_returns.head(10))


if __name__ == "__main__":
    main()
