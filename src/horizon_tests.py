from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling_utils import get_model_feature_columns
from src.walk_forward_split import (
    FEATURES_PATH,
    KEY_COLUMNS,
    LABELS_PATH,
    TARGET_COLUMN,
)
from src.tree_models import (
    create_tree_models,
    predict_tree_models_for_windows,
)
from src.walk_forward_split import (
    build_walk_forward_date_windows,
    get_sorted_dates,
)
from src.backtester import (
    apply_price_limit_execution_rules,
    attach_liquidity_features,
    calculate_backtest_returns,
    calculate_performance_summary,
    load_liquidity_features,
)
from src.optimizer import (
    ISSUER_GROUP_COLUMN,
    build_optimized_weights,
    load_daily_herding_state,
    load_universe_metadata,
)

HORIZON_RESULTS_PATH: str = "reports/tables/horizon_results.csv"
HORIZON_PREDICTIONS_PATH: str = "data/processed/horizon_tree_predictions.parquet"
MODEL_NAME: str = "gradient_boosting"

HORIZON_LABELS: dict[int, str] = {
    1: "forward_relative_return_1d",
    5: "forward_relative_return_5d",
    10: "forward_relative_return_10d",
}


def build_horizon_modeling_dataset(
    horizon: int,
) -> pd.DataFrame:
    if horizon not in HORIZON_LABELS:
        raise ValueError(f"Unsupported horizon: {horizon}")

    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    horizon_target_column = HORIZON_LABELS[horizon]

    selected_labels = labels[
        KEY_COLUMNS + [horizon_target_column]
    ].copy()

    selected_labels = selected_labels.rename(
        columns={
            horizon_target_column: TARGET_COLUMN,
        }
    )

    modeling_dataset = features.merge(
        selected_labels,
        on=KEY_COLUMNS,
        how="left",
    )

    return modeling_dataset

def build_horizon_predictions() -> pd.DataFrame:
    prediction_frames = []

    for horizon in HORIZON_LABELS:
        print("Running horizon:", horizon)

        modeling_dataset = build_horizon_modeling_dataset(horizon)
        historical_rows = modeling_dataset[
            modeling_dataset[TARGET_COLUMN].notna()
        ].copy()

        feature_columns = get_model_feature_columns(historical_rows)
        dates = get_sorted_dates(historical_rows)

        windows = build_walk_forward_date_windows(
            dates=dates,
            train_size=3,
            validation_size=1,
            test_size=1,
            purge_size=0,
            step_size=1,
        )

        models = {
            MODEL_NAME: create_tree_models()[MODEL_NAME],
        }

        predictions, feature_importance = predict_tree_models_for_windows(
            models=models,
            historical_rows=historical_rows,
            windows=windows,
            feature_columns=feature_columns,
        )

        predictions["forecast_horizon_days"] = horizon
        predictions["horizon_label"] = HORIZON_LABELS[horizon]
        predictions["feature_count"] = len(feature_columns)

        prediction_frames.append(predictions)

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )

def summarize_horizon_predictions(
    predictions: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    summary_rows = []

    for horizon, horizon_predictions in predictions.groupby(
        "forecast_horizon_days"
    ):
        daily_rows = []

        for date, date_predictions in horizon_predictions.groupby("date"):
            if date_predictions["actual_return"].nunique() <= 1:
                continue

            rank_ic = date_predictions["predicted_return"].corr(
                date_predictions["actual_return"],
                method="spearman",
            )

            selected = date_predictions.sort_values(
                [
                    "predicted_return",
                    "ticker",
                ],
                ascending=[
                    False,
                    True,
                ],
            ).head(top_n)

            daily_rows.append(
                {
                    "date": date,
                    "rank_ic": rank_ic,
                    "top_n_hit_rate": (
                        selected["actual_return"] > 0
                    ).mean(),
                    "top_n_average_actual_return": (
                        selected["actual_return"].mean()
                    ),
                    "selected_count": len(selected),
                }
            )

        daily_summary = pd.DataFrame(daily_rows)

        summary_rows.append(
            {
                "forecast_horizon_days": horizon,
                "horizon_label": horizon_predictions[
                    "horizon_label"
                ].iloc[0],
                "evaluated_dates": daily_summary["date"].nunique(),
                "feature_count": horizon_predictions[
                    "feature_count"
                ].iloc[0],
                "average_rank_ic": daily_summary["rank_ic"].mean(),
                "average_top_5_hit_rate": daily_summary[
                    "top_n_hit_rate"
                ].mean(),
                "average_top_5_actual_return": daily_summary[
                    "top_n_average_actual_return"
                ].mean(),
                "average_selected_count": daily_summary[
                    "selected_count"
                ].mean(),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
        [
            "forecast_horizon_days",
        ]
    ).reset_index(drop=True)

def build_horizon_backtest_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    liquidity = load_liquidity_features()
    daily_herding = load_daily_herding_state()
    universe = load_universe_metadata()

    summary_rows = []

    for horizon, horizon_predictions in predictions.groupby(
        "forecast_horizon_days"
    ):
        print("Backtesting horizon:", horizon)

        horizon_predictions = horizon_predictions.merge(
            daily_herding,
            on="date",
            how="left",
        )

        horizon_predictions = horizon_predictions.merge(
            universe,
            on="ticker",
            how="left",
        )

        missing_herding_rows = horizon_predictions[
            "high_herding_day"
        ].isna().sum()

        if missing_herding_rows > 0:
            raise ValueError(
                f"Missing herding state for {missing_herding_rows} rows"
            )

        missing_issuer_group_rows = horizon_predictions[
            ISSUER_GROUP_COLUMN
        ].isna().sum()

        if missing_issuer_group_rows > 0:
            raise ValueError(
                f"Missing issuer group for {missing_issuer_group_rows} rows"
            )

        weights = build_optimized_weights(
            predictions=horizon_predictions,
            model_name=MODEL_NAME,
            optimization_mode="normal",
        )

        weights_with_liquidity = attach_liquidity_features(
            weights=weights,
            liquidity=liquidity,
        )

        execution_reference = horizon_predictions[
            [
                "date",
                "ticker",
                "predicted_return",
                "actual_return",
            ]
        ].merge(
            liquidity,
            on=[
                "date",
                "ticker",
            ],
            how="left",
        )

        execution_weights = apply_price_limit_execution_rules(
            weights=weights_with_liquidity,
            execution_reference=execution_reference,
        )

        backtest_returns = calculate_backtest_returns(execution_weights)
        backtest_returns["execution_mode"] = "price_limit_aware"

        performance_summary = calculate_performance_summary(backtest_returns)

        row = performance_summary.iloc[0].to_dict()
        row["forecast_horizon_days"] = horizon
        row["horizon_label"] = horizon_predictions["horizon_label"].iloc[0]

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

def main() -> None:
    for horizon in HORIZON_LABELS:
        modeling_dataset = build_horizon_modeling_dataset(horizon)
        historical_rows = modeling_dataset[
            modeling_dataset[TARGET_COLUMN].notna()
        ].copy()

        feature_columns = get_model_feature_columns(historical_rows)

        print("Forecast horizon:", horizon)
        print("Original horizon label:", HORIZON_LABELS[horizon])
        print("Modeling rows:", len(modeling_dataset))
        print("Historical rows:", len(historical_rows))
        print("Feature count:", len(feature_columns))
        print("Target column:", TARGET_COLUMN)
        print()
    output_path = Path(HORIZON_PREDICTIONS_PATH)

    if output_path.exists():
        horizon_predictions = pd.read_parquet(output_path)
        print("Loaded existing horizon predictions:", output_path)
    else:
        horizon_predictions = build_horizon_predictions()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
     )

    horizon_predictions.to_parquet(
        output_path,
        index=False,
    )

    print("Horizon predictions path:", output_path)

    print("Horizon prediction rows:", len(horizon_predictions))
    print(
        "Forecast horizons:",
        sorted(horizon_predictions["forecast_horizon_days"].unique().tolist()),
    )

    prediction_summary = summarize_horizon_predictions(horizon_predictions)
    backtest_summary = build_horizon_backtest_summary(horizon_predictions)

    horizon_summary = prediction_summary.merge(
    backtest_summary[
        [
            "forecast_horizon_days",
            "average_after_cost_return",
            "return_volatility",
            "diagnostic_sharpe",
            "max_active_drawdown",
            "average_turnover",
            "maximum_turnover",
            "final_cumulative_after_cost_active_return",
        ]
    ],
    on="forecast_horizon_days",
    how="left",
    )
    results_path = Path(HORIZON_RESULTS_PATH)
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizon_summary.to_csv(
        results_path,
        index=False,
    )

    print("\nHorizon prediction summary:")
    print(horizon_summary.round(6).to_string(index=False))
    print("Horizon results path:", results_path)

if __name__ == "__main__":
    main()