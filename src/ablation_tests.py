from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baselines import load_baseline_dataset
from src.modeling_utils import get_model_feature_columns
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


ABLATION_RESULTS_PATH: str = "reports/tables/ablation_results.csv"
ABLATION_PREDICTIONS_PATH: str = "data/processed/ablation_tree_predictions.parquet"
MODEL_NAME: str = "gradient_boosting"

VOLUME_LIQUIDITY_FEATURES: list[str] = [
    "average_daily_volume_20d",
    "average_daily_value_20d",
    "traded_value_rank_20d",
    "volume_z_20",
    "value_traded_z_20",
    "volume_change_5d",
    "amihud_illiquidity_raw",
    "amihud_illiquidity",
    "abnormal_volume_flag",
    "volume_concentration_top5",
    "herding_volume_concentration_score",
]

PRICE_LIMIT_FEATURES: list[str] = [
    "reference_price",
    "estimated_ceiling_price",
    "estimated_floor_price",
    "distance_to_ceiling",
    "distance_to_floor",
    "hit_ceiling_today",
    "hit_floor_today",
    "close_at_ceiling_today",
    "close_at_floor_today",
    "consecutive_ceiling_days",
    "consecutive_floor_days",
    "percent_hitting_ceiling",
    "percent_hitting_floor",
    "price_limit_pressure",
    "herding_price_limit_score",
]

HERDING_FEATURES: list[str] = [
    "vn30_return_dispersion",
    "percent_stocks_up",
    "percent_stocks_down",
    "rolling_avg_pairwise_corr",
    "market_direction_agreement",
    "herding_corr_score",
    "herding_low_dispersion_score",
    "herding_direction_score",
    "herding_volume_concentration_score",
    "herding_index",
]

RISK_FEATURES: list[str] = [
    "rolling_vol_20d",
    "drawdown",
]


ABLATION_GROUPS: dict[str, list[str]] = {
    "all_features": [],
    "without_volume_liquidity": VOLUME_LIQUIDITY_FEATURES,
    "without_price_limit": PRICE_LIMIT_FEATURES,
    "without_herding": HERDING_FEATURES,
    "without_risk": RISK_FEATURES,
}


def build_ablation_feature_sets(
    all_feature_columns: list[str],
) -> dict[str, list[str]]:
    feature_sets = {}

    for ablation_name, removed_features in ABLATION_GROUPS.items():
        removed_feature_set = set(removed_features)

        feature_sets[ablation_name] = [
            feature
            for feature in all_feature_columns
            if feature not in removed_feature_set
        ]

    return feature_sets

def build_ablation_predictions(
    historical_rows: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    dates = get_sorted_dates(historical_rows)

    windows = build_walk_forward_date_windows(
        dates=dates,
        train_size=3,
        validation_size=1,
        test_size=1,
        purge_size=0,
        step_size=1,
    )

    model = create_tree_models()[MODEL_NAME]
    models = {
        MODEL_NAME: model,
    }

    prediction_frames = []

    for ablation_name, feature_columns in feature_sets.items():
        print("Running ablation:", ablation_name)
        print("Feature count:", len(feature_columns))

        predictions, feature_importance = predict_tree_models_for_windows(
            models=models,
            historical_rows=historical_rows,
            windows=windows,
            feature_columns=feature_columns,
        )

        predictions["ablation_name"] = ablation_name
        predictions["feature_count"] = len(feature_columns)

        prediction_frames.append(predictions)

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )

def summarize_ablation_predictions(
    predictions: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    summary_rows = []

    for ablation_name, ablation_predictions in predictions.groupby(
        "ablation_name"
    ):
        daily_rows = []

        for date, date_predictions in ablation_predictions.groupby("date"):
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
                "ablation_name": ablation_name,
                "evaluated_dates": daily_summary["date"].nunique(),
                "feature_count": ablation_predictions[
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
            "average_rank_ic",
            "average_top_5_actual_return",
            "ablation_name",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

def build_ablation_backtest_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    liquidity = load_liquidity_features()
    daily_herding = load_daily_herding_state()
    universe = load_universe_metadata()

    summary_rows = []

    for ablation_name, ablation_predictions in predictions.groupby(
        "ablation_name"
    ):
        print("Backtesting ablation:", ablation_name)

        ablation_predictions = ablation_predictions.merge(
            daily_herding,
            on="date",
            how="left",
        )
        ablation_predictions = ablation_predictions.merge(
            universe,
            on="ticker",
            how="left",
        )
        missing_herding_rows = ablation_predictions[
            "high_herding_day"
        ].isna().sum()

        if missing_herding_rows > 0:
            raise ValueError(
                f"Missing herding state for {missing_herding_rows} rows"
            )

        weights = build_optimized_weights(
            predictions=ablation_predictions,
            model_name=MODEL_NAME,
            optimization_mode="normal",
        )

        weights_with_liquidity = attach_liquidity_features(
            weights=weights,
            liquidity=liquidity,
        )

        execution_reference = ablation_predictions[
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
        row["ablation_name"] = ablation_name

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

def main() -> None:
    modeling_dataset, historical_rows, prediction_rows = load_baseline_dataset()

    all_feature_columns = get_model_feature_columns(historical_rows)
    feature_sets = build_ablation_feature_sets(all_feature_columns)

    print("Full feature count:", len(all_feature_columns))

    for ablation_name, feature_columns in feature_sets.items():
        removed_count = len(all_feature_columns) - len(feature_columns)

        print(
            ablation_name,
            "feature_count:",
            len(feature_columns),
            "removed_count:",
            removed_count,
        )

    output_path = Path(ABLATION_PREDICTIONS_PATH)

    if output_path.exists():
        ablation_predictions = pd.read_parquet(output_path)
        print("Loaded existing ablation predictions:", output_path)
    else:
        ablation_predictions = build_ablation_predictions(
            historical_rows=historical_rows,
            feature_sets=feature_sets,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ablation_predictions.to_parquet(
            output_path,
            index=False,
        )

        print("Ablation predictions path:", output_path)

    print("Ablation prediction rows:", len(ablation_predictions))
    print("Ablation names:", sorted(ablation_predictions["ablation_name"].unique()))

    prediction_summary = summarize_ablation_predictions(ablation_predictions)
    backtest_summary = build_ablation_backtest_summary(ablation_predictions)

    ablation_summary = prediction_summary.merge(
        backtest_summary[
            [
                "ablation_name",
                "average_after_cost_return",
                "return_volatility",
                "diagnostic_sharpe",
                "max_active_drawdown",
                "average_turnover",
                "maximum_turnover",
                "final_cumulative_after_cost_active_return",
            ]
        ],
        on="ablation_name",
        how="left",
    )

    results_path = Path(ABLATION_RESULTS_PATH)
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = Path(ABLATION_RESULTS_PATH)
    ablation_summary.to_csv(
        results_path,
        index=False,
    )

    print("\nAblation prediction summary:")
    print(ablation_summary.round(6).to_string(index=False))
    print("Ablation results path:", results_path)

if __name__ == "__main__":
    main()