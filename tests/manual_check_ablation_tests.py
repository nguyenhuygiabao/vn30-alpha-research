from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ablation_tests import (
    ABLATION_PREDICTIONS_PATH,
    ABLATION_RESULTS_PATH,
    ABLATION_GROUPS,
)


TOLERANCE: float = 1e-8


REQUIRED_RESULT_COLUMNS: list[str] = [
    "ablation_name",
    "evaluated_dates",
    "feature_count",
    "average_rank_ic",
    "average_top_5_hit_rate",
    "average_top_5_actual_return",
    "average_selected_count",
    "average_after_cost_return",
    "return_volatility",
    "diagnostic_sharpe",
    "max_active_drawdown",
    "average_turnover",
    "maximum_turnover",
    "final_cumulative_after_cost_active_return",
]


def main() -> None:
    predictions_path = Path(ABLATION_PREDICTIONS_PATH)
    results_path = Path(ABLATION_RESULTS_PATH)

    predictions_file_exists = predictions_path.exists()
    results_file_exists = results_path.exists()

    predictions = pd.read_parquet(predictions_path)
    results = pd.read_csv(results_path)

    expected_ablation_names = sorted(ABLATION_GROUPS.keys())
    prediction_ablation_names = sorted(
        predictions["ablation_name"].unique().tolist()
    )
    result_ablation_names = sorted(
        results["ablation_name"].unique().tolist()
    )

    required_result_columns_present = all(
        column in results.columns
        for column in REQUIRED_RESULT_COLUMNS
    )

    expected_prediction_ablations_present = (
        prediction_ablation_names == expected_ablation_names
    )

    expected_result_ablations_present = (
        result_ablation_names == expected_ablation_names
    )

    one_result_row_per_ablation = (
        results["ablation_name"].is_unique
        and len(results) == len(expected_ablation_names)
    )

    prediction_rows_valid = len(predictions) > 0

    evaluated_dates_valid = results["evaluated_dates"].min() > 0

    feature_counts_valid = results["feature_count"].min() > 0

    hit_rates_valid = (
        results["average_top_5_hit_rate"].between(0, 1).all()
    )

    sharpe_values_valid = results["diagnostic_sharpe"].notna().all()

    drawdowns_valid = results["max_active_drawdown"].le(0).all()

    turnover_valid = (
        results["average_turnover"].ge(0).all()
        and results["maximum_turnover"].le(0.40 + TOLERANCE).all()
    )

    print("Ablation prediction rows:", len(predictions))
    print("Ablation result rows:", len(results))
    print("Prediction ablations:", prediction_ablation_names)
    print("Result ablations:", result_ablation_names)
    print("\nAblation results:")
    print(results.round(6).to_string(index=False))

    print("\nPredictions file exists:", predictions_file_exists)
    print("Results file exists:", results_file_exists)
    print("Required result columns present:", required_result_columns_present)
    print(
        "Expected prediction ablations present:",
        expected_prediction_ablations_present,
    )
    print(
        "Expected result ablations present:",
        expected_result_ablations_present,
    )
    print("One result row per ablation:", one_result_row_per_ablation)
    print("Prediction rows valid:", prediction_rows_valid)
    print("Evaluated dates valid:", evaluated_dates_valid)
    print("Feature counts valid:", feature_counts_valid)
    print("Hit rates valid:", hit_rates_valid)
    print("Sharpe values valid:", sharpe_values_valid)
    print("Drawdowns valid:", drawdowns_valid)
    print("Turnover valid:", turnover_valid)


if __name__ == "__main__":
    main()