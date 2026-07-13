from __future__ import annotations

import pandas as pd

from src.optimizer import (
    ISSUER_GROUP_COLUMN,
    MAX_SECTOR_WEIGHT,
    KEY_COLUMNS,
    MAX_ISSUER_GROUP_WEIGHT,
    MAX_TURNOVER,
    MAX_WEIGHT,
    MIN_WEIGHT,
    MODEL_NAME,
    OPTIMIZED_WEIGHTS_PATH,
    TREE_MODEL_PREDICTIONS_PATH,
    calculate_turnover,
    HERDING_SCORE_COLUMN,
    SECTOR_COLUMN,
)

TOLERANCE: float = 1e-8

REQUIRED_COLUMNS: list[str] = [
    "date",
    "ticker",
    "weight",
    "predicted_return",
    "actual_return",
    "model_name",
    "optimization_mode",
    "portfolio_turnover",
    "issuer_group",
    "sector",
    HERDING_SCORE_COLUMN,
    "high_herding_day",
]


def main() -> None:
    weights = pd.read_parquet(OPTIMIZED_WEIGHTS_PATH)
    predictions = pd.read_parquet(TREE_MODEL_PREDICTIONS_PATH)

    weights["date"] = pd.to_datetime(weights["date"])
    predictions["date"] = pd.to_datetime(predictions["date"])

    model_predictions = predictions[
        predictions["model_name"] == MODEL_NAME
    ]

    required_columns_present = all(
        column in weights.columns
        for column in REQUIRED_COLUMNS
    )

    duplicate_keys_absent = weights.duplicated(
    [
        "optimization_mode",
        *KEY_COLUMNS,
    ]
    ).sum() == 0

    weights_non_negative = weights["weight"].min() >= 0

    tiny_weights_removed = weights["weight"].min() >= MIN_WEIGHT

    max_weight_respected = weights["weight"].max() <= MAX_WEIGHT + TOLERANCE

    daily_weight_sum = weights.groupby(
    [
        "optimization_mode",
        "date",
    ]
    )["weight"].sum()

    daily_issuer_group_weight = weights.groupby(
    [
        "optimization_mode",
        "date",
        ISSUER_GROUP_COLUMN,
    ]
    )["weight"].sum()
    daily_sector_weight = weights.groupby(
    [
        "optimization_mode",
        "date",
        SECTOR_COLUMN,
    ]
    )["weight"].sum()
    issuer_group_cap_respected = (
        daily_issuer_group_weight.max()
        <= MAX_ISSUER_GROUP_WEIGHT + TOLERANCE
    )

    issuer_group_complete = weights[ISSUER_GROUP_COLUMN].notna().all()
    sector_complete = weights[SECTOR_COLUMN].notna().all()

    sector_aware_weights = daily_sector_weight.loc[
        daily_sector_weight.index.get_level_values("optimization_mode")
        == "sector_aware"
    ]
    sector_cap_respected = (
        sector_aware_weights.max() <= MAX_SECTOR_WEIGHT + TOLERANCE
    )

    weight_sum_respected = daily_weight_sum.max() <= 1.0 + TOLERANCE

    stored_turnover_respected = (
        weights.groupby(
    [
        "optimization_mode",
        "date",
    ]
    )["portfolio_turnover"].max().max()
        <= MAX_TURNOVER + TOLERANCE
    )

    recalculated_turnovers = []

    for optimization_mode, mode_weights in weights.groupby("optimization_mode"):
        previous_weights = pd.Series(dtype=float)

        for date, date_weights in mode_weights.groupby("date"):
            current_weights = date_weights.set_index("ticker")["weight"]

            turnover = calculate_turnover(
                old_weights=previous_weights,
                new_weights=current_weights,
            )

            recalculated_turnovers.append(
                {
                    "optimization_mode": optimization_mode,
                    "date": date,
                    "recalculated_turnover": turnover,
                    "stored_turnover": date_weights["portfolio_turnover"].max(),
                }
            )

            previous_weights = current_weights

    turnover_check = pd.DataFrame(recalculated_turnovers)

    maximum_turnover_difference = (
        turnover_check["recalculated_turnover"]
        - turnover_check["stored_turnover"]
    ).abs().max()

    recalculated_turnover_respected = (
        turnover_check["recalculated_turnover"].max()
        <= MAX_TURNOVER + TOLERANCE
    )

    stored_turnover_matches_recalculated = (
    maximum_turnover_difference <= TOLERANCE
    )

    dates_match_model_predictions = (
        weights["date"].nunique()
        == model_predictions["date"].nunique()
    )

    only_expected_model = weights["model_name"].eq(MODEL_NAME).all()

    optimization_modes = set(weights["optimization_mode"].unique())

    expected_optimization_modes_present = optimization_modes == {
        "normal",
        "herding_aware",
        "sector_aware",
    }

    daily_mode_summary = (
        weights.groupby(
            [
                "optimization_mode",
                "high_herding_day",
                "date",
            ]
        )
        .agg(
            total_weight=("weight", "sum"),
            max_stock_weight=("weight", "max"),
            selected_count=("ticker", "count"),
        )
        .reset_index()
    )

    herding_summary = daily_mode_summary.groupby(
        [
            "optimization_mode",
            "high_herding_day",
        ]
    ).agg(
        dates=("date", "nunique"),
        average_total_weight=("total_weight", "mean"),
        average_max_stock_weight=("max_stock_weight", "mean"),
        average_selected_count=("selected_count", "mean"),
    )

    normal_high_herding = daily_mode_summary[
        (daily_mode_summary["optimization_mode"] == "normal")
        & daily_mode_summary["high_herding_day"]
    ]

    herding_aware_high_herding = daily_mode_summary[
        (daily_mode_summary["optimization_mode"] == "herding_aware")
        & daily_mode_summary["high_herding_day"]
    ]

    herding_aware_reduces_exposure = (
        herding_aware_high_herding["total_weight"].mean()
        < normal_high_herding["total_weight"].mean()
    )

    herding_aware_reduces_max_weight = (
        herding_aware_high_herding["max_stock_weight"].mean()
        < normal_high_herding["max_stock_weight"].mean()
    )

    herding_aware_reduces_selected_count = (
        herding_aware_high_herding["selected_count"].mean()
        < normal_high_herding["selected_count"].mean()
    )

    print("Optimized weight rows:", len(weights))
    print("Date count:", weights["date"].nunique())
    print("Ticker count:", weights["ticker"].nunique())
    print("Minimum weight:", weights["weight"].min())
    print("Maximum weight:", weights["weight"].max())
    print("Maximum daily weight sum:", daily_weight_sum.max())
    print("Maximum issuer group weight:", daily_issuer_group_weight.max())
    print("Maximum sector weight:", daily_sector_weight.max())
    print("Maximum stored turnover:", weights.groupby(
    [
        "optimization_mode",
        "date",
    ]
    )["portfolio_turnover"].max().max())

    print("Maximum recalculated turnover:", turnover_check["recalculated_turnover"].max())
    print("Maximum turnover difference:", maximum_turnover_difference)
    print("\nHerding control summary:")
    print(herding_summary.round(6))
    print("\nRequired columns present:", required_columns_present)
    print("Duplicate keys absent:", duplicate_keys_absent)
    print("Weights non-negative:", weights_non_negative)
    print("Tiny weights removed:", tiny_weights_removed)
    print("Max weight respected:", max_weight_respected)
    print("Weight sum respected:", weight_sum_respected)
    print("Stored turnover respected:", stored_turnover_respected)
    print("Recalculated turnover respected:", recalculated_turnover_respected)
    print("Stored turnover matches recalculated:", stored_turnover_matches_recalculated)
    print("Dates match model predictions:", dates_match_model_predictions)
    print("Only expected model:", only_expected_model)
    print("Issuer group complete:", issuer_group_complete)
    print("Sector complete:", sector_complete)
    print("Issuer group cap respected:", issuer_group_cap_respected)
    print("Sector-aware cap respected:", sector_cap_respected)
    print("Expected optimization modes present:", expected_optimization_modes_present)
    print("Herding-aware reduces exposure:", herding_aware_reduces_exposure)
    print("Herding-aware reduces max weight:", herding_aware_reduces_max_weight)
    print("Herding-aware reduces selected count:", herding_aware_reduces_selected_count)

if __name__ == "__main__":
    main()
