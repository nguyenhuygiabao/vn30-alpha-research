from __future__ import annotations

from pathlib import Path

import pandas as pd


TREE_MODEL_PREDICTIONS_PATH: str = "data/processed/tree_model_predictions.parquet"
OPTIMIZED_WEIGHTS_PATH: str = "data/processed/optimized_weights.parquet"
VN30_UNIVERSE_PATH: str = "config/vn30_universe.csv"
FEATURES_PATH: str = "data/processed/features_combined.parquet"

MODEL_NAME: str = "gradient_boosting"

TOP_N: int = 5
MAX_WEIGHT: float = 0.20
MAX_TURNOVER: float = 0.40
MIN_WEIGHT: float = 1e-8
TURNOVER_BUFFER: float = 1e-6
MAX_ISSUER_GROUP_WEIGHT: float = 0.40
ISSUER_GROUP_COLUMN: str = "issuer_group"
HERDING_SCORE_COLUMN: str = "herding_corr_score"
HIGH_HERDING_THRESHOLD: float = 0.80
HERDING_AWARE_TOP_N: int = 5
HERDING_AWARE_MAX_WEIGHT: float = 0.15
HERDING_AWARE_TARGET_EXPOSURE: float = 0.75
HERDING_AWARE_MIN_PREDICTED_RETURN: float = 0.0

KEY_COLUMNS: list[str] = [
    "date",
    "ticker",
]


def load_model_predictions(
    path: str = TREE_MODEL_PREDICTIONS_PATH,
) -> pd.DataFrame:
    predictions = pd.read_parquet(path)

    predictions["date"] = pd.to_datetime(predictions["date"])

    universe = load_universe_metadata()

    predictions = predictions.merge(
        universe,
        on="ticker",
        how="left",
    )

    missing_issuer_group = predictions[ISSUER_GROUP_COLUMN].isna().sum()

    if missing_issuer_group > 0:
        raise ValueError(
            f"Missing issuer group for {missing_issuer_group} prediction rows"
        )

    return predictions

def load_universe_metadata(
    path: str = VN30_UNIVERSE_PATH,
) -> pd.DataFrame:
    universe = pd.read_csv(path)

    required_columns = [
        "ticker",
        ISSUER_GROUP_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in universe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing universe columns: {missing_columns}")

    return universe[required_columns].copy()

def load_daily_herding_state(
    path: str = FEATURES_PATH,
) -> pd.DataFrame:
    features = pd.read_parquet(path)

    features["date"] = pd.to_datetime(features["date"])

    required_columns = [
        "date",
        HERDING_SCORE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing herding columns: {missing_columns}")

    daily_herding = (
        features.groupby("date")[HERDING_SCORE_COLUMN]
        .first()
        .reset_index()
    )

    daily_herding["high_herding_day"] = (
        daily_herding[HERDING_SCORE_COLUMN] >= HIGH_HERDING_THRESHOLD
    )

    return daily_herding

def build_target_weights_for_date(
    date_predictions: pd.DataFrame,
    top_n: int = TOP_N,
    max_weight: float = MAX_WEIGHT,
    target_exposure: float = 1.0,
    min_predicted_return: float | None = None,
) -> pd.DataFrame:

    eligible_predictions = date_predictions.copy()

    if min_predicted_return is not None:
        eligible_predictions = eligible_predictions[
            eligible_predictions["predicted_return"] > min_predicted_return
        ]

    if eligible_predictions.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                ISSUER_GROUP_COLUMN,
                "predicted_return",
                "target_weight",
            ]
        )

    ranked = eligible_predictions.sort_values(
        [
            "predicted_return",
            "ticker",
        ],
        ascending=[
            False,
            True,
        ],
    ).head(top_n)

    selected_count = len(ranked)

    weight = min(
        target_exposure / selected_count,
        max_weight,
    )

    target_weights = ranked[
        [
            "date",
            "ticker",
            ISSUER_GROUP_COLUMN,
            "predicted_return",
        ]
    ].copy()

    target_weights["target_weight"] = weight
    target_weights = apply_issuer_group_cap(target_weights)

    return target_weights

def apply_issuer_group_cap(
    target_weights: pd.DataFrame,
    max_issuer_group_weight: float = MAX_ISSUER_GROUP_WEIGHT,
) -> pd.DataFrame:
    capped = target_weights.copy()

    group_weight = capped.groupby(ISSUER_GROUP_COLUMN)["target_weight"].transform(
        "sum"
    )

    scale = (max_issuer_group_weight / group_weight).clip(upper=1.0)

    capped["target_weight"] = capped["target_weight"] * scale

    return capped

def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    all_tickers = old_weights.index.union(new_weights.index)

    old_aligned = old_weights.reindex(
        all_tickers,
        fill_value=0.0,
    )

    new_aligned = new_weights.reindex(
        all_tickers,
        fill_value=0.0,
    )

    return 0.5 * (new_aligned - old_aligned).abs().sum()


def apply_turnover_cap(
    previous_weights: pd.Series,
    target_weights: pd.Series,
    max_turnover: float = MAX_TURNOVER,
) -> pd.Series:
    turnover = calculate_turnover(
        old_weights=previous_weights,
        new_weights=target_weights,
    )

    if turnover <= max_turnover or turnover == 0:
        return target_weights[target_weights > MIN_WEIGHT]

    scale = max(
        (max_turnover - TURNOVER_BUFFER) / turnover,
        0.0,
    )

    all_tickers = previous_weights.index.union(target_weights.index)

    previous_aligned = previous_weights.reindex(
        all_tickers,
        fill_value=0.0,
    )

    target_aligned = target_weights.reindex(
        all_tickers,
        fill_value=0.0,
    )

    adjusted_weights = previous_aligned + scale * (
        target_aligned - previous_aligned
    )

    return adjusted_weights[adjusted_weights > MIN_WEIGHT]


def build_optimized_weights(
    predictions: pd.DataFrame,
    model_name: str = MODEL_NAME,
    top_n: int = TOP_N,
    max_weight: float = MAX_WEIGHT,
    max_turnover: float = MAX_TURNOVER,
    optimization_mode: str = "normal",
) -> pd.DataFrame:
    model_predictions = predictions[
        predictions["model_name"] == model_name
    ].copy()

    if model_predictions.empty:
        raise ValueError(f"No predictions found for model_name={model_name}")

    optimized_frames = []
    previous_weights = pd.Series(dtype=float)
    universe_metadata = load_universe_metadata()

    for date, date_predictions in model_predictions.groupby("date"):
        is_high_herding_day = bool(
            date_predictions["high_herding_day"].iloc[0]
        )

        use_herding_controls = (
            optimization_mode == "herding_aware"
            and is_high_herding_day
        )

        if use_herding_controls:
            date_top_n = HERDING_AWARE_TOP_N
            date_max_weight = HERDING_AWARE_MAX_WEIGHT
            date_target_exposure = HERDING_AWARE_TARGET_EXPOSURE
            date_min_predicted_return = HERDING_AWARE_MIN_PREDICTED_RETURN
        else:
            date_top_n = top_n
            date_max_weight = max_weight
            date_target_exposure = 1.0
            date_min_predicted_return = None

        target_frame = build_target_weights_for_date(
            date_predictions=date_predictions,
            top_n=date_top_n,
            max_weight=date_max_weight,
            target_exposure=date_target_exposure,
            min_predicted_return=date_min_predicted_return,
        )

        target_weights = target_frame.set_index("ticker")["target_weight"]

        optimized_weights = apply_turnover_cap(
            previous_weights=previous_weights,
            target_weights=target_weights,
            max_turnover=max_turnover,
        )

        output = (
            optimized_weights.rename("weight")
            .reset_index()
            .rename(columns={"index": "ticker"})
        )

        output["date"] = date
        output = output.merge(
            universe_metadata,
            on="ticker",
            how="left",
        )

        output = output.merge(
            date_predictions[
                [
                    "date",
                    "ticker",
                    "predicted_return",
                    "actual_return",
                    HERDING_SCORE_COLUMN,
                    "high_herding_day",
                ]
            ],
            on=[
                "date",
                "ticker",
            ],
            how="left",
        )

        output["model_name"] = model_name
        output["optimization_mode"] = optimization_mode
        output["portfolio_turnover"] = calculate_turnover(
            old_weights=previous_weights,
            new_weights=optimized_weights,
        )

        optimized_frames.append(output)

        previous_weights = optimized_weights

    return pd.concat(
        optimized_frames,
        ignore_index=True,
    ).sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(drop=True)


def save_optimized_weights(
    weights: pd.DataFrame,
    output_path: str = OPTIMIZED_WEIGHTS_PATH,
) -> str:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights.to_parquet(
        path,
        index=False,
    )

    return str(path)


def main() -> None:
    predictions = load_model_predictions()
    daily_herding = load_daily_herding_state()
    print("Herding dates:", len(daily_herding))
    print("High-herding dates:", daily_herding["high_herding_day"].sum())
    print("High-herding threshold:", HIGH_HERDING_THRESHOLD)
    predictions = predictions.merge(
    daily_herding,
    on="date",
    how="left",
    )

    missing_herding_rows = predictions["high_herding_day"].isna().sum()

    if missing_herding_rows > 0:
        raise ValueError(
            f"Missing herding state for {missing_herding_rows} prediction rows"
        )

    print("Prediction rows:", len(predictions))
    print(
        "Prediction rows on high-herding dates:",
        predictions["high_herding_day"].sum(),
    )
    normal_weights = build_optimized_weights(
        predictions=predictions,
        optimization_mode="normal",
    )

    herding_aware_weights = build_optimized_weights(
        predictions=predictions,
        optimization_mode="herding_aware",
    )

    weights = pd.concat(
        [
            normal_weights,
            herding_aware_weights,
        ],
        ignore_index=True,
    )

    output_path = save_optimized_weights(weights)

    daily_weight_sum = weights.groupby(
        [
            "optimization_mode",
            "date",
        ]
    )["weight"].sum()

    daily_turnover = weights.groupby(
        [
            "optimization_mode",
            "date",
        ]
    )["portfolio_turnover"].max()

    daily_issuer_group_weight = weights.groupby(
        [
            "optimization_mode",
            "date",
            ISSUER_GROUP_COLUMN,
        ]
    )["weight"].sum()

    print("Optimized weights path:", output_path)
    print("Optimized weight rows:", len(weights))
    print("Date count:", weights["date"].nunique())
    print("Ticker count:", weights["ticker"].nunique())
    print("Minimum weight:", weights["weight"].min())
    print("Maximum weight:", weights["weight"].max())
    print("Maximum daily weight sum:", daily_weight_sum.max())
    print("Maximum daily turnover:", daily_turnover.max())
    print("Maximum issuer group weight:", daily_issuer_group_weight.max())
    duplicate_keys = weights.duplicated(
    [
        "optimization_mode",
        "date",
        "ticker",
    ]
    ).sum()

    print("Duplicate mode-date-ticker keys:", duplicate_keys)
    print("\nSample:")
    print(weights.head(20))


if __name__ == "__main__":
    main()
