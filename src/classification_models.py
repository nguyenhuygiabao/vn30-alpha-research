from __future__ import annotations

from math import ceil
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines import load_baseline_dataset
from src.modeling_utils import (
    get_model_feature_columns,
    prepare_classification_window_data,
)
from src.walk_forward_split import (
    TARGET_COLUMN,
    build_walk_forward_date_windows,
    get_sorted_dates,
)


CLASSIFICATION_PREDICTIONS_PATH: str = (
    "data/processed/classification_model_predictions.parquet"
)

TOP_QUINTILE_LABEL_COLUMN: str = "is_top_quintile_next_5d"
TOP_QUINTILE_FRACTION: float = 0.20


def add_top_quintile_label(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    label_column: str = TOP_QUINTILE_LABEL_COLUMN,
    top_fraction: float = TOP_QUINTILE_FRACTION,
) -> pd.DataFrame:
    working = data.copy()

    working[label_column] = pd.NA

    for date, date_rows in working.groupby("date"):
        known_target_rows = date_rows[
            date_rows[target_column].notna()
        ].copy()

        if known_target_rows.empty:
            continue

        positive_count = max(
            1,
            ceil(len(known_target_rows) * top_fraction),
        )

        top_index = known_target_rows.sort_values(
            [
                target_column,
                "ticker",
            ],
            ascending=[
                False,
                True,
            ],
        ).head(positive_count).index

        known_index = known_target_rows.index

        working.loc[known_index, label_column] = 0
        working.loc[top_index, label_column] = 1

    working[label_column] = working[label_column].astype("Int64")

    return working


def fit_predict_logistic_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> pd.Series:
    clean_y_train = y_train.astype(int)
    clean_x_train = x_train.astype(float)
    clean_x_test = x_test.astype(float)

    if clean_y_train.nunique() < 2:
        constant_probability = float(clean_y_train.iloc[0])

        return pd.Series(
            [constant_probability] * len(x_test),
            index=x_test.index,
        )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )

    pipeline.fit(
        clean_x_train,
        clean_y_train,
    )

    model = pipeline.named_steps["model"]
    positive_class_index = list(model.classes_).index(1)

    predicted_probabilities = pipeline.predict_proba(clean_x_test)[
        :,
        positive_class_index,
    ]

    return pd.Series(
        predicted_probabilities,
        index=x_test.index,
    )


def build_classification_prediction_frame(
    test_rows: pd.DataFrame,
    predicted_probabilities: pd.Series,
    label_column: str = TOP_QUINTILE_LABEL_COLUMN,
) -> pd.DataFrame:
    prediction_frame = test_rows[
        [
            "date",
            "ticker",
            TARGET_COLUMN,
            label_column,
        ]
    ].copy()

    prediction_frame["predicted_probability"] = predicted_probabilities

    prediction_frame = prediction_frame.rename(
        columns={
            TARGET_COLUMN: "actual_return",
            label_column: "actual_label",
        }
    )

    return prediction_frame[
        [
            "date",
            "ticker",
            "predicted_probability",
            "actual_label",
            "actual_return",
        ]
    ]


def rank_predictions_by_probability(
    predictions: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    ranked_predictions = predictions.copy()

    ranked_predictions["probability_rank"] = ranked_predictions.groupby("date")[
        "predicted_probability"
    ].rank(
        ascending=False,
        method="first",
    )

    ranked_predictions["selected_top_5"] = ranked_predictions[
        "probability_rank"
    ] <= top_n

    return ranked_predictions.sort_values(
        [
            "date",
            "probability_rank",
            "ticker",
        ]
    ).reset_index(drop=True)


def predict_one_classification_window(
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
    label_column: str = TOP_QUINTILE_LABEL_COLUMN,
) -> pd.DataFrame:
    x_train, y_train, x_test, test_rows = prepare_classification_window_data(
        historical_rows=historical_rows,
        window=window,
        feature_columns=feature_columns,
        label_column=label_column,
    )

    predicted_probabilities = fit_predict_logistic_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
    )

    prediction_frame = build_classification_prediction_frame(
        test_rows=test_rows,
        predicted_probabilities=predicted_probabilities,
        label_column=label_column,
    )

    ranked_predictions = rank_predictions_by_probability(
        predictions=prediction_frame,
        top_n=5,
    )

    return ranked_predictions


def predict_classification_windows(
    historical_rows: pd.DataFrame,
    windows: list[dict[str, pd.DatetimeIndex]],
    feature_columns: list[str],
) -> pd.DataFrame:
    prediction_frames = []

    for window in windows:
        prediction_frame = predict_one_classification_window(
            historical_rows=historical_rows,
            window=window,
            feature_columns=feature_columns,
        )

        prediction_frames.append(prediction_frame)

    if not prediction_frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "predicted_probability",
                "actual_label",
                "actual_return",
                "probability_rank",
                "selected_top_5",
            ]
        )

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


def build_classification_model_predictions(
    train_size: int = 3,
    validation_size: int = 1,
    test_size: int = 1,
    purge_size: int = 0,
    step_size: int = 1,
) -> pd.DataFrame:
    modeling_dataset, historical_rows, prediction_rows = load_baseline_dataset()

    feature_columns = get_model_feature_columns(historical_rows)

    labeled_historical_rows = add_top_quintile_label(
        data=historical_rows,
    )

    dates = get_sorted_dates(labeled_historical_rows)

    windows = build_walk_forward_date_windows(
        dates=dates,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        purge_size=purge_size,
        step_size=step_size,
    )

    predictions = predict_classification_windows(
        historical_rows=labeled_historical_rows,
        windows=windows,
        feature_columns=feature_columns,
    )

    return predictions


def save_classification_model_predictions(
    predictions: pd.DataFrame,
    output_path: str = CLASSIFICATION_PREDICTIONS_PATH,
) -> str:
    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        output,
        index=False,
    )

    return str(output)


def calculate_classification_metrics_by_date(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metric_rows = []

    for date, date_predictions in predictions.groupby("date"):
        evaluation_rows = date_predictions.dropna(
            subset=[
                "actual_label",
                "selected_top_5",
            ]
        ).copy()

        if evaluation_rows.empty:
            continue

        selected_rows = evaluation_rows[
            evaluation_rows["selected_top_5"]
        ]

        true_positive_count = (
            selected_rows["actual_label"] == 1
        ).sum()

        selected_count = len(selected_rows)

        actual_positive_count = (
            evaluation_rows["actual_label"] == 1
        ).sum()

        precision = (
            true_positive_count / selected_count
            if selected_count > 0
            else pd.NA
        )

        recall = (
            true_positive_count / actual_positive_count
            if actual_positive_count > 0
            else pd.NA
        )

        metric_rows.append(
            {
                "date": date,
                "selected_count": selected_count,
                "actual_positive_count": actual_positive_count,
                "true_positive_count": true_positive_count,
                "precision": precision,
                "recall": recall,
            }
        )

    return pd.DataFrame(
        metric_rows,
        columns=[
            "date",
            "selected_count",
            "actual_positive_count",
            "true_positive_count",
            "precision",
            "recall",
        ],
    )


def summarize_classification_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metrics_by_date = calculate_classification_metrics_by_date(
        predictions=predictions,
    )

    if metrics_by_date.empty:
        return pd.DataFrame(
            columns=[
                "evaluated_dates",
                "average_precision",
                "average_recall",
                "average_selected_count",
            ]
        )

    summary = pd.DataFrame(
        [
            {
                "evaluated_dates": len(metrics_by_date),
                "average_precision": metrics_by_date["precision"].mean(),
                "average_recall": metrics_by_date["recall"].mean(),
                "average_selected_count": metrics_by_date[
                    "selected_count"
                ].mean(),
            }
        ]
    )

    return summary


def main() -> None:
    predictions = build_classification_model_predictions()

    output_path = save_classification_model_predictions(
        predictions=predictions,
    )

    metrics_summary = summarize_classification_metrics(
        predictions=predictions,
    )

    print("Classification model predictions path:", output_path)
    print("Prediction rows:", len(predictions))
    print("Prediction columns:", predictions.columns.tolist())

    print("\nPrediction sample:")
    print(predictions.round(6))

    print("\nClassification metrics summary:")
    print(metrics_summary.round(6))


if __name__ == "__main__":
    main()
