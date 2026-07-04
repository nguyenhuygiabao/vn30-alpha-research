from __future__ import annotations
from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.baselines import load_baseline_dataset
from src.linear_models import summarize_model_performance
from src.modeling_utils import (
    build_prediction_frame,
    get_model_feature_columns,
    prepare_regression_window_data,
)
from src.walk_forward_split import (
    TARGET_COLUMN,
    build_walk_forward_date_windows,
    get_sorted_dates,
)

TREE_MODEL_PREDICTIONS_PATH: str = "data/processed/tree_model_predictions.parquet"
TREE_FEATURE_IMPORTANCE_PATH: str = "data/processed/tree_feature_importance.parquet"


def create_tree_models() -> dict[str, object]:
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    return models


def fit_predict_tree_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[pd.Series, object]:
    clean_x_train = x_train.astype(float)
    clean_x_test = x_test.astype(float)

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
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        clean_x_train,
        y_train,
    )

    predictions = pipeline.predict(clean_x_test)

    return (
        pd.Series(
            predictions,
            index=x_test.index,
            name="predicted_return",
        ),
        pipeline,
    )


def extract_feature_importance(
    fitted_pipeline,
    feature_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    fitted_model = fitted_pipeline.named_steps["model"]

    if not hasattr(fitted_model, "feature_importances_"):
        return pd.DataFrame(
            columns=[
                "model_name",
                "feature",
                "importance",
            ]
        )

    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": feature_columns,
            "importance": fitted_model.feature_importances_,
        }
    )

    return importance.sort_values(
        [
            "importance",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


def predict_prepared_tree_window(
    model,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    test_rows: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, fitted_pipeline = fit_predict_tree_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
    )

    prediction_frame = build_prediction_frame(
        test_rows=test_rows,
        predictions=predictions,
        model_name=model_name,
    )

    feature_importance = extract_feature_importance(
        fitted_pipeline=fitted_pipeline,
        feature_columns=feature_columns,
        model_name=model_name,
    )

    return prediction_frame, feature_importance


def predict_one_tree_window(
    model,
    model_name: str,
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, y_train, x_test, test_rows = prepare_regression_window_data(
        historical_rows=historical_rows,
        window=window,
        feature_columns=feature_columns,
    )

    return predict_prepared_tree_window(
        model=model,
        model_name=model_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        test_rows=test_rows,
        feature_columns=feature_columns,
    )


def predict_tree_models_for_window(
    models: dict[str, object],
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    feature_importance_frames = []

    x_train, y_train, x_test, test_rows = prepare_regression_window_data(
        historical_rows=historical_rows,
        window=window,
        feature_columns=feature_columns,
    )

    for model_name, model in models.items():
        prediction_frame, feature_importance = predict_prepared_tree_window(
            model=model,
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            test_rows=test_rows,
            feature_columns=feature_columns,
        )

        feature_importance["test_start_date"] = window["test_dates"].min()
        feature_importance["test_end_date"] = window["test_dates"].max()

        prediction_frames.append(prediction_frame)
        feature_importance_frames.append(feature_importance)

    if prediction_frames:
        predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
    else:
        predictions = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "predicted_return",
                "actual_return",
                "model_name",
            ]
        )

    if feature_importance_frames:
        feature_importance = pd.concat(
            feature_importance_frames,
            ignore_index=True,
        )
    else:
        feature_importance = pd.DataFrame(
            columns=[
                "model_name",
                "feature",
                "importance",
                "test_start_date",
                "test_end_date",
            ]
        )

    return predictions, feature_importance


def predict_tree_models_for_windows(
    models: dict[str, object],
    historical_rows: pd.DataFrame,
    windows: list[dict[str, pd.DatetimeIndex]],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    feature_importance_frames = []

    for window in windows:
        predictions, feature_importance = predict_tree_models_for_window(
            models=models,
            historical_rows=historical_rows,
            window=window,
            feature_columns=feature_columns,
        )

        prediction_frames.append(predictions)
        feature_importance_frames.append(feature_importance)

    if prediction_frames:
        all_predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
    else:
        all_predictions = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "predicted_return",
                "actual_return",
                "model_name",
            ]
        )

    if feature_importance_frames:
        all_feature_importance = pd.concat(
            feature_importance_frames,
            ignore_index=True,
        )
    else:
        all_feature_importance = pd.DataFrame(
            columns=[
                "model_name",
                "feature",
                "importance",
                "test_start_date",
                "test_end_date",
            ]
        )

    return all_predictions, all_feature_importance


def build_tree_model_outputs(
    train_size: int = 3,
    validation_size: int = 1,
    test_size: int = 1,
    purge_size: int = 0,
    step_size: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeling_dataset, historical_rows, prediction_rows = load_baseline_dataset()

    feature_columns = get_model_feature_columns(historical_rows)

    dates = get_sorted_dates(historical_rows)

    windows = build_walk_forward_date_windows(
        dates=dates,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        purge_size=purge_size,
        step_size=step_size,
    )

    models = create_tree_models()

    predictions, feature_importance = predict_tree_models_for_windows(
        models=models,
        historical_rows=historical_rows,
        windows=windows,
        feature_columns=feature_columns,
    )

    return predictions, feature_importance


def save_tree_model_outputs(
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    predictions_path: str = TREE_MODEL_PREDICTIONS_PATH,
    feature_importance_path: str = TREE_FEATURE_IMPORTANCE_PATH,
) -> tuple[str, str]:
    predictions_output = Path(predictions_path)
    feature_importance_output = Path(feature_importance_path)

    predictions_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_importance_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        predictions_output,
        index=False,
    )

    feature_importance.to_parquet(
        feature_importance_output,
        index=False,
    )

    return str(predictions_output), str(feature_importance_output)


def summarize_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    if feature_importance.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "feature",
                "average_importance",
                "importance_observations",
            ]
        )

    summary = feature_importance.groupby(
        [
            "model_name",
            "feature",
        ],
        as_index=False,
    ).agg(
        average_importance=("importance", "mean"),
        importance_observations=("importance", "count"),
    )

    return summary.sort_values(
        [
            "model_name",
            "average_importance",
            "feature",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).groupby("model_name").head(top_n).reset_index(drop=True)


def compare_tree_models_with_linear_models(
    tree_predictions: pd.DataFrame,
    linear_predictions_path: str = "data/processed/linear_model_predictions.parquet",
    top_n: int = 5,
) -> pd.DataFrame:
    comparison_frames = []

    if not tree_predictions.empty:
        tree_summary = summarize_model_performance(
            predictions=tree_predictions,
            top_n=top_n,
        )

        comparison_frames.append(tree_summary)

    linear_path = Path(linear_predictions_path)

    if linear_path.exists():
        linear_predictions = pd.read_parquet(linear_path)

        linear_summary = summarize_model_performance(
            predictions=linear_predictions,
            top_n=top_n,
        )

        comparison_frames.append(linear_summary)

    if not comparison_frames:
        return pd.DataFrame(
            columns=[
                "model_name",
                "evaluated_dates",
                "average_rank_ic",
                "average_top_n_actual_return",
                "average_selected_count",
            ]
        )

    comparison = pd.concat(
        comparison_frames,
        ignore_index=True,
    )

    return comparison.sort_values(
        [
            "average_rank_ic",
            "average_top_n_actual_return",
            "model_name",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)


def main() -> None:
    predictions, feature_importance = build_tree_model_outputs()

    predictions_path, feature_importance_path = save_tree_model_outputs(
        predictions=predictions,
        feature_importance=feature_importance,
    )

    tree_performance_summary = summarize_model_performance(
        predictions=predictions,
        top_n=5,
    )

    model_comparison = compare_tree_models_with_linear_models(
        tree_predictions=predictions,
        top_n=5,
    )

    feature_importance_summary = summarize_feature_importance(
        feature_importance=feature_importance,
        top_n=20,
    )

    print("Tree model predictions path:", predictions_path)
    print("Tree feature importance path:", feature_importance_path)
    print("Prediction rows:", len(predictions))
    print("Feature importance rows:", len(feature_importance))

    print("\nTree model performance summary:")
    print(tree_performance_summary.round(6))

    print("\nTree vs linear model comparison:")
    print(model_comparison.round(6))

    print("\nTop feature importance:")
    print(feature_importance_summary.round(6))


if __name__ == "__main__":
    main()
