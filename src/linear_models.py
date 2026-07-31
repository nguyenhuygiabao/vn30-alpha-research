from __future__ import annotations
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines import load_baseline_dataset
from src.modeling_utils import (
    build_prediction_frame,
    get_model_feature_columns,
)
from src.walk_forward_split import (
    DEFAULT_TRAIN_SIZE,
    build_walk_forward_date_windows,
    get_sorted_dates,
    split_window_data,
)


PREDICTIONS_PATH: str = "data/processed/linear_model_predictions.parquet"

def fit_predict_linear_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> pd.Series:

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
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    pipeline.fit(clean_x_train, y_train)

    predictions = pipeline.predict(clean_x_test)

    return pd.Series(
        predictions,
        index=x_test.index,
        name="predicted_return",
    )


def _empty_linear_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "predicted_return",
            "actual_return",
            "model_name",
        ]
    )


def _prepare_linear_window(
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    x_train, y_train, _x_val, _y_val, x_test, _y_test = split_window_data(
        historical_rows=historical_rows,
        window=window,
        feature_columns=feature_columns,
    )

    test_rows = historical_rows[
        historical_rows["date"].isin(window["test_dates"])
    ].copy()

    return x_train, y_train, x_test, test_rows


def predict_prepared_linear_window(
    model,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> pd.DataFrame:
    predictions = fit_predict_linear_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
    )

    return build_prediction_frame(
        test_rows=test_rows,
        predictions=predictions,
        model_name=model_name,
    )


def predict_models_for_prepared_window(
    models: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> pd.DataFrame:
    prediction_frames = [
        predict_prepared_linear_window(
            model=model,
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            test_rows=test_rows,
        )
        for model_name, model in models.items()
    ]

    if not prediction_frames:
        return _empty_linear_predictions()

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


def predict_one_window(
    model,
    model_name: str,
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
) -> pd.DataFrame:
    x_train, y_train, x_test, test_rows = _prepare_linear_window(
        historical_rows=historical_rows,
        window=window,
        feature_columns=feature_columns,
    )

    return predict_prepared_linear_window(
        model=model,
        model_name=model_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        test_rows=test_rows,
    )


def create_linear_models() -> dict[str, object]:
    models = {
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10000,
        ),
    }

    return models


def predict_models_for_window(
    models: dict[str, object],
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
) -> pd.DataFrame:
    x_train, y_train, x_test, test_rows = _prepare_linear_window(
        historical_rows=historical_rows,
        window=window,
        feature_columns=feature_columns,
    )

    return predict_models_for_prepared_window(
        models=models,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        test_rows=test_rows,
    )


def predict_models_for_windows(
    models: dict[str, object],
    historical_rows: pd.DataFrame,
    windows: list[dict[str, pd.DatetimeIndex]],
    feature_columns: list[str],
    n_jobs: int = -1,
) -> pd.DataFrame:
    # Each walk-forward window is an independent fit. Slicing happens up
    # front (cheap, sequential) so only the small per-window frames -- not
    # the full historical_rows table -- cross process boundaries; results
    # come back in submission order, so output is identical to the
    # sequential loop regardless of n_jobs.
    prepared_windows = [
        _prepare_linear_window(
            historical_rows=historical_rows,
            window=window,
            feature_columns=feature_columns,
        )
        for window in windows
    ]

    prediction_frames = Parallel(n_jobs=n_jobs)(
        delayed(predict_models_for_prepared_window)(
            models=models,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            test_rows=test_rows,
        )
        for x_train, y_train, x_test, test_rows in prepared_windows
    )

    if not prediction_frames:
        return _empty_linear_predictions()

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


def build_linear_model_predictions(
    train_size: int = DEFAULT_TRAIN_SIZE,
    validation_size: int = 1,
    test_size: int = 1,
    purge_size: int = 5,
    step_size: int = 1,
) -> pd.DataFrame:
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

    models = create_linear_models()

    predictions = predict_models_for_windows(
        models=models,
        historical_rows=historical_rows,
        windows=windows,
        feature_columns=feature_columns,
    )

    return predictions


def save_linear_model_predictions(
    predictions: pd.DataFrame,
    output_path: str = PREDICTIONS_PATH,
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


def calculate_rank_ic_by_date(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rank_ic_rows = []

    grouped_predictions = predictions.groupby(
        [
            "model_name",
            "date",
        ]
    )

    for (model_name, date), group in grouped_predictions:
        evaluation_group = group[
            [
                "predicted_return",
                "actual_return",
            ]
        ].dropna()

        if len(evaluation_group) < 2:
            continue

        if evaluation_group["predicted_return"].nunique() < 2:
            continue

        if evaluation_group["actual_return"].nunique() < 2:
            continue

        rank_ic = evaluation_group["predicted_return"].corr(
            evaluation_group["actual_return"],
            method="spearman",
        )

        rank_ic_rows.append(
            {
                "model_name": model_name,
                "date": date,
                "rank_ic": rank_ic,
                "ticker_count": len(evaluation_group),
            }
        )

    return pd.DataFrame(
        rank_ic_rows,
        columns=[
            "model_name",
            "date",
            "rank_ic",
            "ticker_count",
        ],
    )


def calculate_top_n_average_return(
    predictions: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    top_n_rows = []

    grouped_predictions = predictions.groupby(
        [
            "model_name",
            "date",
        ]
    )

    for (model_name, date), group in grouped_predictions:
        evaluation_group = group.dropna(
            subset=[
                "predicted_return",
                "actual_return",
            ]
        ).copy()

        if evaluation_group.empty:
            continue

        selected = evaluation_group.nlargest(
            top_n,
            "predicted_return",
        )

        top_n_rows.append(
            {
                "model_name": model_name,
                "date": date,
                "top_n": top_n,
                "selected_count": len(selected),
                "top_n_average_actual_return": selected["actual_return"].mean(),
            }
        )

    return pd.DataFrame(
        top_n_rows,
        columns=[
            "model_name",
            "date",
            "top_n",
            "selected_count",
            "top_n_average_actual_return",
        ],
    )


def summarize_model_performance(
    predictions: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    rank_ic_by_date = calculate_rank_ic_by_date(
        predictions=predictions,
    )

    top_n_returns = calculate_top_n_average_return(
        predictions=predictions,
        top_n=top_n,
    )

    rank_ic_summary = rank_ic_by_date.groupby(
        "model_name",
        as_index=False,
    ).agg(
        evaluated_dates=("date", "count"),
        average_rank_ic=("rank_ic", "mean"),
    )

    top_n_summary = top_n_returns.groupby(
        "model_name",
        as_index=False,
    ).agg(
        average_top_n_actual_return=("top_n_average_actual_return", "mean"),
        average_selected_count=("selected_count", "mean"),
    )

    summary = rank_ic_summary.merge(
        top_n_summary,
        on="model_name",
        how="outer",
        validate="one_to_one",
    )

    return summary[
        [
            "model_name",
            "evaluated_dates",
            "average_rank_ic",
            "average_top_n_actual_return",
            "average_selected_count",
        ]
    ]


def main() -> None:
    predictions = build_linear_model_predictions()

    output_path = save_linear_model_predictions(
        predictions=predictions,
    )

    performance_summary = summarize_model_performance(
        predictions=predictions,
        top_n=5,
    )

    print("Linear model predictions path:", output_path)
    print("Prediction rows:", len(predictions))
    print("Prediction columns:", predictions.columns.tolist())

    print("\nModels evaluated:")
    print(sorted(predictions["model_name"].unique().tolist()))

    print("\nPrediction sample:")
    print(predictions.round(6))

    print("\nModel performance summary:")
    print(performance_summary.round(6))


if __name__ == "__main__":
    main()
