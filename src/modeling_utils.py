from __future__ import annotations

import pandas as pd

from src.walk_forward_split import (
    KEY_COLUMNS,
    LABEL_COLUMNS,
    TARGET_COLUMN,
)


RAW_MARKET_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "value_traded",
]


def get_model_feature_columns(
    data: pd.DataFrame,
) -> list[str]:
    forbidden_columns = set(
        KEY_COLUMNS
        + LABEL_COLUMNS
        + RAW_MARKET_COLUMNS
    )

    candidate_columns = [
        column
        for column in data.columns
        if column not in forbidden_columns
    ]

    feature_columns = [
        column
        for column in candidate_columns
        if not data[column].isna().all()
    ]

    return feature_columns


def build_prediction_frame(
    test_rows: pd.DataFrame,
    predictions: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    prediction_frame = test_rows[
        [
            "date",
            "ticker",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_frame["predicted_return"] = predictions
    prediction_frame["model_name"] = model_name

    prediction_frame = prediction_frame.rename(
        columns={
            TARGET_COLUMN: "actual_return",
        }
    )

    return prediction_frame[
        [
            "date",
            "ticker",
            "predicted_return",
            "actual_return",
            "model_name",
        ]
    ]


def prepare_regression_window_data(
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    train_rows = historical_rows.loc[
        historical_rows["date"].isin(window["train_dates"])
    ].copy()
    test_rows = historical_rows.loc[
        historical_rows["date"].isin(window["test_dates"])
    ].copy()

    x_train = train_rows[feature_columns].copy()
    y_train = train_rows[target_column].copy()
    x_test = test_rows[feature_columns].copy()

    return x_train, y_train, x_test, test_rows


def prepare_classification_window_data(
    historical_rows: pd.DataFrame,
    window: dict[str, pd.DatetimeIndex],
    feature_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    train_rows = historical_rows.loc[
        historical_rows["date"].isin(window["train_dates"])
    ].copy()
    test_rows = historical_rows.loc[
        historical_rows["date"].isin(window["test_dates"])
    ].copy()

    train_rows = train_rows.loc[
        train_rows[label_column].notna()
    ].copy()
    test_rows = test_rows.loc[
        test_rows[label_column].notna()
    ].copy()

    x_train = train_rows[feature_columns].copy()
    y_train = train_rows[label_column].copy()
    x_test = test_rows[feature_columns].copy()

    return x_train, y_train, x_test, test_rows
