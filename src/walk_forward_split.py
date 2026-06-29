from __future__ import annotations
import pandas as pd

FEATURES_PATH: str = "data/processed/features_momentum.parquet"
LABELS_PATH: str = "data/processed/labels.parquet"

TARGET_COLUMN: str = "forward_relative_return_5d"

LABEL_COLUMNS: list[str] = [
    "forward_return_5d",
    "vn30_forward_return_5d",
    TARGET_COLUMN,
]

KEY_COLUMNS: list[str] = [
    "date",
    "ticker",
]

PURGE_DAYS: int = 5

def load_modeling_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    selected_labels = labels[
        KEY_COLUMNS + LABEL_COLUMNS
    ].copy()

    return features, selected_labels

def build_modeling_dataset(
        features: pd.DataFrame,
        labels: pd.DataFrame,
) -> pd.DataFrame:
    modeling_dataset = features.merge(
        labels, 
        on = KEY_COLUMNS,
        how = "left",
        validate= "one_to_one",
    )
    return modeling_dataset

def separate_target_rows(
        modeling_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical_rows = (
        modeling_dataset[
            modeling_dataset[TARGET_COLUMN].notna()
        ].copy()
    )

    prediction_rows = (
        modeling_dataset[
            modeling_dataset[TARGET_COLUMN].isna()
        ].copy()
    )

    return historical_rows, prediction_rows
def get_sorted_dates(historical_rows: pd.DataFrame,) -> pd.DatetimeIndex:
    unique_dates = pd.DatetimeIndex(historical_rows["date"].drop_duplicates())

    return unique_dates.sort_values()

def build_walk_forward_date_windows(
        dates: pd.DatetimeIndex,
        train_size: int,
        validation_size: int,
        test_size: int,
        purge_size: int,
        step_size: int,

) -> list[dict[str, pd.DatetimeIndex]]:
    windows = []

    required_dates = (
        train_size + purge_size + validation_size + purge_size + test_size
    )

    window_start = 0
    while window_start + required_dates <= len(dates):
        train_start = window_start
        train_end = train_start + train_size

        validation_start = train_end + purge_size
        validation_end = validation_start + validation_size

        test_start = validation_end + purge_size
        test_end = test_start + test_size

        windows.append(
            {
                "train_dates": dates[train_start:train_end],
                "validation_dates": dates[validation_start:validation_end],
                "test_dates": dates[test_start: test_end]
            }
        )

        window_start += step_size
        
    return windows

def split_window_data(
        historical_rows: pd.DataFrame,
        window: dict[str, pd.DatetimeIndex],
        feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series
]: 
    forbidden_columns = KEY_COLUMNS + LABEL_COLUMNS

    if any(
        column in feature_columns
        for column in forbidden_columns
    ): 
        raise ValueError(
            "Features columns cannot contain date, ticker, or target."
        )
   

    train_rows = historical_rows[historical_rows["date"].isin(window["train_dates"])].copy()

    validation_rows = historical_rows[historical_rows["date"].isin(window["validation_dates"])].copy()

    test_rows = historical_rows[historical_rows["date"].isin(window["test_dates"])].copy()

    x_train = train_rows[feature_columns].copy()
    y_train = train_rows[TARGET_COLUMN].copy()

    x_val = validation_rows[feature_columns].copy()
    y_val = validation_rows[TARGET_COLUMN].copy()

    x_test = test_rows[feature_columns].copy()
    y_test = test_rows[TARGET_COLUMN].copy()

    return x_train, y_train, x_val, y_val, x_test, y_test

def main() -> None: 
    print("Features path:", FEATURES_PATH)
    print("labels path:", LABELS_PATH)
    print("Target column:", TARGET_COLUMN)
    print("Key columns:", KEY_COLUMNS)
    print("Purge days:", PURGE_DAYS)

    features, labels = load_modeling_inputs()

    print("\nFeature rows", len(features))
    print("Selected label rows:", len(labels))
    print("Selected label columns:", labels.columns.tolist())

    modeling_dataset = build_modeling_dataset(
        features,
        labels
    )

    duplicate_keys = modeling_dataset.duplicated(KEY_COLUMNS).sum()

    print("\nModeling dataset rows:", len(modeling_dataset))
    print("Modeling dataset columns:", len(modeling_dataset.columns))
    print("Duplicate ticker-date keys:", duplicate_keys)
    print(
        "Valid target rows:",
        modeling_dataset[TARGET_COLUMN].notna().sum(),
    )

    print(
        "Missing target rows:",
        modeling_dataset[TARGET_COLUMN].isna().sum(),
    )

    historical_rows, prediction_rows = separate_target_rows(
        modeling_dataset
    )

    print("\nHistorical rows:", len(historical_rows))
    print("Prediction-only rows:", len(prediction_rows))
    print(
    "Historical targets all known:",
    historical_rows[TARGET_COLUMN].notna().all(),
    )
    print(
    "Prediction targets all missing:",
    prediction_rows[TARGET_COLUMN].isna().all(),
    )

    historical_dates = get_sorted_dates(historical_rows)
    print("\nHistorical date count", len(historical_dates))
    print("Earliest historical date:", historical_dates.min())
    print("Latest historical date:", historical_dates.max())    

    windows = build_walk_forward_date_windows(dates = historical_dates,
                                              train_size=1,
                                              validation_size= 1, 
                                              test_size= 1,
                                              purge_size= PURGE_DAYS, 
                                              step_size= 1)
    print("Walk-forward windows from official sample:", len(windows))

if __name__ == "__main__":
    main()