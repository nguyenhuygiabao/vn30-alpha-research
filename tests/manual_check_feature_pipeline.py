import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.feature_pipeline import build_combined_features
from src.linear_models import get_model_feature_columns
from src.walk_forward_split import (
    KEY_COLUMNS,
    LABEL_COLUMNS,
    build_modeling_dataset,
    load_modeling_inputs,
    separate_target_rows,
)


data = load_ohlcv_csv("sample_data/sample_ohlcv.csv")

features = build_combined_features(data)

features_from_path, labels_from_path = load_modeling_inputs()

modeling_dataset = build_modeling_dataset(
    features=features_from_path,
    labels=labels_from_path,
)

historical_rows, prediction_rows = separate_target_rows(modeling_dataset)

feature_columns = get_model_feature_columns(historical_rows)

required_columns = [
    "date",
    "ticker",
    "simple_return_1d",
    "return_1d",
    "return_5d",
    "distance_to_ceiling",
    "distance_to_floor",
    "volume_z_20",
    "amihud_illiquidity",
    "herding_index",
]

missing_columns = [
    column
    for column in required_columns
    if column not in features.columns
]

duplicate_keys = features.duplicated(KEY_COLUMNS).sum()

forbidden_columns = set(KEY_COLUMNS + LABEL_COLUMNS)
forbidden_feature_columns = [
    column
    for column in feature_columns
    if column in forbidden_columns
]

non_numeric_features = historical_rows[feature_columns].select_dtypes(
    exclude=[
        "number",
        "bool",
    ]
).columns.tolist()

can_cast_features_to_float = True

try:
    historical_rows[feature_columns].astype(float)
except ValueError:
    can_cast_features_to_float = False
except TypeError:
    can_cast_features_to_float = False

rows_preserved = len(features) == len(data)
required_columns_exist = missing_columns == []
duplicate_keys_absent = duplicate_keys == 0
modeling_rows_correct = len(modeling_dataset) == len(features_from_path)
historical_targets_known = historical_rows["forward_relative_return_5d"].notna().all()
prediction_targets_missing = prediction_rows["forward_relative_return_5d"].isna().all()
forbidden_columns_excluded = forbidden_feature_columns == []
features_castable = can_cast_features_to_float

print("Input rows:", len(data))
print("Combined feature rows:", len(features))
print("Combined feature columns:", len(features.columns))
print("Features from official path rows:", len(features_from_path))
print("Modeling dataset rows:", len(modeling_dataset))
print("Historical rows:", len(historical_rows))
print("Prediction rows:", len(prediction_rows))
print("Candidate model feature columns:", len(feature_columns))
print("Missing required columns:", missing_columns)
print("Duplicate ticker-date keys:", duplicate_keys)
print("Forbidden feature columns:", forbidden_feature_columns)
print("Non-numeric feature columns:", non_numeric_features)

print("\nRows preserved:", rows_preserved)
print("Required columns exist:", required_columns_exist)
print("Duplicate keys absent:", duplicate_keys_absent)
print("Modeling rows correct:", modeling_rows_correct)
print("Historical targets known:", historical_targets_known)
print("Prediction targets missing:", prediction_targets_missing)
print("Forbidden columns excluded:", forbidden_columns_excluded)
print("Features castable to float:", features_castable)
