from __future__ import annotations

import pandas as pd


RAW_DATA_PATH: str = "data/raw/yahoo/vn30_test_ohlcv.csv"
LABELS_PATH: str = "data/processed/labels.parquet"

MAIN_TARGET: str = "forward_relative_return_5d"
FORWARD_RETURN: str = "forward_return_5d"
BENCHMARK_RETURN: str = "vn30_forward_return_5d"


raw_data = pd.read_csv(RAW_DATA_PATH)
labels = pd.read_parquet(LABELS_PATH)

raw_data["date"] = pd.to_datetime(raw_data["date"])
labels["date"] = pd.to_datetime(labels["date"])

required_columns = [
    "date",
    "ticker",
    "forward_return_1d",
    "forward_return_5d",
    "forward_return_10d",
    "vn30_forward_return_1d",
    "vn30_forward_return_5d",
    "vn30_forward_return_10d",
    "forward_relative_return_1d",
    "forward_relative_return_5d",
    "forward_relative_return_10d",
]

missing_columns = [
    column
    for column in required_columns
    if column not in labels.columns
]

duplicate_keys = labels.duplicated(
    [
        "date",
        "ticker",
    ]
).sum()

expected_tickers = sorted(raw_data["ticker"].unique().tolist())
actual_tickers = sorted(labels["ticker"].unique().tolist())

missing_main_target_rows = labels[MAIN_TARGET].isna().sum()
valid_main_target_rows = labels[MAIN_TARGET].notna().sum()

missing_main_target_by_ticker = (
    labels
    .groupby("ticker")[MAIN_TARGET]
    .apply(lambda series: series.isna().sum())
)

valid_rows = labels[
    labels[
        [
            FORWARD_RETURN,
            BENCHMARK_RETURN,
            MAIN_TARGET,
        ]
    ]
    .notna()
    .all(axis=1)
].copy()

relative_difference = (
    valid_rows[FORWARD_RETURN]
    - valid_rows[BENCHMARK_RETURN]
    - valid_rows[MAIN_TARGET]
).abs()

date_return_sum = (
    labels
    .groupby("date")[FORWARD_RETURN]
    .transform("sum")
)

date_return_count = (
    labels
    .groupby("date")[FORWARD_RETURN]
    .transform("count")
)

expected_leave_one_out_benchmark = (
    date_return_sum
    - labels[FORWARD_RETURN]
) / (
    date_return_count
    - 1
)

benchmark_check_rows = labels[
    labels[
        [
            FORWARD_RETURN,
            BENCHMARK_RETURN,
        ]
    ]
    .notna()
    .all(axis=1)
].copy()

benchmark_difference = (
    benchmark_check_rows[BENCHMARK_RETURN]
    - expected_leave_one_out_benchmark.loc[benchmark_check_rows.index]
).abs()

required_columns_present = missing_columns == []
rows_match_raw_data = len(labels) == len(raw_data)
duplicate_keys_absent = duplicate_keys == 0
tickers_match_raw_data = actual_tickers == expected_tickers
valid_targets_exist = valid_main_target_rows > 10000
missing_target_rows_correct = missing_main_target_rows == 30
missing_target_by_ticker_correct = (
    missing_main_target_by_ticker == 5
).all()
relative_formula_correct = relative_difference.max() < 1e-12
leave_one_out_benchmark_correct = benchmark_difference.max() < 1e-12

print("Raw rows:", len(raw_data))
print("Label rows:", len(labels))
print("Expected tickers:", expected_tickers)
print("Actual tickers:", actual_tickers)
print("Missing columns:", missing_columns)
print("Duplicate ticker-date keys:", duplicate_keys)
print("Valid main-target rows:", valid_main_target_rows)
print("Missing main-target rows:", missing_main_target_rows)
print("Missing main-target rows by ticker:")
print(missing_main_target_by_ticker.to_string())
print("Max relative formula difference:", relative_difference.max())
print("Max leave-one-out benchmark difference:", benchmark_difference.max())

print("\nRequired columns present:", required_columns_present)
print("Rows match raw data:", rows_match_raw_data)
print("Duplicate keys absent:", duplicate_keys_absent)
print("Tickers match raw data:", tickers_match_raw_data)
print("Valid targets exist:", valid_targets_exist)
print("Missing target rows correct:", missing_target_rows_correct)
print("Missing target by ticker correct:", missing_target_by_ticker_correct)
print("Relative formula correct:", relative_formula_correct)
print("Leave-one-out benchmark correct:", leave_one_out_benchmark_correct)