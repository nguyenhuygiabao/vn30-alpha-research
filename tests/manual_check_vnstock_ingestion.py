from __future__ import annotations

import pandas as pd


RAW_DATA_PATH: str = "data/raw/vnstock/vn30_ohlcv.csv"
AUDIT_PATH: str = "data/raw/vnstock/vn30_coverage_audit.csv"
UNIVERSE_PATH: str = "config/vn30_universe.csv"


data = pd.read_csv(RAW_DATA_PATH)
audit = pd.read_csv(AUDIT_PATH)
universe = pd.read_csv(UNIVERSE_PATH)

required_data_columns = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "value_traded",
]

required_audit_columns = [
    "ticker",
    "vnstock_symbol",
    "issuer_group",
    "status",
    "rows",
    "start",
    "end",
    "error",
]

required_universe_columns = [
    "ticker",
    "vnstock_symbol",
    "issuer_group",
]

missing_data_columns = [
    column
    for column in required_data_columns
    if column not in data.columns
]

missing_audit_columns = [
    column
    for column in required_audit_columns
    if column not in audit.columns
]

missing_universe_columns = [
    column
    for column in required_universe_columns
    if column not in universe.columns
]

data["date"] = pd.to_datetime(data["date"])

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "value_traded",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

duplicate_keys = data.duplicated(
    [
        "date",
        "ticker",
    ]
).sum()

expected_tickers = sorted(universe["ticker"].unique().tolist())
actual_tickers = sorted(data["ticker"].unique().tolist())
audit_tickers = sorted(audit["ticker"].unique().tolist())

rows_by_ticker = data.groupby("ticker").size()

non_positive_prices = (
    data[
        [
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
        ]
    ]
    <= 0
).sum().sum()

negative_volume_rows = (data["volume"] < 0).sum()
negative_value_rows = (data["value_traded"] < 0).sum()

missing_numeric_values = data[numeric_columns].isna().sum().sum()

audit_ok_count = (audit["status"] == "ok").sum()
audit_error_count = (audit["status"] == "error").sum()

short_history_tickers = rows_by_ticker[
    rows_by_ticker < 1000
].index.tolist()

has_expected_short_history = short_history_tickers == ["VPL"]

merged = data.merge(
    universe,
    on="ticker",
    how="left",
    validate="many_to_one",
)

missing_issuer_group_rows = merged["issuer_group"].isna().sum()

vingroup_tickers = sorted(
    universe
    .query("issuer_group == 'Vingroup'")
    ["ticker"]
    .tolist()
)

required_data_columns_present = missing_data_columns == []
required_audit_columns_present = missing_audit_columns == []
required_universe_columns_present = missing_universe_columns == []
duplicate_keys_absent = duplicate_keys == 0
tickers_match_universe = actual_tickers == expected_tickers
audit_tickers_match_universe = audit_tickers == expected_tickers
all_audit_rows_ok = audit_ok_count == len(universe)
no_audit_errors = audit_error_count == 0
enough_total_rows = len(data) > 40000
prices_positive = non_positive_prices == 0
volumes_non_negative = negative_volume_rows == 0
values_non_negative = negative_value_rows == 0
numeric_values_complete = missing_numeric_values == 0
issuer_groups_complete = missing_issuer_group_rows == 0
vingroup_metadata_present = vingroup_tickers == [
    "VHM",
    "VIC",
    "VPL",
    "VRE",
]

print("Raw data rows:", len(data))
print("Raw data columns:", len(data.columns))
print("Audit rows:", len(audit))
print("Universe rows:", len(universe))
print("Earliest date:", data["date"].min())
print("Latest date:", data["date"].max())
print("Expected tickers:", expected_tickers)
print("Actual tickers:", actual_tickers)
print("Rows by ticker:")
print(rows_by_ticker.to_string())
print("Short-history tickers:", short_history_tickers)
print("Vingroup tickers:", vingroup_tickers)
print("Missing data columns:", missing_data_columns)
print("Missing audit columns:", missing_audit_columns)
print("Missing universe columns:", missing_universe_columns)
print("Duplicate ticker-date keys:", duplicate_keys)
print("Non-positive price values:", non_positive_prices)
print("Negative volume rows:", negative_volume_rows)
print("Negative value-traded rows:", negative_value_rows)
print("Missing numeric values:", missing_numeric_values)
print("Audit OK count:", audit_ok_count)
print("Audit error count:", audit_error_count)
print("Missing issuer-group rows:", missing_issuer_group_rows)

print("\nRequired data columns present:", required_data_columns_present)
print("Required audit columns present:", required_audit_columns_present)
print("Required universe columns present:", required_universe_columns_present)
print("Duplicate keys absent:", duplicate_keys_absent)
print("Tickers match universe:", tickers_match_universe)
print("Audit tickers match universe:", audit_tickers_match_universe)
print("All audit rows OK:", all_audit_rows_ok)
print("No audit errors:", no_audit_errors)
print("Enough total rows:", enough_total_rows)
print("Expected short-history ticker only:", has_expected_short_history)
print("Prices positive:", prices_positive)
print("Volumes non-negative:", volumes_non_negative)
print("Values non-negative:", values_non_negative)
print("Numeric values complete:", numeric_values_complete)
print("Issuer groups complete:", issuer_groups_complete)
print("Vingroup metadata present:", vingroup_metadata_present)