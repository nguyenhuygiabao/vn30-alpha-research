import pandas as pd

from src.walk_forward_split import (
    PURGE_DAYS,
    TARGET_COLUMN,
    build_walk_forward_date_windows,
    split_window_data,
)

dates = pd.bdate_range(
    start="2024-01-02",
    periods=40,
)

windows = build_walk_forward_date_windows(
    dates=dates,
    train_size=12,
    validation_size=5,
    test_size=5,
    purge_size=PURGE_DAYS,
    step_size=5,
)

print("Window count:", len(windows))

synthetic_rows = []

for ticker_number, ticker in enumerate(["AAA", "BBB"]):
    for date_number, date in enumerate(dates):
        synthetic_rows.append(
            {
                "date": date,
                "ticker": ticker,
                "return_1d": date_number/100, 
                "return_5d": date_number/50, 
                TARGET_COLUMN: date_number/200 + ticker_number/1000,
            }
        )

synthetic_panel = pd.DataFrame(synthetic_rows)

feature_columns = [
    "return_1d",
    "return_5d",
]

(
    x_train, 
    y_train, 
    x_val, 
    y_val,
    x_test,
    y_test,
) = split_window_data(
    historical_rows = synthetic_panel,
    window = windows[0],
    feature_columns = feature_columns,
)

print("\nFirst window shapes:")
print("x_train:", x_train.shape, "y_train", y_train.shape)
print("x_val:", x_val.shape, "y_val", y_val.shape)
print("x_test:", x_test.shape, "y_test", y_test.shape)

first_window = windows[0]

train_dates = first_window["train_dates"]
validation_dates = first_window["validation_dates"]
test_dates = first_window["test_dates"]

chronological_order_valid = (
    train_dates.max()
    < validation_dates.min()
    < test_dates.min()
)

no_date_overlap = (
    set(train_dates).isdisjoint(validation_dates)
    and set(train_dates).isdisjoint(test_dates)
    and set(validation_dates).isdisjoint(test_dates)
)

print("\nSplit integrity checks:")
print("Dates are chronological:", chronological_order_valid)
print("No date appears in multiple sets:", no_date_overlap)

expected_tickers_per_date = synthetic_panel["ticker"].nunique()
train_panel = synthetic_panel[synthetic_panel["date"].isin(train_dates)]
validation_panel = synthetic_panel[synthetic_panel["date"].isin(validation_dates)]
test_panel = synthetic_panel[synthetic_panel["date"].isin(test_dates)]

all_tickers_stay_together = all(
    panel.groupby("date")["ticker"]
    .nunique()
    .eq(expected_tickers_per_date)
    .all()
    for panel in [
        train_panel,
        validation_panel,
        test_panel,
    ]
)

print(
    "All tickers on each date stay together:",
    all_tickers_stay_together,
)

try: 
    split_window_data(
        historical_rows = synthetic_panel,
        window = windows[0],
        feature_columns = [
            "return_1d",
            TARGET_COLUMN
        ],
    )

    leakage_guard_valid = False

except ValueError:
    leakage_guard_valid = True

print(
    "Target leakage guard works:",
    leakage_guard_valid,
)

row_counts_match = (
    len(x_train) == len(y_train)
    and len(x_val) == len(y_val)
    and len(x_test) == len(y_test)
)

all_targets_known = (
    y_train.notna().all()
    and y_val.notna().all()
    and y_test.notna().all()
)

feature_columns_match = (
    x_train.columns.tolist() == feature_columns
    and x_val.columns.tolist() == feature_columns
    and x_test.columns.tolist() == feature_columns
)

print("Feature and target row counts match:", row_counts_match)
print("All split targets are known:", all_targets_known)
print("Feature columns are correct:", feature_columns_match)

for window_number, window in enumerate(windows, start=1):
    print(f"\nWindow {window_number}")

    print(
        "Train:",
        window["train_dates"].min().date(),
        "to",
        window["train_dates"].max().date(),
    )

    print(
        "Validation:",
        window["validation_dates"].min().date(),
        "to",
        window["validation_dates"].max().date(),
    )

    print(
        "Test:",
        window["test_dates"].min().date(),
        "to",
        window["test_dates"].max().date(),
    )