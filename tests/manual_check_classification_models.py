import pandas as pd

from src.classification_models import (
    TOP_QUINTILE_LABEL_COLUMN,
    add_top_quintile_label,
    calculate_classification_metrics_by_date,
    predict_classification_windows,
    summarize_classification_metrics,
)
from src.walk_forward_split import (
    TARGET_COLUMN,
    build_walk_forward_date_windows,
    get_sorted_dates,
)


data = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2024-01-01"] * 5
            + ["2024-01-02"] * 5
            + ["2024-01-03"] * 5
        ),
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"] * 3,
        "feature_a": [
            0,
            1,
            2,
            3,
            4,
            4,
            3,
            2,
            1,
            0,
            0,
            1,
            2,
            3,
            4,
        ],
        "feature_b": [
            4,
            3,
            2,
            1,
            0,
            0,
            1,
            2,
            3,
            4,
            4,
            3,
            2,
            1,
            0,
        ],
        TARGET_COLUMN: [
            0.00,
            0.01,
            0.02,
            0.03,
            0.04,
            0.04,
            0.03,
            0.02,
            0.01,
            0.00,
            0.00,
            0.01,
            0.02,
            0.03,
            0.04,
        ],
    }
)

labeled_data = add_top_quintile_label(data)

dates = get_sorted_dates(labeled_data)

windows = build_walk_forward_date_windows(
    dates=dates,
    train_size=1,
    validation_size=0,
    test_size=1,
    purge_size=0,
    step_size=1,
)

predictions = predict_classification_windows(
    historical_rows=labeled_data,
    windows=windows,
    feature_columns=[
        "feature_a",
        "feature_b",
    ],
)

metrics_by_date = calculate_classification_metrics_by_date(predictions)
metrics_summary = summarize_classification_metrics(predictions)

required_prediction_columns = [
    "date",
    "ticker",
    "predicted_probability",
    "actual_label",
    "actual_return",
    "probability_rank",
    "selected_top_5",
]

positive_label_counts = labeled_data.groupby("date")[
    TOP_QUINTILE_LABEL_COLUMN
].sum().tolist()

required_columns_exist = all(
    column in predictions.columns
    for column in required_prediction_columns
)

positive_label_counts_correct = positive_label_counts == [
    1,
    1,
    1,
]

prediction_rows_correct = len(predictions) == 10
selected_rows_correct = int(predictions["selected_top_5"].sum()) == 10
metrics_dates_correct = len(metrics_by_date) == 2

print("Labeled data:")
print(
    labeled_data[
        [
            "date",
            "ticker",
            TARGET_COLUMN,
            TOP_QUINTILE_LABEL_COLUMN,
        ]
    ].to_string(index=False)
)

print("\nClassification predictions:")
print(predictions.round(6).to_string(index=False))

print("\nMetrics by date:")
print(metrics_by_date.round(6).to_string(index=False))

print("\nMetrics summary:")
print(metrics_summary.round(6).to_string(index=False))

print("\nRequired columns exist:", required_columns_exist)
print("Positive label counts correct:", positive_label_counts_correct)
print("Prediction rows correct:", prediction_rows_correct)
print("Selected rows correct:", selected_rows_correct)
print("Metrics dates correct:", metrics_dates_correct)
