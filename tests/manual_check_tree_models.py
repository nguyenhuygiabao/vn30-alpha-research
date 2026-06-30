import pandas as pd

from src.linear_models import summarize_model_performance
from src.tree_models import (
    compare_tree_models_with_linear_models,
    create_tree_models,
    predict_tree_models_for_windows,
    summarize_feature_importance,
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

dates = get_sorted_dates(data)

windows = build_walk_forward_date_windows(
    dates=dates,
    train_size=1,
    validation_size=0,
    test_size=1,
    purge_size=0,
    step_size=1,
)

models = create_tree_models()

predictions, feature_importance = predict_tree_models_for_windows(
    models=models,
    historical_rows=data,
    windows=windows,
    feature_columns=[
        "feature_a",
        "feature_b",
    ],
)

performance_summary = summarize_model_performance(
    predictions=predictions,
    top_n=5,
)

feature_importance_summary = summarize_feature_importance(
    feature_importance=feature_importance,
    top_n=2,
)

model_comparison = compare_tree_models_with_linear_models(
    tree_predictions=predictions,
    linear_predictions_path="missing_file.parquet",
    top_n=5,
)

required_prediction_columns = [
    "date",
    "ticker",
    "predicted_return",
    "actual_return",
    "model_name",
]

required_importance_columns = [
    "model_name",
    "feature",
    "importance",
    "test_start_date",
    "test_end_date",
]

models_correct = sorted(predictions["model_name"].unique().tolist()) == [
    "gradient_boosting",
    "random_forest",
]

prediction_rows_correct = len(predictions) == 20
importance_rows_correct = len(feature_importance) == 8
prediction_columns_correct = all(
    column in predictions.columns
    for column in required_prediction_columns
)
importance_columns_correct = all(
    column in feature_importance.columns
    for column in required_importance_columns
)
performance_rows_correct = len(performance_summary) == 2
comparison_rows_correct = len(model_comparison) == 2

print("Tree predictions:")
print(predictions.round(6).to_string(index=False))

print("\nFeature importance:")
print(feature_importance.round(6).to_string(index=False))

print("\nPerformance summary:")
print(performance_summary.round(6).to_string(index=False))

print("\nFeature importance summary:")
print(feature_importance_summary.round(6).to_string(index=False))

print("\nModel comparison:")
print(model_comparison.round(6).to_string(index=False))

print("\nModels correct:", models_correct)
print("Prediction rows correct:", prediction_rows_correct)
print("Importance rows correct:", importance_rows_correct)
print("Prediction columns correct:", prediction_columns_correct)
print("Importance columns correct:", importance_columns_correct)
print("Performance rows correct:", performance_rows_correct)
print("Comparison rows correct:", comparison_rows_correct)
