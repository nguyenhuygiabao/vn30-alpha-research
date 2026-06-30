import pandas as pd

from src.metrics import (
    calculate_max_drawdown,
    calculate_selection_turnover_by_date,
    calculate_sharpe_ratio,
    calculate_spearman_ic_by_date,
    calculate_top_minus_bottom_spread_by_date,
    calculate_top_n_hit_rate_by_date,
    calculate_top_n_portfolio_returns_by_date,
    summarize_baseline_returns,
    summarize_classification_predictions,
    summarize_predictions_by_model,
)
from src.model_report import build_model_report


predictions = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2024-01-01"] * 4
            + ["2024-01-02"] * 4
            + ["2024-01-01"] * 4
            + ["2024-01-02"] * 4
        ),
        "ticker": ["AAA", "BBB", "CCC", "DDD"] * 4,
        "predicted_return": [
            0.4,
            0.3,
            0.2,
            0.1,
            0.1,
            0.2,
            0.3,
            0.4,
            0.1,
            0.2,
            0.3,
            0.4,
            0.4,
            0.3,
            0.2,
            0.1,
        ],
        "actual_return": [
            0.04,
            0.03,
            -0.01,
            -0.02,
            -0.02,
            -0.01,
            0.03,
            0.04,
            0.04,
            0.03,
            -0.01,
            -0.02,
            -0.02,
            -0.01,
            0.03,
            0.04,
        ],
        "model_name": ["good_model"] * 8 + ["bad_model"] * 8,
    }
)

good_model_predictions = predictions[predictions["model_name"] == "good_model"].copy()

ic_by_date = calculate_spearman_ic_by_date(good_model_predictions)

hit_rate_by_date = calculate_top_n_hit_rate_by_date(
    predictions=good_model_predictions,
    top_n=2,
)

spread_by_date = calculate_top_minus_bottom_spread_by_date(
    predictions=good_model_predictions,
    top_n=2,
)

portfolio_returns = calculate_top_n_portfolio_returns_by_date(
    predictions=good_model_predictions,
    top_n=2,
    transaction_cost_rate=0.001,
)

selections = good_model_predictions.copy()
selections["selected"] = selections.groupby("date")["predicted_return"].rank(
    ascending=False,
    method="first",
) <= 2

turnover_by_date = calculate_selection_turnover_by_date(selections)

model_summary = summarize_predictions_by_model(
    predictions=predictions,
    top_n=2,
    transaction_cost_rate=0.001,
)

classification_predictions = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-01"] * 3),
        "ticker": ["AAA", "BBB", "CCC"],
        "predicted_probability": [0.8, 0.6, 0.2],
        "actual_label": [1, 0, 1],
        "actual_return": [0.04, 0.01, 0.03],
        "selected_top_5": [True, True, False],
    }
)

classification_summary = summarize_classification_predictions(
    classification_predictions
)

baseline_returns = pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
            ]
        ),
        "strategy": [
            "equal_weight_all",
            "equal_weight_all",
        ],
        "portfolio_forward_return_5d": [
            0.02,
            0.03,
        ],
        "vn30_forward_return_5d": [
            0.01,
            0.02,
        ],
        "active_return_vs_vn30_5d": [
            0.01,
            0.01,
        ],
        "selected_count": [
            3,
            3,
        ],
    }
)

baseline_summary = summarize_baseline_returns(baseline_returns)

report = build_model_report(
    top_n=5,
    transaction_cost_rate=0.001,
)

ic_correct = ic_by_date["rank_ic"].round(6).tolist() == [
    1.0,
    1.0,
]

hit_rate_correct = hit_rate_by_date["hit_rate"].round(6).tolist() == [
    1.0,
    1.0,
]

spread_correct = spread_by_date["top_minus_bottom_spread"].round(6).tolist() == [
    0.05,
    0.05,
]

turnover_correct = (
    pd.isna(turnover_by_date.loc[0, "turnover"])
    and turnover_by_date.loc[1, "turnover"] == 2.0
)

after_cost_correct = portfolio_returns["after_cost_return"].round(6).tolist() == [
    0.035,
    0.033,
]

model_order_correct = model_summary["model_name"].tolist() == [
    "good_model",
    "bad_model",
]

classification_correct = (
    classification_summary.loc[0, "average_precision"] == 0.5
    and classification_summary.loc[0, "average_recall"] == 0.5
)

baseline_correct = (
    baseline_summary.loc[0, "average_portfolio_return"].round(6) == 0.025
    and baseline_summary.loc[0, "average_active_return"].round(6) == 0.01
)

sharpe_available = not pd.isna(
    calculate_sharpe_ratio(pd.Series([0.01, 0.02, -0.01, 0.03]))
)

drawdown_correct = calculate_max_drawdown(
    pd.Series([0.01, -0.02, 0.03, -0.01])
).round(6) == -0.02

report_sections_correct = all(
    section in report
    for section in [
        "# VN30 Model Evaluation Report",
        "## Regression model comparison",
        "## Classification model comparison",
        "## Baseline strategy comparison",
        "## Tree model feature importance",
        "## Provisional model choice",
    ]
)

print("IC by date:")
print(ic_by_date.round(6).to_string(index=False))

print("\nHit rate by date:")
print(hit_rate_by_date.round(6).to_string(index=False))

print("\nSpread by date:")
print(spread_by_date.round(6).to_string(index=False))

print("\nPortfolio returns:")
print(portfolio_returns.round(6).to_string(index=False))

print("\nModel summary:")
print(model_summary.round(6).to_string(index=False))

print("\nClassification summary:")
print(classification_summary.round(6).to_string(index=False))

print("\nBaseline summary:")
print(baseline_summary.round(6).to_string(index=False))

print("\nIC correct:", ic_correct)
print("Hit rate correct:", hit_rate_correct)
print("Spread correct:", spread_correct)
print("Turnover correct:", turnover_correct)
print("After-cost correct:", after_cost_correct)
print("Model order correct:", model_order_correct)
print("Classification correct:", classification_correct)
print("Baseline correct:", baseline_correct)
print("Sharpe available:", sharpe_available)
print("Drawdown correct:", drawdown_correct)
print("Report sections correct:", report_sections_correct)
