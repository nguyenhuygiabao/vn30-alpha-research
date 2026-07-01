from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics import (
    summarize_baseline_returns,
    summarize_classification_predictions,
    summarize_predictions_by_model,
)
from src.tree_models import summarize_feature_importance


REPORT_PATH: str = "reports/model_report.md"

LINEAR_MODEL_PREDICTIONS_PATH: str = "data/processed/linear_model_predictions.parquet"
TREE_MODEL_PREDICTIONS_PATH: str = "data/processed/tree_model_predictions.parquet"
CLASSIFICATION_MODEL_PREDICTIONS_PATH: str = (
    "data/processed/classification_model_predictions.parquet"
)
BASELINE_RETURNS_PATH: str = "data/processed/baseline_returns.parquet"
TREE_FEATURE_IMPORTANCE_PATH: str = "data/processed/tree_feature_importance.parquet"


def read_parquet_if_exists(path: str) -> pd.DataFrame:
    parquet_path = Path(path)

    if not parquet_path.exists():
        return pd.DataFrame()

    return pd.read_parquet(parquet_path)


def format_dataframe_for_report(data: pd.DataFrame) -> str:
    if data.empty:
        return "_No data available._"

    return "```text\n" + data.round(6).to_string(index=False) + "\n```"


def load_regression_predictions() -> pd.DataFrame:
    prediction_frames = []

    for path in [
        LINEAR_MODEL_PREDICTIONS_PATH,
        TREE_MODEL_PREDICTIONS_PATH,
    ]:
        predictions = read_parquet_if_exists(path)

        if not predictions.empty:
            prediction_frames.append(predictions)

    if not prediction_frames:
        return pd.DataFrame()

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


def choose_best_regression_model(
    regression_summary: pd.DataFrame,
) -> str:
    if regression_summary.empty:
        return "No regression model can be selected because no predictions are available."

    best_row = regression_summary.sort_values(
        [
            "average_rank_ic",
            "average_after_cost_return",
            "average_top_minus_bottom_spread",
            "model_name",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).iloc[0]

    return (
        f"Best provisional regression model: {best_row['model_name']} "
        f"with average Rank IC {best_row['average_rank_ic']:.6f} "
        f"and average after-cost top-5 return "
        f"{best_row['average_after_cost_return']:.6f}."
    )


def build_model_report(
    top_n: int = 5,
    transaction_cost_rate: float = 0.001,
) -> str:
    regression_predictions = load_regression_predictions()
    classification_predictions = read_parquet_if_exists(
        CLASSIFICATION_MODEL_PREDICTIONS_PATH
    )
    baseline_returns = read_parquet_if_exists(BASELINE_RETURNS_PATH)
    tree_feature_importance = read_parquet_if_exists(TREE_FEATURE_IMPORTANCE_PATH)

    if regression_predictions.empty:
        regression_summary = pd.DataFrame()
    else:
        regression_summary = summarize_predictions_by_model(
            predictions=regression_predictions,
            top_n=top_n,
            transaction_cost_rate=transaction_cost_rate,
        )

    if classification_predictions.empty:
        classification_summary = pd.DataFrame()
    else:
        classification_summary = summarize_classification_predictions(
            predictions=classification_predictions,
        )

    if baseline_returns.empty:
        baseline_summary = pd.DataFrame()
    else:
        baseline_summary = summarize_baseline_returns(
            baseline_returns=baseline_returns,
        )

    if tree_feature_importance.empty:
        feature_importance_summary = pd.DataFrame()
    else:
        feature_importance_summary = summarize_feature_importance(
            feature_importance=tree_feature_importance,
            top_n=20,
        )

    best_model_text = choose_best_regression_model(regression_summary)

    report_sections = [
        "# VN30 Model Evaluation Report",
        "",
        "## Scope",
        "",
        (
            "This report compares model outputs using out-of-sample "
            "walk-forward prediction rows currently available in "
            "`data/processed/`."
        ),
        "",
        (
            "Important caution: the current dataset uses the current VN30 constituent "
            "list applied backward through time. The numbers below are real-data "
            "walk-forward results, but they are not final market evidence until "
            "survivorship bias, liquidity limits, and transaction costs are handled "
            "more carefully."
        ),
        "",
        "## Regression model comparison",
        "",
        (
            "Regression models are compared using Rank IC, top-5 hit rate, "
            "top-minus-bottom spread, turnover, transaction cost, "
            "after-cost return, Sharpe ratio, and max drawdown."
        ),
        "",
        format_dataframe_for_report(regression_summary),
        "",
        "## Classification model comparison",
        "",
        (
            "The logistic regression model is evaluated separately because "
            "it predicts top-quintile probability, not raw return."
        ),
        "",
        format_dataframe_for_report(classification_summary),
        "",
        "## Baseline strategy comparison",
        "",
        (
            "The baseline strategy is evaluated as a portfolio return series, "
            "not as a ranking model."
        ),
        "",
        format_dataframe_for_report(baseline_summary),
        "",
        "## Tree model feature importance",
        "",
        (
            "Feature importance shows which variables the tree models used "
            "most often for splits. It is useful for sanity checking, but it "
            "is not causal proof."
        ),
        "",
        format_dataframe_for_report(feature_importance_summary),
        "",
        "## Provisional model choice",
        "",
        best_model_text,
        "",
        (
            "The final model should be chosen using out-of-sample ranking "
            "quality and after-cost portfolio behavior, not in-sample fit. "
            "With real data, a model with slightly lower raw return but lower "
            "turnover may be preferable after transaction costs."
        ),
        "",
    ]

    return "\n".join(report_sections)


def save_model_report(
    report: str,
    output_path: str = REPORT_PATH,
) -> str:
    report_path = Path(output_path)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return str(report_path)


def main() -> None:
    report = build_model_report()
    report_path = save_model_report(report)

    print("Model report path:", report_path)
    print(report)


if __name__ == "__main__":
    main()
