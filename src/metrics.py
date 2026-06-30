from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_spearman_ic_by_date(
    predictions: pd.DataFrame,
    prediction_column: str = "predicted_return",
    actual_return_column: str = "actual_return",
) -> pd.DataFrame:
    ic_rows = []

    for date, date_rows in predictions.groupby("date"):
        valid_rows = date_rows[
            [
                prediction_column,
                actual_return_column,
            ]
        ].dropna()

        if len(valid_rows) < 2:
            rank_ic = np.nan
        else:
            rank_ic = valid_rows[prediction_column].corr(
                valid_rows[actual_return_column],
                method="spearman",
            )

        ic_rows.append(
            {
                "date": date,
                "rank_ic": rank_ic,
                "stock_count": len(valid_rows),
            }
        )

    return pd.DataFrame(ic_rows)


def calculate_top_n_hit_rate_by_date(
    predictions: pd.DataFrame,
    prediction_column: str = "predicted_return",
    actual_return_column: str = "actual_return",
    top_n: int = 5,
) -> pd.DataFrame:
    hit_rate_rows = []

    for date, date_rows in predictions.groupby("date"):
        valid_rows = date_rows[
            [
                "ticker",
                prediction_column,
                actual_return_column,
            ]
        ].dropna()

        selected_rows = valid_rows.sort_values(
            [
                prediction_column,
                "ticker",
            ],
            ascending=[
                False,
                True,
            ],
        ).head(top_n)

        if selected_rows.empty:
            hit_rate = np.nan
            average_selected_return = np.nan
        else:
            hit_rate = (selected_rows[actual_return_column] > 0).mean()
            average_selected_return = selected_rows[actual_return_column].mean()

        hit_rate_rows.append(
            {
                "date": date,
                "selected_count": len(selected_rows),
                "hit_rate": hit_rate,
                "average_selected_return": average_selected_return,
            }
        )

    return pd.DataFrame(hit_rate_rows)


def calculate_top_minus_bottom_spread_by_date(
    predictions: pd.DataFrame,
    prediction_column: str = "predicted_return",
    actual_return_column: str = "actual_return",
    top_n: int = 5,
) -> pd.DataFrame:
    spread_rows = []

    for date, date_rows in predictions.groupby("date"):
        valid_rows = date_rows[
            [
                "ticker",
                prediction_column,
                actual_return_column,
            ]
        ].dropna()

        top_rows = valid_rows.sort_values(
            [
                prediction_column,
                "ticker",
            ],
            ascending=[
                False,
                True,
            ],
        ).head(top_n)

        bottom_rows = valid_rows.sort_values(
            [
                prediction_column,
                "ticker",
            ],
            ascending=[
                True,
                True,
            ],
        ).head(top_n)

        if top_rows.empty or bottom_rows.empty:
            top_return = np.nan
            bottom_return = np.nan
            top_minus_bottom_spread = np.nan
        else:
            top_return = top_rows[actual_return_column].mean()
            bottom_return = bottom_rows[actual_return_column].mean()
            top_minus_bottom_spread = top_return - bottom_return

        spread_rows.append(
            {
                "date": date,
                "top_return": top_return,
                "bottom_return": bottom_return,
                "top_minus_bottom_spread": top_minus_bottom_spread,
            }
        )

    return pd.DataFrame(spread_rows)


def calculate_selection_turnover_by_date(
    selections: pd.DataFrame,
    selected_column: str = "selected",
) -> pd.DataFrame:
    turnover_rows = []

    selected_by_date = {
        date: set(date_rows.loc[date_rows[selected_column], "ticker"])
        for date, date_rows in selections.groupby("date")
    }

    sorted_dates = sorted(selected_by_date)

    previous_selection = None

    for date in sorted_dates:
        current_selection = selected_by_date[date]

        if previous_selection is None:
            turnover = np.nan
        elif not current_selection and not previous_selection:
            turnover = 0.0
        else:
            removed_names = previous_selection - current_selection
            added_names = current_selection - previous_selection
            turnover = (
                len(removed_names) + len(added_names)
            ) / max(
                len(previous_selection),
                len(current_selection),
                1,
            )

        turnover_rows.append(
            {
                "date": date,
                "selected_count": len(current_selection),
                "turnover": turnover,
            }
        )

        previous_selection = current_selection

    return pd.DataFrame(turnover_rows)


def build_top_n_selection_frame(
    predictions: pd.DataFrame,
    prediction_column: str = "predicted_return",
    top_n: int = 5,
) -> pd.DataFrame:
    ranked = predictions.copy()

    ranked["prediction_rank"] = ranked.groupby("date")[prediction_column].rank(
        ascending=False,
        method="first",
    )

    ranked["selected"] = ranked["prediction_rank"] <= top_n

    return ranked


def calculate_top_n_portfolio_returns_by_date(
    predictions: pd.DataFrame,
    prediction_column: str = "predicted_return",
    actual_return_column: str = "actual_return",
    top_n: int = 5,
    transaction_cost_rate: float = 0.001,
) -> pd.DataFrame:
    selections = build_top_n_selection_frame(
        predictions=predictions,
        prediction_column=prediction_column,
        top_n=top_n,
    )

    turnover_by_date = calculate_selection_turnover_by_date(
        selections=selections,
        selected_column="selected",
    )

    selected_rows = selections[selections["selected"]].copy()

    gross_returns = selected_rows.groupby(
        "date",
        as_index=False,
    ).agg(
        gross_top_n_return=(actual_return_column, "mean"),
        selected_count=("ticker", "count"),
    )

    portfolio_returns = gross_returns.merge(
        turnover_by_date[
            [
                "date",
                "turnover",
            ]
        ],
        on="date",
        how="left",
    )

    portfolio_returns["turnover_for_cost"] = portfolio_returns["turnover"].fillna(
        0.0
    )

    portfolio_returns["transaction_cost"] = (
        portfolio_returns["turnover_for_cost"] * transaction_cost_rate
    )

    portfolio_returns["after_cost_return"] = (
        portfolio_returns["gross_top_n_return"]
        - portfolio_returns["transaction_cost"]
    )

    return portfolio_returns.drop(
        columns=[
            "turnover_for_cost",
        ]
    )


def summarize_portfolio_returns(
    portfolio_returns: pd.DataFrame,
    return_column: str = "after_cost_return",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if portfolio_returns.empty:
        return pd.DataFrame(
            [
                {
                    "evaluated_dates": 0,
                    "average_return": np.nan,
                    "sharpe_ratio": np.nan,
                    "max_drawdown": np.nan,
                    "average_turnover": np.nan,
                    "average_transaction_cost": np.nan,
                }
            ]
        )

    returns = portfolio_returns[return_column]

    return pd.DataFrame(
        [
            {
                "evaluated_dates": len(portfolio_returns),
                "average_return": returns.mean(),
                "sharpe_ratio": calculate_sharpe_ratio(
                    returns=returns,
                    periods_per_year=periods_per_year,
                ),
                "max_drawdown": calculate_max_drawdown(returns),
                "average_turnover": portfolio_returns["turnover"].mean(),
                "average_transaction_cost": portfolio_returns[
                    "transaction_cost"
                ].mean(),
            }
        ]
    )


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return np.nan

    volatility = clean_returns.std(ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return clean_returns.mean() / volatility * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    returns: pd.Series,
) -> float:
    clean_returns = returns.fillna(0.0)

    if clean_returns.empty:
        return np.nan

    cumulative_returns = (1.0 + clean_returns).cumprod()
    running_peak = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_peak - 1.0

    return drawdown.min()


def summarize_predictions_by_model(
    predictions: pd.DataFrame,
    prediction_column: str = "predicted_return",
    actual_return_column: str = "actual_return",
    top_n: int = 5,
    transaction_cost_rate: float = 0.001,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    summary_rows = []

    for model_name, model_predictions in predictions.groupby("model_name"):
        ic_by_date = calculate_spearman_ic_by_date(
            predictions=model_predictions,
            prediction_column=prediction_column,
            actual_return_column=actual_return_column,
        )

        hit_rate_by_date = calculate_top_n_hit_rate_by_date(
            predictions=model_predictions,
            prediction_column=prediction_column,
            actual_return_column=actual_return_column,
            top_n=top_n,
        )

        spread_by_date = calculate_top_minus_bottom_spread_by_date(
            predictions=model_predictions,
            prediction_column=prediction_column,
            actual_return_column=actual_return_column,
            top_n=top_n,
        )

        portfolio_returns = calculate_top_n_portfolio_returns_by_date(
            predictions=model_predictions,
            prediction_column=prediction_column,
            actual_return_column=actual_return_column,
            top_n=top_n,
            transaction_cost_rate=transaction_cost_rate,
        )

        portfolio_summary = summarize_portfolio_returns(
            portfolio_returns=portfolio_returns,
            return_column="after_cost_return",
            periods_per_year=periods_per_year,
        ).iloc[0]

        summary_rows.append(
            {
                "model_name": model_name,
                "evaluated_dates": len(ic_by_date),
                "average_rank_ic": ic_by_date["rank_ic"].mean(),
                "average_hit_rate": hit_rate_by_date["hit_rate"].mean(),
                "average_top_minus_bottom_spread": spread_by_date[
                    "top_minus_bottom_spread"
                ].mean(),
                "average_gross_top_n_return": portfolio_returns[
                    "gross_top_n_return"
                ].mean(),
                "average_after_cost_return": portfolio_summary[
                    "average_return"
                ],
                "sharpe_ratio": portfolio_summary["sharpe_ratio"],
                "max_drawdown": portfolio_summary["max_drawdown"],
                "average_turnover": portfolio_summary["average_turnover"],
                "average_transaction_cost": portfolio_summary[
                    "average_transaction_cost"
                ],
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
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
    ).reset_index(drop=True)


def summarize_classification_predictions(
    predictions: pd.DataFrame,
    probability_column: str = "predicted_probability",
    actual_label_column: str = "actual_label",
    actual_return_column: str = "actual_return",
    selected_column: str = "selected_top_5",
    model_name: str = "logistic_regression",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "evaluated_dates",
                "average_precision",
                "average_recall",
                "average_selected_return",
                "sharpe_ratio",
                "max_drawdown",
            ]
        )

    rows = []

    for date, date_rows in predictions.groupby("date"):
        selected_rows = date_rows[date_rows[selected_column]].copy()

        actual_positive_count = int(date_rows[actual_label_column].sum())
        true_positive_count = int(selected_rows[actual_label_column].sum())
        selected_count = len(selected_rows)

        if selected_count == 0:
            precision = np.nan
            average_selected_return = np.nan
        else:
            precision = true_positive_count / selected_count
            average_selected_return = selected_rows[actual_return_column].mean()

        if actual_positive_count == 0:
            recall = np.nan
        else:
            recall = true_positive_count / actual_positive_count

        rows.append(
            {
                "date": date,
                "precision": precision,
                "recall": recall,
                "selected_count": selected_count,
                "average_selected_return": average_selected_return,
            }
        )

    by_date = pd.DataFrame(rows)
    returns = by_date["average_selected_return"]

    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "evaluated_dates": len(by_date),
                "average_precision": by_date["precision"].mean(),
                "average_recall": by_date["recall"].mean(),
                "average_selected_return": returns.mean(),
                "sharpe_ratio": calculate_sharpe_ratio(
                    returns=returns,
                    periods_per_year=periods_per_year,
                ),
                "max_drawdown": calculate_max_drawdown(returns),
            }
        ]
    )


def summarize_baseline_returns(
    baseline_returns: pd.DataFrame,
    return_column: str = "portfolio_forward_return_5d",
    active_return_column: str = "active_return_vs_vn30_5d",
    strategy_column: str = "strategy",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    summary_rows = []

    for strategy, strategy_rows in baseline_returns.groupby(strategy_column):
        returns = strategy_rows[return_column]
        active_returns = strategy_rows[active_return_column]

        summary_rows.append(
            {
                "strategy": strategy,
                "evaluated_dates": len(strategy_rows),
                "average_portfolio_return": returns.mean(),
                "average_active_return": active_returns.mean(),
                "sharpe_ratio": calculate_sharpe_ratio(
                    returns=returns,
                    periods_per_year=periods_per_year,
                ),
                "max_drawdown": calculate_max_drawdown(returns),
                "average_selected_count": strategy_rows[
                    "selected_count"
                ].mean(),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
        [
            "average_active_return",
            "average_portfolio_return",
            "strategy",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)
