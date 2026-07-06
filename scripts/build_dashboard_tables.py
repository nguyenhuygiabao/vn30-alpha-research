from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "reports" / "tables"

BASELINE_RETURNS_PATH = DATA_DIR / "baseline_returns.parquet"
BACKTEST_RETURNS_PATH = DATA_DIR / "backtest_returns.parquet"
OPTIMIZED_WEIGHTS_PATH = DATA_DIR / "optimized_weights.parquet"
TREE_PREDICTIONS_PATH = DATA_DIR / "tree_model_predictions.parquet"
LINEAR_PREDICTIONS_PATH = DATA_DIR / "linear_model_predictions.parquet"
CLASSIFICATION_PREDICTIONS_PATH = DATA_DIR / "classification_model_predictions.parquet"
HORIZON_RESULTS_PATH = TABLES_DIR / "horizon_results.csv"

BENCHMARK_RESULTS_PATH = TABLES_DIR / "benchmark_results.csv"
CONCENTRATION_SUMMARY_PATH = TABLES_DIR / "concentration_summary.csv"
ISSUER_GROUP_EXPOSURE_PATH = TABLES_DIR / "issuer_group_exposure_latest.csv"
LATEST_RANK_DIAGNOSTIC_PATH = TABLES_DIR / "latest_rank_diagnostic.csv"
HORIZON_DISCLOSURE_PATH = TABLES_DIR / "horizon_sample_disclosure.csv"
MODEL_COMPARISON_PATH = TABLES_DIR / "model_comparison_results.csv"
OPTIMIZER_BOUND_DIAGNOSTIC_PATH = TABLES_DIR / "optimizer_bound_diagnostic.csv"

MAX_SINGLE_NAME_WEIGHT = 0.20
MAX_ISSUER_GROUP_WEIGHT = 0.40
TOLERANCE = 1e-8


def read_required_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path.relative_to(ROOT)}")

    data = pd.read_parquet(path)

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])

    return data


def diagnostic_sharpe(returns: pd.Series) -> float:
    volatility = returns.std()

    if pd.isna(volatility) or volatility <= 1e-12:
        return float("nan")

    return returns.mean() / volatility


def max_drawdown_from_period_returns(returns: pd.Series) -> float:
    cumulative = returns.cumsum()
    drawdown = cumulative - cumulative.cummax()

    return drawdown.min()


def build_benchmark_results() -> pd.DataFrame:
    baseline = read_required_parquet(BASELINE_RETURNS_PATH)
    backtest = read_required_parquet(BACKTEST_RETURNS_PATH)

    common_dates = set(backtest["date"].dropna().unique())
    baseline = baseline[baseline["date"].isin(common_dates)].copy()

    rows: list[dict[str, object]] = []

    for keys, group in backtest.groupby(["optimization_mode", "execution_mode"]):
        optimization_mode, execution_mode = keys
        ordered = group.sort_values("date").copy()
        period_returns = ordered["after_cost_return"]

        rows.append(
            {
                "comparison_type": "ml_strategy",
                "strategy": f"ml_{optimization_mode}_{execution_mode}",
                "display_name": f"ML {optimization_mode} / {execution_mode}",
                "forecast_horizon_days": 5,
                "evaluated_dates": ordered["date"].nunique(),
                "average_period_active_return": period_returns.mean(),
                "return_volatility": period_returns.std(),
                "diagnostic_sharpe": diagnostic_sharpe(period_returns),
                "max_active_drawdown": max_drawdown_from_period_returns(period_returns),
                "final_cumulative_active_return": period_returns.cumsum().iloc[-1],
                "average_selected_count": ordered["selected_count"].mean(),
                "return_basis": "after-cost active return per 5-day forecast period",
                "cost_note": "Includes estimated commission, slippage, and liquidity penalty.",
            }
        )

    for strategy, group in baseline.groupby("strategy"):
        ordered = group.sort_values("date").copy()
        period_returns = ordered["active_return_vs_vn30_5d"]

        rows.append(
            {
                "comparison_type": "naive_baseline",
                "strategy": strategy,
                "display_name": strategy.replace("_", " "),
                "forecast_horizon_days": 5,
                "evaluated_dates": ordered["date"].nunique(),
                "average_period_active_return": period_returns.mean(),
                "return_volatility": period_returns.std(),
                "diagnostic_sharpe": diagnostic_sharpe(period_returns),
                "max_active_drawdown": max_drawdown_from_period_returns(period_returns),
                "final_cumulative_active_return": period_returns.cumsum().iloc[-1],
                "average_selected_count": ordered["selected_count"].mean(),
                "return_basis": "before-cost active return versus VN30-style reference per 5-day forecast period",
                "cost_note": "No transaction-cost adjustment applied to this naive baseline.",
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["comparison_type", "diagnostic_sharpe", "final_cumulative_active_return"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return result


def build_concentration_summary() -> pd.DataFrame:
    weights = read_required_parquet(OPTIMIZED_WEIGHTS_PATH)

    latest_date = weights["date"].max()
    latest = weights[weights["date"].eq(latest_date)].copy()

    rows: list[dict[str, object]] = []

    for optimization_mode, group in latest.groupby("optimization_mode"):
        group = group.copy()
        issuer_exposure = (
            group.groupby("issuer_group")["weight"]
            .sum()
            .sort_values(ascending=False)
        )

        top_issuer_group = issuer_exposure.index[0]
        top_issuer_group_weight = issuer_exposure.iloc[0]
        top_issuer_tickers = sorted(
            group[group["issuer_group"].eq(top_issuer_group)]["ticker"].tolist()
        )

        max_single_name_weight = group["weight"].max()
        hhi = (group["weight"] ** 2).sum()

        rows.append(
            {
                "signal_date": latest_date.strftime("%Y-%m-%d"),
                "optimization_mode": optimization_mode,
                "holding_count": len(group),
                "total_weight": group["weight"].sum(),
                "max_single_name_weight": max_single_name_weight,
                "positions_at_max_single_name_weight": int(
                    group["weight"].sub(max_single_name_weight).abs().le(1e-8).sum()
                ),
                "positions_at_or_above_20pct": int(group["weight"].ge(0.20 - 1e-8).sum()),
                "hhi": hhi,
                "effective_position_count": 1.0 / hhi if hhi > 0 else float("nan"),
                "top_issuer_group": top_issuer_group,
                "top_issuer_group_weight": top_issuer_group_weight,
                "top_issuer_group_tickers": ", ".join(top_issuer_tickers),
                "issuer_groups_at_or_above_40pct": int(
                    issuer_exposure.ge(0.40 - 1e-8).sum()
                ),
                "portfolio_turnover": group["portfolio_turnover"].max(),
            }
        )

    return pd.DataFrame(rows).sort_values("optimization_mode").reset_index(drop=True)


def build_latest_issuer_group_exposure() -> pd.DataFrame:
    weights = read_required_parquet(OPTIMIZED_WEIGHTS_PATH)

    latest_date = weights["date"].max()
    latest = weights[weights["date"].eq(latest_date)].copy()

    rows: list[dict[str, object]] = []

    for keys, group in latest.groupby(["optimization_mode", "issuer_group"]):
        optimization_mode, issuer_group = keys
        ordered = group.sort_values("weight", ascending=False).copy()
        exposure = ordered["weight"].sum()

        rows.append(
            {
                "signal_date": latest_date.strftime("%Y-%m-%d"),
                "optimization_mode": optimization_mode,
                "issuer_group": issuer_group,
                "issuer_group_weight": exposure,
                "position_count": len(ordered),
                "tickers": ", ".join(ordered["ticker"].tolist()),
                "max_single_name_weight_in_group": ordered["weight"].max(),
                "weighted_realized_forward_return": (
                    (ordered["weight"] * ordered["actual_return"]).sum() / exposure
                    if exposure > 0
                    else float("nan")
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["optimization_mode", "issuer_group_weight"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_latest_rank_diagnostic() -> pd.DataFrame:
    predictions = read_required_parquet(TREE_PREDICTIONS_PATH)
    weights = read_required_parquet(OPTIMIZED_WEIGHTS_PATH)

    latest_date = predictions["date"].max()
    latest = predictions[
        predictions["date"].eq(latest_date)
        & predictions["model_name"].eq("gradient_boosting")
    ].copy()

    if latest.empty:
        latest = predictions[predictions["date"].eq(latest_date)].copy()

    latest = latest.sort_values(["predicted_return", "ticker"], ascending=[False, True])
    latest["predicted_rank"] = range(1, len(latest) + 1)

    latest = latest.sort_values(["actual_return", "ticker"], ascending=[False, True])
    latest["realized_rank"] = range(1, len(latest) + 1)

    latest["rank_gap_realized_minus_predicted"] = (
        latest["realized_rank"] - latest["predicted_rank"]
    )
    latest["absolute_rank_gap"] = latest["rank_gap_realized_minus_predicted"].abs()

    latest_weights = weights[weights["date"].eq(weights["date"].max())].copy()

    for optimization_mode in sorted(latest_weights["optimization_mode"].dropna().unique()):
        selected = set(
            latest_weights[
                latest_weights["optimization_mode"].eq(optimization_mode)
            ]["ticker"]
        )
        latest[f"in_{optimization_mode}_portfolio"] = latest["ticker"].isin(selected)

    latest["diagnostic_flag"] = "normal"
    latest.loc[
        latest["predicted_rank"].le(5) & latest["realized_rank"].gt(15),
        "diagnostic_flag",
    ] = "top_ranked_miss"
    latest.loc[
        latest["predicted_rank"].le(5) & latest["realized_rank"].le(5),
        "diagnostic_flag",
    ] = "top_ranked_hit"
    latest.loc[
        latest["predicted_rank"].le(5) & latest["actual_return"].lt(0),
        "diagnostic_flag",
    ] = "top_ranked_negative_return"

    latest = latest.sort_values("predicted_rank").rename(
        columns={
            "date": "signal_date",
            "predicted_return": "model_score",
            "actual_return": "realized_forward_return",
        }
    )

    latest["signal_date"] = pd.to_datetime(latest["signal_date"]).dt.strftime("%Y-%m-%d")

    columns = [
        "signal_date",
        "ticker",
        "model_name",
        "predicted_rank",
        "realized_rank",
        "rank_gap_realized_minus_predicted",
        "absolute_rank_gap",
        "model_score",
        "realized_forward_return",
        "diagnostic_flag",
    ]

    portfolio_columns = [
        column
        for column in latest.columns
        if column.startswith("in_") and column.endswith("_portfolio")
    ]

    return latest[columns + portfolio_columns].reset_index(drop=True)


def build_horizon_sample_disclosure() -> pd.DataFrame:
    if not HORIZON_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {HORIZON_RESULTS_PATH.relative_to(ROOT)}")

    horizon = pd.read_csv(HORIZON_RESULTS_PATH).copy()

    horizon["period_label"] = (
        horizon["forecast_horizon_days"].astype(int).astype(str)
        + "d forecast period"
    )

    horizon["approx_non_overlapping_periods"] = (
        horizon["evaluated_dates"] // horizon["forecast_horizon_days"]
    ).astype(int)

    horizon["overlap_disclosure"] = horizon.apply(
        lambda row: (
            f'{int(row["evaluated_dates"]):,} evaluated dates '
            f'(~{int(row["approx_non_overlapping_periods"]):,} non-overlapping '
            f'{int(row["forecast_horizon_days"])}-day periods).'
        ),
        axis=1,
    )

    horizon["metric_period_note"] = (
        "Average after-cost return is measured per forecast period, not annualized."
    )

    return horizon[
        [
            "forecast_horizon_days",
            "period_label",
            "evaluated_dates",
            "approx_non_overlapping_periods",
            "overlap_disclosure",
            "metric_period_note",
        ]
    ].copy()


def build_enhanced_benchmark_results() -> pd.DataFrame:
    baseline = read_required_parquet(BASELINE_RETURNS_PATH)
    backtest = read_required_parquet(BACKTEST_RETURNS_PATH)

    common_dates = set(backtest["date"].dropna().unique())
    baseline = baseline[baseline["date"].isin(common_dates)].copy()

    rows: list[dict[str, object]] = []

    for keys, group in backtest.groupby(["optimization_mode", "execution_mode"]):
        optimization_mode, execution_mode = keys
        ordered = group.sort_values("date").copy()
        period_returns = ordered["after_cost_return"]

        rows.append(
            {
                "comparison_type": "ml_strategy",
                "strategy": f"ml_{optimization_mode}_{execution_mode}",
                "display_name": f"ML {optimization_mode} / {execution_mode}",
                "forecast_horizon_days": 5,
                "evaluated_dates": ordered["date"].nunique(),
                "average_period_active_return": period_returns.mean(),
                "average_return_period_label": "per 5-day forecast period",
                "return_volatility": period_returns.std(),
                "diagnostic_sharpe": diagnostic_sharpe(period_returns),
                "max_active_drawdown": max_drawdown_from_period_returns(period_returns),
                "final_cumulative_active_return_sum": period_returns.cumsum().iloc[-1],
                "final_cumulative_active_return_note": (
                    "Sum of overlapping forecast-period active returns, not a compounded portfolio return."
                ),
                "average_selected_count": ordered["selected_count"].mean(),
                "return_basis": "after-cost active return per 5-day forecast period",
                "cost_note": "Includes estimated commission, slippage, and liquidity penalty.",
            }
        )

    for strategy, group in baseline.groupby("strategy"):
        ordered = group.sort_values("date").copy()
        period_returns = ordered["active_return_vs_vn30_5d"]

        rows.append(
            {
                "comparison_type": "naive_baseline",
                "strategy": strategy,
                "display_name": strategy.replace("_", " "),
                "forecast_horizon_days": 5,
                "evaluated_dates": ordered["date"].nunique(),
                "average_period_active_return": period_returns.mean(),
                "average_return_period_label": "per 5-day forecast period",
                "return_volatility": period_returns.std(),
                "diagnostic_sharpe": diagnostic_sharpe(period_returns),
                "max_active_drawdown": max_drawdown_from_period_returns(period_returns),
                "final_cumulative_active_return_sum": period_returns.cumsum().iloc[-1],
                "final_cumulative_active_return_note": (
                    "Sum of overlapping forecast-period active returns, not a compounded portfolio return."
                ),
                "average_selected_count": ordered["selected_count"].mean(),
                "return_basis": "before-cost active return versus VN30-style reference per 5-day forecast period",
                "cost_note": "No transaction-cost adjustment applied to this naive baseline.",
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "comparison_type",
            "diagnostic_sharpe",
            "final_cumulative_active_return_sum",
        ],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_daily_concentration(weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for keys, group in weights.groupby(["optimization_mode", "date"]):
        optimization_mode, date = keys
        group = group.copy()

        issuer_exposure = group.groupby("issuer_group")["weight"].sum()
        top_issuer_group = issuer_exposure.idxmax()
        top_issuer_group_weight = issuer_exposure.max()
        hhi = (group["weight"] ** 2).sum()

        rows.append(
            {
                "optimization_mode": optimization_mode,
                "date": date,
                "holding_count": len(group),
                "total_weight": group["weight"].sum(),
                "max_single_name_weight": group["weight"].max(),
                "positions_at_or_above_20pct": int(
                    group["weight"].ge(MAX_SINGLE_NAME_WEIGHT - TOLERANCE).sum()
                ),
                "single_name_cap_hit_share": (
                    group["weight"].ge(MAX_SINGLE_NAME_WEIGHT - TOLERANCE).mean()
                ),
                "all_positions_at_single_name_cap": bool(
                    group["weight"].ge(MAX_SINGLE_NAME_WEIGHT - TOLERANCE).all()
                ),
                "hhi": hhi,
                "effective_position_count": 1.0 / hhi if hhi > 0 else float("nan"),
                "top_issuer_group": top_issuer_group,
                "top_issuer_group_weight": top_issuer_group_weight,
                "issuer_groups_at_or_above_40pct": int(
                    issuer_exposure.ge(MAX_ISSUER_GROUP_WEIGHT - TOLERANCE).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_concentration_flag(row: pd.Series) -> str:
    flags = []

    if int(row["positions_at_or_above_20pct"]) == int(row["holding_count"]):
        flags.append("all_positions_at_20pct_cap")

    if row["top_issuer_group_weight"] >= MAX_ISSUER_GROUP_WEIGHT - TOLERANCE:
        flags.append("issuer_group_at_40pct_cap")

    if row["effective_position_count"] <= 5.0 + TOLERANCE:
        flags.append("low_effective_position_count")

    if flags:
        return "; ".join(flags)

    return "normal"


def build_enhanced_concentration_summary() -> pd.DataFrame:
    weights = read_required_parquet(OPTIMIZED_WEIGHTS_PATH)
    concentration = build_daily_concentration(weights)

    latest_date = concentration["date"].max()
    latest = concentration[concentration["date"].eq(latest_date)].copy()
    latest["signal_date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest["concentration_flag"] = latest.apply(build_concentration_flag, axis=1)

    latest_weights = weights[weights["date"].eq(latest_date)].copy()
    ticker_map = (
        latest_weights.groupby(["optimization_mode", "issuer_group"])["ticker"]
        .apply(lambda values: ", ".join(sorted(values)))
        .reset_index()
        .rename(columns={"ticker": "top_issuer_group_tickers"})
    )

    latest = latest.merge(
        ticker_map,
        left_on=["optimization_mode", "top_issuer_group"],
        right_on=["optimization_mode", "issuer_group"],
        how="left",
    )

    latest = latest.drop(columns=["date", "issuer_group"])

    columns = [
        "signal_date",
        "optimization_mode",
        "holding_count",
        "total_weight",
        "max_single_name_weight",
        "positions_at_or_above_20pct",
        "single_name_cap_hit_share",
        "hhi",
        "effective_position_count",
        "top_issuer_group",
        "top_issuer_group_weight",
        "top_issuer_group_tickers",
        "issuer_groups_at_or_above_40pct",
        "concentration_flag",
    ]

    return latest[columns].sort_values("optimization_mode").reset_index(drop=True)


def build_optimizer_bound_diagnostic() -> pd.DataFrame:
    weights = read_required_parquet(OPTIMIZED_WEIGHTS_PATH)
    concentration = build_daily_concentration(weights)

    latest_date = concentration["date"].max()
    rows: list[dict[str, object]] = []

    for optimization_mode, group in concentration.groupby("optimization_mode"):
        ordered = group.sort_values("date").copy()
        latest = ordered[ordered["date"].eq(latest_date)].iloc[0]

        rows.append(
            {
                "signal_date": latest_date.strftime("%Y-%m-%d"),
                "optimization_mode": optimization_mode,
                "latest_holding_count": int(latest["holding_count"]),
                "latest_positions_at_20pct_cap": int(latest["positions_at_or_above_20pct"]),
                "latest_single_name_cap_hit_share": latest["single_name_cap_hit_share"],
                "latest_all_positions_at_cap": bool(latest["all_positions_at_single_name_cap"]),
                "mean_single_name_cap_hit_share": ordered["single_name_cap_hit_share"].mean(),
                "share_of_dates_all_positions_at_cap": ordered[
                    "all_positions_at_single_name_cap"
                ].mean(),
                "max_single_name_cap_hit_share": ordered["single_name_cap_hit_share"].max(),
                "mean_hhi": ordered["hhi"].mean(),
                "latest_hhi": latest["hhi"],
                "mean_effective_position_count": ordered["effective_position_count"].mean(),
                "latest_effective_position_count": latest["effective_position_count"],
                "diagnostic_note": (
                    "High cap-hit share means the optimizer is often constrained by the 20% single-name limit."
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("optimization_mode").reset_index(drop=True)


def build_enhanced_latest_issuer_group_exposure() -> pd.DataFrame:
    weights = read_required_parquet(OPTIMIZED_WEIGHTS_PATH)

    latest_date = weights["date"].max()
    latest = weights[weights["date"].eq(latest_date)].copy()

    rows: list[dict[str, object]] = []

    for keys, group in latest.groupby(["optimization_mode", "issuer_group"]):
        optimization_mode, issuer_group = keys
        ordered = group.sort_values("weight", ascending=False).copy()
        exposure = ordered["weight"].sum()

        rows.append(
            {
                "signal_date": latest_date.strftime("%Y-%m-%d"),
                "optimization_mode": optimization_mode,
                "issuer_group": issuer_group,
                "issuer_group_weight": exposure,
                "position_count": len(ordered),
                "tickers": ", ".join(ordered["ticker"].tolist()),
                "max_single_name_weight_in_group": ordered["weight"].max(),
                "weighted_realized_forward_return": (
                    (ordered["weight"] * ordered["actual_return"]).sum() / exposure
                    if exposure > 0
                    else float("nan")
                ),
                "exposure_flag": (
                    "issuer_group_at_40pct_cap"
                    if exposure >= MAX_ISSUER_GROUP_WEIGHT - TOLERANCE
                    else "normal"
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["optimization_mode", "issuer_group_weight"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_enhanced_horizon_sample_disclosure() -> pd.DataFrame:
    if not HORIZON_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing required file: {HORIZON_RESULTS_PATH.relative_to(ROOT)}"
        )

    horizon = pd.read_csv(HORIZON_RESULTS_PATH).copy()

    horizon["period_label"] = (
        horizon["forecast_horizon_days"].astype(int).astype(str)
        + "d forecast period"
    )

    horizon["approx_non_overlapping_periods"] = (
        horizon["evaluated_dates"] // horizon["forecast_horizon_days"]
    ).astype(int)

    horizon["overlap_share_estimate"] = 1.0 - (
        horizon["approx_non_overlapping_periods"] / horizon["evaluated_dates"]
    )

    horizon["overlap_disclosure"] = horizon.apply(
        lambda row: (
            f'{int(row["evaluated_dates"]):,} evaluated dates '
            f'(~{int(row["approx_non_overlapping_periods"]):,} non-overlapping '
            f'{int(row["forecast_horizon_days"])}-day periods).'
        ),
        axis=1,
    )

    horizon["metric_period_note"] = (
        "Average after-cost return is measured per forecast period, not annualized."
    )

    return horizon[
        [
            "forecast_horizon_days",
            "period_label",
            "evaluated_dates",
            "approx_non_overlapping_periods",
            "overlap_share_estimate",
            "overlap_disclosure",
            "metric_period_note",
        ]
    ].copy()


def first_existing_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in data.columns:
            return column

    return None


def normalize_prediction_source(path: Path, model_family: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    data = read_required_parquet(path)

    score_column = first_existing_column(
        data,
        [
            "predicted_return",
            "predicted_probability",
            "positive_class_probability",
            "probability",
            "score",
        ],
    )
    actual_column = first_existing_column(
        data,
        [
            "actual_return",
            "forward_relative_return_5d",
            "target_return",
            "target",
        ],
    )

    if score_column is None or actual_column is None:
        return pd.DataFrame()

    output = data[["date", "ticker", score_column, actual_column]].copy()
    output = output.rename(
        columns={
            score_column: "model_score",
            actual_column: "realized_forward_return",
        }
    )

    if "model_name" in data.columns:
        output["model_name"] = data["model_name"].astype(str)
    else:
        output["model_name"] = model_family

    output["model_family"] = model_family
    output = output.dropna(
        subset=[
            "date",
            "ticker",
            "model_score",
            "realized_forward_return",
        ]
    )

    return output


def summarize_model_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for keys, group in predictions.groupby(["model_family", "model_name"]):
        model_family, model_name = keys
        daily_rows = []

        for date, date_data in group.groupby("date"):
            date_data = date_data.copy()

            if len(date_data) < 5:
                continue

            date_data = date_data.sort_values(
                ["model_score", "ticker"],
                ascending=[False, True],
            )
            realized = date_data.sort_values(
                ["realized_forward_return", "ticker"],
                ascending=[False, True],
            )

            realized_top5 = set(realized.head(5)["ticker"])
            predicted_top5 = date_data.head(5).copy()

            if (
                date_data["model_score"].nunique() > 1
                and date_data["realized_forward_return"].nunique() > 1
            ):
                rank_ic = date_data["model_score"].corr(
                    date_data["realized_forward_return"],
                    method="spearman",
                )
            else:
                rank_ic = float("nan")

            daily_rows.append(
                {
                    "date": date,
                    "rank_ic": rank_ic,
                    "top5_hit_rate": predicted_top5["ticker"].isin(realized_top5).mean(),
                    "top5_equal_weight_realized_return": predicted_top5[
                        "realized_forward_return"
                    ].mean(),
                    "selected_count": len(predicted_top5),
                }
            )

        daily = pd.DataFrame(daily_rows)

        if daily.empty:
            continue

        returns = daily["top5_equal_weight_realized_return"]

        rows.append(
            {
                "model_family": model_family,
                "model_name": model_name,
                "evaluated_dates": len(daily),
                "average_rank_ic": daily["rank_ic"].mean(),
                "average_top5_hit_rate": daily["top5_hit_rate"].mean(),
                "average_top5_realized_return_per_5d_period": returns.mean(),
                "return_volatility": returns.std(),
                "diagnostic_sharpe": diagnostic_sharpe(returns),
                "max_drawdown_from_top5_return_sum": max_drawdown_from_period_returns(returns),
                "final_cumulative_top5_return_sum": returns.cumsum().iloc[-1],
                "average_selected_count": daily["selected_count"].mean(),
                "comparison_note": (
                    "Equal-weight top-5 by model score; diagnostic only, not optimizer/backtest execution."
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["diagnostic_sharpe", "average_rank_ic"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_model_comparison_results() -> pd.DataFrame:
    frames = [
        normalize_prediction_source(TREE_PREDICTIONS_PATH, "tree"),
        normalize_prediction_source(LINEAR_PREDICTIONS_PATH, "linear"),
        normalize_prediction_source(CLASSIFICATION_PREDICTIONS_PATH, "classification"),
    ]

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(
            columns=[
                "model_family",
                "model_name",
                "evaluated_dates",
                "average_rank_ic",
                "average_top5_hit_rate",
                "average_top5_realized_return_per_5d_period",
                "return_volatility",
                "diagnostic_sharpe",
                "max_drawdown_from_top5_return_sum",
                "final_cumulative_top5_return_sum",
                "average_selected_count",
                "comparison_note",
            ]
        )

    predictions = pd.concat(frames, ignore_index=True)

    return summarize_model_predictions(predictions)


def write_table(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def main() -> None:
    outputs = {
        BENCHMARK_RESULTS_PATH: build_enhanced_benchmark_results(),
        CONCENTRATION_SUMMARY_PATH: build_enhanced_concentration_summary(),
        ISSUER_GROUP_EXPOSURE_PATH: build_enhanced_latest_issuer_group_exposure(),
        LATEST_RANK_DIAGNOSTIC_PATH: build_latest_rank_diagnostic(),
        HORIZON_DISCLOSURE_PATH: build_enhanced_horizon_sample_disclosure(),
        MODEL_COMPARISON_PATH: build_model_comparison_results(),
        OPTIMIZER_BOUND_DIAGNOSTIC_PATH: build_optimizer_bound_diagnostic(),
    }

    for path, data in outputs.items():
        write_table(data, path)
        print(f"Wrote {path.relative_to(ROOT)} rows={len(data)}")

    print()
    print("Benchmark comparison:")
    print(outputs[BENCHMARK_RESULTS_PATH].to_string(index=False))

    print()
    print("Concentration summary:")
    print(outputs[CONCENTRATION_SUMMARY_PATH].to_string(index=False))

    print()
    print("Latest rank diagnostic, top 10 predicted:")
    print(outputs[LATEST_RANK_DIAGNOSTIC_PATH].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
