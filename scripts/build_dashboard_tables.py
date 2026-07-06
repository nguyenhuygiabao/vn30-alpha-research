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
HORIZON_RESULTS_PATH = TABLES_DIR / "horizon_results.csv"

BENCHMARK_RESULTS_PATH = TABLES_DIR / "benchmark_results.csv"
CONCENTRATION_SUMMARY_PATH = TABLES_DIR / "concentration_summary.csv"
ISSUER_GROUP_EXPOSURE_PATH = TABLES_DIR / "issuer_group_exposure_latest.csv"
LATEST_RANK_DIAGNOSTIC_PATH = TABLES_DIR / "latest_rank_diagnostic.csv"
HORIZON_DISCLOSURE_PATH = TABLES_DIR / "horizon_sample_disclosure.csv"


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


def write_table(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def main() -> None:
    outputs = {
        BENCHMARK_RESULTS_PATH: build_benchmark_results(),
        CONCENTRATION_SUMMARY_PATH: build_concentration_summary(),
        ISSUER_GROUP_EXPOSURE_PATH: build_latest_issuer_group_exposure(),
        LATEST_RANK_DIAGNOSTIC_PATH: build_latest_rank_diagnostic(),
        HORIZON_DISCLOSURE_PATH: build_horizon_sample_disclosure(),
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
