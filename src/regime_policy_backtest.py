from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.model_candidates import (
    DEFAULT_REGIME_POLICY,
    build_historical_market_regimes,
    build_rank_ensemble_history,
    build_rolling_rank_ensemble_history,
    paired_block_bootstrap_mean_difference,
)


HISTORICAL_TREE_PREDICTION_HORIZON_DAYS = 5


def _turnover(previous: pd.Series, target: pd.Series) -> tuple[float, float, float]:
    tickers = previous.index.union(target.index)
    old = previous.reindex(tickers, fill_value=0.0)
    new = target.reindex(tickers, fill_value=0.0)
    buys = (new - old).clip(lower=0.0).sum()
    sells = (old - new).clip(upper=0.0).abs().sum()
    return float(0.5 * (new - old).abs().sum()), float(buys), float(sells)


def _apply_turnover_cap(
    previous: pd.Series,
    target: pd.Series,
    max_turnover: float,
) -> pd.Series:
    turnover, _, _ = _turnover(previous, target)
    if turnover <= max_turnover or turnover == 0.0:
        return target[target > 1e-12]

    tickers = previous.index.union(target.index)
    old = previous.reindex(tickers, fill_value=0.0)
    desired = target.reindex(tickers, fill_value=0.0)
    adjusted = old + (max_turnover / turnover) * (desired - old)
    return adjusted[adjusted > 1e-12]


def build_market_drawdown_overlay(
    market_data: pd.DataFrame,
    trigger_drawdown: float = -0.10,
    reduced_exposure: float = 0.50,
    normal_exposure: float = 0.97,
) -> pd.DataFrame:
    """Create an observable equal-weight market drawdown exposure overlay."""
    if not -1.0 < trigger_drawdown < 0.0:
        raise ValueError("trigger_drawdown must be between -1 and 0")
    if not 0.0 <= reduced_exposure <= normal_exposure <= 1.0:
        raise ValueError("Exposure inputs must satisfy 0 <= reduced <= normal <= 1")
    required = {"date", "ticker", "adjusted_close"}
    missing = sorted(required.difference(market_data.columns))
    if missing:
        raise ValueError(f"Market data are missing columns: {missing}")

    prices = market_data[["date", "ticker", "adjusted_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices["ticker"] = prices["ticker"].astype(str).str.strip().str.upper()
    prices = prices.sort_values(["ticker", "date"])
    prices["stock_return"] = prices.groupby("ticker")["adjusted_close"].pct_change()
    overlay = prices.groupby("date", as_index=False)["stock_return"].mean()
    overlay["market_return"] = overlay.pop("stock_return").fillna(0.0)
    overlay["market_nav"] = (1.0 + overlay["market_return"]).cumprod()
    overlay["market_drawdown"] = overlay["market_nav"] / overlay["market_nav"].cummax() - 1.0
    overlay["risk_off"] = overlay["market_drawdown"] <= trigger_drawdown
    overlay["target_exposure"] = np.where(
        overlay["risk_off"], reduced_exposure, normal_exposure
    )
    return overlay[
        ["date", "market_drawdown", "risk_off", "target_exposure"]
    ].reset_index(drop=True)


def build_non_overlapping_policy_returns(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    policy: Mapping[str, str] | None = None,
    top_n: int = 8,
    holding_period_days: int = HISTORICAL_TREE_PREDICTION_HORIZON_DAYS,
    settlement_lag_days: int = 2,
    target_exposure: float = 0.97,
    max_turnover: float = 0.25,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    sell_tax_rate: float = 0.001,
    target_exposure_by_date: pd.Series | None = None,
    signals_are_pre_spaced: bool = False,
    rolling_performance_window: int = 126,
    rolling_label_availability_lag: int = 10,
    realized_return_column: str = "actual_return",
) -> pd.DataFrame:
    """Backtest a fixed regime policy on non-overlapping forward-return labels.

    Rebalances are spaced by ``holding_period_days`` observed trading dates.  If
    that period is at least the settlement lag, proceeds from a prior rebalance
    would settle before the next rebalance; this is reported as a T+2-compatible
    schedule, not as a reconstruction of historical broker fills.
    """
    if top_n <= 0 or holding_period_days <= 0 or settlement_lag_days < 0:
        raise ValueError("top_n and holding_period_days must be positive")
    if not 0.0 <= target_exposure <= 1.0 or not 0.0 < max_turnover <= 1.0:
        raise ValueError("Exposure and turnover inputs are outside valid bounds")
    if realized_return_column not in predictions.columns:
        raise ValueError(
            f"Predictions lack realized return column: {realized_return_column}"
        )
    if target_exposure_by_date is not None:
        target_exposure_by_date = target_exposure_by_date.copy()
        target_exposure_by_date.index = pd.to_datetime(
            target_exposure_by_date.index,
            errors="raise",
        )
        if target_exposure_by_date.index.has_duplicates:
            raise ValueError("Exposure overlay has duplicate dates")
        if ((target_exposure_by_date < 0.0) | (target_exposure_by_date > 1.0)).any():
            raise ValueError("Exposure overlay values must be between 0 and 1")

    policy_map = dict(DEFAULT_REGIME_POLICY if policy is None else policy)
    expected_regimes = {"trend_up", "trend_down", "high_volatility"}
    missing = sorted(expected_regimes.difference(policy_map))
    if missing:
        raise ValueError(f"Regime policy is missing selections: {missing}")

    ensemble = build_rank_ensemble_history(predictions)
    rolling_ensemble = build_rolling_rank_ensemble_history(
        predictions,
        performance_window=rolling_performance_window,
        label_availability_lag=rolling_label_availability_lag,
    )
    combined = pd.concat(
        [predictions.copy(), ensemble, rolling_ensemble], ignore_index=True
    )
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()
    available_models = set(combined["model_name"])
    invalid = sorted(
        {
            model for model in policy_map.values()
            if model != "cash" and model not in available_models
        }
    )
    if invalid:
        raise ValueError(f"Regime policy selects unavailable models: {invalid}")

    regimes = build_historical_market_regimes(market_data)
    realized_returns = predictions[["date", "ticker", realized_return_column]].copy()
    realized_returns["date"] = pd.to_datetime(realized_returns["date"], errors="raise")
    realized_returns = realized_returns.drop_duplicates(["date", "ticker"])
    complete_actuals = (
        realized_returns.groupby(["date", "ticker"])[realized_return_column]
        .agg(lambda values: values.notna().all())
        .groupby(level="date")
        .all()
    )
    model_dates = complete_actuals.loc[complete_actuals].rename("complete_actuals")
    schedule = regimes.merge(
        model_dates,
        left_on="date",
        right_index=True,
        how="inner",
    ).sort_values("date")
    if schedule.empty:
        raise ValueError("No regime dates overlap available predictions")
    if signals_are_pre_spaced:
        market_dates = pd.DatetimeIndex(regimes["date"].drop_duplicates()).sort_values()
        market_positions = pd.Series(range(len(market_dates)), index=market_dates)
        schedule_positions = market_positions.reindex(schedule["date"]).to_numpy()
        if len(schedule_positions) > 1 and (
            np.diff(schedule_positions) < holding_period_days
        ).any():
            raise ValueError(
                "Pre-spaced prediction dates overlap the requested holding period"
            )
        schedule = schedule.copy()
    else:
        schedule = schedule.iloc[::holding_period_days].copy()
    schedule["selected_model"] = schedule["market_regime"].map(policy_map)

    previous = pd.Series(dtype=float)
    rows: list[dict[str, object]] = []

    for schedule_row in schedule.itertuples(index=False):
        date = schedule_row.date
        selected_model = schedule_row.selected_model
        date_target_exposure = target_exposure
        if target_exposure_by_date is not None:
            if date not in target_exposure_by_date.index:
                raise ValueError(f"Exposure overlay lacks date {date}")
            date_target_exposure = float(target_exposure_by_date.loc[date])
        daily_actual_returns = (
            realized_returns.loc[realized_returns["date"] == date]
            .set_index("ticker")[realized_return_column]
            .dropna()
        )
        unavailable_previous = previous.index.difference(daily_actual_returns.index)
        forced_exit_weight = float(previous.reindex(unavailable_previous).sum())
        carried_previous = previous.drop(unavailable_previous)

        if selected_model == "cash":
            target = pd.Series(dtype=float)
        else:
            daily = combined.loc[
                (combined["date"] == date)
                & (combined["model_name"] == selected_model)
            ].nlargest(top_n, "predicted_return")
            if len(daily) < top_n:
                raise ValueError(f"Insufficient predictions for {selected_model} on {date}")
            target = pd.Series(
                date_target_exposure / top_n,
                index=daily["ticker"],
                dtype=float,
            )

        if not target.index.isin(daily_actual_returns.index).all():
            raise ValueError(f"Selected targets lack realized returns on {date}")

        forced_exit_turnover = 0.5 * forced_exit_weight
        optional_turnover_capacity = max(max_turnover - forced_exit_turnover, 0.0)
        if optional_turnover_capacity > 0.0:
            weights = _apply_turnover_cap(
                carried_previous,
                target,
                max_turnover=optional_turnover_capacity,
            )
        else:
            weights = carried_previous
        turnover, buys, sells = _turnover(previous, weights)
        realized = daily_actual_returns.reindex(weights.index)
        if realized.isna().any():
            raise ValueError(f"Missing realized returns for selected holdings on {date}")
        before_cost = float((weights * realized.to_numpy()).sum())
        total_cost = (
            (buys + sells) * (commission_rate + slippage_rate)
            + sells * sell_tax_rate
        )
        rows.append(
            {
                "date": date,
                "market_regime": schedule_row.market_regime,
                "selected_model": selected_model,
                "before_cost_return": before_cost,
                "total_cost": total_cost,
                "after_cost_return": before_cost - total_cost,
                "portfolio_turnover": turnover,
                "total_weight": float(weights.sum()),
                "selected_count": len(weights),
                "target_exposure": date_target_exposure,
                "forced_exit_weight": forced_exit_weight,
                "turnover_cap_breached_by_forced_exit": (
                    turnover > max_turnover + 1e-12
                ),
                "settlement_compatible": holding_period_days >= settlement_lag_days,
            }
        )
        previous = weights

    history = pd.DataFrame(rows)
    history["cumulative_after_cost_return"] = (1.0 + history["after_cost_return"]).cumprod() - 1.0
    return history


def summarize_non_overlapping_policy_returns(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        raise ValueError("Policy backtest history is empty")
    ordered = history.sort_values("date").reset_index(drop=True)
    returns = ordered["after_cost_return"]
    volatility = returns.std()
    portfolio_nav = (1.0 + returns).cumprod()
    cumulative = portfolio_nav - 1.0
    drawdown = portfolio_nav / portfolio_nav.cummax().clip(lower=1.0) - 1.0
    summary = {
                "rebalance_dates": ordered["date"].nunique(),
                "average_after_cost_return": returns.mean(),
                "return_volatility": volatility,
                "diagnostic_sharpe": returns.mean() / volatility if volatility else np.nan,
                "average_turnover": ordered["portfolio_turnover"].mean(),
                "maximum_turnover": ordered["portfolio_turnover"].max(),
                "average_target_exposure": ordered["target_exposure"].mean(),
                "minimum_target_exposure": ordered["target_exposure"].min(),
                "forced_exit_dates": int((ordered["forced_exit_weight"] > 0.0).sum()),
                "maximum_forced_exit_weight": ordered["forced_exit_weight"].max(),
                "final_cumulative_after_cost_return": cumulative.iloc[-1],
                "max_after_cost_drawdown": drawdown.min(),
                "settlement_compatible": bool(ordered["settlement_compatible"].all()),
    }
    if "benchmark_return" in ordered.columns:
        benchmark_nav = (1.0 + ordered["benchmark_return"]).cumprod()
        active_nav = portfolio_nav / benchmark_nav
        active_drawdown = active_nav / active_nav.cummax().clip(lower=1.0) - 1.0
        summary.update(
            {
                "final_cumulative_benchmark_return": benchmark_nav.iloc[-1] - 1.0,
                "final_cumulative_active_return": active_nav.iloc[-1] - 1.0,
                "max_active_drawdown": active_drawdown.min(),
            }
        )
    return pd.DataFrame([summary])


def build_paired_overlay_returns(
    baseline_history: pd.DataFrame,
    overlay_history: pd.DataFrame,
) -> pd.DataFrame:
    """Align two non-overlapping histories for a like-for-like overlay test."""
    required = {"date", "after_cost_return"}
    for name, history in (("baseline", baseline_history), ("overlay", overlay_history)):
        missing = sorted(required.difference(history.columns))
        if missing:
            raise ValueError(f"{name} history is missing columns: {missing}")

    baseline = baseline_history[["date", "after_cost_return"]].rename(
        columns={"after_cost_return": "baseline_after_cost_return"}
    )
    overlay = overlay_history[["date", "after_cost_return"]].rename(
        columns={"after_cost_return": "overlay_after_cost_return"}
    )
    paired = baseline.merge(overlay, on="date", how="inner", validate="one_to_one")
    if len(paired) < 2:
        raise ValueError("At least two matched rebalance dates are required")
    paired["after_cost_return_difference"] = (
        paired["overlay_after_cost_return"] - paired["baseline_after_cost_return"]
    )
    return paired.sort_values("date").reset_index(drop=True)


def summarize_paired_overlay_stability(
    paired_returns: pd.DataFrame,
    block_size: int = 5,
    bootstrap_samples: int = 2_000,
) -> pd.DataFrame:
    """Summarize the overlay's paired return difference and block-bootstrap CI."""
    required = {"date", "after_cost_return_difference"}
    missing = sorted(required.difference(paired_returns.columns))
    if missing:
        raise ValueError(f"Paired returns are missing columns: {missing}")

    differences = paired_returns["after_cost_return_difference"]
    mean_difference, lower, upper = paired_block_bootstrap_mean_difference(
        differences,
        block_size=block_size,
        bootstrap_samples=bootstrap_samples,
    )
    years = pd.to_datetime(paired_returns["date"]).dt.year
    yearly = pd.DataFrame({"year": years, "difference": differences}).groupby("year")[
        "difference"
    ].mean()
    return pd.DataFrame(
        [
            {
                "paired_rebalance_dates": len(paired_returns),
                "mean_after_cost_return_difference": mean_difference,
                "bootstrap_95pct_lower": lower,
                "bootstrap_95pct_upper": upper,
                "positive_rebalance_dates": int((differences > 0.0).sum()),
                "positive_years": int((yearly > 0.0).sum()),
                "available_years": len(yearly),
            }
        ]
    )
