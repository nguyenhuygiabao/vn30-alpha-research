"""Measure drawdown on a compounded five-tranche overlapping portfolio.

`src/backtester.py` publishes `max_active_drawdown` as a running-peak
difference on `cumulative_after_cost_active_return`, which is a plain `cumsum`
of per-date five-day forward active returns. Signal dates are daily and the
forecast horizon is five trading days, so every five-day period is counted
about five times over. The cumulative series is in summed-return units, not in
units of capital, and a drawdown read off it is not a percentage of anything.

Sampling every fifth date and compounding removes the double counting but
introduces a free parameter -- which of the five weekday phases the sampling
starts on. For the `normal` optimisation mode under price-limit-aware execution
that parameter alone moves final equity between 0.73 and 1.39.

This module removes the parameter rather than choosing a value for it. One
fifth of capital rebalances into each date's target weights and is held for the
five-day forecast horizon, so all five phases run at once -- which is what
daily signals with a five-day hold actually imply. The five phase sleeves
collapse into one daily return series, one compounded equity curve, and one
peak-to-trough drawdown expressed as a fraction of capital.

Within a holding period the tranche is buy-and-hold: weights drift with prices
and are not maintained. That is deliberate, because it reconciles exactly with
the backtester's arithmetic -- a buy-and-hold sleeve holding `w` over the
horizon ends the period at `1 + sum(w * horizon_return)`, which is the same sum
the backtester forms.

It is a diagnostic. It reads existing artefacts, writes no production output,
and changes no published number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtester import (
    COMMISSION_RATE,
    LIQUIDITY_PENALTY_RATE,
    SLIPPAGE_RATE,
)

HOLD_HORIZON_DAYS: int = 5
TRANCHE_COUNT: int = HOLD_HORIZON_DAYS
TRADING_DAYS_PER_YEAR: int = 252

KEY_COLUMNS: list[str] = [
    "date",
    "ticker",
]


def build_daily_total_returns(
    market_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-ticker daily returns from adjusted close.

    Adjusted prices drive returns; raw prices drive execution and price limits.
    This is the return leg, so it reads `adjusted_close` only.
    """
    required_columns = [
        "date",
        "ticker",
        "adjusted_close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in market_history.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing market history columns: {missing_columns}")

    daily = (
        market_history[required_columns]
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
        .copy()
    )

    daily["daily_return"] = (
        daily.groupby("ticker")["adjusted_close"].pct_change()
    )

    return daily[
        [
            "date",
            "ticker",
            "daily_return",
        ]
    ]


def build_equal_weight_benchmark_weights(
    universe: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build equal-weight target weights over the point-in-time constituents.

    `src/labels.py` defines the benchmark leg of every published "active"
    number as the leave-one-out equal-weight mean of the point-in-time VN30
    constituents over the same forward horizon. The canonical history carries no
    VN30 index price series, so the portfolio-level analogue is an equal-weight
    basket of those constituents.

    The weights are returned rather than a return series so the benchmark can be
    run through the same tranche simulation as the portfolio. That matters: an
    equal-weight basket rebalanced daily is not the same thing as one rebalanced
    on the forecast horizon, and comparing a five-day tranche portfolio against
    a daily-rebalanced basket would credit or charge the strategy for a
    rebalancing frequency it never had.
    """
    missing_columns = [
        column
        for column in KEY_COLUMNS
        if column not in universe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing universe columns: {missing_columns}")

    eligible = (
        universe[KEY_COLUMNS]
        .drop_duplicates()
        .copy()
    )

    eligible = eligible[eligible["date"].isin(rebalance_dates)]

    constituent_count = (
        eligible.groupby("date")["ticker"]
        .transform("size")
    )

    if (constituent_count == 0).any():
        raise ValueError("A rebalance date has no point-in-time constituents")

    eligible["weight"] = 1.0 / constituent_count

    missing_dates = pd.DatetimeIndex(rebalance_dates).difference(
        pd.DatetimeIndex(eligible["date"].unique())
    )

    if len(missing_dates) > 0:
        raise ValueError(
            f"{len(missing_dates)} rebalance dates have no point-in-time "
            "constituents"
        )

    return eligible.reset_index(drop=True)


def calculate_tranche_turnover(
    weights: pd.DataFrame,
    horizon: int = HOLD_HORIZON_DAYS,
) -> pd.DataFrame:
    """Turnover of each tranche against the tranche it replaces.

    The published `portfolio_turnover` compares each date's target weights with
    the previous *date's*. A tranche is replaced by the tranche `horizon` dates
    later, so the turnover that a tranche portfolio actually pays for is the
    change over `horizon` dates, not over one.

    The comparison is against the previous target weights rather than against
    the drifted holding, which is the convention `src/backtester.py` uses.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    weight_matrix = (
        weights.pivot_table(
            index="date",
            columns="ticker",
            values="weight",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )

    previous_matrix = weight_matrix.shift(horizon).fillna(0.0)

    turnover = (
        0.5 * (weight_matrix - previous_matrix).abs().sum(axis=1)
    ).rename("tranche_turnover")

    return turnover.reset_index()


def calculate_tranche_costs(
    weights: pd.DataFrame,
    liquidity: pd.DataFrame,
    horizon: int = HOLD_HORIZON_DAYS,
) -> pd.DataFrame:
    """Rebalance cost per tranche, using the backtester's cost model.

    Commission, slippage and the low-liquidity penalty use the rates and the
    functional form in `src/backtester.py`. The only substitution is the
    turnover input: a tranche's turnover is measured over `horizon` dates
    because that is how often a tranche is actually traded.

    The cost is expressed as a fraction of the tranche, not of total capital.
    """
    turnover = calculate_tranche_turnover(
        weights=weights,
        horizon=horizon,
    )

    working = weights.merge(
        liquidity,
        on=KEY_COLUMNS,
        how="left",
    )

    working["is_low_liquidity"] = working["is_low_liquidity"].eq(True)

    working["low_liquidity_weight_component"] = working["weight"].where(
        working["is_low_liquidity"],
        0.0,
    )

    low_liquidity_weight = (
        working.groupby("date")["low_liquidity_weight_component"]
        .sum()
        .rename("low_liquidity_weight")
        .reset_index()
    )

    costs = turnover.merge(
        low_liquidity_weight,
        on="date",
        how="left",
    )

    costs["low_liquidity_weight"] = costs["low_liquidity_weight"].fillna(0.0)

    costs["commission_cost"] = costs["tranche_turnover"] * COMMISSION_RATE
    costs["slippage_cost"] = costs["tranche_turnover"] * SLIPPAGE_RATE
    costs["liquidity_penalty"] = (
        costs["tranche_turnover"]
        * costs["low_liquidity_weight"]
        * LIQUIDITY_PENALTY_RATE
    )

    costs["tranche_cost"] = (
        costs["commission_cost"]
        + costs["slippage_cost"]
        + costs["liquidity_penalty"]
    )

    return costs


def build_tranche_values(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    horizon: int = HOLD_HORIZON_DAYS,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Value each tranche on every day of its holding period.

    Returns the return calendar, the rebalance dates, the calendar positions of
    those dates, and a `(rebalance dates, horizon + 1)` array of tranche values
    normalised to 1.0 on the rebalance date. Column `h` is the tranche's value
    after `h` trading days of buy-and-hold.

    Nothing in this function reads a price on or before the rebalance date other
    than through the weights it is handed, and column `h` reads only the `h`
    trading dates that follow the rebalance date.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    required_weight_columns = [
        "date",
        "ticker",
        "weight",
    ]

    missing_columns = [
        column
        for column in required_weight_columns
        if column not in weights.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing weight columns: {missing_columns}")

    calendar = pd.DatetimeIndex(sorted(pd.unique(daily_returns["date"])))
    calendar_position = pd.Series(
        np.arange(len(calendar)),
        index=calendar,
    )

    weight_matrix = (
        weights.pivot_table(
            index="date",
            columns="ticker",
            values="weight",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )

    rebalance_dates = pd.DatetimeIndex(weight_matrix.index)

    unknown_dates = rebalance_dates.difference(calendar)

    if len(unknown_dates) > 0:
        raise ValueError(
            f"{len(unknown_dates)} rebalance dates are absent from the "
            "return calendar"
        )

    positions = calendar_position.loc[rebalance_dates].to_numpy()

    if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
        raise ValueError(
            "Rebalance dates are not a contiguous block of the return calendar"
        )

    if len(rebalance_dates) <= horizon:
        raise ValueError(
            f"At least {horizon + 1} rebalance dates are required, "
            f"received {len(rebalance_dates)}"
        )

    if positions[-1] + horizon > len(calendar) - 1:
        raise ValueError(
            "The return calendar ends before the final tranche is unwound; "
            f"{horizon} trading dates after {rebalance_dates[-1].date()} "
            "are required"
        )

    gross_matrix = (
        daily_returns.pivot_table(
            index="date",
            columns="ticker",
            values="daily_return",
            aggfunc="sum",
        )
        .sort_index()
        .reindex(calendar)
    )

    held_tickers = pd.Index(weight_matrix.columns)
    unpriced_tickers = held_tickers.difference(gross_matrix.columns)

    if len(unpriced_tickers) > 0:
        raise ValueError(
            f"Held tickers absent from the return history: "
            f"{sorted(unpriced_tickers)}"
        )

    gross_matrix = gross_matrix.reindex(columns=held_tickers)
    gross = (1.0 + gross_matrix.to_numpy(dtype=float))
    gross = np.nan_to_num(gross, nan=1.0)

    held_weights = weight_matrix.to_numpy(dtype=float)
    cash_weight = 1.0 - held_weights.sum(axis=1)

    growth = np.ones_like(held_weights)
    tranche_value = np.ones((len(rebalance_dates), horizon + 1))

    for holding_day in range(1, horizon + 1):
        growth = growth * gross[positions + holding_day, :]
        tranche_value[:, holding_day] = (
            cash_weight + (held_weights * growth).sum(axis=1)
        )

    return calendar, rebalance_dates, positions, tranche_value


def calculate_tranche_horizon_returns(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    horizon: int = HOLD_HORIZON_DAYS,
) -> pd.DataFrame:
    """Absolute buy-and-hold return of each tranche over its holding period.

    This is the reconciliation handle. A buy-and-hold tranche ends its period at
    `1 + sum(weight * horizon_return)`, so this column must equal the sum the
    backtester forms from `actual_return` once the benchmark leg is added back.
    """
    _, rebalance_dates, _, tranche_value = build_tranche_values(
        weights=weights,
        daily_returns=daily_returns,
        horizon=horizon,
    )

    return pd.DataFrame(
        {
            "date": rebalance_dates,
            "horizon_return": tranche_value[:, horizon] - 1.0,
        }
    )


def simulate_tranche_portfolio(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    costs: pd.DataFrame | None = None,
    horizon: int = HOLD_HORIZON_DAYS,
) -> pd.DataFrame:
    """Run the overlapping tranche simulation for one weight set.

    Returns one row per trading date on which all `horizon` phase sleeves are
    live, with the before-cost and after-cost daily return of each phase sleeve
    and of the blended portfolio.

    The portfolio is the equal-weight, daily-reallocated blend of the phase
    sleeves: each phase sleeve is a standalone portfolio that rebalances every
    `horizon` dates, and the portfolio holds one `horizon`-th of capital in
    each. That is the same thing as saying one fifth of capital rebalances into
    each date's targets.
    """
    calendar, rebalance_dates, positions, tranche_value = build_tranche_values(
        weights=weights,
        daily_returns=daily_returns,
        horizon=horizon,
    )

    tranche_daily_return = (
        tranche_value[:, 1:] / tranche_value[:, :-1] - 1.0
    )

    phase_of_tranche = np.arange(len(rebalance_dates)) % horizon

    phase_return = np.full((len(calendar), horizon), np.nan)

    for holding_day in range(1, horizon + 1):
        phase_return[
            positions + holding_day,
            phase_of_tranche,
        ] = tranche_daily_return[:, holding_day - 1]

    phase_cost = np.zeros((len(calendar), horizon))

    if costs is not None:
        cost_by_date = (
            costs.set_index("date")["tranche_cost"]
            .reindex(rebalance_dates)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        phase_cost[positions, phase_of_tranche] = cost_by_date

    live = ~np.isnan(phase_return).any(axis=1)

    before_cost = phase_return[live, :]
    after_cost = before_cost - phase_cost[live, :]

    simulation = pd.DataFrame(
        {
            "date": calendar[live],
            "before_cost_return": before_cost.mean(axis=1),
            "after_cost_return": after_cost.mean(axis=1),
            "tranche_cost": phase_cost[live, :].sum(axis=1) / horizon,
        }
    )

    for phase in range(horizon):
        simulation[f"phase_{phase}_before_cost_return"] = before_cost[:, phase]
        simulation[f"phase_{phase}_after_cost_return"] = after_cost[:, phase]

    return simulation.reset_index(drop=True)


def summarize_equity_curve(
    returns: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compound a daily return series and describe the resulting equity curve."""
    clean_returns = returns.dropna()

    if clean_returns.empty:
        raise ValueError("Cannot summarize an empty return series")

    equity = (1.0 + clean_returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0

    periods = len(clean_returns)
    years = periods / trading_days_per_year
    final_equity = float(equity.iloc[-1])

    if final_equity > 0 and years > 0:
        annualised_return = final_equity ** (1.0 / years) - 1.0
    else:
        annualised_return = float("nan")

    return {
        "periods": periods,
        "years": years,
        "final_equity": final_equity,
        "max_drawdown": float(drawdown.min()),
        "annualised_return": float(annualised_return),
        "annualised_volatility": float(
            clean_returns.std(ddof=1) * np.sqrt(trading_days_per_year)
        ),
    }


def build_equity_curves(
    simulation: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizon: int = HOLD_HORIZON_DAYS,
) -> pd.DataFrame:
    """Attach the benchmark and the active return to a tranche simulation."""
    curves = simulation.merge(
        benchmark,
        on="date",
        how="left",
    )

    missing_benchmark = curves["benchmark_return"].isna().sum()

    if missing_benchmark > 0:
        raise ValueError(
            f"Missing benchmark returns on {missing_benchmark} dates"
        )

    curves["active_before_cost_return"] = (
        curves["before_cost_return"] - curves["benchmark_return"]
    )

    curves["active_after_cost_return"] = (
        curves["after_cost_return"] - curves["benchmark_return"]
    )

    for phase in range(horizon):
        curves[f"phase_{phase}_active_after_cost_return"] = (
            curves[f"phase_{phase}_after_cost_return"]
            - curves["benchmark_return"]
        )

    return curves


def summarize_phase_dispersion(
    curves: pd.DataFrame,
    return_column_template: str,
    horizon: int = HOLD_HORIZON_DAYS,
) -> dict[str, float]:
    """Compare the single-phase curves with the blended tranche curve.

    A single-phase curve is what you get by sampling every `horizon`-th date and
    compounding: a real portfolio, but one whose identity depends on an
    arbitrary starting offset. The blend has no such offset. This quantifies how
    much the offset is worth so the claim can be checked rather than asserted.
    """
    phase_finals = []
    phase_drawdowns = []

    for phase in range(horizon):
        phase_summary = summarize_equity_curve(
            curves[return_column_template.format(phase=phase)]
        )
        phase_finals.append(phase_summary["final_equity"])
        phase_drawdowns.append(phase_summary["max_drawdown"])

    return {
        "phase_final_equity_min": float(min(phase_finals)),
        "phase_final_equity_max": float(max(phase_finals)),
        "phase_final_equity_spread": float(max(phase_finals) - min(phase_finals)),
        "phase_final_equity_log_dispersion": float(
            np.std(np.log(phase_finals), ddof=1)
        ),
        "phase_max_drawdown_min": float(min(phase_drawdowns)),
        "phase_max_drawdown_max": float(max(phase_drawdowns)),
        "phase_max_drawdown_spread": float(
            max(phase_drawdowns) - min(phase_drawdowns)
        ),
    }


def summarize_start_offset_sensitivity(
    returns: pd.Series,
    horizon: int = HOLD_HORIZON_DAYS,
) -> dict[str, float]:
    """Restart the blended curve on each of `horizon` consecutive dates.

    The single-phase curves have a starting offset that changes which periods
    they hold. The blended curve has no offset in that sense -- every phase is
    always held -- so shifting the first date only trims a few days off an
    otherwise identical return series. Measuring the residual spread is the
    honest version of the claim.
    """
    finals = []
    drawdowns = []

    for offset in range(horizon):
        summary = summarize_equity_curve(returns.iloc[offset:])
        finals.append(summary["final_equity"])
        drawdowns.append(summary["max_drawdown"])

    return {
        "offset_final_equity_min": float(min(finals)),
        "offset_final_equity_max": float(max(finals)),
        "offset_final_equity_spread": float(max(finals) - min(finals)),
        "offset_final_equity_log_dispersion": float(
            np.std(np.log(finals), ddof=1)
        ),
        "offset_max_drawdown_min": float(min(drawdowns)),
        "offset_max_drawdown_max": float(max(drawdowns)),
        "offset_max_drawdown_spread": float(max(drawdowns) - min(drawdowns)),
    }
