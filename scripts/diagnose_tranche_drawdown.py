"""Report drawdown on a compounded five-tranche overlapping portfolio.

The published `max_active_drawdown` in `reports/` is a running-peak difference
on `cumulative_after_cost_active_return`, a plain `cumsum` of per-date five-day
forward active returns. Because signal dates are daily and the horizon is five
trading days, every period is counted about five times over and the series is
in summed-return units, not in units of capital.

This script builds the alternative: one fifth of capital rebalances into each
date's target weights and is held for five trading days, so all five phases are
held at once. That collapses to a single daily return series, a single
compounded equity curve, and a drawdown that is a genuine fraction of capital.

It also prints the five single-phase curves, which are what you get by sampling
every fifth date, so the size of the arbitrary starting-offset parameter that
the tranche construction removes can be read rather than taken on trust.

It is a diagnostic. It reads existing artefacts, writes
`reports/tables/tranche_portfolio_diagnostic.csv` and, beside it, the daily
price-limit-aware series those summaries compound
(`reports/tables/tranche_daily_returns.csv`), and changes no published number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtester import (
    BACKTEST_RETURNS_PATH,
    OPTIMIZED_WEIGHTS_PATH,
    apply_price_limit_execution_rules,
    load_execution_reference,
    load_liquidity_features,
    load_optimized_weights,
)
from src.feature_pipeline import (
    ADJUSTED_MARKET_PATH,
    MEMBERSHIP_HISTORY_DIR,
    RAW_MARKET_PATH,
    load_verified_membership_history,
)
from src.market_data_schema import load_canonical_market_history
from src.tranche_portfolio import (
    HOLD_HORIZON_DAYS,
    build_daily_total_returns,
    build_equal_weight_benchmark_weights,
    build_equity_curves,
    calculate_tranche_costs,
    calculate_tranche_turnover,
    calculate_tranche_horizon_returns,
    simulate_tranche_portfolio,
    summarize_equity_curve,
    summarize_phase_dispersion,
    summarize_start_offset_sensitivity,
)
from src.universe_history import filter_to_point_in_time_universe

OUTPUT_PATH: str = "reports/tables/tranche_portfolio_diagnostic.csv"
DAILY_OUTPUT_PATH: str = "reports/tables/tranche_daily_returns.csv"

FRICTIONLESS_EXECUTION: str = "frictionless"
PRICE_LIMIT_EXECUTION: str = "price_limit_aware_tranche"

PHASE_SEPARATOR: str = "__phase_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compound a five-tranche overlapping portfolio and report its "
            "drawdown as a percentage of capital."
        ),
    )
    parser.add_argument("--weights-path", default=OPTIMIZED_WEIGHTS_PATH)
    parser.add_argument("--backtest-returns-path", default=BACKTEST_RETURNS_PATH)
    parser.add_argument("--raw-market-path", default=RAW_MARKET_PATH)
    parser.add_argument("--adjusted-market-path", default=ADJUSTED_MARKET_PATH)
    parser.add_argument("--membership-dir", default=MEMBERSHIP_HISTORY_DIR)
    parser.add_argument("--horizon", type=int, default=HOLD_HORIZON_DAYS)
    parser.add_argument(
        "--skip-price-limit",
        action="store_true",
        help=(
            "Report only the frictionless execution case. The price-limit "
            "reconstruction chains the production execution rules phase by "
            "phase and is the slow part of this script."
        ),
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--daily-output",
        default=DAILY_OUTPUT_PATH,
        help=(
            "Where to write the daily price-limit-aware series the summary "
            "compounds. Not written under --skip-price-limit."
        ),
    )
    return parser.parse_args()


def build_phase_partitioned_weights(
    weights: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Relabel each date's weights with the phase sleeve that trades them.

    `apply_price_limit_execution_rules` chains dates within an
    `optimization_mode` group, carrying the previous date's executed weights
    forward as the starting book. A tranche's starting book is not yesterday's
    portfolio but the tranche it replaces, `horizon` dates earlier. Splitting
    each mode into `horizon` phase groups makes the production function chain
    over exactly those dates, so the blocking rules run unmodified against the
    right prior holding.
    """
    partitioned = weights.copy()

    date_rank = (
        pd.Series(
            np.arange(weights["date"].nunique()),
            index=pd.DatetimeIndex(sorted(pd.unique(weights["date"]))),
        )
        % horizon
    )

    partitioned["optimization_mode"] = (
        partitioned["optimization_mode"]
        + PHASE_SEPARATOR
        + partitioned["date"].map(date_rank).astype(str)
    )

    return partitioned


def strip_phase_labels(weights: pd.DataFrame) -> pd.DataFrame:
    restored = weights.copy()

    restored["optimization_mode"] = (
        restored["optimization_mode"]
        .str.rsplit(PHASE_SEPARATOR, n=1)
        .str[0]
    )

    return restored


def reconcile_against_backtester(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    labels: pd.DataFrame,
    backtest_returns: pd.DataFrame,
    optimization_mode: str,
    horizon: int,
) -> dict[str, float]:
    """Check the absolute return leg against the published relative one.

    `actual_return` in the pipeline is `forward_relative_return_5d`: a stock's
    five-day forward return minus the leave-one-out equal-weight mean of its
    point-in-time peers. So the backtester's `before_cost_return` is already an
    active number and there is no absolute return anywhere in
    `backtest_returns.parquet`.

    The tranche simulation is built on absolute prices, so the two only line up
    once the benchmark leg is added back:

        buy-and-hold horizon return == sum(weight * actual_return)
                                       + sum(weight * peer benchmark return)

    The residual is not expected to be exactly zero. The optimizer's turnover
    cap strands a holding whenever a name leaves the point-in-time universe
    faster than the cap lets it be sold, and such a row carries no prediction,
    so its `actual_return` is null. `calculate_backtest_returns` sums that
    column, and a null is skipped, so the published backtest charges the
    stranded holding a return of exactly zero relative to its peers while still
    counting its weight. The tranche simulation prices it from the market
    history instead. `unpriced_weight_rows` counts how many rows that affects.
    Any residual on a date where every holding has an `actual_return` would mean
    the absolute leg is wrong and everything downstream of it should be
    discarded.
    """
    horizon_returns = calculate_tranche_horizon_returns(
        weights=weights,
        daily_returns=daily_returns,
        horizon=horizon,
    )

    benchmark_column = f"vn30_forward_return_{horizon}d"

    priced = weights.merge(
        labels[["date", "ticker", benchmark_column]],
        on=["date", "ticker"],
        how="left",
    )

    published_legs = priced.groupby("date").apply(
        lambda rows: pd.Series(
            {
                "active_leg": float(
                    (rows["weight"] * rows["actual_return"]).sum()
                ),
                "benchmark_leg": float(
                    (rows["weight"] * rows[benchmark_column]).sum()
                ),
            }
        ),
        include_groups=False,
    )

    check = horizon_returns.merge(
        published_legs.reset_index(),
        on="date",
        how="inner",
    )

    published = backtest_returns[
        (backtest_returns["optimization_mode"] == optimization_mode)
        & (backtest_returns["execution_mode"] == "normal")
    ][["date", "before_cost_return"]]

    check = check.merge(published, on="date", how="left")

    reconstruction_error = (
        check["horizon_return"] - check["active_leg"] - check["benchmark_leg"]
    ).abs()

    published_error = (
        check["active_leg"] - check["before_cost_return"]
    ).abs()

    tolerance = 1e-10
    fully_priced = (
        weights.groupby("date")["actual_return"].apply(lambda rows: rows.notna().all())
    )
    unexplained = check.set_index("date")[
        reconstruction_error.to_numpy() > tolerance
    ].index.intersection(fully_priced[fully_priced].index)

    return {
        "reconciled_dates": int(len(check)),
        "unpriced_weight_rows": int(weights["actual_return"].isna().sum()),
        "dates_with_residual": int((reconstruction_error > tolerance).sum()),
        "dates_with_unexplained_residual": int(len(unexplained)),
        "max_absolute_reconstruction_error": float(
            reconstruction_error.max(skipna=True)
        ),
        "mean_absolute_reconstruction_error": float(
            reconstruction_error.mean(skipna=True)
        ),
        "max_published_active_leg_error": float(
            published_error.max(skipna=True)
        ),
    }


def summarize_published_phase_sampling(
    published: pd.DataFrame,
    horizon: int,
) -> dict[str, float]:
    """Reproduce the every-fifth-date sampling of the published series.

    The published `after_cost_return` is a five-day forward active return
    recorded on every date, so taking every `horizon`-th row and compounding
    gives one non-overlapping active equity curve per starting offset. This is
    the measurement the tranche construction replaces, computed here from the
    published artefact so the two sit side by side without either being
    reconstructed twice.
    """
    ordered = published.sort_values("date")["after_cost_return"].to_numpy()

    finals = []
    drawdowns = []
    means = []

    for offset in range(horizon):
        sampled = pd.Series(ordered[offset::horizon])
        equity = (1.0 + sampled).cumprod()
        finals.append(float(equity.iloc[-1]))
        drawdowns.append(float((equity / equity.cummax() - 1.0).min()))
        means.append(float(sampled.mean()))

    return {
        "published_phase_final_equity_min": float(min(finals)),
        "published_phase_final_equity_max": float(max(finals)),
        "published_phase_final_equity_spread": float(max(finals) - min(finals)),
        "published_phase_max_drawdown_percent_worst": 100.0 * float(min(drawdowns)),
        "published_phase_max_drawdown_percent_best": 100.0 * float(max(drawdowns)),
        "published_phase_mean_of_phase_means": float(np.mean(means)),
    }


def build_daily_series_table(
    after_cost_returns: dict[str, pd.DataFrame],
    tranche_turnover: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    benchmark_turnover: pd.DataFrame,
) -> pd.DataFrame:
    """One row per date: each optimizer's after-cost return, the benchmark's
    return, and the turnover of the tranche rebalanced on that date.

    Returns are present only on dates when every phase sleeve is live, which are
    exactly the dates the summary table compounds, so compounding a return
    column reproduces that table's final equity and drawdown. Turnover is
    present on every rebalance date, including the warm-up dates before every
    phase is live, because the published average turnover is the mean over all
    of them.
    """
    frames = [
        benchmark[["date", "benchmark_return"]],
        benchmark_turnover.rename(
            columns={"tranche_turnover": "benchmark_tranche_turnover"}
        ),
    ]

    for optimization_mode in sorted(after_cost_returns):
        frames.append(
            after_cost_returns[optimization_mode][
                ["date", "after_cost_return"]
            ].rename(
                columns={
                    "after_cost_return": (
                        f"{optimization_mode}_after_cost_return"
                    )
                }
            )
        )
        frames.append(
            tranche_turnover[optimization_mode].rename(
                columns={
                    "tranche_turnover": f"{optimization_mode}_tranche_turnover"
                }
            )
        )

    for frame in frames:
        if frame["date"].duplicated().any():
            raise ValueError("Daily series carries a duplicated date")

    table = frames[0]
    for frame in frames[1:]:
        table = table.merge(frame, on="date", how="outer")

    table = table.sort_values("date").reset_index(drop=True)
    table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")

    return table


def summarize_mode(
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    benchmark: pd.DataFrame,
    liquidity: pd.DataFrame,
    horizon: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    costs = calculate_tranche_costs(
        weights=weights,
        liquidity=liquidity,
        horizon=horizon,
    )

    simulation = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        costs=costs,
        horizon=horizon,
    )

    curves = build_equity_curves(
        simulation=simulation,
        benchmark=benchmark,
        horizon=horizon,
    )

    before_cost = summarize_equity_curve(curves["before_cost_return"])
    after_cost = summarize_equity_curve(curves["after_cost_return"])
    benchmark_summary = summarize_equity_curve(curves["benchmark_return"])
    active_before_cost = summarize_equity_curve(
        curves["active_before_cost_return"]
    )
    active_after_cost = summarize_equity_curve(
        curves["active_after_cost_return"]
    )

    phase_dispersion = summarize_phase_dispersion(
        curves=curves,
        return_column_template="phase_{phase}_after_cost_return",
        horizon=horizon,
    )
    active_phase_dispersion = summarize_phase_dispersion(
        curves=curves,
        return_column_template="phase_{phase}_active_after_cost_return",
        horizon=horizon,
    )
    offset_sensitivity = summarize_start_offset_sensitivity(
        returns=curves["after_cost_return"],
        horizon=horizon,
    )
    active_offset_sensitivity = summarize_start_offset_sensitivity(
        returns=curves["active_after_cost_return"],
        horizon=horizon,
    )

    row = {
        "periods": after_cost["periods"],
        "years": after_cost["years"],
        "first_date": curves["date"].iloc[0].date().isoformat(),
        "last_date": curves["date"].iloc[-1].date().isoformat(),
        "average_tranche_turnover": float(costs["tranche_turnover"].mean()),
        "average_daily_cost": float(curves["tranche_cost"].mean()),
        "final_equity_before_cost": before_cost["final_equity"],
        "final_equity_after_cost": after_cost["final_equity"],
        "max_drawdown_percent_before_cost": 100.0 * before_cost["max_drawdown"],
        "max_drawdown_percent_after_cost": 100.0 * after_cost["max_drawdown"],
        "annualised_return_percent_before_cost": (
            100.0 * before_cost["annualised_return"]
        ),
        "annualised_return_percent_after_cost": (
            100.0 * after_cost["annualised_return"]
        ),
        "annualised_volatility_percent_after_cost": (
            100.0 * after_cost["annualised_volatility"]
        ),
        "benchmark_final_equity": benchmark_summary["final_equity"],
        "benchmark_max_drawdown_percent": (
            100.0 * benchmark_summary["max_drawdown"]
        ),
        "benchmark_annualised_return_percent": (
            100.0 * benchmark_summary["annualised_return"]
        ),
        "active_final_equity_before_cost": active_before_cost["final_equity"],
        "active_final_equity_after_cost": active_after_cost["final_equity"],
        "active_max_drawdown_percent_before_cost": (
            100.0 * active_before_cost["max_drawdown"]
        ),
        "active_max_drawdown_percent_after_cost": (
            100.0 * active_after_cost["max_drawdown"]
        ),
        "active_annualised_return_percent_after_cost": (
            100.0 * active_after_cost["annualised_return"]
        ),
        "phase_final_equity_min": phase_dispersion["phase_final_equity_min"],
        "phase_final_equity_max": phase_dispersion["phase_final_equity_max"],
        "phase_final_equity_spread": phase_dispersion[
            "phase_final_equity_spread"
        ],
        "phase_max_drawdown_percent_worst": (
            100.0 * phase_dispersion["phase_max_drawdown_min"]
        ),
        "phase_max_drawdown_percent_best": (
            100.0 * phase_dispersion["phase_max_drawdown_max"]
        ),
        "active_phase_final_equity_min": active_phase_dispersion[
            "phase_final_equity_min"
        ],
        "active_phase_final_equity_max": active_phase_dispersion[
            "phase_final_equity_max"
        ],
        "active_phase_final_equity_spread": active_phase_dispersion[
            "phase_final_equity_spread"
        ],
        "active_phase_max_drawdown_percent_worst": (
            100.0 * active_phase_dispersion["phase_max_drawdown_min"]
        ),
        "active_phase_max_drawdown_percent_best": (
            100.0 * active_phase_dispersion["phase_max_drawdown_max"]
        ),
        "active_offset_final_equity_spread": active_offset_sensitivity[
            "offset_final_equity_spread"
        ],
        "offset_final_equity_spread": offset_sensitivity[
            "offset_final_equity_spread"
        ],
        "offset_max_drawdown_percent_spread": (
            100.0 * offset_sensitivity["offset_max_drawdown_spread"]
        ),
        "phase_to_offset_spread_ratio": (
            phase_dispersion["phase_final_equity_spread"]
            / offset_sensitivity["offset_final_equity_spread"]
            if offset_sensitivity["offset_final_equity_spread"] > 0
            else float("inf")
        ),
    }

    return row, curves


def main() -> None:
    args = parse_args()

    weights = load_optimized_weights(args.weights_path)
    liquidity = load_liquidity_features()[
        ["date", "ticker", "is_low_liquidity"]
    ]
    backtest_returns = pd.read_parquet(args.backtest_returns_path)
    backtest_returns["date"] = pd.to_datetime(backtest_returns["date"])
    labels = pd.read_parquet("data/processed/labels.parquet")
    labels["date"] = pd.to_datetime(labels["date"])

    market_history = load_canonical_market_history(
        args.raw_market_path,
        args.adjusted_market_path,
    )
    membership = load_verified_membership_history(args.membership_dir)

    daily_returns = build_daily_total_returns(market_history)

    point_in_time_universe = filter_to_point_in_time_universe(
        market_data=market_history[["date", "ticker"]].copy(),
        membership=membership,
    )

    rebalance_dates = pd.DatetimeIndex(sorted(pd.unique(weights["date"])))

    benchmark_weights = build_equal_weight_benchmark_weights(
        universe=point_in_time_universe,
        rebalance_dates=rebalance_dates,
    )

    # The benchmark runs through the identical tranche construction, so the
    # active number compares two portfolios that trade on the same cadence and
    # differ only in what they hold. It carries no cost: it is the yardstick,
    # not a tradeable competitor.
    benchmark = simulate_tranche_portfolio(
        weights=benchmark_weights,
        daily_returns=daily_returns,
        costs=None,
        horizon=args.horizon,
    )[["date", "before_cost_return"]].rename(
        columns={"before_cost_return": "benchmark_return"}
    )

    print("TRANCHE PORTFOLIO DRAWDOWN DIAGNOSTIC")
    print("=" * 80)
    print("Weights:", args.weights_path)
    print("Rebalance dates:", weights["date"].nunique())
    print("Hold horizon (trading days):", args.horizon)
    print("Tranches held simultaneously:", args.horizon)
    print("Return leg: adjusted close, buy-and-hold within the holding period")
    print(
        "Benchmark: equal-weight point-in-time VN30 constituents, run through "
        "the same five-tranche construction"
    )
    print(
        "           (the canonical history carries no VN30 index price series)"
    )
    print(
        "Benchmark constituents per date: "
        f"{benchmark_weights.groupby('date').size().min()} to "
        f"{benchmark_weights.groupby('date').size().max()}"
    )
    print()

    modes = sorted(weights["optimization_mode"].unique())

    print("RECONCILIATION AGAINST THE PUBLISHED BACKTEST")
    print("-" * 80)
    for optimization_mode in modes:
        mode_weights = weights[
            weights["optimization_mode"] == optimization_mode
        ]
        reconciliation = reconcile_against_backtester(
            weights=mode_weights,
            daily_returns=daily_returns,
            labels=labels,
            backtest_returns=backtest_returns,
            optimization_mode=optimization_mode,
            horizon=args.horizon,
        )
        print(
            f"{optimization_mode:<20} "
            f"dates={reconciliation['reconciled_dates']:>5}  "
            f"published-leg error="
            f"{reconciliation['max_published_active_leg_error']:.2e}  "
            f"unpriced weight rows="
            f"{reconciliation['unpriced_weight_rows']:>3}  "
            f"dates with residual="
            f"{reconciliation['dates_with_residual']:>3} "
            f"(unexplained {reconciliation['dates_with_unexplained_residual']}), "
            f"max {reconciliation['max_absolute_reconstruction_error']:.2e}, "
            f"mean {reconciliation['mean_absolute_reconstruction_error']:.2e}"
        )
    print()

    execution_weight_sets: dict[str, pd.DataFrame] = {
        FRICTIONLESS_EXECUTION: weights,
    }

    if not args.skip_price_limit:
        print("Reconstructing price-limit execution phase by phase...", flush=True)
        execution_reference = load_execution_reference()
        phase_weights = build_phase_partitioned_weights(
            weights=weights,
            horizon=args.horizon,
        )
        executed = apply_price_limit_execution_rules(
            weights=phase_weights,
            execution_reference=execution_reference,
        )
        execution_weight_sets[PRICE_LIMIT_EXECUTION] = strip_phase_labels(
            executed
        )
        print("Done.", flush=True)
        print()

    summary_rows = []
    daily_after_cost_returns: dict[str, pd.DataFrame] = {}
    daily_tranche_turnover: dict[str, pd.DataFrame] = {}

    for execution_mode, execution_weights in execution_weight_sets.items():
        for optimization_mode in modes:
            mode_weights = execution_weights[
                execution_weights["optimization_mode"] == optimization_mode
            ]

            row, curves = summarize_mode(
                weights=mode_weights[["date", "ticker", "weight"]],
                daily_returns=daily_returns,
                benchmark=benchmark,
                liquidity=liquidity,
                horizon=args.horizon,
            )

            if execution_mode == PRICE_LIMIT_EXECUTION:
                daily_after_cost_returns[optimization_mode] = curves
                daily_tranche_turnover[optimization_mode] = (
                    calculate_tranche_turnover(
                        weights=mode_weights[["date", "ticker", "weight"]],
                        horizon=args.horizon,
                    )
                )

            published = backtest_returns[
                (backtest_returns["optimization_mode"] == optimization_mode)
                & (
                    backtest_returns["execution_mode"]
                    == (
                        "normal"
                        if execution_mode == FRICTIONLESS_EXECUTION
                        else "price_limit_aware"
                    )
                )
            ].sort_values("date")

            published_cumulative = published[
                "cumulative_after_cost_active_return"
            ]
            published_drawdown = float(
                (published_cumulative - published_cumulative.cummax()).min()
            )

            summary_rows.append(
                {
                    "optimization_mode": optimization_mode,
                    "execution_mode": execution_mode,
                    **row,
                    **summarize_published_phase_sampling(
                        published=published,
                        horizon=args.horizon,
                    ),
                    "published_summed_max_active_drawdown": published_drawdown,
                    "published_summed_final_cumulative_active_return": float(
                        published_cumulative.iloc[-1]
                    ),
                    "published_average_daily_turnover": float(
                        published["portfolio_turnover"].mean()
                    ),
                }
            )

            print(
                f"{optimization_mode:<20} {execution_mode:<26} "
                f"equity={row['final_equity_after_cost']:>7.4f}  "
                f"maxDD={row['max_drawdown_percent_after_cost']:>8.2f}%  "
                f"active equity={row['active_final_equity_after_cost']:>7.4f}  "
                f"active maxDD="
                f"{row['active_max_drawdown_percent_after_cost']:>8.2f}%",
                flush=True,
            )

            if execution_mode == FRICTIONLESS_EXECUTION:
                print(
                    "    single-phase curves: final equity "
                    f"{row['phase_final_equity_min']:.4f} to "
                    f"{row['phase_final_equity_max']:.4f} "
                    f"(spread {row['phase_final_equity_spread']:.4f}); "
                    "tranche curve restarted on each of the "
                    f"{args.horizon} offsets spans only "
                    f"{row['offset_final_equity_spread']:.4f}"
                )

    summary = pd.DataFrame(summary_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    if daily_after_cost_returns:
        daily_table = build_daily_series_table(
            after_cost_returns=daily_after_cost_returns,
            tranche_turnover=daily_tranche_turnover,
            benchmark=benchmark,
            benchmark_turnover=calculate_tranche_turnover(
                weights=benchmark_weights[["date", "ticker", "weight"]],
                horizon=args.horizon,
            ),
        )
        daily_path = Path(args.daily_output)
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        daily_table.to_csv(daily_path, index=False)

    print()
    print("DRAWDOWN: COMPOUNDED CAPITAL VERSUS PUBLISHED SUMMED UNITS")
    print("-" * 80)
    print(
        summary[
            [
                "optimization_mode",
                "execution_mode",
                "final_equity_after_cost",
                "max_drawdown_percent_after_cost",
                "active_max_drawdown_percent_after_cost",
                "published_summed_max_active_drawdown",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("PHASE INDEPENDENCE")
    print("-" * 80)
    print(
        "Single-phase curves sample every fifth date; their starting offset is "
        "a free parameter."
    )
    print(
        "The tranche curve has no such parameter -- restarting it on each of "
        f"the {args.horizon} offsets only trims a few days."
    )
    print()
    print(
        summary[
            [
                "optimization_mode",
                "execution_mode",
                "phase_final_equity_min",
                "phase_final_equity_max",
                "phase_final_equity_spread",
                "offset_final_equity_spread",
                "phase_to_offset_spread_ratio",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    print("THE SAME SAMPLING APPLIED TO THE PUBLISHED ACTIVE SERIES")
    print("-" * 80)
    print(
        summary[
            [
                "optimization_mode",
                "execution_mode",
                "published_phase_final_equity_min",
                "published_phase_final_equity_max",
                "published_phase_final_equity_spread",
                "published_phase_max_drawdown_percent_worst",
                "published_phase_max_drawdown_percent_best",
                "published_phase_mean_of_phase_means",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print()
    print("Output:", output_path)
    if daily_after_cost_returns:
        print("Daily series:", Path(args.daily_output))
    else:
        print("Daily series: not written under --skip-price-limit")
    print(
        "Diagnostic only. No production default, published table, or model "
        "configuration was changed."
    )


if __name__ == "__main__":
    main()
