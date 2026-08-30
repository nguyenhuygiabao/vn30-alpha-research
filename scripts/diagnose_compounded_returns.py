"""Measure what the backtest returns look like compounded rather than summed.

The published per-period figures are means over overlapping forecast periods,
and the published cumulative figures are sums of those overlapping periods.
Neither answers the question a reader actually has, which is what a portfolio
running this strategy would have done.

Taking every Nth date gives non-overlapping periods that can be compounded,
but there are N ways to do that depending on which date the sampling starts
from. This script reports all of them, because the spread between them is the
result: if the outcome depends on the starting weekday, the edge is not
distinguishable from noise at this sample size.

Writes reports/tables/compounded_phase_sensitivity.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKTEST_RETURNS_PATH = ROOT / "data" / "processed" / "backtest_returns.parquet"
OUTPUT_PATH = ROOT / "reports" / "tables" / "compounded_phase_sensitivity.csv"

DEFAULT_HORIZON = 5

OUTPUT_COLUMNS = (
    "optimization_mode",
    "execution_mode",
    "phase",
    "periods",
    "average_period_active_return",
    "final_equity",
    "max_drawdown_percent",
)


def compound(returns: pd.Series) -> tuple[float, float]:
    """Final equity and peak-to-trough drawdown of a compounded series."""
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(equity.iloc[-1]), float(drawdown.min())


def build_phase_sensitivity(
    backtest_returns: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """One row per optimization mode, execution mode and overlap phase."""
    rows: list[dict[str, object]] = []

    grouped = backtest_returns.groupby(["optimization_mode", "execution_mode"])

    for (optimization_mode, execution_mode), group in grouped:
        ordered = group.sort_values("date")
        series = ordered["after_cost_return"].reset_index(drop=True)

        for phase in range(horizon):
            disjoint = series.iloc[phase::horizon]

            if disjoint.empty:
                continue

            final_equity, drawdown = compound(disjoint)

            rows.append(
                {
                    "optimization_mode": optimization_mode,
                    "execution_mode": execution_mode,
                    "phase": phase,
                    "periods": len(disjoint),
                    "average_period_active_return": float(disjoint.mean()),
                    "final_equity": final_equity,
                    "max_drawdown_percent": drawdown * 100.0,
                }
            )

    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def reconcile(
    backtest_returns: pd.DataFrame,
    phases: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Check the phase means average back to the published overlapping mean.

    They must, since every observation appears in exactly one phase. A
    mismatch would mean the phase split lost or duplicated rows.
    """
    published = (
        backtest_returns.groupby(["optimization_mode", "execution_mode"])[
            "after_cost_return"
        ]
        .mean()
        .rename("published_overlapping_mean")
    )

    # Phases do not all hold the same number of periods when the series
    # length is not a multiple of the horizon, so the phase means have to be
    # weighted by their counts to recover the pooled mean.
    weighted = phases.assign(
        total=phases["average_period_active_return"] * phases["periods"]
    )
    totals = weighted.groupby(["optimization_mode", "execution_mode"])[
        ["total", "periods"]
    ].sum()
    recovered = (totals["total"] / totals["periods"]).rename(
        "weighted_mean_of_phase_means"
    )

    check = pd.concat([published, recovered], axis=1)
    check["absolute_difference"] = (
        check["published_overlapping_mean"] - check["weighted_mean_of_phase_means"]
    ).abs()
    return check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forecast horizon in trading days, which sets the phase count.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the table without writing it.",
    )
    args = parser.parse_args()

    if not BACKTEST_RETURNS_PATH.exists():
        raise SystemExit(
            f"Missing {BACKTEST_RETURNS_PATH}. Run the backtester first."
        )

    backtest_returns = pd.read_parquet(BACKTEST_RETURNS_PATH)
    phases = build_phase_sensitivity(backtest_returns, args.horizon)

    check = reconcile(backtest_returns, phases, args.horizon)
    worst = float(check["absolute_difference"].max())

    print("PHASE SENSITIVITY")
    print(phases.to_string(index=False))
    print()
    print("RECONCILIATION AGAINST THE PUBLISHED OVERLAPPING MEAN")
    print(check.to_string())
    print()
    print(f"Worst absolute difference: {worst:.12f}")

    if worst > 1e-9:
        raise SystemExit(
            "Phase means do not average back to the published mean, so the "
            "phase split lost or duplicated observations."
        )

    spread = (
        phases.groupby(["optimization_mode", "execution_mode"])["final_equity"]
        .agg(["min", "max"])
    )
    spread["spread"] = spread["max"] - spread["min"]
    print()
    print("FINAL EQUITY SPREAD ACROSS PHASES")
    print(spread.to_string())

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    phases.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
