"""Rebuild the paper-trading OHLCV file from the canonical FiinQuant history.

The paper-trading file was originally downloaded from vnstock, which serves a
single back-adjusted price series: every price column held adjusted prices and
``adjusted_close`` merely repeated ``close``. That is wrong for a file whose
prices drive execution. Back-adjusted history is anchored at the present, so
the most recent bar matches the raw bar, but everything behind it is restated
by later corporate actions. Anything reading price levels from history - an
account replay, an average-daily-value lookback - was reading restated prices.

This script rebuilds the file from the canonical FiinQuant history with the
same contract the research pipeline uses:

    open, high, low, close   raw prices, for execution and price limits
    adjusted_close           adjusted close, for returns
    volume                   vendor volume
    value_traded             close * volume, matching the daily update

Prices stay in thousands of VND, matching ``price_multiplier_to_vnd`` in the
paper-trading configuration. Coverage is unchanged: the current index members
over the same history the file already spans.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.data_update import OUTPUT_COLUMNS
from src.research_market_data import (
    add_market_data_arguments,
    load_research_market_history,
)
from src.vn30_universe import constituents_on_date


DEFAULT_START_DATE = "2020-01-02"


def build_paper_trading_ohlcv(
    market: pd.DataFrame,
    members: tuple[str, ...],
    start_date: str,
) -> pd.DataFrame:
    selected = market.loc[
        market["ticker"].isin(set(members))
        & (market["date"] >= pd.Timestamp(start_date))
    ].copy()

    missing = sorted(set(members).difference(selected["ticker"]))

    if missing:
        raise ValueError(f"Canonical history is missing index members: {missing}")

    # Recomputed rather than taken from the vendor turnover column so the
    # rebuilt history and the daily append share one definition.
    selected["value_traded"] = selected["close"] * selected["volume"]
    rebuilt = selected[list(OUTPUT_COLUMNS)].sort_values(["date", "ticker"])

    if rebuilt.duplicated(["date", "ticker"]).any():
        raise ValueError("Rebuilt OHLCV contains duplicate ticker-date rows")

    if rebuilt.isna().any().any():
        raise ValueError("Rebuilt OHLCV contains missing values")

    return rebuilt.reset_index(drop=True)


def summarize_change(existing: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    comparison = existing.merge(
        rebuilt,
        on=["date", "ticker"],
        how="outer",
        suffixes=("_existing", "_rebuilt"),
        indicator=True,
    )
    only_existing = int((comparison["_merge"] == "left_only").sum())
    only_rebuilt = int((comparison["_merge"] == "right_only").sum())
    shared = comparison.loc[comparison["_merge"] == "both"]
    difference = (
        shared["close_rebuilt"] - shared["close_existing"]
    ).abs() / shared["close_existing"].abs()

    print()
    print("CHANGE AGAINST THE EXISTING FILE")
    print("-" * 80)
    print(f"Rows only in the existing file: {only_existing}")
    print(f"Rows only in the rebuilt file:  {only_rebuilt}")
    print(f"Shared ticker-dates:            {len(shared)}")
    print(f"Closes differing by >0.1%:      {int((difference > 0.001).sum())}")
    print(f"Largest relative close change:  {difference.max():.4%}")

    latest = shared.loc[shared["date"] == shared["date"].max()]
    latest_difference = (
        latest["close_rebuilt"] - latest["close_existing"]
    ).abs().max()
    print(f"Largest change on the last date: {latest_difference:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
        help="Path to the paper-trading YAML configuration.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="First date to include in the rebuilt file.",
    )
    parser.add_argument(
        "--asof-date",
        default=None,
        help="Membership date. Defaults to the last date in the history.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the change without writing the file.",
    )
    add_market_data_arguments(parser, point_in_time_default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paper_trading_config(args.config)
    output_path = ROOT / config["model"]["raw_ohlcv_path"]

    market = load_research_market_history(
        args.raw_market_path,
        args.adjusted_market_path,
        point_in_time=False,
    )
    asof_date = args.asof_date or market["date"].max().date().isoformat()
    members = constituents_on_date(asof_date)
    rebuilt = build_paper_trading_ohlcv(
        market=market,
        members=members,
        start_date=args.start_date,
    )

    print()
    print("REBUILD PAPER-TRADING OHLCV FROM FIINQUANT")
    print("=" * 80)
    print(f"Source: {args.raw_market_path}")
    print(f"        {args.adjusted_market_path}")
    print(f"Destination: {output_path}")
    print(f"Membership as of: {asof_date} ({len(members)} tickers)")
    print(f"Rows: {len(rebuilt)}")
    print(f"Date range: {rebuilt['date'].min().date()} to {rebuilt['date'].max().date()}")

    if output_path.is_file():
        existing = pd.read_csv(output_path, parse_dates=["date"])
        summarize_change(existing, rebuilt)

    if args.dry_run:
        print()
        print("DRY RUN. No file was written.")
        print()
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_suffix(output_path.suffix + ".staging")

    try:
        rebuilt.to_csv(staging, index=False, lineterminator="\n")
        staged = pd.read_csv(staging)

        if len(staged) != len(rebuilt):
            raise ValueError("Staged row count does not match the rebuilt data")

        if list(staged.columns) != list(OUTPUT_COLUMNS):
            raise ValueError("Staged columns do not match the expected schema")

        staging.replace(output_path)
    finally:
        staging.unlink(missing_ok=True)

    print()
    print("PAPER-TRADING OHLCV REBUILT")
    print()


if __name__ == "__main__":
    main()
