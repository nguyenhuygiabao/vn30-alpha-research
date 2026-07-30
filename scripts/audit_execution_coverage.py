from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_pipeline import (
    ADJUSTED_MARKET_PATH,
    MEMBERSHIP_HISTORY_DIR,
    RAW_MARKET_PATH,
    load_verified_membership_history,
)
from src.market_data_schema import load_canonical_market_history
from src.universe_history import normalize_membership_history

SUMMARY_PATH = ROOT / "reports" / "tables" / "execution_coverage_summary.csv"
GAPS_PATH = ROOT / "reports" / "tables" / "execution_coverage_gaps.csv"

RAW_PRICE_COLUMNS = ["open", "high", "low", "close"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether every active point-in-time VN30 constituent has "
            "genuine execution-valid data (raw OHLC, volume, traded value) "
            "on every date in the dataset's trading calendar."
        )
    )
    parser.add_argument(
        "--raw-market-path",
        type=Path,
        default=Path(RAW_MARKET_PATH),
    )
    parser.add_argument(
        "--adjusted-market-path",
        type=Path,
        default=Path(ADJUSTED_MARKET_PATH),
    )
    parser.add_argument(
        "--membership-history-dir",
        type=Path,
        default=Path(MEMBERSHIP_HISTORY_DIR),
    )
    return parser.parse_args()


def build_expected_ticker_dates(
    trading_dates: pd.DatetimeIndex,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Every (date, ticker) pair where ticker is active per verified
    point-in-time membership on a date that is part of the dataset's
    trading calendar (so exchange holidays are not treated as gaps)."""
    normalized = normalize_membership_history(membership)
    trading_dates = pd.DatetimeIndex(trading_dates).sort_values()

    expected_frames = []
    for interval in normalized.itertuples(index=False):
        start = interval.effective_from
        end = interval.effective_to if pd.notna(interval.effective_to) else trading_dates.max()
        active_dates = trading_dates[
            (trading_dates >= start) & (trading_dates <= end)
        ]
        if len(active_dates) == 0:
            continue
        expected_frames.append(
            pd.DataFrame(
                {
                    "date": active_dates,
                    "ticker": interval.ticker,
                }
            )
        )

    expected = pd.concat(expected_frames, ignore_index=True)
    duplicate_keys = expected.duplicated(["date", "ticker"]).sum()
    if duplicate_keys:
        raise ValueError(
            f"Membership intervals produce {duplicate_keys} duplicate "
            "active date/ticker pairs -- overlapping intervals for the "
            "same ticker."
        )
    return expected


def classify_execution_status(
    expected: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    market_columns = [
        "date",
        "ticker",
        *RAW_PRICE_COLUMNS,
        "volume",
        "value_traded",
        "value_traded_is_estimated",
    ]
    joined = expected.merge(
        market[market_columns],
        on=["date", "ticker"],
        how="left",
        indicator=True,
    )

    row_present = joined["_merge"].eq("both")
    raw_price_available = joined[RAW_PRICE_COLUMNS].notna().all(axis=1)
    volume_available = joined["volume"].notna()
    value_traded_real = joined["value_traded"].notna()
    value_traded_estimated = (
        joined["value_traded_is_estimated"].astype("boolean").fillna(False)
    )

    status = pd.Series("missing_row", index=joined.index)
    status = status.mask(row_present & ~raw_price_available, "raw_price_unavailable")
    status = status.mask(
        row_present & raw_price_available & ~volume_available,
        "volume_unavailable",
    )
    status = status.mask(
        row_present & raw_price_available & volume_available & value_traded_real,
        "full_raw_execution_data",
    )
    status = status.mask(
        row_present
        & raw_price_available
        & volume_available
        & ~value_traded_real
        & value_traded_estimated,
        "estimated_traded_value",
    )
    status = status.mask(
        row_present
        & raw_price_available
        & volume_available
        & ~value_traded_real
        & ~value_traded_estimated,
        "value_traded_unavailable",
    )

    joined["execution_status"] = status
    return joined[["date", "ticker", "execution_status"]]


def build_date_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    counts = (
        coverage.groupby(["date", "execution_status"])
        .size()
        .unstack(fill_value=0)
    )
    for column in (
        "full_raw_execution_data",
        "estimated_traded_value",
        "raw_price_unavailable",
        "volume_unavailable",
        "value_traded_unavailable",
        "missing_row",
    ):
        if column not in counts.columns:
            counts[column] = 0

    counts["active_constituents"] = counts.sum(axis=1)
    counts["execution_complete"] = (
        counts["raw_price_unavailable"].eq(0)
        & counts["volume_unavailable"].eq(0)
        & counts["value_traded_unavailable"].eq(0)
        & counts["missing_row"].eq(0)
    )
    return counts.reset_index().sort_values("date")


def main() -> None:
    args = parse_args()

    market = load_canonical_market_history(
        args.raw_market_path,
        args.adjusted_market_path,
    )
    membership = load_verified_membership_history(args.membership_history_dir)

    trading_dates = pd.DatetimeIndex(market["date"].unique())
    expected = build_expected_ticker_dates(trading_dates, membership)
    coverage = classify_execution_status(expected, market)
    date_summary = build_date_summary(coverage)

    gaps = coverage.loc[
        coverage["execution_status"] != "full_raw_execution_data"
    ].sort_values(["date", "ticker"])

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    date_summary.to_csv(SUMMARY_PATH, index=False)
    gaps.to_csv(GAPS_PATH, index=False)

    blocked_dates = date_summary.loc[~date_summary["execution_complete"], "date"]
    status_totals = coverage["execution_status"].value_counts()

    print("EXECUTION COVERAGE AUDIT")
    print("=" * 80)
    print(f"Trading dates in dataset: {len(trading_dates)}")
    print(f"Expected active constituent-dates: {len(expected)}")
    print()
    print("Status totals:")
    for status, count in status_totals.items():
        print(f"  {status}: {count}")
    print()
    print(f"Dates with 100% execution-valid coverage: "
          f"{int(date_summary['execution_complete'].sum())} / {len(date_summary)}")
    print(f"Blocked dates (raw price, volume, or row missing): {len(blocked_dates)}")
    if len(blocked_dates):
        print(
            "Blocked date range: "
            f"{blocked_dates.min().date()} to {blocked_dates.max().date()}"
        )
    print(f"Dates relying on estimated traded value (FLC/ROS): "
          f"{int((date_summary['estimated_traded_value'] > 0).sum())}")
    print()
    print(f"Per-date summary written to: {SUMMARY_PATH}")
    print(f"Per-ticker-date gap detail written to: {GAPS_PATH}")


if __name__ == "__main__":
    main()
