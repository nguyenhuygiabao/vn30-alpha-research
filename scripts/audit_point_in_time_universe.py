from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.universe_history import (
    filter_to_point_in_time_universe,
    normalize_membership_history,
    summarize_membership_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dated VN30 constituent membership against market history."
    )
    parser.add_argument(
        "--membership-path",
        default="config/vn30_membership_history.csv",
        help="CSV columns: ticker,effective_from,effective_to",
    )
    parser.add_argument(
        "--market-data-path", default="data/raw/vnstock/vn30_ohlcv.csv"
    )
    parser.add_argument("--expected-constituents", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    membership_path = Path(args.membership_path)
    if not membership_path.exists():
        raise FileNotFoundError(
            f"Point-in-time membership file not found: {membership_path}. "
            "Do not substitute the current universe for historical membership."
        )
    membership = normalize_membership_history(pd.read_csv(membership_path))
    market_data = pd.read_csv(args.market_data_path)
    market_data["date"] = pd.to_datetime(market_data["date"]).dt.normalize()
    coverage = summarize_membership_coverage(
        membership,
        market_data["date"],
        expected_constituents=args.expected_constituents,
    )
    filtered = filter_to_point_in_time_universe(market_data, membership)

    print("\nPOINT-IN-TIME VN30 UNIVERSE AUDIT")
    print("=" * 80)
    print(f"Membership intervals: {len(membership)}")
    print(f"Membership tickers: {membership['ticker'].nunique()}")
    print(f"Market history starts: {market_data['date'].min().date()}")
    print(f"Filtered market rows: {len(filtered)}")
    print(f"Incomplete membership dates: {(~coverage['coverage_complete']).sum()}")
    print(f"Minimum active constituents: {coverage['active_constituents'].min()}")
    print(f"Maximum active constituents: {coverage['active_constituents'].max()}")
    if not coverage["coverage_complete"].all():
        raise ValueError("Point-in-time membership coverage is incomplete")
    print("Point-in-time membership coverage passed.")


if __name__ == "__main__":
    main()
