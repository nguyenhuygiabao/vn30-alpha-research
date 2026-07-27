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
    summarize_membership_coverage,
)
from src.universe_reconstruction import (
    reconstruct_constituent_history,
)
from src.vn30_universe import DEFAULT_HISTORY_DIR


HISTORY_FILES = {
    "initial_snapshot": "initial_snapshot.csv",
    "review_calendar": "review_calendar.csv",
    "membership_changes": "membership_changes.csv",
    "source_manifest": "source_manifest.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit authoritative point-in-time VN30 membership "
            "against explicit market history."
        )
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help=(
            "Directory containing the four authoritative VN30 "
            "history CSV files."
        ),
    )
    parser.add_argument(
        "--market-data-path",
        type=Path,
        required=True,
        help=(
            "Full historical market-data CSV. An explicit path is "
            "required to prevent accidental use of legacy data."
        ),
    )
    parser.add_argument(
        "--expected-constituents",
        type=int,
        default=30,
    )
    return parser.parse_args()


def load_reconstruction(
    history_dir: Path,
    expected_constituents: int,
):
    paths = {
        name: history_dir / filename
        for name, filename in HISTORY_FILES.items()
    }
    missing = [
        str(path)
        for path in paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing authoritative history files: "
            + ", ".join(missing)
        )

    return reconstruct_constituent_history(
        initial_snapshot=pd.read_csv(
            paths["initial_snapshot"]
        ),
        review_calendar=pd.read_csv(
            paths["review_calendar"]
        ),
        membership_changes=pd.read_csv(
            paths["membership_changes"]
        ),
        source_manifest=pd.read_csv(
            paths["source_manifest"]
        ),
        expected_size=expected_constituents,
    )


def load_market_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Market-data file not found: {path}"
        )

    data = pd.read_csv(path)
    required = {
        "ticker",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
    }
    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            f"Market data are missing columns: {missing}"
        )
    if data.empty:
        raise ValueError("Market data are empty")

    data["date"] = pd.to_datetime(
        data.pop("timestamp"),
        errors="raise",
    ).dt.normalize()
    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if data["ticker"].eq("").any():
        raise ValueError("Market data contain blank tickers")
    if data.duplicated(["date", "ticker"]).any():
        raise ValueError(
            "Market data contain duplicate date/ticker rows"
        )

    return data


def main() -> None:
    args = parse_args()

    result = load_reconstruction(
        history_dir=args.history_dir,
        expected_constituents=args.expected_constituents,
    )
    membership = result.membership_history
    market_data = load_market_data(args.market_data_path)

    historical_tickers = set(
        result.ticker_pool["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    missing_market_tickers = sorted(
        historical_tickers.difference(
            set(market_data["ticker"])
        )
    )
    if missing_market_tickers:
        raise ValueError(
            "Market data missing historical tickers: "
            f"{missing_market_tickers}"
        )

    coverage = summarize_membership_coverage(
        membership,
        market_data["date"],
        expected_constituents=args.expected_constituents,
    )
    filtered = filter_to_point_in_time_universe(
        market_data,
        membership,
    )

    incomplete_dates = int(
        (~coverage["coverage_complete"]).sum()
    )

    print("\nPOINT-IN-TIME VN30 UNIVERSE AUDIT")
    print("=" * 80)
    print(f"Snapshot rows: {len(result.snapshots)}")
    print(
        "Review dates: "
        f"{result.snapshots['effective_date'].nunique()}"
    )
    print(f"Membership intervals: {len(membership)}")
    print(f"Membership tickers: {len(result.ticker_pool)}")
    print(f"Market rows: {len(market_data)}")
    print(
        f"Market tickers: {market_data['ticker'].nunique()}"
    )
    print(
        "Market history starts: "
        f"{market_data['date'].min().date()}"
    )
    print(
        "Market history ends: "
        f"{market_data['date'].max().date()}"
    )
    print(f"Filtered market rows: {len(filtered)}")
    print(f"Incomplete membership dates: {incomplete_dates}")
    print(
        "Minimum active constituents: "
        f"{coverage['active_constituents'].min()}"
    )
    print(
        "Maximum active constituents: "
        f"{coverage['active_constituents'].max()}"
    )

    if incomplete_dates:
        raise ValueError(
            "Point-in-time membership coverage is incomplete"
        )

    print("Point-in-time membership coverage passed.")


if __name__ == "__main__":
    main()
