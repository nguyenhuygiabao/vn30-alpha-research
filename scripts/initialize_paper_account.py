from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_trading.calendar import normalize_date
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.storage import PaperAccountStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the local VN30 paper-trading account ledgers.",
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
        help="Path to the paper-trading YAML configuration.",
    )
    parser.add_argument(
        "--asof-date",
        help="Opening account date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reset existing local paper ledgers. This deletes prior paper state.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paper_trading_config(args.config)
    timezone = ZoneInfo(config["timing"]["timezone"])
    opening_date = (
        normalize_date(args.asof_date)
        if args.asof_date
        else datetime.now(timezone).date()
    )
    storage = PaperAccountStorage(config["output"]["directory"])

    with storage.account_lock():
        broker = storage.initialize(
            initial_cash=config["account"]["initial_cash_vnd"],
            asof_date=opening_date,
            overwrite=args.force,
        )

    print()
    print("PAPER ACCOUNT INITIALIZED")
    print("=" * 80)
    print(f"Account: {config['account']['account_id']}")
    print(f"Opening date: {opening_date.isoformat()}")
    print(f"Currency: {config['account']['currency']}")
    print(f"Settled cash: {broker.settled_cash}")
    print(f"Output directory: {storage.output_directory}")
    print("Paper trading only. No real orders were placed.")
    print()


if __name__ == "__main__":
    main()
