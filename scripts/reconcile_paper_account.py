from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.storage import PaperAccountStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and reconcile the local VN30 paper account.",
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
        help="Path to the paper-trading YAML configuration.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paper_trading_config(args.config)
    storage = PaperAccountStorage(config["output"]["directory"])

    with storage.account_lock():
        storage.validate_all_ledgers()
        broker = storage.load_broker_state()

    pending = [
        settlement
        for settlement in broker.pending_settlements
        if settlement.status.value == "PENDING"
    ]

    print()
    print("PAPER ACCOUNT RECONCILIATION PASSED")
    print("=" * 80)
    print(f"As of date: {broker.asof_date}")
    print(f"Settled cash: {broker.settled_cash}")
    print(f"Unsettled cash: {broker.unsettled_cash}")
    print(f"Buying power: {broker.buying_power}")
    print(f"Tracked positions: {len(broker.positions)}")
    print(f"Pending settlements: {len(pending)}")
    print("Paper trading only. No real orders were placed.")
    print()


if __name__ == "__main__":
    main()
