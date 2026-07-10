from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.market_data import load_and_validate_completed_market_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate completed VN30 data and paper-trading timing.",
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
    timezone = ZoneInfo(config["timing"]["timezone"])
    generated_at = datetime.now(timezone)
    validation = load_and_validate_completed_market_data(
        data_path=config["model"]["raw_ohlcv_path"],
        universe_path=config["model"]["universe_path"],
        generated_at=generated_at,
        timezone_name=config["timing"]["timezone"],
        data_update_cutoff=config["timing"]["earliest_data_update_time"],
        execution_submission_cutoff=config["timing"][
            "execution_submission_cutoff_time"
        ],
        holiday_dates=config["market_calendar"]["holiday_dates"],
    )
    timing = validation.timing

    print()
    print("PAPER-TRADING TIMING VALIDATION PASSED")
    print("=" * 80)
    print(f"Generated at: {timing.generated_at.isoformat()}")
    print(f"Data as of date: {timing.data_asof_date}")
    print(f"Signal date: {timing.signal_date}")
    print(f"Intended execution date: {timing.intended_execution_date}")
    print(
        f"Latest ticker coverage: {validation.latest_row_count}/"
        f"{len(validation.expected_tickers)}"
    )

    if validation.warnings:
        print("Warnings:")

        for warning in validation.warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")

    print("Paper trading only. No real orders were placed.")
    print()


if __name__ == "__main__":
    main()
