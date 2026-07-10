from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_ohlcv_csv
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.data_update import (
    build_daily_update,
    load_universe_metadata,
    write_staged_update,
)


class VnstockDailyProvider:
    def __init__(self, source: str) -> None:
        self.source = source.strip().upper()

    def history(
        self,
        ticker: str,
        provider_symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        try:
            from vnstock import Quote
        except ImportError as error:
            raise RuntimeError(
                "vnstock Quote is required. See the Python setup instructions "
                "in README.md."
            ) from error

        quote = Quote(symbol=provider_symbol, source=self.source)

        return quote.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="d",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely update the local completed VN30 daily OHLCV file.",
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
        help="Path to the paper-trading YAML configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and validate without replacing local files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paper_trading_config(args.config)
    timezone = ZoneInfo(config["timing"]["timezone"])
    generated_at = datetime.now(timezone)
    data_path = config["model"]["raw_ohlcv_path"]
    existing = load_ohlcv_csv(data_path)
    universe = load_universe_metadata(config["model"]["universe_path"])
    update_config = config["data_update"]
    provider = VnstockDailyProvider(update_config["source"])
    result = build_daily_update(
        existing_data=existing,
        universe=universe,
        provider=provider,
        generated_at=generated_at,
        timezone_name=config["timing"]["timezone"],
        data_update_cutoff=config["timing"]["earliest_data_update_time"],
        execution_submission_cutoff=config["timing"][
            "execution_submission_cutoff_time"
        ],
        holiday_dates=config["market_calendar"]["holiday_dates"],
        overlap_calendar_days=update_config["overlap_calendar_days"],
        request_interval_seconds=update_config["request_interval_seconds"],
        max_attempts=update_config["max_attempts"],
        overlap_close_tolerance=update_config["overlap_close_tolerance"],
        suspicious_return_threshold=update_config[
            "suspicious_return_threshold"
        ],
    )
    write_staged_update(
        result=result,
        output_path=data_path,
        audit_path=update_config["audit_path"],
        dry_run=args.dry_run,
    )

    timing = result.validation.timing
    print()
    print("DAILY DATA UPDATE VALIDATION PASSED")
    print("=" * 80)
    print(f"Mode: {'dry run' if args.dry_run else 'updated local file'}")
    print(f"Update window: {result.update_start_date} to {result.update_end_date}")
    print(f"Data as of date: {timing.data_asof_date}")
    print(f"Signal date: {timing.signal_date}")
    print(f"Intended execution date: {timing.intended_execution_date}")
    print(f"Ticker coverage: {result.validation.latest_row_count}/{len(universe)}")
    print(f"Total rows: {len(result.combined_data)}")
    print(f"Suspicious move flags: {len(result.suspicious_moves)}")

    if result.validation.warnings:
        print("Warnings:")

        for warning in result.validation.warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")

    print("No paper orders or real orders were placed.")
    print()


if __name__ == "__main__":
    main()
