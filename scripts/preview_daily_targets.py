from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_ohlcv_csv
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.market_data import (
    load_universe_tickers,
    validate_completed_market_data,
)
from src.paper_trading.scoring import score_completed_market_data
from src.paper_trading.targets import build_constrained_target_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview constrained target weights from completed VN30 data.",
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
    market_data = load_ohlcv_csv(config["model"]["raw_ohlcv_path"])
    universe = pd.read_csv(config["model"]["universe_path"])
    expected_tickers = load_universe_tickers(config["model"]["universe_path"])
    validation = validate_completed_market_data(
        data=market_data,
        expected_tickers=expected_tickers,
        generated_at=generated_at,
        timezone_name=config["timing"]["timezone"],
        data_update_cutoff=config["timing"]["earliest_data_update_time"],
        execution_submission_cutoff=config["timing"][
            "execution_submission_cutoff_time"
        ],
        holiday_dates=config["market_calendar"]["holiday_dates"],
    )
    scoring = score_completed_market_data(
        market_data=market_data,
        expected_tickers=expected_tickers,
        horizon_days=config["timing"]["signal_horizon_trading_days"],
        model_name=config["model"]["model_name"],
    )
    portfolio = config["portfolio"]
    targets = build_constrained_target_weights(
        predictions=scoring.predictions,
        universe=universe,
        target_holdings=portfolio["target_holdings"],
        target_invested_weight=portfolio["target_invested_weight"],
        max_single_name_weight=portfolio["max_single_name_weight"],
        max_issuer_group_weight=portfolio["max_issuer_group_weight"],
        max_sector_weight=portfolio["max_sector_weight"],
    )

    if targets.signal_date.date() != validation.timing.signal_date:
        raise ValueError("Target signal date does not match validated market timing")

    display = targets.target_weights.copy()
    display["target_weight"] = display["target_weight"].map(float)
    group_weights = display.groupby("issuer_group")["target_weight"].sum()
    sector_weights = display.groupby("sector")["target_weight"].sum()

    print()
    print("DAILY CONSTRAINED TARGET PREVIEW PASSED")
    print("=" * 80)
    print(f"Signal date: {validation.timing.signal_date}")
    print(
        "Intended execution date: "
        f"{validation.timing.intended_execution_date}"
    )
    print(f"Selected holdings: {len(display)}")
    print(f"Invested weight: {display['target_weight'].sum():.6f}")
    print(f"Cash buffer: {1.0 - display['target_weight'].sum():.6f}")
    print(f"Maximum single-name weight: {display['target_weight'].max():.6f}")
    print(f"Maximum issuer-family weight: {group_weights.max():.6f}")
    print(f"Maximum risk-sector weight: {sector_weights.max():.6f}")

    if targets.capacity_replacements:
        print("Capacity-driven ranking replacements:")
        for removed, added in targets.capacity_replacements:
            print(f"- Removed {removed}; added {added}")
    else:
        print("Capacity-driven ranking replacements: none")

    print()
    print(
        display[
            [
                "predicted_rank",
                "ticker",
                "issuer_group",
                "sector",
                "score",
                "target_weight",
            ]
        ]
        .round({"score": 8, "target_weight": 6})
        .to_string(index=False)
    )
    print()
    print("This is a target-weight preview only.")
    print("No ledgers, paper orders, or real orders were written.")
    print()


if __name__ == "__main__":
    main()
