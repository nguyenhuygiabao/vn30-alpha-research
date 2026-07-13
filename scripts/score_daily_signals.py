from __future__ import annotations

import argparse
import sys
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path
from zoneinfo import ZoneInfo


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the selected model and preview latest VN30 signal ranks.",
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
        help="Path to the paper-trading YAML configuration.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Temporary scoring-model override, for example gradient_boosting, "
            "random_forest, or rank_ensemble. Does not modify configuration."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = load_paper_trading_config(args.config)
    timezone = ZoneInfo(config["timing"]["timezone"])
    generated_at = datetime.now(timezone)
    market_data = load_ohlcv_csv(config["model"]["raw_ohlcv_path"])
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
    selected_model = args.model or config["model"]["model_name"]
    result = score_completed_market_data(
        market_data=market_data,
        expected_tickers=expected_tickers,
        horizon_days=config["timing"]["signal_horizon_trading_days"],
        model_name=selected_model,
    )

    if result.signal_date != validation.timing.signal_date:
        raise ValueError("Model signal date does not match validated market timing")

    top_count = config["portfolio"]["target_holdings"]
    top_predictions = result.predictions.head(top_count).copy()

    print()
    print("DAILY ML SIGNAL SCORING PASSED")
    print("=" * 80)
    print(f"Data as of date: {validation.timing.data_asof_date}")
    print(f"Signal date: {result.signal_date}")
    print(
        "Intended execution date: "
        f"{validation.timing.intended_execution_date}"
    )
    print(f"Model: {selected_model}")
    print(
        "Forecast horizon: "
        f"{config['timing']['signal_horizon_trading_days']} trading days"
    )
    print(
        f"Training window: {result.training_start_date} "
        f"to {result.training_end_date}"
    )
    print(f"Training dates: {result.training_date_count}")
    print(f"Training rows: {result.training_row_count}")
    print(f"Model features: {len(result.feature_columns)}")
    print(f"Scored tickers: {len(result.predictions)}")
    print()
    print(f"Top {top_count} ranking preview:")
    print(
        top_predictions[["predicted_rank", "ticker", "score"]]
        .round({"score": 8})
        .to_string(index=False)
    )
    print()
    print("This is a scoring preview only.")
    print("No signal ledger, target weights, paper orders, or real orders were written.")
    print()


if __name__ == "__main__":
    main()
