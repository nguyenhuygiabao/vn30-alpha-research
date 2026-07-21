from __future__ import annotations

import argparse
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_ohlcv_csv
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.daily_exports import write_daily_exports
from src.paper_trading.execution_rules import ExecutionConstraints, MarketSnapshot
from src.paper_trading.market_data import (
    load_universe_tickers,
    validate_completed_market_data,
)
from src.paper_trading.order_sizing import TargetWeight, build_order_plan
from src.paper_trading.plan_recording import record_daily_paper_plan
from src.paper_trading.scoring import (
    build_signal_ledger_rows,
    score_completed_market_data,
)
from src.paper_trading.storage import PaperAccountStorage
from src.paper_trading.targets import build_constrained_target_weights
from src.price_limit import add_estimated_price_limits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview account-aware VN30 paper orders."
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
    )
    parser.add_argument(
        "--model",
        default="rank_ensemble",
        choices=["gradient_boosting", "random_forest", "rank_ensemble"],
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the validated plan to local paper ledgers.",
    )
    return parser.parse_args()


def main():
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
        model_name=args.model,
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
        raise ValueError("Targets do not match validated signal timing")

    storage = PaperAccountStorage(config["output"]["directory"])
    storage.validate_all_ledgers()
    broker = storage.load_broker_state()

    # Settlement is calculated in memory only during this preview.
    broker.settle_due(validation.timing.signal_date)

    multiplier = Decimal(
        str(config["market_data"]["price_multiplier_to_vnd"])
    )

    working = market_data.copy()
    working = working.sort_values(["ticker", "date"])
    working["value_vnd"] = (
        working["close"] * working["volume"] * float(multiplier)
    )
    working["adv20_vnd"] = working.groupby("ticker")["value_vnd"].transform(
        lambda values: values.rolling(20, min_periods=1).mean()
    )
    working = add_estimated_price_limits(working)

    latest = working[
        pd.to_datetime(working["date"]).dt.date
        == validation.timing.data_asof_date
    ].copy()

    prediction_tickers = set(scoring.predictions["ticker"])
    snapshots = {}

    for row in latest.itertuples(index=False):
        snapshots[row.ticker] = MarketSnapshot(
            ticker=row.ticker,
            data_date=validation.timing.data_asof_date,
            price=Decimal(str(row.close)) * multiplier,
            average_daily_value=Decimal(str(row.adv20_vnd)),
            prediction_available=row.ticker in prediction_tickers,
            at_ceiling=bool(row.close_at_ceiling_today),
            at_floor=bool(row.close_at_floor_today),
        )

    target_objects = []

    for row in targets.target_weights.itertuples(index=False):
        signal_id = (
            f"signal-{validation.timing.signal_date:%Y%m%d}-"
            f"{args.model}-{row.horizon_days}d-{row.ticker}"
        )
        target_objects.append(
            TargetWeight(
                ticker=row.ticker,
                issuer_group=row.issuer_group,
                sector=row.sector,
                target_weight=row.target_weight,
                signal_id=signal_id,
                predicted_rank=int(row.predicted_rank),
            )
        )

    plan = build_order_plan(
        broker=broker,
        targets=target_objects,
        snapshots=snapshots,
        constraints=ExecutionConstraints.from_config(config),
        signal_date=validation.timing.signal_date,
        intended_execution_date=validation.timing.intended_execution_date,
        expected_data_date=validation.timing.data_asof_date,
    )

    executable = pd.DataFrame(plan.order_rows())
    executable = executable[executable["status"] == "PENDING"]

    print()
    print("DAILY PAPER ORDER PREVIEW PASSED")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Signal date: {validation.timing.signal_date}")
    print(
        "Intended execution date: "
        f"{validation.timing.intended_execution_date}"
    )
    print(f"Paper portfolio value: {plan.portfolio_value:,.0f} VND")
    print(f"Executable paper orders: {len(plan.executable_orders)}")
    print(f"Skipped/deferred trades: {len(plan.skipped_trades)}")
    print(f"Estimated turnover: {plan.estimated_turnover:.4%}")
    print(
        "Spendable cash after proposed orders: "
        f"{plan.spendable_cash_after_orders:,.0f} VND"
    )

    if executable.empty:
        print("\nNo executable orders.")
    else:
        print()
        print(
            executable[
                [
                    "ticker",
                    "side",
                    "requested_quantity",
                    "estimated_price",
                    "requested_value",
                ]
            ].to_string(index=False)
        )

    if plan.skipped_trades:
        reasons = pd.Series(
            [trade.reason_code.value for trade in plan.skipped_trades]
        ).value_counts()
        print("\nSkipped/deferred reasons:")
        print(reasons.to_string())

    if args.write:
        signal_rows = build_signal_ledger_rows(
            result=scoring,
            data_asof_date=validation.timing.data_asof_date,
            intended_execution_date=(
                validation.timing.intended_execution_date
            ),
            created_at=generated_at,
        )

        portfolio_id = (
            f"portfolio-{validation.timing.signal_date:%Y%m%d}-"
            f"{args.model}-"
            f"{config['timing']['signal_horizon_trading_days']}d"
        )

        target_rows = []

        for row in targets.target_weights.itertuples(index=False):
            target_rows.append({
                "portfolio_id": portfolio_id,
                "signal_date": (
                    validation.timing.signal_date.isoformat()
                ),
                "intended_execution_date": (
                    validation.timing.intended_execution_date.isoformat()
                ),
                "ticker": row.ticker,
                "issuer_group": row.issuer_group,
                "target_weight": str(row.target_weight),
                "rebalance_flag": "true",
                "created_at": generated_at.isoformat(),
            })

        recorded = record_daily_paper_plan(
            storage=storage,
            broker=broker,
            rows_by_ledger={
                "signals.csv": signal_rows,
                "target_weights.csv": target_rows,
                "orders.csv": plan.order_rows(),
                "skipped_trades.csv": plan.skipped_trade_rows(),
            },
        )

        print()
        if recorded:
            print("PAPER PLAN RECORDED SUCCESSFULLY")
        else:
            print("PAPER PLAN ALREADY RECORDED; NO DUPLICATES ADDED")

        export_paths = write_daily_exports(
            output_directory=ROOT / "reports" / "daily",
            model_name=args.model,
            signal_date=validation.timing.signal_date,
            intended_execution_date=(
                validation.timing.intended_execution_date
            ),
            targets=targets.target_weights,
            plan=plan,
            broker=broker,
            snapshots=snapshots,
        )

        print("Daily snapshot files:")
        for path in export_paths.values():
            print(f"- {path.relative_to(ROOT)}")

        print("No real orders were submitted.")
    else:
        print()
        print("Preview only. No ledger rows or real orders were written.")


if __name__ == "__main__":
    main()
