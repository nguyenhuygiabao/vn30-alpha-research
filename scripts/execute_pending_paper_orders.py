from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.data_loader import load_ohlcv_csv
from src.paper_trading.calendar import TradingCalendar, normalize_date
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.paper_execution import (
    ExecutionCostRates,
    build_execution_records,
    record_execution_batch,
)
from src.paper_trading.storage import PaperAccountStorage


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or record due VN30 paper-order executions.",
    )
    parser.add_argument(
        "--config",
        default="config/paper_trading_config.yaml",
    )
    parser.add_argument(
        "--execution-date",
        required=True,
        help="Trading date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Record executions and update the paper account.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    execution_date = normalize_date(args.execution_date)
    config = load_paper_trading_config(args.config)

    storage = PaperAccountStorage(config["output"]["directory"])
    storage.validate_all_ledgers()

    orders = storage.read_ledger("orders.csv")
    due = orders[
        (orders["status"] == "PENDING")
        & (
            orders["intended_execution_date"]
            == execution_date.isoformat()
        )
    ].copy()

    print()
    print("VN30 PAPER EXECUTION")
    print("=" * 80)
    print(f"Execution date: {execution_date}")
    print(f"Mode: {'record' if args.write else 'preview'}")
    print(f"Due pending orders: {len(due)}")

    if due.empty:
        print("No pending orders are due.")
        return 0

    market_data = load_ohlcv_csv(
        config["model"]["raw_ohlcv_path"]
    )
    market_data["date"] = pd.to_datetime(
        market_data["date"]
    ).dt.date

    execution_rows = market_data[
        market_data["date"] == execution_date
    ].copy()

    if execution_rows.empty:
        print()
        print(
            "EXECUTION BLOCKED: market data for the execution "
            f"date is unavailable: {execution_date}"
        )
        return 1

    duplicate_tickers = execution_rows["ticker"].duplicated()

    if duplicate_tickers.any():
        duplicates = sorted(
            execution_rows.loc[
                duplicate_tickers,
                "ticker",
            ].astype(str)
        )
        print(
            "EXECUTION BLOCKED: duplicate ticker rows for "
            f"{execution_date}: {duplicates}"
        )
        return 1

    multiplier = Decimal(
        str(config["market_data"]["price_multiplier_to_vnd"])
    )

    open_prices = {
        str(row.ticker).strip().upper(): (
            Decimal(str(row.open)) * multiplier
        )
        for row in execution_rows.itertuples(index=False)
    }

    due_tickers = set(due["ticker"].str.upper())
    missing_tickers = sorted(
        due_tickers.difference(open_prices)
    )

    if missing_tickers:
        print(
            "EXECUTION BLOCKED: missing opening prices for "
            f"{missing_tickers}"
        )
        return 1

    execution_config = config["execution"]
    rates = ExecutionCostRates.from_values(
        commission_rate=execution_config["commission_rate"],
        slippage_rate=execution_config["slippage_rate"],
        sell_tax_rate=execution_config["sell_tax_rate"],
    )

    executions = build_execution_records(
        order_rows=due.to_dict(orient="records"),
        execution_date=execution_date,
        open_prices=open_prices,
        cost_rates=rates,
    )

    preview = pd.DataFrame(
        execution.to_row()
        for execution in executions
    )

    print()
    print(
        preview[
            [
                "ticker",
                "side",
                "filled_quantity",
                "execution_price",
                "gross_value",
                "commission",
                "tax",
                "slippage",
                "net_cash_effect",
            ]
        ].to_string(index=False)
    )

    total_gross = sum(
        (execution.gross_value for execution in executions),
        start=Decimal("0"),
    )
    total_costs = sum(
        (execution.total_costs for execution in executions),
        start=Decimal("0"),
    )

    print()
    print(f"Execution count: {len(executions)}")
    print(f"Gross traded value: {total_gross:,.0f} VND")
    print(f"Estimated costs: {total_costs:,.0f} VND")

    if not args.write:
        print("Preview only. No account files were changed.")
        return 0

    universe = pd.read_csv(config["model"]["universe_path"])
    issuer_groups = {
        str(row.ticker).strip().upper(): str(
            row.issuer_group
        ).strip()
        for row in universe.itertuples(index=False)
    }

    calendar = TradingCalendar.from_weekdays(
        execution_date - timedelta(days=30),
        execution_date + timedelta(days=60),
        holidays=config["market_calendar"]["holiday_dates"],
    )

    broker = storage.load_broker_state()

    recorded = record_execution_batch(
        storage=storage,
        broker=broker,
        executions=executions,
        calendar=calendar,
        issuer_groups=issuer_groups,
        settlement_lag_trading_days=int(
            config["settlement"]["lag_trading_days"]
        ),
        mark_prices=open_prices,
    )

    print()
    if recorded:
        print("PAPER EXECUTIONS RECORDED SUCCESSFULLY")
    else:
        print("EXECUTIONS ALREADY RECORDED; NO DUPLICATES ADDED")

    print("No real broker orders were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
