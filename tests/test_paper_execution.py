from __future__ import annotations

from decimal import Decimal

import pytest

from src.paper_trading.paper_execution import (
    ExecutionCostRates,
    build_execution_records,
)


def pending_order(
    *,
    order_id: str = "order-1",
    ticker: str = "FPT",
    side: str = "BUY",
    quantity: int = 100,
    execution_date: str = "2026-07-22",
    status: str = "PENDING",
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "intended_execution_date": execution_date,
        "ticker": ticker,
        "side": side,
        "requested_quantity": quantity,
        "status": status,
    }


def rates() -> ExecutionCostRates:
    return ExecutionCostRates.from_values(
        commission_rate="0.001",
        slippage_rate="0.001",
        sell_tax_rate="0.001",
    )


def test_build_buy_execution_from_next_open() -> None:
    executions = build_execution_records(
        order_rows=[pending_order()],
        execution_date="2026-07-22",
        open_prices={"FPT": "100000"},
        cost_rates=rates(),
    )

    execution = executions[0]

    assert execution.execution_price == Decimal("100000")
    assert execution.gross_value == Decimal("10000000")
    assert execution.commission == Decimal("10000")
    assert execution.slippage == Decimal("10000")
    assert execution.tax == Decimal("0")
    assert execution.net_cash_effect == Decimal("-10020000")


def test_sell_execution_includes_tax() -> None:
    executions = build_execution_records(
        order_rows=[
            pending_order(
                side="SELL",
                quantity=200,
            )
        ],
        execution_date="2026-07-22",
        open_prices={"FPT": "50000"},
        cost_rates=rates(),
    )

    execution = executions[0]

    assert execution.gross_value == Decimal("10000000")
    assert execution.commission == Decimal("10000")
    assert execution.slippage == Decimal("10000")
    assert execution.tax == Decimal("10000")
    assert execution.net_cash_effect == Decimal("9970000")


def test_ignores_orders_not_due_for_execution() -> None:
    executions = build_execution_records(
        order_rows=[
            pending_order(execution_date="2026-07-23"),
            pending_order(order_id="order-2", status="HOLD"),
        ],
        execution_date="2026-07-22",
        open_prices={"FPT": "100000"},
        cost_rates=rates(),
    )

    assert executions == []


def test_missing_open_price_fails() -> None:
    with pytest.raises(
        ValueError,
        match="Missing valid execution price for FPT",
    ):
        build_execution_records(
            order_rows=[pending_order()],
            execution_date="2026-07-22",
            open_prices={},
            cost_rates=rates(),
        )


def test_duplicate_pending_order_fails() -> None:
    order = pending_order()

    with pytest.raises(
        ValueError,
        match="Duplicate pending order",
    ):
        build_execution_records(
            order_rows=[order, order],
            execution_date="2026-07-22",
            open_prices={"FPT": "100000"},
            cost_rates=rates(),
        )


from pathlib import Path

import pandas as pd

from src.paper_trading.broker_state import PaperBrokerState
from src.paper_trading.calendar import TradingCalendar
from src.paper_trading.paper_execution import record_execution_batch
from src.paper_trading.schemas import LEDGER_SCHEMAS
from src.paper_trading.storage import PaperAccountStorage


def initialize_storage(directory: Path) -> PaperAccountStorage:
    storage = PaperAccountStorage(directory)
    storage.initialize(
        initial_cash="100000000",
        asof_date="2026-07-21",
    )

    order = {
        column: ""
        for column in LEDGER_SCHEMAS["orders.csv"]
    }
    order.update({
        "order_id": "order-1",
        "signal_id": "signal-1",
        "order_date": "2026-07-21",
        "intended_execution_date": "2026-07-22",
        "ticker": "FPT",
        "side": "BUY",
        "requested_quantity": "100",
        "estimated_price": "100000",
        "requested_value": "10000000",
        "status": "PENDING",
        "reason_code": "",
        "created_at": "2026-07-21T08:00:00+00:00",
    })
    storage.append_rows(
        "orders.csv",
        [order],
        unique_by=("order_id",),
    )

    return storage


def test_record_execution_batch_updates_account_atomically(
    tmp_path,
) -> None:
    storage = initialize_storage(tmp_path)
    broker = storage.load_broker_state()
    calendar = TradingCalendar.from_weekdays(
        "2026-07-20",
        "2026-07-31",
    )

    executions = build_execution_records(
        order_rows=storage.read_ledger(
            "orders.csv"
        ).to_dict(orient="records"),
        execution_date="2026-07-22",
        open_prices={"FPT": "100000"},
        cost_rates=rates(),
    )

    recorded = record_execution_batch(
        storage=storage,
        broker=broker,
        executions=executions,
        calendar=calendar,
        issuer_groups={"FPT": "FPT"},
        settlement_lag_trading_days=2,
        mark_prices={"FPT": "100000"},
    )

    assert recorded is True

    orders = storage.read_ledger("orders.csv")
    execution_rows = storage.read_ledger("executions.csv")
    positions = storage.read_ledger("positions.csv")
    settlements = storage.read_ledger("settlement_ledger.csv")

    assert orders.loc[0, "status"] == "EXECUTED"
    assert len(execution_rows) == 1
    assert positions.loc[0, "unsettled_buy_shares"] == "100"
    assert settlements.loc[0, "status"] == "PENDING"
    assert settlements.loc[0, "settlement_date"] == "2026-07-24"


def test_record_execution_batch_is_idempotent(tmp_path) -> None:
    storage = initialize_storage(tmp_path)
    broker = storage.load_broker_state()
    calendar = TradingCalendar.from_weekdays(
        "2026-07-20",
        "2026-07-31",
    )

    executions = build_execution_records(
        order_rows=storage.read_ledger(
            "orders.csv"
        ).to_dict(orient="records"),
        execution_date="2026-07-22",
        open_prices={"FPT": "100000"},
        cost_rates=rates(),
    )

    assert record_execution_batch(
        storage=storage,
        broker=broker,
        executions=executions,
        calendar=calendar,
        issuer_groups={"FPT": "FPT"},
        settlement_lag_trading_days=2,
    )

    reloaded = storage.load_broker_state()

    assert record_execution_batch(
        storage=storage,
        broker=reloaded,
        executions=executions,
        calendar=calendar,
        issuer_groups={"FPT": "FPT"},
        settlement_lag_trading_days=2,
    ) is False

    assert len(storage.read_ledger("executions.csv")) == 1
