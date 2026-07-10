from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from src.paper_trading.broker_state import PaperBrokerState, PositionState
from src.paper_trading.calendar import TradingCalendar
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.execution_rules import (
    ExecutionConstraints,
    MarketSnapshot,
)
from src.paper_trading.order_sizing import TargetWeight, build_order_plan
from src.paper_trading.schemas import (
    LEDGER_SCHEMAS,
    OrderAction,
    OrderStatus,
    Side,
    SkipReason,
)
from src.paper_trading.settlement import ExecutionRecord


DATA_DATE = "2026-07-10"
EXECUTION_DATE = "2026-07-13"


def constraints() -> ExecutionConstraints:
    return ExecutionConstraints.from_config(load_paper_trading_config())


def snapshot(
    ticker: str,
    price: str = "10000",
    adv: str | None = "1000000000",
    data_date: str = DATA_DATE,
    prediction_available: bool = True,
    at_ceiling: bool = False,
    at_floor: bool = False,
) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        data_date=data_date,
        price=price,
        average_daily_value=adv,
        prediction_available=prediction_available,
        at_ceiling=at_ceiling,
        at_floor=at_floor,
    )


def build_cash_broker() -> PaperBrokerState:
    return PaperBrokerState.initialize("100000000", DATA_DATE)


def test_basic_target_becomes_executable_buy_order() -> None:
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.10", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT")},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert len(plan.executable_orders) == 1
    order = plan.executable_orders[0]
    assert order.action == OrderAction.BUY
    assert order.requested_quantity == 1000
    assert order.requested_value == Decimal("10000000")
    assert order.status == OrderStatus.PENDING
    assert plan.estimated_turnover == Decimal("0.05")
    assert plan.skipped_trades == []
    assert list(order.to_row()) == list(LEDGER_SCHEMAS["orders.csv"])


def test_single_name_cap_creates_executable_and_skipped_portions() -> None:
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.20", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT")},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders[0].requested_quantity == 1500
    assert any(
        trade.reason_code == SkipReason.MAX_SINGLE_NAME_WEIGHT_REACHED
        and trade.requested_quantity == 500
        for trade in plan.skipped_trades
    )


def test_issuer_group_cap_scales_related_tickers() -> None:
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[
            TargetWeight("VHM", "Vingroup", "0.15", "signal-vhm", 1),
            TargetWeight("VIC", "Vingroup", "0.15", "signal-vic", 2),
        ],
        snapshots={
            "VHM": snapshot("VHM"),
            "VIC": snapshot("VIC"),
        },
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    quantities = {
        order.ticker: order.requested_quantity
        for order in plan.executable_orders
    }
    assert quantities == {"VHM": 1250, "VIC": 1250}
    assert sum(plan.constrained_target_weights.values()) == Decimal("0.25")
    assert sum(
        trade.requested_quantity
        for trade in plan.skipped_trades
        if trade.reason_code == SkipReason.MAX_ISSUER_GROUP_WEIGHT_REACHED
    ) == 500


def test_turnover_cap_defers_part_of_rebalance() -> None:
    custom = replace(
        constraints(),
        max_single_name_weight=Decimal("1"),
        max_issuer_group_weight=Decimal("1"),
        max_daily_turnover=Decimal("0.10"),
    )
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.50", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT")},
        constraints=custom,
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders[0].requested_value == Decimal("20000000")
    assert plan.estimated_turnover == Decimal("0.10")
    assert any(
        trade.reason_code == SkipReason.MAX_TURNOVER_REACHED
        and trade.requested_value == Decimal("30000000")
        for trade in plan.skipped_trades
    )


def test_adv_limit_creates_partial_order_and_skip_record() -> None:
    custom = replace(
        constraints(),
        max_single_name_weight=Decimal("1"),
        max_issuer_group_weight=Decimal("1"),
    )
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.20", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT", adv="100000000")},
        constraints=custom,
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders[0].requested_quantity == 500
    skipped = [
        trade
        for trade in plan.skipped_trades
        if trade.reason_code == SkipReason.ADV_CAPACITY_LIMIT
    ]
    assert len(skipped) == 1
    assert skipped[0].requested_quantity == 1500
    assert list(skipped[0].to_row()) == list(
        LEDGER_SCHEMAS["skipped_trades.csv"]
    )


def test_cash_buffer_and_costs_prevent_overspending() -> None:
    custom = replace(
        constraints(),
        max_single_name_weight=Decimal("1"),
        max_issuer_group_weight=Decimal("1"),
        max_daily_turnover=Decimal("1"),
    )
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.97", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT", adv="100000000000")},
        constraints=custom,
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    order = plan.executable_orders[0]
    estimated_outflow = -custom.estimated_cash_effect(
        OrderAction.BUY,
        order.requested_value,
    )
    assert estimated_outflow <= Decimal("97000000")
    assert plan.spendable_cash_after_orders >= Decimal("0")
    assert any(
        trade.reason_code == SkipReason.INSUFFICIENT_SETTLED_CASH
        for trade in plan.skipped_trades
    )


def test_buy_at_ceiling_is_skipped() -> None:
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.10", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT", at_ceiling=True)},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders == []
    assert plan.skipped_trades[0].reason_code == (
        SkipReason.PRICE_CEILING_BUY_BLOCK
    )


def test_sell_at_floor_is_skipped() -> None:
    broker = PaperBrokerState(settled_cash="90000000", asof_date=DATA_DATE)
    broker.positions["FPT"] = PositionState(
        ticker="FPT",
        issuer_group="FPT",
        settled_shares=1000,
        average_cost=Decimal("10000"),
    )
    broker.reconcile()
    plan = build_order_plan(
        broker=broker,
        targets=[],
        snapshots={"FPT": snapshot("FPT", at_floor=True)},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders == []
    assert plan.skipped_trades[0].reason_code == (
        SkipReason.PRICE_FLOOR_SELL_BLOCK
    )


def test_stale_data_and_missing_prediction_are_skipped() -> None:
    stale_plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.10", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT", data_date="2026-07-09")},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )
    missing_prediction_plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.10", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT", prediction_available=False)},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert stale_plan.skipped_trades[0].reason_code == SkipReason.STALE_DATA
    assert missing_prediction_plan.skipped_trades[0].reason_code == (
        SkipReason.MISSING_PREDICTION
    )


def test_missing_adv_blocks_trade() -> None:
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.10", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT", adv=None)},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders == []
    assert plan.skipped_trades[0].reason_code == SkipReason.ADV_CAPACITY_LIMIT


def test_no_weight_change_creates_hold_row() -> None:
    broker = PaperBrokerState(settled_cash="90000000", asof_date=DATA_DATE)
    broker.positions["FPT"] = PositionState(
        ticker="FPT",
        issuer_group="FPT",
        settled_shares=1000,
        average_cost=Decimal("10000"),
    )
    broker.reconcile()
    plan = build_order_plan(
        broker=broker,
        targets=[TargetWeight("FPT", "FPT", "0.10", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT")},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert len(plan.orders) == 1
    assert plan.orders[0].action == OrderAction.HOLD
    assert plan.orders[0].status == OrderStatus.HOLD


def test_round_lot_rule_records_odd_lot_remainder() -> None:
    custom = replace(
        constraints(),
        allow_odd_lots=False,
        max_single_name_weight=Decimal("1"),
        max_issuer_group_weight=Decimal("1"),
    )
    plan = build_order_plan(
        broker=build_cash_broker(),
        targets=[TargetWeight("FPT", "FPT", "0.015", "signal-fpt", 1)],
        snapshots={"FPT": snapshot("FPT")},
        constraints=custom,
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders[0].requested_quantity == 100
    assert any(
        trade.reason_code == SkipReason.BELOW_MIN_TRADE_VALUE
        and trade.requested_quantity == 50
        for trade in plan.skipped_trades
    )


def test_unsettled_bought_shares_are_excluded_from_sell_order() -> None:
    calendar = TradingCalendar.from_weekdays("2026-07-01", "2026-07-31")
    broker = PaperBrokerState.initialize("100000000", "2026-07-06")
    first_buy = ExecutionRecord(
        execution_id="execution-first-buy",
        order_id="order-first-buy",
        execution_date="2026-07-06",
        ticker="FPT",
        side=Side.BUY,
        filled_quantity=100,
        execution_price="10000",
    )
    second_buy = ExecutionRecord(
        execution_id="execution-second-buy",
        order_id="order-second-buy",
        execution_date="2026-07-09",
        ticker="FPT",
        side=Side.BUY,
        filled_quantity=100,
        execution_price="10000",
    )
    broker.apply_execution(first_buy, calendar)
    broker.settle_due("2026-07-08")
    broker.apply_execution(second_buy, calendar)
    plan = build_order_plan(
        broker=broker,
        targets=[],
        snapshots={"FPT": snapshot("FPT")},
        constraints=constraints(),
        signal_date=DATA_DATE,
        intended_execution_date=EXECUTION_DATE,
        expected_data_date=DATA_DATE,
    )

    assert plan.executable_orders[0].action == OrderAction.SELL
    assert plan.executable_orders[0].requested_quantity == 100
    assert any(
        trade.reason_code == SkipReason.INSUFFICIENT_SELLABLE_QUANTITY
        and trade.requested_quantity == 100
        for trade in plan.skipped_trades
    )
