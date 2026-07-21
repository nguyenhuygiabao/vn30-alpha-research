from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable, Mapping

from src.paper_trading.calendar import DateLike, normalize_date
from src.paper_trading.schemas import OrderStatus, Side
from src.paper_trading.settlement import ExecutionRecord, to_decimal


@dataclass(frozen=True)
class PendingOrder:
    order_id: str
    intended_execution_date: date
    ticker: str
    side: Side
    requested_quantity: int
    status: OrderStatus

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "PendingOrder":
        return cls(
            order_id=str(row["order_id"]).strip(),
            intended_execution_date=normalize_date(
                str(row["intended_execution_date"])
            ),
            ticker=str(row["ticker"]).strip().upper(),
            side=Side(str(row["side"])),
            requested_quantity=int(row["requested_quantity"]),
            status=OrderStatus(str(row["status"])),
        )


@dataclass(frozen=True)
class ExecutionCostRates:
    commission_rate: Decimal
    slippage_rate: Decimal
    sell_tax_rate: Decimal

    @classmethod
    def from_values(
        cls,
        commission_rate: Decimal | int | float | str,
        slippage_rate: Decimal | int | float | str,
        sell_tax_rate: Decimal | int | float | str,
    ) -> "ExecutionCostRates":
        rates = cls(
            commission_rate=to_decimal(commission_rate),
            slippage_rate=to_decimal(slippage_rate),
            sell_tax_rate=to_decimal(sell_tax_rate),
        )

        if any(
            rate < 0
            for rate in (
                rates.commission_rate,
                rates.slippage_rate,
                rates.sell_tax_rate,
            )
        ):
            raise ValueError("Execution cost rates cannot be negative")

        return rates


def build_execution_records(
    *,
    order_rows: Iterable[Mapping[str, object]],
    execution_date: DateLike,
    open_prices: Mapping[str, Decimal | int | float | str],
    cost_rates: ExecutionCostRates,
) -> list[ExecutionRecord]:
    target_date = normalize_date(execution_date)
    normalized_prices = {
        ticker.strip().upper(): to_decimal(price)
        for ticker, price in open_prices.items()
    }

    executions: list[ExecutionRecord] = []
    seen_order_ids: set[str] = set()

    for row in order_rows:
        order = PendingOrder.from_row(row)

        if order.status != OrderStatus.PENDING:
            continue

        if order.intended_execution_date != target_date:
            continue

        if not order.order_id:
            raise ValueError("Pending order ID cannot be empty")

        if order.order_id in seen_order_ids:
            raise ValueError(f"Duplicate pending order: {order.order_id}")

        if order.requested_quantity <= 0:
            raise ValueError(
                f"Pending order quantity must be positive: {order.order_id}"
            )

        price = normalized_prices.get(order.ticker)

        if price is None or price <= 0:
            raise ValueError(
                f"Missing valid execution price for {order.ticker}"
            )

        gross_value = price * order.requested_quantity
        commission = gross_value * cost_rates.commission_rate
        slippage = gross_value * cost_rates.slippage_rate
        tax = (
            gross_value * cost_rates.sell_tax_rate
            if order.side == Side.SELL
            else Decimal("0")
        )

        executions.append(
            ExecutionRecord(
                execution_id=f"execution-{order.order_id}",
                order_id=order.order_id,
                execution_date=target_date,
                ticker=order.ticker,
                side=order.side,
                filled_quantity=order.requested_quantity,
                execution_price=price,
                gross_value=gross_value,
                commission=commission,
                tax=tax,
                slippage=slippage,
            )
        )
        seen_order_ids.add(order.order_id)

    return executions


EXECUTION_ROLLBACK_FILES = (
    "orders.csv",
    "executions.csv",
    "positions.csv",
    "cash_ledger.csv",
    "settlement_ledger.csv",
)


def record_execution_batch(
    *,
    storage: object,
    broker: object,
    executions: Iterable[ExecutionRecord],
    calendar: object,
    issuer_groups: Mapping[str, str],
    settlement_lag_trading_days: int,
    mark_prices: Mapping[str, Decimal | int | float | str] | None = None,
) -> bool:
    execution_list = list(executions)

    if not execution_list:
        return False

    execution_ids = [execution.execution_id for execution in execution_list]
    order_ids = [execution.order_id for execution in execution_list]

    if len(execution_ids) != len(set(execution_ids)):
        raise ValueError("Duplicate execution IDs in execution batch")

    if len(order_ids) != len(set(order_ids)):
        raise ValueError("Duplicate order IDs in execution batch")

    with storage.account_lock():
        storage.validate_all_ledgers()

        originals = {
            filename: storage.read_ledger(filename)
            for filename in EXECUTION_ROLLBACK_FILES
        }

        existing_executions = originals["executions.csv"]

        if not existing_executions.empty:
            existing_ids = set(existing_executions["execution_id"])

            if set(execution_ids).issubset(existing_ids):
                return False

            if set(execution_ids) & existing_ids:
                raise RuntimeError(
                    "Partial or conflicting execution batch already exists"
                )

        orders = originals["orders.csv"].copy()
        matching = orders["order_id"].isin(order_ids)

        if int(matching.sum()) != len(order_ids):
            missing = sorted(
                set(order_ids).difference(set(orders.loc[matching, "order_id"]))
            )
            raise ValueError(f"Execution references missing orders: {missing}")

        invalid_status = orders.loc[
            matching & (orders["status"] != OrderStatus.PENDING.value),
            ["order_id", "status"],
        ]

        if not invalid_status.empty:
            raise ValueError(
                "Execution requires PENDING orders: "
                f"{invalid_status.to_dict(orient='records')}"
            )

        try:
            for execution in execution_list:
                broker.apply_execution(
                    execution,
                    calendar,
                    issuer_group=issuer_groups.get(execution.ticker, ""),
                    settlement_lag_trading_days=(
                        settlement_lag_trading_days
                    ),
                )

            storage.append_rows(
                "executions.csv",
                [execution.to_row() for execution in execution_list],
                unique_by=("execution_id",),
            )

            orders.loc[matching, "status"] = OrderStatus.EXECUTED.value
            storage.replace_rows(
                "orders.csv",
                orders.to_dict(orient="records"),
            )

            storage.save_broker_state(
                broker,
                mark_prices=mark_prices,
            )
        except Exception:
            for filename, frame in originals.items():
                storage.replace_rows(
                    filename,
                    frame.to_dict(orient="records"),
                )
            raise

    return True
