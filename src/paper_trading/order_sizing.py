from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Iterable, Mapping

from src.paper_trading.broker_state import PaperBrokerState
from src.paper_trading.calendar import DateLike, normalize_date
from src.paper_trading.execution_rules import (
    ExecutionConstraints,
    MarketSnapshot,
    trade_block_reason,
)
from src.paper_trading.schemas import OrderAction, OrderStatus, SkipReason
from src.paper_trading.settlement import ZERO, to_decimal


@dataclass(frozen=True)
class TargetWeight:
    ticker: str
    issuer_group: str
    target_weight: Decimal | int | float | str
    signal_id: str = ""
    predicted_rank: int | None = None

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        target_weight = to_decimal(self.target_weight)

        if not ticker:
            raise ValueError("ticker cannot be empty")

        if not ZERO <= target_weight <= Decimal("1"):
            raise ValueError("target_weight must be between 0 and 1")

        if self.predicted_rank is not None and self.predicted_rank <= 0:
            raise ValueError("predicted_rank must be positive")

        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "target_weight", target_weight)


@dataclass(frozen=True)
class PlannedOrder:
    order_id: str
    signal_id: str
    order_date: date
    intended_execution_date: date
    ticker: str
    action: OrderAction
    requested_quantity: int
    estimated_price: Decimal
    requested_value: Decimal
    status: OrderStatus
    reason_code: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_row(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "order_date": self.order_date.isoformat(),
            "intended_execution_date": self.intended_execution_date.isoformat(),
            "ticker": self.ticker,
            "side": self.action.value,
            "requested_quantity": self.requested_quantity,
            "estimated_price": str(self.estimated_price),
            "requested_value": str(self.requested_value),
            "status": self.status.value,
            "reason_code": self.reason_code,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SkippedTrade:
    skip_id: str
    order_id: str
    date: date
    ticker: str
    side: OrderAction
    requested_quantity: int
    requested_value: Decimal
    reason_code: SkipReason
    details: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_row(self) -> dict[str, object]:
        return {
            "skip_id": self.skip_id,
            "order_id": self.order_id,
            "date": self.date.isoformat(),
            "ticker": self.ticker,
            "side": self.side.value,
            "requested_quantity": self.requested_quantity,
            "requested_value": str(self.requested_value),
            "reason_code": self.reason_code.value,
            "details": self.details,
            "created_at": self.created_at,
        }


@dataclass
class OrderPlan:
    orders: list[PlannedOrder]
    skipped_trades: list[SkippedTrade]
    portfolio_value: Decimal
    estimated_turnover: Decimal
    spendable_cash_before_orders: Decimal
    spendable_cash_after_orders: Decimal
    constrained_target_weights: dict[str, Decimal]

    @property
    def executable_orders(self) -> list[PlannedOrder]:
        return [
            order
            for order in self.orders
            if order.status == OrderStatus.PENDING
            and order.action in {OrderAction.BUY, OrderAction.SELL}
        ]

    def order_rows(self) -> list[dict[str, object]]:
        return [order.to_row() for order in self.orders]

    def skipped_trade_rows(self) -> list[dict[str, object]]:
        return [trade.to_row() for trade in self.skipped_trades]


@dataclass(frozen=True)
class _TradeRequest:
    ticker: str
    action: OrderAction
    quantity: int
    price: Decimal
    signal_id: str
    predicted_rank: int | None

    @property
    def value(self) -> Decimal:
        return self.price * self.quantity


def _target_map(targets: Iterable[TargetWeight]) -> dict[str, TargetWeight]:
    mapped: dict[str, TargetWeight] = {}

    for target in targets:
        if target.ticker in mapped:
            raise ValueError(f"Duplicate target ticker: {target.ticker}")

        mapped[target.ticker] = target

    return mapped


def _issuer_group(target: TargetWeight) -> str:
    return target.issuer_group.strip() or target.ticker


def _cap_target_weights(
    targets: dict[str, TargetWeight],
    constraints: ExecutionConstraints,
) -> tuple[
    dict[str, Decimal],
    dict[str, list[tuple[SkipReason, Decimal]]],
]:
    maximum_invested_weight = Decimal("1") - constraints.cash_buffer_weight
    raw_total = sum(
        (target.target_weight for target in targets.values()),
        start=ZERO,
    )

    if raw_total > maximum_invested_weight + Decimal("1e-12"):
        raise ValueError(
            "Target weights exceed the configured invested-weight limit"
        )

    constrained: dict[str, Decimal] = {}
    rejected_weights: dict[str, list[tuple[SkipReason, Decimal]]] = {}

    for ticker, target in targets.items():
        capped = min(target.target_weight, constraints.max_single_name_weight)
        constrained[ticker] = capped

        if capped < target.target_weight:
            rejected_weights.setdefault(ticker, []).append(
                (
                    SkipReason.MAX_SINGLE_NAME_WEIGHT_REACHED,
                    target.target_weight - capped,
                )
            )

    group_members: dict[str, list[str]] = {}

    for ticker, target in targets.items():
        group_members.setdefault(_issuer_group(target), []).append(ticker)

    for tickers in group_members.values():
        group_total = sum((constrained[ticker] for ticker in tickers), start=ZERO)

        if group_total <= constraints.max_issuer_group_weight or group_total == ZERO:
            continue

        scale = constraints.max_issuer_group_weight / group_total

        for ticker in tickers:
            previous_weight = constrained[ticker]
            constrained[ticker] = previous_weight * scale
            rejected_weights.setdefault(ticker, []).append(
                (
                    SkipReason.MAX_ISSUER_GROUP_WEIGHT_REACHED,
                    previous_weight - constrained[ticker],
                )
            )

    return constrained, rejected_weights


def _portfolio_value(
    broker: PaperBrokerState,
    snapshots: Mapping[str, MarketSnapshot],
) -> Decimal:
    value = broker.settled_cash + broker.unsettled_cash

    for ticker, position in broker.positions.items():
        if position.economic_quantity == 0:
            continue

        snapshot = snapshots.get(ticker)

        if snapshot is None or snapshot.price is None or snapshot.price <= ZERO:
            raise ValueError(f"Missing price for held position: {ticker}")

        value += snapshot.price * position.economic_quantity

    if value <= ZERO:
        raise ValueError("Paper portfolio value must be positive")

    return value


def _turnover_constrained_weights(
    current_weights: dict[str, Decimal],
    target_weights: dict[str, Decimal],
    max_turnover: Decimal,
) -> tuple[dict[str, Decimal], bool]:
    tickers = set(current_weights).union(target_weights)
    turnover = Decimal("0.5") * sum(
        (
            abs(target_weights.get(ticker, ZERO) - current_weights.get(ticker, ZERO))
            for ticker in tickers
        ),
        start=ZERO,
    )

    if turnover <= max_turnover or turnover == ZERO:
        return {
            ticker: target_weights.get(ticker, ZERO)
            for ticker in tickers
        }, False

    scale = max_turnover / turnover

    return {
        ticker: current_weights.get(ticker, ZERO)
        + scale
        * (target_weights.get(ticker, ZERO) - current_weights.get(ticker, ZERO))
        for ticker in tickers
    }, True


def _desired_quantity(
    current_quantity: int,
    target_value: Decimal,
    price: Decimal,
) -> int:
    raw_quantity = target_value / price

    if raw_quantity >= current_quantity:
        return int(raw_quantity.to_integral_value(rounding=ROUND_FLOOR))

    return int(raw_quantity.to_integral_value(rounding=ROUND_CEILING))


def build_order_plan(
    broker: PaperBrokerState,
    targets: Iterable[TargetWeight],
    snapshots: Mapping[str, MarketSnapshot],
    constraints: ExecutionConstraints,
    signal_date: DateLike,
    intended_execution_date: DateLike,
    expected_data_date: DateLike,
) -> OrderPlan:
    order_date = normalize_date(signal_date)
    execution_date = normalize_date(intended_execution_date)

    if execution_date <= order_date:
        raise ValueError("intended_execution_date must be after signal_date")

    broker.reconcile()
    target_by_ticker = _target_map(targets)
    normalized_snapshots = {
        ticker.strip().upper(): snapshot
        for ticker, snapshot in snapshots.items()
    }
    constrained_targets, rejected_cap_weights = _cap_target_weights(
        target_by_ticker,
        constraints,
    )
    portfolio_value = _portfolio_value(broker, normalized_snapshots)
    all_tickers = set(target_by_ticker).union(broker.positions)
    current_weights: dict[str, Decimal] = {}

    for ticker in all_tickers:
        position = broker.positions.get(ticker)
        snapshot = normalized_snapshots.get(ticker)
        quantity = position.economic_quantity if position else 0

        if quantity > 0:
            if snapshot is None or snapshot.price is None or snapshot.price <= ZERO:
                raise ValueError(f"Missing price for held position: {ticker}")

            current_weights[ticker] = snapshot.price * quantity / portfolio_value
        else:
            current_weights[ticker] = ZERO

    planned_weights, turnover_was_capped = _turnover_constrained_weights(
        current_weights,
        constrained_targets,
        constraints.max_daily_turnover,
    )
    orders: list[PlannedOrder] = []
    skipped: list[SkippedTrade] = []
    requests: list[_TradeRequest] = []
    sequence = 0

    def next_order_id(ticker: str, label: str) -> str:
        nonlocal sequence
        sequence += 1
        return f"order-{order_date:%Y%m%d}-{sequence:03d}-{ticker}-{label.lower()}"

    def add_skip(
        ticker: str,
        requested_action: OrderAction,
        quantity: int,
        price: Decimal,
        reason: SkipReason,
        details: str,
        signal_id: str = "",
    ) -> None:
        order_id = next_order_id(ticker, "skip")
        value = price * quantity
        orders.append(
            PlannedOrder(
                order_id=order_id,
                signal_id=signal_id,
                order_date=order_date,
                intended_execution_date=execution_date,
                ticker=ticker,
                action=OrderAction.SKIP,
                requested_quantity=quantity,
                estimated_price=price,
                requested_value=value,
                status=OrderStatus.SKIPPED,
                reason_code=reason.value,
            )
        )
        skipped.append(
            SkippedTrade(
                skip_id=f"skip-{order_id}",
                order_id=order_id,
                date=order_date,
                ticker=ticker,
                side=requested_action,
                requested_quantity=quantity,
                requested_value=value,
                reason_code=reason,
                details=details,
            )
        )

    for ticker in sorted(all_tickers):
        target = target_by_ticker.get(ticker)
        signal_id = target.signal_id if target else ""
        predicted_rank = target.predicted_rank if target else None
        snapshot = normalized_snapshots.get(ticker)
        position = broker.positions.get(ticker)
        current_quantity = position.economic_quantity if position else 0
        price = snapshot.price if snapshot and snapshot.price else ZERO

        if price <= ZERO:
            add_skip(
                ticker=ticker,
                requested_action=OrderAction.SKIP,
                quantity=0,
                price=ZERO,
                reason=SkipReason.MISSING_PRICE,
                details="No valid price was available for order sizing.",
                signal_id=signal_id,
            )
            continue

        constrained_weight = constrained_targets.get(ticker, ZERO)

        for reason, rejected_weight in rejected_cap_weights.get(ticker, []):
            rejected_quantity = int(
                (rejected_weight * portfolio_value / price).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )

            if rejected_quantity > 0:
                add_skip(
                    ticker=ticker,
                    requested_action=OrderAction.BUY,
                    quantity=rejected_quantity,
                    price=price,
                    reason=reason,
                    details="Target exposure above the configured portfolio cap.",
                    signal_id=signal_id,
                )

        planned_weight = max(planned_weights.get(ticker, ZERO), ZERO)

        if turnover_was_capped:
            turnover_shortfall = abs(constrained_weight - planned_weight)
            shortfall_quantity = int(
                (turnover_shortfall * portfolio_value / price).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )

            if shortfall_quantity > 0:
                shortfall_action = (
                    OrderAction.BUY
                    if constrained_weight > current_weights.get(ticker, ZERO)
                    else OrderAction.SELL
                )
                add_skip(
                    ticker=ticker,
                    requested_action=shortfall_action,
                    quantity=shortfall_quantity,
                    price=price,
                    reason=SkipReason.MAX_TURNOVER_REACHED,
                    details="Part of the rebalance was deferred by the turnover cap.",
                    signal_id=signal_id,
                )

        desired_quantity = _desired_quantity(
            current_quantity,
            planned_weight * portfolio_value,
            price,
        )
        quantity_difference = desired_quantity - current_quantity

        if quantity_difference == 0:
            orders.append(
                PlannedOrder(
                    order_id=next_order_id(ticker, "hold"),
                    signal_id=signal_id,
                    order_date=order_date,
                    intended_execution_date=execution_date,
                    ticker=ticker,
                    action=OrderAction.HOLD,
                    requested_quantity=0,
                    estimated_price=price,
                    requested_value=ZERO,
                    status=OrderStatus.HOLD,
                )
            )
            continue

        action = (
            OrderAction.BUY if quantity_difference > 0 else OrderAction.SELL
        )
        unrounded_quantity = abs(quantity_difference)
        requested_quantity = constraints.round_trade_quantity(unrounded_quantity)

        if requested_quantity < unrounded_quantity:
            add_skip(
                ticker=ticker,
                requested_action=action,
                quantity=unrounded_quantity - requested_quantity,
                price=price,
                reason=SkipReason.BELOW_MIN_TRADE_VALUE,
                details="Quantity remainder was below the permitted trading lot.",
                signal_id=signal_id,
            )

        if requested_quantity == 0:
            continue

        block_reason = trade_block_reason(
            action=action,
            snapshot=snapshot,
            expected_data_date=expected_data_date,
            constraints=constraints,
        )

        if block_reason is not None:
            add_skip(
                ticker=ticker,
                requested_action=action,
                quantity=requested_quantity,
                price=price,
                reason=block_reason,
                details="Execution rule blocked the requested trade.",
                signal_id=signal_id,
            )
            continue

        requested_value = price * requested_quantity

        if requested_value < constraints.min_trade_value_vnd:
            add_skip(
                ticker=ticker,
                requested_action=action,
                quantity=requested_quantity,
                price=price,
                reason=SkipReason.BELOW_MIN_TRADE_VALUE,
                details="Requested value was below the minimum trade value.",
                signal_id=signal_id,
            )
            continue

        requests.append(
            _TradeRequest(
                ticker=ticker,
                action=action,
                quantity=requested_quantity,
                price=price,
                signal_id=signal_id,
                predicted_rank=predicted_rank,
            )
        )

    minimum_cash = portfolio_value * constraints.cash_buffer_weight
    spendable_cash = max(broker.buying_power - minimum_cash, ZERO)
    initial_spendable_cash = spendable_cash
    sell_requests = sorted(
        (request for request in requests if request.action == OrderAction.SELL),
        key=lambda request: (-request.value, request.ticker),
    )
    buy_requests = sorted(
        (request for request in requests if request.action == OrderAction.BUY),
        key=lambda request: (
            request.predicted_rank if request.predicted_rank is not None else 10**9,
            -request.value,
            request.ticker,
        ),
    )

    for request in sell_requests + buy_requests:
        snapshot = normalized_snapshots[request.ticker]
        quantity = request.quantity

        if snapshot.average_daily_value is None:
            add_skip(
                ticker=request.ticker,
                requested_action=request.action,
                quantity=quantity,
                price=request.price,
                reason=SkipReason.ADV_CAPACITY_LIMIT,
                details="Average daily value was missing.",
                signal_id=request.signal_id,
            )
            continue

        adv_quantity = constraints.adv_quantity_limit(
            snapshot.average_daily_value,
            request.price,
        )

        if adv_quantity < quantity:
            skipped_quantity = quantity - adv_quantity
            add_skip(
                ticker=request.ticker,
                requested_action=request.action,
                quantity=skipped_quantity,
                price=request.price,
                reason=SkipReason.ADV_CAPACITY_LIMIT,
                details="Trade quantity above the configured ADV capacity was deferred.",
                signal_id=request.signal_id,
            )
            quantity = adv_quantity

        if request.action == OrderAction.SELL:
            sellable = broker.sellable_quantity(request.ticker)

            if sellable < quantity:
                skipped_quantity = quantity - sellable
                add_skip(
                    ticker=request.ticker,
                    requested_action=request.action,
                    quantity=skipped_quantity,
                    price=request.price,
                    reason=SkipReason.INSUFFICIENT_SELLABLE_QUANTITY,
                    details="Requested shares were not yet sellable.",
                    signal_id=request.signal_id,
                )
                quantity = constraints.round_trade_quantity(sellable)
        else:
            affordable = constraints.affordable_buy_quantity(
                spendable_cash,
                request.price,
            )

            if affordable < quantity:
                skipped_quantity = quantity - affordable
                add_skip(
                    ticker=request.ticker,
                    requested_action=request.action,
                    quantity=skipped_quantity,
                    price=request.price,
                    reason=SkipReason.INSUFFICIENT_SETTLED_CASH,
                    details="Settled cash after the cash buffer could not fund the trade.",
                    signal_id=request.signal_id,
                )
                quantity = affordable

        executable_value = request.price * quantity

        if quantity <= 0 or executable_value < constraints.min_trade_value_vnd:
            if quantity > 0:
                add_skip(
                    ticker=request.ticker,
                    requested_action=request.action,
                    quantity=quantity,
                    price=request.price,
                    reason=SkipReason.BELOW_MIN_TRADE_VALUE,
                    details="Executable remainder was below the minimum trade value.",
                    signal_id=request.signal_id,
                )

            continue

        order = PlannedOrder(
            order_id=next_order_id(request.ticker, request.action.value),
            signal_id=request.signal_id,
            order_date=order_date,
            intended_execution_date=execution_date,
            ticker=request.ticker,
            action=request.action,
            requested_quantity=quantity,
            estimated_price=request.price,
            requested_value=executable_value,
            status=OrderStatus.PENDING,
        )
        orders.append(order)

        if request.action == OrderAction.BUY:
            spendable_cash += constraints.estimated_cash_effect(
                OrderAction.BUY,
                executable_value,
            )

    executable_value = sum(
        (order.requested_value for order in orders if order.status == OrderStatus.PENDING),
        start=ZERO,
    )
    estimated_turnover = Decimal("0.5") * executable_value / portfolio_value

    return OrderPlan(
        orders=orders,
        skipped_trades=skipped,
        portfolio_value=portfolio_value,
        estimated_turnover=estimated_turnover,
        spendable_cash_before_orders=initial_spendable_cash,
        spendable_cash_after_orders=spendable_cash,
        constrained_target_weights=planned_weights,
    )
