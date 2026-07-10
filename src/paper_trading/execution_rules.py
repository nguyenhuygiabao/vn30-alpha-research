from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Any

from src.paper_trading.calendar import DateLike, normalize_date
from src.paper_trading.schemas import OrderAction, SkipReason
from src.paper_trading.settlement import ZERO, to_decimal


@dataclass(frozen=True)
class ExecutionConstraints:
    cash_buffer_weight: Decimal
    max_single_name_weight: Decimal
    max_issuer_group_weight: Decimal
    max_daily_turnover: Decimal
    round_lot_size: int
    allow_odd_lots: bool
    min_trade_value_vnd: Decimal
    max_trade_adv_fraction: Decimal
    commission_rate: Decimal
    slippage_rate: Decimal
    sell_tax_rate: Decimal
    block_buy_at_ceiling: bool
    block_sell_at_floor: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExecutionConstraints:
        portfolio = config["portfolio"]
        execution = config["execution"]

        return cls(
            cash_buffer_weight=to_decimal(portfolio["cash_buffer_weight"]),
            max_single_name_weight=to_decimal(
                portfolio["max_single_name_weight"]
            ),
            max_issuer_group_weight=to_decimal(
                portfolio["max_issuer_group_weight"]
            ),
            max_daily_turnover=to_decimal(portfolio["max_daily_turnover"]),
            round_lot_size=int(execution["round_lot_size"]),
            allow_odd_lots=bool(execution["allow_odd_lots"]),
            min_trade_value_vnd=to_decimal(execution["min_trade_value_vnd"]),
            max_trade_adv_fraction=to_decimal(
                execution["max_trade_adv_fraction"]
            ),
            commission_rate=to_decimal(execution["commission_rate"]),
            slippage_rate=to_decimal(execution["slippage_rate"]),
            sell_tax_rate=to_decimal(execution["sell_tax_rate"]),
            block_buy_at_ceiling=bool(execution["block_buy_at_ceiling"]),
            block_sell_at_floor=bool(execution["block_sell_at_floor"]),
        )

    def round_trade_quantity(self, quantity: int) -> int:
        if quantity <= 0:
            return 0

        if self.allow_odd_lots:
            return quantity

        return (quantity // self.round_lot_size) * self.round_lot_size

    def estimated_cash_effect(
        self,
        action: OrderAction,
        gross_value: Decimal,
    ) -> Decimal:
        if action == OrderAction.BUY:
            return -gross_value * (
                Decimal("1") + self.commission_rate + self.slippage_rate
            )

        if action == OrderAction.SELL:
            return gross_value * (
                Decimal("1")
                - self.commission_rate
                - self.slippage_rate
                - self.sell_tax_rate
            )

        return ZERO

    def affordable_buy_quantity(
        self,
        available_cash: Decimal,
        price: Decimal,
    ) -> int:
        if available_cash <= ZERO or price <= ZERO:
            return 0

        unit_cash_required = price * (
            Decimal("1") + self.commission_rate + self.slippage_rate
        )
        raw_quantity = int(
            (available_cash / unit_cash_required).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )

        return self.round_trade_quantity(raw_quantity)

    def adv_quantity_limit(
        self,
        average_daily_value: Decimal,
        price: Decimal,
    ) -> int:
        if average_daily_value <= ZERO or price <= ZERO:
            return 0

        maximum_value = average_daily_value * self.max_trade_adv_fraction
        raw_quantity = int(
            (maximum_value / price).to_integral_value(rounding=ROUND_FLOOR)
        )

        return self.round_trade_quantity(raw_quantity)


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    data_date: DateLike | None
    price: Decimal | int | float | str | None
    average_daily_value: Decimal | int | float | str | None
    prediction_available: bool = True
    at_ceiling: bool = False
    at_floor: bool = False

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()

        if not ticker:
            raise ValueError("ticker cannot be empty")

        object.__setattr__(self, "ticker", ticker)

        if self.data_date is not None:
            object.__setattr__(self, "data_date", normalize_date(self.data_date))

        if self.price is not None:
            object.__setattr__(self, "price", to_decimal(self.price))

        if self.average_daily_value is not None:
            object.__setattr__(
                self,
                "average_daily_value",
                to_decimal(self.average_daily_value),
            )


def trade_block_reason(
    action: OrderAction,
    snapshot: MarketSnapshot | None,
    expected_data_date: DateLike,
    constraints: ExecutionConstraints,
) -> SkipReason | None:
    if snapshot is None or snapshot.data_date is None:
        return SkipReason.STALE_DATA

    if snapshot.data_date != normalize_date(expected_data_date):
        return SkipReason.STALE_DATA

    if not snapshot.prediction_available:
        return SkipReason.MISSING_PREDICTION

    if snapshot.price is None or snapshot.price <= ZERO:
        return SkipReason.MISSING_PRICE

    if action == OrderAction.BUY and constraints.block_buy_at_ceiling:
        if snapshot.at_ceiling:
            return SkipReason.PRICE_CEILING_BUY_BLOCK

    if action == OrderAction.SELL and constraints.block_sell_at_floor:
        if snapshot.at_floor:
            return SkipReason.PRICE_FLOOR_SELL_BLOCK

    return None
