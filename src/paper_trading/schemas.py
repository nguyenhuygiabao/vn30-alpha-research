from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pandas as pd


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    HOLD = "HOLD"
    EXECUTED = "EXECUTED"
    PARTIALLY_EXECUTED = "PARTIALLY_EXECUTED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class SettlementStatus(StrEnum):
    PENDING = "PENDING"
    SETTLED = "SETTLED"
    FAILED = "FAILED"


class SkipReason(StrEnum):
    INSUFFICIENT_SETTLED_CASH = "insufficient_settled_cash"
    INSUFFICIENT_SELLABLE_QUANTITY = "insufficient_sellable_quantity"
    MAX_TURNOVER_REACHED = "max_turnover_reached"
    MAX_SINGLE_NAME_WEIGHT_REACHED = "max_single_name_weight_reached"
    MAX_ISSUER_GROUP_WEIGHT_REACHED = "max_issuer_group_weight_reached"
    ADV_CAPACITY_LIMIT = "adv_capacity_limit"
    PRICE_CEILING_BUY_BLOCK = "price_ceiling_buy_block"
    PRICE_FLOOR_SELL_BLOCK = "price_floor_sell_block"
    MISSING_PRICE = "missing_price"
    MISSING_PREDICTION = "missing_prediction"
    STALE_DATA = "stale_data"
    BELOW_MIN_TRADE_VALUE = "below_min_trade_value"
    RECONCILIATION_FAILED = "reconciliation_failed"


LEDGER_SCHEMAS: dict[str, tuple[str, ...]] = {
    "signals.csv": (
        "signal_id",
        "data_asof_date",
        "signal_date",
        "intended_execution_date",
        "ticker",
        "model_name",
        "horizon_days",
        "score",
        "predicted_rank",
        "created_at",
    ),
    "target_weights.csv": (
        "portfolio_id",
        "signal_date",
        "intended_execution_date",
        "ticker",
        "issuer_group",
        "target_weight",
        "rebalance_flag",
        "created_at",
    ),
    "orders.csv": (
        "order_id",
        "signal_id",
        "order_date",
        "intended_execution_date",
        "ticker",
        "side",
        "requested_quantity",
        "estimated_price",
        "requested_value",
        "status",
        "reason_code",
        "created_at",
    ),
    "executions.csv": (
        "execution_id",
        "order_id",
        "execution_date",
        "ticker",
        "side",
        "filled_quantity",
        "execution_price",
        "gross_value",
        "commission",
        "tax",
        "slippage",
        "net_cash_effect",
        "created_at",
    ),
    "positions.csv": (
        "asof_date",
        "ticker",
        "issuer_group",
        "settled_shares",
        "unsettled_buy_shares",
        "pending_sell_shares",
        "sellable_quantity",
        "average_cost",
        "mark_price",
        "market_value",
        "weight",
    ),
    "cash_ledger.csv": (
        "entry_id",
        "event_date",
        "settlement_date",
        "entry_type",
        "amount",
        "settled_cash_delta",
        "unsettled_cash_delta",
        "reference_id",
        "settled_cash_balance",
        "unsettled_cash_balance",
        "created_at",
    ),
    "settlement_ledger.csv": (
        "settlement_id",
        "trade_date",
        "settlement_date",
        "ticker",
        "side",
        "quantity",
        "gross_amount",
        "fees_and_taxes",
        "status",
        "settled_at",
        "reference_id",
    ),
    "daily_performance.csv": (
        "date",
        "portfolio_value",
        "settled_cash",
        "unsettled_cash",
        "market_value",
        "daily_return",
        "cumulative_return",
        "benchmark_value",
        "benchmark_return",
        "cumulative_benchmark_return",
        "active_return",
        "drawdown",
        "turnover",
        "cash_weight",
        "holdings_count",
        "skipped_trade_count",
    ),
    "skipped_trades.csv": (
        "skip_id",
        "order_id",
        "date",
        "ticker",
        "side",
        "requested_quantity",
        "requested_value",
        "reason_code",
        "details",
        "created_at",
    ),
}


def validate_ledger_columns(filename: str, columns: list[str]) -> None:
    if filename not in LEDGER_SCHEMAS:
        raise KeyError(f"Unknown paper-trading ledger: {filename}")

    expected = list(LEDGER_SCHEMAS[filename])

    if columns != expected:
        raise ValueError(
            f"Invalid columns for {filename}. Expected {expected}, received {columns}"
        )


def initialize_empty_ledgers(
    output_directory: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    created_paths: list[Path] = []

    for filename, columns in LEDGER_SCHEMAS.items():
        ledger_path = output_path / filename

        if ledger_path.exists() and not overwrite:
            continue

        pd.DataFrame(columns=columns).to_csv(ledger_path, index=False)
        created_paths.append(ledger_path)

    return created_paths
