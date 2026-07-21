from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(destination)


def _atomic_text(content: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def write_daily_exports(
    *,
    output_directory: str | Path,
    model_name: str,
    signal_date: object,
    intended_execution_date: object,
    targets: pd.DataFrame,
    plan: Any,
    broker: Any,
    snapshots: Mapping[str, Any],
) -> dict[str, Path]:
    output = Path(output_directory)

    mark_prices = {
        ticker: snapshot.price
        for ticker, snapshot in snapshots.items()
        if getattr(snapshot, "price", None) is not None
    }

    current = pd.DataFrame(broker.position_rows(mark_prices))

    current_columns = [
        column
        for column in [
            "ticker",
            "settled_shares",
            "unsettled_buy_shares",
            "pending_sell_shares",
            "sellable_quantity",
            "economic_quantity",
            "average_cost",
            "mark_price",
            "market_value",
            "weight",
        ]
        if column in current.columns
    ]

    portfolio = targets.copy()

    if not current.empty:
        portfolio = portfolio.merge(
            current[current_columns],
            on="ticker",
            how="left",
        )

    numeric_defaults = {
        "settled_shares": 0,
        "unsettled_buy_shares": 0,
        "pending_sell_shares": 0,
        "sellable_quantity": 0,
        "economic_quantity": 0,
        "average_cost": 0,
        "mark_price": 0,
        "market_value": 0,
        "weight": 0,
    }

    for column, default in numeric_defaults.items():
        if column not in portfolio:
            portfolio[column] = default
        else:
            portfolio[column] = portfolio[column].fillna(default)

    portfolio = portfolio.rename(columns={"weight": "current_weight"})
    portfolio["target_weight"] = pd.to_numeric(
        portfolio["target_weight"],
        errors="coerce",
    ).fillna(0.0)
    portfolio["current_weight"] = pd.to_numeric(
        portfolio["current_weight"],
        errors="coerce",
    ).fillna(0.0)
    portfolio["weight_difference"] = (
        portfolio["target_weight"] - portfolio["current_weight"]
    )

    orders = pd.DataFrame(plan.order_rows())

    account = pd.DataFrame([
        {
            "signal_date": str(signal_date),
            "intended_execution_date": str(intended_execution_date),
            "model": model_name,
            "broker_asof_date": str(broker.asof_date or ""),
            "settled_cash": str(broker.settled_cash),
            "unsettled_cash": str(broker.unsettled_cash),
            "buying_power": str(broker.buying_power),
            "portfolio_value": str(plan.portfolio_value),
            "spendable_cash_before_orders": str(
                plan.spendable_cash_before_orders
            ),
            "spendable_cash_after_orders": str(
                plan.spendable_cash_after_orders
            ),
            "estimated_turnover": str(plan.estimated_turnover),
            "position_count": sum(
                1
                for position in broker.positions.values()
                if position.economic_quantity > 0
            ),
            "pending_settlement_count": sum(
                1
                for settlement in broker.pending_settlements
                if settlement.status.value == "PENDING"
            ),
            "executable_order_count": len(plan.executable_orders),
            "skipped_trade_count": len(plan.skipped_trades),
        }
    ])

    summary = "\n".join([
        "# VN30 Daily Paper-Trading Summary",
        "",
        f"- Signal date: {signal_date}",
        f"- Intended execution date: {intended_execution_date}",
        f"- Model: {model_name}",
        f"- Target holdings: {len(portfolio)}",
        f"- Executable orders: {len(plan.executable_orders)}",
        f"- Skipped or deferred trades: {len(plan.skipped_trades)}",
        f"- Portfolio value: {plan.portfolio_value} VND",
        f"- Estimated turnover: {plan.estimated_turnover}",
        (
            "- Spendable cash after proposed orders: "
            f"{plan.spendable_cash_after_orders} VND"
        ),
        "",
        "No real orders were submitted.",
        "",
    ])

    paths = {
        "portfolio": output / "latest_portfolio.csv",
        "orders": output / "latest_orders.csv",
        "summary": output / "latest_signal_summary.md",
        "account": output / "latest_account_state.csv",
    }

    _atomic_csv(portfolio, paths["portfolio"])
    _atomic_csv(orders, paths["orders"])
    _atomic_csv(account, paths["account"])
    _atomic_text(summary, paths["summary"])

    return paths
