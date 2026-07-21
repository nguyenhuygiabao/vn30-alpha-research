from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

from src.paper_trading.daily_exports import write_daily_exports


class FakeBroker:
    asof_date = None
    settled_cash = Decimal("1000")
    unsettled_cash = Decimal("0")
    buying_power = Decimal("1000")
    positions = {}
    pending_settlements = []

    def position_rows(self, mark_prices):
        del mark_prices
        return []


class FakePlan:
    portfolio_value = Decimal("1000")
    spendable_cash_before_orders = Decimal("900")
    spendable_cash_after_orders = Decimal("700")
    estimated_turnover = Decimal("0.10")
    executable_orders = [object()]
    skipped_trades = []

    def order_rows(self):
        return [{
            "ticker": "FPT",
            "side": "BUY",
            "requested_quantity": 100,
            "status": "PENDING",
        }]


def test_write_daily_exports_creates_latest_files(tmp_path) -> None:
    targets = pd.DataFrame([
        {
            "ticker": "FPT",
            "predicted_rank": 1,
            "target_weight": 0.15,
        }
    ])

    paths = write_daily_exports(
        output_directory=tmp_path,
        model_name="rank_ensemble",
        signal_date="2026-07-21",
        intended_execution_date="2026-07-22",
        targets=targets,
        plan=FakePlan(),
        broker=FakeBroker(),
        snapshots={
            "FPT": SimpleNamespace(price=Decimal("100000")),
        },
    )

    assert all(path.exists() for path in paths.values())

    portfolio = pd.read_csv(paths["portfolio"])
    account = pd.read_csv(paths["account"])
    orders = pd.read_csv(paths["orders"])

    assert portfolio.loc[0, "ticker"] == "FPT"
    assert portfolio.loc[0, "weight_difference"] == 0.15
    assert account.loc[0, "executable_order_count"] == 1
    assert orders.loc[0, "side"] == "BUY"
