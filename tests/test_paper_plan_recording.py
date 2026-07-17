from __future__ import annotations

from datetime import date

import pytest

from src.paper_trading.plan_recording import record_daily_paper_plan
from src.paper_trading.storage import PaperAccountStorage


def plan_rows(suffix: str = "001"):
    signal_id = f"signal-{suffix}"
    order_id = f"order-{suffix}"

    return {
        "signals.csv": [{
            "signal_id": signal_id,
            "data_asof_date": "2026-07-17",
            "signal_date": "2026-07-17",
            "intended_execution_date": "2026-07-20",
            "ticker": "FPT",
            "model_name": "rank_ensemble",
            "horizon_days": 10,
            "score": "0.9",
            "predicted_rank": 1,
            "created_at": "2026-07-17T16:00:00+07:00",
        }],
        "target_weights.csv": [{
            "portfolio_id": f"portfolio-{suffix}",
            "signal_date": "2026-07-17",
            "intended_execution_date": "2026-07-20",
            "ticker": "FPT",
            "issuer_group": "FPT",
            "target_weight": "0.12",
            "rebalance_flag": "true",
            "created_at": "2026-07-17T16:00:00+07:00",
        }],
        "orders.csv": [{
            "order_id": order_id,
            "signal_id": signal_id,
            "order_date": "2026-07-17",
            "intended_execution_date": "2026-07-20",
            "ticker": "FPT",
            "side": "BUY",
            "requested_quantity": 100,
            "estimated_price": "100000",
            "requested_value": "10000000",
            "status": "PENDING",
            "reason_code": "",
            "created_at": "2026-07-17T16:00:00+07:00",
        }],
        "skipped_trades.csv": [],
    }


def initialized_storage(tmp_path):
    storage = PaperAccountStorage(tmp_path / "paper")
    storage.initialize("100000000", date(2026, 7, 17))
    return storage


def test_daily_plan_is_recorded_once_and_is_idempotent(tmp_path) -> None:
    storage = initialized_storage(tmp_path)
    broker = storage.load_broker_state()
    rows = plan_rows()

    assert record_daily_paper_plan(storage, broker, rows) is True
    assert len(storage.read_ledger("signals.csv")) == 1
    assert len(storage.read_ledger("target_weights.csv")) == 1
    assert len(storage.read_ledger("orders.csv")) == 1

    assert record_daily_paper_plan(storage, broker, rows) is False
    assert len(storage.read_ledger("signals.csv")) == 1
    assert len(storage.read_ledger("orders.csv")) == 1


def test_failed_plan_write_rolls_back_every_ledger(tmp_path) -> None:
    storage = initialized_storage(tmp_path)
    broker = storage.load_broker_state()
    rows = plan_rows()
    rows["target_weights.csv"][0].pop("created_at")

    with pytest.raises(ValueError, match="Invalid row keys"):
        record_daily_paper_plan(storage, broker, rows)

    assert storage.read_ledger("signals.csv").empty
    assert storage.read_ledger("target_weights.csv").empty
    assert storage.read_ledger("orders.csv").empty
