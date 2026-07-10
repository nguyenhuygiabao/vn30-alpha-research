from __future__ import annotations

import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from src.paper_trading.calendar import TradingCalendar
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.schemas import LEDGER_SCHEMAS, Side
from src.paper_trading.settlement import ExecutionRecord
from src.paper_trading.storage import PaperAccountStorage


ROOT = Path(__file__).resolve().parents[1]


def test_initialize_creates_all_ledgers_and_round_trips(tmp_path) -> None:
    storage = PaperAccountStorage(tmp_path / "paper")
    initialized = storage.initialize("100000000", "2026-07-10")

    assert initialized.settled_cash == Decimal("100000000")

    for filename in LEDGER_SCHEMAS:
        assert storage.ledger_path(filename).exists()

    loaded = storage.load_broker_state()

    assert loaded.asof_date.isoformat() == "2026-07-10"
    assert loaded.settled_cash == Decimal("100000000")
    assert loaded.unsettled_cash == Decimal("0")
    assert loaded.positions == {}
    assert len(loaded.cash_ledger_entries) == 1


def test_initialize_refuses_to_replace_existing_account(tmp_path) -> None:
    storage = PaperAccountStorage(tmp_path / "paper")
    storage.initialize("100000000", "2026-07-10")

    with pytest.raises(FileExistsError, match="already contains ledger files"):
        storage.initialize("200000000", "2026-07-11")

    loaded = storage.load_broker_state()
    assert loaded.settled_cash == Decimal("100000000")


def test_pending_settlement_survives_save_and_load(tmp_path) -> None:
    storage = PaperAccountStorage(tmp_path / "paper")
    broker = storage.initialize("100000000", "2026-07-06")
    calendar = TradingCalendar.from_weekdays("2026-07-01", "2026-07-31")
    execution = ExecutionRecord(
        execution_id="execution-fpt-buy",
        order_id="order-fpt-buy",
        execution_date="2026-07-06",
        ticker="FPT",
        side=Side.BUY,
        filled_quantity=100,
        execution_price="10000",
        commission="1000",
        slippage="1000",
    )
    broker.apply_execution(execution, calendar, issuer_group="FPT")
    storage.append_rows(
        "executions.csv",
        [execution.to_row()],
        unique_by=("execution_id",),
    )
    storage.save_broker_state(broker, mark_prices={"FPT": "10000"})

    loaded = storage.load_broker_state()
    position = loaded.positions["FPT"]

    assert loaded.settled_cash == Decimal("98998000")
    assert position.settled_shares == 0
    assert position.unsettled_buy_shares == 100
    assert position.sellable_quantity == 0
    assert len(loaded.pending_settlements) == 1
    assert "execution-fpt-buy" in loaded.processed_execution_ids

    loaded.settle_due("2026-07-08")
    storage.save_broker_state(loaded, mark_prices={"FPT": "10000"})
    settled_reload = storage.load_broker_state()

    assert settled_reload.positions["FPT"].settled_shares == 100
    assert settled_reload.positions["FPT"].unsettled_buy_shares == 0
    assert settled_reload.sellable_quantity("FPT") == 100


def test_schema_corruption_is_rejected(tmp_path) -> None:
    storage = PaperAccountStorage(tmp_path / "paper")
    storage.initialize("100000000", "2026-07-10")
    storage.ledger_path("positions.csv").write_text(
        "ticker,bad_column\nFPT,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid columns"):
        storage.read_ledger("positions.csv")


def test_cash_running_balance_corruption_is_rejected(tmp_path) -> None:
    storage = PaperAccountStorage(tmp_path / "paper")
    storage.initialize("100000000", "2026-07-10")
    cash = storage.read_ledger("cash_ledger.csv")
    cash.loc[0, "settled_cash_balance"] = "99999999"
    cash.to_csv(storage.ledger_path("cash_ledger.csv"), index=False)

    with pytest.raises(ValueError, match="Settled cash running balance failed"):
        storage.load_broker_state()


def test_append_rows_rejects_duplicate_identifiers(tmp_path) -> None:
    storage = PaperAccountStorage(tmp_path / "paper")
    storage.initialize("100000000", "2026-07-10")
    row = {
        "signal_id": "signal-1",
        "data_asof_date": "2026-07-10",
        "signal_date": "2026-07-10",
        "intended_execution_date": "2026-07-13",
        "ticker": "FPT",
        "model_name": "gradient_boosting",
        "horizon_days": 10,
        "score": "0.01",
        "predicted_rank": 1,
        "created_at": "2026-07-10T15:15:00+07:00",
    }
    storage.append_rows("signals.csv", [row], unique_by=("signal_id",))

    with pytest.raises(ValueError, match="Duplicate ledger identifiers"):
        storage.append_rows("signals.csv", [row], unique_by=("signal_id",))

    assert len(storage.read_ledger("signals.csv")) == 1


def test_account_lock_rejects_concurrent_operation(tmp_path) -> None:
    storage = PaperAccountStorage(tmp_path / "paper")

    with storage.account_lock():
        with pytest.raises(RuntimeError, match="locked by another operation"):
            with storage.account_lock():
                pass

    assert not storage.lock_path.exists()


def test_initialize_and_reconcile_scripts(tmp_path) -> None:
    config = load_paper_trading_config()
    config["output"]["directory"] = str(tmp_path / "paper")
    config_path = tmp_path / "paper_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    initialize = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "initialize_paper_account.py"),
            "--config",
            str(config_path),
            "--asof-date",
            "2026-07-10",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert initialize.returncode == 0, initialize.stderr
    assert "PAPER ACCOUNT INITIALIZED" in initialize.stdout
    assert "No real orders were placed" in initialize.stdout

    reconcile = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reconcile_paper_account.py"),
            "--config",
            str(config_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert reconcile.returncode == 0, reconcile.stderr
    assert "PAPER ACCOUNT RECONCILIATION PASSED" in reconcile.stdout
    assert "Pending settlements: 0" in reconcile.stdout
