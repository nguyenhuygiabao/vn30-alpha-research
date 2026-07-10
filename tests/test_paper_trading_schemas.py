from __future__ import annotations

import pandas as pd
import pytest

from src.paper_trading.schemas import (
    LEDGER_SCHEMAS,
    SkipReason,
    initialize_empty_ledgers,
    validate_ledger_columns,
)


def test_all_required_ledgers_are_defined() -> None:
    assert set(LEDGER_SCHEMAS) == {
        "signals.csv",
        "target_weights.csv",
        "orders.csv",
        "executions.csv",
        "positions.csv",
        "cash_ledger.csv",
        "settlement_ledger.csv",
        "daily_performance.csv",
        "skipped_trades.csv",
    }


def test_empty_ledger_initialization_uses_exact_schemas(tmp_path) -> None:
    created = initialize_empty_ledgers(tmp_path)

    assert len(created) == len(LEDGER_SCHEMAS)

    for filename, expected_columns in LEDGER_SCHEMAS.items():
        ledger = pd.read_csv(tmp_path / filename)
        assert list(ledger.columns) == list(expected_columns)


def test_initialization_does_not_overwrite_existing_ledgers(tmp_path) -> None:
    initialize_empty_ledgers(tmp_path)
    signals_path = tmp_path / "signals.csv"
    signals_path.write_text("do-not-overwrite\n1\n", encoding="utf-8")

    initialize_empty_ledgers(tmp_path)

    assert signals_path.read_text(encoding="utf-8") == "do-not-overwrite\n1\n"


def test_skip_reasons_cover_settlement_and_execution_blocks() -> None:
    values = {reason.value for reason in SkipReason}

    assert "insufficient_settled_cash" in values
    assert "insufficient_sellable_quantity" in values
    assert "price_ceiling_buy_block" in values
    assert "price_floor_sell_block" in values
    assert "reconciliation_failed" in values


def test_column_validation_rejects_reordered_columns() -> None:
    columns = list(LEDGER_SCHEMAS["orders.csv"])
    columns.reverse()

    with pytest.raises(ValueError, match="Invalid columns"):
        validate_ledger_columns("orders.csv", columns)
