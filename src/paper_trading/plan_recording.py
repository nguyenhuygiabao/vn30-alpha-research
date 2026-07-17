from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from src.paper_trading.broker_state import PaperBrokerState
from src.paper_trading.storage import PaperAccountStorage


PLAN_LEDGER_KEYS = {
    "signals.csv": ("signal_id",),
    "target_weights.csv": ("portfolio_id", "ticker"),
    "orders.csv": ("order_id",),
    "skipped_trades.csv": ("skip_id",),
}

ROLLBACK_LEDGER_FILES = (
    "signals.csv",
    "target_weights.csv",
    "orders.csv",
    "skipped_trades.csv",
    "positions.csv",
    "cash_ledger.csv",
    "settlement_ledger.csv",
)


def _identifier_set(
    rows: list[dict[str, Any]],
    key_columns: tuple[str, ...],
    filename: str,
) -> set[tuple[str, ...]]:
    identifiers = []

    for row in rows:
        try:
            identifier = tuple(str(row[column]) for column in key_columns)
        except KeyError as error:
            raise ValueError(
                f"Missing identifier column in {filename}: {error.args[0]}"
            ) from error

        if any(not value.strip() for value in identifier):
            raise ValueError(f"Empty ledger identifier in {filename}")

        identifiers.append(identifier)

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate incoming identifiers in {filename}")

    return set(identifiers)


def _existing_identifier_set(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> set[tuple[str, ...]]:
    return {
        tuple(str(getattr(row, column)) for column in key_columns)
        for row in frame.itertuples(index=False)
    }


def _restore_ledgers(
    storage: PaperAccountStorage,
    originals: Mapping[str, pd.DataFrame],
) -> None:
    for filename, frame in originals.items():
        storage.replace_rows(
            filename,
            frame.to_dict(orient="records"),
        )


def record_daily_paper_plan(
    storage: PaperAccountStorage,
    broker: PaperBrokerState,
    rows_by_ledger: Mapping[
        str,
        Iterable[Mapping[str, object]],
    ],
) -> bool:
    received = set(rows_by_ledger)
    expected = set(PLAN_LEDGER_KEYS)

    if received != expected:
        raise ValueError(
            "Daily plan must provide exactly these ledgers: "
            f"{sorted(expected)}"
        )

    normalized = {
        filename: [dict(row) for row in rows_by_ledger[filename]]
        for filename in PLAN_LEDGER_KEYS
    }

    incoming_identifiers = {
        filename: _identifier_set(
            normalized[filename],
            key_columns,
            filename,
        )
        for filename, key_columns in PLAN_LEDGER_KEYS.items()
    }

    with storage.account_lock():
        storage.validate_all_ledgers()

        originals = {
            filename: storage.read_ledger(filename)
            for filename in ROLLBACK_LEDGER_FILES
        }

        existing_identifiers = {
            filename: _existing_identifier_set(
                originals[filename],
                key_columns,
            )
            for filename, key_columns in PLAN_LEDGER_KEYS.items()
        }

        overlaps = {
            filename: incoming_identifiers[filename]
            & existing_identifiers[filename]
            for filename in PLAN_LEDGER_KEYS
        }

        if any(overlaps.values()):
            fully_recorded = all(
                incoming_identifiers[filename].issubset(
                    existing_identifiers[filename]
                )
                for filename in PLAN_LEDGER_KEYS
            )

            if fully_recorded:
                return False

            raise RuntimeError(
                "A partial or conflicting paper plan already exists; "
                "no ledger changes were made"
            )

        try:
            for filename, key_columns in PLAN_LEDGER_KEYS.items():
                storage.append_rows(
                    filename,
                    normalized[filename],
                    unique_by=key_columns,
                )

            storage.save_broker_state(broker)
        except Exception:
            _restore_ledgers(storage, originals)
            raise

    return True
