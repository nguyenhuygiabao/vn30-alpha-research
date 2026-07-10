from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import pandas as pd

from src.paper_trading.broker_state import (
    CashLedgerEntry,
    PaperBrokerState,
    PositionState,
)
from src.paper_trading.calendar import DateLike, normalize_date
from src.paper_trading.schemas import (
    LEDGER_SCHEMAS,
    SettlementStatus,
    Side,
    initialize_empty_ledgers,
    validate_ledger_columns,
)
from src.paper_trading.settlement import PendingSettlement, to_decimal


STATE_LEDGER_FILES = (
    "positions.csv",
    "cash_ledger.csv",
    "settlement_ledger.csv",
    "executions.csv",
)


def _optional_date(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return normalize_date(text)


def _required_date(value: object, field_name: str) -> date:
    result = _optional_date(value)

    if result is None:
        raise ValueError(f"Missing required date field: {field_name}")

    return result


def _required_text(value: object, field_name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"Missing required text field: {field_name}")

    text = str(value).strip()

    if not text:
        raise ValueError(f"Missing required text field: {field_name}")

    return text


def _integer(value: object, field_name: str) -> int:
    decimal_value = to_decimal(_required_text(value, field_name))

    if decimal_value != decimal_value.to_integral_value():
        raise ValueError(f"Expected integer value for {field_name}: {value}")

    return int(decimal_value)


def _decimal(value: object, field_name: str) -> Decimal:
    return to_decimal(_required_text(value, field_name))


def _validate_row_keys(filename: str, row: Mapping[str, object]) -> None:
    expected = set(LEDGER_SCHEMAS[filename])
    received = set(row)

    if received != expected:
        missing = sorted(expected.difference(received))
        unexpected = sorted(received.difference(expected))
        raise ValueError(
            f"Invalid row keys for {filename}. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )


class PaperAccountStorage:
    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.lock_path = self.output_directory / ".account.lock"

    def ledger_path(self, filename: str) -> Path:
        if filename not in LEDGER_SCHEMAS:
            raise KeyError(f"Unknown paper-trading ledger: {filename}")

        return self.output_directory / filename

    def initialize(
        self,
        initial_cash: Decimal | int | float | str,
        asof_date: DateLike,
        overwrite: bool = False,
    ) -> PaperBrokerState:
        existing = [
            self.ledger_path(filename)
            for filename in LEDGER_SCHEMAS
            if self.ledger_path(filename).exists()
        ]

        if existing and not overwrite:
            raise FileExistsError(
                "Paper account already contains ledger files. "
                "Use overwrite only when intentionally resetting the account."
            )

        initialize_empty_ledgers(
            self.output_directory,
            overwrite=overwrite,
        )
        broker = PaperBrokerState.initialize(initial_cash, asof_date)
        self.save_broker_state(broker)

        return broker

    def read_ledger(self, filename: str) -> pd.DataFrame:
        path = self.ledger_path(filename)

        if not path.exists():
            raise FileNotFoundError(f"Missing paper-trading ledger: {path}")

        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        validate_ledger_columns(filename, list(frame.columns))

        return frame

    def _atomic_write(self, filename: str, frame: pd.DataFrame) -> None:
        validate_ledger_columns(filename, list(frame.columns))
        self.output_directory.mkdir(parents=True, exist_ok=True)
        destination = self.ledger_path(filename)
        temporary = destination.with_suffix(destination.suffix + ".tmp")

        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(destination)

    def replace_rows(
        self,
        filename: str,
        rows: Iterable[Mapping[str, object]],
    ) -> None:
        columns = list(LEDGER_SCHEMAS[filename])
        normalized_rows = [dict(row) for row in rows]

        for row in normalized_rows:
            _validate_row_keys(filename, row)

        frame = pd.DataFrame(normalized_rows, columns=columns)
        self._atomic_write(filename, frame)

    def append_rows(
        self,
        filename: str,
        rows: Iterable[Mapping[str, object]],
        unique_by: tuple[str, ...] = (),
    ) -> None:
        new_rows = [dict(row) for row in rows]

        if not new_rows:
            return

        expected_columns = list(LEDGER_SCHEMAS[filename])

        for row in new_rows:
            _validate_row_keys(filename, row)

        existing = self.read_ledger(filename)
        additions = pd.DataFrame(new_rows, columns=expected_columns).astype(str)
        combined = pd.concat([existing, additions], ignore_index=True)

        if unique_by:
            missing_unique_columns = [
                column for column in unique_by if column not in expected_columns
            ]

            if missing_unique_columns:
                raise ValueError(
                    f"Unknown uniqueness columns for {filename}: "
                    f"{missing_unique_columns}"
                )

            duplicates = combined.duplicated(subset=list(unique_by), keep=False)

            if duplicates.any():
                duplicate_rows = combined.loc[duplicates, list(unique_by)]
                raise ValueError(
                    f"Duplicate ledger identifiers in {filename}: "
                    f"{duplicate_rows.to_dict(orient='records')}"
                )

        self._atomic_write(filename, combined)

    def save_broker_state(
        self,
        broker: PaperBrokerState,
        mark_prices: Mapping[str, Decimal | int | float | str] | None = None,
    ) -> None:
        broker.reconcile()
        self.replace_rows("positions.csv", broker.position_rows(mark_prices))
        self.replace_rows("cash_ledger.csv", broker.cash_ledger_rows())
        self.replace_rows("settlement_ledger.csv", broker.settlement_rows())

    def load_broker_state(self) -> PaperBrokerState:
        for filename in STATE_LEDGER_FILES:
            self.read_ledger(filename)

        cash_frame = self.read_ledger("cash_ledger.csv")

        if cash_frame.empty:
            raise ValueError(
                "Paper account is not initialized because cash_ledger.csv is empty"
            )

        cash_entries = [
            CashLedgerEntry(
                entry_id=_required_text(row.entry_id, "entry_id"),
                event_date=_required_date(row.event_date, "event_date"),
                settlement_date=_required_date(
                    row.settlement_date,
                    "settlement_date",
                ),
                entry_type=_required_text(row.entry_type, "entry_type"),
                amount=_decimal(row.amount, "amount"),
                settled_cash_delta=_decimal(
                    row.settled_cash_delta,
                    "settled_cash_delta",
                ),
                unsettled_cash_delta=_decimal(
                    row.unsettled_cash_delta,
                    "unsettled_cash_delta",
                ),
                reference_id=_required_text(row.reference_id, "reference_id"),
                settled_cash_balance=_decimal(
                    row.settled_cash_balance,
                    "settled_cash_balance",
                ),
                unsettled_cash_balance=_decimal(
                    row.unsettled_cash_balance,
                    "unsettled_cash_balance",
                ),
                created_at=_required_text(row.created_at, "created_at"),
            )
            for row in cash_frame.itertuples(index=False)
        ]
        cash_entry_ids: set[str] = set()
        running_settled_cash = Decimal("0")
        running_unsettled_cash = Decimal("0")

        for entry in cash_entries:
            if entry.entry_id in cash_entry_ids:
                raise ValueError(f"Duplicate cash ledger entry: {entry.entry_id}")

            cash_entry_ids.add(entry.entry_id)
            running_settled_cash += entry.settled_cash_delta
            running_unsettled_cash += entry.unsettled_cash_delta

            if running_settled_cash != entry.settled_cash_balance:
                raise ValueError(
                    f"Settled cash running balance failed at {entry.entry_id}"
                )

            if running_unsettled_cash != entry.unsettled_cash_balance:
                raise ValueError(
                    f"Unsettled cash running balance failed at {entry.entry_id}"
                )

        latest_cash = cash_entries[-1]

        position_frame = self.read_ledger("positions.csv")
        positions: dict[str, PositionState] = {}
        asof_candidates: list[date] = [latest_cash.event_date]

        for row in position_frame.itertuples(index=False):
            ticker = _required_text(row.ticker, "ticker").upper()

            if ticker in positions:
                raise ValueError(f"Duplicate position ticker: {ticker}")

            position_asof = _required_date(row.asof_date, "asof_date")
            asof_candidates.append(position_asof)
            positions[ticker] = PositionState(
                ticker=ticker,
                issuer_group=str(row.issuer_group).strip(),
                settled_shares=_integer(row.settled_shares, "settled_shares"),
                unsettled_buy_shares=_integer(
                    row.unsettled_buy_shares,
                    "unsettled_buy_shares",
                ),
                pending_sell_shares=_integer(
                    row.pending_sell_shares,
                    "pending_sell_shares",
                ),
                average_cost=_decimal(row.average_cost, "average_cost"),
            )

        settlement_frame = self.read_ledger("settlement_ledger.csv")
        settlements: list[PendingSettlement] = []
        settlement_ids: set[str] = set()

        for row in settlement_frame.itertuples(index=False):
            settlement_id = _required_text(row.settlement_id, "settlement_id")

            if settlement_id in settlement_ids:
                raise ValueError(f"Duplicate settlement ID: {settlement_id}")

            settlement_ids.add(settlement_id)
            side = Side(_required_text(row.side, "side"))
            gross_amount = _decimal(row.gross_amount, "gross_amount")
            total_costs = _decimal(row.fees_and_taxes, "fees_and_taxes")
            net_cash_effect = (
                -(gross_amount + total_costs)
                if side == Side.BUY
                else gross_amount - total_costs
            )
            settled_at = _optional_date(row.settled_at)

            if settled_at is not None:
                asof_candidates.append(settled_at)

            settlements.append(
                PendingSettlement(
                    settlement_id=settlement_id,
                    reference_id=_required_text(row.reference_id, "reference_id"),
                    trade_date=_required_date(row.trade_date, "trade_date"),
                    settlement_date=_required_date(
                        row.settlement_date,
                        "settlement_date",
                    ),
                    ticker=_required_text(row.ticker, "ticker").upper(),
                    side=side,
                    quantity=_integer(row.quantity, "quantity"),
                    gross_amount=gross_amount,
                    total_costs=total_costs,
                    net_cash_effect=net_cash_effect,
                    status=SettlementStatus(
                        _required_text(row.status, "status")
                    ),
                    settled_at=settled_at,
                )
            )

        execution_frame = self.read_ledger("executions.csv")

        if execution_frame["execution_id"].duplicated().any():
            raise ValueError("Duplicate execution IDs in executions.csv")

        processed_execution_ids = {
            _required_text(value, "execution_id")
            for value in execution_frame["execution_id"].tolist()
        }
        processed_execution_ids.update(
            settlement.reference_id for settlement in settlements
        )

        broker = PaperBrokerState(
            settled_cash=latest_cash.settled_cash_balance,
            unsettled_cash=latest_cash.unsettled_cash_balance,
            asof_date=max(asof_candidates),
            positions=positions,
            pending_settlements=settlements,
            cash_ledger_entries=cash_entries,
            processed_execution_ids=processed_execution_ids,
        )
        broker.reconcile()

        return broker

    def validate_all_ledgers(self) -> None:
        for filename in LEDGER_SCHEMAS:
            self.read_ledger(filename)

        self.load_broker_state()

    @contextmanager
    def account_lock(self) -> Iterator[None]:
        self.output_directory.mkdir(parents=True, exist_ok=True)

        try:
            with self.lock_path.open("x", encoding="utf-8") as lock_file:
                lock_file.write("paper-account operation in progress\n")
        except FileExistsError as error:
            raise RuntimeError(
                f"Paper account is locked by another operation: {self.lock_path}"
            ) from error

        try:
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)
