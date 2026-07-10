from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable


DateLike = date | datetime | str


def normalize_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        return date.fromisoformat(value)

    raise TypeError(f"Unsupported date value: {value!r}")


@dataclass(frozen=True)
class TradingCalendar:
    trading_days: tuple[date, ...]

    def __post_init__(self) -> None:
        normalized_days = tuple(
            sorted({normalize_date(value) for value in self.trading_days})
        )

        if not normalized_days:
            raise ValueError("Trading calendar must contain at least one trading day")

        object.__setattr__(self, "trading_days", normalized_days)

    @classmethod
    def from_dates(cls, trading_days: Iterable[DateLike]) -> TradingCalendar:
        return cls(tuple(normalize_date(value) for value in trading_days))

    @classmethod
    def from_weekdays(
        cls,
        start_date: DateLike,
        end_date: DateLike,
        holidays: Iterable[DateLike] = (),
    ) -> TradingCalendar:
        start = normalize_date(start_date)
        end = normalize_date(end_date)

        if end < start:
            raise ValueError("Trading calendar end date cannot be before start date")

        holiday_dates = {normalize_date(value) for value in holidays}
        trading_days: list[date] = []
        current = start

        while current <= end:
            if current.weekday() < 5 and current not in holiday_dates:
                trading_days.append(current)

            current += timedelta(days=1)

        return cls(tuple(trading_days))

    def is_trading_day(self, value: DateLike) -> bool:
        target = normalize_date(value)
        index = bisect_left(self.trading_days, target)

        return index < len(self.trading_days) and self.trading_days[index] == target

    def add_trading_days(self, value: DateLike, offset: int) -> date:
        target = normalize_date(value)
        index = bisect_left(self.trading_days, target)

        if index >= len(self.trading_days) or self.trading_days[index] != target:
            raise ValueError(f"Date is not in the trading calendar: {target}")

        result_index = index + offset

        if result_index < 0 or result_index >= len(self.trading_days):
            raise ValueError(
                f"Trading calendar does not cover offset {offset} from {target}"
            )

        return self.trading_days[result_index]

    def next_trading_day(self, value: DateLike) -> date:
        return self.add_trading_days(value, 1)

    def latest_on_or_before(self, value: DateLike) -> date:
        target = normalize_date(value)
        index = bisect_right(self.trading_days, target) - 1

        if index < 0:
            raise ValueError(f"Trading calendar has no date on or before {target}")

        return self.trading_days[index]

    def earliest_on_or_after(self, value: DateLike) -> date:
        target = normalize_date(value)
        index = bisect_left(self.trading_days, target)

        if index >= len(self.trading_days):
            raise ValueError(f"Trading calendar has no date on or after {target}")

        return self.trading_days[index]

    def settlement_date(self, trade_date: DateLike, lag_trading_days: int = 2) -> date:
        if lag_trading_days < 0:
            raise ValueError("Settlement lag cannot be negative")

        return self.add_trading_days(trade_date, lag_trading_days)
