from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

from src.paper_trading.calendar import DateLike, TradingCalendar, normalize_date


@dataclass(frozen=True)
class SignalTiming:
    data_asof_date: date
    signal_date: date
    intended_execution_date: date
    generated_at: datetime

    def to_row(self) -> dict[str, str]:
        return {
            "data_asof_date": self.data_asof_date.isoformat(),
            "signal_date": self.signal_date.isoformat(),
            "intended_execution_date": self.intended_execution_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
        }


def parse_market_time(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise ValueError(f"Invalid market time: {value}. Expected HH:MM") from error


def build_operational_calendar(
    reference_date: DateLike,
    holiday_dates: Iterable[DateLike] = (),
    buffer_days: int = 45,
) -> TradingCalendar:
    reference = normalize_date(reference_date)

    if buffer_days < 7:
        raise ValueError("Operational calendar buffer must be at least 7 days")

    return TradingCalendar.from_weekdays(
        reference - timedelta(days=buffer_days),
        reference + timedelta(days=buffer_days),
        holidays=holiday_dates,
    )


def expected_completed_trading_day(
    generated_at: datetime,
    timezone_name: str,
    data_update_cutoff: str,
    holiday_dates: Iterable[DateLike] = (),
) -> date:
    timezone = ZoneInfo(timezone_name)

    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")

    local_time = generated_at.astimezone(timezone)
    cutoff = parse_market_time(data_update_cutoff)
    calendar = build_operational_calendar(
        local_time.date(),
        holiday_dates=holiday_dates,
    )

    if calendar.is_trading_day(local_time.date()) and local_time.time() >= cutoff:
        return local_time.date()

    return calendar.latest_on_or_before(local_time.date() - timedelta(days=1))


def resolve_signal_timing(
    data_asof_date: DateLike,
    generated_at: datetime,
    timezone_name: str,
    data_update_cutoff: str,
    execution_submission_cutoff: str,
    holiday_dates: Iterable[DateLike] = (),
    enforce_execution_cutoff: bool = True,
) -> SignalTiming:
    data_date = normalize_date(data_asof_date)
    timezone = ZoneInfo(timezone_name)

    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")

    local_time = generated_at.astimezone(timezone)
    expected_data_date = expected_completed_trading_day(
        generated_at=local_time,
        timezone_name=timezone_name,
        data_update_cutoff=data_update_cutoff,
        holiday_dates=holiday_dates,
    )

    if data_date != expected_data_date:
        raise ValueError(
            f"Latest data date {data_date} does not match the expected completed "
            f"trading day {expected_data_date}"
        )

    calendar = build_operational_calendar(
        data_date,
        holiday_dates=holiday_dates,
    )
    execution_date = calendar.next_trading_day(data_date)
    submission_cutoff = parse_market_time(execution_submission_cutoff)

    if enforce_execution_cutoff and local_time.date() > execution_date:
        raise ValueError(
            f"Signal is stale because intended execution date {execution_date} "
            f"has already passed"
        )

    if (
        enforce_execution_cutoff
        and local_time.date() == execution_date
        and local_time.time() > submission_cutoff
    ):
        raise ValueError(
            f"Signal is stale because the {submission_cutoff.strftime('%H:%M')} "
            "execution submission cutoff has passed"
        )

    return SignalTiming(
        data_asof_date=data_date,
        signal_date=data_date,
        intended_execution_date=execution_date,
        generated_at=local_time,
    )
