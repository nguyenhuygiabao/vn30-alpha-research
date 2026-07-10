from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src.paper_trading.calendar import TradingCalendar
from src.paper_trading.timing import (
    expected_completed_trading_day,
    resolve_signal_timing,
)


TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def local_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=TIMEZONE)


def test_after_close_data_executes_on_next_trading_day() -> None:
    timing = resolve_signal_timing(
        data_asof_date="2026-07-10",
        generated_at=local_datetime("2026-07-10T16:00:00"),
        timezone_name="Asia/Ho_Chi_Minh",
        data_update_cutoff="15:15",
        execution_submission_cutoff="08:45",
    )

    assert timing.data_asof_date == date(2026, 7, 10)
    assert timing.signal_date == date(2026, 7, 10)
    assert timing.intended_execution_date == date(2026, 7, 13)


def test_partial_same_day_data_is_rejected_before_cutoff() -> None:
    with pytest.raises(ValueError, match="expected completed trading day"):
        resolve_signal_timing(
            data_asof_date="2026-07-10",
            generated_at=local_datetime("2026-07-10T12:00:00"),
            timezone_name="Asia/Ho_Chi_Minh",
            data_update_cutoff="15:15",
            execution_submission_cutoff="08:45",
        )


def test_weekend_generation_uses_friday_data_for_monday() -> None:
    timing = resolve_signal_timing(
        data_asof_date="2026-07-10",
        generated_at=local_datetime("2026-07-11T10:00:00"),
        timezone_name="Asia/Ho_Chi_Minh",
        data_update_cutoff="15:15",
        execution_submission_cutoff="08:45",
    )

    assert timing.intended_execution_date == date(2026, 7, 13)


def test_execution_morning_is_allowed_before_submission_cutoff() -> None:
    timing = resolve_signal_timing(
        data_asof_date="2026-07-10",
        generated_at=local_datetime("2026-07-13T08:30:00"),
        timezone_name="Asia/Ho_Chi_Minh",
        data_update_cutoff="15:15",
        execution_submission_cutoff="08:45",
    )

    assert timing.intended_execution_date == date(2026, 7, 13)


def test_signal_is_rejected_after_execution_submission_cutoff() -> None:
    with pytest.raises(ValueError, match="submission cutoff has passed"):
        resolve_signal_timing(
            data_asof_date="2026-07-10",
            generated_at=local_datetime("2026-07-13T08:46:00"),
            timezone_name="Asia/Ho_Chi_Minh",
            data_update_cutoff="15:15",
            execution_submission_cutoff="08:45",
        )


def test_explicit_holiday_moves_execution_date() -> None:
    timing = resolve_signal_timing(
        data_asof_date="2026-07-10",
        generated_at=local_datetime("2026-07-10T16:00:00"),
        timezone_name="Asia/Ho_Chi_Minh",
        data_update_cutoff="15:15",
        execution_submission_cutoff="08:45",
        holiday_dates=["2026-07-13"],
    )

    assert timing.intended_execution_date == date(2026, 7, 14)


def test_expected_completed_day_rolls_back_before_close() -> None:
    completed = expected_completed_trading_day(
        generated_at=local_datetime("2026-07-10T12:00:00"),
        timezone_name="Asia/Ho_Chi_Minh",
        data_update_cutoff="15:15",
    )

    assert completed == date(2026, 7, 9)


def test_calendar_can_find_nearest_covered_trading_dates() -> None:
    calendar = TradingCalendar.from_weekdays("2026-07-01", "2026-07-20")

    assert calendar.latest_on_or_before("2026-07-12") == date(2026, 7, 10)
    assert calendar.earliest_on_or_after("2026-07-12") == date(2026, 7, 13)
