from __future__ import annotations

import pytest

from src.exchange_rules import (
    FIRST_TRADING_DAY,
    RESUMPTION_25_PLUS,
    TREASURY_SHARE_DISTRIBUTION,
    price_limit_fraction,
    tick_size_vnd,
)


@pytest.mark.parametrize(
    ("exchange", "expected"),
    [
        ("HOSE", 0.07),
        ("HNX", 0.10),
        ("UPCOM", 0.15),
    ],
)
def test_regular_price_limits(
    exchange: str,
    expected: float,
) -> None:
    assert price_limit_fraction(exchange) == expected


@pytest.mark.parametrize(
    ("exchange", "expected"),
    [
        ("HOSE", 0.20),
        ("HNX", 0.30),
        ("UPCOM", 0.40),
    ],
)
def test_first_day_price_limits(
    exchange: str,
    expected: float,
) -> None:
    assert price_limit_fraction(
        exchange,
        FIRST_TRADING_DAY,
    ) == expected


def test_resumption_uses_special_limit() -> None:
    assert price_limit_fraction(
        "UPCOM",
        RESUMPTION_25_PLUS,
    ) == 0.40


def test_treasury_share_distribution_uses_special_limit() -> None:
    assert price_limit_fraction(
        "HOSE",
        TREASURY_SHARE_DISTRIBUTION,
    ) == 0.20


def test_hose_historical_tick_change() -> None:
    assert tick_size_vnd(
        "HOSE",
        9_000,
        "2016-09-09",
    ) == 100

    assert tick_size_vnd(
        "HOSE",
        9_000,
        "2016-09-12",
    ) == 10


def test_current_hose_tiered_ticks() -> None:
    assert tick_size_vnd(
        "HOSE",
        9_990,
        "2026-01-02",
    ) == 10

    assert tick_size_vnd(
        "HOSE",
        10_000,
        "2026-01-02",
    ) == 50

    assert tick_size_vnd(
        "HOSE",
        49_950,
        "2026-01-02",
    ) == 50

    assert tick_size_vnd(
        "HOSE",
        50_000,
        "2026-01-02",
    ) == 100


@pytest.mark.parametrize("exchange", ["HNX", "UPCOM"])
def test_hnx_and_upcom_ticks(
    exchange: str,
) -> None:
    assert tick_size_vnd(
        exchange,
        12_345,
        "2026-01-02",
    ) == 100
