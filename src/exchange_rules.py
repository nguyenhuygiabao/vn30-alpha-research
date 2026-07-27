from __future__ import annotations

from typing import Any

import pandas as pd


REGULAR_SESSION = "REGULAR"
FIRST_TRADING_DAY = "FIRST_TRADING_DAY"
RESUMPTION_25_PLUS = "RESUMPTION_25_PLUS"
TREASURY_SHARE_DISTRIBUTION = (
    "TREASURY_SHARE_DISTRIBUTION"
)

SUPPORTED_SESSION_KINDS = frozenset(
    {
        REGULAR_SESSION,
        FIRST_TRADING_DAY,
        RESUMPTION_25_PLUS,
        TREASURY_SHARE_DISTRIBUTION,
    }
)

REGULAR_PRICE_LIMITS = {
    "HOSE": 0.07,
    "HNX": 0.10,
    "UPCOM": 0.15,
}

SPECIAL_PRICE_LIMITS = {
    "HOSE": 0.20,
    "HNX": 0.30,
    "UPCOM": 0.40,
}

HOSE_TIERED_TICK_START = pd.Timestamp("2016-09-12")


def normalize_exchange(exchange: str) -> str:
    normalized = str(exchange).strip().upper()

    if normalized not in REGULAR_PRICE_LIMITS:
        raise ValueError(
            f"Unsupported exchange: {exchange}"
        )

    return normalized


def price_limit_fraction(
    exchange: str,
    session_kind: str = REGULAR_SESSION,
) -> float:
    normalized_exchange = normalize_exchange(exchange)
    normalized_session = (
        str(session_kind).strip().upper()
    )

    if normalized_session not in SUPPORTED_SESSION_KINDS:
        raise ValueError(
            f"Unsupported session kind: {session_kind}"
        )

    if normalized_session == REGULAR_SESSION:
        return REGULAR_PRICE_LIMITS[
            normalized_exchange
        ]

    return SPECIAL_PRICE_LIMITS[
        normalized_exchange
    ]


def tick_size_vnd(
    exchange: str,
    price_vnd: float,
    trade_date: Any,
) -> int:
    normalized_exchange = normalize_exchange(exchange)

    price = float(price_vnd)

    if not pd.notna(price) or price <= 0:
        raise ValueError(
            "Price must be a positive finite value"
        )

    date = pd.Timestamp(trade_date).normalize()

    if normalized_exchange in {"HNX", "UPCOM"}:
        return 100

    if date < HOSE_TIERED_TICK_START:
        return 100

    if price < 10_000:
        return 10

    if price < 50_000:
        return 50

    return 100
