from __future__ import annotations

import pandas as pd
import pytest

from src.exchange_history import (
    attach_exchange_history,
    normalize_exchange_history,
)


def exchange_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "effective_from": [
                "2020-01-01",
                "2020-01-03",
                "2020-01-01",
            ],
            "effective_to": [
                "2020-01-02",
                "",
                "",
            ],
            "exchange": [
                "UPCOM",
                "HOSE",
                "HNX",
            ],
        }
    )


def test_attach_exchange_respects_transfer_date() -> None:
    market = pd.DataFrame(
        {
            "date": [
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
            ],
            "ticker": ["AAA", "AAA", "BBB"],
            "close": [10.0, 11.0, 20.0],
        }
    )

    attached = attach_exchange_history(
        market,
        exchange_history(),
    )

    assert attached["exchange"].tolist() == [
        "UPCOM",
        "HOSE",
        "HNX",
    ]


def test_exchange_ranges_are_inclusive() -> None:
    market = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "ticker": ["AAA"],
        }
    )

    attached = attach_exchange_history(
        market,
        exchange_history(),
    )

    assert attached.loc[0, "exchange"] == "UPCOM"


def test_missing_exchange_history_fails() -> None:
    market = pd.DataFrame(
        {
            "date": ["2019-12-31"],
            "ticker": ["AAA"],
        }
    )

    with pytest.raises(
        ValueError,
        match="No exchange history",
    ):
        attach_exchange_history(
            market,
            exchange_history(),
        )


def test_overlapping_exchange_ranges_fail() -> None:
    overlapping = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "effective_from": [
                "2020-01-01",
                "2020-01-03",
            ],
            "effective_to": [
                "2020-01-05",
                "",
            ],
            "exchange": [
                "UPCOM",
                "HOSE",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="overlaps",
    ):
        normalize_exchange_history(overlapping)
