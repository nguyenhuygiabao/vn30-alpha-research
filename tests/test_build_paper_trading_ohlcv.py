from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_paper_trading_ohlcv import build_paper_trading_ohlcv
from src.paper_trading.data_update import OUTPUT_COLUMNS


MEMBERS = ("FPT", "VCB")


def market_history() -> pd.DataFrame:
    rows = []

    for ticker, base in (("FPT", 100.0), ("VCB", 70.0), ("FLC", 5.0)):
        for offset, day in enumerate(("2019-12-31", "2020-01-02", "2020-01-03")):
            close = base + offset
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 0.5,
                    "low": close - 1.0,
                    "close": close,
                    "adjusted_close": close * 0.98,
                    "volume": 1_000.0,
                    "value_traded": 12345.0,
                    "value_traded_is_estimated": False,
                }
            )

    return pd.DataFrame(rows)


def build(**overrides: object) -> pd.DataFrame:
    arguments: dict[str, object] = {
        "market": market_history(),
        "members": MEMBERS,
        "start_date": "2020-01-02",
    }
    arguments.update(overrides)

    return build_paper_trading_ohlcv(**arguments)


def test_output_matches_the_daily_update_schema() -> None:
    rebuilt = build()

    assert list(rebuilt.columns) == list(OUTPUT_COLUMNS)


def test_only_current_members_are_kept() -> None:
    rebuilt = build()

    assert sorted(rebuilt["ticker"].unique()) == ["FPT", "VCB"]


def test_history_before_the_start_date_is_dropped() -> None:
    rebuilt = build()

    assert rebuilt["date"].min() == pd.Timestamp("2020-01-02")


def test_raw_and_adjusted_prices_stay_separate() -> None:
    # Collapsing these is the defect the rebuild exists to correct.
    rebuilt = build()
    row = rebuilt.loc[rebuilt["ticker"] == "FPT"].reset_index(drop=True).loc[0]

    assert row["close"] == pytest.approx(101.0)
    assert row["adjusted_close"] == pytest.approx(101.0 * 0.98)


def test_value_traded_is_recomputed_not_taken_from_the_vendor() -> None:
    # The daily append computes close * volume, so the rebuilt history has to
    # use the same definition or the two halves of the file disagree.
    rebuilt = build()

    assert (
        rebuilt["value_traded"] == rebuilt["close"] * rebuilt["volume"]
    ).all()
    assert 12345.0 not in set(rebuilt["value_traded"])


def test_rows_are_sorted_by_date_then_ticker() -> None:
    rebuilt = build()
    expected = rebuilt.sort_values(["date", "ticker"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(rebuilt, expected)


def test_a_member_missing_from_the_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing index members.*SSB"):
        build(members=("FPT", "SSB"))


def test_missing_values_are_rejected() -> None:
    market = market_history()
    market.loc[market["ticker"] == "FPT", "adjusted_close"] = None

    with pytest.raises(ValueError, match="missing values"):
        build(market=market)


def test_duplicate_ticker_dates_are_rejected() -> None:
    history = market_history()
    duplicate = history.loc[history["ticker"] == "FPT"].tail(1)
    market = pd.concat([history, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate ticker-date rows"):
        build(market=market)
