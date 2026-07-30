from __future__ import annotations

import pandas as pd

from scripts.audit_execution_coverage import (
    build_date_summary,
    build_expected_ticker_dates,
    classify_execution_status,
)


def membership_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "effective_from": ["2026-01-01", "2026-01-01"],
            "effective_to": [None, "2026-01-03"],
        }
    )


def test_build_expected_ticker_dates_respects_membership_exit() -> None:
    trading_dates = pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    )

    expected = build_expected_ticker_dates(trading_dates, membership_frame())

    aaa_dates = set(expected.loc[expected["ticker"] == "AAA", "date"])
    bbb_dates = set(expected.loc[expected["ticker"] == "BBB", "date"])

    assert aaa_dates == set(trading_dates)
    assert bbb_dates == set(trading_dates[:3])
    assert pd.Timestamp("2026-01-04") not in bbb_dates


def test_classify_execution_status_detects_every_category() -> None:
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
            "ticker": ["AAA"] * 4,
        }
    )

    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "ticker": ["AAA"] * 3,
            "open": [10.0, None, 10.0],
            "high": [10.5, None, 10.5],
            "low": [9.5, None, 9.5],
            "close": [10.2, None, 10.2],
            "volume": [1000.0, None, 1000.0],
            "value_traded": [10000.0, None, None],
            "value_traded_is_estimated": [False, False, True],
        }
    )

    coverage = classify_execution_status(expected, market)
    statuses = coverage.set_index("date")["execution_status"]

    assert statuses[pd.Timestamp("2026-01-01")] == "full_raw_execution_data"
    assert statuses[pd.Timestamp("2026-01-02")] == "raw_price_unavailable"
    assert statuses[pd.Timestamp("2026-01-03")] == "estimated_traded_value"
    assert statuses[pd.Timestamp("2026-01-04")] == "missing_row"


def test_build_date_summary_marks_estimated_value_dates_as_complete() -> None:
    coverage = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "execution_status": [
                "full_raw_execution_data",
                "estimated_traded_value",
                "full_raw_execution_data",
                "raw_price_unavailable",
            ],
        }
    )

    summary = build_date_summary(coverage).set_index("date")

    # Estimated traded value is a disclosed substitute, not a blocker.
    assert bool(summary.loc[pd.Timestamp("2026-01-01"), "execution_complete"])
    # A genuinely unavailable raw price blocks the date.
    assert not bool(summary.loc[pd.Timestamp("2026-01-02"), "execution_complete"])
    assert summary.loc[pd.Timestamp("2026-01-01"), "active_constituents"] == 2
