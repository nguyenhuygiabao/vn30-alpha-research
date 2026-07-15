from __future__ import annotations

import pandas as pd

from src.universe_history import (
    filter_to_point_in_time_universe,
    normalize_membership_history,
    summarize_membership_coverage,
)


def membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "effective_from": ["2020-01-01", "2020-01-01", "2020-01-03"],
            "effective_to": ["2020-01-02", "", ""],
        }
    )


def test_point_in_time_filter_excludes_inactive_ticker_dates() -> None:
    market = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-03"],
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "close": [10.0, 11.0, 20.0, 30.0],
        }
    )
    filtered = filter_to_point_in_time_universe(market, membership())

    assert list(zip(filtered["date"], filtered["ticker"])) == [
        (pd.Timestamp("2020-01-02"), "AAA"),
        (pd.Timestamp("2020-01-03"), "BBB"),
        (pd.Timestamp("2020-01-03"), "CCC"),
    ]


def test_membership_history_rejects_overlapping_intervals() -> None:
    overlapping = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "effective_from": ["2020-01-01", "2020-01-03"],
            "effective_to": ["2020-01-05", "2020-01-10"],
        }
    )
    try:
        normalize_membership_history(overlapping)
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Overlapping membership intervals were accepted")


def test_membership_coverage_reports_incomplete_dates() -> None:
    coverage = summarize_membership_coverage(
        membership(),
        pd.to_datetime(["2020-01-02", "2020-01-03"]),
        expected_constituents=2,
    )

    assert coverage["active_constituents"].tolist() == [2, 2]
    assert coverage["coverage_complete"].all()
