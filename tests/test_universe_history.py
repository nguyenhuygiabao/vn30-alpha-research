from __future__ import annotations

import pandas as pd

from src.universe_history import (
    filter_to_expanding_alumni_universe,
    filter_to_point_in_time_universe,
    filter_to_universe,
    historical_ticker_pool,
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



def reentry_membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "effective_from": [
                "2020-01-01",
                "2020-01-05",
                "2020-01-01",
                "2020-01-03",
            ],
            "effective_to": [
                "2020-01-02",
                "",
                "",
                "",
            ],
        }
    )


def test_strict_universe_respects_exit_and_reentry() -> None:
    market = pd.DataFrame(
        {
            "date": [
                "2020-01-02",
                "2020-01-03",
                "2020-01-05",
            ],
            "ticker": ["AAA", "AAA", "AAA"],
            "close": [10.0, 11.0, 12.0],
        }
    )

    filtered = filter_to_universe(
        market,
        reentry_membership(),
        mode="strict_vn30",
    )

    assert filtered["date"].tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-05"),
    ]


def test_expanding_alumni_keeps_former_member_after_exit() -> None:
    market = pd.DataFrame(
        {
            "date": [
                "2019-12-31",
                "2020-01-02",
                "2020-01-03",
                "2020-01-05",
            ],
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "close": [9.0, 10.0, 11.0, 12.0],
        }
    )

    filtered = filter_to_expanding_alumni_universe(
        market,
        reentry_membership(),
    )

    assert filtered["date"].tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-05"),
    ]


def test_expanding_alumni_prevents_future_constituent_leakage() -> None:
    market = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "ticker": ["CCC", "CCC"],
            "close": [20.0, 21.0],
        }
    )

    filtered = filter_to_universe(
        market,
        reentry_membership(),
        mode="expanding_alumni",
    )

    assert filtered["date"].tolist() == [pd.Timestamp("2020-01-03")]


def test_historical_ticker_pool_contains_full_constituent_union() -> None:
    pool = historical_ticker_pool(reentry_membership())

    assert pool == ["AAA", "BBB", "CCC"]


def test_filter_to_universe_rejects_unknown_mode() -> None:
    market = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "ticker": ["AAA"],
        }
    )

    try:
        filter_to_universe(
            market,
            reentry_membership(),
            mode="future_constituents",
        )
    except ValueError as error:
        assert "Unknown universe mode" in str(error)
    else:
        raise AssertionError("Unknown universe mode was accepted")
