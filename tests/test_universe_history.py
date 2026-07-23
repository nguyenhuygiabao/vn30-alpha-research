from __future__ import annotations

import pandas as pd

from src.universe_history import (
    filter_to_expanding_alumni_universe,
    filter_to_point_in_time_universe,
    filter_to_universe,
    historical_ticker_pool,
    normalize_constituent_snapshots,
    snapshots_to_membership_history,
    validate_constituent_snapshot_coverage,
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


def constituent_snapshots() -> pd.DataFrame:
    rows = []
    baskets = {
        "2020-01-01": ["AAA", "BBB", "CCC"],
        "2020-07-01": ["BBB", "CCC", "DDD"],
        "2021-01-01": ["AAA", "CCC", "DDD"],
    }

    for effective_date, tickers in baskets.items():
        for ticker in tickers:
            rows.append(
                {
                    "effective_date": effective_date,
                    "ticker": ticker,
                    "source_url": f"https://example.com/{effective_date}",
                }
            )

    return pd.DataFrame(rows)


def test_normalize_constituent_snapshots_requires_complete_baskets() -> None:
    incomplete = constituent_snapshots().loc[
        lambda data: ~(
            data["effective_date"].eq("2020-01-01")
            & data["ticker"].eq("AAA")
        )
    ]

    try:
        normalize_constituent_snapshots(incomplete, expected_size=3)
    except ValueError as error:
        assert "must contain 3 tickers" in str(error)
    else:
        raise AssertionError("An incomplete constituent snapshot was accepted")


def test_normalize_constituent_snapshots_rejects_duplicates() -> None:
    snapshots = constituent_snapshots()
    duplicated = pd.concat([snapshots, snapshots.iloc[[0]]], ignore_index=True)

    try:
        normalize_constituent_snapshots(duplicated, expected_size=3)
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("A duplicate constituent row was accepted")


def test_snapshot_conversion_handles_exit_and_reentry() -> None:
    history = snapshots_to_membership_history(
        constituent_snapshots(),
        expected_size=3,
    )

    aaa = history.loc[history["ticker"].eq("AAA")].reset_index(drop=True)

    assert aaa["effective_from"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2021-01-01"),
    ]
    assert aaa.loc[0, "effective_to"] == pd.Timestamp("2020-06-30")
    assert pd.isna(aaa.loc[1, "effective_to"])
    assert historical_ticker_pool(history) == ["AAA", "BBB", "CCC", "DDD"]


def test_snapshot_coverage_rejects_missing_review_period() -> None:
    snapshots = constituent_snapshots().loc[
        lambda data: ~data["effective_date"].eq("2020-07-01")
    ]

    try:
        validate_constituent_snapshot_coverage(
            snapshots,
            start_date="2020-01-02",
            end_date="2021-01-01",
            expected_size=3,
            maximum_gap_days=220,
        )
    except ValueError as error:
        assert "excessive gaps" in str(error)
    else:
        raise AssertionError("Missing review-period coverage was accepted")


def test_snapshot_coverage_requires_pre_start_snapshot() -> None:
    try:
        validate_constituent_snapshot_coverage(
            constituent_snapshots(),
            start_date="2019-12-31",
            end_date="2020-02-01",
            expected_size=3,
        )
    except ValueError as error:
        assert "backtest start" in str(error)
    else:
        raise AssertionError("Coverage without a starting snapshot was accepted")
