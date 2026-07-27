from __future__ import annotations

import shutil

import pandas as pd
import pytest

from src.vn30_universe import (
    DEFAULT_HISTORY_DIR,
    build_constituent_snapshots,
    constituents_on_date,
    historical_tickers,
)


def test_builds_complete_constituent_snapshots() -> None:
    snapshots = build_constituent_snapshots()
    counts = snapshots.groupby("effective_date").size()

    assert len(snapshots) == 630
    assert snapshots["effective_date"].nunique() == 21
    assert counts.eq(30).all()
    assert not snapshots.duplicated(
        ["effective_date", "ticker"]
    ).any()

    expected_tickers = tuple(
        sorted(snapshots["ticker"].unique())
    )

    assert len(expected_tickers) == 60
    assert historical_tickers() == expected_tickers


def test_snapshots_apply_every_membership_change() -> None:
    snapshots = build_constituent_snapshots()

    changes = pd.read_csv(
        DEFAULT_HISTORY_DIR / "membership_changes.csv",
        dtype="string",
    )
    changes["effective_date"] = pd.to_datetime(
        changes["effective_date"],
        format="%Y-%m-%d",
    )
    changes["action"] = changes["action"].str.lower()
    changes["ticker"] = changes["ticker"].str.upper()

    snapshots_by_date = {
        effective_date: set(group["ticker"])
        for effective_date, group in snapshots.groupby(
            "effective_date"
        )
    }

    review_dates = sorted(snapshots_by_date)
    previous_members = snapshots_by_date[review_dates[0]]

    for effective_date in review_dates[1:]:
        period_changes = changes.loc[
            changes["effective_date"].eq(effective_date)
        ]

        removals = set(
            period_changes.loc[
                period_changes["action"].eq("remove"),
                "ticker",
            ]
        )
        additions = set(
            period_changes.loc[
                period_changes["action"].eq("add"),
                "ticker",
            ]
        )

        expected_members = (
            previous_members - removals
        ) | additions

        assert (
            snapshots_by_date[effective_date]
            == expected_members
        )
        assert len(expected_members) == 30

        previous_members = expected_members


def test_constituents_on_date_uses_latest_snapshot() -> None:
    snapshots = build_constituent_snapshots()
    review_dates = sorted(
        snapshots["effective_date"].unique()
    )

    first_date = pd.Timestamp(review_dates[0])
    second_date = pd.Timestamp(review_dates[1])
    last_date = pd.Timestamp(review_dates[-1])

    first_members = tuple(
        snapshots.loc[
            snapshots["effective_date"].eq(first_date),
            "ticker",
        ].sort_values()
    )
    second_members = tuple(
        snapshots.loc[
            snapshots["effective_date"].eq(second_date),
            "ticker",
        ].sort_values()
    )
    last_members = tuple(
        snapshots.loc[
            snapshots["effective_date"].eq(last_date),
            "ticker",
        ].sort_values()
    )

    assert constituents_on_date(first_date) == first_members
    assert (
        constituents_on_date(second_date - pd.Timedelta(days=1))
        == first_members
    )
    assert constituents_on_date(second_date) == second_members
    assert (
        constituents_on_date(last_date + pd.Timedelta(days=365))
        == last_members
    )

    with pytest.raises(
        ValueError,
        match="precedes the first VN30 snapshot",
    ):
        constituents_on_date(
            first_date - pd.Timedelta(days=1)
        )


def test_rejects_unexpected_membership_action(
    tmp_path,
) -> None:
    history_dir = tmp_path / "vn30_history"
    shutil.copytree(DEFAULT_HISTORY_DIR, history_dir)

    changes_path = history_dir / "membership_changes.csv"
    changes = pd.read_csv(changes_path, dtype="string")
    changes.loc[0, "action"] = "replace"
    changes.to_csv(changes_path, index=False)

    with pytest.raises(
        ValueError,
        match="Unexpected membership actions",
    ):
        build_constituent_snapshots(history_dir)
