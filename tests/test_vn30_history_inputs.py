from pathlib import Path

import pandas as pd

from src.universe_reconstruction import (
    reconstruct_constituent_history,
)


BASE = Path("config/vn30_history")


def test_complete_point_in_time_vn30_history() -> None:
    result = reconstruct_constituent_history(
        initial_snapshot=pd.read_csv(BASE / "initial_snapshot.csv"),
        review_calendar=pd.read_csv(BASE / "review_calendar.csv"),
        membership_changes=pd.read_csv(BASE / "membership_changes.csv"),
        source_manifest=pd.read_csv(BASE / "source_manifest.csv"),
        expected_size=30,
    )

    counts = (
        result.snapshots
        .groupby("effective_date")["ticker"]
        .nunique()
    )

    assert len(counts) == 21
    assert counts.index.min() == pd.Timestamp("2016-07-25")
    assert counts.index.max() == pd.Timestamp("2026-05-13")
    assert counts.eq(30).all()
    assert len(result.snapshots) == 630
    assert len(result.ticker_pool) == 60
    assert result.audit["verified"].all()
    assert result.audit["status"].eq("pass").all()


def test_history_preserves_former_constituents() -> None:
    result = reconstruct_constituent_history(
        initial_snapshot=pd.read_csv(BASE / "initial_snapshot.csv"),
        review_calendar=pd.read_csv(BASE / "review_calendar.csv"),
        membership_changes=pd.read_csv(BASE / "membership_changes.csv"),
        source_manifest=pd.read_csv(BASE / "source_manifest.csv"),
        expected_size=30,
    )

    initial = set(
        result.snapshots.loc[
            result.snapshots["effective_date"].eq("2016-07-25"),
            "ticker",
        ]
    )

    assert {
        "FLC",
        "HAG",
        "HNG",
        "ITA",
        "NT2",
        "PVD",
    }.issubset(initial)


def test_mid_cycle_dgc_bsr_replacement() -> None:
    result = reconstruct_constituent_history(
        initial_snapshot=pd.read_csv(BASE / "initial_snapshot.csv"),
        review_calendar=pd.read_csv(BASE / "review_calendar.csv"),
        membership_changes=pd.read_csv(BASE / "membership_changes.csv"),
        source_manifest=pd.read_csv(BASE / "source_manifest.csv"),
        expected_size=30,
    )

    latest = set(
        result.snapshots.loc[
            result.snapshots["effective_date"].eq("2026-05-13"),
            "ticker",
        ]
    )

    assert "BSR" in latest
    assert "DGC" not in latest
