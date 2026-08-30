from __future__ import annotations

import pandas as pd
import pytest

from scripts.extend_fiinquant_history import (
    active_tickers,
    compare_raw_overlap,
    measure_adjustment_factors,
    merge_history,
    rescale_stored_adjusted,
)


STORED_DATES = ("2026-07-15", "2026-07-16", "2026-07-17")
WINDOW_START = pd.Timestamp("2026-07-16")


def adjusted_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "timestamp": pd.Timestamp(day),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
            }
            for ticker, day, close in rows
        ]
    )


def raw_frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "timestamp": pd.Timestamp(day),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": volume,
                "value": close * volume,
            }
            for ticker, day, close, volume in rows
        ]
    )


def test_unchanged_basis_reports_unit_factors() -> None:
    stored = adjusted_frame(
        [("FPT", day, 100.0) for day in STORED_DATES]
        + [("VCB", day, 70.0) for day in STORED_DATES]
    )

    factors, inconsistent = measure_adjustment_factors(stored, stored)

    assert inconsistent == []
    assert factors == pytest.approx({"FPT": 1.0, "VCB": 1.0})


def test_corporate_action_is_measured_as_a_constant_factor() -> None:
    """A split restates every earlier adjusted price by one ratio."""
    stored = adjusted_frame([("FPT", day, 100.0) for day in STORED_DATES])
    # A 2-for-1 split after the stored history halves every earlier price.
    fetched = adjusted_frame([("FPT", day, 50.0) for day in STORED_DATES])

    factors, inconsistent = measure_adjustment_factors(stored, fetched)

    assert inconsistent == []
    assert factors["FPT"] == pytest.approx(0.5)


def test_non_constant_ratio_is_rejected_rather_than_averaged() -> None:
    """Prices that move by different ratios are not a clean restatement."""
    stored = adjusted_frame([("FPT", day, 100.0) for day in STORED_DATES])
    fetched = adjusted_frame(
        [
            ("FPT", "2026-07-15", 50.0),
            ("FPT", "2026-07-16", 80.0),
            ("FPT", "2026-07-17", 95.0),
        ]
    )

    factors, inconsistent = measure_adjustment_factors(stored, fetched)

    assert "FPT" not in factors
    assert len(inconsistent) == 1
    assert "FPT" in inconsistent[0]


def test_rescaling_touches_only_restated_tickers_before_the_window() -> None:
    stored = adjusted_frame(
        [("FPT", day, 100.0) for day in STORED_DATES]
        + [("VCB", day, 70.0) for day in STORED_DATES]
    )

    rescaled, applied = rescale_stored_adjusted(
        stored,
        {"FPT": 0.5, "VCB": 1.0},
        WINDOW_START,
    )

    before = rescaled.loc[
        rescaled["ticker"].eq("FPT")
        & rescaled["timestamp"].lt(WINDOW_START)
    ]
    inside = rescaled.loc[
        rescaled["ticker"].eq("FPT")
        & rescaled["timestamp"].ge(WINDOW_START)
    ]
    untouched = rescaled.loc[rescaled["ticker"].eq("VCB")]

    assert before["close"].tolist() == [50.0]
    assert inside["close"].tolist() == [100.0, 100.0]
    assert untouched["close"].tolist() == [70.0, 70.0, 70.0]
    assert len(applied) == 1
    assert "FPT" in applied[0]


def test_rescaling_applies_to_every_price_column() -> None:
    stored = adjusted_frame([("FPT", "2026-07-15", 100.0)])

    rescaled, _ = rescale_stored_adjusted(stored, {"FPT": 0.5}, WINDOW_START)

    row = rescaled.iloc[0]
    assert row["open"] == pytest.approx(50.0)
    assert row["high"] == pytest.approx(50.0)
    assert row["low"] == pytest.approx(50.0)
    assert row["close"] == pytest.approx(50.0)


def test_merge_prefers_fetched_rows_inside_the_window() -> None:
    stored = raw_frame([("FPT", day, 100.0, 1000.0) for day in STORED_DATES])
    fetched = raw_frame(
        [
            ("FPT", "2026-07-16", 111.0, 1100.0),
            ("FPT", "2026-07-17", 112.0, 1200.0),
            ("FPT", "2026-07-20", 113.0, 1300.0),
        ]
    )

    merged = merge_history(stored, fetched, WINDOW_START)

    assert len(merged) == 4
    assert merged["timestamp"].max() == pd.Timestamp("2026-07-20")
    # The pre-window row survives untouched; the overlap takes vendor values.
    assert merged.loc[
        merged["timestamp"].eq(pd.Timestamp("2026-07-15")), "close"
    ].tolist() == [100.0]
    assert merged.loc[
        merged["timestamp"].eq(pd.Timestamp("2026-07-16")), "close"
    ].tolist() == [111.0]


def test_merge_rejects_duplicate_keys() -> None:
    stored = raw_frame([("FPT", "2026-07-15", 100.0, 1000.0)])
    fetched = raw_frame(
        [
            ("FPT", "2026-07-20", 113.0, 1300.0),
            ("FPT", "2026-07-20", 114.0, 1400.0),
        ]
    )

    with pytest.raises(SystemExit, match="duplicate"):
        merge_history(stored, fetched, WINDOW_START)


def test_raw_overlap_difference_is_reported() -> None:
    stored = raw_frame([("FPT", day, 100.0, 1000.0) for day in STORED_DATES])
    fetched = raw_frame(
        [
            ("FPT", "2026-07-16", 100.0, 1000.0),
            ("FPT", "2026-07-17", 140.0, 1000.0),
        ]
    )

    differences = compare_raw_overlap(stored, fetched)

    assert differences
    assert any("close" in line for line in differences)


def test_identical_raw_overlap_reports_nothing() -> None:
    stored = raw_frame([("FPT", day, 100.0, 1000.0) for day in STORED_DATES])

    assert compare_raw_overlap(stored, stored) == []


def test_active_tickers_excludes_archive_and_delisted_names() -> None:
    rows = [("FPT", day, 100.0, 1000.0) for day in STORED_DATES]
    rows += [("FLC", day, 5.0, 500.0) for day in STORED_DATES]
    # Stopped trading long before the end of the stored history.
    rows += [("ROS", "2021-01-29", 3.0, 300.0)]
    rows += [("HNG", "2024-01-05", 6.0, 600.0)]

    tickers = active_tickers(raw_frame(rows))

    assert tickers == ["FPT"]


def test_missing_overlap_is_refused() -> None:
    stored = adjusted_frame([("FPT", "2026-07-15", 100.0)])
    fetched = adjusted_frame([("FPT", "2026-08-20", 100.0)])

    with pytest.raises(SystemExit, match="overlap"):
        measure_adjustment_factors(stored, fetched)
