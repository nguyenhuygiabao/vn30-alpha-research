from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.market_data_schema import (
    load_canonical_market_history,
)


def write_market_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def market_row(
    *,
    ticker: str = "AAA",
    timestamp: str = "2026-07-22",
    open_price: float = 10_000,
    high: float = 11_000,
    low: float = 9_000,
    close: float = 10_500,
    volume: float = 100,
    value: float = 1_000_000,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "timestamp": timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "value": value,
    }


def test_loader_uses_raw_ohlc_and_adjusted_close(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [market_row()],
    )
    write_market_csv(
        adjusted_path,
        [
            market_row(
                open_price=5_000,
                high=5_500,
                low=4_500,
                close=5_250,
            )
        ],
    )

    market = load_canonical_market_history(
        raw_path,
        adjusted_path,
    )
    row = market.iloc[0]

    assert row["date"] == pd.Timestamp("2026-07-22")
    assert row["ticker"] == "AAA"
    assert row["open"] == 10.0
    assert row["high"] == 11.0
    assert row["low"] == 9.0
    assert row["close"] == 10.5
    assert row["adjusted_close"] == 5.25
    assert row["volume"] == 100
    assert row["value_traded"] == 1_000_000
    assert market.attrs["raw_price_fallback_rows"] == 0
    assert market.attrs["raw_price_fallback_tickers"] == []
    assert market.attrs["price_unit"] == "thousand_vnd"


def test_loader_preserves_adjusted_only_history(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [market_row()],
    )
    write_market_csv(
        adjusted_path,
        [
            market_row(close=5_250),
            market_row(
                ticker="BBB",
                open_price=20_000,
                high=21_000,
                low=19_000,
                close=20_500,
                volume=200,
                value=4_000_000,
            ),
        ],
    )

    market = load_canonical_market_history(
        raw_path,
        adjusted_path,
    )
    fallback = market.loc[
        market["ticker"].eq("BBB")
    ].iloc[0]

    assert len(market) == 2
    assert pd.isna(fallback["open"])
    assert pd.isna(fallback["high"])
    assert pd.isna(fallback["low"])
    assert pd.isna(fallback["close"])
    assert fallback["adjusted_close"] == 20.5
    assert fallback["volume"] == 200
    assert fallback["value_traded"] == 4_000_000
    assert market.attrs["raw_price_fallback_rows"] == 1
    assert market.attrs[
        "raw_price_fallback_tickers"
    ] == ["BBB"]


def test_loader_preserves_missing_adjusted_only_value(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [market_row()],
    )
    write_market_csv(
        adjusted_path,
        [
            market_row(),
            market_row(
                ticker="FLC",
                value=float("nan"),
            ),
        ],
    )

    market = load_canonical_market_history(
        raw_path,
        adjusted_path,
    )
    flc = market.loc[
        market["ticker"].eq("FLC")
    ].iloc[0]

    assert pd.isna(flc["open"])
    assert pd.isna(flc["value_traded"])
    assert flc["adjusted_close"] == 10.5
    assert flc["volume"] == 100
    assert market.attrs["missing_value_traded_rows"] == 1
    assert market.attrs[
        "missing_value_traded_tickers"
    ] == ["FLC"]


def test_loader_rejects_missing_value_when_raw_exists(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [market_row()],
    )
    write_market_csv(
        adjusted_path,
        [market_row(value=float("nan"))],
    )

    with pytest.raises(
        ValueError,
        match="despite available raw market data",
    ):
        load_canonical_market_history(
            raw_path,
            adjusted_path,
        )


def test_loader_allows_missing_value_for_known_archive_tickers(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [
            market_row(),
            market_row(
                ticker="FLC",
                value=float("nan"),
            ),
        ],
    )
    write_market_csv(
        adjusted_path,
        [
            market_row(),
            market_row(
                ticker="FLC",
                value=float("nan"),
            ),
        ],
    )

    market = load_canonical_market_history(
        raw_path,
        adjusted_path,
    )
    flc = market.loc[
        market["ticker"].eq("FLC")
    ].iloc[0]

    assert flc["open"] == 10.0
    assert flc["close"] == 10.5
    assert pd.isna(flc["value_traded"])
    assert market.attrs["raw_price_fallback_rows"] == 0
    assert market.attrs["missing_value_traded_rows"] == 1
    assert market.attrs[
        "missing_value_traded_tickers"
    ] == ["FLC"]


def test_loader_estimates_traded_value_only_for_missing_rows(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [
            market_row(),
            market_row(
                ticker="FLC",
                value=float("nan"),
            ),
        ],
    )
    write_market_csv(
        adjusted_path,
        [
            market_row(),
            market_row(
                ticker="FLC",
                value=float("nan"),
            ),
        ],
    )

    market = load_canonical_market_history(
        raw_path,
        adjusted_path,
    )
    aaa = market.loc[market["ticker"].eq("AAA")].iloc[0]
    flc = market.loc[market["ticker"].eq("FLC")].iloc[0]

    # AAA has a real vendor value; the estimate must stay unused.
    assert pd.isna(aaa["value_traded_estimated"])
    assert aaa["value_traded_is_estimated"] == False  # noqa: E712

    # FLC has no vendor value; typical price (H+L+C)/3 * volume fills in.
    expected_typical_price = (11.0 + 9.0 + 10.5) / 3.0
    assert flc["value_traded_estimated"] == pytest.approx(
        expected_typical_price * 100
    )
    assert flc["value_traded_is_estimated"] == True  # noqa: E712


def test_loader_still_rejects_missing_value_for_non_archive_ticker_with_raw(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [
            market_row(),
            market_row(
                ticker="BBB",
                value=float("nan"),
            ),
        ],
    )
    write_market_csv(
        adjusted_path,
        [
            market_row(),
            market_row(
                ticker="BBB",
                value=float("nan"),
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="despite available raw market data",
    ):
        load_canonical_market_history(
            raw_path,
            adjusted_path,
        )


def test_loader_rejects_raw_rows_without_adjusted_match(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [
            market_row(),
            market_row(ticker="BBB"),
        ],
    )
    write_market_csv(
        adjusted_path,
        [market_row()],
    )

    with pytest.raises(
        ValueError,
        match="lack adjusted counterparts",
    ):
        load_canonical_market_history(
            raw_path,
            adjusted_path,
        )


def test_loader_rejects_volume_disagreement(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    write_market_csv(
        raw_path,
        [market_row(volume=100)],
    )
    write_market_csv(
        adjusted_path,
        [market_row(volume=101)],
    )

    with pytest.raises(
        ValueError,
        match="disagree on volume",
    ):
        load_canonical_market_history(
            raw_path,
            adjusted_path,
        )
