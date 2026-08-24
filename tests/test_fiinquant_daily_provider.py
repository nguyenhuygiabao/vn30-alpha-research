from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.paper_trading.data_update import normalize_provider_ohlcv
from src.paper_trading.fiinquant_provider import (
    ADJUSTED_FIELDS,
    RAW_FIELDS,
    FiinQuantDailyProvider,
    normalize_daily_frame,
)


TICKERS = ("FPT", "VCB", "VNM")
DAYS = ("2026-07-08", "2026-07-09", "2026-07-10")
START = date(2026, 7, 8)
END = date(2026, 7, 10)

# The adjusted close sits below the raw close, as it does for any ticker that
# has paid a dividend since the bar.
ADJUSTMENT = 0.98


def raw_frame(tickers: tuple[str, ...] = TICKERS) -> pd.DataFrame:
    rows = []

    for ticker, base in zip(tickers, (100_000.0, 70_000.0, 60_000.0), strict=False):
        for offset, day in enumerate(DAYS):
            close = base + offset * 1_000.0
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": day,
                    "open": close - 500.0,
                    "high": close + 500.0,
                    "low": close - 1_000.0,
                    "close": close,
                    "volume": 1_000_000,
                }
            )

    return pd.DataFrame(rows)


def adjusted_frame(tickers: tuple[str, ...] = TICKERS) -> pd.DataFrame:
    frame = raw_frame(tickers)[["ticker", "timestamp", "close"]].copy()
    frame["close"] = frame["close"] * ADJUSTMENT

    return frame


class FakeResponse:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def get_data(self) -> pd.DataFrame:
        return self.data


class FakeClient:
    def __init__(
        self,
        raw: pd.DataFrame | None = None,
        adjusted: pd.DataFrame | None = None,
    ) -> None:
        self.raw = raw_frame() if raw is None else raw
        self.adjusted = adjusted_frame() if adjusted is None else adjusted
        self.calls: list[dict[str, object]] = []

    def Fetch_Trading_Data(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        data = self.adjusted if kwargs["adjusted"] else self.raw

        return FakeResponse(data.copy())


def test_prices_are_converted_to_thousands_of_vnd() -> None:
    # The local file stores prices in thousands; FiinQuant reports whole VND.
    # Getting this wrong would silently scale the whole paper-trading book.
    normalized = normalize_daily_frame(raw_frame())
    fpt = normalized.loc[normalized["ticker"] == "FPT"].reset_index(drop=True)

    assert fpt.loc[0, "close"] == pytest.approx(100.0)
    assert fpt.loc[0, "open"] == pytest.approx(99.5)


def test_volume_is_not_rescaled() -> None:
    normalized = normalize_daily_frame(raw_frame())

    assert normalized.loc[0, "volume"] == 1_000_000


def test_timestamp_becomes_a_normalized_date() -> None:
    vendor = raw_frame()
    vendor["timestamp"] = vendor["timestamp"] + " 15:00:00"

    normalized = normalize_daily_frame(vendor)

    assert (normalized["date"] == normalized["date"].dt.normalize()).all()
    assert normalized["date"].min() == pd.Timestamp("2026-07-08")


def test_missing_vendor_columns_are_rejected() -> None:
    vendor = raw_frame().drop(columns=["volume"])

    with pytest.raises(ValueError, match="missing columns.*volume"):
        normalize_daily_frame(vendor)


def test_nonnumeric_vendor_values_are_rejected() -> None:
    vendor = raw_frame()
    vendor["close"] = vendor["close"].astype(object)
    vendor.loc[0, "close"] = "n/a"

    with pytest.raises(ValueError, match="nonnumeric"):
        normalize_daily_frame(vendor)


def test_duplicate_vendor_rows_are_rejected() -> None:
    vendor = pd.concat([raw_frame(), raw_frame().head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate ticker-date rows"):
        normalize_daily_frame(vendor)


def test_empty_vendor_response_is_rejected() -> None:
    with pytest.raises(ValueError, match="no daily rows"):
        normalize_daily_frame(pd.DataFrame())


def test_a_window_costs_one_raw_and_one_adjusted_request() -> None:
    client = FakeClient()
    provider = FiinQuantDailyProvider(client=client, symbols=TICKERS)

    for ticker in TICKERS:
        history = provider.history(
            ticker=ticker,
            provider_symbol=ticker,
            start_date=START,
            end_date=END,
        )

        assert len(history) == 3
        assert "ticker" not in history.columns

    assert len(client.calls) == 2

    raw_call, adjusted_call = client.calls
    assert raw_call["adjusted"] is False
    assert raw_call["fields"] == list(RAW_FIELDS)
    assert adjusted_call["adjusted"] is True
    assert adjusted_call["fields"] == list(ADJUSTED_FIELDS)

    for call in client.calls:
        assert call["tickers"] == sorted(TICKERS)
        assert call["by"] == "1d"
        assert call["from_date"] == "2026-07-08"
        assert call["to_date"] == "2026-07-10"


def test_raw_prices_drive_ohlc_and_adjusted_drives_adjusted_close() -> None:
    # The two series must not be mixed: execution reads close, returns read
    # adjusted_close, and collapsing them is the defect this replaced.
    client = FakeClient()
    provider = FiinQuantDailyProvider(client=client, symbols=TICKERS)

    history = provider.history(
        ticker="FPT",
        provider_symbol="FPT",
        start_date=START,
        end_date=END,
    )

    assert history.loc[0, "close"] == pytest.approx(100.0)
    assert history.loc[0, "adjusted_close"] == pytest.approx(100.0 * ADJUSTMENT)


def test_a_new_window_triggers_new_requests() -> None:
    client = FakeClient()
    provider = FiinQuantDailyProvider(client=client, symbols=TICKERS)

    provider.history(
        ticker="FPT",
        provider_symbol="FPT",
        start_date=START,
        end_date=END,
    )
    provider.history(
        ticker="FPT",
        provider_symbol="FPT",
        start_date=START,
        end_date=date(2026, 7, 11),
    )

    assert len(client.calls) == 4


def test_an_adjusted_series_with_holes_is_rejected() -> None:
    client = FakeClient(adjusted=adjusted_frame().head(4))
    provider = FiinQuantDailyProvider(client=client, symbols=TICKERS)

    with pytest.raises(ValueError, match="adjusted series does not cover"):
        provider.history(
            ticker="FPT",
            provider_symbol="FPT",
            start_date=START,
            end_date=END,
        )


def test_a_ticker_absent_from_the_response_raises() -> None:
    client = FakeClient(
        raw=raw_frame(tickers=("FPT", "VCB")),
        adjusted=adjusted_frame(tickers=("FPT", "VCB")),
    )
    provider = FiinQuantDailyProvider(client=client, symbols=TICKERS)

    with pytest.raises(ValueError, match="no rows for VNM"):
        provider.history(
            ticker="VNM",
            provider_symbol="VNM",
            start_date=START,
            end_date=END,
        )


def test_a_failed_request_is_not_cached() -> None:
    # download_update_rows retries each ticker, so a transient vendor failure
    # must not be remembered as a permanent one.
    class FailingOnceClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def Fetch_Trading_Data(self, **kwargs: object) -> FakeResponse:
            if not self.failed:
                self.failed = True
                raise RuntimeError("vendor timeout")

            return super().Fetch_Trading_Data(**kwargs)

    client = FailingOnceClient()
    provider = FiinQuantDailyProvider(client=client, symbols=TICKERS)

    with pytest.raises(RuntimeError, match="vendor timeout"):
        provider.history(
            ticker="FPT",
            provider_symbol="FPT",
            start_date=START,
            end_date=END,
        )

    history = provider.history(
        ticker="FPT",
        provider_symbol="FPT",
        start_date=START,
        end_date=END,
    )

    assert len(history) == 3


def test_empty_symbol_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty symbols"):
        FiinQuantDailyProvider(client=FakeClient(), symbols=[])


def test_provider_output_satisfies_the_daily_update_schema() -> None:
    client = FakeClient()
    provider = FiinQuantDailyProvider(client=client, symbols=TICKERS)
    history = provider.history(
        ticker="FPT",
        provider_symbol="FPT",
        start_date=START,
        end_date=END,
    )

    normalized = normalize_provider_ohlcv(history, "FPT")

    assert list(normalized.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "value_traded",
    ]
    # The provider supplies a genuine adjusted series, so normalize_provider_ohlcv
    # must keep it rather than falling back to the raw close.
    assert normalized.loc[0, "adjusted_close"] == pytest.approx(100.0 * ADJUSTMENT)
    assert normalized.loc[0, "value_traded"] == pytest.approx(100.0 * 1_000_000)
