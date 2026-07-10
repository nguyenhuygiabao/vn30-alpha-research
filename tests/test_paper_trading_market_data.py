from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.paper_trading.market_data import validate_completed_market_data


TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
GENERATED_AT = datetime(2026, 7, 10, 16, 0, tzinfo=TIMEZONE)


def market_data() -> pd.DataFrame:
    rows = []

    for ticker, close in (("FPT", 100), ("VNM", 60), ("VCB", 70)):
        rows.append(
            {
                "date": "2026-07-09",
                "ticker": ticker,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "adjusted_close": close,
                "volume": 1000000,
                "value_traded": close * 1000000,
            }
        )
        rows.append(
            {
                "date": "2026-07-10",
                "ticker": ticker,
                "open": close,
                "high": close + 2,
                "low": close - 1,
                "close": close + 1,
                "adjusted_close": close + 1,
                "volume": 1100000,
                "value_traded": (close + 1) * 1100000,
            }
        )

    return pd.DataFrame(rows)


def validate(data: pd.DataFrame):
    return validate_completed_market_data(
        data=data,
        expected_tickers=("FPT", "VNM", "VCB"),
        generated_at=GENERATED_AT,
        timezone_name="Asia/Ho_Chi_Minh",
        data_update_cutoff="15:15",
        execution_submission_cutoff="08:45",
    )


def test_complete_latest_market_date_passes() -> None:
    result = validate(market_data())

    assert result.latest_row_count == 3
    assert result.timing.data_asof_date.isoformat() == "2026-07-10"
    assert result.timing.intended_execution_date.isoformat() == "2026-07-13"
    assert result.missing_tickers == ()
    assert len(result.warnings) == 1
    assert "No explicit market holidays" in result.warnings[0]


def test_missing_latest_ticker_is_rejected() -> None:
    data = market_data()
    data = data.loc[
        ~((data["date"] == "2026-07-10") & (data["ticker"] == "VCB"))
    ]

    with pytest.raises(ValueError, match="missing tickers.*VCB"):
        validate(data)


def test_duplicate_ticker_date_is_rejected() -> None:
    data = market_data()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate keys"):
        validate(data)


def test_nonpositive_latest_price_is_rejected() -> None:
    data = market_data()
    mask = (data["date"] == "2026-07-10") & (data["ticker"] == "FPT")
    data.loc[mask, "close"] = 0

    with pytest.raises(ValueError, match="nonpositive prices.*FPT"):
        validate(data)


def test_invalid_latest_ohlc_range_is_rejected() -> None:
    data = market_data()
    mask = (data["date"] == "2026-07-10") & (data["ticker"] == "FPT")
    data.loc[mask, "high"] = 50

    with pytest.raises(ValueError, match="invalid OHLC ranges.*FPT"):
        validate(data)


def test_zero_volume_is_visible_warning() -> None:
    data = market_data()
    mask = (data["date"] == "2026-07-10") & (data["ticker"] == "FPT")
    data.loc[mask, "volume"] = 0
    result = validate(data)

    assert any("volume for: FPT" in warning for warning in result.warnings)


def test_zero_traded_value_is_visible_warning() -> None:
    data = market_data()
    mask = (data["date"] == "2026-07-10") & (data["ticker"] == "FPT")
    data.loc[mask, "value_traded"] = 0
    result = validate(data)

    assert any("traded value for: FPT" in warning for warning in result.warnings)


def test_stale_latest_market_date_is_rejected() -> None:
    data = market_data().loc[lambda frame: frame["date"] == "2026-07-09"]

    with pytest.raises(ValueError, match="does not match.*2026-07-10"):
        validate(data)
