from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.paper_trading.calendar import DateLike
from src.paper_trading.timing import SignalTiming, resolve_signal_timing


LATEST_PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")


@dataclass(frozen=True)
class CompletedMarketDataValidation:
    timing: SignalTiming
    expected_tickers: tuple[str, ...]
    observed_tickers: tuple[str, ...]
    latest_row_count: int
    warnings: tuple[str, ...]

    @property
    def missing_tickers(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.expected_tickers) - set(self.observed_tickers)))


def load_universe_tickers(path: str | Path) -> tuple[str, ...]:
    universe = pd.read_csv(path)

    if "ticker" not in universe.columns:
        raise ValueError("Universe file must contain a ticker column")

    tickers = tuple(
        universe["ticker"].astype(str).str.strip().str.upper().tolist()
    )

    if not tickers:
        raise ValueError("Universe cannot be empty")

    if len(tickers) != len(set(tickers)):
        raise ValueError("Universe contains duplicate tickers")

    return tickers


def validate_completed_market_data(
    data: pd.DataFrame,
    expected_tickers: Iterable[str],
    generated_at: datetime,
    timezone_name: str,
    data_update_cutoff: str,
    execution_submission_cutoff: str,
    holiday_dates: Iterable[DateLike] = (),
) -> CompletedMarketDataValidation:
    if data.empty:
        raise ValueError("OHLCV data is empty")

    holidays = tuple(holiday_dates)

    required_columns = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "value_traded",
    }
    missing_columns = sorted(required_columns.difference(data.columns))

    if missing_columns:
        raise ValueError(f"OHLCV data is missing columns: {missing_columns}")

    working = data.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise").dt.normalize()
    working["ticker"] = working["ticker"].astype(str).str.strip().str.upper()
    duplicate_count = int(working.duplicated(["date", "ticker"]).sum())

    if duplicate_count:
        raise ValueError(f"OHLCV data contains {duplicate_count} duplicate keys")

    latest_date = working["date"].max().date()
    timing = resolve_signal_timing(
        data_asof_date=latest_date,
        generated_at=generated_at,
        timezone_name=timezone_name,
        data_update_cutoff=data_update_cutoff,
        execution_submission_cutoff=execution_submission_cutoff,
        holiday_dates=holidays,
    )
    latest = working.loc[working["date"].dt.date == latest_date].copy()
    expected = tuple(sorted({str(ticker).strip().upper() for ticker in expected_tickers}))
    observed = tuple(sorted(latest["ticker"].unique().tolist()))
    missing_tickers = sorted(set(expected).difference(observed))
    unexpected_tickers = sorted(set(observed).difference(expected))

    if missing_tickers:
        raise ValueError(f"Latest market date is missing tickers: {missing_tickers}")

    if unexpected_tickers:
        raise ValueError(
            f"Latest market date contains unexpected tickers: {unexpected_tickers}"
        )

    if len(latest) != len(expected):
        raise ValueError(
            f"Latest market date has {len(latest)} rows; expected {len(expected)}"
        )

    numeric_latest = latest[list(LATEST_PRICE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_latest.isna().any().any():
        raise ValueError("Latest market date contains missing or nonnumeric prices")

    invalid_price_rows = latest.loc[(numeric_latest <= 0).any(axis=1), "ticker"]

    if not invalid_price_rows.empty:
        raise ValueError(
            "Latest market date contains nonpositive prices for: "
            f"{sorted(invalid_price_rows.tolist())}"
        )

    invalid_ranges = latest.loc[
        (numeric_latest["high"] < numeric_latest["low"])
        | (numeric_latest["open"] < numeric_latest["low"])
        | (numeric_latest["open"] > numeric_latest["high"])
        | (numeric_latest["close"] < numeric_latest["low"])
        | (numeric_latest["close"] > numeric_latest["high"]),
        "ticker",
    ]

    if not invalid_ranges.empty:
        raise ValueError(
            "Latest market date contains invalid OHLC ranges for: "
            f"{sorted(invalid_ranges.tolist())}"
        )

    warnings: list[str] = []
    numeric_volume = pd.to_numeric(latest["volume"], errors="coerce")
    zero_volume_tickers = sorted(
        latest.loc[numeric_volume.isna() | (numeric_volume <= 0), "ticker"].tolist()
    )

    if zero_volume_tickers:
        warnings.append(
            "Nonpositive or missing latest volume for: "
            + ", ".join(zero_volume_tickers)
        )

    numeric_value_traded = pd.to_numeric(latest["value_traded"], errors="coerce")
    invalid_value_tickers = sorted(
        latest.loc[
            numeric_value_traded.isna() | (numeric_value_traded <= 0),
            "ticker",
        ].tolist()
    )

    if invalid_value_tickers:
        warnings.append(
            "Nonpositive or missing latest traded value for: "
            + ", ".join(invalid_value_tickers)
        )

    if not holidays:
        warnings.append(
            "No explicit market holidays are configured; future dates use weekdays only."
        )

    return CompletedMarketDataValidation(
        timing=timing,
        expected_tickers=expected,
        observed_tickers=observed,
        latest_row_count=len(latest),
        warnings=tuple(warnings),
    )


def load_and_validate_completed_market_data(
    data_path: str | Path,
    universe_path: str | Path,
    generated_at: datetime,
    timezone_name: str,
    data_update_cutoff: str,
    execution_submission_cutoff: str,
    holiday_dates: Iterable[DateLike] = (),
) -> CompletedMarketDataValidation:
    data = load_ohlcv_csv(str(data_path))
    tickers = load_universe_tickers(universe_path)

    return validate_completed_market_data(
        data=data,
        expected_tickers=tickers,
        generated_at=generated_at,
        timezone_name=timezone_name,
        data_update_cutoff=data_update_cutoff,
        execution_submission_cutoff=execution_submission_cutoff,
        holiday_dates=holiday_dates,
    )
