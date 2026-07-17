from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from src.paper_trading.market_data import (
    CompletedMarketDataValidation,
    validate_completed_market_data,
)
from src.paper_trading.timing import expected_completed_trading_day


OUTPUT_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "value_traded",
)


class DailyMarketDataProvider(Protocol):
    def history(
        self,
        ticker: str,
        provider_symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class DailyUpdateResult:
    combined_data: pd.DataFrame
    audit: pd.DataFrame
    validation: CompletedMarketDataValidation
    update_start_date: date
    update_end_date: date
    suspicious_moves: pd.DataFrame


def normalize_provider_ohlcv(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        raise ValueError(f"No provider rows returned for {ticker}")

    normalized = data.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if "time" in normalized.columns and "date" not in normalized.columns:
        normalized = normalized.rename(columns={"time": "date"})

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(normalized.columns))

    if missing:
        raise ValueError(f"Missing provider columns for {ticker}: {missing}")

    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="raise",
    ).dt.normalize()
    normalized["ticker"] = ticker.strip().upper()

    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "adjusted_close" in normalized.columns:
        normalized["adjusted_close"] = pd.to_numeric(
            normalized["adjusted_close"],
            errors="coerce",
        )
    else:
        normalized["adjusted_close"] = normalized["close"]

    normalized["value_traded"] = normalized["close"] * normalized["volume"]
    normalized = normalized[list(OUTPUT_COLUMNS)].copy()

    if normalized.duplicated(["date", "ticker"]).any():
        raise ValueError(f"Provider returned duplicate ticker-date rows for {ticker}")

    if normalized[list(OUTPUT_COLUMNS[2:])].isna().any().any():
        raise ValueError(f"Provider returned nonnumeric OHLCV values for {ticker}")

    return normalized.sort_values("date").reset_index(drop=True)


def load_universe_metadata(path: str | Path) -> pd.DataFrame:
    universe = pd.read_csv(path)
    required = {"ticker", "vnstock_symbol", "issuer_group"}
    missing = sorted(required.difference(universe.columns))

    if missing:
        raise ValueError(f"Universe is missing columns: {missing}")

    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["vnstock_symbol"] = (
        universe["vnstock_symbol"].astype(str).str.strip().str.upper()
    )

    if universe["ticker"].duplicated().any():
        raise ValueError("Universe contains duplicate tickers")

    if universe["ticker"].eq("").any() or universe["vnstock_symbol"].eq("").any():
        raise ValueError("Universe contains empty ticker or provider symbol")

    return universe.sort_values("ticker").reset_index(drop=True)


def choose_update_window(
    existing_data: pd.DataFrame,
    completed_date: date,
    overlap_calendar_days: int,
) -> tuple[date, date]:
    if existing_data.empty:
        raise ValueError("Existing OHLCV data cannot be empty for incremental update")

    latest_existing = pd.to_datetime(existing_data["date"], errors="raise").max().date()

    if latest_existing > completed_date:
        raise ValueError(
            f"Existing data date {latest_existing} is later than completed date "
            f"{completed_date}"
        )

    start_date = latest_existing - timedelta(days=overlap_calendar_days)

    return start_date, completed_date


def download_update_rows(
    provider: DailyMarketDataProvider,
    universe: pd.DataFrame,
    start_date: date,
    end_date: date,
    request_interval_seconds: float,
    max_attempts: int,
    sleep_function: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for row in universe.itertuples(index=False):
        last_error: Exception | None = None
        normalized: pd.DataFrame | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                downloaded = provider.history(
                    ticker=row.ticker,
                    provider_symbol=row.vnstock_symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                normalized = normalize_provider_ohlcv(downloaded, row.ticker)
                last_error = None
                break
            except Exception as error:
                last_error = error

                if attempt < max_attempts:
                    sleep_function(request_interval_seconds)

        if normalized is None:
            audit_rows.append(
                {
                    "ticker": row.ticker,
                    "provider_symbol": row.vnstock_symbol,
                    "status": "error",
                    "rows": 0,
                    "start_date": "",
                    "end_date": "",
                    "attempts": max_attempts,
                    "error": repr(last_error),
                }
            )
        else:
            frames.append(normalized)
            audit_rows.append(
                {
                    "ticker": row.ticker,
                    "provider_symbol": row.vnstock_symbol,
                    "status": "ok",
                    "rows": len(normalized),
                    "start_date": normalized["date"].min().date().isoformat(),
                    "end_date": normalized["date"].max().date().isoformat(),
                    "attempts": attempt,
                    "error": "",
                }
            )

        sleep_function(request_interval_seconds)

    audit = pd.DataFrame(audit_rows)
    failed = audit.loc[audit["status"] != "ok", "ticker"].tolist()

    if failed:
        raise ValueError(f"Daily data download failed for tickers: {failed}")

    if not frames:
        raise ValueError("Daily data provider returned no usable ticker data")

    combined = pd.concat(frames, ignore_index=True)

    return combined, audit


def validate_overlap_consistency(
    existing_data: pd.DataFrame,
    downloaded_data: pd.DataFrame,
    close_tolerance: float,
    replacement_start_date: date,
) -> None:
    existing = existing_data[["date", "ticker", "close"]].copy()
    downloaded = downloaded_data[["date", "ticker", "close"]].copy()
    existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
    downloaded["date"] = pd.to_datetime(downloaded["date"]).dt.normalize()
    overlap = existing.merge(
        downloaded,
        on=["date", "ticker"],
        how="inner",
        suffixes=("_existing", "_downloaded"),
    )

    if overlap.empty:
        raise ValueError("Downloaded data has no overlap with the existing OHLCV file")

    overlap_start = pd.Timestamp(replacement_start_date)
    overlap_end = existing["date"].max()
    expected_keys = existing.loc[
        existing["date"].between(overlap_start, overlap_end),
        ["date", "ticker"],
    ].drop_duplicates()
    downloaded_keys = downloaded.loc[
        downloaded["date"].between(overlap_start, overlap_end),
        ["date", "ticker"],
    ].drop_duplicates()
    key_coverage = expected_keys.merge(
        downloaded_keys,
        on=["date", "ticker"],
        how="left",
        indicator=True,
    )
    missing_keys = key_coverage.loc[key_coverage["_merge"] != "both"]

    if not missing_keys.empty:
        examples = missing_keys[["date", "ticker"]].head(10).to_dict(
            orient="records"
        )
        raise ValueError(
            "Downloaded overlap is missing existing ticker-date keys. "
            f"Examples: {examples}"
        )

    denominator = overlap["close_existing"].abs()
    relative_difference = (
        overlap["close_downloaded"] - overlap["close_existing"]
    ).abs() / denominator
    invalid = overlap.loc[
        denominator.eq(0) | relative_difference.gt(close_tolerance),
        ["date", "ticker", "close_existing", "close_downloaded"],
    ]

    if not invalid.empty:
        examples = invalid.head(10).to_dict(orient="records")
        raise ValueError(
            "Downloaded overlap differs from existing closes beyond tolerance. "
            f"Examples: {examples}"
        )


def merge_incremental_data(
    existing_data: pd.DataFrame,
    downloaded_data: pd.DataFrame,
    update_start_date: date,
    update_end_date: date,
) -> pd.DataFrame:
    existing = existing_data.copy()
    downloaded = downloaded_data.copy()
    existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
    downloaded["date"] = pd.to_datetime(downloaded["date"]).dt.normalize()

    future_rows = downloaded.loc[downloaded["date"].dt.date > update_end_date]

    if not future_rows.empty:
        raise ValueError(
            "Provider returned rows after the completed market date: "
            f"{sorted(future_rows['date'].dt.date.unique().tolist())}"
        )

    replacement = downloaded.loc[
        downloaded["date"].dt.date >= update_start_date
    ].copy()
    preserved = existing.loc[existing["date"].dt.date < update_start_date].copy()
    combined = pd.concat([preserved, replacement], ignore_index=True)
    combined = combined[list(OUTPUT_COLUMNS)].sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if combined.duplicated(["date", "ticker"]).any():
        raise ValueError("Merged OHLCV data contains duplicate ticker-date rows")

    return combined


def find_suspicious_moves(
    data: pd.DataFrame,
    threshold: float,
    start_date: date | None = None,
) -> pd.DataFrame:
    working = data[["date", "ticker", "adjusted_close"]].copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    working = working.sort_values(["ticker", "date"])
    working["return_1d"] = working.groupby("ticker")["adjusted_close"].pct_change(
        fill_method=None
    )

    mask = working["return_1d"].abs() > threshold

    if start_date is not None:
        mask &= working["date"].dt.date >= start_date

    return working.loc[
        mask,
        ["date", "ticker", "adjusted_close", "return_1d"],
    ].reset_index(drop=True)


def build_daily_update(
    existing_data: pd.DataFrame,
    universe: pd.DataFrame,
    provider: DailyMarketDataProvider,
    generated_at: datetime,
    timezone_name: str,
    data_update_cutoff: str,
    execution_submission_cutoff: str,
    holiday_dates: list[date | str],
    overlap_calendar_days: int,
    request_interval_seconds: float,
    max_attempts: int,
    overlap_close_tolerance: float,
    suspicious_return_threshold: float,
    sleep_function: Callable[[float], None] = time.sleep,
) -> DailyUpdateResult:
    completed_date = expected_completed_trading_day(
        generated_at=generated_at,
        timezone_name=timezone_name,
        data_update_cutoff=data_update_cutoff,
        holiday_dates=holiday_dates,
    )
    start_date, end_date = choose_update_window(
        existing_data,
        completed_date=completed_date,
        overlap_calendar_days=overlap_calendar_days,
    )
    downloaded, audit = download_update_rows(
        provider=provider,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        request_interval_seconds=request_interval_seconds,
        max_attempts=max_attempts,
        sleep_function=sleep_function,
    )
    validate_overlap_consistency(
        existing_data=existing_data,
        downloaded_data=downloaded,
        close_tolerance=overlap_close_tolerance,
        replacement_start_date=start_date,
    )
    combined = merge_incremental_data(
        existing_data=existing_data,
        downloaded_data=downloaded,
        update_start_date=start_date,
        update_end_date=end_date,
    )
    validation = validate_completed_market_data(
        data=combined,
        expected_tickers=universe["ticker"].tolist(),
        generated_at=generated_at,
        timezone_name=timezone_name,
        data_update_cutoff=data_update_cutoff,
        execution_submission_cutoff=execution_submission_cutoff,
        holiday_dates=holiday_dates,
        enforce_execution_cutoff=False,
    )
    suspicious_moves = find_suspicious_moves(
        combined,
        threshold=suspicious_return_threshold,
        start_date=start_date,
    )

    return DailyUpdateResult(
        combined_data=combined,
        audit=audit,
        validation=validation,
        update_start_date=start_date,
        update_end_date=end_date,
        suspicious_moves=suspicious_moves,
    )


def write_staged_update(
    result: DailyUpdateResult,
    output_path: str | Path,
    audit_path: str | Path,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return

    destination = Path(output_path)
    audit_destination = Path(audit_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(destination.suffix + ".staging")
    audit_staging = audit_destination.with_suffix(
        audit_destination.suffix + ".staging"
    )

    try:
        result.combined_data.to_csv(staging, index=False, lineterminator="\n")
        staged = pd.read_csv(staging)

        if len(staged) != len(result.combined_data):
            raise ValueError("Staged OHLCV row count does not match validated data")

        if list(staged.columns) != list(OUTPUT_COLUMNS):
            raise ValueError("Staged OHLCV columns do not match the validated schema")

        if staged.duplicated(["date", "ticker"]).any():
            raise ValueError("Staged OHLCV data contains duplicate keys")

        staged_latest_date = pd.to_datetime(staged["date"]).max().date()

        if staged_latest_date != result.validation.timing.data_asof_date:
            raise ValueError("Staged OHLCV latest date changed during serialization")

        result.audit.to_csv(audit_staging, index=False, lineterminator="\n")
        staging.replace(destination)
        audit_staging.replace(audit_destination)
    finally:
        staging.unlink(missing_ok=True)
        audit_staging.unlink(missing_ok=True)
