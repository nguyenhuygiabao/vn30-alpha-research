from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.paper_trading.data_update import (
    DailyUpdateResult,
    build_daily_update,
    merge_incremental_data,
    normalize_provider_ohlcv,
    restrict_universe_to_members,
    validate_overlap_consistency,
    write_staged_update,
)


TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
GENERATED_AT = datetime(2026, 7, 10, 16, 0, tzinfo=TIMEZONE)
TICKERS = ("FPT", "VCB", "VNM")


def ohlcv_row(day: str, ticker: str, close: float) -> dict[str, object]:
    return {
        "date": day,
        "ticker": ticker,
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "adjusted_close": close,
        "volume": 1000000,
        "value_traded": close * 1000000,
    }


def existing_data() -> pd.DataFrame:
    rows = []

    for ticker, base in (("FPT", 100), ("VCB", 70), ("VNM", 60)):
        rows.append(ohlcv_row("2026-07-08", ticker, base))
        rows.append(ohlcv_row("2026-07-09", ticker, base + 1))

    return pd.DataFrame(rows)


def provider_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}

    for ticker, base in (("FPT", 100), ("VCB", 70), ("VNM", 60)):
        data[ticker] = pd.DataFrame(
            [
                ohlcv_row("2026-07-08", ticker, base),
                ohlcv_row("2026-07-09", ticker, base + 1),
                ohlcv_row("2026-07-10", ticker, base + 2),
            ]
        ).drop(columns=["ticker", "adjusted_close", "value_traded"])

    return data


def universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(TICKERS),
            "provider_symbol": list(TICKERS),
            "issuer_group": list(TICKERS),
        }
    )


class FakeProvider:
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        failing_tickers: set[str] | None = None,
    ) -> None:
        self.data = data
        self.failing_tickers = failing_tickers or set()
        self.calls: list[tuple[str, date, date]] = []

    def history(
        self,
        ticker: str,
        provider_symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        self.calls.append((ticker, start_date, end_date))

        if ticker in self.failing_tickers:
            raise RuntimeError(f"provider failure for {ticker}")

        return self.data[ticker].copy()


def build(provider: FakeProvider) -> DailyUpdateResult:
    return build_daily_update(
        existing_data=existing_data(),
        universe=universe(),
        provider=provider,
        generated_at=GENERATED_AT,
        timezone_name="Asia/Ho_Chi_Minh",
        data_update_cutoff="15:15",
        execution_submission_cutoff="08:45",
        holiday_dates=[],
        overlap_calendar_days=14,
        request_interval_seconds=0,
        max_attempts=2,
        overlap_close_tolerance=0.05,
        suspicious_return_threshold=0.15,
        sleep_function=lambda _: None,
    )


def test_incremental_update_replaces_overlap_and_adds_completed_day() -> None:
    provider = FakeProvider(provider_data())
    result = build(provider)

    assert result.validation.timing.data_asof_date == date(2026, 7, 10)
    assert result.validation.latest_row_count == 3
    assert len(result.combined_data) == 9
    assert len(result.audit) == 3
    assert set(result.audit["status"]) == {"ok"}
    assert len(provider.calls) == 3


def test_provider_failure_blocks_entire_update() -> None:
    provider = FakeProvider(provider_data(), failing_tickers={"VCB"})

    with pytest.raises(ValueError, match="failed for tickers.*VCB"):
        build(provider)

    assert sum(call[0] == "VCB" for call in provider.calls) == 2


def test_overlap_price_mismatch_is_rejected() -> None:
    downloaded = provider_data()
    downloaded["FPT"].loc[
        downloaded["FPT"]["date"] == "2026-07-09",
        "close",
    ] = 1000
    provider = FakeProvider(downloaded)

    with pytest.raises(ValueError, match="differs from existing closes"):
        build(provider)


def test_missing_existing_overlap_key_is_rejected() -> None:
    downloaded = provider_data()
    downloaded["FPT"] = downloaded["FPT"].loc[
        downloaded["FPT"]["date"] != "2026-07-09"
    ]
    provider = FakeProvider(downloaded)

    with pytest.raises(ValueError, match="missing existing ticker-date keys"):
        build(provider)


def test_truncated_overlap_download_is_rejected() -> None:
    downloaded = provider_data()

    for ticker in downloaded:
        downloaded[ticker] = downloaded[ticker].loc[
            downloaded[ticker]["date"] >= "2026-07-09"
        ]

    provider = FakeProvider(downloaded)

    with pytest.raises(ValueError, match="missing existing ticker-date keys"):
        build(provider)


def test_future_provider_row_is_rejected() -> None:
    downloaded = provider_data()
    future = ohlcv_row("2026-07-13", "FPT", 103)
    future_frame = pd.DataFrame([future]).drop(
        columns=["ticker", "adjusted_close", "value_traded"]
    )
    downloaded["FPT"] = pd.concat(
        [downloaded["FPT"], future_frame],
        ignore_index=True,
    )
    provider = FakeProvider(downloaded)

    with pytest.raises(ValueError, match="rows after the completed market date"):
        build(provider)


def test_provider_time_column_is_normalized() -> None:
    raw = provider_data()["FPT"].rename(columns={"date": "time"})
    normalized = normalize_provider_ohlcv(raw, "fpt")

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
    assert set(normalized["ticker"]) == {"FPT"}


def test_merge_does_not_mutate_existing_dataframe() -> None:
    existing = existing_data()
    original = existing.copy(deep=True)
    downloaded = pd.concat(
        [
            normalize_provider_ohlcv(frame, ticker)
            for ticker, frame in provider_data().items()
        ],
        ignore_index=True,
    )
    merge_incremental_data(
        existing,
        downloaded,
        update_start_date=date(2026, 6, 25),
        update_end_date=date(2026, 7, 10),
    )

    pd.testing.assert_frame_equal(existing, original)


def test_dry_run_does_not_replace_existing_file(tmp_path) -> None:
    result = build(FakeProvider(provider_data()))
    output_path = tmp_path / "vn30_ohlcv.csv"
    audit_path = tmp_path / "audit.csv"
    output_path.write_text("original-content\n", encoding="utf-8")

    write_staged_update(
        result,
        output_path=output_path,
        audit_path=audit_path,
        dry_run=True,
    )

    assert output_path.read_text(encoding="utf-8") == "original-content\n"
    assert not audit_path.exists()


def test_validated_update_replaces_files_without_staging_residue(tmp_path) -> None:
    result = build(FakeProvider(provider_data()))
    output_path = tmp_path / "vn30_ohlcv.csv"
    audit_path = tmp_path / "audit.csv"
    existing_data().to_csv(output_path, index=False)

    write_staged_update(
        result,
        output_path=output_path,
        audit_path=audit_path,
    )

    updated = pd.read_csv(output_path)
    audit = pd.read_csv(audit_path)
    assert pd.to_datetime(updated["date"]).max().date() == date(2026, 7, 10)
    assert len(audit) == 3
    assert not Path(str(output_path) + ".staging").exists()
    assert not Path(str(audit_path) + ".staging").exists()


def test_membership_filter_drops_alumni_tickers() -> None:
    metadata = pd.DataFrame(
        {
            "ticker": ["FLC", "FPT", "ROS", "VCB"],
            "provider_symbol": ["FLC", "FPT", "ROS", "VCB"],
            "issuer_group": ["FLC Group", "FPT", "FLC Group", "VCB"],
        }
    )

    restricted = restrict_universe_to_members(metadata, ["VCB", "FPT"])

    assert restricted["ticker"].tolist() == ["FPT", "VCB"]


def test_membership_filter_rejects_an_unknown_member() -> None:
    metadata = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "provider_symbol": ["FPT"],
            "issuer_group": ["FPT"],
        }
    )

    with pytest.raises(ValueError, match="missing index members.*VCB"):
        restrict_universe_to_members(metadata, ["FPT", "VCB"])


def test_membership_filter_rejects_empty_membership() -> None:
    metadata = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "provider_symbol": ["FPT"],
            "issuer_group": ["FPT"],
        }
    )

    with pytest.raises(ValueError, match="membership cannot be empty"):
        restrict_universe_to_members(metadata, [])
