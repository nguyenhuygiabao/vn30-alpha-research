import builtins
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "download_fiinquant_history.py"
)

SPEC = spec_from_file_location(
    "download_fiinquant_history",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

downloader = module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def test_sdk_is_loaded_only_when_a_client_is_created(
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "FiinQuantX":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        builtins,
        "__import__",
        blocked_import,
    )

    with pytest.raises(
        RuntimeError,
        match="FiinQuantX is required",
    ):
        downloader.create_fiinquant_client(
            "user",
            "password",
        )


def test_vendor_cleaner_rejects_duplicate_normalized_columns():
    frame = pd.DataFrame(
        [[
            "HPG",
            "2026-07-01",
            10.0,
            10.0,
            10.0,
            10.0,
            100.0,
            1_000.0,
            10.0,
        ]],
        columns=[
            "ticker",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "value",
            " OPEN ",
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate normalized columns",
    ):
        downloader.clean_vendor_frame(
            frame,
            {
                "ticker",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "value",
            },
            "Raw FiinQuant history",
        )


def test_invalid_output_is_not_published(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_path = tmp_path / "raw_market_history.csv"
    adjusted_path = (
        tmp_path / "adjusted_market_history.csv"
    )
    foreign_path = (
        tmp_path / "foreign_flow_history.csv"
    )
    raw_path.write_text("existing\n", encoding="utf-8")

    monkeypatch.setattr(
        downloader,
        "RAW_PATH",
        raw_path,
    )
    monkeypatch.setattr(
        downloader,
        "ADJUSTED_PATH",
        adjusted_path,
    )
    monkeypatch.setattr(
        downloader,
        "FOREIGN_PATH",
        foreign_path,
    )

    market = pd.DataFrame(
        {
            "ticker": ["HPG"],
            "timestamp": ["2026-07-01"],
            "open": [27_000.0],
            "high": [28_000.0],
            "low": [26_000.0],
            "close": [27_500.0],
            "volume": [1_000.0],
            "value": [27_500_000.0],
        }
    )
    adjusted = market.copy()
    invalid_flow = pd.DataFrame(
        {
            "ticker": ["HPG"],
            "timestamp": ["2026-07-01"],
            "fb": [60_000.0],
            "fs": [50_000.0],
            "fn": [999.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="foreign_net_value is inconsistent",
    ):
        downloader.write_canonical_outputs(
            market,
            adjusted,
            invalid_flow,
        )

    assert raw_path.read_text(
        encoding="utf-8"
    ) == "existing\n"
    assert not adjusted_path.exists()
    assert not foreign_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_repair_adjusted_ohlc_from_raw_preserves_close():
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "timestamp": pd.to_datetime(
                ["2026-01-02", "2026-01-05"]
            ),
            "open": [100.0, 110.0],
            "high": [120.0, 125.0],
            "low": [90.0, 100.0],
            "close": [110.0, 120.0],
        }
    )

    adjusted = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-05",
                ]
            ),
            "open": [50.0, 60.0, 10.0],
            "high": [60.0, 50.0, 11.0],
            "low": [45.0, 55.0, 9.0],
            "close": [55.0, 60.0, 10.0],
        }
    )

    repaired, audit = (
        downloader.repair_adjusted_ohlc_from_raw(
            raw,
            adjusted,
        )
    )

    target = repaired.loc[
        repaired["ticker"].eq("AAA")
        & repaired["timestamp"].eq(
            pd.Timestamp("2026-01-05")
        )
    ].iloc[0]

    assert target["open"] == pytest.approx(55.0)
    assert target["high"] == pytest.approx(62.5)
    assert target["low"] == pytest.approx(50.0)
    assert target["close"] == pytest.approx(60.0)

    valid_row = repaired.loc[
        repaired["ticker"].eq("AAA")
        & repaired["timestamp"].eq(
            pd.Timestamp("2026-01-02")
        )
    ].iloc[0]

    assert valid_row["open"] == pytest.approx(50.0)
    assert valid_row["high"] == pytest.approx(60.0)
    assert valid_row["low"] == pytest.approx(45.0)
    assert valid_row["close"] == pytest.approx(55.0)

    adjusted_only = repaired.loc[
        repaired["ticker"].eq("BBB")
    ].iloc[0]

    assert adjusted_only["open"] == pytest.approx(10.0)
    assert adjusted_only["high"] == pytest.approx(11.0)
    assert adjusted_only["low"] == pytest.approx(9.0)
    assert adjusted_only["close"] == pytest.approx(10.0)

    assert audit["targeted_rows"] == 1
    assert audit["repaired_fields"] == [
        "open",
        "high",
        "low",
    ]
    assert audit["adjusted_close_changed"] is False
    assert audit["adjusted_only_rows_changed"] is False


def test_repair_requires_raw_counterpart():
    raw = pd.DataFrame(
        columns=[
            "ticker",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    adjusted = pd.DataFrame(
        {
            "ticker": ["CCC"],
            "timestamp": pd.to_datetime(
                ["2026-01-05"]
            ),
            "open": [12.0],
            "high": [10.0],
            "low": [11.0],
            "close": [12.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="raw counterpart",
    ):
        downloader.repair_adjusted_ohlc_from_raw(
            raw,
            adjusted,
        )


def exchange_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "BBB"],
            "effective_from": [
                "2020-01-01",
                "2020-01-01",
                "2020-02-10",
            ],
            "effective_to": [
                "",
                "2020-02-09",
                "",
            ],
            "exchange": ["HOSE", "UPCOM", "HOSE"],
        }
    )


def market_pair(
    ticker: str,
    dates: list[str],
    raw_close: list[float],
    adjusted_close: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted_values = (
        raw_close
        if adjusted_close is None
        else adjusted_close
    )
    raw = pd.DataFrame(
        {
            "ticker": [ticker] * len(dates),
            "timestamp": pd.to_datetime(dates),
            "open": raw_close,
            "high": raw_close,
            "low": raw_close,
            "close": raw_close,
            "volume": [100.0] * len(dates),
            "value": [
                close * 100.0
                for close in raw_close
            ],
        }
    )
    adjusted = pd.DataFrame(
        {
            "ticker": [ticker] * len(dates),
            "timestamp": pd.to_datetime(dates),
            "close": adjusted_values,
        }
    )
    return raw, adjusted


def test_raw_exchange_rules_accept_regular_hose_move():
    raw, adjusted = market_pair(
        "AAA",
        ["2020-01-02", "2020-01-03"],
        [10_000.0, 10_650.0],
    )

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        exchange_history(),
    )

    assert audit == []


def test_raw_exchange_rules_reject_limit_breach():
    raw, adjusted = market_pair(
        "AAA",
        ["2020-01-02", "2020-01-03"],
        [10_000.0, 11_000.0],
    )

    with pytest.raises(
        ValueError,
        match="exchange-rule anomalies",
    ):
        downloader.validate_raw_exchange_rules(
            raw,
            adjusted,
            exchange_history(),
        )


def test_raw_exchange_rules_reject_off_tick_price():
    raw, adjusted = market_pair(
        "AAA",
        ["2020-01-02", "2020-01-03"],
        [10_000.0, 10_625.0],
    )

    with pytest.raises(
        ValueError,
        match="exchange-rule anomalies",
    ):
        downloader.validate_raw_exchange_rules(
            raw,
            adjusted,
            exchange_history(),
        )


def test_raw_exchange_rules_report_unavailable_action_reference():
    raw, adjusted = market_pair(
        "AAA",
        ["2020-01-02", "2020-01-03"],
        [100_000.0, 50_000.0],
        [50_000.0, 50_000.0],
    )

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        exchange_history(),
    )

    assert len(audit) == 1
    assert audit[0]["ticker"] == "AAA"
    assert audit[0]["timestamp"] == "2020-01-03"
    assert (
        audit[0]["reference_source"]
        == "corporate_action_reference_unavailable"
    )
    assert (
        audit[0]["status"]
        == "corporate_action_reference_unavailable"
    )
    assert audit[0]["reference_price"] is None


def test_raw_exchange_rules_detect_exchange_transfer():
    raw, adjusted = market_pair(
        "BBB",
        ["2020-02-07", "2020-02-10"],
        [10_000.0, 11_900.0],
    )

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        exchange_history(),
    )

    assert len(audit) == 1
    assert audit[0]["ticker"] == "BBB"
    assert audit[0]["timestamp"] == "2020-02-10"
    assert audit[0]["exchange"] == "HOSE"
    assert (
        audit[0]["session_kind"]
        == "FIRST_TRADING_DAY"
    )
    assert audit[0]["reference_price"] is None
    assert (
        audit[0]["status"]
        == "special_session_reference_unavailable"
    )


def test_raw_exchange_rules_detect_25_session_resumption():
    dates = pd.bdate_range(
        "2020-01-01",
        periods=27,
    )
    filler_raw, filler_adjusted = market_pair(
        "AAA",
        dates.strftime("%Y-%m-%d").tolist(),
        [10_000.0] * len(dates),
    )
    resumed_raw, resumed_adjusted = market_pair(
        "BBB",
        [
            dates[0].strftime("%Y-%m-%d"),
            dates[-1].strftime("%Y-%m-%d"),
        ],
        [10_000.0, 13_900.0],
    )

    raw = pd.concat(
        [filler_raw, resumed_raw],
        ignore_index=True,
    )
    adjusted = pd.concat(
        [filler_adjusted, resumed_adjusted],
        ignore_index=True,
    )

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        exchange_history(),
    )

    assert any(
        row["ticker"] == "BBB"
        and row["session_kind"] == "RESUMPTION_25_PLUS"
        for row in audit
    )


def test_raw_exchange_rules_skip_unavailable_upcom_band():
    raw = pd.DataFrame(
        {
            "ticker": ["BBB", "BBB"],
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-01-03"]
            ),
            "open": [10_000.0, 8_500.0],
            "high": [12_000.0, 11_500.0],
            "low": [9_000.0, 8_500.0],
            "close": [12_000.0, 10_000.0],
            "volume": [100.0, 100.0],
            "value": [1_000_000.0, 1_000_000.0],
        }
    )
    adjusted = raw[
        ["ticker", "timestamp", "close"]
    ].copy()

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        exchange_history(),
    )

    assert len(audit) == 1
    assert audit[0]["ticker"] == "*"
    assert audit[0]["exchange"] == "UPCOM"
    assert audit[0]["status"] == "validation_scope"
    assert audit[0]["rows"] == 1
    assert (
        audit[0]["reference_source"]
        == "historical_eod_reference_unavailable"
    )


def test_raw_exchange_rules_allow_upcom_average_close():
    raw = pd.DataFrame(
        {
            "ticker": ["BBB", "BBB"],
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-01-03"]
            ),
            "open": [7_900.0, 7_900.0],
            "high": [8_000.0, 8_000.0],
            "low": [7_800.0, 7_800.0],
            "close": [7_900.0, 7_898.0],
            "volume": [100.0, 100.0],
            "value": [790_000.0, 789_800.0],
        }
    )
    adjusted = raw[
        ["ticker", "timestamp", "close"]
    ].copy()

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        exchange_history(),
    )

    assert len(audit) == 1
    assert audit[0]["ticker"] == "*"
    assert audit[0]["exchange"] == "UPCOM"
    assert audit[0]["status"] == "validation_scope"


def test_raw_exchange_rules_still_reject_upcom_off_tick_trade():
    raw = pd.DataFrame(
        {
            "ticker": ["BBB", "BBB"],
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-01-03"]
            ),
            "open": [7_900.0, 7_950.0],
            "high": [8_000.0, 8_000.0],
            "low": [7_800.0, 7_800.0],
            "close": [7_900.0, 7_898.0],
            "volume": [100.0, 100.0],
            "value": [790_000.0, 789_800.0],
        }
    )
    adjusted = raw[
        ["ticker", "timestamp", "close"]
    ].copy()

    with pytest.raises(
        ValueError,
        match="exchange-rule anomalies",
    ):
        downloader.validate_raw_exchange_rules(
            raw,
            adjusted,
            exchange_history(),
        )


def test_raw_exchange_rules_verify_gvr_transfer_reference():
    raw = pd.DataFrame(
        {
            "ticker": ["GVR", "GVR"],
            "timestamp": pd.to_datetime(
                ["2020-03-06", "2020-03-17"]
            ),
            "open": [12_700.0, 10_500.0],
            "high": [12_700.0, 11_500.0],
            "low": [12_300.0, 9_600.0],
            "close": [12_500.0, 11_200.0],
            "volume": [100.0, 100.0],
            "value": [1_250_000.0, 1_120_000.0],
        }
    )
    adjusted = raw[
        ["ticker", "timestamp", "close"]
    ].copy()
    history = pd.DataFrame(
        {
            "ticker": ["GVR", "GVR"],
            "effective_from": [
                "2018-03-21",
                "2020-03-17",
            ],
            "effective_to": [
                "2020-03-09",
                "",
            ],
            "exchange": ["UPCOM", "HOSE"],
        }
    )

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        history,
    )

    assert len(audit) == 1
    assert audit[0]["reference_price"] == 11_570.0
    assert audit[0]["expected_floor"] == 9_260.0
    assert audit[0]["expected_ceiling"] == 13_850.0
    assert audit[0]["status"] == "verified_special_session"


def test_raw_exchange_rules_verify_vhm_listing_reference():
    raw = pd.DataFrame(
        {
            "ticker": ["VHM", "VHM"],
            "timestamp": pd.to_datetime(
                ["2017-07-13", "2018-05-17"]
            ),
            "open": [30_000.0, 110_500.0],
            "high": [30_000.0, 110_500.0],
            "low": [30_000.0, 110_500.0],
            "close": [30_000.0, 110_500.0],
            "volume": [100.0, 100.0],
            "value": [3_000_000.0, 11_050_000.0],
        }
    )
    adjusted = raw[
        ["ticker", "timestamp", "close"]
    ].copy()
    history = pd.DataFrame(
        {
            "ticker": ["VHM"],
            "effective_from": ["2017-07-13"],
            "effective_to": [""],
            "exchange": ["HOSE"],
        }
    )

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        history,
    )

    assert len(audit) == 1
    assert audit[0]["reference_price"] == 92_100.0
    assert audit[0]["expected_floor"] == 73_700.0
    assert audit[0]["expected_ceiling"] == 110_500.0
    assert audit[0]["status"] == "verified_special_session"


def test_raw_exchange_rules_verify_kdc_treasury_bonus():
    raw = pd.DataFrame(
        {
            "ticker": ["KDC", "KDC"],
            "timestamp": pd.to_datetime(
                ["2021-11-29", "2021-11-30"]
            ),
            "open": [60_500.0, 60_000.0],
            "high": [62_000.0, 60_000.0],
            "low": [60_500.0, 51_100.0],
            "close": [62_000.0, 58_200.0],
            "volume": [100.0, 100.0],
            "value": [6_200_000.0, 5_820_000.0],
        }
    )
    adjusted = raw[
        ["ticker", "timestamp", "close"]
    ].copy()
    history = pd.DataFrame(
        {
            "ticker": ["KDC"],
            "effective_from": ["2005-12-12"],
            "effective_to": [""],
            "exchange": ["HOSE"],
        }
    )

    audit = downloader.validate_raw_exchange_rules(
        raw,
        adjusted,
        history,
    )

    assert len(audit) == 1
    assert (
        audit[0]["session_kind"]
        == "TREASURY_SHARE_DISTRIBUTION"
    )
    assert audit[0]["reference_price"] == 62_000.0
    assert audit[0]["expected_floor"] == 49_600.0
    assert audit[0]["expected_ceiling"] == 74_400.0
    assert audit[0]["status"] == "verified_special_session"
