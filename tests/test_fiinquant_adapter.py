from __future__ import annotations

import pandas as pd
import pytest

from src.fiinquant_adapter import (
    normalize_fiinquant_foreign_flow,
    normalize_fiinquant_market_data,
)


def raw_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["hpg", "hpg"],
            "timestamp": [
                "2026-07-20 00:00",
                "2026-07-21 00:00",
            ],
            "open": [100_000, 52_000],
            "high": [102_000, 54_000],
            "low": [98_000, 50_000],
            "close": [100_000, 52_000],
            "volume": [1_000, 2_000],
            "value": [100_000_000, 104_000_000],
        }
    )


def adjusted_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["HPG", "HPG"],
            "timestamp": [
                "2026-07-20 00:00",
                "2026-07-21 00:00",
            ],
            "open": [50_000, 52_000],
            "high": [51_000, 54_000],
            "low": [49_000, 50_000],
            "close": [50_000, 52_000],
        }
    )


def test_market_adapter_maps_price_units() -> None:
    normalized = normalize_fiinquant_market_data(
        raw_market(),
        adjusted_market(),
    )

    first = normalized.iloc[0]

    assert first["ticker"] == "HPG"
    assert first["close"] == 100.0
    assert first["adjusted_close"] == 50.0
    assert first["value_traded"] == 100_000_000


def test_market_adapter_preserves_raw_prices() -> None:
    normalized = normalize_fiinquant_market_data(
        raw_market(),
        adjusted_market(),
    )

    assert normalized.loc[0, "close"] == 100.0
    assert normalized.loc[0, "adjusted_close"] == 50.0


def test_market_adapter_rejects_unmatched_rows() -> None:
    incomplete = adjusted_market().iloc[[0]]

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        normalize_fiinquant_market_data(
            raw_market(),
            incomplete,
        )


def test_foreign_flow_adapter_maps_fields() -> None:
    flow = pd.DataFrame(
        {
            "ticker": ["fpt"],
            "timestamp": ["2026-07-22 00:00"],
            "fb": [15_000_000],
            "fs": [11_000_000],
            "fn": [4_000_000],
        }
    )

    normalized = normalize_fiinquant_foreign_flow(
        flow
    )

    assert normalized.loc[
        0,
        "foreign_buy_value",
    ] == 15_000_000

    assert normalized.loc[
        0,
        "foreign_net_value",
    ] == 4_000_000


def test_foreign_flow_adapter_checks_fn() -> None:
    flow = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "timestamp": ["2026-07-22 00:00"],
            "fb": [15_000_000],
            "fs": [11_000_000],
            "fn": [9_000_000],
        }
    )

    with pytest.raises(
        ValueError,
        match="foreign_net_value is inconsistent",
    ):
        normalize_fiinquant_foreign_flow(flow)
