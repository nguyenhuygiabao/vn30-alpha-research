from __future__ import annotations

import pandas as pd
import pytest

from src.market_data_schema import (
    build_execution_market_view,
    validate_corporate_actions_frame,
    validate_foreign_flow_frame,
    validate_market_data_frame,
)


def sample_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-07-21", "2026-07-22"],
            "ticker": ["hpg", "hpg"],
            "open": [27.0, 28.0],
            "high": [28.0, 29.0],
            "low": [26.5, 27.5],
            "close": [27.5, 28.5],
            "adjusted_close": [25.0, 25.9],
            "volume": [1_000_000, 1_200_000],
            "value_traded": [
                27_500_000_000,
                34_200_000_000,
            ],
        }
    )


def test_execution_view_uses_only_raw_prices() -> None:
    market = sample_market_data()

    execution = build_execution_market_view(market)

    assert execution.loc[0, "close"] == 27.5
    assert "adjusted_close" not in execution.columns


def test_market_data_normalizes_keys() -> None:
    validated = validate_market_data_frame(
        sample_market_data()
    )

    assert validated["ticker"].tolist() == [
        "HPG",
        "HPG",
    ]

    assert validated["date"].dtype.kind == "M"


def test_market_data_rejects_duplicate_keys() -> None:
    market = sample_market_data()

    market = pd.concat(
        [market, market.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate ticker-date",
    ):
        validate_market_data_frame(market)


def test_partial_adjusted_ohlc_is_rejected() -> None:
    market = sample_market_data()
    market["adjusted_open"] = [24.5, 25.5]

    with pytest.raises(
        ValueError,
        match="Adjusted OHLC",
    ):
        validate_market_data_frame(market)


def test_foreign_flow_derives_net_values() -> None:
    flow = pd.DataFrame(
        {
            "date": ["2026-07-22"],
            "ticker": ["fpt"],
            "foreign_buy_volume": [150],
            "foreign_sell_volume": [100],
            "foreign_buy_value": [15_000_000],
            "foreign_sell_value": [11_000_000],
        }
    )

    validated = validate_foreign_flow_frame(flow)

    assert validated.loc[0, "ticker"] == "FPT"
    assert validated.loc[0, "foreign_net_volume"] == 50
    assert validated.loc[0, "foreign_net_value"] == 4_000_000


def test_inconsistent_foreign_net_value_is_rejected() -> None:
    flow = pd.DataFrame(
        {
            "date": ["2026-07-22"],
            "ticker": ["FPT"],
            "foreign_buy_volume": [150],
            "foreign_sell_volume": [100],
            "foreign_buy_value": [15_000_000],
            "foreign_sell_value": [11_000_000],
            "foreign_net_value": [9_000_000],
        }
    )

    with pytest.raises(
        ValueError,
        match="foreign_net_value is inconsistent",
    ):
        validate_foreign_flow_frame(flow)


def test_valid_corporate_action_is_normalized() -> None:
    actions = pd.DataFrame(
        {
            "ticker": ["vnm"],
            "announcement_date": ["2026-06-01"],
            "effective_date": ["2026-06-20"],
            "action_type": ["CASH_DIVIDEND"],
            "cash_amount_vnd": [2_000],
            "share_ratio": [0],
        }
    )

    validated = validate_corporate_actions_frame(
        actions
    )

    assert validated.loc[0, "ticker"] == "VNM"
    assert (
        validated.loc[0, "action_type"]
        == "cash_dividend"
    )


def test_late_corporate_action_announcement_is_rejected() -> None:
    actions = pd.DataFrame(
        {
            "ticker": ["VNM"],
            "announcement_date": ["2026-06-21"],
            "effective_date": ["2026-06-20"],
            "action_type": ["cash_dividend"],
            "cash_amount_vnd": [2_000],
            "share_ratio": [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="announced after",
    ):
        validate_corporate_actions_frame(actions)


def test_value_only_foreign_flow_is_valid() -> None:
    flow = pd.DataFrame(
        {
            "date": ["2026-07-22"],
            "ticker": ["FPT"],
            "foreign_buy_value": [15_000_000],
            "foreign_sell_value": [11_000_000],
        }
    )

    validated = validate_foreign_flow_frame(flow)

    assert validated.loc[
        0,
        "foreign_net_value",
    ] == 4_000_000

    assert "foreign_net_volume" not in validated.columns
