from __future__ import annotations

from copy import deepcopy

import pytest

from src.paper_trading.config import (
    load_paper_trading_config,
    validate_paper_trading_config,
)


def test_default_paper_trading_config_is_valid() -> None:
    config = load_paper_trading_config()

    assert config["timing"]["signal_horizon_trading_days"] == 10
    assert config["timing"]["execution_delay_trading_days"] == 1
    assert config["settlement"]["cycle"] == "T+2"
    assert config["portfolio"]["max_single_name_weight"] == 0.15


def test_same_day_execution_is_rejected() -> None:
    config = deepcopy(load_paper_trading_config())
    config["timing"]["execution_delay_trading_days"] = 0

    with pytest.raises(ValueError, match="same-day execution"):
        validate_paper_trading_config(config)


def test_infeasible_single_name_cap_is_rejected() -> None:
    config = deepcopy(load_paper_trading_config())
    config["portfolio"]["target_holdings"] = 5

    with pytest.raises(ValueError, match="cannot reach invested weight"):
        validate_paper_trading_config(config)


def test_unsettled_cash_reuse_is_rejected() -> None:
    config = deepcopy(load_paper_trading_config())
    config["settlement"]["allow_unsettled_sale_proceeds_for_buys"] = True

    with pytest.raises(ValueError, match="unsettled sale proceeds"):
        validate_paper_trading_config(config)


def test_invalid_market_time_is_rejected() -> None:
    config = deepcopy(load_paper_trading_config())
    config["timing"]["execution_submission_cutoff_time"] = "8:75"

    with pytest.raises(ValueError, match="HH:MM"):
        validate_paper_trading_config(config)


def test_duplicate_market_holidays_are_rejected() -> None:
    config = deepcopy(load_paper_trading_config())
    config["market_calendar"]["holiday_dates"] = [
        "2026-09-02",
        "2026-09-02",
    ]

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        validate_paper_trading_config(config)
