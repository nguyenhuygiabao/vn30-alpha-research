from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config/paper_trading_config.yaml")


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)

    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {section}")

    return value


def _require_keys(section: dict[str, Any], section_name: str, keys: set[str]) -> None:
    missing = sorted(keys.difference(section))

    if missing:
        raise ValueError(f"Missing {section_name} config keys: {missing}")


def validate_paper_trading_config(config: dict[str, Any]) -> None:
    if config.get("version") != 1:
        raise ValueError("paper-trading config version must be 1")

    account = _require_mapping(config, "account")
    timing = _require_mapping(config, "timing")
    market_calendar = _require_mapping(config, "market_calendar")
    portfolio = _require_mapping(config, "portfolio")
    execution = _require_mapping(config, "execution")
    settlement = _require_mapping(config, "settlement")
    model = _require_mapping(config, "model")
    output = _require_mapping(config, "output")

    _require_keys(account, "account", {"account_id", "currency", "initial_cash_vnd"})
    _require_keys(
        timing,
        "timing",
        {
            "timezone",
            "earliest_data_update_time",
            "require_completed_market_day",
            "signal_horizon_trading_days",
            "rebalance_frequency_trading_days",
            "execution_delay_trading_days",
            "execution_price_source",
            "execution_submission_cutoff_time",
        },
    )
    _require_keys(market_calendar, "market_calendar", {"holiday_dates"})
    _require_keys(
        portfolio,
        "portfolio",
        {
            "target_holdings",
            "target_invested_weight",
            "cash_buffer_weight",
            "max_single_name_weight",
            "max_issuer_group_weight",
            "max_daily_turnover",
        },
    )
    _require_keys(
        execution,
        "execution",
        {
            "round_lot_size",
            "allow_odd_lots",
            "min_trade_value_vnd",
            "adv_lookback_trading_days",
            "max_trade_adv_fraction",
            "commission_rate",
            "slippage_rate",
            "sell_tax_rate",
            "block_buy_at_ceiling",
            "block_sell_at_floor",
        },
    )
    _require_keys(
        settlement,
        "settlement",
        {
            "cycle",
            "lag_trading_days",
            "availability_time",
            "allow_unsettled_sale_proceeds_for_buys",
            "allow_unsettled_bought_shares_for_sales",
        },
    )
    _require_keys(
        model,
        "model",
        {"model_name", "predictions_path", "universe_path", "raw_ohlcv_path"},
    )
    _require_keys(output, "output", {"directory"})

    if account["initial_cash_vnd"] <= 0:
        raise ValueError("initial_cash_vnd must be positive")

    if timing["signal_horizon_trading_days"] <= 0:
        raise ValueError("signal_horizon_trading_days must be positive")

    if timing["rebalance_frequency_trading_days"] <= 0:
        raise ValueError("rebalance_frequency_trading_days must be positive")

    if timing["execution_delay_trading_days"] < 1:
        raise ValueError("execution_delay_trading_days must prevent same-day execution")

    if timing["execution_price_source"] != "next_open":
        raise ValueError("execution_price_source must be next_open in config version 1")

    for name in ("earliest_data_update_time", "execution_submission_cutoff_time"):
        try:
            datetime.strptime(str(timing[name]), "%H:%M")
        except ValueError as error:
            raise ValueError(f"{name} must use HH:MM format") from error

    holiday_dates = market_calendar["holiday_dates"]

    if not isinstance(holiday_dates, list):
        raise ValueError("holiday_dates must be a list")

    normalized_holidays = []

    for holiday in holiday_dates:
        try:
            normalized_holidays.append(date.fromisoformat(str(holiday)))
        except ValueError as error:
            raise ValueError(f"Invalid market holiday date: {holiday}") from error

    if len(normalized_holidays) != len(set(normalized_holidays)):
        raise ValueError("holiday_dates cannot contain duplicates")

    target_holdings = portfolio["target_holdings"]
    invested_weight = portfolio["target_invested_weight"]
    cash_buffer = portfolio["cash_buffer_weight"]
    max_single_name = portfolio["max_single_name_weight"]
    max_issuer_group = portfolio["max_issuer_group_weight"]

    if target_holdings <= 0:
        raise ValueError("target_holdings must be positive")

    for name in (
        "target_invested_weight",
        "cash_buffer_weight",
        "max_single_name_weight",
        "max_issuer_group_weight",
        "max_daily_turnover",
    ):
        if not 0 <= portfolio[name] <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    if abs(invested_weight + cash_buffer - 1.0) > 1e-9:
        raise ValueError("target_invested_weight and cash_buffer_weight must sum to 1")

    if target_holdings * max_single_name < invested_weight:
        raise ValueError("target holdings and single-name cap cannot reach invested weight")

    if max_issuer_group < max_single_name:
        raise ValueError("issuer-group cap cannot be below the single-name cap")

    if execution["round_lot_size"] <= 0:
        raise ValueError("round_lot_size must be positive")

    if execution["min_trade_value_vnd"] < 0:
        raise ValueError("min_trade_value_vnd cannot be negative")

    if execution["adv_lookback_trading_days"] <= 0:
        raise ValueError("adv_lookback_trading_days must be positive")

    if not 0 < execution["max_trade_adv_fraction"] <= 1:
        raise ValueError("max_trade_adv_fraction must be in (0, 1]")

    for name in ("commission_rate", "slippage_rate", "sell_tax_rate"):
        if execution[name] < 0:
            raise ValueError(f"{name} cannot be negative")

    if settlement["cycle"] != "T+2" or settlement["lag_trading_days"] != 2:
        raise ValueError("config version 1 requires a T+2 settlement cycle")

    if settlement["allow_unsettled_sale_proceeds_for_buys"]:
        raise ValueError("unsettled sale proceeds cannot be enabled in conservative mode")

    if settlement["allow_unsettled_bought_shares_for_sales"]:
        raise ValueError("unsettled bought shares cannot be enabled in conservative mode")


def load_paper_trading_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("paper-trading config must contain a mapping")

    validate_paper_trading_config(config)

    return config
