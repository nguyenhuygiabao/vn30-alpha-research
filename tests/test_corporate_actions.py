from __future__ import annotations

import pandas as pd
import pytest

from src.corporate_actions import (
    apply_adjustment_factors,
    build_adjustment_audit,
    validate_adjustment_factors_frame,
)


def market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2026-07-20",
                "2026-07-21",
            ],
            "ticker": [
                "hpg",
                "hpg",
            ],
            "open": [100.0, 52.0],
            "high": [102.0, 54.0],
            "low": [98.0, 50.0],
            "close": [100.0, 52.0],
            "adjusted_close": [100.0, 52.0],
            "volume": [1_000, 2_000],
            "value_traded": [
                100_000,
                104_000,
            ],
        }
    )


def factor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2026-07-20",
                "2026-07-21",
            ],
            "ticker": [
                "HPG",
                "HPG",
            ],
            "price_adjustment_factor": [
                0.5,
                1.0,
            ],
            "source": [
                "fixture",
                "fixture",
            ],
        }
    )


def test_factor_validation_normalizes_ticker() -> None:
    factors = factor_frame()
    factors.loc[0, "ticker"] = " hpg "

    validated = validate_adjustment_factors_frame(
        factors
    )

    assert validated.loc[0, "ticker"] == "HPG"


def test_duplicate_factor_keys_are_rejected() -> None:
    factors = factor_frame()

    factors = pd.concat(
        [factors, factors.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate ticker-date",
    ):
        validate_adjustment_factors_frame(
            factors
        )


def test_non_positive_factor_is_rejected() -> None:
    factors = factor_frame()
    factors.loc[0, "price_adjustment_factor"] = 0

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        validate_adjustment_factors_frame(
            factors
        )


def test_adjustment_preserves_raw_prices() -> None:
    adjusted = apply_adjustment_factors(
        market_frame(),
        factor_frame(),
    )

    first = adjusted.iloc[0]

    assert first["close"] == 100.0
    assert first["adjusted_close"] == 50.0
    assert first["adjusted_open"] == 50.0
    assert first["adjusted_high"] == 51.0
    assert first["adjusted_low"] == 49.0


def test_missing_factor_coverage_is_rejected() -> None:
    incomplete = factor_frame().iloc[[0]]

    with pytest.raises(
        ValueError,
        match="do not cover every market row",
    ):
        apply_adjustment_factors(
            market_frame(),
            incomplete,
        )


def test_missing_factor_can_default_to_one() -> None:
    incomplete = factor_frame().iloc[[0]]

    adjusted = apply_adjustment_factors(
        market_frame(),
        incomplete,
        require_full_coverage=False,
    )

    second = adjusted.iloc[1]

    assert second["price_adjustment_factor"] == 1.0
    assert second["adjusted_close"] == 52.0


def test_volume_adjustment_is_explicit() -> None:
    factors = factor_frame()
    factors["volume_adjustment_factor"] = [
        2.0,
        1.0,
    ]

    adjusted = apply_adjustment_factors(
        market_frame(),
        factors,
    )

    assert adjusted.loc[0, "volume"] == 1_000
    assert adjusted.loc[0, "adjusted_volume"] == 2_000


def test_audit_detects_removed_raw_jump() -> None:
    adjusted = apply_adjustment_factors(
        market_frame(),
        factor_frame(),
    )

    audit = build_adjustment_audit(adjusted)

    second = audit.iloc[1]

    assert bool(second["raw_jump"])
    assert not bool(second["adjusted_jump"])
    assert bool(
        second["corporate_action_jump_removed"]
    )
