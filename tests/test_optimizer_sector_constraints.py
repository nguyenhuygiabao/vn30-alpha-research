from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer import (
    MAX_SECTOR_WEIGHT,
    apply_sector_cap,
    build_target_weights_for_date,
)


def ranked_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2026-07-10"),
            "ticker": ["ACB", "MBB", "TCB", "FPT", "GAS", "VNM"],
            "issuer_group": [
                "ACB",
                "MB",
                "Techcombank",
                "FPT",
                "PV GAS",
                "Vinamilk",
            ],
            "sector": ["Banks", "Banks", "Banks", "Technology", "Energy", "Consumer Staples"],
            "predicted_return": [0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        }
    )


def test_sector_aware_targets_cap_bank_cluster() -> None:
    targets = build_target_weights_for_date(
        ranked_predictions(),
        top_n=5,
        max_weight=0.20,
        max_sector_weight=MAX_SECTOR_WEIGHT,
    )
    sector_weights = targets.groupby("sector")["target_weight"].sum()

    assert sector_weights["Banks"] == pytest.approx(MAX_SECTOR_WEIGHT)
    assert sector_weights.max() <= MAX_SECTOR_WEIGHT
    assert targets["target_weight"].sum() == pytest.approx(0.75)


def test_sector_cap_requires_sector_metadata() -> None:
    with pytest.raises(ValueError, match="missing sector"):
        apply_sector_cap(pd.DataFrame({"ticker": ["ACB"], "target_weight": [0.2]}))


def test_sector_cap_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="max_sector_weight"):
        apply_sector_cap(
            pd.DataFrame(
                {"ticker": ["ACB"], "sector": ["Banks"], "target_weight": [0.2]}
            ),
            max_sector_weight=0,
        )
