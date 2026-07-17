from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.paper_trading.targets import TOLERANCE, build_constrained_target_weights


def predictions(
    groups: list[str], sectors: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = [f"T{index:02d}" for index in range(1, len(groups) + 1)]
    prediction_frame = pd.DataFrame(
        {
            "date": pd.Timestamp("2026-07-10"),
            "ticker": tickers,
            "model_name": "gradient_boosting",
            "horizon_days": 10,
            "score": [1.0 - index / 100 for index in range(len(groups))],
            "predicted_rank": range(1, len(groups) + 1),
        }
    )
    universe = pd.DataFrame(
        {
            "ticker": tickers,
            "issuer_group": groups,
            "sector": sectors or groups,
        }
    )
    return prediction_frame, universe


def build(groups: list[str], sectors: list[str] | None = None):
    prediction_frame, universe = predictions(groups, sectors)
    return build_constrained_target_weights(
        predictions=prediction_frame,
        universe=universe,
        target_holdings=7,
        target_invested_weight="0.97",
        max_single_name_weight="0.15",
        max_issuer_group_weight="0.25",
        max_sector_weight="1",
    )


def test_seven_distinct_groups_receive_full_capped_allocation() -> None:
    result = build([f"Group {index}" for index in range(10)])
    targets = result.target_weights

    assert len(targets) == 7
    assert abs(result.invested_weight - Decimal("0.97")) <= TOLERANCE
    assert max(targets["target_weight"]) <= Decimal("0.15")
    assert result.capacity_replacements == ()
    assert targets["predicted_rank"].tolist() == list(range(1, 8))


def test_concentrated_top_ranks_are_diversified_until_feasible() -> None:
    groups = [
        "Vingroup",
        "Vingroup",
        "Vingroup",
        "Vingroup",
        "Group A",
        "Group B",
        "Group C",
        "Group D",
        "Group E",
        "Group F",
    ]
    result = build(groups)
    targets = result.target_weights
    group_weights = targets.groupby("issuer_group")["target_weight"].sum()

    assert result.invested_weight == Decimal("0.97")
    assert set(targets.loc[targets["issuer_group"] == "Vingroup", "ticker"]) == {
        "T01",
        "T02",
    }
    assert result.capacity_replacements == (("T04", "T08"), ("T03", "T09"))
    assert group_weights["Vingroup"] == Decimal("0.25")
    assert max(group_weights) <= Decimal("0.25")


def test_impossible_issuer_concentration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot construct"):
        build(["One Group"] * 10)


def test_rank_mismatch_is_rejected() -> None:
    prediction_frame, universe = predictions([f"Group {index}" for index in range(10)])
    prediction_frame.loc[0, "predicted_rank"] = 2

    with pytest.raises(ValueError, match="ranks do not match"):
        build_constrained_target_weights(
            predictions=prediction_frame,
            universe=universe,
            target_holdings=7,
            target_invested_weight="0.97",
            max_single_name_weight="0.15",
            max_issuer_group_weight="0.25",
            max_sector_weight="1",
        )


def test_current_vingroup_family_contains_all_four_members() -> None:
    universe = pd.read_csv("config/vn30_universe.csv")
    vingroup_tickers = set(
        universe.loc[universe["issuer_group"] == "Vingroup", "ticker"]
    )

    assert vingroup_tickers == {"VHM", "VIC", "VPL", "VRE"}


def test_bank_heavy_top_ranks_are_replaced_until_sector_cap_is_feasible() -> None:
    groups = [f"Issuer {index}" for index in range(12)]
    sectors = ["Banks"] * 6 + [
        "Consumer Staples", "Energy", "Technology", "Materials", "Real Estate", "Industrials"
    ]
    prediction_frame, universe = predictions(groups, sectors)

    result = build_constrained_target_weights(
        predictions=prediction_frame,
        universe=universe,
        target_holdings=8,
        target_invested_weight="0.97",
        max_single_name_weight="0.15",
        max_issuer_group_weight="0.25",
        max_sector_weight="0.35",
    )
    sector_weights = result.target_weights.groupby("sector")["target_weight"].sum()

    assert abs(result.invested_weight - Decimal("0.97")) <= TOLERANCE
    assert abs(sector_weights["Banks"] - Decimal("0.35")) <= TOLERANCE
    assert max(sector_weights) <= Decimal("0.35") + TOLERANCE
    assert len(result.capacity_replacements) == 3


def test_current_universe_identifies_all_bank_members_for_risk_control() -> None:
    universe = pd.read_csv("config/vn30_universe.csv")

    assert set(universe.loc[universe["sector"] == "Banks", "ticker"]) == {
        "ACB", "BID", "CTG", "HDB", "LPB", "MBB", "SHB", "SSB", "STB",
        "TCB", "TPB", "VCB", "VIB", "VPB",
    }

def test_crossed_issuer_and_sector_caps_use_joint_feasible_allocation() -> None:
    prediction_frame, universe = predictions(
        ["Group 1", "Group 1", "Group 2"],
        ["Sector 1", "Sector 2", "Sector 1"],
    )

    result = build_constrained_target_weights(
        predictions=prediction_frame,
        universe=universe,
        target_holdings=3,
        target_invested_weight="0.90",
        max_single_name_weight="0.50",
        max_issuer_group_weight="0.50",
        max_sector_weight="0.50",
    )
    targets = result.target_weights

    assert abs(result.invested_weight - Decimal("0.90")) <= TOLERANCE
    assert targets.groupby("issuer_group")["target_weight"].sum().max() <= Decimal(
        "0.50"
    )
    assert targets.groupby("sector")["target_weight"].sum().max() <= Decimal(
        "0.50"
    )
