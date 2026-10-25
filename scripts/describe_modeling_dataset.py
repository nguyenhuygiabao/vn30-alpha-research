"""Describe the dataset the models are fitted on, for the report page.

Writes two small tables so the report page never carries a hand-entered row
count, date range or feature-group size:

- `reports/tables/dataset_description.csv`, one row: the span, row count and
  ticker count of the canonical market history and of the modeling dataset.
- `reports/tables/feature_groups.csv`: the model-facing feature columns
  partitioned into the six groups `reports/methodology.md` names, with the
  ablation that removes each group.

The four ablated groups are read from `src/ablation_tests.py` rather than
restated, so the partition cannot drift from what the ablation removes. One
column, `herding_volume_concentration_score`, is in both the herding and the
volume/liquidity removal lists; it is counted once, under herding. Foreign-flow
columns are identified by prefix, and every remaining column must be a return
or momentum column, or this raises.

It reads existing artefacts and fits nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablation_tests import (
    ABLATION_RESULTS_PATH,
    HERDING_FEATURES,
    PRICE_LIMIT_FEATURES,
    RISK_FEATURES,
    VOLUME_LIQUIDITY_FEATURES,
)
from src.baselines import load_baseline_dataset
from src.feature_pipeline import ADJUSTED_MARKET_PATH, RAW_MARKET_PATH
from src.market_data_schema import load_canonical_market_history
from src.modeling_utils import get_model_feature_columns
from src.walk_forward_split import TARGET_COLUMN

DATASET_OUTPUT_PATH: str = "reports/tables/dataset_description.csv"
FEATURE_GROUPS_OUTPUT_PATH: str = "reports/tables/feature_groups.csv"

FOREIGN_FLOW_PREFIX: str = "foreign_"
ESTIMATED_VALUE_PREFIX: str = "value_traded"
MOMENTUM_RETURN_MARKERS: tuple[str, ...] = ("return", "distance_from_", "shock")

# Display order, and the precedence used when a column is in more than one
# removal list: earlier groups claim it first.
ABLATED_GROUPS: tuple[tuple[str, str, str, list[str]], ...] = (
    ("risk", "Risk", "without_risk", RISK_FEATURES),
    ("herding", "Herding", "without_herding", HERDING_FEATURES),
    ("price_limit", "Price limit", "without_price_limit", PRICE_LIMIT_FEATURES),
    (
        "liquidity",
        "Liquidity",
        "without_volume_liquidity",
        VOLUME_LIQUIDITY_FEATURES,
    ),
)

DISPLAY_ORDER: tuple[str, ...] = (
    "momentum_return",
    "risk",
    "liquidity",
    "price_limit",
    "herding",
    "foreign_flow",
)


def partition_features(feature_columns: list[str]) -> pd.DataFrame:
    """Assign every model-facing column to exactly one of six groups."""
    assigned: dict[str, str] = {}

    for group, _, _, members in ABLATED_GROUPS:
        for column in members:
            if column not in feature_columns:
                raise ValueError(
                    f"Ablation column {column} is not a model feature"
                )
            assigned.setdefault(column, group)

    for column in feature_columns:
        if column in assigned:
            continue
        if column.startswith(FOREIGN_FLOW_PREFIX):
            assigned[column] = "foreign_flow"
        elif column.startswith(ESTIMATED_VALUE_PREFIX):
            assigned[column] = "liquidity"
        elif any(marker in column for marker in MOMENTUM_RETURN_MARKERS):
            assigned[column] = "momentum_return"
        else:
            raise ValueError(f"Feature column {column} belongs to no group")

    labels = {group: label for group, label, _, _ in ABLATED_GROUPS}
    labels["momentum_return"] = "Momentum and return"
    labels["foreign_flow"] = "Foreign flow"
    ablations = {group: name for group, _, name, _ in ABLATED_GROUPS}

    rows = []
    for group in DISPLAY_ORDER:
        members = [column for column in feature_columns if assigned[column] == group]
        rows.append(
            {
                "group": group,
                "display_name": labels[group],
                "feature_count": len(members),
                "ablation_name": ablations.get(group, ""),
                "features": ";".join(members),
            }
        )

    return pd.DataFrame(rows)


def verify_against_ablation(
    groups: pd.DataFrame,
    feature_columns: list[str],
    ablation: pd.DataFrame,
) -> None:
    """The partition must reproduce the published ablation feature counts."""
    counts = ablation.set_index("ablation_name")["feature_count"]

    if int(groups["feature_count"].sum()) != len(feature_columns):
        raise ValueError("Feature groups do not cover every feature exactly once")

    if len(feature_columns) != int(counts["all_features"]):
        raise ValueError(
            f"{len(feature_columns)} model features against "
            f"{int(counts['all_features'])} in the published ablation"
        )

    for _, _, name, members in ABLATED_GROUPS:
        remaining = len(feature_columns) - len(set(members))
        if remaining != int(counts[name]):
            raise ValueError(
                f"{name} leaves {remaining} features against "
                f"{int(counts[name])} in the published ablation"
            )


def describe_dataset() -> tuple[pd.DataFrame, list[str]]:
    """The dataset's span and size, and the model-facing feature columns."""
    market = load_canonical_market_history(RAW_MARKET_PATH, ADJUSTED_MARKET_PATH)
    modeling, historical, _ = load_baseline_dataset()

    return pd.DataFrame(
        [
            {
                "market_history_rows": len(market),
                "market_history_tickers": int(market["ticker"].nunique()),
                "market_history_first_date": pd.Timestamp(market["date"].min()).date(),
                "market_history_last_date": pd.Timestamp(market["date"].max()).date(),
                "modeling_rows": len(modeling),
                "modeling_tickers": int(modeling["ticker"].nunique()),
                "modeling_first_date": pd.Timestamp(modeling["date"].min()).date(),
                "modeling_last_date": pd.Timestamp(modeling["date"].max()).date(),
                "target_column": TARGET_COLUMN,
                "last_labeled_date": pd.Timestamp(historical["date"].max()).date(),
            }
        ]
    ), get_model_feature_columns(historical)


def main() -> None:
    description, feature_columns = describe_dataset()
    groups = partition_features(feature_columns)
    verify_against_ablation(
        groups=groups,
        feature_columns=feature_columns,
        ablation=pd.read_csv(ABLATION_RESULTS_PATH),
    )

    for path, frame in (
        (DATASET_OUTPUT_PATH, description),
        (FEATURE_GROUPS_OUTPUT_PATH, groups),
    ):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print("Output:", output)

    print(description.T.to_string(header=False))
    print(groups[["display_name", "feature_count", "ablation_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
