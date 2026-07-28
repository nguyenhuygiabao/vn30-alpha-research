from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.market_data_schema import (
    validate_foreign_flow_frame,
    validate_market_data_frame,
)

FOREIGN_FLOW_WINDOWS = (3, 5, 10, 20)

FROZEN_FOREIGN_FLOW_FEATURES = (
    "foreign_net_value_share_1d",
    "foreign_gross_value_share_1d",
    "foreign_net_value_share_3d",
    "foreign_net_value_share_5d",
    "foreign_net_value_share_10d",
    "foreign_net_value_share_20d",
    "foreign_net_buy_persistence_5d",
    "foreign_net_value_z_60",
    "foreign_net_value_rank_pct",
)

FB_FS_FOREIGN_FLOW_FEATURES = (
    "foreign_flow_imbalance_1d",
    "foreign_buy_value_share_3d",
    "foreign_sell_value_share_3d",
    "foreign_buy_value_share_5d",
    "foreign_sell_value_share_5d",
    "foreign_buy_value_share_10d",
    "foreign_sell_value_share_10d",
    "foreign_buy_value_share_20d",
    "foreign_sell_value_share_20d",
    "foreign_buy_value_z_60",
    "foreign_sell_value_z_60",
    "foreign_buy_value_rank_pct",
    "foreign_sell_value_rank_pct",
)

ALL_FOREIGN_FLOW_LEARNING_FEATURES = (
    FROZEN_FOREIGN_FLOW_FEATURES
    + FB_FS_FOREIGN_FLOW_FEATURES
)

CANONICAL_FOREIGN_FLOW_COLUMNS = {
    "ticker",
    "timestamp",
    "fb",
    "fs",
    "fn",
}

CANONICAL_FOREIGN_FLOW_RENAME = {
    "timestamp": "date",
    "fb": "foreign_buy_value",
    "fs": "foreign_sell_value",
    "fn": "foreign_net_value",
}


def load_canonical_foreign_flow_csv(
    file_path: str | Path,
) -> pd.DataFrame:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical foreign-flow file not found: {path}"
        )

    canonical = pd.read_csv(path)
    canonical.columns = [
        str(column).strip().lower()
        for column in canonical.columns
    ]

    duplicate_columns = canonical.columns[
        canonical.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            "Canonical foreign-flow data contain duplicate "
            "normalized columns: "
            f"{sorted(set(duplicate_columns))}"
        )

    missing = sorted(
        CANONICAL_FOREIGN_FLOW_COLUMNS.difference(
            canonical.columns
        )
    )

    if missing:
        raise ValueError(
            "Canonical foreign-flow data are missing "
            f"columns: {missing}"
        )

    normalized = (
        canonical[
            sorted(CANONICAL_FOREIGN_FLOW_COLUMNS)
        ]
        .rename(columns=CANONICAL_FOREIGN_FLOW_RENAME)
        .copy()
    )

    return validate_foreign_flow_frame(normalized)


def _rolling_sums(
    data: pd.DataFrame,
    columns: list[str],
    window: int,
) -> pd.DataFrame:
    rolling = (
        data.groupby(
            "ticker",
            sort=False,
        )[columns]
        .rolling(
            window,
            min_periods=window,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    return rolling.reindex(data.index)


def _add_rolling_z_scores(
    data: pd.DataFrame,
    source_to_output: dict[str, str],
) -> None:
    source_columns = list(source_to_output)
    grouped = data.groupby(
        "ticker",
        sort=False,
    )[source_columns]
    rolling = grouped.rolling(
        60,
        min_periods=20,
    )
    rolling_mean = (
        rolling.mean()
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(data.index)
    )
    rolling_std = (
        rolling.std(ddof=0)
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(data.index)
    )

    for source_column, output_column in (
        source_to_output.items()
    ):
        data[output_column] = (
            data[source_column]
            - rolling_mean[source_column]
        ) / rolling_std[source_column].where(
            rolling_std[source_column].gt(0)
        )


def add_foreign_flow_features(
    market_data: pd.DataFrame,
    foreign_flow: pd.DataFrame,
) -> pd.DataFrame:
    market = validate_market_data_frame(
        market_data,
        allow_missing_raw_prices=True,
        allow_missing_value_traded=True,
        allow_raw_price_exceptions=True,
    )
    flow = validate_foreign_flow_frame(foreign_flow)

    flow_columns = [
        "date",
        "ticker",
        "foreign_buy_value",
        "foreign_sell_value",
        "foreign_net_value",
    ]

    volume_columns = {
        "foreign_buy_volume",
        "foreign_sell_volume",
        "foreign_net_volume",
    }
    has_foreign_volume = volume_columns.issubset(
        flow.columns
    )

    if has_foreign_volume:
        flow_columns.extend(sorted(volume_columns))

    combined = market.merge(
        flow[flow_columns],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    combined = combined.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    value_availability_columns = [
        "foreign_buy_value",
        "foreign_sell_value",
        "foreign_net_value",
    ]
    combined["foreign_flow_available"] = (
        combined[value_availability_columns]
        .notna()
        .all(axis=1)
        .astype("int8")
    )
    combined["foreign_gross_value"] = (
        combined["foreign_buy_value"]
        + combined["foreign_sell_value"]
    )

    active_gross_flow = combined["foreign_gross_value"].gt(0)
    combined["foreign_flow_imbalance_1d"] = (
        combined["foreign_net_value"].div(
            combined["foreign_gross_value"].where(
                active_gross_flow
            )
        )
    )
    combined["foreign_flow_imbalance_1d"] = (
        combined["foreign_flow_imbalance_1d"].mask(
            combined["foreign_buy_value"].eq(0)
            & combined["foreign_sell_value"].eq(0),
            0.0,
        )
    )

    traded_value_denominator = (
        combined["value_traded"]
        .where(combined["value_traded"].gt(0))
    )

    for side in ("buy", "sell", "net"):
        value_column = f"foreign_{side}_value"
        combined[
            f"foreign_{side}_value_share_1d"
        ] = (
            combined[value_column]
            / traded_value_denominator
        )

    combined["foreign_gross_value_share_1d"] = (
        combined["foreign_gross_value"]
        / traded_value_denominator
    )

    if has_foreign_volume:
        combined["foreign_gross_volume"] = (
            combined["foreign_buy_volume"]
            + combined["foreign_sell_volume"]
        )
        volume_denominator = (
            combined["volume"]
            .where(combined["volume"].gt(0))
        )
        combined[
            "foreign_net_volume_share_1d"
        ] = (
            combined["foreign_net_volume"]
            / volume_denominator
        )

    for window in FOREIGN_FLOW_WINDOWS:
        rolling_columns = [
            "value_traded",
            "foreign_buy_value",
            "foreign_sell_value",
            "foreign_net_value",
        ]

        if has_foreign_volume:
            rolling_columns.extend(
                [
                    "volume",
                    "foreign_net_volume",
                ]
            )

        rolling_sums = _rolling_sums(
            combined,
            rolling_columns,
            window,
        )
        traded_value_sum = rolling_sums[
            "value_traded"
        ]

        for side in ("buy", "sell", "net"):
            value_column = f"foreign_{side}_value"
            value_sum = rolling_sums[value_column]
            combined[
                f"{value_column}_{window}d"
            ] = value_sum
            combined[
                f"{value_column}_share_{window}d"
            ] = (
                value_sum
                / traded_value_sum.where(
                    traded_value_sum.gt(0)
                )
            )

        if has_foreign_volume:
            combined[
                f"foreign_net_volume_share_{window}d"
            ] = (
                rolling_sums["foreign_net_volume"]
                / rolling_sums["volume"].where(
                    rolling_sums["volume"].gt(0)
                )
            )

    net_buy_indicator = pd.Series(
        np.where(
            combined["foreign_net_value"].isna(),
            np.nan,
            combined[
                "foreign_net_value"
            ].gt(0).astype(float),
        ),
        index=combined.index,
    )
    combined["_foreign_net_buy_day"] = (
        net_buy_indicator
    )
    net_buy_rolling = _rolling_sums(
        combined,
        ["_foreign_net_buy_day"],
        5,
    )
    combined["foreign_net_buy_days_5d"] = (
        net_buy_rolling["_foreign_net_buy_day"]
    )
    combined[
        "foreign_net_buy_persistence_5d"
    ] = (
        combined["foreign_net_buy_days_5d"]
        / 5.0
    )

    ratio_to_z_score = {
        "foreign_net_value_share_1d":
            "foreign_net_value_z_60",
        "foreign_buy_value_share_1d":
            "foreign_buy_value_z_60",
        "foreign_sell_value_share_1d":
            "foreign_sell_value_z_60",
    }
    _add_rolling_z_scores(
        combined,
        ratio_to_z_score,
    )

    rank_sources = list(ratio_to_z_score)
    ranks = combined.groupby(
        "date",
        sort=False,
    )[rank_sources].rank(
        method="average",
        pct=True,
    )
    for side in ("net", "buy", "sell"):
        ratio_column = (
            f"foreign_{side}_value_share_1d"
        )
        combined[
            f"foreign_{side}_value_rank_pct"
        ] = ranks[ratio_column]

    combined = combined.drop(
        columns=[
            "_foreign_net_buy_day",
            "foreign_buy_value_share_1d",
            "foreign_sell_value_share_1d",
        ]
    )

    return combined.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)
