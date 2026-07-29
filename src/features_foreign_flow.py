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

# Second-generation foreign buy/sell features.
#
# The original FB_FS_FOREIGN_FLOW_FEATURES set is information-equivalent to
# the frozen net features -- buy_share_Nd - sell_share_Nd reproduces
# net_share_Nd to floating-point exactness -- yet it produced no measured
# improvement. The diagnosed reason is representational, not informational:
# buy and sell shares correlate at ~0.88 (common-mode foreign turnover) and
# the net signal is only ~0.49 of their standard deviation, so a depth-2
# axis-aligned gradient-boosting tree cannot express the diagonal
# "buy minus sell" contrast it would need to recover the net signal.
#
# These features therefore supply contrasts the model cannot construct for
# itself, as single pre-computed axes, and are deliberately nonlinear in
# (buy, sell) so they are not spanned by the existing net/gross features:
#
# - foreign_flow_imbalance_{N}d: net divided by gross foreign value. The
#   existing frozen set carries this ratio at 1d only; these extend it to
#   the horizons where net flow actually showed value. Bounded [-1, 1] and
#   scale-free, so it separates flow *direction* from flow *magnitude*.
#   It also depends only on fb/fs, never on traded value, so it stays
#   defined for tickers whose vendor traded value is missing.
# - foreign_flow_return_corr_20d: rolling correlation between daily flow
#   imbalance and daily return. Distinguishes momentum-following foreign
#   flow from contrarian flow -- a property of flow *character* that no
#   level-based feature encodes.
# - foreign_buy_downday_share_20d: fraction of foreign buying that occurred
#   on down days, i.e. accumulation into weakness.
ENHANCED_FB_FS_FEATURES = (
    "foreign_flow_imbalance_3d",
    "foreign_flow_imbalance_5d",
    "foreign_flow_imbalance_10d",
    "foreign_flow_imbalance_20d",
    "foreign_flow_return_corr_20d",
    "foreign_buy_downday_share_20d",
)

# Every foreign-flow column the combined feature output must contain.
# ALL_FOREIGN_FLOW_LEARNING_FEATURES is deliberately left unchanged because
# it defines the already-published frozen_plus_fb_fs comparison.
FOREIGN_FLOW_FEATURE_COLUMNS = (
    ALL_FOREIGN_FLOW_LEARNING_FEATURES
    + ENHANCED_FB_FS_FEATURES
)

FOREIGN_FLOW_INTERACTION_WINDOW = 20

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


def _daily_adjusted_return(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Backward-looking one-day return from adjusted close, within each ticker.

    Derived here rather than read from an upstream feature column so this
    module does not depend on pipeline ordering. adjusted_close is a
    required market column, so this is always computable.
    """
    previous_close = data.groupby(
        "ticker",
        sort=False,
    )["adjusted_close"].shift(1)

    return data["adjusted_close"].div(
        previous_close.where(previous_close.gt(0))
    ) - 1.0


def _rolling_correlation(
    data: pd.DataFrame,
    left_column: str,
    right_column: str,
    window: int,
) -> pd.Series:
    """
    Backward-looking rolling Pearson correlation within each ticker.

    Built from rolling moments rather than Series.rolling().corr() so a
    zero-variance window resolves to missing instead of infinity, which the
    combined-feature validator rejects outright.
    """
    frame = data[
        [
            "ticker",
            left_column,
            right_column,
        ]
    ].copy()
    frame["_product"] = (
        frame[left_column] * frame[right_column]
    )
    frame["_left_squared"] = frame[left_column] ** 2
    frame["_right_squared"] = frame[right_column] ** 2

    moment_columns = [
        left_column,
        right_column,
        "_product",
        "_left_squared",
        "_right_squared",
    ]
    means = (
        frame.groupby(
            "ticker",
            sort=False,
        )[moment_columns]
        .rolling(
            window,
            min_periods=window,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(data.index)
    )

    covariance = (
        means["_product"]
        - means[left_column] * means[right_column]
    )
    # Clip variances that floating-point cancellation drove slightly
    # negative; a negative variance would produce a NaN square root that
    # hides a genuinely zero-variance window.
    left_variance = (
        means["_left_squared"] - means[left_column] ** 2
    ).clip(lower=0.0)
    right_variance = (
        means["_right_squared"] - means[right_column] ** 2
    ).clip(lower=0.0)

    denominator = np.sqrt(left_variance * right_variance)
    correlation = covariance.div(
        denominator.where(denominator.gt(0))
    )

    return correlation.clip(
        lower=-1.0,
        upper=1.0,
    )


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

        # Direction of foreign flow normalized by its own gross size, so
        # the contrast survives independently of how much the stock traded.
        # Follows the 1d convention: a window with no foreign activity at
        # all is balanced (0.0), not missing.
        buy_sum = rolling_sums["foreign_buy_value"]
        sell_sum = rolling_sums["foreign_sell_value"]
        gross_sum = buy_sum + sell_sum
        combined[
            f"foreign_flow_imbalance_{window}d"
        ] = (
            (buy_sum - sell_sum)
            .div(gross_sum.where(gross_sum.gt(0)))
            .mask(
                buy_sum.eq(0) & sell_sum.eq(0),
                0.0,
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

    interaction_window = FOREIGN_FLOW_INTERACTION_WINDOW
    daily_return = _daily_adjusted_return(combined)

    # Is foreign flow in this name momentum-following or contrarian? This
    # describes the character of foreign participation, which no level or
    # magnitude feature captures.
    combined[
        f"foreign_flow_return_corr_{interaction_window}d"
    ] = _rolling_correlation(
        combined.assign(_daily_return=daily_return),
        "foreign_flow_imbalance_1d",
        "_daily_return",
        interaction_window,
    )

    # Share of foreign buying that landed on down days: accumulation into
    # weakness. A missing return leaves the day's indicator missing rather
    # than silently counting it as an up day.
    down_day = pd.Series(
        np.where(
            daily_return.isna(),
            np.nan,
            daily_return.lt(0).astype(float),
        ),
        index=combined.index,
    )
    combined["_foreign_buy_on_down_day"] = (
        combined["foreign_buy_value"] * down_day
    )

    interaction_sums = _rolling_sums(
        combined,
        [
            "_foreign_buy_on_down_day",
            "foreign_buy_value",
        ],
        interaction_window,
    )
    buy_value_total = interaction_sums["foreign_buy_value"]
    combined[
        f"foreign_buy_downday_share_{interaction_window}d"
    ] = interaction_sums[
        "_foreign_buy_on_down_day"
    ].div(
        buy_value_total.where(buy_value_total.gt(0))
    )

    combined = combined.drop(
        columns=[
            "_foreign_net_buy_day",
            "_foreign_buy_on_down_day",
            "foreign_buy_value_share_1d",
            "foreign_sell_value_share_1d",
        ]
    )

    return combined.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)
