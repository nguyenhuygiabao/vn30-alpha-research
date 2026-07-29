from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features_basic import build_basic_features
from src.features_foreign_flow import (
    FOREIGN_FLOW_FEATURE_COLUMNS,
    add_foreign_flow_features,
    load_canonical_foreign_flow_csv,
)
from src.features_herding import add_herding_features
from src.features_liquidity import (
    add_liquidity_features,
    winsorize_series,
)
from src.features_momentum import add_momentum_features
from src.market_data_schema import (
    load_canonical_market_history,
)
from src.price_limit import add_estimated_price_limits
from src.universe_history import filter_to_universe
from src.universe_reconstruction import (
    reconstruct_constituent_history,
)

RAW_MARKET_PATH: str = (
    "data/raw/fiinquant/canonical/"
    "raw_market_history.csv"
)
ADJUSTED_MARKET_PATH: str = (
    "data/raw/fiinquant/canonical/"
    "adjusted_market_history.csv"
)
FOREIGN_FLOW_PATH: str = (
    "data/raw/fiinquant/canonical/"
    "foreign_flow_history.csv"
)
MEMBERSHIP_HISTORY_DIR: str = "config/vn30_history"
UNIVERSE_MODE: str = "strict_vn30"
COMBINED_FEATURES_PATH: str = "data/processed/features_combined.parquet"

REQUIRED_COMBINED_FEATURE_COLUMNS = (
    "date",
    "ticker",
    "adjusted_close",
    "simple_return_1d",
    "return_1d",
    "return_5d",
    "rolling_vol_20d",
    "distance_to_ceiling",
    "distance_to_floor",
    "volume_z_20",
    "amihud_illiquidity",
    "herding_index",
    "foreign_flow_available",
    *FOREIGN_FLOW_FEATURE_COLUMNS,
)


def load_verified_membership_history(
    history_dir: str | Path = MEMBERSHIP_HISTORY_DIR,
) -> pd.DataFrame:
    directory = Path(history_dir)
    inputs = {
        "initial_snapshot": "initial_snapshot.csv",
        "review_calendar": "review_calendar.csv",
        "membership_changes": "membership_changes.csv",
        "source_manifest": "source_manifest.csv",
    }
    loaded: dict[str, pd.DataFrame] = {}

    for name, filename in inputs.items():
        path = directory / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"VN30 history file not found: {path}"
            )

        loaded[name] = pd.read_csv(
            path,
            dtype="string",
            encoding="utf-8-sig",
        )

    reconstruction = reconstruct_constituent_history(
        initial_snapshot=loaded["initial_snapshot"],
        review_calendar=loaded["review_calendar"],
        membership_changes=loaded["membership_changes"],
        source_manifest=loaded["source_manifest"],
        expected_size=30,
    )

    return reconstruction.membership_history


def mask_unavailable_raw_price_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    masked = features.copy()
    raw_price_available = masked[
        ["open", "high", "low", "close"]
    ].notna().all(axis=1)

    numeric_columns = [
        "reference_price",
        "estimated_ceiling_price",
        "estimated_floor_price",
        "distance_to_ceiling",
        "distance_to_floor",
        "consecutive_ceiling_days",
        "consecutive_floor_days",
    ]
    boolean_columns = [
        "hit_ceiling_today",
        "hit_floor_today",
        "close_at_ceiling_today",
        "close_at_floor_today",
    ]

    for column in numeric_columns:
        if column in masked.columns:
            masked[column] = masked[column].where(
                raw_price_available
            )

    for column in boolean_columns:
        if column in masked.columns:
            masked[column] = (
                masked[column]
                .astype("boolean")
                .where(raw_price_available)
            )

    return masked


def recompute_cross_sectional_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    recomputed = features.copy()

    if "average_daily_value_20d" in recomputed:
        recomputed["traded_value_rank_20d"] = (
            recomputed.groupby(
                "date",
                sort=False,
            )["average_daily_value_20d"].rank(
                method="average",
                ascending=True,
                pct=True,
            )
        )

    if "amihud_illiquidity_raw" in recomputed:
        recomputed["amihud_illiquidity"] = (
            recomputed.groupby(
                "date",
                sort=False,
            )["amihud_illiquidity_raw"].transform(
                winsorize_series
            )
        )

    for side in ("net", "buy", "sell"):
        source = (
            f"foreign_{side}_value_share_1d"
        )

        if source not in recomputed:
            continue

        recomputed[
            f"foreign_{side}_value_rank_pct"
        ] = recomputed.groupby(
            "date",
            sort=False,
        )[source].rank(
            method="average",
            pct=True,
        )

    return recomputed


def build_combined_features(
    data: pd.DataFrame,
    foreign_flow: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    universe_mode: str = UNIVERSE_MODE,
) -> pd.DataFrame:
    features = build_basic_features(data)

    features = add_momentum_features(features)

    features = add_estimated_price_limits(features)

    features = mask_unavailable_raw_price_features(
        features
    )

    features = add_liquidity_features(features)

    if foreign_flow is not None:
        features = add_foreign_flow_features(
            features,
            foreign_flow,
        )

    if membership is not None:
        features = filter_to_universe(
            features,
            membership,
            mode=universe_mode,
        )
        features = recompute_cross_sectional_features(
            features
        )

    features = add_herding_features(features)

    features = normalize_numeric_feature_dtypes(features)

    return features.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(drop=True)


def normalize_numeric_feature_dtypes(
    features: pd.DataFrame,
) -> pd.DataFrame:
    normalized = features.copy()

    excluded_columns = {
        "date",
        "ticker",
    }

    for column in normalized.columns:
        if column in excluded_columns:
            continue

        if normalized[column].dtype == "object":
            try:
                normalized[column] = pd.to_numeric(normalized[column])
            except (TypeError, ValueError):
                pass

    return normalized


def validate_combined_features(
    features: pd.DataFrame,
) -> dict[str, object]:
    if features.empty:
        raise ValueError("Combined feature output is empty")

    missing_columns = sorted(
        set(REQUIRED_COMBINED_FEATURE_COLUMNS).difference(
            features.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Combined feature output is missing required "
            f"columns: {missing_columns}"
        )

    duplicate_keys = int(
        features.duplicated(["date", "ticker"]).sum()
    )

    if duplicate_keys:
        raise ValueError(
            "Combined feature output contains "
            f"{duplicate_keys} duplicate ticker-date rows"
        )

    numeric = features.select_dtypes(include="number")
    infinite_columns: dict[str, int] = {}

    for column in numeric.columns:
        values = numeric[column].to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
        count = int(np.isinf(values).sum())

        if count:
            infinite_columns[column] = count

    if infinite_columns:
        raise ValueError(
            "Combined feature output contains infinite "
            f"values: {infinite_columns}"
        )

    return {
        "duplicate_keys": duplicate_keys,
        "missing_columns": missing_columns,
        "infinite_columns": infinite_columns,
    }


def save_combined_features(
    features: pd.DataFrame,
    output_path: str = COMBINED_FEATURES_PATH,
    *,
    validate: bool = True,
) -> str:
    if validate:
        validate_combined_features(features)

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        path,
        index=False,
    )

    return str(path)


def main() -> None:
    raw_market_path = Path(RAW_MARKET_PATH)
    adjusted_market_path = Path(
        ADJUSTED_MARKET_PATH
    )
    foreign_flow_path = Path(
        FOREIGN_FLOW_PATH
    )
    membership_history_dir = Path(
        MEMBERSHIP_HISTORY_DIR
    )

    data = load_canonical_market_history(
        raw_market_path,
        adjusted_market_path,
    )
    foreign_flow = load_canonical_foreign_flow_csv(
        foreign_flow_path
    )
    membership = load_verified_membership_history(
        membership_history_dir
    )

    market_keys = pd.MultiIndex.from_frame(
        data[["date", "ticker"]]
    )
    flow_keys = pd.MultiIndex.from_frame(
        foreign_flow[["date", "ticker"]]
    )
    market_without_flow = market_keys.difference(
        flow_keys
    )
    flow_without_market = flow_keys.difference(
        market_keys
    )

    features = build_combined_features(
        data,
        foreign_flow=foreign_flow,
        membership=membership,
        universe_mode=UNIVERSE_MODE,
    )

    feature_audit = validate_combined_features(
        features
    )
    output_path = save_combined_features(
        features,
        validate=False,
    )

    print("Combined feature generation completed.")
    print("Input rows:", len(data))
    print("Output rows:", len(features))
    print("Output columns:", len(features.columns))
    print(
        "Duplicate ticker-date keys:",
        feature_audit["duplicate_keys"],
    )
    print(
        "Missing required columns:",
        feature_audit["missing_columns"],
    )
    print(
        "Infinite numeric columns:",
        feature_audit["infinite_columns"],
    )
    print("Raw-market source:", raw_market_path)
    print(
        "Adjusted-market source:",
        adjusted_market_path,
    )
    print(
        "Foreign-flow source:",
        foreign_flow_path,
    )
    print(
        "Membership source:",
        membership_history_dir,
    )
    print("Universe mode:", UNIVERSE_MODE)
    print(
        "Raw-price fallback rows:",
        data.attrs["raw_price_fallback_rows"],
    )
    print(
        "Raw-price fallback tickers:",
        data.attrs["raw_price_fallback_tickers"],
    )
    print(
        "Missing traded-value rows:",
        data.attrs["missing_value_traded_rows"],
    )
    print(
        "Missing traded-value tickers:",
        data.attrs["missing_value_traded_tickers"],
    )
    print(
        "Market keys without foreign flow:",
        len(market_without_flow),
    )
    print(
        "Foreign-flow keys without market data:",
        len(flow_without_market),
    )
    print(
        "Foreign-flow available rows:",
        int(features["foreign_flow_available"].sum()),
    )
    print(
        "Rows excluded after lag construction:",
        len(data) - len(features),
    )
    print("Output path:", output_path)


if __name__ == "__main__":
    main()
