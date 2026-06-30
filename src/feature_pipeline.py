from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.features_basic import build_basic_features
from src.features_herding import add_herding_features
from src.features_liquidity import add_liquidity_features
from src.features_momentum import add_momentum_features
from src.price_limit import add_estimated_price_limits


SOURCE_PATH: str = "sample_data/sample_ohlcv.csv"
COMBINED_FEATURES_PATH: str = "data/processed/features_combined.parquet"


def build_combined_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    features = build_basic_features(data)

    features = add_momentum_features(features)

    features = add_estimated_price_limits(features)

    features = add_liquidity_features(features)

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
            except ValueError:
                pass
            except TypeError:
                pass

    return normalized


def save_combined_features(
    features: pd.DataFrame,
    output_path: str = COMBINED_FEATURES_PATH,
) -> str:
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
    data = load_ohlcv_csv(SOURCE_PATH)

    features = build_combined_features(data)

    output_path = save_combined_features(features)

    duplicate_keys = features.duplicated(
        [
            "date",
            "ticker",
        ]
    ).sum()

    required_columns = [
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
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    print("Combined feature generation completed.")
    print("Input rows:", len(data))
    print("Output rows:", len(features))
    print("Output columns:", len(features.columns))
    print("Duplicate ticker-date keys:", duplicate_keys)
    print("Missing required columns:", missing_columns)
    print("Output path:", output_path)


if __name__ == "__main__":
    main()
