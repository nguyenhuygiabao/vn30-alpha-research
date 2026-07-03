from pathlib import Path

import pandas as pd 
import numpy as np

from .data_loader import load_ohlcv_csv

SOURCE_PATH = "data/raw/vnstock/vn30_ohlcv.csv"
OUTPUT_PATH = "data/processed/features_basic.parquet"

RETURN_WINDOW_SHORT = 5
RETURN_WINDOW_LONG  =20 
VOLATILITY_WINDOW = 20 
LIQUIDITY_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252

def build_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create basic return, risk, and liquidity features."""

    features = (
        data
        .sort_values(["ticker", "date"])
        .reset_index(drop = True)
        .copy()
    )

    previous_adjusted_close = (
        features
        .groupby("ticker")["adjusted_close"]
        .shift(1)
    )

    features["simple_return_1d"] = (
        features["adjusted_close"]/previous_adjusted_close -1
    )

    features["log_return_1d"] = np.log(
        features["adjusted_close"]/previous_adjusted_close
    )

    adjusted_close_5d_ago = (
        features
        .groupby("ticker")["adjusted_close"]
        .shift(RETURN_WINDOW_SHORT)
    )

    features["rolling_return_5d"] = (
        features["adjusted_close"]/adjusted_close_5d_ago -1
    )

    adjusted_close_20d_ago = (
        features
        .groupby("ticker")["adjusted_close"]
        .shift(RETURN_WINDOW_LONG)
    )
    
    features["rolling_return_20d"] = (
        features["adjusted_close"]/adjusted_close_20d_ago -1 
    )

    features["rolling_vol_20d"] = (
        features 
            .groupby("ticker")["simple_return_1d"]
            .transform(
                lambda returns: ( 
                    returns
                    .rolling(
                        window = VOLATILITY_WINDOW,
                        min_periods = VOLATILITY_WINDOW,
                    )
                .std()
                *np.sqrt(TRADING_DAYS_PER_YEAR)
             )
        )
    )

    running_peak = (
        features
        .groupby("ticker")["adjusted_close"]
        .cummax()
    )

    features["drawdown"] = (
        features["adjusted_close"]/running_peak -1 
    )

    features["average_daily_volume_20d"] = (
        features 
        .groupby("ticker")["volume"]
        .transform(
            lambda volume: (
                volume. 
                rolling(
                    window = LIQUIDITY_WINDOW,
                    min_periods = LIQUIDITY_WINDOW,
                )
                .mean()
            )
        )
    )

    if "value_traded" in features.columns:
        features["average_daily_value_20d"] = (
            features
            .groupby("ticker")["value_traded"]
            .transform(
                lambda traded_value: (
                    traded_value
                    .rolling(
                        window = LIQUIDITY_WINDOW,
                        min_periods = LIQUIDITY_WINDOW,
                    )
                    .mean()
                )
            )
        )

    features["traded_value_rank_20d"] = (
        features
        .groupby("date")["average_daily_value_20d"]
        .rank(
            method = "average",
            ascending= True, 
            pct = True, 
        )
    )

    return features

def main() -> None: 
    """Build, save, and verify the basic feature dataset"""

    data = load_ohlcv_csv(SOURCE_PATH)

    features = build_basic_features(data)

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents = True, 
        exist_ok = True,    
    )

    features.to_parquet(
        output_path,
        index = False,
    ) 

    reloaded_features = pd.read_parquet(output_path)

    print("Basic feature generation completed.")
    print(f"Input rows: {len(data)}")
    print(f"Output rows: {len(features)}")
    print(f"Reloaded rows: {len(reloaded_features)}")
    print(f"Tickers: {reloaded_features['ticker'].nunique()}")
    print(f"Output file: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()