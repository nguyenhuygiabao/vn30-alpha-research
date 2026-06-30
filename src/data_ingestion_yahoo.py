from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


UNIVERSE_PATH: str = "config/vn30_test_universe.csv"
OUTPUT_PATH: str = "data/raw/yahoo/vn30_test_ohlcv.csv"

START_DATE: str = "2020-01-01"


def flatten_yahoo_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    flattened = data.copy()

    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = flattened.columns.get_level_values(0)

    return flattened


def normalize_yahoo_ohlcv(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    normalized = flatten_yahoo_columns(data)

    normalized = normalized.reset_index()

    normalized = normalized.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
    )

    normalized["ticker"] = ticker

    required_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in normalized.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing Yahoo columns for {ticker}: {missing_columns}"
        )

    normalized = normalized[required_columns].copy()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized["date"] = pd.to_datetime(normalized["date"])

    normalized["value_traded"] = (
        normalized["adjusted_close"]
        * normalized["volume"]
    )

    return normalized


def download_one_ticker(
    ticker: str,
    yahoo_symbol: str,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    data = yf.download(
        yahoo_symbol,
        start=start_date,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(
            f"No data returned for {ticker} / {yahoo_symbol}"
        )

    return normalize_yahoo_ohlcv(
        data=data,
        ticker=ticker,
    )


def download_universe(
    universe_path: str = UNIVERSE_PATH,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    universe = pd.read_csv(universe_path)

    required_universe_columns = [
        "ticker",
        "yahoo_symbol",
        "issuer_group",
    ]

    missing_columns = [
        column
        for column in required_universe_columns
        if column not in universe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing universe columns: {missing_columns}"
        )

    frames = []

    for row in universe.itertuples(index=False):
        ticker_data = download_one_ticker(
            ticker=row.ticker,
            yahoo_symbol=row.yahoo_symbol,
            start_date=start_date,
        )

        frames.append(ticker_data)

        print(
            row.ticker,
            row.yahoo_symbol,
            "rows:",
            len(ticker_data),
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(drop=True)

    return combined


def save_raw_ohlcv(
    data: pd.DataFrame,
    output_path: str = OUTPUT_PATH,
) -> str:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )

    return str(path)


def main() -> None:
    data = download_universe()

    output_path = save_raw_ohlcv(data)

    duplicate_keys = data.duplicated(
        [
            "date",
            "ticker",
        ]
    ).sum()

    print("Yahoo ingestion completed.")
    print("Output rows:", len(data))
    print("Output columns:", len(data.columns))
    print("Tickers:", sorted(data["ticker"].unique().tolist()))
    print("Earliest date:", data["date"].min())
    print("Latest date:", data["date"].max())
    print("Duplicate ticker-date keys:", duplicate_keys)
    print("Output path:", output_path)


if __name__ == "__main__":
    main()