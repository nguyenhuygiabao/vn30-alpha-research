from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from vnstock import Quote


UNIVERSE_PATH: str = "config/vn30_universe.csv"
OUTPUT_PATH: str = "data/raw/vnstock/vn30_ohlcv.csv"
AUDIT_PATH: str = "data/raw/vnstock/vn30_coverage_audit.csv"

SOURCE: str = "kbs"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-06-30"
SLEEP_SECONDS: int = 4


def normalize_vnstock_ohlcv(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    normalized = data.copy()

    normalized = normalized.rename(
        columns={
            "time": "date",
        }
    )

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in normalized.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing vnstock columns for {ticker}: {missing_columns}"
        )

    normalized = normalized[required_columns].copy()

    normalized["ticker"] = ticker

    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized["adjusted_close"] = normalized["close"]

    normalized["value_traded"] = (
        normalized["close"]
        * normalized["volume"]
    )

    output_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "value_traded",
    ]

    return normalized[output_columns].copy()


def download_one_symbol(
    ticker: str,
    vnstock_symbol: str,
) -> pd.DataFrame:
    quote = Quote(
        source=SOURCE,
        symbol=vnstock_symbol,
    )

    data = quote.history(
        symbol=vnstock_symbol,
        start=START_DATE,
        end=END_DATE,
        interval="1D",
    )

    if data.empty:
        raise ValueError(
            f"No data returned for {ticker} / {vnstock_symbol}"
        )

    return normalize_vnstock_ohlcv(
        data=data,
        ticker=ticker,
    )


def main() -> None:
    universe = pd.read_csv(UNIVERSE_PATH)

    required_universe_columns = [
        "ticker",
        "vnstock_symbol",
        "issuer_group",
    ]

    missing_universe_columns = [
        column
        for column in required_universe_columns
        if column not in universe.columns
    ]

    if missing_universe_columns:
        raise ValueError(
            f"Missing universe columns: {missing_universe_columns}"
        )

    frames = []
    audit_rows = []

    for index, row in enumerate(universe.itertuples(index=False), start=1):
        print(
            f"[{index:02d}/{len(universe)}] "
            f"Downloading {row.ticker} / {row.vnstock_symbol}..."
        )

        try:
            ticker_data = download_one_symbol(
                ticker=row.ticker,
                vnstock_symbol=row.vnstock_symbol,
            )

            frames.append(ticker_data)

            audit_rows.append(
                {
                    "ticker": row.ticker,
                    "vnstock_symbol": row.vnstock_symbol,
                    "issuer_group": row.issuer_group,
                    "status": "ok",
                    "rows": len(ticker_data),
                    "start": ticker_data["date"].min(),
                    "end": ticker_data["date"].max(),
                    "error": "",
                }
            )

            print(
                row.ticker,
                "rows:",
                len(ticker_data),
            )

        except Exception as error:
            audit_rows.append(
                {
                    "ticker": row.ticker,
                    "vnstock_symbol": row.vnstock_symbol,
                    "issuer_group": row.issuer_group,
                    "status": "error",
                    "rows": 0,
                    "start": pd.NaT,
                    "end": pd.NaT,
                    "error": repr(error),
                }
            )

            print(
                row.ticker,
                "error:",
                repr(error),
            )

        time.sleep(SLEEP_SECONDS)

    if not frames:
        raise ValueError("No ticker data was downloaded.")

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

    output_path = Path(OUTPUT_PATH)
    audit_path = Path(AUDIT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        audit_path,
        index=False,
    )

    duplicate_keys = combined.duplicated(
        [
            "date",
            "ticker",
        ]
    ).sum()

    print("vnstock ingestion completed.")
    print("Output rows:", len(combined))
    print("Output columns:", len(combined.columns))
    print("Tickers:", combined["ticker"].nunique())
    print("Earliest date:", combined["date"].min())
    print("Latest date:", combined["date"].max())
    print("Duplicate ticker-date keys:", duplicate_keys)
    print("Output path:", output_path)
    print("Audit path:", audit_path)

    print("\nCoverage summary:")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()