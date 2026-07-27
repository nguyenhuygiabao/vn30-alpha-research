from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = [
    "TradingDateId",
    "Ticker",
    "OpenPriceAdjusted",
    "HighPriceAdjusted",
    "LowPriceAdjusted",
    "ClosePriceAdjusted",
    "TotalMatchVolume",
    "ForeignBuyValueTotal",
    "ForeignSellValueTotal",
]

COLUMN_MAPPING = {
    "TradingDateId": "timestamp",
    "Ticker": "ticker",
    "OpenPriceAdjusted": "open",
    "HighPriceAdjusted": "high",
    "LowPriceAdjusted": "low",
    "ClosePriceAdjusted": "close",
    "TotalMatchVolume": "volume",
    "ForeignBuyValueTotal": "fb",
    "ForeignSellValueTotal": "fs",
}

EXPECTED = {
    "FLC": {
        "rows": 127,
        "first_date": "2016-07-26",
        "last_date": "2017-01-20",
    },
    "ROS": {
        "rows": 886,
        "first_date": "2017-07-24",
        "last_date": "2021-01-29",
    },
}


def read_archive_sheet(
    workbook_path: Path,
    ticker: str,
) -> pd.DataFrame:
    data = pd.read_excel(
        workbook_path,
        sheet_name=ticker,
        header=6,
        usecols="B:J",
    ).dropna(how="all")

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    missing = sorted(
        set(SOURCE_COLUMNS).difference(data.columns)
    )

    if missing:
        raise ValueError(
            f"{ticker} sheet is missing columns: {missing}"
        )

    data = (
        data[SOURCE_COLUMNS]
        .rename(columns=COLUMN_MAPPING)
        .copy()
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="raise",
    ).dt.normalize()

    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if set(data["ticker"].unique()) != {ticker}:
        raise ValueError(
            f"{ticker} sheet contains unexpected tickers"
        )

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in price_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    for column in ["volume", "fb", "fs"]:
        numeric = pd.to_numeric(
            data[column],
            errors="raise",
        )

        if numeric.dropna().mod(1).ne(0).any():
            raise ValueError(
                f"{ticker} {column} contains non-integers"
            )

        data[column] = numeric.astype("Int64")

    required = [
        "timestamp",
        "ticker",
        *price_columns,
        "volume",
    ]

    if data[required].isna().any().any():
        raise ValueError(
            f"{ticker} contains missing required values"
        )

    if data.duplicated(
        ["ticker", "timestamp"],
        keep=False,
    ).any():
        raise ValueError(
            f"{ticker} contains duplicate keys"
        )

    if data[price_columns].le(0).any().any():
        raise ValueError(
            f"{ticker} contains non-positive prices"
        )

    if data["volume"].lt(0).any():
        raise ValueError(
            f"{ticker} contains negative volume"
        )

    for column in ["fb", "fs"]:
        if data[column].lt(0).fillna(False).any():
            raise ValueError(
                f"{ticker} contains negative {column}"
            )

    if data["high"].lt(
        data[["open", "low", "close"]].max(axis=1)
    ).any():
        raise ValueError(
            f"{ticker} contains invalid high prices"
        )

    if data["low"].gt(
        data[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError(
            f"{ticker} contains invalid low prices"
        )

    rule = EXPECTED[ticker]

    if len(data) != rule["rows"]:
        raise ValueError(
            f"{ticker}: expected {rule['rows']} rows, "
            f"found {len(data)}"
        )

    if data["timestamp"].min() != pd.Timestamp(
        rule["first_date"]
    ):
        raise ValueError(
            f"{ticker} has an unexpected first date"
        )

    if data["timestamp"].max() != pd.Timestamp(
        rule["last_date"]
    ):
        raise ValueError(
            f"{ticker} has an unexpected last date"
        )

    data["fn"] = (
        data["fb"] - data["fs"]
    ).astype("Int64")

    return data[
        [
            "ticker",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "fb",
            "fs",
            "fn",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    combined = (
        pd.concat(
            [
                read_archive_sheet(args.input, ticker)
                for ticker in EXPECTED
            ],
            ignore_index=True,
        )
        .sort_values(["ticker", "timestamp"])
        .reset_index(drop=True)
    )

    if combined.duplicated(
        ["ticker", "timestamp"]
    ).any():
        raise ValueError(
            "Combined archive contains duplicate keys"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        args.output,
        index=False,
        date_format="%Y-%m-%d",
    )

    metadata = {
        "source_file": args.input.name,
        "source_sha256": hashlib.sha256(
            args.input.read_bytes()
        ).hexdigest(),
        "price_basis": "vendor-adjusted",
        "same_method_as_fiinquant_adjusted": True,
        "raw_ohlc_included": False,
        "corporate_action_records_included": False,
        "foreign_net_value_definition": "fb - fs",
        "foreign_zero_policy": (
            "Explicit zero means no recorded activity; "
            "blank values remain missing."
        ),
    }

    metadata_path = args.output.with_suffix(
        ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("ARCHIVED FIINQUANT IMPORT")
    print(f"Total rows: {len(combined)}")
    print(
        "Duplicate ticker-date keys:",
        combined.duplicated(
            ["ticker", "timestamp"]
        ).sum(),
    )
    print(
        "Missing foreign-buy values:",
        combined["fb"].isna().sum(),
    )
    print(
        "Missing foreign-sell values:",
        combined["fs"].isna().sum(),
    )
    print()

    for ticker, group in combined.groupby(
        "ticker",
        sort=True,
    ):
        print(
            f"{ticker}: rows={len(group)}, "
            f"dates={group['timestamp'].min().date()} "
            f"to {group['timestamp'].max().date()}, "
            f"fb_zero={group['fb'].eq(0).sum()}, "
            f"fs_zero={group['fs'].eq(0).sum()}"
        )

    print()
    print(f"Output: {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
