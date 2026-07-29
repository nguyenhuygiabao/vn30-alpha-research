from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_fiinquant_history import (
    ARCHIVE_TICKERS,
    METADATA_PATH,
    RAW_ARCHIVE_PATH,
    RAW_PATH,
    load_raw_archive,
)
from src.market_data_schema import load_canonical_market_history

ADJUSTED_PATH = (
    ROOT
    / "data"
    / "raw"
    / "fiinquant"
    / "canonical"
    / "adjusted_market_history.csv"
)


def backfill_raw_archive() -> None:
    """
    Append the FLC/ROS raw-OHLC archive into the already-downloaded local
    canonical raw history, without touching the live FiinQuant API.

    This applies the same archive-merge logic that
    scripts/download_fiinquant_history.py runs during a full live rebuild
    (via the shared load_raw_archive helper), so a future live rebuild and
    this standalone backfill cannot silently diverge.
    """
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Canonical metadata not found: {METADATA_PATH}"
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    existing_raw = pd.read_csv(RAW_PATH, dtype={"ticker": "string"})
    existing_raw.columns = [
        str(column).strip().lower() for column in existing_raw.columns
    ]
    existing_raw["timestamp"] = pd.to_datetime(
        existing_raw["timestamp"],
        errors="raise",
    ).dt.normalize()

    already_present = set(
        existing_raw.loc[
            existing_raw["ticker"].isin(ARCHIVE_TICKERS), "ticker"
        ].unique()
    )

    if already_present:
        raise ValueError(
            "Raw market history already contains archive tickers: "
            f"{sorted(already_present)}. Refusing to duplicate rows."
        )

    raw_archive = load_raw_archive(
        metadata["from_date"],
        metadata["to_date"],
    )

    combined_raw = pd.concat(
        [existing_raw, raw_archive],
        ignore_index=True,
    ).sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    if combined_raw.duplicated(["ticker", "timestamp"]).any():
        raise ValueError(
            "Combined raw history contains duplicate ticker-date keys"
        )

    temporary_raw = RAW_PATH.with_suffix(".tmp")

    try:
        combined_raw.to_csv(
            temporary_raw,
            index=False,
            date_format="%Y-%m-%d",
        )

        # Validate against the real adjusted history before committing;
        # this is the same gate write_canonical_outputs uses for a live
        # rebuild.
        market = load_canonical_market_history(
            temporary_raw,
            ADJUSTED_PATH,
        )

        os.replace(temporary_raw, RAW_PATH)
    finally:
        temporary_raw.unlink(missing_ok=True)

    raw_source_sha256 = hashlib.sha256(
        RAW_ARCHIVE_PATH.read_bytes()
    ).hexdigest()

    metadata["raw_tickers"] = int(combined_raw["ticker"].nunique())
    metadata["raw_rows"] = int(len(combined_raw))
    metadata["raw_history_unavailable"] = []
    metadata["raw_traded_value_unavailable"] = sorted(ARCHIVE_TICKERS)
    metadata["raw_traded_value_unavailable_source"] = {
        "archive_path": str(
            RAW_ARCHIVE_PATH.relative_to(ROOT)
        ).replace("\\", "/"),
        "archive_sha256": raw_source_sha256,
        "policy": (
            "Vendor never supplied a total traded-value (VND) figure for "
            "these raw prices. Traded value stays missing for FLC/ROS "
            "rather than being derived from price * volume."
        ),
    }

    fallback = market.loc[
        market[["open", "high", "low", "close"]].isna().all(axis=1)
    ]

    print("FLC/ROS RAW ARCHIVE BACKFILL")
    print("=" * 72)
    print(f"Raw rows before: {len(existing_raw)}")
    print(f"Raw rows after: {len(combined_raw)}")
    print(f"Rows added: {len(combined_raw) - len(existing_raw)}")
    print(
        "Remaining raw-price fallback rows: "
        f"{market.attrs['raw_price_fallback_rows']}"
    )
    print(
        "Remaining raw-price fallback tickers: "
        f"{market.attrs['raw_price_fallback_tickers']}"
    )
    print(
        "Rows still missing traded value: "
        f"{market.attrs['missing_value_traded_rows']}"
    )
    print(
        "Tickers still missing traded value: "
        f"{market.attrs['missing_value_traded_tickers']}"
    )

    if not fallback.empty:
        raise AssertionError(
            "Unexpected residual raw-price fallback rows after backfill"
        )

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Updated: {RAW_PATH}")
    print(f"Updated: {METADATA_PATH}")


if __name__ == "__main__":
    backfill_raw_archive()
