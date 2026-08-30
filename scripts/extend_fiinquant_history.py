"""Extend the canonical FiinQuant history forward without refetching it.

FiinQuant serves only the last 3650 days: a request reaching further back is
refused with `TimeFrameLimitFailed`. The canonical history starts 2016-07-26
and was downloaded on 2026-07-24, the last day that date was still inside the
window, so the early history can no longer be retrieved from the vendor and
the local files are the only copy. `download_fiinquant_history.py` writes only
what it fetched, with no merge against what is already on disk, so it can no
longer be used to refresh the history: narrowing its `--from-date` to a
recent window would replace ten years of data with that window.

This script fetches a recent window and merges it into the existing files.

The merge is not a plain append. Adjusted prices are back-adjusted with the
present as the anchor, so any corporate action inside the fetched window
restates every earlier adjusted price by a constant factor. Appending
new-basis rows onto old-basis history would produce a series that is silently
wrong before the join. The script therefore refetches an overlap window,
measures the per-ticker ratio between the new and the stored adjusted prices,
and rescales the stored history by that ratio before appending, so the result
sits on one basis throughout.

Raw prices are never restated, so they are compared instead: a difference in
the overlap means the vendor changed history and is reported rather than
silently absorbed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_fiinquant_history import (  # noqa: E402
    ADJUSTED_FIELDS,
    ADJUSTED_PATH,
    ARCHIVE_TICKERS,
    FOREIGN_PATH,
    KEYS,
    METADATA_PATH,
    RAW_FIELDS,
    RAW_PATH,
    clean_vendor_frame,
    create_fiinquant_client,
    fetch_history,
    get_credentials,
    write_canonical_outputs,
)

VENDOR_WINDOW_DAYS = 3650
BACKUP_ROOT = ROOT / "data" / "raw" / "fiinquant" / "backups"

PRICE_COLUMNS = ("open", "high", "low", "close")
RAW_COLUMNS = ["open", "high", "low", "close", "volume", "value"]
FOREIGN_COLUMNS = ["fb", "fs", "fn"]

# A ratio this far from 1.0 is treated as a real restatement rather than
# vendor rounding.
RESTATEMENT_TOLERANCE = 1e-6
# Within one ticker the restatement factor must be constant. Allow a little
# slack for the vendor rounding adjusted prices to a fixed precision.
FACTOR_CONSISTENCY_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a recent FiinQuant window and merge it into the existing "
            "canonical history."
        ),
    )
    parser.add_argument(
        "--to-date",
        default=date.today().isoformat(),
        help="Last date to fetch. Defaults to today.",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=120,
        help=(
            "Calendar days of already-stored history to refetch alongside "
            "the new rows. The overlap is what makes restatement detectable, "
            "so it should comfortably span several corporate actions."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Tickers per vendor request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch, compare and report, but write nothing.",
    )
    parser.add_argument(
        "--allow-raw-drift",
        action="store_true",
        help=(
            "Write even though the vendor changed raw prices inside the "
            "overlap. Off by default because raw prices should never move."
        ),
    )
    return parser.parse_args()


def load_existing() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in (RAW_PATH, ADJUSTED_PATH, FOREIGN_PATH):
        if not path.exists():
            raise SystemExit(
                f"Missing canonical file: {path}. This script extends an "
                "existing history and cannot create one."
            )
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frames.append(frame)
    return frames[0], frames[1], frames[2]


def check_window(from_date: pd.Timestamp, to_date: pd.Timestamp) -> None:
    earliest = pd.Timestamp(date.today()) - timedelta(days=VENDOR_WINDOW_DAYS)
    if from_date < earliest:
        raise SystemExit(
            f"Requested start {from_date.date()} is outside the vendor's "
            f"{VENDOR_WINDOW_DAYS}-day window, which today begins "
            f"{earliest.date()}. Reduce --overlap-days."
        )
    if to_date < from_date:
        raise SystemExit("--to-date is before the computed start date.")


def active_tickers(raw: pd.DataFrame) -> list[str]:
    """Tickers still trading at the end of the stored history.

    Delisted names such as FLC and ROS are served from a vendor archive, not
    the API, and asking for them would fail the whole batch.
    """
    latest = raw["timestamp"].max()
    recent = raw.loc[raw["timestamp"] > latest - pd.Timedelta(days=30)]
    tickers = set(recent["ticker"].unique()) - set(ARCHIVE_TICKERS)
    return sorted(tickers)


def measure_adjustment_factors(
    stored: pd.DataFrame,
    fetched: pd.DataFrame,
) -> tuple[dict[str, float], list[str]]:
    """Per-ticker ratio between newly fetched and stored adjusted closes."""
    overlap = stored.merge(
        fetched,
        on=KEYS,
        how="inner",
        suffixes=("_stored", "_new"),
    )
    if overlap.empty:
        raise SystemExit(
            "The fetched window does not overlap the stored history, so a "
            "restatement could not be ruled out. Increase --overlap-days."
        )

    factors: dict[str, float] = {}
    inconsistent: list[str] = []

    for ticker, group in overlap.groupby("ticker"):
        usable = group.loc[
            group["close_stored"].gt(0) & group["close_new"].gt(0)
        ]
        if usable.empty:
            continue

        ratio = usable["close_new"] / usable["close_stored"]
        spread = float(ratio.max() - ratio.min())
        factor = float(ratio.median())

        if spread > FACTOR_CONSISTENCY_TOLERANCE * max(factor, 1.0):
            inconsistent.append(
                f"{ticker}: ratio ranges {ratio.min():.8f} to "
                f"{ratio.max():.8f} across {len(usable)} overlapping rows"
            )
            continue

        factors[ticker] = factor

    return factors, inconsistent


def compare_raw_overlap(
    stored: pd.DataFrame,
    fetched: pd.DataFrame,
) -> list[str]:
    """Raw prices are never restated, so any overlap difference is a flag."""
    overlap = stored.merge(
        fetched,
        on=KEYS,
        how="inner",
        suffixes=("_stored", "_new"),
    )
    differences: list[str] = []

    for column in PRICE_COLUMNS + ("volume",):
        stored_values = overlap[f"{column}_stored"].fillna(0)
        new_values = overlap[f"{column}_new"].fillna(0)
        delta = (new_values - stored_values).abs()
        moved = int((delta > 0.5).sum())
        if moved:
            worst = overlap.loc[delta.idxmax()]
            differences.append(
                f"{column}: {moved} of {len(overlap)} overlapping rows "
                f"differ, worst {worst['ticker']} on "
                f"{pd.Timestamp(worst['timestamp']).date()} "
                f"{stored_values.loc[delta.idxmax()]:.2f} -> "
                f"{new_values.loc[delta.idxmax()]:.2f}"
            )

    return differences


def rescale_stored_adjusted(
    stored: pd.DataFrame,
    factors: dict[str, float],
    before: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    """Put stored adjusted prices onto the newly fetched adjustment basis."""
    rescaled = stored.copy()
    applied: list[str] = []

    for ticker, factor in sorted(factors.items()):
        if abs(factor - 1.0) <= RESTATEMENT_TOLERANCE:
            continue

        mask = rescaled["ticker"].eq(ticker) & rescaled["timestamp"].lt(before)
        rows = int(mask.sum())
        if not rows:
            continue

        for column in PRICE_COLUMNS:
            rescaled.loc[mask, column] = rescaled.loc[mask, column] * factor

        applied.append(
            f"{ticker}: {rows} stored rows multiplied by {factor:.10f}"
        )

    return rescaled, applied


def merge_history(
    stored: pd.DataFrame,
    fetched: pd.DataFrame,
    window_start: pd.Timestamp,
) -> pd.DataFrame:
    """Keep stored rows before the window; take fetched rows inside it."""
    kept = stored.loc[stored["timestamp"] < window_start]
    combined = pd.concat([kept, fetched], ignore_index=True)
    combined = combined.sort_values(KEYS).reset_index(drop=True)

    if combined.duplicated(KEYS).any():
        raise SystemExit("Merged history contains duplicate ticker-date keys")

    return combined


def backup_canonical() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_ROOT / f"{stamp}_before_extension"
    destination.mkdir(parents=True, exist_ok=True)
    for path in (RAW_PATH, ADJUSTED_PATH, FOREIGN_PATH, METADATA_PATH):
        if path.exists():
            shutil.copy2(path, destination / path.name)
    return destination


def describe(name: str, frame: pd.DataFrame) -> None:
    print(
        f"  {name:<14} {len(frame):>7} rows  "
        f"{frame['ticker'].nunique():>3} tickers  "
        f"{frame['timestamp'].min().date()} -> "
        f"{frame['timestamp'].max().date()}"
    )


def main() -> None:
    args = parse_args()

    stored_raw, stored_adjusted, stored_foreign = load_existing()
    stored_end = stored_raw["timestamp"].max()

    to_date = pd.Timestamp(args.to_date)
    from_date = stored_end - pd.Timedelta(days=args.overlap_days)
    check_window(from_date, to_date)

    print("STORED HISTORY")
    describe("raw", stored_raw)
    describe("adjusted", stored_adjusted)
    describe("foreign flow", stored_foreign)

    tickers = active_tickers(stored_raw)
    print(
        f"\nFetching {from_date.date()} -> {to_date.date()} for "
        f"{len(tickers)} tickers "
        f"({args.overlap_days} days of overlap for restatement detection)\n"
    )

    username, password = get_credentials()
    client = create_fiinquant_client(username, password)

    fetched_raw = fetch_history(
        client,
        tickers,
        RAW_FIELDS,
        adjusted=False,
        from_date=from_date.date().isoformat(),
        to_date=to_date.date().isoformat(),
        batch_size=args.batch_size,
    )
    fetched_raw = clean_vendor_frame(
        fetched_raw,
        set(KEYS + RAW_FIELDS),
        "Raw FiinQuant window",
    )

    fetched_adjusted = fetch_history(
        client,
        tickers,
        ADJUSTED_FIELDS,
        adjusted=True,
        from_date=from_date.date().isoformat(),
        to_date=to_date.date().isoformat(),
        batch_size=args.batch_size,
    )
    fetched_adjusted = clean_vendor_frame(
        fetched_adjusted,
        set(KEYS + ADJUSTED_FIELDS),
        "Adjusted FiinQuant window",
        reference=fetched_raw,
    )

    new_raw = fetched_raw[KEYS + RAW_COLUMNS].copy()
    new_foreign = fetched_raw[KEYS + FOREIGN_COLUMNS].copy()
    new_adjusted = fetched_adjusted[KEYS + list(ADJUSTED_FIELDS)].merge(
        fetched_raw[KEYS + ["volume", "value"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    print("\nFETCHED WINDOW")
    describe("raw", new_raw)
    describe("adjusted", new_adjusted)
    describe("foreign flow", new_foreign)

    added = new_raw.loc[new_raw["timestamp"] > stored_end]
    print(
        f"\nRows after {stored_end.date()}: {len(added)} across "
        f"{added['ticker'].nunique()} tickers"
    )
    if added.empty:
        raise SystemExit(
            "The vendor returned nothing newer than the stored history. "
            "Nothing to extend."
        )
    print(
        f"New trading dates: "
        f"{sorted(d.date().isoformat() for d in added['timestamp'].unique())}"
    )

    print("\nRAW OVERLAP CHECK")
    raw_differences = compare_raw_overlap(stored_raw, new_raw)
    if raw_differences:
        for line in raw_differences:
            print(f"  !! {line}")
    else:
        print("  raw prices and volume unchanged in the overlap")

    print("\nADJUSTMENT BASIS CHECK")
    factors, inconsistent = measure_adjustment_factors(
        stored_adjusted,
        new_adjusted,
    )
    if inconsistent:
        for line in inconsistent:
            print(f"  !! {line}")
        raise SystemExit(
            "\nThe new adjusted prices do not differ from the stored ones by "
            "a constant per-ticker factor, so this is not a clean "
            "back-adjustment restatement. Refusing to write."
        )

    restated = {
        ticker: factor
        for ticker, factor in factors.items()
        if abs(factor - 1.0) > RESTATEMENT_TOLERANCE
    }
    if restated:
        print(f"  {len(restated)} tickers restated by a corporate action:")
        for ticker, factor in sorted(restated.items()):
            print(f"    {ticker}: stored prices scale by {factor:.10f}")
    else:
        print(
            f"  no restatement: all {len(factors)} tickers match the stored "
            "basis"
        )

    if raw_differences and not args.allow_raw_drift:
        raise SystemExit(
            "\nRaw prices changed inside the overlap. Raw prices are not "
            "restated by corporate actions, so this means the vendor revised "
            "history. Review the differences above, then rerun with "
            "--allow-raw-drift to accept them."
        )

    rescaled_adjusted, applied = rescale_stored_adjusted(
        stored_adjusted,
        factors,
        from_date,
    )
    if applied:
        print("\nRESCALED STORED HISTORY ONTO THE NEW BASIS")
        for line in applied:
            print(f"  {line}")

    merged_raw = merge_history(stored_raw, new_raw, from_date)
    merged_adjusted = merge_history(rescaled_adjusted, new_adjusted, from_date)
    merged_foreign = merge_history(stored_foreign, new_foreign, from_date)

    raw_keys = set(map(tuple, merged_raw[KEYS].values))
    adjusted_keys = set(map(tuple, merged_adjusted[KEYS].values))
    if raw_keys != adjusted_keys:
        raise SystemExit(
            "Merged raw and adjusted histories do not cover the same "
            f"ticker-dates: {len(raw_keys ^ adjusted_keys)} keys differ"
        )

    print("\nMERGED HISTORY")
    describe("raw", merged_raw)
    describe("adjusted", merged_adjusted)
    describe("foreign flow", merged_foreign)
    print(
        f"\n  raw rows {len(stored_raw)} -> {len(merged_raw)} "
        f"({len(merged_raw) - len(stored_raw):+})"
    )

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    destination = backup_canonical()
    print(f"\nBacked up previous canonical files to {destination}")

    write_canonical_outputs(merged_raw, merged_adjusted, merged_foreign)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["extended_at_vietnam"] = datetime.now(
        ZoneInfo("Asia/Ho_Chi_Minh")
    ).isoformat()
    metadata["to_date"] = to_date.date().isoformat()
    metadata["raw_rows"] = len(merged_raw)
    metadata["adjusted_rows"] = len(merged_adjusted)
    metadata["foreign_flow_rows"] = len(merged_foreign)
    metadata["extension_history"] = metadata.get("extension_history", []) + [
        {
            "fetched_from": from_date.date().isoformat(),
            "fetched_to": to_date.date().isoformat(),
            "rows_added": len(merged_raw) - len(stored_raw),
            "previous_end": stored_end.date().isoformat(),
            "restated_tickers": {
                ticker: factor for ticker, factor in sorted(restated.items())
            },
            "raw_overlap_differences": raw_differences,
            "backup": destination.name,
        }
    ]
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nCANONICAL FIINQUANT HISTORY EXTENDED")
    print("Raw output:", RAW_PATH)
    print("Adjusted output:", ADJUSTED_PATH)
    print("Foreign-flow output:", FOREIGN_PATH)


if __name__ == "__main__":
    main()
