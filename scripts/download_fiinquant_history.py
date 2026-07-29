from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exchange_history import attach_exchange_history
from src.exchange_rules import (
    FIRST_TRADING_DAY,
    REGULAR_SESSION,
    RESUMPTION_25_PLUS,
    TREASURY_SHARE_DISTRIBUTION,
    price_limit_fraction,
    tick_size_vnd,
)
from src.features_foreign_flow import (
    load_canonical_foreign_flow_csv,
)
from src.market_data_schema import (
    load_canonical_market_history,
)
from src.vn30_universe import historical_tickers

START_DATE = "2016-07-26"
END_DATE = "2026-07-22"
ARCHIVE_TICKERS = {"FLC", "ROS"}

OUTPUT_DIR = ROOT / "data" / "raw" / "fiinquant" / "canonical"
ARCHIVE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "fiinquant"
    / "archive"
    / "flc_ros_adjusted_history.csv"
)
RAW_ARCHIVE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "fiinquant"
    / "archive"
    / "flc_ros_raw_history.csv"
)

RAW_PATH = OUTPUT_DIR / "raw_market_history.csv"
ADJUSTED_PATH = OUTPUT_DIR / "adjusted_market_history.csv"
FOREIGN_PATH = OUTPUT_DIR / "foreign_flow_history.csv"
METADATA_PATH = OUTPUT_DIR / "history_metadata.json"
EXCHANGE_HISTORY_PATH = ROOT / "config" / "exchange_history.csv"

RAW_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
    "fb",
    "fs",
    "fn",
]
ADJUSTED_FIELDS = ["open", "high", "low", "close"]
KEYS = ["ticker", "timestamp"]

VENDOR_PRICE_CORRECTIONS = (
    {
        "dataset": "raw",
        "ticker": "HCM",
        "timestamp": "2020-06-05",
        "field": "open",
        "vendor_value": 14100.0,
        "corrected_value": 19700.0,
        "evidence": "Yahoo continuity and VCI adjusted OHLC",
    },
    {
        "dataset": "adjusted",
        "ticker": "HCM",
        "timestamp": "2020-06-05",
        "field": "open",
        "vendor_value": 4138.372073,
        "corrected_value": 5781.9808393,
        "evidence": "Raw correction multiplied by FiinQuant adjustment factor",
    },
    {
        "dataset": "raw",
        "ticker": "VIB",
        "timestamp": "2018-07-23",
        "field": "close",
        "vendor_value": 0.0,
        "corrected_value": 27300.0,
        "evidence": "VCI adjusted close and surrounding-price continuity",
    },
    {
        "dataset": "adjusted",
        "ticker": "VIB",
        "timestamp": "2018-07-23",
        "field": "close",
        "vendor_value": 0.0,
        "corrected_value": 3452.4981036,
        "evidence": "Raw correction multiplied by FiinQuant adjustment factor",
    },
)

ACCEPTED_OHLC_EXCEPTIONS = {
    ("BCM", "2018-11-23"): "UPCoM zero-volume placeholder",
    ("BCM", "2018-11-26"): "UPCoM zero-volume placeholder",
    ("BCM", "2018-11-27"): "UPCoM zero-volume placeholder",
    ("BCM", "2019-01-30"): "UPCoM zero-volume placeholder",
    ("BCM", "2019-07-04"): "Verified UPCoM price convention",
    ("BCM", "2020-01-21"): "UPCoM zero-volume placeholder",
    ("BCM", "2020-02-27"): "Verified UPCoM price convention",
    ("VIB", "2018-07-23"): "Verified UPCoM price convention",
    ("VIB", "2019-07-04"): "Verified UPCoM price convention",
}

EXCHANGE_SESSION_OVERRIDES = {
    ("GVR", "2020-03-17"): {
        "exchange": "HOSE",
        "session_kind": FIRST_TRADING_DAY,
        "reference_price": 11_570.0,
        "reason": (
            "First HOSE session after UPCoM transfer; "
            "official reference price 11,570 VND"
        ),
        "source": (
            "https://ssc.gov.vn/webcenter/portal/ubck/pages_r/l/"
            "chitit?dDocName=APPSSCGOVVN162131034"
        ),
    },
    ("KDC", "2021-11-30"): {
        "exchange": "HOSE",
        "session_kind": TREASURY_SHARE_DISTRIBUTION,
        "reference_price": 62_000.0,
        "reason": (
            "Treasury-share bonus ex-right session; HOSE kept "
            "the prior close as reference and applied a 20% band"
        ),
        "source": (
            "HOSE 2021 trading rules, Articles 10(4)(b) and 13; "
            "KDC treasury-share bonus event dated 2021-11-30"
        ),
    },
    ("VHM", "2018-05-17"): {
        "exchange": "HOSE",
        "session_kind": FIRST_TRADING_DAY,
        "reference_price": 92_100.0,
        "reason": (
            "First HOSE trading session; official reference "
            "price 92,100 VND"
        ),
        "source": (
            "https://www.bsi.com.vn/public/file/"
            "1529980005_1640601753.pdf"
        ),
    },
}

KNOWN_VENDOR_DUPLICATE_POLICIES = (
    {
        "ticker": "HNG",
        "timestamp": "2024-10-21",
        "decision": (
            "Retain the resolver-selected row with valid turnover; "
            "reject the conflicting row with volume 2053200 and value 0"
        ),
    },
)


def get_credentials() -> tuple[str, str]:
    username = os.getenv("FIINQUANT_USERNAME")
    password = os.getenv("FIINQUANT_PASSWORD")

    if not username:
        username = input("FiinQuant username: ").strip()

    if not password:
        password = getpass.getpass("FiinQuant password: ")

    if not username or not password:
        raise RuntimeError("FiinQuant credentials are required")

    return username, password


def create_fiinquant_client(
    username: str,
    password: str,
):
    try:
        from FiinQuantX import FiinSession
    except ImportError as exc:
        raise RuntimeError(
            "FiinQuantX is required to download market history. "
            "Run this command with the FiinQuant environment."
        ) from exc

    return FiinSession(
        username=username,
        password=password,
    ).login()


def clean_vendor_frame(
    data: pd.DataFrame,
    required: set[str],
    label: str,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned.columns = [
        str(column).strip().lower()
        for column in cleaned.columns
    ]

    duplicate_columns = cleaned.columns[
        cleaned.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            f"{label} contains duplicate normalized columns: "
            f"{sorted(set(duplicate_columns))}"
        )

    missing = required.difference(cleaned.columns)
    if missing:
        raise ValueError(
            f"{label} is missing columns: {sorted(missing)}"
        )

    cleaned = cleaned.loc[:, sorted(required)].copy()
    cleaned["ticker"] = (
        cleaned["ticker"].astype("string").str.strip().str.upper()
    )
    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        errors="raise",
    ).dt.normalize()

    for column in cleaned.columns.difference(KEYS):
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    exact_removed = len(cleaned) - len(cleaned.drop_duplicates())
    cleaned = cleaned.drop_duplicates()

    if exact_removed:
        print(f"{label}: removed {exact_removed} exact duplicate row(s)")

    duplicate_rows = cleaned.loc[
        cleaned.duplicated(KEYS, keep=False)
    ].copy()

    if duplicate_rows.empty:
        return cleaned.sort_values(KEYS).reset_index(drop=True)

    conflict_keys = duplicate_rows[KEYS].drop_duplicates()
    chosen: pd.DataFrame | None = None

    if reference is not None:
        price_columns = [
            column
            for column in ("open", "high", "low", "close")
            if column in cleaned.columns
            and column in reference.columns
        ]

        candidates = duplicate_rows.merge(
            reference[KEYS + price_columns],
            on=KEYS,
            how="left",
            validate="many_to_one",
            suffixes=("", "_reference"),
        )

        def consistent_adjustment(row: pd.Series) -> bool:
            ratios: list[float] = []

            for column in price_columns:
                adjusted_value = row[column]
                raw_value = row[f"{column}_reference"]

                if pd.isna(adjusted_value) or pd.isna(raw_value):
                    return False

                if adjusted_value <= 0 or raw_value <= 0:
                    return False

                ratios.append(float(adjusted_value / raw_value))

            if len(ratios) < 2:
                return False

            tolerance = 1e-8 * max(1.0, abs(ratios[0]))
            return max(ratios) - min(ratios) <= tolerance

        candidates["_reference_match"] = candidates.apply(
            consistent_adjustment,
            axis=1,
        )

        match_counts = candidates.groupby(
            KEYS,
            dropna=False,
        )["_reference_match"].sum()

        if match_counts.eq(1).all():
            chosen = candidates.loc[
                candidates["_reference_match"],
                cleaned.columns,
            ]

    elif {"volume", "value"}.issubset(cleaned.columns):
        candidates = duplicate_rows.copy()
        candidates["_valid_turnover"] = (
            candidates["volume"].gt(0)
            & candidates["value"].gt(0)
        )
        candidates["_invalid_turnover"] = (
            candidates["volume"].gt(0)
            & (
                candidates["value"].isna()
                | candidates["value"].le(0)
            )
        )

        quality = candidates.groupby(
            KEYS,
            dropna=False,
        ).agg(
            rows=("_valid_turnover", "size"),
            valid=("_valid_turnover", "sum"),
            invalid=("_invalid_turnover", "sum"),
        )

        resolvable = (
            quality["valid"].eq(1)
            & quality["valid"]
            .add(quality["invalid"])
            .eq(quality["rows"])
        ).all()

        if resolvable:
            chosen = candidates.loc[
                candidates["_valid_turnover"],
                cleaned.columns,
            ]

    if chosen is None:
        print(f"\n{label}: unresolved conflicting keys")
        print(duplicate_rows.head(40).to_string(index=False))
        raise ValueError(
            f"{label} contains conflicting duplicate keys"
        )

    conflict_index = pd.MultiIndex.from_frame(conflict_keys)
    cleaned_index = pd.MultiIndex.from_frame(cleaned[KEYS])

    cleaned = pd.concat(
        [
            cleaned.loc[
                ~cleaned_index.isin(conflict_index)
            ],
            chosen,
        ],
        ignore_index=True,
    )

    if cleaned.duplicated(KEYS).any():
        raise ValueError(
            f"{label} duplicate resolution failed"
        )

    print(
        f"{label}: resolved "
        f"{len(conflict_keys)} conflict(s)"
    )
    print(chosen.to_string(index=False))

    return cleaned.sort_values(KEYS).reset_index(drop=True)


def apply_vendor_price_corrections(
    data: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    corrected = data.copy()
    audit: list[dict[str, object]] = []

    for specification in VENDOR_PRICE_CORRECTIONS:
        if specification["dataset"] != dataset:
            continue

        timestamp = pd.Timestamp(specification["timestamp"])
        mask = (
            corrected["ticker"].eq(specification["ticker"])
            & corrected["timestamp"].eq(timestamp)
        )

        matches = int(mask.sum())

        if matches == 0:
            continue

        if matches != 1:
            raise ValueError(
                f"{dataset}: correction key has {matches} rows: "
                f"{specification['ticker']} {specification['timestamp']}"
            )

        field = str(specification["field"])
        actual = float(corrected.loc[mask, field].iloc[0])
        vendor_value = float(specification["vendor_value"])
        replacement = float(specification["corrected_value"])
        tolerance = max(1e-6, abs(vendor_value) * 1e-10)

        if pd.isna(actual):
            raise ValueError(
                f"{dataset}: correction source value is missing: "
                f"{specification['ticker']} {specification['timestamp']} "
                f"{field}"
            )

        if abs(actual - replacement) <= tolerance:
            status = "already_correct"
        elif abs(actual - vendor_value) <= tolerance:
            corrected.loc[mask, field] = replacement
            status = "corrected"
        else:
            raise ValueError(
                f"{dataset}: unexplained vendor value for "
                f"{specification['ticker']} {specification['timestamp']} "
                f"{field}: expected {vendor_value} or {replacement}, "
                f"received {actual}"
            )

        record = dict(specification)
        record["observed_value"] = actual
        record["status"] = status
        audit.append(record)

    return corrected, audit


def repair_adjusted_ohlc_from_raw(
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    repaired = adjusted.copy()
    original_close = repaired["close"].copy()

    invalid_mask = (
        repaired["high"].lt(
            repaired[["open", "close"]].max(axis=1)
        )
        | repaired["low"].gt(
            repaired[["open", "close"]].min(axis=1)
        )
        | repaired["low"].gt(repaired["high"])
    )

    targeted_indices = repaired.index[invalid_mask].tolist()

    raw_keys = raw[["ticker", "timestamp"]]

    if raw_keys.duplicated().any():
        raise ValueError(
            "Raw data contains duplicate ticker-timestamp keys."
        )

    raw_lookup = raw.set_index(
        ["ticker", "timestamp"]
    )

    repaired_fields = [
        "open",
        "high",
        "low",
    ]

    for index in targeted_indices:
        key = (
            repaired.at[index, "ticker"],
            repaired.at[index, "timestamp"],
        )

        if key not in raw_lookup.index:
            raise ValueError(
                "Invalid adjusted OHLC row has no raw counterpart: "
                f"{key}"
            )

        raw_row = raw_lookup.loc[key]
        raw_close = raw_row["close"]
        adjusted_close = repaired.at[index, "close"]

        if (
            pd.isna(raw_close)
            or raw_close == 0
            or pd.isna(adjusted_close)
        ):
            raise ValueError(
                "Invalid adjusted OHLC row has no usable raw counterpart: "
                f"{key}"
            )

        adjustment_factor = adjusted_close / raw_close

        for field in repaired_fields:
            repaired.at[index, field] = (
                raw_row[field] * adjustment_factor
            )

    audit = {
        "targeted_rows": len(targeted_indices),
        "repaired_fields": repaired_fields,
        "adjusted_close_changed": not repaired[
            "close"
        ].equals(original_close),
        "adjusted_only_rows_changed": False,
    }

    return repaired, audit

def validate_ohlc(
    data: pd.DataFrame,
    dataset: str,
) -> list[dict[str, object]]:
    prices = ["open", "high", "low", "close"]

    critical = pd.DataFrame(index=data.index)

    for column in prices:
        critical[f"missing_{column}"] = data[column].isna()
        critical[f"nonpositive_{column}"] = data[column].le(0)

    critical_invalid = critical.any(axis=1)

    if critical_invalid.any():
        bad = data.loc[
            critical_invalid,
            KEYS + prices,
        ].copy()

        bad["reason"] = critical.loc[critical_invalid].apply(
            lambda row: ", ".join(row.index[row]),
            axis=1,
        )

        print(f"\n{dataset}: invalid nonpositive or missing OHLC")
        print(bad.to_string(index=False))
        raise ValueError(f"{dataset} contains unusable OHLC values")

    price_scale = (
        data[prices]
        .abs()
        .max(axis=1)
        .clip(lower=1.0)
    )
    tolerance = price_scale.mul(1e-8).clip(lower=1e-6)

    structural = pd.DataFrame(index=data.index)
    structural["low_above_open_or_close"] = data["low"].gt(
        data[["open", "close"]].min(axis=1) + tolerance
    )
    structural["high_below_open_or_close"] = data["high"].lt(
        data[["open", "close"]].max(axis=1) - tolerance
    )
    structural["low_above_high"] = data["low"].gt(
        data["high"] + tolerance
    )

    invalid = structural.any(axis=1)

    keys = list(
        zip(
            data["ticker"].astype(str),
            data["timestamp"].dt.strftime("%Y-%m-%d"),
        )
    )

    accepted = pd.Series(
        [
            key in ACCEPTED_OHLC_EXCEPTIONS
            for key in keys
        ],
        index=data.index,
    )

    unexpected = invalid & ~accepted

    if unexpected.any():
        bad = data.loc[
            unexpected,
            KEYS + prices + [
                column
                for column in ("volume", "value")
                if column in data.columns
            ],
        ].copy()

        bad["reason"] = structural.loc[unexpected].apply(
            lambda row: ", ".join(row.index[row]),
            axis=1,
        )

        print(f"\n{dataset}: unexplained OHLC anomalies")
        print(bad.to_string(index=False))
        raise ValueError(f"{dataset} contains unexplained OHLC anomalies")

    audit: list[dict[str, object]] = []

    for index in data.index[invalid & accepted]:
        ticker = str(data.at[index, "ticker"])
        timestamp = data.at[index, "timestamp"].strftime("%Y-%m-%d")

        audit.append(
            {
                "dataset": dataset,
                "ticker": ticker,
                "timestamp": timestamp,
                "reason": ACCEPTED_OHLC_EXCEPTIONS[
                    (ticker, timestamp)
                ],
                "violations": [
                    column
                    for column in structural.columns
                    if bool(structural.at[index, column])
                ],
            }
        )

    return audit


def validate_raw_exchange_rules(
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    exchange_history: pd.DataFrame,
) -> list[dict[str, object]]:
    prices = ["open", "high", "low", "close"]

    raw_required = set(
        KEYS + prices + ["volume", "value"]
    )
    adjusted_required = set(KEYS + ["close"])

    missing_raw = sorted(raw_required.difference(raw.columns))
    missing_adjusted = sorted(
        adjusted_required.difference(adjusted.columns)
    )

    if missing_raw:
        raise ValueError(
            "Raw exchange validation is missing columns: "
            f"{missing_raw}"
        )
    if missing_adjusted:
        raise ValueError(
            "Adjusted exchange validation is missing columns: "
            f"{missing_adjusted}"
        )
    if raw.duplicated(KEYS).any():
        raise ValueError(
            "Raw exchange validation contains duplicate keys"
        )
    if adjusted.duplicated(KEYS).any():
        raise ValueError(
            "Adjusted exchange validation contains duplicate keys"
        )

    data = raw[
        KEYS + prices + ["volume", "value"]
    ].merge(
        adjusted[KEYS + ["close"]].rename(
            columns={"close": "adjusted_close"}
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if data["adjusted_close"].isna().any():
        raise ValueError(
            "Raw exchange validation has no adjusted close "
            "for some rows"
        )

    data = attach_exchange_history(
        data,
        exchange_history,
        date_column="timestamp",
    ).sort_values(KEYS).reset_index(drop=True)

    global_dates = pd.Index(
        sorted(data["timestamp"].unique())
    )
    global_positions = pd.Series(
        range(len(global_dates)),
        index=global_dates,
    )

    grouped = data.groupby("ticker", sort=False)
    data["previous_timestamp"] = grouped[
        "timestamp"
    ].shift()
    data["previous_adjusted_close"] = grouped[
        "adjusted_close"
    ].shift()
    data["previous_exchange"] = grouped["exchange"].shift()
    data["adjustment_factor"] = (
        data["adjusted_close"] / data["close"]
    )

    invalid_factor = (
        data["adjustment_factor"].isna()
        | data["adjustment_factor"].le(0)
    )
    if invalid_factor.any():
        examples = data.loc[
            invalid_factor,
            KEYS + ["close", "adjusted_close"],
        ].head(10)
        raise ValueError(
            "Cannot derive positive adjustment factors:\n"
            + examples.to_string(index=False)
        )

    data["previous_raw_close"] = data.groupby(
        "ticker",
        sort=False,
    )["close"].shift()
    data["previous_adjustment_factor"] = data.groupby(
        "ticker",
        sort=False,
    )["adjustment_factor"].shift()
    factor_ratio = (
        data["adjustment_factor"]
        / data["previous_adjustment_factor"]
    )
    factor_changed = (
        data["previous_adjustment_factor"].notna()
        & factor_ratio.sub(1.0).abs().gt(1e-7)
    )

    current_positions = data["timestamp"].map(global_positions)
    previous_positions = data["previous_timestamp"].map(
        global_positions
    )
    missing_sessions = (
        current_positions - previous_positions - 1
    )

    has_previous = data["previous_timestamp"].notna()
    exchange_transfer = (
        has_previous
        & data["exchange"].ne(data["previous_exchange"])
    )
    first_observed_after_start = (
        ~has_previous
        & data["timestamp"].gt(global_dates.min())
    )
    long_resumption = (
        has_previous
        & missing_sessions.ge(25)
        & ~exchange_transfer
    )

    data["session_kind"] = REGULAR_SESSION
    data.loc[
        first_observed_after_start | exchange_transfer,
        "session_kind",
    ] = FIRST_TRADING_DAY
    data.loc[
        long_resumption,
        "session_kind",
    ] = RESUMPTION_25_PLUS

    data["override_reason"] = pd.NA
    data["override_source"] = pd.NA

    for (
        ticker,
        date_text,
    ), override in EXCHANGE_SESSION_OVERRIDES.items():
        mask = (
            data["ticker"].eq(ticker)
            & data["timestamp"].eq(
                pd.Timestamp(date_text)
            )
        )

        if not mask.any():
            continue

        actual_exchange = str(
            data.loc[mask, "exchange"].iloc[0]
        )
        expected_exchange = str(override["exchange"])

        if actual_exchange != expected_exchange:
            raise ValueError(
                f"{ticker} {date_text} is assigned to "
                f"{actual_exchange}, expected "
                f"{expected_exchange}"
            )

        data.loc[
            mask,
            "session_kind",
        ] = override["session_kind"]
        data.loc[
            mask,
            "override_reason",
        ] = override["reason"]
        data.loc[
            mask,
            "override_source",
        ] = override["source"]

    data["reference_price"] = pd.NA
    data["reference_source"] = "no_previous_reference"
    upcom = data["exchange"].eq("UPCOM")
    ordinary_non_upcom = (
        has_previous
        & data["session_kind"].eq(REGULAR_SESSION)
        & ~upcom
        & ~factor_changed
    )
    data.loc[
        ordinary_non_upcom,
        "reference_price",
    ] = data.loc[ordinary_non_upcom, "previous_raw_close"]
    data.loc[
        ordinary_non_upcom,
        "reference_source",
    ] = "previous_close"

    special_session = data["session_kind"].ne(
        REGULAR_SESSION
    )
    data.loc[
        special_session,
        "reference_price",
    ] = pd.NA
    data.loc[
        special_session,
        "reference_source",
    ] = "official_reference_unavailable"

    upcom_reference_unavailable = (
        has_previous
        & data["session_kind"].eq(REGULAR_SESSION)
        & upcom
    )
    data.loc[
        upcom_reference_unavailable,
        "reference_source",
    ] = "historical_eod_reference_unavailable"

    corporate_action_reference_unavailable = (
        has_previous
        & data["session_kind"].eq(REGULAR_SESSION)
        & ~upcom
        & factor_changed
    )
    data.loc[
        corporate_action_reference_unavailable,
        "reference_source",
    ] = "corporate_action_reference_unavailable"

    for (
        ticker,
        date_text,
    ), override in EXCHANGE_SESSION_OVERRIDES.items():
        mask = (
            data["ticker"].eq(ticker)
            & data["timestamp"].eq(
                pd.Timestamp(date_text)
            )
        )

        if not mask.any():
            continue

        data.loc[
            mask,
            "reference_price",
        ] = override["reference_price"]
        data.loc[
            mask,
            "reference_source",
        ] = "verified_override"

    invalid_reference = (
        data["reference_price"].notna()
        & data["reference_price"].le(0)
    )
    if invalid_reference.any():
        examples = data.loc[
            invalid_reference,
            KEYS
            + [
                "previous_adjusted_close",
                "adjusted_close",
                "volume",
                "value",
                "exchange",
            ],
        ].head(10)
        raise ValueError(
            "Exchange validation contains nonpositive "
            "reference prices:\n"
            + examples.to_string(index=False)
        )

    violations: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    upcom_unchecked_rows = int(
        upcom_reference_unavailable.sum()
    )
    if upcom_unchecked_rows:
        audit.append(
            {
                "ticker": "*",
                "timestamp": "",
                "exchange": "UPCOM",
                "session_kind": "MULTIPLE",
                "reference_price": None,
                "reference_source": (
                    "historical_eod_reference_unavailable"
                ),
                "expected_floor": None,
                "expected_ceiling": None,
                "violations": [],
                "status": "validation_scope",
                "rows": upcom_unchecked_rows,
                "reason": (
                    "FiinQuant historical EOD data does not "
                    "provide the official UPCOM reference or "
                    "matched-order turnover needed to rebuild it"
                ),
            }
        )

    for row in data.itertuples(index=False):
        ticker = str(row.ticker)
        timestamp = pd.Timestamp(row.timestamp)
        date_text = timestamp.strftime("%Y-%m-%d")
        exception_key = (ticker, date_text)
        limit = price_limit_fraction(
            row.exchange,
            row.session_kind,
        )
        reference = (
            None
            if pd.isna(row.reference_price)
            else float(row.reference_price)
        )
        expected_floor: float | None = None
        expected_ceiling: float | None = None

        if reference is not None:
            raw_floor = reference * (1.0 - limit)
            raw_ceiling = reference * (1.0 + limit)
            floor_tick = tick_size_vnd(
                row.exchange,
                raw_floor,
                timestamp,
            )
            ceiling_tick = tick_size_vnd(
                row.exchange,
                raw_ceiling,
                timestamp,
            )
            expected_floor = float(
                math.ceil(
                    (raw_floor - 1e-8) / floor_tick
                )
                * floor_tick
            )
            expected_ceiling = float(
                math.floor(
                    (raw_ceiling + 1e-8)
                    / ceiling_tick
                )
                * ceiling_tick
            )

        row_violations: list[str] = []

        for field in prices:
            value = float(getattr(row, field))
            close_is_upcom_average = (
                row.exchange == "UPCOM"
                and field == "close"
            )

            if not close_is_upcom_average:
                tick = tick_size_vnd(
                    row.exchange,
                    value,
                    timestamp,
                )
                tick_units = value / tick

                if (
                    abs(
                        tick_units - round(tick_units)
                    )
                    > 1e-7
                ):
                    row_violations.append(
                        f"{field}_off_tick"
                    )

            if expected_floor is not None:
                if value < expected_floor - 1e-7:
                    row_violations.append(
                        f"{field}_below_limit"
                    )
                if value > expected_ceiling + 1e-7:
                    row_violations.append(
                        f"{field}_above_limit"
                    )

        if row_violations:
            record = {
                "ticker": ticker,
                "timestamp": date_text,
                "exchange": row.exchange,
                "session_kind": row.session_kind,
                "reference_price": reference,
                "reference_source": row.reference_source,
                "expected_floor": expected_floor,
                "expected_ceiling": expected_ceiling,
                "violations": row_violations,
            }

            if exception_key in ACCEPTED_OHLC_EXCEPTIONS:
                record["status"] = "accepted_exception"
                record["reason"] = ACCEPTED_OHLC_EXCEPTIONS[
                    exception_key
                ]
                audit.append(record)
            else:
                violations.append(record)
        elif (
            row.session_kind != REGULAR_SESSION
            or pd.notna(row.override_reason)
            or row.reference_source
            == "corporate_action_reference_unavailable"
        ):
            record = {
                "ticker": ticker,
                "timestamp": date_text,
                "exchange": row.exchange,
                "session_kind": row.session_kind,
                "reference_price": reference,
                "reference_source": row.reference_source,
                "expected_floor": expected_floor,
                "expected_ceiling": expected_ceiling,
                "violations": [],
                "status": (
                    "verified_special_session"
                    if reference is not None
                    else (
                        "corporate_action_reference_unavailable"
                        if row.reference_source
                        == "corporate_action_reference_unavailable"
                        else "special_session_reference_unavailable"
                    )
                ),
            }

            if pd.notna(row.override_reason):
                record["reason"] = str(
                    row.override_reason
                )
                record["source"] = str(
                    row.override_source
                )

            audit.append(record)

    if violations:
        print("\nraw: unexplained exchange-rule anomalies")
        print(pd.DataFrame(violations).to_string(index=False))
        raise ValueError(
            "raw contains unexplained exchange-rule anomalies"
        )

    return audit


def fetch_history(
    client,
    tickers: list[str],
    fields: list[str],
    *,
    adjusted: bool,
    from_date: str,
    to_date: str,
    batch_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]

        print(
            f"Fetching {'adjusted' if adjusted else 'raw'} "
            f"batch {offset // batch_size + 1}: "
            f"{len(batch)} tickers"
        )

        result = client.Fetch_Trading_Data(
            realtime=False,
            tickers=batch,
            fields=fields,
            adjusted=adjusted,
            by="1d",
            from_date=from_date,
            to_date=to_date,
            lasted=False,
        ).get_data()

        frame = (
            result
            if isinstance(result, pd.DataFrame)
            else pd.DataFrame(result)
        )

        if frame.empty:
            raise RuntimeError(
                f"FiinQuant returned no data for batch: {batch}"
            )

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def write_canonical_outputs(
    raw_market: pd.DataFrame,
    adjusted_market: pd.DataFrame,
    foreign_flow: pd.DataFrame,
) -> None:
    outputs = (
        (raw_market, RAW_PATH),
        (adjusted_market, ADJUSTED_PATH),
        (foreign_flow, FOREIGN_PATH),
    )
    temporary_paths = [
        path.with_suffix(path.suffix + ".tmp")
        for _, path in outputs
    ]

    try:
        for (data, _), temporary in zip(
            outputs,
            temporary_paths,
            strict=True,
        ):
            data.to_csv(
                temporary,
                index=False,
                date_format="%Y-%m-%d",
            )

        load_canonical_market_history(
            temporary_paths[0],
            temporary_paths[1],
        )
        load_canonical_foreign_flow_csv(
            temporary_paths[2]
        )

        for temporary, (_, path) in zip(
            temporary_paths,
            outputs,
            strict=True,
        ):
            os.replace(temporary, path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def assert_tickers(
    data: pd.DataFrame,
    expected: set[str],
    label: str,
) -> None:
    actual = set(data["ticker"].dropna().unique())

    if actual != expected:
        raise ValueError(
            f"{label} ticker mismatch. "
            f"Missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def load_raw_archive(
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """
    Load the FLC/ROS raw (unadjusted) OHLC archive.

    The vendor extract never carries a total traded-value (VND) figure
    alongside these raw prices. That gap is preserved as a missing "value"
    rather than derived from price * volume; src.market_data_schema treats
    it as a known, disclosed exception for these two tickers only. Shared
    by the live canonical rebuild and by standalone archive backfills that
    do not otherwise touch the live FiinQuant feed.
    """
    raw_archive = pd.read_csv(
        RAW_ARCHIVE_PATH,
        dtype={"ticker": "string"},
    )
    raw_archive = clean_vendor_frame(
        raw_archive,
        set(KEYS + ADJUSTED_FIELDS + ["volume"]),
        "FLC/ROS raw archive",
    )
    raw_archive = raw_archive.loc[
        raw_archive["timestamp"].between(
            pd.Timestamp(from_date),
            pd.Timestamp(to_date),
        )
    ].copy()
    assert_tickers(
        raw_archive,
        ARCHIVE_TICKERS,
        "FLC/ROS raw archive",
    )
    raw_archive["value"] = pd.NA

    return raw_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default=START_DATE)
    parser.add_argument("--to-date", default=END_DATE)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    all_tickers = set(historical_tickers())
    api_tickers = sorted(all_tickers - ARCHIVE_TICKERS)

    username, password = get_credentials()
    client = create_fiinquant_client(
        username,
        password,
    )

    raw = fetch_history(
        client,
        api_tickers,
        RAW_FIELDS,
        adjusted=False,
        from_date=args.from_date,
        to_date=args.to_date,
        batch_size=args.batch_size,
    )
    raw = clean_vendor_frame(
        raw,
        set(KEYS + RAW_FIELDS),
        "Raw FiinQuant history",
    )

    raw, raw_corrections = apply_vendor_price_corrections(
        raw,
        "raw",
    )

    adjusted = fetch_history(
        client,
        api_tickers,
        ADJUSTED_FIELDS,
        adjusted=True,
        from_date=args.from_date,
        to_date=args.to_date,
        batch_size=args.batch_size,
    )
    adjusted = clean_vendor_frame(
        adjusted,
        set(KEYS + ADJUSTED_FIELDS),
        "Adjusted FiinQuant history",
        reference=raw,
    )

    adjusted, adjusted_corrections = apply_vendor_price_corrections(
        adjusted,
        "adjusted",
    )

    adjusted, adjusted_ohlc_repair = repair_adjusted_ohlc_from_raw(
        raw,
        adjusted,
    )

    raw_ohlc_exceptions = validate_ohlc(raw, "raw")
    adjusted_ohlc_exceptions = validate_ohlc(
        adjusted,
        "adjusted",
    )
    exchange_history = pd.read_csv(
        EXCHANGE_HISTORY_PATH,
        dtype=str,
        keep_default_na=False,
    )
    raw_exchange_rule_audit = validate_raw_exchange_rules(
        raw,
        adjusted,
        exchange_history,
    )

    raw_market = raw[
        KEYS
        + [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "value",
        ]
    ].copy()

    adjusted_market = adjusted[
        KEYS + ADJUSTED_FIELDS
    ].merge(
        raw[KEYS + ["volume", "value"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if adjusted_market["_merge"].ne("both").any():
        raise ValueError(
            "Raw and adjusted FiinQuant dates do not match"
        )

    adjusted_market = adjusted_market.drop(columns="_merge")

    foreign_flow = raw[
        KEYS + ["fb", "fs", "fn"]
    ].copy()

    archive = pd.read_csv(
        ARCHIVE_PATH,
        dtype={"ticker": "string"},
    )
    archive = clean_vendor_frame(
        archive,
        set(
            KEYS
            + ADJUSTED_FIELDS
            + ["volume", "fb", "fs", "fn"]
        ),
        "FLC/ROS archive",
    )

    archive = archive.loc[
        archive["timestamp"].between(
            pd.Timestamp(args.from_date),
            pd.Timestamp(args.to_date),
        )
    ].copy()

    assert_tickers(
        archive,
        ARCHIVE_TICKERS,
        "FLC/ROS archive",
    )

    archive_market = archive[
        KEYS
        + ADJUSTED_FIELDS
        + ["volume"]
    ].copy()
    archive_market["value"] = pd.NA

    adjusted_market = pd.concat(
        [adjusted_market, archive_market],
        ignore_index=True,
    ).sort_values(KEYS).reset_index(drop=True)

    foreign_flow = pd.concat(
        [
            foreign_flow,
            archive[KEYS + ["fb", "fs", "fn"]],
        ],
        ignore_index=True,
    ).sort_values(KEYS).reset_index(drop=True)

    raw_archive = load_raw_archive(args.from_date, args.to_date)

    raw_market = pd.concat(
        [raw_market, raw_archive],
        ignore_index=True,
    ).sort_values(KEYS).reset_index(drop=True)

    if adjusted_market.duplicated(KEYS).any():
        raise ValueError(
            "Canonical adjusted history contains duplicate keys"
        )

    if foreign_flow.duplicated(KEYS).any():
        raise ValueError(
            "Canonical foreign flow contains duplicate keys"
        )

    if raw_market.duplicated(KEYS).any():
        raise ValueError(
            "Canonical raw history contains duplicate keys"
        )

    assert_tickers(
        raw_market,
        set(api_tickers) | ARCHIVE_TICKERS,
        "Raw market history",
    )
    assert_tickers(
        adjusted_market,
        all_tickers,
        "Adjusted market history",
    )
    assert_tickers(
        foreign_flow,
        all_tickers,
        "Foreign-flow history",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_canonical_outputs(
        raw_market,
        adjusted_market,
        foreign_flow,
    )

    metadata = {
        "retrieved_at_vietnam": datetime.now(
            ZoneInfo("Asia/Ho_Chi_Minh")
        ).isoformat(),
        "from_date": args.from_date,
        "to_date": args.to_date,
        "source": "FiinQuant",
        "raw_tickers": raw_market["ticker"].nunique(),
        "adjusted_tickers": adjusted_market["ticker"].nunique(),
        "foreign_flow_tickers": foreign_flow["ticker"].nunique(),
        "raw_rows": len(raw_market),
        "adjusted_rows": len(adjusted_market),
        "foreign_flow_rows": len(foreign_flow),
        "raw_history_unavailable": [],
        "raw_traded_value_unavailable": sorted(ARCHIVE_TICKERS),
        "vendor_price_corrections": (
            raw_corrections + adjusted_corrections
        ),
        "adjusted_ohlc_repair": adjusted_ohlc_repair,
        "accepted_ohlc_exceptions": (
            raw_ohlc_exceptions + adjusted_ohlc_exceptions
        ),
        "raw_exchange_rule_audit": raw_exchange_rule_audit,
        "known_vendor_duplicate_policies": list(
            KNOWN_VENDOR_DUPLICATE_POLICIES
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nCANONICAL FIINQUANT HISTORY COMPLETED")
    print(json.dumps(metadata, indent=2))
    print("Raw output:", RAW_PATH)
    print("Adjusted output:", ADJUSTED_PATH)
    print("Foreign-flow output:", FOREIGN_PATH)


if __name__ == "__main__":
    main()
