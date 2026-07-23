from __future__ import annotations

import pandas as pd


MEMBERSHIP_COLUMNS = ("ticker", "effective_from", "effective_to")


def normalize_membership_history(membership: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MEMBERSHIP_COLUMNS).difference(membership.columns))
    if missing:
        raise ValueError(f"Membership history is missing columns: {missing}")

    normalized = membership[list(MEMBERSHIP_COLUMNS)].copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()
    normalized["effective_from"] = pd.to_datetime(
        normalized["effective_from"], errors="raise"
    ).dt.normalize()
    raw_effective_to = normalized["effective_to"]
    has_effective_to = raw_effective_to.notna() & raw_effective_to.astype(
        str
    ).str.strip().ne("")
    normalized["effective_to"] = pd.NaT
    normalized.loc[has_effective_to, "effective_to"] = pd.to_datetime(
        raw_effective_to.loc[has_effective_to], errors="raise"
    ).dt.normalize()
    if normalized["ticker"].eq("").any():
        raise ValueError("Membership history contains blank tickers")
    invalid_ranges = normalized["effective_to"].notna() & (
        normalized["effective_to"] < normalized["effective_from"]
    )
    if invalid_ranges.any():
        raise ValueError("Membership history contains reversed date ranges")

    normalized = normalized.sort_values(["ticker", "effective_from"]).reset_index(
        drop=True
    )
    for ticker, intervals in normalized.groupby("ticker", sort=False):
        previous_end: pd.Timestamp | None = None
        for interval in intervals.itertuples(index=False):
            if previous_end is not None and interval.effective_from <= previous_end:
                raise ValueError(f"Membership history overlaps for {ticker}")
            if pd.isna(interval.effective_to):
                previous_end = pd.Timestamp.max.normalize()
            else:
                previous_end = interval.effective_to
    return normalized


def filter_to_point_in_time_universe(
    market_data: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    required = {"date", "ticker"}
    missing = sorted(required.difference(market_data.columns))
    if missing:
        raise ValueError(f"Market data are missing columns: {missing}")
    normalized_membership = normalize_membership_history(membership)
    data = market_data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data["_source_row"] = range(len(data))
    joined = data.merge(normalized_membership, on="ticker", how="inner")
    active = joined.loc[
        (joined["date"] >= joined["effective_from"])
        & (
            joined["effective_to"].isna()
            | (joined["date"] <= joined["effective_to"])
        )
    ].copy()
    if active["_source_row"].duplicated().any():
        raise ValueError("A market row matches multiple membership intervals")
    original_columns = list(market_data.columns)
    return active.sort_values("_source_row")[original_columns].reset_index(drop=True)


def summarize_membership_coverage(
    membership: pd.DataFrame,
    market_dates: pd.Series | pd.DatetimeIndex,
    expected_constituents: int = 30,
) -> pd.DataFrame:
    if expected_constituents <= 0:
        raise ValueError("Expected constituent count must be positive")
    normalized = normalize_membership_history(membership)
    dates = pd.DatetimeIndex(pd.to_datetime(market_dates)).normalize().unique().sort_values()
    rows = []
    for date in dates:
        active = normalized.loc[
            (normalized["effective_from"] <= date)
            & (normalized["effective_to"].isna() | (normalized["effective_to"] >= date))
        ]
        rows.append(
            {
                "date": date,
                "active_constituents": active["ticker"].nunique(),
                "expected_constituents": expected_constituents,
                "coverage_complete": active["ticker"].nunique()
                == expected_constituents,
            }
        )
    return pd.DataFrame(rows)


UNIVERSE_MODES = ("strict_vn30", "expanding_alumni")


def historical_ticker_pool(membership: pd.DataFrame) -> list[str]:
    """Return every ticker that appears in verified membership history."""
    normalized = normalize_membership_history(membership)
    return sorted(normalized["ticker"].unique().tolist())


def filter_to_expanding_alumni_universe(
    market_data: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Keep each ticker from its first known VN30 entry onward."""
    required = {"date", "ticker"}
    missing = sorted(required.difference(market_data.columns))
    if missing:
        raise ValueError(f"Market data are missing columns: {missing}")

    normalized_membership = normalize_membership_history(membership)
    first_entries = (
        normalized_membership.groupby("ticker", as_index=False)["effective_from"]
        .min()
        .rename(columns={"effective_from": "first_vn30_entry"})
    )

    data = market_data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data["_source_row"] = range(len(data))

    joined = data.merge(first_entries, on="ticker", how="inner")
    eligible = joined.loc[joined["date"] >= joined["first_vn30_entry"]].copy()

    original_columns = list(market_data.columns)
    return eligible.sort_values("_source_row")[original_columns].reset_index(drop=True)


def filter_to_universe(
    market_data: pd.DataFrame,
    membership: pd.DataFrame,
    mode: str = "strict_vn30",
) -> pd.DataFrame:
    """Apply strict VN30 or expanding VN30-alumni eligibility."""
    if mode == "strict_vn30":
        return filter_to_point_in_time_universe(market_data, membership)
    if mode == "expanding_alumni":
        return filter_to_expanding_alumni_universe(market_data, membership)

    raise ValueError(
        f"Unknown universe mode: {mode!r}. Expected one of {UNIVERSE_MODES}"
    )


SNAPSHOT_REQUIRED_COLUMNS = {
    "effective_date",
    "ticker",
    "source_url",
}


def normalize_constituent_snapshots(
    snapshots: pd.DataFrame,
    expected_size: int = 30,
) -> pd.DataFrame:
    """Validate and normalize verified VN30 constituent snapshots."""
    missing = sorted(SNAPSHOT_REQUIRED_COLUMNS.difference(snapshots.columns))
    if missing:
        raise ValueError(f"Constituent snapshots are missing columns: {missing}")

    data = snapshots.copy()

    if data[list(SNAPSHOT_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Constituent snapshots contain missing required values")

    data["effective_date"] = pd.to_datetime(
        data["effective_date"],
        errors="raise",
    ).dt.normalize()
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data["source_url"] = data["source_url"].astype(str).str.strip()

    if data["effective_date"].isna().any():
        raise ValueError("Constituent snapshots contain missing effective dates")
    if data["ticker"].eq("").any():
        raise ValueError("Constituent snapshots contain blank tickers")
    if data["source_url"].eq("").any():
        raise ValueError("Constituent snapshots contain blank source URLs")

    duplicates = data.duplicated(
        subset=["effective_date", "ticker"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            "Constituent snapshots contain duplicate effective-date/ticker rows"
        )

    counts = data.groupby("effective_date")["ticker"].nunique()
    invalid_counts = counts.loc[counts != expected_size]
    if not invalid_counts.empty:
        details = {
            str(date.date()): int(count)
            for date, count in invalid_counts.items()
        }
        raise ValueError(
            f"Each constituent snapshot must contain {expected_size} tickers: "
            f"{details}"
        )

    return data.sort_values(
        ["effective_date", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def validate_constituent_snapshot_coverage(
    snapshots: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    expected_size: int = 30,
    maximum_gap_days: int = 220,
) -> pd.DataFrame:
    """Fail when snapshot history cannot cover the requested backtest period."""
    normalized = normalize_constituent_snapshots(
        snapshots,
        expected_size=expected_size,
    )

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if pd.isna(start) or pd.isna(end):
        raise ValueError("Coverage dates cannot be missing")
    if end < start:
        raise ValueError("Coverage end date cannot precede start date")
    if maximum_gap_days <= 0:
        raise ValueError("maximum_gap_days must be positive")

    snapshot_dates = [
        pd.Timestamp(value)
        for value in normalized["effective_date"].drop_duplicates().sort_values()
    ]
    starting_snapshots = [date for date in snapshot_dates if date <= start]

    if not starting_snapshots:
        raise ValueError(
            "No constituent snapshot is effective on or before the backtest start"
        )

    anchor = starting_snapshots[-1]
    timeline = (
        [anchor]
        + [date for date in snapshot_dates if start < date <= end]
        + [end]
    )

    excessive_gaps = [
        (earlier, later, (later - earlier).days)
        for earlier, later in zip(timeline, timeline[1:])
        if (later - earlier).days > maximum_gap_days
    ]
    if excessive_gaps:
        raise ValueError(
            f"Constituent snapshot coverage contains excessive gaps: "
            f"{excessive_gaps}"
        )

    return normalized


def snapshots_to_membership_history(
    snapshots: pd.DataFrame,
    expected_size: int = 30,
) -> pd.DataFrame:
    """Convert complete snapshots into entry, exit, and re-entry intervals."""
    normalized = normalize_constituent_snapshots(
        snapshots,
        expected_size=expected_size,
    )

    snapshot_dates = [
        pd.Timestamp(value)
        for value in normalized["effective_date"].drop_duplicates().sort_values()
    ]

    active_starts: dict[str, pd.Timestamp] = {}
    records: list[dict[str, object]] = []

    for effective_date in snapshot_dates:
        members = set(
            normalized.loc[
                normalized["effective_date"].eq(effective_date),
                "ticker",
            ]
        )

        for ticker in sorted(set(active_starts).difference(members)):
            records.append(
                {
                    "ticker": ticker,
                    "effective_from": active_starts.pop(ticker),
                    "effective_to": effective_date - pd.Timedelta(days=1),
                }
            )

        for ticker in sorted(members.difference(active_starts)):
            active_starts[ticker] = effective_date

    for ticker, effective_from in sorted(active_starts.items()):
        records.append(
            {
                "ticker": ticker,
                "effective_from": effective_from,
                "effective_to": pd.NaT,
            }
        )

    history = pd.DataFrame.from_records(
        records,
        columns=["ticker", "effective_from", "effective_to"],
    )
    return normalize_membership_history(history)
