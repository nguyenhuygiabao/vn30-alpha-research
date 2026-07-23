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
