from __future__ import annotations

import pandas as pd


EXCHANGE_HISTORY_COLUMNS = (
    "ticker",
    "effective_from",
    "effective_to",
    "exchange",
)

SUPPORTED_EXCHANGES = frozenset(
    {
        "HOSE",
        "HNX",
        "UPCOM",
    }
)


def normalize_exchange_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(
        set(EXCHANGE_HISTORY_COLUMNS).difference(
            history.columns
        )
    )
    if missing:
        raise ValueError(
            f"Exchange history is missing columns: {missing}"
        )

    normalized = history[
        list(EXCHANGE_HISTORY_COLUMNS)
    ].copy()

    normalized["ticker"] = (
        normalized["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    normalized["exchange"] = (
        normalized["exchange"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    normalized["effective_from"] = pd.to_datetime(
        normalized["effective_from"],
        errors="raise",
    ).dt.normalize()

    raw_effective_to = normalized["effective_to"]
    has_effective_to = (
        raw_effective_to.notna()
        & raw_effective_to.astype(str).str.strip().ne("")
    )

    normalized["effective_to"] = pd.NaT
    normalized.loc[
        has_effective_to,
        "effective_to",
    ] = pd.to_datetime(
        raw_effective_to.loc[has_effective_to],
        errors="raise",
    ).dt.normalize()

    if normalized["ticker"].eq("").any():
        raise ValueError(
            "Exchange history contains blank tickers"
        )

    unsupported = sorted(
        set(normalized["exchange"]).difference(
            SUPPORTED_EXCHANGES
        )
    )
    if unsupported:
        raise ValueError(
            f"Exchange history contains unsupported exchanges: "
            f"{unsupported}"
        )

    reversed_ranges = (
        normalized["effective_to"].notna()
        & (
            normalized["effective_to"]
            < normalized["effective_from"]
        )
    )
    if reversed_ranges.any():
        raise ValueError(
            "Exchange history contains reversed date ranges"
        )

    normalized = normalized.sort_values(
        ["ticker", "effective_from"],
        kind="stable",
    ).reset_index(drop=True)

    for ticker, intervals in normalized.groupby(
        "ticker",
        sort=False,
    ):
        previous_end: pd.Timestamp | None = None

        for interval in intervals.itertuples(index=False):
            if (
                previous_end is not None
                and interval.effective_from <= previous_end
            ):
                raise ValueError(
                    f"Exchange history overlaps for {ticker}"
                )

            if pd.isna(interval.effective_to):
                previous_end = pd.Timestamp.max.normalize()
            else:
                previous_end = interval.effective_to

    return normalized


def attach_exchange_history(
    market_data: pd.DataFrame,
    history: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    required = {date_column, "ticker"}
    missing = sorted(
        required.difference(market_data.columns)
    )
    if missing:
        raise ValueError(
            f"Market data are missing columns: {missing}"
        )

    if "exchange" in market_data.columns:
        raise ValueError(
            "Market data already contain an exchange column"
        )

    normalized_history = normalize_exchange_history(
        history
    )

    data = market_data.copy()
    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="raise",
    ).dt.normalize()
    data["ticker"] = (
        data["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    data["_source_row"] = range(len(data))

    joined = data.merge(
        normalized_history,
        on="ticker",
        how="left",
    )

    active = joined.loc[
        (
            joined[date_column]
            >= joined["effective_from"]
        )
        & (
            joined["effective_to"].isna()
            | (
                joined[date_column]
                <= joined["effective_to"]
            )
        )
    ].copy()

    match_counts = active.groupby(
        "_source_row"
    ).size()

    duplicate_rows = match_counts.loc[
        match_counts > 1
    ]
    if not duplicate_rows.empty:
        raise ValueError(
            "A market row matches multiple exchange intervals"
        )

    matched_rows = set(match_counts.index)
    missing_rows = [
        index
        for index in data["_source_row"]
        if index not in matched_rows
    ]

    if missing_rows:
        examples = data.loc[
            data["_source_row"].isin(missing_rows),
            [date_column, "ticker"],
        ].head(10).to_dict("records")

        raise ValueError(
            "No exchange history covers some market rows: "
            f"{examples}"
        )

    original_columns = list(market_data.columns)

    return active.sort_values(
        "_source_row"
    )[
        original_columns + ["exchange"]
    ].reset_index(drop=True)
