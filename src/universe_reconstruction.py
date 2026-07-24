from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.universe_history import snapshots_to_membership_history


INITIAL_COLUMNS = {"effective_date", "ticker"}
CALENDAR_COLUMNS = {"effective_date"}
CHANGE_COLUMNS = {"effective_date", "action", "ticker"}
SOURCE_COLUMNS = {
    "effective_date",
    "publication_date",
    "source_url",
    "verified",
}
VALID_CHANGE_ACTIONS = {"add", "remove"}


@dataclass(frozen=True)
class ReconstructionResult:
    """Validated outputs from historical constituent reconstruction."""

    snapshots: pd.DataFrame
    membership_history: pd.DataFrame
    ticker_pool: pd.DataFrame
    audit: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    frame_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            f"{frame_name} is missing columns: {missing}"
        )


def _normalize_dates(
    values: pd.Series,
    field_name: str,
) -> pd.Series:
    try:
        return pd.to_datetime(
            values,
            errors="raise",
        ).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} contains invalid dates"
        ) from exc


def _normalize_tickers(
    values: pd.Series,
    frame_name: str,
) -> pd.Series:
    normalized = (
        values.astype("string")
        .str.strip()
        .str.upper()
    )

    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError(f"{frame_name} contains blank tickers")

    return normalized


def _normalize_initial_snapshot(
    initial_snapshot: pd.DataFrame,
    expected_size: int,
) -> pd.DataFrame:
    _require_columns(
        initial_snapshot,
        INITIAL_COLUMNS,
        "Initial snapshot",
    )

    normalized = initial_snapshot.loc[
        :,
        ["effective_date", "ticker"],
    ].copy()

    normalized["effective_date"] = _normalize_dates(
        normalized["effective_date"],
        "Initial snapshot effective_date",
    )
    normalized["ticker"] = _normalize_tickers(
        normalized["ticker"],
        "Initial snapshot",
    )

    if normalized.empty:
        raise ValueError("Initial snapshot is empty")

    if normalized["effective_date"].nunique() != 1:
        raise ValueError(
            "Initial snapshot must contain exactly one effective date"
        )

    if normalized["ticker"].duplicated().any():
        raise ValueError(
            "Initial snapshot contains duplicate tickers"
        )

    if len(normalized) != expected_size:
        raise ValueError(
            "Initial snapshot must contain exactly "
            f"{expected_size} unique tickers"
        )

    return normalized.sort_values(
        "ticker",
        kind="stable",
    ).reset_index(drop=True)


def _normalize_review_calendar(
    review_calendar: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        review_calendar,
        CALENDAR_COLUMNS,
        "Review calendar",
    )

    normalized = review_calendar.loc[
        :,
        ["effective_date"],
    ].copy()

    normalized["effective_date"] = _normalize_dates(
        normalized["effective_date"],
        "Review calendar effective_date",
    )

    if normalized.empty:
        raise ValueError("Review calendar is empty")

    if normalized["effective_date"].duplicated().any():
        raise ValueError(
            "Review calendar contains duplicate effective dates"
        )

    if not normalized["effective_date"].is_monotonic_increasing:
        raise ValueError(
            "Review calendar effective dates must be strictly increasing"
        )

    return normalized.reset_index(drop=True)


def _parse_verified(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "verified",
    }


def _normalize_source_manifest(
    source_manifest: pd.DataFrame,
    calendar_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    _require_columns(
        source_manifest,
        SOURCE_COLUMNS,
        "Source manifest",
    )

    normalized = source_manifest.loc[
        :,
        [
            "effective_date",
            "publication_date",
            "source_url",
            "verified",
        ],
    ].copy()

    normalized["effective_date"] = _normalize_dates(
        normalized["effective_date"],
        "Source manifest effective_date",
    )
    normalized["publication_date"] = _normalize_dates(
        normalized["publication_date"],
        "Source manifest publication_date",
    )
    normalized["source_url"] = (
        normalized["source_url"]
        .astype("string")
        .str.strip()
    )
    normalized["verified"] = normalized["verified"].map(
        _parse_verified
    )

    if normalized["effective_date"].duplicated().any():
        raise ValueError(
            "Source manifest contains duplicate effective dates"
        )

    if (
        normalized["source_url"].isna().any()
        or normalized["source_url"].eq("").any()
    ):
        raise ValueError("Source manifest contains blank source URLs")

    source_dates = set(normalized["effective_date"])
    missing_dates = sorted(calendar_dates.difference(source_dates))
    extra_dates = sorted(source_dates.difference(calendar_dates))

    if missing_dates:
        formatted = [
            date.date().isoformat()
            for date in missing_dates
        ]
        raise ValueError(
            f"Source manifest is missing review dates: {formatted}"
        )

    if extra_dates:
        formatted = [
            date.date().isoformat()
            for date in extra_dates
        ]
        raise ValueError(
            f"Source manifest contains unknown review dates: {formatted}"
        )

    unverified = normalized.loc[
        ~normalized["verified"],
        "effective_date",
    ]

    if not unverified.empty:
        formatted = [
            date.date().isoformat()
            for date in unverified
        ]
        raise ValueError(
            f"Source manifest contains unverified dates: {formatted}"
        )

    return normalized.sort_values(
        "effective_date",
        kind="stable",
    ).reset_index(drop=True)


def _normalize_membership_changes(
    membership_changes: pd.DataFrame,
    calendar_dates: set[pd.Timestamp],
    initial_date: pd.Timestamp,
) -> pd.DataFrame:
    _require_columns(
        membership_changes,
        CHANGE_COLUMNS,
        "Membership changes",
    )

    normalized = membership_changes.loc[
        :,
        ["effective_date", "action", "ticker"],
    ].copy()

    if normalized.empty:
        normalized["effective_date"] = pd.to_datetime(
            normalized["effective_date"]
        )
        return normalized

    normalized["effective_date"] = _normalize_dates(
        normalized["effective_date"],
        "Membership changes effective_date",
    )
    normalized["action"] = (
        normalized["action"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    normalized["ticker"] = _normalize_tickers(
        normalized["ticker"],
        "Membership changes",
    )

    invalid_actions = sorted(
        set(normalized["action"]).difference(
            VALID_CHANGE_ACTIONS
        )
    )

    if invalid_actions:
        raise ValueError(
            f"Membership changes contain invalid actions: "
            f"{invalid_actions}"
        )

    change_dates = set(normalized["effective_date"])
    unknown_dates = sorted(change_dates.difference(calendar_dates))

    if unknown_dates:
        formatted = [
            date.date().isoformat()
            for date in unknown_dates
        ]
        raise ValueError(
            f"Membership changes contain unknown review dates: "
            f"{formatted}"
        )

    if initial_date in change_dates:
        raise ValueError(
            "Initial review date cannot also contain membership changes"
        )

    duplicate_rows = normalized.duplicated(
        subset=["effective_date", "action", "ticker"],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Membership changes contain duplicate action rows"
        )

    conflicting = (
        normalized.groupby(
            ["effective_date", "ticker"],
            sort=False,
        )["action"]
        .nunique()
        .gt(1)
    )

    if conflicting.any():
        raise ValueError(
            "A ticker cannot be both added and removed "
            "on the same effective date"
        )

    return normalized.sort_values(
        ["effective_date", "action", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def reconstruct_constituent_history(
    initial_snapshot: pd.DataFrame,
    review_calendar: pd.DataFrame,
    membership_changes: pd.DataFrame,
    source_manifest: pd.DataFrame,
    expected_size: int = 30,
) -> ReconstructionResult:
    """
    Reconstruct complete point-in-time constituent snapshots.

    The first review date uses the verified initial basket. Each later
    date applies that date's removals and additions to the prior basket.
    Reviews with no changes still produce a complete snapshot.
    """
    if expected_size <= 0:
        raise ValueError("Expected constituent count must be positive")

    initial = _normalize_initial_snapshot(
        initial_snapshot,
        expected_size,
    )
    calendar = _normalize_review_calendar(review_calendar)

    initial_date = initial["effective_date"].iloc[0]
    first_calendar_date = calendar["effective_date"].iloc[0]

    if initial_date != first_calendar_date:
        raise ValueError(
            "Initial snapshot date must equal the first review date"
        )

    calendar_dates = set(calendar["effective_date"])

    sources = _normalize_source_manifest(
        source_manifest,
        calendar_dates,
    )
    changes = _normalize_membership_changes(
        membership_changes,
        calendar_dates,
        initial_date,
    )

    source_lookup = sources.set_index("effective_date")
    basket = set(initial["ticker"])

    snapshot_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for effective_date in calendar["effective_date"]:
        date_changes = changes.loc[
            changes["effective_date"].eq(effective_date)
        ]

        removals = set(
            date_changes.loc[
                date_changes["action"].eq("remove"),
                "ticker",
            ]
        )
        additions = set(
            date_changes.loc[
                date_changes["action"].eq("add"),
                "ticker",
            ]
        )

        missing_removals = sorted(removals.difference(basket))

        if missing_removals:
            raise ValueError(
                f"Cannot remove non-members on "
                f"{effective_date.date()}: {missing_removals}"
            )

        existing_additions = sorted(additions.intersection(basket))

        if existing_additions:
            raise ValueError(
                f"Cannot add existing members on "
                f"{effective_date.date()}: {existing_additions}"
            )

        next_basket = basket.difference(removals).union(additions)

        if len(next_basket) != expected_size:
            raise ValueError(
                f"Review {effective_date.date()} produces "
                f"{len(next_basket)} constituents instead of "
                f"{expected_size}"
            )

        basket = next_basket
        source = source_lookup.loc[effective_date]

        for ticker in sorted(basket):
            snapshot_rows.append(
                {
                    "effective_date": effective_date,
                    "ticker": ticker,
                    "source_url": source["source_url"],
                }
            )

        audit_rows.append(
            {
                "effective_date": effective_date,
                "publication_date": source["publication_date"],
                "source_url": source["source_url"],
                "verified": bool(source["verified"]),
                "additions": len(additions),
                "removals": len(removals),
                "constituent_count": len(basket),
                "status": "pass",
            }
        )

    snapshots = pd.DataFrame(snapshot_rows).sort_values(
        ["effective_date", "ticker"],
        kind="stable",
    ).reset_index(drop=True)

    membership_history = snapshots_to_membership_history(
        snapshots,
        expected_size=expected_size,
    )

    ticker_pool = pd.DataFrame(
        {
            "ticker": sorted(
                snapshots["ticker"].drop_duplicates()
            )
        }
    )

    audit = pd.DataFrame(audit_rows).sort_values(
        "effective_date",
        kind="stable",
    ).reset_index(drop=True)

    return ReconstructionResult(
        snapshots=snapshots,
        membership_history=membership_history,
        ticker_pool=ticker_pool,
        audit=audit,
    )