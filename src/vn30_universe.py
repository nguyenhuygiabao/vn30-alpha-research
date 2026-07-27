from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "config" / "vn30_history"
EXPECTED_CONSTITUENTS = 30


def _read_csv(
    path: Path,
    required_columns: set[str],
) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        dtype="string",
        encoding="utf-8-sig",
    )

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    return data


def _parse_dates(
    data: pd.DataFrame,
    column: str,
    source_name: str,
) -> pd.DataFrame:
    result = data.copy()

    result[column] = pd.to_datetime(
        result[column],
        format="%Y-%m-%d",
        errors="raise",
    ).dt.normalize()

    if result[column].isna().any():
        raise ValueError(
            f"{source_name} contains missing dates"
        )

    return result


def build_constituent_snapshots(
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> pd.DataFrame:
    history_dir = Path(history_dir)

    initial = _read_csv(
        history_dir / "initial_snapshot.csv",
        {"effective_date", "ticker"},
    )
    changes = _read_csv(
        history_dir / "membership_changes.csv",
        {"effective_date", "action", "ticker"},
    )
    calendar = _read_csv(
        history_dir / "review_calendar.csv",
        {"effective_date"},
    )

    initial = _parse_dates(
        initial,
        "effective_date",
        "initial_snapshot.csv",
    )
    changes = _parse_dates(
        changes,
        "effective_date",
        "membership_changes.csv",
    )
    calendar = _parse_dates(
        calendar,
        "effective_date",
        "review_calendar.csv",
    )

    initial["ticker"] = (
        initial["ticker"]
        .str.strip()
        .str.upper()
    )
    changes["ticker"] = (
        changes["ticker"]
        .str.strip()
        .str.upper()
    )
    changes["action"] = (
        changes["action"]
        .str.strip()
        .str.lower()
    )

    review_dates = (
        calendar["effective_date"]
        .sort_values()
        .drop_duplicates()
        .tolist()
    )

    if len(review_dates) != len(calendar):
        raise ValueError(
            "Review calendar contains duplicate dates"
        )

    if not review_dates:
        raise ValueError("Review calendar is empty")

    initial_dates = initial["effective_date"].unique()

    if (
        len(initial_dates) != 1
        or initial_dates[0] != review_dates[0]
    ):
        raise ValueError(
            "Initial snapshot must match the first review date"
        )

    if initial["ticker"].duplicated().any():
        raise ValueError(
            "Initial snapshot contains duplicate tickers"
        )

    members = set(initial["ticker"])

    if len(members) != EXPECTED_CONSTITUENTS:
        raise ValueError(
            "Initial snapshot must contain exactly "
            f"{EXPECTED_CONSTITUENTS} tickers"
        )

    allowed_actions = {"add", "remove"}
    unexpected_actions = (
        set(changes["action"]) - allowed_actions
    )

    if unexpected_actions:
        raise ValueError(
            "Unexpected membership actions: "
            f"{sorted(unexpected_actions)}"
        )

    if changes.duplicated(
        ["effective_date", "action", "ticker"]
    ).any():
        raise ValueError(
            "Membership changes contain duplicate actions"
        )

    invalid_dates = (
        set(changes["effective_date"])
        - set(review_dates[1:])
    )

    if invalid_dates:
        raise ValueError(
            "Membership changes contain dates outside "
            "the review calendar"
        )

    records: list[dict[str, object]] = []

    for effective_date in review_dates:
        if effective_date != review_dates[0]:
            period_changes = changes.loc[
                changes["effective_date"].eq(
                    effective_date
                )
            ]

            removals = set(
                period_changes.loc[
                    period_changes["action"].eq("remove"),
                    "ticker",
                ]
            )
            additions = set(
                period_changes.loc[
                    period_changes["action"].eq("add"),
                    "ticker",
                ]
            )

            missing_removals = removals - members
            existing_additions = additions & members

            if missing_removals:
                raise ValueError(
                    f"{effective_date.date()}: cannot remove "
                    f"non-members {sorted(missing_removals)}"
                )

            if existing_additions:
                raise ValueError(
                    f"{effective_date.date()}: cannot add "
                    f"existing members {sorted(existing_additions)}"
                )

            members.difference_update(removals)
            members.update(additions)

        if len(members) != EXPECTED_CONSTITUENTS:
            raise ValueError(
                f"{effective_date.date()}: expected "
                f"{EXPECTED_CONSTITUENTS} constituents, "
                f"found {len(members)}"
            )

        records.extend(
            {
                "effective_date": effective_date,
                "ticker": ticker,
            }
            for ticker in sorted(members)
        )

    return pd.DataFrame.from_records(
        records,
        columns=["effective_date", "ticker"],
    )


def constituents_on_date(
    date: str | pd.Timestamp,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> tuple[str, ...]:
    target_date = pd.Timestamp(date).normalize()
    snapshots = build_constituent_snapshots(history_dir)

    eligible_dates = snapshots.loc[
        snapshots["effective_date"].le(target_date),
        "effective_date",
    ]

    if eligible_dates.empty:
        raise ValueError(
            "Requested date precedes the first VN30 snapshot"
        )

    active_date = eligible_dates.max()

    return tuple(
        snapshots.loc[
            snapshots["effective_date"].eq(active_date),
            "ticker",
        ].sort_values()
    )


def historical_tickers(
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> tuple[str, ...]:
    snapshots = build_constituent_snapshots(history_dir)

    return tuple(
        sorted(snapshots["ticker"].unique())
    )
