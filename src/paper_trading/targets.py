from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

import pandas as pd


ZERO = Decimal("0")
TOLERANCE = Decimal("1e-18")


@dataclass(frozen=True)
class TargetConstructionResult:
    signal_date: pd.Timestamp
    model_name: str
    horizon_days: int
    target_weights: pd.DataFrame
    capacity_replacements: tuple[tuple[str, str], ...]

    @property
    def invested_weight(self) -> Decimal:
        return sum(self.target_weights["target_weight"], start=ZERO)


def _to_decimal(value: Decimal | int | float | str) -> Decimal:
    if isinstance(value, Decimal):
        return value

    return Decimal(str(value))


def _validate_limit(
    value: Decimal | int | float | str,
    name: str,
    allow_zero: bool = False,
) -> Decimal:
    normalized = _to_decimal(value)
    lower_bound_valid = normalized >= ZERO if allow_zero else normalized > ZERO

    if not lower_bound_valid or normalized > Decimal("1"):
        boundary = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {boundary}")

    return normalized


def _prepare_ranked_predictions(
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    required_prediction_columns = {
        "date",
        "ticker",
        "model_name",
        "horizon_days",
        "score",
        "predicted_rank",
    }
    missing_prediction_columns = sorted(
        required_prediction_columns.difference(predictions.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            f"Predictions are missing columns: {missing_prediction_columns}"
        )

    required_universe_columns = {"ticker", "issuer_group"}
    missing_universe_columns = sorted(
        required_universe_columns.difference(universe.columns)
    )

    if missing_universe_columns:
        raise ValueError(
            f"Universe is missing columns: {missing_universe_columns}"
        )

    ranked = predictions.copy()
    ranked["date"] = pd.to_datetime(ranked["date"], errors="raise").dt.normalize()
    ranked["ticker"] = ranked["ticker"].astype(str).str.strip().str.upper()
    ranked["score"] = pd.to_numeric(ranked["score"], errors="raise")
    ranked["predicted_rank"] = pd.to_numeric(
        ranked["predicted_rank"],
        errors="raise",
        downcast="integer",
    )

    if ranked.empty:
        raise ValueError("Predictions cannot be empty")

    if ranked["ticker"].duplicated().any():
        raise ValueError("Predictions contain duplicate tickers")

    for column in ("date", "model_name", "horizon_days"):
        if ranked[column].nunique() != 1:
            raise ValueError(f"Predictions must contain exactly one {column}")

    ranked = ranked.sort_values(
        ["score", "ticker"],
        ascending=[False, True],
    ).reset_index(drop=True)
    expected_ranks = list(range(1, len(ranked) + 1))

    if ranked["predicted_rank"].tolist() != expected_ranks:
        raise ValueError("Prediction ranks do not match score and ticker ordering")

    metadata = universe[["ticker", "issuer_group"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.strip().str.upper()
    metadata["issuer_group"] = metadata["issuer_group"].astype(str).str.strip()

    if metadata["ticker"].duplicated().any():
        raise ValueError("Universe contains duplicate tickers")

    ranked = ranked.merge(
        metadata,
        on="ticker",
        how="left",
        validate="one_to_one",
    )

    missing_groups = ranked.loc[
        ranked["issuer_group"].isna() | ranked["issuer_group"].eq(""),
        "ticker",
    ].tolist()

    if missing_groups:
        raise ValueError(f"Missing issuer groups for tickers: {missing_groups}")

    return ranked


def _selection_capacity(
    selected: pd.DataFrame,
    max_single_name_weight: Decimal,
    max_issuer_group_weight: Decimal,
) -> Decimal:
    group_counts = selected.groupby("issuer_group")["ticker"].count()

    return sum(
        (
            min(
                _to_decimal(count) * max_single_name_weight,
                max_issuer_group_weight,
            )
            for count in group_counts
        ),
        start=ZERO,
    )


def _select_capacity_feasible_names(
    ranked: pd.DataFrame,
    target_holdings: int,
    target_invested_weight: Decimal,
    max_single_name_weight: Decimal,
    max_issuer_group_weight: Decimal,
) -> tuple[pd.DataFrame, tuple[tuple[str, str], ...]]:
    if target_holdings > len(ranked):
        raise ValueError(
            f"Requested {target_holdings} holdings from only {len(ranked)} predictions"
        )

    selected = ranked.head(target_holdings).copy()
    replacements: list[tuple[str, str]] = []

    while (
        _selection_capacity(
            selected,
            max_single_name_weight,
            max_issuer_group_weight,
        )
        + TOLERANCE
        < target_invested_weight
    ):
        selected_tickers = set(selected["ticker"])
        current_capacity = _selection_capacity(
            selected,
            max_single_name_weight,
            max_issuer_group_weight,
        )
        replacement_found = False

        for candidate in ranked.loc[~ranked["ticker"].isin(selected_tickers)].itertuples(
            index=False
        ):
            for removed in selected.sort_values(
                "predicted_rank",
                ascending=False,
            ).itertuples(index=False):
                trial = selected.loc[selected["ticker"] != removed.ticker].copy()
                candidate_row = ranked.loc[ranked["ticker"] == candidate.ticker]
                trial = pd.concat([trial, candidate_row], ignore_index=True)
                trial_capacity = _selection_capacity(
                    trial,
                    max_single_name_weight,
                    max_issuer_group_weight,
                )

                if trial_capacity <= current_capacity + TOLERANCE:
                    continue

                selected = trial
                replacements.append((removed.ticker, candidate.ticker))
                replacement_found = True
                break

            if replacement_found:
                break

        if not replacement_found:
            raise ValueError(
                "Cannot construct the requested invested weight under the "
                "single-name and issuer-group caps"
            )

    return (
        selected.sort_values("predicted_rank").reset_index(drop=True),
        tuple(replacements),
    )


def _allocate_capped_weights(
    selected: pd.DataFrame,
    target_invested_weight: Decimal,
    max_single_name_weight: Decimal,
    max_issuer_group_weight: Decimal,
) -> dict[str, Decimal]:
    groups = selected.set_index("ticker")["issuer_group"].to_dict()
    weights = {ticker: ZERO for ticker in selected["ticker"]}
    remaining = target_invested_weight

    while remaining > TOLERANCE:
        group_weights = {
            group: sum(
                (
                    weights[ticker]
                    for ticker, ticker_group in groups.items()
                    if ticker_group == group
                ),
                start=ZERO,
            )
            for group in set(groups.values())
        }
        active = [
            ticker
            for ticker in weights
            if weights[ticker] + TOLERANCE < max_single_name_weight
            and group_weights[groups[ticker]] + TOLERANCE
            < max_issuer_group_weight
        ]

        if not active:
            raise ValueError("Target-weight capacity was exhausted before allocation")

        equal_increment = remaining / len(active)
        active_group_counts = {
            group: sum(groups[ticker] == group for ticker in active)
            for group in set(groups[ticker] for ticker in active)
        }
        increments: dict[str, Decimal] = {}

        for ticker in active:
            group = groups[ticker]
            single_capacity = max_single_name_weight - weights[ticker]
            group_capacity_per_name = (
                max_issuer_group_weight - group_weights[group]
            ) / active_group_counts[group]
            increments[ticker] = min(
                equal_increment,
                single_capacity,
                group_capacity_per_name,
            )

        allocated = sum(increments.values(), start=ZERO)

        if allocated > remaining:
            last_ticker = active[-1]
            increments[last_ticker] -= allocated - remaining
            allocated = remaining

        if allocated <= TOLERANCE:
            raise ValueError("Target-weight allocation made no progress")

        for ticker, increment in increments.items():
            weights[ticker] += increment

        remaining -= allocated

    residual = target_invested_weight - sum(weights.values(), start=ZERO)

    if residual > ZERO:
        for ticker in weights:
            group = groups[ticker]
            group_weight = sum(
                (
                    weight
                    for member, weight in weights.items()
                    if groups[member] == group
                ),
                start=ZERO,
            )
            capacity = min(
                max_single_name_weight - weights[ticker],
                max_issuer_group_weight - group_weight,
            )
            increment = min(residual, capacity)
            weights[ticker] += increment
            residual -= increment

            if residual <= ZERO:
                break

    if abs(target_invested_weight - sum(weights.values(), start=ZERO)) > TOLERANCE:
        raise ValueError("Allocated target weights do not match invested-weight target")

    return weights


def build_constrained_target_weights(
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
    target_holdings: int,
    target_invested_weight: Decimal | int | float | str,
    max_single_name_weight: Decimal | int | float | str,
    max_issuer_group_weight: Decimal | int | float | str,
) -> TargetConstructionResult:
    if target_holdings <= 0:
        raise ValueError("target_holdings must be positive")

    invested_weight = _validate_limit(
        target_invested_weight,
        "target_invested_weight",
    )
    single_name_cap = _validate_limit(
        max_single_name_weight,
        "max_single_name_weight",
    )
    issuer_group_cap = _validate_limit(
        max_issuer_group_weight,
        "max_issuer_group_weight",
    )

    if target_holdings * single_name_cap < invested_weight:
        raise ValueError("Single-name capacity cannot reach target invested weight")

    ranked = _prepare_ranked_predictions(predictions, universe)
    selected, replacements = _select_capacity_feasible_names(
        ranked=ranked,
        target_holdings=target_holdings,
        target_invested_weight=invested_weight,
        max_single_name_weight=single_name_cap,
        max_issuer_group_weight=issuer_group_cap,
    )
    allocated_weights = _allocate_capped_weights(
        selected=selected,
        target_invested_weight=invested_weight,
        max_single_name_weight=single_name_cap,
        max_issuer_group_weight=issuer_group_cap,
    )
    targets = selected[
        [
            "date",
            "ticker",
            "issuer_group",
            "model_name",
            "horizon_days",
            "score",
            "predicted_rank",
        ]
    ].copy()
    targets["target_weight"] = targets["ticker"].map(allocated_weights)

    return TargetConstructionResult(
        signal_date=targets["date"].iloc[0],
        model_name=str(targets["model_name"].iloc[0]),
        horizon_days=int(targets["horizon_days"].iloc[0]),
        target_weights=targets,
        capacity_replacements=replacements,
    )
