from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Iterable

import pandas as pd
from scipy.optimize import linprog


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

    required_universe_columns = {"ticker", "issuer_group", "sector"}
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

    metadata = universe[["ticker", "issuer_group", "sector"]].copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.strip().str.upper()
    metadata["issuer_group"] = metadata["issuer_group"].astype(str).str.strip()
    metadata["sector"] = metadata["sector"].astype(str).str.strip()

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

    missing_sectors = ranked.loc[
        ranked["sector"].isna() | ranked["sector"].eq(""), "ticker"
    ].tolist()

    if missing_sectors:
        raise ValueError(f"Missing sectors for tickers: {missing_sectors}")

    return ranked


def _selection_capacity(
    selected: pd.DataFrame,
    max_single_name_weight: Decimal,
    max_issuer_group_weight: Decimal,
    max_sector_weight: Decimal,
) -> Decimal:
    tickers = selected["ticker"].tolist()
    groups = selected.set_index("ticker")["issuer_group"].to_dict()
    sectors = selected.set_index("ticker")["sector"].to_dict()
    inequalities: list[list[float]] = []
    limits: list[float] = []

    for group in sorted(set(groups.values())):
        inequalities.append(
            [float(groups[ticker] == group) for ticker in tickers]
        )
        limits.append(float(max_issuer_group_weight))

    for sector in sorted(set(sectors.values())):
        inequalities.append(
            [float(sectors[ticker] == sector) for ticker in tickers]
        )
        limits.append(float(max_sector_weight))

    solution = linprog(
        c=[-1.0] * len(tickers),
        A_ub=inequalities,
        b_ub=limits,
        bounds=[(0.0, float(max_single_name_weight))] * len(tickers),
        method="highs",
    )

    if not solution.success:
        raise ValueError("Unable to calculate joint target-weight capacity")

    return Decimal(str(round(-float(solution.fun), 10)))

def _select_capacity_feasible_names(
    ranked: pd.DataFrame,
    target_holdings: int,
    target_invested_weight: Decimal,
    max_single_name_weight: Decimal,
    max_issuer_group_weight: Decimal,
    max_sector_weight: Decimal,
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
            max_sector_weight,
        )
        + TOLERANCE
        < target_invested_weight
    ):
        selected_tickers = set(selected["ticker"])
        current_capacity = _selection_capacity(
            selected,
            max_single_name_weight,
            max_issuer_group_weight,
            max_sector_weight,
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
                    max_sector_weight,
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
                "single-name, issuer-group, and sector caps"
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
    max_sector_weight: Decimal,
) -> dict[str, Decimal]:
    tickers = selected["ticker"].tolist()
    groups = selected.set_index("ticker")["issuer_group"].to_dict()
    sectors = selected.set_index("ticker")["sector"].to_dict()
    count = len(tickers)
    objective = [0.0] * (count + 1)
    objective[-1] = -1.0
    inequalities: list[list[float]] = []
    limits: list[float] = []

    for group in sorted(set(groups.values())):
        inequalities.append(
            [float(groups[ticker] == group) for ticker in tickers] + [0.0]
        )
        limits.append(float(max_issuer_group_weight))

    for sector in sorted(set(sectors.values())):
        inequalities.append(
            [float(sectors[ticker] == sector) for ticker in tickers] + [0.0]
        )
        limits.append(float(max_sector_weight))

    for index in range(count):
        row = [0.0] * (count + 1)
        row[index] = -1.0
        row[-1] = 1.0
        inequalities.append(row)
        limits.append(0.0)

    solution = linprog(
        c=objective,
        A_ub=inequalities,
        b_ub=limits,
        A_eq=[[1.0] * count + [0.0]],
        b_eq=[float(target_invested_weight)],
        bounds=[(0.0, float(max_single_name_weight))] * count
        + [(0.0, float(max_single_name_weight))],
        method="highs",
    )

    if not solution.success:
        raise ValueError("Target-weight capacity was exhausted before allocation")

    quantum = Decimal("1e-12")
    weights = {
        ticker: Decimal(str(max(0.0, float(solution.x[index])))).quantize(
            quantum,
            rounding=ROUND_FLOOR,
        )
        for index, ticker in enumerate(tickers)
    }
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
            sector = sectors[ticker]
            sector_weight = sum(
                (
                    weight
                    for member, weight in weights.items()
                    if sectors[member] == sector
                ),
                start=ZERO,
            )
            capacity = min(
                max_single_name_weight - weights[ticker],
                max_issuer_group_weight - group_weight,
                max_sector_weight - sector_weight,
            )
            increment = min(residual, capacity)
            weights[ticker] += increment
            residual -= increment

            if residual <= ZERO:
                break

    if residual < ZERO:
        for ticker in sorted(weights, key=weights.get, reverse=True):
            reduction = min(-residual, weights[ticker])
            weights[ticker] -= reduction
            residual += reduction

            if residual >= ZERO:
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
    max_sector_weight: Decimal | int | float | str,
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
    sector_cap = _validate_limit(max_sector_weight, "max_sector_weight")

    if target_holdings * single_name_cap < invested_weight:
        raise ValueError("Single-name capacity cannot reach target invested weight")

    ranked = _prepare_ranked_predictions(predictions, universe)
    selected, replacements = _select_capacity_feasible_names(
        ranked=ranked,
        target_holdings=target_holdings,
        target_invested_weight=invested_weight,
        max_single_name_weight=single_name_cap,
        max_issuer_group_weight=issuer_group_cap,
        max_sector_weight=sector_cap,
    )
    allocated_weights = _allocate_capped_weights(
        selected=selected,
        target_invested_weight=invested_weight,
        max_single_name_weight=single_name_cap,
        max_issuer_group_weight=issuer_group_cap,
        max_sector_weight=sector_cap,
    )
    targets = selected[
        [
            "date",
            "ticker",
            "issuer_group",
            "sector",
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
