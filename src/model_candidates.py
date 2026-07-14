from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MEMBER_MODELS: tuple[str, ...] = (
    "gradient_boosting",
    "random_forest",
)
RANK_ENSEMBLE_MODEL_NAME = "rank_ensemble"
REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "model_name",
    "predicted_return",
    "actual_return",
}


def build_rank_ensemble_history(
    predictions: pd.DataFrame,
    member_models: tuple[str, ...] = DEFAULT_MEMBER_MODELS,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")

    members = predictions.loc[
        predictions["model_name"].isin(member_models)
    ].copy()
    if members.empty:
        raise ValueError("No requested ensemble-member predictions were found")

    counts = members.groupby(["date", "ticker"])["model_name"].nunique()
    incomplete = counts[counts != len(member_models)]
    if not incomplete.empty:
        raise ValueError("Ensemble members lack matched date-ticker coverage")

    actual_counts = members.groupby(["date", "ticker"])["actual_return"].nunique()
    if (actual_counts > 1).any():
        raise ValueError("Ensemble members disagree on realized returns")

    scores = members.pivot(
        index=["date", "ticker"],
        columns="model_name",
        values="predicted_return",
    )
    if set(scores.columns) != set(member_models):
        raise ValueError("Ensemble member prediction columns are incomplete")

    ranks = scores.groupby(level="date").rank(
        ascending=False,
        method="first",
    )
    mean_rank = ranks.mean(axis=1)
    counts_per_date = mean_rank.groupby(level="date").transform("count")
    ensemble_score = 1.0 - mean_rank / (counts_per_date + 1.0)
    actual = members.groupby(["date", "ticker"])["actual_return"].first()

    return pd.DataFrame(
        {
            "predicted_return": ensemble_score,
            "actual_return": actual,
            "model_name": RANK_ENSEMBLE_MODEL_NAME,
        }
    ).reset_index()


def summarize_model_candidates(
    predictions: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    ensemble = build_rank_ensemble_history(predictions)
    combined = pd.concat([predictions.copy(), ensemble], ignore_index=True)
    rows: list[dict[str, object]] = []

    for model_name, model_rows in combined.groupby("model_name"):
        daily_rank_ic = []
        daily_top_returns = []
        for _, daily in model_rows.groupby("date"):
            if daily["actual_return"].nunique() < 2:
                continue
            rank_ic = daily["predicted_return"].corr(
                daily["actual_return"], method="spearman"
            )
            top_returns = daily.nlargest(top_n, "predicted_return")["actual_return"]
            daily_rank_ic.append(rank_ic)
            daily_top_returns.append(top_returns.mean())

        rows.append(
            {
                "model_name": model_name,
                "evaluated_dates": len(daily_rank_ic),
                "average_rank_ic": np.nanmean(daily_rank_ic),
                f"average_top_{top_n}_actual_return": np.nanmean(daily_top_returns),
            }
        )

    return pd.DataFrame(rows).sort_values("average_rank_ic", ascending=False).reset_index(drop=True)
