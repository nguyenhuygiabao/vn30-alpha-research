from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MEMBER_MODELS: tuple[str, ...] = (
    "gradient_boosting",
    "random_forest",
)
RANK_ENSEMBLE_MODEL_NAME = "rank_ensemble"
ROLLING_RANK_ENSEMBLE_MODEL_NAME = "rolling_rank_ensemble"
REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "model_name",
    "predicted_return",
    "actual_return",
}
REQUIRED_MARKET_COLUMNS = {"date", "ticker", "adjusted_close"}
DEFAULT_REGIME_POLICY = {
    "trend_up": "random_forest",
    "high_volatility": "random_forest",
    "trend_down": "cash",
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

    actual_bounds = members.groupby(["date", "ticker"])["actual_return"].agg(
        ["min", "max"]
    )
    actuals_match = np.isclose(
        actual_bounds["min"],
        actual_bounds["max"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    if not actuals_match.all():
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


def build_rolling_rank_ensemble_history(
    predictions: pd.DataFrame,
    member_models: tuple[str, ...] = DEFAULT_MEMBER_MODELS,
    performance_window: int = 126,
    label_availability_lag: int = 10,
) -> pd.DataFrame:
    """Blend member ranks using only previously available out-of-sample Rank IC.

    A label from a signal date is not used until ``label_availability_lag`` later
    observed signal dates.  This makes the changing ensemble weights safe for a
    forecast horizon whose realized returns arrive after the signal date.
    """
    if performance_window <= 0 or label_availability_lag < 0:
        raise ValueError("performance_window must be positive and lag non-negative")

    static_ensemble = build_rank_ensemble_history(predictions, member_models)
    members = predictions.loc[predictions["model_name"].isin(member_models)].copy()
    members["date"] = pd.to_datetime(members["date"], errors="raise")
    scores = members.pivot(
        index=["date", "ticker"],
        columns="model_name",
        values="predicted_return",
    )
    ranks = scores.groupby(level="date").rank(ascending=False, method="first")
    rank_scores = 1.0 - ranks.div(
        ranks.groupby(level="date").transform("count") + 1.0
    )
    daily_ic = (
        members.groupby(["date", "model_name"])
        .apply(
            lambda daily: daily["predicted_return"].corr(
                daily["actual_return"], method="spearman"
            )
            if daily["actual_return"].nunique() >= 2
            else np.nan,
            include_groups=False,
        )
        .rename("rank_ic")
        .unstack("model_name")
        .reindex(columns=list(member_models))
    )
    dates = list(daily_ic.index.sort_values())
    actual = static_ensemble.set_index(["date", "ticker"])["actual_return"]
    output_frames: list[pd.DataFrame] = []

    for position, market_date in enumerate(dates):
        available_end = max(position - label_availability_lag, 0)
        available_dates = dates[max(0, available_end - performance_window):available_end]
        if available_dates:
            performance = daily_ic.loc[available_dates].mean(skipna=True).clip(lower=0.0)
        else:
            performance = pd.Series(0.0, index=list(member_models))
        if performance.sum() <= 0.0:
            weights = pd.Series(1.0 / len(member_models), index=list(member_models))
        else:
            weights = performance / performance.sum()

        date_scores = rank_scores.xs(market_date, level="date").copy()
        blended = date_scores.mul(weights, axis="columns").sum(axis=1)
        frame = pd.DataFrame(
            {
                "date": market_date,
                "ticker": blended.index,
                "predicted_return": blended.to_numpy(),
                "actual_return": actual.reindex(
                    pd.MultiIndex.from_product([[market_date], blended.index])
                ).to_numpy(),
                "model_name": ROLLING_RANK_ENSEMBLE_MODEL_NAME,
            }
        )
        for model_name, weight in weights.items():
            frame[f"weight_{model_name}"] = weight
        output_frames.append(frame)

    return pd.concat(output_frames, ignore_index=True)


def summarize_model_candidates(
    predictions: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    ensemble = build_rank_ensemble_history(predictions)
    rolling_ensemble = build_rolling_rank_ensemble_history(predictions)
    combined = pd.concat(
        [predictions.copy(), ensemble, rolling_ensemble], ignore_index=True
    )
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


def build_daily_candidate_metrics(
    predictions: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    ensemble = build_rank_ensemble_history(predictions)
    rolling_ensemble = build_rolling_rank_ensemble_history(predictions)
    combined = pd.concat(
        [predictions.copy(), ensemble, rolling_ensemble], ignore_index=True
    )
    rows: list[dict[str, object]] = []

    for (model_name, market_date), daily in combined.groupby(["model_name", "date"]):
        if daily["actual_return"].nunique() < 2:
            continue
        rows.append(
            {
                "date": market_date,
                "model_name": model_name,
                "rank_ic": daily["predicted_return"].corr(
                    daily["actual_return"], method="spearman"
                ),
                "top_n_actual_return": daily.nlargest(
                    top_n, "predicted_return"
                )["actual_return"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(["date", "model_name"]).reset_index(drop=True)


def paired_block_bootstrap_mean_difference(
    differences: pd.Series | np.ndarray,
    block_size: int = 10,
    bootstrap_samples: int = 2_000,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("At least two finite paired differences are required")
    if block_size <= 0 or bootstrap_samples <= 0:
        raise ValueError("block_size and bootstrap_samples must be positive")

    generator = np.random.default_rng(random_seed)
    sample_means = np.empty(bootstrap_samples)
    block_size = min(block_size, len(values))
    starts = np.arange(len(values) - block_size + 1)
    blocks_needed = int(np.ceil(len(values) / block_size))

    for index in range(bootstrap_samples):
        chosen = generator.choice(starts, size=blocks_needed, replace=True)
        resampled = np.concatenate([values[start : start + block_size] for start in chosen])
        sample_means[index] = resampled[: len(values)].mean()

    lower, upper = np.quantile(sample_means, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def summarize_paired_candidate_stability(
    predictions: pd.DataFrame,
    challenger_model: str = RANK_ENSEMBLE_MODEL_NAME,
    top_n: int = 8,
    rolling_window: int = 126,
) -> pd.DataFrame:
    if rolling_window <= 1:
        raise ValueError("rolling_window must be greater than one")

    daily = build_daily_candidate_metrics(predictions, top_n=top_n)
    challenger = daily.loc[daily["model_name"] == challenger_model].set_index("date")
    rows: list[dict[str, object]] = []

    for model_name in sorted(set(daily["model_name"]) - {challenger_model}):
        baseline = daily.loc[daily["model_name"] == model_name].set_index("date")
        paired = challenger.join(baseline, how="inner", lsuffix="_challenger", rsuffix="_baseline")
        if paired.empty:
            raise ValueError(f"No matched dates for {challenger_model} and {model_name}")

        rank_ic_difference = paired["rank_ic_challenger"] - paired["rank_ic_baseline"]
        mean_diff, lower, upper = paired_block_bootstrap_mean_difference(rank_ic_difference)
        rolling = rank_ic_difference.rolling(rolling_window, min_periods=rolling_window).mean()
        rows.append(
            {
                "challenger_model": challenger_model,
                "baseline_model": model_name,
                "paired_dates": len(paired),
                "mean_rank_ic_difference": mean_diff,
                "bootstrap_95pct_lower": lower,
                "bootstrap_95pct_upper": upper,
                "rolling_window": rolling_window,
                "positive_rolling_windows": int((rolling > 0).sum()),
                "available_rolling_windows": int(rolling.notna().sum()),
            }
        )

    return pd.DataFrame(rows)


def build_historical_market_regimes(
    market_data: pd.DataFrame,
    return_window: int = 20,
    volatility_baseline_window: int = 126,
) -> pd.DataFrame:
    """Classify each date using only the market history available that day.

    The proxy is the equal-weight return of the stocks present in the supplied
    universe.  High-volatility dates take precedence; remaining dates are split
    by the sign of their trailing return.  The volatility threshold is a rolling
    median, so it never uses future observations.
    """
    missing = sorted(REQUIRED_MARKET_COLUMNS.difference(market_data.columns))
    if missing:
        raise ValueError(f"Market data are missing columns: {missing}")
    if return_window <= 1 or volatility_baseline_window <= 1:
        raise ValueError("Regime windows must be greater than one")

    prices = market_data[["date", "ticker", "adjusted_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices = prices.sort_values(["ticker", "date"])
    prices["stock_return"] = prices.groupby("ticker")["adjusted_close"].pct_change()
    market_returns = (
        prices.groupby("date", as_index=False)["stock_return"]
        .mean()
        .rename(columns={"stock_return": "market_return"})
    )
    market_returns["trailing_return"] = (
        (1.0 + market_returns["market_return"])
        .rolling(return_window, min_periods=return_window)
        .apply(np.prod, raw=True)
        - 1.0
    )
    market_returns["trailing_volatility"] = market_returns["market_return"].rolling(
        return_window,
        min_periods=return_window,
    ).std()
    market_returns["volatility_baseline"] = market_returns[
        "trailing_volatility"
    ].rolling(
        volatility_baseline_window,
        min_periods=volatility_baseline_window,
    ).median()
    ready = market_returns.dropna(
        subset=["trailing_return", "trailing_volatility", "volatility_baseline"]
    ).copy()
    ready["market_regime"] = np.where(
        ready["trailing_volatility"] > ready["volatility_baseline"],
        "high_volatility",
        np.where(ready["trailing_return"] >= 0.0, "trend_up", "trend_down"),
    )
    return ready[
        [
            "date",
            "market_regime",
            "trailing_return",
            "trailing_volatility",
            "volatility_baseline",
        ]
    ].reset_index(drop=True)


def summarize_candidates_by_market_regime(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    top_n: int = 8,
    return_window: int = 20,
    volatility_baseline_window: int = 126,
) -> pd.DataFrame:
    """Summarize historical candidates separately across observable regimes."""
    daily = build_daily_candidate_metrics(predictions, top_n=top_n)
    regimes = build_historical_market_regimes(
        market_data,
        return_window=return_window,
        volatility_baseline_window=volatility_baseline_window,
    )
    combined = daily.merge(regimes[["date", "market_regime"]], on="date", how="inner")
    if combined.empty:
        raise ValueError("No prediction dates overlap the available market regimes")

    summary = (
        combined.groupby(["market_regime", "model_name"], as_index=False)
        .agg(
            evaluated_dates=("date", "nunique"),
            average_rank_ic=("rank_ic", "mean"),
            average_top_n_actual_return=("top_n_actual_return", "mean"),
        )
        .sort_values(["market_regime", "average_rank_ic"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def evaluate_regime_policy(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    policy: dict[str, str] | None = None,
    top_n: int = 8,
    return_window: int = 20,
    volatility_baseline_window: int = 126,
) -> pd.DataFrame:
    """Evaluate a fixed, observable-regime model-selection policy historically.

    This is a diagnostic on overlapping forward-return labels, not a live trading
    backtest.  A ``cash`` selection contributes a zero top-N return and has no
    rank IC because no stocks are selected.
    """
    selected_models = dict(DEFAULT_REGIME_POLICY if policy is None else policy)
    expected_regimes = {"trend_up", "trend_down", "high_volatility"}
    missing_regimes = sorted(expected_regimes.difference(selected_models))
    if missing_regimes:
        raise ValueError(f"Regime policy is missing selections: {missing_regimes}")

    daily = build_daily_candidate_metrics(predictions, top_n=top_n)
    regimes = build_historical_market_regimes(
        market_data,
        return_window=return_window,
        volatility_baseline_window=volatility_baseline_window,
    )
    available = set(daily["model_name"])
    invalid_models = sorted(
        {
            model_name
            for model_name in selected_models.values()
            if model_name != "cash" and model_name not in available
        }
    )
    if invalid_models:
        raise ValueError(f"Regime policy selects unavailable models: {invalid_models}")

    prediction_dates = daily[["date"]].drop_duplicates()
    selected = regimes[["date", "market_regime"]].merge(
        prediction_dates,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    if selected.empty:
        raise ValueError("No prediction dates overlap the available market regimes")
    selected["selected_model"] = selected["market_regime"].map(selected_models)
    model_metrics = daily.rename(columns={"model_name": "selected_model"})
    invested = selected.loc[selected["selected_model"] != "cash"].merge(
        model_metrics,
        on=["date", "selected_model"],
        how="left",
        validate="one_to_one",
    )
    if invested[["rank_ic", "top_n_actual_return"]].isna().any().any():
        raise ValueError("Policy selections lack matched daily model metrics")

    cash = selected.loc[selected["selected_model"] == "cash"].assign(
        rank_ic=np.nan,
        top_n_actual_return=0.0,
    )
    combined = pd.concat([invested, cash], ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


def summarize_regime_policy(policy_history: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive diagnostics for a regime-policy history."""
    required = {"date", "selected_model", "top_n_actual_return"}
    missing = sorted(required.difference(policy_history.columns))
    if missing:
        raise ValueError(f"Policy history is missing columns: {missing}")
    if policy_history.empty:
        raise ValueError("Policy history is empty")

    returns = policy_history["top_n_actual_return"]
    return pd.DataFrame(
        [
            {
                "evaluated_dates": policy_history["date"].nunique(),
                "invested_dates": int((policy_history["selected_model"] != "cash").sum()),
                "cash_dates": int((policy_history["selected_model"] == "cash").sum()),
                "average_top_n_actual_return": returns.mean(),
                "top_n_return_volatility": returns.std(),
                "cumulative_top_n_actual_return": (1.0 + returns).prod() - 1.0,
            }
        ]
    )
