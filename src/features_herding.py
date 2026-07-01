from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



ROLLING_CORRELATION_WINDOW: int = 20
TOP_TRADED_NAMES_COUNT: int = 5


def calculate_volume_concentration_top_n(
    values: pd.Series,
    top_n: int = TOP_TRADED_NAMES_COUNT,
) -> float:
    total_value_traded = values.sum()

    if total_value_traded <= 0:
        return float("nan")

    top_value_traded = values.nlargest(top_n).sum()

    return top_value_traded / total_value_traded

def calculate_rolling_average_pairwise_correlation(
    data: pd.DataFrame,
    window: int = ROLLING_CORRELATION_WINDOW,
) -> pd.DataFrame:
    returns_by_date = data.pivot(
        index="date",
        columns="ticker",
        values="return_1d",
    ).sort_index()

    correlations = []

    for end_position, date in enumerate(returns_by_date.index):
        if end_position + 1 < window:
            correlations.append(
                {
                    "date": date,
                    "rolling_avg_pairwise_corr": pd.NA,
                }
            )
            continue

        window_returns = returns_by_date.iloc[
            end_position + 1 - window : end_position + 1
        ]

        correlation_matrix = window_returns.corr(
            min_periods=2,
        )

        pairwise_correlations = []

        for first_ticker, second_ticker in combinations(
            correlation_matrix.columns,
            2,
        ):
            correlation = correlation_matrix.loc[
                first_ticker,
                second_ticker,
            ]

            if pd.notna(correlation):
                pairwise_correlations.append(correlation)

        if pairwise_correlations:
            average_correlation = sum(pairwise_correlations) / len(
                pairwise_correlations
            )
        else:
            average_correlation = pd.NA

        correlations.append(
            {
                "date": date,
                "rolling_avg_pairwise_corr": average_correlation,
            }
        )

    return pd.DataFrame(correlations)

def add_cross_sectional_herding_features(
    data: pd.DataFrame,
    top_n: int = TOP_TRADED_NAMES_COUNT,
) -> pd.DataFrame:
    working = data.copy()

    if "hit_ceiling_today" not in working.columns:
        working["hit_ceiling_today"] = False

    if "hit_floor_today" not in working.columns:
        working["hit_floor_today"] = False

    working = working.sort_values(
        [
            "date",
            "ticker",
        ]
    )

    daily_features = working.groupby("date").agg(
        vn30_return_dispersion=("return_1d", lambda values: values.std(ddof=0)),
        percent_stocks_up=("return_1d", lambda values: (values > 0).mean()),
        percent_stocks_down=("return_1d", lambda values: (values < 0).mean()),
        percent_hitting_ceiling=("hit_ceiling_today", "mean"),
        percent_hitting_floor=("hit_floor_today", "mean"),
    )

    volume_concentration = working.groupby("date")["value_traded"].apply(
        lambda values: calculate_volume_concentration_top_n(
            values=values,
            top_n=top_n,
        )
    )

    daily_features["volume_concentration_top5"] = volume_concentration

    daily_features = daily_features.reset_index()

    return working.merge(
        daily_features,
        on="date",
        how="left",
    )


def add_rolling_correlation_feature(
    data: pd.DataFrame,
    window: int = ROLLING_CORRELATION_WINDOW,
) -> pd.DataFrame:
    working = data.copy()

    rolling_correlation = calculate_rolling_average_pairwise_correlation(
        data=working,
        window=window,
    )

    return working.merge(
        rolling_correlation,
        on="date",
        how="left",
    )


def calculate_expanding_percentile_score(
    values: pd.Series,
    higher_is_more_herding: bool = True,
) -> pd.Series:
    scores = []
    observed_values = []

    for value in values:
        if pd.isna(value):
            scores.append(pd.NA)
            continue

        current_value = float(value)
        observed_values.append(current_value)

        if higher_is_more_herding:
            score = sum(
                historical_value <= current_value
                for historical_value in observed_values
            ) / len(observed_values)
        else:
            score = sum(
                historical_value >= current_value
                for historical_value in observed_values
            ) / len(observed_values)

        scores.append(score)

    return pd.Series(
        scores,
        index=values.index,
    )


def add_herding_index(
    data: pd.DataFrame,
) -> pd.DataFrame:
    working = data.copy()

    daily_features = working[
        [
            "date",
            "vn30_return_dispersion",
            "percent_stocks_up",
            "percent_stocks_down",
            "percent_hitting_ceiling",
            "percent_hitting_floor",
            "volume_concentration_top5",
            "rolling_avg_pairwise_corr",
        ]
    ].drop_duplicates().sort_values("date").reset_index(drop=True)

    daily_features["market_direction_agreement"] = daily_features[
        [
            "percent_stocks_up",
            "percent_stocks_down",
        ]
    ].max(axis=1)

    daily_features["price_limit_pressure"] = (
        daily_features["percent_hitting_ceiling"]
        + daily_features["percent_hitting_floor"]
    )

    daily_features["herding_corr_score"] = calculate_expanding_percentile_score(
        daily_features["rolling_avg_pairwise_corr"],
        higher_is_more_herding=True,
    )

    daily_features["herding_low_dispersion_score"] = (
        calculate_expanding_percentile_score(
            daily_features["vn30_return_dispersion"],
            higher_is_more_herding=False,
        )
    )

    daily_features["herding_direction_score"] = calculate_expanding_percentile_score(
        daily_features["market_direction_agreement"],
        higher_is_more_herding=True,
    )

    daily_features["herding_price_limit_score"] = calculate_expanding_percentile_score(
        daily_features["price_limit_pressure"],
        higher_is_more_herding=True,
    )

    daily_features["herding_volume_concentration_score"] = (
        calculate_expanding_percentile_score(
            daily_features["volume_concentration_top5"],
            higher_is_more_herding=True,
        )
    )

    herding_component_columns = [
        "herding_corr_score",
        "herding_low_dispersion_score",
        "herding_direction_score",
        "herding_price_limit_score",
        "herding_volume_concentration_score",
    ]

    daily_features["herding_index"] = daily_features[
        herding_component_columns
    ].mean(
        axis=1,
        skipna=True,
    )

    return working.merge(
        daily_features[
            [
                "date",
                "market_direction_agreement",
                "price_limit_pressure",
                "herding_corr_score",
                "herding_low_dispersion_score",
                "herding_direction_score",
                "herding_price_limit_score",
                "herding_volume_concentration_score",
                "herding_index",
            ]
        ],
        on="date",
        how="left",
    )


def add_herding_features(
    data: pd.DataFrame,
    correlation_window: int = ROLLING_CORRELATION_WINDOW,
    top_n: int = TOP_TRADED_NAMES_COUNT,
) -> pd.DataFrame:
    working = add_cross_sectional_herding_features(
        data=data,
        top_n=top_n,
    )

    working = add_rolling_correlation_feature(
        data=working,
        window=correlation_window,
    )

    working = add_herding_index(working)

    return working


def get_extreme_herding_dates(
    data: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    daily_herding = data[
        [
            "date",
            "herding_index",
            "vn30_return_dispersion",
            "percent_stocks_up",
            "percent_stocks_down",
            "percent_hitting_ceiling",
            "percent_hitting_floor",
            "volume_concentration_top5",
            "rolling_avg_pairwise_corr",
        ]
    ].drop_duplicates().sort_values("date")

    return daily_herding.sort_values(
        [
            "herding_index",
            "date",
        ],
        ascending=[
            False,
            True,
        ],
    ).head(top_n).reset_index(drop=True)


def plot_herding_index(
    data: pd.DataFrame,
    output_path: str = "reports/plots/herding_index.png",
) -> Path:
    daily_herding = data[
        [
            "date",
            "herding_index",
        ]
    ].drop_duplicates().sort_values("date")

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        daily_herding["date"],
        daily_herding["herding_index"],
    )
    plt.title("Herding Index Through Time")
    plt.xlabel("Date")
    plt.ylabel("Herding Index")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path
