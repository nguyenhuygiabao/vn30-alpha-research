from pathlib import Path

import pandas as pd

from src.features_herding import (
    add_herding_features,
    get_extreme_herding_dates,
    plot_herding_index,
)


rows = []

dates = pd.date_range(
    "2024-01-01",
    periods=10,
)

returns = [
    0.01,
    0.02,
    -0.01,
    0.03,
    0.04,
    -0.05,
    -0.04,
    0.06,
    0.01,
    -0.02,
]

for date, daily_return in zip(dates, returns):
    rows.extend(
        [
            {
                "date": date,
                "ticker": "AAA",
                "return_1d": daily_return,
                "value_traded": 100,
                "hit_ceiling_today": daily_return > 0.05,
                "hit_floor_today": daily_return < -0.05,
            },
            {
                "date": date,
                "ticker": "BBB",
                "return_1d": daily_return * 2,
                "value_traded": 300,
                "hit_ceiling_today": daily_return * 2 > 0.05,
                "hit_floor_today": daily_return * 2 < -0.05,
            },
            {
                "date": date,
                "ticker": "CCC",
                "return_1d": -daily_return,
                "value_traded": 600,
                "hit_ceiling_today": -daily_return > 0.05,
                "hit_floor_today": -daily_return < -0.05,
            },
        ]
    )

data = pd.DataFrame(rows)

features = add_herding_features(
    data=data,
    correlation_window=5,
)

extreme_dates = get_extreme_herding_dates(
    data=features,
    top_n=3,
)

plot_path = plot_herding_index(
    data=features,
    output_path="reports/plots/manual_check_herding_index.png",
)

required_columns = [
    "vn30_return_dispersion",
    "percent_stocks_up",
    "percent_stocks_down",
    "percent_hitting_ceiling",
    "percent_hitting_floor",
    "volume_concentration_top5",
    "rolling_avg_pairwise_corr",
    "market_direction_agreement",
    "price_limit_pressure",
    "herding_index",
]

required_columns_exist = all(
    column in features.columns
    for column in required_columns
)

herding_index_between_zero_and_one = features["herding_index"].between(
    0,
    1,
).all()

final_date = features["date"].max()

final_correlation = features.loc[
    features["date"] == final_date,
    "rolling_avg_pairwise_corr",
].iloc[0]

final_correlation_correct = round(
    final_correlation,
    6,
) == -0.333333

plot_exists = Path(plot_path).exists()

print("Latest herding features:")
print(
    features[
        features["date"] == final_date
    ][
        [
            "date",
            "ticker",
            "vn30_return_dispersion",
            "percent_stocks_up",
            "percent_stocks_down",
            "percent_hitting_ceiling",
            "percent_hitting_floor",
            "volume_concentration_top5",
            "rolling_avg_pairwise_corr",
            "herding_index",
        ]
    ].round(6).to_string(index=False)
)

print("\nExtreme herding dates:")
print(
    extreme_dates.round(6).to_string(index=False)
)

print("\nRequired columns exist:", required_columns_exist)
print("Herding index between 0 and 1:", herding_index_between_zero_and_one)
print("Final correlation correct:", final_correlation_correct)
print("Plot exists:", plot_exists)
print("Plot path:", plot_path)
