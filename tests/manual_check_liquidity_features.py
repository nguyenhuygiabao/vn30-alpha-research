import pandas as pd

from src.features_liquidity import (
    add_liquidity_features,
    rank_stocks_by_liquidity,
)


rows = []

dates = pd.date_range(
    "2024-01-01",
    periods=25,
)

for i, date in enumerate(dates):
    rows.append(
        {
            "date": date,
            "ticker": "AAA",
            "volume": 1_000 + i * 100,
            "value_traded": 1_000_000 + i * 100_000,
            "return_1d": 0.01,
            "shares_outstanding": 1_000_000,
        }
    )

    rows.append(
        {
            "date": date,
            "ticker": "BBB",
            "volume": 5_000 + i * 200,
            "value_traded": 10_000_000 + i * 500_000,
            "return_1d": 0.005,
            "shares_outstanding": 2_000_000,
        }
    )

    rows.append(
        {
            "date": date,
            "ticker": "CCC",
            "volume": 3_000 + i * 150,
            "value_traded": 4_000_000 + i * 250_000,
            "return_1d": 0.02,
            "shares_outstanding": 1_500_000,
        }
    )

data = pd.DataFrame(rows)

features = add_liquidity_features(data)

latest_features = features[
    features["date"] == features["date"].max()
].copy()

ranking = rank_stocks_by_liquidity(features)

required_columns = [
    "volume_z_20",
    "value_traded_z_20",
    "volume_change_5d",
    "turnover_z_20",
    "amihud_illiquidity",
    "abnormal_volume_flag",
]

required_columns_exist = all(
    column in features.columns
    for column in required_columns
)

latest_z_scores_available = latest_features[
    [
        "volume_z_20",
        "value_traded_z_20",
        "turnover_z_20",
    ]
].notna().all().all()

ranking_order_correct = ranking["ticker"].tolist() == [
    "BBB",
    "CCC",
    "AAA",
]

print("Latest liquidity features:")
print(
    latest_features[
        [
            "date",
            "ticker",
            "volume",
            "value_traded",
            "volume_z_20",
            "value_traded_z_20",
            "volume_change_5d",
            "turnover_z_20",
            "amihud_illiquidity",
            "abnormal_volume_flag",
        ]
    ].round(6).to_string(index=False)
)

print("\nLiquidity ranking:")
print(
    ranking[
        [
            "date",
            "ticker",
            "volume",
            "value_traded",
            "amihud_illiquidity",
            "value_traded_rank",
            "volume_rank",
            "amihud_rank",
            "liquidity_rank",
        ]
    ].to_string(index=False)
)

print("\nRequired columns exist:", required_columns_exist)
print("Latest z-scores available:", latest_z_scores_available)
print("Ranking order correct:", ranking_order_correct)
