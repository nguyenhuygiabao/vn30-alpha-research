from pathlib import Path
import numpy as np
import pandas as pd

source = Path("data/processed/historical_account_replay.csv")
rolling_output = Path("reports/tables/historical_replay_rolling.csv")
regime_output = Path("reports/tables/historical_replay_regimes.csv")
yearly_output = Path("reports/tables/historical_replay_yearly.csv")

data = pd.read_csv(source)
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date").reset_index(drop=True)

numeric = [
    "portfolio_value",
    "benchmark_value",
    "turnover",
    "cash_weight",
    "holdings_count",
    "skipped_trade_count",
    "hhi",
    "effective_positions",
    "max_single_weight",
    "max_issuer_weight",
    "max_sector_weight",
]

for column in numeric:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data["strategy_return"] = (
    data["portfolio_value"].pct_change().fillna(0.0)
)
data["benchmark_return"] = (
    data["benchmark_value"].pct_change().fillna(0.0)
)
data["active_return"] = (
    data["strategy_return"] - data["benchmark_return"]
)

data["strategy_63d_return"] = (
    data["portfolio_value"]
    / data["portfolio_value"].shift(63)
    - 1.0
)
data["benchmark_63d_return"] = (
    data["benchmark_value"]
    / data["benchmark_value"].shift(63)
    - 1.0
)
data["active_63d_return"] = (
    (1.0 + data["strategy_63d_return"])
    / (1.0 + data["benchmark_63d_return"])
    - 1.0
)

strategy_std = data["strategy_return"].rolling(
    63,
    min_periods=40,
).std()

benchmark_std = data["benchmark_return"].rolling(
    63,
    min_periods=40,
).std()

data["strategy_63d_sharpe"] = (
    data["strategy_return"].rolling(63, min_periods=40).mean()
    / strategy_std
    * np.sqrt(252)
)

data["benchmark_63d_sharpe"] = (
    data["benchmark_return"].rolling(63, min_periods=40).mean()
    / benchmark_std
    * np.sqrt(252)
)

data["benchmark_20d_volatility"] = (
    data["benchmark_return"]
    .rolling(20, min_periods=15)
    .std()
    * np.sqrt(252)
)

volatility_cutoff = data[
    "benchmark_20d_volatility"
].median()

data["market_direction"] = np.where(
    data["benchmark_63d_return"].ge(0),
    "Bull",
    "Bear",
)

data["volatility_regime"] = np.where(
    data["benchmark_20d_volatility"].ge(volatility_cutoff),
    "High Volatility",
    "Low Volatility",
)

data["regime"] = (
    data["market_direction"]
    + " / "
    + data["volatility_regime"]
)

valid = data.dropna(
    subset=[
        "benchmark_63d_return",
        "benchmark_20d_volatility",
    ]
).copy()

def annualized_return(series):
    return series.mean() * 252

def sharpe(series):
    std = series.std(ddof=1)
    return (
        series.mean() / std * np.sqrt(252)
        if std > 0
        else np.nan
    )

regime_rows = []

for regime, group in valid.groupby("regime"):
    regime_rows.append(
        {
            "regime": regime,
            "days": len(group),
            "strategy_annualized_return": annualized_return(
                group["strategy_return"]
            ),
            "benchmark_annualized_return": annualized_return(
                group["benchmark_return"]
            ),
            "active_annualized_return": annualized_return(
                group["active_return"]
            ),
            "strategy_sharpe": sharpe(
                group["strategy_return"]
            ),
            "benchmark_sharpe": sharpe(
                group["benchmark_return"]
            ),
            "daily_benchmark_win_rate": (
                group["active_return"].gt(0).mean()
            ),
            "average_cash_weight": group[
                "cash_weight"
            ].mean(),
            "average_hhi": group["hhi"].mean(),
            "average_holdings": group[
                "holdings_count"
            ].mean(),
        }
    )

regimes = pd.DataFrame(regime_rows).sort_values(
    "active_annualized_return",
    ascending=False,
)

yearly_rows = []

for year, group in data.groupby(data["date"].dt.year):
    strategy_return = (
        group["portfolio_value"].iloc[-1]
        / group["portfolio_value"].iloc[0]
        - 1.0
    )
    benchmark_return = (
        group["benchmark_value"].iloc[-1]
        / group["benchmark_value"].iloc[0]
        - 1.0
    )

    yearly_nav = group["portfolio_value"]
    yearly_drawdown = (
        yearly_nav / yearly_nav.cummax() - 1.0
    )

    yearly_rows.append(
        {
            "year": year,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "active_return": (
                strategy_return - benchmark_return
            ),
            "strategy_sharpe": sharpe(
                group["strategy_return"]
            ),
            "max_drawdown": yearly_drawdown.min(),
            "average_cash_weight": group[
                "cash_weight"
            ].mean(),
            "average_holdings": group[
                "holdings_count"
            ].mean(),
        }
    )

yearly = pd.DataFrame(yearly_rows)

rolling_output.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(rolling_output, index=False)
regimes.to_csv(regime_output, index=False)
yearly.to_csv(yearly_output, index=False)

rolling = data.dropna(subset=["active_63d_return"])
trade_days = data[data["turnover"].gt(0)]

print("HISTORICAL REPLAY DIAGNOSTICS")
print("=" * 70)
print(
    "63-day benchmark win rate:",
    f'{rolling["active_63d_return"].gt(0).mean():.2%}',
)
print(
    "Median 63-day active return:",
    f'{rolling["active_63d_return"].median():.2%}',
)
print(
    "Worst 63-day active return:",
    f'{rolling["active_63d_return"].min():.2%}',
)
print(
    "Best 63-day active return:",
    f'{rolling["active_63d_return"].max():.2%}',
)
print(
    "Average cash weight:",
    f'{data["cash_weight"].mean():.2%}',
)
print(
    "Average holdings:",
    f'{data["holdings_count"].mean():.2f}',
)
print(
    "Average trade-day turnover:",
    f'{trade_days["turnover"].mean():.2%}',
)
print(
    "Average HHI:",
    f'{data["hhi"].mean():.4f}',
)
print(
    "Maximum issuer weight:",
    f'{data["max_issuer_weight"].max():.2%}',
)
print(
    "Maximum sector weight:",
    f'{data["max_sector_weight"].max():.2%}',
)
print()
print("REGIME RESULTS")
print(
    regimes.to_string(
        index=False,
        formatters={
            "strategy_annualized_return": "{:.2%}".format,
            "benchmark_annualized_return": "{:.2%}".format,
            "active_annualized_return": "{:.2%}".format,
            "strategy_sharpe": "{:.3f}".format,
            "benchmark_sharpe": "{:.3f}".format,
            "daily_benchmark_win_rate": "{:.2%}".format,
            "average_cash_weight": "{:.2%}".format,
            "average_hhi": "{:.4f}".format,
            "average_holdings": "{:.2f}".format,
        },
    )
)
print()
print("YEARLY RESULTS")
print(
    yearly.to_string(
        index=False,
        formatters={
            "strategy_return": "{:.2%}".format,
            "benchmark_return": "{:.2%}".format,
            "active_return": "{:.2%}".format,
            "strategy_sharpe": "{:.3f}".format,
            "max_drawdown": "{:.2%}".format,
            "average_cash_weight": "{:.2%}".format,
            "average_holdings": "{:.2f}".format,
        },
    )
)
