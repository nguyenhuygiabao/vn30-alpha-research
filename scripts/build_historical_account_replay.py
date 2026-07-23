from __future__ import annotations


import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_ohlcv_csv
from src.paper_trading.broker_state import PaperBrokerState
from src.paper_trading.calendar import TradingCalendar
from src.paper_trading.config import load_paper_trading_config
from src.paper_trading.daily_performance import (
    build_daily_performance_row,
    equal_weight_benchmark_return,
)
from src.paper_trading.execution_rules import ExecutionConstraints, MarketSnapshot
from src.paper_trading.order_sizing import PlannedOrder, TargetWeight, build_order_plan
from src.paper_trading.schemas import OrderAction, Side
from src.paper_trading.settlement import ExecutionRecord, ZERO, to_decimal
from src.paper_trading.targets import build_constrained_target_weights


PERFORMANCE_PATH = ROOT / "data" / "processed" / "historical_account_replay.csv"
TRADES_PATH = ROOT / "reports" / "tables" / "historical_account_replay_trades.csv"
SUMMARY_PATH = ROOT / "reports" / "tables" / "historical_account_replay_summary.csv"


def prepare_market_data(data: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    required = {
        "date", "ticker", "open", "high", "low",
        "close", "volume", "value_traded",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Market data missing columns: {missing}")

    market = data.copy()
    market.columns = [str(column).strip().lower() for column in market.columns]
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market["ticker"] = market["ticker"].astype(str).str.strip().str.upper()

    if market.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate market date/ticker rows")

    numeric = ["open", "high", "low", "close", "volume", "value_traded"]
    for column in numeric:
        market[column] = pd.to_numeric(market[column], errors="raise")

    if (market[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Market prices must be positive")

    market = market.sort_values(["ticker", "date"]).reset_index(drop=True)
    market["previous_close"] = market.groupby("ticker")["close"].shift(1)
    market["average_daily_value_20d"] = market.groupby("ticker")[
        "value_traded"
    ].transform(lambda series: series.rolling(20, min_periods=1).mean())

    market["open_at_ceiling"] = (
        market["previous_close"].notna()
        & market["open"].ge(market["previous_close"] * 1.069)
    )
    market["open_at_floor"] = (
        market["previous_close"].notna()
        & market["open"].le(market["previous_close"] * 0.931)
    )
    market["close_at_ceiling"] = (
        market["previous_close"].notna()
        & market["close"].ge(market["previous_close"] * 1.069)
    )
    market["close_at_floor"] = (
        market["previous_close"].notna()
        & market["close"].le(market["previous_close"] * 0.931)
    )

    for column in (
        "open", "close", "previous_close",
        "average_daily_value_20d", "value_traded",
    ):
        market[f"{column}_vnd"] = market[column] * float(multiplier)

    return market.sort_values(["date", "ticker"]).reset_index(drop=True)


def prepare_predictions(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "ticker", "predicted_return",
        "model_name", "forecast_horizon_days",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Prediction data missing columns: {missing}")

    predictions = data.copy()
    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="raise"
    ).dt.normalize()
    predictions["ticker"] = (
        predictions["ticker"].astype(str).str.strip().str.upper()
    )
    predictions["predicted_return"] = pd.to_numeric(
        predictions["predicted_return"], errors="raise"
    )
    predictions = predictions.loc[
        predictions["forecast_horizon_days"].eq(10)
        & predictions["model_name"].eq("gradient_boosting")
    ].copy()

    if predictions.empty:
        raise ValueError("No 10-day gradient-boosting predictions found")
    if predictions.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate prediction date/ticker rows")

    return predictions.sort_values(["date", "ticker"]).reset_index(drop=True)


def price_map(rows: pd.DataFrame, column: str) -> dict[str, Decimal]:
    return {
        row.ticker: to_decimal(getattr(row, column))
        for row in rows.itertuples(index=False)
    }


def portfolio_value(
    broker: PaperBrokerState,
    prices: Mapping[str, Decimal],
) -> Decimal:
    market_value = sum(
        prices.get(ticker, ZERO) * position.economic_quantity
        for ticker, position in broker.positions.items()
    )
    return broker.settled_cash + broker.unsettled_cash + market_value


def snapshot_map(rows: pd.DataFrame) -> dict[str, MarketSnapshot]:
    return {
        row.ticker: MarketSnapshot(
            ticker=row.ticker,
            data_date=pd.Timestamp(row.date).date(),
            price=row.close_vnd,
            average_daily_value=row.average_daily_value_20d_vnd,
            at_ceiling=bool(row.close_at_ceiling),
            at_floor=bool(row.close_at_floor),
        )
        for row in rows.itertuples(index=False)
    }


def prediction_frame(predictions: pd.DataFrame, signal_date: date) -> pd.DataFrame:
    frame = predictions.loc[
        predictions["date"].dt.date.eq(signal_date),
        ["date", "ticker", "predicted_return"],
    ].copy()

    if len(frame) < 8:
        raise ValueError(
            f"Only {len(frame)} predictions available on {signal_date}"
        )

    frame = frame.sort_values(
        ["predicted_return", "ticker"],
        ascending=[False, True],
    ).reset_index(drop=True)
    frame["model_name"] = "gradient_boosting"
    frame["horizon_days"] = 10
    frame["score"] = frame["predicted_return"]
    frame["predicted_rank"] = range(1, len(frame) + 1)

    return frame[
        [
            "date", "ticker", "model_name",
            "horizon_days", "score", "predicted_rank",
        ]
    ]




def target_objects(weights: pd.DataFrame, signal_date: date) -> list[TargetWeight]:
    return [
        TargetWeight(
            ticker=row.ticker,
            issuer_group=row.issuer_group,
            sector=row.sector,
            target_weight=row.target_weight,
            signal_id=(
                f"historical-{signal_date:%Y%m%d}-"
                f"gradient_boosting-10d-{row.ticker}"
            ),
            predicted_rank=int(row.predicted_rank),
        )
        for row in weights.itertuples(index=False)
    ]


def execution_record(
    order: PlannedOrder,
    execution_date: date,
    price: Decimal,
    quantity: int,
    constraints: ExecutionConstraints,
) -> ExecutionRecord:
    gross = price * quantity
    commission = gross * constraints.commission_rate
    slippage = gross * constraints.slippage_rate
    tax = (
        gross * constraints.sell_tax_rate
        if order.action == OrderAction.SELL
        else ZERO
    )
    return ExecutionRecord(
        execution_id=f"historical-execution-{order.order_id}",
        order_id=order.order_id,
        execution_date=execution_date,
        ticker=order.ticker,
        side=Side(order.action.value),
        filled_quantity=quantity,
        execution_price=price,
        gross_value=gross,
        commission=commission,
        slippage=slippage,
        tax=tax,
    )


def execute_orders(
    broker: PaperBrokerState,
    orders: Iterable[PlannedOrder],
    execution_date: date,
    market_rows: pd.DataFrame,
    valuation_rows: pd.DataFrame,
    constraints: ExecutionConstraints,
    calendar: TradingCalendar,
    issuer_groups: Mapping[str, str],
    settlement_lag: int,
) -> tuple[list[dict[str, object]], int]:
    by_ticker = {
        row.ticker: row for row in market_rows.itertuples(index=False)
    }
    open_prices = price_map(valuation_rows, "close_vnd")
    open_prices.update(price_map(market_rows, "open_vnd"))
    ordered = sorted(
        list(orders),
        key=lambda order: (
            0 if order.action == OrderAction.SELL else 1,
            order.order_id,
        ),
    )

    rows: list[dict[str, object]] = []
    skips = 0

    for order in ordered:
        market = by_ticker.get(order.ticker)
        quantity = int(order.requested_quantity)
        reason = ""
        price = ZERO

        if market is None:
            quantity = 0
            reason = "MISSING_MARKET_DATA"
        else:
            price = to_decimal(market.open_vnd)
            if order.action == OrderAction.BUY and bool(market.open_at_ceiling):
                quantity = 0
                reason = "PRICE_CEILING_BUY_BLOCK"
            elif order.action == OrderAction.SELL and bool(market.open_at_floor):
                quantity = 0
                reason = "PRICE_FLOOR_SELL_BLOCK"

        if quantity > 0 and order.action == OrderAction.SELL:
            quantity = constraints.round_trade_quantity(
                min(quantity, broker.sellable_quantity(order.ticker))
            )
            if quantity <= 0:
                reason = "INSUFFICIENT_SELLABLE_QUANTITY"

        if quantity > 0 and order.action == OrderAction.BUY:
            opening_value = portfolio_value(broker, open_prices)
            minimum_cash = opening_value * constraints.cash_buffer_weight
            available_cash = max(broker.buying_power - minimum_cash, ZERO)
            affordable = constraints.affordable_buy_quantity(
                available_cash, price
            )
            quantity = constraints.round_trade_quantity(
                min(quantity, affordable)
            )
            if quantity <= 0:
                reason = "INSUFFICIENT_SETTLED_CASH"

        if quantity > 0 and price * quantity < constraints.min_trade_value_vnd:
            quantity = 0
            reason = "BELOW_MIN_TRADE_VALUE"

        if quantity > 0:
            execution = execution_record(
                order, execution_date, price, quantity, constraints
            )
            broker.apply_execution(
                execution,
                calendar,
                issuer_group=issuer_groups.get(order.ticker, ""),
                settlement_lag_trading_days=settlement_lag,
            )
            gross = execution.gross_value
            commission = execution.commission
            slippage = execution.slippage
            tax = execution.tax
        else:
            skips += 1
            gross = commission = slippage = tax = ZERO

        rows.append(
            {
                "signal_date": order.order_date.isoformat(),
                "execution_date": execution_date.isoformat(),
                "order_id": order.order_id,
                "ticker": order.ticker,
                "side": order.action.value,
                "requested_quantity": int(order.requested_quantity),
                "filled_quantity": quantity,
                "execution_price": str(price),
                "gross_value": str(gross),
                "commission": str(commission),
                "slippage": str(slippage),
                "tax": str(tax),
                "reason_code": reason,
            }
        )

    return rows, skips


def risk_metrics(
    broker: PaperBrokerState,
    prices: Mapping[str, Decimal],
    issuer_groups: Mapping[str, str],
    sectors: Mapping[str, str],
) -> dict[str, float]:
    value = portfolio_value(broker, prices)
    if value <= ZERO:
        return {
            "hhi": 0.0,
            "effective_positions": 0.0,
            "max_single_weight": 0.0,
            "max_issuer_weight": 0.0,
            "max_sector_weight": 0.0,
        }

    weights = {
        ticker: float(
            prices.get(ticker, ZERO)
            * position.economic_quantity
            / value
        )
        for ticker, position in broker.positions.items()
        if position.economic_quantity > 0
        and prices.get(ticker, ZERO) > ZERO
    }
    hhi = sum(weight * weight for weight in weights.values())

    issuer_weights: dict[str, float] = {}
    sector_weights: dict[str, float] = {}
    for ticker, weight in weights.items():
        issuer = issuer_groups.get(ticker, ticker)
        sector = sectors.get(ticker, issuer)
        issuer_weights[issuer] = issuer_weights.get(issuer, 0.0) + weight
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    return {
        "hhi": hhi,
        "effective_positions": 1.0 / hhi if hhi > 0 else 0.0,
        "max_single_weight": max(weights.values(), default=0.0),
        "max_issuer_weight": max(issuer_weights.values(), default=0.0),
        "max_sector_weight": max(sector_weights.values(), default=0.0),
    }


def summarize(performance: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    data = performance.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    numeric_columns = [
        "portfolio_value", "daily_return", "benchmark_value",
        "benchmark_return", "drawdown", "turnover", "cash_weight",
        "hhi", "effective_positions", "max_single_weight",
        "max_issuer_weight", "max_sector_weight",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    initial = data["portfolio_value"].iloc[0]
    final = data["portfolio_value"].iloc[-1]
    total_return = final / initial - 1.0
    years = max(
        (data["date"].iloc[-1] - data["date"].iloc[0]).days / 365.25,
        1.0 / 365.25,
    )
    cagr = (final / initial) ** (1.0 / years) - 1.0

    returns = data["daily_return"].iloc[1:]
    std = returns.std(ddof=1)
    volatility = std * np.sqrt(252.0)
    sharpe = returns.mean() / std * np.sqrt(252.0) if std > 0 else np.nan

    downside = returns.where(returns < 0, 0.0)
    downside_deviation = np.sqrt((downside * downside).mean()) * np.sqrt(252.0)
    sortino = (
        returns.mean() * 252.0 / downside_deviation
        if downside_deviation > 0 else np.nan
    )

    max_drawdown = data["drawdown"].min()
    benchmark_return = (
        data["benchmark_value"].iloc[-1]
        / data["benchmark_value"].iloc[0]
        - 1.0
    )

    if trades.empty:
        requested = filled = skipped = 0
        commission = slippage = tax = 0.0
    else:
        trade_data = trades.copy()
        for column in (
            "requested_quantity", "filled_quantity",
            "commission", "slippage", "tax",
        ):
            trade_data[column] = pd.to_numeric(
                trade_data[column], errors="raise"
            )
        requested = int(trade_data["requested_quantity"].sum())
        filled = int(trade_data["filled_quantity"].sum())
        skipped = int(trade_data["filled_quantity"].eq(0).sum())
        commission = float(trade_data["commission"].sum())
        slippage = float(trade_data["slippage"].sum())
        tax = float(trade_data["tax"].sum())

    return pd.DataFrame(
        [
            {
                "strategy": "gradient_boosting_10d_historical_replay",
                "validation_status": "PROVISIONAL",
                "first_date": data["date"].iloc[0].date(),
                "last_date": data["date"].iloc[-1].date(),
                "initial_value_vnd": initial,
                "final_value_vnd": final,
                "total_return": total_return,
                "cagr": cagr,
                "annualized_volatility": volatility,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_drawdown,
                "benchmark_total_return": benchmark_return,
                "active_total_return": total_return - benchmark_return,
                "average_turnover": data["turnover"].mean(),
                "maximum_turnover": data["turnover"].max(),
                "average_cash_weight": data["cash_weight"].mean(),
                "average_hhi": data["hhi"].mean(),
                "minimum_effective_positions": data[
                    "effective_positions"
                ].replace(0.0, np.nan).min(),
                "maximum_single_weight": data["max_single_weight"].max(),
                "maximum_issuer_weight": data["max_issuer_weight"].max(),
                "maximum_sector_weight": data["max_sector_weight"].max(),
                "fill_rate": filled / requested if requested > 0 else np.nan,
                "skipped_execution_count": skipped,
                "total_commission_vnd": commission,
                "total_slippage_vnd": slippage,
                "total_sell_tax_vnd": tax,
                "total_trading_cost_vnd": commission + slippage + tax,
                "validation_notes": (
                    "Current 30-stock universe used historically; "
                    "point-in-time membership unavailable; "
                    "prices are not corporate-action adjusted; "
                    "open price-limit detection uses a 6.9% threshold."
                ),
            }
        ]
    )


def main() -> None:
    config = load_paper_trading_config(
        "config/paper_trading_config.yaml"
    )
    raw = load_ohlcv_csv(config["model"]["raw_ohlcv_path"])
    market = prepare_market_data(
        raw,
        float(config["market_data"]["price_multiplier_to_vnd"]),
    )
    predictions = prepare_predictions(
        pd.read_parquet(
            "data/processed/horizon_tree_predictions.parquet"
        )
    )
    universe = pd.read_csv(config["model"]["universe_path"])
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["issuer_group"] = universe["issuer_group"].astype(str).str.strip()
    universe["sector"] = universe["sector"].astype(str).str.strip()

    issuer_groups = universe.set_index("ticker")["issuer_group"].to_dict()
    sectors = universe.set_index("ticker")["sector"].to_dict()
    constraints = ExecutionConstraints.from_config(config)

    trading_days = tuple(
        pd.Timestamp(value).date()
        for value in sorted(market["date"].unique())
    )
    calendar = TradingCalendar.from_dates(trading_days)
    market_by_date = {
        pd.Timestamp(timestamp).date(): frame.copy()
        for timestamp, frame in market.groupby("date", sort=True)
    }

    candidate_dates = tuple(
        sorted(
            date_value
            for date_value in predictions["date"].dt.date.unique()
            if calendar.is_trading_day(date_value)
        )
    )
    rebalance_dates = tuple(
        signal_date
        for signal_date in candidate_dates[
            ::int(config["timing"]["rebalance_frequency_trading_days"])
        ]
        if signal_date != trading_days[-1]
    )
    if not rebalance_dates:
        raise ValueError("No valid rebalance dates")

    first_date = rebalance_dates[0]
    broker = PaperBrokerState.initialize(
        config["account"]["initial_cash_vnd"],
        first_date,
    )

    pending: dict[date, list[PlannedOrder]] = {}
    performance_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    previous_closes: dict[str, Decimal] | None = None
    latest_rows_by_ticker: dict[str, dict[str, object]] = {}
    signal_dates = set(rebalance_dates)

    for current_date in trading_days:
        if current_date < first_date:
            continue

        rows = market_by_date[current_date]
        for market_row in rows.to_dict(orient="records"):
            latest_rows_by_ticker[str(market_row["ticker"])] = market_row
        valuation_rows = pd.DataFrame(latest_rows_by_ticker.values())

        day_trades: list[dict[str, object]] = []
        execution_skips = 0

        due = pending.pop(current_date, [])
        if due:
            day_trades, execution_skips = execute_orders(
                broker,
                due,
                current_date,
                rows,
                valuation_rows,
                constraints,
                calendar,
                issuer_groups,
                int(config["settlement"]["lag_trading_days"]),
            )
            trade_rows.extend(day_trades)

        # Settlement becomes available after the morning execution.
        broker.settle_due(current_date)

        plan_skips = 0
        if current_date in signal_dates:
            next_date = calendar.next_trading_day(current_date)
            ranked = prediction_frame(predictions, current_date)
            targets = build_constrained_target_weights(
                predictions=ranked,
                universe=universe,
                target_holdings=int(config["portfolio"]["target_holdings"]),
                target_invested_weight=config["portfolio"][
                    "target_invested_weight"
                ],
                max_single_name_weight=config["portfolio"][
                    "max_single_name_weight"
                ],
                max_issuer_group_weight=config["portfolio"][
                    "max_issuer_group_weight"
                ],
                max_sector_weight=config["portfolio"]["max_sector_weight"],
            )
            plan = build_order_plan(
                broker=broker,
                targets=target_objects(targets.target_weights, current_date),
                snapshots=snapshot_map(valuation_rows),
                constraints=constraints,
                signal_date=current_date,
                intended_execution_date=next_date,
                expected_data_date=current_date,
            )
            pending[next_date] = plan.executable_orders
            plan_skips = len(plan.skipped_trades)

        closes = price_map(valuation_rows, "close_vnd")
        benchmark = (
            ZERO
            if previous_closes is None
            else equal_weight_benchmark_return(previous_closes, closes)
        )

        previous_value = (
            to_decimal(performance_rows[-1]["portfolio_value"])
            if performance_rows
            else to_decimal(config["account"]["initial_cash_vnd"])
        )
        gross_traded = sum(
            to_decimal(row["gross_value"]) for row in day_trades
        )
        turnover = (
            Decimal("0.5") * gross_traded / previous_value
            if previous_value > ZERO else ZERO
        )

        performance = build_daily_performance_row(
            performance_date=current_date,
            broker=broker,
            mark_prices=closes,
            previous_rows=pd.DataFrame(performance_rows),
            benchmark_return=benchmark,
            turnover=turnover,
            skipped_trade_count=execution_skips + plan_skips,
        )
        performance.update(
            {
                key: str(value)
                for key, value in risk_metrics(
                    broker, closes, issuer_groups, sectors
                ).items()
            }
        )
        performance_rows.append(performance)
        previous_closes = closes

    performance = pd.DataFrame(performance_rows)
    trades = pd.DataFrame(trade_rows)
    summary = summarize(performance, trades)

    if float(summary.loc[0, "max_drawdown"]) < -1.0:
        raise ValueError("Invalid drawdown below -100%")
    if (pd.to_numeric(performance["portfolio_value"]) <= 0).any():
        raise ValueError("Non-positive portfolio value detected")
    configured_turnover_cap = float(
        config["portfolio"]["max_daily_turnover"]
    )
    if (
        float(summary.loc[0, "maximum_turnover"])
        > configured_turnover_cap + 0.001
    ):
        raise ValueError("Turnover cap was violated")

    PERFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    performance.to_csv(PERFORMANCE_PATH, index=False)
    trades.to_csv(TRADES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    result = summary.iloc[0]
    print("HISTORICAL 100M REPLAY COMPLETED")
    print("=" * 60)
    print("Range:", result["first_date"], "to", result["last_date"])
    print("Final value:", f'{result["final_value_vnd"]:,.0f} VND')
    print("Total return:", f'{result["total_return"]:.2%}')
    print("CAGR:", f'{result["cagr"]:.2%}')
    print("Sharpe:", f'{result["sharpe"]:.3f}')
    print("Sortino:", f'{result["sortino"]:.3f}')
    print("Max drawdown:", f'{result["max_drawdown"]:.2%}')
    print("Benchmark return:", f'{result["benchmark_total_return"]:.2%}')
    print("Active return:", f'{result["active_total_return"]:.2%}')
    print("Average turnover:", f'{result["average_turnover"]:.2%}')
    print("Fill rate:", f'{result["fill_rate"]:.2%}')
    print("Trading costs:", f'{result["total_trading_cost_vnd"]:,.0f} VND')
    print("Validation: PROVISIONAL")


if __name__ == "__main__":
    main()
