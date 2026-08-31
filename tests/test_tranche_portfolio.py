from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    COMMISSION_RATE,
    LIQUIDITY_PENALTY_RATE,
    SLIPPAGE_RATE,
)
from src.tranche_portfolio import (
    build_daily_total_returns,
    build_equal_weight_benchmark_weights,
    build_equity_curves,
    calculate_tranche_costs,
    calculate_tranche_horizon_returns,
    calculate_tranche_turnover,
    simulate_tranche_portfolio,
    summarize_equity_curve,
    summarize_phase_dispersion,
    summarize_start_offset_sensitivity,
)

CALENDAR = pd.to_datetime(
    [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
    ]
)


def build_constant_daily_returns(
    daily_return_by_ticker: dict[str, float],
    calendar: pd.DatetimeIndex = CALENDAR,
) -> pd.DataFrame:
    rows = []

    for ticker, daily_return in daily_return_by_ticker.items():
        for date in calendar:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "daily_return": daily_return,
                }
            )

    return pd.DataFrame(rows)


def build_alternating_weights(
    calendar: pd.DatetimeIndex,
    rebalance_count: int,
) -> pd.DataFrame:
    """Hold AAA on even-indexed dates and BBB on odd-indexed dates."""
    rows = []

    for position in range(rebalance_count):
        rows.append(
            {
                "date": calendar[position],
                "ticker": "AAA" if position % 2 == 0 else "BBB",
                "weight": 1.0,
            }
        )

    return pd.DataFrame(rows)


def test_blended_return_is_the_mean_of_the_phase_sleeves() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.10, "BBB": -0.10})
    weights = build_alternating_weights(CALENDAR, rebalance_count=5)

    simulation = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )

    phase_columns = [
        "phase_0_before_cost_return",
        "phase_1_before_cost_return",
    ]

    pd.testing.assert_series_equal(
        simulation["before_cost_return"],
        simulation[phase_columns].mean(axis=1),
        check_names=False,
    )

    assert simulation[phase_columns].notna().all().all()


def test_hand_built_case_produces_the_expected_equity_path() -> None:
    """Two phases that are exact mirrors must blend to a flat equity curve.

    AAA gains ten percent a day and BBB loses ten percent a day. The even
    tranches hold only AAA and the odd tranches hold only BBB, so one sleeve
    compounds to 1.1 ** 4 and the other to 0.9 ** 4 while the blend, which holds
    half of each every day, returns exactly zero every day.
    """
    daily_returns = build_constant_daily_returns({"AAA": 0.10, "BBB": -0.10})
    weights = build_alternating_weights(CALENDAR, rebalance_count=5)

    simulation = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )

    # Live dates are the ones on which both sleeves are invested: the third
    # through the sixth calendar dates.
    assert list(simulation["date"]) == list(CALENDAR[2:6])

    np.testing.assert_allclose(
        simulation["phase_0_before_cost_return"].to_numpy(),
        0.10,
    )
    np.testing.assert_allclose(
        simulation["phase_1_before_cost_return"].to_numpy(),
        -0.10,
    )
    np.testing.assert_allclose(
        simulation["before_cost_return"].to_numpy(),
        0.0,
        atol=1e-15,
    )

    blended = summarize_equity_curve(simulation["before_cost_return"])

    assert blended["periods"] == 4
    assert blended["final_equity"] == pytest.approx(1.0)
    assert blended["max_drawdown"] == pytest.approx(0.0)

    phase_dispersion = summarize_phase_dispersion(
        curves=simulation,
        return_column_template="phase_{phase}_before_cost_return",
        horizon=2,
    )

    assert phase_dispersion["phase_final_equity_max"] == pytest.approx(1.1**4)
    assert phase_dispersion["phase_final_equity_min"] == pytest.approx(0.9**4)


def test_cash_residual_is_held_flat_within_the_holding_period() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.10})

    weights = pd.DataFrame(
        {
            "date": list(CALENDAR[:5]),
            "ticker": ["AAA"] * 5,
            "weight": [0.5] * 5,
        }
    )

    simulation = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )

    # Day one of a tranche: 0.5 cash plus 0.5 grown to 0.55 is 1.05.
    # Day two: 0.5 cash plus 0.5 grown to 0.605 is 1.105, so 1.105 / 1.05 - 1.
    expected_second_day = 1.105 / 1.05 - 1.0

    np.testing.assert_allclose(
        simulation["phase_0_before_cost_return"].to_numpy()[:2],
        [expected_second_day, 0.05],
    )


def test_horizon_return_is_the_weighted_sum_of_holding_returns() -> None:
    """The reconciliation identity the backtester's arithmetic depends on.

    A buy-and-hold tranche ends its period at `1 + sum(weight * horizon
    return)`. Without this the tranche curve cannot be compared with the
    published `before_cost_return` at all.
    """
    daily_returns = build_constant_daily_returns({"AAA": 0.10, "BBB": -0.05})

    weights = pd.DataFrame(
        {
            "date": list(CALENDAR[:5]) * 2,
            "ticker": ["AAA"] * 5 + ["BBB"] * 5,
            "weight": [0.3] * 5 + [0.7] * 5,
        }
    )

    horizon_returns = calculate_tranche_horizon_returns(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )

    expected = 0.3 * (1.10**2 - 1.0) + 0.7 * (0.95**2 - 1.0)

    np.testing.assert_allclose(
        horizon_returns["horizon_return"].to_numpy(),
        expected,
    )


def test_appending_future_dates_leaves_earlier_returns_identical() -> None:
    """The construction must contain no lookahead.

    The tranche that rebalances on a date reads only the `horizon` dates that
    follow it, so extending the history forward must leave every already
    computed portfolio return bit-identical.
    """
    extended_calendar = CALENDAR.append(
        pd.to_datetime(["2020-01-10", "2020-01-13", "2020-01-14"])
    )

    daily_returns = build_constant_daily_returns(
        {"AAA": 0.10, "BBB": -0.10},
        calendar=extended_calendar,
    )
    weights = build_alternating_weights(extended_calendar, rebalance_count=8)

    truncated_calendar = extended_calendar[:7]

    original = simulate_tranche_portfolio(
        weights=weights[weights["date"].isin(truncated_calendar[:5])],
        daily_returns=daily_returns[
            daily_returns["date"].isin(truncated_calendar)
        ],
        horizon=2,
    )

    appended = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )

    overlap = appended[appended["date"].isin(original["date"])].reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(original, overlap)


def test_a_tranche_never_reads_a_return_from_its_own_rebalance_date() -> None:
    """Changing a rebalance date's own return must not move that tranche.

    Positions are established at the close of the rebalance date, so the return
    printed on that date belongs to the tranche that is retiring, not to the one
    being opened.
    """
    daily_returns = build_constant_daily_returns({"AAA": 0.10})

    weights = pd.DataFrame(
        {
            "date": list(CALENDAR[:5]),
            "ticker": ["AAA"] * 5,
            "weight": [1.0] * 5,
        }
    )

    baseline = calculate_tranche_horizon_returns(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )

    tampered = daily_returns.copy()
    tampered.loc[tampered["date"] == CALENDAR[3], "daily_return"] = 5.0

    perturbed = calculate_tranche_horizon_returns(
        weights=weights,
        daily_returns=tampered,
        horizon=2,
    )

    # The tranche opened on the tampered date is unaffected; the two tranches
    # that were already holding through it are.
    assert perturbed["horizon_return"].iloc[3] == pytest.approx(
        baseline["horizon_return"].iloc[3]
    )
    assert perturbed["horizon_return"].iloc[1] != pytest.approx(
        baseline["horizon_return"].iloc[1]
    )


def test_tranche_turnover_compares_against_the_tranche_it_replaces() -> None:
    weights = pd.DataFrame(
        {
            "date": [
                CALENDAR[0],
                CALENDAR[1],
                CALENDAR[2],
                CALENDAR[3],
            ],
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    turnover = calculate_tranche_turnover(weights, horizon=2)

    # Nothing is held two dates before the first two rebalances, so those are
    # full purchases. The third replaces an identical AAA book, so it trades
    # nothing. The fourth swaps AAA for BBB, a complete turnover.
    np.testing.assert_allclose(
        turnover["tranche_turnover"].to_numpy(),
        [0.5, 0.5, 0.0, 1.0],
    )


def test_tranche_costs_use_the_backtester_cost_model() -> None:
    weights = pd.DataFrame(
        {
            "date": [CALENDAR[0], CALENDAR[1]],
            "ticker": ["AAA", "BBB"],
            "weight": [1.0, 1.0],
        }
    )
    liquidity = pd.DataFrame(
        {
            "date": [CALENDAR[0], CALENDAR[1]],
            "ticker": ["AAA", "BBB"],
            "is_low_liquidity": [False, True],
        }
    )

    costs = calculate_tranche_costs(
        weights=weights,
        liquidity=liquidity,
        horizon=2,
    )

    expected_liquid = 0.5 * (COMMISSION_RATE + SLIPPAGE_RATE)
    expected_illiquid = 0.5 * (
        COMMISSION_RATE + SLIPPAGE_RATE + 1.0 * LIQUIDITY_PENALTY_RATE
    )

    np.testing.assert_allclose(
        costs["tranche_cost"].to_numpy(),
        [expected_liquid, expected_illiquid],
    )


def test_costs_are_charged_to_one_horizon_th_of_capital() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.0})

    weights = pd.DataFrame(
        {
            "date": list(CALENDAR[:5]),
            "ticker": ["AAA"] * 5,
            "weight": [1.0] * 5,
        }
    )
    costs = pd.DataFrame(
        {
            "date": list(CALENDAR[:5]),
            "tranche_cost": [0.02] * 5,
        }
    )

    simulation = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        costs=costs,
        horizon=2,
    )

    # Cost is charged on the date a tranche opens. While rebalancing is
    # continuous, exactly one of the two sleeves opens each day, so the
    # portfolio pays half of that sleeve's cost.
    rebalancing = simulation.iloc[:-1]
    np.testing.assert_allclose(
        rebalancing["after_cost_return"].to_numpy(),
        -0.01,
    )
    np.testing.assert_allclose(
        rebalancing["tranche_cost"].to_numpy(),
        0.01,
    )

    # The final live date sits past the last rebalance, because the weight
    # series ends while both sleeves are still holding. No tranche opens, so
    # nothing is charged. A continuous run has no such date; this asserts the
    # tail behaviour rather than leaving it to chance.
    tail = simulation.iloc[-1]
    assert tail["tranche_cost"] == 0.0
    assert tail["after_cost_return"] == tail["before_cost_return"]


def test_daily_total_returns_come_from_adjusted_close() -> None:
    market_history = pd.DataFrame(
        {
            "date": list(CALENDAR[:3]) * 2,
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "close": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "adjusted_close": [10.0, 11.0, 22.0, 5.0, 4.0, 4.0],
        }
    )

    daily_returns = build_daily_total_returns(market_history)

    aaa = daily_returns[daily_returns["ticker"] == "AAA"]

    assert aaa["daily_return"].isna().iloc[0]
    np.testing.assert_allclose(
        aaa["daily_return"].to_numpy()[1:],
        [0.1, 1.0],
    )


def test_equal_weight_benchmark_weights_sum_to_one_on_every_date() -> None:
    universe = pd.DataFrame(
        {
            "date": [
                CALENDAR[0],
                CALENDAR[0],
                CALENDAR[1],
                CALENDAR[1],
                CALENDAR[1],
            ],
            "ticker": ["AAA", "BBB", "AAA", "BBB", "CCC"],
        }
    )

    benchmark_weights = build_equal_weight_benchmark_weights(
        universe=universe,
        rebalance_dates=CALENDAR[:2],
    )

    weight_by_date = benchmark_weights.groupby("date")["weight"].sum()

    np.testing.assert_allclose(weight_by_date.to_numpy(), 1.0)
    assert benchmark_weights.groupby("date").size().to_list() == [2, 3]


def test_benchmark_weights_reject_a_date_with_no_constituents() -> None:
    universe = pd.DataFrame(
        {
            "date": [CALENDAR[0]],
            "ticker": ["AAA"],
        }
    )

    with pytest.raises(ValueError, match="no point-in-time constituents"):
        build_equal_weight_benchmark_weights(
            universe=universe,
            rebalance_dates=CALENDAR[:2],
        )


def test_active_return_subtracts_the_benchmark_from_every_sleeve() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.10, "BBB": -0.10})
    weights = build_alternating_weights(CALENDAR, rebalance_count=5)

    simulation = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )
    benchmark = pd.DataFrame(
        {
            "date": simulation["date"],
            "benchmark_return": 0.02,
        }
    )

    curves = build_equity_curves(
        simulation=simulation,
        benchmark=benchmark,
        horizon=2,
    )

    np.testing.assert_allclose(
        curves["active_before_cost_return"].to_numpy(),
        -0.02,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        curves["phase_0_active_after_cost_return"].to_numpy(),
        0.08,
    )


def test_missing_benchmark_dates_raise() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.10, "BBB": -0.10})
    weights = build_alternating_weights(CALENDAR, rebalance_count=5)

    simulation = simulate_tranche_portfolio(
        weights=weights,
        daily_returns=daily_returns,
        horizon=2,
    )
    benchmark = pd.DataFrame(
        {
            "date": simulation["date"].iloc[:1],
            "benchmark_return": [0.02],
        }
    )

    with pytest.raises(ValueError, match="Missing benchmark returns"):
        build_equity_curves(
            simulation=simulation,
            benchmark=benchmark,
            horizon=2,
        )


def test_drawdown_is_a_fraction_of_compounded_capital() -> None:
    returns = pd.Series([0.25, -0.50, 0.20])

    summary = summarize_equity_curve(returns, trading_days_per_year=3)

    assert summary["final_equity"] == pytest.approx(1.25 * 0.5 * 1.2)
    assert summary["max_drawdown"] == pytest.approx(0.625 / 1.25 - 1.0)
    assert summary["annualised_return"] == pytest.approx(0.75 - 1.0)


def test_start_offset_sensitivity_reports_a_spread_per_offset() -> None:
    returns = pd.Series([0.10, 0.0, 0.0, 0.0])

    sensitivity = summarize_start_offset_sensitivity(returns, horizon=2)

    assert sensitivity["offset_final_equity_max"] == pytest.approx(1.1)
    assert sensitivity["offset_final_equity_min"] == pytest.approx(1.0)
    assert sensitivity["offset_final_equity_spread"] == pytest.approx(
        0.1, abs=1e-12
    )


def test_non_contiguous_rebalance_dates_raise() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.10})

    weights = pd.DataFrame(
        {
            "date": [CALENDAR[0], CALENDAR[2], CALENDAR[3]],
            "ticker": ["AAA"] * 3,
            "weight": [1.0] * 3,
        }
    )

    with pytest.raises(ValueError, match="contiguous block"):
        simulate_tranche_portfolio(
            weights=weights,
            daily_returns=daily_returns,
            horizon=2,
        )


def test_unwound_calendar_is_required() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.10})

    weights = pd.DataFrame(
        {
            "date": list(CALENDAR),
            "ticker": ["AAA"] * len(CALENDAR),
            "weight": [1.0] * len(CALENDAR),
        }
    )

    with pytest.raises(ValueError, match="before the final tranche is unwound"):
        simulate_tranche_portfolio(
            weights=weights,
            daily_returns=daily_returns,
            horizon=2,
        )


def test_held_ticker_without_price_history_raises() -> None:
    daily_returns = build_constant_daily_returns({"AAA": 0.10})

    weights = pd.DataFrame(
        {
            "date": list(CALENDAR[:5]),
            "ticker": ["ZZZ"] * 5,
            "weight": [1.0] * 5,
        }
    )

    with pytest.raises(ValueError, match="absent from the return history"):
        simulate_tranche_portfolio(
            weights=weights,
            daily_returns=daily_returns,
            horizon=2,
        )
