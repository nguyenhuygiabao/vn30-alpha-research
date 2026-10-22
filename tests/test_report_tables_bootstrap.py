"""The bootstrap interval that replaced an unsupported significance claim.

The report page used to describe a diagnostic Sharpe of 0.004 as
"statistically indistinguishable from zero" with no test behind it anywhere in
the repository. These tests pin the interval that now stands in its place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_report_tables import (
    BOOTSTRAP_BLOCK_SIZE,
    block_bootstrap_mean_interval,
)


def test_interval_brackets_the_sample_mean() -> None:
    generator = np.random.default_rng(0)
    returns = pd.Series(generator.normal(0.0, 0.02, 2_000))

    lower, upper = block_bootstrap_mean_interval(returns)

    assert lower < returns.mean() < upper


def test_a_series_centred_on_zero_gives_an_interval_spanning_zero() -> None:
    generator = np.random.default_rng(1)
    returns = pd.Series(generator.normal(0.0, 0.02, 2_000))

    lower, upper = block_bootstrap_mean_interval(returns)

    assert lower < 0.0 < upper


def test_a_strong_mean_gives_an_interval_excluding_zero() -> None:
    generator = np.random.default_rng(2)
    returns = pd.Series(generator.normal(0.01, 0.002, 2_000))

    lower, upper = block_bootstrap_mean_interval(returns)

    assert lower > 0.0


def test_the_interval_is_deterministic_for_a_given_seed() -> None:
    returns = pd.Series(np.linspace(-0.01, 0.01, 500))

    assert block_bootstrap_mean_interval(returns) == block_bootstrap_mean_interval(
        returns
    )


def test_blocking_widens_the_interval_on_an_overlapping_series() -> None:
    # Consecutive 5-day forward returns share four of their five days. An
    # i.i.d. bootstrap treats them as independent and reports a narrower
    # interval than the data supports, which is the error the block size
    # exists to avoid. A rolling sum reproduces that dependence.
    generator = np.random.default_rng(3)
    daily = pd.Series(generator.normal(0.0, 0.01, 3_000))
    overlapping = daily.rolling(BOOTSTRAP_BLOCK_SIZE).sum().dropna()

    iid_lower, iid_upper = block_bootstrap_mean_interval(overlapping, block_size=1)
    blocked_lower, blocked_upper = block_bootstrap_mean_interval(overlapping)

    assert (blocked_upper - blocked_lower) > (iid_upper - iid_lower)


def test_a_degenerate_series_returns_not_a_number_rather_than_raising() -> None:
    lower, upper = block_bootstrap_mean_interval(pd.Series([0.01]))

    assert np.isnan(lower) and np.isnan(upper)


def test_non_finite_values_are_dropped() -> None:
    clean = pd.Series([0.01, -0.02, 0.03, 0.00])
    dirty = pd.Series([0.01, np.nan, -0.02, np.inf, 0.03, 0.00])

    assert block_bootstrap_mean_interval(clean) == block_bootstrap_mean_interval(dirty)
