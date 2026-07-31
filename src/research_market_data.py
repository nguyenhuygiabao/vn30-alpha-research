"""Shared FiinQuant market-data loading for research and validation scripts.

Scripts under `scripts/` historically read a single flat vnstock CSV
(`data/raw/vnstock/vn30_ohlcv.csv`) with `pandas.read_csv`. That file holds
only the current 30 VN30 tickers applied backward through time, so any
result computed from it carries survivorship bias and does not match the
production pipeline.

This module gives those scripts the same data the pipeline uses: the
FiinQuant canonical raw + adjusted history, optionally filtered to
point-in-time VN30 membership. The returned frame is a strict superset of
the old flat CSV schema, so callers that expected
`date, ticker, open, high, low, close, adjusted_close, volume, value_traded`
keep working and additionally receive `value_traded_estimated` and
`value_traded_is_estimated`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.feature_pipeline import (
    ADJUSTED_MARKET_PATH,
    MEMBERSHIP_HISTORY_DIR,
    RAW_MARKET_PATH,
    UNIVERSE_MODE,
    load_verified_membership_history,
)
from src.market_data_schema import load_canonical_market_history
from src.universe_history import filter_to_universe

DEFAULT_RAW_MARKET_PATH: str = RAW_MARKET_PATH
DEFAULT_ADJUSTED_MARKET_PATH: str = ADJUSTED_MARKET_PATH
DEFAULT_MEMBERSHIP_DIR: str = MEMBERSHIP_HISTORY_DIR
DEFAULT_UNIVERSE_MODE: str = UNIVERSE_MODE


def add_market_data_arguments(
    parser: argparse.ArgumentParser,
    *,
    point_in_time_default: bool = True,
) -> None:
    """Register the standard FiinQuant market-data options on a parser."""
    parser.add_argument(
        "--raw-market-path",
        default=DEFAULT_RAW_MARKET_PATH,
        help="FiinQuant canonical raw OHLC history.",
    )
    parser.add_argument(
        "--adjusted-market-path",
        default=DEFAULT_ADJUSTED_MARKET_PATH,
        help="FiinQuant canonical adjusted history.",
    )
    parser.add_argument(
        "--membership-dir",
        default=DEFAULT_MEMBERSHIP_DIR,
        help="Directory holding verified VN30 membership history.",
    )

    if point_in_time_default:
        parser.add_argument(
            "--include-inactive",
            action="store_true",
            help=(
                "Skip point-in-time VN30 filtering and keep every ticker-date "
                "in the canonical history. Results will not match the "
                "production pipeline."
            ),
        )


def load_research_market_history(
    raw_market_path: str = DEFAULT_RAW_MARKET_PATH,
    adjusted_market_path: str = DEFAULT_ADJUSTED_MARKET_PATH,
    *,
    membership_dir: str = DEFAULT_MEMBERSHIP_DIR,
    point_in_time: bool = True,
    universe_mode: str = DEFAULT_UNIVERSE_MODE,
) -> pd.DataFrame:
    """Load FiinQuant canonical market history for research scripts.

    With `point_in_time=True` (the default) the result contains only
    ticker-dates where the ticker was an actual VN30 constituent, matching
    `src.feature_pipeline`. Pass `point_in_time=False` only when the caller
    specifically needs the unfiltered vendor history, such as a coverage or
    survivorship-bias audit.
    """
    market = load_canonical_market_history(
        raw_market_path,
        adjusted_market_path,
    )

    if not point_in_time:
        return market

    membership = load_verified_membership_history(membership_dir)

    return filter_to_universe(
        market,
        membership,
        mode=universe_mode,
    )


def load_research_market_history_from_args(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Load market history using arguments from `add_market_data_arguments`."""
    return load_research_market_history(
        args.raw_market_path,
        args.adjusted_market_path,
        membership_dir=args.membership_dir,
        point_in_time=not getattr(args, "include_inactive", False),
    )
