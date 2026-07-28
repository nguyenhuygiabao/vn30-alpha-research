from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.feature_pipeline as pipeline
from src.features_herding import (
    calculate_volume_concentration_top_n,
)


def sample_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-22")],
            "ticker": ["HPG"],
            "open": [27.0],
            "high": [28.0],
            "low": [26.0],
            "close": [27.5],
            "adjusted_close": [27.5],
            "volume": [1_000],
            "value_traded": [27_500],
        }
    )


def patch_existing_feature_builders(
    monkeypatch,
) -> None:
    def identity(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    monkeypatch.setattr(
        pipeline,
        "build_basic_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_momentum_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_estimated_price_limits",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_liquidity_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_herding_features",
        identity,
    )


def test_pipeline_uses_canonical_fiinquant_paths() -> None:
    assert pipeline.RAW_MARKET_PATH == (
        "data/raw/fiinquant/canonical/"
        "raw_market_history.csv"
    )
    assert pipeline.ADJUSTED_MARKET_PATH == (
        "data/raw/fiinquant/canonical/"
        "adjusted_market_history.csv"
    )
    assert pipeline.FOREIGN_FLOW_PATH == (
        "data/raw/fiinquant/canonical/"
        "foreign_flow_history.csv"
    )
    assert pipeline.MEMBERSHIP_HISTORY_DIR == (
        "config/vn30_history"
    )
    assert pipeline.UNIVERSE_MODE == "strict_vn30"


def test_verified_membership_history_is_reconstructed(
    tmp_path: Path,
) -> None:
    tickers = [f"T{index:02d}" for index in range(30)]
    pd.DataFrame(
        {
            "effective_date": ["2026-01-01"] * 30,
            "ticker": tickers,
        }
    ).to_csv(
        tmp_path / "initial_snapshot.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "effective_date": [
                "2026-01-01",
                "2026-07-01",
            ]
        }
    ).to_csv(
        tmp_path / "review_calendar.csv",
        index=False,
    )
    pd.DataFrame(
        columns=["effective_date", "action", "ticker"]
    ).to_csv(
        tmp_path / "membership_changes.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "effective_date": [
                "2026-01-01",
                "2026-07-01",
            ],
            "publication_date": [
                "2025-12-20",
                "2026-06-20",
            ],
            "source_url": [
                "https://example.com/one",
                "https://example.com/two",
            ],
            "verified": [True, True],
        }
    ).to_csv(
        tmp_path / "source_manifest.csv",
        index=False,
    )

    membership = (
        pipeline.load_verified_membership_history(
            tmp_path
        )
    )

    assert len(membership) == 30
    assert set(membership["ticker"]) == set(tickers)
    assert membership["effective_from"].eq(
        pd.Timestamp("2026-01-01")
    ).all()
    assert membership["effective_to"].isna().all()


def test_missing_turnover_invalidates_concentration() -> None:
    concentration = calculate_volume_concentration_top_n(
        pd.Series([100.0, float("nan"), 50.0])
    )

    assert pd.isna(concentration)


def complete_feature_output() -> pd.DataFrame:
    row = {
        column: 1.0
        for column in pipeline.REQUIRED_COMBINED_FEATURE_COLUMNS
        if column not in {"date", "ticker"}
    }
    row["date"] = pd.Timestamp("2026-07-22")
    row["ticker"] = "HPG"
    return pd.DataFrame([row])


def test_feature_output_validation_rejects_duplicate_keys() -> None:
    features = complete_feature_output()
    duplicated = pd.concat(
        [features, features],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate ticker-date",
    ):
        pipeline.validate_combined_features(duplicated)


def test_feature_output_validation_rejects_infinity() -> None:
    features = complete_feature_output()
    features.loc[0, "volume_change_5d"] = np.inf

    with pytest.raises(
        ValueError,
        match="infinite values",
    ):
        pipeline.validate_combined_features(features)


def test_save_validates_before_writing(
    tmp_path: Path,
) -> None:
    features = complete_feature_output().drop(
        columns=["foreign_flow_imbalance_1d"]
    )
    output_path = tmp_path / "features.parquet"

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        pipeline.save_combined_features(
            features,
            str(output_path),
        )

    assert not output_path.exists()


def test_pipeline_adds_foreign_features_when_supplied(
    monkeypatch,
) -> None:
    patch_existing_feature_builders(monkeypatch)
    foreign_flow = pd.DataFrame(
        {
            "date": ["2026-07-22"],
            "ticker": ["HPG"],
        }
    )

    def fake_foreign_builder(
        market: pd.DataFrame,
        supplied_flow: pd.DataFrame,
    ) -> pd.DataFrame:
        assert supplied_flow is foreign_flow

        result = market.copy()
        result["foreign_feature_marker"] = 1
        return result

    monkeypatch.setattr(
        pipeline,
        "add_foreign_flow_features",
        fake_foreign_builder,
    )
    result = pipeline.build_combined_features(
        sample_market(),
        foreign_flow=foreign_flow,
    )

    assert result.loc[
        0,
        "foreign_feature_marker",
    ] == 1


def test_pipeline_skips_foreign_features_when_absent(
    monkeypatch,
) -> None:
    patch_existing_feature_builders(monkeypatch)

    def unexpected_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "Foreign builder should not be called"
        )

    monkeypatch.setattr(
        pipeline,
        "add_foreign_flow_features",
        unexpected_call,
    )
    result = pipeline.build_combined_features(
        sample_market()
    )

    assert "foreign_feature_marker" not in result.columns


def test_point_in_time_filter_precedes_cross_sectional_ranks(
    monkeypatch,
) -> None:
    market = pd.concat(
        [
            sample_market().assign(
                ticker=ticker,
                value_traded=value,
            )
            for ticker, value in (
                ("AAA", 100.0),
                ("BBB", 200.0),
                ("CCC", 10_000.0),
            )
        ],
        ignore_index=True,
    )
    membership = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "effective_from": [
                "2026-01-01",
                "2026-01-01",
                "2027-01-01",
            ],
            "effective_to": ["", "", ""],
        }
    )

    def basic_builder(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        result = data.copy()
        result["average_daily_value_20d"] = (
            result["value_traded"]
        )
        result["traded_value_rank_20d"] = -1.0
        return result

    def identity(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    def foreign_builder(
        data: pd.DataFrame,
        foreign_flow: pd.DataFrame,
    ) -> pd.DataFrame:
        del foreign_flow
        result = data.copy()
        result["foreign_net_value_share_1d"] = (
            result["value_traded"] / 100.0
        )
        result["foreign_flow_imbalance_1d"] = 0.0
        result["foreign_net_value_rank_pct"] = -1.0
        result["foreign_buy_value_rank_pct"] = -1.0
        result["foreign_sell_value_rank_pct"] = -1.0
        return result

    def herding_builder(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        assert set(data["ticker"]) == {"AAA", "BBB"}
        return data.copy()

    monkeypatch.setattr(
        pipeline,
        "build_basic_features",
        basic_builder,
    )
    monkeypatch.setattr(
        pipeline,
        "add_momentum_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_estimated_price_limits",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_liquidity_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_foreign_flow_features",
        foreign_builder,
    )
    monkeypatch.setattr(
        pipeline,
        "add_herding_features",
        herding_builder,
    )

    result = pipeline.build_combined_features(
        market,
        foreign_flow=pd.DataFrame(),
        membership=membership,
    ).set_index("ticker")

    assert set(result.index) == {"AAA", "BBB"}
    assert result.loc[
        "AAA",
        "traded_value_rank_20d",
    ] == 0.5
    assert result.loc[
        "BBB",
        "traded_value_rank_20d",
    ] == 1.0
    assert result.loc[
        "AAA",
        "foreign_net_value_rank_pct",
    ] == 0.5
    assert result.loc[
        "BBB",
        "foreign_net_value_rank_pct",
    ] == 1.0


def test_pre_entry_rows_remain_available_for_lags(
    monkeypatch,
) -> None:
    dates = pd.date_range(
        "2026-07-01",
        periods=6,
        freq="D",
    )
    close = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 6,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adjusted_close": close,
            "volume": [1_000] * 6,
            "value_traded": close * 1_000,
        }
    )
    membership = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "effective_from": ["2026-07-06"],
            "effective_to": [""],
        }
    )

    def identity(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    monkeypatch.setattr(
        pipeline,
        "add_momentum_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_estimated_price_limits",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_liquidity_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_herding_features",
        identity,
    )

    result = pipeline.build_combined_features(
        market,
        membership=membership,
    )

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp(
        "2026-07-06"
    )
    assert result.iloc[0]["rolling_return_5d"] == 0.5


def test_missing_raw_prices_mask_price_limit_features(
    monkeypatch,
) -> None:
    market = sample_market()
    market.loc[
        0,
        ["open", "high", "low", "close"],
    ] = pd.NA

    def identity(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    def price_builder(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        result = data.copy()
        result["distance_to_ceiling"] = 0.1
        result["hit_ceiling_today"] = True
        return result

    monkeypatch.setattr(
        pipeline,
        "build_basic_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_momentum_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_estimated_price_limits",
        price_builder,
    )
    monkeypatch.setattr(
        pipeline,
        "add_liquidity_features",
        identity,
    )
    monkeypatch.setattr(
        pipeline,
        "add_herding_features",
        identity,
    )

    result = pipeline.build_combined_features(
        market
    )

    assert pd.isna(
        result.loc[0, "distance_to_ceiling"]
    )
    assert pd.isna(
        result.loc[0, "hit_ceiling_today"]
    )
