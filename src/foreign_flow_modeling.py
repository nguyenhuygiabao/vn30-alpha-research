from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features_foreign_flow import (
    ALL_FOREIGN_FLOW_LEARNING_FEATURES,
    FB_FS_FOREIGN_FLOW_FEATURES,
    FROZEN_FOREIGN_FLOW_FEATURES,
)
from src.labels import (
    add_leave_one_out_equal_weight_relative_labels,
)
from src.model_candidates import (
    paired_block_bootstrap_mean_difference,
)
from src.paper_horizon_history import (
    build_historical_paper_predictions,
)
from src.tree_models import (
    create_tree_models,
    extract_feature_importance,
    fit_predict_tree_model,
)
from src.universe_history import (
    filter_to_point_in_time_universe,
)
from src.walk_forward_split import KEY_COLUMNS, TARGET_COLUMN


DEFAULT_HORIZON_DAYS = 10
DEFAULT_MINIMUM_TRAINING_DATES = 252
DEFAULT_SIGNAL_STEP_DAYS = DEFAULT_HORIZON_DAYS
DEFAULT_TOP_N = 8
DEFAULT_RESEARCH_END = pd.Timestamp("2026-07-22")
FUTURE_HOLDOUT_START = pd.Timestamp("2026-07-23")
FROZEN_MODEL_NAME = "gradient_boosting"

FEATURE_SET_MARKET_BASELINE = "market_baseline"
FEATURE_SET_FROZEN_FOREIGN = "frozen_foreign_flow"
FEATURE_SET_FB_FS = "fb_fs"
FEATURE_SET_ALL_FOREIGN = "frozen_plus_fb_fs"

FOREIGN_FLOW_FEATURE_SET_ORDER = (
    FEATURE_SET_MARKET_BASELINE,
    FEATURE_SET_FROZEN_FOREIGN,
    FEATURE_SET_FB_FS,
    FEATURE_SET_ALL_FOREIGN,
)

# This is the exact 51-feature market baseline used by the established
# FiinQuant comparison. It is intentionally explicit so that later feature
# pipeline additions cannot silently alter the control model.
FROZEN_MARKET_BASELINE_FEATURES = (
    "simple_return_1d",
    "log_return_1d",
    "rolling_return_5d",
    "rolling_return_20d",
    "rolling_vol_20d",
    "drawdown",
    "average_daily_volume_20d",
    "average_daily_value_20d",
    "traded_value_rank_20d",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "negative_shock_1d",
    "positive_shock_1d",
    "reference_price",
    "estimated_ceiling_price",
    "estimated_floor_price",
    "distance_to_ceiling",
    "distance_to_floor",
    "hit_ceiling_today",
    "hit_floor_today",
    "close_at_ceiling_today",
    "close_at_floor_today",
    "consecutive_ceiling_days",
    "consecutive_floor_days",
    "volume_z_20",
    "value_traded_z_20",
    "volume_change_5d",
    "amihud_illiquidity_raw",
    "amihud_illiquidity",
    "abnormal_volume_flag",
    "vn30_return_dispersion",
    "percent_stocks_up",
    "percent_stocks_down",
    "percent_hitting_ceiling",
    "percent_hitting_floor",
    "volume_concentration_top5",
    "rolling_avg_pairwise_corr",
    "market_direction_agreement",
    "price_limit_pressure",
    "herding_corr_score",
    "herding_low_dispersion_score",
    "herding_direction_score",
    "herding_price_limit_score",
    "herding_volume_concentration_score",
    "herding_index",
)


@dataclass(frozen=True)
class PaperHorizonSignal:
    signal_id: int
    signal_date: pd.Timestamp
    training_cutoff: pd.Timestamp


def target_column_for_horizon(horizon_days: int) -> str:
    if horizon_days not in {1, 5, 10}:
        raise ValueError(
            "Foreign-flow modeling supports 1, 5, or 10 trading days"
        )

    return f"forward_relative_return_{horizon_days}d"


def _normalize_keyed_frame(
    data: pd.DataFrame,
    *,
    frame_name: str,
) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS).difference(data.columns))

    if missing:
        raise ValueError(
            f"{frame_name} is missing key columns: {missing}"
        )

    normalized = data.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"],
        errors="raise",
    ).dt.normalize()
    normalized["ticker"] = (
        normalized["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if normalized["ticker"].isna().any():
        raise ValueError(f"{frame_name} contains missing tickers")

    duplicate_count = int(
        normalized.duplicated(KEY_COLUMNS).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{frame_name} contains {duplicate_count} "
            "duplicate ticker-date rows"
        )

    return normalized


def build_common_calendar_forward_labels(
    stock_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build returns to exact future exchange dates.

    A ticker missing the exchange session at t+h receives a missing label.
    The calculation never substitutes that ticker's next observed session.
    """
    required = {"date", "ticker", "adjusted_close"}
    missing = sorted(required.difference(stock_data.columns))

    if missing:
        raise ValueError(
            "Stock data are missing label columns: "
            f"{missing}"
        )

    labels = _normalize_keyed_frame(
        stock_data,
        frame_name="Stock label data",
    ).sort_values(
        KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)
    labels["adjusted_close"] = pd.to_numeric(
        labels["adjusted_close"],
        errors="raise",
    )
    invalid_prices = (
        labels["adjusted_close"].notna()
        & labels["adjusted_close"].le(0)
    )

    if invalid_prices.any():
        raise ValueError(
            "Adjusted closes must be positive when present"
        )

    exchange_dates = pd.DatetimeIndex(
        labels["date"].drop_duplicates()
    ).sort_values()
    price_lookup = labels.set_index(
        ["ticker", "date"]
    )["adjusted_close"]

    for horizon_days in (1, 5, 10):
        future_date_map = pd.Series(
            exchange_dates[horizon_days:],
            index=exchange_dates[:-horizon_days],
        )
        future_dates = labels["date"].map(future_date_map)
        future_keys = pd.MultiIndex.from_arrays(
            [
                labels["ticker"],
                future_dates,
            ],
            names=["ticker", "date"],
        )
        future_close = price_lookup.reindex(
            future_keys
        ).to_numpy(dtype="float64")
        labels[f"forward_return_{horizon_days}d"] = (
            future_close
            / labels["adjusted_close"].to_numpy(
                dtype="float64",
            )
            - 1.0
        )

    return labels


def build_common_calendar_point_in_time_relative_labels(
    stock_data: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Build common-calendar returns, then apply signal-date membership."""
    full_history_labels = build_common_calendar_forward_labels(
        stock_data
    )
    eligible_labels = filter_to_point_in_time_universe(
        market_data=full_history_labels,
        membership=membership,
    )

    return add_leave_one_out_equal_weight_relative_labels(
        eligible_labels
    )


def build_canonical_modeling_dataset(
    features: pd.DataFrame,
    market_data: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Join common-calendar, point-in-time labels to canonical features."""
    target_column = target_column_for_horizon(horizon_days)
    feature_rows = _normalize_keyed_frame(
        features,
        frame_name="Feature data",
    )
    label_input = market_data[
        ["date", "ticker", "adjusted_close"]
    ].copy()
    labels = (
        build_common_calendar_point_in_time_relative_labels(
            stock_data=label_input,
            membership=membership,
        )
    )
    selected_labels = labels[
        KEY_COLUMNS + [target_column]
    ].copy()
    modeling = feature_rows.merge(
        selected_labels,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    if modeling[target_column].notna().sum() == 0:
        raise ValueError(
            f"No usable {target_column} labels overlap the feature data"
        )

    return modeling.sort_values(
        KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)


def _expected_feature_sets() -> dict[str, tuple[str, ...]]:
    baseline = tuple(FROZEN_MARKET_BASELINE_FEATURES)

    return {
        FEATURE_SET_MARKET_BASELINE: baseline,
        FEATURE_SET_FROZEN_FOREIGN: (
            baseline + tuple(FROZEN_FOREIGN_FLOW_FEATURES)
        ),
        FEATURE_SET_FB_FS: (
            baseline + tuple(FB_FS_FOREIGN_FLOW_FEATURES)
        ),
        FEATURE_SET_ALL_FOREIGN: (
            baseline
            + tuple(ALL_FOREIGN_FLOW_LEARNING_FEATURES)
        ),
    }


def build_foreign_flow_feature_sets(
    modeling_data: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    """Return the explicit, frozen feature comparison."""
    return validate_foreign_flow_feature_sets(
        modeling_data,
        _expected_feature_sets(),
    )


def validate_foreign_flow_feature_sets(
    modeling_data: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    expected = _expected_feature_sets()
    expected_names = set(expected)
    supplied_names = set(feature_sets)

    if supplied_names != expected_names:
        raise ValueError(
            "Foreign-flow feature-set names differ from the "
            f"frozen design: expected {sorted(expected_names)}, "
            f"received {sorted(supplied_names)}"
        )

    normalized = {
        name: tuple(feature_sets[name])
        for name in FOREIGN_FLOW_FEATURE_SET_ORDER
    }

    for feature_set_name, columns in normalized.items():
        if columns != expected[feature_set_name]:
            raise ValueError(
                f"{feature_set_name} differs from its explicit "
                "frozen feature definition"
            )

        duplicate_columns = sorted(
            {
                column
                for column in columns
                if columns.count(column) > 1
            }
        )

        if duplicate_columns:
            raise ValueError(
                f"{feature_set_name} contains duplicate features: "
                f"{duplicate_columns}"
            )

        leakage_columns = sorted(
            column
            for column in columns
            if column.startswith(
                (
                    "forward_return_",
                    "forward_relative_return_",
                    "vn30_forward_return_",
                )
            )
        )

        if leakage_columns:
            raise ValueError(
                f"{feature_set_name} contains target leakage: "
                f"{leakage_columns}"
            )

        missing_columns = sorted(
            set(columns).difference(modeling_data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{feature_set_name} is missing source columns: "
                f"{missing_columns}"
            )

    if len(normalized[FEATURE_SET_MARKET_BASELINE]) != 51:
        raise ValueError(
            "The frozen market baseline must contain 51 features"
        )

    return normalized


def build_modeling_sample(
    modeling_data: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, object]]:
    """Keep all valid target rows and preserve missing feature values."""
    target_column = target_column_for_horizon(horizon_days)
    feature_union = tuple(
        dict.fromkeys(
            column
            for columns in feature_sets.values()
            for column in columns
        )
    )
    required = {
        *KEY_COLUMNS,
        target_column,
        *feature_union,
    }
    missing = sorted(required.difference(modeling_data.columns))

    if missing:
        raise ValueError(
            f"Modeling data are missing comparison columns: {missing}"
        )

    working = _normalize_keyed_frame(
        modeling_data,
        frame_name="Modeling data",
    )
    numeric_features = working.loc[
        :,
        list(feature_union),
    ].apply(
        pd.to_numeric,
        errors="raise",
    )
    feature_values = numeric_features.to_numpy(
        dtype="float64",
        na_value=np.nan,
    )
    infinite_mask = np.isinf(feature_values)

    if infinite_mask.any():
        infinite_columns = {
            column: int(infinite_mask[:, index].sum())
            for index, column in enumerate(feature_union)
            if infinite_mask[:, index].any()
        }
        raise ValueError(
            "Modeling features contain infinite values: "
            f"{infinite_columns}"
        )

    working.loc[:, list(feature_union)] = numeric_features
    target = pd.to_numeric(
        working[target_column],
        errors="raise",
    )
    target_values = target.to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    if np.isinf(target_values).any():
        raise ValueError(
            f"{target_column} contains infinite values"
        )

    working[target_column] = target
    valid_target = working[target_column].notna()
    date_target_coverage = (
        working.assign(_valid_target=valid_target)
        .groupby("date", sort=True)["_valid_target"]
        .agg(total_rows="size", valid_rows="sum")
    )
    eligible_signal_dates = pd.DatetimeIndex(
        date_target_coverage.index[
            date_target_coverage["total_rows"].eq(
                date_target_coverage["valid_rows"]
            )
            & date_target_coverage["total_rows"].gt(1)
        ]
    ).sort_values()
    sample = working.loc[valid_target].copy()

    if sample.empty:
        raise ValueError("No finite target rows are available")

    missing_rows_by_feature_set: dict[str, int] = {}
    missing_values_by_feature_set: dict[str, int] = {}

    for feature_set_name, columns in feature_sets.items():
        missing_frame = working.loc[:, list(columns)].isna()
        missing_rows_by_feature_set[feature_set_name] = int(
            missing_frame.any(axis=1).sum()
        )
        missing_values_by_feature_set[feature_set_name] = int(
            missing_frame.sum().sum()
        )

    audit: dict[str, object] = {
        "input_rows": int(len(working)),
        "valid_target_rows": int(valid_target.sum()),
        "excluded_target_rows": int((~valid_target).sum()),
        "input_dates": int(working["date"].nunique()),
        "eligible_signal_dates": int(
            len(eligible_signal_dates)
        ),
        "feature_missing_rows_excluded": 0,
        "missing_rows_by_feature_set": (
            missing_rows_by_feature_set
        ),
        "missing_values_by_feature_set": (
            missing_values_by_feature_set
        ),
    }

    return (
        sample.reset_index(drop=True),
        eligible_signal_dates,
        audit,
    )


def build_paper_horizon_signal_schedule(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    minimum_training_dates: int = (
        DEFAULT_MINIMUM_TRAINING_DATES
    ),
    signal_step_days: int = DEFAULT_SIGNAL_STEP_DAYS,
    evaluation_start: str | pd.Timestamp | None = None,
    evaluation_end: str | pd.Timestamp | None = None,
) -> list[PaperHorizonSignal]:
    """Mirror the established expanding paper-horizon refit schedule."""
    if (
        horizon_days <= 0
        or minimum_training_dates <= 0
        or signal_step_days <= 0
    ):
        raise ValueError(
            "Horizon, training dates, and signal step must be positive"
        )

    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Index(dates), errors="raise")
    ).normalize().unique().sort_values()
    first_signal_position = (
        minimum_training_dates - 1 + horizon_days
    )

    if first_signal_position >= len(normalized_dates):
        raise ValueError(
            "Not enough dates for the requested paper-horizon history"
        )

    start = (
        pd.Timestamp(normalized_dates.min())
        if evaluation_start is None
        else pd.Timestamp(evaluation_start).normalize()
    )
    end = (
        pd.Timestamp(normalized_dates.max())
        if evaluation_end is None
        else pd.Timestamp(evaluation_end).normalize()
    )

    if end < start:
        raise ValueError(
            "Evaluation end cannot precede evaluation start"
        )

    signals: list[PaperHorizonSignal] = []

    for signal_position in range(
        first_signal_position,
        len(normalized_dates),
        signal_step_days,
    ):
        signal_date = pd.Timestamp(
            normalized_dates[signal_position]
        )

        if signal_date < start or signal_date > end:
            continue

        training_cutoff = pd.Timestamp(
            normalized_dates[
                signal_position - horizon_days
            ]
        )
        training_date_count = int(
            (normalized_dates <= training_cutoff).sum()
        )

        if training_date_count < minimum_training_dates:
            raise ValueError(
                "A signal has insufficient expanding training history"
            )

        signals.append(
            PaperHorizonSignal(
                signal_id=len(signals) + 1,
                signal_date=signal_date,
                training_cutoff=training_cutoff,
            )
        )

    if not signals:
        raise ValueError(
            "No paper-horizon signals overlap the evaluation period"
        )

    return signals


def _new_frozen_model() -> object:
    models = create_tree_models()

    if FROZEN_MODEL_NAME not in models:
        raise ValueError(
            f"Production model factory lacks {FROZEN_MODEL_NAME}"
        )

    return models[FROZEN_MODEL_NAME]


def run_foreign_flow_paper_horizon(
    modeling_sample: pd.DataFrame,
    eligible_signal_dates: pd.DatetimeIndex,
    feature_sets: Mapping[str, Sequence[str]],
    signals: Sequence[PaperHorizonSignal],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    model_factory: Callable[[], object] | None = None,
    max_workers: int = 1,
    progress_callback: (
        Callable[[int, int, PaperHorizonSignal], None] | None
    ) = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the frozen production pipeline for every spaced signal."""
    target_column = target_column_for_horizon(horizon_days)
    allowed_sets = set(FOREIGN_FLOW_FEATURE_SET_ORDER)
    unexpected_sets = sorted(
        set(feature_sets).difference(allowed_sets)
    )

    if unexpected_sets:
        raise ValueError(
            f"Unexpected feature sets: {unexpected_sets}"
        )

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    signal_date_set = set(
        pd.DatetimeIndex(eligible_signal_dates)
    )
    runnable_signals = [
        signal
        for signal in signals
        if signal.signal_date in signal_date_set
    ]
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    create_model = model_factory or _new_frozen_model

    for completed_count, signal in enumerate(
        runnable_signals,
        start=1,
    ):
        training = modeling_sample.loc[
            modeling_sample["date"].le(
                signal.training_cutoff
            )
        ].copy()
        test = modeling_sample.loc[
            modeling_sample["date"].eq(signal.signal_date)
        ].copy()

        if training.empty or test.empty:
            continue

        y_train = pd.to_numeric(
            training[target_column],
            errors="raise",
        ).astype("float64")

        if not np.isfinite(y_train).all():
            raise ValueError(
                f"Signal {signal.signal_id} has invalid training labels"
            )

        if y_train.nunique() < 2:
            raise ValueError(
                f"Signal {signal.signal_id} has constant training labels"
            )

        requested_feature_sets = [
            feature_set_name
            for feature_set_name in FOREIGN_FLOW_FEATURE_SET_ORDER
            if feature_set_name in feature_sets
        ]

        def fit_feature_set(
            feature_set_name: str,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            feature_columns = list(
                feature_sets[feature_set_name]
            )
            predictions, fitted_pipeline = (
                fit_predict_tree_model(
                    model=create_model(),
                    x_train=training[feature_columns],
                    y_train=y_train,
                    x_test=test[feature_columns],
                )
            )
            predicted_values = np.asarray(
                predictions,
                dtype="float64",
            )

            if not np.isfinite(predicted_values).all():
                raise ValueError(
                    f"{feature_set_name} produced non-finite "
                    f"predictions for {signal.signal_date.date()}"
                )

            prediction_frame = test[
                KEY_COLUMNS + [target_column]
            ].copy()
            prediction_frame = prediction_frame.rename(
                columns={target_column: "actual_return"}
            )
            prediction_frame["predicted_return"] = (
                predicted_values
            )
            prediction_frame["feature_set"] = feature_set_name
            prediction_frame["model_name"] = FROZEN_MODEL_NAME
            prediction_frame["signal_id"] = signal.signal_id
            prediction_frame["training_cutoff"] = (
                signal.training_cutoff
            )
            prediction_frame["training_date_count"] = int(
                training["date"].nunique()
            )
            prediction_frame["training_row_count"] = len(training)
            prediction_frame["feature_count"] = len(
                feature_columns
            )

            importance = extract_feature_importance(
                fitted_pipeline=fitted_pipeline,
                feature_columns=feature_columns,
                model_name=FROZEN_MODEL_NAME,
            )
            importance["feature_set"] = feature_set_name
            importance["signal_id"] = signal.signal_id
            importance["signal_date"] = signal.signal_date
            importance["training_cutoff"] = (
                signal.training_cutoff
            )
            return prediction_frame, importance

        if max_workers == 1 or len(requested_feature_sets) == 1:
            signal_results = [
                fit_feature_set(feature_set_name)
                for feature_set_name in requested_feature_sets
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=min(
                    max_workers,
                    len(requested_feature_sets),
                )
            ) as executor:
                signal_results = list(
                    executor.map(
                        fit_feature_set,
                        requested_feature_sets,
                    )
                )

        for prediction_frame, importance in signal_results:
            prediction_frames.append(prediction_frame)
            importance_frames.append(importance)

        if progress_callback is not None:
            progress_callback(
                completed_count,
                len(runnable_signals),
                signal,
            )

    if not prediction_frames:
        raise ValueError(
            "No complete paper-horizon predictions were produced"
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        ["date", "feature_set", "ticker"],
        kind="stable",
    ).reset_index(drop=True)
    importance = (
        pd.concat(
            importance_frames,
            ignore_index=True,
        )
        if importance_frames
        else pd.DataFrame(
            columns=[
                "model_name",
                "feature",
                "importance",
                "feature_set",
                "signal_id",
                "signal_date",
                "training_cutoff",
            ]
        )
    )
    _assert_matched_prediction_coverage(
        predictions,
        expected_feature_sets=set(feature_sets),
    )

    return predictions, importance


def build_production_market_baseline_reference(
    modeling_data: pd.DataFrame,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    minimum_training_dates: int = (
        DEFAULT_MINIMUM_TRAINING_DATES
    ),
    signal_step_days: int = DEFAULT_SIGNAL_STEP_DAYS,
    evaluation_start: str | pd.Timestamp | None = None,
    evaluation_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run the established paper-horizon code on the frozen baseline."""
    target_column = target_column_for_horizon(horizon_days)
    required = {
        *KEY_COLUMNS,
        target_column,
        *FROZEN_MARKET_BASELINE_FEATURES,
    }
    missing = sorted(required.difference(modeling_data.columns))

    if missing:
        raise ValueError(
            "Production baseline reference is missing columns: "
            f"{missing}"
        )

    working = _normalize_keyed_frame(
        modeling_data,
        frame_name="Production baseline reference",
    )

    if evaluation_end is not None:
        end = pd.Timestamp(evaluation_end).normalize()
        working = working.loc[
            working["date"].le(end)
        ].copy()

    reference_input = working[
        KEY_COLUMNS
        + list(FROZEN_MARKET_BASELINE_FEATURES)
        + [target_column]
    ].rename(
        columns={target_column: TARGET_COLUMN}
    )
    reference = build_historical_paper_predictions(
        modeling_dataset=reference_input,
        horizon_days=horizon_days,
        minimum_training_dates=minimum_training_dates,
        signal_step_days=signal_step_days,
    )
    reference = reference.loc[
        reference["model_name"].eq(FROZEN_MODEL_NAME)
    ].copy()

    if evaluation_start is not None:
        start = pd.Timestamp(evaluation_start).normalize()
        reference = reference.loc[
            reference["date"].ge(start)
        ].copy()

    if evaluation_end is not None:
        end = pd.Timestamp(evaluation_end).normalize()
        reference = reference.loc[
            reference["date"].le(end)
        ].copy()

    if reference.empty:
        raise ValueError(
            "Production baseline reference produced no predictions"
        )

    return reference.sort_values(
        ["date", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def assert_market_baseline_parity(
    experiment_predictions: pd.DataFrame,
    production_reference: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    baseline = experiment_predictions.loc[
        experiment_predictions["feature_set"].eq(
            FEATURE_SET_MARKET_BASELINE
        )
    ].copy()
    comparison_columns = [
        "date",
        "ticker",
        "predicted_return",
        "actual_return",
        "training_cutoff",
        "training_date_count",
    ]
    left = baseline[comparison_columns].sort_values(
        KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)
    right = production_reference[
        comparison_columns
    ].sort_values(
        KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)

    for frame in (left, right):
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        ).dt.normalize()
        frame["ticker"] = (
            frame["ticker"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    if list(
        left[KEY_COLUMNS].itertuples(
            index=False,
            name=None,
        )
    ) != list(
        right[KEY_COLUMNS].itertuples(
            index=False,
            name=None,
        )
    ):
        raise ValueError(
            "Market baseline does not match production date-ticker coverage"
        )

    for column in (
        "training_cutoff",
        "training_date_count",
    ):
        if not left[column].equals(right[column]):
            raise ValueError(
                f"Market baseline differs from production {column}"
            )

    prediction_difference = (
        left["predicted_return"]
        - right["predicted_return"]
    ).abs()
    actual_difference = (
        left["actual_return"]
        - right["actual_return"]
    ).abs()

    if (
        prediction_difference.gt(tolerance).any()
        or actual_difference.gt(tolerance).any()
    ):
        raise ValueError(
            "Market baseline predictions do not reproduce the "
            "production paper-horizon path"
        )

    return {
        "status": "passed",
        "rows": int(len(left)),
        "signal_dates": int(left["date"].nunique()),
        "maximum_prediction_difference": float(
            prediction_difference.max()
        ),
        "maximum_actual_return_difference": float(
            actual_difference.max()
        ),
        "tolerance": tolerance,
    }


def _assert_matched_prediction_coverage(
    predictions: pd.DataFrame,
    *,
    expected_feature_sets: set[str] | None = None,
) -> None:
    observed_sets = set(
        predictions["feature_set"].unique()
    )

    if (
        expected_feature_sets is not None
        and observed_sets != expected_feature_sets
    ):
        raise ValueError(
            "Prediction output lacks a requested feature set"
        )

    coverage = predictions.groupby(
        ["date", "ticker"],
        sort=False,
    )["feature_set"].agg(set)

    if not coverage.map(
        lambda values: values == observed_sets
    ).all():
        raise ValueError(
            "Feature sets do not have matched date-ticker coverage"
        )

    actual_bounds = predictions.groupby(
        ["date", "ticker"],
        sort=False,
    )["actual_return"].agg(["min", "max"])
    matching_actuals = np.isclose(
        actual_bounds["min"],
        actual_bounds["max"],
        rtol=1e-10,
        atol=1e-12,
    )

    if not matching_actuals.all():
        raise ValueError(
            "Feature sets disagree on realized returns"
        )


def build_daily_foreign_flow_metrics(
    predictions: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    _assert_matched_prediction_coverage(predictions)
    rows: list[dict[str, object]] = []

    for (
        model_name,
        feature_set_name,
        market_date,
    ), daily in predictions.groupby(
        ["model_name", "feature_set", "date"],
        sort=True,
    ):
        if daily["actual_return"].nunique() < 2:
            continue

        selected = daily.sort_values(
            ["predicted_return", "ticker"],
            ascending=[False, True],
            kind="stable",
        ).head(top_n)
        rows.append(
            {
                "date": market_date,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "rank_ic": daily["predicted_return"].corr(
                    daily["actual_return"],
                    method="spearman",
                ),
                "top_n_average_relative_return": (
                    selected["actual_return"].mean()
                ),
                "top_n_positive_rate": (
                    selected["actual_return"].gt(0).mean()
                ),
                "cross_section_size": len(daily),
                "selected_count": len(selected),
            }
        )

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        raise ValueError(
            "No foreign-flow metrics could be calculated"
        )

    date_counts = metrics.groupby(
        ["model_name", "feature_set"]
    )["date"].nunique()

    if date_counts.nunique() != 1:
        raise ValueError(
            "Feature sets have different evaluated-date counts"
        )

    return metrics.sort_values(
        ["date", "model_name", "feature_set"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_foreign_flow_metrics(
    daily_metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        daily_metrics.groupby(
            ["model_name", "feature_set"],
            as_index=False,
            sort=False,
        )
        .agg(
            evaluated_dates=("date", "nunique"),
            rank_ic_observations=("rank_ic", "count"),
            average_rank_ic=("rank_ic", "mean"),
            rank_ic_volatility=("rank_ic", "std"),
            positive_rank_ic_rate=(
                "rank_ic",
                lambda values: values.dropna().gt(0).mean(),
            ),
            average_top_n_relative_return=(
                "top_n_average_relative_return",
                "mean",
            ),
            average_top_n_positive_rate=(
                "top_n_positive_rate",
                "mean",
            ),
            average_cross_section_size=(
                "cross_section_size",
                "mean",
            ),
        )
        .sort_values(
            ["average_rank_ic", "feature_set"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def summarize_paired_foreign_flow_differences(
    daily_metrics: pd.DataFrame,
    *,
    comparisons: Sequence[tuple[str, str]] | None = None,
    block_size: int = 3,
    bootstrap_samples: int = 2_000,
) -> pd.DataFrame:
    requested = comparisons or (
        (
            FEATURE_SET_FROZEN_FOREIGN,
            FEATURE_SET_MARKET_BASELINE,
        ),
        (
            FEATURE_SET_FB_FS,
            FEATURE_SET_MARKET_BASELINE,
        ),
        (
            FEATURE_SET_ALL_FOREIGN,
            FEATURE_SET_MARKET_BASELINE,
        ),
        (
            FEATURE_SET_ALL_FOREIGN,
            FEATURE_SET_FROZEN_FOREIGN,
        ),
    )
    model_names = tuple(
        daily_metrics["model_name"].drop_duplicates()
    )
    rows: list[dict[str, object]] = []

    for model_name in model_names:
        model_metrics = daily_metrics.loc[
            daily_metrics["model_name"].eq(model_name)
        ]
        available = set(model_metrics["feature_set"])

        for challenger, baseline in requested:
            missing = sorted(
                {challenger, baseline}.difference(available)
            )

            if missing:
                raise ValueError(
                    f"Paired comparison lacks feature sets: {missing}"
                )

            challenger_rows = model_metrics.loc[
                model_metrics["feature_set"].eq(challenger),
                ["date", "rank_ic"],
            ].set_index("date")
            baseline_rows = model_metrics.loc[
                model_metrics["feature_set"].eq(baseline),
                ["date", "rank_ic"],
            ].set_index("date")
            paired = challenger_rows.join(
                baseline_rows,
                how="inner",
                lsuffix="_challenger",
                rsuffix="_baseline",
                validate="one_to_one",
            ).dropna(
                subset=[
                    "rank_ic_challenger",
                    "rank_ic_baseline",
                ]
            )

            if len(paired) < 2:
                raise ValueError(
                    f"Paired comparison {challenger} versus "
                    f"{baseline} has fewer than two finite dates"
                )

            difference = (
                paired["rank_ic_challenger"]
                - paired["rank_ic_baseline"]
            )
            mean, lower, upper = (
                paired_block_bootstrap_mean_difference(
                    difference,
                    block_size=block_size,
                    bootstrap_samples=bootstrap_samples,
                )
            )
            rows.append(
                {
                    "model_name": model_name,
                    "challenger": challenger,
                    "baseline": baseline,
                    "paired_dates": len(paired),
                    "mean_rank_ic_difference": mean,
                    "bootstrap_95pct_lower": lower,
                    "bootstrap_95pct_upper": upper,
                    "positive_difference_rate": (
                        difference.gt(0).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_foreign_feature_importance(
    importance: pd.DataFrame,
) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "feature_set",
                "feature",
                "average_importance",
                "importance_observations",
            ]
        )

    foreign = importance.loc[
        importance["feature"].isin(
            ALL_FOREIGN_FLOW_LEARNING_FEATURES
        )
    ]

    return (
        foreign.groupby(
            ["model_name", "feature_set", "feature"],
            as_index=False,
        )
        .agg(
            average_importance=("importance", "mean"),
            importance_observations=("importance", "count"),
        )
        .sort_values(
            [
                "model_name",
                "feature_set",
                "average_importance",
                "feature",
            ],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
