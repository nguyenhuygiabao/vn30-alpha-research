from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from src.features_foreign_flow import (
    ALL_FOREIGN_FLOW_LEARNING_FEATURES,
    FB_FS_FOREIGN_FLOW_FEATURES,
    FROZEN_FOREIGN_FLOW_FEATURES,
)
from src.labels import build_point_in_time_relative_labels
from src.model_candidates import (
    paired_block_bootstrap_mean_difference,
)
from src.modeling_utils import get_model_feature_columns
from src.walk_forward_split import KEY_COLUMNS


DEFAULT_HORIZON_DAYS = 10
DEFAULT_INITIAL_TRAINING_DATES = 504
DEFAULT_PURGE_DATES = DEFAULT_HORIZON_DAYS
DEFAULT_TEST_BLOCK_DATES = 126
DEFAULT_TOP_N = 8
DEFAULT_DEVELOPMENT_END = pd.Timestamp("2023-12-31")
DEFAULT_HOLDOUT_START = pd.Timestamp("2024-01-01")

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


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    train_dates: pd.DatetimeIndex
    purge_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    @property
    def training_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_dates.max())


def target_column_for_horizon(horizon_days: int) -> str:
    if horizon_days not in {1, 5, 10}:
        raise ValueError(
            "Foreign-flow modeling supports 1, 5, or 10 trading days"
        )

    return f"forward_relative_return_{horizon_days}d"


def build_canonical_modeling_dataset(
    features: pd.DataFrame,
    market_data: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Join point-in-time labels to the canonical feature rows."""
    target_column = target_column_for_horizon(horizon_days)
    feature_rows = features.copy()
    feature_rows["date"] = pd.to_datetime(
        feature_rows["date"],
        errors="raise",
    ).dt.normalize()
    feature_rows["ticker"] = (
        feature_rows["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if feature_rows.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Feature data contain duplicate ticker-date rows"
        )

    label_input = market_data[
        ["date", "ticker", "adjusted_close"]
    ].copy()
    labels = build_point_in_time_relative_labels(
        stock_data=label_input,
        membership=membership,
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


def build_foreign_flow_feature_sets(
    modeling_data: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    """Build fixed, auditable feature sets for the foreign-flow test."""
    model_candidates = get_model_feature_columns(modeling_data)
    market_baseline = tuple(
        column
        for column in model_candidates
        if not column.startswith(
            (
                "forward_return_",
                "forward_relative_return_",
                "vn30_forward_return_",
            )
        )
        if not column.startswith("foreign_")
    )

    if not market_baseline:
        raise ValueError("No non-foreign model features are available")

    missing_foreign = sorted(
        set(ALL_FOREIGN_FLOW_LEARNING_FEATURES).difference(
            modeling_data.columns
        )
    )

    if missing_foreign:
        raise ValueError(
            "Modeling data are missing frozen foreign-flow "
            f"features: {missing_foreign}"
        )

    feature_sets = {
        FEATURE_SET_MARKET_BASELINE: market_baseline,
        FEATURE_SET_FROZEN_FOREIGN: (
            market_baseline
            + tuple(FROZEN_FOREIGN_FLOW_FEATURES)
        ),
        FEATURE_SET_FB_FS: (
            market_baseline
            + tuple(FB_FS_FOREIGN_FLOW_FEATURES)
        ),
        FEATURE_SET_ALL_FOREIGN: (
            market_baseline
            + tuple(ALL_FOREIGN_FLOW_LEARNING_FEATURES)
        ),
    }

    return validate_foreign_flow_feature_sets(
        modeling_data,
        feature_sets,
    )


def validate_foreign_flow_feature_sets(
    modeling_data: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    expected_names = set(FOREIGN_FLOW_FEATURE_SET_ORDER)
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
    baseline = normalized[FEATURE_SET_MARKET_BASELINE]
    expected_foreign = {
        FEATURE_SET_MARKET_BASELINE: set(),
        FEATURE_SET_FROZEN_FOREIGN: set(
            FROZEN_FOREIGN_FLOW_FEATURES
        ),
        FEATURE_SET_FB_FS: set(
            FB_FS_FOREIGN_FLOW_FEATURES
        ),
        FEATURE_SET_ALL_FOREIGN: set(
            ALL_FOREIGN_FLOW_LEARNING_FEATURES
        ),
    }

    for feature_set_name, columns in normalized.items():
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

        unexpected_foreign = sorted(
            column
            for column in columns
            if column.startswith("foreign_")
            and column
            not in ALL_FOREIGN_FLOW_LEARNING_FEATURES
        )

        if unexpected_foreign:
            raise ValueError(
                f"{feature_set_name} contains unapproved "
                f"foreign-flow features: {unexpected_foreign}"
            )

        label_features = sorted(
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

        if label_features:
            raise ValueError(
                f"{feature_set_name} contains target leakage: "
                f"{label_features}"
            )

        missing_columns = sorted(
            set(columns).difference(modeling_data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{feature_set_name} is missing source columns: "
                f"{missing_columns}"
            )

        actual_foreign = {
            column
            for column in columns
            if column.startswith("foreign_")
        }

        if actual_foreign != expected_foreign[feature_set_name]:
            raise ValueError(
                f"{feature_set_name} differs from its frozen "
                "foreign-flow feature definition"
            )

        actual_market = tuple(
            column
            for column in columns
            if not column.startswith("foreign_")
        )

        if actual_market != baseline:
            raise ValueError(
                f"{feature_set_name} does not use the same "
                "market baseline"
            )

    return normalized


def build_common_complete_case_sample(
    modeling_data: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, int]]:
    """Use one complete sample for every feature-set comparison.

    Missing foreign flow is excluded. It is never filled with zero.
    A signal date is eligible only when every constituent row on that date
    has complete values for the union of all compared model features.
    """
    target_column = target_column_for_horizon(horizon_days)
    required = {
        *KEY_COLUMNS,
        target_column,
        "foreign_flow_available",
    }
    feature_union = tuple(
        dict.fromkeys(
            column
            for columns in feature_sets.values()
            for column in columns
        )
    )
    required.update(feature_union)
    missing = sorted(required.difference(modeling_data.columns))

    if missing:
        raise ValueError(
            f"Modeling data are missing comparison columns: {missing}"
        )

    working = modeling_data.copy()
    working["date"] = pd.to_datetime(
        working["date"],
        errors="raise",
    ).dt.normalize()
    working["ticker"] = (
        working["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if working.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Modeling data contain duplicate ticker-date rows"
        )

    numeric_features = working.loc[:, feature_union].apply(
        pd.to_numeric,
        errors="raise",
    )
    finite_features = np.isfinite(
        numeric_features.to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
    ).all(axis=1)
    complete_rows = (
        working["foreign_flow_available"].eq(1).to_numpy()
        & finite_features
        & working[target_column].notna().to_numpy()
    )
    working["_comparison_complete"] = complete_rows

    date_coverage = working.groupby(
        "date",
        sort=True,
    )["_comparison_complete"].agg(
        total_rows="size",
        complete_rows="sum",
    )
    complete_signal_dates = pd.DatetimeIndex(
        date_coverage.index[
            date_coverage["total_rows"].eq(
                date_coverage["complete_rows"]
            )
            & date_coverage["total_rows"].gt(1)
        ]
    )

    sample = working.loc[
        working["_comparison_complete"]
    ].drop(columns=["_comparison_complete"])

    if sample.empty:
        raise ValueError(
            "No common complete-case rows are available"
        )

    audit = {
        "input_rows": int(len(working)),
        "complete_rows": int(len(sample)),
        "excluded_rows": int(len(working) - len(sample)),
        "input_dates": int(working["date"].nunique()),
        "complete_signal_dates": int(
            len(complete_signal_dates)
        ),
    }

    return (
        sample.reset_index(drop=True),
        complete_signal_dates.sort_values(),
        audit,
    )


def build_expanding_walk_forward_windows(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    initial_training_dates: int = DEFAULT_INITIAL_TRAINING_DATES,
    purge_dates: int = DEFAULT_PURGE_DATES,
    test_block_dates: int = DEFAULT_TEST_BLOCK_DATES,
    evaluation_start: str | pd.Timestamp | None = None,
    evaluation_end: str | pd.Timestamp | None = None,
) -> list[WalkForwardWindow]:
    if (
        initial_training_dates <= 0
        or purge_dates <= 0
        or test_block_dates <= 0
    ):
        raise ValueError(
            "Training, purge, and test sizes must be positive"
        )

    normalized_dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Index(dates), errors="raise")
    ).normalize().unique().sort_values()
    first_test_position = initial_training_dates + purge_dates

    if first_test_position >= len(normalized_dates):
        raise ValueError(
            "Not enough dates for the requested walk-forward design"
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

    windows: list[WalkForwardWindow] = []
    window_id = 0

    for test_start in range(
        first_test_position,
        len(normalized_dates),
        test_block_dates,
    ):
        test_end = min(
            test_start + test_block_dates,
            len(normalized_dates),
        )
        raw_test_dates = normalized_dates[
            test_start:test_end
        ]
        test_dates = raw_test_dates[
            (raw_test_dates >= start)
            & (raw_test_dates <= end)
        ]

        if test_dates.empty:
            continue

        train_end = test_start - purge_dates
        train_dates = normalized_dates[:train_end]
        purge_window = normalized_dates[
            train_end:test_start
        ]

        if len(train_dates) < initial_training_dates:
            raise ValueError(
                "A walk-forward window has insufficient training dates"
            )

        if len(purge_window) != purge_dates:
            raise ValueError(
                "A walk-forward window has an incomplete purge"
            )

        window_id += 1
        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_dates=train_dates,
                purge_dates=purge_window,
                test_dates=test_dates,
            )
        )

    if not windows:
        raise ValueError(
            "No walk-forward windows overlap the evaluation period"
        )

    return windows


def _default_model_factory() -> object:
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=2,
        random_state=42,
    )


def _validate_finite_model_frame(
    frame: pd.DataFrame,
    *,
    frame_name: str,
) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{frame_name} contains missing or non-finite values"
        )

    return numeric.astype("float64")


def run_foreign_flow_walk_forward(
    common_sample: pd.DataFrame,
    complete_signal_dates: pd.DatetimeIndex,
    feature_sets: Mapping[str, Sequence[str]],
    windows: Sequence[WalkForwardWindow],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    model_factory: Callable[[], object] = _default_model_factory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_column = target_column_for_horizon(horizon_days)
    signal_date_set = set(complete_signal_dates)
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for window in windows:
        training = common_sample.loc[
            common_sample["date"].isin(window.train_dates)
        ].copy()
        test = common_sample.loc[
            common_sample["date"].isin(window.test_dates)
            & common_sample["date"].isin(signal_date_set)
        ].copy()

        if test.empty:
            continue

        if training.empty:
            raise ValueError(
                f"Window {window.window_id} has no training rows"
            )

        y_train = pd.to_numeric(
            training[target_column],
            errors="raise",
        ).astype("float64")

        if not np.isfinite(y_train).all():
            raise ValueError(
                f"Window {window.window_id} has invalid training labels"
            )

        for feature_set_name in FOREIGN_FLOW_FEATURE_SET_ORDER:
            if feature_set_name not in feature_sets:
                continue

            feature_columns = list(
                feature_sets[feature_set_name]
            )
            x_train = _validate_finite_model_frame(
                training[feature_columns],
                frame_name=(
                    f"{feature_set_name} training data "
                    f"for window {window.window_id}"
                ),
            )
            x_test = _validate_finite_model_frame(
                test[feature_columns],
                frame_name=(
                    f"{feature_set_name} test data "
                    f"for window {window.window_id}"
                ),
            )
            model = clone(model_factory())
            model.fit(x_train, y_train)
            predicted = np.asarray(
                model.predict(x_test),
                dtype="float64",
            )

            prediction_frame = test[
                KEY_COLUMNS + [target_column]
            ].copy()
            prediction_frame = prediction_frame.rename(
                columns={target_column: "actual_return"}
            )
            prediction_frame["predicted_return"] = predicted
            prediction_frame["feature_set"] = feature_set_name
            prediction_frame["window_id"] = window.window_id
            prediction_frame["training_cutoff"] = (
                window.training_cutoff
            )
            prediction_frame["feature_count"] = len(
                feature_columns
            )
            prediction_frames.append(prediction_frame)

            if hasattr(model, "feature_importances_"):
                importance_frames.append(
                    pd.DataFrame(
                        {
                            "feature_set": feature_set_name,
                            "window_id": window.window_id,
                            "training_cutoff": (
                                window.training_cutoff
                            ),
                            "feature": feature_columns,
                            "importance": np.asarray(
                                model.feature_importances_,
                                dtype="float64",
                            ),
                        }
                    )
                )

    if not prediction_frames:
        raise ValueError(
            "No complete walk-forward predictions were produced"
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
                "feature_set",
                "window_id",
                "training_cutoff",
                "feature",
                "importance",
            ]
        )
    )

    _assert_matched_prediction_coverage(predictions)

    return predictions, importance


def _assert_matched_prediction_coverage(
    predictions: pd.DataFrame,
) -> None:
    expected_sets = set(predictions["feature_set"].unique())
    coverage = predictions.groupby(
        ["date", "ticker"],
        sort=False,
    )["feature_set"].agg(
        set
    )

    if not coverage.map(
        lambda values: values == expected_sets
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
        feature_set_name,
        market_date,
    ), daily in predictions.groupby(
        ["feature_set", "date"],
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
            "No daily foreign-flow metrics could be calculated"
        )

    date_counts = metrics.groupby(
        "feature_set"
    )["date"].nunique()

    if date_counts.nunique() != 1:
        raise ValueError(
            "Feature sets have different evaluated-date counts"
        )

    return metrics.sort_values(
        ["date", "feature_set"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_foreign_flow_metrics(
    daily_metrics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        daily_metrics.groupby(
            "feature_set",
            as_index=False,
            sort=False,
        )
        .agg(
            evaluated_dates=("date", "nunique"),
            rank_ic_observations=(
                "rank_ic",
                "count",
            ),
            average_rank_ic=("rank_ic", "mean"),
            rank_ic_volatility=("rank_ic", "std"),
            positive_rank_ic_rate=(
                "rank_ic",
                lambda values: (
                    values.dropna().gt(0).mean()
                ),
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
    block_size: int = 10,
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
    available = set(daily_metrics["feature_set"])
    rows: list[dict[str, object]] = []

    for challenger, baseline in requested:
        missing = sorted(
            {challenger, baseline}.difference(available)
        )

        if missing:
            raise ValueError(
                f"Paired comparison lacks feature sets: {missing}"
            )

        challenger_rows = daily_metrics.loc[
            daily_metrics["feature_set"].eq(challenger),
            ["date", "rank_ic"],
        ].set_index("date")
        baseline_rows = daily_metrics.loc[
            daily_metrics["feature_set"].eq(baseline),
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
            ["feature_set", "feature"],
            as_index=False,
        )
        .agg(
            average_importance=("importance", "mean"),
            importance_observations=(
                "importance",
                "count",
            ),
        )
        .sort_values(
            [
                "feature_set",
                "average_importance",
                "feature",
            ],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
