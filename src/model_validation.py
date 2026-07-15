from __future__ import annotations


DEFAULT_FIXED_MODEL_CANDIDATES = (
    "gradient_boosting",
    "random_forest",
    "rank_ensemble",
    "rolling_rank_ensemble",
)
REGIMES = ("trend_up", "trend_down", "high_volatility")


def build_fixed_model_policies(
    model_names: tuple[str, ...] = DEFAULT_FIXED_MODEL_CANDIDATES,
) -> dict[str, dict[str, str]]:
    """Build comparable no-switch policies for a fixed candidate set."""
    if not model_names:
        raise ValueError("At least one model candidate is required")
    if len(set(model_names)) != len(model_names):
        raise ValueError("Model candidates must be unique")
    return {
        model_name: {regime: model_name for regime in REGIMES}
        for model_name in model_names
    }
