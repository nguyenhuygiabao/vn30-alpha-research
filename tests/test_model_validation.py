from __future__ import annotations

from src.model_validation import (
    DEFAULT_FIXED_MODEL_CANDIDATES,
    REGIMES,
    build_fixed_model_policies,
)


def test_fixed_model_policies_do_not_switch_models_by_regime() -> None:
    policies = build_fixed_model_policies()

    assert set(policies) == set(DEFAULT_FIXED_MODEL_CANDIDATES)
    assert all(set(policy) == set(REGIMES) for policy in policies.values())
    assert all(
        set(policy.values()) == {model_name}
        for model_name, policy in policies.items()
    )
