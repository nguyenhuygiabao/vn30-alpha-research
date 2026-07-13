from __future__ import annotations

from scripts.score_daily_signals import parse_args


def test_model_override_is_optional() -> None:
    assert parse_args([]).model is None


def test_model_override_does_not_change_config_argument() -> None:
    args = parse_args(
        ["--config", "custom.yaml", "--model", "rank_ensemble"]
    )

    assert args.config == "custom.yaml"
    assert args.model == "rank_ensemble"
