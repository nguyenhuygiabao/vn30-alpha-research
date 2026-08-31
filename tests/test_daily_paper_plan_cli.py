from __future__ import annotations

import sys

from scripts.preview_daily_paper_orders import parse_args


def test_paper_order_planner_defaults_to_preview(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["preview_daily_paper_orders.py"],
    )

    args = parse_args()

    assert args.model == "rank_ensemble"
    assert args.write is False


def test_paper_order_planner_requires_explicit_write_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preview_daily_paper_orders.py",
            "--model",
            "random_forest",
            "--write",
        ],
    )

    args = parse_args()

    assert args.model == "random_forest"
    assert args.write is True
