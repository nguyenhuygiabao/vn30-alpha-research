from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_report_page import (
    ARROW_DOWN,
    ARROW_UP,
    MINUS,
    TABLES,
    build_page,
    delta_cell,
    num,
)

REQUIRED_TABLES = (
    "horizon_results.csv",
    "ablation_results.csv",
    "benchmark_results.csv",
    "model_comparison_results.csv",
    "training_window_diagnostic.csv",
    "horizon_sample_disclosure.csv",
    "concentration_summary.csv",
    "latest_rank_diagnostic.csv",
)

needs_tables = pytest.mark.skipif(
    not all((TABLES / name).is_file() for name in REQUIRED_TABLES),
    reason="result tables not generated; run scripts/run_full_pipeline.py",
)


def test_num_uses_a_true_minus_sign_not_a_hyphen() -> None:
    assert num(-0.000481) == f"{MINUS}0.000481"
    assert "-" not in num(-1.234, 3)


def test_delta_cell_carries_a_glyph_in_both_directions() -> None:
    # Colour alone must never signal direction: the success and critical steps
    # fail CVD separation against each other, so an improvement and a regression
    # look identical to a red-green colourblind reader without the arrow.
    improved = delta_cell(0.03, 0.02)
    worsened = delta_cell(0.01, 0.02)

    assert ARROW_UP in improved and 'class="up"' in improved
    assert ARROW_DOWN in worsened and 'class="down"' in worsened


def test_delta_cell_omits_the_glyph_when_unchanged() -> None:
    unchanged = delta_cell(0.02, 0.02)

    assert ARROW_UP not in unchanged
    assert ARROW_DOWN not in unchanged


@needs_tables
def test_page_renders_figures_from_the_live_tables() -> None:
    # The point of generating this page is that it cannot disagree with the
    # tables. Every headline figure must appear exactly as the CSV holds it.
    html = build_page()

    horizon = pd.read_csv(TABLES / "horizon_results.csv").set_index(
        "forecast_horizon_days"
    )
    ablation = pd.read_csv(TABLES / "ablation_results.csv").set_index("ablation_name")
    models = pd.read_csv(TABLES / "model_comparison_results.csv").set_index(
        "model_name"
    )

    for horizon_days in (1, 5, 10):
        assert num(horizon.loc[horizon_days, "average_rank_ic"]) in html
        assert num(horizon.loc[horizon_days, "average_after_cost_return"]) in html

    for name in ablation.index:
        assert num(ablation.loc[name, "average_rank_ic"]) in html

    for name in models.index:
        assert num(models.loc[name, "average_rank_ic"]) in html


@needs_tables
def test_every_coloured_value_also_carries_an_arrow() -> None:
    html = build_page()

    coloured = re.findall(r'class="(?:up|down)">([^<]*)<', html)

    assert coloured, "expected the page to contain coloured delta cells"
    for value in coloured:
        assert value.lstrip().startswith((ARROW_UP, ARROW_DOWN)), value


@needs_tables
def test_page_is_self_contained() -> None:
    # Published under a strict CSP on GitHub Pages, and useful offline.
    html = build_page()

    assert not re.search(r'(?:src|href)="https?://', html)
    assert "<script" not in html


@needs_tables
def test_page_reports_the_training_window_actually_in_use() -> None:
    from src.walk_forward_split import DEFAULT_TRAIN_SIZE

    html = build_page()

    assert f"DEFAULT_TRAIN_SIZE = {DEFAULT_TRAIN_SIZE}" in html


@needs_tables
def test_no_reference_to_the_removed_dashboard() -> None:
    html = build_page()
    lowered = html.lower()

    assert "dashboard" not in lowered
    assert "interactive/" not in lowered
