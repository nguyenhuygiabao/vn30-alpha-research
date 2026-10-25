from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_report_page import (
    AUTHOR_LINKEDIN_URL,
    MINUS,
    REPO_URL,
    TABLES,
    build_page,
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
    "dataset_description.csv",
    "feature_groups.csv",
)

needs_tables = pytest.mark.skipif(
    not all((TABLES / name).is_file() for name in REQUIRED_TABLES),
    reason="result tables not generated; run scripts/run_full_pipeline.py",
)


def test_num_uses_a_true_minus_sign_not_a_hyphen() -> None:
    assert num(-0.000481) == f"{MINUS}0.000481"
    assert "-" not in num(-1.234, 3)


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
def test_no_value_is_encoded_by_colour_alone() -> None:
    # The success and critical steps fail CVD separation against each other
    # (OKLab dE 4.6 protan on the light surface, 4.1 deutan on the dark), so
    # hue must never be the only carrier of meaning. The page states verdicts
    # in words and signs with a minus glyph, and colours no value cell.
    html = build_page()

    assert not re.findall(r'class="(?:up|down)"', html)
    assert "behind the benchmark" in html or "ahead of the benchmark" in html


@needs_tables
def test_page_is_self_contained() -> None:
    # Published under a strict CSP on GitHub Pages, and useful offline: the
    # page loads nothing from anywhere. Links out are navigation rather than
    # resources, and go only to the repository and the author's profile.
    html = build_page()

    assert not re.search(r'src="https?://', html)
    assert "<script" not in html
    assert "<link" not in html
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?https?://", html)

    for href in re.findall(r'href="(https?://[^"]+)"', html):
        assert (
            href == REPO_URL
            or href.startswith(f"{REPO_URL}/")
            or href == AUTHOR_LINKEDIN_URL
        ), f"unexpected outbound link {href}"


@needs_tables
def test_page_reports_the_training_window_actually_in_use() -> None:
    from src.walk_forward_split import DEFAULT_TRAIN_SIZE

    html = build_page()

    # Named in the header and in the section that justifies it, so the page
    # cannot describe one window while the pipeline runs another.
    assert f"walk-forward training window {DEFAULT_TRAIN_SIZE} trading dates" in html
    assert f"Why the training window is {DEFAULT_TRAIN_SIZE} dates" in html


@needs_tables
def test_no_reference_to_the_removed_dashboard() -> None:
    html = build_page()
    lowered = html.lower()

    assert "dashboard" not in lowered
    assert "interactive/" not in lowered
