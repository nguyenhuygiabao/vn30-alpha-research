"""Tests for the overview, charts, performance summary and method sections.

The charts and the performance summary are computed at build time from
`reports/tables/tranche_daily_returns.csv`. These tests recompute each figure
independently from that file, check that the page shows it, and check that
the file reproduces the published tranche table the Deployability section
reads, so the two halves of the page cannot disagree.
"""
from __future__ import annotations

import math
import re
import subprocess

import pandas as pd
import pytest

from scripts.build_report_page import (
    AUTHOR_AFFILIATION,
    AUTHOR_LINKEDIN_URL,
    AUTHOR_NAME,
    MINUS,
    PAPER_TRADING_CONFIG_PATH,
    REPO_BRANCH,
    REPO_URL,
    ROOT,
    TABLES,
    TRANCHE_EXECUTION_MODE,
    build_page,
    reconcile_tranche_statistics,
    tranche_curves,
    tranche_statistics,
)
from scripts.describe_modeling_dataset import partition_features
from scripts.diagnose_tranche_drawdown import build_daily_series_table
from src.ablation_tests import ABLATION_GROUPS
from src.backtester import (
    COMMISSION_RATE,
    SELL_TAX_RATE,
    SLIPPAGE_RATE,
    TRADED_LEGS_PER_UNIT_TURNOVER,
)
from src.paper_trading.config import load_paper_trading_config

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
    "tranche_portfolio_diagnostic.csv",
    "tranche_daily_returns.csv",
)

needs_tables = pytest.mark.skipif(
    not all((TABLES / name).is_file() for name in REQUIRED_TABLES),
    reason="result tables not generated; run scripts/run_full_pipeline.py",
)


def prose(html: str) -> str:
    """Tag-stripped, whitespace-collapsed text."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def section(html: str, section_id: str) -> str:
    start = html.index(f'<section id="{section_id}">')
    return html[start : html.index("</section>", start)]


def limited_table() -> pd.DataFrame:
    tranche = pd.read_csv(TABLES / "tranche_portfolio_diagnostic.csv")
    return (
        tranche[tranche["execution_mode"] == TRANCHE_EXECUTION_MODE]
        .sort_values("active_final_equity_after_cost", ascending=False)
        .set_index("optimization_mode")
    )


def tracked_files() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return completed.stdout.splitlines()


# ---------------------------------------------------------------------------
# Order and overview
# ---------------------------------------------------------------------------
@needs_tables
def test_the_new_sections_come_first_in_reading_order() -> None:
    html = build_page()
    order = [
        "Overview",
        "Growth of 1.0 and drawdown, after cost",
        "Performance summary",
        "Data and method",
        "Cost and execution assumptions",
        "Metric definitions",
        "Deployability",
        "Limitations",
        "Reproduce",
    ]
    positions = [html.index(f"<h2>{title}</h2>") for title in order]

    assert positions == sorted(positions)


@needs_tables
def test_the_author_line_links_the_repository_and_the_profile() -> None:
    html = build_page()
    header = html[html.index("<header>") : html.index("</header>")]

    assert f"{AUTHOR_NAME}, {AUTHOR_AFFILIATION}" in header
    assert f'<a href="{REPO_URL}">' in header
    assert f'<a href="{AUTHOR_LINKEDIN_URL}">' in header


@needs_tables
def test_the_overview_states_the_three_findings_from_the_tables() -> None:
    overview = prose(section(build_page(), "overview"))
    best = limited_table().iloc[0]
    models = pd.read_csv(TABLES / "model_comparison_results.csv").set_index(
        "model_name"
    )
    benchmark = pd.read_csv(TABLES / "benchmark_results.csv").set_index("strategy")
    top5 = "average_top5_realized_return_per_5d_period"
    momentum = float(benchmark.loc["top5_momentum", "average_period_active_return"])
    best_model = str(models[top5].idxmax())
    share = float(models.loc[best_model, top5]) / momentum * 100

    assert "No portfolio beats VN30 after costs" in overview
    assert f"{float(best['final_equity_after_cost']):.4f}" in overview
    assert f"{float(best['benchmark_final_equity']):.4f}" in overview
    assert f"{best_model} , reaches {share:.0f}% of it" in overview
    assert "excludes zero before costs" in overview
    assert "span zero after costs" in overview


# ---------------------------------------------------------------------------
# The daily series and what is drawn from it
# ---------------------------------------------------------------------------
@needs_tables
def test_the_daily_series_reproduces_the_published_tranche_table() -> None:
    limited = limited_table()
    modes = [str(mode) for mode in limited.index]
    daily = pd.read_csv(TABLES / "tranche_daily_returns.csv")
    live = tranche_curves(daily, modes)

    reconcile_tranche_statistics(tranche_statistics(daily, live, modes), live, limited)

    tampered = daily.copy()
    column = f"{modes[0]}_after_cost_return"
    tampered.loc[tampered[column].first_valid_index(), column] += 0.01
    tampered_live = tranche_curves(tampered, modes)

    with pytest.raises(ValueError, match="does not reproduce"):
        reconcile_tranche_statistics(
            tranche_statistics(tampered, tampered_live, modes),
            tampered_live,
            limited,
        )


@needs_tables
def test_a_row_carrying_only_some_returns_is_rejected() -> None:
    limited = limited_table()
    modes = [str(mode) for mode in limited.index]
    daily = pd.read_csv(TABLES / "tranche_daily_returns.csv")
    column = f"{modes[-1]}_after_cost_return"
    daily.loc[daily[column].last_valid_index(), column] = float("nan")

    with pytest.raises(ValueError, match="some returns but not all"):
        tranche_curves(daily, modes)


@needs_tables
def test_the_performance_summary_is_recomputed_from_the_daily_series() -> None:
    table = section(build_page(), "performance-summary")
    limited = limited_table()
    modes = [str(mode) for mode in limited.index]
    daily = pd.read_csv(TABLES / "tranche_daily_returns.csv")
    live = tranche_curves(daily, modes)

    def expected(returns: pd.Series, turnover: pd.Series) -> list[str]:
        equity = (1.0 + returns).cumprod()
        years = len(returns) / 252
        cells = [
            f"{(equity.iloc[-1] ** (1 / years) - 1) * 100:.2f}%",
            f"{returns.std(ddof=1) * math.sqrt(252) * 100:.2f}%",
            f"{returns.mean() / returns.std(ddof=1) * math.sqrt(252):.2f}",
            f"{(equity / equity.cummax() - 1).min() * 100:.1f}%",
            f"{turnover.dropna().mean() * 100:.1f}%",
        ]
        return [cell.replace("-", MINUS) for cell in cells]

    rows = {
        f"<code>{mode}</code>": expected(
            live[f"{mode}_after_cost_return"], daily[f"{mode}_tranche_turnover"]
        )
        for mode in modes
    }
    rows["VN30 benchmark"] = expected(
        live["benchmark_return"], daily["benchmark_tranche_turnover"]
    )

    for label, cells in rows.items():
        match = re.search(rf"<tr[^>]*><td>{re.escape(label)}</td>(.*?)</tr>", table)
        assert match, f"no performance row for {label}"
        assert re.findall(r"<td>(.*?)</td>", match.group(1)) == cells

    assert "risk-free rate of 0%" in prose(table)


@needs_tables
def test_the_charts_label_the_published_end_values_and_troughs() -> None:
    svgs = re.findall(
        r"<svg.*?</svg>", section(build_page(), "performance"), flags=re.S
    )
    best = limited_table().iloc[0]

    assert len(svgs) == 2
    growth, drawdown = svgs

    assert f">{float(best['final_equity_after_cost']):.4f}</text>" in growth
    assert f">{float(best['benchmark_final_equity']):.4f}</text>" in growth
    assert (
        f"{MINUS}{abs(float(best['max_drawdown_percent_after_cost'])):.1f}%</text>"
        in drawdown
    )
    assert (
        f"{MINUS}{abs(float(best['benchmark_max_drawdown_percent'])):.1f}%</text>"
        in drawdown
    )

    for svg in svgs:
        assert 'role="img"' in svg
        assert "aria-label=" in svg
        assert "<title>" in svg


def test_the_daily_series_table_keeps_warm_up_turnover_and_live_returns() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    returns = {
        "normal": pd.DataFrame(
            {"date": dates[1:], "after_cost_return": [0.01, -0.02]}
        )
    }
    turnover = {
        "normal": pd.DataFrame(
            {"date": dates, "tranche_turnover": [0.5, 0.2, 0.1]}
        )
    }
    benchmark = pd.DataFrame({"date": dates[1:], "benchmark_return": [0.0, 0.01]})
    benchmark_turnover = pd.DataFrame(
        {"date": dates, "tranche_turnover": [0.5, 0.0, 0.0]}
    )

    table = build_daily_series_table(returns, turnover, benchmark, benchmark_turnover)

    assert table["date"].tolist() == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert table.loc[0, ["benchmark_return", "normal_after_cost_return"]].isna().all()
    assert table["normal_tranche_turnover"].tolist() == [0.5, 0.2, 0.1]

    with pytest.raises(ValueError, match="duplicated date"):
        build_daily_series_table(
            returns,
            turnover,
            pd.concat([benchmark, benchmark.iloc[[0]]]),
            benchmark_turnover,
        )


# ---------------------------------------------------------------------------
# Method, costs and definitions
# ---------------------------------------------------------------------------
@needs_tables
def test_the_feature_groups_reproduce_the_published_ablation_counts() -> None:
    groups = pd.read_csv(TABLES / "feature_groups.csv")
    ablation = pd.read_csv(TABLES / "ablation_results.csv").set_index("ablation_name")
    total = int(groups["feature_count"].sum())
    features = [
        feature for cell in groups["features"] for feature in cell.split(";")
    ]

    assert len(groups) == 6
    assert total == int(ablation.loc["all_features", "feature_count"])
    assert len(features) == len(set(features)) == total

    for name, removed in ABLATION_GROUPS.items():
        if removed:
            assert total - len(set(removed)) == int(
                ablation.loc[name, "feature_count"]
            )


def test_a_feature_column_that_fits_no_group_raises() -> None:
    listed = sorted({column for removed in ABLATION_GROUPS.values() for column in removed})
    known = [*listed, "foreign_net_value", "return_5d", "value_traded_estimated"]

    groups = partition_features(known).set_index("group")

    assert int(groups.loc["foreign_flow", "feature_count"]) == 1
    assert int(groups.loc["momentum_return", "feature_count"]) == 1
    assert int(groups["feature_count"].sum()) == len(known)

    with pytest.raises(ValueError, match="belongs to no group"):
        partition_features([*known, "unexplained_column"])


@needs_tables
def test_the_cost_section_states_what_the_backtester_charges() -> None:
    text = prose(section(build_page(), "costs"))
    per_turnover = (
        TRADED_LEGS_PER_UNIT_TURNOVER * (COMMISSION_RATE + SLIPPAGE_RATE)
        + SELL_TAX_RATE
    )
    config = load_paper_trading_config(PAPER_TRADING_CONFIG_PATH)

    assert f"Commission {COMMISSION_RATE * 100:.2f}% per side" in text
    assert f"Slippage {SLIPPAGE_RATE * 100:.2f}% per side" in text
    assert f"Sell tax {SELL_TAX_RATE * 100:.2f}% of sale value" in text
    assert f"costs {per_turnover * 100:.2f}% × t" in text
    # Board lots, the ADV cap and T+2 live in the paper-trading engine, and
    # the page must not claim them for the backtest.
    assert "ADV cap and board lots Not applied" in text
    assert "Settlement Not modelled" in text
    assert f"{config['execution']['round_lot_size']} shares" in text


@needs_tables
def test_every_metric_the_tables_use_is_defined() -> None:
    glossary = prose(section(build_page(), "definitions"))

    for term in (
        "Rank IC",
        "Top-5 hit rate",
        "Diagnostic Sharpe",
        "Cumulative sum",
        "Active (compounded)",
        "Equity ratio",
    ):
        assert term in glossary


@needs_tables
def test_reproduce_covers_setup_both_platforms_and_data_access() -> None:
    text = prose(section(build_page(), "reproduce"))

    assert "python -m pip install -r requirements.txt" in text
    assert ".venv\\Scripts\\activate" in text
    assert "source .venv/bin/activate" in text
    assert "The FiinQuant data is not in the repository" in text


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------
@needs_tables
def test_every_repository_link_points_at_a_tracked_path() -> None:
    html = build_page()
    tracked = tracked_files()
    prefix = f"{REPO_URL}/"
    links = [
        href[len(prefix) :].split("/", 2)
        for href in re.findall(r'href="([^"]+)"', html)
        if href.startswith(prefix)
    ]

    assert links

    for kind, branch, path in links:
        assert branch == REPO_BRANCH
        if kind == "blob":
            assert path in tracked, f"{path} is linked but not tracked"
        else:
            assert kind == "tree"
            assert any(name.startswith(f"{path}/") for name in tracked), (
                f"{path}/ is linked but holds no tracked file"
            )


@needs_tables
def test_every_tracked_file_the_page_names_is_a_link() -> None:
    html = build_page()
    tracked = set(tracked_files())
    unlinked = re.sub(r"<a [^>]*>.*?</a>", "", html, flags=re.S)

    for name in set(re.findall(r"<code>([^<\s]+)</code>", unlinked)):
        for candidate in (name, f"reports/tables/{name}"):
            assert candidate not in tracked, f"{name} is named but not linked"
