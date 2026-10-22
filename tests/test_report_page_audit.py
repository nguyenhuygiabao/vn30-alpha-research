"""Regression tests for the report-page audit of 2026-10-21.

Each test here pins one claim the page used to get wrong. They are written
against the generated HTML rather than against intermediate strings, because
every one of these defects was invisible in the generator and visible only in
the output.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

import pandas as pd
import pytest

from scripts.build_report_page import (
    MINUS,
    TABLES,
    build_page,
    strip_document_shell,
    train_row_shortfall_sentence,
)
from src.optimizer import MAX_ISSUER_GROUP_WEIGHT, MAX_WEIGHT
from src.vn30_universe import EXPECTED_CONSTITUENTS

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

needs_tranche = pytest.mark.skipif(
    not (TABLES / "tranche_portfolio_diagnostic.csv").is_file(),
    reason="run scripts/diagnose_tranche_drawdown.py",
)


class _TableShape(HTMLParser):
    """Collect the cell width of every row of every table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[tuple[int, int]]]] = []
        self.rows: list[list[tuple[int, int]]] | None = None
        self.row: list[tuple[int, int]] | None = None
        self.empty_headers = 0
        self._in_header = False
        self._header_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "table":
            self.rows = []
            self.tables.append(self.rows)
        elif tag == "tr" and self.rows is not None:
            self.row = []
            self.rows.append(self.row)
        elif tag in {"th", "td"} and self.row is not None:
            self.row.append(
                (int(values.get("colspan", 1)), int(values.get("rowspan", 1)))
            )
            if tag == "th":
                self._in_header = True
                self._header_text = ""

    def handle_data(self, data: str) -> None:
        if self._in_header:
            self._header_text += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "th" and self._in_header:
            if not self._header_text.strip():
                self.empty_headers += 1
            self._in_header = False


def prose(html: str) -> str:
    """Tag-stripped, whitespace-collapsed text.

    Prose assertions must not depend on where the generator's source happens
    to break a line, which is an artefact of implicit string concatenation.
    """
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def row_widths(rows: list[list[tuple[int, int]]]) -> list[int]:
    """Row widths accounting for cells carried down by rowspan."""
    widths: list[int] = []
    carried: list[int] = []

    for row in rows:
        widths.append(sum(colspan for colspan, _ in row) + len(carried))
        next_carried = [remaining - 1 for remaining in carried if remaining - 1 > 0]
        next_carried += [rowspan - 1 for _, rowspan in row if rowspan > 1]
        carried = next_carried

    return widths


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@needs_tables
def test_no_table_has_a_ragged_header_or_an_empty_header_cell() -> None:
    # The group-header rows used to open with `<th></th>` placeholders, which
    # render as blank cells and read as nothing at all to a screen reader.
    shape = _TableShape()
    shape.feed(build_page())

    assert shape.tables, "expected the page to contain tables"
    assert shape.empty_headers == 0

    for index, rows in enumerate(shape.tables, start=1):
        widths = set(row_widths(rows))
        assert len(widths) == 1, f"table {index} has ragged rows: {widths}"


@needs_tables
def test_sweep_chart_labels_do_not_overlap() -> None:
    # A negative bar used to print its value label two pixels from its own
    # category label, so the only negative bar in the sweep was illegible.
    html = build_page()
    svg = html[html.index("<svg") : html.index("</svg>")]

    labels = [
        (float(x), float(y))
        for x, y in re.findall(r'<text x="([\d.]+)" y="([\d.]+)"', svg)
    ]

    by_column: dict[float, list[float]] = {}
    for x, y in labels:
        by_column.setdefault(x, []).append(y)

    for x, ys in by_column.items():
        ordered = sorted(ys)
        for first, second in zip(ordered, ordered[1:]):
            assert second - first >= 10.0, f"labels collide at x={x}: {ordered}"


@needs_tables
def test_standalone_page_keeps_its_head_and_body_only_form_drops_it() -> None:
    # The file is served directly, so it needs its own head. A host that wraps
    # it in a shell of its own renders that head as body text, which is what
    # put "meta-viewport" and "title" on the page as paragraphs.
    html = build_page()

    assert html.startswith("<!doctype html>")
    assert '<meta name="viewport"' in html
    assert "<title>" in html

    embedded = strip_document_shell(html)

    lowered = embedded.lower()

    assert "<!doctype" not in lowered
    assert "<head>" not in lowered and "</head>" not in lowered
    assert "<html" not in lowered
    assert "<meta" not in lowered
    assert "<body" not in lowered and "</body>" not in lowered
    assert "<style>" in embedded
    assert "<h1>" in embedded


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
@needs_tranche
@needs_tables
def test_active_equity_is_defined_and_the_equity_ratio_is_published_beside_it() -> None:
    # Active is the compounded daily active return, not the ratio of final
    # equities. Readers divided one column by another, got a different number,
    # and concluded the table was wrong.
    html = build_page()
    tranche = pd.read_csv(TABLES / "tranche_portfolio_diagnostic.csv")
    limited = tranche[tranche["execution_mode"] == "price_limit_aware_tranche"]

    text = prose(html)

    assert "compounded daily active return" in text
    assert "not strategy equity divided by benchmark equity" in text
    assert "Equity ratio" in text

    for _, row in limited.iterrows():
        active = float(row["active_final_equity_after_cost"])
        ratio = float(row["final_equity_after_cost"]) / float(
            row["benchmark_final_equity"]
        )
        assert f"{active:.4f}" in html
        assert f"{ratio:.4f}" in html


@needs_tranche
@needs_tables
def test_the_two_drawdown_conventions_are_named_and_separated() -> None:
    html = build_page()

    assert "true percentage of capital" in html
    assert "not as a percentage of\ncapital" in html or (
        "not as a percentage of" in html and "capital" in html
    )
    # The summed-return caveat must say which table it governs, because it does
    # not govern the deployability table above it.
    assert "Cumulative figures in <em>this</em> table" in html
    assert "The two conventions share no number" in html


@needs_tables
def test_model_ratios_say_which_model_each_one_belongs_to() -> None:
    # "62% to 88%" compared random_forest before with elastic_net after.
    text = prose(build_page())

    assert "best model at each stage" in text
    assert "Like for like" in text


@needs_tables
def test_the_momentum_baseline_move_is_attributed_to_the_date_count() -> None:
    # top5_momentum is untrained and before cost, so nothing but the evaluated
    # date count can move it.
    html = build_page()
    benchmark = pd.read_csv(TABLES / "benchmark_results.csv").set_index("strategy")
    dates = int(benchmark.loc["top5_momentum", "evaluated_dates"])

    text = prose(html)

    assert "fixed, untrained rule" in text
    assert f"{dates:,} dates" in text
    assert "2,478 dates" in text
    assert "dates that left the sample" in text


@needs_tables
def test_the_sweep_figure_is_distinguished_from_the_headline_figure() -> None:
    html = build_page()

    assert "The sweep figure is not the headline\nfigure" in html.replace(
        "<strong>", ""
    ).replace("</strong>", "") or "sweep figure is not the headline" in html
    assert "not required to agree" in html


def test_train_row_shortfall_is_explained_from_the_measured_counts() -> None:
    sweep = pd.DataFrame(
        {
            "train_size_dates": [3, 126, 252],
            "approximate_train_rows": [90, 3779, 7555],
        }
    )

    sentence = train_row_shortfall_sentence(sweep)

    assert "3,779 rather than 3,780" in sentence
    assert "7,555 rather than 7,560" in sentence
    assert "data gap" in sentence


def test_train_row_shortfall_says_so_when_there_is_none() -> None:
    sweep = pd.DataFrame(
        {
            "train_size_dates": [3, 21],
            "approximate_train_rows": [
                3 * EXPECTED_CONSTITUENTS,
                21 * EXPECTED_CONSTITUENTS,
            ],
        }
    )

    assert "exactly its length times" in train_row_shortfall_sentence(sweep)


# ---------------------------------------------------------------------------
# Claims that need evidence
# ---------------------------------------------------------------------------
@needs_tables
def test_near_zero_is_reported_with_an_interval_not_asserted() -> None:
    # The page used to call a diagnostic Sharpe of 0.004 "statistically
    # indistinguishable from zero" with no test anywhere behind it.
    html = build_page()

    assert "statistically indistinguishable" not in html

    benchmark = pd.read_csv(TABLES / "benchmark_results.csv")

    if "mean_active_return_95pct_lower" not in benchmark.columns:
        pytest.skip("regenerate benchmark_results.csv for the bootstrap columns")

    text = prose(html)

    assert "moving-block bootstrap" in text
    assert "95% interval" in text


@needs_tables
def test_the_issuer_group_weight_is_scoped_to_the_issuer_group_cap() -> None:
    # A 20.1 percent issuer group next to a 20.0 percent single-name cap read
    # as a breached constraint. Two different caps, neither breached.
    text = prose(build_page())

    assert f"{MAX_WEIGHT * 100:.0f} percent figure is the single-name cap" in text
    assert f"issuer-group cap of {MAX_ISSUER_GROUP_WEIGHT * 100:.0f} percent" in text
    assert "not the single-name cap" in text


@needs_tables
def test_period_boundaries_are_stated_rather_than_mixed() -> None:
    html = build_page()

    assert "252 trading days per year" in html
    assert "is the\nbacktest period" in html or "is the backtest period" in html
    assert "last date\nin the dataset" in html or "last date in the dataset" in html


@needs_tables
def test_the_ticker_counts_are_all_three_stated() -> None:
    html = build_page()

    assert "60 is the\ndataset-wide ticker count" in html or (
        "60 is the" in html and "dataset-wide ticker count" in html
    )
    assert "54 is this file's" in html
    assert "30 is the count on any" in html


@needs_tables
def test_the_paper_horizon_row_count_is_decomposed() -> None:
    text = prose(build_page())

    assert "stacks two models" in text
    assert "30 × 2 = 60 rows" in text


@needs_tables
def test_a_negative_change_is_not_rendered_as_a_positive_delta() -> None:
    # `f"+{change}"` printed "+-1" when a count fell.
    html = build_page()

    assert "+-" not in html
    assert f"+{MINUS}" not in html
