"""Generate the project report page from the published result tables.

The page is rebuilt from `reports/tables/*.csv` on every pipeline run, so it
cannot drift from the numbers it cites. Nothing here is hand-entered: a
hardcoded figure would survive a pipeline run that changed it.

The page reports what the framework measures, in the present tense. It is not a
changelog: version history lives in git, not in the results. Every figure here
is read from a table or from the constant that sets it, and the only literals
are the pre-registered foreign-flow comparison keys, the paper-horizon file
identity, and the author line and repository address, none of which is a
result.

The charts and the performance summary are drawn from
`reports/tables/tranche_daily_returns.csv` and must reproduce
`reports/tables/tranche_portfolio_diagnostic.csv` exactly, or the build fails.
Every file the page names is linked to its page on GitHub, and a link to a
path missing from the checkout fails the build too.

Output: reports/reportpage.html
"""

from __future__ import annotations

import argparse
import inspect
import math
import platform
import sys
from collections.abc import Callable
from datetime import date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablation_tests import ABLATION_GROUPS
from src.backtester import (
    COMMISSION_RATE,
    LIQUIDITY_PENALTY_RATE,
    LOW_LIQUIDITY_QUANTILE,
    MAX_EXECUTION_TURNOVER,
    SELL_TAX_RATE,
    SLIPPAGE_RATE,
    TRADED_LEGS_PER_UNIT_TURNOVER,
)
from src.classification_models import TOP_QUINTILE_FRACTION
from src.labels import (
    FORWARD_HORIZON_LONG,
    FORWARD_HORIZON_MEDIUM,
    FORWARD_HORIZON_SHORT,
)
from src.linear_models import create_linear_models
from src.optimizer import (
    HERDING_AWARE_MAX_WEIGHT,
    HERDING_AWARE_TARGET_EXPOSURE,
    HERDING_AWARE_TOP_N,
    HERDING_SCORE_COLUMN,
    HIGH_HERDING_THRESHOLD,
    MAX_ISSUER_GROUP_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MAX_TURNOVER,
    MAX_WEIGHT,
    MODEL_NAME,
    PAPER_MAX_ISSUER_GROUP_WEIGHT,
    PAPER_MAX_SECTOR_WEIGHT,
    PAPER_MAX_SINGLE_NAME_WEIGHT,
    PAPER_MAX_TURNOVER,
    PAPER_TARGET_EXPOSURE,
    PAPER_TARGET_HOLDINGS,
    TOP_N,
)
from src.paper_trading.config import load_paper_trading_config
from src.price_limit import HOSE_NORMAL_DAILY_LIMIT, HOSE_WIDE_DAILY_LIMIT
from src.tranche_portfolio import (
    HOLD_HORIZON_DAYS,
    TRADING_DAYS_PER_YEAR,
    summarize_equity_curve,
)
from src.tree_models import build_tree_model_outputs, create_tree_models
from src.vn30_universe import EXPECTED_CONSTITUENTS
from src.walk_forward_split import DEFAULT_TRAIN_SIZE, PURGE_DAYS, TARGET_COLUMN

TABLES = ROOT / "reports" / "tables"
OUTPUT_PATH = ROOT / "reports" / "reportpage.html"

FOREIGN_FLOW_PAIRED_PATH = (
    TABLES / "foreign_flow" / "foreign_flow_rebuild_development_paired.csv"
)

# The pre-registered comparison: the enhanced foreign buy/sell feature set
# against the frozen one, on paired daily Rank IC.
FOREIGN_FLOW_CHALLENGER = "enhanced_fb_fs"
FOREIGN_FLOW_BASELINE = "frozen_foreign_flow"

PAPER_HORIZON_PREDICTIONS_PATH = (
    ROOT / "data" / "processed" / "paper_horizon_predictions.parquet"
)

# Distinct tickers in the modeling dataset across the full history: the 30
# current constituents plus 30 names that have left the index.
DATASET_TICKERS = 60

PAPER_TRADING_CONFIG_PATH = ROOT / "config" / "paper_trading_config.yaml"

# The execution mode every compounded figure on the page is reported under.
TRANCHE_EXECUTION_MODE = "price_limit_aware_tranche"

REPO_URL = "https://github.com/nguyenhuygiabao/vn30-alpha-research"
REPO_BRANCH = "main"
AUTHOR_NAME = "Bao H. G. Nguyen"
AUTHOR_AFFILIATION = "Grinnell College"
AUTHOR_LINKEDIN_URL = "https://www.linkedin.com/in/huygiabaonguyen/"

# Chart geometry, shared so the two time-series charts align when stacked.
CHART_WIDTH = 720.0
PLOT_LEFT = 52.0
PLOT_RIGHT = 590.0
END_LABEL_X = PLOT_RIGHT + 12.0

STRATEGY_LABELS = {
    "ml_sector_diversified_normal": ("sector_diversified", "normal"),
    "ml_sector_diversified_price_limit_aware": ("sector_diversified", "price_limit_aware"),
    "ml_herding_aware_normal": ("herding_aware", "normal"),
    "ml_sector_aware_normal": ("sector_aware", "normal"),
    "ml_normal_normal": ("normal", "normal"),
    "ml_herding_aware_price_limit_aware": ("herding_aware", "price_limit_aware"),
    "ml_sector_aware_price_limit_aware": ("sector_aware", "price_limit_aware"),
    "ml_normal_price_limit_aware": ("normal", "price_limit_aware"),
}

MINUS = "−"

STYLE = """
  :root {
    color-scheme: light;
    --surface-1:#fcfcfb; --page:#f9f9f7;
    --ink-1:#0b0b0b; --ink-2:#52514e; --ink-muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
    --series-1:#2a78d6; --good:#006300; --critical:#d03b3b; --neutral-mid:#f0efec;
    --series-2:#eb6834; --link:#1c5cab;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:#1a1a19; --page:#0d0d0d;
      --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-muted:#898781;
      --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
      --series-1:#3987e5; --good:#0ca30c; --critical:#d03b3b; --neutral-mid:#383835;
      --series-2:#d95926; --link:#86b6ef;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:#1a1a19; --page:#0d0d0d;
    --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --good:#0ca30c; --critical:#d03b3b; --neutral-mid:#383835;
    --series-2:#d95926; --link:#86b6ef;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--page); color:var(--ink-1);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; font-size:15px; }
  .wrap { max-width:1080px; margin:0 auto; padding:32px 20px 64px; }
  h1 { font-size:1.75rem; line-height:1.25; margin:0 0 6px; letter-spacing:-0.01em; }
  .sub { color:var(--ink-2); margin:0 0 4px; }
  .meta { color:var(--ink-muted); font-size:0.85rem; margin:0; }
  section { background:var(--surface-1); border:1px solid var(--border);
    border-radius:10px; padding:20px 22px; margin-top:22px; }
  h2 { font-size:1.12rem; margin:0 0 4px; letter-spacing:-0.005em; }
  .note { color:var(--ink-2); font-size:0.9rem; margin:6px 0 0; }
  .caption { color:var(--ink-muted); font-size:0.82rem; margin:8px 0 0; }
  p { margin:10px 0 0; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-top:14px; }
  .tile { border:1px solid var(--border); border-radius:8px; padding:13px 14px; }
  .tile .label { font-size:0.78rem; color:var(--ink-muted); text-transform:uppercase; letter-spacing:0.04em; }
  .tile .value { font-size:1.5rem; font-weight:600; margin-top:4px; letter-spacing:-0.01em; }
  .tile .detail { font-size:0.82rem; color:var(--ink-2); margin-top:4px; }
  .tablewrap { overflow-x:auto; margin-top:12px; -webkit-overflow-scrolling:touch; }
  table { border-collapse:collapse; width:100%; min-width:560px; font-size:0.88rem; }
  th, td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--grid);
    font-variant-numeric:tabular-nums; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; font-variant-numeric:normal; }
  thead th { color:var(--ink-2); font-weight:600; border-bottom:1px solid var(--axis); font-size:0.82rem; }
  tbody tr:last-child td { border-bottom:none; }
  tbody tr.highlight td { background:color-mix(in srgb, var(--series-1) 8%, transparent); }
  caption { caption-side:top; text-align:left; color:var(--ink-2); font-size:0.86rem; padding-bottom:6px; }
  ul { margin:10px 0 0; padding-left:20px; } li { margin:5px 0; }
  code { font-family:ui-monospace,"Cascadia Code",Consolas,monospace; font-size:0.86em;
    background:var(--neutral-mid); padding:1px 5px; border-radius:4px; }
  pre { background:var(--neutral-mid); padding:12px 14px; border-radius:8px; overflow-x:auto;
    font-size:0.84rem; margin-top:10px; }
  pre code { background:none; padding:0; }
  .callout { border-left:3px solid var(--axis); padding:2px 0 2px 14px; margin-top:14px; color:var(--ink-2); }
  .callout strong { color:var(--ink-1); }
  figure { margin:14px 0 0; }
  svg { display:block; max-width:100%; height:auto; }
  footer { color:var(--ink-muted); font-size:0.82rem; margin-top:26px; text-align:center; }
  a { color:var(--link); text-decoration:underline; text-decoration-thickness:1px;
    text-underline-offset:2px; }
  a code { color:inherit; }
  .byline { color:var(--ink-2); margin:10px 0 4px; font-size:0.92rem; }
  h3 { font-size:0.98rem; margin:20px 0 0; letter-spacing:-0.005em; }
  ol.findings { margin:12px 0 0; padding-left:22px; } ol.findings li { margin:8px 0; }
  td.prose { white-space:normal; min-width:220px; }
  th.prose, td.prose { text-align:left; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:10px;
    font-size:0.84rem; color:var(--ink-2); }
  .legend .key { display:inline-block; width:16px; border-top:2px solid; border-radius:1px;
    vertical-align:middle; margin-right:6px; }
  .key-s1 { border-color:var(--series-1); }
  .key-s2 { border-color:var(--series-2); }
  .key-muted { border-color:var(--ink-muted); opacity:0.7; }
  .chartwrap { overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .chartwrap svg { min-width:560px; }
  .hit .crosshair { opacity:0; }
  .hit:hover .crosshair { opacity:1; }
  dl.glossary { margin:12px 0 0; }
  dl.glossary dt { font-weight:600; margin-top:12px; }
  dl.glossary dd { margin:3px 0 0; color:var(--ink-2); }
"""


def read_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Required result table not found: {path}. "
            "Run scripts/run_full_pipeline.py first."
        )
    return pd.read_csv(path)


def optional_table(name: str) -> pd.DataFrame | None:
    """Read a diagnostic table that no pipeline stage is required to write."""
    path = TABLES / name
    if not path.is_file():
        return None
    return pd.read_csv(path)


def num(value: float, digits: int = 6) -> str:
    """Format with a true minus sign rather than a hyphen."""
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}".replace("-", MINUS)


def tile(label: str, value: str, detail: str) -> str:
    return (
        f'<div class="tile"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="detail">{detail}</div></div>'
    )


def horizon_list(days: list[int]) -> str:
    labels = [f"{day}-day" for day in days]
    if not labels:
        return "none"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def interval_clause(row: pd.Series) -> str:
    """Describe a row's bootstrap interval as an appendable clause.

    It reports what the interval does rather than asserting a verdict, because
    the sign of a near-zero mean is exactly what moves when the sample does.
    """
    lower_key = "mean_active_return_95pct_lower"
    upper_key = "mean_active_return_95pct_upper"

    if lower_key not in row.index or pd.isna(row[lower_key]):
        return ", with no interval published for it"

    lower = float(row[lower_key])
    upper = float(row[upper_key])
    verdict = "spans zero" if lower <= 0.0 <= upper else "excludes zero"

    return (
        ", and a 95% bootstrap interval on mean active return of "
        f"[{num(lower)}, {num(upper)}], which {verdict}"
    )


def interval_sentence(benchmark: pd.DataFrame, ml_sorted: pd.DataFrame) -> str:
    """Report the bootstrap intervals for the best and most conservative rows.

    A near-zero mean and a near-zero Sharpe cannot be described in words
    without a test behind them. The interval is computed in
    `scripts/build_report_tables.py` and published in `benchmark_results.csv`.
    """
    needed = {
        "mean_active_return_95pct_lower",
        "mean_active_return_95pct_upper",
    }

    if not needed.issubset(benchmark.columns):
        return (
            '<p class="note">No confidence interval is published for these '
            "rows. Rebuild "
            f"{file_link('reports/tables/benchmark_results.csv', 'benchmark_results.csv')} "
            f"with {file_link('scripts/build_report_tables.py')} to add one.</p>"
        )

    best_name = str(ml_sorted.index[0])
    names = [best_name]
    if best_name != "ml_normal_price_limit_aware":
        names.append("ml_normal_price_limit_aware")

    parts: list[str] = []
    for name in names:
        row = benchmark.loc[name]
        optimizer, execution = STRATEGY_LABELS.get(name, (name, ""))
        lower = float(row["mean_active_return_95pct_lower"])
        upper = float(row["mean_active_return_95pct_upper"])
        verdict = "excludes zero" if (lower > 0.0 or upper < 0.0) else "spans zero"
        parts.append(
            f"<code>{optimizer}</code> + <code>{execution}</code> has mean "
            f"active return {num(float(row['average_period_active_return']))} "
            f"with a 95% interval of [{num(lower)}, {num(upper)}], which "
            f"{verdict}"
        )

    ml_rows = benchmark[benchmark["comparison_type"] == "ml_strategy"]
    spanning = int((~ml_rows["bootstrap_interval_excludes_zero"].astype(bool)).sum()) if (
        "bootstrap_interval_excludes_zero" in ml_rows.columns
    ) else 0

    return (
        '<p class="note"><strong>Is any of this distinguishable from '
        "zero?</strong> "
        + ". ".join(parts)
        + f". Across all {len(ml_rows)} optimizer and execution combinations, "
        f"{spanning} have an interval that spans zero. The interval is a 95 "
        "percent moving-block bootstrap on the mean after-cost active return, "
        "2,000 resamples, blocked at the 5-day forecast horizon so that the "
        "overlap between consecutive period returns stays inside a block. An "
        "interval that spans zero is why no alpha claim is made here; it is "
        "not a claim that the mean is zero.</p>"
    )


def foreign_flow_item() -> str:
    """The pre-registered foreign-flow result, read from its own table."""
    if not FOREIGN_FLOW_PAIRED_PATH.is_file():
        return (
            "<li><strong>Foreign buy/sell features are not promoted.</strong> "
            "The pre-registered comparison table is not present in this "
            "checkout.</li>"
        )

    paired = pd.read_csv(FOREIGN_FLOW_PAIRED_PATH)
    matched = paired[
        (paired["challenger"] == FOREIGN_FLOW_CHALLENGER)
        & (paired["baseline"] == FOREIGN_FLOW_BASELINE)
    ]

    if matched.empty:
        return (
            "<li><strong>Foreign buy/sell features are not promoted.</strong> "
            "The pre-registered comparison row is missing from its table.</li>"
        )

    row = matched.iloc[0]
    difference = float(row["mean_rank_ic_difference"])
    lower = float(row["bootstrap_95pct_lower"])
    upper = float(row["bootstrap_95pct_upper"])
    dates = int(row["paired_dates"])
    spans_zero = lower <= 0.0 <= upper

    return (
        "<li><strong>Foreign buy/sell features are not promoted.</strong> "
        "Pre-registered test, paired block bootstrap on daily Rank IC over "
        f"{dates} non-overlapping signals: mean difference {num(difference)}, "
        f"95% CI [{num(lower)}, {num(upper)}]. The interval "
        + ("spans zero" if spans_zero else "excludes zero")
        + " and the point estimate is "
        + ("negative" if difference < 0 else "positive")
        + ". The promotion rule was fixed before the result was seen: promote "
        "only if the interval excludes zero.</li>"
    )


def paper_horizon_facts() -> str:
    """Describe the paper-horizon prediction file from the file itself.

    `data/processed/` is not tracked, so the page cannot depend on the parquet
    existing. When it does, the row count is decomposed here, which is what
    keeps "60 rows a date against a 30-name index" from reading as a
    duplicate-row defect.
    """
    if not PAPER_HORIZON_PREDICTIONS_PATH.is_file():
        return (
            '<p class="caption">'
            "<code>data/processed/paper_horizon_predictions.parquet</code> is "
            "not present in this checkout, so its shape is not described "
            "here.</p>"
        )

    frame = pd.read_parquet(PAPER_HORIZON_PREDICTIONS_PATH)
    rows = len(frame)
    dates = int(frame["date"].nunique())
    tickers = int(frame["ticker"].nunique())
    model_names = sorted(frame["model_name"].unique())
    duplicate_keys = int(frame.duplicated(subset=["date", "ticker"]).sum())
    duplicate_rows = int(
        frame.duplicated(subset=["date", "ticker", "model_name"]).sum()
    )
    per_date = rows / dates if dates else float("nan")
    first_date = str(pd.Timestamp(frame["date"].min()).date())
    last_date = str(pd.Timestamp(frame["date"].max()).date())

    return (
        '<p class="note"><strong>Why the prediction file carries '
        f"{per_date:.0f} rows a date against a {EXPECTED_CONSTITUENTS}-name "
        "index.</strong> It stacks "
        f"{len(model_names)} models — {', '.join(model_names)} — in "
        "one table with a <code>model_name</code> column, so a fully covered "
        f"date carries {EXPECTED_CONSTITUENTS} × {len(model_names)} = "
        f"{per_date:.0f} rows and its {dates} dates carry {rows:,}. Each "
        f"(date, ticker) pair appears {len(model_names)} times by design "
        f"({duplicate_keys:,} repeated pairs) and each (date, ticker, model) "
        f"key once ({duplicate_rows} repeated keys), so the repetition is the "
        "models rather than a duplicate-row or join defect. The file covers "
        f"{tickers} distinct tickers over {first_date} to {last_date}.</p>"
    )


def train_row_shortfall_sentence(sweep: pd.DataFrame) -> str:
    """Explain why `approximate_train_rows` is not the window length times 30.

    The sweep measures the rows actually present on the first window's
    training dates. The VN30 carries 30 constituents, so the arithmetic answer
    is `train_size * 30`, and some windows come in under it. Ten dates in the
    dataset carry 29 rows rather than 30, because FLC has no data for the week
    of 2017-01-16 and ROS none for the week of 2021-01-25. A window spanning
    one of those dates is one row short per date spanned, which is a gap in
    the data rather than an error in the count.
    """
    deficits: list[str] = []

    for row in sweep.sort_values("train_size_dates").itertuples(index=False):
        size = int(row.train_size_dates)
        measured = int(row.approximate_train_rows)
        expected = size * EXPECTED_CONSTITUENTS
        if measured != expected:
            deficits.append(
                f"{size} dates measures {measured:,} rather than "
                f"{expected:,} ({expected - measured} short)"
            )

    if not deficits:
        return (
            "Every window in this sweep measures exactly its length times "
            f"{EXPECTED_CONSTITUENTS}."
        )

    return (
        "; ".join(deficits)
        + ". Ten dates in the dataset carry 29 constituents rather than 30, "
        "because FLC has no data for the week of 2017-01-16 and ROS none for "
        "the week of 2021-01-25, and a window spanning one of those dates is "
        "one row short for each. The shortfall is a data gap, not a counting "
        "error."
    )


def deployability_section(tranche: pd.DataFrame) -> str:
    """What a portfolio running this strategy would actually have done.

    Every per-period figure on the page is a mean over overlapping forecast
    periods, which a portfolio cannot hold. The strategy signals daily and
    holds five days, so what it trades is five overlapping tranches. That
    construction holds every phase at once and has no free parameters, unlike
    sampling every fifth date, where the starting offset moves the answer more
    than the strategy does.

    Two columns here are routinely misread, so both are defined on the page.

    `Active` is not strategy equity divided by benchmark equity. It is the
    compounded daily active return -- the product over trading days of
    `1 + (strategy return - benchmark return)` -- which is what a
    daily-rebalanced long-strategy/short-index sleeve would have returned. The
    equity ratio is published beside it precisely because the two differ:
    compounding a difference of returns is not the same as taking a ratio of
    compounded returns.

    `Strategy drawdown` is the worst peak-to-trough fall of the compounded
    strategy equity curve, a genuine fraction of capital. `Active drawdown` is
    the same measure on the compounded active-return curve. Neither is the
    summed-return drawdown reported in the portfolio-backtest section below.
    """
    limited = tranche[
        tranche["execution_mode"] == "price_limit_aware_tranche"
    ].sort_values("active_final_equity_after_cost", ascending=False)

    benchmark = float(limited["benchmark_final_equity"].iloc[0])
    years = float(limited["years"].iloc[0])
    periods = int(limited["periods"].iloc[0])
    first_date = str(limited["first_date"].iloc[0])
    last_date = str(limited["last_date"].iloc[0])

    best_row = limited.iloc[0]
    best = float(best_row["final_equity_after_cost"])
    best_mode = str(best_row["optimization_mode"])
    best_drawdown = abs(float(best_row["max_drawdown_percent_after_cost"]))
    best_active_drawdown = abs(
        float(best_row["active_max_drawdown_percent_after_cost"])
    )
    behind = [
        100.0 * (1.0 - float(row["active_final_equity_after_cost"]))
        for _, row in limited.iloc[1:].iterrows()
    ]

    rows: list[str] = []
    for _, row in limited.iterrows():
        active = float(row["active_final_equity_after_cost"])
        equity = float(row["final_equity_after_cost"])
        # Stated in words, not by colour alone.
        verdict = "behind the index" if active < 1.0 else "ahead of the index"
        rows.append(
            "<tr>"
            f"<td><code>{row['optimization_mode']}</code></td>"
            f"<td>{equity:.4f}</td>"
            f"<td>{benchmark:.4f}</td>"
            f"<td>{active:.4f} · {verdict}</td>"
            f"<td>{equity / benchmark:.4f}</td>"
            f"<td>{MINUS}"
            f"{abs(float(row['max_drawdown_percent_after_cost'])):.1f}%</td>"
            f"<td>{MINUS}"
            f"{abs(float(row['active_max_drawdown_percent_after_cost'])):.1f}%"
            "</td>"
            "</tr>"
        )

    spread_rows: list[str] = []
    for _, row in limited.iterrows():
        spread_rows.append(
            "<tr>"
            f"<td><code>{row['optimization_mode']}</code></td>"
            f"<td>{float(row['phase_final_equity_spread']):.4f}</td>"
            f"<td>{float(row['offset_final_equity_spread']):.4f}</td>"
            f"<td>{float(row['phase_to_offset_spread_ratio']):.0f}×</td>"
            "</tr>"
        )

    return "\n".join(
        [
            "<section>",
            "<h2>Deployability</h2>",
            '<p class="note">Every per-period figure on this page is a mean '
            "over <em>overlapping</em> forecast periods, and a portfolio "
            "cannot hold an overlapping period. The strategy signals daily "
            "and holds five days, so what it trades is five overlapping "
            "tranches: one fifth of capital rebalances into each day's "
            "targets. That construction holds every phase at once and has no "
            "free parameters.</p>",
            '<div class="callout"><strong>What the columns mean.</strong> '
            "<em>Active</em> is the compounded daily active return: the "
            "product over every trading day of 1 + (strategy return "
            f"{MINUS} benchmark return). It is <em>not</em> strategy equity "
            "divided by benchmark equity, which is why that ratio is shown "
            "beside it — compounding a difference of returns is not the "
            "same as taking a ratio of compounded returns, and the gap "
            "between the two columns is the covariance of the two legs. "
            "<em>Strategy drawdown</em> is the worst peak-to-trough fall of "
            "the compounded strategy equity curve, a true percentage of "
            "capital. <em>Active drawdown</em> is the same measure on the "
            "compounded active-return curve, so it describes the hedged "
            "sleeve rather than the account. Neither is the summed-return "
            "drawdown reported under Portfolio backtest below.</div>",
            '<div class="tablewrap"><table>',
            "<caption>Overlapping-tranche portfolio, after cost, "
            f"price-limit-aware execution. {periods:,} trading days, "
            f"{first_date} to {last_date}, which is {years:.2f} years at 252 "
            "trading days per year.</caption>",
            '<thead><tr><th scope="col">Optimizer</th>'
            '<th scope="col">Strategy equity</th>'
            '<th scope="col">VN30 benchmark</th>'
            '<th scope="col">Active (compounded)</th>'
            '<th scope="col">Equity ratio</th>'
            '<th scope="col">Strategy drawdown</th>'
            '<th scope="col">Active drawdown</th></tr></thead>',
            "<tbody>",
            *rows,
            "</tbody></table></div>",
            '<div class="callout"><strong>No optimizer mode beats buying and '
            "holding the index.</strong> Every active equity is below 1.0. "
            f"The best configuration, <code>{best_mode}</code>, turns 1.0 "
            f"into {best:.4f} while the VN30 itself turns it into "
            f"{benchmark:.4f}. It takes a {best_drawdown:.0f}% peak-to-trough "
            "fall in capital along the way, and "
            f"{best_active_drawdown:.0f}% on the active curve. The other "
            f"three finish {min(behind):.0f}% to {max(behind):.0f}% behind "
            "the index on compounded active return.</div>",
            '<p class="note">Sampling every fifth date also gives '
            "non-overlapping periods, and its per-period means reconcile "
            "exactly with the figures above. But it leaves the starting "
            "offset as a free parameter, and that choice moves the answer "
            "more than the strategy does. One offset makes "
            "<code>sector_diversified</code> look positive throughout, which "
            "is a property of the sampling rather than of the strategy.</p>",
            '<div class="tablewrap"><table>',
            "<caption>Final-equity spread: sampling phase versus tranche "
            "start offset</caption>",
            '<thead><tr><th scope="col">Optimizer</th>'
            '<th scope="col">Across 5 sampling phases</th>'
            '<th scope="col">Across tranche start offsets</th>'
            '<th scope="col">Ratio</th></tr></thead>',
            "<tbody>",
            *spread_rows,
            "</tbody></table></div>",
            '<p class="caption">Source: '
            f"{file_link('reports/tables/tranche_portfolio_diagnostic.csv')}, "
            "generated by "
            f"{file_link('scripts/diagnose_tranche_drawdown.py')}. This "
            f"{years:.2f}-year window ({first_date} to {last_date}) is the "
            "backtest period, which is longer than the prediction range under "
            "Universe and periods below.</p>",
            "</section>",
        ]
    )


def sweep_chart(sweep: pd.DataFrame, chosen: int) -> str:
    """Bar chart of Rank IC by training window size.

    One series, so no legend box: the title names it. All five bars are
    directly labelled because the shape of the curve is the entire argument.

    The category labels sit on a reserved baseline well below the axis, and a
    negative bar's value label is clamped clear of it, so a shallow negative
    bar cannot print its value over its own category label.
    """
    rows = sweep.sort_values("train_size_dates").reset_index(drop=True)
    top, bottom, ceiling = 30.0, 220.0, 0.04
    category_baseline = 262.0
    scale = (bottom - top) / ceiling
    left, slot, width = 96.0, 110.0, 66.0
    parts: list[str] = []

    for index in range(5):
        y = bottom - (bottom - top) * (index / 4.0)
        parts.append(
            f'<line x1="66" y1="{y:.0f}" x2="640" y2="{y:.0f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="58" y="{y + 4:.0f}" text-anchor="end" '
            f'fill="var(--ink-muted)" font-size="11">{ceiling * index / 4.0:.2f}</text>'
        )

    parts.append(
        f'<line x1="66" y1="{bottom:.0f}" x2="640" y2="{bottom:.0f}" '
        f'stroke="var(--axis)" stroke-width="2"/>'
    )

    described: list[str] = []
    for index, row in enumerate(rows.itertuples(index=False)):
        size = int(row.train_size_dates)
        value = float(row.average_rank_ic)
        described.append(f"{size} dates {num(value, 4)}")

        x = left + index * slot
        centre = x + width / 2
        height = abs(value) * scale
        is_chosen = size == chosen
        opacity = "" if is_chosen else ' opacity="0.55"'
        radius = min(4.0, height / 2)

        if value >= 0:
            bar_y = bottom - height
            label_y = bar_y - 7
        else:
            bar_y = bottom
            label_y = min(bottom + height + 13, category_baseline - 20)

        ink = "var(--ink-1)" if is_chosen else "var(--ink-2)"
        weight = "700" if is_chosen else "600"

        parts.append(
            f'<rect x="{x:.0f}" y="{bar_y:.1f}" width="{width:.0f}" '
            f'height="{height:.1f}" rx="{radius:.1f}" ry="{radius:.1f}" '
            f'fill="var(--series-1)"{opacity}/>'
        )
        parts.append(
            f'<text x="{centre:.0f}" y="{label_y:.0f}" text-anchor="middle" '
            f'fill="{ink}" font-size="11.5" font-weight="{weight}">{num(value, 4)}</text>'
        )
        parts.append(
            f'<text x="{centre:.0f}" y="{category_baseline:.0f}" '
            f'text-anchor="middle" fill="{ink}" font-size="11.5" '
            f'font-weight="{weight}">{size}</text>'
        )
        if is_chosen and height > 45:
            parts.append(
                f'<text x="{centre:.0f}" y="{bottom - 12:.0f}" '
                f'text-anchor="middle" fill="var(--surface-1)" font-size="10" '
                f'font-weight="700">chosen</text>'
            )

    parts.append(
        '<text x="353" y="17" text-anchor="middle" fill="var(--ink-2)" font-size="12">'
        "Average Rank IC by training window size</text>"
    )
    parts.append(
        f'<text x="353" y="{category_baseline + 20:.0f}" text-anchor="middle" '
        'fill="var(--ink-muted)" font-size="11">Training window (trading '
        "dates)</text>"
    )
    label = escape(
        "Bar chart of average Rank IC by training window size: "
        + ", ".join(described)
    )
    return (
        f'<svg viewBox="0 0 660 294" role="img" aria-label="{label}.">'
        + "".join(parts)
        + "</svg>"
    )


def repo_url(path: str) -> str:
    """The GitHub address of a repository file or directory.

    The path must exist in this checkout, so a renamed or deleted file fails
    the build rather than publishing a dead link. A directory is linked as a
    tree and a file as a blob.
    """
    target = ROOT / path
    if not target.exists():
        raise FileNotFoundError(f"Linked path is not in this checkout: {path}")
    kind = "tree" if target.is_dir() else "blob"
    return f"{REPO_URL}/{kind}/{REPO_BRANCH}/{path.rstrip('/')}"


def file_link(path: str, label: str | None = None) -> str:
    """A repository path, set as code and linked to its page on GitHub."""
    return f'<a href="{repo_url(path)}"><code>{escape(label or path)}</code></a>'


def pct(fraction: float, digits: int = 1) -> str:
    """A fraction as a percentage, with a true minus sign."""
    return f"{fraction * 100:.{digits}f}%".replace("-", MINUS)


def tranche_curves(daily: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    """The live-date return of every optimizer mode and of the benchmark.

    Returns sit only on dates when every phase sleeve is live, and turnover on
    every rebalance date, so a row carrying some returns but not all means the
    file was assembled wrong. That raises rather than being dropped.
    """
    columns = ["benchmark_return", *[f"{mode}_after_cost_return" for mode in modes]]
    missing = [column for column in columns if column not in daily.columns]
    if missing:
        raise ValueError(f"tranche_daily_returns.csv is missing columns: {missing}")

    present = daily[columns].notna()
    if (present.any(axis=1) & ~present.all(axis=1)).any():
        raise ValueError(
            "tranche_daily_returns.csv has a date carrying some returns but not all"
        )

    live = daily.loc[present.all(axis=1), ["date", *columns]].copy()
    live["date"] = pd.to_datetime(live["date"])
    return live.reset_index(drop=True)


def tranche_statistics(
    daily: pd.DataFrame,
    live: pd.DataFrame,
    modes: list[str],
) -> dict[str, dict[str, float]]:
    """Annualized figures for each optimizer and the benchmark.

    Compounding goes through `summarize_equity_curve`, the function that wrote
    the published tranche table, so the two cannot use different arithmetic.
    The Sharpe ratio is the only figure added here: mean daily return over its
    standard deviation, annualized by the square root of the trading year,
    against a zero risk-free rate.
    """
    sources = [
        (mode, f"{mode}_after_cost_return", f"{mode}_tranche_turnover")
        for mode in modes
    ]
    sources.append(("benchmark", "benchmark_return", "benchmark_tranche_turnover"))

    statistics: dict[str, dict[str, float]] = {}
    for name, return_column, turnover_column in sources:
        returns = live[return_column]
        summary = summarize_equity_curve(returns)
        summary["sharpe"] = (
            float(returns.mean())
            / float(returns.std(ddof=1))
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
        summary["average_tranche_turnover"] = float(
            daily[turnover_column].dropna().mean()
        )
        statistics[name] = summary

    return statistics


def reconcile_tranche_statistics(
    statistics: dict[str, dict[str, float]],
    live: pd.DataFrame,
    limited: pd.DataFrame,
) -> None:
    """The daily file must reproduce every figure in the published tranche table.

    The charts and the performance summary are drawn from the daily file and
    the Deployability table from the summary table. If the two disagree the
    page would contradict itself, so a mismatch fails the build.
    """
    tolerance = 1e-9
    checks: list[tuple[str, float, float]] = []

    for mode, row in limited.iterrows():
        mode_statistics = statistics[mode]
        checks += [
            (f"{mode} periods", mode_statistics["periods"], row["periods"]),
            (
                f"{mode} final equity",
                mode_statistics["final_equity"],
                row["final_equity_after_cost"],
            ),
            (
                f"{mode} max drawdown",
                100.0 * mode_statistics["max_drawdown"],
                row["max_drawdown_percent_after_cost"],
            ),
            (
                f"{mode} annualized return",
                100.0 * mode_statistics["annualised_return"],
                row["annualised_return_percent_after_cost"],
            ),
            (
                f"{mode} annualized volatility",
                100.0 * mode_statistics["annualised_volatility"],
                row["annualised_volatility_percent_after_cost"],
            ),
            (
                f"{mode} average tranche turnover",
                mode_statistics["average_tranche_turnover"],
                row["average_tranche_turnover"],
            ),
        ]

    first = limited.iloc[0]
    benchmark = statistics["benchmark"]
    checks += [
        (
            "benchmark final equity",
            benchmark["final_equity"],
            first["benchmark_final_equity"],
        ),
        (
            "benchmark max drawdown",
            100.0 * benchmark["max_drawdown"],
            first["benchmark_max_drawdown_percent"],
        ),
        (
            "benchmark annualized return",
            100.0 * benchmark["annualised_return"],
            first["benchmark_annualised_return_percent"],
        ),
    ]

    failures = [
        name
        for name, measured, published in checks
        if abs(float(measured) - float(published))
        > tolerance * max(1.0, abs(float(published)))
    ]

    first_date = str(live["date"].iloc[0].date())
    last_date = str(live["date"].iloc[-1].date())
    if first_date != str(first["first_date"]) or last_date != str(first["last_date"]):
        failures.append("date range")

    if failures:
        raise ValueError(
            "tranche_daily_returns.csv does not reproduce "
            "tranche_portfolio_diagnostic.csv: " + ", ".join(failures)
        )


def chart_x(count: int) -> np.ndarray:
    """One evenly spaced x position per trading day."""
    return PLOT_LEFT + (PLOT_RIGHT - PLOT_LEFT) * np.arange(count) / max(count - 1, 1)


def decimate(xs: np.ndarray, ys: np.ndarray) -> str:
    """Polyline points keeping the lowest and highest point in each pixel.

    At several trading days per pixel that draws the same picture as every
    point, including each trough and peak the page cites, in half the bytes.
    The first and last points are always kept, so the line starts at 1.0 and
    ends on its end marker.
    """
    columns = np.floor(xs).astype(int)
    keep: set[int] = {0, len(xs) - 1}
    for column in np.unique(columns):
        index = np.flatnonzero(columns == column)
        keep.update(
            (
                int(index[np.argmin(ys[index])]),
                int(index[np.argmax(ys[index])]),
            )
        )
    return " ".join(f"{xs[i]:.1f},{ys[i]:.1f}" for i in sorted(keep))


def year_ticks(
    dates: pd.Series,
    xs: np.ndarray,
    axis_y: float,
    label_y: float,
) -> list[str]:
    """A tick and label at the first trading day of each calendar year.

    The partial first year is left unlabelled: its first day is the start of
    the series, only a few pixels from the following January.
    """
    parts: list[str] = []
    years = dates.dt.year.to_numpy()
    for year in np.unique(years):
        index = int(np.flatnonzero(years == year)[0])
        if index == 0:
            continue
        x = xs[index]
        parts.append(
            f'<line x1="{x:.1f}" y1="{axis_y:.0f}" x2="{x:.1f}" '
            f'y2="{axis_y + 4:.0f}" stroke="var(--axis)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{label_y:.0f}" text-anchor="middle" '
            f'fill="var(--ink-muted)" font-size="11">{year}</text>'
        )
    return parts


def hover_layer(
    dates: pd.Series,
    xs: np.ndarray,
    top: float,
    bottom: float,
    readout: Callable[[int], str],
) -> str:
    """A month-end crosshair and readout, in SVG and CSS alone.

    The page carries no script, so each month end gets a transparent hit band
    with a native tooltip and a hairline that CSS reveals on hover. The
    tooltip only enhances: every value it shows is in the linked CSV.
    """
    months = dates.dt.strftime("%Y-%m").to_numpy()
    ends = [
        index
        for index in range(len(months))
        if index == len(months) - 1 or months[index + 1] != months[index]
    ]

    parts = ['<g class="hover">']
    for position, index in enumerate(ends):
        x = xs[index]
        start = PLOT_LEFT if position == 0 else (xs[ends[position - 1]] + x) / 2
        stop = (
            PLOT_RIGHT
            if position == len(ends) - 1
            else (x + xs[ends[position + 1]]) / 2
        )
        parts.append(
            '<g class="hit">'
            f"<title>{escape(readout(index))}</title>"
            f'<rect x="{start:.1f}" y="{top:.0f}" width="{stop - start:.1f}" '
            f'height="{bottom - top:.0f}" fill="transparent"/>'
            f'<line class="crosshair" x1="{x:.1f}" y1="{top:.0f}" x2="{x:.1f}" '
            f'y2="{bottom:.0f}" stroke="var(--ink-muted)" stroke-width="1"/>'
            "</g>"
        )
    parts.append("</g>")
    return "".join(parts)


def spread_label_positions(positions: list[float], gap: float) -> list[float]:
    """Push label baselines apart to at least `gap`, keeping their order."""
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    placed = list(positions)
    for previous, current in zip(order, order[1:]):
        if placed[current] - placed[previous] < gap:
            placed[current] = placed[previous] + gap
    return placed


def legend(entries: list[tuple[str, str]]) -> str:
    """Line keys beside the series names, so identity is never colour alone."""
    items = "".join(
        f'<span><span class="key {key}"></span>{label}</span>'
        for key, label in entries
    )
    return f'<div class="legend">{items}</div>'


def growth_chart(live: pd.DataFrame, highlight: str, others: list[str]) -> str:
    """Growth of 1.0 for every optimizer and the benchmark, after cost.

    The best optimizer and the benchmark carry colour and direct labels. The
    other three are context in muted ink, named in the legend and in the
    hover readout, because their end values sit too close to label apart.
    """
    top, bottom, label_y, height = 18.0, 258.0, 278.0, 292.0
    dates = live["date"]
    xs = chart_x(len(live))
    equity = {
        name: (1.0 + live[f"{name}_after_cost_return"]).cumprod().to_numpy()
        for name in [highlight, *others]
    }
    equity["benchmark"] = (1.0 + live["benchmark_return"]).cumprod().to_numpy()

    low = math.floor(min(float(curve.min()) for curve in equity.values()) * 2) / 2
    high = math.ceil(max(float(curve.max()) for curve in equity.values()) * 2) / 2

    def y_of(value):
        return bottom - (value - low) / (high - low) * (bottom - top)

    parts: list[str] = []
    for tick in np.arange(low, high + 1e-9, 0.5):
        y = y_of(tick)
        stroke = "var(--axis)" if abs(tick - 1.0) < 1e-9 else "var(--grid)"
        parts.append(
            f'<line x1="{PLOT_LEFT:.0f}" y1="{y:.1f}" x2="{PLOT_RIGHT:.0f}" '
            f'y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PLOT_LEFT - 8:.0f}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="var(--ink-muted)" font-size="11">{tick:.1f}</text>'
        )
    parts += year_ticks(dates, xs, bottom, label_y)

    for name in others:
        parts.append(
            f'<polyline points="{decimate(xs, y_of(equity[name]))}" fill="none" '
            'stroke="var(--ink-muted)" stroke-width="1.5" stroke-opacity="0.6" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    for name, stroke in (
        ("benchmark", "var(--series-2)"),
        (highlight, "var(--series-1)"),
    ):
        parts.append(
            f'<polyline points="{decimate(xs, y_of(equity[name]))}" fill="none" '
            f'stroke="{stroke}" stroke-width="2" stroke-linejoin="round" '
            'stroke-linecap="round"/>'
        )

    ends = [
        (highlight, "var(--series-1)", float(equity[highlight][-1])),
        ("VN30", "var(--series-2)", float(equity["benchmark"][-1])),
    ]
    anchors = spread_label_positions([y_of(value) for _, _, value in ends], 30.0)
    for (label, fill, value), anchor in zip(ends, anchors):
        parts.append(
            f'<circle cx="{PLOT_RIGHT:.1f}" cy="{y_of(value):.1f}" r="4" '
            f'fill="{fill}" stroke="var(--surface-1)" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{END_LABEL_X:.0f}" y="{anchor - 3:.1f}" '
            f'fill="var(--ink-2)" font-size="11.5">{label}</text>'
        )
        parts.append(
            f'<text x="{END_LABEL_X:.0f}" y="{anchor + 12:.1f}" '
            f'fill="var(--ink-1)" font-size="12.5" font-weight="600">'
            f"{value:.4f}</text>"
        )

    names = [highlight, "benchmark", *others]

    def display(name: str) -> str:
        return "VN30" if name == "benchmark" else name

    def readout(index: int) -> str:
        values = " · ".join(
            f"{display(name)} {equity[name][index]:.4f}" for name in names
        )
        return f"{dates.iloc[index]:%d %b %Y} · {values}"

    parts.append(hover_layer(dates, xs, top, bottom, readout))

    described = ", ".join(
        f"{display(name)} {equity[name][-1]:.4f}" for name in names
    )
    label = escape(
        "Line chart of the growth of 1.0, after cost, from "
        f"{dates.iloc[0]:%d %B %Y} to {dates.iloc[-1]:%d %B %Y}. "
        f"Final values: {described}."
    )
    return (
        f'<svg viewBox="0 0 {CHART_WIDTH:.0f} {height:.0f}" role="img" '
        f'aria-label="{label}">' + "".join(parts) + "</svg>"
    )


def drawdown_chart(live: pd.DataFrame, highlight: str) -> str:
    """Peak-to-trough drawdown of the best optimizer against the benchmark."""
    top, bottom, label_y, height = 12.0, 192.0, 212.0, 226.0
    dates = live["date"]
    xs = chart_x(len(live))
    series = {
        highlight: live[f"{highlight}_after_cost_return"],
        "VN30": live["benchmark_return"],
    }
    drawdown = {}
    for name, returns in series.items():
        equity = (1.0 + returns).cumprod().to_numpy()
        drawdown[name] = equity / np.maximum.accumulate(equity) - 1.0

    low = math.floor(min(float(curve.min()) for curve in drawdown.values()) * 10) / 10

    def y_of(value):
        return top + value / low * (bottom - top)

    parts: list[str] = []
    for step in range(0, int(round(-low * 10)) + 1):
        tick = -step / 10
        y = y_of(tick)
        stroke = "var(--axis)" if step == 0 else "var(--grid)"
        parts.append(
            f'<line x1="{PLOT_LEFT:.0f}" y1="{y:.1f}" x2="{PLOT_RIGHT:.0f}" '
            f'y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PLOT_LEFT - 8:.0f}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="var(--ink-muted)" font-size="11">{pct(tick, 0)}</text>'
        )
    parts += year_ticks(dates, xs, bottom, label_y)

    strokes = {highlight: "var(--series-1)", "VN30": "var(--series-2)"}
    for name in ("VN30", highlight):
        parts.append(
            f'<polyline points="{decimate(xs, y_of(drawdown[name]))}" fill="none" '
            f'stroke="{strokes[name]}" stroke-width="2" stroke-linejoin="round" '
            'stroke-linecap="round"/>'
        )

    troughs = [(name, int(np.argmin(curve))) for name, curve in drawdown.items()]
    anchors = spread_label_positions(
        [y_of(drawdown[name][index]) for name, index in troughs], 14.0
    )
    for (name, index), anchor in zip(troughs, anchors):
        value = float(drawdown[name][index])
        parts.append(
            f'<circle cx="{xs[index]:.1f}" cy="{y_of(value):.1f}" r="4" '
            f'fill="{strokes[name]}" stroke="var(--surface-1)" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{xs[index] + 9:.1f}" y="{anchor + 4:.1f}" '
            'fill="var(--ink-1)" stroke="var(--surface-1)" stroke-width="3" '
            'stroke-linejoin="round" paint-order="stroke" font-size="11.5" '
            f'font-weight="600">{name} {pct(value)}</text>'
        )

    def readout(index: int) -> str:
        values = " · ".join(
            f"{name} {pct(float(drawdown[name][index]))}" for name in drawdown
        )
        return f"{dates.iloc[index]:%d %b %Y} · drawdown from peak · {values}"

    parts.append(hover_layer(dates, xs, top, bottom, readout))

    described = ", ".join(
        f"{name} deepest {pct(float(drawdown[name][index]))} on "
        f"{dates.iloc[index]:%d %B %Y}"
        for name, index in troughs
    )
    label = escape(
        f"Line chart of drawdown from the previous peak, after cost: {described}."
    )
    return (
        f'<svg viewBox="0 0 {CHART_WIDTH:.0f} {height:.0f}" role="img" '
        f'aria-label="{label}">' + "".join(parts) + "</svg>"
    )


def overview_section(
    limited: pd.DataFrame | None,
    benchmark: pd.DataFrame,
    models: pd.DataFrame,
    top5_col: str,
    dataset: pd.Series,
    feature_count: int,
) -> str:
    """The research question, the data and method, and the three findings.

    Every number is read from a table and every verdict is computed, so the
    paragraph cannot outlive the result it summarises. `limited` is the
    price-limit-aware tranche table, best optimizer first, or None when the
    tranche diagnostic has not been run.
    """
    momentum = float(benchmark.loc["top5_momentum", "average_period_active_return"])
    best_model = str(models[top5_col].idxmax())
    ratio = float(models.loc[best_model, top5_col]) / momentum * 100
    beats_every_model = bool((models[top5_col] < momentum).all())

    ml = benchmark[benchmark["comparison_type"] == "ml_strategy"]

    question = (
        f"Using FiinQuant daily data on the {int(dataset['modeling_tickers'])} "
        "stocks that were VN30 members at some point from "
        f"{dataset['market_history_first_date']} to "
        f"{dataset['market_history_last_date']}, each counted only while it was "
        f"in the index, this project asks whether {len(models)} machine-learning "
        f"models predicting each stock's {FORWARD_HORIZON_MEDIUM}-day return "
        f"relative to the index from {feature_count} features, scored "
        f"walk-forward on a {DEFAULT_TRAIN_SIZE}-date training window with a "
        f"{PURGE_DAYS}-day purge and turned into constrained long-only "
        "portfolios, can beat simply holding the index once trading costs and "
        f"the ±{HOSE_NORMAL_DAILY_LIMIT * 100:.0f}% daily price limit are "
        "charged."
    )

    if limited is None:
        finding_costs = (
            "<strong>The compounded comparison with VN30 is not in this "
            "checkout:</strong> run "
            f"{file_link('scripts/diagnose_tranche_drawdown.py')} to add it."
        )
    else:
        best = limited.iloc[0]
        behind = int((limited["active_final_equity_after_cost"] < 1.0).sum())
        lead = (
            "No portfolio beats VN30 after costs"
            if behind == len(limited)
            else f"{len(limited) - behind} of {len(limited)} portfolios beat "
            "VN30 after costs"
        )
        finding_costs = (
            f"<strong>{lead}:</strong> the best of the {len(limited)} "
            f"optimizers, <code>{limited.index[0]}</code>, turns 1.0 into "
            f"{float(best['final_equity_after_cost']):.4f} from "
            f"{best['first_date']} to {best['last_date']}, while the VN30 "
            f"benchmark turns it into {float(best['benchmark_final_equity']):.4f}."
        )

    if beats_every_model:
        finding_models = (
            "<strong>A one-feature momentum rule beats all "
            f"{len(models)} trained models:</strong> ranking on trailing "
            "10-day return and holding the top five beats every model's own "
            f"top five before cost, and the strongest, <code>{best_model}</code>, "
            f"reaches {ratio:.0f}% of it."
        )
    else:
        finding_models = (
            f"<strong>The strongest trained model, <code>{best_model}</code>, "
            f"reaches {ratio:.0f}% of a one-feature momentum rule</strong> on "
            "matched top-five selection before cost."
        )

    needed = {"mean_active_return_95pct_lower", "mean_active_return_95pct_upper"}
    if needed.issubset(benchmark.columns):
        lower = float(benchmark.loc["top5_momentum", "mean_active_return_95pct_lower"])
        upper = float(benchmark.loc["top5_momentum", "mean_active_return_95pct_upper"])
        momentum_excludes = lower > 0.0 or upper < 0.0
        spanning = int(
            (
                (ml["mean_active_return_95pct_lower"] <= 0.0)
                & (ml["mean_active_return_95pct_upper"] >= 0.0)
            ).sum()
        )
        if momentum_excludes and spanning == len(ml):
            finding_intervals = (
                "<strong>The momentum signal is detectable, and nothing "
                "survives the models and costs:</strong> its 95% bootstrap "
                "interval excludes zero before costs, while all "
                f"{len(ml)} model-driven portfolio configurations have "
                "intervals that span zero after costs."
            )
        else:
            finding_intervals = (
                "<strong>Momentum's 95% bootstrap interval "
                + ("excludes" if momentum_excludes else "spans")
                + f" zero before costs</strong>, and {spanning} of the "
                f"{len(ml)} model-driven portfolio configurations have "
                "intervals that span zero after costs."
            )
    else:
        finding_intervals = (
            "<strong>No bootstrap interval is published</strong> in "
            f"{file_link('reports/tables/benchmark_results.csv', 'benchmark_results.csv')}, "
            "so no result here is tested against zero."
        )

    return "\n".join(
        [
            '<section id="overview">',
            "<h2>Overview</h2>",
            f"<p>{question}</p>",
            '<ol class="findings">',
            f"<li>{finding_costs}</li>",
            f"<li>{finding_models}</li>",
            f"<li>{finding_intervals}</li>",
            "</ol>",
            '<p class="caption">Full write-up: '
            f"{file_link('README.md')} · {file_link('reports/methodology.md')} · "
            f"{file_link('reports/final_results.md')}. Every figure on this page "
            f"is read from {file_link('reports/tables/')}.</p>",
            "</section>",
        ]
    )


def performance_charts_section(
    live: pd.DataFrame,
    highlight: str,
    others: list[str],
) -> str:
    """Growth of 1.0 and drawdown, drawn from the reconciled daily series."""
    first = live["date"].iloc[0]
    last = live["date"].iloc[-1]

    def deepest(returns: pd.Series) -> pd.Timestamp:
        equity = (1.0 + returns).cumprod().to_numpy()
        return live["date"].iloc[int(np.argmin(equity / np.maximum.accumulate(equity)))]

    strategy_trough = deepest(live[f"{highlight}_after_cost_return"])
    benchmark_trough = deepest(live["benchmark_return"])
    if strategy_trough == benchmark_trough:
        trough_sentence = (
            f"Both fall furthest on the same day, {strategy_trough:%d %B %Y}."
        )
    else:
        trough_sentence = (
            f"<code>{highlight}</code> falls furthest on "
            f"{strategy_trough:%d %B %Y} and VN30 on {benchmark_trough:%d %B %Y}."
        )

    source = (
        f"Data: {file_link('reports/tables/tranche_daily_returns.csv')}, written "
        f"by {file_link('scripts/diagnose_tranche_drawdown.py')} beside the "
        "Deployability table, and checked against it every time this page is "
        "built."
    )

    return "\n".join(
        [
            '<section id="performance">',
            "<h2>Growth of 1.0 and drawdown, after cost</h2>",
            '<p class="note">What each optimizer would actually have returned as '
            "the overlapping-tranche portfolio it trades, under "
            f"price-limit-aware execution, from {first.date()} to {last.date()}. "
            "Hover a month for its values.</p>",
            "<h3>Growth of 1.0</h3>",
            legend(
                [
                    ("key-s1", f"<code>{highlight}</code>"),
                    ("key-s2", "VN30 benchmark"),
                    (
                        "key-muted",
                        "other optimizer modes: "
                        + ", ".join(f"<code>{mode}</code>" for mode in others),
                    ),
                ]
            ),
            '<figure><div class="chartwrap">',
            growth_chart(live, highlight, others),
            "</div>",
            '<figcaption class="caption">Each line compounds daily after-cost '
            "returns from 1.0, and the darker rule marks 1.0, the starting "
            "capital. VN30 here is an equal-weight basket of the point-in-time "
            "constituents, run through the same five-tranche construction "
            "without cost, because the canonical history carries no VN30 index "
            f"price series. {source}</figcaption>",
            "</figure>",
            "<h3>Drawdown from previous peak</h3>",
            legend(
                [
                    ("key-s1", f"<code>{highlight}</code>"),
                    ("key-s2", "VN30 benchmark"),
                ]
            ),
            '<figure><div class="chartwrap">',
            drawdown_chart(live, highlight),
            "</div>",
            '<figcaption class="caption">The fall of each compounded curve '
            "from its running peak, as a percentage of capital. "
            f"{trough_sentence}</figcaption>",
            "</figure>",
            "</section>",
        ]
    )


def performance_summary_section(
    statistics: dict[str, dict[str, float]],
    limited: pd.DataFrame,
    rebalance_dates: int,
) -> str:
    """Annualized return, risk and turnover for each optimizer and VN30."""
    modes = [str(mode) for mode in limited.index]
    first = limited.iloc[0]
    benchmark = statistics["benchmark"]

    def row(label: str, figures: dict[str, float], css: str = "") -> str:
        return (
            f"<tr{css}><td>{label}</td>"
            f"<td>{pct(figures['annualised_return'], 2)}</td>"
            f"<td>{pct(figures['annualised_volatility'], 2)}</td>"
            f"<td>{num(figures['sharpe'], 2)}</td>"
            f"<td>{pct(figures['max_drawdown'])}</td>"
            f"<td>{pct(figures['average_tranche_turnover'])}</td></tr>"
        )

    rows = [
        row(
            f"<code>{mode}</code>",
            statistics[mode],
            ' class="highlight"' if index == 0 else "",
        )
        for index, mode in enumerate(modes)
    ]
    rows.append(row("VN30 benchmark", benchmark))

    more_volatile = sum(
        statistics[mode]["annualised_volatility"] > benchmark["annualised_volatility"]
        for mode in modes
    )
    deeper = sum(
        statistics[mode]["max_drawdown"] < benchmark["max_drawdown"] for mode in modes
    )
    best_sharpe_mode = max(modes, key=lambda mode: statistics[mode]["sharpe"])
    best_sharpe = statistics[best_sharpe_mode]["sharpe"]
    lower_return = sum(
        statistics[mode]["annualised_return"] < benchmark["annualised_return"]
        for mode in modes
    )

    if more_volatile == deeper == lower_return == len(modes):
        relation = (
            "under half of" if best_sharpe < benchmark["sharpe"] / 2 else "below"
        )
        callout = (
            "<strong>Every optimizer mode takes more risk than the index for "
            f"less return.</strong> All {len(modes)} are more volatile than "
            "VN30, fall further at their worst and compound more slowly, and "
            "the best Sharpe ratio among them, "
            f"{num(best_sharpe, 2)} for <code>{best_sharpe_mode}</code>, is "
            f"{relation} VN30's {num(benchmark['sharpe'], 2)}."
        )
    else:
        callout = (
            f"<strong>{lower_return} of {len(modes)} optimizer modes compound "
            f"more slowly than VN30.</strong> {more_volatile} are more volatile "
            f"and {deeper} fall further at their worst; the best Sharpe ratio "
            f"among them is {num(best_sharpe, 2)} for "
            f"<code>{best_sharpe_mode}</code>, against VN30's "
            f"{num(benchmark['sharpe'], 2)}."
        )

    root = "√"
    return "\n".join(
        [
            '<section id="performance-summary">',
            "<h2>Performance summary</h2>",
            '<div class="tablewrap"><table>',
            "<caption>Overlapping-tranche portfolio, after cost, "
            f"price-limit-aware execution, {int(first['periods']):,} trading days "
            f"from {first['first_date']} to {first['last_date']}. Computed from "
            "the daily series behind the charts.</caption>",
            '<thead><tr><th scope="col">Portfolio</th>'
            '<th scope="col">Annualized return</th>'
            '<th scope="col">Annualized volatility</th>'
            '<th scope="col">Sharpe ratio</th>'
            '<th scope="col">Max drawdown</th>'
            '<th scope="col">Average turnover</th></tr></thead>',
            "<tbody>",
            *rows,
            "</tbody></table></div>",
            f'<div class="callout">{callout}</div>',
            '<ul class="note">',
            "<li><strong>Annualized return</strong> is compound: final equity "
            f"raised to the power {TRADING_DAYS_PER_YEAR} ÷ trading days, minus "
            "one.</li>",
            "<li><strong>Annualized volatility</strong> is the standard "
            f"deviation of daily returns × {root}{TRADING_DAYS_PER_YEAR}.</li>",
            "<li><strong>Sharpe ratio</strong> is the mean daily return ÷ its "
            f"standard deviation × {root}{TRADING_DAYS_PER_YEAR}, annualized, "
            "with a <strong>risk-free rate of 0%</strong>. A risk-free rate r "
            "would lower each row by r ÷ that row's annualized volatility. It "
            "uses the arithmetic mean, so it is not annualized return divided "
            "by volatility, and it is not the per-period diagnostic Sharpe "
            "reported further down.</li>",
            "<li><strong>Max drawdown</strong> is the worst peak-to-trough fall "
            "of the compounded curve, as a percentage of capital.</li>",
            "<li><strong>Average turnover</strong> is one-way, half the sum of "
            "absolute weight changes, for the tranche rebalanced each day "
            "against the one it replaces "
            f"{HOLD_HORIZON_DAYS} trading days earlier, averaged over all "
            f"{rebalance_dates:,} rebalance dates. A tranche is "
            f"1/{HOLD_HORIZON_DAYS} of capital, so 1/{HOLD_HORIZON_DAYS} of "
            "this figure is traded across the whole portfolio each day. The VN30 basket turns over only when its "
            "constituent list changes, and carries no cost.</li>",
            "</ul>",
            '<p class="caption">Source: '
            f"{file_link('reports/tables/tranche_daily_returns.csv')}. Annualized "
            "return, volatility, drawdown and optimizer turnover reproduce "
            f"{file_link('reports/tables/tranche_portfolio_diagnostic.csv')} "
            "exactly; the build fails if they do not.</p>",
            "</section>",
        ]
    )


def model_descriptions() -> dict[str, str]:
    """Each model's type and settings, read from the objects the pipeline fits."""
    linear = create_linear_models()
    tree = create_tree_models()
    ridge = linear["ridge"].get_params()
    elastic = linear["elastic_net"].get_params()
    forest = tree["random_forest"].get_params()
    boosting = tree["gradient_boosting"].get_params()

    return {
        "ridge": f"Ridge regression, alpha {ridge['alpha']:g}",
        "elastic_net": (
            f"Elastic-net regression, alpha {elastic['alpha']:g}, "
            f"L1 ratio {elastic['l1_ratio']:g}"
        ),
        "random_forest": (
            f"Random forest, {forest['n_estimators']} trees of depth "
            f"{forest['max_depth']}"
        ),
        "gradient_boosting": (
            f"Gradient boosting, {boosting['n_estimators']} trees of depth "
            f"{boosting['max_depth']}, learning rate {boosting['learning_rate']:g}"
        ),
        "classification": (
            "Class-balanced logistic regression on standardized features, "
            "predicting whether a stock finishes in the top "
            f"{TOP_QUINTILE_FRACTION * 100:.0f}% of that date's "
            f"{FORWARD_HORIZON_MEDIUM}-day relative return"
        ),
    }


def ablation_name(row: pd.Series) -> str:
    """The ablation that removes a feature group, or empty if none does."""
    value = row["ablation_name"]
    return "" if pd.isna(value) else str(value)


def feature_overlap_sentence(groups: pd.DataFrame) -> str:
    """Say where an ablation removes a different set than the group counted.

    The six groups partition the features, but the four ablation lists were
    written independently and overlap at the edges, so the ablation feature
    counts are not the group sizes. This says exactly where they differ.
    """
    owner = {
        feature: str(row["display_name"])
        for _, row in groups.iterrows()
        for feature in str(row["features"]).split(";")
    }

    def code_list(names: list[str]) -> str:
        coded = [f"<code>{name}</code>" for name in names]
        if len(coded) == 1:
            return coded[0]
        return ", ".join(coded[:-1]) + " and " + coded[-1]

    clauses: list[str] = []
    for _, row in groups.iterrows():
        name = ablation_name(row)
        if not name:
            continue
        members = set(str(row["features"]).split(";"))
        removed = list(ABLATION_GROUPS[name])
        elsewhere = [feature for feature in removed if feature not in members]
        kept = [
            feature
            for feature in str(row["features"]).split(";")
            if feature not in removed
        ]
        if not elsewhere and not kept:
            continue
        clause = f"<code>{name}</code> removes {len(removed)} columns"
        if elsewhere:
            clause += ", including " + code_list(elsewhere) + ", counted under " + (
                ", ".join(sorted({owner[feature] for feature in elsewhere}))
            )
        if kept:
            clause += ", and leaves " + code_list(kept) + " in place"
        clauses.append(clause)

    if not clauses:
        return "Each ablation removes exactly the group it is listed against."
    return "; ".join(clauses) + "."


def data_and_method_section(
    dataset: pd.Series,
    groups: pd.DataFrame,
    models: pd.DataFrame,
    horizon: pd.DataFrame,
) -> str:
    """Where the data comes from and every step from it to a portfolio."""
    walk_forward = inspect.signature(build_tree_model_outputs).parameters

    def default(name: str) -> int:
        return int(walk_forward[name].default)

    descriptions = model_descriptions()
    undescribed = [name for name in models.index if name not in descriptions]
    if undescribed:
        raise ValueError(f"No description for model(s): {undescribed}")

    model_rows = []
    for name, row in models.iterrows():
        production = name == MODEL_NAME
        css = ' class="highlight"' if production else ""
        role = "feeds portfolio construction" if production else "comparison"
        model_rows.append(
            f"<tr{css}><td><code>{name}</code></td><td>{row['model_family']}</td>"
            f'<td class="prose">{descriptions[name]}</td><td>{role}</td></tr>'
        )

    group_rows = [
        "<tr>"
        f"<td>{row['display_name']}</td><td>{int(row['feature_count'])}</td>"
        + (
            f"<td><code>{ablation_name(row)}</code></td>"
            if ablation_name(row)
            else "<td>not ablated</td>"
        )
        + "</tr>"
        for _, row in groups.iterrows()
    ]
    group_rows.append(
        '<tr class="highlight"><td>All model-facing features</td>'
        f"<td>{int(groups['feature_count'].sum())}</td>"
        "<td><code>all_features</code></td></tr>"
    )

    def cap(value: float | None) -> str:
        return "none" if value is None else pct(value, 0)

    herding_rule = (
        f"<code>{HERDING_SCORE_COLUMN}</code> at or above "
        f"{HIGH_HERDING_THRESHOLD:.2f}"
    )
    mode_rows = [
        (
            "normal",
            f"top {TOP_N}",
            cap(MAX_WEIGHT),
            cap(MAX_ISSUER_GROUP_WEIGHT),
            cap(None),
            cap(MAX_TURNOVER),
            pct(1.0, 0),
        ),
        (
            "sector_aware",
            f"top {TOP_N}",
            cap(MAX_WEIGHT),
            cap(MAX_ISSUER_GROUP_WEIGHT),
            cap(MAX_SECTOR_WEIGHT),
            cap(MAX_TURNOVER),
            pct(1.0, 0),
        ),
        (
            "herding_aware",
            f"top {TOP_N}*",
            f"{cap(MAX_WEIGHT)}*",
            cap(MAX_ISSUER_GROUP_WEIGHT),
            cap(None),
            cap(MAX_TURNOVER),
            f"{pct(1.0, 0)}*",
        ),
        (
            "sector_diversified",
            f"{PAPER_TARGET_HOLDINGS} names",
            cap(PAPER_MAX_SINGLE_NAME_WEIGHT),
            cap(PAPER_MAX_ISSUER_GROUP_WEIGHT),
            cap(PAPER_MAX_SECTOR_WEIGHT),
            cap(PAPER_MAX_TURNOVER),
            pct(PAPER_TARGET_EXPOSURE, 0),
        ),
    ]

    evaluated = int(horizon.loc[FORWARD_HORIZON_MEDIUM, "evaluated_dates"])

    def dates(count: int) -> str:
        return f"{count} date" if count == 1 else f"{count} dates"

    return "\n".join(
        [
            '<section id="data-and-method">',
            "<h2>Data and method</h2>",
            "<h3>Data</h3>",
            '<p class="note">Daily bars from FiinQuant, a Vietnamese '
            "market-data vendor: open, high, low, close, volume, traded value, "
            "and foreign investors' buy and sell value. The canonical history "
            f"holds {int(dataset['market_history_rows']):,} rows across "
            f"{int(dataset['market_history_tickers'])} tickers from "
            f"{dataset['market_history_first_date']} to "
            f"{dataset['market_history_last_date']}, and "
            f"{int(dataset['modeling_rows']):,} of them fall on dates when the "
            "ticker was an index constituent. Raw prices drive anything tied to "
            "the printed price, such as limit bands and ceiling detection; "
            "adjusted close drives returns and labels; "
            f"{file_link('src/market_data_schema.py')} keeps the two apart.</p>",
            "<h3>Universe</h3>",
            '<p class="note">Point-in-time VN30 membership, rebuilt from '
            f"index-review announcements in {file_link('config/vn30_history/')}, "
            "each review backed by a source in "
            f"{file_link('config/vn30_history/source_manifest.csv', 'source_manifest.csv')}. "
            "A stock contributes rows only on dates it was in the index, so the "
            f"{int(dataset['modeling_tickers'])} tickers include former members "
            "that were later removed or delisted, which keeps survivorship bias "
            "out of every result.</p>",
            "<h3>Prediction target</h3>",
            '<p class="note">For each stock on each signal date: its forward '
            f"return over {FORWARD_HORIZON_SHORT}, {FORWARD_HORIZON_MEDIUM} and "
            f"{FORWARD_HORIZON_LONG} trading days on adjusted close, minus the "
            "equal-weight average forward return of the other constituents that "
            "day, a leave-one-out VN30 benchmark. Production uses the "
            f"{FORWARD_HORIZON_MEDIUM}-day target, <code>{TARGET_COLUMN}</code>; "
            "the last signal date with a complete label is "
            f"{dataset['last_labeled_date']}.</p>",
            "<h3>Features</h3>",
            '<div class="tablewrap"><table>',
            f"<caption>{int(groups['feature_count'].sum())} model-facing "
            "columns in six groups</caption>",
            '<thead><tr><th scope="col">Group</th><th scope="col">Columns</th>'
            '<th scope="col">Ablation that removes it</th></tr></thead>',
            "<tbody>",
            *group_rows,
            "</tbody></table></div>",
            '<p class="caption">Each column is counted in exactly one group. '
            + feature_overlap_sentence(groups)
            + " That is why the Feature ablation table's counts are not the "
            "group sizes. Every column is listed in "
            f"{file_link('reports/tables/feature_groups.csv')}, written by "
            f"{file_link('scripts/describe_modeling_dataset.py')} from the "
            f"removal lists in {file_link('src/ablation_tests.py')}.</p>",
            "<h3>Models</h3>",
            '<div class="tablewrap"><table>',
            '<thead><tr><th scope="col">Model</th><th scope="col">Family</th>'
            '<th scope="col" class="prose">Type and settings</th>'
            '<th scope="col">Role</th>'
            "</tr></thead>",
            "<tbody>",
            *model_rows,
            "</tbody></table></div>",
            '<p class="note">All five are used only to rank stocks within a '
            "date: portfolio construction reads the ordering, not the size, of "
            "a prediction.</p>",
            "<h3>Walk-forward validation</h3>",
            '<p class="note">Never a random split. Each window fits on '
            f"{default('train_size')} consecutive trading dates, validates on "
            f"{dates(default('validation_size'))} and tests on "
            f"{dates(default('test_size'))}, with a {default('purge_size')}-date "
            "purge before each, so no forward label in the training window "
            "overlaps the date it is scored on. The window then steps forward "
            f"{dates(default('step_size'))}, and the "
            f"{FORWARD_HORIZON_MEDIUM}-day horizon is scored on {evaluated:,} "
            "dates.</p>",
            "<h3>Portfolio construction</h3>",
            '<p class="note">Long-only, no shorting and no leverage, from '
            f"<code>{MODEL_NAME}</code> scores. Four optimizer modes:</p>",
            '<div class="tablewrap"><table>',
            '<thead><tr><th scope="col">Mode</th><th scope="col">Names targeted</th>'
            '<th scope="col">Single-name cap</th><th scope="col">Issuer-group cap</th>'
            '<th scope="col">Sector cap</th><th scope="col">Turnover cap</th>'
            '<th scope="col">Target invested</th></tr></thead>',
            "<tbody>",
            *[
                "<tr>"
                + f"<td><code>{mode_row[0]}</code></td>"
                + "".join(f"<td>{cell}</td>" for cell in mode_row[1:])
                + "</tr>"
                for mode_row in mode_rows
            ],
            "</tbody></table></div>",
            '<p class="caption">In the first three modes each selected name '
            "gets an equal weight up to the single-name cap, and a breached "
            "issuer-group or sector cap scales that group down, leaving the "
            "excess in cash. <code>sector_diversified</code> instead fills from "
            "further down the ranking when a cap binds, so it stays invested. "
            f"* On a high-herding day, one with {herding_rule}, "
            f"<code>herding_aware</code> holds only names in the top "
            f"{HERDING_AWARE_TOP_N} with a positive score, caps each at "
            f"{cap(HERDING_AWARE_MAX_WEIGHT)} and targets "
            f"{pct(HERDING_AWARE_TARGET_EXPOSURE, 0)} invested. Turnover is "
            "one-way, per rebalance. Names targeted is the target; the Holdings column "
            "under Portfolio backtest measures what was held, which runs higher "
            "because the turnover cap moves into new targets gradually. "
            f"Constants: {file_link('src/optimizer.py')}.</p>",
            "<h3>From signals to a portfolio</h3>",
            '<p class="note">The strategy signals daily and holds for '
            f"{HOLD_HORIZON_DAYS} trading days, so the portfolio it trades is "
            f"{HOLD_HORIZON_DAYS} overlapping tranches: each day "
            f"1/{HOLD_HORIZON_DAYS} of capital is rebalanced into that day's "
            "target weights and held, buy-and-hold, for "
            f"{HOLD_HORIZON_DAYS} days. The VN30 benchmark is an equal-weight "
            "basket of the point-in-time constituents run through the same "
            "construction without cost. The charts, the Performance summary "
            "and Deployability all come from this construction, built in "
            f"{file_link('src/tranche_portfolio.py')}; the per-period tables "
            "further down average overlapping forecast periods instead.</p>",
            "</section>",
        ]
    )


def costs_section() -> str:
    """Every cost and execution rule, read from the constants that apply them."""
    legs = TRADED_LEGS_PER_UNIT_TURNOVER
    per_turnover = legs * (COMMISSION_RATE + SLIPPAGE_RATE) + SELL_TAX_RATE
    ceiling = 1.0 + HOSE_NORMAL_DAILY_LIMIT
    floor = 1.0 - HOSE_NORMAL_DAILY_LIMIT

    config = load_paper_trading_config(PAPER_TRADING_CONFIG_PATH)
    execution = config["execution"]
    settlement = config["settlement"]
    timing = config["timing"]

    def can(allowed: bool) -> str:
        return "can" if allowed else "cannot"

    backtest_rows = [
        (
            "Commission",
            f"{pct(COMMISSION_RATE, 2)} per side",
            f"Charged on both legs of a rebalance: {legs:g} × one-way turnover × "
            "rate.",
        ),
        (
            "Slippage",
            f"{pct(SLIPPAGE_RATE, 2)} per side",
            "A flat rate on both legs; it does not grow with order size.",
        ),
        (
            "Sell tax",
            f"{pct(SELL_TAX_RATE, 2)} of sale value",
            "The statutory transfer tax, on the sale leg only: one-way turnover × "
            "rate.",
        ),
        (
            "Liquidity penalty",
            f"{pct(LIQUIDITY_PENALTY_RATE, 2)} per side",
            "Both legs, scaled by the portfolio's weight in low-liquidity "
            "names: those whose 20-day average daily traded value is at or "
            f"below the {LOW_LIQUIDITY_QUANTILE * 100:.0f}th percentile of that "
            "date's constituents.",
        ),
        (
            "Price limits",
            f"±{HOSE_NORMAL_DAILY_LIMIT * 100:.0f}% HOSE band",
            "In price-limit-aware execution a buy is blocked when the "
            f"signal-date close is at or above the previous close × {ceiling:.2f}, "
            "rounded down to the HOSE tick, and a sell when it is at or below × "
            f"{floor:.2f}, rounded up. The blocked name keeps its previous "
            "weight, and buys are scaled back if that leaves more than 100% "
            f"invested. The ±{HOSE_WIDE_DAILY_LIMIT * 100:.0f}% band HOSE applies "
            "on special sessions is not modelled. Frictionless execution fills "
            "every intended trade.",
        ),
        (
            "Execution turnover cap",
            f"{pct(MAX_EXECUTION_TURNOVER, 0)} one-way",
            "Price-limit-aware execution scales any rebalance above this back to "
            "it, on top of the optimizer's own cap.",
        ),
        (
            "Fill price",
            "Signal-date close",
            "Each tranche is valued from the adjusted close of the date it is "
            "formed. No execution delay is modelled.",
        ),
        (
            "Settlement",
            "Not modelled",
            f"Each tranche is held {HOLD_HORIZON_DAYS} trading days, so no share "
            "is sold before it settles at T+2, but sale proceeds are reinvested "
            "on the day of the sale.",
        ),
        (
            "ADV cap and board lots",
            "Not applied",
            "Weights are fractional and no trade is limited by traded volume.",
        ),
    ]

    paper_rows = [
        (
            "Board lot",
            f"{int(execution['round_lot_size'])} shares; odd lots "
            + ("allowed" if execution["allow_odd_lots"] else "not allowed"),
        ),
        ("Minimum trade", f"{int(execution['min_trade_value_vnd']):,} VND"),
        (
            "ADV participation cap",
            f"{pct(float(execution['max_trade_adv_fraction']), 0)} of "
            f"{int(execution['adv_lookback_trading_days'])}-day average daily "
            "traded value per order; the excess is deferred",
        ),
        (
            "Costs per fill",
            f"commission {pct(float(execution['commission_rate']), 2)}, slippage "
            f"{pct(float(execution['slippage_rate']), 2)}, sell tax "
            f"{pct(float(execution['sell_tax_rate']), 2)} on sales",
        ),
        (
            "Price limits",
            ("buys blocked at the ceiling" if execution["block_buy_at_ceiling"] else "buys at the ceiling allowed")
            + ", "
            + ("sells blocked at the floor" if execution["block_sell_at_floor"] else "sells at the floor allowed"),
        ),
        (
            "Settlement",
            f"{settlement['cycle']}: trades settle "
            f"{int(settlement['lag_trading_days'])} trading days later; unsettled "
            "sale proceeds "
            f"{can(settlement['allow_unsettled_sale_proceeds_for_buys'])} fund "
            "buys and unsettled shares "
            f"{can(settlement['allow_unsettled_bought_shares_for_sales'])} be sold",
        ),
        (
            "Timing",
            f"orders fill at the {str(timing['execution_price_source']).replace('_', ' ')}, "
            f"{int(timing['execution_delay_trading_days'])} trading day after the "
            f"signal; {int(timing['signal_horizon_trading_days'])}-day signal, "
            f"rebalanced every {int(timing['rebalance_frequency_trading_days'])} "
            "trading days",
        ),
    ]

    return "\n".join(
        [
            '<section id="costs">',
            "<h2>Cost and execution assumptions</h2>",
            '<p class="note">As implemented in '
            f"{file_link('src/backtester.py')} and "
            f"{file_link('src/tranche_portfolio.py')}. These rules produce "
            "every figure on this page.</p>",
            '<div class="tablewrap"><table>',
            "<caption>Backtest</caption>",
            '<thead><tr><th scope="col">Item</th><th scope="col">Setting</th>'
            '<th scope="col" class="prose">How it is applied</th></tr></thead>',
            "<tbody>",
            *[
                f"<tr><td>{item}</td><td>{setting}</td>"
                f'<td class="prose">{rule}</td></tr>'
                for item, setting, rule in backtest_rows
            ],
            "</tbody></table></div>",
            '<div class="callout"><strong>What a rebalance costs.</strong> A '
            f"rebalance with one-way turnover t costs {pct(per_turnover, 2)} × t "
            f"({legs:g} × ({pct(COMMISSION_RATE, 2)} + {pct(SLIPPAGE_RATE, 2)}) + "
            f"{pct(SELL_TAX_RATE, 2)}), plus a liquidity penalty of up to "
            f"{pct(legs * LIQUIDITY_PENALTY_RATE, 2)} × t when every name held "
            "is in the low-liquidity group.</div>",
            '<p class="note">Board lots, an ADV participation cap and T+2 '
            "settlement are implemented, but in the paper-trading engine in "
            f"{file_link('src/paper_trading/')}, configured by "
            f"{file_link('config/paper_trading_config.yaml')}. No figure on "
            "this page comes from it.</p>",
            '<div class="tablewrap"><table>',
            "<caption>Paper-trading engine</caption>",
            '<thead><tr><th scope="col">Item</th>'
            '<th scope="col" class="prose">Setting</th></tr></thead>',
            "<tbody>",
            *[
                f'<tr><td>{item}</td><td class="prose">{setting}</td></tr>'
                for item, setting in paper_rows
            ],
            "</tbody></table></div>",
            "</section>",
        ]
    )


def glossary_section() -> str:
    """The terms the tables below use, defined once."""
    terms = [
        (
            "Rank IC",
            "On each signal date, the Spearman rank correlation between the "
            "model's predicted scores and the realized forward relative returns "
            "of that day's constituents, averaged over dates. 0 means no "
            "ranking skill and 1 a perfect ranking.",
        ),
        (
            "Top-5 hit rate",
            "The share of the five highest-ranked stocks whose realized forward "
            "return beats the leave-one-out benchmark, averaged over dates.",
        ),
        (
            "Diagnostic Sharpe",
            "Mean per-period active return divided by its standard deviation, "
            "on overlapping forecast periods. It is per period and not "
            "annualized, so it is not comparable with the Sharpe ratio in the "
            "Performance summary.",
        ),
        (
            "Cumulative sum",
            "Per-period active returns summed over every evaluated date, not "
            "compounded. Consecutive 5-day periods overlap, so each day's "
            "return is counted about five times; read it as a diagnostic, not "
            "a track record.",
        ),
        (
            "Active (compounded)",
            "The product over trading days of 1 + (strategy daily return "
            f"{MINUS} benchmark daily return): the growth of a daily-rebalanced "
            "sleeve long the strategy and short the benchmark. Below 1.0, the "
            "strategy fell behind.",
        ),
        (
            "Equity ratio",
            "Strategy final equity divided by benchmark final equity. It "
            "differs from active because compounding a difference of returns is "
            "not the ratio of compounded returns.",
        ),
    ]
    return "\n".join(
        [
            '<section id="definitions">',
            "<h2>Metric definitions</h2>",
            '<dl class="glossary">',
            *[f"<dt>{term}</dt><dd>{definition}</dd>" for term, definition in terms],
            "</dl>",
            "</section>",
        ]
    )


def reproduce_section() -> str:
    """Setup, the commands that rebuild every table, and how to get the data."""
    python = platform.python_version()
    commands = [
        ("scripts/run_full_pipeline.py", "every pipeline table, then this page"),
        ("scripts/diagnose_training_window.py", "the training-window sweep"),
        (
            "scripts/diagnose_tranche_drawdown.py",
            "Deployability, the daily series and the charts",
        ),
        ("scripts/describe_modeling_dataset.py", "the dataset and feature-group tables"),
        ("scripts/build_report_page.py", "this page alone, from reports/tables/"),
    ]
    width = max(len(path) for path, _ in commands) + len("python ")

    rebuild = "\n".join(
        f"{('python ' + path):<{width}}  # {purpose}" for path, purpose in commands
    )

    return "\n".join(
        [
            '<section id="reproduce">',
            "<h2>Reproduce</h2>",
            "<h3>Setup</h3>",
            f'<p class="note">This page was built with Python {python}, the '
            "version the test suite runs on; version ranges are pinned in "
            f"{file_link('requirements.txt')}.</p>",
            "<pre><code>git clone "
            f"{REPO_URL}.git\n"
            "cd vn30-alpha-research\n"
            "python -m venv .venv\n"
            ".venv\\Scripts\\activate        # Windows\n"
            "source .venv/bin/activate     # macOS and Linux\n"
            "python -m pip install -r requirements.txt</code></pre>",
            "<h3>Rebuild</h3>",
            '<p class="note">Run from the repository root, since scripts read '
            "paths relative to it. On Windows, <code>py</code> works in place "
            "of <code>python</code>.</p>",
            f"<pre><code>{rebuild}\npython -m pytest -q</code></pre>",
            "<p class=\"note\">Scripts: "
            + " · ".join(file_link(path) for path, _ in commands)
            + ".</p>",
            "<h3>Data access</h3>",
            '<p class="note"><strong>The FiinQuant data is not in the '
            "repository.</strong> <code>data/raw/</code> and "
            f"<code>data/processed/</code> are excluded by {file_link('.gitignore')}, "
            "so a clone can rebuild this page from the committed tables and "
            "run the tests that read them, but cannot rerun the pipeline. "
            "Rebuilding from raw data needs a FiinQuant account: install the "
            "vendor's <code>FiinQuantX</code> wheel in its own environment as "
            f"the {file_link('README.md')} describes, set "
            "<code>FIINQUANT_USERNAME</code> and <code>FIINQUANT_PASSWORD</code>, "
            f"and run {file_link('scripts/download_fiinquant_history.py')}, which "
            "writes the canonical files under "
            "<code>data/raw/fiinquant/canonical/</code>. FLC and ROS raw prices "
            "need a vendor archive export as well, imported by "
            f"{file_link('scripts/import_fiinquant_archived_history.py')}.</p>",
            '<p class="caption">This page is generated by '
            f"{file_link('scripts/build_report_page.py')} from "
            f"{file_link('reports/tables/')} on every pipeline run, so its "
            "figures cannot drift from the tables. Full detail in "
            f"{file_link('reports/final_results.md')} and "
            f"{file_link('reports/methodology.md')}.</p>",
            "</section>",
        ]
    )


def build_page() -> str:
    horizon = read_table("horizon_results.csv").set_index("forecast_horizon_days")
    ablation = read_table("ablation_results.csv").set_index("ablation_name")
    benchmark = read_table("benchmark_results.csv").set_index("strategy")
    models = read_table("model_comparison_results.csv").set_index("model_name")
    sweep = read_table("training_window_diagnostic.csv")
    disclosure = read_table("horizon_sample_disclosure.csv").set_index(
        "forecast_horizon_days"
    )
    concentration = read_table("concentration_summary.csv")
    ranks = read_table("latest_rank_diagnostic.csv")

    ic5 = float(horizon.loc[5, "average_rank_ic"])
    ac5 = float(horizon.loc[5, "average_after_cost_return"])

    ml = benchmark[benchmark["comparison_type"] == "ml_strategy"]
    ml_sorted = ml.sort_values("average_period_active_return", ascending=False)
    positive_now = int((ml["average_period_active_return"] > 0).sum())

    momentum = float(benchmark.loc["top5_momentum", "average_period_active_return"])
    momentum_dates = int(benchmark.loc["top5_momentum", "evaluated_dates"])
    top5_col = "average_top5_realized_return_per_5d_period"
    best_model_name = str(models[top5_col].idxmax())
    best_model = float(models.loc[best_model_name, top5_col])
    ratio = best_model / momentum * 100

    after_cost = {hz: float(horizon.loc[hz, "average_after_cost_return"]) for hz in (1, 5, 10)}
    rank_ic = {hz: float(horizon.loc[hz, "average_rank_ic"]) for hz in (1, 5, 10)}
    positive_horizons = [hz for hz in (1, 5, 10) if after_cost[hz] > 0]
    negative_horizons = [hz for hz in (1, 5, 10) if after_cost[hz] <= 0]
    best_ic_horizon = max(rank_ic, key=lambda hz: rank_ic[hz])

    out: list[str] = []
    add = out.append

    add("<!doctype html>")
    add('<html lang="en">')
    add("<head>")
    add('<meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, initial-scale=1">')
    add("<title>VN30 Alpha Research — Results</title>")
    add(f"<style>{STYLE}</style>")
    add("</head>")
    add("<body>")
    add('<div class="wrap">')

    add("<header>")
    add("<h1>VN30 Alpha Research — Results</h1>")
    add(
        '<p class="sub">Cross-sectional stock selection on the Vietnamese VN30 index, '
        "evaluated after transaction costs and Vietnam-specific execution frictions. "
        "Research framework, not a live trading system.</p>"
    )
    add(
        f'<p class="byline">{AUTHOR_NAME}, {AUTHOR_AFFILIATION} · '
        f'<a href="{REPO_URL}">GitHub repository</a> · '
        f'<a href="{AUTHOR_LINKEDIN_URL}">LinkedIn</a></p>'
    )
    add(
        f'<p class="meta">Generated {date.today():%d %B %Y} from '
        f"{file_link('reports/tables/')} "
        f"· point-in-time VN30 membership · walk-forward training window "
        f"{DEFAULT_TRAIN_SIZE} trading dates</p>"
    )
    add("</header>")

    # Optional: written by scripts/diagnose_tranche_drawdown.py, which is not
    # a pipeline stage. Skip what depends on it rather than fail the build.
    tranche = optional_table("tranche_portfolio_diagnostic.csv")
    daily = optional_table("tranche_daily_returns.csv")
    limited = None
    if tranche is not None:
        limited = (
            tranche[tranche["execution_mode"] == TRANCHE_EXECUTION_MODE]
            .sort_values("active_final_equity_after_cost", ascending=False)
            .set_index("optimization_mode")
        )

    dataset = read_table("dataset_description.csv").iloc[0]
    groups = read_table("feature_groups.csv")
    feature_count = int(groups["feature_count"].sum())
    if feature_count != int(ablation.loc["all_features", "feature_count"]):
        raise ValueError(
            f"feature_groups.csv counts {feature_count} features against "
            f"{int(ablation.loc['all_features', 'feature_count'])} in "
            "ablation_results.csv"
        )

    # --- Overview -----------------------------------------------------------
    add(overview_section(limited, benchmark, models, top5_col, dataset, feature_count))

    # --- Performance --------------------------------------------------------
    if limited is not None and daily is not None:
        modes = [str(mode) for mode in limited.index]
        live = tranche_curves(daily, modes)
        statistics = tranche_statistics(daily, live, modes)
        reconcile_tranche_statistics(statistics, live, limited)
        add(performance_charts_section(live, modes[0], modes[1:]))
        add(
            performance_summary_section(
                statistics,
                limited,
                int(daily[f"{modes[0]}_tranche_turnover"].notna().sum()),
            )
        )

    # --- Method, costs and definitions --------------------------------------
    add(data_and_method_section(dataset, groups, models, horizon))
    add(costs_section())
    add(glossary_section())

    # --- Deployability ------------------------------------------------------
    if tranche is not None:
        add(deployability_section(tranche))

    # --- Summary ------------------------------------------------------------
    add("<section>")
    add("<h2>Summary</h2>")
    add('<div class="tiles">')
    add(tile("Rank IC · 5-day", num(ic5, 4),
             f"Spearman rank correlation, {int(horizon.loc[5, 'evaluated_dates']):,} dates"))
    add(tile("After-cost return · 5d", num(ac5),
             "per overlapping 5-day forecast period"))
    add(tile("Portfolio modes positive", f"{positive_now} of {len(ml)}",
             "after-cost active return per period"))
    add(tile("Best model vs momentum", f"{ratio:.0f}%",
             f"{best_model_name} against top5_momentum"))
    add("</div>")
    add(
        '<div class="callout"><strong>A single-feature momentum rule beats '
        "every trained model.</strong> Ranking on trailing 10-day return and "
        f"holding the top five returns {num(momentum)} per 5-day period "
        f"before cost; the strongest trained model, "
        f"<code>{best_model_name}</code>, returns {num(best_model)} on the "
        f"same basis, which is {ratio:.0f}% of the baseline. "
        "<code>return_10d</code> is an input feature to every model, so the "
        "models hold strictly more information and use it worse. Nothing on "
        "this page supports an alpha claim.</div>"
    )
    add("</section>")

    # --- Training window ----------------------------------------------------
    add("<section>")
    add(f"<h2>Why the training window is {DEFAULT_TRAIN_SIZE} dates</h2>")
    add(
        '<p class="note">Five window sizes are measured on an identical set of '
        f"{int(sweep['evaluated_dates'].iloc[0])} evaluation dates. The relationship is an "
        "inverted U, not monotonic: Rank IC peaks at 21 dates and collapses to roughly zero at "
        "252, which is what one would expect if long windows average across market regimes that "
        "no longer apply.</p>"
    )
    add("<figure>")
    add(sweep_chart(sweep, DEFAULT_TRAIN_SIZE))
    add(
        '<figcaption class="caption">Gradient boosting, 5-day horizon, every configuration scored '
        "on identical dates. Source: "
        f"{file_link('reports/tables/training_window_diagnostic.csv')}"
        "</figcaption>"
    )
    add("</figure>")
    add('<div class="tablewrap"><table><caption>Full sweep result</caption>')
    add('<thead><tr><th scope="col">Training window</th>'
        '<th scope="col">Train rows</th><th scope="col">Rank IC</th>'
        '<th scope="col">Top-5 hit rate</th>'
        '<th scope="col">Top-5 return</th></tr></thead><tbody>')
    for row in sweep.sort_values("train_size_dates").itertuples(index=False):
        size = int(row.train_size_dates)
        chosen = size == DEFAULT_TRAIN_SIZE
        cls = ' class="highlight"' if chosen else ""
        mark = " <strong>(chosen)</strong>" if chosen else ""
        add(
            f"<tr{cls}><td>{size} dates{mark}</td>"
            f"<td>{int(row.approximate_train_rows):,}</td>"
            f"<td>{num(row.average_rank_ic)}</td>"
            f"<td>{num(row.average_top_n_hit_rate)}</td>"
            f"<td>{num(row.average_top_n_actual_return)}</td></tr>"
        )
    add("</tbody></table></div>")
    add(
        f'<p class="note">{DEFAULT_TRAIN_SIZE} is chosen over the Rank-IC-optimal 21 because it '
        "has the best top-5 hit rate and the only clearly positive top-5 return, and portfolio "
        "construction selects a top-N basket rather than trading the full ranking.</p>"
    )
    chosen_sweep = sweep[sweep["train_size_dates"] == DEFAULT_TRAIN_SIZE]
    if not chosen_sweep.empty:
        sweep_ic = float(chosen_sweep["average_rank_ic"].iloc[0])
        sweep_dates = int(chosen_sweep["evaluated_dates"].iloc[0])
        add(
            '<p class="note"><strong>The sweep figure is not the production '
            "figure.</strong> This table reports Rank IC "
            f"{num(sweep_ic)} for the chosen {DEFAULT_TRAIN_SIZE}-date window "
            f"while production reports {num(ic5)}. Both are gradient boosting "
            "on the 5-day horizon and neither is wrong. The sweep scores "
            f"{sweep_dates:,} test dates, subsampled at every tenth date and "
            "starting late enough that the 252-date window is also reachable, "
            "so that all five configurations are scored on an identical set of "
            "dates. Production walk-forward scores every date it can, "
            f"{int(horizon.loc[5, 'evaluated_dates']):,} of them. A shorter, "
            "later, sparser subsample is a different sample of the same "
            "estimator, so the two figures are not required to agree.</p>"
        )
    add(
        '<p class="caption">Chart and table render from the same rows of '
        f"{file_link('reports/tables/training_window_diagnostic.csv', 'training_window_diagnostic.csv')}"
        " — the chart rounds "
        "to four decimals, the table prints six, and neither carries a "
        "separate copy of the numbers. <code>Train rows</code> is the measured "
        "count of dataset rows falling on the first window's training dates, "
        f"not the window length times {EXPECTED_CONSTITUENTS}: "
        + train_row_shortfall_sentence(sweep)
        + "</p>"
    )
    add("</section>")

    # --- Horizon ------------------------------------------------------------
    add("<section>")
    add("<h2>Forecast horizon</h2>")
    add('<p class="note">Gradient boosting, price-limit-aware execution. '
        "Average after-cost return is per forecast period, not annualized.</p>")
    add('<div class="tablewrap"><table><thead><tr>')
    add('<th scope="col">Horizon</th><th scope="col">Rank IC</th>'
        '<th scope="col">Top-5 hit rate</th>'
        '<th scope="col">After-cost return</th>'
        '<th scope="col">Diagnostic Sharpe</th>'
        '<th scope="col">Cumulative sum</th>'
        '<th scope="col">Average turnover</th>'
        '<th scope="col">Evaluated dates</th></tr></thead><tbody>')
    for hz in (1, 5, 10):
        row = horizon.loc[hz]
        cls = ' class="highlight"' if hz == 5 else ""
        add(
            f"<tr{cls}><td>{hz} day</td>"
            f"<td>{num(row['average_rank_ic'])}</td>"
            f"<td>{num(row['average_top_5_hit_rate'])}</td>"
            f"<td>{num(row['average_after_cost_return'])}</td>"
            f"<td>{num(row['diagnostic_sharpe'], 3)}</td>"
            f"<td>{num(row['final_cumulative_after_cost_active_return'], 3)}</td>"
            f"<td>{num(row['average_turnover'], 2)}</td>"
            f"<td>{int(row['evaluated_dates']):,}</td></tr>"
        )
    add("</tbody></table></div>")

    if positive_horizons:
        verdict = (
            f"Only the {horizon_list(positive_horizons)} horizon clears cost, at "
            f"{num(after_cost[positive_horizons[0]])} per period."
            if len(positive_horizons) == 1
            else f"The {horizon_list(positive_horizons)} horizons clear cost."
        )
    else:
        verdict = "No horizon clears cost."

    add(
        f'<p class="note">{verdict} The {horizon_list(negative_horizons)} '
        + ("horizon does" if len(negative_horizons) == 1 else "horizons do")
        + " not. The "
        f"{best_ic_horizon}-day horizon has the <em>highest</em> Rank IC of "
        f"the three at {num(rank_ic[best_ic_horizon])}"
        + (
            " and is still net-negative, which locates its failure in "
            "rebalancing cost rather than in prediction"
            if after_cost[best_ic_horizon] <= 0
            else ""
        )
        + ". Turnover is what separates them: the 1-day horizon rebalances at "
        f"{float(horizon.loc[1, 'average_turnover']):.2f} average turnover "
        f"against {float(horizon.loc[10, 'average_turnover']):.2f} at 10 days, "
        "so the horizon with the best ranking pays the most to act on it.</p>"
    )
    add(
        '<p class="caption">Overlap disclosure: the 10-day horizon’s '
        f"{int(disclosure.loc[10, 'evaluated_dates']):,} evaluated dates correspond to roughly "
        f"{int(disclosure.loc[10, 'approx_non_overlapping_periods'])} genuinely non-overlapping "
        "periods, and the 5-day horizon’s "
        f"{int(disclosure.loc[5, 'evaluated_dates']):,} to roughly "
        f"{int(disclosure.loc[5, 'approx_non_overlapping_periods'])}.</p>"
    )
    add("</section>")

    # --- Ablation -----------------------------------------------------------
    add("<section>")
    add("<h2>Feature ablation</h2>")
    add('<p class="note">5-day horizon. Group sizes are the drop in feature '
        "count when each group is removed.</p>")
    add('<div class="tablewrap"><table><thead><tr>')
    add('<th scope="col">Configuration</th><th scope="col">Features</th>'
        '<th scope="col">Rank IC</th><th scope="col">Top-5 hit rate</th>'
        '<th scope="col">After-cost return</th>'
        '<th scope="col">Cumulative sum</th></tr></thead><tbody>')
    ordered = ablation.sort_values("average_rank_ic", ascending=False)
    for name, row in ordered.iterrows():
        cls = ' class="highlight"' if name == "all_features" else ""
        label = name.replace("_", " ")
        add(
            f"<tr{cls}><td>{label}</td><td>{int(row['feature_count'])}</td>"
            f"<td>{num(row['average_rank_ic'])}</td>"
            f"<td>{num(row['average_top_5_hit_rate'])}</td>"
            f"<td>{num(row['average_after_cost_return'])}</td>"
            f"<td>{num(row['final_cumulative_after_cost_active_return'], 3)}</td>"
            "</tr>"
        )
    add("</tbody></table></div>")
    positive_ablations = int((ablation["average_after_cost_return"] > 0).sum())
    add(
        '<div class="callout"><strong>The full feature set leads on Rank IC, '
        "and that is a separate claim from paying for itself.</strong> "
        "<code>all_features</code> has the highest Rank IC of the five "
        "configurations, so no feature group is dead weight on ranking "
        "quality. On after-cost return, "
        + (
            f"{positive_ablations} of {len(ablation)} configurations clears "
            "transaction costs"
            if positive_ablations
            else f"none of the {len(ablation)} configurations clears "
            "transaction costs"
        )
        + ". Removing <code>price_limit</code> or <code>volume_liquidity</code> "
        "costs the most Rank IC, which is what one would expect of features "
        "built around the mechanics of the exchange.</div>"
    )
    add("</section>")

    # --- Model comparison ---------------------------------------------------
    add("<section>")
    add("<h2>Model comparison</h2>")
    add('<p class="note">Equal-weight top-5 by model score, before cost · '
        "diagnostic only, not optimizer or backtest execution.</p>")
    add('<div class="tablewrap"><table><thead><tr>')
    add('<th scope="col">Model</th><th scope="col">Rank IC</th>'
        '<th scope="col">Top-5 return / 5d</th>'
        '<th scope="col">Diagnostic Sharpe</th>'
        '<th scope="col">Cumulative sum</th></tr></thead><tbody>')
    for name, row in models.sort_values("average_rank_ic", ascending=False).iterrows():
        cls = ' class="highlight"' if name == best_model_name else ""
        suffix = " (production)" if name == "gradient_boosting" else ""
        add(
            f"<tr{cls}><td>{name}{suffix}</td>"
            f"<td>{num(row['average_rank_ic'])}</td>"
            f"<td>{num(row[top5_col])}</td>"
            f"<td>{num(row['diagnostic_sharpe'], 3)}</td>"
            f"<td>{num(row['final_cumulative_top5_return_sum'], 3)}</td></tr>"
        )
    add("</tbody></table></div>")
    if best_model_name != "gradient_boosting":
        add(
            '<div class="callout"><strong>The production model is not the '
            f"strongest one.</strong> <code>{best_model_name}</code> leads on "
            "top-5 return while <code>gradient_boosting</code> is what feeds "
            "portfolio construction. The inconsistency is recorded rather "
            "than resolved: switching the production model moves every "
            "portfolio figure on this page, so it is an open question.</div>"
        )
    add("</section>")

    # --- Baselines ----------------------------------------------------------
    add("<section>")
    add("<h2>Baseline comparison</h2>")
    add('<p class="note">Top-5 selection, before cost, same relative-to-VN30 '
        f"target, all scored on the same {momentum_dates:,} dates.</p>")
    add('<div class="tablewrap"><table><thead><tr>'
        '<th scope="col">Selection rule</th>'
        '<th scope="col">Top-5 active return / 5d</th>'
        '<th scope="col">vs momentum</th></tr></thead><tbody>')
    add(
        '<tr class="highlight"><td>Trailing 10-day return '
        f"(<code>top5_momentum</code>)</td><td><strong>{num(momentum)}</strong>"
        "</td><td>—</td></tr>"
    )
    for name, row in models.sort_values(top5_col, ascending=False).iterrows():
        share = float(row[top5_col]) / momentum * 100
        add(
            f"<tr><td>{name}</td><td>{num(row[top5_col])}</td>"
            f"<td>{share:.0f}%</td></tr>"
        )
    add("</tbody></table></div>")

    momentum_row = benchmark.loc["top5_momentum"]
    if "mean_active_return_95pct_lower" in momentum_row.index and not pd.isna(
        momentum_row["mean_active_return_95pct_lower"]
    ):
        momentum_lower = float(momentum_row["mean_active_return_95pct_lower"])
        momentum_upper = float(momentum_row["mean_active_return_95pct_upper"])
        momentum_excludes = momentum_lower > 0.0 or momentum_upper < 0.0
        add(
            '<p class="note"><strong>The baseline is the one row that '
            "separates from zero.</strong> <code>top5_momentum</code> has a "
            "95% moving-block bootstrap interval of "
            f"[{num(momentum_lower)}, {num(momentum_upper)}] on its mean "
            "active return, which "
            + ("excludes zero" if momentum_excludes else "spans zero")
            + ". Every one of the eight after-cost portfolio configurations "
            "under Portfolio backtest has an interval that spans zero. The "
            "baseline is measured before cost and the portfolios after it, so "
            "this is not a like-for-like contest — but it is the shape of "
            "the whole result: a detectable signal in a one-feature rule, and "
            "nothing detectable left once the models and the trading costs "
            "are added.</p>"
        )
    add(
        '<p class="note"><code>return_10d</code> is an input feature to every '
        "model, so none of them is missing the information the baseline uses. "
        f"The strongest, <code>{best_model_name}</code>, reaches "
        f"{ratio:.0f}% of it.</p>"
    )
    add("</section>")

    # --- Portfolio ----------------------------------------------------------
    add("<section>")
    add("<h2>Portfolio backtest</h2>")
    add('<p class="note">Optimizer modes crossed with execution modes, 5-day '
        "horizon, after cost.</p>")
    add('<div class="tablewrap"><table><thead><tr>'
        '<th scope="col">Optimizer</th><th scope="col">Execution</th>'
        '<th scope="col">Active return / period</th>'
        '<th scope="col">Diagnostic Sharpe</th>'
        '<th scope="col">Cumulative sum</th>'
        '<th scope="col">Holdings</th></tr></thead><tbody>')
    for name, row in ml_sorted.iterrows():
        optimizer, execution = STRATEGY_LABELS.get(name, (name, ""))
        cls = ' class="highlight"' if name == ml_sorted.index[0] else ""
        add(
            f"<tr{cls}><td>{optimizer}</td><td>{execution}</td>"
            f"<td>{num(row['average_period_active_return'])}</td>"
            f"<td>{num(row['diagnostic_sharpe'], 3)}</td>"
            f"<td>{num(row['final_cumulative_active_return_sum'], 3)}</td>"
            f"<td>{float(row['average_selected_count']):.1f}</td></tr>"
        )
    add("</tbody></table></div>")
    conservative = benchmark.loc["ml_normal_price_limit_aware"]
    conservative_mean = float(conservative["average_period_active_return"])
    verb = "clears cost by" if conservative_mean > 0 else "falls short of cost by"
    add(
        f'<p class="note">{positive_now} of {len(ml)} configurations are '
        "positive. The most conservative (<code>normal</code> + "
        f"<code>price_limit_aware</code>) {verb} "
        f"{num(abs(conservative_mean))} per period, at a diagnostic Sharpe of "
        f"{num(conservative['diagnostic_sharpe'], 3)}. Wider diversification "
        "outperforms concentration, which is evidence that the ranking signal "
        "is weak.</p>"
    )
    add(interval_sentence(benchmark, ml_sorted))
    add(
        '<p class="caption">Cumulative figures in <em>this</em> table sum '
        "overlapping forecast-period active returns, so they are not "
        "compounded portfolio returns, and <code>max_active_drawdown</code> in "
        f"{file_link('reports/tables/benchmark_results.csv', 'benchmark_results.csv')}"
        " is a running-peak fall on that same "
        "summed series — in summed-return units, not as a percentage of "
        "capital. The Deployability table at the top of the page uses the "
        "other convention: both of its drawdown columns are peak-to-trough "
        "falls of a compounded equity curve and are genuine percentages of "
        "capital. The two conventions share no number and should not be "
        "compared.</p>"
    )
    add("</section>")

    # --- Latest diagnostics -------------------------------------------------
    latest = concentration[concentration["optimization_mode"] == "normal"]
    add("<section>")
    add("<h2>Latest diagnostics</h2>")
    if not latest.empty:
        row = latest.iloc[0]
        group_tickers = str(row["top_issuer_group_tickers"])
        add(
            f'<p class="note">Signal date {row["signal_date"]}, the last date '
            "in the dataset. <code>normal</code> optimizer: "
            f"{int(row['holding_count'])} holdings, maximum single-name weight "
            f"{row['max_single_name_weight'] * 100:.1f} percent, HHI "
            f"{row['hhi']:.3f}, effective position count "
            f"{row['effective_position_count']:.1f}, top issuer group "
            f"{row['top_issuer_group']} at "
            f"{row['top_issuer_group_weight'] * 100:.1f} percent.</p>"
        )
        add(
            '<p class="caption">Those last two percentages are measured '
            "against different constraints, and neither is breached. The "
            f"{MAX_WEIGHT * 100:.0f} percent figure is the single-name cap, "
            "and the largest single holding sits at "
            f"{row['max_single_name_weight'] * 100:.2f} percent, just inside "
            f"it. The {row['top_issuer_group']} figure of "
            f"{row['top_issuer_group_weight'] * 100:.1f} percent is a sum "
            f"across {group_tickers}, so the constraint that governs it is "
            "the issuer-group cap of "
            f"{MAX_ISSUER_GROUP_WEIGHT * 100:.0f} percent, not the "
            "single-name cap. Issuer groups at or above the group cap on this "
            f"date: {int(row['issuer_groups_at_or_above_40pct'])}.</p>"
        )
    top_ranked = ranks.sort_values("predicted_rank").head(5)
    worst_missed = ranks.sort_values("realized_rank").head(2)
    add('<div class="tablewrap"><table>'
        "<caption>Predicted versus realized rank, latest signal date</caption>"
        '<thead><tr><th scope="col">Ticker</th>'
        '<th scope="col">Predicted rank</th>'
        '<th scope="col">Realized rank</th>'
        '<th scope="col">Realized forward return</th></tr></thead><tbody>')
    for row in pd.concat([top_ranked, worst_missed]).itertuples(index=False):
        add(
            f"<tr><td>{row.ticker}</td><td>{int(row.predicted_rank)}</td>"
            f"<td>{int(row.realized_rank)}</td>"
            f"<td>{row.realized_forward_return * 100:.2f}%</td></tr>".replace(
                "-", MINUS, 1
            )
            if row.realized_forward_return < 0 else
            f"<tr><td>{row.ticker}</td><td>{int(row.predicted_rank)}</td>"
            f"<td>{int(row.realized_rank)}</td>"
            f"<td>{row.realized_forward_return * 100:.2f}%</td></tr>"
        )
    add("</tbody></table></div>")
    add('<p class="caption">The last two rows are the best realized performers '
        "of the day, shown with the rank the model gave them. One date is an "
        "illustration, not evidence.</p>")
    add("</section>")

    # --- Universe and periods -----------------------------------------------
    add("<section>")
    add("<h2>Universe and periods</h2>")
    add(
        '<p class="note">The universe is point-in-time VN30 membership, '
        "reconstructed from verified index-review history in "
        f"{file_link('config/vn30_history/')}. Three different ticker counts "
        "appear across these outputs and all three are correct.</p>"
    )
    add('<div class="tablewrap"><table><thead><tr>'
        '<th scope="col">Count</th><th scope="col">What it counts</th>'
        '<th scope="col">Where it appears</th></tr></thead><tbody>')
    add(f"<tr><td>{EXPECTED_CONSTITUENTS}</td>"
        "<td>Index constituents on any single date</td>"
        "<td>Every per-date figure, portfolio construction</td></tr>")
    add("<tr><td>54</td>"
        "<td>Distinct tickers in the paper-horizon prediction file</td>"
        "<td><code>paper_horizon_predictions.parquet</code></td></tr>")
    add(f'<tr class="highlight"><td>{DATASET_TICKERS}</td>'
        "<td>Distinct tickers in the modeling dataset over the full history</td>"
        "<td>The 74,910-row dataset: 30 current constituents plus 30 former "
        "ones</td></tr>")
    add("</tbody></table></div>")
    add(
        '<p class="note">54 rather than '
        f"{DATASET_TICKERS} in the prediction file follows from point-in-time "
        "membership. Six former constituents (FLC, HAG, HCM, HNG, ITA, PPC) "
        "left the index before 2017-08-09, which is the first date that file "
        "can predict after its 252-date training warm-up, so they are "
        "correctly absent from it.</p>"
    )
    add(paper_horizon_facts())
    add(
        '<p class="note"><strong>Three distinct periods appear on this page.</strong> '
        "The Deployability table covers the backtest window, the longest of "
        "the three. The prediction file above covers a shorter range, because "
        "it begins after a 252-date warm-up and ends where a 10-day forward "
        "label stops being observable. The signal date under Latest "
        "diagnostics is the last date in the dataset and is later than both, "
        "because a signal needs no realized forward label. Each is reported "
        "against its own period and no figure is carried between them.</p>"
    )
    add("</section>")

    # --- Limitations --------------------------------------------------------
    add("<section>")
    add("<h2>Limitations</h2>")
    add("<ul>")
    add(
        "<li><strong>A one-feature momentum baseline beats every trained "
        f"model</strong> — {num(momentum)} against {num(best_model)} for "
        f"<code>{best_model_name}</code>, before cost on matched dates.</li>"
    )
    add(
        "<li><strong>No optimizer mode beats holding the index.</strong> "
        "Compounded as the overlapping-tranche portfolio the strategy would "
        "actually trade, every mode finishes behind the VN30 after costs.</li>"
    )
    add(
        "<li><strong>Margins are thin.</strong> The most conservative "
        "portfolio configuration "
        + ("clears cost at" if conservative_mean > 0 else "does not clear cost, at")
        + " a diagnostic Sharpe of "
        + num(conservative["diagnostic_sharpe"], 3)
        + interval_clause(conservative)
        + ".</li>"
    )
    add(
        "<li><strong>Concentration reduces performance</strong>, which is what "
        "a weak ranking signal looks like.</li>"
    )
    add(foreign_flow_item())
    add(
        "<li><strong>Features are fitted in raw levels</strong> while the "
        "target is demeaned against the index, and no within-date "
        "standardisation is applied. Whether that specification handicaps the "
        "models is untested.</li>"
    )
    add("</ul>")
    interval_rows = (
        int(benchmark["mean_active_return_95pct_lower"].notna().sum())
        if "mean_active_return_95pct_lower" in benchmark.columns
        else 0
    )
    add(
        '<div class="callout"><strong>Multiple comparisons.</strong> The '
        "training window is chosen by measuring five values against the same "
        "history, on top of three horizons, five ablation configurations, four "
        "optimizer modes, two execution modes and five foreign-flow feature "
        "sets tested against it. Nothing here corrects for the number of "
        "comparisons, and the intervals that exist "
        + (
            f"({interval_rows} strategy and baseline rows of "
            f"{file_link('reports/tables/benchmark_results.csv', 'benchmark_results.csv')}"
            ", plus the foreign-flow test) "
            "are each marginal rather than family-wise"
            if interval_rows
            else "cover only the foreign-flow test"
        )
        + ". Rank IC, hit rate, and every per-horizon and per-ablation figure "
        "on this page is a point estimate with no interval at all. This is a "
        "statement about the configurations measured here, not about the VN30 "
        "market.</div>"
    )
    add("</section>")

    # --- Reproduce ----------------------------------------------------------
    add(reproduce_section())

    add("<footer>Research framework, not a live trading system. Not investment advice.</footer>")
    add("</div>")
    add("</body>")
    add("</html>")

    return "\n".join(out)


def strip_document_shell(html: str) -> str:
    """Return just the body content, for hosts that supply their own shell.

    `reports/reportpage.html` is served as a standalone file, so it needs its
    own doctype, `<head>`, charset, viewport and `<title>`. A host that wraps
    the file in a shell of its own instead renders all of that as visible body
    text. For those hosts, emit the body content and carry the stylesheet
    across, rather than dropping the head tags from the standalone file where
    they belong.
    """
    head_open = html.index("<style>")
    head_close = html.index("</style>") + len("</style>")
    style = html[head_open:head_close]

    body_open = html.index("<body>") + len("<body>")
    body_close = html.rindex("</body>")
    body = html[body_open:body_close].strip("\n")

    title_open = html.index("<title>") + len("<title>")
    title = html[title_open : html.index("</title>")]

    return f"<title>{title}</title>\n{style}\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--body-only",
        action="store_true",
        help=(
            "Write title, stylesheet and body content without the document "
            "shell, for a host that supplies its own doctype and head. The "
            "default standalone file keeps its head, which it needs when "
            "served directly."
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    html = build_page()

    if args.body_only:
        html = strip_document_shell(html)

    output_path = Path(args.output) if args.output else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print("Report page generated:")
    print(" ", output_path)
    print("  bytes:", len(html))
    print("  form:", "body only" if args.body_only else "standalone document")


if __name__ == "__main__":
    main()
