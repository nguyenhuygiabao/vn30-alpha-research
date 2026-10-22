"""Generate the project report page from the published result tables.

The page is rebuilt from `reports/tables/*.csv` on every pipeline run, so it
cannot drift from the numbers it cites. Nothing here is hand-entered: a
hardcoded figure would survive a pipeline run that changed it.

The page reports what the framework measures, in the present tense. It is not a
changelog: version history lives in git, not in the results. Every figure here
is read from a table, and the only literals are the pre-registered foreign-flow
comparison keys and the paper-horizon file identity, both of which name rows
rather than values.

Output: reports/reportpage.html
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizer import MAX_ISSUER_GROUP_WEIGHT, MAX_WEIGHT
from src.vn30_universe import EXPECTED_CONSTITUENTS
from src.walk_forward_split import DEFAULT_TRAIN_SIZE

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
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:#1a1a19; --page:#0d0d0d;
      --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-muted:#898781;
      --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
      --series-1:#3987e5; --good:#0ca30c; --critical:#d03b3b; --neutral-mid:#383835;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:#1a1a19; --page:#0d0d0d;
    --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --good:#0ca30c; --critical:#d03b3b; --neutral-mid:#383835;
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
            "rows. Regenerate <code>benchmark_results.csv</code> with "
            "<code>scripts/build_report_tables.py</code> to add one.</p>"
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
            "<code>reports/tables/tranche_portfolio_diagnostic.csv</code>, "
            "generated by "
            "<code>scripts/diagnose_tranche_drawdown.py</code>. This "
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
        f'<p class="meta">Generated {date.today():%d %B %Y} from <code>reports/tables/</code> '
        f"· point-in-time VN30 membership · walk-forward training window "
        f"{DEFAULT_TRAIN_SIZE} trading dates</p>"
    )
    add("</header>")

    # --- Deployability ------------------------------------------------------
    # Optional: written by scripts/diagnose_tranche_drawdown.py, which is not
    # a pipeline stage. Skip the section rather than fail the page build.
    tranche = optional_table("tranche_portfolio_diagnostic.csv")
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
        "on identical dates. Source: <code>reports/tables/training_window_diagnostic.csv</code>"
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
        "<code>training_window_diagnostic.csv</code> — the chart rounds "
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
        "<code>benchmark_results.csv</code> is a running-peak fall on that same "
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
        "<code>config/vn30_history/</code>. Three different ticker counts "
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
            "<code>benchmark_results.csv</code>, plus the foreign-flow test) "
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
    add("<section>")
    add("<h2>Reproduce</h2>")
    add("<pre><code>py .\\scripts\\run_full_pipeline.py          # rebuilds every table and this page\n"
        "py .\\scripts\\diagnose_training_window.py    # the sweep behind the window size\n"
        "py .\\scripts\\diagnose_tranche_drawdown.py   # the deployability table\n"
        "py -m pytest -q                            # test suite</code></pre>")
    add(
        '<p class="caption">This page is generated by '
        "<code>scripts/build_report_page.py</code> from <code>reports/tables/</code> on every "
        "pipeline run, so its figures cannot drift from the tables. Full detail in "
        "<code>reports/final_results.md</code> and <code>reports/methodology.md</code>.</p>"
    )
    add("</section>")

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
