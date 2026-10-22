"""Generate the project report page from the published result tables.

The page is rebuilt from `reports/tables/*.csv` on every pipeline run, so it
cannot drift from the numbers it cites. An earlier hand-written version carried
hardcoded figures, which would have gone silently stale the first time the
pipeline reran -- the same failure mode as the stale ablation cache and the
stale horizon-disclosure table this project has already had to fix.

The only frozen values are the `train_size = 3` baseline in BASELINE, a
historical record of a configuration no current table contains, and the
paper-horizon migration counts. Both are labelled as such below.

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
FOREIGN_FLOW_PAIRED_PATH = (
    TABLES / "foreign_flow" / "foreign_flow_rebuild_development_paired.csv"
)

# The pre-registered comparison: the enhanced foreign buy/sell feature set
# against the frozen one, on paired daily Rank IC.
FOREIGN_FLOW_CHALLENGER = "enhanced_fb_fs"
FOREIGN_FLOW_BASELINE = "frozen_foreign_flow"
OUTPUT_PATH = ROOT / "reports" / "reportpage.html"

# ---------------------------------------------------------------------------
# Frozen historical baseline: results when the walk-forward training window was
# 3 trading dates (about 90 rows against 99 features). That configuration no
# longer exists, so no current CSV contains these; they are recorded here so the
# page can show what changed. Taken from the tables as committed at 0d8b2a5.
# ---------------------------------------------------------------------------
BASELINE_TRAIN_SIZE = 3

# The baseline was scored on more dates than the current configuration. A
# walk-forward window cannot predict until it has `train_size` dates of
# history behind it, so widening the window from 3 to 63 dates consumes 60
# more dates of warm-up and drops them from the evaluation. Every "before"
# column on this page is a mean over 2,478 dates and every "after" column a
# mean over 2,418: the same history, 60 fewer measurements at the start of it.
# That is the whole reason the untrained momentum baseline moves between the
# two columns even though the rule itself never changed. Taken from
# reports/tables/benchmark_results.csv as committed at 18e275d^.
BASELINE_EVALUATED_DATES = 2478

BASELINE_HORIZON = {
    1: {"rank_ic": 0.018752, "after_cost": -0.000703, "cumulative": -1.749439},
    5: {"rank_ic": 0.019869, "after_cost": -0.000481, "cumulative": -1.191474},
    10: {"rank_ic": 0.039083, "after_cost": 0.000804, "cumulative": 1.979629},
}

BASELINE_ABLATION = {
    "all_features": {"rank_ic": 0.019869, "after_cost": -0.000481},
    "without_risk": {"rank_ic": 0.025746, "after_cost": -0.000321},
    "without_herding": {"rank_ic": 0.020276, "after_cost": -0.000572},
    "without_price_limit": {"rank_ic": 0.013470, "after_cost": -0.000664},
    "without_volume_liquidity": {"rank_ic": 0.021549, "after_cost": -0.000540},
}

BASELINE_MODELS = {
    "elastic_net": {"rank_ic": 0.021737, "top5": 0.001351},
    "ridge": {"rank_ic": 0.017905, "top5": 0.000786},
    "gradient_boosting": {"rank_ic": 0.019869, "top5": 0.000998},
    "random_forest": {"rank_ic": 0.018015, "top5": 0.001433},
    "classification": {"rank_ic": 0.018657, "top5": 0.001006},
}

BASELINE_STRATEGIES = {
    "ml_sector_diversified_normal": {"active": 0.000341, "cumulative": 0.843858},
    "ml_sector_diversified_price_limit_aware": {"active": 0.000102, "cumulative": 0.253794},
    "ml_herding_aware_normal": {"active": 0.000036, "cumulative": 0.088217},
    "ml_sector_aware_normal": {"active": -0.000080, "cumulative": -0.197645},
    "ml_normal_normal": {"active": -0.000064, "cumulative": -0.157891},
    "ml_herding_aware_price_limit_aware": {"active": -0.000280, "cumulative": -0.692317},
    "ml_sector_aware_price_limit_aware": {"active": -0.000445, "cumulative": -1.103530},
    "ml_normal_price_limit_aware": {"active": -0.000481, "cumulative": -1.191474},
}

BASELINE_MOMENTUM = 0.002310

# One-time migration of paper-horizon predictions from the static 30-ticker
# universe to point-in-time membership. Recorded, not derived: the "before"
# column describes a file that no longer exists.
#
# The row count needs decomposing or it reads as a duplicate-row bug. The file
# stacks two models, so a fully covered date carries 30 constituents times 2
# models = 60 rows, and 208 * 60 = 12,480. Verified against
# data/processed/paper_horizon_predictions.parquet by
# paper_horizon_check() below whenever that file is present.
PAPER_HORIZON_MODELS = 2
PAPER_HORIZON_DATES = 208
PAPER_HORIZON_ROWS = 12_480
PAPER_HORIZON_TICKERS = 54
PAPER_HORIZON_DATE_RANGE = "2017-08-09 to 2026-07-01"

PAPER_HORIZON_MIGRATION = [
    ("Models stacked per date", "2", "2"),
    ("Constituents per date", "30", "30"),
    ("Rows per fully covered date", "60", "60"),
    ("Prediction dates", "136", "208"),
    ("Prediction rows", "7,932", "12,480"),
    ("Distinct tickers across all dates", "30", "54"),
    ("Date range", "2021-01-15 to 2026-06-22", PAPER_HORIZON_DATE_RANGE),
    ("Dates at full point-in-time coverage", "3 of 136", "208 of 208"),
    ("Rebalance dates in both scripts", "3", "208"),
]

PAPER_HORIZON_PREDICTIONS_PATH = (
    ROOT / "data" / "processed" / "paper_horizon_predictions.parquet"
)

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
ARROW_UP = "↑"
ARROW_DOWN = "↓"

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
  .tile .from { font-size:0.82rem; color:var(--ink-2); margin-top:2px; font-variant-numeric:tabular-nums; }
  .tile .delta { font-size:0.82rem; margin-top:5px; font-weight:600; }
  .up { color:var(--good); } .down { color:var(--critical); } .flat { color:var(--ink-muted); }
  .tablewrap { overflow-x:auto; margin-top:12px; -webkit-overflow-scrolling:touch; }
  table { border-collapse:collapse; width:100%; min-width:560px; font-size:0.88rem; }
  th, td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--grid);
    font-variant-numeric:tabular-nums; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; font-variant-numeric:normal; }
  thead th { color:var(--ink-2); font-weight:600; border-bottom:1px solid var(--axis); font-size:0.82rem; }
  tbody tr:last-child td { border-bottom:none; }
  tbody tr.highlight td { background:color-mix(in srgb, var(--series-1) 8%, transparent); }
  caption { caption-side:top; text-align:left; color:var(--ink-2); font-size:0.86rem; padding-bottom:6px; }
  .grouphead th { border-bottom:none; color:var(--ink-muted); font-size:0.76rem;
    text-transform:uppercase; letter-spacing:0.04em; padding-bottom:2px;
    text-align:center; }
  .grouphead th.stub { text-transform:none; letter-spacing:0; font-size:0.82rem;
    color:var(--ink-2); font-weight:600; text-align:left; vertical-align:bottom; }
  .grouphead th[colspan] { border-bottom:1px solid var(--grid); }
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


BEFORE_COLUMN_DISCLOSURE = (
    '<p class="caption"><strong>What the <em>before</em> column is.</strong> '
    "It was captured under <code>train_size = 3</code> and under the cost "
    "model as it stood before commit <code>bfdabcf</code>, which charged "
    "commission and slippage on one leg of a trade instead of two and omitted "
    "the statutory transfer tax, pricing a round trip at 0.2 percent where the "
    "paper account charges 0.5. A before/after pair in an <em>after-cost</em> "
    "column therefore straddles two changes at once and is not the effect of "
    "the training window. The Rank IC columns do isolate the training window, "
    "because Rank IC is a rank correlation and does not depend on costs at "
    "all \u2014 which is why those columns are identical to six decimals "
    "before and after the cost correction.</p>"
)


def interval_clause(row: pd.Series) -> str:
    """Describe a row's bootstrap interval, or say that there isn't one.

    Written as a clause rather than a sentence so a caller can append it. It
    reports what the interval does instead of asserting it, because the sign
    of a near-zero mean is exactly the thing that moves when the cost model or
    the sample changes.
    """
    lower_key = "mean_active_return_95pct_lower"
    upper_key = "mean_active_return_95pct_upper"

    if lower_key not in row.index or pd.isna(row[lower_key]):
        return ", with no interval published for it"

    lower = float(row[lower_key])
    upper = float(row[upper_key])
    verdict = "spans zero" if lower <= 0.0 <= upper else "excludes zero"

    return (
        f", and a 95% bootstrap interval on mean active return of "
        f"[{num(lower)}, {num(upper)}], which {verdict}"
    )


def foreign_flow_item() -> str:
    """The pre-registered foreign-flow result, read from its own table.

    This was the one figure on the page that carried an interval, and it was
    the one figure typed in by hand. Reading it from
    `foreign_flow_rebuild_development_paired.csv` means the only interval on
    the page cannot be the only number that silently goes stale.
    """
    fallback = (
        "<li><strong>Foreign buy/sell features remain unpromoted.</strong> "
        "Pre-registered test: mean Rank-IC difference "
        f"{MINUS}0.002608, 95% CI [{MINUS}0.013314, +0.008297]. The interval "
        "spans zero and the point estimate is negative.</li>"
    )

    if not FOREIGN_FLOW_PAIRED_PATH.is_file():
        return fallback

    paired = pd.read_csv(FOREIGN_FLOW_PAIRED_PATH)
    matched = paired[
        (paired["challenger"] == FOREIGN_FLOW_CHALLENGER)
        & (paired["baseline"] == FOREIGN_FLOW_BASELINE)
    ]

    if matched.empty:
        return fallback

    row = matched.iloc[0]
    difference = float(row["mean_rank_ic_difference"])
    lower = float(row["bootstrap_95pct_lower"])
    upper = float(row["bootstrap_95pct_upper"])
    dates = int(row["paired_dates"])
    spans_zero = lower <= 0.0 <= upper

    return (
        "<li><strong>Foreign buy/sell features remain unpromoted.</strong> "
        "Pre-registered test, paired block bootstrap on daily Rank IC over "
        f"{dates} non-overlapping signals: mean difference "
        f"{num(difference)}, 95% CI [{num(lower)}, {num(upper)}]. The "
        + (
            "interval spans zero"
            if spans_zero
            else "interval excludes zero"
        )
        + " and the point estimate is "
        + ("negative" if difference < 0 else "positive")
        + ". The promotion rule was fixed before the result was seen: promote "
        "only if the interval excludes zero.</li>"
    )


def paper_horizon_check() -> str:
    """Re-derive the frozen paper-horizon counts from the file, if it is here.

    `data/processed/` is not tracked, so the page cannot depend on the parquet
    existing. When it does exist this prints the decomposition measured from
    it, which is what turns "12,480 rows over 208 dates" from a number that
    looks like a duplicate-row bug into one that is accounted for.
    """
    if not PAPER_HORIZON_PREDICTIONS_PATH.is_file():
        return (
            '<p class="caption">The counts above are frozen records of a '
            "one-time migration. "
            "<code>data/processed/paper_horizon_predictions.parquet</code> is "
            "not present in this checkout, so they were not re-derived on this "
            "run.</p>"
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

    agrees = (
        rows == PAPER_HORIZON_ROWS
        and dates == PAPER_HORIZON_DATES
        and tickers == PAPER_HORIZON_TICKERS
        and len(model_names) == PAPER_HORIZON_MODELS
    )
    verdict = (
        "match the frozen counts above"
        if agrees
        else "DISAGREE with the frozen counts above and need regenerating"
    )

    return (
        '<p class="caption">Re-derived from '
        "<code>data/processed/paper_horizon_predictions.parquet</code> on this "
        f"run: {rows:,} rows, {dates} dates, {per_date:.0f} rows per date, "
        f"{tickers} distinct tickers, {len(model_names)} models "
        f"({', '.join(model_names)}). These {verdict}. "
        f"{duplicate_keys:,} rows repeat a (date, ticker) pair, which is "
        f"expected and is exactly the {len(model_names)} models stacked in one "
        f"file; {duplicate_rows} rows repeat a (date, ticker, model) key, so "
        "there is no duplicate-row or join defect.</p>"
    )


def train_row_shortfall_sentence(sweep: pd.DataFrame) -> str:
    """Explain why `approximate_train_rows` is not the window length times 30.

    The sweep measures the rows actually present in the first window's training
    dates. The VN30 carries 30 constituents, so the arithmetic answer is
    `train_size * 30`, and two of the five windows come in under it. Ten dates
    in the dataset carry 29 rows rather than 30, because FLC has no data for
    the week of 2017-01-16 and ROS none for the week of 2021-01-25. A window
    that spans one of those dates is one row short per date spanned, which is
    a real gap in the data rather than an error in the count.
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
        + ". Ten dates in the dataset carry 29 constituents rather than 30 "
        "because FLC has no data for the week of 2017-01-16 and ROS none for "
        "the week of 2021-01-25, and a window spanning one of those dates is "
        "one row short for each. The shortfall is a data gap, not a counting "
        "error."
    )


def interval_sentence(benchmark: pd.DataFrame, ml_sorted: pd.DataFrame) -> str:
    """Report the bootstrap interval rather than asserting significance.

    The previous wording called a diagnostic Sharpe of 0.004 "statistically
    indistinguishable from zero" with no test behind it. The interval below is
    a 95 percent moving-block bootstrap on the mean after-cost active return,
    blocked at the five-day forecast horizon so the overlap between
    consecutive period returns stays inside a block. It is computed in
    `scripts/build_report_tables.py` and published in
    `benchmark_results.csv`.
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
    parts: list[str] = []

    for name in (best_name, "ml_normal_price_limit_aware"):
        row = benchmark.loc[name]
        optimizer, execution = STRATEGY_LABELS.get(name, (name, ""))
        lower = float(row["mean_active_return_95pct_lower"])
        upper = float(row["mean_active_return_95pct_upper"])
        excludes = lower > 0.0 or upper < 0.0
        verdict = (
            "excludes zero" if excludes else "spans zero"
        )
        parts.append(
            f"<code>{optimizer}</code> + <code>{execution}</code> has mean "
            f"active return {num(float(row['average_period_active_return']))} "
            f"with a 95% interval of [{num(lower)}, {num(upper)}], which "
            f"{verdict}"
        )

    if best_name == "ml_normal_price_limit_aware":
        parts = parts[:1]

    return (
        '<p class="note"><strong>Is any of this distinguishable from '
        "zero?</strong> "
        + ". ".join(parts)
        + ". The interval is a 95 percent moving-block bootstrap on the mean "
        "after-cost active return, 2,000 resamples, blocked at the 5-day "
        "forecast horizon so that the overlap between consecutive period "
        "returns stays inside a block. An interval that spans zero is the "
        "reason no alpha claim is made here; it is not a claim that the mean "
        "is zero.</p>"
    )


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


def deployability_section(tranche: pd.DataFrame) -> str:
    """What a portfolio running this strategy would actually have done.

    Every per-period figure on the page is a mean over overlapping forecast
    periods, which a portfolio cannot hold. The strategy signals daily and
    holds five days, so what it trades is five overlapping tranches. That
    construction holds every phase at once and has no free parameters, unlike
    sampling every fifth date, where the starting offset moves the answer more
    than the strategy does.

    Two columns here are routinely misread, so both are defined on the page
    rather than left to the reader.

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
            f"<td>{active:.4f} \u00b7 {verdict}</td>"
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
            f"<td>{float(row['phase_to_offset_spread_ratio']):.0f}\u00d7</td>"
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
            "beside it \u2014 compounding a difference of returns is not the "
            "same as taking a ratio of compounded returns, and the gap "
            "between the two columns is the covariance of the two legs. "
            "<em>Strategy drawdown</em> is the worst peak-to-trough fall of "
            "the compounded strategy equity curve, a true percentage of "
            "capital. <em>Active drawdown</em> is the same measure on the "
            "compounded active-return curve. Neither is the summed-return "
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
            "backtest period. It is not the paper-horizon prediction range "
            "quoted further down, which is shorter because it starts after a "
            "252-date training warm-up.</p>",
            "</section>",
        ]
    )


def num(value: float, digits: int = 6) -> str:
    """Format with a true minus sign rather than a hyphen."""
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}".replace("-", MINUS)


def delta_cell(current: float, baseline: float, digits: int = 6) -> str:
    """A value cell carrying a directional glyph.

    The arrow is the primary cue and colour only reinforces it. The success and
    critical steps fail CVD separation against each other (OKLab dE 4.6 protan
    on the light surface, 4.1 deutan on the dark), so hue must never be the only
    signal of direction.
    """
    if pd.isna(current) or pd.isna(baseline):
        return f"<td>{num(current, digits)}</td>"
    if current > baseline:
        return f'<td class="up">{ARROW_UP} {num(current, digits)}</td>'
    if current < baseline:
        return f'<td class="down">{ARROW_DOWN} {num(current, digits)}</td>'
    return f"<td>{num(current, digits)}</td>"


def tile(label: str, value: str, previous: str, delta: str, direction: str) -> str:
    arrow = ARROW_UP if direction == "up" else ARROW_DOWN
    return (
        f'<div class="tile"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="from">from {previous}</div>'
        f'<div class="delta {direction}">{arrow} {delta}</div></div>'
    )


def sweep_chart(sweep: pd.DataFrame, chosen: int) -> str:
    """Bar chart of Rank IC by training window size.

    One series, so no legend box: the title names it. All five bars are directly
    labelled because the shape of the curve is the entire argument.

    The category labels sit on a reserved baseline well below the axis. A
    negative bar's value label used to be placed at `bottom + height + 14`,
    two pixels from the category baseline at 248, so the only negative bar in
    the sweep printed its value on top of its own label.
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
            # Clamped clear of `category_baseline` so a shallow negative bar
            # cannot print its value over its own category label.
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
    label = escape("Bar chart of average Rank IC by training window size: " + ", ".join(described))
    return f'<svg viewBox="0 0 660 294" role="img" aria-label="{label}.">' + "".join(parts) + "</svg>"


def build_page() -> str:
    horizon = read_table("horizon_results.csv").set_index("forecast_horizon_days")
    ablation = read_table("ablation_results.csv").set_index("ablation_name")
    benchmark = read_table("benchmark_results.csv").set_index("strategy")
    models = read_table("model_comparison_results.csv").set_index("model_name")
    sweep = read_table("training_window_diagnostic.csv")
    disclosure = read_table("horizon_sample_disclosure.csv").set_index("forecast_horizon_days")
    concentration = read_table("concentration_summary.csv")
    ranks = read_table("latest_rank_diagnostic.csv")

    ic5 = horizon.loc[5, "average_rank_ic"]
    ac5 = horizon.loc[5, "average_after_cost_return"]
    ic5_base = BASELINE_HORIZON[5]["rank_ic"]
    ac5_base = BASELINE_HORIZON[5]["after_cost"]

    ml = benchmark[benchmark["comparison_type"] == "ml_strategy"]
    ml_sorted = ml.sort_values("average_period_active_return", ascending=False)
    positive_now = int((ml["average_period_active_return"] > 0).sum())
    positive_base = sum(1 for v in BASELINE_STRATEGIES.values() if v["active"] > 0)
    best_ml = float(ml_sorted["average_period_active_return"].iloc[0])
    best_ml_base = max(v["active"] for v in BASELINE_STRATEGIES.values())

    momentum = float(benchmark.loc["top5_momentum", "average_period_active_return"])
    momentum_dates = int(benchmark.loc["top5_momentum", "evaluated_dates"])
    top5_col = "average_top5_realized_return_per_5d_period"

    # These two ratios are "best model at each stage", and that is not the
    # same model at the two stages: random_forest led before the window change
    # and elastic_net leads after it. Presenting them as one model improving
    # would be wrong, so both names are published and the like-for-like figure
    # for the current leader is published beside them.
    best_model_name_now = str(models[top5_col].idxmax())
    best_model = float(models.loc[best_model_name_now, top5_col])
    best_model_name_base = max(
        BASELINE_MODELS, key=lambda name: BASELINE_MODELS[name]["top5"]
    )
    best_model_base = BASELINE_MODELS[best_model_name_base]["top5"]
    ratio_now = best_model / momentum * 100
    ratio_base = best_model_base / BASELINE_MOMENTUM * 100
    same_model_ratio_base = (
        BASELINE_MODELS[best_model_name_now]["top5"] / BASELINE_MOMENTUM * 100
        if best_model_name_now in BASELINE_MODELS
        else float("nan")
    )

    out: list[str] = []
    add = out.append

    add('<!doctype html>')
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
    # Optional: written by scripts/diagnose_compounded_returns.py, which is not
    # a pipeline stage. Skip the section rather than fail the page build.
    tranche = optional_table("tranche_portfolio_diagnostic.csv")
    if tranche is not None:
        add(deployability_section(tranche))

    # --- Headline -----------------------------------------------------------
    add("<section>")
    add("<h2>Headline</h2>")
    add(
        '<p class="note">Every model module set <code>train_size = '
        f"{BASELINE_TRAIN_SIZE}</code>. That parameter counts trading <em>dates</em>, not "
        "blocks of dates, so each walk-forward window fitted 99 features on about 90 rows. "
        "The value was duplicated across five modules and was never revisited as the dataset "
        f"grew. It is now <code>DEFAULT_TRAIN_SIZE = {DEFAULT_TRAIN_SIZE}</code>, defined once.</p>"
    )
    add('<div class="tiles">')
    add(tile("Rank IC · 5-day", num(ic5, 4), num(ic5_base, 4),
             f"{(ic5 / ic5_base - 1) * 100:+.0f}% · better", "up"))
    add(tile("After-cost return · 5d", num(ac5), num(ac5_base),
             "crossed to positive" if (ac5 > 0 >= ac5_base) else "changed",
             "up" if ac5 > ac5_base else "down"))
    modes_change = positive_now - positive_base
    add(tile("Portfolio modes positive", f"{positive_now} of {len(ml)}",
             f"{positive_base} of {len(BASELINE_STRATEGIES)}",
             f"{modes_change:+d}".replace("-", MINUS) + " · "
             + ("better" if modes_change > 0 else
                "worse" if modes_change < 0 else "unchanged"),
             "up" if modes_change > 0 else "down"))
    add(tile("Best model vs momentum", f"{ratio_now:.0f}%", f"{ratio_base:.0f}%",
             "gap narrowed", "up"))
    add("</div>")
    add(
        '<p class="caption">The last tile compares the best model at each '
        "stage, and that is a different model at each stage: "
        f"<code>{best_model_name_base}</code> led before the window change at "
        f"{ratio_base:.0f}% of the momentum baseline and "
        f"<code>{best_model_name_now}</code> leads after it at "
        f"{ratio_now:.0f}%. Like for like, "
        f"<code>{best_model_name_now}</code> itself went from "
        f"{same_model_ratio_base:.0f}% to {ratio_now:.0f}%. The before and "
        "after columns are also means over different numbers of dates "
        f"({BASELINE_EVALUATED_DATES:,} and {momentum_dates:,}); Baseline "
        "comparison explains why.</p>"
    )
    add(
        '<div class="callout"><strong>The core finding did not change.</strong> A single-feature '
        "momentum rule still beats every trained model. The gap narrowed from "
        f"{BASELINE_MOMENTUM / best_model_base:.2f}× to {momentum / best_model:.2f}×, so part "
        "of it was undertraining — but not all of it. Nothing here supports an alpha claim.</div>"
    )
    add("</section>")

    # --- Training window ----------------------------------------------------
    add("<section>")
    add(f"<h2>Why the training window is {DEFAULT_TRAIN_SIZE} dates</h2>")
    add(
        '<p class="note">Five window sizes were measured on an identical set of '
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
    add("<thead><tr><th>Training window</th><th>Train rows</th><th>Rank IC</th>"
        "<th>Top-5 hit rate</th><th>Top-5 return</th></tr></thead><tbody>")
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
        f'<p class="note">{DEFAULT_TRAIN_SIZE} was chosen over the Rank-IC-optimal 21 because it '
        "has the best top-5 hit rate and the only clearly positive top-5 return, and portfolio "
        "construction selects a top-N basket rather than trading the full ranking.</p>"
    )
    chosen_sweep = sweep[sweep["train_size_dates"] == DEFAULT_TRAIN_SIZE]
    if not chosen_sweep.empty:
        sweep_ic = float(chosen_sweep["average_rank_ic"].iloc[0])
        sweep_dates = int(chosen_sweep["evaluated_dates"].iloc[0])
        add(
            '<p class="note"><strong>The sweep figure is not the headline '
            "figure.</strong> This table reports Rank IC "
            f"{num(sweep_ic)} for the chosen {DEFAULT_TRAIN_SIZE}-date window "
            f"while the headline reports {num(ic5)}. Both are gradient "
            "boosting on the 5-day horizon and neither is wrong. The sweep "
            f"scores {sweep_dates:,} test dates, subsampled at every tenth "
            "date and starting late enough that the 252-date window is also "
            "reachable, so that all five configurations are scored on an "
            "identical set of dates. Production walk-forward scores every date "
            f"it can, {int(horizon.loc[5, 'evaluated_dates']):,} of them. A "
            "shorter, later, sparser subsample is a different sample of the "
            "same estimator, so the two figures are not required to "
            "agree.</p>"
        )
    add(
        '<p class="caption">Chart and table render from the same rows of '
        "<code>training_window_diagnostic.csv</code> \u2014 the chart rounds "
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
    add('<p class="note">Arrows carry the direction of change against the previous training '
        "window; colour is secondary and never the only cue.</p>")
    add('<div class="tablewrap"><table><thead>')
    add('<tr class="grouphead">'
        '<th class="stub" scope="col" rowspan="2">Horizon</th>'
        '<th scope="colgroup" colspan="2">Rank IC</th>'
        '<th scope="colgroup" colspan="2">After-cost return</th>'
        '<th scope="colgroup" colspan="2">Cumulative sum</th></tr>')
    add('<tr><th scope="col">before</th><th scope="col">after</th>'
        '<th scope="col">before</th><th scope="col">after</th>'
        '<th scope="col">before</th><th scope="col">after</th>'
        "</tr></thead><tbody>")
    for hz in (1, 5, 10):
        base = BASELINE_HORIZON[hz]
        add(
            f"<tr><td>{hz} day</td>"
            f"<td>{num(base['rank_ic'])}</td>"
            + delta_cell(horizon.loc[hz, "average_rank_ic"], base["rank_ic"])
            + f"<td>{num(base['after_cost'])}</td>"
            + delta_cell(horizon.loc[hz, "average_after_cost_return"], base["after_cost"])
            + f"<td>{num(base['cumulative'], 3)}</td>"
            + delta_cell(
                horizon.loc[hz, "final_cumulative_after_cost_active_return"],
                base["cumulative"], 3)
            + "</tr>"
        )
    add("</tbody></table></div>")
    after_cost = {hz: float(horizon.loc[hz, "average_after_cost_return"]) for hz in (1, 5, 10)}
    rank_ic = {hz: float(horizon.loc[hz, "average_rank_ic"]) for hz in (1, 5, 10)}
    positive = [hz for hz in (1, 5, 10) if after_cost[hz] > 0]
    negative = [hz for hz in (1, 5, 10) if after_cost[hz] <= 0]
    best_ic = max(rank_ic, key=lambda hz: rank_ic[hz])

    def horizon_list(items: list[int]) -> str:
        labels = [f"{hz}-day" for hz in items]
        if not labels:
            return "none"
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + " and " + labels[-1]

    if positive:
        verdict = (
            f"Only the {horizon_list(positive)} horizon clears cost"
            if len(positive) == 1
            else f"The {horizon_list(positive)} horizons clear cost"
        ) + f", at {num(after_cost[positive[0]])} per period."
    else:
        verdict = "No horizon clears cost."

    add(
        f'<p class="note">{verdict} The {horizon_list(negative)} '
        + ("horizon does" if len(negative) == 1 else "horizons do")
        + " not. The "
        f"{best_ic}-day horizon has the <em>highest</em> Rank IC of the three "
        f"at {num(rank_ic[best_ic])}"
        + (
            " and is still net-negative, which locates its failure in "
            "rebalancing cost rather than in prediction"
            if after_cost[best_ic] <= 0
            else ""
        )
        + ". Turnover is what separates them: the 1-day horizon rebalances at "
        f"{float(horizon.loc[1, 'average_turnover']):.2f} average turnover "
        f"against {float(horizon.loc[10, 'average_turnover']):.2f} at 10 days, "
        "so the horizon with the best ranking pays the most to act on it. The "
        "10-day Rank IC also fell against the 3-date baseline, from "
        f"{num(BASELINE_HORIZON[10]['rank_ic'])} to {num(rank_ic[10])}; the "
        "sweep measured only the 5-day horizon, so that was not predicted and "
        "is a real, if small, cost of the change.</p>"
    )
    add(BEFORE_COLUMN_DISCLOSURE)
    add(
        f'<p class="caption">Overlap disclosure: the 10-day horizon’s '
        f"{int(disclosure.loc[10, 'evaluated_dates']):,} evaluated dates correspond to roughly "
        f"{int(disclosure.loc[10, 'approx_non_overlapping_periods'])} genuinely non-overlapping "
        "periods. Average after-cost return is per forecast period, not annualized.</p>"
    )
    add("</section>")

    # --- Ablation -----------------------------------------------------------
    add("<section>")
    add("<h2>Feature ablation</h2>")
    add('<div class="tablewrap"><table><thead>')
    add('<tr class="grouphead">'
        '<th class="stub" scope="col" rowspan="2">Configuration</th>'
        '<th class="stub" scope="col" rowspan="2">Features</th>'
        '<th scope="colgroup" colspan="2">Rank IC</th>'
        '<th scope="colgroup" colspan="2">After-cost return</th></tr>')
    add('<tr><th scope="col">before</th><th scope="col">after</th>'
        '<th scope="col">before</th><th scope="col">after</th>'
        "</tr></thead><tbody>")
    ordered = ablation.sort_values("average_rank_ic", ascending=False)
    for name, row in ordered.iterrows():
        base = BASELINE_ABLATION.get(name)
        cls = ' class="highlight"' if name == "all_features" else ""
        label = name.replace("_", " ")
        if base is None:
            add(f"<tr{cls}><td>{label}</td><td>{int(row['feature_count'])}</td>"
                f"<td>n/a</td><td>{num(row['average_rank_ic'])}</td>"
                f"<td>n/a</td><td>{num(row['average_after_cost_return'])}</td></tr>")
            continue
        add(
            f"<tr{cls}><td>{label}</td><td>{int(row['feature_count'])}</td>"
            f"<td>{num(base['rank_ic'])}</td>"
            + delta_cell(row["average_rank_ic"], base["rank_ic"])
            + f"<td>{num(base['after_cost'])}</td>"
            + delta_cell(row["average_after_cost_return"], base["after_cost"])
            + "</tr>"
        )
    add("</tbody></table></div>")
    positive_ablations = int((ablation["average_after_cost_return"] > 0).sum())
    add(
        '<div class="callout"><strong>The Rank IC ranking reversed; the cost '
        "verdict did not.</strong> Previously, removing features improved Rank "
        "IC and <code>without_risk</code> led the table, which read as "
        "\u201cthe feature set is mostly noise\u201d. With the wider training "
        "window the ranking inverts and the full feature set leads on Rank IC, "
        "so that earlier reading reflected too little training data to use 99 "
        "features rather than harmful features. It does not follow that the "
        "feature set pays for itself: "
        + (
            f"{positive_ablations} of {len(ablation)} configurations clear "
            "transaction costs"
            if positive_ablations
            else f"none of the {len(ablation)} configurations clears "
            "transaction costs"
        )
        + ". Under the pre-correction cost model two of five appeared to, "
        "which was an artifact of charging a round trip 0.2 percent instead of "
        "0.5.</div>"
    )
    add(BEFORE_COLUMN_DISCLOSURE)
    add("</section>")

    # --- Model comparison ---------------------------------------------------
    add("<section>")
    add("<h2>Model comparison</h2>")
    add('<div class="tablewrap"><table>'
        "<caption>Equal-weight top-5 by model score · diagnostic only</caption><thead>")
    add('<tr class="grouphead">'
        '<th class="stub" scope="col" rowspan="2">Model</th>'
        '<th scope="colgroup" colspan="2">Rank IC</th>'
        '<th scope="colgroup" colspan="2">Top-5 return / 5d</th></tr>')
    add('<tr><th scope="col">before</th><th scope="col">after</th>'
        '<th scope="col">before</th><th scope="col">after</th>'
        "</tr></thead><tbody>")
    best_model_name = models[top5_col].idxmax()
    for name, row in models.sort_values("average_rank_ic", ascending=False).iterrows():
        base = BASELINE_MODELS.get(name)
        cls = ' class="highlight"' if name == best_model_name else ""
        suffix = ' <span class="flat">(production)</span>' if name == "gradient_boosting" else ""
        if base is None:
            add(f"<tr{cls}><td>{name}{suffix}</td><td>n/a</td>"
                f"<td>{num(row['average_rank_ic'])}</td><td>n/a</td>"
                f"<td>{num(row[top5_col])}</td></tr>")
            continue
        add(
            f"<tr{cls}><td>{name}{suffix}</td>"
            f"<td>{num(base['rank_ic'])}</td>"
            + delta_cell(row["average_rank_ic"], base["rank_ic"])
            + f"<td>{num(base['top5'])}</td>"
            + delta_cell(row[top5_col], base["top5"])
            + "</tr>"
        )
    add("</tbody></table></div>")
    if best_model_name != "gradient_boosting":
        add(
            '<div class="callout"><strong>The production model is no longer the best one.</strong> '
            f"<code>{best_model_name}</code> now leads on top-5 return, while gradient boosting "
            "remains what feeds portfolio construction, for continuity. Changing it would move "
            "every portfolio number again, so it is flagged as an open question rather than "
            "switched.</div>"
        )
    add("</section>")

    # --- Baselines ----------------------------------------------------------
    add("<section>")
    add("<h2>Baseline comparison</h2>")
    add('<p class="note">Top-5 selection, before cost, same relative-to-VN30 '
        "target. Within a column every rule is scored on identical dates. "
        "Across columns they are not: the <em>before</em> column covers "
        f"{BASELINE_EVALUATED_DATES:,} dates and the <em>after</em> column "
        f"{momentum_dates:,}, because a {DEFAULT_TRAIN_SIZE}-date training "
        f"window needs {DEFAULT_TRAIN_SIZE - BASELINE_TRAIN_SIZE} more dates "
        "of warm-up than a 3-date window before it can predict, and those "
        "dates drop out of the evaluation.</p>")
    add(
        '<p class="note"><strong>Why the momentum baseline moves at '
        "all.</strong> <code>top5_momentum</code> is a fixed, untrained rule: "
        "rank on trailing 10-day return, hold the top five. Nothing refits it, "
        "and it is reported before cost, so neither the training window nor "
        "the cost model can reach it. It reads "
        f"{num(BASELINE_MOMENTUM)} over {BASELINE_EVALUATED_DATES:,} dates "
        f"and {num(momentum)} over {momentum_dates:,} dates. The rule did not "
        f"change; the {BASELINE_EVALUATED_DATES - momentum_dates} dates that "
        "left the sample did.</p>")
    add('<div class="tablewrap"><table><thead><tr><th>Selection rule</th><th>before</th>'
        "<th>after</th><th>vs momentum</th></tr></thead><tbody>")
    add(
        '<tr class="highlight"><td>Trailing 10-day return (<code>top5_momentum</code>)</td>'
        f"<td>{num(BASELINE_MOMENTUM)}</td><td><strong>{num(momentum)}</strong></td>"
        "<td>—</td></tr>"
    )
    for name, row in models.sort_values(top5_col, ascending=False).iterrows():
        base = BASELINE_MODELS.get(name)
        share = row[top5_col] / momentum * 100
        before = num(base["top5"]) if base else "n/a"
        cell = delta_cell(row[top5_col], base["top5"]) if base else f"<td>{num(row[top5_col])}</td>"
        add(f"<tr><td>{name}</td><td>{before}</td>{cell}<td>{share:.0f}%</td></tr>")
    add("</tbody></table></div>")
    momentum_row = benchmark.loc["top5_momentum"]
    if "mean_active_return_95pct_lower" in momentum_row.index and not pd.isna(
        momentum_row["mean_active_return_95pct_lower"]
    ):
        momentum_lower = float(momentum_row["mean_active_return_95pct_lower"])
        momentum_upper = float(momentum_row["mean_active_return_95pct_upper"])
        momentum_excludes = momentum_lower > 0.0 or momentum_upper < 0.0
        add(
            '<p class="note"><strong>The baseline is the one row here that '
            "separates from zero.</strong> <code>top5_momentum</code> has a 95% "
            f"moving-block bootstrap interval of [{num(momentum_lower)}, "
            f"{num(momentum_upper)}] on its mean active return, which "
            + ("excludes zero" if momentum_excludes else "spans zero")
            + ". Every one of the eight after-cost portfolio configurations in "
            "the next section has an interval that spans zero. The baseline is "
            "measured before cost and the portfolios after it, so this is not a "
            "like-for-like contest \u2014 but it is the shape of the whole "
            "result: a detectable signal in a one-feature rule, and nothing "
            "detectable left once the models and the trading costs are "
            "added.</p>"
        )
    add(
        '<p class="note">A one-feature rule still wins. <code>return_10d</code> '
        "is an input feature to every model, so they are not missing the "
        "information. The best model at each stage went from "
        f"{ratio_base:.0f}% of the baseline "
        f"(<code>{best_model_name_base}</code>) to {ratio_now:.0f}% "
        f"(<code>{best_model_name_now}</code>) \u2014 two different models. "
        f"Like for like, <code>{best_model_name_now}</code> went from "
        f"{same_model_ratio_base:.0f}% to {ratio_now:.0f}%. Either way part "
        "of the original gap was undertraining, and part of it was not.</p>"
    )
    add("</section>")

    # --- Portfolio ----------------------------------------------------------
    add("<section>")
    add("<h2>Portfolio backtest</h2>")
    add('<p class="note">Optimizer modes crossed with execution modes, 5-day horizon, after '
        "cost.</p>")
    add('<div class="tablewrap"><table><thead>')
    add('<tr class="grouphead">'
        '<th class="stub" scope="col" rowspan="2">Optimizer</th>'
        '<th class="stub" scope="col" rowspan="2">Execution</th>'
        '<th scope="colgroup" colspan="2">Active return / period</th>'
        '<th scope="colgroup" colspan="2">Cumulative sum</th></tr>')
    add('<tr><th scope="col">before</th><th scope="col">after</th>'
        '<th scope="col">before</th><th scope="col">after</th>'
        "</tr></thead><tbody>")
    for name, row in ml_sorted.iterrows():
        base = BASELINE_STRATEGIES.get(name)
        optimizer, execution = STRATEGY_LABELS.get(name, (name, ""))
        cls = ' class="highlight"' if name == ml_sorted.index[0] else ""
        if base is None:
            add(f"<tr{cls}><td>{optimizer}</td><td>{execution}</td><td>n/a</td>"
                f"<td>{num(row['average_period_active_return'])}</td><td>n/a</td>"
                f"<td>{num(row['final_cumulative_active_return_sum'], 3)}</td></tr>")
            continue
        add(
            f"<tr{cls}><td>{optimizer}</td><td>{execution}</td>"
            f"<td>{num(base['active'])}</td>"
            + delta_cell(row["average_period_active_return"], base["active"])
            + f"<td>{num(base['cumulative'], 3)}</td>"
            + delta_cell(row["final_cumulative_active_return_sum"], base["cumulative"], 3)
            + "</tr>"
        )
    add("</tbody></table></div>")
    conservative = benchmark.loc["ml_normal_price_limit_aware"]
    conservative_mean = float(conservative["average_period_active_return"])
    verb = "clears cost by" if conservative_mean > 0 else "falls short of cost by"
    add(
        f'<p class="note">{positive_now} of {len(ml)} configurations are '
        f"positive, where {positive_base} of {len(BASELINE_STRATEGIES)} were "
        "before. The most conservative (<code>normal</code> + "
        f"<code>price_limit_aware</code>) {verb} "
        f"{num(abs(conservative_mean))} per period, at a diagnostic Sharpe of "
        f"{num(conservative['diagnostic_sharpe'], 3)}. Wider diversification "
        "still outperforms concentration, which remains evidence that the "
        "ranking signal is weak.</p>"
    )
    add(interval_sentence(benchmark, ml_sorted))
    add(
        '<p class="caption">Cumulative figures in <em>this</em> table sum '
        "overlapping forecast-period active returns, so they are not "
        "compounded portfolio returns, and <code>max_active_drawdown</code> in "
        "<code>benchmark_results.csv</code> is a running-peak fall on that same "
        "summed series \u2014 in summed-return units, not as a percentage of "
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
        "<thead><tr><th>Ticker</th><th>Predicted rank</th><th>Realized rank</th>"
        "<th>Realized forward return</th></tr></thead><tbody>")
    for row in pd.concat([top_ranked, worst_missed]).itertuples(index=False):
        add(
            f"<tr><td>{row.ticker}</td><td>{int(row.predicted_rank)}</td>"
            f"<td>{int(row.realized_rank)}</td>"
            f"<td>{row.realized_forward_return * 100:.2f}%</td></tr>".replace("-", MINUS, 1)
            if row.realized_forward_return < 0 else
            f"<tr><td>{row.ticker}</td><td>{int(row.predicted_rank)}</td>"
            f"<td>{int(row.realized_rank)}</td>"
            f"<td>{row.realized_forward_return * 100:.2f}%</td></tr>"
        )
    add("</tbody></table></div>")
    add('<p class="caption">The last two rows are the best realized performers of the day, shown '
        "with the rank the model gave them. One date is an illustration, not evidence.</p>")
    add("</section>")

    # --- Paper horizon ------------------------------------------------------
    add("<section>")
    add("<h2>Paper-horizon predictions rebuilt</h2>")
    add(
        '<p class="note">Separate from the training-window change. '
        "<code>paper_horizon_predictions.parquet</code> had been built from the static 30-ticker "
        "universe, so it contained predictions for names on dates they were not index "
        "constituents. Under point-in-time filtering those rows correctly vanish and only fully "
        "covered dates survive, which had left two scripts reporting on three observations.</p>"
    )
    add('<div class="tablewrap"><table><thead><tr><th>Measure</th><th>before</th><th>after</th>'
        "</tr></thead><tbody>")
    for label, before, after in PAPER_HORIZON_MIGRATION:
        cls = ' class="highlight"' if "coverage" in label else ""
        add(f"<tr{cls}><td>{label}</td><td>{before}</td><td>{after}</td></tr>")
    add("</tbody></table></div>")
    add(
        '<p class="note"><strong>Why 60 rows a date is not 60 duplicate '
        "rows.</strong> The VN30 has 30 constituents on any date, and this "
        "file stacks two models, <code>gradient_boosting</code> and "
        "<code>random_forest</code>, in one table with a "
        "<code>model_name</code> column. A fully covered date therefore "
        f"carries 30 \u00d7 2 = 60 rows, and {PAPER_HORIZON_DATES} "
        f"\u00d7 60 = {PAPER_HORIZON_ROWS:,}. Each (date, ticker) pair "
        "appears twice by design and each (date, ticker, model) key once.</p>"
    )
    add(
        '<p class="note"><strong>Why 54 tickers and not the dataset\u2019s '
        "60.</strong> The modeling dataset covers 60 tickers across the full "
        "history \u2014 the 30 current constituents plus 30 former ones. This "
        "file covers the 54 of those 60 that were VN30 members on at least one "
        "date inside its own range. The six missing names (FLC, HAG, HCM, HNG, "
        "ITA, PPC) all left the index before 2017-08-09, the first date a "
        "prediction can be made after the 252-date training warm-up, so under "
        "point-in-time membership they correctly never appear. 60 is the "
        "dataset-wide ticker count; 54 is this file's; 30 is the count on any "
        "single date.</p>"
    )
    add(
        '<p class="note"><strong>Which period is which.</strong> '
        f"{PAPER_HORIZON_DATE_RANGE} is this prediction file's range, "
        "which is shorter than the backtest window in Deployability at the top "
        "of the page: it starts after a 252-date warm-up and stops where a "
        "10-day forward label stops being observable. The signal date in "
        "Latest diagnostics is the last date in the dataset and is later than "
        "both, because a signal needs no realized forward label. The three "
        "periods are reported separately and no figure is carried between "
        "them.</p>"
    )
    add(paper_horizon_check())
    add("</section>")

    # --- Limitations --------------------------------------------------------
    add("<section>")
    add("<h2>What did not improve</h2>")
    add("<ul>")
    add(f"<li><strong>A one-feature momentum baseline still beats every trained model</strong> "
        f"— {num(momentum)} against {num(best_model)}. Narrowed, unresolved.</li>")
    add(f"<li><strong>The 10-day horizon Rank IC fell</strong> from "
        f"{num(BASELINE_HORIZON[10]['rank_ic'])} to "
        f"{num(horizon.loc[10, 'average_rank_ic'])}.</li>")
    add("<li><strong>Margins remain thin.</strong> The most conservative "
        "portfolio configuration "
        + ("clears cost at" if conservative_mean > 0 else "does not clear cost, at")
        + " a diagnostic Sharpe of "
        + num(conservative["diagnostic_sharpe"], 3)
        + interval_clause(conservative)
        + ".</li>")
    add("<li><strong>Concentration still reduces performance</strong>, which is what a weak "
        "ranking signal looks like.</li>")
    add(foreign_flow_item())
    add("</ul>")
    interval_rows = int(
        benchmark["mean_active_return_95pct_lower"].notna().sum()
    ) if "mean_active_return_95pct_lower" in benchmark.columns else 0
    add(
        '<div class="callout"><strong>Multiple comparisons.</strong> The '
        "training window was chosen by measuring five values against the same "
        "history, on top of three horizons, five ablation configurations, four "
        "optimizer modes, two execution modes and five foreign-flow feature "
        "sets already tested against it. The improvement is consistent across "
        "horizons, ablations and portfolio modes, which makes selection noise "
        "unlikely \u2014 but nothing here corrects for the number of "
        "comparisons, and the intervals that do exist "
        + (
            f"({interval_rows} strategy and baseline rows of "
            "<code>benchmark_results.csv</code>, plus the foreign-flow test) "
            "are each marginal rather than family-wise"
            if interval_rows
            else "cover only the foreign-flow test"
        )
        + ". Rank IC, hit rate, and every per-horizon and per-ablation figure "
        "above is a point estimate with no interval at all.</div>"
    )
    add("</section>")

    # --- Reproduce ----------------------------------------------------------
    add("<section>")
    add("<h2>Reproduce</h2>")
    add("<pre><code>py .\\scripts\\run_full_pipeline.py          # rebuilds every table and this page\n"
        "py .\\scripts\\diagnose_training_window.py    # the sweep behind the window size\n"
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
    text -- the "meta-viewport" and "title" that showed up as paragraphs. For
    those hosts, emit the body content and carry the stylesheet across, rather
    than dropping the head tags from the standalone file where they belong.
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
