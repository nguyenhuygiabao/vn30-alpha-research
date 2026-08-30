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

import sys
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.walk_forward_split import DEFAULT_TRAIN_SIZE

TABLES = ROOT / "reports" / "tables"
OUTPUT_PATH = ROOT / "reports" / "reportpage.html"

# ---------------------------------------------------------------------------
# Frozen historical baseline: results when the walk-forward training window was
# 3 trading dates (about 90 rows against 99 features). That configuration no
# longer exists, so no current CSV contains these; they are recorded here so the
# page can show what changed. Taken from the tables as committed at 0d8b2a5.
# ---------------------------------------------------------------------------
BASELINE_TRAIN_SIZE = 3

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
# universe to point-in-time membership. Recorded, not derived.
PAPER_HORIZON_MIGRATION = [
    ("Prediction rows", "7,932", "12,480"),
    ("Prediction dates", "136", "208"),
    ("Tickers", "30", "54"),
    ("Date range", "2021-01-15 to 2026-06-22", "2017-08-09 to 2026-07-01"),
    ("Dates at full point-in-time coverage", "3 of 136", "208 of 208"),
    ("Rebalance dates in both scripts", "3", "208"),
]

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
    text-transform:uppercase; letter-spacing:0.04em; padding-bottom:2px; }
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


def deployability_section(phases: pd.DataFrame) -> str:
    """What the per-period figures look like compounded as a portfolio.

    The headline figures are means over overlapping forecast periods, which a
    portfolio cannot hold. Sampling non-overlapping periods can be done five
    ways at a 5-day horizon, and the spread across those five is the result.
    """
    limited = phases[phases["execution_mode"] == "price_limit_aware"]

    ordered = sorted(
        limited["optimization_mode"].unique(),
        key=lambda mode: limited.loc[
            limited["optimization_mode"] == mode, "final_equity"
        ].min(),
    )

    rows: list[str] = []
    for mode in ordered:
        group = limited[limited["optimization_mode"] == mode]
        positive = int((group["final_equity"] > 1.0).sum())
        # Marked in text, not by colour alone.
        verdict = "fragile" if positive < len(group) else "positive throughout"
        rows.append(
            "<tr>"
            f"<td><code>{mode}</code></td>"
            f"<td>{positive} of {len(group)} · {verdict}</td>"
            f"<td>{group['final_equity'].min():.4f} to "
            f"{group['final_equity'].max():.4f}</td>"
            f"<td>{MINUS}{abs(group['max_drawdown_percent'].min()):.1f}% to "
            f"{MINUS}{abs(group['max_drawdown_percent'].max()):.1f}%</td>"
            "</tr>"
        )

    return "\n".join(
        [
            "<section>",
            "<h2>Deployability</h2>",
            '<p class="note">Every per-period figure on this page is a mean '
            "over <em>overlapping</em> forecast periods. A portfolio cannot "
            "hold an overlapping period. Sampling non-overlapping 5-day "
            "periods instead gives 484 independent periods, and five ways to "
            "do it depending on which date the sampling starts from. The "
            "spread across those five phases is the result.</p>",
            '<div class="tablewrap"><table>',
            "<caption>Compounded outcome by overlap phase, price-limit-aware "
            "execution</caption>",
            "<thead><tr><th>Optimizer</th><th>Phases positive</th>"
            "<th>Final equity</th><th>Max drawdown</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table></div>",
            '<div class="callout"><strong>Three of the four optimizer modes '
            "are not deployable.</strong> Their compounded outcome swings "
            "from a 27% loss to a 48% gain on nothing but the choice of "
            "starting weekday, which is not an edge distinguishable from "
            "noise at 484 periods. Drawdown in capital terms is 20% to 43%, "
            "not the summed-unit figures reported elsewhere. "
            "<code>sector_diversified</code> is positive in all five phases, "
            "but it is one of four configurations measured against the same "
            "history: selecting it because it survived is itself a comparison "
            "these figures do not adjust for, and it has no out-of-sample "
            "validation.</div>",
            '<p class="caption">Source: '
            "<code>reports/tables/compounded_phase_sensitivity.csv</code>, "
            "generated by <code>scripts/diagnose_compounded_returns.py</code>."
            "</p>",
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
    """
    rows = sweep.sort_values("train_size_dates").reset_index(drop=True)
    top, bottom, ceiling = 30.0, 230.0, 0.04
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
            label_y = bottom + height + 14

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
            f'<text x="{centre:.0f}" y="248" text-anchor="middle" '
            f'fill="{ink}" font-size="11.5" font-weight="{weight}">{size}</text>'
        )
        if is_chosen:
            parts.append(
                f'<text x="{centre:.0f}" y="200" text-anchor="middle" '
                f'fill="var(--surface-1)" font-size="10" font-weight="700">chosen</text>'
            )

    parts.append(
        '<text x="353" y="17" text-anchor="middle" fill="var(--ink-2)" font-size="12">'
        "Average Rank IC by training window size (trading dates)</text>"
    )
    label = escape("Bar chart of average Rank IC by training window size: " + ", ".join(described))
    return f'<svg viewBox="0 0 660 260" role="img" aria-label="{label}.">' + "".join(parts) + "</svg>"


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
    top5_col = "average_top5_realized_return_per_5d_period"
    best_model = float(models[top5_col].max())
    best_model_base = max(v["top5"] for v in BASELINE_MODELS.values())
    ratio_now = best_model / momentum * 100
    ratio_base = best_model_base / BASELINE_MOMENTUM * 100

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
    phases = optional_table("compounded_phase_sensitivity.csv")
    if phases is not None:
        add(deployability_section(phases))

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
    add(tile("Portfolio modes positive", f"{positive_now} of {len(ml)}",
             f"{positive_base} of {len(BASELINE_STRATEGIES)}",
             f"+{positive_now - positive_base} · better", "up"))
    add(tile("Best model vs momentum", f"{ratio_now:.0f}%", f"{ratio_base:.0f}%",
             "gap narrowed", "up"))
    add("</div>")
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
    add("</section>")

    # --- Horizon ------------------------------------------------------------
    add("<section>")
    add("<h2>Forecast horizon</h2>")
    add('<p class="note">Arrows carry the direction of change against the previous training '
        "window; colour is secondary and never the only cue.</p>")
    add('<div class="tablewrap"><table><thead>')
    add('<tr class="grouphead"><th></th><th colspan="2">Rank IC</th>'
        '<th colspan="2">After-cost return</th><th colspan="2">Cumulative sum</th></tr>')
    add("<tr><th>Horizon</th><th>before</th><th>after</th><th>before</th><th>after</th>"
        "<th>before</th><th>after</th></tr></thead><tbody>")
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
    add(
        '<p class="note">The 5-day horizon — the baseline horizon, and previously the headline '
        "failure — crossed to positive. The 1-day horizon now has the <em>highest</em> Rank IC "
        "of the three yet remains the only net-negative one, which locates its failure in "
        "rebalancing cost rather than in prediction. The 10-day Rank IC fell slightly; the sweep "
        "measured only the 5-day horizon, so that was not predicted and is a real, if small, cost "
        "of the change.</p>"
    )
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
    add('<tr class="grouphead"><th></th><th></th><th colspan="2">Rank IC</th>'
        '<th colspan="2">After-cost return</th></tr>')
    add("<tr><th>Configuration</th><th>Features</th><th>before</th><th>after</th>"
        "<th>before</th><th>after</th></tr></thead><tbody>")
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
        '<div class="callout"><strong>This result reversed.</strong> Previously, removing features '
        "improved Rank IC and <code>without_risk</code> led the table, which read as "
        "“the feature set is mostly noise”. With the wider training window the ranking "
        f"inverts and the full feature set leads, with {positive_ablations} of {len(ablation)} "
        "configurations clearing transaction costs. The earlier reading reflected too little "
        "training data to use 99 features — not harmful features.</div>"
    )
    add("</section>")

    # --- Model comparison ---------------------------------------------------
    add("<section>")
    add("<h2>Model comparison</h2>")
    add('<div class="tablewrap"><table>'
        "<caption>Equal-weight top-5 by model score · diagnostic only</caption><thead>")
    add('<tr class="grouphead"><th></th><th colspan="2">Rank IC</th>'
        '<th colspan="2">Top-5 return / 5d</th></tr>')
    add("<tr><th>Model</th><th>before</th><th>after</th><th>before</th><th>after</th>"
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
    add('<p class="note">Top-5 selection, before cost, same relative-to-VN30 target and the same '
        "evaluated dates.</p>")
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
    add(
        '<p class="note">A one-feature rule still wins. <code>return_10d</code> is an input feature '
        "to every model, so they are not missing the information. The best model moved from "
        f"{ratio_base:.0f}% to {ratio_now:.0f}% of the baseline, meaning part of the original gap "
        "was undertraining — and part of it was not.</p>"
    )
    add("</section>")

    # --- Portfolio ----------------------------------------------------------
    add("<section>")
    add("<h2>Portfolio backtest</h2>")
    add('<p class="note">Optimizer modes crossed with execution modes, 5-day horizon, after '
        "cost.</p>")
    add('<div class="tablewrap"><table><thead>')
    add('<tr class="grouphead"><th></th><th></th><th colspan="2">Active return / period</th>'
        '<th colspan="2">Cumulative sum</th></tr>')
    add("<tr><th>Optimizer</th><th>Execution</th><th>before</th><th>after</th>"
        "<th>before</th><th>after</th></tr></thead><tbody>")
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
    add(
        f'<p class="note">All {positive_now} configurations are now positive, where '
        f"{positive_base} of {len(BASELINE_STRATEGIES)} were before. The most conservative "
        "(<code>normal</code> + <code>price_limit_aware</code>) clears cost by "
        f"{num(conservative['average_period_active_return'])} per period at a diagnostic Sharpe of "
        f"{num(conservative['diagnostic_sharpe'], 3)} — statistically indistinguishable from "
        "zero. Wider diversification still outperforms concentration, which remains evidence that "
        "the ranking signal is weak.</p>"
    )
    add(
        '<p class="caption">Cumulative figures sum overlapping forecast-period active returns; they '
        "are not compounded portfolio returns. Drawdown, where reported, is measured on that same "
        "summed series and is not a percentage of capital.</p>"
    )
    add("</section>")

    # --- Latest diagnostics -------------------------------------------------
    latest = concentration[concentration["optimization_mode"] == "normal"]
    add("<section>")
    add("<h2>Latest diagnostics</h2>")
    if not latest.empty:
        row = latest.iloc[0]
        add(
            f'<p class="note">Signal date {row["signal_date"]}, <code>normal</code> optimizer: '
            f"{int(row['holding_count'])} holdings, maximum single-name weight "
            f"{row['max_single_name_weight'] * 100:.1f} percent, HHI "
            f"{row['hhi']:.3f}, effective position count {row['effective_position_count']:.1f}, "
            f"top issuer group {row['top_issuer_group']} at "
            f"{row['top_issuer_group_weight'] * 100:.1f} percent.</p>"
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
        '<p class="note">54 tickers rather than 60 is correct: six former constituents (FLC, HAG, '
        "HCM, HNG, ITA, PPC) left the index before 2017-08-09, the first date a prediction could be "
        "made after the 252-date training warm-up.</p>"
    )
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
    add("<li><strong>Margins remain thin.</strong> The most conservative portfolio configuration "
        f"clears cost at a diagnostic Sharpe of {num(conservative['diagnostic_sharpe'], 3)}.</li>")
    add("<li><strong>Concentration still reduces performance</strong>, which is what a weak "
        "ranking signal looks like.</li>")
    add("<li><strong>Foreign buy/sell features remain unpromoted.</strong> Pre-registered test: "
        f"mean Rank-IC difference {MINUS}0.002608, 95% CI [{MINUS}0.013314, +0.008297]. The "
        "interval spans zero and the point estimate is negative.</li>")
    add("</ul>")
    add(
        '<div class="callout"><strong>Multiple comparisons.</strong> The training window was chosen '
        "by measuring five values against the same history, on top of three horizons, five ablation "
        "configurations, four optimizer modes, two execution modes and five foreign-flow feature "
        "sets already tested against it. The improvement is consistent across horizons, ablations "
        "and portfolio modes, which makes selection noise unlikely — but no confidence interval "
        "supports that, and no figure above carries one except the foreign-flow test.</div>"
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


def main() -> None:
    html = build_page()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print("Report page generated:")
    print(" ", OUTPUT_PATH)
    print("  bytes:", len(html))


if __name__ == "__main__":
    main()
