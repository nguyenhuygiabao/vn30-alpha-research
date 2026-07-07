from pathlib import Path
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
INTERACTIVE = REPORTS / "interactive"

FIG_BG = "#0b1020"
AX_BG = "#111827"
GRID = "#334155"
TEXT = "#e5e7eb"
MUTED = "#cbd5e1"
EDGE = "#475569"
ACCENT = "#60a5fa"
ACCENT_2 = "#34d399"
NEG = "#fb7185"

DISPLAY_LABELS = {
    "all_features": "All features",
    "without_volume_liquidity": "No volume/liquidity",
    "without_risk": "No risk features",
    "without_herding": "No herding features",
    "without_price_limit": "No price-limit features",

    "ml_normal_normal": "ML: standard",
    "ml_normal_price_limit_aware": "ML: price-limit aware",
    "ml_herding_aware_normal": "ML: herding-aware",
    "ml_herding_aware_price_limit_aware": "ML: herding + price-limit aware",

    "top5_momentum": "Top-5 momentum",
    "top5_reversal": "Top-5 reversal",
    "low_volatility_top10": "Low-volatility top 10",
    "equal_weight_all": "Equal-weight universe",

    "Best Feature Set": "Best Ablation Variant",
    "BEST FEATURE SET": "BEST ABLATION VARIANT",
    "Best ML vs Naive Baseline": "ML vs Best Naive Baseline",
    "BEST ML VS NAIVE BASELINE": "ML VS BEST NAIVE BASELINE",

    "Interactive Weekly Active Drawdown": "Interactive Active-Return Drawdown",
    "Weekly active drawdown": "Active-return drawdown",
    "weekly active drawdown": "active-return drawdown",
}


def pretty_label(value):
    text = str(value)
    if text in DISPLAY_LABELS:
        return DISPLAY_LABELS[text]

    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()

    replacements = {
        "rank ic": "Rank IC",
        "ml": "ML",
        "vn30": "VN30",
        "hhi": "HHI",
        "adv": "ADV",
        "ohlcv": "OHLCV",
        "rsi": "RSI",
        "macd": "MACD",
        "atr": "ATR",
        "vol": "volatility",
        "z": "z-score",
    }

    lower = text.lower()
    if lower in replacements:
        return replacements[lower]

    return text.title().replace("Ml", "ML").replace("Vn30", "VN30").replace("Ic", "IC")


def percent_if_small(series):
    vals = pd.to_numeric(series, errors="coerce")
    if vals.dropna().empty:
        return vals
    if vals.abs().max() <= 1.5:
        return vals * 100
    return vals


def apply_public_labels_to_text_files():
    paths = []
    paths.extend(REPORTS.rglob("*.html"))
    paths.extend(REPORTS.rglob("*.md"))

    readme = ROOT / "README.md"
    if readme.exists():
        paths.append(readme)

    changed = []

    for path in sorted(set(paths)):
        text = path.read_text(encoding="utf-8", errors="replace")
        original = text

        for raw, clean in DISPLAY_LABELS.items():
            text = text.replace(raw, clean)

        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
            changed.append(path.relative_to(ROOT))

    return changed


def setup_matplotlib():
    plt.rcParams.update({
        "figure.facecolor": FIG_BG,
        "axes.facecolor": AX_BG,
        "savefig.facecolor": FIG_BG,
        "axes.edgecolor": EDGE,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.titlecolor": TEXT,
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 180,
    })


def style_axis(ax):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)


def save_fig(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=FIG_BG, edgecolor=FIG_BG)
    plt.close(fig)
    return out.relative_to(ROOT)


def existing_csv(name):
    path = TABLES / name
    return path if path.exists() else None


def find_col(df, candidates, contains=None):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if contains:
        for col in df.columns:
            low = col.lower()
            if all(token in low for token in contains):
                return col

    return None


def build_horizon_figures():
    path = existing_csv("horizon_results.csv")
    if path is None:
        return []

    df = pd.read_csv(path).copy()
    horizon_col = find_col(df, ["forecast_horizon_days", "horizon", "horizon_days"])
    sharpe_col = find_col(df, ["diagnostic_sharpe", "sharpe"])
    rank_col = find_col(df, ["average_rank_ic", "rank_ic"])

    if horizon_col is None:
        return []

    df = df.sort_values(horizon_col)
    df["horizon_label"] = df[horizon_col].astype(int).astype(str) + "d"

    written = []

    if sharpe_col:
        fig, ax = plt.subplots(figsize=(9.5, 5.3))
        colors = [NEG if v < 0 else ACCENT for v in pd.to_numeric(df[sharpe_col], errors="coerce")]
        ax.bar(df["horizon_label"], df[sharpe_col], color=colors, edgecolor="none")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title("Diagnostic Sharpe by forecast horizon", pad=14)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Diagnostic Sharpe")
        style_axis(ax)
        written.append(save_fig(fig, FIGURES / "horizon_diagnostic_sharpe.png"))

    if rank_col:
        fig, ax = plt.subplots(figsize=(9.5, 5.3))
        colors = [NEG if v < 0 else ACCENT_2 for v in pd.to_numeric(df[rank_col], errors="coerce")]
        ax.bar(df["horizon_label"], df[rank_col], color=colors, edgecolor="none")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title("Average Rank IC by forecast horizon", pad=14)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Average Rank IC")
        style_axis(ax)
        written.append(save_fig(fig, FIGURES / "horizon_rank_ic.png"))

    return written


def build_ablation_figure():
    path = existing_csv("ablation_results.csv")
    if path is None:
        return []

    df = pd.read_csv(path).copy()
    name_col = find_col(df, ["ablation_name", "feature_set", "scenario", "name"])
    sharpe_col = find_col(df, ["diagnostic_sharpe", "sharpe"])

    if name_col is None or sharpe_col is None:
        return []

    df["label"] = df[name_col].map(pretty_label)
    df[sharpe_col] = pd.to_numeric(df[sharpe_col], errors="coerce")
    df = df.dropna(subset=[sharpe_col]).sort_values(sharpe_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [NEG if v < 0 else ACCENT for v in df[sharpe_col]]
    ax.barh(df["label"], df[sharpe_col], color=colors, edgecolor="none")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title("Diagnostic Sharpe by ablation variant", pad=14)
    ax.set_xlabel("Diagnostic Sharpe")
    ax.set_ylabel("")
    style_axis(ax)

    return [save_fig(fig, FIGURES / "ablation_diagnostic_sharpe.png")]


def build_feature_importance_figure():
    candidates = []
    for path in sorted(TABLES.glob("*.csv")):
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        feature_col = find_col(df, ["feature", "feature_name", "variable", "input"])
        importance_col = find_col(df, ["importance", "feature_importance", "average_feature_importance"], contains=["importance"])

        if feature_col and importance_col:
            score = 10 if "gradient" in path.stem.lower() else 0
            candidates.append((score, path, feature_col, importance_col))

    if not candidates:
        return []

    _, path, feature_col, importance_col = sorted(candidates, reverse=True)[0]
    df = pd.read_csv(path).copy()

    df[importance_col] = pd.to_numeric(df[importance_col], errors="coerce")
    df = df.dropna(subset=[importance_col])
    if df.empty:
        return []

    df = df.sort_values(importance_col, ascending=False).head(18).copy()
    df["label"] = df[feature_col].map(pretty_label)
    df = df.sort_values(importance_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.barh(df["label"], df[importance_col], color=ACCENT, edgecolor="none")
    ax.set_title("Top gradient boosting feature importances", pad=14)
    ax.set_xlabel("Average feature importance")
    ax.set_ylabel("")
    style_axis(ax)

    return [save_fig(fig, FIGURES / "top_gradient_boosting_feature_importance.png")]


def build_benchmark_figure():
    path = existing_csv("benchmark_results.csv")
    if path is None:
        return []

    df = pd.read_csv(path).copy()
    name_col = find_col(df, ["strategy_name", "scenario_name", "baseline_name", "name"])
    sharpe_col = find_col(df, ["diagnostic_sharpe", "sharpe"])

    if name_col is None or sharpe_col is None:
        return []

    df[sharpe_col] = pd.to_numeric(df[sharpe_col], errors="coerce")
    df = df.dropna(subset=[sharpe_col])
    if df.empty:
        return []

    df["label"] = df[name_col].map(pretty_label)
    df = df.sort_values(sharpe_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    colors = [NEG if v < 0 else ACCENT for v in df[sharpe_col]]
    ax.barh(df["label"], df[sharpe_col], color=colors, edgecolor="none")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title("ML and baseline diagnostic Sharpe comparison", pad=14)
    ax.set_xlabel("Diagnostic Sharpe")
    ax.set_ylabel("")
    style_axis(ax)

    return [save_fig(fig, FIGURES / "benchmark_diagnostic_sharpe.png")]


def build_latest_rank_scatter():
    path = existing_csv("latest_rank_diagnostic.csv")
    if path is None:
        return []

    df = pd.read_csv(path).copy()
    pred_col = find_col(df, ["predicted_rank", "model_rank", "pred_rank"], contains=["predicted", "rank"])
    real_col = find_col(df, ["realized_rank", "actual_rank"], contains=["realized", "rank"])
    ticker_col = find_col(df, ["ticker", "symbol"])

    if pred_col is None or real_col is None:
        return []

    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df[real_col] = pd.to_numeric(df[real_col], errors="coerce")
    df = df.dropna(subset=[pred_col, real_col])
    if df.empty:
        return []

    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.scatter(df[pred_col], df[real_col], s=58, color=ACCENT, alpha=0.88, edgecolor=FIG_BG, linewidth=0.8)

    if ticker_col:
        for _, row in df.iterrows():
            ax.annotate(str(row[ticker_col]), (row[pred_col], row[real_col]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)

    lo = min(df[pred_col].min(), df[real_col].min())
    hi = max(df[pred_col].max(), df[real_col].max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", alpha=0.7)

    ax.set_title("Latest predicted rank vs realized rank", pad=14)
    ax.set_xlabel("Predicted rank")
    ax.set_ylabel("Realized rank")
    style_axis(ax)

    return [save_fig(fig, FIGURES / "latest_rank_diagnostic.png")]


def build_issuer_group_exposure():
    path = existing_csv("issuer_group_exposure_latest.csv")
    if path is None:
        return []

    df = pd.read_csv(path).copy()
    group_col = find_col(df, ["issuer_group", "group", "issuer"])
    weight_col = find_col(df, ["weight", "group_weight", "exposure", "portfolio_weight"], contains=["weight"])

    if group_col is None or weight_col is None:
        return []

    df[weight_col] = percent_if_small(df[weight_col])
    df = df.dropna(subset=[weight_col]).sort_values(weight_col, ascending=False).head(10)
    if df.empty:
        return []

    df = df.sort_values(weight_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(df[group_col].map(pretty_label), df[weight_col], color=ACCENT_2, edgecolor="none")
    ax.set_title("Latest issuer-group exposure", pad=14)
    ax.set_xlabel("Portfolio weight (%)")
    ax.set_ylabel("")
    style_axis(ax)

    return [save_fig(fig, FIGURES / "issuer_group_exposure_latest.png")]


def build_optimizer_bound_diagnostic():
    path = existing_csv("optimizer_bound_diagnostic.csv")
    if path is None:
        return []

    df = pd.read_csv(path).copy()
    date_col = find_col(df, ["date", "signal_date", "rebalance_date"], contains=["date"])
    value_col = find_col(df, ["cap_hit_count", "max_weight_count", "bound_hit_count"], contains=["hit"])

    if date_col is None or value_col is None:
        return []

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if df.empty:
        return []

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(df[date_col], df[value_col], color=ACCENT, linewidth=1.8)
    ax.fill_between(df[date_col], df[value_col], color=ACCENT, alpha=0.18)
    ax.set_title("Optimizer cap-hit diagnostic", pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of cap hits")
    style_axis(ax)

    return [save_fig(fig, FIGURES / "optimizer_bound_diagnostic.png")]


DARK_PATCH_START = "<!-- VN30_DARK_INTERACTIVE_PATCH_START -->"
DARK_PATCH_END = "<!-- VN30_DARK_INTERACTIVE_PATCH_END -->"

DARK_INTERACTIVE_PATCH = f"""
{DARK_PATCH_START}
<style>
html, body {{
  background: #0b1020 !important;
  color: #e5e7eb !important;
  margin: 0 !important;
}}
.plotly-graph-div, .js-plotly-plot, .svg-container {{
  background: #0b1020 !important;
}}
.modebar {{
  background: transparent !important;
}}
.modebar-btn svg path {{
  fill: #cbd5e1 !important;
}}
</style>
<script>
(function () {{
  const bg = "#0b1020";
  const panel = "#111827";
  const text = "#e5e7eb";
  const muted = "#cbd5e1";
  const grid = "rgba(148, 163, 184, 0.24)";
  const border = "#334155";

  function patchPlot(gd) {{
    if (!window.Plotly || !gd) return;

    const layout = gd.layout || {{}};
    const patch = {{
      paper_bgcolor: bg,
      plot_bgcolor: bg,
      font: Object.assign({{}}, layout.font || {{}}, {{color: text}}),
      title: Object.assign({{}}, layout.title || {{}}, {{
        font: Object.assign({{}}, (layout.title || {{}}).font || {{}}, {{color: text}})
      }}),
      legend: Object.assign({{}}, layout.legend || {{}}, {{
        bgcolor: "rgba(0,0,0,0)",
        font: Object.assign({{}}, (layout.legend || {{}}).font || {{}}, {{color: text}})
      }}),
      hoverlabel: Object.assign({{}}, layout.hoverlabel || {{}}, {{
        bgcolor: panel,
        bordercolor: border,
        font: {{color: "#f8fafc"}}
      }})
    }};

    ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"].forEach(function(axis) {{
      patch[axis] = Object.assign({{}}, layout[axis] || {{}}, {{
        color: muted,
        gridcolor: grid,
        zerolinecolor: border,
        linecolor: border,
        tickfont: Object.assign({{}}, ((layout[axis] || {{}}).tickfont || {{}}), {{color: muted}}),
        title: Object.assign({{}}, ((layout[axis] || {{}}).title || {{}}), {{
          font: Object.assign({{}}, (((layout[axis] || {{}}).title || {{}}).font || {{}}), {{color: text}})
        }})
      }});

      if (axis.startsWith("xaxis")) {{
        patch[axis].rangeslider = Object.assign({{}}, ((layout[axis] || {{}}).rangeslider || {{}}), {{
          bgcolor: "#0f172a",
          bordercolor: border,
          borderwidth: 1
        }});
        patch[axis].rangeselector = Object.assign({{}}, ((layout[axis] || {{}}).rangeselector || {{}}), {{
          bgcolor: panel,
          activecolor: border,
          font: Object.assign({{}}, (((layout[axis] || {{}}).rangeselector || {{}}).font || {{}}), {{color: text}})
        }});
      }}
    }});

    if (Array.isArray(layout.updatemenus)) {{
      patch.updatemenus = layout.updatemenus.map(function(menu) {{
        return Object.assign({{}}, menu, {{
          bgcolor: panel,
          activecolor: border,
          bordercolor: border,
          font: Object.assign({{}}, menu.font || {{}}, {{color: text}})
        }});
      }});
    }}

    window.Plotly.relayout(gd, patch).then(function() {{
      window.Plotly.Plots.resize(gd);
    }});
  }}

  function patchAll() {{
    if (!window.Plotly) return false;
    const plots = document.querySelectorAll(".js-plotly-plot");
    if (!plots.length) return false;
    plots.forEach(patchPlot);
    document.body.style.background = bg;
    return true;
  }}

  let tries = 0;
  const timer = setInterval(function() {{
    tries += 1;
    if (patchAll() || tries > 120) clearInterval(timer);
  }}, 100);

  window.addEventListener("resize", function() {{
    setTimeout(patchAll, 100);
  }});
}})();
</script>
{DARK_PATCH_END}
"""


def remove_existing_patch(text):
    while DARK_PATCH_START in text and DARK_PATCH_END in text:
        before, rest = text.split(DARK_PATCH_START, 1)
        _, after = rest.split(DARK_PATCH_END, 1)
        text = before + after
    return text


def patch_interactive_charts():
    if not INTERACTIVE.exists():
        return []

    changed = []

    for path in sorted(INTERACTIVE.glob("*.html")):
        text = path.read_text(encoding="utf-8", errors="replace")
        original = text

        for raw, clean in DISPLAY_LABELS.items():
            text = text.replace(raw, clean)

        text = remove_existing_patch(text)

        if "</body>" in text:
            text = text.replace("</body>", DARK_INTERACTIVE_PATCH + "\n</body>", 1)
        else:
            text = text.rstrip() + "\n" + DARK_INTERACTIVE_PATCH + "\n"

        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
            changed.append(path.relative_to(ROOT))

    return changed


def main():
    if not REPORTS.exists():
        raise SystemExit(f"Missing reports directory: {REPORTS}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    text_changed = apply_public_labels_to_text_files()

    written = []
    written.extend(build_horizon_figures())
    written.extend(build_ablation_figure())
    written.extend(build_feature_importance_figure())
    written.extend(build_benchmark_figure())
    written.extend(build_latest_rank_scatter())
    written.extend(build_issuer_group_exposure())
    written.extend(build_optimizer_bound_diagnostic())

    interactive_changed = patch_interactive_charts()

    print("Dark public graphs rebuilt.")
    print(f"Public text files changed: {len(text_changed)}")
    for path in text_changed:
        print(f"  text: {path}")

    print(f"Static figures rebuilt: {len(written)}")
    for path in written:
        print(f"  png:  {path}")

    print(f"Interactive charts patched: {len(interactive_changed)}")
    for path in interactive_changed:
        print(f"  html: {path}")


if __name__ == "__main__":
    main()
