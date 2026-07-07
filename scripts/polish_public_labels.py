from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
INTERACTIVE = REPORTS / "interactive"

DISPLAY_LABELS = {
    # Ablation labels
    "all_features": "All features",
    "without_volume_liquidity": "No volume/liquidity",
    "without_risk": "No risk features",
    "without_herding": "No herding features",
    "without_price_limit": "No price-limit features",

    # ML scenario labels
    "ml_normal_normal": "ML: standard",
    "ml_normal_price_limit_aware": "ML: price-limit aware",
    "ml_herding_aware_normal": "ML: herding-aware",
    "ml_herding_aware_price_limit_aware": "ML: herding + price-limit aware",

    # Baseline labels
    "top5_momentum": "Top-5 momentum",
    "top5_reversal": "Top-5 reversal",
    "low_volatility_top10": "Low-volatility top 10",
    "equal_weight_all": "Equal-weight universe",

    # Dashboard/card wording
    "Best Feature Set": "Best Ablation Variant",
    "BEST FEATURE SET": "BEST ABLATION VARIANT",
    "Best ML vs Naive Baseline": "ML vs Best Naive Baseline",
    "BEST ML VS NAIVE BASELINE": "ML VS BEST NAIVE BASELINE",

    # Active drawdown wording
    "Interactive Weekly Active Drawdown": "Interactive Active-Return Drawdown",
    "Weekly active drawdown": "Active-return drawdown",
    "weekly active drawdown": "active-return drawdown",
}


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

  .plotly-graph-div,
  .js-plotly-plot,
  .main-svg,
  .svg-container {{
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
  const axisGrid = "rgba(148, 163, 184, 0.22)";
  const axisLine = "rgba(203, 213, 225, 0.55)";
  const textColor = "#e5e7eb";
  const mutedText = "#cbd5e1";
  const bg = "#0b1020";
  const panel = "#111827";
  const border = "#334155";

  function patchOnePlot(gd) {{
    if (!window.Plotly || !gd) return;

    const patch = {{
      "paper_bgcolor": bg,
      "plot_bgcolor": bg,
      "font.color": textColor,
      "title.font.color": textColor,
      "legend.font.color": textColor,
      "legend.bgcolor": "rgba(0,0,0,0)",
      "hoverlabel.bgcolor": panel,
      "hoverlabel.bordercolor": border,
      "hoverlabel.font.color": "#f8fafc"
    }};

    const layout = gd.layout || {{}};
    const axisNames = Object.keys(layout).filter(function (key) {{
      return /^xaxis\\d*$/.test(key) || /^yaxis\\d*$/.test(key);
    }});

    if (!axisNames.includes("xaxis")) axisNames.push("xaxis");
    if (!axisNames.includes("yaxis")) axisNames.push("yaxis");

    axisNames.forEach(function (axis) {{
      patch[axis + ".color"] = mutedText;
      patch[axis + ".gridcolor"] = axisGrid;
      patch[axis + ".zerolinecolor"] = axisLine;
      patch[axis + ".linecolor"] = axisLine;
      patch[axis + ".tickfont.color"] = mutedText;
      patch[axis + ".title.font.color"] = textColor;

      if (axis.startsWith("xaxis")) {{
        patch[axis + ".rangeslider.bgcolor"] = "#0f172a";
        patch[axis + ".rangeslider.bordercolor"] = border;
        patch[axis + ".rangeslider.borderwidth"] = 1;
        patch[axis + ".rangeselector.bgcolor"] = panel;
        patch[axis + ".rangeselector.activecolor"] = border;
        patch[axis + ".rangeselector.font.color"] = textColor;
      }}
    }});

    if (Array.isArray(layout.updatemenus)) {{
      patch["updatemenus"] = layout.updatemenus.map(function (menu) {{
        return Object.assign({{}}, menu, {{
          bgcolor: panel,
          activecolor: border,
          bordercolor: border,
          font: Object.assign({{}}, menu.font || {{}}, {{ color: textColor }})
        }});
      }});
    }}

    window.Plotly.relayout(gd, patch).then(function () {{
      window.Plotly.Plots.resize(gd);
    }});
  }}

  function patchAllPlots() {{
    if (!window.Plotly) return false;

    const plots = document.querySelectorAll(".js-plotly-plot");
    if (!plots.length) return false;

    plots.forEach(patchOnePlot);
    document.body.style.background = bg;
    return true;
  }}

  let attempts = 0;
  const timer = setInterval(function () {{
    attempts += 1;
    const done = patchAllPlots();

    if (done || attempts > 80) {{
      clearInterval(timer);
    }}
  }}, 100);

  window.addEventListener("resize", function () {{
    setTimeout(patchAllPlots, 100);
  }});
}})();
</script>
{DARK_PATCH_END}
"""


def replace_exact_public_labels(text):
    for raw, clean in DISPLAY_LABELS.items():
        text = text.replace(raw, clean)
    return text


def remove_existing_dark_patch(text):
    while DARK_PATCH_START in text and DARK_PATCH_END in text:
        before, rest = text.split(DARK_PATCH_START, 1)
        _, after = rest.split(DARK_PATCH_END, 1)
        text = before + after
    return text


def patch_interactive_html(text):
    text = remove_existing_dark_patch(text)

    if "</body>" in text:
        text = text.replace("</body>", DARK_INTERACTIVE_PATCH + "\n</body>", 1)
    else:
        text = text.rstrip() + "\n" + DARK_INTERACTIVE_PATCH + "\n"

    return text


def process_text_file(path, interactive=False):
    text = path.read_text(encoding="utf-8", errors="replace")
    original = text

    text = replace_exact_public_labels(text)

    if interactive:
        text = patch_interactive_html(text)

    if text != original:
        path.write_text(text, encoding="utf-8", newline="\n")
        return True

    return False


def main():
    changed = []

    public_text_paths = []
    public_text_paths.extend(REPORTS.rglob("*.html"))
    public_text_paths.extend(REPORTS.rglob("*.md"))

    readme = ROOT / "README.md"
    if readme.exists():
        public_text_paths.append(readme)

    interactive_paths = set(INTERACTIVE.rglob("*.html")) if INTERACTIVE.exists() else set()

    for path in sorted(set(public_text_paths)):
        changed_file = process_text_file(path, interactive=path in interactive_paths)
        if changed_file:
            changed.append(path.relative_to(ROOT))

    print("Public labels and interactive dark theme polished.")
    print(f"Files changed: {len(changed)}")
    for path in changed:
        print(f"  changed: {path}")


if __name__ == "__main__":
    main()
