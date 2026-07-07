from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"


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


def pretty_label(value):
    text = str(value)
    return DISPLAY_LABELS.get(text, text.replace("_", " ").title())


def replace_exact_public_labels():
    """
    Safe public-label cleanup.

    Important:
    - This only replaces exact known raw labels.
    - It does NOT replace every snake_case string.
    - Therefore it should not break img src, iframe src, href, or file paths.
    """
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


def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (9.5, 5.4),
        "figure.dpi": 160,
        "savefig.dpi": 180,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 10,
    })


def save_horizon_figures():
    path = TABLES / "horizon_results.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path).sort_values("forecast_horizon_days").copy()
    df["horizon_label"] = df["forecast_horizon_days"].astype(int).astype(str) + "d"

    written = []

    fig, ax = plt.subplots()
    ax.bar(df["horizon_label"], df["diagnostic_sharpe"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Diagnostic Sharpe by forecast horizon")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Diagnostic Sharpe")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIGURES / "horizon_diagnostic_sharpe.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(out.relative_to(ROOT))

    fig, ax = plt.subplots()
    ax.bar(df["horizon_label"], df["average_rank_ic"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Average Rank IC by forecast horizon")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Average Rank IC")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIGURES / "horizon_rank_ic.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(out.relative_to(ROOT))

    return written


def save_ablation_figure():
    path = TABLES / "ablation_results.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path).copy()
    df["label"] = df["ablation_name"].map(pretty_label)
    df = df.sort_values("diagnostic_sharpe", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(df["label"], df["diagnostic_sharpe"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Diagnostic Sharpe by ablation variant")
    ax.set_xlabel("Diagnostic Sharpe")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = FIGURES / "ablation_diagnostic_sharpe.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return [out.relative_to(ROOT)]


def save_feature_importance_figures():
    written = []
    candidates = sorted(TABLES.glob("*feature_importance*.csv"))

    for path in candidates:
        df = pd.read_csv(path).copy()

        feature_col = "feature" if "feature" in df.columns else None
        if feature_col is None and "feature_name" in df.columns:
            feature_col = "feature_name"

        importance_col = "importance" if "importance" in df.columns else None
        if importance_col is None and "feature_importance" in df.columns:
            importance_col = "feature_importance"

        if feature_col is None or importance_col is None:
            continue

        df = df.sort_values(importance_col, ascending=False).head(15).copy()
        df["label"] = df[feature_col].map(lambda x: str(x).replace("_", " ").title())
        df = df.sort_values(importance_col, ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df["label"], df[importance_col])
        ax.set_title("Top feature importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        out_name = path.stem.replace("_feature_importance", "") + "_feature_importance.png"
        out = FIGURES / out_name
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out.relative_to(ROOT))

    return written


def main():
    if not REPORTS.exists():
        raise SystemExit(f"Missing reports directory: {REPORTS}")

    FIGURES.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    changed = replace_exact_public_labels()
    written = []
    written.extend(save_horizon_figures())
    written.extend(save_ablation_figure())
    written.extend(save_feature_importance_figures())

    print("Safe public label polish complete.")
    print(f"Text files changed: {len(changed)}")
    for path in changed:
        print(f"  changed: {path}")

    print(f"Figures regenerated: {len(written)}")
    for path in written:
        print(f"  figure:  {path}")


if __name__ == "__main__":
    main()
