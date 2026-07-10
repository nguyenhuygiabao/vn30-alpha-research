from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

OUTPUT_PATH = REPORTS_DIR / "dashboard.html"

HORIZON_RESULTS_PATH = TABLES_DIR / "horizon_results.csv"
ABLATION_RESULTS_PATH = TABLES_DIR / "ablation_results.csv"
BENCHMARK_RESULTS_PATH = TABLES_DIR / "benchmark_results.csv"
CONCENTRATION_SUMMARY_PATH = TABLES_DIR / "concentration_summary.csv"
ISSUER_GROUP_EXPOSURE_PATH = TABLES_DIR / "issuer_group_exposure_latest.csv"
LATEST_RANK_DIAGNOSTIC_PATH = TABLES_DIR / "latest_rank_diagnostic.csv"
HORIZON_DISCLOSURE_PATH = TABLES_DIR / "horizon_sample_disclosure.csv"
MODEL_COMPARISON_PATH = TABLES_DIR / "model_comparison_results.csv"
OPTIMIZER_BOUND_DIAGNOSTIC_PATH = TABLES_DIR / "optimizer_bound_diagnostic.csv"

TREE_PREDICTIONS_PATH = DATA_DIR / "tree_model_predictions.parquet"
OPTIMIZED_WEIGHTS_PATH = DATA_DIR / "optimized_weights.parquet"


INTERACTIVE_SECTIONS = [
    {
        "title": "Cumulative after-cost active return",
        "file": "interactive/interactive_cumulative_return.html",
        "guidance": "Use the date buttons or bottom slider to zoom. Click a legend item to hide/show a scenario. Double-click a legend item to isolate it.",
        "meaning": "Shows whether the strategy builds active value over time after estimated trading costs.",
        "read": "A rising line means the strategy is adding active return. Click legend items to hide, show, or isolate scenarios.",
        "watch": "Look for steady growth, long flat periods, and sharp drops.",
    },
    {
        "title": "Active drawdown",
        "file": "interactive/interactive_active_drawdown.html",
        "guidance": "Use the date buttons or bottom slider to zoom. Click a legend item to hide/show a scenario. Double-click a legend item to isolate it.",
        "meaning": "Shows how far the strategy falls below its previous active-return peak.",
        "read": "Values closer to zero are better. Deep negative drops mean the strategy gave back previous gains.",
        "watch": "The 2026 drawdown is inside the walk-forward backtest window and should be interpreted as part of the out-of-sample diagnostic period, not as live-trading evidence.",
    },
    {
        "title": "Portfolio turnover",
        "file": "interactive/interactive_portfolio_turnover.html",
        "guidance": "Use the date buttons or bottom slider to zoom. Click a legend item to hide/show a scenario. Double-click a legend item to isolate it.",
        "meaning": "Shows how much the portfolio changes between rebalancing dates.",
        "read": "Higher turnover means more trading. More trading can make live execution less realistic because costs rise.",
        "watch": "Look for long periods near the maximum turnover level.",
    },
    {
        "title": "Rolling 60-day diagnostic Sharpe with approximate band",
        "file": "interactive/interactive_rolling_diagnostic_sharpe.html",
        "guidance": "Use the date buttons or bottom slider to zoom. Click a legend item to hide/show a scenario. Double-click a legend item to isolate it.",
        "meaning": "Shows short-term risk-adjusted performance over rolling 60-trading-day windows with an approximate visual uncertainty band.",
        "read": "Higher values mean better return per unit of volatility during the recent window. The shaded band is approximate and should not be read as formal statistical proof.",
        "watch": "Look for unstable periods where the rolling Sharpe drops sharply or the band is wide.",
    },
    {
        "title": "Single-name cap-hit share",
        "file": "interactive/interactive_optimizer_cap_hits.html",
        "guidance": "Use the date buttons or bottom slider to zoom. Click a legend item to hide/show a scenario. Double-click a legend item to isolate it.",
        "meaning": "Shows how often holdings sit at the 20 percent single-name cap.",
        "read": "A value near 100 percent means nearly every holding is at the maximum allowed weight.",
        "watch": "Persistent high values mean portfolio construction is heavily shaped by the cap constraint.",
    },
]


STATIC_SECTIONS = [
    {
        "title": "Top gradient boosting feature importance",
        "file": "figures/top_gradient_boosting_feature_importance.png",
        "meaning": "Shows which inputs the gradient boosting model relied on most.",
        "read": "Larger bars mean higher model importance.",
    },
    {
        "title": "Horizon diagnostic Sharpe",
        "file": "figures/horizon_diagnostic_sharpe.png",
        "meaning": "Compares risk-adjusted behavior across prediction horizons.",
        "read": "Higher is better for this diagnostic metric.",
    },
    {
        "title": "Horizon Rank IC",
        "file": "figures/horizon_rank_ic.png",
        "meaning": "Compares how well the model ranks stocks across forecast horizons.",
        "read": "Higher means the ranking aligns better with realized future relative returns.",
    },
    {
        "title": "Ablation diagnostic Sharpe",
        "file": "figures/ablation_diagnostic_sharpe.png",
        "meaning": "Shows what happens when feature groups are removed.",
        "read": "If performance falls after removing a group, that group likely adds useful information.",
    },
]


GLOSSARY_ROWS = [
    ("Rank IC (unitless, -1 to +1)", "Correlation between the model's stock ranking and realized future return ranking."),
    ("Diagnostic Sharpe", "Return divided by volatility, annualized. Used here as a diagnostic comparison metric."),
    ("Active return", "Return relative to the reference portfolio or benchmark."),
    ("After-cost return", "Return after estimated commission, slippage, and liquidity penalties."),
    ("Drawdown", "Fall from a previous performance peak."),
    ("Turnover", "How much the portfolio changes between rebalancing dates."),
    ("Feature ablation", "Removing feature groups to test whether they help."),
    ("Forecast horizon", "How far ahead the model predicts, such as 1 day, 5 days, or 10 days."),
    ("Overlapping windows", "Forecast periods that share trading days. This increases raw sample counts versus independent periods."),
    ("HHI", "Sum of squared portfolio weights. Higher values mean more concentration."),
    ("Effective position count", "Inverse of HHI. It translates concentration into an approximate number of equally weighted positions."),
    ("Issuer-group exposure", "Total portfolio weight in tickers that belong to the same issuer group."),
    ("Cap-hit share", "Share of current holdings that sit at the optimizer's maximum single-name weight."),
    ("Cumulative active-return sum", "A sum of overlapping active forecast-period returns. It is not a compounded live portfolio return."),
    ("Model comparison", "A diagnostic comparison using equal-weight top-5 selections by model score. It is not the same as the optimized execution backtest."),
    ("Top-5 hit rate", "How often top-ranked stocks are among better realized performers."),
    ("Model score", "The model's predicted relative return signal."),
]


DISPLAY_COLUMN_NAMES = {
    "average_rank_ic": "Rank IC (unitless, -1 to +1)",
    "final_cumulative_after_cost_active_return": "Cumulative active-return sum",
    "final_cumulative_active_return": "Cumulative active-return sum",
    "final_cumulative_active_return_sum": "Cumulative active-return sum",
    "final_cumulative_top5_return_sum": "Cumulative top-5 return sum",
    "final_cumulative_active_return_note": "Cumulative active-return note",
    "max_active_drawdown": "Max active drawdown",
    "max_drawdown_from_top5_return_sum": "Max drawdown from top-5 return sum",
}

CUMULATIVE_ACTIVE_RETURN_NOTE = (
    "Sum of overlapping forecast-period active returns, not a compounded portfolio return."
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def format_number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def format_percent(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"

    try:
        return f"{float(value) * 100:,.{digits}f}%"
    except Exception:
        return str(value)


def format_table(
    data: pd.DataFrame,
    max_rows: int = 20,
    percent_columns: set[str] | None = None,
    decimal_columns: set[str] | None = None,
) -> str:
    if data.empty:
        return '<p class="muted">No data available.</p>'

    percent_columns = percent_columns or set()
    decimal_columns = decimal_columns or set()

    display = data.head(max_rows).copy()

    for column in display.columns:
        if column in percent_columns:
            display[column] = display[column].map(format_percent)
        elif column in decimal_columns:
            display[column] = display[column].map(lambda value: format_number(value, 4))

    display = display.rename(columns=DISPLAY_COLUMN_NAMES)

    return display.to_html(index=False, classes="data-table", border=0, escape=True)


def cumulative_active_return_note_html() -> str:
    return f'<p class="muted table-note">{escape(CUMULATIVE_ACTIVE_RETURN_NOTE)}</p>'


def latest_signal_date() -> str:
    if not TREE_PREDICTIONS_PATH.exists():
        return "N/A"

    predictions = pd.read_parquet(TREE_PREDICTIONS_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"])

    return predictions["date"].max().strftime("%Y-%m-%d")


def latest_vhm_vic_tie_note() -> str:
    if not TREE_PREDICTIONS_PATH.exists():
        return ""

    predictions = pd.read_parquet(TREE_PREDICTIONS_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"])

    latest_date = predictions["date"].max()
    latest = predictions[
        predictions["date"].eq(latest_date)
        & predictions["model_name"].eq("gradient_boosting")
        & predictions["ticker"].isin(["VHM", "VIC"])
    ].copy()

    if latest["ticker"].nunique() < 2:
        return ""

    scores = latest.set_index("ticker")["predicted_return"]
    vhm_score = scores["VHM"]
    vic_score = scores["VIC"]

    if vhm_score == vic_score:
        return (
            "VHM and VIC have identical model scores in the latest snapshot; "
            "rank ordering is tie-broken by ticker/order for display."
        )

    if format_percent(vhm_score) == format_percent(vic_score):
        return (
            "VHM and VIC round to the same displayed score; underlying full-precision "
            "scores differ."
        )

    return ""


def metric_card(title: str, value: str, subtext: str) -> str:
    return f"""
      <div class="metric-card">
        <div class="metric-title">{escape(title)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-subtext">{escape(subtext)}</div>
      </div>
    """


def summary_cards_html() -> str:
    horizon = read_csv(HORIZON_RESULTS_PATH)
    ablation = read_csv(ABLATION_RESULTS_PATH)

    cards = []

    if not horizon.empty:
        selected_rows = horizon.loc[
            horizon["forecast_horizon_days"].eq(10)
        ]
        selected = (
            selected_rows.iloc[0]
            if not selected_rows.empty
            else horizon.loc[horizon["average_rank_ic"].idxmax()]
        )

        cards.append(
            metric_card(
                "Selected research horizon",
                f'{int(selected["forecast_horizon_days"])} trading days',
                (
                    f'Rank IC {format_number(selected["average_rank_ic"])} · '
                    f'diagnostic Sharpe {format_number(selected["diagnostic_sharpe"])}'
                ),
            )
        )

    if not ablation.empty:
        best_ablation = ablation.loc[ablation["diagnostic_sharpe"].idxmax()]
        cards.append(
            metric_card(
                "Best feature set",
                str(best_ablation["ablation_name"]),
                f'Sharpe {format_number(best_ablation["diagnostic_sharpe"])}',
            )
        )

    cards.append(
        metric_card(
            "Latest backtest signal",
            latest_signal_date(),
            "historical walk-forward snapshot, not a live signal",
        )
    )
    cards.append(
        metric_card(
            "Evidence status",
            "Research only",
            "paper validation in progress · no real-money execution",
        )
    )

    return "\n".join(cards)


def latest_stock_ranking() -> pd.DataFrame:
    if not TREE_PREDICTIONS_PATH.exists():
        return pd.DataFrame()

    predictions = pd.read_parquet(TREE_PREDICTIONS_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"])

    latest_date = predictions["date"].max()
    latest = predictions[
        predictions["date"].eq(latest_date)
        & predictions["model_name"].eq("gradient_boosting")
    ].copy()

    if latest.empty:
        latest = predictions[predictions["date"].eq(latest_date)].copy()

    latest = latest.sort_values("predicted_return", ascending=False)
    latest["rank"] = range(1, len(latest) + 1)

    latest = latest[
        ["rank", "ticker", "date", "model_name", "predicted_return", "actual_return"]
    ].rename(
        columns={
            "date": "signal_date",
            "model_name": "model",
            "predicted_return": "model_score",
            "actual_return": "realized_forward_return",
        }
    )

    latest["signal_date"] = pd.to_datetime(latest["signal_date"]).dt.strftime("%Y-%m-%d")

    return latest


def latest_portfolio_weights() -> pd.DataFrame:
    if not OPTIMIZED_WEIGHTS_PATH.exists():
        return pd.DataFrame()

    weights = pd.read_parquet(OPTIMIZED_WEIGHTS_PATH)
    weights["date"] = pd.to_datetime(weights["date"])

    latest_date = weights["date"].max()
    latest = weights[weights["date"].eq(latest_date)].copy()
    preferred = latest[latest["optimization_mode"].eq("normal")].copy()

    if preferred.empty:
        preferred = latest.copy()

    preferred = preferred[
        ["date", "ticker", "issuer_group", "optimization_mode", "weight", "predicted_return", "actual_return"]
    ].sort_values(
        ["optimization_mode", "weight", "predicted_return"],
        ascending=[True, False, False],
    )

    preferred["date"] = preferred["date"].dt.strftime("%Y-%m-%d")

    return preferred.rename(
        columns={
            "date": "signal_date",
            "predicted_return": "model_score",
            "actual_return": "realized_forward_return",
        }
    )


def horizon_table_html() -> str:
    horizon = read_csv(HORIZON_RESULTS_PATH)

    columns = [
        "forecast_horizon_days",
        "evaluated_dates",
        "average_rank_ic",
        "average_top_5_hit_rate",
        "average_after_cost_return",
        "diagnostic_sharpe",
        "max_active_drawdown",
        "average_turnover",
        "final_cumulative_after_cost_active_return",
        "final_cumulative_active_return",
        "final_cumulative_active_return_sum",
    ]

    available = [column for column in columns if column in horizon.columns]

    return format_table(
        horizon[available],
        percent_columns={
            "average_top_5_hit_rate",
            "average_after_cost_return",
            "max_active_drawdown",
            "average_turnover",
        },
        decimal_columns={
            "average_rank_ic",
            "diagnostic_sharpe",
            "final_cumulative_after_cost_active_return",
            "final_cumulative_active_return",
            "final_cumulative_active_return_sum",
        },
    )


def ablation_table_html() -> str:
    ablation = read_csv(ABLATION_RESULTS_PATH)

    columns = [
        "ablation_name",
        "evaluated_dates",
        "feature_count",
        "average_rank_ic",
        "average_top_5_hit_rate",
        "average_after_cost_return",
        "diagnostic_sharpe",
        "max_active_drawdown",
        "average_turnover",
        "final_cumulative_after_cost_active_return",
        "final_cumulative_active_return",
        "final_cumulative_active_return_sum",
    ]

    available = [column for column in columns if column in ablation.columns]

    return format_table(
        ablation[available],
        percent_columns={
            "average_top_5_hit_rate",
            "average_after_cost_return",
            "max_active_drawdown",
            "average_turnover",
        },
        decimal_columns={
            "average_rank_ic",
            "diagnostic_sharpe",
            "final_cumulative_after_cost_active_return",
            "final_cumulative_active_return",
            "final_cumulative_active_return_sum",
        },
    )


def benchmark_table_html() -> str:
    benchmark = read_csv(BENCHMARK_RESULTS_PATH)

    columns = [
        "comparison_type",
        "display_name",
        "forecast_horizon_days",
        "evaluated_dates",
        "average_period_active_return",
        "return_volatility",
        "diagnostic_sharpe",
        "max_active_drawdown",
        "final_cumulative_after_cost_active_return",
        "final_cumulative_active_return",
        "final_cumulative_active_return_sum",
        "average_selected_count",
        "average_return_period_label",
        "final_cumulative_active_return_note",
        "cost_note",
    ]

    available = [column for column in columns if column in benchmark.columns]

    return format_table(
        benchmark[available],
        max_rows=12,
        percent_columns={
            "average_period_active_return",
            "return_volatility",
            "max_active_drawdown",
        },
        decimal_columns={
            "diagnostic_sharpe",
            "final_cumulative_after_cost_active_return",
            "final_cumulative_active_return",
            "final_cumulative_active_return_sum",
            "average_selected_count",
        },
    )


def concentration_summary_table_html() -> str:
    concentration = read_csv(CONCENTRATION_SUMMARY_PATH)

    columns = [
        "signal_date",
        "optimization_mode",
        "holding_count",
        "total_weight",
        "max_single_name_weight",
        "positions_at_or_above_20pct",
        "single_name_cap_hit_share",
        "hhi",
        "effective_position_count",
        "top_issuer_group",
        "top_issuer_group_weight",
        "top_issuer_group_tickers",
        "issuer_groups_at_or_above_40pct",
        "concentration_flag",
    ]

    available = [column for column in columns if column in concentration.columns]

    return format_table(
        concentration[available],
        max_rows=10,
        percent_columns={
            "total_weight",
            "max_single_name_weight",
            "top_issuer_group_weight",
            "single_name_cap_hit_share",
        },
        decimal_columns={
            "hhi",
            "effective_position_count",
        },
    )


def issuer_group_exposure_table_html() -> str:
    exposure = read_csv(ISSUER_GROUP_EXPOSURE_PATH)

    columns = [
        "signal_date",
        "optimization_mode",
        "issuer_group",
        "issuer_group_weight",
        "position_count",
        "tickers",
        "max_single_name_weight_in_group",
        "weighted_realized_forward_return",
        "exposure_flag",
    ]

    available = [column for column in columns if column in exposure.columns]

    return format_table(
        exposure[available],
        max_rows=12,
        percent_columns={
            "issuer_group_weight",
            "max_single_name_weight_in_group",
            "weighted_realized_forward_return",
        },
    )


def latest_rank_diagnostic_table_html() -> str:
    diagnostic = read_csv(LATEST_RANK_DIAGNOSTIC_PATH)

    columns = [
        "signal_date",
        "ticker",
        "predicted_rank",
        "realized_rank",
        "rank_gap_realized_minus_predicted",
        "absolute_rank_gap",
        "model_score",
        "realized_forward_return",
        "diagnostic_flag",
        "in_normal_portfolio",
        "in_herding_aware_portfolio",
    ]

    available = [column for column in columns if column in diagnostic.columns]

    return format_table(
        diagnostic[available],
        max_rows=15,
        percent_columns={
            "model_score",
            "realized_forward_return",
        },
    )


def horizon_disclosure_table_html() -> str:
    disclosure = read_csv(HORIZON_DISCLOSURE_PATH)

    columns = [
        "forecast_horizon_days",
        "period_label",
        "evaluated_dates",
        "approx_non_overlapping_periods",
        "overlap_share_estimate",
        "overlap_disclosure",
        "metric_period_note",
    ]

    available = [column for column in columns if column in disclosure.columns]

    return format_table(
        disclosure[available],
        max_rows=10,
        percent_columns={"overlap_share_estimate"},
    )


def optimizer_bound_diagnostic_table_html() -> str:
    diagnostic = read_csv(OPTIMIZER_BOUND_DIAGNOSTIC_PATH)

    columns = [
        "signal_date",
        "optimization_mode",
        "latest_holding_count",
        "latest_positions_at_20pct_cap",
        "latest_single_name_cap_hit_share",
        "latest_all_positions_at_cap",
        "mean_single_name_cap_hit_share",
        "share_of_dates_all_positions_at_cap",
        "latest_hhi",
        "latest_effective_position_count",
        "diagnostic_note",
    ]

    available = [column for column in columns if column in diagnostic.columns]

    return format_table(
        diagnostic[available],
        max_rows=10,
        percent_columns={
            "latest_single_name_cap_hit_share",
            "mean_single_name_cap_hit_share",
            "share_of_dates_all_positions_at_cap",
        },
        decimal_columns={
            "latest_hhi",
            "latest_effective_position_count",
        },
    )


def model_comparison_table_html() -> str:
    comparison = read_csv(MODEL_COMPARISON_PATH)

    columns = [
        "model_family",
        "model_name",
        "evaluated_dates",
        "average_rank_ic",
        "average_top5_hit_rate",
        "average_top5_realized_return_per_5d_period",
        "return_volatility",
        "diagnostic_sharpe",
        "max_drawdown_from_top5_return_sum",
        "final_cumulative_top5_return_sum",
        "average_selected_count",
        "comparison_note",
    ]

    available = [column for column in columns if column in comparison.columns]

    return format_table(
        comparison[available],
        max_rows=12,
        percent_columns={
            "average_top5_hit_rate",
            "average_top5_realized_return_per_5d_period",
            "return_volatility",
            "max_drawdown_from_top5_return_sum",
        },
        decimal_columns={
            "average_rank_ic",
            "diagnostic_sharpe",
            "final_cumulative_top5_return_sum",
            "average_selected_count",
        },
    )


def interactive_section_html(item: dict[str, str]) -> str:
    path = REPORTS_DIR / item["file"]

    if not path.exists():
        return ""

    versioned = f'{item["file"]}?v={int(path.stat().st_mtime)}'
    guidance = item.get(
        "guidance",
        "Use the date buttons or bottom slider to zoom. Click a legend item to hide/show a scenario.",
    )

    return f"""
    <section class="interactive-card">
      <h2>{escape(item["title"])}</h2>
      <p class="helper">
        {escape(guidance)}
      </p>
      <iframe src="{escape(versioned)}" title="{escape(item["title"])}" loading="lazy"></iframe>
      <p><a href="{escape(item["file"])}" target="_blank" rel="noopener">Open chart in a full page</a></p>
      <div class="annotation">
        <p><strong>What this means:</strong> {escape(item["meaning"])}</p>
        <p><strong>How to read it:</strong> {escape(item["read"])}</p>
        <p><strong>What to watch:</strong> {escape(item["watch"])}</p>
      </div>
    </section>
    """


def static_section_html(item: dict[str, str]) -> str:
    path = REPORTS_DIR / item["file"]

    if not path.exists() or path.stat().st_size == 0:
        return ""

    return f"""
      <article class="static-card">
        <h3>{escape(item["title"])}</h3>
        <img src="{escape(item["file"])}" alt="{escape(item["title"])}">
        <div class="small-note">
          <p><strong>Meaning:</strong> {escape(item["meaning"])}</p>
          <p><strong>Read:</strong> {escape(item["read"])}</p>
        </div>
      </article>
    """


def static_grid_html() -> str:
    cards = "\n".join(static_section_html(item) for item in STATIC_SECTIONS)

    return f"""
    <section class="section-card">
      <h2>Additional static diagnostics</h2>
      <div class="static-grid">
        {cards}
      </div>
    </section>
    """



def research_validity_html() -> str:
    return """
    <section class="section-card" id="validity">
      <div class="section-kicker">Read before interpreting results</div>
      <h2>Research validity notes</h2>
      <p class="muted">
        This dashboard reports historical diagnostics from a research backtest. The results should not be read as
        live-trading evidence yet.
      </p>
      <div class="validity-grid">
        <article>
          <h3>Static universe</h3>
          <p>The current VN30 universe is fixed rather than point-in-time, so survivorship bias remains a known limitation.</p>
        </article>
        <article>
          <h3>Multiple comparisons</h3>
          <p>The project tests several horizons, ablations, optimizer variants, and execution settings, so best-case metrics may overstate stable skill.</p>
        </article>
        <article>
          <h3>Point estimates</h3>
          <p>Sharpe, Rank IC, drawdown, and return metrics are currently point estimates. Bootstrap intervals or deflated Sharpe diagnostics are future upgrades.</p>
        </article>
        <article>
          <h3>Live-market frictions</h3>
          <p>Corporate actions, foreign ownership room, liquidity, and paper-trading validation must be handled before any live execution claim.</p>
        </article>
      </div>
    </section>
    """


def glossary_html() -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td>{escape(term)}</td>
          <td>{escape(explanation)}</td>
        </tr>
        """
        for term, explanation in GLOSSARY_ROWS
    )

    return f"""
    <section class="section-card" id="glossary">
      <div class="section-kicker">Reference</div>
      <h2>Glossary: what the metrics mean</h2>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Term</th>
              <th>Simple meaning</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>
    """


def page_html() -> str:
    ranking = latest_stock_ranking()
    weights = latest_portfolio_weights()
    vhm_vic_note = latest_vhm_vic_tie_note()
    vhm_vic_note_html = (
        f'<p class="muted table-note">{escape(vhm_vic_note)}</p>'
        if vhm_vic_note
        else ""
    )

    interactive_sections = "\n".join(
        interactive_section_html(item)
        for item in INTERACTIVE_SECTIONS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VN30 Alpha Research Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Interactive VN30 machine-learning research dashboard covering signals, portfolio risk, and historical validation evidence.">
  <meta name="theme-color" content="#050814">
  <style>
    :root {{
      --bg: #050814;
      --bg-2: #070d1c;
      --panel: rgba(11, 18, 32, 0.96);
      --panel-soft: rgba(17, 27, 46, 0.96);
      --panel-glow: rgba(34, 211, 238, 0.12);
      --line: rgba(148, 163, 184, 0.22);
      --line-strong: rgba(125, 211, 252, 0.38);
      --text: #eaf2ff;
      --muted: #94a3b8;
      --accent: #22d3ee;
      --accent-2: #34d399;
      --danger: #fb7185;
      --white: #ffffff;
    }}

    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.13), transparent 32rem),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.10), transparent 26rem),
        linear-gradient(180deg, var(--bg), #030712 55%, #020617);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Arial, Helvetica, sans-serif;
      line-height: 1.5;
    }}

    a {{
      color: #7dd3fc;
    }}

    a:focus-visible,
    button:focus-visible {{
      outline: 3px solid rgba(34, 211, 238, 0.72);
      outline-offset: 3px;
    }}

    .skip-link {{
      position: fixed;
      left: 16px;
      top: -80px;
      z-index: 100;
      padding: 10px 14px;
      border-radius: 10px;
      color: #001018;
      background: #67e8f9;
      font-weight: 800;
      transition: top 0.2s ease;
    }}

    .skip-link:focus {{
      top: 12px;
    }}

    .topbar {{
      position: sticky;
      top: 0;
      z-index: 50;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      background: rgba(3, 7, 18, 0.84);
      backdrop-filter: blur(18px);
    }}

    .topbar-inner {{
      width: min(1500px, calc(100% - 64px));
      min-height: 68px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }}

    .brand {{
      display: inline-flex;
      align-items: center;
      gap: 11px;
      color: var(--text);
      text-decoration: none;
      font-size: 14px;
      font-weight: 900;
      letter-spacing: -0.01em;
      white-space: nowrap;
    }}

    .brand-mark {{
      display: grid;
      width: 34px;
      height: 34px;
      place-items: center;
      border: 1px solid rgba(103, 232, 249, 0.34);
      border-radius: 11px;
      color: #a5f3fc;
      background: linear-gradient(145deg, rgba(8, 145, 178, 0.26), rgba(16, 185, 129, 0.16));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }}

    .topnav {{
      display: flex;
      align-items: center;
      gap: 4px;
      overflow-x: auto;
      scrollbar-width: none;
    }}

    .topnav::-webkit-scrollbar {{
      display: none;
    }}

    .topnav a {{
      padding: 8px 11px;
      border-radius: 9px;
      color: #a8b6cc;
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }}

    .topnav a:hover {{
      color: var(--text);
      background: rgba(148, 163, 184, 0.10);
    }}

    .hero {{
      width: min(1500px, calc(100% - 64px));
      margin: 24px auto 20px;
      padding: 42px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(8, 13, 28, 0.98)),
        radial-gradient(circle at 20% 0%, rgba(34, 211, 238, 0.18), transparent 22rem);
      box-shadow:
        0 24px 70px rgba(0, 0, 0, 0.38),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
      position: relative;
      overflow: hidden;
      scroll-margin-top: 88px;
    }}

    .hero::after {{
      content: "";
      position: absolute;
      width: 420px;
      height: 420px;
      right: -170px;
      bottom: -250px;
      border-radius: 50%;
      background: rgba(34, 211, 238, 0.10);
      filter: blur(12px);
      pointer-events: none;
    }}

    .hero-grid {{
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.75fr);
      gap: 42px;
      align-items: center;
    }}

    .eyebrow,
    .section-kicker {{
      color: #67e8f9;
      text-transform: uppercase;
      letter-spacing: 0.13em;
      font-size: 11px;
      font-weight: 900;
    }}

    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      margin-bottom: 16px;
    }}

    .eyebrow::before {{
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #34d399;
      box-shadow: 0 0 16px rgba(52, 211, 153, 0.8);
    }}

    h1 {{
      margin: 0 0 10px;
      font-size: clamp(34px, 4vw, 54px);
      line-height: 1.03;
      letter-spacing: -0.055em;
      background: linear-gradient(90deg, #e0f2fe, #38bdf8 42%, #34d399);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }}

    h2 {{
      margin: 0 0 16px;
      font-size: 25px;
      letter-spacing: -0.02em;
    }}

    h3 {{
      margin: 0 0 12px;
      font-size: 17px;
      letter-spacing: -0.01em;
    }}

    p {{
      margin: 0;
    }}

    .subtitle {{
      max-width: 760px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.7;
    }}

    .hero-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 11px;
      margin-top: 24px;
    }}

    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 10px 16px;
      border: 1px solid rgba(125, 211, 252, 0.34);
      border-radius: 11px;
      color: #02131d;
      background: linear-gradient(135deg, #67e8f9, #34d399);
      text-decoration: none;
      font-size: 13px;
      font-weight: 900;
      box-shadow: 0 10px 28px rgba(34, 211, 238, 0.16);
    }}

    .button.secondary {{
      color: #dbeafe;
      background: rgba(15, 23, 42, 0.72);
      box-shadow: none;
    }}

    .hero-status {{
      padding: 20px;
      border: 1px solid rgba(125, 211, 252, 0.20);
      border-radius: 18px;
      background: rgba(2, 6, 23, 0.48);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }}

    .status-label {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      color: #dbeafe;
      font-size: 13px;
      font-weight: 850;
    }}

    .status-badge {{
      padding: 5px 9px;
      border: 1px solid rgba(251, 191, 36, 0.26);
      border-radius: 999px;
      color: #fde68a;
      background: rgba(161, 98, 7, 0.14);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .status-list {{
      display: grid;
      gap: 12px;
      margin: 16px 0 0;
    }}

    .status-list div {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 12px;
    }}

    .status-list dt {{
      color: var(--muted);
      font-size: 12px;
    }}

    .status-list dd {{
      margin: 0;
      color: #e2e8f0;
      font-size: 12px;
      font-weight: 750;
    }}

    main {{
      width: min(1500px, calc(100% - 64px));
      margin: 0 auto 60px;
    }}

    main section[id] {{
      scroll-margin-top: 88px;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 22px;
    }}

    .metric-card,
    .section-card,
    .interactive-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(8, 13, 28, 0.96));
      box-shadow:
        0 18px 46px rgba(0, 0, 0, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.035);
      margin-bottom: 22px;
    }}

    .metric-card {{
      padding: 22px;
      position: relative;
      overflow: hidden;
    }}

    .metric-card::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 3px;
      background: linear-gradient(180deg, var(--accent), var(--accent-2));
      opacity: 0.9;
    }}

    .metric-title {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 12px;
    }}

    .metric-value {{
      color: var(--text);
      font-size: 32px;
      font-weight: 900;
      margin-bottom: 7px;
      letter-spacing: -0.04em;
    }}

    .metric-subtext,
    .muted,
    .helper {{
      color: var(--muted);
    }}

    .section-card,
    .interactive-card {{
      padding: 24px;
    }}

    .section-kicker {{
      margin-bottom: 8px;
    }}

    .section-intro {{
      margin: 48px 0 18px;
      padding: 0 4px;
    }}

    .section-intro h2 {{
      margin-bottom: 7px;
    }}

    .section-intro p {{
      max-width: 780px;
      color: var(--muted);
    }}

    .section-card > .muted,
    .interactive-card > .helper {{
      margin-top: -6px;
      margin-bottom: 14px;
      font-size: 14px;
    }}

    .table-note {{
      margin: 10px 0 14px;
      font-size: 13px;
    }}

    .table-wrap {{
      overflow-x: auto;
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 14px;
      background: rgba(2, 6, 23, 0.34);
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}

    .data-table th {{
      background: rgba(15, 23, 42, 0.92);
      color: #dbeafe;
      text-align: left;
      padding: 12px 13px;
      white-space: nowrap;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.055em;
      border-bottom: 1px solid var(--line);
    }}

    .data-table td {{
      border-top: 1px solid rgba(148, 163, 184, 0.14);
      padding: 11px 13px;
      color: #d7e3f8;
      white-space: nowrap;
    }}

    .data-table tr:hover td {{
      background: rgba(34, 211, 238, 0.055);
    }}

    .interactive-card {{
      background:
        linear-gradient(180deg, rgba(9, 14, 27, 0.98), rgba(3, 7, 18, 0.98));
      border-color: rgba(125, 211, 252, 0.20);
    }}

    .interactive-card iframe {{
      display: block;
      width: 100%;
      height: 620px;
      border: 1px solid rgba(125, 211, 252, 0.18);
      border-radius: 16px;
      background: #050a14;
      margin: 14px 0 0;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.025);
    }}

    .annotation {{
      margin-top: 16px;
      padding: 16px 18px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 14px;
      background: rgba(15, 23, 42, 0.74);
      color: var(--muted);
    }}

    .annotation p {{
      margin: 0;
    }}

    .annotation p + p {{
      margin-top: 8px;
    }}

    .annotation strong,
    .small-note strong {{
      color: var(--text);
    }}

    .validity-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}

    .validity-grid article {{
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 14px;
      background: rgba(8, 13, 28, 0.88);
      padding: 15px;
    }}

    .validity-grid h3 {{
      margin: 0 0 8px;
      color: var(--text);
      font-size: 15px;
    }}

    .validity-grid p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}

    .static-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}

    .static-card {{
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 15px;
      background: rgba(8, 13, 28, 0.92);
      padding: 16px;
      min-width: 0;
    }}

    .static-card img {{
      display: block;
      width: 100%;
      height: 300px;
      object-fit: contain;
      background: #000000;
      border: 1px solid rgba(255, 255, 255, 0.10);
      border-radius: 12px;
      margin-bottom: 12px;
      filter: none;
    }}

    .small-note {{
      color: var(--muted);
      font-size: 13px;
    }}

    .small-note p {{
      margin: 0;
    }}

    .small-note p + p {{
      margin-top: 7px;
    }}

    .disclaimer {{
      margin-top: 24px;
      padding: 18px 20px;
      border: 1px solid rgba(251, 113, 133, 0.28);
      border-radius: 16px;
      background: rgba(127, 29, 29, 0.16);
      color: #fecdd3;
    }}

    .disclaimer strong {{
      color: #ffe4e6;
    }}

    @media (max-width: 1100px) {{
      .topbar-inner,
      .hero,
      main {{
        width: min(100% - 28px, 1500px);
      }}

      .hero {{
        padding: 28px 22px;
      }}

      .hero-grid {{
        grid-template-columns: 1fr;
        gap: 26px;
      }}

      .metric-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .static-grid {{
        grid-template-columns: 1fr;
      }}

      .validity-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 700px) {{
      .topbar-inner {{
        min-height: 62px;
      }}

      .brand span:last-child {{
        display: none;
      }}

      .topnav a {{
        padding-inline: 8px;
        font-size: 12px;
      }}

      .hero {{
        margin-top: 14px;
      }}

      .hero-status {{
        padding: 16px;
      }}

      .metric-grid {{
        grid-template-columns: 1fr;
      }}

      .interactive-card iframe {{
        height: 560px;
      }}

      .static-card img {{
        height: 250px;
      }}
    }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to dashboard content</a>

  <div class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="#overview" aria-label="VN30 Alpha Research home">
        <span class="brand-mark" aria-hidden="true">V30</span>
        <span>VN30 Alpha Research</span>
      </a>
      <nav class="topnav" aria-label="Dashboard sections">
        <a href="#overview">Overview</a>
        <a href="#rankings">Signals</a>
        <a href="#portfolio">Portfolio</a>
        <a href="#horizons">Evidence</a>
        <a href="#charts">Charts</a>
        <a href="#validity">Methodology</a>
      </nav>
    </div>
  </div>

  <header class="hero" id="overview">
    <div class="hero-grid">
      <div>
        <div class="eyebrow">Machine-learning research · VN30 universe</div>
        <h1>Signals, portfolio risk, and evidence in one view.</h1>
        <p class="subtitle">
          Explore how a Vietnam-aware ranking model behaves across forecast horizons,
          portfolio constraints, and execution assumptions. Results are historical
          diagnostics, not an investment recommendation.
        </p>
        <div class="hero-actions">
          <a class="button" href="#rankings">Explore latest ranking</a>
          <a class="button secondary" href="#validity">Read research limitations</a>
        </div>
      </div>
      <aside class="hero-status" aria-label="Research configuration">
        <div class="status-label">
          Research configuration
          <span class="status-badge">Validation mode</span>
        </div>
        <dl class="status-list">
          <div><dt>Selected model</dt><dd>Gradient Boosting</dd></div>
          <div><dt>Forecast horizon</dt><dd>10 trading days</dd></div>
          <div><dt>Universe</dt><dd>Current VN30 constituents</dd></div>
          <div><dt>Publication</dt><dd>Static historical snapshot</dd></div>
          <div><dt>Execution</dt><dd>No real-money orders</dd></div>
        </dl>
      </aside>
    </div>
  </header>

  <main id="main-content">
    <section class="metric-grid">
      {summary_cards_html()}
    </section>

    {research_validity_html()}

    <section class="section-card" id="benchmarks">
      <div class="section-kicker">Context</div>
      <h2>Baseline comparison</h2>
      <p class="muted">
        Compares the ML strategy with equal-weight VN30-style exposure and simple rule-based baselines.
        ML strategy rows use after-cost active returns. Naive baseline rows use before-cost active returns
        versus the VN30-style reference. This is a diagnostic comparison, not live evidence.
      </p>
      {cumulative_active_return_note_html()}
      <div class="table-wrap">
        {benchmark_table_html()}
      </div>
    </section>

    <section class="section-card" id="portfolio">
      <div class="section-kicker">Portfolio risk</div>
      <h2>Latest concentration risk</h2>
      <p class="muted">
        Shows single-name concentration, issuer-group concentration, HHI, and effective position count
        for the latest optimized portfolio snapshot.
      </p>
      <div class="table-wrap">
        {concentration_summary_table_html()}
      </div>
    </section>

    <section class="section-card" id="issuer-risk">
      <h2>Latest issuer-group exposure</h2>
      <p class="muted">
        Surfaces issuer groups such as Vingroup where multiple tickers can create hidden concentration.
        Rows at the 40 percent issuer-group cap are flagged directly.
      </p>
      <div class="table-wrap">
        {issuer_group_exposure_table_html()}
      </div>
    </section>

    <section class="section-card" id="optimizer-diagnostic">
      <h2>Optimizer bound diagnostic</h2>
      <p class="muted">
        Shows whether the optimizer is genuinely diversifying or mainly hitting the 20 percent single-name cap.
        The latest snapshot has all five holdings at the single-name cap, so this is a portfolio-construction risk check.
      </p>
      <div class="table-wrap">
        {optimizer_bound_diagnostic_table_html()}
      </div>
    </section>

    <section class="section-card" id="rankings">
      <div class="section-kicker">Signal diagnostics</div>
      <h2>Latest predicted rank vs realized rank</h2>
      <p class="muted">
        Compares the latest model ranking with realized forward-return rank so hits and misses are visible.
      </p>
      {vhm_vic_note_html}
      <div class="table-wrap">
        {latest_rank_diagnostic_table_html()}
      </div>
    </section>

    <section class="section-card" id="stock-ranking">
      <h2>Latest stock ranking</h2>
      <p class="muted">Stocks ranked by the latest gradient boosting model score.</p>
      <div class="table-wrap">
        {format_table(
            ranking,
            max_rows=15,
            percent_columns={"model_score", "realized_forward_return"},
        )}
      </div>
    </section>

    <section class="section-card" id="portfolio-weights">
      <h2>Latest optimized portfolio weights</h2>
      <p class="muted">Latest generated long-only optimized weights from the framework.</p>
      <div class="table-wrap">
        {format_table(
            weights,
            max_rows=20,
            percent_columns={"weight", "model_score", "realized_forward_return"},
        )}
      </div>
    </section>

    <section class="section-card" id="horizons">
      <div class="section-kicker">Model evidence</div>
      <h2>Forecast horizon results</h2>
      <p class="muted">
        Compares whether 1-day, 5-day, or 10-day prediction targets work better.
        Average after-cost return is measured per forecast period, not annualized. Rank IC is unitless, from -1 to +1.
      </p>
      {cumulative_active_return_note_html()}
      <div class="table-wrap">
        {horizon_table_html()}
      </div>
      <h3 style="margin-top: 18px;">Overlapping-window disclosure</h3>
      <p class="muted">
        Multi-day forecast horizons use overlapping evaluated dates, so the raw count is larger than the
        approximate number of independent non-overlapping periods.
      </p>
      <p class="muted table-note">
        10-day horizon: 1,584 evaluated dates (~158 non-overlapping 10-day periods).
        Overlapping forecast windows inflate the apparent sample count.
      </p>
      <div class="table-wrap">
        {horizon_disclosure_table_html()}
      </div>
    </section>

    <section class="section-card">
      <h2>Model comparison diagnostic</h2>
      <p class="muted">
        Compares available linear, tree-based, and classification prediction files using an equal-weight top-5 diagnostic.
        This is not the same as the optimized transaction-cost-aware backtest. Rank IC is unitless, from -1 to +1.
      </p>
      <div class="table-wrap">
        {model_comparison_table_html()}
      </div>
    </section>

    <section class="section-card">
      <h2>Feature ablation results</h2>
      <p class="muted">
        Checks whether the full feature set beats reduced feature groups. Rank IC is unitless, from -1 to +1.
      </p>
      {cumulative_active_return_note_html()}
      <div class="table-wrap">
        {ablation_table_html()}
      </div>
    </section>

    <section class="section-intro" id="charts">
      <div class="section-kicker">Interactive exploration</div>
      <h2>Performance and risk charts</h2>
      <p>Zoom into the walk-forward history, compare scenarios, and inspect where the model's behavior changes.</p>
    </section>

    {interactive_sections}

    {static_grid_html()}

    {glossary_html()}

    <section class="disclaimer">
      <strong>Research disclaimer:</strong>
      This dashboard summarizes a historical research backtest. It is not trade-ready and does not account for
      all live execution constraints, data-vendor issues, point-in-time VN30 membership, tax, borrow constraints,
      or real-time market impact.
    </section>
  </main>
</body>
</html>
"""


def polish_dashboard_html(html: str) -> str:
    import re

    cumulative_note = (
        '<p class="muted"><strong>Unit note:</strong> '
        'Cumulative active-return sum is the sum of overlapping forecast-period active returns, '
        'not a compounded portfolio return.</p>'
    )

    baseline_note = (
        '<p class="muted"><strong>Baseline comparison is included:</strong> '
        'ML strategy rows use after-cost active returns. Naive baseline rows use before-cost '
        'active returns versus the VN30-style reference. This is a diagnostic comparison, '
        'not live-trading evidence.</p>'
    )

    overlap_note = (
        '<p class="muted"><strong>Overlapping-window note:</strong> '
        '10-day horizon: 1,584 evaluated dates (~158 non-overlapping 10-day periods). '
        'Overlapping forecast windows inflate the apparent sample count.</p>'
    )

    equal_weight_note = (
        '<p class="muted"><strong>Equal-weight baseline note:</strong> '
        'Equal-weight active return is approximately zero by construction when compared with an equal-weight VN30-style reference. '
        'This row confirms benchmark alignment, not a standalone alpha strategy result.</p>'
    )

    issuer_guidance = (
        '<p class="muted">Hover over each bar to inspect tickers, weights, and concentration flags. '
        'The 40 percent reference line marks the issuer-group cap.</p>'
    )

    drawdown_note = (
        '<p class="muted"><strong>Drawdown window note:</strong> '
        'The 2026 drawdown is inside the walk-forward backtest window and should be interpreted '
        'as part of the out-of-sample diagnostic period, not live-trading evidence.</p>'
    )

    model_comparison_note = (
        '<p class="muted"><strong>Model-comparison note:</strong> '
        'Hit rates here use a simpler equal-weight top-5 diagnostic by raw model score, not the optimizer backtest. '
        'Use the forecast-horizon table for optimizer-based hit-rate results.</p>'
    )

    def format_cumulative_sum_tables(page_html: str) -> str:
        def format_number_cell(cell_html: str) -> str:
            raw_text = re.sub(r"<.*?>", "", cell_html).strip()

            if not raw_text or raw_text.upper() in {"N/A", "NA", "NAN"}:
                return cell_html

            if "%" in raw_text:
                return cell_html

            try:
                value = float(raw_text.replace(",", ""))
            except ValueError:
                return cell_html

            return re.sub(
                r"(<td[^>]*>).*?(</td>)",
                rf"\g<1>{value:.2f}%\2",
                cell_html,
                count=1,
                flags=re.DOTALL,
            )

        def process_table(match: re.Match) -> str:
            table = match.group(0)

            if "Cumulative active-return sum" not in table:
                return table

            header_match = re.search(r"<tr[^>]*>(.*?)</tr>", table, flags=re.DOTALL)
            if not header_match:
                return table

            headers = re.findall(r"<th[^>]*>.*?</th>", header_match.group(1), flags=re.DOTALL)
            header_names = [
                re.sub(r"<.*?>", "", header).strip()
                for header in headers
            ]

            if "Cumulative active-return sum" not in header_names:
                return table

            target_index = header_names.index("Cumulative active-return sum")

            def process_row(row_match: re.Match) -> str:
                row = row_match.group(0)

                if "<th" in row:
                    return row

                cells = re.findall(r"<td[^>]*>.*?</td>", row, flags=re.DOTALL)

                if target_index >= len(cells):
                    return row

                old_cell = cells[target_index]
                new_cell = format_number_cell(old_cell)

                if old_cell == new_cell:
                    return row

                return row.replace(old_cell, new_cell, 1)

            return re.sub(r"<tr[^>]*>.*?</tr>", process_row, table, flags=re.DOTALL)

        return re.sub(r"<table[^>]*>.*?</table>", process_table, page_html, flags=re.DOTALL)


    # Rename raw/internal cumulative-return headers wherever they leak into HTML.
    html = html.replace(
        "final_cumulative_after_cost_active_return",
        "Cumulative active-return sum",
    )
    html = html.replace(
        "final_cumulative_active_return_sum",
        "Cumulative active-return sum",
    )
    html = html.replace(
        "final_cumulative_active_return",
        "Cumulative active-return sum",
    )

    html = format_cumulative_sum_tables(html)

    # Make Rank IC explicitly unitless.
    html = html.replace(
        "<th>average_rank_ic</th>",
        "<th>Rank IC (unitless, -1 to +1)</th>",
    )
    html = html.replace(
        "<th>rank_ic</th>",
        "<th>Rank IC (unitless, -1 to +1)</th>",
    )

    # Add a visible cumulative-return note near horizon/baseline tables.
    if "Sum of overlapping forecast-period active returns, not a compounded portfolio return." not in html:
        inserted = False
        for pattern in [
            r"(<h2>Forecast horizon[^<]*</h2>)",
            r"(<h2>Horizon[^<]*</h2>)",
            r"(<h2>Baseline comparison</h2>)",
        ]:
            new_html, count = re.subn(
                pattern,
                r"\1\n" + cumulative_note,
                html,
                count=1,
                flags=re.IGNORECASE,
            )
            if count:
                html = new_html
                inserted = True
                break

        if not inserted:
            html = html.replace("</main>", cumulative_note + "\n</main>", 1)

    # Make baseline section impossible to miss.
    if "Baseline comparison is included:" not in html:
        html = re.sub(
            r"(<h2>Baseline comparison</h2>)",
            r"\1\n" + baseline_note,
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    # Make overlapping-window disclosure impossible to miss.
    if "Overlapping-window note:" not in html:
        inserted = False
        for pattern in [
            r"(<h2>Forecast horizon[^<]*</h2>)",
            r"(<h2>Horizon[^<]*</h2>)",
        ]:
            new_html, count = re.subn(
                pattern,
                r"\1\n" + overlap_note,
                html,
                count=1,
                flags=re.IGNORECASE,
            )
            if count:
                html = new_html
                inserted = True
                break

        if not inserted:
            html = html.replace("</main>", overlap_note + "\n</main>", 1)

    # Replace misleading time-series zoom instruction on the cross-sectional issuer chart.
    issuer_pattern = (
        r"(<h2>Latest issuer-group exposure chart</h2>\s*)"
        r"<p class=\"muted\">.*?</p>"
    )
    new_html, count = re.subn(
        issuer_pattern,
        r"\1" + issuer_guidance,
        html,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    html = new_html

    if count == 0 and "Latest issuer-group exposure chart" in html and "Hover over each bar" not in html:
        html = re.sub(
            r"(<h2>Latest issuer-group exposure chart</h2>)",
            r"\1\n" + issuer_guidance,
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    # Add drawdown window clarification.
    if "walk-forward backtest window" not in html:
        new_html, count = re.subn(
            r"(<h2>[^<]*drawdown[^<]*</h2>)",
            r"\1\n" + drawdown_note,
            html,
            count=1,
            flags=re.IGNORECASE,
        )
        html = new_html

        if count == 0:
            html = html.replace("</main>", drawdown_note + "\n</main>", 1)

    return html


def postprocess_dashboard_file() -> None:
    dashboard_path = ROOT / "reports" / "dashboard.html"

    if not dashboard_path.exists():
        return

    html = dashboard_path.read_text(encoding="utf-8")
    polished = polish_dashboard_html(html)
    dashboard_path.write_text(polished, encoding="utf-8", newline="\n")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    html = page_html()
    clean_html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    OUTPUT_PATH.write_text(clean_html, encoding="utf-8", newline="\n")

    print()
    print("Fresh HTML dashboard generated:")
    print(f"  {OUTPUT_PATH}")


def force_final_review_notes() -> None:
    import re

    dashboard_path = ROOT / "reports" / "dashboard.html"

    if not dashboard_path.exists():
        return

    html = dashboard_path.read_text(encoding="utf-8")

    equal_weight_note = (
        '<p class="muted"><strong>Equal-weight baseline note:</strong> '
        'Equal-weight active return is approximately zero by construction when compared with an equal-weight VN30-style reference. '
        'This row confirms benchmark alignment, not a standalone alpha strategy result.</p>'
    )

    model_hit_note = (
        '<p class="muted"><strong>Model-comparison hit-rate note:</strong> '
        'Hit rates here use a simpler equal-weight top-5 diagnostic by raw model score, not the optimizer backtest. '
        'Use the forecast-horizon table for optimizer-based hit-rate results.</p>'
    )

    if "Equal-weight active return is approximately zero" not in html:
        html, count = re.subn(
            r"(<h2>Baseline comparison</h2>)",
            r"\1\n" + equal_weight_note,
            html,
            count=1,
            flags=re.IGNORECASE,
        )

        if count == 0:
            html = html.replace("</main>", equal_weight_note + "\n</main>", 1)

    if "Hit rates here use a simpler equal-weight top-5 diagnostic" not in html:
        html, count = re.subn(
            r"(<h2>Model comparison diagnostic</h2>)",
            r"\1\n" + model_hit_note,
            html,
            count=1,
            flags=re.IGNORECASE,
        )

        if count == 0:
            html = html.replace("</main>", model_hit_note + "\n</main>", 1)

    dashboard_path.write_text(html, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
    postprocess_dashboard_file()
    force_final_review_notes()
    postprocess_dashboard_file()
