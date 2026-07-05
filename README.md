# VN30 Long-Only Machine Learning Framework

Ranks VN30 stocks by expected 1-, 5-, and 10-day outperformance, then tests whether those signals can support a constrained long-only portfolio.

The framework uses daily OHLCV data, walk-forward validation, transaction costs, portfolio constraints, and Vietnam-specific market frictions. It is a research project, not a live trading system or investment advice.

## Key result

**The 10-day horizon is the strongest tested setup so far.**

In the current historical backtest, the 10-day label has the best Rank IC, top-5 hit rate, diagnostic Sharpe, and final after-cost active return among the tested horizons.

Known limitations remain: the VN30 universe is static, results are point estimates, and multiple tested configurations create selection risk.

## Dashboard

Main report:

    reports/dashboard.html

Rebuild and open locally:

    cd vn30-long-only-ml-framework
    py .\scripts\build_interactive_charts.py
    py .\scripts\build_html_report.py
    start .\reports\dashboard.html

Quick terminal summary:

    py .\scripts\report_summary.py

## Method summary

The project includes:

- OHLCV feature engineering
- Forward relative-return labels versus VN30 benchmark
- Walk-forward model validation
- Linear, tree-based, gradient boosting, and classification models
- Forecast-horizon and feature-ablation tests
- Long-only portfolio optimization
- Transaction-cost-aware backtesting
- Price-limit-aware execution comparison
- Herding-aware portfolio risk control
- Static reports and interactive dashboard

## Key terms

- **Rank IC**: correlation between predicted stock ranking and realized future ranking.
- **Top-5 hit rate**: how often top-ranked stocks become strong realized performers.
- **Diagnostic Sharpe**: risk-adjusted comparison metric used inside this research framework.
- **Active return**: return relative to the benchmark or reference portfolio.
- **After-cost active return**: active return after estimated trading costs.
- **Max active drawdown**: largest fall from a previous active-return peak.
- **Walk-forward validation**: train on past data, test on later unseen dates.
- **Feature ablation**: remove feature groups to test whether they add value.

## Metric conventions

- Decimal values in result tables are often percent-like units. Example: `-0.219472` means about `-21.95%`.
- Rank IC and diagnostic Sharpe are unitless.
- Final after-cost active return is a historical backtest metric, not a live-return claim.

## Data

Main raw data source:

```text
data/raw/vnstock/vn30_ohlcv.csv
```

Current dataset:

- 46,863 daily OHLCV rows
- 30 VN30 tickers
- Date range: 2020-01-02 to 2026-06-30
- Daily end-of-day data, not intraday data

Important limitation:

The current VN30 universe is static, not fully point-in-time. This may introduce survivorship bias and should be addressed in future work.

## Current VN30 universe

```text
ACB, BID, BSR, CTG, FPT, GAS, GVR, HDB, HPG, LPB,
MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB,
TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE
```

Issuer-group metadata is stored in:

```text
config/vn30_universe.csv
```

## Framework components

The project includes:

- Feature engineering
- Forward relative-return labels
- Walk-forward validation
- Linear, tree-based, and classification models
- Baseline portfolios
- Rolling past-only risk model
- Constrained long-only optimizer
- Transaction-cost-aware backtester
- Price-limit-aware execution comparison
- Herding-aware portfolio risk control
- Feature ablation testing
- Forecast horizon testing
- Result visualization
- Final methodology, results, and audit reports

## Main research finding

The strongest tested forecast horizon is currently the 10-day forward relative-return label.

Forecast horizon results:

| Horizon | Average Rank IC | Top-5 Hit Rate | Diagnostic Sharpe | Max Active Drawdown | Final After-Cost Active Return |
|---|---:|---:|---:|---:|---:|
| 1d | -0.003844 | 0.447861 | -0.140805 | -1.525574 | -1.508238 |
| 5d | 0.338033 | 0.676072 | 0.840073 | -0.355852 | 29.360326 |
| 10d | 0.546504 | 0.801122 | 1.398924 | -0.219472 | 77.915433 |

Interpretation:

- 1-day prediction is weak and noisy.
- 5-day prediction is a strong baseline.
- 10-day prediction is the strongest tested research horizon so far.

## Feature ablation summary

The full feature set performs best overall.

| Feature Set | Feature Count | Average Rank IC | Diagnostic Sharpe | Final After-Cost Active Return |
|---|---:|---:|---:|---:|
| all_features | 51 | 0.338033 | 0.840073 | 29.360326 |
| without_herding | 41 | 0.337983 | 0.827557 | 29.281588 |
| without_price_limit | 36 | 0.316762 | 0.784717 | 27.268516 |
| without_risk | 49 | 0.314453 | 0.749285 | 26.301430 |
| without_volume_liquidity | 40 | 0.307164 | 0.742006 | 26.341446 |

Interpretation:

- Volume/liquidity, price-limit, and risk features matter more for raw predictive performance.
- Herding features add limited standalone alpha.
- Herding remains useful as a portfolio-level risk-control mechanism.

## Important reports

Open these first:

```text
reports/final_results.md
reports/methodology.md
reports/final_audit.md
```

Additional reports:

```text
reports/model_report.md
reports/data_quality_report.md
```

## Tables

```text
reports/tables/ablation_results.csv
reports/tables/horizon_results.csv
```

## Figures

```text
reports/figures/cumulative_after_cost_active_return.png
reports/figures/active_drawdown.png
reports/figures/portfolio_turnover.png
reports/figures/rolling_diagnostic_sharpe.png
reports/figures/top_gradient_boosting_feature_importance.png
reports/figures/ablation_diagnostic_sharpe.png
reports/figures/horizon_diagnostic_sharpe.png
reports/figures/horizon_rank_ic.png
```

## How to view outputs locally

Open reports in VS Code:

```powershell
code .\reports\final_results.md
code .\reports\methodology.md
code .\reports\final_audit.md
```

Preview Markdown in VS Code:

```text
Ctrl + Shift + V
```

Open figures:

```powershell
explorer .\reports\figures
```

Open tables:

```powershell
explorer .\reports\tables
```

## Pipeline runner workflows

The main local runner is:

```powershell
py .\scripts\run_full_pipeline.py
```

This runner refreshes outputs from existing local raw data only. It does not download or update live market data yet.

### Full local rebuild

Use this when you want to rebuild generated outputs from existing raw data:

```powershell
py .\scripts\run_full_pipeline.py --clean-first --audit-after
```

### Faster local refresh

Use this when you want to preserve expensive ML prediction caches and reuse them only when they are still fresh:

```powershell
py .\scripts\run_full_pipeline.py --clean-first --keep-expensive-ml --reuse-existing-ml --audit-after
```

Cache reuse is stale-aware. If features, labels, or relevant model source files change, the affected ML stages rerun automatically.

A cleaner dry run may report missing generated files. This is normal when optional or older generated outputs are listed by the cleaner but are not currently present locally.

## Repository hygiene

Generated data files are intentionally ignored by Git:

```text
data/processed/
data/raw/vnstock/
data/raw/yahoo/
```

These files are local reproducible outputs and should not be committed unless they are final report tables or figures.

## Project status and next upgrades

Completed usability upgrades:

- Updated the GitHub README as the project front page
- Added PROJECT_CONTEXT.md as the reusable project handoff prompt
- Added reports/report_index.md to link reports, tables, and figures
- Added scripts/report_summary.py for a quick command-line project summary
- Added scripts/run_full_pipeline.py for controlled one-command pipeline refresh
- Added --clean-first to the pipeline runner for optional stale-output cleanup
- Added --keep-expensive-ml to preserve costly model prediction caches during cleanup
- Added --reuse-existing-ml to skip expensive ML stages only when cached outputs are fresh
- Refactored shared modeling helpers into src/modeling_utils.py
- Added --summary-only to the pipeline runner for quick report checks
- Added scripts/clean_generated_outputs.py for safe stale-output preview and cleanup
- Added .gitignore rules so generated raw and processed data stay local
- Removed the UTF-8 BOM from requirements.txt
- Made the data-quality report deterministic by removing runtime timestamp noise

Priority next upgrades:

- Keep PROJECT_CONTEXT.md updated after major project changes
- Add a clean audit/update workflow after each rerun
- Add optional daily auto-update later through Windows Task Scheduler or GitHub Actions
- Add a Streamlit dashboard for tables and figures

Medium-term upgrades:

- Add stronger out-of-sample robustness tests
- Add non-overlapping portfolio evaluation
- Improve execution realism
- Add HTML or Excel report export if needed

Long-term research upgrades:

- Add point-in-time VN30 membership
- Add fundamental data
- Add foreign ownership or flow data
- Add macro variables
- Add reproducible sentiment or news features if reliable sources are available

## Disclaimer

This repository is for research and educational purposes only. It does not provide investment advice and does not claim to be a deployable trading strategy.

## Research validity limitations and next steps

This project is currently a research-grade backtesting framework, not a live trading system. The dashboard and reports should be read as historical diagnostics rather than investment recommendations.

Current methodological limitations:

- Static VN30 universe: the current dataset uses a fixed VN30 stock universe rather than point-in-time VN30 membership. This can create survivorship bias because today's constituents may not fully represent the investable VN30 universe at earlier dates.
- Multiple comparisons: the project tests multiple forecast horizons, feature ablations, optimizer variants, and execution assumptions on the same historical dataset. The best-looking result may partly reflect search across configurations rather than stable predictive skill.
- Point-estimate metrics: current Sharpe, Rank IC, drawdown, and return metrics are reported as point estimates. Future versions should add bootstrap confidence intervals or a deflated Sharpe ratio.
- Baseline comparison: future reports should display ML results directly against simple baselines such as equal-weight VN30 and a simple momentum-only rule.
- Corporate actions: before relying on automated daily updates, the project should further audit whether OHLCV prices are consistently adjusted for splits, stock dividends, and other HOSE corporate actions.
- Live-execution frictions: a future live or paper-trading workflow should consider VN-specific constraints such as foreign ownership room, liquidity, price limits, turnover, and realistic order execution.

Priority research upgrades before live trading claims:

1. Add explicit equal-weight and momentum-only baseline comparisons to the dashboard.
2. Add bootstrap confidence intervals or deflated Sharpe diagnostics.
3. Replace the static VN30 universe with point-in-time VN30 membership when reliable constituent history is available.
4. Audit corporate-action adjustment quality in the raw OHLCV source.
5. Add paper-trading validation before considering any live execution workflow.
