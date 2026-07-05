# VN30 Long-Only Machine Learning Framework

A research-grade framework for ranking VN30 stocks, forecasting forward relative returns versus the VN30 benchmark, and constructing constrained long-only portfolios under Vietnam-specific market frictions.

This project is a transparent research framework. It is not a live trading system and it is not financial advice.

## Project objective

The goal is to study whether daily public OHLCV-based features can support medium-frequency VN30 stock ranking and long-only portfolio construction.

The framework focuses on:

- Forward relative-return prediction versus the VN30 benchmark
- Walk-forward validation instead of random train-test splitting
- Long-only constrained portfolio construction
- Vietnam-specific transaction costs and price-limit frictions
- Feature ablation and forecast horizon robustness testing
- Final report artifacts for methodology, results, audit, tables, and figures

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
