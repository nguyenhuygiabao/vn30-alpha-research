# VN30 Alpha Research Framework

Ranks VN30 stocks by expected 1-, 5-, and 10-day outperformance, then tests whether those signals can support a constrained long-only portfolio.

The framework uses FiinQuant-sourced daily OHLCV and foreign buy/sell data, point-in-time VN30 membership, walk-forward validation, transaction costs, portfolio constraints, and Vietnam-specific market frictions. It is a research project, not a live trading system or investment advice.

## Key result

After purged walk-forward validation and realistic transaction costs, the signal is weak and mostly does not survive a naive momentum baseline. Average Rank IC across the tested models is ~0.02 (5-day horizon) to ~0.04 (10-day horizon). Most after-cost ML portfolio variants are flat to slightly negative; the 10-day horizon is the one configuration that is genuinely after-cost positive on the tested history. A single-feature trailing-momentum baseline (`top5_momentum`, before cost) still outperforms every trained model's raw top-5 pick, before cost or portfolio construction even enter the picture — see `reports/final_results.md` and `Running Results Tracker.txt` for the full diagnosis.

The current project should be read as a reproducible VN30 research framework and data/leakage-control case study, not evidence of a deployable alpha strategy.

## Dashboard

Public dashboard:

    https://nguyenhuygiabao.github.io/vn30-alpha-research/reports/dashboard.html

Repository copy:

    reports/dashboard.html

Quick terminal summary:

    py .\scripts\report_summary.py

## Method summary

The project includes:

- FiinQuant raw + adjusted OHLCV and foreign buy/sell feature engineering
- Point-in-time VN30 membership (not a static universe)
- Forward relative-return labels versus VN30 benchmark
- Walk-forward model validation with purge/embargo gaps
- Linear, tree-based, gradient boosting, and classification models
- Forecast-horizon and feature-ablation tests
- Long-only portfolio optimization with issuer-group and sector caps
- Transaction-cost-aware backtesting
- Price-limit-aware execution comparison
- Herding-aware portfolio risk control
- Execution-data coverage auditing
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

Return columns are stored as decimals in the source CSV files. For example, `0.001` means `0.10%`, and `-0.219472` means `-21.95%`.

Dashboard and README tables display return-like values as percentages for readability.

Rank IC and diagnostic Sharpe are unitless.

Cumulative active-return figures are a **sum of overlapping forecast-period active returns**, not a compounded portfolio return. Read them as research diagnostics, not live portfolio performance.

## Data

Main data source (research and backtesting pipeline):

```text
data/raw/fiinquant/canonical/raw_market_history.csv
data/raw/fiinquant/canonical/adjusted_market_history.csv
data/raw/fiinquant/canonical/foreign_flow_history.csv
```

Current dataset:

- 137,719 raw daily OHLCV rows; 74,910 rows after point-in-time universe filtering and lag construction
- 60 tickers: the current 30 VN30 constituents plus all verified historical alumni
- Date range: 2016-07-26 to 2026-07-15
- Daily end-of-day data, not intraday data
- Raw price and volume are genuinely available for every active point-in-time constituent-date except two tickers (FLC, ROS) that predate FiinQuant's own raw-price coverage; those use a real vendor-archive raw-price backfill. Traded value for those same two tickers (1,013 of 74,910 rows) is not available from any source and uses a clearly-flagged typical-price estimate, never silently substituted. See `scripts/audit_execution_coverage.py` and `reports/tables/execution_coverage_summary.csv`.

Point-in-time VN30 membership is implemented (`config/vn30_history/`, `src/universe_reconstruction.py`, `src/universe_history.py`), so the modeling universe reflects who was actually in the index on each historical date rather than today's static 30 — this addresses the survivorship-bias limitation noted in earlier versions of this README.

**Separate, known limitation:** the live daily-update path used by the paper-trading foundation (`scripts/update_daily_data.py`, `config/paper_trading_config.yaml`) still uses vnstock, not FiinQuant — FiinQuant's bulk historical downloader is used for the research pipeline, but the paper-trading incremental updater has not yet been migrated. This does not affect any research or backtest result above; it only affects the live paper-trading data-refresh workflow described later in this file.

## Current VN30 universe

```text
ACB, BID, BSR, CTG, FPT, GAS, GVR, HDB, HPG, LPB,
MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB,
TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE
```

`config/vn30_universe.csv` covers all 60 historical constituents (current members plus alumni such as FLC, ROS, HAG, HNG, and others), with issuer-group and sector metadata used for the optimizer's concentration caps. Issuer-group entries deliberately link known affiliated companies for concentration-risk purposes (for example FLC and ROS share a controlling shareholder and are capped jointly).

## Framework components

The project includes:

- FiinQuant raw/adjusted market data and foreign buy/sell flow ingestion
- Point-in-time VN30 membership reconstruction and auditing
- Feature engineering, including foreign-flow features (not currently promoted to production models — see below)
- Forward relative-return labels
- Walk-forward validation with purge/embargo gaps
- Linear, tree-based, and classification models
- Baseline portfolios, including a simple trailing-momentum benchmark
- Rolling past-only risk model
- Constrained long-only optimizer with issuer-group and sector caps
- Transaction-cost-aware backtester
- Price-limit-aware execution comparison
- Herding-aware portfolio risk control
- Feature ablation testing
- Forecast horizon testing
- Execution-data coverage auditing
- Result visualization
- Final methodology, results, and audit reports

## Main research finding

Forecast horizon results (current, from a full pipeline rerun against the FiinQuant point-in-time dataset):

| Horizon | Average Rank IC | Top-5 Hit Rate | Average After-Cost Return / Period | Diagnostic Sharpe | Final Cumulative After-Cost Active Return | Evaluated Dates |
|---|---:|---:|---:|---:|---:|---:|
| 1d | 0.018752 | 0.467203 | -0.000703 | -0.114101 | -174.9% | 2488 |
| 5d | 0.019869 | 0.469492 | -0.000481 | -0.025742 | -119.1% | 2478 |
| 10d | 0.039083 | 0.487130 | 0.000804 | 0.030871 | 197.9% | 2463 |

Interpretation:

- 1-day prediction is weak and net-negative after cost.
- 5-day prediction (the original headline horizon) has roughly double the Rank IC of 1-day but is still net-negative after cost.
- 10-day prediction has the strongest Rank IC and is the only horizon that is after-cost positive on the tested history. Even so, this does not clear a naive momentum baseline on a like-for-like basis (see below), so it should be read as the best of the tested configurations, not as validated alpha.

## Feature ablation summary

| Feature Set | Feature Count | Average Rank IC | Top-5 Hit Rate | Diagnostic Sharpe | Final Cumulative After-Cost Active Return | Evaluated Dates |
|---|---:|---:|---:|---:|---:|---:|
| Excluding risk features (rolling vol, drawdown) | 97 | 0.025746 | 0.476755 | 0.038422 | -79.7% | 2478 |
| Excluding volume/liquidity features | 88 | 0.021549 | 0.469169 | 0.029155 | -133.7% | 2478 |
| Excluding herding features | 89 | 0.020276 | 0.468846 | 0.030913 | -141.8% | 2478 |
| All features | 99 | 0.019869 | 0.469492 | 0.025742 | -119.1% | 2478 |
| Excluding price-limit features | 84 | 0.013470 | 0.466990 | 0.037221 | -164.4% | 2478 |

Interpretation:

- Every configuration is net-negative after cost on this sample — no feature set tested here clears transaction costs.
- Removing the risk features (rolling volatility, drawdown — 2 features) gives the best Rank IC and top-5 return of the tested variants; this is a more promising ablation lead than feature-set size alone would suggest, and has not been pursued further.
- Price-limit features are clearly load-bearing: removing them is unambiguously worst on every metric.
- Read these as diagnostics, not stable feature-selection evidence — the ranking has already changed once this session after fixing a stale-cache bug in the ablation script, so it should not be treated as final.

## Foreign buy/sell flow (FB/FS)

Foreign investor buy/sell flow features were built and evaluated against a pre-registered decision rule (paired block bootstrap on daily Rank IC, promote only if the 95% CI excludes zero). Result: does not clear the bar (mean difference -0.0026, 95% CI [-0.0133, +0.0083], point estimate negative) and is **not promoted** to production models. The frozen net-flow features alone outperform every configuration that adds a buy/sell-split representation. Full diagnosis, including why (the features are used heavily by the model but their relationship with returns flips sign year to year — non-stationary, not simply noisy or badly represented), is in `docs/enhanced_fb_fs_checkpoint.md`.

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
docs/enhanced_fb_fs_checkpoint.md
Running Results Tracker.txt
```

## Tables

```text
reports/tables/ablation_results.csv
reports/tables/horizon_results.csv
reports/tables/benchmark_results.csv
reports/tables/execution_coverage_summary.csv
reports/tables/foreign_flow/
```

## Figures

```text
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

This runner refreshes outputs from existing local raw data only. It does not download or update live market data.

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

Cache reuse at the `run_full_pipeline.py` level is stale-aware (`--reuse-existing-ml` checks dependency file mtimes). Note that `ablation_tests.py` and `horizon_tests.py` previously had a separate, non-stale-aware internal cache that could silently serve weeks-old results; this was removed, and both scripts now always rebuild when invoked directly.

### Execution-data coverage audit

Checks, for every active point-in-time VN30 constituent on every trading date, whether raw OHLC, volume, and traded value (real or clearly-flagged estimate) are actually available:

```powershell
py .\scripts\audit_execution_coverage.py
```

### Foreign-flow experiment

Evaluates the foreign buy/sell flow feature sets (including FB/FS) against the frozen production model and the pre-registered decision rule. This is a long-running, expanding-window walk-forward refit (multiple hours on a full 9-year history) and is separate from the production pipeline:

```powershell
py .\scripts\run_foreign_flow_experiment.py
```

A cleaner dry run may report missing generated files. This is normal when optional or older generated outputs are listed by the cleaner but are not currently present locally.

## Paper-trading foundation

The repository includes a settlement-aware paper-trading foundation. It is
still an implementation and validation layer, not a live trading system, and its live daily-update path is currently vnstock-based (see the Data section above), not FiinQuant.

Current components include:

- explicit signal-date and next-trading-day execution timing
- completed-market-day and full-universe data validation
- configurable T+2 settlement assumptions
- settled and unsettled cash and share balances
- executable order sizing with concentration, turnover, liquidity, lot-size,
  price-limit, and minimum-trade constraints
- persistent local paper-account ledgers with reconciliation checks

Validate that the latest OHLCV file represents a completed market day:

```powershell
py .\scripts\validate_paper_trading_timing.py
```

Safely download and validate an incremental daily update without replacing the
current raw file:

```powershell
py .\scripts\update_daily_data.py --dry-run
```

On Python 3.14, Vnstock's optional chart dependency currently selects a NumPy
release without a compatible Windows wheel. Install the core Quote dependencies
without that optional chart package:

```powershell
py -m pip install pydantic openpyxl importlib-metadata tenacity "vnai>=2.4.8"
py -m pip install --no-deps "vnstock==4.0.4"
py -c "from vnstock import Quote; print('VNSTOCK QUOTE IMPORT PASSED')"
```

After the dry run passes, replace the local raw file with the validated update:

```powershell
py .\scripts\update_daily_data.py
```

The updater must run after the configured market-data cutoff. It downloads an
overlapping window, verifies provider continuity, requires full latest-date
universe coverage, rejects future or malformed rows, and stages the output before
replacing the existing CSV.

After setup, double-click `run_daily_data_update.cmd` in the repository root for
the guarded daily update. The window stays open and reports success or failure.
It validates the Vnstock Quote import before downloading, stops on any failed
validation, and does not run ML models or place orders. The same launcher can be
checked without replacing data from PowerShell:

```powershell
py .\scripts\run_daily_data_update.py --dry-run
```

Preview the selected 10-day Gradient Boosting ranking from completed data:

```powershell
py .\scripts\score_daily_signals.py
```

The scorer rebuilds features and 10-day relative-return labels directly from the
validated raw data. It fits only on dates whose complete forward outcome is known
at the signal cutoff, then scores all 30 tickers on the latest completed date.
It does not reuse historical backtest predictions. This command is a preview and
does not write ledgers, construct target weights, or place orders.

Preview constrained target weights after signal scoring passes:

```powershell
py .\scripts\preview_daily_targets.py
```

The target builder selects seven names and allocates 97 percent invested weight
subject to the 15 percent single-name cap and 25 percent issuer-family cap. If a
cluster of highly ranked names from one family makes the requested exposure
infeasible, lower-ranked names from other families are substituted in rank order.
The current Vingroup family is explicitly verified as VHM, VIC, VPL, and VRE.
This command remains a preview and writes no ledgers or orders.

After the paper configuration and opening capital are finalized, initialize the
local paper account once:

```powershell
py .\scripts\initialize_paper_account.py --asof-date YYYY-MM-DD
```

Reconcile an initialized account:

```powershell
py .\scripts\reconcile_paper_account.py
```

Generated paper-account CSV ledgers are local mutable state and are ignored by
Git. No real orders are placed by these scripts. Target-weight construction, the
daily paper runner, and the monitoring dashboard remain future work.

## Repository hygiene

Generated data files are intentionally ignored by Git:

```text
data/processed/
data/raw/
```

This covers both the FiinQuant canonical research data and the vnstock paper-trading data. These files are local reproducible outputs and should not be committed unless they are final report tables or figures.

## Testing

```powershell
py -m pytest -q
```

342 tests as of the last full run.

## Next upgrades

Priority next upgrades:

- Migrate the paper-trading live daily-update path from vnstock to FiinQuant
- Investigate why a naive trailing-momentum baseline beats every trained model's raw top-5 pick before cost or portfolio construction (see `Running Results Tracker.txt`)
- Decide whether to pursue the flagged foreign-flow/herding-regime interaction, or wait for a genuine Vietnam market-classification upgrade before re-testing FB/FS (see `docs/enhanced_fb_fs_checkpoint.md`)

Medium-term upgrades:

- Add non-overlapping portfolio evaluation
- Improve execution realism
- Add HTML or Excel report export if needed

Long-term research upgrades:

- Add fundamental data
- Add macro variables
- Add reproducible sentiment or news features if reliable sources are available

## License

This project is released under the MIT License. See `LICENSE` for details.

## Disclaimer

This repository is for research and educational purposes only. It does not provide investment advice and does not claim to be a deployable trading strategy.

## Research validity limitations and next steps

This project is currently a research-grade backtesting framework, not a live trading system. The dashboard and reports should be read as historical diagnostics rather than investment recommendations.

Current methodological limitations:

- Multiple comparisons: the project tests multiple forecast horizons, feature ablations, foreign-flow feature sets, optimizer variants, and execution assumptions on the same historical dataset. The best-looking result may partly reflect search across configurations rather than stable predictive skill. The foreign-flow experiment explicitly tracks this: the FB/FS result above is the fifth feature set tested on the same development data, and the pre-registered decision rule and multiplicity caveat are documented in `docs/enhanced_fb_fs_checkpoint.md`.
- Point-estimate metrics: most reported Sharpe, Rank IC, drawdown, and return metrics are point estimates without confidence intervals, except the foreign-flow comparison, which uses a paired block bootstrap. Extending bootstrap CIs to the main horizon/ablation results would strengthen those conclusions.
- Baseline comparison: `reports/tables/benchmark_results.csv` now compares ML results directly against simple baselines (equal-weight, top-5 momentum, top-5 reversal, low-volatility). The naive momentum baseline currently beats every trained model on a before-cost, matched-concentration basis; the root cause has been diagnosed (see `Running Results Tracker.txt`) but not resolved.
- Corporate actions: before relying on automated daily updates, the project should further audit whether OHLCV prices are consistently adjusted for splits, stock dividends, and other HOSE corporate actions.
- Live-execution frictions: a future live or paper-trading workflow should consider VN-specific constraints such as foreign ownership room, liquidity, price limits, turnover, and realistic order execution.
- Paper trading is on a different data source (vnstock) than the research pipeline (FiinQuant) and has not been reconciled against it.
