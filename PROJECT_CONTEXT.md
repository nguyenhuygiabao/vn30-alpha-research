# VN30 Long-Only Machine Learning Framework - Project Context

This file is the compact handoff prompt for continuing the VN30 Long-Only Machine Learning Framework.

Use this file when starting a new ChatGPT session or when asking another coding assistant to continue the project.

## Repository

GitHub:

https://github.com/nguyenhuygiabao/vn30-long-only-ml-framework

Local Windows path:

C:\Users\HUY BAO\vn30-long-only-ml-framework

## Project purpose

This project builds a research-grade VN30 long-only machine learning framework.

The framework ranks VN30 stocks, predicts forward relative returns versus the VN30 benchmark, constructs constrained long-only portfolios, accounts for Vietnam-specific trading frictions, and produces methodology/results/report artifacts.

This project should be presented as a transparent research framework, not a live deployable trading strategy and not financial advice.

## Mentoring workflow preference

Use a coding mentor workflow.

Preferred style:

1. Give executable commands or code first.
2. Keep routine verification grouped when safe.
3. For risky edits, go one step at a time.
4. Use exact file paths and exact PowerShell commands.
5. Wait for verification output before moving to risky next steps.
6. Do not commit generated raw/parquet data.
7. Keep generated data local and untracked unless it is a final report table or figure.

Important Python variable preference:

Use lowercase matrix/data names where relevant:

x_train, x_val, x_test, y_train, y_val, y_test

## Current data

Main raw data source:

data/raw/vnstock/vn30_ohlcv.csv

Current dataset:

- 46,863 rows
- 30 tickers
- Date range: 2020-01-02 to 2026-06-30
- Daily end-of-day OHLCV data
- Not intraday data
- Not HFT

Current VN30 universe:

ACB, BID, BSR, CTG, FPT, GAS, GVR, HDB, HPG, LPB,
MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB,
TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE

Universe metadata:

config/vn30_universe.csv

Issuer group note:

The Vingroup issuer group includes VHM, VIC, VPL, VRE.

Important limitation:

The current universe is static, not fully point-in-time. This may introduce survivorship bias.

## Labels

The project predicts forward relative returns versus VN30 benchmark return.

Created labels:

- forward_relative_return_1d
- forward_relative_return_5d
- forward_relative_return_10d

Original baseline label:

forward_relative_return_5d

Current horizon conclusion:

- 1d is weak/noisy
- 5d is strong
- 10d is strongest in current tests

## Feature groups

Feature families include:

- Momentum and returns
- Risk
- Liquidity
- Volume shocks
- Price-limit behavior
- Herding proxies
- Market breadth
- Rolling correlation
- Issuer-group metadata

Important feature file:

data/processed/features_combined.parquet

## Important source modules

src/feature_pipeline.py
src/labels.py
src/walk_forward_split.py
src/linear_models.py
src/tree_models.py
src/classification_models.py
src/baseline_portfolios.py
src/risk_model.py
src/optimizer.py
src/backtester.py
src/price_limit.py
src/ablation_tests.py
src/horizon_tests.py
src/visualize_results.py

## Validation

The project uses walk-forward validation.

Random train/test splitting should not be used for the main financial time-series evaluation.

Purging logic is used to reduce leakage from overlapping forward-return labels.

Features should be past-only.

The rolling risk model uses shifted returns before rolling calculations.

## Risk model

File:

src/risk_model.py

Settings:

RISK_WINDOW = 60
MIN_RISK_OBSERVATIONS = 20

The risk model uses past-only rolling beta and residual risk.

## Optimizer

File:

src/optimizer.py

Core constraints:

MAX_WEIGHT = 0.20
MAX_TURNOVER = 0.40
MAX_ISSUER_GROUP_WEIGHT = 0.40

Optimizer modes:

- normal
- herding_aware

Herding-aware settings:

HERDING_SCORE_COLUMN = herding_corr_score
HIGH_HERDING_THRESHOLD = 0.80
HERDING_AWARE_TOP_N = 5
HERDING_AWARE_MAX_WEIGHT = 0.15
HERDING_AWARE_TARGET_EXPOSURE = 0.75
HERDING_AWARE_MIN_PREDICTED_RETURN = 0.0

Herding-aware risk-control result:

- normal high-herding average exposure: 0.988337
- herding-aware high-herding average exposure: 0.738502
- normal high-herding average max weight: 0.193131
- herding-aware high-herding average max weight: 0.148358
- normal high-herding average selected count: 10.165217
- herding-aware high-herding average selected count: 7.713043

Interpretation:

Herding-aware optimization reduces exposure, concentration, and turnover during high-herding regimes, but gives up some return.

## Backtester

File:

src/backtester.py

Transaction-cost assumptions:

COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
LIQUIDITY_PENALTY_RATE = 0.001
low-liquidity quantile = 0.20

Execution modes:

- normal
- price_limit_aware

Price-limit-aware execution:

- Blocks buys at close-at-ceiling
- Blocks sells at close-at-floor
- Recomputes feasible trades after restrictions

Important Vietnam-specific price detail:

VN stock prices from vnstock are in thousands of VND.

HOSE tick-size logic in src/price_limit.py should use:

- price < 10: tick size 0.01
- price < 50: tick size 0.05
- otherwise: tick size 0.10

## Main backtest results

Day 24 main backtest results:

normal optimizer + normal execution:

- diagnostic Sharpe: 0.873634
- max active drawdown: -0.278142
- average turnover: 0.357871
- final cumulative after-cost active return: 32.377818

normal optimizer + price-limit-aware execution:

- diagnostic Sharpe: 0.860436
- max active drawdown: -0.370818
- average turnover: 0.353277
- final cumulative after-cost active return: 31.487395

herding-aware optimizer + normal execution:

- diagnostic Sharpe: 0.872999
- max active drawdown: -0.278142
- average turnover: 0.351899
- final cumulative after-cost active return: 31.703161

herding-aware optimizer + price-limit-aware execution:

- diagnostic Sharpe: 0.860766
- max active drawdown: -0.370818
- average turnover: 0.347622
- final cumulative after-cost active return: 30.885591

Interpretation:

The normal optimizer gives the strongest return. The herding-aware optimizer reduces exposure/concentration/turnover but gives up some return. Price-limit-aware execution is more conservative and more realistic.

## Feature ablation results

File:

reports/tables/ablation_results.csv

Results:

all_features:

- feature count: 51
- average rank IC: 0.338033
- hit rate: 0.676072
- top-5 actual return: 0.023634
- diagnostic Sharpe: 0.840073
- drawdown: -0.355852
- average turnover: 0.354637
- final return: 29.360326

without_herding:

- feature count: 41
- average rank IC: 0.337983
- hit rate: 0.676321
- top-5 actual return: 0.023677
- diagnostic Sharpe: 0.827557
- drawdown: -0.355852
- average turnover: 0.356006
- final return: 29.281588

without_price_limit:

- feature count: 36
- average rank IC: 0.316762
- hit rate: 0.666128
- top-5 actual return: 0.022343
- diagnostic Sharpe: 0.784717
- drawdown: -0.359316
- average turnover: 0.361541
- final return: 27.268516

without_risk:

- feature count: 49
- average rank IC: 0.314453
- hit rate: 0.657303
- top-5 actual return: 0.021847
- diagnostic Sharpe: 0.749285
- drawdown: -0.363631
- average turnover: 0.365078
- final return: 26.301430

without_volume_liquidity:

- feature count: 40
- average rank IC: 0.307164
- hit rate: 0.655065
- top-5 actual return: 0.021507
- diagnostic Sharpe: 0.742006
- drawdown: -0.322599
- average turnover: 0.361687
- final return: 26.341446

Ablation conclusion:

The full feature set is best overall. Herding features add limited raw predictive alpha. Volume/liquidity, price-limit, and risk features matter more. Herding remains useful as a portfolio-level risk-control mechanism.

## Forecast horizon results

File:

reports/tables/horizon_results.csv

Results:

1d:

- average rank IC: -0.003844
- top-5 hit rate: 0.447861
- top-5 actual return: -0.000067
- diagnostic Sharpe: -0.140805
- max active drawdown: -1.525574
- average turnover: 0.389600
- final cumulative after-cost active return: -1.508238

5d:

- average rank IC: 0.338033
- top-5 hit rate: 0.676072
- top-5 actual return: 0.023634
- diagnostic Sharpe: 0.840073
- max active drawdown: -0.355852
- average turnover: 0.354637
- final cumulative after-cost active return: 29.360326

10d:

- average rank IC: 0.546504
- top-5 hit rate: 0.801122
- top-5 actual return: 0.052681
- diagnostic Sharpe: 1.398924
- max active drawdown: -0.219472
- average turnover: 0.317395
- final cumulative after-cost active return: 77.915433

Horizon conclusion:

10d is best on core metrics. 1d is noisy/weak. 5d remains a strong original baseline.

## Main reports

reports/final_results.md
reports/methodology.md
reports/final_audit.md
reports/model_report.md
reports/data_quality_report.md
reports/report_index.md

## Main result tables

reports/tables/ablation_results.csv
reports/tables/horizon_results.csv

## Main figures

reports/figures/cumulative_after_cost_active_return.png
reports/figures/active_drawdown.png
reports/figures/portfolio_turnover.png
reports/figures/rolling_diagnostic_sharpe.png
reports/figures/top_gradient_boosting_feature_importance.png
reports/figures/ablation_diagnostic_sharpe.png
reports/figures/horizon_diagnostic_sharpe.png
reports/figures/horizon_rank_ic.png

There may also be extra exploratory figures such as reports/figures/fpt_momentum_features.png.

## How to view outputs

Open main reports:

code .\reports\final_results.md
code .\reports\methodology.md
code .\reports\final_audit.md
code .\reports\report_index.md

Preview Markdown in VS Code:

Ctrl + Shift + V

Open figures:

explorer .\reports\figures

Open tables:

explorer .\reports\tables

Quick command-line summary:

py .\scripts\report_summary.py

## Git hygiene

Generated data files should stay local and untracked.

Ignored generated data paths include:

data/processed/
data/raw/vnstock/
data/raw/yahoo/

Do not add generated raw/parquet files to Git unless explicitly requested.

Final report tables and figures can be committed.

## Current usability upgrades already completed

- Updated README.md as the GitHub front page
- Added PROJECT_CONTEXT.md as the reusable project handoff prompt
- Added reports/report_index.md
- Added scripts/report_summary.py
- Added scripts/run_full_pipeline.py for controlled one-command pipeline refresh
- Added scripts/clean_generated_outputs.py for safe stale-output preview and cleanup
- Added .gitignore rules for generated data
- Removed BOM from requirements.txt
- Made reports/data_quality_report.md deterministic by removing runtime timestamp noise

## One-command pipeline runner

File:

scripts/run_full_pipeline.py

Main commands:

py .\scripts\run_full_pipeline.py
py .\scripts\clean_generated_outputs.py
py .\scripts\clean_generated_outputs.py --confirm
py .\scripts\run_full_pipeline.py --stop-after data_quality
py .\scripts\run_full_pipeline.py --start-at features --stop-after labels
py .\scripts\run_full_pipeline.py --start-at linear_models --stop-after classification_models
py .\scripts\run_full_pipeline.py --start-at baseline_portfolios --stop-after backtester
py .\scripts\run_full_pipeline.py --start-at ablation_tests --stop-after report_summary
py .\scripts\run_full_pipeline.py

Important implementation note:

The runner executes src modules with py -m style through sys.executable, so relative imports work correctly.

## Priority next upgrades

1. Keep PROJECT_CONTEXT.md updated when project state changes.
2. Connect stale-output cleaning into the pipeline runner after more testing.
3. Add clean audit/update workflow after each rerun.
4. Add optional daily auto-update through Windows Task Scheduler or GitHub Actions later.
5. Add Streamlit dashboard after the command-line workflow is stable.

## Future research upgrades

- Point-in-time VN30 membership
- Non-overlapping portfolio evaluation
- More realistic execution model
- Fundamental data
- Foreign ownership or flow data
- Macro variables
- Reproducible sentiment/news features
- Stronger out-of-sample robustness tests

## Current next instruction for future assistants

First check:

git status --short
git log --oneline -5

If the repo is clean, continue with the next priority upgrade.

Do not assume generated parquet/raw data should be committed.

If adding automation, prefer scripts/ for PowerShell-runnable entry points.

If running src modules that use relative imports, run them as modules:

py -m src.module_name

not as direct files like:

py .\src\module_name.py
