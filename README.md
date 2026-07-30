# VN30 Alpha Research Framework

Cross-sectional stock selection on the Vietnamese VN30 index. The framework ranks constituents by expected outperformance against the VN30 benchmark, converts rankings into a constrained long-only portfolio, and evaluates the result after transaction costs and Vietnam-specific execution frictions.

Research framework, not a live trading system. Not investment advice.

## Results

Forecast horizon comparison, gradient boosting, price-limit-aware execution:

| Horizon | Rank IC | Top-5 Hit Rate | After-Cost Return / Period | Diagnostic Sharpe | Evaluated Dates |
|---:|---:|---:|---:|---:|---:|
| 1d | 0.018752 | 0.467203 | -0.000703 | -0.114101 | 2488 |
| 5d | 0.019869 | 0.469492 | -0.000481 | -0.025742 | 2478 |
| 10d | 0.039083 | 0.487130 | 0.000804 | 0.030871 | 2463 |

The 10-day horizon is the only after-cost positive configuration.

Feature ablation, 5-day horizon:

| Configuration | Features | Rank IC | After-Cost Return / Period |
|---|---:|---:|---:|
| Excluding risk features | 97 | 0.025746 | -0.000321 |
| Excluding volume/liquidity | 88 | 0.021549 | -0.000540 |
| Excluding herding | 89 | 0.020276 | -0.000572 |
| All features | 99 | 0.019869 | -0.000481 |
| Excluding price limit | 84 | 0.013470 | -0.000664 |

No configuration clears transaction costs at the 5-day horizon.

Baseline comparison, top-5 selection, before cost, matched concentration and target:

| Selection Rule | Top-5 Active Return / 5d Period |
|---|---:|
| Trailing 10-day return (`top5_momentum`) | 0.002310 |
| random_forest | 0.001433 |
| elastic_net | 0.001351 |
| gradient_boosting | 0.000998 |
| ridge | 0.000786 |

A single-feature momentum rule outperforms every trained model. `return_10d` is already an input feature to all five models.

Foreign buy/sell features were tested against a decision rule fixed before results were seen (paired block bootstrap on daily Rank IC, 222 non-overlapping signals, promote only if the 95 percent confidence interval excludes zero). Mean difference -0.002608, CI [-0.013314, +0.008297]. Not promoted.

Cumulative return figures in this repository sum overlapping forecast-period active returns and are not compounded portfolio returns.

## Data

FiinQuant daily data, 137,719 rows across 60 tickers, 2016-07-26 to 2026-07-15. After point-in-time universe filtering and lag construction, 74,910 rows reach the modeling stage.

```text
data/raw/fiinquant/canonical/raw_market_history.csv
data/raw/fiinquant/canonical/adjusted_market_history.csv
data/raw/fiinquant/canonical/foreign_flow_history.csv
```

Raw prices are used for price limits and execution reference; adjusted close is used for returns and labels. The loader enforces the separation.

The universe is point-in-time VN30 membership reconstructed from verified index-review history in `config/vn30_history/`, covering 60 tickers across the full period (current 30 constituents plus 30 former constituents).

FLC and ROS raw OHLC was backfilled from a vendor archive. Traded value for those two tickers is unavailable from any source; 1,013 rows use a typical-price estimate flagged per row by `value_traded_is_estimated`. Execution-data coverage is otherwise complete: 73,897 of 74,910 constituent-dates have full real raw OHLC, volume, and traded value, with zero missing rows.

## Method

- Labels: forward return relative to VN30 at 1, 5, and 10 days
- Features: 99 model-facing columns across momentum, risk, liquidity, price-limit, herding, and foreign-flow groups
- Models: ridge, elastic net, random forest, gradient boosting, gradient-boosting classifier
- Validation: walk-forward with a 5-day purge at each boundary, roughly 2,480 windows
- Portfolio: long-only, 20 percent single-name cap, 40 percent turnover cap, 40 percent issuer-group cap, 35 percent sector cap
- Costs: 10 bps commission, 10 bps slippage, 10 bps sell tax, liquidity penalty, 5 percent ADV cap, 100-share lots, T+2 settlement
- Execution modes: frictionless and price-limit-aware

Full detail in [`reports/methodology.md`](reports/methodology.md).

## Reports

```text
reports/final_results.md      main findings and tables
reports/methodology.md        data, features, validation, costs, bias controls
reports/final_audit.md        audit output, test suite, known gaps
reports/report_index.md       index of all outputs
reports/model_report.md       model evaluation detail
reports/data_quality_report.md
docs/enhanced_fb_fs_checkpoint.md   foreign-flow experiment and result
```

Dashboard: [nguyenhuygiabao.github.io/vn30-alpha-research/reports/dashboard.html](https://nguyenhuygiabao.github.io/vn30-alpha-research/reports/dashboard.html) (repository copy at `reports/dashboard.html`)

## Usage

Full pipeline, 19 stages, from existing local raw data:

```powershell
py .\scripts\run_full_pipeline.py
```

Rebuild everything, then audit:

```powershell
py .\scripts\run_full_pipeline.py --clean-first --audit-after
```

Reuse expensive model stages only when their outputs are newer than their dependencies:

```powershell
py .\scripts\run_full_pipeline.py --clean-first --keep-expensive-ml --reuse-existing-ml --audit-after
```

Audits and checks:

```powershell
py .\scripts\audit_project_state.py
py .\scripts\audit_fiinquant_core.py --require-data
py .\scripts\audit_execution_coverage.py
py .\scripts\report_summary.py
```

Foreign-flow experiment, separate from the production pipeline and several hours to run:

```powershell
py .\scripts\run_foreign_flow_experiment.py
```

Tests:

```powershell
py -m pytest -q
```

342 tests.

## Paper Trading

A settlement-aware paper-trading layer exists for signal timing, T+2 settlement, order sizing under concentration and liquidity constraints, and local account ledgers. No real orders are placed.

Its live daily-update path still uses vnstock rather than FiinQuant and has not been reconciled against the research pipeline. This does not affect the results above, which run entirely on FiinQuant data.

```powershell
py .\scripts\validate_paper_trading_timing.py
py .\scripts\update_daily_data.py --dry-run
py .\scripts\score_daily_signals.py
py .\scripts\preview_daily_targets.py
```

On Python 3.14, install the vnstock core dependencies without the optional chart package:

```powershell
py -m pip install pydantic openpyxl importlib-metadata tenacity "vnai>=2.4.8"
py -m pip install --no-deps "vnstock==4.0.4"
```

Generated paper-account ledgers are local state and are not tracked.

## Limitations

- Multiple comparisons are not corrected. Three horizons, five ablation configurations, four optimizer modes, two execution modes, and five foreign-flow feature sets were tested against the same history.
- Only the foreign-flow comparison reports confidence intervals. Other metrics are point estimates.
- Cumulative figures sum overlapping periods rather than compounding.
- Daily bars only: no intraday queue priority, order-book depth, or hidden liquidity. Foreign ownership room is not modeled.
- No fundamental, macro, or news data.
- Issuer-group and sector labels for the 30 alumni tickers were compiled manually and should be verified against an authoritative classification source.
- Corporate-action adjustment is verified only to the extent that raw and adjusted files agree on volume and traded value.

## Repository

Generated data is not tracked:

```text
data/processed/
data/raw/
```

## License

MIT. See `LICENSE`.

## Disclaimer

For research and educational purposes only. Not investment advice, and not a deployable trading strategy.
