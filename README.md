# VN30 Alpha Research Framework

Cross-sectional stock selection on the Vietnamese VN30 index. The framework ranks constituents by expected outperformance against the VN30 benchmark, converts rankings into a constrained long-only portfolio, and evaluates the result after transaction costs and Vietnam-specific execution frictions.

Research framework, not a live trading system. Not investment advice.

## Results

Forecast horizon comparison, gradient boosting, price-limit-aware execution:

| Horizon | Rank IC | Top-5 Hit Rate | After-Cost Return / Period | Diagnostic Sharpe | Evaluated Dates |
|---:|---:|---:|---:|---:|---:|
| 1d | 0.044204 | 0.482784 | -0.000244 | -0.031919 | 2428 |
| 5d | 0.033360 | 0.481307 | 0.000076 | 0.003907 | 2418 |
| 10d | 0.037407 | 0.482147 | 0.001120 | 0.039957 | 2403 |

The 5-day and 10-day horizons are after-cost positive; the 1-day horizon is not.

Feature ablation, 5-day horizon:

| Configuration | Features | Rank IC | After-Cost Return / Period |
|---|---:|---:|---:|
| All features | 99 | 0.033360 | 0.000076 |
| Excluding risk | 97 | 0.033263 | -0.000347 |
| Excluding herding | 89 | 0.033124 | 0.000081 |
| Excluding price limit | 84 | 0.029790 | -0.000202 |
| Excluding volume/liquidity | 88 | 0.029133 | -0.000309 |

The full feature set is now the strongest configuration, and two of five clear transaction costs.

Baseline comparison, top-5 selection, before cost, matched concentration and target:

| Selection Rule | Top-5 Active Return / 5d Period |
|---|---:|
| Trailing 10-day return (`top5_momentum`) | 0.002264 |
| elastic_net | 0.001992 |
| ridge | 0.001478 |
| gradient_boosting | 0.001080 |
| random_forest | 0.001030 |

A single-feature momentum rule still outperforms every trained model, though the gap has narrowed: the best model now reaches 88 percent of the baseline, against 62 percent before the training window was widened. `return_10d` is already an input feature to all five models.

Foreign buy/sell features were tested against a decision rule fixed before results were seen (paired block bootstrap on daily Rank IC, 222 non-overlapping signals, promote only if the 95 percent confidence interval excludes zero). Mean difference -0.002608, CI [-0.013314, +0.008297]. Not promoted.

Cumulative return figures in this repository sum overlapping forecast-period active returns and are not compounded portfolio returns. Drawdown is measured on that same summed series, so it is reported in summed-return units rather than as a percentage of capital.

### Deployability

The per-period figures above are means over overlapping forecast periods. A
portfolio cannot hold an overlapping period, so they do not answer what a
portfolio running this strategy would have done.

The strategy produces a signal every day and holds for five, so what it
actually trades is five overlapping tranches: one fifth of capital rebalances
into each day's targets and is held for five days. That construction holds
every phase at once and has no free parameters. Measured over 9.58 years,
after costs, under price-limit-aware execution:

| Optimizer | Strategy equity | VN30 benchmark | Active | Active drawdown |
|---|---:|---:|---:|---:|
| `sector_diversified` | 2.6252 | 2.6918 | 0.9828 | -26.3% |
| `normal` | 2.1042 | 2.6918 | 0.8021 | -37.5% |
| `herding_aware` | 2.1172 | 2.6918 | 0.7704 | -36.9% |
| `sector_aware` | 1.7880 | 2.6918 | 0.6526 | -45.7% |

**No optimizer mode beats buying and holding the index.** Every active equity
is below 1.0: the best configuration turns 1.0 into 2.6252 while the VN30
itself turns it into 2.6918, and it takes a 53 percent peak-to-trough drawdown
along the way. The three other modes end 20 to 35 percent behind the index.

Sampling non-overlapping periods instead leaves the starting offset as a free
parameter, and that choice moves the answer more than the strategy does: final
equity spreads by 0.48 to 1.06 across the five offsets, against 0.023 to 0.030
for the tranche portfolio — a factor of 20 to 45. One of those offsets makes
`sector_diversified` look positive throughout. That is a property of the
sampling, not of the strategy, which is why the tranche construction is the
one reported here.

This is a statement about the configurations measured here, not about the VN30
market. Features are fitted in raw levels while the target is demeaned against
the index, and no within-date standardisation is applied; whether that
specification handicaps the models is untested.

Sources: `reports/tables/tranche_portfolio_diagnostic.csv` from
`scripts/diagnose_tranche_drawdown.py`, and
`reports/tables/compounded_phase_sensitivity.csv` from
`scripts/diagnose_compounded_returns.py`.

## Data

FiinQuant daily data, 137,719 rows across 60 tickers, 2016-07-26 to 2026-07-22. After point-in-time universe filtering and lag construction, 74,910 rows reach the modeling stage.

```text
data/raw/fiinquant/canonical/raw_market_history.csv
data/raw/fiinquant/canonical/adjusted_market_history.csv
data/raw/fiinquant/canonical/foreign_flow_history.csv
```

Raw prices are used for price limits and execution reference; adjusted close is used for returns and labels. The loader enforces the separation.

The universe is point-in-time VN30 membership reconstructed from verified index-review history in `config/vn30_history/`, covering 60 tickers across the full period (current 30 constituents plus 30 former constituents).

FLC and ROS raw OHLC was backfilled from a vendor archive. Traded value for those two tickers is unavailable from any source; 1,013 rows use a typical-price estimate flagged per row by `value_traded_is_estimated`. Execution-data coverage is otherwise complete: 73,897 of 74,910 constituent-dates have full real raw OHLC, volume, and traded value, with zero missing rows.

## Method

Full detail in [`reports/methodology.md`](reports/methodology.md).

### Labels

Forward return relative to the VN30 benchmark at 1, 5, and 10 trading days.
The production target is `forward_relative_return_5d`. Because the label is
demeaned against the index, the market-wide component of any feature is
orthogonal to it by construction.

### Features

99 model-facing columns. Group sizes below are measured as the drop in feature
count when each group is ablated, so they are reproducible from
`reports/tables/ablation_results.csv`. One column belongs to two groups, so the
groups do not sum to 99.

| Group | Columns | Contents |
|---|---:|---|
| Foreign flow | 46 | Foreign buy, sell and net value, their rolling means and z-scores, cross-sectional rank percentiles, gross participation share, and buy-on-down-day indicators |
| Momentum and return | 16 | 1, 3, 5, 10, 20 and 60-day returns, rolling returns, log and simple daily return, distance from 20-day high and low, positive and negative shock flags |
| Price limit | 15 | Estimated ceiling and floor from HOSE tick rules, distance to each, hit and close-at-limit flags, consecutive ceiling and floor day counts, market-wide limit pressure |
| Volume and liquidity | 11 | Volume z-scores, abnormal volume flags, volume change, Amihud illiquidity, traded-value rank, turnover measures |
| Herding and market state | 10 | Cross-sectional correlation score, direction agreement, return dispersion, percent of constituents up, down, and at limits, top-5 volume concentration, rolling average pairwise correlation |
| Risk | 2 | Realised volatility measures |

Features are standardised by a scaler fitted over the pooled training window.
No standardisation is applied within a date.

### Models

Five models are fitted independently on every walk-forward window. Gradient
boosting feeds portfolio construction.

| Model | Configuration |
|---|---|
| Gradient boosting *(production)* | 100 estimators, learning rate 0.05, max depth 2, min samples per leaf 2 |
| Random forest | 100 estimators, max depth 3, min samples per leaf 2 |
| Ridge | alpha 1.0 |
| Elastic net | alpha 0.001, L1 ratio 0.5 |
| Gradient-boosting classifier | Direction classification on the same features |

Elastic net currently beats gradient boosting on every diagnostic measured,
while gradient boosting remains the production model. That inconsistency is
recorded rather than resolved.

### Validation

Walk-forward over roughly 2,480 windows, each fitting on `DEFAULT_TRAIN_SIZE`
= 63 trading dates and predicting the next, with a 5-day purge at every
boundary so no training row overlaps the label window of a test row. The
window size was chosen by a recorded sweep over {3, 21, 63, 126, 252} scored on
an identical set of evaluation dates
(`reports/tables/training_window_diagnostic.csv`).

Parallel and sequential walk-forward output are asserted to be exactly equal,
so process-based fitting is a speed change only.

### Portfolio construction

Long-only, top-5 selection by model score, in four optimizer modes: `normal`,
`herding_aware`, `sector_aware`, and `sector_diversified`. Constraints are a
20 percent single-name cap, 40 percent turnover cap, 40 percent issuer-group
cap, and 35 percent sector cap.

### Costs and execution

10 bps commission, 10 bps slippage, 10 bps sell tax, a liquidity penalty, a
5 percent ADV participation cap, 100-share board lots, and T+2 settlement.
Two execution modes are reported: frictionless, and price-limit-aware, which
blocks buys at the ceiling and sells at the floor.

### Invariants enforced by tests

- Appending future rows leaves every already-computed feature bit-identical
- Raw prices drive price limits and execution; adjusted prices drive returns
  and labels, and mixing them raises
- Point-in-time VN30 membership, never today's constituents applied backwards
- The four published foreign-flow feature sets stay frozen at 51, 60, 64 and 73
  features
- No infinite or duplicate feature values reach the modeling stage

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

Report page: [nguyenhuygiabao.github.io/vn30-alpha-research](https://nguyenhuygiabao.github.io/vn30-alpha-research/) (repository copy at `reports/reportpage.html`, regenerated from `reports/tables/` by `scripts/build_report_page.py` on every pipeline run)

## Usage

Full pipeline, 18 stages, from existing local raw data:

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

409 tests.

## Paper Trading

A settlement-aware paper-trading layer exists for signal timing, T+2 settlement, order sizing under concentration and liquidity constraints, and local account ledgers. No real orders are placed.

The live daily-update path runs on FiinQuant, the same vendor as the research pipeline. It downloads the current index members for the trailing overlap window, checks the overlap against the local file before replacing anything, and writes through a staging file.

```powershell
py .\scripts\validate_paper_trading_timing.py
py .\scripts\update_daily_data.py --dry-run
py .\scripts\score_daily_signals.py
py .\scripts\preview_daily_targets.py
```

The daily update needs the FiinQuantX vendor wheel, which lives in its own
environment. `run_daily_data_update.py` finds that interpreter automatically:

```powershell
py -m venv .venv-fiinquant
.\.venv-fiinquant\Scripts\python.exe -m pip install FiinQuantX pandas pyarrow PyYAML
```

Credentials are read from `FIINQUANT_USERNAME` and `FIINQUANT_PASSWORD`, and
prompted for interactively when either is unset.

The paper-trading price file can be rebuilt from the canonical FiinQuant
history at any time:

```powershell
py .\scripts\build_paper_trading_ohlcv.py --dry-run
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
