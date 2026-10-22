# VN30 Alpha Research Framework

Cross-sectional stock selection on the Vietnamese VN30 index. The framework ranks constituents by expected outperformance against the VN30 benchmark, converts rankings into a constrained long-only portfolio, and evaluates the result after transaction costs and Vietnam-specific execution frictions.

Research framework, not a live trading system. Not investment advice.

## Results

> **Partly stale as of 2026-09-05.** The cost model was corrected in commit
> `bfdabcf` — commission and slippage now charge both legs of a trade and the
> statutory transfer tax is applied to sales, so a round trip costs 0.5 percent
> rather than 0.2. Rank IC and hit rate do not depend on the cost model and are
> unaffected anywhere.
>
> | Table | State |
> |---|---|
> | Deployability | **Regenerated**, correct |
> | Feature ablation | **Regenerated**, correct |
> | Horizon comparison *(immediately below)* | **Regenerated 2026-10-21**, correct |
> | Benchmark / portfolio comparison | **Regenerated 2026-10-21**, correct |
>
> Every table in this README is now on the corrected cost model. Rank IC and
> hit rate do not depend on the cost model and never moved.

Forecast horizon comparison, gradient boosting, price-limit-aware execution.
**Regenerated 2026-10-21 under the corrected cost model:**

| Horizon | Rank IC | Top-5 Hit Rate | After-Cost Return / Period | Diagnostic Sharpe | Evaluated Dates |
|---:|---:|---:|---:|---:|---:|
| 1d | 0.044204 | 0.482784 | -0.001420 | -0.185495 | 2428 |
| 5d | 0.033360 | 0.481307 | -0.000822 | -0.042290 | 2418 |
| 10d | 0.037407 | 0.482147 | 0.000351 | 0.012532 | 2403 |

The 10-day horizon is after-cost positive; the 1-day and 5-day horizons are not.

Feature ablation, 5-day horizon. **Regenerated 2026-09-05 under the corrected
cost model** (`bfdabcf`):

| Configuration | Features | Rank IC | After-Cost Return / Period |
|---|---:|---:|---:|
| All features | 99 | 0.033360 | **-0.000822** |
| Excluding risk | 97 | 0.033263 | -0.001244 |
| Excluding herding | 89 | 0.033124 | -0.000818 |
| Excluding price limit | 84 | 0.029790 | -0.001127 |
| Excluding volume/liquidity | 88 | 0.029133 | -0.001224 |

The full feature set is still the strongest configuration, and **none of the
five clears transaction costs.** Before the cost correction two of five appeared
to; that was an artifact of charging a round trip 0.2 percent instead of 0.5.

The same correction moves the portfolio comparison: 2 of
8 optimizer/execution configurations are after-cost positive per
period, where the pre-correction tables showed 8 of 8.

Every row of `benchmark_results.csv` now carries a 95 percent moving-block bootstrap interval on its mean after-cost active return (2,000 resamples, blocked at the 5-day horizon so the overlap between consecutive period returns stays inside a block). The best configuration, `sector_diversified_normal`, has mean 0.000225 with interval [-0.000624, 0.001085], which spans zero. The most conservative, `normal` + `price_limit_aware`, has mean -0.000822 with interval [-0.002056, 0.000462], which spans zero. This replaces the earlier description of a near-zero Sharpe as "statistically indistinguishable from zero", which was an assertion with no test behind it.

Rank IC is unchanged, as expected — it does not depend on the cost model. Only
the after-cost columns moved.

Baseline comparison, top-5 selection, before cost, matched concentration and target:

| Selection Rule | Top-5 Active Return / 5d Period | Evaluated Dates |
|---|---:|---:|
| Trailing 10-day return (`top5_momentum`) | 0.002264 | 2418 |
| elastic_net | 0.001992 | 2418 |
| classification | 0.001811 | 2418 |
| ridge | 0.001478 | 2418 |
| random_forest | 0.001076 | 2418 |
| gradient_boosting | 0.001073 | 2418 |

A single-feature momentum rule still outperforms every trained model, though
the gap has narrowed. `return_10d` is already an input feature to all five
models.

**The 62 and 88 percent figures are the best model at each stage, and that is
not the same model at each stage.** Before the training window was widened the
leader was `random_forest` at 0.001433 against a 0.002310 baseline, which is 62
percent. After widening it is `elastic_net` at 0.001992 against 0.002264, which
is 88 percent. Like for like, `elastic_net` itself went from 58 percent to 88
percent. Both readings are reported so neither can be mistaken for the other.

**The momentum baseline moves from 0.002310 to 0.002264 even though the rule
never changed.** It is a fixed, untrained rule reported before cost, so neither
the training window nor the cost model can touch it. What changed is the sample:
a walk-forward window cannot predict until it has `train_size` dates behind it,
so going from 3 to 63 dates consumes 60 more dates of warm-up. The baseline is a
mean over 2,478 dates before and 2,418 after — 2,478 − 2,418 = 60 = 63 − 3. The
dates left the sample; the rule did not move. Every before/after pair in this
README has that same 60-date difference between its two columns.

Foreign buy/sell features were tested against a decision rule fixed before results were seen (paired block bootstrap on daily Rank IC, 222 non-overlapping signals, promote only if the 95 percent confidence interval excludes zero). Mean difference -0.002608, CI [-0.013314, +0.008297]. Not promoted.

Cumulative return figures in the per-period tables above sum overlapping forecast-period active returns and are not compounded portfolio returns. Drawdown in those tables is measured on that same summed series, so it is reported in summed-return units rather than as a percentage of capital. The Deployability section below is the exception and states its own convention: there, drawdown is a peak-to-trough fall of a compounded equity curve and is a genuine percentage of capital.

### Deployability

The per-period figures above are means over overlapping forecast periods. A
portfolio cannot hold an overlapping period, so they do not answer what a
portfolio running this strategy would have done.

The strategy produces a signal every day and holds for five, so what it
actually trades is five overlapping tranches: one fifth of capital rebalances
into each day's targets and is held for five days. That construction holds
every phase at once and has no free parameters. Measured over 2,414 trading
days, 2016-11-15 to 2026-07-16 — which is 9.58 years at 252 trading days per
year, not 9.67 calendar years — after costs, under price-limit-aware execution:

| Optimizer | Strategy equity | VN30 benchmark | Active (compounded) | Equity ratio | Strategy drawdown | Active drawdown |
|---|---:|---:|---:|---:|---:|---:|
| `sector_diversified` | 1.4850 | 2.6918 | 0.5557 | 0.5517 | -58.3% | -50.3% |
| `normal` | 1.1699 | 2.6918 | 0.4458 | 0.4346 | -63.5% | -61.7% |
| `herding_aware` | 1.1945 | 2.6918 | 0.4345 | 0.4437 | -57.7% | -61.4% |
| `sector_aware` | 1.0105 | 2.6918 | 0.3687 | 0.3754 | -62.1% | -66.1% |

**No optimizer mode beats buying and holding the index.** Every active equity
is below 1.0: the best configuration turns 1.0 into 1.4850 while the VN30
itself turns it into 2.6918, and takes a 58 percent peak-to-trough fall in
capital on the way. The three other modes end 55 to 63 percent behind the index
on compounded active return.

**`Active` is not `Strategy equity / VN30 benchmark`.** It is the compounded
daily active return, the product over trading days of 1 + (strategy return −
benchmark return), which is what a daily-rebalanced long-strategy/short-index
sleeve would have returned. The equity ratio is published beside it because the
two differ — compounding a difference of returns is not the same as taking a
ratio of compounded returns, and the gap is the covariance of the two legs, not
rounding.

**The 53 percent and -26.3 percent drawdowns previously published side by side
are two different curves, not two readings of one.** 53.1 percent was the
peak-to-trough fall of `sector_diversified`'s own compounded equity curve —
capital down 53.1 percent from its high-water mark — while -26.3 percent was
the peak-to-trough fall of the compounded *active* curve, the long-strategy /
short-index sleeve. Both are percentages of capital and both are compounded;
they differ because a portfolio and a hedged version of that portfolio draw
down at different times and by different amounts. Under the corrected cost
model the same pair reads -58.3 percent and -50.3 percent. The strategy
drawdown column answers "how far did the account fall", the active drawdown
column answers "how far did the bet against the index fall", and neither
substitutes for the other.

**A third, non-comparable drawdown convention appears in the per-period
tables.** `max_active_drawdown` in `benchmark_results.csv` is a running-peak
fall on `cumsum` of overlapping five-day forward active returns. It is in
summed-return units, counts every period about five times over, and is not a
percentage of capital at all — the conservative configuration's -3.30 there is
not a 330 percent loss. No number is carried between that convention and the
two compounded columns above.

These figures were regenerated on 2026-09-05 after a correction to the cost
model, and they are substantially worse than the ones published before that
date — `sector_diversified` previously read a strategy equity of 2.6252, an
active 0.9828, and a 53.1 percent strategy drawdown.
Nothing about the strategy changed. What changed is that it is now charged what
it would actually pay.

`portfolio_turnover` is one-way, so a rebalance of `t` sells `t` and buys `t`
and trades `2t` of notional. The backtester applied commission and slippage to
`t` rather than `2t`, and omitted Vietnam's statutory 0.1 percent transfer tax
on sale proceeds altogether. A round trip was therefore priced at 0.2 percent
while `config/paper_trading_config.yaml` charged 0.5 percent for the same
trade — visible in `paper_trading/executions.csv`, where a 5,900,000 VND buy
carries commission 5,900 and slippage 5,900, both per side. The live harness was
right and the backtest was not. See commit `bfdabcf`.

The earlier figures understated cost by a factor of 2.5 and should not be cited.

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

Three different ticker counts appear in this repository and all three are
correct:

| Count | What it counts | Where it appears |
|---:|---|---|
| 30 | Index constituents on any single date | Every per-date figure, portfolio construction |
| 54 | Distinct tickers inside `paper_horizon_predictions.parquet` | The paper-horizon section of the report page |
| 60 | Distinct tickers in the modeling dataset over the full history | This section, and the 74,910-row dataset |

54 rather than 60 in the paper-horizon file is correct rather than a gap: six
former constituents (FLC, HAG, HCM, HNG, ITA, PPC) left the index before
2017-08-09, which is the first date that file can make a prediction after its
252-date training warm-up, so under point-in-time membership they never appear
in it. That file also stacks two models in one table with a `model_name`
column, so a fully covered date carries 30 constituents x 2 models = 60 rows
and its 208 dates carry 12,480 rows. The repeated (date, ticker) pairs are the
two models, not duplicate rows.

Ten of the 2,492 modeling dates carry 29 constituents rather than 30, because
FLC has no data for the week of 2017-01-16 and ROS none for the week of
2021-01-25. This is why `approximate_train_rows` in
`reports/tables/training_window_diagnostic.csv` is a measured count rather than
the window length times 30: the 126-date window measures 3,779 rows against an
arithmetic 3,780, and the 252-date window 7,555 against 7,560.

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

The sweep reports Rank IC 0.028126 for the chosen 63-date window while the
production figure is 0.033360. Both are gradient boosting on the 5-day horizon
and neither is wrong: the sweep scores 223 test dates, subsampled at every
tenth date and starting late enough that the 252-date window is also reachable,
so that all five configurations are scored on identical dates. Production
walk-forward scores every date it can, 2,418 of them. A shorter, later, sparser
subsample is a different sample of the same estimator, not a different method.

Parallel and sequential walk-forward output are asserted to be exactly equal,
so process-based fitting is a speed change only.

### Portfolio construction

Long-only, top-5 selection by model score, in four optimizer modes: `normal`,
`herding_aware`, `sector_aware`, and `sector_diversified`. Constraints are a
20 percent single-name cap, 40 percent turnover cap, 40 percent issuer-group
cap, and 35 percent sector cap.

The single-name cap and the issuer-group cap are separate constraints, which
matters when reading the concentration diagnostics: on 2026-07-15 the `normal`
optimizer holds a largest single position of 20.00 percent, at the single-name
cap, and a Vingroup issuer group of 20.1 percent spread across VHM, VIC and
VPL. The 20.1 percent figure is governed by the 40 percent issuer-group cap,
not by the 20 percent single-name cap, and no constituent of that group exceeds
20 percent on its own. Neither constraint is breached;
`issuer_groups_at_or_above_40pct` is 0 for every optimizer mode on that date.

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
- Rank IC, hit rate, and the per-horizon and per-ablation figures are point estimates with no interval. Intervals are reported for the foreign-flow comparison (paired block bootstrap) and, since 2026-10-21, for the mean after-cost active return of every row of `benchmark_results.csv` (95 percent moving-block bootstrap, 2,000 resamples, blocked at the 5-day horizon so the overlap between consecutive period returns stays inside a block).
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
