# Methodology

## 1. Objective

This project tests whether cross-sectional stock selection inside the VN30 universe can be turned into a constrained long-only portfolio that survives transaction costs and Vietnam-specific execution frictions.

For each trading date the model ranks the active VN30 constituents by expected outperformance against the VN30 benchmark over a forward horizon. Those rankings are converted into portfolio weights under long-only, turnover, single-name, issuer-group, and sector constraints, then backtested with commission, slippage, a liquidity penalty, and price-limit execution rules.

This is a research pipeline, not a trading strategy. Several of the results reported below are negative.

## 2. Investment Universe

The universe is **point-in-time VN30 membership**, not the current constituent list applied backward.

Membership is reconstructed from four verified source files in `config/vn30_history/`:

- `initial_snapshot.csv` — constituents as of 2016-07-25
- `review_calendar.csv` — every index review date
- `membership_changes.csv` — additions and removals at each review
- `source_manifest.csv` — the source URL backing each review

`src/universe_reconstruction.py` replays these into per-ticker `(effective_from, effective_to)` intervals, asserting that every review date resolves to exactly 30 constituents. `src/universe_history.py` then filters market data so a ticker only contributes rows on dates when it was actually in the index.

Across the full history this pool contains **60 tickers**: the current 30 plus 30 former constituents such as FLC, ROS, HAG, HNG, ITA, and PVD. Including delisted and demoted names is what removes survivorship bias — earlier versions of this project used a static 30-ticker universe and were biased as a result.

`config/vn30_universe.csv` carries issuer-group and sector metadata for all 60 tickers. Issuer groups deliberately link affiliated companies so concentration caps apply to shared economic exposure rather than per-ticker: FLC and ROS share a controlling shareholder, HAG holds a controlling stake in HNG, POW holds roughly 59 percent of NT2, and the Vingroup names (VHM, VIC, VPL, VRE) are capped together.

## 3. Market Data

Data comes from **FiinQuant**, stored as three canonical CSV files:

```text
data/raw/fiinquant/canonical/raw_market_history.csv
data/raw/fiinquant/canonical/adjusted_market_history.csv
data/raw/fiinquant/canonical/foreign_flow_history.csv
```

The dataset holds **137,719 daily rows across 60 tickers from 2016-07-26 to 2026-07-15**. After point-in-time universe filtering and lag construction, **74,910 rows** reach the modeling stage.

### Raw versus adjusted price contract

The two price files are not interchangeable, and `src/market_data_schema.py` enforces the distinction:

- **Raw prices** are used for anything that depends on the price actually printed on the exchange: price-limit bands, ceiling and floor detection, and execution reference prices. Adjusted prices would silently break these, because a split-adjusted close does not sit at the historical ceiling.
- **Adjusted close** is used for returns, labels, and any return-derived feature, so corporate actions do not appear as spurious jumps.

The loader cross-checks that raw and adjusted rows agree on volume and traded value, and raises an error on mismatch rather than reconciling the two.

### FLC and ROS coverage

FiinQuant's own raw-price history does not reach back over FLC and ROS. Raw OHLC for those two tickers was backfilled from a vendor archive export (`scripts/import_fiinquant_archived_history.py`, `scripts/backfill_flc_ros_raw_archive.py`), which brought raw-price fallback rows to zero.

Traded value for those tickers is not available from any source. Rather than zero-fill it, the loader leaves `value_traded` genuinely missing and publishes a separate typical-price estimate — `(high + low + close) / 3 × volume` — in `value_traded_estimated`, with `value_traded_is_estimated` marking exactly which rows are affected (1,013 of 74,910). Features that need a liquidity denominator opt into the estimate explicitly, so estimated and vendor-reported values remain distinguishable downstream.

`scripts/audit_execution_coverage.py` verifies this end to end: for every active constituent on every trading date, it checks that raw OHLC, volume, and traded value are actually present. The current result is 73,897 rows with complete real data, 1,013 rows on the flagged estimate, and zero rows with missing prices, missing volume, or missing rows entirely.

## 4. Labels

Labels are forward returns relative to the VN30 benchmark over the same window, at 1-day, 5-day, and 10-day horizons. The primary target is `forward_relative_return_5d`; the horizon test in section 10 compares all three.

Labels are built from the same point-in-time membership and adjusted-price history as the features. An earlier version of the pipeline built labels from a static 30-ticker file while features came from the 60-ticker point-in-time history, so the feature/label merge silently dropped every former-constituent row and made the membership filtering a no-op. The merge is now 1:1 with zero unmatched rows.

Features use only information available strictly before the prediction date. Forward labels are never fed back in as inputs.

## 5. Features

`data/processed/features_combined.parquet` holds 109 columns, of which 99 are model-facing features after keys, labels, and raw-price columns are excluded. They fall into six groups:

**Momentum and return** — simple and log returns, rolling returns over several windows, medium-term momentum, distance from rolling highs and lows, drawdown.

**Risk** — 20-day rolling volatility and drawdown.

**Liquidity** — average daily volume and traded value, traded-value rank, volume z-score, abnormal-volume flags, and Amihud illiquidity. These matter in Vietnam because a signal that only fires on illiquid names is not tradable.

**Price limit** — estimated ceiling and floor prices, distance to each band, whether a stock hit or closed at a limit, consecutive limit days, and market-wide limit pressure. This is the group the ablation test finds most load-bearing.

**Herding** — cross-sectional return dispersion, share of stocks moving in the same direction, rolling average pairwise correlation, and a composite herding index.

**Foreign flow** — foreign buy and sell value from FiinQuant, expressed as net and gross shares of traded value across several windows, plus persistence, z-scores, and cross-sectional ranks. A second-generation set of buy/sell contrast features was also built and tested; see section 11 for why it is not promoted.

## 6. Models

Five models are run through identical walk-forward machinery: ridge and elastic-net regression, random forest, gradient boosting, and a gradient-boosting classifier over a binary outperformance label.

Gradient boosting (`n_estimators=100`, `max_depth=2`) is the model carried into portfolio construction, for continuity with earlier versions of the project rather than because it dominates. On the current data it does not: its Rank IC is 0.0334 against elastic net at 0.0367 and ridge at 0.0361, and elastic net also leads on hit rate, top-5 return, and diagnostic Sharpe. Reconsidering the production model is a live question, not a formality.

Models are evaluated as rankers, since portfolio construction only uses predicted values to sort candidates.

## 7. Walk-Forward Validation

Validation is walk-forward, never a random split. Random splitting on time-ordered market data leaks future regimes into training and inflates every metric.

Each window trains on **63 trading dates**, validates on 1, and tests on 1, stepping forward one date at a time, which yields about 2,420 windows over the history. A **5-day purge** is applied at each train/validation and validation/test boundary.

`train_size` counts dates, not blocks, so with roughly 30 active constituents each window fits on about **1,890 rows** against 99 features. This was previously 3 dates, or about 90 rows, a value duplicated across five model modules that appears never to have been revisited as the dataset grew. It is now the single constant `DEFAULT_TRAIN_SIZE` in `src/walk_forward_split.py`.

The value was selected by sweeping {3, 21, 63, 126, 252} against a shared set of 223 evaluation dates (`scripts/diagnose_training_window.py`). The relationship is an inverted U rather than monotonic: Rank IC peaks at 21 dates and falls to roughly zero at 252, which is consistent with long windows averaging across market regimes that no longer apply. 63 was chosen over the Rank-IC-optimal 21 because it had the best top-5 hit rate and the only clearly positive top-5 return, and portfolio construction selects a top-N basket rather than trading the full ranking.

Widening the window raised Rank IC at 1-day and 5-day horizons, moved the 5-day horizon from net-negative to net-positive after cost, and reversed the earlier ablation finding that removing features helped. Section 11 of `reports/final_results.md` carries the full before/after comparison.

The purge exists because 5-day and 10-day forward labels overlap adjacent dates. Without it, a training row's label window extends into the test period and the model is partly scored on information it already saw. Adding purging is what collapsed this project's earlier headline results, and those earlier numbers should be treated as leakage-inflated.

Model fitting across windows is parallelized with process-based `joblib` workers. Output is asserted identical to the sequential path, so parallelism is a speed change only.

## 8. Portfolio Construction

`src/optimizer.py` turns predicted rankings into weights under:

- long-only, no shorting or leverage
- 20 percent maximum single-name weight
- 40 percent maximum portfolio turnover per rebalance
- 40 percent maximum issuer-group weight
- 35 percent maximum sector weight

Four optimizer modes are compared:

- `normal` — constrained long-only on the raw ranking
- `herding_aware` — reduced exposure and tighter caps during high-herding states
- `sector_aware` — adds the sector cap
- `sector_diversified` — widest diversification, roughly 26 names, with a 25 percent issuer-group cap

The optimizer refuses to run if any prediction row lacks issuer-group or sector metadata rather than dropping it, which is how the 60-ticker metadata gap in `config/vn30_universe.csv` was caught.

## 9. Costs and Execution Frictions

The backtester applies commission (10 bps), slippage (10 bps), a sell-side tax (10 bps), and a liquidity penalty scaled to how large an order is against 20-day average daily value. Trades are capped at 5 percent of ADV and rounded to 100-share lots. Settlement is T+2, so sale proceeds are not immediately available for reinvestment.

Two execution modes are compared:

- `normal` — assumes intended trades fill
- `price_limit_aware` — blocks buys into a ceiling close and sells into a floor close, then recomputes feasible trades

Price-limit-aware execution is consistently worse than normal execution. The gap between the two measures how much of the result depends on fills that HOSE's daily bands would not have allowed.

## 10. Robustness Tests

### Feature ablation

Five configurations, removing one feature group at a time, all under price-limit-aware execution:

| Configuration | Features | Rank IC | Top-5 Hit Rate | After-Cost Return | Cumulative After-Cost |
|---|---:|---:|---:|---:|---:|
| All features | 99 | 0.033360 | 0.481307 | 0.000076 | 0.184 |
| Excluding risk | 97 | 0.033263 | 0.481472 | -0.000347 | -0.840 |
| Excluding herding | 89 | 0.033124 | 0.482299 | 0.000081 | 0.195 |
| Excluding price limit | 84 | 0.029790 | 0.477916 | -0.000202 | -0.489 |
| Excluding volume/liquidity | 88 | 0.029133 | 0.476427 | -0.000309 | -0.747 |

The full feature set is now the strongest configuration and two of five clear transaction costs. This reverses the earlier reading: under a 3-date training window, removing features improved Rank IC and `without_risk` led, which suggested the feature set was largely noise. With 63 dates the ranking inverts. The earlier result reflected too little training data to use 99 features, not harmful features.

Removing price-limit or volume/liquidity features is now clearly worse on both Rank IC and after-cost return. `without_herding` is essentially tied with the full set, so herding features remain unproven as an alpha source, consistent with their original purpose as a portfolio risk control.

These numbers come from a fresh run. A stale-cache bug in `src/ablation_tests.py` previously served three-week-old predictions from a 51-feature snapshot, and the conclusions drawn from that run were wrong. The internal cache has been removed.

### Forecast horizon

| Horizon | Rank IC | Top-5 Hit Rate | After-Cost Return | Diagnostic Sharpe | Cumulative After-Cost | Dates |
|---:|---:|---:|---:|---:|---:|---:|
| 1d | 0.044204 | 0.482784 | -0.000244 | -0.031919 | -0.593 | 2428 |
| 5d | 0.033360 | 0.481307 | 0.000076 | 0.003907 | 0.184 | 2418 |
| 10d | 0.037407 | 0.482147 | 0.001120 | 0.039957 | 2.691 | 2403 |

The 10-day horizon remains strongest after cost, and the 5-day horizon has crossed from negative to marginally positive. The 1-day horizon now has the highest Rank IC of the three yet is still the only net-negative one, which locates its failure in rebalancing cost rather than in prediction.

### Baseline comparison

`reports/tables/benchmark_results.csv` compares the ML portfolios against naive rules:

| Strategy | Active Return per 5d Period | Basis | Names Held |
|---|---:|---|---:|
| top5_momentum | 0.002264 | before cost | 5 |
| ML sector_diversified / normal | 0.000930 | after cost | 12.9 |
| ML sector_diversified / price_limit_aware | 0.000696 | after cost | 12.9 |
| ML normal / price_limit_aware | 0.000076 | after cost | 6.3 |

`top5_momentum` sorts on a single feature — trailing 10-day return — and holds the top five. It still beats every ML portfolio variant, though all eight ML variants are now positive after cost, where five of eight were negative under the previous training window.

The cost basis differs (baselines are before cost, ML is after), so this is not a like-for-like table. Recomputing each model's own top-5 pick on a matched basis — same concentration, same relative-to-VN30 target, before any cost or portfolio construction — gives:

| Selection Rule | Top-5 Active Return, Before Cost |
|---|---:|
| top5_momentum (`return_10d` only) | 0.002264 |
| elastic_net top-5 | 0.001992 |
| ridge top-5 | 0.001478 |
| gradient_boosting top-5 | 0.001080 |
| random_forest top-5 | 0.001030 |

Every trained model still picks worse top-5 baskets than a single trailing-return sort, and `return_10d` is already one of their input features. The margin narrowed from 1.6x to 1.14x when the training window was widened, so part of the original gap was undertraining rather than a property of the signal. The remainder is unresolved.

## 11. Foreign Flow Experiment

Foreign buy/sell (FB/FS) features were tested against a decision rule fixed **before** results were seen: paired block bootstrap on daily Rank IC over 222 non-overlapping 10-day signals, promote only if the 95 percent confidence interval excludes zero.

Result: mean Rank-IC difference **-0.002608**, 95 percent CI **[-0.013314, +0.008297]**. The interval spans zero and the point estimate is negative. FB/FS is **not promoted**.

The full ranking is informative:

| Feature Set | Rank IC |
|---|---:|
| Frozen foreign flow (net-style only) | 0.063616 |
| Frozen plus original buy/sell | 0.062448 |
| Frozen plus enhanced buy/sell | 0.061007 |
| Market baseline (no foreign flow) | 0.055389 |
| Buy/sell only | 0.053736 |

Net-style foreign-flow features are genuinely useful — they beat the market baseline by 0.0082. Every set that adds a buy/sell split sits below the net-only set, and stacking more buy/sell features makes it progressively worse.

The reason is not that the model ignores them. The two strongest new features rank second and third in the model's own importance output. It is that their relationship with forward returns is not sign-stable: the year-by-year Rank-IC difference is positive in six years and negative in four, the per-signal win rate is 50.5 percent, and the day-to-day standard deviation of the difference is about 29 times its mean. At 222 signals there is not enough power to separate the true effect from zero in either direction. Details are in `docs/enhanced_fb_fs_checkpoint.md`.

## 12. Bias Controls

- Point-in-time VN30 membership, so delisted and demoted names are included on the dates they were actually investable.
- Walk-forward validation with horizon-aware purging at every boundary.
- Past-only feature construction, with a leakage test asserting that appending future rows leaves already-computed feature values bit-identical.
- Raw versus adjusted price separation enforced at the loader.
- Transaction costs, liquidity penalties, ADV caps, lot sizes, and T+2 settlement in the backtest.
- Price-limit-aware execution reported alongside frictionless execution rather than instead of it.
- Estimated traded value flagged per row, never silently substituted.
- Pre-registered decision rule and paired bootstrap confidence intervals for the foreign-flow comparison.

## 13. Limitations

**Cumulative returns are sums, not compounded.** Every cumulative figure in this project sums overlapping forecast-period active returns. At the 10-day horizon, 2,463 evaluated dates correspond to roughly 246 genuinely non-overlapping periods. Read cumulative figures as diagnostics, not portfolio track records. Drawdown is computed on that same summed series and is therefore also in summed-return units, not a percentage of capital.

**Multiple comparisons are not corrected.** This project has tested three horizons, five ablation configurations, four optimizer modes, two execution modes, and five foreign-flow feature sets against the same history. Only the foreign-flow comparison carries a confidence interval, and even that one does not adjust for being the fifth feature set on the same development data. The best-looking number in this repository is partly a product of search.

**Most metrics are point estimates.** Rank IC, diagnostic Sharpe, drawdown, and return figures outside the foreign-flow experiment have no confidence intervals. Extending bootstrap intervals to the horizon and ablation results would materially strengthen or weaken those conclusions.

**Daily data only.** No intraday queue priority, order-book depth, or hidden liquidity. Foreign ownership room is not modeled.

**No fundamental, macro, or news data.** Features are derived from price, volume, and foreign flow.

**Paper trading runs on a different data source.** The live daily-update path still uses vnstock rather than FiinQuant, and has not been reconciled against the research pipeline.

**Corporate-action adjustment quality has not been fully audited** beyond the raw/adjusted consistency checks in the loader.

## 14. Summary

Pipeline components: point-in-time VN30 universe, enforced raw/adjusted price contract, purged walk-forward validation, constrained long-only optimization, transaction costs and price-limit execution, feature-ablation and forecast-horizon tests, and a pre-registered bootstrap comparison for the foreign-flow feature sets.

Current results: only the 10-day horizon is after-cost positive; no ablation configuration clears transaction costs at the 5-day horizon; a single-feature trailing-return rule outperforms every trained model on a matched basis; foreign buy/sell features did not clear their pre-registered confidence-interval criterion.
