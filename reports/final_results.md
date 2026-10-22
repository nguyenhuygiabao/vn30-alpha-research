# Final Results

## 1. Executive Summary

This project ranks VN30 stocks by expected outperformance against the VN30 benchmark, converts those rankings into a constrained long-only portfolio, and measures what survives transaction costs and Vietnam-specific execution frictions.


The main finding remains negative:

**A single-feature momentum rule still beats every trained model.** Sorting on trailing 10-day return and holding the top five names returns 0.002264 per 5-day period before cost. The best trained model's top-five basket returns 0.001992 on the same basis. `return_10d` is already an input feature to every model, so the models have strictly more information and still use it worse.

Beyond that:

- Of the three tested horizons, only the **10-day horizon** is after-cost positive, at 0.000351 per period. The 1-day and 5-day horizons are not.
- **None of five feature-ablation configurations** clears transaction costs. The full feature set is the strongest on Rank IC, which is a separate claim from paying for itself.
- **Foreign buy/sell features failed a pre-registered test** and are not promoted; the 95 percent interval spans zero.
- **2 of the eight ML portfolio variants** have positive after-cost active return per period, and **all eight have a 95 percent bootstrap interval that spans zero.** The `top5_momentum` baseline is the only row in `benchmark_results.csv` whose interval excludes zero, and it is measured before cost.

These figures reflect a walk-forward training window of 63 trading dates, which fits 99 features on roughly 1,890 rows per window. Section 11 records the sweep behind that choice.

Pipeline components: point-in-time VN30 membership, an enforced raw/adjusted price contract, purged walk-forward validation, modeled transaction costs and price limits, and a pre-registered bootstrap test for the foreign-flow feature sets.

Every cumulative figure below sums overlapping forecast-period active returns. It is not a compounded portfolio return and should not be read as a track record. Max active drawdown is measured on that same summed series, so it is reported as a plain number in summed-return units, not as a percentage of capital.

Run as the overlapping-tranche portfolio it would actually trade, no optimizer
mode beats buying and holding the index after costs. Section 12 gives the
figures.

## 2. Main Portfolio Backtest

Four optimizer modes crossed with two execution modes, at the 5-day horizon, after cost:

| Optimizer | Execution | Active Return / Period | Diagnostic Sharpe | Max Active Drawdown | Cumulative Sum | Avg Names |
|---|---|---:|---:|---:|---:|---:|
| sector_diversified | normal | 0.000225 | 0.017245 | -1.644 | 0.544 | 12.9 |
| sector_diversified | price_limit_aware | 0.000003 | 0.000226 | -1.879 | 0.007 | 12.9 |
| herding_aware | normal | -0.000352 | -0.020218 | -2.219 | -0.852 | 6.0 |
| sector_aware | normal | -0.000384 | -0.022559 | -2.233 | -0.928 | 6.0 |
| normal | normal | -0.000502 | -0.026302 | -2.610 | -1.214 | 6.3 |
| herding_aware | price_limit_aware | -0.000645 | -0.036346 | -2.830 | -1.560 | 6.0 |
| sector_aware | price_limit_aware | -0.000764 | -0.044028 | -3.025 | -1.846 | 6.1 |
| normal | price_limit_aware | -0.000822 | -0.042290 | -3.304 | -1.988 | 6.3 |

Drawdown here is the largest fall of the cumulative active-return *sum* below its running peak, in the same units as that sum. The -3.304 for `normal / price_limit_aware` is a fall of 3.304 in summed active return, not a 330% loss of capital. Individual 5-day period returns range only -12.6% to +8.5%.

**2 of eight configurations have positive after-cost active return per period.** The largest is 0.000225 at a diagnostic Sharpe of 0.02.

Each row carries a 95 percent moving-block bootstrap interval on that mean, blocked at the 5-day forecast horizon so the overlap between consecutive period returns stays inside a block:

| Optimizer / Execution | Mean | 95% Lower | 95% Upper | Verdict |
|---|---:|---:|---:|---|
| sector_diversified / normal | 0.000225 | -0.000624 | 0.001085 | spans zero |
| sector_diversified / price_limit_aware | 0.000003 | -0.000873 | 0.000868 | spans zero |
| herding_aware / normal | -0.000352 | -0.001446 | 0.000774 | spans zero |
| sector_aware / normal | -0.000384 | -0.001437 | 0.000659 | spans zero |
| normal / normal | -0.000502 | -0.001677 | 0.000725 | spans zero |
| herding_aware / price_limit_aware | -0.000645 | -0.001763 | 0.000518 | spans zero |
| sector_aware / price_limit_aware | -0.000764 | -0.001828 | 0.000388 | spans zero |
| normal / price_limit_aware | -0.000822 | -0.002056 | 0.000462 | spans zero |

**Every interval spans zero.** None of the eight configurations is distinguishable from no active return at all.

Two patterns run through the table. Price-limit-aware execution is worse than normal execution in every matched pair, which quantifies how much of the result depends on fills HOSE's daily bands would have blocked. And wider diversification helps: `sector_diversified` (about 13 names) outperforms the concentrated `normal` mode (about 6 names). A ranking with strong edge should do better when allowed to concentrate, so this is evidence that the signal is weak.

The most conservative configuration is **`normal` + `price_limit_aware`** at -0.000822, since it neither relaxes execution assumptions nor relies on wide diversification.

## 3. Forecast Horizon

| Horizon | Rank IC | Top-5 Hit Rate | Top-5 Return | After-Cost Return | Diagnostic Sharpe | Cumulative Sum | Turnover | Dates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d | 0.044204 | 0.482784 | 0.001144 | -0.001420 | -0.185495 | -3.451 | 0.368 | 2428 |
| 5d | 0.033360 | 0.481307 | 0.001073 | -0.000822 | -0.042290 | -1.988 | 0.281 | 2418 |
| 10d | 0.037407 | 0.482147 | 0.002144 | 0.000351 | 0.012532 | 0.845 | 0.240 | 2403 |

**Only the 10-day horizon clears transaction costs**, at 0.000351 per period. The 1-day and 5-day horizons do not.

The 1-day horizon has the *highest* Rank IC of the three (0.0442) and is still net-negative after cost. That combination is informative: raw predictive accuracy at one day is real, but daily rebalancing turnover consumes it. It is a cost problem, not a prediction problem, and turnover is what separates the three — 0.368 at one day against 0.240 at ten.

The 5-day horizon is the production target, at -0.000822 per period with a diagnostic Sharpe of -0.042.

Overlap disclosure: the 10-day horizon's 2,403 evaluated dates correspond to roughly **240 genuinely non-overlapping periods**, and the 5-day horizon's 2,418 to roughly 483. Average after-cost return is per forecast period, not annualized.

## 4. Feature Ablation

One feature group removed at a time, all under price-limit-aware execution:

| Configuration | Features | Rank IC | Top-5 Hit Rate | Top-5 Return | After-Cost Return | Cumulative Sum |
|---|---:|---:|---:|---:|---:|---:|
| All features | 99 | 0.033360 | 0.481307 | 0.001073 | -0.000822 | -1.988 |
| Excluding risk | 97 | 0.033263 | 0.481472 | 0.000794 | -0.001244 | -3.007 |
| Excluding herding | 89 | 0.033124 | 0.482299 | 0.001097 | -0.000818 | -1.978 |
| Excluding price limit | 84 | 0.029790 | 0.477916 | 0.000920 | -0.001127 | -2.726 |
| Excluding volume/liquidity | 88 | 0.029133 | 0.476427 | 0.000724 | -0.001224 | -2.960 |

**The full feature set is the strongest configuration on Rank IC**, so no feature group is dead weight on ranking quality. That is a separate claim from paying for itself: none of the five clears transaction costs.

Removing volume/liquidity or price-limit features costs the most Rank IC, so both groups carry information — which is what one would expect of features built around the mechanics of the exchange.

`without_herding` is essentially tied with the full set on both Rank IC and after-cost return. Herding features are unproven as an alpha source, which is consistent with their original purpose as a portfolio risk control.

`src/ablation_tests.py` holds no prediction cache. An earlier cache had no freshness check and could serve predictions from a stale feature snapshot, so it was removed; these figures come from a full run.

## 5. Model Comparison

Equal-weight top-5 by model score, diagnostic only:

| Model | Rank IC | Top-5 Hit Rate | Top-5 Return / 5d | Diagnostic Sharpe |
|---|---:|---:|---:|---:|
| elastic_net | 0.036697 | 0.198677 | 0.001992 | 0.095810 |
| ridge | 0.036075 | 0.192721 | 0.001478 | 0.075068 |
| gradient_boosting | 0.033360 | 0.187924 | 0.001073 | 0.053650 |
| random_forest | 0.031098 | 0.191150 | 0.001076 | 0.051486 |
| classification | 0.020905 | 0.207858 | 0.001811 | 0.086431 |

Every model improved with the wider training window, but the *linear* models improved most and now lead clearly: elastic net at 0.0367 against gradient boosting at 0.0334, and elastic net returns roughly 85 percent more per top-5 basket. Under the previous window the spread was narrow (0.0179 to 0.0217) and the ordering close to arbitrary.

Gradient boosting is still the model carried into portfolio construction, for continuity with earlier versions of this project. On the current evidence that choice is now actively questionable rather than merely unjustified: elastic net beats it on Rank IC, hit rate, top-5 return, and diagnostic Sharpe simultaneously. Switching would change every portfolio number and has not been done here.

## 6. Baseline Comparison

| Strategy | Active Return / Period | Basis | Cumulative Sum | Names |
|---|---:|---|---:|---:|
| top5_momentum | 0.002264 | before cost | 5.474 | 5 |
| ML sector_diversified / normal | 0.000225 | after cost | 0.544 | 12.9 |
| ML sector_diversified / price_limit_aware | 0.000003 | after cost | 0.007 | 12.9 |
| ML normal / normal | -0.000502 | after cost | -1.214 | 6.3 |
| ML normal / price_limit_aware | -0.000822 | after cost | -1.988 | 6.3 |
| equal_weight_all | -0.000000 | before cost | -0.000 | 30 |
| low_volatility_top10 | -0.000411 | before cost | -0.994 | 10 |
| top5_reversal | -0.002731 | before cost | -6.603 | 5 |

`top5_momentum` sorts on `return_10d` alone and holds the top five. It is an order of magnitude ahead of the best ML configuration.

The cost basis differs, so this table alone is not conclusive. Recomputing each model's own top-5 pick on a fully matched basis — same concentration, same relative-to-VN30 target, before any cost or portfolio construction, same 2,478 evaluated dates — removes that objection:

| Selection Rule | Top-5 Active Return, Before Cost |
|---|---:|
| top5_momentum (`return_10d` only) | 0.002264 |
| elastic_net | 0.001992 |
| classification | 0.001811 |
| ridge | 0.001478 |
| random_forest | 0.001076 |
| gradient_boosting | 0.001073 |

Momentum leads. The strongest model, `elastic_net`, reaches 88 percent of the baseline, so momentum leads it by 1.14x.

Part of the earlier gap was therefore an artefact of undertrained models rather than a property of the signal. The remaining gap is real but much smaller, and the two best models are now the linear ones, which is what one would expect if the useful structure is close to a linear combination of a few momentum-like features.

Still unresolved: a single trailing-return sort continues to beat a 99-feature model that receives `return_10d` as an input. Candidate explanations, none yet tested: the tree depth limit (`max_depth=2`) cannot represent the interaction that makes momentum work; the remaining features still dilute a small number of informative axes; or the relative-return target is a poor fit for a momentum-dominated market.

## 7. Foreign Flow

FB/FS features were evaluated against a rule fixed before results were seen: paired block bootstrap on daily Rank IC over 222 non-overlapping 10-day signals, promote only if the 95 percent CI excludes zero.

**Result: mean difference -0.002608, 95 percent CI [-0.013314, +0.008297]. Not promoted.**

| Feature Set | Rank IC |
|---|---:|
| Frozen foreign flow (net-style only) | 0.063616 |
| Frozen plus original buy/sell | 0.062448 |
| Frozen plus enhanced buy/sell | 0.061007 |
| Market baseline (no foreign flow) | 0.055389 |
| Buy/sell only | 0.053736 |

Net-style foreign-flow features add real value: they beat the market baseline by 0.0082. But every set that adds a buy/sell split ranks below the net-only set, and adding more buy/sell features makes it monotonically worse.

The cause is not that the model ignores the new features — two of them rank second and third in its own importance output. It is that their relationship with forward returns flips sign across the sample: positive in six years, negative in four, a 50.5 percent per-signal win rate, and a day-to-day standard deviation about 29 times the mean difference. This looks like a non-stationary relationship rather than a representation problem or pure noise. Full analysis in `docs/enhanced_fb_fs_checkpoint.md`.

## 8. Concentration and Latest Diagnostics

Latest signal date 2026-07-15, `normal` optimizer: 16 holdings, maximum single-name weight 20.0 percent, HHI 0.147, effective position count 6.8, top issuer group Vingroup at 20.1 percent. No issuer group approaches the 40 percent cap. The single-name cap is now binding exactly at 20 percent, and the portfolio is more concentrated than under the previous training window (16 holdings against 21).

The rank diagnostic for the same date still shows misses at the top:

| Ticker | Predicted Rank | Realized Rank | Realized Forward Return |
|---|---:|---:|---:|
| FPT | 1 | 10 | +2.61% |
| SSI | 2 | 24 | -2.57% |
| MSN | 3 | 11 | -0.22% |
| VJC | 4 | 7 | +3.06% |
| VHM | 5 | 14 | -0.91% |
| LPB | 28 | 1 | +11.84% |
| VNM | 30 | 2 | +11.35% |

Two of the top five realized positive returns, an improvement on the previous run where four of five were negative. But the two best performers of the day were still ranked 28th and 30th out of 30. One date is not evidence, and this remains a concrete illustration of what a 0.03 Rank IC means in practice.

## 9. Figures

```text
reports/figures/top_gradient_boosting_feature_importance.png
reports/figures/ablation_diagnostic_sharpe.png
reports/figures/horizon_diagnostic_sharpe.png
reports/figures/horizon_rank_ic.png
```

The assembled report page is `reports/reportpage.html`, generated from `reports/tables/` by `scripts/build_report_page.py` so its figures cannot drift from the tables.

## 10. Weaknesses

**A one-feature baseline beats the models.** Section 6. Narrowed from 1.6x to 1.14x by widening the training window, but unresolved.

**Margins are thin.** Section 2. 2 of eight portfolio configurations clear cost, the most conservative does not, and every one of the eight has a 95 percent bootstrap interval on mean active return that spans zero.

**Multiple comparisons are uncorrected.** Three horizons, five ablations, four optimizer modes, two execution modes, five foreign-flow feature sets, all against the same history. Only the foreign-flow test carries a confidence interval, and it does not adjust for being the fifth feature set on the same development data. The best number in this repository is partly a product of search.

**Most metrics have no confidence intervals.** Rank IC, Sharpe, and return figures outside the foreign-flow experiment are point estimates. Given the day-to-day dispersion observed in the foreign-flow analysis, several rankings in this report would likely not survive interval estimation.

**Cumulative figures are sums of overlapping periods**, not compounded returns.

**Concentration reduces performance** rather than improving it, which is consistent with a weak ranking signal.

**The production model is no longer the best one.** Section 5. Elastic net now beats gradient boosting on every diagnostic, but gradient boosting is still what feeds portfolio construction.

**Data scope.** Daily bars only: no intraday queue priority, order-book depth, or hidden liquidity. No fundamentals, macro, or news. Foreign ownership room is not modeled.

**The live daily-update path has never made a vendor call.** It now reads the same FiinQuant history as the research pipeline, on the same raw-for-execution, adjusted-for-returns contract, but since the migration it has only been exercised against a stub client.

## 11. Training Window

Every figure in this report uses a walk-forward training window of **63 trading
dates**, defined once as `DEFAULT_TRAIN_SIZE` in `src/walk_forward_split.py`.
`train_size` in `build_walk_forward_date_windows` counts dates, not blocks of
dates, so a 63-date window fits on roughly 63 x 30 = 1,890 rows.

The value is chosen by sweeping {3, 21, 63, 126, 252} on
223 shared evaluation dates
(`scripts/diagnose_training_window.py`, results in
`reports/tables/training_window_diagnostic.csv`):

| Training window | Train rows | Rank IC | Top-5 hit rate | Top-5 return |
|---|---:|---:|---:|---:|
| 3 dates | 90 | 0.016773 | 0.482511 | 0.000002 |
| 21 dates | 630 | 0.031102 | 0.489686 | -0.001508 |
| 63 dates | 1,890 | 0.028126 | 0.498655 | 0.001074 |
| 126 dates | 3,779 | 0.023372 | 0.480717 | -0.000116 |
| 252 dates | 7,555 | -0.000485 | 0.449327 | -0.002341 |

The relationship is an inverted U, not monotonic: Rank IC peaks at 21 dates and
collapses to roughly zero at 252, consistent with long windows averaging across
market regimes that no longer apply. 63 is chosen over the Rank-IC-optimal 21
because it has the best top-5 hit rate and the only clearly positive top-5
return, and portfolio construction selects a top-N basket rather than trading
the full ranking.

`Train rows` is the measured count of dataset rows on the first window's
training dates, not the window length times 30. Ten dates in the dataset carry
29 constituents rather than 30, because FLC has no data for the week of
2017-01-16 and ROS none for the week of 2021-01-25, so the 126-date window
measures 3,779 rows against an arithmetic 3,780 and the 252-date window 7,555
against 7,560.

The sweep reports Rank IC
0.028126
for the chosen window while section 3 reports
0.033360 for the same estimator. The sweep
scores 223 test dates, subsampled at every
tenth date and starting late enough that the 252-date window is also reachable,
so all five configurations are scored on identical dates. Production
walk-forward scores every date it can,
2,418 of them. A shorter, later, sparser
subsample is a different sample of the same estimator, so the two are not
required to agree.

One caveat: this is a parameter chosen by measuring five values against the
same history, which adds to the multiple-comparisons problem in section 10. No
confidence interval supports the choice.

## 12. Conclusion

**No optimizer mode beats buying and holding the VN30 after costs.** The
per-period figures above are after-cost positive, but they are means over
overlapping forecast periods, and a portfolio cannot hold an overlapping
period. What decides deployability is what a portfolio running this strategy
would have done.

The strategy signals daily and holds five days, so it trades five overlapping
tranches: one fifth of capital rebalances into each day's targets and is held
for the horizon. That construction holds every phase simultaneously and has no
free parameters. Over 2,414 trading days, 2016-11-15 to
2026-07-16 — 9.58 years at 252 trading days per year — after costs,
price-limit-aware execution:

| Optimizer | Strategy equity | VN30 benchmark | Active (compounded) | Equity ratio | Strategy drawdown | Active drawdown |
|---|---:|---:|---:|---:|---:|---:|
| `sector_diversified` | 1.4850 | 2.6918 | 0.5557 | 0.5517 | -58.3% | -50.3% |
| `normal` | 1.1699 | 2.6918 | 0.4458 | 0.4346 | -63.5% | -61.7% |
| `herding_aware` | 1.1945 | 2.6918 | 0.4345 | 0.4437 | -57.7% | -61.4% |
| `sector_aware` | 1.0105 | 2.6918 | 0.3687 | 0.3754 | -62.1% | -66.1% |

Every active equity is below 1.0. The best configuration turns 1.0 into
1.4850 against the index's 2.6918, with a
58 percent peak-to-trough fall in capital;
the others finish 55 to 63 percent behind the index on
compounded active return. Nothing here is deployable, and the ranking among the
four modes is a ranking of how far behind they finish.

Three columns here are read wrongly often enough to be worth defining:

- **Active** is the compounded daily active return, the product over trading
  days of 1 + (strategy return − benchmark return). It is not strategy equity
  divided by benchmark equity, which is published beside it as **Equity
  ratio** precisely because the two differ. Compounding a difference of
  returns is not the same as taking a ratio of compounded returns.
- **Strategy drawdown** is the worst peak-to-trough fall of the compounded
  strategy equity curve: a genuine percentage of capital.
- **Active drawdown** is the same measure on the compounded active curve, so
  it describes the hedged sleeve rather than the account. The two drawdown
  columns differ because a portfolio and a hedged version of it fall at
  different times.

None of the three is the `max_active_drawdown` reported in section 3 and in
`benchmark_results.csv`, which is a running-peak fall on a `cumsum` of
overlapping five-day active returns, in summed-return units rather than as a
percentage of capital.

### Why not non-overlapping sampling

Sampling every fifth date also produces non-overlapping periods, and the
per-period means reconcile exactly with section 3 once weighted by period
count. But it leaves the starting offset as a free parameter, and that choice
moves the answer more than the strategy does:

| Optimizer | Final-equity spread across the 5 phases | Across tranche start offsets |
|---|---:|---:|
| `herding_aware` | 1.0608 | 0.0235 |
| `normal` | 0.7719 | 0.0279 |
| `sector_aware` | 0.6483 | 0.0233 |
| `sector_diversified` | 0.4803 | 0.0303 |

A factor of 20 to 45. One offset makes `sector_diversified` appear positive in
all five phases, which is a property of the sampling rather than of the
strategy. The tranche construction is reported above for that reason.

Sources: `reports/tables/tranche_portfolio_diagnostic.csv` from
`scripts/diagnose_tranche_drawdown.py`, and
`reports/tables/compounded_phase_sensitivity.csv` from
`scripts/diagnose_compounded_returns.py`.

Three further results point the same way. A single-feature momentum rule
outperforms every trained model on a matched basis. The best portfolio
configuration earns 0.000930 per 5-day period at a diagnostic Sharpe of 0.07.
Foreign buy/sell features failed a pre-registered test. None of the figures in
this document carry confidence intervals, and each additional configuration
tested against the same history compounds the multiple-comparisons problem in
section 10.

### What this does not establish

This is a negative result about a specification, not about the market. Two
things remain untested and would have to be ruled out before concluding that
no tradable structure exists in VN30:

- Features enter the models as raw levels, standardised only over the pooled
  training window and never within a date, while the target is demeaned
  against the index by construction. The market-wide component of every
  feature is therefore orthogonal to the target but still shapes split
  thresholds and coefficients. A within-date sort, which is what the momentum
  baseline performs, cancels that component exactly. Whether this explains the
  baseline gap has not been measured.
- Drawdown has never been evaluated against an exposure control matched on
  average exposure, so the existing downside-overlay results cannot separate
  risk timing from holding less risk.

Adding features answers neither question, and each attempt costs
multiple-comparisons budget against a history already used for three horizons,
five ablations, four optimizer modes, two execution modes, and five
foreign-flow feature sets.
