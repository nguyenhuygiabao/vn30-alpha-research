# Final Results

## 1. Executive Summary

This project ranks VN30 stocks by expected outperformance against the VN30 benchmark, converts those rankings into a constrained long-only portfolio, and measures what survives transaction costs and Vietnam-specific execution frictions.

The main finding remains negative, but by a smaller margin than before:

**A single-feature momentum rule still beats every trained model.** Sorting on trailing 10-day return and holding the top five names returns 0.002264 per 5-day period before cost. The best trained model's top-five basket returns 0.001992 on the same basis. `return_10d` is already an input feature to every model, so the models have strictly more information and still use it worse.

Beyond that:

- The **5-day and 10-day horizons** are now after-cost positive. The 1-day horizon is not.
- **Two of five feature-ablation configurations** clear transaction costs, and the full feature set is the strongest.
- **Foreign buy/sell features failed a pre-registered test** and are not promoted.
- All eight ML portfolio variants now have positive after-cost active return.

These figures reflect a walk-forward training window of 63 trading dates. It was previously 3, which fit about 90 rows against 99 features. See section 11 for the before/after comparison.

Pipeline components: point-in-time VN30 membership, an enforced raw/adjusted price contract, purged walk-forward validation, modeled transaction costs and price limits, and a pre-registered bootstrap test for the foreign-flow feature sets.

Every cumulative figure below sums overlapping forecast-period active returns. It is not a compounded portfolio return and should not be read as a track record. Max active drawdown is measured on that same summed series, so it is reported as a plain number in summed-return units, not as a percentage of capital.

Compounded as a portfolio instead, the strategy is not deployable. Section 12
gives the figures.

## 2. Main Portfolio Backtest

Four optimizer modes crossed with two execution modes, at the 5-day horizon, after cost:

| Optimizer | Execution | Active Return / Period | Diagnostic Sharpe | Max Active Drawdown | Cumulative Sum | Avg Names |
|---|---|---:|---:|---:|---:|---:|
| sector_diversified | normal | 0.000930 | 0.071309 | -1.027 | 2.249 | 12.9 |
| sector_diversified | price_limit_aware | 0.000696 | 0.052945 | -1.199 | 1.682 | 12.9 |
| herding_aware | normal | 0.000512 | 0.029370 | -1.161 | 1.238 | 6.0 |
| sector_aware | normal | 0.000488 | 0.028693 | -1.116 | 1.181 | 6.0 |
| normal | normal | 0.000419 | 0.021966 | -1.437 | 1.014 | 6.3 |
| herding_aware | price_limit_aware | 0.000205 | 0.011565 | -1.493 | 0.496 | 6.0 |
| sector_aware | price_limit_aware | 0.000089 | 0.005159 | -1.560 | 0.216 | 6.1 |
| normal | price_limit_aware | 0.000076 | 0.003907 | -1.778 | 0.184 | 6.3 |

Drawdown here is the largest fall of the cumulative active-return *sum* below its running peak, in the same units as that sum. The -1.778 for `normal / price_limit_aware` is a fall of 1.778 in summed active return, not a 178% loss of capital. Individual 5-day period returns range only -12.6% to +8.5%.

All eight configurations are now positive, where five of eight were negative under the previous 3-date training window. The largest is 0.000930 per period at a diagnostic Sharpe of 0.07 — better than before, still small.

Two patterns persist. Price-limit-aware execution is worse than normal execution in every matched pair, which quantifies how much of the result depends on fills HOSE's daily bands would have blocked. And wider diversification still helps: `sector_diversified` (about 13 names) outperforms the concentrated `normal` mode (about 6 names). A ranking with strong edge should do better when allowed to concentrate, so this remains evidence that the signal is weak, even though every mode now clears cost.

The most conservative configuration is **`normal` + `price_limit_aware`** at 0.000076, since it neither relaxes execution assumptions nor relies on wide diversification.

## 3. Forecast Horizon

| Horizon | Rank IC | Top-5 Hit Rate | Top-5 Return | After-Cost Return | Diagnostic Sharpe | Cumulative Sum | Turnover | Dates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d | 0.044204 | 0.482784 | 0.000387 | -0.000244 | -0.031919 | -0.593 | 0.391 | 2428 |
| 5d | 0.033360 | 0.481307 | 0.001073 | 0.000076 | 0.003907 | 0.184 | 0.366 | 2418 |
| 10d | 0.037407 | 0.482147 | 0.001943 | 0.001120 | 0.039957 | 2.691 | 0.346 | 2403 |

The 10-day horizon remains the strongest after cost, and the 5-day horizon has crossed from negative to marginally positive. Both clear transaction costs.

The 1-day horizon now has the *highest* Rank IC of the three (0.0442) yet is still the only one that is net-negative after cost. That combination is informative: raw predictive accuracy at one day is real, but daily rebalancing turnover consumes it. It is a cost problem, not a prediction problem.

The 5-day horizon is this project's original baseline. Under the previous 3-date training window it was net-negative; it is now marginally positive at 0.000076 per period, with a diagnostic Sharpe of 0.004 that is still indistinguishable from zero.

Overlap disclosure: the 10-day horizon's 2,403 evaluated dates correspond to roughly **240 genuinely non-overlapping periods**. Average after-cost return is per forecast period, not annualized.

## 4. Feature Ablation

One feature group removed at a time, all under price-limit-aware execution:

| Configuration | Features | Rank IC | Top-5 Hit Rate | Top-5 Return | After-Cost Return | Cumulative Sum |
|---|---:|---:|---:|---:|---:|---:|
| All features | 99 | 0.033360 | 0.481307 | 0.001073 | 0.000076 | 0.184 |
| Excluding risk | 97 | 0.033263 | 0.481472 | 0.000794 | -0.000347 | -0.840 |
| Excluding herding | 89 | 0.033124 | 0.482299 | 0.001097 | 0.000081 | 0.195 |
| Excluding price limit | 84 | 0.029790 | 0.477916 | 0.000920 | -0.000202 | -0.489 |
| Excluding volume/liquidity | 88 | 0.029133 | 0.476427 | 0.000724 | -0.000309 | -0.747 |

**The full feature set is now the strongest configuration**, and two of five clear transaction costs.

This reverses the previous reading. Under the 3-date training window, removing features improved Rank IC and `without_risk` led the table, which suggested the feature set was mostly noise. With 63 dates of training the ranking inverts: `all_features` leads, and every reduced set is worse or no better. The earlier result was a symptom of too little training data to use 99 features, not evidence that the features were harmful.

Removing volume/liquidity or price-limit features is now clearly worse on both Rank IC and after-cost return, so both groups carry information.

`without_herding` is a marginal exception, essentially tied with the full set (0.000081 against 0.000076 after cost, on a Rank IC 0.0002 lower). Herding features remain unproven as an alpha source, which is consistent with their original purpose as a portfolio risk control.

These numbers are from a fresh run. `src/ablation_tests.py` previously carried a cache with no freshness check and silently served three-week-old predictions from a 51-feature snapshot of the pipeline. That output showed several configurations as after-cost profitable; those figures described an obsolete pipeline. The cache has been removed.

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
| ML sector_diversified / normal | 0.000930 | after cost | 2.249 | 12.9 |
| ML sector_diversified / price_limit_aware | 0.000696 | after cost | 1.682 | 12.9 |
| ML normal / normal | 0.000419 | after cost | 1.014 | 6.3 |
| ML normal / price_limit_aware | 0.000076 | after cost | 0.184 | 6.3 |
| equal_weight_all | ~0 | before cost | ~0 | 30 |
| low_volatility_top10 | -0.000411 | before cost | -0.994 | 10 |
| top5_reversal | -0.002731 | before cost | -6.603 | 5 |

`top5_momentum` sorts on `return_10d` alone and holds the top five. It is an order of magnitude ahead of the best ML configuration.

The cost basis differs, so this table alone is not conclusive. Recomputing each model's own top-5 pick on a fully matched basis — same concentration, same relative-to-VN30 target, before any cost or portfolio construction, same 2,478 evaluated dates — removes that objection:

| Selection Rule | Top-5 Active Return, Before Cost |
|---|---:|
| top5_momentum (`return_10d` only) | 0.002264 |
| elastic_net | 0.001992 |
| ridge | 0.001478 |
| gradient_boosting | 0.001080 |
| random_forest | 0.001030 |

Momentum still leads, but the margin has narrowed substantially. The best model now reaches 88 percent of the baseline, against 62 percent under the 3-date training window. Momentum led the best model by 1.6x before; it now leads by 1.14x.

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

**Margins are thin.** Section 2. All eight portfolio configurations now clear cost, but the most conservative does so by 0.000076 per period at a diagnostic Sharpe of 0.004.

**Multiple comparisons are uncorrected.** Three horizons, five ablations, four optimizer modes, two execution modes, five foreign-flow feature sets, all against the same history. Only the foreign-flow test carries a confidence interval, and it does not adjust for being the fifth feature set on the same development data. The best number in this repository is partly a product of search.

**Most metrics have no confidence intervals.** Rank IC, Sharpe, and return figures outside the foreign-flow experiment are point estimates. Given the day-to-day dispersion observed in the foreign-flow analysis, several rankings in this report would likely not survive interval estimation.

**Cumulative figures are sums of overlapping periods**, not compounded returns.

**Concentration reduces performance** rather than improving it, which is consistent with a weak ranking signal.

**The production model is no longer the best one.** Section 5. Elastic net now beats gradient boosting on every diagnostic, but gradient boosting is still what feeds portfolio construction.

**Data scope.** Daily bars only: no intraday queue priority, order-book depth, or hidden liquidity. No fundamentals, macro, or news. Foreign ownership room is not modeled.

**The live daily-update path has never made a vendor call.** It now reads the same FiinQuant history as the research pipeline, on the same raw-for-execution, adjusted-for-returns contract, but since the migration it has only been exercised against a stub client.

## 11. Training Window Change

Every figure in this report reflects a walk-forward training window of **63 trading dates**. It was previously **3**.

`train_size` in `build_walk_forward_date_windows` counts dates, not blocks of dates, so the old setting fit 99 features on about 90 rows per window. The value was duplicated across five model modules and appears never to have been revisited as the dataset grew. It is now a single constant, `DEFAULT_TRAIN_SIZE` in `src/walk_forward_split.py`.

The value was chosen by sweeping {3, 21, 63, 126, 252} on 223 shared evaluation dates (`scripts/diagnose_training_window.py`, results in `reports/tables/training_window_diagnostic.csv`). The relationship is an inverted U, not monotonic: Rank IC peaks at 21 dates and collapses to roughly zero at 252, consistent with long windows averaging across market regimes that no longer apply. 63 was chosen over the Rank-IC-optimal 21 because it had the best top-5 hit rate and the only clearly positive top-5 return, and portfolio construction selects a top-N basket.

Effect on the headline numbers:

| Metric | train_size=3 | train_size=63 |
|---|---:|---:|
| Rank IC, 1d | 0.018752 | 0.044204 |
| Rank IC, 5d | 0.019869 | 0.033360 |
| Rank IC, 10d | 0.039083 | 0.037407 |
| After-cost return, 5d | -0.000481 | 0.000076 |
| After-cost return, 10d | 0.000804 | 0.001120 |
| Best ML portfolio | 0.000341 | 0.000930 |
| ML configurations positive | 3 of 8 | 8 of 8 |
| Ablation configurations positive | 0 of 5 | 2 of 5 |
| Best model vs momentum baseline | 62% | 88% |

Two results did not improve. The 10-day Rank IC is marginally lower (0.0374 against 0.0391), and the 10-day top-5 hit rate fell slightly. The sweep only measured the 5-day horizon, so the 10-day outcome was not predicted by it and is a genuine, if small, cost of the change.

One caveat carried forward: this is a parameter chosen by measuring five values against the same history, which adds to the multiple-comparisons problem in section 10. The improvement is large and consistent enough across horizons, ablations, and portfolio modes that it is unlikely to be selection noise, but no confidence interval supports that judgement.

## 12. Conclusion

**The strategy is not deployable on VN30 over this sample.** The per-period
figures above are after-cost positive, but they are sums of overlapping
forecast periods, and the property that decides deployability is what happens
when the same returns are compounded as a portfolio.

Taking non-overlapping 5-day periods for the production configuration
(`normal` optimizer, price-limit-aware execution) gives 484 independent
periods, and a result that depends on which of the five overlap phases the
curve starts on:

| Starting phase | Mean per period | Final equity | Max drawdown |
|---:|---:|---:|---:|
| 0 | -0.000463 | 0.7256 | -43.15% |
| 1 | -0.000244 | 0.8062 | -40.03% |
| 2 | +0.000861 | 1.3938 | -25.29% |
| 3 | -0.000171 | 0.8392 | -43.05% |
| 4 | +0.000398 | 1.1095 | -31.82% |

The five phase means average to +0.0000762, which reconciles exactly with the
+0.000076 in section 3. Nothing is wrong with the headline arithmetic. But an
outcome ranging from 0.73 to 1.39 on the choice of starting weekday is not an
edge that survives contact with implementation, and the drawdown expressed in
capital terms is 25 to 43 percent rather than the summed-unit figures reported
elsewhere in this document.

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
