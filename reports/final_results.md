# Final Results

## 1. Executive Summary

This project ranks VN30 stocks by expected outperformance against the VN30 benchmark, converts those rankings into a constrained long-only portfolio, and measures what survives transaction costs and Vietnam-specific execution frictions.

The main finding is negative:

**A single-feature momentum rule beats every trained model.** Sorting on trailing 10-day return and holding the top five names returns 0.002310 per 5-day period before cost. The best trained model's top-five basket returns 0.001433 on the same basis. `return_10d` is already an input feature to every model, so the models have strictly more information and use it worse.

Beyond that:

- Only the **10-day horizon** is after-cost positive. The 1-day and 5-day horizons are both net-negative.
- **No feature-ablation configuration** clears transaction costs at the 5-day horizon. All five are negative.
- **Foreign buy/sell features failed a pre-registered test** and are not promoted.
- Five of eight ML portfolio variants have negative after-cost active return.

Pipeline components: point-in-time VN30 membership, an enforced raw/adjusted price contract, purged walk-forward validation, modeled transaction costs and price limits, and a pre-registered bootstrap test for the foreign-flow feature sets.

Every cumulative figure below sums overlapping forecast-period active returns. It is not a compounded portfolio return and should not be read as a track record. Max active drawdown is measured on that same summed series, so it is reported as a plain number in summed-return units, not as a percentage of capital.

## 2. Main Portfolio Backtest

Four optimizer modes crossed with two execution modes, at the 5-day horizon, after cost:

| Optimizer | Execution | Active Return / Period | Diagnostic Sharpe | Max Active Drawdown | Cumulative Sum | Avg Names |
|---|---|---:|---:|---:|---:|---:|
| sector_diversified | normal | 0.000341 | 0.029910 | -0.710 | 0.844 | 25.8 |
| sector_diversified | price_limit_aware | 0.000102 | 0.008918 | -0.922 | 0.254 | 25.8 |
| herding_aware | normal | 0.000036 | 0.002103 | -1.361 | 0.088 | 11.6 |
| normal | normal | -0.000064 | -0.003511 | -1.454 | -0.158 | 13.2 |
| sector_aware | normal | -0.000080 | -0.004878 | -1.230 | -0.198 | 11.1 |
| herding_aware | price_limit_aware | -0.000280 | -0.016299 | -1.455 | -0.692 | 11.6 |
| sector_aware | price_limit_aware | -0.000445 | -0.026249 | -1.667 | -1.104 | 11.1 |
| normal | price_limit_aware | -0.000481 | -0.025742 | -1.876 | -1.191 | 13.2 |

Drawdown here is the largest fall of the cumulative active-return *sum* below its running peak, in the same units as that sum. The -1.876 for `normal / price_limit_aware` is a fall of 1.876 in summed active return from a peak of +0.607 to a trough of -1.269; it is not a 187% loss of capital. Individual 5-day period returns range only -12.6% to +8.5%.

Only three of eight configurations are positive, and the largest is 0.000341 per period with a diagnostic Sharpe of 0.03 — economically indistinguishable from zero.

Two patterns are consistent. Price-limit-aware execution is worse than normal execution in every matched pair, which quantifies how much of the paper result depends on fills HOSE's daily bands would have blocked. And wider diversification helps: `sector_diversified` (roughly 26 names) outperforms the concentrated `normal` mode (13 names), the opposite of what a genuinely strong ranking signal would produce. A model with real edge should do better when allowed to concentrate into its top picks.

The most conservative configuration is **`normal` + `price_limit_aware`** at -0.000481, since it neither relaxes execution assumptions nor relies on wide diversification.

## 3. Forecast Horizon

| Horizon | Rank IC | Top-5 Hit Rate | Top-5 Return | After-Cost Return | Diagnostic Sharpe | Cumulative Sum | Turnover | Dates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d | 0.018752 | 0.467203 | 0.000446 | -0.000703 | -0.114101 | -1.749 | 0.391 | 2488 |
| 5d | 0.019869 | 0.469492 | 0.000998 | -0.000481 | -0.025742 | -1.191 | 0.366 | 2478 |
| 10d | 0.039083 | 0.487130 | 0.001910 | 0.000804 | 0.030871 | 1.980 | 0.346 | 2463 |

The 10-day horizon is the only after-cost positive configuration in the project. It roughly doubles the Rank IC of the shorter horizons, has the best hit rate, and turns over least, which is why it costs less to run.

The 1-day horizon is the clearest failure. Rank IC is comparable to 5-day but after-cost return is the worst of the three, because turnover is highest and a 0.019 Rank IC cannot pay for rebalancing every session.

The 5-day horizon is this project's original baseline. It is net-negative after cost and should no longer be described as strong. Earlier versions of this report claimed it "performed strongly"; that claim came from a pre-purge run and was leakage-inflated.

Overlap disclosure: the 10-day horizon's 2,463 evaluated dates correspond to roughly **246 genuinely non-overlapping periods**. Average after-cost return is per forecast period, not annualized.

## 4. Feature Ablation

One feature group removed at a time, all under price-limit-aware execution:

| Configuration | Features | Rank IC | Top-5 Hit Rate | Top-5 Return | After-Cost Return | Cumulative Sum |
|---|---:|---:|---:|---:|---:|---:|
| Excluding risk | 97 | 0.025746 | 0.476755 | 0.001234 | -0.000321 | -0.797 |
| Excluding volume/liquidity | 88 | 0.021549 | 0.469169 | 0.000773 | -0.000540 | -1.337 |
| Excluding herding | 89 | 0.020276 | 0.468846 | 0.000899 | -0.000572 | -1.418 |
| All features | 99 | 0.019869 | 0.469492 | 0.000998 | -0.000481 | -1.191 |
| Excluding price limit | 84 | 0.013470 | 0.466990 | 0.000567 | -0.000664 | -1.644 |

**Every configuration is net-negative after cost.**

Removing the two risk features (`rolling_vol_20d`, `drawdown`) gives the best Rank IC, the best hit rate, and the least-negative after-cost result. This has not been investigated further.

Removing price-limit features is worst on every metric, indicating that group carries information about Vietnamese market microstructure rather than acting as noise.

Rank IC and after-cost outcome do not move together. Excluding volume/liquidity improves Rank IC (0.0215 vs 0.0199) but worsens after-cost return (-0.000540 vs -0.000481). The plausible mechanism is that liquidity features add noise to the raw prediction while still steering the optimizer away from names that are expensive to trade. Prediction accuracy and portfolio outcome are separate objectives here.

These numbers are from a fresh run. `src/ablation_tests.py` previously carried a cache with no freshness check and silently served three-week-old predictions from a 51-feature snapshot of the pipeline. That output showed several configurations as after-cost profitable; those figures described an obsolete pipeline. The cache has been removed.

## 5. Model Comparison

Equal-weight top-5 by model score, diagnostic only:

| Model | Rank IC | Top-5 Hit Rate | Top-5 Return / 5d | Diagnostic Sharpe |
|---|---:|---:|---:|---:|
| elastic_net | 0.021737 | 0.185069 | 0.001351 | 0.066882 |
| gradient_boosting | 0.019869 | 0.185230 | 0.000998 | 0.046962 |
| classification | 0.018657 | 0.189427 | 0.001006 | 0.050339 |
| random_forest | 0.018015 | 0.190638 | 0.001433 | 0.066590 |
| ridge | 0.017905 | 0.183697 | 0.000786 | 0.039614 |

Rank IC spans 0.0179 to 0.0217 across five model families of very different capacity. That narrow a spread, with linear models at the top, indicates the ceiling is set by the features and the signal-to-noise of the problem, not by model choice. Gradient boosting is carried into portfolio construction for continuity with earlier versions of this project, not because it wins.

## 6. Baseline Comparison

| Strategy | Active Return / Period | Basis | Cumulative Sum | Names |
|---|---:|---|---:|---:|
| top5_momentum | 0.002310 | before cost | 5.724 | 5 |
| ML sector_diversified / normal | 0.000341 | after cost | 0.844 | 25.8 |
| ML sector_diversified / price_limit_aware | 0.000102 | after cost | 0.254 | 25.8 |
| equal_weight_all | ~0 | before cost | ~0 | 30 |
| low_volatility_top10 | -0.000347 | before cost | -0.858 | 10 |
| ML normal / price_limit_aware | -0.000481 | after cost | -1.191 | 13.2 |
| top5_reversal | -0.002630 | before cost | -6.503 | 5 |

`top5_momentum` sorts on `return_10d` alone and holds the top five. It is an order of magnitude ahead of the best ML configuration.

The cost basis differs, so this table alone is not conclusive. Recomputing each model's own top-5 pick on a fully matched basis — same concentration, same relative-to-VN30 target, before any cost or portfolio construction, same 2,478 evaluated dates — removes that objection:

| Selection Rule | Top-5 Active Return, Before Cost |
|---|---:|
| top5_momentum (`return_10d` only) | 0.002310 |
| random_forest | 0.001433 |
| elastic_net | 0.001351 |
| gradient_boosting | 0.000998 |
| ridge | 0.000786 |

Momentum leads the best model by 1.6x and the worst by 2.9x, before cost or diversification enter the picture. Since `return_10d` is an input feature to all five models, they are not missing the information — they are diluting it. The `without_risk` ablation result points the same direction: removing features improved Rank IC.

This is the central unresolved problem in the project. Candidate explanations, none yet tested: the tree depth limit (`max_depth=2`) cannot represent the interaction that makes momentum work; the 99-feature input is mostly noise that swamps a small number of informative axes; or the relative-return target is a poor fit for a momentum-dominated market.

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

Latest signal date 2026-07-15, `normal` optimizer: 21 holdings, 100 percent invested, maximum single-name weight 18.9 percent (VPB), HHI 0.112, effective position count 8.9, top issuer group VPBank at 18.9 percent. No issuer group at or above 40 percent, and no position at or above 20 percent. Constraints are binding as designed.

The rank diagnostic for the same date shows the model missing badly at the top:

| Ticker | Predicted Rank | Realized Rank | Realized Forward Return |
|---|---:|---:|---:|
| CTG | 1 | 19 | -1.87% |
| TPB | 3 | 20 | -2.10% |
| SHB | 4 | 16 | -1.37% |
| SSI | 5 | 24 | -2.57% |
| LPB | 24 | 1 | +11.84% |
| VNM | 22 | 2 | +11.35% |

Four of the top five picks realized negative returns while two of the bottom seven were the two best performers of the day. This is a single date and not independent evidence, but it is a concrete illustration of what a 0.02 Rank IC means in practice.

## 9. Figures

```text
reports/figures/top_gradient_boosting_feature_importance.png
reports/figures/ablation_diagnostic_sharpe.png
reports/figures/horizon_diagnostic_sharpe.png
reports/figures/horizon_rank_ic.png
```

Interactive charts covering cumulative return, active drawdown, turnover, rolling diagnostic Sharpe, and optimizer cap hits are in `reports/interactive/`, and the assembled dashboard is `reports/dashboard.html`.

## 10. Weaknesses

**A one-feature baseline beats the models.** Section 6. Unresolved.

**Nothing clears cost at the 5-day horizon.** Section 4. Only the 10-day horizon is positive at all.

**Multiple comparisons are uncorrected.** Three horizons, five ablations, four optimizer modes, two execution modes, five foreign-flow feature sets, all against the same history. Only the foreign-flow test carries a confidence interval, and it does not adjust for being the fifth feature set on the same development data. The best number in this repository is partly a product of search.

**Most metrics have no confidence intervals.** Rank IC, Sharpe, and return figures outside the foreign-flow experiment are point estimates. Given the day-to-day dispersion observed in the foreign-flow analysis, several rankings in this report would likely not survive interval estimation.

**Cumulative figures are sums of overlapping periods**, not compounded returns.

**Concentration reduces performance** rather than improving it, which is consistent with a weak ranking signal.

**Data scope.** Daily bars only: no intraday queue priority, order-book depth, or hidden liquidity. No fundamentals, macro, or news. Foreign ownership room is not modeled.

**Paper trading runs on a different data source** (vnstock) than the research pipeline (FiinQuant) and has not been reconciled against it.

## 11. Conclusion

The results do not support an alpha claim. One horizon out of three is after-cost positive, no feature configuration clears cost at the 5-day baseline horizon, a single-feature momentum rule outperforms every trained model on a matched basis, and the most recent feature addition did not clear its pre-registered criterion.

The open question is why a single trailing-return sort outperforms a 99-feature gradient-boosting model that already receives that feature as an input. Until that is answered, additional features are unlikely to help, and each further configuration tested against the same history increases the multiple-comparisons problem described in section 10.
