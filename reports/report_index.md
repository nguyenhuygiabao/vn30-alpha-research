# Report Index

Links to the main outputs of the VN30 Alpha Research Framework.

## Start here

- [Report Page](reportpage.html) — headline results, before/after comparisons, and the training-window sweep. Generated from `reports/tables/` on every pipeline run.
- [Final Results](final_results.md) — main findings, backtest results, horizon and ablation tests, baseline comparison, limitations.
- [Methodology](methodology.md) — universe, data contract, labels, features, models, validation, optimizer, costs, bias controls.
- [Audit](final_audit.md) — automated audit output, test suite, reproducibility and consistency checks, known gaps.

## Supporting reports

- [Model Report](model_report.md) — model evaluation detail.
- [Data Quality Report](data_quality_report.md) — data coverage and quality checks.
- [Foreign Flow Checkpoint](../docs/enhanced_fb_fs_checkpoint.md) — foreign buy/sell experiment, pre-registered decision rule, and result.
- [FiinQuant Core Checkpoint](../docs/fiinquant_core_checkpoint.md) — data-path scope and contract.
- [Foreign Flow Modeling](../docs/foreign_flow_modeling.md) — frozen comparison design.

## Result tables

- [Horizon Results](tables/horizon_results.csv) — forecast-horizon comparison.
- [Ablation Results](tables/ablation_results.csv) — feature-ablation comparison.
- [Benchmark Results](tables/benchmark_results.csv) — ML strategies against equal-weight and rule-based baselines.
- [Model Comparison](tables/model_comparison_results.csv) — Rank IC and top-5 diagnostics by model family.
- [Concentration Summary](tables/concentration_summary.csv) — max single-name weight, HHI, effective position count, top issuer-group exposure.
- [Issuer Group Exposure](tables/issuer_group_exposure_latest.csv) — latest issuer-group exposure by optimization mode.
- [Latest Rank Diagnostic](tables/latest_rank_diagnostic.csv) — predicted versus realized rank by ticker.
- [Horizon Sample Disclosure](tables/horizon_sample_disclosure.csv) — overlapping-window disclosure and non-overlapping period counts.
- [Optimizer Bound Diagnostic](tables/optimizer_bound_diagnostic.csv) — which constraints bind, and how often.
- [Execution Coverage Summary](tables/execution_coverage_summary.csv) — per-date execution-data availability.
- [Execution Coverage Gaps](tables/execution_coverage_gaps.csv) — per-ticker-date detail for rows not on full real data.
- [Foreign Flow Tables](tables/foreign_flow/) — per-feature-set summary, paired bootstrap differences, daily series, feature importance.

## Main figures

- [Gradient Boosting Feature Importance](figures/top_gradient_boosting_feature_importance.png)
- [Ablation Diagnostic Sharpe](figures/ablation_diagnostic_sharpe.png)
- [Horizon Diagnostic Sharpe](figures/horizon_diagnostic_sharpe.png)
- [Horizon Rank IC](figures/horizon_rank_ic.png)


## Current summary

Point-in-time VN30 membership across 60 historical constituents, FiinQuant raw and adjusted price data, and foreign-flow data, evaluated with purged walk-forward validation and after transaction costs and price-limit execution rules.

**The strategy is not deployable over this sample.** Compounding
non-overlapping 5-day periods for the production configuration gives a final
equity between 0.73 and 1.39 depending only on which of the five overlap
phases the curve starts on, with a maximum drawdown of 25 to 43 percent of
capital. The per-period figures below are correct; they simply do not survive
being compounded as a portfolio. See section 12 of the final results.

Of the three tested horizons, the 5-day and 10-day horizons are after-cost
positive and the 1-day horizon is not. Two of five feature-ablation
configurations clear transaction costs. A single-feature trailing-return rule
(`top5_momentum`, 0.002264 before cost) outperforms every trained model's
top-5 basket on a matched basis, the best of which returns 0.001992. Foreign
buy/sell features did not clear their pre-registered confidence-interval
criterion and are not promoted.

These figures reflect a walk-forward training window of 63 trading dates.

This is a research framework, not a live trading strategy.

## Local viewing commands

```powershell
code .\reports\final_results.md
code .\reports\methodology.md
code .\reports\final_audit.md
code .\reports\report_index.md
explorer .\reports\figures
explorer .\reports\tables
```
