# Report Index

This page links the main outputs of the VN30 Long-Only Machine Learning Framework.

## Start here

- [Final Results](final_results.md): main findings, backtest results, horizon tests, ablation tests, figures, limitations, and conclusion.
- [Methodology](methodology.md): data, labels, features, models, validation, optimizer, backtester, transaction costs, and bias controls.
- [Final Audit](final_audit.md): final repo state, script reruns, verification checks, and artifact audit.

## Supporting reports

- [Model Report](model_report.md): model evaluation details.
- [Data Quality Report](data_quality_report.md): data coverage and quality checks.

## Result tables

- [Ablation Results](tables/ablation_results.csv): feature-ablation results.
- [Horizon Results](tables/horizon_results.csv): forecast-horizon results.

## Main figures

- [Cumulative After-Cost Active Return](figures/cumulative_after_cost_active_return.png)
- [Active Drawdown](figures/active_drawdown.png)
- [Portfolio Turnover](figures/portfolio_turnover.png)
- [Rolling Diagnostic Sharpe](figures/rolling_diagnostic_sharpe.png)
- [Top Gradient Boosting Feature Importance](figures/top_gradient_boosting_feature_importance.png)
- [Ablation Diagnostic Sharpe](figures/ablation_diagnostic_sharpe.png)
- [Horizon Diagnostic Sharpe](figures/horizon_diagnostic_sharpe.png)
- [Horizon Rank IC](figures/horizon_rank_ic.png)

## Current conclusion

The project should be presented as a transparent research framework, not a live trading strategy.

Current evidence suggests that the 10-day forward relative-return label is the strongest tested horizon, while the 5-day label remains a strong baseline. The full feature set performs best overall. Price-limit, risk, volume, and liquidity features matter more than herding features for raw predictive performance, while herding remains useful as portfolio-level risk control.

## Local viewing commands

- code .\reports\final_results.md
- code .\reports\methodology.md
- code .\reports\final_audit.md
- code .\reports\report_index.md
- explorer .\reports\figures
- explorer .\reports\tables
