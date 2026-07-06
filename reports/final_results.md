# Final Results

## 1. Executive Summary

This project builds a VN30 alpha research framework for cross-sectional stock selection. The framework ranks VN30 stocks using daily OHLCV-based features, converts predictions into constrained portfolio weights, and evaluates performance after transaction costs, liquidity penalties, turnover limits, issuer-group caps, and Vietnam-specific price-limit execution rules.

The final research conclusion is that medium-horizon stock selection is more promising than very short-horizon prediction in this dataset. The 1-day horizon performed poorly and appeared close to noise. The 5-day horizon performed strongly and remains the original baseline. The 10-day horizon produced the strongest tested results across prediction quality, after-cost performance, drawdown, turnover, and diagnostic Sharpe.

The project should be interpreted as a research framework, not a live trading system. Because the backtest uses overlapping forward returns, cumulative active returns and diagnostic Sharpe values should be read as research diagnostics rather than final investable performance statistics.

## 2. Main Backtest Results

The main portfolio backtest compares two optimizer modes and two execution modes.

The optimizer modes are:

- `normal`: standard constrained long-only optimizer.
- `herding_aware`: reduced exposure and concentration during high-herding market states.

The execution modes are:

- `normal`: assumes intended trades can be executed.
- `price_limit_aware`: blocks unrealistic buys at ceiling closes and sells at floor closes.

| Optimizer Mode | Execution Mode | Diagnostic Sharpe | Max Active Drawdown | Average Turnover | Final Cumulative After-Cost Active Return |
|---|---:|---:|---:|---:|---:|
| normal | normal | 0.873634 | -0.278142 | 0.357871 | 32.377818 |
| normal | price_limit_aware | 0.860436 | -0.370818 | 0.353277 | 31.487395 |
| herding_aware | normal | 0.872999 | -0.278142 | 0.351899 | 31.703161 |
| herding_aware | price_limit_aware | 0.860766 | -0.370818 | 0.347622 | 30.885591 |

The normal optimizer produced the highest final cumulative after-cost active return. The herding-aware optimizer slightly reduced turnover and concentration but gave up some return. Price-limit-aware execution reduced performance compared with normal execution, which is expected because it makes the backtest more conservative and realistic.

The strongest pure-return configuration was:

`normal optimizer + normal execution`

The more realistic Vietnam-friction-aware configuration was:

`normal optimizer + price_limit_aware execution`

For final interpretation, the price-limit-aware result is more useful because it accounts for Vietnam’s daily ceiling and floor trading constraints.

## 3. Forecast Horizon Results

The forecast horizon test compares 1-day, 5-day, and 10-day forward relative-return labels.

| Forecast Horizon | Average Rank IC | Top-5 Hit Rate | Top-5 Average Actual Return | Diagnostic Sharpe | Max Active Drawdown | Average Turnover | Final Cumulative After-Cost Active Return |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d | -0.003844 | 0.447861 | -0.000067 | -0.140805 | -1.525574 | 0.389600 | -1.508238 |
| 5d | 0.338033 | 0.676072 | 0.023634 | 0.840073 | -0.355852 | 0.354637 | 29.360326 |
| 10d | 0.546504 | 0.801122 | 0.052681 | 1.398924 | -0.219472 | 0.317395 | 77.915433 |

The 10-day horizon is the strongest tested horizon. It produced the best rank IC, best hit rate, highest diagnostic Sharpe, lowest drawdown, lowest average turnover, and highest final cumulative after-cost active return.

The 1-day horizon performed poorly. This suggests that daily next-day stock selection inside the VN30 universe is too noisy for this feature set and model design.

The 5-day horizon remains useful as the original baseline because it is more responsive than 10d while still producing strong predictive and portfolio results. However, the evidence from the horizon experiment favors 10d as the best-tested forecast horizon.

## 4. Feature Ablation Results

Feature ablation removes one feature group at a time and compares the resulting model and portfolio diagnostics. This checks whether the model depends too heavily on a single type of signal.

| Ablation Setting | Feature Count | Average Rank IC | Top-5 Hit Rate | Top-5 Average Actual Return | Diagnostic Sharpe | Max Active Drawdown | Average Turnover | Final Cumulative After-Cost Active Return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_features | 51 | 0.338033 | 0.676072 | 0.023634 | 0.840073 | -0.355852 | 0.354637 | 29.360326 |
| without_herding | 41 | 0.337983 | 0.676321 | 0.023677 | 0.827557 | -0.355852 | 0.356006 | 29.281588 |
| without_price_limit | 36 | 0.316762 | 0.666128 | 0.022343 | 0.784717 | -0.359316 | 0.361541 | 27.268516 |
| without_risk | 49 | 0.314453 | 0.657303 | 0.021847 | 0.749285 | -0.363631 | 0.365078 | 26.301430 |
| without_volume_liquidity | 40 | 0.307164 | 0.655065 | 0.021507 | 0.742006 | -0.322599 | 0.361687 | 26.341446 |

The full feature set produced the strongest overall result. Removing herding features had only a small effect on raw prediction quality, which suggests that herding features are not the main source of alpha in the model.

The largest performance drops came from removing volume/liquidity, price-limit, and risk features. This is important because these feature groups are closely connected to Vietnam-specific market behavior. The result supports the idea that liquidity, price limits, and risk conditions are central to VN30 stock-selection performance.

## 5. Selected Model and Configuration

The selected predictive model is:

`gradient_boosting`

The selected research horizon is:

`10-day forward relative return`

The selected realistic execution setting is:

`price_limit_aware`

The selected portfolio construction style is:

`normal constrained long-only optimizer`

This combination is selected because the 10-day horizon produced the strongest tested prediction and portfolio metrics, while the normal optimizer preserved stronger return generation than the herding-aware optimizer. Price-limit-aware execution is preferred for final interpretation because it reflects Vietnam-specific trading constraints more realistically than the normal execution setting.

The herding-aware optimizer remains useful as a risk-control extension. It successfully reduces exposure and concentration during high-herding regimes, but in this experiment it did not improve final return or drawdown enough to replace the normal optimizer as the main configuration.

## 6. Main Figures

The final report figures are stored in:

`reports/figures/`

The key figures are:

- `top_gradient_boosting_feature_importance.png`
- `ablation_diagnostic_sharpe.png`
- `horizon_diagnostic_sharpe.png`
- `horizon_rank_ic.png`

Together, these figures show portfolio performance, drawdown behavior, turnover, stability over time, feature importance, feature ablation, and forecast-horizon comparison.

## 7. Failure Cases and Weaknesses

The clearest failure case is the 1-day forecast horizon. It produced a negative average rank IC, weak hit rate, negative after-cost performance, and poor diagnostic Sharpe. This suggests that next-day prediction is too noisy for this public daily-data framework.

Another weakness is that herding features did not materially improve raw predictive performance. Herding remains conceptually useful as a portfolio risk-control idea, but the ablation test shows that it is not a major standalone alpha source in this model.

The price-limit-aware execution test also shows that normal backtests can be too optimistic. Once ceiling and floor execution restrictions are added, performance declines. This is not a bug; it is evidence that Vietnam-specific execution frictions matter.

The project also has a static-universe limitation. It uses the current VN30 universe rather than a fully point-in-time historical constituent list. This makes the experiment cleaner and easier to reproduce, but it may introduce survivorship bias.

Finally, the framework uses daily OHLCV data only. It does not include fundamentals, foreign ownership flows, proprietary order-book data, macro variables, earnings revisions, or news sentiment. Therefore, the model cannot capture every important driver of Vietnamese equity returns.

## 8. Interpretation

The main result is not simply that one model has a high backtest number. The stronger conclusion is methodological: a Vietnam-specific ML stock-selection framework needs more than standard return features.

The most useful parts of the framework are:

- Cross-sectional relative-return labels
- Walk-forward validation
- Liquidity and price-limit features
- Transaction-cost-aware backtesting
- Price-limit-aware execution rules
- Issuer-group concentration caps
- Forecast-horizon testing
- Feature ablation testing

The 10-day result suggests that the model needs enough time for signals to play out. The poor 1-day result suggests that daily noise, market microstructure, and short-term execution frictions dominate very short-horizon prediction.

## 9. Final Conclusion

The final framework supports the idea that medium-horizon VN30 stock selection can be studied with a disciplined machine learning pipeline using public daily data. The 10-day forecast horizon is the strongest tested horizon, while the 5-day horizon remains a strong baseline.

The project does not claim to be a deployable trading strategy yet. Its value is that it builds a transparent, reproducible, and Vietnam-aware research pipeline that connects data, features, labels, models, validation, portfolio construction, execution frictions, robustness testing, and final reporting.

Future improvements should focus on point-in-time VN30 membership, non-overlapping portfolio evaluation, more realistic order execution, fundamental data, ownership-flow data, and stronger out-of-sample robustness testing.
