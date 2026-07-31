# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the numbers below are real-data walk-forward results on a point-in-time VN30 universe, with transaction costs and price-limit execution rules applied. They are still research diagnostics rather than final market evidence: cumulative figures sum overlapping forecast-period returns, metrics other than the foreign-flow comparison are point estimates without confidence intervals, and multiple horizons, feature sets, and optimizer variants were tested against the same history.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
      elastic_net             2418         0.036697          0.490157                         0.004160                    0.001992                   0.001409      1.076748     -0.808193          0.583368                  0.000583
            ridge             2418         0.036075          0.485277                         0.003742                    0.001478                   0.000779      0.628199     -0.879017          0.699545                  0.000699
gradient_boosting             2418         0.033360          0.481307                         0.002682                    0.001073                   0.000444      0.352768     -0.783934          0.628713                  0.000628
    random_forest             2418         0.031098          0.480562                         0.002669                    0.001076                   0.000507      0.385638     -0.866005          0.568804                  0.000569
```

## Classification model comparison

The logistic regression model is evaluated separately because it predicts top-quintile probability, not raw return.

```text
         model_name  evaluated_dates  average_precision  average_recall  average_selected_return  sharpe_ratio  max_drawdown
logistic_regression             2418           0.241522        0.201268                 0.001811      1.372052     -0.707662
```

## Baseline strategy comparison

The baseline strategy is evaluated as a portfolio return series, not as a ranking model.

```text
            strategy  evaluated_dates  average_portfolio_return  average_active_return  sharpe_ratio  max_drawdown  average_selected_count
       top5_momentum             2482                  0.004604               0.002294      2.071568     -0.902732                5.000000
    equal_weight_all             2492                  0.002385              -0.000000      1.288258     -0.962741               29.995987
low_volatility_top10             2472                  0.002013              -0.000347      1.232174     -0.930533               10.000000
       top5_reversal             2473                 -0.000193              -0.002630     -0.074396     -0.997989                5.000000
```

## Tree model feature importance

Feature importance shows which variables the tree models used most often for splits. It is useful for sanity checking, but it is not causal proof.

```text
       model_name                       feature  average_importance  importance_observations
gradient_boosting                    return_60d            0.077856                     2418
gradient_boosting                      drawdown            0.075066                     2418
gradient_boosting      average_daily_volume_20d            0.057009                     2418
gradient_boosting               rolling_vol_20d            0.056798                     2418
gradient_boosting       average_daily_value_20d            0.053787                     2418
gradient_boosting       estimated_ceiling_price            0.028252                     2418
gradient_boosting               reference_price            0.027500                     2418
gradient_boosting    foreign_flow_imbalance_20d            0.026892                     2418
gradient_boosting         estimated_floor_price            0.026586                     2418
gradient_boosting                    return_10d            0.026407                     2418
gradient_boosting  foreign_flow_return_corr_20d            0.024516                     2418
gradient_boosting        foreign_sell_value_20d            0.024488                     2418
gradient_boosting   foreign_net_value_share_20d            0.023592                     2418
gradient_boosting         foreign_net_value_20d            0.022605                     2418
gradient_boosting foreign_buy_downday_share_20d            0.021294                     2418
gradient_boosting  foreign_sell_value_share_20d            0.020494                     2418
gradient_boosting         foreign_buy_value_20d            0.019834                     2418
gradient_boosting                     return_3d            0.018742                     2418
gradient_boosting   foreign_buy_value_share_20d            0.018548                     2418
gradient_boosting        distance_from_20d_high            0.018308                     2418
    random_forest                    return_60d            0.083584                     2418
    random_forest                      drawdown            0.071677                     2418
    random_forest      average_daily_volume_20d            0.057549                     2418
    random_forest               rolling_vol_20d            0.054459                     2418
    random_forest       average_daily_value_20d            0.054437                     2418
    random_forest        value_traded_estimated            0.030826                     2418
    random_forest       estimated_ceiling_price            0.028220                     2418
    random_forest               reference_price            0.028056                     2418
    random_forest                    return_10d            0.026292                     2418
    random_forest    foreign_flow_imbalance_20d            0.025565                     2418
    random_forest         estimated_floor_price            0.025190                     2418
    random_forest                     return_3d            0.025032                     2418
    random_forest        foreign_sell_value_20d            0.023177                     2418
    random_forest         foreign_net_value_20d            0.022656                     2418
    random_forest   foreign_net_value_share_20d            0.022501                     2418
    random_forest        distance_from_20d_high            0.021314                     2418
    random_forest  foreign_sell_value_share_20d            0.019135                     2418
    random_forest  foreign_flow_return_corr_20d            0.017279                     2418
    random_forest         foreign_buy_value_20d            0.017232                     2418
    random_forest foreign_buy_downday_share_20d            0.016634                     2418
```

## Provisional model choice

Best provisional regression model: elastic_net with average Rank IC 0.036697 and average after-cost top-5 return 0.001409.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
