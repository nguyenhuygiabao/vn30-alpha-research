# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the current dataset uses the current VN30 constituent list applied backward through time. The numbers below are real-data walk-forward results, but they are not final market evidence until survivorship bias, liquidity limits, and transaction costs are handled more carefully.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
gradient_boosting             1609         0.353113          0.690367                         0.045097                    0.025058                   0.024153     15.220779     -0.208974          0.905473                  0.000905
    random_forest             1609         0.325848          0.666377                         0.041925                    0.022929                   0.022126     13.677701     -0.179925          0.803483                  0.000803
      elastic_net             1609         0.268957          0.623369                         0.034363                    0.018691                   0.017797     11.874938     -0.172125          0.894527                  0.000894
            ridge             1609         0.261638          0.620758                         0.033692                    0.018673                   0.017722     11.618704     -0.164531          0.951990                  0.000951
```

## Classification model comparison

The logistic regression model is evaluated separately because it predicts top-quintile probability, not raw return.

```text
         model_name  evaluated_dates  average_precision  average_recall  average_selected_return  sharpe_ratio  max_drawdown
logistic_regression             1609           0.356495        0.297079                 0.015283     10.526802     -0.254114
```

## Baseline strategy comparison

The baseline strategy is evaluated as a portfolio return series, not as a ranking model.

```text
            strategy  evaluated_dates  average_portfolio_return  average_active_return  sharpe_ratio  max_drawdown  average_selected_count
       top5_momentum             1603                  0.007818               0.003623      3.092889     -0.873921                5.000000
    equal_weight_all             1613                  0.004414              -0.000000      2.126402     -0.904494               28.960322
       top5_reversal             1594                  0.004323              -0.000131      1.530188     -0.948201                5.000000
low_volatility_top10             1593                  0.003499              -0.000972      1.941379     -0.835925               10.000000
```

## Tree model feature importance

Feature importance shows which variables the tree models used most often for splits. It is useful for sanity checking, but it is not causal proof.

```text
       model_name                  feature  average_importance  importance_observations
gradient_boosting               return_60d            0.098324                     1609
gradient_boosting          rolling_vol_20d            0.092698                     1609
gradient_boosting                 drawdown            0.088177                     1609
gradient_boosting average_daily_volume_20d            0.078365                     1609
gradient_boosting          reference_price            0.075907                     1609
gradient_boosting        distance_to_floor            0.050169                     1609
gradient_boosting  average_daily_value_20d            0.049755                     1609
gradient_boosting               return_10d            0.042370                     1609
gradient_boosting    distance_from_20d_low            0.040775                     1609
gradient_boosting   consecutive_floor_days            0.036552                     1609
gradient_boosting   distance_from_20d_high            0.034053                     1609
gradient_boosting                return_3d            0.032757                     1609
gradient_boosting consecutive_ceiling_days            0.032026                     1609
gradient_boosting      distance_to_ceiling            0.031530                     1609
gradient_boosting       rolling_return_20d            0.027583                     1609
gradient_boosting    traded_value_rank_20d            0.024911                     1609
gradient_boosting               return_20d            0.023805                     1609
gradient_boosting         volume_change_5d            0.021396                     1609
gradient_boosting        rolling_return_5d            0.018879                     1609
gradient_boosting                return_5d            0.018426                     1609
    random_forest               return_60d            0.099604                     1609
    random_forest          rolling_vol_20d            0.089469                     1609
    random_forest                 drawdown            0.083978                     1609
    random_forest          reference_price            0.073319                     1609
    random_forest average_daily_volume_20d            0.072162                     1609
    random_forest        distance_to_floor            0.053167                     1609
    random_forest               return_10d            0.045184                     1609
    random_forest    distance_from_20d_low            0.042080                     1609
    random_forest  average_daily_value_20d            0.039424                     1609
    random_forest   distance_from_20d_high            0.034615                     1609
    random_forest                return_3d            0.033217                     1609
    random_forest   consecutive_floor_days            0.032918                     1609
    random_forest      distance_to_ceiling            0.032779                     1609
    random_forest consecutive_ceiling_days            0.030282                     1609
    random_forest               return_20d            0.029610                     1609
    random_forest    traded_value_rank_20d            0.028040                     1609
    random_forest       rolling_return_20d            0.025015                     1609
    random_forest         volume_change_5d            0.023429                     1609
    random_forest        rolling_return_5d            0.022281                     1609
    random_forest                return_5d            0.019775                     1609
```

## Provisional model choice

Best provisional regression model: gradient_boosting with average Rank IC 0.353113 and average after-cost top-5 return 0.024153.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
