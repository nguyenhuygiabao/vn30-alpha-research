# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the current dataset uses the current VN30 constituent list applied backward through time. The numbers below are real-data walk-forward results, but they are not final market evidence until survivorship bias, liquidity limits, and transaction costs are handled more carefully.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
            ridge             1611         0.023545          0.459342                         0.003888                    0.001424                   0.000507      0.341657     -0.835371          0.917267                  0.000917
    random_forest             1611         0.019393          0.466046                         0.003272                    0.001878                   0.001003      0.656812     -0.746576          0.875776                  0.000875
gradient_boosting             1611         0.018370          0.460832                         0.002153                    0.001231                   0.000263      0.183613     -0.775049          0.969441                  0.000969
      elastic_net             1611         0.017069          0.455990                         0.003089                    0.001012                   0.000137      0.092273     -0.843263          0.874783                  0.000874
```

## Classification model comparison

The logistic regression model is evaluated separately because it predicts top-quintile probability, not raw return.

```text
         model_name  evaluated_dates  average_precision  average_recall  average_selected_return  sharpe_ratio  max_drawdown
logistic_regression             1611           0.231161        0.192634                 0.001841      1.293504     -0.548176
```

## Baseline strategy comparison

The baseline strategy is evaluated as a portfolio return series, not as a ranking model.

```text
            strategy  evaluated_dates  average_portfolio_return  average_active_return  sharpe_ratio  max_drawdown  average_selected_count
       top5_momentum             1615                  0.007656               0.003589      3.034853     -0.873921                   5.000
    equal_weight_all             1625                  0.004285              -0.000000      2.068608     -0.904494                  28.968
       top5_reversal             1606                  0.004250              -0.000072      1.508461     -0.948201                   5.000
low_volatility_top10             1605                  0.003401              -0.000939      1.888739     -0.835925                  10.000
```

## Tree model feature importance

Feature importance shows which variables the tree models used most often for splits. It is useful for sanity checking, but it is not causal proof.

```text
       model_name                  feature  average_importance  importance_observations
gradient_boosting               return_60d            0.108622                     1611
gradient_boosting          rolling_vol_20d            0.102749                     1611
gradient_boosting                 drawdown            0.098845                     1611
gradient_boosting average_daily_volume_20d            0.091727                     1611
gradient_boosting  average_daily_value_20d            0.057326                     1611
gradient_boosting               return_10d            0.047919                     1611
gradient_boosting  estimated_ceiling_price            0.047787                     1611
gradient_boosting    estimated_floor_price            0.046434                     1611
gradient_boosting    distance_from_20d_low            0.046321                     1611
gradient_boosting          reference_price            0.043511                     1611
gradient_boosting   distance_from_20d_high            0.037639                     1611
gradient_boosting                return_3d            0.036278                     1611
gradient_boosting    traded_value_rank_20d            0.032276                     1611
gradient_boosting       rolling_return_20d            0.029197                     1611
gradient_boosting               return_20d            0.028032                     1611
gradient_boosting         volume_change_5d            0.025120                     1611
gradient_boosting        rolling_return_5d            0.021869                     1611
gradient_boosting                return_5d            0.020310                     1611
gradient_boosting              volume_z_20            0.017142                     1611
gradient_boosting        value_traded_z_20            0.015720                     1611
    random_forest               return_60d            0.111017                     1611
    random_forest          rolling_vol_20d            0.099279                     1611
    random_forest                 drawdown            0.095236                     1611
    random_forest average_daily_volume_20d            0.084676                     1611
    random_forest               return_10d            0.051079                     1611
    random_forest  estimated_ceiling_price            0.047477                     1611
    random_forest    distance_from_20d_low            0.046202                     1611
    random_forest  average_daily_value_20d            0.046178                     1611
    random_forest    estimated_floor_price            0.044893                     1611
    random_forest          reference_price            0.044548                     1611
    random_forest   distance_from_20d_high            0.038587                     1611
    random_forest                return_3d            0.037266                     1611
    random_forest    traded_value_rank_20d            0.032774                     1611
    random_forest               return_20d            0.030537                     1611
    random_forest       rolling_return_20d            0.030495                     1611
    random_forest         volume_change_5d            0.027652                     1611
    random_forest        rolling_return_5d            0.024613                     1611
    random_forest                return_5d            0.022758                     1611
    random_forest              volume_z_20            0.018206                     1611
    random_forest        value_traded_z_20            0.017601                     1611
```

## Provisional model choice

Best provisional regression model: ridge with average Rank IC 0.023545 and average after-cost top-5 return 0.000507.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
