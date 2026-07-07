# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the current dataset uses the current VN30 constituent list applied backward through time. The numbers below are real-data walk-forward results, but they are not final market evidence until survivorship bias, liquidity limits, and transaction costs are handled more carefully.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
            ridge             1599         0.023182          0.459037                         0.003872                    0.001435                   0.000519      0.348807     -0.835371          0.916896                  0.000916
    random_forest             1599         0.018807          0.465791                         0.003208                    0.001861                   0.000986      0.644320     -0.746576          0.875594                  0.000875
gradient_boosting             1599         0.018626          0.460913                         0.002203                    0.001259                   0.000289      0.201684     -0.775049          0.970713                  0.000970
      elastic_net             1599         0.016393          0.454784                         0.002995                    0.000963                   0.000088      0.059139     -0.843263          0.875594                  0.000875
```

## Classification model comparison

The logistic regression model is evaluated separately because it predicts top-quintile probability, not raw return.

```text
         model_name  evaluated_dates  average_precision  average_recall  average_selected_return  sharpe_ratio  max_drawdown
logistic_regression             1599            0.23127        0.192725                 0.001857      1.302348     -0.548176
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
gradient_boosting               return_60d            0.107505                     1599
gradient_boosting          rolling_vol_20d            0.102326                     1599
gradient_boosting                 drawdown            0.099091                     1599
gradient_boosting average_daily_volume_20d            0.092138                     1599
gradient_boosting  average_daily_value_20d            0.057202                     1599
gradient_boosting  estimated_ceiling_price            0.048085                     1599
gradient_boosting               return_10d            0.047719                     1599
gradient_boosting    estimated_floor_price            0.046702                     1599
gradient_boosting    distance_from_20d_low            0.046522                     1599
gradient_boosting          reference_price            0.043771                     1599
gradient_boosting   distance_from_20d_high            0.037260                     1599
gradient_boosting                return_3d            0.036399                     1599
gradient_boosting    traded_value_rank_20d            0.032258                     1599
gradient_boosting       rolling_return_20d            0.029215                     1599
gradient_boosting               return_20d            0.028073                     1599
gradient_boosting         volume_change_5d            0.025262                     1599
gradient_boosting        rolling_return_5d            0.021963                     1599
gradient_boosting                return_5d            0.020410                     1599
gradient_boosting              volume_z_20            0.017105                     1599
gradient_boosting        value_traded_z_20            0.015663                     1599
    random_forest               return_60d            0.109820                     1599
    random_forest          rolling_vol_20d            0.098646                     1599
    random_forest                 drawdown            0.095517                     1599
    random_forest average_daily_volume_20d            0.085095                     1599
    random_forest               return_10d            0.050762                     1599
    random_forest  estimated_ceiling_price            0.047748                     1599
    random_forest    distance_from_20d_low            0.046378                     1599
    random_forest  average_daily_value_20d            0.046180                     1599
    random_forest    estimated_floor_price            0.045171                     1599
    random_forest          reference_price            0.044823                     1599
    random_forest   distance_from_20d_high            0.038298                     1599
    random_forest                return_3d            0.037413                     1599
    random_forest    traded_value_rank_20d            0.032793                     1599
    random_forest               return_20d            0.030567                     1599
    random_forest       rolling_return_20d            0.030559                     1599
    random_forest         volume_change_5d            0.027716                     1599
    random_forest        rolling_return_5d            0.024735                     1599
    random_forest                return_5d            0.022879                     1599
    random_forest              volume_z_20            0.018234                     1599
    random_forest        value_traded_z_20            0.017617                     1599
```

## Provisional model choice

Best provisional regression model: ridge with average Rank IC 0.023182 and average after-cost top-5 return 0.000519.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
