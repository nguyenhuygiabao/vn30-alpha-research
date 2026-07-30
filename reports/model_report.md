# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the current dataset uses the current VN30 constituent list applied backward through time. The numbers below are real-data walk-forward results, but they are not final market evidence until survivorship bias, liquidity limits, and transaction costs are handled more carefully.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
      elastic_net             2478         0.021737          0.476513                         0.003250                    0.001351                   0.000437      0.342716     -0.714492          0.915301                  0.000915
gradient_boosting             2478         0.019869          0.469492                         0.003010                    0.000998                   0.000006      0.004154     -0.826714          0.992814                  0.000992
    random_forest             2478         0.018015          0.471429                         0.003039                    0.001433                   0.000551      0.405961     -0.738247          0.882519                  0.000882
            ridge             2478         0.017905          0.470056                         0.002297                    0.000786                  -0.000197     -0.157276     -0.858161          0.983448                  0.000983
```

## Classification model comparison

The logistic regression model is evaluated separately because it predicts top-quintile probability, not raw return.

```text
         model_name  evaluated_dates  average_precision  average_recall  average_selected_return  sharpe_ratio  max_drawdown
logistic_regression             2478           0.220823        0.184019                 0.001006      0.799103      -0.55947
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
gradient_boosting                    return_60d            0.059645                     2478
gradient_boosting                      drawdown            0.056539                     2478
gradient_boosting               rolling_vol_20d            0.055500                     2478
gradient_boosting      average_daily_volume_20d            0.043067                     2478
gradient_boosting  foreign_flow_return_corr_20d            0.031661                     2478
gradient_boosting foreign_buy_downday_share_20d            0.027678                     2478
gradient_boosting                    return_10d            0.026065                     2478
gradient_boosting       average_daily_value_20d            0.025843                     2478
gradient_boosting               reference_price            0.024049                     2478
gradient_boosting       estimated_ceiling_price            0.023279                     2478
gradient_boosting         estimated_floor_price            0.022967                     2478
gradient_boosting         distance_from_20d_low            0.021841                     2478
gradient_boosting  foreign_sell_value_share_20d            0.021830                     2478
gradient_boosting                     return_3d            0.020435                     2478
gradient_boosting        distance_from_20d_high            0.020435                     2478
gradient_boosting    foreign_flow_imbalance_20d            0.019109                     2478
gradient_boosting   foreign_net_value_share_20d            0.018927                     2478
gradient_boosting         foreign_net_value_20d            0.018903                     2478
gradient_boosting   foreign_buy_value_share_20d            0.017378                     2478
gradient_boosting                    return_20d            0.016893                     2478
    random_forest                    return_60d            0.059848                     2478
    random_forest               rolling_vol_20d            0.057690                     2478
    random_forest                      drawdown            0.052306                     2478
    random_forest      average_daily_volume_20d            0.042446                     2478
    random_forest  foreign_flow_return_corr_20d            0.030357                     2478
    random_forest                    return_10d            0.027316                     2478
    random_forest foreign_buy_downday_share_20d            0.026637                     2478
    random_forest               reference_price            0.024838                     2478
    random_forest         distance_from_20d_low            0.023660                     2478
    random_forest       estimated_ceiling_price            0.022023                     2478
    random_forest       average_daily_value_20d            0.021891                     2478
    random_forest         estimated_floor_price            0.020907                     2478
    random_forest        distance_from_20d_high            0.020833                     2478
    random_forest  foreign_sell_value_share_20d            0.020291                     2478
    random_forest                     return_3d            0.019990                     2478
    random_forest    foreign_flow_imbalance_20d            0.018910                     2478
    random_forest         foreign_net_value_20d            0.018466                     2478
    random_forest   foreign_net_value_share_20d            0.017446                     2478
    random_forest                    return_20d            0.017050                     2478
    random_forest            rolling_return_20d            0.016980                     2478
```

## Provisional model choice

Best provisional regression model: elastic_net with average Rank IC 0.021737 and average after-cost top-5 return 0.000437.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
