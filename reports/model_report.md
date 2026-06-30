# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the current dataset is still a tiny toy sample. The numbers below validate the evaluation pipeline, but they should not be treated as real market evidence.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
gradient_boosting                1              1.0               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
      elastic_net                1              0.5               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
    random_forest                1              0.5               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
            ridge                1              0.5               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
```

## Classification model comparison

The logistic regression model is evaluated separately because it predicts top-quintile probability, not raw return.

```text
         model_name  evaluated_dates  average_precision  average_recall  average_selected_return  sharpe_ratio  max_drawdown
logistic_regression                1           0.333333             1.0                 0.016671           NaN           0.0
```

## Baseline strategy comparison

The baseline strategy is evaluated as a portfolio return series, not as a ranking model.

```text
        strategy  evaluated_dates  average_portfolio_return  average_active_return  sharpe_ratio  max_drawdown  average_selected_count
equal_weight_all                5                  0.034607               0.018356    511.407292           0.0                     3.0
```

## Tree model feature importance

Feature importance shows which variables the tree models used most often for splits. It is useful for sanity checking, but it is not causal proof.

```text
       model_name                      feature  average_importance  importance_observations
gradient_boosting            distance_to_floor            0.606112                        1
gradient_boosting      estimated_ceiling_price            0.107998                        1
gradient_boosting        estimated_floor_price            0.106009                        1
gradient_boosting              reference_price            0.104628                        1
gradient_boosting            positive_shock_1d            0.019882                        1
gradient_boosting          distance_to_ceiling            0.013878                        1
gradient_boosting                    return_1d            0.008974                        1
gradient_boosting             simple_return_1d            0.008905                        1
gradient_boosting       vn30_return_dispersion            0.008312                        1
gradient_boosting                log_return_1d            0.005993                        1
gradient_boosting herding_low_dispersion_score            0.003138                        1
gradient_boosting            percent_stocks_up            0.003126                        1
gradient_boosting   market_direction_agreement            0.003045                        1
gradient_boosting         abnormal_volume_flag            0.000000                        1
gradient_boosting           amihud_illiquidity            0.000000                        1
gradient_boosting       amihud_illiquidity_raw            0.000000                        1
gradient_boosting      average_daily_value_20d            0.000000                        1
gradient_boosting     average_daily_volume_20d            0.000000                        1
gradient_boosting     consecutive_ceiling_days            0.000000                        1
gradient_boosting       consecutive_floor_days            0.000000                        1
    random_forest            distance_to_floor            0.314858                        1
    random_forest        estimated_floor_price            0.119184                        1
    random_forest          distance_to_ceiling            0.101323                        1
    random_forest            positive_shock_1d            0.100538                        1
    random_forest                    return_1d            0.092712                        1
    random_forest             simple_return_1d            0.086096                        1
    random_forest                log_return_1d            0.060361                        1
    random_forest              reference_price            0.049480                        1
    random_forest      estimated_ceiling_price            0.042271                        1
    random_forest   market_direction_agreement            0.021467                        1
    random_forest herding_low_dispersion_score            0.010101                        1
    random_forest            percent_stocks_up            0.000940                        1
    random_forest       vn30_return_dispersion            0.000669                        1
    random_forest         abnormal_volume_flag            0.000000                        1
    random_forest           amihud_illiquidity            0.000000                        1
    random_forest       amihud_illiquidity_raw            0.000000                        1
    random_forest      average_daily_value_20d            0.000000                        1
    random_forest     average_daily_volume_20d            0.000000                        1
    random_forest     consecutive_ceiling_days            0.000000                        1
    random_forest       consecutive_floor_days            0.000000                        1
```

## Provisional model choice

Best provisional regression model: gradient_boosting with average Rank IC 1.000000 and average after-cost top-5 return 0.016671.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
