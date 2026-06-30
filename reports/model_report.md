# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the current dataset is still a tiny toy sample. The numbers below validate the evaluation pipeline, but they should not be treated as real market evidence.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
gradient_boosting                1         0.866025               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
      elastic_net                1         0.500000               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
    random_forest                1         0.500000               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
            ridge                1         0.500000               1.0                              0.0                    0.016671                   0.016671           NaN           0.0               NaN                       0.0
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
       model_name                feature  average_importance  importance_observations
gradient_boosting      positive_shock_1d            0.610226                        1
gradient_boosting              return_1d            0.389774                        1
gradient_boosting distance_from_20d_high            0.000000                        1
gradient_boosting  distance_from_20d_low            0.000000                        1
gradient_boosting      negative_shock_1d            0.000000                        1
gradient_boosting             return_10d            0.000000                        1
gradient_boosting             return_20d            0.000000                        1
gradient_boosting              return_3d            0.000000                        1
gradient_boosting              return_5d            0.000000                        1
gradient_boosting             return_60d            0.000000                        1
    random_forest              return_1d            0.578482                        1
    random_forest      positive_shock_1d            0.421518                        1
    random_forest distance_from_20d_high            0.000000                        1
    random_forest  distance_from_20d_low            0.000000                        1
    random_forest      negative_shock_1d            0.000000                        1
    random_forest             return_10d            0.000000                        1
    random_forest             return_20d            0.000000                        1
    random_forest              return_3d            0.000000                        1
    random_forest              return_5d            0.000000                        1
    random_forest             return_60d            0.000000                        1
```

## Provisional model choice

Best provisional regression model: gradient_boosting with average Rank IC 0.866025 and average after-cost top-5 return 0.016671.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
