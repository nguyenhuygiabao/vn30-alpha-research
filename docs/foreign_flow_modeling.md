# Foreign-flow modeling protocol

This experiment tests whether FiinQuant foreign buy (`fb`) and foreign sell
(`fs`) data add cross-sectional predictive value without changing the
production strategy.

## Fixed design

- Forecast horizon: 10 trading days
- Model: the existing gradient-boosting regressor with frozen parameters
- Universe: verified point-in-time VN30 membership
- Labels: leave-one-out relative returns built from canonical adjusted prices
- Initial training history: 504 trading dates
- Purge: 10 trading dates
- Test block: 126 trading dates
- Portfolio diagnostic: top eight stocks

The feature comparison is fixed before the holdout is opened:

| Feature set | Included features |
|---|---|
| `market_baseline` | Existing non-foreign model features |
| `frozen_foreign_flow` | Baseline plus the nine previously frozen net/gross foreign-flow features |
| `fb_fs` | Baseline plus 14 separate foreign-buy and foreign-sell features |
| `frozen_plus_fb_fs` | Baseline plus all 23 approved foreign-flow features |

Raw `fb`, `fs`, and `fn` values and intermediate rolling sums are not admitted
directly to the model.

## Missing data

Missing foreign flow is never replaced with zero. Every feature set is trained
and evaluated on the same finite, complete-case rows. A signal date is evaluated
only when every constituent row on that date is complete for the union of all
compared features.

Observed zeros remain valid observations.

## Leakage controls

- Features use information available on or before the signal-date close.
- The intended execution remains the next trading session.
- Forward labels are built from adjusted prices after point-in-time eligibility
  is established.
- Every test block is separated from its training history by at least the
  10-day forecast horizon.
- Hyperparameters and feature definitions are not tuned during this comparison.

## Two phases

The default command runs development data only:

```powershell
py .\scripts\run_foreign_flow_experiment.py
```

The development period ends on 31 December 2023. The 2024 onward holdout must
remain unopened until one foreign-flow candidate is frozen.

Running the experiment does not change strategy configuration, paper targets,
paper orders, or real orders.
