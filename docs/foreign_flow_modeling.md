# Foreign-flow modeling rebuild

This experiment tests whether FiinQuant foreign buy (`fb`) and foreign sell
(`fs`) features add cross-sectional predictive value without changing the
production strategy.

## Frozen comparison

- Forecast horizon: 10 common exchange sessions
- Signal spacing: 10 exchange sessions
- Training frame: expanding history
- Initial training history: 252 exchange sessions
- Model: the existing production Gradient Boosting regressor
- Missing-data handling: the existing tree-model imputation pipeline
- Universe: every verified historical VN30 member, restricted by
  point-in-time membership on each signal date
- Labels: leave-one-out relative returns from canonical adjusted prices
- Portfolio diagnostic: top eight stocks

The ticker pool and feature count are different concepts. The historical pool
can contain any number of former and current VN30 members. The market baseline
contains exactly 51 model features for each eligible ticker-date row.

| Feature set | Feature count | Included features |
|---|---:|---|
| `market_baseline` | 51 | Explicit frozen market feature list |
| `frozen_foreign_flow` | 60 | Baseline plus the original nine net/gross flow features |
| `fb_fs` | 64 | Baseline plus 14 separate foreign-buy and foreign-sell features |
| `frozen_plus_fb_fs` | 73 | Baseline plus all 23 approved foreign-flow features |

Raw `fb`, `fs`, and `fn` values, availability flags, and intermediate rolling
sums are not admitted directly to the model.

## Corrected missing-data rule

Feature missingness does not remove a ticker-date row. Missing values remain
missing in the stored modeling frame and are handled inside the existing
training pipeline. Only rows without a valid target are excluded.

This corrects the earlier complete-case experiment, which removed about 41.2%
of the canonical rows and changed the baseline sample.

## Calendar and leakage controls

- A 10-day return ends on the exact tenth future exchange date.
- If a ticker lacks that future exchange session, its target remains missing.
  The label does not jump to the ticker's next observed session.
- Membership is applied at the signal date.
- All forward-return columns are excluded from every model input.
- The training cutoff is ten exchange sessions before each signal.
- Every signal is independently refitted using expanding history.
- All four feature sets use identical training and evaluation rows.

## Baseline parity gate

Before the full comparison, the 51-feature baseline is run through both the
rebuilt path and the established production paper-horizon path for one real
historical signal. The signal keeps its complete expanding training history.
This bounded gate avoids rerunning every production model over the entire
history merely to discard the non-Gradient-Boosting results.

The date-ticker coverage, training cutoff, realized returns, and predictions
must match within `1e-12`. The full four-set comparison stops if this gate
fails. After it passes, all four feature sets are trained once over the full
matched signal schedule. The four independent feature-set fits run in parallel
for each signal, and progress is printed every ten completed signals. Model
parameters, expanding windows, random seeds, and final output ordering remain
unchanged.

## Research boundary

All data through 22 July 2026 are historical development research. A clean
holdout begins with future observations after 22 July 2026. This script cannot
run beyond the historical research cutoff.

The experiment does not alter strategy configuration, paper targets, paper
orders, or real orders. Results must be reviewed before any feature is promoted.
