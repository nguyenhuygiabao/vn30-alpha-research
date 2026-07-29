# Enhanced FB/FS feature checkpoint

Status as of 2026-07-29. Features are **built and verified but not yet
evaluated**. No result is known. Nothing is promoted.

## Why the original FB/FS set failed

The original `fb_fs` set measured no improvement over the market baseline
(paired Rank-IC difference -0.000546, CI crossing zero). The cause was
diagnosed on the real dataset rather than assumed:

1. `buy_share_Nd - sell_share_Nd` reproduces `net_share_Nd` to within
   5.4e-15 at every horizon. The `fb_fs` set therefore already **contained**
   all the information in the frozen net features.
2. But `corr(buy, sell) = +0.87 to +0.88` — both sides are dominated by a
   common-mode foreign turnover factor — and `std(net) / std(buy) = 0.47 to
   0.50`. The net signal is a small difference of two large, highly
   correlated numbers.
3. A `max_depth=2` axis-aligned gradient-boosting tree cannot express the
   diagonal `buy - sell` contrast. Splitting on `buy` alone exposes mostly
   common-mode variation (`corr(buy, net)` is only +0.28 to +0.38).

The failure was **representational, not informational**. The fix is not more
raw buy/sell columns; it is supplying contrasts the model cannot build for
itself, as single pre-computed axes.

Reproduce the diagnosis with the numbers above from
`data/processed/features_combined.parquet`.

## What was built

Six features in `src/features_foreign_flow.py`, exported as
`ENHANCED_FB_FS_FEATURES`. Each is deliberately **nonlinear** in (buy, sell)
so it is not spanned by the existing net/gross features:

| Feature | Rationale |
|---|---|
| `foreign_flow_imbalance_3d/5d/10d/20d` | net / gross foreign value. The frozen set carries this ratio at 1d only; these extend it to the horizons where net flow actually showed value. Bounded [-1, 1] and scale-free, so flow *direction* is separated from flow *magnitude*. |
| `foreign_flow_return_corr_20d` | Rolling correlation of daily flow imbalance with daily return. Separates momentum-following foreign flow from contrarian flow — a property of flow *character* that no level feature encodes. |
| `foreign_buy_downday_share_20d` | Fraction of foreign buying that landed on down days: accumulation into weakness. |

A useful side property: the imbalance features depend only on `fb`/`fs` and
never on traded value, so they remain defined for FLC and ROS, where every
traded-value-normalized foreign share is missing (973–1009 of 1013 rows
covered, versus 0).

## What is verified

- Full suite **339/339 passing** (16 new tests in
  `tests/test_enhanced_foreign_flow_features.py`).
- Hand-computed value checks: constant buy=300 / sell=100 gives imbalance
  exactly 0.5; one-sided buying saturates at +1.0, one-sided selling at -1.0.
- **Leakage guard**: appending future rows leaves every already-computed
  value bit-identical, so no window reaches forward.
- Zero-variance windows resolve to missing, never infinity — the combined
  feature validator rejects infinities outright.
- Real-data ranges are sane: imbalance within [-1, 1], correlation within
  [-0.91, 0.94], down-day share within [0, 1]. Coverage 97.6–99.9%.
- `audit_fiinquant_core.py --require-data` and `audit_project_state.py` pass.
- The four published feature sets are **byte-identical**; sizes 51/60/64/73
  are still asserted in `test_feature_sets_freeze_exact_51_feature_baseline`.
  The new set is strictly appended as a fifth, `enhanced_fb_fs` (66
  features = 51 baseline + 9 frozen foreign + 6 enhanced).

## What is NOT known

**Whether any of this improves the model.** The experiment has not been run.
Do not describe these features as an improvement until it has.

## Sequencing warning for the rerun

`src/modeling_utils.get_model_feature_columns` auto-discovers every
non-forbidden column, so the 6 new features **will enter the production
models automatically** on the next `run_full_pipeline.py` — before any
evaluation justifies them. That would promote FB/FS by side effect.

Correct order:

1. Run the foreign-flow experiment first. It is insulated: it uses explicit
   frozen feature lists validated by `validate_foreign_flow_feature_sets`,
   not auto-discovery.
2. Review the result against the decision rule below.
3. Only then decide whether to rerun the production pipeline.

## Pre-registered decision rule

Fixed **before** results are seen, so the outcome cannot be selected after
the fact:

- **Primary test**: `enhanced_fb_fs` vs `frozen_foreign_flow`, paired block
  bootstrap on daily Rank IC.
- **Promote to "credible development improvement" only if the 95% CI
  excludes zero.** A CI that touches or crosses zero is recorded as no
  value, and the features stay unpromoted.
- Secondary, reported but not decisive: `enhanced_fb_fs` vs
  `market_baseline`.
- A positive result is **not** a deployment decision. It is development-set
  evidence only. The post-2026-07-22 holdout stays closed.
- Honest caveat to carry into any write-up: this is the **fifth** feature
  set tested against the same development data. The bootstrap CI does not
  adjust for that multiplicity, so a marginal result should be read as
  weaker than its nominal interval suggests.
- Do not iterate on feature definitions to chase a better number. That
  converts this from a test into a search, and the reported CI would no
  longer mean anything.

## Command to run

```powershell
py .\scripts\run_foreign_flow_experiment.py
```

Writes to `reports/tables/foreign_flow/` and
`data/processed/foreign_flow_rebuild_development_predictions.parquet`.
The script is guarded: it changes no production configuration, no paper
orders, and no real orders, and it cannot run past the 2026-07-22 research
cutoff.

Expect it to be slower than the previous run — five feature sets instead of
four, refit per signal.

## Guardrails still held

- FB/FS not promoted.
- 2024+ / post-2026-07-22 holdout not opened.
- No production config, paper orders, or real orders changed.
- The published four-set comparison remains reproducible.
