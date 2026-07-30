# Enhanced FB/FS feature checkpoint

Status as of 2026-07-30. Features are **built, verified, and evaluated
against the pre-registered decision rule. FB/FS does not clear the bar.
Not promoted.** See "Result" below.

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

## Result (2026-07-30)

`scripts/run_foreign_flow_experiment.py` was run to completion: 222
non-overlapping 10-day signals, 2017-08-09 through 2026-07-01, expanding
training window, frozen `gradient_boosting` model, paired block bootstrap
on daily Rank IC. ~4 hours wall clock. Exit code 0. Full test suite
(342/342) still passing afterward. No production config, `feature_
pipeline.py` auto-discovery, paper trading, or real orders were touched.

**Pre-registered primary test result:** `enhanced_fb_fs` vs
`frozen_foreign_flow`, mean Rank-IC difference **-0.002608**, 95% CI
**[-0.013314, +0.008297]**. The CI spans zero and the point estimate is
negative. **Does not clear the bar. FB/FS is not promoted.**

Secondary test (reported, not decisive): `enhanced_fb_fs` vs
`market_baseline` = +0.005619, CI [-0.009386, +0.020961] — also spans
zero.

Full ranking by average Rank IC across all 5 sets (222 signals each):

| feature set | average Rank IC |
|---|---|
| `frozen_foreign_flow` (market baseline + net-style foreign features only, no buy/sell split) | 0.0636 — best |
| `frozen_plus_fb_fs` (adds original buy/sell shares) | 0.0624 |
| `enhanced_fb_fs` (adds the 6 new nonlinear features on top) | 0.0610 |
| `market_baseline` (no foreign flow features at all) | 0.0554 |
| `fb_fs` (buy/sell shares only, no net) | 0.0537 — worst |

Two clean facts fall out of this ranking on their own: (1) the frozen
*net*-style foreign flow features are genuinely useful —
`frozen_foreign_flow` beats `market_baseline` by +0.0082, a real,
non-trivial gain; (2) every set that adds a *buy/sell-split*
representation, original or enhanced, sits below `frozen_foreign_flow`,
and adding more buy/sell-derived features (`fb_fs` → `frozen_plus_fb_fs`
→ `enhanced_fb_fs`, going the other direction) makes it progressively
worse, not better.

### Why — revised diagnosis

The original representational diagnosis (below) explained why the
*original* `fb_fs` failed: `buy_share - sell_share` reproduces `net_share`
to 5.4e-15, `corr(buy, sell)` is +0.87 to +0.88, and a `max_depth=2` tree
cannot express the diagonal `buy - sell` contrast from raw, correlated
inputs. Building bounded, pre-computed, nonlinear contrasts
(`foreign_flow_imbalance_Nd`, `foreign_flow_return_corr_20d`,
`foreign_buy_downday_share_20d`) was the correct fix for *that* specific
failure mode. It did not fix the outcome, and checking why revealed a
different, deeper issue:

- **The new features are not being ignored.** In `enhanced_fb_fs`'s own
  feature-importance output, `foreign_flow_imbalance_20d` and
  `foreign_flow_return_corr_20d` rank #2 and #3 by average importance,
  right behind the frozen net feature. The model uses them heavily — this
  rules out "diluted into irrelevance by noise" as the explanation.
- **The relationship is not stable over time.** Breaking the daily
  Rank-IC difference (`enhanced_fb_fs` minus `frozen_foreign_flow`) down
  by year: positive in 2017, 2020, 2022, 2023, 2025, 2026; negative in
  2018, 2019, 2021, 2024 — no visible trend, roughly 6 up years against 4
  down years. Per-signal, `enhanced_fb_fs` beats `frozen_foreign_flow` on
  50.5% of the 222 dates, statistically a coin flip. The day-to-day
  standard deviation of the difference (0.075) is about 29x the overall
  mean (-0.0026) — at 222 non-overlapping signals across 9 years, there
  is not enough statistical power to distinguish the true average effect
  from zero in either direction.

Conclusion: this looks like a **non-stationary or regime-dependent
signal**, not a representational limit and not pure noise. The model can
access the information; whatever relationship exists between foreign flow
imbalance and forward returns does not hold a consistent sign across the
sample period. No further feature re-encoding on the same 9-year sample is
expected to fix that.

**One specific, honestly-flagged lead, not yet pursued:** the year-by-year
sign flips loosely track known stress/calm alternation, and this project
already has herding/regime features in production (`herding_index`,
`high_herding_day`). Whether foreign-flow imbalance's predictive sign is
conditional on herding regime is a real question the data raises. It has
not been tested. Pursuing it would be a 6th feature-set test on the same
development data (compounding the multiplicity caveat below) and should be
a deliberate, separately-scoped decision, not a default next step.

**Second flagged re-test trigger: a Vietnam market classification upgrade
(MSCI/FTSE frontier -> emerging market).** The entire evaluated sample
(2017-2026) predates any such reclassification. A material, sustained
increase in foreign participation is exactly the kind of regime shift that
would plausibly change the sign-stability problem identified above -- it
is a structural change in *who* is trading, not a hope that re-running the
same test on the same regime yields a different number. Do not re-test on
this basis until the reclassification is real (not anticipatory), since
testing on a hypothesis rather than realized data would reintroduce the
multiplicity/spec-search problem this checkpoint exists to avoid. The
feature code (`ENHANCED_FB_FS_FEATURES` in
`src/features_foreign_flow.py`) is not being removed and remains available
for that re-test when warranted.

## Sequencing warning for the rerun (historical — followed correctly)

`src/modeling_utils.get_model_feature_columns` auto-discovers every
non-forbidden column, so the 6 new features would have entered the
production models automatically on the next `run_full_pipeline.py` —
before any evaluation justified them. That would have promoted FB/FS by
side effect. This was avoided: the foreign-flow experiment (insulated,
explicit frozen feature lists, not auto-discovery) was run and evaluated
before any production pipeline rerun, and the result above means no such
rerun with FB/FS enabled is warranted.

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

## Command to run (already executed 2026-07-30 — see Result above)

```powershell
py .\scripts\run_foreign_flow_experiment.py
```

Wrote to `reports/tables/foreign_flow/` and
`data/processed/foreign_flow_rebuild_development_predictions.parquet`.
The script is guarded: it changes no production configuration, no paper
orders, and no real orders, and it cannot run past the 2026-07-22 research
cutoff. Confirmed via `git status` after the run — none of those were
touched.

Took ~4 hours wall clock (expanding training window refit per signal, 222
signals × 5 feature sets, threaded 4-way — not directly comparable in cost
to the production pipeline's fixed-window, process-parallel refits).

## Guardrails held

- FB/FS tested against the pre-registered rule and **not promoted** — the
  result is a decision, not just an absence of promotion.
- 2024+ / post-2026-07-22 holdout not opened.
- No production config, paper orders, or real orders changed.
- The published four-set comparison remains reproducible.
- Feature definitions were not iterated after seeing results.
