# Audit

## 1. Purpose

This file records the reproducibility and consistency checks run against the repository, and the state of the automated audits.

## 2. Automated Audits

Four audit scripts are maintained in `scripts/`. All four currently pass.

### Project state audit

```powershell
py .\scripts\audit_project_state.py
```

Checks that every required source file, report, table, figure, and config file exists and is non-empty, that no generated data files have leaked into Git, and that all PNG figures are non-empty. It also reports working-tree status.

### FiinQuant core audit

```powershell
py .\scripts\audit_fiinquant_core.py --require-data
```

Validates the data contract end to end. Current output:

```text
Canonical market rows: 137719
Canonical market tickers: 60
Raw-price fallback rows: 0
Canonical foreign-flow rows: 137719
Combined feature rows: 74910
Combined feature columns: 109
Duplicate feature keys: 0
Infinite feature columns: {}
```

Zero raw-price fallback rows means every ticker-date resolves to genuine raw OHLC, including FLC and ROS after the vendor-archive backfill. Zero duplicate keys and zero infinite columns are hard requirements enforced by `validate_combined_features`.

### Execution coverage audit

```powershell
py .\scripts\audit_execution_coverage.py
```

For every active point-in-time VN30 constituent on every trading date, checks whether raw OHLC, volume, and traded value are present. Current output:

```text
Trading dates in dataset: 2497
Expected active constituent-dates: 74910
  full_raw_execution_data: 73897
  estimated_traded_value: 1013
Dates with 100% execution-valid coverage: 2497 / 2497
Blocked dates (raw price, volume, or row missing): 0
```

The 1,013 estimated rows are FLC (127) and ROS (886) between 2016-07-26 and 2021-01-29, where no vendor traded-value figure exists. These are flagged per row by `value_traded_is_estimated`. There are no rows with missing raw prices, missing volume, or missing rows entirely.

Per-date and per-ticker-date detail is written to:

```text
reports/tables/execution_coverage_summary.csv
reports/tables/execution_coverage_gaps.csv
```

### Point-in-time universe audit

```powershell
py .\scripts\audit_point_in_time_universe.py --market-data-path <path>
```

Verifies that reconstructed VN30 membership resolves to exactly 30 constituents on every review date and that market data covers every historical member.

## 3. Test Suite

```powershell
py -m pytest -q
```

409 tests, all passing. Coverage of note:

- **Leakage guards** — appending future rows must leave every already-computed feature value bit-identical, asserted with `pd.testing.assert_frame_equal`.
- **Parallelism equivalence** — parallel walk-forward output must equal sequential output exactly, so process-based fitting is a speed change only.
- **Data-contract enforcement** — raw/adjusted mismatch, missing traded value on non-exempt tickers, and infinite feature values must all raise.
- **Feature-set freeze guards** — the four published foreign-flow feature sets must remain at exactly 51/60/64/73 features so committed comparisons stay reproducible.
- **Execution coverage classification** — membership-exit filtering, each execution-status category, and the rule that a flagged estimate does not block date completeness.
- **Daily-update price basis** — the FiinQuant daily provider must return raw prices in `close` and adjusted prices in `adjusted_close`, converted from whole VND to thousands. Collapsing the two, or missing the conversion, is asserted against directly.
- **Tranche portfolio construction** — sleeve weights sum correctly, a hand-built case produces the expected equity path, costs are charged on the date a tranche opens rather than across its holding period, and the construction contains no lookahead.
- **History extension** — the per-ticker adjustment factor between a refetched window and the stored history must be constant before any rescaling is applied, and a merge that would drop rows, duplicate ticker-dates, or silently absorb a raw-price revision must raise.

## 4. Pipeline Reproducibility

The full pipeline runs 18 stages:

```powershell
py .\scripts\run_full_pipeline.py
```

Stage order: data quality, features, labels, walk-forward checks, linear models, tree models, classification models, baseline portfolios, risk model, optimizer, backtester, ablation tests, horizon tests, report tables, model report, figures, report page, report summary.

Cache behaviour: `--reuse-existing-ml` skips expensive model stages only when their outputs exist and are newer than their dependency files. `src/ablation_tests.py` and `src/horizon_tests.py` previously carried a separate internal cache with no freshness check, which served stale predictions indefinitely; that cache has been removed and both stages now always rebuild when invoked directly.

**Verified reproducible.** A complete rerun from `--clean-first`, rebuilding every processed artefact from the canonical CSVs, produced output byte-identical to the committed tables, figures and report page. `git status` reported no modification to any tracked file. Stage timings for that run: ablation tests 3,230s, horizon tests 1,936s, tree models 1,193s, optimizer 181s, linear models 102s, all other stages under 70s.

## 5. Consistency Checks

Two independently regenerated outputs are cross-checked for agreement:

- `horizon_results.csv` at the 5-day horizon and `ablation_results.csv` for `all_features` describe the same configuration (gradient boosting, all features, price-limit-aware execution) and agree exactly: Rank IC 0.033360, after-cost return -0.000822, cumulative -1.988493. This agreement was broken between 2026-09-05 and 2026-10-21, when `ablation_results.csv` had been regenerated under the corrected cost model and `horizon_results.csv` had not: the two read -0.000822 and +0.000076 for the same configuration. Both are now on the corrected model.
- `ablation_results.csv` for `all_features` and `benchmark_results.csv` for `ml_normal_price_limit_aware` agree at -0.000822. The ablation backtest applies price-limit execution unconditionally, so it corresponds to the `price_limit_aware` benchmark row, not the `normal` one.
- The phase-sampled columns of `tranche_portfolio_diagnostic.csv` and the whole of `compounded_phase_sensitivity.csv` are produced by two independent implementations and agree to the digit, for example `normal` under price-limit-aware execution at 0.725619 to 1.393828 final equity and -43.145238 to -25.288476 percent drawdown. Both diagnostics were confirmed deterministic across reruns.
- Weighted by period count, the five phase means recover the published overlapping per-period mean exactly, to machine zero. The weighting is necessary because the series length is not a multiple of the horizon, so phases hold 484 or 483 periods.

## 6. Known Gaps

- **Paper-trading account history predates the price-basis correction.** The paper-trading OHLCV file was previously downloaded from vnstock, which serves a single back-adjusted series: every price column held adjusted prices and `adjusted_close` merely repeated `close`. Because back-adjustment is anchored at the present, the most recent bar was correct but history was restated by later corporate actions. The file has been rebuilt from the canonical FiinQuant history with raw prices in the OHLC columns and a genuine adjusted close (`scripts/build_paper_trading_ohlcv.py`). Row coverage is unchanged at 47,343 and the last date is bit-identical, so the live account is unaffected, but 41,279 historical closes changed. Any account replay produced before this rebuild used the old basis.
- **Issuer-group and sector metadata provenance.** `config/vn30_universe.csv` covers all 60 historical constituents, but the issuer-group and sector labels for the 30 alumni tickers were compiled manually rather than sourced from a vendor classification file. They should be verified against an authoritative source.
- **Corporate-action adjustment quality.** Verified only to the extent that raw and adjusted files agree on volume and traded value. A dedicated split and stock-dividend audit has not been run.
- **Confidence intervals.** Only the foreign-flow comparison reports bootstrap confidence intervals. Horizon, ablation, and portfolio metrics are point estimates.
- **Frozen baseline in the report page.** `scripts/build_report_page.py` reads every current figure from `reports/tables/`, but the `train_size = 3` comparison column is a hardcoded historical record, since no current table contains that configuration. It is labelled as such in the script.

## 7. Status

Automated audits pass, the test suite passes at 409 tests, the pipeline reruns byte-identical from clean, and independently regenerated outputs agree where they describe the same configuration. Research results are documented in `reports/final_results.md`, methodology in `reports/methodology.md`, and the foreign-flow experiment in `docs/enhanced_fb_fs_checkpoint.md`.

**The headline research result is negative.** Run as the overlapping-tranche portfolio the strategy would actually trade, no optimizer mode beats buying and holding the VN30 after costs: the best configuration returns 1.4850 against the index's 2.6918 over 9.58 years (2,414 trading days at 252 per year), with a 58 percent peak-to-trough fall in capital and a compounded active equity of 0.5557. These are the figures under the corrected cost model of commit `bfdabcf`; before that correction the same row read 2.6252 and 0.9828. Section 12 of `reports/final_results.md` gives the full table and explains why the tranche construction, rather than non-overlapping sampling, is the instrument reported.

This repository is a research framework. It is not a live trading system and the reports are historical diagnostics, not investment recommendations.
