# FiinQuant core checkpoint

This checkpoint isolates the production FiinQuant data path from unrelated
dashboard and research work.

## Included

- Canonical raw, adjusted, and foreign-flow history ingestion
- Explicit raw-price versus adjusted-price data contracts
- Point-in-time VN30 membership reconstruction
- Historical exchange and price-limit rules
- Corporate-action validation
- Foreign-flow feature construction
- Liquidity safeguards for zero turnover and zero lagged volume
- Focused regression tests and a repeatable core audit

Raw prices remain the source for execution and price-limit logic. Adjusted
prices remain the source for returns, labels, and model features. A genuine
zero foreign-flow value remains zero; missing vendor data remains missing.

## Deliberately excluded

- Dashboard changes and generated reports
- FiinQuant probe scripts and entitlement experiments
- Trial comparison and one-off diagnostic scripts
- ZIP archives, patch files, and root-level source copies
- Generated raw data and feature artifacts

The reusable `fiinquant_adapter` module is retained as the tested API
normalization boundary for a future incremental updater. The old trial
downloader is excluded because the canonical historical downloader supersedes
it.

## Validation gates

The checkpoint is committed only after:

1. Focused FiinQuant tests pass.
2. The complete repository test suite passes.
3. Features rebuild from the local canonical FiinQuant data.
4. Canonical keys, required feature columns, duplicate keys, and infinite
   numeric values pass the core audit.
5. Python compilation and Git whitespace checks pass.

The preparation tool does not push the checkpoint branch.
