---
type: reference
description: Rules for history, locks, labels, purge, embargo, annualisation, costs and benchmarks that hold for every time-ordered dataset.
---

# Time rules for every dataset

Details and tests: [time-series checklist](../../../docs/time-series.md), [panels and the locked test](../../../docs/panel.md), [walk-forward](../../../docs/walkforward.md).

History
- Use every row from the first bar of the source to the newest. A later start needs the owner's written approval, recorded as `approved_start`. Warm-up is not a later start: features are NaN until their lookback is filled.
- A panel starts at the first membership date the membership rule gives. That follows from `--min-history` and is not a cut; record the date.

Locked test
- Declare the locked test with the data, on the first store of the chain (`--locked-test START,END` of `rcsr prep`, `rcsr panel build` or `rcsr enrich`), before any development result exists. Every later store inherits it.
- Development data ends before the lock. Evaluators refuse `[START - purge, END + embargo)`.
- Files that hold rows inside the lock are downloaded and checksummed only. They are never opened, decoded, listed by content or summarised before the seal. Stage them by name.
- A store that must hold locked rows (a read store, built after the candidates are frozen) is built with `--allow-lockbox`, and with `--lock-record` on every command that takes it (`rcsr enrich` and the `rcsr panel` builders), so its sidecars say nothing about the range. `rcsr lock check --record <record> <dir>` exits 0 on every directory.
- Never print a symbol, date or value from inside the lock, and never write one into an issue, a pull request or a tracked file. Globs and greps stay inside the staged and store directories; a `*-sealed/` directory is never listed.
- A tail guard runs before each phase of a campaign: no run, walk-forward or path file may hold a window that reaches the lock start.
- The lock is read once, at the end, through the ledger: register, identity check on development rows, freeze, seal, read, verify, verdict. A second read of a candidate is refused. A null store never enters a ledger.

Labels and splits
- A label enters at the next open after the decision bar and exits `horizon` bars later. Its purge is `entry_offset + horizon + delay`, derived from the spec.
- Overlapping labels: purge at every boundary, or sample every `horizon` bars with `stride` and one `anchor` for all symbols so every series decides at the same stamps. Do not mix the two.
- The embargo is the longest feature lookback of the view. Use `--embargo-seconds auto` of `rcsr protocol cpcv` (needs `--vars` and `--bar-seconds`), not a hand table.
- Split chronologically. Combinatorial purged cross-validation over groups (`rcsr protocol cpcv`), walk-forward origins as a second lens. A split that reaches the lock is trimmed or refused by the engine; check the last group end.
- Count members per split group before choosing a cross-section floor (see the type stage).

Annualisation
- One period is one label horizon wide: `S = max(horizon_seconds, stride_seconds)`, `periods_per_year = 31536000 / S`. `rcsr walkforward` and `rcsr trials` refuse an explicit `--periods-per-year` that differs from it by more than 1 %. An hourly-period Sharpe is not a daily Sharpe.

Costs and position rules
- Spot only, long only: no short, no margin, no options, no borrow term. A label has actions 0 and +1.
- Costs are symmetric per side and enter the label (`fee`), and every result is also read at twice the cost.

Benchmarks
- Sharpe and Sortino are measured against non-interest cash at 0. Excess returns are over cash at 0, buy-and-hold, or the equal-weight book of the members. No bond, bill, interest-rate, lending or stablecoin-yield benchmark is used anywhere.

Null stores
- A search-capacity null is a store with permuted labels (`rcsr panel permute-labels`). Its header and manifest carry the `null_store` marker, the marker follows every copy and append, and a marked store is refused by the ledger and the locked read.

Registry
- Record `feature_registry_version` and `schema_version` of `rcsr schema` with every store. A store written under an older registry version still verifies its hashes; its registry text may differ, so rewrite the schema sidecar before a comparison.
