---
type: reference
description: What every type stage 04_* of the data pipeline reads, writes and proves.
---

# Type-stage contract

A type stage turns the raw development files of one data type into a base store and a recipe that the later stages follow without knowing the type.

Reads: `../01_intake/output/card.md`, `../02_acquire/output/acquire.md`, `../03_profile/output/profile.md`, each with a `check.md` that says `pass`, and its own `references/`.

Writes into its `output/`: `adapt.md` copied from `../../_templates/adapt.md`, and `check.md`. Writes the base store `$STORE` with its manifest.

`adapt.md` holds these keys, one per line, `key: value`:

| Key | Meaning |
|---|---|
| `type` | the folder name of the type stage, for example `04_ohlcv` |
| `variant` | the column variant inside the type (for bars: `ohlc`, `ohlcv`, `ohlcv_flow`) |
| `layout` | the file layout that was detected |
| `bar_seconds` | spacing of the store rows; `none` for irregular data |
| `source_bar_seconds` | spacing of the raw rows |
| `first_bar` | first row of the source, ISO UTC |
| `store_first_bar` | first row of the store; equals `first_bar` unless `approved_start` is set |
| `approved_start` | `none`, or the later start with who approved it and when |
| `dev_end` | first instant that is not development data |
| `locked_test` | `START,END` of the declared locked test, or `none` with the reason |
| `series` | `single` or `panel` |
| `symbols` | number of series |
| `raw_columns` | the raw value columns the source really holds |
| `zero_volume_rows`, `zero_volume_policy` | count, and `keep` or `drop` with the reason |
| `feature_groups` | the `--features` value stage 06 uses |
| `view_excludes` | names or families the variant cannot support |
| `labels` | the `--label` values stage 06 uses |
| `periods_per_year` | `31536000 / max(horizon_seconds, stride_seconds)` of the primary label |
| `store` | path of the base store |
| `deviations` | everything skipped, with the reason; `none` only if nothing was skipped |

Proves in its Check: every key is present, `store_first_bar` equals `first_bar` or `approved_start` is not `none`, the store and its manifest exist, and no development file reaches `dev_end`.

A type stage never builds features, labels, panels or splits. It never opens a file inside the lock.
