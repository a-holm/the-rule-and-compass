# Memory model

`rcsr` stores a loaded table **column-major**: one contiguous `f64` block per
numeric column, and one `u32` code per row plus a small level table for text or
categorical columns. There is no row-major copy and no per-cell `String` after
parsing.

## Bytes per cell

Let `n` be the number of rows and `C` the number of distinct levels in a text
column.

| column kind | resident bytes per cell | note |
|---|---:|---|
| numeric, high cardinality | **8** | one `f64` in a contiguous column |
| numeric, `<= 4096` distinct values | **12** | `f64` + `u32` code; the exact source text is kept for dates/labels |
| text / categorical | **4** + `O(C · label)` / `n` | `u32` code + one copy of each label |
| missing numeric cell | 8 | `NaN` is "not data"; every kernel is NaN-aware |
| boolean input | 8 | a `f64` 0/1 column (the search's view) |

A 6-column numeric table with six non-missing columns therefore costs about
**48 B/row** of cell storage. The rest of a run is the fixed binary and the
search's samples; row selection itself (train/test/folds) is a **selector**, not
a second copy of the table.

This replaces the old representation, which kept both `Vec<Vec<f64>>`
(32 B/cell plus 24 B of `Vec` header per row) and `Vec<Vec<String>>` (24 B of
`String` header plus one allocation per cell). Measured on a 20 000-row,
6-column CSV, the loader's peak RSS fell from **38.5 MB to ~9.4 MB** (about
0.47 KiB/row), a 4.1x reduction.

## What is not stored

* **No general `f64` compression.** Dictionary encoding is for text and
  low-cardinality columns only; a high-cardinality `f64` column stays plain
  `f64` (a lossless double codec decodes slower than the DRAM it replaces).
* **No row groups while resident.** Row groups bound the working set of an
  out-of-core backing; they do not make the data smaller. The in-memory store
  exposes `load_group`/`group_view`; the file backing added in #71 uses the same
  shape with `pread`, so call sites do not branch. See
  [`data-format.md`](data-format.md).
* **No `f32`.** Storage stays `f64`; results are the numbers the user supplied.

## Projected reads and the upload ring (issue #294)

A projection is an index vector, not a copied table. With 20 variables plus
time from a 364-column store, the resident storage term falls from about
`364 · 8 · n` bytes to `21 · 8 · n`; the guard charges the projected columns
(`storage_bytes`), a projected group buffer instead of the full-header
`GroupBuf`, and the pinned ring (`ring_bytes`). The device tile is reported as
`device_tile_bytes` but is not host RSS. `store_memory` in `--json` exposes all
of these terms.

The GPU transport pins **3 slots × one projected group**, capped at **64 MB**.
At `group_rows = 65 536` and 21 `f64`/`i64` columns that is 11 MB per slot,
33 MB for the ring. The pin is constant for the life of the run; groups stream
through it. A group wider than the cap is a refusal, not a fallback to a full
table. The reader writes straight into the pinned slot (`pread`), so there is
no intermediate `Vec<f64>` and no second host copy.

## Row selectors

A split, fold or evaluation subset is described by a [`RowSelector`]:

| selector | when `auto` chooses it | cost |
|---|---|---|
| `All` | no split | zero |
| `Range` | contiguous rows (temporal blocks, folds of an ordered table) | zero |
| `Set` | **reused and resident** and `\|S\| <= 7 %` of `n`, with an absolute cap of 1 000 000 rows | 4 B/selected row |
| `Hash` | everything else (one-shot subsets, large folds) | O(1) memory; ~1.5 ns/row |

A 20 % fold is a scan, not a gather: under the I/O model a scan beats a gather
when `|S| / n < 1/B`, and a resident cache line holds eight `f64` values
(`B ≈ 8`). A one-shot `Set` never wins because building it costs the scan it was
meant to avoid. The hash is a pinned 64-bit multiply (`RowSelector::hash_row`),
not `DefaultHasher`.

The textual split report includes the selector that was used, e.g.

```text
split=temporal | selector: train range 0..600 (600 rows) | test range 600..800 (200 rows)
```

`Median`, `Interquartile`, `RankCorrelation` and `Auc` sort a full vector and
therefore cannot be computed from a streaming group scan; when an out-of-core
backing is added those metrics must stay resident, be capped, or be reported as
approximate.
