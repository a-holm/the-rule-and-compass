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

### Registry views and the grouped writer (issue #260)

`--project n` picks 24-48 registered columns before the read, so a 200-column or 364-column store is projected to the view and the resident term follows the view size. `store::write_store_grouped` holds one group of every column (`group_rows x columns x 8` bytes), not the table. `tests/registry.rs` writes a 128-column, 1,000,000-row store (about 1 GiB) in a child process and asserts that the writer and a 13-column projected read of it each stay under 256 MiB peak RSS (`VmHWM`). The 364-column, 20-variable peak-RSS and prep-wall measurements on the real store are taken in the final full test.

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

## Enrichment output staging (#300)

`rcsr enrich` stages each feature column to a temporary `f64` file before the
append pass. The writer uses a 64 KiB staging block (`ENRICH_TEMP_WRITE_CHUNK_BYTES`),
and the GPU path adds a per-family output arena plus its pinned host mirror.
Both are capped at **64 MiB** (`GPU_OUTPUT_ARENA_BYTES` /
`GPU_PINNED_STAGE_BYTES`); a family block whose columns do not fit falls back to
the per-column staging path, so the bound does not grow with the 358-column
catalogue. `memory::enrich_output_stage_bytes(rows, columns)` returns the
charged bytes, `min(rows * columns * 8, 64 MiB) + 64 KiB`.

At 16.1M rows one `f64` column is 128 MiB, so the arena cannot hold even one
full column; that case uses per-column staging and is reported as such. At the
3.87M-row scale the arena holds two columns, and a 500k-row parity fixture holds
eight. The pinned mirror is reused across blocks, never allocated per column.

## Device memory plan (#293)

Before the first device allocation the run computes a plan: a pure function in `src/memory.rs` (`device_plan`) of the workload, the free device memory and the caps. The device is queried once with `cudaMemGetInfo` after the CUDA context exists, so the context's own overhead is already subtracted from `free`.

```text
budget = min(cap 10.5 GiB, free - reserve 1 GiB, --gpu-memory)
data_f32     = rows x (vars + 1) x 4 B                      scoring rows and target
side_vectors = rows x 12 B (label sign) + rows x 8 B (weights), when present
fit_twin     = rows x (vars + 1) x 8 B                      --fit-strategy gpu, or the exact rank/order scorer
                (--fit-rows N < rows: N x (vars + 1) x 12 B + N x 8 B with weights, the subsample's f64 twin and f32 rows;
                 the exact scorer adds a full-row twin)
buffers      = 1 MiB + program slots, score, hash and fit regions, linear in the candidate count
stat_wave    = max(512 MiB, 6 x (rows x 64 B + 1 MiB))      exact rank/order scorer only
```

Decision order:

1. More than `i32::MAX` rows: refused at stage `eval`, because the kernels index rows with 32 bits.
2. `data_f32 + side_vectors + buffers` within the budget: `resident`. An explicit device fit needs the twin as well, and the exact scorer needs the twin and the wave. A device fit that does not fit is refused at stage `fit`. A scorer that does not fit has `stat_route = cpu` and the run keeps the CPU.
3. Otherwise `tiled` when two tiles of `tile_rows` rows, the side vectors, the buffers and at least one candidate's partials fit. `tile_rows` is the largest power of two with `2 x tile_rows x vars x 4 B` within L2, clamped to 512..8192. In this mode the feature columns stay on the host and stream through two device tiles. The target, label sign and weights stay resident. An explicit device fit is refused at stage `fit` ("tiled residency has no device fit"). The default fit and the exact rank and order-statistic scorers run on the CPU, and the plan routes name `device_memory`. The candidates are scored in waves of `partials / 4,112 B` so the lane carry stays under the partials cap (64 MiB).
4. Otherwise refused at stage `eval`.

A refusal names the stage, the bytes it needs, the budget and the smallest fixes, each computed by re-running the plan: `--vars K` (largest count that fits), the largest training row count, `--fit-strategy exact` (stage `fit`), raising `--gpu-memory` or freeing device memory held by other processes, `--cpu`.

```text
error: GPU memory plan refused the fit stage: the f64 fit twin does not fit beside the resident rows
  need 19.50 GiB (20,933,704,236 bytes): rows f32 6.49 GiB + f64 fit twin 12.99 GiB + buffers 15.15 MiB (4,775,757 rows x 364 vars, P=10,000)
  budget 10.09 GiB (10,830,741,504 bytes) = min(cap 10.50 GiB, free 11.09 GiB - reserve 1.00 GiB)
  smallest fix: --vars 187; or at most 2457450 training rows (--sample-rows on store input, scaled by the train fraction); or --fit-strategy exact; or --cpu
```

Outcomes:

* A refused `fit` stage exits 2, because a device fit is always explicit. A tiled run refuses `--fit-strategy gpu` the same way.
* A refused `eval` stage exits 2 under `--gpu`. Without `--gpu` the run keeps the CPU, `backend` is `cpu`, `backend_reason` starts with `device_memory` and the route ledger names `device_memory`.
* A device allocation that still fails is reported as `device_memory: the device refused an allocation of N bytes ...`, never as "no usable CUDA device".

`--json` reports the plan as `gpu.memory_plan` (mode, budget and its source, free and total bytes, required bytes, every term, rows, vars, candidates, L2 bytes, tile rows, routes, the refusal with its fixes, and `observed`, the device allocation ledger of the run). The test hooks `RCSR_TEST_GPU_TILE_ROWS` and `RCSR_TEST_GPU_PARTIALS_CAP_BYTES` override the tile height and the partials cap.

### Plan matrix

Planned for the RTX 3060 with 11,353 MiB free (budget 10.09 GiB) and 4,775,757 rows. `P` is the candidate count; 10,000 is the largest case, and 1,000 changes only the buffer term. `memory::tests::device_plan_matrix` checks every cell.

| vars | device fit | f32 rows | f64 twin | resident need | mode | fit route |
|---:|---|---:|---:|---:|---|---|
| 20 | off | 0.37 GiB | 0 | 0.44 GiB | resident | none |
| 20 | on | 0.37 GiB | 0.75 GiB | 1.18 GiB | resident | device |
| 56 | off | 1.01 GiB | 0 | 1.08 GiB | resident | none |
| 56 | on | 1.01 GiB | 2.03 GiB | 3.11 GiB | resident | device |
| 364 | off | 6.49 GiB | 0 | 6.56 GiB | resident | none |
| 364 | on | 6.49 GiB | 12.99 GiB | 19.54 GiB | resident | refused: `--vars 187`, fewer rows, or `--fit-strategy exact` |

When other processes hold the device, the budget shrinks. At a 1.0 GiB budget the 56-variable run without a device fit needs 1.05 GiB, so it runs `tiled` with 4,096-row tiles. The same budget refuses an explicit device fit at the fit stage. A 364-variable run needs about 48 GB of host memory when every column is used, so the host guard refuses it before the device plan matters.

### Wide stores: 364 columns, 20 variables

The projection (see above) reads only the columns a run uses, so the host and device footprint follow the 20 searched variables and not the 364-column header. The test `projection_gpu_search` builds a 364-column x 65,536-row store (ts, symbol, y, x0..x360), runs `--vars x0..x19 --time ts --group-by symbol --gpu --fit-strategy gpu --generations 3 --json --run-manifest --dev-log`, and asserts:

* exit 0;
* `projection.columns` in the run manifest has 23 names (ts, symbol, y, x0..x19);
* `gpu.memory_plan.mode == "resident"`;
* `peak_rss_bytes` of the development log under 8 GiB;
* `gpu.memory_plan.observed.peak_live_bytes` under 10.5 GiB.

It runs in about 7 seconds. The same builder writes the full-scale store with `cargo test --release --features cuda --test projection_gpu_search -- --ignored build_wide_364_store` (4,775,757 rows x 364 columns, 13.9 GB, one `write_store_appending` pass from the 23-column base and four shared filler column files; 135 s). The search is launched on that file with the `rcsr` binary:

```sh
rcsr --data 293-wide364.rcsb --target y --vars x0,...,x19 --time ts --group-by symbol \
  --gpu --fit-strategy gpu --generations 3 --islands 4 --pop 128 --seed 7 \
  --json --quiet --run-manifest m.json --dev-log dev.log
```

Measured on the RTX 3060 (12 GB), the machine otherwise in use at the lowest priority:

| quantity | value | cap |
|---|---:|---:|
| exit status | 0 | |
| projected columns | 23 | |
| `memory_plan.mode` | resident | |
| training rows | 3,581,818 | |
| peak RSS (`peak_rss_bytes` of the dev log) | 4.77 GB (4.44 GiB) | 8 GiB |
| peak RSS (`time -v`, maximum resident set) | 4.67 GB | 8 GiB |
| observed device use (`peak_live_bytes`) | 945,905,560 B (0.88 GiB) | 10.5 GiB |
| planned device need (`required_bytes`) | 947,408,336 B | 10.09 GiB |
| prep wall (`timing.prep_seconds`) | 8.7 s | |
| search wall (`timing.search_seconds`) | 1,705 s | |
| baselines wall | 101 s | |
| total wall (`timing.total_seconds`) | 1,820 s | |

The search wall is dominated by host work: 47 candidates of generation 0 fall back to the CPU fit (132 s), and the process runs at the lowest CPU priority. The device kernels take 2.7 s of the run (`gpu_seconds`).

### Resident scoring throughput

Resident scoring of P candidates over 4,775,757 x 20 f32 rows (squared score, `score_pool`, median of 7 warm calls):

| P | median | min | target |
|---:|---:|---:|---:|
| 400 | 0.139 s | 0.138 s | 0.25 s |
| 4,096 | 1.285 s | 1.278 s | 2 s |

The record is `resident_throughput_record` in `tests/projection_gpu_search.rs` (`--ignored --nocapture`). It is a measurement, not a gate.
