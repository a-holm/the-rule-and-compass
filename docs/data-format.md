# The column store and the memory guard

Issue **#71**. Two related pieces:

1. `rcsr import` writes a versioned, inspectable binary **column store**.
2. `--max-memory` refuses an input that would not fit, before the resident
   `Table` and its search copies are built.

A store that does not fit is still searchable: the loader scans it group by
group and materialises a bounded, hash-selected sample. RSS is bounded by the
group buffer and the sample, not by the file.

## Memory guard

```sh
rcsr --data large.csv --target y --max-memory 2GB
```

The estimate is derived from the table's **shape**, not its file size (issue
#112), and the CSV path **streams** it (issue #126): a quote-aware scan reads the
file once to measure the header width, data-row count and byte count, then the
parser makes its two streaming passes. No whole-file `String` is built, so the
text is not a resident term:

```text
text     = bytes read into one String (the current loader; a streaming path
           removes this term -- see issue #126)
table    = rows x columns x 12 B          (f64 value + u32 dictionary code)
prep     = rows x used_columns x 24 B      (cells, train and test copies)
         + rows x 240 B                    (sample headers, selectors, row copies)
estimate = (text + table + prep) x 1.25    (allocator slack)
```

`used_columns` is the search's projection (`--vars`, `--target`, `--weight`);
with no `--vars` the whole header is assumed. `rcsr import` has no search
preparation, so its guard uses the `text + table` terms only. The old
`33 x file size` factor was calibrated on ~6-column data and predicted 560 GB
for a 17 GB, 93-column panel that is at most ~27 GB resident. The guard reports
its estimate **and this basis** (rows x columns, the three terms and slack) in
the normal preparation block, in text and in `--json`/`--ndjson`, not only when
it refuses a file.

### Calibration (issue #112)

The fixed preparation term was **72 B/row** before this calibration. On the
1M-row x 6-column fixture that model predicted 378 MiB, but the search run
peaked at 552 MiB, so the fixed term was raised to **240 B/row**. The 12 B/cell
and 24 B/used-cell terms were left unchanged. The raise is justified by the
measurements below: the estimate is now above the measured peak everywhere,
but never more than 1.55x it.

Machine: 31 GiB RAM, 8 threads, `/mnt/lager`. `old` is the `read_to_string`
loader; `stream` is the #126 streaming loader (same estimator without the text
term). The `search` command is `--target y --generations 1 --pop 8
--baseline-samples 16 --seed 1`; `import` is `rcsr import ... --quiet`.

| fixture | rows x cols | file | path | peak RSS | estimate | estimate / peak |
|---|---|---:|---|---:|---:|---:|
| 6-column | 1,000,000 x 6 | 29.0 MiB | old import | 91.6 MiB | 122.0 MiB | 1.33 |
| 6-column | 1,000,000 x 6 | 29.0 MiB | stream import | 61.8 MiB | 85.8 MiB | 1.39 |
| 6-column | 1,000,000 x 6 | 29.0 MiB | old search | 552.2 MiB | 579.8 MiB | 1.05 |
| 6-column | 1,000,000 x 6 | 29.0 MiB | stream search | 422.1 MiB | 543.7 MiB | 1.29 |
| 90-column | 150,000 x 90 | 88.4 MiB | old import | 196.1 MiB | 303.6 MiB | 1.55 |
| 90-column | 150,000 x 90 | 88.4 MiB | stream import | 140.0 MiB | 193.1 MiB | 1.38 |
| 90-column | 150,000 x 90 | 88.4 MiB | old search | 553.9 MiB | 732.8 MiB | 1.32 |
| 90-column | 150,000 x 90 | 88.4 MiB | stream search | 452.5 MiB | 622.3 MiB | 1.38 |

The narrow fixture is 30,364,090 bytes; the wide one is 92,696,166 bytes. A
wide but small-on-disk file is therefore no longer mistaken for cheap: the
estimate grows with rows x columns, and a narrow but large file is no longer
multiplied by 33. Tests for both shapes live in
`my-version/tests/memory_estimate.rs`.

When the estimate is above the budget, `rcsr` prints the file size, the
rows x columns basis, the estimate and the budget, suggests `--force` and
`rcsr import`, and exits **3** (input error). It does not allocate the table
first, so it cannot become `memory allocation of N bytes failed` (SIGABRT) or a
silent SIGKILL.

`--force` overrides the guard. The kernel can still kill the process; that is
the deliberate meaning of "force".

`--max-memory` accepts bytes or `KB`/`MB`/`GB` (decimal) and
`KiB`/`MiB`/`GiB` (binary), case-insensitively.

The guard is opt-in for a CSV, but it also fires when a **hard** wall is
detected: a cgroup v2/v1 memory limit or `ulimit -v` (`Max address space`).
That is what makes a container or `ulimit` run fail cleanly instead of being
OOM-killed. The soft `MemAvailable` figure is deliberately not used to refuse a
CSV (the shape estimate is conservative); it is used, together with the hard
limits, to decide whether a `.rcsb` store is loaded resident or streamed.

## Import

```sh
rcsr import data.csv data.rcsb [--group-rows 65536] [--time ts]
rcsr --data data.rcsb --target y --vars x1,x2
```

`import` uses the same CSV reader as a search, so ragged rows are refused
unless `--skip-malformed` is passed. The output defaults to the input path with
its extension replaced by `.rcsb`. The same input and `--group-rows` produce
the same bytes.

The reader is RFC 4180-style and streams: one logical record is held in memory
at a time, and the file is read twice (typing, then building) rather than loaded
as one string. A quoted field may contain the delimiter, a tab, a newline, or an
escaped `""`:

```text
x,y,label
1,2,"a,b"
3,4,"c	d"
5,6,"two
lines"
7,8,"he said ""hi"""
```

Both LF and CRLF record separators are accepted; a newline inside quotes is
kept as `\n`. Blank lines and lines starting with `#` outside quotes are
ignored. The delimiter (tab, comma or semicolon, else whitespace) is detected on
the first record *after* quote-aware parsing and confirmed against the second,
so a delimiter inside quotes cannot choose the wrong split. A genuinely ragged
row still fails with its 1-based physical line; an unterminated quote names the
line where the quoted field starts.

When the store fits the budget it is loaded into the ordinary resident
`Table`; every later path (split, preparation, search, diagnostics) is
identical to the CSV path. On the fixed `tests/fixtures/clean.csv` trajectory
the imported store reproduces the pinned front exactly:

```text
CSV   sha256  e4140058d01882540e877259014b07559bf1b9d1805b4d4e7c776220d18c2992
store sha256  e4140058d01882540e877259014b07559bf1b9d1805b4d4e7c776220d18c2992
```

When it does not fit, the run prints an `out of core:` note and searches a
sample selected by the pinned one-multiply hash while scanning groups. The
sample is a sample: report it as such. (The full-data re-score of the Pareto
front on a streamed store is follow-up work, not silently claimed here.)

## File format, versions 1 and 2

All integers and floating-point values are little-endian. The layout is:

```text
offset 0   8 bytes   magic "RCSRBIN\0"
offset 8   4 bytes   u32 version (writers emit 2; readers accept 1 and 2)
offset 12  4 bytes   u32 header_len
offset 16  header    UTF-8 JSON, header_len bytes
           data      per-group, per-column value and code blocks
```

### Per-column dtypes

| `dtype` | value block | code block | used for |
|---|---|---|---|
| `f64` | `n * 8` bytes | `n * 4` bytes when `has_codes` | features and targets |
| `i64` | `n * 8` bytes | none | the time column, nanosecond-safe |
| `u32` | none | `n * 4` bytes | text/categorical dictionary columns |

Dictionary **codes never enter arithmetic**: a `u32` column has no evaluator
values, and a numeric dictionary column keeps its `f64` value block while the
`u32` codes in `DictColumn` resolve to `levels`. The evaluator always sees a
level or a value, never a code.

The header is self-describing and deterministic:

```json
{"format":"rcsr-column-store","version":2,"n_rows":100000,
 "n_cols":3,"group_rows":65536,"columns":[
  {"name":"x1","numeric":true,"dtype":"f64","levels":[],"has_codes":false,
   "offsets":[0,262144],"code_offsets":[]},
  {"name":"site","numeric":false,"dtype":"u32","levels":["A","B"],"has_codes":true,
   "offsets":[0,0],"code_offsets":[1835008,262144]},
  {"name":"y","numeric":true,"dtype":"f64","levels":[],"has_codes":false,
   "offsets":[524288,786432],"code_offsets":[]}]}
```

* `n_rows`, `n_cols`, `group_rows`: the logical shape. `n_groups` is
  `ceil(n_rows / group_rows)`; the last group is short, never padded.
* `numeric`: the column has an evaluator value block (`false` for `u32`).
* `dtype`: `f64`, `i64` or `u32`, as in the table above.
* `levels`: the dictionary (empty when the column has no dictionary).
* `has_codes`: the column carries a code block in every group.
* `offsets[g]` / `code_offsets[g]`: byte offsets **relative to the data
  section**, for group `g`. The data section begins at `16 + header_len`.
  Relative offsets keep the header independent of its own length. A `u32`
  column has no value bytes, so its value offsets do not advance.

Blocks are written in this order, for every group `g`:

```text
for column c in 0..n_cols:
    value block   group_len(g) * 8 bytes   (f64 NaN / i64::MIN when missing;
                                            absent for u32)
    code block    group_len(g) * 4 bytes   (v2, only when has_codes)
```

Version 1 is the original layout: every value block and every code block is
`f64` (8 bytes), and a v1 header has no `dtype` field. A v1 header that does
carry `"dtype":"i64"` (the #127 extension) is read as an i64 time block. The
version stays authoritative for code width: v1 codes are `f64`, v2 codes are
`u32`. A v2 store is therefore detected by its version field, not by trying
to parse it as v1.

### Time columns

`rcsr import` selects the time column in this order:

1. `--time <column>`, if given. Its cells must be whole numbers; otherwise
   the import refuses rather than silently writing a lossy store.
2. The first column whose name is time-like (`ts`, `timestamp`, `time`,
   `date`, `datetime`, `day`, `dt`) and whose cells are all integers.
3. No `i64` column: all values stay `f64`.

Ordering, duplicate handling, gap checks and split reports use the exact `i64`
view. The evaluator sees an `f64` copy so a mis-specified variable still has a
value, but a time axis should not be passed as a feature.

`group_rows = 65_536` is 512 KiB per `f64` column, the measured I/O block from
#70. It is a streaming unit, not an L2 tile.

A wrong magic is "not an rcsr column store"; a different version says which
version the build reads and asks for `rcsr import` again; a truncated block is
refused with the group, the column and both byte counts.

## Backing: resident vs file

The measured rule in #70 is a capacity rule, not a speed rule: a resident
`Vec<f64>` is never slower than a file. The store therefore loads resident
whenever the estimate fits the budget.

Out of core, the empirical policy is:

| workload | mechanism |
|---|---|
| streamed groups / full scans | **`pread`** (`read_exact_at`) into reusable group buffers |
| sorted or coalesced reads | `pread` (1.2–2.8× faster than `mmap` at ≥1 GiB) |
| sparse random gather | never a syscall per element; read the containing group sequentially |
| `mmap` + `MADV_RANDOM` | not used here: group streaming with `pread` already bounds RSS, and needs no `madvise(MADV_DONTNEED)` |

The public shape is #70's `GroupView::col(c) -> &[f64]`, produced by
`FileStore::load_group`. No whole-column slice is exposed, so the backing can
change without touching the evaluator. The core crate remains pure `std`: no
`libc`, no third-party crate.

Selector policy is unchanged from #70: `Set` only when it is reused, resident
and small (≤ ~7 %, absolute cap 10⁶ rows); everything else is a `Range` or the
pinned one-multiply `Hash` predicate. The hashed sample and the split selector
are the same function, so row selection is reproducible across passes.

## Measured

### Streaming CSV (issue #126)

Machine: 31 GiB RAM, 8 threads, `/mnt/lager`. Fixture: a generated
**1,000,000 rows × 6 columns** CSV, **30,364,090 bytes (29.0 MiB)**. Both
binaries are release builds; `import` writes the same 58 MiB store.

| path | command | wall | peak RSS |
|---|---|---|---|
| old `read_to_string` | `rcsr import rows1m.csv out.rcsb --quiet` | 6.28 s | 93,316 KiB (91.1 MiB) |
| streaming | `rcsr import rows1m.csv out.rcsb --quiet` | 5.50 s | 63,492 KiB (62.0 MiB) |
| old `read_to_string` | `rcsr --data rows1m.csv --target y --generations 1 --pop 8 --baseline-samples 16 --seed 1` | 21.71 s | 565,812 KiB (552.5 MiB) |
| streaming | same command | 25.26 s | 432,164 KiB (422.0 MiB) |

The import difference is the 29 MiB whole-file `String` (plus allocator
growth); the search difference is larger because the peak included both the
file text and the prepared samples. The streaming path still builds the
resident `Table` and search samples, so the guard estimates those terms from the
measured shape.

### Column store

Machine: 31 GiB RAM, 8 threads, `/mnt/lager` on a spinning disk. Fixture: a
directly generated version-1 store, **100,000,000 rows × 5 columns = 4.0 GB**,
searched under `ulimit -v 900000` (900 MiB address space) with
`--max-memory 256MB --generations 2 --pop 16 --baseline-samples 512`:

```text
out of core: 100000000 rows x 5 cols (estimated 139.70 GiB);
             searched a 102401-row hash sample (seed 0)
wall      5.29 s
max RSS   43,356 KiB (42.3 MiB)
exit      0
```

The 139.70 GiB estimate is deliberately conservative (300 B/cell); the real
resident preparation of this shape would also not fit. The run read the whole
4 GB store for the sample scan while RSS stayed at one group plus the sample.

Guard, same machine, a 40.4 MiB / 1,000,000-row CSV under `ulimit -v 300000`
and `--max-memory 512MB`:

```text
error: input would exceed the memory budget
  guard.csv: 40.40 MiB on disk; estimated resident footprint 1.30 GiB
  memory budget is 488.28 MiB (512000000 bytes; set it with --max-memory)
  ... pass --force ... or convert it once and search out of core:
    rcsr import <input.csv> <store.rcsb>
exit 3
```

That refusal predates the #112 fixed-row calibration; the same shape is now
estimated higher (the calibration above records the current numbers).

## Measured: v1 versus v2

Machine: 31 GiB RAM, `/mnt/lager` spinning disk. Both files were read by the
same v2 binary after `POSIX_FADV_DONTNEED` (cold cache), twice, on the
3,943,856-row x 56-column hourly compact store (`group_rows = 65_536`):

| store | bytes | import wall | cold load + split wall |
|---|---|---|---|
| v1 (all f64, codes f64) | 1,924,648,087 | 202.8 s (v1 binary) | 11.06 s / 11.58 s |
| v2 (i64 time, u32 codes) | 1,814,220,771 | 140.3 s (v2 binary) | 9.22 s / 9.96 s |

v2 is **110,427,316 bytes smaller (-5.7 %)**. The saving is exactly the five
dictionary columns: `symbol` loses its unused 8-byte value block (-31.5 MB) and
all five code blocks halve from 8 to 4 bytes (-78.9 MB). The 100-symbol daily
panel (`panel_daily_100`, 154,674 rows x 28 columns) shrinks from 39,627,060 to
35,915,272 bytes (**-3.7 MB, -9.4 %**), because v1 also gave the timestamp a
dictionary and a code block; v2 stores it as i64 with no codes.

## Tests

`my-version/tests/store.rs` covers: byte-exact round trip (numeric columns,
numeric dictionaries and text dictionaries), a hand-built v1 store read with
correct short/last groups, a version mismatch, truncation, hash sampling and
row alignment, the CSV-vs-store search front, the guard (exit 3) for both
search and import, magic detection without the `.rcsb` extension, v2 per-column
dtypes, 4-byte code blocks, a 100-symbol daily-panel v1-to-v2 migration, and
the exact ms/us/ns time path.
