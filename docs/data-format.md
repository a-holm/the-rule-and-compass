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
#112): once the header and row count are known, the guard models

```text
text     = bytes read into one String
table    = rows x columns x 12 B          (f64 value + u32 dictionary code)
prep     = rows x used_columns x 24 B      (cells, train and test copies)
         + rows x 72 B                     (Vec headers and selectors)
estimate = (text + table + prep) x 1.25    (allocator slack)
```

`used_columns` is the search's projection (`--vars`, `--target`, `--weight`);
with no `--vars` the whole header is assumed. `rcsr import` has no search
preparation, so its guard uses the `text + table` terms only. The old `33 x file size` factor
was calibrated on ~6-column data and predicted 560 GB for a 17 GB, 93-column
panel that is at most ~27 GB resident. The text term is not optional: the
loader reads the whole CSV into one `String`, so a 17 GB CSV cannot be less
than 17 GB resident.

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

## File format, version 1

All integers and floating-point values are little-endian. The layout is:

```text
offset 0   8 bytes   magic "RCSRBIN\0"
offset 8   4 bytes   u32 version (currently 1)
offset 12  4 bytes   u32 header_len
offset 16  header    UTF-8 JSON, header_len bytes
           data      per-group, per-column f64 blocks
```

The header is self-describing and deterministic:

```json
{"format":"rcsr-column-store","version":1,"n_rows":100000,
 "n_cols":3,"group_rows":65536,"columns":[
  {"name":"x1","numeric":true,"dtype":"f64","levels":[],"has_codes":false,
   "offsets":[0,524288],"code_offsets":[]},
  {"name":"site","numeric":false,"dtype":"f64","levels":["A","B"],"has_codes":true,
   "offsets":[1048576,1572864],"code_offsets":[2097152,2621440]},
  {"name":"y","numeric":true,"dtype":"f64","levels":[],"has_codes":false,
   "offsets":[3145728,3670016],"code_offsets":[]}]}
```

* `n_rows`, `n_cols`, `group_rows`: the logical shape. `n_groups` is
  `ceil(n_rows / group_rows)`; the last group is short, never padded.
* `numeric`: the column has a value block.
* `dtype`: the value block's element type, `"f64"` (default) or `"i64"`.
  The time column is `i64` so nanosecond prints survive exactly; every
  feature and target column stays `f64`.
* `levels`: the dictionary (empty when the column has no dictionary).
* `has_codes`: the column also has a `u32` code block per group, stored as
  `f64` (small integers are exact in binary64) so the hot representation has
  one element type. Text columns have `numeric:false`, `has_codes:true` and no
  value blocks.
* `offsets[g]` / `code_offsets[g]`: byte offsets **relative to the data
  section**, for group `g`. The data section begins at `16 + header_len`.
  Relative offsets keep the header independent of its own length.

Blocks are written in this order, for every group `g`:

```text
for column c in 0..n_cols:
    value block   group_len(g) * 8 bytes   (NaN when the column has no values;
                                            i64::MIN when an i64 time cell is missing)
    code block    group_len(g) * 8 bytes   (only when has_codes)
```

### dtypes and the version rule

`dtype` is an **additive** field: a version-1 header that omits it is the
original all-`f64` layout, and both forms are read by this build. The version
stays `1` because the block sizes do not change — only the interpretation of
the same eight bytes for the time column. `rcsr import` selects the time
column in this order:

1. `--time <column>`, if given. Its cells must be whole numbers; otherwise
   the import refuses rather than silently writing a lossy store.
2. The first column whose name is time-like (`ts`, `timestamp`, `time`,
   `date`, `datetime`, `day`, `dt`) and whose cells are all integers.
3. No `i64` column: the store is byte-compatible with the old all-`f64`
   layout.

A column stored as `i64` is exposed to the evaluator as `f64` (for split
reports and accidental use), but ordering, duplicate handling and gap checks
read the exact `i64` view. The evaluator never performs arithmetic on the
integer view.

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

## Tests

`my-version/tests/store.rs` covers: byte-exact round trip (numeric columns,
numeric dictionaries and text dictionaries), a hand-built store read with
correct short/last groups, a version mismatch, truncation, hash sampling and
row alignment, the CSV-vs-store search front, the guard (exit 3) for both
search and import, magic detection without the `.rcsb` extension, and the
`i64` time path: exact ms/us/ns round trips, two prints 1 ns apart, duplicate
instants, gap detection, and an old v1 header with no `dtype` reading as
`f64`.
