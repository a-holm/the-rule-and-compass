# Preparing data: missing values, outliers, smoothing, scaling and filters

A symbolic search will fit whatever it is given, including gaps, spikes and
typos. `rcsr` therefore has an explicit preparation step. **Every option is
opt-in, and the defaults reproduce the behaviour of a plain run**: no filling
(a gap is an error, not a silent `NaN`), no outlier removal, no smoothing,
mean/standard-deviation normalisation and no row filter.

Time columns follow the strict policy in [`time-parsing.md`](time-parsing.md):
offset-aware ISO 8601, an explicit date-only choice, and numeric s/ms/us/ns
detection with a refusal on ambiguity.

The one rule that shapes the whole design is the same as for splitting:
**nothing about the test rows may influence the numbers used to prepare the
training rows.** See [`data-splitting.md`](data-splitting.md) for the split
rules themselves.

## The order of operations (and why)

```
raw table
  → row filter                     (which rows are candidates at all)
  → train/test split               (on the raw values)
  → fill or drop missing values    (statistics from training rows)
  → outlier removal                (median/IQR from training rows)
  → smoothing                      (inside each side, never across the cut)
  → centre/scale                   (statistics from the surviving training rows)
  → search and scoring
```

Everything that is a statistic — a mean, a median, a quantile, an IQR, a
standard deviation, a range — is computed from the **training rows only** and
then applied to both sides. `previous` and `interpolate` walk each side
separately in row order, so a training gap can never be filled from a test
value. Smoothing windows shrink at the train/test boundary for the same reason.
Rows that preparation removes disappear from both the training and the test
set; they are never silently kept.

## Copies, ordering and normalisation cost

The default path (no fill, no outlier rule, no smoothing) reads the used
columns straight from the loaded table: it does not make a row-major copy of
them (`#133`). Only a *mutating* stage (fill, outlier removal, smoothing) needs
that copy, and it is made once. The split itself is a selector over the table,
not a second copy.

Centre/scale still computes the training mean and standard deviation in the
same two-pass order, but it reads the column slices directly instead of
collecting a per-column copy. Median, min, max, IQR and range need an ordered
copy; mean/std (the default) does not sort. This changes no number: the same
training rows produce the same constants and the same samples in the same
order, including the final `(x - mean) / std` division (not a reciprocal
multiply).

Sample construction allocates the row vectors once and fills them in 256-row
blocks, so each source column is read sequentially rather than through 53
distant cache lines per row. For a single fold the pooled test block is built
directly and `Prepared::fold_samples` stays empty; consumers fall back to
`test`. With `--folds K > 1` the per-fold samples are still materialised, in
fold order.

## Measured (#133)

Release binary, Intel i7-3770K (8 threads), 31 GiB, Linux; fixture
`/mnt/lager/tmp/opencode/panel_hourly_v2.rcsb` (1.81 GB, 3,943,856 x 56) via
`examples/prep_measure`:

| fixture | load | prepare | total | peak RSS |
|---|---:|---:|---:|---:|
| 3,943,856 x 56 `.rcsb`, target `y60`, `--fill drop` | 2.12 s | **8.46 s** | 10.59 s | **3.64 GB** |
| 76,695 x 56 CSV, default split | 2.95 s | **0.21 s** | 3.16 s | 74 MB |

The 3.94M-row preparation target was `<= 12 s` and `<= 5.5 GB`; both are met.
A full 76k-row search (`--generations 1 --pop 30 --fit-iters 0`,
`--baseline-samples 64`) runs in **4.05 s** wall with 145 MB RSS, inside the
`<= 8 s` target. The earlier dead `x_raw`, the unconditional per-column sort,
the row-major copy for `--fill drop`, the table clone for an all-numeric fill,
and the duplicate pooled-test build are gone.

The SoA transpose count is pinned by a unit test: a five-generation run builds
the full-training view once and the pooled test view once — **2 transposes**,
not the previous 8 (the test prints the count).

## Ragged rows

The field count is part of the data contract. A data row with more or fewer
fields than the header cannot be interpreted without guessing which cell is
missing or extra, so the loader refuses the file instead of dropping the row:

```
error: line 3: expected 2 field(s), found 3 (ragged row; pass --skip-malformed to drop such rows instead)
```

The line number is 1-based in the file. Blank lines and lines whose first
non-space character is `#` are not data rows and never trigger this error.

`--skip-malformed` is the explicit opt-out: it drops ragged rows instead of
refusing the file and prints how many were dropped in the preparation report
(`# preparation: --skip-malformed dropped 1 ragged row(s)`; the JSON reports
`rows_skipped_malformed`). Do not confuse this with a missing **cell** in a
correct-width row: `x,y` followed by `1,` has the right field count and is a
gap for `--fill` to handle.

## Missing values

At load time a cell is **missing** when it is empty, `NA`, `NaN` or `null`,
case-insensitively. A column that otherwise contains numbers is still numeric
with gaps in it, so it remains usable as an input or a target.

Without `--fill`, a gap in a used column is an **error**, not a silent `NaN`:
complete tables stay bit-for-bit unchanged, and a gappy table fails with the
count and the affected columns:

```
error: 1 missing cell(s) in used columns (y: 1); pass --fill drop|mean|median|zero|previous|interpolate
```

That is deliberate. A `NaN` that reaches the search rejects every candidate it
touches, so the run finishes with an empty Pareto front and `inf` baselines —
which reads as "no structure found" when the real answer is "the data has
gaps". Failing loudly names the problem and the fix.

Feature columns produced by `rcsr enrich` follow a stricter rule than the search
preparation fills above: a trailing window that spans a missing bucket, listing,
delisting, migration or outage is `NaN`, and Wilder/EMA state restarts after the
gap. Preparation `--fill` is a separate, explicit choice made later; it never
changes the stored feature value or bridges a gap in the producer. The compact
52-name profile follows the same rule, even where the historical compact builder
used positional windows that bridged a hole.

`--fill <method>` chooses what to do:

| method | what happens | use when | do not use when |
|---|---|---|---|
| `none` (default) | gaps are an error that names the column and the fix | the table is complete | you have gaps — choose one of the methods below |
| `drop` | drop any row with a gap in a used column | gaps are rare and random | dropping loses an important subgroup, or rows are ordered (dropping creates holes) |
| `mean` | fill with the **training** column mean | values are roughly symmetric | the column is skewed or has outliers (the mean is dragged) |
| `median` | fill with the **training** column median | skewed values or outliers | the column is multi-modal and the median is unrepresentative |
| `zero` | fill with `0` | zero is a real, meaningful baseline (a count, a return) | zero is outside the data's range or changes the meaning |
| `previous` | carry the last non-missing value forward **inside the same side**, in row order | the series is a step function / last observation carried forward | the value changes quickly; the first gap in a side has no predecessor |
| `interpolate` | linear interpolation between the two neighbouring non-missing values **inside the same side** | a smooth, slowly changing series with isolated gaps | gaps are long, or at an edge (there is nothing to interpolate from) |

`previous` may propagate values that were themselves filled; `interpolate`
always uses the two observed neighbours. A gap that cannot be resolved (the
first row of a side for `previous`, a leading/trailing gap for `interpolate`)
is left missing and then its row is dropped, because leaving `NaN` in a filled
table defeats the purpose of asking for a fill.

**Unparseable cells.** Without `--fill`, a non-numeric token such as `abc`
still makes the column non-numeric and it cannot be used — a data error, as
before. With an explicit `--fill`, a junk token in an otherwise numeric **named**
input or target is treated as missing (so it is filled or its row dropped);
this is the only way typos are forgiven. A column never named with `--vars` or
`--target` is not coerced, and text columns with no numeric values at all
(dates, tickers) are never coerced.

## Outlier removal

`--outlier-iqr <k>` drops a row when **any used column** (input or target) lies
more than `k` interquartile ranges from that column's median:

```
lo = median − k · IQR        hi = median + k · IQR
IQR = Q3 − Q1
```

The median and quartiles are Hyndman–Fan type-7 quantiles computed from the
**training rows after filling**. Both sides are filtered with the same bounds.
A typical fence is `1.5`; `3` is conservative and removes only extreme points.
Do not use it when the "outliers" are the rare events you are trying to model
(a crisis, a spike, a failure): removing them answers a different question.

## Smoothing

`--smooth <k>` replaces every value in every used column with the mean of a
`k`-wide window centred on it. The window starts `floor(k/2)` rows before the
value and is clipped at the edge of its side, so it **shrinks** there rather
than reading across the train/test boundary. With `k = 3` row `i` becomes the
mean of `i-1, i, i+1`; at the first row it is the mean of rows `0` and `1`.
Missing cells in a window are skipped.

Smoothing is applied **after** outlier removal, so a spike is removed rather
than smeared into its neighbours. It is a moving average, not a low-pass
filter: it shifts peaks and destroys sharp transitions. Do not smooth a
piecewise or threshold response, or a target whose whole signal is the spike.
`--smooth 1` is a no-op.

## Normalisation: `--center` and `--scale`

Inputs are transformed affinely before the search:

```
x' = (x − center) / scale
```

| `--center` | subtract | | `--scale` | divide by |
|---|---|---|---|---|
| `mean` (default) | training mean | | `std` (default) | training standard deviation |
| `median` | training median | | `iqr` | Q3 − Q1 |
| `min` | training minimum | | `range` | max − min |
| `max` | training maximum | | `none` | 1 |
| `none` | 0 | | | |

The default is exactly today's mean/std normalisation. Every combination stays
affine, so the printed equation is denormalised with the same two numbers:

```
original = normalised · scale + center
```

`Prepared::means` holds the centre and `Prepared::stds` the scale for every
combination, which is why `--eval-expr`, the `y = ...` output and the fitted
constants remain in original units. A degenerate scale (zero IQR or range, or
a standard deviation below `1e-12`) is replaced by `1` so the transform stays
invertible.

Median/IQR is the robust choice for skewed or outlier-prone columns; min/max
range scaling makes the largest and smallest training value map to 0 and 1.
`none` is useful when units are already meaningful and the search should see
them; it is a bad idea if columns differ by orders of magnitude.

## Row filters

`--filter <expr>` keeps only the rows where the expression evaluates to a
positive number; `0`, negative and `NaN` drop the row. The expression is
parsed by the same parser as `--eval-expr`, so it may reference **any numeric
column by name** (not just the chosen inputs) and supports arithmetic,
comparisons (`<`, `<=`, `>`, `>=`, `==`), `and`, `or`, `xor`, `not`, functions
and parentheses:

```sh
rcsr --data panel.csv --target next_close --filter "volume > 100000 and close > 0"
```

The filter is applied **before** the split, so `--test-frac` is a fraction of
the rows that pass it. A syntax error is a usage error (exit code 2) and the
message contains the parser's 1-based column:

```
error: cannot parse --filter: unexpected end of input (column 5)
  expression: x >
```

Filtering on the target is allowed, but be careful: selecting rows by the value
you are trying to predict changes the question being asked.

## The preparation report

The text report prints one commented line per active transformation (silent
when everything is at its default):

```
# preparation: fill=median rows_filled=2 rows_dropped_missing=0
# preparation: outlier-iqr=1.5 rows_dropped_outlier=1
# preparation: smooth=3 cells_smoothed=42
# preparation: center=median scale=iqr
# preparation: filter removed 5 row(s)
```

`--json` always includes a `preparation` object with the methods and counts:

```json
"preparation": {
  "fill": "median",
  "rows_filled": 12,
  "rows_dropped_missing": 3,
  "rows_dropped_outlier": 2,
  "smoothed": 0,
  "center": "mean",
  "scale": "std",
  "filtered": 5
}
```

`--ndjson` emits a matching `preparation` event before the solutions when a
transformation is active:

```json
{"event":"preparation","fill":"median","rows_filled":12,...}
```

`rows_filled` counts rows with at least one filled cell; `smoothed` counts
cells replaced by a smoothing window.

When the input is a `.rcsb` store the same object also carries the memory
decision (issue #112): `store_memory` is the header-derived estimate with every
term (`estimate_bytes`, `rows`, `columns`, `used_columns`, `scan_columns`,
`storage_bytes`, `directory_bytes`, `group_buffer_bytes`, `search_bytes`,
`fit_copy_bytes`, `fixed_bytes`, `slack_bytes`, `saturated`, `basis`),
`full_rows` is `true` for a full load, `false` for an explicit `--sample-rows`
sample and `null` for a CSV, and `sampled_rows` is the requested sample size.
The text report prints the same basis:

```
# preparation: memory estimate 7826847852 B from store header: 3943856 rows x 56 columns (54 search + 2 scan used); basis: storage 1814179393 B + directory/group 30150456 B + search 3470593280 B + fit 0 B + fixed 946525440 B + slack 1565369571 B
```

A sampled run adds `# preparation: warning: full_rows=false; used an explicit
N-row sample, not the full store`, so a sample can never be mistaken for the
full data.

## Worked example

`gap.csv` has eight rows and two gaps in `x`, one in the target. Train on the
first six rows and test on the last two, fill with the training median, remove
nothing, smooth with a 3-row window and use robust scaling:

```sh
rcsr --data gap.csv --target y --vars x --split temporal --time t \
     --test-frac 0.25 --fill median --smooth 3 --center median --scale iqr
```

```
# preparation: fill=median rows_filled=2 rows_dropped_missing=0
# preparation: smooth=3 cells_smoothed=16
# preparation: center=median scale=iqr
# target: y = f(x) | split=temporal | train rows: 6 | test rows: 2
```

The fill statistic is the median of the six training rows; the two test rows
cannot move it. The test rows are transformed with the same centre and scale,
so their error is comparable, and the reported equation is converted back to
the units of `x` and `y`.

## Checks

`tests/prep.rs` covers each method with hand-computed values, proves that a
wild test-side value cannot change a mean/median fill, that `drop` removes
exactly the expected rows, that every centre/scale pair denormalises exactly,
that a filter keeps exactly the matching rows, and that a handful of
median-filled gaps still recover `y = 2x + 1`. The existing split tests run
unmodified, which is the regression guard for the default configuration.
