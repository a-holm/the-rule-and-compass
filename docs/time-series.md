# Time-series hazards

Every option here can change a result. None of them is silent: each is either
refused on time-ordered data, requires an explicit flag, or is reported in the
`preparation` block (text and JSON).

## Smoothing reads the past, not the future

`--smooth k` is a **trailing** moving average: row `t` averages rows
`t-k+1 ..= t`, shrinking at the start of each side. It never reads a later row,
so it cannot leak the future into training.

The old centred window is still available as `--smooth-centred k`. It averages
`t-k/2 ..= t+k/2` and therefore reads future bars. It is an explicit opt-in and
is reported as:

```text
# preparation: smooth=3 mode=centred cells_smoothed=12
# preparation: warning: centred smoothing reads future bars (look-ahead)
```

Windows still never cross the train/test boundary: each side is smoothed
separately. Do not use centred smoothing on a forecasting problem.

## `--fill interpolate` is refused on time-ordered data

Linear interpolation fills a gap from the **next** non-missing row. In a time
series that is a future bar. When `--time` is set (or the split is temporal), the
run fails:

```text
--fill interpolate reads the next bar, which is look-ahead on time-ordered data;
use --fill previous or --fill drop, or pass --allow-interpolate if the rows are
not a time series
```

`--fill previous` (last observation carried forward) and `--fill drop` are
causal. `--allow-interpolate` exists for genuinely exchangeable rows, e.g. a
spatial grid or a table where row order carries no time meaning.

## `--gap` needs a temporal split

The embargo (`--gap k`, in **rows**) drops rows between the training prefix and
the test blocks of a temporal split. It is meaningless for `random`/`group`
splits and for an explicit `--train-select`/`--test-select`, where the user's
windows define the boundary. In those cases `rcsr` emits:

```text
# note: the embargo gap does not apply to a row selection
```

and the reported `gap` is not a purge. If you select rows explicitly, leave your
own gap between the windows. `--gap` is row-based; across sessions, weekends or
timeframes a row gap is not a time gap. Use `--gap-break` for lag series.

## Time units, duplicates and monotonicity

The preparation block reports what the time column actually was:

```text
# preparation: time unit=ms rows=1200 duplicates=2 non_monotonic=1 converted=1200
# preparation: warning: 2 duplicate time instant(s); de-duplicate by instant ...
# preparation: warning: time is not monotonic in row order (1 step(s)); ...
```

- **Units**: numeric columns are matched to disjoint epoch windows (s, ms, us,
  ns) and normalised to seconds. A column that fits none or more than one is
  refused; pass `--time-unit`. Binance monthly files switched from milliseconds
  to microseconds in 2025, so this is not hypothetical.
- **Duplicates** are counted by **numeric instant**, so `1673343780` and
  `1673343780.0` (108 such rows in one BTC file) collide even though their text
  differs. A duplicated instant can make a temporal split place the same bar on
  both sides; de-duplicate upstream.
- **Monotonicity** is checked in row order. A non-monotonic step is reported.
  The temporal split orders rows itself, but a series that jumps backwards is
  usually a concatenation or a unit change and should be fixed at the source.

See also [`time-parsing.md`](time-parsing.md) for the strict ISO 8601/date-only
rules and [`data-splitting.md`](data-splitting.md) for the split semantics.

## Selector syntax

`--filter`, `--exclude`, `--train-select` and `--test-select` accept both
`and`/`or`/`xor` and the symbols `&&`/`||`, so
`--filter "x > 0.5 && y < 3"` is the same expression as
`--filter "x > 0.5 and y < 3"`.
