# Splitting data: i.i.d., time series and panel data

Searching for an equation is only half the job. The other half is deciding
**which rows the search is allowed to learn from, and which rows it is scored
on**. Get that wrong and the reported error is a lie — not because the maths is
wrong, but because the model was tested on information it had already seen.

This document explains the four splitting rules `rcsr` implements, why the
default (a random split) is wrong for anything with a time order, and how the
extras (`--gap`, `--folds`, `--split auto`) work. Once the split is chosen,
run the full audit in [`time-series.md`](time-series.md): purge/embargo,
overlapping labels, look-ahead preparation, lockboxes, per-group reporting,
multiple testing and costs.

## The one rule that matters: no leakage

A test set is only meaningful if the model could not have learned anything about
it. There are two ways to break that rule without noticing:

1. **Looking into the future.** A time series is ordered. If you shuffle the
   rows and hold out a random 25%, then for every training row there is close to
   it in time a test row — usually a test row *from tomorrow*. A model that
   interpolates between neighbouring training rows will look brilliant on the
   test set and be useless tomorrow. The shuffle itself is the leak.

2. **Leaking across groups.** Suppose you have OHLC prices for 200 stocks over
   the same year. The rows are not independent: all rows for one stock are
   driven by that stock's own dynamics and price level. A random split puts
   Monday of AAPL in training and Tuesday of AAPL in test; the model can learn
   "remember yesterday's AAPL price" rather than a rule that generalises. If
   what you actually want is a rule that works on a **new** stock, you must test
   on stocks the model has never seen.

Both are the same mistake: the test rows are not exchangeable with the training
rows, so the average test error estimates something other than what you want.

## The four modes

`--split <mode>` (or `--time` / `--group-by`, which imply a mode):

| mode | training rows | test rows | the question it answers |
|---|---|---|---|
| `random` | a random share | the remaining share | "can the model fit points from the same distribution?" |
| `temporal` | the first rows, in time | the last rows, in time | "can the model predict the *future*?" |
| `group` | whole groups | *other* whole groups | "does the rule generalise to a new stock / subject / sensor?" |
| `temporal-group` | the past of every group | the future of every group | "can the model predict tomorrow, for these kinds of series?" |

`random` is the default because for genuinely exchangeable rows (measurements
that could have been taken in any order) it is the right thing and it uses all
of the data evenly. It is *only* right there.

The other two axes that recur:

* **`--gap N`** — an *embargo*. In `temporal` mode `rcsr` drops `N` rows
  immediately after the boundary between training and testing; in
  `temporal-group` mode it drops `N` rows in each series. A suffixed value
  (`--gap 90m`, `1.5h`, `30s`, `2d`) is **wall time** and is converted with
  `ceil(seconds / median_dt)`, so an embargo is never silently rounded to zero;
  if explicit `--train-select`/`--test-select` replace the positional split,
  the gap is dropped with a warning. Why: neighbouring rows
  in a series are correlated, so the last training row and the first test row
  would otherwise be almost the same observation. Purged, embargoed splits are
  standard practice in financial machine learning for exactly this reason.
* **`--folds K`** — split the test region into `K` contiguous blocks and report
  each block's error as well as the mean. A model that gets lucky on one future
  window will show it: the block errors disagree. This is *evaluation only* — the
  model is trained once, so `K` folds cost one training run, not `K`.

## Building a split by hand (what `rcsr` does)

For `temporal`, with `n` rows, held-out fraction `f` and embargo `g`:

```
rows, in time order:   [ 0 ......................... n-1 ]
                        |<--- training --->|
                        |<-- (1-f)·n -->|  g   |<-- f·n -->|
                                            ^
                                     dropped (embargo)
```

Concretely, with 100 rows, `--test-frac 0.25 --gap 5`: rows 0–74 train, rows
75–79 dropped, rows 80–99 test.

### Purge by time with a target spec

`--gap` counts rows. On a panel of 100 symbols `--gap 2` is 0.02 hours, so it
cannot purge a label that looks one hour ahead. With a target spec
(`--target-spec`, see `docs/time-series.md` §2b) the engine purges by time: it
drops every training row whose stamp is closer than `entry_offset_seconds +
delay_seconds + horizon_seconds` to any test stamp, and every fit row that close
to a validation stamp, for every split mode and on the global time axis. The
counts and the realised minimum distances are in `split`
(`purged_rows`, `purge_seconds`, `validation_purge_seconds`). `--gap` applies on
top. Without a spec no row is dropped, and the realised distances are still
recorded.

### Validation slice and the test lockbox

The test block answers one question: how well does the chosen law generalise?
If you rank candidates by `test_error`, you have already spent that block — the
winner was chosen *because* it looked good on it, so its reported error is
optimistic. The fix is a second holdout inside the training period:

```sh
rcsr --data d.csv --target y --vars x1,x2 --split temporal --time t \
     --test-frac 0.25 --validation-frac 0.2 --gap 5
```

The training period is now split chronologically into **fit-train** and
**validation**, with the same embargo `--gap` on both sides of the validation
slice:

```
rows, in time order:   [ 0 ......................... n-1 ]
                        |<----- fit-train ----->|g|<- val ->|g|<- test ->|
```

Model selection ranks on the validation slice by default (`--select-by
validation`); `--select-by train` and `--select-by aicc` are also available.
`test_error` is **scoring-only** and is never a selection criterion. The JSON
front carries `train_error`, `validation_error`, `test_error` and
`archive_rank` (the position in the original archive), and the split object
carries `validation_rows`. If you ask for `--select-by validation` without a
slice, `rcsr` exits 2 instead of quietly falling back to train.

`--train-select` / `--test-select` still work exactly as before: they replace
the positional split and are the right tool when the boundary is a date or a
flag rather than a fraction. A validation slice is carved from the selected
training rows in the same way, so `--validation-frac` composes with them; the
purge note is still reported.

For `group`, groups (distinct values of `--group-by`) are ordered by a stable
hash of their name and whole groups move to the test side until the test share
is reached. **A group is never on both sides.** The assignment is deterministic,
so the same data and seed give the same split.

For `temporal-group`, the `temporal` recipe above is applied **inside each
group**, and the group's rows are ordered by `--time` first. Every group
contributes to both the training and the test side — so the test set still asks
"predict the future", but now over many series at once, which is far more data
than a single series gives you.

## Assigning rows by expression

Sometimes you know exactly which rows are training and which are validation: a
date range, a set of machines, a flag column. Say so directly.

```sh
# explicit sides
rcsr --data d.csv --target y --vars x1,x2 \
     --train-select "year < 2020" --test-select "year >= 2020"

# drop the commissioning rows first, then split the rest as usual
rcsr --data d.csv --target y --vars x1,x2 \
     --exclude "warmup == 1"
```

Any expression the model language accepts works, and it may name any numeric
column — not just the inputs. Rows are assigned by the same truth rule
everywhere: **a value greater than zero is true**.

| situation | rule | reported as |
|---|---|---|
| matches `--train-select` only | training | `train_rows` |
| matches `--test-select` only | test | `test_rows` |
| matches **both** | **training** | `overlap`, plus a note |
| matches **neither** (both given) | unused | `unassigned`, plus a note |
| only one selection given | the other side is the complement | — |

Two consequences worth stating plainly:

* **A selection overrides `--split`.** If you say which rows belong where, a
  positional mode cannot also say it, so the mode is ignored and the report
  says `mode=select` with a note naming the mode it replaced. `--exclude` is
  different: it only removes rows, so it composes with any mode — the population
  shrinks and the chosen mode splits what remains.
* **Nothing is discarded silently.** Overlap and unused counts are printed and
  appear in `--json` as `split.overlap` and `split.unassigned`, and the excluded
  count as `split.excluded`.

```
# row selection assigned 8 train / 4 test row(s)
# 1 row(s) matched both selections; they were assigned to training
# row selection replaced the random split
```

Selection is deterministic: the same table and the same expressions always give
the same sides, independent of `--seed`.

## Normalisation must not peek either

`rcsr` standardises each input column to zero mean and unit variance before
searching. It computes those statistics from the **training rows only**, then
applies the same numbers to the test rows. If you computed them over the whole
table, the test set's values would nudge the scaling of the inputs — a small,
sneaky leak. `tests/split.rs` guards this with an explicit test.

The returned equations are denormalised back to the original units, so they are
still written in terms of your columns.

## Letting the tool decide

`--split auto` inspects the columns that are neither the target nor a chosen
input and picks a mode:

* a **time-like** column (a name containing `time`/`date`, or a monotone,
  near-all-distinct column) → `temporal`;
* a **group-like** column (a text column, or a low-cardinality number) →
  `group`;
* both → `temporal-group`;
* neither → `random`.

It prints what it decided and why. If you do *not* pass `--split auto`, `rcsr`
still runs this check, but only warns — it will not silently change the split
you asked for. The warning describes what it *observed*, never a split it did
not apply:

```
warning: the rows do not look exchangeable: found a time-like column 'date' and a group-like column 'stock'
         a random split would let the model see the future (or the same
         group on both sides). Consider `--split auto` (it would
         choose: temporal-group), or pass explicitly:  --time date --group-by stock
```

`--vars` defaults to every numeric column except the target and any
`--time`/`--group-by` column: a timestamp must not become a predictor by
accident.

## Worked example: a panel of stocks

`panel.csv` has one row per (date, stock), with `close`, `volume` and a target
`next_close`:

```sh
# 3 stocks x 200 days, temporal split inside each stock:
rcsr --data panel.csv --target next_close --split temporal-group \
     --time date --group-by stock

# Reported as:
#   split=temporal-group | train rows: 450 | test rows: 150 | groups: 3/3 in train
```

Each stock is trained on its past and tested on its future; nothing about a
test day was visible during training, and no stock is missing from either side.

If instead the goal were "a model for a stock we have never traded", you would
use `--group-by stock` alone: whole stocks are held out.

## History and delayed variables

A forecast needs the past. `rcsr` supplies it explicitly:

* **`--lags K`** appends `x_lag1 … x_lagK` for every input variable. A lag is
  taken from the *previous row of the same series*, so nothing crosses a group
  boundary or a hole in the time axis.
* **`--history-frac f`** reserves the first `f` share of each series as
  *history*. Those rows are dropped from the scored data — they exist only so the
  rows just after them have a past. (With 100 rows and `--history-frac 0.05 --lags 3`,
  the first 5 rows of each series are history and row 6 onward is scored.)
* **`--gap-break f`** starts a new series whenever the step in `--time` is more
  than `f` times the median step. This is how a gap in a time series is handled:
  the lag chain stops at the gap, so a "yesterday" that is actually three months
  ago is never used as yesterday.

The series is the same unit the split works on, so `--lags` composes with every
split mode:

```sh
# AR-style model: predict tomorrow from today and yesterday, per stock,
# trained on the past and tested on the future.
rcsr --data panel.csv --target next_close --split temporal-group \
     --time date --group-by stock --lags 2 --history-frac 0.05 --gap-break 3
```

The engine's report says what it did:

```
# 1 series, 2 history row(s) dropped, 2 delayed column(s) added
# target: next_close = f(close,volume,close_lag1,volume_lag1,...) | split=temporal-group
```

`tests/split.rs` checks that a lag equals the previous row, that it never crosses
a group or a gap, and that history rows supply the first lags.

### Operator form (planned)

Today the delays are *columns*. The richer form is a `delay(x, k)` operator
inside expressions, alongside moving averages, a moving median and a derivative
— which would also allow things like `mavg(x, 5) * y`. That needs a row-aware
evaluator (each evaluation carrying the series position), so it is planned
rather than done. Materialised lags cover the common case today at no extra
evaluation cost.


## Row selectors (issue #70)

The split is stored as a **selector** over the one column-major table, not as a
second copy of the rows: `All`, a contiguous `Range`, a pinned `Hash { seed,
pct }` predicate, or an explicit `Set`. The text report names the selector that ran, for example:

```
split=temporal | selector: train range 0..600 (600 rows) | test range 600..800 (200 rows)
```

`RowSelector::auto` follows the measured policy: contiguous rows are a `Range`;
a **reused, resident** `Set` is used only up to ~7 % of the rows (and at most
1 000 000 rows); a one-shot `Set` never wins; everything else is a `Hash`
predicate. See [`memory-model.md`](memory-model.md) for the thresholds and the
bytes-per-cell table.
