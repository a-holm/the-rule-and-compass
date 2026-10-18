# Splitting data: i.i.d., time series and panel data

Searching for an equation is only half the job. The other half is deciding
**which rows the search is allowed to learn from, and which rows it is scored
on**. Get that wrong and the reported error is a lie — not because the maths is
wrong, but because the model was tested on information it had already seen.

This document explains the four splitting rules `rcsr` implements, why the
default (a random split) is wrong for anything with a time order, and how the
extras (`--gap`, `--folds`, `--split auto`) work.

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
  `temporal-group` mode it drops `N` rows in each series. Why: neighbouring rows
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

For `group`, groups (distinct values of `--group-by`) are ordered by a stable
hash of their name and whole groups move to the test side until the test share
is reached. **A group is never on both sides.** The assignment is deterministic,
so the same data and seed give the same split.

For `temporal-group`, the `temporal` recipe above is applied **inside each
group**, and the group's rows are ordered by `--time` first. Every group
contributes to both the training and the test side — so the test set still asks
"predict the future", but now over many series at once, which is far more data
than a single series gives you.

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
you asked for:

```
warning: the rows do not look exchangeable: found both a time-like and a
         group-like column; using a temporal split inside each group
         a random split would let the model see the future (or the same
         group on both sides). Consider `--split auto` (chosen: temporal-group),
         or pass explicitly:  --time date --group-by stock
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

## What is not (yet) here

A forecast usually needs **history**: `y_t` may depend on `x_{t-1}`, `x_{t-2}`,
or on a moving average. The clean way to express that is as *operators over the
series* — `delay(x, k)`, a moving average, a derivative — evaluated per row with
access to the previous rows. `rcsr`'s evaluator is currently row-local, so the
planned route is:

1. an optional "history" reserve at the start of each series (rows kept for
   their past values but not scored), and
2. a `delay` / moving-average / derivative operator family that indexes the
   series, with a maximum history offset.

These are semantic extensions of the same split machinery: the series is the
unit of order, and the split decides which part of each series is the future.
A cheaper stop-gap that already works today is to feed **lagged columns**
explicitly (`close_lag1`, `close_lag2`, …) as ordinary inputs and let the split
handle the rest.
