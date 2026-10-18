# Dataset aggregates: statistics as expression leaves

A **dataset aggregate** reduces one input column over the rows the search is
training on. `mean(x1)` is the mean of column `x1`, not a function of the row
being evaluated: every row of the model sees the same number.

Aggregates are **opt-in**. They are accepted by `--ops` and listed by
`rcsr schema` under `"operators"`, but they are not in the default operator set,
so a broad search is not diluted by a statistic that is usually constant.

## Semantics

| operator | value over the training rows |
|---|---|
| `mean(x)` | arithmetic mean of the finite values |
| `std(x)` | **population** standard deviation, `sqrt(Σ(v-mean)² / n)` |
| `median(x)` | 50th percentile (linear interpolation) |
| `iqr(x)` | `q75 − q25`, the same interpolated quantiles |
| `sum(x)` | sum of the finite values |
| `amin(x)` | smallest finite value (printed `amin`, also `min(x)` with one argument) |
| `amax(x)` | largest finite value (printed `amax`, also `max(x)` with one argument) |
| `count()` | number of **training rows**, including rows with a missing cell |

The column is the aggregate's single child and must be a bare input variable:
`mean(x1)` is valid, `mean(x1 + 1)` is a syntax error. `count()` is zero-arity.
Quantiles use the same linear interpolation as the data-preparation code: for
sorted values `v[0..n]`, position `p·(n−1)` is interpolated between its floor
and ceiling. `std` divides by `n` (population), not `n−1`.

Every aggregate except `count` ignores non-finite cells. A column with **no
finite value at all** produces `NaN`; that is a non-finite prediction, which the
metric turns into `+inf` and the candidate is infeasible, exactly like
`ln(-1)`. No value is ever invented.

Weights do not change an aggregate: `mean(x)` is the unweighted mean, even when
`--weight` is in use. Weighting a dataset statistic would silently change what
the printed formula means; the unweighted definition is the interpretable one.

## The cache rule

Aggregates are computed **once**, before the first generation, into a small
`expr::AggTable` (one row of values per input column). The sampler reads that
table when it emits an `Op::Agg` node and stores the number in `Node::val`.

* `expr::eval` returns `val` without walking the column child and without
  looking at the row. Per-row evaluation cost does not grow.
* `fit` gives the node no constant slot and no gradient: it is a dataset fact,
  not a fitted parameter. `count_constants` ignores it, so `k_constants`,
  AIC/AICc/BIC and the `constants(e)` meta operator never count it.
* `simplify` never folds an aggregate (a fold would erase the name), and keeps
  its cached value; the column child is an input and cannot be rewritten by
  simplification.
* `mutate`/`crossover` refresh the cache bottom-up when the column under an
  aggregate changes, and the sampler always gives an aggregate a bare input
  child.
* A parsed aggregate (`mean(x1)` in `--eval-expr`, `--seed-expr` or a seed file)
  starts with `val = NaN`; `expr::bind_aggregates` resolves it against a table.
  `--eval-expr` builds that table from the **raw** training rows, because it
  evaluates in original units. Search seeds are resolved against the search's
  own (normalised) training sample.

## The leakage rule

The table is built from `search::run`'s **training** sample and from nothing
else. The test split is never read to compute a statistic:

* no test row contributes to `mean/std/median/iqr/sum/amin/amax/count`;
* the values are not recomputed per generation, not even when
  `--predictor-samples` scores a subset of the training rows — the subset
  changes which rows grade a candidate, not what the dataset is;
* the archive and every reported error use the same table, so train and test
  errors compare the same law.

A checkpoint stores the cached `val` with the tree, so a resumed run continues
with exactly the statistics the saved run used.

## Worked example

Column `x1 = 1, 2, 3, 4, 5` (five rows):

| expression | value |
|---|---|
| `mean(x1)` | `3` |
| `std(x1)` | `sqrt(2) ≈ 1.4142135623730951` |
| `median(x1)` | `3` |
| `iqr(x1)` | `4 − 2 = 2` |
| `sum(x1)` | `15` |
| `amin(x1)` | `1` |
| `amax(x1)` | `5` |
| `count()` | `5` |

A law can then be centred or scaled by the data itself:

```text
(x1 - median(x1)) / iqr(x1)
x1 / mean(x1)
x1 - mean(x1)
```

### Recovery

On `y = x1 / mean(x1)` over the non-round column
`0.7, 1.9, 2.3, 3.1, 5.4, 6.6, 8.2, 9.7` (mean `4.7375`), an unseeded search
with constant fitting **disabled** found

```text
y = (x1 / mean(x1))        complexity 4, training error 0.000e+00
```

The best plain-constant alternative, `x1 / 4.738` (three-decimal constants),
has training error `1.055e-04`. With fitting enabled, `x1 / c` can of course
reach the same slope by fitting `c`; the aggregate's advantage is that the
constant is *named* and comes from the data, and costs no parameter.

The acceptance-criterion target `y = (x1 - mean(x1)) / std(x1)` is recovered
too, by the same unseeded setup:

```text
y = ((x1 - mean(x1)) / std(x1))    complexity 7, training error 0.000e+00
```

```sh
cargo run --release -- --data data.csv --target y --vars x1 \
  --ops add,sub,mul,div,mean,median,count --center none --scale none
```

## Normalisation caveat

The search centres and scales its inputs (`--center mean --scale std` by
default), so the sample the aggregate sees is the **normalised** column, and
`mean(x1)` is the mean of that column (approximately `0`). The statistic is still correct for
the model the search builds, but it is not the raw data's mean. When a law is
meant to be read in original units, disable the transform:

```sh
--center none --scale none
```

With that, search space and original units coincide, the printed
`x1 / mean(x1)` re-scores identically under `--eval-expr`, and the statistic is
the one a reader expects. `denormalize` leaves an aggregate subtree untouched
for the same reason it leaves a meta argument untouched: rewriting the input
inside it would change what it means.

## Names and `--ops`

Canonical `--ops` names:

```text
mean, std, median, iqr, sum, amin, amax, count

They are ordinary operators for `--op-complexity` too (`--op-complexity "mean=2"`),
and the aggregate's column is its single child, so it is charged like any other
unary node.
```

`std` also accepts `stddev`; `amin`/`amax` also accept `agg_min`/`agg_max` and
`min_over`/`max_over`. `min` and `max` remain the **binary** primitives in
`--ops`; in an expression, `min(x1)`/`max(x1)` with one argument parse as the
column reductions and `min(a, b)`/`max(a, b)` stay binary. The printed form of
the reductions is `amin(x1)`/`amax(x1)`.

```sh
# aggregate-enabled search
cargo run --release -- --data data.csv --target y --vars x1 x2 \
  --ops add,sub,mul,div,mean,std,median,iqr

# score a printed law (statistics from the raw training rows)
cargo run --release -- --data data.csv --target y --vars x1 \
  --eval-expr "x1 / mean(x1)"
```

Aggregates are deliberately excluded from `default_set()`. `density` is not
implemented: its definition depends on the variable's type and unit, and a
wrong definition is worse than none.
