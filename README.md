# rule-and-compass (🎻 `rcsr`)

A **symbolic-regression** engine written in Rust.

Given a table of numeric columns, `rcsr` searches for short closed-form
expressions `y = f(x1, x2, ...)` that fit the data, trading **error** against
**complexity**, and reports the non-dominated set (the *Pareto front*).

> The name is an old-fashioned English pub sign (1700s style, like *The Three
> Compasses*): the **rule** is the law/equation we recover, the **compass** is
> the measurement — the instrument that finds it.

## What it does

Symbolic regression finds formulas, not just fitted parameters. `rcsr` evolves a
population of expression trees, scores each on **two axes (error and
complexity)**, and returns the front of non-dominated equations — the most
accurate equation at each level of complexity.

Constants inside a candidate are not guessed: each candidate's constants are
fitted by **Levenberg–Marquardt**, so a candidate with the right *shape* is
immediately usable even if its numbers started out wrong.

See [`docs/how-symbolic-regression-works.md`](docs/how-symbolic-regression-works.md)
for a plain-language walkthrough,
[`docs/optimisation.md`](docs/optimisation.md) for the theory and measurements
behind the search strategy, and [`SPEC.md`](SPEC.md) for the precise behaviour.

### Implicit relations

Some laws are level sets rather than functions: a circle `x² + y² = 1` has no
single-valued `y = g(x)`. `--implicit` appends the target as one more input and
searches `f(x1, …, xn, y) = 0`. `--metric implicit` (or the shorthand
`--implicit-derivative`) scores the distance to the level set,
`|f| / ||∇f||`, using the adjoint gradient the fitter already computes. See
[`docs/implicit.md`](docs/implicit.md) for the model, both implicit metrics, the
zero-gradient guard and a worked unit-circle example.

## Build & run

Requires a Rust toolchain (`rustup` recommended). No third-party crates are used.

```sh
cd my-version
cargo build --release

# demo on the bundled sample: recover y = 2x + 1
cargo run --release -- \
  --data examples/line.csv --target y --vars x \
  --generations 200 --islands 4 --pop 200 --metric absolute
```

Output (human-readable form):

```
  1  complexity=1   train_err=1.455120      test_err=1.216192       y = (0.8446 * x)
  2  complexity=2   train_err=1.096793      test_err=0.609935       y = exp((0.8446 * x))
  3  complexity=3   train_err=0.999970      test_err=1.000090       y = (2.0044 * x)
  4  complexity=4   train_err=0.651494      test_err=0.423752       y = (exp((0.8446 * x)) + (0.8446 * x))
  5  complexity=5   train_err=0.000001      test_err=0.000001       y = (1 + (2 * x))
```

The last row is the answer: `y = 1 + 2x`, exact to the rounding of the data, and
the smaller rows are the cheaper approximations it beat along the way.

```sh
# a trigonometric example: recover y = sin(x)
cargo run --release -- --data examples/sine.csv --target y --vars x \
  --generations 300 --islands 4 --pop 300 --metric squared
```

### Score a formula without searching

`--eval-expr` parses a formula, scores it on the split and exits — no search. It
is the inverse of the printed output, so a printed right-hand side can be pasted
straight back in to re-check it:

```sh
# original units: variables by name, *, ^, sin(...), if(...), ...
cargo run --release -- --data examples/line.csv --target y --vars x \
  --eval-expr "2*x + 1"
#   1  complexity=5   k_constants=2   train_err=0.000000      test_err=0.000000      y = ((2 * x) + 1)
```

A syntax error names the offending column:
`--eval-expr "2 * (x + )"` fails with
`expected a number, variable, function or '(' (column 10)` and exit code 2.

### Walk-forward evaluation and backtest

`rcsr walkforward` evaluates one supplied expression over time windows on a
`.rcsb` store (CSV is also accepted), applies a long/flat rule and charges costs
on position changes. It reads the store group by group, uses the same f64
evaluator as `--eval-expr`, and reports per-window and pooled Sharpe, CAGR, max
drawdown, turnover, exposure, hit rate, break-even hit rate, trades and rows as
JSON or Markdown:

```sh
rcsr walkforward --data panel_daily_long.rcsb --target fwd1 \
  --expr 'logistic(r60) - tanh(r120)' --vars r60,r120 \
  --time ts --group-by symbol --horizon 172800 \
  --window '2019,0,1546128000,1546300800,1577836800' \
  --cost-bps 10 --threshold-quantile 0.7 --json
```

Test windows are defined only by time; the purge gap must be at least the
horizon, and the threshold comes from training (or an optional trailing
validation slice), never from the test block. See
[`docs/walkforward.md`](docs/walkforward.md) for the semantics, metric
definitions and the reference-agreement check against the Python evaluator.

Grammar: `+ - * / ^` (`^` is right-associative and binds tighter than unary
minus), function calls with any arity (`sin(x)`, `min(x, y)`, `if(c, a, b)`),
comparisons (`x > 0`, `x <= y`, `x == 0`), the logical operators
`and`/`or`/`xor`/`not`, the positional constraint `between(x, lo, hi)`,
`require`/`constrain` (a violated constraint makes the candidate infeasible
rather than merely wrong), the opt-in meta operators `complexity`,
`variables`, `constants`, `contains` and `contains_op` (see
[`docs/meta-operators.md`](docs/meta-operators.md)), the opt-in dataset
aggregates `mean(x1)`, `std(x1)`, `median(x1)`, `iqr(x1)`, `sum(x1)`,
`amin(x1)`, `amax(x1)` and `count()` (see
[`docs/aggregation.md`](docs/aggregation.md)), parentheses, the constants
`pi` and `e`, and variable names from your table (the `x0`, `x1`, … fallback
also works). Comparisons are non-associative, so write
`(a < b) and (b < c)`.

`cargo test` runs the unit tests and the end-to-end recovery tests
(`y = x`, `y = x²`, `y = x1·x2`), plus a regression test that constant fitting
reaches machine precision.

## GUI

A local web front end ships as a second binary. It is a self-contained
single-page app embedded in the executable, so it has no CDN, framework or
network dependency at runtime and binds loopback only:

```sh
cd my-version
cargo build --release
./target/release/rcsr_serve --port 8080 --open
# open http://127.0.0.1:8080
```

The tour: paste or open a CSV/TSV and give each column a role (input, target,
time, group, weight, ignore); the Search pane is generated from `rcsr schema`,
with metric, generations, population, islands, complexity cap and seed pinned
and every other flag under **More options**; Run streams NDJSON and grows a
log-scale Pareto chart as solutions arrive; Solutions lists the front and the
selected row shows the formula, constants ± standard errors, train/test error,
AICc/BIC, Durbin–Watson, residual checks, baselines, bootstrap and
cross-validation; **Copy as CLI** emits the exact `rcsr` invocation (including a
here-document when the table was pasted). The page consumes only the JSON
contract: `/api/schema`, `/api/example` and the `/api/run` event stream. See
[`docs/gui.md`](docs/gui.md) for the as-built notes, the event shapes and what
is deferred.
### Start from what you already know

`--seed-expr` (repeatable) and `--seed-file` inject prior solutions into the
search instead of making it start from random trees alone:

```sh
cargo run --release -- --data examples/sine.csv --target y --vars x \
  --seed-expr "sin(x)" --generations 15
```

and reports the processed seed on stderr:

```text
# seeded: complexity=... train_error=0.000000  y = sin(x)
```

Each seed is parsed in original units, simplified, its constants fitted on the
training sample, then inserted into the archive and the initial population of
**every** island. A seed is an elite: it is kept even above `--max-complexity`,
which limits generated offspring, and the run prints a `# note:` when that
happens. `--seed-file` takes one expression per line, skipping `#` comments and
blank lines; both flags can be combined. A malformed seed is a usage error
(exit 2) naming the flag, text and parser column; an unreadable file is an
input error (exit 3). `--json` reports a `seeds` array and `--ndjson` a `seed`
event per seed. See [`docs/seeds.md`](docs/seeds.md) for the processing order,
the elite rule, the file format and the measured before/after numbers.

## Building blocks

`--function` (repeatable, up to four) supplies a sub-formula the search may
reuse as an operator of its own:

```sh
cargo run --release -- --data examples/sine.csv --target y --vars x \
  --function "wave(a)=sin(a)" --generations 100
```

A definition is `name(a,b)=<expr>` or `name=<expr>` (parameters inferred from
the free variables). The recovered model prints the name, e.g. `y = wave(x)`,
and the definition's constants are shared by every call.
A block costs 1 by default and can be priced with
`--op-complexity "wave=2"` (or `custom0=2`); the Pareto front ranks on that
cost. See [`docs/building-blocks.md`](docs/building-blocks.md) for the syntax,
parameter inference, shared-constant fitting and limits.

## Choosing what "complexity" means

By default complexity is the node count. `--op-complexity "pow=2,sin=1.5"`,
`--complexity-weights "var=1,const=3"` and `--depth-weight 0.5` price
operators, terminals and nesting instead. The configured value is what the
Pareto front is ranked on, what `--max-complexity` limits, and what every row
reports; the defaults reproduce the node count exactly. See
[`docs/complexity.md`](docs/complexity.md) for the formula and worked examples.

## Dimensional constraints

`--units 'x=m,t=s,y=m/s'` declares the dimensions of the inputs and the target;
`--require-dimensions` then rejects every candidate whose dimensions cannot be
reconciled with the target's with an infinite error. The algebra knows the SI
names `m`, `s`, `kg`, `A`, `K`, `mol` and `cd`, and **any other identifier is
its own base dimension**, so `price=USD,volume=shares` catches `price + volume`.
It models dimensions only, not unit identity or scale: declare `volume=m^3`, not
`litre`. Fitted constants are allowed to carry dimensions (a coefficient absorbs
a unit ratio, an intercept absorbs the target's), and each solution reports their
apparent dimensions as a plausibility check. Undeclared names are dimensionless
and column names are never inspected. Units are strictly optional in three
layers: no `--units` changes nothing and adds no output fields; `--units` alone
only reports and never rejects; `--require-dimensions` turns the hard filter on
and is a usage error without `--units`. See
[`docs/units.md`](docs/units.md) for the operator table, the radian/degree and
dB/pH caveats, a worked example and the limitations of a hard filter.

## Meta operators

`complexity(e)`, `variables(e)`, `constants(e)`, `contains(e)` and
`contains_op(e)` describe the shape of a subexpression: node count, distinct
input variables, constant leaves, and whether an input or an operator is
present. They are opt-in (`--ops complexity,...`), return one value per model
rather than one per row, and are cached when the node is built. Because they
describe the model rather than the data, they are useful for self-describing
models and structural conditions, not as a source of row-level signal. See
[`docs/meta-operators.md`](docs/meta-operators.md) for the semantics table and
the caching rule.

## Dataset aggregates

`mean(x1)`, `std(x1)`, `median(x1)`, `iqr(x1)`, `sum(x1)`, `amin(x1)`,
`amax(x1)` and `count()` reduce one input column over the training rows. They
are opt-in (`--ops mean,std,...`), computed once into a small per-column table
before the search and cached in the node, so per-row evaluation cost does not
grow. They are dataset facts, not fitted parameters: no constant slot, no
gradient and no `k_constants`; and the table is built from the training rows
only, never the test split. See
[`docs/aggregation.md`](docs/aggregation.md) for the semantics table, the cache
and leakage rules, and a worked recovery example.

## Save and resume a run

A long search can be checkpointed and continued later. The checkpoint carries
the complete search state — configuration, populations, per-island RNG states,
stagnation and patience counters, archive and a data fingerprint — so a resumed
run follows the identical trajectory:

```sh
rcsr --data examples/line.csv --target y --vars x \
     --generations 20 --seed 7 --save run.rcsr
rcsr --data examples/line.csv --target y --vars x \
     --generations 40 --seed 7 --resume run.rcsr
```

`--save-every <N>` sets the interval (default 25 generations); a checkpoint is
always written on exit. Saving does not consume randomness, so it cannot change
the result. The saved search settings win on resume; only the generation budget
may be raised, and data that does not match the saved fingerprint is refused.
See [`docs/checkpoint.md`](docs/checkpoint.md) for the versioned text format,
the determinism guarantee and its limits.

## Using it from other programs

`rcsr` is built to be driven by scripts and agents.

* **stdout carries only the result; stderr carries the human log.** So
  `rcsr ... --json > result.json` and `rcsr ... --json | jq .pareto[0]` do the
  right thing.
* **`rcsr schema`** prints a self-describing JSON document (every option, its
  type, default and meaning; plus the exit codes). An agent can discover the
  interface without reading the source.
* **`--json`** emits one JSON object; **`--ndjson`** streams one object per line
  (`start`, `progress`…, `solution`…, `done`) for long runs.
* **Deterministic**: the same `--seed` gives the same answer, and the seed is
  echoed in the output.
* **Bounded time**: `--time-budget <seconds>` guarantees the process returns.
* **stdin**: `--data -` reads the table from standard input.
* **Bounded memory**: `--max-memory 2GB` refuses a CSV whose estimated
  footprint (from rows x columns, not `33 x` the file) would exceed the budget,
  with exit 3 and a fix, instead of being SIGABRT/SIGKILL'ed. `rcsr import`
  writes a versioned `.rcsb` store that can be scanned group by group and
  searched from a bounded hash sample when it does not fit; see
  [`docs/data-format.md`](docs/data-format.md).
* **Stable exit codes**: `0` success, `2` usage error, `3` input error,
  `4` no finite solution.

```sh
# machine-readable
rcsr --data examples/sine.csv --target y --json | jq '.pareto[-1].expression'

# discover the interface
rcsr schema

# stream progress as NDJSON while it runs
rcsr --data examples/sine.csv --target y --ndjson --progress --time-budget 5

# read the table from a pipe
cat examples/line.csv | rcsr --data - --target y --top 3
```

```jsonc
// rcsr --data examples/line.csv --target y --json
{
  "tool": "rcsr", "version": "0.1.0", "seed": 0, "metric": "absolute",
  "target": "y", "variables": ["x"], "train_rows": 30, "test_rows": 10,
  "generations_run": 200, "converged_early": false, "elapsed_seconds": 1.9,
  "null_control": null,
  "baselines": {
    "mean":   { "train_error": 2.0034, "test_error": 2.2496 },
    "linear": { "train_error": 4.8e-7, "test_error": 7.6e-7 },
    "knn":    { "k": 5, "train_error": 0.1299, "test_error": 0.2503,
                "reference_rows": 30, "train_query_rows": 30,
                "train_query_sem": null, "test_query_rows": 10,
                "test_query_sem": null, "capped": false }
  },
  "residual_tests": {
    "n": 30, "lags": 6, "durbin_watson": 2.0,
    "ljung_box":     { "statistic": 2.01, "df": 6, "p_value": 0.919 },
    "breusch_pagan": { "statistic": 0.159, "df": 1, "p_value": 0.690 },
    "skewness": 0.008, "kurtosis": -0.667
  },
  "pareto": [
    { "rank": 5, "complexity": 5, "train_error": 5.3e-7,
      "test_error": 5.2e-7,
      "constant_errors": {
        "dof": 28, "ill_conditioned": false,
        "constants": [ { "value": 1.0, "std_error": 4.1e-7, "relative": 4.1e-7 },
                       { "value": 2.0, "std_error": 3.2e-7, "relative": 1.6e-7 } ] },
      "expression": "(1 + (2 * x))" }
  ]
}
```

A graphical front end for the same contract is designed in
[`docs/gui.md`](docs/gui.md): schema-driven forms, a live Pareto plot, the
split made visible on the grid, and a "Copy as CLI" action on every search.

### Full option list

```
rcsr --data <file|-> --target <col> [--vars a,b,..]
     [--dtype col=continuous|boolean|categorical,..] [--max-levels N]
     [--metric <name>] [--generations N] [--islands I]
     [--pop P] [--max-complexity K] [--seed S] [--test-frac f] [--ops list]
     [--seed-expr expr] [--seed-file path]
     [--op-complexity spec] [--complexity-weights spec] [--depth-weight w]
     [--split random|temporal|group|temporal-group|auto] [--time col]
     [--group-by col] [--gap N] [--folds K] [--lags K] [--history-frac f]
     [--validation-frac f] [--select-by train|validation|aicc]
     [--gap-break f] [--fill method] [--outlier-iqr k] [--smooth k]
     [--center mean|median|min|max|none] [--scale std|iqr|range|none]
     [--filter expr] [--exclude expr] [--train-select expr] [--test-select expr]
     [--null-runs N] [--null-block N] [--baseline-samples N] [--bootstrap-error N] [--cv mode] [--cv-k K]
     [--stability B] [--eval-expr expr]
     [--units name=unit,..] [--require-dimensions]
     [--fit-iters N] [--no-fit] [--no-dedup] [--no-simd] [--predictor-samples N]
     [--no-front-parents]
     [--fit-samples N]
     [--patience N]
     [--time-budget seconds] [--save file] [--save-every N] [--resume file]
     [--format text|json|ndjson] [--progress]
     [--top N] [--quiet] [--out file.tsv]
     [--report summary|sensitivity|benchmark] [--report-out file.md]
     [--report-repeats N]
```

Input may be CSV, TSV, or whitespace-delimited, with a header row. Numeric
columns can be used as inputs or target; text columns (dates, ticker symbols)
are kept and can be used for splitting. Blank lines and `#` comment lines are
ignored. A data row with a different number of fields from the header is
refused with an input error (exit 3) naming the 1-based line and both counts;
`--skip-malformed` drops such rows instead and reports how many it skipped.
A numeric cell may be missing (empty,
`NA`, `NaN` or `null`, case-insensitively); `--fill` chooses whether those rows
are dropped or filled, and the default leaves them untouched.

## Choosing how rows are split

Which rows go to training and which to testing is the difference between a
number you can trust and one you cannot. `--split` selects the rule:

| `--split` | training rows | test rows | use when |
|---|---|---|---|
| `random` (default) | a random 75% | the rest | rows are exchangeable (i.i.d. samples) |
| `temporal` | the **first** rows | the **last** rows | one time series; the future must stay unseen |
| `group` | whole groups | other whole groups | you must generalise to *unseen* entities |
| `temporal-group` | the past of every group | the future of every group | panel data: many series measured over one period |
| `auto` | inferred from the columns, and printed | | you want the tool to decide and say why |

A random split of a time series lets the model see the future, and a random
split of panel data puts the same stock on both sides. `rcsr` warns when the
rows look non-exchangeable and a random split was requested.

```sh
# One series: train on the past, test on the future, with a 5-row embargo
# between them so autocorrelation cannot leak across the boundary.
rcsr --data prices.csv --target y --split temporal --time date --gap 5

# Panel data (OHLC for several symbols over the same period): train on the
# past of every symbol, test on its future.
rcsr --data panel.csv --target next_close --split temporal-group \
     --time date --group-by symbol

# Generalise to unseen symbols: hold out whole symbols.
rcsr --data panel.csv --target next_close --group-by symbol

# Let it decide, and report the choice:
rcsr --data panel.csv --target next_close --split auto

# You know exactly which rows belong where: say so.
rcsr --data d.csv --target y --vars x1,x2 \
     --train-select "year < 2020" --test-select "year >= 2020"

# Drop rows first, then let a mode split the rest.
rcsr --data d.csv --target y --vars x1,x2 --exclude "warmup == 1"
```

* `--time <col>` may be a number or a date (`2024-01-31`, `2024/01/31`,
  `2024-01-31T09:30:00`).
* `--group-by <col>` may be text (tickers) or a low-cardinality number.
* `--gap N` drops `N` rows after the train/test boundary (an *embargo*), per
  series in `temporal-group` mode.
* `--folds K` splits the test region into `K` contiguous blocks and reports the
  error of each block next to the mean, so an unstable model cannot hide behind
  a lucky pooled number.
* `--validation-frac f` holds out the last `f` of the training period as a
  chronological validation slice (with a `--gap` purge either side) and
  `--select-by train|validation|aicc` chooses the ranking criterion; the test
  block is a scoring-only lockbox and is never used to select. See
  [`docs/overfitting.md`](docs/overfitting.md) for the lockbox argument.
* Normalisation statistics are computed from the **training rows only**; the
  test set never influences the model's input scaling.
* `--train-select` / `--test-select` assign rows by expression (`> 0` is true)
  and override `--split`; `--exclude` drops rows and composes with any mode.
  A row matching both selections goes to training, one matching neither is
  unused, and both counts are reported — nothing is discarded silently.

See [`docs/data-splitting.md`](docs/data-splitting.md) for the reasoning and the
worked examples.

## Cleaning and preparing data

Real tables have gaps, spikes and rows that should never be scored. Every
preparation option is opt-in, and the defaults are exactly the plain mean/std
run:

| flag | effect |
|---|---|
| `--fill <method>` | what to do with a missing cell: `none` (default), `drop`, `mean`, `median`, `zero`, `previous`, `interpolate` |
| `--outlier-iqr k` | drop rows beyond `k` interquartile ranges from the column median |
| `--smooth k` | `k`-wide centred moving average per column; edges shrink |
| `--center <how>` | what to subtract before the search: `mean` (default), `median`, `min`, `max`, `none` |
| `--scale <how>` | what to divide by: `std` (default), `iqr`, `range`, `none` |
| `--filter expr` | keep rows where the expression is `> 0`, e.g. `"volume > 1e5 and close > 0"` |
| `--weight col` | per-row weight column (finite, `>= 0`, at least one positive); a weight of 2 means "this row twice" |
| `--skip-malformed` | drop rows whose field count differs from the header and report the count (default: refuse the file with exit 3) |

```sh
# Robust preparation: fill gaps with the training median, drop extreme
# points, smooth a 3-row window and scale by median/IQR.
rcsr --data panel.csv --target y --vars x1,x2 --split temporal --time date \
     --fill median --outlier-iqr 1.5 --smooth 3 --center median --scale iqr

# Only score rows that pass a condition:
rcsr --data panel.csv --target next_close --filter "volume > 100000"
```

The report says what ran, one commented line per active transformation:

```
# preparation: fill=median rows_filled=12 rows_dropped_missing=0
# preparation: outlier-iqr=1.5 rows_dropped_outlier=2
# preparation: center=median scale=iqr
```

`--json` carries the same numbers in a `preparation` object (methods, filled /
dropped / smoothed / filtered counts), and `--ndjson` emits a `preparation`
event when something is active.

### Boolean and categorical variables

Real tables mix continuous measurements with switches and labels. `--dtype`
says what each input is:

| kind | meaning |
|---|---|
| `continuous` (default) | a number, centred and scaled on the training rows |
| `boolean` | a `0`/`1` switch; validated, and **never** standardised |
| `categorical` | a text or low-cardinality numeric label; expanded to one-hot `col=value` boolean indicators |

```sh
# x1 AND NOT x2, from two 0/1 switches
rcsr --data and_not.csv --target y --dtype x1=boolean,x2=boolean \
     --ops and,not

# One indicator per symbol, all levels kept and sorted
rcsr --data panel.csv --target next_close --dtype symbol=categorical
```

A boolean column must contain only `0`/`1`; any other value is a usage-level
data error naming the row and the value. It is not centred or scaled, because
the evaluator's truthiness rule (`> 0` is true) and comparisons such as
`flag == 1` depend on the literal `0`/`1`. Categorical levels are the distinct
values in the **whole table**, sorted (numerically when they are numbers,
lexicographically otherwise), with missing cells as the final
`col=<missing>` level. `--max-levels N` (default 16) refuses a column with too
many levels rather than silently creating a huge input set. All levels are
kept: the search selects terms, so it does not need a dropped reference level.

`--json` reports the result as
`"variables_detail":[{"name":"flag","dtype":"boolean","source":null}, ...]`,
and the text log prints `# variables: flag (boolean), ...`. See
[`docs/variable-types.md`](docs/variable-types.md) for the exact rules, why
boolean columns bypass normalisation, and the `x1 AND NOT x2` truth table.

### Weighing rows

`--weight <col>` gives every row its own weight. The weight says **how much a
row counts**, not how large its target is: recent measurements, reliable sensors
or duplicated observations can be emphasised, and a row given weight zero has no
influence on the fit or the error.

```sh
# y = 2x + 1, with the second observation counted twice
rcsr --data d.csv --target y --vars x --weight w --metric squared
```

`absolute`, `squared`, `signed`, `log_squashed`, `correlation` and
`correlation_signed` become weighted means (weighted Pearson for the
correlations); `aic`, `aicc` and `bic`
use the weighted RSS and the number of positive-weight rows as `n`. A weight of
2 is exactly a duplicated row because the normalisation is `Σwᵢ·lossᵢ / Σwᵢ`,
not division by the row count. The fitter solves the weighted least-squares
problem (`sqrt(w)` on every residual), and the standard errors use the weighted
`JᵀWJ` with `dof = #positive weights − k`.

Metrics with no standard weighted form (`median`, `maximum`, `interquartile`,
`slope`, `rank_correlation`, `auc`, `log_loss`, `hinge`) **refuse** `--weight`
with a usage error that lists the supported ones: weights are never silently
ignored. The text report says

```
# weights: column w, sum 12.5 over 20 rows (2 zero-weight)
```

and `--json` echoes `"weight_column"`, `"weight_sum"` and `"weight_rows"`.
Stability resamples carry the weights with their rows; cross-validation and
`--eval-expr` are weighted too. Null baselines and residual tests remain
unweighted, and the report says so. See
[`docs/weights.md`](docs/weights.md) for the formulas, the doubling proof, why
the other metrics refuse weights, and a worked example.

**No leakage:** filling statistics, outlier bounds and centre/scale are all
computed from the **training rows only** and applied to both sides; smoothing
and `previous`/`interpolate` never cross the train/test boundary. See
[`docs/data-preparation.md`](docs/data-preparation.md) for each method, when
*not* to use it, and a worked example.

### Checking for leakage

Leakage is caught differentially, not by a significance test. The
[`leakage` module](docs/leakage.md) provides truncation replay (recompute at
cut points; every completed row must be bit-identical), a future-shift probe,
canary detection and synthetic leak/no-leak fixtures. They run in
`cargo test --release --test leakage` and can be pointed at your own panel with
`Panel::from_csv`; `assert_clean()` panics on a real leak instead of logging it.

### Forecasting: history and delayed variables

A forecast usually needs *history*: today's value may depend on yesterday's.
`--lags K` materialises `x_lag1 … x_lagK` for every input, computed inside each
series (group boundary, time order and time gaps respected), so the search can
use them like any other column:

```sh
# y_t = 1 + 3·x_{t-1}:  let the engine look one step back
rcsr --data lag.csv --target y --vars x --split temporal --time t --lags 1
```

`--history-frac f` keeps the first share of every series as *history*: those
rows provide the past for later rows but are not scored, so a delay near the
start of a series never reaches before it. `--gap-break f` cuts a series when
the time step exceeds `f` times the median step, which stops a lag from jumping
a hole in the data.

## Is it real, or is it overfitting?

A symbolic search can fit noise, so a small error proves nothing on its own.
Every solution therefore comes with the checks below, and `--null-runs` adds the
decisive one:

```sh
rcsr --data data.csv --target y --vars x1,x2 --null-runs 19
#  3  complexity=4  train_err=0.0004  test_err=0.0009  aicc=-402.1
# diagnostics: n=90 k=3 constants | overfit index (test/train)=2.25 | Durbin-Watson=2.01
# residual tests (best solution): Ljung-Box h=10 Q=9.4 p=0.494 | Breusch-Pagan LM=1.1 p=0.290 | skewness=0.02 excess kurtosis=-0.41
# null baselines (absolute): mean train=0.88 test=0.91 | linear train=0.42 test=0.46 | 5-NN train=0.31 test=0.35
# null control (19 shuffled-target runs): best=0.61 median=0.72 p=0.05 — the real fit beats the noise control
```

For autocorrelated series or overlapping labels, add `--null-block N` (N at
least the label horizon) to permute contiguous blocks instead of rows; the
report names the method, block length and block count, and the i.i.d. shuffle is
anti-conservative there. See [`docs/overfitting.md`](docs/overfitting.md).

| check | question it answers |
|---|---|
| `train_error` vs `test_error` | does it survive data it never saw? |
| `overfit_index` | how much worse does it get off-sample? |
| `fold_errors` | is the fit stable across blocks of the test set? |
| `k_constants` | how many free parameters did it really earn? |
| `aicc`, `bic` | is the extra complexity paid for by the fit? |
| `durbin_watson` | did anything get left in the residuals? |
| `constant_errors` | are the constants well determined, or do they trade off? |
| `baselines` | does it beat the mean, a straight line and a 5-NN learner? |
| `residual_tests` | Ljung–Box autocorrelation, Breusch–Pagan changing variance, residual skew/kurtosis |
| `null_control` | could the search have produced this from noise? |
| `bootstrap_error` | apparent, out-of-bag and .632+ error estimates (`--bootstrap-error`) |
| `cross_validation` | mean and spread of held-out error over `k` folds (`--cv`) |
| `stability` | how often each form and variable is selected on resampled rows (`--stability`) |

The null baselines are fit on the training rows and scored on the training rows
and the test folds in the search's own metric: a model that does not beat the
5-nearest-neighbour learner has found no structure the data did not already
contain. The k-NN baseline's reference set is always all training rows; for up
to eight inputs a k-d tree answers exact queries, and above that the exact
brute-force path reuses one buffer (still `O(query_rows × reference_rows × d)`,
so the 200k-under-a-minute result is a `d <= 8` claim). Only query rows can be
sampled, with `--baseline-samples N` (default `0` = every row, and only for
absolute/squared/log_squashed). That is an unbiased estimate of the row-average
loss, and `--json` reports train/test query rows and SEMs with the
finite-population correction. The residual tests (Ljung–Box, Breusch–Pagan,
skewness, excess kurtosis) are reported for the best solution; a small p-value
means the model left autocorrelation or changing variance behind.

The null control shuffles the target, runs the *identical* search, and repeats:
if shuffled targets fit as well as the real one, the "law" is manufactured. With
`r` runs the smallest attainable p-value is `1/(r+1)`, so use at least 19.

On a small sample, `--bootstrap-error n` refits each solution's constants on `n`
bootstrap resamples of the training rows and reports the **apparent**, the
**out-of-bag** and the **.632+** error (Efron & Tibshirani 1997) per solution;
the `.632+` estimate interpolates the optimistic and pessimistic numbers and
always lies between them. Above `n^0.7 >= 1000` rows (about 19 300) the draw is
bounded with a bag-of-little-bootstraps subsample of `b = n^0.7` rows, and the
resample size is reported in text and `--json`.

`--cv k-fold|rolling|purged` goes further: it rebuilds the split as `k` folds,
re-runs the whole search on each fold's training rows (normalised on those rows
alone) and reports the mean and spread of the held-out error, so a single lucky
split cannot flatter the answer. Use `rolling` for time series (it never trains
on the future) and `purged` to embargo `--gap` rows around each test block. It
costs about `k` extra searches.

`--stability B` asks the question that matters most: **does the same law come
back?** It bootstraps the training rows `B` times, re-runs the identical search
on each resample and ranks the structural forms and input variables by how often
they are selected. A variable at 100% is one the data insists on; one near 0% is
noise the search only sometimes needed. Cost: `B` extra searches. The same BLB
cap as `--bootstrap-error` applies above about 19 300 rows, and the resample size
is reported.

See [`docs/overfitting.md`](docs/overfitting.md) for the full checklist, the
formulas, what is still missing, and the literature.

## Reports for other people

`--report` turns a completed run into a Markdown document. One report per run;
a later flag replaces an earlier one. Use `--report-out` for a file, or let a
text run print it to stdout. In `--json`/`--ndjson` the report goes to stderr so
the machine-readable stream is never corrupted.

```sh
# everything a reader needs: command, data, law, constants + errors, diagnostics,
# baselines, the whole Pareto front and a trust verdict
rcsr --data data.csv --target y --report summary --report-out summary.md

# which input actually moves the prediction? analytic d f / d x per row
rcsr --data data.csv --target y --report sensitivity --report-out sensitivity.md

# repeat the search N times and report median/IQR error and wall-clock time
rcsr --data data.csv --target y --report benchmark --report-repeats 9 \
     --report-out benchmark.md
```

The summary's closing paragraph is derived from the statistics in the document
(it counts the baselines beaten, reads the overfit index, the permutation
p-value and the residual tests), so it states what the run does *not* establish
as well as what it does. The sensitivity report uses the **analytic** adjoint
derivative and ranks inputs by `mean(|∂f/∂x_i| · σ_x_i / σ_f)`, which is
scale-free and therefore comparable across columns with different units. The
benchmark honours the same `--generations`, `--pop` and `--time-budget`, derives
its seeds from `--seed`, and reports Hyndman–Fan type-7 quartiles.

See [`docs/reports.md`](docs/reports.md) for one worked example of each report,
with real output, the seed derivation and the stdout/stderr rules.

## Known-truth benchmarks

To check that the engine *recovers* laws rather than overfitting samples, a
separate harness runs it on a suite of **25 problems whose true formula is
known** — small Feynman-style physics laws and elementary dynamical systems —
with and without noise:

```sh
cd my-version
cargo run --release --bin rcsr-bench -- --suite known-truth --noise 0.0,0.01,0.1
cargo run --release --bin rcsr-bench -- --tiny          # fast CI subset, seconds
cargo run --release --bin rcsr-bench -- --tiny --json   # machine-readable
cargo run --release --bin rcsr-bench -- --list          # the problem table
```

A problem counts as **recovered** when the engine's Pareto front contains an
expression reaching test R² ≥ 0.99 on fresh *noiseless* data. The harness prints
a per-run table and a summary (recovery rate overall and per noise level, median
complexity) in text or with `--json`, and a run is reproducible from `--seed`.
See [`docs/benchmarks.md`](docs/benchmarks.md) for the criterion, the problem
list and the recorded results.

## Features

- expression trees over a broad primitive set (`+ - * / ^`, `exp ln sqrt abs
  sin cos tan sinh cosh tanh logistic gauss square cube`, `if`), plus the
  opt-in `floor ceil round mod min max sign step erf erfc atan2 asinh acosh
  atanh`, the conditional family
  `equal less less_or_equal greater greater_or_equal and or xor not require
  between constrain`, the meta operators
  `complexity variables constants contains contains_op` and the dataset
  aggregates `mean std median iqr sum amin amax count` selected with `--ops`
  (all with domain guards; the meta operators describe a subexpression's shape
  and are evaluated once per candidate, not per row — see
  [`docs/meta-operators.md`](docs/meta-operators.md); the aggregates are
  training-column statistics cached once per search — see
  [`docs/aggregation.md`](docs/aggregation.md));
- **twenty error metrics** (`--metric`): absolute, squared, correlation and
  signed correlation, median, maximum, signed, log-squashed, rank correlation
  and signed rank correlation, interquartile, slope, the two implicit
  objectives, AUC, log loss, hinge, and AIC/AICc/BIC as search objectives.
  The unsigned correlations accept an inverted model; the `_signed` variants
  require the right direction, and the binary metrics reject targets with more
  than two distinct values;
- **Levenberg–Marquardt constant fitting** using adjoint (backpropagation)
  derivatives — one backward sweep gives every constant's gradient, with
  analytic derivatives for every differentiable primitive (`erf`/`erfc`,
  `atan2`, `asinh`/`acosh`/`atanh`, and documented subgradients for
  `mod`/`min`/`max` and `and`/`or`/`xor`; `0` a.e. for the piecewise-constant
  `floor`/`ceil`/`round`/`sign`/`step`, the comparisons and `not`);
- an **island-model, multi-objective** evolutionary search over an
  error/complexity **Pareto front**, with mutation, crossover, migration and
  restarts, run in parallel across islands;
- **ε-dominance** archiving and **lexicographic parsimony pressure**, so the
  front stays small and expressions stay short;
- **Pareto-front parent feedback** (15%): one parent in six is drawn from the
  global archive instead of the island tournament, so a shape found on another
  island or generations earlier can be recombined rather than lost.
  Measured on the known-truth suite at three seeds: **72 → 74/75**, **72 → 75/75**
  and **72 → 74/75**, with `pythagoras` recovered in **7/9** cells instead of
  **0/9**, for about 5% more wall time. `--no-front-parents` disables it and
  restores the previous trajectory exactly;
- **configurable complexity** (`--op-complexity`, `--complexity-weights`,
  `--depth-weight`): per-operator costs, variable/constant weights and a
  nesting-depth penalty, with the node-count default preserved exactly;
- **structural de-duplication** and **convergence/time-budget stopping**, so the
  same expression is never evaluated twice and the search stops when there is
  nothing left to gain;
- **subsampled fitness** (`--predictor-samples N`): each generation scores and
  fits candidates on `N` training rows drawn without replacement, while the
  archive and every reported error are re-scored on the full training set.
  Carried individuals are refitted on the generation's subset before they are
  scored, so everyone is compared as `fit(S_g) -> score(S_g)` rather than with
  stale constants. A request at least the training size is an exact no-op, and
  a request above the measured break-even (3/5 of the rows) is widened to the
  full sample, because rebuilding the sample and refitting the carried
  population cost more than the rows saved;
- **screened constant fitting** (`--fit-samples N`): within each generation,
  all offspring are fitted and scored on a fresh deterministic `N`-row subset,
  the top 1/16 are promoted to a full-data refit and re-score, and the next
  population and archive are selected from those full-data scores. This is a
  throughput-for-statistics trade, not a free win: the sampled best is
  optimistically biased (the winner's curse), promotion reduces but does not
  remove it, and on deterministic, noiseless data the screen is a pure loss.
  `0` (the default) keeps the historical path; the flag is independent of
  `--predictor-samples`;
- **seeded starts** (`--seed-expr`, `--seed-file`): prior solutions are parsed,
  simplified, fitted and inserted into the archive and every island before
  generation 1; a seed is an elite, kept even above `--max-complexity`;
- **split modes for time series and panel data** (`random`, `temporal`, `group`,
  `temporal-group`, `auto`) with an embargo gap, per-block test errors and
  **delayed variables** (`--lags`, `--history-frac`, `--gap-break`), so a
  forecast is never scored on data it has already seen;
- **opt-in data preparation** — missing-value fills (`--fill`, missing marked
  by an empty cell, `NA`, `NaN` or `null`), IQR outlier removal, moving-average
  smoothing, median/IQR/range scaling and expression row filters (`--filter`) —
  with every statistic computed from the training rows only and a preparation
  report in text, JSON and NDJSON;
- **multi-output regression** (`--target y1,y2,...`): one shared tree shape,
  one fitted constant vector per output, combined error = mean per-output
  error; see [`docs/multi-output.md`](docs/multi-output.md);
- **overfitting diagnostics** on every solution (AICc, BIC, Durbin–Watson,
  overfit index, and per-constant **standard errors** from the Gauss–Newton
  Jacobian that flag constants the data cannot pin down), **null baselines**
  (mean, least-squares line, 5-NN) and **residual tests** (Ljung–Box,
  Breusch–Pagan, skewness/kurtosis) reported automatically, plus a
  `--null-runs` permutation control that runs the identical search on shuffled
  targets, a `--bootstrap-error` `.632+` bootstrap estimate that refits each
  solution's constants on resamples, `--cv` cross-validation (`k-fold`,
  `rolling`, `purged`) that reports the mean and spread of the held-out error,
  and `--stability` bootstrap selection that ranks forms and variables by how
  often they are rediscovered;
- **Markdown reports** (`--report summary|sensitivity|benchmark`): a
  self-contained model summary with a statistics-driven trust verdict, an
  analytic per-input sensitivity ranking (`mean(|∂f/∂x_i| · σ_x_i / σ_f)`, no
  finite differences), and a repeated-search benchmark with per-complexity
  median/IQR error and wall-clock time;
- bottom-up simplification with affine normalisation, and denormalisation back
  to original units.

See [`docs/optimisation.md`](docs/optimisation.md) for the measured before/after
numbers and the hardware plan.

## Is the search converging?

`--progress` reports four bounded numbers every generation — **confidence**
(the best error stopped moving), **stability** (the front's structural forms
survive), **maturity** (the oldest surviving form's age) and **percent
converged** (their weighted headline) — and the final `--json`/NDJSON output
carries the same `"convergence"` object. They are computed from the archive
alone, cost O(front size) and never touch a training row. See
[`docs/convergence.md`](docs/convergence.md) for the exact formulas, the
20-generation window, how to read them and the easy/hard evidence.

## Layout

```
SPEC.md                       behaviour specification
docs/how-symbolic-regression-works.md   plain-language explainer
docs/optimisation.md          theory, literature and measurements behind the search
docs/data-splitting.md        train/test splitting for time series and panel data
docs/data-preparation.md      missing values, outliers, smoothing, scaling and filters
docs/memory-model.md          bytes per cell, dictionary encoding and row selectors
docs/data-format.md           versioned .rcsb column store, import and the memory guard
docs/walkforward.md           walk-forward evaluation, costs, metrics and the long/flat rule
docs/variable-types.md        boolean and categorical inputs, one-hot encoding
docs/weights.md               per-row weights, weighted metrics and weighted least squares
docs/implicit.md              implicit relations f(x, y) = 0 and the level-set metric
docs/overfitting.md           avoiding and detecting overfitting, with references
docs/meta-operators.md        structural meta operators and their cache rule
docs/aggregation.md           dataset aggregates: semantics, cache and leakage rules
docs/seeds.md                 start the search from a known formula (--seed-expr/--seed-file)
docs/building-blocks.md       user-defined reusable sub-formulas (--function)
docs/multi-output.md          one shared shape, per-output constants (--target y1,y2)
docs/benchmarks.md            known-truth recovery benchmarks and recorded results
docs/convergence.md           confidence, stability, maturity and percent converged
docs/gui.md                   design for a GUI over the JSON/NDJSON contract
docs/gpu-backend.md           the optional CUDA evaluator and the DGX Spark plan
my-version/
  Cargo.toml
  build.rs      compiles the CUDA backend when --features cuda is used
  src/
    rng.rs        MT19937 RNG for reproducible runs
    expr.rs       expression trees, primitives, evaluator, mutation, simplification
    fit.rs        Levenberg–Marquardt constant fitting
    metrics.rs    error metrics
    pareto.rs     the error/complexity Pareto front
    data.rs       table loading, split modes, normalisation
    bench.rs      known-truth benchmark suite: problems, generators, scoring
    search.rs     island-model GP loop
    gpu.rs        bytecode compiler + FFI for the optional GPU evaluator
    cuda/eval.cu  the CUDA kernel
    main.rs       the `rcsr` CLI
    bin/
      gpu_bench.rs  GPU vs CPU evaluation benchmark (--features cuda)
      rcsr_bench.rs the `rcsr-bench` known-truth benchmark harness
  tests/
    recover.rs    end-to-end recovery tests + unit tests
    split.rs      split-mode tests (temporal, group, panel, folds)
    prep.rs       data-preparation tests (fill, outliers, smoothing, scaling, filters)
    weights.rs    row weights, weighted metrics and weighted least squares
    bench.rs      known-truth benchmark harness tests
  examples/
    line.csv, sine.csv
```

## License

MIT.
