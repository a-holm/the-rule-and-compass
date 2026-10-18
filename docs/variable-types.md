# Boolean and categorical variables

By default every input column is **continuous**: it is centred and scaled on the
training rows before the search. Real tables also contain switches (`0`/`1`) and
labels (a ticker, a region, a regime). `--dtype` tells the engine how to treat
them:

```sh
rcsr --data d.csv --target y --vars flag,symbol \
     --dtype flag=boolean,symbol=categorical
```

One entry per non-default column; anything not named stays `continuous`. Kinds
are `continuous` (the default), `boolean` and `categorical`
(`numeric`/`float`, `bool`/`binary` and `category`/`factor` are accepted
aliases). Unknown kinds, unknown columns and naming the target are usage errors
(exit 2).

## Boolean variables

A `boolean` column must contain only the numbers `0` and `1`. Any other value is
rejected before the search, naming the column, the row and the value:

```text
error: boolean column 'flag' must contain only 0 and 1;
       row 2 (1-based, header excluded) has '2'
```

`0.0` and `1.0` are the same numbers and are accepted. A missing cell (empty,
`NA`, `NaN` or `null`) is not a value error: it is left to `--fill`, exactly
like any other column. A fill that would average truth values into a fraction
(`mean`, `median`, `interpolate`) is refused once it produces a non-0/1 value;
use `--fill zero`, `--fill drop` or `--fill previous` instead.

### Why boolean columns are not standardised

The evaluator uses **truthiness**: a value is true when it is strictly greater
than zero. `and`, `or`, `xor`, `not`, `if` and the feasibility constraints all
build on that rule, and comparisons such as `flag == 1` read the literal value.

Centring and scaling a `0`/`1` column replaces the two values by
`(0 - μ)/σ` and `(1 - μ)/σ`. Those numbers are no longer booleans: `== 1` stops
matching, arithmetic over truth values changes meaning, and the exact contract
"false is 0, true is 1" is lost. (The signs may happen to survive for a balanced
column, but that is a coincidence, not a guarantee.)

So a boolean input bypasses normalisation completely:

* `means[j] = 0` and `stds[j] = 1` in the denormalisation bookkeeping;
* every value reaching the search and the evaluator is exactly the `0`/`1` from
  the file;
* a printed formula is still in original units, because the identity transform
  is what denormalisation applies;
* `--center` and `--scale` apply to continuous inputs only, and `--smooth`
  skips boolean inputs (a moving average would create fractions).

Boolean inputs are ordinary inputs for the sampler: they can appear in any
expression, feed `if(flag, a, b)`, or be compared and combined with the logical
operators. Enable those primitives with `--ops`, for example
`--ops add,sub,mul,and,not,greater`.

## Categorical variables

`--dtype symbol=categorical` expands one column into **one-hot indicator
columns**, one per distinct value:

| original `symbol` | `symbol=AAPL` | `symbol=MSFT` | `symbol=GOOG` |
|---|---:|---:|---:|
| AAPL | 1 | 0 | 0 |
| MSFT | 0 | 1 | 0 |
| GOOG | 0 | 0 | 1 |

Each indicator is a `boolean` input with the rules above, so it is never
standardised either.

### Level order

The levels are the **distinct values in the whole table**, not just the training
rows, and they are sorted so the encoding does not depend on row order:

* if every level parses as a number, they sort by value (`1, 2, 10`, not
  `1, 10, 2`);
* otherwise the level text sorts lexicographically;
* a missing or empty cell becomes its own final level, `column=<missing>`.

The whole-table rule matters for reproducibility: the same file always yields
the same column names and the same 0/1 layout, whichever rows happen to be
training or test. It is a naming/encoding choice only; the split still decides
which rows are scored.

A literal value equal to `<missing>` is reserved for gaps and is a usage error.

### How many levels?

`--max-levels N` (default `16`) bounds the number of distinct values. More than
that is a usage error (exit 2) naming the column and the count:

```text
error: categorical column 'symbol' has 20 distinct values, more than
       --max-levels 16; raise the limit (e.g. --max-levels 20)
```

Raise the limit deliberately: a free-text column with hundreds of levels turns
into hundreds of inputs and makes the search mostly a search over noise.

### All levels are kept

The encoding keeps **every** level; no reference level is dropped. One-hot
encodings for linear models often drop one level to avoid perfect collinearity,
but this engine does not invert a design matrix: it selects terms. Keeping all
levels preserves the meaning of each indicator (`symbol=AAPL` is exactly "the
symbol is AAPL") and cannot make an otherwise valid expression unsolvable.
Dropping a level would silently make one category the "baseline" and change what
the remaining indicators mean.

### Interaction with `--vars` and split columns

A declared datatype is also a request to use the column as an input:

* when `--vars` is omitted, a text categorical column named by `--dtype` is
  added to the default inputs (numeric columns are already there);
* when `--vars` is given explicitly, it wins: naming a column there is required,
  and `--dtype` naming a column outside it is a usage error rather than a silent
  addition;
* a column used for `--time`, `--group-by` or `--weight` cannot also be an
  input, so it cannot be named by `--dtype`;
* `--lags K` runs after expansion, so a delayed boolean or indicator column is
  boolean too.

A non-numeric column left as `continuous` is rejected by the data loader when
the inputs are resolved; declare it `categorical` to encode it instead.

## Reports

The text log names every input and its datatype:

```text
# variables: x1 (boolean), x2 (boolean), symbol=AAPL (boolean), ...
```

`--json` and `--ndjson` carry the same information:

```json
"variables_detail": [
  {"name": "flag", "dtype": "boolean", "source": null},
  {"name": "symbol=AAPL", "dtype": "boolean", "source": "symbol"}
]
```

`source` is the original column for a one-hot indicator and `null` otherwise.
The `--report summary` Markdown report adds a `variable types` row to its data
table.

## Worked example: `x1 AND NOT x2`

Build a table with every combination of two switches and the target

```text
x1,x2,y
0,0,0
0,1,0
1,0,1
1,1,0
...
```

(the four rows repeated, so the split has rows of both classes). Then:

```sh
rcsr --data and_not.csv --target y --dtype x1=boolean,x2=boolean \
     --ops and,not --generations 80 --pop 300 --islands 2 --seed 1
```

The truth table the recovered law must reproduce is

| `x1` | `x2` | `x1 AND NOT x2` |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

With the command above the engine reports (among the equivalent forms)

```text
  3  complexity=4   train_err=0.000000   test_err=0.000000   y = ((not x2) and x1)
```

and `tests/dtypes.rs::recovers_x1_and_not_x2` checks two things independently of
which equivalent form wins: the best front member has zero training and test
error, and evaluating it on the four rows above gives exactly `0, 0, 1, 0`.
Because the inputs are raw `0`/`1`, `and`/`not` see clean truth values rather
than scaled approximations.

## `--eval-expr` and one-hot names

Printed formulas use the input names verbatim, so a one-hot term appears as
`symbol=AAPL`. The expression parser accepts identifiers and the positional
`x0, x1, …` fallback; if you want to paste a formula containing `=`-names back
into `--eval-expr`, use the `xN` form for those inputs (the order is the
`variables` array in the JSON output).
