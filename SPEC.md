# SPEC — `my-version`: a symbolic-regression engine (Rust)

Goal: a working Linux implementation of a symbolic-regression engine, written in
Rust, with tests. This spec describes **behaviour and interfaces**.

---

## 1. Problem

Given a table of numeric columns `x_0..x_{n-1}` (inputs) and `y` (target), find
short closed-form expressions `f` minimising a chosen error between `f(x)` and
`y`, while also minimising the **complexity** of `f`. The output is the set of
**non-dominated** solutions (the error/complexity Pareto front).

## 2. CLI

```
rcsr --data <file.csv>
     --target <col>
     --vars <c1,c2,...>
     --metric absolute|squared|correlation
     --generations <N>          # default 200
     --islands <I>              # default 1
     --pop <P>                  # per-island population, default 200
     --max-complexity <K>       # path-length constraint
     --seed <S>                 # default 0 (MT19937)
     --test-frac <f>            # default 0.25
     --ops <list>               # default: built-in set (see section 4)
     --out <file>               # optional; write the front as TSV
```
`--help` prints usage. Exit non-zero on unreadable or empty data.

## 3. Data handling

- Parse CSV / TSV / whitespace-delimited numeric data (no external crates).
- Drop or reject rows containing non-numeric or missing values, and report how
  many were dropped.
- **Normalise** each variable to zero mean / unit standard deviation for the
  search; keep the affine map and **denormalise** expressions for output.
- Split rows into **training** (`1 - test_frac`) and **test**. Fitness uses the
  training rows only; the reported error also uses the test rows.
- Shuffle deterministically with the seeded RNG.

## 4. Representation

Expression **tree** over these primitives:

- terminals: `Const` (real value), `Input(i)`
- unary: `Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh, Logistic, Step, Gauss, Square, Cube, Floor, Ceil, Round, Sign, Erf, Erfc, Asinh, Acosh, Atanh, Not, Require, Constrain, Complexity, Variables, Constants, Contains, ContainsOp`
- binary: `Add, Sub, Mul, Div, Pow, Mod, Min, Max, Atan2, Equal, Less, LessOrEqual, Greater, GreaterOrEqual, And, Or, Xor`
- ternary: `If(cond, a, b)` where `cond` is `a > 0`, `Between(x, lo, hi)`
- aggregates: `Agg(kind)` with `kind` in `Mean, Std, Median, Iqr, Sum, Min,
  Max, Count`; the column is a single `Input` child (`Count` is zero-arity)

`Floor`…`Atanh`, `Mod`/`Min`/`Max`/`Atan2`, the whole conditional
family (`Equal`…`GreaterOrEqual`, `And`/`Or`/`Xor`/`Not`,
`Require`/`Between`/`Constrain`), the meta operators
(`Complexity`…`ContainsOp`) and the dataset aggregates (`Agg`) are **opt-in**
primitives: they are not part of the default operator set but are accepted by
`--ops` and listed by `rcsr schema` under `"operators"`. `erf`/`erfc` are computed to machine
precision with no external crates, and the Levenberg–Marquardt adjoint pass uses
the analytic derivative of each operator where one exists (`0` a.e. for
`floor`/`ceil`/`round`/`sign`/`step`, the comparisons and `not`; a documented
symmetric subgradient for `min`/`max` and for `and`/`or`/`xor`;
`d mod(a,b) = (1, -floor(a/b))` a.e.; `1` through a satisfied
`require`/`constrain`/`between`).

**Comparisons** (`equal`, `less`, `less_or_equal`, `greater`,
`greater_or_equal`) return `1.0` when the relation holds and `0.0` otherwise;
`NaN` in either operand gives `NaN`.

**Logical** operators treat a value as true when it is **strictly positive**
(`0` is false): `and`/`or`/`xor` return `1.0`/`0.0`, `not` negates, and a `NaN`
operand propagates.

**Constraints** make an infeasible candidate a *hard failure* rather than a
large finite error: a violated constraint evaluates to a non-finite value, which
the metric turns into `+inf` (section 6), so the row — and the candidate — is
rejected. While satisfied they pass their argument through unchanged:

| operator | satisfied | violated |
|---|---|---|
| `require(cond)` | `cond` when `cond > 0` | `NaN` |
| `constrain(x)` | `x` when `x > 0` (the unary spelling of `require`) | `NaN` |
| `between(x, lo, hi)` | `x` when `lo <= x <= hi` | `NaN` |

**Meta operators** describe the *shape* of their argument instead of evaluating
it, so their value is the same on every row:

| op | returns |
|---|---|
| `complexity(e)` | the node count of `e` |
| `variables(e)` | the number of **distinct** input variables appearing in `e` |
| `constants(e)` | the number of constant leaves in `e` |
| `contains(e)` | `1` if `e` contains at least one input variable, else `0` |
| `contains_op(e)` | `1` if `e` contains at least one operator node (i.e. is not a bare leaf), else `0` |

They are unary, computed **once** when the node is built (or when `simplify`
changes the argument's shape) and cached in `Node::val`; `eval` returns that
cache without walking the argument. A meta node has no fitted constant slot and
a zero gradient in the adjoint sweep, exactly like `floor`/`step`, so
`count_constants` never counts it. `complexity` and `variables` count nested
meta nodes and their arguments like ordinary structure. See
[`docs/meta-operators.md`](docs/meta-operators.md) for the cache rule and the
caveat that a meta operator describes the model, not the data.

**Dataset aggregates** reduce one input column over the **training rows** and
return the same value on every row:

| op | value |
|---|---|
| `mean(x)` | arithmetic mean |
| `std(x)` | population standard deviation (divide by `n`) |
| `median(x)` | 50th percentile (linear interpolation) |
| `iqr(x)` | `q75 - q25`, the same interpolated quantiles |
| `sum(x)` | sum |
| `amin(x)` / `min(x)` | smallest finite value (`amin` is the `--ops` name) |
| `amax(x)` / `max(x)` | largest finite value (`amax` is the `--ops` name) |
| `count()` | number of training rows (zero-arity) |

The column is the aggregate's single `Input` child; `mean(x1 + 1)` is a parse
error. Non-finite cells are ignored; a column with no finite value yields `NaN`,
which the domain guard turns into an infeasible candidate (`+inf` score), never
an invented number.

The values are computed **once**, before the first generation, into a
per-column `AggTable` and cached in `Node::val`. `eval` returns the cache
without walking the child, so per-row cost does not grow. An aggregate has no
constant slot, no gradient in the adjoint sweep, and is ignored by
`count_constants` and the `constants(e)` meta operator: a dataset fact is not a
fitted parameter. `simplify` never folds an aggregate, and `mutate`/`crossover`
refresh the cache after a structural edit.

The table is built from the search's **training** sample only. Test rows never
contribute, and it is not rebuilt for `--predictor-samples` subsets; the test
split is only ever scored with the training statistics. Because the search
normalises inputs by default, the statistic is that of the normalised column;
use `--center none --scale none` to read aggregates in original units. See
[`docs/aggregation.md`](docs/aggregation.md).

Evaluation is a recursive `f64` evaluator with **domain guards**, so
out-of-domain operations yield a non-finite value (treated as a hard failure):
`Ln(a<=0) -> NaN`, `Sqrt(a<0) -> NaN`, `Div(|b|<eps) -> NaN`,
`Pow(a<0, non-integer b) -> NaN`, `Tan(x)` guarded near poles,
`Acosh(a<1) -> NaN`, `Atanh(|a|>=1) -> NaN`, `Mod(_,|b|<eps) -> NaN`, and a
violated constraint as above.

Evaluation is done **column-wise over all training rows**.

### 4.1 Textual expression language

`expr::to_string` prints a tree; `expr::parse` (module `parse`, re-exported as
`expr::parse`) reads one back:

```
expr    := or_expr
or_expr := xor_expr ('or' xor_expr)*
xor_expr:= and_expr ('xor' and_expr)*
and_expr:= cmp_expr ('and' cmp_expr)*
cmp_expr:= sum (('<' | '<=' | '>' | '>=' | '==') sum)?   // non-associative
sum     := term (('+' | '-') term)*
term    := unary (('*' | '/') unary)*
unary   := ('-' | '+' | 'not') unary | power
power   := primary ('^' unary)?          // right associative
primary := number
         | meta '(' expr ')'
         | agg '(' expr? ')'
         | name '(' expr (',' expr)* ')'
         | name
         | '(' expr ')'
meta    := 'complexity' | 'variables' | 'constants' | 'contains' | 'contains_op'
agg     := 'mean' | 'std' | 'median' | 'iqr' | 'sum' | 'amin' | 'amax' | 'count'
```

Precedence, loosest to tightest: `or`, `xor`, `and`, comparisons
`< <= > >= ==`, `+`/`-`, `*`/`/`, unary `-`/`+`/`not`, `^`. So `-x^2` is
`-(x^2)`, `2^3^2` is `2^(3^2)` and `x + 1 > y * 2` is `(x + 1) > (y * 2)`.
`or`/`xor`/`and` are left-associative; comparisons are **non-associative**
(`a < b < c` is a syntax error — write `(a < b) and (b < c)`); `not` is a prefix
operator at the unary level, so `not x > 0` is `(not x) > 0` and
`(not (x > 0))` is the negated comparison. `to_string` prints every binary
conditional operator parenthesised (`(x > 0)`, `(a and b)`, `(not a)`) and the
constraints as calls (`between(x, -1, 1)`), so the printed form parses back to
the same tree. Function names are the primitives above, via
`Op::from_name`, sharing the alias table with `--ops` (`log` = `ln`,
`sig` = `logistic`, `lt`/`le`/`gt`/`ge`/`eq` = the comparisons, `&&`/`||`/`!` =
`and`/`or`/`not`). Variable names resolve against the table's column names,
then the `x0`, `x1`, … fallback; `pi` and `e` are constants when they are not
declared variables. Unknown names, unmatched parentheses and wrong argument
counts produce a `ParseError` carrying the **1-based column**. The meta
operators are ordinary unary calls in this grammar
(`complexity(x1 * x2)`), and the parsed node caches the structural value.
The aggregates are calls too (`mean(x1)`, `count()`); their argument must be a
bare variable, and a parsed node's value is `NaN` until it is bound to the
training table (`expr::bind_aggregates`). `min(x1)`/`max(x1)` with one
argument are the column reductions; `min(a, b)`/`max(a, b)` stay binary.
`--eval-expr` exposes the parser on the command line.

### 4.2 User-defined building blocks (`--function`)

Up to four `--function` definitions make a sub-formula available as an operator
of its own arity. The syntax is `name(a,b)=<expr>` or `name=<expr>`; in the
latter form the parameters are the free variables in order of first appearance.
A body is parsed with the same grammar and may call only blocks defined before
it, so recursion and forward references are rejected. Names must be unique and
must not shadow a primitive or a canonical slot name (`custom0`..`custom3`).
Each definition occupies the slot matching its position. A block has 0 to 4
parameters.

A call `name(e1, e2)` is an expression node with two children. Evaluation
substitutes the argument values for the body's parameters and evaluates the
body. The printed form uses the user's name (`name(e1, e2)`) and parses back
when the same definitions are supplied. Constants inside a body are **shared**
by every call in a candidate: the constant fitter maps them to one parameter
per `(block slot, constant ordinal)`, writes the fitted value back into every
use, and `count_constants` counts them once per block, not once per call.

A block costs its configured operator cost (default `1`) plus the cost of its
arguments; the body's internal nodes are not charged again. `--op-complexity`
accepts either the user's name or the canonical slot. Supplying a block adds its
slot to the run's operator set. Without `--function` no custom operator exists
and behaviour is unchanged. `--function` is incompatible with `--save` and
`--resume`, because the checkpoint format does not store definitions.

See [`docs/building-blocks.md`](docs/building-blocks.md) for examples and the
shared-constant fitting rule.

## 5. Complexity

Every candidate has a **configured complexity**

```
complexity(f) = sum(op_cost)              # over operator nodes only
              + var_weight  * n_vars
              + const_weight * n_consts
              + depth_weight * max_depth
```

- `sum(op_cost)` prices every node that is neither `Const` nor `Input`.
  Terminals are priced by the two terminal weights, never by `op_cost`.
- `n_vars` and `n_consts` count `Input` and `Const` nodes.
- `max_depth` is the number of edges on the longest root-to-node path. A
  single terminal has depth `0`; `sin(x)` has depth `1`; `sin(cos(x))` has
  depth `2`.
- **Defaults:** every operator costs `1`, `var_weight = const_weight = 1` and
  `depth_weight = 0`. The value is then exactly the number of nodes, bit for
  bit, which is the historical behaviour.
- `--op-complexity "pow=2,sin=1.5,if=3"` sets per-operator costs. Names use the
  same table as `--ops`; any operator not listed keeps the cost `1`. A
  user-defined block is priced by its name or `customN`; its body is not
  re-priced, so the cost is the price of the block as one primitive.
- `--complexity-weights "var=1,const=1"` sets terminal weights. The keys are
  `var` (aliases `variable`, `input`) and `const` (alias `constant`); `depth`
  is also accepted as a synonym for `--depth-weight`.
- `--depth-weight 0.5` adds `0.5` per nesting level.
- Every weight must be positive and finite; the depth weight may also be zero.
  An unknown operator or key, a missing `=`, an empty entry or a non-positive
  weight is a **usage error (exit 2)** whose message names the offending token.
- The configured complexity is the value on which the Pareto front is ranked
  and pruned, the value compared against `--max-complexity`, and the value
  reported for every solution (`complexity=` in text, `"complexity"` in JSON).
- Complexity is computed for every candidate, so the hot path is one
  allocation-free recursive walk over the tree.

See [`docs/complexity.md`](docs/complexity.md) for worked examples.

## 6. Error metrics

For training pairs `(x_k, y_k)`, prediction `p_k = f(x_k)`:

- **absolute** (default): `mean_k |p_k - y_k|`
- **squared**: `mean_k (p_k - y_k)^2`
- **correlation**: `1 - corr(p, y)` (minimise is equivalent to maximising the
  correlation coefficient)

Any non-finite `p_k` makes the metric `+inf`.

### Implicit metrics (issue #14)

`--implicit` appends the target as one more input (normalised with its training
mean/std) and sets the fitting target to `0`, so the model is
`f(x1..xn, y) = 0`. Two metrics score it:

- **implicit** (alias `implicit_derivative`): `mean_k |f(p_k)| / ||∇f(p_k)||`,
  the first-order distance to the zero level set;
- **implicit_residual**: `mean_k |f(p_k)|`, the plain residual.

Both use the adjoint sweep for `∇f`. A row with `||∇f|| = 0`, a non-finite
`f`/gradient component, or an overflowing distance makes the candidate `+inf`.
The implicit-derivative metric fits constants on the signed geometric residual
`f / ||∇f||`; the plain residual keeps the ordinary least-squares fit against
the zero target. See [`docs/implicit.md`](docs/implicit.md).

## 7. Search algorithm

**Island-model, multi-objective evolutionary search.**

State:
- `I` islands, each with `P` individuals (`{ tree, train_error, complexity }`);
- a global archive: the **Pareto front** on `(train_error, complexity)`. A new
  individual is kept if no archive member is `<=` on both axes (with one
  strictly `<`); dominated members are removed. On equal complexity, the lower
  error is kept.

Per generation, per island:
1. Evaluate any unevaluated individuals on the training data.
2. Merge into the global Pareto archive.
3. Produce `P` offspring:
   - **selection**: tournament of size 4 drawn from the island; the winner is
     the dominating individual, or a random one if they are incomparable;
   - **variation**: point-mutate an operator, replace/insert/delete a random
     subtree, perturb a constant, or **crossover** (swap random subtrees of two
     parents);
   - **constant fitting**: for each `Const` leaf, try a small set of
     multiplicative/additive steps and keep any improvement against the
     training error, repeated for a few passes;
   - respect `max_complexity`; retry a bounded number of times.
4. Replace the island population with the offspring, keeping the island's best
   individual (elitism).

Every `migration_interval` generations, copy the best member of each island into
the next island (ring topology). If an island stagnates for `restart_interval`
generations, restart it from random individuals.

Optional quality-diversity coverage (`--qd`) observes every evaluated
individual in a rectangular grid over two descriptor axes (variable usage and
prediction skewness). Each occupied cell keeps a bounded local Pareto front.
The layer is passive: it does not alter selection, migration, restarts or the
global archive. `--json` reports the grid size, occupied cells and coverage
fraction.

Termination: `--generations` reached. Report progress as generations.

## 8. Simplification

Before output, apply bottom-up:
- constant folding (`add(const,const)`, etc.);
- identities: `x+0 -> x`, `x*1 -> x`, `x*0 -> 0`, `x/1 -> x`, `x^1 -> x`,
  `x^0 -> 1`, `--x -> x`;
- denormalise to the original variable units.

Report the final form's error on the **test** set.

## 9. Output

For each archive member (ascending complexity), print: rank, complexity,
training error, test error, and the expression string. With `--out`, write
machine-readable rows.

## 10. Edge cases

- Fewer than 2 usable rows -> error out.
- Constant target column -> any constant expression is a perfect solution
  (complexity 1).
- Duplicate variable names / unknown column -> error.
- All-NaN predictions -> `+inf` error; the individual never enters the archive.
- Very large/small constants are clamped to a sane range to avoid overflow.

## 11. Tests (must all pass)

1. `expr`: evaluation of each primitive; domain guards return non-finite;
   complexity counts nodes.
2. `metrics`: known vectors give the expected absolute/squared/correlation
   values.
3. `pareto`: domination insert/remove behaves correctly.
4. **recovery** (integration), search recovers to test error `< 1e-6`:
   - `y = x`
   - `y = x^2`
   - `y = x1 * x2`
   with fixed seeds for determinism.
5. `simplify`: `x*1 + 0` folds to `x`; constant subexpressions fold.
6. conditional operators: evaluation of each comparison/logical/constraint
   (including the true/false boundary at `0`), a parser round-trip for
   `x > 0`, `and`, `not` and `between(a, lo, hi)`, a recovery test on the
   piecewise target `y = x*(x>0)`, and a test that a violated constraint yields
   a non-finite evaluation (a hard failure), not a small finite error.

## 12. Out of scope for v1

GUI, project files, distributed/cloud compute, time-series operators
(`delay`, moving averages), implicit-derivative metrics, licensing.
