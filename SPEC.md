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
     --ops <list>               # default: all primitives in section 4
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
- unary: `Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh, Logistic, Step, Gauss, Square, Cube`
- binary: `Add, Sub, Mul, Div, Pow`
- ternary: `If(cond, a, b)` where `cond` is `a > 0`

Evaluation is a recursive `f64` evaluator with **domain guards**, so
out-of-domain operations yield a non-finite value (treated as a hard failure):
`Ln(a<=0) -> NaN`, `Sqrt(a<0) -> NaN`, `Div(|b|<eps) -> NaN`,
`Pow(a<0, non-integer b) -> NaN`, `Tan(x)` guarded near poles.

Evaluation is done **column-wise over all training rows**.

## 5. Complexity

- Per-operator cost table; default **1.0 per node** (configurable).
- `complexity(expr) = sum of the cost of each node`.
- Reject individuals whose complexity exceeds `max_complexity`.

## 6. Error metrics

For training pairs `(x_k, y_k)`, prediction `p_k = f(x_k)`:

- **absolute** (default): `mean_k |p_k - y_k|`
- **squared**: `mean_k (p_k - y_k)^2`
- **correlation**: `1 - corr(p, y)` (minimise is equivalent to maximising the
  correlation coefficient)

Any non-finite `p_k` makes the metric `+inf`.

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

## 12. Out of scope for v1

GUI, project files, distributed/cloud compute, time-series operators
(`delay`, moving averages), implicit-derivative metrics, licensing.
