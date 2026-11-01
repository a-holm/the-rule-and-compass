# How symbolic regression works — and how this engine is built

*Design notes for `rule-and-compass`, a symbolic-regression engine written in Rust.*

## 1. The problem

Symbolic regression searches for a **closed-form mathematical expression** that
fits a dataset, rather than fitting fixed parameters of a model chosen in
advance. Given inputs `x1..xn` and a target `y`, we want a short expression `f`
with `f(x) ≈ y`.

Two things are wanted at once: **accuracy** and **simplicity**. A huge
expression can always fit noise, so we also want the expression *small*. Because
those pull in opposite directions, the search is naturally a **multi-objective**
problem, and because the space of expressions is combinatorial, it is driven by
an **evolutionary algorithm**.

## 2. Data

- Tabular numeric data (CSV, TSV or whitespace-delimited) with one header row.
- Each input variable is **normalised** (zero mean, unit standard deviation)
  before the search, so the search does not have to rediscover scales. The
  affine map is kept so results can be printed in the original units.
- Rows are split into a **training** set (used to score candidates) and a
  held-out **test** set (used to report how well a result generalises).
- Rows and variables may be **weighted**.

## 3. Representation

A candidate is an **expression tree**. Leaves are constants and input variables;
internal nodes are operators. Trees are cheap to build, mutate and recombine.

Evaluation is **column-wise**: the whole training column is evaluated in a
single pass, so one candidate costs one traversal of the operator tree over all
rows. This is what makes millions of candidate evaluations affordable.

Primitives (grouped as the UI groups them):

- **terminals** — real constant, integer constant, input variable
- **arithmetic** — `+ - * / ^ mod`, negation, factorial, `min`, `max`
- **rounding** — floor, ceiling, round
- **elementary** — `exp`, `ln`, `sqrt`, `abs`
- **trigonometry** — `sin cos tan`, and the inverse/hyperbolic family, `atan2`
- **squashing** — `logistic`, `step`, `sign`, `gauss`, `erf`, `erfc`
- **logical** — `if(c, a, b)`, `= < <= > >= and or xor not`
- **history** (optional) — delayed variables, moving averages

In this engine the conditional family is **opt-in** (`--ops`): the comparisons
`equal less less_or_equal greater greater_or_equal`, the logical operators
`and or xor not`, and the positional constraints `require constrain between`.
A comparison evaluates to `0`/`1`; a value counts as *true* when it is strictly
positive. A **constraint** is different from a wrong prediction: `require(cond)`
and `constrain(x)` return their (positive) argument when satisfied and a
non-finite value when violated, and `between(x, lo, hi)` returns `x` only on the
closed interval. Because a non-finite prediction makes the error metric infinite
(§5), a violated constraint marks the candidate **infeasible** and removes it
from the search rather than scoring it as a merely large error.

The **meta operators** — `complexity`, `variables`, `constants`, `contains`,
`contains_op` — are also opt-in. They inspect the shape of their argument
instead of evaluating it, so they return the same value on every row and are
cached when the node is built. They describe the model, not the data; see
[`meta-operators.md`](meta-operators.md).

## 4. Complexity

The second objective is a size measure of the expression. The simplest choice is
the **number of nodes**; a **per-operator cost table** can refine it so that,
say, `sin` costs more than `+`. A **path-length constraint** caps how large a
candidate may grow.

## 5. Error metrics

The score of a candidate on the training data. Lower is better:

- absolute error (default), squared error
- maximum (worst-case) error
- median error, inter-quartile error, log-squashed error
- correlation coefficient, `R²`, rank correlation (these are maximised)
- AIC-adjusted variants
- classification-oriented: AUC, log loss, hinge loss, accuracy
- implicit-derivative error, for relations written `F(x, y) = 0`

Any candidate producing a non-finite prediction is treated as a hard failure.

## 6. The search

**Island-model, multi-objective evolutionary search.**

- A population of expression individuals is split into several **islands**.
  Islands evolve independently (which maintains diversity) and can exchange
  members.
- Every candidate is scored on **two axes — error and complexity** — and the
  engine keeps the **Pareto front** of non-dominated solutions rather than a
  single "best". The answer is therefore a *set*: at each complexity level, the
  most accurate equation found so far.
- The loop advances in **generations**. Each generation: evaluate the
  population, update the front, then breed the next generation.
- **Selection** is by tournament within an island. With probability 15 %, a
  parent is instead drawn uniformly from the global Pareto front: the front is
  non-dominated, so each member is the best shape found at some complexity, and
  this lets a building block found on another island or generations earlier
  re-enter the population. `--no-front-parents` disables the feedback.
- **Variation** has two parts:
  - *mutation* — the edit kind is drawn first (25 % point change, 35 %
    constant/input perturb, 40 % subtree grow/shrink) and the site is chosen
    among nodes that the edit can actually change, so a mutation never returns
    the parent unchanged;
  - *crossover* — swap subtrees between two parents.
- **Constant fitting** tunes the numeric constants of a candidate by local
  search. Without it, `y = a·sin(b·x) + c` would almost never be found, because
  random constants rarely land near the right values.
- **Linear scaling** (`--linear-scaling`, off by default) wraps every candidate
  shape `f` in a free affine map `a + b·f(x)`. The two coefficients are ordinary
  constants, fitted with the others (closed-form least-squares start on the
  CPU), so a shape that is right up to scale and offset scores as right at
  once. Variation edits the bare shape; the front holds the wrapped formula
  with its full complexity. It needs a least-squares metric.
- **Migration** periodically copies an island's best member into a neighbouring
  island; **stagnant islands are restarted** to avoid getting stuck.
- Termination is by generation count or a time/effort budget.

## 7. Simplification and presentation

Scoring and de-duplication use the sound normal form (`expr::simplify`): constant
folding and only those identities that preserve every IEEE value, including
`NaN`, `±inf` and signed zero. Before output, the display relation
(`expr::simplify_display`) additionally applies affine normalisation and
`x+0 -> x` for a readable equation, and the tree is **denormalised** back to the
original variable units.

## 8. Concurrency

The search is embarrassingly parallel: islands, and candidate evaluations within
an island, map onto a worker-thread pool. Data loading and per-column processing
are parallelised as well. A thread-safe tracker keeps the shared Pareto front
consistent between workers.

## 9. Output

A set of equations on the error/complexity front, each with its complexity,
training error, test error and expression text. The front can be written to a
file for downstream use.

## 10. Roadmap

- more metrics and primitives (history and derivative operators)
- a configurable per-operator complexity table
- parallel islands via real threads
- project save/load
