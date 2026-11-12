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
- **Selection** is by tournament within an island. A finite error always beats a non-finite one; parsimony only breaks ties among finite errors (and among non-finite ones). With probability 15 %, a
  parent is instead drawn uniformly from the global Pareto front: the front is
  non-dominated, so each member is the best shape found at some complexity, and
  this lets a building block found on another island or generations earlier
  re-enter the population. `--no-front-parents` disables the feedback.
- **Constant models.** A tree with no input outside an aggregate or meta operator (`exp(1.3)`, `mean(x1) + 0.2`) predicts one value for every row, so after its constant fit it equals the training-mean model. With `--reject-constant-offspring` such a child is dropped after simplification and before de-duplication and the fit, and counted in `variation.constant_rejected`. The first constant tree of an island is kept, so the front keeps its complexity-1 baseline. Random fill, QD immigrant and restart trees are drawn again (up to 8 times) when they are constant and the island already holds one. Seeds and migrants are not filtered. The flag is off by default.
- **Survivor selection** is generational by default: the island is replaced by
  its offspring and only the best individual is carried over. With
  `--selection afp` parents and offspring compete: the next population is the
  Pareto-best set on (error, complexity, age), plus a few random immigrants per
  generation. Small correct shapes and new lineages survive next to large
  approximations, and individuals that are no better than the mean model are
  dropped by their rank. Parents are then drawn by Pareto rank and crowding
  distance.
- **Variation** has two parts:
  - *mutation* — the edit kind is drawn first (25 % point change, 35 %
    constant/input perturb, 40 % subtree grow/shrink) and the site is chosen
    among nodes that the edit can actually change, so a mutation never returns
    the parent unchanged;
    `--mutation-mix extended` adds four structural edits: *insert* wraps a node
    in a random operator, *delete* replaces a node by one of its children,
    *swap* exchanges the operands of a non-commutative binary operator and
    *rotate* turns `o1(a, o2(b, c))` into `o2(o1(a, b), c)`. Weights: point
    0.20, terminal 0.30, subtree 0.20, insert 0.10, delete 0.10, swap 0.05,
    rotate 0.05. `--adaptive-mix` lets each island reweight the seven kinds
    from the success rate (a child beats its first parent) over the last 10
    generations, `w = max(0.02, (s + 1) / (a + 2))`, normalised; it learns
    from CPU-scored generations only;
  - *crossover* — swap subtrees between two parents; `--crossover-internal P`
    picks an internal node as the cut point with probability `P`;
- **Constant fitting** tunes the numeric constants of a candidate by local
  search. Without it, `y = a·sin(b·x) + c` would almost never be found, because
  random constants rarely land near the right values.
- **Migration** periodically copies an island's best member into a neighbouring
  island; **stagnant islands are restarted** to avoid getting stuck. Every
  restart interval (`--restart-interval`, default 60 generations) the default rule (`--restart-policy archive`)
  restarts an island after two checks in which its best error is worse than the
  archive best. Once migration has copied the archive best into every island,
  no island is behind it and that rule never fires. `--restart-policy island`
  (#1022) reads the island's own stall instead: an island restarts when its best
  error has not improved for one restart interval, except the keeper (the lowest
  island whose best equals the archive best, so the best expression is never
  discarded). A restarted island is isolated for `--restart-isolation`
  generations (default 25): it draws no parent from the global front, no
  `--qd-policy` parent (#1067) and takes no migrant, so its new random material
  is not pulled back at once to the region the other islands share; it breeds
  from its own members only. With one island nothing restarts. One island-policy
  module decides migration, restarts and parent sources for the owned-tree loop
  and the flat postfix loop alike (#1081).
- **Explore tier** (`--explore-islands K`, #1100, default 0). Without it a run
  behaves as one trial: the global front and the ring spread a new best to
  every island within a few migration intervals, and the restart rules above
  are fixed cutoffs. With it the highest `K` islands are explore islands. Each
  is isolated for its whole life (no front parent, no `--qd-policy` parent, no
  migrant in or out) and restarts at the end of each lifetime of the Luby
  universal sequence 1, 1, 2, 1, 1, 2, 4, ... in units of the restart interval
  (`--restart-interval`, default 60 generations; 0 turns every restart off,
  explore restarts included). One sequence is spread over the explore islands: each next
  lifetime goes to the island whose lifetimes add up to the least, ties to the
  lowest island (the list schedule of Cire, Kadioglu and Sellmann 2014). Luby,
  Sinclair and Zuckerman (1993) show that this sequence reaches a target within
  a logarithmic factor of the best fixed cutoff without knowing the run-time
  distribution, provided the trials are independent. The restart generations
  depend on the generation alone, so a checkpoint needs no extra record. The
  other islands form the exploit tier: they keep `--restart-policy` and a
  directed ring among themselves, and they see explore results only through the
  archive and its front parents. QD niche immigrants and AFP immigrants still
  arrive on an explore island; they are fresh random trees. On the flat postfix
  path a child that duplicates a program on any island is redrawn, which is the
  one remaining link between an explore island and the others.
- **Front scope** (`--front-scope global|island`, #1070, default `global`).
  With `global` every island draws its 15 % front parents from the one Pareto
  front of all islands, a channel far stronger than the ring: a new best
  reaches every island within about one generation. With `island` each island
  keeps its own error/complexity front of the members it scored, with the
  archive's insert rule, cap and fixed-position filter. Fronts travel only with
  the migrants: at a migration the destination island's front becomes the
  canonical front of its own and the source's members (`canonical_front`
  depends on the member set only, so this merge is associative, commutative
  and idempotent). An island that receives no migrant keeps its front, and a
  restart empties the restarted island's front. The global archive still takes
  every member of every island each generation, so the report, patience and
  the restart rules read it unchanged. `--qd-policy` is refused with `island`:
  its QD archive is one parent pool for every island.
- Termination is by generation count or a time/effort budget.

## 7. Simplification and presentation

Scoring and de-duplication use the sound normal form (`expr::simplify`): constant
folding and only those identities that preserve every IEEE value, including
`NaN`, `±inf` and signed zero. Before output, the display relation
(`expr::simplify_display`) additionally applies affine normalisation and
`x+0 -> x` for a readable equation, and the tree is **denormalised** back to the
original variable units.

**Front prune (#466).** The sound simplifier must keep code that is neutral only on the training data: `x ^ (0 * f)` is 1 only for finite `f`, `if(c, a, b)` is `a` only where `c > 0`, `0 + n` differs from `n` at `-0`. With `--prune` (off by default), after the generation loop, and so also after the `solved` stop, each front member is pruned by measurement. For every internal node the prune tries each child and a constant (the mean of the subtree) in its place, refits the constants with the exact CPU fitter and keeps the smallest result whose training error is within `1e-6 * (1 + |e|)` of the original, the archive's epsilon-dominance tolerance. It repeats until nothing is accepted, offers every accepted tree to the front, and the front is rebuilt on all training rows in f64. The pass uses no RNG, so its result is a function of the tree and the data. `--prune` enables it, `--prune-rows` caps the rows it fits on and `--prune-budget` caps the fits. `expr::simplify_equal_branches` folds `if(c, a, a)` to `a`, which is exact for every condition including `NaN`; the prune uses it, the search does not, so a default run keeps the search trajectory.

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
