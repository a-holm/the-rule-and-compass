# Making the search faster, and why it works

This document is in four parts:

1. **The shape of the problem** — a plain-language account of what the search is
   actually searching over, and a mathematical (category-theoretic) framing that
   tells us *where the wasted work is*.
2. **What the literature says** — the techniques that are known to reduce
   evaluations and wall-clock time, with citations.
3. **What this engine now does**, and the measured before/after numbers.
4. **Hardware** — what to do on the machine we have, and what to do on an
   NVIDIA DGX Spark.

Throughout, claims are labelled **[measured]**, **[implemented]**,
**[literature]**, or **[planned]**.

---

## 1. The shape of the problem

### 1.1 In plain language

We want a short formula `f` such that `f(x) ≈ y` for a table of data. "Short" and
"accurate" pull against each other, so there is no single right answer — there is
a *trade-off curve* of formulas, from tiny ones that are a bit wrong to large
ones that are nearly exact. The engine's job is to walk that curve down.

Two things make this hard:

* **The space is enormous.** With ~15 operators and ~5 variables there are more
  distinct expressions of 20 nodes than there are atoms in the visible universe.
* **Most expressions are useless.** Nearly all of them are far worse than "just
  predict the average".

So the goal is not "search harder". It is "spend every evaluation where it can
still matter, and never evaluate the same thing twice".

### 1.2 The mathematical framing

This is not decoration — each piece of the framing below corresponds directly to
a speed-up in the code.

**Syntax is a free algebra.** Expressions are terms built from a signature
`Σ` (the operators: `+`, `×`, `sin`, …) and a set of variables `X`. The set of
all expressions is the *free algebra* `T(Σ, X)`: it is "free" because it imposes
no laws beyond the formal construction — `x + 1` and `1 + x` are different
elements of it.

**Semantics is an algebra, and evaluation is its unique homomorphism.** A
dataset fixes an algebra `A` (real numbers with real functions). By the universal
property of the free algebra there is exactly one *homomorphism* of `Σ`-algebras

```
eval : T(Σ, X) ──▶ A
```

sending each formal term to the function it computes. This map is the bridge
from syntax (`x + 1`) to semantics (the function `x ↦ x + 1`).

Free ⊣ forgetful: `T(Σ, ·)` is left adjoint to the forgetful functor from
`Σ`-algebras to sets. The unit of this adjunction is "regard a variable as a
term"; this is precisely why *any* choice of meaning for the variables extends
uniquely to a meaning for every term.

**All of the waste lives in the kernel of `eval`.** The kernel is the set of
pairs of terms that mean the same function:

```
ker(eval) = { (s, t) : eval(s) = eval(t) as functions }
```

`x + 1` and `1 + x` are in it. `sin(x)² + cos(x)²` and `1` are in it. Every pair
in the kernel is *redundant work*: if you evaluate both, you have paid twice for
the same information. The quotient `T(Σ, X)/ker(eval)` — the **coequalizer** of
the two projections — is the set of *distinct behaviours*, which is what the
search should really be exploring.

That single observation motivates three of the optimisations below:

| Framing | Concrete optimisation |
|---|---|
| Kernel of `eval` is waste | structural hashing + de-duplication **[implemented]** |
| Coequalizer = distinct behaviours | semantic hashing, e-graphs / equality saturation **[planned]** |
| `eval` is a homomorphism | evaluate shared sub-terms once; cache sub-term columns **[planned]** |

**The trade-off curve is an antichain.** Order formulas by two criteria:
error (lower is better) and complexity (lower is better). This is a *product
order*. The useful answers are the maximal elements — an **antichain**, the
Pareto front. Anything dominated by another point is strictly useless and need
not be stored or bred from. Keeping an archive of the front is therefore not
bookkeeping, it is the search's memory of "what is still worth exploring".

**Mutation and crossover are endomorphisms.** The operators act on `T(Σ, X)`.
They are *random*, not structure-preserving, which is why the population drifts.
Two consequences:

* because operators act on *syntax*, they can easily produce semantically equal
  terms (walking straight into the kernel again) — hence de-duplication;
* because fitness is measured in *semantics*, the operators are "blind": this is
  the gap that semantic operators (below) try to close.

**Simplification is a rewriting system and a proof obligation.** The engine has
two relations with different jobs:

* **`expr::simplify` — the sound normal form (N_core).** It is used for scoring,
  de-duplication and CSE. Every rule is value-preserving on IEEE-754 values,
  including `NaN`, `±inf` and signed zero; a rule that needs a side condition is
  applied only when the non-relational definedness/range analysis
  (`expr::analyze_total`, O(nodes)) proves it. The rules are: constant folding
  when the result is finite; `-(-x) → x`; `x + -0 → x` (and `x + +0 → x` only
  when `x` is provably non-zero); `x - +0 → x`; `x - x → 0` only when `x` is
  finite; `x * 1 → x`, `x / 1 → x`, `pow(x,1) → x`, `pow(x,0) → 1`;
  `x / x → 1` only when `x` is finite and non-zero; `0 * x → ±0` only when `x`
  is finite with a known sign (the sign follows `x`); and branch pruning for a
  constant `If` condition. `Add` and `Mul` children are ordered by a total
  structural key (`op index`, value bits, arity, children, block definition),
  so `a+b` and `b+a` share one normal form. There is **no** associativity,
  distributivity or affine re-association: f64 `+` and `*` are not associative,
  and those rules belong to display only.
* **`expr::simplify_display` — the display relation (R_approx).** It adds affine
  normalisation (`x/1.1839 → 0.8447*x`), branch pruning and tolerance-based
  identity absorption for a readable printed equation. It may change a value at
  an IEEE edge case and is **never** used for scoring, dedup, CSE or a
  checkpoint hash.

The sound core terminates (every rule either shrinks the tree or folds it) and
is confluent on its guarded fragment: it is applied bottom-up in one pass, the
commutative orientation is a total order, and the result is idempotent
(`simplify(simplify(t)) = simplify(t)`). The tests exercise idempotence,
critical pairs, a 10k-tree differential corpus over edge values, and a 10k-tree
soundness/precision check of the analysis. Equality saturation with an e-graph
would compute a larger quotient by searching the theory directly **[planned]**.


---

## 2. What the literature says

### 2.1 Constants are the biggest single lever

Random constants almost never land on the right values; the shape can be right
while the numbers are hopelessly wrong. The standard fix is to treat the leaves
as the parameters of a nonlinear least-squares problem and solve it:

* **Levenberg–Marquardt / Gauss–Newton** — the workhorse. Benchmarking eight
  constant-optimisation schemes over ten problems, dos Reis, Caminha & Penna
  (2024) report mean squared error ≈ **117 with no optimisation** versus ≈ **6.5
  with Levenberg–Marquardt**, and conclude that "symbolic regression without
  optimisation does not converge". Kommenda et al. (2013) report the same
  direction at a modest per-evaluation cost.
* **Linear scaling** (Keijzer 2003/2004) — solve `y = a·f(x) + b` in closed form
  with two dot products. Nearly free, and it "relieves the search from
  coefficient learning" (Chen, Xue, Banzhaf & Zhang, GECCO 2023). Best
  accuracy-per-unit-effort trick there is.
* **Ephemeral random constants** — random constants at each leaf. Cheap, weak;
  best used as seeds for the methods above.

### 2.2 Semantic operators

Pawlak, Wieloch & Krawiec (IEEE TEVC 2015) propose **semantic backpropagation**:
decompose a tree into a sub-program and its context, compute the *desired*
semantics of the sub-tree, and replace it with the library member closest to that
target. Reported: shorter programs and significantly lower test error than
forward-propagation operators. **Geometric semantic operators** (Moraglio,
Krawiec & Johnson, PPSN 2012) guarantee the offspring lies between the parents in
semantic space, smoothing the landscape — at the cost of heavy bloat, so they
need slack control. Survey: Vanneschi, Castelli & Silva (GPEM 2014).

### 2.3 Multi-objective search

For two objectives, **NSGA-II** (Deb et al. 2002) — fast non-dominated sort plus
**crowding distance** — is still the sweet spot. Two cheap additions matter:

* **ε-dominance** (Laumanns, Thiele, Deb & Zitzler 2002): treat errors within a
  tolerance as equal. This bounds the archive, removes near-duplicate error
  levels and stops the front filling with numerically-equivalent variants.
* **ParetoPG** (Smits & Kotanchek 2005): exploit the front to accelerate the
  search directly.

Many-objective variants (NSGA-III, MOEA/D) only pay off with three or more
objectives.

### 2.4 Bloat control

Bloat multiplies evaluation cost and hurts generalisation. The best-return,
lowest-risk options (Luke & Panait 2006) are:

* **lexicographic parsimony pressure** — pick on fitness, break ties by size;
* **double tournament** — first tournament on fitness, second on size;
* **ALPS / age-layering** (Hornby 2006) — new material competes only within its
  own age layer, which also fights premature convergence.

All three change only the selection step; no new tuning parameter is introduced.

### 2.5 Strong ideas from other systems

* **Linear-tree encoding + cached sub-trees** (Burlacu, Kronberger & Kommenda
  2020) — cache a sub-tree's output so re-evaluation after a local change touches
  only the path to the root.
* **Simplification rules, per-operator complexity weights, warm starts**
  (Cranmer 2023) — restart from a saved front rather than from nothing.
* **FFX** (McConaghy 2011), **SINDy** (Brunton, Proctor & Kutz 2016, PNAS) —
  cast the problem as *sparse regression over a fixed library*; selection becomes
  convex and cheap. Excellent when a sensible library exists; weak otherwise.
* **AI Feynman** (Udrescu & Tegmark, Science 2020 / NeurIPS 2020) — use
  dimensional analysis, symmetry, separability and modularity to *decompose* a
  hard multivariate problem into small univariate ones before searching.
* **Hash-based best-first search** (Kahlmeyer, Giesen et al., IJCAI 2024) and
  **MCTS with sub-tree reuse** (Kamienny et al., ICML 2023; NeurIPS 2025) —
  credit partial expressions so the same sub-expression is never recomputed.

### 2.6 Where the state of the art stands

**SRBench** (La Cava et al., arXiv:2107.14351) evaluates 14 symbolic-regression
and 7 machine-learning methods on 252 problems and reports that *genetic
programming with nonlinear least-squares constant optimisation plus parsimony*
remains the most accurate family on noisy, real-world tasks. Neural and
reinforcement-learning methods (DSO; EQL) are strongest at exact recovery on
synthetic data. The practical conclusion for this engine: keep the GP core,
invest in constant optimisation, parsimony and the front — and use neural or
library methods only to *seed* or *pre-process*.

---

## 3. What this engine does now

**[implemented]** — all of the following are in the code and covered by tests:

| Change | Why |
|---|---|
| Adjoint-based **Levenberg–Marquardt** constant fitting | One backward sweep gives the whole Jacobian: O(nodes) per row, independent of the number of constants. An undamped Gauss–Newton polish reaches machine precision on well-conditioned fits. |
| **Sound normal form** in scoring/dedup/CSE, **affine normalisation** in display | `simplify` preserves every IEEE value (including NaN/±inf/±0) and applies guarded cancellations only with a definedness proof; `simplify_display` keeps the readable affine forms. Smaller printed forms *and* deterministic dedup keys. |
| **ε-dominance** archiving | Removes numerically-equivalent, larger expressions from the front. |
| **Structural hashing + de-duplication** | Never evaluate two identical candidates in a generation. |
| **Lexicographic parsimony pressure** | No size-penalty coefficient to tune; bloat is controlled at selection time. |
| **Parallel islands** | Islands are independent, so they evolve on separate threads. |
| **Convergence / time-budget stopping** | Stop when there is nothing left to gain, or when a caller's deadline arrives. |

### 3.0 Constants are fitted to the scoring loss

The constants of a candidate are fitted against the **configured `--metric`**,
not always against squared error (issue #117). The objective that ranks the
front and the objective that moves the constants are now the same wherever that
is mathematically well posed:

| metric | fitting strategy |
|---|---|
| `squared`, `aic`, `aicc`, `bic` | the exact historical Levenberg–Marquardt least-squares path, bit for bit |
| `absolute` | IRLS with weights `1 / max(|r|, 1e-6)`, each weighted solve on the same LM/QR solver |
| `log_squashed` | IRLS with weights `1 / (1 + r²)`, the derivative of `ln(1 + r²)` |
| `auc`, `log_loss`, `hinge` | logistic IRLS on the two-class label, with the model output as a logit; `auc`/`hinge` use the smooth logistic surrogate and the configured metric accepts or rejects each step |
| `correlation`, `rank_correlation`, `signed`, `median`, `maximum`, `interquartile`, `slope` | squared fit, then a deterministic coordinate polish on the configured metric (no least-squares normal equations exist for rank/order statistics) |
| information criteria | least squares for the constants, with the parameter penalty applied at scoring time |

Every strategy is deterministic, uses the same adjoint/Jacobian and QR solve,
and writes a trial back only when the configured loss improves. `squared`
results are bit-identical to the pre-#117 engine (`tests` pin the constants and
SSE). The weighted normal equations now scale **both** the Jacobian and the
residual by `sqrt(w)`; before this change the weighted fit mixed a weighted
Jacobian with a half-weighted gradient, which is corrected as part of the same
work.

The fitted loss is reported in `--json` under `fitting` (`loss` names the
objective, `value` is the best fitted training loss). A non-smooth metric whose
polish cannot improve on the squared fit keeps the squared constants and the same
reported score.

### 3.1 Measured results

Same seeds, same generations, same population; wall-clock on the development
machine (4 physical cores; details in §4). Errors are the best **training** error
reached (`squared` metric unless noted).

| Problem | Rows | Baseline error | Baseline time | Now | Now time | Generations used |
|---|---|---|---|---|---|---|
| `y = 3x² − 2x + 1` | 100 | 2.13 × 10⁻¹ | 9.91 s | **9.4 × 10⁻²⁰** | **2.83 s** | 50 / 400 |
| `y = sin(x₁) + x₂²` | 400 | 1.59 × 10⁻³ | 62.9 s | **5.7 × 10⁻²¹** | **44.9 s** | 180 / 500 |
| `y = e^{−0.4x}·cos(3x)` | 120 | 2.59 × 10⁻² | 12.0 s | **2.47 × 10⁻⁵** | 12.6 s | 155 / 400 |
| `y = 2x + 1` | 40 | — | — | **5.3 × 10⁻⁷** (rounding floor) | 1.8 s | 200 |
| `y = sin x` | 60 | — | — | **3 × 10⁻³⁰** | 2.2 s | 300 |

### 3.2 The constant-fitting budget

`--fit-iters N` sets the Levenberg–Marquardt budget for every candidate, and the
budget is honoured literally: `--fit-iters 0` (the same as `--no-fit`) performs
**no fitting at all** — no damped LM iteration and no undamped Gauss–Newton
polish — so the constants stay exactly as the tree was generated. Earlier
releases silently ran five steps per candidate under `--no-fit` (one forced LM
iteration plus a four-step polish); after the #135 fix the same daily-panel
search went from about 3 200 to 7 240 candidate evaluations per second. With the
default `--fit-iters 8` the fit is bit-for-bit unchanged.

Three effects are visible. First, **accuracy**: constant fitting turns
"approximately right" into "exact", which is why the errors drop by 13–20 orders
of magnitude on the first two problems. Second, **speed**: the third problem is
solved with 10⁻² → 10⁻⁵ error in the *same* time; the first is 3.5× faster
because the engine converges and stops early instead of burning all 400
generations. Third, **parsimony**: recovered formulas print as `1 + 2·x` and
`sin(x)` rather than as a hairball of nested divisions.

### 3.2 What is left on the algorithm side

Ordered by expected gain per unit effort **[planned]**:

1. **Linear scaling** around the fitness function (`y = a·f(x) + b` in closed
   form). Cheap, and it makes the *ranking* of unscaled candidates much more
   informative during early generations.
2. **Column-oriented (SoA) evaluation with cached sub-terms.** Move the data
   matrix to one contiguous array per variable and evaluate node-by-node over
   whole columns, caching each node's column. Re-evaluation after a local
   mutation then touches only the path to the root. The evaluator is implemented,
   wired into the search and measured in §3.3; caching sub-term columns is the
   remaining step.
3. **Crowding distance** on top of ε-dominance, for a better-spread front.
4. **Node-level memoisation** across generations (a hash map from sub-term to
   column), which is the practical, incremental version of "quotient by the
   kernel".
5. **Semantic backpropagation mutation** (replace a sub-tree with the library
   member whose semantics is closest to the desired one).
6. **Warm starts from a persisted front**: save the archive and restart from
   it, so later runs begin with the best shapes found so far, rather than
   rediscovering them. (Pareto-front parent feedback *within* a run is
   implemented and measured in §3.2.1.)

### 3.2.1 Pareto-front parent feedback (#76)

The archive used to be write-only: updated every generation and read only to
decide whether an island was stagnant. A shape found on another island, or ten
generations earlier, survived only if its own lineage did.

Now, with probability **15%**, one of the two parents is drawn uniformly from
the current Pareto front instead of the island tournament; the other still comes
from the island. The draw is seeded, so a run remains reproducible. The front is
copied once per generation into the island threads (a small `Vec` of front
members), so the islands stay independent.

Measured on the known-truth suite (seed 1/7/42, full documented budget), off vs
on:

| seed | off | on | `pythagoras` R² at 0% / 1% / 10% (off → on) | wall |
|---|---:|---:|---|---:|
| 1 | 72/75 | **74/75** | 0.987 / 0.721 / 0.987 → **0.993 / 0.987 / 1.000** | 272 → 288 s |
| 7 | 72/75 | **75/75** | 0.955 / 0.962 / 0.957 → **0.998 / 1.000 / 1.000** | 263 → 278 s |
| 42 | 72/75 | **74/75** | 0.942 / 0.766 / 0.952 → **0.996 / 1.000 / 0.981** | 286 → 289 s |

`pythagoras` was the suite's only miss, and it is recovered in **7 of 9** cells
with the feedback versus **0 of 9** without. The cost is about 5% wall time.

This changes the default search trajectory, so it is a **quality change, not an
optimisation**: the exactness hashes from before this feature no longer apply to
a default run. `--no-front-parents` restores the previous trajectory exactly
(both pinned hashes hold with it), and a checkpoint records the setting and
refuses to resume across a change of it.

### 3.3 Structure-of-arrays CPU evaluator (stages 1–2, wired)

**[implemented, wired into the search and measured]** — `src/eval_soa.rs`
compiles one expression to a flat postfix program and evaluates that program
over whole columns. It is the CPU counterpart of the postfix bytecode in
`src/gpu.rs`: the per-row pointer walk becomes one linear instruction stream
over a structure-of-arrays (SoA) buffer.

**Stage 2 wiring.** `search::eval_error` now compiles each candidate once and
evaluates its rows in batches. The sample is transposed to `cols[var][row]`
**once per sample** (per generation, and once for the test set), not once per
candidate, and the transpose is shared by every individual scored on it.
`Config::simd` selects the path; `--no-simd` forces the historical per-row
`expr::eval` walk. The flag is an escape hatch for debugging and A/B runs, not a
search setting: it is not saved in a checkpoint and a checkpoint written by one
path resumes under the other. Both paths are required to produce bit-identical
errors, so the choice cannot move the archive.

A tree with any operator outside the supported set below makes `compile` return
`None`, and **that tree** transparently falls back to the scalar walk. The
aggregates, meta operators, constraints, comparisons and user-defined blocks all
take this fallback. Multi-output samples (`Sample::ys`) and the implicit metrics
keep their own per-output fit/metric paths unchanged and do not use the SoA
kernel; the search's single-output, non-implicit fitness path does.

The API is deliberately small:

```rust
use rcsr::eval_soa::{columns, compile, eval_batch, eval_batch_scalar, sse};

// cols[var][row], row-major inside each column.
let cols = columns(&x);                            // x[row][var] -> cols[var][row]
let program = compile(&tree).expect("supported tree"); // None => fall back
let mut out = vec![0.0; rows];
eval_batch(&program, &cols, rows, &mut out);        // AVX if available
eval_batch_scalar(&program, &cols, rows, &mut out); // portable SoA loop
let sum_sq = sse(&program, &cols, rows, &y);        // convenience
```

`compile` computes the postfix stream and its peak operand-stack depth once;
`eval_batch` allocates the value stack once per call, not per row. `rows` is
clamped to `out.len()`, so zero rows and short buffers are safe. A trailing
group shorter than the SIMD width is evaluated by the scalar loop, so any row
count works.

**Supported operators.** Unary: `neg`, `exp`, `ln`, `abs`, `sqrt`, `sin`,
`cos`, `tan`, `sinh`, `cosh`, `tanh`, `logistic`, `step`, `gauss`, `square`,
`cube`, `floor`, `ceil`, `round`, `sign`. Binary: `add`, `sub`, `mul`, `div`,
`pow`, `min`, `max`. Ternary: `if`. This covers the whole default primitive set
plus `min`/`max`.

Everything else returns `None` from `compile` so the caller can fall back:
`mod`, `atan2`, `erf`, `erfc`, `asinh`, `acosh`, `atanh`, the comparisons and
logical operators, the constraint operators (`require`/`between`/`constrain`)
and the meta operators (`complexity`/`variables`/`constants`/`contains`/
`contains_op`).

**Domain guards.** The guards are copied from `expr::eval` verbatim:
`exp(a > 700) = +inf`, `ln(a <= 0) = NaN`, `sqrt(a < 0) = NaN`,
`div(|b| < 1e-12) = NaN`, `pow(negative, non-integer) = NaN`, `min`/`max`
propagate `NaN`, and `if` selects `cond > 0`. The tests compare every row of a
deterministic generated corpus against `expr::eval`; the measured maximum
relative difference is **0** (bit-identical) on the scalar and AVX paths.

**What vectorises.** On `x86_64`, the kernel is compiled with
`#[target_feature(enable = "avx")]` and selected at runtime with
`is_x86_feature_detected!("avx")`. The development CPU has AVX but no AVX2 and
no FMA, so the kernel is four `f64` lanes and needs neither. Nothing requires
nightly.

| operation | AVX kernel | portable fallback |
|---|---|---|
| `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sqrt`, `square`, `cube`, `min`, `max` | vectorised (4 rows per instruction) | scalar per row |
| `if` | vectorised as a lane mask + blend | per-row select |
| `exp`, `ln`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `logistic`, `step`, `gauss`, `floor`, `ceil`, `round`, `sign`, `pow` | one scalar `f64` call per lane (no AVX instruction for them) | scalar per row |

Even where the transcendental itself is scalar, the tree walk and the
per-row recursion are gone; those cases therefore still gain from the flat
stream, just less than pure arithmetic.

#### Measured: scalar walk vs SoA

`cargo run --release --bin cpu_bench` evaluates four deterministic trees (5, 9,
15 and 25 nodes) over 1 000, 20 000 and 200 000 rows. The scalar column is the
existing `expr::eval` tree walk on row vectors (what the search uses today);
the SoA columns are `eval_batch_scalar` and the runtime-selected AVX path.
Times are µs for one full pass, best of five runs, each sized to about 50 ms.

| nodes | rows | walk µs | SoA scalar µs | SoA AVX µs | SoA speed-up | AVX speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 000 | 19.37 | 16.16 | 3.56 | 1.20× | **5.44×** |
| 5 | 20 000 | 381.87 | 324.25 | 72.09 | 1.18× | **5.30×** |
| 5 | 200 000 | 4 106.38 | 3 322.97 | 742.16 | 1.24× | **5.53×** |
| 9 | 1 000 | 59.93 | 48.87 | 34.58 | 1.23× | **1.73×** |
| 9 | 20 000 | 1 219.78 | 970.00 | 688.92 | 1.26× | **1.77×** |
| 9 | 200 000 | 12 185.48 | 9 764.19 | 8 217.21 | 1.25× | **1.48×** |
| 15 | 1 000 | 131.20 | 102.61 | 82.62 | 1.28× | **1.59×** |
| 15 | 20 000 | 2 613.77 | 2 087.46 | 1 572.26 | 1.25× | **1.66×** |
| 15 | 200 000 | 29 489.78 | 20 722.41 | 16 505.77 | 1.42× | **1.79×** |
| 25 | 1 000 | 235.90 | 215.96 | 151.77 | 1.09× | **1.55×** |
| 25 | 20 000 | 5 020.49 | 4 295.85 | 3 063.77 | 1.17× | **1.64×** |
| 25 | 200 000 | 49 051.38 | 43 449.00 | 29 596.99 | 1.13× | **1.66×** |

Measured on the development machine in §4.1 (AVX, no AVX2/FMA). The host is
shared, so each cell is the minimum over eight alternating runs of
`cpu_bench`; an individual run can be several percent slower. The 5-node tree
is pure arithmetic and shows the ceiling of the approach. The 9-, 15- and
25-node trees mix arithmetic with `sin`/`cos`/`exp`/`tanh`, which AVX alone
cannot vectorise, so their 1.5–1.8× is the more realistic range for search
candidates.

**Agreement:** maximum relative difference against the scalar tree walk is
`0.0` for both SoA paths on all twelve tree/row combinations (same `f64`
operations, same guards). The benchmark prints this check on every run.

The 200 000-row cases are memory-bound and noisier than the smaller ones; the
trend that matters is the AVX speed-up over the existing walk, which grows with
the arithmetic share of the tree. Trees dominated by `sin`/`cos`/`exp` gain
less because AVX alone cannot vectorise those functions.

#### Constants behind `sign` and `step`

`sign` and `step` are piecewise constant: their derivative is zero almost
everywhere, and so is the adjoint that [`fit.rs`](../my-version/src/fit.rs)
uses to move constants. A constant that reaches the output only through one of
them therefore never changes during fitting; whatever threshold the search
happened to create is printed, but it is not an optimum. When the fitted Pareto
front contains such a constant the CLI prints a warning naming `sign`/`step`.

The workable pattern is **fit a smooth surrogate, threshold at inference**:
search and fit `logistic(k·(x - t))` or `tanh(k·(x - t))`, which have usable
gradients in `k` and `t`, then replace the surrogate with the hard switch only
when the final rule is written down. Alternatively pin the threshold as a
literal constant and do not ask the fitter to move it. A model using `sign` or
`step` is still valid as a *given* rule; it is only the joint fitting that is
ill-posed.

#### Measured: search-level effect

The kernel table measures one evaluation. In a real search the same candidate is
also fitted with Levenberg–Marquardt: with the default `--fit-iters 8`, every
offspring is walked roughly eight more times (value and adjoint gradient per
row) before its fitness is taken once. Simplification, structural hashing,
archive bookkeeping, RNG draws and migration add work that neither evaluator
touches. The end-to-end gain is therefore much smaller than the kernel gain.

Two reproducible tables, three inputs, `--islands 2 --seed 42 --patience 0`,
scalar (`--no-simd`) alternated with the SoA path, five runs each. The host was
shared (load average ~12 on 8 cores), so **wall-clock is noisy**; the table
reports median wall-clock and median user+sys CPU time, and the CPU ratio is the
stable number. Archives were byte-identical between the paths in every pair
(bit-for-bit `error`, `cost`, tree and test error are asserted by
`tests/search_simd.rs`).

| rows | generations × pop | fit | scalar wall | SoA wall | scalar CPU | SoA CPU | CPU speed-up |
|---:|---|---:|---:|---:|---:|---:|---:|
| 800 | 30 × 120 | off | 0.81 s | 0.64 s | 0.43 s | 0.33 s | **1.30×** |
| 5 000 | 15 × 100 | off | 1.61 s | 1.43 s | 1.67 s | 1.45 s | **1.15×** |
| 800 | 30 × 120 | 8 | 4.10 s | 4.55 s | 6.32 s | 6.28 s | **1.01×** |
| 5 000 | 15 × 100 | 8 | 9.23 s | 9.20 s | 14.34 s | 14.03 s | **1.02×** |

The first two rows are `--no-fit`, which isolates the part of the search the
evaluator actually owns: a **1.15–1.30×** end-to-end gain. The last two keep
the default LM fitter, and the gain collapses to **1.01–1.02×**. Archive
complexities were 1–12 (800 rows) and 1–10 (5 000 rows); stage-1 trees of 9–25
nodes gained 1.5–1.8× in isolation, but the search's small candidates carry more
constant-per-tree overhead and the fitter dwarfs the scored evaluation. The
5 000-row case gains less than the 800-row one because its columns no longer fit
in the same cache level.

Wiring the LM fitter's value/Jacobian passes onto the same SoA layout is the
next real win; it is deliberately out of scope here, where the requirement is
that every existing fit and metric path keeps its exact behaviour.

#### The exact final-SSE reuse (#87)

The fitter's final SSE is the score for a `squared` fit on the same rows, so the
search reuses it instead of scoring the fitted tree a second time. This changes
no fit and no trajectory. The **initial** SSE is deliberately still computed by
the standalone `sse(c)`: using the first `gn_step`'s residual sum as the
acceptance baseline changed the LM trajectory on guard-free trees, so that
reuse was reverted. `tests/trajectory.rs` pins the complete front for the
documented baseline command, so a future shift fails loudly.

The remaining final-reuse saving is small. The 12 `gn_step` sweeps also run the
adjoint `backward` pass and build `JᵀJ`/`Jᵀr`, so they cost several times a
forward `sse`; the one removed scoring sweep is the cheap kind. End-to-end it is
within measurement noise on `w2.csv` (the earlier 2.6–4.8 % figure included the
now-reverted initial reuse). The reuse is exact and applies to `squared` only —
this crate has no `rmse` metric — and is refused when the fit and score rows
differ, the tree folded constants, the sample is weighted/multi-output, or a
guard makes the fitted SSE non-finite. `fit.rs::value_of` now applies the two
guards that can leave a *finite* fitter value where the scorer rejects the
tree — `Exp > 700 → +inf` and `Div |denominator| < 1e-12 → NaN` — and the
fitter rejects `x.len() != y.len()`. The other operators already produce a
non-finite fitter value in the same cases (`Ln(0) → -inf`, `Sqrt(<0) → NaN`,
`Tan(inf) → NaN`, `Pow → NaN`, `Atan2 → NaN`), so guarding them too changed the
multi-output search trajectory without fixing a finite-vs-infinite divergence.

#### The exact linear least-squares arm (#116)

The default fitter is still the LM/adjoint path described above. Some trees do
not need iterations at all: when the value is **affine in every constant** —
`f(x; c) = f(x; c₀) + Σⱼ (cⱼ − c₀ⱼ)·gⱼ(x)` — the minimiser is the solution of a
linear least-squares problem. `--fit-strategy auto` (or `linear`) detects that
class structurally and solves it directly:

* the adjoint sweep at the current constants gives the coefficients `gⱼ(x)`
  exactly and independently of `c₀`;
* the weighted design matrix `√w·J` is solved by **column-scaled,
  column-pivoted Householder QR**, never the normal equations, so the condition
  number is not squared;
* the constants are written back only if the closed-form weighted SSE is finite
  and no worse than the starting point.

The classifier is structural, not a numerical test: a product of two
constant-dependent factors is bilinear and rejected; any nonlinear operator is
accepted only when its whole subtree is constant-free; `if` may choose between
affine branches only when its condition is constant-free. `a*x + b`,
`a*x² + b*sin(x) + c` and `a*f(x) + b*g(x) + c` qualify; `a*sin(b*x) + c`,
`a*(b + x)` and `exp(a*x)` do not.

**Exactness.** The arm is an exact global minimiser for its class, not an
approximation. The pre-registered acceptance rules are:

| rule | tolerance / behaviour |
|---|---|
| affine objective versus exact LM | QR SSE `<=` LM SSE × `(1 + 1e-12)` |
| constants versus exact LM | relative difference `<= 1e-8` on affine tests |
| rank test | column-scaled pivot norm `> 1e-12`; otherwise exact LM |
| size cap | `rows × constants <= 8e6` and `<= 64` constants; otherwise exact LM |
| ineligible, rank-deficient, non-finite or non-improving | exact LM, tree untouched, bit-identical |
| `--fit-strategy exact` (default) | historical LM, bit for bit; pinned constants and archive hashes unchanged |

Because the linear arm reaches the optimum in one solve instead of iterating,
it **can** change the search trajectory (usually by fitting a linear candidate
better than an 8-iteration LM run). It is therefore opt-in, and it is saved in
checkpoints as search state; `auto` currently has exactly these two arms.

`--fit-report` prints the arm counts for the run. On the daily panel
(`panel_daily_100.csv`, 154,675 rows, target `y1`, inputs `r1,r3,r7,ma7`,
`--seed 3 --islands 1 --pop 200 --generations 5 --patience 0`), two alternating
pairs on a shared host:

| strategy | linear / exact calls | user CPU | wall (median) |
|---|---:|---:|---:|
| `exact` | 0 / 995 | 80.0 s | 110.6 s |
| `auto` | 491 / 504 (49.3 %) | 53.4 s | 66.9 s |

That is **1.50× user-CPU** and **1.65× wall-clock** for the run. The HOURLY panel
(`panel_hourly_compact.csv`, 3.9M rows, target `y60`, `--fill drop --pop 100
--generations 2`) measured 513.6 s → 402.9 s user CPU (**1.28×**, 49.0 % linear
calls) and 631 s → 437 s wall. The archive improvements on the daily run were
strictly non-worse: `auto` reported two extra Pareto members with lower train
and test error (`0.30848993 / 0.31136481` versus the `exact` front's best
`0.30861096 / 0.31146704` at a larger complexity), because the linear arm
solved candidates that LM's eight iterations left short of the optimum.

**Not yet implemented** (tracked on #116): VarPro for separable nonlinear
models, batched GPU LM and the adaptive cost-model dispatcher. This PR lands
only the exact CPU linear arm plus its classifier, fallback and reporting.

#### Subsampled scoring: `--predictor-samples`

`--predictor-samples N` draws a fresh `N`-row subset every generation and uses
it for both the constant fit and the score. #86 made two changes to the original
implementation:

* **Carried individuals are refitted before they are scored.** Previously the
  carried population was only re-scored, so it competed as `fit(S_{g-1}) ->
  score(S_g)` against offspring on `fit(S_g) -> score(S_g)`, and the elite was
  never refitted at all. The refit uses one warm-started LM iteration (plus the
  fitter's four polish iterations) because the constants already come from the
  previous subset; a cold refit doubled the generation's fit cost and erased
  the win at 750/1500.
* **Near-full requests are widened to the full sample.** The fitter-only
  break-even, `m/n < (24 - 1/G)/25 ~ 0.96`, assumed that only a scoring sweep
  was saved for carried individuals. With the refit the model is about
  `(26 + 6) m` against `26 n`, i.e. `m/n < 0.81` before the sample rebuild and
  the extra final archive pass. On the `w2.csv` workload (1 500 train rows,
  `--fit-iters 8`, 10 generations, min of three `%U`) `900/1500 = 0.60` still
  wins but `1000/1500 = 0.67` is slower than full data, so the measured rule is
  `m <= 3/5 n`. `750/1500` stays active; `1400/1500` is an exact no-op that
  uses the full sample. The text header and the `--json`/NDJSON output report
  the requested and effective sizes, and whether the request was widened.

#### Screening the fit rows: `--fit-samples`

The fitter reads every training row on each LM sweep, so the cheapest way to
make it smaller is to spend those sweeps on fewer rows. `--fit-samples N` is a
**within-generation successive-halving** screen (Karnin et al. 2013; Jamieson &
Talwalkar 2016), not just a cheaper refit:

1. every offspring is fitted and scored on a **fresh deterministic** `N`-row
   subset of the generation's sample (a salted stream, independent of
   `--predictor-samples`; the subset is redrawn every generation);
2. the top **1/16** of that screening ranking is promoted;
3. each promoted candidate is **refitted and re-scored on all training rows**;
4. the next population is selected from the full-data parents plus the promoted
   offspring, and only full-data scores enter the archive.

There is no Hyperband schedule across generations: candidates are replaced, not
trained longer, so one promotion round per generation is the relevant
algorithm. `0` (the default) is the historical full-data path bit for bit; a
request at least as large as the sample, or `--no-fit`, is an exact no-op. The
flag is orthogonal to `--predictor-samples`, which also changes the scoring
rows.

This is a **throughput-for-statistics trade, not a free win**. Screening `P`
offspring on `N` rows and keeping the best is optimistically biased (the
winner's curse); promotion and full-data selection reduce that bias but cannot
remove it, because the candidates that never make the top 1/16 are still lost.
On **deterministic, noiseless data the screen is a pure loss**: there is no
noise for the extra rows to average out, so it only spends fewer rows on a
problem a full fit would solve exactly. Use it when the fitter dominates the
run and the data are noisy enough that an `N`-row subset is representative.

Measured on the `w2.csv` workload (2 000 rows, 1 500 train, 30 generations,
pop 100, seed 1, min of three `%U` runs on a shared host), with `m = 200` and a
1/16 promotion fraction:

| config | full data | `--fit-samples 200` | speed-up | refitted best error |
|---|---:|---:|---:|---|
| G=30, absolute | 5.03 s | **2.11 s** | **2.38×** | 0.1633 → **0.0121** (13.5× better) |
| G=30, squared | 4.58 s | **1.65 s** | **2.78×** | **0.0259 → 0.0398** (1.54× worse) |
| G=10, absolute | 1.09 s | **0.50 s** | **2.18×** | 0.1762 → **0.2261** (1.28× worse) |
| G=10, squared | 1.60 s | **0.37 s** | **4.32×** | **0.0259 → 0.0869** (3.36× worse) |

The baselines are the full run's own best shapes **refitted** on all rows, not
their unrefitted errors (the earlier draft compared against the unrefitted
0.0865, which was wrong). On the squared metric — the one the 0.0259 baseline
comes from — the promoted screen is **1.54× worse at G=30** and **3.36× worse
at G=10**, so the quality cost is visible next to the speed-up. The G=30
absolute error improves sharply on this seed (0.1633 → 0.0121), which is why
the delta is reported per metric and not as a general quality claim; the
recovered squared law is an 11-node expression, while the absolute winner is a
22-node polynomial/sine form. A larger promotion fraction (1/8) reduced the
squared error to 0.0042 but erased most of the speed-up (4.03 s, 1.25×), so the
shipped fraction is 1/16. These are one seed each; the flag stays opt-in.

#### Opt-in codegen: `target-cpu=native` and PGO

Neither flag is forced in the repository; the default build is portable. Both
are worth trying on the machine that will run the search.

```sh
# Native code generation (may produce a binary that only runs on this CPU).
cd my-version
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin cpu_bench
./target/release/cpu_bench
```

Across the twelve tree/row cells, `target-cpu=native` changed the AVX
evaluation time by between −2 % and +8 % relative to the default build, with no
consistent direction (for example 15 nodes × 200 000 rows: 16.51 ms default vs
15.86 ms native; 5 nodes × 1 000 rows: 3.56 µs vs 3.64 µs). The explicit
intrinsics are selected at runtime, so this flag mostly affects the surrounding
scalar code. Keep it as an opt-in recipe rather than a requirement; it did not
change the agreement result (still exactly zero difference).

PGO needs the `llvm-tools` component (`rustup component add llvm-tools-preview`).
Instrument with a representative workload, merge the profiles, then rebuild
with the merged profile; combine with `target-cpu=native` if desired:

```sh
cd my-version
PROFDATA=$(find "$(rustc --print sysroot)" -name llvm-profdata | head -1)
PGO=/mnt/lager/tmp/opencode/pgo
rm -rf "$PGO" && mkdir -p "$PGO"

RUSTFLAGS="-Cprofile-generate=$PGO" cargo build --release --bin cpu_bench
./target/release/cpu_bench >/dev/null

"$PROFDATA" merge -o "$PGO/merged.profdata" "$PGO"
RUSTFLAGS="-Cprofile-use=$PGO/merged.profdata \
           -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release --bin cpu_bench
```

The benchmark is a small, branch-heavy interpreter, so PGO mainly helps the
instruction dispatch; the explicit AVX intrinsics do not depend on it. As with
`target-cpu=native`, keep these flags in the build recipe rather than in a
committed configuration, so a release built on one machine still runs on
another.

### 3.4 Dehosting stage 1: pool, canonical hash, parallel fallback (#122)

The fit-free search is host-bound: at `N = 5 000`, `P = 10 000` the evaluation
kernel is ~2 % of a generation, and the rest is variation, simplification,
hashing and — with the default operator set — the **serial** CPU scoring of
`if` programs the kernel cannot compile. Stage 1 of #122 attacks the parts that
do not change the search trajectory:

* `genome::GenomePool` — a fixed-capacity, double-buffered postfix arena with
  SoA metadata. Width is `2*max_complexity + 4` words (64 words = 256 B at the
  default), so slots never move and peak memory is constant (2.6 MB per 10 000
  genomes per buffer).
* a persistent CUDA arena with the same layout (`rc_pool_reserve`,
  `rc_pool_upload`, `rc_pool_eval`): no `cudaMalloc`/`cudaFree` inside the
  generation loop;
* a device canonical hash (`rc_pool_hash`) matching `genome::postfix_hash`, and
  Philox4x32-10 with the positional `(seed, island, generation, slot, draw)`
  contract, both tested;
* a bounded CPU pool for unsupported programs, replacing the serial tail
  (`RCSR_CPU_FALLBACK_THREADS`, default `min(cores, 8)`);
* `Archive::try_insert_view`, which clones a tree only after it survives the
  dominance test.

Measured (RTX 3060, i7-3770K, `N = 5 000`, `--no-fit`, `--gpu`, default
operators, `P = 10 000`, 10 generations, three interleaved runs):
**18.3/22.8 s → 8.8/10.7 s (min/median, 2.1–2.4×)**. The unsupported tail
alone fell from 2.15 s to 0.73 s for 8 521 trees (3.0× on four physical cores);
without `if` the same cell is flat (5.44 s → 5.34 s), since variation and the
startup population dominate there. The JSON report for a 5-generation
`P = 2 000` run is byte-identical to the pre-#122 build apart from
`elapsed_seconds`. Device variation and a device archive remain the next
stages; fitting and all f64 reporting stay on the CPU.

---

## 4. Hardware

### 4.1 The machine we have

| | |
|---|---|
| CPU | Intel Core i7-3770K, 4 cores / 8 threads, 3.5 GHz |
| Vector ISA | **SSE4.2 + AVX (128/256-bit), no AVX2, no FMA** |
| RAM | 31 GB |
| GPU | NVIDIA RTX 3060 (Ampere, `sm_86`, 12 GB) |
| Toolchain | rustc 1.99 (pure `std`, no dependencies), CUDA 13.0 runtime, `nvcc` 12.0 |

What this implies:

* **Only 4 physical cores**, so island parallelism tops out around 4×; the
  measured 3.5× on the first problem is close to that ceiling.
* **No AVX2/FMA**, so 4-wide `f64` and 8-wide `f32` are the practical SIMD
  widths, and hand-written SIMD is needed (the compiler will not vectorise a
  tree walk — LLVM refuses to vectorise loops containing a `switch`).
* **FP64 on the RTX 3060 is 1/64 of FP32** (~0.2 vs ~12.7 TFLOPS), so any GPU
  work must be FP32 to be worthwhile.

### 4.2 The NVIDIA DGX Spark (GB10)

| | |
|---|---|
| SoC | GB10 Grace Blackwell, CPU and GPU dies joined by NVLink-C2C |
| CPU | 20 Arm cores (10× Cortex-X925 + 10× Cortex-A725), SVE2 |
| GPU | 48 SMs, 6 144 CUDA cores, ~2.5 GHz → **≈ 31 TFLOPS FP32** |
| Compute capability | **`sm_121`** (consumer Blackwell, not datacentre `sm_100`) |
| FP64 | **1/64 of FP32 ≈ 0.5 TFLOPS** |
| Memory | **128 GB LPDDR5X, ~273 GB/s, coherent and unified** |
| Software | DGX OS (Ubuntu, aarch64), CUDA 13.0+ |

Sources: NVIDIA DGX Spark hardware guide; Chips and Cheese's GB10 analysis;
NVIDIA RTX Blackwell whitepaper (FP64 = 1/32 or 1/64 of FP32).

The two facts that matter: **unified memory** (the dataset and the whole
population can stay resident — no per-generation upload) and **weak FP64** (the
GPU path must be FP32, with FP64 reserved for final verification of survivors).

### 4.3 Plan

**On this machine [partly implemented]**

1. Keep the CPU search authoritative and dependency-free (**done**).
2. Parallel islands with `std::thread` (**done**).
3. De-duplication and early stopping (**done**).
4. A batched **GPU evaluator** (`--features cuda`) with a measured 274–715×
   speed-up over the reference `f64` tree walk on the RTX 3060 (**done**;
   see `docs/gpu-backend.md`). Wiring it into the generation loop is the next
   step.
5. SoA data + column-oriented evaluation: the evaluator with explicit 4-wide
   `f64` AVX is implemented, wired into the search (`--no-simd` to disable) and
   measured in §3.3; caching sub-term columns is the remaining step.
   **[wired]**
6. `-C target-cpu=native` and PGO recipes are documented and measured once in
   §3.3; neither is forced in the repository. **[recipes documented]**

**On the DGX Spark [planned]**

The kernel already has the right shape (postfix programs, block-per-expression,
FP32, fused reduction); the work is to scale and target it:

1. Build for **`sm_121`** with CUDA 13 (`RCSR_CUDA_ARCH=sm_121`). This is a
   consumer Blackwell target, so datacentre flags such as `sm_100` will not run.
2. Move to **unified memory** (`cudaMallocManaged`) so the dataset and bytecode
   need no explicit copy — 128 GB of coherent memory is the point of the GB10.
3. Batch the **whole population** (all islands) into a single launch, so launch
   latency disappears at ~10⁶ evaluations per launch.
4. Use larger blocks and consider splitting very wide datasets across several
   blocks per expression with a two-pass reduction.
5. Keep **FP64 for the final check** of the surviving front only: FP64 on GB10 is
   1/64 of FP32 (~0.5 TFLOPS).
6. **Do not oversubscribe the 273 GB/s** memory bus: hand off between CPU
   evolution and GPU evaluation rather than running both flat out.

Reported GPU speed-ups for this style of workload range from ~18× (EvoGP) to
~40× (stack-based GP) over tuned CPU implementations, so a realistic first
target on GB10 is **10–40×** over an optimised AVX-512 CPU, rising further with
row tiling and CUDA graphs. The status of the GPU path in this repository is in
`docs/gpu-backend.md`.

---

## Bibliography

1. dos Reis, Caminha & Penna (2024). *Benchmarking symbolic regression constant
   optimization schemes.* arXiv:2412.02126.
2. Kommenda, Kronberger et al. (2013). *Effects of constant optimization by
   nonlinear least squares in symbolic regression.* EUROCAST.
3. Keijzer (2003). *Improving symbolic regression with interval arithmetic and
   linear scaling.* EuroGP.
4. Chen, Xue, Banzhaf & Zhang (2023). *Relieving genetic programming from
   coefficient learning via correlation and linear scaling.* GECCO.
5. Pawlak, Wieloch & Krawiec (2015). *Semantic backpropagation for designing
   search operators in genetic programming.* IEEE TEVC 19(3).
6. Moraglio, Krawiec & Johnson (2012). *Geometric semantic genetic programming.*
   PPSN.
7. Vanneschi, Castelli & Silva (2014). *A survey of semantic methods in genetic
   programming.* GPEM.
8. Deb, Pratap, Agarwal & Meyarivan (2002). *A fast and elitist multi-objective
   genetic algorithm: NSGA-II.* IEEE TEVC 6(2).
9. Laumanns, Thiele, Deb & Zitzler (2002). *Combining convergence and diversity
   in evolutionary multi-objective optimization.* Evolutionary Computation 10(3).
10. Smits & Kotanchek (2005). *Pareto-front exploitation in symbolic regression.*
    GPTP II.
11. Luke & Panait (2006). *A comparison of bloat control methods for genetic
    programming.* Evolutionary Computation 14(3).
12. Hornby (2006). *ALPS: the age-layered population structure.* GECCO.
13. Burlacu, Kronberger & Kommenda (2020). *Operon C++: an efficient genetic
    programming framework.* GECCO Companion.
14. Cranmer (2023). *Interpretable machine learning for science with PySR and
    SymbolicRegression.jl.* arXiv:2305.01582.
15. McConaghy (2011). *FFX: fast, scalable, deterministic symbolic regression
    technology.* GPTP IX.
16. Brunton, Proctor & Kutz (2016). *Discovering governing equations from data
    by sparse identification of nonlinear dynamical systems.* PNAS 113(15).
17. Udrescu & Tegmark (2020). *AI Feynman: a physics-inspired method for
    symbolic regression.* Science Advances 6(16).
18. Kahlmeyer, Giesen et al. (2024). *Scaling up unbiased search-based symbolic
    regression.* IJCAI.
19. Kamienny et al. (2023). *Deep generative symbolic regression with
    Monte-Carlo tree search.* ICML.
20. La Cava, Orzechowski, Burlacu, de França, Virgolin, Jin, Kommenda & Moore
    (2021). *Contemporary symbolic regression methods and their relative
    performance (SRBench).* arXiv:2107.14351.
21. Laumanns et al. — see 9. Deb & Jain (2014). *NSGA-III.* IEEE TEVC 18(4).
22. NVIDIA. *DGX Spark hardware guide.*
23. Chips and Cheese. *Analyzing NVIDIA GB10's GPU.*
24. NVIDIA. *RTX Blackwell GPU architecture whitepaper.*
