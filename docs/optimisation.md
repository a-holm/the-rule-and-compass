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

**Simplification is a rewriting system.** `simplify` applies oriented equations
`l → r` of a `Σ`-algebra theory; if it terminates and is confluent it computes
normal forms, i.e. a section of the quotient map. It is deliberately *not* fully
confluent here, so it is a heuristic — but every rule it applies permanently
shrinks the term, which reduces both evaluation cost and the printed complexity.
Equality saturation with an e-graph would compute the *true* normal form up to a
budget by searching the quotient directly **[planned]**.

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
| **Affine normalisation** in the simplifier | `(x/c₁)/c₂ → k·x`, constant chains folded, branch pruning for constant conditions. Smaller printed forms *and* fewer nodes to evaluate. |
| **ε-dominance** archiving | Removes numerically-equivalent, larger expressions from the front. |
| **Structural hashing + de-duplication** | Never evaluate two identical candidates in a generation. |
| **Lexicographic parsimony pressure** | No size-penalty coefficient to tune; bloat is controlled at selection time. |
| **Parallel islands** | Islands are independent, so they evolve on separate threads. |
| **Convergence / time-budget stopping** | Stop when there is nothing left to gain, or when a caller's deadline arrives. |

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
   mutation then touches only the path to the root.
3. **Crowding distance** on top of ε-dominance, for a better-spread front.
4. **Node-level memoisation** across generations (a hash map from sub-term to
   column), which is the practical, incremental version of "quotient by the
   kernel".
5. **Semantic backpropagation mutation** (replace a sub-tree with the library
   member whose semantics is closest to the desired one).
6. **Warm starts**: persist the front and restart from it.

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
5. Next: SoA data + column-oriented evaluation with cached sub-term columns, and
   explicit 4-wide `f64` SIMD for the arithmetic nodes.
6. Compile with `-C target-cpu=native`; consider PGO. Both are free wins of a
   few percent on branchy code. **[not yet done]**

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
