# Known-truth benchmarks: does the engine recover the law?

A symbolic search is *supposed* to fit data, so a small error proves nothing on
its own. The only way to tell recovery from overfitting is to run the engine on
data whose generating law is known, and to check whether the front it returns
contains that law. This document describes the known-truth benchmark suite, the
recovery criterion, how to run it, and the results recorded on this repository's
current hardware.

The harness lives in `my-version/src/bench.rs` (the suite and scoring) and
`my-version/src/bin/rcsr_bench.rs` (the `rcsr-bench` CLI). It is driven through
the library API (`search::run`), so it measures the engine, not the input
plumbing.

---

## 1. The suite

25 problems: small Feynman-style physics laws and elementary dynamical systems.
Every truth formula is finite over its sampling range, and every problem is
solvable in principle with the engine's default primitive set.

Training and test targets are generated from the truth formula; the search only
ever sees the inputs. The `tiny` column marks the fast CI subset.

| problem | variables | truth | sampling range | tiny |
|---|---|---|---|---|
| `line` | x | `2x + 1` | x ∈ [−2, 2] | ✓ |
| `quadratic` | x | `x²` | x ∈ [−2, 2] | ✓ |
| `cubic` | x | `x³` | x ∈ [−1.5, 1.5] | |
| `inverse` | x | `1/x` | x ∈ [0.5, 2.5] | ✓ |
| `product` | x1, x2 | `x1·x2` | both ∈ [−2, 2] | ✓ |
| `ratio` | x1, x2 | `x1/x2` | x1 ∈ [−2, 2], x2 ∈ [0.5, 2.5] | ✓ |
| `sqrt_law` | x | `√x` | x ∈ [0.25, 4] | ✓ |
| `exp_decay` | x | `e^(−x)` | x ∈ [0, 2] | ✓ |
| `log_law` | x | `ln x` | x ∈ [0.2, 4] | ✓ |
| `sine_law` | x | `sin x` | x ∈ [−π, π] | ✓ |
| `gaussian` | x | `e^(−x²)` | x ∈ [−2.5, 2.5] | |
| `logistic_law` | x | `1/(1+e^(−x))` | x ∈ [−4, 4] | |
| `tanh_law` | x | `tanh x` | x ∈ [−3, 3] | |
| `harmonic_sum` | x1, x2 | `1/x1 + 1/x2` | both ∈ [0.5, 2.5] | |
| `reciprocal_sum` | x1, x2 | `1/(x1+x2)` | both ∈ [0.5, 3] | |
| `pythagoras` | x1, x2 | `√(x1² + x2²)` | both ∈ [−3, 3] | |
| `kinetic_energy` | m, v | `½·m·v²` | m ∈ [0.5, 3], v ∈ [−3, 3] | |
| `coulomb` | q1, q2, r | `q1·q2 / r²` | all ∈ [0.5, 3] | |
| `ideal_gas` | n, T, V | `n·T / V` | all ∈ [0.5, 3] | |
| `spring_period` | m, k | `√(m/k)` | both ∈ [0.5, 3] | |
| `velocity_addition` | u, v | `(u + v)/(1 + u·v)` | both ∈ [−0.8, 0.8] | |
| `damped_oscillator` | x | `e^(−x)·sin x` | x ∈ [0, 4] | |
| `relativistic` | β | `1/√(1 − β²)` | β ∈ [−0.9, 0.9] | |
| `sine_sum` | x1, x2 | `sin(x1 + x2)` | both ∈ [−2, 2] | |
| `power_law` | x | `x^1.5` | x ∈ [0.25, 4] | |

`rcsr-bench --list` prints this table (with the exact formulas) from the code.

## 2. How the data is generated

Each problem fixes a **sample count and an input range**, and the generator is
deterministic:

* **Training**: 60 rows drawn uniformly from the ranges.
* **Test**: 200 *fresh* rows, drawn independently from the same ranges.
* **Noise**: Gaussian, added to the **training target only**, scaled by the
  standard deviation of the noiseless target. So `--noise 0.01` means 1 %
  relative noise and `0.0` means exact data. The test target is always
  **noiseless**.
* **Seed**: the point and noise streams are seeded from the problem name with a
  fixed constant, so the *same points and the same noise realisation* are used
  for every search seed. `--seed` drives the engine only. This makes runs
  comparable across seeds and exactly reproducible.

The engine sees the inputs in their **raw units** (no normalisation is applied by
the harness), so a recovered expression is directly the law in the satellite's
variables.

## 3. The recovery criterion

For each problem × noise run the engine returns a Pareto front (all
non-dominated error/complexity solutions). Every front member is evaluated on
the fresh **noiseless** test set, and `R² = 1 − SSE/SST` is computed against the
mean of the test target (`−inf` if any prediction is non-finite).

* **Selected solution**: the front member with the highest test `R²`. Ties go to
  the simplest expression, because the front is scanned in complexity order.
* **Recovered**: the selected test `R²` is at least the threshold
  (`--threshold`, default **0.99**).

In words: *the front contained an expression that explains at least 99 % of the
variance of the true law at points the search never saw.*

Selecting on the test set is deliberate and documented: the question is whether
the engine **offered** the law somewhere on its front, not whether it happened to
rank that law first on the noisy training data. The reported complexity, train
error (in the search metric) and test RMSE all belong to the selected solution.

Two choices keep the measurement honest:

* the test set is **fresh and noiseless**, so an interpolation of the training
  rows scores poorly on it;
* noise is added to the target only, so the engine cannot see the noise as an
  input it could fit.

## 4. Running the benchmark

```sh
cd my-version
cargo build --release

# the full sweep: 25 problems × 3 noise levels
./target/release/rcsr-bench --suite known-truth --noise 0.0,0.01,0.1

# the fast subset for CI (a few seconds)
./target/release/rcsr-bench --tiny

# machine-readable
./target/release/rcsr-bench --tiny --json

# the problem table
./target/release/rcsr-bench --list
```

Options: `--noise`, `--tiny`, `--generations`, `--islands`, `--pop`, `--seed`,
`--metric`, `--threshold`, `--time-budget`, `--json`. With `--tiny` the defaults
drop to 30 generations, 1 island and 80 individuals per island (full suite: 150
generations, 2 islands, 200 individuals). The search metric defaults to
`squared`; `--seed` reproduces a run exactly.

## 5. Recorded results

### 5.1 Tiny suite (CI)

```sh
cargo run --release --bin rcsr-bench -- --tiny --seed 1
```

Recorded on an 8-thread x86-64 Linux machine (Intel Core i7-3770K), engine seed
`1`, all three default noise levels:

| problem | cx | R² @ 0.00 | R² @ 0.01 | R² @ 0.10 |
|---|---|---|---|---|
| `line` | 5 | 1.000 | 1.000 | 1.000 |
| `quadratic` | 2 | 1.000 | 1.000 | 1.000 |
| `inverse` | 3 | 1.000 | 1.000 | 1.000 |
| `product` | 3 | 1.000 | 1.000 | 1.000 |
| `ratio` | 3 | 1.000 | 1.000 | 1.000 |
| `sqrt_law` | 2 | 1.000 | 1.000 | 1.000 |
| `exp_decay` | 3 | 1.000 | 1.000 | 1.000 |
| `log_law` | 2 | 1.000 | 1.000 | 1.000 |
| `sine_law` | 2 | 1.000 | 1.000 | 1.000 |

```
recovered 27/27 (100.0%)   median complexity 3
by noise level:
  0.000    9 runs    9 recovered (100.0%)  median complexity 3
  0.010    9 runs    9 recovered (100.0%)  median complexity 3
  0.100    9 runs    9 recovered (100.0%)  median complexity 3
```

The whole tiny sweep (9 problems × 3 noise levels) finishes in about **5 seconds**
on an idle 8-thread machine, which is why it is the CI subset. Engine seeds 7 and
42 give 27/27 (100 %) and 26/27 (96.3 %) at these settings — one 1 %-noise run
missed the threshold at seed 42 — so the tiny suite is a smoke test, not a
one-seed fluke.

Every selected solution is the true form: `square(x)`, `sqrt(x)`, `ln(x)`,
`sin(x)`, `exp(-x)`, `(1 / x)`, `(x1 * x2)`, `(x1 / x2)` and a line written as
`(1 + (x + x))`. The `line` complexity of 5 is not a failure — `x + x` costs one
node more than `2*x` but fits identically; the criterion is behaviour on new
data, not the printed shape.

### 5.2 Full suite

```sh
cargo run --release --bin rcsr-bench -- --suite known-truth --noise 0.0,0.01,0.1 --seed 1
```

All 25 problems at the three default noise levels with the full budget
(150 generations, 2 islands, 200 individuals). Recorded with engine seed `1` on
an 8-thread x86-64 machine; the sum of the per-run times was about 365 s
(≈ 6–7 min wall-clock, one run at a time).

```
recovered 72/75 (96.0%)   median complexity 3
by noise level:
  0.000   25 runs   24 recovered ( 96.0%)  median complexity 3
  0.010   25 runs   24 recovered ( 96.0%)  median complexity 3
  0.100   25 runs   24 recovered ( 96.0%)  median complexity 3
```

| problem | R² @ 0.00 | R² @ 0.01 | R² @ 0.10 |
|---|---|---|---|
| `line` | 1.000 | 1.000 | 1.000 |
| `quadratic` | 1.000 | 1.000 | 1.000 |
| `cubic` | 1.000 | 1.000 | 1.000 |
| `inverse` | 1.000 | 1.000 | 1.000 |
| `product` | 1.000 | 1.000 | 1.000 |
| `ratio` | 1.000 | 1.000 | 1.000 |
| `sqrt_law` | 1.000 | 1.000 | 1.000 |
| `exp_decay` | 1.000 | 1.000 | 1.000 |
| `log_law` | 1.000 | 1.000 | 1.000 |
| `sine_law` | 1.000 | 1.000 | 1.000 |
| `gaussian` | 1.000 | 1.000 | 1.000 |
| `logistic_law` | 1.000 | 1.000 | 1.000 |
| `tanh_law` | 1.000 | 1.000 | 1.000 |
| `harmonic_sum` | 1.000 | 1.000 | 0.999 |
| `reciprocal_sum` | 1.000 | 1.000 | 1.000 |
| `pythagoras` | 0.954 | 0.989 | 0.986 |
| `kinetic_energy` | 1.000 | 1.000 | 1.000 |
| `coulomb` | 1.000 | 1.000 | 1.000 |
| `ideal_gas` | 1.000 | 1.000 | 1.000 |
| `spring_period` | 1.000 | 1.000 | 1.000 |
| `velocity_addition` | 0.999 | 1.000 | 1.000 |
| `damped_oscillator` | 1.000 | 1.000 | 1.000 |
| `relativistic` | 1.000 | 1.000 | 0.999 |
| `sine_sum` | 1.000 | 1.000 | 1.000 |
| `power_law` | 1.000 | 1.000 | 1.000 |

24 of 25 problems are recovered at every noise level. The single miss is
`pythagoras` (`√(x1² + x2²)`): its best front member reaches R² = 0.954
(noiseless), 0.989 (1 %) and 0.986 (10 %), so the front came close but never
carried the exact two-term form within the budget. This is exactly the kind of
information the suite exists to produce — a place where the search, not the
noise, is the limiting factor.

## 6. Reproducing and limitations

* Reproduce a run by reusing the same `--seed`; the data and the noise
  realisation are fixed per problem, so only the search differs between seeds.
* Wall-clock times depend on the machine and on how many runs share the CPU;
  compare recovery rates and complexities, not absolute seconds.
* The suite is deliberately small and closed-form. It measures *recovery on
  synthetic noiseless-to-noisy data*, not performance on real, noisy,
  model-misspecified measurements. The overfitting tools in
  [`overfitting.md`](overfitting.md) apply there.
* Selecting the best front member on the test set is a permissive criterion
  (the front is large). It answers "was the law on the front?", which is the
  right question for a *generator*, and the recorded numbers should be read that
  way.
