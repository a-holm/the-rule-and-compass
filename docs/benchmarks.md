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

**These numbers are a recording, not a promise about the current tree.** They are
pinned to the commit and date named in each subsection. The search is
stochastic across seeds: a different `--seed` changes the trajectory, and a
later change to mutation, fitting or parent selection can move the counts even
at the same commit. Reproduce a row with the stated commit and seed, not by
assuming `main` still produces it.

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
(150 generations, 2 islands, 200 individuals). Recorded with engine seed `1` at
commit `c4c1d9d` (2026-10-20) on an 8-thread x86-64 machine; the 75 runs took
180.27 s of search time (wall-clock includes process overhead).

This replaces the earlier `d26f2fd` recording. `main` has moved since that
commit (data preparation, fitting, GPU scheduling and checkpoint work), so the
old numbers must not be read as a current-tree claim. The table below is the
fresh recording; it is not a claim that every later commit reproduces it.

```
recovered 74/75 (98.7%)   median complexity 3
by noise level:
  0.000   25 runs   25 recovered (100.0%)  median complexity 3
  0.010   25 runs   24 recovered (96.0%)   median complexity 3
  0.100   25 runs   25 recovered (100.0%)  median complexity 3
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
| `pythagoras` | 0.993 | 0.987 | 1.000 |
| `kinetic_energy` | 1.000 | 1.000 | 1.000 |
| `coulomb` | 1.000 | 1.000 | 1.000 |
| `ideal_gas` | 1.000 | 1.000 | 1.000 |
| `spring_period` | 1.000 | 1.000 | 1.000 |
| `velocity_addition` | 1.000 | 1.000 | 1.000 |
| `damped_oscillator` | 1.000 | 1.000 | 1.000 |
| `relativistic` | 1.000 | 1.000 | 0.999 |
| `sine_sum` | 1.000 | 1.000 | 1.000 |
| `power_law` | 1.000 | 1.000 | 1.000 |

24 of 25 problems are recovered at every noise level; the single miss is
`pythagoras` at 1 % noise. The three `pythagoras` rows, with the selected form
and complexity, are:

| noise | R² | complexity | selected form |
|---|---:|---:|---|
| 0.00 | 0.993 | 22 | `(6.229 - ((4.4857 - tan(((0.1617 - (-0.0856 * abs(x1))) - cos((1.0596 - cos(x1)))))) ^ gauss((0.2074 * x2))))` |
| 0.01 | 0.987 | 27 | `((3.1835 / (0.9929 ^ ((0.687 / (0.0181 + abs(x2))) ^ (-1.5532 + cos(x1))))) * ((0.482 / (0.785 ^ abs(x2))) ^ (1.121 + cos(x1))))` |
| 0.10 | 1.000 | 10 | `(((0.0218 + square(x1)) + abs(square(x2))) ^ 0.4982)` |

At 0 % and 10 % the front clears the 0.99 recovery threshold with an
approximation; at 1 % its best member (complexity 27) falls just below it
(R² = 0.9867). The exact two-term form is reachable — an earlier build found it
at complexity 7 — so this is a stochastic limit of the search at this budget,
not an impossibility.

### 5.3 Pareto-front parent feedback (#76)

From #76 the search draws 15 % of parents uniformly from the global Pareto
front (`--no-front-parents` restores the old trajectory exactly). The full
suite was run with the feature on and off for three seeds, same budget and same
machine, one run after another:

```sh
rcsr-bench --suite known-truth --noise 0.0,0.01,0.1 --seed <S> [--no-front-parents]
```

| seed | policy | recovered | median cx | wall (s) | `pythagoras` R² @ 0 / 0.01 / 0.10 |
|---:|---|---:|---:|---:|---|
| 1 | off | 72/75 | 3 | 271.67 | 0.987 / 0.721 / 0.987 |
| 1 | **on** | **74/75** | 3 | 287.52 | **0.993** / 0.987 / **1.000** |
| 7 | off | 72/75 | 4 | 262.65 | 0.955 / 0.962 / 0.957 |
| 7 | **on** | **75/75** | **3** | 278.43 | **0.998** / **1.000** / **1.000** |
| 42 | off | 72/75 | 3 | 285.97 | 0.942 / 0.766 / 0.952 |
| 42 | **on** | **74/75** | 3 | 289.21 | **0.996** / **1.000** / 0.981 |

The gain is consistent across seeds: **+2, +3 and +2** recovered, entirely on
`pythagoras`, which is recovered at 0 % noise for all three seeds (0.993, 0.998,
0.996) and at every noise level for seed 7. No problem that was recovered with
the feature off is lost with it on, and median complexity does not rise (seed 7
falls from 4 to 3). Wall-clock is 1–6 % higher, which is within the
run-to-run noise seen on this machine; it is not a speed claim.

At a smaller budget (`--generations 60 --pop 100 --islands 1 --seed 1`) the
total is **neutral**: 71/75 both ways, median complexity 3 both ways, 48.70 s
off versus 46.68 s on. The composition differs. Without feedback all 25
noiseless problems are recovered and `pythagoras` is exact (R² 1.000); with
feedback 23 are recovered at 0 % but 24 at each noise level, and `pythagoras`
is not recovered at any noise level (0.961 / 0.651 / 0.962 versus
1.000 / 0.833 / 0.675 off). So the full-budget gain does not transfer to this
smaller budget; the feature is kept on because the recorded full suite improves
consistently across three seeds, but the budget sensitivity is real.

Bloat is bounded by the existing machinery: the child of a front parent goes
through `simplify`, the `max_complexity` check (before and after constant
fitting) and structural de-duplication, and only the tree is taken — the front
member's stored error is never used as fitness. The median complexity numbers
above confirm there is no size drift at this budget.

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

## 7. Timing schema (`--json`, `--ndjson`, reports)

Every search run reports a non-overlapping wall-clock ledger in a `timing`
object (and text reports print a `# time:` line plus a `## Time` section):

| field | covers |
|---|---|
| `prep_seconds` | argument parsing, loading, splitting and data preparation |
| `search_seconds` | the evolutionary search only |
| `baselines_seconds` | the mean, OLS and k-NN null baselines |
| `diagnostics_seconds` | Pareto-row diagnostics, bootstrap, stability, CV, null control |
| `report_seconds` | building the Markdown report |
| `total_seconds` | measured wall from the start of the command to the end of the phases |

The phases are disjoint and are checked against `total_seconds` (within 1% in
the test on a real CLI run), so a large baseline or bootstrap cannot hide
outside the numbers. `elapsed_seconds` is **kept for compatibility and means
`search_seconds`**; use `total_seconds` for the cost of the whole run. The same
fields appear in the NDJSON `done` event.

Multi-output runs (`--target y1,y2`) use the same ledger in text, `--json` and
the NDJSON `done` event. They do not run the single-target baseline or
diagnostics phases, so `baselines_seconds` and `diagnostics_seconds` are
reported as `0`, not omitted.

The k-NN baseline reports its own sampling fields (`reference_rows`,
`query_rows`, `capped`, `sem`, `seconds`); on an uncapped run
`baselines_seconds` is the magnitude of that exact O(N²·d) work, not zero.

### GPU run ledger (`gpu.ledger`, #284)

`timing` is the process ledger. `gpu.ledger` is the run-scoped device ledger,
with one wall clock and exclusive main-thread stages:

| field | meaning |
|---|---|
| `wall_s` | search-phase run wall measured by the ledger clock |
| `stages.{vary,eval,fit,select,archive,features,pnl,io,host_other}_s` | exclusive stage seconds; `fit` is partitioned out of `eval` exactly once |
| `stage_sum_s`, `unaccounted_s`, `closure` | `Σ stage_s + unaccounted_s = wall_s`; `closure` is the `{value,numerator,denominator,units,scope}` residual, acceptance `< 1 %` |
| `device_busy_s` | **union** of device intervals across streams, never their sum |
| `device_busy_sum_s`, `device_overlap_s`, `device_intervals` | the naive sum, the double count it would introduce, and the interval count |
| `duty_run` | `device_busy_s / wall_s`, units `fraction`, scope `run` |
| `eval_kernel_duty`, `fit_kernel_duty` | scoped sub-ratios (`evaluator_window`, `fit_stage`) |
| `device_split` | evaluator kernel/launch/copy/alloc seconds plus the fit kernel split, all in seconds |
| `fit.kernel_s`, `fit.fallback_s`, `fit.refit_s` | raw device kernel, measured exact-CPU fallback and finalist-refit seconds |
| `fit.admitted/device_fits/fallback_fits/fallback_by_cause` | fit counters and every labelled fallback cause |
| `eval_fallback.programs/seconds` | evaluator CPU-fallback work |
| `quality_per_gpu_hour` | decades per run wall-hour, with like-unit errors and full provenance |

`gpu.eval_kernel_duty` (formerly `gpu.duty_cycle`) is the evaluator-window kernel
fraction (scope `evaluator_window`, window `gpu.eval_window_s`). It is not the run-level number; use
`gpu.ledger.duty_run` for “how busy was the device during the run?”. A GPU run
answers `Σ stage_s + unaccounted_s = wall_s` within 1 %; a CPU run still reports
`"gpu": null`.

