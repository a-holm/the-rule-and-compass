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
| `device_interval_basis` | how intervals were placed: `basis` is `events` (CUDA event timestamps read back after the sync), `host_envelope` (fallback when events are unavailable), `mixed` or `none`, with the `events` and `host_envelope` counts |
| `stages.io_s` | checkpoint snapshot and write windows inside the search clock |
| `structural_zero_stages` | stages with no work inside a search run by design (`features`, `pnl`): enrichment and P&L are separate commands, so a zero there is structural, not a missing measurement |
| `outside_run` | windows before the search clock starts, outside the closure: `io_s` (table load) and `features_s` (fill, outlier, lag and split preparation); the JSON write happens after the report is serialised and is not measurable inside it |
| `duty_run` | `device_busy_s / wall_s`, units `fraction`, scope `run` |
| `eval_kernel_duty`, `fit_kernel_duty` | scoped sub-ratios (`evaluator_window`, `fit_stage`) |
| `device_split` | evaluator kernel/launch/copy/alloc seconds plus the fit kernel split, all in seconds |
| `fit.kernel_s`, `fit.fallback_s`, `fit.refit_s` | raw device kernel, measured exact-CPU fallback and finalist-refit seconds |
| `fit.admitted/device_fits/fallback_fits/fallback_by_cause` | fit counters and every labelled fallback cause |
| `fit.fallback_improved`, `fit.fallback_improved_by_cause` | fallbacks whose exact CPU refit lowered the exact error, in total and by cause; the share of `fallback_fits` is the refit yield |
| `eval_fallback.programs/seconds` | evaluator CPU-fallback work |
| `quality_per_gpu_hour` | decades per run wall-hour, with like-unit errors and full provenance |

### Phase ledger of `enrich`, `walkforward` and `ladder` (#284)

Each command ends its JSON report with a `ledger` object: the same keys as `gpu.ledger` (`wall_s`, `stages.*_s`, `stage_sum_s`, `unaccounted_s`, `closure`, `device_busy_s`, `device_busy_sum_s`, `device_overlap_s`, `device_intervals`, `device_interval_basis`, `duty_run`) plus `command` and the command's own `structural_zero_stages`. The wall clock is the command's own run wall. Stages are exclusive and sum to no more than `wall_s`.

| command | `io_s` | `features_s` / `pnl_s` | device intervals |
|---|---|---|---|
| `enrich` | input read, feature reads, GPU output staging writes, output write | `features_s`: context, GPU, CPU and cross-section producer work | the kernel seconds of each device region, placed in that region's host window (`host_envelope`) |
| `walkforward` | `.rcsb` row-group reads or the CSV table load | `pnl_s`: scoring, windows, controls, `--batch-pnl` | the `--batch-pnl` device tile seconds per seam (`host_envelope`) |
| `ladder` | panel, windows and fold load | `pnl_s`: the whole ladder | none |

`host_other_s` is the rest of the wall. `walkforward` keeps the ledger outside the reproduction hash (`report_sha256` hashes the report without it). `--device-pnl` does not time its device reduction, so it adds no interval.

`gpu.eval_kernel_duty` (formerly `gpu.duty_cycle`) is the evaluator-window kernel
fraction (scope `evaluator_window`, window `gpu.eval_window_s`). It is not the run-level number; use
`gpu.ledger.duty_run` for “how busy was the device during the run?”. A GPU run
answers `Σ stage_s + unaccounted_s = wall_s` within 1 %; a CPU run still reports
`"gpu": null`.


## 8. Hard and signal suites, multi-seed runs and paired comparison

`known-truth` is saturated at its criterion: at the default budget the front holds a member above R² 0.99 in 75 of 75 runs, and the seed-to-seed spread at a small budget is as large as the effect of a search change. The `hard` and `signal` suites, the oracle criterion, the time-to-hit fields, many seeds and `rcsr-bench compare` exist to rank a search change with a known error bar. `known-truth` is unchanged: `rcsr-bench --suite known-truth --seed 1 --cpu --json` gives the same `recovered`, `test_r2`, `complexity`, `train_error`, `test_rmse` and `expression` for all 75 runs as the harness before this section.

The suites live in `my-version/src/bench_suites.rs`, the statistics and the comparison in `my-version/src/bench_stats.rs`.

### 8.1 The `hard` suite

20 problems, 100 training rows, 500 fresh noiseless test rows, uniform inputs, the same noise model as `known-truth`. `truth_cx` is the node count of the truth tree under the default complexity model (`x^2` is one `square` node, `x^4` is `square(square(x))`); a unit test checks it against the tree. `rcsr-bench --list --suite hard` prints the table from the code.

| problem | truth | ranges | `truth_cx` |
|---|---|---|---:|
| `nguyen5` | `sin(x²)·cos(x) − 1` | x ∈ [−1, 1] | 8 |
| `nguyen6` | `sin(x) + sin(x + x²)` | x ∈ [−1, 1] | 8 |
| `nguyen7` | `ln(x + 1) + ln(x² + 1)` | x ∈ [0, 2] | 10 |
| `nguyen9` | `sin(x1) + sin(x2²)` | both ∈ [−1, 1] | 6 |
| `nguyen10` | `2·sin(x1)·cos(x2)` | both ∈ [−1, 1] | 7 |
| `nguyen12` | `x1⁴ − x1³ + 0.5·x2² − x2` | both ∈ [−1, 1] | 13 |
| `keijzer11` | `x1·x2 + sin((x1 − 1)(x2 − 1))` | both ∈ [−3, 3] | 12 |
| `keijzer14` | `8 / (2 + x1² + x2²)` | both ∈ [−3, 3] | 9 |
| `keijzer15` | `0.2·x1³ + 0.5·x2³ − x2 − x1` | both ∈ [−3, 3] | 13 |
| `vladislavleva1` | `exp(−(x1 − 1)²) / (1.2 + (x2 − 2.5)²)` | both ∈ [0.3, 4] | 13 |
| `pagie1` | `x1⁴/(1 + x1⁴) + x2⁴/(1 + x2⁴)` | both ∈ [−5, 5] | 19 |
| `korns1` | `1.57 + 24.3·x4` (inputs x1..x5) | all ∈ [−50, 50] | 5 |
| `korns4` | `−2.3 + 0.13·sin(x3)` (inputs x1..x5) | all ∈ [−50, 50] | 6 |
| `feynman_i6_2` | `exp(−0.5 (t/s)²) / (2.5066283·s)` | t, s ∈ [1, 3] | 11 |
| `feynman_i12_11` | `q·(E + B·v·sin(th))` | all ∈ [1, 5] | 10 |
| `feynman_i15_3x` | `(x − u·t) / sqrt(1 − u²/c²)` | x [5, 10], u [1, 2], t [2, 5], c [3, 10] | 14 |
| `feynman_i29_16` | `sqrt(x1² + x2² − 2·x1·x2·cos(th1 − th2))` | all ∈ [1, 5] | 17 |
| `freq_sine` | `1.5·sin(2.7·x + 0.4)` | x ∈ [−3, 3] | 8 |
| `rational_r1` | `(x + 1)³ / (x² − x + 1)` | x ∈ [−1, 1] | 11 |
| `damped_freq` | `exp(−0.4·x)·cos(3·x)` | x ∈ [0, 6] | 9 |

`korns1` and `korns4` carry four unused input columns; a problem lists them in `distractors`.

### 8.2 The `signal` suite

Three low-SNR regression tasks that mimic a long-only signal on crypto data. Each has 2000 training, 1000 validation and 2000 test rows. The target is `y = s + e` with the noiseless signal `s` and Student-t noise `e` with 4 degrees of freedom (`N(0,1) / sqrt(chi2_4 / 4)`). The noise scale is solved on the training rows so that the realised `var(s) / var(y)` equals the task's oracle R² (unit test `bench_signal_oracle_r2`). Three distractor columns `d1..d3 ~ N(0, 1)` are appended to every task. The suite has no noise sweep: `--noise` is ignored and the run line reports `"noise": null`.

| task | inputs | signal `s` | oracle R² | rows |
|---|---|---|---:|---|
| `sig_tanh` | r1 ~ N(0,1), vol = exp(0.5·N(0,1)), z ~ N(0,1) | `0.3·tanh(2·r1) − 0.2·z·vol` | 0.20 | i.i.d. |
| `sig_regime` | r1, vol | `if(1.2 − vol, 0.25·r1, −0.1·r1)` | 0.10 | i.i.d. |
| `sig_ar` | xf, xs AR(1) with phi 0.5 and 0.95 (unit marginal sd), vol | `0.15·(xf − xs) / vol` | 0.05 | time-ordered; train, validation and test are consecutive blocks |

The front member with the lowest error on the **validation** rows is selected, never one chosen on the test rows. `signal_r2` is its R² against the noiseless test signal, `signal_recovered` is `signal_r2 >= 0.8`, `selected_cx` its complexity and `distractors_used` the number of distractor columns in the expression. For these runs `test_r2`, `complexity` and `expression` describe the selected member and `recovered` applies `--threshold` to `signal_r2`.

### 8.3 Criteria and per-run fields

| field | meaning |
|---|---|
| `seed`, `suite`, `truth_cx`, `generations_run`, `stop_reason`, `backend` | run identity and how it ended; `backend` is what ran (`cpu` or `gpu`) |
| `oracle_r2` | test R² of `c0 + c1·truth` refit on the training rows with every truth constant free (50 LM iterations from the true constants) |
| `oracle_recovered` | some front member has `1 − R² <= 2 (1 − oracle_r2) + 1e-9`. At noise 0 this demands the law; with noise it demands what the true structure reaches on these rows. `null` for signal tasks |
| `exact_cx`, `excess_cx` | complexity of the simplest front member that meets the oracle criterion, and its difference to `truth_cx` |
| `front_hv` | hypervolume of the final front in the plane `(min(1, max(0, 1 − R²)), cost / max_complexity)` with reference point `(1, 1)`; higher is better, 1 means the exact law at near-zero cost |
| `hit_gen`, `hit_s`, `hit_evals` | first generation (1-based), elapsed search seconds and evaluated candidates (`generation · islands · pop`) at which the front held a member with test R² >= `--threshold`; for signal tasks the validation-selected member must reach `signal_r2 >= 0.8` |
| `oracle_hit_gen`, `oracle_hit_s`, `oracle_hit_evals` | the same for the oracle criterion |
| `signal_r2`, `signal_recovered`, `selected_cx`, `distractors_used` | signal tasks only |
| `fit_attempts`, `fit_gpu`, `fit_fallback` | device fit counters, `null` on a CPU run |
| `prune_s` | seconds spent in the front prune (#466), `0` with `--no-prune` |

The time-to-hit fields come from a reporting-only hook, `search::Config::front_observer`, called once per generation after the archive update. It does not change the trajectory (test `front_observer_does_not_change_the_archive`, and `front_observer_does_not_change_the_postfix_archive` for the flat arena path). The bench scores each new front member once, so `seconds` and `hit_s` include that scoring. A run stopped by `--time-budget` still reports its hits.

### 8.4 Running

```sh
cd my-version
rcsr-bench --suite hard --noise 0 --seeds 1-3 --cpu --ndjson > hard.ndjson
rcsr-bench --suite signal --seeds 1 --cpu --ndjson > signal.ndjson
rcsr-bench --suite all --problems nguyen9,sig_tanh --seeds 1,2 --generations 20 --json
```

| flag | effect |
|---|---|
| `--suite known-truth\|hard\|signal\|all` | suites to run; a comma list runs several |
| `--problems a,b` | run only these names; an unknown name exits 2 |
| `--seeds 1,2,3` or `--seeds 1-10` | engine seeds; `--seed S` stays a single seed |
| `--cpu` | `Config::gpu = false`, no environment variable; `--gpu` needs a device and exits 2 without one |
| `--fit-strategy exact\|auto\|linear\|varpro\|stream\|gpu\|default` | `exact` stays the bench default; `default` resolves the arm like the production CLI (`FitRequest::Unset`) |
| `--fit-iters`, `--ops`, `--max-complexity`, `--patience`, `--migration-interval`, `--restart-interval`, `--variation`, `--dispatch` | copied into `search::Config` |
| `--no-prune` | skip the front prune (#466) to measure its effect |
| `--threads N` | worker threads for fitting and scoring inside an island (default: all cores; results do not depend on it) |
| `--jobs N` | N CPU runs at once; each run keeps its seed, so results are identical, but `seconds` and `hit_s` are contended. Exits 2 with a device backend |
| `--ndjson` | one JSON line per finished run, flushed at once, then one summary line |
| `--json` | one JSON object, as before |

Every NDJSON line carries a `config` object: `generator` (the data generator and criteria version), the thresholds, `knobs` (every search setting) and `build` (version, CUDA feature, backend request, jobs, `git describe`).

**Chunks.** A run killed by a lock-hold timeout keeps every line it flushed. Run seeds as separate invocations, each under `timeout 580`, and merge the files in `compare` with a comma list. Seeds are independent, so the merged result equals one long invocation.

### 8.5 `rcsr-bench compare`

```sh
rcsr-bench compare --base a.ndjson[,a2.ndjson] --treat b.ndjson[,b2.ndjson] \
    [--criterion recovered|oracle_recovered|signal_recovered] [--json]
```

* Pairs runs by `(suite, problem, noise, seed)`. It exits 2 when the two sides do not hold the same set of keys, when a key repeats, or when the `config` objects differ in anything other than `knobs` and `build` (a different data generator or threshold cannot be compared by accident).
* Prints both rates with Wilson 95 % intervals, the discordant counts `b` (base only) and `c` (treat only) and the exact two-sided McNemar p-value `min(1, 2 P(X <= min(b, c)))`, `X ~ Binomial(b + c, 1/2)`.
* Prints the median paired ratio `treat.hit_s / base.hit_s` and `hit_evals` over pairs where both hit, the median paired `front_hv` difference, the summed `seconds` of both sides and a per-problem table (base hits, treat hits, wins, losses).
* `hit_evals` is exact and independent of machine load; `hit_s` is wall time. Compare time on an otherwise idle machine and prefer `hit_evals` when the load varies.

Unit tests pin the statistics: Wilson 7/10 is [0.397, 0.892], McNemar (0, 6) is 0.03125 and (3, 3) is 1.

### 8.6 Recorded baselines

Recorded on the 8-thread i7-3770K with `--features cuda` for the GPU rows, with other jobs running on the machine, so absolute seconds are inflated and noisy. CPU data recorded at `b1f6390`, GPU data at `e6431cc` (the commits differ only in the `--dispatch` knob). Raw NDJSON is not committed; each row lists its command.

| row | command (after `cd my-version`) | runs | recovered | oracle_recovered | median `hit_s` | summed seconds |
|---|---|---:|---|---|---:|---:|
| `known-truth`, seed 1, 3 noise levels, CPU | `rcsr-bench --suite known-truth --seed 1 --cpu --json` | 75 | 75/75 | | | 445.0 |
| `hard`, noise 0, seeds 1-3, CPU | `rcsr-bench --suite hard --noise 0 --seeds 1 --cpu --ndjson` and `--seeds 2-3` | 60 | 53/60 [0.778, 0.942] | 23/60 (38.3 %) [0.271, 0.510] | 0.33 s (n = 53) | 242.4 |
| `hard`, noise 0, seeds 1-3, GPU, default fit arm | `rcsr-bench --suite hard --noise 0 --seeds 1-3 --gpu --fit-strategy default --ndjson` | 60 | 50/60 [0.720, 0.907] | 19/60 (31.7 %) [0.213, 0.442] | 0.14 s (n = 50) | 62.3 |
| `signal`, seeds 1-2, CPU | `rcsr-bench --suite signal --seeds 1-2 --cpu --ndjson` | 6 | `signal_recovered` 6/6 | | 10.0 s (n = 6) | 406.7 |

The `hard` baseline `oracle_recovered` rate at noise 0 is 38.3 %, inside the 20 % to 80 % target band, so the default budget (150 generations, 2 islands, 200 individuals) stays. Per problem, the number of seeds (of 3) whose front holds the oracle-grade member on CPU: `damped_freq`, `freq_sine`, `korns1`, `nguyen9`, `nguyen10` 3; `feynman_i12_11`, `keijzer15`, `korns4` 2; `nguyen6`, `nguyen12` 1; the other nine problems 0. The `signal` runs select models with `signal_r2` between 0.89 and 1.00 (cost 7 to 24); `distractors_used` is 0 in four of six runs, 1 in one and 3 in one, so it separates runs where `signal_recovered` does not.

Paired comparison of the CPU baseline with the same seeds at `--generations 60` (a deliberately weaker treatment, to exercise the tool):

```
rcsr-bench compare --base hard-s1.ndjson,hard-s2-3.ndjson --treat hard-g60.ndjson --criterion oracle_recovered
base   23/60 (38.3%)  Wilson 95% [0.271, 0.510]
treat  17/60 (28.3%)  Wilson 95% [0.185, 0.408]
discordant: b (base only) = 6, c (treat only) = 0
McNemar exact two-sided p = 0.03125
median paired hit_evals ratio treat/base = 1.0000 over 46 pairs
```

The `hit_evals` ratio of exactly 1 shows that the shorter run follows the same trajectory as the first 60 generations of the longer one. Comparing the CPU baseline with the GPU default-arm run: `oracle_recovered` b = 6, c = 2, p = 0.289; `recovered` b = 3, c = 0, p = 0.25; median paired `hit_s` ratio 0.33.

**Full baselines.** The recorded rows are small seed sets. The full baselines, one invocation per chunk so that each fits a 10-minute lock hold (measured: `hard` about 70 to 90 s per seed, `known-truth` about 240 to 450 s per seed, `signal` about 200 s per seed):

```sh
# known-truth, seeds 1-5, CPU: one seed per invocation
for s in 1 2 3 4 5; do timeout 580 rcsr-bench --suite known-truth --seed $s --cpu --ndjson > kt-s$s.ndjson; done
# hard, seeds 1-10, CPU, noise 0: three seeds per invocation
for r in 1-3 4-6 7-9 10; do timeout 580 rcsr-bench --suite hard --noise 0 --seeds $r --cpu --ndjson > hard-$r.ndjson; done
# signal, seeds 1-10, CPU: two seeds per invocation
for r in 1-2 3-4 5-6 7-8 9-10; do timeout 580 rcsr-bench --suite signal --seeds $r --cpu --ndjson > signal-$r.ndjson; done
# hard, seeds 1-3, GPU, production fit arm
timeout 580 rcsr-bench --suite hard --noise 0 --seeds 1-3 --gpu --fit-strategy default --ndjson > hard-gpu.ndjson
```

Each chunk is one lock hold. They are not recorded here yet; the 10-seed `hard` rate decides whether the budget band holds.

### 8.7 Limits

* `signal_recovered` (R² >= 0.8 against the noiseless signal) is saturated on the recorded runs. `distractors_used`, `selected_cx` and `signal_r2` still vary.
* The oracle criterion is relative to the training rows. A front member can meet it with a different, equally good form; `exact_cx` and `excess_cx` show the cost.
* Wall-clock fields depend on machine load. `hit_evals` does not.
