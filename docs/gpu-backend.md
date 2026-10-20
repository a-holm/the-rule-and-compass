# The optional GPU backend

The search's hot loop is *score P candidate expressions against N rows of data*.
Those are P × N independent evaluations, which is exactly the shape a GPU likes.
This document describes the backend in `src/cuda/eval.cu` + `src/gpu.rs`, how to
build and run it, what it measured on the available hardware, and how the same
design maps onto an NVIDIA DGX Spark.

The backend is **preferred automatically**: a binary built with
`--features cuda` probes the device once at startup and uses the GPU when one is
available; a build without the feature, or a machine with no usable device,
falls back to the CPU with a recorded reason. The default build remains pure
`std` with no dependencies and no CUDA requirement.

## Backend identity and resumes

Checkpoints record `backend cpu|gpu` (#140). `--resume` refuses a CPU/GPU
change with exit code `3` and an error naming both backends; pass
`--allow-backend-change` to override it deliberately. The override is printed on
stderr and recorded in `--run-manifest` under `resume.backend_change`. This is
the fix for the observed incident where `resume_10h.py` omitted `--gpu` and
silently continued a GPU-selected population on the CPU. Always state `--gpu`
or `--cpu` explicitly in a launcher, even though CPU is the default.

## Build and run

```sh
cd my-version

# default: CPU only, no CUDA needed
cargo build --release

# with the GPU evaluator (RTX 3060 / Ampere)
RCSR_CUDA_ARCH=sm_86 cargo build --release --features cuda

# benchmark GPU vs the reference CPU evaluator
./target/release/rcsr-gpu-bench --data examples/sine.csv --target y --vars x \
    --batch 8000 --depth 5 --seed 5

# opt in to GPU-guided search when a device is present (--gpu is required)
# no flag: a CUDA build with a device uses the GPU automatically
./target/release/rcsr --data examples/sine.csv --target y --vars x \
    --generations 200 --pop 200 --islands 4 --seed 7 --metric squared --gpu

# assert that the GPU must exist (exit 2 when it does not)
./target/release/rcsr ... --gpu

# force the CPU evaluator (alias: --cpu), or simulate a missing device
./target/release/rcsr ... --gpu --cpu
RCSR_GPU=0 ./target/release/rcsr ... --gpu
```

`RCSR_CUDA_ARCH` is validated before `nvcc` runs. A malformed value, or a
toolkit that is too old for the requested target, produces a short,
actionable error naming the target, the detected toolkit, and the minimum
supported CUDA version. For example, on a CUDA 12.0 machine:

```text
error: RCSR_CUDA_ARCH=sm_121 needs CUDA 12.8 or newer, but `nvcc` is CUDA 12.0.
error: sm_121 is the GB10 in the DGX Spark; its DGX OS images ship CUDA 13.0.
error: Install a newer toolkit and point RCSR_NVCC at it, for example:
error:   RCSR_NVCC=/usr/local/cuda-12.8/bin/nvcc RCSR_CUDA_ARCH=sm_121 cargo build --release --features cuda
error: or build for the GPU in this machine, for example:
error:   RCSR_CUDA_ARCH=sm_86 cargo build --release --features cuda
```

Environment overrides for the build:

| Variable | Meaning | Default |
|---|---|---|
| `RCSR_CUDA_ARCH` | target architecture passed to `nvcc` as `-arch=` | `sm_86` (Ampere) |
| `RCSR_NVCC` | `nvcc` binary to use | `nvcc` |
| `RCSR_GPU_UNIFIED` | `1`/`true`/`yes`/`on` selects `cudaMallocManaged` for every buffer; `0`/`false`/anything else selects plain device memory | `1` for `sm_121`, `0` otherwise |

Minimum toolkit per target, as used by the build-script check:

| Target | Minimum CUDA | Notes |
|---|---|---|
| `sm_86` | 11.1 | RTX 3060 / Ampere; verified here with CUDA 12.0 |
| `sm_90` / `sm_90a` | 11.8 / 12.0 | Hopper |
| `sm_100` | 12.8 | datacentre Blackwell |
| `sm_121` | 12.8 | GB10 / DGX Spark; DGX OS ships CUDA 13.x |


## How it works

**Bytecode.** Each expression is compiled once into a compact postfix program of
32-bit words: an opcode, plus an operand for `const` (the float bits) and
`var` (the variable index). This is the same representation that makes the CPU
side cache-friendly, and it is trivial to send to the device.

**One block per expression.** The kernel launches with `grid.x = P` (one block
per candidate). Every thread in a block executes the *same* instruction stream;
threads differ only in which row they are processing. Program structure
therefore causes **no warp divergence** — the classic reason tree-walking GP on
GPUs is slow.

**Threads across rows.** Each thread walks rows `tid, tid+B, tid+2B, …`, keeping
its operand stack in local memory. The block's sum of squared errors is reduced
in shared memory and written once per expression.

**Branchless operators, matched semantics.** Division, `log`, `sqrt`, `tan` and
`pow` use the *same* domain guards as the host evaluator, expressed as predicated
selects rather than branches. Anything non-finite, or larger than a shared
magnitude limit of 1e30, counts as a hard failure on **both** sides — so the
GPU and CPU agree on which candidates are hopeless.

**Native hard operators (#131).** `floor`, `ceil`, `round`, `min`, `max`, `if`,
`sign`, the comparisons, `and`/`or`/`xor`/`not`, `mod`, `atan2`, `erf` and `erfc`
compile to native opcodes instead of falling back to the serial CPU evaluator.
Both kernels — the packed `eval_kernel` and the fixed-slot `eval_pool_kernel`
used by the search — call one `rc_exec_row` device function, so an operator has
exactly one implementation and cannot silently differ between the pipeline and
pool paths. `floor`/`ceil` are one `FRND`; `round` is `roundf` (ties away from
zero, matching Rust — not `cvt.rni`, which is ties-to-even); `min`/`max` guard
NaN explicitly so either NaN operand yields NaN like `expr.rs`; `if` evaluates
both arms but a true select discards the unselected one; `mod` uses the host's
floored `a - b*floor(a/b)` with the `|b|<1e-12` and finite-quotient guards;
`atan2`, `erf` and `erfc` call libdevice. `erf(-0)` is normalised to `+0` to
match the host.

Every tree reaches the device through `gpu::compile_program`, the one guarded
entry point: it applies the #121 `lower()` rewrites, rejects a postfix program
that would need more than 56 operand-stack slots, and returns the opcodes that
were actually emitted. `Population::add` (pipeline path) and the fixed-slot
batch (synchronous path) both use it, so they agree on `asinh`/`acosh`/`atanh`
and on the stack limit.

**Exactness limits of the native opcodes (f32).** `floor`, `ceil`, `round`,
`sign`, `min`, `max`, `if`, comparisons and logicals are exact on the f32 grid
(0 ULP); `mod` is asserted to 1 ULP, `atan2` to 3 ULP, `erf` to 2 ULP and
`erfc` to 4 ULP against the f64 CPU reference
(`per_opcode_ulp_bounds_against_the_f64_reference`). These bounds are
relative-ULP claims for the normal f32
range. Outside it the contract is different: `erff` saturates to 1 at
`|x| >= 3.9196`, `erfcf` reaches the subnormal boundary around `x = 9.19` and
returns zero at `x = 10.05`, so a relative-ULP bound is meaningless there;
`native_special_values_are_exact` asserts those cases explicitly: at
`x = 10.05` libdevice returns the smallest subnormal (1.4e-45) or zero, not a
relative-accurate tail. A second limit belongs to `mod`: its quotient is f32,
so above `|a/b| = 2^25` the fractional part of the quotient is not representable
and the result is not the f64 remainder; a range correction keeps it inside
`[0,b)` (or `(b,0]` for negative `b`), and the fixture checks that documented
behaviour instead of a ULP bound. Below `2^25` (which covers the `8 388 608 /
0.7` case) the correction makes `mod` match the f64 reference to 1 ULP. `mod`
returns NaN for a non-finite operand, `|b| < 1e-12` or a non-finite f32
quotient, as the host does, and `atan2` returns NaN if either operand is NaN.
Discontinuous
operators (`floor`, `ceil`, `round`, `mod`, `sign`, comparisons, `if`) can jump
by O(1) on a 1-ULP input change, so they are never approximated in the tree and
the final archive is always re-scored by the CPU f64 evaluator. The accumulation
itself is f32 (see the 200 000-row caveat below), so the f32 fitness is a
search signal, not a reported number.

Static kernel cost measured on `sm_86` (CUDA 12.0, `-O3`), before and after this
change:

| kernel | SASS before | SASS after | registers | stack |
|---|---:|---:|---:|---:|
| `eval_kernel` (packed / pipeline) | 1 312 | 1 336 (+1.8%) | 27 → 27 | 288 B → 288 B |
| `eval_pool_kernel` (fixed slots / synchronous) | 1 008 | 1 336 (+32.6%) | 28 → 30 | 288 B → 288 B |
| `predict_kernel` (test fixtures only) | — | 1 296 | 26 | 288 B |

The pool kernel grew because it previously **did not implement** the #131
opcodes at all (it fell through to `r = 0.0f`); the growth is the cost of the
operators it now shares with the packed kernel, not a regression in a working
path. No kernel's stack frame increased. A CUDA-guided run reports the opcode
and fallback counts for each generation under `gpu.generations` in `--json`,
aggregated from the emitted programs of both the pipeline batches and the
synchronous fallback batch; a CPU run's JSON is unchanged. Operators without an
opcode (`asinh`, `acosh`, `atanh`, constraints, aggregates) still take the CPU
fallback, and the final f64 CPU rescore remains the exactness guarantee for
reported results.

**CPU fallback pool (#121).** Unsupported programs are not scored on the
scoring thread. Each generation collects them into jobs and hands them to a
persistent worker pool created with the GPU search (`RCSR_CPU_FALLBACK_THREADS`,
default `min(cores, 8)`); the `sync_channel` queue holds at most one chunk per
worker, so a burst cannot grow memory without bound. Results come back tagged
with their absolute candidate index and are merged in candidate order, so the
fitness bits and the archive are identical to the serial walk for any worker
count (`search::tests::cpu_fallback_is_identical_for_one_and_many_threads`, and
the mixed supported/unsupported integration test). `gpu.generations[]` reports
`programs`, `native`, `lowered`, `fallbacks`/`unsupported_programs`,
`fallback_seconds` and `fallback_workers`; `gpu.totals` aggregates the run.

**Static and device stack guards (#121).** The lowering pass can expand a tree,
so the host computes the postfix peak stack after lowering and rejects anything
above 56 slots before a word is emitted; that tree goes to the CPU pool. The
device is still guarded independently: every push checks `RC_MAX_STACK` (64) and
every operator checks its operand depth, returning NaN (a hard failure) instead
of writing past the local stack if a malformed or future program ever slips
through. `gpu::deep_lowered_tree_is_rejected_before_emission` and
`gpu::device_stack_guard_rejects_an_over_deep_program` pin both halves.

**FP32 throughout.** On the RTX 3060 (Ampere) and on the GB10 in the DGX Spark,
FP64 runs at **1/64 of FP32**, so an FP64 kernel would throw away the whole
advantage. FP64 is reserved for re-checking the final surviving equations on the
host.

**Resident data.** `RcCtx` uploads the dataset once; subsequent batches of
candidates only upload the (small) bytecode and download P errors.

**Whole-population batching.** `src/gpu.rs` has a `Population` type that
compiles many trees into one byte string plus an offsets table, and
`GpuEvaluator::eval_population` scores all of them in **one call and one kernel
launch**, returning one SSE per program. `rcsr-gpu-bench` uses this path and
prints the population size, so the capability is measured rather than assumed;
`cargo test --features cuda` checks that a whole-population batch returns the
same SSEs as scoring each program on its own. Since issue #26 the search uses
this path for every generation (see below).

**Unified memory.** With `RCSR_GPU_UNIFIED=1` (the default for `sm_121`) every
context buffer is allocated with `cudaMallocManaged`: the host writes into it
directly, the device migrates pages on demand, and there is no explicit
host→device copy for the dataset. That is what lets a 128 GB GB10 hold a
dataset larger than device VRAM. On a discrete GPU with its own VRAM the
managed path is slower for these short, repeated launches, so `sm_86` defaults
to plain `cudaMalloc` plus one copy at context creation. The benchmark prints
`memory : device` or `memory : unified` so every recorded run says which path
it measured.


## Measured results

Machine: Intel Core i7-3770K (4C/8T, AVX only) + NVIDIA GeForce RTX 3060
(`sm_86`, 28 SMs, 12 GB), CUDA 12.0. Every run below scores a whole
`Population` in one launch and uses plain device memory (the `sm_86` default).
"Batch" is the number of compiled expressions; the CPU reference is the
engine's own interpreted `f64` tree walk (`cpu_sse`), timed once.

| Dataset | Rows | Batch | GPU batch | CPU batch | Speed-up | Agreement (median / p99 / within 1e-3) |
|---|---|---|---|---|---|---|
| `examples/sine.csv` | 60 | 8 000 | 0.178 ms | 35.8 ms | **201×** | 2.9e-8 / 2.3e-4 / 99.3% |
| two-variable (y = sin x₁ + x₂²) | 400 | 8 000 | 0.360 ms | 220.2 ms | **612×** | 3.3e-8 / 3.5e-4 / 99.2% |
| large two-variable (200 000 rows) | 200 000 | 500 | 7.290 ms | 12 234 ms | **1 678×** | 3.8e-8 / 1.6e-1 / 98.0% |

`Agreement` is the relative difference between the f32 kernel and the f64
reference, over candidates that both sides scored as ordinary finite results.
A median of ~3 × 10⁻⁸ is f32 epsilon: the kernel is computing the same thing.
Roughly 30% of candidates are *hard failures* on both sides (random
`exp`/`cosh`/`cube` trees exceed the magnitude limit) and are excluded from the
agreement statistic, not from the search.

Three honest caveats:

* The CPU column is a **reference interpreter in `f64`**, not the engine's
  fastest CPU path. A column-oriented SIMD `f64` evaluator would narrow the gap;
  the fair summary is "hundreds to ~1 700 times faster than a tree walk", not
  "faster than the whole engine".
* The 200 000-row run accumulates a large sum in `f32`, so its p99 relative
  error is 0.16 even though the median is still f32 epsilon: 98% of programs
  agree within 1e-3. Near-tied candidates on very large datasets should be
  re-checked in FP64 before the Pareto front is reported.
* CPU wall-clock on this 2012 machine varies by tens of percent with background
  load (the same sine CPU batch took 35.8 ms in one run and 47.5 ms in another);
  the GPU numbers are CUDA-event kernel times and are stable.

The kernel time grows from 0.178 ms (60 rows) to 7.29 ms (200 000 rows), so on
short datasets it is still **latency-bound, not throughput-bound**: with only
32–256 threads per block and a per-thread local-memory stack, occupancy is low.
At 500 programs × 200 000 rows it reaches ~13 × 10⁹ evaluations/s, still a small
fraction of the card's FP32 peak, so there is headroom left.

### Device memory versus unified memory (same RTX 3060)

| Dataset | Batch | Device (ms) | Unified (ms) | Unified overhead |
|---|---|---|---|---|
| `examples/sine.csv` | 8 000 | 0.178 | 1.496 | 8.4× slower |
| large two-variable | 500 | 7.290 | 8.243 | 1.13× slower |

Managed memory works correctly on the RTX 3060 — the same tests and agreement
checks pass — but on a discrete GPU the page migrations for the small
per-launch buffers dominate a short kernel, which is why `sm_86` keeps the
plain-device-memory path. On a GB10 with coherent unified memory there is no
PCIe migration and the trade-off is expected to invert, but that is **not
measured here**.


## Inside the search loop (issue #26)

When the crate is built with `--features cuda`, `run_search` probes for a
device once at startup. If a device answers, it creates one resident
`GpuSearch` context for the full training sample. Library callers get `gpu = true`
by default in a CUDA build (`Config::default()` follows `cfg!(feature = "cuda")`),
and `false` otherwise; the CLI resolves the same preference through the probe.
Each generation then runs as:

1. the island threads generate, mutate, de-duplicate and simplify candidates,
   and fit their constants with the existing **CPU** Levenberg–Marquardt code;
2. every candidate still pending is compiled into one `Population` and scored
   with a single `eval_population` call — one kernel launch for the whole
   generation, all islands included;
3. selection, migration and restart decisions consume the returned fitness.

The GPU never produces a number a user reads. The kernel sums squared errors;
the search divides that by the row count and uses it as a mean-squared fitness
(exact for `--metric squared`, a ranking proxy for the other plain regression
metrics). At the end `full_data_archive` re-scores the surviving trees on the
full training sample with the CPU evaluator in `f64`, and the reported
train/test errors, Pareto front and convergence statistics come from that
rebuild. Constants stay on the CPU and in `f64` throughout.

The GPU path is used for single-output, unweighted, full-sample runs with a
plain regression metric. Multi-output (`--target y1,y2`), `--weight`, the
implicit metrics, classification metrics, AIC/AICc/BIC and
`--predictor-samples` keep the CPU evaluator, because they need more than one
prediction per row or a different set of rows every generation; the reason is
printed in the one-line note.

Fallback and the flags:

* With neither flag, a CUDA build with a usable device selects the GPU. A build
  without the feature prints
  `# gpu: unavailable (built without --features cuda); using the CPU`; a CUDA
  build with no usable device prints the same line with the probe's reason.
* `--gpu` is an **assertion**: it exits 2 with a clear error when the build or
  the device cannot provide a GPU. It no longer silently falls back.
* `--no-gpu` (alias `--cpu`) forces the CPU. This is the reproducibility escape
  hatch: it is silent and yields the byte-identical CPU trajectory.
* `RCSR_GPU=0` makes a CUDA build behave as if no device were present, which is
  how the fallback test runs on a machine that has one.
* Any CUDA error during a batch prints the same `# gpu: unavailable (...)`
  note, re-scores the pending generation on the CPU and disables the GPU for the
  rest of the run. There is no panic and no stale result.

### Reported backend and the startup summary

The probe runs once, before the dataset is uploaded, and the effective backend
is recorded from what ran rather than inferred from the flags. Non-quiet runs
print one line:

```
# resource: backend=gpu device="NVIDIA GeForce RTX 3060 (sm_86, 28 SMs, 12288 MB)" threads=8 rows=2048 cols=3
# resource: backend=cpu device=- threads=8 rows=2048 cols=3 (no CUDA device found)
```

`rows`/`cols` are the training table. `threads` is the machine's available
parallelism; the current search loop uses one thread per island, and a
`--threads` control and the #132 baseline cap are separate workstreams.

`--json` carries `"backend": "gpu"|"cpu"`, `"device": <name>|null` and
`"backend_reason": <why the CPU ran>|null`; NDJSON emits one
`{"event":"backend",...}` object before the results. The run manifest from
#140 will carry the same fields when it lands; until then JSON is the record.

A GPU-guided run also carries `gpu.generations`, one object per scoring
generation, aggregated from the programs actually emitted by the guarded
compiler (pipeline batches and the synchronous fallback batch share a
generation key):

```json
"gpu": {"generations": [
  {"generation": 0, "programs": 200, "fallbacks": 0,
   "opcodes": {"OP_ADD": 431, "OP_ERF": 12, "OP_FLOOR": 7, "OP_MUL": 260}}
]}
```

`programs` counts trees uploaded to the device, `fallbacks` counts candidates
in the same batch that the guarded compiler rejected, and `opcodes` counts
operator words in the uploaded programs. The counters describe what ran, not
what the operator set allowed.

### Exactness

Cross-backend runs are **not bit-identical by construction**. The GPU scores a
generation in `f32` and that fitness can change selection, so a CUDA default run
may reach a different (often algebraically equivalent) form than `--cpu`. Every
number a user reads — train/test error, constants, diagnostics and the Pareto
front — is still rebuilt from the surviving trees on the full training sample
with the exact CPU `f64` path. Use `--cpu` when a byte-identical reproduction
across machines or binary builds is required.

The failure marker is part of the C API: `rc_eval` returns a negative time when
`cudaGetLastError` reports a launch or execution failure, and `eval_batch`
returns an empty result when the program upload fails.

### Overlapping generation with scoring (issue #119)

Steps 1 and 2 above share no data except the candidate trees: variation,
simplification, fitting and de-duplication are CPU work, and a candidate's SSE
is a function of its tree alone. Since #119 the main thread therefore drains a
bounded queue while the island threads keep producing, instead of waiting for
every island before compiling the whole population:

* each island thread offers a candidate to a `SyncSender` immediately after it
  is fitted and pushed into the next population (its own FIFO order);
* the main thread takes batches of `pop / depth` candidates from the queue, so
  one batch is always being scored while later ones are still being generated
  and fitted;
* selection, migration, restart and the archive still happen at the generation
  barrier, after every result has been received.

`--pipeline-depth N` (default 2) sets the queue depth; `0` restores the
synchronous path. The queue holds at most `N` batches, plus at most one batch in
the scorer's hands, so live candidate trees are bounded by
`(N + 1) * ceil(pop / N)` — independent of the number of generations. A scoring
error stops further GPU work, keeps draining (so no producer can block), and
re-scores every still-pending candidate on the CPU in `f64`.

The result mapping is fixed by `(island, slot)`; the arrival order of concurrent
island threads is not part of the result. `tests/gpu_loop.rs` pins the whole
archive and every `f64` test-error bit against `pipeline_depth = 0`, and the two
queue unit tests force a slow consumer to show FIFO ordering and the
`capacity + batch` memory bound. A CPU-only run constructs no queue, so any
depth is an exact no-op.

Measured on the RTX 3060 (`sm_86`, CUDA 12.0), shared host with another GPU
process active, alternating depths:

| workload | depth 0 | depth 2 | speed-up |
|---|---:|---:|---:|
| `d5k.csv`, 5 000 rows, pop 400, 20 generations, `--fit-iters 8` | 106.6 s wall, 95.1 s user | 100.2 s wall, 93.0 s user | **1.06× wall, 1.02× user** |
| `d5k.csv`, same but `--no-fit` (evaluator-bound), 30 generations | 6.10 s wall | 5.66 s wall (depth 4) | **1.08×** |
| `d200k.csv`, 200 000 rows, pop 400, `--no-fit`, 30 generations | 167.6 s wall, 150.2 s user | 153.0 s wall, 144.2 s user (depth 4) | **1.09× wall, 1.04× user** |

The archives are byte-identical between depths (single- and two-island runs),
including the exact `f64` test errors. With the default fitter the GPU batch is
only about **2–3 %** of wall time on `d5k` (a 30-generation `--no-fit` run takes
6.1 s against 101 s for 20 fitted generations), so Amdahl caps any overlap of
scoring alone at about 1.03×. The **≥1.1× end-to-end target in #119 is therefore
not met on this workload**; the gain is 1.08–1.09× only when evaluation, not
constant fitting, dominates. Closing the gap needs work *inside* the CPU
generation (fitting on the SoA layout, or the batched GPU LM), not more overlap
against a 2 % GPU phase. The batched LM kernel is #169; #116 supplies the
`BatchedFitter` integration point and the exact `f64` verifier in
`src/fit/gpu.rs`, so a GPU proposal is accepted only when it is finite, has the
right shape and scores no worse than the starting constants, and every
rejection is refit on the CPU. See
[`optimisation.md`](optimisation.md#batched-gpu-fitting-169).

### Measured end to end

Machine and build: the same RTX 3060 (`sm_86`, CUDA 12.0), plain device memory.
Searches are whole `rcsr` runs on the same table and the same
`--generations 200 --pop 200 --islands 4 --seed 7 --metric squared`; every cell
is three interleaved CPU/GPU runs and reports **min / median** wall-clock
seconds. The host was shared with other builds during the measurement (load
average 9–15), so absolute times move by tens of percent; each pair ran
back-to-back on the same host.

| Dataset | Rows | Generations run | CPU wall (s) | GPU wall (s) | Speed-up (min) |
|---|---|---|---|---|---|
| `examples/sine.csv` (`--ops add,sub,mul,sin`) | 60 | 41 (early stop) | 1.66 / 2.37 | 1.89 / 2.02 | **0.88×** |
| two-variable `y = sin x₁ + x₂²` (`--ops add,sub,mul,sin,square`) | 400 | 58 (early stop) | 18.70 / 18.93 | 19.05 / 19.61 | **0.98×** |

Both backends recovered the known law (the GPU-selected forms are algebraically
equivalent: the sine phase differs by a multiple of 2π, the square is scaled).
The reported best errors agree to **≤ 5.7 × 10⁻¹⁹ absolute**; the relative
difference is ~3 × 10⁻⁶ only because both errors are at the f64 noise floor
(~1.7 × 10⁻¹³ and ~5.5 × 10⁻¹⁹). `tests/gpu_loop.rs` additionally asserts the
recovered predictions agree to 1e-12 and the reported train/test errors to
1e-9; on its fixture the observed difference is exactly 0.

The honest conclusion: **at this budget the end-to-end speed-up is zero to
slightly negative**. The GPU only replaces the tree evaluation used for
fitness; the CPU Levenberg–Marquardt iterations that fit each candidate's
constants dominate (roughly 7× more arithmetic than one fitness evaluation),
and on 60–400 rows the per-generation upload, launch and event sync plus the
final `full_data_archive` rebuild are a larger fraction than the evaluation
work saved. Raw kernel throughput (hundreds of times the tree walk) does not
translate into a whole-search win while fitting stays CPU-side.

Taking the fitting out isolates the evaluation and shows the wiring is doing
what it should (`--no-fit --patience 0`, two runs each):

| Dataset | Rows | Gens | CPU wall (s) | GPU wall (s) | Speed-up |
|---|---|---|---|---|---|
| two-variable | 400 | 100 | 1.41 / 1.42 | 0.77 / 0.91 | **1.7–1.8×** |
| two-variable | 5 000 | 100 | 14.02 / 14.38 | 1.93 / 1.96 | **7.2–7.5×** |

So the GPU path pays off exactly where the issue says it should — when
evaluation, rather than constant fitting, is the bottleneck. Making the fit
cheaper or moving it to the GPU is a separate workstream; batch size and launch
overhead are addressed by the fixed-capacity pool below.

### Fixed-capacity genome pool and flat postfix variation (#122)

The GPU search keeps the historical owned-tree loop for fitting runs and for
configurations the flat engine cannot reproduce exactly. A second engine removes
the per-child tree work from the fit-free, large-population path:

* **One persistent program arena.** `genome::GenomePool` is a fixed-capacity,
  double-buffered postfix arena. Every genome occupies exactly
  `width_for(max_complexity) = 2*max_complexity + 4` words (64 words = 256 B at
  the default 30), so a slot never moves and there is no free list. The CUDA
  context reserves `d_pool[2]` once; a generation uploads into the back buffer
  and never calls `cudaMalloc`/`cudaFree`. At 10 000 genomes that is 2.6 MB per
  buffer, bounded by capacity and flat across generations.
* **Device-canonical hash.** `rc_pool_hash` computes the same FNV-1a postfix
  hash as `genome::postfix_hash`, bit for bit; the pool can therefore dedup in
  stable `(hash, slot)` order without a host hash pass. `Philox4x32-10` is
  implemented with the same positional `(seed, island, generation, slot, draw)`
  contract the full device-variation stage will use.
* **Flat variation arena.** `postfix::Arena` holds `islands × pop` fixed-width
  programs plus SoA length/cost/hash/error arrays. `postfix::Variation` performs
  tournament selection, subtree crossover, point/terminal/subtree mutation,
  complexity and width bounds, structural hashing and duplicate rejection
  directly on the arena. It allocates no tree and calls no simplifier per child.
  Every random decision comes from Philox4x32-10 keyed by
  `(seed, island, generation, slot)`.
* **Survivor-only archive hand-off.** The archive is tested on `(error, cost)`
  first (`Archive::admits`); a tree is rebuilt from postfix only when the
  candidate actually enters the front. `Archive::canonicalize` then rebuilds a
  monotone `(cost, error)` staircase, and `Archive::hash` is the determinism
  gate. The final reported front is rebuilt from those survivors only and
  rescored in `f64` on the full training sample.
* **The full opcode set in the pool kernel.** The fixed-slot kernel used to fall
  through to `r = 0` for `min`, `max`, `mod`, comparisons, logical operators and
  `if`; it now calls the same `rc_exec_row`
  as the packed kernel (`gpu::tests::pool_kernel_implements_every_emitted_opcode`).
* **Unsupported programs no longer serialise the batch.** Before #131 `if` had
  no kernel opcode, and the old `GpuSearch::score` scored every such tree one
  after another on the calling thread. `if` is now a native opcode; the
  remaining unsupported programs (constraints, aggregates, unlowed blocks) are
  submitted to a persistent bounded CPU worker pool *before* the kernel launch,
  so the fallback overlaps the batch instead of forming a tail. Each tree is
  independent, so the result is bit-identical to the serial walk and merged in
  candidate order; `search::tests::cpu_fallback_is_identical_for_one_and_many_threads`
  pins the one-thread and four-thread results together and
  `fallback_pool_is_bounded_and_merges_in_candidate_order` pins the queue bound
  and the merge order with a non-contiguous pending set.
* **Archive insertion without a pre-emptive clone.** `Archive::try_insert_view`
  tests dominance on `(cost, error)` first and clones the tree only for a
  candidate that enters the front.

The path is selected before generation 0 and recorded in the backend note:

```sh
rcsr ... --no-fit --gpu --variation auto      # flat arena when it qualifies
rcsr ... --variation node                      # historical owned-tree loop
rcsr ... --variation postfix                   # require the flat path
```

`auto` requires a GPU, `--fit-iters 0`, `islands × pop >= 4096`, the default
complexity model, no aggregates/custom blocks/dimension constraints, no
screening or predictor subsampling and no QD layer. The flat engine draws parents
from the island tournament only: the archive-guided `--front-parents` draw is
not ported, so `auto` can produce a different (but deterministic) trajectory. Everything else stays on the
owned-tree path, including all constant fitting (#116), simplification of
reported expressions, checkpoints and exact `f64` reporting. A fitting run can
therefore never be silently approximated by the flat engine.

### The `if` cell and the worker pool (#121)

The regression in `gpu-search/2` §2.2 was measured as a GPU run with default
operators (11.16 s) against the same run with `--ops` excluding `if` (1.17 s):
40 generations, `pop 200`, 4 islands, `--no-fit`, `N = 1 500` training rows,
seed 1, RTX 3060. Remeasured on the same host with the finished lowering and
worker pool:

| Run | Wall (three runs) | vs recorded 11.16 s |
|---|---|---|
| default operators (includes `if`) | 1.04 / 1.09 / 1.17 s | **9.5–10.7× faster** |
| `--ops` excluding `if` | 0.87 / 0.90 / 0.93 s | — |

The `if` cell is now within 1.15× of the no-`if` cell, so the serialisation is
gone rather than merely shortened. `if` is native (#131); the pool covers the
programs that still have no opcode.

Fallback-only cell, `N = 15 000` training rows, `pop 2 000`, 5 generations,
`--ops add,mul,div,require`, 5 704 unsupported programs:

| `RCSR_CPU_FALLBACK_THREADS` | fallback s | whole run s |
|---:|---:|---:|
| 1 (serial) | 4.77 | 10.44 |
| 8 (default) | 1.44 (**3.3×**) | 7.48 (1.4×) |

A mixed 40-generation cell (`if` plus `require`, 15 756 fallback programs) gives
the same Pareto archive at 1, 4 and 8 workers — SHA-256 of
(expression, train error, test error, complexity) is `06b05c11e491dbf2` for all
three — with fallback time 2.07 / 1.19 / 0.79 s and wall 3.48 / 2.45 / 1.74 s.

**The 5× fallback-only target is not reachable on this host and is not claimed.**
The i7-3770K has 4 physical cores (8 hyperthreads); the pool achieves 3.3× of
the 4× ceiling (83 % parallel efficiency), and Amdahl's law caps the whole
run at 2.0× because variation and simplification stay serial. Reaching 5× on
unsupported programs needs either at least 8 physical cores or native opcodes
for the remaining operators (`require`/`between`/`constrain`, aggregates),
which is the opcode-request path in `gpu-lowering/C-lowering-pass.md` §6, not a
scheduling change. The 9.5× `if` cell itself is met at 9.5–10.7× because the
native `if` opcode removes the fallback rather than parallelising it.

Measured on the RTX 3060 host (i7-3770K, 4C/8T), `N = 3 750` training rows from
a 5 000-row two-input sample, `--no-fit --gpu --seed 1`, `P = 10 000`, default
operators, 10 generations, three interleaved runs each on a busy shared host
(the absolute numbers are noisy; the ratio is pairwise):

| Run | wall, s | search, s | generations/s |
|---|---:|---:|---:|
| `--variation node` (historical default pipeline) | 5.71 / 6.82 / 7.20 | 3.54–7.24 | 1.47 |
| `--variation postfix` | 0.94 / 1.02 / 1.45 | 0.69–1.04 | 9.80 (**6.7×**) |
| `--variation node --pipeline-depth 0` (best tree setting) | 5.01 / 5.15 / 5.28 | 3.19 | 1.94 |

At a fixed wall of about 2.2–4.1 s the flat path reached a best training error
of 0.022 (40 and 80 generations) against 0.118 for the owned-tree path in 10
generations. A 2-D hypervolume over `(complexity, train error)` with reference
point `(40, 1)` was 34.6 for the 40-generation flat run at 2.2 s against 31.6
for the 10-generation tree run at 3.5 s. The archive is deterministic for a
seed, repeats across 1 and 4 islands, and a checkpoint/resume trip reproduces
the uninterrupted archive hash (`tests/postfix_loop.rs`).

What remains staged: device-side mutation/crossover and a fused device
selection/archive. The pool, hash and Philox contracts above are the interface
those stages consume; the CPU variation, simplification, fitting, checkpoints
and f64 reporting stay authoritative meanwhile.

## Batched constant fitting (issue #169)

Scoring was on the GPU long before fitting was: non-affine candidates still ran
the scalar CPU Levenberg–Marquardt loop, which dominated wall time at large `N`.
`--fit-strategy gpu` (opt-in; `exact` remains the default) moves that loop into
one CUDA block per candidate:

1. The postorder tree is flattened into 4-word records `[op, kid0, kid1, slot]`.
   Only operators with an analytic value and adjoint are eligible.
2. The block evaluates the value and the reverse-mode Jacobian for its `p`
   constants, and accumulates `JᵀJ`, `Jᵀr` and SSE. Each statistic is owned by
   one thread and summed in fixed chunk order, so the reduction is deterministic
   and needs no atomics.
3. Thread 0 solves `(JᵀJ + λ diag) δ = −Jᵀr` in f64 with a Cholesky
   conditioning gate, then the block scores the trial and accepts/rejects in the
   same order as the CPU loop. Damping retries reuse the cached normal
   equations.
4. After the damped loop, four undamped Gauss–Newton steps polish the constants
   in f64 on an exact f64 copy of the training rows (uploaded once, only when
   this arm is selected), so the returned constants are not limited by the f32
   evaluator copy or by f32 row arithmetic.

**Eligibility and fallback.** A candidate is offered to the kernel only for the
squared/AIC family, with `p <= 16`, at most 96 flattened nodes, no shared block
constants, an analytic derivative for every operator, and at least `32p` rows.
The kernel returns a status: `0` usable, `1` ill-conditioned or rank-deficient
(the undamped Cholesky diagonal ratio exceeds `1e4` or a pivot hits the
`1e-12` floor), `2` a non-finite value or residual. Status `1`/`2`, unsupported
trees and a failed launch are refit by the exact CPU arm, seeded with the GPU
constants. `--fit-report` and the `gpu.fit` JSON object count
`attempts`/`gpu`/`fallback`/`refits`; the run manifest records the same ledger.

**Exactness.** Search fitness is still f32 and only guides selection. Every
finalist is refit by the exact CPU f64 LM on all training rows before the final
archive is rebuilt, and the archive is then scored by the CPU in f64. Reported
constants and metrics therefore never come from f32. The unit tests upload the
same f64 rows the CPU arm uses; the kernel's final constants then match the CPU
LM to machine precision (median `2e-15` in the p=1..16 test, median relative
constant error `<= 1e-5` and SSE relative `<= 1e-8` asserted) and every
ill-conditioned/non-finite candidate carries a fallback status. Without the f64
copy (a device-allocation failure) the polish runs on f32-rounded inputs and the
constants differ by roughly `1e-7` relative; the finalist refit still makes the
reported numbers exact.

**Measurements (RTX 3060, i7-3770K, this host).** Fit stage at `N = 154 000`,
8-thread CPU wall versus the whole GPU batch (upload + kernel + download):

| p | P | CPU wall | GPU wall | fit speed-up |
|---:|---:|---:|---:|---:|
| 4 | 256 | 28.0 s | 1.63 s | 17.1× |
| 8 | 64 | 20.5 s | 0.82 s | 25.2× |

`P = 256` is a quarter of the issue's `P >= 512` gate (a full run would exceed
the one-minute benchmark budget); the p=4 cell therefore falls just short of the
20× gate while p=8 clears it. End-to-end, N=154k, 4 islands, `--pop 64`, two
generations, `--metric squared`, smooth finite operators: **12.53 s (`exact`) ->
10.46 s (`gpu`), 1.20x**. The fit kernel itself was 1.38 s, i.e. about 13 % of
end-to-end wall; 454 candidates were fitted on the device and 306 fell back (the
random population contains many rank-deficient or non-finite starts), and 235
finalists were refit exactly on the CPU. The `>= 1.3x` end-to-end and `>= 85 %`
duty gates are therefore **not met by this first arm**; the measured lever is
the fit-stage ratio on eligible candidates, and the fallback/refit share is what
limits Amdahl. A dispatcher that avoids offering degenerate candidates and a
lower-cost finalist refit are the follow-ups, not a default change.

**Overlap.** The overlap pipeline (`--pipeline-depth`) is not combined with this
arm yet: it scores a tree as soon as it is offered, before the barrier fit
batch, so the GPU-fit path uses synchronous generation batches.

## The DGX Spark path

| | |
|---|---|
| SoC | GB10 Grace Blackwell (CPU and GPU dies joined by NVLink-C2C) |
| CPU | 20 Arm cores (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | 48 SMs, ~31 TFLOPS FP32, **`sm_121`** |
| FP64 | 1/64 of FP32 |
| Memory | 128 GB LPDDR5X, ~273 GB/s, **coherent and unified** |
| Software | DGX OS (aarch64), CUDA 13.0+ |

Exact build commands:

```sh
# RTX 3060 / Ampere: plain device memory, CUDA 11.1+ (verified with 12.0)
RCSR_CUDA_ARCH=sm_86 cargo build --release --features cuda

# DGX Spark / GB10: CUDA 12.8+ (13.0+ on DGX OS), unified memory on by default
RCSR_CUDA_ARCH=sm_121 cargo build --release --features cuda

# unified memory on any other target, e.g. to test the path on a discrete GPU
RCSR_CUDA_ARCH=sm_86 RCSR_GPU_UNIFIED=1 cargo build --release --features cuda

# sm_121 with plain device memory instead (not recommended: 128 GB is the point)
RCSR_CUDA_ARCH=sm_121 RCSR_GPU_UNIFIED=0 cargo build --release --features cuda
```

Plan and status:

1. **Native build.** `RCSR_CUDA_ARCH=sm_121`, CUDA 12.8+ `nvcc` (CUDA 13 on DGX
   OS). Implemented: the architecture is accepted by the validator and the
   build script reports the required toolkit instead of dumping nvcc output.
   `sm_121` is a *consumer Blackwell* target; datacentre flags such as `sm_100`
   will not run on it. **The binary itself has not been built or run here**
   because this machine's `nvcc` is CUDA 12.0.
2. **Unified memory.** Implemented. Building for `sm_121` compiles the context
   with `-DRCSR_UNIFIED=1`, which allocates every buffer with
   `cudaMallocManaged` (host-accessible, explicit copies removed). The same
   block compiles and runs on `sm_86` with `RCSR_GPU_UNIFIED=1`, where it is
   measurably slower; see the table above.
3. **Whole-generation batches.** Implemented as `Population` +
   `eval_population`, measured here with 8 000 programs in one launch, and
   wired into the search loop by issue #26 (one batch per generation).
4. **Bigger blocks and wide-dataset splitting.** Planned: with 48 SMs, use
   larger blocks and split very wide datasets across several blocks per
   expression with an atomic or two-pass reduction.
5. **Keep FP32, verify in FP64.** Score on the GPU in FP32; re-check the
   surviving Pareto front on the CPU in FP64 before reporting. This matters most
   on the 200 000-row dataset, where f32 accumulation moves the p99 agreement
   even though the median stays at f32 epsilon.
6. **Don't oversubscribe the memory bus.** The CPU evolution and the GPU
   evaluation share 273 GB/s; hand off between them rather than running both
   flat out.

The expected gain over an optimised AVX-512 CPU implementation of the same
evaluation is **10–40×** (reported figures for this style of workload range from
~18× to ~40×; the 201–1 678× measured above is against a much slower
single-threaded reference), rising further with row tiling and CUDA graphs.

## Measuring the full GPU path (issue #118)

The older benchmark placed an event pair immediately after the launch, so
`cudaEventElapsedTime` reported only kernel execution. Sync, the device→host
copy of the SSEs and the per-generation `cudaMalloc`/`cudaFree` happened outside
that window — the overhead that made the end-to-end loop look worse than the
kernel table.

`eval.cu` now records **disjoint host windows** for the whole batch:
`rc_stats_begin` → allocation (including the synchronising
`cudaMalloc`/`cudaFree`) → H2D upload → launch → `cudaEventSynchronize` → D2H
download. `rc_last_stats` exposes them; `GpuStats` (`src/gpu_stats.rs`)
accumulates them. `kernel_s` is GPU-event time *inside* that wall: it is the
numerator of duty, never another additive phase.

The `--json`/NDJSON `gpu` block contains:

`kernel_s`, `launch_s`, `sync_s`, `alloc_s`, `h2d_s`, `d2h_s`, `transfer_s`,
`window_s`, `host_s`, `accounted_s`, `wall_s`, `idle_s`, `residual_fraction`, `h2d_bytes`,
`d2h_bytes`, `batches`, `failed_batches`, `programs`, `evals`, `row_evals`,
`evals_per_s`, `row_evals_per_s`, `generations_per_s`, `duty_cycle`,
`duty_source`, `sm_occupancy`, `occupancy_source`, `quality_per_gpu_hour`, and
the #131 per-generation `generations` opcode/fallback report. A CPU run reports
`"gpu": null`.

Definitions that matter:

* **wall** is the whole evaluator path measured by the caller. `window_s` is
  the five C host windows and `host_s` the Rust glue outside them; their sum
  `accounted_s` closes to wall, and `residual_fraction` reports any leftover.
  `tests/gpu_loop.rs` asserts closure within 1 % on a real GPU run.
* **evals/s** counts candidate evaluations (`programs / wall`); **row-evals/s**
  counts `programs × rows / wall`. A failed batch contributes wall and bytes but
  zero evaluations.
* **duty** is `kernel_s / wall_s`, a kernel-time fraction. It is not
  `nvidia-smi utilization.gpu` (a 1 Hz sample of "some kernel was resident") and
  it is not SM occupancy.
* **quality/GPU-hour** is decades of error reduction against the constant-mean
  baseline per measured full-path GPU-hour:
  `-log10(best_sse / mean_baseline_sse) / hours`. A throughput gain is not a
  search gain: compare quality at matched cumulative evaluations and paired
  seeds.
* **`sm_occupancy` is `null`** with `occupancy_source: "ncu (not run)"`.
  Occupancy requires `ncu` and is never inferred from sampled utilisation.

`rcsr-gpu-bench` prints the ledger, full-path candidate and row throughput,
duty, quality per GPU-hour, and a comparison against both the interpreted walk
and the engine's own SoA/AVX evaluator (the baseline the #118 gate names).

How to measure properly:

```sh
# full-path ledger from the harness itself
./target/release/rcsr-gpu-bench --data cell5000.csv --target y --vars x1,x2 \
  --batch 800 --depth 5

# continuous duty/timeline and host gaps: nsys sees the real API sequence
nsys profile --stats=true -t cuda,nvtx \
  ./target/release/rcsr-gpu-bench --data cell5000.csv --batch 800 --depth 5

# achieved occupancy / SM throughput: ncu, one diagnostic cell, separate run
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./target/release/rcsr-gpu-bench --data cell5000.csv --batch 512 --depth 5

# CUPTI activity tracing (continuous duty, not a 1 Hz sample)
nsys profile -t cuda --cuda-um-cpu-page-faults=false -o gpu-duty ...
```

Do not use `nvidia-smi --query-gpu=utilization.gpu` for a sub-10 ms kernel: it
samples at 1 Hz (or 100 ms at best) and measures "some kernel was resident", not
occupancy or SM throughput. Use it only as a coarse cross-check.

A/B the evaluator that matters: `--no-simd` (scalar walk), default (SoA/AVX),
and `--gpu`. A speed-up against the interpreted walk is not the gate.

### Measured on the RTX 3060 cell (issue #118)

Cell: `cell5000.csv`, `N = 5 000`, `V = 2`, `P = 800`, depth 5, seed 7,
`sm_86`, CUDA 12.0, plain device memory. Three interleaved repeats, 25 batches
each, on a shared 8-core host with other builds active; absolute times move by
tens of percent, so medians are quoted.

| | before (event-only timer) | after (full path) |
|---|---:|---:|
| GPU batch wall | 0.519 ms (kernel only) | **0.523 ms** (alloc+H2D+launch+sync+D2H) |
| row-evals/s | 7.70e9 (kernel only) | **7.66e9** (full path) |
| SoA/AVX CPU batch | not measured | 367.2 ms |
| speed-up vs SoA/AVX | not measured | **~702×** |
| speed-up vs interpreted walk | 847–864× | ~1 035× |
| C-window / host coverage | not measured | **99.1–99.2 % / 0.8–0.9 %** |
| kernel-time duty | not measured | **70.1–83.5 % (median 80.2 %)** |

The old "8e9 evals/s" was the kernel event only. The new number is the same
order on this cell because the batch is large enough that copies and sync are
small; the *reported* number is now the real path and the decomposition is
visible. On a small cell (`N = 1 000`, `P = 100`, depth 5) the full path was
0.095–0.221 ms against 5.96–16.52 ms for SoA/AVX, i.e. 27–128× faster in every
repeat; there is no small-cell regression in evaluator throughput.

Occupancy is **not** verified: `ncu` returns `ERR_NVGPUCTRPERM` (no permission
to read GPU performance counters) on this host, so `sm_occupancy` stays `null`.
`nsys profile -t cuda` traces the run, but its importer is missing here, so no
CUPTI duty number is claimed. Kernel-time duty from the ledger is not a
substitute for either.

## Verified versus unverified

| Item | Where it was tested | Status |
|---|---|---|
| Architecture parser/validator (accept/reject, toolkit minimums) | unit tests, any machine | **verified** |
| CUDA 12.0 rejects `sm_121` with the target, toolkit and minimum version named | RTX 3060 host, `nvcc` 12.0 | **verified** |
| Plain device-memory path: build, tests, benchmark | RTX 3060, `sm_86`, CUDA 12.0 | **verified** |
| Unified-memory path: build, tests, benchmark | RTX 3060, `sm_86`, `RCSR_GPU_UNIFIED=1` | **verified on a discrete GPU** |
| Whole-population batch equals one-at-a-time SSEs | RTX 3060, `cargo test --features cuda` | **verified** |
| GPU-vs-CPU agreement | RTX 3060, test + benchmark | **verified** (tables above) |
| One GPU batch per generation in the search loop | RTX 3060, `tests/gpu_loop.rs` + end-to-end table | **verified** |
| Bounded CPU/GPU overlap: archive and every `f64` error identical to the synchronous path | RTX 3060, `tests/gpu_loop.rs` (`pipeline_depth` 0 vs 2), single and two islands | **verified** |
| Queue FIFO order and `capacity + batch + 1` memory bound under a forced slow consumer | `src/pipeline.rs` unit tests, any machine | **verified** |
| ≥1.1× end-to-end with the default fitter at pop 400, N ≥ 5 000 | RTX 3060, `d5k.csv`, shared host | **not met** — 1.02–1.06×; GPU phase is only 2–3 % of wall |
| Reported law/train/test errors agree with the CPU backend | RTX 3060, `tests/gpu_loop.rs` | **verified** (law predictions within 1e-12; errors within 1e-9, observed 0) |
| Full-path ledger (alloc/H2D/launch/sync/D2H) and duty | RTX 3060, `src/gpu_stats.rs` tests + `tests/gpu_loop.rs` + `rcsr-gpu-bench` | **verified** (host windows close to wall; kernel-time duty, not occupancy) |
| Speed-up against the SoA/AVX path (not the interpreted walk) | RTX 3060, `rcsr-gpu-bench` | **measured** (see the PR/issue; the benchmark prints both baselines) |
| Failed batch is timed but adds no evaluations | RTX 3060, `tests/gpu_loop.rs` | **verified** |
| SM occupancy / achieved SM throughput | needs `ncu` with counter permission | **unverified here** — reported as `null`, never invented |
| `--no-gpu` forces the CPU, with and without the feature | any machine, `tests/gpu_loop.rs` | **verified** |
| Missing device / CUDA error falls back with one note | `RCSR_GPU=0`, `tests/gpu_loop.rs` | **verified** (no-device path; an injected kernel failure is not exercised) |
| `sm_121` compilation and execution | needs a GB10 with CUDA 12.8+ / 13 | **unverified** — no such hardware here; the build fails early with the version requirement |
| Dataset larger than device VRAM through managed memory | needs a GB10 (or a >12 GB dataset) | **unverified** — the allocation path is compiled and exercised on `sm_86`, but not at that size |
| On-GPU FP64 verification of survivors | planned | **unverified** — survivors are re-checked by the CPU `f64` evaluator today |

What would close the gaps: on a DGX Spark (or any `sm_121` + CUDA 13 machine),
run

```sh
RCSR_CUDA_ARCH=sm_121 cargo test --release --features cuda
RCSR_CUDA_ARCH=sm_121 ./target/release/rcsr-gpu-bench --data <set> --batch 8000
```

then repeat the benchmark with a dataset whose f32 columns exceed device VRAM,
and compare the unified run with `RCSR_GPU_UNIFIED=0`. The benchmark prints the
device, the architecture and `memory : device|unified`, so the result records
which path produced it.

