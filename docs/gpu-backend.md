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
are compiled to native opcodes instead of falling back to the serial CPU
evaluator. `floor`/`ceil` are one `FRND`; `round` is `roundf` (ties away from
zero, matching Rust — not `cvt.rni`, which is ties-to-even); `min`/`max` guard
NaN explicitly so either NaN operand yields NaN like `expr.rs`; `if` evaluates
both arms but a true select discards the unselected one; `mod` uses the host's
floored `a - b*floor(a/b)` with the `|b|<1e-12` and finite-quotient guards;
`atan2`, `erf` and `erfc` call libdevice (2/2/4 ulp documented; `erf(-0)` is
normalised to `+0` to match the host). The static kernel grew from 1000 to 1312
SASS (+31%), registers from 26 to 27, and the 288-byte stack frame is unchanged
on `sm_86`. A CUDA-guided run reports the opcode and fallback counts for each
generation under `gpu.generations` in `--json`; a CPU run's JSON is unchanged.
Operators without an opcode (`asinh`, `acosh`, `atanh`, constraints, aggregates)
still take the CPU fallback, and the final f64 CPU rescore remains the
exactness guarantee for reported results.

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

Variation, simplification, fitting and de-duplication are CPU work; a
candidate's SSE is a function of its tree alone. The only legal overlap is
therefore **inside one generation**: produce chunk `c+1` while chunk `c` is on
the device. Selection, migration, restart and the archive still happen at the
generation barrier, after every result has been received. Overlapping
*generations* (generating `g+1` while `g` scores) is not legal without changing
the trajectory, because `vary(g+1)` needs the population that `select(g)`
produces from the completed errors of `g`.

Since #119 the main thread drains a bounded queue while the island threads keep
producing:

* each island thread offers a candidate to a `SyncSender` immediately after it
  is fitted and pushed into the next population (its own FIFO order);
* `pipeline::chunk_plan(pop, depth)` splits the generation into `depth` chunks of
  `ceil(pop / depth)` candidates. The scorer takes one chunk at a time, so one
  chunk is on the device while later ones are still being generated and fitted;
* the channel holds at most `min(pop, depth * batch)` jobs and the scorer holds
  one more batch. Beyond the population the search already owns, the queue adds
  at most one generation of cloned trees plus one batch — independent of the
  generation count. The `(island, slot)` result table is `O(pop)` bits and is
  what makes the mapping order-independent;
* selection, migration, restart and the archive still happen at the generation
  barrier, after every result has been received.

`--pipeline-depth N` (default 2) sets the number of chunks; `0` restores the
synchronous path. A larger depth starts the device earlier (smaller first chunk)
at the cost of more launches; it does not change the ceiling below. A scoring
error discards every `f32` value the overlap scorer already wrote, stops further
GPU work, keeps draining (so no producer can block), and re-scores the whole
generation in `f64` on the CPU through the same bounded fallback as the
synchronous path. A tree the kernel cannot compile is left pending and scored
exactly at the barrier, so the fallback set and the #131 program/fallback
accounting are identical to the synchronous path (asserted in
`tests/gpu_loop.rs`).

**Ceiling.** Let `F` be CPU seconds per generation (variation + simplify + fit +
dedup) and `E` the GPU seconds. The synchronous cycle is `F + E`; chunked
overlap is `F + E/depth` at best (the last chunk still tails), so

```
speed-up <= (F + E) / (F + E/depth) <= 1 + E/F .
```

With the default fitter on the target cell (`d5k.csv`, 5 000 rows, pop 400) the
GPU phase is a fraction of a percent of the generation, so the measured ceiling
is about `1.02×` (below). The obstruction is Amdahl's, not the queue's: constant
fitting is ~98 % of the generation and is not work the evaluator can absorb.
Raising `--pipeline-depth` buys at most the last `E/depth` tail; it cannot move
the fit fraction. The `score_chunk` seam in `search.rs` is where a deferred-fit
arm joins the same batch; until #169 lands, `fit_defers_to_gpu` keeps overlap off
for a fit strategy that would otherwise hand an unfitted tree to the scorer.

Measured on the RTX 3060 (`sm_86`, CUDA 12.0), shared host with other processes
active (load average 16–25), alternating depths. Absolute seconds are noisy; the
useful numbers are the user time and the no-fit baseline:

| workload (`d5k.csv`, 5 000 rows, pop 400, 12 generations, `--fit-iters 8`) | depth 0 | depth 2 | depth 4 |
|---|---:|---:|---:|
| search wall, round 1 / round 2 (s) | 30.4 / 21.1 | 27.6 / 30.5 | 37.5 / 19.3 |
| user CPU, round 1 / round 2 (s) | 17.3 / 16.2 | 17.4 / 16.7 | 17.4 / 15.6 |
| same, `--no-fit` (8 generations, search s) | 0.31 | — | 0.22 (depth 8) |

The depth-to-depth spread is host noise: no depth shows a repeatable wall gain,
and user CPU does not increase (the ~1–2 s spread is contention, not removed
work), which is the point — overlap moves *when* the scoring happens, it does
not remove work. The no-fit run puts the whole generation at ~38 ms; an
event-timed `rcsr-gpu-bench` batch of the same 400 programs × 5 000 rows takes
**0.238 ms**. Against a fitted generation of ~1.76 s that leaves the fitter at
≈98 % of wall and an overlap ceiling of

```
1 + E/F <= 1 + 0.038/1.76 ~= 1.02.
```

**The ≥1.1× end-to-end target is not reachable by any amount of overlap at this
budget.** Reaching it requires the fit stage to leave the CPU (the batched GPU LM
in #169, whose own target is a ≥20× fit-stage speed-up) or the fit budget to
fall; more chunks against a 0.24 ms GPU phase cannot buy 10 %.

The archives are byte-identical between depths (single- and two-island runs,
depths 0/2/4/8), including the exact `f64` test errors. So are the #131
per-generation program and fallback counts; a tree the kernel cannot compile
(e.g. `asinh`, which the kernel does not lower in either path) is a `f64` CPU
fallback in both schedules (`tests/gpu_loop.rs`).

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

### Fixed-capacity genome pool and the unsupported tail (#122)

The first dehosting stage removes two pieces of host work that did not depend on
the GPU at all:

* **One persistent program arena.** `genome::GenomePool` is a fixed-capacity,
  double-buffered postfix arena. Every genome occupies exactly
  `width_for(max_complexity) = 2*max_complexity + 4` words (64 words = 256 B at
  the default 30), so a slot never moves and there is no free list. The CUDA
  context now reserves `d_pool[2]` once; a generation uploads into the back
  buffer and never calls `cudaMalloc`/`cudaFree`. At 10 000 genomes that is
  2.6 MB per buffer, bounded by capacity and flat across generations.
* **Device-canonical hash.** `rc_pool_hash` computes the same FNV-1a postfix
  hash as `genome::postfix_hash`, bit for bit; the pool can therefore dedup in
  stable `(hash, slot)` order without a host hash pass. `Philox4x32-10` is
  implemented with the same positional `(seed, island, generation, slot, draw)`
  contract the full device-variation stage will use.
* **Unsupported programs no longer serialise the batch.** `if` has no kernel
  opcode, and the old `GpuSearch::score` scored every such tree one after
  another on the calling thread. They are now split into contiguous slices and
  scored on a bounded CPU pool (`RCSR_CPU_FALLBACK_THREADS`, default
  `min(cores, 8)`). Each tree is independent, so the result is bit-identical;
  `search::tests::cpu_fallback_is_identical_for_one_and_many_threads` pins the
  one-thread and four-thread results together.
* **Archive insertion without a pre-emptive clone.** `Archive::try_insert_view`
  tests dominance on `(cost, error)` first and clones the tree only for a
  candidate that enters the front.

Measured on the RTX 3060 host (i7-3770K, 4C/8T), `N = 5 000`,
`--no-fit --gpu --seed 1`, `P = 10 000`, default operators (including `if`),
three interleaved runs each, wall for 10 generations:

| Run | min / median |
|---|---:|
| `origin/main` | 18.3 s / 22.8 s |
| #122 stage | 8.8 s / 10.7 s (**2.1–2.4×**) |

The per-generation CPU fallback for 8 521 unsupported trees fell from 2.15 s
(serial) to 0.73 s on the 8-thread pool (3.0× on 4 physical cores). With the
operator set that has no unsupported opcode (`--ops` without `if`) the same
`P = 10 000`, 20-generation cell is unchanged (5.44 s vs 5.34 s): the pool
removes allocation and clone overhead there, but variation and the startup
population dominate. The kernel itself stays ~1–3 ms per generation: at the
`--no-fit` slope this is under 1 % duty, which is why the remaining plan targets
full device variation and fitting (#116), not the launch path. Exactness: the `P = 2 000`, 5-generation JSON
report is byte-identical to `origin/main` apart from `elapsed_seconds`, i.e.
the same constants, expressions and errors.

What remains staged: device-side mutation/crossover and a fused device
selection/archive. The pool, hash and Philox contracts above are the interface
those stages consume; the CPU variation, simplification, fitting, checkpoints
and f64 reporting stay authoritative meanwhile.

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
| Bounded CPU/GPU overlap: archive, every `f64` error and #131 accounting identical to the synchronous path | RTX 3060, `tests/gpu_loop.rs` (`pipeline_depth` 0 vs 2/4/8), single and two islands | **verified** |
| Queue FIFO order per producer, consumer-error propagation and the `capacity + batch` memory bound under a forced slow consumer | `src/pipeline.rs` unit tests, any machine | **verified** |
| A failed overlap batch discards earlier `f32` chunks before the `f64` re-score | `search::tests::a_failed_overlap_batch_discards_earlier_f32_chunks`, any machine | **verified** |
| `chunk_plan` is total, covers the population and bounds the queue by one generation | `src/pipeline.rs` unit test, any machine | **verified** |
| ≥1.1× end-to-end with the default fitter at pop 400, N ≥ 5 000 | RTX 3060, `d5k.csv`, shared host | **not reachable by overlap** — fit is ≈98 % of wall; measured ceiling ≈1.02×; needs fit offload (#169) |
| Reported law/train/test errors agree with the CPU backend | RTX 3060, `tests/gpu_loop.rs` | **verified** (law predictions within 1e-12; errors within 1e-9, observed 0) |
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

