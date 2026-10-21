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
the fix for the observed incident where the archived `resume_10h.py` omitted
`--gpu` and silently continued a GPU-selected population on the CPU. Always
state `--gpu` or `--cpu` explicitly in a launcher, even though CPU is the
default.

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
error: RCSR_CUDA_ARCH=sm_121 needs CUDA 12.9 or newer, but `nvcc` is CUDA 12.0.
error: sm_121 is the GB10 in the DGX Spark; its DGX OS images ship CUDA 13.0.
error: Install a newer toolkit and point RCSR_NVCC at it, for example:
error:   RCSR_NVCC=/usr/local/cuda-12.9/bin/nvcc RCSR_CUDA_ARCH=sm_121 cargo build --release --features cuda
error: or build for the GPU in this machine, for example:
error:   RCSR_CUDA_ARCH=sm_86 cargo build --release --features cuda
```

CUDA 12.8 was the wrong floor: it added Blackwell (`sm_100`, `sm_120`) but its
`nvcc --list-gpu-arch` stops at `compute_120`. `sm_121` first appears in CUDA
12.9 (`compute_121`), so a 12.8 toolkit fails with "not defined for option
'gpu-architecture'" even though the build script used to let it through.

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
| `sm_120` | 12.8 | consumer Blackwell |
| `sm_121` | 12.9 | GB10 / DGX Spark; DGX OS ships CUDA 13.x |

The build-script check is deliberately about the **compiler**, not the device:
`RCSR_CUDA_ARCH=sm_121` succeeds on any machine with CUDA ≥ 12.9 and needs no
GB10 attached.

### `sm_121` compile-only validation (#28)

[`tests/sm121_compile.rs`](../my-version/tests/sm121_compile.rs) compiles the
real `src/cuda/eval.cu` for `sm_121` and inspects the cubin. It needs CUDA ≥
12.9 and `cuobjdump`, but no GPU; when no such toolkit is found it prints a skip
note and the rest of the suite is unaffected. Run it explicitly:

```sh
RCSR_SM121_NVCC=/path/to/cuda-13/bin/nvcc \
  cargo test --release --test sm121_compile -- --nocapture
```

The test asserts that `nvcc`/`ptxas` accept `-arch=sm_121`, that
`cuobjdump --list-elf` names an `.sm_121.cubin`, that `--dump-sass` is tagged
`code for sm_121` / `EF_CUDA_SM121`, and that `ptxas -v` reports a resource line
for `eval_pool_kernel` and `fit_kernel`.

Recorded on 2026-10-21 with CUDA 13.0.88 (V13.0.88) on an x86_64 host, no GPU
involved:

| Kernel | Registers | Stack | Static smem |
|---|---|---|---|
| `eval_pool_kernel` | 30 | 288 B | 1 024 B |
| `eval_kernel` | 33 | 288 B | 1 024 B |
| `fit_kernel` | 110 | 1 744 B | 40 672 B |
| `predict_kernel` | 29 | 288 B | — |
| `hash_kernel` | 14 | — | — |

`cuobjdump --dump-sass` reports `code for sm_121` and
`EF_CUDA_SM121 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM121)`; the disassembly contains
Blackwell instructions (`UMOV.64`, `DFMA`, `LDS.64`). This is **compile
evidence**: the source and toolchain produce real `sm_121` device code, and the
resource budget is recorded. It is **not** evidence that a GB10 loaded or ran
the kernels, and it says nothing about occupancy or speed.

**The runtime half stays open.** `sm_121` execution needs a GB10 (or another
`sm_121` device) and its driver. Acceptance is therefore split explicitly:
*compiles for `sm_121`* is local and permanent (the test above); *runs on GB10*
is one dated remote session recording `nvidia-smi` (device, driver, compute
capability), the GPU test suite and a benchmark. The second must never be
marked from cross-compilation alone.

**PTX portability is not `sm_121` performance evidence.** The current `sm_86`
build embeds `compute_86` PTX (`cuobjdump -lptx` shows `*.sm_86.ptx`), and PTX
is forward-compatible, so that fatbin can JIT for `sm_121` and run (set
`CUDA_FORCE_PTX_JIT=1` to force the path). That proves the source is portable
and is the cheapest first check on a rented Spark; it does not measure native
`sm_121` codegen, so it cannot replace either the compile test above or the
native run.


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

### Register and occupancy contract (#131)

The old acceptance wording used a single number ("27 registers") measured on one
kernel before the pool kernel implemented the #131 opcodes. On `sm_86` that
number was never the constraint: 27 and 30 registers round to the **same**
allocation, and the actual occupancy cliff is at 41 registers. The contract is
therefore per kernel and expressed as occupancy, not as a bare register count:

* `eval_kernel`, `eval_pool_kernel` and `predict_kernel` carry
  `__launch_bounds__(RC_MAX_THREADS, 6)` (`RC_MAX_THREADS` is 256), i.e. PTX
  `.maxntid 256, 1, 1` and `.minnctapersm 6`;
* at 256 threads each kernel must stay at or below **40 registers**, keep its
  288-byte stack frame, and produce **no spill stores/loads**;
* that gives six resident blocks per SM, 6 × 256 = 1 536 threads = 48 warps =
  **100 % theoretical occupancy** on `sm_86`.

Measured with the build script's flags (`sm_86`, CUDA 12.0, `-O3`) on
`main` at `ce3201c` (before) and on this branch (after):

```sh
nvcc -O3 -std=c++14 -arch=sm_86 -cubin src/cuda/eval.cu \
     -o /tmp/eval.cubin -Xptxas -v,-warn-spills
cuobjdump -res-usage /tmp/eval.cubin
```

| kernel | REG before | REG after | allocated regs/thread | STACK | SHARED | LOCAL | spills | occupancy at 256 threads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `eval_kernel` (packed / pipeline) | 32 | 31 | 32 | 288 B | 1 024 B | 0 | 0 | 100 % (6 blocks) |
| `eval_pool_kernel` (fixed slots / synchronous) | 32 | 31 | 32 | 288 B | 1 024 B | 0 | 0 | 100 % (6 blocks) |
| `predict_kernel` (test fixtures only) | 28 | 26 | 32 | 288 B | 0 | 0 | 0 | 100 % (6 blocks) |
| `fit_kernel` (batched LM, untouched) | 93 | 93 | 96 | 1 744 B | 40 672 B | 0 | 0 | not subject to this contract |
| `hash_kernel` / `cast_f32_to_f64_kernel` (untouched) | 16 / 12 | 16 / 12 | 16 / 16 | 0 | 0 | 0 | 0 | — |

The before column is today's `main`, after the #121 device stack guards; the
PR #180 snapshot (27 / 30 registers, same 288-byte stack) predates those guards
and remains correct for that commit. Both snapshots round to 32 allocated
registers per thread, so the launch-bounds build changes no occupancy.

Why 32 allocated registers is the floor: for a 256-thread block (8 warps) the
allocator charges `ceil(regs × 32 / 256) × 256` registers per warp. At 26–32
registers that is 4 units = 1 024 registers per warp = 32 effective registers
per thread. A block therefore uses 8 × 1 024 = 8 192 registers, and the SM's
65 536 registers admit 8 blocks — but the 1 536-thread limit admits only 6, so
threads bind and registers do not. The cliff is at 41: six blocks need at most
`65 536 / 6 / 8 = 1 365` registers per warp, which rounds to 5 units = 1 280 =
**40 registers per thread**. At 41–48 registers the allocator charges 6 units
and the SM falls to five blocks = 40 warps = 83 % occupancy. `predict_kernel`
and the 32-thread launch used for tiny `N` are capped by the 16-block limit
instead (16 warps = 33 %). The harness
`evaluator_kernels_carry_the_sm86_occupancy_contract` pins the launch bounds,
the 256-thread constant and the 6 × 256 = 1 536 arithmetic.

The pool kernel's SASS growth came from implementing the #131 opcodes at all
(it previously fell through to `r = 0.0f`); the growth is the cost of the
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
short datasets it is still **latency-bound, not throughput-bound**: at 256
threads the register/occupancy contract above is designed for full residency,
but a small `N` launches only 32 threads, where the 16-block cap limits the SM
to 16 warps (33 %) regardless of registers. The per-thread local-memory stack
then adds memory latency that a full block would hide.
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

The GPU never produces a number a user reads. The kernel accumulates one
**configured objective** per row; the search divides that by the row count and
uses it as the fitness. Today the device has two native objectives:

| `--metric` | device objective per row | status |
|---|---|---|
| `squared` | `r²` | exact objective (up to `f32`) |
| `log_squashed` | `ln(1 + r²)` | exact objective (up to `f32`), #229 |

Every other metric is scored by the CPU evaluator: the kernel has no MAE, rank,
minimax, correlation, slope or classification objective, and using MSE as a
ranking proxy silently changes the search (`--metric slope` was observed to
select a different Pareto point, #236). At the end `full_data_archive`
re-scores the surviving trees on the full training sample with the CPU
evaluator in `f64`, and the reported train/test errors, Pareto front and
convergence statistics come from that rebuild. Constants stay on the CPU and in
`f64` throughout.

The GPU path is used for single-output, unweighted, full-sample runs whose
metric is in the table above. Multi-output (`--target y1,y2`), `--weight`,
every other metric (including implicit, classification, correlation, rank,
slope and AIC/AICc/BIC) and `--predictor-samples` keep the CPU evaluator,
because they need more than one prediction per row, a different objective, a
per-row weight array the evaluator does not carry, or a different set of rows
every generation; the reason is printed in the one-line note.

Fallback and the flags:

* With neither flag, a CUDA build with a usable device selects the GPU **for
  `--metric squared`**. Every other metric, including the default `absolute`,
  is scored on the CPU and prints
  `# gpu: off (metric is not scored natively on the GPU)` once; the resource
  line and the `--json` `backend` both say `cpu` and `backend_reason` carries
  the same cause.
* A build without the feature prints
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
screening or predictor subsampling and no QD layer. The flat engine now has
`--front-parents` parity with the owned-tree loop: at the start of a generation
it compiles the previous front once into fixed-width postfix programs
(`postfix::FrontParents`) and takes 15 % of parent draws uniformly from it,
using the same draw order as `pick_parent` (#76). `--no-front-parents` passes an
empty pool and `front_p = 0`, consumes no random draw and is bit-identical to
the pre-#76 flat trajectory. Everything else stays on the owned-tree path,
including all constant fitting (#116), simplification of reported expressions,
checkpoints and exact `f64` reporting. A fitting run can therefore never be
silently approximated by the flat engine.

#### Archive-hash contract (#122)

The hash box has two well-posed readings, and only those are claimed:

1. **Same-path determinism.** For a fixed seed, build and configuration the
   archive hash repeats exactly: repeated runs, one-island and four-island
   configurations, and a checkpoint/resume trip all reproduce it
   (`tests/postfix_loop.rs`).
2. **Content equivalence on a fixed candidate stream.** The front is reduced by
   `Archive::canonicalize` to a monotone `(cost, error)` staircase: sort by
   `(cost, error, structural hash)`, keep the best per cost, keep only strictly
   improving errors as cost rises, then trim the cap deterministically. The
   hash is FNV-1a over `(cost bits, error bits, structural hash)` in that order,
   so a fixed member set hashes the same however it arrived
   (`pareto::tests::hash_is_content_equivalent_for_a_fixed_candidate_stream`).
3. **Not equality to the pre-#122 trajectory.** That comparison is ill-posed,
   not merely unimplemented. The old archive was updated by ε-dominance, which
   is not transitive, and when over capacity `prune` removed
   `items[len / 2]`; both depend on insertion order, so there is no canonical
   old hash. The flat engine also deliberately uses a different counter-based
   RNG contract. We therefore do not claim that the flat archive equals the
   historical owned-tree archive for the same seed.

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

**Re-measured at the stated `N >= 5 000` (2026-10-21).** The 6.7× above used
3 750 training rows (a 5 000-row file at the default 25 % test fraction), which
is below the issue's `N >= 5 000` wording. The same host, file and budget with
`--test-frac 0` (so **N = 5 000**), three interleaved runs each:

```sh
./target/release/rcsr --data gpu5k.csv --target y --vars x0,x1 \
  --no-fit --gpu --pop 10000 --islands 1 --generations 10 --seed 1 \
  --patience 0 --test-frac 0 --variation node|postfix
```

| Run | wall, s (three runs) | median wall | generations/s |
|---|---:|---:|---:|
| `--variation node` (overlap pipeline, the default) | 6.68 / 7.22 / 6.74 | 6.74 | 1.48 |
| `--variation postfix` | 0.90 / 0.85 / 0.91 | 0.90 | **11.1 (7.5×)** |

Peak RSS was 128.4 MB for the owned-tree run and **113.5 MB** for the flat run;
the flat value did not grow from one to ten generations (113.68 MB at one
generation). The 3 750-row and 5 000-row cells agree, so the target is met at
the stated row count. This is a no-fit screening cell, not production search
throughput; fitting remains on the owned-tree path (#116).

**Fallback and memory boxes.** At `P = 4 000`, `--variation auto` keeps the
owned-tree path and reports `node variation (population 4000 is below the
4096-genome gate)`; forcing `--variation postfix` refuses the run. The arena is
fixed capacity: 2.56 MB of program words plus 20 B of metadata per slot at
`P = 10 000`, and `postfix::tests::ten_thousand_slots_have_a_fixed_memory_bound`
pins that the next generation has exactly the same byte count
(`tests/postfix_loop.rs` runs the `N = 5 000` cell).

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

**Two instantiations.** The kernel is compiled twice and dispatched on the
batch's largest constant count. `p <= 16` keeps the 256-thread block and 256-row
chunk it was measured with (now 43 744 B of shared memory: #229 added the f64
per-row loss buffers); `17 <= p <= 20` uses a 128-thread block and a 128-row
chunk (41 040 B). Both fit two blocks per SM on `sm_86`. The larger alternative — 20 constants with a 256-row chunk —
needs about 50.8 KB, past the 48 KB static ceiling and only one block per SM, so
the shared block is dynamic (`extern __shared__`) and the opt-in limit is raised
once with `cudaFuncSetAttribute`; it is measured below. A block narrower than
the statistic count (231 entries at `p = 20`) gives each thread several
statistics (`t`, `t + blockDim`, …), so every entry is still owned by exactly
one thread and accumulated over rows in ascending order. That keeps the sum
independent of the chunk size: a candidate fitted alone and the same candidate
in a padded `kmax = 20` batch are **bit-identical**, which
`a_mixed_p_batch_matches_single_fits_and_the_cpu_lm` asserts for `p = 4` beside
`p = 17..20`.

**Eligibility, admission and fallback.** A candidate is offered to the kernel
only for a metric with a native device objective (squared/AIC family, and
`log_squashed` since #229), with `p <= 20`, at most 96 flattened nodes, no
shared block constants, an analytic derivative for every operator, and at least
`32p` rows. The device objective is also a **CLI contract** (#223, extended by
#229): an explicit `--fit-strategy gpu` with any metric *without* a native
device objective — that is, outside `squared`, `aic`, `aicc`, `bic` and
`log_squashed` — is a usage error (exit code `2`) raised before any data is
read, because the only alternative would be a silent CPU fit of a different
objective. The message names the two objectives the kernel minimises and every
way out (`--metric squared`, `--metric log_squashed`, or a CPU fit strategy). The device-missing fallback below is unchanged: it never
claims the GPU arm and is recorded with its cause.

The arm also screens before it fits: every unfitted offspring is
scored once on the GPU with its seed constants, and the screened ranking decides
admission. Below **1 048 576 training rows** only the best
`max(32, pop/4)` candidates of each island are admitted to the fitted LM batch
(ties break on complexity and then the fixed population index), and the rest
keep their screening score. At or above that size every candidate with
constants is admitted, because the seed-constant screen no longer predicts the
fitted ranking well enough to leave three quarters of the offspring unfitted:
the 3.9M hourly probe admitted `pop/4`, and its best error (0.4753) was far
worse than the exact arm's (0.1612) at equal evaluations. The threshold is a
row count rather than a timing measurement, so admission stays a pure function
of the screened errors, the row count and the fixed island/population order and
repeated runs are reproducible. Admission never changes the evaluation count or
the exact finalist refit.

The kernel returns a status: `0` usable, `1` ill-conditioned or rank-deficient
(the undamped Cholesky diagonal ratio exceeds `1e4` or a pivot hits the
`1e-12` floor), `2` a non-finite value or residual. Status `1`/`2`, unsupported
trees and a failed launch are refit by the exact CPU arm, seeded with the GPU
constants. Every CPU fallback is labelled by cause: `unsupported_plan`
(no analytic derivative, too many nodes, shared block constants or no
constants), `too_few_rows`, `too_many_constants` (`p > 20`), `metric_not_native`,
`ill_conditioned`, `non_finite` or `launch`. `--fit-report` and the `gpu.fit` JSON object count
`screened`/`attempts`/`gpu`/`fallback`, the per-cause map, `refits`, the kernel
`seconds` and the whole fit stage's `host_seconds`; the run manifest records the
same accepted/fallback/refit totals.

**Exactness.** Search fitness is still f32 and only guides selection. Every
finalist is refit by the exact CPU f64 LM on all training rows before the final
archive is rebuilt, and the archive is then scored by the CPU in f64. Reported
constants and metrics therefore never come from f32. The unit tests upload the
same f64 rows the CPU arm uses; the kernel's final constants then match the CPU
LM to machine precision (median `2e-15` in the p=1..20 test, median relative
constant error `<= 1e-5` and SSE relative `<= 1e-8` asserted) and every
ill-conditioned/non-finite candidate carries a fallback status. For
`p = 17..20` the measured worst cases are SSE relative `6.0e-13` and constant
error `1.2e-12`, and a `p = 20` collinear design is still labelled status 1.
Without the f64
copy (a device-allocation failure) the polish runs on f32-rounded inputs and the
constants differ by roughly `1e-7` relative; the finalist refit still makes the
reported numbers exact.

**Constant cap and the p=17..20 instantiation (#169).** The cap is 20 constants.
`p = 1..16` keeps the original 256-thread instantiation; `p = 17..20` runs the
second one. Their resources, from `cuobjdump -res-usage` on the built
`librcsrcuda.so` and from `gpu::fit_kernel_shape` at runtime, are:

| constant cap | block | row chunk | dynamic shared | registers | stack (local frame) |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 256 | 40 672 B | 89 | 1 792 B |
| 20 | 128 | 128 | 38 992 B | 85 | 1 824 B |

Both stay under the 48 KB static ceiling and fit two blocks per SM on `sm_86`.
The rejected alternative — 20 constants with a 256-thread block and a 256-row
chunk — is 50 768 B (one block per SM). At `N = 1e5`/`1e6`, `P = 256`, eight
iterations, one batch per cell:

| `p` instantiation | shared | kernel at `N=1e5` | kernel at `N=1e6` |
|---|---:|---:|---:|
| 20, 128 threads / 128 rows | 38 992 B | 4 038.7 ms | 40 232.7 ms |
| 20, 256 threads / 256 rows | 50 768 B | 4 404.6 ms (+9.1 %) | 45 237.0 ms (+12.4 %) |

The two-block-per-SM variant wins, so `FT_CHUNK_LARGE` is 128. Fit-stage
throughput against the 8-thread CPU LM on the same cells (the host was busy with
other suites, so the CPU side is indicative rather than a clean baseline):

| p | N | P | CPU wall | GPU wall | fit speed-up | fallbacks |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1e5 | 32 | 2.10 s | 0.16 s | 13.2× | 0 |
| 16 | 1e5 | 32 | 8.72 s | 0.54 s | 16.1× | 0 |
| 20 | 1e5 | 32 | 14.43 s | 0.70 s | 20.7× | 0 |
| 20 | 1e5 | 256 | — | 4.04 s (15.8 ms/candidate) | — | 0 of 256 |
| 4 | 1e6 | 32 | 25.51 s | 1.47 s | 17.3× | 0 |
| 16 | 1e6 | 32 | 80.63 s | 4.98 s | 16.2× | 0 |
| 20 | 1e6 | 32 | 76.11 s | 6.98 s | 10.9× | 0 |
| 20 | 1e6 | 256 | — | 40.23 s (157 ms/candidate) | — | 0 of 256 |

Every `p = 17..20` fixture ends with status 0, so none of them falls back. The
only constant-count fallback left is `too_many_constants` at `p > 20`, which the
plan rejects before the kernel sees the candidate.

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

**Admission/promotion A/B (WP-169-D).** The pinned fit-heavy 154k cell
(4 islands, `--pop 256`, five generations, no screening, `--fit-iters 8`,
smooth finite operators) at equal 5 120 evaluations:

| arm | wall | best error | hypervolume | TTT (exact final × 1.02) |
|---|---:|---:|---:|---|
| `exact` CPU | 218.68 s | 1.8589e-2 | 0.5056 | gen 5 / 218.7 s |
| `gpu` + admission | 93.38 s | 1.0861e-2 | 0.5386 | gen 4 / 19.1 s |

The GPU arm is **2.34×** end-to-end. It screened 6 124 offspring, admitted
1 536, fitted 1 437 on the device and sent 99 to the CPU (**6.4 %** of admitted;
all `ill_conditioned`, every other cause zero), then refit 997 finalists
exactly. The whole fit stage was 32.00 s (34.3 % of wall; kernel 4.39 s) and GPU
duty was **96.8 %**. Hypervolume and time-to-target are both better than the
CPU arm at equal evaluations.

The measured fit share is **34.3 %, below the 50 % construction** the issue's
`>= 1.3×` clause names, so this cell does not evidence that conditional box even
though its measured e2e and duty exceed the numeric thresholds.

**3.9M hourly cell, equal evaluations.** The first probe (`pop 128`, one island,
two generations) failed quality: seed-constant screening admitted only `pop/4`
and the GPU arm reached best 0.4753 against 0.1612 for the exact arm. More
generations alone do **not** fix a screen that admits one quarter of the
offspring. Five generations at the same 640 evaluations, still one island:

| arm | wall | best error | hypervolume | TTT (exact final × 1.02) |
|---|---:|---:|---:|---|
| `exact` CPU | 1421.5 s | 9.6745e-3 | 0.7251 | gen 5 / 1421.5 s |
| `gpu`, old `pop/4` | 251.2 s | 3.7786e-2 | 0.5725 | not reached |
| `gpu`, all fittable | 444.6 s | 2.9472e-1 | 0.1948 | not reached |

The old policy screens 763 offspring, admits 192 (32 per generation), fits 178
on the device and falls back 14 (7.3 %, all `ill_conditioned`); fit share 48.7 %,
kernel 38.0 s, duty 94.6 %. The widened policy admits 553, fits 493 and falls
back 60 (10.8 %); fit share 71.9 %, kernel 52.6 s, duty 92.2 %. A single island
is dominated by mutation luck: the exact arm's late `sin`/`tanh` discovery did
not appear in either GPU trajectory, so a best-of-one comparison at 640
evaluations is not a stable quality gate.

The gate therefore uses four independent islands at the same total evaluation
count — `pop 64`, four islands, five generations (1 280 evaluations) — where
best-of-four is not decided by one lucky mutation. Both arms use seed 9, the
same operators and budget; `exact` is still the CPU reference and `gpu` is the
opt-in arm with all fittable offspring admitted:

| arm | wall | best error | hypervolume | TTT (exact final × 1.02) |
|---|---:|---:|---:|---|
| `exact` CPU | 1593.6 s | 3.3828e-2 | 0.5723 | gen 4 / 1200.0 s |
| `gpu`, all fittable | 865.2 s | **1.0861e-2** | **0.5762** | **gen 3 / 266.1 s** |

The GPU arm **beats the exact arm at equal evaluations** on all three quality
controls: best error is 3.1× lower, hypervolume is slightly higher (0.5762 vs
0.5723) and the target is reached one generation and 4.5× earlier. End-to-end
speed-up is **1.84×** (1593.6 s → 865.2 s). It screened 1 516 offspring,
admitted 1 014 (every candidate with constants; 502 had no constants), fitted
939 on the device and sent 75 to the CPU (**7.4 %**, all `ill_conditioned`;
every other cause zero), then refit 252 finalists exactly. The whole fit stage
was 569.6 s (**65.8 %** of wall; kernel 85.6 s) and GPU duty was **92.0 %**.
This cell meets the conditional construction (`fit ≥ 50 %`) with e2e ≥ 1.3×
and duty ≥ 85 %. A separate ignored diagnostic
(`diagnostic_gpu_fit_matches_cpu_at_one_million_rows`) confirms the kernel still
matches the CPU LM at 1 050 000 rows: status-0 candidates agree to
`1e-8`–`1e-14` relative SSE, and the one ill-conditioned candidate is labelled
and falls back.

**Overlap.** The overlap pipeline (`--pipeline-depth`) is not combined with this
arm yet: it scores a tree as soon as it is offered, before the barrier fit
batch, so the GPU-fit path uses synchronous generation batches.

### LogSquashed on the batched kernel (issue #229)

`--metric log_squashed` minimises the mean `ln(1 + r²)` — the Cauchy/Lorentzian
shape, Barron's robust loss at `α = 0`. It is non-convex and redescending: the
true Gauss–Newton curvature `ρ″(r)JᵀJ` is indefinite once `|r| > 1`, and the
loss has several local minima, so a fit is a *basin refinement*, not a global
solve. The kernel therefore runs **weighted LM/IRLS** in the same block as the
squared arm; nothing else in the scheduling changes.

Per row, at the constants the pass already evaluates (`r = f(x) − y`, `w` the
base `--weight`):

```
sc = w · 2/(1 + r²)          // MM/IRLS weight; 2 is ρ′(r)/r / (1/(1+r²))
A += (√sc·g)(√sc·g)ᵀ ,  b += √sc·g · √sc·r
loss += w · log1pf(r·r)      // ONE f64 add per row
```

* The weight enters only through the rank-one update, so `JᵀWJ` stays positive
  semidefinite where the true Hessian is not. `ρ′/r = 2/(1 + r²)`; the CPU
  `irls_fit` uses `1/(1 + r²)`, and the constant 2 cannot move the weighted
  least-squares argmin.
* The weight is refreshed **every sweep** (the CPU refreshes 4 times and then
  solves once), which is closer to true majorise-minimise and costs `iters`
  passes instead of about `2·iters`.
* The step, damping law (`λ₀ = 1e-3`, `/8`, `×6`, `10⁹`, clamp ±1e6), and the
  four undamped polish steps are the squared arm's, bit for bit.
* `sw` is formed in f64 from the f32 residual, and the trial loss is an f64
  sum in the existing single-owner, ascending-row reduction (no f64 atomics: an
  atomic's order can flip an accept/reject, and that decision is part of the
  trajectory). The per-row `ρ` uses the f32 `log1pf` (1 ULP); it is a
  **comparator, never a reported number**.
* A base `--weight` vector is uploaded with the f64 fit rows and multiplies the
  IRLS weight. A zero-weight row is skipped before any arithmetic, so a masked
  row cannot contribute `0 · inf = NaN`. Weighted **runs** still use the CPU
  evaluator, because the evaluator has no per-row weight array and would rank
  them on unweighted loss.

**Gate.** The robust objective is native in both kernels: `eval_kernel` and
`eval_pool_kernel` accumulate `Σ ln(1+r²)` and the host divides by the row
count, so the search ranks a `log_squashed` run on exactly the configured
objective. `gpu_blocker` admits a metric only when
`fit::DeviceLoss::for_metric` names a native loss; `log_squashed` passes because
of this scorer, and the #236 rule ("no proxy ranking") is unchanged.

**Conditioning gate.** Robust weights make the normal equations badly *scaled*
without making them ill-conditioned, so the squared arm's
`hi/lo ≤ 10⁴` Cholesky-diagonal gate is replaced for `log_squashed` by the
scale-free

```
ln det Ã = 2 Σ ln L_ii − Σ ln A_ii ,   Ã = D^{-1/2} A D^{-1/2}
```

which fails (status 1 → exact CPU refit) when the data no longer identifies the
parameters (`ln det Ã < ln 1e-12`) and no longer uses the absolute `1e-18`
pivot floor. Negative curvature of the true Hessian is **not** gated: `ρ″ < 0`
at `|r| > 1` is a property of the objective, not a defect.

**Verification and finalists.** `fit/gpu.rs` is metric-aware: a proposal is
accepted only if it improves the *configured* loss (a robust proposal that
makes SSE worse is exactly the case a squared gate would discard), and
`refit_finalists_exact` uses `fit_weighted_metric`, which starts from
`best = metric_score(tree)` and writes only improving trials. Reported numbers
come from the CPU `f64` rebuild like every other metric.

**Cost model.** Per accepted sweep the row cost is forward `m` nodes, adjoint
`≈ 2.5m`, the `(p²+p)/2 + p + 1` statistic adds, plus for the robust arm one
f64 divide (`2/(1+r²)`), one `sqrtf`, `p` extra multiplies and one f32 `log1p`
plus one f64 add. On `sm_86` MUFU-class ops run at roughly 1/8 the FMA rate, so
`log + rcp ≈ 12–16` FMA-equivalents: with `C(p) = 3 + (p²+p)/16` the relative
per-row cost is ≈6× at `p = 1`, ≈3× at `p = 8`, ≈1.8× at `p = 16` and ≈1.5× at
`p = 20`. Against the CPU the robust arm is *not* the expensive one: the CPU
`log_squashed` fit runs ~4 IRLS rounds × 2 sweeps plus an 8-sweep final solve
(≈16 passes plus scoring) against the squared arm's 8 + 4.

**Fallback labels.** Every CPU fallback keeps a cause; the robust arm adds
`metric_not_native` (a metric with no native loss reaching the fit path) beside
`unsupported_plan`, `too_few_rows`, `too_many_constants`, `ill_conditioned`,
`non_finite` and `launch`.

**Measured (paired, RTX 3060, this host).** Fit stage, `N = 50 000` rows with a
heavy-tailed target (10 % of rows at +15), 8 f64 iterations per candidate,
`--fit-iters 8`, the exact CPU IRLS arm versus the whole GPU batch (upload +
kernel + download) in one process:

| p | P | CPU wall | GPU wall (kernel) | fit speed-up | no worse | endpoint mean loss |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 6.49 s | 0.189 s (0.188 s) | **34.4×** | 64/64 | 5.420539e-1 = CPU |
| 8 | 64 | 12.14 s | 0.248 s (0.248 s) | **48.9×** | 64/64 | 5.420539e-1 = CPU |

The squared arm on the same machine at the same time measured 41.8× (`p=4`),
31.9× (`p=16`) and 31.9× (`p=20`) at `N = 100 000`, so the robust arm lands in
the same band: the CPU pays the transcendental cost too, in ~16 IRLS/LM passes,
against the kernel's 8 + 4. A repeat of the robust bench while the host was more
loaded measured CPU 19.1 s / 27.7 s with the kernel unchanged at 0.191 s /
0.221 s (96.6× / 124.9×) — device time is stable, the host arm follows host
load. The 11 `ill_conditioned` candidates in the end-to-end run below are the
only fallback cause observed; `metric_not_native` is 0 because the gate keeps
such runs off the path.

End to end (`bench_end_to_end` with `RCSR_BENCH_METRIC=log_squashed`,
`N = 20 000`, `pop 64`, 2 islands, 3 generations, 384 evaluations, seed 9, both
arms in one process):

| arm | wall | best mean `ln(1+r²)` | hypervolume | TTT |
|---|---:|---:|---:|---|
| `exact` CPU | 8.90 s | 8.943851e-1 | 0.6353 | gen 1 / 1.7 s |
| `gpu` + admission | **3.41 s** | **8.901082e-1** | **0.6616** | gen 1 / 0.5 s |

**2.61×** end to end, with a 0.5 % lower best robust loss, a higher hypervolume
and a 3.4× earlier time-to-target at equal evaluations. The arm screened 506
offspring, admitted 252, fitted 241 on the device and sent 11 to the CPU
(**4.4 %**, all `ill_conditioned`), then refit 127 finalists exactly in CPU f64.
The fit stage was 1.02 s (30.0 % of wall; kernel 0.29 s) and kernel-time duty
75.3 %. The device was shared with an active 10-hour production run and other
test suites, so both walls are inflated; the arms are paired within one process.

**Quality, limits.** On the convex affine fixture the GPU and CPU IRLS endpoints
agree to all printed digits (mean loss bit-equal, 64/64 candidates). The
non-convex risk the note pre-registered — the GPU landing in a different basin
from the CPU on heavy tails — is asserted only per fixture
(`gpu ≤ cpu × (1+1e-6)`), not measured over a corpus; graduated non-convexity
for finalists is the follow-up.

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

# DGX Spark / GB10: CUDA 12.9+ (13.0+ on DGX OS), unified memory on by default
RCSR_CUDA_ARCH=sm_121 cargo build --release --features cuda

# unified memory on any other target, e.g. to test the path on a discrete GPU
RCSR_CUDA_ARCH=sm_86 RCSR_GPU_UNIFIED=1 cargo build --release --features cuda

# sm_121 with plain device memory instead (not recommended: 128 GB is the point)
RCSR_CUDA_ARCH=sm_121 RCSR_GPU_UNIFIED=0 cargo build --release --features cuda
```

Plan and status:

1. **Native build.** `RCSR_CUDA_ARCH=sm_121`, CUDA 12.9+ `nvcc` (CUDA 13 on DGX
   OS). Implemented: the architecture is accepted by the validator, the build
   script reports the required toolkit instead of dumping nvcc output, and
   `tests/sm121_compile.rs` compile-verifies a real `sm_121` cubin with CUDA
   13.0.88 (see “`sm_121` compile-only validation” above). `sm_121` is a
   *consumer Blackwell* target; datacentre flags such as `sm_100` will not run
   on it. **The kernels have not been run on a GB10 here**: compilation is
   local and permanent, execution is hardware-gated.
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
| `sm_121` compilation (cubin + SASS + register/smem audit) | x86_64 host, CUDA 13.0.88, **no GPU** (`tests/sm121_compile.rs`) | **verified — compiled, not run** |
| `sm_121` execution on a GB10 (device, driver, GPU tests, benchmark) | needs a dated session on `sm_121` hardware | **hardware-gated — not run here** |
| Dataset larger than device VRAM through managed memory | needs a GB10 (or a >12 GB dataset) | **unverified** — the allocation path is compiled and exercised on `sm_86`, but not at that size |
| On-GPU FP64 verification of survivors | planned | **unverified** — survivors are re-checked by the CPU `f64` evaluator today |
| `log_squashed` native loss in the evaluator and fit kernel, weighted-LM parity with CPU IRLS, determinism, ill-conditioning/fallback labels, `--no-fit` no-op (#229) | RTX 3060, `cargo test --release --features cuda` (`src/gpu.rs` fit tests, `src/fit/gpu.rs`, `tests/gpu_fit.rs`) | **verified** (see the #229 section) |
| `log_squashed` quality of the robust arm against the CPU IRLS over a corpus (basin splits on heavy tails) | RTX 3060, paired fit-stage and end-to-end benchmarks | **measured on one workload** — 34–49× fit-stage, 2.61× end-to-end, no-worse 64/64 at `p=4,8`; the corpus-wide basin-split rate is not measured |

The compile gap is closed by the test and table above; it can be re-run on any
machine with CUDA ≥ 12.9. The runtime gap needs a DGX Spark (or any `sm_121` +
CUDA 13 machine), where the dated session should record:

```sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
RCSR_CUDA_ARCH=sm_121 cargo test --release --features cuda
RCSR_CUDA_ARCH=sm_121 ./target/release/rcsr-gpu-bench --data <set> --batch 8000

# first, the cheap portability check on the existing sm_86 build:
CUDA_FORCE_PTX_JIT=1 ./target/release/rcsr-gpu-bench --data <set> --batch 8000
```

then repeat the benchmark with a dataset whose f32 columns exceed device VRAM,
and compare the unified run with `RCSR_GPU_UNIFIED=0`. The benchmark prints the
device, the architecture and `memory : device|unified`, so the result records
which path produced it. The PTX-JIT run proves portability only; the native
`sm_121` build is what the performance claim needs. Neither run is claimed
here.

