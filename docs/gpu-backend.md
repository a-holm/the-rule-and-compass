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
    --batch 8000 --depth 5 --seed 5 --reps 25

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
| `RCSR_GPU_SYNC` | `1`/`true`/`yes`/`on` forces the blocking copy path for the batch arena (the labelled fallback, #285) | unset (pinned asynchronous path) |

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


## Projected transport (#294)

A wide store is a **read policy** problem before it is a device problem. When
`--data` is a `.rcsb` store and `--vars` and `--time` are given, the search
loads only the variables, targets, `--weight`, `--time` and `--group-by`
columns. `--no-project` restores the full-column load for an A/B run. Filters,
`--dtype` expansion and `--sample-rows` also keep the full path because they
need the whole header.

`Projection` resolves exact column names against the header, refuses a missing
or dictionary-only name instead of dropping it, stores header indices in header
order, and hashes `index:name:dtype;` with FNV-1a 64. The hash and the exact
column list are written to `--run-manifest` as `projection.{hash,columns,indices}`.
`--json` for `rcsr enrich` carries the same pair for its device read.

The device path does not build a `Table` at all. `ProjectedRing` owns two or
three `cudaHostAlloc` slots capped at 64 MB total, plus one reader thread that
`pread`s the projected blocks straight into the pinned slot. The main thread
enqueues one `cudaMemcpyAsync` per projected column on a copy stream and runs
the next group's segment-id and elementwise feature tile (`r*`, `lag_r1_*`,
`range1`, `clv1`, calendar) on a compute stream. A copy event guards slot reuse;
a compute event guards the slot's device writes. The only stream
synchronisation is the final join after the last group. A group larger than the
cap is refused with its byte count rather than silently shrinking the
projection.

The reader and the main thread share one slot state. A filled group owns its
slot until it is uploaded or dropped, the reader only ever waits for a free
slot, and `finish` or drop cancels the reader instead of draining it, so no
early exit can leave either thread waiting on the other (#352). Every wait for
the reader is bounded: `max(60 s, one slot at 1 MiB/s)` by default, or
`RCSR_RING_WAIT_MS`. When a wait expires, the transport fails with the ring
state (reader step and group, free slots, groups sent and received), `enrich`
prints one `# enrich: projected transport failed: ...` line and uses the
whole-column upload. A reader that does not exit within the bound after
cancellation is detached and its pinned slots stay allocated, so a stuck read
cannot hold the process. `RCSR_TEST_RING_STALL_GROUP=<g>` is a test hook that
stalls the reader before group `g`.

`rcsr enrich --gpu on` uses this transport by default. It reads only
`ts/open/high/low/close` plus optional `volume/quote_volume`; the transport
upload is measured in `gpu.transfer_bytes`. On a store that cannot be projected
(e.g. a text-only field required as a numeric kernel input) the historical
whole-column upload is used and reported, never silently.

`rcsr enrich --json` partitions `timing.feature_s` on the host thread into
read, context, GPU section, `gpu.write_s`, CPU pass and a non-negative
`feature_other_s` remainder. The device times `kernel_s`, `h2d_s` and `d2h_s`
are busy measures that may overlap that timeline (docs/ohlcv.md).

Measured on the RTX 3060 with the 6 400-row fixture: a 3-slot 7-column ring is
constant across 100 groups, tiled `r1/r5/lag/range/clv/calendar` columns are
bit-identical to the whole-column kernels, and the copy/compute event windows
overlap (`tests/gpu_ring.rs`). `RCSR_NO_PROJECTED_UPLOAD=1` forces the
historical whole-column upload for a paired run; it is a measurement escape
hatch, not a supported production mode.

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

The GPU never produces a number a user reads. The objective kernel computes one
**configured score objective**; the returned value is already the metric value
(the mean, the RSS + AIC penalty, the maximum or `1 − ρ`), so the search never
divides a raw sum by the row count. The exact order/rank statistics (#288) come
back from the batch entry point as the statistic itself. The reported numbers are still rebuilt by
the CPU in `f64` at the end.

#287 replaced the old three-entry loss gate with a total
`Metric -> ScoreDispatch` table (`metrics::score_dispatch`). A metric either has
a native `ScoreObjective` or a **named refusal**; it can never fall through to
squared error:

| `--metric` | native objective | notes |
|---|---|---|
| `squared` | mean `r²` | one pass |
| `absolute` | mean `|r|` | one pass |
| `log_squashed` | mean `ln(1 + r²)` | one pass, `log1p` |
| `signed` | `|mean r|` | one pass |
| `maximum` | `max |r|` | integer max on the `|r|` bits, lowest-row tie |
| `correlation` | `1 − |ρ|` | two-pass centred moment, cached `ȳ, S_yy` |
| `correlation_signed` | `1 − ρ` | same moment, signed |
| `aic`, `aicc`, `bic` | RSS + penalty | one pass; `k` per candidate |
| `median` | median of `|r|` (even `n`: two-middle mean) | exact `f64` batch scorer, #288 (≥ 4096 rows) |
| `interquartile` | mean of `|r|` on ranks `[⌊n/4⌋, ⌊3n/4⌋)` | exact `f64` batch scorer, #288 (≥ 4096 rows) |
| `rank_correlation` | `1 − |ρ_s|` from exact integer ranks | exact `f64` batch scorer, #288 (≥ 4096 rows) |
| `rank_correlation_signed` | `1 − ρ_s` from exact integer ranks | exact `f64` batch scorer, #288 (≥ 4096 rows) |
| `auc` | `1 − U/(n₊n₋)` from exact integer ranks | exact `f64` batch scorer, #288 (≥ 4096 rows, two-valued target) |
| `log_loss` | mean `max(−u,0)+log1p(exp(−|u|))` | stable softplus, `u=t·p` (#289) |
| `brier` | mean `(σ(p) − l)²` | proper Brier on the probability (#289) |
| `hinge` | mean `1 − t·(2σ(p) − 1)` | the bounded sigmoid **surrogate** (#289) |
| `hinge_ordinary` | mean `max(0, 1 − u)` | the SVM hinge (#289) |
| `hinge_squared` | mean `max(0, 1 − u)²` | squared SVM hinge (#289) |

The device kernel uses a fixed reduction tree, ascending row order and no
floating-point atomics, so the same candidate scores identical bits alone or in
a batch of 7 or 447. `maximum` compares the IEEE-754 bits of `|r|` and breaks an
exact tie on the lowest row. A non-finite prediction (or one above the same
`1e30` hard-magnitude limit used elsewhere) returns **`+inf` with status 2**,
never the old finite `1e30/n`. `score_batch`/`score_pool` return `f64` values
plus a per-candidate status; the legacy `eval_batch`/`eval_pool` f32 SSE path
remains for the fit and test APIs.

`median`, `interquartile`, `rank_correlation`, `rank_correlation_signed` and
`auc` are `ScoreDispatch::Exact` rows of the same table (#288): they are scored
by `GpuEvaluator::score_stat_batch` over flattened plans on the exact `f64` forward
pass, from the measured 4 096-row floor (see the #288 section below). `slope`
and the implicit metrics remain CPU. Their named reason is printed and recorded
in `backend_reason`, and so is the row floor for an `Exact` metric.
`log_loss`, `brier` and the three hinge forms are native since #289 — see
[Classification fitness](#classification-fitness-issue-289) below. Their
fitness, the label sign and (for the classification family) the per-row
weights live on the device; every reported number is still the exact CPU
rebuild.

The GPU path is used for single-output, full-sample runs whose metric has a
native score objective or an exact order/rank statistic. Multi-output (`--target y1,y2`), a refused metric and
`--predictor-samples` keep the CPU evaluator, because they need more than one
prediction per row, a different objective, or a different set of rows every
generation. `--weight` is admitted for every native metric with a weighted form:
the regression family (`squared`, `absolute`, `log_squashed`, `signed`,
`correlation`, `correlation_signed`, `aic`, `aicc`, `bic`) and the classification
objectives. The device carries the weight vector, divides by `Σw`, skips
zero-weight rows before the program runs and scores all-zero weights as `+inf`
with status 4. `maximum` has no weighted form and keeps the CPU evaluator; the
reason is printed in the one-line note.

The **fit** stage is a separate decision. `--fit-strategy gpu` moves the
constants of a rank (`rank_correlation` / `rank_correlation_signed`), median,
interquartile or maximum run on the device — the exact rank stage of
[#230](#exact-rank-correlation-fitting-issue-230), the exact median polish of
[#226](#median-absolute-error-fitting-issue-226), the middle-50 % window polish
of [#232](#interquartile-absolute-error-fitting-issue-232) and the exact
minimum-error polish of #227. Median, interquartile and the rank family are
**scored** natively too (#288), so for those metrics the backend is `gpu` and
the order/rank statistic is the fitness, not an SSE proxy; `maximum` is scored by
the #287 objective kernel. Below the 4 096-row scoring floor the run keeps the
exact CPU statistic (and can still use the device fit-only arm); that case
reports `cpu` with `backend_reason = "order/rank scoring: the sample is below
the 4096-row device floor"`, and `GpuSearch::native_metric`/`native_stat` are
inactive. The sort-based arms need at least
[`MEDIAN_GPU_MIN_ROWS`](../my-version/src/fit.rs) = 4 096 training rows
(median, interquartile, rank, AUC) and the maximum fit arm
[`MAXIMUM_GPU_MIN_ROWS`](../my-version/src/fit.rs) = 512, the measured
crossovers below which the CPU statistic is cheaper.

Fallback and the flags:

* With neither flag, a CUDA build with a usable device selects the GPU when the
  #287 table names a native scorer for the metric (subject to the row floor and
  the binary-target rule for AUC). Every other metric is scored on the CPU and
  prints the table's named reason (for example `# gpu: off (slope needs a
  cross-row first-difference pass)`) once; the resource line and the `--json`
  `backend` both say `cpu` and `backend_reason` carries the same cause.
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

### Exact order-statistic and rank scoring (issue #288)

The fit arms of #226/#227/#230/#232 already computed an exact device statistic;
#288 exposes it as a **scoring entry point over arbitrary candidates** and adds
the dispatch gate that admits those metrics to the search.

**Entry point.** `GpuEvaluator::score_stat_batch(plans, constants, stat)` takes one
[`GpuFitPlan`](../my-version/src/fit.rs) and one constant vector per candidate,
runs the exact `f64` forward pass from the resident rows, and returns one
`Option<f64>` per candidate for a [`ScoreStat`](../my-version/src/gpu.rs):

| `ScoreStat` | device work | exactness contract |
|---|---|---|
| `Median` | segmented sort of the `|r|` bit patterns, then the middle pick (even `n` averages the two middles) | bit-identical to `metrics::median_abs` |
| `Interquartile` | the same sort, mean of ranks `[⌊n/4⌋, ⌊3n/4⌋)` (all rows below `n = 4`) | fixed-order tree mean, `1e-12` relative |
| `Maximum` | integer max on the `|r|` bits plus the lowest maximiser | bit-identical, order-free |
| `RankCorrelation` | exact integer rank sums of #230, then `1 − |ρ|` | bit-identical to the exact `RankTarget` statistic |
| `RankCorrelationSigned` | the same sums under `1 − ρ` | bit-identical, orientation preserved |
| `Auc` | the same integer sums read as the Mann–Whitney `U` | bit-identical to `metrics::auc` for the same prediction vector |

The return value is tagged: `Some(None)` is a candidate whose prediction was not
finite and maps to `+inf` exactly like the CPU scorer; `None` is a whole-call
decline (allocation, launch, bad shape, or an AUC target that is not two-valued)
and sends the batch to the exact CPU path. No candidate can be left with a
fabricated finite value.

**Why AUC is exact without a label kernel.** For a two-valued target the cached
doubled centred ranks `B` are `nneg` on the positives and `−npos` on the
negatives, so `Σ_pos A = Σ A_i B_i / n` exactly and

```text
U   = (Σ_pos A + npos·nneg) / 2
AUC = U / (npos·nneg) = 1 − error
```

`Σ_pos A` and `npos·nneg` are exact integers, so `2U = Σ_pos A + npos·nneg` is
exact; `U` itself may be a half-integer when a tied positive/negative pair
contributes one half (`2U` odd, e.g. a constant prediction over an odd
`npos·nneg`). Both the integer and the half-integer are exact in `f64`; the only
rounding is the same single division and subtraction `metrics::auc` performs,
and the CPU's rank sum is exact for `n < 2^53`. The class sizes come from the
same exact `f64` target uploaded by `set_fit_data`; `binary_class_sizes`
verifies the two-run rank shape instead of assuming it, so a three-valued target
is refused. All-equal targets and constant predictions are exactly `0.5`
(chance).

**Determinism and waves.** The rank sums are exact `i128` additions, so they do
not depend on reduction order. The sort statistics depend only on the multiset
of residuals; ties share one integer and `−0.0` canonicalises to `+0.0` exactly
like the CPU's `==`. A call is split into device waves by candidate count only
(`RANK_WAVE_BYTES` is 512 MB, i.e. 64 B/row, so a 2.46 M-row wave holds a few
candidates); a candidate's value is identical alone, in a batch of 7 and in a
batch of 447. A test pins batch composition and repetition bit for bit.

**Dispatch gate.** The #287 table `metrics::score_dispatch` is the only
metric-to-statistic table. `median`, `interquartile`, `rank_correlation`,
`rank_correlation_signed` and `auc` are `ScoreDispatch::Exact(ScoreStat)` rows;
`ScoreStat::for_metric` reads that row and nothing else. `maximum` is a
`ScoreDispatch::Native` objective row (#287), so the search ranks it with the
objective kernel at any size; `ScoreStat::Maximum` stays as the exact `f64`
entry of `score_stat_batch` for direct callers and the parity tests. `gpu_blocker`
names every refusal:

* no scorer for the metric: the table's own reason (for example `slope needs a
  cross-row first-difference pass`);
* sample below the measured floor: `order/rank scoring: the sample is below the
  4096-row device floor`;
* AUC with a target that is not two-valued: `auc needs a two-valued target`;
* weighted `maximum`: `weighted rows: maximum has no weighted form`;
* weighted rows with a metric that has no native weighted scorer: `weighted rows
  need a native scorer with a weighted form`.

When admitted, `GpuSearch::evaluate_pending_stat` flattens each pending tree with
`gpu_fit_plan_reason`, scores the whole generation through `score_stat_batch`, and
runs the bounded CPU fallback pool for trees the flattening cannot represent.
The flat postfix and overlap-pipeline paths score compiled programs through the
objective kernel, so they are disabled for these metrics: the run stays on the
tree scorer rather than being ranked by SSE. A failed device batch returns an
error, the evaluator is dropped and the generation is re-scored on the CPU, as
for every other device failure.

**Measurements.** The reproducible paired bench is
`cargo test --release --features cuda --lib bench_score_batch -- --ignored
--nocapture` (`RCSR_BENCH_N`, `RCSR_BENCH_P` override the sample/candidates).
On the RTX 3060 at `N = 100 000`, `P = 8`, one wave and a 6.4 MB wave charge:

| statistic | device s/row/candidate | one-thread exact CPU s/row/candidate |
|---|---:|---:|
| `median` | 7.6e-9 | 1.13e-8 (rank-sum reference) |
| `rank_correlation` | 1.29e-8 (includes the first-call rank prepare) | 1.13e-8 |
| `rank_correlation_signed` | 9.3e-9 | 1.13e-8 |

Against the search's **8-worker** CPU scorer (ideal ≈ `1.4e-9 s/row`) the
sort-based path is ≈1× or slower at this size, as report
`01-native-scoring.md` §5 predicted (≈4.5 s each at `N = 2.456 M`, `P = 447`,
where a wave holds three candidates). The production radix histogram/select path
is the follow-up that removes the `2N` sort buffer and the 64 B/row wave; this
change is the correctness entry point and the dispatch gate, so the run-level
gain for rank/AUC runs is device residency, not a claimed speed-up. The
`N = 2.456 M`, `P = 447` cell is reproducible with `RCSR_BENCH_N=2456306
RCSR_BENCH_P=447` on a device with enough memory.

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

### Row ranges and the lane carry (#293)

`score_kernel` gives lane `l` (thread `l` of the block, `L = blockDim.x`, 256 for production runs) the rows `r` with `r mod L = l`, in ascending order, and folds the lanes with one fixed tree. `rc_score_rows` scores an explicit partition `0 = b0 < b1 < ... < bk = n` with the same decomposition:

- The lane of a row is `r mod L` in absolute rows. It does not depend on the partition, the chunk count or the grid.
- Per candidate, the carry holds `L` lane states (two f64 each) and an `int` bad flag. Maximum keeps its key bits and row in the two slots. The region is zeroed at the start of a batch.
- `score_rows_kernel<MODE, PASS>` applies one row range: lane `l` continues from its carried state over the rows of the range with `r mod L = l`. Correlation runs two passes with `score_mid_kernel` between them (mean prediction, lanes reset).
- `score_finish_kernel<MODE>` runs the resident tree and epilogue: `+inf` and status 2 on a bad row, status 4 for all-zero weights, the AIC penalty on the host. A weighted launch carries `Σ w·term` in lane A and `Σ w` in lane B (weighted correlation: `Σ w·dp²` and `Σ w·dp·dy`, with the weighted mean prediction from the mid kernel); the weighted target moments are computed once when the weights are uploaded.

The ranges form a left fold in row order, so every score kind returns the same bits as the resident kernel for any partition, including unaligned cuts and 1-row ranges. A lane index relative to the range start gives different bits on unaligned cuts. The production resident path keeps `score_kernel`.

The carry is `P x (16 L + 16)` bytes (4,112 B per candidate), allocated once through the ledger with tag `carry` and grown only when the candidate count grows. Tests: `gpu::carry_tests`.

`score_kernel` and `score_rows_kernel` fold rows through one `__device__` function (`score_step`), so the resident and the row-range paths cannot drift apart.

### Tiled residency (#293)

When the plan is `tiled`, `GpuEvaluator::new_tiled` (C: `rc_create_tiled`) keeps the column-major f32 feature columns on the host and allocates two device tile slots and two pinned staging slots of `tile_rows x vars x 4 B`. The target, label sign and weights stay resident by absolute row. Every score call runs the row-range kernels over the tiles:

- Per wave of candidates, the tiles are swept in order (twice for the correlation family). A wave holds `partials cap / 4,112 B` candidates; one wave covers the whole population when it fits the cap.
- For a tile, the host copies the `vars` column segments into the staging slot, the copy runs on a transfer stream behind the last kernel that read the device slot, and the kernel runs on the default stream behind the copy (events `ev_tile_copied`, `ev_tile_ready`, `ev_tile_used`). The next tile uploads while the current kernel runs.
- The lane of a row is its absolute row modulo `L`, so scores are bit-identical to the resident kernels for every tile height and wave size.
- The legacy evaluator, the batched fit, the exact rank and order-statistic scorers and the f64 twin are refused in a tiled context (the call returns a failure). The plan keeps them on the CPU.

`ArenaReport` carries `tile_rows`, `tiles` (tile kernel launches), `tile_uploads`, `tile_overlapped_uploads`, `tile_pinned_bytes` and `waves`. A `# gpu: tiled residency (...)` note names the tile height and the waves. The eval stage of the route planner adds the streamed tile bytes to its transfer estimate. Tests: `tests/gpu_tiled.rs`.

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
`--fit-strategy gpu` moves that loop into one CUDA block per candidate. It is
the default where admissible (see [Default fit arm](#default-fit-arm)); an
explicit `gpu` is always honoured:

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
only for a metric with a native device fit objective (squared/AIC family,
`log_squashed` since #229, and `absolute` since #222; scoring is a separate
#287 table), with `p <= 20`, at most
96 flattened nodes, no shared block constants, an analytic derivative for every
operator, and at least `32p` rows. The device objective is also a **CLI
contract** (#223, extended by #229/#222): an explicit `--fit-strategy gpu` with
any metric *without* a native device objective — that is, outside `squared`,
`aic`, `aicc`, `bic`, `log_squashed` and `absolute` — is a usage error (exit
code `2`) raised before any data is read, because the only alternative would be
a silent CPU fit of a different objective. The message names the objectives the
kernel minimises and every way out (`--metric squared`, `--metric
log_squashed`, `--metric absolute`, or a CPU fit strategy). The device-missing
fallback below is unchanged: it never claims the GPU arm and is recorded with
its cause.

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

The kernel returns a status. Status `0` is the ordinary success; `4`–`8` are
**labels on a finite answer, never CPU fallbacks**: `4` `rank_deficient` (the
undamped, Jacobi-scaled `ln det Ã` gate says a direction is unidentified, but
the damped step and the exact finalist refit own the result), `5` `recovered`
(the last accepted finite iterate survived a failed trial), `6`
`no_improvement` (the start was kept), `7` `start_unusable` (the start itself
has no finite value; the CPU arm could not move it either, so this is log-only)
and `8` `recovered_skipped` (a finite answer whose non-finite *gradient* rows
were skipped, up to the built-in row guard). Only status `1` (no finite step) and
`2` (no finite iterate at all) route to the exact CPU arm, together with
unsupported trees and a failed launch. Status `1`/`2`, unsupported
trees and a failed launch are refit by the exact CPU arm, seeded with the GPU
constants. Every CPU fallback is labelled by cause: `unsupported_plan`
(no analytic derivative, too many nodes, shared block constants or no
constants), `too_few_rows`, `too_many_constants` (`p > 20`),
`metric_not_native` (no device fit objective for the configured metric),
`ill_conditioned`, `non_finite`, `launch`, or — for the fit-only arms — the
`rank_declined` (#230), `median_declined` (#226), `interquartile_declined`
(#232) and `maximum_declined` (#227) causes (the device declined the call and
the exact CPU arm took over). `--fit-report` and the `gpu.fit` JSON object count
`screened`/`attempts`/`gpu`/`fallback`, the per-cause map, the
`status_by_label` map of finite device answers, `refits`, the kernel `seconds`
and the whole fit stage's `host_seconds`; the run manifest records the same
accepted/fallback/refit totals.

**Parallel fallback batch and its bound (#255).** The exact CPU refits are
independent per candidate and are therefore split across scoped workers with
the same fixed-order pattern as the finalist refits in `full_data_archive`:
`ceil(n / workers)` contiguous jobs per worker, one thread-local block registry
per worker, and the fitted trees written back by their original island/slot in
admission order. The result is bit-identical to the old serial `for` loop for
any worker count; `search::tests::fit_fallback_is_identical_for_one_and_many_workers`
pins one versus eight workers and compares every tree bit with the serial arm.
The worker count is the same bounded pool setting as the evaluator fallback
(`RCSR_CPU_FALLBACK_THREADS`, default `min(cores, 8)`, maximum 64).

`--fit-fallback-max N` is a hard per-generation cap on those refits (default
`4096`, `0` = unbounded). The first `N` fallbacks in deterministic admission
order are refit; any remainder keeps the finite constants the device produced.
That case is **reported, never silent**: `gpu.fit.fallback_bounded` and each
generation's `fit_fallback_bounded` count it, and `fallback_max` records the
configured bound. The default is above every generation of the recorded cell
(at most 540 admitted), so normal runs are unchanged; the cap exists so one
pathological generation cannot spend minutes in an unbounded CPU tail.
`gpu.fit.fallback_seconds` reports the measured fallback subset, and each
generation carries `fit_fallback_seconds`, `fit_fallback_bounded` and
`fit_fallback_workers`. `fallback_seconds` is a **subset** of `host_seconds`
(kernel + launch + fallback + write-back), so the phase ledger still closes
without double counting.

Measured on the fit-heavy pinned cell (`tests/gpu_fit.rs::bench_end_to_end`:
154 000 rows, 21 variables, 4 islands × pop 256, 5 generations, `--fit-iters 8`,
seed 9, `--metric squared`) on an RTX 3060 with an Intel i7-3770K (4 cores / 8
logical CPUs), same binary and seed, only `RCSR_CPU_FALLBACK_THREADS` changed:

| fallback workers | fit host | fallback subset | wall | best error | archive hash |
|---:|---:|---:|---:|---:|---|
| 1 (serial reference) | 35.00 s | 30.13 s | 93.54 s | 1.086148e-2 | `c257d304ee910746` |
| **8** | **11.49 s** | **6.93 s** | **63.10 s** | **1.086148e-2** | **`c257d304ee910746`** |

The exact fallback loop is **4.35×** faster, the fit stage **3.05×** and the
end-to-end run **1.48×**, with the same per-generation trajectory, archive best,
archive hash and 996 finalist refits. The untouched serial binary recorded 23.33 s
fit host at 67.02 s wall on the same cell in a separate run, so the change does
not borrow time from another phase; run-to-run host load moves the absolute
numbers, which is why the worker pair above is the controlled comparison. The
recorded 10 h cell is fallback-dominated (95.40 % of process wall in 5 835
refits), so at this measured ratio its fallback residue projects from 36 362 s
to about 8 400 s; that projection is not a run on that cell and is labelled as
such.

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

**Overlap (#286).** The overlap pipeline (`--pipeline-depth`) now carries the
fit arm where doing so cannot change the trajectory: at `N >= 1 048 576` rows
the serial admission set is already every pending candidate with constants, so
the queue can score an earlier candidate's seed constants, fit it on the device
(or the structural CPU residual) and re-score it while the island threads keep
generating later candidates. Below that row floor, with a fitting subset, and
for the metric-specific gated arms (`log_loss`, rank, median, interquartile,
maximum), the synchronous generation batch is still used. Selection and the
archive still wait at the generation barrier, so all errors, the archive hash
and the exact CPU finalist are unchanged; `--pipeline-depth 0` remains the
synchronous schedule.

**Zero diagonal (#399).** `ft_solve` freezes a slot whose normal-equation diagonal is exactly `0.0` on every arm. A constant with a zero adjoint on all active rows cannot move the prediction, so its row and column become the identity, its right-hand side is `0` and its step is `0`. The other constants are still fitted, as the CPU ridge solve (`fit.rs::solve`, ridge `1e-12`) does. A negative or non-finite diagonal is still status `1`. `tests/gpu_fit.rs::zero_adjoint_constant_is_frozen_and_the_rest_are_fitted` pins it.

### Default fit arm

A run that names no `--fit-strategy`, or names `auto`, resolves its fit arm once, before generation 0 and before the device memory plan. An explicit arm (`exact`, `linear`, `varpro`, `stream`, `gpu`) is never resolved, and a resume keeps the saved arm and admission. The resolved arm is `gpu` when all of these hold, checked in this order; the first failing factor is the labelled reason:

1. `metric_not_least_squares`: the metric is `squared`, `aic`, `aicc` or `bic`.
2. `no_fit`: `--fit-iters` is above 0.
3. The backend reason: the GPU evaluator runs the search (`--cpu/--no-gpu`, `no usable CUDA device`, `built without --features cuda` and the evaluator refusals keep the CPU).
4. `device_memory`: the memory plan has no `fit` refusal for the f64 fit rows.
5. `dispatch_cost`: the run-level cost model predicts a win.

When one fails, no flag keeps `exact` and `auto` keeps the structural CPU `auto`, so every CPU-only run (non-CUDA build, `--cpu`, no device, an ineligible run) keeps its archive and JSON byte for byte.

A resolved `gpu` arm admits every candidate with constants at every row count (`gpu.fit.admission` is `all`). An explicit `--fit-strategy gpu` keeps the `max(32, pop/4)` admission below 1 048 576 rows (`screened`). The admission is saved in checkpoints (`config fit_admission all`; a missing line means `screened`) and is part of the config hash.

The cost model is pure and reads no timing. With `I` island threads, `C` cores, `W = min(C, 8)` fallback workers, `s = 1` when every configured operator has a device fit opcode and no custom block is defined (else `0`), `k = 0.05` (device cost of one fit-row relative to one CPU thread), `E = 1.32` (overhead of the batched arm) and `h = 0.3` (required margin):

```text
cpu = 1 / min(I, C)
gpu = (s * k + (1 - s) / W) * E * (1 + h)
device iff gpu < cpu
```

With `W = 8`: without full operator coverage (the default operator set, because `if` has no fit opcode) the device is chosen up to 4 islands and `exact` from 5 (`dispatch_cost`); with full coverage the device is chosen up to 11 islands. There is no row floor. The per-candidate floor of `32p` rows in the fit stage is unchanged. This run-level rule is separate from the per-candidate stage model in `docs/dispatch.md`.

Evidence for the default is a paired comparison against `exact` on the GPU evaluator at equal generations (default operators, `--fit-iters 8`): admitting every candidate matched `exact` in test error and hypervolume and ran 1.5x to 5.1x faster; the screened fraction was faster still but lost on hypervolume. Fallbacks in these runs were `unsupported_plan` only, from `if`.

| cell (train rows, islands x pop, generations) | arm | seeds | speed-up | test error better/worse/tie | hypervolume better/worse |
|---|---|---:|---:|---|---|
| 300, 1 x 200, 20 | screened | 3 | 3.75x | 3/0/0 | 2/1 |
| 300, 1 x 200, 20 | all | 3 | 2.60x | 2/1/0 | 2/1 |
| 1 500, 1 x 200, 20 | screened | 13 | 6.11x | 3/9/1 | 3/10 |
| 1 500, 1 x 200, 20 | all | 13 | 5.08x | 6/5/2 | 9/4 |
| 1 500, 4 x 50, 20 | all | 13 | 1.80x | 6/6/1 | 5/7 |
| 15 000, 1 x 200, 10 | screened | 3 | 7.75x | 1/2/0 | 1/2 |
| 15 000, 1 x 200, 10 | all | 3 | 4.20x | 2/1/0 | 2/1 |
| 115 500, 4 x 64, 3 | all | 2 | 1.52x | 0/2/0 | 0/2 |

Provenance. The non-quiet stderr line `# gpu-fit: default -> gpu (squared, admit all)` (or `-> exact (dispatch_cost)`) appears when the GPU evaluator runs the search. On a GPU run `--json` `fitting` carries `strategy_source` (`default` or `auto`) and `strategy_reason` (`admissible` or the failing factor). `--run-manifest` carries `fit_resolution` and the development log `run_start` carries `fit_strategy_source` and `fit_strategy_reason`, both for every resolved run.

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
  row cannot contribute `0 · inf = NaN`. Weighted **runs** are scored on the
  device with the same weights.

**Gate.** The robust objective is native in both kernels: `eval_kernel` and
`eval_pool_kernel` accumulate `Σ ln(1+r²)` and the host divides by the row
count, so the search ranks a `log_squashed` run on exactly the configured
objective. `gpu_blocker` admits a metric only when
`metrics::score_dispatch` names a native `ScoreObjective` (#287);
`log_squashed` passes because of this scorer, and the #236 rule ("no proxy
ranking") is unchanged.

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

`--metric signed` now has a native one-pass score objective (#287); its exact
fit arm remains the host-side projection of #228 (the fit section below).

### Absolute (MAE) on the batched kernel (issue #222)

`--metric absolute` minimises the mean absolute residual `(1/n) Σ |r|`. It is
convex in the prediction but not in `c`, and its subgradient is set-valued at
`r = 0`; `r = 0` is also where the classical Lawson weight `1/|r|` is infinite.
The arm is therefore **weighted LM/IRLS with true-`|r|` acceptance**, not an
SSE fit with an MAE report.

Per row, at the constants the pass already evaluates:

```
sc = w / max(|r|, 1e-6)          // L1 majoriser; 1e-6 is the CPU floor
A += (√sc·g)(√sc·g)ᵀ ,  b += √sc·g · √sc·r
loss += w · |r|                  // the accepted objective, f64 sum
```

* `rounds = min(iters, 4)` outer IRLS rounds of `iters/rounds` damped sweeps
  mirror the CPU `irls_fit` split; the weights are re-derived at the start of
  every round and after every accepted step, so each sweep is one majorise step.
* Every trial is scored by the configured `Σ|r|` (f32 per row, f64 reduction).
  If a trial is not better, the step is **halved up to twice** (Armijo,
  `ft_loss` passes only) before it is rejected; the f64 polish gets one exact
  half-step. A lower weighted surrogate can therefore never be accepted while
  the true MAE rises.
* `sc` is computed in f64 from the f64 residual; the row scaling `√sc` enters
  the existing rank-one update, so `A` stays PSD. The f64 polish re-assembles
  the same weighted system on the exact f64 rows.
* The conditioning gate is the scale-free `ln det Ã` test of #229/#286, used by
  every arm: L1 weights span six decades by construction and the squared arm's
  old `hi/lo ≤ 10⁴` compared derivative-column units. #286 Jacobi-equilibrates
  the step (`Ã δ̃ = −D^{-1/2}jᵀr`, `δ = D^{-1/2}δ̃`) and labels a rank-deficient
  finite answer `rank_deficient`; only a system with no finite step remains
  status `1` and goes to the exact CPU IRLS.
* The base `--weight` vector multiplies the L1 weight exactly as in #229; a
  weighted *run* is scored on the device with the same weights.

**Gate.** `eval_kernel`/`eval_pool_kernel` accumulate `Σ|r|` for the new loss
code (`FitObjective::Absolute = 4`), so the search ranks a `--metric absolute`
run on MAE itself. `fit::fit_loss` names the
objective, `defers_fit_to_gpu` admits it, and `verify_proposal` +
`refit_finalists_exact` score and refit with `fit_weighted_metric(.., Absolute)`
— the exact CPU L1 IRLS. A squared-only backend still declines `absolute`
(the `BatchedFitter` default), so a device can never answer it with SSE.

**Degeneracy and fallback.** Zero residual rows are legitimate but make the
majoriser unbounded; the `1e-6` floor and the `ln det Ã` test decide between an
exact fit and a labelled `ill_conditioned` fallback. Non-finite rows are status
`2`; every other CPU fallback keeps its cause label from #169/#229.

**Cost model.** Per accepted sweep the row cost is the forward/adjoint of the
squared arm plus one f64 divide (`1/max(|r|,eps)`), one `sqrtf` and `p` extra
multiplies; there is no transcendental, so the absolute arm costs the squared
arm plus ~5 % of the statistic work, against the CPU's ~4 IRLS rounds ×
`iters/4` sweeps plus a final `max(iters,4)`-sweep solve. On `N` rows, `p`
constants and `P` candidates:

```
T_cpu(L1)  ≈ P · (S · n · a + (4 + S) · n · (m + p²/2) · b)      S = iters
T_gpu(L1)  ≈ P · n · S · c / B + launch                          c ≈ 3 + (p²+p)/16
```

with `B` the derated candidate-rows/s of #169; below `n ≈ 2 000`, or `P < 2·SM`
where one block per candidate leaves the device half empty, the CPU arm is the
cheaper one and stays. The measured paired numbers are recorded in the table
below and in #222.

**Measured (paired, RTX 3060, this host).** Fit stage, `N = 100 000` rows with
10 % of rows at +15 (the `bench_batched_fit_absolute` fixture), 8 iterations
per candidate, the exact CPU L1 IRLS versus the whole GPU batch (upload +
kernel + download):

| p | P | CPU wall | GPU wall (kernel) | fit speed-up | no worse | mean endpoint MAE (CPU / GPU) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 16.19 s | 0.499 s (0.499 s) | **32.4×** | 64/64 | 1.500500 / 1.500495 |
| 8 | 64 | 30.33 s | 0.851 s (0.850 s) | **35.7×** | 64/64 | 1.500522 / 1.500500 |

The L1 arm has no transcendental, so it is not slower than the #229 robust
arm (34–49× on its fixture); the GPU mean is slightly *better* than the CPU
because the outer IRLS rounds and Armijo steps land closer to the kink.

End to end (`bench_end_to_end`, `RCSR_BENCH_METRIC=absolute`, `N = 20 000`,
pop 64, 2 islands, 3 generations, 384 evaluations, seed 9, 10 % outliers, both
arms in one process):

| arm | wall | best mean `|r|` | hypervolume | TTT |
|---|---:|---:|---:|---:|
| `exact` CPU | 12.23 s | 6.186247e0 | 0.005127 | gen 1 / 2.6 s |
| `gpu` + admission | **3.45 s** | **6.176220e0** | 0.005127 | gen 1 / 0.7 s |

**3.55×** end to end at equal evaluations, with a 0.16 % lower best MAE and a
3.7× earlier time-to-target. The arm screened 506 offspring, admitted 252,
fitted 237 on the device and sent 15 to the CPU (**6.0 %**, all
`ill_conditioned`; every other cause zero), then refit 130 finalists exactly in
CPU f64. The fit stage was 1.39 s (40.4 % of wall; kernel 0.28 s) and
kernel-time duty was 81.6 %.

### LogLoss on the batched kernel (issue #234)

`--metric log_loss` is the mean binary cross-entropy
`−1/n Σ [l ln q + (1−l) ln(1−q)]` with `q = σ(p)`, the model output `p` read as
a **logit** and `l = [y > (min y + max y)/2]` (#113). Unlike `log_squashed`
this is a *likelihood*, not a robust loss: the optimum is finite, it fixes the
scale and the offset, and its Hessian `JᵀWJ` with `W = q(1−q) ≤ 1/4` is
positive definite — the one classification metric the note in
`arbeidsmappe/crypto-rcsr/gpu-metrics/classification.md` calls optimisable
without a surrogate.

**Why the arm was fit-only, and what #289 changed.** The CPU fitness of
`log_loss` is one `O(n)` pass, while the CPU fit (`logistic_irls`) is up to 16
rounds of `tree.clone()` + two LM sweeps + one metric pass. Measured on the
pinned cell below, the fit owned **97.5 %** of a `log_loss` run's CPU work — the
opposite of the AUC arm, where the exact fitness owned 70–77 % of the wall and a
fit-only arm could not pay for itself (#233). The original #234 arm was therefore
admitted for the **fit** only. #289 closes the remaining gap with a label-aware
per-row scorer (`ScoreObjective::LogLoss`, [Classification
fitness](#classification-fitness-issue-289)): the search now ranks
log-loss/Brier/hinge on the device, and only the reported numbers stay the exact
CPU rebuild. `DeviceLoss::for_metric(LogLoss)` is still `None` — the regression
row loss has no label semantics — so the score objective is a separate
`ScoreObjective` in the #287 table with disjoint launch codes; a fit launch can
never be mistaken for a score launch.

**The step is the CPU's step, in the same kernel.** With the label sign
`t = 2l − 1`, the margin `u = t·p` and `q = σ(u)` clamped to
`[1e-9, 1−1e-9]` exactly like `logistic_irls`:

```
sw  = w · q(1-q)            // Fisher weight, ≤ 0.25
resp = −t/q                 // pseudo-response offset, |resp| ≤ 1e9
A  += (√sw·g)(√sw·g)ᵀ ,  b += √sw·g · √sw·resp
loss += w · [max(−u,0) + log1p(exp(−|u|))]
```

The accumulator forms `Σ (s·g)(s·resp) = Σ sw·g·resp = −Σ t(1−q)·g`, which is
exactly the gradient of `Σ ln(1 + exp(−t·p))` that `ft_solve`'s
`A δ = −Jᵀr` consumes, and `A` is the same `JᵀWJ` the CPU's pseudo-response
`z = p + t/q` produces. Nothing else in the kernel changes: same layout, same
`λ` schedule, same four-step f64 polish, same statistic ownership.

* **Stability.** `q` is formed from `u` in f64 with the branch
  `1/(1+e^{−u})` for `u ≥ 0` and `e^u/(1+e^u)` otherwise, so no step
  overflows at `|u| = 10⁶`; the loss is the Lin–Lin–Weng form
  `max(−u,0) + log1p(exp(−|u|))`, which never overflows and is exact for small
  `|u|` where the naive form cancels. Both row factors are bounded by the clamp,
  so neither an f32 product nor `--use_fast_math`'s flush-to-zero can hide a
  row: `√sw ≤ 0.5`, `|resp| ≤ 1e9`.
* **The label sign is uploaded once** (`rc_set_fit_sign`), because a
  one-block-per-candidate kernel cannot reduce the target midpoint over the
  sample. A log-loss launch **without** an uploaded sign is refused on the
  device, so the arm can never run on another objective, and the host refuses a
  **single-class** sample (`single_class`), whose unconstrained optimum is at
  infinity.
* **Class imbalance is not reweighted.** The configured metric is the
  *unweighted* mean cross-entropy, so balancing the classes would silently
  change the objective (the balanced logistic is only the right surrogate for
  `auc`). Imbalance is a conditioning problem instead: at 1 % positives the
  optimum is finite, `q(1−q)` spans decades and the scale-free `ln det Ã` gate
  is what decides identification. `q(1−q)` never reaches zero, so a saturated
  row keeps a (tiny) weight rather than disappearing from the Hessian.
* **Metric-aware acceptance.** After the batch, every accepted proposal is
  judged by the exact `f64` `log_loss` at the proposal against the exact value
  at the starting constants (`fit::gate_device_proposal`). Worse → revert and label
  `log_loss_worse`, non-finite → revert and label `non_finite`. Finalists are
  refit by `fit_weighted_metric` in CPU `f64` and every reported error is the
  exact CPU log-loss of the stored tree.

**Fallback labels.** Beside the existing causes the arm reports
`single_class` (one class in the fit sample) and `log_loss_worse` (the kernel's
constants lost the exact comparison). `ill_conditioned` is the common one on
binary data, and it is the *correct* answer: a candidate that separates the fit
rows has no finite MLE, so the kernel's determinant gate declines it and the
exact CPU arm keeps the candidate.

**Measured (paired, RTX 3060 shared with a 10-hour production run).** `N =
20 000` rows, 5 % positives, 15 % label noise, `pop 64`, 2 islands, 5
generations, **640 evaluations**, `--fit-iters 8`, `--fit-strategy gpu`:

| seed | CPU wall | GPU wall | speed-up | CPU best log-loss | GPU best log-loss | Δ | CPU hv | GPU hv | fallback | kernel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 93 | 21.80 s | 9.17 s | **2.38×** | 0.099964 | 0.105188 | +5.2e-3 | 0.5496 | 0.5950 | 41 (10.9 %) | 0.34 s |
| 94 | 24.96 s | 6.37 s | **3.92×** | 0.097171 | 0.093748 | −3.4e-3 | 0.7361 | 0.7295 | 38 (10.3 %) | 0.38 s |
| 95 | 22.76 s | 7.24 s | **3.14×** | 0.104713 | 0.093063 | −11.7e-3 | 0.5798 | 0.5952 | 38 (10.2 %) | 0.35 s |

Both walls carry the shared device and host load; the arms are paired in one
process and each seed is deterministic (the same archive and best value on every
repeat). Quality is a wash, if anything slightly positive: the GPU arm is better
on two of three seeds and on the hypervolume in two of three, and at
`--fit-iters 16` it is better by 5.4e-3 at 3.06×, at `--fit-iters 4` better by
1.0e-3 at 2.62× — the one worse seed is a trajectory difference at equal
evaluations, not a systematic loss.

**Where the time goes** (seed 93): the CPU arm spends 40.70 s of CPU work in the
fit (97.5 %) against 1.06 s in the fitness. The device arm's fit stage is
6.38 s of wall for 376 admitted candidates — **17.0 ms per candidate** against
**63.6 ms** on the CPU, of which 0.92 ms is the kernel (**61–69× on the
kernel**) — and the rest is the exact `f64` gate (two metric passes per
proposal), the 41 labelled CPU fallbacks and the write-back. In the #234 measurement the arm also
scored every candidate's fitness on the CPU (0.74 s, 8.1 % of wall); #289 now
removes that term by scoring classification natively on the device, so the same
fit-stage gain is no longer capped by CPU fitness.

**Tests** (`tests/logloss_gpu.rs`): the label rule and its subset
independence, single-class counting, the constant model against the closed form
`ln(n₊/n₋)` on the device, parity with the CPU `logistic_irls` on the objective
(relative ≤ 1e-6), extreme logits (±300, +5 000), determinism and batch
independence, the single-class refusal, a 28/32/64-iteration termination guard,
the proposal gate (worse / non-finite / kept), exact-CPU-reported errors at the
search level, the `--no-fit` no-op, and the paired measurement as an ignored
benchmark.

### Classification fitness (#289)

The #287 score table now has a label-aware block for long-only binary
classification. The objective is the **configured metric**, never squared error
and never a bounded surrogate with a different name:

```text
log_loss        w · [ max(−u, 0) + log1p(exp(−|u|)) ]      u = t·p
brier           w · (σ(p) − l)²
hinge           w · (1 − t·(2σ(p) − 1))                    bounded sigmoid
hinge_ordinary  w · max(0, 1 − u)
hinge_squared   w · max(0, 1 − u)²
```

`hinge` keeps its historical name and formula and the docs now say explicitly
that it is a **bounded sigmoid surrogate**, not the SVM hinge; `hinge_ordinary`
and `hinge_squared` are the separate SVM forms. A constant model has
`b* = ln(n₊/n₋)` under log-loss and `σ(b*) = n₊/n` under Brier, and the tests
pin both optima.

**Codes and dispatch.** The classification objectives occupy the score codes
10–14, after the regression block 0–9 (#287); the fit `RC_LOSS_*` codes remain
0–5. `metrics::score_dispatch` maps every classification metric to one of them,
so `auc` stays a **named** refusal (`auc needs the exact rank scorer with
labels (#288)`) rather than the old generic "classification metric" gate. The
CPU reference is `metrics::score_native_weighted`, which the kernel mirrors.

**Numerics.** `log_loss` is the Lin–Lin–Weng softplus, which never overflows and
is exact for small `|u|`; the CPU `metrics::log_loss` uses the same form, so an
extreme logit (`|p| = 10⁶`) is the analytic value on both sides rather than an
arbitrary `1e-12` probability clamp. `σ` is the branch form
`1/(1+e^{−x})` / `e^x/(1+e^x)`. Per-row arithmetic is `f32` (this evaluator's
contract); the accumulator is a **fixed-order `f64` shared-memory tree**, never
an atomic and never a `f32` sum. The per-row weight product and `Σw` are `f64`.

**Weights and labels.** The host uploads the per-row weight vector and the label
sign once per sample (`rc_set_score_weights`, `rc_set_fit_sign`; the same sign
vector serves the log-loss fit). The weighted midpoint and class counts are
taken over positive-weight rows, so a masked row cannot relabel a row that
counts. A row with `w = 0` is skipped **before** `rc_exec_row`, so an undefined
branch in it contributes nothing; a positive-weight non-finite prediction
yields `+inf` with status 2, and an all-zero weight vector is status 4
(`AllZeroWeight`). The kernel divides its weighted sum by `Σw` in `f64`, exactly
`metrics::score_weighted`. A classification launch without an uploaded sign is
refused (negative time), so the search takes the exact CPU scorer rather than
silently ranking by MSE.

**Determinism.** One owner block per candidate, fixed row stride, fixed thread
count from `N`, and a fixed reduction tree: a candidate's score is bit-identical
alone, inside a 512-candidate batch, and on a repeated call. The integer paths
are unaffected; `f32` device scores are not claimed to be bit-identical across
devices or toolkits, and every reported number is the exact CPU `f64` rebuild.

**Tests** (`tests/classification_gpu.rs`): the objective map and disjoint
launch codes, weighted duplicate-row parity and `Σw` scaling, masked
target/prediction invisibility, constant-model optima, the metric-aware proposal
gate (a proposal that lowers log-loss but raises Brier is rejected under
`--metric brier`; a proposal that lowers log-loss but lowers AUC is rejected
under `--metric auc`), the labelled `auc` refusal, device-vs-CPU parity on the
same `f32` rows (relative ≤ 1e-6) and against the `f64` model (≤ 1e-5), no rank
inversion, weighted `Σw` and batch determinism, the zero-weight `ln(-1)` skip,
and a native-Brier search whose archive is the exact CPU rebuild. The paired
native-fitness vs CPU scorer benchmark is an ignored test.

**Measured cost** (`bench_classification_fitness`, RTX 3060 shared host,
`P = 32` models, 5 repeats; both arms score the same rows, the device uploads
them once and the CPU arm pays the per-candidate prediction walk plus the metric
pass, which is exactly the cost the search paid before #289):

| `N` | objective | weights | CPU | device | kernel | speed-up | CPU row-evals/s | device row-evals/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 154 000 | `log_loss` | no | 1.889 s | 0.0085 s | 0.0069 s | **222×** | 1.304e7 | 2.895e9 |
| 154 000 | `brier` | no | 1.513 s | 0.0092 s | 0.0069 s | **165×** | 1.628e7 | 2.691e9 |
| 154 000 | `log_loss` | yes | 2.214 s | 0.0112 s | 0.0085 s | **197×** | 1.113e7 | 2.197e9 |
| 2 456 306 | `log_loss` | no | 30.30 s | 0.0809 s | 0.0771 s | **374×** | 1.297e7 | 4.856e9 |
| 2 456 306 | `brier` | yes | 27.11 s | 0.0930 s | 0.0904 s | **292×** | 1.449e7 | 4.227e9 |

This is the scorer stage only, not an end-to-end search claim; the best value
agreed within 1e-5 relative to the CPU arm (f32 rows). The device threshold is
clearly above the measured CPU path at both sizes, so the 8.1 % fitness share
from the #234 cell is removed rather than moved.

### Metric-aware verification and the signed arm (#228)

`fit/gpu.rs` judges every proposal on the **configured** objective (the #229
change), and since #228 it also guarantees *where the proposal came from*:

* `BatchedFitter::fit_batch_metric` receives the metric. Its default
  implementation delegates for the squared family (`squared`, `aic`, `aicc`,
  `bic`) and returns `None` for every other metric, so a backend that only knows
  squared error can never answer `signed`, `absolute`, `median`,
  `interquartile`, `maximum`, `correlation`, … . `BatchedGpu` overrides it with
  the device loss registry (`fit::fit_loss`) plus the signed warm start, and the
  exact median/interquartile order-statistic polish (#226/#232) and the exact
  maximum-error polish (#227) below.
* `verify_proposal` completes a `signed` proposal with the exact `f64`
  constrained-LS projection (`fit::signed_projection`) **before** comparing, and
  accepts only when the achieved `|Σ wᵢ rᵢ| / Σ wᵢ` is finite, no worse, and
  either certified on the manifold or strictly better. A model that cannot move
  the mean is labelled `not_identifiable` and keeps its constants — never
  accepted on an equal objective, which would be a squared answer in disguise.
* Statuses stay labelled: `gpu`, `unsupported`, `non_finite`, `worse`,
  `not_identifiable`, `no_backend`.

`BatchedGpu` is the CUDA LM kernel as a `BatchedFitter`; it fits on the
evaluator's resident `f64` rows (and the resident weights) and returns the
kernel's status-0 proposals. For `signed` those constants are only the **warm
start**: the host projection decides, exactly as `docs/optimisation.md` §3.0.1
describes. That is deliberate — the projection is two passes on `p` constants
and must be exact `f64`, so putting it on the device could only add rounding,
not speed on the row sweep the kernel has already paid for.

**The search still keeps `--metric signed` off the device.** An explicit
`--fit-strategy gpu --metric signed` is a usage error (`gpu_fit_metric_error`),
because `fit::fit_loss` is `None` for it: the fit could be warmed up on the
device, but the per-generation ranking would fall back to MSE, which is exactly
what #236's gate forbids. `tests/gpu_fit.rs::signed_arm_projects_the_device_proposal_exactly`
exercises the device-warm-start arm directly against the real kernel; the search
path opens when a native signed loss lands.

**Cost model.** With `n` rows and `p` constants a signed CPU fit costs `iters`
LM assemblies plus `min(iters,12)·p·6` polish evaluations (each a full row
sweep); the projection replaces the second term with **two** assemblies (one per
round, one round for affine trees) plus an `O(p³)` factorisation and two `O(p²)`
substitutions:

```text
T_cpu_signed   ≈ (iters + 48·p) · n · c_sweep
T_projected    ≈ (iters + 1)  · n · c_sweep + O(p³)
```

Measured on the shared development machine (`signed_cost_model`,
`n = 20 000`, `--fit-iters 8`, the polish cost taken from a metric that still
runs it on the same tree — indicative, not a quiet box): `signed`
22.9 / 70.1 / 125.3 / 238.6 ms against the polish path's
44.5 / 230.3 / 824.6 / 3 648.3 ms at `p = 2 / 3 / 4 / 8`
(1.9× / 3.3× / 6.6× / 15.3×), of which the projection itself is
4.7 / 6.4 / 11.2 / 56.8 ms. Bias `before → after` on the same fixtures:
`5.85e-13 → 1.28e-16` (p=2), unchanged at `~1e-16` where the squared fit was
already feasible (p=3, 4), and `4.37e-4 → 1.08e-8` (p=8, near-collinear basis,
`not_converged` — two rounds did not certify, and the polish it replaces cost
15× the arm and left the bias at `4.37e-4`).

### Deferred-fit contract

The deferred device fit does per candidate what the CPU arm does.

* **One exact gate.** `fit::device_fit_is_gated` is true for `log_loss`, `correlation`, `correlation_signed` and `signed`. For these, `GpuSearch::fit_pending` passes every usable proposal through `fit::gate_device_proposal`, the same function `fit_batch_verified` uses: rank, signed projection, the configured metric in exact `f64`, or squared SSE. `Rejected` restores the incoming constants and sends the candidate to the exact CPU arm with the label `worse` (`log_loss_worse` for `log_loss`); `NonFinite` does the same with the label `non_finite`. `NotIdentifiable` (signed only) uses the label `not_identifiable`.
* **Refresh.** After the device write-back, after the CPU fallback batch and after the pipeline write-back, `refresh_after_fit` simplifies each fitted tree and recomputes its complexity, as `evolve_island` does after the CPU fit. Dead `1*` and `+ 0` nodes left by the kernel no longer reach tie-breaks or the in-search archive. A tree the dimensional filter rejects keeps an infinite error.
* **Pipeline exclusion.** `pipeline_fit_safe` is false for every gated metric, because the pipeline write-back applies kernel constants without the gate.
* **Wide admission.** `wide_admission` is true at 1 048 576 training rows or more, and for `correlation`, `correlation_signed` and `signed` at any size. With `wide`, `select_promoted` takes every candidate with constants and `GpuSearch::score` screens only the candidates without constants; the others are fitted and scored once. Other metrics keep `max(32, pop/4)` below the threshold.

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

## Persistent batch arena and asynchronous copies (issue #285)

The packed batch path (`rc_create` → `rc_set_programs` → `rc_eval`) used to call
`cudaFree`/`cudaMalloc` for `d_prog`, `d_off` and the output on **every batch**,
then launch on the default stream, `cudaEventSynchronize`, and block in
`cudaMemcpy`. Both allocator calls synchronise the device, so that schedule could
never overlap a generation with the previous one. `rc_set_programs` now owns no
allocator call in steady state.

* **One device arena.** `arena_prog[2]`, `arena_off[2]` and `arena_out` are
  allocated once at the high-water mark of the batch shape. `arena_prog` is
  double-buffered: a batch is uploaded into the back slot while the front slot's
  kernel runs. A slot is only overwritten after its own `ev_slot_done` event.
  Capacity growth is explicit (`rc_arena_reserve`) and refused while a batch is
  in flight; it never happens inside a stable generation loop.
* **Pinned host staging.** Two upload slots and two download slots are pinned
  with `cudaHostAlloc` and copied with `cudaMemcpyAsync` on `s_h2d`/`s_d2h`. The
  rings are hard-capped at **64 MiB total**; a request that does not fit, a
  failed `cudaHostAlloc`, a failed stream/event creation, a unified-memory build
  or `RCSR_GPU_SYNC=1` switches to the blocking path and records the reason in
  `rc_arena_report`. The 8 000-program batch above needs 804 392 B of pinned
  staging — well inside the cap — and that number is constant across 100
  generations.
* **Events, not stream synchronisation.** `rc_eval_submit` records the launch
  and D2H events and returns; `rc_eval_collect` waits on the download event and
  copies the pinned errors. `rc_eval` is the compatibility shim that calls the
  two in sequence, so every existing caller keeps its synchronous contract.
  There is no `cudaStreamSynchronize` in the batch path.
* **Event-ordering test.** `tests/gpu_arena.rs` runs 100 generations and asserts
  `alloc_s` stays at zero after warm-up and the pinned ring never grows. It also
  compares the asynchronous path against the forced synchronous path **bit for
  bit**, and uses `eval_batch_pipelined` to enqueue batch B's upload while batch
  A's kernel is still outstanding; `overlapped_uploads` must increase and both
  batches must equal the sequential result. Arithmetic is unchanged: the same
  kernel, launch shape and accumulation order execute; only the stream and
  buffer lifetime differ.
* **Telemetry.** `GpuEvaluator::arena_report()` exposes `async_copy`, `reason`,
  `pinned_bytes`, `alloc_calls`, `steady_alloc_calls`, `overlapped_uploads` and
  `batches`. `alloc_calls` counts cold arena/staging setup; any increase in
  `steady_alloc_calls` after the first batch is a hot-path allocation bug.
  `rcsr-gpu-bench --reps N` prints the arena line next to the #118 ledger.

Measured on the RTX 3060 cell used above (200 000 rows × 2 variables, 8 000
programs, depth 5, CUDA 12.0, plain device memory). The pre-#285 binary reported
25 repetitions; the post-#285 binary reports 100. Per-batch host windows:

| window | before #285 | after #285 |
|---|---:|---:|
| `alloc_s` | 53.0 µs | **0.2 µs** (steady allocator calls: 0) |
| `h2d_s` | 148.1 µs | **97.2 µs** |
| `d2h_s` | 58.3 µs | **5.0 µs** |
| `launch_s` | 41.7 µs | 37.4 µs |
| host windows + sync | ~137.5 ms | ~128.8 ms |
| kernel | ~137.0 ms | ~128.6 ms |

The honest conclusion is the one in the issue: this is a
**correctness-of-schedule** change, not a 10-hour throughput lever. The measured
host overhead falls by roughly 160 µs per batch (less than 0.2 % of the
kernel-dominated wall), but the batch loop no longer allocates, no longer
synchronises through the allocator, and can overlap the next upload with the
current kernel. That is the prerequisite for #286/#291/#292, not a stand-alone
speed-up.

## Persistent fit and score buffers (issue #334)

`rc_fit_batch`, `rc_rank_score_batch`, `rc_order_stat_score_batch` and
`rc_max_scan_batch` used to call `cudaMalloc`/`cudaFree` for their inputs,
outputs and sort workspace on every call (8 allocations and 8 frees per fit
batch), copy through pageable memory with blocking `cudaMemcpy` and launch on the
legacy stream. They now share two per-context regions:

* **One device region** (`fs_dev`) holds, per call, `[inputs | outputs | work]`
  with every part aligned to 256 bytes. The work part carries the rank
  workspace (keys, sorted keys, rows, flags, run starts, prefix sums) and CUB's
  temporary storage, or the order-statistic keys and sort storage.
* **One pinned host region** (`fs_host`) mirrors `[inputs | outputs]`, so a
  batch is one `cudaMemcpyAsync` up, the unchanged kernels, one
  `cudaMemcpyAsync` down, all on the non-blocking `s_fit` stream. The only
  synchronisation is one `cudaEventSynchronize` on `ev_fit_done` per batch. An
  event recorded on the legacy stream orders the batch after the synchronous
  dataset uploads without blocking the host.
* **High-water growth.** A region is reallocated only when a batch needs more
  bytes than it holds, with 25 % headroom. Each device or pinned allocate and
  free is counted in `fit_alloc_calls`, each growth in `fit_growths`. A call is a
  steady-state call (`fit_steady_alloc_calls`) when the same kind of batch (fit,
  rank, order statistic, maximum) has already completed once. The lazy f64 twins
  of the resident rows and the cached target ranks are dataset setup, counted in
  `fit_alloc_calls` only.
* **Arithmetic is unchanged.** Same kernels, launch shapes, owner threads and
  reduction orders; only buffer lifetime and stream differ. A failed launch
  drains the stream before returning, and the caller keeps the CPU arm.
* **Correlation LM step is single-threaded.** `ft_corr_mask_gauge` negates and
  masks the shared `jtj`/`jtr` in place, so only thread 0 runs it, in the same
  branch as `ft_solve`. Running it on every thread raced on the in-place
  negation and made the correlation fit differ in the last digits on about one
  run in a thousand. `gpu::fit_tests::correlation_fit_is_bit_stable_over_repeats`
  guards it.

`GpuEvaluator::arena_report()` exposes `fit_alloc_calls`, `fit_growths`,
`fit_steady_alloc_calls`, `fit_batches`, `fit_device_bytes` and
`fit_pinned_bytes`. The run ledger reports the same counters as
`gpu.ledger.fit.alloc` (`calls`, `growths`, `steady_alloc_calls`, `batches`,
scope `run`, units `allocator calls`); `rcsr schema` documents the field.
`steady_alloc_calls` is 0 while batch shapes stay inside the high-water mark; a
search whose batches keep growing shows the growths there.

Tests: `gpu::persistent_tests` in `src/gpu.rs` asserts 0 allocator calls over
100 fit batches and 100 batches each of the rank, median, interquartile and
maximum stages after warm-up, the counted growth for a larger batch, and a
bitwise hash of fit constants, SSE, status, rank sums, order statistics and
maximum scans that was captured on the allocating path. `tests/gpu_persistent.rs`
holds the archive hashes of GPU-fit searches captured on the allocating path and
a code inspection of the entry points (no legacy-stream launch, no synchronous
copy, no allocator call).

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
`evals_per_s`, `row_evals_per_s`, `batches_per_s` (evaluator batches per
evaluator second, not generations), `eval_kernel_duty` (the **evaluator-window**
kernel fraction as a value with numerator, denominator, units and scope),
`eval_window_s`, `device_seconds_incl_fit` (evaluator wall plus fit kernel
seconds), `device_fraction_of_run` and `eval_fraction_of_run` (over
`timing.total_seconds`), `process_total_s`, `run_wall_s`, `device_busy_s`, `duty_run`, `ledger`, `sm_occupancy`,
`occupancy_source`, `quality_per_gpu_hour`, `quality_per_gpu_hour_run`,
`quality_scope`, and the #131
per-generation `generations` opcode/fallback report. A CPU run reports
`"gpu": null`.

Definitions that matter:

* **wall** is the whole evaluator path measured by the caller. `window_s` is
  the five C host windows and `host_s` the Rust glue outside them; their sum
  `accounted_s` closes to wall, and `residual_fraction` reports any leftover.
  `tests/gpu_loop.rs` asserts closure within 1 % on a real GPU run.
* **evals/s** counts candidate evaluations (`programs / wall`); **row-evals/s**
  counts `programs × rows / wall`. A failed batch contributes wall and bytes but
  zero evaluations.
* **evaluator duty** is `kernel_s / wall_s` inside the evaluator window. It is
  not a run-level number and it is not `nvidia-smi utilization.gpu` (a 1 Hz
  sample of "some kernel was resident") or SM occupancy. The run-level headline
  is `gpu.ledger.duty_run`: the **union** of all device intervals divided by the
  run wall. Overlapping streams are merged, never added; the naive sum and the
  double-counted overlap are both reported (`device_busy_sum_s`,
  `device_overlap_s`).
* **run ledger** (`gpu.ledger`, #284) has one wall clock and exclusive stage
  seconds. `Σ stage_s + unaccounted_s = wall_s`, and more than 1 % unaccounted
  is a telemetry defect. `fit` separates raw device kernel seconds from the
  measured exact-CPU fallback and finalist-refit seconds, with
  `fallback_by_cause`; `eval_fallback` does the same for the evaluator's CPU
  pool. Every derived value is
  `{value,numerator,denominator,units,scope}`. `features`, `pnl` and `io` are
  present even when their stage did not run, so a reader never has to infer a
  missing key.
* **quality/GPU-hour** is decades of error reduction against the constant-mean
  baseline per measured run-wall GPU-hour:
  `-log10(best / baseline) / hours`. The two errors **must be the same unit and
  the same rows** (both MSE or both SSE); the #256 bug was an SSE baseline
  divided by an MSE best. It is reported only for `--metric squared`; every
  other metric reports `null` rather than a mixed-unit ratio. The ledger form
  carries its numerator, denominator,
  units (`decades_per_gpu_hour`) and scope (`run`). The baseline is the
  constant-mean MSE of the training rows, `Σ(y−ȳ)²/n`. `gpu.quality_per_gpu_hour`
  divides by the evaluator window (`gpu.eval_window_s`); `quality_per_gpu_hour_run`
  and `gpu.ledger.quality_per_gpu_hour` divide by the run wall. A throughput gain is not a
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

## Long-only P&L reduction (issue #304)

`src/cuda/pnl.cu` is a separate, fused long-only `{0, +1}` P&L reduction over a
candidate's score vector. It is not the search scorer and it does not evaluate
expressions; it consumes scores and accumulates the same book the scalar
`walkforward` evaluator builds. `rcsr walkforward --device-pnl` exercises it as
a cross-check beside the reported CPU metrics.

The design follows the determinism contract of the batched fit:

* rows are split into fixed 8,192-row **canonical segments**; inside a segment
  rows go to 32 virtual lanes by `(row - segment_start) % 32`, and the lanes
  merge through a fixed pairwise tree;
* a segment is always split over the same lanes, so 128, 256 and 512 threads
  change warp ownership only, never summation order; a warp reads 32 consecutive
  rows at a time instead of striding by the whole segment;
* chunks that end inside a segment resume the per-segment accumulator in device
  memory, so alone/padded/4 k/64 k chunks are bit-identical;
* all P&L terms are `f64`; only the input score is `f32`;
* candidates stream through a declared tile of at most 8 candidates, so the
  path never materialises `P x N` scores;
* the final exit is charged once per group, positions are only `{0, 1}`, and a
  negative position is rejected (no short kernel).

The CPU canonical path in `src/trading_gpu.rs` uses the same segments and merge
order, so device/CPU parity is bit-identical on the fixtures (and within
`1e-9` relative, the acceptance gate, even if a future input exposes a rounding
difference).

## Band, cooldown and hindsight DP scan (issue #306)

`rc_band_scan` and `rc_dp_scan` in `src/cuda/pnl.cu` run the no-trade band, its
minimum-hold (cooldown) variant and the hindsight DP of
[`docs/walkforward.md`](walkforward.md#associative-carries-and-chunk-invariance).
`rcsr walkforward --gpu` uses them; `trading_gpu::band_scan_device`,
`cooldown_scan_device`, their `_batch` forms and `dp_scan_device` are the Rust
entry points.

The contract is stricter than the P&L reduction: the device result equals the
serial CPU scan **bit for bit**, or the call returns `Err(reason)` and the
caller runs the CPU scan.

* **No reassociation.** The CPU folds each chain left to right: one chain per
  `(candidate, group, start state)` for the band, one per `(group, source
  state)` for the DP. The carry algebra is associative, but its rewards are
  `f64` sums and `f64` addition is not (`[1, 2^-53, 2^-53]` is `1.0` left to
  right and `1 + 2^-52` right first). A parallel prefix over carries, CUB's
  `DeviceScan` included, therefore cannot reproduce the CPU bits; CCCL's
  determinism levels promise run-to-run or GPU-to-GPU reproducibility, not
  equality with a serial fold. The device folds each chain in one thread, in
  ascending row order, with the CPU's operations in the CPU's order
  (`-fmad=false`).
* **Layout.** A 1-D grid-stride loop over chains. Band chain
  `(candidate * groups + group) * (K + 1) + start`, so the start states of one
  group sit in adjacent threads and read the same rows; DP chain
  `2 * group + source`. Any block size from 1 to `rc_band_dp_max_threads()`
  (the smaller `maxThreadsPerBlock` of the two kernels) gives the same bits:
  it only changes which thread owns a chain. One block per candidate (the P&L
  layout) would leave a one-candidate walk-forward on a single SM.
* **Chunks.** The host launches once per row chunk. A chain whose group starts
  inside the chunk starts from its initial state; any other chain resumes its
  `(state, value)` from device memory, as the P&L kernel resumes a segment.
* **Memory.** Returns, group offsets and the per-chain state are resident for
  the call; scores are `f64` and stream through the declared 8-candidate tile;
  the start-flat chain writes `u8` positions for the tile. Every allocation
  goes through `rc_checked_malloc`.
* **Host side.** Inputs are validated by the CPU scan's own checks first
  (finite returns, contiguous groups, parameters), and the C entry points
  re-check the group offsets before any upload. The per-group carries come
  back to the host, where `BandScan::net`, `CooldownScan::net` and
  `DpScan::bound` add them with the CPU code.
* **Parallelism.** Chains are the unit of parallel work: `candidates x groups x
  (K + 1)` for the band, `2 x groups` for the DP. A single long group is one
  serial chain per start state. A launch only runs the chains of the groups
  its row chunk touches, so `rcsr walkforward --gpu` scans all rows in one
  launch; smaller chunks give the same bits but serialise the groups.

Measured on the RTX 3060 (`sm_86`), 3,943,856 rows, `β = 0.1`, `K = 1`, one
candidate, block size 256, one launch, device times including upload and
download (`bench_band_dp_device_vs_cpu`, log
`/mnt/lager/tmp/eureqa-gates/issue306-final.log`):

| groups | band CPU | band device | DP CPU | DP device |
|---:|---:|---:|---:|---:|
| 439 (9,001 rows each) | 0.100 s | 0.098 s | 0.030 s | 0.033 s |
| 3,944 (1,000 rows each) | 0.094 s | 0.083 s | 0.032 s | 0.029 s |
| 1 | 0.106 s | 1.131 s | 0.031 s | 0.905 s |

The device scan is exact, not faster: on a panel it is level with the CPU, and
one long series is 10 to 30 times slower, because a chain is one dependent
`f64` add and two loads per row in one thread. With 64 k-row chunks the
439-group band takes 0.267 s. The device path pays off only with many chains
(many groups or a candidate batch); the walk-forward runs one candidate.

Tests (`tests/pnl_gpu.rs`, module `device_band_dp`): the band, cooldown and DP
carries, positions, net and bound equal the CPU at 1, 32, 128, 256, 512 and the maximum
block size and at 4 k and 64 k chunks; every candidate of a padded
two-tile batch equals its alone run; the `n <= 20` brute-force DP and the
`β = 0` threshold identity hold on the device; seams, mid-series entry, final
exit, the cooldown minimum, one-row groups and non-finite scores match the CPU;
refusals carry a reason; and `rcsr walkforward --gpu` reports the same JSON as
the CPU run apart from the backend labels.

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
| Persistent batch arena, pinned <=64 MiB ring, zero steady-state allocator calls, async-vs-sync bit equality, upload-behind-kernel event ordering (#285) | RTX 3060, `tests/gpu_arena.rs` (3 tests) and `rcsr-gpu-bench --reps 100` | **verified** (steady allocator calls 0; 804 392 B pinned ring constant; identical `f32` bits) |
| Speed-up against the SoA/AVX path (not the interpreted walk) | RTX 3060, `rcsr-gpu-bench` | **measured** (see the PR/issue; the benchmark prints both baselines) |
| Failed batch is timed but adds no evaluations | RTX 3060, `tests/gpu_loop.rs` | **verified** |
| SM occupancy / achieved SM throughput | needs `ncu` with counter permission | **unverified here** — reported as `null`, never invented |
| `--no-gpu` forces the CPU, with and without the feature | any machine, `tests/gpu_loop.rs` | **verified** |
| Missing device / CUDA error falls back with one note | `RCSR_GPU=0`, `tests/gpu_loop.rs` | **verified** (no-device path; an injected kernel failure is not exercised) |
| `sm_121` compilation (cubin + SASS + register/smem audit) | x86_64 host, CUDA 13.0.88, **no GPU** (`tests/sm121_compile.rs`) | **verified — compiled, not run** |
| `sm_121` execution on a GB10 (device, driver, GPU tests, benchmark) | needs a dated session on `sm_121` hardware | **hardware-gated — not run here** |
| Dataset larger than device VRAM through managed memory | needs a GB10 (or a >12 GB dataset) | **unverified** — the allocation path is compiled and exercised on `sm_86`, but not at that size |
| Band, cooldown and hindsight DP scan bit-identical to the serial CPU scan at block sizes 1/32/128/256/512/max and 4 k/64 k chunks, padded batches, brute force `n <= 20`, `β = 0`, labelled refusal, `walkforward --gpu` (#306) | RTX 3060, `cargo test --release --features cuda --test pnl_gpu device_band_dp` | **verified** (see the #306 section) |
| On-GPU FP64 verification of survivors | planned | **unverified** — survivors are re-checked by the CPU `f64` evaluator today |
| `log_squashed` native loss in the evaluator and fit kernel, weighted-LM parity with CPU IRLS, determinism, ill-conditioning/fallback labels, `--no-fit` no-op (#229) | RTX 3060, `cargo test --release --features cuda` (`src/gpu.rs` fit tests, `src/fit/gpu.rs`, `tests/gpu_fit.rs`) | **verified** (see the #229 section) |
| logistic log-loss constants on the batched kernel: Fisher weights `q(1-q)`, uploaded label sign, softplus trial objective, closed-form and parity fixtures, extreme logits, single-class refusal, proposal gate, determinism, `--no-fit`, 28/32/64-iteration termination guard (#234) | RTX 3060, `cargo test --release --features cuda --test logloss_gpu` (21 tests) and the ignored `bench_logloss_end_to_end` | **verified** on three paired seeds (2.38×/3.92×/3.14×, quality a wash) |
| **Native classification fitness**: `log_loss`, `brier`, `hinge`, `hinge_ordinary`, `hinge_squared` in the label-aware score kernel; stable softplus/sigmoid; uploaded sign and weights; zero-weight skip; `f64` fixed reduction; exact CPU-finalist rebuild; metric-aware gate (#289) | RTX 3060, `cargo test --release --features cuda --test classification_gpu` and the ignored paired benchmark | **verified** (device/CPU ≤1e-6 on the same f32 rows, ≤1e-5 vs f64; weighted parity; no rank inversion; masked-row skip; deterministic; native-Brier search reports exact CPU values). The large-`N` paired cost is the table in [Classification fitness](#classification-fitness-issue-289) |
| **Weighted regression scoring**: the nine weighted regression objectives in the score kernels (resident and row-range paths), weighted target moments, `Σw` normalisation, zero-weight skip, all-zero weights as status 4, weighted `maximum` refused (`gpu::tests::weighted_regression_scores_match_the_cpu_weighted_score`, `gpu::carry_tests`, `tests/gpu_fit.rs`) | RTX 3060, `cargo test --release --features cuda` | **verified** (device/CPU 1e-12 relative, 1e-9 for the correlations; reported finalists equal CPU `score_weighted` bit for bit) |
| `log_squashed` quality of the robust arm against the CPU IRLS over a corpus (basin splits on heavy tails) | RTX 3060, paired fit-stage and end-to-end benchmarks | **measured on one workload** — 34–49× fit-stage, 2.61× end-to-end, no-worse 64/64 at `p=4,8`; the corpus-wide basin-split rate is not measured |
| Exact median selection (odd/even `n`, ties, constant predictions), CPU parity of the polished constants, determinism, fallback labels and the `--no-fit` no-op (#226) | RTX 3060, `cargo test --release --features cuda` (`src/gpu.rs` median tests, `src/fit/median.rs`, `src/fit/gpu.rs`, `tests/gpu_median.rs`) | **verified** (see the #226 section) |
| Exact maximum and its lowest maximiser (unique/non-unique/ties/all-zero), the exact level interval on an affine coordinate, the exact 1-D minimiser (hand-solved midrange and Chebyshev line), dominance over the start and over the CPU polish, determinism, fallback labels, cost-model floor and the `--no-fit` no-op (#227) | RTX 3060, `cargo test --release --features cuda` (`src/gpu.rs` maximum tests, `src/fit/minimax.rs`, `src/fit/gpu.rs`, `tests/gpu_max.rs`) | **verified** (see the #227 section) |
| Maximum paired fit stage (CPU ladder vs bisection on the error level) | RTX 3060, `tests/gpu_max.rs` (`the_paired_maximum_measurement_shows_the_device_tail_is_faster`) | **measured on three sample sizes** — 5.8–9.1× at 24 candidates, 2.6–8.4× at the search's 32-candidate batch; one-candidate batches lose (0.2–0.5×) and are never used |
| Maximum end-to-end (`rcsr --metric maximum --fit-strategy gpu`) | RTX 3060, three seeds, 20k rows (`--fit-iters 8`), 5 generations, pop 60 | **measured on three seeds** — 1.75–4.0× end-to-end (9.38/3.72/8.48 s exact vs 2.36/2.12/2.89 s device), best front error 3.79/3.91/3.66 vs 3.90/3.63/3.42 (trajectory noise, not a quality claim); fit fallbacks 8.6–10.2 %, all `non_finite`/`unsupported_plan`/`ill_conditioned`, zero `maximum_declined`; the fitness is still CPU, so Amdahl caps the gain |
| `absolute` native `\|r\|` loss in the evaluator and fit kernel, L1/IRLS weights, true-MAE Armijo, parity with CPU IRLS, determinism, zero-residual/ill-conditioned labels, `--no-fit` no-op, metric-aware gate and finalist refit (#222) | RTX 3060, `cargo test --release --features cuda` (`src/gpu.rs` fit tests, `src/fit/gpu.rs`, `tests/gpu_fit.rs`, `src/fit.rs`) | **verified** (see the #222 section) |
| Exact interquartile window mean (rank `[⌊n/4⌋, ⌊3n/4⌋)`, all rows below `n = 4`, ties at both boundaries, constant predictions, odd/even `n`), CPU parity of the score and of the polished constants, determinism across batch composition, ill-conditioned/declined labels and the `--no-fit` no-op (#232) | RTX 3060, `cargo test --release --features cuda` (`src/gpu.rs` order-statistic tests, `src/fit/interquartile.rs`, `src/fit/gpu.rs`, `tests/gpu_interquartile.rs`) | **verified** (see the #232 section) |
| Exact order/rank **scoring** entry point (#288): median/IQR/maximum/rank/AUC bit-identical to the CPU for the same prediction vector, AUC integer identity and chance cases, non-binary AUC refusal, `+inf` for non-finite predictions, batch-composition/repeat determinism, gate admission and named row-floor refusals | RTX 3060, `cargo test --release --features cuda` (`src/gpu.rs` `score_batch_*` and `auc_*` tests, `src/search.rs` gate tests) | **verified** (see the #288 section) |
| `interquartile` paired fit stage and end-to-end quality | RTX 3060, `bench_interquartile_fit_stage` and `bench_end_to_end` (`RCSR_BENCH_METRIC=interquartile`) | **measured on two workloads** — 8.1–32.2× fit stage at 4 096–154 000 rows, 2.5–4.1× end-to-end at equal evaluations, same best error and hypervolume at 50 000 rows, 1.2–3.1 % fallbacks (all `ill_conditioned`); corpus-wide basin-split rate not measured |
| `absolute` paired fit stage and end-to-end quality | RTX 3060, `bench_batched_fit_absolute` and `bench_end_to_end` (`RCSR_BENCH_METRIC=absolute`) | **measured on one workload** — 32.4–35.7× fit stage, no-worse 64/64 at `p=4,8`, 3.55× end-to-end, 6.0 % fallbacks (all `ill_conditioned`); corpus-wide basin-split rate not measured |

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


## Exact rank-correlation fitting (issue #230)

`--metric rank_correlation` (`1 − |ρ|`) and `rank_correlation_signed`
(`1 − ρ`) are fitted on the CPU as a squared LM start followed by a
deterministic coordinate polish on the rank objective. The polish is the cost
centre: `coordinate_polish` re-sorts the prediction **and** the target on every
trial, up to `12·p·6 + 1` `O(n log n)` sorts per candidate on one core. #230
moves that tail to the device and leaves the objective exactly as it is.

**Why the score can be exact.** With average ranks `r`, twice the centred rank

```
A_i = 2 r_i − (n + 1)          B_i = 2 rank_y(i) − (n + 1)
ρ = Σ A_i B_i / sqrt(Σ A_i² · Σ B_i²)
```

is an integer in `[−(n−1), n−1]` with `Σ A_i = 0`, ties included: the average
rank of a tie block is a half-integer and twice the centred value is its integer
midpoint. The three sums are therefore exact integers, independent of the order
in which they are reduced. The device accumulates them as two `u64` halves of a
128-bit two's-complement value, one block per segment, with a carry-aware tree
reduction and **no atomics**. The host turns them into `i128` and performs the
single division, so a CPU and a GPU score are equal bit for bit, and 1 versus 4
islands cannot change a fitted constant.

**Pipeline per generation step.** For the admitted candidates, one call scores
all six ladder trials of all of them:

1. `k_rank_predict` evaluates each candidate's program in **f64**
   (`ft_forward_d`) and writes an order-preserving `u64` key and the row index;
2. `cub::DeviceSegmentedRadixSort::SortPairs` sorts every `(candidate, trial)`
   segment (the same primitive `cuda/enrich.cu` already uses);
3. `k_rank_flags` marks run starts (exact `f64` equality, so `-0.0` and `+0.0`
   share a rank like the CPU's `==`) and gathers the cached target ranks;
4. two CUB integer scans give the run ordinals and the per-row prefix sums of
   `B`;
5. `k_rank_contrib` turns each run's `[start, end)` into
   `A = start + end − n` and accumulates `A·ΣB` and `A²·length`, reducing the
   block deterministically.

The target's `B` is computed once per fit sample and cached on the device, so
the target is never re-sorted per trial.

**Trajectory compatibility.** The device arm runs the *same* ladder
`(+10 %, −10 %, +3 %, −3 %, +1 %, −1 %) × max(|c|, 1)` in the same constant
order and **commits the first strict improver**, never the best of the six
trials. A test drives `coordinate_polish` and the device polish from the same
starting constants and asserts bit-identical constants, for the unsigned and the
signed objective. `--no-fit` never enters the arm (nothing to fit), and the
CPU `--fit-strategy exact` arm stays bit for bit.

**Fallbacks.** A candidate is offered only when it has a GPU fit plan (`p ≤ 20`,
≤ 96 nodes, no shared block constants, analytic operators) and at least `32p`
rows. A non-finite prediction makes that segment unscorable; a failed
allocation/launch, a declined device call, or a non-finite constant is returned
as a non-zero status. Every such candidate is refit by the exact CPU arm
(`fit_weighted_metric`, i.e. squared LM plus the exact rank polish) and counted
under the labelled causes (`unsupported_plan`, `too_few_rows`,
`too_many_constants`, `non_finite`, `launch`, and `rank_declined` when a device
rank call itself fails). The fitness of every candidate,
GPU-fitted or not, is the CPU rank score, and every reported finalist is still
refit exactly by the CPU before the archive is rebuilt.

**Measured cost (RTX 3060, shared with a 10-hour GPU run, so indicative).**
One candidate, `p = 4`, up to 8 sweeps, CPU exact polish versus the device
polish on the same start (identical constants in every row):

| rows `n` | CPU polish | GPU polish | speed-up | one call (6 trials) |
|---|---|---|---|---|
| 5 000 | 38.5 ms | 9.6 ms | 4.0× | 0.55 ms |
| 20 000 | 237.7 ms | 26.7 ms | 8.9× | 3.1 ms |
| 50 000 | 430.2 ms | 74.5 ms | 5.8× | 3.3 ms |
| 154 000 | 1 335.8 ms | 112.9 ms | 11.8× | 7.5 ms |

The stage cost splits into a fixed per-call part and a per-segment part. On the
same run at `n = 154 000`, one segment cost 5.20 ms and six segments (all six
ladder trials) 7.46 ms, i.e. **≈ 4.7 ms fixed + 0.45 ms per segment**. The fixed
part is allocation, uploads, CUB temporary-size queries and eight launches; the
per-segment part is the segmented radix sort of `n` 64-bit keys plus the integer
reduction, and it grows with `n` (at 3.9M rows it is the dominant term). Because
one call covers every live candidate, the fixed part is amortised across the
batch and the per-candidate cost approaches the sort. The cost model is
therefore

```
T_polish ≈ calls · (t_fixed + segs · t_sort(n)),   calls ≈ 1 + sweeps · p
```

with `segs` = live candidates × 6. Reusing the per-call buffers in the context
(and folding the two scans into one) is the obvious next optimisation; it would
remove most of `t_fixed` and is not needed for exactness. End to end on `n = 20 000`, `pop = 48`, 2 generations, 1 island, 96
evaluations: 3.74 s exact CPU versus 2.10 s GPU/rank (1.78×) at the same best
error (`1.095749e-1`), with 72/78 admitted fits on the device, 6 labelled CPU
fallbacks (7.7 %) and the fit stage at 39.5 % of wall.

**When it is used.** Only with `--fit-strategy gpu`, a CUDA build, a usable
device, single-output unweighted full-sample data, and `--fit-iters > 0`. Below
roughly `n ≈ 10⁴` the CPU polish is latency-competitive for a single candidate,
so the device arm pays off through batching; the threshold is not a hidden
switch — any configuration outside the list above is labelled by
`gpu_blocker`'s reason in the one-line note.

## Signed rank correlation (issue #231)

`--metric rank_correlation_signed` scores `1 − ρ`, not `1 − |ρ|`: the direction
of the monotone relation is part of the objective. #230 landed the exact integer
rank stage and left the orientation as one host-side flag; #231 proves and pins
that orientation end to end. There is no second kernel: the integer sums
`Σ A·B`, `Σ A·A`, `Σ B·B` are shared, and `signed` is applied once, after the
single division.

**The identities the arm is built on.** For any prediction `f` and its
reflection `−f`, `A(−f) = −A(f)` exactly, so

```
Σ AB(−f) = −Σ AB(f),   Σ A² and Σ B² unchanged
ρ(−f) = −ρ(f)
1 − ρ(−f) = 2 − (1 − ρ(f))         1 − |ρ(−f)| = 1 − |ρ(f)|   (unsigned)
```

A reflection is therefore *penalised* by `2 − score` under the signed metric and
*invisible* under the unsigned one. That is the only difference between the two
arms, and it is a bit-exact statement about the integer sums, tested on the CPU
(`reflection_penalty_is_a_score_identity`) and on the device
(`reflection_flips_the_cross_sum_exactly_on_the_device`: the device returns
`AB(−c) = −AB(c)` with equal `AA`, `BB`).

**Where the orientation is decided.**

* `coordinate_polish` and the batched `polish_rank_batch` score every trial with
  `RankTarget::score(pred, signed)`. A coordinate step *can* cross a zero
  constant and so reflect the model; it is accepted **only** when that improves
  `1 − ρ` strictly. No procedural reflection is ever applied.
* The keep/replace decision after the polish is `fit::rank::choose_rank_fit`,
  shared by the search arm (`GpuEvaluator::rank_fit_batch`), the batched
  verifier (`BatchedGpu`) and the unit tests: strict improvement on the exact
  oriented score, equal keeps the start, and an unscorable **baseline** is a
  labelled `rank_declined` fallback rather than a blind acceptance of the
  polished constants.
* The shared proposal gate (`fit::gpu::verify_proposal`) now judges a rank
  proposal on the **exact integer statistic** under the metric's orientation
  (before #231 it compared the f64 Pearson value; the ordering is identical, but
  only the integer form is the same arithmetic the polish decides on). A
  reflected proposal is rejected for `rank_correlation_signed` and accepted as
  an equal for `rank_correlation`.
* `BatchedGpu::fit_batch_metric` answers rank metrics with the device rank arm
  (`rank_fit_batch`), so the batched verifier is rank-native; the search path
  already used the same arm directly. Weights, multi-output and non-finite rows
  are still declined to the exact CPU arm.

**MSE can never rank.** `GpuSearch::native_metric` is `false` for the rank
family, and the run is scored through the exact batch entry point
(`native_stat`, #288): the same integer statistic that decides the polish also
scores every candidate (see the #288 section), and the squared LM is only the
*start* of the polish. A test pins the gate direction
(`a_signed_rank_proposal_is_accepted_even_when_it_costs_sse`): a proposal that is
much worse in SSE but perfect in signed ρ is accepted, which an SSE (or `|ρ|`)
gate would reject. `--metric rank_*` below the 4 096-row scoring floor reports
`cpu` with the named floor reason and scores on the CPU; the fit-only arm may
still run there.

**`--no-fit`.** `fit_iters = 0` never builds the evaluator (the arm is not
offered), and `rank_fit_batch(…, 0, …)` returns the incoming constants without a
sweep, for both orientations.

**Measured (RTX 3060, machine shared with other builds — load average ≈ 12, so
indicative).** Fit-stage polish, `p = 4`, up to 8 sweeps, same start, identical
constants:

| `n` | CPU exact polish | GPU polish (unsigned) | GPU polish (signed) | speed-up |
|---|---|---|---|---|---|
| 154 000 | 1 218.1 ms | 93.2 ms | 93.9 ms | 11.95× |

The two orientations cost the same, as expected: identical sort work, one sign
on the host. One call carrying `s` ladder segments at `n = 154 000` (mean of 3):
5.02, 5.99, 6.28, 7.17, 6.74, 7.29, 7.97, 8.19 ms for `s = 1…8`, a least-squares
fit of **`t = 4.97 ms + 0.41 ms × segments`**. The fixed part is
allocation/upload/launch and amortises across live candidates; the per-segment
part is the segmented radix sort plus the integer reduction.

Paired end-to-end (`n = 20 000`, `pop = 48`, 2 generations, 1 island, 96
evaluations, same seed and rows):

| metric | CPU exact | GPU rank fit | speed-up | best error | fits on device | labelled fallbacks | fit share of wall |
|---|---|---|---|---|---|---|---|
| `rank_correlation` | 3.91 s | 2.00 s | 1.96× | 8.553000e-2 → 8.551545e-2 | 72 / 78 | 6 (7.7 %) | 40.1 % |
| `rank_correlation_signed` | 3.80 s | 1.69 s | 2.25× | 1.095749e-1 (identical) | 75 / 80 | 5 (6.2 %) | 44.3 % |

Fitness stays on the CPU in both rows; the speed-up is the fit stage only,
which is why it is smaller than the 11.95× polish speed-up.

**What is not claimed.** The 3.9M-row cell is not exercised (the cost model
extrapolates from 154k); buffer reuse across rank calls is not implemented; and
1-island versus 4-island *archives* are not asserted equal (the island RNG
streams differ by design) — determinism is pinned as repeated identical runs at
each island count.

## Median absolute error fitting (issue #226)

`--metric median` scores `medianᵢ |yᵢ − f(xᵢ;c)|`. The CPU arm is a squared
fit followed by `coordinate_polish` on that median: per constant six fixed
ladder steps (`±10 %, ±3 %, ±1 %` of `max(|c|,1)`), the **first strict
improver** committed, up to 12 sweeps. Every trial costs a full prediction
sweep **and a full `n log n` sort** of the residuals, and a sweep is `6·p`
trials — the sort is the cost centre, exactly like the rank family of #230.

#226 moves that tail to the device without changing what it optimises.

**Exact selection.** With `rᵢ = pred(xᵢ) − yᵢ` and `|rᵢ| ≥ 0`, the IEEE-754
`u64` bit pattern of `|rᵢ|` orders exactly like the value. The device

1. evaluates each candidate's program in **exact `f64`** (`ft_forward_d`) and
   writes `bits(|rᵢ|)` for every row of every `(candidate, trial)` segment
   (`k_median_keys`; a non-finite prediction or residual marks the segment,
   never a `1e30` stand-in);
2. sorts each segment's keys with `cub::DeviceSegmentedRadixSort::SortKeys`
   (four byte passes over keys already in memory, no atomics, no float
   reduction);
3. picks the median (`k_median_pick`): the middle key for odd `n`,
   `0.5·(a + b)` of the two middle keys for even `n` — the same expression
   `metrics::median_abs` evaluates.

`median` is a symmetric function of the multiset `{|rᵢ|}`, so **ties cannot
change the value** and need no row-index tie-break: an all-tied residual vector
selects the same number however the batch is composed. A test drives odd and
even row counts, an entirely tied multiset, `n = 2..5`, a constant prediction
and several constant vectors, and asserts the device median equals
`metrics::score_weighted(.., median)` **bit for bit**.

**Pipeline per generation step.** For the admitted candidates:

1. the resident rows are fitted by the landed squared LM kernel (the same warm
   start `fit_weighted_metric` uses, with the same `--fit-iters` count);
2. one call scores all six ladder trials of all live candidates
   (`rc_median_score_batch`);
3. the host commits the first strict improver per constant, exactly like
   `coordinate_polish`, and repeats for at most `--fit-iters` sweeps;
4. the arm keeps the **incoming** constants unless the polished median is
   strictly lower, so a squared start that is worse on the median cannot be
   reported.

**Trajectory compatibility.** The device runs the *same* ladder in the same
constant order with the same first-improver rule on the same exact `f64`
median. From the same starting constants the batched polish and
`coordinate_polish` commit **bit-identical** constants and scores (pinned by a
device test). From *different* warm starts (the device LM against the CPU LM)
the two arms may reach different local optima, because the median objective is
not convex; that is why the reported number is always the exact CPU median of
the reported constants and why every finalist is refit by
`refit_finalists_exact` → `fit_weighted_metric(.., median)`.

**Metric-aware gate, not SSE.** The device proposal passes through
`fit_batch_verified`/`fit_pending`, which score the **configured median** in
exact `f64` and accept only a finite, no-worse value. A test constructs a
proposal whose median is exactly `0` while its SSE is strictly larger than the
least-squares optimum's and asserts that it is accepted for `median` — the SSE
gate would have rejected the only median answer. A squared-only backend returns
`None` for a median request and the exact CPU median arm runs instead.

**Fallbacks (all labelled).** A candidate is offered only with a GPU fit plan
(`p ≤ 20`, ≤ 96 nodes, no shared block constants, analytic operators) and at
least `32p` rows; every other candidate, and every batch the device marks
non-finite (`non_finite`), ill-conditioned (`ill_conditioned`), launches
badly (`launch`) or whose median call declines (`median_declined`), is refit by
the exact CPU median arm. The run-level decision has a reason too: the device
arm needs `n ≥ MEDIAN_GPU_MIN_ROWS = 4 096`, and below it the exact CPU arm
runs with `# gpu-fit: off (median fit arm: the sample is too small for the
device polish)` and the same string in `backend_reason`. `--no-fit` never
enters the arm: the evaluator is not created and the archive is identical to
the exact CPU run, bit for bit.

**Measured cost (RTX 3060, machine shared with other jobs, so indicative).**
One candidate, `p = 4`, `--fit-iters 8`, sin/linear tree; best of three runs:

| rows `n` | CPU arm | CPU polish tail | GPU arm | one call (6 trials) | whole-arm speed-up |
|---|---|---|---|---|---|
| 5 000 | 50.0 ms | 35.1 ms | 13.4 ms | 0.47 ms | 3.7× |
| 20 000 | 151.2 ms | 91.3 ms | 22.7 ms | 1.05 ms | 6.7× |
| 50 000 | 515.3 ms | 435.7 ms | 63.1 ms | 2.34 ms | 8.2× |
| 154 000 | 1 503.7 ms | 756.3 ms | 254.5 ms | 4.00 ms | 5.9× |

The tail cost splits into a fixed per-call part and a per-row part:

```
T_cpu_tail ≈ calls · (c_cpu · n)                    c_cpu ≈ 3.3e-8 s/row
T_gpu_tail ≈ calls · (t_fixed + 6 · c_gpu · n)      t_fixed ≈ 0.35 ms, c_gpu ≈ 4.0e-9 s/row
calls = 1 + sweeps · p
```

which is `MEDIAN_CPU_ROW_S`, `MEDIAN_GPU_CALL_S` and `MEDIAN_GPU_ROW_S` in
`src/fit.rs` and gives a crossover near `n ≈ 2 000` for `p = 4`. The admission
floor is set to **4 096 rows** — the smallest round count with a modelled margin
of ~1.7× and inside the measured never-worse regime — and it is a row count,
never a timing measurement, so a run is reproducible. `median_fit_cost(n, p,
sweeps)` exposes the model and a unit test pins the floor and its monotonicity.

**When it is used.** The **scoring** arm is the exact device median from
`n ≥ 4 096` under the same single-output/unweighted/full-sample gates (#288), so
the backend is `gpu` and the fitness is the configured median. The **fit** polish
additionally needs `--fit-strategy gpu` and `--fit-iters > 0`; when the fit arm
is declined, `median_declined` is one of the labelled causes and the exact CPU
arm refits. Below the scoring floor the exact CPU median runs and records the
floor reason. The interquartile arm is
[#232](#interquartile-absolute-error-fitting-issue-232) below.

## Interquartile absolute error fitting (issue #232)

`--metric interquartile` scores the **mean of the absolute residuals between the
25th and 75th percentile**, `metrics::interquartile_mean_abs`: with
`a = sorted |rᵢ|` and `n = a.len()`,

```
n < 4 :  mean(a)                        (no window; the CPU small-n rule)
else  :  mean(a[n / 4 .. (3 * n) / 4])   (integer division, so ⌊·⌋)
```

The CPU arm is the same family as `median` (#226): a squared pre-fit followed by
`coordinate_polish` on that window mean — six ladder steps per constant
(`±10 %, ±3 %, ±1 %` of `max(|c|,1)`), the **first strict improver** committed,
up to 12 sweeps, each trial paying a full prediction sweep **and** a full sort.

#232 runs that ladder on the device through the same driver as #226
(`fit::polish_order_stat_batch`, one `fit::OrderStat` switch), so the two arms
cannot drift:

1. the same `LADDER` table, the same constant order, the first strict improver,
   at most `min(sweeps, 12)` sweeps;
2. the same objective, scored from the resident **exact `f64`** rows;
3. the same contract: a declined call (`None`) hands the batch to the exact CPU
   arm with a labelled cause.

**Rank window, not a value range.** The window is `[⌊n/4⌋, ⌊3n/4⌋)` by *rank*.
Selecting "every key in `[a[lo], a[hi−1]]`" is wrong when the top boundary value
repeats: `a = 1, 2, 3, 3` (`n = 4`) averages `2, 3` in the window (`2.5`) but
`2, 3, 3` over the value range (`8/3`). A device test pins `2.5` and asserts the
value-range answer is rejected. The device therefore selects ranks from the same
`u64` bit-pattern sort the median arm uses (`rc_order_stat_score_batch`, `what =
1`) and averages exactly those ranks.

**Ties.** The window mean is a function of the multiset `{|rᵢ|}`, so ties cannot
change the value at either boundary, and CUB's radix sort is stable (equal keys
keep row order), so the selected *set* is reproducible as well. Tests cover all
tied residuals, an all-equal multiset, a top-boundary tie, a bottom-boundary
tie, half the rows tied at the top, and `n = 2..9` (the `n < 4` fallback).

**Exactness and the f32 boundary risk.** The window sum is a **fixed 256-thread
tree reduction** over ranks `[lo, hi)` (`k_order_stat_pick`), because a
single-thread left-to-right sum is bit-identical to the CPU but costs ~200 µs
per trial at 154 000 rows. The consequence is stated exactly:

* the interquartile device score is deterministic (fixed window, fixed thread
  count, no float atomics) and agrees with `metrics::score_weighted` to
  **≤ 1e-12 relative** — not bit for bit, unlike the median pick, which is a
  single value. Tests assert the relative bound, not equality of bits;
* the boundary *risk* is one ulp: moving one boundary row by one ulp moves the
  mean by at most `ulp / window`, pinned by a unit test. The arm predicts in
  `f64` from the exact resident rows, so this does not arise on the scored path;
  only the missing-f64-twin fallback in `rc_order_stat_score_batch` casts
  `f32 → f64`, where one boundary row can move by one **f32** ulp (2.4e-7 at
  `|r| = 2`), i.e. ~3e-8 of the window mean — far above the 1e-12 parity
  tolerance. That is why the search always uploads the f64 twins and why the
  f32 path is not a parity path.

**Metric-aware gate and refit.** The device proposal passes through
`fit_batch_verified`/`fit_pending`, which score the **configured interquartile
window mean** in exact `f64`; a proposal is accepted only when the window mean is
finite and no worse. A test builds a proposal whose window mean is exactly `0`
while its SSE is strictly larger than the least-squares optimum's and asserts it
is accepted for `interquartile` — an SSE gate would reject it. A squared-only
backend returns `None` for an interquartile request and the exact CPU arm runs
instead. Finalists are refit by `refit_finalists_exact` →
`fit_weighted_metric(.., interquartile)`. That refit is monotone but not
bit-idempotent: the arm restarts from a squared pre-fit, so a second pass begins
the LM at the polished constants and can land on a marginally different squared
optimum. The guarantee is "never worse", and that is what the test pins.

**Fallbacks (all labelled).** A candidate is offered only with a GPU fit plan
(`p ≤ 20`, ≤ 96 nodes, no shared block constants, analytic operators) and at
least `32p` rows. Every other candidate — and every batch the device marks
non-finite (`non_finite`), ill-conditioned (`ill_conditioned`), launches badly
(`launch`) or whose interquartile call declines (`interquartile_declined`) — is
refit by the exact CPU interquartile arm. The run-level decision has a reason
too: the arm needs `n ≥ INTERQUARTILE_GPU_MIN_ROWS = 4 096`, and below it the
CPU arm runs with `# gpu-fit: off (interquartile fit arm: the sample is too
small for the device polish)` and the same string in `backend_reason`.
`--no-fit` never enters the arm: the evaluator is not created and the archive
is identical to the exact CPU run, bit for bit.

**Measured fit stage (RTX 3060, machine shared with other jobs, best of three;
3-constant sin/linear tree, `--fit-iters 8`, p = 3).**

| rows `n` | CPU arm | CPU polish tail | GPU arm | one call (6 trials) | tail ratio |
|---|---|---|---|---|---|
| 2 000 | 15.3 ms | 8.2 ms | 5.0 ms | 0.18 ms | 3.4× |
| 4 096 | 63.0 ms | 53.8 ms | 10.8 ms | 0.22 ms | 8.1× |
| 10 000 | 109.7 ms | 92.0 ms | 15.6 ms | 0.37 ms | 14.7× |
| 20 000 | 207.6 ms | 162.4 ms | 29.8 ms | 0.61 ms | 14.9× |
| 50 000 | 710.7 ms | 563.8 ms | 75.9 ms | 1.68 ms | 18.0× |
| 154 000 | 3 850.0 ms | 3 307.0 ms | 325.3 ms | 3.75 ms | 32.2× |

The parallel window sum does not show up above the sort's noise: the six-trial
call costs 3.75 ms at 154 000 rows, the same as the median arm's 4.00 ms. The
cost model is the median arm's shape with measured constants
(`IQR_GPU_CALL_S = 1.6e-4`, `IQR_GPU_ROW_S = 3.9e-9`, CPU side
`MEDIAN_CPU_ROW_S`):

```
T_cpu_tail ≈ calls · (c_cpu · n)                    c_cpu ≈ 3.3e-8 s/row
T_gpu_tail ≈ calls · (t_fixed + 6 · c_gpu · n)      t_fixed ≈ 0.16 ms, c_gpu ≈ 3.9e-9 s/row
calls = 1 + sweeps · p
```

`interquartile_fit_cost(n, p, sweeps)` exposes it, `interquartile_gpu_fit_worthwhile`
is the decision, and the admission floor is **4 096 rows** — a row count, never
a timing measurement. The model is deliberately conservative on the CPU side:
the IQR ladder more often improves on every sweep and so runs all of them, while
the model charges the median's per-trial coefficient.

**Measured end to end (equal evaluations, seed 9, `bench_end_to_end`).**

| cell | exact CPU | GPU fit + CPU scoring | speed-up | best error | fallbacks | fit stage |
|---|---|---|---|---|---|---|
| `N = 20 000`, `pop = 48`, 2 gens, 1 island (a 3-constant tanh fit reaches the final answer on the device) | 2.86 s | 1.13 s | 2.5× | 5.39e-1 / **1.57e-1** | 1/80 = 1.2 % (`ill_conditioned`) | 0.33 s = 29 % of wall |
| `N = 50 000`, `pop = 96`, 2 gens, 1 island | 17.33 s | 4.25 s | 4.1× | 4.05e-1 / 4.05e-1 (same) | 3/96 = 3.1 % (`ill_conditioned`) | 1.28 s = 30 % of wall |

The 50 000-row cell reaches the same archive best and hypervolume (0.348) and the
same time-to-target generation, 9.6× sooner in wall time (1.8 s vs 17.3 s). The
20 000-row cell's lower error is a trajectory effect (the device warm start finds
a different minimum of a non-convex objective), not a quality guarantee; the
guarantee is per candidate: the gate never accepts a worse window mean. The
`gpu.accounting` duty counter reads 0.0 % on these runs because a fit-only arm
does not feed the evaluator accounting; the device-time share is the fit stage's
`fit_ms / wall` (0.23 s / 4.25 s in the 50 000-row cell).

**When it is used.** The exact window mean **scores** the population on the
device from `n ≥ 4 096` (#288), so the backend is `gpu` and the fitness is the
configured statistic. The **fit** polish additionally needs `--fit-strategy gpu`
and `--fit-iters > 0`; a declined polish is labelled `interquartile_declined`
and refit by the exact CPU arm. Below the scoring floor the CPU window mean runs
with the named floor reason.

## Maximum absolute error fitting (issue #227)

`--metric maximum` scores `maxᵢ |yᵢ − f(xᵢ;c)|` (Chebyshev / L∞). The CPU arm
is a squared fit followed by `coordinate_polish` on that maximum: per constant
six fixed ladder steps, so it can only land on `±1 %, ±3 %, ±10 %` of
`max(|c|,1)` and cannot resolve a minimum between two rungs — and it pays
`6·p·sweeps` full prediction sweeps. The objective is not an SSE and has no
smooth Hessian: with a unique maximiser its subgradient is the rank-one
`sign(r*)·g*`, so the squared LM cannot drive it.

#227 replaces that tail with an **exact one-dimensional minimisation per
constant**, batched over a whole generation, and proves two dominance claims.

**The level test.** With the other constants fixed and a tree affine in the
chosen coordinate, the residual is `rᵢ(δ) = aᵢ + bᵢ·δ`, where `aᵢ = rᵢ` and
`bᵢ = ∂rᵢ/∂c` from the exact adjoint sweep the fit kernel already forms. The
level `t` is attainable along the coordinate **exactly** when the per-row
intervals

```text
−t ≤ aᵢ + bᵢ·δ ≤ t   ⇔   δ ∈ [min(e₁,e₂), max(e₁,e₂)],
                          e₁ = (−t − aᵢ)/bᵢ,  e₂ = (t − aᵢ)/bᵢ
```

intersect: `max loᵢ ≤ min hiᵢ`. A row with `bᵢ = 0` constrains no `δ` and only
vetoes the level when `|aᵢ| > t`. The device returns that intersection with the
maximum in one scan (`k_max_scan`, `rc_max_scan_batch`), and the host
**bisects on the error level** `t ∈ [0, J(c)]`: `J(c)` is feasible at `δ = 0`
by definition, `t = 0` is the lower end, and the committed step is the midpoint
of the intersecting interval at the last feasible level. At most
`MINIMAX_BISECT_ROUNDS = 32` rounds, stopping as soon as the bracket is within
`1e-9` relative — a closed, ordered, deterministic search with no step ladder
and no unbounded optimiser.

For a tree that is **not** affine in the coordinate (a fitted exponent, a
threshold, `log(c·x)`) the residual is not affine in `δ`, so the interval test
is a *linearised* feasibility test, not an exact one. It is documented as a
heuristic where it applies: it proposes a step, and the exact accept test keeps
the monotonicity claim true. The exactness claim is made only for affine
coordinates, and the tests pin it there (a hand-solved midrange, a two-constant
Chebyshev line, and the exact interval at the exact optimum).

**The exact maximum and its maximiser.** `k_max_scan` reduces, one block per
`(candidate, coordinate)` segment,

* `max |pred − y|` in `f64`. A maximum is **order-free** — reordering the rows
  cannot change it — and `|r| ≥ 0`, so its IEEE bit pattern orders exactly like
  the value and the reduction is an **integer max**: no float summation, no
  rounding, no run-to-run variation;
* the **lowest** row index that attains it (`min` over ties), so a non-unique
  maximiser is reported identically on every run and the active set is
  reproducible;
* the level interval above, when a coordinate and a level are requested.

Every reduction is a fixed-structure shared-memory tree of order-free
operators; there are no atomics anywhere. Tests pin the device maximum against
`metrics::score_weighted(.., maximum)` **bit for bit**, the lowest maximiser on
a tied fixture, and identical results alone, repeated, or batched next to other
candidates.

**Pipeline per generation step.** For the admitted candidates:

1. the resident rows are fitted by the squared LM kernel (the same warm start
   `fit_weighted_metric` uses, same `--fit-iters`);
2. one scan scores the baseline `max |r|` of the incoming constants;
3. per sweep, 32 level scans run over all candidates and all their coordinates
   at once, then one batched call scores the proposed constants; the best
   **strict** improver is committed (ties go to the lowest constant index);
   a sweep that commits nothing ends the polish, at most `min(iters, 12)`
   sweeps;
4. the arm keeps the **incoming** constants unless the polished maximum is
   strictly lower.

**Dominance, twice.**

* *Over the start* — structural: a step is committed only when the exact
  `max |r|` of the trial is strictly less than the best so far, so the arm can
  never report a constant it made worse, whatever the device returned.
* *Over the CPU polish* — `maximum_finalist` runs `fit_weighted_metric(..,
  maximum)` (the CPU ladder) on the device constants for every finalist and
  keeps whichever exact objective is lower, with a plain `f64` `<` and **zero
  tolerance**. A test asserts `J_device ≤ J_cpu_polish` on the same candidates
  from the same warm start.

**Metric-aware gate, not SSE.** Proposals pass through
`fit_batch_verified`/`fit_pending`, which score the **configured maximum** in
exact `f64`; the device's own `GpuFitOutput.sse` field carries `max |r|` for
this arm, never an SSE. A squared-only backend returns `None` for a maximum
request and the exact CPU arm runs instead.

**Fallbacks (all labelled).** A candidate is offered only with a GPU fit plan
(`p ≤ 20`, ≤ 96 nodes, no shared block constants, analytic operators) and at
least `32p` rows; every other candidate, and every batch the device marks
non-finite (`non_finite`), ill-conditioned (`ill_conditioned`), launches badly
(`launch`) or whose maximum scan declines (`maximum_declined`), is refit by the
exact CPU maximum arm. The run-level decision has a reason too: the device arm
needs `n ≥ MAXIMUM_GPU_MIN_ROWS = 512`, and below it the exact CPU arm runs
with `# gpu-fit: off (maximum fit arm: the sample is too small for the device
polish)` and the same string in `backend_reason`. `--no-fit` never enters the
arm: the evaluator is not created and the archive is identical to the exact
CPU run, bit for bit.

**Paired measurement (RTX 3060, shared machine, indicative).** 24 affine
candidates `a·x + b`, `--fit-iters 8`, same squared warm start on both arms:

| rows `n` | CPU arm | GPU arm | speed-up |
|---|---|---|---|
| 5 000 | 277.9 ms | 48.1 ms | 5.8× |
| 20 000 | 1 116.3 ms | 123.5 ms | 9.0× |
| 50 000 | 2 620.6 ms | 287.3 ms | 9.1× |

and at the search's **minimum admitted batch** (32 candidates, `p = 2`, best of
three): 256 rows 23.4 vs 9.1 ms (2.6×), 512 rows 43.1 vs 10.9 ms (3.95×),
1 024 rows 78.3 vs 10.4 ms (7.6×), 8 192 rows 623.8 vs 74.2 ms (8.4×). On a
**single** candidate the device loses (0.2–0.5×): 266 sequential launches
cannot be amortised over one block, which is why the arm is only ever reached
through the batched search.

The cost model in `src/fit.rs` is per candidate:

```text
cpu_s = START_S + sweeps·p·6·CPU_EVAL_ROW_S·n
gpu_s = START_S + calls·GPU_CALL_S + sweeps·ROUNDS·p·GPU_SCAN_ROW_S·n
calls = 2 + sweeps·(ROUNDS + 1)
START_S = 1.5e-4, CPU_EVAL_ROW_S = 2.46e-8, GPU_CALL_S = 8.2e-7, GPU_SCAN_ROW_S = 5.0e-10
```

Every coordinate of a candidate rides in **one** call, so the device term scales
with `p` per row but not per call, and the modelled speed-up grows with `p`. The
one-call packing matters: six separately synchronised `cudaMemcpy` calls per
round were most of the latency before the request buffer was packed into a
single upload (`rc_max_scan_batch`). The admission floor is **512 rows** — above
the measured crossover, where the model predicts a speed-up for every `p = 1..=20`
and the measurement shows at least 3.9× at the search's batch size.
`maximum_fit_cost(n, p, sweeps)` exposes the model and a unit test pins the
floor, the monotonicity in `n` and in `p`, and the coefficients' signs.

**Measured end-to-end (three seeds, 20 000 rows, 5 generations, `--pop 60`).**
CPU `--fit-strategy exact` 9.38 / 3.72 / 8.48 s against the device arm
2.36 / 2.12 / 2.89 s (1.75–4.0×, the machine is shared so these are
indicative), with best front error 3.90 / 3.63 / 3.42 against 3.79 / 3.91 /
3.66 — the fit is kept monotone per candidate, but the search trajectory
differs, so this is not a quality claim. Fallback share is
8.6–10.2 %, entirely `non_finite`, `unsupported_plan` and `ill_conditioned`;
zero `maximum_declined` after the mixed-`p` batch fix (below). This measurement
predates #288's native maximum scoring, which removes the CPU fitness pass and
raises the end-to-end share the arm can recover; the fit stage numbers are
otherwise unchanged.

**Mixed constant counts.** The driver sends one level/interval vector per
constant of **each** candidate (`plan.order`), not one per slot of the batch's
largest `p`. Mixing those up made the device decline every batch that held a
`p = 1` candidate next to a `p = 3` one, which the search then reported as
`maximum_declined` and refit on the CPU: correct, but a large silent regression
in device use. Two tests pin it — a driver test that polishes a `p = 1, 2, 3`
batch and compares it bit for bit with three single-candidate runs, and a device
test that requires `status == 0` for every candidate of a mixed batch.

**When it is used.** The **fit** bisection needs `--fit-strategy gpu`, a CUDA build,
a usable device, single-output unweighted full-sample data (a **weighted**
maximum has no defined objective — `weights_supported` excludes `maximum` — so it
is never fitted on the device), `--fit-iters > 0` and `n ≥ 512`. Scoring is
native through the #287 maximum reducer; a declined call is labelled
`maximum_declined` and refit by the exact CPU arm. Lawson IRLS on the weighted
assembly (`minimax-bias.md` §2.3) and native maximum scoring remain open; the
bisection is the research note's `R1`, which the note recommends first.

## Per-stage dispatch and the route ledger (#290)

Dispatch is planned per stage in `src/dispatch.rs`: `plan(stage, workload)`
returns `device`, `cpu` or `split` with a machine reason. The model uses
`R = W·N·m`, `T_dev = R/r_eval`, `T_fit = W/r_fit(N,p,sweeps)`,
`T_xfer = bytes/BW`, a measured launch term and `T_cpu` from telemetry, with a
hysteresis band `h = 0.3` so a stage does not oscillate at the crossover.
`vary` stays CPU until #291; `select`, `archive`, `features`, `pnl`, `io` and
the exact `refit` are CPU by contract.

A ~100 ms startup probe measures the host rate and anchors the device rates;
the values, source, sample count and reproducibility band are written to
`--json` under `dispatch.calibration` (the probe wall time is provenance and
appears only in the manifest, so the result JSON stays byte-stable) and to
`--run-manifest` under `dispatch`.
The per-stage `dispatch.route_ledger` carries `route`, `reason`, `calls`,
`candidates`, `predicted_s`, `actual_s`, `actual_over_predicted`, `bytes` and
`switches`.

Structural and arithmetic refusals (`launch`, `unsupported_plan`,
`too_few_rows`, `too_many_constants`, `ill_conditioned`, `non_finite`,
`metric_not_native`, `no_device`, `exactness`) are checked before any timing
comparison. A fast device can never turn an arithmetic failure into a CPU
route. The fit arm adds `dispatch_cost` for a valid plan the model routes to
the CPU, so a tuning decision is never reported as an arithmetic fallback.
The existing median / interquartile / maximum row floors are expressed through
the same planner with a zero-hysteresis compatibility calibration, preserving
their measured crossovers.

See [`dispatch.md`](dispatch.md) for the formulas, JSON shape and the tests.

## Device memory plan (#293)

The run plans its device memory before the first allocation. `--gpu-memory <size>` lowers the budget below `min(10.5 GiB, free - 1 GiB)`. A stage that does not fit is refused with its bytes and the smallest fix: `--gpu` exits 2, a plain run keeps the CPU with `backend_reason` starting `device_memory`, and an explicit `--fit-strategy gpu` whose f64 fit rows do not fit exits 2. `gpu.memory_plan` in `--json` records the plan. A failed device allocation is reported as `device_memory` with the byte count. The terms, the decision order and the refusal text are in [`memory-model.md`](memory-model.md).

### Device allocation ledger

Every device allocation of an evaluator goes through one tagged wrapper. It counts requests, keeps the largest request and its tag, tracks live and peak live bytes, and refuses a request that would push the live bytes over the context budget (the plan budget) with the tag and the byte count. `gpu.memory_plan.observed` in `--json` reports `requests`, `max_request_bytes`, `max_request_tag`, `live_bytes`, `peak_live_bytes`, `refused` and `refused_tag`. A refused allocation is reported as `device_memory: the device refused an allocation of N bytes (tag T) for ...`.

No production request scales with `P x N`. The n-scaled requests are the dataset (`cols`, `y`), its f64 twin (`cols_f64`, `y_f64`, `w_f64`), the label sign (`tsign`, `tsign_f64`), the rank target (`rank_b`) and the fit and score region (`fit_region`). `GpuEvaluator::predictions`, which writes `P x N` values, exists only in test builds, and `rc_predict` refuses an output above 64 MiB.

The rank and order-statistic score calls split their trials into sub-waves, so one call stays inside 512 MiB (64 bytes per row and segment) even when six trials of one candidate exceed it. The fit and score region caps its growth slack at 512 MiB when the batch itself fits.

### Throughput at the wide-panel scale

Resident scoring of 400 candidates over 4,775,757 x 20 f32 rows takes 0.139 s (median) and 4,096 candidates take 1.285 s, inside the 0.25 s and 2 s targets. Tiled scoring at 1,000,000 x 20 and P = 400 with 8,192-row tiles takes 0.95 times the resident time, because the next tile uploads while the current kernel runs. A 364-column store searched with 20 variables is `resident` and holds 0.88 GiB on the device; the numbers and the command are in [`memory-model.md`](memory-model.md).
