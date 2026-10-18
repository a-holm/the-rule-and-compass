# The optional GPU backend

The search's hot loop is *score P candidate expressions against N rows of data*.
Those are P × N independent evaluations, which is exactly the shape a GPU likes.
This document describes the backend in `src/cuda/eval.cu` + `src/gpu.rs`, how to
build and run it, what it measured on the available hardware, and how the same
design maps onto an NVIDIA DGX Spark.

The backend is **opt-in**. The default build remains pure `std` with no
dependencies and no CUDA requirement.

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
same SSEs as scoring each program on its own. The search loop still owns its
own scheduling and is not wired to `Population` yet.

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


## Where it plugs into the search

Compiling a candidate and scoring a whole generation in one launch is the natural
integration: generate offspring → batch-evaluate on the GPU → spend the
(CPU-side) Levenberg–Marquardt budget only on the candidates whose *raw* error is
promising. That is the concrete "avoid useless work" win on top of raw
throughput. `Population` and `GpuEvaluator::eval_population` provide the batch
shape; wiring them into the evolutionary loop is issue #26, a separate
workstream, because that file is owned elsewhere. Until then the search loop
calls the CPU evaluator, so the integration is a **planned** step rather than a
shipped one.

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
   `eval_population` and measured here with 8 000 programs in one launch.
   Wiring the search's generation into it belongs to issue #26.
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

