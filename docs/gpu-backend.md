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

# with the GPU evaluator
cargo build --release --features cuda

# benchmark GPU vs the reference CPU evaluator
./target/release/rcsr-gpu-bench --data examples/sine.csv --target y --vars x \
    --batch 8000 --depth 5 --seed 5
```

Environment overrides for the build:

| Variable | Meaning | Default |
|---|---|---|
| `RCSR_CUDA_ARCH` | target architecture passed to `nvcc` as `-arch=` | `sm_86` (Ampere) |
| `RCSR_NVCC` | `nvcc` binary to use | `nvcc` |

For a DGX Spark, build with `RCSR_CUDA_ARCH=sm_121` under CUDA 13 (see below).

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
candidates only upload the (small) bytecode and download P errors. This is the
shape that matters on a machine with unified memory.

## Measured results

Machine: Intel Core i7-3770K (4C/8T, AVX only) + NVIDIA GeForce RTX 3060
(`sm_86`, 28 SMs, 12 GB). Batch of 8 000 random depth-5 expressions; the CPU
reference is the engine's own interpreted `f64` tree walk (`cpu_sse`).

| Dataset | Rows | GPU batch | CPU batch | Speed-up | Agreement (median / p99 / within 1e-3) |
|---|---|---|---|---|---|
| `examples/sine.csv` | 60 | 0.231 ms | 63.3 ms | **274×** | 2.9e-8 / 2.3e-4 / 99.3% |
| two-variable (y = sin x₁ + x₂²) | 400 | 0.365 ms | 260.8 ms | **715×** | 4.5e-8 / 3.2e-4 / 99.3% |
| poly (y = 3x² − 2x + 1) | 100 | 0.204 ms | 57.6 ms | **283×** | 4.9e-8 / 4.9e-5 / 99.7% |
| bump (y = e^{−0.4x}cos 3x) | 120 | 0.210 ms | 98.4 ms | **468×** | 3.0e-8 / 8.7e-5 / 99.6% |

`Agreement` is the relative difference between the f32 kernel and the f64
reference, over candidates that both sides scored as ordinary finite results.
A median of ~3 × 10⁻⁸ is f32 epsilon: the kernel is computing the same thing.
The remaining ~30% of candidates are *hard failures* on both sides (random
`exp`/`cosh`/`cube` trees exceed the magnitude limit) and are excluded from the
agreement statistic, not from the search.

Two honest caveats:

* The CPU column is a **reference interpreter in `f64`**, not the engine's
  fastest CPU path. A column-oriented SIMD `f64` evaluator would narrow the gap;
  the fair summary is "hundreds of times faster than a tree walk", not
  "hundreds of times faster than the whole engine".
* The kernel time barely grows from 60 to 400 rows (0.231 → 0.365 ms), which
  says it is **latency-bound, not throughput-bound**: with only 32–256 threads
  per block and a per-thread local-memory stack, occupancy is low. Even so it is
  roughly 1.4% of the card's FP32 peak, so there is a lot of headroom left.

## Where it plugs into the search

Compiling a candidate and scoring a whole generation in one launch is the natural
integration: generate offspring → batch-evaluate on the GPU → spend the
(CPU-side) Levenberg–Marquardt budget only on the candidates whose *raw* error is
promising. That is the concrete "avoid useless work" win on top of raw
throughput. The pieces (`compile`, `GpuEvaluator`, `cpu_sse`) are in place; the
search loop still calls the CPU evaluator, so the integration is a **planned**
step rather than a shipped one.

## The DGX Spark path

| | |
|---|---|
| SoC | GB10 Grace Blackwell (CPU and GPU dies joined by NVLink-C2C) |
| CPU | 20 Arm cores (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | 48 SMs, ~31 TFLOPS FP32, **`sm_121`** |
| FP64 | 1/64 of FP32 |
| Memory | 128 GB LPDDR5X, ~273 GB/s, **coherent and unified** |
| Software | DGX OS (aarch64), CUDA 13.0+ |

Concrete plan:

1. **Native build.** `RCSR_CUDA_ARCH=sm_121`, CUDA 13 `nvcc`. Note that
   `sm_121` is a *consumer Blackwell* target; datacentre flags such as `sm_100`
   will not run on it.
2. **Unified memory.** Replace `cudaMalloc`/`cudaMemcpy` with
   `cudaMallocManaged` (or map host memory) so the dataset and the bytecode never
   need an explicit copy, and the GPU can read the CPU's population directly.
3. **Whole-generation batches.** With ~10⁶ evaluations per launch, launch
   latency disappears; batch the entire population (all islands) in one kernel.
4. **Bigger blocks, fewer per-expression split.** With 48 SMs and high
   occupancy, use larger blocks and consider splitting very wide datasets across
   several blocks per expression, with an atomic or two-pass reduction.
5. **Keep FP32, verify in FP64.** Score on the GPU in FP32; re-check the
   surviving Pareto front on the CPU in FP64 before reporting.
6. **Don't oversubscribe the memory bus.** The CPU evolution and the GPU
   evaluation share 273 GB/s; hand off between them rather than running both
   flat out.

The expected gain over an optimised AVX-512 CPU implementation of the same
evaluation is **10–40×** (reported figures for this style of workload range from
~18× to ~40×, and the measured 274–715× above is against a much slower
single-threaded reference), rising further with row tiling and CUDA graphs.
