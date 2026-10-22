/*
 * Batched fitness evaluation on the GPU.
 *
 * The workload is: given one dataset (N rows, V variables) and a batch of P
 * candidate expressions, compute the sum of squared errors of each expression
 * over all rows. That is P x N independent evaluations -- embarrassingly
 * parallel -- and it is the hot loop of the whole search.
 *
 * Design (mirrors the plan in docs/gpu-backend.md):
 *   - one thread block per expression, threads spread across rows;
 *   - therefore every thread in a block executes the *same* instruction stream,
 *     so there is no warp divergence caused by program structure;
 *   - operators are branchless (protected division/log/sqrt, no `if`);
 *   - the fitness reduction is fused into the kernel;
 *   - the dataset stays resident in device memory across evaluations.
 *
 * Arithmetic is float (FP32) on purpose: on both the consumer Ampere parts and
 * the GB10 in the DGX Spark, FP64 runs at 1/64 of FP32, so an FP64 kernel would
 * throw away the entire advantage. FP64 is reserved for re-checking the final
 * surviving equations on the host.
 */

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Opcodes. Keep in sync with src/gpu.rs. */
enum {
    OP_CONST = 0,  /* next word: float bits            */
    OP_VAR = 1,    /* next word: variable index        */
    OP_ADD = 2,
    OP_SUB = 3,
    OP_MUL = 4,
    OP_DIV = 5,
    OP_SIN = 6,
    OP_COS = 7,
    OP_TAN = 8,
    OP_EXP = 9,
    OP_LOG = 10,
    OP_SQRT = 11,
    OP_ABS = 12,
    OP_TANH = 13,
    OP_SIG = 14,   /* logistic                        */
    OP_SQUARE = 15,
    OP_CUBE = 16,
    OP_NEG = 17,
    OP_GAUSS = 18,
    OP_STEP = 19,
    OP_POW = 20,
    OP_SINH = 21,
    OP_COSH = 22,
    /* #131: native hard operators. Floor/ceil/round are one FRND; min/max
       use fminf/fmaxf behind an explicit NaN guard to match expr.rs; if is a
       select; mod is the host's floored formula; the transcendental ops call
       libdevice. */
    OP_FLOOR = 23,
    OP_CEIL = 24,
    OP_ROUND = 25,
    OP_MOD = 26,
    OP_ATAN2 = 27,
    OP_ERF = 28,
    OP_ERFC = 29,
    OP_MIN = 30,
    OP_MAX = 31,
    OP_IF = 32,
    OP_SIGN = 33,
    OP_EQUAL = 34,
    OP_LESS = 35,
    OP_LESS_OR_EQUAL = 36,
    OP_GREATER = 37,
    OP_GREATER_OR_EQUAL = 38,
    OP_AND = 39,
    OP_OR = 40,
    OP_XOR = 41,
    OP_NOT = 42,
    OP_HALT = 255
};

#define RC_MAX_STACK 64
#define RC_MAX_THREADS 256
#define RC_MIN_THREADS 32

/* #118: host-side ledger for one batch, in milliseconds. The windows are
 * disjoint: allocation (including synchronising cudaMalloc/cudaFree), H2D
 * upload, kernel launch, event synchronisation and D2H download. Their sum is
 * the measured wall, so a report can check the decomposition instead of
 * assuming it. `kernel_ms` is GPU-event time *inside* that wall; it is the
 * numerator of the duty estimate, never another additive host phase. */
typedef struct RcStats {
    double kernel_ms;
    double launch_ms;
    double sync_ms;
    double alloc_ms;
    double h2d_ms;
    double d2h_ms;
    double wall_ms;
    unsigned long long h2d_bytes;
    unsigned long long d2h_bytes;
    int batch;
    int failed;
} RcStats;

static double rc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

struct RcCtx {
    float *d_cols;
    float *d_y;
    /* #169: f64 twins used by the exact polish of the batched fit. They are
       allocated lazily on the first fit call, so a search-only run never pays
       for them. */
    double *d_cols64;
    double *d_y64;
    /* #229: the base `--weight` row vector for the batched fit (null when the
       run is unweighted). Uploaded with the f64 fit rows, so the fit kernel's
       per-row IRLS weight can be `sqrt(w_i * 2/(1 + r^2))`. */
    double *d_w64;
    unsigned int *d_prog;
    unsigned int *d_off;
    float *d_out;
    /* #122: fixed-capacity, double-buffered genome arena. */
    unsigned int *d_pool[2];
    unsigned long long *d_hash;
    int pool_cap;
    int pool_width;
    int hash_cap;
    /* #230: cached target ranks. `B` is twice the centred average rank of `y`
       (an integer), computed once per fit sample; the rank-correlation stage
       reuses it for every trial instead of re-sorting the target. */
    long long *d_rank_b;
    int rank_ready;
    int n;
    int v;
    int nprog;
    int out_cap;
    cudaEvent_t t0, t1;
    /* #118: the current batch's ledger and its start on the same monotonic
       clock, so `wall_ms` covers the whole begin..download path. */
    RcStats stats;
    double batch_start_ms;
};

__device__ __forceinline__ float logisticf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* #121: one row of a postfix program, shared by the offset kernel and the
   fixed-stride pool kernel so both execute the same instruction set.

   The 64-slot stack is guarded on the device as well as on the host: a push
   past `RC_MAX_STACK` or an operator reading below the current depth returns
   NaN, which both kernels score as a hard failure. Without these checks a
   program emitted without the host's `peak_stack` gate would write past `s`
   and corrupt device state instead of failing that candidate. `sp > 0` on
   HALT means a missing operand (or an empty program) is a hard failure too. */
__device__ __forceinline__ float rc_exec_row(const unsigned int *__restrict__ code,
                                             const float *__restrict__ cols,
                                             int n,
                                             int row)
{
    float s[RC_MAX_STACK];
    int sp = 0;
    for (;;) {
        unsigned int w = *code++;
        if (w == OP_CONST) {
            if (sp >= RC_MAX_STACK) return CUDART_NAN_F;
            s[sp++] = __uint_as_float(*code++);
        } else if (w == OP_VAR) {
            unsigned int k = *code++;
            if (sp >= RC_MAX_STACK) return CUDART_NAN_F;
            s[sp++] = cols[(size_t)k * (size_t)n + (size_t)row];
        } else if (w == OP_HALT) {
            return sp > 0 ? s[0] : CUDART_NAN_F;
        } else if (w == OP_ADD) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            s[sp - 1] += b;
        } else if (w == OP_SUB) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            s[sp - 1] -= b;
        } else if (w == OP_MUL) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            s[sp - 1] *= b;
        } else if (w == OP_DIV) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            /* Match the host evaluator: a near-zero denominator is a domain
               error (NaN), which the host then scores as a hard failure. */
            s[sp - 1] = (fabsf(b) < 1e-12f) ? CUDART_NAN_F : s[sp - 1] / b;
        } else if (w == OP_POW) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            float a = s[sp - 1];
            s[sp - 1] = (a < 0.0f && fabsf(b - roundf(b)) > 1e-9f)
                            ? CUDART_NAN_F
                            : powf(a, b);
        } else if (w == OP_MOD) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            float a = s[sp - 1];
            /* Host semantics: floored remainder with the same guards as
               expr::eval. `floorf(q)` (not fmodf) makes the sign follow the
               divisor. The quotient is f32, so above |a/b| = 2^25 its
               fractional part is not representable and the result is no
               longer the f64 remainder (documented in docs/gpu-backend.md;
               the final archive is re-scored on the CPU). For the range the
               search can use, one range correction removes the single ULP of
               quotient rounding that can push r across 0 or |b|. */
            float r = CUDART_NAN_F;
            if (isfinite(a) && isfinite(b) && fabsf(b) >= 1e-12f) {
                float q = a / b;
                if (isfinite(q)) {
                    r = fmaf(-b, floorf(q), a);
                    float ab = fabsf(b);
                    float rs = (b > 0.0f) ? r : -r;
                    if (rs < 0.0f) {
                        r += b;
                    } else if (rs >= ab) {
                        r -= b;
                    }
                }
            }
            s[sp - 1] = r;
        } else if (w == OP_ATAN2) {
            if (sp < 2) return CUDART_NAN_F;
            float x = s[--sp];
            float y = s[sp - 1];
            s[sp - 1] = (isnan(y) || isnan(x)) ? CUDART_NAN_F : atan2f(y, x);
        } else if (w == OP_MIN || w == OP_MAX) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            float a = s[sp - 1];
            if (isnan(a) || isnan(b)) {
                s[sp - 1] = CUDART_NAN_F;
            } else {
                s[sp - 1] = (w == OP_MIN) ? fminf(a, b) : fmaxf(a, b);
            }
        } else if (w == OP_EQUAL || w == OP_LESS || w == OP_LESS_OR_EQUAL
                   || w == OP_GREATER || w == OP_GREATER_OR_EQUAL) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            float a = s[sp - 1];
            if (isnan(a) || isnan(b)) {
                s[sp - 1] = CUDART_NAN_F;
            } else {
                int r = 0;
                switch (w) {
                    case OP_EQUAL: r = a == b; break;
                    case OP_LESS: r = a < b; break;
                    case OP_LESS_OR_EQUAL: r = a <= b; break;
                    case OP_GREATER: r = a > b; break;
                    default: r = a >= b; break;
                }
                s[sp - 1] = r ? 1.0f : 0.0f;
            }
        } else if (w == OP_AND || w == OP_OR || w == OP_XOR) {
            if (sp < 2) return CUDART_NAN_F;
            float b = s[--sp];
            float a = s[sp - 1];
            if (isnan(a) || isnan(b)) {
                s[sp - 1] = CUDART_NAN_F;
            } else {
                int r = 0;
                switch (w) {
                    case OP_AND: r = (a > 0.0f) && (b > 0.0f); break;
                    case OP_OR: r = (a > 0.0f) || (b > 0.0f); break;
                    default: r = (a > 0.0f) != (b > 0.0f); break;
                }
                s[sp - 1] = r ? 1.0f : 0.0f;
            }
        } else if (w == OP_IF) {
            /* Both arms have already been evaluated on the stack; a true
               select discards the unselected value, including NaN. */
            if (sp < 3) return CUDART_NAN_F;
            float b = s[--sp];
            float a = s[--sp];
            s[sp - 1] = (s[sp - 1] > 0.0f) ? a : b;
        } else {
            /* unary operators */
            if (sp < 1) return CUDART_NAN_F;
            float a = s[sp - 1];
            float r = a;
            switch (w) {
                case OP_SIN: r = sinf(a); break;
                case OP_COS: r = cosf(a); break;
                case OP_TAN: {
                    float t = tanf(a);
                    r = isfinite(t) ? t : CUDART_NAN_F;
                } break;
                case OP_EXP: r = expf(a); break;
                case OP_LOG: r = (a > 0.0f) ? logf(a) : CUDART_NAN_F; break;
                case OP_SQRT: r = (a >= 0.0f) ? sqrtf(a) : CUDART_NAN_F; break;
                case OP_ABS: r = fabsf(a); break;
                case OP_TANH: r = tanhf(a); break;
                case OP_SIG: r = logisticf(a); break;
                case OP_SQUARE: r = a * a; break;
                case OP_CUBE: r = a * a * a; break;
                case OP_NEG: r = -a; break;
                case OP_GAUSS: r = expf(-a * a); break;
                case OP_STEP: r = a > 0.0f ? 1.0f : 0.0f; break;
                case OP_SINH: r = sinhf(a); break;
                case OP_COSH: r = coshf(a); break;
                case OP_FLOOR: r = floorf(a); break;
                case OP_CEIL: r = ceilf(a); break;
                case OP_ROUND: r = roundf(a); break;
                case OP_ERF:
                    /* expr::erf returns +0 for -0; erff preserves the
                       sign bit, so normalise zeros. */
                    r = (a == 0.0f) ? 0.0f : erff(a);
                    break;
                case OP_ERFC: r = erfcf(a); break;
                case OP_SIGN:
                    r = isnan(a) ? CUDART_NAN_F : (a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f));
                    break;
                case OP_NOT:
                    r = isnan(a) ? CUDART_NAN_F : (a > 0.0f ? 0.0f : 1.0f);
                    break;
                default: r = CUDART_NAN_F; break;
            }
            s[sp - 1] = r;
        }
    }
}

/* --- #229: the loss the evaluator accumulates -----------------------------
 *
 * #236 closed the gate on every non-squared metric because the evaluator only
 * summed squared error: a slope/absolute/rank run was silently ranked by MSE.
 * LogSquashed is admitted again only because the evaluator computes its own
 * objective, `Σ ln(1 + r^2)`, per row. The host still divides by the row count,
 * so the number the search ranks on is exactly `Metric::LogSquashed`'s mean
 * loss (up to f32 rounding); reported numbers are rebuilt in f64 by the CPU.
 *
 * `log1pf` is a 1-ULP single-precision device function (CUDA C++ Programming
 * Guide, mathematical functions). The f32 residual square overflows above
 * |r| = 1.8e19, where `ln(1+r^2)` is `2 ln|r|` to within 1e-38, so that branch
 * is both cheaper and correct.
 */
#define RC_LOSS_SQUARED 0
#define RC_LOSS_LOG_SQUASHED 1
/* #224: fit-only correlation objectives (the evaluator still has no native
   correlation scorer; these codes are never passed to rc_eval/rc_pool_eval). */
#define RC_LOSS_CORRELATION 2
#define RC_LOSS_CORRELATION_SIGNED 3

__device__ __forceinline__ float rc_row_loss(float r, int loss)
{
    if (loss == RC_LOSS_LOG_SQUASHED) {
        const float rr = r * r;
        return isfinite(rr) ? log1pf(rr) : 2.0f * logf(fabsf(r));
    }
    return r * r;
}

/* #131 register contract: 6 blocks of RC_MAX_THREADS per SM. On sm_86 the
   thread limit (1536) allows 6 blocks of 256, so 6 is the occupancy target;
   ptxas derives a hard ~40-register ceiling from it. Crossing that ceiling is
   a compile-time error/spill, not a silent occupancy loss. */
__global__ void __launch_bounds__(RC_MAX_THREADS, 6) eval_kernel(const float *__restrict__ cols,
                            const float *__restrict__ y,
                            int n,
                            int v,
                            const unsigned int *__restrict__ prog,
                            const unsigned int *__restrict__ off,
                            float *__restrict__ out,
                            int loss)
{
    const int p = blockIdx.x;
    float acc = 0.0f;
    const unsigned int *code = prog + off[p];

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float pred = rc_exec_row(code, cols, n, i);
        /* Shared magnitude limit: anything non-finite or astronomically large is
           a hard failure. Applying the *same* rule on the host keeps the f32
           kernel and the f64 reference on the same fitness landscape. */
        if (!isfinite(pred) || fabsf(pred) > 1e30f) {
            acc += 1e30f;
            continue;
        }
        float r = pred - y[i];
        acc += rc_row_loss(r, loss);
    }

    __shared__ float sh[RC_MAX_THREADS];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[p] = sh[0];
}

/* #122: the same evaluator over a fixed-stride genome pool. One block per
   program; slot `p` starts at `p * width` and ends at its HALT word. */
__global__ void __launch_bounds__(RC_MAX_THREADS, 6) eval_pool_kernel(const float *__restrict__ cols,
                                 const float *__restrict__ y,
                                 int n,
                                 int v,
                                 const unsigned int *__restrict__ prog,
                                 int width,
                                 float *__restrict__ out,
                                 int loss)
{
    const int p = blockIdx.x;
    const unsigned int *code = prog + (size_t)p * (size_t)width;
    float acc = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float pred = rc_exec_row(code, cols, n, i);
        if (!isfinite(pred) || fabsf(pred) > 1e30f) {
            acc += 1e30f;
            continue;
        }
        float r = pred - y[i];
        acc += rc_row_loss(r, loss);
    }

    __shared__ float sh[RC_MAX_THREADS];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[p] = sh[0];
}

/* #131 fixtures: raw per-row predictions of a packed batch, P x N row-major.
   Unlike the fitness kernels this does not apply the hard-failure magnitude
   limit, so a fixture can see NaN, infinities and signed zeros. */
__global__ void __launch_bounds__(RC_MAX_THREADS, 6) predict_kernel(const float *__restrict__ cols,
                               int n,
                               const unsigned int *__restrict__ prog,
                               const unsigned int *__restrict__ off,
                               float *__restrict__ out)
{
    const int p = blockIdx.x;
    const unsigned int *pr = prog + off[p];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[(size_t)p * (size_t)n + (size_t)i] = rc_exec_row(pr, cols, n, i);
    }
}

/* --- #169: batched Levenberg-Marquardt constant fitting ------------------
 *
 * One block per candidate runs the whole LM loop: value + reverse-mode
 * Jacobian for the p constants, J^T J / J^T r / SSE accumulation, a damped
 * f64 Cholesky solve, a trial-SSE sweep and the accept/reject sequence,
 * all without a grid barrier. The block reduction is a fixed-order,
 * deterministic sum (no atomics). Per-row arithmetic is f32; every
 * accumulator and the solve are f64.
 *
 * A program is a postorder array of 4-word records [op, kid0, kid1, slot].
 * Constants are per-candidate f64 values in shared memory and are what the
 * loop moves; the block writes the final constants, SSE and status back.
 */

/* Two instantiations, selected by the batch's largest constant count. The
 * shared footprint grows with both the constant cap and the row chunk:
 * p=20 with a 256-row chunk needs about 50.8 KB, above the 48 KB static limit
 * and only one block per SM, while the 128-row chunk keeps it near 38.1 KB so
 * two blocks still fit per SM on sm_86. p<=16 keeps the 256-row chunk it was
 * measured with. Both instantiate the same code, and every statistic is
 * accumulated over rows in ascending index order by one thread, so the chunk
 * size changes which thread owns which row, never the sum: a candidate's
 * constants are bit-identical whichever instantiation fits it. */
#define FT_MAX_NODES 96
#define FT_MAX_CONST_SMALL 16
#define FT_CHUNK_SMALL 256
#define FT_MAX_CONST_LARGE 20
#define FT_CHUNK_LARGE 128
#define FT_PCHUNK 128
/* Lower triangle + right-hand side + SSE. */
#define FT_STATS(MAXC) (((MAXC) * ((MAXC) + 1) / 2) + (MAXC) + 1)

/* Dynamic shared block of one instantiation. The kernel takes it from
 * `extern __shared__` so the same code serves both sizes and the p=20 block is
 * not limited by the 48 KB static ceiling. */
template <int MAXC, int CHUNK, int PCHUNK>
struct FtSmem {
    double sc[MAXC];
    double sjtj[FT_STATS(MAXC)];
    double sjtr[MAXC];
    double sL[MAXC * MAXC];
    double strial[MAXC];
    double sdelta[MAXC];
    double sred[CHUNK];
    float sg[CHUNK * MAXC];
    float sr[CHUNK];
    double sdl[CHUNK];
    double sdg[PCHUNK * MAXC];
    double sdr[PCHUNK];
    double sdl64[PCHUNK];
    /* #224: correlation moments and conditioning probes. */
    double scorr[6 + 3 * MAXC];
    double svar;
    double syy;
    double srho;
    int sbad;
    int ssolved;
    double slambda;
    double sbest;
};

__device__ __forceinline__ void ft_forward(const unsigned int *nd,
                                           int m,
                                           const float *__restrict__ cols,
                                           int n,
                                           int row,
                                           const double *c,
                                           float *v)
{
    for (int i = 0; i < m; ++i) {
        const unsigned int op = nd[4 * i];
        const unsigned int k0 = nd[4 * i + 1];
        const unsigned int k1 = nd[4 * i + 2];
        const unsigned int slot = nd[4 * i + 3];
        if (op == OP_CONST) {
            v[i] = (float)c[slot];
        } else if (op == OP_VAR) {
            v[i] = cols[(size_t)slot * (size_t)n + (size_t)row];
        } else if (op == OP_ADD) {
            v[i] = v[k0] + v[k1];
        } else if (op == OP_SUB) {
            v[i] = v[k0] - v[k1];
        } else if (op == OP_MUL) {
            v[i] = v[k0] * v[k1];
        } else if (op == OP_DIV) {
            float b = v[k1];
            v[i] = (fabsf(b) < 1e-12f) ? CUDART_NAN_F : v[k0] / b;
        } else if (op == OP_POW) {
            float a = v[k0];
            float b = v[k1];
            v[i] = (a < 0.0f && fabsf(b - roundf(b)) > 1e-9f) ? CUDART_NAN_F
                                                              : powf(a, b);
        } else {
            float a = v[k0];
            float r = a;
            switch (op) {
                case OP_SIN: r = sinf(a); break;
                case OP_COS: r = cosf(a); break;
                case OP_TAN: {
                    float t = tanf(a);
                    r = isfinite(t) ? t : CUDART_NAN_F;
                } break;
                case OP_EXP: r = (a > 700.0f) ? CUDART_INF_F : expf(a); break;
                case OP_LOG: r = (a > 0.0f) ? logf(a) : CUDART_NAN_F; break;
                case OP_SQRT: r = (a >= 0.0f) ? sqrtf(a) : CUDART_NAN_F; break;
                case OP_ABS: r = fabsf(a); break;
                case OP_TANH: r = tanhf(a); break;
                case OP_SIG: r = logisticf(a); break;
                case OP_SQUARE: r = a * a; break;
                case OP_CUBE: r = a * a * a; break;
                case OP_NEG: r = -a; break;
                case OP_GAUSS: r = expf(-a * a); break;
                case OP_SINH: r = sinhf(a); break;
                case OP_COSH: r = coshf(a); break;
                case OP_ERF: r = (a == 0.0f) ? 0.0f : erff(a); break;
                case OP_ERFC: r = erfcf(a); break;
                default: r = CUDART_NAN_F; break;
            }
            v[i] = r;
        }
    }
}

/* Reverse-mode adjoint sweep of the same program. `g[j] = d root / d c_j`. */
template <int MAXC>
__device__ __forceinline__ void ft_backward(const unsigned int *nd,
                                            int m,
                                            const float *v,
                                            float *adj,
                                            float *g,
                                            int p)
{
    for (int i = 0; i < m; ++i) adj[i] = 0.0f;
    for (int j = 0; j < p; ++j) g[j] = 0.0f;
    adj[m - 1] = 1.0f;
    for (int i = m - 1; i >= 0; --i) {
        const float a = adj[i];
        if (a == 0.0f) continue;
        const unsigned int op = nd[4 * i];
        const unsigned int k0 = nd[4 * i + 1];
        const unsigned int k1 = nd[4 * i + 2];
        const unsigned int slot = nd[4 * i + 3];
        if (op == OP_CONST) {
            g[slot] += a;
            continue;
        }
        if (op == OP_VAR) continue;
        const float x = v[k0];
        const float y = v[k1];
        switch (op) {
            case OP_NEG: adj[k0] += -a; break;
            case OP_EXP: adj[k0] += a * expf(x); break;
            case OP_LOG: adj[k0] += a / x; break;
            case OP_ABS: adj[k0] += a * (x >= 0.0f ? 1.0f : -1.0f); break;
            case OP_SQRT: adj[k0] += a / (2.0f * sqrtf(x)); break;
            case OP_SIN: adj[k0] += a * cosf(x); break;
            case OP_COS: adj[k0] += -a * sinf(x); break;
            case OP_TAN: {
                const float t = cosf(x);
                adj[k0] += a / (t * t);
            } break;
            case OP_SINH: adj[k0] += a * coshf(x); break;
            case OP_COSH: adj[k0] += a * sinhf(x); break;
            case OP_TANH: {
                const float t = tanhf(x);
                adj[k0] += a * (1.0f - t * t);
            } break;
            case OP_SIG: {
                const float s = logisticf(x);
                adj[k0] += a * s * (1.0f - s);
            } break;
            case OP_GAUSS: adj[k0] += a * (-2.0f * x) * expf(-x * x); break;
            case OP_SQUARE: adj[k0] += a * 2.0f * x; break;
            case OP_CUBE: adj[k0] += a * 3.0f * x * x; break;
            case OP_ERF:
                adj[k0] += a * 1.1283791670955126f * expf(-x * x);
                break;
            case OP_ERFC:
                adj[k0] += -a * 1.1283791670955126f * expf(-x * x);
                break;
            case OP_ADD:
                adj[k0] += a;
                adj[k1] += a;
                break;
            case OP_SUB:
                adj[k0] += a;
                adj[k1] += -a;
                break;
            case OP_MUL:
                adj[k0] += a * y;
                adj[k1] += a * x;
                break;
            case OP_DIV:
                adj[k0] += a / y;
                adj[k1] += -a * x / (y * y);
                break;
            case OP_POW:
                if (x > 0.0f) {
                    adj[k0] += a * y * powf(x, y - 1.0f);
                    adj[k1] += a * powf(x, y) * logf(x);
                }
                break;
            default: break;
        }
    }
}


/* f64 twins of the forward/adjoint/assembly/sse passes, used by the final
 * undamped polish. The LM search itself keeps f32 row arithmetic; the last
 * four Gauss-Newton steps run in f64 so the returned constants are not limited
 * by the f32 derivative floor. */
__device__ __forceinline__ void ft_forward_d(const unsigned int *nd,
                                             int m,
                                             const double *__restrict__ cols,
                                             int n,
                                             int row,
                                             const double *c,
                                             double *v)
{
    for (int i = 0; i < m; ++i) {
        const unsigned int op = nd[4 * i];
        const unsigned int k0 = nd[4 * i + 1];
        const unsigned int k1 = nd[4 * i + 2];
        const unsigned int slot = nd[4 * i + 3];
        if (op == OP_CONST) {
            v[i] = c[slot];
        } else if (op == OP_VAR) {
            v[i] = cols[(size_t)slot * (size_t)n + (size_t)row];
        } else if (op == OP_ADD) {
            v[i] = v[k0] + v[k1];
        } else if (op == OP_SUB) {
            v[i] = v[k0] - v[k1];
        } else if (op == OP_MUL) {
            v[i] = v[k0] * v[k1];
        } else if (op == OP_DIV) {
            double b = v[k1];
            v[i] = (fabs(b) < 1e-12) ? CUDART_NAN : v[k0] / b;
        } else if (op == OP_POW) {
            double a = v[k0];
            double b = v[k1];
            v[i] = (a < 0.0 && fabs(b - round(b)) > 1e-9) ? CUDART_NAN
                                                           : pow(a, b);
        } else {
            double a = v[k0];
            double r = a;
            switch (op) {
                case OP_SIN: r = sin(a); break;
                case OP_COS: r = cos(a); break;
                case OP_TAN: {
                    double t = tan(a);
                    r = isfinite(t) ? t : CUDART_NAN;
                } break;
                case OP_EXP: r = (a > 700.0) ? CUDART_INF : exp(a); break;
                case OP_LOG: r = (a > 0.0) ? log(a) : CUDART_NAN; break;
                case OP_SQRT: r = (a >= 0.0) ? sqrt(a) : CUDART_NAN; break;
                case OP_ABS: r = fabs(a); break;
                case OP_TANH: r = tanh(a); break;
                case OP_SIG: r = 1.0 / (1.0 + exp(-a)); break;
                case OP_SQUARE: r = a * a; break;
                case OP_CUBE: r = a * a * a; break;
                case OP_NEG: r = -a; break;
                case OP_GAUSS: r = exp(-a * a); break;
                case OP_SINH: r = sinh(a); break;
                case OP_COSH: r = cosh(a); break;
                case OP_ERF: r = (a == 0.0) ? 0.0 : erf(a); break;
                case OP_ERFC: r = erfc(a); break;
                default: r = CUDART_NAN; break;
            }
            v[i] = r;
        }
    }
}

template <int MAXC>
__device__ __forceinline__ void ft_backward_d(const unsigned int *nd,
                                              int m,
                                              const double *v,
                                              double *adj,
                                              double *g,
                                              int p)
{
    for (int i = 0; i < m; ++i) adj[i] = 0.0;
    for (int j = 0; j < p; ++j) g[j] = 0.0;
    adj[m - 1] = 1.0;
    for (int i = m - 1; i >= 0; --i) {
        const double a = adj[i];
        if (a == 0.0) continue;
        const unsigned int op = nd[4 * i];
        const unsigned int k0 = nd[4 * i + 1];
        const unsigned int k1 = nd[4 * i + 2];
        const unsigned int slot = nd[4 * i + 3];
        if (op == OP_CONST) {
            g[slot] += a;
            continue;
        }
        if (op == OP_VAR) continue;
        const double x = v[k0];
        const double y = v[k1];
        switch (op) {
            case OP_NEG: adj[k0] += -a; break;
            case OP_EXP: adj[k0] += a * exp(x); break;
            case OP_LOG: adj[k0] += a / x; break;
            case OP_ABS: adj[k0] += a * (x >= 0.0 ? 1.0 : -1.0); break;
            case OP_SQRT: adj[k0] += a / (2.0 * sqrt(x)); break;
            case OP_SIN: adj[k0] += a * cos(x); break;
            case OP_COS: adj[k0] += -a * sin(x); break;
            case OP_TAN: {
                const double t = cos(x);
                adj[k0] += a / (t * t);
            } break;
            case OP_SINH: adj[k0] += a * cosh(x); break;
            case OP_COSH: adj[k0] += a * sinh(x); break;
            case OP_TANH: {
                const double t = tanh(x);
                adj[k0] += a * (1.0 - t * t);
            } break;
            case OP_SIG: {
                const double s = 1.0 / (1.0 + exp(-x));
                adj[k0] += a * s * (1.0 - s);
            } break;
            case OP_GAUSS: adj[k0] += a * (-2.0 * x) * exp(-x * x); break;
            case OP_SQUARE: adj[k0] += a * 2.0 * x; break;
            case OP_CUBE: adj[k0] += a * 3.0 * x * x; break;
            case OP_ERF: adj[k0] += a * 1.1283791670955126 * exp(-x * x); break;
            case OP_ERFC: adj[k0] += -a * 1.1283791670955126 * exp(-x * x); break;
            case OP_ADD:
                adj[k0] += a;
                adj[k1] += a;
                break;
            case OP_SUB:
                adj[k0] += a;
                adj[k1] += -a;
                break;
            case OP_MUL:
                adj[k0] += a * y;
                adj[k1] += a * x;
                break;
            case OP_DIV:
                adj[k0] += a / y;
                adj[k1] += -a * x / (y * y);
                break;
            case OP_POW:
                if (x > 0.0) {
                    adj[k0] += a * y * pow(x, y - 1.0);
                    adj[k1] += a * pow(x, y) * log(x);
                }
                break;
            default: break;
        }
    }
}

/* Entries owned by one thread when the block is narrower than the statistic
 * count: p=20 needs 231 entries but the 128-row-chunk instantiation has 128
 * threads, so each thread owns at most two. */
#define FT_OWNED(MAXC, CHUNK) (((FT_STATS(MAXC)) + (CHUNK) - 1) / (CHUNK))

/* Decode statistic `s` of a p-constant system: 0 = J^T J entry (ei, ej),
 * 1 = J^T r entry, 2 = the objective `Σ w ρ(r)` (SSE for `squared`, the
 * log-squashed loss for `log_squashed`). */
__device__ __forceinline__ int ft_stat_kind(int s, int tri, int p, int *ei, int *ej)
{
    if (s < tri) {
        int k = 0;
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j <= i; ++j, ++k) {
                if (k == s) {
                    *ei = i;
                    *ej = j;
                    return 0;
                }
            }
        }
        return 0;
    }
    if (s < tri + p) return 1;
    return 2;
}

template <int MAXC, int PCHUNK>
__device__ void ft_assemble_d(const unsigned int *nd,
                              int m,
                              const double *__restrict__ cols,
                              const double *__restrict__ y,
                              const double *__restrict__ w,
                              int n,
                              int p,
                              int metric,
                              const double *c,
                              double *dg,
                              double *dr,
                              double *dl,
                              double *jtj,
                              double *jtr,
                              double *ploss,
                              int *bad)
{
    const int tid = threadIdx.x;
    const int tri = p * (p + 1) / 2;
    const int total = tri + p + 1;
    /* Thread `t` owns entries `t`, `t + PCHUNK`, ... A block narrower than the
       statistic count therefore still covers every entry, and each entry is
       accumulated by one thread over rows in ascending order: ownership moves
       a statistic between threads, it never changes its sum. */
    int own[FT_OWNED(MAXC, PCHUNK)];
    int own_kind[FT_OWNED(MAXC, PCHUNK)];
    int own_i[FT_OWNED(MAXC, PCHUNK)];
    int own_j[FT_OWNED(MAXC, PCHUNK)];
    double acc[FT_OWNED(MAXC, PCHUNK)];
    int nown = 0;
    for (int s = tid; s < total; s += PCHUNK) {
        own[nown] = s;
        own_kind[nown] = ft_stat_kind(s, tri, p, &own_i[nown], &own_j[nown]);
        acc[nown] = 0.0;
        ++nown;
    }
    for (int base = 0; base < n; base += PCHUNK) {
        const int rows = min(PCHUNK, n - base);
        if (tid < PCHUNK) {
            const int row = base + tid;
            const double wr = (row < n && w) ? w[row] : 1.0;
            if (row < n && wr > 0.0) {
                double v[FT_MAX_NODES];
                double adj[FT_MAX_NODES];
                double g[MAXC];
                ft_forward_d(nd, m, cols, n, row, c, v);
                const double pred = v[m - 1];
                const double r = pred - y[row];
                bool ok = isfinite(pred) && fabs(pred) <= 1e30 && isfinite(r);
                ft_backward_d<MAXC>(nd, m, v, adj, g, p);
                /* #229: the MM/IRLS weight. `2/(1 + r^2)` is `rho'(r)/r`; the
                   CPU arm uses `1/(1 + r^2)` and the constant 2 cannot move the
                   weighted least-squares argmin. The weight scales the whole
                   row (adjoint and residual), so J^T W J stays PSD even where
                   the true Hessian is indefinite. */
                double sw = wr;
                if (metric == RC_LOSS_LOG_SQUASHED) sw = wr * 2.0 / (1.0 + r * r);
                const double s = sqrt(sw);
                for (int j = 0; j < p; ++j) {
                    if (!isfinite(g[j])) ok = false;
                    dg[tid * p + j] = ok ? s * g[j] : 0.0;
                }
                dr[tid] = ok ? s * r : 0.0;
                dl[tid] = ok ? wr * (metric == RC_LOSS_LOG_SQUASHED
                                         ? log1p(r * r)
                                         : r * r)
                             : 0.0;
                if (!ok) *bad = 1;
            } else {
                for (int j = 0; j < p; ++j) dg[tid * p + j] = 0.0;
                dr[tid] = 0.0;
                dl[tid] = 0.0;
            }
        }
        __syncthreads();
        for (int o = 0; o < nown; ++o) {
            if (own_kind[o] == 0) {
                const int ei = own_i[o];
                const int ej = own_j[o];
                for (int k = 0; k < rows; ++k) acc[o] += dg[k * p + ei] * dg[k * p + ej];
            } else if (own_kind[o] == 1) {
                const int which = own[o] - tri;
                for (int k = 0; k < rows; ++k) acc[o] += dg[k * p + which] * dr[k];
            } else {
                for (int k = 0; k < rows; ++k) acc[o] += dl[k];
            }
        }
        __syncthreads();
    }
    for (int o = 0; o < nown; ++o) {
        if (own_kind[o] == 0) {
            jtj[own[o]] = acc[o];
        } else if (own_kind[o] == 1) {
            jtr[own[o] - tri] = acc[o];
        } else {
            *ploss = acc[o];
        }
    }
    __syncthreads();
}

/* Value-only objective sweep of `c` for the trial step. Fixed-order f64
 * reduction: `Σ w ρ(r)` with ρ the configured loss. */
template <int PCHUNK>
__device__ double ft_loss_d(const unsigned int *nd,
                            int m,
                            const double *__restrict__ cols,
                            const double *__restrict__ y,
                            const double *__restrict__ w,
                            int n,
                            int metric,
                            const double *c,
                            double *red)
{
    const int tid = threadIdx.x;
    double local = 0.0;
    if (tid < PCHUNK) {
        for (int row = tid; row < n; row += PCHUNK) {
            const double wr = w ? w[row] : 1.0;
            /* A zero-weight row is invisible: it is skipped before any
               arithmetic so a non-finite prediction there cannot poison the
               sum (`0 * inf`), exactly like `metrics::score_weighted`. */
            if (wr <= 0.0) continue;
            double v[FT_MAX_NODES];
            ft_forward_d(nd, m, cols, n, row, c, v);
            const double pred = v[m - 1];
            const double r = pred - y[row];
            if (!isfinite(pred) || fabs(pred) > 1e30 || !isfinite(r)) {
                local += 1e30;
            } else {
                local += wr * (metric == RC_LOSS_LOG_SQUASHED ? log1p(r * r) : r * r);
            }
        }
    }
    red[tid] = local;
    __syncthreads();
    for (int st = PCHUNK / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const double total = red[0];
    __syncthreads();
    return total;
}

/* Assemble J^T J, J^T r and the objective at `c`. Every statistic is owned by
 * one thread and accumulated in block-local fixed row order, so the reduction is
 * deterministic and needs no atomics. When the block is narrower than the
 * statistic count a thread owns several entries (see ft_assemble_d).
 *
 * #229: the objective statistic is `Σ w ρ(r)` in f64, one add per row. The
 * per-row ρ is the f32 `log1pf` (a comparator, never a reported number);
 * `squared` keeps the historical `r * r`, so its trajectory is unchanged. */
template <int MAXC, int CHUNK>
__device__ void ft_assemble(const unsigned int *nd,
                            int m,
                            const float *__restrict__ cols,
                            const float *__restrict__ y,
                            const double *__restrict__ w,
                            int n,
                            int p,
                            int metric,
                            const double *c,
                            float *sg,
                            float *sr,
                            double *sl,
                            double *jtj,
                            double *jtr,
                            double *ploss,
                            int *bad)
{
    const int tid = threadIdx.x;
    const int tri = p * (p + 1) / 2;
    const int total = tri + p + 1;
    int own[FT_OWNED(MAXC, CHUNK)];
    int own_kind[FT_OWNED(MAXC, CHUNK)];
    int own_i[FT_OWNED(MAXC, CHUNK)];
    int own_j[FT_OWNED(MAXC, CHUNK)];
    double acc[FT_OWNED(MAXC, CHUNK)];
    int nown = 0;
    for (int s = tid; s < total; s += CHUNK) {
        own[nown] = s;
        own_kind[nown] = ft_stat_kind(s, tri, p, &own_i[nown], &own_j[nown]);
        acc[nown] = 0.0;
        ++nown;
    }
    for (int base = 0; base < n; base += CHUNK) {
        const int rows = min(CHUNK, n - base);
        const int row = base + tid;
        const double wr = (row < n && w) ? w[row] : 1.0;
        if (row < n && wr > 0.0) {
            float vt[FT_MAX_NODES];
            float adj[FT_MAX_NODES];
            float g[MAXC];
            ft_forward(nd, m, cols, n, row, c, vt);
            const float pred = vt[m - 1];
            const float r = pred - y[row];
            bool ok = isfinite(pred) && fabsf(pred) <= 1e30f && isfinite(r);
            ft_backward<MAXC>(nd, m, vt, adj, g, p);
            /* #229: `sqrt(w_i · 2/(1 + r_i^2))` scales the row's adjoint and
               residual, so the rank-one update below is unchanged and `A`
               stays positive semidefinite even where ρ'' < 0. */
            double sw = wr;
            if (metric == RC_LOSS_LOG_SQUASHED) {
                const double rr = (double)r * (double)r;
                sw = wr * 2.0 / (1.0 + rr);
            }
            const float s = sqrtf((float)sw);
            for (int j = 0; j < p; ++j) {
                if (!isfinite(g[j])) ok = false;
                sg[tid * p + j] = ok ? s * g[j] : 0.0f;
            }
            sr[tid] = ok ? s * r : 0.0f;
            sl[tid] = ok ? (double)wr * (double)rc_row_loss(r, metric) : 0.0;
            if (!ok) *bad = 1;
        } else {
            for (int j = 0; j < p; ++j) sg[tid * p + j] = 0.0f;
            sr[tid] = 0.0f;
            sl[tid] = 0.0;
        }
        __syncthreads();
        for (int o = 0; o < nown; ++o) {
            if (own_kind[o] == 0) {
                const int ei = own_i[o];
                const int ej = own_j[o];
                for (int k = 0; k < rows; ++k) {
                    acc[o] += (double)(sg[k * p + ei] * sg[k * p + ej]);
                }
            } else if (own_kind[o] == 1) {
                const int which = own[o] - tri;
                for (int k = 0; k < rows; ++k) {
                    acc[o] += (double)(sg[k * p + which] * sr[k]);
                }
            } else {
                for (int k = 0; k < rows; ++k) acc[o] += sl[k];
            }
        }
        __syncthreads();
    }
    for (int o = 0; o < nown; ++o) {
        if (own_kind[o] == 0) {
            jtj[own[o]] = acc[o];
        } else if (own_kind[o] == 1) {
            jtr[own[o] - tri] = acc[o];
        } else {
            *ploss = acc[o];
        }
    }
    __syncthreads();
}

/* Value-only objective sweep of `c` for the trial step. Fixed-order f64
 * reduction of `Σ w ρ(r)`. */
template <int CHUNK>
__device__ double ft_loss(const unsigned int *nd,
                          int m,
                          const float *__restrict__ cols,
                          const float *__restrict__ y,
                          const double *__restrict__ w,
                          int n,
                          int metric,
                          const double *c,
                          double *red)
{
    const int tid = threadIdx.x;
    double local = 0.0;
    for (int row = tid; row < n; row += CHUNK) {
        const double wr = w ? w[row] : 1.0;
        if (wr <= 0.0) continue; /* a masked row never influences the fit */
        float vt[FT_MAX_NODES];
        ft_forward(nd, m, cols, n, row, c, vt);
        const float pred = vt[m - 1];
        const float r = pred - y[row];
        if (!isfinite(pred) || fabsf(pred) > 1e30f || !isfinite(r)) {
            local += 1e30;
        } else if (metric == RC_LOSS_LOG_SQUASHED) {
            local += wr * (double)rc_row_loss(r, metric);
        } else {
            /* Keep the historical f64 square: the f32 product would round the
               trial score differently from the assembly's own statistic. */
            local += wr * (double)r * (double)r;
        }
    }
    red[tid] = local;
    __syncthreads();
    for (int st = CHUNK / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const double total = red[0];
    __syncthreads();
    return total;
}

/* Damped Cholesky solve in f64. Returns 0 on success, 1 on a non-positive
 * pivot or a conditioning gate.
 *
 * `scale_free = 0` (the squared family) keeps the historical gate: an absolute
 * pivot floor of 1e-18 and a Cholesky diagonal ratio `hi/lo <= 1e4`.
 *
 * #229: robust IRLS weights make `A` badly *scaled* without being
 * ill-conditioned, so for `log_squashed` the gate is the scale-free
 * `ln det Ã = 2Σ ln L_ii − Σ ln A_ii`, `Ã = D^{-1/2} A D^{-1/2}`, which is
 * invariant to units and to a global weight scale. `ln det Ã < ln 1e-12` means
 * the data no longer identifies the parameters, which is exactly when the
 * exact CPU arm must take over. */
/* ln(1e-12), the smallest identified fraction of parameter space. */
#define FT_LN_DET_FLOOR (-27.631021115928547)

template <int MAXC>
__device__ int ft_solve(const double *jtj,
                        const double *jtr,
                        double lambda,
                        int p,
                        double *delta,
                        double *L,
                        int scale_free)
{
    /* Conditioning gate on the UNDAMPED J^T J: damping deliberately hides a
       rank-deficient direction (that is what it is for), so the gate must not
       see it. */
    double sum_ln_L = 0.0;
    double sum_ln_A = 0.0;
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = jtj[i * (i + 1) / 2 + j];
            if (scale_free && i == j) {
                /* The absolute 1e-12 floor is meaningless once the weights span
                   decades; only a genuinely non-positive pivot fails here. */
                if (!(jtj[i * (i + 1) / 2 + i] > 0.0) ||
                    !isfinite(jtj[i * (i + 1) / 2 + i])) return 1;
                sum_ln_A += log(jtj[i * (i + 1) / 2 + i]);
            } else if (i == j) {
                s += 1e-12;
            }
            for (int k = 0; k < j; ++k) s -= L[i * p + k] * L[j * p + k];
            if (i == j) {
                if (scale_free ? (!(s > 0.0) || !isfinite(s)) : (!(s > 1e-18) || !isfinite(s)))
                    return 1;
                L[i * p + i] = sqrt(s);
                if (scale_free) sum_ln_L += 0.5 * log(s);
            } else {
                L[i * p + j] = s / L[j * p + j];
            }
        }
    }
    if (scale_free) {
        const double ln_det = 2.0 * sum_ln_L - sum_ln_A;
        if (!isfinite(ln_det) || ln_det < FT_LN_DET_FLOOR) return 1;
    } else {
        double lo = L[0], hi = L[0];
        for (int i = 1; i < p; ++i) {
            lo = fmin(lo, L[i * p + i]);
            hi = fmax(hi, L[i * p + i]);
        }
        if (!(lo > 0.0) || !isfinite(hi) || hi / lo > 1e4) return 1;
    }
    /* Damped factorisation for the step itself. */
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = jtj[i * (i + 1) / 2 + j];
            if (i == j) s = s * (1.0 + lambda) + 1e-12;
            for (int k = 0; k < j; ++k) s -= L[i * p + k] * L[j * p + k];
            if (i == j) {
                if (!(s > 1e-18) || !isfinite(s)) return 1;
                L[i * p + i] = sqrt(s);
            } else {
                L[i * p + j] = s / L[j * p + j];
            }
        }
    }
    double rhs[MAXC];
    for (int i = 0; i < p; ++i) rhs[i] = -jtr[i];
    for (int i = 0; i < p; ++i) {
        double s = rhs[i];
        for (int k = 0; k < i; ++k) s -= L[i * p + k] * rhs[k];
        rhs[i] = s / L[i * p + i];
    }
    for (int i = p - 1; i >= 0; --i) {
        double s = rhs[i];
        for (int k = i + 1; k < p; ++k) s -= L[k * p + i] * delta[k];
        delta[i] = s / L[i * p + i];
    }
    return 0;
}

__global__ void cast_f32_to_f64_kernel(const float *__restrict__ src,
                                      double *__restrict__ dst,
                                      long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}

/* --- #224: exact correlation fit objective -------------------------------
 *
 * The kernel minimises exactly 1-|rho| (RC_LOSS_CORRELATION) or 1-rho
 * (RC_LOSS_CORRELATION_SIGNED), never SSE. Moments are accumulated in f64 from
 * the f32 forward/adjoint values with the same resident row weights the other
 * objectives use. One owner thread accumulates each statistic in ascending row
 * order, so the reduction is deterministic and needs no atomics.
 *
 * Statistic layout: triangle C_jk, sum w*g_j, sum w*g_j*f, sum w*g_j*y, then
 * six globals (W, sum w*f, sum w*f^2, sum w*f*y, sum w*y, sum w*y^2).
 */
#define FT_CORR_STATS(MAXC) (((MAXC) * ((MAXC) + 1) / 2) + 4 * (MAXC) + 6)
#define FT_CORR_OWNED(MAXC, CHUNK) \
    (((FT_CORR_STATS(MAXC)) + (CHUNK) - 1) / (CHUNK))

/* Decode correlation statistic `s`: 0 = C_ij, 1 = sum w*g_i,
 * 2 = sum w*g_i*f, 3 = sum w*g_i*y, 4 = global moment `g`. */
__device__ __forceinline__ int ft_corr_kind(int s, int tri, int p, int *i, int *j, int *g)
{
    if (s < tri) {
        int k = 0;
        for (int a = 0; a < p; ++a) {
            for (int b = 0; b <= a; ++b, ++k) {
                if (k == s) {
                    *i = a;
                    *j = b;
                    return 0;
                }
            }
        }
        *i = 0;
        *j = 0;
        return 0;
    }
    if (s < tri + p) {
        *i = s - tri;
        return 1;
    }
    if (s < tri + 2 * p) {
        *i = s - tri - p;
        return 2;
    }
    if (s < tri + 3 * p) {
        *i = s - tri - 2 * p;
        return 3;
    }
    *g = s - tri - 3 * p;
    return 4;
}

template <int MAXC, int CHUNK>
__device__ void ft_corr_assemble(const unsigned int *nd,
                                 int m,
                                 const float *__restrict__ cols,
                                 const float *__restrict__ y,
                                 const double *__restrict__ w,
                                 int n,
                                 int p,
                                 const double *c,
                                 float *sg,
                                 float *sr,
                                 double *jtj,
                                 double *jtr,
                                 double *mom,
                                 double *psvar,
                                 double *psyy,
                                 double *pscore,
                                 double *prho,
                                 int *bad,
                                 int metric)
{
    const int tid = threadIdx.x;
    const int tri = p * (p + 1) / 2;
    const int total = FT_CORR_STATS(p);
    int own[FT_CORR_OWNED(MAXC, CHUNK)];
    int own_kind[FT_CORR_OWNED(MAXC, CHUNK)];
    int own_i[FT_CORR_OWNED(MAXC, CHUNK)];
    int own_j[FT_CORR_OWNED(MAXC, CHUNK)];
    int own_g[FT_CORR_OWNED(MAXC, CHUNK)];
    double acc[FT_CORR_OWNED(MAXC, CHUNK)];
    int nown = 0;
    for (int s = tid; s < total; s += CHUNK) {
        own[nown] = s;
        own_kind[nown] = ft_corr_kind(s, tri, p, &own_i[nown], &own_j[nown], &own_g[nown]);
        acc[nown] = 0.0;
        ++nown;
    }
    for (int base = 0; base < n; base += CHUNK) {
        const int rows = min(CHUNK, n - base);
        const int row = base + tid;
        const double wr = (row < n && w) ? w[row] : 1.0;
        if (row < n && wr > 0.0) {
            float vt[FT_MAX_NODES];
            float adj[FT_MAX_NODES];
            float g[MAXC];
            ft_forward(nd, m, cols, n, row, c, vt);
            const float pred = vt[m - 1];
            const float t = y[row];
            bool ok = isfinite(pred) && fabsf(pred) <= 1e30f && isfinite(t);
            ft_backward<MAXC>(nd, m, vt, adj, g, p);
            for (int j = 0; j < p; ++j) {
                if (!isfinite(g[j])) ok = false;
                sg[tid * p + j] = ok ? g[j] : 0.0f;
            }
            sr[tid] = ok ? pred : 0.0f;
            if (!ok) *bad = 1;
        } else {
            for (int j = 0; j < p; ++j) sg[tid * p + j] = 0.0f;
            sr[tid] = 0.0f;
        }
        __syncthreads();
        for (int o = 0; o < nown; ++o) {
            const int kind = own_kind[o];
            if (kind == 0) {
                const int ei = own_i[o];
                const int ej = own_j[o];
                for (int k = 0; k < rows; ++k) {
                    const double wk = (w) ? w[base + k] : 1.0;
                    acc[o] += wk * (double)(sg[k * p + ei] * sg[k * p + ej]);
                }
            } else if (kind == 1) {
                const int ei = own_i[o];
                for (int k = 0; k < rows; ++k) {
                    const double wk = (w) ? w[base + k] : 1.0;
                    acc[o] += wk * (double)sg[k * p + ei];
                }
            } else if (kind == 2) {
                const int ei = own_i[o];
                for (int k = 0; k < rows; ++k) {
                    const double wk = (w) ? w[base + k] : 1.0;
                    acc[o] += wk * (double)sg[k * p + ei] * (double)sr[k];
                }
            } else if (kind == 3) {
                const int ei = own_i[o];
                for (int k = 0; k < rows; ++k) {
                    const double wk = (w) ? w[base + k] : 1.0;
                    acc[o] += wk * (double)sg[k * p + ei] * (double)y[base + k];
                }
            } else {
                const int gi = own_g[o];
                for (int k = 0; k < rows; ++k) {
                    const double wk = (w) ? w[base + k] : 1.0;
                    const double f = (double)sr[k];
                    const double t = (double)y[base + k];
                    if (gi == 0) acc[o] += wk;
                    else if (gi == 1) acc[o] += wk * f;
                    else if (gi == 2) acc[o] += wk * f * f;
                    else if (gi == 3) acc[o] += wk * f * t;
                    else if (gi == 4) acc[o] += wk * t;
                    else acc[o] += wk * t * t;
                }
            }
        }
        __syncthreads();
    }
    for (int o = 0; o < nown; ++o) {
        const int kind = own_kind[o];
        if (kind == 0) {
            jtj[own[o]] = acc[o];
        } else if (kind < 4) {
            mom[(kind - 1) * p + own_i[o]] = acc[o];
        } else {
            mom[3 * p + own_g[o]] = acc[o];
        }
    }
    __syncthreads();

    if (tid == 0) {
        const double W = mom[3 * p + 0];
        const double sum_f = mom[3 * p + 1];
        const double sum_f2 = mom[3 * p + 2];
        const double sum_ft = mom[3 * p + 3];
        const double sum_t = mom[3 * p + 4];
        const double sum_t2 = mom[3 * p + 5];
        double sff = 0.0, syy = 0.0, rho = 0.0;
        if (W > 0.0) {
            const double mf = sum_f / W;
            const double mt = sum_t / W;
            sff = sum_f2 - W * mf * mf;
            syy = sum_t2 - W * mt * mt;
            const double sft = sum_ft - W * mf * mt;
            if (sff > 0.0 && syy > 0.0) rho = sft / sqrt(sff * syy);
            /* A_j and B_j overwrite sum g and sum g*y; gbar goes into sum g*f. */
            for (int j = 0; j < p; ++j) {
                const double gbar = mom[j] / W;
                const double a = mom[p + j] - W * gbar * mf;
                const double b = mom[2 * p + j] - W * gbar * mt;
                mom[j] = a;
                mom[2 * p + j] = b;
                mom[p + j] = gbar;
            }
            if (sff > 0.0 && syy > 0.0) {
                const double root = sqrt(sff * syy);
                for (int j = 0; j < p; ++j) {
                    jtr[j] = mom[2 * p + j] / root - rho * mom[j] / sff;
                }
                int k = 0;
                for (int i = 0; i < p; ++i) {
                    for (int j = 0; j <= i; ++j, ++k) {
                        const double centred = jtj[k] - W * mom[p + i] * mom[p + j];
                        jtj[k] = (centred - mom[i] * mom[j] / sff) / sff;
                    }
                }
            } else {
                for (int j = 0; j < p; ++j) jtr[j] = 0.0;
                int k = 0;
                for (int i = 0; i < p; ++i) {
                    for (int j = 0; j <= i; ++j, ++k) jtj[k] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
        *psvar = sff;
        *psyy = syy;
        *prho = rho;
        *pscore = (metric == RC_LOSS_CORRELATION_SIGNED) ? (1.0 - rho) : (1.0 - fabs(rho));
        if (!isfinite(*pscore) || !isfinite(sff) || !isfinite(syy)) *bad = 1;
    }
    __syncthreads();
}

/* Value-only weighted correlation score at `c`: one row pass, f64 reduction,
 * the same chordal objective. A non-finite row returns a large finite value so
 * the LM rejects the trial; the host gate then labels the candidate. */
template <int CHUNK>
__device__ double ft_corr_score(const unsigned int *nd,
                                int m,
                                const float *__restrict__ cols,
                                const float *__restrict__ y,
                                const double *__restrict__ w,
                                int n,
                                const double *c,
                                double *red,
                                int metric)
{
    const int tid = threadIdx.x;
    double sw = 0.0, sf = 0.0, sf2 = 0.0, sft = 0.0, st = 0.0, st2 = 0.0;
    for (int row = tid; row < n; row += CHUNK) {
        const double wr = (w) ? w[row] : 1.0;
        if (wr <= 0.0) continue;
        float vt[FT_MAX_NODES];
        ft_forward(nd, m, cols, n, row, c, vt);
        const double f = (double)vt[m - 1];
        const double t = (double)y[row];
        if (!isfinite(f) || fabs(f) > 1e30 || !isfinite(t)) {
            sw = 1.0;
            sf = 1e30;
            sf2 = 1e60;
            sft = 0.0;
            st = 0.0;
            st2 = 0.0;
            break;
        }
        sw += wr;
        sf += wr * f;
        sf2 += wr * f * f;
        sft += wr * f * t;
        st += wr * t;
        st2 += wr * t * t;
    }
    double vals[6] = {sw, sf, sf2, sft, st, st2};
    for (int q = 0; q < 6; ++q) {
        red[tid] = vals[q];
        __syncthreads();
        for (int s = CHUNK / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        vals[q] = red[0];
        __syncthreads();
    }
    const double W = vals[0];
    if (!(W > 0.0)) return 1e30;
    const double mf = vals[1] / W;
    const double mt = vals[4] / W;
    const double sff = vals[2] - W * mf * mf;
    const double syy = vals[5] - W * mt * mt;
    const double sft_c = vals[3] - W * mf * mt;
    if (!(sff > 0.0) || !(syy > 0.0)) return 1.0;
    const double rho = sft_c / sqrt(sff * syy);
    return (metric == RC_LOSS_CORRELATION_SIGNED) ? (1.0 - rho) : (1.0 - fabs(rho));
}

/* Mask the affine gauge slots and turn the rho gradient into the objective
 * gradient: L = 1 - sigma*rho, so ft_solve with jtr = dL gives
 * delta = -H^{-1} dL. A gauge slot gets a zero right-hand side and an identity
 * row/column, so its step is exactly zero. */
__device__ void ft_corr_mask_gauge(double *jtj, double *jtr, int p, double rho, int metric,
                                   const int *gauge)
{
    const double sigma = (metric == RC_LOSS_CORRELATION_SIGNED)
                             ? 1.0
                             : (rho >= 0.0 ? 1.0 : -1.0);
    for (int j = 0; j < p; ++j) jtr[j] = -sigma * jtr[j];
    for (int i = 0; i < p; ++i) {
        const bool gi = (gauge[0] == i) || (gauge[1] == i);
        for (int j = 0; j <= i; ++j) {
            const bool gj = (gauge[0] == j) || (gauge[1] == j);
            if (gi || gj) {
                jtj[i * (i + 1) / 2 + j] = (i == j) ? 1.0 : 0.0;
            }
        }
        if (gi) jtr[i] = 0.0;
    }
}

template <int MAXC, int CHUNK, int PCHUNK>
__global__ void fit_kernel(const float *__restrict__ cols,
                           const float *__restrict__ y,
                           const double *__restrict__ cols64,
                           const double *__restrict__ y64,
                           const double *__restrict__ w64,
                           int n,
                           int v,
                           const unsigned int *__restrict__ prog,
                           const unsigned int *__restrict__ off,
                           const int *__restrict__ pcount,
                           int kmax,
                           const double *__restrict__ c0,
                           const int *__restrict__ gauge,
                           int iters,
                           int metric,
                           double *__restrict__ outc,
                           double *__restrict__ outsse,
                           int *__restrict__ outstatus)
{
    const int cand = blockIdx.x;
    const int tid = threadIdx.x;
    const int p = pcount[cand];
    const int words = (int)(off[cand + 1] - off[cand]);
    const int m = words / 4;
    const unsigned int *nd = prog + off[cand];
    /* #229: the scale-free determinant gate replaces the ratio gate for the
       robust arm, whose normal equations are badly scaled by construction. */
    const int scale_free = (metric == RC_LOSS_LOG_SQUASHED);

    extern __shared__ char ft_smem_raw[];
    FtSmem<MAXC, CHUNK, PCHUNK> *S =
        reinterpret_cast<FtSmem<MAXC, CHUNK, PCHUNK> *>(ft_smem_raw);

    if (tid < p) S->sc[tid] = c0[(size_t)cand * (size_t)kmax + (size_t)tid];
    if (tid == 0) {
        S->sbad = 0;
        S->slambda = 1e-3;
        S->sbest = 0.0;
    }
    __syncthreads();

    /* #224: exact correlation LM. The affine gauge pair is renormalised once
       to (sign(a), 0) and then frozen: its step is exactly zero, so the reduced
       curvature has no null direction and the gate below measures only shape
       collinearity or saturation. */
    if (metric == RC_LOSS_CORRELATION || metric == RC_LOSS_CORRELATION_SIGNED) {
        const int *g = gauge + 2 * cand;
        if (tid == 0) {
            const int gs = g[0];
            const int go = g[1];
            if (gs >= 0 && gs < p && isfinite(S->sc[gs]) && S->sc[gs] != 0.0) {
                S->sc[gs] = S->sc[gs] > 0.0 ? 1.0 : -1.0;
            }
            if (go >= 0 && go < p) S->sc[go] = 0.0;
        }
        __syncthreads();
        int status = 0;
        double best = 1.0;
        if (iters > 0) {
            ft_corr_assemble<MAXC, CHUNK>(nd, m, cols, y, w64, n, p, S->sc, S->sg, S->sr,
                                          S->sjtj, S->sjtr, S->scorr, &S->svar, &S->syy,
                                          &S->sbest, &S->srho, &S->sbad, metric);
            if (S->sbad) {
                status = 2;
            } else if (!(S->svar > 0.0) || !(S->syy > 0.0)) {
                best = 1.0;
            } else if (S->svar < 1e-24 * S->syy) {
                status = 1;
            } else {
                best = S->sbest;
                double lambda = 1e-3;
                int assembled = 1;
                for (int it = 0; it < iters && status == 0; ++it) {
                    if (!assembled) {
                        ft_corr_assemble<MAXC, CHUNK>(nd, m, cols, y, w64, n, p, S->sc, S->sg,
                                                      S->sr, S->sjtj, S->sjtr, S->scorr,
                                                      &S->svar, &S->syy, &S->sbest, &S->srho,
                                                      &S->sbad, metric);
                        assembled = 1;
                        if (S->sbad) {
                            status = 2;
                            break;
                        }
                        if (S->svar < 1e-24 * S->syy) {
                            status = 1;
                            break;
                        }
                    }
                    ft_corr_mask_gauge(S->sjtj, S->sjtr, p, S->srho, metric, g);
                    if (tid == 0) {
                        S->ssolved = (ft_solve<MAXC>(S->sjtj, S->sjtr, lambda, p, S->sdelta,
                                                     S->sL, 0) == 0);
                    }
                    __syncthreads();
                    if (!S->ssolved) {
                        status = 1;
                        break;
                    }
                    if (tid < p) {
                        double t = S->sc[tid] + S->sdelta[tid];
                        S->strial[tid] = fmin(1e6, fmax(-1e6, t));
                    }
                    __syncthreads();
                    double trial = ft_corr_score<CHUNK>(nd, m, cols, y, w64, n, S->strial,
                                                        S->sred, metric);
                    if (trial < best) {
                        if (tid < p) S->sc[tid] = S->strial[tid];
                        best = trial;
                        lambda = fmax(lambda / 8.0, 1e-12);
                        assembled = 0;
                        if (best <= 0.0) break;
                    } else {
                        lambda *= 6.0;
                        if (lambda > 1e9) break;
                    }
                    __syncthreads();
                }
                for (int it = 0; it < 4 && status == 0 && best > 0.0; ++it) {
                    ft_corr_assemble<MAXC, CHUNK>(nd, m, cols, y, w64, n, p, S->sc, S->sg, S->sr,
                                                  S->sjtj, S->sjtr, S->scorr, &S->svar, &S->syy,
                                                  &S->sbest, &S->srho, &S->sbad, metric);
                    if (S->sbad) {
                        status = 2;
                        break;
                    }
                    ft_corr_mask_gauge(S->sjtj, S->sjtr, p, S->srho, metric, g);
                    if (tid == 0) {
                        S->ssolved = (ft_solve<MAXC>(S->sjtj, S->sjtr, 0.0, p, S->sdelta,
                                                     S->sL, 0) == 0);
                    }
                    __syncthreads();
                    if (!S->ssolved) break;
                    if (tid < p) {
                        double t = S->sc[tid] + S->sdelta[tid];
                        S->strial[tid] = fmin(1e6, fmax(-1e6, t));
                    }
                    __syncthreads();
                    double trial = ft_corr_score<CHUNK>(nd, m, cols, y, w64, n, S->strial,
                                                        S->sred, metric);
                    if (trial < best) {
                        if (tid < p) S->sc[tid] = S->strial[tid];
                        best = trial;
                    } else {
                        break;
                    }
                    __syncthreads();
                }
            }
        } else {
            best = ft_corr_score<CHUNK>(nd, m, cols, y, w64, n, S->sc, S->sred, metric);
        }
        __syncthreads();
        if (tid == 0) {
            for (int i = 0; i < p; ++i) outc[(size_t)cand * (size_t)kmax + (size_t)i] = S->sc[i];
            outsse[cand] = best;
            outstatus[cand] = status;
        }
        return;
    }

    int status = 0;
    int assembled = 0;
    double best = 0.0;

    if (iters > 0) {
        /* First assembly also supplies the baseline objective. */
        ft_assemble<MAXC, CHUNK>(nd, m, cols, y, w64, n, p, metric, S->sc, S->sg, S->sr, S->sdl, S->sjtj, S->sjtr, &S->sbest, &S->sbad);
        assembled = 1;
        best = S->sbest;
        if (S->sbad) status = 2;
        for (int it = 0; it < iters && status == 0; ++it) {
            if (!assembled) {
                ft_assemble<MAXC, CHUNK>(nd, m, cols, y, w64, n, p, metric, S->sc, S->sg, S->sr, S->sdl, S->sjtj, S->sjtr, &S->sbest, &S->sbad);
                assembled = 1;
                if (S->sbad) {
                    status = 2;
                    break;
                }
            }
            if (tid == 0) {
                S->ssolved = (ft_solve<MAXC>(S->sjtj, S->sjtr, S->slambda, p, S->sdelta, S->sL, scale_free) == 0);
            }
            __syncthreads();
            if (!S->ssolved) {
                status = 1;
                break;
            }
            if (tid < p) {
                double t = S->sc[tid] + S->sdelta[tid];
                S->strial[tid] = fmin(1e6, fmax(-1e6, t));
            }
            __syncthreads();
            double tsse = ft_loss<CHUNK>(nd, m, cols, y, w64, n, metric, S->strial, S->sred);
            if (tsse < best) {
                if (tid < p) S->sc[tid] = S->strial[tid];
                best = tsse;
                if (tid == 0) S->slambda = fmax(S->slambda / 8.0, 1e-12);
                assembled = 0;
                if (best == 0.0) break;
            } else {
                if (tid == 0) S->slambda *= 6.0;
                if (S->slambda > 1e9) break;
            }
            __syncthreads();
        }
        /* Re-baseline the acceptance objective in f64: `best` so far came from
           the f32 assembly, and comparing an f64 trial against it would reject
           every polish step on a systematic rounding difference alone. */
        if (status == 0) {
            const double exact = ft_loss_d<PCHUNK>(nd, m, cols64, y64, w64, n, metric, S->sc, S->sred);
            if (isfinite(exact)) best = exact;
        }
        /* Undamped Gauss-Newton polish, exactly like the CPU path. */
        for (int it = 0; it < 4 && status == 0 && best != 0.0; ++it) {
            ft_assemble_d<MAXC, PCHUNK>(nd, m, cols64, y64, w64, n, p, metric, S->sc, S->sdg, S->sdr, S->sdl64, S->sjtj, S->sjtr, &S->sbest, &S->sbad);
            assembled = 1;
            if (S->sbad) {
                status = 2;
                break;
            }
            if (tid == 0) S->ssolved = (ft_solve<MAXC>(S->sjtj, S->sjtr, 0.0, p, S->sdelta, S->sL, scale_free) == 0);
            __syncthreads();
            if (!S->ssolved) break;
            if (tid < p) {
                double t = S->sc[tid] + S->sdelta[tid];
                S->strial[tid] = fmin(1e6, fmax(-1e6, t));
            }
            __syncthreads();
            double tsse = ft_loss_d<PCHUNK>(nd, m, cols64, y64, w64, n, metric, S->strial, S->sred);
            if (tsse < best) {
                if (tid < p) S->sc[tid] = S->strial[tid];
                best = tsse;
            } else {
                break;
            }
            __syncthreads();
        }
    }

    (void)v;
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < p; ++i) outc[(size_t)cand * (size_t)kmax + (size_t)i] = S->sc[i];
        outsse[cand] = best;
        outstatus[cand] = status;
    }
}

/* Shape of the instantiation that owns a batch whose largest constant count is
 * `kmax`. Returns 0 and fills the out-parameters, or -1 when `kmax` is past the
 * largest instantiation (the caller then keeps the CPU arm). */
int ft_kernel_shape(int kmax, int *threads, int *smem_bytes, int *max_constants)
{
    if (kmax <= FT_MAX_CONST_SMALL) {
        *threads = FT_CHUNK_SMALL;
        *smem_bytes = (int)sizeof(FtSmem<FT_MAX_CONST_SMALL, FT_CHUNK_SMALL, FT_PCHUNK>);
        *max_constants = FT_MAX_CONST_SMALL;
        return 0;
    }
    if (kmax <= FT_MAX_CONST_LARGE) {
        *threads = FT_CHUNK_LARGE;
        *smem_bytes = (int)sizeof(FtSmem<FT_MAX_CONST_LARGE, FT_CHUNK_LARGE, FT_PCHUNK>);
        *max_constants = FT_MAX_CONST_LARGE;
        return 0;
    }
    return -1;
}

/* Launch one instantiation of the fit kernel. The shared block is dynamic, so
 * the p=20 instantiation is not limited by the 48 KB static ceiling; the
 * opt-in limit is raised once per instantiation. Returns 0 on success. */
template <int MAXC, int CHUNK, int PCHUNK>
static int ft_launch(RcCtx *c,
                     int nprog,
                     int kmax,
                     const unsigned int *d_prog,
                     const unsigned int *d_off,
                     const int *d_p,
                     const double *d_c0,
                     const int *d_gauge,
                     int iters,
                     int metric,
                     double *d_c,
                     double *d_sse,
                     int *d_status)
{
    const int bytes = (int)sizeof(FtSmem<MAXC, CHUNK, PCHUNK>);
    static int opted_in = 0;
    if (!opted_in) {
        if (cudaFuncSetAttribute((const void *)fit_kernel<MAXC, CHUNK, PCHUNK>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        opted_in = 1;
    }
    fit_kernel<MAXC, CHUNK, PCHUNK><<<nprog, CHUNK, bytes>>>(
        c->d_cols, c->d_y, c->d_cols64, c->d_y64, c->d_w64, c->n, c->v, d_prog, d_off, d_p,
        kmax, d_c0, d_gauge, iters, metric, d_c, d_sse, d_status);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* #122: canonical postfix hash, FNV-1a over the words up to and including
   HALT. Mirrors `genome::postfix_hash` exactly. */
__global__ void hash_kernel(const unsigned int *__restrict__ prog,
                            int width,
                            int nprog,
                            unsigned long long *__restrict__ out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nprog) return;
    const unsigned int *slot = prog + (size_t)p * (size_t)width;
    unsigned long long h = 0xcbf29ce484222325ULL;
    for (int q = 0; q < width; ++q) {
        unsigned int w = slot[q];
        h ^= (unsigned long long)w;
        h *= 0x00000100000001b3ULL;
        if (w == OP_HALT) break;
    }
    out[p] = h;
}

/* Allocation strategy.
 *
 * With RCSR_UNIFIED every buffer lives in managed (unified) memory: the host
 * writes directly into it, the device migrates pages on demand, and there is no
 * explicit host<->device copy. That is what lets a GB10 with 128 GB of coherent
 * memory hold a dataset larger than the device's VRAM.
 *
 * Without the macro the dataset is copied once into plain device memory, which
 * is faster on a discrete GPU with its own VRAM (the RTX 3060 default).
 */
#ifdef RCSR_UNIFIED
static int rc_alloc(void **p, size_t bytes)
{
    return (int)cudaMallocManaged(p, bytes, cudaMemAttachGlobal);
}
static void rc_upload(void *dst, const void *src, size_t bytes)
{
    memcpy(dst, src, bytes);
}
static void rc_download(void *dst, const void *src, size_t bytes)
{
    memcpy(dst, src, bytes);
}
#else
static int rc_alloc(void **p, size_t bytes)
{
    return (int)cudaMalloc(p, bytes);
}
static void rc_upload(void *dst, const void *src, size_t bytes)
{
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
static void rc_download(void *dst, const void *src, size_t bytes)
{
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
#endif


/* --- #230: exact rank-correlation stage ----------------------------------
 *
 * `1 - rho` / `1 - |rho|` over average ranks. Twice the centred rank
 *
 *     A_i = 2 r_i - (n + 1)
 *
 * is an integer in [-(n-1), n-1] and `Sigma A_i = 0`, so
 *
 *     rho = Sigma A_i B_i / sqrt(Sigma A_i^2 Sigma B_i^2)
 *
 * with `B` the same quantity for the target. The three sums are exact and
 * order-independent, which is why this stage needs no atomics and gives the
 * same bits on one island and on four.
 *
 * Pipeline per call, over `ns` segments of `n` rows:
 *   1. `k_rank_predict` writes an order-preserving u64 key per (segment, row);
 *   2. `cub::DeviceSegmentedRadixSort` sorts the keys with the row as payload;
 *   3. `k_rank_flags` marks run starts and gathers the cached target ranks;
 *   4. two CUB scans turn the flags into run ordinals and the gathered ranks
 *      into prefix sums (both exact integer sums);
 *   5. `k_rank_starts` scatters each run's first position;
 *   6. `k_rank_contrib` accumulates, per run, `A * Sigma B` and `A^2 * length`
 *      into a 128-bit pair built from two u64 halves, then reduces the block.
 *
 * Ties are exact float equality: `-0.0` is canonicalised to `+0.0` before
 * keying so the two share a rank exactly like the CPU's `==`, and the order
 * inside a tie run is irrelevant because the whole run gets the average rank.
 */

__device__ __forceinline__ unsigned long long rank_key(double v)
{
    if (v == 0.0) v = 0.0;              /* -0.0 == +0.0 is one tie */
    if (isnan(v)) return 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long bits;
    memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x8000000000000000ULL) ? ~bits : (bits | 0x8000000000000000ULL);
}

/* Two's-complement 128-bit addition of (ah:al) into (hi:lo). */
__device__ __forceinline__ void rank_add128(unsigned long long *hi,
                                            unsigned long long *lo,
                                            unsigned long long ah,
                                            unsigned long long al)
{
    const unsigned long long old = *lo;
    *lo += al;
    *hi += ah + (*lo < old ? 1ULL : 0ULL);
}

/* Exact 64x64 -> 128 product of two signed values, as a u64 pair. */
__device__ __forceinline__ void rank_mul128(long long a, long long b,
                                            unsigned long long *hi,
                                            unsigned long long *lo)
{
    const unsigned long long ua =
        (a < 0) ? (unsigned long long)(-(a + 1)) + 1ULL : (unsigned long long)a;
    const unsigned long long ub =
        (b < 0) ? (unsigned long long)(-(b + 1)) + 1ULL : (unsigned long long)b;
    *lo = ua * ub;
    *hi = __umul64hi(ua, ub);
    if ((a < 0) != (b < 0)) {
        *lo = ~*lo + 1ULL;
        *hi = ~*hi + (*lo == 0 ? 1ULL : 0ULL);
    }
}

__global__ void k_rank_predict(const unsigned int *__restrict__ prog,
                               const unsigned int *__restrict__ off,
                               const double *__restrict__ c0, int kmax,
                               const double *__restrict__ cols, int n, int ns,
                               int ntrials,
                               unsigned long long *__restrict__ keys,
                               unsigned int *__restrict__ rows,
                               int *__restrict__ status)
{
    const long long total = (long long)n * (long long)ns;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        const int seg = (int)(i / n);
        const int row = (int)(i % n);
        /* `off` has one entry per *candidate*; a candidate owns `ntrials`
           consecutive segments (its ladder trials). */
        const int cand = seg / ntrials;
        const unsigned int *nd = prog + off[cand];
        const int m = (int)((off[cand + 1] - off[cand]) / 4u);
        const double *c = c0 + (size_t)seg * (size_t)kmax;
        double v[FT_MAX_NODES];
        ft_forward_d(nd, m, cols, n, row, c, v);
        const double pred = v[m - 1];
        if (!isfinite(pred)) {
            keys[i] = 0xFFFFFFFFFFFFFFFFULL;
            status[seg] = 2; /* non-finite prediction, mapped to `non_finite` */
        } else {
            keys[i] = rank_key(pred);
        }
        rows[i] = (unsigned int)row;
    }
}

/* One key per row, without a program: the target side. */
__global__ void k_rank_keys_from_double(const double *__restrict__ v, long long total,
                                        unsigned long long *__restrict__ keys,
                                        unsigned int *__restrict__ rows)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        keys[i] = rank_key(v[i]);
        rows[i] = (unsigned int)i;
    }
}

__global__ void k_rank_flags(const unsigned long long *__restrict__ keys,
                             const unsigned int *__restrict__ rows,
                             const long long *__restrict__ b, int n, long long total,
                             int *__restrict__ flags, long long *__restrict__ gather)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < total;
         p += stride) {
        const bool start = (p % n == 0) || keys[p] != keys[p - 1];
        flags[p] = start ? 1 : 0;
        gather[p] = (b != 0) ? b[rows[p]] : 0LL;
    }
}

/* Close the exclusive prefix sums: `prefix[total]` is the total of all rows,
 * which a run ending at the last position needs. */
__global__ void k_rank_tail(long long *__restrict__ prefix,
                            const long long *__restrict__ gather, long long total)
{
    if (total > 0) prefix[total] = prefix[total - 1] + gather[total - 1];
}

__global__ void k_rank_starts(const int *__restrict__ flags,
                              const int *__restrict__ ord,
                              long long *__restrict__ starts, long long total)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < total;
         p += stride) {
        if (flags[p]) starts[ord[p] - 1] = p;
    }
}

/* Per-run contribution to the two 128-bit sums, reduced inside one block per
 * segment. No atomics: the block tree is a fixed order, so the result is the
 * same on every run and on every device. */
__global__ void k_rank_contrib(const long long *__restrict__ starts,
                               const long long *__restrict__ prefix,
                               const int *__restrict__ ord, int n, long long total,
                               int nsegs, unsigned long long *__restrict__ out)
{
    __shared__ unsigned long long shi[256];
    __shared__ unsigned long long slo[256];
    const int tid = threadIdx.x;
    const int seg = blockIdx.x;
    const long long first = (long long)seg * n;
    const long long last = first + n - 1;
    const int base = ord[first] - 1;
    const int end = (last < total) ? ord[last] : ord[total - 1];
    const int runs = end - base;
    unsigned long long hi_ab = 0ULL, lo_ab = 0ULL;
    unsigned long long hi_aa = 0ULL, lo_aa = 0ULL;
    for (int r = base + tid; r < end; r += 256) {
        const long long i = starts[r];
        const long long j = (r + 1 < end) ? starts[r + 1] : (first + n);
        const long long a = (i - first) + (j - first) - (long long)n;
        const long long bsum = prefix[j] - prefix[i];
        unsigned long long h, l;
        rank_mul128(a, bsum, &h, &l);
        rank_add128(&hi_ab, &lo_ab, h, l);
        rank_mul128(a * a, j - i, &h, &l);
        rank_add128(&hi_aa, &lo_aa, h, l);
    }
    shi[tid] = hi_ab;
    slo[tid] = lo_ab;
    __syncthreads();
    /* Reduce the pair (hi, lo) with a carry-aware add. */
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            rank_add128(&shi[tid], &slo[tid], shi[tid + st], slo[tid + st]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[(size_t)seg * 4 + 0] = shi[0];
        out[(size_t)seg * 4 + 1] = slo[0];
    }
    __syncthreads();
    shi[tid] = hi_aa;
    slo[tid] = lo_aa;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            rank_add128(&shi[tid], &slo[tid], shi[tid + st], slo[tid + st]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[(size_t)seg * 4 + 2] = shi[0];
        out[(size_t)seg * 4 + 3] = slo[0];
        (void)runs;
    }
}

/* Write `A` (the centred doubled rank of a run) back to every row of the run. */
__global__ void k_rank_fill_b(const unsigned int *__restrict__ rows_sorted,
                              const long long *__restrict__ starts, int nruns, int n,
                              long long *__restrict__ b)
{
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < nruns;
         r += gridDim.x * blockDim.x) {
        const int i = (int)starts[r];
        const int j = (r + 1 < nruns) ? (int)starts[r + 1] : n;
        const long long a = (long long)(i + j) - (long long)n;
        for (int q = i; q < j; ++q) b[rows_sorted[q]] = a;
    }
}

/* `Sigma B^2` in a 128-bit pair. Used once per call and by the tests. */
__global__ void k_rank_bb(const long long *__restrict__ b, int n,
                          unsigned long long *__restrict__ out)
{
    __shared__ unsigned long long shi[256];
    __shared__ unsigned long long slo[256];
    const int tid = threadIdx.x;
    unsigned long long hi = 0ULL, lo = 0ULL;
    for (int r = tid; r < n; r += 256) {
        unsigned long long h, l;
        rank_mul128(b[r], b[r], &h, &l);
        rank_add128(&hi, &lo, h, l);
    }
    shi[tid] = hi;
    slo[tid] = lo;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) rank_add128(&shi[tid], &slo[tid], shi[tid + st], slo[tid + st]);
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = shi[0];
        out[1] = slo[0];
    }
}

extern "C" {

void *rc_create(const float *cols, int n, int v, const float *y)
{
    RcCtx *c = (RcCtx *)calloc(1, sizeof(RcCtx));
    if (!c) return 0;
    c->n = n;
    c->v = v;
    c->nprog = 0;
    c->out_cap = 0;
    size_t col_bytes = sizeof(float) * (size_t)n * (size_t)v;
    size_t y_bytes = sizeof(float) * (size_t)n;
    if (rc_alloc((void **)&c->d_cols, col_bytes) != 0 ||
        rc_alloc((void **)&c->d_y, y_bytes) != 0) {
        if (c->d_cols) cudaFree(c->d_cols);
        if (c->d_y) cudaFree(c->d_y);
        free(c);
        return 0;
    }
    rc_upload(c->d_cols, cols, col_bytes);
    rc_upload(c->d_y, y, y_bytes);
    cudaEventCreate(&c->t0);
    cudaEventCreate(&c->t1);
    return (void *)c;
}

void rc_destroy(void *ctx)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return;
    cudaFree(c->d_cols);
    cudaFree(c->d_y);
    if (c->d_cols64) cudaFree(c->d_cols64);
    if (c->d_y64) cudaFree(c->d_y64);
    if (c->d_w64) cudaFree(c->d_w64);
    if (c->d_prog) cudaFree(c->d_prog);
    if (c->d_off) cudaFree(c->d_off);
    if (c->d_out) cudaFree(c->d_out);
    if (c->d_pool[0]) cudaFree(c->d_pool[0]);
    if (c->d_pool[1]) cudaFree(c->d_pool[1]);
    if (c->d_hash) cudaFree(c->d_hash);
    if (c->d_rank_b) cudaFree(c->d_rank_b);
    cudaEventDestroy(c->t0);
    cudaEventDestroy(c->t1);
    free(c);
}

int rc_set_programs(void *ctx, const unsigned int *prog, const unsigned int *off, int nprog)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return -1;
    RcStats *st = &c->stats;
    double t = rc_now_ms();
    if (c->d_prog) { cudaFree(c->d_prog); c->d_prog = 0; }
    if (c->d_off) { cudaFree(c->d_off); c->d_off = 0; }
    c->nprog = nprog;
    size_t prog_bytes = sizeof(unsigned int) * (size_t)off[nprog];
    rc_alloc((void **)&c->d_prog, prog_bytes);
    rc_alloc((void **)&c->d_off, sizeof(unsigned int) * (size_t)(nprog + 1));
    if (!c->d_prog || !c->d_off) {
        st->alloc_ms += rc_now_ms() - t;
        st->failed = 1;
        st->wall_ms = rc_now_ms() - c->batch_start_ms;
        return -1;
    }
    if (c->out_cap < nprog) {
        if (c->d_out) cudaFree(c->d_out);
        if (rc_alloc((void **)&c->d_out, sizeof(float) * (size_t)nprog) != 0) {
            c->d_out = 0;
            c->out_cap = 0;
            st->alloc_ms += rc_now_ms() - t;
            st->failed = 1;
            st->wall_ms = rc_now_ms() - c->batch_start_ms;
            return -1;
        }
        c->out_cap = nprog;
    }
    /* cudaMalloc/cudaFree synchronise, so this window includes the real cost
       of (re)allocating the program and output buffers. */
    st->alloc_ms += rc_now_ms() - t;

    t = rc_now_ms();
    rc_upload(c->d_prog, prog, prog_bytes);
    rc_upload(c->d_off, off, sizeof(unsigned int) * (size_t)(nprog + 1));
    st->h2d_ms += rc_now_ms() - t;
    st->h2d_bytes += (unsigned long long)(prog_bytes + sizeof(unsigned int) * (size_t)(nprog + 1));
    st->batch = nprog;
    st->wall_ms = rc_now_ms() - c->batch_start_ms;
    return 0;
}

/* Returns elapsed milliseconds; writes nprog errors into `out`. The context's
   RcStats records the full launch+sync+download path. */
float rc_eval(void *ctx, float *out, int loss)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return -1.0f;
    RcStats *st = &c->stats;
    /* One block per expression; threads spread over rows. Size the block to the
       row count (rounded up to a power of two) so short datasets do not launch
       mostly-idle warps. */
    int threads = RC_MIN_THREADS;
    while (threads < RC_MAX_THREADS && threads < c->n) threads <<= 1;
    double t = rc_now_ms();
    cudaEventRecord(c->t0, 0);
    eval_kernel<<<c->nprog, threads>>>(c->d_cols, c->d_y, c->n, c->v,
                                       c->d_prog, c->d_off, c->d_out, loss);
    cudaEventRecord(c->t1, 0);
    st->launch_ms += rc_now_ms() - t;
    /* Wait for the kernel before reading the result. In the unified build the
       output lives in managed memory, so a plain memcpy (page migration) is the
       download; there is no explicit device-to-host copy. */
    t = rc_now_ms();
    cudaEventSynchronize(c->t1);
    st->sync_ms += rc_now_ms() - t;
    /* A launch or execution error must be visible to the caller: the search
       turns it into a CPU fallback instead of trusting a stale output buffer.
       A negative time is the error marker; the benchmark treats it the same. */
    if (cudaGetLastError() != cudaSuccess) {
        st->failed = 1;
        st->wall_ms = rc_now_ms() - c->batch_start_ms;
        return -1.0f;
    }
    t = rc_now_ms();
    rc_download(out, c->d_out, sizeof(float) * (size_t)c->nprog);
    st->d2h_ms += rc_now_ms() - t;
    st->d2h_bytes += sizeof(float) * (unsigned long long)c->nprog;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c->t0, c->t1);
    st->kernel_ms = (double)ms;
    st->batch = c->nprog;
    st->wall_ms = rc_now_ms() - c->batch_start_ms;
    return ms;
}

/* #131 fixtures: upload a packed batch and write its P x N raw predictions
   (row-major) into `out`. Test-only; the fitness path still uses rc_eval /
   rc_pool_eval. Returns 0 on success, -1 on any CUDA error. */
int rc_predict(void *ctx, const unsigned int *prog, const unsigned int *off,
               int nprog, float *out)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nprog <= 0) return -1;
    if (rc_set_programs(ctx, prog, off, nprog) != 0) return -1;
    size_t total = (size_t)nprog * (size_t)c->n;
    float *d_pred = 0;
    if (cudaMalloc((void **)&d_pred, sizeof(float) * total) != 0) return -1;
    int threads = RC_MIN_THREADS;
    while (threads < RC_MAX_THREADS && threads < c->n) threads <<= 1;
    predict_kernel<<<nprog, threads>>>(c->d_cols, c->n, c->d_prog, c->d_off, d_pred);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_pred);
        return -1;
    }
    if (cudaMemcpy(out, d_pred, sizeof(float) * total, cudaMemcpyDeviceToHost)
        != cudaSuccess) {
        cudaFree(d_pred);
        return -1;
    }
    cudaFree(d_pred);
    return 0;
}

/* --- #122: persistent fixed-capacity program pool ------------------------ */

/* Reserve both arena buffers once. A later, larger batch may grow the arena;
   a generation never allocates or frees it. */
int rc_pool_reserve(void *ctx, int nprog, int width)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nprog <= 0 || width <= 0) return -1;
    double t = rc_now_ms();
    if (c->pool_cap >= nprog && c->pool_width == width) {
        c->stats.alloc_ms += rc_now_ms() - t;
        return 0;
    }
    if (c->d_pool[0]) { cudaFree(c->d_pool[0]); c->d_pool[0] = 0; }
    if (c->d_pool[1]) { cudaFree(c->d_pool[1]); c->d_pool[1] = 0; }
    size_t bytes = sizeof(unsigned int) * (size_t)nprog * (size_t)width;
    if (rc_alloc((void **)&c->d_pool[0], bytes) != 0 ||
        rc_alloc((void **)&c->d_pool[1], bytes) != 0) {
        c->stats.failed = 1;
        c->stats.alloc_ms += rc_now_ms() - t;
        return -1;
    }
    if (c->hash_cap < nprog) {
        if (c->d_hash) { cudaFree(c->d_hash); c->d_hash = 0; }
        if (rc_alloc((void **)&c->d_hash, sizeof(unsigned long long) * (size_t)nprog) != 0) {
            c->d_hash = 0;
            c->hash_cap = 0;
            c->stats.failed = 1;
            c->stats.alloc_ms += rc_now_ms() - t;
            return -1;
        }
        c->hash_cap = nprog;
    }
    c->pool_cap = nprog;
    c->pool_width = width;
    c->stats.alloc_ms += rc_now_ms() - t;
    return 0;
}

/* Upload `nprog` fixed-width programs into buffer `which` (0 or 1). */
int rc_pool_upload(void *ctx, int which, const unsigned int *prog, int nprog)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !c->d_pool[which & 1] || nprog <= 0) return -1;
    size_t bytes = sizeof(unsigned int) * (size_t)nprog * (size_t)c->pool_width;
    double t = rc_now_ms();
    rc_upload(c->d_pool[which & 1], prog, bytes);
    c->stats.h2d_ms += rc_now_ms() - t;
    c->stats.h2d_bytes += (unsigned long long)bytes;
    c->stats.batch = nprog;
    return 0;
}

/* Evaluate the programs already uploaded to `which`; same return contract as
   rc_eval (milliseconds, negative on failure). */
float rc_pool_eval(void *ctx, int which, int nprog, int width, float *out, int loss)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !c->d_pool[which & 1] || nprog <= 0) return -1.0f;
    RcStats *st = &c->stats;
    double t = rc_now_ms();
    if (c->out_cap < nprog) {
        if (c->d_out) cudaFree(c->d_out);
        if (rc_alloc((void **)&c->d_out, sizeof(float) * (size_t)nprog) != 0) {
            c->d_out = 0;
            c->out_cap = 0;
            st->failed = 1;
            st->alloc_ms += rc_now_ms() - t;
            st->wall_ms = rc_now_ms() - c->batch_start_ms;
            return -1.0f;
        }
        c->out_cap = nprog;
    }
    st->alloc_ms += rc_now_ms() - t;
    int threads = RC_MIN_THREADS;
    while (threads < RC_MAX_THREADS && threads < c->n) threads <<= 1;
    t = rc_now_ms();
    cudaEventRecord(c->t0, 0);
    eval_pool_kernel<<<nprog, threads>>>(c->d_cols, c->d_y, c->n, c->v,
                                         c->d_pool[which & 1], width, c->d_out, loss);
    cudaEventRecord(c->t1, 0);
    st->launch_ms += rc_now_ms() - t;
    t = rc_now_ms();
    cudaEventSynchronize(c->t1);
    st->sync_ms += rc_now_ms() - t;
    if (cudaGetLastError() != cudaSuccess) {
        st->failed = 1;
        st->wall_ms = rc_now_ms() - c->batch_start_ms;
        return -1.0f;
    }
    t = rc_now_ms();
    rc_download(out, c->d_out, sizeof(float) * (size_t)nprog);
    st->d2h_ms += rc_now_ms() - t;
    st->d2h_bytes += sizeof(float) * (unsigned long long)nprog;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c->t0, c->t1);
    st->kernel_ms = (double)ms;
    st->batch = nprog;
    st->wall_ms = rc_now_ms() - c->batch_start_ms;
    return ms;
}

/* #118: start a fresh batch ledger. The Rust evaluator calls this once before
   reserve+upload+eval, so allocation, upload, launch, sync and download are all
   in one window and `wall_ms` is their independently measured envelope. */
void rc_stats_begin(void *ctx)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return;
    c->stats.kernel_ms = 0.0;
    c->stats.launch_ms = 0.0;
    c->stats.sync_ms = 0.0;
    c->stats.alloc_ms = 0.0;
    c->stats.h2d_ms = 0.0;
    c->stats.d2h_ms = 0.0;
    c->stats.wall_ms = 0.0;
    c->stats.h2d_bytes = 0;
    c->stats.d2h_bytes = 0;
    c->stats.batch = 0;
    c->stats.failed = 0;
    c->batch_start_ms = rc_now_ms();
}

/* #118: copy the current (last) batch's ledger. Returns 0 on success. */
int rc_last_stats(void *ctx, RcStats *out)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !out) return -1;
    *out = c->stats;
    return 0;
}

/* Canonical postfix hashes of the last uploaded buffer. */
int rc_pool_hash(void *ctx, int which, int nprog, int width, unsigned long long *out)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !c->d_pool[which & 1] || !c->d_hash || nprog <= 0) return -1;
    int threads = 128;
    int blocks = (nprog + threads - 1) / threads;
    hash_kernel<<<blocks, threads>>>(c->d_pool[which & 1], width, nprog, c->d_hash);
    if (cudaGetLastError() != cudaSuccess) return -1;
    rc_download(out, c->d_hash, sizeof(unsigned long long) * (size_t)nprog);
    return 0;
}

/* #169: upload exact f64 training columns for the fit kernel's final polish.
   The evaluator's resident dataset is f32; the batched fit needs the original
   f64 values so its polished constants are not limited by the f32 input
   rounding. Called once per run when the GPU-fit arm is selected.
   #229: `w` is the base `--weight` row vector, or null for an unweighted run.
   It is uploaded once here rather than per batch. */
int rc_set_fit_data(void *ctx, const double *cols, const double *y, const double *w)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !cols || !y) return -1;
    if (c->d_cols64) { cudaFree(c->d_cols64); c->d_cols64 = 0; }
    if (c->d_y64) { cudaFree(c->d_y64); c->d_y64 = 0; }
    if (c->d_w64) { cudaFree(c->d_w64); c->d_w64 = 0; }
    size_t cb = sizeof(double) * (size_t)c->n * (size_t)c->v;
    size_t yb = sizeof(double) * (size_t)c->n;
    if (rc_alloc((void **)&c->d_cols64, cb) != 0) return -1;
    if (rc_alloc((void **)&c->d_y64, yb) != 0) {
        cudaFree(c->d_cols64);
        c->d_cols64 = 0;
        return -1;
    }
    rc_upload(c->d_cols64, cols, cb);
    rc_upload(c->d_y64, y, yb);
    if (w) {
        if (rc_alloc((void **)&c->d_w64, yb) != 0) {
            cudaFree(c->d_cols64);
            c->d_cols64 = 0;
            cudaFree(c->d_y64);
            c->d_y64 = 0;
            return -1;
        }
        rc_upload(c->d_w64, w, yb);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* --- #169: batched LM fit entry point -------------------------------------
 *
 * `prog` is the concatenated 4-word node records; `off` has nprog+1 word
 * offsets; `pcount` has one constant count per candidate; constants start at
 * `c0[(cand*kmax)+i]`. `metric` selects the objective the loop minimises: 0 is
 * the sum of squared errors, 1 the log-squashed sum `Σ w ln(1 + r^2)` with
 * per-row IRLS weights `2/(1 + r^2)` (#229). Returns the kernel time in
 * milliseconds, or a negative value when the launch failed (the caller then
 * keeps the CPU arm).
 */
float rc_fit_batch(void *ctx,
                   const unsigned int *prog,
                   const unsigned int *off,
                   const int *pcount,
                   int nprog,
                   int kmax,
                   const double *c0,
                   const int *gauge,
                   int iters,
                   int metric,
                   double *outc,
                   double *outsse,
                   int *outstatus)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nprog <= 0 || kmax <= 0) return -1.0f;
    if (kmax > FT_MAX_CONST_LARGE) return -1.0f;
    unsigned int *d_prog = 0;
    unsigned int *d_off = 0;
    int *d_p = 0;
    double *d_c0 = 0;
    int *d_gauge = 0;
    double *d_c = 0;
    double *d_sse = 0;
    int *d_status = 0;
    size_t prog_words = (size_t)off[nprog];
    size_t cbytes = sizeof(double) * (size_t)nprog * (size_t)kmax;
    int ok = 1;
    if (rc_alloc((void **)&d_prog, sizeof(unsigned int) * prog_words) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_off, sizeof(unsigned int) * (size_t)(nprog + 1)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_p, sizeof(int) * (size_t)nprog) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_c0, cbytes) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_gauge, sizeof(int) * (size_t)nprog * 2) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_c, cbytes) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_sse, sizeof(double) * (size_t)nprog) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_status, sizeof(int) * (size_t)nprog) != 0) ok = 0;
    if (ok && (!c->d_cols64 || !c->d_y64)) {
        size_t cb = sizeof(double) * (size_t)c->n * (size_t)c->v;
        size_t yb = sizeof(double) * (size_t)c->n;
        if (rc_alloc((void **)&c->d_cols64, cb) != 0) {
            c->d_cols64 = 0;
            ok = 0;
        }
        if (ok && rc_alloc((void **)&c->d_y64, yb) != 0) {
            c->d_y64 = 0;
            ok = 0;
        }
        if (ok) {
            int threads = 256;
            long long total = (long long)c->n * (long long)c->v;
            long long blocks = (total + threads - 1) / threads;
            cast_f32_to_f64_kernel<<<(int)blocks, threads>>>(c->d_cols, c->d_cols64, total);
            long long yblocks = (c->n + threads - 1) / threads;
            cast_f32_to_f64_kernel<<<(int)yblocks, threads>>>(c->d_y, c->d_y64, c->n);
            if (cudaGetLastError() != cudaSuccess) ok = 0;
        }
    }
    if (ok) {
        rc_upload(d_prog, prog, sizeof(unsigned int) * prog_words);
        rc_upload(d_off, off, sizeof(unsigned int) * (size_t)(nprog + 1));
        rc_upload(d_p, pcount, sizeof(int) * (size_t)nprog);
        rc_upload(d_c0, c0, cbytes);
        rc_upload(d_gauge, gauge, sizeof(int) * (size_t)nprog * 2);
        cudaEventRecord(c->t0, 0);
        int launched;
        if (kmax <= FT_MAX_CONST_SMALL) {
            launched = ft_launch<FT_MAX_CONST_SMALL, FT_CHUNK_SMALL, FT_PCHUNK>(
                c, nprog, kmax, d_prog, d_off, d_p, d_c0, d_gauge, iters, metric, d_c, d_sse, d_status);
        } else {
            launched = ft_launch<FT_MAX_CONST_LARGE, FT_CHUNK_LARGE, FT_PCHUNK>(
                c, nprog, kmax, d_prog, d_off, d_p, d_c0, d_gauge, iters, metric, d_c, d_sse, d_status);
        }
        cudaEventRecord(c->t1, 0);
        cudaEventSynchronize(c->t1);
        if (launched != 0 || cudaGetLastError() != cudaSuccess) {
            ok = 0;
        } else {
            rc_download(outc, d_c, cbytes);
            rc_download(outsse, d_sse, sizeof(double) * (size_t)nprog);
            rc_download(outstatus, d_status, sizeof(int) * (size_t)nprog);
        }
    }
    if (d_prog) cudaFree(d_prog);
    if (d_off) cudaFree(d_off);
    if (d_p) cudaFree(d_p);
    if (d_c0) cudaFree(d_c0);
    if (d_gauge) cudaFree(d_gauge);
    if (d_c) cudaFree(d_c);
    if (d_sse) cudaFree(d_sse);
    if (d_status) cudaFree(d_status);
    if (!ok) return -1.0f;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c->t0, c->t1);
    return ms;
}

/* Shape of the fit-kernel instantiation that would run a batch whose largest
 * constant count is `kmax`: block width, dynamic shared bytes and the largest
 * constant count of that instantiation. Returns 0, or -1 when `kmax` is past
 * every instantiation (such a batch stays on the CPU). Diagnostics only. */
int rc_fit_kernel_info(int kmax, int *threads, int *smem_bytes, int *max_constants)
{
    if (!threads || !smem_bytes || !max_constants) return -1;
    return ft_kernel_shape(kmax, threads, smem_bytes, max_constants);
}


/* --- #230: host side of the exact rank stage ------------------------------ */

typedef struct RcRankWork {
    unsigned long long *keys;
    unsigned long long *keys_out;
    unsigned int *rows;
    unsigned int *rows_out;
    int *flags;
    int *ord;
    long long *gather;
    long long *prefix;
    long long *starts;
    void *temp;
    size_t temp_bytes;
} RcRankWork;

static void rc_rank_work_free(RcRankWork *w)
{
    if (w->keys) cudaFree(w->keys);
    if (w->keys_out) cudaFree(w->keys_out);
    if (w->rows) cudaFree(w->rows);
    if (w->rows_out) cudaFree(w->rows_out);
    if (w->flags) cudaFree(w->flags);
    if (w->ord) cudaFree(w->ord);
    if (w->gather) cudaFree(w->gather);
    if (w->prefix) cudaFree(w->prefix);
    if (w->starts) cudaFree(w->starts);
    if (w->temp) cudaFree(w->temp);
    memset(w, 0, sizeof(*w));
}

static int rc_rank_work_alloc(RcRankWork *w, long long total)
{
    memset(w, 0, sizeof(*w));
    int ok = 1;
    if (rc_alloc((void **)&w->keys, sizeof(unsigned long long) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&w->keys_out, sizeof(unsigned long long) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&w->rows, sizeof(unsigned int) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&w->rows_out, sizeof(unsigned int) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&w->flags, sizeof(int) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&w->ord, sizeof(int) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&w->gather, sizeof(long long) * (size_t)total) != 0) ok = 0;
    /* `prefix` has one extra slot: a run's end may be `total`, and the
       contribution kernel reads `prefix[end]`. The scan writes into the base
       allocation (CUB expects aligned output pointers; scanning into
       `prefix + 1` is misaligned and produced wrong sums), and
       `k_rank_tail` fills the last slot afterwards. */
    if (ok && rc_alloc((void **)&w->prefix, sizeof(long long) * (size_t)(total + 1)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&w->starts, sizeof(long long) * (size_t)total) != 0) ok = 0;
    if (!ok) rc_rank_work_free(w);
    return ok ? 0 : -1;
}

/* Sort, mark run starts, prefix-sum the gathered target ranks and scatter the
 * run starts. `d_b` may be NULL for a run-only sweep (the target stage). */
static int rc_rank_runs(RcRankWork *w, const long long *d_b, int n, long long total,
                        int nsegs, int *d_off)
{
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    /* Size the temporary storage for the largest of the three primitives. */
    size_t tb = 0, ts = 0, tsc = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        NULL, tb, w->keys, w->keys_out, w->rows, w->rows_out, total, nsegs, d_off,
        d_off + 1);
    cub::DeviceScan::InclusiveSum(NULL, ts, w->flags, w->ord, (int)total);
    cub::DeviceScan::ExclusiveSum(NULL, tsc, w->gather, w->prefix, (int)total);
    if (ts > tb) tb = ts;
    if (tsc > tb) tb = tsc;
    /* CUB may report a zero-byte requirement for a tiny input; cudaMalloc(0)
       is not portable, so ask for at least one byte. */
    if (tb == 0) tb = 1;
    if (rc_alloc(&w->temp, tb) != 0) return -1;
    w->temp_bytes = tb;
    cub::DeviceSegmentedRadixSort::SortPairs(
        w->temp, tb, w->keys, w->keys_out, w->rows, w->rows_out, total, nsegs, d_off,
        d_off + 1);
    if (cudaGetLastError() != cudaSuccess) return -1;
    k_rank_flags<<<(int)blocks, threads>>>(w->keys_out, w->rows_out, d_b, n, total,
                                           w->flags, w->gather);
    if (cudaGetLastError() != cudaSuccess) return -1;
    cub::DeviceScan::InclusiveSum(w->temp, tb, w->flags, w->ord, (int)total);
    cub::DeviceScan::ExclusiveSum(w->temp, tb, w->gather, w->prefix, (int)total);
    k_rank_tail<<<1, 1>>>(w->prefix, w->gather, total);
    if (cudaGetLastError() != cudaSuccess) return -1;
    k_rank_starts<<<(int)blocks, threads>>>(w->flags, w->ord, w->starts, total);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return 0;
}

/* Ensure the f64 twin of the target exists (the rank stage needs f64 input). */
static int rc_rank_ensure_y64(RcCtx *c)
{
    if (c->d_y64) return 0;
    if (rc_alloc((void **)&c->d_y64, sizeof(double) * (size_t)c->n) != 0) {
        c->d_y64 = 0;
        return -1;
    }
    int threads = 256;
    long long yblocks = (c->n + threads - 1) / threads;
    cast_f32_to_f64_kernel<<<(int)yblocks, threads>>>(c->d_y, c->d_y64, c->n);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(c->d_y64);
        c->d_y64 = 0;
        return -1;
    }
    return 0;
}

/* Compute and cache `B`, twice the centred average rank of the target rows. */
static int rc_rank_prepare(void *ctx)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return -1;
    if (c->rank_ready) return 0;
    if (c->n <= 0) return -1;
    if (rc_rank_ensure_y64(c) != 0) return -1;
    if (rc_alloc((void **)&c->d_rank_b, sizeof(long long) * (size_t)c->n) != 0) {
        c->d_rank_b = 0;
        return -1;
    }
    RcRankWork w;
    long long total = c->n;
    if (rc_rank_work_alloc(&w, total) != 0) return -1;
    int *d_off = 0;
    int host_off[2] = {0, c->n};
    int ok = (rc_alloc((void **)&d_off, sizeof(int) * 2) == 0);
    if (ok) {
        rc_upload(d_off, host_off, sizeof(int) * 2);
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        k_rank_keys_from_double<<<(int)blocks, threads>>>(c->d_y64, total, w.keys, w.rows);
        ok = (cudaGetLastError() == cudaSuccess) && (rc_rank_runs(&w, NULL, c->n, total, 1, d_off) == 0);
    }
    if (ok) {
        int nruns = 0;
        rc_download(&nruns, w.ord + (total - 1), sizeof(int));
        long long nrun = nruns;
        int tb = 256;
        long long rb = (nrun + tb - 1) / tb;
        if (rb > 65535) rb = 65535;
        k_rank_fill_b<<<(int)rb, tb>>>(w.rows_out, w.starts, (int)nrun, c->n, c->d_rank_b);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (d_off) cudaFree(d_off);
    rc_rank_work_free(&w);
    if (!ok) {
        cudaFree(c->d_rank_b);
        c->d_rank_b = 0;
        return -1;
    }
    c->rank_ready = 1;
    return 0;
}

/* Exact rank sums of `nsets * ntrials` prediction vectors against the target.
 *
 * Constants are `c0[(seg * kmax) + slot]` with `seg = set * ntrials + trial`,
 * matching `rc_fit_batch`'s per-candidate layout extended by one trial axis.
 * `out` holds four u64 per segment: `(num_hi, num_lo, aa_hi, aa_lo)`, the two
 * exact 128-bit sums `Sigma A B` and `Sigma A A`. `out_bb` receives the
 * target's `Sigma B B` as `(hi, lo)`. Returns the stage time in ms, or a
 * negative value when nothing may be trusted (the caller then uses the CPU).
 */
float rc_rank_score_batch(void *ctx,
                          const unsigned int *prog,
                          const unsigned int *off,
                          const int *pcount,
                          int nsets,
                          int ntrials,
                          int kmax,
                          const double *c0,
                          unsigned long long *out,
                          int *out_status,
                          unsigned long long *out_bb)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nsets <= 0 || ntrials <= 0 || kmax <= 0) return -1.0f;
    if (c->n <= 1) return -1.0f;
    if (rc_rank_prepare(ctx) != 0) return -1.0f;
    if (rc_rank_ensure_y64(c) != 0) return -1.0f;
    const int nsegs = nsets * ntrials;
    const long long total = (long long)nsegs * (long long)c->n;
    if (total <= 0 || total > 0x7FFFFFFFLL) return -1.0f;
    unsigned int *d_prog = 0, *d_off = 0;
    int *d_p = 0;
    double *d_c0 = 0;
    unsigned long long *d_out = 0;
    int *d_status = 0;
    unsigned long long *d_bb = 0;
    int *d_seg_off = 0;
    RcRankWork w;
    memset(&w, 0, sizeof(w));
    int ok = 1;
    const size_t prog_words = (size_t)off[nsets];
    if (rc_alloc((void **)&d_prog, sizeof(unsigned int) * prog_words) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_off, sizeof(unsigned int) * (size_t)(nsets + 1)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_p, sizeof(int) * (size_t)nsets) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_c0, sizeof(double) * (size_t)nsegs * (size_t)kmax) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_out, sizeof(unsigned long long) * 4 * (size_t)nsegs) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_status, sizeof(int) * (size_t)nsegs) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_bb, sizeof(unsigned long long) * 2) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_seg_off, sizeof(int) * (size_t)(nsegs + 1)) != 0) ok = 0;
    if (ok) {
        rc_upload(d_prog, prog, sizeof(unsigned int) * prog_words);
        rc_upload(d_off, off, sizeof(unsigned int) * (size_t)(nsets + 1));
        rc_upload(d_p, pcount, sizeof(int) * (size_t)nsets);
        rc_upload(d_c0, c0, sizeof(double) * (size_t)nsegs * (size_t)kmax);
        int *host_seg = (int *)malloc(sizeof(int) * (size_t)(nsegs + 1));
        if (!host_seg) {
            ok = 0;
        } else {
            for (int s = 0; s <= nsegs; ++s) host_seg[s] = s * c->n;
            rc_upload(d_seg_off, host_seg, sizeof(int) * (size_t)(nsegs + 1));
            free(host_seg);
        }
        cudaMemset(d_status, 0, sizeof(int) * (size_t)nsegs);
    }
    if (ok && rc_rank_work_alloc(&w, total) != 0) ok = 0;
    float ms = -1.0f;
    if (ok) {
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        cudaEventRecord(c->t0, 0);
        k_rank_predict<<<(int)blocks, threads>>>(d_prog, d_off, d_c0, kmax, c->d_cols64,
                                                 c->n, nsegs, ntrials, w.keys, w.rows, d_status);
        ok = cudaGetLastError() == cudaSuccess;
        if (ok && rc_rank_runs(&w, c->d_rank_b, c->n, total, nsegs, d_seg_off) != 0) ok = 0;
        if (ok) {
            k_rank_contrib<<<nsegs, 256>>>(w.starts, w.prefix, w.ord, c->n, total, nsegs, d_out);
            k_rank_bb<<<1, 256>>>(c->d_rank_b, c->n, d_bb);
            ok = cudaGetLastError() == cudaSuccess;
        }
        cudaEventRecord(c->t1, 0);
        cudaEventSynchronize(c->t1);
        if (ok) {
            cudaEventElapsedTime(&ms, c->t0, c->t1);
            rc_download(out, d_out, sizeof(unsigned long long) * 4 * (size_t)nsegs);
            rc_download(out_status, d_status, sizeof(int) * (size_t)nsegs);
            rc_download(out_bb, d_bb, sizeof(unsigned long long) * 2);
        }
    }
    if (d_prog) cudaFree(d_prog);
    if (d_off) cudaFree(d_off);
    if (d_p) cudaFree(d_p);
    if (d_c0) cudaFree(d_c0);
    if (d_out) cudaFree(d_out);
    if (d_status) cudaFree(d_status);
    if (d_bb) cudaFree(d_bb);
    if (d_seg_off) cudaFree(d_seg_off);
    rc_rank_work_free(&w);
    if (!ok) return -1.0f;
    return ms;
}

/* #230 test hook: copy the cached target ranks to the host so a parity test
   can check the per-row `B` instead of only the sums. */
int rc_rank_debug_b(void *ctx, long long *out)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !out) return -1;
    if (rc_rank_prepare(ctx) != 0) return -1;
    rc_download(out, c->d_rank_b, sizeof(long long) * (size_t)c->n);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* 1 when the context uses managed (unified) memory, 0 for plain device memory.
   The benchmark prints this so a recorded run states which path it measured. */
int rc_unified(void)
{
#ifdef RCSR_UNIFIED
    return 1;
#else
    return 0;
#endif
}

const char *rc_device_name(void)
{
    static char name[256];
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "unknown";
    snprintf(name, sizeof(name), "%s (sm_%d%d, %d SMs, %d MB)",
             p.name, p.major, p.minor, p.multiProcessorCount,
             (int)(p.totalGlobalMem / (1024 * 1024)));
    return name;
}

/* Number of CUDA devices, or a negative value when the runtime/driver query
   itself fails. Used by the one-time startup probe (issue #141); it allocates
   nothing, so it can run before the dataset is uploaded. */
int rc_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return -1;
    return n;
}

} /* extern "C" */
