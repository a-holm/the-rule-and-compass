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
#include <math_constants.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
    unsigned int *d_prog;
    unsigned int *d_off;
    float *d_out;
    /* #122: fixed-capacity, double-buffered genome arena. */
    unsigned int *d_pool[2];
    unsigned long long *d_hash;
    int pool_cap;
    int pool_width;
    int hash_cap;
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

__global__ void eval_kernel(const float *__restrict__ cols,
                            const float *__restrict__ y,
                            int n,
                            int v,
                            const unsigned int *__restrict__ prog,
                            const unsigned int *__restrict__ off,
                            float *__restrict__ out)
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
        acc += r * r;
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
__global__ void eval_pool_kernel(const float *__restrict__ cols,
                                 const float *__restrict__ y,
                                 int n,
                                 int v,
                                 const unsigned int *__restrict__ prog,
                                 int width,
                                 float *__restrict__ out)
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
        acc += r * r;
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
__global__ void predict_kernel(const float *__restrict__ cols,
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

#define FT_MAX_CONST 16
#define FT_MAX_NODES 96
/* Lower triangle + right-hand side + SSE. */
#define FT_MAX_STATS (FT_MAX_CONST * (FT_MAX_CONST + 1) / 2 + FT_MAX_CONST + 1)
#define FT_CHUNK 256
#define FT_PCHUNK 128

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

__device__ void ft_assemble_d(const unsigned int *nd,
                              int m,
                              const double *__restrict__ cols,
                              const double *__restrict__ y,
                              int n,
                              int p,
                              const double *c,
                              double *dg,
                              double *dr,
                              double *jtj,
                              double *jtr,
                              double *psse,
                              int *bad)
{
    const int tid = threadIdx.x;
    const int tri = p * (p + 1) / 2;
    const int total = tri + p + 1;
    int kind = -1;
    int ei = 0, ej = 0, which = 0;
    if (tid < total) {
        if (tid < tri) {
            kind = 0;
            int k = 0;
            for (int i = 0; i < p; ++i) {
                for (int j = 0; j <= i; ++j, ++k) {
                    if (k == tid) {
                        ei = i;
                        ej = j;
                    }
                }
            }
        } else if (tid < tri + p) {
            kind = 1;
            which = tid - tri;
        } else {
            kind = 2;
        }
    }
    double acc = 0.0;
    for (int base = 0; base < n; base += FT_PCHUNK) {
        const int rows = min(FT_PCHUNK, n - base);
        if (tid < FT_PCHUNK) {
            const int row = base + tid;
            if (row < n) {
                double v[FT_MAX_NODES];
                double adj[FT_MAX_NODES];
                double g[FT_MAX_CONST];
                ft_forward_d(nd, m, cols, n, row, c, v);
                const double pred = v[m - 1];
                const double r = pred - y[row];
                bool ok = isfinite(pred) && fabs(pred) <= 1e30 && isfinite(r);
                ft_backward_d(nd, m, v, adj, g, p);
                for (int j = 0; j < p; ++j) {
                    if (!isfinite(g[j])) ok = false;
                    dg[tid * p + j] = ok ? g[j] : 0.0;
                }
                dr[tid] = ok ? r : 0.0;
                if (!ok) *bad = 1;
            } else {
                for (int j = 0; j < p; ++j) dg[tid * p + j] = 0.0;
                dr[tid] = 0.0;
            }
        }
        __syncthreads();
        if (kind == 0) {
            for (int k = 0; k < rows; ++k) acc += dg[k * p + ei] * dg[k * p + ej];
        } else if (kind == 1) {
            for (int k = 0; k < rows; ++k) acc += dg[k * p + which] * dr[k];
        } else if (kind == 2) {
            for (int k = 0; k < rows; ++k) acc += dr[k] * dr[k];
        }
        __syncthreads();
    }
    if (kind == 0) {
        int k = 0;
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j <= i; ++j, ++k) {
                if (k == tid) jtj[k] = acc;
            }
        }
    } else if (kind == 1) {
        jtr[which] = acc;
    } else if (kind == 2) {
        *psse = acc;
    }
    __syncthreads();
}

__device__ double ft_sse_d(const unsigned int *nd,
                           int m,
                           const double *__restrict__ cols,
                           const double *__restrict__ y,
                           int n,
                           const double *c,
                           double *red)
{
    const int tid = threadIdx.x;
    double local = 0.0;
    if (tid < FT_PCHUNK) {
        for (int row = tid; row < n; row += FT_PCHUNK) {
            double v[FT_MAX_NODES];
            ft_forward_d(nd, m, cols, n, row, c, v);
            const double pred = v[m - 1];
            const double r = pred - y[row];
            if (!isfinite(pred) || fabs(pred) > 1e30 || !isfinite(r)) {
                local += 1e30;
            } else {
                local += r * r;
            }
        }
    }
    red[tid] = local;
    __syncthreads();
    for (int st = FT_PCHUNK / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const double total = red[0];
    __syncthreads();
    return total;
}

/* Assemble J^T J, J^T r and the SSE at `c`. Each statistic is owned by one
 * thread (thread t < stats owns entry t) and accumulated in block-local fixed
 * row order, so the reduction is deterministic and needs no atomics. */
__device__ void ft_assemble(const unsigned int *nd,
                            int m,
                            const float *__restrict__ cols,
                            const float *__restrict__ y,
                            int n,
                            int p,
                            const double *c,
                            float *sg,
                            float *sr,
                            double *jtj,
                            double *jtr,
                            double *psse,
                            int *bad)
{
    const int tid = threadIdx.x;
    const int tri = p * (p + 1) / 2;
    const int total = tri + p + 1;
    int kind = -1;   /* 0 = jtj, 1 = jtr, 2 = sse */
    int ei = 0, ej = 0, which = 0;
    if (tid < total) {
        if (tid < tri) {
            kind = 0;
            int k = 0;
            for (int i = 0; i < p; ++i) {
                for (int j = 0; j <= i; ++j, ++k) {
                    if (k == tid) {
                        ei = i;
                        ej = j;
                    }
                }
            }
        } else if (tid < tri + p) {
            kind = 1;
            which = tid - tri;
        } else {
            kind = 2;
        }
    }
    double acc = 0.0;
    for (int base = 0; base < n; base += FT_CHUNK) {
        const int rows = min(FT_CHUNK, n - base);
        const int row = base + tid;
        if (row < n) {
            float vt[FT_MAX_NODES];
            float adj[FT_MAX_NODES];
            float g[FT_MAX_CONST];
            ft_forward(nd, m, cols, n, row, c, vt);
            const float pred = vt[m - 1];
            const float r = pred - y[row];
            bool ok = isfinite(pred) && fabsf(pred) <= 1e30f && isfinite(r);
            ft_backward(nd, m, vt, adj, g, p);
            for (int j = 0; j < p; ++j) {
                if (!isfinite(g[j])) ok = false;
                sg[tid * p + j] = ok ? g[j] : 0.0f;
            }
            sr[tid] = ok ? r : 0.0f;
            if (!ok) *bad = 1;
        } else {
            for (int j = 0; j < p; ++j) sg[tid * p + j] = 0.0f;
            sr[tid] = 0.0f;
        }
        __syncthreads();
        if (kind == 0) {
            for (int k = 0; k < rows; ++k) {
                acc += (double)(sg[k * p + ei] * sg[k * p + ej]);
            }
        } else if (kind == 1) {
            for (int k = 0; k < rows; ++k) {
                acc += (double)(sg[k * p + which] * sr[k]);
            }
        } else if (kind == 2) {
            for (int k = 0; k < rows; ++k) {
                acc += (double)(sr[k] * sr[k]);
            }
        }
        __syncthreads();
    }
    if (kind == 0) {
        int k = 0;
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j <= i; ++j, ++k) {
                if (k == tid) jtj[k] = acc;
            }
        }
    } else if (kind == 1) {
        jtr[which] = acc;
    } else if (kind == 2) {
        *psse = acc;
    }
    __syncthreads();
}

/* Value-only SSE sweep of `c` for the trial step. Fixed-order f64 reduction. */
__device__ double ft_sse(const unsigned int *nd,
                         int m,
                         const float *__restrict__ cols,
                         const float *__restrict__ y,
                         int n,
                         const double *c,
                         double *red)
{
    const int tid = threadIdx.x;
    double local = 0.0;
    for (int row = tid; row < n; row += FT_CHUNK) {
        float vt[FT_MAX_NODES];
        ft_forward(nd, m, cols, n, row, c, vt);
        const float pred = vt[m - 1];
        const float r = pred - y[row];
        if (!isfinite(pred) || fabsf(pred) > 1e30f || !isfinite(r)) {
            local += 1e30;
        } else {
            local += (double)r * (double)r;
        }
    }
    red[tid] = local;
    __syncthreads();
    for (int st = FT_CHUNK / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    const double total = red[0];
    __syncthreads();
    return total;
}

/* Damped Cholesky solve in f64. Returns 0 on success, 1 on a non-positive
 * pivot or a condition-number gate (`rho = max L_ii / min L_ii > 1e4`). */
__device__ int ft_solve(const double *jtj,
                        const double *jtr,
                        double lambda,
                        int p,
                        double *delta,
                        double *L)
{
    /* Conditioning gate on the UNDAMPED J^T J: damping deliberately hides a
       rank-deficient direction (that is what it is for), so the gate must not
       see it. A pivot at the 1e-12 diagonal floor or a Cholesky diagonal ratio
       above 1e4 marks the candidate for the exact CPU refit. */
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = jtj[i * (i + 1) / 2 + j];
            if (i == j) s += 1e-12;
            for (int k = 0; k < j; ++k) s -= L[i * p + k] * L[j * p + k];
            if (i == j) {
                if (!(s > 1e-18) || !isfinite(s)) return 1;
                L[i * p + i] = sqrt(s);
            } else {
                L[i * p + j] = s / L[j * p + j];
            }
        }
    }
    double lo = L[0], hi = L[0];
    for (int i = 1; i < p; ++i) {
        lo = fmin(lo, L[i * p + i]);
        hi = fmax(hi, L[i * p + i]);
    }
    if (!(lo > 0.0) || !isfinite(hi) || hi / lo > 1e4) return 1;
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
    double rhs[FT_MAX_CONST];
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

__global__ void fit_kernel(const float *__restrict__ cols,
                           const float *__restrict__ y,
                           const double *__restrict__ cols64,
                           const double *__restrict__ y64,
                           int n,
                           int v,
                           const unsigned int *__restrict__ prog,
                           const unsigned int *__restrict__ off,
                           const int *__restrict__ pcount,
                           int kmax,
                           const double *__restrict__ c0,
                           int iters,
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
    const int tri = p * (p + 1) / 2;

    __shared__ double sc[FT_MAX_CONST];
    __shared__ double sjtj[FT_MAX_STATS];
    __shared__ double sjtr[FT_MAX_CONST];
    __shared__ double sL[FT_MAX_CONST * FT_MAX_CONST];
    __shared__ double strial[FT_MAX_CONST];
    __shared__ double sdelta[FT_MAX_CONST];
    __shared__ double sred[FT_CHUNK];
    __shared__ float sg[FT_CHUNK * FT_MAX_CONST];
    __shared__ float sr[FT_CHUNK];
    __shared__ double sdg[FT_PCHUNK * FT_MAX_CONST];
    __shared__ double sdr[FT_PCHUNK];
    __shared__ int sbad;
    __shared__ int ssolved;
    __shared__ double slambda;
    __shared__ double sbest;

    if (tid < p) sc[tid] = c0[(size_t)cand * (size_t)kmax + (size_t)tid];
    if (tid == 0) {
        sbad = 0;
        slambda = 1e-3;
        sbest = 0.0;
    }
    __syncthreads();

    int status = 0;
    int assembled = 0;
    double best = 0.0;

    if (iters > 0) {
        /* First assembly also supplies the baseline SSE. */
        ft_assemble(nd, m, cols, y, n, p, sc, sg, sr, sjtj, sjtr, &sbest, &sbad);
        assembled = 1;
        best = sbest;
        if (sbad) status = 2;
        for (int it = 0; it < iters && status == 0; ++it) {
            if (!assembled) {
                ft_assemble(nd, m, cols, y, n, p, sc, sg, sr, sjtj, sjtr, &sbest, &sbad);
                assembled = 1;
                if (sbad) {
                    status = 2;
                    break;
                }
            }
            if (tid == 0) {
                ssolved = (ft_solve(sjtj, sjtr, slambda, p, sdelta, sL) == 0);
            }
            __syncthreads();
            if (!ssolved) {
                status = 1;
                break;
            }
            if (tid < p) {
                double t = sc[tid] + sdelta[tid];
                strial[tid] = fmin(1e6, fmax(-1e6, t));
            }
            __syncthreads();
            double tsse = ft_sse(nd, m, cols, y, n, strial, sred);
            if (tsse < best) {
                if (tid < p) sc[tid] = strial[tid];
                best = tsse;
                if (tid == 0) slambda = fmax(slambda / 8.0, 1e-12);
                assembled = 0;
                if (best == 0.0) break;
            } else {
                if (tid == 0) slambda *= 6.0;
                if (slambda > 1e9) break;
            }
            __syncthreads();
        }
        /* Re-baseline the acceptance SSE in f64: `best` so far came from the
           f32 assembly, and comparing an f64 trial against it would reject
           every polish step on a systematic rounding difference alone. */
        if (status == 0) {
            const double exact = ft_sse_d(nd, m, cols64, y64, n, sc, sred);
            if (isfinite(exact)) best = exact;
        }
        /* Undamped Gauss-Newton polish, exactly like the CPU path. */
        for (int it = 0; it < 4 && status == 0 && best != 0.0; ++it) {
            ft_assemble_d(nd, m, cols64, y64, n, p, sc, sdg, sdr, sjtj, sjtr, &sbest, &sbad);
            assembled = 1;
            if (sbad) {
                status = 2;
                break;
            }
            if (tid == 0) ssolved = (ft_solve(sjtj, sjtr, 0.0, p, sdelta, sL) == 0);
            __syncthreads();
            if (!ssolved) break;
            if (tid < p) {
                double t = sc[tid] + sdelta[tid];
                strial[tid] = fmin(1e6, fmax(-1e6, t));
            }
            __syncthreads();
            double tsse = ft_sse_d(nd, m, cols64, y64, n, strial, sred);
            if (tsse < best) {
                if (tid < p) sc[tid] = strial[tid];
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
        for (int i = 0; i < p; ++i) outc[(size_t)cand * (size_t)kmax + (size_t)i] = sc[i];
        outsse[cand] = best;
        outstatus[cand] = status;
    }
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
    if (c->d_prog) cudaFree(c->d_prog);
    if (c->d_off) cudaFree(c->d_off);
    if (c->d_out) cudaFree(c->d_out);
    if (c->d_pool[0]) cudaFree(c->d_pool[0]);
    if (c->d_pool[1]) cudaFree(c->d_pool[1]);
    if (c->d_hash) cudaFree(c->d_hash);
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
float rc_eval(void *ctx, float *out)
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
                                       c->d_prog, c->d_off, c->d_out);
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
float rc_pool_eval(void *ctx, int which, int nprog, int width, float *out)
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
                                         c->d_pool[which & 1], width, c->d_out);
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
   rounding. Called once per run when the GPU-fit arm is selected. */
int rc_set_fit_data(void *ctx, const double *cols, const double *y)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !cols || !y) return -1;
    if (c->d_cols64) { cudaFree(c->d_cols64); c->d_cols64 = 0; }
    if (c->d_y64) { cudaFree(c->d_y64); c->d_y64 = 0; }
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
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* --- #169: batched LM fit entry point -------------------------------------
 *
 * `prog` is the concatenated 4-word node records; `off` has nprog+1 word
 * offsets; `pcount` has one constant count per candidate; constants start at
 * `c0[(cand*kmax)+i]`. Returns the kernel time in milliseconds, or a negative
 * value when the launch failed (the caller then keeps the CPU arm).
 */
float rc_fit_batch(void *ctx,
                   const unsigned int *prog,
                   const unsigned int *off,
                   const int *pcount,
                   int nprog,
                   int kmax,
                   const double *c0,
                   int iters,
                   double *outc,
                   double *outsse,
                   int *outstatus)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nprog <= 0 || kmax <= 0) return -1.0f;
    if (kmax > FT_MAX_CONST) return -1.0f;
    unsigned int *d_prog = 0;
    unsigned int *d_off = 0;
    int *d_p = 0;
    double *d_c0 = 0;
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
        cudaEventRecord(c->t0, 0);
        fit_kernel<<<nprog, FT_CHUNK>>>(c->d_cols, c->d_y, c->d_cols64, c->d_y64, c->n, c->v,
                                        d_prog, d_off, d_p, kmax, d_c0,
                                        iters, d_c, d_sse, d_status);
        cudaEventRecord(c->t1, 0);
        cudaEventSynchronize(c->t1);
        if (cudaGetLastError() != cudaSuccess) {
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
    if (d_c) cudaFree(d_c);
    if (d_sse) cudaFree(d_sse);
    if (d_status) cudaFree(d_status);
    if (!ok) return -1.0f;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c->t0, c->t1);
    return ms;
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
