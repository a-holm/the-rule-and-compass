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

/* #285: host-visible arena state. `steady_alloc_calls` counts allocator calls
   after the first successful batch; the hot-path acceptance is that it stays
   at zero while the batch shape is stable. */
typedef struct RcArenaReport {
    int async_copy;
    char reason[96];
    unsigned long long pinned_bytes;
    unsigned long long alloc_calls;
    unsigned long long steady_alloc_calls;
    unsigned long long overlapped_uploads;
    unsigned long long batches;
} RcArenaReport;

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
    /* #234: the per-row label sign `t = 2l - 1` (`l` from the target
       midpoint), uploaded once by `rc_set_fit_sign` and read only by the
       log-loss arm of the batched fit. Both precisions are resident because the
       f32 assembly sweep and the f64 polish sweep read the same rows. Null
       until uploaded, which makes a log-loss launch a refused launch rather
       than a fit on another objective. */
    float *d_tsign;
    double *d_tsign64;
    int has_tsign;
    unsigned int *d_prog;
    unsigned int *d_off;
    float *d_out;
    /* #287: native-score readback buffers. `d_score_out` is f64 and
       `d_score_status` carries 0 (ok) or 2 (non-finite); they are separate from
       `d_out` so the legacy f32 SSE evaluator keeps its exact ABI. */
    double *d_score_out;
    int *d_score_status;
    int score_cap;
    /* #287: cached target moments for the two-pass correlation scorer. They are
       computed once from the resident f32 target at rc_create time. */
    double ybar;
    double syy;
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
    /* #227: reusable device arena for the exact maximum-error scan. The
       bisection on the error level calls the scan once per round, so the
       per-call `cudaMalloc`/`cudaFree` churn of the one-shot selection arms
       would dominate; the arena is grown, never shrunk. */
    void *d_mm;
    size_t mm_bytes;
    int n;
    int v;
    int nprog;
    int out_cap;
    cudaEvent_t t0, t1;
    /* #118: the current batch's ledger and its start on the same monotonic
       clock, so `wall_ms` covers the whole begin..download path. */
    RcStats stats;
    double batch_start_ms;

    /* --- #285: persistent batch arena, pinned staging, streams and events ---
     *
     * The batch API used to call `cudaFree`/`cudaMalloc` for `d_prog`/`d_off`
     * and `d_out` on every batch, and every copy and kernel ran on the default
     * stream. Both allocator calls synchronise the device, so a generation
     * could never overlap the previous one. The arena below is allocated once
     * at the high-water mark and grown only through an explicit reserve; the
     * steady-state batch loop performs no allocator call at all.
     *
     * Two program slots let the host upload batch B while batch A is still
     * executing: each slot carries its own upload-done and kernel-done event,
     * and a slot is only overwritten after its own completion event. The two
     * pinned rings (`h_up`, `h_down`) are bounded by RC_PIN_BUDGET_BYTES; a
     * copy that cannot be staged in pinned memory falls back to the blocking
     * path with a recorded reason instead of silently becoming synchronous. */
    unsigned int *arena_prog[2];
    unsigned int *arena_off[2];
    size_t arena_prog_words;   /* capacity of each slot, in u32 words */
    size_t arena_off_words;    /* capacity of each slot, in u32 words */
    float *arena_out;          /* persistent device output, nprog floats */
    size_t arena_out_cap;      /* capacity in floats */
    int arena_slot;            /* slot selected by the last rc_set_programs */
    int arena_upload_async;    /* this batch's upload was enqueued on s_h2d */
    /* Host staging rings. `h_up` holds one batch's packed programs plus its
       offsets; `h_down` holds one batch's errors. The two slots alternate. */
    unsigned char *h_up[2];
    unsigned char *h_down[2];
    size_t h_up_bytes;
    size_t h_down_bytes;
    size_t pinned_bytes;
    cudaStream_t s_h2d;
    cudaStream_t s_d2h;
    cudaStream_t s_eval;
    cudaEvent_t ev_upload[2];     /* program slot -> upload complete */
    cudaEvent_t ev_slot_done[2];  /* program slot -> kernel complete */
    cudaEvent_t ev_up_free[2];    /* host staging slot -> copy out complete */
    cudaEvent_t ev_down_free[2];  /* download staging slot -> copy complete */
    int up_next;
    int down_next;
    /* #285: one in-flight batch, so the compatibility `rc_eval` can be split
       into an async submit and a collecting wait for the overlap test. */
    struct {
        int active;
        int slot;
        int nprog;
        size_t out_bytes;
        int down_slot;
        int d2h_async;
        cudaEvent_t done;
    } pending;
    /* #285: accounting that tells cold allocation apart from a steady-state
       request. `warm` is set after the first successful batch; any allocator
       call after that is a bug the tests can see. */
    int async_copy;            /* 1 = pinned async copies, 0 = synchronous */
    int warm;
    unsigned long long alloc_calls;
    unsigned long long steady_alloc_calls;
    unsigned long long overlapped_uploads;
    unsigned long long arena_batches;
    char sync_reason[96];
};

/* #285: the pinned ring is capped at 64 MiB. Two upload slots plus two
   download slots must fit inside it, so each direction can use at most half. */
#define RC_PIN_BUDGET_BYTES (64u * 1024u * 1024u)

/* #285: an allocator call is part of the ledger. Before the first batch it is
   cold setup; afterwards it is a steady-state request and must stay zero on
   the hot path. */
static int rc_arena_dev_alloc(RcCtx *c, void **p, size_t bytes)
{
    int rc = (int)cudaMalloc(p, bytes);
    if (rc == (int)cudaSuccess) {
        c->alloc_calls++;
        if (c->warm) c->steady_alloc_calls++;
    }
    return rc;
}

static void rc_arena_dev_free(RcCtx *c, void *p)
{
    if (!p) return;
    cudaFree(p);
    c->alloc_calls++;
    if (c->warm) c->steady_alloc_calls++;
}

/* Reserve the device arena at the requested high-water mark. Capacity only
   grows, and only when no batch is in flight: reallocating a slot that a
   pending kernel still reads would be a use-after-free. */
static int rc_arena_reserve(RcCtx *c, int nprog, size_t prog_words)
{
    size_t off_words = (size_t)nprog + 1;
    size_t out_len = (size_t)nprog;
    if (prog_words <= c->arena_prog_words && off_words <= c->arena_off_words &&
        out_len <= c->arena_out_cap) {
        return 0;
    }
    if (c->pending.active) {
        snprintf(c->sync_reason, sizeof(c->sync_reason),
                 "arena growth requested while a batch is in flight");
        return -1;
    }
    size_t want_prog = prog_words > c->arena_prog_words ? prog_words : c->arena_prog_words;
    size_t want_off = off_words > c->arena_off_words ? off_words : c->arena_off_words;
    size_t want_out = out_len > c->arena_out_cap ? out_len : c->arena_out_cap;
    unsigned int *p0 = 0, *p1 = 0, *o0 = 0, *o1 = 0;
    float *out = 0;
    int ok = 1;
    if (rc_arena_dev_alloc(c, (void **)&p0, sizeof(unsigned int) * want_prog) != 0) ok = 0;
    if (ok && rc_arena_dev_alloc(c, (void **)&p1, sizeof(unsigned int) * want_prog) != 0) ok = 0;
    if (ok && rc_arena_dev_alloc(c, (void **)&o0, sizeof(unsigned int) * want_off) != 0) ok = 0;
    if (ok && rc_arena_dev_alloc(c, (void **)&o1, sizeof(unsigned int) * want_off) != 0) ok = 0;
    if (ok && rc_arena_dev_alloc(c, (void **)&out, sizeof(float) * want_out) != 0) ok = 0;
    if (!ok) {
        if (p0) cudaFree(p0);
        if (p1) cudaFree(p1);
        if (o0) cudaFree(o0);
        if (o1) cudaFree(o1);
        if (out) cudaFree(out);
        snprintf(c->sync_reason, sizeof(c->sync_reason),
                 "device arena allocation failed");
        return -1;
    }
    if (c->arena_prog[0]) cudaFree(c->arena_prog[0]);
    if (c->arena_prog[1]) cudaFree(c->arena_prog[1]);
    if (c->arena_off[0]) cudaFree(c->arena_off[0]);
    if (c->arena_off[1]) cudaFree(c->arena_off[1]);
    if (c->arena_out) cudaFree(c->arena_out);
    c->arena_prog[0] = p0;
    c->arena_prog[1] = p1;
    c->arena_off[0] = o0;
    c->arena_off[1] = o1;
    c->arena_out = out;
    c->arena_prog_words = want_prog;
    c->arena_off_words = want_off;
    c->arena_out_cap = want_out;
    return 0;
}

/* Reserve the pinned staging rings. The total is hard-capped at 64 MiB; a
   request that does not fit is a labelled fallback to the blocking path, never
   an over-budget pin. A failed cudaHostAlloc is handled the same way. */
static int rc_pin_reserve(RcCtx *c, size_t up_bytes, size_t down_bytes)
{
    if (up_bytes <= c->h_up_bytes && down_bytes <= c->h_down_bytes) return 0;
    size_t want_up = up_bytes > c->h_up_bytes ? up_bytes : c->h_up_bytes;
    size_t want_down = down_bytes > c->h_down_bytes ? down_bytes : c->h_down_bytes;
    if (2 * (want_up + want_down) > RC_PIN_BUDGET_BYTES) {
        snprintf(c->sync_reason, sizeof(c->sync_reason),
                 "pinned staging request exceeds the 64 MiB ring");
        return -1;
    }
    unsigned char *u0 = 0, *u1 = 0, *d0 = 0, *d1 = 0;
    int ok = 1;
    if (cudaHostAlloc((void **)&u0, want_up, cudaHostAllocDefault) != cudaSuccess) ok = 0;
    if (ok && cudaHostAlloc((void **)&u1, want_up, cudaHostAllocDefault) != cudaSuccess) ok = 0;
    if (ok && cudaHostAlloc((void **)&d0, want_down, cudaHostAllocDefault) != cudaSuccess) ok = 0;
    if (ok && cudaHostAlloc((void **)&d1, want_down, cudaHostAllocDefault) != cudaSuccess) ok = 0;
    if (!ok) {
        if (u0) cudaFreeHost(u0);
        if (u1) cudaFreeHost(u1);
        if (d0) cudaFreeHost(d0);
        if (d1) cudaFreeHost(d1);
        snprintf(c->sync_reason, sizeof(c->sync_reason),
                 "cudaHostAlloc for the pinned staging ring failed");
        return -1;
    }
    if (c->h_up[0]) cudaFreeHost(c->h_up[0]);
    if (c->h_up[1]) cudaFreeHost(c->h_up[1]);
    if (c->h_down[0]) cudaFreeHost(c->h_down[0]);
    if (c->h_down[1]) cudaFreeHost(c->h_down[1]);
    c->h_up[0] = u0;
    c->h_up[1] = u1;
    c->h_down[0] = d0;
    c->h_down[1] = d1;
    c->h_up_bytes = want_up;
    c->h_down_bytes = want_down;
    c->pinned_bytes = 2 * (want_up + want_down);
    c->alloc_calls++;
    if (c->warm) c->steady_alloc_calls++;
    return 0;
}

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
/* #222: mean absolute error. The evaluator, the batched fit's objective and
   its per-row IRLS weight all derive from the same code, so a run can never
   fit or rank |r| with a squared-error surrogate. */
#define RC_LOSS_ABSOLUTE 4
/* #234: the logistic cross-entropy on the binary label, `Σ w ln(1 + exp(-t·p))`
   with `p` read as a logit and `t = 2l - 1`. Fit-only: the evaluator has no
   label or logit semantics, so this code is never passed to
   `rc_eval`/`rc_pool_eval`. The per-row weight is the Fisher weight `q(1-q)`
   and the response is `t/q`, which reproduce `logistic_irls`'s Newton step. */
#define RC_LOSS_LOGISTIC 5

/* #287: native score-objective codes (`ScoreObjective::code` in metrics.rs).
   These are deliberately separate from the fit `RC_LOSS_*` codes: fitting and
   scoring are different decisions, and `log_loss` remains fit-only. */
#define RC_SCORE_SQUARED 0
#define RC_SCORE_ABSOLUTE 1
#define RC_SCORE_LOG_SQUASHED 2
#define RC_SCORE_SIGNED 3
#define RC_SCORE_MAXIMUM 4
#define RC_SCORE_CORRELATION 5
#define RC_SCORE_CORRELATION_SIGNED 6
#define RC_SCORE_AIC 7
#define RC_SCORE_AICC 8
#define RC_SCORE_BIC 9
#define RC_SCORE_NONFINITE 2

/* #222: the L1 majoriser floor, the same `1e-6` the CPU `irls_fit` uses. It is
   deliberately not smaller: at |r| = 0 the exact weight is infinite, and a
   finite floor keeps a single interpolated row from reaching a weight that
   overflows the f32 product with a large adjoint. */
#define FT_L1_EPS 1e-6

__device__ __forceinline__ float rc_row_loss(float r, int loss)
{
    if (loss == RC_LOSS_LOG_SQUASHED) {
        const float rr = r * r;
        return isfinite(rr) ? log1pf(rr) : 2.0f * logf(fabsf(r));
    }
    if (loss == RC_LOSS_ABSOLUTE) {
        return fabsf(r);
    }
    if (loss != RC_LOSS_SQUARED) {
        /* #234: codes 2, 3 and 5 are fit-only objectives. Returning NaN makes
           an accidental evaluator launch visibly wrong instead of silently
           ranking that run by squared error (the #236 defect). */
        return CUDART_NAN_F;
    }
    return r * r;
}

/* #222: the per-row weight of the least-squares surrogate. `base` is the
   user's `--weight` (1.0 when unweighted). `log_squashed` uses
   `2/(1 + r^2)`, the derivative of `ln(1 + r^2)` split over the residual;
   `absolute` uses the Lawson/L1 majoriser `1/max(|r|, eps)`. Both are
   computed in f64 so an L1 weight never picks up an f32 rounding step of its
   own. For every other objective the weight is the base weight, bit for bit. */
__device__ __forceinline__ double ft_row_weight(double base, double r, int metric)
{
    if (metric == RC_LOSS_LOG_SQUASHED) return base * 2.0 / (1.0 + r * r);
    if (metric == RC_LOSS_ABSOLUTE) return base / fmax(fabs(r), FT_L1_EPS);
    return base;
}

/* #234: sigmoid clamped exactly like `logistic_irls` (q in [1e-9, 1-1e-9]).
   The clamp bounds the pseudo-response `t/q`, keeps `q(1-q)` off the f32
   denormal range (which `--use_fast_math` would flush to zero) and makes both
   row factors finite for every finite margin. */
__device__ __forceinline__ double rc_logistic_q(double u)
{
    double q;
    if (u >= 0.0) {
        q = 1.0 / (1.0 + exp(-u));
    } else {
        const double e = exp(u);
        q = e / (1.0 + e);
    }
    return fmin(1.0 - 1e-9, fmax(1e-9, q));
}

/* #234: `ln(1 + exp(-u))` in the Lin–Lin–Weng form
   `max(-u, 0) + log1p(exp(-|u|))`: never overflows, exact for small `|u|`
   (where the naive form cancels) and monotone, so the trial gate compares a
   true log-likelihood. */
__device__ __forceinline__ double rc_softplus_neg(double u)
{
    const double a = fabs(u);
    return (u < 0.0 ? -u : 0.0) + log1p(exp(-a));
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

/* --- #287: native score-objective kernels -------------------------------
 *
 * One block per candidate, threads stride rows. The reduction tree is fixed
 * (256 slots) and every thread visits rows in ascending order, so the result
 * depends only on `(program, n)` and never on batch composition, grid size or
 * which block a candidate happened to land in. No floating-point atomics are
 * used; the only atomic is the integer "a row was non-finite" flag.
 *
 * MODE is a compile-time RC_SCORE_* code: the statistic is selected by the
 * template, not by a runtime switch inside the row loop. Correlation needs two
 * passes (sum p, then centred moments); every other objective is one pass.
 *
 * A non-finite prediction (or one above the same 1e30 hard-magnitude limit the
 * old f32 evaluator used) writes `+inf` and status 2. It never writes a finite
 * sentinel that the host could divide by n.
 */
template <int MODE>
__global__ void __launch_bounds__(RC_MAX_THREADS, 6) score_kernel(
    const float *__restrict__ cols,
    const float *__restrict__ y,
    int n,
    int v,
    const unsigned int *__restrict__ prog,
    const unsigned int *__restrict__ off,
    int width,
    int pooled,
    double *__restrict__ out,
    int *__restrict__ status,
    double ybar,
    double syy)
{
    const int tid = threadIdx.x;
    const int p = blockIdx.x;
    const unsigned int *code = pooled
        ? prog + (size_t)p * (size_t)width
        : prog + off[p];
    __shared__ double sh1[RC_MAX_THREADS];
    __shared__ double sh2[RC_MAX_THREADS];
    __shared__ unsigned long long shk[RC_MAX_THREADS];
    __shared__ int shi[RC_MAX_THREADS];
    __shared__ int sbad;

    int bad = 0;
    double a = 0.0;
    double b = 0.0;
    unsigned long long best_key = 0;
    int best_row = 0;

    if (MODE == RC_SCORE_CORRELATION || MODE == RC_SCORE_CORRELATION_SIGNED) {
        /* Pass 1: mean prediction. */
        for (int i = tid; i < n; i += blockDim.x) {
            const float pred = rc_exec_row(code, cols, n, i);
            if (!isfinite(pred) || fabsf(pred) > 1e30f) {
                bad = 1;
            } else {
                a += (double)pred;
            }
        }
        if (tid == 0) sbad = 0;
        sh1[tid] = a;
        __syncthreads();
        if (bad) atomicOr(&sbad, 1);
        __syncthreads();
        if (sbad) {
            if (tid == 0) {
                out[p] = INFINITY;
                status[p] = RC_SCORE_NONFINITE;
            }
            return;
        }
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if ((int)tid < st) sh1[tid] += sh1[tid + st];
            __syncthreads();
        }
        const double pbar = sh1[0] / (double)n;

        /* Pass 2: centred sums. ybar/syy were cached per fit sample. */
        bad = 0;
        a = 0.0;
        b = 0.0;
        for (int i = tid; i < n; i += blockDim.x) {
            const float pred = rc_exec_row(code, cols, n, i);
            if (!isfinite(pred) || fabsf(pred) > 1e30f) {
                bad = 1;
                continue;
            }
            const double dp = (double)pred - pbar;
            const double dy = (double)y[i] - ybar;
            a += dp * dp;
            b += dp * dy;
        }
        if (tid == 0) sbad = 0;
        sh1[tid] = a;
        sh2[tid] = b;
        __syncthreads();
        if (bad) atomicOr(&sbad, 1);
        __syncthreads();
        if (sbad) {
            if (tid == 0) {
                out[p] = INFINITY;
                status[p] = RC_SCORE_NONFINITE;
            }
            return;
        }
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if ((int)tid < st) {
                sh1[tid] += sh1[tid + st];
                sh2[tid] += sh2[tid + st];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const double spp = sh1[0];
            const double spy = sh2[0];
            /* The CPU rule: zero variance on either side is rho = 0. */
            const double rho = (spp <= 0.0 || syy <= 0.0)
                ? 0.0
                : spy / (sqrt(spp) * sqrt(syy));
            out[p] = (MODE == RC_SCORE_CORRELATION_SIGNED)
                ? 1.0 - rho
                : 1.0 - fabs(rho);
            status[p] = 0;
        }
        return;
    }

    /* One-pass reductions. Modes 7/8/9 accumulate RSS; the host applies the
       AIC family's `k` penalty, which differs per candidate. */
    for (int i = tid; i < n; i += blockDim.x) {
        const float pred = rc_exec_row(code, cols, n, i);
        if (!isfinite(pred) || fabsf(pred) > 1e30f) {
            bad = 1;
            continue;
        }
        const double r = (double)pred - (double)y[i];
        if (MODE == RC_SCORE_SQUARED || MODE == RC_SCORE_AIC ||
            MODE == RC_SCORE_AICC || MODE == RC_SCORE_BIC) {
            a += r * r;
        } else if (MODE == RC_SCORE_ABSOLUTE) {
            a += fabs(r);
        } else if (MODE == RC_SCORE_LOG_SQUASHED) {
            const double rr = r * r;
            a += isfinite(rr) ? log1p(rr) : 2.0 * log(fabs(r));
        } else if (MODE == RC_SCORE_SIGNED) {
            a += r;
        } else if (MODE == RC_SCORE_MAXIMUM) {
            const double ar = fabs(r);
            const unsigned long long key = (unsigned long long)__double_as_longlong(ar);
            if (key > best_key) {
                best_key = key;
                best_row = i;
            }
        }
    }
    if (tid == 0) sbad = 0;
    sh1[tid] = a;
    shk[tid] = best_key;
    shi[tid] = best_row;
    __syncthreads();
    if (bad) atomicOr(&sbad, 1);
    __syncthreads();
    if (sbad) {
        if (tid == 0) {
            out[p] = INFINITY;
            status[p] = RC_SCORE_NONFINITE;
        }
        return;
    }
    if (MODE == RC_SCORE_MAXIMUM) {
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if ((int)tid < st) {
                const unsigned long long rk = shk[tid + st];
                const int rr = shi[tid + st];
                if (rk > shk[tid] || (rk == shk[tid] && rr < shi[tid])) {
                    shk[tid] = rk;
                    shi[tid] = rr;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[p] = __longlong_as_double((long long)shk[0]);
            status[p] = 0;
        }
    } else {
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if ((int)tid < st) sh1[tid] += sh1[tid + st];
            __syncthreads();
        }
        if (tid == 0) {
            const double sum = sh1[0];
            if (MODE == RC_SCORE_AIC || MODE == RC_SCORE_AICC || MODE == RC_SCORE_BIC) {
                out[p] = sum; /* RSS; host applies the penalty. */
            } else if (MODE == RC_SCORE_SIGNED) {
                out[p] = fabs(sum) / (double)n;
            } else {
                out[p] = sum / (double)n;
            }
            status[p] = 0;
        }
    }
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
    /* #286: label-only outcomes: rank-deficient design, rows whose gradient
       was skipped, and the number of accepted trials. */
    int srank;
    int sskip;
    int simproved;
    int sstart;
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
                              const double *__restrict__ tsign,
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
                              int *bad,
                              int *skipped)
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
                /* #286: `1e30` is an f32 overflow guard; the f64 assembly and
                   loss accept any finite value (the CPU arm fits 1e31). */
                bool ok = isfinite(pred) && isfinite(r);
                ft_backward_d<MAXC>(nd, m, v, adj, g, p);
                /* #286: a finite value with a non-finite derivative (e.g.
                   `pow(x,c)` at x=0, where `x^c ln x = 0·−∞`) is a bad
                   *gradient* row, not a bad prediction. Drop that row's Gram
                   contribution and keep its loss so one row in 2.46M no longer
                   aborts the candidate. */
                bool grad_ok = true;
                for (int j = 0; j < p; ++j) {
                    if (!isfinite(g[j])) grad_ok = false;
                }
                /* #229/#222: the MM/IRLS weight. `2/(1 + r^2)` is the
                   log-squashed `rho'(r)/r`; `1/max(|r|, eps)` is the Lawson
                   L1 majoriser. The constant factors a positive weight picks
                   up cannot move the weighted least-squares argmin. The
                   weight scales the whole row (adjoint and residual), so
                   J^T W J stays PSD even where the true Hessian is indefinite
                   (L1) or negative (log-squashed beyond |r| = 1). */
                /* #234: `sw = w·q(1-q)` and `resp = -t/q` give the same
                   `J^T r = -Σ t(1-q) g` (the loss gradient) as the f32 sweep,
                   in exact f64. */
                double sw;
                double resp = r;
                double rho;
                if (metric == RC_LOSS_LOGISTIC) {
                    const double tt = tsign ? tsign[row] : 0.0;
                    const double u = tt * pred;
                    if (!tsign || !isfinite(tt)) ok = false;
                    const double q = rc_logistic_q(u);
                    sw = wr * q * (1.0 - q);
                    resp = -tt / q;
                    rho = rc_softplus_neg(u);
                } else {
                    sw = ft_row_weight(wr, r, metric);
                    /* The objective statistic is always the configured loss, in
                       f64: `|r|` for absolute, `ln(1+r²)` for log-squashed, and
                       the historical `r*r` for squared. */
                    if (metric == RC_LOSS_SQUARED) {
                        rho = r * r;
                    } else if (metric == RC_LOSS_ABSOLUTE) {
                        rho = fabs(r);
                    } else {
                        rho = log1p(r * r);
                    }
                }
                const double s = sqrt(sw);
                const bool use_grad = ok && grad_ok;
                for (int j = 0; j < p; ++j) {
                    dg[tid * p + j] = use_grad ? s * g[j] : 0.0;
                }
                dr[tid] = use_grad ? s * resp : 0.0;
                dl[tid] = ok ? wr * rho : 0.0;
                if (!ok) *bad = 1;
                if (ok && !grad_ok) atomicAdd(skipped, 1);
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
                            const double *__restrict__ tsign,
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
            if (!isfinite(pred) || !isfinite(r)) {
                local += 1e30;
            } else if (metric == RC_LOSS_LOGISTIC) {
                const double tt = tsign ? tsign[row] : 0.0;
                local += isfinite(tt) ? wr * rc_softplus_neg(tt * pred) : 1e30;
            } else if (metric == RC_LOSS_SQUARED) {
                local += wr * r * r;
            } else if (metric == RC_LOSS_ABSOLUTE) {
                local += wr * fabs(r);
            } else {
                local += wr * log1p(r * r);
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
                            const float *__restrict__ tsign,
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
            /* #229/#222: `sqrt(w_i · 2/(1 + r_i^2))` (log-squashed) or
               `sqrt(w_i / max(|r_i|, eps))` (absolute) scales the row's
               adjoint and residual, so the rank-one update below is unchanged
               and `A` stays positive semidefinite even where rho'' < 0.
               #234: the logistic arm folds the Fisher weight `q(1-q)` into the
               same square root and carries `-t/q` in the residual, so
               `Σ (s g)(s resp) = -Σ t(1-q) g` is exactly the gradient of the
               loss that `ft_solve`'s `A δ = -J^T r` consumes. */
            double sw;
            double resp = (double)r;
            double rowloss;
            if (metric == RC_LOSS_LOGISTIC) {
                const double tt = tsign ? (double)tsign[row] : 0.0;
                const double u = tt * (double)pred;
                if (!tsign || !isfinite(tt)) ok = false;
                const double q = rc_logistic_q(u);
                sw = wr * q * (1.0 - q);
                resp = -tt / q;
                rowloss = wr * rc_softplus_neg(u);
            } else {
                sw = ft_row_weight(wr, (double)r, metric);
                rowloss = (double)wr * (double)rc_row_loss(r, metric);
            }
            const float s = sqrtf((float)sw);
            for (int j = 0; j < p; ++j) {
                if (!isfinite(g[j])) ok = false;
                sg[tid * p + j] = ok ? s * g[j] : 0.0f;
            }
            sr[tid] = ok ? (float)(s * resp) : 0.0f;
            sl[tid] = ok ? rowloss : 0.0;
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
                          const float *__restrict__ tsign,
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
        } else if (metric == RC_LOSS_LOGISTIC) {
            /* #234: the configured loss is the logistic cross-entropy of the
               signed margin, never an SSE of the 0/1 target. */
            const double tt = tsign ? (double)tsign[row] : 0.0;
            local += isfinite(tt) ? wr * rc_softplus_neg(tt * (double)pred) : 1e30;
        } else if (metric == RC_LOSS_SQUARED) {
            /* Keep the historical f64 square: the f32 product would round the
               trial score differently from the assembly's own statistic. */
            local += wr * (double)r * (double)r;
        } else {
            /* #222: the configured objective itself (`|r|`, `ln(1+r²)`), in
               f32 per row and f64 in the sum, exactly like the assembly. */
            local += wr * (double)rc_row_loss(r, metric);
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

/* Damped Cholesky solve in f64. Returns 0 on success, 4 when the *undamped*
 * normal equations are rank deficient but the damped step was still solved,
 * and 1 when no finite step can be produced.
 *
 * #286: every arm uses the scale-free `ln det Ã = 2Σ ln L_ii − Σ ln A_ii`
 * gate on `Ã = D^{-1/2} A D^{-1/2}`, with `D = diag(JᵀJ)`. The old absolute
 * pivot floor and `hi/lo <= 1e4` ratio compared derivative-column *units*
 * (they rejected a perfectly orthogonal design whose constants span decades
 * and accepted κ = 2e8). The step itself is Jacobi-equilibrated: solve
 * `Ã δ̃ = −D^{-1/2} jᵀr`, then `δ = D^{-1/2} δ̃`. A rank-deficient direction
 * is *reported* (status 4), not handed to a 70× more expensive CPU refit:
 * the damping exists precisely to make it well-posed and the exact finalist
 * refit owns every reported number. */
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
    (void)scale_free; /* #286: the scale-free gate is universal now. */
    double dscale[MAXC];
    for (int i = 0; i < p; ++i) {
        const double aii = jtj[i * (i + 1) / 2 + i];
        if (!(aii > 0.0) || !isfinite(aii)) return 1;
        dscale[i] = sqrt(aii);
    }

    /* Conditioning gate on the UNDAMPED, Jacobi-scaled J^T J: damping
       deliberately hides a rank-deficient direction, so the gate must not see
       it. A non-positive pivot or a determinant below the floor means the data
       does not identify all p parameters; that is reported (4), not fatal. */
    int rank_deficient = 0;
    double sum_ln_L = 0.0;
    int ok = 1;
    for (int i = 0; i < p && ok; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = jtj[i * (i + 1) / 2 + j] / (dscale[i] * dscale[j]);
            if (!isfinite(s)) return 1;
            for (int k = 0; k < j; ++k) s -= L[i * p + k] * L[j * p + k];
            if (i == j) {
                if (!(s > 0.0) || !isfinite(s)) {
                    ok = 0;
                    break;
                }
                L[i * p + i] = sqrt(s);
                sum_ln_L += 0.5 * log(s);
            } else {
                L[i * p + j] = s / L[j * p + j];
            }
        }
    }
    if (!ok) {
        rank_deficient = 1;
    } else {
        const double ln_det = 2.0 * sum_ln_L;
        if (!isfinite(ln_det) || ln_det < FT_LN_DET_FLOOR) rank_deficient = 1;
    }

    /* Damped factorisation of the equilibrated system for the step itself. */
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = jtj[i * (i + 1) / 2 + j] / (dscale[i] * dscale[j]);
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
    for (int i = 0; i < p; ++i) rhs[i] = -jtr[i] / dscale[i];
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
    for (int i = 0; i < p; ++i) delta[i] /= dscale[i];
    return rank_deficient ? 4 : 0;
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
                           const float *__restrict__ tsign,
                           const double *__restrict__ cols64,
                           const double *__restrict__ y64,
                           const double *__restrict__ tsign64,
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
    /* #229/#222: the scale-free determinant gate replaces the ratio gate for
       the robust arms, whose normal equations are badly scaled by
       construction (L1 weights span 1..1e6). */
    /* #234: the logistic arm is row-weighted by `q(1-q) <= 0.25` and asks the
       same identification question, so it uses the scale-free gate too. */
    const int scale_free = (metric == RC_LOSS_LOG_SQUASHED || metric == RC_LOSS_ABSOLUTE ||
                            metric == RC_LOSS_LOGISTIC);

    extern __shared__ char ft_smem_raw[];
    FtSmem<MAXC, CHUNK, PCHUNK> *S =
        reinterpret_cast<FtSmem<MAXC, CHUNK, PCHUNK> *>(ft_smem_raw);

    if (tid < p) S->sc[tid] = c0[(size_t)cand * (size_t)kmax + (size_t)tid];
    if (tid == 0) {
        S->sbad = 0;
        S->srank = 0;
        S->sskip = 0;
        S->simproved = 0;
        S->sstart = 0;
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
            if (tid == 0) S->sstart = S->sbad ? 0 : 1;
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
                        const int solve = ft_solve<MAXC>(S->sjtj, S->sjtr, lambda, p, S->sdelta,
                                                         S->sL, 0);
                        S->ssolved = (solve != 1);
                        if (solve == 4) S->srank = 1;
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
                        if (tid == 0) S->simproved = 1;
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
                        const int solve = ft_solve<MAXC>(S->sjtj, S->sjtr, 0.0, p, S->sdelta,
                                                         S->sL, 0);
                        S->ssolved = (solve != 1);
                        if (solve == 4) S->srank = 1;
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
                        if (tid == 0) S->simproved = 1;
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
            /* #286: a finite correlation answer is returned, never refit for a
               rank/gauge label the exact finalist refit already owns. */
            if (status == 1 || status == 2) {
                if (!S->sstart) status = 7;
                else status = S->simproved ? 5 : 6;
            } else if (status == 0 && S->srank) {
                status = 4;
            }
            for (int i = 0; i < p; ++i) outc[(size_t)cand * (size_t)kmax + (size_t)i] = S->sc[i];
            outsse[cand] = best;
            outstatus[cand] = status;
        }
        return;
    }

    int status = 0;
    int assembled = 0;
    double best = 0.0;

    /* #222: the L1 arm re-derives its per-row weights from the residuals of
       the current constants. `rounds` outer IRLS rounds of `inner` damped LM
       sweeps mirror the CPU `irls_fit` split (four weight updates, `iters/4`
       sweeps each for the default of 8), and every trial is judged by `|r|`.
       The squared and log-squashed arms keep one round and their historical
       step sequence bit for bit. */
    const int irls = (metric == RC_LOSS_ABSOLUTE);
    const int rounds = irls ? (iters < 1 ? 1 : (iters < 4 ? iters : 4)) : 1;
    const int inner = irls ? (iters / rounds < 1 ? 1 : iters / rounds) : iters;

    if (iters > 0) {
        /* First assembly also supplies the baseline objective. */
        ft_assemble<MAXC, CHUNK>(nd, m, cols, y, tsign, w64, n, p, metric, S->sc, S->sg, S->sr, S->sdl, S->sjtj, S->sjtr, &S->sbest, &S->sbad);
        assembled = 1;
        best = S->sbest;
        if (tid == 0) S->sstart = S->sbad ? 0 : 1;
        if (S->sbad) status = 2;
        for (int round = 0; round < rounds && status == 0 && best != 0.0; ++round) {
            if (irls && round > 0) assembled = 0; /* new weights, new Gram */
            for (int it = 0; it < inner && status == 0; ++it) {
                if (!assembled) {
                    ft_assemble<MAXC, CHUNK>(nd, m, cols, y, tsign, w64, n, p, metric, S->sc, S->sg, S->sr, S->sdl, S->sjtj, S->sjtr, &S->sbest, &S->sbad);
                    assembled = 1;
                    if (S->sbad) {
                        status = 2;
                        break;
                    }
                }
                if (tid == 0) {
                    const int solve = ft_solve<MAXC>(S->sjtj, S->sjtr, S->slambda, p, S->sdelta, S->sL, scale_free);
                    S->ssolved = (solve != 1);
                    if (solve == 4) S->srank = 1;
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
                double tsse = ft_loss<CHUNK>(nd, m, cols, y, tsign, w64, n, metric, S->strial, S->sred);
                /* #222: true-metric Armijo. The weighted surrogate can point
                   uphill for `|r|`; halving the step twice costs two forward
                   passes and never accepts a trial that is worse in the
                   configured loss. Every other objective keeps the historical
                   single-trial acceptance. */
                for (int half = 0; half < (irls ? 2 : 0) && !(tsse < best); ++half) {
                    if (tid < p) {
                        S->strial[tid] = S->sc[tid] + 0.5 * (S->strial[tid] - S->sc[tid]);
                    }
                    __syncthreads();
                    tsse = ft_loss<CHUNK>(nd, m, cols, y, tsign, w64, n, metric, S->strial, S->sred);
                }
                if (tsse < best) {
                    if (tid < p) S->sc[tid] = S->strial[tid];
                    best = tsse;
                    if (tid == 0) {
                        S->slambda = fmax(S->slambda / 8.0, 1e-12);
                        S->simproved = 1;
                    }
                    assembled = 0;
                    if (best == 0.0) break;
                } else {
                    if (tid == 0) S->slambda *= 6.0;
                    /* #234: `S->slambda` is written by thread 0 only, so the
                       exit test needs a barrier to be block-uniform. Without
                       it the `break` can be taken by a subset of the block and
                       the rest then waits forever at the next `__syncthreads()`
                       (a deadlock that shows as 100 % device use and a host
                       spinning in `cudaEventSynchronize`). Any fit that stalls
                       reaches it once λ crosses 1e9; the logistic arm's f32
                       objective is flat once converged, so it does so
                       reliably for `--fit-iters >= 28`. */
                    __syncthreads();
                    if (S->slambda > 1e9) break;
                }
                __syncthreads();
            }
        }
        /* Re-baseline the acceptance objective in f64: `best` so far came from
           the f32 assembly, and comparing an f64 trial against it would reject
           every polish step on a systematic rounding difference alone. */
        if (status == 0) {
            const double exact = ft_loss_d<PCHUNK>(nd, m, cols64, y64, tsign64, w64, n, metric, S->sc, S->sred);
            if (isfinite(exact)) best = exact;
        }
        /* Undamped Gauss-Newton polish, exactly like the CPU path. */
        for (int it = 0; it < 4 && status == 0 && best != 0.0; ++it) {
            ft_assemble_d<MAXC, PCHUNK>(nd, m, cols64, y64, tsign64, w64, n, p, metric, S->sc, S->sdg, S->sdr, S->sdl64, S->sjtj, S->sjtr, &S->sbest, &S->sbad, &S->sskip);
            assembled = 1;
            if (S->sbad) {
                status = 2;
                break;
            }
            if (tid == 0) {
                const int solve = ft_solve<MAXC>(S->sjtj, S->sjtr, 0.0, p, S->sdelta, S->sL, scale_free);
                S->ssolved = (solve != 1);
                if (solve == 4) S->srank = 1;
            }
            __syncthreads();
            if (!S->ssolved) break;
            if (tid < p) {
                double t = S->sc[tid] + S->sdelta[tid];
                S->strial[tid] = fmin(1e6, fmax(-1e6, t));
            }
            __syncthreads();
            double tsse = ft_loss_d<PCHUNK>(nd, m, cols64, y64, tsign64, w64, n, metric, S->strial, S->sred);
            /* #222: one exact f64 backtrack for the non-smooth objective. */
            if (irls && !(tsse < best)) {
                if (tid < p) {
                    S->strial[tid] = S->sc[tid] + 0.5 * (S->strial[tid] - S->sc[tid]);
                }
                __syncthreads();
                tsse = ft_loss_d<PCHUNK>(nd, m, cols64, y64, tsign64, w64, n, metric, S->strial, S->sred);
            }
            if (tsse < best) {
                if (tid < p) S->sc[tid] = S->strial[tid];
                best = tsse;
                if (tid == 0) S->simproved = 1;
            } else {
                break;
            }
            __syncthreads();
        }
    }

    /* #286: label a finite answer instead of discarding it.
     *   4 rank_deficient   finite; null direction; the damping is the point
     *   5 recovered        last accepted finite iterate after a failed trial
     *   6 no_improvement   finite start, no trial improved it
     *   7 start_unusable   the start itself has no finite value (log only)
     *   8 recovered_skipped a finite answer whose bad-gradient rows were skipped
     * Only a genuine arithmetic failure with no finite start remains status 1,
     * and only a non-finite start with no accepted iterate is status 7; neither
     * needs the CPU arm, which cannot move a non-finite start either. */
    if (tid == 0) {
        if (status == 1 || status == 2) {
            if (!S->sstart) status = 7;
            else if (S->simproved) status = 5;
            else status = 6;
        } else if (status == 0) {
            if (S->sskip > 0) status = 8;
            else if (S->srank) status = 4;
            else if (!S->simproved && iters > 0) status = 6;
        }
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
        c->d_cols, c->d_y, c->d_tsign, c->d_cols64, c->d_y64, c->d_tsign64, c->d_w64, c->n,
        c->v, d_prog, d_off, d_p, kmax, d_c0, d_gauge, iters, metric, d_c, d_sse, d_status);
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
    c->async_copy = 1;
    c->sync_reason[0] = '\0';
#ifdef RCSR_UNIFIED
    /* Managed pages migrate on access; an explicit pinned copy would add a
       second transport. Keep the managed build on its proven synchronous
       path and label it. */
    c->async_copy = 0;
    snprintf(c->sync_reason, sizeof(c->sync_reason),
             "unified-memory build uses synchronous managed copies");
#endif
    const char *force_sync = getenv("RCSR_GPU_SYNC");
    if (force_sync && force_sync[0] && strcmp(force_sync, "0") != 0 &&
        strcmp(force_sync, "false") != 0 && strcmp(force_sync, "off") != 0) {
        c->async_copy = 0;
        snprintf(c->sync_reason, sizeof(c->sync_reason),
                 "RCSR_GPU_SYNC requests the synchronous copy path");
    }
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
    /* #287: cache the target mean and centred sum of squares. The device
       correlator re-executes predictions for its second pass but never pays to
       recompute the target side. */
    c->ybar = 0.0;
    c->syy = 0.0;
    if (n > 0) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += (double)y[i];
        c->ybar = s / (double)n;
        double ss = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = (double)y[i] - c->ybar;
            ss += d * d;
        }
        c->syy = ss;
    }
    cudaEventCreate(&c->t0);
    cudaEventCreate(&c->t1);
    /* #285: copy streams and the per-slot events. A failed stream/event
       creation is not fatal: the batch path then keeps its synchronous
       semantics with a recorded reason. */
    int streams_ok = 1;
    if (cudaStreamCreateWithFlags(&c->s_h2d, cudaStreamNonBlocking) != cudaSuccess) streams_ok = 0;
    if (streams_ok && cudaStreamCreateWithFlags(&c->s_d2h, cudaStreamNonBlocking) != cudaSuccess) streams_ok = 0;
    if (streams_ok && cudaStreamCreateWithFlags(&c->s_eval, cudaStreamNonBlocking) != cudaSuccess) streams_ok = 0;
    for (int i = 0; streams_ok && i < 2; ++i) {
        if (cudaEventCreateWithFlags(&c->ev_upload[i], cudaEventDisableTiming) != cudaSuccess) streams_ok = 0;
        if (streams_ok && cudaEventCreateWithFlags(&c->ev_slot_done[i], cudaEventDisableTiming) != cudaSuccess) streams_ok = 0;
        if (streams_ok && cudaEventCreateWithFlags(&c->ev_up_free[i], cudaEventDisableTiming) != cudaSuccess) streams_ok = 0;
        if (streams_ok && cudaEventCreateWithFlags(&c->ev_down_free[i], cudaEventDisableTiming) != cudaSuccess) streams_ok = 0;
    }
    if (!streams_ok) {
        c->async_copy = 0;
        snprintf(c->sync_reason, sizeof(c->sync_reason),
                 "CUDA stream/event creation failed; synchronous copy path");
    }
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
    if (c->d_tsign) cudaFree(c->d_tsign);
    if (c->d_tsign64) cudaFree(c->d_tsign64);
    if (c->d_prog) cudaFree(c->d_prog);
    if (c->d_off) cudaFree(c->d_off);
    if (c->d_out) cudaFree(c->d_out);
    if (c->d_score_out) cudaFree(c->d_score_out);
    if (c->d_score_status) cudaFree(c->d_score_status);
    if (c->d_pool[0]) cudaFree(c->d_pool[0]);
    if (c->d_pool[1]) cudaFree(c->d_pool[1]);
    if (c->d_hash) cudaFree(c->d_hash);
    if (c->d_rank_b) cudaFree(c->d_rank_b);
    if (c->d_mm) cudaFree(c->d_mm);
    /* #285: arena, staging rings, streams and events. */
    for (int i = 0; i < 2; ++i) {
        if (c->arena_prog[i]) cudaFree(c->arena_prog[i]);
        if (c->arena_off[i]) cudaFree(c->arena_off[i]);
        if (c->h_up[i]) cudaFreeHost(c->h_up[i]);
        if (c->h_down[i]) cudaFreeHost(c->h_down[i]);
        if (c->ev_upload[i]) cudaEventDestroy(c->ev_upload[i]);
        if (c->ev_slot_done[i]) cudaEventDestroy(c->ev_slot_done[i]);
        if (c->ev_up_free[i]) cudaEventDestroy(c->ev_up_free[i]);
        if (c->ev_down_free[i]) cudaEventDestroy(c->ev_down_free[i]);
    }
    if (c->arena_out) cudaFree(c->arena_out);
    if (c->s_h2d) cudaStreamDestroy(c->s_h2d);
    if (c->s_d2h) cudaStreamDestroy(c->s_d2h);
    if (c->s_eval) cudaStreamDestroy(c->s_eval);
    cudaEventDestroy(c->t0);
    cudaEventDestroy(c->t1);
    free(c);
}

int rc_set_programs(void *ctx, const unsigned int *prog, const unsigned int *off, int nprog)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !prog || !off || nprog <= 0) return -1;
    RcStats *st = &c->stats;
    size_t prog_words = (size_t)off[nprog];
    size_t prog_bytes = sizeof(unsigned int) * prog_words;
    size_t off_bytes = sizeof(unsigned int) * (size_t)(nprog + 1);
    double t = rc_now_ms();
    /* #285: the arena is allocated at its high-water mark, not per batch. A
       growth is a cold event and is refused while a batch is in flight. */
    if (rc_arena_reserve(c, nprog, prog_words) != 0) {
        st->alloc_ms += rc_now_ms() - t;
        st->failed = 1;
        st->wall_ms = rc_now_ms() - c->batch_start_ms;
        return -1;
    }
    st->alloc_ms += rc_now_ms() - t;
    /* Alternate program slots. A slot is only overwritten after the kernel
       that last read it has completed; the other slot may still be running,
       which is exactly what lets the next upload overlap it. */
    t = rc_now_ms();
    int slot = c->arena_slot ^ 1;
    if (c->ev_slot_done[slot]) cudaEventSynchronize(c->ev_slot_done[slot]);
    if (c->pending.active && cudaEventQuery(c->t1) == cudaErrorNotReady) {
        c->overlapped_uploads++;
    }
    st->sync_ms += rc_now_ms() - t;

    t = rc_now_ms();
    int upload_async = 0;
    if (c->async_copy) {
        size_t pin_before = c->pinned_bytes;
        if (rc_pin_reserve(c, prog_bytes + off_bytes,
                           sizeof(float) * (size_t)nprog) == 0) {
            upload_async = 1;
        } else {
            c->async_copy = 0;
        }
        if (c->pinned_bytes != pin_before) {
            st->alloc_ms += rc_now_ms() - t;
            t = rc_now_ms();
        }
    }
    if (upload_async) {
        int ui = c->up_next;
        c->up_next ^= 1;
        /* The staging slot was last read by the copy issued two uploads ago;
           wait for that copy before overwriting it. A never-recorded event is
           complete, so the first batch does not block. */
        cudaEventSynchronize(c->ev_up_free[ui]);
        memcpy(c->h_up[ui], prog, prog_bytes);
        memcpy(c->h_up[ui] + prog_bytes, off, off_bytes);
        cudaMemcpyAsync(c->arena_prog[slot], c->h_up[ui], prog_bytes,
                        cudaMemcpyHostToDevice, c->s_h2d);
        cudaMemcpyAsync(c->arena_off[slot], c->h_up[ui] + prog_bytes, off_bytes,
                        cudaMemcpyHostToDevice, c->s_h2d);
        cudaEventRecord(c->ev_upload[slot], c->s_h2d);
        cudaEventRecord(c->ev_up_free[ui], c->s_h2d);
    } else {
        /* Blocking fallback: exact same bytes, no overlap. */
        rc_upload(c->arena_prog[slot], prog, prog_bytes);
        rc_upload(c->arena_off[slot], off, off_bytes);
    }
    c->arena_slot = slot;
    c->arena_upload_async = upload_async;
    c->nprog = nprog;
    st->h2d_ms += rc_now_ms() - t;
    st->h2d_bytes += (unsigned long long)(prog_bytes + off_bytes);
    st->batch = nprog;
    st->wall_ms = rc_now_ms() - c->batch_start_ms;
    return 0;
}

/* #285: enqueue the current batch's kernel, then its asynchronous download.
   No host synchronisation happens here; `rc_eval_collect` owns the wait. */
int rc_eval_submit(void *ctx, int loss)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || c->nprog <= 0 || c->pending.active) return -1;
    int slot = c->arena_slot;
    if (!c->arena_prog[slot] || !c->arena_off[slot] || !c->arena_out) return -1;
    RcStats *st = &c->stats;
    int threads = RC_MIN_THREADS;
    while (threads < RC_MAX_THREADS && threads < c->n) threads <<= 1;
    if (c->arena_upload_async) {
        cudaStreamWaitEvent(c->s_eval, c->ev_upload[slot], 0);
    }
    double t = rc_now_ms();
    cudaEventRecord(c->t0, c->s_eval);
    eval_kernel<<<c->nprog, threads, 0, c->s_eval>>>(
        c->d_cols, c->d_y, c->n, c->v, c->arena_prog[slot], c->arena_off[slot],
        c->arena_out, loss);
    cudaEventRecord(c->t1, c->s_eval);
    cudaEventRecord(c->ev_slot_done[slot], c->s_eval);
    st->launch_ms += rc_now_ms() - t;
    if (cudaGetLastError() != cudaSuccess) {
        st->failed = 1;
        st->wall_ms = rc_now_ms() - c->batch_start_ms;
        return -1;
    }
    size_t out_bytes = sizeof(float) * (size_t)c->nprog;
    int down_slot = -1;
    int d2h_async = 0;
    if (c->async_copy && c->h_down[0] && c->h_down_bytes >= out_bytes) {
        down_slot = c->down_next;
        c->down_next ^= 1;
        cudaEventSynchronize(c->ev_down_free[down_slot]);
        cudaStreamWaitEvent(c->s_d2h, c->t1, 0);
        cudaMemcpyAsync(c->h_down[down_slot], c->arena_out, out_bytes,
                        cudaMemcpyDeviceToHost, c->s_d2h);
        cudaEventRecord(c->ev_down_free[down_slot], c->s_d2h);
        d2h_async = 1;
    }
    c->pending.active = 1;
    c->pending.slot = slot;
    c->pending.nprog = c->nprog;
    c->pending.out_bytes = out_bytes;
    c->pending.down_slot = down_slot;
    c->pending.d2h_async = d2h_async;
    c->pending.done = d2h_async ? c->ev_down_free[down_slot] : c->t1;
    return 0;
}

/* #285: wait for the submitted batch and copy its errors into `out`. This is
   the only place the compatibility shim blocks, and it blocks on an event, not
   `cudaStreamSynchronize`. */
float rc_eval_collect(void *ctx, float *out)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !out || !c->pending.active) return -1.0f;
    RcStats *st = &c->stats;
    double t = rc_now_ms();
    cudaEventSynchronize(c->pending.done);
    st->sync_ms += rc_now_ms() - t;
    t = rc_now_ms();
    if (c->pending.d2h_async) {
        memcpy(out, c->h_down[c->pending.down_slot], c->pending.out_bytes);
    } else {
        rc_download(out, c->arena_out, c->pending.out_bytes);
    }
    st->d2h_ms += rc_now_ms() - t;
    st->d2h_bytes += (unsigned long long)c->pending.out_bytes;
    int failed = 0;
    if (cudaGetLastError() != cudaSuccess) failed = 1;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c->t0, c->t1);
    st->kernel_ms = (double)ms;
    st->batch = c->pending.nprog;
    st->wall_ms = rc_now_ms() - c->batch_start_ms;
    if (failed) st->failed = 1;
    c->pending.active = 0;
    c->warm = 1;
    c->arena_batches++;
    return failed ? -1.0f : ms;
}

/* Returns elapsed milliseconds; writes nprog errors into `out`. The context's
   RcStats records the full launch+sync+download path. #285: this is the
   compatibility shim over submit+collect, so existing callers keep their
   synchronous contract while the copy/launch machinery stays asynchronous. */
float rc_eval(void *ctx, float *out, int loss)
{
    if (rc_eval_submit(ctx, loss) != 0) return -1.0f;
    return rc_eval_collect(ctx, out);
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
    int slot = c->arena_slot;
    /* #285: the upload may still be in flight on S_h2d. */
    if (c->arena_upload_async) cudaStreamWaitEvent(0, c->ev_upload[slot], 0);
    size_t total = (size_t)nprog * (size_t)c->n;
    float *d_pred = 0;
    if (cudaMalloc((void **)&d_pred, sizeof(float) * total) != 0) return -1;
    int threads = RC_MIN_THREADS;
    while (threads < RC_MAX_THREADS && threads < c->n) threads <<= 1;
    predict_kernel<<<nprog, threads>>>(c->d_cols, c->n, c->arena_prog[slot],
                                       c->arena_off[slot], d_pred);
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

/* #285: snapshot of the arena/copy mode for tests, the benchmark and the
   report. `reason` is the recorded cause when the synchronous path is active. */
int rc_arena_report(void *ctx, RcArenaReport *out)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !out) return -1;
    memset(out, 0, sizeof(*out));
    out->async_copy = c->async_copy;
    snprintf(out->reason, sizeof(out->reason), "%s", c->sync_reason);
    out->pinned_bytes = (unsigned long long)c->pinned_bytes;
    out->alloc_calls = c->alloc_calls;
    out->steady_alloc_calls = c->steady_alloc_calls;
    out->overlapped_uploads = c->overlapped_uploads;
    out->batches = c->arena_batches;
    return 0;
}

/* #285: the labelled fallback switch. `sync != 0` forces the blocking copy
   path (used by the parity test and by a caller that knows its staging will
   not fit); `sync == 0` re-enables async copies if the streams exist. Returns
   the previous mode. */
int rc_arena_force_sync(void *ctx, int sync)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return -1;
    int previous = c->async_copy;
    if (sync) {
        c->async_copy = 0;
        snprintf(c->sync_reason, sizeof(c->sync_reason),
                 "caller forced the synchronous copy path");
    } else if (c->s_h2d && c->s_d2h && c->s_eval) {
        c->async_copy = 1;
        c->sync_reason[0] = '\0';
    }
    return previous;
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

/* #287: grow the f64 score readback buffers. They are separate from the
   legacy f32 `d_out`, so the SSE ABI is unchanged. Returns 0 on success. */
static int rc_score_reserve(RcCtx *c, int nprog)
{
    if (c->score_cap >= nprog) return 0;
    if (c->d_score_out) { cudaFree(c->d_score_out); c->d_score_out = 0; }
    if (c->d_score_status) { cudaFree(c->d_score_status); c->d_score_status = 0; }
    if (rc_alloc((void **)&c->d_score_out, sizeof(double) * (size_t)nprog) != 0 ||
        rc_alloc((void **)&c->d_score_status, sizeof(int) * (size_t)nprog) != 0) {
        if (c->d_score_out) { cudaFree(c->d_score_out); c->d_score_out = 0; }
        if (c->d_score_status) { cudaFree(c->d_score_status); c->d_score_status = 0; }
        c->score_cap = 0;
        return -1;
    }
    c->score_cap = nprog;
    return 0;
}

/* #287: AIC-family post-processing, kept on the host so `k` can differ per
   candidate without a device upload. The formula mirrors `metrics.rs`. */
static double rc_aic_value(int kind, double rss, int n, int k)
{
    const double dn = (double)n;
    const double dk = (double)k;
    const double base = (rss <= 0.0) ? -INFINITY : dn * log(rss / dn);
    if (kind == RC_SCORE_AIC) return base + 2.0 * dk;
    if (kind == RC_SCORE_BIC) return base + dk * log(dn);
    /* AICc */
    if (!isfinite(base)) return base;
    const double denom = dn - dk - 1.0;
    if (denom <= 0.0) return INFINITY;
    return base + 2.0 * dk + 2.0 * dk * (dk + 1.0) / denom;
}

/* Shared launcher for the packed and fixed-slot pool entry points. */
static float rc_score_launch(RcCtx *c, const unsigned int *prog,
                             const unsigned int *off, int width, int pooled,
                             int nprog, int kind, const int *pcount,
                             double *out, int *status)
{
    RcStats *st = &c->stats;
    double t = rc_now_ms();
    if (rc_score_reserve(c, nprog) != 0) {
        st->failed = 1;
        st->alloc_ms += rc_now_ms() - t;
        st->wall_ms = rc_now_ms() - c->batch_start_ms;
        return -1.0f;
    }
    st->alloc_ms += rc_now_ms() - t;
    int threads = RC_MIN_THREADS;
    while (threads < RC_MAX_THREADS && threads < c->n) threads <<= 1;

    t = rc_now_ms();
    cudaEventRecord(c->t0, 0);
    switch (kind) {
        case RC_SCORE_SQUARED:
            score_kernel<RC_SCORE_SQUARED><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_ABSOLUTE:
            score_kernel<RC_SCORE_ABSOLUTE><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_LOG_SQUASHED:
            score_kernel<RC_SCORE_LOG_SQUASHED><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_SIGNED:
            score_kernel<RC_SCORE_SIGNED><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_MAXIMUM:
            score_kernel<RC_SCORE_MAXIMUM><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_CORRELATION:
            score_kernel<RC_SCORE_CORRELATION><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_CORRELATION_SIGNED:
            score_kernel<RC_SCORE_CORRELATION_SIGNED><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_AIC:
            score_kernel<RC_SCORE_AIC><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_AICC:
            score_kernel<RC_SCORE_AICC><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        case RC_SCORE_BIC:
            score_kernel<RC_SCORE_BIC><<<nprog, threads>>>(
                c->d_cols, c->d_y, c->n, c->v, prog, off, width, pooled,
                c->d_score_out, c->d_score_status, c->ybar, c->syy);
            break;
        default:
            st->failed = 1;
            st->wall_ms = rc_now_ms() - c->batch_start_ms;
            return -1.0f;
    }
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
    rc_download(out, c->d_score_out, sizeof(double) * (size_t)nprog);
    rc_download(status, c->d_score_status, sizeof(int) * (size_t)nprog);
    st->d2h_ms += rc_now_ms() - t;
    st->d2h_bytes += sizeof(double) * (unsigned long long)nprog
                   + sizeof(int) * (unsigned long long)nprog;
    /* AIC family: convert the device RSS to the configured penalty here. */
    if (kind == RC_SCORE_AIC || kind == RC_SCORE_AICC || kind == RC_SCORE_BIC) {
        for (int i = 0; i < nprog; ++i) {
            if (status[i] == 0) {
                const int k = pcount ? pcount[i] : 0;
                out[i] = rc_aic_value(kind, out[i], c->n, k);
            }
        }
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c->t0, c->t1);
    st->kernel_ms = (double)ms;
    st->batch = nprog;
    st->wall_ms = rc_now_ms() - c->batch_start_ms;
    return ms;
}

/* #287: score a packed batch. Returns milliseconds, negative on failure. */
float rc_score_batch(void *ctx, const unsigned int *prog,
                     const unsigned int *off, int nprog, int kind,
                     const int *pcount, double *out, int *status)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nprog <= 0 || !out || !status) return -1.0f;
    if (rc_set_programs(ctx, prog, off, nprog) != 0) return -1.0f;
    /* #285 moved the packed upload into the persistent arena. The legacy
       `d_prog`/`d_off` pointers are no longer filled, so score the arena slot
       `rc_set_programs` just wrote. Wait for its H2D copy before launching. */
    const int slot = c->arena_slot;
    if (!c->arena_prog[slot] || !c->arena_off[slot]) return -1.0f;
    if (c->arena_upload_async) {
        cudaStreamWaitEvent(0, c->ev_upload[slot], 0);
    }
    return rc_score_launch(c, c->arena_prog[slot], c->arena_off[slot], 0, 0,
                           nprog, kind, pcount, out, status);
}

/* #287: score the fixed-slot programs already uploaded to `which`. */
float rc_pool_score(void *ctx, int which, int nprog, int width, int kind,
                    const int *pcount, double *out, int *status)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || !c->d_pool[which & 1] || nprog <= 0 || !out || !status)
        return -1.0f;
    return rc_score_launch(c, c->d_pool[which & 1], 0, width, 1, nprog, kind,
                           pcount, out, status);
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

/* --- #234: the label sign the log-loss arm of the batched fit reads -------
 *
 * `sign[row]` is `t = 2l - 1` with `l = [y > (min y + max y)/2]`, the same rule
 * `logistic_irls` uses. It is uploaded once, next to the f64 fit rows, because
 * a one-block-per-candidate kernel cannot compute the midpoint over the sample
 * by itself. Passing `NULL` clears it, and every log-loss launch is then
 * refused.
 */
int rc_set_fit_sign(void *ctx, const double *sign)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return -1;
    if (c->d_tsign) { cudaFree(c->d_tsign); c->d_tsign = 0; }
    if (c->d_tsign64) { cudaFree(c->d_tsign64); c->d_tsign64 = 0; }
    c->has_tsign = 0;
    if (!sign) return 0;
    size_t yb = sizeof(double) * (size_t)c->n;
    size_t yb32 = sizeof(float) * (size_t)c->n;
    float *host = (float *)malloc(yb32);
    if (!host) return -1;
    for (int i = 0; i < c->n; ++i) host[i] = (float)sign[i];
    int ok = 1;
    if (rc_alloc((void **)&c->d_tsign64, yb) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_tsign, yb32) != 0) ok = 0;
    if (ok) {
        rc_upload(c->d_tsign64, sign, yb);
        rc_upload(c->d_tsign, host, yb32);
        if (cudaGetLastError() != cudaSuccess) ok = 0;
    }
    free(host);
    if (!ok) {
        if (c->d_tsign) { cudaFree(c->d_tsign); c->d_tsign = 0; }
        if (c->d_tsign64) { cudaFree(c->d_tsign64); c->d_tsign64 = 0; }
        return -1;
    }
    c->has_tsign = 1;
    return 0;
}

/* --- #169: batched LM fit entry point -------------------------------------
 *
 * `metric` is one of the `RC_LOSS_*` codes: 0 squared, 1 log-squashed,
 * 4 absolute, 5 the logistic cross-entropy (#234, which also needs the sign
 * uploaded by `rc_set_fit_sign`). 2 and 3 are the fit-only correlation codes,
 * which run their own kernel.
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
    /* #234: a log-loss batch without the uploaded label sign is refused here,
       so the arm can never run on another objective. */
    if (metric == RC_LOSS_LOGISTIC && !c->has_tsign) return -1.0f;
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

/* --- #226/#232: host side of the exact order-statistic stage --------------- */

/* `|r|` as a `u64`. The absolute residual is non-negative, so its IEEE-754
 * bit pattern orders exactly like the value; sorting the patterns is sorting
 * the residuals. `-0.0` and `+0.0` both map to `+0.0` bits, which is what the
 * CPU's `f64::abs` produces too. */
__device__ __forceinline__ unsigned long long md_abs_key(double r)
{
    double a = fabs(r);
    unsigned long long k;
    memcpy(&k, &a, sizeof(k));
    return k;
}

__device__ __forceinline__ double md_key_value(unsigned long long k)
{
    double a;
    memcpy(&a, &k, sizeof(a));
    return a;
}

/* One key per row of every `(candidate, trial)` segment: the bit pattern of
 * `|pred − y|` in exact f64. `status[seg] = 2` marks a segment with a
 * non-finite prediction or residual, which the caller reads as `non_finite`
 * and hands to the exact CPU arm (the landed #169 guard, never a silent 1e30). */
__global__ void k_order_stat_keys(const unsigned int *__restrict__ prog,
                              const unsigned int *__restrict__ off,
                              const double *__restrict__ c0, int kmax,
                              const double *__restrict__ cols,
                              const double *__restrict__ y, int n, int ns,
                              int ntrials,
                              unsigned long long *__restrict__ keys,
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
        const double r = pred - y[row];
        if (!isfinite(pred) || fabs(pred) > 1e30 || !isfinite(r)) {
            keys[i] = 0xFFFFFFFFFFFFFFFFULL;
            status[seg] = 2; /* non-finite row: the caller falls back */
        } else {
            keys[i] = md_abs_key(r);
        }
    }
}

/* Threads per segment in the order-statistic pick. Fixed on purpose: the
 * interquartile window sum is a tree reduction, and a fixed width is what makes
 * that reduction's rounding order (hence its value) reproducible. */
#define RC_OS_PICK_THREADS 256

/* The rank window `[lo, hi)` that `metrics::interquartile_mean_abs` averages:
 * `lo = n/4`, `hi = (3n)/4` for `n >= 4`. Below four rows the CPU arm has no
 * interquartile window and scores the plain mean of **all** rows, so the window
 * is `[0, n)` there. */
__device__ __forceinline__ void md_iqr_window(int n, int *lo, int *hi)
{
    if (n < 4) {
        *lo = 0;
        *hi = n;
        return;
    }
    *lo = n / 4;
    *hi = (3 * n) / 4;
}

/* Turn one sorted segment into its order statistic. `what` 0 is the median,
 * `what` 1 is the interquartile mean of `|r|`.
 *
 * Median. Odd `n`: the middle value. Even `n`: `0.5 * (a + b)` of the two
 * middle values, the same expression `metrics::median_abs` evaluates.
 *
 * Interquartile. The mean of the sorted keys at ranks `[lo, hi)` where
 * `lo = floor(n/4)` and `hi = floor(3n/4)` (all rows when `n < 4`), exactly the
 * slice `metrics::interquartile_mean_abs` averages. Membership is decided by
 * *rank*, never by a value interval: with a repeated boundary value the two
 * differ (sorted `1,2,3,3` with `n = 4` averages `2,3` in the window but
 * `2,3,3` over the value range). The sum is a fixed 256-thread tree reduction
 * over the window in block order, so the value is deterministic for a fixed
 * `n`; it agrees with the CPU's left-to-right `f64` sum to rounding, not to the
 * last bit. (The median is a single value and does agree bit for bit.)
 *
 * Ties need no tie-break in either case: both statistics are functions of the
 * multiset `{|r_i|}`, so equal keys cannot change the value.
 */
__global__ void k_order_stat_pick(const unsigned long long *__restrict__ keys,
                                  int n, int ns, int what,
                                  double *__restrict__ out,
                                  int *__restrict__ status)
{
    __shared__ double part[RC_OS_PICK_THREADS];
    const int seg = blockIdx.x;
    const int tid = threadIdx.x;
    if (seg >= ns) return;
    if (status[seg] != 0 || n <= 0) {
        if (tid == 0) {
            if (n <= 0) status[seg] = 2;
            out[seg] = CUDART_NAN;
        }
        return;
    }
    const unsigned long long *k = keys + (size_t)seg * (size_t)n;
    if (what == 0) {
        if (tid != 0) return;
        if (n & 1) {
            out[seg] = md_key_value(k[n / 2]);
        } else {
            out[seg] = 0.5 * (md_key_value(k[n / 2 - 1]) + md_key_value(k[n / 2]));
        }
        return;
    }
    int lo, hi;
    md_iqr_window(n, &lo, &hi);
    double s = 0.0;
    for (int i = lo + tid; i < hi; i += RC_OS_PICK_THREADS) s += md_key_value(k[i]);
    part[tid] = s;
    __syncthreads();
    for (int stride = RC_OS_PICK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) part[tid] += part[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[seg] = part[0] / (double)(hi - lo);
}

/* #227: the exact maximum-error scan, one block per `(candidate, trial)`
 * segment.
 *
 * Every thread strides the resident exact rows and the block reduces
 *
 *   - `max |pred − y|` in f64. A maximum is order-free — reordering the rows
 *     cannot change the value — and `|r| >= 0`, so its IEEE-754 `u64` pattern
 *     orders exactly like the value and the reduction is an **integer** max.
 *     There is no float summation anywhere, so the statistic carries no
 *     rounding and no run-to-run variation;
 *   - the **lowest** row index that attains it, as a min over ties, so the
 *     active set and any tie-break built on it are reproducible;
 *   - when `coords[seg]` is a constant slot and `levels[seg]` is a finite
 *     level `t`, the intersection of the per-row feasibility intervals of the
 *     affine coordinate model `r_i(delta) = a_i + b_i·delta`:
 *
 *         −t <= a_i + b_i·delta <= t   =>   delta in [min(e1,e2), max(e1,e2)]
 *
 *     with `a_i = r_i` and `b_i = ∂r_i/∂c` from the exact adjoint sweep. The
 *     intersection is `[max lo_i, min hi_i]` and it is non-empty exactly when
 *     `t` is attainable along that coordinate. A row with `b_i == 0` can only
 *     veto the level (`|a_i| > t`), it constrains no `delta`. That is the
 *     feasibility test the host-side bisection on the error level drives.
 *
 * No atomics: every reduction is a fixed-structure shared-memory tree of
 * order-free operators (integer max, min, max/min on f64, bitwise or), so the
 * result cannot depend on how the batch is composed or scheduled. */
#define FT_MM_THREADS 256

struct RcMaxOut {
    double vmax;  /* exact max |r|, NaN when status != 0 */
    double lo;    /* max of the per-row lower bounds (`t` requested) */
    double hi;    /* min of the per-row upper bounds (`t` requested) */
    int row;      /* lowest-index maximiser, -1 when status != 0 */
    int status;   /* 0 ok; 2 a row was not finite */
};

__global__ void k_max_scan(const unsigned int *__restrict__ prog,
                           const unsigned int *__restrict__ off,
                           const int *__restrict__ pcount,
                           const double *__restrict__ c0, int kmax,
                           const int *__restrict__ coords,
                           const double *__restrict__ levels,
                           const double *__restrict__ cols,
                           const double *__restrict__ y, int n, int ns,
                           int ntrials,
                           RcMaxOut *__restrict__ out)
{
    __shared__ unsigned long long skey[FT_MM_THREADS];
    __shared__ int srow[FT_MM_THREADS];
    __shared__ double slo[FT_MM_THREADS];
    __shared__ double shi[FT_MM_THREADS];
    __shared__ int sbad[FT_MM_THREADS];
    const int seg = blockIdx.x;
    if (seg >= ns) return;
    const int tid = threadIdx.x;
    /* `off` has one entry per *candidate*; a candidate owns `ntrials`
       consecutive segments (its trials, or its coordinates). */
    const int cand = seg / ntrials;
    const unsigned int *nd = prog + off[cand];
    const int m = (int)((off[cand + 1] - off[cand]) / 4u);
    const int p = pcount[cand];
    const double *c = c0 + (size_t)seg * (size_t)kmax;
    const int coord = (coords && p > 0 && coords[seg] >= 0 && coords[seg] < p)
                          ? coords[seg]
                          : -1;
    const double level = levels ? levels[seg] : CUDART_NAN;
    const int want_level = (coord >= 0) && isfinite(level) && level >= 0.0;

    unsigned long long mykey = 0ULL;
    int myrow = 0x7FFFFFFF;
    double mylo = -CUDART_INF;
    double myhi = CUDART_INF;
    int mybad = 0;    /* bit 0: non-finite row; bit 1: level infeasible */
    for (int row = tid; row < n; row += blockDim.x) {
        double v[FT_MAX_NODES];
        ft_forward_d(nd, m, cols, n, row, c, v);
        const double pred = v[m - 1];
        const double r = pred - y[row];
        if (!isfinite(pred) || fabs(pred) > 1e30 || !isfinite(r)) {
            mybad |= 1;
            break;
        }
        const unsigned long long key = md_abs_key(r);
        if (key > mykey || (key == mykey && row < myrow)) {
            mykey = key;
            myrow = row;
        }
        if (want_level && (mybad & 2) == 0) {
            double adj[FT_MAX_NODES];
            double g[FT_MAX_CONST_LARGE];
            ft_backward_d<FT_MAX_CONST_LARGE>(nd, m, v, adj, g, p);
            const double b = g[coord];
            if (!isfinite(b)) {
                mybad |= 1;
                break;
            }
            if (b == 0.0) {
                /* A row the coordinate cannot move: it only vetoes. */
                if (fabs(r) > level) mybad |= 2;
            } else {
                const double e1 = (-level - r) / b;
                const double e2 = (level - r) / b;
                const double lo_i = fmin(e1, e2);
                const double hi_i = fmax(e1, e2);
                if (lo_i > mylo) mylo = lo_i;
                if (hi_i < myhi) myhi = hi_i;
            }
        }
    }
    skey[tid] = mykey;
    srow[tid] = myrow;
    slo[tid] = mylo;
    shi[tid] = myhi;
    sbad[tid] = mybad;
    __syncthreads();
    for (int st = FT_MM_THREADS / 2; st > 0; st >>= 1) {
        if (tid < st) {
            const int j = tid + st;
            if (skey[j] > skey[tid] || (skey[j] == skey[tid] && srow[j] < srow[tid])) {
                skey[tid] = skey[j];
                srow[tid] = srow[j];
            }
            if (slo[j] > slo[tid]) slo[tid] = slo[j];
            if (shi[j] < shi[tid]) shi[tid] = shi[j];
            sbad[tid] |= sbad[j];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int bad = sbad[0];
        RcMaxOut o;
        o.status = (bad & 1) ? 2 : 0;
        o.vmax = (bad & 1) ? CUDART_NAN : md_key_value(skey[0]);
        o.row = (bad & 1) ? -1 : srow[0];
        o.lo = (bad & 2) ? CUDART_INF : slo[0];
        o.hi = (bad & 2) ? -CUDART_INF : shi[0];
        out[seg] = o;
    }
}

/* #227: make sure the f64 twins of the resident rows exist. Same lazy cast the
 * median arm does, factored out because the maximum arm calls the scan dozens
 * of times per candidate. */
static int rc_mm_f64(RcCtx *c)
{
    if (c->d_cols64 && c->d_y64) return 0;
    if (!c->d_cols64) {
        if (rc_alloc((void **)&c->d_cols64, sizeof(double) * (size_t)c->n * (size_t)c->v) != 0) {
            c->d_cols64 = 0;
            return -1;
        }
        int threads = 256;
        long long total = (long long)c->n * (long long)c->v;
        long long blocks = (total + threads - 1) / threads;
        cast_f32_to_f64_kernel<<<(int)blocks, threads>>>(c->d_cols, c->d_cols64, total);
    }
    if (!c->d_y64) {
        if (rc_alloc((void **)&c->d_y64, sizeof(double) * (size_t)c->n) != 0) {
            c->d_y64 = 0;
            return -1;
        }
        int threads = 256;
        long long yblocks = (c->n + threads - 1) / threads;
        cast_f32_to_f64_kernel<<<(int)yblocks, threads>>>(c->d_y, c->d_y64, c->n);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Reserve `bytes` from the reusable arena, aligned to 256 bytes. */
static int rc_mm_reserve(RcCtx *c, size_t bytes)
{
    if (bytes <= c->mm_bytes) return 0;
    if (c->d_mm) {
        cudaFree(c->d_mm);
        c->d_mm = 0;
        c->mm_bytes = 0;
    }
    size_t want = bytes + (bytes >> 1) + 4096;
    if (rc_alloc(&c->d_mm, want) != 0) {
        c->d_mm = 0;
        return -1;
    }
    c->mm_bytes = want;
    return 0;
}

/* Exact maximum-error scan of `nsets * ntrials` prediction vectors.
 *
 * Constants are `c0[(seg * kmax) + slot]` with `seg = set * ntrials + trial`,
 * the layout `rc_fit_batch`/`rc_rank_score_batch`/`rc_order_stat_score_batch` use.
 * `coords` and `levels` hold one entry per segment: `coords[seg] < 0` (or a
 * level that is not finite) asks for the maximum only, otherwise the segment
 * also returns the interval intersection at that level.
 *
 * Returns the stage time in ms, or a negative value when nothing may be
 * trusted (the caller then uses the exact CPU arm). */
float rc_max_scan_batch(void *ctx,
                        const unsigned int *prog,
                        const unsigned int *off,
                        const int *pcount,
                        int nsets,
                        int ntrials,
                        int kmax,
                        const double *c0,
                        const int *coords,
                        const double *levels,
                        RcMaxOut *out)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nsets <= 0 || ntrials <= 0 || kmax <= 0 || kmax > FT_MAX_CONST_LARGE) return -1.0f;
    if (c->n <= 0) return -1.0f;
    if (rc_mm_f64(c) != 0) return -1.0f;
    const int nsegs = nsets * ntrials;
    if ((long long)nsegs * (long long)c->n > 0x7FFFFFFFLL) return -1.0f;
    const size_t prog_words = (size_t)off[nsets];
    const size_t align = 256;
    const size_t sz_prog = ((sizeof(unsigned int) * prog_words + align - 1) / align) * align;
    const size_t sz_off = ((sizeof(unsigned int) * (size_t)(nsets + 1) + align - 1) / align) * align;
    const size_t sz_p = ((sizeof(int) * (size_t)nsets + align - 1) / align) * align;
    const size_t sz_c0 =
        ((sizeof(double) * (size_t)nsegs * (size_t)kmax + align - 1) / align) * align;
    const size_t sz_coords = ((sizeof(int) * (size_t)nsegs + align - 1) / align) * align;
    const size_t sz_levels = ((sizeof(double) * (size_t)nsegs + align - 1) / align) * align;
    const size_t sz_out = ((sizeof(RcMaxOut) * (size_t)nsegs + align - 1) / align) * align;
    const size_t total_bytes = sz_prog + sz_off + sz_p + sz_c0 + sz_coords + sz_levels + sz_out;
    if (rc_mm_reserve(c, total_bytes) != 0) return -1.0f;
    char *base = (char *)c->d_mm;
    unsigned int *d_prog = (unsigned int *)base;
    unsigned int *d_off = (unsigned int *)(base + sz_prog);
    int *d_p = (int *)(base + sz_prog + sz_off);
    double *d_c0 = (double *)(base + sz_prog + sz_off + sz_p);
    int *d_coords = (int *)(base + sz_prog + sz_off + sz_p + sz_c0);
    double *d_levels = (double *)(base + sz_prog + sz_off + sz_p + sz_c0 + sz_coords);
    RcMaxOut *d_out = (RcMaxOut *)(base + sz_prog + sz_off + sz_p + sz_c0 + sz_coords + sz_levels);

    /* One host->device copy for the whole request. The bisection calls this
       once per round, so six separately synchronised `cudaMemcpy` calls per
       round would be most of its latency on a small batch. The host block uses
       the same aligned layout as the arena. A null `coords`/`levels` means
       "maximum only": `-1` / NaN for every segment, filled here on the host. */
    const size_t in_bytes = sz_prog + sz_off + sz_p + sz_c0 + sz_coords + sz_levels;
    char *host = (char *)calloc(1, in_bytes);
    if (!host) return -1.0f;
    memcpy(host, prog, sizeof(unsigned int) * prog_words);
    memcpy(host + sz_prog, off, sizeof(unsigned int) * (size_t)(nsets + 1));
    memcpy(host + sz_prog + sz_off, pcount, sizeof(int) * (size_t)nsets);
    memcpy(host + sz_prog + sz_off + sz_p, c0, sizeof(double) * (size_t)nsegs * (size_t)kmax);
    int *h_coords = (int *)(host + sz_prog + sz_off + sz_p + sz_c0);
    double *h_levels = (double *)(host + sz_prog + sz_off + sz_p + sz_c0 + sz_coords);
    for (int seg = 0; seg < nsegs; ++seg) {
        h_coords[seg] = coords ? coords[seg] : -1;
        if (levels && coords) {
            h_levels[seg] = levels[seg];
        } else {
            memcpy(&h_levels[seg], "\xff\xff\xff\xff\xff\xff\xff\xff", sizeof(double));
        }
    }
    rc_upload(base, host, in_bytes);
    free(host);
    float ms = -1.0f;
    int ok = cudaGetLastError() == cudaSuccess;
    if (ok) {
        cudaEventRecord(c->t0, 0);
        k_max_scan<<<nsegs, FT_MM_THREADS>>>(d_prog, d_off, d_p, d_c0, kmax, d_coords,
                                             d_levels, c->d_cols64, c->d_y64, c->n, nsegs,
                                             ntrials, d_out);
        ok = cudaGetLastError() == cudaSuccess;
        cudaEventRecord(c->t1, 0);
        cudaEventSynchronize(c->t1);
        if (ok && cudaGetLastError() != cudaSuccess) ok = 0;
    }
    if (ok) {
        cudaEventElapsedTime(&ms, c->t0, c->t1);
        rc_download(out, d_out, sizeof(RcMaxOut) * (size_t)nsegs);
        if (cudaGetLastError() != cudaSuccess) ok = 0;
    }
    if (!ok) return -1.0f;
    return ms;
}

/* Exact median (what = 0) or interquartile mean of the absolute errors
 * (what = 1) of `nsets * ntrials` prediction vectors.
 *
 * Constants are `c0[(seg * kmax) + slot]` with `seg = set * ntrials + trial`,
 * the same layout `rc_fit_batch`/`rc_rank_score_batch` use. `out_value`
 * receives the exact `f64` statistic per segment and `out_status` the labelled
 * status (0 ok, 2 non-finite). Returns the stage time in ms, or a negative
 * value when nothing may be trusted (the caller then uses the exact CPU arm).
 *
 * The keys are predictably cheap to sort: one prediction sweep, then CUB's
 * segmented radix sort of the `u64` patterns (four byte passes over keys
 * already in memory), then one pick. No atomics, no float atomics and no
 * data-dependent reduction order, so the value is bit-identical however the
 * batch is composed (the interquartile mean's own tree reduction is fixed by
 * the constant window and the constant thread count).
 */
float rc_order_stat_score_batch(void *ctx,
                                const unsigned int *prog,
                                const unsigned int *off,
                                const int *pcount,
                                int nsets,
                                int ntrials,
                                int kmax,
                                const double *c0,
                                int what,
                                double *out_value,
                                int *out_status)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c || nsets <= 0 || ntrials <= 0 || kmax <= 0) return -1.0f;
    if (what != 0 && what != 1) return -1.0f;
    if (c->n <= 0) return -1.0f;
    if (!c->d_cols64 || !c->d_y64) {
        /* The f64 twins are normally uploaded once by `rc_set_fit_data`; a
           direct caller without them gets the f32-rounded inputs, exactly like
           `rc_fit_batch`'s own fallback. */
        size_t cb = sizeof(double) * (size_t)c->n * (size_t)c->v;
        size_t yb = sizeof(double) * (size_t)c->n;
        if (!c->d_cols64) {
            if (rc_alloc((void **)&c->d_cols64, cb) != 0) {
                c->d_cols64 = 0;
                return -1.0f;
            }
            int threads = 256;
            long long total = (long long)c->n * (long long)c->v;
            long long blocks = (total + threads - 1) / threads;
            cast_f32_to_f64_kernel<<<(int)blocks, threads>>>(c->d_cols, c->d_cols64, total);
        }
        if (!c->d_y64) {
            if (rc_alloc((void **)&c->d_y64, yb) != 0) {
                c->d_y64 = 0;
                return -1.0f;
            }
            int threads = 256;
            long long yblocks = (c->n + threads - 1) / threads;
            cast_f32_to_f64_kernel<<<(int)yblocks, threads>>>(c->d_y, c->d_y64, c->n);
        }
        if (cudaGetLastError() != cudaSuccess) return -1.0f;
    }
    const int nsegs = nsets * ntrials;
    const long long total = (long long)nsegs * (long long)c->n;
    if (total <= 0 || total > 0x7FFFFFFFLL) return -1.0f;
    unsigned int *d_prog = 0, *d_off = 0;
    int *d_p = 0;
    double *d_c0 = 0;
    unsigned long long *d_keys = 0;
    unsigned long long *d_keys_out = 0;
    double *d_val = 0;
    int *d_status = 0;
    int *d_seg_off = 0;
    void *d_temp = 0;
    size_t temp_bytes = 0;
    int ok = 1;
    const size_t prog_words = (size_t)off[nsets];
    if (rc_alloc((void **)&d_prog, sizeof(unsigned int) * prog_words) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_off, sizeof(unsigned int) * (size_t)(nsets + 1)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_p, sizeof(int) * (size_t)nsets) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_c0, sizeof(double) * (size_t)nsegs * (size_t)kmax) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_keys, sizeof(unsigned long long) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_keys_out, sizeof(unsigned long long) * (size_t)total) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_val, sizeof(double) * (size_t)nsegs) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_status, sizeof(int) * (size_t)nsegs) != 0) ok = 0;
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
        /* Size the CUB temporary storage once per call (it depends on the
           segment count and length, not on the values). */
        cub::DeviceSegmentedRadixSort::SortKeys(
            NULL, temp_bytes, d_keys, d_keys_out, (int)total, nsegs, d_seg_off,
            d_seg_off + 1);
        if (temp_bytes == 0) temp_bytes = 1;
        if (rc_alloc(&d_temp, temp_bytes) != 0) ok = 0;
    }
    float ms = -1.0f;
    if (ok) {
        int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        if (blocks < 1) blocks = 1;
        cudaEventRecord(c->t0, 0);
        k_order_stat_keys<<<(int)blocks, threads>>>(d_prog, d_off, d_c0, kmax, c->d_cols64,
                                                c->d_y64, c->n, nsegs, ntrials, d_keys,
                                                d_status);
        ok = cudaGetLastError() == cudaSuccess;
        if (ok) {
            cub::DeviceSegmentedRadixSort::SortKeys(
                d_temp, temp_bytes, d_keys, d_keys_out, (int)total, nsegs, d_seg_off,
                d_seg_off + 1);
            ok = cudaGetLastError() == cudaSuccess;
        }
        if (ok) {
            k_order_stat_pick<<<nsegs, RC_OS_PICK_THREADS>>>(d_keys_out, c->n, nsegs, what,
                                                             d_val, d_status);
            ok = cudaGetLastError() == cudaSuccess;
        }
        cudaEventRecord(c->t1, 0);
        cudaEventSynchronize(c->t1);
        if (ok && cudaGetLastError() != cudaSuccess) ok = 0;
        if (ok) {
            cudaEventElapsedTime(&ms, c->t0, c->t1);
            rc_download(out_value, d_val, sizeof(double) * (size_t)nsegs);
            rc_download(out_status, d_status, sizeof(int) * (size_t)nsegs);
            if (cudaGetLastError() != cudaSuccess) ok = 0;
        }
    }
    if (d_prog) cudaFree(d_prog);
    if (d_off) cudaFree(d_off);
    if (d_p) cudaFree(d_p);
    if (d_c0) cudaFree(d_c0);
    if (d_keys) cudaFree(d_keys);
    if (d_keys_out) cudaFree(d_keys_out);
    if (d_val) cudaFree(d_val);
    if (d_status) cudaFree(d_status);
    if (d_seg_off) cudaFree(d_seg_off);
    if (d_temp) cudaFree(d_temp);
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
