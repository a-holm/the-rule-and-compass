/*
 * #304: fused long-only {0, +1} P&L reduction.
 *
 * One block per candidate. Within a block, thread `t` owns every canonical
 * segment `s` with `s % blockDim.x == t`. A segment is always reduced by one
 * thread in ascending row order, so the bit pattern does not depend on the
 * thread count: 128, 256 and 512 threads only move ownership, not order.
 *
 * Chunk carry. `rc_pnl_eval` may launch several times over disjoint row ranges.
 * The per-(candidate, segment) partial lives in device memory. A launch that
 * starts inside a segment resumes that partial and keeps adding rows in order;
 * a launch that starts on a segment boundary starts the segment from zero. This
 * is what makes alone/padded/4k/64k chunking bit-identical.
 *
 * The kernel never allocates P x N. The host uploads a declared tile of at most
 * `tile_candidates` scores; the population is streamed through that tile.
 */

#include <cuda_runtime.h>
#include "alloc_guard.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PNL_LANES 32

typedef struct RcPnlTotals {
    double gross;
    double cost;
    double turnover;
    double net;
    unsigned long long rows;
    unsigned long long long_rows;
    unsigned long long entries;
    unsigned long long exits;
    unsigned long long trades;
    unsigned long long hits;
    unsigned long long final_exits;
    int status; /* 0 ok, 2 non-finite return */
} RcPnlTotals;

typedef struct RcPnlCtx {
    int n;
    int segment_rows;
    int nsegments;
    int tile_candidates;
    double *d_returns;
    unsigned long long *d_groups;
    float *d_scores;      /* tile_candidates x n, candidate major */
    RcPnlTotals *d_partials; /* tile_candidates x nsegments */
} RcPnlCtx;

static int rc_pnl_zero(RcPnlTotals *t)
{
    memset(t, 0, sizeof(*t));
    return 0;
}

static void rc_pnl_merge(RcPnlTotals *dst, const RcPnlTotals *src)
{
    dst->gross += src->gross;
    dst->cost += src->cost;
    dst->turnover += src->turnover;
    dst->net += src->net;
    dst->rows += src->rows;
    dst->long_rows += src->long_rows;
    dst->entries += src->entries;
    dst->exits += src->exits;
    dst->trades += src->trades;
    dst->hits += src->hits;
    dst->final_exits += src->final_exits;
    if (src->status != 0) dst->status = src->status;
}

/* The fixed 32 -> 1 lane tree. The host reference in trading_gpu.rs uses the
 * same pairing, so a segment's bits are identical on CPU and device. */
static void rc_pnl_merge_lanes(RcPnlTotals *out, RcPnlTotals *lanes)
{
    int width = PNL_LANES;
    while (width > 1) {
        int half = width / 2;
        for (int i = 0; i < half; ++i) {
            rc_pnl_merge(&lanes[i], &lanes[i + half]);
        }
        width = half;
    }
    *out = lanes[0];
}

__device__ __forceinline__ double pnl_position(float score, double threshold)
{
    return (isfinite((double)score) && (double)score > threshold) ? 1.0 : 0.0;
}

__global__ void pnl_kernel(const double *__restrict__ returns,
                           const unsigned long long *__restrict__ groups,
                           const float *__restrict__ scores,
                           const double *__restrict__ thresholds,
                           double side,
                           int n,
                           int segment_rows,
                           int nsegments,
                           long long row_lo,
                           long long row_hi,
                           RcPnlTotals *__restrict__ partials)
{
    const int candidate = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (PNL_LANES - 1);
    const int warp = tid / PNL_LANES;
    const int warps = blockDim.x / PNL_LANES;
    const double threshold = thresholds[candidate];
    const float *zc = scores + (size_t)candidate * (size_t)n;
    RcPnlTotals *pc = partials +
        (size_t)candidate * (size_t)nsegments * (size_t)PNL_LANES;

    for (int s = warp; s < nsegments; s += warps) {
        long long seg_lo = (long long)s * segment_rows;
        long long seg_hi = seg_lo + segment_rows;
        if (seg_hi > n) seg_hi = n;
        long long lo = row_lo > seg_lo ? row_lo : seg_lo;
        long long hi = row_hi < seg_hi ? row_hi : seg_hi;
        if (lo >= hi) continue;

        RcPnlTotals *lane_slot = pc + (size_t)s * (size_t)PNL_LANES;
        RcPnlTotals acc = {};
        if (lo > seg_lo) {
            acc = lane_slot[lane];
        }

        /* Lane `L` owns rows with (row - seg_lo) % 32 == L. The first such row
         * at or after `lo`, so a mid-segment chunk resumes without replaying
         * earlier rows. */
        long long first = seg_lo + lane;
        if (first < lo) {
            first += ((lo - first + PNL_LANES - 1) / PNL_LANES) * PNL_LANES;
        }

        for (long long i = first; i < hi; i += PNL_LANES) {
            double r = returns[i];
            if (!isfinite(r)) {
                acc.status = 2;
                continue;
            }
            double p = pnl_position(zc[i], threshold);
            double prev = 0.0;
            if (i > 0 && groups[i] == groups[i - 1]) {
                prev = pnl_position(zc[i - 1], threshold);
            }
            double d = p - prev;
            int last = (i + 1 == n) || (groups[i + 1] != groups[i]);
            double gross = p * r;
            double cost = side * fabs(d);
            if (last && p != 0.0) {
                cost += side * fabs(p);
                acc.final_exits += 1;
                acc.exits += 1;
            }
            acc.rows += 1;
            if (p != 0.0) {
                acc.long_rows += 1;
                if (r > 0.0) acc.hits += 1;
            }
            if (d > 0.0) acc.entries += 1;
            else if (d < 0.0) acc.exits += 1;
            if (d != 0.0) acc.trades += 1;
            acc.gross += gross;
            acc.cost += cost;
            acc.turnover += fabs(d);
            acc.net += gross - cost;
        }
        lane_slot[lane] = acc;
    }
}

extern "C" {

void *rc_pnl_create(const double *returns,
                    const unsigned long long *groups,
                    int n,
                    int segment_rows,
                    int tile_candidates)
{
    if (returns == NULL || groups == NULL || n <= 0 ||
        segment_rows <= 0 || tile_candidates <= 0) {
        return NULL;
    }
    int nsegments = (n + segment_rows - 1) / segment_rows;
    RcPnlCtx *ctx = (RcPnlCtx *)calloc(1, sizeof(RcPnlCtx));
    if (ctx == NULL) return NULL;
    ctx->n = n;
    ctx->segment_rows = segment_rows;
    ctx->nsegments = nsegments;
    ctx->tile_candidates = tile_candidates;

    size_t ret_bytes = (size_t)n * sizeof(double);
    size_t group_bytes = (size_t)n * sizeof(unsigned long long);
    size_t score_bytes = (size_t)n * (size_t)tile_candidates * sizeof(float);
    size_t part_bytes = (size_t)nsegments * (size_t)tile_candidates *
                        (size_t)PNL_LANES * sizeof(RcPnlTotals);

    if (rc_checked_malloc((void **)&ctx->d_returns, ret_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_groups, group_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_scores, score_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_partials, part_bytes) != cudaSuccess) goto fail;

    if (cudaMemcpy(ctx->d_returns, returns, ret_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(ctx->d_groups, groups, group_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    return ctx;

fail:
    if (ctx->d_returns) cudaFree(ctx->d_returns);
    if (ctx->d_groups) cudaFree(ctx->d_groups);
    if (ctx->d_scores) cudaFree(ctx->d_scores);
    if (ctx->d_partials) cudaFree(ctx->d_partials);
    free(ctx);
    return NULL;
}

void rc_pnl_destroy(void *opaque)
{
    RcPnlCtx *ctx = (RcPnlCtx *)opaque;
    if (ctx == NULL) return;
    if (ctx->d_returns) cudaFree(ctx->d_returns);
    if (ctx->d_groups) cudaFree(ctx->d_groups);
    if (ctx->d_scores) cudaFree(ctx->d_scores);
    if (ctx->d_partials) cudaFree(ctx->d_partials);
    free(ctx);
}

int rc_pnl_tile_capacity(void *opaque)
{
    RcPnlCtx *ctx = (RcPnlCtx *)opaque;
    return ctx == NULL ? 0 : ctx->tile_candidates;
}

int rc_pnl_segment_rows(void *opaque)
{
    RcPnlCtx *ctx = (RcPnlCtx *)opaque;
    return ctx == NULL ? 0 : ctx->segment_rows;
}

int rc_pnl_eval(void *opaque,
                const float *scores,
                int candidates,
                const double *thresholds,
                double side,
                int threads,
                int chunk_rows,
                RcPnlTotals *out)
{
    RcPnlCtx *ctx = (RcPnlCtx *)opaque;
    if (ctx == NULL || scores == NULL || thresholds == NULL || out == NULL) return 1;
    if (candidates <= 0 || candidates > ctx->tile_candidates) return 1;
    if (threads <= 0 || threads > 1024 || (threads % PNL_LANES) != 0 ||
        chunk_rows <= 0) return 1;

    size_t score_bytes = (size_t)ctx->n * (size_t)candidates * sizeof(float);
    size_t part_bytes = (size_t)ctx->nsegments * (size_t)candidates *
                        (size_t)PNL_LANES * sizeof(RcPnlTotals);

    if (cudaMemcpy(ctx->d_scores, scores, score_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
    }
    if (cudaMemset(ctx->d_partials, 0, part_bytes) != cudaSuccess) return 3;

    for (long long lo = 0; lo < ctx->n; lo += chunk_rows) {
        long long hi = lo + chunk_rows;
        if (hi > ctx->n) hi = ctx->n;
        pnl_kernel<<<candidates, threads>>>(ctx->d_returns,
                                            ctx->d_groups,
                                            ctx->d_scores,
                                            thresholds,
                                            side,
                                            ctx->n,
                                            ctx->segment_rows,
                                            ctx->nsegments,
                                            lo,
                                            hi,
                                            ctx->d_partials);
        if (cudaGetLastError() != cudaSuccess) return 3;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;

    RcPnlTotals *host = (RcPnlTotals *)malloc(part_bytes);
    if (host == NULL) return 1;
    if (cudaMemcpy(host, ctx->d_partials, part_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(host);
        return 3;
    }

    for (int c = 0; c < candidates; ++c) {
        RcPnlTotals total;
        rc_pnl_zero(&total);
        const RcPnlTotals *pc = host +
            (size_t)c * (size_t)ctx->nsegments * (size_t)PNL_LANES;
        for (int s = 0; s < ctx->nsegments; ++s) {
            RcPnlTotals lanes[PNL_LANES];
            const RcPnlTotals *src = pc + (size_t)s * (size_t)PNL_LANES;
            for (int l = 0; l < PNL_LANES; ++l) lanes[l] = src[l];
            RcPnlTotals segment;
            rc_pnl_merge_lanes(&segment, lanes);
            rc_pnl_merge(&total, &segment);
        }
        out[c] = total;
    }
    free(host);
    return 0;
}

} /* extern "C" */

/* ------------------------------------------------------------------------- */
/* #305: exact type-7 threshold radix selection and shared valid-row mask.    */
/* ------------------------------------------------------------------------- */

/*
 * Exact type-7 threshold selection and the shared valid-row mask (#305).
 *
 * The threshold is a discrete input: a one-ulp difference flips every position
 * whose score sits on the boundary. Device selection is therefore exact, not
 * approximate:
 *
 *   1. map every finite f32 to an order-preserving u32 key (`-0` -> `+0`,
 *      NaN/+-inf -> the 0xffffffff sentinel);
 *   2. select the two ranks bracketing h = (n - 1) q with four 256-bin radix
 *      count passes per rank. Counts are integer atomics, so the result does
 *      not depend on thread count, block count or launch order;
 *   3. return the two order statistics as f64. The host interpolates in f64
 *      with the same expression as the CPU reference; that arithmetic is not
 *      re-associated here.
 *
 * The mask kernel sets bit i iff score[i] and target[i] are both finite, in the
 * same little-endian-within-u64 layout as Rust's `trading_gpu::ValidMask`.
 *
 * Both entry points take host slices, upload, run and download. They are the
 * parity/selection path for a small discrete input, not a per-candidate
 * streaming path; the CPU result remains the reported authority.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define RC_PNL_THREADS 256
#define RC_PNL_MAX_THREADS 512
#define RC_NONFINITE_KEY 0xffffffffu

__device__ __forceinline__ unsigned int rc_pnl_key(float v)
{
    if (!isfinite(v)) return RC_NONFINITE_KEY;
    unsigned int bits = (v == 0.0f) ? 0u : __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

/* Host-side inverse of the key transform. The device never needs it: it only
 * writes keys, and the two order statistics come back as f64. */
static float rc_pnl_value(unsigned int key)
{
    unsigned int bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Keys plus the exact count of finite rows. Integer atomics make both
 * deterministic. */
__global__ void rc_pnl_key_kernel(const float *__restrict__ scores,
                                  long long n,
                                  unsigned int *__restrict__ keys,
                                  unsigned long long *__restrict__ finite)
{
    __shared__ unsigned int count[RC_PNL_MAX_THREADS];
    const int tid = threadIdx.x;
    if (tid < RC_PNL_MAX_THREADS) count[tid] = 0u;
    for (long long i = (long long)blockIdx.x * blockDim.x + tid; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        unsigned int key = rc_pnl_key(scores[i]);
        keys[i] = key;
        if (key != RC_NONFINITE_KEY) count[tid] += 1u;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) count[tid] += count[tid + stride];
        __syncthreads();
    }
    if (tid == 0 && count[0] != 0u) {
        atomicAdd(finite, (unsigned long long)count[0]);
    }
}

/* One radix pass: 256-bin histogram of `keys` restricted to the prefix already
 * fixed by the host. */
__global__ void rc_pnl_hist_kernel(const unsigned int *__restrict__ keys,
                                   long long n,
                                   int shift,
                                   unsigned int keep_mask,
                                   unsigned int keep_val,
                                   unsigned int *__restrict__ hist)
{
    __shared__ unsigned int bins[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) bins[b] = 0u;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        unsigned int key = keys[i];
        if (key == RC_NONFINITE_KEY) continue;
        if ((key & keep_mask) != keep_val) continue;
        atomicAdd(&bins[(key >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        if (bins[b] != 0u) atomicAdd(&hist[b], bins[b]);
    }
}

/* Select one rank. The host advances `prefix` with the chosen byte; the next
 * pass only counts keys that agree on every byte fixed so far. */
static int rc_pnl_select_rank(const unsigned int *d_keys,
                              long long key_count,
                              long long finite_count,
                              long long rank,
                              int threads,
                              unsigned int *d_hist,
                              unsigned int *h_hist,
                              unsigned int *out_key)
{
    if (rank < 0 || rank >= finite_count || threads <= 0 ||
        threads > RC_PNL_MAX_THREADS)
        return 1;
    unsigned int prefix = 0u;
    const int blocks = (int)((key_count + threads - 1) / threads);
    for (int shift = 24; shift >= 0; shift -= 8) {
        unsigned int keep_mask =
            (shift >= 24) ? 0u : (0xffffffffu << (shift + 8));
        cudaError_t err = cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
        if (err != cudaSuccess) return 2;
        rc_pnl_hist_kernel<<<blocks, threads>>>(
            d_keys, key_count, shift, keep_mask, prefix, d_hist);
        err = cudaGetLastError();
        if (err != cudaSuccess) return 3;
        err = cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return 4;
        long long before = 0;
        int bucket = -1;
        for (int b = 0; b < 256; ++b) {
            if (rank < before + (long long)h_hist[b]) {
                bucket = b;
                break;
            }
            before += (long long)h_hist[b];
        }
        if (bucket < 0) return 5;
        rank -= before;
        prefix |= ((unsigned int)bucket) << shift;
    }
    *out_key = prefix;
    return 0;
}

extern "C" int rc_threshold_select_f32_threads(const float *scores,
                                               long long n,
                                               double q,
                                               int threads,
                                               long long *out_finite,
                                               double *out_lo,
                                               double *out_hi)
{
    if (n < 0 || out_finite == NULL || out_lo == NULL || out_hi == NULL ||
        threads <= 0 || threads > RC_PNL_MAX_THREADS)
        return 1;
    *out_finite = 0;
    *out_lo = NAN;
    *out_hi = NAN;
    if (n == 0) return 0;
    if (scores == NULL) return 2;

    float *d_scores = NULL;
    unsigned int *d_keys = NULL;
    unsigned int *d_hist = NULL;
    unsigned long long *d_finite = NULL;
    unsigned long long h_finite = 0;
    unsigned int *h_hist = NULL;
    int status = 0;

    cudaError_t err = rc_checked_malloc((void **)&d_scores, (size_t)n * sizeof(float));
    if (err != cudaSuccess) { status = 3; goto done; }
    err = rc_checked_malloc((void **)&d_keys, (size_t)n * sizeof(unsigned int));
    if (err != cudaSuccess) { status = 4; goto done; }
    err = rc_checked_malloc((void **)&d_hist, 256 * sizeof(unsigned int));
    if (err != cudaSuccess) { status = 5; goto done; }
    err = rc_checked_malloc((void **)&d_finite, sizeof(unsigned long long));
    if (err != cudaSuccess) { status = 6; goto done; }
    h_hist = (unsigned int *)malloc(256 * sizeof(unsigned int));
    if (h_hist == NULL) { status = 7; goto done; }

    err = cudaMemcpy(d_scores, scores, (size_t)n * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { status = 8; goto done; }
    err = cudaMemset(d_finite, 0, sizeof(unsigned long long));
    if (err != cudaSuccess) { status = 9; goto done; }

    {
        const int blocks = (int)((n + threads - 1) / threads);
        rc_pnl_key_kernel<<<blocks, threads>>>(d_scores, n, d_keys, d_finite);
        err = cudaGetLastError();
        if (err != cudaSuccess) { status = 10; goto done; }
    }
    err = cudaMemcpy(&h_finite, d_finite, sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { status = 11; goto done; }
    *out_finite = (long long)h_finite;
    if (h_finite == 0) { status = 0; goto done; }
    if (h_finite == 1) {
        unsigned int key = 0u;
        status = rc_pnl_select_rank(d_keys, n, (long long)h_finite, 0, threads,
                                    d_hist, h_hist, &key);
        if (status != 0) goto done;
        *out_lo = (double)rc_pnl_value(key);
        *out_hi = *out_lo;
        status = 0;
        goto done;
    }

    {
        double qq = q;
        if (qq < 0.0) qq = 0.0;
        if (qq > 1.0) qq = 1.0;
        double h = (double)(h_finite - 1u) * qq;
        long long lo = 0;
        long long hi = 0;
        if (!isnan(h)) {
            lo = (long long)floor(h);
            hi = (long long)ceil(h);
        }
        if (lo < 0) lo = 0;
        if (hi < 0) hi = 0;
        if (lo >= (long long)h_finite) lo = (long long)h_finite - 1;
        if (hi >= (long long)h_finite) hi = (long long)h_finite - 1;
        unsigned int lo_key = 0u, hi_key = 0u;
        status = rc_pnl_select_rank(d_keys, n, (long long)h_finite, lo, threads,
                                    d_hist, h_hist, &lo_key);
        if (status != 0) goto done;
        if (hi == lo) {
            hi_key = lo_key;
        } else {
            status = rc_pnl_select_rank(d_keys, n, (long long)h_finite, hi,
                                        threads, d_hist, h_hist, &hi_key);
            if (status != 0) goto done;
        }
        *out_lo = (double)rc_pnl_value(lo_key);
        *out_hi = (double)rc_pnl_value(hi_key);
        status = 0;
    }

done:
    if (h_hist != NULL) free(h_hist);
    if (d_finite != NULL) cudaFree(d_finite);
    if (d_hist != NULL) cudaFree(d_hist);
    if (d_keys != NULL) cudaFree(d_keys);
    if (d_scores != NULL) cudaFree(d_scores);
    if (status == 0) cudaGetLastError(); /* clear a stale error, if any */
    return status;
}

/* Default entry point: the canonical 256-thread launch. The `_threads` variant
 * exists so the thread-invariance contract can be tested on the device. */
extern "C" int rc_threshold_select_f32(const float *scores,
                                       long long n,
                                       double q,
                                       long long *out_finite,
                                       double *out_lo,
                                       double *out_hi)
{
    return rc_threshold_select_f32_threads(scores, n, q, RC_PNL_THREADS,
                                           out_finite, out_lo, out_hi);
}

__global__ void rc_pnl_mask_kernel(const float *__restrict__ scores,
                                   const float *__restrict__ targets,
                                   long long n,
                                   unsigned long long *__restrict__ words)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        if (isfinite(scores[i]) && isfinite(targets[i])) {
            atomicOr(&words[i >> 6], 1ull << (i & 63));
        }
    }
}

extern "C" int rc_valid_mask_f32(const float *scores,
                                 const float *targets,
                                 long long n,
                                 unsigned long long *words)
{
    if (n < 0 || (n > 0 && (scores == NULL || targets == NULL || words == NULL)))
        return 1;
    if (n == 0) return 0;

    const size_t words_bytes = (size_t)((n + 63) / 64) * sizeof(unsigned long long);
    float *d_scores = NULL;
    float *d_targets = NULL;
    unsigned long long *d_words = NULL;
    int status = 0;

    cudaError_t err = rc_checked_malloc((void **)&d_scores, (size_t)n * sizeof(float));
    if (err != cudaSuccess) { status = 2; goto done; }
    err = rc_checked_malloc((void **)&d_targets, (size_t)n * sizeof(float));
    if (err != cudaSuccess) { status = 3; goto done; }
    err = rc_checked_malloc((void **)&d_words, words_bytes);
    if (err != cudaSuccess) { status = 4; goto done; }
    err = cudaMemcpy(d_scores, scores, (size_t)n * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { status = 5; goto done; }
    err = cudaMemcpy(d_targets, targets, (size_t)n * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { status = 6; goto done; }
    err = cudaMemset(d_words, 0, words_bytes);
    if (err != cudaSuccess) { status = 7; goto done; }
    {
        const int blocks = (int)((n + RC_PNL_THREADS - 1) / RC_PNL_THREADS);
        rc_pnl_mask_kernel<<<blocks, RC_PNL_THREADS>>>(d_scores, d_targets, n,
                                                       d_words);
        err = cudaGetLastError();
        if (err != cudaSuccess) { status = 8; goto done; }
    }
    err = cudaMemcpy(words, d_words, words_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { status = 9; goto done; }

done:
    if (d_words != NULL) cudaFree(d_words);
    if (d_targets != NULL) cudaFree(d_targets);
    if (d_scores != NULL) cudaFree(d_scores);
    if (status == 0) cudaGetLastError();
    return status;
}

/* ------------------------------------------------------------------------- */
/* #306: band/cooldown and hindsight DP chain scans.                          */
/* ------------------------------------------------------------------------- */

/*
 * The CPU reference (`band_scan`, `cooldown_scan`, `dp_scan`) folds each chain
 * left to right: one chain per (candidate, group, start state) for the band,
 * one per (group, source state) for the DP. f64 addition is not associative,
 * so a parallel prefix over carries would round differently from that fold.
 * Each chain is therefore folded by exactly one thread in ascending row order;
 * the block and grid size only change which thread owns a chain.
 *
 * Chunk carry. The host launches once per [row_lo, row_hi) chunk. A chain
 * whose group starts inside the chunk starts from its initial state; any other
 * chain resumes its (state, value) from device memory. The fold order is the
 * same for every chunk size.
 *
 * Bit identity with the CPU also relies on `-fmad=false` (build.rs): every
 * product and sum below is rounded on its own, in the CPU's order.
 */

#include <limits.h>

#define RC_SCAN_MAX_BLOCKS 65535

__global__ void rc_band_chain_kernel(const double *__restrict__ scores,
                                     const double *__restrict__ returns,
                                     const long long *__restrict__ offsets,
                                     const double *__restrict__ thresholds,
                                     const double *__restrict__ bands,
                                     int groups,
                                     int states,
                                     long long n,
                                     long long chains,
                                     double side,
                                     long long row_lo,
                                     long long row_hi,
                                     int *__restrict__ chain_state,
                                     double *__restrict__ chain_reward,
                                     unsigned char *__restrict__ positions)
{
    const int hold = states - 1;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < chains;
         c += (long long)gridDim.x * blockDim.x) {
        const int start = (int)(c % states);
        const long long cg = c / states;
        const int g = (int)(cg % groups);
        const long long cand = cg / groups;
        const long long gs = offsets[g];
        const long long ge = offsets[g + 1];
        const long long lo = row_lo > gs ? row_lo : gs;
        const long long hi = row_hi < ge ? row_hi : ge;
        if (lo >= hi) continue;

        int state = start;
        double acc = 0.0;
        if (lo > gs) {
            state = chain_state[c];
            acc = chain_reward[c];
        }
        const double theta = thresholds[cand];
        const double beta = bands[cand];
        const double enter_above = theta + beta;
        const double exit_at = theta - beta;
        const double *zc = scores + cand * n;
        unsigned char *pc = positions + cand * n;
        for (long long i = lo; i < hi; ++i) {
            const double z = zc[i];
            int next;
            if (state == 0) {
                next = (isfinite(z) && z > enter_above) ? 1 : 0;
            } else if ((!isfinite(z) || z <= exit_at) && state >= hold) {
                next = 0;
            } else {
                next = state < hold ? state + 1 : hold;
            }
            const double p0 = state != 0 ? 1.0 : 0.0;
            const double p1 = next != 0 ? 1.0 : 0.0;
            double w = p1 * returns[i] - side * fabs(p1 - p0);
            if (i + 1 == ge && p1 != 0.0) w -= side;
            acc = acc + w;
            state = next;
            if (start == 0) pc[i] = (unsigned char)(next != 0);
        }
        chain_state[c] = state;
        chain_reward[c] = acc;
    }
}

/* `values` holds (V(flat), V(long)) per chain; chain c = 2 * group + source. */
__global__ void rc_dp_chain_kernel(const double *__restrict__ returns,
                                   const long long *__restrict__ offsets,
                                   int groups,
                                   double side,
                                   long long row_lo,
                                   long long row_hi,
                                   double *__restrict__ values)
{
    const long long chains = 2LL * (long long)groups;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < chains;
         c += (long long)gridDim.x * blockDim.x) {
        const int source = (int)(c & 1);
        const long long g = c >> 1;
        const long long gs = offsets[g];
        const long long ge = offsets[g + 1];
        const long long lo = row_lo > gs ? row_lo : gs;
        const long long hi = row_hi < ge ? row_hi : ge;
        if (lo >= hi) continue;

        double v0 = source == 0 ? 0.0 : -INFINITY;
        double v1 = source == 0 ? -INFINITY : 0.0;
        if (lo > gs) {
            v0 = values[2 * c];
            v1 = values[2 * c + 1];
        }
        const double sell = -side;
        for (long long i = lo; i < hi; ++i) {
            const double r = returns[i];
            const double buy = r - side;
            double flat = -INFINITY;
            double cand = 0.0 + v0;
            if (cand > flat) flat = cand;
            cand = sell + v1;
            if (cand > flat) flat = cand;
            double held = -INFINITY;
            cand = buy + v0;
            if (cand > held) held = cand;
            cand = r + v1;
            if (cand > held) held = cand;
            v0 = flat;
            v1 = held;
        }
        values[2 * c] = v0;
        values[2 * c + 1] = v1;
    }
}

static int rc_scan_offsets_ok(const long long *offsets, int groups, long long n)
{
    if (offsets[0] != 0 || offsets[groups] != n) return 0;
    for (int g = 0; g < groups; ++g) {
        if (offsets[g] >= offsets[g + 1]) return 0;
    }
    return 1;
}

static int rc_scan_blocks(long long chains, int threads)
{
    long long blocks = (chains + threads - 1) / threads;
    return blocks > RC_SCAN_MAX_BLOCKS ? RC_SCAN_MAX_BLOCKS : (int)blocks;
}

extern "C" {

/* The largest block size both chain kernels accept on this device, 0 when the
 * device or its attributes are unavailable. */
int rc_band_dp_max_threads(void)
{
    cudaFuncAttributes band, dp;
    if (cudaFuncGetAttributes(&band, rc_band_chain_kernel) != cudaSuccess ||
        cudaFuncGetAttributes(&dp, rc_dp_chain_kernel) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return band.maxThreadsPerBlock < dp.maxThreadsPerBlock ? band.maxThreadsPerBlock
                                                           : dp.maxThreadsPerBlock;
}

/*
 * Band/cooldown scan for `candidates` score vectors (candidate major, f64).
 * `states` is min_hold + 1. Outputs, all host memory:
 *   positions  candidates x n, the start-flat path (0 or 1);
 *   end_states candidates x groups x states, the end-state index per start;
 *   rewards    candidates x groups x states, the folded f64 reward per start.
 * Returns 0 on success; 1 invalid argument, 2 device unavailable, 3 allocation
 * refused, 4 upload, 5 launch, 6 kernel or synchronize, 7 download.
 */
int rc_band_scan(const double *scores,
                 const double *returns,
                 const long long *offsets,
                 int groups,
                 long long n,
                 int candidates,
                 const double *thresholds,
                 const double *bands,
                 int states,
                 double side,
                 int threads,
                 long long chunk_rows,
                 unsigned char *positions,
                 int *end_states,
                 double *rewards)
{
    if (scores == NULL || returns == NULL || offsets == NULL || thresholds == NULL ||
        bands == NULL || positions == NULL || end_states == NULL || rewards == NULL)
        return 1;
    if (groups <= 0 || n <= 0 || candidates <= 0 || states < 2 || chunk_rows <= 0)
        return 1;
    if (!rc_scan_offsets_ok(offsets, groups, n)) return 1;
    if ((long long)candidates * (long long)groups > LLONG_MAX / states) return 1;
    if ((unsigned long long)n > (unsigned long long)(SIZE_MAX / sizeof(double)) / (unsigned long long)candidates)
        return 1;
    const int max_threads = rc_band_dp_max_threads();
    if (max_threads <= 0) return 2;
    if (threads < 1 || threads > max_threads) return 1;

    const long long chains = (long long)candidates * (long long)groups * (long long)states;
    const size_t cells = (size_t)n * (size_t)candidates;
    double *d_scores = NULL, *d_returns = NULL, *d_thr = NULL, *d_band = NULL;
    double *d_reward = NULL;
    long long *d_offsets = NULL;
    int *d_state = NULL;
    unsigned char *d_pos = NULL;
    int status = 0;

    if (rc_checked_malloc((void **)&d_scores, cells * sizeof(double)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_returns, (size_t)n * sizeof(double)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_offsets, (size_t)(groups + 1) * sizeof(long long)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_thr, (size_t)candidates * sizeof(double)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_band, (size_t)candidates * sizeof(double)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_state, (size_t)chains * sizeof(int)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_reward, (size_t)chains * sizeof(double)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_pos, cells) != cudaSuccess) {
        status = 3;
        goto done;
    }
    if (cudaMemcpy(d_scores, scores, cells * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_returns, returns, (size_t)n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_offsets, offsets, (size_t)(groups + 1) * sizeof(long long), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_thr, thresholds, (size_t)candidates * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_band, bands, (size_t)candidates * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        status = 4;
        goto done;
    }
    {
        const int blocks = rc_scan_blocks(chains, threads);
        for (long long lo = 0; lo < n; lo += chunk_rows) {
            long long hi = n - lo > chunk_rows ? lo + chunk_rows : n;
            rc_band_chain_kernel<<<blocks, threads>>>(d_scores, d_returns, d_offsets, d_thr,
                                                      d_band, groups, states, n, chains, side,
                                                      lo, hi, d_state, d_reward, d_pos);
            if (cudaGetLastError() != cudaSuccess) {
                status = 5;
                goto done;
            }
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        status = 6;
        goto done;
    }
    if (cudaMemcpy(positions, d_pos, cells, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(end_states, d_state, (size_t)chains * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(rewards, d_reward, (size_t)chains * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        status = 7;
        goto done;
    }

done:
    if (d_pos != NULL) cudaFree(d_pos);
    if (d_reward != NULL) cudaFree(d_reward);
    if (d_state != NULL) cudaFree(d_state);
    if (d_band != NULL) cudaFree(d_band);
    if (d_thr != NULL) cudaFree(d_thr);
    if (d_offsets != NULL) cudaFree(d_offsets);
    if (d_returns != NULL) cudaFree(d_returns);
    if (d_scores != NULL) cudaFree(d_scores);
    if (status != 0) cudaGetLastError();
    return status;
}

/*
 * Hindsight DP scan. `values` (host) receives groups x 2 x 2 doubles: for
 * group g and source state j, (V(flat), V(long)) at the end of the group, i.e.
 * column j of the group's max-plus carry. Return codes as rc_band_scan.
 */
int rc_dp_scan(const double *returns,
               const long long *offsets,
               int groups,
               long long n,
               double side,
               int threads,
               long long chunk_rows,
               double *values)
{
    if (returns == NULL || offsets == NULL || values == NULL) return 1;
    if (groups <= 0 || n <= 0 || chunk_rows <= 0) return 1;
    if (!rc_scan_offsets_ok(offsets, groups, n)) return 1;
    const int max_threads = rc_band_dp_max_threads();
    if (max_threads <= 0) return 2;
    if (threads < 1 || threads > max_threads) return 1;

    const long long chains = 2LL * (long long)groups;
    double *d_returns = NULL, *d_values = NULL;
    long long *d_offsets = NULL;
    int status = 0;

    if (rc_checked_malloc((void **)&d_returns, (size_t)n * sizeof(double)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_offsets, (size_t)(groups + 1) * sizeof(long long)) != cudaSuccess ||
        rc_checked_malloc((void **)&d_values, (size_t)chains * 2 * sizeof(double)) != cudaSuccess) {
        status = 3;
        goto done;
    }
    if (cudaMemcpy(d_returns, returns, (size_t)n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_offsets, offsets, (size_t)(groups + 1) * sizeof(long long), cudaMemcpyHostToDevice) != cudaSuccess) {
        status = 4;
        goto done;
    }
    {
        const int blocks = rc_scan_blocks(chains, threads);
        for (long long lo = 0; lo < n; lo += chunk_rows) {
            long long hi = n - lo > chunk_rows ? lo + chunk_rows : n;
            rc_dp_chain_kernel<<<blocks, threads>>>(d_returns, d_offsets, groups, side, lo, hi,
                                                    d_values);
            if (cudaGetLastError() != cudaSuccess) {
                status = 5;
                goto done;
            }
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        status = 6;
        goto done;
    }
    if (cudaMemcpy(values, d_values, (size_t)chains * 2 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        status = 7;
        goto done;
    }

done:
    if (d_values != NULL) cudaFree(d_values);
    if (d_offsets != NULL) cudaFree(d_offsets);
    if (d_returns != NULL) cudaFree(d_returns);
    if (status != 0) cudaGetLastError();
    return status;
}

} /* extern "C" */
