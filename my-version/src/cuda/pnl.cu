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

    if (cudaMalloc((void **)&ctx->d_returns, ret_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void **)&ctx->d_groups, group_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void **)&ctx->d_scores, score_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void **)&ctx->d_partials, part_bytes) != cudaSuccess) goto fail;

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
