/*
 * #309: batch walk-forward long-only P&L over (candidate, window).
 *
 * The host lays every window's valid rows out in chain order. A chain is the
 * maximal run of rows that share one position state: one group in pooled
 * mode (the position carries across windows), one (window, group) in
 * per-window mode. Rows of a chain are ascending in (window, time). Inside a
 * chain a window is one contiguous block; the final exit is charged on the
 * last row of every block, and the position state survives into the next
 * block of the chain.
 *
 * One thread folds one (candidate, chain) in ascending row order, so the f64
 * sums of a block never depend on the launch shape. The host merges blocks
 * into window cells in ascending block order.
 *
 * Nothing here is P x N: the context holds one declared tile of at most
 * `tile` candidates, and the host streams the population through it.
 */

#include <cuda_runtime.h>
#include "alloc_guard.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct RcWfbCell {
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
} RcWfbCell;

typedef struct RcWfbCtx {
    long long n;
    int chains;
    int blocks;
    int windows;
    int tile;
    double *d_returns;
    int *d_win;
    long long *d_chain_off;
    int *d_chain_block0;
    double *d_scores;     /* tile x n, candidate major */
    double *d_thr;        /* tile x windows */
    RcWfbCell *d_out;     /* tile x blocks */
    size_t device_bytes;
} RcWfbCtx;

__global__ void rc_wfb_kernel(const double *__restrict__ returns,
                              const int *__restrict__ win,
                              const long long *__restrict__ chain_off,
                              const int *__restrict__ chain_block0,
                              const double *__restrict__ scores,
                              const double *__restrict__ thr,
                              double side,
                              long long n,
                              int chains,
                              int blocks,
                              int windows,
                              int candidates,
                              RcWfbCell *__restrict__ out)
{
    const long long total = (long long)candidates * (long long)chains;
    for (long long job = (long long)blockIdx.x * blockDim.x + threadIdx.x; job < total;
         job += (long long)gridDim.x * blockDim.x) {
        const int cand = (int)(job / chains);
        const int chain = (int)(job % chains);
        const long long lo = chain_off[chain];
        const long long hi = chain_off[chain + 1];
        const double *zc = scores + (size_t)cand * (size_t)n;
        const double *tc = thr + (size_t)cand * (size_t)windows;
        RcWfbCell *oc = out + (size_t)cand * (size_t)blocks;

        int block = chain_block0[chain];
        int cur = win[lo];
        double prev = 0.0;
        RcWfbCell acc;
        memset(&acc, 0, sizeof(acc));

        for (long long i = lo; i < hi; ++i) {
            const int w = win[i];
            if (w != cur) {
                oc[block++] = acc;
                memset(&acc, 0, sizeof(acc));
                cur = w;
            }
            const double r = returns[i];
            if (!isfinite(r)) {
                acc.status = 2;
                continue;
            }
            const double s = zc[i];
            const double p = (isfinite(s) && s > tc[w]) ? 1.0 : 0.0;
            const double d = p - prev;
            const bool last = (i + 1 == hi) || (win[i + 1] != w);
            const double gross = p * r;
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
            prev = p;
        }
        oc[block] = acc;
    }
}

static void rc_wfb_free(RcWfbCtx *ctx)
{
    if (ctx == NULL) return;
    if (ctx->d_returns) cudaFree(ctx->d_returns);
    if (ctx->d_win) cudaFree(ctx->d_win);
    if (ctx->d_chain_off) cudaFree(ctx->d_chain_off);
    if (ctx->d_chain_block0) cudaFree(ctx->d_chain_block0);
    if (ctx->d_scores) cudaFree(ctx->d_scores);
    if (ctx->d_thr) cudaFree(ctx->d_thr);
    if (ctx->d_out) cudaFree(ctx->d_out);
    free(ctx);
}

extern "C" {

/*
 * Upload the layout once. `chain_off` has chains + 1 entries ending at n;
 * `chain_block0[k]` is the first block index of chain k, blocks are numbered
 * in chain order. Returns NULL on invalid arguments, a refused allocation or a
 * failed copy; the cause is cleared from the runtime's error slot.
 */
void *rc_wfb_create(const double *returns,
                    const int *win,
                    const long long *chain_off,
                    const int *chain_block0,
                    long long n,
                    int chains,
                    int blocks,
                    int windows,
                    int tile)
{
    if (returns == NULL || win == NULL || chain_off == NULL || chain_block0 == NULL ||
        n <= 0 || chains <= 0 || blocks <= 0 || windows <= 0 || tile <= 0) {
        return NULL;
    }
    if (chain_off[0] != 0 || chain_off[chains] != n) return NULL;
    for (int k = 0; k < chains; ++k) {
        if (chain_off[k + 1] <= chain_off[k]) return NULL;
        if (chain_block0[k] < 0 || chain_block0[k] >= blocks) return NULL;
    }
    for (long long i = 0; i < n; ++i) {
        if (win[i] < 0 || win[i] >= windows) return NULL;
    }
    for (int k = 0; k < chains; ++k) {
        long long changes = 0;
        for (long long i = chain_off[k] + 1; i < chain_off[k + 1]; ++i) {
            if (win[i] < win[i - 1]) return NULL;
            if (win[i] != win[i - 1]) ++changes;
        }
        if ((long long)chain_block0[k] + changes >= blocks) return NULL;
    }
    if ((unsigned long long)n > (unsigned long long)(SIZE_MAX / sizeof(double)) / (unsigned long long)tile) {
        return NULL;
    }
    RcWfbCtx *ctx = (RcWfbCtx *)calloc(1, sizeof(RcWfbCtx));
    if (ctx == NULL) return NULL;
    ctx->n = n;
    ctx->chains = chains;
    ctx->blocks = blocks;
    ctx->windows = windows;
    ctx->tile = tile;

    const size_t ret_bytes = (size_t)n * sizeof(double);
    const size_t win_bytes = (size_t)n * sizeof(int);
    const size_t off_bytes = (size_t)(chains + 1) * sizeof(long long);
    const size_t b0_bytes = (size_t)chains * sizeof(int);
    const size_t score_bytes = (size_t)n * (size_t)tile * sizeof(double);
    const size_t thr_bytes = (size_t)tile * (size_t)windows * sizeof(double);
    const size_t out_bytes = (size_t)tile * (size_t)blocks * sizeof(RcWfbCell);

    if (rc_checked_malloc((void **)&ctx->d_returns, ret_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_win, win_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_chain_off, off_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_chain_block0, b0_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_scores, score_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_thr, thr_bytes) != cudaSuccess) goto fail;
    if (rc_checked_malloc((void **)&ctx->d_out, out_bytes) != cudaSuccess) goto fail;
    ctx->device_bytes = ret_bytes + win_bytes + off_bytes + b0_bytes + score_bytes + thr_bytes + out_bytes;

    if (cudaMemcpy(ctx->d_returns, returns, ret_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(ctx->d_win, win, win_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(ctx->d_chain_off, chain_off, off_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(ctx->d_chain_block0, chain_block0, b0_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    return ctx;

fail:
    cudaGetLastError();
    rc_wfb_free(ctx);
    return NULL;
}

void rc_wfb_destroy(void *opaque)
{
    rc_wfb_free((RcWfbCtx *)opaque);
}

int rc_wfb_tile(void *opaque)
{
    RcWfbCtx *ctx = (RcWfbCtx *)opaque;
    return ctx == NULL ? 0 : ctx->tile;
}

long long rc_wfb_device_bytes(void *opaque)
{
    RcWfbCtx *ctx = (RcWfbCtx *)opaque;
    return ctx == NULL ? 0 : (long long)ctx->device_bytes;
}

/*
 * Evaluate `candidates` (<= tile) score rows. `scores` is candidates x n,
 * `thresholds` candidates x windows, `out` candidates x blocks (host).
 * Returns 0 on success; 1 invalid argument, 4 upload, 5 launch,
 * 6 kernel or synchronize, 7 download.
 */
int rc_wfb_eval(void *opaque,
                const double *scores,
                const double *thresholds,
                int candidates,
                double side,
                int threads,
                RcWfbCell *out)
{
    RcWfbCtx *ctx = (RcWfbCtx *)opaque;
    if (ctx == NULL || scores == NULL || thresholds == NULL || out == NULL) return 1;
    if (candidates <= 0 || candidates > ctx->tile) return 1;
    if (threads <= 0 || threads > 1024 || (threads % 32) != 0) return 1;

    const size_t score_bytes = (size_t)ctx->n * (size_t)candidates * sizeof(double);
    const size_t thr_bytes = (size_t)candidates * (size_t)ctx->windows * sizeof(double);
    const size_t out_bytes = (size_t)candidates * (size_t)ctx->blocks * sizeof(RcWfbCell);
    int status = 0;

    if (cudaMemcpy(ctx->d_scores, scores, score_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ctx->d_thr, thresholds, thr_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(ctx->d_out, 0, out_bytes) != cudaSuccess) {
        status = 4;
        goto done;
    }
    {
        const long long jobs = (long long)candidates * (long long)ctx->chains;
        long long grid = (jobs + threads - 1) / threads;
        if (grid > 65535) grid = 65535;
        rc_wfb_kernel<<<(unsigned int)grid, threads>>>(ctx->d_returns, ctx->d_win, ctx->d_chain_off,
                                                       ctx->d_chain_block0, ctx->d_scores, ctx->d_thr,
                                                       side, ctx->n, ctx->chains, ctx->blocks,
                                                       ctx->windows, candidates, ctx->d_out);
        if (cudaGetLastError() != cudaSuccess) {
            status = 5;
            goto done;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        status = 6;
        goto done;
    }
    if (cudaMemcpy(out, ctx->d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        status = 7;
        goto done;
    }
done:
    if (status != 0) cudaGetLastError();
    return status;
}

} /* extern "C" */
