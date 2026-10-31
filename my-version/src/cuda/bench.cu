/*
 * Device benchmark joins, relative returns, rolling beta and per-timestamp
 * cross means (issue #299).
 *
 * CPU reference: `enrich::benchmark_return`, `benchmark_vol`, `benchmark_beta`,
 * `beta_window`, `compact_all` (beta384, rel_btc96) and `market_state_pass`.
 *
 * Benchmark side (`RcBench`): the benchmark series (BTC, optionally ETH) is a
 * small sorted `(time, planes...)` array that the CPU already derives from the
 * benchmark symbol. It is uploaded once. Each base row binary-searches its own
 * timestamp for an exact match; a missing timestamp is NaN, never a stale value.
 * Planes: r1 r16 r20 r96 r672 vol20 vol96 volume.
 *
 *   lookup  btc_ and eth_ returns and vols      pure gather, bitwise
 *   rel     own k-bar log return - plane      own `log` is CUDA, tier 1e-12
 *   beta    rolling cov/var, clip +-5         sequential window sum in CPU order
 *
 * The beta window is summed in the same ascending order as the CPU (`-fmad=false`
 * in build.rs, no atomics, one thread per row), so beta differs from the CPU
 * only through the device `log` of the asset's own return.
 *
 * Cross-section side (`rc_bench_market_*`): operates on the resident `RcEnrich`
 * context after `rc_enrich_xs_prepare` built the (time, symbol, row) order. One
 * thread per timestamp group walks the group in that order, so sums use the CPU
 * addition order exactly. Every row is written by exactly one group thread.
 */

#include <cuda_runtime.h>
#include "alloc_guard.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "enrich_common.h"

#define RC_BENCH_PLANES 8
#define RC_BENCH_R1 0
#define RC_BENCH_R16 1
#define RC_BENCH_R20 2
#define RC_BENCH_R96 3
#define RC_BENCH_R672 4
#define RC_BENCH_VOL20 5
#define RC_BENCH_VOL96 6
#define RC_BENCH_VOLUME 7

#define RC_BENCH_LOOKUP 0
#define RC_BENCH_REL 1
#define RC_BENCH_BETA 2
#define RC_BENCH_BETA_CLIPCOV 3

#define RC_MIN_CROSS_MEMBERS 20

typedef struct RcBenchSeries {
    int nb;
    const long long *times;
    const double *planes[RC_BENCH_PLANES];
} RcBenchSeries;

typedef struct RcBench {
    int n;
    int nseg;
    long long step;
    long long *d_times;
    double *d_close;
    int *d_seg_start;
    int *d_seg_id;
    unsigned char *d_self; /* [2][nseg]: segment is the benchmark symbol */
    int nb[2];
    long long *d_bt[2];
    double *d_bp[2][RC_BENCH_PLANES];
    int *d_idx[2];   /* matched benchmark row per base row, -1 if absent */
    double *d_x;     /* own one-bar log return */
    double *d_y[2];  /* benchmark r1 joined to the base rows */
    double *d_out;
    int y_ready[2];
    int x_ready;
    RcEnrichStats last;
    double batch_start_ms;
} RcBench;

static double rc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int rc_alloc(void **p, size_t bytes)
{
#ifdef RCSR_UNIFIED
    return (int)rc_checked_malloc_managed(p, bytes);
#else
    return (int)rc_checked_malloc(p, bytes);
#endif
}

static void rc_upload(void *dst, const void *src, size_t bytes)
{
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

static void rc_download(void *dst, const void *src, size_t bytes)
{
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

static void bstats_begin(RcBench *c)
{
    memset(&c->last, 0, sizeof(c->last));
    c->batch_start_ms = rc_now_ms();
}

static void bstats_end(RcBench *c)
{
    c->last.wall_ms = rc_now_ms() - c->batch_start_ms;
    c->last.batch = 1;
}

static int blocks_for(int n, int threads)
{
    return (n + threads - 1) / threads;
}

/* ------------------------------------------------------------------------- */
/* Benchmark kernels                                                         */
/* ------------------------------------------------------------------------- */

__global__ void k_bench_seg_id(const int *__restrict__ seg_start, int nseg, int n,
                               int *__restrict__ seg_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int lo = 0, hi = nseg - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (seg_start[mid] <= i) lo = mid; else hi = mid - 1;
    }
    seg_id[i] = lo;
}

/* Exact timestamp equality against the sorted benchmark times. */
__global__ void k_bench_index(const long long *__restrict__ t,
                              const long long *__restrict__ bt, int nb, int n,
                              int *__restrict__ idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long v = t[i];
    int lo = 0, hi = nb;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (bt[mid] < v) lo = mid + 1; else hi = mid;
    }
    idx[i] = (lo < nb && bt[lo] == v) ? lo : -1;
}

__global__ void k_bench_gather(const int *__restrict__ idx,
                               const double *__restrict__ plane, int n,
                               double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = idx[i];
    out[i] = (j >= 0) ? plane[j] : NAN;
}

/* Own k-bar log return, identical gate to `enrich::ret_k` (series-local). */
__device__ __forceinline__ double bench_own_ret(const long long *t, const double *c,
                                                const int *seg_start, const int *seg_id,
                                                int i, int k, long long step)
{
    int lo = seg_start[seg_id[i]];
    if (i - lo >= k && t[i] - t[i - k] == (long long)k * step &&
        c[i] > 0.0 && c[i - k] > 0.0) {
        return log(c[i] / c[i - k]);
    }
    return NAN;
}

__global__ void k_bench_own_ret(const long long *__restrict__ t,
                                const double *__restrict__ c,
                                const int *__restrict__ seg_start,
                                const int *__restrict__ seg_id, int n, int k,
                                long long step, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bench_own_ret(t, c, seg_start, seg_id, i, k, step);
}

__global__ void k_bench_rel(const long long *__restrict__ t,
                            const double *__restrict__ c,
                            const int *__restrict__ seg_start,
                            const int *__restrict__ seg_id,
                            const int *__restrict__ idx,
                            const double *__restrict__ plane, int n, int k,
                            long long step, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double own = bench_own_ret(t, c, seg_start, seg_id, i, k, step);
    int j = idx[i];
    double b = (j >= 0) ? plane[j] : NAN;
    out[i] = (isfinite(own) && isfinite(b)) ? own - b : NAN;
}

/* Rolling beta: same window order and arithmetic as `enrich::beta_window`. */
__global__ void k_bench_beta(const long long *__restrict__ t,
                             const int *__restrict__ seg_start,
                             const int *__restrict__ seg_id,
                             const unsigned char *__restrict__ self_flag,
                             const double *__restrict__ x,
                             const double *__restrict__ y, int n, int w,
                             long long step, int clip_cov, int use_self,
                             double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = NAN;
    int s = seg_id[i];
    int li = i - seg_start[s];
    int contig = (step > 0) && (li + 1 >= w) &&
                 (t[i + 1 - w] == t[i] - (long long)(w - 1) * step);
    if (contig) {
        if (use_self && self_flag[s]) {
            v = 1.0;
        } else {
            double sx = 0.0, sy = 0.0, syy = 0.0, sxy = 0.0;
            int ok = 1;
            for (int j = i + 1 - w; j <= i; ++j) {
                double xv = x[j];
                double yv = y[j];
                if (!isfinite(xv) || !isfinite(yv)) { ok = 0; break; }
                sx += xv;
                sy += yv;
                syy += yv * yv;
                sxy += xv * yv;
            }
            if (ok) {
                double nf = (double)w;
                double cov = sxy / nf - (sx / nf) * (sy / nf);
                double var = syy / nf - (sy / nf) * (sy / nf);
                if (clip_cov) cov = fmin(fmax(cov, -0.05), 0.05);
                if (var > 0.0) v = fmin(fmax(cov / var, -5.0), 5.0);
            }
        }
    }
    out[i] = v;
}

/* ------------------------------------------------------------------------- */
/* Benchmark context                                                         */
/* ------------------------------------------------------------------------- */

extern "C" {

void rc_bench_destroy(RcBench *c);

RcBench *rc_bench_create(const long long *times, const double *close, int n,
                         const int *seg_start, const int *seg_len, int nseg,
                         long long step, const RcBenchSeries *bench0,
                         const RcBenchSeries *bench1,
                         const unsigned char *self_flags)
{
    if (n <= 0 || nseg <= 0 || nseg > n || !bench0) return NULL;
    (void)seg_len;
    RcBench *c = (RcBench *)calloc(1, sizeof(RcBench));
    if (!c) return NULL;
    c->n = n;
    c->nseg = nseg;
    c->step = step;
    bstats_begin(c);
    size_t rows = (size_t)n;
    int ok = 1;
    double t0 = rc_now_ms();
    ok &= rc_alloc((void **)&c->d_times, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_close, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_start, (size_t)nseg * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_id, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_self, 2 * (size_t)nseg) == 0;
    ok &= rc_alloc((void **)&c->d_out, rows * sizeof(double)) == 0;
    const RcBenchSeries *b[2] = {bench0, bench1};
    unsigned long long bench_bytes = 0;
    for (int k = 0; ok && k < 2; ++k) {
        c->nb[k] = (b[k] && b[k]->nb > 0) ? b[k]->nb : 0;
        if (c->nb[k] == 0) continue;
        size_t nb = (size_t)c->nb[k];
        ok &= rc_alloc((void **)&c->d_bt[k], nb * sizeof(long long)) == 0;
        for (int p = 0; ok && p < RC_BENCH_PLANES; ++p) {
            ok &= rc_alloc((void **)&c->d_bp[k][p], nb * sizeof(double)) == 0;
        }
        ok &= rc_alloc((void **)&c->d_idx[k], rows * sizeof(int)) == 0;
        ok &= rc_alloc((void **)&c->d_y[k], rows * sizeof(double)) == 0;
        bench_bytes += nb * (sizeof(long long) + RC_BENCH_PLANES * sizeof(double));
    }
    c->last.alloc_ms = rc_now_ms() - t0;
    if (!ok) {
        rc_bench_destroy(c);
        return NULL;
    }
    t0 = rc_now_ms();
    rc_upload(c->d_times, times, rows * sizeof(long long));
    rc_upload(c->d_close, close, rows * sizeof(double));
    rc_upload(c->d_seg_start, seg_start, (size_t)nseg * sizeof(int));
    if (self_flags) rc_upload(c->d_self, self_flags, 2 * (size_t)nseg);
    else cudaMemset(c->d_self, 0, 2 * (size_t)nseg);
    for (int k = 0; k < 2; ++k) {
        if (c->nb[k] == 0) continue;
        size_t nb = (size_t)c->nb[k];
        rc_upload(c->d_bt[k], b[k]->times, nb * sizeof(long long));
        for (int p = 0; p < RC_BENCH_PLANES; ++p)
            rc_upload(c->d_bp[k][p], b[k]->planes[p], nb * sizeof(double));
    }
    c->last.h2d_ms = rc_now_ms() - t0;
    c->last.h2d_bytes = rows * (sizeof(long long) + sizeof(double)) +
                        (unsigned long long)nseg * (sizeof(int) + 2) + bench_bytes;

    const int threads = 256;
    int nbk = blocks_for(n, threads);
    k_bench_seg_id<<<nbk, threads>>>(c->d_seg_start, nseg, n, c->d_seg_id);
    for (int k = 0; k < 2; ++k) {
        if (c->nb[k] == 0) continue;
        k_bench_index<<<nbk, threads>>>(c->d_times, c->d_bt[k], c->nb[k], n,
                                        c->d_idx[k]);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        bstats_end(c);
        rc_bench_destroy(c);
        return NULL;
    }
    bstats_end(c);
    return c;
}

void rc_bench_destroy(RcBench *c)
{
    if (!c) return;
    cudaFree(c->d_times);
    cudaFree(c->d_close);
    cudaFree(c->d_seg_start);
    cudaFree(c->d_seg_id);
    cudaFree(c->d_self);
    cudaFree(c->d_out);
    cudaFree(c->d_x);
    for (int k = 0; k < 2; ++k) {
        cudaFree(c->d_bt[k]);
        for (int p = 0; p < RC_BENCH_PLANES; ++p) cudaFree(c->d_bp[k][p]);
        cudaFree(c->d_idx[k]);
        cudaFree(c->d_y[k]);
    }
    free(c);
}

void rc_bench_last_stats(RcBench *c, RcEnrichStats *out)
{
    if (c && out) *out = c->last;
}

/* One output column. kind: lookup / rel / beta / beta with clipped covariance.
   `a` is the plane (lookup, rel) or the window (beta); `k` is the return
   horizon for rel. `use_self` applies the CPU "benchmark against itself" 1.0. */
int rc_bench_column(RcBench *c, int bench, int kind, int a, int use_self,
                    double *host_out)
{
    if (!c || !host_out || bench < 0 || bench > 1 || c->nb[bench] == 0) return 1;
    bstats_begin(c);
    const int threads = 256;
    int nbk = blocks_for(c->n, threads);
    const double *plane = NULL;
    if (kind == RC_BENCH_LOOKUP || kind == RC_BENCH_REL) {
        if (kind == RC_BENCH_LOOKUP) {
            if (a < 0 || a >= RC_BENCH_PLANES) return 2;
            plane = c->d_bp[bench][a];
            k_bench_gather<<<nbk, threads>>>(c->d_idx[bench], plane, c->n, c->d_out);
        } else {
            int pl;
            switch (a) {
            case 1: pl = RC_BENCH_R1; break;
            case 16: pl = RC_BENCH_R16; break;
            case 20: pl = RC_BENCH_R20; break;
            case 96: pl = RC_BENCH_R96; break;
            default: return 2;
            }
            k_bench_rel<<<nbk, threads>>>(c->d_times, c->d_close, c->d_seg_start,
                                          c->d_seg_id, c->d_idx[bench],
                                          c->d_bp[bench][pl], c->n, a, c->step,
                                          c->d_out);
        }
    } else {
        if (a <= 0) return 2;
        if (!c->x_ready) {
            if (!c->d_x && rc_alloc((void **)&c->d_x, (size_t)c->n * sizeof(double)) != 0) {
                c->last.failed = 1;
                bstats_end(c);
                return 3;
            }
            k_bench_own_ret<<<nbk, threads>>>(c->d_times, c->d_close, c->d_seg_start,
                                              c->d_seg_id, c->n, 1, c->step, c->d_x);
            c->x_ready = 1;
        }
        if (!c->y_ready[bench]) {
            k_bench_gather<<<nbk, threads>>>(c->d_idx[bench],
                                             c->d_bp[bench][RC_BENCH_R1], c->n,
                                             c->d_y[bench]);
            c->y_ready[bench] = 1;
        }
        k_bench_beta<<<nbk, threads>>>(
            c->d_times, c->d_seg_start, c->d_seg_id, c->d_self + (size_t)bench * c->nseg,
            c->d_x, c->d_y[bench], c->n, a, c->step,
            kind == RC_BENCH_BETA_CLIPCOV ? 1 : 0, use_self, c->d_out);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    c->last.kernel_ms = rc_now_ms() - c->batch_start_ms;
    if (err != cudaSuccess) {
        c->last.failed = 1;
        bstats_end(c);
        return 4;
    }
    double t0 = rc_now_ms();
    rc_download(host_out, c->d_out, (size_t)c->n * sizeof(double));
    c->last.d2h_ms = rc_now_ms() - t0;
    c->last.d2h_bytes = (unsigned long long)c->n * sizeof(double);
    bstats_end(c);
    return 0;
}

} /* extern "C" */

/* ------------------------------------------------------------------------- */
/* Per-timestamp cross means on the resident (time, symbol, row) order       */
/* ------------------------------------------------------------------------- */

/* One thread per timestamp group, CPU addition order (order[] is the CPU's
   (time, symbol, row) order). Outputs: mean, breadth, dispersion, ex-self rel.
   Groups under 20 finite PIT members, and non-member rows, are NaN. */
__global__ void k_market_group(const double *__restrict__ src,
                               const unsigned int *__restrict__ order,
                               const unsigned char *__restrict__ member,
                               const int *__restrict__ off, int groups,
                               double *__restrict__ mean_out,
                               double *__restrict__ breadth_out,
                               double *__restrict__ disp_out,
                               double *__restrict__ rel_out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    int k0 = off[g], k1 = off[g + 1];
    double sum = 0.0;
    int cnt = 0;
    for (int k = k0; k < k1; ++k) {
        unsigned int r = order[k];
        double v = src[r];
        if (member[r] && isfinite(v)) {
            sum += v;
            ++cnt;
        }
    }
    if (cnt < RC_MIN_CROSS_MEMBERS) {
        for (int k = k0; k < k1; ++k) {
            unsigned int r = order[k];
            mean_out[r] = NAN;
            breadth_out[r] = NAN;
            disp_out[r] = NAN;
            rel_out[r] = NAN;
        }
        return;
    }
    double nf = (double)cnt;
    double mean = sum / nf;
    double ss = 0.0;
    int positive = 0;
    for (int k = k0; k < k1; ++k) {
        unsigned int r = order[k];
        double v = src[r];
        if (member[r] && isfinite(v)) {
            double d = v - mean;
            ss += d * d;
            if (v > 0.0) ++positive;
        }
    }
    double sd = sqrt(fmax(ss / nf, 0.0));
    double br = (double)positive / nf;
    for (int k = k0; k < k1; ++k) {
        unsigned int r = order[k];
        if (!member[r]) {
            mean_out[r] = NAN;
            breadth_out[r] = NAN;
            disp_out[r] = NAN;
            rel_out[r] = NAN;
            continue;
        }
        double v = src[r];
        mean_out[r] = mean;
        breadth_out[r] = br;
        disp_out[r] = sd;
        rel_out[r] = (isfinite(v) && cnt > 1) ? v - (sum - v) / (double)(cnt - 1) : NAN;
    }
}

/* btc_volume_share: benchmark volume at the group's timestamp / total member
   volume. Exact timestamp match, no forward fill. */
__global__ void k_market_volume_share(const double *__restrict__ vol,
                                      const unsigned int *__restrict__ order,
                                      const unsigned char *__restrict__ member,
                                      const int *__restrict__ off, int groups,
                                      const long long *__restrict__ times,
                                      const long long *__restrict__ bt,
                                      const double *__restrict__ bvol, int nb,
                                      double *__restrict__ out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    int k0 = off[g], k1 = off[g + 1];
    double total = 0.0;
    int cnt = 0;
    for (int k = k0; k < k1; ++k) {
        unsigned int r = order[k];
        double v = vol[r];
        if (member[r] && isfinite(v)) {
            total += v;
            ++cnt;
        }
    }
    double share = NAN;
    if (cnt >= RC_MIN_CROSS_MEMBERS && total > 0.0) {
        long long t = times[order[k0]];
        int lo = 0, hi = nb;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (bt[mid] < t) lo = mid + 1; else hi = mid;
        }
        if (lo < nb && bt[lo] == t) {
            double b = bvol[lo];
            if (isfinite(b)) share = b / total;
        }
    }
    for (int k = k0; k < k1; ++k) {
        unsigned int r = order[k];
        out[r] = member[r] ? share : NAN;
    }
}

/* Prefix sums of the member's r1 against the market r1 per segment, laid out
   with a stride of one extra slot per segment so every segment owns nt+1
   entries (exactly the CPU's per-segment prefix vectors). */
__global__ void k_corr_prefix(const double *__restrict__ x,
                              const double *__restrict__ mkt,
                              const unsigned char *__restrict__ member,
                              const int *__restrict__ seg_start,
                              const int *__restrict__ seg_len, int nseg,
                              double *__restrict__ px, double *__restrict__ py,
                              double *__restrict__ pxx, double *__restrict__ pyy,
                              double *__restrict__ pxy, double *__restrict__ pc)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int nt = seg_len[s];
    if (nt < 96) return;
    size_t base = (size_t)seg_start[s] + (size_t)s;
    double a = 0.0, b = 0.0, aa = 0.0, bb = 0.0, ab = 0.0, cc = 0.0;
    px[base] = 0.0; py[base] = 0.0; pxx[base] = 0.0; pyy[base] = 0.0;
    pxy[base] = 0.0; pc[base] = 0.0;
    for (int j = 0; j < nt; ++j) {
        int g = seg_start[s] + j;
        double xv = x[g];
        double yv = member[g] ? mkt[g] : NAN;
        int ok = isfinite(xv) && isfinite(yv);
        a += ok ? xv : 0.0;
        b += ok ? yv : 0.0;
        aa += ok ? xv * xv : 0.0;
        bb += ok ? yv * yv : 0.0;
        ab += ok ? xv * yv : 0.0;
        cc += ok ? 1.0 : 0.0;
        px[base + j + 1] = a;
        py[base + j + 1] = b;
        pxx[base + j + 1] = aa;
        pyy[base + j + 1] = bb;
        pxy[base + j + 1] = ab;
        pc[base + j + 1] = cc;
    }
}

__global__ void k_corr_eval(const long long *__restrict__ t,
                            const int *__restrict__ seg_start,
                            const int *__restrict__ seg_len,
                            const int *__restrict__ seg_id,
                            const double *__restrict__ px, const double *__restrict__ py,
                            const double *__restrict__ pxx, const double *__restrict__ pyy,
                            const double *__restrict__ pxy, const double *__restrict__ pc,
                            int n, long long step, double *__restrict__ out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    out[g] = NAN;
    int s = seg_id[g];
    int nt = seg_len[s];
    if (nt < 96) return;
    int i = g - seg_start[s];
    if (step <= 0 || i + 1 < 96) return;
    if (t[g + 1 - 96] != t[g] - 95LL * step) return;
    size_t base = (size_t)seg_start[s] + (size_t)s;
    int lo = i + 1 - 96;
    if (pc[base + i + 1] - pc[base + lo] != 96.0) return;
    double w = 96.0;
    double sx = px[base + i + 1] - px[base + lo];
    double sy = py[base + i + 1] - py[base + lo];
    double sxx = pxx[base + i + 1] - pxx[base + lo];
    double syy = pyy[base + i + 1] - pyy[base + lo];
    double sxy = pxy[base + i + 1] - pxy[base + lo];
    double vx = sxx - sx * sx / w;
    double vy = syy - sy * sy / w;
    if (vx > 0.0 && vy > 0.0) {
        out[g] = fmin(fmax((sxy - sx * sy / w) / sqrt(vx * vy), -1.0), 1.0);
    }
}

static void market_ledger(RcEnrich *c, double t_start, double h2d_ms,
                          double d2h_ms, unsigned long long h2d_bytes,
                          unsigned long long d2h_bytes, cudaError_t err)
{
    c->last.h2d_ms = h2d_ms;
    c->last.d2h_ms = d2h_ms;
    c->last.h2d_bytes = h2d_bytes;
    c->last.d2h_bytes = d2h_bytes;
    c->last.kernel_ms = rc_now_ms() - t_start - h2d_ms - d2h_ms;
    c->last.wall_ms = rc_now_ms() - t_start;
    c->last.batch = 1;
    c->last.failed = (err != cudaSuccess) ? 1 : 0;
}

extern "C" {

/* Group statistics of `src` (host, n doubles). `want` bit k downloads output k
   (0 mean, 1 breadth, 2 dispersion, 3 rel). Requires xs_prepare. */
int rc_bench_market_group(void *v, const double *src, unsigned int want,
                          double *mean, double *breadth, double *disp, double *rel)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || !src || !c->xs_ready) return 1;
    double t_start = rc_now_ms();
    size_t n = (size_t)c->n;
    rc_upload(c->d_xs_src, src, n * sizeof(double));
    double h2d = rc_now_ms() - t_start;
    int threads = 128;
    int blocks = blocks_for(c->xs_groups, threads);
    k_market_group<<<blocks, threads>>>(c->d_xs_src, c->d_xs_order, c->d_xs_member,
                                        c->d_xs_off, c->xs_groups, c->d_out,
                                        c->d_out2, c->d_out3, c->d_d);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    double t_dl = rc_now_ms();
    unsigned long long bytes = 0;
    if (err == cudaSuccess) {
        if ((want & 1u) && mean) { rc_download(mean, c->d_out, n * sizeof(double)); bytes += n * 8; }
        if ((want & 2u) && breadth) { rc_download(breadth, c->d_out2, n * sizeof(double)); bytes += n * 8; }
        if ((want & 4u) && disp) { rc_download(disp, c->d_out3, n * sizeof(double)); bytes += n * 8; }
        if ((want & 8u) && rel) { rc_download(rel, c->d_d, n * sizeof(double)); bytes += n * 8; }
    }
    market_ledger(c, t_start, h2d, rc_now_ms() - t_dl, n * 8, bytes, err);
    return err == cudaSuccess ? 0 : 2;
}

/* btc_volume_share for the host volume column against the benchmark volume. */
int rc_bench_market_volume_share(void *v, const double *vol, const long long *bt,
                                 const double *bvol, int nb, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || !vol || !bt || !bvol || nb <= 0 || !c->xs_ready) return 1;
    double t_start = rc_now_ms();
    size_t n = (size_t)c->n;
    long long *d_bt = NULL;
    double *d_bv = NULL;
    if (rc_alloc((void **)&d_bt, (size_t)nb * sizeof(long long)) != 0 ||
        rc_alloc((void **)&d_bv, (size_t)nb * sizeof(double)) != 0) {
        cudaFree(d_bt);
        cudaFree(d_bv);
        return 3;
    }
    rc_upload(c->d_xs_src, vol, n * sizeof(double));
    rc_upload(d_bt, bt, (size_t)nb * sizeof(long long));
    rc_upload(d_bv, bvol, (size_t)nb * sizeof(double));
    double h2d = rc_now_ms() - t_start;
    int threads = 128;
    int blocks = blocks_for(c->xs_groups, threads);
    k_market_volume_share<<<blocks, threads>>>(c->d_xs_src, c->d_xs_order,
                                               c->d_xs_member, c->d_xs_off,
                                               c->xs_groups, c->d_times, d_bt, d_bv,
                                               nb, c->d_out);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    double t_dl = rc_now_ms();
    if (err == cudaSuccess) rc_download(out, c->d_out, n * sizeof(double));
    market_ledger(c, t_start, h2d, rc_now_ms() - t_dl,
                  n * 8 + (unsigned long long)nb * 16, n * 8, err);
    cudaFree(d_bt);
    cudaFree(d_bv);
    return err == cudaSuccess ? 0 : 2;
}

/* Average-correlation proxy: group means of the per-member trailing 96-bar
   correlation with the equal-weight market r1. `r1` is the host r1 column. */
int rc_bench_market_avg_corr(void *v, const double *r1, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || !r1 || !out || !c->xs_ready) return 1;
    double t_start = rc_now_ms();
    size_t n = (size_t)c->n;
    size_t padded = n + (size_t)c->nseg + 1;
    double *scratch[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    double *d_x = NULL, *d_corr = NULL;
    int ok = 1;
    for (int k = 0; k < 6 && ok; ++k)
        ok = rc_alloc((void **)&scratch[k], padded * sizeof(double)) == 0;
    ok = ok && rc_alloc((void **)&d_x, n * sizeof(double)) == 0;
    ok = ok && rc_alloc((void **)&d_corr, n * sizeof(double)) == 0;
    if (!ok) {
        for (int k = 0; k < 6; ++k) cudaFree(scratch[k]);
        cudaFree(d_x);
        cudaFree(d_corr);
        return 3;
    }
    rc_upload(d_x, r1, n * sizeof(double));
    rc_upload(c->d_xs_src, r1, n * sizeof(double));
    double h2d = rc_now_ms() - t_start;
    int threads = 128;
    int gblocks = blocks_for(c->xs_groups, threads);
    /* Market r1 mean per timestamp (d_out), breadth/dispersion/rel are scratch. */
    k_market_group<<<gblocks, threads>>>(c->d_xs_src, c->d_xs_order, c->d_xs_member,
                                         c->d_xs_off, c->xs_groups, c->d_out,
                                         c->d_out2, c->d_out3, c->d_d);
    int sblocks = blocks_for(c->nseg, threads);
    k_corr_prefix<<<sblocks, threads>>>(d_x, c->d_out, c->d_xs_member, c->d_seg_start,
                                        c->d_seg_len, c->nseg, scratch[0], scratch[1],
                                        scratch[2], scratch[3], scratch[4], scratch[5]);
    int rblocks = blocks_for(c->n, 256);
    k_corr_eval<<<rblocks, 256>>>(c->d_times, c->d_seg_start, c->d_seg_len, c->d_seg_id,
                                  scratch[0], scratch[1], scratch[2], scratch[3],
                                  scratch[4], scratch[5], c->n, (long long)c->step,
                                  d_corr);
    /* Mean of the member correlations per timestamp. */
    k_market_group<<<gblocks, threads>>>(d_corr, c->d_xs_order, c->d_xs_member,
                                         c->d_xs_off, c->xs_groups, c->d_out,
                                         c->d_out2, c->d_out3, c->d_d);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    double t_dl = rc_now_ms();
    if (err == cudaSuccess) rc_download(out, c->d_out, n * sizeof(double));
    market_ledger(c, t_start, h2d, rc_now_ms() - t_dl, n * 16, n * 8, err);
    for (int k = 0; k < 6; ++k) cudaFree(scratch[k]);
    cudaFree(d_x);
    cudaFree(d_corr);
    return err == cudaSuccess ? 0 : 2;
}

} /* extern "C" */
