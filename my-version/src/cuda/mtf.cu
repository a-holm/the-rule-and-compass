/*
 * Device multi-timeframe resample and causal as-of join (issue #296).
 *
 * The CPU `enrich::mtf_all` output is the byte contract. For each symbol:
 *
 *   1. bucket the base rows with `floor(t / tf)`;
 *   2. fold each bucket in input order: open = first, high = max,
 *      low = min, close = last (volume is not an MTF leaf);
 *   3. evaluate the eleven value leaves on the smaller higher series with the
 *      same sequential arithmetic and prefix-subtraction order as the CPU;
 *   4. join with `higher.time + tf <= base.time`, then require the last closed
 *      bucket to be exactly `bucket_start(base.time) - tf`. A stale or missing
 *      bucket is NaN for every value leaf, while `bar_age` stays finite.
 *
 * No atomics touch an output. Bucket indices come from a deterministic
 * exclusive scan of the new-bucket flags, and every leaf gather is a pure map
 * from a precomputed join index. The context owns the resident base columns,
 * the reusable higher-timeframe arena and the statistics ledger.
 */

#include <cuda_runtime.h>
#include "alloc_guard.h"
#include "cub_guard.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cub/cub.cuh>

#include "win_stats.h"

#define RC_MTF_LEAVES 11
#define RC_MTF_RET1 0
#define RC_MTF_RET4 1
#define RC_MTF_RET24 2
#define RC_MTF_MA 3
#define RC_MTF_VOL 4
#define RC_MTF_ATR 5
#define RC_MTF_RSI 6
#define RC_MTF_CLOSE_POS 7
#define RC_MTF_DONCHIAN 8
#define RC_MTF_RANGE 9
#define RC_MTF_TREND 10
#define RC_MTF_BAR_AGE 11

typedef struct RcMtfStats {
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
} RcMtfStats;

typedef struct RcMtf {
    int n;
    int nseg;
    int step;
    int max_buckets;
    int total_buckets;
    long long prepared_tf;

    long long *d_times;
    double *d_open;
    double *d_high;
    double *d_low;
    double *d_close;
    int *d_seg_start;
    int *d_seg_len;
    int *d_seg_id;

    int *d_flags;
    int *d_bucket_of_row;
    int *d_bucket_lo;
    int *d_seg_bucket_start;
    int *d_seg_bucket_len;
    void *d_scan_tmp;
    size_t scan_tmp_bytes;

    long long *d_h_time;
    double *d_h_open;
    double *d_h_high;
    double *d_h_low;
    double *d_h_close;
    double *d_leaf;
    double *d_ps;
    double *d_ps2;
    double *d_pc;
    double *d_rs;
    double *d_rs2;
    double *d_rc;
    double *d_rs3;
    double *d_rs4;
    int *d_join_idx;
    double *d_age;

    RcMtfStats last;
    double batch_start_ms;
} RcMtf;

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

static void stats_begin(RcMtf *c)
{
    memset(&c->last, 0, sizeof(c->last));
    c->batch_start_ms = rc_now_ms();
}

static void stats_end(RcMtf *c)
{
    c->last.wall_ms = rc_now_ms() - c->batch_start_ms;
    c->last.kernel_ms = c->last.wall_ms;
    c->last.batch = 1;
}

__device__ __forceinline__ long long rc_floor_div(long long a, long long b)
{
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

__device__ __forceinline__ long long rc_bucket(long long t, long long tf)
{
    return rc_floor_div(t, tf) * tf;
}

/* New bucket when the symbol changes or the epoch bucket changes. */
__global__ void k_bucket_flags(const long long *__restrict__ t,
                               const int *__restrict__ seg,
                               int n, long long tf,
                               int *__restrict__ flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || seg[i] != seg[i - 1] ||
                rc_bucket(t[i], tf) != rc_bucket(t[i - 1], tf))
                   ? 1
                   : 0;
}

__global__ void k_bucket_lo(const int *__restrict__ flags,
                            const int *__restrict__ bucket_of_row,
                            int n, int *__restrict__ lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flags[i]) return;
    lo[bucket_of_row[i]] = i;
}

/* One thread per bucket: fold its contiguous base rows in input order. */
__global__ void k_resample(const long long *__restrict__ t,
                           const double *__restrict__ open,
                           const double *__restrict__ high,
                           const double *__restrict__ low,
                           const double *__restrict__ close,
                           const int *__restrict__ bucket_lo,
                           int n, int total, long long tf,
                           long long *__restrict__ h_time,
                           double *__restrict__ h_open,
                           double *__restrict__ h_high,
                           double *__restrict__ h_low,
                           double *__restrict__ h_close)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total) return;
    int lo = bucket_lo[b];
    int hi = (b + 1 < total) ? bucket_lo[b + 1] : n;
    long long start = rc_bucket(t[lo], tf);
    double o = open[lo];
    double h = high[lo];
    double l = low[lo];
    /* The CPU starts from group[0] high/low and then folds every row,
       including the first, so the max/min semantics are identical. */
    for (int i = lo; i < hi; ++i) {
        h = fmax(h, high[i]);
        l = fmin(l, low[i]);
    }
    h_time[b] = start;
    h_open[b] = o;
    h_high[b] = h;
    h_low[b] = l;
    h_close[b] = close[hi - 1];
}

__global__ void k_seg_buckets(const int *__restrict__ seg_start,
                              const int *__restrict__ bucket_of_row,
                              int nseg, int total,
                              int *__restrict__ seg_bs,
                              int *__restrict__ seg_bl)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int bs = bucket_of_row[seg_start[s]];
    int be = (s + 1 < nseg) ? bucket_of_row[seg_start[s + 1]] : total;
    seg_bs[s] = bs;
    seg_bl[s] = be - bs;
}

__device__ __forceinline__ bool rc_contig_local(const long long *t, int base,
                                                int p, int w, int step)
{
    if (w == 0) return true;
    if (p + 1 < w || step <= 0) return false;
    return t[base + p + 1 - w] ==
           t[base + p] - (long long)(w - 1) * step;
}

/* #364: window-statistic arena of one input. `ci` holds the finite count and
   the accumulation distance of each slot as an int pair. */
struct MtfArena {
    double *s1;
    double *c1;
    double *s2;
    double *c2;
    int *ci;
    const double *xp; /* x at slot q is xp[q] */
    __device__ double s(int k, int q) const { return k == 0 ? s1[q] : s2[q]; }
    __device__ double c(int k, int q) const { return k == 0 ? c1[q] : c2[q]; }
    __device__ int bk(int q) const { return ci[2 * q + 1]; }
    __device__ int cn(int q) const { return ci[2 * q]; }
    __device__ double x(int q) const { return xp[q]; }
};

/* Sequential prefix over one segment in the CPU order (restart at the segment
   start, every gap and every RW_BLOCK buckets). Slot `pb + 1 + p` is row `p`. */
__device__ __forceinline__ void rc_prefix_local(const MtfArena &a, const double *x,
                                                const long long *time, int start,
                                                int len, int pb, int step, bool sq)
{
    a.s1[pb] = 0.0;
    a.c1[pb] = 0.0;
    a.s2[pb] = 0.0;
    a.c2[pb] = 0.0;
    a.ci[2 * pb] = 0;
    a.ci[2 * pb + 1] = 0;
    RwAcc acc;
    rw_acc_init(&acc);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        int q = pb + 1 + p;
        bool restart = (p % RW_BLOCK) == 0 ||
                       (p > 0 && time[i] - time[i - 1] != (long long)step);
        rw_acc_push(&acc, x[i], q, restart, sq);
        a.s1[q] = acc.s1;
        a.c1[q] = acc.c1;
        if (sq) {
            a.s2[q] = acc.s2;
            a.c2[q] = acc.c2;
        }
        a.ci[2 * q] = acc.cnt;
        a.ci[2 * q + 1] = q - acc.start;
    }
}

__device__ __forceinline__ double rc_rsi_norm(double gain, double loss)
{
    double rsi = (loss > 0.0)
                     ? 100.0 - 100.0 / (1.0 + gain / loss)
                     : ((gain > 0.0) ? 100.0 : 50.0);
    return rsi / 50.0 - 1.0;
}

/* One thread per segment over its higher buckets. The arithmetic mirrors
   `mtf_ret`, `mtf_atr`, `mtf_rsi`, `ma_ratio`, `rstd_py`, `mtf_close_pos`,
   `mtf_donchian_pos`, `mtf_range_pct` and `mtf_trend_slope` exactly. */
__global__ void k_mtf_leaves(const long long *__restrict__ h_time,
                             const double *__restrict__ h_open,
                             const double *__restrict__ h_high,
                             const double *__restrict__ h_low,
                             const double *__restrict__ h_close,
                             const int *__restrict__ seg_bs,
                             const int *__restrict__ seg_bl,
                             int nseg, int step, int max_buckets,
                             double *ps, double *ps2, double *pc,
                             double *rs, double *rs2, double *rc,
                             double *rs3, double *rs4, double *leaf)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int bs = seg_bs[s];
    int len = seg_bl[s];
    if (len <= 0) return;

    double *ret1 = leaf + (size_t)RC_MTF_RET1 * max_buckets;
    double *ret4 = leaf + (size_t)RC_MTF_RET4 * max_buckets;
    double *ret24 = leaf + (size_t)RC_MTF_RET24 * max_buckets;
    double *ma = leaf + (size_t)RC_MTF_MA * max_buckets;
    double *vol = leaf + (size_t)RC_MTF_VOL * max_buckets;
    double *atrp = leaf + (size_t)RC_MTF_ATR * max_buckets;
    double *rsi = leaf + (size_t)RC_MTF_RSI * max_buckets;
    double *cpos = leaf + (size_t)RC_MTF_CLOSE_POS * max_buckets;
    double *don = leaf + (size_t)RC_MTF_DONCHIAN * max_buckets;
    double *rng = leaf + (size_t)RC_MTF_RANGE * max_buckets;
    double *trend = leaf + (size_t)RC_MTF_TREND * max_buckets;

    /* Returns. CPU: contiguity against the *higher* step, finite closes,
       close at i-k > 0. */
    for (int p = 0; p < len; ++p) {
        int i = bs + p;
        double r1 = NAN;
        if (p >= 1 && h_time[i] - h_time[i - 1] == (long long)step &&
            isfinite(h_close[i]) && isfinite(h_close[i - 1]) &&
            h_close[i - 1] > 0.0) {
            r1 = h_close[i] / h_close[i - 1] - 1.0;
        }
        ret1[i] = r1;
        ret4[i] = NAN;
        ret24[i] = NAN;
        if (p >= 4 && h_time[i] - h_time[i - 4] == 4LL * step &&
            isfinite(h_close[i]) && isfinite(h_close[i - 4]) &&
            h_close[i - 4] > 0.0) {
            ret4[i] = h_close[i] / h_close[i - 4] - 1.0;
        }
        if (p >= 24 && h_time[i] - h_time[i - 24] == 24LL * step &&
            isfinite(h_close[i]) && isfinite(h_close[i - 24]) &&
            h_close[i - 24] > 0.0) {
            ret24[i] = h_close[i] / h_close[i - 24] - 1.0;
        }
    }

    /* Elementwise leaves. */
    for (int p = 0; p < len; ++p) {
        int i = bs + p;
        cpos[i] = NAN;
        rng[i] = NAN;
        if (isfinite(h_high[i]) && isfinite(h_low[i]) && isfinite(h_close[i])) {
            double range = h_high[i] - h_low[i];
            if (range > 0.0) cpos[i] = (h_close[i] - h_low[i]) / range - 0.5;
            if (h_close[i] > 0.0) rng[i] = range / h_close[i];
        }
    }

    /* Wilder ATR14 with the CPU's period-sized seed and gap reset. */
    {
        double sum = 0.0;
        double prev = NAN;
        int count = 0;
        for (int p = 0; p < len; ++p) {
            int i = bs + p;
            if (p > 0 && h_time[i] - h_time[i - 1] != (long long)step) {
                count = 0;
                sum = 0.0;
                prev = NAN;
            }
            double tr;
            if (!(isfinite(h_high[i]) && isfinite(h_low[i]))) {
                count = 0;
                sum = 0.0;
                prev = NAN;
                atrp[i] = NAN;
                continue;
            }
            if (p == 0 || !isfinite(h_close[i - 1]) ||
                h_time[i] - h_time[i - 1] != (long long)step) {
                tr = h_high[i] - h_low[i];
            } else {
                double pc = h_close[i - 1];
                tr = fmax(h_high[i] - h_low[i],
                          fmax(fabs(h_high[i] - pc), fabs(h_low[i] - pc)));
            }
            if (!isfinite(tr)) {
                count = 0;
                sum = 0.0;
                prev = NAN;
                atrp[i] = NAN;
                continue;
            }
            if (count < 14) {
                sum += tr;
                count += 1;
                if (count == 14) {
                    prev = sum / 14.0;
                    atrp[i] = (isfinite(h_close[i]) && h_close[i] > 0.0)
                                  ? prev / h_close[i]
                                  : NAN;
                } else {
                    atrp[i] = NAN;
                }
            } else {
                prev += (tr - prev) / 14.0;
                atrp[i] = (isfinite(h_close[i]) && h_close[i] > 0.0)
                              ? prev / h_close[i]
                              : NAN;
            }
        }
    }

    /* Wilder RSI14, normalised RSI/50 - 1, with the CPU's seed. */
    {
        double gain = 0.0, loss = 0.0;
        double avg_gain = NAN, avg_loss = NAN;
        int count = 0;
        rsi[bs] = NAN;
        for (int p = 1; p < len; ++p) {
            int i = bs + p;
            if (h_time[i] - h_time[i - 1] != (long long)step ||
                !isfinite(h_close[i]) || !isfinite(h_close[i - 1])) {
                gain = 0.0;
                loss = 0.0;
                count = 0;
                avg_gain = NAN;
                avg_loss = NAN;
                rsi[i] = NAN;
                continue;
            }
            double d = h_close[i] - h_close[i - 1];
            double up = fmax(d, 0.0);
            double dn = fmax(-d, 0.0);
            if (!isfinite(avg_gain)) {
                gain += up;
                loss += dn;
                count += 1;
                if (count == 14) {
                    avg_gain = gain / 14.0;
                    avg_loss = loss / 14.0;
                    rsi[i] = rc_rsi_norm(avg_gain, avg_loss);
                } else {
                    rsi[i] = NAN;
                }
            } else {
                avg_gain = (avg_gain * 13.0 + up) / 14.0;
                avg_loss = (avg_loss * 13.0 + dn) / 14.0;
                rsi[i] = rc_rsi_norm(avg_gain, avg_loss);
            }
        }
    }

    /* Donchian position over 20 contiguous higher bars. */
    for (int p = 0; p < len; ++p) {
        int i = bs + p;
        don[i] = NAN;
        if (!rc_contig_local(h_time, bs, p, 20, step)) continue;
        bool finite = true;
        double hi = -INFINITY;
        double lo = INFINITY;
        for (int j = p + 1 - 20; j <= p; ++j) {
            if (!isfinite(h_high[bs + j]) || !isfinite(h_low[bs + j])) {
                finite = false;
                break;
            }
            hi = fmax(hi, h_high[bs + j]);
            lo = fmin(lo, h_low[bs + j]);
        }
        if (finite && hi > lo && isfinite(h_close[i])) {
            don[i] = (h_close[i] - lo) / (hi - lo) - 0.5;
        }
    }

    /* Prefix sums for close (ma_ratio, trend_slope) and ret1 (vol). */
    int pb = bs + s;
    MtfArena ca = {ps, ps2, ps2, ps2, (int *)pc, h_close + bs - pb - 1};
    MtfArena ra = {rs, rs2, rs3, rs4, (int *)rc, ret1 + bs - pb - 1};
    rc_prefix_local(ca, h_close, h_time, bs, len, pb, step, false);
    rc_prefix_local(ra, ret1, h_time, bs, len, pb, step, true);

    for (int p = 0; p < len; ++p) {
        int i = bs + p;
        ma[i] = NAN;
        if (!rc_contig_local(h_time, bs, p, 20, step)) continue;
        double m = rw_mean_strict(ca, pb + 1 + p, 20);
        if (isfinite(m) && m > 0.0) ma[i] = h_close[i] / m - 1.0;
    }

    for (int p = 0; p < len; ++p) {
        int i = bs + p;
        vol[i] = NAN;
        if (p < 20 || !rc_contig_local(h_time, bs, p, 20, step)) continue;
        double sd = rw_std_pop(ra, pb + 1 + p, 20);
        if (isfinite(sd)) vol[i] = sd;
    }

    for (int p = 0; p < len; ++p) {
        int i = bs + p;
        trend[i] = NAN;
        if (p < 39 || !rc_contig_local(h_time, bs, p, 40, step)) continue;
        double now = rw_mean_strict(ca, pb + 1 + p, 20);
        double prev = rw_mean_strict(ca, pb + 1 + p - 20, 20);
        if (isfinite(now) && isfinite(prev) && prev > 0.0) {
            trend[i] = now / prev - 1.0;
        }
    }
}

/* One thread per segment: the monotone causal merge cursor. */
__global__ void k_join(const long long *__restrict__ t,
                       const int *__restrict__ seg_start,
                       const int *__restrict__ seg_len,
                       const int *__restrict__ seg_bs,
                       const int *__restrict__ seg_bl,
                       const long long *__restrict__ h_time,
                       int nseg, long long tf,
                       int *__restrict__ join_idx,
                       double *__restrict__ age)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = seg_start[s];
    int len = seg_len[s];
    int bs = seg_bs[s];
    int be = bs + seg_bl[s];
    int cursor = bs;
    for (int i = start; i < start + len; ++i) {
        long long now = t[i];
        while (cursor < be && h_time[cursor] + tf <= now) ++cursor;
        int j = cursor - 1;
        if (j >= bs) {
            double a = (double)(now - (h_time[j] + tf));
            age[i] = a;
            long long expected = rc_bucket(now, tf) - tf;
            join_idx[i] = (h_time[j] == expected) ? j : -1;
        } else {
            age[i] = NAN;
            join_idx[i] = -1;
        }
    }
}

__global__ void k_gather(const int *__restrict__ join_idx,
                         const double *__restrict__ src,
                         int n, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = join_idx[i];
    out[i] = (j >= 0) ? src[j] : NAN;
}

__global__ void k_copy_age(const double *__restrict__ age, int n,
                           double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = age[i];
}

static int blocks_for(int n, int threads)
{
    return (n + threads - 1) / threads;
}

extern "C" {

void rc_mtf_destroy(RcMtf *c);

RcMtf *rc_mtf_create(const long long *times, const double *open,
                     const double *high, const double *low,
                     const double *close, int n,
                     const int *seg_start, const int *seg_len, int nseg,
                     int step, int max_buckets)
{
    if (n <= 0 || nseg <= 0 || nseg > n || max_buckets <= 0) return NULL;
    RcMtf *c = (RcMtf *)calloc(1, sizeof(RcMtf));
    if (!c) return NULL;
    c->n = n;
    c->nseg = nseg;
    c->step = step;
    c->prepared_tf = -1;

    c->max_buckets = max_buckets > n ? n : max_buckets;

    int ok = 1;
    size_t rows = (size_t)n;
    size_t mb = (size_t)c->max_buckets;
    size_t pref = mb + (size_t)nseg + 2;
    stats_begin(c);
    ok &= rc_alloc((void **)&c->d_times, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_open, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_high, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_low, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_close, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_start, (size_t)nseg * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_len, (size_t)nseg * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_id, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_flags, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_bucket_of_row, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_bucket_lo, mb * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_bucket_start, (size_t)nseg * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_bucket_len, (size_t)nseg * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_h_time, mb * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_h_open, mb * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_h_high, mb * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_h_low, mb * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_h_close, mb * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_leaf, (size_t)RC_MTF_LEAVES * mb * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_ps, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_ps2, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_pc, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_rs, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_rs2, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_rc, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_rs3, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_rs4, pref * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_join_idx, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_age, rows * sizeof(double)) == 0;
    if (!ok) {
        rc_mtf_destroy(c);
        return NULL;
    }

    /* CUB scan temporary storage for the new-bucket flags. */
    c->scan_tmp_bytes = 0;
    ok &= rc_cub_status(cub::DeviceScan::ExclusiveSum(
              NULL, c->scan_tmp_bytes, (int *)NULL, (int *)NULL, n)) ==
          cudaSuccess;
    ok = ok && rc_alloc(&c->d_scan_tmp, c->scan_tmp_bytes) == 0;
    if (!ok) {
        rc_mtf_destroy(c);
        return NULL;
    }

    rc_upload(c->d_times, times, rows * sizeof(long long));
    rc_upload(c->d_open, open, rows * sizeof(double));
    rc_upload(c->d_high, high, rows * sizeof(double));
    rc_upload(c->d_low, low, rows * sizeof(double));
    rc_upload(c->d_close, close, rows * sizeof(double));
    rc_upload(c->d_seg_start, seg_start, (size_t)nseg * sizeof(int));
    rc_upload(c->d_seg_len, seg_len, (size_t)nseg * sizeof(int));

    /* Segment id for the new-bucket rule; the CPU runs the family per symbol. */
    {
        int *ids = (int *)malloc(rows * sizeof(int));
        if (!ids) {
            rc_mtf_destroy(c);
            return NULL;
        }
        for (int s = 0; s < nseg; ++s) {
            int start = seg_start[s];
            int len = seg_len[s];
            for (int j = 0; j < len; ++j) ids[start + j] = s;
        }
        rc_upload(c->d_seg_id, ids, rows * sizeof(int));
        free(ids);
    }

    c->last.h2d_bytes = (unsigned long long)rows * (sizeof(long long) + 5ULL * sizeof(double)) +
                        (unsigned long long)nseg * 2ULL * sizeof(int) +
                        (unsigned long long)rows * sizeof(int);
    stats_end(c);
    return c;
}

void rc_mtf_destroy(RcMtf *c)
{
    if (!c) return;
    cudaFree(c->d_times);
    cudaFree(c->d_open);
    cudaFree(c->d_high);
    cudaFree(c->d_low);
    cudaFree(c->d_close);
    cudaFree(c->d_seg_start);
    cudaFree(c->d_seg_len);
    cudaFree(c->d_seg_id);
    cudaFree(c->d_flags);
    cudaFree(c->d_bucket_of_row);
    cudaFree(c->d_bucket_lo);
    cudaFree(c->d_seg_bucket_start);
    cudaFree(c->d_seg_bucket_len);
    cudaFree(c->d_scan_tmp);
    cudaFree(c->d_h_time);
    cudaFree(c->d_h_open);
    cudaFree(c->d_h_high);
    cudaFree(c->d_h_low);
    cudaFree(c->d_h_close);
    cudaFree(c->d_leaf);
    cudaFree(c->d_ps);
    cudaFree(c->d_ps2);
    cudaFree(c->d_pc);
    cudaFree(c->d_rs);
    cudaFree(c->d_rs2);
    cudaFree(c->d_rc);
    cudaFree(c->d_rs3);
    cudaFree(c->d_rs4);
    cudaFree(c->d_join_idx);
    cudaFree(c->d_age);
    free(c);
}

int rc_mtf_total_buckets(RcMtf *c)
{
    return c ? c->total_buckets : -1;
}

void rc_mtf_last_stats(RcMtf *c, RcMtfStats *out)
{
    if (c && out) *out = c->last;
}

static thread_local int g_force_cub_fail = 0;

extern "C" void rc_mtf_test_force_cub_fail(int on)
{
    g_force_cub_fail = on;
}

/* Resample to one timeframe, compute all eleven higher leaves and the join
   cursor. A later `rc_mtf_gather` is then a pure row map. */
int rc_mtf_prepare(RcMtf *c, long long tf_seconds)
{
    if (!c || tf_seconds <= 0) return 1;
    if (tf_seconds < 3600 || tf_seconds % 3600 != 0) return 6;
    stats_begin(c);
    const int threads = 256;
    int nb = blocks_for(c->n, threads);

    k_bucket_flags<<<nb, threads>>>(c->d_times, c->d_seg_id, c->n,
                                    tf_seconds, c->d_flags);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 2;
    }

    size_t tmp = c->scan_tmp_bytes;
    err = RC_CUB(g_force_cub_fail, cub::DeviceScan::ExclusiveSum(
        c->d_scan_tmp, tmp, c->d_flags, c->d_bucket_of_row, c->n));
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 3;
    }

    /* The last bucket index plus its flag is the bucket count. */
    int tail[2] = {0, 0};
    rc_download(&tail[0], c->d_bucket_of_row + (c->n - 1), sizeof(int));
    rc_download(&tail[1], c->d_flags + (c->n - 1), sizeof(int));
    c->total_buckets = tail[0] + tail[1];
    if (c->total_buckets < 0 || c->total_buckets > c->max_buckets) {
        c->last.failed = 1;
        stats_end(c);
        return 4;
    }
    c->last.d2h_bytes += 2ULL * sizeof(int);

    k_bucket_lo<<<nb, threads>>>(c->d_flags, c->d_bucket_of_row, c->n,
                                 c->d_bucket_lo);
    int bres = c->total_buckets > 0 ? blocks_for(c->total_buckets, threads) : 0;
    if (bres > 0) {
        k_resample<<<bres, threads>>>(
            c->d_times, c->d_open, c->d_high, c->d_low, c->d_close,
            c->d_bucket_lo, c->n, c->total_buckets, tf_seconds,
            c->d_h_time, c->d_h_open, c->d_h_high, c->d_h_low, c->d_h_close);
    }
    int sres = blocks_for(c->nseg, threads);
    k_seg_buckets<<<sres, threads>>>(c->d_seg_start, c->d_bucket_of_row,
                                     c->nseg, c->total_buckets,
                                     c->d_seg_bucket_start,
                                     c->d_seg_bucket_len);
    k_mtf_leaves<<<sres, threads>>>(
        c->d_h_time, c->d_h_open, c->d_h_high, c->d_h_low, c->d_h_close,
        c->d_seg_bucket_start, c->d_seg_bucket_len, c->nseg, (int)tf_seconds,
        c->max_buckets, c->d_ps, c->d_ps2, c->d_pc, c->d_rs, c->d_rs2,
        c->d_rc, c->d_rs3, c->d_rs4, c->d_leaf);
    k_join<<<sres, threads>>>(c->d_times, c->d_seg_start, c->d_seg_len,
                              c->d_seg_bucket_start, c->d_seg_bucket_len,
                              c->d_h_time, c->nseg, tf_seconds,
                              c->d_join_idx, c->d_age);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 5;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 6;
    }
    c->prepared_tf = tf_seconds;
    stats_end(c);
    return 0;
}

/* Gather leaf `leaf` (0..10) or bar_age (11) into a base-length f64 column. */
int rc_mtf_gather(RcMtf *c, int leaf, double *out)
{
    if (!c || !out || c->prepared_tf <= 0) return 1;
    if (leaf < 0 || leaf > RC_MTF_BAR_AGE) return 2;
    stats_begin(c);
    const int threads = 256;
    int nb = blocks_for(c->n, threads);
    if (leaf == RC_MTF_BAR_AGE) {
        k_copy_age<<<nb, threads>>>(c->d_age, c->n, out);
    } else {
        const double *src = c->d_leaf + (size_t)leaf * c->max_buckets;
        k_gather<<<nb, threads>>>(c->d_join_idx, src, c->n, out);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 3;
    }
    cudaDeviceSynchronize();
    c->last.d2h_bytes = (unsigned long long)c->n * sizeof(double);
    stats_end(c);
    return 0;
}

} /* extern "C" */
