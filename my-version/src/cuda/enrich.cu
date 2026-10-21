/*
 * Exact f64 feature enrichment on the GPU (issue #200).
 *
 * Unlike the fitness kernel in `eval.cu`, this backend is not approximate: the
 * CPU enrichment path is the reference and these kernels use the same f64
 * arithmetic. The sequential families (rolling sums, Wilder/EMA scans,
 * Donchian and labels) run one thread per symbol segment and walk the segment
 * in the same order as the CPU, so their columns are bit-identical. The
 * elementwise families use one thread per row. Cross-sectional ranks use a
 * stable segmented radix sort (CUB), preserving the CPU tie order.
 *
 * The source columns are uploaded once per `rc_enrich_create`; every family
 * reuses the same device context, scratch arena and statistics ledger
 * (`RcEnrichStats`, the same windows as #118's `RcStats`).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cub/cub.cuh>

typedef struct RcEnrichStats {
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
} RcEnrichStats;

typedef struct RcEnrich {
    int n;
    int nseg;
    int step;
    long long *d_times;
    double *d_open;
    double *d_high;
    double *d_low;
    double *d_close;
    double *d_volume;
    double *d_qvolume;
    int *d_seg_start;
    int *d_seg_len;
    int *d_seg_id;
    /* Scratch arena, reused by every family. */
    double *d_a;
    double *d_b;
    double *d_c;
    double *d_d;
    double *d_out;
    double *d_out2;
    double *d_out3;
    RcEnrichStats last;
    double batch_start_ms;
} RcEnrich;

static double rc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int rc_alloc(void **p, size_t bytes)
{
#ifdef RCSR_UNIFIED
    return cudaMallocManaged(p, bytes, cudaMemAttachGlobal);
#else
    return cudaMalloc(p, bytes);
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

static void stats_begin(RcEnrich *c)
{
    memset(&c->last, 0, sizeof(c->last));
    c->batch_start_ms = rc_now_ms();
}

static void stats_end(RcEnrich *c)
{
    c->last.wall_ms = rc_now_ms() - c->batch_start_ms;
    c->last.batch = 1;
}

/* ------------------------------------------------------------------------- */
/* Device helpers                                                            */
/* ------------------------------------------------------------------------- */

__device__ __forceinline__ bool rc_finite(double v)
{
    return isfinite(v);
}

__device__ __forceinline__ bool rc_contig(const long long *t, int i, int w, int step)
{
    if (w == 0) return true;
    if (i + 1 < w || step <= 0) return false;
    return t[i + 1 - w] == t[i] - (long long)(w - 1) * step;
}

/* Sequential prefix over one segment. Each segment gets its own base
   (`start + segment_id`) inside the scratch arena so two segment threads never
   write the same boundary slot. */
__device__ void rc_prefix(double *sum, double *sum2, double *cnt,
                          const double *x, int start, int len, int pb)
{
    sum[pb] = 0.0;
    sum2[pb] = 0.0;
    cnt[pb] = 0.0;
    double s = 0.0, s2 = 0.0, c = 0.0;
    for (int j = 0; j < len; ++j) {
        double v = x[start + j];
        if (rc_finite(v)) {
            s += v;
            s2 += v * v;
            c += 1.0;
        }
        sum[pb + j + 1] = s;
        sum2[pb + j + 1] = s2;
        cnt[pb + j + 1] = c;
    }
}

__device__ __forceinline__ double rc_std(const double *sum, const double *sum2,
                                         const double *cnt, int pb, int p, int w)
{
    int lo = pb + p + 1 - w;
    int hi = pb + p + 1;
    if (cnt[hi] - cnt[lo] < w) return NAN;
    double m = (sum[hi] - sum[lo]) / (double)w;
    double s2 = (sum2[hi] - sum2[lo]) / (double)w;
    return sqrt(fmax(s2 - m * m, 0.0));
}

__device__ __forceinline__ double rc_mean(const double *sum, const double *cnt,
                                          int pb, int p, int w)
{
    int lo = pb + p + 1 - w;
    int hi = pb + p + 1;
    if (cnt[hi] - cnt[lo] < w) return NAN;
    return (sum[hi] - sum[lo]) / (double)w;
}

__device__ __forceinline__ int rc_weekday(long long days)
{
    long long r = (days + 4) % 7;
    if (r < 0) r += 7;
    return (int)r;
}

__device__ void rc_civil_from_days(long long z, int *yy, int *mm, int *dd)
{
    z += 719468;
    long long era = (z >= 0 ? z : z - 146096) / 146097;
    long long doe = z - era * 146097;
    long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long long y = yoe + era * 400;
    long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    long long mp = (5 * doy + 2) / 153;
    long long d = doy - (153 * mp + 2) / 5 + 1;
    long long m = mp + (mp < 10 ? 3 : -9);
    *yy = (int)(y + (m <= 2 ? 1 : 0));
    *mm = (int)m;
    *dd = (int)d;
}

/* ------------------------------------------------------------------------- */
/* Elementwise kernels                                                       */
/* ------------------------------------------------------------------------- */

__global__ void k_ret(const long long *__restrict__ t, const double *__restrict__ c,
                      const int *__restrict__ seg, int n, int step, int k,
                      double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = NAN;
    if (i >= k && seg[i] == seg[i - k] &&
        t[i] - t[i - k] == (long long)k * step &&
        c[i] > 0.0 && c[i - k] > 0.0) {
        v = log(c[i] / c[i - k]);
    }
    out[i] = v;
}

__global__ void k_lag(const long long *__restrict__ t, const double *__restrict__ c,
                      const int *__restrict__ seg, int n, int step, int k,
                      double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = NAN;
    int j = i - k;
    if (j >= 1 && seg[i] == seg[j] && t[i] - t[j] == (long long)k * step &&
        t[j] - t[j - 1] == (long long)step) {
        double a = c[j], b = c[j - 1];
        if (a > 0.0 && b > 0.0) v = log(a / b);
    }
    out[i] = v;
}

__global__ void k_range(const double *__restrict__ high, const double *__restrict__ low,
                        const double *__restrict__ close, int n, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = close[i] > 0.0 ? (high[i] - low[i]) / close[i] : NAN;
}

__global__ void k_clv(const double *__restrict__ high, const double *__restrict__ low,
                      const double *__restrict__ close, int n, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double r = high[i] - low[i];
    out[i] = r > 0.0 ? ((close[i] - low[i]) - (high[i] - close[i])) / r : NAN;
}

__global__ void k_calendar(const long long *__restrict__ t, int n, int which,
                           double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long ts = t[i];
    double frac, a;
    const double pi = 3.14159265358979323846264338327950288;
    switch (which) {
    case 0: /* dow_sin */
    case 1: /* dow_cos */
        a = 2.0 * pi * (double)rc_weekday(ts / 86400) / 7.0;
        out[i] = (which == 0) ? sin(a) : cos(a);
        break;
    case 2: /* hour_sin */
    case 3: /* hour_cos */
        frac = (double)(ts % 86400) / 86400.0;
        if (frac < 0.0) frac += 1.0;
        a = 2.0 * pi * frac;
        out[i] = (which == 2) ? sin(a) : cos(a);
        break;
    case 4: /* month_sin */
    case 5: /* month_cos */ {
        int y, m, d;
        rc_civil_from_days(ts / 86400, &y, &m, &d);
        a = 2.0 * pi * (double)m / 12.0;
        out[i] = (which == 4) ? sin(a) : cos(a);
        break;
    }
    default: { /* dom_sin / dom_cos */
        int y, m, d;
        rc_civil_from_days(ts / 86400, &y, &m, &d);
        a = 2.0 * pi * (double)d / 31.0;
        out[i] = (which == 6) ? sin(a) : cos(a);
        break;
    }
    }
}

/* ------------------------------------------------------------------------- */
/* Sequential (one thread per segment) kernels                               */
/* ------------------------------------------------------------------------- */

__global__ void k_ma(const long long *__restrict__ t, const double *__restrict__ close,
                     const int *__restrict__ ss, const int *__restrict__ sl,
                     int nseg, int step, int w,
                     double *sa, double *sb, double *sc, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    int pb = start + s;
    rc_prefix(sa, sb, sc, close, start, len, pb);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = NAN;
        if (p + 1 < w || !rc_contig(t, i, w, step)) continue;
        double m = rc_mean(sa, sc, pb, p, w);
        if (isfinite(m) && m > 0.0) out[i] = close[i] / m - 1.0;
    }
}

__global__ void k_vol(const long long *__restrict__ t, const double *__restrict__ close,
                      const int *__restrict__ ss, const int *__restrict__ sl,
                      int nseg, int step, int w,
                      double *sa, double *sb, double *sc, double *ret, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        double v = NAN;
        if (p > 0 && t[i] - t[i - 1] == (long long)step &&
            close[i] > 0.0 && close[i - 1] > 0.0) {
            v = log(close[i] / close[i - 1]);
        }
        ret[i] = v;
    }
    int pb = start + s;
    rc_prefix(sa, sb, sc, ret, start, len, pb);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = NAN;
        if (p < w || !rc_contig(t, i, w, step)) continue;
        out[i] = rc_std(sa, sb, sc, pb, p, w);
    }
}

__global__ void k_bb_width(const long long *__restrict__ t, const double *__restrict__ close,
                           const int *__restrict__ ss, const int *__restrict__ sl,
                           int nseg, int step, int w,
                           double *sa, double *sb, double *sc, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    int pb = start + s;
    rc_prefix(sa, sb, sc, close, start, len, pb);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = NAN;
        if (p + 1 < w || !rc_contig(t, i, w, step)) continue;
        double m = rc_mean(sa, sc, pb, p, w);
        double sd = rc_std(sa, sb, sc, pb, p, w);
        if (isfinite(m) && isfinite(sd) && m > 0.0) out[i] = 4.0 * sd / m;
    }
}

__global__ void k_vz(const long long *__restrict__ t, const double *__restrict__ volume,
                     const int *__restrict__ ss, const int *__restrict__ sl,
                     int nseg, int step, int w,
                     double *sa, double *sb, double *sc, double *x, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        x[i] = volume[i] >= 0.0 ? log1p(volume[i]) : NAN;
    }
    int pb = start + s;
    rc_prefix(sa, sb, sc, x, start, len, pb);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = NAN;
        if (!rc_contig(t, i, w, step)) continue;
        double m = rc_mean(sa, sc, pb, p, w);
        double sd = rc_std(sa, sb, sc, pb, p, w);
        if (isfinite(m) && isfinite(sd) && sd > 0.0) out[i] = (x[i] - m) / sd;
    }
}

__global__ void k_donchian(const long long *__restrict__ t,
                           const double *__restrict__ high, const double *__restrict__ low,
                           const double *__restrict__ close,
                           const int *__restrict__ ss, const int *__restrict__ sl,
                           int nseg, int step, int w, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = NAN;
        if (p + 1 < w || !rc_contig(t, i, w, step)) continue;
        int lo = i + 1 - w;
        double hi = -INFINITY, ll = INFINITY;
        for (int j = lo; j <= i; ++j) {
            hi = fmax(hi, high[j]);
            ll = fmin(ll, low[j]);
        }
        if (hi > ll && isfinite(ll)) out[i] = (close[i] - ll) / (hi - ll);
    }
}

__global__ void k_ema_ratio(const long long *__restrict__ t, const double *__restrict__ close,
                            const int *__restrict__ ss, const int *__restrict__ sl,
                            int nseg, int step, int span, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    double a = 2.0 / ((double)span + 1.0);
    double prev = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool gap = p > 0 && (t[i] - t[i - 1] != (long long)step);
        if (gap) prev = NAN;
        double v = close[i];
        if (!isfinite(v)) {
            prev = NAN;
            out[i] = NAN;
            continue;
        }
        prev = isfinite(prev) ? a * v + (1.0 - a) * prev : v;
        out[i] = (isfinite(prev) && v > 0.0) ? v / prev - 1.0 : NAN;
    }
}

__global__ void k_macd(const long long *__restrict__ t, const double *__restrict__ close,
                       const int *__restrict__ ss, const int *__restrict__ sl,
                       int nseg, int step, double *macd, double *sig, double *hist)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    double a12 = 2.0 / 13.0, a26 = 2.0 / 27.0;
    double p12 = NAN, p26 = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool gap = p > 0 && (t[i] - t[i - 1] != (long long)step);
        if (gap) {
            p12 = NAN;
            p26 = NAN;
        }
        double v = close[i];
        if (!isfinite(v)) {
            p12 = NAN;
            p26 = NAN;
            macd[i] = NAN;
            continue;
        }
        p12 = isfinite(p12) ? a12 * v + (1.0 - a12) * p12 : v;
        p26 = isfinite(p26) ? a26 * v + (1.0 - a26) * p26 : v;
        macd[i] = (isfinite(p12) && isfinite(p26) && p26 > 0.0) ? p12 / p26 - 1.0 : NAN;
    }
    /* Signal = EMA(9) of MACD, same recurrence as the CPU `ema`. */
    double a = 2.0 / 10.0;
    double prev = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool gap = p > 0 && (t[i] - t[i - 1] != (long long)step);
        if (gap) prev = NAN;
        double v = macd[i];
        if (!isfinite(v)) {
            prev = NAN;
            sig[i] = NAN;
            hist[i] = NAN;
            continue;
        }
        prev = isfinite(prev) ? a * v + (1.0 - a) * prev : v;
        sig[i] = prev;
        hist[i] = (isfinite(macd[i]) && isfinite(prev)) ? macd[i] - prev : NAN;
    }
}

__device__ void rc_wilder_inplace(double *x, const long long *t, int start, int len,
                                  int step, int period)
{
    double a = 1.0 / (double)period;
    double prev = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool gap = p > 0 && (t[i] - t[i - 1] != (long long)step);
        if (gap) prev = NAN;
        double v = x[i];
        if (!isfinite(v)) continue;
        prev = isfinite(prev) ? prev + a * (v - prev) : v;
        x[i] = prev;
    }
}

__global__ void k_rsi_py(const long long *__restrict__ t, const double *__restrict__ close,
                         const int *__restrict__ ss, const int *__restrict__ sl,
                         int nseg, int step, int period,
                         double *d, double *up, double *dn, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        double delta = 0.0;
        if (p > 0 && isfinite(close[i]) && isfinite(close[i - 1])) {
            delta = close[i] - close[i - 1];
        }
        d[i] = delta;
        up[i] = fmax(delta, 0.0);
        dn[i] = fmax(-delta, 0.0);
    }
    rc_wilder_inplace(up, t, start, len, step, period);
    rc_wilder_inplace(dn, t, start, len, step, period);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        if (!(isfinite(up[i]) && isfinite(dn[i])) || dn[i] <= 0.0) {
            out[i] = NAN;
        } else {
            out[i] = (100.0 - 100.0 / (1.0 + up[i] / dn[i])) / 50.0 - 1.0;
        }
    }
}

__global__ void k_atr_py(const long long *__restrict__ t, const double *__restrict__ close,
                         const double *__restrict__ high, const double *__restrict__ low,
                         const int *__restrict__ ss, const int *__restrict__ sl,
                         int nseg, int step, int period,
                         double *tr, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        double v = NAN;
        if (isfinite(high[i]) && isfinite(low[i])) {
            if (p == 0 || !isfinite(close[i - 1])) {
                v = high[i] - low[i];
            } else {
                double pc = close[i - 1];
                v = fmax(high[i] - low[i], fmax(fabs(high[i] - pc), fabs(low[i] - pc)));
            }
        }
        tr[i] = v;
    }
    rc_wilder_inplace(tr, t, start, len, step, period);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = (isfinite(tr[i]) && close[i] > 0.0) ? tr[i] / close[i] : NAN;
    }
}

__global__ void k_label(const long long *__restrict__ t, const double *__restrict__ open,
                        const double *__restrict__ close,
                        const int *__restrict__ ss, const int *__restrict__ sl,
                        int nseg, int step, int horizon, double cost,
                        int vol_window, int stride, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    int end_seg = start + len;
    double cost2 = 2.0 * cost / 10000.0;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = NAN;
        if (p + horizon >= len) continue;
        int end = i + horizon;
        if (end >= end_seg) end = end_seg - 1;
        bool gap = false;
        for (int j = i + 1; j <= end; ++j) {
            /* matches ohlcv::is_gap_step: a longer bucket is a gap */
            if (t[j] - t[j - 1] > (long long)step) {
                gap = true;
                break;
            }
        }
        if (gap) continue;
        if (stride > 1 && (p % stride) != 0) continue;
        double entry = open[i + 1];
        double exit = close[i + horizon];
        if (!(entry > 0.0 && exit > 0.0)) continue;
        double y = log(exit / entry) - cost2;
        if (vol_window > 1) {
            int lo = p + 1 - vol_window;
            if (lo < 0) lo = 0;
            /* trailing population std of log returns, matching features.rs */
            double sum = 0.0;
            int cnt = 0;
            bool ok = true;
            for (int j = lo; j <= p; ++j) {
                int k = start + j;
                double r = NAN;
                if (j > 0 && close[k] > 0.0 && close[k - 1] > 0.0) {
                    r = log(close[k] / close[k - 1]);
                }
                if (!isfinite(r)) {
                    ok = false;
                    break;
                }
                sum += r;
                cnt += 1;
            }
            if (!ok || cnt == 0) {
                continue;
            }
            double m = sum / (double)cnt;
            double ssq = 0.0;
            for (int j = lo; j <= p; ++j) {
                int k = start + j;
                double r = log(close[k] / close[k - 1]);
                ssq += (r - m) * (r - m);
            }
            double sd = sqrt(ssq / (double)cnt);
            if (sd > 0.0) y /= sd;
            else continue;
        }
        out[i] = y;
    }
}

/* ------------------------------------------------------------------------- */
/* Cross-sectional rank                                                      */
/* ------------------------------------------------------------------------- */

__device__ __forceinline__ unsigned long long rc_sort_key(double v)
{
    if (isnan(v)) return 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long bits;
    memcpy(&bits, &v, sizeof(bits));
    if (bits & 0x8000000000000000ULL) return ~bits;
    return bits | 0x8000000000000000ULL;
}

__global__ void k_count_finite(const unsigned long long *__restrict__ keys,
                               const int *__restrict__ off, int ngroups,
                               unsigned int *__restrict__ counts)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    unsigned int c = 0;
    for (int p = off[g]; p < off[g + 1]; ++p) {
        if (keys[p] != 0xFFFFFFFFFFFFFFFFULL) c += 1;
    }
    counts[g] = c;
}

__global__ void k_rank_scatter(const unsigned long long *__restrict__ keys,
                               const unsigned int *__restrict__ rows,
                               const int *__restrict__ off,
                               const unsigned int *__restrict__ counts,
                               const unsigned int *__restrict__ gid,
                               int n, double *__restrict__ out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    unsigned int g = gid[p];
    unsigned int finite = counts[g];
    unsigned int pos = (unsigned int)p - (unsigned int)off[g];
    double v = NAN;
    if (pos < finite && finite >= 2 &&
        keys[p] != 0xFFFFFFFFFFFFFFFFULL) {
        v = (double)pos / (double)(finite - 1);
    }
    out[rows[p]] = v;
}

/* ------------------------------------------------------------------------- */
/* C API                                                                     */
/* ------------------------------------------------------------------------- */

/* Stats slot for the rank call, which does not own an RcEnrich context. */
static RcEnrichStats g_rank_last;

extern "C" void rc_enrich_destroy(void *v);

extern "C" {

void *rc_enrich_create(const long long *times, const double *open, const double *high,
                       const double *low, const double *close, const double *volume,
                       const double *qvolume, int n, const int *seg_start,
                       const int *seg_len, int nseg, int step)
{
    if (n <= 0 || nseg <= 0) return NULL;
    RcEnrich *c = (RcEnrich *)calloc(1, sizeof(RcEnrich));
    if (!c) return NULL;
    c->n = n;
    c->nseg = nseg;
    c->step = step;
    size_t rows = (size_t)n;
    double t0 = rc_now_ms();
    int ok = 1;
    if (rc_alloc((void **)&c->d_times, rows * sizeof(long long)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_open, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_high, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_low, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_close, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_volume, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_qvolume, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_start, (size_t)nseg * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_len, (size_t)nseg * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_id, rows * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_a, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_b, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_c, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_d, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_out, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_out2, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_out3, rows * sizeof(double)) != 0) ok = 0;
    c->last.alloc_ms = rc_now_ms() - t0;
    if (!ok) {
        rc_enrich_destroy(c);
        return NULL;
    }
    t0 = rc_now_ms();
    rc_upload(c->d_times, times, rows * sizeof(long long));
    rc_upload(c->d_open, open, rows * sizeof(double));
    rc_upload(c->d_high, high, rows * sizeof(double));
    rc_upload(c->d_low, low, rows * sizeof(double));
    rc_upload(c->d_close, close, rows * sizeof(double));
    rc_upload(c->d_volume, volume, rows * sizeof(double));
    rc_upload(c->d_qvolume, qvolume, rows * sizeof(double));
    rc_upload(c->d_seg_start, seg_start, (size_t)nseg * sizeof(int));
    rc_upload(c->d_seg_len, seg_len, (size_t)nseg * sizeof(int));
    c->last.h2d_ms = rc_now_ms() - t0;
    c->last.h2d_bytes = 8ULL * (unsigned long long)rows * 8ULL +
                        8ULL * (unsigned long long)nseg;
    /* Segment id per row, derived once on the host and uploaded with the rest
       of the resident data. It lets elementwise kernels refuse to cross a
       symbol boundary without a per-row lookup table on the device. */
    int *seg_id = (int *)malloc(rows * sizeof(int));
    if (!seg_id) {
        rc_enrich_destroy(c);
        return NULL;
    }
    for (int s = 0; s < nseg; ++s) {
        for (int j = 0; j < seg_len[s]; ++j) seg_id[seg_start[s] + j] = s;
    }
    rc_upload(c->d_seg_id, seg_id, rows * sizeof(int));
    free(seg_id);
    c->last.h2d_bytes += (unsigned long long)rows * sizeof(int);
    stats_end(c);
    c->last.batch = 0;
    return c;
}

void rc_enrich_destroy(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return;
    cudaFree(c->d_times);
    cudaFree(c->d_open);
    cudaFree(c->d_high);
    cudaFree(c->d_low);
    cudaFree(c->d_close);
    cudaFree(c->d_volume);
    cudaFree(c->d_qvolume);
    cudaFree(c->d_seg_start);
    cudaFree(c->d_seg_len);
    cudaFree(c->d_seg_id);
    cudaFree(c->d_a);
    cudaFree(c->d_b);
    cudaFree(c->d_c);
    cudaFree(c->d_d);
    cudaFree(c->d_out);
    cudaFree(c->d_out2);
    cudaFree(c->d_out3);
    free(c);
}

void rc_enrich_last_stats(void *v, RcEnrichStats *out)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || !out) return;
    *out = c->last;
}

#define RC_ELEM_BEGIN(c, blocks, threads)                                     \
    do {                                                                      \
        stats_begin(c);                                                       \
        cudaEvent_t ev0, ev1;                                                 \
        cudaEventCreate(&ev0);                                                \
        cudaEventCreate(&ev1);                                                \
        cudaEventRecord(ev0, 0);                                              \
        double _l = rc_now_ms();

#define RC_ELEM_LAUNCH(c, call)                                               \
        call;                                                                 \
        c->last.launch_ms += rc_now_ms() - _l;                                \
        double _s = rc_now_ms();                                              \
        cudaEventRecord(ev1, 0);                                              \
        cudaEventSynchronize(ev1);                                            \
        c->last.sync_ms += rc_now_ms() - _s;                                  \
        float _ms = 0.0f;                                                     \
        cudaEventElapsedTime(&_ms, ev0, ev1);                                 \
        c->last.kernel_ms += _ms;                                             \
        cudaEventDestroy(ev0);                                                \
        cudaEventDestroy(ev1);                                                \
    } while (0)

#define RC_DOWNLOAD(c, dst, src)                                              \
    do {                                                                      \
        double _d = rc_now_ms();                                              \
        rc_download(dst, src, (size_t)c->n * sizeof(double));                 \
        c->last.d2h_ms += rc_now_ms() - _d;                                   \
        c->last.d2h_bytes += (unsigned long long)c->n * sizeof(double);       \
    } while (0)

#define RC_END(c)                                                             \
    do {                                                                      \
        stats_end(c);                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        c->last.failed = (err != cudaSuccess) ? 1 : 0;                        \
        return c->last.failed ? -1 : 0;                                       \
    } while (0)

int rc_enrich_ret(void *v, int k, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_ret<<<blocks, threads>>>(c->d_times, c->d_close, c->d_seg_id,
                                                  c->n, c->step, k, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_lag(void *v, int k, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_lag<<<blocks, threads>>>(c->d_times, c->d_close, c->d_seg_id,
                                                  c->n, c->step, k, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_range(void *v, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_range<<<blocks, threads>>>(c->d_high, c->d_low, c->d_close,
                                                    c->n, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_clv(void *v, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_clv<<<blocks, threads>>>(c->d_high, c->d_low, c->d_close,
                                                  c->n, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_calendar(void *v, int which, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_calendar<<<blocks, threads>>>(c->d_times, c->n, which, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_ma(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_ma<<<blocks, threads>>>(c->d_times, c->d_close,
                                                 c->d_seg_start, c->d_seg_len, c->nseg,
                                                 c->step, w, c->d_a, c->d_b, c->d_c, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_vol(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_vol<<<blocks, threads>>>(c->d_times, c->d_close,
                                                  c->d_seg_start, c->d_seg_len, c->nseg,
                                                  c->step, w, c->d_a, c->d_b, c->d_c,
                                                  c->d_d, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_bb_width(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_bb_width<<<blocks, threads>>>(c->d_times, c->d_close,
                                                       c->d_seg_start, c->d_seg_len, c->nseg,
                                                       c->step, w, c->d_a, c->d_b, c->d_c,
                                                       c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_vz(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_vz<<<blocks, threads>>>(c->d_times, c->d_volume,
                                                 c->d_seg_start, c->d_seg_len, c->nseg,
                                                 c->step, w, c->d_a, c->d_b, c->d_c,
                                                 c->d_d, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_donchian(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_donchian<<<blocks, threads>>>(c->d_times, c->d_high, c->d_low,
                                                       c->d_close, c->d_seg_start,
                                                       c->d_seg_len, c->nseg, c->step, w,
                                                       c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_ema(void *v, int span, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_ema_ratio<<<blocks, threads>>>(c->d_times, c->d_close,
                                                        c->d_seg_start, c->d_seg_len,
                                                        c->nseg, c->step, span, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_macd(void *v, double *macd, double *sig, double *hist)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_macd<<<blocks, threads>>>(c->d_times, c->d_close,
                                                   c->d_seg_start, c->d_seg_len, c->nseg,
                                                   c->step, c->d_out, c->d_out2, c->d_out3)));
    RC_DOWNLOAD(c, macd, c->d_out);
    RC_DOWNLOAD(c, sig, c->d_out2);
    RC_DOWNLOAD(c, hist, c->d_out3);
    RC_END(c);
}

int rc_enrich_rsi_py(void *v, int period, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_rsi_py<<<blocks, threads>>>(c->d_times, c->d_close,
                                                     c->d_seg_start, c->d_seg_len, c->nseg,
                                                     c->step, period, c->d_a, c->d_b,
                                                     c->d_c, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_atr_py(void *v, int period, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_atr_py<<<blocks, threads>>>(c->d_times, c->d_close, c->d_high,
                                                     c->d_low, c->d_seg_start, c->d_seg_len,
                                                     c->nseg, c->step, period, c->d_a,
                                                     c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_label(void *v, int horizon, double cost_bps, int vol_window, int stride,
                    double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    int threads = 128, blocks = (c->nseg + threads - 1) / threads;
    RC_ELEM_LAUNCH(c, (k_label<<<blocks, threads>>>(c->d_times, c->d_open, c->d_close,
                                                    c->d_seg_start, c->d_seg_len, c->nseg,
                                                    c->step, horizon, cost_bps, vol_window,
                                                    stride, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

/* Stable segmented rank. `keys`, `rows`, `offsets` and `gid` are host arrays
   built in the same (time, symbol, row) order as the CPU reference; the sort
   is stable, so equal values keep that tie order. */
int rc_enrich_rank(const unsigned long long *keys, const unsigned int *rows,
                   const int *offsets, const unsigned int *gid, int n, int ngroups,
                   double *out)
{
    double t0 = rc_now_ms();
    RcEnrichStats st;
    memset(&st, 0, sizeof(st));
    unsigned long long *d_keys = NULL, *d_keys_out = NULL;
    unsigned int *d_rows = NULL, *d_rows_out = NULL, *d_counts = NULL, *d_gid = NULL;
    int *d_off = NULL;
    void *d_temp = NULL;
    double *d_out = NULL;
    size_t temp_bytes = 0;
    int ok = 1;
    if (rc_alloc((void **)&d_keys, (size_t)n * sizeof(unsigned long long)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_keys_out, (size_t)n * sizeof(unsigned long long)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_rows, (size_t)n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_rows_out, (size_t)n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_off, (size_t)(ngroups + 1) * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_gid, (size_t)n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_counts, (size_t)ngroups * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&d_out, (size_t)n * sizeof(double)) != 0) ok = 0;
    st.alloc_ms = rc_now_ms() - t0;
    if (ok) {
        t0 = rc_now_ms();
        rc_upload(d_keys, keys, (size_t)n * sizeof(unsigned long long));
        rc_upload(d_rows, rows, (size_t)n * sizeof(unsigned int));
        rc_upload(d_off, offsets, (size_t)(ngroups + 1) * sizeof(int));
        rc_upload(d_gid, gid, (size_t)n * sizeof(unsigned int));
        st.h2d_ms = rc_now_ms() - t0;
        st.h2d_bytes = (unsigned long long)n * (sizeof(unsigned long long) +
                                                sizeof(unsigned int) +
                                                sizeof(unsigned int)) +
                       (unsigned long long)(ngroups + 1) * sizeof(int);
        cub::DeviceSegmentedRadixSort::SortPairs(
            NULL, temp_bytes, d_keys, d_keys_out, d_rows, d_rows_out,
            n, ngroups, d_off, d_off + 1);
        t0 = rc_now_ms();
        if (rc_alloc(&d_temp, temp_bytes) != 0) {
            ok = 0;
        } else {
            st.alloc_ms += rc_now_ms() - t0;
            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0, 0);
            double l = rc_now_ms();
            cub::DeviceSegmentedRadixSort::SortPairs(
                d_temp, temp_bytes, d_keys, d_keys_out, d_rows, d_rows_out,
                n, ngroups, d_off, d_off + 1);
            st.launch_ms = rc_now_ms() - l;
            double sy = rc_now_ms();
            cudaEventRecord(ev1, 0);
            cudaEventSynchronize(ev1);
            st.sync_ms = rc_now_ms() - sy;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            st.kernel_ms += ms;
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }
    }
    if (ok) {
        int threads = 128;
        int cblocks = (ngroups + threads - 1) / threads;
        k_count_finite<<<cblocks, threads>>>(d_keys_out, d_off, ngroups, d_counts);
        int rblocks = (n + threads - 1) / threads;
        k_rank_scatter<<<rblocks, threads>>>(d_keys_out, d_rows_out, d_off, d_counts,
                                             d_gid, n, d_out);
        cudaDeviceSynchronize();
        t0 = rc_now_ms();
        rc_download(out, d_out, (size_t)n * sizeof(double));
        st.d2h_ms = rc_now_ms() - t0;
        st.d2h_bytes = (unsigned long long)n * sizeof(double);
        st.batch = 1;
    } else {
        st.failed = 1;
    }
    st.wall_ms = rc_now_ms() - t0;
    cudaFree(d_keys);
    cudaFree(d_keys_out);
    cudaFree(d_rows);
    cudaFree(d_rows_out);
    cudaFree(d_off);
    cudaFree(d_gid);
    cudaFree(d_counts);
    cudaFree(d_out);
    cudaFree(d_temp);
    /* The caller reads this through a small host shim. */
    g_rank_last = st;
    return ok ? 0 : -1;
}

void rc_enrich_rank_stats(RcEnrichStats *out)
{
    if (out) *out = g_rank_last;
}

} /* extern "C" */
