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
#include "alloc_guard.h"
#include "cub_guard.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cub/cub.cuh>

#include <atomic>
#include <map>
#include <mutex>

#include "enrich_common.h"
#include "win_stats.h"

/* #300: hard cap for the per-family output arena and the pinned D2H mirror. */
#define RC_BATCH_ARENA_BYTES (64ULL * 1024ULL * 1024ULL)
#define RC_BATCH_CHUNK_BYTES (4ULL * 1024ULL * 1024ULL)

/* #300: operation descriptors for the batched output path. The parameter
   meanings match the single-column entry points below. */
enum {
    RC_OP_RET = 1,
    RC_OP_LAG,
    RC_OP_RANGE,
    RC_OP_CLV,
    RC_OP_CALENDAR,
    RC_OP_MA,
    RC_OP_VOL,
    RC_OP_BB_WIDTH,
    RC_OP_VZ,
    RC_OP_DONCHIAN,
    RC_OP_EMA,
    RC_OP_RSI,
    RC_OP_ATR,
    RC_OP_LABEL,
    RC_OP_MACD
};

typedef struct RcEnrichOp {
    int kind;
    int a;
    int b;
    int c;
    double param;
} RcEnrichOp;

static double rc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static std::mutex g_budget_mu;
static size_t g_budget = 0;
static size_t g_budget_live = 0;
static std::map<void *, size_t> g_output_bytes;

extern "C" int rc_budget_take(size_t bytes)
{
    std::lock_guard<std::mutex> g(g_budget_mu);
    if (g_budget && g_budget_live + bytes > g_budget) return -1;
    g_budget_live += bytes;
    return 0;
}

extern "C" void rc_budget_give(size_t bytes)
{
    std::lock_guard<std::mutex> g(g_budget_mu);
    g_budget_live = g_budget_live >= bytes ? g_budget_live - bytes : 0;
}

extern "C" void rc_enrich_set_test_budget(size_t bytes)
{
    std::lock_guard<std::mutex> g(g_budget_mu);
    g_budget = bytes;
    g_budget_live = 0;
}

/* Test-only fault injection. Allocation faults refuse the k-th allocation
   made through the guard after arming; launch faults make one launch site
   return failure before any kernel is queued. Both are off by default. */
volatile int rc_fault_alloc_armed = 0;
static std::atomic<unsigned long long> g_fault_alloc_count(0);
static std::atomic<unsigned long long> g_fault_alloc_at(0);
static volatile int g_fault_launch_site = 0;
static std::atomic<unsigned long long> g_fault_launch_hits(0);

enum {
    RC_FAULT_SEG_ID = 1,
    RC_FAULT_RING_FEATURE = 2,
    RC_FAULT_BATCH = 3,
    RC_FAULT_XS_PREPARE = 4,
    RC_FAULT_XS_RANK = 5,
    RC_FAULT_ELEMENT = 6
};

extern "C" int rc_fault_alloc_hit(void)
{
    unsigned long long n = g_fault_alloc_count.fetch_add(1) + 1;
    return n == g_fault_alloc_at.load();
}

extern "C" void rc_enrich_set_test_alloc_fault(unsigned long long at)
{
    g_fault_alloc_count = 0;
    g_fault_alloc_at = at;
    rc_fault_alloc_armed = at != 0;
}

extern "C" unsigned long long rc_enrich_test_alloc_count(void)
{
    return g_fault_alloc_count.load();
}

extern "C" void rc_enrich_set_test_launch_fault(int site)
{
    g_fault_launch_hits = 0;
    g_fault_launch_site = site;
}

static int rc_launch_fault(int site)
{
    if (g_fault_launch_site != site) return 0;
    g_fault_launch_hits.fetch_add(1);
    return 1;
}

extern "C" unsigned long long rc_enrich_test_launch_hits(void)
{
    return g_fault_launch_hits.load();
}

#define RC_LAUNCH_FAULT(site) rc_launch_fault(site)


static thread_local int g_force_cub_fail = 0;

extern "C" void rc_enrich_test_force_cub_fail(int on)
{
    g_force_cub_fail = on;
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

/* #364: window-statistic arena. Slot `pb + 1 + p` holds the running sums of
   segment row `p`; slot `pb` is the zero entry before the first row. Each
   segment owns `len + 1` slots starting at `pb = start + segment_id`. */
struct RwArena {
    double *s1;
    double *c1;
    double *s2;
    double *c2;
    int *cn;
    int *bk;
};

struct RwArenaView {
    RwArena a;
    const double *xp; /* x at slot q is xp[q] */
    __device__ double s(int k, int q) const { return k == 0 ? a.s1[q] : a.s2[q]; }
    __device__ double c(int k, int q) const { return k == 0 ? a.c1[q] : a.c2[q]; }
    __device__ int bk(int q) const { return a.bk[q]; }
    __device__ int cn(int q) const { return a.cn[q]; }
    __device__ double x(int q) const { return xp[q]; }
};

/* Kernel A: Sum2 (and Dot2 of x*x when `sq`) prefix of one segment, one thread
   per RW_BLOCK-row block. The accumulation restarts at every block start and
   gap row, so blocks are independent except for the finite-cell count, which
   runs across the whole segment: a block pass counts, one thread scans the
   block counts, then every block is built from its offset. One CTA per
   segment; the block count lives in the `cn` slot of the block's first row
   until the block is built. */
__global__ void k_rw_pre(const double *__restrict__ x, const long long *__restrict__ t,
                         const int *__restrict__ ss, const int *__restrict__ sl,
                         int step, bool sq, RwArena a)
{
    int s = blockIdx.x;
    int start = ss[s], len = sl[s];
    int pb = start + s;
    int nb = (len + RW_BLOCK - 1) / RW_BLOCK;
    if (threadIdx.x == 0) {
        a.s1[pb] = 0.0;
        a.c1[pb] = 0.0;
        a.s2[pb] = 0.0;
        a.c2[pb] = 0.0;
        a.cn[pb] = 0;
        a.bk[pb] = 0;
    }
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int p1 = min(len, p0 + RW_BLOCK);
        int cnt = 0;
        for (int p = p0; p < p1; ++p) cnt += isfinite(x[start + p]) ? 1 : 0;
        a.cn[pb + 1 + p0] = cnt;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int b = 0; b < nb; ++b) {
            int slot = pb + 1 + b * RW_BLOCK;
            int cnt = a.cn[slot];
            a.cn[slot] = run;
            run += cnt;
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int p1 = min(len, p0 + RW_BLOCK);
        RwAcc acc;
        rw_acc_init(&acc);
        acc.cnt = a.cn[pb + 1 + p0];
        for (int p = p0; p < p1; ++p) {
            int i = start + p;
            int q = pb + 1 + p;
            bool restart = (p % RW_BLOCK) == 0 || (p > 0 && t[i] - t[i - 1] != (long long)step);
            rw_acc_push(&acc, x[i], q, restart, sq);
            a.s1[q] = acc.s1;
            a.c1[q] = acc.c1;
            if (sq) {
                a.s2[q] = acc.s2;
                a.c2[q] = acc.c2;
            }
            a.cn[q] = acc.cnt;
            a.bk[q] = q - acc.start;
        }
    }
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
    if (!(isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i]))) {
        out[i] = NAN;
        return;
    }
    double r = high[i] - low[i];
    // Compact definition: (C-L)/(H-L) - 1/2; H==L uses the compact -1/2.
    out[i] = r > 0.0 ? (close[i] - low[i]) / r - 0.5 : -0.5;
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
/* Tiled elementwise kernels (#294 direct projected upload)                  */
/* ------------------------------------------------------------------------- */

/* The same arithmetic as k_ret/k_lag/k_range/k_clv/k_calendar, restricted to
   [start, start+len). The transport runs one of these per uploaded group on
   the compute stream, so the H2D copy of group g+1 overlaps the kernel of g.
   State (close/times/seg_id) before `start` is already resident and ready. */

__global__ void k_ret_tile(const long long *__restrict__ t, const double *__restrict__ c,
                           const int *__restrict__ seg, int step, int k,
                           int start, int len, double *__restrict__ out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= len) return;
    int i = start + p;
    double v = NAN;
    if (i >= k && seg[i] == seg[i - k] &&
        t[i] - t[i - k] == (long long)k * step &&
        c[i] > 0.0 && c[i - k] > 0.0) {
        v = log(c[i] / c[i - k]);
    }
    out[i] = v;
}

__global__ void k_lag_tile(const long long *__restrict__ t, const double *__restrict__ c,
                           const int *__restrict__ seg, int step, int k,
                           int start, int len, double *__restrict__ out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= len) return;
    int i = start + p;
    double v = NAN;
    int j = i - k;
    if (j >= 1 && seg[i] == seg[j] && t[i] - t[j] == (long long)k * step &&
        t[j] - t[j - 1] == (long long)step) {
        double a = c[j], b = c[j - 1];
        if (a > 0.0 && b > 0.0) v = log(a / b);
    }
    out[i] = v;
}

__global__ void k_range_tile(const double *__restrict__ high, const double *__restrict__ low,
                             const double *__restrict__ close, int start, int len,
                             double *__restrict__ out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= len) return;
    int i = start + p;
    out[i] = close[i] > 0.0 ? (high[i] - low[i]) / close[i] : NAN;
}

__global__ void k_clv_tile(const double *__restrict__ high, const double *__restrict__ low,
                           const double *__restrict__ close, int start, int len,
                           double *__restrict__ out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= len) return;
    int i = start + p;
    if (!(isfinite(high[i]) && isfinite(low[i]) && isfinite(close[i]))) {
        out[i] = NAN;
        return;
    }
    double r = high[i] - low[i];
    out[i] = r > 0.0 ? (close[i] - low[i]) / r - 0.5 : -0.5;
}

__global__ void k_calendar_tile(const long long *__restrict__ t, int which,
                                int start, int len, double *__restrict__ out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= len) return;
    int i = start + p;
    long long ts = t[i];
    double frac, a;
    const double pi = 3.14159265358979323846264338327950288;
    switch (which) {
    case 0:
    case 1:
        a = 2.0 * pi * (double)rc_weekday(ts / 86400) / 7.0;
        out[i] = (which == 0) ? sin(a) : cos(a);
        break;
    case 2:
    case 3:
        frac = (double)(ts % 86400) / 86400.0;
        if (frac < 0.0) frac += 1.0;
        a = 2.0 * pi * frac;
        out[i] = (which == 2) ? sin(a) : cos(a);
        break;
    case 4:
    case 5: {
        int y, m, d;
        rc_civil_from_days(ts / 86400, &y, &m, &d);
        a = 2.0 * pi * (double)m / 12.0;
        out[i] = (which == 4) ? sin(a) : cos(a);
        break;
    }
    default: {
        int y, m, d;
        rc_civil_from_days(ts / 86400, &y, &m, &d);
        a = 2.0 * pi * (double)d / 31.0;
        out[i] = (which == 6) ? sin(a) : cos(a);
        break;
    }
    }
}

/* Segment id for [start, start+len), replacing the host loop that used to build
   it before upload. */
__global__ void k_seg_id_tile(const int *__restrict__ ss, const int *__restrict__ sl,
                              int nseg, int start, int len, int *__restrict__ seg)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= len) return;
    int i = start + p;
    int s = 0;
    for (int k = 0; k < nseg; ++k) {
        if (i >= ss[k] && i < ss[k] + sl[k]) { s = k; break; }
    }
    seg[i] = s;
}

/* ------------------------------------------------------------------------- */
/* Window statistics: per-block prefix (k_rw_pre), per-row evaluation       */
/* ------------------------------------------------------------------------- */

__global__ void k_ma(const long long *__restrict__ t, const double *__restrict__ close,
                     const int *__restrict__ seg_id, const int *__restrict__ ss,
                     int n, int step, int w, int mode, RwArena ar,
                     double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = seg_id[i];
    int start = ss[s];
    int p = i - start;
    int pb = start + s;
    RwArenaView v = {ar, close + start - pb - 1};
    out[i] = NAN;
    if (p + 1 < w || !rc_contig(t, i, w, step)) return;
    int hi = pb + 1 + p;
    if (mode == RC_WIN_MEAN_FINITE) {
        /* Compact engine columns ma96/ma672 average the finite closes of
           the window. */
        double m = rw_mean_finite(v, hi, w);
        if (!(close[i] > 0.0)) return;
        if (isfinite(close[i]) && isfinite(m) && m != 0.0) out[i] = close[i] / m - 1.0;
        return;
    }
    /* The catalogue maN columns need the full window. */
    double m = rw_mean_strict(v, hi, w);
    if (isfinite(m) && m > 0.0) out[i] = close[i] / m - 1.0;
}

__global__ void k_vol_ret(const long long *__restrict__ t, const double *__restrict__ close,
                          const int *__restrict__ seg_id, const int *__restrict__ ss,
                          int n, int step, double *__restrict__ ret)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int p = i - ss[seg_id[i]];
    double v = NAN;
    if (p > 0 && t[i] - t[i - 1] == (long long)step &&
        close[i] > 0.0 && close[i - 1] > 0.0) {
        v = log(close[i] / close[i - 1]);
    }
    ret[i] = v;
}

__global__ void k_vol(const long long *__restrict__ t, const double *__restrict__ ret,
                      const int *__restrict__ seg_id, const int *__restrict__ ss,
                      int n, int step, int w, RwArena ar, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = seg_id[i];
    int start = ss[s];
    int p = i - start;
    int pb = start + s;
    RwArenaView v = {ar, ret + start - pb - 1};
    out[i] = NAN;
    if (p < w || !rc_contig(t, i, w, step)) return;
    out[i] = rw_std_pop(v, pb + 1 + p, w);
}

__global__ void k_bb_width(const long long *__restrict__ t, const double *__restrict__ close,
                           const int *__restrict__ seg_id, const int *__restrict__ ss,
                           int n, int step, int w, RwArena ar,
                           double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = seg_id[i];
    int start = ss[s];
    int p = i - start;
    int pb = start + s;
    RwArenaView v = {ar, close + start - pb - 1};
    out[i] = NAN;
    if (p + 1 < w || !rc_contig(t, i, w, step)) return;
    int hi = pb + 1 + p;
    double m = rw_mean_strict(v, hi, w);
    double sd = rw_std_pop(v, hi, w);
    if (isfinite(m) && isfinite(sd) && m > 0.0) out[i] = 4.0 * sd / m;
}

__global__ void k_vz_x(const double *__restrict__ volume, int n, double *__restrict__ x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = volume[i] >= 0.0 ? log1p(volume[i]) : NAN;
}

__global__ void k_vz(const long long *__restrict__ t, const double *__restrict__ x,
                     const int *__restrict__ seg_id, const int *__restrict__ ss,
                     int n, int step, int w, RwArena ar, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = seg_id[i];
    int start = ss[s];
    int p = i - start;
    int pb = start + s;
    RwArenaView v = {ar, x + start - pb - 1};
    out[i] = NAN;
    if (!rc_contig(t, i, w, step)) return;
    int hi = pb + 1 + p;
    double m = rw_mean_strict(v, hi, w);
    double sd = rw_std_pop(v, hi, w);
    if (isfinite(m) && isfinite(sd) && sd > 0.0) out[i] = (x[i] - m) / sd;
}

/* ------------------------------------------------------------------------- */
/* Sequential (one thread per segment) kernels                               */
/* ------------------------------------------------------------------------- */

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
        if (p > 0 && t[i] - t[i - 1] != (long long)step) {
            /* The close-to-close move spans the hole; it is missing, not zero. */
            delta = NAN;
        } else if (p > 0 && isfinite(close[i]) && isfinite(close[i - 1])) {
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
            if (p == 0 || !isfinite(close[i - 1]) ||
                t[i] - t[i - 1] != (long long)step) {
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
                bool gap_r = j > 0 && (t[k] - t[k - 1] != (long long)step);
                if (j > 0 && !gap_r && close[k] > 0.0 && close[k - 1] > 0.0) {
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

#define RC_RANK_INVALID 0xFFFFFFFFFFFFFFFFULL

/* Order key for an f64 that matches the CPU's `partial_cmp` on finite values:
   -0.0 and +0.0 are the same tie, and NaN/invalid rows sort last. */
__device__ __forceinline__ unsigned long long rc_sort_key(double v)
{
    if (isnan(v)) return RC_RANK_INVALID;
    unsigned long long bits;
    memcpy(&bits, &v, sizeof(bits));
    if (bits == 0x8000000000000000ULL) bits = 0ULL;
    if (bits & 0x8000000000000000ULL) return ~bits;
    return bits | 0x8000000000000000ULL;
}

__global__ void k_iota_u32(unsigned int *__restrict__ order, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) order[p] = (unsigned int)p;
}

__global__ void k_keys_from_sym(const unsigned int *__restrict__ sym,
                                unsigned long long *__restrict__ keys, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) keys[p] = (unsigned long long)sym[p];
}

/* Signed time as an order-preserving u64; the permutation sort is radix, so a
   raw two's-complement negative would order after every positive timestamp. */
__global__ void k_keys_from_time(const long long *__restrict__ times,
                                 const unsigned int *__restrict__ order,
                                 unsigned long long *__restrict__ keys, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) {
        unsigned long long t = (unsigned long long)times[order[p]];
        keys[p] = t ^ 0x8000000000000000ULL;
    }
}

/* One flag per slot of the time-ordered permutation. An inclusive sum turns
   the flags into group ids without an atomic counter, so the layout is
   deterministic. */
__global__ void k_group_boundary(const long long *__restrict__ times,
                                 const unsigned int *__restrict__ order,
                                 unsigned int *__restrict__ flags, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    flags[p] = (p == 0 || times[order[p]] != times[order[p - 1]]) ? 1u : 0u;
}

/* Inclusive sum of the boundary flags counts groups from one. */
__global__ void k_gid_zero_based(unsigned int *__restrict__ gid, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) gid[p] -= 1u;
}

__global__ void k_group_starts(const unsigned int *__restrict__ flags,
                               const unsigned int *__restrict__ gid,
                               int *__restrict__ off, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n && flags[p]) off[gid[p]] = p;
}

__global__ void k_count_finite(const unsigned long long *__restrict__ keys,
                               const int *__restrict__ off, int ngroups,
                               unsigned int *__restrict__ counts)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    unsigned int c = 0;
    for (int p = off[g]; p < off[g + 1]; ++p) {
        if (keys[p] != RC_RANK_INVALID) c += 1;
    }
    counts[g] = c;
}

/* Build one column's sort keys in permutation slot order. Non-members and
   non-finite sources share the invalid key, so they sort to the end of their
   timestamp and never enter the finite count or a rank. */
__global__ void k_rank_keys(const double *__restrict__ src,
                            const unsigned int *__restrict__ order,
                            const unsigned char *__restrict__ member,
                            unsigned long long *__restrict__ keys, int n)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    unsigned int r = order[p];
    keys[p] = member[r] ? rc_sort_key(src[r]) : RC_RANK_INVALID;
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
    if (pos < finite && finite >= 20 &&
        keys[p] != RC_RANK_INVALID) {
        v = 2.0 * (double)pos / (double)(finite - 1) - 1.0;
    }
    out[rows[p]] = v;
}

/* ------------------------------------------------------------------------- */
/* C API                                                                     */
/* ------------------------------------------------------------------------- */

extern "C" void rc_enrich_destroy(void *v);

/* #298 engine-tail entry points live in `tail.cu`; they take raw device
   pointers so that translation unit does not need the RcEnrich layout. */
extern "C" {
int rc_tail_donchian(const long long *t, const double *high, const double *low,
                     const double *close, const int *seg_id, const int *ss,
                     const int *sl, int nseg, int n, int step, int w, int centred,
                     double *out);
int rc_tail_stoch(const long long *t, const double *high, const double *low,
                  const double *close, const int *seg_id, const int *ss,
                  const int *sl, int nseg, int n, int step, int w, int mean_w,
                  int which, double *out, double *scratch);
int rc_tail_atr(const long long *t, const double *high, const double *low,
                const double *close, const int *ss, const int *sl, int nseg,
                int n, int step, int period, int which, double *out);
int rc_tail_batr(const long long *t, const double *open, const double *high,
                 const double *low, const double *close, const int *ss,
                 const int *sl, int nseg, int n, int step, int period,
                 double *out);
int rc_tail_gapatr(const long long *t, const double *open, const double *high,
                   const double *low, const double *close, const int *ss,
                   const int *sl, int nseg, int n, int step, int period,
                   double *out);
int rc_tail_volrat(const long long *t, const double *high, const double *low,
                   const double *close, const int *ss, const int *sl, int nseg,
                   int n, int step, int p1, int p2, double *out);
int rc_tail_adx(const long long *t, const double *high, const double *low,
                const double *close, const int *ss, const int *sl, int nseg,
                int n, int step, int period, int which, double *out);
int rc_tail_bb_z(const long long *t, const double *close, const int *ss,
                 const int *sl, int nseg, int n, int step, int w, int mode,
                 double *out);
int rc_tail_ma_slope(const long long *t, const double *close, const int *ss,
                     const int *sl, int nseg, int n, int step, int w, int lag,
                     double *out);
int rc_tail_tod(const long long *t, int n, int step, int want_sin, double *out);
int rc_tail_exotic(const long long *t, const double *high, const double *low,
                   const double *close, const int *seg_id, const int *ss,
                   const int *sl, int nseg, int n, int step, int which,
                   double *out);
}

/* Free only the cross-section permutation/rank arena. */
static void rc_xs_free(RcEnrich *c)
{
    if (!c) return;
    cudaFree(c->d_xs_order);
    cudaFree(c->d_xs_order_tmp);
    cudaFree(c->d_xs_rows_out);
    cudaFree(c->d_xs_gid);
    cudaFree(c->d_xs_counts);
    cudaFree(c->d_xs_sym);
    cudaFree(c->d_xs_off);
    cudaFree(c->d_xs_keys);
    cudaFree(c->d_xs_keys_out);
    cudaFree(c->d_xs_member);
    cudaFree(c->d_xs_src);
    cudaFree(c->d_xs_temp);
    c->d_xs_order = NULL;
    c->d_xs_order_tmp = NULL;
    c->d_xs_rows_out = NULL;
    c->d_xs_gid = NULL;
    c->d_xs_counts = NULL;
    c->d_xs_sym = NULL;
    c->d_xs_off = NULL;
    c->d_xs_keys = NULL;
    c->d_xs_keys_out = NULL;
    c->d_xs_member = NULL;
    c->d_xs_src = NULL;
    c->d_xs_temp = NULL;
    c->xs_temp_bytes = 0;
    c->xs_groups = 0;
    c->xs_ready = 0;
}

extern "C" {

void *rc_enrich_create(const long long *times, const double *open, const double *high,
                       const double *low, const double *close, const double *volume,
                       const double *qvolume, const double *trades,
                       const double *taker_buy_base, const double *taker_buy_quote,
                       unsigned int present, int n, const int *seg_start,
                       const int *seg_len, int nseg, int step)
{
    if (n <= 0 || nseg <= 0) return NULL;
    RcEnrich *c = (RcEnrich *)calloc(1, sizeof(RcEnrich));
    if (!c) return NULL;
    c->n = n;
    c->nseg = nseg;
    c->step = step;
    c->present = present;
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
    if (ok && rc_alloc((void **)&c->d_trades, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_tb, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_tbq, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_start, (size_t)nseg * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_len, (size_t)nseg * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_id, rows * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_run_id, rows * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_a, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_b, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_c, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_d, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_e, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_cn, (rows + (size_t)nseg + 1) * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_bk, (rows + (size_t)nseg + 1) * sizeof(int)) != 0) ok = 0;
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
    rc_upload(c->d_trades, trades, rows * sizeof(double));
    rc_upload(c->d_tb, taker_buy_base, rows * sizeof(double));
    rc_upload(c->d_tbq, taker_buy_quote, rows * sizeof(double));
    rc_upload(c->d_seg_start, seg_start, (size_t)nseg * sizeof(int));
    rc_upload(c->d_seg_len, seg_len, (size_t)nseg * sizeof(int));
    c->last.h2d_ms = rc_now_ms() - t0;
    c->last.h2d_bytes = 8ULL * (unsigned long long)rows * 11ULL +
                        8ULL * (unsigned long long)nseg;
    /* Segment id per row, derived once on the host and uploaded with the rest
       of the resident data. It lets elementwise kernels refuse to cross a
       symbol boundary without a per-row lookup table on the device. */
    int *seg_id = (int *)malloc(rows * sizeof(int));
    int *run_id = (int *)malloc(rows * sizeof(int));
    if (!seg_id || !run_id) {
        free(seg_id);
        free(run_id);
        rc_enrich_destroy(c);
        return NULL;
    }
    int run = 0;
    for (int s = 0; s < nseg; ++s) {
        for (int j = 0; j < seg_len[s]; ++j) {
            int i = seg_start[s] + j;
            seg_id[i] = s;
            if (j > 0 && times[i] - times[i - 1] != (long long)step) run += 1;
            run_id[i] = run;
        }
        run += 1;
    }
    rc_upload(c->d_seg_id, seg_id, rows * sizeof(int));
    rc_upload(c->d_run_id, run_id, rows * sizeof(int));
    free(seg_id);
    free(run_id);
    c->last.h2d_bytes += (unsigned long long)rows * sizeof(int) * 2ULL;
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
    cudaFree(c->d_trades);
    cudaFree(c->d_tb);
    cudaFree(c->d_tbq);
    cudaFree(c->d_seg_start);
    cudaFree(c->d_seg_len);
    cudaFree(c->d_seg_id);
    cudaFree(c->d_run_id);
    cudaFree(c->d_a);
    cudaFree(c->d_b);
    cudaFree(c->d_c);
    cudaFree(c->d_d);
    cudaFree(c->d_e);
    cudaFree(c->d_cn);
    cudaFree(c->d_bk);
    cudaFree(c->d_out);
    cudaFree(c->d_out2);
    cudaFree(c->d_out3);
    cudaFree(c->d_batch);
#if defined(RCSR_UNIFIED)
    /* Managed memory needs no host free. */
#else
    if (c->h_pinned) cudaFreeHost(c->h_pinned);
#endif
    if (c->d_feat) rc_budget_give((size_t)c->feat_cap * (size_t)c->n * sizeof(double));
    cudaFree(c->d_feat);
    rc_xs_free(c);
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
        if (RC_LAUNCH_FAULT(RC_FAULT_ELEMENT)) {                              \
            cudaEventDestroy(ev0);                                            \
            cudaEventDestroy(ev1);                                            \
            c->last.failed = 1;                                               \
            stats_end(c);                                                     \
            return -1;                                                        \
        }                                                                     \
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

/* Tail entries own their kernels and synchronise internally; this wrapper keeps
   the same ledger windows as the in-file kernels. */
#define RC_TAIL(c, call, out)                                                 \
    do {                                                                      \
        stats_begin(c);                                                       \
        cudaEvent_t _e0, _e1;                                                 \
        cudaEventCreate(&_e0);                                                \
        cudaEventCreate(&_e1);                                                \
        cudaEventRecord(_e0, 0);                                              \
        double _l = rc_now_ms();                                              \
        int _rc = RC_LAUNCH_FAULT(RC_FAULT_ELEMENT) ? -1 : (call);            \
        c->last.launch_ms += rc_now_ms() - _l;                                \
        double _s = rc_now_ms();                                              \
        cudaEventRecord(_e1, 0);                                              \
        cudaEventSynchronize(_e1);                                            \
        c->last.sync_ms += rc_now_ms() - _s;                                  \
        float _ms = 0.0f;                                                     \
        cudaEventElapsedTime(&_ms, _e0, _e1);                                 \
        c->last.kernel_ms += _ms;                                             \
        cudaEventDestroy(_e0);                                                \
        cudaEventDestroy(_e1);                                                \
        if (_rc != 0) {                                                       \
            c->last.failed = 1;                                               \
            stats_end(c);                                                     \
            return -1;                                                        \
        }                                                                     \
        RC_DOWNLOAD(c, out, c->d_out);                                        \
        c->last.failed = (cudaGetLastError() != cudaSuccess) ? 1 : 0;         \
        stats_end(c);                                                         \
        return c->last.failed ? -1 : 0;                                       \
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

static RwArena rw_arena(const RcEnrich *c)
{
    RwArena a = {c->d_a, c->d_b, c->d_c, c->d_e, c->d_cn, c->d_bk};
    return a;
}

static void rw_launch_pre(const RcEnrich *c, const double *x, bool sq)
{
    k_rw_pre<<<c->nseg, 32>>>(x, c->d_times, c->d_seg_start, c->d_seg_len, c->step, sq,
                              rw_arena(c));
}

static void rw_launch_ma(const RcEnrich *c, int w, int mode, double *out)
{
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    rw_launch_pre(c, c->d_close, false);
    k_ma<<<blocks, threads>>>(c->d_times, c->d_close, c->d_seg_id, c->d_seg_start, c->n,
                              c->step, w, mode, rw_arena(c), out);
}

static void rw_launch_vol(const RcEnrich *c, int w, double *out)
{
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    k_vol_ret<<<blocks, threads>>>(c->d_times, c->d_close, c->d_seg_id, c->d_seg_start, c->n,
                                   c->step, c->d_d);
    rw_launch_pre(c, c->d_d, true);
    k_vol<<<blocks, threads>>>(c->d_times, c->d_d, c->d_seg_id, c->d_seg_start, c->n, c->step,
                               w, rw_arena(c), out);
}

static void rw_launch_bb_width(const RcEnrich *c, int w, double *out)
{
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    rw_launch_pre(c, c->d_close, true);
    k_bb_width<<<blocks, threads>>>(c->d_times, c->d_close, c->d_seg_id, c->d_seg_start, c->n,
                                    c->step, w, rw_arena(c), out);
}

static void rw_launch_vz(const RcEnrich *c, int w, double *out)
{
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    k_vz_x<<<blocks, threads>>>(c->d_volume, c->n, c->d_d);
    rw_launch_pre(c, c->d_d, true);
    k_vz<<<blocks, threads>>>(c->d_times, c->d_d, c->d_seg_id, c->d_seg_start, c->n, c->step,
                              w, rw_arena(c), out);
}

int rc_enrich_ma(void *v, int w, int mode, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    RC_ELEM_LAUNCH(c, (rw_launch_ma(c, w, mode, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_vol(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    RC_ELEM_LAUNCH(c, (rw_launch_vol(c, w, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_bb_width(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    RC_ELEM_LAUNCH(c, (rw_launch_bb_width(c, w, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_vz(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_ELEM_BEGIN(c, 0, 0);
    RC_ELEM_LAUNCH(c, (rw_launch_vz(c, w, c->d_out)));
    RC_DOWNLOAD(c, out, c->d_out);
    RC_END(c);
}

int rc_enrich_donchian(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_donchian(c->d_times, c->d_high, c->d_low, c->d_close,
                                c->d_seg_id, c->d_seg_start, c->d_seg_len,
                                c->nseg, c->n, c->step, w, 0, c->d_out)),
            out);
}

int rc_enrich_donchian_centred(void *v, int w, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_donchian(c->d_times, c->d_high, c->d_low, c->d_close,
                                c->d_seg_id, c->d_seg_start, c->d_seg_len,
                                c->nseg, c->n, c->step, w, 1, c->d_out)),
            out);
}

int rc_enrich_stoch(void *v, int which, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    /* The mean3 prefix needs (n + nseg + 1) doubles for sum and count each. */
    double *scratch = NULL;
    if (which == 1) {
        size_t nb = (size_t)c->n + (size_t)c->nseg + 1;
        if (rc_checked_malloc((void **)&scratch, 2 * nb * sizeof(double)) != cudaSuccess) {
            return -1;
        }
    }
    stats_begin(c);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    double l = rc_now_ms();
    int rc = rc_tail_stoch(c->d_times, c->d_high, c->d_low, c->d_close,
                           c->d_seg_id, c->d_seg_start, c->d_seg_len, c->nseg,
                           c->n, c->step, 14, 3, which, c->d_out, scratch);
    c->last.launch_ms += rc_now_ms() - l;
    double s = rc_now_ms();
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    c->last.sync_ms += rc_now_ms() - s;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    c->last.kernel_ms += ms;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (scratch) cudaFree(scratch);
    if (rc != 0) {
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    RC_DOWNLOAD(c, out, c->d_out);
    c->last.failed = (cudaGetLastError() != cudaSuccess) ? 1 : 0;
    stats_end(c);
    return c->last.failed ? -1 : 0;
}

int rc_enrich_atr(void *v, int period, int which, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_atr(c->d_times, c->d_high, c->d_low, c->d_close,
                           c->d_seg_start, c->d_seg_len, c->nseg, c->n,
                           c->step, period, which, c->d_out)),
            out);
}

int rc_enrich_batr(void *v, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_batr(c->d_times, c->d_open, c->d_high, c->d_low,
                            c->d_close, c->d_seg_start, c->d_seg_len, c->nseg,
                            c->n, c->step, 14, c->d_out)),
            out);
}

int rc_enrich_gapatr(void *v, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_gapatr(c->d_times, c->d_open, c->d_high, c->d_low,
                              c->d_close, c->d_seg_start, c->d_seg_len, c->nseg,
                              c->n, c->step, 14, c->d_out)),
            out);
}

int rc_enrich_volrat(void *v, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_volrat(c->d_times, c->d_high, c->d_low, c->d_close,
                              c->d_seg_start, c->d_seg_len, c->nseg, c->n,
                              c->step, 16, 96, c->d_out)),
            out);
}

int rc_enrich_adx(void *v, int which, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_adx(c->d_times, c->d_high, c->d_low, c->d_close,
                           c->d_seg_start, c->d_seg_len, c->nseg, c->n,
                           c->step, 14, which, c->d_out)),
            out);
}

int rc_enrich_bb_z(void *v, int w, int mode, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_bb_z(c->d_times, c->d_close, c->d_seg_start,
                            c->d_seg_len, c->nseg, c->n, c->step, w, mode,
                            c->d_out)),
            out);
}

int rc_enrich_ma_slope(void *v, int w, int lag, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_ma_slope(c->d_times, c->d_close, c->d_seg_start,
                                c->d_seg_len, c->nseg, c->n, c->step, w, lag,
                                c->d_out)),
            out);
}

int rc_enrich_tod(void *v, int want_sin, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_tod(c->d_times, c->n, c->step, want_sin, c->d_out)),
            out);
}

int rc_enrich_exotic(void *v, int which, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    RC_TAIL(c, (rc_tail_exotic(c->d_times, c->d_high, c->d_low, c->d_close,
                              c->d_seg_id, c->d_seg_start, c->d_seg_len,
                              c->nseg, c->n, c->step, which, c->d_out)),
            out);
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

/* #300: launch one descriptor into `c->d_out` (and `d_out2/d_out3` for MACD),
   with no host download. Returns the number of output slots written. */
static int rc_batch_launch(RcEnrich *c, const RcEnrichOp *op)
{
    if (RC_LAUNCH_FAULT(RC_FAULT_BATCH)) return 0;
    if (!c->d_times || !c->d_open || !c->d_high || !c->d_low || !c->d_close || !c->d_volume ||
        !c->d_seg_id || !c->d_seg_start || !c->d_seg_len || !c->d_a || !c->d_b || !c->d_c ||
        !c->d_d || !c->d_e || !c->d_cn || !c->d_bk || !c->d_out) {
        return 0;
    }
    if (op->kind == RC_OP_MACD && (!c->d_out2 || !c->d_out3)) return 0;
    int threads = 256, blocks = (c->n + threads - 1) / threads;
    int sthreads = 128, sblocks = (c->nseg + sthreads - 1) / sthreads;
    switch (op->kind) {
    case RC_OP_RET:
        k_ret<<<blocks, threads>>>(c->d_times, c->d_close, c->d_seg_id,
                                   c->n, c->step, op->a, c->d_out);
        return 1;
    case RC_OP_LAG:
        k_lag<<<blocks, threads>>>(c->d_times, c->d_close, c->d_seg_id,
                                   c->n, c->step, op->a, c->d_out);
        return 1;
    case RC_OP_RANGE:
        k_range<<<blocks, threads>>>(c->d_high, c->d_low, c->d_close,
                                     c->n, c->d_out);
        return 1;
    case RC_OP_CLV:
        k_clv<<<blocks, threads>>>(c->d_high, c->d_low, c->d_close,
                                   c->n, c->d_out);
        return 1;
    case RC_OP_CALENDAR:
        k_calendar<<<blocks, threads>>>(c->d_times, c->n, op->a, c->d_out);
        return 1;
    case RC_OP_MA:
        rw_launch_ma(c, op->a, op->b, c->d_out);
        return 1;
    case RC_OP_VOL:
        rw_launch_vol(c, op->a, c->d_out);
        return 1;
    case RC_OP_BB_WIDTH:
        rw_launch_bb_width(c, op->a, c->d_out);
        return 1;
    case RC_OP_VZ:
        rw_launch_vz(c, op->a, c->d_out);
        return 1;
    case RC_OP_DONCHIAN:
        k_donchian<<<sblocks, sthreads>>>(c->d_times, c->d_high, c->d_low,
                                          c->d_close, c->d_seg_start, c->d_seg_len,
                                          c->nseg, c->step, op->a, c->d_out);
        return 1;
    case RC_OP_EMA:
        k_ema_ratio<<<sblocks, sthreads>>>(c->d_times, c->d_close,
                                           c->d_seg_start, c->d_seg_len, c->nseg,
                                           c->step, op->a, c->d_out);
        return 1;
    case RC_OP_RSI:
        k_rsi_py<<<sblocks, sthreads>>>(c->d_times, c->d_close, c->d_seg_start,
                                        c->d_seg_len, c->nseg, c->step, op->a,
                                        c->d_a, c->d_b, c->d_c, c->d_out);
        return 1;
    case RC_OP_ATR:
        k_atr_py<<<sblocks, sthreads>>>(c->d_times, c->d_close, c->d_high,
                                        c->d_low, c->d_seg_start, c->d_seg_len,
                                        c->nseg, c->step, op->a, c->d_a, c->d_out);
        return 1;
    case RC_OP_LABEL:
        k_label<<<sblocks, sthreads>>>(c->d_times, c->d_open, c->d_close,
                                       c->d_seg_start, c->d_seg_len, c->nseg,
                                       c->step, op->a, op->param, op->b,
                                       op->c, c->d_out);
        return 1;
    case RC_OP_MACD:
        k_macd<<<sblocks, sthreads>>>(c->d_times, c->d_close, c->d_seg_start,
                                      c->d_seg_len, c->nseg, c->step, c->d_out,
                                      c->d_out2, c->d_out3);
        return 3;
    default:
        return 0;
    }
}

int rc_enrich_batch_capacity(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || c->n <= 0) return 0;
    size_t col_bytes = (size_t)c->n * sizeof(double);
    if (col_bytes == 0 || col_bytes > RC_BATCH_ARENA_BYTES) return 0;
    return (int)(RC_BATCH_ARENA_BYTES / col_bytes);
}

double *rc_enrich_batch_host(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    return c ? c->h_pinned : NULL;
}

int rc_enrich_batch_run(void *v, const RcEnrichOp *ops, int nops)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || !ops || nops <= 0) return -1;
    int cap = rc_enrich_batch_capacity(c);
    if (cap <= 0) return -1;
    int total_slots = 0;
    for (int i = 0; i < nops; ++i) {
        int s = ops[i].kind == RC_OP_MACD ? 3 : 1;
        total_slots += s;
    }
    if (total_slots <= 0 || total_slots > cap) return -1;
    size_t col_bytes = (size_t)c->n * sizeof(double);
    size_t bytes = col_bytes * (size_t)total_slots;
    int ok = 1;
    stats_begin(c);
    if (!c->d_batch) {
        double t0 = rc_now_ms();
        if (rc_alloc((void **)&c->d_batch, (size_t)cap * col_bytes) != 0) ok = 0;
        c->last.alloc_ms += rc_now_ms() - t0;
    }
#if defined(RCSR_UNIFIED)
    if (ok && !c->h_pinned) c->h_pinned = c->d_batch;
#else
    if (ok && !c->h_pinned) {
        double t0 = rc_now_ms();
        if (cudaHostAlloc((void **)&c->h_pinned, (size_t)cap * col_bytes,
                          cudaHostAllocDefault) != cudaSuccess) {
            ok = 0;
        }
        c->last.alloc_ms += rc_now_ms() - t0;
    }
#endif
    if (!ok) {
        stats_end(c);
        c->last.failed = 1;
        return -1;
    }
    c->batch_cap = cap;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    double l = rc_now_ms();
    int slot = 0;
    for (int i = 0; i < nops && ok; ++i) {
        int written = rc_batch_launch(c, &ops[i]);
        if (written <= 0 || slot + written > total_slots) {
            ok = 0;
            break;
        }
        cudaMemcpyAsync(c->d_batch + (size_t)slot * c->n, c->d_out,
                        col_bytes, cudaMemcpyDeviceToDevice, 0);
        if (written == 3) {
            cudaMemcpyAsync(c->d_batch + (size_t)(slot + 1) * c->n, c->d_out2,
                            col_bytes, cudaMemcpyDeviceToDevice, 0);
            cudaMemcpyAsync(c->d_batch + (size_t)(slot + 2) * c->n, c->d_out3,
                            col_bytes, cudaMemcpyDeviceToDevice, 0);
        }
        slot += written;
    }
    c->last.launch_ms += rc_now_ms() - l;
    if (ok && slot != total_slots) ok = 0;
    double s = rc_now_ms();
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    c->last.sync_ms += rc_now_ms() - s;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    c->last.kernel_ms += ms;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    if (ok) {
        /* One chunked 2-D copy for the whole block: the source and destination
           pitch are one full column, so each chunk carries every column's
           same row range and no per-column download is issued. */
        size_t chunk_rows = RC_BATCH_CHUNK_BYTES / (size_t)(total_slots * sizeof(double));
        if (chunk_rows < 1) chunk_rows = 1;
        double d = rc_now_ms();
        for (size_t off = 0; off < (size_t)c->n && ok; off += chunk_rows) {
            size_t rows = (size_t)c->n - off;
            if (rows > chunk_rows) rows = chunk_rows;
            cudaError_t rc = cudaMemcpy2DAsync(
                c->h_pinned + off, col_bytes,
                c->d_batch + off, col_bytes,
                rows * sizeof(double), (size_t)total_slots,
                cudaMemcpyDeviceToHost, 0);
            if (rc != cudaSuccess) ok = 0;
        }
        if (ok) {
            cudaError_t rc = cudaDeviceSynchronize();
            if (rc != cudaSuccess) ok = 0;
        }
        c->last.d2h_ms += rc_now_ms() - d;
        c->last.d2h_bytes += (unsigned long long)bytes;
    }
    stats_end(c);
    c->last.batch = 1;
    c->last.failed = ok ? 0 : 1;
    return ok ? 0 : -1;
}


/* ------------------------------------------------------------------------- */
/* Cross-section permutation and segmented rank (#295)                       */
/* ------------------------------------------------------------------------- */

/* Build the (time, symbol, row) permutation once and keep it resident.
   Two stable radix sorts reproduce the CPU's `sort_unstable_by` order:
   symbol first, then time. Equal times therefore keep symbol and row order. */
int rc_enrich_xs_prepare(void *v, const unsigned int *sym_key,
                         const unsigned char *member)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return -1;
    stats_begin(c);
    if (c->n <= 0 || !sym_key) {
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    if (c->xs_ready) {
        stats_end(c);
        return 0;
    }
    size_t n = (size_t)c->n;
    int ok = 1;
    double t_alloc = rc_now_ms();
    if (rc_alloc((void **)&c->d_xs_order, n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_order_tmp, n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_rows_out, n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_gid, n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_sym, n * sizeof(unsigned int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_keys, n * sizeof(unsigned long long)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_keys_out, n * sizeof(unsigned long long)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_member, n * sizeof(unsigned char)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_xs_src, n * sizeof(double)) != 0) ok = 0;
    if (ok) {
        t_alloc = rc_now_ms() - t_alloc;
        c->last.alloc_ms = t_alloc;
    }
    if (!ok) {
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    double t0 = rc_now_ms();
    rc_upload(c->d_xs_sym, sym_key, n * sizeof(unsigned int));
    if (member) {
        rc_upload(c->d_xs_member, member, n * sizeof(unsigned char));
    } else {
        cudaMemset(c->d_xs_member, 1, n * sizeof(unsigned char));
    }
    c->last.h2d_ms = rc_now_ms() - t0;
    c->last.h2d_bytes = (unsigned long long)n * (sizeof(unsigned int) +
                                                 sizeof(unsigned char));

    size_t t1 = 0, t2 = 0;
    if (rc_cub_status(cub::DeviceRadixSort::SortPairs(
            NULL, t1, c->d_xs_keys, c->d_xs_keys_out,
            c->d_xs_order, c->d_xs_order_tmp, c->n)) != cudaSuccess ||
        rc_cub_status(cub::DeviceScan::InclusiveSum(
            NULL, t2, c->d_xs_rows_out, c->d_xs_gid, c->n)) != cudaSuccess) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    size_t temp_bytes = t1 > t2 ? t1 : t2;
    t0 = rc_now_ms();
    if (rc_alloc(&c->d_xs_temp, temp_bytes) != 0) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    c->last.alloc_ms += rc_now_ms() - t0;
    c->xs_temp_bytes = temp_bytes;

    if (RC_LAUNCH_FAULT(RC_FAULT_XS_PREPARE) || !c->d_times || !c->d_xs_order ||
        !c->d_xs_order_tmp || !c->d_xs_rows_out || !c->d_xs_gid || !c->d_xs_sym ||
        !c->d_xs_keys || !c->d_xs_keys_out || !c->d_xs_temp) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }

    int threads = 256;
    int blocks = (c->n + threads - 1) / threads;
    double l0 = rc_now_ms();
    k_iota_u32<<<blocks, threads>>>(c->d_xs_order, c->n);
    k_keys_from_sym<<<blocks, threads>>>(c->d_xs_sym, c->d_xs_keys, c->n);
    c->last.launch_ms += rc_now_ms() - l0;

    /* Pass 1: stable by symbol. The value carried along is the row index. */
    cudaError_t cub_err = RC_CUB(g_force_cub_fail, cub::DeviceRadixSort::SortPairs(
        c->d_xs_temp, temp_bytes, c->d_xs_keys, c->d_xs_keys_out,
        c->d_xs_order, c->d_xs_order_tmp, c->n));
    if (cub_err != cudaSuccess) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    /* Pass 2: stable by time. Equal times retain the symbol/row order. */
    k_keys_from_time<<<blocks, threads>>>(c->d_times, c->d_xs_order_tmp,
                                          c->d_xs_keys, c->n);
    cub_err = RC_CUB(g_force_cub_fail, cub::DeviceRadixSort::SortPairs(
        c->d_xs_temp, temp_bytes, c->d_xs_keys, c->d_xs_keys_out,
        c->d_xs_order_tmp, c->d_xs_order, c->n));
    if (cub_err != cudaSuccess) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }

    /* Group boundaries, ids and starts from the time-ordered permutation. */
    k_group_boundary<<<blocks, threads>>>(c->d_times, c->d_xs_order,
                                          c->d_xs_rows_out, c->n);
    cub_err = RC_CUB(g_force_cub_fail, cub::DeviceScan::InclusiveSum(
        c->d_xs_temp, temp_bytes, c->d_xs_rows_out, c->d_xs_gid, c->n));
    if (cub_err != cudaSuccess) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    k_gid_zero_based<<<blocks, threads>>>(c->d_xs_gid, c->n);
    unsigned int last_gid = 0;
    cudaMemcpy(&last_gid, c->d_xs_gid + (c->n - 1), sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    int groups = (int)last_gid + 1;
    if (groups < 1) groups = 1;
    c->xs_groups = groups;

    t0 = rc_now_ms();
    ok = rc_alloc((void **)&c->d_xs_off, (size_t)(groups + 1) * sizeof(int)) == 0 &&
         rc_alloc((void **)&c->d_xs_counts,
                  (size_t)groups * sizeof(unsigned int)) == 0;
    c->last.alloc_ms += rc_now_ms() - t0;
    if (ok) {
        size_t t3 = 0;
        ok = rc_cub_status(cub::DeviceSegmentedRadixSort::SortPairs(
                 NULL, t3, c->d_xs_keys, c->d_xs_keys_out, c->d_xs_order,
                 c->d_xs_rows_out, c->n, groups, c->d_xs_off,
                 c->d_xs_off + 1)) == cudaSuccess;
        if (ok && t3 > c->xs_temp_bytes) {
            cudaFree(c->d_xs_temp);
            c->d_xs_temp = NULL;
            ok = rc_alloc(&c->d_xs_temp, t3) == 0;
            if (ok) c->xs_temp_bytes = t3;
        }
    }
    if (ok) {
        k_group_starts<<<blocks, threads>>>(c->d_xs_rows_out, c->d_xs_gid,
                                            c->d_xs_off, c->n);
        int end = c->n;
        cudaMemcpy(c->d_xs_off + groups, &end, sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaFree(c->d_xs_sym);
        c->d_xs_sym = NULL;
        c->xs_ready = 1;
    }

    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    c->last.kernel_ms = ms;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (!ok) {
        rc_xs_free(c);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    c->last.batch = 1;
    stats_end(c);
    return 0;
}

/* Rank one source column across the cached permutation. The stable segmented
   sort uses (value, symbol, row) via the cached order, and only finite PIT
   members enter the count; every other row is NaN. */
int rc_enrich_xs_rank(void *v, const double *src, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return -1;
    stats_begin(c);
    if (c->xs_ready && (RC_LAUNCH_FAULT(RC_FAULT_XS_RANK) || !c->d_xs_src || !c->d_xs_order ||
                        !c->d_xs_member || !c->d_xs_keys || !c->d_xs_keys_out ||
                        !c->d_xs_rows_out || !c->d_xs_off || !c->d_xs_counts ||
                        !c->d_xs_gid || !c->d_xs_temp)) {
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    if (!c->xs_ready || !src || !out) {
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    size_t n = (size_t)c->n;
    double t0 = rc_now_ms();
    rc_upload(c->d_xs_src, src, n * sizeof(double));
    c->last.h2d_ms = rc_now_ms() - t0;
    c->last.h2d_bytes = (unsigned long long)n * sizeof(double);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    int threads = 256;
    int blocks = (c->n + threads - 1) / threads;
    double l0 = rc_now_ms();
    k_rank_keys<<<blocks, threads>>>(c->d_xs_src, c->d_xs_order,
                                     c->d_xs_member, c->d_xs_keys, c->n);
    c->last.launch_ms = rc_now_ms() - l0;
    double l1 = rc_now_ms();
    cudaError_t cub_err = rc_cub_status(cub::DeviceSegmentedRadixSort::SortPairs(
        c->d_xs_temp, c->xs_temp_bytes, c->d_xs_keys, c->d_xs_keys_out,
        c->d_xs_order, c->d_xs_rows_out, c->n, c->xs_groups,
        c->d_xs_off, c->d_xs_off + 1));
    c->last.launch_ms += rc_now_ms() - l1;
    if (cub_err != cudaSuccess) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        c->last.failed = 1;
        stats_end(c);
        return -1;
    }
    int cblocks = (c->xs_groups + threads - 1) / threads;
    k_count_finite<<<cblocks, threads>>>(c->d_xs_keys_out, c->d_xs_off,
                                         c->xs_groups, c->d_xs_counts);
    k_rank_scatter<<<blocks, threads>>>(c->d_xs_keys_out, c->d_xs_rows_out,
                                        c->d_xs_off, c->d_xs_counts,
                                        c->d_xs_gid, c->n, c->d_xs_src);
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    c->last.kernel_ms = ms;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    t0 = rc_now_ms();
    rc_download(out, c->d_xs_src, n * sizeof(double));
    c->last.d2h_ms = rc_now_ms() - t0;
    c->last.d2h_bytes = (unsigned long long)n * sizeof(double);
    c->last.batch = 1;
    cudaError_t err = cudaGetLastError();
    c->last.failed = (err != cudaSuccess) ? 1 : 0;
    stats_end(c);
    return c->last.failed ? -1 : 0;
}

} /* extern "C" */

/* ------------------------------------------------------------------------- */
/* Projected pinned ring and direct H2D upload (issue #294)                  */
/* ------------------------------------------------------------------------- */

typedef struct RcRing {
    int slots;
    size_t slot_bytes;
    void **host;
    cudaStream_t copy_stream;
    cudaStream_t compute_stream;
    cudaEvent_t *copy_done;
    cudaEvent_t *compute_done;
    unsigned long long copy_count;
    unsigned long long compute_count;
    int failed;
} RcRing;

extern "C" {

void rc_ring_destroy(void *v);

void *rc_ring_create(int slots, size_t slot_bytes)
{
    if (slots < 2 || slot_bytes == 0) return NULL;
    RcRing *r = (RcRing *)calloc(1, sizeof(RcRing));
    if (!r) return NULL;
    r->slots = slots;
    r->slot_bytes = slot_bytes;
    r->host = (void **)calloc((size_t)slots, sizeof(void *));
    r->copy_done = (cudaEvent_t *)calloc((size_t)slots, sizeof(cudaEvent_t));
    r->compute_done = (cudaEvent_t *)calloc((size_t)slots, sizeof(cudaEvent_t));
    if (!r->host || !r->copy_done || !r->compute_done) { rc_ring_destroy((void *)r); return NULL; }
    if (cudaStreamCreateWithFlags(&r->copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
        rc_ring_destroy((void *)r); return NULL;
    }
    if (cudaStreamCreateWithFlags(&r->compute_stream, cudaStreamNonBlocking) != cudaSuccess) {
        rc_ring_destroy((void *)r); return NULL;
    }
    for (int i = 0; i < slots; ++i) {
        /* cudaHostAlloc gives the asynchronous copy engine page-locked host
           memory. Write-combined is deliberately not requested: the reader
           writes and the device reads, which is the case cudaHostAllocDefault
           documents as the safe/default choice. */
        if (cudaHostAlloc(&r->host[i], slot_bytes, cudaHostAllocDefault) != cudaSuccess) {
            rc_ring_destroy((void *)r); return NULL;
        }
        if (cudaEventCreateWithFlags(&r->copy_done[i], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&r->compute_done[i], cudaEventDisableTiming) != cudaSuccess) {
            rc_ring_destroy((void *)r); return NULL;
        }
        /* A freshly recorded event makes the first wait a no-op. */
        cudaEventRecord(r->compute_done[i], r->compute_stream);
    }
    cudaStreamSynchronize(r->compute_stream);
    return (void *)r;
}

void rc_ring_destroy(void *v)
{
    RcRing *r = (RcRing *)v;
    if (!r) return;
    if (r->copy_stream) cudaStreamDestroy(r->copy_stream);
    if (r->compute_stream) cudaStreamDestroy(r->compute_stream);
    if (r->host) {
        for (int i = 0; i < r->slots; ++i) {
            if (r->host[i]) cudaFreeHost(r->host[i]);
        }
        free(r->host);
    }
    if (r->copy_done) {
        for (int i = 0; i < r->slots; ++i) {
            if (r->copy_done[i]) cudaEventDestroy(r->copy_done[i]);
        }
        free(r->copy_done);
    }
    if (r->compute_done) {
        for (int i = 0; i < r->slots; ++i) {
            if (r->compute_done[i]) cudaEventDestroy(r->compute_done[i]);
        }
        free(r->compute_done);
    }
    free(r);
}

void *rc_ring_host(void *v, int slot)
{
    RcRing *r = (RcRing *)v;
    if (!r || slot < 0 || slot >= r->slots) return NULL;
    return r->host[slot];
}

size_t rc_ring_slot_bytes(void *v)
{
    RcRing *r = (RcRing *)v;
    return r ? r->slot_bytes : 0;
}

int rc_ring_slots(void *v)
{
    RcRing *r = (RcRing *)v;
    return r ? r->slots : 0;
}

/* The rc_ring_* status calls return 0, -1 for a bad argument, or the
   positive cudaError_t of the failing CUDA call (named by rc_ring_error_name). */
const char *rc_ring_error_name(int code)
{
    return cudaGetErrorName((cudaError_t)code);
}

/* Enqueue one async H2D copy per projected column, then mark the slot's copy
   event. The caller may immediately enqueue compute that waits on it. */
int rc_ring_upload(void *v, int slot, void *const *dev_ptrs, const size_t *dev_offsets,
                   const size_t *host_offsets, const size_t *bytes, int ncols)
{
    RcRing *r = (RcRing *)v;
    if (!r || slot < 0 || slot >= r->slots) return -1;
    for (int i = 0; i < ncols; ++i) {
        if (bytes[i] == 0) continue;
        char *dst = (char *)dev_ptrs[i] + dev_offsets[i];
        char *src = (char *)r->host[slot] + host_offsets[i];
        cudaError_t e = cudaMemcpyAsync(dst, src, bytes[i], cudaMemcpyHostToDevice,
                                        r->copy_stream);
        if (e != cudaSuccess) { r->failed = 1; return (int)e; }
    }
    cudaError_t e = cudaEventRecord(r->copy_done[slot], r->copy_stream);
    if (e != cudaSuccess) { r->failed = 1; return (int)e; }
    r->copy_count += 1;
    return 0;
}

int rc_ring_compute_begin(void *v, int slot)
{
    RcRing *r = (RcRing *)v;
    if (!r || slot < 0 || slot >= r->slots) return -1;
    cudaError_t e = cudaStreamWaitEvent(r->compute_stream, r->copy_done[slot], 0);
    if (e != cudaSuccess) { r->failed = 1; return (int)e; }
    return 0;
}

int rc_ring_compute_end(void *v, int slot)
{
    RcRing *r = (RcRing *)v;
    if (!r || slot < 0 || slot >= r->slots) return -1;
    cudaError_t e = cudaEventRecord(r->compute_done[slot], r->compute_stream);
    if (e != cudaSuccess) { r->failed = 1; return (int)e; }
    r->compute_count += 1;
    return 0;
}

/* Wait until the copy that reads this pinned slot has completed, so the reader
   can reuse it. This is an event wait, not a stream-wide synchronisation. A
   slot's first wait sees a never-recorded event, which is already complete.
   copy_stream never waits on another stream, so this cannot wait on the host. */
int rc_ring_wait_copy(void *v, int slot)
{
    RcRing *r = (RcRing *)v;
    if (!r || slot < 0 || slot >= r->slots) return -1;
    cudaError_t e = cudaEventSynchronize(r->copy_done[slot]);
    return e == cudaSuccess ? 0 : (int)e;
}

int rc_ring_wait_slot(void *v, int slot)
{
    RcRing *r = (RcRing *)v;
    if (!r || slot < 0 || slot >= r->slots) return -1;
    return cudaEventSynchronize(r->compute_done[slot]) == cudaSuccess ? 0 : -1;
}

/* One synchronisation at the end of the whole projected upload, never per
   group on the hot path. */
int rc_ring_sync(void *v)
{
    RcRing *r = (RcRing *)v;
    if (!r) return -1;
    if (cudaStreamSynchronize(r->compute_stream) != cudaSuccess) { r->failed = 1; return -1; }
    if (cudaStreamSynchronize(r->copy_stream) != cudaSuccess) { r->failed = 1; return -1; }
    return r->failed ? -1 : 0;
}

int rc_ring_stats(void *v, unsigned long long *copies, unsigned long long *computes,
                  unsigned long long *bytes, int *failed)
{
    RcRing *r = (RcRing *)v;
    if (!r) return -1;
    if (copies) *copies = r->copy_count;
    if (computes) *computes = r->compute_count;
    if (bytes) *bytes = (unsigned long long)r->slot_bytes * (unsigned long long)r->slots;
    if (failed) *failed = r->failed;
    return 0;
}

/* Run the per-tile compute needed by the projected path. `kind`:
   0 ret(k), 1 lag(k), 2 range, 3 clv, 4 calendar(which). */
int rc_ring_launch_seg_id(void *rv, void *cv, int start, int len)
{
    RcRing *r = (RcRing *)rv;
    RcEnrich *c = (RcEnrich *)cv;
    if (!r || !c || len <= 0) return -1;
    if (!c->d_seg_start || !c->d_seg_len || !c->d_seg_id) return -1;
    if (RC_LAUNCH_FAULT(RC_FAULT_SEG_ID)) return (int)cudaErrorLaunchFailure;
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    /* cudaGetLastError also reports an earlier call's error on this thread,
       e.g. an optional tile output whose allocation failed and was skipped.
       Clear it so only this launch decides the result. */
    (void)cudaGetLastError();
    k_seg_id_tile<<<blocks, threads, 0, r->compute_stream>>>(
        c->d_seg_start, c->d_seg_len, c->nseg, start, len, c->d_seg_id);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

int rc_ring_launch_feature(void *rv, void *cv, int kind, int k, int which,
                           int start, int len, double *out)
{
    RcRing *r = (RcRing *)rv;
    RcEnrich *c = (RcEnrich *)cv;
    if (!r || !c || !out || len <= 0) return -1;
    if (!c->d_times || !c->d_open || !c->d_high || !c->d_low || !c->d_close || !c->d_seg_id) return -1;
    if (RC_LAUNCH_FAULT(RC_FAULT_RING_FEATURE)) return (int)cudaErrorLaunchFailure;
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    cudaStream_t s = r->compute_stream;
    (void)cudaGetLastError();
    switch (kind) {
    case 0:
        k_ret_tile<<<blocks, threads, 0, s>>>(c->d_times, c->d_close, c->d_seg_id,
                                              c->step, k, start, len, out);
        break;
    case 1:
        k_lag_tile<<<blocks, threads, 0, s>>>(c->d_times, c->d_close, c->d_seg_id,
                                              c->step, k, start, len, out);
        break;
    case 2:
        k_range_tile<<<blocks, threads, 0, s>>>(c->d_high, c->d_low, c->d_close,
                                                start, len, out);
        break;
    case 3:
        k_clv_tile<<<blocks, threads, 0, s>>>(c->d_high, c->d_low, c->d_close,
                                              start, len, out);
        break;
    case 4:
        k_calendar_tile<<<blocks, threads, 0, s>>>(c->d_times, which, start, len, out);
        break;
    default:
        return -1;
    }
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

/* An RcEnrich that owns the device arrays but has not uploaded the source
   columns. The projected ring fills them group by group. `nan_mask` marks
   value arrays that must start as NaN (bit 0..5 = open,high,low,close,volume,
   qvolume). */
void *rc_enrich_create_arena(int n, int nseg, int step, const int *seg_start,
                             const int *seg_len, unsigned int nan_mask)
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
    if (ok && rc_alloc((void **)&c->d_trades, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_tb, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_tbq, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_start, (size_t)nseg * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_len, (size_t)nseg * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_seg_id, rows * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_a, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_b, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_c, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_d, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_e, (rows + (size_t)nseg + 1) * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_cn, (rows + (size_t)nseg + 1) * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_bk, (rows + (size_t)nseg + 1) * sizeof(int)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_out, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_out2, rows * sizeof(double)) != 0) ok = 0;
    if (ok && rc_alloc((void **)&c->d_out3, rows * sizeof(double)) != 0) ok = 0;
    c->last.alloc_ms = rc_now_ms() - t0;
    if (!ok) { rc_enrich_destroy(c); return NULL; }
    t0 = rc_now_ms();
    rc_upload(c->d_seg_start, seg_start, (size_t)nseg * sizeof(int));
    rc_upload(c->d_seg_len, seg_len, (size_t)nseg * sizeof(int));
    c->last.h2d_ms = rc_now_ms() - t0;
    c->last.h2d_bytes = 2ULL * (unsigned long long)nseg * sizeof(int);
    if (nan_mask & 1u) cudaMemset(c->d_open, 0xFF, rows * sizeof(double));
    if (nan_mask & 2u) cudaMemset(c->d_high, 0xFF, rows * sizeof(double));
    if (nan_mask & 4u) cudaMemset(c->d_low, 0xFF, rows * sizeof(double));
    if (nan_mask & 8u) cudaMemset(c->d_close, 0xFF, rows * sizeof(double));
    if (nan_mask & 16u) cudaMemset(c->d_volume, 0xFF, rows * sizeof(double));
    if (nan_mask & 32u) cudaMemset(c->d_qvolume, 0xFF, rows * sizeof(double));
    if (nan_mask & 64u) cudaMemset(c->d_trades, 0xFF, rows * sizeof(double));
    if (nan_mask & 128u) cudaMemset(c->d_tb, 0xFF, rows * sizeof(double));
    if (nan_mask & 256u) cudaMemset(c->d_tbq, 0xFF, rows * sizeof(double));
    /* #297: presence bits derive from the same mask; a clear bit means the
       optional column was not projected and must stay NaN. */
    c->present = 0;
    if (!(nan_mask & 16u)) c->present |= RC_PRESENT_VOLUME;
    if (!(nan_mask & 32u)) c->present |= RC_PRESENT_QUOTE_VOLUME;
    if (!(nan_mask & 64u)) c->present |= RC_PRESENT_TRADES;
    if (!(nan_mask & 128u)) c->present |= RC_PRESENT_TAKER_BUY;
    if (!(nan_mask & 256u)) c->present |= RC_PRESENT_TAKER_BUY_QV;
    stats_end(c);
    c->last.batch = 0;
    return c;
}

long long *rc_enrich_times_ptr(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    return c ? c->d_times : NULL;
}

/* which: 0 open, 1 high, 2 low, 3 close, 4 volume, 5 qvolume,
   6 trades, 7 taker-buy base, 8 taker-buy quote. */
double *rc_enrich_value_ptr(void *v, int which)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return NULL;
    switch (which) {
    case 0: return c->d_open;
    case 1: return c->d_high;
    case 2: return c->d_low;
    case 3: return c->d_close;
    case 4: return c->d_volume;
    case 5: return c->d_qvolume;
    case 6: return c->d_trades;
    case 7: return c->d_tb;
    case 8: return c->d_tbq;
    default: return NULL;
    }
}

double *rc_enrich_new_output(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return NULL;
    double *p = NULL;
    size_t bytes = (size_t)c->n * sizeof(double);
    if (rc_budget_take(bytes) != 0) return NULL;
    if (rc_alloc((void **)&p, bytes) != 0) {
        rc_budget_give(bytes);
        return NULL;
    }
    std::lock_guard<std::mutex> g(g_budget_mu);
    g_output_bytes[p] = bytes;
    return p;
}

void rc_enrich_release_output(double *p)
{
    if (!p) return;
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> g(g_budget_mu);
        auto it = g_output_bytes.find(p);
        if (it != g_output_bytes.end()) {
            bytes = it->second;
            g_output_bytes.erase(it);
        }
    }
    cudaFree(p);
    rc_budget_give(bytes);
}

int rc_enrich_download_ptr(const double *src, double *dst, int n)
{
    if (!src || !dst || n <= 0) return -1;
    return cudaMemcpy(dst, src, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess
               ? 0
               : -1;
}

void rc_enrich_note_projected_upload(void *v, unsigned long long bytes, double ms)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return;
    c->last.h2d_bytes += bytes;
    c->last.h2d_ms += ms;
}

void rc_enrich_note_projected_download(void *v, unsigned long long bytes, double ms)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return;
    c->last.d2h_bytes += bytes;
    c->last.d2h_ms += ms;
}

} /* extern "C" */

__global__ void k_ring_busy(unsigned long long cycles)
{
    unsigned long long t0 = clock64();
    while (clock64() - t0 < cycles) { __nanosleep(64); }
}

/* Event-ordering probe for the transport (tests/gpu_ring.rs). It exercises the
   same ring/stream/event calls as the projected path: a copy of slot i+1 is
   enqueued while the compute stream's busy kernel for slot i is still running.
   Reports the two windows and whether they overlap. No cudaStreamSynchronize is
   used until the final join. */
extern "C" int rc_ring_overlap_probe(size_t bytes, int iters, unsigned long long cycles,
                                     double *copy_ms, double *compute_ms, int *overlapped,
                                     unsigned long long *copies, int *failed)
{
    if (bytes == 0 || iters < 2) return -1;
    void *ring = rc_ring_create(2, bytes);
    if (!ring) return -1;
    void *d0 = NULL, *d1 = NULL;
    if (rc_alloc(&d0, bytes) != 0 || rc_alloc(&d1, bytes) != 0) {
        cudaFree(d0); cudaFree(d1); rc_ring_destroy(ring);
        return -1;
    }
    cudaEvent_t cs, ce, ks, ke;
    cudaEventCreate(&cs); cudaEventCreate(&ce);
    cudaEventCreate(&ks); cudaEventCreate(&ke);
    RcRing *r = (RcRing *)ring;
    cudaEventRecord(cs, r->copy_stream);
    cudaEventRecord(ks, r->compute_stream);
    int ok = 1;
    for (int i = 0; i < iters && ok; ++i) {
        int slot = i % 2;
        void *dev[1] = { (slot == 0) ? d0 : d1 };
        size_t off[1] = { 0 };
        size_t hoff[1] = { 0 };
        size_t len[1] = { bytes };
        if (rc_ring_upload(ring, slot, dev, off, hoff, len, 1) != 0) { ok = 0; break; }
        if (rc_ring_compute_begin(ring, slot) != 0) { ok = 0; break; }
        k_ring_busy<<<1, 1, 0, r->compute_stream>>>(cycles);
        if (cudaGetLastError() != cudaSuccess) { ok = 0; break; }
        if (rc_ring_compute_end(ring, slot) != 0) { ok = 0; break; }
    }
    if (ok) {
        cudaEventRecord(ce, r->copy_stream);
        cudaEventRecord(ke, r->compute_stream);
        if (cudaEventSynchronize(ke) != cudaSuccess) ok = 0;
        if (cudaEventSynchronize(ce) != cudaSuccess) ok = 0;
    }
    float cm = 0.0f, km = 0.0f, kstart = 0.0f, kend = 0.0f;
    if (ok) {
        cudaEventElapsedTime(&cm, cs, ce);
        cudaEventElapsedTime(&km, ks, ke);
        cudaEventElapsedTime(&kstart, cs, ks);
        cudaEventElapsedTime(&kend, cs, ke);
        if (copy_ms) *copy_ms = (double)cm;
        if (compute_ms) *compute_ms = (double)km;
        /* The compute stream's window starts before the copy stream finishes,
           and it is still running after it. */
        if (overlapped) *overlapped = (kstart < cm && kend > cm) ? 1 : 0;
    }
    if (copies) *copies = r->copy_count;
    if (failed) *failed = r->failed;
    cudaEventDestroy(cs); cudaEventDestroy(ce);
    cudaEventDestroy(ks); cudaEventDestroy(ke);
    cudaFree(d0); cudaFree(d1);
    rc_ring_destroy(ring);
    return ok ? 0 : -1;
}


