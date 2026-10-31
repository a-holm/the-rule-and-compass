/*
 * Engine-tail enrichment kernels (issue #298).
 *
 * This translation unit owns the parameterised tail of the engine catalogue:
 * blocked rolling extrema (Donchian, stochastic, close position and nr4-ready
 * primitives), ATR/ADX/DI, Bollinger bands, the exotic statistic family and the
 * remaining calendar/MA parameters. It never owns the resident context: every
 * entry point takes raw device pointers, so `enrich.cu` keeps the only copy of
 * the `RcEnrich` layout and this file stays independent of it.
 *
 * Arithmetic policy:
 *  - f64 everywhere, `-fmad=false`;
 *  - one gap rule: a window is NaN when its timestamps are not contiguous;
 *  - rolling extrema record window validity before masking, so CUDA's
 *    `fmax(NaN, x) == x` behaviour can never bridge an invalid bar;
 *  - recurrences are run-serial with the CPU seed/reset order, so ATR/ADX/DI
 *    columns are deterministic and match the reference within the declared tier.
 */

#include <cuda_runtime.h>
#include "alloc_guard.h"
#include <math.h>
#include <stdint.h>

#include "win_stats.h"

#define TAIL_MAX_W 1024

__device__ __forceinline__ bool tail_contig(const long long *t, int i, int w, int step)
{
    if (w == 0) return true;
    if (i + 1 < w || step <= 0) return false;
    return t[i + 1 - w] == t[i] - (long long)(w - 1) * step;
}

__device__ __forceinline__ double tail_clip(double v, double lo, double hi)
{
    if (!isfinite(v)) return NAN;
    return fmax(lo, fmin(hi, v));
}

__device__ __forceinline__ double tail_div_pos(double a, double b)
{
    if (!isfinite(a) || !isfinite(b) || b <= 0.0) return NAN;
    return a / b;
}

template <int CAP>
struct TailRing {
    double b[CAP];
    int win;
    int pos;
    int n;
    double sum;
    double sum2;
    int cnt;

    __device__ TailRing()
        : win(CAP), pos(0), n(0), sum(0.0), sum2(0.0), cnt(0) {}

    __device__ void set_window(int w)
    {
        win = w < 1 ? 1 : (w > CAP ? CAP : w);
        pos = 0;
        n = 0;
        sum = 0.0;
        sum2 = 0.0;
        cnt = 0;
    }

    __device__ void push(double v)
    {
        if (n == win) {
            double old = b[pos];
            if (isfinite(old)) {
                sum -= old;
                sum2 -= old * old;
                cnt -= 1;
            }
        } else {
            n += 1;
        }
        b[pos] = v;
        if (isfinite(v)) {
            sum += v;
            sum2 += v * v;
            cnt += 1;
        }
        pos += 1;
        if (pos == win) pos = 0;
    }

    __device__ double back(int k) const
    {
        int idx = pos - 1 - k;
        while (idx < 0) idx += win;
        return b[idx];
    }

    __device__ bool full() const { return n == win; }
    __device__ double mean_finite() const { return cnt > 0 ? sum / (double)cnt : NAN; }
    __device__ double mean_full() const { return cnt == win ? sum / (double)win : NAN; }
    __device__ double std_full() const
    {
        if (cnt < win) return NAN;
        double m = sum / (double)win;
        double s2 = sum2 / (double)win;
        return sqrt(fmax(s2 - m * m, 0.0));
    }
};

/* Causal compensated prefix ring (#364). It keeps the last `CAP + 1` rows of
   the block-local Sum2 (and, when `SQ`, Dot2 of x*x) running sums, so a window
   statistic is computed by the same code as the CPU `WinStats`. Rows are
   pushed once each, in segment order, with `restart` set at every
   `RW_BLOCK`-th row and at every gap row. */
template <int CAP, bool SQ = false>
struct TailPrefix {
    static const int SPAN = CAP + 1;
    double s1[SPAN];
    double c1[SPAN];
    double s2[SQ ? SPAN : 1];
    double c2[SQ ? SPAN : 1];
    double xv[SQ ? SPAN : 1];
    int cn[SPAN];
    int bk[SPAN];
    int win;
    int n;
    RwAcc acc;

    __device__ TailPrefix() : win(CAP), n(0) { rw_acc_init(&acc); }

    __device__ void set_window(int w)
    {
        win = w < 1 ? 1 : (w > CAP ? CAP : w);
        n = 0;
        rw_acc_init(&acc);
    }

    struct View {
        const TailPrefix *r;
        __device__ int slot(int q) const { return q % SPAN; }
        __device__ double s(int k, int q) const
        {
            return q < 0 ? 0.0 : (k == 0 ? r->s1[slot(q)] : r->s2[SQ ? slot(q) : 0]);
        }
        __device__ double c(int k, int q) const
        {
            return q < 0 ? 0.0 : (k == 0 ? r->c1[slot(q)] : r->c2[SQ ? slot(q) : 0]);
        }
        __device__ int bk(int q) const { return r->bk[slot(q)]; }
        __device__ int cn(int q) const { return q < 0 ? 0 : r->cn[slot(q)]; }
        __device__ double x(int q) const { return r->xv[SQ ? slot(q) : 0]; }
    };

    __device__ void store(int q)
    {
        int j = q % SPAN;
        s1[j] = acc.s1;
        c1[j] = acc.c1;
        if (SQ) {
            s2[j] = acc.s2;
            c2[j] = acc.c2;
        }
        cn[j] = acc.cnt;
        bk[j] = q - acc.start;
    }

    __device__ void push(double v, bool restart)
    {
        int q = n;
        rw_acc_push(&acc, v, q, restart, SQ);
        if (SQ) xv[q % SPAN] = v;
        store(q);
        n += 1;
    }

    /* Dot2 of a*b into the first stream (pair ring). */
    __device__ void push_dot(double a, double b, bool restart)
    {
        int q = n;
        rw_acc_push_dot(&acc, a, b, q, restart);
        store(q);
        n += 1;
    }

    __device__ bool full() const { return n >= win; }

    /* Window of the last `win` rows (clamped at the segment start; the
       callers mask rows with fewer than `win` rows). */
    __device__ int span_w() const { return win < n ? win : n; }

    __device__ double mean_finite() const
    {
        View v = {this};
        return rw_mean_finite(v, n - 1, span_w());
    }

    __device__ double mean_full() const
    {
        View v = {this};
        return rw_mean_strict(v, n - 1, span_w());
    }

    __device__ double std_full() const
    {
        View v = {this};
        return rw_std_pop(v, n - 1, span_w());
    }
};

/* Ring restart rule shared with the CPU `acc0_of`. */
__device__ __forceinline__ bool tail_restart(const long long *t, int i, int p, int step)
{
    return (p % RW_BLOCK) == 0 || (p > 0 && t[i] - t[i - 1] != (long long)step);
}

/* Chunked walk of a segment. One CTA per segment; thread `b` owns the rows of
   block `b` (RW_BLOCK rows) and replays `warm_blocks` blocks before it, so its
   prefix rings hold the accumulations of the sequential walk. Accumulations
   restart at every block start, so a replay from a block start is exact.
   `warm_blocks * RW_BLOCK` must cover every look-back of the kernel. */
__device__ __forceinline__ int tail_warm_start(int p0, int warm_blocks)
{
    int pw = p0 - warm_blocks * RW_BLOCK;
    return pw > 0 ? pw : 0;
}

/* ------------------------------------------------------------------------- */
/* Blocked rolling extrema (van Herk/Gil-Werman prefix-suffix)               */
/* ------------------------------------------------------------------------- */

__global__ void k_tail_fill_nan(double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = NAN;
}

/* Prefix extrema inside blocks of `w` rows, relative to the segment start.
   A non-finite cell poisons the rest of its block; `pre[j]` stores the extrema
   of [block_start, j], which is exactly what the combine step needs. */
__global__ void k_tail_extrema_prefix(const double *__restrict__ x, int n,
                                      const int *__restrict__ seg_id,
                                      const int *__restrict__ ss,
                                      const int *__restrict__ sl,
                                      int w, int want_max,
                                      double *__restrict__ pre)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = seg_id[i];
    int start = ss[s];
    int end = start + sl[s];
    int off = i - start;
    if (off % w != 0) return;
    int stop = start + off + w;
    if (stop > end) stop = end;
    double best = want_max ? -INFINITY : INFINITY;
    bool ok = true;
    for (int j = start + off; j < stop; ++j) {
        double v = x[j];
        if (!isfinite(v)) {
            ok = false;
        } else {
            best = want_max ? fmax(best, v) : fmin(best, v);
        }
        pre[j] = ok ? best : NAN;
    }
}

/* Suffix pass. Each block computes its backward extrema once and writes the
   window ending at `j + w - 1`, combining the suffix at `j` with the prefix at
   the window end. Every valid window is written exactly once. */
__global__ void k_tail_extrema_suffix(const long long *__restrict__ t,
                                      const double *__restrict__ x, int n,
                                      const int *__restrict__ seg_id,
                                      const int *__restrict__ ss,
                                      const int *__restrict__ sl,
                                      int step, int w, int want_max,
                                      const double *__restrict__ pre,
                                      double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = seg_id[i];
    int start = ss[s];
    int end = start + sl[s];
    int off = i - start;
    if (off % w != 0) return;
    int block_start = start + off;
    int block_end = block_start + w;
    if (block_end > end) block_end = end;
    double best = want_max ? -INFINITY : INFINITY;
    bool ok = true;
    for (int j = block_end - 1; j >= block_start; --j) {
        double v = x[j];
        if (!isfinite(v)) {
            ok = false;
        } else {
            best = want_max ? fmax(best, v) : fmin(best, v);
        }
        int out_i = j + w - 1;
        if (out_i < start || out_i >= end) continue;
        double suf = ok ? best : NAN;
        double pv = pre[out_i];
        if (tail_contig(t, out_i, w, step) && isfinite(suf) && isfinite(pv)) {
            out[out_i] = want_max ? fmax(suf, pv) : fmin(suf, pv);
        }
    }
}

/* Launch one rolling-extrema pass into `out`. `pre` is caller-owned scratch. */
static int tail_extrema_launch(const long long *t, const double *x, int n,
                               const int *seg_id, const int *ss, const int *sl,
                               int step, int w, int want_max,
                               double *pre, double *out)
{
    if (n <= 0 || w <= 0) return -1;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_tail_fill_nan<<<blocks, threads>>>(out, n);
    k_tail_extrema_prefix<<<blocks, threads>>>(x, n, seg_id, ss, sl, w, want_max, pre);
    k_tail_extrema_suffix<<<blocks, threads>>>(t, x, n, seg_id, ss, sl, step, w,
                                               want_max, pre, out);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void k_tail_donchian_combine(const double *__restrict__ close,
                                        const double *__restrict__ hi,
                                        const double *__restrict__ lo,
                                        int n, int centred,
                                        double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double h = hi[i], l = lo[i];
    if (!isfinite(h) || !isfinite(l)) {
        out[i] = NAN;
        return;
    }
    if (h > l) {
        double v = (close[i] - l) / (h - l);
        out[i] = centred ? v - 0.5 : v;
    } else {
        out[i] = centred ? 0.0 : NAN;
    }
}

__global__ void k_tail_close_pos_combine(const double *__restrict__ close,
                                         const double *__restrict__ hi,
                                         const double *__restrict__ lo,
                                         int n, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double h = hi[i], l = lo[i];
    out[i] = (isfinite(h) && isfinite(l) && h > l) ? (close[i] - l) / (h - l) : NAN;
}

extern "C" int rc_tail_donchian(const long long *t, const double *high,
                                const double *low, const double *close,
                                const int *seg_id, const int *ss, const int *sl,
                                int nseg, int n, int step, int w, int centred,
                                double *out)
{
    (void)nseg;
    double *pre = NULL, *hi = NULL, *lo = NULL;
    if (rc_checked_malloc((void **)&pre, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
    if (rc_checked_malloc((void **)&hi, (size_t)n * sizeof(double)) != cudaSuccess) {
        cudaFree(pre);
        return -1;
    }
    if (rc_checked_malloc((void **)&lo, (size_t)n * sizeof(double)) != cudaSuccess) {
        cudaFree(pre);
        cudaFree(hi);
        return -1;
    }
    int rc = tail_extrema_launch(t, high, n, seg_id, ss, sl, step, w, 1, pre, hi);
    if (rc == 0) rc = tail_extrema_launch(t, low, n, seg_id, ss, sl, step, w, 0, pre, lo);
    if (rc == 0) {
        int threads = 256, blocks = (n + threads - 1) / threads;
        k_tail_donchian_combine<<<blocks, threads>>>(close, hi, lo, n, centred, out);
        if (cudaGetLastError() != cudaSuccess) rc = -1;
    }
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    cudaFree(pre);
    cudaFree(hi);
    cudaFree(lo);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Stochastic oscillator                                                     */
/* ------------------------------------------------------------------------- */

__global__ void k_tail_stoch_k(const long long *__restrict__ t,
                               const double *__restrict__ close,
                               const double *__restrict__ hi,
                               const double *__restrict__ lo,
                               int n, int step, int w, int mean_w,
                               double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double h = hi[i], l = lo[i];
    double v = NAN;
    if (tail_contig(t, i, w, step) && isfinite(h) && isfinite(l) && h > l) {
        v = tail_clip((close[i] - l) / (h - l) - 0.5, -0.5, 0.5);
    }
    out[i] = v;
    (void)mean_w;
}

/* Stochastic %D: the rolling mean of %K, in place over `out`. */
__global__ void k_tail_stoch_d(const long long *__restrict__ t,
                               const int *__restrict__ ss,
                               const int *__restrict__ sl, int nseg,
                               int step, int w, double *out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    TailPrefix<8> ring;
    ring.set_window(w);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        ring.push(out[i], tail_restart(t, i, p, step));
        if (w <= 0 || p + 1 < w || !tail_contig(t, i, w, step)) {
            out[i] = NAN;
        } else {
            out[i] = ring.mean_finite();
        }
    }
}

 extern "C" int rc_tail_stoch(const long long *t, const double *high,
                             const double *low, const double *close,
                             const int *seg_id, const int *ss, const int *sl,
                             int nseg, int n, int step, int w, int mean_w,
                             int which, double *out, double *scratch)
{
    double *pre = NULL, *hi = NULL, *lo = NULL;
    if (rc_checked_malloc((void **)&pre, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
    if (rc_checked_malloc((void **)&hi, (size_t)n * sizeof(double)) != cudaSuccess) {
        cudaFree(pre);
        return -1;
    }
    if (rc_checked_malloc((void **)&lo, (size_t)n * sizeof(double)) != cudaSuccess) {
        cudaFree(pre);
        cudaFree(hi);
        return -1;
    }
    int rc = tail_extrema_launch(t, high, n, seg_id, ss, sl, step, w, 1, pre, hi);
    if (rc == 0) rc = tail_extrema_launch(t, low, n, seg_id, ss, sl, step, w, 0, pre, lo);
    if (rc == 0) {
        int threads = 256, blocks = (n + threads - 1) / threads;
        k_tail_stoch_k<<<blocks, threads>>>(t, close, hi, lo, n, step, w, mean_w, out);
        if (cudaGetLastError() != cudaSuccess) rc = -1;
    }
    if (rc == 0 && which == 1) {
        int threads = 128, sblocks = (nseg + threads - 1) / threads;
        k_tail_stoch_d<<<sblocks, threads>>>(t, ss, sl, nseg, step, mean_w, out);
        if (cudaGetLastError() != cudaSuccess) rc = -1;
    }
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    cudaFree(pre);
    cudaFree(hi);
    cudaFree(lo);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* ATR / ADX / DI                                                            */
/* ------------------------------------------------------------------------- */

__device__ __forceinline__ double tail_true_range(const long long *t,
                                                  const double *high,
                                                  const double *low,
                                                  const double *close,
                                                  int i, int start, int step,
                                                  bool gap_aware)
{
    if (!(isfinite(high[i]) && isfinite(low[i]))) return NAN;
    if (i == start || !isfinite(close[i - 1]) ||
        (gap_aware && t[i] - t[i - 1] != (long long)step)) {
        return high[i] - low[i];
    }
    double pc = close[i - 1];
    return fmax(high[i] - low[i],
                fmax(fabs(high[i] - pc), fabs(low[i] - pc)));
}

/* Python `wilder()`: seed on the first finite value, reset state at a gap. */
__device__ void tail_wilder_py(double *x, const long long *t, int start, int len,
                               int step, int period, double *out)
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
        if (out) out[i] = prev;
        else x[i] = prev;
    }
}

/* #125 `wilder()`: seed on the first complete finite window, restart on a gap
   mask, and mask the warm-up. This is the canonical standalone `atr14`. */
__device__ void tail_wilder_legacy(const double *x, const long long *t,
                                   int start, int len, int step, int period,
                                   double *out)
{
    int n = len;
    for (int p = 0; p < n; ++p) out[start + p] = NAN;
    if (n < period) return;
    int first = 0;
    while (first + period <= n) {
        bool ok = true;
        for (int j = 0; j < period; ++j) {
            if (!isfinite(x[start + first + j])) { ok = false; break; }
        }
        if (ok) break;
        first += 1;
    }
    if (first + period > n) return;
    double seed = 0.0;
    for (int j = 0; j < period; ++j) seed += x[start + first + j];
    double prev = seed / (double)period;
    out[start + first + period - 1] = prev;
    int look = period + 1;
    for (int p = first + period; p < n; ++p) {
        int i = start + p;
        bool gap = false;
        int lo = p - (look - 1);
        if (lo < 0) lo = 0;
        for (int j = lo + 1; j <= p; ++j) {
            if (t[start + j] - t[start + j - 1] != (long long)step) { gap = true; break; }
        }
        if (gap || !isfinite(x[i])) {
            if (p + period <= n) {
                bool ok = true;
                for (int j = 0; j < period; ++j) {
                    if (!isfinite(x[start + p + j])) { ok = false; break; }
                }
                if (ok) {
                    double s = 0.0;
                    for (int j = 0; j < period; ++j) s += x[start + p + j];
                    prev = s / (double)period;
                    out[start + p + period - 1] = prev;
                }
            }
            continue;
        }
        prev = (prev * (double)(period - 1) + x[i]) / (double)period;
        out[i] = prev;
    }
}

/* which: 0 = price ATR (python), 1 = NATR (python), 2 = fractional ATR
   (python, used by atr50), 3 = legacy #125 atr14 with its gap mask. */
__global__ void k_tail_atr(const long long *__restrict__ t,
                           const double *__restrict__ high,
                           const double *__restrict__ low,
                           const double *__restrict__ close,
                           const int *__restrict__ ss,
                           const int *__restrict__ sl, int nseg,
                           int step, int period, int which,
                           double *__restrict__ tr, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    bool gap_aware = (which != 3);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        tr[i] = tail_true_range(t, high, low, close, i, start, step, gap_aware);
    }
    if (which == 3) {
        tail_wilder_legacy(tr, t, start, len, step, period, out);
        int look = period + 1;
        for (int p = 0; p < len; ++p) {
            int i = start + p;
            bool gap = false;
            int lo = p - (look - 1);
            if (lo < 0) lo = 0;
            for (int j = lo + 1; j <= p; ++j) {
                if (t[start + j] - t[start + j - 1] != (long long)step) { gap = true; break; }
            }
            if (gap) out[i] = NAN;
        }
        return;
    }
    tail_wilder_py(tr, t, start, len, step, period, NULL);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        double a = tr[i];
        if (!isfinite(a)) {
            out[i] = NAN;
        } else if (which == 2) {
            out[i] = close[i] > 0.0 ? a / close[i] : NAN;
        } else if (which == 1) {
            out[i] = close[i] > 0.0 ? 100.0 * a / close[i] : NAN;
        } else {
            out[i] = a;
        }
    }
}

/* batr / gapatr / volrat are separate small kernels because they consume
   different ATR periods and inputs. */
__global__ void k_tail_batr(const long long *__restrict__ t,
                            const double *__restrict__ open,
                            const double *__restrict__ high,
                            const double *__restrict__ low,
                            const double *__restrict__ close,
                            const int *__restrict__ ss,
                            const int *__restrict__ sl, int nseg,
                            int step, int period,
                            double *__restrict__ tr, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        tr[i] = tail_true_range(t, high, low, close, i, start, step, true);
    }
    tail_wilder_py(tr, t, start, len, step, period, NULL);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        double a = tr[i];
        out[i] = (isfinite(a) && a > 0.0) ? (close[i] - open[i]) / a : NAN;
    }
}

__global__ void k_tail_gapatr(const long long *__restrict__ t,
                              const double *__restrict__ open,
                              const double *__restrict__ high,
                              const double *__restrict__ low,
                              const double *__restrict__ close,
                              const int *__restrict__ ss,
                              const int *__restrict__ sl, int nseg,
                              int step, int period,
                              double *__restrict__ tr, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        tr[i] = tail_true_range(t, high, low, close, i, start, step, true);
    }
    tail_wilder_py(tr, t, start, len, step, period, NULL);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        double a = tr[i];
        double prev = (p > 0 && t[i] - t[i - 1] == (long long)step) ? close[i - 1] : NAN;
        out[i] = (isfinite(a) && a > 0.0 && isfinite(prev))
                     ? (open[i] - prev) / a
                     : NAN;
    }
}

__global__ void k_tail_volrat(const long long *__restrict__ t,
                              const double *__restrict__ high,
                              const double *__restrict__ low,
                              const double *__restrict__ close,
                              const int *__restrict__ ss,
                              const int *__restrict__ sl, int nseg,
                              int step, int p1, int p2,
                              double *__restrict__ tr,
                              double *__restrict__ a1,
                              double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        tr[i] = tail_true_range(t, high, low, close, i, start, step, true);
    }
    for (int p = 0; p < len; ++p) a1[start + p] = tr[start + p];
    tail_wilder_py(tr, t, start, len, step, p1, NULL);
    tail_wilder_py(a1, t, start, len, step, p2, NULL);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = (isfinite(tr[i]) && isfinite(a1[i]) && a1[i] > 0.0)
                     ? tr[i] / a1[i]
                     : NAN;
    }
}

/* which: 0 = adx14/100, 1 = di_plus/100, 2 = di_minus/100, 3 = di_diff14. */
__global__ void k_tail_adx(const long long *__restrict__ t,
                           const double *__restrict__ high,
                           const double *__restrict__ low,
                           const double *__restrict__ close,
                           const int *__restrict__ ss,
                           const int *__restrict__ sl, int nseg,
                           int step, int period, int which,
                           double *__restrict__ tr,
                           double *__restrict__ pdi,
                           double *__restrict__ mdi,
                           double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    double ap = 1.0 / (double)period;
    double atr_prev = NAN, p_prev = NAN, m_prev = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool gap = p > 0 && (t[i] - t[i - 1] != (long long)step);
        if (gap) {
            atr_prev = NAN;
            p_prev = NAN;
            m_prev = NAN;
        }
        double trv = tail_true_range(t, high, low, close, i, start, step, true);
        double pdm = NAN, mdm = NAN;
        if (p > 0) {
            if (gap) {
                pdm = NAN;
                mdm = NAN;
            } else {
                double up = high[i] - high[i - 1];
                double dn = low[i - 1] - low[i];
                pdm = (up > dn && up > 0.0) ? up : 0.0;
                mdm = (dn > up && dn > 0.0) ? dn : 0.0;
            }
        } else {
            pdm = 0.0;
            mdm = 0.0;
        }
        if (isfinite(trv)) atr_prev = isfinite(atr_prev) ? atr_prev + ap * (trv - atr_prev) : trv;
        if (isfinite(pdm)) p_prev = isfinite(p_prev) ? p_prev + ap * (pdm - p_prev) : pdm;
        if (isfinite(mdm)) m_prev = isfinite(m_prev) ? m_prev + ap * (mdm - m_prev) : mdm;
        double pv = (isfinite(atr_prev) && atr_prev > 0.0) ? 100.0 * p_prev / atr_prev : NAN;
        double mv = (isfinite(atr_prev) && atr_prev > 0.0) ? 100.0 * m_prev / atr_prev : NAN;
        pdi[i] = pv;
        mdi[i] = mv;
        tr[i] = atr_prev;
        if (which == 1) out[i] = pv / 100.0;
        else if (which == 2) out[i] = mv / 100.0;
        else if (which == 3) out[i] = (isfinite(pv) && isfinite(mv)) ? (pv - mv) / 100.0 : NAN;
    }
    if (which > 0) return;
    double dx_prev = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool gap = p > 0 && (t[i] - t[i - 1] != (long long)step);
        if (gap) dx_prev = NAN;
        double pv = pdi[i], mv = mdi[i];
        double dx = (isfinite(pv) && isfinite(mv) && pv + mv > 0.0)
                        ? 100.0 * fabs(pv - mv) / (pv + mv)
                        : NAN;
        if (isfinite(dx)) dx_prev = isfinite(dx_prev) ? dx_prev + ap * (dx - dx_prev) : dx;
        out[i] = isfinite(dx_prev) ? dx_prev / 100.0 : NAN;
    }
}

extern "C" int rc_tail_atr(const long long *t, const double *high,
                           const double *low, const double *close,
                           const int *ss, const int *sl, int nseg, int n,
                           int step, int period, int which, double *out)
{
    double *tr = NULL;
    if (rc_checked_malloc((void **)&tr, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
    int threads = 128, blocks = (nseg + threads - 1) / threads;
    k_tail_atr<<<blocks, threads>>>(t, high, low, close, ss, sl, nseg, step,
                                    period, which, tr, out);
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    cudaFree(tr);
    return rc;
}

extern "C" int rc_tail_batr(const long long *t, const double *open,
                            const double *high, const double *low,
                            const double *close, const int *ss, const int *sl,
                            int nseg, int n, int step, int period, double *out)
{
    double *tr = NULL;
    if (rc_checked_malloc((void **)&tr, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
    int threads = 128, blocks = (nseg + threads - 1) / threads;
    k_tail_batr<<<blocks, threads>>>(t, open, high, low, close, ss, sl, nseg,
                                     step, period, tr, out);
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    cudaFree(tr);
    return rc;
}

extern "C" int rc_tail_gapatr(const long long *t, const double *open,
                              const double *high, const double *low,
                              const double *close, const int *ss, const int *sl,
                              int nseg, int n, int step, int period, double *out)
{
    double *tr = NULL;
    if (rc_checked_malloc((void **)&tr, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
    int threads = 128, blocks = (nseg + threads - 1) / threads;
    k_tail_gapatr<<<blocks, threads>>>(t, open, high, low, close, ss, sl, nseg,
                                       step, period, tr, out);
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    cudaFree(tr);
    return rc;
}

extern "C" int rc_tail_volrat(const long long *t, const double *high,
                              const double *low, const double *close,
                              const int *ss, const int *sl, int nseg, int n,
                              int step, int p1, int p2, double *out)
{
    double *tr = NULL, *a1 = NULL;
    if (rc_checked_malloc((void **)&tr, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
    if (rc_checked_malloc((void **)&a1, (size_t)n * sizeof(double)) != cudaSuccess) {
        cudaFree(tr);
        return -1;
    }
    int threads = 128, blocks = (nseg + threads - 1) / threads;
    k_tail_volrat<<<blocks, threads>>>(t, high, low, close, ss, sl, nseg, step,
                                       p1, p2, tr, a1, out);
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    cudaFree(tr);
    cudaFree(a1);
    return rc;
}

extern "C" int rc_tail_adx(const long long *t, const double *high,
                           const double *low, const double *close,
                           const int *ss, const int *sl, int nseg, int n,
                           int step, int period, int which, double *out)
{
    double *tr = NULL, *pdi = NULL, *mdi = NULL;
    if (rc_checked_malloc((void **)&tr, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
    if (rc_checked_malloc((void **)&pdi, (size_t)n * sizeof(double)) != cudaSuccess) {
        cudaFree(tr);
        return -1;
    }
    if (rc_checked_malloc((void **)&mdi, (size_t)n * sizeof(double)) != cudaSuccess) {
        cudaFree(tr);
        cudaFree(pdi);
        return -1;
    }
    int threads = 128, blocks = (nseg + threads - 1) / threads;
    k_tail_adx<<<blocks, threads>>>(t, high, low, close, ss, sl, nseg, step,
                                    period, which, tr, pdi, mdi, out);
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    cudaFree(tr);
    cudaFree(pdi);
    cudaFree(mdi);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Bollinger, MA slope, time-of-day                                          */
/* ------------------------------------------------------------------------- */

/* mode 0 = raw z-score (the #125 reference sums the window directly),
   1 = clip((C-mean)/(2 sd), -1, 1) (the compact/py15m prefix formula). */
__global__ void k_tail_bb_z_direct(const long long *__restrict__ t,
                                   const double *__restrict__ close,
                                   const int *__restrict__ ss,
                                   const int *__restrict__ sl, int nseg,
                                   int step, int w,
                                   double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    TailRing<TAIL_MAX_W> ring;
    ring.set_window(w);
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        ring.push(close[i]);
        if (!tail_contig(t, i, w, step) || p + 1 < w) {
            out[i] = NAN;
            continue;
        }
        double sum = 0.0;
        for (int k = w - 1; k >= 0; --k) sum += ring.back(k);
        double m = sum / (double)w;
        double ss2 = 0.0;
        for (int k = w - 1; k >= 0; --k) {
            double d = ring.back(k) - m;
            ss2 += d * d;
        }
        double sd = sqrt(ss2 / (double)w);
        out[i] = (isfinite(m) && isfinite(sd) && sd > 0.0)
                     ? (close[i] - m) / sd
                     : NAN;
    }
}

__global__ void k_tail_bb_z_prefix(const long long *__restrict__ t,
                                   const double *__restrict__ close,
                                   const int *__restrict__ ss,
                                   const int *__restrict__ sl, int nseg,
                                   int step, int w, int mode,
                                   double *__restrict__ out)
{
    int s = blockIdx.x;
    int start = ss[s], len = sl[s];
    TailPrefix<TAIL_MAX_W, true> ring;
    for (int b = threadIdx.x; b * RW_BLOCK < len; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int pe = min(len, p0 + RW_BLOCK);
        ring.set_window(w);
        for (int p = tail_warm_start(p0, 1); p < pe; ++p) {
            int i = start + p;
            ring.push(close[i], tail_restart(t, i, p, step));
            if (p < p0) continue;
            if (!tail_contig(t, i, w, step) || p + 1 < w) {
                out[i] = NAN;
                continue;
            }
            double m = ring.mean_full();
            double sd = ring.std_full();
            if (!isfinite(m) || !isfinite(sd) || sd <= 0.0) {
                out[i] = NAN;
            } else if (mode == 0) {
                out[i] = (close[i] - m) / sd;
            } else {
                out[i] = tail_clip((close[i] - m) / (2.0 * sd), -1.0, 1.0);
            }
        }
    }
}

extern "C" int rc_tail_bb_z(const long long *t, const double *close,
                            const int *ss, const int *sl, int nseg, int n,
                            int step, int w, int mode, double *out)
{
    int threads = 128, blocks = (nseg + threads - 1) / threads;
    if (mode == 0) {
        k_tail_bb_z_direct<<<blocks, threads>>>(t, close, ss, sl, nseg, step, w, out);
    } else {
        k_tail_bb_z_prefix<<<nseg, 32>>>(t, close, ss, sl, nseg, step, w, mode, out);
    }
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    (void)n;
    return rc;
}

__global__ void k_tail_ma_slope(const long long *__restrict__ t,
                                const double *__restrict__ close,
                                const int *__restrict__ ss,
                                const int *__restrict__ sl, int nseg,
                                int step, int w, int lag,
                                double *__restrict__ out)
{
    int s = blockIdx.x;
    int start = ss[s], len = sl[s];
    int warm = (w + lag + RW_BLOCK - 1) / RW_BLOCK;
    TailPrefix<TAIL_MAX_W> cring;
    TailRing<TAIL_MAX_W> mring;
    for (int b = threadIdx.x; b * RW_BLOCK < len; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int pe = min(len, p0 + RW_BLOCK);
        cring.set_window(w);
        mring.set_window(lag + 1);
        for (int p = tail_warm_start(p0, warm); p < pe; ++p) {
            int i = start + p;
            cring.push(close[i], tail_restart(t, i, p, step));
            double ma = NAN;
            if (tail_contig(t, i, w, step) && p + 1 >= w) ma = cring.mean_finite();
            mring.push(ma);
            if (p < p0) continue;
            double old = mring.back(lag);
            if (p >= lag && t[i] - t[i - lag] == (long long)lag * step &&
                isfinite(ma) && isfinite(old) && old != 0.0) {
                out[i] = ma / old - 1.0;
            } else {
                out[i] = NAN;
            }
        }
    }
}

extern "C" int rc_tail_ma_slope(const long long *t, const double *close,
                                const int *ss, const int *sl, int nseg, int n,
                                int step, int w, int lag, double *out)
{
    k_tail_ma_slope<<<nseg, 32>>>(t, close, ss, sl, nseg, step, w, lag, out);
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    (void)n;
    return rc;
}

__global__ void k_tail_tod(const long long *__restrict__ t, int n, int step,
                           int want_sin, double *__restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long day = t[i] % 86400;
    if (day < 0) day += 86400;
    long long idx = step > 0 ? day / step : 0;
    long long bars = step > 0 ? 86400 / step : 1;
    if (bars <= 0) bars = 1;
    const double pi = 3.14159265358979323846264338327950288;
    double a = 2.0 * pi * (double)idx / (double)bars;
    out[i] = want_sin ? sin(a) : cos(a);
}

extern "C" int rc_tail_tod(const long long *t, int n, int step, int want_sin,
                           double *out)
{
    int threads = 256, blocks = (n + threads - 1) / threads;
    k_tail_tod<<<blocks, threads>>>(t, n, step, want_sin, out);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* ------------------------------------------------------------------------- */
/* Exotic statistics                                                         */
/* ------------------------------------------------------------------------- */

__device__ __forceinline__ double tail_ret1(const long long *t, const double *c,
                                            int i, int step)
{
    if (i <= 0 || t[i] - t[i - 1] != (long long)step) return NAN;
    if (!(c[i] > 0.0 && c[i - 1] > 0.0)) return NAN;
    return log(c[i] / c[i - 1]);
}

__device__ __forceinline__ double tail_ret_k(const long long *t, const double *c,
                                             int i, int k, int step)
{
    if (i < k || t[i] - t[i - k] != (long long)k * step) return NAN;
    if (!(c[i] > 0.0 && c[i - k] > 0.0)) return NAN;
    return log(c[i] / c[i - k]);
}

__global__ void k_tail_exotic_simple5(const long long *__restrict__ t,
                                      const double *__restrict__ close,
                                      const int *__restrict__ ss,
                                      const int *__restrict__ sl, int nseg,
                                      int step, int want_std,
                                      double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    TailRing<5> ring;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        ring.push(tail_ret1(t, close, i, step));
        if (p < 5 || !ring.full()) {
            out[i] = NAN;
            continue;
        }
        bool ok = true;
        double sum = 0.0;
        for (int k = 4; k >= 0; --k) {
            double v = ring.back(k);
            if (!isfinite(v)) ok = false;
            sum += v;
        }
        if (!ok) {
            out[i] = NAN;
            continue;
        }
        double m = sum / 5.0;
        if (!want_std) {
            out[i] = m;
        } else {
            double ss2 = 0.0;
            for (int k = 4; k >= 0; --k) {
                double d = ring.back(k) - m;
                ss2 += d * d;
            }
            out[i] = sqrt(ss2 / 5.0);
        }
    }
}

/* `cov / var` of x against its lag over the last 96 rows, from double-word
   co-moments (the device twin of `window_stats::comoment_ratio`). */
template <class VX, class VL, class VXL>
__device__ double tail_comoment_ratio(const VX &vx, const VL &vl, const VXL &vxl, int hi, int w)
{
    int lo = hi + 1 - w;
    double na = (double)(vxl.cn(hi) - vxl.cn(lo - 1));
    double nb = (double)(vx.cn(hi) - vx.cn(lo - 1));
    double nc = (double)(vl.cn(hi) - vl.cn(lo - 1));
    if (!(na > 0.0 && nb > 0.0 && nc > 0.0)) return NAN;
    RwDw a = rw_win(vx, 0, lo, hi);
    RwDw b = rw_win(vl, 0, lo, hi);
    RwDw c = rw_win(vxl, 0, lo, hi);
    RwDw d = rw_win(vl, 1, lo, hi);
    RwDw numc = rw_dw_add(rw_dw_mul_f(c, __dmul_rn(nb, nc)),
                          rw_dw_neg(rw_dw_mul_f(rw_dw_mul(a, b), na)));
    double cov = __ddiv_rn(numc.h, __dmul_rn(__dmul_rn(na, nb), nc));
    RwDw numv = rw_dw_add(rw_dw_mul_f(d, nc), rw_dw_neg(rw_dw_mul(b, b)));
    double var = __ddiv_rn(rw_guarded(vl, numv, d, nc, lo, hi), __dmul_rn(nc, nc));
    return var > 0.0 ? __ddiv_rn(cov, var) : NAN;
}

__global__ void k_tail_exotic_ou(const long long *__restrict__ t,
                                 const double *__restrict__ close,
                                 const int *__restrict__ ss,
                                 const int *__restrict__ sl, int nseg,
                                 int step, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    TailPrefix<96> cr, xr, xxr;
    TailPrefix<96, true> xlr;
    double prev_x = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool rs = tail_restart(t, i, p, step);
        cr.push(close[i], rs);
        double ma = tail_contig(t, i, 96, step) && p + 1 >= 96 ? cr.mean_finite() : NAN;
        double x = (close[i] > 0.0 && isfinite(ma) && ma > 0.0)
                       ? rw_log1p_det((close[i] - ma) / ma)
                       : NAN;
        double xl = (p > 0 && t[i] - t[i - 1] == (long long)step) ? prev_x : NAN;
        prev_x = x;
        xr.push(x, rs);
        xlr.push(xl, rs);
        xxr.push_dot(x, xl, rs);
        if (!tail_contig(t, i, 96, step) || p + 1 < 96) {
            out[i] = NAN;
            continue;
        }
        TailPrefix<96>::View vx = {&xr};
        TailPrefix<96, true>::View vl = {&xlr};
        TailPrefix<96>::View vxl = {&xxr};
        double ratio = tail_comoment_ratio(vx, vl, vxl, p, 96);
        if (isfinite(ratio)) {
            double phi = fmax(1e-6, fmin(0.999999, ratio));
            double hl = -0.69314718055994530942 / log(phi);
            out[i] = log(1.0 + fmax(0.0, fmin(1000.0, hl)));
        } else {
            out[i] = NAN;
        }
    }
}

__global__ void k_tail_exotic_acf(const long long *__restrict__ t,
                                  const double *__restrict__ close,
                                  const int *__restrict__ ss,
                                  const int *__restrict__ sl, int nseg,
                                  int step, double *__restrict__ out)
{
    int s = blockIdx.x;
    int start = ss[s], len = sl[s];
    TailPrefix<384, true> lr, lrl;
    TailPrefix<384> prod;
    for (int b = threadIdx.x; b * RW_BLOCK < len; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int pe = min(len, p0 + RW_BLOCK);
        int pw = tail_warm_start(p0, 1);
        lr.set_window(384);
        lrl.set_window(384);
        prod.set_window(384);
        double prev = pw > 0 ? tail_ret1(t, close, start + pw - 1, step) : NAN;
        for (int p = pw; p < pe; ++p) {
            int i = start + p;
            bool rs = tail_restart(t, i, p, step);
            double v = tail_ret1(t, close, i, step);
            double l = (p > 0 && t[i] - t[i - 1] == (long long)step) ? prev : NAN;
            prev = v;
            lr.push(v, rs);
            lrl.push(l, rs);
            prod.push(v * l, rs);
            if (p < p0) continue;
            bool ok = tail_contig(t, i, 384, step) && p + 1 >= 384;
            double mlr = lr.mean_finite();
            double mlrl = lrl.mean_finite();
            double mp = prod.mean_finite();
            double sdlr = lr.std_full();
            double sdlrl = lrl.std_full();
            double den = sdlr * sdlrl;
            if (ok && isfinite(mp) && isfinite(mlr) && isfinite(mlrl) && den > 0.0) {
                out[i] = (mp - mlr * mlrl) / den;
            } else {
                out[i] = NAN;
            }
        }
    }
}

__global__ void k_tail_exotic_vr4(const long long *__restrict__ t,
                                  const double *__restrict__ close,
                                  const int *__restrict__ ss,
                                  const int *__restrict__ sl, int nseg,
                                  int step, double *__restrict__ out)
{
    int s = blockIdx.x;
    int start = ss[s], len = sl[s];
    TailPrefix<384, true> lr, r4;
    for (int b = threadIdx.x; b * RW_BLOCK < len; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int pe = min(len, p0 + RW_BLOCK);
        lr.set_window(384);
        r4.set_window(384);
        for (int p = tail_warm_start(p0, 1); p < pe; ++p) {
            int i = start + p;
            bool rs = tail_restart(t, i, p, step);
            lr.push(tail_ret1(t, close, i, step), rs);
            r4.push(tail_ret_k(t, close, i, 4, step), rs);
            if (p < p0) continue;
            double sdlr = lr.std_full();
            double sdr4 = r4.std_full();
            double den = 4.0 * sdlr * sdlr;
            if (tail_contig(t, i, 384, step) && p + 1 >= 384 && den > 0.0) {
                out[i] = (sdr4 * sdr4) / den;
            } else {
                out[i] = NAN;
            }
        }
    }
}

__global__ void k_tail_exotic_rskew(const long long *__restrict__ t,
                                    const double *__restrict__ close,
                                    const int *__restrict__ ss,
                                    const int *__restrict__ sl, int nseg,
                                    int step, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    TailPrefix<96> lr, d2, d3;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool rs = tail_restart(t, i, p, step);
        lr.push(tail_ret1(t, close, i, step), rs);
        double m = tail_contig(t, i, 96, step) && p + 1 >= 96 ? lr.mean_finite() : NAN;
        double d = tail_ret1(t, close, i, step) - m;
        d2.push(d * d, rs);
        d3.push(d * d * d, rs);
        double m2 = d2.mean_finite();
        double m3 = d3.mean_finite();
        if (tail_contig(t, i, 96, step) && p + 1 >= 96 && m2 > 0.0 &&
            isfinite(m2) && isfinite(m3)) {
            out[i] = m3 / pow(m2, 1.5);
        } else {
            out[i] = NAN;
        }
    }
}

__global__ void k_tail_exotic_jump(const long long *__restrict__ t,
                                   const double *__restrict__ close,
                                   const int *__restrict__ ss,
                                   const int *__restrict__ sl, int nseg,
                                   int step, double *__restrict__ out)
{
    int s = blockIdx.x;
    int start = ss[s], len = sl[s];
    TailPrefix<96> rv, bv;
    for (int b = threadIdx.x; b * RW_BLOCK < len; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int pe = min(len, p0 + RW_BLOCK);
        int pw = tail_warm_start(p0, 1);
        rv.set_window(96);
        bv.set_window(96);
        double prev = pw > 0 ? tail_ret1(t, close, start + pw - 1, step) : NAN;
        for (int p = pw; p < pe; ++p) {
            int i = start + p;
            bool rs = tail_restart(t, i, p, step);
            double v = tail_ret1(t, close, i, step);
            double l = (p > 0 && t[i] - t[i - 1] == (long long)step) ? prev : NAN;
            prev = v;
            rv.push(v * v, rs);
            bv.push(fabs(v) * fabs(l), rs);
            if (p < p0) continue;
            double mrv = rv.mean_finite();
            double mbv = bv.mean_finite();
            if (tail_contig(t, i, 96, step) && p + 1 >= 96 && isfinite(mrv) &&
                isfinite(mbv) && mrv > 0.0) {
                out[i] = fmax(mrv - (3.14159265358979323846 / 2.0) * mbv, 0.0) / mrv;
            } else {
                out[i] = NAN;
            }
        }
    }
}

__global__ void k_tail_exotic_semivar(const long long *__restrict__ t,
                                      const double *__restrict__ close,
                                      const int *__restrict__ ss,
                                      const int *__restrict__ sl, int nseg,
                                      int step, double *__restrict__ out)
{
    int s = blockIdx.x;
    int start = ss[s], len = sl[s];
    TailPrefix<96> rv, dn, up;
    for (int b = threadIdx.x; b * RW_BLOCK < len; b += blockDim.x) {
        int p0 = b * RW_BLOCK;
        int pe = min(len, p0 + RW_BLOCK);
        rv.set_window(96);
        dn.set_window(96);
        up.set_window(96);
        for (int p = tail_warm_start(p0, 1); p < pe; ++p) {
            int i = start + p;
            bool rs = tail_restart(t, i, p, step);
            double v = tail_ret1(t, close, i, step);
            rv.push(v * v, rs);
            dn.push(isfinite(v) && v < 0.0 ? v * v : (isfinite(v) ? 0.0 : NAN), rs);
            up.push(isfinite(v) && v > 0.0 ? v * v : (isfinite(v) ? 0.0 : NAN), rs);
            if (p < p0) continue;
            double mrv = rv.mean_finite();
            double mdn = dn.mean_finite();
            double mup = up.mean_finite();
            if (tail_contig(t, i, 96, step) && p + 1 >= 96 && isfinite(mrv) &&
                isfinite(mdn) && isfinite(mup) && mrv > 0.0) {
                out[i] = (mdn - mup) / mrv;
            } else {
                out[i] = NAN;
            }
        }
    }
}

__global__ void k_tail_exotic_cusum(const long long *__restrict__ t,
                                    const double *__restrict__ close,
                                    const int *__restrict__ ss,
                                    const int *__restrict__ sl, int nseg,
                                    int step, double *__restrict__ out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int start = ss[s], len = sl[s];
    TailPrefix<96, true> lr;
    TailRing<97> cs_ring;
    double acc = 0.0;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        bool rs = tail_restart(t, i, p, step);
        bool gap = p > 0 && (t[i] - t[i - 1] != (long long)step);
        if (gap) acc = 0.0;
        lr.push(tail_ret1(t, close, i, step), rs);
        double m = tail_contig(t, i, 96, step) && p + 1 >= 96 ? lr.mean_finite() : NAN;
        double sd = lr.std_full();
        double z = (isfinite(tail_ret1(t, close, i, step)) && isfinite(m) &&
                    isfinite(sd) && sd > 0.0)
                       ? (tail_ret1(t, close, i, step) - m) / sd
                       : NAN;
        if (isfinite(z)) acc += z;
        cs_ring.push(acc);
        if (tail_contig(t, i, 96, step) && p >= 96) {
            bool run_starts = t[i - 95] - t[i - 96] != (long long)step;
            double old = run_starts ? 0.0 : cs_ring.back(96);
            out[i] = fabs(acc - old) / sqrt(96.0);
        } else {
            out[i] = NAN;
        }
    }
}

/* which: 0 rmean5, 1 rstd5, 2 close_pos20, 3 ou_hl96, 4 acf1, 5 vr4,
   6 rskew96, 7 jump_share, 8 semivar_asym, 9 cusum_abs_z. */
extern "C" int rc_tail_exotic(const long long *t, const double *high,
                              const double *low, const double *close,
                              const int *seg_id, const int *ss, const int *sl,
                              int nseg, int n, int step, int which,
                              double *out)
{
    int threads = 128, blocks = (nseg + threads - 1) / threads;
    switch (which) {
    case 0:
    case 1:
        k_tail_exotic_simple5<<<blocks, threads>>>(t, close, ss, sl, nseg, step,
                                                   which, out);
        break;
    case 2: {
        double *pre = NULL, *hi = NULL, *lo = NULL;
        if (rc_checked_malloc((void **)&pre, (size_t)n * sizeof(double)) != cudaSuccess) return -1;
        if (rc_checked_malloc((void **)&hi, (size_t)n * sizeof(double)) != cudaSuccess) {
            cudaFree(pre);
            return -1;
        }
        if (rc_checked_malloc((void **)&lo, (size_t)n * sizeof(double)) != cudaSuccess) {
            cudaFree(pre);
            cudaFree(hi);
            return -1;
        }
        int rc = tail_extrema_launch(t, high, n, seg_id, ss, sl, step, 20, 1, pre, hi);
        if (rc == 0) rc = tail_extrema_launch(t, low, n, seg_id, ss, sl, step, 20, 0, pre, lo);
        if (rc == 0) {
            int tb = 256, bb = (n + tb - 1) / tb;
            k_tail_close_pos_combine<<<bb, tb>>>(close, hi, lo, n, out);
            if (cudaGetLastError() != cudaSuccess) rc = -1;
        }
        if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
        cudaFree(pre);
        cudaFree(hi);
        cudaFree(lo);
        return rc;
    }
    case 3:
        k_tail_exotic_ou<<<blocks, threads>>>(t, close, ss, sl, nseg, step, out);
        break;
    case 4:
        k_tail_exotic_acf<<<nseg, 32>>>(t, close, ss, sl, nseg, step, out);
        break;
    case 5:
        k_tail_exotic_vr4<<<nseg, 32>>>(t, close, ss, sl, nseg, step, out);
        break;
    case 6:
        k_tail_exotic_rskew<<<blocks, threads>>>(t, close, ss, sl, nseg, step, out);
        break;
    case 7:
        k_tail_exotic_jump<<<nseg, 32>>>(t, close, ss, sl, nseg, step, out);
        break;
    case 8:
        k_tail_exotic_semivar<<<nseg, 32>>>(t, close, ss, sl, nseg, step, out);
        break;
    case 9:
        k_tail_exotic_cusum<<<blocks, threads>>>(t, close, ss, sl, nseg, step, out);
        break;
    default:
        return -1;
    }
    int rc = cudaGetLastError() == cudaSuccess ? 0 : -1;
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = -1;
    return rc;
}
