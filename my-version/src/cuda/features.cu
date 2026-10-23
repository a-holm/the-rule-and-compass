/*
 * Fused flow and price-action feature producers (#297).
 *
 * CPU reference: `enrich::flow_all`, `enrich::price_action_all` and the shared
 * rolling helpers in `src/enrich.rs`. The device split is deliberately small:
 *
 *   flow:          one elementwise base kernel (vwap, per-trade, Amihud,
 *                  OBV step, CVD net, Chaikin money-flow value) + one run-serial
 *                  cumulative kernel (OBV, CVD) + one consumer kernel per
 *                  output column.
 *   price_action:  one elementwise geometry kernel (clv, body, range, validity,
 *                  true range) + one Wilder-14 pass + one consumer kernel per
 *                  output column.
 *
 * The base kernels read `trades` and `taker_buy_base` exactly once regardless
 * of how many consumer columns are requested; `base_reads` counts that, and a
 * test pins it. Rolling consumers run one thread per symbol segment and build
 * the same prefix sums as the CPU in the existing scratch arena, so their
 * addition order is identical without a per-thread ring or an n-length
 * per-column allocation. Two-bar geometry consumers run one thread per row.
 *
 * Gap policy: every state, contiguity and cumulative test is segment-relative.
 * `d_run_id` increments at a symbol boundary and at every non-contiguous time
 * step; `gap(i) = i>0 && same_segment && run_id differs`, which is exactly the
 * per-symbol `gap_mask` of the CPU. Segment start therefore behaves as CPU
 * `i == 0` (`gap[0] = false`, fresh streak, OBV = 0).
 *
 * Parity tiers: arithmetic-only columns are bitwise (build.rs compiles with
 * `-fmad=false` and IEEE sqrt); the three columns that take `log` are declared
 * `1e-12`. `zero_ret_share96` is bitwise because it tests the exact close ratio
 * rather than a rounded `log`, which is equivalent to `log(ratio) == 0`.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "enrich_common.h"

/* Flow output indices. Must match `features_gpu::FLOW_DEVICE_COLUMNS`. */
enum {
    RC_FLOW_VWAP = 0,
    RC_FLOW_TAKER_RATIO96,
    RC_FLOW_TAKER_IMB16,
    RC_FLOW_TRADE_INTENSITY96,
    RC_FLOW_AVG_TRADE_SIZE96,
    RC_FLOW_AMIHUD96,
    RC_FLOW_ZERO_RET_SHARE96,
    RC_FLOW_OBV_Z96,
    RC_FLOW_CVD_PROXY,
    RC_FLOW_VWAP_DEV96,
    RC_FLOW_CMF20,
    RC_FLOW_VWAP20_DEV,
    RC_FLOW_COUNT
};

/* Price-action output indices. Must match `features_gpu::PRICE_ACTION_DEVICE_COLUMNS`. */
enum {
    RC_PA_CLV16 = 0,
    RC_PA_BODY16,
    RC_PA_RNG_ATR,
    RC_PA_PREV_RANGE_POS,
    RC_PA_INSIDE_S,
    RC_PA_OUTSIDE_S,
    RC_PA_ENGULF_S,
    RC_PA_HARAMI_S,
    RC_PA_RUN8,
    RC_PA_NR4,
    RC_PA_TREND96,
    RC_PA_COUNT
};

/* Flow base planes. */
enum {
    RC_FB_VWAP = 0,
    RC_FB_PER_TRADE,
    RC_FB_AMIHUD,
    RC_FB_OBV,
    RC_FB_CVD,
    RC_FB_MFV,
    RC_FB_NET,
    RC_FB_STEP,
    RC_FB_PLANES
};

/* Price-action base planes (tr is overwritten in place by Wilder ATR). */
enum {
    RC_PB_CLV = 0,
    RC_PB_BODY,
    RC_PB_RNG,
    RC_PB_VALID,
    RC_PB_ATR,
    RC_PB_PLANES
};

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

__device__ __forceinline__ bool rc_seg_start(const RcEnrich *c, int i)
{
    return i == 0 || c->d_seg_id[i] != c->d_seg_id[i - 1];
}

__device__ __forceinline__ bool rc_gap(const RcEnrich *c, int i)
{
    if (i <= 0 || c->d_seg_id[i] != c->d_seg_id[i - 1]) return false;
    /* The projected-arena path (#294) uploads times through the ring and does
       not precompute d_run_id; the exact same test is one subtraction. */
    if (c->d_run_id) return c->d_run_id[i] != c->d_run_id[i - 1];
    return c->d_times[i] - c->d_times[i - 1] != (long long)c->step;
}

/* CPU `contig(times, i, w, step)`: the whole window is one contiguous run. */
__device__ __forceinline__ bool rc_contig(const RcEnrich *c, int i, int w)
{
    if (i + 1 < w) return false;
    return c->d_times[i + 1 - w] ==
           c->d_times[i] - (long long)(w - 1) * (long long)c->step;
}

/* Same as `rc_contig` plus the window does not cross a symbol boundary. */
__device__ __forceinline__ bool rc_contig_seg(const RcEnrich *c, int i, int w)
{
    return rc_contig(c, i, w) && c->d_seg_id[i + 1 - w] == c->d_seg_id[i];
}

__device__ __forceinline__ double rc_div_pos(double a, double b)
{
    if (isfinite(a) && isfinite(b) && b > 0.0) return a / b;
    return NAN;
}

__device__ __forceinline__ double rc_relu(double x)
{
    if (!isfinite(x)) return NAN;
    return x > 0.0 ? x : 0.0;
}

__device__ __forceinline__ double rc_clamp01(double x)
{
    if (!isfinite(x)) return NAN;
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

__device__ __forceinline__ double rc_pos_part(double x)
{
    if (!isfinite(x)) return NAN;
    return rc_relu(x) / (fabs(x) + 1e-12);
}

__device__ __forceinline__ double rc_neg_part(double x)
{
    if (!isfinite(x)) return NAN;
    return rc_relu(-x) / (fabs(x) + 1e-12);
}

/* One segment's prefix over the scratch arena. `pb` is a per-segment base so
 * two segment threads never write the same boundary slot (same scheme as
 * `enrich.cu::rc_prefix`). */
/* ------------------------------------------------------------------------- */
/* Flow base                                                                 */
/* ------------------------------------------------------------------------- */

__global__ void k_flow_base(RcEnrich *c, double *f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c->n) return;
    const int n = c->n;
    double *vwap = f + (size_t)RC_FB_VWAP * n;
    double *per_trade = f + (size_t)RC_FB_PER_TRADE * n;
    double *amihud = f + (size_t)RC_FB_AMIHUD * n;
    double *mfv = f + (size_t)RC_FB_MFV * n;
    double *net = f + (size_t)RC_FB_NET * n;
    double *stepv = f + (size_t)RC_FB_STEP * n;

    bool vol_present = (c->present & RC_PRESENT_VOLUME) != 0;
    bool qv_present = (c->present & RC_PRESENT_QUOTE_VOLUME) != 0;
    bool trades_present = (c->present & RC_PRESENT_TRADES) != 0;
    bool tb_present = (c->present & RC_PRESENT_TAKER_BUY) != 0;

    double v = vol_present ? c->d_volume[i] : NAN;
    double qv = qv_present ? c->d_qvolume[i] : NAN;
    double trades = trades_present ? c->d_trades[i] : NAN;
    double tb = tb_present ? c->d_tb[i] : NAN;
    double h = c->d_high[i];
    double l = c->d_low[i];
    double cl = c->d_close[i];

    /* The fused base reads each optional flow column once. These branches are
       on a uniform presence mask, not a per-consumer reload. */
    vwap[i] = rc_div_pos(qv, v);

    if (isfinite(qv) && qv >= 0.0 && isfinite(trades)) {
        double den = trades > 1.0 ? trades : 1.0;
        per_trade[i] = qv / den;
    } else {
        per_trade[i] = NAN;
    }

    bool lr_ok = i > 0 && !rc_gap(c, i) && isfinite(cl) &&
                 isfinite(c->d_close[i - 1]) && cl > 0.0 &&
                 c->d_close[i - 1] > 0.0;
    /* The Amihud plane stores the per-bar |log return| / quote volume: the
       96-bar consumer averages this value directly. */
    if (lr_ok && isfinite(qv) && qv > 0.0) {
        amihud[i] = fabs(log(cl / c->d_close[i - 1])) / qv;
    } else {
        amihud[i] = NAN;
    }

    if (isfinite(h) && isfinite(l) && isfinite(cl) && h > l) {
        mfv[i] = (((cl - l) - (h - cl)) / (h - l)) * v;
    } else {
        mfv[i] = NAN;
    }

    if (isfinite(v) && isfinite(tb)) {
        net[i] = 2.0 * tb - v;
    } else {
        net[i] = NAN;
    }

    /* OBV step = sign(C_t - C_{t-1}) * volume, valid only inside a run. The CPU
       recurrence requires finite closes and volume, not positive closes. */
    if (i > 0 && !rc_gap(c, i) && isfinite(cl) &&
        isfinite(c->d_close[i - 1]) && isfinite(v)) {
        double pc = c->d_close[i - 1];
        double sign = cl > pc ? 1.0 : (cl < pc ? -1.0 : 0.0);
        stepv[i] = sign * v;
    } else {
        stepv[i] = NAN;
    }
}

/* Cumulative OBV/CVD. One block per symbol segment, staged in shared memory so
 * global reads and writes are coalesced while thread 0 keeps the CPU's exact
 * left-to-right recurrence order. */
#define ST_TILE 512
__global__ void k_flow_state(RcEnrich *c, double *f)
{
    int s = blockIdx.x;
    if (s >= c->nseg) return;
    __shared__ double sv[ST_TILE];
    __shared__ double sstep[ST_TILE];
    __shared__ double snet[ST_TILE];
    __shared__ double sobv[ST_TILE];
    __shared__ double scvd[ST_TILE];
    const int n = c->n;
    double *obv = f + (size_t)RC_FB_OBV * n;
    double *cvd = f + (size_t)RC_FB_CVD * n;
    double *net = f + (size_t)RC_FB_NET * n;
    double *stepv = f + (size_t)RC_FB_STEP * n;
    int start = c->d_seg_start[s];
    int len = c->d_seg_len[s];
    bool vol_present = (c->present & RC_PRESENT_VOLUME) != 0;
    int tid = threadIdx.x;
    double obv_prev = NAN;
    double cvd_prev = NAN;

    for (int base = 0; base < len; base += ST_TILE) {
        int tlen = len - base;
        if (tlen > ST_TILE) tlen = ST_TILE;
        for (int j = tid; j < tlen; j += blockDim.x) {
            int i = start + base + j;
            sv[j] = vol_present ? c->d_volume[i] : NAN;
            sstep[j] = stepv[i];
            snet[j] = net[i];
        }
        __syncthreads();
        if (tid == 0) {
            for (int j = 0; j < tlen; ++j) {
                int p = base + j;
                int i = start + p;
                /* OBV: a gap starts a fresh state at 0 when volume is present;
                   an internal non-finite bar poisons the cumulative state but
                   does not reset it, exactly like `flow_all`. */
                double o;
                if (p == 0 || rc_gap(c, i)) {
                    o = isfinite(sv[j]) ? 0.0 : NAN;
                } else if (!isfinite(obv_prev) || !isfinite(sstep[j])) {
                    o = NAN;
                } else {
                    o = obv_prev + sstep[j];
                }
                obv_prev = o;
                sobv[j] = o;

                /* CVD: the gap bar itself is NaN; the next contiguous bar
                   restarts at its own net because the previous value is not
                   finite. */
                double cv;
                if (p > 0 && rc_gap(c, i)) {
                    cv = NAN;
                } else if (!isfinite(snet[j])) {
                    cv = NAN;
                } else {
                    double prev = p == 0 ? 0.0 : cvd_prev;
                    cv = isfinite(prev) ? prev + snet[j] : snet[j];
                }
                cvd_prev = cv;
                scvd[j] = cv;
            }
        }
        __syncthreads();
        for (int j = tid; j < tlen; j += blockDim.x) {
            int i = start + base + j;
            obv[i] = sobv[j];
            cvd[i] = scvd[j];
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------------- */
/* Flow consumers                                                            */
/* ------------------------------------------------------------------------- */

/* Rolling consumers use one CUDA block per symbol segment. The block stages a
 * tile of the source plane in shared memory (coalesced global access), one
 * thread builds the running prefix in CPU order, and the block then evaluates
 * one output row per lane. This keeps the exact `rmean_py` addition order while
 * avoiding the uncoalesced per-segment walk of a single thread. */
#define RT_TILE 512
#define RT_W 97
#define RT_SLOTS (RT_W + RT_TILE + 1)

typedef struct RtShared {
    double p1[RT_SLOTS];
    double p2[RT_SLOTS];
    int pc[RT_SLOTS];
    double xb[RT_TILE];
    double m1[RT_TILE];
    double carry1[RT_W];
    double carry2[RT_W];
    int carryc[RT_W];
} RtShared;

__device__ __forceinline__ void rt_prefix_tile(RtShared *sh, int tlen,
                                               double *s1, double *s2, int *sc)
{
    for (int k = 0; k < RT_W; ++k) {
        sh->p1[k] = sh->carry1[k];
        sh->p2[k] = sh->carry2[k];
        sh->pc[k] = sh->carryc[k];
    }
    sh->p1[RT_W] = *s1;
    sh->p2[RT_W] = *s2;
    sh->pc[RT_W] = *sc;
    for (int j = 0; j < tlen; ++j) {
        double v = sh->xb[j];
        if (isfinite(v)) {
            *s1 += v;
            *s2 += v * v;
            *sc += 1;
        }
        sh->p1[RT_W + j + 1] = *s1;
        sh->p2[RT_W + j + 1] = *s2;
        sh->pc[RT_W + j + 1] = *sc;
    }
    for (int k = 0; k < RT_W; ++k) {
        sh->carry1[k] = sh->p1[tlen + k];
        sh->carry2[k] = sh->p2[tlen + k];
        sh->carryc[k] = sh->pc[tlen + k];
    }
}

/* `rmean_py`: finite-count divisor, index `j` inside the staged tile. */
__device__ __forceinline__ double rt_mean(const RtShared *sh, int j, int w)
{
    int lo = RT_W + j + 1 - w;
    int hi = RT_W + j + 1;
    double n = sh->pc[hi] - sh->pc[lo];
    if (!(n > 0.0)) return NAN;
    return (sh->p1[hi] - sh->p1[lo]) / n;
}

/* `rstd_py`: full finite window, population formula. */
__device__ __forceinline__ double rt_std(const RtShared *sh, int j, int w)
{
    int lo = RT_W + j + 1 - w;
    int hi = RT_W + j + 1;
    if (sh->pc[hi] - sh->pc[lo] < w) return NAN;
    double m = (sh->p1[hi] - sh->p1[lo]) / (double)w;
    double s2 = (sh->p2[hi] - sh->p2[lo]) / (double)w;
    double var = s2 - m * m;
    if (var < 0.0) var = 0.0;
    return sqrt(var);
}

/* `rmean_strict`: every cell of the window must be finite. */
__device__ __forceinline__ double rt_strict(const RtShared *sh, int j, int w)
{
    int lo = RT_W + j + 1 - w;
    int hi = RT_W + j + 1;
    if (sh->pc[hi] - sh->pc[lo] != w) return NAN;
    return (sh->p1[hi] - sh->p1[lo]) / (double)w;
}

__device__ __forceinline__ void rt_zero_carry(RtShared *sh)
{
    for (int k = 0; k < RT_W; ++k) {
        sh->carry1[k] = 0.0;
        sh->carry2[k] = 0.0;
        sh->carryc[k] = 0;
    }
}

__device__ __forceinline__ int rt_tile_len(int len, int base)
{
    int t = len - base;
    return t > RT_TILE ? RT_TILE : t;
}

/* Copy one plane; one thread per row so the access is coalesced. */
__global__ void k_flow_copy(const double *src, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[i];
}

/* `vwap20_dev`: direct 20-value sums in CPU order, one thread per row. */
__global__ void k_flow_vwap20(const RcEnrich *c, double *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c->n) return;
    out[i] = NAN;
    if (i + 1 < 20 || !rc_contig_seg(c, i, 20)) return;
    double q = 0.0;
    double vol = 0.0;
    for (int j = i + 1 - 20; j <= i; ++j) {
        q += c->d_qvolume[j];
        vol += c->d_volume[j];
    }
    if (vol > 0.0) {
        double vw = q / vol;
        if (vw > 0.0) out[i] = c->d_close[i] / vw - 1.0;
    }
}

/* Single-source rolling columns: avg_trade_size96 (log mean), amihud96
 * (mean x 1e6), obv_z96 (z-score), cvd_proxy (cumulative / std) and
 * zero_ret_share96 (0/1 mean). */
__global__ void k_flow_roll_single(const RcEnrich *c, const double *f,
                                   int which, double *out)
{
    int s = blockIdx.x;
    if (s >= c->nseg) return;
    __shared__ RtShared sh;
    int start = c->d_seg_start[s];
    int len = c->d_seg_len[s];
    const int n = c->n;
    const double *obv = f + (size_t)RC_FB_OBV * n;
    const double *cvd = f + (size_t)RC_FB_CVD * n;
    const double *src = which == RC_FLOW_AVG_TRADE_SIZE96
        ? f + (size_t)RC_FB_PER_TRADE * n
        : (which == RC_FLOW_AMIHUD96 ? f + (size_t)RC_FB_AMIHUD * n
           : (which == RC_FLOW_OBV_Z96 ? obv
              : (which == RC_FLOW_CVD_PROXY ? cvd : NULL)));
    const int w = 96;
    int tid = threadIdx.x;
    double s1 = 0.0, s2 = 0.0;
    int sc = 0;
    if (tid == 0) rt_zero_carry(&sh);
    __syncthreads();
    for (int base = 0; base < len; base += RT_TILE) {
        int tlen = rt_tile_len(len, base);
        for (int j = tid; j < tlen; j += blockDim.x) {
            int i = start + base + j;
            double v;
            if (which == RC_FLOW_ZERO_RET_SHARE96) {
                int p = base + j;
                v = NAN;
                if (p > 0 && !rc_gap(c, i) && isfinite(c->d_close[i]) &&
                    isfinite(c->d_close[i - 1]) && c->d_close[i] > 0.0 &&
                    c->d_close[i - 1] > 0.0) {
                    v = (c->d_close[i] == c->d_close[i - 1]) ? 1.0 : 0.0;
                }
            } else {
                v = src[i];
            }
            sh.xb[j] = v;
        }
        __syncthreads();
        if (tid == 0) rt_prefix_tile(&sh, tlen, &s1, &s2, &sc);
        __syncthreads();
        for (int j = tid; j < tlen; j += blockDim.x) {
            int p = base + j;
            int i = start + p;
            if (p < w - 1 || !rc_contig(c, i, w)) {
                out[i] = NAN;
                continue;
            }
            if (which == RC_FLOW_AVG_TRADE_SIZE96) {
                double m = rt_mean(&sh, j, w);
                out[i] = (isfinite(m) && m >= 0.0) ? log(1.0 + m) : NAN;
            } else if (which == RC_FLOW_AMIHUD96) {
                out[i] = rt_mean(&sh, j, w) * 1e6;
            } else if (which == RC_FLOW_ZERO_RET_SHARE96) {
                out[i] = rt_mean(&sh, j, w);
            } else if (which == RC_FLOW_OBV_Z96) {
                double m = rt_mean(&sh, j, w);
                double sd = rt_std(&sh, j, w);
                out[i] = (sd > 0.0 && isfinite(obv[i]) && isfinite(m))
                    ? (obv[i] - m) / sd : NAN;
            } else {
                double sd = rt_std(&sh, j, w);
                out[i] = (isfinite(cvd[i]) && sd > 0.0) ? cvd[i] / sd : NAN;
            }
        }
        __syncthreads();
    }
}

/* Two-source rolling columns: taker_ratio96, taker_imb16,
 * trade_intensity96, vwap_dev96 and cmf20. Pass A stages the x1 mean into the
 * global scratch plane, pass B stages x2 and writes the ratio. */
__global__ void k_flow_roll_two(const RcEnrich *c, const double *f, int which,
                                const double *x1, const double *x2, double *out)
{
    int s = blockIdx.x;
    if (s >= c->nseg) return;
    __shared__ RtShared sh;
    int start = c->d_seg_start[s];
    int len = c->d_seg_len[s];
    int w = which == RC_FLOW_TAKER_IMB16 ? 16 : (which == RC_FLOW_CMF20 ? 20 : 96);
    int tid = threadIdx.x;

    /* Pass A: mean of x1 into the scratch plane. */
    double s1 = 0.0, s2 = 0.0;
    int sc = 0;
    if (tid == 0) rt_zero_carry(&sh);
    __syncthreads();
    for (int base = 0; base < len; base += RT_TILE) {
        int tlen = rt_tile_len(len, base);
        for (int j = tid; j < tlen; j += blockDim.x) sh.xb[j] = x1[start + base + j];
        __syncthreads();
        if (tid == 0) rt_prefix_tile(&sh, tlen, &s1, &s2, &sc);
        __syncthreads();
        for (int j = tid; j < tlen; j += blockDim.x) {
            int p = base + j;
            int i = start + p;
            sh.m1[j] = NAN;
            if (p >= w - 1 && rc_contig(c, i, w)) sh.m1[j] = rt_mean(&sh, j, w);
            c->d_d[i] = sh.m1[j];
        }
        __syncthreads();
    }

    /* Pass B: mean of x2 and the consumer expression. */
    s1 = 0.0;
    s2 = 0.0;
    sc = 0;
    if (tid == 0) rt_zero_carry(&sh);
    __syncthreads();
    for (int base = 0; base < len; base += RT_TILE) {
        int tlen = rt_tile_len(len, base);
        for (int j = tid; j < tlen; j += blockDim.x) sh.xb[j] = x2[start + base + j];
        __syncthreads();
        if (tid == 0) rt_prefix_tile(&sh, tlen, &s1, &s2, &sc);
        __syncthreads();
        for (int j = tid; j < tlen; j += blockDim.x) {
            int p = base + j;
            int i = start + p;
            out[i] = NAN;
            if (p < w - 1 || !rc_contig(c, i, w)) continue;
            double m1 = c->d_d[i];
            double m2 = rt_mean(&sh, j, w);
            if (which == RC_FLOW_TAKER_RATIO96) {
                out[i] = rc_div_pos(m1, m2);
            } else if (which == RC_FLOW_TAKER_IMB16) {
                if (isfinite(m1) && isfinite(m2)) {
                    out[i] = rc_div_pos(2.0 * m1 - m2, m2);
                }
            } else if (which == RC_FLOW_TRADE_INTENSITY96) {
                if (isfinite(m1) && isfinite(m2) && m2 > 0.0 && m1 > 0.0) {
                    out[i] = log(1.0 + m1 / m2);
                }
            } else if (which == RC_FLOW_VWAP_DEV96) {
                double vw = rc_div_pos(m1, m2);
                if (isfinite(c->d_close[i]) && vw > 0.0) {
                    out[i] = (c->d_close[i] - vw) / vw;
                }
            } else {
                out[i] = rc_div_pos(m1, m2);
            }
        }
        __syncthreads();
    }
}



/* Price-action base                                                         */
/* ------------------------------------------------------------------------- */

__global__ void k_price_base(RcEnrich *c, double *f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c->n) return;
    const int n = c->n;
    double *clv = f + (size_t)RC_PB_CLV * n;
    double *body = f + (size_t)RC_PB_BODY * n;
    double *rng = f + (size_t)RC_PB_RNG * n;
    double *valid = f + (size_t)RC_PB_VALID * n;
    double *tr = f + (size_t)RC_PB_ATR * n;

    double o = c->d_open[i];
    double h = c->d_high[i];
    double l = c->d_low[i];
    double cl = c->d_close[i];
    double v = (c->present & RC_PRESENT_VOLUME) ? c->d_volume[i] : NAN;

    bool volume_ok = !isfinite(v) || v > 0.0;
    bool finite = isfinite(o) && isfinite(h) && isfinite(l) && isfinite(cl);
    double r = h - l;
    if (!finite || !volume_ok || r <= 0.0) {
        clv[i] = NAN;
        body[i] = NAN;
        rng[i] = NAN;
        valid[i] = 0.0;
    } else {
        clv[i] = (cl - l) / r - 0.5;
        body[i] = (cl - o) / r;
        rng[i] = r;
        valid[i] = 1.0;
    }

    if (!(isfinite(h) && isfinite(l))) {
        tr[i] = NAN;
    } else if (i == 0 || rc_seg_start(c, i) || !isfinite(c->d_close[i - 1]) ||
               rc_gap(c, i)) {
        tr[i] = h - l;
    } else {
        double pc = c->d_close[i - 1];
        double a = h - l;
        double b = fabs(h - pc);
        double d = fabs(l - pc);
        double m = a > b ? a : b;
        tr[i] = m > d ? m : d;
    }
}

/* Wilder state, in place over the true-range plane. */
__global__ void k_price_wilder(RcEnrich *c, double *f)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= c->nseg) return;
    const int n = c->n;
    double *atr = f + (size_t)RC_PB_ATR * n;
    int start = c->d_seg_start[s];
    int len = c->d_seg_len[s];
    const double a = 1.0 / 14.0;
    double prev = NAN;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        if (p > 0 && rc_gap(c, i)) prev = NAN;
        double v = atr[i];
        if (!isfinite(v)) continue;
        prev = isfinite(prev) ? prev + a * (v - prev) : v;
        atr[i] = prev;
    }
}

/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */
/* Price-action consumers                                                    */
/* ------------------------------------------------------------------------- */

/* Strict rolling columns: clv16, body16 and trend96. One block per segment
 * with the same shared-tile prefix as the flow consumers. */
__global__ void k_price_roll(const RcEnrich *c, const double *f, int which, double *out)
{
    int s = blockIdx.x;
    if (s >= c->nseg) return;
    __shared__ RtShared sh;
    int start = c->d_seg_start[s];
    int len = c->d_seg_len[s];
    const int n = c->n;
    const double *clv = f + (size_t)RC_PB_CLV * n;
    const double *body = f + (size_t)RC_PB_BODY * n;
    const double *valid = f + (size_t)RC_PB_VALID * n;
    int w = which == RC_PA_TREND96 ? 96 : 16;
    int tid = threadIdx.x;
    double s1 = 0.0, s2 = 0.0;
    int sc = 0;
    if (tid == 0) rt_zero_carry(&sh);
    __syncthreads();
    for (int base = 0; base < len; base += RT_TILE) {
        int tlen = rt_tile_len(len, base);
        for (int j = tid; j < tlen; j += blockDim.x) {
            int i = start + base + j;
            sh.xb[j] = which == RC_PA_CLV16 ? clv[i]
                     : (which == RC_PA_BODY16 ? body[i]
                        : (valid[i] != 0.0 ? c->d_close[i] : NAN));
        }
        __syncthreads();
        if (tid == 0) rt_prefix_tile(&sh, tlen, &s1, &s2, &sc);
        __syncthreads();
        for (int j = tid; j < tlen; j += blockDim.x) {
            int p = base + j;
            int i = start + p;
            out[i] = NAN;
            if (p < w - 1 || !rc_contig(c, i, w)) continue;
            double m = rt_strict(&sh, j, w);
            if (which == RC_PA_TREND96) {
                if (valid[i] != 0.0 && m > 0.0 && c->d_close[i] > 0.0) {
                    out[i] = c->d_close[i] / m - 1.0;
                }
            } else {
                out[i] = m;
            }
        }
        __syncthreads();
    }
}

/* Run streak: one thread per segment; the state is a short integer recurrence. */
__global__ void k_price_run8(const RcEnrich *c, double *out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= c->nseg) return;
    int start = c->d_seg_start[s];
    int len = c->d_seg_len[s];
    int streak = 0;
    for (int p = 0; p < len; ++p) {
        int i = start + p;
        out[i] = NAN;
        if (p == 0 || rc_gap(c, i) || !isfinite(c->d_close[i]) ||
            !isfinite(c->d_close[i - 1])) {
            streak = 0;
            continue;
        }
        double cur = c->d_close[i];
        double prev = c->d_close[i - 1];
        if (cur > prev) {
            streak = streak > 0 ? streak + 1 : 1;
        } else if (cur < prev) {
            streak = streak < 0 ? streak - 1 : -1;
        } else {
            streak = 0;
        }
        int clipped = streak > 8 ? 8 : (streak < -8 ? -8 : streak);
        out[i] = (double)clipped / 8.0;
    }
}

/* Two-bar and single-bar columns: one thread per row. */
__global__ void k_price_elem(const RcEnrich *c, const double *f, int which, double *out)
{
    const int n = c->n;
    const double *body = f + (size_t)RC_PB_BODY * n;
    const double *rng = f + (size_t)RC_PB_RNG * n;
    const double *valid = f + (size_t)RC_PB_VALID * n;
    const double *atr = f + (size_t)RC_PB_ATR * n;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = NAN;
    double v0 = valid[i];
    double v1 = (i > 0 && !rc_seg_start(c, i)) ? valid[i - 1] : 0.0;
    bool gap = i == 0 || rc_seg_start(c, i) || rc_gap(c, i);
    double a = atr[i];
    switch (which) {
    case RC_PA_RNG_ATR:
        if (v0 != 0.0 && atr[i] > 0.0) out[i] = rng[i] / atr[i];
        break;
    case RC_PA_PREV_RANGE_POS:
        if (v0 != 0.0 && v1 != 0.0 && !gap) {
            double r0 = c->d_high[i - 1] - c->d_low[i - 1];
            if (r0 > 0.0) {
                out[i] = (c->d_close[i] - c->d_low[i - 1]) / r0 - 0.5;
            }
        }
        break;
    case RC_PA_INSIDE_S:
        if (v0 != 0.0 && v1 != 0.0 && !gap && isfinite(a) && a > 0.0) {
            double h = c->d_high[i];
            double l = c->d_low[i];
            double hp = c->d_high[i - 1];
            double lp = c->d_low[i - 1];
            out[i] = rc_relu(1.0 - rc_relu(h - hp) / a) *
                     rc_relu(1.0 - rc_relu(lp - l) / a);
        }
        break;
    case RC_PA_OUTSIDE_S:
        if (v0 != 0.0 && v1 != 0.0 && !gap && isfinite(a) && a > 0.0) {
            double h = c->d_high[i];
            double l = c->d_low[i];
            double hp = c->d_high[i - 1];
            double lp = c->d_low[i - 1];
            out[i] = rc_relu(1.0 - rc_relu(hp - h) / a) *
                     rc_relu(1.0 - rc_relu(l - lp) / a);
        }
        break;
    case RC_PA_ENGULF_S: {
        if (v0 != 0.0 && v1 != 0.0 && !gap && isfinite(a) && a > 0.0) {
            double b0 = body[i - 1];
            double b1 = body[i];
            double size = rc_relu((fabs(b1) - fabs(b0)) /
                                  (fabs(b1) + fabs(b0) + 1e-12));
            double bull = rc_pos_part(b1) * rc_neg_part(b0) * size *
                          rc_clamp01(rc_relu(c->d_close[i] - c->d_open[i - 1]) / a);
            double bear = rc_neg_part(b1) * rc_pos_part(b0) * size *
                          rc_clamp01(rc_relu(c->d_open[i] - c->d_close[i - 1]) / a);
            out[i] = bull - bear;
        }
        break;
    }
    case RC_PA_HARAMI_S: {
        if (v0 != 0.0 && v1 != 0.0 && !gap && isfinite(a) && a > 0.0) {
            double b0 = body[i - 1];
            double b1 = body[i];
            double shrink = fabs(b0) > 1e-12
                ? rc_relu(1.0 - fabs(b1) / (0.5 * fabs(b0))) : 0.0;
            double r0 = c->d_high[i - 1] - c->d_low[i - 1];
            double hi0 = fmax(c->d_open[i - 1], c->d_close[i - 1]);
            double lo0 = fmin(c->d_open[i - 1], c->d_close[i - 1]);
            double hi1 = fmax(c->d_open[i], c->d_close[i]);
            double lo1 = fmin(c->d_open[i], c->d_close[i]);
            double contain = 0.0;
            if (r0 > 0.0) {
                contain = rc_relu(1.0 - rc_relu(hi1 - hi0) / r0) *
                          rc_relu(1.0 - rc_relu(lo0 - lo1) / r0);
            }
            double opp = rc_pos_part(b0) * rc_neg_part(b1) +
                         rc_neg_part(b0) * rc_pos_part(b1);
            double dir = rc_pos_part(b1) - rc_neg_part(b1);
            out[i] = dir * opp * shrink * contain;
        }
        break;
    }
    case RC_PA_NR4: {
        if (i < 3 || !rc_contig_seg(c, i, 4)) break;
        bool ok = true;
        double min_r = INFINITY;
        for (int j = i - 3; j <= i; ++j) {
            if (valid[j] == 0.0) {
                ok = false;
                break;
            }
            double r = c->d_high[j] - c->d_low[j];
            if (r < min_r) min_r = r;
        }
        if (ok && min_r > 0.0) {
            out[i] = (c->d_high[i] - c->d_low[i]) / min_r;
        }
        break;
    }
    default:
        break;
    }
}


/* C API                                                                     */
/* ------------------------------------------------------------------------- */

/* Lazily allocate the fused base planes. The flow and price-action bases are
 * never live at the same time, so both families reuse one arena. */
static int feat_ensure(RcEnrich *c, int planes)
{
    if (c->feat_cap >= planes && c->d_feat) return 0;
    if (c->d_feat) {
        cudaFree(c->d_feat);
        c->d_feat = NULL;
        c->feat_cap = 0;
    }
    size_t bytes = (size_t)planes * (size_t)c->n * sizeof(double);
    if (rc_alloc((void **)&c->d_feat, bytes) != 0) return -1;
    c->feat_cap = planes;
    return 0;
}

#define FEAT_BEGIN(c)                                                         \
    do {                                                                      \
        stats_begin(c);                                                       \
    } while (0)

#define FEAT_LAUNCH(c, call)                                                  \
    do {                                                                      \
        cudaEvent_t ev0, ev1;                                                 \
        cudaEventCreate(&ev0);                                                \
        cudaEventCreate(&ev1);                                                \
        cudaEventRecord(ev0, 0);                                              \
        double _l = rc_now_ms();                                              \
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

#define FEAT_DOWNLOAD(c, dst)                                                 \
    do {                                                                      \
        double _d = rc_now_ms();                                              \
        rc_download(dst, c->d_out, (size_t)c->n * sizeof(double));            \
        c->last.d2h_ms += rc_now_ms() - _d;                                   \
        c->last.d2h_bytes += (unsigned long long)c->n * sizeof(double);       \
    } while (0)

#define FEAT_END(c)                                                           \
    do {                                                                      \
        stats_end(c);                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "rcsr features CUDA error: %s\n",                 \
                    cudaGetErrorString(err));                                 \
        }                                                                     \
        c->last.failed = (err != cudaSuccess) ? 1 : 0;                        \
        return c->last.failed ? -1 : 0;                                       \
    } while (0)

extern "C" {

int rc_features_flow_base(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return -1;
    if (feat_ensure(c, RC_FB_PLANES) != 0) return -1;
    FEAT_BEGIN(c);
    int threads = 256;
    int blocks = (c->n + threads - 1) / threads;
    FEAT_LAUNCH(c, (k_flow_base<<<blocks, threads>>>(c, c->d_feat)));
    int seg_blocks = (c->nseg + 127) / 128;
    FEAT_LAUNCH(c, (k_flow_state<<<c->nseg, 256>>>(c, c->d_feat)));
    /* Counted reads: the elementwise base consumed each present optional flow
       column exactly once, independent of the requested output count. */
    if (c->present & RC_PRESENT_TRADES) c->base_reads += 1;
    if (c->present & RC_PRESENT_TAKER_BUY) c->base_reads += 1;
    c->base_passes += 2;
    FEAT_END(c);
}

int rc_features_flow_col(void *v, int which, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || !out || which < 0 || which >= RC_FLOW_COUNT) return -1;
    if (!c->d_feat) return -1;
    FEAT_BEGIN(c);
    int rows = (c->n + 255) / 256;
    int segs = c->nseg;
    switch (which) {
    case RC_FLOW_VWAP:
        FEAT_LAUNCH(c, (k_flow_copy<<<rows, 256>>>(c->d_feat + (size_t)RC_FB_VWAP * c->n, c->d_out, c->n)));
        break;
    case RC_FLOW_VWAP20_DEV:
        FEAT_LAUNCH(c, (k_flow_vwap20<<<rows, 256>>>(c, c->d_out)));
        break;
    case RC_FLOW_AVG_TRADE_SIZE96:
    case RC_FLOW_AMIHUD96:
    case RC_FLOW_ZERO_RET_SHARE96:
    case RC_FLOW_OBV_Z96:
    case RC_FLOW_CVD_PROXY:
        FEAT_LAUNCH(c, (k_flow_roll_single<<<segs, 256>>>(c, c->d_feat, which, c->d_out)));
        break;
    case RC_FLOW_TAKER_RATIO96:
    case RC_FLOW_TAKER_IMB16:
    case RC_FLOW_TRADE_INTENSITY96:
    case RC_FLOW_VWAP_DEV96:
    case RC_FLOW_CMF20: {
        const int n = c->n;
        const double *x1 = which == RC_FLOW_TRADE_INTENSITY96
            ? c->d_trades
            : (which == RC_FLOW_VWAP_DEV96 ? c->d_qvolume
               : (which == RC_FLOW_CMF20 ? c->d_feat + (size_t)RC_FB_MFV * n
                  : c->d_tb));
        FEAT_LAUNCH(c, (k_flow_roll_two<<<segs, 256>>>(c, c->d_feat, which, x1, c->d_volume, c->d_out)));
        break;
    }
    default:
        return -1;
    }
    FEAT_DOWNLOAD(c, out);
    FEAT_END(c);
}

int rc_features_price_base(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c) return -1;
    if (feat_ensure(c, RC_PB_PLANES) != 0) return -1;
    FEAT_BEGIN(c);
    int threads = 256;
    int blocks = (c->n + threads - 1) / threads;
    FEAT_LAUNCH(c, (k_price_base<<<blocks, threads>>>(c, c->d_feat)));
    int seg_blocks = (c->nseg + 127) / 128;
    FEAT_LAUNCH(c, (k_price_wilder<<<seg_blocks, 128>>>(c, c->d_feat)));
    c->base_passes += 2;
    FEAT_END(c);
}

int rc_features_price_col(void *v, int which, double *out)
{
    RcEnrich *c = (RcEnrich *)v;
    if (!c || !out || which < 0 || which >= RC_PA_COUNT) return -1;
    if (!c->d_feat) return -1;
    FEAT_BEGIN(c);
    int rows = (c->n + 255) / 256;
    int segs = c->nseg;
    switch (which) {
    case RC_PA_CLV16:
    case RC_PA_BODY16:
    case RC_PA_TREND96:
        FEAT_LAUNCH(c, (k_price_roll<<<segs, 256>>>(c, c->d_feat, which, c->d_out)));
        break;
    case RC_PA_RUN8:
        FEAT_LAUNCH(c, (k_price_run8<<<segs, 128>>>(c, c->d_out)));
        break;
    default:
        FEAT_LAUNCH(c, (k_price_elem<<<rows, 256>>>(c, c->d_feat, which, c->d_out)));
        break;
    }
    FEAT_DOWNLOAD(c, out);
    FEAT_END(c);
}

int rc_features_base_reads(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    return c ? c->base_reads : -1;
}

int rc_features_base_passes(void *v)
{
    RcEnrich *c = (RcEnrich *)v;
    return c ? c->base_passes : -1;
}

int rc_features_flow_count(void)
{
    return RC_FLOW_COUNT;
}

int rc_features_price_count(void)
{
    return RC_PA_COUNT;
}

} /* extern "C" */
