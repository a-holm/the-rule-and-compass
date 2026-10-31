/*
 * Device panel preparation: fused duplicate scan, 15 m -> 1 h resample and gap
 * tags (issue #301).
 *
 * Input is one already-decoded symbol-major block. Nothing here parses text,
 * decides units or writes files. The CPU reference is
 * `src/panel_prep.rs::prepare`; this file must produce the same bytes.
 *
 * Per block:
 *
 *   1. stable segmented radix sort by `(symbol segment, time)`, carrying the
 *      original row index so equal timestamps keep their input order;
 *   2. adjacent equal-time runs: bit-identical payloads collapse with
 *      `dup_rows = run_len`; differing payloads are folded exactly once
 *      (first/max/min/last, flows left-to-right, complete AND, traded OR) and
 *      marked a conflict;
 *   3. the conflict list is compacted with an exclusive prefix sum of the run
 *      conflict flags -- never an atomic;
 *   4. unique rows are bucketed with another exclusive scan; one thread per
 *      bucket folds four contiguous input-step instants in order;
 *   5. structurally incomplete buckets are either dropped or emitted with
 *      `cov15m < 4` and PARTIAL_BUCKET; emitted buckets are compacted with a
 *      third scan;
 *   6. gap tags run over the compacted output, reset by a segment change
 *      (listing/delisting) or a chain change (migration/member=0).
 *
 * No atomic touches an output and the block size cannot change the result.
 * All timestamps stay `i64` UTC seconds.
 */

#include <cuda_runtime.h>
#include "alloc_guard.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cub/cub.cuh>

#define RC_PREP_COMPLETE 1u
#define RC_PREP_TRADED 2u
#define RC_PREP_PARTIAL 4u
#define RC_PREP_CONFLICT 8u
#define RC_PREP_OUTAGE 16u

typedef struct RcPrepStats {
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
} RcPrepStats;

typedef struct RcPrep {
    int n;
    int nseg;
    long long in_step;
    long long out_step;
    int keep_partial;

    long long *d_time;
    double *d_open;
    double *d_high;
    double *d_low;
    double *d_close;
    double *d_volume;
    double *d_quote_volume;
    double *d_trades;
    double *d_taker_buy_base;
    double *d_taker_buy_quote;
    unsigned char *d_complete;
    unsigned char *d_traded;
    long long *d_chain;
    int *d_seg_start;
    int *d_seg_len;
    int *d_seg_end;
    int *d_seg_id;

    /* Stable sort by time, carrying the original row index. */
    long long *d_s_time;
    double *d_s_open;
    double *d_s_high;
    double *d_s_low;
    double *d_s_close;
    double *d_s_volume;
    double *d_s_quote_volume;
    double *d_s_trades;
    double *d_s_taker_buy_base;
    double *d_s_taker_buy_quote;
    unsigned char *d_s_complete;
    unsigned char *d_s_traded;
    long long *d_s_chain;
    int *d_s_seg;
    int *d_iota;
    int *d_s_index;
    void *d_sort_tmp;
    size_t sort_tmp_bytes;

    /* Equal-time runs and their compacted unique rows. */
    int *d_run_flags;
    int *d_run_of;
    int *d_run_lo;
    long long *d_u_time;
    double *d_u_open;
    double *d_u_high;
    double *d_u_low;
    double *d_u_close;
    double *d_u_volume;
    double *d_u_quote_volume;
    double *d_u_trades;
    double *d_u_taker_buy_base;
    double *d_u_taker_buy_quote;
    unsigned char *d_u_complete;
    unsigned char *d_u_traded;
    long long *d_u_chain;
    int *d_u_seg;
    int *d_u_dup_rows;
    int *d_u_conflict;
    int *d_u_first_row;
    int *d_u_len;
    int *d_conf_off;
    long long *d_conf_time;
    long long *d_conf_chain;
    int *d_conf_seg;
    int *d_conf_dup_rows;
    int *d_conf_first_row;
    int *d_conf_len;

    /* Bucket flags, compaction and output. */
    int *d_bucket_flags;
    int *d_bucket_of;
    int *d_bucket_lo;
    int *d_emit;
    int *d_out_at;
    long long *d_o_time;
    double *d_o_open;
    double *d_o_high;
    double *d_o_low;
    double *d_o_close;
    double *d_o_volume;
    double *d_o_quote_volume;
    double *d_o_trades;
    double *d_o_taker_buy_base;
    double *d_o_taker_buy_quote;
    unsigned int *d_o_cov;
    unsigned int *d_o_dup_rows;
    unsigned int *d_o_gap;
    unsigned int *d_o_flags;
    long long *d_o_chain;
    int *d_o_seg;
    void *d_scan_tmp;
    size_t scan_tmp_bytes;

    int n_unique;
    int n_out;
    int n_conflict;

    RcPrepStats last;
    double batch_start_ms;
} RcPrep;

typedef struct RcPrepHostOut {
    long long *time;
    double *open;
    double *high;
    double *low;
    double *close;
    double *volume;
    double *quote_volume;
    double *trades;
    double *taker_buy_base;
    double *taker_buy_quote;
    unsigned int *cov15m;
    unsigned int *dup_rows;
    unsigned int *gap;
    unsigned int *flags;
    long long *chain;
    unsigned int *segment;
    long long *conf_time;
    unsigned int *conf_segment;
    long long *conf_chain;
    unsigned int *conf_dup_rows;
    unsigned int *conf_first_row;
    unsigned int *conf_len;
    int n_out;
    int n_conflict;
} RcPrepHostOut;

static double rc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int rc_alloc(void **p, size_t bytes)
{
    if (bytes == 0) {
        *p = NULL;
        return 0;
    }
#ifdef RCSR_UNIFIED
    return (int)rc_checked_malloc_managed(p, bytes);
#else
    return (int)rc_checked_malloc(p, bytes);
#endif
}

static void rc_free(void *p)
{
    if (p) cudaFree(p);
}

static void rc_upload(void *dst, const void *src, size_t bytes)
{
    if (bytes) cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

static void rc_download(void *dst, const void *src, size_t bytes)
{
    if (bytes) cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

static void stats_begin(RcPrep *c)
{
    memset(&c->last, 0, sizeof(c->last));
    c->batch_start_ms = rc_now_ms();
}

/* phase 0: device pass (kernel_ms = wall of the synchronised pass);
 * 1: creation (alloc_ms set by the caller, remainder is the upload);
 * 2: result download. */
static void stats_end_phase(RcPrep *c, int phase)
{
    c->last.wall_ms = rc_now_ms() - c->batch_start_ms;
    if (phase == 0) c->last.kernel_ms = c->last.wall_ms;
    else if (phase == 1) c->last.h2d_ms = c->last.wall_ms - c->last.alloc_ms;
    else c->last.d2h_ms = c->last.wall_ms;
    c->last.batch = 1;
}

#define stats_end(c) stats_end_phase((c), 0)

static int blocks_for(int n, int threads)
{
    return (n + threads - 1) / threads;
}

__device__ __forceinline__ long long rc_prep_floor_div(long long a, long long b)
{
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

__device__ __forceinline__ long long rc_prep_bucket(long long t, long long step)
{
    return rc_prep_floor_div(t, step) * step;
}

/* Bitwise payload equality for one equal-time run. */
__device__ __forceinline__ bool rc_same_payload(RcPrep *c, int a, int b)
{
    if (c->d_s_complete[a] != c->d_s_complete[b]) return false;
    if (c->d_s_traded[a] != c->d_s_traded[b]) return false;
    return __double_as_longlong(c->d_s_open[a]) == __double_as_longlong(c->d_s_open[b]) &&
           __double_as_longlong(c->d_s_high[a]) == __double_as_longlong(c->d_s_high[b]) &&
           __double_as_longlong(c->d_s_low[a]) == __double_as_longlong(c->d_s_low[b]) &&
           __double_as_longlong(c->d_s_close[a]) == __double_as_longlong(c->d_s_close[b]) &&
           __double_as_longlong(c->d_s_volume[a]) == __double_as_longlong(c->d_s_volume[b]) &&
           __double_as_longlong(c->d_s_quote_volume[a]) == __double_as_longlong(c->d_s_quote_volume[b]) &&
           __double_as_longlong(c->d_s_trades[a]) == __double_as_longlong(c->d_s_trades[b]) &&
           __double_as_longlong(c->d_s_taker_buy_base[a]) == __double_as_longlong(c->d_s_taker_buy_base[b]) &&
           __double_as_longlong(c->d_s_taker_buy_quote[a]) == __double_as_longlong(c->d_s_taker_buy_quote[b]);
}

__device__ __forceinline__ bool rc_structural(const long long *time, int lo, int hi,
                                              long long in_step, long long out_step)
{
    if (hi - lo != 4) return false;
    long long start = rc_prep_bucket(time[lo], out_step);
    if (time[lo] != start) return false;
    for (int k = 1; k < 4; ++k) {
        if (time[lo + k] != start + (long long)k * in_step) return false;
    }
    return true;
}

__global__ void rc_prep_iota(int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

__global__ void rc_prep_gather_sorted(const long long *__restrict__ time,
                                const double *__restrict__ open,
                                const double *__restrict__ high,
                                const double *__restrict__ low,
                                const double *__restrict__ close,
                                const double *__restrict__ volume,
                                const double *__restrict__ quote_volume,
                                const double *__restrict__ trades,
                                const double *__restrict__ taker_buy_base,
                                const double *__restrict__ taker_buy_quote,
                                const unsigned char *__restrict__ complete,
                                const unsigned char *__restrict__ traded,
                                const long long *__restrict__ chain,
                                const int *__restrict__ seg,
                                const int *__restrict__ index,
                                int n,
                                long long *__restrict__ s_time,
                                double *__restrict__ s_open,
                                double *__restrict__ s_high,
                                double *__restrict__ s_low,
                                double *__restrict__ s_close,
                                double *__restrict__ s_volume,
                                double *__restrict__ s_quote_volume,
                                double *__restrict__ s_trades,
                                double *__restrict__ s_taker_buy_base,
                                double *__restrict__ s_taker_buy_quote,
                                unsigned char *__restrict__ s_complete,
                                unsigned char *__restrict__ s_traded,
                                long long *__restrict__ s_chain,
                                int *__restrict__ s_seg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = index[i];
    s_time[i] = time[j];
    s_open[i] = open[j];
    s_high[i] = high[j];
    s_low[i] = low[j];
    s_close[i] = close[j];
    s_volume[i] = volume[j];
    s_quote_volume[i] = quote_volume[j];
    s_trades[i] = trades[j];
    s_taker_buy_base[i] = taker_buy_base[j];
    s_taker_buy_quote[i] = taker_buy_quote[j];
    s_complete[i] = complete[j];
    s_traded[i] = traded[j];
    s_chain[i] = chain[j];
    s_seg[i] = seg[j];
}

/* Run start when the segment or the timestamp changes. */
__global__ void rc_prep_run_flags(const long long *__restrict__ time,
                            const int *__restrict__ seg,
                            int n, int *__restrict__ flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || seg[i] != seg[i - 1] || time[i] != time[i - 1]) ? 1 : 0;
}

__global__ void rc_prep_run_lo(const int *__restrict__ run_of,
                         const int *__restrict__ flags,
                         int n, int *__restrict__ run_lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flags[i]) run_lo[run_of[i]] = i;
}

/*
 * One thread per equal-time run. Identical copies are a pure copy; a conflict
 * folds every copy exactly once in stable-sorted (input) order.
 */
__global__ void rc_prep_unique(RcPrep *c, int n_unique)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_unique) return;
    int lo = c->d_run_lo[r];
    int hi = (r + 1 < n_unique) ? c->d_run_lo[r + 1] : c->n;
    bool same = true;
    for (int j = lo + 1; j < hi; ++j) {
        if (!rc_same_payload(c, lo, j)) {
            same = false;
            break;
        }
    }
    c->d_u_time[r] = c->d_s_time[lo];
    c->d_u_chain[r] = c->d_s_chain[lo];
    c->d_u_seg[r] = c->d_s_seg[lo];
    c->d_u_complete[r] = c->d_s_complete[lo];
    c->d_u_traded[r] = c->d_s_traded[lo];
    c->d_u_open[r] = c->d_s_open[lo];
    c->d_u_high[r] = c->d_s_high[lo];
    c->d_u_low[r] = c->d_s_low[lo];
    c->d_u_close[r] = c->d_s_close[lo];
    c->d_u_volume[r] = c->d_s_volume[lo];
    c->d_u_quote_volume[r] = c->d_s_quote_volume[lo];
    c->d_u_trades[r] = c->d_s_trades[lo];
    c->d_u_taker_buy_base[r] = c->d_s_taker_buy_base[lo];
    c->d_u_taker_buy_quote[r] = c->d_s_taker_buy_quote[lo];
    c->d_u_dup_rows[r] = hi - lo;
    c->d_u_first_row[r] = c->d_s_index[lo];
    c->d_u_len[r] = hi - lo;
    c->d_u_conflict[r] = same ? 0 : 1;
    if (same) return;

    unsigned char complete = 1;
    unsigned char traded = 0;
    c->d_u_volume[r] = 0.0;
    c->d_u_quote_volume[r] = 0.0;
    c->d_u_trades[r] = 0.0;
    c->d_u_taker_buy_base[r] = 0.0;
    c->d_u_taker_buy_quote[r] = 0.0;
    for (int j = lo; j < hi; ++j) {
        if (c->d_s_high[j] > c->d_u_high[r]) c->d_u_high[r] = c->d_s_high[j];
        if (c->d_s_low[j] < c->d_u_low[r]) c->d_u_low[r] = c->d_s_low[j];
        c->d_u_volume[r] += c->d_s_volume[j];
        c->d_u_quote_volume[r] += c->d_s_quote_volume[j];
        c->d_u_trades[r] += c->d_s_trades[j];
        c->d_u_taker_buy_base[r] += c->d_s_taker_buy_base[j];
        c->d_u_taker_buy_quote[r] += c->d_s_taker_buy_quote[j];
        complete &= c->d_s_complete[j];
        traded |= c->d_s_traded[j];
    }
    c->d_u_close[r] = c->d_s_close[hi - 1];
    c->d_u_complete[r] = complete;
    c->d_u_traded[r] = traded;
}

__global__ void rc_prep_conflict_scatter(const long long *__restrict__ time,
                                   const long long *__restrict__ chain,
                                   const int *__restrict__ seg,
                                   const int *__restrict__ dup_rows,
                                   const int *__restrict__ first_row,
                                   const int *__restrict__ len,
                                   const int *__restrict__ conflict,
                                   const int *__restrict__ off,
                                   int n_unique,
                                   long long *__restrict__ out_time,
                                   long long *__restrict__ out_chain,
                                   int *__restrict__ out_seg,
                                   int *__restrict__ out_dup_rows,
                                   int *__restrict__ out_first_row,
                                   int *__restrict__ out_len)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_unique || !conflict[r]) return;
    int o = off[r];
    out_time[o] = time[r];
    out_chain[o] = chain[r];
    out_seg[o] = seg[r];
    out_dup_rows[o] = dup_rows[r];
    out_first_row[o] = first_row[r];
    out_len[o] = len[r];
}

/* New bucket when the segment changes or the epoch bucket changes. */
__global__ void rc_prep_bucket_flags(const long long *__restrict__ time,
                               const int *__restrict__ seg,
                               int n, long long out_step,
                               int *__restrict__ flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || seg[i] != seg[i - 1] ||
                rc_prep_bucket(time[i], out_step) != rc_prep_bucket(time[i - 1], out_step))
                   ? 1
                   : 0;
}

__global__ void rc_prep_bucket_lo(const int *__restrict__ bucket_of,
                            const int *__restrict__ flags,
                            int n, int *__restrict__ lo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flags[i]) lo[bucket_of[i]] = i;
}

__global__ void rc_prep_emit_flags(const long long *__restrict__ time,
                             const int *__restrict__ bucket_lo,
                             int n_buckets, int n_unique,
                             long long in_step, long long out_step,
                             int keep_partial, int *__restrict__ emit)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_buckets) return;
    int lo = bucket_lo[b];
    int hi = (b + 1 < n_buckets) ? bucket_lo[b + 1] : n_unique;
    bool structural = rc_structural(time, lo, hi, in_step, out_step);
    emit[b] = (structural || keep_partial) ? 1 : 0;
}

__global__ void rc_prep_fold(RcPrep *c, int n_buckets)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_buckets || !c->d_emit[b]) return;
    int o = c->d_out_at[b];
    int lo = c->d_bucket_lo[b];
    int hi = (b + 1 < n_buckets) ? c->d_bucket_lo[b + 1] : c->n_unique;
    long long start = rc_prep_bucket(c->d_u_time[lo], c->out_step);
    bool structural = rc_structural(c->d_u_time, lo, hi, c->in_step, c->out_step);

    c->d_o_time[o] = start;
    c->d_o_chain[o] = c->d_u_chain[lo];
    c->d_o_seg[o] = c->d_u_seg[lo];
    c->d_o_open[o] = c->d_u_open[lo];
    c->d_o_high[o] = c->d_u_high[lo];
    c->d_o_low[o] = c->d_u_low[lo];
    c->d_o_close[o] = c->d_u_close[lo];
    c->d_o_volume[o] = 0.0;
    c->d_o_quote_volume[o] = 0.0;
    c->d_o_trades[o] = 0.0;
    c->d_o_taker_buy_base[o] = 0.0;
    c->d_o_taker_buy_quote[o] = 0.0;
    unsigned int cov = (unsigned int)(hi - lo);
    unsigned int dup = 0;
    unsigned char complete = structural ? 1 : 0;
    unsigned char conflict = 0;
    for (int j = lo; j < hi; ++j) {
        if (c->d_u_high[j] > c->d_o_high[o]) c->d_o_high[o] = c->d_u_high[j];
        if (c->d_u_low[j] < c->d_o_low[o]) c->d_o_low[o] = c->d_u_low[j];
        c->d_o_volume[o] += c->d_u_volume[j];
        c->d_o_quote_volume[o] += c->d_u_quote_volume[j];
        c->d_o_trades[o] += c->d_u_trades[j];
        c->d_o_taker_buy_base[o] += c->d_u_taker_buy_base[j];
        c->d_o_taker_buy_quote[o] += c->d_u_taker_buy_quote[j];
        dup += (unsigned int)c->d_u_dup_rows[j];
        complete &= c->d_u_complete[j];
        conflict |= (unsigned char)c->d_u_conflict[j];
    }
    c->d_o_close[o] = c->d_u_close[hi - 1];
    c->d_o_cov[o] = cov;
    c->d_o_dup_rows[o] = dup;
    c->d_o_gap[o] = 0;
    unsigned int flags = 0;
    if (complete) flags |= RC_PREP_COMPLETE;
    if (c->d_o_volume[o] > 0.0 || c->d_o_trades[o] > 0.0) flags |= RC_PREP_TRADED;
    if (!structural) flags |= RC_PREP_PARTIAL;
    if (conflict) flags |= RC_PREP_CONFLICT;
    c->d_o_flags[o] = flags;
}

/* Gap tags over the compacted output; segment or chain changes reset. */
__global__ void rc_prep_gap(long long *__restrict__ time,
                      long long *__restrict__ chain,
                      int *__restrict__ seg,
                      unsigned int *__restrict__ gap,
                      unsigned int *__restrict__ flags,
                      int n, long long out_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || i == 0) return;
    unsigned int g;
    if (seg[i] != seg[i - 1] || chain[i] != chain[i - 1]) {
        g = 0;
    } else {
        long long missing = (time[i] - time[i - 1]) / out_step - 1;
        g = missing > 0 ? (unsigned int)missing : 0u;
    }
    gap[i] = g;
    if (g > 0) flags[i] |= RC_PREP_OUTAGE;
}

extern "C" {

void rc_prep_destroy(RcPrep *c);

RcPrep *rc_prep_create(const long long *times, const double *open,
                       const double *high, const double *low,
                       const double *close, const double *volume,
                       const double *quote_volume, const double *trades,
                       const double *taker_buy_base,
                       const double *taker_buy_quote,
                       const unsigned char *complete,
                       const unsigned char *traded,
                       const long long *chain, int n,
                       const int *seg_start, const int *seg_len, int nseg)
{
    if (n <= 0 || nseg <= 0 || nseg > n) return NULL;
    RcPrep *c = (RcPrep *)calloc(1, sizeof(RcPrep));
    if (!c) return NULL;
    c->n = n;
    c->nseg = nseg;
    c->in_step = 900;
    c->out_step = 3600;
    stats_begin(c);
    int ok = 1;
    size_t rows = (size_t)n;
    size_t nr = (size_t)nseg;

    ok &= rc_alloc((void **)&c->d_time, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_open, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_high, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_low, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_close, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_quote_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_trades, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_taker_buy_base, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_taker_buy_quote, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_complete, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_traded, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_chain, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_start, nr * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_len, nr * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_end, nr * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_id, rows * sizeof(int)) == 0;

    ok &= rc_alloc((void **)&c->d_s_time, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_s_open, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_high, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_low, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_close, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_quote_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_trades, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_taker_buy_base, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_taker_buy_quote, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_s_complete, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_s_traded, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_s_chain, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_s_seg, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_iota, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_s_index, rows * sizeof(int)) == 0;

    ok &= rc_alloc((void **)&c->d_run_flags, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_run_of, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_run_lo, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_u_time, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_u_open, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_high, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_low, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_close, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_quote_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_trades, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_taker_buy_base, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_taker_buy_quote, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_u_complete, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_u_traded, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_u_chain, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_u_seg, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_u_dup_rows, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_u_conflict, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_u_first_row, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_u_len, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_conf_off, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_conf_time, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_conf_chain, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_conf_seg, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_conf_dup_rows, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_conf_first_row, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_conf_len, rows * sizeof(int)) == 0;

    ok &= rc_alloc((void **)&c->d_bucket_flags, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_bucket_of, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_bucket_lo, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_emit, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_out_at, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_o_time, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_o_open, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_high, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_low, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_close, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_quote_volume, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_trades, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_taker_buy_base, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_taker_buy_quote, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_o_cov, rows * sizeof(unsigned int)) == 0;
    ok &= rc_alloc((void **)&c->d_o_dup_rows, rows * sizeof(unsigned int)) == 0;
    ok &= rc_alloc((void **)&c->d_o_gap, rows * sizeof(unsigned int)) == 0;
    ok &= rc_alloc((void **)&c->d_o_flags, rows * sizeof(unsigned int)) == 0;
    ok &= rc_alloc((void **)&c->d_o_chain, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_o_seg, rows * sizeof(int)) == 0;

    if (!ok) {
        rc_prep_destroy(c);
        return NULL;
    }
    cudaDeviceSynchronize();
    c->last.alloc_ms = rc_now_ms() - c->batch_start_ms;

    /* Segment table and per-row segment id; the caller guarantees contiguity. */
    {
        int *starts = (int *)malloc(nr * sizeof(int));
        int *lens = (int *)malloc(nr * sizeof(int));
        int *ends = (int *)malloc(nr * sizeof(int));
        int *ids = (int *)malloc(rows * sizeof(int));
        if (!starts || !lens || !ends || !ids) {
            free(starts);
            free(lens);
            free(ends);
            free(ids);
            rc_prep_destroy(c);
            return NULL;
        }
        for (int s = 0; s < nseg; ++s) {
            starts[s] = seg_start[s];
            lens[s] = seg_len[s];
            ends[s] = seg_start[s] + seg_len[s];
            for (int j = 0; j < seg_len[s]; ++j) ids[seg_start[s] + j] = s;
        }
        if (seg_start[0] != 0 || ends[nseg - 1] != n) {
            free(starts);
            free(lens);
            free(ends);
            free(ids);
            rc_prep_destroy(c);
            return NULL;
        }
        rc_upload(c->d_seg_start, starts, nr * sizeof(int));
        rc_upload(c->d_seg_len, lens, nr * sizeof(int));
        rc_upload(c->d_seg_end, ends, nr * sizeof(int));
        rc_upload(c->d_seg_id, ids, rows * sizeof(int));
        free(starts);
        free(lens);
        free(ends);
        free(ids);
    }

    rc_upload(c->d_time, times, rows * sizeof(long long));
    rc_upload(c->d_open, open, rows * sizeof(double));
    rc_upload(c->d_high, high, rows * sizeof(double));
    rc_upload(c->d_low, low, rows * sizeof(double));
    rc_upload(c->d_close, close, rows * sizeof(double));
    rc_upload(c->d_volume, volume, rows * sizeof(double));
    rc_upload(c->d_quote_volume, quote_volume, rows * sizeof(double));
    rc_upload(c->d_trades, trades, rows * sizeof(double));
    rc_upload(c->d_taker_buy_base, taker_buy_base, rows * sizeof(double));
    rc_upload(c->d_taker_buy_quote, taker_buy_quote, rows * sizeof(double));
    rc_upload(c->d_complete, complete, rows * sizeof(unsigned char));
    rc_upload(c->d_traded, traded, rows * sizeof(unsigned char));
    rc_upload(c->d_chain, chain, rows * sizeof(long long));

    /* CUB workspaces: one stable segmented sort and one int scan buffer. */
    c->sort_tmp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        NULL, c->sort_tmp_bytes, (long long *)NULL, (long long *)NULL,
        (int *)NULL, (int *)NULL, n, nseg, c->d_seg_start, c->d_seg_end,
        0, (int)(sizeof(long long) * 8));
    c->scan_tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(NULL, c->scan_tmp_bytes, (int *)NULL,
                                  (int *)NULL, n);
    ok &= rc_alloc(&c->d_sort_tmp, c->sort_tmp_bytes) == 0;
    ok &= rc_alloc(&c->d_scan_tmp, c->scan_tmp_bytes) == 0;
    if (!ok) {
        rc_prep_destroy(c);
        return NULL;
    }

    /* iota is sorted as the carried value; it is never overwritten. */
    {
        int threads = 256;
        int nb = blocks_for(n, threads);
        rc_prep_iota<<<nb, threads>>>(c->d_iota, n);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        rc_prep_destroy(c);
        return NULL;
    }
    cudaDeviceSynchronize();

    c->last.h2d_bytes =
        rows * (sizeof(long long) + 9ULL * sizeof(double) +
                sizeof(unsigned char) * 2ULL + sizeof(long long));
    c->last.h2d_bytes += nr * 2ULL * sizeof(int);
    c->last.h2d_bytes += rows * sizeof(int); /* segment ids */
    stats_end_phase(c, 1);
    return c;
}

void rc_prep_destroy(RcPrep *c)
{
    if (!c) return;
    rc_free(c->d_time);
    rc_free(c->d_open);
    rc_free(c->d_high);
    rc_free(c->d_low);
    rc_free(c->d_close);
    rc_free(c->d_volume);
    rc_free(c->d_quote_volume);
    rc_free(c->d_trades);
    rc_free(c->d_taker_buy_base);
    rc_free(c->d_taker_buy_quote);
    rc_free(c->d_complete);
    rc_free(c->d_traded);
    rc_free(c->d_chain);
    rc_free(c->d_seg_start);
    rc_free(c->d_seg_len);
    rc_free(c->d_seg_end);
    rc_free(c->d_seg_id);
    rc_free(c->d_s_time);
    rc_free(c->d_s_open);
    rc_free(c->d_s_high);
    rc_free(c->d_s_low);
    rc_free(c->d_s_close);
    rc_free(c->d_s_volume);
    rc_free(c->d_s_quote_volume);
    rc_free(c->d_s_trades);
    rc_free(c->d_s_taker_buy_base);
    rc_free(c->d_s_taker_buy_quote);
    rc_free(c->d_s_complete);
    rc_free(c->d_s_traded);
    rc_free(c->d_s_chain);
    rc_free(c->d_s_seg);
    rc_free(c->d_iota);
    rc_free(c->d_s_index);
    rc_free(c->d_sort_tmp);
    rc_free(c->d_run_flags);
    rc_free(c->d_run_of);
    rc_free(c->d_run_lo);
    rc_free(c->d_u_time);
    rc_free(c->d_u_open);
    rc_free(c->d_u_high);
    rc_free(c->d_u_low);
    rc_free(c->d_u_close);
    rc_free(c->d_u_volume);
    rc_free(c->d_u_quote_volume);
    rc_free(c->d_u_trades);
    rc_free(c->d_u_taker_buy_base);
    rc_free(c->d_u_taker_buy_quote);
    rc_free(c->d_u_complete);
    rc_free(c->d_u_traded);
    rc_free(c->d_u_chain);
    rc_free(c->d_u_seg);
    rc_free(c->d_u_dup_rows);
    rc_free(c->d_u_conflict);
    rc_free(c->d_u_first_row);
    rc_free(c->d_u_len);
    rc_free(c->d_conf_off);
    rc_free(c->d_conf_time);
    rc_free(c->d_conf_chain);
    rc_free(c->d_conf_seg);
    rc_free(c->d_conf_dup_rows);
    rc_free(c->d_conf_first_row);
    rc_free(c->d_conf_len);
    rc_free(c->d_bucket_flags);
    rc_free(c->d_bucket_of);
    rc_free(c->d_bucket_lo);
    rc_free(c->d_emit);
    rc_free(c->d_out_at);
    rc_free(c->d_o_time);
    rc_free(c->d_o_open);
    rc_free(c->d_o_high);
    rc_free(c->d_o_low);
    rc_free(c->d_o_close);
    rc_free(c->d_o_volume);
    rc_free(c->d_o_quote_volume);
    rc_free(c->d_o_trades);
    rc_free(c->d_o_taker_buy_base);
    rc_free(c->d_o_taker_buy_quote);
    rc_free(c->d_o_cov);
    rc_free(c->d_o_dup_rows);
    rc_free(c->d_o_gap);
    rc_free(c->d_o_flags);
    rc_free(c->d_o_chain);
    rc_free(c->d_o_seg);
    rc_free(c->d_scan_tmp);
    free(c);
}

void rc_prep_last_stats(RcPrep *c, RcPrepStats *out)
{
    if (c && out) *out = c->last;
}

void rc_prep_counts(RcPrep *c, int *n_out, int *n_conflict, int *n_unique)
{
    if (!c) return;
    if (n_out) *n_out = c->n_out;
    if (n_conflict) *n_conflict = c->n_conflict;
    if (n_unique) *n_unique = c->n_unique;
}

int rc_prep_run(RcPrep *c, long long in_step, long long out_step,
                int keep_partial, int threads)
{
    if (!c || in_step <= 0 || out_step <= 0 || out_step % in_step != 0) return 1;
    if (out_step != 4 * in_step) return 1;
    if (threads < 32 || threads > 1024 || threads % 32 != 0) return 2;
    stats_begin(c);
    c->in_step = in_step;
    c->out_step = out_step;
    c->keep_partial = keep_partial;
    const int n = c->n;
    int nb = blocks_for(n, threads);

    cub::DeviceSegmentedRadixSort::SortPairs(
        c->d_sort_tmp, c->sort_tmp_bytes, c->d_time, c->d_s_time,
        c->d_iota, c->d_s_index, n, c->nseg, c->d_seg_start, c->d_seg_end,
        0, (int)(sizeof(long long) * 8));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 3;
    }

    rc_prep_gather_sorted<<<nb, threads>>>(
        c->d_time, c->d_open, c->d_high, c->d_low, c->d_close,
        c->d_volume, c->d_quote_volume, c->d_trades,
        c->d_taker_buy_base, c->d_taker_buy_quote, c->d_complete,
        c->d_traded, c->d_chain, c->d_seg_id, c->d_s_index, n,
        c->d_s_time, c->d_s_open, c->d_s_high, c->d_s_low, c->d_s_close,
        c->d_s_volume, c->d_s_quote_volume, c->d_s_trades,
        c->d_s_taker_buy_base, c->d_s_taker_buy_quote, c->d_s_complete,
        c->d_s_traded, c->d_s_chain, c->d_s_seg);

    rc_prep_run_flags<<<nb, threads>>>(c->d_s_time, c->d_s_seg, n, c->d_run_flags);
    cub::DeviceScan::ExclusiveSum(c->d_scan_tmp, c->scan_tmp_bytes,
                                  c->d_run_flags, c->d_run_of, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 4;
    }
    int run_tail[2] = {0, 0};
    rc_download(&run_tail[0], c->d_run_of + (n - 1), sizeof(int));
    rc_download(&run_tail[1], c->d_run_flags + (n - 1), sizeof(int));
    c->n_unique = run_tail[0] + run_tail[1];
    c->last.d2h_bytes += 2ULL * sizeof(int);
    if (c->n_unique <= 0 || c->n_unique > n) {
        c->last.failed = 1;
        stats_end(c);
        return 5;
    }

    rc_prep_run_lo<<<nb, threads>>>(c->d_run_of, c->d_run_flags, n, c->d_run_lo);
    int ub = blocks_for(c->n_unique, threads);
    rc_prep_unique<<<ub, threads>>>(c, c->n_unique);

    /* Deterministic conflict list: exclusive scan, then scatter. */
    cub::DeviceScan::ExclusiveSum(c->d_scan_tmp, c->scan_tmp_bytes,
                                  c->d_u_conflict, c->d_conf_off, c->n_unique);
    int conf_tail[2] = {0, 0};
    rc_download(&conf_tail[0], c->d_conf_off + (c->n_unique - 1), sizeof(int));
    rc_download(&conf_tail[1], c->d_u_conflict + (c->n_unique - 1), sizeof(int));
    c->n_conflict = conf_tail[0] + conf_tail[1];
    c->last.d2h_bytes += 2ULL * sizeof(int);
    if (c->n_conflict > 0) {
        rc_prep_conflict_scatter<<<ub, threads>>>(
            c->d_u_time, c->d_u_chain, c->d_u_seg, c->d_u_dup_rows,
            c->d_u_first_row, c->d_u_len, c->d_u_conflict, c->d_conf_off,
            c->n_unique, c->d_conf_time, c->d_conf_chain, c->d_conf_seg,
            c->d_conf_dup_rows, c->d_conf_first_row, c->d_conf_len);
    }

    rc_prep_bucket_flags<<<ub, threads>>>(c->d_u_time, c->d_u_seg, c->n_unique,
                                    c->out_step, c->d_bucket_flags);
    cub::DeviceScan::ExclusiveSum(c->d_scan_tmp, c->scan_tmp_bytes,
                                  c->d_bucket_flags, c->d_bucket_of, c->n_unique);
    int bucket_tail[2] = {0, 0};
    rc_download(&bucket_tail[0], c->d_bucket_of + (c->n_unique - 1), sizeof(int));
    rc_download(&bucket_tail[1], c->d_bucket_flags + (c->n_unique - 1), sizeof(int));
    int n_buckets = bucket_tail[0] + bucket_tail[1];
    c->last.d2h_bytes += 2ULL * sizeof(int);
    if (n_buckets <= 0 || n_buckets > n) {
        c->last.failed = 1;
        stats_end(c);
        return 6;
    }

    rc_prep_bucket_lo<<<ub, threads>>>(c->d_bucket_of, c->d_bucket_flags,
                                 c->n_unique, c->d_bucket_lo);
    int bb = blocks_for(n_buckets, threads);
    rc_prep_emit_flags<<<bb, threads>>>(c->d_u_time, c->d_bucket_lo, n_buckets,
                                  c->n_unique, in_step, out_step,
                                  keep_partial, c->d_emit);
    cub::DeviceScan::ExclusiveSum(c->d_scan_tmp, c->scan_tmp_bytes,
                                  c->d_emit, c->d_out_at, n_buckets);
    int emit_tail[2] = {0, 0};
    rc_download(&emit_tail[0], c->d_out_at + (n_buckets - 1), sizeof(int));
    rc_download(&emit_tail[1], c->d_emit + (n_buckets - 1), sizeof(int));
    c->n_out = emit_tail[0] + emit_tail[1];
    c->last.d2h_bytes += 2ULL * sizeof(int);
    if (c->n_out < 0 || c->n_out > n_buckets) {
        c->last.failed = 1;
        stats_end(c);
        return 7;
    }

    if (c->n_out > 0) {
        rc_prep_fold<<<bb, threads>>>(c, n_buckets);
        int ob = blocks_for(c->n_out, threads);
        rc_prep_gap<<<ob, threads>>>(c->d_o_time, c->d_o_chain, c->d_o_seg,
                               c->d_o_gap, c->d_o_flags, c->n_out, out_step);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 8;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        stats_end(c);
        return 9;
    }
    stats_end(c);
    return 0;
}

int rc_prep_download(RcPrep *c, RcPrepHostOut *out)
{
    if (!c || !out) return 1;
    if (out->n_out != c->n_out || out->n_conflict != c->n_conflict) return 2;
    stats_begin(c);
    int n = c->n_out;
    int m = c->n_conflict;
    if (n > 0) {
        if (!out->time || !out->open || !out->high || !out->low || !out->close ||
            !out->volume || !out->quote_volume || !out->trades ||
            !out->taker_buy_base || !out->taker_buy_quote || !out->cov15m ||
            !out->dup_rows || !out->gap || !out->flags || !out->chain ||
            !out->segment) {
            stats_end_phase(c, 2);
            return 3;
        }
        rc_download(out->time, c->d_o_time, (size_t)n * sizeof(long long));
        rc_download(out->open, c->d_o_open, (size_t)n * sizeof(double));
        rc_download(out->high, c->d_o_high, (size_t)n * sizeof(double));
        rc_download(out->low, c->d_o_low, (size_t)n * sizeof(double));
        rc_download(out->close, c->d_o_close, (size_t)n * sizeof(double));
        rc_download(out->volume, c->d_o_volume, (size_t)n * sizeof(double));
        rc_download(out->quote_volume, c->d_o_quote_volume, (size_t)n * sizeof(double));
        rc_download(out->trades, c->d_o_trades, (size_t)n * sizeof(double));
        rc_download(out->taker_buy_base, c->d_o_taker_buy_base, (size_t)n * sizeof(double));
        rc_download(out->taker_buy_quote, c->d_o_taker_buy_quote, (size_t)n * sizeof(double));
        rc_download(out->cov15m, c->d_o_cov, (size_t)n * sizeof(unsigned int));
        rc_download(out->dup_rows, c->d_o_dup_rows, (size_t)n * sizeof(unsigned int));
        rc_download(out->gap, c->d_o_gap, (size_t)n * sizeof(unsigned int));
        rc_download(out->flags, c->d_o_flags, (size_t)n * sizeof(unsigned int));
        rc_download(out->chain, c->d_o_chain, (size_t)n * sizeof(long long));
        rc_download(out->segment, c->d_o_seg, (size_t)n * sizeof(int));
    }
    if (m > 0) {
        if (!out->conf_time || !out->conf_segment || !out->conf_chain ||
            !out->conf_dup_rows || !out->conf_first_row || !out->conf_len) {
            stats_end_phase(c, 2);
            return 4;
        }
        rc_download(out->conf_time, c->d_conf_time, (size_t)m * sizeof(long long));
        rc_download(out->conf_segment, c->d_conf_seg, (size_t)m * sizeof(int));
        rc_download(out->conf_chain, c->d_conf_chain, (size_t)m * sizeof(long long));
        rc_download(out->conf_dup_rows, c->d_conf_dup_rows, (size_t)m * sizeof(int));
        rc_download(out->conf_first_row, c->d_conf_first_row, (size_t)m * sizeof(int));
        rc_download(out->conf_len, c->d_conf_len, (size_t)m * sizeof(int));
    }
    c->last.d2h_bytes += (unsigned long long)n *
                         (sizeof(long long) + 9ULL * sizeof(double) +
                          4ULL * sizeof(unsigned int) + sizeof(long long) +
                          sizeof(int));
    c->last.d2h_bytes += (unsigned long long)m *
                         (2ULL * sizeof(long long) + 4ULL * sizeof(int));
    stats_end_phase(c, 2);
    return 0;
}

} /* extern "C" */
