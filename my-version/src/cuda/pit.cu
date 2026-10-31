/*
 * Device point-in-time as-of joins (issue #302).
 *
 * One kernel family serves two joins:
 *
 *   membership   key = decision `as_of`,     delay = 0
 *   MTF          key = higher bar `time`,    delay = timeframe seconds
 *
 * A leaf (decision or higher bar) is visible to a base row at time `t` when
 * `key + delay <= t`. The predicate is `<=`, never `<`: a decision at `as_of`
 * is visible to a row at `as_of`. Both leaf lists and the base rows are
 * non-decreasing inside one symbol segment, so the join is a monotone merge.
 *
 * The decisions themselves (ranking, ADV tie-breaks, `member`) stay on the CPU
 * and arrive as data. Nothing here parses or ranks.
 *
 * Per run:
 *
 *   1. k_asof: each thread owns a chunk of consecutive base rows. It binary
 *      searches the chunk's first row once, then advances a cursor; the
 *      remaining rows cost one comparison each instead of a fresh search;
 *   2. k_gap_rows: pointwise `keep` (decision found and member), `bar_age`, the
 *      freshness flag and the gap tag. A gap is counted only between two kept
 *      adjacent rows of one segment and one input chain, so `member=0`, listing,
 *      delisting and migration reset the chain and nothing bridges them;
 *   3. an inclusive scan of the chain starts numbers the chains (from 1 inside each
 *      segment), an exclusive
 *      scan of `keep` compacts the kept row indices. No atomic touches an
 *      output.
 *
 * `rc_pit_check` re-derives the no-look-ahead predicate for any leaf index
 * column and counts the violations (an integer sum, so the count is exact).
 *
 * The chunk size and the block size cannot change the result.
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

typedef struct RcPitStats {
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
    unsigned long long loads;      /* dependent key loads in the merge */
    unsigned long long violations; /* look-ahead violations of the last join */
    int n_kept;
    int n_chunks;
} RcPitStats;

typedef struct RcPit {
    int n;
    int nseg;
    int nleaf;

    long long *d_time;
    long long *d_chain;
    int *d_seg_start;
    int *d_seg_len;
    int *d_leaf_start;
    int *d_leaf_len;
    long long *d_key;
    unsigned char *d_member;

    int *d_ch_seg;
    int *d_ch_lo;
    int *d_ch_hi;
    int chunk_cap;

    int *d_leaf;
    double *d_age;
    unsigned char *d_fresh;
    unsigned char *d_keep;
    unsigned char *d_found;
    long long *d_as_of;
    unsigned int *d_gap;
    int *d_start;
    int *d_chain_ord;
    int *d_chain_loc;
    int *d_dest;
    int *d_kept_rows;
    int *d_check_idx;
    int *d_row_seg;
    int *d_first;
    int n_kept;
    int n_chunks;
    unsigned long long *d_counter;
    void *d_scan_tmp;
    size_t scan_tmp_bytes;

    int ran;
    RcPitStats last;
    double batch_start_ms;
} RcPit;

typedef struct RcPitHostOut {
    int *leaf;
    double *age;
    unsigned char *fresh;
    unsigned char *keep;
    unsigned char *found;
    long long *as_of;
    unsigned int *gap;
    int *chain_ord;
    int *kept_rows;
} RcPitHostOut;

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

static void stats_begin(RcPit *c)
{
    memset(&c->last, 0, sizeof(c->last));
    c->batch_start_ms = rc_now_ms();
}

/* phase 0: device pass; 1: creation; 2: download. */
static void stats_end_phase(RcPit *c, int phase)
{
    c->last.wall_ms = rc_now_ms() - c->batch_start_ms;
    if (phase == 0) c->last.kernel_ms = c->last.wall_ms;
    else if (phase == 1) c->last.h2d_ms = c->last.wall_ms - c->last.alloc_ms;
    else c->last.d2h_ms = c->last.wall_ms;
    c->last.batch = 1;
}

static int blocks_for(long long n, int threads)
{
    return (int)((n + threads - 1) / threads);
}

__device__ __forceinline__ long long rc_pit_floor_div(long long a, long long b)
{
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

/* One thread per chunk of consecutive rows of one segment. */
__global__ void k_asof(const long long *__restrict__ t,
                       const int *__restrict__ ch_seg,
                       const int *__restrict__ ch_lo,
                       const int *__restrict__ ch_hi,
                       int nchunk,
                       const int *__restrict__ leaf_start,
                       const int *__restrict__ leaf_len,
                       const long long *__restrict__ key,
                       const unsigned char *__restrict__ member,
                       long long delay, long long fresh_tf,
                       int *__restrict__ leaf,
                       double *__restrict__ age,
                       unsigned char *__restrict__ fresh,
                       unsigned char *__restrict__ keep,
                       unsigned char *__restrict__ found,
                       long long *__restrict__ as_of,
                       unsigned long long *__restrict__ loads)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunk) return;
    int s = ch_seg[c];
    int lo = ch_lo[c];
    int hi = ch_hi[c];
    int ls = leaf_start[s];
    int le = ls + leaf_len[s];
    unsigned long long n_loads = 0;

    long long first = t[lo];
    int a = ls;
    int b = le;
    while (a < b) {
        int m = a + ((b - a) >> 1);
        ++n_loads;
        if (key[m] + delay <= first) a = m + 1;
        else b = m;
    }
    int cursor = a;
    for (int r = lo; r < hi; ++r) {
        long long now = t[r];
        while (cursor < le) {
            ++n_loads;
            if (key[cursor] + delay <= now) ++cursor;
            else break;
        }
        int j = cursor - 1;
        if (j >= ls) {
            long long visible = key[j] + delay;
            leaf[r] = j;
            age[r] = (double)(now - visible);
            fresh[r] = (fresh_tf <= 0 ||
                        key[j] == rc_pit_floor_div(now, fresh_tf) * fresh_tf - fresh_tf)
                           ? 1
                           : 0;
            found[r] = 1;
            keep[r] = member[j] ? 1 : 0;
            as_of[r] = key[j];
        } else {
            leaf[r] = -1;
            age[r] = NAN;
            fresh[r] = 0;
            found[r] = 0;
            keep[r] = 0;
            as_of[r] = 0;
        }
    }
    atomicAdd(loads, n_loads);
}

/* Pointwise gap tag and chain start. Row r continues the previous row when
 * both are kept, in one segment and in one input chain. */
__global__ void k_gap_rows(const long long *__restrict__ t,
                           const long long *__restrict__ chain,
                           const int *__restrict__ first_of_seg,
                           int n, long long step,
                           const unsigned char *__restrict__ keep,
                           unsigned int *__restrict__ gap,
                           int *__restrict__ start)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    bool cont = keep[r] && r > first_of_seg[r] && keep[r - 1] && chain[r] == chain[r - 1];
    unsigned int g = 0;
    if (cont) {
        long long d = (t[r] - t[r - 1]) / step - 1;
        g = d > 0 ? (unsigned int)d : 0u;
    }
    gap[r] = g;
    start[r] = (keep[r] && !cont) ? 1 : 0;
}

/* Number chains from 1 inside each segment, so a prefix of the timeline (or of
 * an earlier symbol) cannot change a later row's id. */
__global__ void k_rebase(const int *__restrict__ first_of_seg,
                         const int *__restrict__ scan, int n,
                         int *__restrict__ local)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    int f = first_of_seg[r];
    local[r] = scan[r] - (f > 0 ? scan[f - 1] : 0);
}

__global__ void k_keep_int(const unsigned char *__restrict__ keep, int n,
                           int *__restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    out[r] = keep[r] ? 1 : 0;
}

__global__ void k_scatter(const unsigned char *__restrict__ keep,
                          const int *__restrict__ dest, int n,
                          int *__restrict__ kept_rows)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    if (keep[r]) kept_rows[dest[r]] = r;
}

/* Independent no-look-ahead check of any leaf index column. */
__global__ void k_check(const long long *__restrict__ t,
                        const int *__restrict__ row_seg,
                        const int *__restrict__ leaf_start,
                        const int *__restrict__ leaf_len,
                        const long long *__restrict__ key,
                        const int *__restrict__ idx, int n, long long delay,
                        unsigned long long *__restrict__ counter)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    int j = idx[r];
    if (j < 0) return;
    int s = row_seg[r];
    int ls = leaf_start[s];
    bool bad = (j < ls) || (j >= ls + leaf_len[s]) || (key[j] + delay > t[r]);
    if (bad) atomicAdd(counter, 1ULL);
}

__global__ void k_row_seg(const int *__restrict__ seg_start,
                          const int *__restrict__ seg_len, int nseg,
                          int *__restrict__ row_seg, int *__restrict__ first_of_seg)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    int a = seg_start[s];
    int e = a + seg_len[s];
    for (int r = a; r < e; ++r) {
        row_seg[r] = s;
        first_of_seg[r] = a;
    }
}

extern "C" {

void rc_pit_destroy(RcPit *c);

RcPit *rc_pit_create(const long long *times, const long long *chain, int n,
                     const int *seg_start, const int *seg_len,
                     const int *leaf_start, const int *leaf_len, int nseg,
                     const long long *key, const unsigned char *member,
                     int nleaf)
{
    if (n <= 0 || nseg <= 0 || nseg > n || nleaf < 0) return NULL;
    RcPit *c = (RcPit *)calloc(1, sizeof(RcPit));
    if (!c) return NULL;
    c->n = n;
    c->nseg = nseg;
    c->nleaf = nleaf;
    stats_begin(c);
    size_t rows = (size_t)n;
    size_t nr = (size_t)nseg;
    size_t nl = (size_t)(nleaf > 0 ? nleaf : 1);
    int ok = 1;
    ok &= rc_alloc((void **)&c->d_time, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_chain, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_start, nr * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_seg_len, nr * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_leaf_start, nr * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_leaf_len, nr * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_key, nl * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_member, nl * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_leaf, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_age, rows * sizeof(double)) == 0;
    ok &= rc_alloc((void **)&c->d_fresh, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_keep, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_found, rows * sizeof(unsigned char)) == 0;
    ok &= rc_alloc((void **)&c->d_as_of, rows * sizeof(long long)) == 0;
    ok &= rc_alloc((void **)&c->d_gap, rows * sizeof(unsigned int)) == 0;
    ok &= rc_alloc((void **)&c->d_start, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_chain_ord, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_chain_loc, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_dest, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_kept_rows, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_check_idx, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_row_seg, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_first, rows * sizeof(int)) == 0;
    ok &= rc_alloc((void **)&c->d_counter, sizeof(unsigned long long)) == 0;
    if (ok) {
        size_t a = 0, b = 0;
        ok &= rc_cub_status(cub::DeviceScan::InclusiveSum(
                  NULL, a, (int *)NULL, (int *)NULL, n)) == cudaSuccess;
        ok &= rc_cub_status(cub::DeviceScan::ExclusiveSum(
                  NULL, b, (int *)NULL, (int *)NULL, n)) == cudaSuccess;
        c->scan_tmp_bytes = a > b ? a : b;
        ok = ok && rc_alloc(&c->d_scan_tmp, c->scan_tmp_bytes) == 0;
    }
    if (!ok) {
        rc_pit_destroy(c);
        return NULL;
    }
    cudaDeviceSynchronize();
    c->last.alloc_ms = rc_now_ms() - c->batch_start_ms;

    rc_upload(c->d_time, times, rows * sizeof(long long));
    rc_upload(c->d_chain, chain, rows * sizeof(long long));
    rc_upload(c->d_seg_start, seg_start, nr * sizeof(int));
    rc_upload(c->d_seg_len, seg_len, nr * sizeof(int));
    rc_upload(c->d_leaf_start, leaf_start, nr * sizeof(int));
    rc_upload(c->d_leaf_len, leaf_len, nr * sizeof(int));
    if (nleaf > 0) {
        rc_upload(c->d_key, key, (size_t)nleaf * sizeof(long long));
        rc_upload(c->d_member, member, (size_t)nleaf * sizeof(unsigned char));
    }
    c->last.h2d_bytes = rows * 16 + nr * 16 + (size_t)nleaf * 9;
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        rc_pit_destroy(c);
        return NULL;
    }
    stats_end_phase(c, 1);
    return c;
}

void rc_pit_destroy(RcPit *c)
{
    if (!c) return;
    rc_free(c->d_time);
    rc_free(c->d_chain);
    rc_free(c->d_seg_start);
    rc_free(c->d_seg_len);
    rc_free(c->d_leaf_start);
    rc_free(c->d_leaf_len);
    rc_free(c->d_key);
    rc_free(c->d_member);
    rc_free(c->d_ch_seg);
    rc_free(c->d_ch_lo);
    rc_free(c->d_ch_hi);
    rc_free(c->d_leaf);
    rc_free(c->d_age);
    rc_free(c->d_fresh);
    rc_free(c->d_keep);
    rc_free(c->d_found);
    rc_free(c->d_as_of);
    rc_free(c->d_gap);
    rc_free(c->d_start);
    rc_free(c->d_chain_ord);
    rc_free(c->d_chain_loc);
    rc_free(c->d_dest);
    rc_free(c->d_kept_rows);
    rc_free(c->d_check_idx);
    rc_free(c->d_row_seg);
    rc_free(c->d_first);
    rc_free(c->d_counter);
    rc_free(c->d_scan_tmp);
    free(c);
}

void rc_pit_last_stats(RcPit *c, RcPitStats *out)
{
    if (c && out) *out = c->last;
}

static int g_force_cub_fail = 0;

extern "C" void rc_pit_test_force_cub_fail(int on)
{
    g_force_cub_fail = on;
}

/* Run one join. `delay` is added to every leaf key; `fresh_tf` > 0 marks a
 * row fresh only when the visible leaf is exactly the previous expected
 * bucket; `step` is the gap bucket size. */
int rc_pit_run(RcPit *c, long long delay, long long fresh_tf, long long step,
               int threads, int chunk_rows)
{
    if (!c || step <= 0 || delay < 0 || fresh_tf < 0) return 1;
    if (threads < 32 || threads > 1024 || threads % 32 != 0) return 2;
    if (chunk_rows < 1) return 3;
    stats_begin(c);

    /* Chunk table: never crosses a segment boundary. */
    int *seg_start = (int *)malloc((size_t)c->nseg * sizeof(int));
    int *seg_len = (int *)malloc((size_t)c->nseg * sizeof(int));
    if (!seg_start || !seg_len) {
        free(seg_start);
        free(seg_len);
        return 4;
    }
    rc_download(seg_start, c->d_seg_start, (size_t)c->nseg * sizeof(int));
    rc_download(seg_len, c->d_seg_len, (size_t)c->nseg * sizeof(int));
    long long nchunk_ll = 0;
    for (int s = 0; s < c->nseg; ++s)
        nchunk_ll += ((long long)seg_len[s] + chunk_rows - 1) / chunk_rows;
    if (nchunk_ll <= 0 || nchunk_ll > 0x7fffffffLL) {
        free(seg_start);
        free(seg_len);
        return 5;
    }
    int nchunk = (int)nchunk_ll;
    int *h = (int *)malloc((size_t)nchunk * 3 * sizeof(int));
    if (!h) {
        free(seg_start);
        free(seg_len);
        return 4;
    }
    {
        int k = 0;
        for (int s = 0; s < c->nseg; ++s) {
            for (int lo = 0; lo < seg_len[s]; lo += chunk_rows) {
                int hi = lo + chunk_rows;
                if (hi > seg_len[s]) hi = seg_len[s];
                h[k] = s;
                h[nchunk + k] = seg_start[s] + lo;
                h[2 * nchunk + k] = seg_start[s] + hi;
                ++k;
            }
        }
    }
    free(seg_start);
    free(seg_len);
    if (nchunk > c->chunk_cap) {
        rc_free(c->d_ch_seg);
        rc_free(c->d_ch_lo);
        rc_free(c->d_ch_hi);
        c->d_ch_seg = c->d_ch_lo = c->d_ch_hi = NULL;
        int ok = rc_alloc((void **)&c->d_ch_seg, (size_t)nchunk * sizeof(int)) == 0;
        ok &= rc_alloc((void **)&c->d_ch_lo, (size_t)nchunk * sizeof(int)) == 0;
        ok &= rc_alloc((void **)&c->d_ch_hi, (size_t)nchunk * sizeof(int)) == 0;
        if (!ok) {
            free(h);
            c->chunk_cap = 0;
            c->last.failed = 1;
            return 6;
        }
        c->chunk_cap = nchunk;
    }
    rc_upload(c->d_ch_seg, h, (size_t)nchunk * sizeof(int));
    rc_upload(c->d_ch_lo, h + nchunk, (size_t)nchunk * sizeof(int));
    rc_upload(c->d_ch_hi, h + 2 * nchunk, (size_t)nchunk * sizeof(int));
    free(h);
    c->last.h2d_bytes += (unsigned long long)nchunk * 12;

    cudaMemset(c->d_counter, 0, sizeof(unsigned long long));
    int nb_chunk = blocks_for(nchunk, threads);
    k_asof<<<nb_chunk, threads>>>(c->d_time, c->d_ch_seg, c->d_ch_lo, c->d_ch_hi,
                                  nchunk, c->d_leaf_start, c->d_leaf_len, c->d_key,
                                  c->d_member, delay, fresh_tf, c->d_leaf, c->d_age,
                                  c->d_fresh, c->d_keep, c->d_found, c->d_as_of,
                                  c->d_counter);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        return 7;
    }
    unsigned long long loads = 0;
    cudaDeviceSynchronize();
    rc_download(&loads, c->d_counter, sizeof(loads));

    /* Per-row segment id and segment start. */
    int *d_row_seg = c->d_row_seg;
    int *d_first = c->d_first;
    int nbs = blocks_for(c->nseg, threads);
    k_row_seg<<<nbs, threads>>>(c->d_seg_start, c->d_seg_len, c->nseg, d_row_seg,
                                d_first);
    int nb = blocks_for(c->n, threads);
    k_gap_rows<<<nb, threads>>>(c->d_time, c->d_chain, d_first, c->n, step,
                                c->d_keep, c->d_gap, c->d_start);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        return 8;
    }

    size_t tmp = c->scan_tmp_bytes;
    err = RC_CUB(g_force_cub_fail, cub::DeviceScan::InclusiveSum(
        c->d_scan_tmp, tmp, c->d_start, c->d_chain_ord, c->n));
    if (err != cudaSuccess) {
        c->last.failed = 1;
        return 11;
    }
    k_rebase<<<nb, threads>>>(d_first, c->d_chain_ord, c->n, c->d_chain_loc);
    k_keep_int<<<nb, threads>>>(c->d_keep, c->n, c->d_start);
    tmp = c->scan_tmp_bytes;
    err = rc_cub_status(cub::DeviceScan::ExclusiveSum(
        c->d_scan_tmp, tmp, c->d_start, c->d_dest, c->n));
    if (err != cudaSuccess) {
        c->last.failed = 1;
        return 12;
    }
    k_scatter<<<nb, threads>>>(c->d_keep, c->d_dest, c->n, c->d_kept_rows);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        return 9;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        c->last.failed = 1;
        return 10;
    }

    int tail_dest = 0;
    unsigned char tail_keep = 0;
    rc_download(&tail_dest, c->d_dest + (c->n - 1), sizeof(int));
    rc_download(&tail_keep, c->d_keep + (c->n - 1), 1);
    c->n_kept = tail_dest + (tail_keep ? 1 : 0);
    c->n_chunks = nchunk;
    c->last.n_kept = c->n_kept;
    c->last.n_chunks = c->n_chunks;
    c->last.loads = loads;
    c->last.d2h_bytes = sizeof(loads) + sizeof(int) + 1;
    c->ran = 1;

    /* Self-check of the join just produced; the expected count is zero. */
    cudaMemset(c->d_counter, 0, sizeof(unsigned long long));
    k_check<<<nb, threads>>>(c->d_time, d_row_seg, c->d_leaf_start, c->d_leaf_len,
                             c->d_key, c->d_leaf, c->n, delay, c->d_counter);
    cudaDeviceSynchronize();
    unsigned long long viol = 0;
    rc_download(&viol, c->d_counter, sizeof(viol));
    c->last.violations = viol;
    c->last.d2h_bytes += sizeof(viol);
    stats_end_phase(c, 0);
    return cudaGetLastError() == cudaSuccess ? 0 : 11;
}

/* Count look-ahead violations of a caller-supplied leaf index column. */
int rc_pit_check(RcPit *c, const int *idx, long long delay,
                 unsigned long long *violations)
{
    if (!c || !idx || !violations || delay < 0) return 1;
    stats_begin(c);
    rc_upload(c->d_check_idx, idx, (size_t)c->n * sizeof(int));
    int *d_row_seg = c->d_row_seg;
    int *d_first = c->d_first;
    int threads = 256;
    k_row_seg<<<blocks_for(c->nseg, threads), threads>>>(
        c->d_seg_start, c->d_seg_len, c->nseg, d_row_seg, d_first);
    cudaMemset(c->d_counter, 0, sizeof(unsigned long long));
    k_check<<<blocks_for(c->n, threads), threads>>>(
        c->d_time, d_row_seg, c->d_leaf_start, c->d_leaf_len, c->d_key,
        c->d_check_idx, c->n, delay, c->d_counter);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    rc_download(violations, c->d_counter, sizeof(unsigned long long));
    c->last.h2d_bytes = (unsigned long long)c->n * sizeof(int);
    c->last.d2h_bytes = sizeof(unsigned long long);
    if (err != cudaSuccess) c->last.failed = 1;
    stats_end_phase(c, 0);
    return err == cudaSuccess ? 0 : 2;
}

int rc_pit_download(RcPit *c, RcPitHostOut *out)
{
    if (!c || !out || !c->ran) return 1;
    stats_begin(c);
    size_t n = (size_t)c->n;
    rc_download(out->leaf, c->d_leaf, n * sizeof(int));
    rc_download(out->age, c->d_age, n * sizeof(double));
    rc_download(out->fresh, c->d_fresh, n);
    rc_download(out->keep, c->d_keep, n);
    rc_download(out->found, c->d_found, n);
    rc_download(out->as_of, c->d_as_of, n * sizeof(long long));
    rc_download(out->gap, c->d_gap, n * sizeof(unsigned int));
    rc_download(out->chain_ord, c->d_chain_loc, n * sizeof(int));
    int keep_n = c->n_kept;
    rc_download(out->kept_rows, c->d_kept_rows, (size_t)keep_n * sizeof(int));
    cudaError_t err = cudaGetLastError();
    c->last.n_kept = c->n_kept;
    c->last.n_chunks = c->n_chunks;
    c->last.d2h_bytes = n * (4 + 8 + 1 + 1 + 1 + 8 + 4 + 4) + (size_t)keep_n * 4;
    stats_end_phase(c, 2);
    return err == cudaSuccess ? 0 : 2;
}

} /* extern "C" */
