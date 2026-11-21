/* #1258: the device arm of `rcsr enumerate`.
 *
 * Included at the end of eval.cu: it runs the interpreter `rc_exec_row` on the
 * resident f32 columns of an RcCtx and allocates through that context's ledger
 * (`rc_dev_alloc`), so a refusal names its bytes and tag and a budget applies.
 *
 * One batch of T trees:
 *   1. rc_enum_eval_kernel: the f32 prediction bits of every tree, a
 *      non-finite flag per tree and the histogram of the top 11 bits of the
 *      order key of every prediction.
 *   2. rc_enum_probe_kernel: the first q distinct values of each tree in row
 *      order (f32 `==`, the probe of `enumerate::grid`); a tree with fewer
 *      keeps all of them and takes the host grid.
 *   3. Exact radix selection of the grid ranks (the same for every tree) on
 *      the u32 order keys, digits of 11, 8, 8 and 5 bits; the order of the
 *      keys is `f64::total_cmp` of the widened predictions. Counts are integer
 *      atomics and picks are integer scans, so a selected key depends on the
 *      key multiset only.
 *   4. rc_enum_score_kernel: per (tree, book, group of 16 cuts) one block of
 *      RC_PNL_WARPS warps walks the 8,192-row segments like pnl_score_kernel.
 *      A row is long for cut k when its order key lies in `[lo_k, hi_k]`, the
 *      host's exact image of `isfinite(p) && p > threshold` with `p` the f64
 *      `pred - c` or `c - pred`. Lane `l` adds the excess of the long rows
 *      `seg_start + l + 32j` in ascending `j` per (block, cut), the lanes fold
 *      16, 8, 4, 2, 1, each warp adds its segment sums in ascending order and
 *      the warp sums fold 4, 2, 1: `FitnessBook::block_totals` term for term.
 *      Flips, exits and long rows are popcounts of the cut's ballot. No
 *      floating-point atomics.
 *
 * Every kernel goes through rc_launch; a failure records its step in
 * rc_enum_last_error. */
#ifndef RCSR_ENUMERATE_H
#define RCSR_ENUMERATE_H

#define RC_ENUM_CUTS 16
#define RC_ENUM_MAX_Q 64
#define RC_ENUM_MAX_RANKS (2 * RC_ENUM_MAX_Q)
#define RC_ENUM_TOP_BINS 2048
#define RC_ENUM_LOW_BINS 256
#define RC_ENUM_SLOT_CHUNK 32
#define RC_ENUM_PROBE_WARPS 4
#define RC_ENUM_PASSES 4

static_assert(RC_ENUM_TOP_BINS == RS_BINS, "pass 0 flushes through rs_flush");
static_assert(RC_ENUM_CUTS <= 32, "lane k owns the counts of cut k");

typedef struct RcEnumTotals {
    double gross;
    unsigned int flips;
    unsigned int exits;
    unsigned int longs;
    unsigned int pad;
} RcEnumTotals;

struct RcEnum {
    RcCtx *c;
    int n;
    int nb;
    int q;
    int nq;
    int groups;
    int tree_cap;
    int word_cap;
    int book_cap;
    unsigned char *d_rows;
    double *d_excess;
    unsigned int *d_prog;
    unsigned int *d_pred;
    unsigned char *d_tree;
    unsigned int *d_hist;
    RcEnumTotals *d_tot;
    /* Views into d_rows, d_prog, d_tree and d_hist. */
    unsigned char *flags;
    unsigned char *bids;
    unsigned int *off;
    int *nonfinite;
    int *dcount;
    int *status;
    int *nslots;
    int *ncuts;
    int *bad;
    int *qslot;
    unsigned int *dvals;
    unsigned int *ranks;
    unsigned int *qpre;
    unsigned int *qres;
    unsigned int *spre;
    unsigned int *lo;
    unsigned int *hi;
    unsigned int *hist1;
    unsigned int *histq;
    cudaEvent_t e0, e1;
    int events;
    double eval_ms;
    double score_ms;
};

static thread_local char g_enum_err[256] = "";

static void rc_enum_fail(const char *step, cudaError_t e)
{
    char who[96] = "";
    const int arg = rc_launch_last_refusal(who, (int)sizeof(who));
    if (e == cudaErrorInvalidValue && arg >= 0 && strcmp(who, step) == 0) {
        snprintf(g_enum_err, sizeof(g_enum_err), "the launch guard refused %s (argument %d)", step, arg);
    } else {
        snprintf(g_enum_err, sizeof(g_enum_err), "%s failed: %s", step, cudaGetErrorString(e));
    }
    cudaGetLastError();
}

/* The u32 key whose unsigned order is the total order of the f32 bits. */
__host__ __device__ __forceinline__ unsigned int rc_enum_key(unsigned int bits)
{
    return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__host__ __device__ __forceinline__ int rc_enum_shift(int pass)
{
    return pass == 0 ? 21 : pass == 1 ? 13 : pass == 2 ? 5 : 0;
}

__host__ __device__ __forceinline__ int rc_enum_width(int pass)
{
    return pass == 0 ? 11 : pass == 3 ? 5 : 8;
}

/* The key bits fixed before pass `pass`. */
__host__ __device__ __forceinline__ unsigned int rc_enum_fixed(int pass)
{
    if (pass == 0) return 0u;
    const int keep = rc_enum_shift(pass) + rc_enum_width(pass);
    return ~((1u << keep) - 1u);
}

/* Grid `(trees, row blocks)`, RS_THREADS threads. */
__global__ void __launch_bounds__(RS_THREADS) rc_enum_eval_kernel(
    const float *__restrict__ cols,
    int n,
    const unsigned int *__restrict__ prog,
    const unsigned int *__restrict__ off,
    unsigned int *__restrict__ pred,
    int *__restrict__ nonfinite,
    unsigned int *__restrict__ hist1)
{
    __shared__ unsigned int bins[RC_ENUM_TOP_BINS];
    const int t = blockIdx.x;
    for (int b = threadIdx.x; b < RC_ENUM_TOP_BINS; b += blockDim.x) bins[b] = 0u;
    __syncthreads();
    const unsigned int *code = prog + off[t];
    unsigned int *out = pred + (size_t)t * (size_t)n;
    int bad = 0;
    const long long stride = (long long)gridDim.y * blockDim.x;
    for (long long base = (long long)blockIdx.y * blockDim.x; base < n; base += stride) {
        const long long row = base + threadIdx.x;
        unsigned int d = RS_NO_DIGIT;
        if (row < n) {
            const float p = rc_exec_row(code, cols, n, (int)row);
            const unsigned int bits = __float_as_uint(p);
            out[row] = bits;
            bad |= !isfinite(p);
            d = rc_enum_key(bits) >> rc_enum_shift(0);
        }
        rs_count(bins, d);
    }
    bad = __syncthreads_or(bad);
    if (threadIdx.x == 0 && bad) atomicOr(&nonfinite[t], 1);
    rs_flush(bins, 1, hist1 + (size_t)t * RC_ENUM_TOP_BINS);
}

/* One warp per tree. A row joins the set when no member equals it; the new
   values of one warp step join in lane (row) order, so the set holds the
   first occurrence of each value, as the CPU probe does. Status: 1 a
   non-finite prediction, 2 fewer than q distinct values, 0 the rank grid. */
__global__ void __launch_bounds__(32 * RC_ENUM_PROBE_WARPS) rc_enum_probe_kernel(
    const unsigned int *__restrict__ pred,
    int n,
    int ntrees,
    int q,
    const int *__restrict__ nonfinite,
    int *__restrict__ dcount,
    unsigned int *__restrict__ dvals,
    int *__restrict__ status)
{
    __shared__ float set[RC_ENUM_PROBE_WARPS][RC_ENUM_MAX_Q];
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * RC_ENUM_PROBE_WARPS + w;
    if (t >= ntrees) return;
    const unsigned int *p = pred + (size_t)t * (size_t)n;
    int cnt = 0;
    for (long long base = 0; base < n && cnt < q; base += 32) {
        const long long row = base + lane;
        const int in = row < n;
        const unsigned int b = in ? p[row] : 0u;
        const float v = __uint_as_float(b);
        int fresh = in;
        for (int j = 0; j < cnt && fresh; ++j) fresh = !(set[w][j] == v);
        unsigned int m = __ballot_sync(0xffffffffu, fresh);
        while (m && cnt < q) {
            const int leader = __ffs(m) - 1;
            const unsigned int lb = __shfl_sync(0xffffffffu, b, leader);
            if (lane == 0) {
                set[w][cnt] = __uint_as_float(lb);
                dvals[(size_t)t * q + cnt] = lb;
            }
            ++cnt;
            if (lane == leader || (fresh && v == __uint_as_float(lb))) fresh = 0;
            m = __ballot_sync(0xffffffffu, fresh);
        }
        __syncwarp();
    }
    if (lane == 0) {
        dcount[t] = cnt;
        status[t] = nonfinite[t] ? 1 : (cnt < q ? 2 : 0);
    }
}

/* Passes 1 to 3: per tree, the digit of every key whose fixed bits equal a
   slot prefix (the distinct prefixes of the queries, ascending), slots in
   shared chunks of RC_ENUM_SLOT_CHUNK. Grid `(trees, row blocks)`. */
__global__ void __launch_bounds__(RS_THREADS) rc_enum_hist_kernel(
    int pass,
    const unsigned int *__restrict__ pred,
    int n,
    int nq,
    const unsigned int *__restrict__ spre,
    const int *__restrict__ nslots,
    const int *__restrict__ status,
    unsigned int *__restrict__ hist)
{
    __shared__ unsigned int bins[RC_ENUM_SLOT_CHUNK * RC_ENUM_LOW_BINS];
    __shared__ unsigned int sp[RC_ENUM_MAX_RANKS];
    const int t = blockIdx.x;
    if (status[t] != 0) return;
    const int ns = nslots[t];
    for (int i = threadIdx.x; i < ns; i += blockDim.x) sp[i] = spre[(size_t)t * nq + i];
    const int shift = rc_enum_shift(pass);
    const unsigned int dmask = (1u << rc_enum_width(pass)) - 1u;
    const unsigned int fixed = rc_enum_fixed(pass);
    const unsigned int *ks = pred + (size_t)t * (size_t)n;
    const long long stride = (long long)gridDim.y * blockDim.x;
    for (int c0 = 0; c0 < ns; c0 += RC_ENUM_SLOT_CHUNK) {
        const int cn = min(RC_ENUM_SLOT_CHUNK, ns - c0);
        for (int b = threadIdx.x; b < cn * RC_ENUM_LOW_BINS; b += blockDim.x) bins[b] = 0u;
        __syncthreads();
        for (long long base = (long long)blockIdx.y * blockDim.x; base < n; base += stride) {
            const long long row = base + threadIdx.x;
            unsigned int d = RS_NO_DIGIT;
            if (row < n) {
                const unsigned int key = rc_enum_key(ks[row]);
                const unsigned int fk = key & fixed;
                int a = c0, z = c0 + cn;
                while (a < z) {
                    const int mid = (a + z) >> 1;
                    if (sp[mid] < fk) a = mid + 1;
                    else z = mid;
                }
                if (a < c0 + cn && sp[a] == fk) {
                    d = ((unsigned int)(a - c0) << 8) | ((key >> shift) & dmask);
                }
            }
            rs_count(bins, d);
        }
        __syncthreads();
        unsigned int *h = hist + ((size_t)t * nq + c0) * RC_ENUM_LOW_BINS;
        for (int b = threadIdx.x; b < cn * RC_ENUM_LOW_BINS; b += blockDim.x) {
            const unsigned int v = bins[b];
            if (v != 0u) atomicAdd(&h[b], v);
        }
        __syncthreads();
    }
}

/* Fix one digit of every query of a tree, then the slots of the next pass.
   One block of RS_THREADS per tree. A rank outside the counted keys cannot
   happen for a consistent histogram; it sets `bad` so the host discards the
   batch. */
__global__ void __launch_bounds__(RS_THREADS) rc_enum_pick_kernel(
    int pass,
    int nq,
    const unsigned int *__restrict__ ranks,
    const unsigned int *__restrict__ hist1,
    const unsigned int *__restrict__ histq,
    const int *__restrict__ status,
    unsigned int *__restrict__ qpre,
    unsigned int *__restrict__ qres,
    int *__restrict__ qslot,
    unsigned int *__restrict__ spre,
    int *__restrict__ nslots,
    int *__restrict__ bad)
{
    __shared__ unsigned int warp_sum[RS_THREADS / 32];
    __shared__ int found_digit;
    __shared__ unsigned int found_below;
    const int t = blockIdx.x;
    if (status[t] != 0) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int bins = 1 << rc_enum_width(pass);
    const int per = bins >= RS_THREADS ? bins / RS_THREADS : 1;
    const int shift = rc_enum_shift(pass);
    const size_t base = (size_t)t * nq;
    for (int j = 0; j < nq; ++j) {
        const unsigned int *h = pass == 0
            ? hist1 + (size_t)t * RC_ENUM_TOP_BINS
            : histq + (base + (size_t)qslot[base + j]) * RC_ENUM_LOW_BINS;
        const unsigned int r = pass == 0 ? ranks[j] : qres[base + j];
        unsigned int mine = 0u;
        for (int i = 0; i < per; ++i) {
            const int b = tid * per + i;
            if (b < bins) mine += h[b];
        }
        unsigned int incl = mine;
        for (int d = 1; d < 32; d <<= 1) {
            const unsigned int v = __shfl_up_sync(0xffffffffu, incl, d);
            if (lane >= d) incl += v;
        }
        if (lane == 31) warp_sum[wid] = incl;
        if (tid == 0) found_digit = -1;
        __syncthreads();
        unsigned int before = 0u;
        for (int w = 0; w < wid; ++w) before += warp_sum[w];
        incl += before;
        const unsigned int excl = incl - mine;
        if (r >= excl && r < incl) {
            unsigned int below = excl;
            for (int i = 0; i < per; ++i) {
                const int b = tid * per + i;
                const unsigned int cnt = b < bins ? h[b] : 0u;
                if (r < below + cnt) {
                    found_digit = b;
                    found_below = below;
                    break;
                }
                below += cnt;
            }
        }
        __syncthreads();
        if (tid == 0) {
            if (found_digit < 0) {
                bad[t] = 1;
            } else {
                const unsigned int p = pass == 0 ? 0u : qpre[base + j];
                qpre[base + j] = p | ((unsigned int)found_digit << shift);
                qres[base + j] = r - found_below;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        int s = -1;
        unsigned int prev = 0u;
        for (int j = 0; j < nq; ++j) {
            const unsigned int p = qpre[base + j];
            if (j == 0 || p != prev) {
                ++s;
                spre[base + s] = p;
                prev = p;
            }
            qslot[base + j] = s;
        }
        nslots[t] = s + 1;
    }
}

/* One block per (tree, book, group of RC_ENUM_CUTS cuts), RC_PNL_WARPS warps.
   NBMAX is 1, RC_PNL_MAX_BLOCKS or RC_PNL_WIDE_BLOCKS; `bids` is null for
   one block. Lane k owns the counts and warp sums of cut k of the group. The
   lane sums of the block being walked stay in registers; the other blocks of
   the segment wait in `sg`, so every (block, lane, cut) sum takes its rows in
   ascending order. */
__device__ int g_enum_red_atomic = 0;

template <int NBMAX>
__global__ void __launch_bounds__(RC_MAX_THREADS, 2) rc_enum_score_kernel(
    const unsigned int *__restrict__ pred,
    int n,
    const double *__restrict__ excess,
    const unsigned char *__restrict__ flags,
    const unsigned char *__restrict__ bids,
    int nb,
    int q,
    int nbooks,
    int groups,
    const unsigned int *__restrict__ lo,
    const unsigned int *__restrict__ hi,
    const int *__restrict__ ncuts,
    RcEnumTotals *__restrict__ tot)
{
    const int g = blockIdx.x % groups;
    const int bk = (blockIdx.x / groups) % nbooks;
    const int t = blockIdx.x / (groups * nbooks);
    const int k0 = g * RC_ENUM_CUTS;
    const int all = ncuts[t];
    if (k0 >= all) return;
    const int nc = min(RC_ENUM_CUTS, all - k0);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ double sh_w[RC_PNL_WARPS][RC_ENUM_CUTS];
    __shared__ unsigned int sh_c[3][RC_ENUM_CUTS];
    __shared__ double sh_red[RC_ENUM_CUTS];
    const size_t cbase = (size_t)t * q + k0;
    unsigned int clo[RC_ENUM_CUTS], chi[RC_ENUM_CUTS];
#pragma unroll
    for (int k = 0; k < RC_ENUM_CUTS; ++k) {
        clo[k] = k < nc ? lo[cbase + k] : 1u;
        chi[k] = k < nc ? hi[cbase + k] : 0u;
    }
    const unsigned int my_lo = lane < nc ? lo[cbase + lane] : 1u;
    const unsigned int my_hi = lane < nc ? hi[cbase + lane] : 0u;
    const unsigned int *pk = pred + (size_t)t * (size_t)n;
    const double *ex = excess + (size_t)bk * (size_t)n;
    const int top = nb - 1;

    double wsum[NBMAX];
    unsigned int cnt[NBMAX][3];
    double sg[NBMAX > 1 ? NBMAX : 1][RC_ENUM_CUTS];
#pragma unroll
    for (int b = 0; b < NBMAX; ++b) {
        wsum[b] = 0.0;
        cnt[b][0] = 0u;
        cnt[b][1] = 0u;
        cnt[b][2] = 0u;
    }
    const long long rows = (long long)n;
    const long long nseg = (rows + RC_PNL_SEGMENT_ROWS - 1) / RC_PNL_SEGMENT_ROWS;
    for (long long s = warp; s < nseg; s += RC_PNL_WARPS) {
        const long long seg_start = s * RC_PNL_SEGMENT_ROWS;
        const long long seg_end = seg_start + RC_PNL_SEGMENT_ROWS < rows ? seg_start + RC_PNL_SEGMENT_ROWS : rows;
        unsigned int carry = 0u;
        if (seg_start > 0) {
            const unsigned int kp = rc_enum_key(pk[seg_start - 1]);
            carry = (kp >= my_lo && kp <= my_hi) ? 1u : 0u;
        }
        int bcur = 0;
        if (NBMAX > 1) bcur = min((int)bids[seg_start], top);
        unsigned long long touched = 0ull;
        double acc[RC_ENUM_CUTS];
#pragma unroll
        for (int k = 0; k < RC_ENUM_CUTS; ++k) acc[k] = 0.0;
        unsigned int cf = 0u, ce = 0u, cl = 0u;
        for (long long base = seg_start; base < seg_end; base += 32) {
            const long long row = base + lane;
            const int in = row < seg_end;
            unsigned int key = 0u, f = 0u;
            int bid = bcur;
            double e = 0.0;
            if (in) {
                key = rc_enum_key(pk[row]);
                e = ex[row];
                f = flags[row];
                if (NBMAX > 1) bid = min((int)bids[row], top);
            }
            const unsigned int fst = __ballot_sync(0xffffffffu, (f & 1u) != 0u);
            const unsigned int lst = __ballot_sync(0xffffffffu, (f & 2u) != 0u);
            const unsigned int val = __ballot_sync(0xffffffffu, in);
            unsigned int mine = 0u, lmask = 0u;
#pragma unroll
            for (int k = 0; k < RC_ENUM_CUTS; ++k) {
                const int lg = in && key >= clo[k] && key <= chi[k];
                lmask |= (unsigned int)lg << k;
                const unsigned int m = __ballot_sync(0xffffffffu, lg);
                if (lane == k) mine = m;
            }
            const unsigned int prev = ((mine << 1) | carry) & ~fst;
            const unsigned int fl = (mine ^ prev) & val;
            const unsigned int xt = mine & lst;
            carry = (mine >> 31) & 1u;
            if (NBMAX == 1 || __all_sync(0xffffffffu, !in || bid == bcur)) {
                cf += __popc(fl);
                ce += __popc(xt);
                cl += __popc(mine);
#pragma unroll
                for (int k = 0; k < RC_ENUM_CUTS; ++k) {
                    if ((lmask >> k) & 1u) acc[k] += e;
                }
                continue;
            }
            if constexpr (NBMAX > 1) {
#pragma unroll
                for (int k = 0; k < RC_ENUM_CUTS; ++k) sg[bcur][k] = acc[k];
                touched |= 1ull << bcur;
                cnt[bcur][0] += cf;
                cnt[bcur][1] += ce;
                cnt[bcur][2] += cl;
                cf = 0u;
                ce = 0u;
                cl = 0u;
                unsigned long long present = 0ull;
                unsigned int todo = val;
                while (todo) {
                    const int kb = __shfl_sync(0xffffffffu, bid, __ffs(todo) - 1);
                    const unsigned int bm = __ballot_sync(0xffffffffu, in && bid == kb);
                    present |= 1ull << kb;
                    cnt[kb][0] += __popc(fl & bm);
                    cnt[kb][1] += __popc(xt & bm);
                    cnt[kb][2] += __popc(mine & bm);
                    todo &= ~bm;
                }
                unsigned long long fresh = present & ~touched;
                while (fresh) {
                    const int b = __ffsll((long long)fresh) - 1;
#pragma unroll
                    for (int k = 0; k < RC_ENUM_CUTS; ++k) sg[b][k] = 0.0;
                    fresh &= fresh - 1ull;
                }
                touched |= present;
                if (in) {
#pragma unroll
                    for (int k = 0; k < RC_ENUM_CUTS; ++k) {
                        if ((lmask >> k) & 1u) sg[bid][k] += e;
                    }
                }
                bcur = __shfl_sync(0xffffffffu, bid, 31 - __clz(val));
#pragma unroll
                for (int k = 0; k < RC_ENUM_CUTS; ++k) acc[k] = sg[bcur][k];
            }
        }
        cnt[NBMAX > 1 ? bcur : 0][0] += cf;
        cnt[NBMAX > 1 ? bcur : 0][1] += ce;
        cnt[NBMAX > 1 ? bcur : 0][2] += cl;
        if constexpr (NBMAX > 1) {
#pragma unroll
            for (int k = 0; k < RC_ENUM_CUTS; ++k) sg[bcur][k] = acc[k];
            touched |= 1ull << bcur;
        } else {
            touched = 1ull;
        }
        while (touched) {
            const int b = __ffsll((long long)touched) - 1;
            touched &= touched - 1ull;
#pragma unroll
            for (int k = 0; k < RC_ENUM_CUTS; ++k) {
                double v;
                if constexpr (NBMAX > 1) {
                    v = sg[b][k];
                } else {
                    v = acc[k];
                }
                for (int half = 16; half >= 1; half >>= 1) v += __shfl_down_sync(0xffffffffu, v, half);
                v = __shfl_sync(0xffffffffu, v, 0);
                if (lane == k) wsum[NBMAX > 1 ? b : 0] += v;
            }
        }
    }
    for (int b = 0; b < nb; ++b) {
        if (threadIdx.x < 3 * RC_ENUM_CUTS) sh_c[threadIdx.x / RC_ENUM_CUTS][threadIdx.x % RC_ENUM_CUTS] = 0u;
        if (threadIdx.x < RC_ENUM_CUTS) sh_red[threadIdx.x] = 0.0;
        __syncthreads();
        if (lane < nc) {
            const int bb = NBMAX > 1 ? b : 0;
            sh_w[warp][lane] = wsum[bb];
            if (g_enum_red_atomic) atomicAdd(&sh_red[lane], wsum[bb]);
            atomicAdd(&sh_c[0][lane], cnt[bb][0]);
            atomicAdd(&sh_c[1][lane], cnt[bb][1]);
            atomicAdd(&sh_c[2][lane], cnt[bb][2]);
        }
        __syncthreads();
        if (threadIdx.x < nc) {
            const int k = threadIdx.x;
            double w[RC_PNL_WARPS];
#pragma unroll
            for (int i = 0; i < RC_PNL_WARPS; ++i) w[i] = sh_w[i][k];
            for (int half = RC_PNL_WARPS / 2; half >= 1; half >>= 1) {
                for (int i = 0; i < half; ++i) w[i] = __dadd_rn(w[i], w[i + half]);
            }
            RcEnumTotals *o = tot + (((size_t)t * nbooks + bk) * q + k0 + k) * nb + b;
            o->gross = w[0];
            if (g_enum_red_atomic) o->gross = sh_red[k];
            o->flips = sh_c[0][k];
            o->exits = sh_c[1][k];
            o->longs = sh_c[2][k];
            o->pad = 0u;
        }
        __syncthreads();
    }
}

static int rc_enum_row_blocks(int n)
{
    return rs_blocks_per_segment(n);
}

static size_t rc_align256(size_t b)
{
    return (b + 255) & ~(size_t)255;
}

static void rc_enum_free(RcEnum *h)
{
    if (!h) return;
    RcCtx *c = h->c;
    rc_dev_free(c, h->d_rows);
    rc_dev_free(c, h->d_excess);
    rc_dev_free(c, h->d_prog);
    rc_dev_free(c, h->d_pred);
    rc_dev_free(c, h->d_tree);
    rc_dev_free(c, h->d_hist);
    rc_dev_free(c, h->d_tot);
    if (h->events) {
        cudaEventDestroy(h->e0);
        cudaEventDestroy(h->e1);
    }
    free(h);
}

/* Per-thread local bytes of the score kernel for `nb` blocks times the
   threads the device can hold: the driver backs that much local memory. */
static size_t rc_enum_local_bytes_for(int nb)
{
    cudaFuncAttributes a;
    cudaError_t e;
    if (nb <= 1) e = cudaFuncGetAttributes(&a, rc_enum_score_kernel<1>);
    else if (nb <= RC_PNL_MAX_BLOCKS) e = cudaFuncGetAttributes(&a, rc_enum_score_kernel<RC_PNL_MAX_BLOCKS>);
    else e = cudaFuncGetAttributes(&a, rc_enum_score_kernel<RC_PNL_WIDE_BLOCKS>);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    int dev = 0, sms = 0, per_sm = 0;
    if (cudaGetDevice(&dev) != cudaSuccess
        || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
        || cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return (size_t)a.localSizeBytes * (size_t)sms * (size_t)per_sm;
}

extern "C" {

void rc_enum_red_atomic_for_tests(int on)
{
    cudaMemcpyToSymbol(g_enum_red_atomic, &on, sizeof(int));
}

const char *rc_enum_last_error(void)
{
    return g_enum_err;
}

size_t rc_enum_local_bytes(int nb)
{
    return rc_enum_local_bytes_for(nb);
}

/* `bids` holds one block id per row when nb > 1 and is ignored otherwise.
   `ranks` are the ascending distinct grid ranks (nq of them, at most
   RC_ENUM_MAX_RANKS; 0 when the rows are too few for a rank grid). */
void *rc_enum_create(void *ctx, const unsigned char *flags, const unsigned char *bids, int nb,
                     int tree_cap, int word_cap, int book_cap, int q,
                     const unsigned int *ranks, int nq)
{
    g_enum_err[0] = 0;
    rc_alloc_fail_clear();
    RcCtx *c = (RcCtx *)ctx;
    if (!c || c->tiled || !c->d_cols || !flags || nb < 1 || nb > RC_PNL_WIDE_BLOCKS || (nb > 1 && !bids)
        || tree_cap < 1 || word_cap < 1 || book_cap < 1 || q < 2 || q > RC_ENUM_MAX_Q || q % 2 != 0
        || nq < 0 || nq > RC_ENUM_MAX_RANKS || (nq > 0 && !ranks) || c->n < 1) {
        snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_create: invalid arguments");
        return NULL;
    }
    RcEnum *h = (RcEnum *)calloc(1, sizeof(RcEnum));
    if (!h) {
        snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_create: host allocation failed");
        return NULL;
    }
    const size_t n = (size_t)c->n;
    const size_t T = (size_t)tree_cap;
    const size_t nqa = nq > 0 ? (size_t)nq : 1;
    h->c = c;
    h->n = c->n;
    h->nb = nb;
    h->q = q;
    h->nq = nq;
    h->groups = (q + RC_ENUM_CUTS - 1) / RC_ENUM_CUTS;
    h->tree_cap = tree_cap;
    h->word_cap = word_cap;
    h->book_cap = book_cap;
    const size_t rows_bytes = rc_align256(n) + (nb > 1 ? rc_align256(n) : 0);
    const size_t prog_bytes = rc_align256(sizeof(unsigned int) * (size_t)word_cap) + rc_align256(sizeof(unsigned int) * (T + 1));
    /* d_tree: 6 ints per tree, dvals, ranks, qpre, qres, spre, qslot, lo, hi. */
    const size_t ints = rc_align256(sizeof(int) * T * 6);
    const size_t dv = rc_align256(sizeof(unsigned int) * T * (size_t)q);
    const size_t rk = rc_align256(sizeof(unsigned int) * nqa);
    const size_t qq = rc_align256(sizeof(unsigned int) * T * nqa);
    const size_t cut = rc_align256(sizeof(unsigned int) * T * (size_t)q);
    const size_t tree_bytes = ints + dv + rk + 4 * qq + 2 * cut;
    const size_t h1 = rc_align256(sizeof(unsigned int) * T * RC_ENUM_TOP_BINS);
    const size_t hist_bytes = h1 + rc_align256(sizeof(unsigned int) * T * nqa * RC_ENUM_LOW_BINS);
    const size_t tot_bytes = sizeof(RcEnumTotals) * T * (size_t)book_cap * (size_t)q * (size_t)nb;
    int rc = 0;
    if (!rc) rc = rc_dev_alloc(c, "enum_rows", (void **)&h->d_rows, rows_bytes);
    if (!rc) rc = rc_dev_alloc(c, "enum_books", (void **)&h->d_excess, sizeof(double) * n * (size_t)book_cap);
    if (!rc) rc = rc_dev_alloc(c, "enum_prog", (void **)&h->d_prog, prog_bytes);
    if (!rc) rc = rc_dev_alloc(c, "enum_pred", (void **)&h->d_pred, sizeof(unsigned int) * n * T);
    if (!rc) rc = rc_dev_alloc(c, "enum_tree", (void **)&h->d_tree, tree_bytes);
    if (!rc) rc = rc_dev_alloc(c, "enum_hist", (void **)&h->d_hist, hist_bytes);
    if (!rc) rc = rc_dev_alloc(c, "enum_totals", (void **)&h->d_tot, tot_bytes);
    if (rc) {
        snprintf(g_enum_err, sizeof(g_enum_err), "allocation refused");
        rc_enum_free(h);
        return NULL;
    }
    h->flags = h->d_rows;
    h->bids = nb > 1 ? h->d_rows + rc_align256(n) : NULL;
    h->off = h->d_prog + rc_align256(sizeof(unsigned int) * (size_t)word_cap) / sizeof(unsigned int);
    unsigned char *p = h->d_tree;
    h->nonfinite = (int *)p;
    h->dcount = h->nonfinite + T;
    h->status = h->dcount + T;
    h->nslots = h->status + T;
    h->ncuts = h->nslots + T;
    h->bad = h->ncuts + T;
    p += ints;
    h->dvals = (unsigned int *)p;
    p += dv;
    h->ranks = (unsigned int *)p;
    p += rk;
    h->qpre = (unsigned int *)p;
    p += qq;
    h->qres = (unsigned int *)p;
    p += qq;
    h->spre = (unsigned int *)p;
    p += qq;
    h->qslot = (int *)p;
    p += qq;
    h->lo = (unsigned int *)p;
    p += cut;
    h->hi = (unsigned int *)p;
    h->hist1 = h->d_hist;
    h->histq = h->d_hist + h1 / sizeof(unsigned int);
    cudaError_t e = cudaMemcpy(h->flags, flags, n, cudaMemcpyHostToDevice);
    if (e == cudaSuccess && nb > 1) e = cudaMemcpy(h->bids, bids, n, cudaMemcpyHostToDevice);
    if (e == cudaSuccess && nq > 0) e = cudaMemcpy(h->ranks, ranks, sizeof(unsigned int) * (size_t)nq, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaEventCreate(&h->e0);
    if (e == cudaSuccess) {
        e = cudaEventCreate(&h->e1);
        if (e != cudaSuccess) cudaEventDestroy(h->e0);
    }
    if (e != cudaSuccess) {
        rc_enum_fail("the upload of the rows", e);
        rc_enum_free(h);
        return NULL;
    }
    h->events = 1;
    return h;
}

void rc_enum_destroy(void *handle)
{
    rc_enum_free((RcEnum *)handle);
}

int rc_enum_set_book(void *handle, int slot, const double *excess)
{
    RcEnum *h = (RcEnum *)handle;
    if (!h || slot < 0 || slot >= h->book_cap || !excess) return -1;
    const cudaError_t e = cudaMemcpy(h->d_excess + (size_t)slot * (size_t)h->n, excess,
                                     sizeof(double) * (size_t)h->n, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        rc_enum_fail("the upload of a book", e);
        return -4;
    }
    return 0;
}

static double rc_enum_elapsed(RcEnum *h)
{
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, h->e0, h->e1) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    return (double)ms;
}

static int rc_enum_launch_failed(const char *name, cudaError_t e)
{
    rc_enum_fail(name, e);
    return -2;
}

/* Evaluate `ntrees` programs and select their grid ranks. Downloads per tree
   the non-finite flag, the distinct count and values (q per tree) and the
   selected f32 bits (nq per tree; meaningful for status 0). */
int rc_enum_eval(void *handle, const unsigned int *prog, const unsigned int *off, int ntrees,
                 int *nonfinite, int *dcount, unsigned int *dvals, unsigned int *sel)
{
    g_enum_err[0] = 0;
    RcEnum *h = (RcEnum *)handle;
    if (!h || !prog || !off || ntrees < 1 || ntrees > h->tree_cap || !nonfinite || !dcount || !dvals
        || (h->nq > 0 && !sel)) {
        snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_eval: invalid arguments");
        return -1;
    }
    if (off[ntrees] > (unsigned int)h->word_cap) {
        snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_eval: %u program words for a capacity of %d", off[ntrees], h->word_cap);
        return -1;
    }
    RcCtx *c = h->c;
    const size_t T = (size_t)ntrees;
    const int nq = h->nq;
    cudaError_t e = cudaMemcpy(h->d_prog, prog, sizeof(unsigned int) * off[ntrees], cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(h->off, off, sizeof(unsigned int) * (T + 1), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        rc_enum_fail("the upload of the programs", e);
        return -4;
    }
    cudaEventRecord(h->e0, 0);
    e = cudaMemsetAsync(h->d_tree, 0, sizeof(int) * (size_t)h->tree_cap * 6, 0);
    if (e == cudaSuccess) e = cudaMemsetAsync(h->hist1, 0, sizeof(unsigned int) * T * RC_ENUM_TOP_BINS, 0);
    if (e != cudaSuccess) {
        rc_enum_fail("the clearing of the tree state", e);
        return -4;
    }
    cudaError_t le = cudaSuccess;
    const dim3 grid((unsigned int)ntrees, (unsigned int)rc_enum_row_blocks(h->n));
    le = rc_launch("rc_enum_eval_kernel", rc_enum_eval_kernel, grid, RS_THREADS, 0, 0,
                   c->d_cols, h->n, h->d_prog, h->off, h->d_pred, h->nonfinite, h->hist1);
    if (le != cudaSuccess) return rc_enum_launch_failed("rc_enum_eval_kernel", le);
    const unsigned int pblocks = (unsigned int)((ntrees + RC_ENUM_PROBE_WARPS - 1) / RC_ENUM_PROBE_WARPS);
    le = rc_launch("rc_enum_probe_kernel", rc_enum_probe_kernel, pblocks, 32 * RC_ENUM_PROBE_WARPS, 0, 0,
                   h->d_pred, h->n, ntrees, h->q, h->nonfinite, h->dcount, h->dvals, h->status);
    if (le != cudaSuccess) return rc_enum_launch_failed("rc_enum_probe_kernel", le);
    if (nq > 0) {
        for (int pass = 0; pass < RC_ENUM_PASSES; ++pass) {
            if (pass > 0) {
                e = cudaMemsetAsync(h->histq, 0, sizeof(unsigned int) * T * (size_t)nq * RC_ENUM_LOW_BINS, 0);
                if (e != cudaSuccess) {
                    rc_enum_fail("the clearing of the histograms", e);
                    return -4;
                }
                le = rc_launch("rc_enum_hist_kernel", rc_enum_hist_kernel, grid, RS_THREADS, 0, 0,
                               pass, h->d_pred, h->n, nq, h->spre, h->nslots, h->status, h->histq);
                if (le != cudaSuccess) return rc_enum_launch_failed("rc_enum_hist_kernel", le);
            }
            le = rc_launch("rc_enum_pick_kernel", rc_enum_pick_kernel, ntrees, RS_THREADS, 0, 0,
                           pass, nq, h->ranks, h->hist1, h->histq, h->status, h->qpre, h->qres, h->qslot,
                           h->spre, h->nslots, h->bad);
            if (le != cudaSuccess) return rc_enum_launch_failed("rc_enum_pick_kernel", le);
        }
    }
    cudaEventRecord(h->e1, 0);
    int *bad = (int *)malloc(sizeof(int) * T);
    if (!bad) {
        snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_eval: host allocation failed");
        return -1;
    }
    e = cudaMemcpy(nonfinite, h->nonfinite, sizeof(int) * T, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(dcount, h->dcount, sizeof(int) * T, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(bad, h->bad, sizeof(int) * T, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(dvals, h->dvals, sizeof(unsigned int) * T * (size_t)h->q, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess && nq > 0) e = cudaMemcpy(sel, h->qpre, sizeof(unsigned int) * T * (size_t)nq, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        free(bad);
        rc_enum_fail("the evaluation and grid selection", e);
        return -4;
    }
    int inconsistent = 0;
    for (size_t i = 0; i < T; ++i) inconsistent |= bad[i];
    free(bad);
    if (inconsistent) {
        snprintf(g_enum_err, sizeof(g_enum_err), "the grid selection found a rank outside its histogram");
        return -5;
    }
    if (nq > 0) {
        for (size_t i = 0; i < T * (size_t)nq; ++i) {
            const unsigned int k = sel[i];
            sel[i] = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
        }
    }
    h->eval_ms += rc_enum_elapsed(h);
    return 0;
}

/* Score `ntrees` trees of the last evaluation on books 0..nbooks of the
   resident slots. `lo`, `hi` hold q entries per tree, `ncuts` the cuts of
   each tree (0 skips it). `out` receives ntrees x nbooks x q x nb totals. */
int rc_enum_score(void *handle, int ntrees, const unsigned int *lo, const unsigned int *hi,
                  const int *ncuts, int nbooks, RcEnumTotals *out)
{
    g_enum_err[0] = 0;
    RcEnum *h = (RcEnum *)handle;
    if (!h || ntrees < 1 || ntrees > h->tree_cap || !lo || !hi || !ncuts || nbooks < 1 || nbooks > h->book_cap || !out) {
        snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_score: invalid arguments");
        return -1;
    }
    for (int i = 0; i < ntrees; ++i) {
        if (ncuts[i] < 0 || ncuts[i] > h->q) {
            snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_score: tree %d has %d cuts for q = %d", i, ncuts[i], h->q);
            return -1;
        }
    }
    const size_t T = (size_t)ntrees;
    const size_t qn = T * (size_t)h->q;
    cudaError_t e = cudaMemcpy(h->lo, lo, sizeof(unsigned int) * qn, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(h->hi, hi, sizeof(unsigned int) * qn, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(h->ncuts, ncuts, sizeof(int) * T, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        rc_enum_fail("the upload of the cuts", e);
        return -4;
    }
    const size_t count = T * (size_t)nbooks * (size_t)h->q * (size_t)h->nb;
    cudaEventRecord(h->e0, 0);
    e = cudaMemsetAsync(h->d_tot, 0, sizeof(RcEnumTotals) * count, 0);
    if (e != cudaSuccess) {
        rc_enum_fail("the clearing of the totals", e);
        return -4;
    }
    const unsigned long long blocks = (unsigned long long)ntrees * (unsigned long long)nbooks * (unsigned long long)h->groups;
    if (blocks > 0x7fffffffULL) {
        snprintf(g_enum_err, sizeof(g_enum_err), "rc_enum_score: %llu blocks exceed the grid", blocks);
        return -1;
    }
    const unsigned int gx = (unsigned int)blocks;
    cudaError_t le = cudaSuccess;
    if (h->nb == 1) {
        le = rc_launch("rc_enum_score_kernel", rc_enum_score_kernel<1>, gx, RC_MAX_THREADS, 0, 0,
                       h->d_pred, h->n, h->d_excess, h->flags, h->bids, h->nb, h->q, nbooks, h->groups,
                       h->lo, h->hi, h->ncuts, h->d_tot);
        if (le != cudaSuccess) return rc_enum_launch_failed("rc_enum_score_kernel", le);
    } else if (h->nb <= RC_PNL_MAX_BLOCKS) {
        le = rc_launch("rc_enum_score_kernel", rc_enum_score_kernel<RC_PNL_MAX_BLOCKS>, gx, RC_MAX_THREADS, 0, 0,
                       h->d_pred, h->n, h->d_excess, h->flags, h->bids, h->nb, h->q, nbooks, h->groups,
                       h->lo, h->hi, h->ncuts, h->d_tot);
        if (le != cudaSuccess) return rc_enum_launch_failed("rc_enum_score_kernel", le);
    } else {
        le = rc_launch("rc_enum_score_kernel", rc_enum_score_kernel<RC_PNL_WIDE_BLOCKS>, gx, RC_MAX_THREADS, 0, 0,
                       h->d_pred, h->n, h->d_excess, h->flags, h->bids, h->nb, h->q, nbooks, h->groups,
                       h->lo, h->hi, h->ncuts, h->d_tot);
        if (le != cudaSuccess) return rc_enum_launch_failed("rc_enum_score_kernel", le);
    }
    cudaEventRecord(h->e1, 0);
    e = cudaMemcpy(out, h->d_tot, sizeof(RcEnumTotals) * count, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        rc_enum_fail("the scoring of the cuts", e);
        return -4;
    }
    h->score_ms += rc_enum_elapsed(h);
    return 0;
}

/* Test hook: the f32 bits of tree `tree` of the last evaluation. */
int rc_enum_pred(void *handle, int tree, unsigned int *out)
{
    RcEnum *h = (RcEnum *)handle;
    if (!h || tree < 0 || tree >= h->tree_cap || !out) return -1;
    const cudaError_t e = cudaMemcpy(out, h->d_pred + (size_t)tree * (size_t)h->n,
                                     sizeof(unsigned int) * (size_t)h->n, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        rc_enum_fail("the download of the predictions", e);
        return -4;
    }
    return 0;
}

/* Device milliseconds of the evaluation and selection kernels and of the
   scoring kernels so far. */
int rc_enum_kernel_ms(void *handle, double *eval_ms, double *score_ms)
{
    RcEnum *h = (RcEnum *)handle;
    if (!h || !eval_ms || !score_ms) return -1;
    *eval_ms = h->eval_ms;
    *score_ms = h->score_ms;
    return 0;
}

} /* extern "C" */

#endif
