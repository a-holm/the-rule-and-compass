/* #288: exact segmented radix selection over u64 keys.
 *
 * The #305 threshold selection (`pnl.cu`) refines one rank of one f32 vector
 * with 256-bin passes driven from the host. This is the same exact idea for
 * many segments at once: every segment of `n` u64 keys refines up to
 * `RS_MAX_QUERIES` ranks in six 11-bit passes, high digit first, and the bucket
 * choice runs on the device, so a whole selection is a fixed sequence of
 * launches on one stream with no host round trip.
 *
 * A pass counts, per segment and per rank, the digit of every key that agrees
 * with the prefix fixed so far. Counts are integer atomics and the pick is an
 * integer prefix scan, so the selected key is a function of the key multiset
 * only: block count, blocks per segment, launch order and batch composition
 * cannot change it.
 *
 * Callers own the key layout (`keys[seg * n + row]`) and the first pass: the
 * key producer counts the top digit of every finite key into query 0 of the
 * pass-0 histogram, which every query of that segment shares. */
#ifndef RCSR_RADIX_SELECT_H
#define RCSR_RADIX_SELECT_H

#include <cuda_runtime.h>
#include <stddef.h>

#define RS_THREADS 256
#define RS_DIGIT_BITS 11
#define RS_BINS (1 << RS_DIGIT_BITS)
#define RS_PASSES 6
#define RS_MAX_QUERIES 2
/* Rows one block covers on average; enough to amortise clearing and flushing
   the shared bins. */
#define RS_ROWS_PER_BLOCK 16384
#define RS_MAX_BLOCKS_PER_SEG 1024
#define RS_NO_DIGIT 0xFFFFFFFFu

/* Pass `p` refines bits `[rs_shift(p), rs_top(p))`; pass 5 is the 9 low bits. */
__host__ __device__ __forceinline__ int rs_top(int pass)
{
    return 64 - RS_DIGIT_BITS * pass;
}

__host__ __device__ __forceinline__ int rs_shift(int pass)
{
    const int s = rs_top(pass) - RS_DIGIT_BITS;
    return s > 0 ? s : 0;
}

__host__ __device__ __forceinline__ int rs_width(int pass)
{
    return rs_top(pass) - rs_shift(pass);
}

/* The bits already fixed before pass `p`. */
__host__ __device__ __forceinline__ unsigned long long rs_fixed_mask(int pass)
{
    const int top = rs_top(pass);
    return top >= 64 ? 0ULL : (~0ULL << top);
}

static inline int rs_blocks_per_segment(int n)
{
    long long b = ((long long)n + RS_ROWS_PER_BLOCK - 1) / RS_ROWS_PER_BLOCK;
    if (b < 1) b = 1;
    if (b > RS_MAX_BLOCKS_PER_SEG) b = RS_MAX_BLOCKS_PER_SEG;
    return (int)b;
}

/* Count `digit` once per lane. Lanes with the same digit are merged into one
 * shared atomic, which keeps a pass whose keys share one or two digits (the
 * exponent bins of pass 0) from serialising. Every lane of the warp must call
 * this; `RS_NO_DIGIT` counts nothing. */
__device__ __forceinline__ void rs_count(unsigned int *bins, unsigned int digit)
{
    const unsigned int peers = __match_any_sync(0xffffffffu, digit);
    const int lane = threadIdx.x & 31;
    if (digit != RS_NO_DIGIT && lane == __ffs(peers) - 1) {
        atomicAdd(&bins[digit], (unsigned int)__popc(peers));
    }
}

/* Add a block's shared bins (`q` queries) to the segment's histogram. */
__device__ __forceinline__ void rs_flush(const unsigned int *bins, int q, unsigned int *seg_hist)
{
    for (int b = threadIdx.x; b < q * RS_BINS; b += blockDim.x) {
        const unsigned int v = bins[b];
        if (v != 0u) atomicAdd(&seg_hist[b], v);
    }
}

/* Passes 1..5. Grid `(segments, blocks per segment)`, `RS_THREADS` threads.
 * A segment whose status is not 0 is skipped. */
__global__ void rs_hist_kernel(const unsigned long long *__restrict__ keys,
                               int n, int nsegs, int q, int pass,
                               const int *__restrict__ status,
                               const unsigned long long *__restrict__ prefix,
                               unsigned int *__restrict__ hist)
{
    __shared__ unsigned int bins[RS_MAX_QUERIES * RS_BINS];
    const int seg = blockIdx.x;
    if (seg >= nsegs || status[seg] != 0) return;
    const int shift = rs_shift(pass);
    const unsigned int dmask = (1u << rs_width(pass)) - 1u;
    const unsigned long long fixed = rs_fixed_mask(pass);
    unsigned long long pfx[RS_MAX_QUERIES];
    for (int k = 0; k < RS_MAX_QUERIES; ++k) {
        pfx[k] = k < q ? prefix[(size_t)seg * RS_MAX_QUERIES + k] : 0ULL;
    }
    for (int b = threadIdx.x; b < q * RS_BINS; b += blockDim.x) bins[b] = 0u;
    __syncthreads();
    const unsigned long long *ks = keys + (size_t)seg * (size_t)n;
    const long long stride = (long long)gridDim.y * blockDim.x;
    for (long long base = (long long)blockIdx.y * blockDim.x; base < n; base += stride) {
        const long long row = base + threadIdx.x;
        const int live = row < n;
        const unsigned long long key = live ? ks[row] : 0ULL;
        for (int k = 0; k < q; ++k) {
            unsigned int d = RS_NO_DIGIT;
            if (live && (key & fixed) == pfx[k]) d = (unsigned int)(key >> shift) & dmask;
            rs_count(bins + k * RS_BINS, d);
        }
    }
    __syncthreads();
    rs_flush(bins, q, hist + (size_t)seg * RS_MAX_QUERIES * RS_BINS);
}

/* Fix one digit of every query. One block per segment. Pass 0 reads the shared
 * histogram of query 0 and the initial ranks `rank0`/`rank1`; later passes read
 * each query's own histogram and residual rank. The bins read are cleared for
 * the next pass. A rank that is not inside the counted keys cannot happen for a
 * consistent histogram; it is reported as status 3 so the caller discards the
 * call instead of trusting it. */
__global__ void rs_pick_kernel(unsigned int *__restrict__ hist, int nsegs, int q, int pass,
                               unsigned int rank0, unsigned int rank1,
                               unsigned long long *__restrict__ prefix,
                               unsigned int *__restrict__ rank,
                               int *__restrict__ status)
{
    __shared__ unsigned int part[RS_THREADS];
    __shared__ int found_digit;
    __shared__ unsigned int found_below;
    const int seg = blockIdx.x;
    if (seg >= nsegs || status[seg] != 0) return;
    const int tid = threadIdx.x;
    const int shift = rs_shift(pass);
    const int per = (1 << rs_width(pass)) / RS_THREADS;
    unsigned int *h0 = hist + (size_t)seg * RS_MAX_QUERIES * RS_BINS;
    const size_t st = (size_t)seg * RS_MAX_QUERIES;
    for (int k = 0; k < q; ++k) {
        const unsigned int *h = h0 + (pass == 0 ? 0 : k) * RS_BINS;
        const unsigned int r = pass == 0 ? (k == 0 ? rank0 : rank1) : rank[st + k];
        unsigned int mine = 0u;
        for (int i = 0; i < per; ++i) mine += h[tid * per + i];
        part[tid] = mine;
        if (tid == 0) found_digit = -1;
        __syncthreads();
        for (int d = 1; d < RS_THREADS; d <<= 1) {
            const unsigned int v = tid >= d ? part[tid - d] : 0u;
            __syncthreads();
            part[tid] += v;
            __syncthreads();
        }
        const unsigned int incl = part[tid];
        const unsigned int excl = incl - mine;
        if (r >= excl && r < incl) {
            unsigned int below = excl;
            for (int i = 0; i < per; ++i) {
                const unsigned int cnt = h[tid * per + i];
                if (r < below + cnt) {
                    found_digit = tid * per + i;
                    found_below = below;
                    break;
                }
                below += cnt;
            }
        }
        __syncthreads();
        if (tid == 0) {
            if (found_digit < 0) {
                status[seg] = 3;
            } else {
                const unsigned long long p = pass == 0 ? 0ULL : prefix[st + k];
                prefix[st + k] = p | ((unsigned long long)found_digit << shift);
                rank[st + k] = r - found_below;
            }
        }
        __syncthreads();
    }
    const int used = pass == 0 ? 1 : q;
    for (int b = tid; b < used * RS_BINS; b += RS_THREADS) h0[b] = 0u;
}

#endif
