/*
 * Exact type-7 threshold selection and the shared valid-row mask (#305).
 *
 * The threshold is a discrete input: a one-ulp difference flips every position
 * whose score sits on the boundary. Device selection is therefore exact, not
 * approximate:
 *
 *   1. map every finite f32 to an order-preserving u32 key (`-0` -> `+0`,
 *      NaN/+-inf -> the 0xffffffff sentinel);
 *   2. select the two ranks bracketing h = (n - 1) q with four 256-bin radix
 *      count passes per rank. Counts are integer atomics, so the result does
 *      not depend on thread count, block count or launch order;
 *   3. return the two order statistics as f64. The host interpolates in f64
 *      with the same expression as the CPU reference; that arithmetic is not
 *      re-associated here.
 *
 * The mask kernel sets bit i iff score[i] and target[i] are both finite, in the
 * same little-endian-within-u64 layout as Rust's `trading_gpu::ValidMask`.
 *
 * Both entry points take host slices, upload, run and download. They are the
 * parity/selection path for a small discrete input, not a per-candidate
 * streaming path; the CPU result remains the reported authority.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RC_PNL_THREADS 256
#define RC_PNL_MAX_THREADS 512
#define RC_NONFINITE_KEY 0xffffffffu

__device__ __forceinline__ unsigned int rc_pnl_key(float v)
{
    if (!isfinite(v)) return RC_NONFINITE_KEY;
    unsigned int bits = (v == 0.0f) ? 0u : __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

/* Host-side inverse of the key transform. The device never needs it: it only
 * writes keys, and the two order statistics come back as f64. */
static float rc_pnl_value(unsigned int key)
{
    unsigned int bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Keys plus the exact count of finite rows. Integer atomics make both
 * deterministic. */
__global__ void rc_pnl_key_kernel(const float *__restrict__ scores,
                                  long long n,
                                  unsigned int *__restrict__ keys,
                                  unsigned long long *__restrict__ finite)
{
    __shared__ unsigned int count[RC_PNL_MAX_THREADS];
    const int tid = threadIdx.x;
    if (tid < RC_PNL_MAX_THREADS) count[tid] = 0u;
    for (long long i = (long long)blockIdx.x * blockDim.x + tid; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        unsigned int key = rc_pnl_key(scores[i]);
        keys[i] = key;
        if (key != RC_NONFINITE_KEY) count[tid] += 1u;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) count[tid] += count[tid + stride];
        __syncthreads();
    }
    if (tid == 0 && count[0] != 0u) {
        atomicAdd(finite, (unsigned long long)count[0]);
    }
}

/* One radix pass: 256-bin histogram of `keys` restricted to the prefix already
 * fixed by the host. */
__global__ void rc_pnl_hist_kernel(const unsigned int *__restrict__ keys,
                                   long long n,
                                   int shift,
                                   unsigned int keep_mask,
                                   unsigned int keep_val,
                                   unsigned int *__restrict__ hist)
{
    __shared__ unsigned int bins[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) bins[b] = 0u;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        unsigned int key = keys[i];
        if (key == RC_NONFINITE_KEY) continue;
        if ((key & keep_mask) != keep_val) continue;
        atomicAdd(&bins[(key >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        if (bins[b] != 0u) atomicAdd(&hist[b], bins[b]);
    }
}

/* Select one rank. The host advances `prefix` with the chosen byte; the next
 * pass only counts keys that agree on every byte fixed so far. */
static int rc_pnl_select_rank(const unsigned int *d_keys,
                              long long key_count,
                              long long finite_count,
                              long long rank,
                              int threads,
                              unsigned int *d_hist,
                              unsigned int *h_hist,
                              unsigned int *out_key)
{
    if (rank < 0 || rank >= finite_count || threads <= 0 ||
        threads > RC_PNL_MAX_THREADS)
        return 1;
    unsigned int prefix = 0u;
    const int blocks = (int)((key_count + threads - 1) / threads);
    for (int shift = 24; shift >= 0; shift -= 8) {
        unsigned int keep_mask =
            (shift >= 24) ? 0u : (0xffffffffu << (shift + 8));
        cudaError_t err = cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
        if (err != cudaSuccess) return 2;
        rc_pnl_hist_kernel<<<blocks, threads>>>(
            d_keys, key_count, shift, keep_mask, prefix, d_hist);
        err = cudaGetLastError();
        if (err != cudaSuccess) return 3;
        err = cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return 4;
        long long before = 0;
        int bucket = -1;
        for (int b = 0; b < 256; ++b) {
            if (rank < before + (long long)h_hist[b]) {
                bucket = b;
                break;
            }
            before += (long long)h_hist[b];
        }
        if (bucket < 0) return 5;
        rank -= before;
        prefix |= ((unsigned int)bucket) << shift;
    }
    *out_key = prefix;
    return 0;
}

extern "C" int rc_threshold_select_f32_threads(const float *scores,
                                               long long n,
                                               double q,
                                               int threads,
                                               long long *out_finite,
                                               double *out_lo,
                                               double *out_hi)
{
    if (n < 0 || out_finite == NULL || out_lo == NULL || out_hi == NULL ||
        threads <= 0 || threads > RC_PNL_MAX_THREADS)
        return 1;
    *out_finite = 0;
    *out_lo = NAN;
    *out_hi = NAN;
    if (n == 0) return 0;
    if (scores == NULL) return 2;

    float *d_scores = NULL;
    unsigned int *d_keys = NULL;
    unsigned int *d_hist = NULL;
    unsigned long long *d_finite = NULL;
    unsigned long long h_finite = 0;
    unsigned int *h_hist = NULL;
    int status = 0;

    cudaError_t err = cudaMalloc((void **)&d_scores, (size_t)n * sizeof(float));
    if (err != cudaSuccess) { status = 3; goto done; }
    err = cudaMalloc((void **)&d_keys, (size_t)n * sizeof(unsigned int));
    if (err != cudaSuccess) { status = 4; goto done; }
    err = cudaMalloc((void **)&d_hist, 256 * sizeof(unsigned int));
    if (err != cudaSuccess) { status = 5; goto done; }
    err = cudaMalloc((void **)&d_finite, sizeof(unsigned long long));
    if (err != cudaSuccess) { status = 6; goto done; }
    h_hist = (unsigned int *)malloc(256 * sizeof(unsigned int));
    if (h_hist == NULL) { status = 7; goto done; }

    err = cudaMemcpy(d_scores, scores, (size_t)n * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { status = 8; goto done; }
    err = cudaMemset(d_finite, 0, sizeof(unsigned long long));
    if (err != cudaSuccess) { status = 9; goto done; }

    {
        const int blocks = (int)((n + threads - 1) / threads);
        rc_pnl_key_kernel<<<blocks, threads>>>(d_scores, n, d_keys, d_finite);
        err = cudaGetLastError();
        if (err != cudaSuccess) { status = 10; goto done; }
    }
    err = cudaMemcpy(&h_finite, d_finite, sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { status = 11; goto done; }
    *out_finite = (long long)h_finite;
    if (h_finite == 0) { status = 0; goto done; }
    if (h_finite == 1) {
        unsigned int key = 0u;
        status = rc_pnl_select_rank(d_keys, n, (long long)h_finite, 0, threads,
                                    d_hist, h_hist, &key);
        if (status != 0) goto done;
        *out_lo = (double)rc_pnl_value(key);
        *out_hi = *out_lo;
        status = 0;
        goto done;
    }

    {
        double qq = q;
        if (qq < 0.0) qq = 0.0;
        if (qq > 1.0) qq = 1.0;
        double h = (double)(h_finite - 1u) * qq;
        long long lo = 0;
        long long hi = 0;
        if (!isnan(h)) {
            lo = (long long)floor(h);
            hi = (long long)ceil(h);
        }
        if (lo < 0) lo = 0;
        if (hi < 0) hi = 0;
        if (lo >= (long long)h_finite) lo = (long long)h_finite - 1;
        if (hi >= (long long)h_finite) hi = (long long)h_finite - 1;
        unsigned int lo_key = 0u, hi_key = 0u;
        status = rc_pnl_select_rank(d_keys, n, (long long)h_finite, lo, threads,
                                    d_hist, h_hist, &lo_key);
        if (status != 0) goto done;
        if (hi == lo) {
            hi_key = lo_key;
        } else {
            status = rc_pnl_select_rank(d_keys, n, (long long)h_finite, hi,
                                        threads, d_hist, h_hist, &hi_key);
            if (status != 0) goto done;
        }
        *out_lo = (double)rc_pnl_value(lo_key);
        *out_hi = (double)rc_pnl_value(hi_key);
        status = 0;
    }

done:
    if (h_hist != NULL) free(h_hist);
    if (d_finite != NULL) cudaFree(d_finite);
    if (d_hist != NULL) cudaFree(d_hist);
    if (d_keys != NULL) cudaFree(d_keys);
    if (d_scores != NULL) cudaFree(d_scores);
    if (status == 0) cudaGetLastError(); /* clear a stale error, if any */
    return status;
}

/* Default entry point: the canonical 256-thread launch. The `_threads` variant
 * exists so the thread-invariance contract can be tested on the device. */
extern "C" int rc_threshold_select_f32(const float *scores,
                                       long long n,
                                       double q,
                                       long long *out_finite,
                                       double *out_lo,
                                       double *out_hi)
{
    return rc_threshold_select_f32_threads(scores, n, q, RC_PNL_THREADS,
                                           out_finite, out_lo, out_hi);
}

__global__ void rc_pnl_mask_kernel(const float *__restrict__ scores,
                                   const float *__restrict__ targets,
                                   long long n,
                                   unsigned long long *__restrict__ words)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        if (isfinite(scores[i]) && isfinite(targets[i])) {
            atomicOr(&words[i >> 6], 1ull << (i & 63));
        }
    }
}

extern "C" int rc_valid_mask_f32(const float *scores,
                                 const float *targets,
                                 long long n,
                                 unsigned long long *words)
{
    if (n < 0 || (n > 0 && (scores == NULL || targets == NULL || words == NULL)))
        return 1;
    if (n == 0) return 0;

    const size_t words_bytes = (size_t)((n + 63) / 64) * sizeof(unsigned long long);
    float *d_scores = NULL;
    float *d_targets = NULL;
    unsigned long long *d_words = NULL;
    int status = 0;

    cudaError_t err = cudaMalloc((void **)&d_scores, (size_t)n * sizeof(float));
    if (err != cudaSuccess) { status = 2; goto done; }
    err = cudaMalloc((void **)&d_targets, (size_t)n * sizeof(float));
    if (err != cudaSuccess) { status = 3; goto done; }
    err = cudaMalloc((void **)&d_words, words_bytes);
    if (err != cudaSuccess) { status = 4; goto done; }
    err = cudaMemcpy(d_scores, scores, (size_t)n * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { status = 5; goto done; }
    err = cudaMemcpy(d_targets, targets, (size_t)n * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { status = 6; goto done; }
    err = cudaMemset(d_words, 0, words_bytes);
    if (err != cudaSuccess) { status = 7; goto done; }
    {
        const int blocks = (int)((n + RC_PNL_THREADS - 1) / RC_PNL_THREADS);
        rc_pnl_mask_kernel<<<blocks, RC_PNL_THREADS>>>(d_scores, d_targets, n,
                                                       d_words);
        err = cudaGetLastError();
        if (err != cudaSuccess) { status = 8; goto done; }
    }
    err = cudaMemcpy(words, d_words, words_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { status = 9; goto done; }

done:
    if (d_words != NULL) cudaFree(d_words);
    if (d_targets != NULL) cudaFree(d_targets);
    if (d_scores != NULL) cudaFree(d_scores);
    if (status == 0) cudaGetLastError();
    return status;
}
