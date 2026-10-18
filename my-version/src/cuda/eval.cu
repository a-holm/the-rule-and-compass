/*
 * Batched fitness evaluation on the GPU.
 *
 * The workload is: given one dataset (N rows, V variables) and a batch of P
 * candidate expressions, compute the sum of squared errors of each expression
 * over all rows. That is P x N independent evaluations -- embarrassingly
 * parallel -- and it is the hot loop of the whole search.
 *
 * Design (mirrors the plan in docs/gpu-backend.md):
 *   - one thread block per expression, threads spread across rows;
 *   - therefore every thread in a block executes the *same* instruction stream,
 *     so there is no warp divergence caused by program structure;
 *   - operators are branchless (protected division/log/sqrt, no `if`);
 *   - the fitness reduction is fused into the kernel;
 *   - the dataset stays resident in device memory across evaluations.
 *
 * Arithmetic is float (FP32) on purpose: on both the consumer Ampere parts and
 * the GB10 in the DGX Spark, FP64 runs at 1/64 of FP32, so an FP64 kernel would
 * throw away the entire advantage. FP64 is reserved for re-checking the final
 * surviving equations on the host.
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Opcodes. Keep in sync with src/gpu.rs. */
enum {
    OP_CONST = 0,  /* next word: float bits            */
    OP_VAR = 1,    /* next word: variable index        */
    OP_ADD = 2,
    OP_SUB = 3,
    OP_MUL = 4,
    OP_DIV = 5,
    OP_SIN = 6,
    OP_COS = 7,
    OP_TAN = 8,
    OP_EXP = 9,
    OP_LOG = 10,
    OP_SQRT = 11,
    OP_ABS = 12,
    OP_TANH = 13,
    OP_SIG = 14,   /* logistic                        */
    OP_SQUARE = 15,
    OP_CUBE = 16,
    OP_NEG = 17,
    OP_GAUSS = 18,
    OP_STEP = 19,
    OP_POW = 20,
    OP_SINH = 21,
    OP_COSH = 22,
    OP_HALT = 255
};

#define RC_MAX_STACK 64
#define RC_MAX_THREADS 256
#define RC_MIN_THREADS 32

struct RcCtx {
    float *d_cols;
    float *d_y;
    unsigned int *d_prog;
    unsigned int *d_off;
    float *d_out;
    int n;
    int v;
    int nprog;
    int out_cap;
    cudaEvent_t t0, t1;
};

__device__ __forceinline__ float logisticf(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void eval_kernel(const float *__restrict__ cols,
                            const float *__restrict__ y,
                            int n,
                            int v,
                            const unsigned int *__restrict__ prog,
                            const unsigned int *__restrict__ off,
                            float *__restrict__ out)
{
    const int p = blockIdx.x;
    float acc = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s[RC_MAX_STACK];
        int sp = 0;
        unsigned int q = off[p];
        for (;;) {
            unsigned int w = prog[q++];
            if (w == OP_CONST) {
                s[sp++] = __uint_as_float(prog[q++]);
            } else if (w == OP_VAR) {
                unsigned int k = prog[q++];
                s[sp++] = cols[(size_t)k * (size_t)n + (size_t)i];
            } else if (w == OP_HALT) {
                break;
            } else if (w == OP_ADD) {
                float b = s[--sp];
                s[sp - 1] += b;
            } else if (w == OP_SUB) {
                float b = s[--sp];
                s[sp - 1] -= b;
            } else if (w == OP_MUL) {
                float b = s[--sp];
                s[sp - 1] *= b;
            } else if (w == OP_DIV) {
                float b = s[--sp];
                /* Match the host evaluator: a near-zero denominator is a domain
                   error (NaN), which the host then scores as a hard failure. */
                s[sp - 1] = (fabsf(b) < 1e-12f) ? CUDART_NAN_F : s[sp - 1] / b;
            } else if (w == OP_POW) {
                float b = s[--sp];
                float a = s[sp - 1];
                s[sp - 1] = (a < 0.0f && fabsf(b - roundf(b)) > 1e-9f)
                                ? CUDART_NAN_F
                                : powf(a, b);
            } else {
                /* unary operators */
                float a = s[sp - 1];
                float r = a;
                switch (w) {
                    case OP_SIN: r = sinf(a); break;
                    case OP_COS: r = cosf(a); break;
                    case OP_TAN: {
                        float t = tanf(a);
                        r = isfinite(t) ? t : CUDART_NAN_F;
                    } break;
                    case OP_EXP: r = expf(a); break;
                    case OP_LOG: r = (a > 0.0f) ? logf(a) : CUDART_NAN_F; break;
                    case OP_SQRT: r = (a >= 0.0f) ? sqrtf(a) : CUDART_NAN_F; break;
                    case OP_ABS: r = fabsf(a); break;
                    case OP_TANH: r = tanhf(a); break;
                    case OP_SIG: r = logisticf(a); break;
                    case OP_SQUARE: r = a * a; break;
                    case OP_CUBE: r = a * a * a; break;
                    case OP_NEG: r = -a; break;
                    case OP_GAUSS: r = expf(-a * a); break;
                    case OP_STEP: r = a > 0.0f ? 1.0f : 0.0f; break;
                    case OP_SINH: r = sinhf(a); break;
                    case OP_COSH: r = coshf(a); break;
                    default: r = 0.0f; break;
                }
                s[sp - 1] = r;
            }
        }
        float pred = s[0];
        /* Shared magnitude limit: anything non-finite or astronomically large is
           a hard failure. Applying the *same* rule on the host keeps the f32
           kernel and the f64 reference on the same fitness landscape. */
        if (!isfinite(pred) || fabsf(pred) > 1e30f) {
            acc += 1e30f;
            continue;
        }
        float r = pred - y[i];
        acc += r * r;
    }

    __shared__ float sh[RC_MAX_THREADS];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[p] = sh[0];
}

extern "C" {

void *rc_create(const float *cols, int n, int v, const float *y)
{
    RcCtx *c = (RcCtx *)calloc(1, sizeof(RcCtx));
    if (!c) return 0;
    c->n = n;
    c->v = v;
    c->nprog = 0;
    c->out_cap = 0;
    cudaMalloc((void **)&c->d_cols, sizeof(float) * (size_t)n * (size_t)v);
    cudaMalloc((void **)&c->d_y, sizeof(float) * (size_t)n);
    cudaMemcpy(c->d_cols, cols, sizeof(float) * (size_t)n * (size_t)v, cudaMemcpyHostToDevice);
    cudaMemcpy(c->d_y, y, sizeof(float) * (size_t)n, cudaMemcpyHostToDevice);
    cudaEventCreate(&c->t0);
    cudaEventCreate(&c->t1);
    return (void *)c;
}

void rc_destroy(void *ctx)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return;
    cudaFree(c->d_cols);
    cudaFree(c->d_y);
    if (c->d_prog) cudaFree(c->d_prog);
    if (c->d_off) cudaFree(c->d_off);
    if (c->d_out) cudaFree(c->d_out);
    cudaEventDestroy(c->t0);
    cudaEventDestroy(c->t1);
    free(c);
}

int rc_set_programs(void *ctx, const unsigned int *prog, const unsigned int *off, int nprog)
{
    RcCtx *c = (RcCtx *)ctx;
    if (!c) return -1;
    if (c->d_prog) { cudaFree(c->d_prog); c->d_prog = 0; }
    if (c->d_off) { cudaFree(c->d_off); c->d_off = 0; }
    c->nprog = nprog;
    size_t prog_bytes = sizeof(unsigned int) * (size_t)off[nprog];
    cudaMalloc((void **)&c->d_prog, prog_bytes);
    cudaMalloc((void **)&c->d_off, sizeof(unsigned int) * (size_t)(nprog + 1));
    cudaMemcpy(c->d_prog, prog, prog_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c->d_off, off, sizeof(unsigned int) * (size_t)(nprog + 1), cudaMemcpyHostToDevice);
    if (c->out_cap < nprog) {
        if (c->d_out) cudaFree(c->d_out);
        cudaMalloc((void **)&c->d_out, sizeof(float) * (size_t)nprog);
        c->out_cap = nprog;
    }
    return 0;
}

/* Returns elapsed milliseconds; writes nprog errors into `out`. */
float rc_eval(void *ctx, float *out)
{
    RcCtx *c = (RcCtx *)ctx;
    /* One block per expression; threads spread over rows. Size the block to the
       row count (rounded up to a power of two) so short datasets do not launch
       mostly-idle warps. */
    int threads = RC_MIN_THREADS;
    while (threads < RC_MAX_THREADS && threads < c->n) threads <<= 1;
    cudaEventRecord(c->t0, 0);
    eval_kernel<<<c->nprog, threads>>>(c->d_cols, c->d_y, c->n, c->v,
                                       c->d_prog, c->d_off, c->d_out);
    cudaEventRecord(c->t1, 0);
    cudaMemcpy(out, c->d_out, sizeof(float) * (size_t)c->nprog, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(c->t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c->t0, c->t1);
    return ms;
}

const char *rc_device_name(void)
{
    static char name[256];
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return "unknown";
    snprintf(name, sizeof(name), "%s (sm_%d%d, %d SMs, %d MB)",
             p.name, p.major, p.minor, p.multiProcessorCount,
             (int)(p.totalGlobalMem / (1024 * 1024)));
    return name;
}

} /* extern "C" */
