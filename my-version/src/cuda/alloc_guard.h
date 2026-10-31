/* Device allocation that never leaves a dangling pointer or a pending error.
 *
 * A refused cudaMalloc sets the runtime's last-error slot. The next kernel
 * launch check would read it as that kernel's own failure, and a caller that
 * ignores the return value would keep whatever the pointer held. Every
 * allocation in the enrichment sources goes through these helpers. */
#ifndef RCSR_ALLOC_GUARD_H
#define RCSR_ALLOC_GUARD_H

#include <cuda_runtime.h>
#include <stddef.h>

static inline cudaError_t rc_checked_malloc(void **p, size_t bytes)
{
    void *q = NULL;
    cudaError_t e = cudaMalloc(&q, bytes);
    if (e != cudaSuccess) {
        cudaGetLastError();
        *p = NULL;
        return e;
    }
    *p = q;
    return cudaSuccess;
}

static inline cudaError_t rc_checked_malloc_managed(void **p, size_t bytes)
{
    void *q = NULL;
    cudaError_t e = cudaMallocManaged(&q, bytes, cudaMemAttachGlobal);
    if (e != cudaSuccess) {
        cudaGetLastError();
        *p = NULL;
        return e;
    }
    *p = q;
    return cudaSuccess;
}

#endif
