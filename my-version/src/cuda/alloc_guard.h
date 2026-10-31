/* Device allocation that never leaves a dangling pointer or a pending error.
 *
 * A refused cudaMalloc sets the runtime's last-error slot. The next kernel
 * launch check would read it as that kernel's own failure, and a caller that
 * ignores the return value would keep whatever the pointer held. Every
 * allocation in the enrichment sources goes through these helpers.
 *
 * Test hook: when rc_fault_alloc_armed is set, the helpers ask
 * rc_fault_alloc_hit() whether to refuse this allocation before cudaMalloc is
 * called. The state lives in enrich.cu. */
#ifndef RCSR_ALLOC_GUARD_H
#define RCSR_ALLOC_GUARD_H

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
extern volatile int rc_fault_alloc_armed;
int rc_fault_alloc_hit(void);
#ifdef __cplusplus
}
#endif

static inline cudaError_t rc_checked_malloc(void **p, size_t bytes)
{
    void *q = NULL;
    if (rc_fault_alloc_armed && rc_fault_alloc_hit()) {
        *p = NULL;
        return cudaErrorMemoryAllocation;
    }
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
    if (rc_fault_alloc_armed && rc_fault_alloc_hit()) {
        *p = NULL;
        return cudaErrorMemoryAllocation;
    }
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
