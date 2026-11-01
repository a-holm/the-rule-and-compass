/* CUB status handling shared by the enrichment sources.
 *
 * Every CUB entry point returns a cudaError_t. A failed call leaves the
 * runtime's last-error slot set, so the next launch check would blame an
 * unrelated kernel. rc_cub_status clears the slot and hands the code back.
 *
 * rc_cub_take implements the test hook: a one-shot counter (thread_local, so a
 * forced failure never reaches another test). A counter of k refuses the k-th
 * guarded call before it reaches CUB and reports a failure; 1 is the next call. */
#ifndef RCSR_CUB_GUARD_H
#define RCSR_CUB_GUARD_H

#include <cuda_runtime.h>

static inline cudaError_t rc_cub_status(cudaError_t e)
{
    if (e != cudaSuccess) cudaGetLastError();
    return e;
}

static inline int rc_cub_take(int *flag)
{
    if (*flag <= 0) return 0;
    return --*flag == 0;
}

#define RC_CUB(flag, call) \
    rc_cub_status(rc_cub_take(&(flag)) ? cudaErrorInvalidValue : (call))

#endif
