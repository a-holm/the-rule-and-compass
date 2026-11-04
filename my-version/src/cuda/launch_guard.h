/* Checked kernel launches: every kernel launch in src/cuda goes through
 * rc_launch. Each pointer argument must be NULL, device or managed memory;
 * anything else refuses the launch before the kernel runs, so a host pointer
 * can neither fault nor migrate a page under HMM.
 *
 * A refused launch runs a zero-block copy of the same kernel. That executes
 * nothing and leaves cudaErrorInvalidValue in the last-error slot,
 * where every caller already reads it. rc_launch_last_refusal names the
 * kernel and argument.
 *
 * RCSR_DEVICE_PTR_CHECK=0 turns the check off (read once per process). The
 * type test uses cudaPointerAttributes::type: under HMM devicePointer is
 * non-NULL for pageable host memory.
 *
 * A by-value struct that carries pointers needs an RcArgCheck specialisation.
 * This header holds the only kernel launch syntax in src/cuda. */
#ifndef RCSR_LAUNCH_GUARD_H
#define RCSR_LAUNCH_GUARD_H

#include <cuda_runtime.h>
#include <type_traits>

#ifdef __cplusplus
extern "C" {
#endif
int rc_launch_check_enabled(void);
void rc_launch_refused(const char *kernel, int arg);
/* Kernel name of this thread's last refusal into buf; returns its argument
 * index, or -1 when there is none. */
int rc_launch_last_refusal(char *buf, int len);
#ifdef __cplusplus
}
#endif

static inline bool rc_dev_ok(const void *p)
{
    if (!p) return true;
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}

template <typename T, typename = void>
struct RcArgCheck {
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
                  "kernel argument type needs an RcArgCheck specialisation");
    static bool ok(const T &) { return true; }
};

template <typename T>
struct RcArgCheck<T *> {
    static bool ok(T *p) { return rc_dev_ok((const void *)p); }
};

template <typename P, typename A>
static inline P rc_launch_conv(A a)
{
    return a;
}

template <typename... P, typename... A>
static cudaError_t rc_launch(const char *name, void (*k)(P...), dim3 grid, dim3 block,
                             size_t smem, cudaStream_t s, A... a)
{
    if (rc_launch_check_enabled()) {
        const bool ok[] = {true, RcArgCheck<std::decay_t<P>>::ok(rc_launch_conv<P>(a))...};
        for (size_t i = 1; i < sizeof(ok) / sizeof(ok[0]); ++i) {
            if (!ok[i]) {
                rc_launch_refused(name, (int)i - 1);
                k<<<dim3(0), block, 0, s>>>(rc_launch_conv<P>(a)...);
                const cudaError_t e = cudaPeekAtLastError();
                return e != cudaSuccess ? e : cudaErrorInvalidValue;
            }
        }
    }
    k<<<grid, block, smem, s>>>(rc_launch_conv<P>(a)...);
    return cudaPeekAtLastError();
}

#endif
