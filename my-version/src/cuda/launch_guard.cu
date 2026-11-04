/* Process state of the launch guard and its test entry; see launch_guard.h. */
#include "launch_guard.h"

#include <mutex>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace {
struct Refusal {
    char kernel[96];
    int arg;
};
thread_local Refusal g_last = {"", -1};
}

extern "C" int rc_launch_check_enabled(void)
{
    static const int enabled = [] {
        const char *e = getenv("RCSR_DEVICE_PTR_CHECK");
        return (e && e[0] == '0' && e[1] == '\0') ? 0 : 1;
    }();
    return enabled;
}

extern "C" int rc_launch_forced(const char *kernel)
{
    static const char *const prefix = getenv("RCSR_FORCE_LAUNCH_REFUSAL");
    return prefix && prefix[0] && strncmp(kernel, prefix, strlen(prefix)) == 0;
}

extern "C" void rc_launch_refused(const char *kernel, int arg)
{
    snprintf(g_last.kernel, sizeof(g_last.kernel), "%s", kernel);
    g_last.arg = arg;
    static std::mutex m;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lock(m);
    if (seen.insert(kernel).second) {
        fprintf(stderr, "rcsr: refused launch of %s: argument %d is not device or managed memory\n",
                kernel, arg);
    }
}

extern "C" int rc_launch_last_refusal(char *buf, int len)
{
    if (buf && len > 0) snprintf(buf, (size_t)len, "%s", g_last.kernel);
    return g_last.arg;
}

namespace {
struct RcTestArg {
    int n;
    int *dev;
    const int *other;
};
}

template <>
struct RcArgCheck<RcTestArg> {
    static bool ok(const RcTestArg &a) { return rc_dev_ok(a.dev) && rc_dev_ok(a.other); }
};

/* Never dereferences src: a launch that wrongly runs cannot touch host memory. */
static __global__ void k_guard_touch(int *counter, const int *src)
{
    (void)src;
    if (threadIdx.x == 0 && blockIdx.x == 0) counter[0] = 1;
}

static __global__ void k_guard_touch_struct(RcTestArg a)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) a.dev[0] = a.n;
}

/* Launches a test kernel with one argument of the given kind.
 * kind: 0 device, 1 malloc, 2 stack, 3 pinned, 4 NULL, 5 managed,
 *       6 struct with a host member, 7 struct with device members.
 * out[0] = rc_launch result, out[1] = first cudaGetLastError, out[2] = second,
 * out[3] = counter read back from the device (1 when the kernel ran),
 * out[4] = refused argument index (-1 none). Returns 0, or -1 on setup failure. */
extern "C" int rc_test_launch_guard(int kind, int *out, char *kernel, int kernel_len)
{
    int *counter = NULL;
    if (cudaMalloc((void **)&counter, sizeof(int)) != cudaSuccess) return -1;
    cudaMemset(counter, 0, sizeof(int));
    int stack_word = 0;
    void *arg = NULL;
    void *host = NULL;
    switch (kind) {
        case 0:
        case 7: cudaMalloc(&arg, 64); break;
        case 1:
        case 6:
            host = malloc(64);
            arg = host;
            break;
        case 2: arg = &stack_word; break;
        case 3: cudaHostAlloc(&arg, 64, cudaHostAllocDefault); break;
        case 5: cudaMallocManaged(&arg, 64); break;
        default: break;
    }
    if (kind != 4 && !arg) {
        cudaFree(counter);
        return -1;
    }
    cudaGetLastError();
    cudaError_t r;
    if (kind == 6 || kind == 7) {
        RcTestArg a = {1, counter, (const int *)arg};
        r = rc_launch("k_guard_touch_struct", k_guard_touch_struct, 1, 32, 0, 0, a);
    } else {
        r = rc_launch("k_guard_touch", k_guard_touch, 1, 32, 0, 0, counter, (const int *)arg);
    }
    out[0] = (int)r;
    out[1] = (int)cudaGetLastError();
    out[2] = (int)cudaGetLastError();
    cudaDeviceSynchronize();
    int back = -1;
    cudaMemcpy(&back, counter, sizeof(int), cudaMemcpyDeviceToHost);
    out[3] = back;
    out[4] = rc_launch_last_refusal(kernel, kernel_len);
    cudaGetLastError();
    cudaFree(counter);
    if (kind == 0 || kind == 7 || kind == 5) cudaFree(arg);
    if (kind == 3) cudaFreeHost(arg);
    free(host);
    return 0;
}
