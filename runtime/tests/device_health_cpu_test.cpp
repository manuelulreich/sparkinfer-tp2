// dual-gpu WP-5: the failure policy's bookkeeping (sparkinfer/device_health.h). Decision (a),
// manifest CHG-0034: any fatal event on EITHER card -- or a tp=2 collective/rendezvous failure --
// downgrades the whole server through the one process-global flag; the first event's reason and
// card are kept for /health, /v1/info and the per-request error.
//
// CPU-only: needs the CUDA headers (cudaError_t) and links cudart for the symbols, but never calls
// into the driver -- the only paths exercised either return before any CUDA call (recoverable
// errors) or take no CUDA call at all (note_device_fatal / note_tp_fatal). The flag is one-way, so
// the cases run in order in one process.

#include "sparkinfer/device_health.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace sparkinfer;

static int g_fail = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL: %s (line %d)\n", #x, __LINE__); ++g_fail; } } while (0)

int main() {
    // Healthy at start; no reason recorded.
    CHECK(!device_lost());
    CHECK(!device_lost_info().lost);
    CHECK(device_lost_info().reason.empty());
    CHECK(device_lost_info().device == -1);

    // Recoverable errors never trip the policy (OOM is capacity, reported as 503/429 per request).
    CHECK(!note_cuda_error(cudaSuccess));
    CHECK(!note_cuda_error(cudaErrorMemoryAllocation));
    CHECK(!note_cuda_error(cudaErrorInvalidValue));
    CHECK(!device_lost());

    // The classification the cu() wrappers rely on (unchanged by WP-5).
    CHECK(is_unrecoverable(cudaErrorIllegalAddress));
    CHECK(is_unrecoverable(cudaErrorLaunchFailure));
    CHECK(!is_unrecoverable(cudaErrorMemoryAllocation));
    CHECK(!is_unrecoverable(cudaErrorUnknown));   // why a link failure needs note_tp_fatal

    // A tp collective failure (rendezvous timeout / GpuLink op failure) downgrades the WHOLE
    // engine: flag set, reason recorded, device unknown (-1 = the link, not one card). Several
    // threads racing (both ranks noticing at once) still record exactly one first event.
    std::vector<std::thread> ts;
    std::vector<int> firsts(4, 0);
    for (int i = 0; i < 4; ++i)
        ts.emplace_back([i, &firsts] { firsts[i] = note_tp_fatal("rendezvous timeout") ? 1 : 0; });
    for (auto& t : ts) t.join();
    CHECK(firsts[0] + firsts[1] + firsts[2] + firsts[3] == 1);
    CHECK(device_lost());
    DeviceLostInfo info = device_lost_info();
    CHECK(info.lost);
    CHECK(info.device == -1);
    CHECK(info.reason == "tp=2 collective failure: rendezvous timeout");

    // Later events are consequences: they keep the flag set but never overwrite the first reason.
    CHECK(!note_device_fatal(1, "cuda: an illegal memory access was encountered"));
    info = device_lost_info();
    CHECK(info.reason == "tp=2 collective failure: rendezvous timeout");
    CHECK(info.device == -1);

    // One-way: nothing un-sets it (only a restart does).
    CHECK(!note_cuda_error(cudaErrorMemoryAllocation));
    CHECK(device_lost());

    if (g_fail) {
        printf("device_health_cpu_test: %d FAILED\n", g_fail);
        return 1;
    }
    printf("device_health_cpu_test: OK\n");
    return 0;
}
