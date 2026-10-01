#pragma once
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <cuda_runtime.h>

namespace sparkinfer {

// Has the CUDA context been lost?
//
// Some CUDA errors are STICKY: once raised, the context is permanently unusable and every
// subsequent call in the process returns the same error. cudaErrorIllegalAddress is the one that
// matters here -- a single out-of-bounds access in any kernel kills the context for good.
//
// The runtime's cu() helpers used to print and continue, which is fine for a recoverable error
// (cudaErrorMemoryAllocation, say) and catastrophic for a sticky one: the process keeps issuing
// work against a dead context, every call fails, and the code proceeds on whatever uninitialised
// or stale host memory the failed copies left behind. Observed under a 16-concurrent burst on a
// nearly-full card: 21,535 identical "illegal memory access" lines followed by
// "malloc(): unaligned tcache chunk detected" and std::length_error -- i.e. the host heap was
// eventually corrupted and the server died. A server must degrade to a clean 503, not that.
//
// So: record it once, let callers ask, and refuse new work instead of grinding on. This is
// deliberately one-way -- there is no recovery short of restarting the process, and pretending
// otherwise would just produce a subtler failure later.
inline std::atomic<bool>& device_lost_flag() {
    static std::atomic<bool> lost{false};
    return lost;
}

inline bool device_lost() { return device_lost_flag().load(std::memory_order_relaxed); }

// True for errors that leave the context unusable. Allocation failures are deliberately NOT here:
// running out of memory is a normal, recoverable capacity condition that the engine already
// reports as 503/429, and treating it as fatal would take a server down for ordinary load.
inline bool is_unrecoverable(cudaError_t e) {
    switch (e) {
        case cudaErrorIllegalAddress:
        case cudaErrorMisalignedAddress:
        case cudaErrorIllegalInstruction:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorLaunchFailure:
        case cudaErrorHardwareStackError:
        case cudaErrorECCUncorrectable:
        case cudaErrorContextIsDestroyed:
        case cudaErrorDeviceUninitialized:
            return true;
        default:
            return false;
    }
}

// ---- Why the engine went down (dual-gpu WP-5, the failure policy) ------------------------------
//
// tp=2 makes the two cards ONE logical device: a shared KV block numbering, one all-reduce per
// layer, one decode loop. The failure policy is therefore decision (a) of the plan (manifest
// CHG-0034): ANY fatal error on EITHER card -- a sticky CUDA error in either rank's context, a
// GpuLink collective that fails to post, or a host rendezvous between the ranks that times out --
// downgrades the WHOLE server through the one process-global flag above. Per-card isolation
// (serving degraded from the surviving card) is the rejected alternative: at 16 GB/card one card
// cannot hold the model, the ranks share per-layer collectives, and a half-reduced hidden state is
// silently wrong rather than slow. Effects (identical to tp=1's): every in-flight job fails at its
// next step with an explicit error, new requests get 503, /health reports 503 + the reason below,
// and only a process restart recovers.
//
// The first fatal event's reason and device are kept (later ones are consequences); device is -1
// when unknown. Recording happens on the error path only -- never on a hot path.
struct DeviceLostInfo {
    bool lost = false;
    int device = -1;          // CUDA ordinal of the card that failed; -1 = unknown / the link
    std::string reason;       // e.g. "cuda: an illegal memory access was encountered"
};

inline std::mutex& device_lost_mutex() {
    static std::mutex m;
    return m;
}
inline DeviceLostInfo& device_lost_info_slot() {
    static DeviceLostInfo info;
    return info;
}

// Mark the engine lost with a reason. The first caller wins the reason; the flag is set either
// way. Returns true if this call was the first fatal event.
inline bool note_device_fatal(int device, const char* reason) {
    bool first = false;
    {
        std::lock_guard<std::mutex> g(device_lost_mutex());
        DeviceLostInfo& info = device_lost_info_slot();
        if (!info.lost) {
            info.lost = true;
            info.device = device;
            info.reason = reason ? reason : "unknown";
            first = true;
        }
    }
    device_lost_flag().store(true, std::memory_order_relaxed);
    if (first)
        fprintf(stderr, "[device_health] engine marked unhealthy (device %d): %s -- in-flight "
                        "requests fail, new requests get 503, restart required\n",
                device, reason ? reason : "unknown");
    return first;
}

// A tp collective failure (a GpuLink op refused or failed to post, or the two ranks' host
// rendezvous timed out): the per-layer reduce did not happen, so both ranks' hidden states are
// undefined from here on. Not a CUDA context loss, but the same verdict -- the output can no
// longer be trusted, and the ranks may be out of step for good.
inline bool note_tp_fatal(const char* what) {
    std::string r = "tp=2 collective failure: ";
    r += what ? what : "unknown";
    return note_device_fatal(-1, r.c_str());
}

inline DeviceLostInfo device_lost_info() {
    std::lock_guard<std::mutex> g(device_lost_mutex());
    DeviceLostInfo info = device_lost_info_slot();
    if (!info.lost && device_lost()) info.lost = true;   // flag set without a reason
    return info;
}

// Call from every cu()-style wrapper. Returns true if this error killed the context. The failing
// card is the calling thread's current device: under tp=2 each rank's work runs on a thread bound
// to that rank's card, so this names which card died (CHG-0034: "logs which card").
inline bool note_cuda_error(cudaError_t e) {
    if (e == cudaSuccess) return false;
    if (!is_unrecoverable(e)) return false;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = -1;
    std::string r = "cuda: ";
    r += cudaGetErrorString(e);
    note_device_fatal(dev, r.c_str());
    return true;
}

}  // namespace sparkinfer
