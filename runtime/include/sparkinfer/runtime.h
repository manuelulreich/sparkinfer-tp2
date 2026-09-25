#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sparkinfer/gpu_stats.h"

namespace sparkinfer {

struct RuntimeConfig {
    int device_id = 0;
    size_t kv_cache_bytes = 0;      // 0 = auto (80% of free VRAM)
    size_t expert_cache_bytes = 0;  // MoE expert residency budget
    int max_batch_size = 256;
    int max_seq_len = 32768;
    bool enable_cuda_graphs = true;
    bool enable_chunked_prefill = true;
    // Tensor-parallel size (dual-gpu WP-4). 1 = today's single-device behaviour.
    int tp = 1;
    // The explicit device ids the tensor-parallel ranks span. The effective device list
    // is `devices` when non-empty, else {device_id} -- the tp=1 default, which is today's
    // single-device path (device 0) byte-identically.
    std::vector<int> devices = {0};
};

// One row of the runtime's per-device property table: what initialize() queried for each
// effective device (a single row for the default tp=1, i.e. today's one card).
struct GpuDeviceInfo {
    int id = 0;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int num_sms = 0;
    float bandwidth_gbps = 0.f;
};

class Runtime {
public:
    static std::unique_ptr<Runtime> create(const RuntimeConfig& cfg);

    virtual ~Runtime() = default;

    virtual void initialize() = 0;
    virtual void shutdown() = 0;

    // Returns device peak memory bandwidth in GB/s
    virtual float memory_bandwidth_gbps() const = 0;

    // Returns number of available CUDA SMs
    virtual int num_sms() const = 0;

    // Engine-level GPU observability: a live sample of heat (°C) + VRAM (+ power/clock) on this
    // runtime's device. Safe to poll periodically (e.g. while decoding) to watch thermals/throttle.
    virtual GpuStats gpu_stats() const = 0;

    // The per-device property table filled by initialize(): one row per effective device, in
    // the config's list order (one row at the default tp=1).
    virtual int device_count() const = 0;
    virtual GpuDeviceInfo device_info(size_t i) const = 0;
};

} // namespace sparkinfer
