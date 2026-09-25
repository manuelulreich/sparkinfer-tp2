// Runtime — device setup and capability query.

#include "sparkinfer/runtime.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <vector>

namespace sparkinfer {

class RuntimeImpl : public Runtime {
public:
    explicit RuntimeImpl(const RuntimeConfig& cfg) : cfg_(cfg) {}

    // The effective device list: cfg_.devices when the config names any, else
    // {cfg_.device_id} -- the tp=1 default, which is today's single-card path.
    std::vector<int> effective_devices() const {
        if (!cfg_.devices.empty()) return cfg_.devices;
        return {cfg_.device_id};
    }

    void initialize() override {
        const std::vector<int> eff = effective_devices();
        // The one cudaSetDevice this runtime makes: on the first effective device (the tp=1
        // default: today's cudaSetDevice(0), and nothing in the default path changes). The
        // further devices are only ever queried by ordinal below -- this runtime establishes
        // no process-global "current device" beyond that one call (dual-gpu plan §2: the
        // later split work drives each card through stream-scoped launches, not setDevice).
        cudaError_t e = cudaSetDevice(eff.front());
        if (e != cudaSuccess) { fprintf(stderr, "[runtime] setDevice: %s\n", cudaGetErrorString(e)); return; }
        // One property row per effective device (the tp=1 default: exactly today's one card,
        // one query per attribute, one log line).
        devices_.clear();
        for (int dev : eff) {
            GpuDeviceInfo d{};
            d.id = dev;
            cudaDeviceProp p{};
            e = cudaGetDeviceProperties(&p, dev);
            if (e != cudaSuccess) {
                // The single-device message stays byte-identical to the legacy one; only a
                // multi-device table names the failing card.
                if (eff.size() > 1) fprintf(stderr, "[runtime] getProps: device %d: %s\n", dev, cudaGetErrorString(e));
                else fprintf(stderr, "[runtime] getProps: %s\n", cudaGetErrorString(e));
                return;
            }
            d.name = p.name;
            d.cc_major = p.major;
            d.cc_minor = p.minor;
            d.num_sms = p.multiProcessorCount;
            // bandwidth = 2 (DDR) * memClock(Hz) * busWidth(bytes) / 1e9.
            // memoryClockRate/memoryBusWidth were removed from cudaDeviceProp in CUDA 13;
            // query them via device attributes (portable across CUDA 12.x and 13.x).
            int mem_clock_khz = 0, bus_width_bits = 0;
            cudaDeviceGetAttribute(&mem_clock_khz,  cudaDevAttrMemoryClockRate,      dev);
            cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
            const double mem_hz = (double)mem_clock_khz * 1e3;
            const double bus_bytes = bus_width_bits / 8.0;
            d.bandwidth_gbps = (float)(2.0 * mem_hz * bus_bytes / 1e9);
            printf("[runtime] %s  cc=%d.%d  SMs=%d  BW=%.0f GB/s\n",
                   p.name, p.major, p.minor, d.num_sms, d.bandwidth_gbps);
            devices_.push_back(std::move(d));
        }
        // The scalar members mirror row 0: the existing num_sms()/memory_bandwidth_gbps()
        // contract. At the default tp=1 row 0 is the only row and today's device 0, so the
        // values are exactly what the old single-device path stored.
        num_sms_ = devices_[0].num_sms;
        bandwidth_gbps_ = devices_[0].bandwidth_gbps;
        cc_major_ = devices_[0].cc_major;
        cc_minor_ = devices_[0].cc_minor;
    }

    void shutdown() override { cudaDeviceSynchronize(); }
    float memory_bandwidth_gbps() const override { return bandwidth_gbps_; }
    int   num_sms() const override { return num_sms_; }
    GpuStats gpu_stats() const override {
        // The first effective device (device 0 at the default); the empty-table fallback is
        // cfg_.device_id, i.e. today's behaviour when initialize() never completed a row.
        return query_gpu_stats(devices_.empty() ? cfg_.device_id : devices_[0].id);
    }
    int device_count() const override { return static_cast<int>(devices_.size()); }
    GpuDeviceInfo device_info(size_t i) const override { return devices_.at(i); }

private:
    RuntimeConfig cfg_;
    std::vector<GpuDeviceInfo> devices_;
    int num_sms_ = 0, cc_major_ = 0, cc_minor_ = 0;
    float bandwidth_gbps_ = 0.f;
};

std::unique_ptr<Runtime> Runtime::create(const RuntimeConfig& cfg) {
    return std::unique_ptr<Runtime>(new RuntimeImpl(cfg));
}

} // namespace sparkinfer
