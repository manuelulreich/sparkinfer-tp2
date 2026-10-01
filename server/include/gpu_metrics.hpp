// Per-card observability rendering (dual-gpu WP-5; manifest CHG-0018 / CHG-0021 / CHG-0034).
//
// One GpuRow per tensor-parallel rank -- a single row at tp=1, two at tp=2. ModelEngine fills the
// rows (live NVML/cudaMemGetInfo sample + that rank's KV pool occupancy); this header only turns
// them into the /metrics exposition text and the /v1/info JSON fragment. Pure C++ -- no CUDA, no
// runtime linkage, no JSON library -- so the exact output shape is unit-testable on any box
// (runtime/tests/gpu_metrics_cpu_test.cpp).
//
// Conventions: a reading the card did not report (NVML absent, or the query failed) is -1 in the
// row; /metrics then OMITS that sample (Prometheus has no null) and /v1/info writes JSON null.
// Every per-card sample carries {rank, device, name} labels so a two-card scrape is unambiguous.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace sparkinfer_server {

struct GpuRow {
    int rank = 0;                 // tensor-parallel rank (0 = the leader / the tp=1 device)
    int device = -1;              // CUDA ordinal
    std::string name;             // e.g. "NVIDIA GeForce RTX 5060 Ti"
    bool valid = false;           // false when not even VRAM could be read
    int temp_c = -1;
    int power_w = -1;
    int sm_clock_mhz = -1;
    int util_pct = -1;
    uint64_t vram_used_bytes = 0;
    uint64_t vram_total_bytes = 0;
    int kv_free_blocks = -1;      // this rank's KV pool; -1 = unknown
    int kv_total_blocks = -1;
};

// The temperature the thermal policy governs on: the hottest card (max over the reported
// readings), -1 when none reported. Mirrors ThermalGovernor::governing_temp_c.
inline int gpu_governing_temp_c(const std::vector<GpuRow>& rows) {
    int m = -1;
    for (const auto& r : rows)
        if (r.temp_c >= 0 && r.temp_c > m) m = r.temp_c;
    return m;
}

namespace gpu_metrics_detail {

// Prometheus label-value escaping: backslash, double quote, newline.
inline std::string prom_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '"') o += "\\\"";
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

// JSON string escaping (control characters as \u00XX).
inline std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c < 0x20) {
            char b[8];
            snprintf(b, sizeof b, "\\u%04x", c);
            o += b;
        } else o += (char)c;
    }
    return o;
}

inline std::string labels(const GpuRow& r) {
    return "{rank=\"" + std::to_string(r.rank) + "\",device=\"" + std::to_string(r.device) +
           "\",name=\"" + prom_escape(r.name) + "\"}";
}

inline std::string json_int_or_null(long long v) { return v < 0 ? "null" : std::to_string(v); }

}  // namespace gpu_metrics_detail

// The /metrics block: tp size, health, and per-card gauges. Appended after the existing
// metrics; at tp=1 every per-card family has exactly one sample (rank 0).
inline std::string render_gpu_metrics(const std::vector<GpuRow>& rows, int tp, bool healthy) {
    using namespace gpu_metrics_detail;
    std::string o;
    o += "# HELP sparkinfer_tp_size Tensor-parallel size (number of GPUs one model spans)\n"
         "# TYPE sparkinfer_tp_size gauge\n"
         "sparkinfer_tp_size " + std::to_string(tp) + "\n";
    o += "# HELP sparkinfer_device_healthy 1 while every GPU context is usable; 0 after a fatal "
         "error on any card or the tp link (restart required)\n"
         "# TYPE sparkinfer_device_healthy gauge\n"
         "sparkinfer_device_healthy " + std::string(healthy ? "1" : "0") + "\n";

    struct Fam {
        const char* name;
        const char* help;
        long long (*get)(const GpuRow&);
        bool need_valid;   // VRAM fields are meaningless when the sample failed
    };
    static const Fam fams[] = {
        {"sparkinfer_gpu_vram_used_bytes", "Resident VRAM per GPU (total - free)",
         [](const GpuRow& r) { return (long long)r.vram_used_bytes; }, true},
        {"sparkinfer_gpu_vram_total_bytes", "VRAM capacity per GPU",
         [](const GpuRow& r) { return (long long)r.vram_total_bytes; }, true},
        {"sparkinfer_gpu_temperature_celsius", "GPU core temperature",
         [](const GpuRow& r) { return (long long)r.temp_c; }, false},
        {"sparkinfer_gpu_power_watts", "Instantaneous board power draw",
         [](const GpuRow& r) { return (long long)r.power_w; }, false},
        {"sparkinfer_gpu_utilization_percent", "GPU utilization over the driver's last sample period",
         [](const GpuRow& r) { return (long long)r.util_pct; }, false},
        {"sparkinfer_gpu_sm_clock_mhz", "SM clock (drops when thermal-throttling)",
         [](const GpuRow& r) { return (long long)r.sm_clock_mhz; }, false},
        {"sparkinfer_kv_pool_free_blocks", "Free KV cache blocks in this rank's pool",
         [](const GpuRow& r) { return (long long)r.kv_free_blocks; }, false},
        {"sparkinfer_kv_pool_total_blocks", "Total KV cache blocks in this rank's pool",
         [](const GpuRow& r) { return (long long)r.kv_total_blocks; }, false},
    };
    for (const Fam& f : fams) {
        std::string samples;
        for (const GpuRow& r : rows) {
            if (f.need_valid && !r.valid) continue;
            const long long v = f.get(r);
            if (v < 0) continue;   // not reported: omit rather than invent a value
            samples += std::string(f.name) + labels(r) + " " + std::to_string(v) + "\n";
        }
        if (samples.empty()) continue;
        o += std::string("# HELP ") + f.name + " " + f.help + "\n# TYPE " + f.name + " gauge\n";
        o += samples;
    }
    const int gt = gpu_governing_temp_c(rows);
    if (gt >= 0)
        o += "# HELP sparkinfer_gpu_max_temperature_celsius Hottest card's temperature (the "
             "thermal policy's governing reading: max over the tp ranks)\n"
             "# TYPE sparkinfer_gpu_max_temperature_celsius gauge\n"
             "sparkinfer_gpu_max_temperature_celsius " + std::to_string(gt) + "\n";
    return o;
}

// The /v1/info fields this WP adds, as a JSON object-member list WITHOUT the surrounding braces
// and with a leading comma, so the handler appends it to its existing object unchanged:
//   ,"tp":2,"devices":[{...},{...}],"link":"p2p-mapped","healthy":true
// `link` is omitted when empty (tp=1); `unhealthy_reason` only appears when !healthy.
inline std::string render_gpu_info_json(const std::vector<GpuRow>& rows, int tp,
                                        const std::string& link, bool healthy,
                                        const std::string& unhealthy_reason, int unhealthy_device) {
    using namespace gpu_metrics_detail;
    std::string o = ",\"tp\":" + std::to_string(tp) + ",\"devices\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const GpuRow& r = rows[i];
        if (i) o += ",";
        o += "{\"rank\":" + std::to_string(r.rank) + ",\"device\":" + std::to_string(r.device) +
             ",\"name\":\"" + json_escape(r.name) + "\"" +
             ",\"vram_used_bytes\":" + (r.valid ? std::to_string(r.vram_used_bytes) : "null") +
             ",\"vram_total_bytes\":" + (r.valid ? std::to_string(r.vram_total_bytes) : "null") +
             ",\"temperature_c\":" + json_int_or_null(r.temp_c) +
             ",\"power_w\":" + json_int_or_null(r.power_w) +
             ",\"utilization_pct\":" + json_int_or_null(r.util_pct) +
             ",\"sm_clock_mhz\":" + json_int_or_null(r.sm_clock_mhz) +
             ",\"kv_free_blocks\":" + json_int_or_null(r.kv_free_blocks) +
             ",\"kv_total_blocks\":" + json_int_or_null(r.kv_total_blocks) + "}";
    }
    o += "]";
    if (!link.empty()) o += ",\"link\":\"" + json_escape(link) + "\"";
    o += std::string(",\"healthy\":") + (healthy ? "true" : "false");
    if (!healthy) {
        o += ",\"unhealthy_reason\":\"" + json_escape(unhealthy_reason) + "\"";
        o += ",\"unhealthy_device\":" + json_int_or_null(unhealthy_device);
    }
    return o;
}

}  // namespace sparkinfer_server
