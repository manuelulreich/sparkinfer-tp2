// dual-gpu WP-5: the per-card observability surface -- /metrics exposition and the /v1/info JSON
// fragment rendered from server/include/gpu_metrics.hpp. Pure C++: no CUDA, no GPU, no runtime
// linkage. Pins the shape at tp=1 (one row, rank 0) and tp=2 (two rows, distinct labels), the
// omit-when-unreported rule, the governing (max) temperature, and the unhealthy fields.

#include "../../server/include/gpu_metrics.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace sparkinfer_server;

static int g_fail = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL: %s (line %d)\n", #x, __LINE__); ++g_fail; } } while (0)

static bool has(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
}
static int count(const std::string& s, const std::string& needle) {
    int n = 0;
    for (size_t p = s.find(needle); p != std::string::npos; p = s.find(needle, p + needle.size())) ++n;
    return n;
}

static GpuRow card(int rank, int dev, int temp, int power, int util) {
    GpuRow r;
    r.rank = rank;
    r.device = dev;
    r.name = "NVIDIA GeForce RTX 5060 Ti";
    r.valid = true;
    r.temp_c = temp;
    r.power_w = power;
    r.util_pct = util;
    r.sm_clock_mhz = 2700;
    r.vram_used_bytes = 10000000000ull;
    r.vram_total_bytes = 17103323136ull;
    r.kv_free_blocks = 100 + rank;
    r.kv_total_blocks = 8200;
    return r;
}

int main() {
    // ---- tp=1: exactly one sample per family, rank 0 -------------------------------------------
    {
        const std::vector<GpuRow> rows = {card(0, 0, 61, 120, 95)};
        const std::string m = render_gpu_metrics(rows, 1, true);
        CHECK(has(m, "sparkinfer_tp_size 1\n"));
        CHECK(has(m, "sparkinfer_device_healthy 1\n"));
        CHECK(has(m, "sparkinfer_gpu_temperature_celsius{rank=\"0\",device=\"0\",name=\"NVIDIA GeForce RTX 5060 Ti\"} 61\n"));
        CHECK(has(m, "sparkinfer_gpu_vram_used_bytes{rank=\"0\",device=\"0\",name=\"NVIDIA GeForce RTX 5060 Ti\"} 10000000000\n"));
        CHECK(has(m, "sparkinfer_gpu_vram_total_bytes{rank=\"0\",device=\"0\",name=\"NVIDIA GeForce RTX 5060 Ti\"} 17103323136\n"));
        CHECK(has(m, "sparkinfer_gpu_power_watts{rank=\"0\""));
        CHECK(has(m, "sparkinfer_gpu_utilization_percent{rank=\"0\",device=\"0\",name=\"NVIDIA GeForce RTX 5060 Ti\"} 95\n"));
        CHECK(has(m, "sparkinfer_kv_pool_free_blocks{rank=\"0\",device=\"0\",name=\"NVIDIA GeForce RTX 5060 Ti\"} 100\n"));
        CHECK(has(m, "sparkinfer_gpu_max_temperature_celsius 61\n"));
        CHECK(count(m, "sparkinfer_gpu_temperature_celsius{") == 1);
        CHECK(count(m, "# TYPE sparkinfer_gpu_temperature_celsius gauge") == 1);
        CHECK(!has(m, "rank=\"1\""));

        const std::string j = render_gpu_info_json(rows, 1, "", true, "", -1);
        CHECK(j.rfind(",\"tp\":1,\"devices\":[{\"rank\":0,\"device\":0,\"name\":\"NVIDIA GeForce RTX 5060 Ti\"", 0) == 0);
        CHECK(has(j, "\"temperature_c\":61"));
        CHECK(has(j, "\"kv_free_blocks\":100,\"kv_total_blocks\":8200}]"));
        CHECK(!has(j, "\"link\""));             // tp=1 has no link
        CHECK(has(j, ",\"healthy\":true"));
        CHECK(!has(j, "unhealthy_reason"));
        CHECK(count(j, "{\"rank\":") == 1);
    }

    // ---- tp=2: two rows, distinct labels; the governing temperature is the hotter card ----------
    {
        const std::vector<GpuRow> rows = {card(0, 0, 58, 150, 90), card(1, 1, 73, 160, 88)};
        const std::string m = render_gpu_metrics(rows, 2, true);
        CHECK(has(m, "sparkinfer_tp_size 2\n"));
        CHECK(count(m, "sparkinfer_gpu_temperature_celsius{") == 2);
        CHECK(count(m, "# HELP sparkinfer_gpu_temperature_celsius ") == 1);   // one family header
        CHECK(has(m, "sparkinfer_gpu_temperature_celsius{rank=\"1\",device=\"1\",name=\"NVIDIA GeForce RTX 5060 Ti\"} 73\n"));
        CHECK(has(m, "sparkinfer_kv_pool_free_blocks{rank=\"1\",device=\"1\",name=\"NVIDIA GeForce RTX 5060 Ti\"} 101\n"));
        CHECK(has(m, "sparkinfer_gpu_max_temperature_celsius 73\n"));
        CHECK(gpu_governing_temp_c(rows) == 73);

        const std::string j = render_gpu_info_json(rows, 2, "p2p-mapped", true, "", -1);
        CHECK(j.rfind(",\"tp\":2,\"devices\":[", 0) == 0);
        CHECK(count(j, "{\"rank\":") == 2);
        CHECK(has(j, "{\"rank\":1,\"device\":1,"));
        CHECK(has(j, "],\"link\":\"p2p-mapped\",\"healthy\":true"));
    }

    // ---- unreported readings are omitted (/metrics) or null (/v1/info), never invented ----------
    {
        GpuRow r = card(0, 3, -1, -1, -1);
        r.sm_clock_mhz = -1;
        r.kv_free_blocks = -1;
        r.kv_total_blocks = -1;
        GpuRow dead = card(1, 4, -1, -1, -1);
        dead.valid = false;   // even VRAM could not be read (e.g. a lost context)
        dead.sm_clock_mhz = -1;   // valid=false means NVML failed too: every reading is -1
        dead.kv_free_blocks = -1;
        dead.kv_total_blocks = -1;
        const std::vector<GpuRow> rows = {r, dead};
        const std::string m = render_gpu_metrics(rows, 2, true);
        CHECK(!has(m, "sparkinfer_gpu_temperature_celsius"));
        CHECK(!has(m, "sparkinfer_gpu_power_watts"));
        CHECK(!has(m, "sparkinfer_gpu_max_temperature_celsius"));
        CHECK(!has(m, "sparkinfer_kv_pool_free_blocks"));
        CHECK(count(m, "sparkinfer_gpu_vram_used_bytes{") == 1);   // the invalid row is skipped
        CHECK(!has(m, "device=\"4\""));
        const std::string j = render_gpu_info_json(rows, 2, "pinned-staging", true, "", -1);
        CHECK(has(j, "\"temperature_c\":null,\"power_w\":null,\"utilization_pct\":null"));
        CHECK(has(j, "{\"rank\":1,\"device\":4,\"name\":\"NVIDIA GeForce RTX 5060 Ti\",\"vram_used_bytes\":null,\"vram_total_bytes\":null"));
        CHECK(gpu_governing_temp_c(rows) == -1);
        CHECK(gpu_governing_temp_c({}) == -1);
    }

    // ---- unhealthy: the gauge drops to 0, /v1/info names the first fatal event -----------------
    {
        const std::vector<GpuRow> rows = {card(0, 0, 60, 100, 50), card(1, 1, 62, 100, 50)};
        const std::string m = render_gpu_metrics(rows, 2, false);
        CHECK(has(m, "sparkinfer_device_healthy 0\n"));
        const std::string j = render_gpu_info_json(rows, 2, "p2p-mapped", false,
                                                   "tp=2 collective failure: rendezvous timeout", -1);
        CHECK(has(j, ",\"healthy\":false,\"unhealthy_reason\":\"tp=2 collective failure: rendezvous timeout\",\"unhealthy_device\":null"));
        const std::string j2 = render_gpu_info_json(rows, 2, "p2p-mapped", false,
                                                    "cuda: an \"illegal\" memory access", 1);
        CHECK(has(j2, "\"unhealthy_reason\":\"cuda: an \\\"illegal\\\" memory access\",\"unhealthy_device\":1"));
    }

    // ---- escaping ----------------------------------------------------------------------------------
    {
        GpuRow r = card(0, 0, 50, 10, 10);
        r.name = "we\"ird\\name\n";
        const std::string m = render_gpu_metrics({r}, 1, true);
        CHECK(has(m, "name=\"we\\\"ird\\\\name\\n\""));
        const std::string j = render_gpu_info_json({r}, 1, "", true, "", -1);
        CHECK(has(j, "\"name\":\"we\\\"ird\\\\name\\u000a\""));
    }

    if (g_fail) {
        printf("gpu_metrics_cpu_test: %d FAILED\n", g_fail);
        return 1;
    }
    printf("gpu_metrics_cpu_test: OK\n");
    return 0;
}
