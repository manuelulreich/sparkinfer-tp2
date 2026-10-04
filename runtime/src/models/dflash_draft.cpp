// DFlash draft runtime: safetensors load + GGUF load + block-parallel forward.
#include "sparkinfer/pinned_pool.h"
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/device_health.h"
#include <atomic>
#include "sparkinfer/models/dflash_kernels.h"
#include "sparkinfer/models/qwen35.h"   // tp_allreduce_bf16_on, tp_run_with_peer (split draft)
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/quant.h"
#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/compressed_tensors.h"
#include "sparkinfer/gguf.h"
// Header-only Muse Glimmer DFlash draft config derivation (mirrors examples/qwen3_gguf_config.h's
// museglimmer_config_from_gguf for the target model). Lives in examples/ by this codebase's
// convention for GGUF-config-from-metadata helpers; reachable here via the runtime library's
// PRIVATE "examples" include dir (see runtime/CMakeLists.txt).
#include "dflash_gguf_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sparkinfer {
namespace {

using bf16 = __nv_bfloat16;

// Every CUDA call cu() has seen fail, over the life of the process. load() and load_gguf() compare
// it across the load: a draft whose weights or scratch did not fit has NOT loaded, however much of
// it did. Without this the loader logged "malloc: out of memory" (and then a lost context), still
// returned true, and the server started with speculative decoding advertised and every request
// failing (#1086).
std::atomic<int> g_cu_errors{0};

inline void cu(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    g_cu_errors.fetch_add(1, std::memory_order_relaxed);
    // See device_health.h: sticky errors kill the context, so record them and let the
    // engine refuse work rather than issuing more against a dead device.
    const bool fatal = note_cuda_error(e);
    static std::atomic<int> logged{0};
    const int n = logged.fetch_add(1, std::memory_order_relaxed);
    // Fatal lines print too, but only the first few: once the context is gone every call in
    // flight fails the same way, and a stress run wrote 2.4 GB of them before it drained.
    static std::atomic<int> fatal_logged{0};
    if (n < 20 || (fatal && fatal_logged.fetch_add(1, std::memory_order_relaxed) < 8))
        fprintf(stderr, "[dflash] %s: %s%s\n", what, cudaGetErrorString(e),
                fatal ? "  [CONTEXT LOST -- server will refuse further work]" : "");
    else if (n == 20)
        fprintf(stderr, "[dflash] (further CUDA errors suppressed)\n");
}

struct TensorView {
    void* data = nullptr;
    size_t nbytes = 0;
    std::vector<int64_t> shape;
    std::string dtype;   // "BF16" | "F32" | "U8" (packed NVFP4) | "F8_E4M3" (group scales)
};

// Minimal safetensors reader (BF16 / F32 / NVFP4 payload). Owns a host mmap-like buffer.
struct SafeTensorsFile {
    std::vector<char> bytes;
    std::unordered_map<std::string, TensorView> tensors; // pointers into bytes

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.seekg(0, std::ios::end);
        const std::streamoff sz = f.tellg();
        if (sz < 8) return false;
        f.seekg(0, std::ios::beg);
        bytes.resize((size_t)sz);
        f.read(bytes.data(), sz);
        if (!f) return false;
        uint64_t hdr_len = 0;
        memcpy(&hdr_len, bytes.data(), 8);
        if (8 + hdr_len > (uint64_t)sz) return false;
        const std::string hdr(bytes.data() + 8, bytes.data() + 8 + hdr_len);
        // Parse each "name":{...} entry for dtype/shape/data_offsets.
        size_t pos = 0;
        while (pos < hdr.size()) {
            size_t key_start = hdr.find('"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = hdr.find('"', key_start + 1);
            if (key_end == std::string::npos) break;
            std::string key = hdr.substr(key_start + 1, key_end - key_start - 1);
            // Tensor entries are always "name": { ... }
            size_t colon = hdr.find(':', key_end);
            if (colon == std::string::npos) break;
            size_t obj = hdr.find('{', colon);
            if (obj == std::string::npos || obj > colon + 4) {
                // Not a tensor object (e.g. string metadata field) — advance past this key.
                pos = key_end + 1;
                continue;
            }
            if (key == "__metadata__") {
                // Skip nested object.
                int depth = 0;
                size_t i = obj;
                for (; i < hdr.size(); i++) {
                    if (hdr[i] == '{') depth++;
                    else if (hdr[i] == '}') {
                        depth--;
                        if (depth == 0) { i++; break; }
                    }
                }
                pos = i;
                continue;
            }
            size_t obj_end = hdr.find('}', obj);
            if (obj_end == std::string::npos) break;
            std::string body = hdr.substr(obj, obj_end - obj + 1);
            // dtype
            std::string dtype;
            size_t d0 = body.find("\"dtype\"");
            if (d0 != std::string::npos) {
                size_t c = body.find(':', d0);
                size_t q0 = body.find('"', c);
                size_t q1 = body.find('"', q0 + 1);
                if (q0 != std::string::npos && q1 != std::string::npos)
                    dtype = body.substr(q0 + 1, q1 - q0 - 1);
            }
            // data_offsets: [start, end]
            int64_t off0 = 0, off1 = 0;
            size_t o0 = body.find("\"data_offsets\"");
            if (o0 != std::string::npos) {
                size_t b0 = body.find('[', o0);
                off0 = strtoll(body.c_str() + b0 + 1, nullptr, 10);
                size_t comma = body.find(',', b0);
                off1 = strtoll(body.c_str() + comma + 1, nullptr, 10);
            }
            std::vector<int64_t> shape;
            size_t s0 = body.find("\"shape\"");
            if (s0 != std::string::npos) {
                size_t b0 = body.find('[', s0);
                size_t b1 = body.find(']', b0);
                std::string ss = body.substr(b0 + 1, b1 - b0 - 1);
                size_t p = 0;
                while (p < ss.size()) {
                    while (p < ss.size() && (ss[p] == ' ' || ss[p] == ',')) p++;
                    if (p >= ss.size()) break;
                    shape.push_back(strtoll(ss.c_str() + p, nullptr, 10));
                    while (p < ss.size() && ss[p] != ',') p++;
                }
            }
            // U8 = NVFP4-packed weight (2x E2M1 per byte), F8_E4M3 = its per-group scale.
            // Both are payload for the NVFP4 dequant below, not tensors to skip.
            if (dtype != "BF16" && dtype != "F32" && dtype != "BOOL" &&
                dtype != "U8" && dtype != "F8_E4M3") {
                fprintf(stderr, "[dflash] skip tensor %s dtype=%s\n", key.c_str(), dtype.c_str());
                pos = obj_end + 1;
                continue;
            }
            TensorView tv;
            tv.dtype = dtype;
            tv.shape = shape;
            tv.nbytes = (size_t)(off1 - off0);
            tv.data = bytes.data() + 8 + hdr_len + off0;
            tensors[key] = tv;
            pos = obj_end + 1;
        }
        if (tensors.empty())
            fprintf(stderr, "[dflash] safetensors parse produced 0 tensors (hdr_len=%llu)\n",
                    (unsigned long long)hdr_len);
        return !tensors.empty();
    }
};

bool parse_config_json(const std::string& path, DFlashDraftConfig& cfg) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string j = ss.str();
    auto find_int = [&](const char* key, int& dst) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = j.find(pat);
        if (p == std::string::npos) return;
        size_t c = j.find(':', p);
        dst = (int)strtol(j.c_str() + c + 1, nullptr, 10);
    };
    auto find_float = [&](const char* key, float& dst) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = j.find(pat);
        if (p == std::string::npos) return;
        size_t c = j.find(':', p);
        dst = strtof(j.c_str() + c + 1, nullptr);
    };
    find_int("hidden_size", cfg.hidden);
    find_int("intermediate_size", cfg.intermediate);
    find_int("num_hidden_layers", cfg.n_layers);
    find_int("num_attention_heads", cfg.n_q_heads);
    find_int("num_key_value_heads", cfg.n_kv_heads);
    find_int("head_dim", cfg.head_dim);
    find_int("vocab_size", cfg.vocab);
    find_int("sliding_window", cfg.sliding_window);
    find_float("rms_norm_eps", cfg.rms_eps);
    find_int("block_size", cfg.block_size);
    // YaRN lives under "rope_parameters". Only applied when rope_type is actually yarn -- a
    // checkpoint carrying factor but rope_type "linear"/"default" must not silently get YaRN.
    if (j.find("\"yarn\"") != std::string::npos) {
        find_float("factor", cfg.yarn_factor);
        find_int("original_max_position_embeddings", cfg.yarn_orig_max_pos);
        find_float("beta_fast", cfg.yarn_beta_fast);
        find_float("beta_slow", cfg.yarn_beta_slow);
    }
    find_int("mask_token_id", cfg.mask_token_id);

    // DSpark (RadixArk/Qwen3.8-27B-DSpark) nests the draft-specific settings under
    // "dflash_config", and its target_layer_ids differ from this struct's Qwen3.6 default both in
    // VALUE and in COUNT: [4,16,28,40,52] (5 captures) vs {1,6,11,16,22,27,32,37} (8). The count
    // is load-bearing -- fc.weight is [hidden, n_cap*hidden], so leaving the default in place
    // makes the loader demand a [5120, 40960] projector against the checkpoint's [5120, 25600]
    // and fail outright. Parse the list rather than inheriting it.
    {
        size_t p = j.find("\"target_layer_ids\"");
        if (p != std::string::npos) {
            size_t a = j.find('[', p), b = j.find(']', a);
            if (a != std::string::npos && b != std::string::npos) {
                std::vector<int> ids;
                const char* q = j.c_str() + a + 1;
                const char* end = j.c_str() + b;
                while (q < end) {
                    char* nx = nullptr;
                    long v = strtol(q, &nx, 10);
                    if (nx == q) { q++; continue; }
                    ids.push_back((int)v);
                    q = nx;
                }
                if (!ids.empty()) cfg.target_layer_ids = ids;
            }
        }
    }
    // rope_theta nested
    size_t rp = j.find("\"rope_theta\"");
    if (rp != std::string::npos) {
        size_t c = j.find(':', rp);
        cfg.rope_theta = strtof(j.c_str() + c + 1, nullptr);
    }
    // dflash_config.block_size / mask_token_id / target_layer_ids
    size_t df = j.find("\"dflash_config\"");
    if (df != std::string::npos) {
        size_t bs = j.find("\"block_size\"", df);
        if (bs != std::string::npos) {
            size_t c = j.find(':', bs);
            cfg.block_size = (int)strtol(j.c_str() + c + 1, nullptr, 10);
        }
        size_t mt = j.find("\"mask_token_id\"", df);
        if (mt != std::string::npos) {
            size_t c = j.find(':', mt);
            cfg.mask_token_id = (int)strtol(j.c_str() + c + 1, nullptr, 10);
        }
        size_t tl = j.find("\"target_layer_ids\"", df);
        if (tl != std::string::npos) {
            size_t b0 = j.find('[', tl);
            size_t b1 = j.find(']', b0);
            cfg.target_layer_ids.clear();
            size_t p = b0 + 1;
            while (p < b1) {
                while (p < b1 && (j[p] == ' ' || j[p] == ',')) p++;
                if (p >= b1) break;
                cfg.target_layer_ids.push_back((int)strtol(j.c_str() + p, nullptr, 10));
                while (p < b1 && j[p] != ',') p++;
            }
        }
    }
    // layer_types
    size_t lt = j.find("\"layer_types\"");
    if (lt != std::string::npos) {
        size_t b0 = j.find('[', lt);
        size_t b1 = j.find(']', b0);
        cfg.sliding_layers.assign(cfg.n_layers, true);
        int idx = 0;
        size_t p = b0;
        while (p < b1 && idx < cfg.n_layers) {
            size_t q0 = j.find('"', p);
            if (q0 == std::string::npos || q0 >= b1) break;
            size_t q1 = j.find('"', q0 + 1);
            std::string t = j.substr(q0 + 1, q1 - q0 - 1);
            cfg.sliding_layers[idx++] = (t == "sliding_attention");
            p = q1 + 1;
        }
    } else {
        cfg.sliding_layers.assign(cfg.n_layers, true);
        if (cfg.n_layers > 0) cfg.sliding_layers[cfg.n_layers - 1] = false;
    }
    // DFlash2: its settings live under dflash_config; absent keys keep the reference defaults.
    {
        size_t a = j.find("\"architectures\"");
        size_t b = a == std::string::npos ? a : j.find(']', a);
        cfg.dflash2 = b != std::string::npos &&
                      j.substr(a, b - a).find("\"DFlash2DraftModel\"") != std::string::npos;
    }
    if (cfg.dflash2) {
        auto in_df = [&](const char* key, float& dst) {
            if (df == std::string::npos) return;
            size_t p = j.find(std::string("\"") + key + "\"", df);
            if (p == std::string::npos) return;
            const char* v = j.c_str() + j.find(':', p) + 1;
            while (*v == ' ') v++;
            if (strncmp(v, "null", 4) != 0) dst = strtof(v, nullptr);
        };
        float taps = (float)cfg.conv_taps, group = (float)cfg.conv_group;
        float rank = (float)cfg.selector_rank, topk = (float)cfg.selector_top_k;
        in_df("conv_kernel_size", taps);
        in_df("conv_group_size", group);
        in_df("selector_rank", rank);
        in_df("selector_top_k", topk);
        in_df("output_multiplier", cfg.output_multiplier);
        in_df("final_logit_softcapping", cfg.logit_softcap);
        in_df("input_embedding_scale", cfg.embed_scale);
        cfg.conv_taps = (int)taps;
        cfg.conv_group = (int)group;
        cfg.selector_rank = (int)rank;
        cfg.selector_top_k = (int)topk;
        size_t ic = j.find("\"is_causal\"");
        if (ic != std::string::npos) {
            const char* v = j.c_str() + j.find(':', ic) + 1;
            while (*v == ' ') v++;
            cfg.is_causal = strncmp(v, "true", 4) == 0 ? 1 : (strncmp(v, "false", 5) == 0 ? 0 : -1);
        }
        cfg.spec_depth = cfg.block_size - 1;
    }
    return true;
}

// ---- DFlash2 NVFP4 cache ----
//
// The DFlash2 checkpoint ships bf16. Its seven Linear weights per layer are quantized to NVFP4 once,
// on the host, into a safetensors file in ModelOpt's layout -- <name> U8 [rows, cols/2] packed e2m1
// (low nibble = even column), <name>_scale F8_E4M3 [rows, cols/16], <name>_scale_2 F32 -- with
// every other tensor copied as stored. Later loads read that file instead, so the draft loads as
// DSpark's NVFP4 checkpoint does. The rounding is launch_quantize_w_nvfp4's (global =
// 448 * 6 / amax(tensor), group scale e4m3(amax(group) / 6 * global), nearest e2m1).
//
// The cache sits beside the checkpoint when that directory is writable, else under
// $XDG_CACHE_HOME/sparkinfer (default ~/.cache/sparkinfer). Its metadata records the source's size
// and mtime; a source that changed rebuilds it. SPARKINFER_DFLASH2_NVFP4_CACHE=0 skips it (the
// load then quantizes on the GPU every time), =rebuild rebuilds it.
constexpr const char* kNvCacheFormat = "sparkinfer-dflash2-nvfp4-v1";

bool nv_cache_quantized(const std::string& name) {
    static const char* kSuffix[] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                    "mlp.gate_proj.weight",    "mlp.up_proj.weight",
                                    "mlp.down_proj.weight"};
    if (name.compare(0, 7, "layers.") != 0) return false;
    for (const char* s : kSuffix) {
        const size_t n = strlen(s);
        if (name.size() > n && name.compare(name.size() - n, n, s) == 0) return true;
    }
    return false;
}

// Positive e4m3 (float8_e4m3fn) values by code, 0x00..0x7E.
const std::vector<float>& e4m3_table() {
    static const std::vector<float> t = [] {
        std::vector<float> v(127);
        for (int b = 0; b < 127; b++) {
            const int e = (b >> 3) & 0xF, m = b & 7;
            v[b] = e == 0 ? std::ldexp((float)m / 8.f, -6) : std::ldexp(1.f + (float)m / 8.f, e - 7);
        }
        return v;
    }();
    return t;
}
// Nearest e4m3 code of a non-negative value, ties to the even code, saturating at 448.
unsigned char e4m3_encode(float a) {
    const std::vector<float>& t = e4m3_table();
    if (!(a > 0.f)) return 0;
    if (a >= t[126]) return 126;
    const int hi = (int)(std::lower_bound(t.begin(), t.end(), a) - t.begin());
    if (t[hi] == a) return (unsigned char)hi;
    const int lo = hi - 1;
    const float dl = a - t[lo], dh = t[hi] - a;
    if (dl < dh) return (unsigned char)lo;
    if (dh < dl) return (unsigned char)hi;
    return (unsigned char)((lo & 1) ? hi : lo);
}
unsigned e2m1_index(float a) {   // as nvfp4_e2m1_idx in dflash_kernels.cu
    if (a < 0.25f) return 0;
    if (a < 0.75f) return 1;
    if (a < 1.25f) return 2;
    if (a < 1.75f) return 3;
    if (a < 2.5f) return 4;
    if (a < 3.5f) return 5;
    if (a < 5.f) return 6;
    return 7;
}

inline float bf16_bits_to_float(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// One bf16 [rows, cols] tensor as packed e2m1, e4m3 group scales and the f32 scale_2.
void quantize_nvfp4_host(const uint16_t* w, int64_t rows, int64_t cols, std::vector<uint8_t>& packed,
                         std::vector<uint8_t>& scales, float& scale_2) {
    const int64_t n = rows * cols, groups = n / 16;
    float amax = 0.f;
    for (int64_t i = 0; i < n; i++) amax = std::max(amax, std::fabs(bf16_bits_to_float(w[i])));
    const float glob = amax > 0.f ? 448.f * 6.f / amax : 1.f;
    scale_2 = 1.f / glob;
    packed.assign((size_t)n / 2, 0);
    scales.assign((size_t)groups, 0);
    const std::vector<float>& t = e4m3_table();
    for (int64_t g = 0; g < groups; g++) {
        float v[16], am = 0.f;
        for (int i = 0; i < 16; i++) {
            v[i] = bf16_bits_to_float(w[g * 16 + i]);
            am = std::max(am, std::fabs(v[i]));
        }
        const unsigned char sb = e4m3_encode(am / 6.f * glob);
        scales[g] = sb;
        const float sq = t[sb];
        const float inv = sq > 0.f ? glob / sq : 0.f;
        for (int i = 0; i < 16; i += 2) {
            const float a = v[i] * inv, b = v[i + 1] * inv;
            const unsigned lo = e2m1_index(std::fabs(a)) | (a < 0.f ? 8u : 0u);
            const unsigned hi = e2m1_index(std::fabs(b)) | (b < 0.f ? 8u : 0u);
            packed[(size_t)g * 8 + i / 2] = (uint8_t)(lo | (hi << 4));
        }
    }
}

// The metadata a cache of `src` must carry (empty: `src` cannot be read).
std::string nv_cache_key(const std::string& src) {
    struct stat sb {};
    if (stat(src.c_str(), &sb) != 0) return {};
    const long long mt = (long long)sb.st_mtim.tv_sec * 1000000000LL + sb.st_mtim.tv_nsec;
    return std::string("\"format\":\"") + kNvCacheFormat + "\",\"source_size\":\"" +
           std::to_string((long long)sb.st_size) + "\",\"source_mtime_ns\":\"" +
           std::to_string(mt) + "\"";
}

// Whether `cache` is a complete cache carrying `key` (it is renamed into place only once written).
bool nv_cache_valid(const std::string& cache, const std::string& key) {
    std::ifstream f(cache, std::ios::binary);
    uint64_t n = 0;
    if (!f || !f.read(reinterpret_cast<char*>(&n), 8) || n == 0 || n > (64u << 20)) return false;
    std::string hdr(n, '\0');
    if (!f.read(&hdr[0], (std::streamsize)n)) return false;
    return hdr.find(key) != std::string::npos;
}

// Write the cache of `src` (a loaded model.safetensors) to `path`, through a temp file renamed
// into place. false: nothing written (the temp file is removed).
bool nv_cache_write(const SafeTensorsFile& src, const std::string& key, const std::string& path) {
    struct Out {
        std::string name, dtype;
        std::vector<int64_t> shape;
        const void* data = nullptr;
        size_t bytes = 0;
    };
    std::vector<std::string> names;
    for (const auto& kv : src.tensors) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    size_t nq = 0;
    for (const std::string& nm : names) nq += nv_cache_quantized(nm);
    std::vector<Out> outs;
    std::vector<std::vector<uint8_t>> payload(2 * nq);   // packed, scales per quantized tensor
    std::vector<float> scale2(nq);
    size_t q = 0;
    for (const std::string& nm : names) {
        const TensorView& t = src.tensors.at(nm);
        if (nv_cache_quantized(nm) && t.dtype == "BF16" && t.shape.size() == 2 &&
            t.shape[1] % 16 == 0) {
            std::vector<uint8_t>& packed = payload[2 * q];
            std::vector<uint8_t>& scales = payload[2 * q + 1];
            quantize_nvfp4_host((const uint16_t*)t.data, t.shape[0], t.shape[1], packed, scales,
                                scale2[q]);
            outs.push_back({nm, "U8", {t.shape[0], t.shape[1] / 2}, packed.data(), packed.size()});
            outs.push_back({nm + "_scale", "F8_E4M3", {t.shape[0], t.shape[1] / 16}, scales.data(),
                            scales.size()});
            outs.push_back({nm + "_scale_2", "F32", {}, &scale2[q], sizeof(float)});
            q++;
        } else {
            outs.push_back({nm, t.dtype, t.shape, t.data, t.nbytes});
        }
    }
    std::string hdr = std::string("{\"__metadata__\":{") + key + "}";
    size_t off = 0;
    for (const Out& o : outs) {
        hdr += ",\"" + o.name + "\":{\"dtype\":\"" + o.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < o.shape.size(); i++)
            hdr += (i ? "," : "") + std::to_string(o.shape[i]);
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + o.bytes) +
               "]}";
        off += o.bytes;
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';
    const std::string tmp = path + ".tmp." + std::to_string((long long)getpid());
    bool ok = false;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (f) {
            const uint64_t n = hdr.size();
            f.write(reinterpret_cast<const char*>(&n), 8);
            f.write(hdr.data(), (std::streamsize)hdr.size());
            for (const Out& o : outs)
                f.write(static_cast<const char*>(o.data), (std::streamsize)o.bytes);
            f.flush();
            ok = (bool)f;
        }
    }
    if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

// The NVFP4 cache of the DFlash2 checkpoint in `dir`, built now if it is missing or stale; empty
// when there is none (disabled, or it could not be written) and the caller loads `dir`'s own
// model.safetensors. Serialised: the two ranks of a split draft load in turn, and the second
// finds the first's file.
std::string dflash2_nvfp4_cache(const std::string& dir) {
    static std::mutex mu;
    static std::vector<std::string> rebuilt;   // =rebuild rebuilds once per process and directory
    std::lock_guard<std::mutex> lock(mu);
    const char* e = getenv("SPARKINFER_DFLASH2_NVFP4_CACHE");
    if (e && e[0] == '0') return {};
    const bool rebuild = e && strcmp(e, "rebuild") == 0 &&
                         std::find(rebuilt.begin(), rebuilt.end(), dir) == rebuilt.end();
    const std::string src = dir + "/model.safetensors";
    const std::string key = nv_cache_key(src);
    if (key.empty()) return {};
    std::vector<std::string> cand{dir + "/sparkinfer-nvfp4.safetensors"};
    {
        const char* xdg = getenv("XDG_CACHE_HOME");
        const char* home = getenv("HOME");
        const std::string base = xdg && xdg[0] ? std::string(xdg)
                               : (home && home[0] ? std::string(home) + "/.cache" : std::string());
        if (!base.empty()) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%016zx", std::hash<std::string>{}(dir));
            cand.push_back(base + "/sparkinfer/dflash2-nvfp4-" + buf + ".safetensors");
        }
    }
    if (!rebuild)
        for (const std::string& c : cand)
            if (nv_cache_valid(c, key)) return c;
    SafeTensorsFile st;
    if (!st.load(src)) return {};
    for (size_t i = 0; i < cand.size(); i++) {
        const std::string& c = cand[i];
        const std::string cdir = c.substr(0, c.rfind('/'));
        if (i > 0) {
            mkdir(cdir.substr(0, cdir.rfind('/')).c_str(), 0755);
            mkdir(cdir.c_str(), 0755);
        }
        if (access(cdir.c_str(), W_OK) != 0) continue;
        fprintf(stderr, "[dflash] DFlash2: quantizing the projections to NVFP4 (once) into %s\n",
                c.c_str());
        if (nv_cache_write(st, key, c)) {
            if (rebuild) rebuilt.push_back(dir);
            return c;
        }
        fprintf(stderr, "[dflash] DFlash2: could not write %s\n", c.c_str());
    }
    return {};
}

struct Q8W { signed char* q = nullptr; float* s = nullptr;
              unsigned char* q4 = nullptr; void* dm = nullptr; };

struct LayerWeights {
    bf16 *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    // Q8_0 mirrors of the four batched projections (Q/K/V, O, gate/up, down).
    Q8W q8_wq, q8_wk, q8_wv, q8_wo, q8_gate, q8_up, q8_down;
    // NVFP4 payloads (SI_QTYPE_NVFP4, see Impl::nv_on) of the same projections, or null.
    void *nv_wq = nullptr, *nv_wk = nullptr, *nv_wv = nullptr, *nv_wo = nullptr,
         *nv_gate = nullptr, *nv_up = nullptr, *nv_down = nullptr;
    bf16 *q_norm = nullptr, *k_norm = nullptr;
    bf16 *input_norm = nullptr, *post_norm = nullptr;
    bf16 *gate = nullptr, *up = nullptr, *down = nullptr;
};

// Draft projection weight format: 4 = asymmetric int4, 8 = Q8_0, 0 = bf16. Default 4.
inline int draft_w_bits() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_DFLASH_WBITS");
        v = e ? atoi(e) : 4;
        if (v != 0 && v != 4 && v != 8) v = 4;
    }
    return v;
}
inline bool q8_on() { return draft_w_bits() != 0; }

// NVFP4 draft projections (SPARKINFER_DFLASH_NVFP4, default on): see Impl::nv_on.
inline bool draft_nvfp4() {
    static const bool v = [] { const char* e = getenv("SPARKINFER_DFLASH_NVFP4");
                               return !(e && e[0] == '0'); }();
    return v;
}

} // namespace

struct DFlashDraftModel::Impl {
    DraftWalk walk;   // DFlash2: the selector walk of the next block (set_walk; greedy after it)
    DFlashDraftConfig cfg;
    int device = 0;               // the card this instance allocated on (set at construction)
    // (dual-GPU) Split draft. On the rank-0 instance: the rank-1 slice and its device, run beside
    // every forward / KV-state call. On both: the target's stream on this card (the draft runs on
    // it, see tp_attach). tp_split: this instance computes half of each layer and sums through the
    // link (true on both ranks once attached).
    DFlashDraftModel* tp_peer = nullptr;
    int tp_peer_dev = -1;
    cudaStream_t tp_stream = nullptr;
    bool tp_split = false;
    std::vector<LayerWeights> layers;
    bf16* fc = nullptr;           // [H, fc_k] as [out, in] for gemv
    // (dual-GPU) fc split by input columns. The target captures, on each card, its own half of
    // every tapped layer's hidden state (columns [r * H/2, (r+1) * H/2) of each layer, see
    // Qwen35Model::set_dflash_capture_split), so a rank's capture rows are [rows, fc_k] and fc's
    // matching columns live on that rank: the two partial projections sum to the whole one
    // through the all-reduce that used to carry rank 0's result to rank 1. fc_k = n_cap * H
    // unsplit. peer_hidden maps a rank-0 capture pointer to rank 1's twin (set by the target).
    int fc_k = 0;
    bool fc_split = false;
    std::function<const void*(const void*)> peer_hidden;

    // NVFP4 projections (SPARKINFER_DFLASH_NVFP4, default on for a safetensors draft). Every
    // backbone projection and fc is one SI_QTYPE_NVFP4 payload (kernels/qtype.h): the checkpoint's
    // own NVFP4 tensors (o/gate/up/down on DSpark) as they are stored, its bf16 ones (q/k/v, fc)
    // quantized once at load. No bf16 or Q4 copy is kept. Up to kNvGemvRows rows run the dp4a
    // GEMV against the same Q8_1 activations the Q4 path used (exact integer dots per 16-weight
    // group, the weights as stored); wider calls (a prompt's context) dequantize the weight to
    // bf16 in slices for the tensor-core GEMM. Activations are never cut to 4 bits.
    bool nv_on = false;
    void* nv_fc = nullptr;
    char* nv_q81 = nullptr;   // Q8_1 staging for up to kNvGemvRows rows of nv_kmax
    int nv_kmax = 0;
    static constexpr int kNvGemvRows = 64;
    // YaRN rotary table (null unless the checkpoint configures rope_type "yarn").
    float* d_yarn_inv_freq = nullptr;   // [head_dim/2]
    float  yarn_att_scale = 1.0f;
    bf16* hidden_norm = nullptr;
    bf16* final_norm = nullptr;
    // Quantized copy of `fc`. The projector is [hidden, n_cap*hidden] = [5120, 25600] on DSpark,
    // 262 MB of bf16 -- and it was the ONE draft matrix still read at full precision, once per
    // block, to project 1-5 context rows. Every other projection has had a Q4 copy since #661.
    Q8W q8_fc;
    std::vector<void*> owned;

    // Shared target pointers
    const void* embed = nullptr;
    // (dual-GPU) Vocab-split embedding: `embed` holds only rows [0, embed_rows) (the tp=2 leader's
    // half); rows at or above it live in `embed_hi` on device `embed_hi_dev`, indexed from
    // embed_rows. 0 = `embed` is the whole table (tp=1). Rows fetched from the peer are cached
    // here -- the mask token is in every block and sits in the upper half on Qwen3.8.
    int embed_rows = 0;
    const void* embed_hi = nullptr;
    int embed_hi_dev = -1;
    std::unordered_map<int, bf16*> embed_hi_cache;
    const void* lm_head = nullptr;
    int lm_head_type = 0;
    signed char* head_nv_xq = nullptr;   // NVFP4-payload head (kLmHeadNvfp4Type): int8 rows
    float* head_nv_xs = nullptr;         // ...and their per-16 scales
    int vocab = 0;
    int hidden = 0;
    bf16* lm_head_bf16 = nullptr;  // eager dequant+transpose cache for the batched LM-head GEMM
    signed char* lm_head_i8 = nullptr;
    float* lm_head_i8_scale = nullptr;
    unsigned char* lm_head_i4 = nullptr;
    float* lm_head_i4_scale = nullptr;

    // DSpark's Markov head (optional -- absent for plain DFlash drafts, e.g. Qwen3.6-35B-A3B's).
    // markov_w1 [vocab, markov_rank]: embedding table indexed by the previous token id.
    // markov_w2 [vocab, markov_rank]: projection back to vocab space, added to the base logits.
    // Shared vocab with the target (this draft has no embed_tokens of its own either), so no
    // verifier/draft vocab remapping is needed.
    bf16* markov_w1 = nullptr;
    bf16* markov_w2 = nullptr;
    // int8 copy of markov_w2 with per-32 scales, built once at load; the bf16 original is released
    // as soon as it exists. See launch_markov_bias_add_q8 for why this is about L2, not DRAM.
    signed char* markov_w2_q = nullptr;
    float* markov_w2_s = nullptr;
    int markov_rank = 0;

    // DSpark's confidence head (AcceptRatePredictor): predicts a per-position accept
    // probability from concat(hidden, markov_latent). Optional -- requires the Markov head
    // (confidence_head_with_markov=True for every DSpark checkpoint released so far).
    bf16* confidence_w = nullptr;   // [hidden + markov_rank]
    float confidence_bias = 0.f;    // scalar, read back to host once at load (cheap, load-time only)
    // [block_size+1][markov_rank]. It USED to be a single [markov_rank] scratch overwritten by
    // every step of the sequential Markov loop, which forced the confidence head to run inside
    // that loop -- four separate grid-of-ONE launches, 8.5 us each, for a dot product. With a
    // per-row stride the four become one grid-4 launch after the chain. The chain itself is
    // unaffected: nothing in it ever read the latent back.
    float* markov_latent = nullptr;

    // DFlash2 (cfg.dflash2). Per layer, the attention's and the MLP's grouped convolution:
    // base [2, taps, H] and kernel_projection [2 * taps * H / group, H], bf16, whole on every rank
    // (the convolution runs on the full hidden state, which both ranks hold).
    struct D2Conv { bf16* base = nullptr; bf16* proj = nullptr; void* proj_nv = nullptr; };
    std::vector<D2Conv> d2_attn, d2_mlp;
    // Candidate selector (rank 0): hidden_projection [rank, H], codebooks [vocab, rank].
    bf16* d2_hproj = nullptr;
    bf16* d2_pred = nullptr;
    bf16* d2_succ = nullptr;
    void* d2_host_cb = nullptr;   // the two codebooks in pinned, mapped host memory (or null)
    bf16* d2_coef = nullptr;     // [B, 2 * taps * H / group]: the row's conv coefficients
    bf16* d2_cv = nullptr;       // [B, H]: a convolution's output
    bf16* d2_hp = nullptr;       // [B, rank]: the selector's hidden projection
    float* d2_q_p = nullptr;     // [B, kDraftQTab]: the sampled walk's proposal distribution
    int* d2_q_i = nullptr;
    float* d2_tk_v = nullptr;    // [B, kRowsTopkMax]: candidates per proposal row, best first
    int* d2_tk_i = nullptr;
    float* d2_h_tk_v = nullptr;  // pinned twins, for the cross-card merge
    int* d2_h_tk_i = nullptr;
    // Rank 1 of a split DFlash2 draft: its card's half of the target head (set_peer_head), the
    // last d2_head_rows rows of the vocabulary.
    std::function<DFlashDraftModel::HeadRef()> d2_head_src;
    int d2_head_rows = 0;

    // Scratch
    cudaStream_t stream{};
    bf16 *noise = nullptr;          // [B, H]
    bf16 *target_proj = nullptr;    // [ctx, H]
    bf16 *x = nullptr, *xn = nullptr, *h = nullptr, *hn = nullptr;
    bf16 *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr, *ao = nullptr;
    bf16 *gate = nullptr, *up = nullptr, *down = nullptr;
    // Per-split online-softmax partials for the row-batched KV-split draft attention.
    float *fa_m = nullptr, *fa_l = nullptr, *fa_acc = nullptr;
    float* logits = nullptr;        // [B, vocab]
    void* head_q8 = nullptr;        // [B] Q8_1 rows of xn for the multi-row head MMVQ
    int *d_ids = nullptr, *d_out = nullptr;
    int *h_ids = nullptr;                        // PINNED staging for the block ids
    // Q8_1 staging for the dp4a backbone: one 36-byte block per 32 values per block row, sized for
    // the widest K any projection uses (the FFN's intermediate).
    void* xq81 = nullptr;

    int *h_out = nullptr;
    float *d_confidence = nullptr, *h_confidence = nullptr;   // [B], confidence head output

    // Per-layer contiguous KV cache: [max_seq, n_kv, d]
    std::vector<bf16*> k_cache, v_cache;
    int seq_len = 0;
    // First position the cache holds: > 0 when the context was ingested from a truncated capture
    // (target_hidden_start > 0). The full-attention layer must then stay windowed for the rest of
    // the generation -- positions below it were never computed.
    int ctx_lo = 0;
    // Position of cache row 0. > 0 once the cache has slid (forward_block): row i holds position
    // kv_base + i, and positions below kv_base are gone (ctx_lo >= kv_base then).
    int kv_base = 0;
    // Lowest position held in every layer: rows a block skipped (outside every window) or that
    // slid out are not (kv_snapshot).
    int kv_valid_lo = 0;
    // The draft's active block width for a proposal depth (see forward_block).
    int width_for(int kProposalDepth) const {
        const Impl& s = *this;
        const auto& c = cfg;
        static const int env_w = []{
            const char* e = getenv("SPARKINFER_DFLASH_BLOCK_WIDTH");
            return e ? atoi(e) : 0;
        }();
        int v = env_w;
        if (v <= 0) {
            v = kProposalDepth + 1;
            // DSpark's full-attention block is bidirectional: even when the verifier only asks
            // for proposal 1, later mask rows are trained context for the base-logit row that
            // produces it. The two-row tier removed too much of that context. Four rows stay on
            // the same fast batched-GEMV tier and measured 106.70 -> 111.57 tok/s (tau 1.561 ->
            // 1.662) on Qwen3.8-27B, while the full seven-row fallback costs 9.27 ms. Plain
            // DFlash checkpoints have no Markov head and retain the old depth+1 choice.
            if (s.markov_w1 && c.block_size == 7 && v < 4) v = 4;
        }
        if (v < kProposalDepth + 1) v = kProposalDepth + 1;
        // Round up to a width the batched-GEMV path is instantiated for; anything else falls
        // back to the per-token GEMV loop, which costs far more than the rows it saves.
        //
        // The rounding targets {2,4,8,16} and is then CLAMPED to block_size, which is 7 on the
        // released DSpark checkpoints -- not a power of two. So every depth from 4 up asked for a
        // width of 7, nothing was instantiated for 7, and the draft fell to the per-token loop:
        // 2.5 ms -> 9.9 ms. Widths 5/6/7 are instantiated now (see dflash_kernels.cu), so the
        // clamp lands on a batched width. This matters most at ctx >= SPARKINFER_DFLASH_DEEP_MIN_SEQ,
        // where the ladder in qwen35.cpp selects depth 7 and therefore width 7 on every block.
        // DO NOT narrow this to depth+1. Tried it: at proposal depth 4 it takes the block from
        // 7 rows to 5 and saves ~0.13 ms of draft, and it LOSES more than that in acceptance --
        // 125.0920 tok/s at tau 1.8824 with the rounding below, against 124.2785 / 1.8551 with an
        // exact width, measured on one binary with the arms alternated. The block's attention is
        // bidirectional, so mask rows the verifier never reads are still trained CONTEXT for the
        // rows it does read; the comment above about the two-row tier is the same effect at the
        // other end, and the ceiling is not free either.
        const int w = v <= 2 ? 2 : (v <= 4 ? 4 : (v <= 8 ? 8 : 16));
        return w > c.block_size ? c.block_size : w;
    }
    // (dual-GPU) forward_blocks' row scratch, sized for kMultiRows rows on first use.
    static constexpr int kMultiRows = 32;
    bool m_ready = false, m_failed = false;
    bf16 *m_noise = nullptr, *m_x = nullptr, *m_xn = nullptr, *m_h = nullptr, *m_hn = nullptr,
         *m_q = nullptr, *m_attn = nullptr, *m_ao = nullptr, *m_gate = nullptr, *m_up = nullptr,
         *m_down = nullptr, *m_knew = nullptr, *m_vnew = nullptr, *m_kctx = nullptr,
         *m_vctx = nullptr, *m_th = nullptr, *m_tp = nullptr;
    float* m_logits = nullptr;
    char *m_head_q8 = nullptr, *m_xq81 = nullptr;
    int *m_d_ids = nullptr, *m_d_out = nullptr, *m_h_ids = nullptr, *m_h_out = nullptr;

    // (B6) Draft head as one NVFP4 tensor-core GEMM over the target's FP4 head copy (rows
    // 0..Vd-1: a whole-atom prefix of its row-major data and N-outer SFB scales).
    const void* hf_w = nullptr;
    const void* hf_sf = nullptr;
    float hf_alpha = 1.f;
    void *hf_a = nullptr, *hf_as = nullptr, *hf_ws = nullptr;
    bf16* hf_x = nullptr;   // [kMultiRows][H] gathered rows (multi path)
    int hf_state = 0;       // 0 untried, 1 ready, -1 off
    // logits rows [0, m) (and garbage rows up to m rounded to 8, which `out` must hold) from the
    // m contiguous bf16 rows at x. False: not taken, nothing launched.
    bool head_fp4(const bf16* x, int m, float* out, int Vd, cudaStream_t st) {
        static const bool env = [] {
            const char* e = getenv("SPARKINFER_DFLASH_HEAD_FP4"); return !(e && e[0] == '0');
        }();
        if (!env || m <= 0 || m > kMultiRows || !head_fp4_ready(Vd)) return false;
        const int H = cfg.hidden, mp = (m + 7) & ~7;
        return kernels::launch_prefill_nvfp4_quant_a(x, hf_a, hf_as, mp, H, st) &&
               kernels::launch_prefill_nvfp4_gemm_f32(hf_a, hf_as, hf_w, hf_sf, out, mp, Vd, H,
                                                      hf_ws, st, hf_alpha);
    }
    bool head_fp4_ready(int Vd) {
        if (!hf_w || !hf_sf || Vd % 128) return false;
        const int H = cfg.hidden;
        if (hf_state == 0) {
            const size_t ws = kernels::prefill_nvfp4_workspace_bytes_f32(kMultiRows, Vd, H);
            bool ok = kernels::prefill_nvfp4_supported(kMultiRows, Vd, H) &&
                      cudaMalloc(&hf_a, kernels::prefill_nvfp4_data_bytes(kMultiRows, H)) == cudaSuccess &&
                      cudaMalloc(&hf_as, kernels::prefill_nvfp4_scale_bytes_a(kMultiRows, H)) == cudaSuccess &&
                      cudaMalloc(&hf_ws, ws ? ws : 16) == cudaSuccess &&
                      cudaMalloc(&hf_x, (size_t)kMultiRows * H * sizeof(bf16)) == cudaSuccess;
            if (!ok) cudaGetLastError();
            hf_state = ok ? 1 : -1;
        }
        return hf_state == 1;
    }

    bool multi_alloc() {
        if (m_ready || m_failed) return m_ready;
        const int R = kMultiRows;
        const int H = cfg.hidden, I = cfg.intermediate;
        const int qdim = cfg.n_q_heads * cfg.head_dim, kvdim = cfg.n_kv_heads * cfg.head_dim;
        const int n_cap = (int)cfg.target_layer_ids.size();
        const int V = cfg.tp_rank == 0 ? std::max(cfg.vocab, 1) : 1;   // the head runs on rank 0
        int kmax = std::max(I, H);
        kmax = std::max(kmax, n_cap * H);
        kmax = std::max(kmax, qdim);
        std::vector<void*> got;
        bool ok = true;
        auto a = [&](size_t bytes) -> void* {
            void* p = nullptr;
            if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
            got.push_back(p);
            return p;
        };
        const size_t rh = (size_t)R * H * sizeof(bf16);
        m_noise = (bf16*)a(rh); m_x = (bf16*)a(rh); m_xn = (bf16*)a(rh); m_h = (bf16*)a(rh);
        m_hn = (bf16*)a(rh); m_ao = (bf16*)a(rh); m_down = (bf16*)a(rh); m_tp = (bf16*)a(rh);
        m_q = (bf16*)a((size_t)R * qdim * sizeof(bf16));
        m_attn = (bf16*)a((size_t)R * qdim * sizeof(bf16));
        m_gate = (bf16*)a((size_t)R * I * sizeof(bf16));
        m_up = (bf16*)a((size_t)R * I * sizeof(bf16));
        m_knew = (bf16*)a((size_t)R * kvdim * sizeof(bf16));
        m_vnew = (bf16*)a((size_t)R * kvdim * sizeof(bf16));
        m_kctx = (bf16*)a((size_t)R * kvdim * sizeof(bf16));
        m_vctx = (bf16*)a((size_t)R * kvdim * sizeof(bf16));
        m_th = (bf16*)a((size_t)R * n_cap * H * sizeof(bf16));
        m_logits = (float*)a((size_t)R * V * sizeof(float));
        m_head_q8 = (char*)a((size_t)R * kernels::llama_q8_1_bytes(H));
        m_xq81 = (char*)a((size_t)R * ((kmax + 31) / 32) * 36);
        m_d_ids = (int*)a((size_t)2 * R * sizeof(int));
        m_d_out = (int*)a((size_t)2 * R * sizeof(int));
        if (ok && cudaHostAlloc(&m_h_ids, (size_t)2 * R * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
            m_h_ids = nullptr; ok = false;
        }
        if (ok && cudaHostAlloc(&m_h_out, (size_t)2 * R * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
            m_h_out = nullptr; ok = false;
        }
        if (!ok) {
            cudaGetLastError();
            for (void* p : got) cudaFree(p);
            if (m_h_ids) cudaFreeHost(m_h_ids);
            m_h_ids = nullptr;
            m_failed = true;
            return false;
        }
        for (void* p : got) owned.push_back(p);
        m_ready = true;
        return true;
    }
    int kv_cap = 0;   // positions k_cache/v_cache hold (cfg.max_seq for the built-in cache)
    // (dual-GPU) Extra KV caches, one per concurrently drafted session (kv_state_*). The active
    // cache is always the one in k_cache/v_cache/seq_len/kv_cap; a state's slot holds it while
    // it is not selected. Slot -1 is the built-in cache.
    struct KvState { std::vector<bf16*> k, v; int seq_len = 0; int cap = 0; int ctx_lo = 0; int base = 0; int valid_lo = 0; bool live = false; };
    std::vector<KvState> kv_states;
    KvState kv_default;
    int kv_cur = -1;
    // Make state `id` (-1 = the built-in cache) the one k_cache/v_cache/seq_len/kv_cap describe.
    // This rank only (DFlashDraftModel::kv_state_select also selects on the peer).
    bool select(int id) {
        if (id == kv_cur) return true;
        if (id >= 0 && (id >= (int)kv_states.size() || !kv_states[id].live)) return false;
        KvState& out = kv_cur < 0 ? kv_default : kv_states[kv_cur];
        out.k.swap(k_cache);
        out.v.swap(v_cache);
        out.seq_len = seq_len;
        out.ctx_lo = ctx_lo;
        out.cap = kv_cap;
        out.base = kv_base;
        out.valid_lo = kv_valid_lo;
        KvState& in = id < 0 ? kv_default : kv_states[id];
        k_cache.swap(in.k);
        v_cache.swap(in.v);
        seq_len = in.seq_len;
        ctx_lo = in.ctx_lo;
        kv_cap = in.cap;
        kv_base = in.base;
        kv_valid_lo = in.valid_lo;
        kv_cur = id;
        return true;
    }
    // Whether forward_blocks can take the batched path for these sessions (steady-state blocks on
    // the dp4a/Q4 tiers, unwindowed, within kMultiRows), filling each session's cache length and
    // context-row offset. Does not allocate (see multi_alloc).
    bool multi_plan(int n, const DFlashDraftModel::DraftSeg* seg, int proposals,
                    std::vector<int>& past, std::vector<int>& ctx_off, int& ctx_total) {
        static const bool multi_env = [] {
            const char* e = getenv("SPARKINFER_DFLASH_MULTI"); return !(e && e[0] == '0');
        }();
        static const int kDp4a = []{ const char* e = getenv("SPARKINFER_DFLASH_DP4A");
                                     return e ? atoi(e) : 15; }();
        static const int kRowShift = []{
            const char* e = getenv("SPARKINFER_DFLASH_ROW_SHIFT"); return (e && e[0] == '0') ? 0 : 1;
        }();
        static const int kFullWindowEnv = []{ const char* e = getenv("SPARKINFER_DFLASH_FULL_WINDOW");
                                              return e ? atoi(e) : -1; }();
        const auto& c = cfg;
        const int depth = std::min(c.block_size, proposals > 0 ? std::min(proposals, 15) : 5);
        const int BW = width_for(depth);
        const int R = kMultiRows;
        const int rows = n * BW;
        const bool fast_w = (BW == 16 || BW == 8 || BW == 7 || BW == 6 || BW == 5 || BW == 4 || BW == 2);
        bool ok = multi_env && n >= 2 && rows <= R && n * depth <= R && fast_w && kDp4a == 15 &&
                  kRowShift == 1 && xq81 && (nv_on ? nv_fc != nullptr : q8_fc.q4 != nullptr) &&
                  head_q8 && lm_head_type == 12 &&
                  kFullWindowEnv < 0 && 3 * c.sliding_window <= 0 && !lm_head_i4 &&
                  !lm_head_i8 && markov_w1 && markov_w2_q;
        for (int L = 0; ok && L < c.n_layers; L++) {
            const auto& w = layers[L];
            ok = nv_on ? (w.nv_wq && w.nv_wk && w.nv_wv && w.nv_wo && w.nv_gate && w.nv_up &&
                          w.nv_down)
                       : (w.q8_wq.q4 && w.q8_wk.q4 && w.q8_wv.q4 && w.q8_wo.q4 && w.q8_gate.q4 &&
                          w.q8_up.q4 && w.q8_down.q4);
        }
        ctx_total = 0;
        for (int j = 0; ok && j < n; j++) {
            const DFlashDraftModel::DraftSeg& g = seg[j];
            ok = g.state >= 0 && g.state < (int)kv_states.size() && kv_states[g.state].live &&
                 g.ctx_len >= 1 && g.ctx_len <= 8 && g.target_hidden_start == 0 && g.target_hidden &&
                 g.ids && g.out_argmax;
            if (!ok) break;
            const bool cur = g.state == kv_cur;
            past[j] = cur ? seq_len : kv_states[g.state].seq_len;
            ctx_off[j] = ctx_total;
            ctx_total += g.ctx_len;
            // Unwindowed contexts only: past 12288, or after a truncated capture (ctx_lo), the full
            // layer attends a window, which forward_block applies and this pass does not.
            const int lo = cur ? ctx_lo : kv_states[g.state].ctx_lo;
            const int cap = cur ? kv_cap : kv_states[g.state].cap;
            const int base = cur ? kv_base : kv_states[g.state].base;
            // A state that slid (base > 0) or would slide in this block goes through forward_block.
            ok = past[j] + g.ctx_len == g.pos0 && past[j] + g.ctx_len + BW <= cap &&
                 past[j] + g.ctx_len < 12288 && lo == 0 && base == 0;
        }
        return ok && ctx_total <= R;
    }

    // The draft's quantized weight copies are built on first use, not at load. Constructing them
    // is what makes merely loading the draft tax the TARGET's decode: measured on RTX 5090 with
    // the same bench_decode call either side of draft.load(), 501.9 -> 460.1 tok/s at 512-ctx, and
    // skipping just this construction removes all of it (501.2). Freeing the draft afterwards
    // restores it too (501.4), so the cost tracks these buffers being resident rather than the
    // ~216 MB they occupy -- 2.5 GB of dummy allocations reproduce none of it.
    //
    // Deferring means a generation that never runs the draft never pays. dflash_generate primes
    // this before its decode clock starts, so a generation that does run the draft is unchanged.
    struct PendingQuant { bf16* w; int N, K; Q8W* dst; };
    std::vector<PendingQuant> pending_quant;
    bool quant_ready = false;

    bool quant_failed = false;   // a quantized copy could not be allocated (device memory)
    void quant_pending() {
        for (auto& pq : pending_quant) {
            *pq.dst = make_q8(pq.w, pq.N, pq.K);
            if (!(pq.dst->q4 && pq.dst->dm) && !(pq.dst->q && pq.dst->s)) quant_failed = true;
        }
        pending_quant.clear();
    }
    void ensure_quant() {
        if (quant_ready) return;
        quant_pending();
        quant_ready = true;
        release_unused_bf16();
    }

    // (dual-GPU C3) Once every quantized mirror exists, the bf16 q/o/gate/up/down copies are dead
    // weight: the block projections read them only on the per-token fallback (a block width the
    // batched kernels are not instantiated for, or a missing mirror), and the default width always
    // rounds onto an instantiated one (see BW in draft_step). k/v (the wide first block's context
    // GEMM reads them) and fc stay. On DSpark that is ~3.3 GB of a 16 GB card -- at tp=2 the draft
    // shares card 0 with half the target, and holding them starved the prefill arena and the
    // draft's own capture buffer. Not when SPARKINFER_DFLASH_BLOCK_WIDTH forces a width (the
    // fallback could then run), nor with SPARKINFER_DFLASH_KEEP_BF16=1 (A/B).
    size_t bf16_freed = 0;   // bf16 bytes released so far (cfg.eager_quant releases per layer)
    void release_unused_bf16(bool report = true) {
        static const bool keep = [] {
            const char* e = getenv("SPARKINFER_DFLASH_KEEP_BF16");
            const char* w = getenv("SPARKINFER_DFLASH_BLOCK_WIDTH");
            return (e && e[0] == '1') || (w && atoi(w) > 0);
        }();
        if (keep || quant_failed || !q8_on()) return;
        auto has = [](const Q8W& q) { return (q.q4 && q.dm) || (q.q && q.s); };
        size_t freed = 0;
        const size_t qd = (size_t)cfg.n_q_heads * cfg.head_dim, H = cfg.hidden,
                     I = cfg.intermediate;
        for (auto& lw : layers) {
            if (!lw.wq || !(has(lw.q8_wq) && has(lw.q8_wo) && has(lw.q8_gate) && has(lw.q8_up) &&
                            has(lw.q8_down)))
                continue;
            release(lw.wq); lw.wq = nullptr;
            release(lw.wo); lw.wo = nullptr;
            release(lw.gate); lw.gate = nullptr;
            release(lw.up); lw.up = nullptr;
            release(lw.down); lw.down = nullptr;
            freed += (2 * qd * H + 3 * I * H) * sizeof(bf16);
        }
        bf16_freed += freed;
        if (report && bf16_freed)
            fprintf(stderr, "[dflash] released the bf16 q/o/gate/up/down copies (%.2f GB): the "
                            "quantized mirrors serve every block width in use\n", bf16_freed / 1e9);
    }

    // A copy that could not be allocated comes back empty (the caller then reports quant_failed)
    // and its kernel is not launched: quantizing into a null buffer was an illegal address, which
    // lost the context on a card too full for the draft -- and so the whole server, where the
    // load check would have declined speculative decoding cleanly a moment later.
    Q8W make_q8(bf16* w, int N, int K) {
        Q8W o;
        if (draft_w_bits() == 4) {
            o.q4 = alloc<unsigned char>((size_t)N * (K / 2));
            o.dm = alloc<short>((size_t)N * (K / 32) * 2);   // __half2 per 32-weight block
            if (!(o.q4 && o.dm)) {
                release(o.q4); release(o.dm);
                cudaGetLastError();
                return Q8W{};
            }
            dflash_kernels::launch_quantize_w_q4(w, o.q4, o.dm, N, K, stream);
        } else {
            o.q = alloc<signed char>((size_t)N * K);
            o.s = alloc<float>((size_t)N * (K / 32));
            if (!(o.q && o.s)) {
                release(o.q); release(o.s);
                cudaGetLastError();
                return Q8W{};
            }
            dflash_kernels::launch_quantize_w_q8(w, o.q, o.s, N, K, stream);
        }
        cudaStreamSynchronize(stream);
        return o;
    }

    // Give a buffer back and drop it from `owned`, so the destructor does not double-free it.
    void release(void* p) {
        if (!p) return;
        for (size_t i = 0; i < owned.size(); i++) {
            if (owned[i] == p) { owned.erase(owned.begin() + i); break; }
        }
        cudaFree(p);
    }

    template <class T> T* alloc(size_t n) {
        void* p = nullptr;
        cu(cudaMalloc(&p, n * sizeof(T)), "malloc");
        owned.push_back(p);
        return (T*)p;
    }

    // Scratch/KV-cache allocation shared by load() (safetensors) and load_gguf(): identical
    // buffers either way, sized purely from cfg (which is fully populated by the time either
    // caller reaches this). Factored out so the two load paths cannot drift on sizing.
    void alloc_scratch() {
        const int H = cfg.hidden;
        const int I = cfg.intermediate;
        const int B = cfg.block_size;
        const int max_ctx = cfg.max_seq;
        const int qdim = cfg.n_q_heads * cfg.head_dim;
        const int kvdim = cfg.n_kv_heads * cfg.head_dim;
        noise = alloc<bf16>((size_t)B * H);
        target_proj = alloc<bf16>((size_t)max_ctx * H);
        x = alloc<bf16>((size_t)B * H);
        xn = alloc<bf16>((size_t)B * H);
        h = alloc<bf16>((size_t)B * H);
        hn = alloc<bf16>((size_t)B * H);
        q = alloc<bf16>((size_t)B * qdim);
        attn = alloc<bf16>((size_t)B * qdim);
        ao = alloc<bf16>((size_t)B * H);
        {
            const size_t parts = (size_t)dflash_kernels::kDFlashAttnMaxRows * cfg.n_q_heads *
                                 dflash_kernels::kDFlashAttnMaxSplits;
            fa_m = alloc<float>(parts);
            fa_l = alloc<float>(parts);
            fa_acc = alloc<float>(parts * 128);
        }
        gate = alloc<bf16>((size_t)B * I);
        up = alloc<bf16>((size_t)B * I);
        down = alloc<bf16>((size_t)B * H);
        // DFlash2 scores its candidates on both cards (see d2_head).
        logits = alloc<float>((size_t)B * (cfg.tp_rank == 0 || cfg.dflash2 ? std::max(cfg.vocab, 1) : 1));
        head_q8 = alloc<char>((size_t)B * kernels::llama_q8_1_bytes(H));
        d_ids = alloc<int>(B);
        // Proposal-indexed, not row-indexed. Under the row-shift mapping the chain writes
        // d_out[r] / d_confidence[r] for r = 1..kProposalDepth, and a block_size-wide block
        // backs kProposalDepth == B -- so index B is live and these need B+1 slots. d_ids stays
        // at B: it holds the block's input ids, one per row.
        d_out = alloc<int>(B + 1);
        cu(cudaHostAlloc(&h_out, (B + 1) * sizeof(int), cudaHostAllocDefault), "h_out");
        // The block ids arrive in a caller-owned std::vector, i.e. PAGEABLE memory, and CUDA
        // performs a stream synchronize before a pageable H2D copy is initiated. Every other host
        // buffer on this path (h_out, h_confidence, and the verify's ph_ids/ph_pos/ph_seq) is
        // already pinned; this one was the exception, and it sits at the very first instruction of
        // the draft block -- immediately after the verify has left a 158 us GDN state commit in
        // flight on the target's stream, which shares no data with the draft and should overlap it.
        cu(cudaHostAlloc(&h_ids, (B + 1) * sizeof(int), cudaHostAllocDefault), "h_ids");
        {
            int kmax = (cfg.intermediate > cfg.hidden ? cfg.intermediate : cfg.hidden);
            const int kfc = (int)cfg.target_layer_ids.size() * cfg.hidden;   // the fc projector
            if (kfc > kmax) kmax = kfc;
            const size_t blocks = (size_t)(B + 1) * ((kmax + 31) / 32);
            xq81 = alloc<char>(blocks * 36);
        }
        if (nv_on) {
            nv_kmax = std::max(std::max(cfg.intermediate, cfg.hidden),
                               std::max(fc_k, cfg.n_q_heads * cfg.head_dim));
            nv_q81 = alloc<char>((size_t)kNvGemvRows * (nv_kmax / 32) * 36);
        }
        if (confidence_w) {
            d_confidence = alloc<float>(B + 1);
            cu(cudaHostAlloc(&h_confidence, (B + 1) * sizeof(float), cudaHostAllocDefault),
               "h_confidence");
        }
        if (cfg.dflash2) {
            const size_t tk = (size_t)B * dflash_kernels::kRowsTopkMax;
            d2_coef = alloc<bf16>((size_t)B * 2 * cfg.conv_taps * (H / cfg.conv_group));
            d2_cv = alloc<bf16>((size_t)B * H);
            d2_tk_v = alloc<float>(tk);
            d2_tk_i = alloc<int>(tk);
            if (cfg.tp_rank == 0) d2_hp = alloc<bf16>((size_t)B * cfg.selector_rank);
            cu(cudaHostAlloc(&d2_h_tk_v, tk * sizeof(float), cudaHostAllocDefault), "d2_h_tk_v");
            cu(cudaHostAlloc(&d2_h_tk_i, tk * sizeof(int), cudaHostAllocDefault), "d2_h_tk_i");
        }

        seq_len = 0;
        // (dual-GPU) A split draft serves the group path, which drafts every session on its own
        // kv_state; the built-in cache (max_seq positions, 168 MB per card for DSpark at 16k) is
        // then only taken if a forward actually runs on it (ensure_builtin_kv).
        if (cfg.tp_size > 1) {
            kv_cap = 0;
            return;
        }
        k_cache.resize(cfg.n_layers);
        v_cache.resize(cfg.n_layers);
        for (int L = 0; L < cfg.n_layers; L++) {
            k_cache[L] = alloc<bf16>((size_t)cfg.max_seq * kvdim);
            v_cache[L] = alloc<bf16>((size_t)cfg.max_seq * kvdim);
        }
        kv_cap = cfg.max_seq;
    }

    // The built-in cache, allocated now if it is not yet (see alloc_scratch). false: no memory.
    bool ensure_builtin_kv() {
        std::vector<bf16*>& bk = kv_cur == -1 ? k_cache : kv_default.k;
        std::vector<bf16*>& bv = kv_cur == -1 ? v_cache : kv_default.v;
        if (!bk.empty()) return true;
        const size_t bytes = (size_t)cfg.max_seq * cfg.n_kv_heads * cfg.head_dim * sizeof(bf16);
        std::vector<bf16*> k, v;
        bool ok = true;
        for (int L = 0; L < cfg.n_layers && ok; L++) {
            void* a = nullptr;
            void* b = nullptr;
            ok = cudaMalloc(&a, bytes) == cudaSuccess;
            if (ok) k.push_back((bf16*)a);
            ok = ok && cudaMalloc(&b, bytes) == cudaSuccess;
            if (ok) v.push_back((bf16*)b);
        }
        if (!ok) {
            cudaGetLastError();
            for (bf16* x : k) cudaFree(x);
            for (bf16* x : v) cudaFree(x);
            return false;
        }
        for (bf16* x : k) owned.push_back(x);
        for (bf16* x : v) owned.push_back(x);
        bk = std::move(k);
        bv = std::move(v);
        if (kv_cur == -1) kv_cap = cfg.max_seq;
        else kv_default.cap = cfg.max_seq;
        return true;
    }

    // ---- NVFP4 projections (see nv_on) ----

    // A checkpoint NVFP4 tensor (ModelOpt: packed e2m1 [rows, cols/2], e4m3 group scales
    // [rows, cols/16], weight_scale_2) as an SI_QTYPE_NVFP4 payload of this rank's slice (by_rows:
    // rows [part * rows/parts, ...), else those columns). The header holds 1 / weight_scale_2,
    // the payload's divisor. The bytes are copied as stored: nothing is re-rounded.
    void* upload_nvfp4_payload(const TensorView& w, const TensorView& sc, float ws2, bool by_rows,
                               int part, int parts, int* n_out, int* k_out) {
        const int rows = (int)w.shape[0], cols = (int)w.shape[1] * 2;
        if ((by_rows ? rows : cols) % parts != 0 || (!by_rows && (cols / parts) % 16 != 0))
            return nullptr;
        const int r0 = by_rows ? part * (rows / parts) : 0, nr = by_rows ? rows / parts : rows;
        const int c0 = by_rows ? 0 : part * (cols / parts), nc = by_rows ? cols : cols / parts;
        const size_t sbytes = (size_t)nr * (nc / 16), pbytes = (size_t)nr * (nc / 2);
        std::vector<unsigned char> host(256 + sbytes + pbytes, 0);
        const float g = ws2 > 0.f ? 1.f / ws2 : 1.f;
        std::memcpy(host.data(), &g, sizeof(float));
        const unsigned char* pw = (const unsigned char*)w.data;
        const unsigned char* ps = (const unsigned char*)sc.data;
        for (int r = 0; r < nr; r++) {
            std::memcpy(host.data() + 256 + (size_t)r * (nc / 16),
                        ps + (size_t)(r0 + r) * (cols / 16) + c0 / 16, nc / 16);
            std::memcpy(host.data() + 256 + sbytes + (size_t)r * (nc / 2),
                        pw + (size_t)(r0 + r) * (cols / 2) + c0 / 2, nc / 2);
        }
        unsigned char* d = alloc<unsigned char>(host.size());
        cu(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice), "upload nvfp4 payload");
        if (n_out) *n_out = nr;
        if (k_out) *k_out = nc;
        return d;
    }

    // A device bf16 [N, K] weight as an NVFP4 payload (round to nearest); the bf16 is released.
    void* quant_nvfp4_payload(bf16* w, int N, int K) {
        if (!w || K % 16) return nullptr;
        unsigned char* d = alloc<unsigned char>(256 + (size_t)N * K / 16 + (size_t)N * K / 2);
        dflash_kernels::launch_quantize_w_nvfp4(w, d, N, K, stream);
        cu(cudaStreamSynchronize(stream), "nvfp4 quantize");
        release(w);
        return d;
    }

    // y_i = x @ W_i^T for up to three NVFP4 payloads on one bf16 activation x [rows, K]: rows up
    // to kNvGemvRows through the dp4a GEMV (x quantized to Q8_1 here, unless `xq81` already holds
    // it), wider through nv_gemm.
    void nv_proj(const bf16* x, const void* xq81, int rows, int K, const void* W0, bf16* y0, int N0,
                 const void* W1, bf16* y1, int N1, const void* W2, bf16* y2, int N2,
                 cudaStream_t st) {
        if (rows <= 0) return;
        if (rows > kNvGemvRows) {
            nv_gemm(x, W0, y0, rows, N0, K, st);
            if (W1) nv_gemm(x, W1, y1, rows, N1, K, st);
            if (W2) nv_gemm(x, W2, y2, rows, N2, K, st);
            return;
        }
        if (!xq81) {
            kernels::launch_quantize_q8_1_rows(x, nv_q81, K, rows, K, st);
            xq81 = nv_q81;
        }
        dflash_kernels::launch_gemv_nvfp4_q81(xq81, W0, W1, W2, y0, y1, y2, N0, W1 ? N1 : 0,
                                              W2 ? N2 : 0, K, rows, st);
    }

    // The wide case: the weight dequantized to bf16 in slices of output rows (at most 32 MB each)
    // for the tensor-core GEMM, whose input stays bf16. Runs at a join, so its two scratch
    // buffers are taken and given back here.
    void nv_gemm(const bf16* x, const void* W, bf16* y, int rows, int N, int K, cudaStream_t st) {
        const char* base = static_cast<const char*>(W);
        const size_t sbytes = (size_t)N * K / 16;
        int ns = N;
        while ((size_t)ns * K * sizeof(bf16) > (32u << 20) && ns % 128 == 0 && ns > 128) ns /= 2;
        bf16* wb = nullptr;
        bf16* yt = nullptr;
        if (cudaMalloc(&wb, (size_t)ns * K * sizeof(bf16)) != cudaSuccess ||
            (ns < N && cudaMalloc(&yt, (size_t)rows * ns * sizeof(bf16)) != cudaSuccess)) {
            cudaGetLastError();
            if (wb) cudaFree(wb);
            // Not enough room for the slices: the GEMV path, slower but allocation-free.
            for (int r0 = 0; r0 < rows; r0 += kNvGemvRows)
                nv_proj(x + (size_t)r0 * K, nullptr, std::min(kNvGemvRows, rows - r0), K, W,
                        y + (size_t)r0 * N, N, nullptr, nullptr, 0, nullptr, nullptr, 0, st);
            return;
        }
        for (int n0 = 0; n0 < N; n0 += ns) {
            const int nn = std::min(ns, N - n0);
            kernels::launch_ct_dequant_nvfp4_dev(base + 256 + sbytes + (size_t)n0 * (K / 2),
                                                 base + 256 + (size_t)n0 * (K / 16),
                                                 static_cast<const float*>(W), wb, nn, K, st);
            if (nn == N) {
                kernels::launch_prefill_gemm(x, wb, y, rows, N, K, st);
            } else {
                kernels::launch_prefill_gemm(x, wb, yt, rows, nn, K, st);
                cu(cudaMemcpy2DAsync(y + n0, (size_t)N * sizeof(bf16), yt, (size_t)nn * sizeof(bf16),
                                     (size_t)nn * sizeof(bf16), rows, cudaMemcpyDeviceToDevice, st),
                   "nvfp4 gemm slice");
            }
        }
        cu(cudaStreamSynchronize(st), "nvfp4 gemm");
        cudaFree(wb);
        if (yt) cudaFree(yt);
    }

    // NVFP4 -> BF16 at load, decoded on the host.
    //
    // Deliberately NOT launch_ct_dequant_nvfp4: that kernel reads the group scale as CUTLASS
    // *unsigned* e4m3, while ModelOpt exports signed float8_e4m3fn. The bit patterns differ, so
    // the kernel mis-scales every group -- which degrades draft acceptance without breaking
    // correctness, because the target verifies every drafted token anyway.
    //
    // Validated against the BF16 source of the same checkpoint:
    //   w = e2m1(nibble) * f8e4m3(group_scale) * weight_scale_2  ->  rel_err 0.0896, ratio 0.9927
    // which is pure 4-bit rounding error. This is a one-time load cost.
    static float decode_e2m1(unsigned char n) {
        static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        float v = mag[n & 0x7];
        return (n & 0x8) ? -v : v;
    }
    static float decode_e4m3(unsigned char b) {   // signed float8_e4m3fn
        const int s = (b >> 7) & 0x1;
        const int e = (b >> 3) & 0xF;
        const int m = b & 0x7;
        float v;
        if (e == 0) v = std::ldexp((float)m / 8.0f, -6);          // subnormal
        else        v = std::ldexp(1.0f + (float)m / 8.0f, e - 7);
        return s ? -v : v;
    }

    bf16* upload_nvfp4(const TensorView& packed, const TensorView& scale, float global_scale) {
        const int rows = (int)packed.shape[0];
        const int cols = (int)packed.shape[1] * 2;
        const int groups = cols / 16;
        const unsigned char* pw = (const unsigned char*)packed.data;
        const unsigned char* ps = (const unsigned char*)scale.data;
        std::vector<bf16> host((size_t)rows * cols);
        for (int r = 0; r < rows; r++) {
            const unsigned char* prow = pw + (size_t)r * (cols / 2);
            const unsigned char* srow = ps + (size_t)r * groups;
            for (int c = 0; c < cols; c += 2) {
                const unsigned char byte = prow[c >> 1];
                const float gs = decode_e4m3(srow[c >> 4]) * global_scale;
                host[(size_t)r * cols + c]     = __float2bfloat16(decode_e2m1(byte & 0x0F) * gs);
                host[(size_t)r * cols + c + 1] = __float2bfloat16(decode_e2m1(byte >> 4) * gs);
            }
        }
        bf16* out = alloc<bf16>((size_t)rows * cols);
        cu(cudaMemcpy(out, host.data(), host.size() * sizeof(bf16), cudaMemcpyHostToDevice),
           "upload nvfp4");
        return out;
    }

    // (dual-GPU) One rank's slice of a stored [rows, cols] projection, decoded on the host exactly
    // as upload() / upload_nvfp4() decode the whole tensor: by_rows takes rows
    // [part * rows/parts, ...), otherwise columns [part * cols/parts, ...). nullptr (logged) when
    // the dimension does not divide.
    bf16* upload_part(const TensorView& w, const TensorView* sc, float gs, bool by_rows, int part,
                      int parts) {
        const bool fp4 = w.dtype == "U8";
        if (w.shape.size() != 2 || (fp4 && !sc)) return nullptr;
        const int rows = (int)w.shape[0];
        const int cols = (int)w.shape[1] * (fp4 ? 2 : 1);
        if ((by_rows ? rows : cols) % parts != 0) {
            fprintf(stderr, "[dflash] cannot split a [%d, %d] projection %d ways by %s\n", rows, cols,
                    parts, by_rows ? "rows" : "columns");
            return nullptr;
        }
        const int r0 = by_rows ? part * (rows / parts) : 0, nr = by_rows ? rows / parts : rows;
        const int c0 = by_rows ? 0 : part * (cols / parts), nc = by_rows ? cols : cols / parts;
        std::vector<bf16> host((size_t)nr * nc);
        for (int r = 0; r < nr; r++) {
            const size_t src_r = (size_t)(r0 + r);
            for (int c = 0; c < nc; c++) {
                const int col = c0 + c;
                float v;
                if (fp4) {
                    const unsigned char byte = ((const unsigned char*)w.data)[src_r * (cols / 2) + (col >> 1)];
                    const unsigned char sb = ((const unsigned char*)sc->data)[src_r * (cols / 16) + (col >> 4)];
                    v = decode_e2m1((col & 1) ? (byte >> 4) : (byte & 0x0F)) * (decode_e4m3(sb) * gs);
                    host[(size_t)r * nc + c] = __float2bfloat16(v);
                } else if (w.dtype == "F32") {
                    host[(size_t)r * nc + c] = __float2bfloat16(((const float*)w.data)[src_r * cols + col]);
                } else {
                    host[(size_t)r * nc + c] = ((const bf16*)w.data)[src_r * cols + col];
                }
            }
        }
        bf16* out = alloc<bf16>(host.size());
        cu(cudaMemcpy(out, host.data(), host.size() * sizeof(bf16), cudaMemcpyHostToDevice),
           "upload part");
        return out;
    }

    // ---- DFlash2 (cfg.dflash2) ----

    // A convolution's coefficients for `rows` rows of x (rows: an instantiated batched-GEMV
    // width) into d2_coef, and its first side on x into d2_cv.
    void d2_prepare(const D2Conv& cv, const bf16* x, int rows, int block, cudaStream_t st) {
        const int H = cfg.hidden, nc = 2 * cfg.conv_taps * (H / cfg.conv_group);
        if (cv.proj_nv)
            nv_proj(x, nullptr, rows, H, cv.proj_nv, d2_coef, nc, nullptr, nullptr, 0, nullptr,
                    nullptr, 0, st);
        else
            dflash_kernels::launch_gemv_batched16(x, cv.proj, d2_coef, nc, H, st, rows);
        dflash_kernels::launch_dflash2_conv(x, d2_coef, cv.base, d2_cv, rows, block, H,
                                            cfg.conv_group, cfg.conv_taps, 0, st);
    }
    // Its second side on x (the wrapped attention's or MLP's output) into d2_cv.
    void d2_finish(const D2Conv& cv, const bf16* x, int rows, int block, cudaStream_t st) {
        dflash_kernels::launch_dflash2_conv(x, d2_coef, cv.base, d2_cv, rows, block, cfg.hidden,
                                            cfg.conv_group, cfg.conv_taps, 1, st);
    }

    // The proposals of one block from the final hidden rows in xn (BW rows): the head's top
    // selector_top_k of rows 1..depth (each card its own vocab half, merged), then the selector's
    // walk from the block's anchor token (d_ids[0]) into out_argmax[1..depth] (rank 0).
    //
    // The walk is greedy, also for sampled requests. The verify accepts a drafted token while it
    // equals the token that row draws as ordinary decode would; that needs no proposal
    // distribution, and the proposal most likely to survive it is the most probable path. A path
    // sampled from the selector's scores (the reference's walk at temperature > 0, made for
    // rejection sampling against those scores) only lowers that.
    //
    // At tp=2 both ranks call this at the same point: it ends in a host exchange.
    bool d2_head(int BW, int depth, int* out_argmax, cudaStream_t st) {
        const int H = cfg.hidden, K = cfg.selector_top_k, TK = dflash_kernels::kRowsTopkMax;
        const bool lead = cfg.tp_rank == 0;
        if (!lead) {
            // Rank 1 scores the target head's half on its own card.
            const DFlashDraftModel::HeadRef h = d2_head_src ? d2_head_src()
                                                            : DFlashDraftModel::HeadRef{};
            lm_head = h.lm_head;
            lm_head_type = h.lm_head_type;
            vocab = d2_head_rows;
            hidden = H;
            hf_w = h.fp4_w;
            hf_sf = h.fp4_sf;
            hf_alpha = h.fp4_alpha;
        }
        const int Vh = vocab;                        // this card's head rows
        const int lo = lead ? 0 : cfg.vocab - Vh;    // the vocab id of the first
        // Rows 0..depth: the tensor-core head computes 8-row tiles anyway, and row 0 starts them.
        const int m = depth + 1;
        bool ok = lm_head && Vh > 0 && depth >= 1 && m <= BW;
        if (ok) {
            bool done = ((m + 7) & ~7) <= cfg.block_size && head_fp4(xn, m, logits, Vh, st);
            if (!done && lm_head_type == kLmHeadNvfp4Type) {
                if (!head_nv_xq) head_nv_xq = alloc<signed char>((size_t)cfg.block_size * H);
                if (!head_nv_xs) head_nv_xs = alloc<float>((size_t)cfg.block_size * (H / 16));
                if (head_nv_xq && head_nv_xs && m <= cfg.block_size) {
                    kernels::launch_gemv_nvfp4_quant_x(xn, head_nv_xq, head_nv_xs, m, H, st);
                    done = kernels::launch_gemv_nvfp4_rows_dp4a_f32(head_nv_xq, head_nv_xs, lm_head,
                                                                    logits, m, Vh, H, st);
                }
                if (!done) ok = false;
            }
            if (!done && ok && head_q8 && (lm_head_type == 12 || lm_head_type == 14)) {
                kernels::launch_quantize_q8_1_rows(xn, head_q8, H, m, H, st);
                done = lm_head_type == 12
                    ? kernels::launch_gemv_q4k_dp4a_multirow_f32(head_q8, lm_head, logits, Vh, H, m, st)
                    : kernels::launch_gemv_q6k_dp4a_multirow_f32(head_q8, lm_head, logits, Vh, H, m, st);
            }
            for (int r = 1; !done && ok && r <= depth; r++) {
                if (lm_head_type)
                    kernels::launch_gemv_q_f32(xn + (size_t)r * H, lm_head, lm_head_type,
                                               logits + (size_t)r * Vh, Vh, H, st);
                else
                    kernels::launch_gemv_f32(xn + (size_t)r * H, lm_head, logits + (size_t)r * Vh,
                                             Vh, H, st);
            }
            dflash_kernels::launch_rows_topk(logits + Vh, depth, Vh, K, d2_tk_v, d2_tk_i, st);
        }
        if (tp_split) {
            const size_t n = (size_t)depth * TK;
            if (ok) {
                cu(cudaMemcpyAsync(d2_h_tk_v, d2_tk_v, n * sizeof(float), cudaMemcpyDeviceToHost, st),
                   "dflash2 candidates");
                cu(cudaMemcpyAsync(d2_h_tk_i, d2_tk_i, n * sizeof(int), cudaMemcpyDeviceToHost, st),
                   "dflash2 candidate ids");
            }
            if (cudaStreamSynchronize(st) != cudaSuccess) ok = false;
            if (!ok)
                for (size_t i = 0; i < n; i++) { d2_h_tk_v[i] = -INFINITY; d2_h_tk_i[i] = INT_MAX; }
            std::vector<float> mv(n);
            std::vector<int> mi(n);
            tp_exchange_topk(d2_h_tk_v, d2_h_tk_i, depth, lead ? Vh : lo, mv.data(), mi.data());
            if (!lead) return true;
            ok = mi[0] != INT_MAX;
            if (!ok) return false;
            std::memcpy(d2_h_tk_v, mv.data(), n * sizeof(float));
            std::memcpy(d2_h_tk_i, mi.data(), n * sizeof(int));
            cu(cudaMemcpyAsync(d2_tk_v, d2_h_tk_v, n * sizeof(float), cudaMemcpyHostToDevice, st),
               "dflash2 merged candidates");
            cu(cudaMemcpyAsync(d2_tk_i, d2_h_tk_i, n * sizeof(int), cudaMemcpyHostToDevice, st),
               "dflash2 merged candidate ids");
        }
        if (!ok || !lead) return ok;
        const int R = cfg.selector_rank;
        dflash_kernels::launch_gemv_batched16(xn, d2_hproj, d2_hp, R, H, st, BW);
        const bool want_q = walk.temperature > 0.f && walk.q_p && walk.q_ids;
        constexpr int QT = dflash_kernels::kDraftQTab;
        if (want_q && !d2_q_p) {
            if (cudaMalloc(&d2_q_p, (size_t)kMultiRows * QT * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d2_q_i, (size_t)kMultiRows * QT * sizeof(int)) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
        }
        dflash_kernels::launch_dflash2_select(d2_tk_i, d2_tk_v, TK, d2_hp + R, R, d_ids, d2_pred,
                                              d2_succ, cfg.vocab, R, K, depth,
                                              cfg.output_multiplier, cfg.logit_softcap, d_out + 1,
                                              st, walk.temperature, walk.top_k, walk.top_p,
                                              walk.seed, walk.step0, want_q ? d2_q_p : nullptr,
                                              want_q ? d2_q_i : nullptr);
        if (want_q) {
            cu(cudaMemcpyAsync(walk.q_p, d2_q_p, (size_t)depth * QT * sizeof(float),
                               cudaMemcpyDeviceToHost, st), "dflash2 q");
            cu(cudaMemcpyAsync(walk.q_ids, d2_q_i, (size_t)depth * QT * sizeof(int),
                               cudaMemcpyDeviceToHost, st), "dflash2 q ids");
        }
        walk = DraftWalk{};
        cu(cudaMemcpyAsync(h_out + 1, d_out + 1, depth * sizeof(int), cudaMemcpyDeviceToHost, st),
           "dflash2 proposals");
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        for (int t = 1; t <= depth; t++) out_argmax[t] = h_out[t];
        return true;
    }

    bf16* upload(const TensorView& tv) {
        bf16* d = alloc<bf16>(tv.nbytes / sizeof(bf16));
        if (tv.nbytes % sizeof(bf16) == 0) {
            cu(cudaMemcpy(d, tv.data, tv.nbytes, cudaMemcpyHostToDevice), "upload bf16");
        } else {
            // F32 -> BF16
            size_t n = tv.nbytes / sizeof(float);
            std::vector<bf16> tmp(n);
            const float* src = (const float*)tv.data;
            for (size_t i = 0; i < n; i++) tmp[i] = __float2bfloat16(src[i]);
            cu(cudaMemcpy(d, tmp.data(), n * sizeof(bf16), cudaMemcpyHostToDevice), "upload f32");
        }
        return d;
    }
};

DFlashDraftModel::DFlashDraftModel(const DFlashDraftConfig& cfg) : p_(new Impl()) {
    p_->cfg = cfg;
    if (p_->cfg.sliding_layers.empty()) {
        p_->cfg.sliding_layers.assign(p_->cfg.n_layers, true);
        if (p_->cfg.n_layers > 0) p_->cfg.sliding_layers.back() = false;
    }
    // Non-blocking, for the same reason as Qwen35Model's stream: see qwen35.cpp.
    cudaStreamCreateWithFlags(&p_->stream, cudaStreamNonBlocking);
    cudaGetDevice(&p_->device);
}

DFlashDraftModel::~DFlashDraftModel() {
    if (!p_) return;
    // Free on the card the buffers came from (the rank-1 slice is destroyed from the leader's
    // thread).
    int prev = -1;
    cudaGetDevice(&prev);
    if (prev != p_->device) cudaSetDevice(p_->device);
    p_->tp_peer = nullptr;   // the peer is its own object, destroyed on its own
    if (!p_->kv_states.empty()) p_->select(-1);
    for (auto& st : p_->kv_states) {
        for (bf16* b : st.k) cudaFree(b);
        for (bf16* b : st.v) cudaFree(b);
    }
    for (void* p : p_->owned) cudaFree(p);
    if (p_->h_out) cudaFreeHost(p_->h_out);
    if (p_->h_ids) cudaFreeHost(p_->h_ids);
    if (p_->h_confidence) cudaFreeHost(p_->h_confidence);
    if (p_->d2_h_tk_v) cudaFreeHost(p_->d2_h_tk_v);
    if (p_->d2_host_cb) cudaFreeHost(p_->d2_host_cb);
    if (p_->d2_h_tk_i) cudaFreeHost(p_->d2_h_tk_i);
    if (p_->m_h_ids) cudaFreeHost(p_->m_h_ids);
    if (p_->m_h_out) cudaFreeHost(p_->m_h_out);
    if (p_->stream) cudaStreamDestroy(p_->stream);
    if (prev >= 0 && prev != p_->device) cudaSetDevice(prev);
    delete p_;
    p_ = nullptr;
}

const DFlashDraftConfig& DFlashDraftModel::config() const { return p_->cfg; }

void DFlashDraftModel::set_embed_split(int local_rows, const void* hi_table, int hi_device) {
    p_->embed_rows = (local_rows > 0 && hi_table) ? local_rows : 0;
    p_->embed_hi = p_->embed_rows ? hi_table : nullptr;
    p_->embed_hi_dev = p_->embed_rows ? hi_device : -1;
    // The cached rows came from the previous table; they are owned allocations, so only forget them.
    p_->embed_hi_cache.clear();
}

void DFlashDraftModel::set_walk(const DraftWalk& walk) { p_->walk = walk; }

void DFlashDraftModel::set_head_fp4(const void* w, const void* sf, float alpha) {
    p_->hf_w = w;
    p_->hf_sf = sf;
    p_->hf_alpha = alpha;
}

void DFlashDraftModel::set_peer_head(std::function<HeadRef()> src, int rows) {
    if (!p_->cfg.dflash2 || !p_->tp_peer) return;
    Impl& q = *p_->tp_peer->p_;
    q.d2_head_src = std::move(src);
    q.d2_head_rows = rows;
}

void DFlashDraftModel::set_shared_weights(const void* embed, const void* lm_head,
                                         int lm_head_type, int vocab, int hidden) {
    p_->embed = embed;
    p_->lm_head = lm_head;
    p_->lm_head_type = lm_head_type;
    p_->vocab = vocab;
    p_->hidden = hidden;
    // Eagerly dequantize the (static, resident) lm_head weight to bf16 exactly once, in its
    // native [vocab,hidden] ("out,in") layout -- dflash_kernels::launch_gemv_batched16_f32
    // reads W the same way a single-row GEMV does, so no relayout is needed. Lets
    // forward_block score a whole block with one batched GEMV instead of a per-token loop.
    // ONLY the BW==16 batched-head path (launch_gemv_batched16_f32 below) reads this bf16 copy;
    // every other block width already scores against the target's QUANTIZED head on-read, via
    // launch_gemv_q4k_dp4a_multirow_f32 or the per-row launch_gemv_q_f32 fallback. Materializing
    // it regardless costs vocab*hidden*2 bytes for nothing at any other block size -- 2.54 GB for
    // DSpark (V=248320, H=5120), which is precisely what pushed Qwen3.8-27B + DSpark past 32 GB
    // and made the draft OOM in its own lm_head dequant while the weights themselves fit.
    if (p_->cfg.block_size == 16 && !p_->lm_head_bf16 && vocab > 0 && hidden > 0) {
        p_->lm_head_bf16 = p_->alloc<bf16>((size_t)vocab * hidden);
        if (lm_head_type != 0) {
            kernels::launch_gguf_dequant(lm_head_type, lm_head, p_->lm_head_bf16,
                                         (long)vocab * hidden, p_->stream);
        } else {
            cu(cudaMemcpyAsync(p_->lm_head_bf16, lm_head, (size_t)vocab * hidden * sizeof(bf16),
                               cudaMemcpyDeviceToDevice, p_->stream), "lm_head copy");
        }
        cu(cudaStreamSynchronize(p_->stream), "lm_head dequant");
    }
    // The draft's head is the single largest read in the draft block, and the draft only has to
    // NOMINATE tokens -- every emitted token is still a target argmax, so the head's precision can
    // only move the accept length, never correctness. int4 halves those bytes again (~254 MB vs
    // ~508 MB at V=248k), and the kernel is at HBM peak, so its runtime is its weight bytes.
    // SPARKINFER_DFLASH_HEAD_I4=0 keeps the int8 head.
    if (!p_->lm_head_i8 && lm_head_type == 12 && vocab > 0 && hidden == 2048) {
        p_->lm_head_i8 = p_->alloc<signed char>((size_t)vocab * hidden);
        p_->lm_head_i8_scale = p_->alloc<float>(vocab);
        if (!kernels::launch_gguf_dequant_rows_i8(
                lm_head_type, lm_head, p_->lm_head_i8, p_->lm_head_i8_scale,
                vocab, hidden, p_->stream)) {
            p_->lm_head_i8 = nullptr;
            p_->lm_head_i8_scale = nullptr;
        }
        static const bool want_i4 = [] {
            const char* e = getenv("SPARKINFER_DFLASH_HEAD_I4");
            return !(e && e[0] == '0');
        }();
        if (want_i4 && p_->lm_head_i8) {
            p_->lm_head_i4 = p_->alloc<unsigned char>((size_t)vocab * (hidden / 2));
            p_->lm_head_i4_scale = p_->alloc<float>(vocab);
            if (p_->lm_head_i4 && p_->lm_head_i4_scale) {
                kernels::launch_pack_i8_rows_i4(p_->lm_head_i8, p_->lm_head_i8_scale,
                                                p_->lm_head_i4, p_->lm_head_i4_scale,
                                                vocab, hidden, p_->stream);
                cudaStreamSynchronize(p_->stream);
            } else {
                p_->lm_head_i4 = nullptr;
                p_->lm_head_i4_scale = nullptr;
            }
        }
        cu(cudaStreamSynchronize(p_->stream), "lm_head int8 prepack");
    }
}

void DFlashDraftModel::reset() {
    p_->seq_len = 0;
    p_->ctx_lo = 0;
    p_->kv_base = 0;
    p_->kv_valid_lo = 0;
    if (p_->tp_peer) p_->tp_peer->reset();   // host state only
}


void DFlashDraftModel::crop(int keep) {
    if (keep < 0) keep = 0;
    if (keep > p_->seq_len) keep = p_->seq_len;
    p_->seq_len = keep;
    if (p_->tp_peer) p_->tp_peer->crop(keep);
}

int DFlashDraftModel::seq_len() const { return p_->seq_len; }

// Defaults keep the draft exactly as it was without sliding: below 12288 it attends everything,
// which a state of 12288 positions plus two blocks holds; from 12288 up (and after a truncated
// capture) its widest window is 2048, which the 4096 kept positions always cover.
int DFlashDraftModel::kv_slide_capacity() const {
    static const int env = [] {
        const char* e = getenv("SPARKINFER_DSPARK_KV_CAP");
        return e ? atoi(e) : 0;
    }();
    const int two_blocks = 2 * (p_->cfg.block_size + 1);
    const int floor = kv_slide_keep() + 2 * two_blocks;
    // DFlash2 windows every layer at any context, so the kept span is all it ever needs.
    const int want = env > 0 ? env : (p_->cfg.dflash2 ? floor : 12288 + two_blocks);
    return std::min(p_->cfg.max_seq, std::max(want, floor));
}

// DFlash2 windows every layer at sliding_window (2048), so a block's queries never read a key
// older than its first position minus the window: that plus two blocks of slack is all a slide has
// to keep. Half of the 4096 kept before, ~25 KB a position per card, so ~50 MB less per running
// stream on each card. SPARKINFER_DFLASH2_TIGHT=0 restores 4096 (and the 4096-row capture).
bool dflash2_tight_on() {
    static const bool v = [] {
        const char* e = getenv("SPARKINFER_DFLASH2_TIGHT");
        return !(e && e[0] == '0');
    }();
    return v;
}

int DFlashDraftModel::kv_slide_keep() const {
    static const int env = [] {
        const char* e = getenv("SPARKINFER_DSPARK_KV_KEEP");
        return e ? atoi(e) : 0;
    }();
    if (env > 0) return env;
    const DFlashDraftConfig& c = p_->cfg;
    if (c.dflash2 && dflash2_tight_on() && c.sliding_window > 0 &&
        !getenv("SPARKINFER_DFLASH_FULL_WINDOW"))
        return c.sliding_window + 2 * (c.block_size + 1);
    return 4096;
}

int DFlashDraftModel::kv_valid_lo() const { return std::max(p_->kv_valid_lo, p_->kv_base); }

bool DFlashDraftModel::kv_snapshot(int lo, int hi, KvSnapshot& out) {
    Impl& s = *p_;
    out = KvSnapshot{};
    if (!s.tp_peer) return kv_snapshot_local(lo, hi, out);
    auto peer = std::make_shared<KvSnapshot>();
    bool ok = false, peer_ok = false;
    tp_run_with_peer(s.tp_peer_dev, [&] { peer_ok = s.tp_peer->kv_snapshot_local(lo, hi, *peer); },
                     [&] { ok = kv_snapshot_local(lo, hi, out); });
    if (!ok || !peer_ok) { out = KvSnapshot{}; return false; }
    out.peer = std::move(peer);
    return true;
}

bool DFlashDraftModel::kv_snapshot_local(int lo, int hi, KvSnapshot& out) {
    Impl& s = *p_;
    if (lo < std::max(s.kv_valid_lo, s.kv_base) || hi > s.seq_len || hi <= lo ||
        s.k_cache.size() != (size_t)s.cfg.n_layers)
        return false;
    const size_t kvdim = (size_t)s.cfg.n_kv_heads * s.cfg.head_dim;
    const size_t rows = (size_t)(hi - lo), plane = rows * kvdim * sizeof(bf16);
    const size_t bytes = plane * 2 * s.cfg.n_layers;
    std::shared_ptr<void> owned = pinned_pool_get(bytes);
    if (!owned) return false;
    void* host = owned.get();
    cudaStream_t st = s.tp_stream ? s.tp_stream : s.stream;
    if (s.tp_stream) cudaStreamSynchronize(s.stream);
    const size_t off = (size_t)(lo - s.kv_base) * kvdim;
    char* h = static_cast<char*>(host);
    for (int L = 0; L < s.cfg.n_layers; L++) {
        cu(cudaMemcpyAsync(h + (2 * L) * plane, s.k_cache[L] + off, plane, cudaMemcpyDeviceToHost, st),
           "draft kv snapshot k");
        cu(cudaMemcpyAsync(h + (2 * L + 1) * plane, s.v_cache[L] + off, plane, cudaMemcpyDeviceToHost,
                           st), "draft kv snapshot v");
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;
    out.host = std::move(owned);
    out.lo = lo;
    out.hi = hi;
    out.bytes = bytes;
    return true;
}

bool DFlashDraftModel::kv_start_at(int from, int seq_len, const KvSnapshot* snap) {
    Impl& s = *p_;
    if (!s.tp_peer) return kv_start_at_local(from, seq_len, snap);
    if (snap && !snap->peer) return false;
    bool ok = false, peer_ok = false;
    tp_run_with_peer(s.tp_peer_dev,
                     [&] { peer_ok = s.tp_peer->kv_start_at_local(from, seq_len,
                                                                 snap ? snap->peer.get() : nullptr); },
                     [&] { ok = kv_start_at_local(from, seq_len, snap); });
    if (ok && peer_ok) return true;
    // One rank started, the other did not: leave both empty (the caller drafts nothing then).
    reset();
    return false;
}

bool DFlashDraftModel::kv_start_at_local(int from, int seq_len, const KvSnapshot* snap) {
    Impl& s = *p_;
    if (from < 0 || seq_len < from || (!snap && from != seq_len)) return false;
    if (snap && (from < snap->lo || seq_len != snap->hi || !snap->host)) return false;
    // Room for the held rows and two blocks, or the very first block slides them out again.
    const int two_blocks = 2 * (s.cfg.block_size + 1);
    if (seq_len - from + two_blocks > s.kv_cap || s.k_cache.size() != (size_t)s.cfg.n_layers)
        return false;
    const size_t kvdim = (size_t)s.cfg.n_kv_heads * s.cfg.head_dim;
    if (snap && seq_len > from) {
        const size_t rows = (size_t)(snap->hi - snap->lo), plane = rows * kvdim * sizeof(bf16);
        const size_t skip = (size_t)(from - snap->lo) * kvdim * sizeof(bf16);
        const size_t n = (size_t)(seq_len - from) * kvdim * sizeof(bf16);
        const char* h = static_cast<const char*>(snap->host.get());
        cudaStream_t st = s.tp_stream ? s.tp_stream : s.stream;
        for (int L = 0; L < s.cfg.n_layers; L++) {
            cu(cudaMemcpyAsync(s.k_cache[L], h + (2 * L) * plane + skip, n, cudaMemcpyHostToDevice, st),
               "draft kv restore k");
            cu(cudaMemcpyAsync(s.v_cache[L], h + (2 * L + 1) * plane + skip, n,
                               cudaMemcpyHostToDevice, st), "draft kv restore v");
        }
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
    }
    s.kv_base = from;
    s.kv_valid_lo = from;
    s.ctx_lo = from;
    s.seq_len = seq_len;
    return true;
}

int DFlashDraftModel::kv_state_create(int capacity) {
    Impl& s = *p_;
    if (!s.tp_peer) return kv_state_create_local(capacity);
    // Both ranks hold one slot per state (each its own KV heads), under the same id: the slot
    // vectors only ever change together, so a free slot on one is free on the other.
    int id = -1, peer_id = -1;
    tp_run_with_peer(s.tp_peer_dev, [&] { peer_id = s.tp_peer->kv_state_create_local(capacity); },
                     [&] { id = kv_state_create_local(capacity); });
    if (id >= 0 && id == peer_id) return id;
    if (id >= 0 || peer_id >= 0)
        fprintf(stderr, "[dflash] kv_state_create: ranks disagree (rank0 %d, rank1 %d); undoing\n",
                id, peer_id);
    if (id >= 0) kv_state_free_local(id);
    if (peer_id >= 0)
        tp_run_with_peer(s.tp_peer_dev, [&] { s.tp_peer->kv_state_free_local(peer_id); }, nullptr);
    return -1;
}

int DFlashDraftModel::kv_state_create_local(int capacity) {
    Impl& s = *p_;
    if (capacity <= 0) return -1;
    if (capacity > s.cfg.max_seq) capacity = s.cfg.max_seq;
    const size_t kvdim = (size_t)s.cfg.n_kv_heads * s.cfg.head_dim;
    Impl::KvState st;
    st.cap = capacity;
    bool ok = true;
    for (int L = 0; L < s.cfg.n_layers && ok; L++) {
        void* k = nullptr;
        void* v = nullptr;
        ok = cudaMalloc(&k, (size_t)capacity * kvdim * sizeof(bf16)) == cudaSuccess;
        if (ok) st.k.push_back((bf16*)k);
        ok = ok && cudaMalloc(&v, (size_t)capacity * kvdim * sizeof(bf16)) == cudaSuccess;
        if (ok) st.v.push_back((bf16*)v);
    }
    if (!ok) {
        cudaGetLastError();
        for (bf16* b : st.k) cudaFree(b);
        for (bf16* b : st.v) cudaFree(b);
        return -1;
    }
    st.live = true;
    for (size_t i = 0; i < s.kv_states.size(); i++)
        if (!s.kv_states[i].live) { s.kv_states[i] = std::move(st); return (int)i; }
    s.kv_states.push_back(std::move(st));
    return (int)s.kv_states.size() - 1;
}

bool DFlashDraftModel::kv_state_select(int id) {
    Impl& s = *p_;
    if (s.tp_peer && !s.tp_peer->p_->select(id)) return false;   // host state only
    return s.select(id);
}

void DFlashDraftModel::kv_state_free(int id) {
    Impl& s = *p_;
    if (s.tp_peer)
        tp_run_with_peer(s.tp_peer_dev, [&] { s.tp_peer->kv_state_free_local(id); },
                         [&] { kv_state_free_local(id); });
    else
        kv_state_free_local(id);
}

void DFlashDraftModel::kv_state_free_local(int id) {
    Impl& s = *p_;
    if (id < 0 || id >= (int)s.kv_states.size() || !s.kv_states[id].live) return;
    if (s.kv_cur == id) s.select(-1);
    cudaStreamSynchronize(s.stream);
    if (s.tp_stream) cudaStreamSynchronize(s.tp_stream);
    Impl::KvState& st = s.kv_states[id];
    for (bf16* b : st.k) cudaFree(b);
    for (bf16* b : st.v) cudaFree(b);
    st = Impl::KvState{};
}

const float* DFlashDraftModel::last_logits() const { return p_->logits; }

// YaRN inverse-frequency table, computed exactly as HuggingFace's _compute_yarn_parameters does
// (transformers/modeling_rope_utils.py) -- the reference dspark.py/dflash.py do not implement RoPE
// themselves, they inherit transformers' Qwen3 classes, so that IS the authority here.
//   inv_freq[i] = interp*(1-ramp_i) + extrap*ramp_i, ramp over the NTK-by-parts correction range
//   att_scale   = 0.1*ln(factor) + 1
// att_scale applies to BOTH q and k, not q alone. HF folds it into the cos/sin tables
// (cos = emb.cos() * attention_scaling) and then uses those same tables for q_embed and k_embed,
// so both are scaled. Reading it as an attention-logit scale and applying it only to q is the
// natural misreading and is wrong -- it changes the logits by att_scale rather than att_scale^2
// and silently degrades acceptance. Both launch_rms_heads_rope calls below pass it for exactly
// this reason.
static void compute_yarn_inv_freq(const DFlashDraftConfig& cfg, std::vector<float>& inv_freq,
                                  float& att_scale) {
    constexpr double kPi = 3.14159265358979323846;
    const int d = cfg.head_dim, half = d / 2;
    const double base = cfg.rope_theta, factor = cfg.yarn_factor;
    const double orig = cfg.yarn_orig_max_pos > 0 ? cfg.yarn_orig_max_pos : 8192.0;
    auto find_dim = [&](double nrot) {
        return (d * std::log(orig / (nrot * 2.0 * kPi))) / (2.0 * std::log(base));
    };
    double low  = std::floor(find_dim(cfg.yarn_beta_fast));
    double high = std::ceil (find_dim(cfg.yarn_beta_slow));
    low  = std::max(low, 0.0);
    high = std::min(high, (double)d - 1.0);
    if (high - low < 1e-3) high = low + 1e-3;   // guard the degenerate range
    inv_freq.resize(half);
    for (int i = 0; i < half; i++) {
        const double pos_freq = std::pow(base, (2.0 * i) / (double)d);
        const double extrap = 1.0 / pos_freq;
        const double interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)i - low) / (high - low);
        ramp = std::min(std::max(ramp, 0.0), 1.0);
        const double extrap_w = 1.0 - ramp;     // 1 => untouched band, 0 => fully interpolated
        inv_freq[i] = (float)(interp * (1.0 - extrap_w) + extrap * extrap_w);
    }
    att_scale = (float)(0.1 * std::log(factor) + 1.0);
}

bool DFlashDraftModel::load(const std::string& dir) {
    Impl& s = *p_;
    const int cu_errors_before = g_cu_errors.load();
    const std::string cfg_path = dir + "/config.json";
    std::string st_path = dir + "/model.safetensors";
    parse_config_json(cfg_path, s.cfg);
    // (dual-GPU) A split rank keeps a contiguous share of the query heads (and with them, GQA
    // being grouped, of the KV heads) and of the FFN columns; from here on cfg holds this rank's
    // counts, so every buffer and kernel below sizes itself for the slice.
    const int tp_n = s.cfg.tp_size > 1 ? s.cfg.tp_size : 1;
    const int tp_r = tp_n > 1 ? s.cfg.tp_rank : 0;
    if (tp_n > 1) {
        const auto& c = s.cfg;
        if (tp_n != 2 || tp_r < 0 || tp_r >= tp_n || c.n_q_heads % tp_n || c.n_kv_heads % tp_n ||
            c.intermediate % (tp_n * 32) || ((c.n_q_heads / tp_n) * c.head_dim) % 32) {
            fprintf(stderr, "[dflash] cannot split this draft %d ways (%d query heads, %d KV heads, "
                            "FFN %d)\n", tp_n, c.n_q_heads, c.n_kv_heads, c.intermediate);
            return false;
        }
        s.cfg.n_q_heads /= tp_n;
        s.cfg.n_kv_heads /= tp_n;
        s.cfg.intermediate /= tp_n;
    }
    const bool lead = tp_r == 0;
    if (s.cfg.dflash2) {
        const auto& c = s.cfg;
        if (c.conv_taps < 1 || c.conv_group < 1 || c.hidden % c.conv_group ||
            (c.hidden / c.conv_group) % 4 || c.selector_rank < 8 || c.selector_rank % 8 ||
            c.selector_top_k < 1 || c.selector_top_k > 32 || c.block_size < 2 ||
            c.block_size > 16) {
            fprintf(stderr, "[dflash] DFlash2 config not supported (taps %d, group %d, rank %d, "
                            "top_k %d, block %d)\n", c.conv_taps, c.conv_group, c.selector_rank,
                    c.selector_top_k, c.block_size);
            return false;
        }
    }
    // DFlash2 ships bf16: load its NVFP4 cache (built on the first load, see dflash2_nvfp4_cache).
    if (s.cfg.dflash2 && draft_nvfp4()) {
        const std::string cache = dflash2_nvfp4_cache(dir);
        if (!cache.empty()) st_path = cache;
    }
    SafeTensorsFile st;
    if (!st.load(st_path)) {
        fprintf(stderr, "[dflash] failed to load %s\n", st_path.c_str());
        return false;
    }
    auto require = [&](const std::string& name) -> TensorView* {
        auto it = st.tensors.find(name);
        if (it == st.tensors.end()) {
            fprintf(stderr, "[dflash] missing tensor %s\n", name.c_str());
            return nullptr;
        }
        return &it->second;
    };
    auto optional = [&](const std::string& name) -> TensorView* {
        auto it = st.tensors.find(name);
        return it == st.tensors.end() ? nullptr : &it->second;
    };

    // Resolve a projection weight whether it is stored BF16 or NVFP4-packed. ModelOpt writes
    // <name>.weight (U8), <name>.weight_scale (ue4m3) and <name>.weight_scale_2 (F32 global);
    // a mixed-precision export leaves untouched projections as plain BF16, so both must work
    // within the same checkpoint.
    // split: 0 = whole, 1 = this rank's rows (output features), 2 = its columns (input features).
    auto load_weight = [&](const std::string& name, int split = 0) -> bf16* {
        TensorView* w = require(name);
        if (!w) return nullptr;
        TensorView* sc = w->dtype == "U8" ? optional(name + "_scale") : nullptr;
        if (w->dtype == "U8" && !sc) {
            fprintf(stderr, "[dflash] %s is NVFP4-packed but %s_scale is missing\n",
                    name.c_str(), name.c_str());
            return nullptr;
        }
        float gs = 1.0f;
        if (sc) {
            TensorView* g2 = optional(name + "_scale_2");
            if (g2 && g2->nbytes >= sizeof(float)) gs = *(const float*)g2->data;
        }
        if (tp_n > 1 && split != 0) return s.upload_part(*w, sc, gs, split == 1, tp_r, tp_n);
        if (w->dtype != "U8") return s.upload(*w);
        return s.upload_nvfp4(*w, *sc, gs);
    };

    const int H = s.cfg.hidden;
    const int I = s.cfg.intermediate;
    const int n_cap = (int)s.cfg.target_layer_ids.size();
    const int B = s.cfg.block_size;

    if (s.cfg.yarn_factor > 1.0f) {
        std::vector<float> ifreq;
        compute_yarn_inv_freq(s.cfg, ifreq, s.yarn_att_scale);
        if (cudaMalloc(&s.d_yarn_inv_freq, ifreq.size() * sizeof(float)) == cudaSuccess) {
            cudaMemcpy(s.d_yarn_inv_freq, ifreq.data(), ifreq.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
            fprintf(stderr, "[dflash] YaRN: factor=%.1f orig_max=%d att_scale=%.4f "
                            "(inv_freq[0]=%.3e inv_freq[%d]=%.3e)\n",
                    s.cfg.yarn_factor, s.cfg.yarn_orig_max_pos, s.yarn_att_scale,
                    ifreq.front(), (int)ifreq.size() - 1, ifreq.back());
        } else {
            fprintf(stderr, "[dflash] YaRN table alloc failed -- falling back to plain RoPE\n");
        }
    }

    auto* fc = require("fc.weight");
    auto* hn = require("hidden_norm.weight");
    auto* nn = require("norm.weight");
    if (!fc || !hn || !nn) return false;
    // (dual-GPU) fc split by input columns (see Impl::fc_split): rank r keeps, of every captured
    // layer's H columns, the half [r * H/2, (r+1) * H/2). SPARKINFER_DFLASH_FC_SPLIT=0 keeps the
    // whole projector on rank 0, which then sends rank 1 the result.
    static const bool kFcSplit = [] { const char* e = getenv("SPARKINFER_DFLASH_FC_SPLIT");
                                      return !(e && e[0] == '0'); }();
    s.fc_split = tp_n == 2 && kFcSplit && fc->dtype == "BF16" && fc->shape.size() == 2 &&
                 (int)fc->shape[0] == s.cfg.hidden && (int)fc->shape[1] == n_cap * s.cfg.hidden &&
                 (s.cfg.hidden / 2) % 256 == 0;
    s.fc_k = s.fc_split ? n_cap * s.cfg.hidden / 2 : n_cap * s.cfg.hidden;
    if (s.fc_split) {
        const int Hh = s.cfg.hidden / 2, K = n_cap * s.cfg.hidden;
        std::vector<bf16> host((size_t)s.cfg.hidden * s.fc_k);
        const bf16* src = (const bf16*)fc->data;
        for (int r = 0; r < s.cfg.hidden; r++)
            for (int l = 0; l < n_cap; l++)
                std::memcpy(host.data() + (size_t)r * s.fc_k + (size_t)l * Hh,
                            src + (size_t)r * K + (size_t)l * s.cfg.hidden + (size_t)tp_r * Hh,
                            (size_t)Hh * sizeof(bf16));
        s.fc = s.alloc<bf16>(host.size());
        cu(cudaMemcpy(s.fc, host.data(), host.size() * sizeof(bf16), cudaMemcpyHostToDevice),
           "upload fc half");
        s.hidden_norm = s.upload(*hn);
    } else if (lead) {
        // Rank 1 of a split draft without the fc split never projects the context (rank 0 sends
        // it the result).
        s.fc = s.upload(*fc);
        s.hidden_norm = s.upload(*hn);
    }
    s.final_norm = s.upload(*nn);
    // NVFP4 projections (Impl::nv_on): fc becomes one payload and its bf16 copy goes.
    s.nv_on = draft_nvfp4();
    if (s.nv_on && s.fc) {
        s.nv_fc = s.quant_nvfp4_payload(s.fc, s.cfg.hidden, s.fc_k);
        s.fc = nullptr;
        if (!s.nv_fc) return false;
    }
    // Quantize the projector alongside the layer weights (see Impl::q8_fc). The bf16 copy stays:
    // the FIRST block projects the whole prompt and routes to the tensor-core GEMM, which is
    // compute-bound and wants bf16.
    if (q8_on() && s.fc)
        s.pending_quant.push_back({s.fc, s.cfg.hidden, s.fc_k, &s.q8_fc});

    s.layers.resize(s.cfg.n_layers);
    for (int L = 0; L < s.cfg.n_layers; L++) {
        auto& lw = s.layers[L];
        const std::string pfx = "layers." + std::to_string(L) + ".";
        auto* wq = require(pfx + "self_attn.q_proj.weight");
        auto* wk = require(pfx + "self_attn.k_proj.weight");
        auto* wv = require(pfx + "self_attn.v_proj.weight");
        auto* wo = require(pfx + "self_attn.o_proj.weight");
        auto* qn = require(pfx + "self_attn.q_norm.weight");
        auto* kn = require(pfx + "self_attn.k_norm.weight");
        auto* in = require(pfx + "input_layernorm.weight");
        auto* pn = require(pfx + "post_attention_layernorm.weight");
        auto* g = require(pfx + "mlp.gate_proj.weight");
        auto* u = require(pfx + "mlp.up_proj.weight");
        auto* d = require(pfx + "mlp.down_proj.weight");
        if (!qn || !kn || !in || !pn)
            return false;
        if (s.cfg.dflash2) {
            // Both convolutions, whole on every rank, bf16 as stored.
            const int taps = s.cfg.conv_taps, nc = 2 * taps * (H / s.cfg.conv_group);
            auto load_conv = [&](const std::string& name, Impl::D2Conv& cv) -> bool {
                TensorView* b = require(pfx + name + ".base_kernel");
                TensorView* p = require(pfx + name + ".kernel_projection.weight");
                if (!b || !p || b->dtype != "BF16" || p->dtype != "BF16" || b->shape.size() != 3 ||
                    b->shape[0] != 2 || b->shape[1] != taps || b->shape[2] != H ||
                    p->shape.size() != 2 || p->shape[0] != nc || p->shape[1] != H) {
                    fprintf(stderr, "[dflash] layer %d: %s is not a [2, %d, %d] kernel with a "
                                    "[%d, %d] bf16 projection\n", L, name.c_str(), taps, H, nc, H);
                    return false;
                }
                cv.base = s.upload(*b);
                cv.proj = s.upload(*p);
                static const bool conv_nv = [] {
                    const char* e = getenv("SPARKINFER_DFLASH2_CONV_NVFP4");
                    return !(e && e[0] == '0');
                }();
                if (conv_nv && s.nv_on) {
                    cv.proj_nv = s.quant_nvfp4_payload(cv.proj, nc, H);
                    if (cv.proj_nv) cv.proj = nullptr;
                }
                return true;
            };
            s.d2_attn.resize(s.cfg.n_layers);
            s.d2_mlp.resize(s.cfg.n_layers);
            if (!load_conv("attention_conv", s.d2_attn[L]) || !load_conv("mlp_conv", s.d2_mlp[L]))
                return false;
        }
        if (s.nv_on) {
            // One NVFP4 payload per projection, this rank's slice (see Impl::nv_on).
            auto load_nv = [&](const std::string& name, int split, int N, int K) -> void* {
                TensorView* t = require(name);
                if (!t) return nullptr;
                const bool part = tp_n > 1 && split != 0;
                if (t->dtype == "U8") {
                    TensorView* sc = optional(name + "_scale");
                    TensorView* g2 = optional(name + "_scale_2");
                    if (!sc) return nullptr;
                    const float gs = g2 && g2->nbytes >= sizeof(float) ? *(const float*)g2->data : 1.f;
                    int n = 0, k = 0;
                    void* d = s.upload_nvfp4_payload(*t, *sc, gs, split == 1, part ? tp_r : 0,
                                                     part ? tp_n : 1, &n, &k);
                    return d && n == N && k == K ? d : nullptr;
                }
                bf16* b = load_weight(name, split);
                return s.quant_nvfp4_payload(b, N, K);
            };
            const int qd = s.cfg.n_q_heads * s.cfg.head_dim, kvd = s.cfg.n_kv_heads * s.cfg.head_dim;
            lw.nv_wq = load_nv(pfx + "self_attn.q_proj.weight", 1, qd, H);
            lw.nv_wk = load_nv(pfx + "self_attn.k_proj.weight", 1, kvd, H);
            lw.nv_wv = load_nv(pfx + "self_attn.v_proj.weight", 1, kvd, H);
            lw.nv_wo = load_nv(pfx + "self_attn.o_proj.weight", 2, H, qd);
            lw.nv_gate = load_nv(pfx + "mlp.gate_proj.weight", 1, I, H);
            lw.nv_up = load_nv(pfx + "mlp.up_proj.weight", 1, I, H);
            lw.nv_down = load_nv(pfx + "mlp.down_proj.weight", 2, H, I);
            lw.q_norm = s.upload(*qn); lw.k_norm = s.upload(*kn);
            lw.input_norm = s.upload(*in); lw.post_norm = s.upload(*pn);
            if (!lw.nv_wq || !lw.nv_wk || !lw.nv_wv || !lw.nv_wo || !lw.nv_gate || !lw.nv_up ||
                !lw.nv_down) {
                fprintf(stderr, "[dflash] layer %d: NVFP4 projections failed to load\n", L);
                return false;
            }
            continue;
        }
        lw.wq = load_weight(pfx + "self_attn.q_proj.weight", 1);
        lw.wk = load_weight(pfx + "self_attn.k_proj.weight", 1);
        lw.wv = load_weight(pfx + "self_attn.v_proj.weight", 1);
        lw.wo = load_weight(pfx + "self_attn.o_proj.weight", 2);
        lw.gate = load_weight(pfx + "mlp.gate_proj.weight", 1);
        lw.up = load_weight(pfx + "mlp.up_proj.weight", 1);
        lw.down = load_weight(pfx + "mlp.down_proj.weight", 2);
        if (!lw.wq || !lw.wk || !lw.wv || !lw.wo || !lw.gate || !lw.up || !lw.down)
            return false;
        lw.q_norm = s.upload(*qn); lw.k_norm = s.upload(*kn);
        lw.input_norm = s.upload(*in); lw.post_norm = s.upload(*pn);
        // Q8_0 mirrors: the batched projections are DRAM-bound at the narrowed diffusion width,
        // so halving their weight bytes is the dominant remaining win. Built once, on load.
        if (q8_on()) {
            const int qd = s.cfg.n_q_heads * s.cfg.head_dim, kvd = s.cfg.n_kv_heads * s.cfg.head_dim;
            s.pending_quant.push_back({lw.wq,   qd,  H,  &lw.q8_wq});
            s.pending_quant.push_back({lw.wk,   kvd, H,  &lw.q8_wk});
            s.pending_quant.push_back({lw.wv,   kvd, H,  &lw.q8_wv});
            s.pending_quant.push_back({lw.wo,   H,   qd, &lw.q8_wo});
            s.pending_quant.push_back({lw.gate, I,   H,  &lw.q8_gate});
            s.pending_quant.push_back({lw.up,   I,   H,  &lw.q8_up});
            s.pending_quant.push_back({lw.down, H,   I,  &lw.q8_down});
            if (s.cfg.eager_quant) {
                s.quant_pending();
                if (s.quant_failed) {
                    fprintf(stderr, "[dflash] quantized copies of layer %d do not fit\n", L);
                    return false;
                }
                s.release_unused_bf16(false);
            }
        }
    }

    // DSpark's Markov head: trained weights sitting in the checkpoint but unused by plain DFlash
    // drafting. Both tensors must be present and agree on rank, or the head is silently skipped
    // (a malformed pair is worth surfacing, but a genuinely draft-without-Markov checkpoint --
    // Qwen3.6-35B-A3B's, for one -- is the normal case and should load exactly as before).
    // DFlash2's candidate selector (rank 0 runs the walk).
    if (s.cfg.dflash2 && lead) {
        const int R = s.cfg.selector_rank;
        TensorView* hp = require("candidate_selector.hidden_projection.weight");
        TensorView* pc = require("candidate_selector.predecessor_codebook");
        TensorView* sc = require("candidate_selector.successor_codebook");
        auto is = [&](const TensorView* t, int64_t rows, int64_t cols) {
            return t && t->dtype == "BF16" && t->shape.size() == 2 && t->shape[0] == rows &&
                   t->shape[1] == cols;
        };
        if (!is(hp, R, H) || !is(pc, s.cfg.vocab, R) || !is(sc, s.cfg.vocab, R)) {
            fprintf(stderr, "[dflash] DFlash2 candidate selector: want a [%d, %d] hidden projection "
                            "and [%d, %d] codebooks, bf16\n", R, H, s.cfg.vocab, R);
            return false;
        }
        s.d2_hproj = s.upload(*hp);
        // (tp=2) The codebooks are [vocab, rank] bf16, 121 MiB each, and the walk reads only the
        // anchor's and the candidates' rows -- at most 1 + depth * k of each a draft, ~58 KB. Keep
        // them in pinned, mapped host memory and let the select kernel read those rows over PCIe:
        // 243 MiB more free on card 0, the card that sets the KV pool. Same bytes, same kernel.
        // SPARKINFER_DFLASH2_SELECTOR_HOST=0 keeps them on the device.
        static const bool cb_host = [] {
            const char* e = getenv("SPARKINFER_DFLASH2_SELECTOR_HOST");
            return !(e && e[0] == '0');
        }();
        if (cb_host && s.cfg.tp_size == 2 && pc->nbytes == sc->nbytes &&
            cudaHostAlloc(&s.d2_host_cb, 2 * pc->nbytes, cudaHostAllocMapped | cudaHostAllocPortable) ==
                cudaSuccess) {
            std::memcpy(s.d2_host_cb, pc->data, pc->nbytes);
            std::memcpy((char*)s.d2_host_cb + pc->nbytes, sc->data, sc->nbytes);
            void* dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, s.d2_host_cb, 0) == cudaSuccess) {
                s.d2_pred = (bf16*)dp;
                s.d2_succ = (bf16*)((char*)dp + pc->nbytes);
            } else {
                cudaGetLastError();
                cudaFreeHost(s.d2_host_cb);
                s.d2_host_cb = nullptr;
            }
        } else {
            cudaGetLastError();
            s.d2_host_cb = nullptr;
        }
        if (!s.d2_host_cb) {
            s.d2_pred = s.upload(*pc);
            s.d2_succ = s.upload(*sc);
        }
    }

    auto* mw1 = optional("markov_head.markov_w1.weight");
    auto* mw2 = optional("markov_head.markov_w2.weight");
    if (lead && mw1 && mw2 && mw1->shape.size() == 2 && mw2->shape.size() == 2 &&
        mw1->shape[1] == mw2->shape[1] && mw1->shape[1] > 0) {
        s.markov_w1 = s.upload(*mw1);
        s.markov_w2 = s.upload(*mw2);
        // SPARKINFER_DFLASH_MARKOV_Q8=0 keeps the bf16 table (A/B).
        {
            static const bool q8_on = []{ const char* e = getenv("SPARKINFER_DFLASH_MARKOV_Q8");
                                          return !(e && e[0] == '0'); }();
            const int rk = (int)mw1->shape[1], nv = (int)mw2->shape[0];
            if (q8_on && rk == 256 && nv > 0) {
                s.markov_w2_q = s.alloc<signed char>((size_t)nv * rk);
                s.markov_w2_s = s.alloc<float>((size_t)nv * (rk / 32));
                dflash_kernels::launch_quantize_w_q8(s.markov_w2, s.markov_w2_q, s.markov_w2_s,
                                                     nv, rk, s.stream);
                cu(cudaStreamSynchronize(s.stream), "markov w2 q8");
                // Release the bf16 table: nothing reads it once the int8 one exists, and holding
                // both would ADD ~72 MB on a card already carrying two full weight copies.
                // Freeing it makes this net -56 MB.
                s.release(s.markov_w2);
                s.markov_w2 = nullptr;
            }
        }
        s.markov_rank = (int)mw1->shape[1];
    } else if (lead && (mw1 || mw2)) {
        fprintf(stderr, "[dflash] markov_head tensors present but malformed "
                        "(w1=%zux%zu w2=%zux%zu) -- skipping\n",
                mw1 ? (size_t)mw1->shape[0] : 0, mw1 ? (size_t)mw1->shape[1] : 0,
                mw2 ? (size_t)mw2->shape[0] : 0, mw2 ? (size_t)mw2->shape[1] : 0);
    }

    // Confidence head: requires the Markov head (confidence_head_with_markov=True on every
    // DSpark checkpoint released so far concatenates the Markov latent into its input), so only
    // look for it once the Markov head itself loaded successfully.
    if (s.markov_w1) {
        auto* cw = optional("confidence_head.proj.weight");
        auto* cb = optional("confidence_head.proj.bias");
        const int want_dim = H + s.markov_rank;
        if (cw && cb && cw->shape.size() == 2 && cw->shape[0] == 1 &&
            (int)cw->shape[1] == want_dim && cb->shape.size() == 1 && cb->shape[0] == 1) {
            s.confidence_w = s.upload(*cw);
            // Bias is a single scalar -- read it back to host once here rather than carrying a
            // device pointer the kernel would have to dereference on every call for no reason.
            const unsigned short raw = *reinterpret_cast<const unsigned short*>(cb->data);
            unsigned int bits = (unsigned int)raw << 16;
            std::memcpy(&s.confidence_bias, &bits, sizeof(float));
            if (cudaMalloc(&s.markov_latent, (size_t)(s.cfg.block_size + 1) * s.markov_rank * sizeof(float)) != cudaSuccess)
                s.confidence_w = nullptr;  // can't run the head without scratch -- disable cleanly
            else
                s.owned.push_back(s.markov_latent);
        } else if (cw || cb) {
            fprintf(stderr, "[dflash] confidence_head tensors present but malformed "
                            "(want proj.weight=[1,%d]) -- skipping\n", want_dim);
        }
    }

    // Scratch + KV cache (shared with load_gguf(), see Impl::alloc_scratch).
    s.alloc_scratch();
    if (const int n = g_cu_errors.load() - cu_errors_before; n > 0) {
        fprintf(stderr, "[dflash] draft NOT loaded: %d CUDA call(s) failed while uploading weights or "
                        "allocating scratch (see above) -- usually device out of memory\n", n);
        return false;
    }
    fprintf(stderr, "[dflash] loaded draft: layers=%d H=%d B=%d n_cap=%d mask=%d markov_rank=%d "
                    "confidence=%d\n",
            s.cfg.n_layers, H, B, n_cap, s.cfg.mask_token_id, s.markov_rank, s.confidence_w != nullptr);
    if (s.cfg.dflash2)
        fprintf(stderr, "[dflash] DFlash2: conv %d taps x group %d, selector top-%d rank %d, window "
                        "%d, %s attention, depth %d\n", s.cfg.conv_taps, s.cfg.conv_group,
                s.cfg.selector_top_k, s.cfg.selector_rank, s.cfg.sliding_window,
                s.cfg.is_causal == 1 ? "causal" : "non-causal", s.cfg.spec_depth);
    return true;
}

namespace {
// launch_gguf_dequant (kernels/quant.h) only implements F32/F16/Q8_0/Q4_K/Q5_K/Q6_K (ggml types
// 0/1/8/12/13/14). Mirrors Qwen35Model's ggml_dequant_supported (qwen35.cpp) so an unsupported
// (or future) quant type is rejected at load time instead of silently falling through as garbage.
bool dflash_gguf_dequant_supported(int ggml_type) {
    switch (ggml_type) {
        case 0: case 1: case 8: case 12: case 13: case 14: return true;
        default: return false;
    }
}
} // namespace

bool DFlashDraftModel::load_gguf(const std::string& path) {
    Impl& s = *p_;
    const int cu_errors_before = g_cu_errors.load();
    GGUF g;
    if (!g.open(path)) {
        fprintf(stderr, "[dflash] failed to open gguf %s\n", path.c_str());
        return false;
    }
    const std::string arch = g.meta_str("general.architecture");
    if (arch != "dflash") {
        fprintf(stderr, "[dflash] %s: expected general.architecture=\"dflash\", got \"%s\"\n",
                path.c_str(), arch.c_str());
        return false;
    }
    museglimmer_dflash_config_from_gguf(g, s.cfg);

    auto require = [&](const std::string& name) -> const GGUFTensor* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) fprintf(stderr, "[dflash] missing tensor %s\n", name.c_str());
        return t;
    };
    // Dense weight -> bf16, kept in its native GGUF [out,in] row-major layout (dims[0]=in
    // fastest, dims[1]=out -- see runtime/examples/qwen3_gguf_config.h's header comment /
    // Qwen35Model::load_gguf for the same convention on the target). That is exactly the layout
    // forward_block already expects for wq/wk/wv/wo/gate/up/down/fc (the safetensors load() above
    // uploads HF nn.Linear.weight raw for the same reason: it too is stored [out,in]), so unlike
    // Qwen35Model::load_gguf's `dense(name, transpose)` this never needs to transpose.
    auto dense = [&](const std::string& name) -> bf16* {
        const GGUFTensor* t = require(name);
        if (!t) return nullptr;
        if (!dflash_gguf_dequant_supported(t->ggml_type)) {
            fprintf(stderr, "[dflash] unsupported ggml type %d for %s\n", t->ggml_type, name.c_str());
            return nullptr;
        }
        void* raw = nullptr;
        cu(cudaMalloc(&raw, t->n_bytes), "gguf raw malloc");
        cu(cudaMemcpy(raw, t->data, t->n_bytes, cudaMemcpyHostToDevice), "gguf raw upload");
        bf16* dst = s.alloc<bf16>(t->n_values);
        kernels::launch_gguf_dequant(t->ggml_type, raw, dst, t->n_values, s.stream);
        cu(cudaStreamSynchronize(s.stream), "gguf dequant sync");
        cudaFree(raw);
        return dst;
    };

    s.fc = dense("fc.weight");
    s.fc_k = (int)s.cfg.target_layer_ids.size() * s.cfg.hidden;
    s.hidden_norm = dense("enc.output_norm.weight");
    s.final_norm = dense("output_norm.weight");
    if (!s.fc || !s.hidden_norm || !s.final_norm) return false;
    if (q8_on())
        s.pending_quant.push_back({s.fc, s.cfg.hidden,
                                   (int)s.cfg.target_layer_ids.size() * s.cfg.hidden, &s.q8_fc});

    const int H = s.cfg.hidden;
    const int I = s.cfg.intermediate;
    const int n_cap = (int)s.cfg.target_layer_ids.size();

    s.layers.resize(s.cfg.n_layers);
    for (int L = 0; L < s.cfg.n_layers; L++) {
        auto& lw = s.layers[L];
        const std::string pfx = "blk." + std::to_string(L) + ".";
        lw.input_norm = dense(pfx + "attn_norm.weight");
        lw.post_norm  = dense(pfx + "ffn_norm.weight");
        lw.wq = dense(pfx + "attn_q.weight");
        lw.wk = dense(pfx + "attn_k.weight");
        lw.wv = dense(pfx + "attn_v.weight");
        lw.wo = dense(pfx + "attn_output.weight");
        lw.q_norm = dense(pfx + "attn_q_norm.weight");
        lw.k_norm = dense(pfx + "attn_k_norm.weight");
        lw.gate = dense(pfx + "ffn_gate.weight");
        lw.up   = dense(pfx + "ffn_up.weight");
        lw.down = dense(pfx + "ffn_down.weight");
        if (!lw.input_norm || !lw.post_norm || !lw.wq || !lw.wk || !lw.wv || !lw.wo ||
            !lw.q_norm || !lw.k_norm || !lw.gate || !lw.up || !lw.down)
            return false;
        // Q8_0/int4 mirrors: same as load()'s safetensors path (see the comment there).
        if (q8_on()) {
            const int qd = s.cfg.n_q_heads * s.cfg.head_dim, kvd = s.cfg.n_kv_heads * s.cfg.head_dim;
            s.pending_quant.push_back({lw.wq,   qd,  H,  &lw.q8_wq});
            s.pending_quant.push_back({lw.wk,   kvd, H,  &lw.q8_wk});
            s.pending_quant.push_back({lw.wv,   kvd, H,  &lw.q8_wv});
            s.pending_quant.push_back({lw.wo,   H,   qd, &lw.q8_wo});
            s.pending_quant.push_back({lw.gate, I,   H,  &lw.q8_gate});
            s.pending_quant.push_back({lw.up,   I,   H,  &lw.q8_up});
            s.pending_quant.push_back({lw.down, H,   I,  &lw.q8_down});
        }
    }

    s.alloc_scratch();
    if (const int n = g_cu_errors.load() - cu_errors_before; n > 0) {
        fprintf(stderr, "[dflash] draft NOT loaded: %d CUDA call(s) failed while uploading weights or "
                        "allocating scratch (see above) -- usually device out of memory\n", n);
        return false;
    }
    fprintf(stderr,
            "[dflash] loaded draft (gguf): layers=%d H=%d B=%d n_cap=%d mask=%d rope_normal=%d\n",
            s.cfg.n_layers, H, s.cfg.block_size, n_cap, s.cfg.mask_token_id,
            (int)s.cfg.rope_normal);
    return true;
}

namespace {
// Rows below this keep the bit-exact row-batched GEMV path. Two reasons for a high bar. The
// prefill GEMM tiles its output 128x128, so a narrow block leaves most of the tile idle and the
// GEMV shape still wins outright. And re-associating this projection perturbs the draft's
// proposals, which at a short context costs more acceptance than the kernel saves (measured at
// 512: mean accept 2.0317 -> 2.0000, a net -0.65% even though the kernel itself got faster).
// Only a genuinely long context ingestion, where the GEMV shape is an order of magnitude off,
// takes the GEMM -- every shorter block stays bit-for-bit what it was.
constexpr int kCtxGemmMinRows = 1024;
bool ctx_gemm_enabled() {
    static const int on = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_GEMM");
                              return (e && e[0] == '0') ? 0 : 1; }();
    return on != 0;
}
}  // namespace

void DFlashDraftModel::ensure_quant() { if (p_) p_->ensure_quant(); }
bool DFlashDraftModel::quant_ok() const {
    return p_ && p_->quant_ready && !p_->quant_failed && (!p_->tp_peer || p_->tp_peer->quant_ok());
}

bool DFlashDraftModel::fc_split() const {
    return p_ && p_->fc_split && p_->tp_peer && p_->tp_peer->p_->fc_split;
}

void DFlashDraftModel::set_peer_hidden_map(std::function<const void*(const void*)> map) {
    p_->peer_hidden = std::move(map);
}

bool DFlashDraftModel::tp_attach(DFlashDraftModel* peer, int peer_device, cudaStream_t stream,
                                 cudaStream_t peer_stream) {
    Impl& s = *p_;
    if (!peer || !stream || !peer_stream || s.cfg.tp_rank != 0 || peer->p_->cfg.tp_rank != 1 ||
        s.cfg.tp_size != 2 || peer->p_->cfg.tp_size != 2)
        return false;
    if (s.fc_split != peer->p_->fc_split) {
        fprintf(stderr, "[dflash] split draft: the ranks disagree on the fc split\n");
        return false;
    }
    // The batched path's row scratch is taken now, on both cards: a rank that could not allocate
    // it lazily, mid-step, would leave the other one waiting in a link op.
    // DFlash2 never takes it (multi_plan wants DSpark's Markov head and an unwindowed draft), and
    // rank 0's share holds a [32, vocab] f32 logits plane: 30 MiB for nothing.
    bool peer_multi = false;
    if (!s.cfg.dflash2)
        tp_run_with_peer(peer_device, [&] { peer_multi = peer->p_->multi_alloc(); },
                         [&] { s.multi_alloc(); });
    s.tp_peer = peer;
    s.tp_peer_dev = peer_device;
    s.tp_stream = stream;
    s.tp_split = true;
    peer->p_->tp_stream = peer_stream;
    peer->p_->tp_split = true;
    fprintf(stderr, "[dflash] split draft: rank 0 on device %d, rank 1 on device %d "
                    "(%d of %d query heads, %d KV heads, FFN %d per card; batched path %s; "
                    "fc %s)\n",
            s.device, peer_device, s.cfg.n_q_heads, s.cfg.n_q_heads * 2, s.cfg.n_kv_heads,
            s.cfg.intermediate, s.m_ready && peer_multi ? "on" : "off",
            s.fc_split ? "split by columns" : "on rank 0");
    return true;
}

bool DFlashDraftModel::forward_block(const void* target_hidden, int ctx_len,
                                     const int* noise_ids, int pos0,
                                     int* out_argmax, cudaStream_t stream, int proposals,
                                     float* out_confidence, int target_hidden_start) {
    Impl& s = *p_;
    s.ensure_quant();
    // Every decline that depends on rank-0-only state happens here, before the peer starts: inside
    // the bodies the two ranks may only return early together, before their first link op.
    if ((!s.fc && !s.nv_fc) || !s.embed || !s.lm_head || !noise_ids || !out_argmax) return false;
    if (!s.tp_peer)
        return forward_block_body(target_hidden, ctx_len, noise_ids, pos0, out_argmax, stream,
                                  proposals, out_confidence, target_hidden_start);
    if (s.kv_cur == -1) {
        // The built-in cache is taken on first use at tp (see alloc_scratch), on both cards.
        bool have = false, peer_have = false;
        tp_run_with_peer(s.tp_peer_dev, [&] { peer_have = s.tp_peer->p_->ensure_builtin_kv(); },
                         [&] { have = s.ensure_builtin_kv(); });
        if (!have || !peer_have) return false;
    }
    bool ok = false, peer_ok = false;
    // fc split: rank 1 projects its own half of the capture (see Impl::fc_split).
    const void* peer_th = s.fc_split && s.peer_hidden && target_hidden ? s.peer_hidden(target_hidden)
                                                                        : nullptr;
    tp_run_with_peer(
        s.tp_peer_dev,
        [&] {
            peer_ok = s.tp_peer->forward_block_body(peer_th, ctx_len, noise_ids, pos0, nullptr,
                                                    nullptr, proposals, nullptr,
                                                    target_hidden_start);
        },
        [&] {
            ok = forward_block_body(target_hidden, ctx_len, noise_ids, pos0, out_argmax, stream,
                                    proposals, out_confidence, target_hidden_start);
        });
    return ok && peer_ok;
}

bool DFlashDraftModel::forward_block_body(const void* target_hidden, int ctx_len,
                                          const int* noise_ids, int pos0,
                                          int* out_argmax, cudaStream_t stream, int proposals,
                                          float* out_confidence, int target_hidden_start) {
    Impl& s = *p_;
    // (dual-GPU) Rank 1 of a split draft: only its half of each layer. Its inputs -- the block's
    // embedding and the projected context -- come from rank 0 through the link.
    const bool lead = s.cfg.tp_rank == 0;
    const bool split = s.tp_split;
    if (ctx_len < 0 || ctx_len + s.cfg.block_size > s.cfg.max_seq + s.cfg.block_size) return false;
    cudaStream_t st = s.tp_stream ? s.tp_stream : (stream ? stream : s.stream);
    const auto& c = s.cfg;
    const int H = c.hidden;
    const int I = c.intermediate;
    const int B = c.block_size;
    const int n_cap = (int)c.target_layer_ids.size();
    const int qdim = c.n_q_heads * c.head_dim;
    const int kvdim = c.n_kv_heads * c.head_dim;
    const int d = c.head_dim;
    // Proposal depth (also sets the draft's active diffusion width, depth+1). The caller selects
    // it by context length and passes it in; the env default only applies when it does not.
    static const int kProposalDepthDefault = []{
        const char* e = getenv("SPARKINFER_DFLASH_PROPOSALS");
        int v = e ? atoi(e) : 5;
        return v < 1 ? 1 : (v > 15 ? 15 : v);
    }();
    // Also clamped to block_size: proposal r reads block row r-1 (or r without the row shift),
    // so a block can never back more proposals than it has rows. Without this a checkpoint whose
    // block_size is below the requested depth reads uninitialised argmax rows as proposals.
    const int kProposalDepth = std::min(c.max_proposals(),
                                        proposals > 0 ? (proposals > 15 ? 15 : proposals)
                                                      : kProposalDepthDefault);
    // Active diffusion width. Only rows 0..kProposalDepth are ever consumed (row 0 is the seed,
    // 1..kProposalDepth the scored proposals), yet the backbone was run at the checkpoint's full
    // block_size=16 — 12 of every 16 rows computed and discarded. The block's attention is
    // bidirectional, so narrowing it DOES change what the draft proposes; that is allowed here
    // because every emitted token is still a target argmax, only the accept length can move.
    // SPARKINFER_DFLASH_BLOCK_WIDTH overrides (0/unset = kProposalDepth+1).
    // Not cached across calls: the proposal depth it derives from is now chosen per generation.
    const int BW = s.width_for(kProposalDepth);
    // DP4A BACKBONE (SPARKINFER_DFLASH_DP4A=0 restores the float path). Quantizes each activation
    // to Q8_1 once and drives every projection that shares it through the dp4a kernel: 1.125 B per
    // element instead of 2, eight dp4a instead of 32 float FMAs per (weight row, block row) per
    // 32-block, and the weights stay packed. Draft-only, so it moves what is PROPOSED, never what
    // is emitted.
    // BITMASK: 1 = Q/K/V, 2 = o-proj, 4 = gate/up, 8 = down. Swept at 512 generated tokens, where
    // tau resolves to 512/222 rather than 128/71 -- at 128 the step-count lottery is 1.4% and
    // swamps the effect, and reading it there produced a confident but WRONG conclusion that the
    // o-projection could not take a quantized activation:
    //
    //     mask           draft ms   step ms   tau      DSPARK
    //     0  (none)       1.475     13.374    2.3198   172.44
    //     12 (FFN)        1.382     13.278    2.3198   173.68
    //     14 (+o_proj)    1.369     13.246    2.3094   173.33
    //     15 (all four)   1.336     13.215    2.3198   174.51   <- default
    //
    // All four take Q8_1 for free. Draft-only either way: this moves what is PROPOSED, never what
    // is emitted, and LOSSLESS stays 1.
    static const int kDp4a = []{ const char* e = getenv("SPARKINFER_DFLASH_DP4A");
                                 return e ? atoi(e) : 15; }();
    // NVFP4 projections (Impl::nv_on) replace every Q4/bf16 projection below.
    const bool nv = s.nv_on;
    const bool dp4a_ok = s.xq81 != nullptr;
    const bool dp4a_qkv  = dp4a_ok && (kDp4a & 1);
    const bool dp4a_o    = dp4a_ok && (kDp4a & 2);
    const bool dp4a_gu   = dp4a_ok && (kDp4a & 4);
    const bool dp4a_down = dp4a_ok && (kDp4a & 8);
    // Set when the producing norm has already emitted Q8_1 of the activation, so the projection
    // that consumes it can skip the standalone quantize launch. The single staging buffer is safe
    // because each fold is immediately followed by its one consumer -- nothing else touches xq81
    // in between.
    bool xn_ready = false, hn_ready = false;
    auto q81n = [&](const bf16* src, int kk, int rows) {
        kernels::launch_quantize_q8_1_rows(src, s.xq81, kk, rows, kk, st);
        return s.xq81;
    };
    auto q81 = [&](const bf16* src, int kk) { return q81n(src, kk, BW); };
    const float scale = 1.f / sqrtf((float)d);
    const int past = s.seq_len;
    // The fixed-size (block_size) projections below can use a batched-GEMV kernel that reads
    // each weight row from DRAM once instead of once per token (see dflash_kernels.cu). It's
    // instantiated for the active width tiers below; an unsupported width falls back to the
    // per-token GEMV loop.
    const bool fast16 = (BW == 16 || BW == 8 || BW == 7 || BW == 6 || BW == 5 ||
                         BW == 4 || BW == 2);
    // DFlash2's convolutions and selector run their projections on the batched GEMV only.
    const bool d2 = c.dflash2;
    if (d2 && !fast16) return false;

    // Stage through pinned memory (see h_ids): a pageable H2D would sync the stream here.
    static const bool kPinIds = []{ const char* e = getenv("SPARKINFER_DFLASH_PIN_IDS");
                                    return !(e && e[0] == '0'); }();
    if (kPinIds && s.h_ids) {
        for (int i = 0; i < BW; i++) s.h_ids[i] = noise_ids[i];
        cu(cudaMemcpyAsync(s.d_ids, s.h_ids, BW * sizeof(int), cudaMemcpyHostToDevice, st), "ids");
    } else {
        cu(cudaMemcpyAsync(s.d_ids, noise_ids, BW * sizeof(int), cudaMemcpyHostToDevice, st), "ids");
    }
    if (!lead) {
        // Rank 1 holds no embedding: zeros here, rank 0's rows arrive in the sum below.
        cu(cudaMemsetAsync(s.noise, 0, (size_t)BW * H * sizeof(bf16), st), "split noise zero");
    } else if (s.embed_rows > 0) {
        // Split table: gather the rows this device owns (zeros elsewhere), then copy each upper-
        // half row in from the peer's half -- the same bytes, so the embedding is bit-identical.
        kernels::launch_embedding_vocab_window(s.d_ids, s.embed, s.noise, BW, H, 0, s.embed_rows, st);
        int dev = 0;
        cu(cudaGetDevice(&dev), "embed split device");
        for (int i = 0; i < BW; i++) {
            const int id = noise_ids[i];
            if (id < s.embed_rows) continue;
            const bf16* src = static_cast<const bf16*>(s.embed_hi) + (size_t)(id - s.embed_rows) * H;
            bf16* dst = s.noise + (size_t)i * H;
            auto it = s.embed_hi_cache.find(id);
            if (it == s.embed_hi_cache.end() && s.embed_hi_cache.size() < 4096) {
                bf16* row = s.alloc<bf16>((size_t)H);
                if (row) {
                    cu(cudaMemcpyPeer(row, dev, src, s.embed_hi_dev, (size_t)H * sizeof(bf16)),
                       "embed split fetch");
                    it = s.embed_hi_cache.emplace(id, row).first;
                }
            }
            if (it != s.embed_hi_cache.end())
                cu(cudaMemcpyAsync(dst, it->second, (size_t)H * sizeof(bf16),
                                   cudaMemcpyDeviceToDevice, st), "embed split row");
            else
                cu(cudaMemcpyPeerAsync(dst, dev, src, s.embed_hi_dev, (size_t)H * sizeof(bf16), st),
                   "embed split peer row");
        }
    } else {
        kernels::launch_embedding(s.d_ids, s.embed, s.noise, BW, H, st);
    }
    if (d2 && lead && c.embed_scale != 1.f)
        dflash_kernels::launch_scale_bf16(s.noise, (long)BW * H, c.embed_scale, st);
    // x + 0 is exact, so both ranks now hold rank 0's embedding rows bit for bit.
    if (split) tp_allreduce_bf16_on(s.noise, (size_t)BW * H, st);

    // target_hidden [ctx, n_cap*H] -> fc -> hidden_norm -> target_proj [ctx, H]
    // fc.weight is [H, n_cap*H] (out, in). Loop gemv per row.
    // Context ingestion (the first block of a generation) runs these projections over the whole
    // prompt: at a 4k context that one step is [4096, n_cap*H] -> [4096, H] here plus [4096, H] ->
    // K/V per layer below. The row-batched GEMV kernels these used give one CTA per (output, row)
    // and reduce K per CTA, so at 4k they hit ~13 TFLOPS -- fine for the 1-6 row steady-state
    // blocks they were written for, an order of magnitude off for a 4096-row one. Route just the
    // wide case to the tensor-core bf16 prefill GEMM (same C[M,N] = A[M,K] @ W^T with the native
    // [N,K] weight, fp32 accumulate); anything at or below kCtxGemmMinRows keeps the existing
    // exact GEMV path bit-for-bit, so every steady-state block is untouched.
    const bool ctx_gemm = ctx_len >= kCtxGemmMinRows && ctx_gemm_enabled();
    // Draft full-attention-layer window (default ON = 3x the draft's sliding_window). The draft is a
    // heuristic proposer whose every token the target verifies, so bounding how far back its "full"
    // attention layer looks can only affect ACCEPTANCE (tau), never correctness. Measured: its
    // proposals are unchanged (accept length identical) with the full layer bounded to 3x the sliding
    // window -- the smallest multiple that stays tau-neutral at both 16k and 32k -- while 2x already
    // costs acceptance. Bounding it lets the split attention (#751) and the ingestion (#752) trim the
    // full layer's read / K-V / fc exactly as they trim the windowed layers, removing the last draft
    // cost that still grew with context. Env override: unset -> 3*sliding_window; 0 -> off (full); N.
    static const int kFullWindowEnv = []{ const char* e = getenv("SPARKINFER_DFLASH_FULL_WINDOW");
                                          return e ? atoi(e) : -1; }();
    // The DSpark checkpoint ships "sliding_window": null. find_int runs strtol over " null" and
    // gets 0, so 3 * c.sliding_window is 0, and BOTH trim paths below are gated behind
    // `kFullWindow > 0` -- the draft has never windowed anything, at any context. Nothing warns:
    // the run succeeds and the draft simply attends every token it has.
    //
    // That is free while the context is short and expensive once it is not. nsys on the decode
    // range, per step: the draft's attention (k_attn_rows_tile_hd128) is 0.32 ms of a 13.6 ms step
    // at ctx=4096 and 2.44 ms of a 20.9 ms step at ctx=32768 -- it grew 7.5x for an 8x context,
    // because it is attending all of it, and the draft goes from 12% of the step to 24%.
    //
    // Windowing is NOT a win everywhere, so it is gated rather than simply switched on. Default vs
    // an 8192 window, one binary, arms alternated, every arm lossless with AR flat:
    //     ctx    default    window 8192
    //     4k     143.03  ->  142.90    0%      inert by construction, tau BIT-IDENTICAL at 1.9394:
    //                                          a window wider than the context trims nothing
    //     16k    231.46  ->  214.05   -7.5%    (second prompt: 161.45 -> 152.17, -5.7%)
    //     24k    174.55  ->  176.82   +1.3%
    //     32k    109.13  ->  122.63  +12.4%    (second prompt: 102.08 -> 116.05, +13.7%)
    // Both ends reproduce on an independent prompt, so neither the win nor the loss is noise. So it
    // engages only once the context is at least 3x the window -- once the draft would be keeping at
    // most a third of what it is paying to attend to.
    //
    // PROVENANCE, because it bounds what these numbers mean: the long prompts above come from
    // bench/scripts/gen_eval_prompt.py --len, which builds a long stream by tiling seed-shuffled
    // paragraph orders of a short corpus. Such a stream is self-similar, and how much exploitable
    // structure sits beyond the window depends on where the truncation lands -- which is why the
    // measured tau is non-monotonic in the context (5.20 at 16k, 3.69 at 24k, 2.30 at 32k). The
    // RELATIVE comparisons are still sound (the target verifies every token, so both arms of a pair
    // emit identical text and a tau difference is a real acceptance difference), but the absolute
    // tok/s are not representative of non-repeating prose, and the sign of this trade is known to
    // move with acceptance: it LOSES in the high-tau regime above.
    //
    // The gate is on `past + ctx_len`, NOT ctx_len. ctx_len is how many NEW target rows this block
    // ingests -- the whole prompt on the first block and then just `keep` (1..8) on every one
    // after -- so gating on it alone engages the window for one block and switches it off for the
    // rest of the run.
    // One window, one threshold. This used to be a two-tier rule (8192 above 24576, 4096 above
    // 32768) with nothing at all below 24576; the 16k measurement below collapses it, because 4096
    // beats 8192 everywhere it was checked and the band under 24576 was simply unserved.
    // kMidCtxMinSeq is deliberately the same 12288 boundary the proposal-depth ladder in
    // qwen35.cpp uses -- the two policies describe the same regime and should not drift apart.
    //
    // 2048 rather than 4096 in the band this can be measured in. Re-swept on the SCORED prompt
    // (first 16384 tokens of bench_prompt_32k.txt) at the shipped proposal depth, not the depth-2
    // ladder the 4096 column above was taken at. Both terms move the right way -- old context is
    // not buying acceptance here, it is only costing draft time:
    //
    //   window    8192     4096     3072     2560     2048     1536     1024
    //   tok/s    117.72   118.94   122.42   122.69   126.11   123.20   123.77   (128 gen tokens)
    //   tau      1.6203   1.6000   1.6410   1.6410   1.6842   1.6410   1.6410
    //   draft ms  1.691    1.370    1.310    1.263    1.235    1.203    1.168
    //
    // tau at 128 generated tokens is quantised (128/80, /78, /76), so the peak was re-measured at
    // 512 to confirm it is a mechanism and not a step boundary: 4096 -> 150.85 tok/s at tau 2.0397
    // and draft 1.287 ms, 2048 -> 155.23 at tau 2.0894 and draft 1.183, with 1536 (154.02) and
    // 3072 (151.87) on either side. Acceptance +2.4% and draft cost -8% at the same time.
    // 2048 reproduces exactly across fresh processes (126.095 / 126.125, tau 1.6842 in both).
    //
    // The band is CLOSED at the top rather than extended: 24576 and above keeps the 4096 that was
    // independently measured there. Narrowing further at 32k is plausible from the trend, but a
    // 32 GB card cannot run ctx=32768 through the batched prefill at all (the scratch arena is
    // ~13 MB short and the run falls back), so it cannot be measured here -- and dspark-decode@32k
    // is both a scored dimension and a no-regression floor. Ship what is measured.
    static const int kMidCtxWindow = 2048;
    static const int kMidCtxMinSeq = 12288;
    static const int kMidCtxMaxSeq = 24576;
    // The long band keeps 4096 while the mid band was narrowed to 2048, but the reason the mid
    // band moved applies harder here, not less: acceptance falls with context while the draft's
    // attention cost rises with it, so at 32k old context buys even less than it does at 16k.
    // Measured at ctx=32768 on bench_prompt_32k.txt, lossless at every point, MEAN_ACCEPT
    // identical (1.2800) across the whole range -- this is draft time removed, not acceptance
    // traded away:
    //
    //     window   4096     3072     2048     1024      256
    //     tok/s   89.28    89.88    90.44    90.79    91.33
    //
    // 2048 rather than the faster 256, for the reason the mid band chose 4096 over 3072: 512 and
    // 768 sit in a reproducible acceptance dip (tau 1.2673 and 1.2549), so the narrow end is not a
    // broad peak on this prompt and picking its best point is fitting one corpus. 2048 is the
    // constant the band below already uses, which collapses the window ladder to one value above
    // 12288.
    static const int kLongCtxWindow = 2048;
    // Sliding KV (kv_state_create): the block's rows go at cache row past - kv_base. When they do
    // not fit, keep the newest kv_slide_keep() positions before the block's first query and drop
    // the rest; positions below the new base are gone, so ctx_lo rises to it, which windows the
    // full-attention layer as a truncated capture does (checked against the kept span below).
    if (past < s.kv_base) return false;
    int slide_base = s.kv_base;
    if (past - s.kv_base + ctx_len + BW > s.kv_cap) {
        slide_base = std::max(s.kv_base, pos0 - kv_slide_keep());
        s.ctx_lo = std::max(s.ctx_lo, slide_base);
    }
    int kFullWindow = kFullWindowEnv >= 0 ? kFullWindowEnv : 3 * c.sliding_window;
    // DFlash2 has no full-attention layer: this only bounds the context rows fc projects.
    if (d2 && kFullWindowEnv < 0) kFullWindow = c.sliding_window;
    if (kFullWindowEnv < 0 && kFullWindow <= 0) {
        const int total_ctx = past + ctx_len;
        // The scored prose prompt has low draft acceptance, so old context buys less than it did
        // on the repeating calibration corpus above. At 32k, narrowing the window from 8192 to
        // 4096 cuts draft time 1.920 -> 1.525 ms while remaining lossless and moving end-to-end
        // throughput 89.85 -> 91.80 tok/s. Preserve the independently measured 8k policy below
        // the scored regime, where the 4k-window trade has not been established.
        // ...and the same is true from 12k up, which is where the draft's attention first becomes
        // a large share of the step. Below 24576 the rule above never fired at all -- kFullWindow
        // is 3 * c.sliding_window and the checkpoint's sliding_window parses to 0 -- so at 16k the
        // draft was still attending all 16384 tokens and costing 2.886 ms of a 15.7 ms step.
        // Measured at ctx=16384 on the first 16384 tokens of bench_prompt_32k.txt (real prose, not
        // a tiling), three fresh processes per arm, every arm lossless and AR flat at 85.39-85.50:
        //     window   none     8192     4096     3072     2048      (at proposal depth 2)
        //     tok/s   82.478   86.093   92.031   92.749   91.594
        // 4096 rather than the marginally better 3072: the peak is broad, 3072 is worth 0.8% on
        // ONE prompt, and reusing the constant the scored regime already uses is worth more than
        // that. So a single threshold now covers everything from 12288 up.
        if (total_ctx >= kMidCtxMinSeq)
            kFullWindow = total_ctx < kMidCtxMaxSeq ? kMidCtxWindow : kLongCtxWindow;
        // A truncated capture (the tp group path keeps a prompt's last 4096 rows) windows the
        // full-attention layer at any length: rows below ctx_lo were never ingested. 2048 is the
        // window both measured bands above settled on.
        else if (std::max(s.ctx_lo, past == 0 ? target_hidden_start : 0) > 0)
            kFullWindow = kMidCtxWindow;
    }
    if (past == 0 && target_hidden_start > 0) s.ctx_lo = std::max(s.ctx_lo, target_hidden_start);
    if (slide_base > 0) {
        // Positions below the base are gone, so every layer must window (as after a truncated
        // capture). With the defaults the window (2048) lies inside the kept span (4096) whenever
        // the state slid, so a slide drops nothing the attention would read; a window wider than
        // the kept span, or a state started past its prompt's beginning (kv_start_at), attends
        // only what is held. SPARKINFER_DFLASH_FULL_WINDOW=0 (no window) cannot slide.
        for (int L = 0; L < c.n_layers; L++) {
            int win = (L < (int)c.sliding_layers.size() && c.sliding_layers[L]) ? c.sliding_window : 0;
            if (win == 0 && kFullWindow > 0) win = kFullWindow;
            if (win <= 0) {
                fprintf(stderr, "[dflash] KV slide: layer %d attends the whole context, which no "
                                "longer starts at 0 (base %d)\n", L, slide_base);
                return false;
            }
        }
    }
    if (past + ctx_len + BW - slide_base > s.kv_cap) {
        fprintf(stderr, "[dflash] KV overflow past=%d new=%d base=%d max=%d\n", past, ctx_len + BW,
                slide_base, s.kv_cap);
        return false;
    }
    if (slide_base > s.kv_base) {
        // Move the kept rows [slide_base, past) to the front. RoPE was applied when they were
        // written, so they stay valid at any row. Copies of at most `shift` rows never overlap.
        const int shift = slide_base - s.kv_base;
        const int keep = std::max(0, past - slide_base);
        const size_t row = (size_t)kvdim * sizeof(bf16);
        for (int L = 0; L < c.n_layers; L++)
            for (int off = 0; off < keep; off += shift) {
                const int rows = std::min(shift, keep - off);
                cu(cudaMemcpyAsync(s.k_cache[L] + (size_t)off * kvdim,
                                   s.k_cache[L] + (size_t)(shift + off) * kvdim, rows * row,
                                   cudaMemcpyDeviceToDevice, st), "kv slide k");
                cu(cudaMemcpyAsync(s.v_cache[L] + (size_t)off * kvdim,
                                   s.v_cache[L] + (size_t)(shift + off) * kvdim, rows * row,
                                   cudaMemcpyDeviceToDevice, st), "kv slide v");
            }
        static const bool kSlideLog = getenv("SPARKINFER_DSPARK_SLIDE_LOG") != nullptr;
        if (kSlideLog && lead)
            fprintf(stderr, "[dflash] KV slide: base %d -> %d (kept %d positions, pos0 %d)\n",
                    s.kv_base, slide_base, keep, pos0);
        s.kv_base = slide_base;
    }
    s.kv_valid_lo = std::max(s.kv_valid_lo, s.kv_base);
    const int kv_rel = past - s.kv_base;   // cache row of position `past`
    int ctx_skip_max = 0;   // context rows some layer did not project (kv_valid_lo)
    // fc trim: once the full-attn layer is windowed, NO layer reads target_proj older than the
    // largest window across layers, so project only that tail. Uses the same attn_gqa_kv_lo bound
    // the per-layer ingestion (#752) applies, at the LARGEST window -> surviving rows byte-identical,
    // and every skipped row is one no layer reads. kFullWindow==0 -> fc_skip 0 (unchanged).
    int fc_skip = 0;
    {
        static const int kCT = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_TRIM");
                                   return (e && e[0] == '0') ? 0 : 1; }();
        if (ctx_len > 0 && kCT && kFullWindow > 0) {
            const int maxwin = c.sliding_window > kFullWindow ? c.sliding_window : kFullWindow;
            const int lo = dflash_kernels::attn_gqa_kv_lo(BW, past + ctx_len + BW, c.n_q_heads,
                                                          c.n_kv_heads, d, pos0, 0, maxwin);
            fc_skip = lo > past ? (lo - past < ctx_len ? lo - past : ctx_len) : 0;
        }
    }
    const int fc_rows = ctx_len - fc_skip;
    // Route the two remaining BF16 weight reads in this block -- the projector above and the
    // per-layer context K/V below -- through the Q4 copies. 0 restores bf16 for both.
    static const bool kCtxQ4 = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_Q4");
                                   return !(e && e[0] == '0'); }();
    if (fc_rows > 0 && !lead && !s.fc_split) {
        if (fc_skip < target_hidden_start) return false;
        cu(cudaMemsetAsync(s.target_proj + (size_t)fc_skip * H, 0, (size_t)fc_rows * H * sizeof(bf16),
                           st), "split ctx zero");
        tp_allreduce_bf16_on(s.target_proj + (size_t)fc_skip * H, (size_t)fc_rows * H, st);
    } else if (fc_rows > 0) {
        if (fc_skip < target_hidden_start) return false;
        const int fk = s.fc_k;
        const bf16* th = (const bf16*)target_hidden +
                         (size_t)(fc_skip - target_hidden_start) * fk;
        bf16* tp = s.target_proj + (size_t)fc_skip * H;
        const bool fc_q4 = kCtxQ4 && s.q8_fc.q4 && fc_rows >= 1 && fc_rows <= 8;
        if (!target_hidden) {
            // A split rank whose capture twin could not be found: its half contributes nothing
            // (the draft proposes worse, nothing emitted changes), but it still joins the sum.
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr, "[dflash] fc split: no capture rows on rank %d\n", c.tp_rank);
            }
            cu(cudaMemsetAsync(tp, 0, (size_t)fc_rows * H * sizeof(bf16), st), "fc half zero");
        } else if (s.nv_fc) {
            s.nv_proj(th, nullptr, fc_rows, fk, s.nv_fc, tp, H, nullptr, nullptr, 0, nullptr,
                      nullptr, 0, st);
        } else if (ctx_gemm) {
            kernels::launch_prefill_gemm(th, s.fc, tp, fc_rows, H, fk, st);
        } else if (fc_q4 && dp4a_ok) {
            dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                q81n(th, fk, fc_rows), s.q8_fc.q4, nullptr, nullptr,
                s.q8_fc.dm, nullptr, nullptr,
                tp, nullptr, nullptr, H, 0, 0, fk, st, fc_rows);
        } else if (fc_q4) {
            dflash_kernels::launch_gemv_batched_q4_fused3(
                th, s.q8_fc.q4, nullptr, nullptr, s.q8_fc.dm, nullptr, nullptr,
                tp, nullptr, nullptr, H, 0, 0, fk, st, fc_rows);
        } else if (fc_rows > 1) {
            dflash_kernels::launch_gemv_rows_batched(th, s.fc, tp, fc_rows, H, fk, st);
        } else {
            kernels::launch_gemv(th, s.fc, tp, H, fk, st);
        }
        // fc split: the two ranks' column halves sum to the whole projection, on both cards.
        if (s.fc_split) tp_allreduce_bf16_on(tp, (size_t)fc_rows * H, st);
        dflash_kernels::launch_rms(tp, s.hidden_norm, tp, fc_rows, H, c.rms_eps, st);
        // Ablation (SPARKINFER_DFLASH_ZERO_CTX=1): blank the projected target features after
        // computing them. The draft then attends over an all-zero context while everything else --
        // shapes, positions, RoPE, the block forward, the Markov chain -- is untouched. If tau is
        // unchanged the draft is IGNORING the target context, which localises the acceptance gap to
        // the fc/hidden_norm/KV-injection path rather than to the draft backbone or the head.
        if (getenv("SPARKINFER_DFLASH_ZERO_CTX"))
            cu(cudaMemsetAsync(tp, 0, (size_t)fc_rows * H * sizeof(bf16), st), "zero ctx");
        // The projected context, to rank 1 (which adds zeros).
        if (split && !s.fc_split) tp_allreduce_bf16_on(tp, (size_t)fc_rows * H, st);
    }

    // x = noise embedding
    cu(cudaMemcpyAsync(s.x, s.noise, (size_t)BW * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
       "noise->x");

    static const int active_layers = [] {
        const char* e = getenv("SPARKINFER_DFLASH_LAYERS");
        return e ? atoi(e) : 0;
    }();
    const int run_layers = active_layers > 0 ? std::min(active_layers, c.n_layers) : c.n_layers;
    // cat(k_ctx, k_noise) is built at exactly the layout and length the cache slice expects, so
    // staging it in k_new/v_new and memcpy'ing it in cost two extra D2D launches per layer -- 12 a
    // block, on a draft that is launch-bound (its kernels are 1-3 us and the gaps between them are
    // the same size). Project straight into the cache slice instead; the RoPE then runs in place
    // there. Same values written to the same addresses in the same order.
    const int new_len_all = ctx_len + BW;
    // Attention geometry for this block, hoisted: the context ingestion below needs the layer's
    // window and the block's key span to know which context rows the attention can still reach.
    const int kv_len_for_block = past + new_len_all;
    const int q_pos0_for_block = pos0;
    for (int L = 0; L < run_layers; L++) {
        const LayerWeights& w = s.layers[L];
        int window_of_layer = (L < (int)c.sliding_layers.size() && c.sliding_layers[L])
                                        ? c.sliding_window : 0;
        if (window_of_layer == 0 && kFullWindow > 0) window_of_layer = kFullWindow;
        bf16* const kdst = s.k_cache[L] + (size_t)kv_rel * kvdim;
        bf16* const vdst = s.v_cache[L] + (size_t)kv_rel * kvdim;
        if (L == 0)
            dflash_kernels::launch_rms(s.x, w.input_norm, s.xn, BW, H, c.rms_eps, st);
        // DFlash2: the attention reads the normed rows through its convolution's first side.
        const bf16* qin = s.xn;
        if (d2) {
            s.d2_prepare(s.d2_attn[L], s.xn, BW, BW, st);
            qin = s.d2_cv;
        }

        // Q from noise, K/V from cat(target, noise)
        if (nv) {
            s.nv_proj(qin, xn_ready ? s.xq81 : q81(qin, H), BW, H, w.nv_wq, s.q, qdim,
                      w.nv_wk, kdst + (size_t)ctx_len * kvdim, kvdim,
                      w.nv_wv, vdst + (size_t)ctx_len * kvdim, kvdim, st);
        } else if (fast16) {
            if (w.q8_wq.q4 && dp4a_qkv)
                dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                    xn_ready ? s.xq81 : q81(qin, H), w.q8_wq.q4, w.q8_wk.q4, w.q8_wv.q4,
                    w.q8_wq.dm, w.q8_wk.dm, w.q8_wv.dm,
                    s.q, kdst + (size_t)ctx_len * kvdim, vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
            else if (w.q8_wq.q4)
                dflash_kernels::launch_gemv_batched_q4_fused3(
                    qin, w.q8_wq.q4, w.q8_wk.q4, w.q8_wv.q4, w.q8_wq.dm, w.q8_wk.dm, w.q8_wv.dm,
                    s.q, kdst + (size_t)ctx_len * kvdim, vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
            else if (w.q8_wq.q)
                dflash_kernels::launch_gemv_batched_q8_fused3(
                    qin, w.q8_wq.q, w.q8_wk.q, w.q8_wv.q, w.q8_wq.s, w.q8_wk.s, w.q8_wv.s,
                    s.q, kdst + (size_t)ctx_len * kvdim, vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
            else
                dflash_kernels::launch_gemv_batched16_fused3(
                    qin, w.wq, w.wk, w.wv,
                    s.q, kdst + (size_t)ctx_len * kvdim,
                    vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
        } else {
            for (int t = 0; t < BW; t++)
                kernels::launch_gemv(s.xn + (size_t)t * H, w.wq, s.q + (size_t)t * qdim, qdim, H, st);
        }
        xn_ready = false;   // consumed above; xq81 is reused by the projections that follow
        // Context half of cat(k_ctx, k_noise), written ahead of the noise rows.
        //
        // A windowed layer never reads a key older than `window` before the query, and
        // launch_attn_gqa now starts its partition above that bound rather than masking it inside
        // the loop -- so on a long prompt the leading context rows are projected into the cache and
        // then never touched. Skip producing them. The bound comes from attn_gqa_kv_lo, the same
        // function the launcher uses, evaluated at THIS block: q_pos0 only grows as the generation
        // proceeds, and the one-split-worth guard it applies gets easier to satisfy as it does
        // (the bound climbs a key per token while a split's width climbs a fraction of one), so a
        // row dead for this block is dead for every later block too. Full-attention layers and
        // every steady-state block (where the window still covers the cache) get skip == 0 and are
        // byte-for-byte unchanged.
        static const int kCtxTrim = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_TRIM");
                                        return (e && e[0] == '0') ? 0 : 1; }();
        const int ctx_kv_lo = (ctx_len > 0 && kCtxTrim)
            ? dflash_kernels::attn_gqa_kv_lo(BW, kv_len_for_block, c.n_q_heads, c.n_kv_heads, d,
                                             q_pos0_for_block, /*k_pos0=*/0, window_of_layer)
            : 0;
        // Context rows below the slid base have no cache row (and are outside every window).
        const int ctx_skip = std::max(ctx_kv_lo > past ? std::min(ctx_kv_lo - past, ctx_len) : 0,
                                      std::min(ctx_len, std::max(0, s.kv_base - past)));
        const int ctx_rows = ctx_len - ctx_skip;
        ctx_skip_max = std::max(ctx_skip_max, ctx_skip);
        const bf16* const ctx_src = s.target_proj + (size_t)ctx_skip * H;
        bf16* const kdst_ctx = kdst + (size_t)ctx_skip * kvdim;
        bf16* const vdst_ctx = vdst + (size_t)ctx_skip * kvdim;
        // The steady-state context rows went through the BF16 wk/wv while every other projection
        // in this block -- including the noise rows' own K/V, from the same two matrices -- goes
        // through the Q4 copies. That is 21 MB of weights per layer against 2.6 MB, on a path that
        // runs 1-5 rows and is therefore purely weight-bound.
        //
        // It was left bf16 out of caution about the context being the one place the TARGET's
        // information enters the draft. Measured, that caution is unfounded: sweeping the draft's
        // whole backbone precision (SPARKINFER_DFLASH_WBITS) does not move acceptance in the
        // direction precision would predict. At 512 generated tokens, MORE precision gives LOWER
        // acceptance -- Q4 tau 2.3733, Q8 tau 2.3624 for +0.47 ms of draft -- so what is being
        // read is numeric luck, not draft quality. Judge changes on this path by `draft ms/call`
        // and `batched ms/call`, which are deterministic to ~0.005 ms; a tau delta under ~2% on a
        // single prompt says nothing. The draft is insensitive to weight precision, so the context
        // rows may use the same Q4 copies the block rows do.
        //
        // The wide first-block case (ctx_gemm) stays on the bf16 tensor-core GEMM: at 4096 rows it
        // is COMPUTE bound and already runs at ~169 TFLOPS, ~80% of this card's bf16 peak, where a
        // dequantising path would only add work. SPARKINFER_DFLASH_CTX_Q4=0 restores bf16.
        // ctx_rows is the ACCEPTED length, so 1 is its modal value -- and the single-row case was
        // the most expensive of all, two separate bf16 GEMVs each streaming the whole 10.5 MB
        // matrix to produce one row. Cover 1..8, not just the multi-row tiers.
        const bool ctx_q4 = kCtxQ4 && w.q8_wk.q4 && w.q8_wv.q4 && ctx_rows >= 1 && ctx_rows <= 8;
        if (ctx_rows > 0 && nv) {
            s.nv_proj(ctx_src, nullptr, ctx_rows, H, w.nv_wk, kdst_ctx, kvdim, w.nv_wv, vdst_ctx,
                      kvdim, nullptr, nullptr, 0, st);
        } else if (ctx_rows > 0 && ctx_gemm) {
            kernels::launch_prefill_gemm(ctx_src, w.wk, kdst_ctx, ctx_rows, kvdim, H, st);
            kernels::launch_prefill_gemm(ctx_src, w.wv, vdst_ctx, ctx_rows, kvdim, H, st);
        } else if (ctx_q4 && dp4a_ok) {
            dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                q81n(ctx_src, H, ctx_rows), w.q8_wk.q4, w.q8_wv.q4, nullptr,
                w.q8_wk.dm, w.q8_wv.dm, nullptr,
                kdst_ctx, vdst_ctx, nullptr, kvdim, kvdim, 0, H, st, ctx_rows);
        } else if (ctx_q4) {
            // Same fused pair the noise rows use; y0/y1 are written as [row][kvdim], which is
            // exactly the cache slice's layout.
            dflash_kernels::launch_gemv_batched_q4_fused3(
                ctx_src, w.q8_wk.q4, w.q8_wv.q4, nullptr, w.q8_wk.dm, w.q8_wv.dm, nullptr,
                kdst_ctx, vdst_ctx, nullptr, kvdim, kvdim, 0, H, st, ctx_rows);
        } else if (ctx_rows > 1) {
            dflash_kernels::launch_gemv_rows_exact_fused2(
                ctx_src, w.wk, w.wv, kdst_ctx, vdst_ctx,
                ctx_rows, kvdim, kvdim, H, st);
        } else if (ctx_rows == 1) {
            kernels::launch_gemv(ctx_src, w.wk, kdst_ctx, kvdim, H, st);
            kernels::launch_gemv(ctx_src, w.wv, vdst_ctx, kvdim, H, st);
        }
        if (!fast16 && !nv) {
            for (int t = 0; t < BW; t++) {
                kernels::launch_gemv(s.xn + (size_t)t * H, w.wk,
                                     kdst + (size_t)(ctx_len + t) * kvdim, kvdim, H, st);
                kernels::launch_gemv(s.xn + (size_t)t * H, w.wv,
                                     vdst + (size_t)(ctx_len + t) * kvdim, kvdim, H, st);
            }
        }

        const int new_len = ctx_len + BW;
        // Q / K RMSNorm per head


        // RoPE: Q at positions past..(past+B) if past≈pos0 for noise-only positions.
        // Match reference: position_ids cover past_len .. start+block_size for the cat length.
        // k positions: past .. past+new_len-1 when past==seq_len and we're appending.
        // Absolute: k_pos0 = pos0 - ctx_len (context features align with tokens just before noise),
        // q_pos0 = pos0.
        const int k_pos0 = pos0 - ctx_len;
        const int q_pos0 = pos0;
        // c.rope_normal selects consecutive-pair ("normal"/LLAMA_ROPE_TYPE_NORM) RoPE instead of
        // the NeoX split-half pairing every other DFlash draft checkpoint (Qwen3.6) uses -- see
        // DFlashDraftConfig::rope_normal and k_rms_heads_rope_normal (dflash_kernels.cu) for why
        // Muse Glimmer's draft sets this. Default false leaves Qwen3.6 byte-for-byte unchanged.
        if (c.rope_normal) {
            dflash_kernels::launch_rms_heads_rope_normal(s.q, w.q_norm, BW, c.n_q_heads, d,
                                                         c.rms_eps, q_pos0, c.rope_theta, st);
            // Norm+RoPE only the rows that were actually produced above; the skipped context
            // prefix holds no K to normalise. Row i of this slice keeps its own absolute
            // position, so the surviving rows get exactly the angles they got before.
            dflash_kernels::launch_rms_heads_rope_normal(kdst + (size_t)ctx_skip * kvdim, w.k_norm,
                                                         new_len - ctx_skip, c.n_kv_heads, d,
                                                         c.rms_eps, k_pos0 + ctx_skip,
                                                         c.rope_theta, st);
        } else {
            // s.d_yarn_inv_freq is null unless the checkpoint configures rope_type "yarn", in
            // which case these two calls are the ONLY behavioural change -- Q and K must be
            // rotated with the same table or their relative phase is wrong.
            dflash_kernels::launch_rms_heads_rope(s.q, w.q_norm, BW, c.n_q_heads, d, c.rms_eps,
                                                 q_pos0, c.rope_theta, st,
                                                 s.d_yarn_inv_freq, s.yarn_att_scale);
            dflash_kernels::launch_rms_heads_rope(kdst + (size_t)ctx_skip * kvdim, w.k_norm,
                                                 new_len - ctx_skip, c.n_kv_heads, d,
                                                 c.rms_eps, k_pos0 + ctx_skip, c.rope_theta, st,
                                                 s.d_yarn_inv_freq, s.yarn_att_scale);
        }

        // K/V are already in the cache at row kv_rel -- attend over the whole cache, whose row 0
        // holds position kv_base.
        const int kv_len = kv_rel + new_len;
        const int window = window_of_layer;
        static int mixed_causal = [] {
            const char* e = getenv("SPARKINFER_DFLASH_MIXED_CAUSAL");
            return (!e || e[0] != '0') ? 1 : 0;
        }();
        // Full-attention DFlash layers are ENCODER_ONLY in the reference/SGLang implementation
        // unless the checkpoint explicitly declares is_causal. This checkpoint does not, so its
        // draft block is bidirectional: row i sees the later mask-token rows it was trained with.
        // SparkInfer used to force every layer causal by default after an ablation made against
        // the old, displaced Markov-logit rows. Repeating the comparison after fixing that row
        // mapping reverses the decision: reference semantics raise tau 1.561 -> 1.600 and decode
        // 106.70 -> 109.36 tok/s at depth 1, with identical 1.999 ms draft cost and lossless output.
        // Sliding layers remain causal through mixed_causal below. The override is retained for
        // checkpoints that need it: SPARKINFER_DFLASH_FORCE_CAUSAL=1 forces every layer causal.
        static const int kForceCausal = []{
            const char* e = getenv("SPARKINFER_DFLASH_FORCE_CAUSAL");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        // DFlash2 follows the checkpoint's is_causal (false: its block is bidirectional).
        const bool causal = kForceCausal ||
                            (d2 ? c.is_causal == 1
                                : (mixed_causal && L < (int)c.sliding_layers.size() &&
                                   c.sliding_layers[L]));
        dflash_kernels::launch_attn_gqa(s.q, s.k_cache[L], s.v_cache[L], s.attn,
                                        BW, kv_len, c.n_q_heads, c.n_kv_heads, d,
                                        q_pos0, /*k_pos0_cache=*/s.kv_base, window, causal, scale,
                                        st, s.fa_m, s.fa_l, s.fa_acc);

        if (nv) {
            s.nv_proj(s.attn, q81(s.attn, qdim), BW, qdim, w.nv_wo, s.ao, H, nullptr, nullptr, 0,
                      nullptr, nullptr, 0, st);
        } else if (fast16) {
            if (w.q8_wo.q4 && dp4a_o)
                dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                    q81(s.attn, qdim), w.q8_wo.q4, nullptr, nullptr,
                    w.q8_wo.dm, nullptr, nullptr,
                    s.ao, nullptr, nullptr, H, 0, 0, qdim, st, BW);
            else if (w.q8_wo.q4)
                dflash_kernels::launch_gemv_batched_q4_fused3(
                    s.attn, w.q8_wo.q4, nullptr, nullptr, w.q8_wo.dm, nullptr, nullptr,
                    s.ao, nullptr, nullptr, H, 0, 0, qdim, st, BW);
            else if (w.q8_wo.q)
                dflash_kernels::launch_gemv_batched_q8_fused3(
                    s.attn, w.q8_wo.q, nullptr, nullptr, w.q8_wo.s, nullptr, nullptr,
                    s.ao, nullptr, nullptr, H, 0, 0, qdim, st, BW);
            else
                dflash_kernels::launch_gemv_batched16(s.attn, w.wo, s.ao, H, qdim, st, BW);
        } else {
            for (int t = 0; t < BW; t++)
                kernels::launch_gemv(s.attn + (size_t)t * qdim, w.wo, s.ao + (size_t)t * H, H, qdim, st);
        }
        // Split: each rank's o_proj covers its own heads (K-split), so the sum is the full product.
        if (split) tp_allreduce_bf16_on(s.ao, (size_t)BW * H, st);
        // DFlash2: the attention's output through its convolution's second side; the MLP then
        // reads the normed rows through its own convolution's first side.
        if (d2) s.d2_finish(s.d2_attn[L], s.ao, BW, BW, st);
        dflash_kernels::launch_add_rms(s.x, d2 ? s.d2_cv : s.ao, s.h, w.post_norm, s.hn, BW, H,
                                       c.rms_eps, st, dp4a_gu && !d2 ? s.xq81 : nullptr);
        hn_ready = dp4a_gu && !d2;
        const bf16* gin = s.hn;
        if (d2) {
            s.d2_prepare(s.d2_mlp[L], s.hn, BW, BW, st);
            gin = s.d2_cv;
        }
        if (nv) {
            s.nv_proj(gin, hn_ready ? s.xq81 : q81(gin, H), BW, H, w.nv_gate, s.gate, I,
                      w.nv_up, s.up, I, nullptr, nullptr, 0, st);
        } else if (fast16) {
            if (w.q8_gate.q4 && dp4a_gu)
                dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                    hn_ready ? s.xq81 : q81(gin, H), w.q8_gate.q4, w.q8_up.q4, nullptr,
                    w.q8_gate.dm, w.q8_up.dm, nullptr,
                    s.gate, s.up, nullptr, I, I, 0, H, st, BW);
            else if (w.q8_gate.q4)
                dflash_kernels::launch_gemv_batched_q4_fused3(
                    gin, w.q8_gate.q4, w.q8_up.q4, nullptr, w.q8_gate.dm, w.q8_up.dm, nullptr,
                    s.gate, s.up, nullptr, I, I, 0, H, st, BW);
            else if (w.q8_gate.q)
                dflash_kernels::launch_gemv_batched_q8_fused3(
                    gin, w.q8_gate.q, w.q8_up.q, nullptr, w.q8_gate.s, w.q8_up.s, nullptr,
                    s.gate, s.up, nullptr, I, I, 0, H, st, BW);
            else
                dflash_kernels::launch_gemv_batched16_fused2(
                    gin, w.gate, w.up, s.gate, s.up, I, I, H, st, BW);
        } else {
            for (int t = 0; t < BW; t++) {
                kernels::launch_gemv(s.hn + (size_t)t * H, w.gate, s.gate + (size_t)t * I, I, H, st);
                kernels::launch_gemv(s.hn + (size_t)t * H, w.up,   s.up   + (size_t)t * I, I, H, st);
            }
        }
        dflash_kernels::launch_swiglu(s.gate, s.up, s.gate, BW * I, st);
        if (nv) {
            s.nv_proj(s.gate, q81(s.gate, I), BW, I, w.nv_down, s.down, H, nullptr, nullptr, 0,
                      nullptr, nullptr, 0, st);
        } else if (fast16) {
            if (w.q8_down.q4 && dp4a_down)
                dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                    q81(s.gate, I), w.q8_down.q4, nullptr, nullptr,
                    w.q8_down.dm, nullptr, nullptr,
                    s.down, nullptr, nullptr, H, 0, 0, I, st, BW);
            else if (w.q8_down.q4)
                dflash_kernels::launch_gemv_batched_q4_fused3(
                    s.gate, w.q8_down.q4, nullptr, nullptr, w.q8_down.dm, nullptr, nullptr,
                    s.down, nullptr, nullptr, H, 0, 0, I, st, BW);
            else if (w.q8_down.q)
                dflash_kernels::launch_gemv_batched_q8_fused3(
                    s.gate, w.q8_down.q, nullptr, nullptr, w.q8_down.s, nullptr, nullptr,
                    s.down, nullptr, nullptr, H, 0, 0, I, st, BW);
            else
                dflash_kernels::launch_gemv_batched16(s.gate, w.down, s.down, H, I, st, BW);
        } else {
            for (int t = 0; t < BW; t++)
                kernels::launch_gemv(s.gate + (size_t)t * I, w.down, s.down + (size_t)t * H, H, I, st);
        }
        // Fold the second residual into the norm that always consumes it: the next layer's input
        // norm, or the final norm after the last layer. Same math, one launch instead of two, and
        // the draft is eager-launched so each saved launch is also a saved gap.
        if (split) tp_allreduce_bf16_on(s.down, (size_t)BW * H, st);   // K-split over the FFN
        if (d2) s.d2_finish(s.d2_mlp[L], s.down, BW, BW, st);
        const bf16* next_norm = (L + 1 < run_layers) ? s.layers[L + 1].input_norm : s.final_norm;
        dflash_kernels::launch_add_rms(s.h, d2 ? s.d2_cv : s.down, s.x, next_norm, s.xn, BW, H,
                                       c.rms_eps, st, dp4a_qkv && !d2 ? s.xq81 : nullptr);
        xn_ready = dp4a_qkv && !d2;
        // Per-layer residual stream, for the differential (see the dump block below). s.x is the
        // layer's output residual; comparing it layer by layer turns "the backbone diverges"
        // into "layer N diverges", which is the difference between a search and a fix.
        if (const char* dd = lead ? getenv("SPARKINFER_DSPARK_DUMP") : nullptr) {
            static int dumped_layers = 0;
            if (dumped_layers <= L) {
                dumped_layers = L + 1;
                cudaStreamSynchronize(st);
                std::vector<bf16> host((size_t)BW * H);
                cudaMemcpy(host.data(), s.x, host.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
                char path[512];
                snprintf(path, sizeof(path), "%s/layer%d_x.bin", dd, L);
                if (FILE* f = fopen(path, "wb")) {
                    fwrite(host.data(), sizeof(bf16), host.size(), f);
                    fclose(f);
                }
            }
        }
    }
    if (run_layers <= 0)
        dflash_kernels::launch_rms(s.x, s.final_norm, s.xn, BW, H, c.rms_eps, st);
    if (ctx_skip_max > 0) s.kv_valid_lo = std::max(s.kv_valid_lo, past + ctx_skip_max);
    if (d2) {
        // DFlash2: the candidate head on both cards, the selector on rank 0 (Impl::d2_head). The
        // proposals the block cannot back are padded with the mask token, which never verifies.
        const bool ok = s.d2_head(BW, kProposalDepth, out_argmax, st);
        if (ok && lead && proposals > kProposalDepth)
            for (int t = kProposalDepth + 1; t <= std::min(proposals, c.block_size); t++)
                out_argmax[t] = c.mask_token_id;
        s.seq_len = std::min(pos0 < 0 ? 0 : pos0, past + ctx_len + BW);
        return ok;
    }
    if (!lead) {
        // The head and the Markov chain run on rank 0 only; this rank just keeps its cache in
        // step (the same advance-then-crop as below).
        s.seq_len = std::min(pos0 < 0 ? 0 : pos0, past + ctx_len + BW);
        return true;
    }

    // LM head (target weights) -> logits / argmax. Batched over the whole block: one batched
    // GEMV against the eagerly dequantized bf16 lm_head cache instead of a per-token
    // quantized GEMV loop.
    const int V = s.vocab > 0 ? s.vocab : c.vocab;
    // DRAFT OUTPUT VOCABULARY (SPARKINFER_DFLASH_DRAFT_VOCAB, 0 = full).
    //
    // Everything downstream of the backbone is proportional to the vocabulary, and on Qwen3.8 that
    // vocabulary is 248320. Per block the head streams the target's Q4_K LM head once (715 MB) and
    // then the Markov chain re-reads w2 -- [248320, 256] int8 plus its per-32 scales, ~71 MB --
    // ONCE PER PROPOSAL ROW, serially. Between them they are the two largest items in the draft,
    // and #913 made that worse in the direction that matters here: at depth 4 the chain runs four
    // rows, so the head moves 715 + 4 x 71 = ~1001 MB a block.
    //
    // Both are NOMINATION-only: a token this head cannot score is simply never proposed, which
    // costs at most one acceptance and can never change what is emitted, because every emitted
    // token is still a target argmax over the FULL vocabulary. So the head may score a prefix.
    //
    // The rows are contiguous in both tables ([vocab, ...], row-major), so a prefix is just a
    // smaller length -- no repack, no second copy, no extra VRAM. Qwen3.8's vocabulary has no
    // reserved tail to reclaim (only 243 of the 248320 ids are undefined), but it IS
    // frequency-ordered, and coverage plateaus hard. Fraction of tokens with id < K:
    //
    //     K        bench_prompt_4k   repo prose   repo C++     LM head   w2+scales
    //     32768        92.61%          90.86%      93.77%        94 MB      9.4 MB
    //     65536        97.53%          96.64%      97.92%       189 MB     18.7 MB
    //     98304        99.95%          99.23%     100.00%       283 MB     28.1 MB
    //    163840        99.95%          99.38%     100.00%       472 MB     46.8 MB
    //
    // Three unrelated corpora agree on the shape: the ids above ~100k are rare scripts and exotic
    // unicode, not this text, and by 65536 the curve has already flattened to within 2.5% of it.
    //
    // 65536 is NOT the throughput peak. Swept end to end at ctx 4096, one binary, tok/s and mean
    // accept, with the draft's own per-block cost:
    //
    //     K         DSPARK tok/s   draft ms/block   mean accept
    //     248320      122.87           2.575          1.8286
    //     131072      125.90           2.250          1.8286
    //      98304      125.98           2.159          1.8286
    //      65536      127.08           2.075          1.8286
    //      49152      127.84           2.03           1.8286   <- peak
    //      32768      125.90           1.99           1.8028
    //      24576      126.42           1.96           1.8028
    //      16384      125.00           1.94           1.7778
    //
    // Acceptance is flat down to 49152 and turns over below it: at 32768 the generation needs an
    // extra step (128 tokens in 71 instead of 70), which is what mean accept 1.8286 -> 1.8028 is.
    // So 49152 sits ONE step from that edge, and its 0.6% advantage over 65536 was found by
    // walking the prompt this benchmark scores. 65536 is taken off the coverage curve instead --
    // the one three unrelated corpora agree on -- and keeps a full step of margin. Giving up 0.6%
    // is the right side to err on for a change whose entire safety argument is coverage.
    //
    // The Markov head's OTHER table, w1, is indexed by the PREVIOUS token and is not pruned: the
    // conditioning still accepts any target token, only the output side is narrowed.
    static const int kDraftVocab = []{
        const char* e = getenv("SPARKINFER_DFLASH_DRAFT_VOCAB");
        return e ? atoi(e) : 65536;
    }();
    // Vd is the logits ROW STRIDE as well as the length: the multi-row head writes y[m*N + n], so
    // the stride has to travel with it or the Markov chain reads the wrong row.
    const int Vd = (kDraftVocab > 0 && kDraftVocab < V) ? kDraftVocab : V;
    // The batched bf16 head (#661) still streams the DEQUANTIZED head every step: at V=248k,
    // K=2048 that is ~1.0 GB per pass versus ~416 MB for the native Q6_K bytes, and it pins ~1 GB
    // of VRAM for the bf16 cache. Prefer a multi-row Q6_K MMVQ that keeps the head quantized: each
    // warp owns one output row, walks its superblocks once and accumulates all B dot products, so
    // the weight streams from HBM a single time in its compact form.
    // SPARKINFER_DFLASH_HEAD_MULTIROW=0 falls back to the bf16 batched path.
    static int head_mr = -1;
    if (head_mr < 0) { const char* e = getenv("SPARKINFER_DFLASH_HEAD_MULTIROW"); head_mr = (e && e[0] == '0') ? 0 : 1; }
    // SGLang consumes base-logit rows [0, depth). Keep the legacy [1, depth+1) producer and
    // consumer together behind one A/B switch; changing only one side reads the wrong (or, on the
    // multi-row fast path, uninitialised) logits and is not a meaningful comparison.
    static const int kRowShift = []{
        const char* e = getenv("SPARKINFER_DFLASH_ROW_SHIFT");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    const int head_row0 = kRowShift ? 0 : 1;
    bool head_done = false;
    // The tensor-core head reads 8 rows from head_row0 (the depth padded to the mma's 8); xn and
    // logits hold BW >= 8 rows.
    if (BW >= 8 + head_row0 && kProposalDepth <= 8)
        head_done = s.head_fp4(s.xn + (size_t)head_row0 * H, kProposalDepth,
                               s.logits + (size_t)head_row0 * Vd, Vd, st);
    if (!head_done && head_mr && s.head_q8 && (s.lm_head_type == 14 || s.lm_head_type == 12)) {
        // Score only the proposal rows the verifier can consume. One row-batched quantize launch
        // instead of kProposalDepth tiny ones (8 CTAs each, so launch latency dominated them).
        // DSpark's Markov chain consumes base-logit rows [0, depth): row 0 plus the anchor token
        // produces proposal 1, row 1 plus proposal 1 produces proposal 2, and so on. Quantizing
        // rows [1, depth+1) here was the other half of the legacy row displacement below: merely
        // fixing the consumer would otherwise make row 0 read uninitialised logits on this fast
        // multi-row head path.
        kernels::launch_quantize_q8_1_rows(s.xn + (size_t)head_row0 * H,
                                           s.head_q8, H, kProposalDepth, H, st);
        // Prefer the Q4_K head when one is bound: this kernel already runs near HBM peak, so its
        // runtime IS its weight bytes -- ~280 MB in Q4_K against ~417 MB in Q6_K at V=248k.
        static const bool head_i8 = [] {
            const char* e = getenv("SPARKINFER_DFLASH_HEAD_I8");
            return !(e && e[0] == '0');
        }();
        if (s.lm_head_type == 12 && s.lm_head_i4)
            head_done = kernels::launch_gemv_i4_q81_multirow_f32(
                s.head_q8, s.lm_head_i4, s.lm_head_i4_scale,
                s.logits + (size_t)head_row0 * Vd, Vd, H, kProposalDepth, st);
        else if (s.lm_head_type == 12 && head_i8 && s.lm_head_i8)
            head_done = kernels::launch_gemv_i8_q81_multirow_f32(
                s.head_q8, s.lm_head_i8, s.lm_head_i8_scale,
                s.logits + (size_t)head_row0 * Vd, Vd, H, kProposalDepth, st);
        else if (s.lm_head_type == 12)
            head_done = kernels::launch_gemv_q4k_dp4a_multirow_f32(
                s.head_q8, s.lm_head, s.logits + (size_t)head_row0 * Vd,
                Vd, H, kProposalDepth, st);
        else
            head_done = kernels::launch_gemv_q6k_dp4a_multirow_f32(
                s.head_q8, s.lm_head, s.logits + (size_t)head_row0 * Vd,
                Vd, H, kProposalDepth, st);
    }
    if (!head_done) {
    if (BW == 16) {
        dflash_kernels::launch_gemv_batched16_f32(s.xn, s.lm_head_bf16, s.logits, Vd, H, st);
    } else {
        for (int t = 0; t < B; t++) {
            const bf16* row = s.xn + (size_t)t * H;
            float* logit_row = s.logits + (size_t)t * Vd;
            if (s.lm_head_type)
                kernels::launch_gemv_q_f32(row, s.lm_head, s.lm_head_type, logit_row, Vd, H, st);
            else
                kernels::launch_gemv_f32(row, s.lm_head, logit_row, Vd, H, st);
        }
    }
    }
    // DSpark's Markov head: position r's logit bias conditions on the token AT position r itself
    // (the anchor for r==0, its own -- just-chosen -- prediction for r>0), predicting position
    // r+1. That token is only known once position r-1's own Markov-corrected argmax has run, so
    // proposals 1..kProposalDepth chain sequentially instead of the single batched argmax below --
    // each launch reads its "previous token" from device memory (s.d_ids[0], the real anchor, for
    // r==1; this same loop's own s.d_out[r-1] for r>1), so the chain needs no host round-trip
    // until the final readback already below. Base row 0 is consumed to form proposal 1; d_out[0]
    // itself is not a proposal because the caller already owns the target-produced anchor token.
    // MARKOV HEAD: ALWAYS ON. It was gated on sequence length (#868, threshold 12288) because it
    // "raises acceptance and costs more than the acceptance is worth below long context". That was
    // measured inside the regime the gate itself creates, and the conclusion was wrong.
    //
    // The head is not an optimisation layered on DSpark -- it IS DSpark. The algorithm is a
    // semi-autoregressive block drafter: one forward emits the gamma-token block, and a lightweight
    // sequential head conditions each position on the PREVIOUS drafted token. Without it the block
    // forward alone cannot differentiate positions and the draft degenerates into repeating one
    // high-frequency token. Measured at ctx=4096 with SPARKINFER_DSPARK_PROBE=1:
    //
    //   off: draft=[13 13 13 13 13 13]   ("." six times)                      accept 0-1
    //   on:  draft=[6337 421 369 524 279 1788]
    //        target=[6337 421 369 524 177460]                                 accept 4
    //
    // Over a generation at ctx=4096, depth 6: tau 1.085 -> 1.600. That also explains why tau used
    // to be INVARIANT to proposal depth (1.076 / 1.085 / 1.085 at depth 1 / 3 / 6) -- positions past
    // the first were never going to be accepted, so drafting more of them changed nothing.
    //
    // The head genuinely costs ~1.41 ms of a 16.3 ms step: w2 is [vocab, rank] = 248320 x 256 bf16
    // = 127 MB re-read per proposal row, and the chain serialises bias -> confidence -> argmax per
    // row. That cost repays only through the BATCHED verify, where accepting k tokens costs ONE
    // target forward instead of k. On the token loop it cannot repay at any tau, because that path
    // runs one target forward per kept token exactly as AR does -- which is what the original
    // measurement was really observing: it priced the head against a verify path structurally
    // incapable of using acceptance.
    //
    // SPARKINFER_DFLASH_MARKOV=0 turns it off for A/B. Draft-only either way: this shifts what is
    // proposed, never what is emitted, so LOSSLESS is unaffected.
    static const int markov_env = []{
        const char* e = getenv("SPARKINFER_DFLASH_MARKOV");
        return e ? (e[0] == '0' ? 0 : 1) : -1;
    }();
    // Default ON at every context. SPARKINFER_DFLASH_DEEP_MIN_SEQ no longer gates this; it still
    // selects the proposal-depth ladder in qwen35.cpp, which is a separate decision.
    const bool markov_on = markov_env >= 0 ? (markov_env != 0) : true;
    if (markov_on && s.markov_w1 && (s.markov_w2 || s.markov_w2_q)) {
        // ROW SHIFT (SPARKINFER_DFLASH_ROW_SHIFT=0 restores the legacy mapping). Which block row
        // backs proposal r?
        //
        // The legacy path reads row r. SGLang reads row r-1: its Markov sampler iterates over ALL
        // gamma rows of base_logits starting at row 0, with first_prev_tokens = the anchor
        // (draft_block_ids[:,0], the bonus token) -- so its FIRST proposal comes from row 0, the
        // row holding the real seed token, and its gamma-th from row gamma-1. Ours discards row 0
        // as "redundant with the target's own verify-row-0 call" and starts at row 1, so every
        // proposal is backed by the row one position later than the one that predicts it. The
        // conditioning is aligned (r==1 uses the seed either way); only the logit row is off.
        //
        // That is consistent with the symptom: tau lands well above 1.0 because the Markov bias,
        // conditioned correctly, partially compensates for the wrong base row -- but it is capped,
        // and deeper drafting buys nothing because every position is displaced.
        if (!head_done) kernels::launch_argmax(s.logits, s.d_out, 1, Vd, st);
        for (int r = 1; r <= kProposalDepth; r++) {
            const int* prev = (r == 1) ? s.d_ids : (s.d_out + (r - 1));
            const size_t row = (size_t)(kRowShift ? r - 1 : r);
            if (s.markov_w2_q)
                dflash_kernels::launch_markov_bias_add_q8(
                    s.markov_w1, s.markov_w2_q, s.markov_w2_s, prev,
                    s.logits + row * Vd, Vd, s.markov_rank, st,
                    s.confidence_w ? s.markov_latent + (size_t)r * s.markov_rank : nullptr);
            else
                dflash_kernels::launch_markov_bias_add(
                    s.markov_w1, s.markov_w2, prev,
                    s.logits + row * Vd, Vd, s.markov_rank, st,
                    s.confidence_w ? s.markov_latent + (size_t)r * s.markov_rank : nullptr);
            kernels::launch_argmax(s.logits + row * Vd, s.d_out + r, 1, Vd, st);
        }
        // Confidence head (optional), for every proposal row at once. It reads row r's hidden
        // state and the Markov latent that row's bias call wrote, and nothing in the chain ever
        // consumed it, so hoisting it out of the loop changes no value -- only the launch count,
        // from kProposalDepth grid-of-one kernels to one.
        if (s.confidence_w)
            dflash_kernels::launch_confidence_head_rows(
                s.xn, H, s.markov_latent, s.markov_rank,
                s.confidence_w, s.confidence_bias, H, s.markov_rank,
                kRowShift ? 0 : 1, kProposalDepth, s.d_confidence, st);
    } else {
        kernels::launch_argmax(s.logits + (size_t)(head_done ? head_row0 : 0) * Vd,
                               s.d_out + (head_done ? 1 : 0),
                               head_done ? kProposalDepth : B, Vd, st);
    }
    if (head_done)
        cu(cudaMemcpyAsync(s.h_out + 1, s.d_out + 1, kProposalDepth * sizeof(int),
                           cudaMemcpyDeviceToHost, st), "argmax");
    else
        cu(cudaMemcpyAsync(s.h_out, s.d_out, B * sizeof(int),
                           cudaMemcpyDeviceToHost, st), "argmax");

    // NUMERICAL DIFFERENTIAL DUMP (SPARKINFER_DSPARK_DUMP=<dir>). Writes ONE block's worth of
    // inputs and intermediates so a Python reference can recompute the same step from the same
    // bytes and diff tensor by tensor. Dumps only the first call of a process -- one block is
    // enough to localise a numerical divergence, and dumping every step would write gigabytes.
    //
    // Exists because every structural hypothesis has been eliminated (block construction, row
    // indexing, context injection, injected-KV positions, NVFP4 scale convention, aux-layer
    // capture point) while acceptance is still ~1.36 against a reported ~3.8. What is left can
    // only be found by comparing numbers, not by reasoning about wiring.
    if (const char* dump_dir = getenv("SPARKINFER_DSPARK_DUMP")) {
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            cudaStreamSynchronize(st);
            auto put = [&](const char* nm, const void* dev, size_t bytes, bool from_device) {
                std::vector<char> host(bytes);
                if (from_device)
                    cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost);
                else
                    memcpy(host.data(), dev, bytes);
                std::string path = std::string(dump_dir) + "/" + nm + ".bin";
                FILE* f = fopen(path.c_str(), "wb");
                if (f) { fwrite(host.data(), 1, bytes, f); fclose(f); }
            };
            const size_t V_ = (size_t)V;
            put("target_hidden", target_hidden, (size_t)ctx_len * s.fc_k * sizeof(bf16), true);
            put("target_proj",   s.target_proj, (size_t)ctx_len * H * sizeof(bf16), true);
            put("xn_last",       s.xn,          (size_t)BW * H * sizeof(bf16), true);
            // The block's token embeddings -- the draft borrows the TARGET's embed table, so this
            // is the only way to get layer 0's input without re-deriving it from a 20 GB sharded
            // NVFP4 checkpoint whose embedding may itself be quantized.
            put("noise_embed",   s.noise,       (size_t)BW * H * sizeof(bf16), true);
            put("logits",        s.logits,      (size_t)BW * V_ * sizeof(float), true);
            put("noise_ids",     noise_ids,     (size_t)BW * sizeof(int), false);
            put("d_out",         s.d_out,       (size_t)BW * sizeof(int), true);
            std::string meta = std::string(dump_dir) + "/meta.txt";
            if (FILE* f = fopen(meta.c_str(), "w")) {
                fprintf(f, "ctx_len %d\npos0 %d\npast %d\nBW %d\ndepth %d\nH %d\nV %d\n"
                           "n_cap %d\nmarkov_rank %d\nblock_size %d\nfc_skip %d\n",
                        ctx_len, pos0, past, BW, kProposalDepth, H, V, n_cap, s.markov_rank,
                        c.block_size, fc_skip);
                fclose(f);
            }
            fprintf(stderr, "[dspark-dump] wrote one block to %s (ctx_len=%d BW=%d depth=%d)\n",
                    dump_dir, ctx_len, BW, kProposalDepth);
        }
    }
    if (out_confidence && s.confidence_w)
        cu(cudaMemcpyAsync(s.h_confidence + 1, s.d_confidence + 1,
                           kProposalDepth * sizeof(float), cudaMemcpyDeviceToHost, st),
           "confidence readback");
    cu(cudaStreamSynchronize(st), "draft sync");
    for (int t = head_done ? 1 : 0; t <= kProposalDepth; t++) out_argmax[t] = s.h_out[t];
    if (out_confidence && s.confidence_w)
        for (int t = 1; t <= kProposalDepth; t++) out_confidence[t] = s.h_confidence[t];

    // Advance past the just-appended ctx+noise, then crop to `pos0` (= block start).
    // Matches z-lab dflash: past_key_values_draft.update(...) then .crop(start).
    // Without this, seq_len stays 0, crop(pos0) clamps to 0, and every step rebuilds
    // from an empty cache — draft quality collapses (τ≈1.x) after the first block.
    s.seq_len = std::min(pos0 < 0 ? 0 : pos0, past + ctx_len + BW);   // this rank's crop(pos0)
    return true;
}


// (dual-GPU) Several sessions' blocks in one pass: every projection and the head run once over all
// sessions' rows (weights streamed once per 16 rows instead of once per session); the context K/V
// append, RoPE and attention run per session against that session's own KV state. Same math as
// forward_block per row, on the steady-state path only -- anything else (a first block ingesting
// a whole prompt, a windowed draft, a non-dp4a or non-Q4 weight, too many rows) runs the sessions
// one by one through forward_block.
bool DFlashDraftModel::forward_blocks(int n, const DraftSeg* seg, int proposals,
                                      cudaStream_t stream) {
    Impl& s = *p_;
    s.ensure_quant();
    if (n < 1 || !seg) return false;
    if (!s.tp_peer) return forward_blocks_body(n, seg, proposals, stream, -1);
    // Split: rank 0 picks the path (it holds the head-side state the choice reads) and both ranks
    // take it. Per-session declines of forward_block are rank-0-only too, so they are checked here.
    if ((!s.fc && !s.nv_fc) || !s.embed || !s.lm_head) return false;
    for (int j = 0; j < n; j++)
        if (!seg[j].ids || !seg[j].out_argmax) return false;
    std::vector<int> past(n), ctx_off(n);
    int ctx_total = 0;
    const int mode = (s.multi_plan(n, seg, proposals, past, ctx_off, ctx_total) && s.m_ready &&
                      s.tp_peer->p_->m_ready) ? 1 : 0;
    bool ok = false, peer_ok = false;
    // fc split: rank 1 projects its own half of each session's capture (see Impl::fc_split).
    std::vector<DraftSeg> peer_seg;
    if (s.fc_split) {
        peer_seg.assign(seg, seg + n);
        for (auto& q : peer_seg)
            q.target_hidden = s.peer_hidden && q.target_hidden ? s.peer_hidden(q.target_hidden)
                                                               : nullptr;
    }
    const DraftSeg* pseg = s.fc_split ? peer_seg.data() : seg;
    tp_run_with_peer(
        s.tp_peer_dev,
        [&] { peer_ok = s.tp_peer->forward_blocks_body(n, pseg, proposals, nullptr, mode); },
        [&] { ok = forward_blocks_body(n, seg, proposals, stream, mode); });
    return ok && peer_ok;
}

bool DFlashDraftModel::forward_blocks_body(int n, const DraftSeg* seg, int proposals,
                                           cudaStream_t stream, int mode) {
    Impl& s = *p_;
    const auto& c = s.cfg;
    const bool lead = c.tp_rank == 0;
    const bool split = s.tp_split;
    auto one_by_one = [&]() {
        for (int j = 0; j < n; j++) {
            if (!s.select(seg[j].state)) return false;
            s.walk = seg[j].walk;
            if (!forward_block_body(seg[j].target_hidden, seg[j].ctx_len, seg[j].ids, seg[j].pos0,
                                    seg[j].out_argmax, stream, proposals, nullptr,
                                    seg[j].target_hidden_start))
                return false;
        }
        return true;
    };
    static const int kDraftVocab = []{
        const char* e = getenv("SPARKINFER_DFLASH_DRAFT_VOCAB"); return e ? atoi(e) : 65536;
    }();
    const int depth = std::min(c.block_size, proposals > 0 ? std::min(proposals, 15) : 5);
    const int BW = s.width_for(depth);
    const int rows = n * BW;
    int ctx_total = 0;
    std::vector<int> ctx_off(n), past(n);
    auto kc = [&](int id, int L) { return id == s.kv_cur ? s.k_cache[L] : s.kv_states[id].k[L]; };
    auto vc = [&](int id, int L) { return id == s.kv_cur ? s.v_cache[L] : s.kv_states[id].v[L]; };
    auto sl = [&](int id) -> int& { return id == s.kv_cur ? s.seq_len : s.kv_states[id].seq_len; };
    bool ok;
    if (mode < 0) {
        ok = s.multi_plan(n, seg, proposals, past, ctx_off, ctx_total) && s.multi_alloc();
    } else {
        // The leader's choice; its checks covered these sessions' states, which are mirrored.
        ok = mode == 1;
        for (int j = 0; ok && j < n; j++) {
            past[j] = sl(seg[j].state);
            ctx_off[j] = ctx_total;
            ctx_total += seg[j].ctx_len;
        }
    }
    if (!ok) {
        static int noted = 0;
        if (getenv("SPARKINFER_DFLASH_MULTI_DEBUG") && n >= 2 && noted++ < 5)
            fprintf(stderr, "[dflash] forward_blocks: one by one (n=%d BW=%d depth=%d xq81=%d fc=%d "
                            "headq8=%d head_t=%d i4=%d i8=%d mk=%d win=%d alloc=%d ctx=%d)\n",
                    n, BW, depth, s.xq81 != nullptr, s.q8_fc.q4 != nullptr, s.head_q8 != nullptr,
                    s.lm_head_type, s.lm_head_i4 != nullptr, s.lm_head_i8 != nullptr,
                    s.markov_w2_q != nullptr, c.sliding_window, s.m_ready, ctx_total);
        return one_by_one();
    }

    {
        static long taken = 0;
        if (lead && getenv("SPARKINFER_DFLASH_MULTI_DEBUG") && (++taken & (taken - 1)) == 0)
            fprintf(stderr, "[dflash] forward_blocks: batched x%ld (n=%d rows=%d)\n", taken, n, n * BW);
    }
    cudaStream_t st = s.tp_stream ? s.tp_stream : (stream ? stream : s.stream);
    const int H = c.hidden, I = c.intermediate, B = c.block_size;
    const int n_cap = (int)c.target_layer_ids.size();
    const int qdim = c.n_q_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim, d = c.head_dim;
    const float scale = 1.f / sqrtf((float)d);
    const int V = s.vocab > 0 ? s.vocab : c.vocab;
    const int Vd = (kDraftVocab > 0 && kDraftVocab < V) ? kDraftVocab : V;
    auto q81 = [&](const bf16* src, int kk, int nr) {
        kernels::launch_quantize_q8_1_rows(src, s.m_xq81, kk, nr, kk, st);
        return s.m_xq81;
    };
    auto proj = [&](const void* xq, const Q8W& w0, const Q8W* w1, const Q8W* w2, void* y0,
                    void* y1, void* y2, int N0, int N1, int N2, int K, int nr) {
        dflash_kernels::launch_gemv_batched_q4_dp4a_fused3_rows(
            xq, w0.q4, w1 ? w1->q4 : nullptr, w2 ? w2->q4 : nullptr, w0.dm,
            w1 ? w1->dm : nullptr, w2 ? w2->dm : nullptr, y0, y1, y2, N0, N1, N2, K, st, nr);
    };

    // Block ids and their embedding.
    for (int j = 0; j < n; j++)
        for (int i = 0; i < BW; i++) s.m_h_ids[j * BW + i] = seg[j].ids[i];
    cu(cudaMemcpyAsync(s.m_d_ids, s.m_h_ids, (size_t)rows * sizeof(int), cudaMemcpyHostToDevice, st),
       "multi ids");
    if (!lead) {
        // Rank 1: zeros; rank 0's embedding and projected context arrive in the sums below.
        cu(cudaMemsetAsync(s.m_noise, 0, (size_t)rows * H * sizeof(bf16), st), "split noise zero");
        cu(cudaMemsetAsync(s.m_tp, 0, (size_t)ctx_total * H * sizeof(bf16), st), "split ctx zero");
    } else if (s.embed_rows > 0) {
        kernels::launch_embedding_vocab_window(s.m_d_ids, s.embed, s.m_noise, rows, H, 0,
                                               s.embed_rows, st);
        int dev = 0;
        cu(cudaGetDevice(&dev), "embed split device");
        for (int r = 0; r < rows; r++) {
            const int id = s.m_h_ids[r];
            if (id < s.embed_rows) continue;
            const bf16* src = static_cast<const bf16*>(s.embed_hi) + (size_t)(id - s.embed_rows) * H;
            bf16* dst = s.m_noise + (size_t)r * H;
            auto it = s.embed_hi_cache.find(id);
            if (it == s.embed_hi_cache.end() && s.embed_hi_cache.size() < 4096) {
                bf16* row = s.alloc<bf16>((size_t)H);
                if (row) {
                    cu(cudaMemcpyPeer(row, dev, src, s.embed_hi_dev, (size_t)H * sizeof(bf16)),
                       "embed split fetch");
                    it = s.embed_hi_cache.emplace(id, row).first;
                }
            }
            if (it != s.embed_hi_cache.end())
                cu(cudaMemcpyAsync(dst, it->second, (size_t)H * sizeof(bf16),
                                   cudaMemcpyDeviceToDevice, st), "embed split row");
            else
                cu(cudaMemcpyPeerAsync(dst, dev, src, s.embed_hi_dev, (size_t)H * sizeof(bf16), st),
                   "embed split peer row");
        }
    } else {
        kernels::launch_embedding(s.m_d_ids, s.embed, s.m_noise, rows, H, st);
    }

    // The sessions' new context rows, concatenated, through fc + hidden_norm.
    // fc split (see Impl::fc_split): each rank projects its half of the capture columns and the
    // sum below completes the projection on both cards; hidden_norm then runs on each.
    if (lead || s.fc_split) {
        const int fk = s.fc_k;
        for (int j = 0; j < n; j++) {
            bf16* dst = s.m_th + (size_t)ctx_off[j] * fk;
            const size_t bytes = (size_t)seg[j].ctx_len * fk * sizeof(bf16);
            if (seg[j].target_hidden)
                cu(cudaMemcpyAsync(dst, seg[j].target_hidden, bytes, cudaMemcpyDeviceToDevice, st),
                   "multi ctx gather");
            else {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    fprintf(stderr, "[dflash] fc split: no capture rows on rank %d (batched)\n",
                            c.tp_rank);
                }
                cu(cudaMemsetAsync(dst, 0, bytes, st), "multi ctx half zero");
            }
        }
        if (s.nv_on)
            s.nv_proj(s.m_th, nullptr, ctx_total, fk, s.nv_fc, s.m_tp, H, nullptr, nullptr, 0,
                      nullptr, nullptr, 0, st);
        else
            proj(q81(s.m_th, fk, ctx_total), s.q8_fc, nullptr, nullptr, s.m_tp, nullptr, nullptr,
                 H, 0, 0, fk, ctx_total);
        if (!s.fc_split)
            dflash_kernels::launch_rms(s.m_tp, s.hidden_norm, s.m_tp, ctx_total, H, c.rms_eps, st);
    }
    if (split) {
        // Both ranks now hold rank 0's rows bit for bit (x + 0 is exact).
        tp_allreduce_bf16_on(s.m_noise, (size_t)rows * H, st);
        tp_allreduce_bf16_on(s.m_tp, (size_t)ctx_total * H, st);
    }
    if (s.fc_split)
        dflash_kernels::launch_rms(s.m_tp, s.hidden_norm, s.m_tp, ctx_total, H, c.rms_eps, st);
    cu(cudaMemcpyAsync(s.m_x, s.m_noise, (size_t)rows * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
       "multi noise->x");

    static int mixed_causal = [] {
        const char* e = getenv("SPARKINFER_DFLASH_MIXED_CAUSAL"); return (!e || e[0] != '0') ? 1 : 0;
    }();
    static const int kForceCausal = []{
        const char* e = getenv("SPARKINFER_DFLASH_FORCE_CAUSAL"); return (e && e[0] == '1') ? 1 : 0;
    }();
    bool xn_ready = false;
    for (int L = 0; L < c.n_layers; L++) {
        const auto& w = s.layers[L];
        if (L == 0) dflash_kernels::launch_rms(s.m_x, w.input_norm, s.m_xn, rows, H, c.rms_eps, st);
        if (s.nv_on) {
            s.nv_proj(s.m_xn, xn_ready ? s.m_xq81 : q81(s.m_xn, H, rows), rows, H, w.nv_wq, s.m_q,
                      qdim, w.nv_wk, s.m_knew, kvdim, w.nv_wv, s.m_vnew, kvdim, st);
            s.nv_proj(s.m_tp, nullptr, ctx_total, H, w.nv_wk, s.m_kctx, kvdim, w.nv_wv, s.m_vctx,
                      kvdim, nullptr, nullptr, 0, st);
        } else {
            proj(xn_ready ? s.m_xq81 : q81(s.m_xn, H, rows), w.q8_wq, &w.q8_wk, &w.q8_wv, s.m_q,
                 s.m_knew, s.m_vnew, qdim, kvdim, kvdim, H, rows);
            proj(q81(s.m_tp, H, ctx_total), w.q8_wk, &w.q8_wv, nullptr, s.m_kctx, s.m_vctx,
                 nullptr, kvdim, kvdim, 0, H, ctx_total);
        }
        const bool causal = kForceCausal ||
                            (mixed_causal && L < (int)c.sliding_layers.size() && c.sliding_layers[L]);
        for (int j = 0; j < n; j++) {
            const DraftSeg& g = seg[j];
            bf16* kd = kc(g.state, L) + (size_t)past[j] * kvdim;
            bf16* vd = vc(g.state, L) + (size_t)past[j] * kvdim;
            const size_t cb = (size_t)g.ctx_len * kvdim * sizeof(bf16);
            const size_t nbytes = (size_t)BW * kvdim * sizeof(bf16);
            cu(cudaMemcpyAsync(kd, s.m_kctx + (size_t)ctx_off[j] * kvdim, cb, cudaMemcpyDeviceToDevice, st), "multi kctx");
            cu(cudaMemcpyAsync(vd, s.m_vctx + (size_t)ctx_off[j] * kvdim, cb, cudaMemcpyDeviceToDevice, st), "multi vctx");
            cu(cudaMemcpyAsync(kd + (size_t)g.ctx_len * kvdim, s.m_knew + (size_t)j * BW * kvdim, nbytes,
                               cudaMemcpyDeviceToDevice, st), "multi knew");
            cu(cudaMemcpyAsync(vd + (size_t)g.ctx_len * kvdim, s.m_vnew + (size_t)j * BW * kvdim, nbytes,
                               cudaMemcpyDeviceToDevice, st), "multi vnew");
            bf16* qj = s.m_q + (size_t)j * BW * qdim;
            const int new_len = g.ctx_len + BW;
            if (c.rope_normal) {
                dflash_kernels::launch_rms_heads_rope_normal(qj, w.q_norm, BW, c.n_q_heads, d,
                                                             c.rms_eps, g.pos0, c.rope_theta, st);
                dflash_kernels::launch_rms_heads_rope_normal(kd, w.k_norm, new_len, c.n_kv_heads, d,
                                                             c.rms_eps, g.pos0 - g.ctx_len,
                                                             c.rope_theta, st);
            } else {
                dflash_kernels::launch_rms_heads_rope(qj, w.q_norm, BW, c.n_q_heads, d, c.rms_eps,
                                                     g.pos0, c.rope_theta, st,
                                                     s.d_yarn_inv_freq, s.yarn_att_scale);
                dflash_kernels::launch_rms_heads_rope(kd, w.k_norm, new_len, c.n_kv_heads, d,
                                                     c.rms_eps, g.pos0 - g.ctx_len, c.rope_theta, st,
                                                     s.d_yarn_inv_freq, s.yarn_att_scale);
            }
            dflash_kernels::launch_attn_gqa(qj, kc(g.state, L), vc(g.state, L),
                                            s.m_attn + (size_t)j * BW * qdim, BW,
                                            past[j] + new_len, c.n_q_heads, c.n_kv_heads, d,
                                            g.pos0, /*k_pos0_cache=*/0, /*window=*/0, causal, scale,
                                            st, s.fa_m, s.fa_l, s.fa_acc);
        }
        if (s.nv_on)
            s.nv_proj(s.m_attn, q81(s.m_attn, qdim, rows), rows, qdim, w.nv_wo, s.m_ao, H, nullptr,
                      nullptr, 0, nullptr, nullptr, 0, st);
        else
            proj(q81(s.m_attn, qdim, rows), w.q8_wo, nullptr, nullptr, s.m_ao, nullptr, nullptr,
                 H, 0, 0, qdim, rows);
        if (split) tp_allreduce_bf16_on(s.m_ao, (size_t)rows * H, st);
        dflash_kernels::launch_add_rms(s.m_x, s.m_ao, s.m_h, w.post_norm, s.m_hn, rows, H,
                                       c.rms_eps, st, s.m_xq81);
        if (s.nv_on) {
            s.nv_proj(s.m_hn, s.m_xq81, rows, H, w.nv_gate, s.m_gate, I, w.nv_up, s.m_up, I,
                      nullptr, nullptr, 0, st);
        } else {
            proj(s.m_xq81, w.q8_gate, &w.q8_up, nullptr, s.m_gate, s.m_up, nullptr, I, I, 0, H, rows);
        }
        dflash_kernels::launch_swiglu(s.m_gate, s.m_up, s.m_gate, rows * I, st);
        if (s.nv_on)
            s.nv_proj(s.m_gate, q81(s.m_gate, I, rows), rows, I, w.nv_down, s.m_down, H, nullptr,
                      nullptr, 0, nullptr, nullptr, 0, st);
        else
            proj(q81(s.m_gate, I, rows), w.q8_down, nullptr, nullptr, s.m_down, nullptr, nullptr,
                 H, 0, 0, I, rows);
        if (split) tp_allreduce_bf16_on(s.m_down, (size_t)rows * H, st);
        const bf16* next_norm = (L + 1 < c.n_layers) ? s.layers[L + 1].input_norm : s.final_norm;
        dflash_kernels::launch_add_rms(s.m_h, s.m_down, s.m_x, next_norm, s.m_xn, rows, H,
                                       c.rms_eps, st, s.m_xq81);
        xn_ready = true;
    }

    if (!lead) {
        for (int j = 0; j < n; j++) {
            int& len = sl(seg[j].state);
            len = std::min(seg[j].pos0, past[j] + seg[j].ctx_len + BW);
        }
        return true;
    }
    // Head over each session's proposal rows (row r - 1 backs proposal r), then the Markov bias
    // and argmax per proposal, chained per session.
    const size_t q8row = kernels::llama_q8_1_bytes(H);
    for (int j = 0; j < n; j++)
        kernels::launch_quantize_q8_1_rows(s.m_xn + (size_t)j * BW * H,
                                           s.m_head_q8 + (size_t)j * depth * q8row, H, depth, H, st);
    const int hrows = n * depth;
    bool head_tc = false;
    if (hrows <= Impl::kMultiRows - 7 && s.head_fp4_ready(Vd)) {
        bool ok = true;
        for (int j = 0; ok && j < n; j++)
            ok = cudaMemcpyAsync(s.hf_x + (size_t)j * depth * H, s.m_xn + (size_t)j * BW * H,
                                 (size_t)depth * H * sizeof(bf16), cudaMemcpyDeviceToDevice,
                                 st) == cudaSuccess;
        head_tc = ok && s.head_fp4(s.hf_x, hrows, s.m_logits, Vd, st);
    }
    for (int r0 = 0; !head_tc && r0 < hrows; r0 += 16) {
        const int m = std::min(16, hrows - r0);
        if (!kernels::launch_gemv_q4k_dp4a_multirow_f32(s.m_head_q8 + (size_t)r0 * q8row, s.lm_head,
                                                        s.m_logits + (size_t)r0 * Vd, Vd, H, m, st))
            return false;
    }
    for (int j = 0; j < n; j++) {
        int* dout = s.m_d_out + (size_t)j * (B + 1);
        for (int r = 1; r <= depth; r++) {
            const int* prev = (r == 1) ? s.m_d_ids + (size_t)j * BW : dout + (r - 1);
            float* lrow = s.m_logits + (size_t)(j * depth + r - 1) * Vd;
            dflash_kernels::launch_markov_bias_add_q8(s.markov_w1, s.markov_w2_q, s.markov_w2_s,
                                                      prev, lrow, Vd, s.markov_rank, st, nullptr);
            kernels::launch_argmax(lrow, dout + r, 1, Vd, st);
        }
    }
    cu(cudaMemcpyAsync(s.m_h_out, s.m_d_out, (size_t)n * (B + 1) * sizeof(int),
                       cudaMemcpyDeviceToHost, st), "multi argmax");
    cu(cudaStreamSynchronize(st), "multi draft sync");
    for (int j = 0; j < n; j++) {
        for (int r = 1; r <= depth; r++) seg[j].out_argmax[r] = s.m_h_out[j * (B + 1) + r];
        int& len = sl(seg[j].state);
        len = std::min(seg[j].pos0, past[j] + seg[j].ctx_len + BW);
    }
    return true;
}

} // namespace sparkinfer
