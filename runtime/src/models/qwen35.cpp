// Qwen single-sequence greedy decoder. Serves every Qwen-family checkpoint here (Qwen3.5-9B,
// Qwen3.6-35B-A3B, Qwen3.8-27B) plus Muse Glimmer; layer counts and FFN shape come from
// QwenConfig, so the shapes below are the pattern, not fixed numbers.
//
// Per token: embed -> [c.n_layers x Qwen layer] -> final RMSNorm -> LM head -> argmax.
// Qwen full-attention layer: RMSNorm -> Q/K/V -> per-head QK-norm -> RoPE ->
//             KV append -> GQA flash decode -> O-proj -> residual -> RMSNorm ->
//             FFN -> residual.
// The FFN is either routed top-k MoE (+ shared expert), or -- when c.dense_ffn is set, as it is
// for Qwen3.8-27B and Qwen3.5-9B -- a single SwiGLU, still dispatched through the MoE kernels
// with one expert.
// Qwen3.5/Qwen3.6/Qwen3.8 hybrid layers replace full attention with a single-token
// Gated DeltaNet recurrent update on the 3-of-4 linear-attention layers.
// Only the sampled id is copied to the host, which autoregressive greedy decoding fundamentally
// requires. Decode is NOT single-stream: besides the main `stream` there are side streams for
// concurrent K/V-side projections (stream_k/stream_v, forked around the GDN alpha/beta pair) and
// an L2-prefetch stream (stream_pf). All of them fork from and rejoin the main stream via events,
// so the ordering the math depends on is still a single chain -- but anything added here has to
// respect those forks, and anything captured into the decode graph has to capture them too.

#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/device_health.h"
#include <atomic>
#include <condition_variable>
#include <map>

#include <mutex>
#include <thread>
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/models/dflash_kernels.h"
#include "qwen35_prefill.h"
#include "sparkinfer/thermal_governor.h"
#include "sparkinfer/kv_ops.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/ternary_ptq1.h"
#include "sparkinfer/prism_hadamard.h"
#include "sparkinfer/gdn_v_regroup.h"
#include "sparkinfer/kernels/proj_requant.h"
#include "sparkinfer/kernels/hadamard.h"
#include "sparkinfer/kernels/ternary.h"
#include "sparkinfer/safetensors.h"
#include "sparkinfer/kernels/compressed_tensors.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/prefill.h"   // launch_prefill_swiglu, for the NVFP4 decode FFN
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/deterministic.h"
#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/quant.h"
#include "sparkinfer/kernels/qtype.h"
#include "sparkinfer/kernels/proj_requant.h"
#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/lmcache_bridge_client.h"
#include "sparkinfer/lmcache_staging.h"
#include "sparkinfer/tp_layout.hpp"
#include "sparkinfer/gpu_link.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <unordered_map>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <climits>
#include <limits>
#include <algorithm>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    // Record context-killing errors so the engine can refuse new work instead of
    // issuing more against a dead context (see device_health.h).
    const bool fatal = note_cuda_error(e);
    // Rate-limit: a lost context makes EVERY subsequent call fail, which produced
    // 21,535 identical lines in one burst and buried the first, real error.
    static std::atomic<int> logged{0};
    const int n = logged.fetch_add(1, std::memory_order_relaxed);
    // Fatal lines print too, but only the first few: once the context is gone every call in
    // flight fails the same way, and a stress run wrote 2.4 GB of them before it drained.
    static std::atomic<int> fatal_logged{0};
    if (n < 20 || (fatal && fatal_logged.fetch_add(1, std::memory_order_relaxed) < 8))
        fprintf(stderr, "[qwen35] %s: %s%s\n", what, cudaGetErrorString(e),
                fatal ? "  [CONTEXT LOST -- server will refuse further work]" : "");
    else if (n == 20)
        fprintf(stderr, "[qwen35] (further CUDA errors suppressed)\n");
}
// End a stream capture and instantiate it. A graph is handed back only if both steps succeeded;
// otherwise whatever was built is destroyed and both handles are left null, so no later replay,
// park or destroy can pass libcuda a graph that does not exist. Capture records without running,
// so on failure the captured step's own work has not happened -- the caller decides what that means.
inline bool finish_capture(cudaStream_t st, cudaGraph_t* graph, cudaGraphExec_t* exec, const char* what) {
    *graph = nullptr;
    *exec = nullptr;
    cudaGraph_t g = nullptr;
    cudaError_t e = cudaStreamEndCapture(st, &g);
    if (e != cudaSuccess || !g) {
        cu(e != cudaSuccess ? e : cudaErrorStreamCaptureInvalidated, what);
        if (g) cudaGraphDestroy(g);
        return false;
    }
    cudaGraphExec_t x = nullptr;
    e = cudaGraphInstantiate(&x, g, 0);
    if (e != cudaSuccess) {
        cu(e, what);
        cudaGraphDestroy(g);
        return false;
    }
    *graph = g;
    *exec = x;
    return true;
}
// SPARKINFER_MUSE_FUSE_TAIL=0 splits Muse Glimmer's sandwich-norm tail back into the original
// launch_norm_then_add + launch_rmsnorm pair (the two produce bit-identical output).
inline bool muse_fuse_tail() {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_MUSE_FUSE_TAIL");
        return !(e && e[0] == '0');
    }();
    return on;
}
using bf16 = unsigned short;

// float -> bf16 on the host. Truncating (not round-to-nearest-even), matching what the rest of
// this file's host-side conversions do; the vision embeddings it converts are already the output
// of a bf16 tower, so the low bits being dropped here carry no information.
inline bf16 f32_to_bf16_host(float f) {
    unsigned u; std::memcpy(&u, &f, sizeof(u));
    return (bf16)(u >> 16);
}

// forward_token() sentinel: the decode graph was enqueued but not yet collected. Never a valid
// token id, so it cannot be confused with a real argmax.
constexpr int kDFlashDeferred = INT_MIN;

// Qwen35Model::set_logit_bias's sparse (id,val) scratch cap -- sparkinfer's own implementation
// bound for scratch-buffer sizing, NOT an OpenAI-documented limit.
constexpr int kMaxLogitBiasEntries = 1024;

// launch_gguf_dequant only implements F32/F16/Q8_0/Q4_K/Q6_K. Reject anything
// else at load time so Q5_K (etc.) cannot silently fall through as F32.
bool ggml_dequant_supported(int ggml_type) {
    switch (ggml_type) {
        case 0:  // F32
        case 1:  // F16
        case 8:  // Q8_0
        case 12: // Q4_K
        case 13: // Q5_K (UD / dynamic quants mix this in)
        case 14: // Q6_K
        case 30: // BF16 (Ternary-Bonsai-2 keeps its GDN alpha/beta projections here)
            return true;
        default:
            return false;
    }
}

// PTQ1_0 (Ternary-Bonsai-2's 1.75-bit weights) has no kernel of its own yet, so it enters the
// runtime as Q4_K: each trit becomes a nibble either side of the zero point, at 0.34% RMS. Every
// upload goes through here rather than each call site testing the type, because a path that forgot
// would read 28-byte ternary blocks as 144-byte Q4_K ones and load whatever followed them.
struct HostBlocks {
    const void* data = nullptr;
    size_t bytes = 0;
    int ggml_type = 0;
    std::vector<uint8_t> converted;   // non-empty only when a transcode actually happened
};

HostBlocks host_blocks_for_upload(const GGUFTensor* t, const std::string& name) {
    HostBlocks hb;
    hb.data = t->data;
    hb.bytes = t->n_bytes;
    hb.ggml_type = t->ggml_type;
    if (t->ggml_type != kPtq1GgmlType) return hb;

    // Two 128-trit groups make one 256-element Q4_K superblock, so a row that is a multiple of 128
    // but not 256 would pair groups across the row boundary and shear every row after the first.
    if (t->dims[0] % kQ4KBlockElems != 0) {
        fprintf(stderr, "[gguf] %s: PTQ1_0 row of %ld is not a multiple of %d, cannot transcode\n",
                name.c_str(), t->dims[0], kQ4KBlockElems);
        hb.ggml_type = -1;   // fails ggml_dequant_supported at the call site
        return hb;
    }
    const size_t supers = (size_t)t->n_values / kQ4KBlockElems;
    hb.converted.resize(supers * kQ4KBlockBytes);
    ptq1_to_q4k(static_cast<const uint8_t*>(t->data), (size_t)t->n_values,
                         hb.converted.data());
    hb.data = hb.converted.data();
    hb.bytes = hb.converted.size();
    hb.ggml_type = 12;   // Q4_K
    return hb;
}

// Folds Ternary-Bonsai-2's Hadamard rotation into the weights at load time.
//
// The checkpoint stores every rotated weight row as R.W[o], where R = H.diag(s), and expects the
// runtime to rotate the activation entering each matmul. Un-rotating the rows instead -- W[o] =
// R^-1.(stored row), and R^-1 = R^T = diag(s).H -- is the same arithmetic moved to the other
// operand, and it leaves an ordinary model behind: no graph needs a rotation inserted before its
// matmuls, and every existing kernel applies unchanged. token_embd is not a special case here;
// it is a lookup whose rows were rotated for the same reason, so it un-rotates identically.
//
// The rotated rows come out dense and Gaussian-ish rather than ternary. Weights that live as
// quantized blocks are therefore refitted to Q4_K, which spends the ternary structure to buy
// correctness; a native ternary kernel that rotates activations instead is what keeps it.
struct UnrotateJob {
    const GGUFTensor* t = nullptr;
    const std::vector<int8_t>* sign = nullptr;
    long width = 0, rows = 0, block = 0;
    // Every rotated tensor, token_embd included, is stored as R.row and undoes with R^-1: measured
    // against the un-quantized checkpoint this model was derived from, that recovers the embedding
    // at cosine 0.89, which is the ternary quantization error and nothing else. The metadata's
    // inverse_weight_names names a RUNTIME obligation -- token_embd is the one tensor with no
    // input activation to rotate, so a runtime that rotates activations must instead apply R^-1 to
    // the row it reads -- not a different storage direction. Kept switchable because that
    // distinction is easy to get backwards and a wrong guess looks exactly like a bad format read.
    bool undo_with_forward = false;

    // prism.hadamard.gdn_v_grouped: wherever the GDN v axis is PRODUCED -- attn_qkv's v rows and
    // the whole of attn_gate -- this checkpoint stores its 48 v-heads transposed, as
    // [heads_per_group][groups] rather than the [groups][heads_per_group] the architecture reads
    // them in. ssm_out, which consumes v, is stored in ordinary order, so a loader that ignores
    // this feeds the gate and the values of one head to another and the layer's output collapses.
    long v_row0 = -1;        // first row of the v block, negative when the tensor has none
    long v_rows = 0;         // n_v_heads * head_dim
    long v_groups = 0;       // ssm.group_count
    long v_head_dim = 0;
};

// Destination row -> the row of the stored tensor that belongs there.
long unrotate_source_row(const UnrotateJob& j, long dst) {
    if (j.v_row0 < 0 || dst < j.v_row0 || dst >= j.v_row0 + j.v_rows) return dst;
    const long r = dst - j.v_row0;
    const long head = r / j.v_head_dim, off = r % j.v_head_dim;
    const long per_group = (j.v_rows / j.v_head_dim) / j.v_groups;
    return j.v_row0 + gdn_v_source_head(head, j.v_groups, per_group) * j.v_head_dim + off;
}

bool bonsai_rot_forward(const char* env, bool def) {
    const char* v = getenv(env);
    if (!v || !v[0]) return def;
    return v[0] == 'f' || v[0] == 'F';   // "fwd" applies R, anything else applies R^-1
}

// Marks the v block of a tensor that produces GDN v, so its heads get regrouped on the way in.
void unrotate_job_set_v_block(UnrotateJob& j, const std::string& name, const Qwen35Config& c) {
    const long head_dim = c.linear_head_dim, v_rows = (long)c.linear_v_heads * head_dim;
    if (head_dim <= 0 || c.linear_q_heads <= 0 || c.linear_v_heads <= 0) return;
    if (c.linear_v_heads % c.linear_q_heads != 0) return;   // no clean group split; leave it alone
    if (name.find(".attn_gate.weight") != std::string::npos) j.v_row0 = 0;
    else if (name.find(".attn_qkv.weight") != std::string::npos)
        j.v_row0 = 2 * (long)c.linear_q_heads * head_dim;   // q and k come first, then v
    else return;
    if (j.v_row0 + v_rows > j.rows) { j.v_row0 = -1; return; }
    j.v_rows = v_rows;
    j.v_groups = c.linear_q_heads;
    j.v_head_dim = head_dim;
}

// The same v-head transpose applies to everything else indexed by GDN v head: the per-head
// scalars ssm_a and ssm_dt.bias, and the alpha/beta projections that emit one value per head.
// None of those is ternary, so they never pass through the un-rotation; they are regrouped here.
// Leaving them alone pairs each head's decay and step size with another head's values.
void* upload_v_regrouped_bf16(const GGUFTensor* t, const std::string& name, long row0,
                              long heads, long groups, long rows_per_head) {
    if (heads <= 0 || groups <= 0 || heads % groups != 0 || rows_per_head <= 0) return nullptr;
    const long row_len = t->n_dims >= 2 ? t->dims[0] : 1;
    if (row_len <= 0 || t->n_values % row_len != 0) return nullptr;
    const long rows = t->n_values / row_len;
    if (row0 < 0 || row0 + heads * rows_per_head > rows) return nullptr;
    if (t->ggml_type != 0 && t->ggml_type != 30) {
        fprintf(stderr, "[bonsai] %s: cannot regroup ggml type %d\n", name.c_str(), t->ggml_type);
        return nullptr;
    }
    const long per_group = heads / groups;
    std::vector<uint16_t> host((size_t)t->n_values);
    const auto* raw = static_cast<const uint8_t*>(t->data);
    for (long d = 0; d < rows; ++d) {
        long src = d;
        if (d >= row0 && d < row0 + heads * rows_per_head) {
            const long r = d - row0, h = r / rows_per_head, off = r % rows_per_head;
            src = row0 + gdn_v_source_head(h, groups, per_group) * rows_per_head + off;
        }
        for (long i = 0; i < row_len; ++i) {
            uint16_t bits;
            if (t->ggml_type == 30) {
                std::memcpy(&bits, raw + ((size_t)src * row_len + i) * 2, 2);
            } else {
                float f;
                std::memcpy(&f, raw + ((size_t)src * row_len + i) * 4, 4);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                u += 0x7FFFu + ((u >> 16) & 1u);
                bits = (uint16_t)(u >> 16);
            }
            host[(size_t)d * row_len + i] = bits;
        }
    }
    void* dev = nullptr;
    if (cudaMalloc(&dev, host.size() * 2) != cudaSuccess) return nullptr;
    if (cudaMemcpy(dev, host.data(), host.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev);
        return nullptr;
    }
    return dev;
}

bool unrotate_job_init(UnrotateJob& j, const GGUFTensor* t, const std::string& name,
                       const std::vector<int8_t>& sign, long block, bool inverse_listed) {
    j.t = t; j.sign = &sign; j.block = block;
    j.undo_with_forward = inverse_listed ? bonsai_rot_forward("SPARKINFER_BONSAI_EMB_ROT", false)
                                         : bonsai_rot_forward("SPARKINFER_BONSAI_W_ROT", false);
    j.width = t->dims[0];
    if (j.width <= 0 || t->n_values % j.width != 0) return false;
    j.rows = t->n_values / j.width;
    if ((long)sign.size() != j.width || j.width % block != 0 || j.width % kPtq1BlockElems != 0) {
        fprintf(stderr, "[bonsai] %s: input width %ld does not fit sign vector / block / group\n",
                name.c_str(), j.width);
        return false;
    }
    return true;
}

// Decodes rows [r0, r0+nr) of a ternary tensor, un-rotates each, and writes them as bf16.
void unrotate_rows_to_bf16(const UnrotateJob& j, long r0, long nr, uint16_t* out) {
    const auto* src = static_cast<const uint8_t*>(j.t->data);
    const unsigned hw = std::max(1u, std::min(std::thread::hardware_concurrency(), 32u));
    auto worker = [&](long lo, long hi) {
        std::vector<float> scratch(j.width);
        for (long r = lo; r < hi; ++r) {
            const size_t blk = (size_t)unrotate_source_row(j, r0 + r) * j.width / kPtq1BlockElems;
            ptq1_dequant(src + blk * kPtq1BlockBytes, (size_t)j.width, scratch.data());
            if (j.undo_with_forward)
                hadamard_rotate_activation(scratch.data(), j.width, j.block, j.sign->data());
            else
                hadamard_unrotate_activation(scratch.data(), j.width, j.block, j.sign->data());
            uint16_t* dst = out + (size_t)r * j.width;
            for (long i = 0; i < j.width; ++i) {
                uint32_t bits;
                std::memcpy(&bits, &scratch[i], 4);
                bits += 0x7FFFu + ((bits >> 16) & 1u);   // round to nearest even on the way to bf16
                dst[i] = (uint16_t)(bits >> 16);
            }
        }
    };
    std::vector<std::thread> pool;
    const long per = (nr + hw - 1) / hw;
    for (unsigned k = 0; k < hw; ++k) {
        const long lo = std::min<long>(nr, (long)k * per), hi = std::min<long>(nr, lo + per);
        if (lo < hi) pool.emplace_back(worker, lo, hi);
    }
    for (auto& th : pool) th.join();
}

// Chunked so a 248k-row embedding never needs its whole bf16 expansion resident at once.
long unrotate_rows_per_chunk(const UnrotateJob& j) {
    return std::max<long>(1, (64L << 20) / (j.width * 2));
}

// Reports a CUDA failure against the tensor that caused it. Without this an error here surfaces
// much later as an unrelated tensor "missing", because every subsequent cudaMalloc inherits it.
bool unrotate_cuda_ok(const char* what, const std::string& name) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    fprintf(stderr, "[bonsai] %s: %s failed: %s\n", name.c_str(), what, cudaGetErrorString(e));
    return false;
}

// Un-rotated weight -> bf16 on device, for the embedding table.
void* unrotate_ternary_to_bf16(const UnrotateJob& j, const std::string& name) {
    void* dev = nullptr;
    if (cudaMalloc(&dev, (size_t)j.t->n_values * 2) != cudaSuccess) {
        unrotate_cuda_ok("bf16 alloc", name);
        return nullptr;
    }
    const long step = unrotate_rows_per_chunk(j);
    std::vector<uint16_t> host((size_t)std::min(j.rows, step) * j.width);
    for (long r0 = 0; r0 < j.rows; r0 += step) {
        const long nr = std::min(step, j.rows - r0);
        unrotate_rows_to_bf16(j, r0, nr, host.data());
        if (cudaMemcpy(static_cast<char*>(dev) + (size_t)r0 * j.width * 2, host.data(),
                       (size_t)nr * j.width * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
            unrotate_cuda_ok("bf16 upload", name);
            cudaFree(dev);
            return nullptr;
        }
    }
    return dev;
}

// Un-rotated weight -> Q4_K on device, for everything that stays quantized.
void* unrotate_ternary_to_q4k(const UnrotateJob& j, const std::string& name, cudaStream_t stream) {
    void* q4k = nullptr;
    const size_t q4k_bytes = (size_t)(j.t->n_values / kQ4KBlockElems) * kQ4KBlockBytes;
    if (cudaMalloc(&q4k, q4k_bytes) != cudaSuccess) {
        unrotate_cuda_ok("q4k alloc", name);
        return nullptr;
    }
    const long step = unrotate_rows_per_chunk(j);
    std::vector<uint16_t> host((size_t)std::min(j.rows, step) * j.width);
    void* dev_bf16 = nullptr;
    if (cudaMalloc(&dev_bf16, host.size() * 2) != cudaSuccess) {
        unrotate_cuda_ok("staging alloc", name);
        cudaFree(q4k);
        return nullptr;
    }

    bool ok = true;
    for (long r0 = 0; r0 < j.rows && ok; r0 += step) {
        const long nr = std::min(step, j.rows - r0);
        unrotate_rows_to_bf16(j, r0, nr, host.data());
        const long n_chunk = nr * j.width;
        if (cudaMemcpy(dev_bf16, host.data(), (size_t)n_chunk * 2,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            ok = unrotate_cuda_ok("staging upload", name);
            break;
        }
        auto* dst = static_cast<char*>(q4k) +
                    (size_t)(r0 * j.width / kQ4KBlockElems) * kQ4KBlockBytes;
        kernels::launch_proj_requant_q4k_lloyd(dev_bf16, dst, n_chunk, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) ok = unrotate_cuda_ok("q4k refit", name);
    }
    cudaFree(dev_bf16);
    if (!ok) { cudaFree(q4k); return nullptr; }
    return q4k;
}

long qwen_moe_meta_int(const GGUF& g, const std::string& key, long def) {
    const long missing = std::numeric_limits<long>::min();
    long v = g.meta_int("qwen35." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen35moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen3moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen3_5_moe." + key, missing);
    return v != missing ? v : def;
}

bool is_qwen35_or_qwen36_hybrid_moe(const GGUF& g) {
    const std::string name = g.meta_str("general.name");
    if (name.find("Qwen3.5-35B-A3B") != std::string::npos ||
        name.find("Qwen3.6-35B-A3B") != std::string::npos)
        return true;

    if (g.tensor("blk.0.attn_qkv.weight") != nullptr &&
        g.tensor("blk.3.attn_q.weight") != nullptr)
        return true;

    const GGUFTensor* emb = g.tensor("token_embd.weight");
    const long vocab = emb ? emb->dims[1] : qwen_moe_meta_int(g, "vocab_size", -1);
    const bool hybrid_tensor_layout =
        g.tensor("blk.0.attn_q.weight") == nullptr &&
        g.tensor("blk.3.attn_q.weight") != nullptr;
    return qwen_moe_meta_int(g, "block_count", -1) == 40 &&
           qwen_moe_meta_int(g, "embedding_length", -1) == 2048 &&
           qwen_moe_meta_int(g, "attention.head_count", -1) == 16 &&
           qwen_moe_meta_int(g, "attention.head_count_kv", -1) == 2 &&
           qwen_moe_meta_int(g, "attention.key_length", -1) == 256 &&
           qwen_moe_meta_int(g, "expert_count", -1) == 256 &&
           qwen_moe_meta_int(g, "expert_used_count", -1) == 8 &&
           qwen_moe_meta_int(g, "expert_feed_forward_length", -1) == 512 &&
           vocab == 248320 &&
           hybrid_tensor_layout;
}

bool is_linear_layer(const Qwen35Config& c, int layer) {
    return c.hybrid && c.full_attn_interval > 0 && ((layer + 1) % c.full_attn_interval) != 0;
}

// True when the stack actually CARRIES Gated-DeltaNet layers. `hybrid` alone does not say that:
// Muse Glimmer sets it to unlock the batched prefill and the attention-gate path while declaring
// full_attn_interval = 0, which makes is_linear_layer() false for every layer -- it has the flag
// and none of the layers. Anything that exists only to hold GDN state has to ask this, not
// `hybrid`, or it allocates a recurrent state for a stack with no recurrence.
bool has_linear_layers(const Qwen35Config& c) {
    if (!c.hybrid || c.full_attn_interval <= 0) return false;
    for (int L = 0; L < c.n_layers; ++L) if (is_linear_layer(c, L)) return true;
    return false;
}

// Whether a session on this stack has to carry Gated-DeltaNet state at all.
// SPARKINFER_MUSE_GDN_GATE=0 restores the old `cfg.hybrid` test, for an A/B out of one binary.
bool needs_linear_state(const Qwen35Config& c) {
    static const bool gate = [] {
        const char* e = getenv("SPARKINFER_MUSE_GDN_GATE");
        return !(e && e[0] == '0');
    }();
    return gate ? has_linear_layers(c) : c.hybrid;
}
}

struct SessionBuffers {
    float* lin_state = nullptr;
    bf16* lin_conv_state = nullptr;
    // This session's GDN state has been compacted to bf16 in the first half of lin_state (see
    // launch_qwen36_gdn_state_to_b16). Set only for sessions that have gone through
    // decode_packed; a speculative-decode session never has, and neither does session 0.
    bool lin_state_b16 = false;
    // Per-request running count of how many times each vocab id has appeared in THIS session's
    // generated completion so far -- for presence_penalty/frequency_penalty. Unlike lin_state/
    // lin_conv_state (hybrid-architecture-only), this exists for EVERY model. vocab-sized, alloc'd
    // once per session slot; see Qwen35Model::reset_penalty_counts for why it must ALSO be
    // explicitly re-zeroed once per REQUEST (not just once per session-slot creation) whenever a
    // session is reused across multiple requests (seq_id 0, the shared prefix session).
    int* penalty_counts = nullptr;
    // Per-request static per-vocab-id additive bias for logit_bias -- unlike penalty_counts (which
    // starts at zero and is incremented device-side every decode step), this is SET ONCE per
    // request (Qwen35Model::set_logit_bias) and stays constant for the rest of the request's
    // decode. Exists for EVERY model, same as penalty_counts, and needs the identical per-REQUEST
    // (not per-session-slot) re-zero-then-set discipline when a session is reused (seq_id 0).
    float* logit_bias = nullptr;
    // Whether this request's set_logit_bias had any entries. Host-side, so the prefill seed can
    // skip the bias pass entirely for the requests (nearly all) that never set one.
    bool logit_bias_set = false;
};

struct Qwen35Model::Impl {
    Qwen35Config cfg;
    KVCacheManager* kv;
    moe::MoEEngine* engine;
    // Dual-GPU Wave 3: which tp-table rank owns this instance's weights and which CUDA device
    // it is built on. (0,0) is the unsplit tp=1 default: the ctor never calls cudaSetDevice,
    // so every allocation and free is byte-identical to the historical single-device path.
    int rank = 0;
    int device = 0;
    Qwen35Weights w;
    cudaStream_t stream{};
    cudaStream_t stream_k{}, stream_v{};         // side streams for concurrent K/V projection
    cudaStream_t stream_pf{};                    // side stream for the L2 weight prefetch
    cudaEvent_t ev_pf_fork{}, ev_pf_done{};      // prefetch fork/join (kept inside the decode graph)
    cudaEvent_t ev_qkv{}, ev_k{}, ev_v{};        // fork/join events (captured into the decode graph)
    cudaEvent_t ev_pipe_fork{}, ev_gdn_z{}, ev_gdn_ab{};
    cudaEvent_t ev_sx_gate{}, ev_sx_done{};
    uint64_t active_seq_id = 0;
    std::atomic<uint64_t> next_session_id{1};
    std::unordered_map<uint64_t, SessionBuffers> sessions;
    int qdim, kvdim;
    int linear_qdim = 0, linear_vdim = 0, linear_qkvdim = 0;
    bool gguf = false;   // true after load_gguf: dense weights are native [out,in], use GEMV
    // Pending image input for the NEXT prefill_batched call, set by set_pending_vision and
    // cleared once consumed. Device-resident so prefill can splice without a host round trip.
    // Null on every text-only request, which is what keeps the vision path off the measured path.
    void* d_vision_emb = nullptr;
    int*  d_vision_pos = nullptr;
    int   vision_n = 0;
    // Interleaved-MRoPE rotary positions for the pending prompt: [n_tokens*3] int32 on device.
    // Null on every text-only request, which is what keeps the MRoPE kernel instantiation off the
    // measured path entirely.
    int*  d_mrope_pos = nullptr;
    int   mrope_n = 0;
    // Added to the DECODE rotary position, never to the cache slot. Vision spans advance the
    // rotary counter by less than their token count, so generated tokens sit at a lower rotary
    // position than their slot index; this is that difference, and it is 0 without vision.
    int   mrope_pos_offset = 0;
    // Guards the capture window against legacy-default-stream work issued by other threads.
    // See Qwen35Model::device_mutex() in the header for why this is required.
    std::recursive_mutex device_mu;
    // CUDA-graph capture of the decode compute (captured once, replayed each token)
    cudaGraph_t cu_graph{};
    cudaGraphExec_t cu_exec{};
    bool graph_ready = false;
    cudaGraph_t cu_prefill_graph{};
    cudaGraphExec_t cu_prefill_exec{};
    bool graph_prefill_ready = false;
    int graph_prefill_attn_mode = -1;
    bool bench_feedback_graph = false;
    int graph_attn_mode = -1;  // host-side flash-decode dispatch class captured in cu_graph
    bool graph_state_b16 = false;  // GDN state representation baked into cu_graph
    // Separate decode graph for DFlash verify (sample=true, dflash_capture=true). The normal
    // cu_graph/cu_exec can't be reused here because it carries the extra per-layer hidden
    // captures. Their destination row varies per call, which is handled the same way the graph
    // already handles token_id/position: cap_row is packed into d_scalars and read on device.
    cudaGraph_t cu_dflash_graph{};
    cudaGraphExec_t cu_dflash_exec{};
    bool dflash_graph_ready = false;
    int dflash_graph_attn_mode = -1;
    bool dflash_graph_sparse = false;
    // The NVFP4 LM-head operand, held here rather than in `owned` because it is releasable.
    void* lm_head_fp4_payload = nullptr;
    void* lm_head_fp4_sf_buf = nullptr;
    // Scratch for the one-off GDN state compaction; sized for one session's state, reused by all.
    void* gdn_state_stage = nullptr;
    // Mirrors sessions[active_seq_id].lin_state_b16, swapped by activate_session() exactly the way
    // lin_state/lin_conv_state are.
    bool active_lin_state_b16 = false;
    // Reusable device pointer arrays for packed decode (see Qwen35Model::decode_packed). Their
    // ADDRESSES are baked into the packed graph; their CONTENTS are rewritten every step, which
    // is what lets one graph per row count serve any set of sessions.
    void* packed_dev_states = nullptr;
    void* packed_dev_convs = nullptr;
    void* packed_dev_tables = nullptr;
    void* packed_dev_tables_win = nullptr;   // ring tables for windowed slices (see kv_cache.h)
    void* packed_host_tables_win = nullptr;
    // Pinned staging + the seq_ids the device arrays currently hold, so an unchanged row set
    // skips the upload entirely.
    void* packed_host_states = nullptr;
    void* packed_host_convs = nullptr;
    void* packed_host_tables = nullptr;
    void* packed_host_seqs = nullptr;
    int   packed_rows_valid = 0;

    // Per-session parking lot for the AR decode graph.
    //
    // activate_session() used to DESTROY cu_graph outright, because every node in it bakes this
    // session's device pointers -- lin_state / lin_conv_state / penalty_counts / logit_bias, and
    // kv->block_table(active_seq_id). Correct, but it made a session switch cost a full capture +
    // instantiate of the whole 64-layer decode, and the continuous-batch worker switches sessions
    // on EVERY token once two requests are in flight. Measured on RTX 5090 / Qwen3.8-27B-NVFP4,
    // that is ~1.09 ms per token: per-token wall time goes 13.07 ms at concurrency 1 to 14.16 ms
    // at concurrency 2, so aggregate throughput FALLS 84.9 -> 74.4 tok/s going 1 -> 2.
    //
    // Parking the outgoing session's graph under its own seq_id and restoring the incoming one
    // keeps exactly the same "a decode graph only ever runs for the session it was captured for"
    // guarantee, while making a switch a pointer swap.
    //
    // Safe because every pointer a decode node bakes is stable for the session's lifetime: the
    // per-session buffers are allocated once in open_session(), and block_table(seq) is
    // d_block_tables + slot * max_blocks_per_seq -- a fixed slab offset whose CONTENTS grow as
    // blocks are appended but whose ADDRESS does not. That is the same assumption the pre-existing
    // code already made by replaying one capture across a session's whole decode phase.
    //
    // n_splits is parked with the graph: the adaptive-splits check invalidates on
    // `want != s.n_splits`, so restoring a graph without restoring the value it was captured at
    // would tear it straight back down.
    struct ParkedDecodeGraph {
        cudaGraph_t graph{};
        cudaGraphExec_t exec{};
        int attn_mode = -1;
        bool sparse = false;
        int n_splits = 0;
        // Which GDN state representation the capture's kernels read. Parked with the graph for
        // the same reason n_splits is: a session parked on the fp32 form can be compacted while
        // it is parked, and restoring the graph without the value it was captured at would
        // replay fp32 kernels over a bf16 state.
        bool state_b16 = false;
    };
    std::unordered_map<uint64_t, ParkedDecodeGraph> parked_graphs;
    // Bounded purely as a leak backstop -- close_session() drops a session's entry, so in steady
    // state this holds one graph per LIVE session, which is the server's concurrency limit.
    static constexpr size_t kMaxParkedGraphs = 64;

    // scratch (bf16)
    bf16 *x, *xn, *q, *k, *v, *attn, *ao, *h, *hn, *routed, *shared;
    bf16 *qraw = nullptr, *qgate = nullptr;
    int tp_cur_pos = 0;        // (dual-GPU) forward_token_tp's decode position, read by tp_attn_layer_tp
    bf16 *dbg_xn_dump = nullptr;   // DEBUG ONLY (SPARKINFER_MG_STAGE_DEBUG): [2*n_layers+1, H] xn/final-norm/hn snapshot
    bf16 *lin_qkv = nullptr, *lin_q = nullptr, *lin_k = nullptr, *lin_v = nullptr;
    bf16 *lin_z = nullptr, *lin_alpha = nullptr, *lin_beta = nullptr;
    bf16 *lin_gdn = nullptr, *lin_norm = nullptr, *shared_gate_tmp = nullptr;
    bf16 *sh_gate = nullptr, *sh_up = nullptr, *sh_h = nullptr;   // shared-expert GEMV scratch [moe_ffn]
    bf16 *nvfp4_g = nullptr, *nvfp4_u = nullptr, *nvfp4_h = nullptr;  // native NVFP4 dense-FFN decode scratch
    bf16 *lin_conv_state = nullptr;
    float* lin_state = nullptr;
    // End-of-prefix snapshot of the hybrid recurrent state, taken by cache_prefix() and
    // replayed by restore_prefix_state(). The prefix's KV blocks can simply be kept, but
    // this state cannot: decoding mutates it in place.
    float* prefix_lin_state = nullptr;
    bf16*  prefix_lin_conv_state = nullptr;
    // "Current" session's penalty_counts (swapped by activate_session(), unconditionally, for
    // every model -- see SessionBuffers). penalty_counts_default backs session 0's entry, alloc'd
    // once at load time; sessions[seq_id].penalty_counts for every other session is alloc'd by
    // open_session().
    int* penalty_counts = nullptr;
    int* penalty_counts_default = nullptr;
    // "Current" session's logit_bias (swapped by activate_session(), unconditionally, for every
    // model -- mirrors penalty_counts exactly). logit_bias_default backs session 0's entry.
    float* logit_bias = nullptr;
    float* logit_bias_default = nullptr;
    bool logit_bias_set = false;   // sessions[active_seq_id].logit_bias_set, swapped with logit_bias
    // Transient scratch for Qwen35Model::set_logit_bias's sparse (id,val) -> device scatter. NOT
    // session-scoped (purely transient staging, safe to share across requests since submit_locked
    // -- the only caller -- always runs with the engine mutex held). Fixed size (kMaxLogitBiasEntries).
    int* h_logit_bias_ids = nullptr; float* h_logit_bias_vals = nullptr;
    int* d_logit_bias_ids = nullptr; float* d_logit_bias_vals = nullptr;
    // Pinned staging for set_logit_bias_dense (cfg.vocab floats), allocated on first use: only
    // constrained requests ever need it.
    float* h_dense_bias = nullptr;
    float* logits;
    int *d_scalars, *d_tok, *d_out_id, *d_pos, *d_seqlen, *d_writepos, *d_shared_ids;
    int *d_cap_row = nullptr;   // dflash capture row, packed into d_scalars[4]
    int *h_scalars = nullptr, *h_out_id = nullptr;
    // Temperature-sampling params, refreshed via cudaMemcpyAsync before every forward_token call
    // (same pattern as h_scalars/d_scalars above) so a captured decode graph can safely replay
    // across separate requests/sessions with different temperature/seed -- see fused.h /
    // launch_temperature_sample's doc comment for why these can't be plain kernel arguments.
    float* h_sample_temp = nullptr;
    unsigned long long *h_sample_seed = nullptr, *h_sample_step = nullptr;
    float* d_sample_temp = nullptr;
    unsigned long long *d_sample_seed = nullptr, *d_sample_step = nullptr;
    // top_k/top_p params, same refresh-before-every-call discipline as the sample params above.
    int* h_sample_top_k = nullptr;
    float* h_sample_top_p = nullptr;
    int* d_sample_top_k = nullptr;
    float* d_sample_top_p = nullptr;
    // presence_penalty/frequency_penalty params, same refresh-before-every-call discipline.
    float* h_sample_presence_penalty = nullptr;
    float* h_sample_frequency_penalty = nullptr;
    float* d_sample_presence_penalty = nullptr;
    float* d_sample_frequency_penalty = nullptr;
    // top_k/top_p truncation scratch: allocated ONCE at load time (vocab-sized), fixed address,
    // never reallocated per-call -- see kernels::launch_topk_topp_mask's doc comment for why this
    // must always process the full vocab regardless of top_k/top_p (CUB's num_items is a host
    // constant, not a per-call device value).
    int* d_vocab_iota = nullptr;
    float* d_sorted_logits = nullptr;
    int* d_sorted_idx = nullptr;
    float* d_topk_exp = nullptr;
    float* d_topk_cumsum = nullptr;
    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    // logprobs/top_logprobs scratch -- same alloc-once-at-load-time discipline as the topk/topp
    // block above. d_rank_by_id[id] = sorted rank of vocab entry `id` (inverse permutation,
    // scattered by topk_topp_exp_kernel); d_chosen_logit = the winning token's raw logit, written
    // by launch_extract_chosen_logit. See Qwen35Model::last_token_logprobs's doc comment.
    int* d_rank_by_id = nullptr;
    float* d_chosen_logit = nullptr;
    // NOTE: there is deliberately no device staging buffer for token_logprob_for()'s token id.
    // The decode graph bakes d_out_id into its captured launch_extract_chosen_logit node, so a
    // teacher-forced caller that wants some OTHER token's logit re-runs that kernel -- but it
    // passes the id BY VALUE (launch_extract_chosen_logit_id). Staging it through a device buffer
    // needs a host->device memcpy on the legacy default stream, which is unordered against
    // s.stream (cudaStreamNonBlocking): the kernel could read the previous call's id. See #1001.
    // Deterministic-mode softmax normalizer (see launch_logprob_denom_det). 256 partials + total.
    float* d_denom_partials = nullptr;
    float* d_denom_det = nullptr;
    float* d_shared_w;
    std::vector<void*> owned;   // device buffers from load_weights / load_gguf
    // GGUF fused-expert decode scratch (allocated by load_gguf)
    float *mf_logits = nullptr, *mf_weights = nullptr, *mf_h = nullptr, *mf_out = nullptr;
    // Decode-side NVFP4 FFN scratch (the default path; SPARKINFER_QWEN38_DECODE_NVFP4=0 opts out):
    // gate and up outputs, and
    // the SwiGLU result that feeds down. bf16 [moe_ffn] each -- ~104 KB total at ffn=17408, so it
    // is allocated unconditionally rather than gated, keeping the decode path branch-free.
    bf16 *nv_gate = nullptr, *nv_up = nullptr, *nv_h = nullptr;
    // int8 staging for the dp4a NVFP4 FFN: one quantize of the layer input feeds gate AND up,
    // a second feeds down. Sized for the widest K the FFN reads (moe_ffn for down).
    signed char *nv_xq = nullptr;
    float *nv_xs = nullptr;
    // Projection-side int8 staging (attention + GDN), separate from the FFN's. `a` holds the
    // layer input xn, shared by every projection that reads it; `b` holds one-off inputs
    // (o_proj's att, ssm_out's lnrm) so quantizing those cannot clobber xn while the parallel
    // K/V and GDN streams are still reading it.
    signed char *nv_pq_a = nullptr, *nv_pq_b = nullptr;
    float *nv_ps_a = nullptr, *nv_ps_b = nullptr;
    float *sx_h = nullptr;   // pipelined shared-expert h_scratch (avoids racing routed mf_h)
    void  *sx_q8 = nullptr;  // pipelined shared-expert Q8_1(h) for down (avoids racing aq81)
    int   *mf_ids = nullptr, *mf_counts = nullptr;
    unsigned int *mf_rc = nullptr;   // fused-router grid-completion counter (persistent, zero-init)
    // Per-row int8 scales of the routed expert weights, for the batched prefill's fused
    // quantized-B MoE GEMM (prefill_moe_q.cu). Laid out [layer][expert * rows]; populated
    // EAGERLY here at load, never lazily -- the scored sweep times each context exactly once
    // (512 first), so a lazy fill would land inside the very pass it is meant to speed up.
    float *moe_rs_gate = nullptr, *moe_rs_up = nullptr, *moe_rs_down = nullptr;
    // Muse Glimmer dense prefill: one pool of per-output-row int8 scales for the native Q4_K/Q5_K
    // attn + FFN gate/up weights (Qwen35LayerWeights::*_rs point into it). See load().
    float *muse_rs = nullptr;
    // flash-decoding (KV-split) attention partials
    static constexpr int MAX_NSPLITS = 256;   // partials sized for this; adaptive n_splits <= this
    int n_splits = 32;
    bool adaptive_splits = true;              // scale n_splits with seq_len (decode graph re-captured on change)
    int split_chunk = 256;                    // target serial KV per split (SPARKINFER_SPLIT_CHUNK)
    float *fa_m = nullptr, *fa_l = nullptr, *fa_acc = nullptr;
    // Sink + sliding-window sparse-KV. Default on; SPARKINFER_SPARSE_KV=0 disables. Per-kv_head block list.
    int*   sparse_sel = nullptr;
    int    sparse_budget = 0;      // max sel slots = 1 + window
    int    sparse_window = 256;    // recent window in KV blocks (16 tokens/block)
    int    sparse_min_ctx = 8192;
    bool   graph_sparse = false;
    // GQA-8 (Qwen3.6 full-attn) sparse rides the dense int8-MMA kernel over a compacted
    // paged-KV view instead of a dedicated sparse walker (issue #559). Decode steps only.
    int*   sparse_vtbl = nullptr;  // compact view block table [sparse_budget]
    int*   sparse_vlen = nullptr;  // compact view seq_len (device scalar)
    int    sparse_vsplits = 128;   // KV splits over the view (sized so MMA chunks >= 2 blocks)
    // Muse Glimmer: pure sliding-window compact view for swa-flagged layers (no sink,
    // mandatory every step -- see fa_kv_compact_view_pure). Built once per decode step,
    // shared by every swa layer that step; global (non-swa) layers use the full btable.
    int*   swa_vtbl = nullptr;     // compact view block table [swa_budget]
    int*   swa_vlen = nullptr;     // compact view seq_len (device scalar)
    int    swa_budget = 0;         // sliding_window tokens / block_size, rounded up
    int    swa_vsplits = 32;       // KV splits over the (small, fixed-size) swa view
    bf16*  emb_norm_ones = nullptr; // [hidden] all-1.0, for the unweighted post-embedding RMSNorm
    // pre-quantized Q8_1 activation (computed once per projection input, shared across Q/K/V)
    signed char* aq8 = nullptr; float *aq8_d = nullptr, *aq8_s = nullptr;
    bool use_pq = true;   // SPARKINFER_PQ=0 disables the pre-quantized GEMV path
    void* aq81 = nullptr; // block_q8_1 activation for the faithful llama mmvq port
    bool use_llama = true; // default ON: faithful llama mmvq for Q4_K attn GEMVs (+9.7%, top1 0.99). =0 disables
    bool use_q6mmvq = true;  // default ON: int8 Q6_K mmvq for attn-V upgrades + LM head. =0 disables
    bool use_qkvstream = true; // default ON: run Q/K/V projections on concurrent streams. =0 disables
    bool use_qkfuse = true;// default ON: fused per-head Q-norm + K-norm (1 kernel). =0 disables
    bool use_ropekv = true;// default ON: fused RoPE + KV-append (1 kernel vs 2). =0 disables
    bool use_attnin = true;// default ON: single fused QK-norm+RoPE+KV-append (1 kernel vs qkfuse+ropekv=2). =0 disables
    bool use_fnq = true;   // default ON: post-MoE add_rmsnorm2 also emits Q8_1(xn), deleting the
                           // next layer's standalone QKV-input quantize node. =0 disables
    bool use_gdn_pipe = true;   // default ON: overlap GDN gate/scalar projections on side streams. =0 disables
    bool use_gdn_quad = false;  // default OFF: one-grid GDN Q4_K quad (H=2048). =1 enables
    bool use_attn_qkv = true;   // default ON: one-grid full-attn QKV MMVQ (Q4_K, H=2048). =0 disables
    bool use_shexp_pipe = true; // default ON: overlap shared expert with routed MoE. =0 disables
    bool use_addnorm3 = true;   // default ON: fold routed+shared residual_add into post-MoE add_rmsnorm. =0 disables
    bool use_router_fused = true; // default ON (256-expert path): fuse the router GEMV + bitonic top-k
                                  // into one kernel (grid-completion), dropping the top-k launch. =0 disables

    // Prefix KV reuse (Genie-style warm prompt): cache_prefix() retains KV + GDN state.
    std::vector<int> prefix_tokens;
    int prefix_len = 0;
    int prefix_next = -1;
    bool prefix_active = false;

    // Optional external KV cache tier (docs/lmcache_bridge_protocol.md). Null = disabled (the
    // default) -- every lookup/store call site below is a no-op when this is null, so nothing
    // about existing behavior changes unless a caller explicitly opts in via
    // set_lmcache_bridge().
    BridgeClient* lmcache_bridge = nullptr;

    // Ternary-Bonsai-2's rotated basis, kept on the device for the native PTQ1_0 path: the
    // weights stay in their 28-byte blocks and the ACTIVATION carries the rotation instead of
    // the weights carrying its inverse. Empty unless the checkpoint declares prism.hadamard and
    // SPARKINFER_BONSAI_NATIVE is on; the folded path needs none of this.
    std::unordered_map<long, void*> bonsai_sign_dev;   // input width -> int8[width] on device
    long bonsai_block = 0;
    bf16* bonsai_rot = nullptr;                        // scratch for one rotated activation
    long bonsai_rot_elems = 0;
    bool bonsai_embed_native = false;                  // token_embd left in its ternary blocks
    // The layer's normed input, rotated once on the main stream before the projections fan out
    // across stream_k/stream_v. Rotating inside each projection would race: they run concurrently
    // and would share one scratch. Written where s.xn is, so s.xn's own visibility carries it.
    bf16* bonsai_rot_xn = nullptr;
    // The dense FFN read natively: its input is the POST-ATTENTION norm, not xn, so it needs its
    // own rotation -- reusing bonsai_rot_xn would feed gate/up the wrong activation. `ffn_h` holds
    // SwiGLU's output and is rotated in place for the down projection, whose input width is the
    // FFN width rather than the residual one (17408 against 5120, a different sign vector).
    bf16* bonsai_rot_hn = nullptr;
    bf16* bonsai_ffn_gate = nullptr;
    bf16* bonsai_ffn_up = nullptr;
    bf16* bonsai_ffn_h = nullptr;
    // Resolved once at load rather than looked up per layer per token.
    const signed char* bonsai_sign_h = nullptr;     // int8[hidden]
    const signed char* bonsai_sign_ffn = nullptr;   // int8[moe_ffn]
    // Decode's own view of the layers (SPARKINFER_BONSAI_DECODE_SHADOW): the folded Q4_K weights
    // stay resident for prefill and the packed batch, and a second, ternary copy of the same
    // tensors -- 0.21875 bytes/weight against Q4_K's 0.5625 -- is what a single-row decode step
    // reads. Empty when off; forward_token then reads s.w.layers as before.
    std::vector<Qwen35LayerWeights> bonsai_dec_layers;
    const void* bonsai_dec_head = nullptr;
    // Its allocations, held here rather than in `owned` because it is releasable: the copy is
    // ~5.5 GB, and concurrent requests need that VRAM more (each carries ~151 MB of GDN state).
    std::vector<void*> bonsai_dec_bufs;

    // DFlash speculative decoding (target-side primitives).
    DFlashDraftModel* dflash_draft = nullptr;
    // Preserve the native Q6_K head for the draft's multi-row MMVQ while the
    // target uses its faster, narrower Q4_K requantized copy.
    const void* dflash_lm_head = nullptr;
    int dflash_lm_head_type = 0;
    bool dflash_capture = false;
    // DFlash verify token 0: enqueue the captured decode graph and collect it later, so the
    // draft block can be issued in between (see dflash_generate).
    bool defer_decode_sync = false;
    bool decode_pending = false;
    std::vector<int> dflash_layer_ids;
    int dflash_n_cap = 0;
    int dflash_max_rows = 16;
    int dflash_cap_row = 0;
    int final_seqlen_hint = -1;   // set by generate()/dflash_generate() before their prefill loop
    int dflash_ctx_len = 0;
    int dflash_ctx_cap = 0;
    int dflash_ctx_start = 0;
    bf16* dflash_hidden = nullptr;    // [max_rows, n_cap * H]
    bf16* dflash_context = nullptr;   // [ctx_cap, n_cap * H]
    // (dual-GPU) Split capture (set_dflash_capture_split): this card stores columns
    // [dflash_cap_off, dflash_cap_off + cap_h()) of each captured layer, so both buffers above
    // are n_cap * cap_h() wide. 0 / 0 = the whole row (tp=1, or an unsplit draft).
    int dflash_cap_off = 0;
    int dflash_cap_h = 0;
    int cap_h() const { return dflash_cap_h > 0 ? dflash_cap_h : cfg.hidden; }
    // Rank 0 only: its split-capture buffers and their rank-1 twins (same size, same layout).
    struct CapAlias { const char* lead; size_t bytes; char* peer; };
    std::vector<CapAlias> cap_alias;
    mutable std::mutex cap_alias_mu;
    // Rank 1 only, while the capture is split: the leader, whose dflash_cap_peer maps its pointers.
    Qwen35Model* cap_lead = nullptr;
    float* spec_lin_snap = nullptr;
    bf16* spec_conv_snap = nullptr;
    // GDN v-head state window (dual-GPU state split, see GdnStateWindow): (0,0) = all heads =
    // the unsplit, byte-identical tp=1 model.
    GdnStateWindow gdn_window{};
    // Full-layout GDN state round-trip arena [slots][v_full][HD][HD] for the prefill scan and the
    // dflash verify/commit paths, which only have full-v kernels: they inject the windowed
    // per-slot region into it, run, and extract it back. Allocated only when a window is set;
    // the degenerate window uses the dense arenas directly, the way the code did before.
    float* gdn_scratch = nullptr;

    // (dual-GPU WP-9) tp=2 scratch, allocated per-model by tp_attach on this model's OWN
    // device (never shared across ranks). tp_xrow is the op-entry embedding-exchange row;
    // tp_ar is the per-layer AR-A/AR-B staging (both all-reduces reuse row 0); the decode
    // epilogue all-reduces the ordinary s.logits row (tp_allreduce_logits). All null until
    // tp_attach, so the tp=1 path never allocates or touches any of them.
    GpuLink* tp_link = nullptr;
    int tp_rank = 0;
    std::vector<Qwen35Model*> tp_peers;   // rank-ordered 2-slot world ([my_rank]=this)
    bf16* tp_xrow = nullptr;   // [hidden]
    bf16* tp_ar   = nullptr;   // [32][2*hidden]
    // GDN tp staging: one N-windowed qkv GEMV (rows [ql q | ql k | vloc v] of the rank's
    // compact weight blob) lands here, then three D2D copies scatter the q/k/v windows into
    // the full-width packed lin_qkv. Null unless tp_attach allocated it (hybrid models).
    bf16* tp_qkv = nullptr;
    // Dense-FFN tp staging (dense_ffn models): one N-windowed gate GEMV and one N-windowed up
    // GEMV (each fl = moe_ffn/2 rows on 27B) land in the first two quarters, the swiglu over
    // the rank's gate/up window in the last; the down K-windowed GEMV reads exactly that h
    // window and lands in the first H of tp_ar. Null unless tp_attach allocated it.
    bf16* tp_ffn = nullptr;      // [3 * moe_ffn/2] bf16
    char* tp_fq81 = nullptr;      // llama q8_1 scratch for the down's K-window activation quant
    signed char* tp_fq8 = nullptr; // q8_1 quant (non-llama path) of the same window
    float *tp_fq8_d = nullptr, *tp_fq8_s = nullptr;
    // Full-attn tp staging (hybrid models, 27B): tp_qraw is the rank's fused Q+gate N-window
    // GEMV output (2*qdim/2 bf16 == qdim: the raw [q|gate] per-head interleave before the split);
    // tp_pos/h_tp_pos carry this rank's own (rotary) position and seq_len for the windowed
    // append/decode, because the tp path never writes d_scalars[1]/[3].
    bf16* tp_qraw = nullptr;
    int* tp_pos = nullptr;
    int* h_tp_pos = nullptr;
    // (dual-GPU WP-12) DSpark multi-row verify scratch (verify_rows_tp), allocated on first use on
    // this rank's own device, sized for kTpVerifyRows rows. rec_* keep each GDN layer's projected
    // rows so a partial accept can replay exactly the kept rows into the restored state.
    bool vr_ready = false;
    bf16 *vr_x = nullptr, *vr_xn = nullptr, *vr_h = nullptr, *vr_hn = nullptr, *vr_ar = nullptr;
    signed char* vr_nq = nullptr;   // NVFP4 activation quant rows [R, kmax]
    float* vr_ns = nullptr;         // its per-16 scales [R, kmax/16]
    char* vr_q81 = nullptr;         // Q8_1 rows [R, q8_1_bytes(kmax)]
    size_t vr_q81_row = 0;
    bf16 *vr_rec_qkv = nullptr, *vr_rec_a = nullptr, *vr_rec_b = nullptr;   // [n_layers][R][...]
    bf16 *vr_full = nullptr, *vr_z = nullptr, *vr_gdn = nullptr, *vr_ln = nullptr;
    bf16 *vr_qraw = nullptr, *vr_q = nullptr, *vr_g = nullptr, *vr_k = nullptr, *vr_v = nullptr,
         *vr_attn = nullptr, *vr_ffn = nullptr;
    float *vr_lh = nullptr, *vr_logits = nullptr;
    int *vr_ids = nullptr, *vr_pos = nullptr, *vr_seq = nullptr, *vr_out = nullptr;
    int* h_vr = nullptr;            // pinned [4][R]: ids | positions | seqlens | argmax
    float* h_vr_maxv = nullptr;     // pinned [R]: each row's local head maximum
    double plain_step_ms[5] = {0, 0, 0, 0, 0};   // note_plain_decode, by rows
    // (plan 06, P) Sampled verify rows: this rank's top-k candidates per row, the merged ones,
    // and the per-row sampler parameters. Device [R][64] / [R], and one pinned host block.
    float *vr_tk_v = nullptr, *vr_mg_v = nullptr, *vr_smp_temp = nullptr, *vr_smp_topp = nullptr;
    int *vr_tk_i = nullptr, *vr_mg_i = nullptr, *vr_smp_topk = nullptr, *vr_smp_out = nullptr;
    unsigned long long *vr_smp_seed = nullptr, *vr_smp_step = nullptr;
    char* h_vr_smp = nullptr;
    // Constrained verify rows: this rank's half of each row's allowed-token bitmask, [R][Vr / 32]
    // on the device and pinned staging for it; allocated on the first constrained verify.
    uint32_t* vr_mask = nullptr;
    uint32_t* h_vr_mask = nullptr;
    int vr_mask_state = 0;          // 0 = untried, 1 = ready, -1 = declined (agreed across ranks)
    float* vr_snap_lin = nullptr;
    bf16* vr_snap_conv = nullptr;
    // (dual-GPU C1b) FP4 tensor-core staging for the multi-session rows path: one activation
    // operand (data + block scales) for the widest K, and one CUTLASS workspace for every shape.
    int vr_tc_state = 0;            // 0 = untried, 1 = ready, -1 = declined (agreed across ranks)
    void *vr_tc_a = nullptr, *vr_tc_as = nullptr, *vr_tc_ws = nullptr;
    // (dual-GPU C1c) Multi-session GDN in one launch per layer: per-row state / conv pointer
    // arrays (device + pinned staging) and full-width q|k|v, alpha/beta and output rows.
    bool vr_mb_ready = false;
    void** vr_mb_ptrs = nullptr;    // device [2R]: R conv-state pointers, then R state pointers
    void** h_vr_mb_ptrs = nullptr;  // pinned twin
    bf16 *vr_mb_q = nullptr, *vr_mb_k = nullptr, *vr_mb_v = nullptr, *vr_mb_a = nullptr,
         *vr_mb_b = nullptr, *vr_mb_o = nullptr;
    // ...and the attention layers' KV append + flash decode in one launch each (TC mode): packed
    // per-row block tables (full and ring) and flash-decode partials for R rows.
    int vr_ma_state = 0;            // 0 untried, 1 ready, -1 declined (agreed)
    const int** vr_ma_tptr = nullptr;      // device [2R]: full-table pointers, then ring-table
    const int** h_vr_ma_tptr = nullptr;    // pinned twin
    int *vr_ma_tab = nullptr, *vr_ma_tab_win = nullptr;   // [R][max_blocks]
    int *vr_ma_pairs = nullptr, *h_vr_ma_pairs = nullptr;   // [2R]: row pairs of one session (B1)
    int *vr_ma_grp = nullptr, *h_vr_ma_grp = nullptr;       // [8R]: nvfp4 row groups (plan 10)
    float *vr_ma_m = nullptr, *vr_ma_l = nullptr, *vr_ma_acc = nullptr;
    // (dual-GPU C2) Segmented multi-session verify: one GDN snapshot per segment (session),
    // grown on demand, agreed across the ranks, released when the group run ends.
    std::vector<float*> vr_seg_snap_lin;
    std::vector<bf16*> vr_seg_snap_conv;
    // ...or, with the multi-step GDN kernels, no snapshot at all: a read-only forward over every
    // step and a commit of the accepted steps. Per linear layer: the full-width q|k|v conv input
    // rows, the conv outputs and alpha/beta, kept from the forward for the commit.
    int vr_ms_state = 0;            // 0 untried, 1 ready, -1 declined (agreed)
    bf16 *vr_ms_full = nullptr, *vr_ms_q = nullptr, *vr_ms_k = nullptr, *vr_ms_v = nullptr,
         *vr_ms_a = nullptr, *vr_ms_b = nullptr;
    int* vr_ms_keep = nullptr;      // device [R]
    int* h_vr_ms_keep = nullptr;    // pinned [R]

    template <class T> T* alloc(size_t n) { void* p=nullptr; cu(cudaMalloc(&p, n*sizeof(T)), "malloc"); return (T*)p; }
};

Qwen35Model::Qwen35Model(const Qwen35Config& cfg, KVCacheManager* kv, moe::MoEEngine* engine,
                         GdnStateWindow gdn_window, int rank, int device)
    : p_(new Impl()) {
    p_->cfg = cfg; p_->kv = kv; p_->engine = engine;
    p_->rank = rank; p_->device = device;
    // One-time device bind (dual-GPU Wave 3): a split instance is built on its own card, and
    // every allocation and launch below (and the dtor's frees) then run on that card. (0,0) --
    // the unsplit tp=1 default -- fires nothing, so that path stays byte-identical to today.
    if (rank != 0 || device != 0)
        cu(cudaSetDevice(device), "ctor setDevice");
    // Normalize this instance's GDN v-head state window (dual-GPU state split). A degenerate
    // request (or tp=1) stays degenerate: every allocation below then sizes the full state,
    // byte-identical to the unsplit model.
    bool gdn_warned = false;
    p_->gdn_window = gdn_window_normalize(cfg, gdn_window, &gdn_warned);
    if (gdn_warned)
        fprintf(stderr, "[gdn-split] requested state window (v_start=%d, v_count=%d) is not "
                        "expressible by this model's GDN qh mapping; using all %d heads\n",
                gdn_window.v_start, gdn_window.v_count, cfg.linear_v_heads);
    // Flash-decode KV-split count is occupancy tuning only (math is identical for any
    // value — empty splits contribute zero), and it's baked into the decode CUDA graph
    // at construction. 16 over-subscribes the GPU for short context (32 q_heads * 16 =
    // 512 single-warp blocks); SPARKINFER_NSPLITS lets the scored regime be tuned/swept
    // without a rebuild. Clamp to [1, 64]; buffers below are sized from it.
    if (const char* ns = getenv("SPARKINFER_NSPLITS")) {
        int v = atoi(ns); if (v < 1) v = 1; if (v > Impl::MAX_NSPLITS) v = Impl::MAX_NSPLITS; p_->n_splits = v;
        p_->adaptive_splits = false;   // fixed n_splits (A/B/sweeps)
        fprintf(stderr, "[nsplits] flash-decode splits = %d (fixed env override)\n", v);
    }
    if (const char* c = getenv("SPARKINFER_SPLIT_CHUNK")) { int v = atoi(c); if (v > 0) p_->split_chunk = v; }
    p_->qdim = cfg.n_q_heads * cfg.head_dim;
    p_->kvdim = cfg.n_kv_heads * cfg.head_dim;
    p_->linear_qdim = cfg.linear_q_heads * cfg.linear_head_dim;
    p_->linear_vdim = cfg.linear_v_heads * cfg.linear_head_dim;
    p_->linear_qkvdim = 2 * p_->linear_qdim + p_->linear_vdim;
    // Non-blocking: a blocking stream implicitly synchronises with the legacy stream, so a
    // graph capture here makes any other thread's legacy-stream work fail with "operation
    // would make the legacy stream depend on a capturing blocking stream". That is what
    // breaks concurrent requests -- capture is per-thread, but the implicit legacy edge is not.
    cudaStreamCreateWithFlags(&p_->stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&p_->stream_k, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&p_->stream_v, cudaStreamNonBlocking);
    // Prefetch stream/events exist ONLY for Muse Glimmer, so every other architecture keeps
    // byte-for-byte the stream and event set it had before this change.
    if (cfg.muse_glimmer) {
        cudaStreamCreateWithFlags(&p_->stream_pf, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&p_->ev_pf_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&p_->ev_pf_done, cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&p_->ev_qkv, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_k, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_v, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_pipe_fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_gdn_z, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_gdn_ab, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_sx_gate, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_sx_done, cudaEventDisableTiming);
    const int H = cfg.hidden;
    p_->x=p_->alloc<bf16>(H); p_->xn=p_->alloc<bf16>(H);
    p_->q=p_->alloc<bf16>(p_->qdim); p_->k=p_->alloc<bf16>(p_->kvdim); p_->v=p_->alloc<bf16>(p_->kvdim);
    p_->attn=p_->alloc<bf16>(p_->qdim); p_->ao=p_->alloc<bf16>(H);
    p_->h=p_->alloc<bf16>(H); p_->hn=p_->alloc<bf16>(H);
    p_->routed=p_->alloc<bf16>(H); p_->shared=p_->alloc<bf16>(H);
    if (cfg.hybrid) {
        p_->qraw=p_->alloc<bf16>(p_->qdim * 2);
        p_->qgate=p_->alloc<bf16>(p_->qdim);
        p_->lin_qkv=p_->alloc<bf16>(p_->linear_qkvdim);
        p_->lin_q=p_->alloc<bf16>(p_->linear_qdim);
        p_->lin_k=p_->alloc<bf16>(p_->linear_qdim);
        p_->lin_v=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_z=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_alpha=p_->alloc<bf16>(cfg.linear_v_heads);
        p_->lin_beta=p_->alloc<bf16>(cfg.linear_v_heads);
        p_->lin_gdn=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_norm=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_conv_state=p_->alloc<bf16>((size_t)cfg.n_layers * (cfg.linear_conv_kernel - 1) * p_->linear_qkvdim);
        // Dense per-v-head arena: a split instance keeps only its own v_count heads per slot.
        // Conv state stays FULL on every device -- its rows are shared q/k/v channels, not
        // v-head blocks -- so the split saves the lin_state bytes (72.0 of the 147.75 MiB/seq
        // on the 27B) and not the conv bytes.
        const int gdn_vloc = p_->gdn_window.v_count > 0 ? p_->gdn_window.v_count : cfg.linear_v_heads;
        p_->lin_state=p_->alloc<float>((size_t)gdn_state_slots(cfg) * gdn_vloc * cfg.linear_head_dim * cfg.linear_head_dim);
        if (p_->gdn_window.v_count > 0) {
            // One shared full-layout round-trip arena for the GDN paths that only have full-v
            // kernels (prefill scan, dflash verify/commit). Zeroed once here at allocation; the
            // v-head blocks OUTSIDE the window may then keep whatever deterministic garbage the
            // kernels leave in them, because the GDN recurrence is block-diagonal in v-head --
            // no windowed head's [HD][HD] block is ever touched by a non-windowed head.
            const size_t scratch_n = (size_t)gdn_state_slots(cfg) * cfg.linear_v_heads * cfg.linear_head_dim * cfg.linear_head_dim;
            p_->gdn_scratch=p_->alloc<float>(scratch_n);
            cu(cudaMemsetAsync(p_->gdn_scratch, 0, scratch_n * sizeof(float), p_->stream), "gdn_scratch zero");
        }
        p_->shared_gate_tmp=p_->alloc<bf16>(1);
    }
    p_->logits=p_->alloc<float>(cfg.vocab);
    // presence_penalty/frequency_penalty per-vocab running count, session-0's own buffer (every
    // OTHER session's own count buffer is alloc'd by open_session()) -- unconditional, every
    // model, unlike lin_state/lin_conv_state above (hybrid-architecture-only).
    p_->penalty_counts_default=p_->alloc<int>(cfg.vocab);
    p_->penalty_counts=p_->penalty_counts_default;
    cu(cudaMemsetAsync(p_->penalty_counts_default, 0, (size_t)cfg.vocab * sizeof(int), p_->stream),
       "penalty_counts_default zero");
    // logit_bias per-vocab additive bias, session-0's own buffer -- mirrors penalty_counts_default
    // exactly (unconditional, every model).
    p_->logit_bias_default=p_->alloc<float>(cfg.vocab);
    p_->logit_bias=p_->logit_bias_default;
    cu(cudaMemsetAsync(p_->logit_bias_default, 0, (size_t)cfg.vocab * sizeof(float), p_->stream),
       "logit_bias_default zero");
    p_->d_logit_bias_ids=p_->alloc<int>(kMaxLogitBiasEntries);
    p_->d_logit_bias_vals=p_->alloc<float>(kMaxLogitBiasEntries);
    cu(cudaHostAlloc(&p_->h_logit_bias_ids, kMaxLogitBiasEntries * sizeof(int), cudaHostAllocDefault),
       "host logit_bias ids");
    cu(cudaHostAlloc(&p_->h_logit_bias_vals, kMaxLogitBiasEntries * sizeof(float), cudaHostAllocDefault),
       "host logit_bias vals");
    p_->d_scalars=p_->alloc<int>(5);
    p_->d_tok=p_->d_scalars + 0; p_->d_pos=p_->d_scalars + 1;
    p_->d_writepos=p_->d_scalars + 2; p_->d_seqlen=p_->d_scalars + 3;
    p_->d_cap_row=p_->d_scalars + 4;
    p_->d_out_id=p_->alloc<int>(1);
    cu(cudaHostAlloc(&p_->h_scalars, 5 * sizeof(int), cudaHostAllocDefault), "host scalars");
    cu(cudaHostAlloc(&p_->h_out_id, sizeof(int), cudaHostAllocDefault), "host out id");
    p_->d_sample_temp=p_->alloc<float>(1);
    p_->d_sample_seed=p_->alloc<unsigned long long>(1);
    p_->d_sample_step=p_->alloc<unsigned long long>(1);
    cu(cudaHostAlloc(&p_->h_sample_temp, sizeof(float), cudaHostAllocDefault), "host sample temp");
    cu(cudaHostAlloc(&p_->h_sample_seed, sizeof(unsigned long long), cudaHostAllocDefault), "host sample seed");
    cu(cudaHostAlloc(&p_->h_sample_step, sizeof(unsigned long long), cudaHostAllocDefault), "host sample step");
    *p_->h_sample_temp = 0.f;
    p_->d_sample_top_k=p_->alloc<int>(1);
    p_->d_sample_top_p=p_->alloc<float>(1);
    cu(cudaHostAlloc(&p_->h_sample_top_k, sizeof(int), cudaHostAllocDefault), "host sample top_k");
    cu(cudaHostAlloc(&p_->h_sample_top_p, sizeof(float), cudaHostAllocDefault), "host sample top_p");
    *p_->h_sample_top_k = 0;
    *p_->h_sample_top_p = 1.f;
    p_->d_sample_presence_penalty=p_->alloc<float>(1);
    p_->d_sample_frequency_penalty=p_->alloc<float>(1);
    cu(cudaHostAlloc(&p_->h_sample_presence_penalty, sizeof(float), cudaHostAllocDefault), "host sample presence_penalty");
    cu(cudaHostAlloc(&p_->h_sample_frequency_penalty, sizeof(float), cudaHostAllocDefault), "host sample frequency_penalty");
    *p_->h_sample_presence_penalty = 0.f;
    *p_->h_sample_frequency_penalty = 0.f;
    // top_k/top_p truncation scratch -- allocated once here (vocab-sized, fixed address), never
    // reallocated per-call. Sizing the two CUB temp-storage buffers is a load-time-only call, no
    // kernel launch involved.
    p_->d_vocab_iota=p_->alloc<int>(cfg.vocab);
    p_->d_sorted_logits=p_->alloc<float>(cfg.vocab);
    p_->d_sorted_idx=p_->alloc<int>(cfg.vocab);
    p_->d_topk_exp=p_->alloc<float>(cfg.vocab);
    p_->d_topk_cumsum=p_->alloc<float>(cfg.vocab);
    kernels::launch_vocab_iota_init(p_->d_vocab_iota, cfg.vocab);
    p_->sort_temp_bytes = kernels::topk_sort_temp_storage_bytes(cfg.vocab);
    p_->scan_temp_bytes = kernels::topk_scan_temp_storage_bytes(cfg.vocab);
    cu(cudaMalloc(&p_->d_sort_temp, p_->sort_temp_bytes), "topk sort temp");
    cu(cudaMalloc(&p_->d_scan_temp, p_->scan_temp_bytes), "topk scan temp");
    // logprobs/top_logprobs scratch, same discipline as above. d_rank_by_id needs no memset --
    // topk_topp_exp_kernel fully overwrites it every decode step before it is ever read.
    p_->d_rank_by_id=p_->alloc<int>(cfg.vocab);
    p_->d_chosen_logit=p_->alloc<float>(1);
    p_->d_denom_partials=p_->alloc<float>(256);
    p_->d_denom_det=p_->alloc<float>(1);
    p_->d_shared_ids=p_->alloc<int>(1); p_->d_shared_w=p_->alloc<float>(1);
    int zero=0; float one=1.f;
    cu(cudaMemcpy(p_->d_shared_ids,&zero,sizeof(int),cudaMemcpyHostToDevice),"shared ids");
    cu(cudaMemcpy(p_->d_shared_w,&one,sizeof(float),cudaMemcpyHostToDevice),"shared w");
    // Fused-expert + flash-decoding decode scratch (batch 1). Allocated here so
    // EVERY load path (set_weights / load_weights / load_gguf) has it — not just
    // GGUF. (fa_* NULL here is what crashed flash_decode_split on the non-GGUF path.)
    p_->mf_logits  = p_->alloc<float>(std::max(1, cfg.n_experts));
    p_->mf_ids     = p_->alloc<int>(std::max(1, cfg.top_k));
    p_->mf_weights = p_->alloc<float>(std::max(1, cfg.top_k));
    p_->mf_counts  = p_->alloc<int>(std::max(1, cfg.n_experts));
    p_->mf_rc      = p_->alloc<unsigned int>(1);
    cu(cudaMemset(p_->mf_rc, 0, sizeof(unsigned int)), "mf_rc zero");   // grid-completion counter starts at 0
    p_->mf_h       = p_->alloc<float>((size_t)std::max(1, cfg.top_k) * cfg.moe_ffn);
    p_->mf_out     = p_->alloc<float>(cfg.hidden);
    p_->nv_gate    = p_->alloc<bf16>(cfg.moe_ffn);
    p_->nv_up      = p_->alloc<bf16>(cfg.moe_ffn);
    p_->nv_h       = p_->alloc<bf16>(cfg.moe_ffn);
    p_->nv_xq      = p_->alloc<signed char>(cfg.moe_ffn);
    p_->nv_xs      = p_->alloc<float>(cfg.moe_ffn / 16 + 1);
    {
        int pw = cfg.hidden;
        if (p_->qdim > pw) pw = p_->qdim;
        if (p_->linear_vdim > pw) pw = p_->linear_vdim;
        p_->nv_pq_a = p_->alloc<signed char>(pw);
        p_->nv_ps_a = p_->alloc<float>(pw / 16 + 1);
        p_->nv_pq_b = p_->alloc<signed char>(pw);
        p_->nv_ps_b = p_->alloc<float>(pw / 16 + 1);
    }
    if (cfg.dense_ffn && cfg.top_k > 0) {
        cu(cudaMemcpy(p_->mf_ids, &zero, sizeof(int), cudaMemcpyHostToDevice), "dense expert id");
        cu(cudaMemcpy(p_->mf_weights, &one, sizeof(float), cudaMemcpyHostToDevice), "dense expert w");
    }
    if (cfg.n_shared > 0) {
        p_->sx_h  = p_->alloc<float>(cfg.moe_ffn);
        p_->sx_q8 = p_->alloc<char>(kernels::llama_q8_1_bytes(cfg.moe_ffn));
    }
    const size_t fa_n = (size_t)cfg.n_q_heads * Impl::MAX_NSPLITS;   // sized for the adaptive max
    p_->fa_m   = p_->alloc<float>(fa_n);
    p_->fa_l   = p_->alloc<float>(fa_n);
    p_->fa_acc = p_->alloc<float>(fa_n * cfg.head_dim);
    // Sink + sliding-window sparse KV: default ON for Qwythos GQA-4 hd256 (int8 KV).
    // SPARKINFER_SPARSE_KV=0 restores dense full-context flash-decode.
    bool sparse_enable = true;
    if (const char* se = getenv("SPARKINFER_SPARSE_KV")) sparse_enable = (se[0] != '0');
    const bool sparse_gqa4 = cfg.head_dim == 256 && cfg.n_kv_heads > 0 &&
                             cfg.n_q_heads == cfg.n_kv_heads * 4;
    // GQA-8 hd256 (Qwen3.6 full-attn layers), issue #559: same sink+window policy, but
    // realized as a compacted paged-KV view fed to the unmodified dense int8-MMA kernel.
    // The compact view is ratio-AGNOSTIC: launch_fa_kv_compact_view builds a sink+window block
    // table, and launch_flash_decode_split is the same dense entry point that already serves this
    // shape, with the head counts passed as runtime arguments. GQA-8 was simply the ratio it was
    // first wired for. Qwen3.8 is GQA-6 (24 q over 4 kv), matched neither predicate, and has
    // therefore been reading the WHOLE KV cache on every decode step -- 8.59 GB at ctx=262144,
    // which nsys puts at 9.56 ms of a 19.87 ms step (48%) and only 50% of DRAM peak.
    //
    // Opt-in (SPARKINFER_SPARSE_GQA6=1), not the default. The view is an approximation, not a
    // lossless kernel: from min_ctx on, decode attends the sink and the last 4096 tokens and
    // nothing in between (#958 said so; its accuracy gate never reached min_ctx). An agent
    // session passes 16K tokens within a few turns, and from there Qwen3.8 could not read back:
    // pi read a 31K-token tool result and described it as a column of integers, and a line
    // 20K tokens back was reported absent (#1088). Exact attention costs decode throughput at
    // long context (95 -> 85.7 tok/s at ctx=32768 in #958); that is the price of the answer.
    const int gqa_ratio = cfg.n_kv_heads > 0 ? cfg.n_q_heads / cfg.n_kv_heads : 0;
    static const bool sparse_gqa6_on = [] {
        const char* e = getenv("SPARKINFER_SPARSE_GQA6");
        return e && e[0] == '1';
    }();
    const bool sparse_gqa6 = sparse_gqa6_on && cfg.head_dim == 256 && cfg.n_kv_heads > 0 &&
                             cfg.n_q_heads == cfg.n_kv_heads * 6;
    const bool sparse_gqa8 = cfg.head_dim == 256 && cfg.n_kv_heads > 0 &&
                             cfg.n_q_heads == cfg.n_kv_heads * 8;
    const bool sparse_view = sparse_gqa6 || sparse_gqa8;   // compact-view tiers
    if (sparse_enable && (sparse_gqa4 || sparse_view)) {
        p_->sparse_window = 256;
        if (const char* w = getenv("SPARKINFER_SPARSE_WINDOW")) { int v = atoi(w); if (v > 0) p_->sparse_window = v; }
        // Legacy aliases from the Quest prototype (blocks, not tokens).
        if (const char* rw = getenv("SPARKINFER_SPARSE_RECENT")) { int v = atoi(rw); if (v > 0) p_->sparse_window = v; }
        if (const char* b = getenv("SPARKINFER_SPARSE_BUDGET")) {
            int v = atoi(b); if (v > 1) p_->sparse_window = v - 1;   // budget included sink
        }
        // GQA-8: the dense hd256 path is already int8 tensor-core, so the O(window) read
        // only clears the dense cost decisively from ~16k context up (bot-measured on this
        // shape: sparse under 16k is a wash-to-regression, 16k/32k are wins). Engaging at
        // 16384 also keeps every mid-length scoring probe on the exact dense path.
        if (sparse_view) p_->sparse_min_ctx = 16384;
        if (const char* mc = getenv("SPARKINFER_SPARSE_MIN_CTX")) { int v = atoi(mc); if (v > 0) p_->sparse_min_ctx = v; }
        p_->sparse_budget = 1 + p_->sparse_window;
        if (sparse_view) {
            p_->sparse_vtbl = p_->alloc<int>(p_->sparse_budget);
            p_->sparse_vlen = p_->alloc<int>(1);
            // Split the compact view so each MMA split still covers >= 2 KV blocks (the
            // dense launcher's tensor-core engagement condition), capped at MAX_NSPLITS.
            // window=256 -> 4112-token view -> 128 splits (x8 kv heads = 1024 CTAs).
            int vs = (p_->sparse_budget * kv->block_size()) / 32;
            if (vs > Impl::MAX_NSPLITS) vs = Impl::MAX_NSPLITS;
            if (vs < 1) vs = 1;
            p_->sparse_vsplits = vs;
        } else {
            p_->sparse_sel = p_->alloc<int>((size_t)cfg.n_kv_heads * p_->sparse_budget);
        }
        fprintf(stderr, "[sparse-kv] sliding-window (default on): gqa=%d window=%d blocks (%d tokens) min_ctx=%d%s\n",
                sparse_view ? gqa_ratio : 4, p_->sparse_window, p_->sparse_window * kv->block_size(),
                p_->sparse_min_ctx, sparse_view ? " (compact-view, decode-only)" : "");
    }
    // Muse Glimmer: mandatory pure sliding-window view for swa-flagged layers, every step,
    // regardless of context length (architectural, not a long-context approximation).
    if (cfg.muse_glimmer && cfg.sliding_window > 0) {
        p_->swa_budget = (cfg.sliding_window + kv->block_size() - 1) / kv->block_size();
        p_->swa_vtbl = p_->alloc<int>(p_->swa_budget);
        p_->swa_vlen = p_->alloc<int>(1);
        int vs = (p_->swa_budget * kv->block_size()) / 32;
        if (vs > Impl::MAX_NSPLITS) vs = Impl::MAX_NSPLITS;
        if (vs < 1) vs = 1;
        p_->swa_vsplits = vs;
        fprintf(stderr, "[muse-glimmer] sliding-window: %d tokens (%d blocks), every-4th-layer global/NoPE\n",
                cfg.sliding_window, p_->swa_budget);
        // Unweighted RMSNorm applied to the token embedding before layer 0 (no learned
        // per-channel weight -- launch_rmsnorm always takes one, so fill a constant-1.0
        // buffer once at load time and reuse it as that "weight").
        p_->emb_norm_ones = p_->alloc<bf16>(H);
        std::vector<bf16> ones(H, (bf16)0x3F80u);   // bf16 bit pattern for 1.0f
        cudaMemcpy(p_->emb_norm_ones, ones.data(), (size_t)H * sizeof(bf16), cudaMemcpyHostToDevice);
    }
    const int kmax = (p_->qdim > H) ? p_->qdim : H;          // largest projection input dim
    p_->aq8   = p_->alloc<signed char>(kmax);
    p_->aq8_d = p_->alloc<float>(kmax >> 5);
    p_->aq8_s = p_->alloc<float>(kmax >> 5);
    p_->aq81  = p_->alloc<char>(kernels::llama_q8_1_bytes(kmax));
    if (const char* e = getenv("SPARKINFER_PQ"))    p_->use_pq    = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_LLAMA")) p_->use_llama = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_Q6MMVQ")) p_->use_q6mmvq = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_QKFUSE")) p_->use_qkfuse = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ROPEKV")) p_->use_ropekv = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_FNQ"))    p_->use_fnq   = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_QKVSTREAM")) p_->use_qkvstream = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ATTNIN")) p_->use_attnin = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_GDN_PIPE")) p_->use_gdn_pipe = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_GDN_QUAD")) p_->use_gdn_quad = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ATTN_QKV")) p_->use_attn_qkv = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_SHEXP_PIPE")) p_->use_shexp_pipe = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ADDNORM3")) p_->use_addnorm3 = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ROUTER_FUSED")) p_->use_router_fused = !(e[0] == '0');
    {
        SessionBuffers d;
        if (cfg.hybrid) {
            d.lin_state = p_->lin_state;
            d.lin_conv_state = p_->lin_conv_state;
        }
        d.penalty_counts = p_->penalty_counts_default;   // unconditional -- every model
        d.logit_bias = p_->logit_bias_default;           // unconditional -- every model
        p_->sessions[0] = d;
    }
}

// (dual-GPU S7a-1) Process-scope state for the prefill o_proj all-reduce: tp_attach publishes the
// shared GpuLink plus the rank's (device, stream) into the slot of the link end it sits on (slot
// 0 = device_a, the all-reduce leader); each prefill pass then registers its per-pass partial
// buffer in its slot, and tp_prefill_allreduce_bf16 posts the one link-wide reduce.
static GpuLink* g_tp_prefill_link = nullptr;
static int g_tp_prefill_dev[2] = {-1, -1};
static cudaStream_t g_tp_prefill_stream[2] = {nullptr, nullptr};
static void* g_tp_prefill_buf[2] = {nullptr, nullptr};

// (dual-GPU) Host rendezvous for the leader-issued link ops (decode AR rows, the epilogue
// maxreduces, the prefill o_proj AR). One GpuLink call posts the reduce on BOTH ranks' streams,
// so the leader may post only once the peer has enqueued the producers of its input, and the
// peer may enqueue the consumers of the result only once the leader has posted onto its stream
// -- otherwise the reduce lands at an arbitrary point of the peer's stream. Both ranks run the
// identical sequence of link ops, so the n-th peer arrival pairs with the n-th leader post. A
// rank that falls out of step (a branch taken on one rank only) times out loudly instead of
// hanging the server.
static std::atomic<unsigned long long> g_tp_peer_arrived{0};
static std::atomic<unsigned long long> g_tp_leader_seq{0};
static std::atomic<unsigned long long> g_tp_leader_posted{0};

static bool tp_spin_until(const std::atomic<unsigned long long>& v, unsigned long long target,
                          const char* what) {
    const auto t0 = std::chrono::steady_clock::now();
    while (v.load(std::memory_order_acquire) < target) {
        // (WP-5 failure policy, device_health.h) Once either rank is lost -- or an earlier
        // rendezvous already timed out -- stop waiting: the ranks are no longer in step and
        // every later op would otherwise burn its own 60 s, i.e. hang the request.
        if (device_lost()) return false;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) {
            fprintf(stderr, "[tp] rendezvous timeout (%s, op %llu): the ranks are out of step\n",
                    what, target);
            note_tp_fatal("rendezvous timeout");
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

// Peer side: "my producers are enqueued", then hold until the leader has posted the op.
static void tp_peer_rendezvous(const char* what) {
    const unsigned long long n = g_tp_peer_arrived.fetch_add(1, std::memory_order_acq_rel) + 1;
    tp_spin_until(g_tp_leader_posted, n, what);
}

// Leader side: wait for the peer's arrival, post the op, then release the peer.
template <class F>
static void tp_leader_rendezvous(const char* what, F&& post) {
    const unsigned long long n = g_tp_leader_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (tp_spin_until(g_tp_peer_arrived, n, what)) post();
    g_tp_leader_posted.store(n, std::memory_order_release);
}

// (dual-GPU) Rank-1 mirroring. The serving engine drives only the rank-0 model; every
// state-changing public call (sessions, penalties, prompt ingest, decode) is replayed on the
// rank-1 model by TpMirrorScope: the peer's call runs on the peer device's persistent worker
// thread (TpWorker below, bound once to that device -- the op path itself never setDevice)
// concurrently with the leader's own body, which the rendezvous above needs, and is waited for
// before the leader returns. The
// thread-local depth makes only the OUTERMOST call mirror (ingest -> prefill -> ... must not
// replay twice); a null peer just holds the depth, which is also how a leader-only call is made.
static thread_local int t_tp_mirror_depth = 0;

// One PERSISTENT worker thread per peer device runs every mirrored op. It binds its device once
// and keeps its thread_local state (prefill arenas, graph keys, events, split-K scratch) across
// ops, so "thread_local" means "per rank" -- the property every such cache in the forward/prefill
// code relies on. A one-shot thread per op (the first design) started each op with EMPTY caches
// and leaked everything they allocated when it exited.
struct TpWorker {
    std::mutex scope_mu;   // one mirrored op at a time (held by TpMirrorScope for its lifetime)
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()>* job = nullptr;
    bool done = false, stop = false;
    std::thread th;
    explicit TpWorker(int dev) {
        th = std::thread([this, dev] {
            if (dev >= 0) cu(cudaSetDevice(dev), "tp worker setDevice");
            t_tp_mirror_depth = 1;   // ops on this thread never mirror again
            std::unique_lock<std::mutex> lk(mu);
            for (;;) {
                cv.wait(lk, [&] { return job != nullptr || stop; });
                if (stop) return;
                std::function<void()>* j = job;
                lk.unlock();
                (*j)();
                lk.lock();
                job = nullptr;
                done = true;
                cv.notify_all();
            }
        });
    }
    void post(std::function<void()>* j) {
        std::lock_guard<std::mutex> lk(mu);
        job = j;
        done = false;
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return done; });
    }
    ~TpWorker() {
        { std::lock_guard<std::mutex> lk(mu); stop = true; }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};

static TpWorker& tp_worker_for(int dev) {
    static std::mutex reg_mu;
    static std::map<int, std::unique_ptr<TpWorker>> reg;
    std::lock_guard<std::mutex> lk(reg_mu);
    auto& w = reg[dev];
    if (!w) w = std::make_unique<TpWorker>(dev);
    return *w;
}

// Lock order, for every mirrored op: the LEADER's device_mu, then the peer worker's scope_mu
// (then, inside the job, the peer's own device_mu). The leader's lock is taken first so a thread
// already holding it (e.g. an HTTP thread inside submit_locked -> open_session) can never wait
// for scope_mu while the scope_mu holder waits for that same device_mu -- the deadlock a
// concurrent decode + session open hit when scope_mu was taken first.
struct TpMirrorScope {
    TpWorker* worker = nullptr;
    std::unique_lock<std::recursive_mutex> leader_hold;
    std::unique_lock<std::mutex> hold;
    std::function<void()> job;
    template <class F>
    TpMirrorScope(std::recursive_mutex* leader_mu, Qwen35Model* peer, F&& fn) {
        ++t_tp_mirror_depth;
        if (!peer) return;
        if (leader_mu) leader_hold = std::unique_lock<std::recursive_mutex>(*leader_mu);
        job = [peer, f = std::forward<F>(fn)]() mutable { f(*peer); };
        worker = &tp_worker_for(peer->tp_rank_view().device);
        hold = std::unique_lock<std::mutex>(worker->scope_mu);
        worker->post(&job);
    }
    ~TpMirrorScope() {
        if (worker) worker->wait();
        --t_tp_mirror_depth;
    }
    TpMirrorScope(const TpMirrorScope&) = delete;
    TpMirrorScope& operator=(const TpMirrorScope&) = delete;
};

// Replays `call` (a member call expression) on the peer for the rest of the enclosing scope.
#define TP_MIRROR(...) \
    TpMirrorScope tp_mirror_scope_(&p_->device_mu, tp_mirror_peer(), \
                                   [&](Qwen35Model& tp_peer_) { (void)tp_peer_.__VA_ARGS__; })

Qwen35Model::~Qwen35Model() {
    // Mirror the ctor's one-time bind: cudaFree is device-bound, so a split instance must free
    // on the card it allocated from -- a cross-device free is the classic silent-corruption trap.
    // The (0,0) default fires nothing (tp=1 invariance).
    if (p_->rank != 0 || p_->device != 0)
        cu(cudaSetDevice(p_->device), "dtor setDevice");
    for (void* b : p_->bonsai_dec_bufs) cudaFree(b);
    if (p_->lm_head_fp4_payload) cudaFree(p_->lm_head_fp4_payload);
    if (p_->lm_head_fp4_sf_buf) cudaFree(p_->lm_head_fp4_sf_buf);
    if (p_->gdn_state_stage) cudaFree(p_->gdn_state_stage);
    for (void* b : p_->owned) cudaFree(b);
    cudaFree(p_->x); cudaFree(p_->xn); cudaFree(p_->q); cudaFree(p_->k); cudaFree(p_->v);
    cudaFree(p_->attn); cudaFree(p_->ao); cudaFree(p_->h); cudaFree(p_->hn);
    cudaFree(p_->routed); cudaFree(p_->shared); cudaFree(p_->logits);
    cudaFree(p_->penalty_counts_default);
    cudaFree(p_->logit_bias_default);
    cudaFree(p_->d_logit_bias_ids); cudaFree(p_->d_logit_bias_vals);
    cudaFreeHost(p_->h_logit_bias_ids); cudaFreeHost(p_->h_logit_bias_vals);
    if (p_->h_dense_bias) cudaFreeHost(p_->h_dense_bias);
    // Packed decode scalars (d_tok/d_pos/d_seqlen/d_writepos alias into d_scalars — not freed separately)
    cudaFree(p_->d_scalars); cudaFree(p_->d_out_id);
    cudaFreeHost(p_->h_scalars); cudaFreeHost(p_->h_out_id);
    cudaFree(p_->d_sample_temp); cudaFree(p_->d_sample_seed); cudaFree(p_->d_sample_step);
    cudaFreeHost(p_->h_sample_temp); cudaFreeHost(p_->h_sample_seed); cudaFreeHost(p_->h_sample_step);
    cudaFree(p_->d_sample_top_k); cudaFree(p_->d_sample_top_p);
    cudaFreeHost(p_->h_sample_top_k); cudaFreeHost(p_->h_sample_top_p);
    cudaFree(p_->d_sample_presence_penalty); cudaFree(p_->d_sample_frequency_penalty);
    cudaFreeHost(p_->h_sample_presence_penalty); cudaFreeHost(p_->h_sample_frequency_penalty);
    cudaFree(p_->d_vocab_iota); cudaFree(p_->d_sorted_logits); cudaFree(p_->d_sorted_idx);
    cudaFree(p_->d_topk_exp); cudaFree(p_->d_topk_cumsum);
    cudaFree(p_->d_sort_temp); cudaFree(p_->d_scan_temp);
    cudaFree(p_->d_rank_by_id); cudaFree(p_->d_chosen_logit);
    cudaFree(p_->d_denom_partials); cudaFree(p_->d_denom_det);
    cudaFree(p_->d_shared_ids); cudaFree(p_->d_shared_w);
    // Qwen3.6 Gated-DeltaNet buffers (allocated only for the hybrid model)
    cudaFree(p_->qraw); cudaFree(p_->qgate);
    cudaFree(p_->dbg_xn_dump);
    cudaFree(p_->lin_qkv); cudaFree(p_->lin_q); cudaFree(p_->lin_k); cudaFree(p_->lin_v);
    cudaFree(p_->lin_z); cudaFree(p_->lin_alpha); cudaFree(p_->lin_beta);
    cudaFree(p_->lin_gdn); cudaFree(p_->lin_norm); cudaFree(p_->lin_conv_state); cudaFree(p_->lin_state);
    cudaFree(p_->gdn_scratch);
    cudaFree(p_->tp_xrow); cudaFree(p_->tp_ar);
    cudaFree(p_->tp_qkv); cudaFree(p_->tp_qraw); cudaFree(p_->tp_pos); free(p_->h_tp_pos);
    cudaFree(p_->tp_ffn); cudaFree(p_->tp_fq81);
    cudaFree(p_->tp_fq8); cudaFree(p_->tp_fq8_d); cudaFree(p_->tp_fq8_s);
    // (S7a-1) Unpublish this instance from the process-scope prefill AR slots.
    g_tp_prefill_link = nullptr;
    g_tp_prefill_dev[0] = -1; g_tp_prefill_dev[1] = -1;
    g_tp_prefill_stream[0] = nullptr; g_tp_prefill_stream[1] = nullptr;
    g_tp_prefill_buf[0] = nullptr; g_tp_prefill_buf[1] = nullptr;
    cudaFree(p_->shared_gate_tmp);
    cudaFree(p_->nvfp4_g); cudaFree(p_->nvfp4_u); cudaFree(p_->nvfp4_h);
    cudaFree(p_->mf_logits); cudaFree(p_->mf_weights); cudaFree(p_->mf_h); cudaFree(p_->mf_out);
    cudaFree(p_->sx_h); cudaFree(p_->sx_q8);
    cudaFree(p_->mf_ids); cudaFree(p_->mf_counts); cudaFree(p_->mf_rc);
    cudaFree(p_->moe_rs_gate); cudaFree(p_->moe_rs_up); cudaFree(p_->moe_rs_down);
    cudaFree(p_->muse_rs);
    cudaFree(p_->fa_m); cudaFree(p_->fa_l); cudaFree(p_->fa_acc);
    cudaFree(p_->sparse_sel);
    cudaFree(p_->sparse_vtbl); cudaFree(p_->sparse_vlen);
    cudaFree(p_->swa_vtbl); cudaFree(p_->swa_vlen); cudaFree(p_->emb_norm_ones);
    cudaFree(p_->aq8); cudaFree(p_->aq8_d); cudaFree(p_->aq8_s); cudaFree(p_->aq81);
    cudaFree(p_->dflash_hidden); cudaFree(p_->dflash_context);
    // spec_lin_snap / spec_conv_snap are in owned[] (allocated via Impl::alloc)
    for (auto& kv : p_->sessions) {
        if (kv.first == 0) continue;
        if (kv.second.lin_state) cudaFree(kv.second.lin_state);
        if (kv.second.lin_conv_state) cudaFree(kv.second.lin_conv_state);
    }
    if (p_->packed_host_states) cudaFreeHost(p_->packed_host_states);
    if (p_->packed_host_convs) cudaFreeHost(p_->packed_host_convs);
    if (p_->packed_host_tables) cudaFreeHost(p_->packed_host_tables);
    if (p_->packed_host_seqs) cudaFreeHost(p_->packed_host_seqs);
    if (p_->packed_dev_states) cudaFree(p_->packed_dev_states);
    if (p_->packed_dev_convs) cudaFree(p_->packed_dev_convs);
    if (p_->packed_dev_tables) cudaFree(p_->packed_dev_tables);
    if (p_->packed_dev_tables_win) cudaFree(p_->packed_dev_tables_win);
    if (p_->packed_host_tables_win) cudaFreeHost(p_->packed_host_tables_win);
    for (auto& kv : p_->parked_graphs) {
        if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        if (kv.second.graph) cudaGraphDestroy(kv.second.graph);
    }
    p_->parked_graphs.clear();
    if (p_->graph_ready) { cudaGraphExecDestroy(p_->cu_exec); cudaGraphDestroy(p_->cu_graph); }
    if (p_->graph_prefill_ready) { cudaGraphExecDestroy(p_->cu_prefill_exec); cudaGraphDestroy(p_->cu_prefill_graph); }
    if (p_->dflash_graph_ready) { cudaGraphExecDestroy(p_->cu_dflash_exec); cudaGraphDestroy(p_->cu_dflash_graph); }
    if (p_->ev_pf_fork) cudaEventDestroy(p_->ev_pf_fork);
    if (p_->ev_pf_done) cudaEventDestroy(p_->ev_pf_done);
    cudaEventDestroy(p_->ev_qkv); cudaEventDestroy(p_->ev_k); cudaEventDestroy(p_->ev_v);
    cudaEventDestroy(p_->ev_pipe_fork); cudaEventDestroy(p_->ev_gdn_z); cudaEventDestroy(p_->ev_gdn_ab);
    cudaEventDestroy(p_->ev_sx_gate); cudaEventDestroy(p_->ev_sx_done);
    if (p_->stream_pf) cudaStreamDestroy(p_->stream_pf);
    cudaStreamDestroy(p_->stream_v); cudaStreamDestroy(p_->stream_k);
    cudaStreamDestroy(p_->stream);
    delete p_;
}

void Qwen35Model::set_weights(const Qwen35Weights& w) { p_->w = w; }
const Qwen35Config& Qwen35Model::config() const { return p_->cfg; }
const GdnStateWindow& Qwen35Model::gdn_state_window() const { return p_->gdn_window; }
int Qwen35Model::gdn_v_local() const {
    return p_->gdn_window.v_count > 0 ? p_->gdn_window.v_count : p_->cfg.linear_v_heads;
}

void Qwen35Model::copy_logits(float* host_logits) const {
    // p_->logits holds the last step's lm-head output; forward_token() syncs the
    // stream before returning, so it is valid to read here.
    cudaMemcpy(host_logits, p_->logits, (size_t)p_->cfg.vocab * sizeof(float), cudaMemcpyDeviceToHost);
}

Qwen35Model::TokenLogprob Qwen35Model::last_token_logprobs(int top_n) const {
    Impl& s = *p_;
    top_n = std::max(0, std::min(top_n, kMaxTopLogprobs));

    TokenLogprob out;
    out.token_id = *s.h_out_id;   // already synced by forward_token() before it returned
    // A lost context makes every copy below fail, leaving the locals at their initialised values
    // -- which would be published as a real, confident-looking distribution. Return the token id
    // with no logprob data instead; the engine is about to fail this request anyway.
    if (device_lost()) return out;

    float denom = 1.f, chosen_logit = 0.f;
    if (deterministic_mode()) {
        // Recompute the normalizer with a pinned summation order instead of reading the CUB
        // scan's last element, whose low bit is timing-dependent -- see launch_logprob_denom_det.
        // Legal to branch on here (unlike inside the captured decode graph) because this runs on
        // the host after forward_token() has already returned.
        kernels::launch_logprob_denom_det(s.d_topk_exp, s.cfg.vocab, s.d_denom_partials,
                                          s.d_denom_det, s.stream);
        cu(cudaStreamSynchronize(s.stream), "logprob denom sync");
        cu(cudaMemcpy(&denom, s.d_denom_det, sizeof(float), cudaMemcpyDeviceToHost), "logprob denom");
    } else {
        cu(cudaMemcpy(&denom, s.d_topk_cumsum + (s.cfg.vocab - 1), sizeof(float), cudaMemcpyDeviceToHost),
           "logprob denom (scan)");
    }
    cu(cudaMemcpy(&chosen_logit, s.d_chosen_logit, sizeof(float), cudaMemcpyDeviceToHost), "logprob chosen");

    float row_max = 0.f;
    std::vector<int> ids;
    std::vector<float> logits;
    if (top_n > 0) {
        ids.resize(top_n);
        logits.resize(top_n);
        cu(cudaMemcpy(ids.data(), s.d_sorted_idx, (size_t)top_n * sizeof(int), cudaMemcpyDeviceToHost),
           "logprob top ids");
        cu(cudaMemcpy(logits.data(), s.d_sorted_logits, (size_t)top_n * sizeof(float), cudaMemcpyDeviceToHost),
           "logprob top logits");
        row_max = logits[0];   // rank 0 is always the row max, free from the descending sort
    } else {
        cu(cudaMemcpy(&row_max, s.d_sorted_logits, sizeof(float), cudaMemcpyDeviceToHost), "logprob row max");
    }

    const float logsumexp = row_max + logf(denom);
    out.logprob = chosen_logit - logsumexp;
    out.top_alternatives.reserve((size_t)top_n);
    for (int i = 0; i < top_n; i++) out.top_alternatives.emplace_back(ids[i], logits[i] - logsumexp);
    return out;
}

Qwen35Model::TokenLogprob Qwen35Model::token_logprob_for(int token_id, int top_n) const {
    Impl& s = *p_;
    TokenLogprob out;
    if (token_id < 0 || token_id >= s.cfg.vocab) return out;   // caller validates; be defensive
    // The sort/scan half of the distribution (d_sorted_logits / d_topk_cumsum / d_rank_by_id) is
    // whatever the preceding forward_token() or prefill left behind and is reused as-is -- it
    // describes the full vocab, so it already contains this token's logit at rank
    // d_rank_by_id[token_id]. Only the single-element "which token did we pick" extraction has to
    // be redone, which is the same <<<1,1>>> kernel the decode path runs, pointed at our id
    // instead of the argmax's. That keeps a teacher-forced score numerically IDENTICAL to what
    // /v1/chat/completions would report for the same token at the same position: same logits,
    // same fp32 logsumexp, same subtraction.
    //
    // The id travels as a by-value kernel ARGUMENT, never through a device buffer. Staging it
    // with cudaMemcpy(..., HostToDevice) queues the write on the legacy default stream while this
    // kernel runs on s.stream (cudaStreamNonBlocking), which does not serialize against it -- so
    // the kernel could read the id from the PREVIOUS call and return the wrong token's logit,
    // with the argmax and top_logprobs (which come from the sort, not from this lookup) still
    // perfectly correct. That was issue #1001: ~22 nats of error at position 1 of every
    // /v1/score, because the stale id was the token scored immediately before.
    kernels::launch_extract_chosen_logit_id(token_id, s.d_rank_by_id, s.d_sorted_logits,
                                            s.d_chosen_logit, s.stream);
    cu(cudaStreamSynchronize(s.stream), "token_logprob_for sync");
    out = last_token_logprobs(top_n);   // reads the d_chosen_logit we just rewrote
    out.token_id = token_id;            // ...whose token_id would otherwise be the argmax's
    return out;
}

void Qwen35Model::dflash_maybe_capture_layer(int layer) {
    Impl& s = *p_;
    if (!s.dflash_capture || !s.dflash_hidden || s.dflash_n_cap <= 0) return;
    int slot = -1;
    for (int i = 0; i < s.dflash_n_cap; i++) {
        if (s.dflash_layer_ids[i] == layer) { slot = i; break; }
    }
    if (slot < 0) return;
    const int H = s.cfg.hidden;
    // Writes dflash_hidden[cap_row][slot] directly. cap_row is read from d_scalars[4] on the
    // device, so this node is graph-capturable even though the row changes every verify token --
    // which retires the staging buffer and the extra out-of-graph flush memcpy that every DFlash
    // verify token paid on top of a plain decode forward.
    // (dual-GPU) A split capture stores this card's columns of the row only.
    const int ch = s.cap_h();
    (void)H;
    dflash_kernels::launch_capture_row(s.x + s.dflash_cap_off, s.dflash_hidden, s.d_cap_row, slot,
                                       ch, s.dflash_n_cap * ch, s.dflash_max_rows, s.stream);
}

// Depth-adaptive KV-split count for a given seqlen: 32 (short) -> 128 (mid) -> 256 (long), plus
// the hd256/GQA occupancy correction. Extracted so both forward_token()'s per-token adaptation
// and dflash_generate()'s one-time pre-capture initialization compute the exact same value —
// see forward_token() below for why dflash_generate needs its own call to this.
int Qwen35Model::adaptive_nsplits_for(int seqlen) const {
    const Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    int want = 32;
    if ((long)seqlen > 2L * s.split_chunk) want = 128;
    if ((long)seqlen > 28L * s.split_chunk && (long)seqlen <= 48L * s.split_chunk)
        want = Impl::MAX_NSPLITS;
    if ((long)seqlen > 64L * s.split_chunk) want = Impl::MAX_NSPLITS;
    if (want > Impl::MAX_NSPLITS) want = Impl::MAX_NSPLITS;
    // hd256/GQA-8 occupancy correction (Qwen3.6 full-attention shape specifically) — see the
    // #707-era measurement notes this replaces for the exact tuning rationale (flat 160 through
    // 32k for GQA-8; GQA-4 promotes further at 64k/128k).
    if (c.head_dim == 256 && c.n_kv_heads > 0 && want >= 128) {
        if (c.n_q_heads == c.n_kv_heads * 8)
            want = 160;
        else if (c.n_q_heads == c.n_kv_heads * 6) {
            // Qwen3.8-27B's 24Q/4KV full-attention shape. The 8:1 and 4:1 groups above have had an
            // occupancy correction since #707; the 6:1 group never got one and fell through to the
            // raw MAX_NSPLITS, which costs it a whole CTA wave.
            //
            // The kernel this dispatches to (fa_split_gqa_mma_i8_kernel<256,6>) holds THREE blocks
            // per SM, so a 170-SM part runs 510 CTAs at once. The grid is n_kv_heads * n_splits, so
            // 256 asks for 1024 CTAs -- two full waves plus a third that carries FOUR. Dropping
            // into the two-wave band deletes that tail; the cost is a slightly longer chunk and one
            // ragged 128-token group at its end, which is worth far less than the wave.
            //
            // Measured at ctx=262144 on the ModelOpt NVFP4 checkpoint, decode tok/s (3 reps each at
            // the ends): 256 -> 55.70 · 255 -> 58.81 · 252 -> 58.86 · 250 -> 57.32 · 248 -> 58.64.
            // 252 is the measured peak of the band, and beats 255 in every rep.
            //
            // Only the long-context band moves: below want == MAX_NSPLITS -- every context up to
            // 16384, which is every concurrent-decode and short-decode path -- this is not reached.
            if (want == Impl::MAX_NSPLITS) want = 252;
        }
        else if (c.n_q_heads == c.n_kv_heads * 4) {
            if ((long)seqlen > 98304L)           want = 128;  // 128k decode (seqlen ~131k)
            else if ((long)seqlen > 65536L)      want = 192;  // 64k decode band
            else                                 want = 160;
        }
    }
    return want;
}

std::recursive_mutex& Qwen35Model::device_mutex() { return p_->device_mu; }

bool Qwen35Model::set_pending_vision(const float* emb, const int* positions, int n_img, int hidden) {
    TP_MIRROR(set_pending_vision(emb, positions, n_img, hidden));
    Impl& s = *p_;
    clear_pending_vision();
    if (!emb || !positions || n_img <= 0) return false;
    if (hidden != s.cfg.hidden) return false;   // merger output must match the LM embedding width
    const size_t n = (size_t)n_img * hidden;
    std::vector<bf16> h(n);
    for (size_t i = 0; i < n; i++) h[i] = f32_to_bf16_host(emb[i]);
    if (cudaMalloc(&s.d_vision_emb, n * sizeof(bf16)) != cudaSuccess) return false;
    if (cudaMalloc((void**)&s.d_vision_pos, (size_t)n_img * sizeof(int)) != cudaSuccess) {
        cudaFree(s.d_vision_emb); s.d_vision_emb = nullptr; return false;
    }
    if (cudaMemcpy(s.d_vision_emb, h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice) != cudaSuccess
     || cudaMemcpy(s.d_vision_pos, positions, (size_t)n_img * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        clear_pending_vision(); return false;
    }
    s.vision_n = n_img;
    return true;
}

void Qwen35Model::clear_pending_vision() {
    TP_MIRROR(clear_pending_vision());
    Impl& s = *p_;
    if (s.d_vision_emb) cudaFree(s.d_vision_emb);
    if (s.d_vision_pos) cudaFree(s.d_vision_pos);
    s.d_vision_emb = nullptr; s.d_vision_pos = nullptr; s.vision_n = 0;
}

bool Qwen35Model::set_pending_mrope(const int* positions, int n_tokens, int decode_offset) {
    TP_MIRROR(set_pending_mrope(positions, n_tokens, decode_offset));
    Impl& s = *p_;
    clear_pending_mrope();
    if (!positions || n_tokens <= 0) return false;
    const size_t n = (size_t)n_tokens * 3;
    if (cudaMalloc((void**)&s.d_mrope_pos, n * sizeof(int)) != cudaSuccess) return false;
    if (cudaMemcpy(s.d_mrope_pos, positions, n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        clear_pending_mrope();
        return false;
    }
    s.mrope_n = n_tokens;
    // Survives clear_pending_mrope deliberately: the positions are consumed by ONE prefill, but
    // the offset has to keep applying to every decode step of the same session afterwards.
    s.mrope_pos_offset = decode_offset;
    return true;
}

void Qwen35Model::clear_pending_mrope() {
    TP_MIRROR(clear_pending_mrope());
    Impl& s = *p_;
    if (s.d_mrope_pos) cudaFree(s.d_mrope_pos);
    s.d_mrope_pos = nullptr;
    s.mrope_n = 0;
}

void Qwen35Model::reset_mrope_offset() {
    TP_MIRROR(reset_mrope_offset());
    p_->mrope_pos_offset = 0;
}

int Qwen35Model::forward_token(int token_id, int position, bool sample, float temperature,
                               unsigned long long seed, unsigned long long sample_step,
                               int top_k, float top_p,
                               float presence_penalty, float frequency_penalty) {
    // (dual-GPU WP-9) tp=2: the weights are split across the GpuLink pair, so the step runs
    // on the per-rank twin instead; this guard is the ONLY tp>1 delta on the tp=1 path.
    if (tp_active()) {
        TP_MIRROR(forward_token(token_id, position, sample, temperature, seed, sample_step,
                                top_k, top_p, presence_penalty, frequency_penalty));
        return forward_token_tp(token_id, position, sample, temperature, seed, sample_step,
                                 top_k, top_p, presence_penalty, frequency_penalty);
    }
    // Held for the whole call, not just the capture window. The window has four exits
    // (three EndCapture sites plus the replay-instead-of-capture early path), and a lock that
    // has to be released on every one of them is a lock that will eventually be leaked by an
    // edit. worker_loop() is single-threaded so this never contends with another decode step --
    // the only waiter is a submit on the HTTP thread, which pays at most one step (~ms).
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    kernels::GemmConfig gc{};
    int seqlen = position + 1;
    cudaStream_t st = s.stream;
    const bool dflash_cap = s.dflash_capture;

    // DEBUG ONLY (Muse Glimmer bring-up bisection): SPARKINFER_MG_STAGE_DEBUG=1 dumps
    // L2 norm + first 3 values of the residual stream at each pipeline stage, for every
    // forward_token call. Forces a fresh capture+launch every step (never replays an old
    // graph) so the "step" label baked into each debug kernel's launch params is always
    // accurate -- acceptable perf hit for a short debug run only. Remove once the bug hunt
    // concludes; harmless no-op (env unset) otherwise.
    static int mg_dbg = -1;
    if (mg_dbg < 0) { const char* e = getenv("SPARKINFER_MG_STAGE_DEBUG"); mg_dbg = (e && e[0] == '1') ? 1 : 0; }
    // Widened to any dense_ffn hybrid model (originally Muse-Glimmer-only) to reuse this
    // instrumentation for the Qwen3.8-27B bring-up too -- still opt-in via the same env var.
    const bool mgd = (c.muse_glimmer || c.dense_ffn) && mg_dbg;
    if (mgd && s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec); cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr; s.cu_graph = nullptr; s.graph_ready = false;
    }
    // The check above only forces recapture of the PLAIN decode graph. DFlash's own forward_token
    // calls (AR+capture isolation, and the token-loop verify) route through dflash_cap=true, which
    // replays s.cu_dflash_exec instead -- an early return further below that skips all the
    // debug/dump code entirely once that graph is warm, so SPARKINFER_MG_DUMP_STEP silently only
    // ever captured the plain-AR-reference call otherwise (found 2026-08-17 bisecting a dspark
    // divergence that turned out to be unrelated -- kept because the gap was real).
    if (mgd && s.dflash_graph_ready) {
        cudaGraphExecDestroy(s.cu_dflash_exec); cudaGraphDestroy(s.cu_dflash_graph);
        s.cu_dflash_exec = nullptr; s.cu_dflash_graph = nullptr; s.dflash_graph_ready = false;
    }
    auto dbg_bf16 = [&](const void* p, int n, int tag, int layer) {
        if (mgd) kernels::launch_mg_debug_bf16(p, n, tag, layer, position, st);
    };
    auto dbg_f32 = [&](const float* p, int n, int tag, int layer) {
        if (mgd) kernels::launch_mg_debug_f32(p, n, tag, layer, position, st);
    };
    // DEBUG ONLY: SPARKINFER_MG_DUMP_STEP=<position> additionally raw-dumps tag=10 (this
    // layer's pre-attn-norm xn) for EVERY layer at that one decode step, then the final norm,
    // then each layer's post-attn hn, into a [2*n_layers+1,H] bf16 device buffer, D2H-copied
    // and written to SPARKINFER_MG_DUMP_FILE (default /tmp/mg_xn_dump.bin) right after this
    // step's graph launch is synced. Lets a later-layer hypothesis be checked against a
    // from-scratch Python reference seeded with sparkinfer's OWN xn for that layer, without needing to replicate every earlier layer.
    static int dump_step = -2;
    if (dump_step < -1) { const char* e = getenv("SPARKINFER_MG_DUMP_STEP"); dump_step = e ? atoi(e) : -1; }
    const bool mgdump = mgd && dump_step == position;
    if (mgdump && !s.dbg_xn_dump)
        cu(cudaMalloc(&s.dbg_xn_dump, (size_t)(2 * c.n_layers + 1) * H * sizeof(bf16)), "dbg_xn_dump alloc");
    auto dbg_xn_snapshot = [&](const void* p, int layer) {
        if (mgdump) cu(cudaMemcpyAsync(s.dbg_xn_dump + (size_t)layer * H, p, (size_t)H * sizeof(bf16),
                                       cudaMemcpyDeviceToDevice, st), "dbg_xn_dump copy");
    };

    s.h_scalars[0] = token_id;
    // [1] is the ROTARY position, [2] the KV cache slot. They are the same number for every
    // text-only request and diverge under MRoPE: a vision span occupies one cache slot per token
    // but advances the rotary counter by only max(grid_h, grid_w)/merge, so everything generated
    // after an image sits at a LOWER rotary position than its slot index.
    //
    // One scalar is enough for the whole decode-side correction because a generated token is text,
    // and text has all three MRoPE axes equal -- so the three-number position collapses back to
    // one, just shifted. mrope_pos_offset is 0 whenever no vision span was ingested, which makes
    // this line identical to what it was.
    s.h_scalars[1] = position + s.mrope_pos_offset;
    s.h_scalars[2] = position;
    s.h_scalars[3] = seqlen;
    s.h_scalars[4] = s.dflash_cap_row;
    cu(cudaMemcpyAsync(s.d_scalars, s.h_scalars, 5 * sizeof(int), cudaMemcpyHostToDevice, st), "decode scalars");
    // Refreshed on every call (both capture and replay paths) so the decode graph's sampling
    // kernel -- always launched unconditionally inside the captured region, see below -- picks
    // up THIS call's temperature/seed/step rather than whatever was baked in at capture time.
    *s.h_sample_temp = temperature;
    *s.h_sample_seed = seed;
    *s.h_sample_step = sample_step;
    cu(cudaMemcpyAsync(s.d_sample_temp, s.h_sample_temp, sizeof(float), cudaMemcpyHostToDevice, st), "sample temp");
    cu(cudaMemcpyAsync(s.d_sample_seed, s.h_sample_seed, sizeof(unsigned long long), cudaMemcpyHostToDevice, st), "sample seed");
    cu(cudaMemcpyAsync(s.d_sample_step, s.h_sample_step, sizeof(unsigned long long), cudaMemcpyHostToDevice, st), "sample step");
    *s.h_sample_top_k = top_k;
    *s.h_sample_top_p = top_p;
    cu(cudaMemcpyAsync(s.d_sample_top_k, s.h_sample_top_k, sizeof(int), cudaMemcpyHostToDevice, st), "sample top_k");
    cu(cudaMemcpyAsync(s.d_sample_top_p, s.h_sample_top_p, sizeof(float), cudaMemcpyHostToDevice, st), "sample top_p");
    *s.h_sample_presence_penalty = presence_penalty;
    *s.h_sample_frequency_penalty = frequency_penalty;
    cu(cudaMemcpyAsync(s.d_sample_presence_penalty, s.h_sample_presence_penalty, sizeof(float), cudaMemcpyHostToDevice, st), "sample presence_penalty");
    cu(cudaMemcpyAsync(s.d_sample_frequency_penalty, s.h_sample_frequency_penalty, sizeof(float), cudaMemcpyHostToDevice, st), "sample frequency_penalty");

    // Depth-adaptive KV-split (see adaptive_nsplits_for() above for the tiers/occupancy math).
    // DFlash DECODE: freeze n_splits once an actual dflash verify graph is captured. Adapting it
    // while that graph is live (e.g. the 32->160 jump at seqlen>2*split_chunk ~= 512) invalidates
    // + re-captures it mid-stream, corrupting the compact-verify state (spurious token 0, then
    // repeat). Gating on dflash_capture alone (rather than dflash_capture && dflash_graph_ready)
    // was wrong: dflash_capture is set true before prefill even starts, so that blanket guard also
    // froze n_splits during PREFILL, where sample=false never captures any graph at all and there
    // is nothing to protect. Prefill then ran every position (short depths included) at whatever
    // single value happened to be inherited from framework state predating this call, instead of
    // ramping 32->128->160 with depth the way the reference (AR / pre-freeze) path does — a small
    // per-split floating-point rounding difference in the quantized-KV reduction that compounds
    // over thousands of positions into hidden states the draft model no longer agrees with
    // (verified: SPEC_AGREE collapses at 4k-ctx even though the final prefill logits/argmax token
    // are unaffected — only DFlash's own stashed hidden-state capture diverges). Gating on
    // dflash_graph_ready specifically lets prefill keep adapting exactly like the non-DFlash path,
    // and only starts freezing once there is a live graph that a change would actually corrupt.
    if (s.adaptive_splits && !(s.dflash_capture && s.dflash_graph_ready)) {
        // DFlash's decode graph freezes n_splits the instant it is captured (below) and never
        // revisits it again -- so if the freeze happens to land on a call where the CURRENT
        // position is still in a lower tier than where the rest of the generation will run (e.g.
        // a prompt just under 512 tokens, whose decode phase crosses seqlen>512 partway through),
        // the frozen value is permanently wrong for the back half of the run. AR's own graph
        // doesn't have this problem in isolation -- it re-adapts every step, safely recapturing
        // as seqlen grows -- but that means AR and a frozen DFlash graph would legitimately use
        // DIFFERENT split counts for the SAME position (AR=32, DFlash=160, both individually
        // correct for their own path but mismatched against each other), which is its own
        // source of divergence. So both sides peek ahead at the very moment their own first
        // decode-graph capture happens (final_seqlen_hint, set by generate()/dflash_generate()
        // before their prefill loop) and settle on the SAME tier for the whole decode phase,
        // rather than one side transitioning mid-run and the other starting there already.
        // Elsewhere (prefill positions, and any later step once a graph is already ready) this
        // branch doesn't fire, so prefill keeps adapting exactly as before.
        int want = adaptive_nsplits_for(seqlen);
        // DFlash only needs the hint applied once, right as its graph freezes (below) -- after
        // that this whole outer branch stops running (dflash_graph_ready gates it off), so the
        // frozen value simply stays. AR has no such freeze: it re-derives want from the CURRENT
        // seqlen on every single decode step and recaptures whenever that differs from s.n_splits
        // (by design, so it can safely track a long prefill's own adaptation). Applying the hint
        // to AR only once would just get overwritten back down on the very next step once seqlen
        // no longer matches the hinted tier -- it has to be applied on every decode step so AR
        // settles on and stays at the same tier DFlash is frozen at, instead of legitimately
        // transitioning mid-decode while DFlash cannot follow.
        const bool about_to_freeze_dflash = s.dflash_capture && !s.dflash_graph_ready;
        const bool ar_decode_step = sample && !s.dflash_capture;
        if (((sample && about_to_freeze_dflash) || ar_decode_step) && s.final_seqlen_hint > seqlen) {
            const int want_final = adaptive_nsplits_for(s.final_seqlen_hint);
            if (want_final > want) want = want_final;
        }
        if (want != s.n_splits) {                       // changed -> invalidate the captured graph
            s.n_splits = want;
            if (s.graph_ready) {
                cudaGraphExecDestroy(s.cu_exec); cudaGraphDestroy(s.cu_graph);
                s.cu_exec = nullptr; s.cu_graph = nullptr;
                s.graph_ready = false;
            }
            if (s.graph_prefill_ready) {
                cudaGraphExecDestroy(s.cu_prefill_exec); cudaGraphDestroy(s.cu_prefill_graph);
                s.cu_prefill_exec = nullptr; s.cu_prefill_graph = nullptr;
                s.graph_prefill_ready = false; s.graph_prefill_attn_mode = -1;
            }
            if (s.dflash_graph_ready) {
                cudaGraphExecDestroy(s.cu_dflash_exec); cudaGraphDestroy(s.cu_dflash_graph);
                s.cu_dflash_exec = nullptr; s.cu_dflash_graph = nullptr;
                s.dflash_graph_ready = false; s.dflash_graph_attn_mode = -1;
            }
        }
    }
    // launch_flash_decode_split chooses its scalar-vs-MMA implementation on the host
    // while the graph is captured. If int8 KV is enabled for a long-context run, a graph
    // captured at a short seqlen would otherwise keep replaying the scalar int8 path after
    // the sequence is large enough for the tensor-core path. Recapture at that mode change.
    static int famma_graph = -1;
    if (famma_graph < 0) {
        const char* e = getenv("SPARKINFER_FAMMA");
        famma_graph = (e && e[0] == '0') ? 0 : 1;
    }
    static int famma4_graph = -1;
    if (famma4_graph < 0) {
        const char* e = getenv("SPARKINFER_FAMMA4");
        famma4_graph = (e && e[0] == '0') ? 0 : 1;
    }
    int attn_graph_mode = 0;
    if (famma_graph && s.kv->int8_kv() && s.kv->block_size() == 16 &&
        c.n_kv_heads > 0 && c.n_q_heads == c.n_kv_heads * 8) {
        const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
        attn_graph_mode = (seqlen > 512 && mma_chunk >= 32) ? 2 : 1;
    } else if (famma4_graph && s.kv->int8_kv() && s.kv->block_size() == 16 &&
               c.n_kv_heads > 0 && c.n_q_heads == c.n_kv_heads * 4) {
        const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
        attn_graph_mode = (seqlen > 512 && mma_chunk >= 32) ? 3 : 1;
    }
    if (s.graph_ready && attn_graph_mode != s.graph_attn_mode) {
        cu(cudaGraphExecDestroy(s.cu_exec), "graph recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_graph), "graph recapture destroy graph");
        s.cu_exec = nullptr;
        s.cu_graph = nullptr;
        s.graph_ready = false;
    }
    if (s.dflash_graph_ready && attn_graph_mode != s.dflash_graph_attn_mode) {
        cu(cudaGraphExecDestroy(s.cu_dflash_exec), "dflash graph recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_dflash_graph), "dflash graph recapture destroy graph");
        s.cu_dflash_exec = nullptr;
        s.cu_dflash_graph = nullptr;
        s.dflash_graph_ready = false;
    }
    const bool sparse_avail = s.sparse_budget > 0 && s.kv->int8_kv() &&
                              c.head_dim == 256 && c.n_q_heads == c.n_kv_heads * 4;
    // GQA-8 compact-view sparse is decode-only (`sample`): prefill and teacher-forced
    // scoring always run the exact dense path, and the windowed view is never baked into
    // the prefill graph.
    // Stays gated on int8 KV. Lifting it does speed the AR leg (+11.2% at ctx=32768), but the
    // DSpark verify path does not consult the view, so the two legs stop agreeing and
    // dspark_tau_check reports LOSSLESS=0 -- a hard gate. Both legs would have to become sparse
    // together before this could be relaxed.
    const bool sparse_view_avail = s.sparse_vtbl != nullptr && sample && s.kv->int8_kv() &&
                                   c.head_dim == 256 && c.n_kv_heads > 0 &&
                                   (c.n_q_heads == c.n_kv_heads * 6 ||
                                    c.n_q_heads == c.n_kv_heads * 8);
    const bool sparse_on = (sparse_avail || sparse_view_avail) && seqlen >= s.sparse_min_ctx;
    if (s.graph_ready && s.graph_sparse != sparse_on) {
        cu(cudaGraphExecDestroy(s.cu_exec), "sparse recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_graph), "sparse recapture destroy graph");
        s.cu_exec = nullptr; s.cu_graph = nullptr; s.graph_ready = false;
    }
    // The capture bakes WHICH GDN state representation its kernels read (launch_qwen36_gdn_ar's
    // state_compact_b16 selects a different kernel instantiation). decode_packed compacts a
    // session's state to bf16 the first time it packs one, which can land between two of that
    // session's own decode steps -- and a request that decodes alone, joins a batch, then outlives
    // it hits exactly that order. Replaying the fp32 capture over the compacted state reads every
    // element at the wrong width. The dflash verify graph needs no equivalent: decode_packed is
    // the only compactor and DSpark's verify never sets packed_rows, so its sessions stay fp32.
    if (s.graph_ready && s.graph_state_b16 != s.active_lin_state_b16) {
        cu(cudaGraphExecDestroy(s.cu_exec), "gdn state recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_graph), "gdn state recapture destroy graph");
        s.cu_exec = nullptr; s.cu_graph = nullptr; s.graph_ready = false;
    }
    if (s.dflash_graph_ready && s.dflash_graph_sparse != sparse_on) {
        cu(cudaGraphExecDestroy(s.cu_dflash_exec), "dflash sparse recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_dflash_graph), "dflash sparse recapture destroy graph");
        s.cu_dflash_exec = nullptr; s.cu_dflash_graph = nullptr; s.dflash_graph_ready = false;
    }
    if (s.graph_prefill_ready && attn_graph_mode != s.graph_prefill_attn_mode) {
        cu(cudaGraphExecDestroy(s.cu_prefill_exec), "prefill graph recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_prefill_graph), "prefill graph recapture destroy graph");
        s.cu_prefill_exec = nullptr;
        s.cu_prefill_graph = nullptr;
        s.graph_prefill_ready = false;
        s.graph_prefill_attn_mode = -1;
    }
    // The buffers behind these two resets are allocated only for a stack that actually carries
    // Gated-DeltaNet layers (see needs_linear_state), so a `hybrid` stack with none of them has
    // nothing to zero and the pointers are null. The sibling reset in qwen35_prefill.cpp guards on
    // the pointers for the same reason; match it, so this is also safe if the alloc ever fails.
    if (c.hybrid && position == 0 && s.lin_state && s.lin_conv_state) {
        cu(cudaMemsetAsync(s.lin_state, 0,
                           (size_t)gdn_state_slots(c) * gdn_v_local() * c.linear_head_dim * c.linear_head_dim * sizeof(float), st),
           "linear state reset");
        cu(cudaMemsetAsync(s.lin_conv_state, 0,
                           (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim * sizeof(bf16), st),
           "linear conv reset");
    }

    // Prefill graph: embed→layers→final norm (no LM head). Decode graph: full path + argmax.
    // DFlash's per-layer hidden capture used to require the eager path entirely (a captured
    // graph bakes fixed destinations, and dflash_hidden[cap_row] varies call to call) -- the
    // capture kernel reads cap_row from device memory instead, so DFlash verify tokens
    // (sample=true) get graph replay with no post-replay fixup. The prefill/prompt path
    // (sample=false) still falls back to eager under capture; not on the scored decode path.
    if (!sample && s.graph_prefill_ready && !dflash_cap) {
        cu(cudaGraphLaunch(s.cu_prefill_exec, st), "prefill graph launch");
        cu(cudaStreamSynchronize(st), "prefill graph sync");
        return token_id;
    }
    if (sample && s.graph_ready && !dflash_cap) {
        cu(cudaGraphLaunch(s.cu_exec, st), "graph launch");
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
        cu(cudaStreamSynchronize(st), "sync");
        return *s.h_out_id;
    }
    if (sample && dflash_cap && s.dflash_graph_ready) {
        cu(cudaGraphLaunch(s.cu_dflash_exec, st), "dflash graph launch");
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
        // Deferred collect (DFlash verify token 0 only): return without blocking so the caller
        // can issue the draft block behind it. The target forward is then already running on the
        // GPU while the host is still issuing the draft's ~100 launches -- and it costs no
        // std::thread spawn/join per decode step.
        if (s.defer_decode_sync) { s.decode_pending = true; return kDFlashDeferred; }
        cu(cudaStreamSynchronize(st), "sync");
        return *s.h_out_id;
    }
    if (sample && s.graph_prefill_ready) {
        cu(cudaGraphExecDestroy(s.cu_prefill_exec), "drop prefill graph for decode");
        cu(cudaGraphDestroy(s.cu_prefill_graph), "drop prefill graph for decode");
        s.cu_prefill_exec = nullptr;
        s.cu_prefill_graph = nullptr;
        s.graph_prefill_ready = false;
        s.graph_prefill_attn_mode = -1;
    }
    // Reaching here means: no ready graph could be replayed above. For sample=true this is
    // always safe to (re)capture -- either the plain decode graph (dflash_cap false) or the
    // dflash decode graph (dflash_cap true, via the row-independent stage buffer); only the
    // prefill/prompt path (sample=false) still avoids capturing while dflash_cap is on.
    const bool capturing_graph = sample || !dflash_cap;
    if (capturing_graph)
        cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), sample ? "begin decode capture" : "begin prefill capture");

    if (s.bonsai_embed_native) {
        // The table's rows are stored rotated -- that is what made quantising them to trits
        // survivable -- so the row comes back in that basis and the inverse comes off here.
        const auto it = s.bonsai_sign_dev.find(H);
        kernels::launch_embedding_ptq1_unrotate(s.d_tok, s.w.embed_tokens,
                                                static_cast<const signed char*>(it->second),
                                                s.x, 1, H, (int)s.bonsai_block, st);
    } else {
        kernels::launch_embedding(s.d_tok, s.w.embed_tokens, s.x, 1, H, st);
    }
    dbg_bf16(s.x, H, 0, -1);   // tag 0: post-embedding, pre emb_norm
    if (c.muse_glimmer && s.emb_norm_ones)
        kernels::launch_rmsnorm(s.x, s.emb_norm_ones, s.x, 1, H, c.rms_eps, st);
    dbg_bf16(s.x, H, 1, -1);   // tag 1: post emb_norm

    int* btable = s.kv->block_table(s.active_seq_id);
    // The windowed layers' ring table (== btable unless this pool has capped slices).
    int* btable_win = s.kv->block_table_win(s.active_seq_id);
    // GQA-8 sparse: materialize the sink+window compact view once per decode step (the
    // logical->physical block map and seq_len are shared by all full-attn layers; the
    // per-layer K/V pool rows for this token are appended before each layer's attention
    // as usual). Inside the capture so replays track the growing sequence.
    if (sparse_on && s.sparse_vtbl)
        kernels::launch_fa_kv_compact_view(s.d_seqlen, btable, s.sparse_vtbl, s.sparse_vlen,
                                           s.kv->block_size(), s.sparse_window, s.sparse_budget, st);
    // Muse Glimmer: pure sliding-window view for swa-flagged layers, every step (mandatory,
    // not gated by context length like the sparse-kv approximation above).
    if (c.muse_glimmer && s.swa_vtbl)
        kernels::launch_fa_kv_compact_view_pure(s.d_seqlen, btable_win, s.swa_vtbl, s.swa_vlen,
                                                s.kv->block_size(), s.swa_budget, s.swa_budget, st);
    // Prime: xn = RMSNorm(x, layer0.input_norm). Each layer's tail then fuses the
    // post-MoE residual with the NEXT layer's input norm (or final_norm), so the
    // per-layer input RMSNorm + two residual-adds collapse into two fused kernels.
    kernels::launch_rmsnorm(s.x, s.w.layers[0].input_norm, s.xn, 1, H, c.rms_eps, st);

    // When the fused norm+quant path is on, each layer's post-MoE add_rmsnorm2 also emits
    // Q8_1(xn) into aq81, so the next layer's QKV-input quantize (and the LM-head quantize)
    // are already done. Only the prime norm above still needs the standalone quant (layer 0).
    const bool fnq = s.gguf && s.use_fnq && s.use_pq && s.use_llama;

    // Muse Glimmer's sandwich tail can now emit the Q8_1 its consumer needs, the way
    // add_rmsnorm2_q8 does for every other architecture. Both flags record what the tail
    // ACTUALLY did (it declines on shapes its register path cannot serve), never what this
    // architecture is assumed to do -- a wrong assumption here is the stale-aq81 failure the
    // comments at the FFN and QKV call sites describe, and it degrades silently.
    // muse_xn_q8 crosses the layer boundary: layer L's post-FFN tail feeds layer L+1's Q/K/V.
    bool muse_hn_q8 = false, muse_xn_q8 = false;

    // ---- L2 weight prefetch into the sandwich-norm tail's idle window ----
    // Decode is DRAM-bound but only ~86% bus-utilized; the gap is time the bus idles inside a
    // latency-bound kernel. Muse Glimmer's two single-CTA sandwich tails per layer are the largest
    // such window (~3.4 us each on one of 170 SMs). Fork a prefetch of the weight matrix the next
    // big GEMV will stream, so it overlaps the tail instead of leaving the bus idle. Each join sits
    // a full block after its fork (FFN between fork1/join1, attention between fork2/join2), so the
    // prefetch never serializes against the main stream. Prefetch has no side effects, so this is
    // bit-identical -- only where a byte is served from changes.
    // SPARKINFER_MG_L2PF_MB=0 disables (one-binary A/B control).
    static int l2pf_mb = -1;
    if (l2pf_mb < 0) {
        const char* e = getenv("SPARKINFER_MG_L2PF_MB");
        l2pf_mb = e ? atoi(e) : 5;   // flat optimum over 3-5 MB
        if (l2pf_mb < 0) l2pf_mb = 0;
    }
    // Each window is forked immediately before a latency-bound stretch and joined immediately
    // before the matrix it prefetched is streamed, so the prefetch always has the full window to
    // land and never stalls the main stream. Forking earlier than this is WORSE, not better: a
    // variant that spanned the whole attention block measured +0.54% against this schedule's
    // +0.99%, because attention streams ~47 MB through a 96 MB L2 and evicts the prefetch before
    // the FFN reads it. Sizing is a flat optimum over 3-6 MB; past ~24 MB the prefetch outruns its
    // window and the join serializes, which costs more than half the step.
    // Window 1 gets its own size: its runway is the whole attention block (~35 us), an order of
    // magnitude longer than the two ~4 us sandwich-tail windows, so it can absorb far more of
    // w.wo than they can of gate/up. Sizing them together undershoots window 1 and overshoots 2/3.
    static int l2pf_wo_mb = -1;
    if (l2pf_wo_mb < 0) {
        const char* e = getenv("SPARKINFER_MG_L2PF_WO_MB");
        l2pf_wo_mb = e ? atoi(e) : 8;   // swept 8/12/16: 101.57 / 101.55 / 101.46 tok/s
        if (l2pf_wo_mb < 0) l2pf_wo_mb = 0;
    }
    const bool l2pf = c.muse_glimmer && s.gguf && l2pf_mb > 0 && s.stream_pf != nullptr;
    const size_t pf_bytes = (size_t)l2pf_mb << 20;
    const size_t pf_wo_bytes = (size_t)l2pf_wo_mb << 20;
    bool pf_outstanding = false;
    auto pf_join = [&]() {
        if (!pf_outstanding) return;
        cu(cudaStreamWaitEvent(st, s.ev_pf_done, 0), "l2 prefetch join");
        pf_outstanding = false;
    };
    auto pf_fork_n = [&](const void* a, const void* b, size_t nbytes) {
        if (!l2pf || !nbytes || (!a && !b)) return;
        cu(cudaEventRecord(s.ev_pf_fork, st), "l2 prefetch fork");
        cu(cudaStreamWaitEvent(s.stream_pf, s.ev_pf_fork, 0), "l2 prefetch fork wait");
        if (a) kernels::launch_l2_prefetch(a, nbytes, s.stream_pf);
        if (b) kernels::launch_l2_prefetch(b, nbytes, s.stream_pf);
        cu(cudaEventRecord(s.ev_pf_done, s.stream_pf), "l2 prefetch done");
        pf_outstanding = true;
    };
    auto pf_fork = [&](const void* a, const void* b) { pf_fork_n(a, b, pf_bytes); };
    // A single fork per layer issuing every prefetch back-to-back was tried and is WORSE than
    // doing nothing (98.37 vs 99.44 tok/s): the side stream then hammers DRAM through the QKV
    // projections it is supposed to hide behind, and gate/up is evicted long before the FFN reads
    // it. Each window must be forked immediately before the stretch it covers.
    // Window mask: 1 = w.wo across the attention block, 2 = gate/up across the post-attn tail,
    // 4 = next layer's wq across the post-FFN tail. A window only earns its place if it beats the
    // ~0.89%/step its own fork+join event nodes cost.
    static int pf_win = -1;
    if (pf_win < 0) { const char* e = getenv("SPARKINFER_MG_L2PF_WIN"); pf_win = e ? atoi(e) : 7; }

    for (int L = 0; L < c.n_layers; L++) {
        // The decode shadow's weights, read through the dp4a GEMV. Native residency keeps the
        // float kernels its packed batch also runs, so a row decodes the same alone or batched.
        const bool dec_shadow = !s.bonsai_dec_layers.empty();
        const Qwen35LayerWeights& w = dec_shadow ? s.bonsai_dec_layers[L] : s.w.layers[L];
        // Which block table this layer's KV lives in. A sliding-window layer may sit in a capped
        // RING slice (KVCacheConfig::window_tokens), whose logical->physical map is its own; a
        // full-causal layer always takes the full one. block_table_win() IS block_table() on a
        // pool with no windowed slices, so this is the identity everywhere else.
        int* ltab = w.swa ? btable_win : btable;
        // Window 3 lands: the previous layer's post-FFN prefetch covered its sandwich tail.
        pf_join();
        // Window 1: the QKV projections are latency-bound, not bandwidth-bound (~32 MB over ~29 us
        // = ~1.1 TB/s of a ~1.66 TB/s bus), so the whole attention block has spare bandwidth. w.wo
        // is streamed at the end of it and only has to survive QKV's ~32 MB through a 96 MB L2, so
        // it is prefetchable across that entire runway -- unlike gate/up, which would have to
        // survive attention's full ~47 MB and measured worse when tried that way.
        if (pf_win & 1) pf_fork_n(w.wo, nullptr, pf_wo_bytes);
        dbg_bf16(s.xn, H, 10, L);   // tag 10: pre-attn-norm output (this layer's normed input)
        dbg_xn_snapshot(s.xn, L);
        int xn_ptq1_q = -1;
        if (s.bonsai_rot_xn) {
            // Once, on the main stream, before the projections fan out across stream_k/stream_v.
            // It also leaves the int8 copy every ternary projection below reads (-1: it did not,
            // and each GEMV quantizes for itself as before).
            const auto it = s.bonsai_sign_dev.find(H);
            if (dec_shadow)
                xn_ptq1_q = kernels::launch_ptq1_rotate_quant(
                    s.xn, s.bonsai_rot_xn, static_cast<const signed char*>(it->second), (int)H,
                    (int)s.bonsai_block, st);
            else
                kernels::launch_hadamard_rotate_bf16(s.xn, s.bonsai_rot_xn,
                                                     static_cast<const signed char*>(it->second),
                                                     H, (int)H, (int)s.bonsai_block, st);
            // tag 15: the rotated xn. R is orthogonal, so this l2 must equal tag 10's exactly --
            // a cheap in-model check that the rotation is what the isolated test says it is.
            dbg_bf16(s.bonsai_rot_xn, H, 15, L);
        }
        // xn_q8_ready assumes the PREVIOUS layer's tail already emitted Q8_1(this layer's xn)
        // into s.aq81 as a side effect (true for architectures whose post-MoE tail runs
        // launch_add_rmsnorm2_q8 / add_rmsnorm3_q8). Muse Glimmer's tail is the sandwich-norm
        // pair launch_norm_then_add + a plain launch_rmsnorm (see the c.muse_glimmer branch
        // below) -- neither emits a Q8 side channel. Trusting xn_q8_ready==true here for L>0
        // left s.aq81 permanently stuck holding layer 0's Q8_1(xn): every K/V projection at
        // L=1..n_layers-1 (both wk_type=12 Q4_K and wv_type=14 Q6_K route through proj_xn's
        // mmvq_q4k/mmvq_q6k branches, which read s.aq81 unconditionally) ran against the wrong
        // layer's quantized activation. Force a fresh quantize every layer for muse_glimmer.
        bool xn_q8_ready = (fnq && L > 0 && !c.muse_glimmer) || muse_xn_q8;
        // Both flags are re-earned every layer by the tail that actually ran; consumed here, so
        // a layer whose tail declines falls back to its own quantize instead of inheriting.
        muse_hn_q8 = false;
        muse_xn_q8 = false;
        auto prepare_xn_quant = [&](bool any_q4k, bool any_q6k, bool any_q80) {
            if (!s.gguf || !s.use_pq) return;
            if (xn_q8_ready) return;
            if (s.use_llama && (any_q4k || any_q80 || (s.use_q6mmvq && any_q6k))) {
                kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
                xn_q8_ready = true;
            } else if (any_q4k) {
                kernels::launch_quantize_q8_1(s.xn, s.aq8, s.aq8_d, s.aq8_s, H, st);
            }
        };
        // NVFP4 dp4a staging for xn. Issued on `st` before any stream fork, exactly like the
        // Q8_1 quantize above, so the parallel K/V and GDN streams that read it inherit the same
        // dependency they already have on xn itself. Not gated on use_pq: that flag selects the
        // mmvq activation format, which this path does not use.
        bool xn_nv_ready = false;
        auto prepare_xn_nvfp4 = [&](bool any_nv) {
            if (!any_nv || xn_nv_ready || !kernels::qwen38_nvfp4_dp4a_proj()) return;
            kernels::launch_gemv_nvfp4_quant_x(s.xn, s.nv_pq_a, s.nv_ps_a, 1, H, st);
            xn_nv_ready = true;
        };
        auto proj_xn = [&](const void* W, int t, void* y, int N, cudaStream_t pst) {
            if (s.gguf) {
                if (s.use_pq && t == 12) {
                    if (s.use_llama) kernels::launch_mmvq_q4k(s.aq81, W, y, N, H, pst);
                    else             kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, H, pst);
                }
                else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14)
                    kernels::launch_mmvq_q6k(s.aq81, W, y, N, H, pst);
                else if (s.use_pq && s.use_llama && t == 8)
                    kernels::launch_mmvq_q80(s.aq81, W, y, N, H, pst);
                else if (t == kernels::SI_QTYPE_FP8)
                    kernels::launch_gemv_fp8(s.xn, W, y, N, H, pst);
                else if (t == kernels::SI_QTYPE_NVFP4) {
                    if (!(xn_nv_ready &&
                          kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_pq_a, s.nv_ps_a, W, y,
                                                               1, N, H, pst)))
                        kernels::launch_gemv_nvfp4(s.xn, W, y, N, H, pst);
                }
                else if (t == kPtq1GgmlType && s.bonsai_rot_xn)
                    dec_shadow ? kernels::launch_gemv_ptq1_q(xn_ptq1_q, s.bonsai_rot_xn, W, y, N, H, pst)
                               : kernels::launch_gemv_ptq1(s.bonsai_rot_xn, W, y, N, H, pst);
                else if (t) kernels::launch_gemv_q(s.xn, W, t, y, N, H, pst);
                else        kernels::launch_gemv(s.xn, W, y, N, H, pst);
            } else {
                kernels::launch_gemm(s.xn, W, y, 1, N, H, 1.f, 0.f, gc, pst);
            }
        };
        auto proj_from = [&](const void* x, const void* W, int t, void* y, int N, int K) {
            if (s.gguf) {
                if (s.use_pq && t == 12) {
                    if (s.use_llama) {
                        kernels::launch_quantize_q8_1_blocks(x, s.aq81, K, st);
                        kernels::launch_mmvq_q4k(s.aq81, W, y, N, K, st);
                    } else {
                        kernels::launch_quantize_q8_1(x, s.aq8, s.aq8_d, s.aq8_s, K, st);
                        kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, K, st);
                    }
                } else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14) {
                    kernels::launch_quantize_q8_1_blocks(x, s.aq81, K, st);
                    kernels::launch_mmvq_q6k(s.aq81, W, y, N, K, st);
                } else if (s.use_pq && s.use_llama && t == 8) {
                    kernels::launch_quantize_q8_1_blocks(x, s.aq81, K, st);
                    kernels::launch_mmvq_q80(s.aq81, W, y, N, K, st);
                } else if (t == kernels::SI_QTYPE_FP8) {
                    kernels::launch_gemv_fp8(x, W, y, N, K, st);
                } else if (t == kernels::SI_QTYPE_NVFP4) {
                    if (kernels::qwen38_nvfp4_dp4a_proj()) {
                        kernels::launch_gemv_nvfp4_quant_x(x, s.nv_pq_b, s.nv_ps_b, 1, K, st);
                        if (kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_pq_b, s.nv_ps_b, W, y,
                                                                 1, N, K, st)) return;
                    }
                    kernels::launch_gemv_nvfp4(x, W, y, N, K, st);
                } else if (t) kernels::launch_gemv_q(x, W, t, y, N, K, st);
                else          kernels::launch_gemv(x, W, y, N, K, st);
            } else {
                kernels::launch_gemm(x, W, y, 1, N, K, 1.f, 0.f, gc, st);
            }
        };

        if (w.linear_attn) {
            const bool any_q4k = (w.wqkv_type == 12 || w.wqkv_gate_type == 12 ||
                                  w.ssm_alpha_type == 12 || w.ssm_beta_type == 12);
            const bool any_q6k = (w.wqkv_type == 14 || w.wqkv_gate_type == 14 ||
                                  w.ssm_alpha_type == 14 || w.ssm_beta_type == 14);
            const bool any_q80 = (w.wqkv_type == 8 || w.wqkv_gate_type == 8 ||
                                  w.ssm_alpha_type == 8 || w.ssm_beta_type == 8);
            prepare_xn_quant(any_q4k, any_q6k, any_q80);
            prepare_xn_nvfp4(w.wqkv_type == kernels::SI_QTYPE_NVFP4 ||
                             w.wqkv_gate_type == kernels::SI_QTYPE_NVFP4 ||
                             w.ssm_alpha_type == kernels::SI_QTYPE_NVFP4 ||
                             w.ssm_beta_type == kernels::SI_QTYPE_NVFP4);
            const bool gdn_quad = s.use_gdn_quad && s.gguf && s.use_pq && s.use_llama && H == 2048
                               && w.wqkv_type == 12 && w.wqkv_gate_type == 12
                               && w.ssm_alpha_type == 12 && w.ssm_beta_type == 12;
            const bool gdn_pipelined = !gdn_quad && s.gguf && s.use_gdn_pipe;
            const bool gdn_fused_proj = [&] {
                static int fuse = -1;
                if (fuse < 0) { const char* e = getenv("SPARKINFER_GDN_QKVZ_FUSE");
                    fuse = (e && e[0] == '0') ? 0 : 1; }
                return fuse && s.gguf && s.use_pq && s.use_llama &&
                       w.wqkv_type == 12 && w.wqkv_gate_type == 12 &&
                       (H == 2048 || H == 4096 || H == 5120) && s.linear_qkvdim > 0 && s.linear_vdim > 0;
            }();
            if (gdn_quad) {
                kernels::launch_gdn_quad_mmvq_q4k(s.aq81, w.wqkv, w.wqkv_gate, w.ssm_alpha, w.ssm_beta,
                    s.lin_qkv, s.lin_z, s.lin_alpha, s.lin_beta,
                    s.linear_qkvdim, s.linear_vdim, c.linear_v_heads, c.linear_v_heads, H, st);
            } else if (gdn_fused_proj && gdn_pipelined) {
                cudaEventRecord(s.ev_pipe_fork, st);
                cudaStreamWaitEvent(s.stream_v, s.ev_pipe_fork, 0);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, s.stream_v);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, s.stream_v);
                cudaEventRecord(s.ev_gdn_ab, s.stream_v);
                kernels::launch_mmvq_gdn_qkv_z_pack2(s.aq81, w.wqkv, w.wqkv_gate,
                                                       s.lin_qkv, s.lin_z,
                                                       s.linear_qkvdim, s.linear_vdim, H, st);
            } else if (gdn_pipelined && !gdn_fused_proj) {
                cudaEventRecord(s.ev_pipe_fork, st);
                cudaStreamWaitEvent(s.stream_k, s.ev_pipe_fork, 0);
                cudaStreamWaitEvent(s.stream_v, s.ev_pipe_fork, 0);
                proj_xn(w.wqkv_gate, w.wqkv_gate_type, s.lin_z, s.linear_vdim, s.stream_k);
                cudaEventRecord(s.ev_gdn_z, s.stream_k);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, s.stream_v);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, s.stream_v);
                cudaEventRecord(s.ev_gdn_ab, s.stream_v);
                proj_xn(w.wqkv, w.wqkv_type, s.lin_qkv, s.linear_qkvdim, st);
            } else if (gdn_fused_proj) {
                kernels::launch_mmvq_gdn_qkv_z_pack2(s.aq81, w.wqkv, w.wqkv_gate,
                                                       s.lin_qkv, s.lin_z,
                                                       s.linear_qkvdim, s.linear_vdim, H, st);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, st);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, st);
            } else {
                proj_xn(w.wqkv, w.wqkv_type, s.lin_qkv, s.linear_qkvdim, st);
                proj_xn(w.wqkv_gate, w.wqkv_gate_type, s.lin_z, s.linear_vdim, st);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, st);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, st);
            }

            bf16* conv_state = s.lin_conv_state +
                (size_t)L * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
            // Fused conv_split + l2_norm: one kernel instead of three (SPARKINFER_GDN_FUSE=0 restores split).
            static int gdn_fuse = -1;
            if (gdn_fuse < 0) { const char* e = getenv("SPARKINFER_GDN_FUSE"); gdn_fuse = (e && e[0] == '0') ? 0 : 1; }
            if (gdn_fuse && c.linear_head_dim == 128 && c.linear_q_heads == 16 &&
                (c.linear_v_heads == 32 || c.linear_v_heads == 48)) {
                kernels::launch_qwen36_conv_split_l2norm_fused(s.lin_qkv, w.ssm_conv, conv_state,
                                                 s.lin_q, s.lin_k, s.lin_v,
                                                 c.linear_q_heads, c.linear_v_heads,
                                                 c.linear_head_dim, c.linear_conv_kernel,
                                                 c.rms_eps, st);
            } else {
                kernels::launch_qwen36_conv_split_l2(s.lin_qkv, w.ssm_conv, conv_state,
                                                 s.lin_q, s.lin_k, s.lin_v,
                                                 c.linear_q_heads, c.linear_v_heads,
                                                 c.linear_head_dim, c.linear_conv_kernel,
                                                 c.rms_eps, st);
            }
            if (gdn_pipelined) cudaStreamWaitEvent(st, s.ev_gdn_ab, 0);
            // The layer's slot is handed over as an OFFSET, not folded into the pointer: under a
            // compacted state it counts bf16 elements, and only the kernel knows that. Folding it
            // in here is what made every GDN layer past the first read its slot at twice the right
            // byte offset the moment a session had been through decode_packed.
            //
            // GDN v-head window (dual-GPU state split): this instance owns v-heads
            // [gdn_v0, gdn_v0 + gdn_vloc), so the state slot is vloc*HD^2 and every per-v-head
            // operand is read at its GLOBAL offset. The fast AR kernel is block-diagonal in the
            // v-head, so this is a pure call-site windowing: shift the q/k/v/alpha/beta/dt/a/out
            // bases, pass the local counts, launch. (0,0) window: zero shifts, full counts, the
            // exact launch arguments of the unsplit model.
            const int gdn_vloc = gdn_v_local();
            const int gdn_v0 = p_->gdn_window.v_count > 0 ? p_->gdn_window.v_start : 0;
            const int gdn_g = c.linear_v_heads / c.linear_q_heads;  // v-heads per q-head group
            // q/k are read through the shared q-head: block mode maps global v-head h to q-head
            // h/g, so the bases move by (v0/g) groups; cyclic mode only admits a leading window
            // (v0 == 0), where the bases do not move at all.
            const int gdn_qshift = c.gdn_qh_block ? (gdn_v0 / gdn_g) * c.linear_head_dim : 0;
            const size_t state_off = (size_t)gdn_state_slot(c, L) * gdn_vloc *
                                      c.linear_head_dim * c.linear_head_dim;
            // The compacted-state flag belongs to the ACTIVE session: a packed batch that declines
            // (a tail chunk of one row) falls back to this path for rows whose state has already
            // been converted, so the two must agree on the representation.
            kernels::launch_qwen36_gdn_ar(s.lin_q + gdn_qshift, s.lin_k + gdn_qshift,
                                          s.lin_v + gdn_v0 * c.linear_head_dim,
                                          s.lin_alpha + gdn_v0, s.lin_beta + gdn_v0,
                                          static_cast<const bf16*>(w.ssm_dt) + gdn_v0, static_cast<const bf16*>(w.ssm_a) + gdn_v0,
                                          s.lin_state, state_off, s.lin_gdn + gdn_v0 * c.linear_head_dim,
                                          p_->gdn_window.v_count > 0
                                              ? (c.gdn_qh_block ? gdn_vloc / gdn_g : gdn_vloc)
                                              : c.linear_q_heads,
                                          gdn_vloc,
                                          c.linear_head_dim, c.gdn_qh_block, st,
                                          s.active_lin_state_b16);
            if (gdn_pipelined && !gdn_fused_proj) cudaStreamWaitEvent(st, s.ev_gdn_z, 0);
            const bool gdn_gn_q8 = s.gguf && s.use_pq && s.use_llama &&
                                   (w.ssm_out_type == 12 || w.ssm_out_type == 8) &&
                                   c.linear_head_dim == 128;
            if (gdn_gn_q8) {
                static int gn_q8 = -1;
                if (gn_q8 < 0) {
                    const char* e = getenv("SPARKINFER_GDN_GNORM_Q8");
                    gn_q8 = (e && e[0] == '0') ? 0 : 1;
                }
                if (gn_q8) {
                    kernels::launch_qwen36_gated_norm_q8(s.lin_gdn, s.lin_z, w.ssm_norm, s.aq81,
                                                         c.linear_v_heads, c.linear_head_dim,
                                                         c.rms_eps, st);
                    if (w.ssm_out_type == 12)
                        kernels::launch_mmvq_q4k(s.aq81, w.ssm_out, s.ao, H, s.linear_vdim, st);
                    else
                        kernels::launch_mmvq_q80(s.aq81, w.ssm_out, s.ao, H, s.linear_vdim, st);
                } else {
                    kernels::launch_qwen36_gated_norm(s.lin_gdn, s.lin_z, w.ssm_norm, s.lin_norm,
                                                      c.linear_v_heads, c.linear_head_dim, c.rms_eps, st);
                    proj_from(s.lin_norm, w.ssm_out, w.ssm_out_type, s.ao, H, s.linear_vdim);
                }
            } else {
                kernels::launch_qwen36_gated_norm(s.lin_gdn, s.lin_z, w.ssm_norm, s.lin_norm,
                                                  c.linear_v_heads, c.linear_head_dim, c.rms_eps, st);
                proj_from(s.lin_norm, w.ssm_out, w.ssm_out_type, s.ao, H, s.linear_vdim);
            }
        } else {
            // ---- Q/K/V projection (q_has_gate-aware; q_has_gate=false is byte-identical to Qwen3-MoE) ----
            if (s.gguf) {
                const bool any_q4k = (w.wq_type == 12 || w.wk_type == 12 || w.wv_type == 12);
                const bool any_q6k = (w.wq_type == 14 || w.wk_type == 14 || w.wv_type == 14);
                const bool any_q80 = (w.wq_type == 8 || w.wk_type == 8 || w.wv_type == 8);
                prepare_xn_quant(any_q4k, any_q6k, any_q80);
                prepare_xn_nvfp4(w.wq_type == kernels::SI_QTYPE_NVFP4 ||
                                 w.wk_type == kernels::SI_QTYPE_NVFP4 ||
                                 w.wv_type == kernels::SI_QTYPE_NVFP4);
                // Muse Glimmer keeps attn_gate as its own quantized tensor (w.wgate), so Q goes
                // straight to s.q and the gate straight to s.qgate -- no [q|gate] interleave to
                // build and no split to undo it. Every other model still fuses them into s.qraw.
                const bool sep_gate = (w.wgate != nullptr);
                void* q_dst = (w.q_has_gate && !sep_gate) ? s.qraw : s.q;
                const int nq = (w.q_has_gate && !sep_gate) ? s.qdim * 2 : s.qdim;
                // Muse Glimmer's Q and attn-gate are two separate Q4_K tensors projected
                // back-to-back on the SAME stream. Merge them into one launch: one fewer graph
                // node per layer, and the launch doubles to 18.4 MB which reads faster per byte.
                // K/V are untouched -- QKVSTREAM overlaps them on side streams and folding those
                // in measured -0.13%. SPARKINFER_MG_QG_FUSE=0 restores the split pair.
                static int mg_qg = -1;
                if (mg_qg < 0) { const char* e = getenv("SPARKINFER_MG_QG_FUSE"); mg_qg = (e && e[0] == '0') ? 0 : 1; }
                auto proj_q_gate = [&](cudaStream_t qs) {
                    if (mg_qg && c.muse_glimmer && sep_gate && s.use_pq && s.use_llama &&
                        w.wq_type == 12 && w.wgate_type == 12 && H == 6656 &&
                        kernels::launch_mmvq_q4k_kfixed2(s.aq81, w.wq, w.wgate, q_dst, s.qgate,
                                                         nq, s.qdim, H, qs))
                        return;
                    proj_xn(w.wq, w.wq_type, q_dst, nq, qs);
                    if (sep_gate) proj_xn(w.wgate, w.wgate_type, s.qgate, s.qdim, qs);
                };
                // The fused QKV kernel writes one contiguous q of width nq and knows nothing about
                // a separate gate tensor, so it cannot serve this path.
                const bool attn_qkv = !sep_gate && s.use_attn_qkv && s.use_pq && s.use_llama
                                   && (H == 2048 || H == 4096 || H == 5120)
                                   && w.wq_type == 12 && w.wk_type == 12 && w.wv_type == 12;
                if (attn_qkv) {
                    kernels::launch_attn_qkv_mmvq_q4k(s.aq81, w.wq, w.wk, w.wv,
                        q_dst, s.k, s.v, nq, s.kvdim, s.kvdim, H, st);
                } else if (s.use_qkvstream) {
                    cudaEventRecord(s.ev_qkv, st);
                    cudaStreamWaitEvent(s.stream_k, s.ev_qkv, 0);
                    cudaStreamWaitEvent(s.stream_v, s.ev_qkv, 0);
                    proj_q_gate(st);
                    proj_xn(w.wk, w.wk_type, s.k, s.kvdim, s.stream_k);
                    proj_xn(w.wv, w.wv_type, s.v, s.kvdim, s.stream_v);
                    cudaEventRecord(s.ev_k, s.stream_k);
                    cudaEventRecord(s.ev_v, s.stream_v);
                    cudaStreamWaitEvent(st, s.ev_k, 0);
                    cudaStreamWaitEvent(st, s.ev_v, 0);
                } else {
                    proj_q_gate(st);
                    proj_xn(w.wk, w.wk_type, s.k, s.kvdim, st);
                    proj_xn(w.wv, w.wv_type, s.v, s.kvdim, st);
                }
            } else {
                kernels::launch_gemm(s.xn, w.wq, w.q_has_gate ? s.qraw : s.q,
                                     1, w.q_has_gate ? s.qdim * 2 : s.qdim, H, 1.f, 0.f, gc, st);
                kernels::launch_gemm(s.xn, w.wk, s.k, 1, s.kvdim, H, 1.f, 0.f, gc, st);
                kernels::launch_gemm(s.xn, w.wv, s.v, 1, s.kvdim, H, 1.f, 0.f, gc, st);
            }
            // ---- QK-norm + RoPE + KV-append ----
            const int kvf = s.kv->kv_dtype();   // KVDtype: 0 bf16, 1 int8, 2 fp8, 3 nvfp4
            const bool kv8 = kvf != 0;   // quantized pool (scale pools present)
            void* kpool = (char*)s.kv->k_pool() + s.kv->kv_bytes(s.kv->layer_base_elems(L));
            void* vpool = (char*)s.kv->v_pool() + s.kv->kv_bytes(s.kv->layer_base_elems(L));
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + s.kv->scale_layer_base_elems(L) * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + s.kv->scale_layer_base_elems(L) * 2 : nullptr;
            const bool partial_rope = (c.rope_dim > 0 && c.rope_dim < c.head_dim);
            const bool qkgate_fuse = w.q_has_gate && partial_rope && kv8 && s.use_qkfuse && H == 2048;
            // w.wgate != nullptr means Q and the gate were projected straight into s.q / s.qgate
            // above, so there is no interleaved s.qraw to split.
            if (w.q_has_gate && !qkgate_fuse && !w.wgate)
                kernels::launch_qwen36_split_q_gate(s.qraw, s.q, s.qgate, c.n_q_heads, c.head_dim, st);
            dbg_bf16(s.q, s.qdim, 11, L);      // tag 11: Q, raw split, pre QK-norm
            dbg_bf16(s.qgate, s.qdim, 12, L);  // tag 12: attn gate proj, pre-sigmoid
            dbg_bf16(s.k, s.kvdim, 13, L);     // tag 13: K, raw, pre QK-norm

            if (!w.q_has_gate && !partial_rope && (s.use_attnin || kv8)) {
                // Qwen3-MoE frontier: fused int8 QK-norm + RoPE + KV-append (unchanged vs main)
                kernels::launch_qknorm_rope_kv_append(s.q, s.k, s.v, w.q_norm, w.k_norm, kpool, vpool,
                                                      ltab, s.d_pos, 1, c.n_q_heads, c.n_kv_heads,
                                                      c.head_dim, c.rope_theta, c.rms_eps,
                                                      s.kv->block_size(), s.kv->max_blocks_per_seq(), st,
                                                      kscale, vscale, kv8 ? 1 : 0);
            } else {
                // Qwen3.6 (gated / partial-rotary): fuse QK-norm + partial-RoPE + KV when enabled.
                if (partial_rope && kv8) {
                    if (s.use_qkfuse && H == 2048) {
                        if (qkgate_fuse) {
                            kernels::launch_qknorm_rope_kv_partial_int8_gated(s.qraw, s.q, s.qgate, s.k, s.v,
                                w.q_norm, w.k_norm, kpool, vpool, kscale, vscale, ltab, s.d_pos, 1,
                                c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                                s.kv->block_size(), s.kv->max_blocks_per_seq(), st, kvf);
                        } else {
                            kernels::launch_qknorm_rope_kv_partial_int8(s.q, s.k, s.v, w.q_norm, w.k_norm,
                                kpool, vpool, kscale, vscale, ltab, s.d_pos, 1,
                                c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                                s.kv->block_size(), s.kv->max_blocks_per_seq(), st, kvf);
                        }
                    } else {
                        if (s.use_qkfuse)
                            kernels::launch_rmsnorm_qk(s.q, s.k, w.q_norm, w.k_norm, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                        else {
                            kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads,  c.head_dim, c.rms_eps, st);
                            kernels::launch_rmsnorm(s.k, w.k_norm, s.k, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                        }
                        kernels::launch_rope_kv_append_partial_int8(s.q, s.k, s.v, kpool, vpool, kscale, vscale,
                            ltab, s.d_pos, 1, c.n_q_heads, c.n_kv_heads,
                            c.head_dim, c.rope_dim, c.rope_theta,
                            s.kv->block_size(), s.kv->max_blocks_per_seq(), st, kvf);
                    }
                } else if (partial_rope && s.use_qkfuse) {
                    kernels::launch_qknorm_rope_kv_partial(s.q, s.k, s.v, w.q_norm, w.k_norm,
                        (bf16*)kpool, (bf16*)vpool, ltab, s.d_pos, 1,
                        c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim,
                        c.rope_theta, c.rms_eps, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                } else {
                    // Muse Glimmer: QK-norm + RoPE/append are two dependent graph nodes per layer
                    // (104/step) doing tiny work. Fuse them into one; bit-identical, see
                    // launch_muse_qknorm_rope_kv. SPARKINFER_MG_QKR_FUSE=0 restores the pair.
                    static int mg_qkr = -1;
                    if (mg_qkr < 0) { const char* e = getenv("SPARKINFER_MG_QKR_FUSE"); mg_qkr = (e && e[0] == '0') ? 0 : 1; }
                    const bool mg_qkr_fuse = mg_qkr && c.muse_glimmer && s.use_qkfuse && !partial_rope && !kv8;
                    if (mg_qkr_fuse) {
                        kernels::launch_muse_qknorm_rope_kv(
                            s.q, s.k, s.v, w.q_norm, w.k_norm, (bf16*)kpool, (bf16*)vpool, ltab,
                            s.d_pos, w.swa ? s.d_pos : s.d_writepos,
                            c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                            s.kv->block_size(), c.rms_eps, /*do_rope=*/w.swa != 0, st);
                    } else if (s.use_qkfuse)
                        kernels::launch_rmsnorm_qk(s.q, s.k, w.q_norm, w.k_norm, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                    else {
                        kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads,  c.head_dim, c.rms_eps, st);
                        kernels::launch_rmsnorm(s.k, w.k_norm, s.k, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                    }
                    dbg_bf16(s.q, s.qdim, 20, L);   // tag 20: Q, post QK-norm, pre-RoPE
                    dbg_bf16(s.k, s.kvdim, 21, L);  // tag 21: K, post QK-norm, pre-RoPE
                    if (mg_qkr_fuse) {
                        // already done in one kernel above
                    } else if (c.muse_glimmer && !w.swa) {
                        // Global/NoPE layer: no rotation at all (Q/K are already QK-normed
                        // above) -- append K/V as-is. Every 4th layer per sliding_window_pattern.
                        // The bf16 append below casts the pool to bf16* unconditionally, so on an
                        // int8 cache it wrote two bytes per one-byte element: correct-looking but
                        // corrupt, and the reason Muse produced garbage at ctx >= 4096 (where the
                        // example mains switch the cache to int8). Quantise instead when kv8.
                        if (kv8)
                            kernels::launch_muse_kv_append_int8(
                                s.q, s.k, s.v, kpool, vpool, kscale, vscale, ltab, s.d_writepos, 1,
                                c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, /*rope_normal=*/false,
                                s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                        else
                        launch_kv_append((bf16*)kpool, (bf16*)vpool, s.k, s.v, ltab, s.d_writepos, 1,
                                         c.n_kv_heads, c.head_dim, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else if (partial_rope) {
                        kernels::launch_rope_kv_append_partial(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, ltab, s.d_pos, 1,
                                                               c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim,
                                                               c.rope_theta, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else if (c.muse_glimmer) {
                        // Muse Glimmer's SWA layers need "normal" (consecutive-pair,
                        // LLAMA_ROPE_TYPE_NORM) rotation, NOT the NeoX (split-half) pairing
                        // every other kernel below implements -- llama.cpp's own
                        // llama_model_rope_type() puts LLM_ARCH_MUSE_GLIMMER in the same
                        // LLAMA_ROPE_TYPE_NORM bucket as LLM_ARCH_LLAMA, while every arch this
                        // codebase was actually built for (Qwen2/3/3MoE, Gemma) is
                        // LLAMA_ROPE_TYPE_NEOX. Reusing launch_rope_kv_append here rotated the
                        // wrong pair of dimensions together for every position > 0 (position 0
                        // is a no-op rotation under either convention, which is why this hid
                        // during the earliest single-token bring-up checks): Q/K stayed
                        // well-formed but phase-wrong, so attention still produced a plausible
                        // softmax over the wrong distribution instead of visibly breaking.
                        // Unconditional (not gated on s.use_ropekv) so a SPARKINFER_ROPEKV=0
                        // override can't silently fall through to the NeoX plain-launch_rope
                        // path in the final else below.
                        // Same int8 hazard as the NoPE branch above: this writes bf16 straight
                        // into the pool, which is wrong when the cache is int8.
                        if (kv8)
                            kernels::launch_muse_kv_append_int8(
                                s.q, s.k, s.v, kpool, vpool, kscale, vscale, ltab, s.d_pos, 1,
                                c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, /*rope_normal=*/true,
                                s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                        else
                        kernels::launch_rope_kv_append_normal(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, ltab, s.d_pos, 1,
                                                              c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                                                              s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else if (s.use_ropekv) {
                        kernels::launch_rope_kv_append(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, ltab, s.d_pos, 1,
                                                       c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                                                       s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else {
                        kernels::launch_rope(s.q, s.k, s.d_pos, 1, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, st);
                        launch_kv_append((bf16*)kpool, (bf16*)vpool, s.k, s.v, ltab, s.d_writepos, 1,
                                         c.n_kv_heads, c.head_dim, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    }
                    dbg_bf16(s.q, s.qdim, 22, L);   // tag 22: Q, post RoPE-or-passthrough (SDPA input)
                    dbg_bf16(s.k, s.kvdim, 23, L);  // tag 23: K, post RoPE-or-passthrough (SDPA input)
                }
            }

            // ---- attention (Q8-emit only when output is not gated: the gate mutates attn after decode) ----
            static int attn_gq8 = -1;
            if (attn_gq8 < 0) { const char* e = getenv("SPARKINFER_ATTN_GQ8"); attn_gq8 = (e && e[0] == '0') ? 0 : 1; }
            // The H test is not a hidden-size requirement -- it is a proxy for "a gated combine has
            // been instantiated for this architecture's head_dim", and only the hd256 models ever
            // had one. fa_combine_gated_q8_kernel is HEAD_DIM-generic, so instantiating it at 128
            // lets Muse Glimmer (H=6656, head_dim=128) qualify too, folding the sigmoid gate and
            // the output quantize into the combine and deleting 2 graph nodes per layer (104/step).
            // SPARKINFER_MG_ATTN_GQ8=0 restores the split path for A/B.
            static int mg_gq8 = -1;
            if (mg_gq8 < 0) { const char* e = getenv("SPARKINFER_MG_ATTN_GQ8"); mg_gq8 = (e && e[0] == '0') ? 0 : 1; }
            const bool mg_gate_ok = c.muse_glimmer && mg_gq8 && c.head_dim == 128;
            const bool attn_gate_q8 = attn_gq8 && w.q_has_gate && s.gguf && s.use_pq && s.use_llama
                                      && (H == 2048 || H == 4096 || mg_gate_ok)
                                      && (w.wo_type == 12 || w.wo_type == 8) && (s.qdim % 32 == 0);
            const bool emit_attn_q8 = !w.q_has_gate && s.use_attnin && s.gguf && s.use_pq && s.use_llama && w.wo_type == 12;
            if (c.muse_glimmer && w.swa && s.swa_vtbl) {
                // Sliding-window layer: same dense flash-decode entry point, pointed at the
                // per-step pure sliding-window compact view (no sink, unlike the sparse-kv
                // path above) instead of the full KV. Mandatory every step at this context
                // regardless of length -- not gated on sparse_on/context-length like the
                // Qwythos/Qwen3.6 approximation.
                kernels::launch_flash_decode_split(s.q, kpool, vpool, s.swa_vtbl, s.swa_vlen,
                                                   s.attn, s.fa_m, s.fa_l, s.fa_acc, 1,
                                                   c.n_q_heads, c.n_kv_heads, c.head_dim,
                                                   s.kv->block_size(), s.swa_budget, s.swa_vsplits,
                                                   1.f / sqrtf((float)c.head_dim), st,
                                                   (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr,
                                                   s.swa_budget * s.kv->block_size(),
                                                   kscale, vscale, kv8 ? 1 : 0,
                                                   attn_gate_q8 ? s.qgate : nullptr,
                                                   mg_gate_ok ? 1 : 0);
            } else if (sparse_on && s.sparse_vtbl) {
                // GQA-8 (Qwen3.6): the same dense flash-decode entry point, pointed at the
                // per-step compact view — sink + last-window blocks, view seq_len carrying
                // the partial tail. The tuned int8-MMA kernel and fused combine run
                // unmodified; only the KV footprint changes (O(window) vs O(context)).
                // Host-side hints are view-sized constants, so the captured graph is stable:
                // max_blocks/seqlen describe the view, and sparse_vsplits keeps every MMA
                // split at >= 2 KV blocks while filling the 5090 (8 kv heads x 128 splits).
                kernels::launch_flash_decode_split(s.q, kpool, vpool, s.sparse_vtbl, s.sparse_vlen,
                                                   s.attn, s.fa_m, s.fa_l, s.fa_acc, 1,
                                                   c.n_q_heads, c.n_kv_heads, c.head_dim,
                                                   s.kv->block_size(), s.sparse_budget, s.sparse_vsplits,
                                                   1.f / sqrtf((float)c.head_dim), st,
                                                   (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr,
                                                   s.sparse_budget * s.kv->block_size(),
                                                   kscale, vscale, kvf,
                                                   attn_gate_q8 ? s.qgate : nullptr);
            } else if (sparse_on) {
                kernels::launch_fa_kv_window_select(s.d_seqlen, s.sparse_sel, c.n_kv_heads,
                    s.kv->block_size(), s.sparse_budget, s.sparse_window, st);
                kernels::launch_flash_decode_split_sparse(s.q, kpool, vpool, ltab, s.d_seqlen,
                    s.sparse_sel, s.fa_m, s.fa_l, s.fa_acc, c.n_q_heads, c.n_kv_heads, c.head_dim,
                    s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits, s.sparse_budget,
                    1.f / sqrtf((float)c.head_dim), kscale, vscale, st);
                kernels::launch_fa_combine_hd256(s.fa_m, s.fa_l, s.fa_acc, s.attn, c.n_q_heads,
                    s.n_splits, (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr, st,
                    attn_gate_q8 ? s.qgate : nullptr);
            } else {
            kernels::launch_flash_decode_split(s.q, kpool, vpool, ltab, s.d_seqlen, s.attn,
                                               s.fa_m, s.fa_l, s.fa_acc, 1, c.n_q_heads, c.n_kv_heads, c.head_dim,
                                               s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
                                               1.f / sqrtf((float)c.head_dim), st,
                                               (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr, seqlen,
                                               kscale, vscale, kvf,
                                               attn_gate_q8 ? s.qgate : nullptr,
                                               mg_gate_ok ? 1 : 0);
            }
            dbg_bf16(s.attn, s.qdim, 30, L);   // tag 30: SDPA output, pre-gate
            if (w.q_has_gate && !attn_gate_q8) {
                kernels::launch_qwen36_mul_sigmoid(s.attn, s.qgate, s.qdim, st);
            }
            dbg_bf16(s.attn, s.qdim, 31, L);   // tag 31: SDPA output, post sigmoid-gate

            // ---- O projection (int8 mmvq path) ----
            if (pf_win & 1) pf_join();   // window 1 lands here: w.wo is read next
            if (s.gguf && s.use_pq && w.wo_type == 12) {
                if (s.use_llama) {
                    if (!emit_attn_q8 && !attn_gate_q8) kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, s.qdim, st);
                    kernels::launch_mmvq_q4k(s.aq81, w.wo, s.ao, H, s.qdim, st);
                } else {
                    kernels::launch_quantize_q8_1(s.attn, s.aq8, s.aq8_d, s.aq8_s, s.qdim, st);
                    kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, w.wo, s.ao, H, s.qdim, st);
                }
            }
            else if (s.gguf && s.use_pq && s.use_llama && w.wo_type == 8) {
                kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, s.qdim, st);
                kernels::launch_mmvq_q80(s.aq81, w.wo, s.ao, H, s.qdim, st);
            }
            // o_proj does not go through proj_from, so it needs the dp4a arm spelled out here.
            // The verify drives w.wo through its own proj(), which does take that arm -- leaving
            // this one on the float GEMV made decode and verify disagree on 16 layers' worth of
            // attention output, which showed up as LOSSLESS=0 rather than as a wrong number.
            else if (s.gguf && w.wo_type == kernels::SI_QTYPE_NVFP4 &&
                     kernels::qwen38_nvfp4_dp4a_proj() &&
                     (kernels::launch_gemv_nvfp4_quant_x(s.attn, s.nv_pq_b, s.nv_ps_b, 1, s.qdim, st),
                      kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_pq_b, s.nv_ps_b, w.wo, s.ao,
                                                           1, H, s.qdim, st))) {}
            else if (s.gguf && w.wo_type) kernels::launch_gemv_q(s.attn, w.wo, w.wo_type, s.ao, H, s.qdim, st);
            else if (s.gguf)         kernels::launch_gemv(s.attn, w.wo, s.ao, H, s.qdim, st);
            else                     kernels::launch_gemm(s.attn, w.wo, s.ao, 1, H, s.qdim, 1.f, 0.f, gc, st);
            dbg_bf16(s.ao, H, 40, L);   // tag 40: attn_o_proj output (post wo)
        }

        if (c.muse_glimmer) {
            // Sandwich norm: h = x + RMSNorm(ao) * post_attn_norm (norm the attention
            // output alone, not the sum -- see launch_norm_then_add), then hn = RMSNorm(h,
            // ffn_norm) is a genuine separate pre-FFN norm, not post_attn_norm doing double
            // duty like every other architecture here.
            //
            // The sandwich (post_attn_norm/post_ffn_norm) RMSNorm uses its OWN eps (1e-8,
            // upstream's `post_norm_eps` in muse-glimmer.cpp), distinct from the model's
            // normal rms_eps (1e-5, used for attn_norm/ffn_norm/q_norm/k_norm) -- reusing
            // c.rms_eps here silently gives wrong post-attn/post-ffn norms.
            // Both halves of the sandwich tail are single-CTA kernels over 6656 elements, so each
            // costs ~3.2 us of launch/reduction latency for 13 KB of traffic. Fuse them.
            // Window 2: the post-attn sandwich tail is a single CTA on a 170-SM device for ~3.4 us.
            // Cover it with the FFN gate/up matrices, which are streamed immediately after it.
            if (pf_win & 2) pf_fork(w.gate_q, w.up_q);
            if (muse_fuse_tail())
                // hn is the FFN's input; let the tail hand it over already quantized. Nothing
                // between here and the dense FFN touches aq81 on this architecture (n_shared=0,
                // no router), so the buffer still holds Q8_1(hn) when gate/up read it.
                muse_hn_q8 = kernels::launch_muse_sandwich_tail(s.x, s.ao, w.post_attn_norm,
                                                   w.ffn_norm, s.h, s.hn,
                                                   fnq ? s.aq81 : nullptr, 1, H, 1e-8f,
                                                   c.rms_eps, st);
            else {
            kernels::launch_norm_then_add(s.x, s.ao, w.post_attn_norm, s.h, 1, H, 1e-8f, st);
            dbg_bf16(s.h, H, 50, L);   // tag 50: h = x + sandwich_norm(ao)  (post-attn residual)
            kernels::launch_rmsnorm(s.h, w.ffn_norm, s.hn, 1, H, c.rms_eps, st);
            }
            dbg_bf16(s.hn, H, 51, L);  // tag 51: hn = pre-FFN norm(h)
        } else if (fnq) {
            // fused: h = x + ao ; hn = RMSNorm(h, post_attn_norm). When fnq, also emit
            // Q8_1(hn) into aq81 so the MoE gate/up mmvq skips its own quantize node (the
            // router below reads bf16 hn).
            kernels::launch_add_rmsnorm2_q8(s.x, s.ao, w.post_attn_norm, s.h, s.hn, s.aq81, H, c.rms_eps, st);
            dbg_bf16(s.h, H, 50, L);
            dbg_bf16(s.hn, H, 51, L);
        } else {
            kernels::launch_add_rmsnorm2(s.x, s.ao, w.post_attn_norm, s.h, s.hn, 1, H, c.rms_eps, st);
            dbg_bf16(s.h, H, 50, L);
            dbg_bf16(s.hn, H, 51, L);
        }
        dbg_xn_snapshot(s.hn, c.n_layers + 1 + L);   // slots n_layers+1.. : post-attn hn

        const bool qmoe = w.shared_gate_q && w.shared_up_q && w.shared_down_q
                       && w.shared_gate_qtype == 8 && c.hidden == 2048 && c.moe_ffn == 512;
        const bool shexp_pipelined = (c.n_shared > 0) && s.gguf && s.use_shexp_pipe;
        if (shexp_pipelined) {
            cudaEventRecord(s.ev_pipe_fork, st);
            cudaStreamWaitEvent(s.stream_k, s.ev_pipe_fork, 0);
            cudaStreamWaitEvent(s.stream_v, s.ev_pipe_fork, 0);
            if (w.shared_gate_inp) {
                if (s.use_pq && w.shared_gate_inp_type == 12) {
                    if (s.use_llama) {
                        if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, s.stream_k);
                        if (H == 2048)
                            kernels::launch_mmvq_q4k_sigmoid(s.aq81, w.shared_gate_inp, s.d_shared_w, H, s.stream_k);
                        else
                            kernels::launch_mmvq_q4k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                    } else {
                        kernels::launch_quantize_q8_1(s.hn, s.aq8, s.aq8_d, s.aq8_s, H, s.stream_k);
                        kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s,
                                                        w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                    }
                } else if (s.use_pq && s.use_llama && s.use_q6mmvq && w.shared_gate_inp_type == 14) {
                    if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, s.stream_k);
                    kernels::launch_mmvq_q6k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                } else if (w.shared_gate_inp_type) {
                    kernels::launch_gemv_q(s.hn, w.shared_gate_inp, w.shared_gate_inp_type,
                                           s.shared_gate_tmp, 1, H, s.stream_k);
                } else {
                    // Fused GEMV + sigmoid for the shared-expert gate scalar:
                    // writes fp32 sigmoid(gate) directly, eliminating the separate
                    // 1-thread sigmoid_scalar_kernel launch. SPARKINFER_GEMV_SIGMOID=0
                    // restores the split path for A/B.
                    static int gemv_sigmoid = -1;
                    if (gemv_sigmoid < 0) { const char* e = getenv("SPARKINFER_GEMV_SIGMOID");
                        gemv_sigmoid = (e && e[0] == '1') ? 1 : 0; }   // default off: fused dot != split-k GEMV
                    if (gemv_sigmoid) {
                        kernels::launch_gemv_sigmoid(s.hn, w.shared_gate_inp, s.shared_gate_tmp, s.d_shared_w, H, s.stream_k);
                    } else {
                        kernels::launch_gemv(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                        kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, s.stream_k);
                    }
                }
                // The dense gate branch above already applies sigmoid (either fused into
                // launch_gemv_sigmoid or as its split follow-up). Only quantized projections
                // that leave a raw scalar in shared_gate_tmp need this common epilogue.
                if (w.shared_gate_inp && w.shared_gate_inp_type != 0 &&
                    !(s.use_pq && s.use_llama && w.shared_gate_inp_type == 12 && H == 2048))
                    kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, s.stream_k);
            }
            if (qmoe) {
                // Pipelined shared overlaps stream_k with MoE on st — accum into routed here
                // races MoE (shared finishes first, MoE overwrites routed). Always write s.shared;
                // fold happens after both complete. SPARKINFER_SHEXP_ACCUM=1 only applies on the
                // non-pipelined path where MoE has already landed in routed.
                kernels::launch_shared_expert_q8_mmvq(
                    s.hn, fnq ? s.aq81 : nullptr,
                    w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                    w.shared_gate_inp ? s.d_shared_w : nullptr,
                    s.shared, s.sx_h, s.sx_q8, H, c.moe_ffn, s.stream_k, false);
            } else {
                kernels::launch_gemv(s.hn, w.shared_gate, s.sh_gate, c.moe_ffn, H, s.stream_k);
                kernels::launch_gemv(s.hn, w.shared_up,   s.sh_up,   c.moe_ffn, H, s.stream_v);
                cudaEventRecord(s.ev_sx_gate, s.stream_v);
                cudaStreamWaitEvent(s.stream_k, s.ev_sx_gate, 0);
                kernels::launch_qwen36_shared_swiglu(s.sh_gate, s.sh_up, s.d_shared_w,
                                                     s.sh_h, c.moe_ffn, s.stream_k);
                kernels::launch_gemv(s.sh_h, w.shared_down, s.shared, H, c.moe_ffn, s.stream_k);
            }
            cudaEventRecord(s.ev_sx_done, s.stream_k);
        }

        if (c.dense_ffn) {
            // Qwen3.5 dense SwiGLU: keep gate/up/down quantized and run the same MMVQ
            // expert-FFN path as MoE decode — bf16 dequant+GEMV diverged ~40pp vs llama.cpp.
            //
            // input_q8 must be a valid Q8_1 quantization of s.hn (the FFN's actual input) or
            // null (letting the kernel quantize s.hn itself). `fnq` only guarantees that for
            // architectures whose post-attention step runs launch_add_rmsnorm2_q8, which emits
            // Q8_1(hn) as a side effect of computing hn. Muse Glimmer's sandwich norm does not:
            // its post-attn step is launch_norm_then_add (writes s.h, no Q8 emission) followed
            // by a plain launch_rmsnorm into s.hn (see the c.muse_glimmer branch above) -- s.aq81
            // at this point still holds Q8_1(xn) from this same layer's QKV-input quantize
            // (prepare_xn_quant, earlier in this iteration), not Q8_1(hn). Passing that stale
            // buffer here fed the gate/up MMVQ kernel the wrong activation vector entirely:
            // garbage FFN output that still looked like a confident (but wrong) distribution
            // downstream, rather than crashing or NaN-ing. Force nullptr for muse_glimmer so
            // launch_moe_expert_ffn_q4k quantizes s.hn fresh instead of trusting the stale cache.
            // Checkpoint-native decode FFN -- the default since 2026-08-21
            // (SPARKINFER_QWEN38_DECODE_NVFP4=0 opts out). gate_q/up_q/down_q
            // are a Q4_K REQUANTIZATION of the NVFP4 the checkpoint ships -- 8.25% mean relative
            // weight error, corr 0.9968 -- so this path exists to answer whether decoding the
            // actual checkpoint numerics changes anything measurable. It is dense-only (top_k==1,
            // one expert): the 256-expert MoE routes per token and has no single payload to read.
            //
            // Deliberately unfused: three GEMVs and an elementwise SwiGLU, against the Q4_K path's
            // single fused kernel. That is the slow-but-obviously-correct shape for an accuracy
            // experiment; if the accuracy win is real, the fusion work follows, and NVFP4's
            // decoded magnitudes are exact int8 so a dp4a path is reachable.
            // Ternary SwiGLU, read in the stored blocks. Same three-GEMV-plus-elementwise shape
            // as the NVFP4 arm below and for the same reason: obviously correct first, and the
            // fused kernel it replaces cannot drive type 143 at all.
            //
            // Two rotations, not one. Gate and up read the post-attention norm at the residual
            // width; down reads SwiGLU's output at the FFN width, which is a different sign
            // vector entirely. The second rotation is in place -- launch_hadamard_rotate_bf16
            // permits aliasing, and each CTA owns its whole 1024-span.
            if (w.gate_qtype == kPtq1GgmlType && w.up_qtype == kPtq1GgmlType &&
                w.down_qtype == kPtq1GgmlType && s.bonsai_ffn_h && c.top_k == 1) {
                if (dec_shadow) {
                    // Gate and up share one int8 copy of the rotated hn, made by the rotation.
                    const int hq = kernels::launch_ptq1_rotate_quant(
                        s.hn, s.bonsai_rot_hn, s.bonsai_sign_h, (int)H, (int)s.bonsai_block, st);
                    kernels::launch_gemv_ptq1_q(hq, s.bonsai_rot_hn, w.gate_q, s.bonsai_ffn_gate,
                                                c.moe_ffn, H, st);
                    kernels::launch_gemv_ptq1_q(hq, s.bonsai_rot_hn, w.up_q, s.bonsai_ffn_up,
                                                c.moe_ffn, H, st);
                } else {
                    kernels::launch_hadamard_rotate_bf16(s.hn, s.bonsai_rot_hn, s.bonsai_sign_h,
                                                         H, (int)H, (int)s.bonsai_block, st);
                    kernels::launch_gemv_ptq1(s.bonsai_rot_hn, w.gate_q, s.bonsai_ffn_gate,
                                              c.moe_ffn, H, st);
                    kernels::launch_gemv_ptq1(s.bonsai_rot_hn, w.up_q, s.bonsai_ffn_up,
                                              c.moe_ffn, H, st);
                }
                kernels::launch_prefill_swiglu(s.bonsai_ffn_gate, s.bonsai_ffn_up,
                                               s.bonsai_ffn_h, c.moe_ffn, st);
                if (dec_shadow) {
                    const int fq = kernels::launch_ptq1_rotate_quant(
                        s.bonsai_ffn_h, s.bonsai_ffn_h, s.bonsai_sign_ffn, (int)c.moe_ffn,
                        (int)s.bonsai_block, st);
                    kernels::launch_gemv_ptq1_q(fq, s.bonsai_ffn_h, w.down_q, s.routed, H,
                                                c.moe_ffn, st);
                } else {
                    kernels::launch_hadamard_rotate_bf16(s.bonsai_ffn_h, s.bonsai_ffn_h,
                                                         s.bonsai_sign_ffn, c.moe_ffn,
                                                         (int)c.moe_ffn, (int)s.bonsai_block, st);
                    kernels::launch_gemv_ptq1(s.bonsai_ffn_h, w.down_q, s.routed, H, c.moe_ffn,
                                              st);
                }
            } else
            if (w.gate_nv && w.up_nv && w.down_nv && c.top_k == 1) {
                // dp4a NVFP4 (SPARKINFER_QWEN38_NVFP4_DP4A=0 restores the float GEMVs). The
                // weights stay exactly what the checkpoint ships -- NVFP4 magnitudes are integers,
                // so the group dot is an exact integer reduction -- and only the activation is
                // quantized, at one scale per 16 values.
                //
                // The verify in qwen35_prefill.cpp switches on the SAME flag and calls the SAME
                // kernel at N rows. Both sides must move together: losslessness here is defined as
                // the speculative output matching the AR output of this build, so a decode on
                // dp4a against a verify on the float path would disagree and report LOSSLESS=0.
                if (kernels::qwen38_nvfp4_dp4a()) {
                    kernels::launch_gemv_nvfp4_quant_x(s.hn, s.nv_xq, s.nv_xs, 1, H, st);
                    // One grid for gate and up (see launch_gemv_nvfp4_rows_dp4a2): same shape,
                    // same activation, and per-row bit-identical to the two singles below.
                    if (!kernels::launch_gemv_nvfp4_rows_dp4a2(s.nv_xq, s.nv_xs, w.gate_nv,
                                                               w.up_nv, s.nv_gate, s.nv_up,
                                                               1, c.moe_ffn, H, st)) {
                    kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_xq, s.nv_xs, w.gate_nv, s.nv_gate,
                                                         1, c.moe_ffn, H, st);
                    kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_xq, s.nv_xs, w.up_nv, s.nv_up,
                                                         1, c.moe_ffn, H, st);
                    }
                    if (!kernels::launch_prefill_swiglu_nvfp4(s.nv_gate, s.nv_up, s.nv_h,
                                                             s.nv_xq, s.nv_xs, c.moe_ffn, st)) {
                    kernels::launch_prefill_swiglu(s.nv_gate, s.nv_up, s.nv_h, c.moe_ffn, st);
                    kernels::launch_gemv_nvfp4_quant_x(s.nv_h, s.nv_xq, s.nv_xs, 1, c.moe_ffn, st);
                    }
                    kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_xq, s.nv_xs, w.down_nv, s.routed,
                                                         1, H, c.moe_ffn, st);
                } else {
                kernels::launch_gemv_nvfp4(s.hn, w.gate_nv, s.nv_gate, c.moe_ffn, H, st);
                kernels::launch_gemv_nvfp4(s.hn, w.up_nv, s.nv_up, c.moe_ffn, H, st);
                kernels::launch_prefill_swiglu(s.nv_gate, s.nv_up, s.nv_h, c.moe_ffn, st);
                kernels::launch_gemv_nvfp4(s.nv_h, w.down_nv, s.routed, H, c.moe_ffn, st);
                }
            } else
            kernels::launch_moe_expert_ffn_q4k(s.hn, w.gate_q, w.up_q, w.down_q,
                                               w.gate_qtype, w.up_qtype, w.down_qtype,
                                               s.mf_ids, s.mf_weights, s.routed, s.mf_h, s.mf_out,
                                               1, c.top_k, H, c.moe_ffn,
                                               (muse_hn_q8 || (fnq && !c.muse_glimmer))
                                                   ? s.aq81 : nullptr, st);
        } else if (w.gate_q) {   // GGUF fused: route, then dequant-on-read only the top_k experts
            // The per-expert token counts only feed the batched-dispatch sort; the single-token
            // decode expert FFN reads ids/weights directly and never touches them. Zeroing that
            // buffer is a per-layer memset node in the replayed decode graph whose fixed cost far
            // outweighs the handful of atomics that fill it, so skip the count on this path.
            // SPARKINFER_MOE_COUNTS=1 restores the memset + on-device counting.
            static int moe_counts = -1;
            if (moe_counts < 0) { const char* mc = getenv("SPARKINFER_MOE_COUNTS"); moe_counts = (mc && mc[0] == '1') ? 1 : 0; }
            const bool rfuse = s.use_router_fused && !moe_counts && c.n_experts == 256 && (c.hidden % 8) == 0;
            if (rfuse) {
                // one kernel: router GEMV -> logits scratch, then in-kernel bitonic top-8 (last block)
                kernels::launch_router_fused(s.hn, w.router_w, s.mf_logits, s.mf_rc,
                                             s.mf_ids, s.mf_weights, c.n_experts, c.hidden, c.top_k, 1, st);
            } else {
                kernels::launch_gemv_f32(s.hn, w.router_w, s.mf_logits, c.n_experts, c.hidden, st);  // router_w native [E,H]
                if (moe_counts) cu(cudaMemsetAsync(s.mf_counts, 0, c.n_experts * sizeof(int), st), "mf counts");
                kernels::launch_moe_router(s.mf_logits, s.mf_ids, s.mf_weights,
                                           moe_counts ? s.mf_counts : nullptr,
                                           1, c.n_experts, c.top_k, 1, st);
            }
            kernels::launch_moe_expert_ffn_q4k(s.hn, w.gate_q, w.up_q, w.down_q,
                                               w.gate_qtype, w.up_qtype, w.down_qtype,
                                               s.mf_ids, s.mf_weights, s.routed, s.mf_h, s.mf_out,
                                               1, c.top_k, c.hidden, c.moe_ffn,
                                               fnq ? s.aq81 : nullptr, st);
        } else {
            s.engine->set_layer_weights(L, {w.router_w, w.gate, w.up, w.down});
            s.engine->forward(s.hn, s.routed, 1, L, st);
        }
        const void* shared_to_fold = nullptr;
        if (c.n_shared > 0) {
            const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
            if (shexp_pipelined) {
                cudaStreamWaitEvent(st, s.ev_sx_done, 0);
                // (residual_add folded into add_rmsnorm3 below — #279)
                if (s.use_addnorm3) {
                    if (fnq)
                        kernels::launch_add_rmsnorm3_q8(s.h, s.routed, s.shared, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
                    else
                        kernels::launch_add_rmsnorm3(s.h, s.routed, s.shared, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
                } else {
                    launch_residual_add(s.routed, s.shared, s.routed, H, st);
                    if (fnq)
                        kernels::launch_add_rmsnorm2_q8(s.h, s.routed, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
                    else
                        kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
                }
                dflash_maybe_capture_layer(L);
                continue;
            }
            if (w.shared_gate_inp) {
                if (s.gguf) {
                    if (s.use_pq && w.shared_gate_inp_type == 12) {
                        if (s.use_llama) {
                            if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, st);
                            if (H == 2048)
                                kernels::launch_mmvq_q4k_sigmoid(s.aq81, w.shared_gate_inp, s.d_shared_w, H, st);
                            else
                                kernels::launch_mmvq_q4k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                        } else {
                            kernels::launch_quantize_q8_1(s.hn, s.aq8, s.aq8_d, s.aq8_s, H, st);
                            kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s,
                                                            w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                        }
                    } else if (s.use_pq && s.use_llama && s.use_q6mmvq && w.shared_gate_inp_type == 14) {
                        if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, st);
                        kernels::launch_mmvq_q6k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                    } else if (w.shared_gate_inp_type) {
                        kernels::launch_gemv_q(s.hn, w.shared_gate_inp, w.shared_gate_inp_type, s.shared_gate_tmp, 1, H, st);
                    } else {
                        static int gs2 = -1;
                        if (gs2 < 0) { const char* e = getenv("SPARKINFER_GEMV_SIGMOID");
                            gs2 = (e && e[0] == '1') ? 1 : 0; }
                        if (gs2) {
                            kernels::launch_gemv_sigmoid(s.hn, w.shared_gate_inp, s.shared_gate_tmp, s.d_shared_w, H, st);
                        } else {
                            kernels::launch_gemv(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                            kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, st);
                        }
                    }
                } else {
                    kernels::launch_gemm(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, 1, H, 1.f, 0.f, gc, st);
                    kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, st);
                }
            }
            if (s.gguf) {
                if (qmoe) {
                    // MoE already wrote routed; safe to accum shared down into it (opt-in).
                    static int shexp_accum = -1;
                    if (shexp_accum < 0) { const char* e = getenv("SPARKINFER_SHEXP_ACCUM");
                        shexp_accum = (e && e[0] == '1') ? 1 : 0; }
                    const bool sx_accum = shexp_accum != 0;
                    kernels::launch_shared_expert_q8_mmvq(
                        s.hn, fnq ? s.aq81 : nullptr,
                        w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                        w.shared_gate_inp ? s.d_shared_w : nullptr,
                        sx_accum ? s.routed : s.shared, s.mf_h, s.aq81, H, c.moe_ffn, st,
                        sx_accum);
                    if (sx_accum) {
                        if (fnq)
                            kernels::launch_add_rmsnorm2_q8(s.h, s.routed, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
                        else
                            kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
                        dflash_maybe_capture_layer(L);
                        continue;
                    }
                } else {
                    kernels::launch_gemv(s.hn, w.shared_gate, s.sh_gate, c.moe_ffn, H, st);
                    kernels::launch_gemv(s.hn, w.shared_up,   s.sh_up,   c.moe_ffn, H, st);
                    kernels::launch_qwen36_shared_swiglu(s.sh_gate, s.sh_up, s.d_shared_w, s.sh_h, c.moe_ffn, st);
                    kernels::launch_gemv(s.sh_h, w.shared_down, s.shared, H, c.moe_ffn, st);
                }
            } else {
                // set_weights path: shared weights are [hidden,ffn]/[ffn,hidden] dense.
                kernels::launch_moe_expert_ffn(s.hn, w.shared_gate, w.shared_up, w.shared_down,
                                               s.d_shared_ids, s.d_shared_w, s.shared,
                                               1, 1, 1, H, c.moe_ffn, st);
            }
            if (s.use_addnorm3) shared_to_fold = s.shared;
            else launch_residual_add(s.routed, s.shared, s.routed, H, st);
        }
        const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        dbg_bf16(s.routed, H, 60, L);   // tag 60: routed = dense FFN(hn) output, pre sandwich-norm
        if (c.muse_glimmer) {
            // Sandwich norm: x = h + RMSNorm(routed) * post_ffn_norm (norm the FFN output
            // alone, not the sum -- mirrors the post-attention step above). No shared
            // expert on this architecture (dense_ffn, n_shared=0), so shared_to_fold is
            // always null here. xn = RMSNorm(x, nextnorm) is the next layer's ordinary
            // pre-attn norm (or final_norm on the last layer), unaffected by the sandwich.
            // Same 1e-8 post_norm_eps as the post-attn sandwich norm above -- see that comment.
            // Window 2 lands (the whole FFN has run since it was forked). Window 3: cover the
            // post-FFN sandwich tail with the next layer's attention projections.
            {
                if (pf_win & 2) pf_join();
                if (pf_win & 4) {
                    const Qwen35LayerWeights* nx = (L + 1 < c.n_layers) ? &s.w.layers[L + 1] : nullptr;
                    if (nx) pf_fork(nx->wq, nx->wgate ? nx->wgate : nx->wk);
                    else    pf_fork(s.w.lm_head, nullptr);
                }
            }
            if (muse_fuse_tail())
                // xn is the NEXT layer's Q/K/V input; emitting Q8_1(xn) here is what finally
                // lets muse_glimmer take the xn_q8_ready path other architectures already get
                // from add_rmsnorm2_q8 / add_rmsnorm3_q8.
                muse_xn_q8 = kernels::launch_muse_sandwich_tail(s.h, s.routed, w.post_ffn_norm,
                                                   nextnorm, s.x, s.xn,
                                                   fnq ? s.aq81 : nullptr, 1, H, 1e-8f,
                                                   c.rms_eps, st);
            else {
            kernels::launch_norm_then_add(s.h, s.routed, w.post_ffn_norm, s.x, 1, H, 1e-8f, st);
            dbg_bf16(s.x, H, 70, L);   // tag 70: x = h + sandwich_norm(routed)  (layer output)
            kernels::launch_rmsnorm(s.x, nextnorm, s.xn, 1, H, c.rms_eps, st);
            }
        } else if (shared_to_fold) {
            if (fnq)
                kernels::launch_add_rmsnorm3_q8(s.h, s.routed, shared_to_fold, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
            else
                kernels::launch_add_rmsnorm3(s.h, s.routed, shared_to_fold, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
        } else if (fnq)
            kernels::launch_add_rmsnorm2_q8(s.h, s.routed, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
        else
            kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
        dflash_maybe_capture_layer(L);
    }
    pf_join();   // last layer's prefetch: must be joined before the capture ends or the graph is malformed
    // xn now holds RMSNorm(x_final, final_norm)
    dbg_bf16(s.xn, H, 80, -2);   // tag 80: final-norm output (lm_head input)
    dbg_xn_snapshot(s.xn, c.n_layers);   // extra slot: final-norm output, for lm_head cross-check
    if (!sample) {
        if (capturing_graph &&
            finish_capture(st, &s.cu_prefill_graph, &s.cu_prefill_exec, "prefill graph capture")) {
            s.graph_prefill_ready = true;
            s.graph_prefill_attn_mode = attn_graph_mode;
            cu(cudaGraphLaunch(s.cu_prefill_exec, st), "prefill graph launch (first)");
        }
        cu(cudaStreamSynchronize(st), "prefill sync");
        return token_id;
    }
    // Third instance of the same stale-Q8_1-cache pattern as prepare_xn_quant's xn_q8_ready
    // (L>0) and the dense_ffn gate/up input above: `fnq` only promises aq81==Q8_1(xn) here
    // because the fnq path's final-layer tail is launch_add_rmsnorm2_q8/add_rmsnorm3_q8,
    // which writes xn AND emits Q8_1(xn) into aq81 as a side effect. Muse Glimmer's final-
    // layer tail is the c.muse_glimmer sandwich-norm branch above (launch_norm_then_add then
    // a plain launch_rmsnorm into s.xn) -- no Q8 side channel. `fnq` itself doesn't check
    // c.muse_glimmer (it's just s.gguf/use_fnq/use_pq/use_llama), so with fnq true this
    // silently fed the LM head a stale aq81 left over from the last layer's fresh
    // prepare_xn_quant(xn) quantize (a *different*, pre-final-norm activation vector) --
    // wrong logits on every single decode step. Force a fresh quantize for muse_glimmer.
    if (s.bonsai_dec_head) {
        // The decode shadow's ternary head: rotate xn into the weights' basis, as the native
        // branch below does, and read 0.21875 bytes/weight instead of the folded head's Q4_K.
        const int hq = kernels::launch_ptq1_rotate_quant(s.xn, s.bonsai_rot, s.bonsai_sign_h,
                                                         (int)H, (int)s.bonsai_block, st);
        kernels::launch_gemv_ptq1_q_f32(hq, s.bonsai_rot, s.bonsai_dec_head, s.logits, c.vocab,
                                        H, st);
    }
    else if (s.gguf && s.use_pq && s.use_llama && s.w.lm_head_type == 12) {
        if (!fnq || c.muse_glimmer) kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
        kernels::launch_mmvq_q4k_f32(s.aq81, s.w.lm_head, s.logits, c.vocab, H, st);
    }
    else if (s.gguf && s.use_q6mmvq && s.w.lm_head_type == 14) {   // int8 Q6_K dp4a LM head (1 warp/row)
        if (!fnq || c.muse_glimmer) kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);  // else aq81 = Q8_1(xn) from final norm
        kernels::launch_gemv_q6k_dp4a_f32(s.aq81, s.w.lm_head, s.logits, c.vocab, H, st);
    }
    else if (s.gguf && s.w.lm_head_type == kPtq1GgmlType && s.bonsai_rot &&
             s.bonsai_sign_dev.count(H) != 0) {
        // Native ternary: the weights are still in the basis they were quantized in, so the
        // activation goes there first rather than the rotation being folded into the weights.
        const auto it = s.bonsai_sign_dev.find(H);
        kernels::launch_hadamard_rotate_bf16(s.xn, s.bonsai_rot,
                                             static_cast<const signed char*>(it->second),
                                             H, (int)H, (int)s.bonsai_block, st);
        kernels::launch_gemv_ptq1_f32(s.bonsai_rot, s.w.lm_head, s.logits, c.vocab, H, st);
    }
    else if (s.gguf && s.w.lm_head_type) kernels::launch_gemv_q_f32(s.xn, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else if (s.gguf)                kernels::launch_gemv_f32(s.xn, s.w.lm_head, s.logits, c.vocab, H, st);  // lm_head native [vocab,H]
    else        kernels::launch_linear_f32(s.xn, s.w.lm_head, s.logits, 1, c.vocab, H, st);
    dbg_f32(s.logits, c.vocab, 90, -2);   // tag 90: raw logits (pre logit_scale/softcap)
    if (c.muse_glimmer && c.final_logit_softcapping > 0.f)
        kernels::launch_logit_softcap(s.logits, 1, c.vocab, c.logit_scale, c.final_logit_softcapping, st);
    dbg_f32(s.logits, c.vocab, 91, -2);   // tag 91: final logits (post logit_scale/softcap)
    // Always launched, never host-gated: logit_bias has NO inertness proof at temperature<=0 (an
    // arbitrary per-vocab additive bias CAN change the greedy-argmax winner on its own) -- see
    // qwen35.h's forward_token doc comment. Reads whatever is CURRENTLY in s.logit_bias, set once
    // per request by set_logit_bias (not refreshed here every decode step, unlike the scalar
    // params below) -- correct because activate_session() swaps s.logit_bias to the request's own
    // session buffer before any of this request's forward_token() calls run.
    kernels::launch_logit_bias(s.logits, s.logit_bias, c.vocab, st);
    // Always launched, never host-gated: presence/frequency penalty has NO inertness proof at
    // presence_penalty==0 && frequency_penalty==0 the way top_k/top_p do at temperature<=0 -- see
    // qwen35.h's forward_token doc comment. Applied BEFORE launch_topk_topp_mask's sort so the
    // penalized distribution flows through top_k/top_p truncation, temperature sampling, AND
    // logprobs reporting -- matches real-world OpenAI/vLLM behavior of reporting logprobs against
    // what was actually sampled from.
    kernels::launch_presence_frequency_penalty(s.logits, s.penalty_counts, c.vocab,
                                               s.d_sample_presence_penalty, s.d_sample_frequency_penalty, st);
    // Always launched, never host-gated on top_k/top_p/temperature: this call site is inside a
    // CUDA graph whose node topology is frozen at capture time and may be replayed for a LATER,
    // separate request (e.g. via use_prefix_session's shared seq_id=0) with different values. Both
    // kernels read their params from device memory refreshed above and no-op (or are provably
    // inert -- see qwen35.h's forward_token doc comment) when disabled/at temperature<=0.
    kernels::launch_topk_topp_mask(s.logits, c.vocab, s.d_vocab_iota, s.d_sorted_logits, s.d_sorted_idx,
                                   s.d_topk_exp, s.d_topk_cumsum, s.d_sort_temp, s.sort_temp_bytes,
                                   s.d_scan_temp, s.scan_temp_bytes,
                                   s.d_sample_top_k, s.d_sample_top_p, s.d_rank_by_id, st);
    kernels::launch_temperature_sample(s.logits, 1, c.vocab, s.d_sample_temp, s.d_sample_seed,
                                       s.d_sample_step, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    // Always launched, never host-gated on logprobs: same CUDA-graph-captured-region rationale as
    // launch_topk_topp_mask/launch_temperature_sample above -- this call site may replay for a
    // LATER, different request whose logprobs setting differs. Negligible cost (single thread).
    kernels::launch_extract_chosen_logit(s.d_out_id, s.d_rank_by_id, s.d_sorted_logits,
                                         s.d_chosen_logit, st);
    // Always launched, unconditionally: records this step's sampled token into the CURRENT
    // session's running penalty count so the NEXT decode step's penalty application sees it.
    kernels::launch_increment_penalty_count(s.penalty_counts, s.d_out_id, st);
    if (s.bench_feedback_graph) kernels::launch_decode_feedback(s.d_scalars, s.d_out_id, st);

    if (capturing_graph && dflash_cap) {
        if (finish_capture(st, &s.cu_dflash_graph, &s.cu_dflash_exec, "dflash graph capture")) {
            s.dflash_graph_ready = true;
            s.dflash_graph_attn_mode = attn_graph_mode;
            s.dflash_graph_sparse = sparse_on;
            {
                static int dbg = -1;
                if (dbg < 0) { const char* e = getenv("SPARKINFER_GRAPH_DEBUG"); dbg = (e && e[0]=='1') ? 1 : 0; }
                if (dbg) {
                    const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
                    fprintf(stderr, "[dflash-graph] capture pos=%d seqlen=%d n_splits=%d attn_mode=%d mma_chunk=%d sparse=%d\n",
                            position, seqlen, s.n_splits, attn_graph_mode, mma_chunk, sparse_on ? 1 : 0);
                }
            }
            cu(cudaGraphLaunch(s.cu_dflash_exec, st), "dflash graph launch (first)");
        }
    } else if (capturing_graph) {
        if (finish_capture(st, &s.cu_graph, &s.cu_exec, "decode graph capture")) {
            s.graph_ready = true;
            s.graph_attn_mode = attn_graph_mode;
            s.graph_sparse = sparse_on;
            s.graph_state_b16 = s.active_lin_state_b16;
            static int graph_dbg = -1;
            if (graph_dbg < 0) {
                const char* e = getenv("SPARKINFER_GRAPH_DEBUG");
                graph_dbg = (e && e[0] == '1') ? 1 : 0;
            }
            if (graph_dbg) {
                const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
                fprintf(stderr, "[graph] capture pos=%d seqlen=%d n_splits=%d attn_mode=%d mma_chunk=%d sparse=%d\n",
                        position, seqlen, s.n_splits, attn_graph_mode, mma_chunk, sparse_on ? 1 : 0);
            }
            cu(cudaGraphLaunch(s.cu_exec, st), "graph launch (first)");
        }
    }

    cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
    cu(cudaStreamSynchronize(st), "sync");
    if (mgdump) {
        std::vector<bf16> host((size_t)(2 * c.n_layers + 1) * H);
        cu(cudaMemcpy(host.data(), s.dbg_xn_dump, host.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
           "dbg_xn_dump readback");
        const char* path = getenv("SPARKINFER_MG_DUMP_FILE");
        // This fires on EVERY call at this position, not just the first, so a process that calls
        // forward_token(_, dump_step, true) more than once (AR ref, then AR+capture isolation, then
        // DFlash's own token loop) used to silently overwrite the same file. Suffix with a call
        // counter so each occurrence survives.
        static int mgdump_call = 0;
        std::string path_s = std::string(path ? path : "/tmp/mg_xn_dump.bin") + "." + std::to_string(mgdump_call++);
        FILE* f = fopen(path_s.c_str(), "wb");
        if (f) { fwrite(host.data(), sizeof(bf16), host.size(), f); fclose(f); }
        fprintf(stderr, "[mg-debug] dumped xn[layer,H=%d] for step=%d, %d layers -> %s\n",
                H, position, c.n_layers, path_s.c_str());
    }
    return *s.h_out_id;
}

// ---------------------------------------------------------------------------
// (dual-GPU WP-9) tensor-parallel=2 decode twin. The ~1,500-line tp=1 body above is
// byte-identical; every tp>1 site below is degenerate (tp_attach is never called there).
// ---------------------------------------------------------------------------

bool Qwen35Model::tp_active() const {
    return p_->tp_link != nullptr && p_->tp_peers.size() == 2;
}

int Qwen35Model::tp_rank() const { return p_->tp_rank; }

Qwen35Model* Qwen35Model::tp_mirror_peer() const {
    const Impl& s = *p_;
    if (t_tp_mirror_depth != 0 || s.tp_rank != 0 || !s.tp_link || s.tp_peers.size() != 2) return nullptr;
    return s.tp_peers[1];
}

Qwen35Model::TpRankView Qwen35Model::tp_rank_view() const {
    const Impl& s = *p_;
    TpRankView v;
    v.device = s.device;
    v.stream = s.stream;
    v.xrow   = s.tp_xrow;
    v.ar     = s.tp_ar;
    v.logits = s.logits;
    return v;
}

void Qwen35Model::tp_attach(GpuLink* link, int my_rank, const std::vector<Qwen35Model*>& peers) {
    if (!link) {
        fprintf(stderr, "[tp] tp_attach: null GpuLink, staying tp=1\n");
        return;
    }
    if (my_rank < 0 || my_rank > 1) {
        fprintf(stderr, "[tp] tp_attach: rank %d unsupported (tp is capped at 2), staying tp=1\n",
                my_rank);
        return;
    }
    // Normalize both accepted attach forms into a rank-ordered 2-slot world: world[my_rank]=this
    // and the other rank's model in world[1-my_rank]. The engine may pass (a) the item-11
    // shorthand -- the leader the single OTHER rank, the peer nothing -- or (b) the full
    // rank-ordered list whose slot r is rank r's model (so peers[my_rank]==this). Any other
    // shape is rejected and the model stays tp=1.
    std::vector<Qwen35Model*> world(2, nullptr);
    world[my_rank] = this;
    if (peers.size() == 2) {
        for (int r = 0; r < 2; r++)
            if (peers[r] && peers[r] != this) world[r] = peers[r];
    } else if (peers.size() == 1) {
        if (peers[0] != this) world[1 - my_rank] = peers[0];
        // peers[0]==this: handed only our own slot; the other slot stays null. That is fine for
        // the peer (rank 1), which never issues the all-reduce; the leader (rank 0) is always
        // handed the peer by item 11, so it never ends up with an empty peer slot.
    } else if (peers.size() != 0) {
        fprintf(stderr, "[tp] tp_attach: unexpected peers size %zu, staying tp=1\n", peers.size());
        return;
    }
    p_->tp_peers = world;
    p_->tp_rank = my_rank;
    // Per-rank scratch on THIS model's own device (never shared across ranks). The ctor bound the
    // main thread to it only when (rank,device)!=(0,0); tp_attach runs later on the engine thread,
    // which may sit on another context, so save/restore the device around the allocs. The op
    // path never setDevice -- GpuLink is (device,stream)-pure.
    Impl& s = *p_;
    int prev = -1;
    cu(cudaGetDevice(&prev), "tp_attach getDevice");
    cu(cudaSetDevice(s.device), "tp_attach setDevice");
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    s.tp_xrow   = s.alloc<bf16>((size_t)H);
    s.tp_ar     = s.alloc<bf16>((size_t)32 * 2 * H);
    // GDN qkv staging, sized from this model's own GDN window (the ctor already normalized it):
    // rank r owns q/k heads [ql*r, ql*(r+1)) and v heads [vloc*r, vloc*(r+1)) with ql =
    // vloc/g in block mode, so the one qkv GEMV writes 2*ql*HD + vloc*HD rows.
    if (c.hybrid) {
        const int HD = c.linear_head_dim;
        const int gl = c.linear_v_heads / c.linear_q_heads;
        const int vl = p_->gdn_window.v_count > 0 ? p_->gdn_window.v_count : c.linear_v_heads;
        const int ql = c.gdn_qh_block ? vl / gl : vl;
        s.tp_qkv = s.alloc<bf16>((size_t)(2 * ql + vl) * HD);
        // Full-attn staging: the fused Q+gate window (qdim bf16) and the pos/seqlen scalars.
        s.tp_qraw  = s.alloc<bf16>((size_t)c.n_q_heads * c.head_dim);
        s.tp_pos   = s.alloc<int>(2);
        s.h_tp_pos = (int*)malloc(2 * sizeof(int));
    }
    // Dense-FFN staging: gate/up each own a fl = moe_ffn/2 N window and down owns the matching
    // K window of the swiglu output, so one 3*fl bf16 block covers all three. The K-window
    // activation quants need their own scratch (aq81/aq8 are kmax-sized, which on 27B is 6144
    // -- smaller than fl = 8704); the NVFP4 variant reuses nv_xq/nv_xs (moe_ffn-sized), as tp=1.
    if (c.dense_ffn && c.moe_ffn / 2 > 0) {
        const int fl = c.moe_ffn / 2;
        s.tp_ffn   = s.alloc<bf16>((size_t)3 * fl);
        s.tp_fq81  = s.alloc<char>(kernels::llama_q8_1_bytes(fl));
        s.tp_fq8   = s.alloc<signed char>(fl);
        s.tp_fq8_d  = s.alloc<float>(fl / 32);
        s.tp_fq8_s  = s.alloc<float>(fl / 32);
    }
    // Zero the exchange rows up front so the (leader-only) all-reduce can never read
    // uninitialized memory in this session.
    cu(cudaMemsetAsync(s.tp_xrow, 0, (size_t)H * sizeof(bf16), s.stream), "tp xrow init");
    cu(cudaMemsetAsync(s.tp_ar, 0, (size_t)32 * 2 * H * sizeof(bf16), s.stream), "tp ar init");
    cu(cudaSetDevice(prev), "tp_attach restore");
    p_->tp_link = link;
    // (S7a-1) Publish for the prefill o_proj all-reduce. Slot by link end, not by my_rank, so the
    // leader is always the rank on the device_a side whatever its logical rank is.
    {
        const int slot = (s.device == link->device_a()) ? 0 : 1;
        g_tp_prefill_link = link;
        // FP4 prefill at any row count (not just m % 8 == 0): a prompt whose length is not a
        // multiple of 8 otherwise runs every projection through the int8 conversion path.
        kernels::prefill_nvfp4_set_any_m(true);
        g_tp_prefill_dev[slot] = s.device;
        g_tp_prefill_stream[slot] = s.stream;
    }
    fprintf(stderr, "[tp] rank %d attached (device %d; H=%d V=%d %d layers; peer slot %s)\n",
            my_rank, s.device, H, c.vocab, c.n_layers,
            world[1 - my_rank] ? "present" : "EMPTY");
}

int Qwen35Model::forward_token_tp(int token_id, int position, bool sample, float temperature,
                                  unsigned long long seed, unsigned long long sample_step,
                                  int top_k, float top_p,
                                  float presence_penalty, float frequency_penalty) {
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    cudaStream_t st = s.stream;

    // tp>1 decode is EAGER-only: a cross-device P2P op is not stream-capturable, so the graph
    // machinery the tp=1 body relies on is deliberately bypassed (one-shot note).
    {
        static bool eager_noted = false;
        if (!eager_noted) {
            eager_noted = true;
            fprintf(stderr, "[tp] tp=2 decode is EAGER-only (no CUDA-graph capture; P2P is "
                            "not stream-capturable)\n");
        }
    }

    // The peer rank runs this same body concurrently: forward_token / decode_packed replay the
    // public call on it (TpMirrorScope), and the link ops below rendezvous the two.

    // -- op entry: exchange the embedding row (design item 3) --
    // The embedding table is row-split on the vocab axis, each rank holding V/2 CONSECUTIVE rows
    // (rank r: rows [r*V/2, (r+1)*V/2)), so the input row lives on exactly one rank. The owner
    // GEMVs its local copy of the row into tp_xrow -- indexed by the LOCAL row (global id minus
    // its slice base, since its table holds only its half), the other rank zeroes its copy, and
    // one all-reduce gives both ranks the full row. It then seeds x and primes layer 0's input
    // norm, exactly as the tp=1 body's op entry does.
    // A sequence's first token starts from zero recurrent + conv state, as in the tp=1 body:
    // open_session does NOT zero lin_state/lin_conv_state (the position-0 reset there and in the
    // prefill does), so without this a rank decodes on whatever its fresh allocation held.
    if (c.hybrid && position == 0 && s.lin_state && s.lin_conv_state) {
        cu(cudaMemsetAsync(s.lin_state, 0,
                           (size_t)gdn_state_slots(c) * gdn_v_local() * c.linear_head_dim *
                               c.linear_head_dim * sizeof(float), st),
           "tp linear state reset");
        cu(cudaMemsetAsync(s.lin_conv_state, 0,
                           (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim * sizeof(bf16), st),
           "tp linear conv reset");
    }

    const int rows_per_rank = c.vocab / 2;
    const int owning_rank = rows_per_rank > 0 ? token_id / rows_per_rank : 0;
    if ((int)s.tp_rank == owning_rank) {
        s.h_scalars[0] = token_id - owning_rank * rows_per_rank;
        cu(cudaMemcpyAsync(s.d_scalars, s.h_scalars, sizeof(int), cudaMemcpyHostToDevice, st),
           "tp entry token");
        kernels::launch_embedding(s.d_tok, s.w.embed_tokens, s.tp_xrow, 1, H, st);
    } else {
        cu(cudaMemsetAsync(s.tp_xrow, 0, (size_t)H * sizeof(bf16), st), "tp xrow zero");
    }
    tp_allreduce_row(s.tp_xrow, (size_t)H, /*is_xrow=*/true);
    cu(cudaMemcpyAsync(s.x, s.tp_xrow, (size_t)H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
       "tp xrow -> x");
    kernels::launch_rmsnorm(s.x, s.w.layers[0].input_norm, s.xn, 1, H, c.rms_eps, st);

    s.tp_cur_pos = position;   // the attention layers' KV slot / rotary position
    // DSpark hidden-state capture (leader only: set_dflash_capture is not mirrored, so the peer's
    // flag stays off). launch_capture_row reads its row from d_scalars[4], as in the tp=1 body.
    if (s.dflash_capture && s.dflash_hidden) {
        s.h_scalars[4] = s.dflash_cap_row;
        cu(cudaMemcpyAsync(s.d_cap_row, s.h_scalars + 4, sizeof(int), cudaMemcpyHostToDevice, st),
           "tp capture row");
    }

    // DEBUG ONLY: SPARKINFER_MG_DUMP_STEP=<position> dumps, at that step, the same
    // [2*n_layers+1, H] bf16 layout as the tp=1 dump (each layer's input-normed xn, the final
    // norm, then each layer's post-attention hn) into <SPARKINFER_MG_DUMP_FILE>.tp<rank>.<n> --
    // for layer-by-layer tp=1 vs tp=2 diffing.
    static int tp_dump_step = -2;
    if (tp_dump_step < -1) { const char* e = getenv("SPARKINFER_MG_DUMP_STEP"); tp_dump_step = e ? atoi(e) : -1; }
    std::vector<bf16> tp_dump;
    if (tp_dump_step == position) tp_dump.resize((size_t)(2 * c.n_layers + 1) * H);
    auto tp_dump_xn = [&](int slot, const void* src = nullptr) {
        if (tp_dump.empty()) return;
        cu(cudaMemcpyAsync(tp_dump.data() + (size_t)slot * H, src ? src : s.xn, (size_t)H * sizeof(bf16),
                           cudaMemcpyDeviceToHost, st), "tp dump");
        cu(cudaStreamSynchronize(st), "tp dump sync");
    };

    // -- 64-layer loop: per-layer partials + one all-reduce per block; layer tails unchanged --
    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        tp_dump_xn(L);
        // The two attention flavors share the same tp shape: each helper computes this rank's
        // partial into the tp_ar row (GDN: windowed qkv GEMV + rank-local conv/recurrence/norm +
        // K-windowed ssm_out; attn: N-windowed q/k/v GEMVs + rank-local QK-norm/RoPE/KV-append/
        // flash-decode + K-windowed o_proj), which the AR-A/AR-B all-reduces below combine.
        if (is_linear_layer(c, L)) tp_gdn_layer_tp(L, s.x, s.xn);
        else                        tp_attn_layer_tp(L, s.x, s.xn);
        // AR-A: combine the attention-block partial (o_proj/s.ao once the body lands) across ranks.
        tp_allreduce_row(s.tp_ar, (size_t)H, /*is_xrow=*/false);   // the partial is row 0 [H]
        // tail1 (unchanged vs tp=1): h = x + attn_out ; hn = RMSNorm(h, post_attn_norm)
        kernels::launch_add_rmsnorm2(s.x, s.tp_ar, w.post_attn_norm, s.h, s.hn, 1, H, c.rms_eps, st);
        tp_dump_xn(c.n_layers + 1 + L, s.hn);
        // FFN partial (tp_ar row 0: this rank's ffn partial; AR-B below combines the ranks).
        tp_ffn_layer_tp(L, s.xn, s.tp_ar);
        // AR-B: combine the ffn_down partial across ranks.
        tp_allreduce_row(s.tp_ar, (size_t)H, /*is_xrow=*/false);   // the partial is row 0 [H]
        // tail2 (unchanged vs tp=1): x = h + ffn_out ; xn = RMSNorm(x, nextnorm)
        const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_add_rmsnorm2(s.h, s.tp_ar, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
        dflash_maybe_capture_layer(L);   // x = this layer's output, replicated on both ranks
    }
    tp_dump_xn(c.n_layers);
    if (!tp_dump.empty()) {
        static int tp_dump_call[2] = {0, 0};
        const char* path = getenv("SPARKINFER_MG_DUMP_FILE");
        const std::string ps = std::string(path ? path : "/tmp/mg_xn_dump.bin") + ".tp" +
                               std::to_string(s.tp_rank) + "." + std::to_string(tp_dump_call[s.tp_rank & 1]++);
        if (FILE* f = fopen(ps.c_str(), "wb")) { fwrite(tp_dump.data(), sizeof(bf16), tp_dump.size(), f); fclose(f); }
        fprintf(stderr, "[tp-debug] dumped xn for step=%d -> %s\n", position, ps.c_str());
    }

    // -- epilogue: vocab-split lm_head GEMV, then ONE f32 all-reduce that hands both ranks the
    // full logits row, then the tp=1 sampling sequence verbatim. Each rank GEMVs the final normed
    // row against its own consecutive half of the lm_head (global rows [rank*Vr, (rank+1)*Vr))
    // into its half of s.logits, the other half zeroed; the sum all-reduce (x + 0 is exact) leaves
    // a bit-identical [V] row on both ranks. Every kernel after it is the tp=1 one on identical
    // inputs and identical device params (same seed/step), so both ranks pick the same token and
    // keep their replicated penalty counts in lockstep -- and top_k/top_p, temperature sampling
    // and logprobs (last_token_logprobs reads this rank's d_topk_exp/d_chosen_logit) are EXACTLY
    // tp=1's, with no per-vocab-half approximation. Cost: one 1 MB f32 reduce per sampled token
    // (~0.15 ms over P2P on the PCIe pair, vs ~30 ms per decode step). The 27B decode lm_head is
    // Q4_K in both variants, so the arm chain mirrors tp=1's with N=Vr.
    const int Vr  = c.vocab / 2;              // rows this rank owns (27B: 124160)
    const int off = (int)s.tp_rank * Vr;      // global base of this rank's window
    float* logits_h = s.logits + off;         // this rank's half of the row

    // !sample: the tp=1 body never runs lm_head on this path either -- sync and return the
    // input token (no logits needed).
    if (!sample) {
        cu(cudaStreamSynchronize(st), "tp noprobs sync");
        return token_id;
    }

    // Sampling params, refreshed on THIS stream (the tp op entry does not run the tp=1 refresh
    // block) so the mask/sampling kernels see this step's values.
    *s.h_sample_temp = temperature;
    *s.h_sample_seed = seed;
    *s.h_sample_step = sample_step;
    cu(cudaMemcpyAsync(s.d_sample_temp, s.h_sample_temp, sizeof(float), cudaMemcpyHostToDevice, st), "tp sample temp");
    cu(cudaMemcpyAsync(s.d_sample_seed, s.h_sample_seed, sizeof(unsigned long long), cudaMemcpyHostToDevice, st), "tp sample seed");
    cu(cudaMemcpyAsync(s.d_sample_step, s.h_sample_step, sizeof(unsigned long long), cudaMemcpyHostToDevice, st), "tp sample step");
    *s.h_sample_top_k = top_k;
    *s.h_sample_top_p = top_p;
    cu(cudaMemcpyAsync(s.d_sample_top_k, s.h_sample_top_k, sizeof(int), cudaMemcpyHostToDevice, st), "tp sample top_k");
    cu(cudaMemcpyAsync(s.d_sample_top_p, s.h_sample_top_p, sizeof(float), cudaMemcpyHostToDevice, st), "tp sample top_p");
    *s.h_sample_presence_penalty = presence_penalty;
    *s.h_sample_frequency_penalty = frequency_penalty;
    cu(cudaMemcpyAsync(s.d_sample_presence_penalty, s.h_sample_presence_penalty, sizeof(float), cudaMemcpyHostToDevice, st), "tp presence");
    cu(cudaMemcpyAsync(s.d_sample_frequency_penalty, s.h_sample_frequency_penalty, sizeof(float), cudaMemcpyHostToDevice, st), "tp frequency");

    cu(cudaMemsetAsync(s.logits, 0, (size_t)c.vocab * sizeof(float), st), "tp logits zero");
    // lm_head N-window GEMV into the arena half (mirror of the tp=1 arm chain, out=arena half).
    if (s.gguf && s.use_pq && s.use_llama && s.w.lm_head_type == 12) {
        kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
        kernels::launch_mmvq_q4k_f32(s.aq81, s.w.lm_head, logits_h, Vr, H, st);
    }
    else if (s.gguf && s.use_q6mmvq && s.w.lm_head_type == 14) {
        kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
        kernels::launch_gemv_q6k_dp4a_f32(s.aq81, s.w.lm_head, logits_h, Vr, H, st);
    }
    else if (s.gguf && s.w.lm_head_type)
        kernels::launch_gemv_q_f32(s.xn, s.w.lm_head, s.w.lm_head_type, logits_h, Vr, H, st);
    else if (s.gguf)
        kernels::launch_gemv_f32(s.xn, s.w.lm_head, logits_h, Vr, H, st);
    else
        kernels::launch_linear_f32(s.xn, s.w.lm_head, logits_h, 1, Vr, H, st);

    tp_allreduce_logits();

    kernels::launch_logit_bias(s.logits, s.logit_bias, c.vocab, st);
    kernels::launch_presence_frequency_penalty(s.logits, s.penalty_counts, c.vocab,
                                               s.d_sample_presence_penalty, s.d_sample_frequency_penalty, st);
    kernels::launch_topk_topp_mask(s.logits, c.vocab, s.d_vocab_iota, s.d_sorted_logits, s.d_sorted_idx,
                                   s.d_topk_exp, s.d_topk_cumsum, s.d_sort_temp, s.sort_temp_bytes,
                                   s.d_scan_temp, s.scan_temp_bytes,
                                   s.d_sample_top_k, s.d_sample_top_p, s.d_rank_by_id, st);
    kernels::launch_temperature_sample(s.logits, 1, c.vocab, s.d_sample_temp, s.d_sample_seed,
                                       s.d_sample_step, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    kernels::launch_extract_chosen_logit(s.d_out_id, s.d_rank_by_id, s.d_sorted_logits,
                                         s.d_chosen_logit, st);
    kernels::launch_increment_penalty_count(s.penalty_counts, s.d_out_id, st);
    cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "tp out_id");
    cu(cudaStreamSynchronize(st), "tp final sync");
    return *s.h_out_id;
}

void Qwen35Model::tp_allreduce_logits() {
    Impl& s = *p_;
    // Leader-issued like tp_allreduce_row: one GpuLink::allreduce posts the in-place f32 sum of
    // s.logits on BOTH ranks' streams; the peer only takes part in the rendezvous.
    if (!s.tp_link || s.tp_peers.size() != 2) return;
    if (s.tp_rank != 0) { tp_peer_rendezvous("logits allreduce"); return; }
    if (!s.tp_peers[1]) return;
    const TpRankView pv = s.tp_peers[1]->tp_rank_view();
    GpuLink::RankRef self_ref{s.device, s.stream, s.logits, s.logits};
    GpuLink::RankRef peer_ref{pv.device, pv.stream, pv.logits, pv.logits};
    GpuLink::RankRef a = self_ref, b = peer_ref;
    if (s.tp_link->device_a() != s.device) { a = peer_ref; b = self_ref; }
    tp_leader_rendezvous("logits allreduce", [&] {
        if (!s.tp_link->allreduce(a, b, (size_t)s.cfg.vocab * sizeof(float), GpuLink::Dtype::Float32))
            cu(cudaErrorUnknown, "tp logits allreduce");
    });
}

constexpr int kTpVerifyRows = 32;   // = kQwen35MaxPackedRows: one rows pass per packed step

// (dual-GPU WP-12) verify_rows_tp's scratch, all-or-nothing and agreed across the ranks. Called
// lazily by the first verify, or up front by reserve_tp_verify() so the draft's load accounts for it.
bool Qwen35Model::tp_verify_alloc() {
    Impl& s = *p_;
    if (s.vr_ready) return true;
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    const int R = kTpVerifyRows;
    const int HD = c.head_dim;
    const int qdim_l = (c.n_q_heads / 2) * HD, kvdim_l = (c.n_kv_heads / 2) * HD;
    const int fl = c.moe_ffn / 2;
    const int lhd = c.linear_head_dim;
    const int vloc = s.gdn_window.v_count;
    const int g = c.linear_v_heads / c.linear_q_heads;
    const int ql = c.gdn_qh_block ? vloc / g : vloc;
    const int wq = (2 * ql + vloc) * lhd;
    const int lqkv = s.linear_qkvdim;
    const int Kw = vloc * lhd;
    const int V = c.vocab, Vr = V / 2;
    const int kmax = std::max(std::max(H, fl), std::max(2 * qdim_l, Kw));
    const size_t ls = (size_t)gdn_state_slots(c) * gdn_v_local() * lhd * lhd;
    const size_t cs = (size_t)c.n_layers * (c.linear_conv_kernel - 1) * lqkv;
    {
        // All-or-nothing: on a failed allocation free what this call took and decline, on BOTH
        // ranks (agreed below) -- a verify running on null scratch writes through null pointers.
        std::vector<void*> got;
        bool alloc_ok = true;
        auto va = [&](size_t bytes) -> void* {
            void* p = nullptr;
            if (!alloc_ok || cudaMalloc(&p, bytes) != cudaSuccess) { alloc_ok = false; return nullptr; }
            got.push_back(p);
            return p;
        };
        const size_t rh = (size_t)R * H * sizeof(bf16);
        s.vr_x = (bf16*)va(rh); s.vr_xn = (bf16*)va(rh); s.vr_h = (bf16*)va(rh);
        s.vr_hn = (bf16*)va(rh); s.vr_ar = (bf16*)va(rh);
        s.vr_nq = (signed char*)va((size_t)R * kmax);
        s.vr_ns = (float*)va((size_t)R * (kmax / 16 + 1) * sizeof(float));
        s.vr_q81_row = kernels::llama_q8_1_bytes(kmax);
        s.vr_q81 = (char*)va((size_t)R * s.vr_q81_row);
        s.vr_rec_qkv = (bf16*)va((size_t)c.n_layers * R * wq * sizeof(bf16));
        s.vr_rec_a = (bf16*)va((size_t)c.n_layers * R * vloc * sizeof(bf16));
        s.vr_rec_b = (bf16*)va((size_t)c.n_layers * R * vloc * sizeof(bf16));
        s.vr_full = (bf16*)va((size_t)R * lqkv * sizeof(bf16));
        s.vr_z = (bf16*)va((size_t)R * Kw * sizeof(bf16));
        s.vr_gdn = (bf16*)va((size_t)R * Kw * sizeof(bf16));
        s.vr_ln = (bf16*)va((size_t)R * Kw * sizeof(bf16));
        s.vr_qraw = (bf16*)va((size_t)R * 2 * qdim_l * sizeof(bf16));
        s.vr_q = (bf16*)va((size_t)R * qdim_l * sizeof(bf16));
        s.vr_g = (bf16*)va((size_t)R * qdim_l * sizeof(bf16));
        s.vr_attn = (bf16*)va((size_t)R * qdim_l * sizeof(bf16));
        s.vr_k = (bf16*)va((size_t)R * kvdim_l * sizeof(bf16));
        s.vr_v = (bf16*)va((size_t)R * kvdim_l * sizeof(bf16));
        s.vr_ffn = (bf16*)va((size_t)3 * R * fl * sizeof(bf16));
        s.vr_lh = (float*)va((size_t)R * Vr * sizeof(float));
        s.vr_logits = (float*)va((size_t)R * V * sizeof(float));
        s.vr_ids = (int*)va((size_t)4 * R * sizeof(int));
        s.vr_pos = s.vr_ids + R; s.vr_seq = s.vr_ids + 2 * R; s.vr_out = s.vr_ids + 3 * R;
        s.vr_snap_lin = (float*)va(ls * sizeof(float));
        s.vr_snap_conv = (bf16*)va(cs * sizeof(bf16));
        {
            const size_t K = dflash_kernels::kRowsTopkMax;
            s.vr_tk_v = (float*)va((size_t)R * K * sizeof(float));
            s.vr_tk_i = (int*)va((size_t)R * K * sizeof(int));
            s.vr_mg_v = (float*)va((size_t)R * K * sizeof(float));
            s.vr_mg_i = (int*)va((size_t)R * K * sizeof(int));
            s.vr_smp_temp = (float*)va((size_t)R * sizeof(float));
            s.vr_smp_topp = (float*)va((size_t)R * sizeof(float));
            s.vr_smp_topk = (int*)va((size_t)R * sizeof(int));
            s.vr_smp_out = (int*)va((size_t)R * sizeof(int));
            s.vr_smp_seed = (unsigned long long*)va((size_t)R * sizeof(unsigned long long));
            s.vr_smp_step = (unsigned long long*)va((size_t)R * sizeof(unsigned long long));
            if (alloc_ok && !s.h_vr_smp &&
                cudaMallocHost(&s.h_vr_smp, (size_t)R * K * 16 + (size_t)R * 32) != cudaSuccess) {
                s.h_vr_smp = nullptr;
                alloc_ok = false;
            }
        }
        if (alloc_ok && !s.h_vr && cudaMallocHost(&s.h_vr, (size_t)4 * R * sizeof(int)) != cudaSuccess) {
            s.h_vr = nullptr;
            alloc_ok = false;
        }
        if (alloc_ok && !s.h_vr_maxv &&
            cudaMallocHost(&s.h_vr_maxv, (size_t)R * sizeof(float)) != cudaSuccess) {
            s.h_vr_maxv = nullptr;
            alloc_ok = false;
        }
        cudaGetLastError();   // a failed cudaMalloc leaves a sticky-free error in the runtime state
        alloc_ok = tp_prefill_agree_min(alloc_ok ? 1 : 0) != 0;
        if (!alloc_ok) {
            for (void* p : got) cudaFree(p);
            static bool noted = false;
            if (!noted) {
                noted = true;
                fprintf(stderr, "[tp] DSpark verify scratch does not fit on this card -- speculative "
                                "verify declined (ordinary decode serves the request); lower --ctx\n");
            }
            return false;
        }
        for (void* p : got) s.owned.push_back(p);
        s.vr_ready = true;
    }
    return true;
}

// Reserve the tp verify scratch now, on both ranks (mirrored), so its memory is taken before
// prompts size their prefill arenas against what is free. false = it does not fit.
bool Qwen35Model::reserve_tp_verify() {
    if (!tp_active()) return true;
    bool peer_ok = true;
    {
        TP_MIRROR(reserve_tp_verify_local(&peer_ok));
        reserve_tp_verify_local(nullptr);
    }
    return p_->vr_ready && peer_ok;
}

void Qwen35Model::reserve_tp_verify_local(bool* ok) {
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    const bool r = tp_verify_alloc();
    if (ok) *ok = r;
}

// (dual-GPU WP-12) DSpark's batched verify at tp=2: `n` consecutive positions of the active
// sequence in one pass, returning the accepted-prefix length (or -1: declined, nothing changed).
// Both ranks run it (batched_forward mirrors the call). It is forward_token_tp with a row axis:
// every projection runs once over all rows on the row-batched kernel whose per-row arithmetic is
// the single-row decode kernel's (NVFP4 rows-dp4a, Q4_K/Q6_K rows MMVQ), or row by row on the
// decode kernel itself where there is no such twin; the norms are row-wise; the KV append takes
// the rows' positions; attention and the GDN recurrence run row by row on the decode kernels;
// each block's all-reduce covers all rows at once; and the head ends in the decode epilogue's
// zero-padded full-vocab all-reduce + argmax. So every row's argmax is the token forward_token_tp
// would have produced at that position -- DSpark stays byte-lossless against tp=2 AR -- while
// the weights stream once per block instead of once per token.
//
// The GDN state cannot be un-stepped, so it is snapshotted on entry; when the draft is only
// partly accepted the snapshot is restored and the kept rows' recorded projections are replayed
// through the same conv + recurrence kernels. KV rows past the accepted prefix are overwritten
// by the next step, as in the tp=1 verify.
int Qwen35Model::verify_rows_tp(const int* ids, int n, int start_pos, void* capture_dst,
                                int* out_argmax) {
    return tp_rows_forward(ids, n, start_pos, nullptr, nullptr, capture_dst, out_argmax);
}

// (dual-GPU C1) The row-batched tp forward behind both verify_rows_tp (row_seq == nullptr: n
// consecutive positions of the ACTIVE session, snapshot + partial-accept replay) and
// decode_packed_tp (row_seq != nullptr: row r is one decode step of session row_seq[r] at
// row_pos[r], every row kept). Per row the arithmetic is the single-row decode body's either way;
// only the state each row reads and writes (GDN conv/recurrent state, KV block table, position)
// is chosen per row.
int Qwen35Model::tp_rows_forward(const int* ids, int n, int start_pos, const int* row_pos,
                                 const uint64_t* row_seq, void* capture_dst, int* out_argmax,
                                 int seg_n, void* const* seg_capture, int* seg_keep,
                                 const SpecSampleRow* row_sample, const uint32_t* const* row_mask) {
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    const bool multi = row_seq != nullptr;
    // (dual-GPU C2) Segmented: seg_n sessions each verify T = n / seg_n consecutive positions,
    // rows STEP-major (row t * seg_n + j is session j's position row_pos[j] + t), so the rows of
    // one step are contiguous and the GDN recurrence advances one step for all sessions per
    // launch. Each session is snapshotted, kept to its own accepted prefix (seg_keep[j]) and
    // replayed like the single-session verify.
    const bool seg = multi && seg_n > 0;
    const int S = seg ? seg_n : 0;
    const int T = seg ? n / seg_n : 1;
    if (seg && (n % seg_n != 0 || !seg_keep)) return -1;
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    const int R = kTpVerifyRows;
    cudaStream_t st = s.stream;
    // Every decline happens here, before the first link op, and depends only on state both ranks
    // share (config, loaded formats, the mirrored session), so the ranks always decline together.
    if (!ids || !out_argmax || n < 1 || n > R) return -1;
    if (!s.gguf || !c.hybrid || !c.dense_ffn || (s.use_pq && !s.use_llama)) return -1;
    if (multi && !row_pos) return -1;
    // Per-row GDN state (multi: each row's own session; verify: the active one for every row).
    float* row_lin[kTpVerifyRows];
    bf16* row_conv[kTpVerifyRows];
    uint64_t row_kv[kTpVerifyRows];
    for (int r = 0; r < n; r++) {
        if (multi) {
            auto it = s.sessions.find(row_seq[r]);
            if (it == s.sessions.end() || it->second.lin_state_b16 || !it->second.lin_state ||
                !it->second.lin_conv_state)
                return -1;
            row_lin[r] = static_cast<float*>(it->second.lin_state);
            row_conv[r] = static_cast<bf16*>(it->second.lin_conv_state);
            row_kv[r] = row_seq[r];
        } else {
            if (s.active_lin_state_b16 || !s.lin_state || !s.lin_conv_state) return -1;
            row_lin[r] = s.lin_state;
            row_conv[r] = s.lin_conv_state;
            row_kv[r] = s.active_seq_id;
        }
    }
    const int HD = c.head_dim;
    const int n_q = c.n_q_heads / 2, n_kv = c.n_kv_heads / 2;
    const int qdim_l = n_q * HD, kvdim_l = n_kv * HD;
    const int fl = c.moe_ffn / 2;
    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        if (!(w.gate_nv && w.up_nv && w.down_nv)) return -1;
        if (is_linear_layer(c, L)) continue;
        if (!(c.rope_dim > 0 && c.rope_dim < HD) || w.wgate != nullptr) return -1;
        // The decode body fuses these into its flash-decode / o_proj quantize; rows here do not.
        const bool attn_gate_q8 = w.q_has_gate && s.use_pq && s.use_llama &&
                                  (H == 2048 || H == 4096) && (w.wo_type == 12 || w.wo_type == 8) &&
                                  (s.qdim % 32 == 0);
        const bool emit_attn_q8 = !w.q_has_gate && s.use_attnin && s.use_pq && s.use_llama &&
                                  w.wo_type == 12;
        static int attn_gq8 = -1;
        if (attn_gq8 < 0) { const char* e = getenv("SPARKINFER_ATTN_GQ8"); attn_gq8 = (e && e[0] == '0') ? 0 : 1; }
        if ((attn_gq8 && attn_gate_q8) || emit_attn_q8) return -1;
    }

    const int lhd = c.linear_head_dim;
    const int vloc = s.gdn_window.v_count;
    const int v0 = s.gdn_window.v_count > 0 ? s.gdn_window.v_start : 0;
    const int g = c.linear_v_heads / c.linear_q_heads;
    const int q0 = c.gdn_qh_block ? v0 / g : 0;
    const int ql = c.gdn_qh_block ? vloc / g : vloc;
    const int wq = (2 * ql + vloc) * lhd;            // the rank's packed q|k|v row
    const int lqkv = s.linear_qkvdim;
    const int lqdim = c.linear_q_heads * lhd;
    const int Kw = vloc * lhd;
    const int V = c.vocab, Vr = V / 2;
    const int kmax = std::max(std::max(H, fl), std::max(2 * qdim_l, Kw));
    const size_t ls = (size_t)gdn_state_slots(c) * gdn_v_local() * lhd * lhd;
    const size_t cs = (size_t)c.n_layers * (c.linear_conv_kernel - 1) * lqkv;

    if (!s.vr_ready && !tp_verify_alloc()) return -1;
    // A sampled row must take the exact sampled draw below (top_k in [1, 64], its scratch
    // allocated); otherwise decline now, before any state is touched, rather than emit an argmax.
    if (row_sample) {
        bool any = false;
        for (int r = 0; r < n; r++)
            if (row_sample[r].temperature > 0.f) {
                any = true;
                if (row_sample[r].top_k < 1 || row_sample[r].top_k > dflash_kernels::kRowsTopkMax)
                    return -1;
            }
        if (any && !s.h_vr_smp) return -1;
    }
    // Constrained rows need their mask staging (both ranks agree, as for the FP4 rows below) and a
    // vocab half that splits into whole mask words.
    bool masked = false;
    if (row_mask)
        for (int r = 0; r < n; r++) masked |= row_mask[r] != nullptr;
    if (masked) {
        if (Vr % 32) return -1;
        if (s.vr_mask_state == 0) {
            const size_t bytes = (size_t)R * (Vr / 32) * sizeof(uint32_t);
            bool ok = cudaMalloc(&s.vr_mask, bytes) == cudaSuccess &&
                      cudaMallocHost(&s.h_vr_mask, bytes) == cudaSuccess;
            ok = tp_prefill_agree_min(ok ? 1 : 0) != 0;
            s.vr_mask_state = ok ? 1 : -1;
        }
        if (s.vr_mask_state < 0) return -1;
    }

    // (dual-GPU C1b) Multi-session rows on the FP4 tensor cores. The dp4a rows kernels re-read
    // the weights for every 8 rows, while the block-scaled GEMM reads them once for up to ~32
    // rows at the weight-read floor (measured on one 5060 Ti: gate|up 111 us at M = 8, 16 and 32;
    // dp4a needs 123 us per 8). Activations are FP4 (W4A4, as the prefill), so a packed row is no
    // longer bit-identical to the same row decoded alone; deterministic mode keeps the dp4a rows.
    // Rows pad to a multiple of 8 (the GEMM's row rule); every scratch buffer holds R rows and
    // rows are independent, so the padding rows compute garbage nobody reads.
    // SPARKINFER_TP_DECODE_TC=0 disables; _MINROWS (default 4) is the smallest batch that uses it.
    static const bool tc_env = [] {
        const char* e = getenv("SPARKINFER_TP_DECODE_TC"); return !(e && e[0] == '0');
    }();
    static const int tc_min = [] {
        // 4: measured at 1k context, aggregate decode C2 93.8 (TC) vs 98.1 (dp4a), C4 173.7 vs
        // 164.0 -- the crossover sits between 2 and 4 rows.
        const char* e = getenv("SPARKINFER_TP_DECODE_TC_MINROWS"); return e ? atoi(e) : 4;
    }();
    // A single session's DSpark verify (rows = consecutive positions) takes the same form: the
    // verify is then W4A4 like the packed decode, so its accepted tokens are no longer the
    // dp4a-decode's bit for bit -- outside deterministic mode only, as the packed decode.
    // Measured (HyperQwen cohort, 512 tokens, C1): verify 31.9 -> 25.2 ms, 77.4 -> 98.4 tok/s at
    // the same mean accept. SPARKINFER_TP_VERIFY_TC=0 keeps the dp4a verify.
    static const bool verify_tc_env = [] {
        const char* e = getenv("SPARKINFER_TP_VERIFY_TC"); return !(e && e[0] == '0');
    }();
    const int mp = (n + 7) & ~7;
    bool tc = (multi || verify_tc_env) && tc_env && n >= tc_min && mp <= R &&
              !deterministic_mode() && s.vr_tc_state >= 0;
    if (tc && s.vr_tc_state == 0) {
        const int kmax_tc = std::max(std::max(H, fl), std::max(qdim_l, Kw));
        size_t ws = kernels::prefill_nvfp4_workspace_bytes_f32(R, Vr, H);
        const int shapes[][2] = {{wq, H}, {Kw, H}, {H, Kw}, {2 * qdim_l, H}, {kvdim_l, H},
                                 {H, qdim_l}, {fl, H}, {H, fl}};
        for (auto& sh : shapes)
            ws = std::max(ws, kernels::prefill_nvfp4_workspace_bytes(R, sh[0], sh[1]));
        bool ok = cudaMalloc(&s.vr_tc_a, kernels::prefill_nvfp4_data_bytes(R, kmax_tc)) == cudaSuccess &&
                  cudaMalloc(&s.vr_tc_as, kernels::prefill_nvfp4_scale_bytes_a(R, kmax_tc)) == cudaSuccess &&
                  cudaMalloc(&s.vr_tc_ws, ws ? ws : 16) == cudaSuccess;
        ok = tp_prefill_agree_min(ok ? 1 : 0) != 0;
        s.vr_tc_state = ok ? 1 : -1;
        if (!ok) tc = false;
    }
    // The block-scaled form of a projection, when this pass runs it (else false: dp4a rows).
    auto proj_tc = [&](const void* fp4, const void* sf, float alpha, const bf16* in, int K,
                       void* out, int N) {
        return tc && fp4 && sf &&
               kernels::launch_prefill_nvfp4_quant_a(in, s.vr_tc_a, s.vr_tc_as, mp, K, st) &&
               kernels::launch_prefill_nvfp4_gemm(s.vr_tc_a, s.vr_tc_as, fp4, sf, out, mp, N, K,
                                                  s.vr_tc_ws, st, alpha);
    };
    // (plan 10, D2) An H-wide projection whose output is all-reduced, in two column halves: each
    // half's all-reduce runs on the side stream (flag kernel) as soon as the half is out, so the
    // first crosses the link while the second is computed. The halves land as [mp][H/2] twice in
    // vr_ar (add_rmsnorm2_split reads them); joined before returning. The A operand must already
    // be quantized (vr_tc_a, mp rows, K). Not bit-identical to the single GEMM (the half-width
    // GEMM splits K differently), which the tensor-core verify is not against plain decode
    // either; deterministic mode never takes the tensor-core verify. Both ranks decide alike.
    // SPARKINFER_TP_VERIFY_AR_SPLIT=0: one GEMM, one synchronous all-reduce.
    static const bool ar_split_env = [] {
        const char* e = getenv("SPARKINFER_TP_VERIFY_AR_SPLIT");
        return !(e && e[0] == '0');
    }();
    const int hh = H / 2;
    auto split_gemm_ar = [&](const void* fp4, const void* sf, float alpha, int K) -> bool {
        if (!ar_split_env || hh % 128 || !kernels::prefill_nvfp4_supported(mp, hh, K)) return false;
        const char* wb = static_cast<const char*>(fp4);
        const char* sb = static_cast<const char*>(sf);
        bf16* lo = s.vr_ar;
        bf16* hi = s.vr_ar + (size_t)mp * hh;
        if (!kernels::launch_prefill_nvfp4_gemm(s.vr_tc_a, s.vr_tc_as, wb, sb, lo, mp, hh, K,
                                                s.vr_tc_ws, st, alpha))
            return false;
        tp_prefill_allreduce_bf16_async(lo, (size_t)n * hh);
        if (!kernels::launch_prefill_nvfp4_gemm(s.vr_tc_a, s.vr_tc_as, wb + (size_t)hh * K / 2,
                                                sb + kernels::prefill_nvfp4_scale_bytes_b(hh, K),
                                                hi, mp, hh, K, s.vr_tc_ws, st, alpha))
            cu(cudaErrorInvalidValue, "tp verify split projection");
        tp_prefill_allreduce_bf16_async(hi, (size_t)n * hh);
        tp_prefill_allreduce_join();
        return true;
    };
    auto proj_tc_split = [&](const void* fp4, const void* sf, float alpha, const bf16* in, int K) {
        return tc && ar_split_env && fp4 && sf &&
               kernels::launch_prefill_nvfp4_quant_a(in, s.vr_tc_a, s.vr_tc_as, mp, K, st) &&
               split_gemm_ar(fp4, sf, alpha, K);
    };

    // Entry: ids / positions / seq_lens, the GDN snapshot, and the embedding (vocab-window gather
    // + all-reduce == the decode entry's owner-row exchange).
    auto pos_of = [&](int r) { return multi ? row_pos[r] : start_pos + r; };
    for (int r = 0; r < n; r++) {
        s.h_vr[r] = ids[r];
        s.h_vr[R + r] = pos_of(r);
        s.h_vr[2 * R + r] = pos_of(r) + 1;
    }
    cu(cudaMemcpyAsync(s.vr_ids, s.h_vr, (size_t)3 * R * sizeof(int), cudaMemcpyHostToDevice, st),
       "tp verify ids");
    // (dual-GPU C2) Segmented rows on the multi-step kernels: one conv + one recurrence launch per
    // layer for all sessions and steps, the state read-only until the accepted steps commit.
    // SPARKINFER_TP_SEG_MULTISTEP=0 keeps the per-step launches + snapshot/replay (A/B).
    static const bool ms_env = [] {
        const char* e = getenv("SPARKINFER_TP_SEG_MULTISTEP"); return !(e && e[0] == '0');
    }();
    std::vector<int> lin_ord(c.n_layers, -1);
    int n_lin = 0;
    for (int L = 0; L < c.n_layers; L++) if (is_linear_layer(c, L)) lin_ord[L] = n_lin++;
    bool ms = seg && ms_env && c.linear_head_dim == 128 && s.vr_ms_state >= 0;
    if (ms && s.vr_ms_state == 0) {
        const size_t rl = (size_t)n_lin * R;
        bool ok = cudaMalloc(&s.vr_ms_full, rl * lqkv * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ms_q, rl * lqdim * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ms_k, rl * lqdim * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ms_v, rl * c.linear_v_heads * lhd * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ms_a, rl * c.linear_v_heads * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ms_b, rl * c.linear_v_heads * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ms_keep, (size_t)R * sizeof(int)) == cudaSuccess &&
                  cudaHostAlloc(&s.h_vr_ms_keep, (size_t)R * sizeof(int), cudaHostAllocDefault) == cudaSuccess;
        if (ok) ok = cudaMemsetAsync(s.vr_ms_full, 0, rl * lqkv * sizeof(bf16), st) == cudaSuccess;
        cudaGetLastError();
        ok = tp_prefill_agree_min(ok ? 1 : 0) != 0;
        s.vr_ms_state = ok ? 1 : -1;
        if (!ok) ms = false;
    }
    if (seg && T > 1 && !ms) {
        if ((int)s.vr_seg_snap_lin.size() < S) {
            bool ok = true;
            while (ok && (int)s.vr_seg_snap_lin.size() < S) {
                void* a = nullptr;
                void* b = nullptr;
                ok = cudaMalloc(&a, ls * sizeof(float)) == cudaSuccess &&
                     cudaMalloc(&b, cs * sizeof(bf16)) == cudaSuccess;
                if (!ok) { if (a) cudaFree(a); if (b) cudaFree(b); cudaGetLastError(); break; }
                s.vr_seg_snap_lin.push_back((float*)a);
                s.vr_seg_snap_conv.push_back((bf16*)b);
            }
            if (tp_prefill_agree_min(ok ? 1 : 0) == 0) return -1;   // nothing touched yet
        }
        for (int j = 0; j < S; j++) {
            cu(cudaMemcpyAsync(s.vr_seg_snap_lin[j], row_lin[j], ls * sizeof(float),
                               cudaMemcpyDeviceToDevice, st), "tp seg snap lin");
            cu(cudaMemcpyAsync(s.vr_seg_snap_conv[j], row_conv[j], cs * sizeof(bf16),
                               cudaMemcpyDeviceToDevice, st), "tp seg snap conv");
        }
    }
    if (!multi) {
        cu(cudaMemcpyAsync(s.vr_snap_lin, s.lin_state, ls * sizeof(float), cudaMemcpyDeviceToDevice, st),
           "tp verify snap lin");
        cu(cudaMemcpyAsync(s.vr_snap_conv, s.lin_conv_state, cs * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "tp verify snap conv");
    }
    kernels::launch_embedding_vocab_window(s.vr_ids, s.w.embed_tokens, s.vr_x, n, H,
                                           (int)s.tp_rank * Vr, Vr, st);
    tp_prefill_allreduce_bf16(s.vr_x, (size_t)n * H);
    kernels::launch_rmsnorm(s.vr_x, s.w.layers[0].input_norm, s.vr_xn, n, H, c.rms_eps, st);

    // One projection over the n rows of `in` ([n][K]) into `out` ([n][N]). The NVFP4 arm is the
    // decode arm with M = n; everything else is the decode body's own single-row call per row.
    auto proj_rows = [&](const void* W, int t, const bf16* in, int K, void* out, int N) {
        bf16* y = static_cast<bf16*>(out);
        static const bool rowwise = getenv("SPARKINFER_TPV_ROWWISE") != nullptr;   // DEBUG A/B
        if (rowwise && t == kernels::SI_QTYPE_NVFP4 && kernels::qwen38_nvfp4_dp4a_proj()) {
            for (int r = 0; r < n; r++) {
                kernels::launch_gemv_nvfp4_quant_x(in + (size_t)r * K, s.vr_nq, s.vr_ns, 1, K, st);
                kernels::launch_gemv_nvfp4_rows_dp4a(s.vr_nq, s.vr_ns, W, y + (size_t)r * N, 1, N, K, st);
            }
            return;
        }
        if (t == kernels::SI_QTYPE_NVFP4 && kernels::qwen38_nvfp4_dp4a_proj()) {
            kernels::launch_gemv_nvfp4_quant_x(in, s.vr_nq, s.vr_ns, n, K, st);
            if (kernels::launch_gemv_nvfp4_rows_dp4a(s.vr_nq, s.vr_ns, W, y, n, N, K, st)) return;
        }
        for (int r = 0; r < n; r++) {
            const bf16* x = in + (size_t)r * K;
            bf16* yr = y + (size_t)r * N;
            char* q81 = s.vr_q81 + (size_t)r * s.vr_q81_row;
            if (s.use_pq && s.use_llama && t == 12) {
                kernels::launch_quantize_q8_1_blocks(x, q81, K, st);
                kernels::launch_mmvq_q4k(q81, W, yr, N, K, st);
            } else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14) {
                kernels::launch_quantize_q8_1_blocks(x, q81, K, st);
                kernels::launch_mmvq_q6k(q81, W, yr, N, K, st);
            } else if (s.use_pq && s.use_llama && t == 8) {
                kernels::launch_quantize_q8_1_blocks(x, q81, K, st);
                kernels::launch_mmvq_q80(q81, W, yr, N, K, st);
            } else if (t == kernels::SI_QTYPE_FP8) {
                kernels::launch_gemv_fp8(x, W, yr, N, K, st);
            } else if (t == kernels::SI_QTYPE_NVFP4) {
                kernels::launch_gemv_nvfp4(x, W, yr, N, K, st);
            } else if (t) {
                kernels::launch_gemv_q(x, W, t, yr, N, K, st);
            } else {
                kernels::launch_gemv(x, W, yr, N, K, st);
            }
        }
    };
    // q|k|v windows of the recorded rank rows -> the full-width packed qkv rows the conv reads.
    auto scatter_qkv = [&](const bf16* rec, int rows, bf16* dst = nullptr) {
        bf16* full = dst ? dst : s.vr_full;
        auto cp = [&](size_t dst_off, size_t src_off, int width) {
            cu(cudaMemcpy2DAsync(full + dst_off, (size_t)lqkv * sizeof(bf16),
                                 rec + src_off, (size_t)wq * sizeof(bf16),
                                 (size_t)width * sizeof(bf16), rows, cudaMemcpyDeviceToDevice, st),
               "tp verify qkv scatter");
        };
        cp((size_t)q0 * lhd, 0, ql * lhd);
        cp((size_t)lqdim + q0 * lhd, (size_t)ql * lhd, ql * lhd);
        cp((size_t)2 * lqdim + v0 * lhd, (size_t)2 * ql * lhd, vloc * lhd);
    };
    // One GDN step for row r of layer L (conv window + recurrence advance in place), exactly as
    // tp_gdn_layer_tp runs it; the gated-delta output lands in `out`.
    auto gdn_step = [&](int L, int r, bf16* out) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        bf16* conv_state = row_conv[r] + (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
        kernels::launch_qwen36_conv_split_l2norm_fused(s.vr_full + (size_t)r * lqkv, w.ssm_conv,
                                                       conv_state, s.lin_q, s.lin_k, s.lin_v,
                                                       c.linear_q_heads, c.linear_v_heads, lhd,
                                                       c.linear_conv_kernel, c.rms_eps, st,
                                                       q0, ql, q0, ql, v0, vloc);
        const size_t state_off = (size_t)gdn_state_slot(c, L) * vloc * lhd * lhd;
        const size_t ab = ((size_t)L * R + r) * vloc;
        kernels::launch_qwen36_gdn_ar(s.lin_q + q0 * lhd, s.lin_k + q0 * lhd, s.lin_v + v0 * lhd,
                                      s.vr_rec_a + ab, s.vr_rec_b + ab,
                                      static_cast<const bf16*>(w.ssm_dt) + v0,
                                      static_cast<const bf16*>(w.ssm_a) + v0,
                                      row_lin[r], state_off, out,
                                      c.gdn_qh_block ? vloc / g : vloc, vloc,
                                      lhd, c.gdn_qh_block, st, /*state_b16=*/false);
    };

    // (dual-GPU C1c) Multi-session rows advance independent sequences, so the GDN conv and
    // recurrence run as ONE batched launch per layer (each row on its own session's state through
    // a pointer array) instead of n launches each -- the per-row launches were the step's floor
    // once the projections went to the tensor cores (~7,400 launches at 32 rows). Same kernels
    // the tp=1 packed decode uses; their batch==1 arithmetic is the single-row kernels'.
    // SPARKINFER_TP_ROWS_GDN_BATCHED=0 keeps the per-row launches (A/B).
    static const bool mb_env = [] {
        const char* e = getenv("SPARKINFER_TP_ROWS_GDN_BATCHED"); return !(e && e[0] == '0');
    }();
    bool mb = multi && mb_env && (seg ? S > 1 : n > 1) && c.linear_head_dim == 128;
    if (ms) mb = true;   // the pointer arrays below
    const int lvdim_full = c.linear_v_heads * lhd;
    if (mb && !s.vr_mb_ready) {
        bool ok = cudaMalloc(&s.vr_mb_ptrs, (size_t)2 * R * sizeof(void*)) == cudaSuccess &&
                  cudaHostAlloc(&s.h_vr_mb_ptrs, (size_t)2 * R * sizeof(void*),
                                cudaHostAllocDefault) == cudaSuccess &&
                  cudaMalloc(&s.vr_mb_q, (size_t)R * lqdim * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_mb_k, (size_t)R * lqdim * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_mb_v, (size_t)R * lvdim_full * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_mb_a, (size_t)R * c.linear_v_heads * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_mb_b, (size_t)R * c.linear_v_heads * sizeof(bf16)) == cudaSuccess &&
                  cudaMalloc(&s.vr_mb_o, (size_t)R * lvdim_full * sizeof(bf16)) == cudaSuccess;
        // The other rank's windows of the packed qkv rows are never written by scatter_qkv; the
        // batched conv skips them (windowed), but keep the bytes defined anyway.
        if (ok) ok = cudaMemsetAsync(s.vr_full, 0, (size_t)R * lqkv * sizeof(bf16), st) == cudaSuccess;
        ok = tp_prefill_agree_min(ok ? 1 : 0) != 0;
        s.vr_mb_ready = ok;
    }
    if (mb && !s.vr_mb_ready) mb = false;
    if (mb) {
        for (int r = 0; r < n; r++) {
            s.h_vr_mb_ptrs[r] = row_conv[r];
            s.h_vr_mb_ptrs[R + r] = row_lin[r];
        }
        cu(cudaMemcpyAsync(s.vr_mb_ptrs, s.h_vr_mb_ptrs, (size_t)2 * R * sizeof(void*),
                           cudaMemcpyHostToDevice, st), "tp rows gdn ptrs");
    }
    // The batched conv + recurrence of layer L over `steps` row groups (all rows at once when
    // unsegmented): group t is rows [t * G, (t + 1) * G), G = S (segmented) or n, launched in
    // order so a session's step t + 1 reads the state its step t wrote. The gathered alpha/beta
    // (vr_mb_a/b) and the full-width q|k|v rows (vr_full) must be in place for all rows.
    auto gdn_batched_steps = [&](int L, int steps) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        const int G = seg ? S : n;
        const size_t conv_off = (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
        const size_t state_off = (size_t)gdn_state_slot(c, L) * vloc * lhd * lhd;
        for (int t = 0; t < steps; t++) {
            const size_t r0 = (size_t)t * G;
            kernels::launch_qwen36_conv_split_l2norm_fused_batched(
                s.vr_full + r0 * lqkv, w.ssm_conv, s.vr_mb_ptrs + r0, conv_off,
                s.vr_mb_q + r0 * lqdim, s.vr_mb_k + r0 * lqdim, s.vr_mb_v + r0 * lvdim_full,
                G, c.linear_q_heads, c.linear_v_heads, lhd, c.linear_conv_kernel, c.rms_eps, st,
                q0, ql, q0, ql, v0, vloc);
            if (!kernels::launch_qwen36_gdn_ar_batched(
                    s.vr_mb_q + r0 * lqdim, s.vr_mb_k + r0 * lqdim, s.vr_mb_v + r0 * lvdim_full,
                    s.vr_mb_a + r0 * c.linear_v_heads, s.vr_mb_b + r0 * c.linear_v_heads,
                    w.ssm_dt, w.ssm_a, reinterpret_cast<float* const*>(s.vr_mb_ptrs + R + r0),
                    state_off, s.vr_mb_o + r0 * lvdim_full, G, c.linear_q_heads, c.linear_v_heads,
                    lhd, c.gdn_qh_block, st, /*state_compact_b16=*/false, v0, vloc)) {
                cu(cudaErrorInvalidValue, "tp rows gdn batched");   // head_dim checked above
            }
        }
    };

    // Attention for all rows at once (TC mode only: the flash-decode split layout follows the
    // batch's longest row, so a row's partial sums are not the single-row call's).
    // SPARKINFER_TP_ROWS_ATTN_BATCHED=0 keeps the per-row launches.
    static const bool ma_env = [] {
        const char* e = getenv("SPARKINFER_TP_ROWS_ATTN_BATCHED"); return !(e && e[0] == '0');
    }();
    bool ma = tc && (mb || !multi || seg) && ma_env;
    auto ms_full = [&](int L) { return s.vr_ms_full + (size_t)lin_ord[L] * R * lqkv; };
    auto ms_q = [&](int L) { return s.vr_ms_q + (size_t)lin_ord[L] * R * lqdim; };
    auto ms_k = [&](int L) { return s.vr_ms_k + (size_t)lin_ord[L] * R * lqdim; };
    auto ms_v = [&](int L) { return s.vr_ms_v + (size_t)lin_ord[L] * R * lvdim_full; };
    auto ms_a = [&](int L) { return s.vr_ms_a + (size_t)lin_ord[L] * R * c.linear_v_heads; };
    auto ms_b = [&](int L) { return s.vr_ms_b + (size_t)lin_ord[L] * R * c.linear_v_heads; };
    // One multi-step conv + recurrence pass of layer L over all S sessions (forward: nsteps
    // null, T steps, outputs, state untouched; commit: nsteps = accepted steps, state written).
    auto ms_pass = [&](int L, const int* nsteps, bool commit) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        const size_t conv_off = (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
        const size_t state_off = (size_t)gdn_state_slot(c, L) * vloc * lhd * lhd;
        kernels::launch_qwen36_conv_split_l2norm_steps(
            ms_full(L), w.ssm_conv, s.vr_mb_ptrs, conv_off, ms_q(L), ms_k(L), ms_v(L), S, T,
            nsteps, commit, !commit, c.linear_q_heads, c.linear_v_heads, lhd,
            c.linear_conv_kernel, c.rms_eps, st, q0, ql, q0, ql, v0, vloc);
        if (!kernels::launch_qwen36_gdn_ar_steps(
                ms_q(L), ms_k(L), ms_v(L), ms_a(L), ms_b(L), w.ssm_dt, w.ssm_a,
                reinterpret_cast<float* const*>(s.vr_mb_ptrs + R), state_off, s.vr_mb_o, S, T,
                nsteps, commit, !commit, c.linear_q_heads, c.linear_v_heads, lhd, c.gdn_qh_block,
                st, v0, vloc))
            cu(cudaErrorInvalidValue, "tp seg gdn steps");
    };
    const int mbs = s.kv->max_blocks_per_seq();
    if (ma && s.vr_ma_state == 0) {
        const size_t fa = (size_t)R * n_q * Impl::MAX_NSPLITS;
        bool ok = cudaMalloc(&s.vr_ma_tptr, (size_t)2 * R * sizeof(int*)) == cudaSuccess &&
                  cudaHostAlloc(&s.h_vr_ma_tptr, (size_t)2 * R * sizeof(int*),
                                cudaHostAllocDefault) == cudaSuccess &&
                  cudaMalloc(&s.vr_ma_tab, (size_t)R * mbs * sizeof(int)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ma_tab_win, (size_t)R * mbs * sizeof(int)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ma_pairs, (size_t)2 * R * sizeof(int)) == cudaSuccess &&
                  cudaHostAlloc(&s.h_vr_ma_pairs, (size_t)2 * R * sizeof(int),
                                cudaHostAllocDefault) == cudaSuccess &&
                  cudaMalloc(&s.vr_ma_grp, (size_t)8 * R * sizeof(int)) == cudaSuccess &&
                  cudaHostAlloc(&s.h_vr_ma_grp, (size_t)8 * R * sizeof(int),
                                cudaHostAllocDefault) == cudaSuccess &&
                  cudaMalloc(&s.vr_ma_m, fa * sizeof(float)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ma_l, fa * sizeof(float)) == cudaSuccess &&
                  cudaMalloc(&s.vr_ma_acc, fa * HD * sizeof(float)) == cudaSuccess;
        ok = tp_prefill_agree_min(ok ? 1 : 0) != 0;
        s.vr_ma_state = ok ? 1 : -1;
    }
    if (ma && s.vr_ma_state != 1) ma = false;
    int ma_maxlen = 0, ma_pairs = 0, ma_groups = 0, ma_grp_rows = 0;
    if (ma) {
        for (int r = 0; r < n; r++) {
            s.h_vr_ma_tptr[r] = s.kv->block_table(row_kv[r]);
            s.h_vr_ma_tptr[R + r] = s.kv->block_table_win(row_kv[r]);
            ma_maxlen = std::max(ma_maxlen, pos_of(r) + 1);
        }
        cu(cudaMemcpyAsync(s.vr_ma_tptr, s.h_vr_ma_tptr, (size_t)2 * R * sizeof(int*),
                           cudaMemcpyHostToDevice, st), "tp rows attn tables");
        dflash_kernels::launch_gather_rows_i32(s.vr_ma_tptr, s.vr_ma_tab, mbs, n, st);
        dflash_kernels::launch_gather_rows_i32(s.vr_ma_tptr + R, s.vr_ma_tab_win, mbs, n, st);
        // (plan 07, B1) Consecutive rows of one session in pairs, for the long-context attention
        // that reads the KV once per pair (launch_flash_decode_split_pairs).
        bool paired[kTpVerifyRows] = {};
        for (int r = 0; r < n; r++) {
            if (paired[r]) continue;
            paired[r] = true;
            int o = -1;
            for (int r2 = r + 1; r2 < n; r2++)
                if (!paired[r2] && row_kv[r2] == row_kv[r]) { o = r2; break; }
            if (o >= 0) paired[o] = true;
            s.h_vr_ma_pairs[2 * ma_pairs] = r;
            s.h_vr_ma_pairs[2 * ma_pairs + 1] = o;
            ma_pairs++;
        }
        if (ma_pairs < n)
            cu(cudaMemcpyAsync(s.vr_ma_pairs, s.h_vr_ma_pairs, (size_t)2 * ma_pairs * sizeof(int),
                               cudaMemcpyHostToDevice, st), "tp rows attn pairs");
        else
            ma_pairs = 0;
        // (plan 10) nvfp4: the rows of one session with one split size in groups of up to 8,
        // for launch_flash_decode_split_rows_nvfp4 (one K/V read a group). Only when every row
        // takes the tensor-core kernel on its own (more than 512 keys and a split of 32+), which is
        // what ordinary decode runs for it; otherwise the rows keep the per-row path.
        static const int kRowsGroupMax = [] {
            const char* e = getenv("SPARKINFER_FA_ROWS_GROUP");
            return (e && atoi(e) == 4) ? 4 : 8;
        }();
        if (s.kv->kv_dtype() == 3) {
            bool all_mma = true;
            for (int r = 0; r < n && all_mma; r++) {
                const int len = pos_of(r) + 1;
                all_mma = len > 512 && (len + s.n_splits - 1) / s.n_splits >= 32;
            }
            int largest = 0;
            bool grouped[kTpVerifyRows] = {};
            std::vector<std::vector<int>> gl;
            for (int r = 0; r < n && all_mma; r++) {
                if (grouped[r]) continue;
                const int chunk = (pos_of(r) + 1 + s.n_splits - 1) / s.n_splits;
                std::vector<int> cur;
                for (int r2 = r; r2 < n; r2++) {
                    if (grouped[r2] || row_kv[r2] != row_kv[r] ||
                        (pos_of(r2) + 1 + s.n_splits - 1) / s.n_splits != chunk)
                        continue;
                    grouped[r2] = true;
                    cur.push_back(r2);
                    if ((int)cur.size() == kRowsGroupMax) { gl.push_back(cur); cur.clear(); }
                }
                if (!cur.empty()) gl.push_back(cur);
            }
            for (const auto& gr : gl) largest = std::max(largest, (int)gr.size());
            if (all_mma && !gl.empty()) {
                ma_grp_rows = largest > 4 ? 8 : 4;
                for (size_t gi = 0; gi < gl.size(); gi++)
                    for (int i = 0; i < ma_grp_rows; i++)
                        s.h_vr_ma_grp[gi * ma_grp_rows + i] = i < (int)gl[gi].size() ? gl[gi][i] : -1;
                ma_groups = (int)gl.size();
                cu(cudaMemcpyAsync(s.vr_ma_grp, s.h_vr_ma_grp,
                                   (size_t)ma_groups * ma_grp_rows * sizeof(int),
                                   cudaMemcpyHostToDevice, st), "tp rows attn groups");
            }
        }
    }

    // DSpark capture (end of each layer below): the leader's, or with a split capture
    // (set_dflash_capture_split) each rank's own columns, rank 1 into the twin of the leader's
    // destination the caller passed.
    const bool cap_here = s.dflash_n_cap > 0 && (s.tp_rank == 0 || s.dflash_cap_h > 0);
    const int ch = s.cap_h();
    Qwen35Model* cap_lead = s.tp_rank != 0 ? s.cap_lead : nullptr;
    auto cap_dst = [&](void* p) -> void* {
        if (!p || s.tp_rank == 0) return p;
        void* q = cap_lead ? const_cast<void*>(cap_lead->dflash_cap_peer(p)) : nullptr;
        static bool warned = false;
        if (!q && !warned) {
            warned = true;
            fprintf(stderr, "[tp] split capture: no rank-1 twin for a verify capture row\n");
        }
        return q;
    };

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        bool mix_split = false;   // the mixer projection went out as two split all-reduces (D2)
        if (is_linear_layer(c, L)) {
            bf16* rec = s.vr_rec_qkv + (size_t)L * R * wq;
            if (!proj_tc(w.gdn_qkv_fp4, w.gdn_qkv_fp4_sf, w.gdn_qkv_fp4_alpha, s.vr_xn, H, rec, wq))
                proj_rows(w.wqkv, w.wqkv_type, s.vr_xn, H, rec, wq);
            if (!proj_tc(w.gdn_z_fp4, w.gdn_z_fp4_sf, w.gdn_z_fp4_alpha, s.vr_xn, H, s.vr_z, Kw))
                proj_rows(w.wqkv_gate, w.wqkv_gate_type, s.vr_xn, H, s.vr_z, Kw);
            bf16* rec_a = s.vr_rec_a + (size_t)L * R * vloc;
            bf16* rec_b = s.vr_rec_b + (size_t)L * R * vloc;
            if (tc && w.ssm_alpha_type == 0 && w.ssm_beta_type == 0) {
                // One launch for both small bf16 projections over all rows (TC mode only: the
                // rows kernel is not the single-row GEMV's reduction order).
                dflash_kernels::launch_gemv_rows_exact_fused2(s.vr_xn, w.ssm_alpha, w.ssm_beta,
                                                              rec_a, rec_b, n, vloc, vloc, H, st);
            } else {
                proj_rows(w.ssm_alpha, w.ssm_alpha_type, s.vr_xn, H, rec_a, vloc);
                proj_rows(w.ssm_beta, w.ssm_beta_type, s.vr_xn, H, rec_b, vloc);
            }
            if (ms) {
                scatter_qkv(rec, n, ms_full(L));
                kernels::launch_gather_rows(ms_a(L) + v0, c.linear_v_heads, rec_a, vloc, vloc, n, st);
                kernels::launch_gather_rows(ms_b(L) + v0, c.linear_v_heads, rec_b, vloc, vloc, n, st);
                ms_pass(L, nullptr, false);
                kernels::launch_gather_rows(s.vr_gdn, Kw, s.vr_mb_o + (size_t)v0 * lhd, lvdim_full,
                                            Kw, n, st);
            } else if (mb) {
                scatter_qkv(rec, n);
                kernels::launch_gather_rows(s.vr_mb_a + v0, c.linear_v_heads, rec_a, vloc, vloc, n, st);
                kernels::launch_gather_rows(s.vr_mb_b + v0, c.linear_v_heads, rec_b, vloc, vloc, n, st);
                gdn_batched_steps(L, T);
                kernels::launch_gather_rows(s.vr_gdn, Kw, s.vr_mb_o + (size_t)v0 * lhd, lvdim_full,
                                            Kw, n, st);
            } else {
                scatter_qkv(rec, n);
                for (int r = 0; r < n; r++) gdn_step(L, r, s.vr_gdn + (size_t)r * Kw);
            }
            kernels::launch_qwen36_gated_norm(s.vr_gdn, s.vr_z, w.ssm_norm, s.vr_ln, n * vloc, lhd,
                                              c.rms_eps, st);
            mix_split = proj_tc_split(w.gdn_out_fp4, w.gdn_out_fp4_sf, w.gdn_out_fp4_alpha, s.vr_ln, Kw);
            if (!mix_split &&
                !proj_tc(w.gdn_out_fp4, w.gdn_out_fp4_sf, w.gdn_out_fp4_alpha, s.vr_ln, Kw, s.vr_ar, H))
                proj_rows(w.ssm_out, w.ssm_out_type, s.vr_ln, Kw, s.vr_ar, H);
        } else {
            if (w.q_has_gate) {
                if (!proj_tc(w.wq_fp4, w.wq_fp4_sf, w.wq_fp4_alpha, s.vr_xn, H, s.vr_qraw, 2 * qdim_l))
                    proj_rows(w.wq, w.wq_type, s.vr_xn, H, s.vr_qraw, 2 * qdim_l);
                kernels::launch_qwen36_split_q_gate(s.vr_qraw, s.vr_q, s.vr_g, n * n_q, HD, st);
            } else {
                proj_rows(w.wq, w.wq_type, s.vr_xn, H, s.vr_q, qdim_l);
            }
            if (!proj_tc(w.wk_fp4, w.wk_fp4_sf, w.wk_fp4_alpha, s.vr_xn, H, s.vr_k, kvdim_l))
                proj_rows(w.wk, w.wk_type, s.vr_xn, H, s.vr_k, kvdim_l);
            if (!proj_tc(w.wv_fp4, w.wv_fp4_sf, w.wv_fp4_alpha, s.vr_xn, H, s.vr_v, kvdim_l))
                proj_rows(w.wv, w.wv_type, s.vr_xn, H, s.vr_v, kvdim_l);
            if (s.use_qkfuse)
                kernels::launch_rmsnorm_qk(s.vr_q, s.vr_k, w.q_norm, w.k_norm, n * n_q, n * n_kv, HD,
                                           c.rms_eps, st);
            else {
                kernels::launch_rmsnorm(s.vr_q, w.q_norm, s.vr_q, n * n_q, HD, c.rms_eps, st);
                kernels::launch_rmsnorm(s.vr_k, w.k_norm, s.vr_k, n * n_kv, HD, c.rms_eps, st);
            }
            const int kvf = s.kv->kv_dtype();   // KVDtype: 0 bf16, 1 int8, 2 fp8, 3 nvfp4
            const bool kv8 = kvf != 0;   // quantized pool (scale pools present)
            void* kpool = (char*)s.kv->k_pool() + s.kv->kv_bytes(s.kv->layer_base_elems(L));
            void* vpool = (char*)s.kv->v_pool() + s.kv->kv_bytes(s.kv->layer_base_elems(L));
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + s.kv->scale_layer_base_elems(L) * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + s.kv->scale_layer_base_elems(L) * 2 : nullptr;
            auto ltab_of = [&](int r) {
                return w.swa ? s.kv->block_table_win(row_kv[r]) : s.kv->block_table(row_kv[r]);
            };
            if (ma) {
                // One call per kernel for all rows: they index the packed per-row tables.
                int* tab = w.swa ? s.vr_ma_tab_win : s.vr_ma_tab;
                if (kv8)
                    kernels::launch_rope_kv_append_partial_int8(s.vr_q, s.vr_k, s.vr_v, kpool, vpool,
                                                                kscale, vscale, tab, s.vr_pos, n, n_q,
                                                                n_kv, HD, c.rope_dim, c.rope_theta,
                                                                s.kv->block_size(), mbs, st, kvf);
                else
                    kernels::launch_rope_kv_append_partial(s.vr_q, s.vr_k, s.vr_v, (bf16*)kpool,
                                                           (bf16*)vpool, tab, s.vr_pos, n, n_q, n_kv,
                                                           HD, c.rope_dim, c.rope_theta,
                                                           s.kv->block_size(), mbs, st);
                if (!(ma_groups > 0 && !w.swa &&
                      kernels::launch_flash_decode_split_rows_nvfp4(
                          s.vr_q, kpool, vpool, tab, s.vr_seq, s.vr_ma_grp, ma_groups, ma_grp_rows,
                          s.vr_attn, s.vr_ma_m, s.vr_ma_l, s.vr_ma_acc, n, n_q, n_kv, HD,
                          s.kv->block_size(), s.n_splits, mbs, 1.f / sqrtf((float)HD), st, kscale,
                          vscale)) &&
                    !(ma_pairs > 0 && !w.swa &&
                      kernels::launch_flash_decode_split_pairs(
                          s.vr_q, kpool, vpool, tab, s.vr_seq, s.vr_ma_pairs, ma_pairs, s.vr_attn,
                          s.vr_ma_m, s.vr_ma_l, s.vr_ma_acc, n, n_q, n_kv, HD, s.kv->block_size(),
                          mbs, s.n_splits, 1.f / sqrtf((float)HD), st, ma_maxlen, kscale, vscale,
                          kvf)))
                    kernels::launch_flash_decode_split(s.vr_q, kpool, vpool, tab, s.vr_seq, s.vr_attn,
                                                       s.vr_ma_m, s.vr_ma_l, s.vr_ma_acc, n, n_q, n_kv,
                                                       HD, s.kv->block_size(), mbs, s.n_splits,
                                                       1.f / sqrtf((float)HD), st, nullptr, ma_maxlen,
                                                       kscale, vscale, kvf, nullptr, 0);
            } else {
            // Row by row, as the decode body appends: these kernels index the block table PER
            // TOKEN (block_table[tok * max_blocks + blk], the packed-sequence layout), so a
            // multi-token call on one sequence's table would put rows 1.. in other tables' blocks.
            for (int r = 0; r < n; r++) {
                bf16* qr = s.vr_q + (size_t)r * qdim_l;
                const bf16* kr = s.vr_k + (size_t)r * kvdim_l;
                const bf16* vrw = s.vr_v + (size_t)r * kvdim_l;
                if (kv8)
                    kernels::launch_rope_kv_append_partial_int8(qr, kr, vrw, kpool, vpool, kscale, vscale,
                                                                ltab_of(r), s.vr_pos + r, 1, n_q, n_kv, HD,
                                                                c.rope_dim, c.rope_theta, s.kv->block_size(),
                                                                s.kv->max_blocks_per_seq(), st, kvf);
                else
                    kernels::launch_rope_kv_append_partial(qr, kr, vrw, (bf16*)kpool, (bf16*)vpool,
                                                           ltab_of(r), s.vr_pos + r, 1, n_q, n_kv, HD,
                                                           c.rope_dim, c.rope_theta, s.kv->block_size(),
                                                           s.kv->max_blocks_per_seq(), st);
            }
            for (int r = 0; r < n; r++)
                kernels::launch_flash_decode_split(s.vr_q + (size_t)r * qdim_l, kpool, vpool, ltab_of(r),
                                                   s.vr_seq + r, s.vr_attn + (size_t)r * qdim_l,
                                                   s.fa_m, s.fa_l, s.fa_acc, 1, n_q, n_kv, HD,
                                                   s.kv->block_size(), s.kv->max_blocks_per_seq(),
                                                   s.n_splits, 1.f / sqrtf((float)HD), st,
                                                   nullptr, pos_of(r) + 1,
                                                   kscale, vscale, kvf, nullptr, 0);
            }
            if (w.q_has_gate)
                kernels::launch_qwen36_mul_sigmoid(s.vr_attn, s.vr_g, n * qdim_l, st);
            mix_split = proj_tc_split(w.wo_fp4, w.wo_fp4_sf, w.wo_fp4_alpha, s.vr_attn, qdim_l);
            if (!mix_split &&
                !proj_tc(w.wo_fp4, w.wo_fp4_sf, w.wo_fp4_alpha, s.vr_attn, qdim_l, s.vr_ar, H))
                proj_rows(w.wo, w.wo_type, s.vr_attn, qdim_l, s.vr_ar, H);
        }
        // AR-A + tail1, as forward_token_tp.
        if (mix_split) {
            kernels::launch_add_rmsnorm2_split(s.vr_x, s.vr_ar, s.vr_ar + (size_t)mp * hh, hh,
                                               w.post_attn_norm, s.vr_h, s.vr_hn, n, H, c.rms_eps, st);
        } else {
            tp_prefill_allreduce_bf16(s.vr_ar, (size_t)n * H);
            kernels::launch_add_rmsnorm2(s.vr_x, s.vr_ar, w.post_attn_norm, s.vr_h, s.vr_hn, n, H,
                                         c.rms_eps, st);
        }
        // FFN (dense NVFP4, the decode arm with M = n), AR-B, tail2.
        bf16* fg = s.vr_ffn;
        bf16* fu = s.vr_ffn + (size_t)R * fl;
        bf16* fh = s.vr_ffn + (size_t)2 * R * fl;
        const bool ffn_tc_front = tc && w.gate_fp4 && w.gate_fp4_sf && w.up_fp4 && w.up_fp4_sf &&
            w.down_fp4 && w.down_fp4_sf &&
            kernels::launch_prefill_nvfp4_quant_a(s.vr_hn, s.vr_tc_a, s.vr_tc_as, mp, H, st) &&
            kernels::launch_prefill_nvfp4_gemm(s.vr_tc_a, s.vr_tc_as, w.gate_fp4, w.gate_fp4_sf,
                                               fg, mp, fl, H, s.vr_tc_ws, st, w.gate_fp4_alpha) &&
            kernels::launch_prefill_nvfp4_gemm(s.vr_tc_a, s.vr_tc_as, w.up_fp4, w.up_fp4_sf,
                                               fu, mp, fl, H, s.vr_tc_ws, st, w.up_fp4_alpha) &&
            kernels::launch_prefill_nvfp4_swiglu_quant_a(fg, fu, s.vr_tc_a, s.vr_tc_as, mp, fl, st);
        const bool down_split = ffn_tc_front && split_gemm_ar(w.down_fp4, w.down_fp4_sf,
                                                             w.down_fp4_alpha, fl);
        const bool ffn_tc = ffn_tc_front &&
            (down_split ||
             kernels::launch_prefill_nvfp4_gemm(s.vr_tc_a, s.vr_tc_as, w.down_fp4, w.down_fp4_sf,
                                                s.vr_ar, mp, H, fl, s.vr_tc_ws, st, w.down_fp4_alpha));
        if (ffn_tc) {
        } else if (kernels::qwen38_nvfp4_dp4a()) {
            kernels::launch_gemv_nvfp4_quant_x(s.vr_hn, s.vr_nq, s.vr_ns, n, H, st);
            if (!kernels::launch_gemv_nvfp4_rows_dp4a2(s.vr_nq, s.vr_ns, w.gate_nv, w.up_nv,
                                                        fg, fu, n, fl, H, st)) {
                kernels::launch_gemv_nvfp4_rows_dp4a(s.vr_nq, s.vr_ns, w.gate_nv, fg, n, fl, H, st);
                kernels::launch_gemv_nvfp4_rows_dp4a(s.vr_nq, s.vr_ns, w.up_nv, fu, n, fl, H, st);
            }
            if (!kernels::launch_prefill_swiglu_nvfp4(fg, fu, fh, s.vr_nq, s.vr_ns, (long)n * fl, st)) {
                kernels::launch_prefill_swiglu(fg, fu, fh, (long)n * fl, st);
                kernels::launch_gemv_nvfp4_quant_x(fh, s.vr_nq, s.vr_ns, n, fl, st);
            }
            if (!kernels::launch_gemv_nvfp4_rows_dp4a(s.vr_nq, s.vr_ns, w.down_nv, s.vr_ar, n, H, fl, st))
                for (int r = 0; r < n; r++)
                    kernels::launch_gemv_nvfp4(fh + (size_t)r * fl, w.down_nv, s.vr_ar + (size_t)r * H,
                                               H, fl, st);
        } else {
            for (int r = 0; r < n; r++) {
                const bf16* x = s.vr_hn + (size_t)r * H;
                bf16 *gr = fg + (size_t)r * fl, *ur = fu + (size_t)r * fl, *hr = fh + (size_t)r * fl;
                kernels::launch_gemv_nvfp4(x, w.gate_nv, gr, fl, H, st);
                kernels::launch_gemv_nvfp4(x, w.up_nv, ur, fl, H, st);
                kernels::launch_prefill_swiglu(gr, ur, hr, fl, st);
                kernels::launch_gemv_nvfp4(hr, w.down_nv, s.vr_ar + (size_t)r * H, H, fl, st);
            }
        }
        const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        if (down_split) {
            kernels::launch_add_rmsnorm2_split(s.vr_h, s.vr_ar, s.vr_ar + (size_t)mp * hh, hh, nextnorm,
                                               s.vr_x, s.vr_xn, n, H, c.rms_eps, st);
        } else {
            tp_prefill_allreduce_bf16(s.vr_ar, (size_t)n * H);
            kernels::launch_add_rmsnorm2(s.vr_h, s.vr_ar, nextnorm, s.vr_x, s.vr_xn, n, H, c.rms_eps, st);
        }
        // DSpark capture of this layer's output rows: on the leader, or on both ranks with a split
        // capture (each its own columns, into its own twin of the leader's destination).
        if (cap_here && seg && seg_capture)
            for (int slot = 0; slot < s.dflash_n_cap; slot++)
                if (s.dflash_layer_ids[slot] == L)
                    for (int j = 0; j < S; j++)
                        if (void* dst = cap_dst(seg_capture[j]))
                            cu(cudaMemcpy2DAsync(static_cast<bf16*>(dst) + (size_t)slot * ch,
                                                 (size_t)s.dflash_n_cap * ch * sizeof(bf16),
                                                 s.vr_x + (size_t)j * H + s.dflash_cap_off,
                                                 (size_t)S * H * sizeof(bf16),
                                                 (size_t)ch * sizeof(bf16), T,
                                                 cudaMemcpyDeviceToDevice, st), "tp seg capture");
        if (cap_here && !multi && capture_dst)
            if (void* dst = cap_dst(capture_dst))
                for (int slot = 0; slot < s.dflash_n_cap; slot++)
                    if (s.dflash_layer_ids[slot] == L)
                        cu(cudaMemcpy2DAsync(static_cast<bf16*>(dst) + (size_t)slot * ch,
                                             (size_t)s.dflash_n_cap * ch * sizeof(bf16),
                                             s.vr_x + s.dflash_cap_off, (size_t)H * sizeof(bf16),
                                             (size_t)ch * sizeof(bf16), n,
                                             cudaMemcpyDeviceToDevice, st), "tp capture");
    }

    // Head: this rank's vocab half per row, then the decode epilogue's zero-padded [V] all-reduce
    // and argmax, row-batched.
    const bool head_q4k = s.use_pq && s.use_llama && s.w.lm_head_type == 12;
    bool head_done = false;
    if (tc) {
        // Both ranks must take the same head form: the FP4 head copy is free-VRAM dependent.
        const bool head_tc = tp_prefill_agree_min(s.w.lm_head_fp4 && s.w.lm_head_fp4_sf ? 1 : 0) != 0;
        head_done = head_tc &&
            kernels::launch_prefill_nvfp4_quant_a(s.vr_xn, s.vr_tc_a, s.vr_tc_as, mp, H, st) &&
            kernels::launch_prefill_nvfp4_gemm_f32(s.vr_tc_a, s.vr_tc_as, s.w.lm_head_fp4,
                                                   s.w.lm_head_fp4_sf, s.vr_lh, mp, Vr, H,
                                                   s.vr_tc_ws, st, s.w.lm_head_fp4_alpha);
    }
    if (!head_done && head_q4k) {
        kernels::launch_quantize_q8_1_rows(s.vr_xn, s.vr_q81, H, n, H, st);
        head_done = kernels::launch_mmvq_rows_f32(12, s.vr_q81, s.w.lm_head, s.vr_lh, n, Vr, H, st);
    }
    if (!head_done) {
        for (int r = 0; r < n; r++) {
            const bf16* x = s.vr_xn + (size_t)r * H;
            float* y = s.vr_lh + (size_t)r * Vr;
            char* q81 = s.vr_q81 + (size_t)r * s.vr_q81_row;
            if (head_q4k) {
                kernels::launch_quantize_q8_1_blocks(x, q81, H, st);
                kernels::launch_mmvq_q4k_f32(q81, s.w.lm_head, y, Vr, H, st);
            } else if (s.use_q6mmvq && s.w.lm_head_type == 14) {
                kernels::launch_quantize_q8_1_blocks(x, q81, H, st);
                kernels::launch_gemv_q6k_dp4a_f32(q81, s.w.lm_head, y, Vr, H, st);
            } else if (s.w.lm_head_type) {
                kernels::launch_gemv_q_f32(x, s.w.lm_head, s.w.lm_head_type, y, Vr, H, st);
            } else {
                kernels::launch_gemv_f32(x, s.w.lm_head, y, Vr, H, st);
            }
        }
    }
    // Constrained rows: this rank's half of each row's mask, applied as the engine's constraint
    // bias, so every draw below (argmax, sampled top-k) is the one ordinary constrained decode makes.
    if (masked) {
        const int Wr = Vr / 32;
        const size_t w0 = (size_t)s.tp_rank * Wr;
        for (int r = 0; r < n; r++) {
            uint32_t* dst = s.h_vr_mask + (size_t)r * Wr;
            if (row_mask[r]) std::memcpy(dst, row_mask[r] + w0, (size_t)Wr * sizeof(uint32_t));
            else std::memset(dst, 0xff, (size_t)Wr * sizeof(uint32_t));
        }
        cu(cudaMemcpyAsync(s.vr_mask, s.h_vr_mask, (size_t)n * Wr * sizeof(uint32_t),
                           cudaMemcpyHostToDevice, st), "tp verify row masks");
        dflash_kernels::launch_rows_mask_bits(s.vr_lh, n, Vr, s.vr_mask, st);
    }
    // Argmax per rank over its own vocab half, then exchange (value, index) per row on the host.
    // This used to zero-pad an [n][V] f32 row block, place the half, all-reduce all of it across
    // the link (8 MB at 8 rows, 32 MB at 32) and argmax the full rows -- the same index, by the
    // lowest-index rule, for ~1.2 ms of link time per verify. SPARKINFER_TP_HEAD_ALLREDUCE=1
    // restores it (A/B).
    static const bool head_ar = [] {
        const char* e = getenv("SPARKINFER_TP_HEAD_ALLREDUCE"); return e && e[0] == '1';
    }();
    // (plan 06, P) Sampled rows: each rank's exact top-k of its half, merged over both (identical
    // on both ranks), then the decode sampler's draw from those candidates -- the same kernel on
    // the same inputs on each rank, so both keep the same tokens. Greedy rows of the same verify
    // take the merged best, which is the argmax the exchange below would give.
    int smp_k = 0;
    if (row_sample)
        for (int r = 0; r < n; r++)
            if (row_sample[r].temperature > 0.f)
                smp_k = std::max(smp_k, std::min(row_sample[r].top_k, dflash_kernels::kRowsTopkMax));
    if (smp_k > 0 && s.h_vr_smp && n <= kTpVerifyRows) {
        const int K = dflash_kernels::kRowsTopkMax;
        dflash_kernels::launch_rows_topk(s.vr_lh, n, Vr, smp_k, s.vr_tk_v, s.vr_tk_i, st);
        float* h_v = reinterpret_cast<float*>(s.h_vr_smp);
        int* h_i = reinterpret_cast<int*>(h_v + (size_t)R * K);
        float* h_mv = reinterpret_cast<float*>(h_i + (size_t)R * K);
        int* h_mi = reinterpret_cast<int*>(h_mv + (size_t)R * K);
        char* h_par = reinterpret_cast<char*>(h_mi + (size_t)R * K);
        float* h_temp = reinterpret_cast<float*>(h_par);
        float* h_topp = h_temp + R;
        int* h_topk = reinterpret_cast<int*>(h_topp + R);
        int* h_out = h_topk + R;
        unsigned long long* h_seed = reinterpret_cast<unsigned long long*>(h_out + R);
        unsigned long long* h_step = h_seed + R;
        cu(cudaMemcpyAsync(h_v, s.vr_tk_v, (size_t)n * K * sizeof(float), cudaMemcpyDeviceToHost, st),
           "tp verify topk v");
        cu(cudaMemcpyAsync(h_i, s.vr_tk_i, (size_t)n * K * sizeof(int), cudaMemcpyDeviceToHost, st),
           "tp verify topk i");
        cu(cudaStreamSynchronize(st), "tp verify topk sync");
        tp_exchange_topk(h_v, h_i, n, Vr, h_mv, h_mi);
        for (int r = 0; r < n; r++) {
            const SpecSampleRow& p = row_sample[r];
            h_temp[r] = p.temperature;
            h_topk[r] = p.top_k;
            h_topp[r] = p.top_p;
            h_seed[r] = p.seed;
            h_step[r] = p.step;
        }
        cu(cudaMemcpyAsync(s.vr_mg_v, h_mv, (size_t)n * K * sizeof(float), cudaMemcpyHostToDevice, st),
           "tp verify cand v");
        cu(cudaMemcpyAsync(s.vr_mg_i, h_mi, (size_t)n * K * sizeof(int), cudaMemcpyHostToDevice, st),
           "tp verify cand i");
        cu(cudaMemcpyAsync(s.vr_smp_temp, h_temp, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, st), "smp temp");
        cu(cudaMemcpyAsync(s.vr_smp_topp, h_topp, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, st), "smp topp");
        cu(cudaMemcpyAsync(s.vr_smp_topk, h_topk, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, st), "smp topk");
        cu(cudaMemcpyAsync(s.vr_smp_seed, h_seed, (size_t)n * sizeof(unsigned long long),
                           cudaMemcpyHostToDevice, st), "smp seed");
        cu(cudaMemcpyAsync(s.vr_smp_step, h_step, (size_t)n * sizeof(unsigned long long),
                           cudaMemcpyHostToDevice, st), "smp step");
        dflash_kernels::launch_rows_sample_candidates(s.vr_mg_v, s.vr_mg_i, n, K, s.vr_smp_temp,
                                                      s.vr_smp_topk, s.vr_smp_topp, s.vr_smp_seed,
                                                      s.vr_smp_step, s.vr_smp_out, st);
        cu(cudaMemcpyAsync(h_out, s.vr_smp_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost, st),
           "smp out");
        cu(cudaStreamSynchronize(st), "tp verify sample sync");
        for (int r = 0; r < n; r++) out_argmax[r] = h_out[r];
        // Rejection sampling (rows carrying the draft's q, see SpecSampleRow): walk each session's
        // rows in order; row t accepts the token it checks (the next row's id) with probability
        // min(1, p/q), else draws from max(0, p - q) and the session stops there. p is this row's
        // sampling distribution as k_rows_sample_candidates draws it (top_k, the top_p prefix at
        // temperature 1, softmax(logit / temperature)); a proposal outside the walk's table (an
        // n-gram lookup) is a point mass, q = 1. A fully accepted session keeps the last row's
        // ordinary draw. Host-side, from data both ranks hold identically, with a counter-based
        // RNG of (seed, step): both ranks keep the same tokens.
        if (seg) {
            constexpr int QT = dflash_kernels::kDraftQTab;
            auto rng = [](unsigned long long seed, unsigned long long step, unsigned long long salt) {
                unsigned long long z = seed ^ (step * 0x9E3779B97F4A7C15ull) ^ (salt * 0xD1B54A32D192ED03ull);
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                z ^= z >> 31;
                return (double)(z >> 11) * (1.0 / 9007199254740992.0);   // [0, 1)
            };
            for (int j = 0; j < S; j++) {
                for (int t = 0; t + 1 < T; t++) {
                    const int r = t * S + j;
                    const SpecSampleRow& rs = row_sample[r];
                    if (!(rs.temperature > 0.f) || !rs.q_p || !rs.q_ids) break;
                    const float* v = h_mv + (size_t)r * K;
                    const int* ix = h_mi + (size_t)r * K;
                    int kk = rs.top_k;
                    if (kk <= 0 || kk > K) kk = K;
                    double cum[dflash_kernels::kRowsTopkMax];
                    double run = 0.0;
                    for (int i = 0; i < kk; i++) { run += std::exp((double)v[i] - v[0]); cum[i] = run; }
                    const bool p_active = rs.top_p >= 0.f && rs.top_p < 1.f;
                    int m = 0;
                    for (int i = 0; i < kk; i++) {
                        if (i > 0 && p_active && !(cum[i - 1] < rs.top_p * cum[kk - 1])) break;
                        if (ix[i] == INT_MAX || v[i] == -INFINITY) break;
                        m = i + 1;
                    }
                    if (m == 0) break;
                    double pw[dflash_kernels::kRowsTopkMax], z = 0.0;
                    for (int i = 0; i < m; i++) { pw[i] = std::exp(((double)v[i] - v[0]) / rs.temperature); z += pw[i]; }
                    for (int i = 0; i < m; i++) pw[i] /= z;
                    const int x = ids[(size_t)(t + 1) * S + j];
                    double px = 0.0, qx = -1.0;
                    for (int i = 0; i < m; i++) if (ix[i] == x) { px = pw[i]; break; }
                    for (int i = 0; i < QT && rs.q_ids[i] >= 0; i++)
                        if (rs.q_ids[i] == x) { qx = rs.q_p[i]; break; }
                    const bool point = !(qx > 0.0);   // not in the walk's table: an n-gram proposal
                    if (point) qx = 1.0;
                    if (rng(rs.seed, rs.step, 1) * qx < px) { out_argmax[r] = x; continue; }
                    // Rejected: draw from max(0, p - q) over this row's candidates.
                    double rw[dflash_kernels::kRowsTopkMax], rz = 0.0;
                    for (int i = 0; i < m; i++) {
                        double qv = 0.0;
                        if (point) qv = ix[i] == x ? 1.0 : 0.0;
                        else
                            for (int q = 0; q < QT && rs.q_ids[q] >= 0; q++)
                                if (rs.q_ids[q] == ix[i]) { qv = rs.q_p[q]; break; }
                        rw[i] = std::max(0.0, pw[i] - qv);
                        rz += rw[i];
                    }
                    if (!(rz > 0.0)) { for (int i = 0; i < m; i++) rw[i] = pw[i]; rz = 1.0; }
                    double u = rng(rs.seed, rs.step, 2) * rz;
                    int pick = ix[m - 1];
                    for (int i = 0; i < m; i++) {
                        if (rw[i] <= 0.0) continue;
                        if (u < rw[i]) { pick = ix[i]; break; }
                        u -= rw[i];
                        pick = ix[i];
                    }
                    out_argmax[r] = pick;
                    break;
                }
            }
        }
    } else if (!head_ar && n <= 64) {
        kernels::launch_argmax(s.vr_lh, s.vr_out, n, Vr, st);
        cu(cudaMemcpyAsync(s.h_vr + 3 * R, s.vr_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost, st),
           "tp verify local argmax");
        cu(cudaStreamSynchronize(st), "tp verify sync");
        float vals[64];
        int lidx[64];
        for (int r = 0; r < n; r++) {
            lidx[r] = s.h_vr[3 * R + r];
            cu(cudaMemcpyAsync(&s.h_vr_maxv[r], s.vr_lh + (size_t)r * Vr + lidx[r], sizeof(float),
                               cudaMemcpyDeviceToHost, st), "tp verify local max");
        }
        cu(cudaStreamSynchronize(st), "tp verify sync max");
        for (int r = 0; r < n; r++) vals[r] = s.h_vr_maxv[r];
        tp_exchange_argmax(vals, lidx, n, Vr, out_argmax);
    } else {
    cu(cudaMemsetAsync(s.vr_logits, 0, (size_t)n * V * sizeof(float), st), "tp verify logits zero");
    cu(cudaMemcpy2DAsync(s.vr_logits + (size_t)s.tp_rank * Vr, (size_t)V * sizeof(float),
                         s.vr_lh, (size_t)Vr * sizeof(float), (size_t)Vr * sizeof(float), n,
                         cudaMemcpyDeviceToDevice, st), "tp verify logits place");
    tp_prefill_allreduce_f32(s.vr_logits, (size_t)n * V);
    kernels::launch_argmax(s.vr_logits, s.vr_out, n, V, st);
    cu(cudaMemcpyAsync(s.h_vr + 3 * R, s.vr_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost, st),
       "tp verify argmax");
    cu(cudaStreamSynchronize(st), "tp verify sync");
    for (int r = 0; r < n; r++) out_argmax[r] = s.h_vr[3 * R + r];
    }

    if (seg) {
        // Session j's accepted prefix, as below; a partly accepted session is restored and its
        // kept steps replayed through the batched kernels, the other sessions' rows of a step
        // pointed at the (otherwise unused) single-verify snapshot as scratch.
        int max_keep = 0;
        bool any_partial = false;
        if (ms) {
            for (int j = 0; j < S; j++) {
                int k = 1;
                while (k < T && ids[(size_t)k * S + j] == out_argmax[(size_t)(k - 1) * S + j]) ++k;
                seg_keep[j] = k;
                s.h_vr_ms_keep[j] = k;
            }
            cu(cudaMemcpyAsync(s.vr_ms_keep, s.h_vr_ms_keep, (size_t)S * sizeof(int),
                               cudaMemcpyHostToDevice, st), "tp seg keep");
            for (int L = 0; L < c.n_layers; L++)
                if (is_linear_layer(c, L)) ms_pass(L, s.vr_ms_keep, true);
            cu(cudaStreamSynchronize(st), "tp seg commit");
            return n;
        }
        for (int j = 0; j < S; j++) {
            int k = 1;
            while (k < T && ids[(size_t)k * S + j] == out_argmax[(size_t)(k - 1) * S + j]) ++k;
            seg_keep[j] = k;
            if (k < T) {
                any_partial = true;
                max_keep = std::max(max_keep, k);
                cu(cudaMemcpyAsync(row_lin[j], s.vr_seg_snap_lin[j], ls * sizeof(float),
                                   cudaMemcpyDeviceToDevice, st), "tp seg restore lin");
                cu(cudaMemcpyAsync(row_conv[j], s.vr_seg_snap_conv[j], cs * sizeof(bf16),
                                   cudaMemcpyDeviceToDevice, st), "tp seg restore conv");
            }
        }
        if (any_partial) {
            if (mb) {
                for (int t = 0; t < max_keep; t++)
                    for (int j = 0; j < S; j++) {
                        const bool live = seg_keep[j] < T && t < seg_keep[j];
                        s.h_vr_mb_ptrs[t * S + j] = live ? (void*)row_conv[j] : (void*)s.vr_snap_conv;
                        s.h_vr_mb_ptrs[R + t * S + j] = live ? (void*)row_lin[j] : (void*)s.vr_snap_lin;
                    }
                cu(cudaMemcpyAsync(s.vr_mb_ptrs, s.h_vr_mb_ptrs, (size_t)2 * R * sizeof(void*),
                                   cudaMemcpyHostToDevice, st), "tp seg replay ptrs");
            }
            for (int L = 0; L < c.n_layers; L++) {
                if (!is_linear_layer(c, L)) continue;
                scatter_qkv(s.vr_rec_qkv + (size_t)L * R * wq, n);
                if (mb) {
                    kernels::launch_gather_rows(s.vr_mb_a + v0, c.linear_v_heads,
                                                s.vr_rec_a + (size_t)L * R * vloc, vloc, vloc, n, st);
                    kernels::launch_gather_rows(s.vr_mb_b + v0, c.linear_v_heads,
                                                s.vr_rec_b + (size_t)L * R * vloc, vloc, vloc, n, st);
                    gdn_batched_steps(L, max_keep);
                } else {
                    for (int t = 0; t < max_keep; t++)
                        for (int j = 0; j < S; j++)
                            if (seg_keep[j] < T && t < seg_keep[j]) gdn_step(L, t * S + j, s.vr_gdn);
                }
            }
        }
        cu(cudaStreamSynchronize(st), "tp seg commit");
        return n;
    }
    if (multi) return n;   // every row is one independent decode step: all kept
    // Accepted prefix: row 0 is always kept (it is the target's own next token); row r is kept
    // while the draft token at r matches the target's argmax at r-1.
    int keep = 1;
    while (keep < n && ids[keep] == out_argmax[keep - 1]) ++keep;
    if (keep < n) {
        cu(cudaMemcpyAsync(s.lin_state, s.vr_snap_lin, ls * sizeof(float), cudaMemcpyDeviceToDevice, st),
           "tp verify restore lin");
        cu(cudaMemcpyAsync(s.lin_conv_state, s.vr_snap_conv, cs * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "tp verify restore conv");
        for (int L = 0; L < c.n_layers; L++) {
            if (!is_linear_layer(c, L)) continue;
            scatter_qkv(s.vr_rec_qkv + (size_t)L * R * wq, keep);
            for (int r = 0; r < keep; r++) gdn_step(L, r, s.vr_gdn);
        }
        cu(cudaStreamSynchronize(st), "tp verify commit");
    }
    return keep;
}

bool Qwen35Model::decode_packed_tp(const int* tokens, const int* positions,
                                   const uint64_t* seq_ids, int n, int* out_sampled,
                                   const SpecSampleRow* row_sample) {
    if (!tokens || !positions || !seq_ids || !out_sampled) return false;
    if (n < 1 || n > kQwen35MaxPackedRows) return false;
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (!s.cfg.hybrid || !s.gguf) return false;

    // (dual-GPU WP-9, design item 9) The tp=2 packed decode is a per-row loop over the SAME
    // single-row windowed path the single-token op runs (forward_token_tp): N one-row tp ARs
    // instead of one packed N-row run. The rows-kernels bit-identity invariant (a packed row is
    // bit-identical to that row's one-step AR) makes this correctness-preserving; the packed-
    // kernel throughput a tp>1 split of the weights cannot use is the accepted perf regression --
    // the M2 gate is code-complete + in-regime unit test.
    //
    // Per-row bookkeeping: the single-row tp path keys its per-rank state off the model's ACTIVE
    // session (GDN/conv state pointers, the session's penalty/logit-bias buffers, and the
    // position read from THIS rank's own KV pool), not on per-row table pointers like the tp=1
    // packed kernels, so the leader activates each row's session before running it. The PEER
    // rank's active session is not touched here: it must be mirrored by the engine (S8 wires
    // model_engine to activate on both ranks) -- a caller driving this directly must keep the
    // peer's active session in step with the rows.
    //
    // Sampling: the engine only packs plain-greedy batches (temperature 0, no truncation, no
    // penalties, no logit bias -- see step_jobs_packed), and forward_token_tp at temperature 0
    // returns the exact argmax (its Gumbel step is a no-op at zero temperature), so each
    // out_sampled[i] is exactly what the tp=1 packed argmax would have emitted for that row.
    // No GDN b16 compaction happens on the tp path (the tp=1 packed compaction loop is
    // bypassed), so every session stays in the fp32 representation it was opened in: a mixed-
    // representation batch, the one hazard the tp=1 packed path declines for, cannot arise.
    for (int i = 0; i < n; i++)
        if (s.sessions.find(seq_ids[i]) == s.sessions.end()) return false;
    // (dual-GPU C1) Row-batched: all rows through one tp_rows_forward (the verify body with each
    // row on its own session), so every block's weights stream once for all rows and the
    // all-reduces cover all rows at once. The rows kernels are the single-row decode kernels' row
    // twins (the invariant DSpark's losslessness already rests on), so each row's argmax is the
    // token the per-row loop below would produce. Groups of kTpVerifyRows; a decline (b16 GDN
    // state, verify scratch unavailable -- decided identically on both ranks) keeps the loop.
    // SPARKINFER_TP_PACKED_ROWS=0 forces the loop (A/B).
    static const bool rows_on = [] {
        const char* e = getenv("SPARKINFER_TP_PACKED_ROWS");
        return !(e && e[0] == '0');
    }();
    if (rows_on && n > 1) {
        bool ok = true;
        for (int i0 = 0; i0 < n && ok; i0 += kTpVerifyRows) {
            const int m = std::min(kTpVerifyRows, n - i0);
            ok = tp_rows_forward(tokens + i0, m, 0, positions + i0, seq_ids + i0, nullptr,
                                 out_sampled + i0, 0, nullptr, nullptr,
                                 row_sample ? row_sample + i0 : nullptr) == m;
            // A decline happens before any state is touched, so the rows from i0 on can still
            // take the loop; rows already run must not run again.
            if (!ok) {
                for (int i = i0; i < n; i++) {
                    activate_session(seq_ids[i]);
                    const SpecSampleRow rs = row_sample ? row_sample[i] : SpecSampleRow{};
                    out_sampled[i] = forward_token_tp(tokens[i], positions[i], true, rs.temperature,
                                                      rs.seed, rs.step, rs.top_k, rs.top_p, 0.f, 0.f);
                }
                return true;
            }
        }
        return true;
    }
    for (int i = 0; i < n; i++) {
        activate_session(seq_ids[i]);
        out_sampled[i] = forward_token_tp(tokens[i], positions[i],
                                          /*sample=*/true, /*temperature=*/0.f,
                                          /*seed=*/0, /*sample_step=*/0,
                                          /*top_k=*/0, /*top_p=*/1.f,
                                          /*presence_penalty=*/0.f, /*frequency_penalty=*/0.f);
    }
    return true;
}

void Qwen35Model::tp_attn_layer_tp(int l, uint16_t* x, const uint16_t* xn) {
    (void)x;
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const Qwen35LayerWeights& w = s.w.layers[l];
    const int H = c.hidden;
    const int HD = c.head_dim;
    cudaStream_t st = s.stream;
    kernels::GemmConfig gc{};

    // This rank's full-attention window at R=2: rank r owns q heads [n_q/2*r, n_q/2*(r+1)) and
    // kv heads [n_kv/2*r, n_kv/2*(r+1)) -- the loader sliced the weight blobs along exactly it.
    // The rank's activations live in the LEADING window of the full-width s.q/s.k/s.v/s.qgate/
    // s.attn buffers: the kernels below take LOCAL head counts and are head-index-agnostic, so
    // they read exactly that window (27B: 12 q heads + 12 gate heads, 2 kv heads).
    const int n_q     = c.n_q_heads / 2;
    const int n_kv    = c.n_kv_heads / 2;
    const int qdim_l   = n_q * HD;    // the rank's q (and gate) window; the o_proj K window
    const int kvdim_l  = n_kv * HD;  // the rank's k/v window

    // Zero the AR-A staging: the o_proj K-windowed partial below lands in the first H of the
    // 2*H all-reduced row, the second H stays zero through the sum.
    cu(cudaMemsetAsync(s.tp_ar, 0, (size_t)2 * H * sizeof(bf16), st), "tp attn ar zero");

    // Only the hybrid full-attn shape is implemented here (27B: gated Q + partial rope; the KV
    // append follows the cache's int8/bf16 kind). Any other combination (a plain-rope hybrid, a
    // separate gate tensor) would corrupt this rank's pool with a wrong-shape append, so note
    // once and leave the row zero rather than guess.
    const bool partial_rope = (c.rope_dim > 0 && c.rope_dim < HD);
    if (!partial_rope || w.wgate != nullptr) {
        static bool attn_shape_note = false;
        if (!attn_shape_note) {
            attn_shape_note = true;
            fprintf(stderr, "[tp] tp_attn_layer_tp: unsupported attn shape (partial_rope=%d "
                            "sep_gate=%d); AR row left zero, layer output not valid\n",
                    (int)partial_rope, (int)(w.wgate != nullptr));
        }
        return;
    }

    // -- this rank's position/seq_len: the decode position forward_token_tp was called with
    // (identical on both ranks: the peer runs the mirrored call with the same arguments), exactly
    // the tp=1 body's h_scalars[2]/[3] (slot = position, seqlen = position + 1). NOT the pool's
    // allocated_tokens(): that is the sequence's block CAPACITY (blocks x block_size, the whole
    // request budget), so every token appended to one fixed slot at one fixed rotary position and
    // flash-decode read unwritten slots. Rotary == slot (mrope offset 0: the 27B is text-only).
    const int pos    = s.tp_cur_pos;
    const int seqlen = pos + 1;
    s.h_tp_pos[0] = pos;
    s.h_tp_pos[1] = seqlen;
    cu(cudaMemcpyAsync(s.tp_pos, s.h_tp_pos, 2 * sizeof(int), cudaMemcpyHostToDevice, st),
       "tp attn pos");

    // -- quantize xn once, shared by the q/k/v GEMVs (same dispatch as the tp=1 body and the
    // GDN tp body; the rank's compact blobs are full-K = H, so the full-K GEMVs read them right) --
    const bool any_q4k = (w.wq_type == 12 || w.wk_type == 12 || w.wv_type == 12);
    const bool any_q6k = (w.wq_type == 14 || w.wk_type == 14 || w.wv_type == 14);
    const bool any_q80 = (w.wq_type == 8 || w.wk_type == 8 || w.wv_type == 8);
    const bool any_nv  = (w.wq_type == kernels::SI_QTYPE_NVFP4 ||
                          w.wk_type == kernels::SI_QTYPE_NVFP4 ||
                          w.wv_type == kernels::SI_QTYPE_NVFP4);
    if (s.gguf && s.use_pq) {
        if (s.use_llama && (any_q4k || any_q80 || (s.use_q6mmvq && any_q6k)))
            kernels::launch_quantize_q8_1_blocks(xn, s.aq81, H, st);
        else if (any_q4k)
            kernels::launch_quantize_q8_1(xn, s.aq8, s.aq8_d, s.aq8_s, H, st);
    }
    const bool xn_nv = s.gguf && any_nv && kernels::qwen38_nvfp4_dp4a_proj();
    if (xn_nv)
        kernels::launch_gemv_nvfp4_quant_x(xn, s.nv_pq_a, s.nv_ps_a, 1, H, st);

    // -- N-window projections into the rank's compact [N'][K=H] blobs (plain GEMVs) --
    auto proj = [&](const void* W, int t, void* y, int N) {
        if (s.gguf) {
            if (s.use_pq && t == 12) {
                if (s.use_llama) kernels::launch_mmvq_q4k(s.aq81, W, y, N, H, st);
                else             kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, H, st);
            }
            else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14)
                kernels::launch_mmvq_q6k(s.aq81, W, y, N, H, st);
            else if (s.use_pq && s.use_llama && t == 8)
                kernels::launch_mmvq_q80(s.aq81, W, y, N, H, st);
            else if (t == kernels::SI_QTYPE_FP8)
                kernels::launch_gemv_fp8(xn, W, y, N, H, st);
            else if (t == kernels::SI_QTYPE_NVFP4) {
                if (!(xn_nv && kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_pq_a, s.nv_ps_a, W, y,
                                                                     1, N, H, st)))
                    kernels::launch_gemv_nvfp4(xn, W, y, N, H, st);
            }
            else if (t) kernels::launch_gemv_q(xn, W, t, y, N, H, st);
            else         kernels::launch_gemv(xn, W, y, N, H, st);
        } else {
            kernels::launch_gemm(xn, W, y, 1, N, H, 1.f, 0.f, gc, st);
        }
    };

    // Q: 27B fuses Q and the attn gate into ONE GEMV whose raw output interleaves per head as
    // [q(HD) | gate(HD)] -- head h owns raw[h*2*HD .. (h+1)*2*HD), q the first half, gate the
    // second (see split_q_gate). The rank's N window is 2*qdim_l of that 2*qdim tensor, so it
    // lands whole in tp_qraw and the split scatters the rank's head blocks into the leading
    // windows of s.q / s.qgate.
    if (w.q_has_gate) {
        proj(w.wq, w.wq_type, s.tp_qraw, 2 * qdim_l);
        kernels::launch_qwen36_split_q_gate(s.tp_qraw, s.q, s.qgate, n_q, HD, st);
    } else {
        proj(w.wq, w.wq_type, s.q, qdim_l);
    }
    proj(w.wk, w.wk_type, s.k, kvdim_l);
    proj(w.wv, w.wv_type, s.v, kvdim_l);

    // -- QK-norm: the [HD] weight is one shared vector, replicated over every head, so the local
    // window needs no shift --
    if (s.use_qkfuse)
        kernels::launch_rmsnorm_qk(s.q, s.k, w.q_norm, w.k_norm, n_q, n_kv, HD, c.rms_eps, st);
    else {
        kernels::launch_rmsnorm(s.q, w.q_norm, s.q, n_q, HD, c.rms_eps, st);
        kernels::launch_rmsnorm(s.k, w.k_norm, s.k, n_kv, HD, c.rms_eps, st);
    }

    // -- RoPE + KV-append into THIS rank's pool, which was built for exactly n_kv heads: pool
    // head j == model kv head kv_head_start + j, and this rank's local kv head j is that model
    // head, so the kernel's [(ctok*n_kv + j)*HD] indexing lands the token in this rank's rows.
    const int kvf = s.kv->kv_dtype();   // KVDtype: 0 bf16, 1 int8, 2 fp8, 3 nvfp4
    const bool kv8 = kvf != 0;   // quantized pool (scale pools present)
    void* kpool = (char*)s.kv->k_pool() + s.kv->kv_bytes(s.kv->layer_base_elems(l));
    void* vpool = (char*)s.kv->v_pool() + s.kv->kv_bytes(s.kv->layer_base_elems(l));
    void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + s.kv->scale_layer_base_elems(l) * 2 : nullptr;
    void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + s.kv->scale_layer_base_elems(l) * 2 : nullptr;
    int* ltab = w.swa ? s.kv->block_table_win(s.active_seq_id) : s.kv->block_table(s.active_seq_id);
    if (kv8)
        kernels::launch_rope_kv_append_partial_int8(s.q, s.k, s.v, kpool, vpool, kscale, vscale,
                                                    ltab, s.tp_pos, 1, n_q, n_kv,
                                                    HD, c.rope_dim, c.rope_theta,
                                                    s.kv->block_size(), s.kv->max_blocks_per_seq(), st, kvf);
    else
        kernels::launch_rope_kv_append_partial(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool,
                                               ltab, s.tp_pos, 1, n_q, n_kv,
                                               HD, c.rope_dim, c.rope_theta,
                                               s.kv->block_size(), s.kv->max_blocks_per_seq(), st);

    // -- flash decode over the rank's own pool: local head counts; n_splits stays at the model
    // value (the rank uses n_q of the 24-head fa_* scratch, inside its 24*MAX_NSPLITS sizing) --
    static int attn_gq8 = -1;
    if (attn_gq8 < 0) { const char* e = getenv("SPARKINFER_ATTN_GQ8"); attn_gq8 = (e && e[0] == '0') ? 0 : 1; }
    const bool attn_gate_q8 = attn_gq8 && w.q_has_gate && s.gguf && s.use_pq && s.use_llama
                               && (H == 2048 || H == 4096) && (w.wo_type == 12 || w.wo_type == 8)
                               && (s.qdim % 32 == 0);
    const bool emit_attn_q8 = !w.q_has_gate && s.use_attnin && s.gguf && s.use_pq && s.use_llama
                               && w.wo_type == 12;
    kernels::launch_flash_decode_split(s.q, kpool, vpool, ltab, s.tp_pos + 1, s.attn,
                                       s.fa_m, s.fa_l, s.fa_acc, 1, n_q, n_kv, HD,
                                       s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
                                       1.f / sqrtf((float)HD), st,
                                       (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr, seqlen,
                                       kscale, vscale, kvf,
                                       attn_gate_q8 ? s.qgate : nullptr, 0);
    // Sigmoid gate over the rank's attn window (fused into the combine only for the gated-combine
    // shapes above, which 27B is not).
    if (w.q_has_gate && !attn_gate_q8)
        kernels::launch_qwen36_mul_sigmoid(s.attn, s.qgate, qdim_l, st);

    // -- o_proj: the K-windowed (row-parallel) partial. The rank's blob is compact [N=H][K=qdim_l]
    // (the K axis is the in-feature window of the rank's q heads -- the GGUF [in][out] layout
    // splits along in), and the rank's local attn window IS exactly that K window, so a plain
    // GEMV at K=qdim_l over the leading window is the partial; it lands in the first H of the AR
    // row for the loop's AR-A, mirroring the GDN body's ssm_out.
    if (s.gguf) {
        if (s.use_pq && w.wo_type == 12) {
            if (s.use_llama) {
                kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, qdim_l, st);
                kernels::launch_mmvq_q4k(s.aq81, w.wo, s.tp_ar, H, qdim_l, st);
            } else {
                kernels::launch_quantize_q8_1(s.attn, s.aq8, s.aq8_d, s.aq8_s, qdim_l, st);
                kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, w.wo, s.tp_ar, H, qdim_l, st);
            }
        }
        else if (s.use_pq && s.use_llama && s.use_q6mmvq && w.wo_type == 14) {
            kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, qdim_l, st);
            kernels::launch_mmvq_q6k(s.aq81, w.wo, s.tp_ar, H, qdim_l, st);
        }
        else if (s.use_pq && s.use_llama && w.wo_type == 8) {
            kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, qdim_l, st);
            kernels::launch_mmvq_q80(s.aq81, w.wo, s.tp_ar, H, qdim_l, st);
        }
        else if (w.wo_type == kernels::SI_QTYPE_FP8) {
            kernels::launch_gemv_fp8(s.attn, w.wo, s.tp_ar, H, qdim_l, st);
        }
        else if (w.wo_type == kernels::SI_QTYPE_NVFP4) {
            if (kernels::qwen38_nvfp4_dp4a_proj()) {
                kernels::launch_gemv_nvfp4_quant_x(s.attn, s.nv_pq_b, s.nv_ps_b, 1, qdim_l, st);
                if (!kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_pq_b, s.nv_ps_b, w.wo, s.tp_ar,
                                                           1, H, qdim_l, st))
                    kernels::launch_gemv_nvfp4(s.attn, w.wo, s.tp_ar, H, qdim_l, st);
            } else {
                kernels::launch_gemv_nvfp4(s.attn, w.wo, s.tp_ar, H, qdim_l, st);
            }
        }
        else if (w.wo_type) {
            kernels::launch_gemv_q(s.attn, w.wo, w.wo_type, s.tp_ar, H, qdim_l, st);
        }
        else {
            kernels::launch_gemv(s.attn, w.wo, s.tp_ar, H, qdim_l, st);
        }
    } else {
        kernels::launch_gemm(s.attn, w.wo, s.tp_ar, 1, H, qdim_l, 1.f, 0.f, gc, st);
    }
}

void Qwen35Model::tp_gdn_layer_tp(int l, uint16_t* x, const uint16_t* xn) {
    (void)x;
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const Qwen35LayerWeights& w = s.w.layers[l];
    const int H = c.hidden;
    const int HD = c.linear_head_dim;
    cudaStream_t st = s.stream;
    kernels::GemmConfig gc{};

    // Zero the AR-A staging: the ssm_out K-windowed partial below lands in the first H of the
    // 2*H all-reduced row, the second H stays zero through the sum.
    cu(cudaMemsetAsync(s.tp_ar, 0, (size_t)2 * H * sizeof(bf16), st), "tp gdn ar zero");

    // -- this rank's GDN window (the loader already sliced its weights along exactly it) --
    const int vloc = p_->gdn_window.v_count;                          // 24 per rank on 27B
    const int v0   = p_->gdn_window.v_count > 0 ? p_->gdn_window.v_start : 0;
    const int g     = c.linear_v_heads / c.linear_q_heads;           // v-heads per q-head group
    const int q0    = c.gdn_qh_block ? v0 / g : 0;                   // q/k heads owned by the rank
    const int ql    = c.gdn_qh_block ? vloc / g : vloc;
    // Rank-local blobs are indexed by LOCAL head id (no offsets); only the full-width activation
    // buffers (q/k/v, alpha/beta, gdn/norm, conv state) are shifted to the rank's global window.

    // -- quantize xn once, shared by every full-K GEMV below (same dispatch as the tp=1 body) --
    const bool any_q4k = (w.wqkv_type == 12 || w.wqkv_gate_type == 12 ||
                          w.ssm_alpha_type == 12 || w.ssm_beta_type == 12);
    const bool any_q6k = (w.wqkv_type == 14 || w.wqkv_gate_type == 14 ||
                          w.ssm_alpha_type == 14 || w.ssm_beta_type == 14);
    const bool any_q80 = (w.wqkv_type == 8 || w.wqkv_gate_type == 8 ||
                          w.ssm_alpha_type == 8 || w.ssm_beta_type == 8);
    const bool any_nv  = (w.wqkv_type == kernels::SI_QTYPE_NVFP4 ||
                          w.wqkv_gate_type == kernels::SI_QTYPE_NVFP4 ||
                          w.ssm_alpha_type == kernels::SI_QTYPE_NVFP4 ||
                          w.ssm_beta_type == kernels::SI_QTYPE_NVFP4);
    bool xn_nv = false;
    if (s.gguf && s.use_pq) {
        if (s.use_llama && (any_q4k || any_q80 || (s.use_q6mmvq && any_q6k)))
            kernels::launch_quantize_q8_1_blocks(xn, s.aq81, H, st);
        else if (any_q4k)
            kernels::launch_quantize_q8_1(xn, s.aq8, s.aq8_d, s.aq8_s, H, st);
    }
    xn_nv = s.gguf && any_nv && kernels::qwen38_nvfp4_dp4a_proj();
    if (xn_nv)
        kernels::launch_gemv_nvfp4_quant_x(xn, s.nv_pq_a, s.nv_ps_a, 1, H, st);

    // -- one full-K GEMV over this rank's compact [N'][K=H] weight blob --
    auto proj = [&](const void* W, int t, void* y, int N) {
        if (s.gguf) {
            if (s.use_pq && t == 12) {
                if (s.use_llama) kernels::launch_mmvq_q4k(s.aq81, W, y, N, H, st);
                else             kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, H, st);
            }
            else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14)
                kernels::launch_mmvq_q6k(s.aq81, W, y, N, H, st);
            else if (s.use_pq && s.use_llama && t == 8)
                kernels::launch_mmvq_q80(s.aq81, W, y, N, H, st);
            else if (t == kernels::SI_QTYPE_FP8)
                kernels::launch_gemv_fp8(xn, W, y, N, H, st);
            else if (t == kernels::SI_QTYPE_NVFP4) {
                if (!(xn_nv && kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_pq_a, s.nv_ps_a, W, y,
                                                                    1, N, H, st)))
                    kernels::launch_gemv_nvfp4(xn, W, y, N, H, st);
            }
            else if (t) kernels::launch_gemv_q(xn, W, t, y, N, H, st);
            else         kernels::launch_gemv(xn, W, y, N, H, st);
        } else {
            kernels::launch_gemm(xn, W, y, 1, N, H, 1.f, 0.f, gc, st);
        }
    };

    // -- qkv: one GEMV into staging (rows [ql q | ql k | vloc v] of the rank's blob), then
    // three D2D copies scatter the windows into the full-width PACKED qkv buffer at their global
    // offsets ([q | k | v], the layout the conv kernel below reads as its input; it writes the
    // split lin_q/lin_k/lin_v itself) --
    const int qdim = c.linear_q_heads * HD;
    proj(w.wqkv, w.wqkv_type, s.tp_qkv, 2 * ql * HD + vloc * HD);
    cu(cudaMemcpyAsync(s.lin_qkv + q0 * HD, s.tp_qkv, (size_t)ql * HD * sizeof(bf16),
                       cudaMemcpyDeviceToDevice, st), "tp gdn q");
    cu(cudaMemcpyAsync(s.lin_qkv + qdim + q0 * HD, s.tp_qkv + ql * HD, (size_t)ql * HD * sizeof(bf16),
                       cudaMemcpyDeviceToDevice, st), "tp gdn k");
    cu(cudaMemcpyAsync(s.lin_qkv + 2 * qdim + v0 * HD, s.tp_qkv + 2 * ql * HD,
                       (size_t)vloc * HD * sizeof(bf16), cudaMemcpyDeviceToDevice, st), "tp gdn v");
    proj(w.wqkv_gate, w.wqkv_gate_type, s.lin_z + v0 * HD, vloc * HD);
    proj(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha + v0, vloc);
    proj(w.ssm_beta, w.ssm_beta_type, s.lin_beta + v0, vloc);

    // -- windowed conv: full-width qkv/conv_state operands, rank-local conv_w rows; heads
    // outside the window return before touching conv_state, so each rank owns the conv-state
    // rows of the heads it computes (and only those). The windowed fused kernel is the only
    // window-capable conv here, so it is the tp path even when the tp=1 A/B guard does not
    // hold for this shape (one-shot note; 27B always holds it).
    bf16* conv_state = s.lin_conv_state +
        (size_t)l * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
    {
        static int gdn_fuse = -1;
        if (gdn_fuse < 0) { const char* e = getenv("SPARKINFER_GDN_FUSE"); gdn_fuse = (e && e[0] == '0') ? 0 : 1; }
        static bool conv_note = false;
        if (!conv_note) {
            conv_note = true;
            if (!(gdn_fuse && c.linear_head_dim == 128 && c.linear_q_heads == 16 &&
                  (c.linear_v_heads == 32 || c.linear_v_heads == 48)))
                fprintf(stderr, "[tp] gdn: tp conv uses the windowed fused kernel (the split "
                                "kernel takes no rank window; tp=1 A/B guard not met)\n");
        }
        kernels::launch_qwen36_conv_split_l2norm_fused(s.lin_qkv, w.ssm_conv, conv_state,
                                                       s.lin_q, s.lin_k, s.lin_v,
                                                       c.linear_q_heads, c.linear_v_heads,
                                                       c.linear_head_dim, c.linear_conv_kernel,
                                                       c.rms_eps, st, q0, ql, q0, ql, v0, vloc);
    }

    // -- recurrence: pure base shifts + LOCAL head counts (the tp=1 windowed convention); a/dt
    // ARE shifted by v0: the blobs are FULL-width (48) on every rank (the CT loader loads them
    // whole on both ranks), so this rank's v window [v0, v0+vloc) reads its entries via the +v0
    // shift -- the same convention as the 2221-side GDN body.
    const size_t state_off = (size_t)gdn_state_slot(c, l) * vloc *
        c.linear_head_dim * c.linear_head_dim;
    kernels::launch_qwen36_gdn_ar(s.lin_q + q0 * HD, s.lin_k + q0 * HD,
                                  s.lin_v + v0 * HD,
                                  s.lin_alpha + v0, s.lin_beta + v0,
                                  static_cast<const bf16*>(w.ssm_dt) + v0, static_cast<const bf16*>(w.ssm_a) + v0,   // typed: void* + n is BYTES
                                  s.lin_state, state_off, s.lin_gdn + v0 * HD,
                                  c.gdn_qh_block ? vloc / g : vloc,
                                  vloc,
                                  c.linear_head_dim, c.gdn_qh_block, st,
                                  s.active_lin_state_b16);

    // -- gated norm: the [HD] weight is a single shared vector (unshifted on every model);
    // x/z/out carry the v-window shift.
    kernels::launch_qwen36_gated_norm(s.lin_gdn + v0 * HD, s.lin_z + v0 * HD,
                                      w.ssm_norm, s.lin_norm + v0 * HD,
                                      vloc, c.linear_head_dim, c.rms_eps, st);

    // -- ssm_out: the K-windowed (row-parallel) partial. The rank's weight blob is COMPACT
    // [H][Kw] (N = full hidden, K = this rank's v window); the plain GEMVs at K = Kw are
    // layout-correct for it in every type. The partial lands in tp_ar for the loop's AR-A.
    const int K0 = v0 * HD, Kw = vloc * HD;
    const uint16_t* ln = s.lin_norm + K0;
    if (s.gguf) {
        if (s.use_pq && w.ssm_out_type == 12) {
            if (s.use_llama) {
                kernels::launch_quantize_q8_1_blocks(ln, s.aq81, Kw, st);
                kernels::launch_mmvq_q4k(s.aq81, w.ssm_out, s.tp_ar, H, Kw, st);
            } else {
                kernels::launch_quantize_q8_1(ln, s.aq8, s.aq8_d, s.aq8_s, Kw, st);
                kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, w.ssm_out, s.tp_ar, H, Kw, st);
            }
        }
        else if (s.use_pq && s.use_llama && s.use_q6mmvq && w.ssm_out_type == 14) {
            kernels::launch_quantize_q8_1_blocks(ln, s.aq81, Kw, st);
            kernels::launch_mmvq_q6k(s.aq81, w.ssm_out, s.tp_ar, H, Kw, st);
        }
        else if (s.use_pq && s.use_llama && w.ssm_out_type == 8) {
            kernels::launch_quantize_q8_1_blocks(ln, s.aq81, Kw, st);
            kernels::launch_mmvq_q80(s.aq81, w.ssm_out, s.tp_ar, H, Kw, st);
        }
        else if (w.ssm_out_type == kernels::SI_QTYPE_FP8) {
            kernels::launch_gemv_fp8(ln, w.ssm_out, s.tp_ar, H, Kw, st);
        }
        else if (w.ssm_out_type == kernels::SI_QTYPE_NVFP4) {
            if (kernels::qwen38_nvfp4_dp4a_proj()) {
                kernels::launch_gemv_nvfp4_quant_x(ln, s.nv_pq_b, s.nv_ps_b, 1, Kw, st);
                if (!kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_pq_b, s.nv_ps_b, w.ssm_out,
                                                           s.tp_ar, 1, H, Kw, st))
                    kernels::launch_gemv_nvfp4(ln, w.ssm_out, s.tp_ar, H, Kw, st);
            } else {
                kernels::launch_gemv_nvfp4(ln, w.ssm_out, s.tp_ar, H, Kw, st);
            }
        }
        else if (w.ssm_out_type) {
            kernels::launch_gemv_q(ln, w.ssm_out, w.ssm_out_type, s.tp_ar, H, Kw, st);
        }
        else {
            kernels::launch_gemv(ln, w.ssm_out, s.tp_ar, H, Kw, st);
        }
    } else {
        kernels::launch_gemm(ln, w.ssm_out, s.tp_ar, 1, H, Kw, 1.f, 0.f, gc, st);
    }
}

void Qwen35Model::tp_ffn_layer_tp(int l, const uint16_t* xn, uint16_t* out) {
    (void)xn;
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const Qwen35LayerWeights& w = s.w.layers[l];
    const int H = c.hidden;
    const int fl = c.moe_ffn / 2;   // this rank's gate/up N window and down K window (8704 on 27B)
    cudaStream_t st = s.stream;
    kernels::GemmConfig gc{};
    if (!c.dense_ffn || fl <= 0) {   // non-dense model: no FFN to split; keep the AR row zero
        cu(cudaMemsetAsync(out, 0, (size_t)2 * H * sizeof(bf16), st), "tp ffn zero");
        return;
    }

    // The layer loop passes the layer INPUT norm, but the FFN's input is the POST-ATTENTION
    // norm: the tp=1 body's gate/up/down all read s.hn, which tail1 above just wrote. Reading it
    // directly here (instead of the argument) mirrors the tp=1 sites without touching the loop.
    const bf16* x = s.hn;

    // Zero the AR-B staging: the down K-windowed partial lands in the first H of the 2*H row,
    // the second H stays zero through the all-reduce sum.
    cu(cudaMemsetAsync(out, 0, (size_t)2 * H * sizeof(bf16), st), "tp ffn ar zero");

    // -- this rank's windows: gate and up each own fl rows of the F-row matrices, h is the
    // swiglu over exactly that pair, and down's K window over h is the same fl values,
    // positionally aligned with this rank's compact down blob (the AR-B sums the partials).
    bf16* g = s.tp_ffn;
    bf16* u = s.tp_ffn + fl;
    bf16* h = s.tp_ffn + 2 * fl;

    // -- NVFP4 ModelOpt 27B: the tp=1 dense-FFN sites (nvfp4 dp4a / bf16 fallback), same
    // windows. nv_xq/nv_xs are moe_ffn-sized, so they hold both activation quants (K = H for
    // gate/up, K = fl for down) without any new buffer; nv_pq_a is kmax-sized and is not used.
    if (w.gate_nv && w.up_nv && w.down_nv) {
        if (s.gguf && kernels::qwen38_nvfp4_dp4a()) {
            kernels::launch_gemv_nvfp4_quant_x(x, s.nv_xq, s.nv_xs, 1, H, st);
            if (!kernels::launch_gemv_nvfp4_rows_dp4a2(s.nv_xq, s.nv_xs, w.gate_nv, w.up_nv,
                                                        g, u, 1, fl, H, st)) {
                kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_xq, s.nv_xs, w.gate_nv, g, 1, fl, H, st);
                kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_xq, s.nv_xs, w.up_nv, u, 1, fl, H, st);
            }
            if (!kernels::launch_prefill_swiglu_nvfp4(g, u, h, s.nv_xq, s.nv_xs, fl, st)) {
                kernels::launch_prefill_swiglu(g, u, h, fl, st);
                kernels::launch_gemv_nvfp4_quant_x(h, s.nv_xq, s.nv_xs, 1, fl, st);
            }
            if (!kernels::launch_gemv_nvfp4_rows_dp4a(s.nv_xq, s.nv_xs, w.down_nv, out, 1, H, fl, st))
                kernels::launch_gemv_nvfp4(h, w.down_nv, out, H, fl, st);
        } else {
            kernels::launch_gemv_nvfp4(x, w.gate_nv, g, fl, H, st);
            kernels::launch_gemv_nvfp4(x, w.up_nv, u, fl, H, st);
            kernels::launch_prefill_swiglu(g, u, h, fl, st);
            kernels::launch_gemv_nvfp4(h, w.down_nv, out, H, fl, st);
        }
        return;
    }

    // -- Gguf Q4_K 27B (and the other quantized types): quantize the FFN input ONCE, keyed on
    // the types this FFN actually carries (same dispatch family as tp_gdn_layer_tp's entry).
    const bool any_q4k = (w.gate_qtype == 12 || w.up_qtype == 12 || w.down_qtype == 12);
    const bool any_q6k = (w.gate_qtype == 14 || w.up_qtype == 14 || w.down_qtype == 14);
    const bool any_q80 = (w.gate_qtype == 8  || w.up_qtype == 8  || w.down_qtype == 8);
    if (s.gguf && s.use_pq) {
        if (s.use_llama && (any_q4k || any_q80 || (s.use_q6mmvq && any_q6k)))
            kernels::launch_quantize_q8_1_blocks(x, s.aq81, H, st);
        else if (any_q4k)
            kernels::launch_quantize_q8_1(x, s.aq8, s.aq8_d, s.aq8_s, H, st);
    }

    // -- gate/up: N-window GEMVs over this rank's compact weight blobs (K = H full; the N axis
    // is the fast one in every layout, so a windowed N is a pure row-range shift, no kernel change).
    auto proj = [&](const void* W, int t, bf16* y) {
        if (s.gguf) {
            if (s.use_pq && t == 12) {
                if (s.use_llama) kernels::launch_mmvq_q4k(s.aq81, W, y, fl, H, st);
                else             kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, fl, H, st);
            }
            else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14)
                kernels::launch_mmvq_q6k(s.aq81, W, y, fl, H, st);
            else if (s.use_pq && s.use_llama && t == 8)
                kernels::launch_mmvq_q80(s.aq81, W, y, fl, H, st);
            else if (t == kernels::SI_QTYPE_FP8)
                kernels::launch_gemv_fp8(x, W, y, fl, H, st);
            else if (t == kernels::SI_QTYPE_NVFP4)
                kernels::launch_gemv_nvfp4(x, W, y, fl, H, st);
            else if (t) kernels::launch_gemv_q(x, W, t, y, fl, H, st);
            else         kernels::launch_gemv(x, W, y, fl, H, st);
        } else {
            kernels::launch_gemm(x, W, y, 1, fl, H, 1.f, 0.f, gc, st);
        }
    };
    proj(w.gate_q, w.gate_qtype, g);
    proj(w.up_q,   w.up_qtype,   u);

    // -- swiglu over the rank's window (h is the down's K window, positionally aligned).
    kernels::launch_prefill_swiglu(g, u, h, fl, st);

    // -- down: K-window GEMV over this rank's compact [K = fl][N = H] blob, partial into the
    // first H of the AR row (the K-windowed quants cannot reuse aq81/aq8: those are kmax-sized
    // (6144 on 27B), below fl = 8704, so they land in the tp_attach scratch).
    if (s.gguf) {
        if (s.use_pq && s.use_llama && w.down_qtype == 12) {
            kernels::launch_quantize_q8_1_blocks(h, s.tp_fq81, fl, st);
            kernels::launch_mmvq_q4k(s.tp_fq81, w.down_q, out, H, fl, st);
        }
        else if (s.use_pq && s.use_llama && s.use_q6mmvq && w.down_qtype == 14) {
            kernels::launch_quantize_q8_1_blocks(h, s.tp_fq81, fl, st);
            kernels::launch_mmvq_q6k(s.tp_fq81, w.down_q, out, H, fl, st);
        }
        else if (s.use_pq && s.use_llama && w.down_qtype == 8) {
            kernels::launch_quantize_q8_1_blocks(h, s.tp_fq81, fl, st);
            kernels::launch_mmvq_q80(s.tp_fq81, w.down_q, out, H, fl, st);
        }
        else if (s.use_pq && !s.use_llama && w.down_qtype == 12) {
            kernels::launch_quantize_q8_1(h, s.tp_fq8, s.tp_fq8_d, s.tp_fq8_s, fl, st);
            kernels::launch_gemv_q_dp4a_pq(s.tp_fq8, s.tp_fq8_d, s.tp_fq8_s, w.down_q, out, H, fl, st);
        }
        else if (w.down_qtype == kernels::SI_QTYPE_FP8) {
            kernels::launch_gemv_fp8(h, w.down_q, out, H, fl, st);
        }
        else if (w.down_qtype) {
            kernels::launch_gemv_q(h, w.down_q, w.down_qtype, out, H, fl, st);
        }
        else {
            kernels::launch_gemv(h, w.down_q, out, H, fl, st);
        }
    } else {
        kernels::launch_gemm(h, w.down_q, out, 1, H, fl, 1.f, 0.f, gc, st);
    }
}

void Qwen35Model::tp_allreduce_row(uint16_t* row, size_t elems, bool is_xrow) {
    Impl& s = *p_;
    // Only the group leader (rank 0) issues the all-reduce: a single GpuLink::allreduce posts
    // the reduce on BOTH ranks' streams (see reduce_impl in gpu_link.cpp), so the peer must not
    // issue a second one. The peer's call only takes part in the rendezvous.
    if (!s.tp_link || s.tp_peers.size() != 2) return;
    if (s.tp_rank != 0) { tp_peer_rendezvous("decode allreduce"); return; }
    if (!s.tp_peers[1]) return;
    const TpRankView pv = s.tp_peers[1]->tp_rank_view();
    uint16_t* p_row = is_xrow ? pv.xrow : pv.ar;
    GpuLink::RankRef self_ref{s.device, s.stream, row, row};
    GpuLink::RankRef peer_ref{pv.device, pv.stream, p_row, p_row};
    // reduce_impl requires ref a on the link's device_a and ref b on device_b; order ours to
    // match the link's (dev_a, dev_b) regardless of which physical rank this instance is.
    GpuLink::RankRef a = self_ref, b = peer_ref;
    if (s.tp_link->device_a() != s.device) { a = peer_ref; b = self_ref; }
    tp_leader_rendezvous("decode allreduce", [&] {
        if (!s.tp_link->allreduce(a, b, elems * sizeof(uint16_t), GpuLink::Dtype::BFloat16))
            { cu(cudaErrorUnknown, "tp allreduce"); note_tp_fatal("GpuLink decode allreduce failed"); }
    });
}

// (dual-GPU B1) Asynchronous twin of tp_prefill_allreduce_bf16. The reduce is posted on a per-rank
// SIDE stream instead of the compute stream, ordered after the compute stream's producers by an
// event, so the copy engines move this partial while the compute stream runs on. Rows the op
// covers must not be touched by the compute stream until tp_prefill_allreduce_join(). The side
// stream is in order, so several async ops queue behind each other (GpuLink's landing scratch is
// then never shared by two ops in flight); the synchronous form joins first for the same reason.
// The result is bit-identical to the synchronous form: the same GpuLink op, only another stream.
struct TpArSide {
    static constexpr int kRing = 64;   // per-op completion events; a wrapped slot is re-recorded
                                       // by a LATER op of the same in-order stream, so waiting on
                                       // it can only over-wait, never under-wait
    cudaStream_t side = nullptr;
    // (C1) The ops' reduce kernels run here, not on `side`, so the copy engines start the next
    // op's copy while this one reduces (GpuLink::allreduce_pipelined). `done` is recorded here.
    cudaStream_t red = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t done[kRing] = {};
    int seq = 0;
    bool pending = false;
};
// SPARKINFER_TP_AR_WIRE=e4m3|int8: the async prefill all-reduces send 8-bit codes with a scale
// per 128 values (48 % fewer bytes over the link; lossy, both ranks keep identical sums). Default
// off. Needs the pipelined path.
static int tp_ar_wire() {
    static const int w = [] {
        const char* e = getenv("SPARKINFER_TP_AR_WIRE");
        if (!e) return 0;
        if (!strcmp(e, "e4m3") || !strcmp(e, "fp8") || !strcmp(e, "1")) return 1;
        if (!strcmp(e, "int8") || !strcmp(e, "2")) return 2;
        return 0;
    }();
    return w;
}
// SPARKINFER_TP_AR_PIPE=0: copy and reduce of an async prefill all-reduce on one stream again.
static bool tp_ar_pipe_on() {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_TP_AR_PIPE");
        return !(e && e[0] == '0');
    }();
    return on;
}
static TpArSide g_tp_ar_side[2];

static int tp_prefill_rank() {
    if (!g_tp_prefill_link || g_tp_prefill_dev[0] < 0 || g_tp_prefill_dev[1] < 0) return -1;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        (dev != g_tp_prefill_dev[0] && dev != g_tp_prefill_dev[1]))
        return -1;
    return (dev == g_tp_prefill_dev[0]) ? 0 : 1;
}

void tp_prefill_allreduce_join() {
    const int r = tp_prefill_rank();
    if (r < 0) return;
    TpArSide& sd = g_tp_ar_side[r];
    if (!sd.pending) return;
    cu(cudaStreamWaitEvent(g_tp_prefill_stream[r], sd.done[(sd.seq - 1) % TpArSide::kRing], 0),
       "tp prefill ar join");
    sd.pending = false;
}

void tp_prefill_allreduce_wait(int ticket) {
    const int r = tp_prefill_rank();
    if (r < 0 || ticket < 0) return;
    TpArSide& sd = g_tp_ar_side[r];
    if (!sd.pending || ticket >= sd.seq) return;
    cu(cudaStreamWaitEvent(g_tp_prefill_stream[r], sd.done[ticket % TpArSide::kRing], 0),
       "tp prefill ar wait");
}

bool tp_prefill_ar_side_prepare() {
    const int r = tp_prefill_rank();
    if (r < 0) return true;
    TpArSide& sd = g_tp_ar_side[r];
    if (sd.side) return true;
    // Created by the rank's own thread, on its own (current) device; all or nothing.
    cudaStream_t side = nullptr, red = nullptr;
    cudaEvent_t ready = nullptr, done[TpArSide::kRing] = {};
    // Highest priority: a verify's row all-reduce (one flag-kernel block) is posted here to run
    // beside the GEMM producing the rest of its layer, and must get an SM as soon as one frees.
    int prio_lo = 0, prio_hi = 0;
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    bool ok = cudaStreamCreateWithPriority(&side, cudaStreamNonBlocking, prio_hi) == cudaSuccess &&
              cudaStreamCreateWithPriority(&red, cudaStreamNonBlocking, prio_hi) == cudaSuccess &&
              cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) == cudaSuccess;
    for (cudaEvent_t& e : done)
        ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        for (cudaEvent_t e : done) if (e) cudaEventDestroy(e);
        if (ready) cudaEventDestroy(ready);
        if (red) cudaStreamDestroy(red);
        if (side) cudaStreamDestroy(side);
        return false;
    }
    sd.side = side;
    sd.red = red;
    sd.ready = ready;
    for (int i = 0; i < TpArSide::kRing; i++) sd.done[i] = done[i];
    return true;
}

int tp_prefill_allreduce_bf16_async(void* in_out, size_t elems) {
    if (!in_out || elems == 0) return -1;
    const int r = tp_prefill_rank();
    if (r < 0) return -1;
    TpArSide& sd = g_tp_ar_side[r];
    if (!sd.side && !tp_prefill_ar_side_prepare())
        cu(cudaErrorMemoryAllocation, "tp ar side stream");
    cu(cudaEventRecord(sd.ready, g_tp_prefill_stream[r]), "tp ar ready record");
    cu(cudaStreamWaitEvent(sd.side, sd.ready, 0), "tp ar side wait");
    // Small ops (a verify's rows) take the flag all-reduce on the side stream: the copy-engine
    // pipeline is for prefill chunks of megabytes.
    const bool pipe = tp_ar_pipe_on() && sd.red && elems * sizeof(bf16) > detail::kFlagMaxBytes;
    // Pipelined: this op lands in scratch half (seq & 1), which op seq-2 read in its reduce on
    // `red`; the copy may not overwrite it before that reduce has run.
    if (pipe && sd.pending && sd.seq >= 2)
        cu(cudaStreamWaitEvent(sd.side, sd.done[(sd.seq - 2) % TpArSide::kRing], 0),
           "tp ar slot wait");
    // Both ranks post the same op sequence, so their seq (and slot) agree.
    const int slot = sd.seq & 1;
    g_tp_prefill_buf[r] = in_out;
    auto post = [&] {
        GpuLink::RankRef a{g_tp_prefill_dev[0], g_tp_ar_side[0].side, g_tp_prefill_buf[0],
                           g_tp_prefill_buf[0]};
        GpuLink::RankRef b{g_tp_prefill_dev[1], g_tp_ar_side[1].side, g_tp_prefill_buf[1],
                           g_tp_prefill_buf[1]};
        const bool ok = pipe
            ? g_tp_prefill_link->allreduce_pipelined(a, b, elems * sizeof(bf16),
                                                     GpuLink::Dtype::BFloat16, g_tp_ar_side[0].red,
                                                     g_tp_ar_side[1].red, slot, tp_ar_wire())
            : g_tp_prefill_link->allreduce(a, b, elems * sizeof(bf16), GpuLink::Dtype::BFloat16);
        if (!ok)
            { cu(cudaErrorUnknown, "tp prefill allreduce async"); note_tp_fatal("GpuLink prefill allreduce failed"); }
    };
    if (r != 0) tp_peer_rendezvous("prefill allreduce async");
    else tp_leader_rendezvous("prefill allreduce async", post);
    // After the rendezvous the op is on both ranks' streams; `done` marks everything posted so far
    // (on `red` when pipelined: the reduces run there in op order).
    const int ticket = sd.seq++;
    cu(cudaEventRecord(sd.done[ticket % TpArSide::kRing], pipe ? sd.red : sd.side),
       "tp ar done record");
    sd.pending = true;
    return ticket;
}

void tp_prefill_allreduce_bf16(void* in_out, size_t elems) {
    tp_prefill_allreduce_join();   // the link's landing scratch may still be in use by an async op
    // Prefill o_proj partial (S7a-1): each rank's pass calls this with its per-pass [N][H] buffer
    // right after enqueuing its K-compact o_proj GEMM. The call registers the buffer in the
    // caller's process-static slot. One link call posts on BOTH ranks' streams, so only the
    // leader issues the reduce (as in tp_allreduce_row), inside the two-way host rendezvous: it
    // posts after the peer has enqueued its GEMM, and the peer enqueues nothing further until the
    // post has landed on its stream (GpuLink's entry events then order the copy after the GEMM).
    if (!g_tp_prefill_link || !in_out || elems == 0) return;
    if (g_tp_prefill_dev[0] < 0 || g_tp_prefill_dev[1] < 0) return;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        (dev != g_tp_prefill_dev[0] && dev != g_tp_prefill_dev[1]))
        return;
    const int r = (dev == g_tp_prefill_dev[0]) ? 0 : 1;
    g_tp_prefill_buf[r] = in_out;
    if (r != 0) { tp_peer_rendezvous("prefill allreduce"); return; }
    tp_leader_rendezvous("prefill allreduce", [&] {
        GpuLink::RankRef a{g_tp_prefill_dev[0], g_tp_prefill_stream[0], g_tp_prefill_buf[0],
                           g_tp_prefill_buf[0]};
        GpuLink::RankRef b{g_tp_prefill_dev[1], g_tp_prefill_stream[1], g_tp_prefill_buf[1],
                           g_tp_prefill_buf[1]};
        if (!g_tp_prefill_link->allreduce(a, b, elems * sizeof(bf16), GpuLink::Dtype::BFloat16))
            { cu(cudaErrorUnknown, "tp prefill allreduce"); note_tp_fatal("GpuLink prefill allreduce failed"); }
    });
}

// (dual-GPU) Host-side agreement for a rank-local decision inside a mirrored pass: both ranks
// call it at the same point with their own value and both get the minimum. Prefill sizes its
// scratch from the card's own free memory, and a rank that shrinks a chunk or falls back on its
// own runs a different sequence of link ops than its peer -- the ranks fall out of step and the
// next collective fails. Agreeing first keeps them on one path. Not tp: returns v unchanged.
static std::atomic<int> g_tp_agree_in[2];
static std::atomic<int> g_tp_agree_out{0};
// Each rank's call site (file, line). Agreements pair by arrival order only, so a rank that
// skipped one pairs every later agreement -- and link op -- with the wrong partner; seen once as
// a prefill FFN chunk "agreed" to 1 and a GpuLink op on a null buffer. The leader names both
// sites when they differ.
static const char* g_tp_agree_file[2] = {nullptr, nullptr};
static std::atomic<int> g_tp_agree_line[2];
static int tp_prefill_agree(int v, int (*op)(int, int), int line, const char* file) {
    if (!g_tp_prefill_link || g_tp_prefill_dev[0] < 0 || g_tp_prefill_dev[1] < 0) return v;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        (dev != g_tp_prefill_dev[0] && dev != g_tp_prefill_dev[1]))
        return v;
    const int r = (dev == g_tp_prefill_dev[0]) ? 0 : 1;
    g_tp_agree_in[r].store(v, std::memory_order_release);
    g_tp_agree_file[r] = file;
    g_tp_agree_line[r].store(line, std::memory_order_release);
    if (r != 0) {
        tp_peer_rendezvous("prefill agree");
        return g_tp_agree_out.load(std::memory_order_acquire);
    }
    int out = v;
    tp_leader_rendezvous("prefill agree", [&] {
        const int peer_line = g_tp_agree_line[1].load(std::memory_order_acquire);
        const char* peer_file = g_tp_agree_file[1];
        if (peer_line != line || !peer_file || strcmp(peer_file, file) != 0) {
            fprintf(stderr, "[tp] agreement out of step: rank 0 at %s:%d, rank 1 at %s:%d\n",
                    file, line, peer_file ? peer_file : "?", peer_line);
            note_tp_fatal("tp agreement out of step");
        }
        out = op(v, g_tp_agree_in[1].load(std::memory_order_acquire));
        g_tp_agree_out.store(out, std::memory_order_release);
    });
    return out;
}
int tp_prefill_agree_min(int v, int line, const char* file) {
    return tp_prefill_agree(v, [](int a, int b) { return std::min(a, b); }, line, file);
}
int tp_prefill_agree_and(int v, int line, const char* file) {
    return tp_prefill_agree(v, [](int a, int b) { return a & b; }, line, file);
}

// (dual-GPU) Row-wise argmax over the vocab-split head without moving the logits: each rank brings
// its half's best (value, local index) per row; both get the global index. Rank 0 holds the lower
// vocab half, so it wins ties -- the lowest-index rule an argmax over the concatenated row applies.
// Host exchange through the same two-way rendezvous as tp_prefill_agree. Not tp: local result.
static float g_tp_amax_val[2][64];
static int g_tp_amax_idx[2][64];
void tp_exchange_argmax(const float* val, const int* idx, int n, int rows_per_rank, int* out) {
    const int r = tp_prefill_rank();
    if (r < 0 || n > 64) { for (int i = 0; i < n; i++) out[i] = idx[i]; return; }
    for (int i = 0; i < n; i++) { g_tp_amax_val[r][i] = val[i]; g_tp_amax_idx[r][i] = idx[i]; }
    auto combine = [&] {
        for (int i = 0; i < n; i++)
            g_tp_amax_idx[0][i] = (g_tp_amax_val[0][i] >= g_tp_amax_val[1][i] ||
                                   g_tp_amax_val[1][i] != g_tp_amax_val[1][i])   // NaN on rank 1
                                      ? g_tp_amax_idx[0][i]
                                      : g_tp_amax_idx[1][i] + rows_per_rank;
    };
    if (r != 0) tp_peer_rendezvous("argmax exchange");
    else tp_leader_rendezvous("argmax exchange", combine);
    for (int i = 0; i < n; i++) out[i] = g_tp_amax_idx[0][i];
    // The peer reads slot 0 after the leader's post; hold the leader until it has, so the next
    // exchange cannot overwrite slot 0 under it.
    tp_prefill_agree_min(0);
}

// Slots for tp_exchange_topk, as for tp_exchange_argmax: each rank writes its own before arriving.
static float g_tp_tk_val[2][32 * dflash_kernels::kRowsTopkMax];
static int g_tp_tk_idx[2][32 * dflash_kernels::kRowsTopkMax];
void tp_exchange_topk(const float* val, const int* idx, int n, int rows_per_rank, float* out_v,
                      int* out_i) {
    const int K = dflash_kernels::kRowsTopkMax;
    const int r = tp_prefill_rank();
    if (r < 0 || n > 32) {
        for (int i = 0; i < n * K; i++) { out_v[i] = val[i]; out_i[i] = idx[i]; }
        return;
    }
    for (int i = 0; i < n * K; i++) {
        g_tp_tk_val[r][i] = val[i];
        g_tp_tk_idx[r][i] = idx[i] == INT_MAX ? INT_MAX : idx[i] + r * rows_per_rank;
    }
    if (r != 0) tp_peer_rendezvous("topk exchange");
    else tp_leader_rendezvous("topk exchange", [] {});
    // Both halves are published: merge them here, the same way on each rank.
    auto better = [](float av, int ai, float bv, int bi) { return av > bv || (av == bv && ai < bi); };
    for (int row = 0; row < n; row++) {
        const float* v0 = g_tp_tk_val[0] + row * K;
        const int* i0 = g_tp_tk_idx[0] + row * K;
        const float* v1 = g_tp_tk_val[1] + row * K;
        const int* i1 = g_tp_tk_idx[1] + row * K;
        int a = 0, b = 0;
        for (int k = 0; k < K; k++) {
            const bool take0 = b >= K || (a < K && better(v0[a], i0[a], v1[b], i1[b]));
            out_v[row * K + k] = take0 ? v0[a] : v1[b];
            out_i[row * K + k] = take0 ? i0[a++] : i1[b++];
        }
    }
    // The peer has read both slots once the leader passes this: the next exchange may overwrite.
    tp_prefill_agree_min(0);
}

// Slots for tp_allreduce_bf16_on: each rank publishes its buffer and stream before arriving, the
// leader reads both inside its post. A rank cannot republish before the leader has posted the
// op it arrived for (the peer spins until then), so a slot is never overwritten under the read.
static void* g_tp_on_buf[2] = {nullptr, nullptr};
static cudaStream_t g_tp_on_stream[2] = {nullptr, nullptr};
void tp_allreduce_bf16_on(void* in_out, size_t elems, cudaStream_t stream) {
    if (!in_out || elems == 0 || !stream) return;
    const int r = tp_prefill_rank();
    if (r < 0) return;
    tp_prefill_allreduce_join();   // an async prefill op may still own the link's landing scratch
    g_tp_on_buf[r] = in_out;
    g_tp_on_stream[r] = stream;
    if (r != 0) { tp_peer_rendezvous("draft allreduce"); return; }
    tp_leader_rendezvous("draft allreduce", [&] {
        GpuLink::RankRef a{g_tp_prefill_dev[0], g_tp_on_stream[0], g_tp_on_buf[0], g_tp_on_buf[0]};
        GpuLink::RankRef b{g_tp_prefill_dev[1], g_tp_on_stream[1], g_tp_on_buf[1], g_tp_on_buf[1]};
        if (!g_tp_prefill_link->allreduce(a, b, elems * sizeof(bf16), GpuLink::Dtype::BFloat16))
            { cu(cudaErrorUnknown, "tp draft allreduce"); note_tp_fatal("GpuLink draft allreduce failed"); }
    });
}

void tp_run_with_peer(int peer_device, const std::function<void()>& peer_fn,
                      const std::function<void()>& leader_fn) {
    if (!peer_fn) {
        if (leader_fn) leader_fn();
        return;
    }
    TpWorker& w = tp_worker_for(peer_device);
    std::unique_lock<std::mutex> hold(w.scope_mu);
    std::function<void()> job = peer_fn;
    w.post(&job);
    if (leader_fn) leader_fn();
    w.wait();
}

void tp_prefill_allreduce_f32(float* in_out, size_t elems) {
    // Same registration + two-way rendezvous as tp_prefill_allreduce_bf16, for an f32 buffer
    // (the prefill seed's zero-padded [vocab] logits row).
    if (!g_tp_prefill_link || !in_out || elems == 0) return;
    if (g_tp_prefill_dev[0] < 0 || g_tp_prefill_dev[1] < 0) return;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        (dev != g_tp_prefill_dev[0] && dev != g_tp_prefill_dev[1]))
        return;
    const int r = (dev == g_tp_prefill_dev[0]) ? 0 : 1;
    g_tp_prefill_buf[r] = in_out;
    if (r != 0) { tp_peer_rendezvous("prefill f32 allreduce"); return; }
    tp_leader_rendezvous("prefill f32 allreduce", [&] {
        GpuLink::RankRef a{g_tp_prefill_dev[0], g_tp_prefill_stream[0], g_tp_prefill_buf[0],
                           g_tp_prefill_buf[0]};
        GpuLink::RankRef b{g_tp_prefill_dev[1], g_tp_prefill_stream[1], g_tp_prefill_buf[1],
                           g_tp_prefill_buf[1]};
        if (!g_tp_prefill_link->allreduce(a, b, elems * sizeof(float), GpuLink::Dtype::Float32))
            cu(cudaErrorUnknown, "tp prefill f32 allreduce");
    });
}

namespace {
bool prefill_samples_lmhead() {
    static int legacy = -1;
    if (legacy < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_LEGACY");
        legacy = (e && e[0] == '1') ? 1 : 0;
    }
    return legacy != 0;
}

// Batched prefill (prefill_batched_run). Default ON; SPARKINFER_PREFILL_BATCHED=0 disables. Supports
// the Qwythos dense-hybrid AND the Qwen3.6-35B-A3B MoE hybrid (dense_ffn=false, n_experts>0) — both
// share the GDN + attention batched kernels; only the FFN differs. From position 0 only.
bool batched_prefill_enabled(bool gguf, const Qwen35Config& cfg, int n_tokens) {
    static int want_batched = -1, batched_maxctx = -1;
    if (want_batched < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_BATCHED");
        want_batched = (e && e[0] == '0') ? 0 : 1;
        // 128k: the windowed prefill attention (#455) is O(N*window) and the FFN scratch is chunked
        // (prefill_batched_run), so the batched pass now fits VRAM and stays flat ~18k pp up to 128k
        // (vs the ~300 pp sequential fallback). Raised from 64k. SPARKINFER_PREFILL_BATCHED_MAXCTX overrides.
        const char* mc = getenv("SPARKINFER_PREFILL_BATCHED_MAXCTX");
        batched_maxctx = mc ? atoi(mc) : 131072;
    }
    // dense hybrid (Qwythos) or the Qwen3.6 MoE hybrid — prefill_batched_run validates the
    // MoE requirements (256 experts, quantized experts + router) itself and returns -1 to
    // fall back if unsupported.
    const bool ffn_ok = cfg.dense_ffn || cfg.n_experts > 0;
    // Muse Glimmer: windowed batched prefill IS implemented in prefill_batched_run -- per-layer
    // NoPE (rotary_dim=0) on the global layers, pure rolling-window attention (launch_prefill_attn_
    // swa_pure_bf16) on the SWA layers, bf16 KV write, and sandwich/embedding norm + logit softcap
    // parity with the decode path. DEFAULT ON: numerically equivalent to the token-loop prefill
    // (batched-vs-token-loop parity PPL 1.232 vs 1.233 & identical argmax; compute-sanitizer 0
    // errors; Qwen3.6 cross-model guard clean). SPARKINFER_MUSE_BATCHED=0 forces the sequential
    // token-loop fallback (kept for A/B and as an escape hatch).
    if (cfg.muse_glimmer) {
        static const int muse_batched = []{ const char* e = getenv("SPARKINFER_MUSE_BATCHED"); return (e && e[0] == '0') ? 0 : 1; }();
        if (!muse_batched) return false;
    }
    return want_batched && gguf && cfg.hybrid && ffn_ok && n_tokens > 0 &&
           n_tokens <= batched_maxctx;
}

// Tokens per batched-prefill window, for a prompt long enough to need windowing at all (see
// prefill_single_pass_max_tokens). The single-pass arena scales with the token count (mostly the
// N*H activations and the attention scratch): it measured ~8.4 GB at 128k, which no longer fits
// beside an NVFP4 27B and its 128k+ KV cache -- the allocation fails and the WHOLE prompt drops
// onto the token loop, ~30x slower. Windowing bounds the arena by this constant instead, so a
// 256k prompt costs the same arena as a 32k one.
//
// 16384 rather than a window as large as will fit: measured at ctx=262144 on an RTX 5090, 16384
// ingests at 2642 pp against 2135 for 32768, i.e. the SMALLER window is 24% faster. The arena a
// window holds is released and reallocated per window once it exceeds the keep-resident budget,
// and a smaller one both costs less to cycle and leaves more of L2 to the attention scan against
// a quarter-million-token KV cache -- which is what a long prefill is actually spending its time
// on. SPARKINFER_PREFILL_WINDOW overrides; 0 restores the single unwindowed pass.
int prefill_window_tokens() {
    static const int w = [] {
        const char* e = getenv("SPARKINFER_PREFILL_WINDOW");
        const int v = e ? atoi(e) : 16384;
        return v < 0 ? 0 : v;
    }();
    return w;
}

// The same window, bounded by what a capped windowed KV slice can serve in one pass. A ring holds
// its window plus window_pass_tokens, so a prefill pass longer than that would overwrite rows its
// own earliest query still reads -- prefill_batched_run refuses one, and clamping here keeps a long
// prompt on the batched path (chunked into ring-sized windows) instead of the token loop.
// Uncapped pools are unaffected: window_pass_limit() is INT_MAX there.
int prefill_single_pass_max_tokens(const KVCacheManager* kv);
int prefill_window_tokens(const KVCacheManager* kv) {
    int w = prefill_window_tokens();
    if (kv && kv->windowed()) {
        const int lim = kv->window_pass_limit();
        if (lim > 0 && (w <= 0 || w > lim)) w = lim;
    }
    return w;
}

// Largest prompt still ingested in ONE pass. Separate from the window size on purpose: windowing
// exists to bound the arena, so below the size where a single pass demonstrably fits there is no
// reason to pay for it (a second LM-head tail, an arena release-and-reallocate per window, and no
// whole-prefill CUDA graph after the first window). 32768 is where the single pass is measured
// working and fast, and holding the threshold there is what keeps every context at or below it on
// exactly the path it had -- including the 4k/16k/32k prefill dimensions, which are no-regression
// floors. The window size below it is then free to be whatever ingests a LONG prompt fastest,
// which is not the same number. SPARKINFER_PREFILL_SINGLE_MAX overrides.
int prefill_single_pass_max_tokens() {
    static const int m = [] {
        const char* e = getenv("SPARKINFER_PREFILL_SINGLE_MAX");
        const int v = e ? atoi(e) : 32768;
        return v < 0 ? 0 : v;
    }();
    return m;
}

// Batched-prefill eligibility for a prompt that may be windowed: every condition of
// batched_prefill_enabled() except the context cap, which a window makes irrelevant -- what has
// to fit is one window, not the prompt.
// Past the ring's reach a prompt MUST be windowed, so the single-pass bound shrinks with it.
int prefill_single_pass_max_tokens(const KVCacheManager* kv) {
    int m = prefill_single_pass_max_tokens();
    if (kv && kv->windowed()) {
        const int lim = kv->window_pass_limit();
        if (lim > 0 && m > lim) m = lim;
    }
    return m;
}

bool batched_prefill_windowed_enabled(bool gguf, const Qwen35Config& cfg, int n_tokens,
                                      const KVCacheManager* kv = nullptr) {
    const int w = prefill_window_tokens(kv);
    if (n_tokens <= 0) return false;
    if (w <= 0 || n_tokens <= prefill_single_pass_max_tokens(kv))
        return batched_prefill_enabled(gguf, cfg, n_tokens);
    return batched_prefill_enabled(gguf, cfg, w);
}

// LMCache chunk size, tokens. Doubles as both the LOOKUP eligibility threshold (below this many
// tokens, always recompute locally rather than pay an IPC round trip) and the STORE alignment
// unit (a store range is rounded down to the nearest multiple, matching what the sidecar's own
// LMCache instance is configured with -- see bridge/lmcache_bridge.py's identically-named env
// var). Not negotiated from the sidecar's HELLO_ACK at runtime (that field exists on the wire
// but is currently informational-only on this side, see lmcache_bridge_client.cpp) -- both sides
// must be configured to agree; a mismatch degrades safely (misaligned STOREs are rejected by the
// sidecar, not corrupted) rather than crashing, so this is a real but non-catastrophic
// simplification, not a hidden correctness trap.
int lmcache_chunk_size_tokens() {
    static int chunk = [] {
        const char* e = getenv("SPARKINFER_LMCACHE_CHUNK_SIZE");
        const int v = e ? atoi(e) : 256;
        return v > 0 ? v : 256;
    }();
    return chunk;
}

BridgeKVLayout lmcache_layout_from_cfg(const Qwen35Config& cfg, const KVCacheManager& kv) {
    BridgeKVLayout layout;
    layout.num_layers = cfg.n_layers;
    layout.num_kv_heads = cfg.n_kv_heads;
    layout.head_dim = cfg.head_dim;
    layout.block_size = kv.block_size();
    layout.int8_kv = kv.int8_kv();
    layout.elem_bytes = kv.int8_kv() ? 1 : 2;
    // model_name is unused for the staging byte-math this layout feeds (stage_kv_to_shm /
    // restore_kv_from_shm never read it) -- the BridgeClient's own layout, fixed at construction
    // and used for HELLO, is what actually needs to match the sidecar's.
    return layout;
}

// Stores tokens[start, end) (rounded down to the nearest lmcache_chunk_size_tokens() boundary --
// a trailing partial chunk is silently dropped, not sent, since the sidecar would reject it as
// misaligned anyway) for seq_id to the external cache tier. No-op if no bridge is attached, the
// bridge is currently unreachable, or nothing chunk-aligned remains. Fire-and-forget: the actual
// network round trip happens on BridgeClient's own background thread, this call only blocks for
// the (synchronous, on-stream) cudaMemcpy staging into shm.
//
// Takes individual fields rather than Impl& -- Impl is a private nested type, so a free function
// outside the class can't name it in a signature even from within the same translation unit
// (only member functions, which is why every call site below reads `s.<field>` from inside an
// actual Qwen35Model member function's `Impl& s = *p_;`).
void lmcache_maybe_store(BridgeClient* bridge, const Qwen35Config& cfg, KVCacheManager& kv,
                         cudaStream_t stream, uint64_t seq_id, const std::vector<int>& tokens,
                         int start, int end) {
    // Deliberately not gated on bridge->is_alive(): that only becomes true as a side effect of
    // a prior successful handshake, so gating on it here would mean the very first store in a
    // freshly started process never fires (nothing has ever connected yet to make it true) --
    // found via the live E2E test (lmcache_e2e_gpu_test.cpp), which is exactly the class of bug
    // that kind of test exists to catch. store_async() (and the LOOKUP call in
    // ingest_prompt_range, same reasoning) already handle "is the bridge actually reachable"
    // internally via their own lazy-connect + cooldown/backoff -- that's what their docstrings
    // mean by "safe to call regardless."
    if (!bridge) return;
    const int chunk = lmcache_chunk_size_tokens();
    const int aligned_end = (end / chunk) * chunk;
    if (aligned_end <= start) return;

    static std::atomic<uint64_t> counter{0};
    const std::string shm_name = "/sparkinfer_kv_store_" + std::to_string(seq_id) + "_" +
                                 std::to_string(counter.fetch_add(1));
    const BridgeKVLayout layout = lmcache_layout_from_cfg(cfg, kv);
    if (!stage_kv_to_shm(kv, layout, seq_id, start, aligned_end, shm_name, stream)) return;
    bridge->store_async(tokens, start, aligned_end, shm_name);
}
} // namespace

// Fills `ids` with the first `n` token ids from SPARKINFER_BENCH_PROMPT_FILE (whitespace-separated
// decimal ids). Returns false -- leaving `ids` untouched for the caller's synthetic fallback -- if
// the var is unset, the file is unreadable, or it holds fewer than `n` ids. Deliberately strict
// about the short-file case: silently padding a real prompt with synthetic filler would produce a
// number that is neither one thing nor the other, and would do it invisibly.
static bool bench_prompt_ids_from_env(std::vector<int>& ids, int n) {
    const char* path = getenv("SPARKINFER_BENCH_PROMPT_FILE");
    if (!path || !*path) return false;
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "[bench] prompt file %s unreadable — using synthetic prompt\n", path); return false; }
    std::vector<int> got;
    got.reserve(n);
    for (long v; f >> v; ) {
        got.push_back((int)v);
        if ((int)got.size() >= n) break;
    }
    if ((int)got.size() < n) {
        fprintf(stderr, "[bench] prompt file %s has %zu ids, need %d — using synthetic prompt\n",
                path, got.size(), n);
        return false;
    }
    ids.assign(got.begin(), got.begin() + n);
    fprintf(stderr, "[bench] prompt: %d real tokens from %s\n", n, path);
    return true;
}

Qwen35Model::BenchDecodeResult Qwen35Model::bench_decode(int warmup, int n, int context_tokens) {
    BenchDecodeResult out{};
    Impl& s = *p_;
    static int last_bench_ctx = -1;
    if (context_tokens != last_bench_ctx && s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec);
        cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr;
        s.cu_graph = nullptr;
        s.graph_ready = false;
    }
    last_bench_ctx = context_tokens;
    if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) { fprintf(stderr, "[bench] kv allocate failed\n"); return out; }
    int start_pos = context_tokens;
    if (const char* e = getenv("SPARKINFER_BENCH_START_POS")) {
        start_pos = atoi(e);
    }
    if (start_pos < 0) start_pos = 0;
    if (start_pos + warmup + n > s.cfg.max_seq) {
        fprintf(stderr, "[bench] requested ctx=%d warmup=%d n=%d exceeds max_seq=%d\n",
                start_pos, warmup, n, s.cfg.max_seq);
        s.kv->free(s.active_seq_id);
        return out;
    }
    static int bench_device_loop = -1;
    if (bench_device_loop < 0) {
        const char* e = getenv("SPARKINFER_BENCH_DEVICE_LOOP");
        bench_device_loop = (e && e[0] == '0') ? 0 : 1;
    }
    s.bench_feedback_graph = bench_device_loop != 0;
    int pos = 0, tok = 100;
    // Batched prefill: one weight-amortized GEMM pass fills the KV cache + Gated-DeltaNet state, then
    // decode continues from start_pos. Default ON for the dense hybrid; SPARKINFER_PREFILL_BATCHED=0
    // (or ctx > SPARKINFER_PREFILL_BATCHED_MAXCTX, default 64k — the O(N^2) prefill attention is still
    // naive) falls back to the token loop below, which is left byte-identical to main on purpose.
    bool batched_done = false;
    if (start_pos > 0) {
        if (batched_prefill_windowed_enabled(s.gguf, s.cfg, start_pos, s.kv)) {
            std::vector<int> ids(start_pos);
            // Default is a synthetic ramp, NOT text. That is fine for a weight-bandwidth-bound
            // dense decode, but it is out-of-distribution for anything whose cost depends on token
            // CONTENT -- MoE expert routing, GDN state, any cache keyed on repeats -- so an
            // optimization that only pays off on this ramp would still score as a real speedup.
            // SPARKINFER_BENCH_PROMPT_FILE (space-separated ids, e.g. produced by the eval bot's
            // own tokenizer) substitutes a real prompt. Left OPT-IN so existing baselines across
            // every model stay comparable; see bench_prompt_ids() for the parsing.
            if (!bench_prompt_ids_from_env(ids, start_pos))
                for (int i = 0; i < start_pos; i++) ids[i] = 100 + (i % 20000);   // deterministic pseudo-prompt
            auto pb0 = std::chrono::high_resolution_clock::now();
            // No out_done: on a refusal the loop below re-ingests from position 0, and since it
            // feeds its own output back as the next input (this is a synthetic bench prompt, not
            // ids[]) resuming mid-way would not reproduce the same sequence anyway.
            int seed = prefill_batched_chunked(ids.data(), start_pos);
            cudaDeviceSynchronize();
            auto pb1 = std::chrono::high_resolution_clock::now();
            if (seed >= 0) {
                out.prefill_pp = start_pos / std::chrono::duration<double>(pb1 - pb0).count();
                pos = start_pos;
                tok = (seed < s.cfg.vocab) ? seed : 100;
                batched_done = true;
            }
        }
    }
    if (start_pos > 0 && !batched_done) {
        auto p0 = std::chrono::high_resolution_clock::now();
        for (; pos < start_pos; pos++) {
            tok = forward_token(tok, pos, prefill_samples_lmhead());
            if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
        }
        cudaDeviceSynchronize();
        auto p1 = std::chrono::high_resolution_clock::now();
        out.prefill_pp = start_pos / std::chrono::duration<double>(p1 - p0).count();
    }
    for (int i = 0; i < warmup; i++) {
        tok = forward_token(tok, pos++, true);
        if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
    }
    if (s.graph_ready) cu(cudaGraphUpload(s.cu_exec, s.stream), "bench graph upload");
    cudaDeviceSynchronize();

    if (bench_device_loop && s.graph_ready) {
        s.h_scalars[0] = tok;
        s.h_scalars[1] = pos;
        s.h_scalars[2] = pos;
        s.h_scalars[3] = pos + 1;
        cu(cudaMemcpyAsync(s.d_scalars, s.h_scalars, 4 * sizeof(int), cudaMemcpyHostToDevice, s.stream), "bench scalars");
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++) {
            cu(cudaGraphLaunch(s.cu_exec, s.stream), "bench graph launch");
        }
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, s.stream), "bench final out");
        cu(cudaStreamSynchronize(s.stream), "bench sync");
        auto t1 = std::chrono::high_resolution_clock::now();
        s.kv->free(s.active_seq_id);
        s.bench_feedback_graph = false;
        double secs = std::chrono::duration<double>(t1 - t0).count();
        out.decode_tps = n / secs;
        return out;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        tok = forward_token(tok, pos++, true);
        if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    s.kv->free(s.active_seq_id);
    s.bench_feedback_graph = false;
    double secs = std::chrono::duration<double>(t1 - t0).count();
    out.decode_tps = n / secs;
    return out;
}

void Qwen35Model::drop_parked_decode_graphs() {
    Impl& s = *p_;
    // Self-guarding: most callers reach this through invalidate_decode_graph(), and while the
    // device-touching ones (cache_prefix, clear_prefix_cache) already hold device_mu, the DSpark
    // ones (set_dflash_capture, restore_spec_snapshot) do not. Destroying a raw graph handle
    // unlocked was merely racy; mutating a std::unordered_map unlocked is undefined. Recursive,
    // so the callers that do hold it are unaffected.
    std::lock_guard<std::recursive_mutex> device_lock(s.device_mu);
    for (auto& kv : s.parked_graphs) {
        if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        if (kv.second.graph) cudaGraphDestroy(kv.second.graph);
    }
    s.parked_graphs.clear();
}

void Qwen35Model::invalidate_decode_graph() {
    Impl& s = *p_;
    // Parked graphs go too. This function means "something the capture depends on changed"
    // (weights, splits policy, sparse budget, bench toggles), and a graph parked under the old
    // setting is exactly as stale as the active one. activate_session() deliberately does NOT
    // route its decode-graph handling through here -- a session switch invalidates nothing, it
    // just moves which capture is current.
    drop_parked_decode_graphs();
    if (s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec);
        cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr;
        s.cu_graph = nullptr;
        s.graph_ready = false;
        s.graph_attn_mode = -1;
        s.graph_sparse = false;
    }
    if (s.dflash_graph_ready) {
        cudaGraphExecDestroy(s.cu_dflash_exec);
        cudaGraphDestroy(s.cu_dflash_graph);
        s.cu_dflash_exec = nullptr;
        s.cu_dflash_graph = nullptr;
        s.dflash_graph_ready = false;
        s.dflash_graph_attn_mode = -1;
        s.dflash_graph_sparse = false;
    }
    // The PREFILL graph has to go too, for the same reason as the two above: every attention
    // kernel node in it was captured with `btable` (kv->block_table(active_seq_id)) baked in as a
    // literal argument, so replaying it after the active session changed reads and appends KV
    // through the OLD session's block table -- which by then is typically freed. This function is
    // what activate_session() calls precisely to mean "the session identity underneath the graphs
    // just moved", and leaving one of the three graphs behind made that promise false.
    if (s.graph_prefill_ready) {
        cudaGraphExecDestroy(s.cu_prefill_exec);
        cudaGraphDestroy(s.cu_prefill_graph);
        s.cu_prefill_exec = nullptr;
        s.cu_prefill_graph = nullptr;
        s.graph_prefill_ready = false;
        s.graph_prefill_attn_mode = -1;
    }
}

bool Qwen35Model::prompt_matches_prefix(const std::vector<int>& prompt) const {
    const Impl& s = *p_;
    if (!s.prefix_active || s.prefix_len <= 0) return false;
    if (prompt.size() < (size_t)s.prefix_len) return false;
    for (int i = 0; i < s.prefix_len; i++)
        if (prompt[(size_t)i] != s.prefix_tokens[(size_t)i]) return false;
    return true;
}

int Qwen35Model::ingest_prompt_range(const int* ids, int start, int end, int chunk_limit,
                                     int* out_pos, bool want_seed_logprob,
                                     bool allow_batched_resume) {
    int tp_peer_pos = 0;
    TP_MIRROR(ingest_prompt_range(ids, start, end, chunk_limit, out_pos ? &tp_peer_pos : nullptr,
                                  want_seed_logprob, allow_batched_resume));
    Impl& s = *p_;
    if (!ids || end <= start) {
        if (out_pos) *out_pos = start;
        return -1;
    }
    const int n = end - start;
    // Batched GEMM prefill. Only eligible on the very first call for this range (start==0) --
    // a windowed pass carries the Gated-DeltaNet recurrence forward from wherever the state
    // already is, so it has to begin where the state does -- and it always covers the whole
    // [0,end), in prefill_window_tokens()-sized windows, regardless of chunk_limit.
    int batched_done = 0;
    if (start == 0 && batched_prefill_windowed_enabled(s.gguf, s.cfg, n, s.kv)) {
        int seed = prefill_batched_chunked(ids, n, want_seed_logprob, &batched_done);
        if (seed >= 0) {
            if (out_pos) *out_pos = end;
            return seed;
        }
    } else if (start > 0 && allow_batched_resume &&
               batched_prefill_windowed_enabled(s.gguf, s.cfg, n, s.kv)) {
        // A prefix-cache hit: KV for [0, start) is shared in and the recurrent state at `start` is
        // restored, which is exactly what a windowed batched pass continues from. Without this the
        // whole remainder would take the token loop, and a cache hit would be slower than
        // recomputing the prompt from zero.
        int seed = prefill_batched_resume(ids, start, end, want_seed_logprob, &batched_done);
        if (seed >= 0) {
            if (out_pos) *out_pos = end;
            return seed;
        }
    }

    // External KV cache lookup (docs/lmcache_bridge_protocol.md). Deliberately gated on
    // batched_prefill_enabled() having already been tried-and-failed (or never eligible) above:
    // the batched path is ~100x faster than the token loop and only ever eligible from position
    // 0, so restoring a prefix from the bridge only when we were headed for the token loop
    // anyway means a partial hit can never downgrade what would otherwise be a fast batched pass
    // into a slower one for the remainder -- it's a strict win, never a trade-off. Only on the
    // very first call for this range (start==0); a chunked resumption call (start>0, continuing
    // a previous partial token-loop advance) never re-queries -- one round trip per prefill.
    // Deliberately not gated on lmcache_bridge->is_alive(): that only becomes true as a side
    // effect of a prior successful handshake, so gating on it here would mean the very first
    // lookup in a freshly started process never fires (found via the live E2E test,
    // lmcache_e2e_gpu_test.cpp -- exactly the class of bug that test exists to catch).
    // lookup() already handles "is the bridge actually reachable" internally via its own
    // lazy-connect + cooldown/backoff, bounded by the 5ms default timeout either way.
    // Whatever the batched windows did land is already correct in the KV cache and in the GDN
    // state, so the token loop resumes from there instead of recomputing it.
    int actual_start = start + batched_done;
    if (actual_start == 0 && s.lmcache_bridge && n >= lmcache_chunk_size_tokens()) {
        LookupResult res = s.lmcache_bridge->lookup(std::vector<int>(ids, ids + end));
        if (res.ok && res.matched_tokens > 0) {
            const BridgeKVLayout layout = lmcache_layout_from_cfg(s.cfg, *s.kv);
            bool restored = true;
            for (const BridgeKVChunk& c : res.chunks) {
                if (!restore_kv_from_shm(*s.kv, layout, s.active_seq_id, res.shm_name,
                                         c.shm_offset_bytes, c.start_tok, c.len_tok, s.stream)) {
                    restored = false;
                    break;
                }
            }
            // A partial restore failure means some positions in [0, matched_tokens) may hold
            // garbage KV -- never resume the token loop past whatever was verified fully
            // restored, and never resume past a failure point at all: fall back to recomputing
            // the entire range from 0 rather than risk attending against corrupted KV.
            //
            // Clamped to end-1, never end itself: a full hit (matched_tokens == end) would
            // otherwise skip the token loop entirely, and the token loop is what produces the
            // decode seed (the LM-head logits at the last prompt position) -- the KV cache lets
            // attention skip recomputing *past* positions, it doesn't let this function skip
            // computing the *current*/last position's own forward pass, which is what next's
            // value actually comes from. Found via lmcache_bench.cpp's benchmark, whose prompt
            // length happened to land on an exact chunk boundary (matched_tokens == end) --
            // earlier tests never hit this because their prompts weren't exact chunk multiples,
            // so the token loop always had at least one real remaining token regardless.
            if (restored) actual_start = std::min(res.matched_tokens, end - 1);
        }
    }

    int limit = end - actual_start;
    if (chunk_limit > 0 && limit > chunk_limit) limit = chunk_limit;
    const int stop = actual_start + limit;
    int next = -1;
    for (int i = actual_start; i < stop; i++) {
        // Only sample (compute logits/argmax) at the true end of the whole range, not at a
        // chunk boundary -- decode doesn't start until the full prompt is ingested, so
        // intermediate chunk-final tokens never need a logits pass (unless the legacy
        // every-step flag is set).
        const bool sample = prefill_samples_lmhead() || i + 1 == end;
        int r = forward_token(ids[i], i, sample);
        if (sample) next = r;
    }
    if (out_pos) *out_pos = stop;
    return (stop >= end) ? next : -1;
}

bool Qwen35Model::cache_prefix(const std::vector<int>& tokens) {
    TP_MIRROR(cache_prefix(tokens));
    // Runs on the HTTP thread while the continuous-batch worker may be mid-forward_token.
    // Everything below touches the device -- and invalidate_decode_graph() calls
    // cudaGraphExecDestroy/cudaGraphDestroy, which would free the very graph the worker is
    // replaying. Guarded by the same lock capture uses; see device_mutex(). The
    // prefix_exclusive (num_active()==0) check the server does before calling these is a
    // snapshot taken without holding anything, so it does not order against the worker.
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    clear_prefix_cache();
    if (tokens.empty()) return false;
    if (tokens.size() > (size_t)s.cfg.max_seq) return false;
    invalidate_decode_graph();
    if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) return false;
    const int n = (int)tokens.size();
    int next = ingest_prompt_range(tokens.data(), 0, n);
    if (next < 0 || next >= s.cfg.vocab) {
        s.kv->free(s.active_seq_id);
        return false;
    }
    cudaDeviceSynchronize();
    // Snapshot the recurrent state as it stands at the end of the prefix. Every later reuse
    // replays this; without it the hybrid layers would resume from whatever the last request
    // left behind, which is wrong output rather than a crash.
    if (s.cfg.hybrid && s.lin_state && s.lin_conv_state) {
        const size_t st_n = (size_t)gdn_state_slots(s.cfg) * gdn_v_local() *
                            s.cfg.linear_head_dim * s.cfg.linear_head_dim;
        const size_t cv_n = (size_t)s.cfg.n_layers * (s.cfg.linear_conv_kernel - 1) *
                            s.linear_qkvdim;
        if (!s.prefix_lin_state)      s.prefix_lin_state      = s.alloc<float>(st_n);
        if (!s.prefix_lin_conv_state) s.prefix_lin_conv_state = s.alloc<bf16>(cv_n);
        if (!s.prefix_lin_state || !s.prefix_lin_conv_state) {
            s.kv->free(s.active_seq_id);
            return false;
        }
        cu(cudaMemcpyAsync(s.prefix_lin_state, s.lin_state, st_n * sizeof(float),
                           cudaMemcpyDeviceToDevice, s.stream), "prefix state snapshot");
        cu(cudaMemcpyAsync(s.prefix_lin_conv_state, s.lin_conv_state, cv_n * sizeof(bf16),
                           cudaMemcpyDeviceToDevice, s.stream), "prefix conv snapshot");
        cu(cudaStreamSynchronize(s.stream), "prefix snapshot sync");
    }
    s.prefix_tokens = tokens;
    s.prefix_len = (int)tokens.size();
    s.prefix_next = next;
    s.prefix_active = true;
    return true;
}

bool Qwen35Model::restore_prefix_state() {
    TP_MIRROR(restore_prefix_state());
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (!s.prefix_active) return false;
    if (!s.cfg.hybrid) return true;            // nothing recurrent to restore
    if (!s.prefix_lin_state || !s.prefix_lin_conv_state) return false;
    if (!s.lin_state || !s.lin_conv_state) return false;
    const size_t st_n = (size_t)gdn_state_slots(s.cfg) * gdn_v_local() *
                        s.cfg.linear_head_dim * s.cfg.linear_head_dim;
    const size_t cv_n = (size_t)s.cfg.n_layers * (s.cfg.linear_conv_kernel - 1) * s.linear_qkvdim;
    cu(cudaMemcpyAsync(s.lin_state, s.prefix_lin_state, st_n * sizeof(float),
                       cudaMemcpyDeviceToDevice, s.stream), "prefix state restore");
    cu(cudaMemcpyAsync(s.lin_conv_state, s.prefix_lin_conv_state, cv_n * sizeof(bf16),
                       cudaMemcpyDeviceToDevice, s.stream), "prefix conv restore");
    cu(cudaStreamSynchronize(s.stream), "prefix restore sync");
    return true;
}

int Qwen35Model::prefix_block_count() const {
    const Impl& s = *p_;
    if (!s.prefix_active || s.prefix_len <= 0) return 0;
    const int bs = s.kv->block_size();
    return (s.prefix_len + bs - 1) / bs;
}

void Qwen35Model::clear_prefix_cache() {
    TP_MIRROR(clear_prefix_cache());
    // Runs on the HTTP thread while the continuous-batch worker may be mid-forward_token.
    // Everything below touches the device -- and invalidate_decode_graph() calls
    // cudaGraphExecDestroy/cudaGraphDestroy, which would free the very graph the worker is
    // replaying. Guarded by the same lock capture uses; see device_mutex(). The
    // prefix_exclusive (num_active()==0) check the server does before calling these is a
    // snapshot taken without holding anything, so it does not order against the worker.
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (s.prefix_active || s.kv->allocated_tokens(s.active_seq_id) > 0) {
        // Store to the external cache tier before the blocks it reads are freed below -- this
        // is the "prefix cache overwrite" eviction point (docs/lmcache_bridge_protocol.md's
        // STORE trigger list): the active prefix is about to be dropped for a different one.
        if (s.prefix_active)
            lmcache_maybe_store(s.lmcache_bridge, s.cfg, *s.kv, s.stream, s.active_seq_id,
                               s.prefix_tokens, 0, s.prefix_len);
        s.kv->free(s.active_seq_id);
        invalidate_decode_graph();
    }
    s.prefix_tokens.clear();
    s.prefix_len = 0;
    s.prefix_next = -1;
    s.prefix_active = false;
}

void Qwen35Model::release_prefix_session() {
    TP_MIRROR(release_prefix_session());
    // Runs on the HTTP thread while the continuous-batch worker may be mid-forward_token.
    // Everything below touches the device -- and invalidate_decode_graph() calls
    // cudaGraphExecDestroy/cudaGraphDestroy, which would free the very graph the worker is
    // replaying. Guarded by the same lock capture uses; see device_mutex(). The
    // prefix_exclusive (num_active()==0) check the server does before calling these is a
    // snapshot taken without holding anything, so it does not order against the worker.
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    // Caller already freed session-0 KV. Keep the token fingerprint for matching;
    // mark inactive so the next request re-runs cache_prefix() instead of
    // attending against an empty block table.
    s.prefix_next = -1;
    s.prefix_active = false;
    invalidate_decode_graph();
}

int Qwen35Model::prefix_cached_len() const { return p_->prefix_active ? p_->prefix_len : 0; }

int Qwen35Model::prefix_seed_token() const {
    const Impl& s = *p_;
    return (s.prefix_active && s.prefix_next >= 0) ? s.prefix_next : -1;
}

double Qwen35Model::bench_ttft(const std::vector<int>& prompt) {
    Impl& s = *p_;
    if (prompt.empty()) return 0.;
    const bool reuse = prompt_matches_prefix(prompt);
    if (!reuse) {
        clear_prefix_cache();
        invalidate_decode_graph();
        if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) return -1.;
    } else if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) {
        return -1.;
    }
    const int start = reuse ? s.prefix_len : 0;
    if (getenv("SPARKINFER_DEBUG_PREFIX"))
        fprintf(stderr, "[prefix] ttft n=%zu start=%d reuse=%d cached=%d\n",
                prompt.size(), start, (int)reuse, s.prefix_len);
    s.bench_feedback_graph = false;
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    (void)ingest_prompt_range(prompt.data(), start, (int)prompt.size());
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!reuse) {
        s.kv->free(s.active_seq_id);
        invalidate_decode_graph();
        if (s.graph_ready) {
            cu(cudaGraphExecDestroy(s.cu_exec), "ttft graph destroy exec");
            cu(cudaGraphDestroy(s.cu_graph), "ttft graph destroy");
            s.cu_exec = nullptr;
            s.cu_graph = nullptr;
            s.graph_ready = false;
        }
        if (s.graph_prefill_ready) {
            cu(cudaGraphExecDestroy(s.cu_prefill_exec), "ttft prefill graph destroy exec");
            cu(cudaGraphDestroy(s.cu_prefill_graph), "ttft prefill graph destroy");
            s.cu_prefill_exec = nullptr;
            s.cu_prefill_graph = nullptr;
            s.graph_prefill_ready = false;
            s.graph_prefill_attn_mode = -1;
        }
    }
    return std::chrono::duration<double>(t1 - t0).count();
}

// Thin adapter: hand the batched-prefill orchestration (qwen35_prefill.cpp) exactly the scratch
// buffers, streams and config it needs, so Impl stays private to this file.
// Give the NVFP4 LM-head operand back. It exists only to serve a packed decode wide enough to
// want a GEMM, and it is the one piece of weight residency in this model that a run can decide it
// does not need.
// Give the decode shadow's ternary copy back; decode then reads the folded weights, exactly as
// with SPARKINFER_BONSAI_DECODE_SHADOW=0. The decode graphs have its pointers baked in, so they go
// first and recapture on the next step. Returns whether anything was freed.
template <class Impl>
static bool release_bonsai_shadow(Impl& s) {
    if (s.bonsai_dec_bufs.empty()) return false;
    cudaGetLastError();   // clear the failed cudaMalloc that brought us here
    cudaDeviceSynchronize();
    if (s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec); cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr; s.cu_graph = nullptr; s.graph_ready = false;
    }
    if (s.dflash_graph_ready) {
        cudaGraphExecDestroy(s.cu_dflash_exec); cudaGraphDestroy(s.cu_dflash_graph);
        s.cu_dflash_exec = nullptr; s.cu_dflash_graph = nullptr; s.dflash_graph_ready = false;
    }
    s.bonsai_dec_layers.clear();
    s.bonsai_dec_head = nullptr;
    for (void* b : s.bonsai_dec_bufs) cudaFree(b);
    s.bonsai_dec_bufs.clear();
    kernels::ptq1_dp_release();
    fprintf(stderr, "[bonsai] decode shadow released: a new session needed the VRAM\n");
    return true;
}

void Qwen35Model::release_lm_head_fp4() {
    Impl& s = *p_;
    if (!s.lm_head_fp4_payload && !s.lm_head_fp4_sf_buf) return;
    if (s.dflash_draft) s.dflash_draft->set_head_fp4(nullptr, nullptr, 1.f);
    s.w.lm_head_fp4 = nullptr;
    s.w.lm_head_fp4_sf = nullptr;
    if (s.lm_head_fp4_payload) cudaFree(s.lm_head_fp4_payload);
    if (s.lm_head_fp4_sf_buf) cudaFree(s.lm_head_fp4_sf_buf);
    s.lm_head_fp4_payload = nullptr;
    s.lm_head_fp4_sf_buf = nullptr;
}

int Qwen35Model::prefill_batched(const int* prompt_ids, int n, bool want_seed_logprob,
                                 int pos0) {
    TP_MIRROR(prefill_batched(prompt_ids, n, want_seed_logprob, pos0));
    Impl& s = *p_;
    // (dual-GPU WP-14) Tile-aligned split on small-SM cards. The prefill GEMMs' fast arms -- the
    // 4-warp 3-stage int8 kernel and the fused gate/up SwiGLU -- need the row count to be a
    // multiple of their 128-row M tile; one ragged row drops the WHOLE pass to the 2-stage
    // kernel. Measured on the 36-SM RTX 5060 Ti (27B, tp=2): 3072 tokens prefill in 1415 ms,
    // 3074 in 1790 ms. So prefill the aligned bulk in one pass and the < 128-row remainder in a
    // second. That second pass is NOT cheap: every pass has a ~150 ms floor on this card (the
    // per-pass NVFP4 -> int8 weight conversion dominates it; a 2-token pass takes 164 ms), so the
    // split only pays above ~2k tokens -- measured: 3074 1790 -> 1584 ms, 3199 1829 -> 1655,
    // 2000 a wash, 1100 a loss (720 -> 755) -- hence SPARKINFER_PREFILL_ALIGN_MIN = 2048.
    // Padding the GEMMs' M inside one pass would avoid the second pass entirely (a deeper
    // arena change, recorded as follow-up). Arch-keyed: the 170-SM 5090's tuning is
    // untouched (SPARKINFER_PREFILL_ALIGN=1/0 forces it on/off). Not with a pending image or
    // MRoPE table, whose positions are rows of THIS pass. Both tp ranks decide identically (n,
    // SM count and the mirrored pending state agree), and the inner calls are nested, so each
    // rank splits its own mirrored call the same way.
    {
        static const int align_mode = [] {
            const char* e = getenv("SPARKINFER_PREFILL_ALIGN");
            return e ? (e[0] == '0' ? 0 : 1) : -1;
        }();
        static const int align_min = [] {
            const char* e = getenv("SPARKINFER_PREFILL_ALIGN_MIN");
            return e ? std::max(129, atoi(e)) : 2048;
        }();
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, s.device);
        const bool on = align_mode >= 0 ? align_mode == 1 : (sms > 0 && sms <= 64);
        constexpr int kTile = 128;
        if (on && n >= align_min && (n % kTile) != 0 && !s.d_vision_emb && !s.d_mrope_pos) {
            const int bulk = n - n % kTile;
            const int first = prefill_batched(prompt_ids, bulk, false, pos0);
            if (first < 0) return first;
            return prefill_batched(prompt_ids + bulk, n - bulk, want_seed_logprob, pos0 + bulk);
        }
    }
    // A long batched prefill needs the scratch arena more than a wide decode needs the head, and
    // on a 32-GB card the two do not both fit: measured at ctx=32768 the arena wants 3.6 GB and
    // the head operand's 0.81 GB is enough to make it fail, which drops the WHOLE prompt onto the
    // token loop -- a ~30x regression on a no-regression floor, to buy a win that only exists at
    // packed widths >= 16. So hand it back before sizing the arena rather than after failing to.
    //
    // A load-time VRAM check cannot make this call: measured free-at-load is 25.2 GB for the 32k
    // DSpark harness against 22.9 GB for the concurrency harness, i.e. the run that must NOT keep
    // the operand is the one with MORE headroom at the moment of the decision -- the KV cache and
    // the arena are both taken afterwards. The prompt length is the signal that actually
    // separates them.
    //
    // Not at tp=2: there each card holds half the head (0.41 GB), the prefill falls back to smaller
    // windows rather than to the token loop, and without the operand every verify and draft reads
    // the q4k head instead (measured with DFlash2 at 20k: 31.3 -> 28.7 ms a step kept, the 20k
    // prefill unchanged at 5.9 s). An explicit SPARKINFER_Q38_HEAD_NVFP4_YIELD_TOKENS still applies.
    static const char* kHeadFp4YieldEnv = getenv("SPARKINFER_Q38_HEAD_NVFP4_YIELD_TOKENS");
    static const int kHeadFp4YieldTokens = [] {
        const int v = kHeadFp4YieldEnv ? atoi(kHeadFp4YieldEnv) : 1024;
        return v < 1 ? 1 : v;
    }();
    const bool head_yield = kHeadFp4YieldEnv || !s.tp_link;
    if (head_yield && n >= kHeadFp4YieldTokens && (s.lm_head_fp4_payload || s.lm_head_fp4_sf_buf)) {
        fprintf(stderr, "[compressed-tensors] NVFP4 lm_head released for a %d-token batched "
                        "prefill (the scratch arena needs the VRAM more)\n", n);
        release_lm_head_fp4();
    }
    // Batched prefill is position-0 only and writes the whole recurrent state as fp32, so a
    // session that had been compacted is back on the fp32 form afterwards.
    {
        auto sit = s.sessions.find(s.active_seq_id);
        if (sit != s.sessions.end()) sit->second.lin_state_b16 = false;
        s.active_lin_state_b16 = false;
    }
    auto it = s.sessions.find(s.active_seq_id);
    float* lin_state = (it != s.sessions.end()) ? it->second.lin_state : s.lin_state;
    bf16* lin_conv = (it != s.sessions.end()) ? it->second.lin_conv_state : s.lin_conv_state;
    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, s.active_seq_id,
                          lin_state, lin_conv,
                          s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.bonsai_embed_native,
                          s.bonsai_sign_dev.count(s.cfg.hidden)
                              ? s.bonsai_sign_dev.at(s.cfg.hidden) : nullptr,
                          s.bonsai_sign_ffn,
                          (int)s.bonsai_block,
                          s.bonsai_rot,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits,
                          s.dflash_capture ? s.dflash_layer_ids.data() : nullptr,
                          s.dflash_capture ? s.dflash_n_cap : 0,
                          s.dflash_capture ? s.dflash_context : nullptr,
                          s.dflash_capture ? s.dflash_ctx_start : 0,
                          // Image input, if set_pending_vision was called for this prompt.
                          // Only prefill_batched gets these: the other two Qwen35PrefillCtx
                          // sites are DSpark verify paths, which replay already-generated
                          // tokens and can never contain an image placeholder.
                          s.d_vision_emb, s.d_vision_pos, s.vision_n,
                          // MRoPE rotary positions, null unless set_pending_mrope ran for this prompt.
                          s.d_mrope_pos };
    ctx.gdn_window = s.gdn_window;
    ctx.gdn_scratch = s.gdn_scratch;
    ctx.capture_off = s.dflash_cap_off;
    ctx.capture_h = s.dflash_cap_h;
    // The Bonsai decode shadow's VRAM is decode-only -- this pass reads the folded Q4_K weights
    // either way -- so on a scratch-alloc failure it is pure margin to give back and retry once.
    // #1154 was rejected for exactly the failure this skips: at a long --ctx the KV pool leaves
    // too little room for prefill's own arena once the shadow is also resident, every batched
    // prefill falls back to the ~30x-slower token loop, and nothing ever recovered because
    // release_bonsai_shadow only ran for a new session's state, never for this.
    bool scratch_oom = false;
    ctx.scratch_oom_out = &scratch_oom;
    int seed = prefill_batched_run(ctx, prompt_ids, n, pos0);
    if (seed < 0 && scratch_oom && release_bonsai_shadow(s)) {
        scratch_oom = false;
        seed = prefill_batched_run(ctx, prompt_ids, n, pos0);
    }
    // Consume it. This is PER-REQUEST state, not model state: leaving it set would splice the
    // previous request's image into the next prompt, which would produce fluent, confident text
    // about an image the caller never sent. Cleared on every path out, including the failure
    // path below, because a prefill that returned -1 has still consumed the staging.
    clear_pending_vision();
    if (seed >= 0 && s.dflash_capture && s.dflash_context && s.dflash_n_cap > 0)
        s.dflash_ctx_len = pos0 + n;
    // logit_bias applies to the FIRST token too. prefill_batched_run's argmax runs on the raw
    // logits -- inside the prefill graph, which does not key on the per-session bias buffer -- so
    // the first token of every batched-prefill response ignored logit_bias: a -100 could not keep
    // a token out of position 0, and a +100 could not put one there. forward_token() adds the bias
    // before its argmax; do the same here and re-pick the seed. Outside the capture (a host `if` is
    // legal here, see below), before the seed logprob so that describes the biased distribution
    // exactly as forward_token()'s entries do, and skipped for requests that set no bias.
    if (seed >= 0 && s.logit_bias_set) {
        kernels::launch_logit_bias(s.logits, s.logit_bias, s.cfg.vocab, s.stream);
        kernels::launch_argmax(s.logits, s.d_out_id, 1, s.cfg.vocab, s.stream);
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, s.stream),
           "prefill seed logit_bias readback");
        cu(cudaStreamSynchronize(s.stream), "prefill seed logit_bias sync");
        seed = *s.h_out_id;
    }
    // Seed-token logprob (see ingest_prompt_range's want_seed_logprob). prefill_batched_run's
    // LM-head tail stops at argmax -- it never runs the sort/scan that last_token_logprobs()
    // reads -- so without this the FIRST token of every response has no logprob entry and
    // `logprobs.content` comes back one short of usage.completion_tokens.
    //
    // Deliberately here in the adapter rather than inside prefill_batched_run: this is the one
    // point both of that function's exits pass through (the full pass AND the CUDA-graph replay
    // fast path), it needs Impl's sampler scratch which Qwen35PrefillCtx does not carry, and it
    // must stay OUTSIDE the prefill graph capture -- the capture is closed by the time control
    // returns here, so a plain host-side `if` is legal, which it would not be inside the
    // captured region.
    //
    // s.logits still holds this call's last-position logits and d_out_id the argmax of them:
    // both are dedicated Impl buffers, not arena scratch, so the arena release at the end of
    // prefill_batched_run cannot have clobbered them. The distribution is computed from those
    // logits UNCHANGED, so the seed token reported here is exactly the token already chosen --
    // this reports, it never re-decides.
    if (seed >= 0 && want_seed_logprob) {
        kernels::launch_logprob_distribution(s.logits, s.cfg.vocab, s.d_vocab_iota,
                                             s.d_sorted_logits, s.d_sorted_idx, s.d_topk_exp,
                                             s.d_topk_cumsum, s.d_sort_temp, s.sort_temp_bytes,
                                             s.d_scan_temp, s.scan_temp_bytes, s.d_rank_by_id,
                                             s.stream);
        kernels::launch_extract_chosen_logit(s.d_out_id, s.d_rank_by_id, s.d_sorted_logits,
                                             s.d_chosen_logit, s.stream);
        // last_token_logprobs() reads this scratch with blocking cudaMemcpy on the legacy default
        // stream, which is not ordered against s.stream -- sync here, exactly as forward_token()
        // does before returning, so that contract ("valid after the call returns") holds.
        cu(cudaStreamSynchronize(s.stream), "prefill seed logprob sync");
    }
    return seed;
}

bool Qwen35Model::ingest_prompts_packed(const uint64_t* seq_ids, const int* const* prompts,
                                        const int* lens, int n_prompts, int* seeds) {
    std::vector<int> tp_peer_seeds(n_prompts > 0 ? n_prompts : 0);
    TP_MIRROR(ingest_prompts_packed(seq_ids, prompts, lens, n_prompts,
                                    seeds ? tp_peer_seeds.data() : nullptr));
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (!seq_ids || !prompts || !lens || !seeds || n_prompts < 2) return false;
    if (!s.gguf || !s.cfg.hybrid || !s.cfg.dense_ffn || s.cfg.muse_glimmer) return false;
    // Per-request staging belongs to exactly one prompt, so a pack cannot carry it.
    if (s.dflash_capture || s.d_vision_emb || s.d_mrope_pos) return false;
    std::vector<int> off((size_t)n_prompts);
    std::vector<float*> lin_state((size_t)n_prompts);
    std::vector<void*> lin_conv((size_t)n_prompts);
    int total = 0;
    for (int i = 0; i < n_prompts; ++i) {
        auto it = s.sessions.find(seq_ids[i]);
        if (it == s.sessions.end() || !prompts[i] || lens[i] <= 0) return false;
        // The pass's seed is a raw argmax; a session with a logit bias needs prefill_batched's re-pick.
        if (!it->second.lin_state || !it->second.lin_conv_state || it->second.logit_bias_set)
            return false;
        off[(size_t)i] = total;
        total += lens[i];
        lin_state[(size_t)i] = it->second.lin_state;
        lin_conv[(size_t)i] = it->second.lin_conv_state;
    }
    if (!batched_prefill_windowed_enabled(s.gguf, s.cfg, total, s.kv) ||
        total > prefill_single_pass_max_tokens(s.kv))
        return false;
    std::vector<int> ids;
    ids.reserve((size_t)total);
    for (int i = 0; i < n_prompts; ++i) ids.insert(ids.end(), prompts[i], prompts[i] + lens[i]);
    for (int i = 0; i < n_prompts; ++i) {
        // Position-0 batched prefill writes the whole recurrent state as fp32 (see prefill_batched).
        s.sessions[seq_ids[i]].lin_state_b16 = false;
        if (s.active_seq_id == seq_ids[i]) s.active_lin_state_b16 = false;
        seeds[i] = -1;
    }
    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, seq_ids[0],
                          lin_state[0], lin_conv[0],
                          s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.bonsai_embed_native,
                          s.bonsai_sign_dev.count(s.cfg.hidden)
                              ? s.bonsai_sign_dev.at(s.cfg.hidden) : nullptr,
                          s.bonsai_sign_ffn,
                          (int)s.bonsai_block,
                          s.bonsai_rot,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits,
                          nullptr, 0, nullptr, 0 };
    ctx.gdn_window = s.gdn_window;
    ctx.gdn_scratch = s.gdn_scratch;
    ctx.multi_n = n_prompts;
    ctx.multi_off = off.data();
    ctx.multi_len = lens;
    ctx.multi_seq_ids = seq_ids;
    ctx.multi_lin_state = lin_state.data();
    ctx.multi_lin_conv = lin_conv.data();
    ctx.multi_seed = seeds;
    if (prefill_batched_run(ctx, ids.data(), total, 0) < 0) return false;
    for (int i = 0; i < n_prompts; ++i)
        if (seeds[i] < 0 || seeds[i] >= s.cfg.vocab) return false;
    return true;
}

// Narrowest window prefill_batched_chunked retries at after a scratch decline.
static constexpr int kMinRetryWindow = 256;

int Qwen35Model::prefill_batched_chunked(const int* prompt_ids, int n, bool want_seed_logprob,
                                         int* out_done) {
    int tp_peer_done = 0;
    TP_MIRROR(prefill_batched_chunked(prompt_ids, n, want_seed_logprob,
                                      out_done ? &tp_peer_done : nullptr));
    if (out_done) *out_done = 0;
    if (!prompt_ids || n <= 0) return -1;
    Impl& s = *p_;
    int window = prefill_window_tokens(s.kv);
    // Short enough for one pass: run exactly the call this function replaced, so no context that
    // already worked changes kernel path, tile shape or arithmetic. The bound is the single-pass
    // threshold, NOT the window size -- a prompt between the two is still one pass.
    if (window <= 0 || n <= prefill_single_pass_max_tokens(s.kv)) {
        const int seed = prefill_batched(prompt_ids, n, want_seed_logprob);
        if (seed >= 0 && seed < s.cfg.vocab) {
            if (out_done) *out_done = n;
            return seed;
        }
        // The single pass declined. Returning here sends the WHOLE prompt to the token loop, and
        // at n == prefill_single_pass_max_tokens() that is exactly what happens in a session whose
        // KV cache is sized for a longer context: the single-pass arena cannot be allocated beside
        // it, while the windowed path -- which the very next context up already uses successfully
        // -- allocates one window and runs. Measured on Muse Glimmer in one model load over
        // 16k/32k/64k (what bench_sweep_run does): prefill@32k 923 pp against 11018 at 16k and
        // 8045 at 64k, i.e. the boundary itself, not the length. Windowing it gives 9629 pp.
        //
        // Try windowing before giving up. It restarts from position 0, so a single pass that had
        // already written part of the cache is simply recomputed -- prefill is deterministic in
        // the prompt, and pos0 == 0 resets the recurrent state the same way the original call did.
        // Nothing here changes a context where the single pass succeeds: that returns above.
        // (dual-GPU) A prompt shorter than the window used to stop here and go to the token
        // loop. A pass that cannot get its scratch declines before touching any state, so a
        // NARROWER window is a clean retry: half the prompt, rounded to the 128-row GEMM tile.
        // Measured: a card carrying the DSpark draft at tp=2 cannot fit a 4k-token arena (the
        // token loop then took 70 s), while a 2k window fits.
        if (window <= 0 || n <= window) {
            const int half = ((n / 2 + 127) / 128) * 128;
            if (half < kMinRetryWindow || half >= n) return -1;
            window = half;
        }
        fprintf(stderr, "[prefill] single pass declined at n=%d -- windowing (%d) instead of the "
                        "token loop\n", n, window);
    }
    int pos = 0;
    while (pos < n) {
        const int len = std::min(window, n - pos);
        const bool last = (pos + len >= n);
        // Only the final window needs the seed logprob -- the earlier ones exist to fill KV and
        // carry the Gated-DeltaNet recurrence, and nothing ever reads their argmax.
        const int seed = prefill_batched(prompt_ids + pos, len, want_seed_logprob && last, pos);
        // Scratch too large even at this width: halve the window and retry from the same position
        // (a declined pass changed nothing). Both tp ranks decline together (the arena check is
        // agreed), so they halve together.
        if (seed < 0 && window > kMinRetryWindow) {
            window = std::max(kMinRetryWindow, ((window / 2 + 127) / 128) * 128);
            fprintf(stderr, "[prefill] window declined at pos=%d -- retrying at %d\n", pos, window);
            continue;
        }
        // A window that declines -- a path with no start position (Muse's rolling-window
        // attention, a DSpark capture), or a scratch allocation that failed even at window size
        // -- leaves [0, pos) correct in the cache and stops there: the caller finishes the rest
        // token by token rather than recomputing what already landed.
        if (seed < 0) return -1;
        // *out_done advances only over windows whose result is trustworthy: a final window that
        // produced a nonsense seed is not counted, so the caller redoes it rather than trusting
        // a KV range whose own LM head disagreed with itself.
        if (last) {
            if (seed >= s.cfg.vocab) return -1;
            if (out_done) *out_done = n;
            return seed;
        }
        if (out_done) *out_done = pos + len;
        pos += len;
    }
    return -1;
}

int Qwen35Model::session_token_budget(size_t prompt_len, int max_new, int max_seq) {
    const long need = (long)prompt_len + max_new + 16;
    if (need <= 0) return 16;
    if (need > max_seq) return max_seq;
    return (int)need;
}

uint64_t Qwen35Model::open_session(int num_tokens, bool* alloc_failed,
                                   const std::vector<int>* shared_prefix_blocks) {
    // (dual-GPU) Both ranks open the session; their per-model id counters advance in lockstep, so
    // the ids agree. If one rank could not open (its pool differs) or the ids diverge, undo the
    // side that did open -- leader-only, since the other rank does not know that id -- and fail.
    if (Qwen35Model* peer = tp_mirror_peer()) {
        uint64_t peer_id = 0, id = 0;
        bool peer_alloc_failed = false;
        {
            TpMirrorScope m(&p_->device_mu, peer, [&](Qwen35Model& pm) {
                peer_id = pm.open_session(num_tokens, &peer_alloc_failed, shared_prefix_blocks);
            });
            id = open_session(num_tokens, alloc_failed, shared_prefix_blocks);
        }
        if (id == peer_id) return id;
        fprintf(stderr, "[tp] open_session: ranks disagree (rank0 id %llu, rank1 id %llu); undoing\n",
                (unsigned long long)id, (unsigned long long)peer_id);
        if (id) {
            TpMirrorScope solo(nullptr, nullptr, [](Qwen35Model&) {});
            close_session(id);
        }
        if (peer_id) {
            TpMirrorScope m(&p_->device_mu, peer, [&](Qwen35Model& pm) { pm.close_session(peer_id); });
        }
        if (alloc_failed && peer_alloc_failed) *alloc_failed = true;
        return 0;
    }
    // cudaMalloc + kv->allocate (block-table copy on the legacy stream). submit_locked already
    // holds this lock around its call, but guard here too so a future caller cannot miss it --
    // the mutex is recursive, so the nested acquisition is free.
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (num_tokens <= 0) return 0;
    const uint64_t seq_id = s.next_session_id.fetch_add(1);
    const bool kv_ok = (shared_prefix_blocks && !shared_prefix_blocks->empty())
                           ? s.kv->allocate_with_prefix(seq_id, *shared_prefix_blocks, num_tokens)
                           : s.kv->allocate(seq_id, num_tokens);
    if (!kv_ok) return 0;   // pool full -- normal, transient
    SessionBuffers buf;
    bool alloc_ok = false;
    for (int attempt = 0;; ++attempt) {
        buf = SessionBuffers{};
        // Unconditional, every model -- unlike lin_state/lin_conv_state below (hybrid-only). This
        // fresh session serves exactly one request end-to-end before close_session() frees it (1:1
        // lifecycle via ContinuousBatchEngine::finish_job), so a one-time zero here is sufficient
        // -- unlike session 0, which is reused across many DIFFERENT requests and needs an explicit
        // per-request reset (see reset_penalty_counts()).
        buf.penalty_counts = s.alloc<int>(s.cfg.vocab);
        buf.logit_bias = s.alloc<float>(s.cfg.vocab);
        alloc_ok = buf.penalty_counts != nullptr && buf.logit_bias != nullptr;
        // Per-session recurrent state, and it is not small: n_layers * v_heads * head_dim^2 floats
        // is 109 MB on Muse Glimmer's 52 layers. That model declares `hybrid` but has no GDN layer
        // at all, so every concurrent request was reserving 109 MB for a recurrence it never runs
        // -- 3.6 GB at 32 requests, on a card whose FP4 prefill operands already leave it with a
        // few hundred MB of headroom. Ask whether the stack has the layers, not whether it has the
        // flag.
        if (needs_linear_state(s.cfg)) {
            buf.lin_state = s.alloc<float>((size_t)gdn_state_slots(s.cfg) * gdn_v_local() *
                                           s.cfg.linear_head_dim * s.cfg.linear_head_dim);
            buf.lin_conv_state = s.alloc<bf16>((size_t)s.cfg.n_layers *
                                               (s.cfg.linear_conv_kernel - 1) * s.linear_qkvdim);
            alloc_ok = alloc_ok && buf.lin_state && buf.lin_conv_state;
        }
        if (alloc_ok) break;
        if (buf.penalty_counts) cudaFree(buf.penalty_counts);
        if (buf.logit_bias) cudaFree(buf.logit_bias);
        if (buf.lin_state) cudaFree(buf.lin_state);
        if (buf.lin_conv_state) cudaFree(buf.lin_conv_state);
        buf = SessionBuffers{};
        // The decode shadow is a cache of weights decode can also read folded: a request that
        // cannot get its state takes the shadow's VRAM, once, rather than failing.
        if (attempt == 0 && release_bonsai_shadow(s)) continue;
        break;
    }
    if (!alloc_ok) {
        // A real cudaMalloc failure (already logged by alloc<T>'s cu() wrapper as
        // "[qwen35] malloc: out of memory") -- not the KV pool being full, which was already
        // checked above. Surfaced separately so callers can tell "genuinely no capacity right
        // now" (retry later) apart from "device is out of memory" (permanent until restart).
        if (alloc_failed) *alloc_failed = true;
        s.kv->free(seq_id);
        if (buf.penalty_counts) cudaFree(buf.penalty_counts);
        if (buf.logit_bias) cudaFree(buf.logit_bias);
        if (buf.lin_state) cudaFree(buf.lin_state);
        if (buf.lin_conv_state) cudaFree(buf.lin_conv_state);
        return 0;
    }
    // Explicit zero, not relying on alloc<T>'s (plain cudaMalloc) zeroing guarantee -- unlike
    // lin_state/lin_conv_state, which are written wholesale before ever being read (GDN state is
    // computed fresh as the prefill/decode loop processes this session's own tokens), a stale,
    // nonzero penalty count would be silently, incorrectly wrong from token 1.
    cu(cudaMemsetAsync(buf.penalty_counts, 0, (size_t)s.cfg.vocab * sizeof(int), s.stream),
       "penalty_counts zero");
    // logit_bias: same one-time-zero-is-sufficient reasoning as penalty_counts above (1:1 fresh-
    // session lifecycle) -- the actual per-request VALUES are set later by set_logit_bias(), called
    // from submit_locked() right after this open_session() returns.
    cu(cudaMemsetAsync(buf.logit_bias, 0, (size_t)s.cfg.vocab * sizeof(float), s.stream),
       "logit_bias zero");
    s.sessions[seq_id] = buf;
    return seq_id;
}

int Qwen35Model::prefill_batched_resume(const int* prompt_ids, int start, int end,
                                        bool want_seed_logprob, int* out_done) {
    int tp_peer_done = 0;
    TP_MIRROR(prefill_batched_resume(prompt_ids, start, end, want_seed_logprob,
                                     out_done ? &tp_peer_done : nullptr));
    if (out_done) *out_done = 0;
    if (!prompt_ids || start <= 0 || end <= start) return -1;
    Impl& s = *p_;
    const int n = end - start;
    const int window = prefill_window_tokens(s.kv);
    int done = 0;
    constexpr int kDeclined = -2;   // a pass could not get its scratch; nothing of it landed
    auto run = [&](int step) -> int {
        for (int pos = start + done; pos < end; pos += step) {
            const int len = std::min(step, end - pos);
            const bool last = (pos + len >= end);
            const int seed = prefill_batched(prompt_ids + pos, len, want_seed_logprob && last, pos);
            if (seed < 0) return kDeclined;
            if (last) {
                if (seed >= s.cfg.vocab) return -1;
                done = n;
                return seed;
            }
            done = pos + len - start;
        }
        return -1;
    };
    // Same split as prefill_batched_chunked: one pass up to the single-pass threshold, windows
    // above it. Every pass here has pos0 > 0, so each runs eager and carries the recurrence forward.
    const bool single = window <= 0 || n <= prefill_single_pass_max_tokens(s.kv);
    int step = single ? n : window;
    int seed = run(step);
    // ...and the same retry when the single pass declines. A cached prefix followed by a long
    // continuation is what an agent sends after a large tool result, and on serve-dspark at
    // --ctx 131072 the 31K-token pass's scratch arena did not fit (free=31 MB): the WHOLE
    // continuation went to the token loop, minutes instead of seconds (#1088). A pass declines
    // before its first kernel runs -- see the pos0 != 0 note in prefill_batched_run -- so
    // nothing of it landed, and windowing from `start` continues the same state.
    if (seed == kDeclined && single && done == 0 && window > 0 && n > window) {
        fprintf(stderr, "[prefill] resumed single pass declined at n=%d -- windowing (%d) instead "
                        "of the token loop\n", n, window);
        step = window;
        seed = run(step);
    }
    // A pass that still declines -- a continuation shorter than the window, or a window too wide
    // for what the card has left -- used to send the rest to the token loop: at --ctx 131072 with
    // the DSpark draft on both cards, a 5.4k-token continuation after a prefix hit took 102 s
    // instead of ~2. Halve and retry from where the landed passes stopped, as
    // prefill_batched_chunked does (both tp ranks decline together, so they halve together).
    while (seed == kDeclined && step > kMinRetryWindow) {
        const int rest = n - done;
        const int next = std::max(kMinRetryWindow, ((std::min(step, rest) / 2 + 127) / 128) * 128);
        if (next >= step) break;
        fprintf(stderr, "[prefill] resumed pass declined at pos=%d -- retrying at %d\n",
                start + done, next);
        step = next;
        seed = run(step);
    }
    if (seed < 0) seed = -1;
    if (out_done) *out_done = done;
    return seed;
}

bool Qwen35Model::snapshot_recurrent_state(uint64_t seq_id, RecurrentStateSnapshot& out) {
    // (dual-GPU) Each rank holds only its own GDN window: the leader snapshots its half into
    // `out` and rank 1's (on rank 1's worker/device) into out.peer; both must succeed.
    if (Qwen35Model* peer = tp_mirror_peer()) {
        auto ps = std::make_shared<RecurrentStateSnapshot>();
        bool ok = false, peer_ok = false;
        {
            TpMirrorScope m(&p_->device_mu, peer, [&](Qwen35Model& pm) {
                peer_ok = pm.snapshot_recurrent_state(seq_id, *ps);
            });
            ok = snapshot_recurrent_state(seq_id, out);
        }
        if (!ok || !peer_ok) { out = RecurrentStateSnapshot{}; return false; }
        out.peer = std::move(ps);
        return true;
    }
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (!needs_linear_state(s.cfg)) {
        out = RecurrentStateSnapshot{};
        return true;
    }
    auto it = s.sessions.find(seq_id);
    if (seq_id == 0 || it == s.sessions.end()) return false;
    const SessionBuffers& b = it->second;
    if (!b.lin_state || !b.lin_conv_state || b.lin_state_b16) return false;
    const size_t st_bytes = (size_t)gdn_state_slots(s.cfg) * gdn_v_local() * s.cfg.linear_head_dim *
                            s.cfg.linear_head_dim * sizeof(float);
    const size_t cv_bytes = (size_t)s.cfg.n_layers * (s.cfg.linear_conv_kernel - 1) *
                            s.linear_qkvdim * sizeof(bf16);
    void* host = nullptr;
    if (cudaHostAlloc(&host, st_bytes + cv_bytes, cudaHostAllocDefault) != cudaSuccess || !host)
        return false;
    std::shared_ptr<void> owned(host, [](void* p) { cudaFreeHost(p); });
    // Prefill may still have work queued on any of the model's streams; the state is final only
    // once all of it has run.
    cudaDeviceSynchronize();
    cu(cudaMemcpyAsync(host, b.lin_state, st_bytes, cudaMemcpyDeviceToHost, s.stream),
       "prefix-cache state snapshot");
    cu(cudaMemcpyAsync(static_cast<char*>(host) + st_bytes, b.lin_conv_state, cv_bytes,
                       cudaMemcpyDeviceToHost, s.stream), "prefix-cache conv snapshot");
    if (cudaStreamSynchronize(s.stream) != cudaSuccess) return false;
    out.host = std::move(owned);
    out.state_bytes = st_bytes;
    out.conv_bytes = cv_bytes;
    return true;
}

bool Qwen35Model::restore_recurrent_state(uint64_t seq_id, const RecurrentStateSnapshot& snap) {
    // (dual-GPU) Restore both halves (see snapshot_recurrent_state); a snapshot without the peer
    // half (taken at tp=1) cannot restore a split model.
    if (Qwen35Model* peer = tp_mirror_peer()) {
        if (!snap.peer) return false;
        bool ok = false, peer_ok = false;
        {
            TpMirrorScope m(&p_->device_mu, peer, [&](Qwen35Model& pm) {
                peer_ok = pm.restore_recurrent_state(seq_id, *snap.peer);
            });
            ok = restore_recurrent_state(seq_id, snap);
        }
        return ok && peer_ok;
    }
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (!needs_linear_state(s.cfg)) return true;
    auto it = s.sessions.find(seq_id);
    if (seq_id == 0 || it == s.sessions.end()) return false;
    SessionBuffers& b = it->second;
    const size_t st_bytes = (size_t)gdn_state_slots(s.cfg) * gdn_v_local() * s.cfg.linear_head_dim *
                            s.cfg.linear_head_dim * sizeof(float);
    const size_t cv_bytes = (size_t)s.cfg.n_layers * (s.cfg.linear_conv_kernel - 1) *
                            s.linear_qkvdim * sizeof(bf16);
    if (!snap.host || snap.state_bytes != st_bytes || snap.conv_bytes != cv_bytes ||
        !b.lin_state || !b.lin_conv_state)
        return false;
    cu(cudaMemcpyAsync(b.lin_state, snap.host.get(), st_bytes, cudaMemcpyHostToDevice, s.stream),
       "prefix-cache state restore");
    cu(cudaMemcpyAsync(b.lin_conv_state, static_cast<char*>(snap.host.get()) + st_bytes, cv_bytes,
                       cudaMemcpyHostToDevice, s.stream), "prefix-cache conv restore");
    if (cudaStreamSynchronize(s.stream) != cudaSuccess) return false;
    // The snapshot is the fp32 form a prefill writes, whatever this session's buffer held before.
    b.lin_state_b16 = false;
    if (s.active_seq_id == seq_id) s.active_lin_state_b16 = false;
    return true;
}

void Qwen35Model::set_lmcache_bridge(BridgeClient* bridge) { p_->lmcache_bridge = bridge; }

void Qwen35Model::close_session(uint64_t seq_id, const std::vector<int>* store_tokens) {
    TP_MIRROR(close_session(seq_id, nullptr));   // the LMCache store is leader-only
    // Device work (lmcache store reads KV, kv->free, buffer frees) reachable from the worker
    // while the HTTP thread may be in clear_prefix_cache()/cache_prefix(). Same lock as capture;
    // see device_mutex(). Recursive, so nesting under an already-held lock is fine.
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    if (seq_id == 0) return;
    // Drop this session's parked decode graph before the buffers its nodes bake are freed below.
    {
        auto parked = s.parked_graphs.find(seq_id);
        if (parked != s.parked_graphs.end()) {
            if (parked->second.exec) cudaGraphExecDestroy(parked->second.exec);
            if (parked->second.graph) cudaGraphDestroy(parked->second.graph);
            s.parked_graphs.erase(parked);
        }
    }
    // The packed row-set cache is keyed on seq_ids; a closing session frees the very buffers those
    // device arrays point at, and a later session can be handed the same id. Force the next packed
    // step to re-upload rather than trust a match.
    s.packed_rows_valid = 0;
    // Store to the external cache tier before freeing the blocks it reads -- this is the
    // "session close" eviction point (docs/lmcache_bridge_protocol.md's STORE trigger list).
    // store_tokens is null for most callers (this model class doesn't itself track a session's
    // original prompt; only ContinuousBatchEngine::finish_job has that context) -- a null
    // pointer is a pure no-op here, not a degraded path.
    if (store_tokens && !store_tokens->empty())
        lmcache_maybe_store(s.lmcache_bridge, s.cfg, *s.kv, s.stream, seq_id, *store_tokens, 0,
                           (int)store_tokens->size());
    s.kv->free(seq_id);
    auto it = s.sessions.find(seq_id);
    if (it != s.sessions.end()) {
        // open_session() only ever stores a freshly cudaMalloc'd, uniquely-owned buffer here
        // (seq_id == 0 -- the one case that could alias the model's persistent default buffers
        // -- returns before this point, see the guard at the top of this function). A `!=
        // s.lin_state` check used to gate these frees, but activate_session(seq_id) (called at
        // the top of every ContinuousBatchEngine::step_job, before finish_job/close_session runs
        // later in that same call) unconditionally aliases s.lin_state to *this* session's own
        // buffer -- so by the time close_session() ran, the two pointers were always equal and
        // the free was always skipped. Every hybrid-model (Muse Glimmer, Qwen3.6) request via the
        // server leaked its full lin_state/lin_conv_state allocation, permanently, confirmed live
        // as a fixed ~108 MiB/request leak independent of token count (#779).
        if (it->second.lin_state) cudaFree(it->second.lin_state);
        if (it->second.lin_conv_state) cudaFree(it->second.lin_conv_state);
        // Unconditional, every model -- mirrors the lin_state/lin_conv_state free above exactly.
        // The seq_id == 0 guard at the top of this function already protects
        // penalty_counts_default from ever being freed here, same aliasing-safety story as #779.
        if (it->second.penalty_counts) cudaFree(it->second.penalty_counts);
        // Unconditional, every model -- mirrors penalty_counts's free exactly, same aliasing-safety
        // story (the seq_id == 0 guard above already protects logit_bias_default).
        if (it->second.logit_bias) cudaFree(it->second.logit_bias);
        s.sessions.erase(it);
    }
    if (s.active_seq_id == seq_id) activate_session(0);
}

int Qwen35Model::max_packed_rows() {
    // Runtime cap, so the widest packed batch can be A/B'd in ONE binary. The compile-time
    // kQwen35MaxPackedRows still sizes every array; this only bounds what a step will pack.
    static const int cap = [] {
        const char* e = getenv("SPARKINFER_PACKED_MAX_ROWS");
        int v = e ? atoi(e) : kQwen35MaxPackedRows;
        if (v < 1) v = 1;
        return v > kQwen35MaxPackedRows ? kQwen35MaxPackedRows : v;
    }();
    return cap;
}

bool Qwen35Model::decode_packed(const int* tokens, const int* positions,
                                const uint64_t* seq_ids, int n, int* out_sampled,
                                const SpecSampleRow* row_sample) {
    // (dual-GPU WP-9) tp=2: same split-weights story as forward_token above -- the step runs on
    // the per-rank twin instead; this guard is the ONLY tp>1 delta on the tp=1 packed path.
    if (tp_active()) {
        std::vector<int> tp_peer_out(n > 0 ? n : 0);
        TP_MIRROR(decode_packed(tokens, positions, seq_ids, n, tp_peer_out.data(), row_sample));
        return decode_packed_tp(tokens, positions, seq_ids, n, out_sampled, row_sample);
    }
    // Sampled rows pack at tp=2 only (the rows pass's sampled draw).
    if (row_sample)
        for (int i = 0; i < n; i++)
            if (row_sample[i].temperature > 0.f) return false;
    Impl& s = *p_;
    if (!tokens || !positions || !seq_ids || !out_sampled) return false;
    if (n < 1 || n > kQwen35MaxPackedRows) return false;
    if (!s.cfg.hybrid || !s.gguf) return false;
    std::lock_guard<std::recursive_mutex> device_lock(s.device_mu);

    // Resolve each row's per-session buffers. A row whose session is missing (or never got its
    // hybrid state) cannot be packed -- decline the whole batch rather than silently decode it
    // against another request's state.
    // PINNED staging, allocated once. The upload below must be async -- a synchronous copy, or an
    // async one from a stack array that has to be waited on, forces the CPU to drain the GPU every
    // step, and the engine's per-row callbacks then run with the device idle. Measured at
    // concurrency 8 that stall was ~4.5 ms of a ~24 ms step, against ~19.5 ms of actual GPU work.
    if (!s.packed_dev_states) {
        const size_t np = kQwen35MaxPackedRows;
        if (cudaMalloc(&s.packed_dev_states, np * sizeof(float*)) != cudaSuccess) return false;
        if (cudaMalloc(&s.packed_dev_convs, np * sizeof(void*)) != cudaSuccess) return false;
        if (cudaMalloc(&s.packed_dev_tables, np * sizeof(const int*)) != cudaSuccess) return false;
        if (cudaMalloc(&s.packed_dev_tables_win, np * sizeof(const int*)) != cudaSuccess) return false;
        if (cudaHostAlloc(&s.packed_host_tables_win, np * sizeof(const int*), cudaHostAllocDefault)
            != cudaSuccess) return false;
        if (cudaHostAlloc(&s.packed_host_states, np * sizeof(float*), cudaHostAllocDefault)
            != cudaSuccess) return false;
        if (cudaHostAlloc(&s.packed_host_convs, np * sizeof(void*), cudaHostAllocDefault)
            != cudaSuccess) return false;
        if (cudaHostAlloc(&s.packed_host_tables, np * sizeof(const int*), cudaHostAllocDefault)
            != cudaSuccess) return false;
        if (cudaHostAlloc(&s.packed_host_seqs, np * sizeof(uint64_t), cudaHostAllocDefault)
            != cudaSuccess) return false;
    }
    float** h_states = static_cast<float**>(s.packed_host_states);
    void**  h_convs  = static_cast<void**>(s.packed_host_convs);
    const int** h_tables = static_cast<const int**>(s.packed_host_tables);
    const int** h_tables_win = static_cast<const int**>(s.packed_host_tables_win);
    uint64_t* h_seqs = static_cast<uint64_t*>(s.packed_host_seqs);

    // Skip the upload entirely when the row set has not moved. Each session's buffers and its
    // block-table slab row are fixed for its lifetime, so the same seq_ids imply the same three
    // arrays -- and a steady stream of concurrent requests decodes the SAME set for many steps in
    // a row.
    bool same = (s.packed_rows_valid == n);
    for (int i = 0; i < n && same; i++) same = h_seqs[i] == seq_ids[i];

    for (int i = 0; i < n; i++) {
        auto it = s.sessions.find(seq_ids[i]);
        if (it == s.sessions.end()) return false;
        // Only a stack that HAS the layers has the buffers; demanding them on one that does not
        // refuses every row for a state the model never carried.
        if (needs_linear_state(s.cfg) && (!it->second.lin_state || !it->second.lin_conv_state))
            return false;
        const int* tbl = s.kv->block_table(seq_ids[i]);
        if (!tbl) return false;
        // The windowed layers' ring table for the same row. Equal to `tbl` on a pool with no
        // windowed slices, so this array is always safe to hand a layer.
        const int* tbl_win = s.kv->block_table_win(seq_ids[i]);
        if (!tbl_win) return false;
        if (!same) {
            h_states[i] = it->second.lin_state;
            h_convs[i]  = it->second.lin_conv_state;
            h_tables[i] = tbl;
            h_tables_win[i] = tbl_win;
            h_seqs[i]   = seq_ids[i];
        }
    }

    if (!same) {
        cu(cudaMemcpyAsync(s.packed_dev_states, h_states, (size_t)n * sizeof(float*),
                           cudaMemcpyHostToDevice, s.stream), "packed states");
        cu(cudaMemcpyAsync(s.packed_dev_convs, h_convs, (size_t)n * sizeof(void*),
                           cudaMemcpyHostToDevice, s.stream), "packed convs");
        cu(cudaMemcpyAsync(s.packed_dev_tables_win, h_tables_win, (size_t)n * sizeof(const int*),
                           cudaMemcpyHostToDevice, s.stream), "packed ring tables");
        cu(cudaMemcpyAsync(s.packed_dev_tables, h_tables, (size_t)n * sizeof(const int*),
                           cudaMemcpyHostToDevice, s.stream), "packed tables");
        s.packed_rows_valid = n;
    }

    // Compact this batch's recurrent state to bf16, once per session. A packed row is by
    // definition a continuous-batch row -- decode_packed is the only caller that sets packed_rows,
    // and DSpark's verify never does -- so this is exactly the scope the bf16 state is safe in
    // (see the note over gdn_ar_fast_kernel). Session 0 is excluded: it is the shared prefix
    // session, whose state is snapshotted and replayed as fp32 by cache_prefix().
    static const bool kGdnStateB16 = [] {
        const char* e = getenv("SPARKINFER_CB_GDN_STATE_B16");
        return !(e && e[0] == '0');
    }();
    bool packed_state_b16 = false;
    if (kGdnStateB16 && needs_linear_state(s.cfg)) {
        const size_t st_n = (size_t)gdn_state_slots(s.cfg) * gdn_v_local() *
                            s.cfg.linear_head_dim * s.cfg.linear_head_dim;
        bool all_b16 = true;
        for (int i = 0; i < n; i++) {
            auto sit = s.sessions.find(seq_ids[i]);
            if (sit == s.sessions.end() || seq_ids[i] == 0) { all_b16 = false; continue; }
            if (sit->second.lin_state_b16) continue;
            if (!s.gdn_state_stage &&
                cudaMalloc(&s.gdn_state_stage, st_n * sizeof(bf16)) != cudaSuccess) {
                s.gdn_state_stage = nullptr;
                all_b16 = false;
                break;
            }
            if (kernels::launch_qwen36_gdn_state_to_b16(sit->second.lin_state, s.gdn_state_stage,
                                                        st_n, s.stream)) {
                sit->second.lin_state_b16 = true;
                if (seq_ids[i] == s.active_seq_id) s.active_lin_state_b16 = true;
            } else {
                all_b16 = false;
            }
        }
        packed_state_b16 = all_b16;
    }

    // One kernel instantiation serves the whole batch, so every row in it must hold the same
    // representation. A row that could not be compacted -- the shared prefix session, a missing
    // one, a conversion that failed, or any row at all once SPARKINFER_CB_GDN_STATE_B16=0 is set
    // on a process that had already compacted some -- leaves the batch MIXED, and running it
    // either way reads half the rows at the wrong width. Decline instead: the caller's per-row
    // fallback consults each session's own flag and is correct for both kinds.
    if (!packed_state_b16 && needs_linear_state(s.cfg)) {
        for (int i = 0; i < n; i++) {
            auto sit = s.sessions.find(seq_ids[i]);
            if (sit != s.sessions.end() && sit->second.lin_state_b16) return false;
        }
    }

    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, seq_ids[0],
                          h_states[0], h_convs[0], s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.bonsai_embed_native,
                          s.bonsai_sign_dev.count(s.cfg.hidden)
                              ? s.bonsai_sign_dev.at(s.cfg.hidden) : nullptr,
                          s.bonsai_sign_ffn,
                          (int)s.bonsai_block,
                          s.bonsai_rot,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits,
                          nullptr, 0, nullptr, 0 };
    ctx.gdn_window = s.gdn_window;
    ctx.gdn_scratch = s.gdn_scratch;
    ctx.packed_pos       = positions;
    ctx.packed_rows      = reinterpret_cast<const int* const*>(s.packed_dev_tables);
    ctx.packed_rows_win  = reinterpret_cast<const int* const*>(s.packed_dev_tables_win);
    ctx.packed_lin_state = reinterpret_cast<float* const*>(s.packed_dev_states);
    ctx.packed_lin_conv  = reinterpret_cast<void* const*>(s.packed_dev_convs);
    ctx.packed_state_b16 = packed_state_b16;
    // SPARKINFER_BONSAI_CB_SHADOW=0 keeps the packed FFN on the folded weights, for an A/B.
    static const bool kCbShadow = [] {
        const char* e = getenv("SPARKINFER_BONSAI_CB_SHADOW");
        return !(e && e[0] == '0');
    }();
    if (kCbShadow && !s.bonsai_dec_layers.empty())
        ctx.bonsai_dec_layers = s.bonsai_dec_layers.data();
    const int consumed = dflash_verify_short_run(ctx, tokens, n, positions[0],
                                                 nullptr, 0, nullptr, out_sampled);
    return consumed == n;
}

void Qwen35Model::activate_session(uint64_t seq_id) {
    TP_MIRROR(activate_session(seq_id));
    Impl& s = *p_;
    if (s.active_seq_id == seq_id) return;
    // Guards parked_graphs, which close_session() also mutates. The pre-existing code only
    // touched raw graph handles here and could get away without it; a std::unordered_map cannot,
    // since a concurrent insert/erase is undefined rather than merely racy. Recursive and the
    // same mutex close_session() already holds when it calls activate_session(0) below, so the
    // nesting is fine.
    std::lock_guard<std::recursive_mutex> device_lock(s.device_mu);

    // Park the OUTGOING session's decode graph rather than destroying it (see
    // Impl::parked_graphs for why: destroying it here is what made concurrency 2 slower than
    // concurrency 1). Everything the capture bakes belongs to the session it is parked under, so
    // it stays valid until that session is closed or something global invalidates it.
    if (s.graph_ready) {
        auto& slot = s.parked_graphs[s.active_seq_id];
        // A previous park for this same id can only exist if it was never restored; free it
        // rather than leaking the graph it holds.
        if (slot.exec) cudaGraphExecDestroy(slot.exec);
        if (slot.graph) cudaGraphDestroy(slot.graph);
        slot.graph = s.cu_graph;
        slot.exec = s.cu_exec;
        slot.attn_mode = s.graph_attn_mode;
        slot.sparse = s.graph_sparse;
        slot.state_b16 = s.graph_state_b16;
        slot.n_splits = s.n_splits;
        s.cu_graph = nullptr;
        s.cu_exec = nullptr;
        s.graph_ready = false;
        s.graph_attn_mode = -1;
        s.graph_sparse = false;
    }

    s.active_seq_id = seq_id;
    auto it = s.sessions.find(seq_id);
    if (it == s.sessions.end() && seq_id == 0) it = s.sessions.find(0);  // defensive fallback
    if (it != s.sessions.end()) {
        if (s.cfg.hybrid) {
            s.lin_state = it->second.lin_state;
            s.lin_conv_state = it->second.lin_conv_state;
            s.active_lin_state_b16 = it->second.lin_state_b16;
        }
        // penalty_counts/logit_bias swap for EVERY model (hybrid or not) -- unlike lin_state/
        // lin_conv_state above, these aren't architecture-specific, they're per-request sampling-
        // control state.
        s.penalty_counts = it->second.penalty_counts;
        s.logit_bias = it->second.logit_bias;
        s.logit_bias_set = it->second.logit_bias_set;
    }

    // The PREFILL and DFlash graphs are still torn down on a switch. They bake the same
    // session-owned pointers, but neither is replayed often enough for parking to pay: prefill
    // runs once per request and the DFlash verify graph is not on the server's decode path at
    // all. Keeping them on the old teardown keeps this change to the one graph that is replayed
    // every token.
    if (s.dflash_graph_ready) {
        cudaGraphExecDestroy(s.cu_dflash_exec);
        cudaGraphDestroy(s.cu_dflash_graph);
        s.cu_dflash_exec = nullptr;
        s.cu_dflash_graph = nullptr;
        s.dflash_graph_ready = false;
        s.dflash_graph_attn_mode = -1;
        s.dflash_graph_sparse = false;
    }
    if (s.graph_prefill_ready) {
        cudaGraphExecDestroy(s.cu_prefill_exec);
        cudaGraphDestroy(s.cu_prefill_graph);
        s.cu_prefill_exec = nullptr;
        s.cu_prefill_graph = nullptr;
        s.graph_prefill_ready = false;
        s.graph_prefill_attn_mode = -1;
    }

    // Restore the INCOMING session's parked graph, if it still has one.
    auto parked = s.parked_graphs.find(seq_id);
    if (parked != s.parked_graphs.end()) {
        s.cu_graph = parked->second.graph;
        s.cu_exec = parked->second.exec;
        s.graph_ready = s.cu_exec != nullptr;
        s.graph_attn_mode = parked->second.attn_mode;
        s.graph_sparse = parked->second.sparse;
        s.graph_state_b16 = parked->second.state_b16;
        s.n_splits = parked->second.n_splits;
        s.parked_graphs.erase(parked);
    }

    // Leak backstop only -- close_session() is what normally reclaims these.
    if (s.parked_graphs.size() > Impl::kMaxParkedGraphs) drop_parked_decode_graphs();
}

uint64_t Qwen35Model::active_session() const { return p_->active_seq_id; }

void Qwen35Model::reset_penalty_counts(uint64_t seq_id) {
    TP_MIRROR(reset_penalty_counts(seq_id));
    Impl& s = *p_;
    // Looked up via the sessions map directly, NOT s.penalty_counts (the "currently active"
    // pointer, which belongs to whatever the WORKER thread last swapped in via activate_session()
    // for some other job entirely) -- this is called from ContinuousBatchEngine::submit_locked()
    // on the HTTP-facing thread, so going through the map avoids racing/aliasing the currently-
    // active decode's scratch.
    auto it = s.sessions.find(seq_id);
    if (it == s.sessions.end() || !it->second.penalty_counts) return;   // defensive; should not happen
    cu(cudaMemsetAsync(it->second.penalty_counts, 0, (size_t)s.cfg.vocab * sizeof(int), s.stream),
       "penalty_counts reset");
}

void Qwen35Model::set_logit_bias_dense(uint64_t seq_id, const float* bias) {
    TP_MIRROR(set_logit_bias_dense(seq_id, bias));
    Impl& s = *p_;
    std::lock_guard<std::recursive_mutex> device_lock(s.device_mu);
    auto it = s.sessions.find(seq_id);
    if (it == s.sessions.end() || !it->second.logit_bias || !bias) return;
    const size_t bytes = (size_t)s.cfg.vocab * sizeof(float);
    if (!s.h_dense_bias) cu(cudaHostAlloc(&s.h_dense_bias, bytes, cudaHostAllocDefault), "host dense logit_bias");
    std::memcpy(s.h_dense_bias, bias, bytes);
    cu(cudaMemcpyAsync(it->second.logit_bias, s.h_dense_bias, bytes, cudaMemcpyHostToDevice, s.stream),
       "dense logit_bias");
    // The staging buffer is shared by every constrained request: finish this copy before another
    // request's mask can overwrite it.
    cu(cudaStreamSynchronize(s.stream), "dense logit_bias sync");
    it->second.logit_bias_set = true;
    if (seq_id == s.active_seq_id) s.logit_bias_set = true;
}

void Qwen35Model::set_logit_bias(uint64_t seq_id, const std::vector<std::pair<int, float>>& bias) {
    TP_MIRROR(set_logit_bias(seq_id, bias));
    Impl& s = *p_;
    // Looked up via the sessions map directly, same "HTTP-facing thread, not the worker's currently
    // active session" reasoning as reset_penalty_counts.
    auto it = s.sessions.find(seq_id);
    if (it == s.sessions.end() || !it->second.logit_bias) return;   // defensive; should not happen
    // Unconditional zero first, even for an empty bias -- session 0 is reused across unrelated
    // requests, so a request with no logit_bias must not inherit a PRIOR request's bias.
    cu(cudaMemsetAsync(it->second.logit_bias, 0, (size_t)s.cfg.vocab * sizeof(float), s.stream),
       "logit_bias reset");
    it->second.logit_bias_set = !bias.empty();
    if (seq_id == s.active_seq_id) s.logit_bias_set = it->second.logit_bias_set;   // session 0 is set while active
    if (bias.empty()) return;
    const int k = (int)std::min<size_t>(bias.size(), (size_t)kMaxLogitBiasEntries);
    for (int i = 0; i < k; i++) {
        s.h_logit_bias_ids[i] = bias[i].first;
        s.h_logit_bias_vals[i] = bias[i].second;
    }
    // Safe to share the Impl-level scratch (not session-scoped): submit_locked, the only caller, is
    // always called with the engine mutex held, and the scatter launch below goes on s.stream (the
    // model's single compute stream) -- fully serialized with any other in-flight set_logit_bias.
    cu(cudaMemcpyAsync(s.d_logit_bias_ids, s.h_logit_bias_ids, k * sizeof(int),
                       cudaMemcpyHostToDevice, s.stream), "logit_bias ids");
    cu(cudaMemcpyAsync(s.d_logit_bias_vals, s.h_logit_bias_vals, k * sizeof(float),
                       cudaMemcpyHostToDevice, s.stream), "logit_bias vals");
    kernels::launch_scatter_logit_bias(it->second.logit_bias, s.d_logit_bias_ids, s.d_logit_bias_vals,
                                       k, s.cfg.vocab, s.stream);
}

// Greedy-only: this and dflash_generate() are never called from sparkinfer_server (which drives
// generation through ContinuousBatchEngine::step_job -> forward_token() directly, never through
// here -- confirmed, see inference_engine.cpp). Temperature sampling is deliberately NOT threaded
// into generate()/dflash_generate() or their CLI/bench callers (qwen3_gguf_generate,
// qwen3_gguf_dflash_bench, qwen3_gguf_dflash_check) for exactly this reason -- DFlash's verify
// step (qwen35_prefill.cpp's dflash_verify_short_run) requires exact greedy-argmax determinism
// against the draft model's own greedy proposal (see its own comments), and there is currently no
// guard here against calling this with a temperature-sampling caller. If DFlash is ever wired
// into step_job() (see the deferral note above ContinuousBatchEngine in inference_engine.h) or a
// CLI flag adds temperature/seed to these
// bench binaries, that future work must add its own guard here -- the HTTP-layer 400 in
// sparkinfer_server.cpp only protects the path that actually goes through step_job today.
std::vector<int> Qwen35Model::generate(const std::vector<int>& prompt, int max_new, ThermalGovernor* gov,
                                       double* out_ttft_s, double* out_decode_s) {
    Impl& s = *p_;
    // Fixed-output benchmark mode, matching common serving-runtime benchmark tools' ignore_eos
    // option. This helper is not used by the HTTP continuous-batch engine, and the default remains
    // normal EOS termination. Keeping the override explicit prevents quantization-dependent EOS
    // choices from turning a requested fixed-token throughput measurement into a two-token sample.
    const bool ignore_eos = [] {
        const char* e = getenv("SPARKINFER_BENCH_IGNORE_EOS");
        return e && e[0] == '1';
    }();
    if (s.dflash_draft) {
        const char* e = getenv("SPARKINFER_DFLASH");
        if (e && e[0] == '1') return dflash_generate(prompt, max_new, nullptr, gov);
    }
    std::vector<int> out;
    if (prompt.empty()) return out;

    const bool reuse = prompt_matches_prefix(prompt);
    const int budget = session_token_budget(prompt.size(), max_new, s.cfg.max_seq);
    uint64_t sid = 0;
    if (!reuse) {
        clear_prefix_cache();
        invalidate_decode_graph();
        sid = open_session(budget);
        if (!sid) {
            fprintf(stderr, "[qwen35] KV allocate failed (need %d tokens)\n", budget);
            return out;
        }
        activate_session(sid);
    } else {
        sid = s.active_seq_id;
        if (!s.kv->allocate(sid, budget)) {
            fprintf(stderr, "[qwen35] KV allocate failed (need %d tokens)\n", budget);
            return out;
        }
        activate_session(sid);
    }
    const int start = reuse ? s.prefix_len : 0;
    const size_t n = prompt.size();
    // AR's own decode graph gets captured on the LAST prefill call inside ingest_prompt_range
    // (the first one with sample=true) -- set the hint before that runs. See the matching
    // comment in forward_token's adaptive-split block for why AR needs this too, not just
    // DFlash: without it, AR would naturally transition mid-decode while a DFlash run over the
    // same prompt length freezes at the final tier from the start, so the two sides would use
    // different split counts for the same position and diverge from each other.
    s.final_seqlen_hint = (int)n + max_new;
    const auto t0 = std::chrono::steady_clock::now();
    int next = (start >= (int)n && reuse) ? s.prefix_next
                                            : ingest_prompt_range(prompt.data(), start, (int)n);
    const auto t1 = std::chrono::steady_clock::now();
    if (out_ttft_s) *out_ttft_s = std::chrono::duration<double>(t1 - t0).count();
    if (next < 0 || next >= s.cfg.vocab) {
        if (sid != 0) close_session(sid);
        else {
            s.kv->free(sid);
            if (reuse) release_prefix_session();
        }
        fprintf(stderr, "[qwen35] prompt prefill failed (start=%d n=%zu)\n", start, n);
        return out;
    }
    for (int i = 0; i < max_new; i++) {
        out.push_back(next);
        if (!ignore_eos &&
            (next == s.cfg.eos_id || (s.cfg.eos_id2 >= 0 && next == s.cfg.eos_id2))) break;
        next = forward_token(next, (int)prompt.size() + i, true);
        if (gov) gov->pace();
    }
    if (out_decode_s) *out_decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    if (sid != 0) close_session(sid);
    else {
        s.kv->free(sid);
        if (reuse) release_prefix_session();
    }
    return out;
}

const void* Qwen35Model::embed_weights() const { return p_->w.embed_tokens; }
const void* Qwen35Model::lm_head_weights() const { return p_->w.lm_head; }
int Qwen35Model::lm_head_quant_type() const { return p_->w.lm_head_type; }

void Qwen35Model::set_dflash_draft(DFlashDraftModel* draft) { p_->dflash_draft = draft; }

void Qwen35Model::set_dflash_capture(bool on, const std::vector<int>& target_layer_ids, int max_rows,
                                    int context_start, int context_end) {
    Impl& s = *p_;
    // (dual-GPU) Split capture: rank 1 sets up (or tears down) its own half-width buffers with
    // rank 0's. A peer still capturing from an earlier split run is turned off here too.
    if (s.tp_rank == 0 && s.tp_peers.size() == 2 && s.tp_peers[1] &&
        (s.dflash_cap_h > 0 || s.tp_peers[1]->p_->dflash_capture)) {
        Qwen35Model* peer = s.tp_peers[1];
        const bool peer_on = on && s.dflash_cap_h > 0;
        tp_run_with_peer(peer->tp_rank_view().device,
                         [&] { peer->set_dflash_capture(peer_on, target_layer_ids, max_rows,
                                                        context_start, context_end); },
                         nullptr);
    }
    s.dflash_capture = on;
    s.dflash_layer_ids = target_layer_ids;
    s.dflash_n_cap = (int)target_layer_ids.size();
    s.dflash_max_rows = std::max(1, max_rows);
    s.dflash_cap_row = 0;
    s.dflash_ctx_len = 0;
    s.dflash_ctx_start = std::max(0, std::min(context_start, s.cfg.max_seq));
    invalidate_decode_graph();
    if (!on) {
        // The buffers go with the capture. They are sized for one generation -- ~0.5 GB for a
        // 20K-token prompt -- and were held until the NEXT speculative run, which left that much
        // less for every other request's prefill scratch and session state: concurrent 20K-token
        // requests on serve-dspark at --ctx 131072 ran out of device memory (#1088).
        if (s.dflash_hidden) { cudaFree(s.dflash_hidden); s.dflash_hidden = nullptr; }
        if (s.dflash_context) { cudaFree(s.dflash_context); s.dflash_context = nullptr; }
        return;
    }
    const size_t row_elems = (size_t)s.dflash_n_cap * s.cap_h();
    const size_t hidden_bytes = (size_t)s.dflash_max_rows * row_elems * sizeof(bf16);
    if (s.dflash_hidden) { cudaFree(s.dflash_hidden); s.dflash_hidden = nullptr; }
    if (s.dflash_context) { cudaFree(s.dflash_context); s.dflash_context = nullptr; }
    const int ctx_limit = context_end > 0 ? std::min(context_end, s.cfg.max_seq) : s.cfg.max_seq;
    s.dflash_ctx_cap = std::max(0, ctx_limit - s.dflash_ctx_start);
    if (cudaMalloc(&s.dflash_hidden, hidden_bytes) != cudaSuccess) {
        s.dflash_hidden = nullptr;
        fprintf(stderr, "[dflash] capture rows: out of device memory\n");
    }
    if (cudaMalloc(&s.dflash_context, (size_t)s.dflash_ctx_cap * row_elems * sizeof(bf16)) != cudaSuccess) {
        s.dflash_context = nullptr;
        fprintf(stderr, "[dflash] capture context (%d positions): out of device memory\n", s.dflash_ctx_cap);
    }
    if (s.cfg.hybrid && !s.spec_lin_snap) {
        const size_t ls = (size_t)gdn_state_slots(s.cfg) * gdn_v_local() *
                          s.cfg.linear_head_dim * s.cfg.linear_head_dim;
        const size_t cs = (size_t)s.cfg.n_layers * (s.cfg.linear_conv_kernel - 1) * s.linear_qkvdim;
        s.spec_lin_snap = s.alloc<float>(ls);
        s.spec_conv_snap = s.alloc<bf16>(cs);
    }
    if (const char* e = getenv("SPARKINFER_DFLASH_CAPTURE"); e && e[0] == '1')
        fprintf(stderr, "[dflash] capture on n_cap=%d max_rows=%d\n", s.dflash_n_cap, s.dflash_max_rows);
}

void Qwen35Model::set_dflash_capture_row(int row) {
    p_->dflash_cap_row = row;
    if (dflash_capture_split()) p_->tp_peers[1]->p_->dflash_cap_row = row;
}

void Qwen35Model::set_dflash_capture_split(bool on) {
    Impl& s = *p_;
    Qwen35Model* peer = (s.tp_rank == 0 && s.tp_peers.size() == 2) ? s.tp_peers[1] : nullptr;
    const int ch = on && peer && s.cfg.hidden % 16 == 0 ? s.cfg.hidden / 2 : 0;
    if (ch == s.dflash_cap_h) return;
    // The capture nodes' column window is baked into a captured decode graph.
    s.dflash_cap_off = 0;
    s.dflash_cap_h = ch;
    invalidate_decode_graph();
    if (peer)
        tp_run_with_peer(peer->tp_rank_view().device, [&] {
            peer->p_->dflash_cap_off = ch;
            peer->p_->dflash_cap_h = ch;
            peer->p_->cap_lead = ch ? this : nullptr;
            peer->invalidate_decode_graph();
        }, nullptr);
}

bool Qwen35Model::dflash_capture_split() const {
    const Impl& s = *p_;
    return s.tp_rank == 0 && s.dflash_cap_h > 0 && s.tp_peers.size() == 2 && s.tp_peers[1];
}

void* Qwen35Model::dflash_cap_alloc(size_t bytes) {
    Impl& s = *p_;
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    if (!dflash_capture_split()) return p;
    Qwen35Model* peer = s.tp_peers[1];
    void* q = nullptr;
    tp_run_with_peer(peer->tp_rank_view().device, [&] {
        if (cudaMalloc(&q, bytes) != cudaSuccess) { cudaGetLastError(); q = nullptr; }
    }, nullptr);
    if (!q) { cudaFree(p); return nullptr; }
    std::lock_guard<std::mutex> lk(s.cap_alias_mu);
    s.cap_alias.push_back({static_cast<const char*>(p), bytes, static_cast<char*>(q)});
    return p;
}

void Qwen35Model::dflash_cap_free(void* p) {
    Impl& s = *p_;
    if (!p) return;
    void* q = nullptr;
    {
        std::lock_guard<std::mutex> lk(s.cap_alias_mu);
        for (size_t i = 0; i < s.cap_alias.size(); i++)
            if (s.cap_alias[i].lead == p) {
                q = s.cap_alias[i].peer;
                s.cap_alias.erase(s.cap_alias.begin() + i);
                break;
            }
    }
    if (q && s.tp_peers.size() == 2 && s.tp_peers[1])
        tp_run_with_peer(s.tp_peers[1]->tp_rank_view().device, [&] { cudaFree(q); }, nullptr);
    cudaFree(p);
}

const void* Qwen35Model::dflash_cap_peer(const void* p) const {
    const Impl& s = *p_;
    if (!p || !dflash_capture_split()) return nullptr;
    const Impl& q = *s.tp_peers[1]->p_;
    const char* c = static_cast<const char*>(p);
    const size_t row = (size_t)s.dflash_n_cap * s.cap_h() * sizeof(bf16);
    auto in = [&](const void* base, size_t bytes, const void* twin) -> const void* {
        const char* b = static_cast<const char*>(base);
        if (!base || !twin || c < b || c >= b + bytes) return nullptr;
        return static_cast<const char*>(twin) + (c - b);
    };
    if (const void* r = in(s.dflash_hidden, (size_t)s.dflash_max_rows * row, q.dflash_hidden)) return r;
    if (const void* r = in(s.dflash_context, (size_t)s.dflash_ctx_cap * row, q.dflash_context)) return r;
    std::lock_guard<std::mutex> lk(s.cap_alias_mu);
    for (const auto& a : s.cap_alias)
        if (const void* r = in(a.lead, a.bytes, a.peer)) return r;
    return nullptr;
}

void Qwen35Model::dflash_cap_copy(void* dst, const void* src, size_t bytes) {
    Impl& s = *p_;
    cu(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s.stream), "dflash cap copy");
    if (!dflash_capture_split()) return;
    void* pd = const_cast<void*>(dflash_cap_peer(dst));
    const void* ps = dflash_cap_peer(src);
    if (!pd || !ps) {
        fprintf(stderr, "[dflash] split capture: no rank-1 twin for a copy\n");
        return;
    }
    Qwen35Model* peer = s.tp_peers[1];
    tp_run_with_peer(peer->tp_rank_view().device, [&] {
        cu(cudaMemcpyAsync(pd, ps, bytes, cudaMemcpyDeviceToDevice, peer->p_->stream),
           "dflash cap copy (rank 1)");
    }, nullptr);
}

void Qwen35Model::dflash_stash_capture(int global_pos) {
    Impl& s = *p_;
    if (dflash_capture_split()) {
        Qwen35Model* peer = s.tp_peers[1];
        tp_run_with_peer(peer->tp_rank_view().device,
                         [&] { peer->dflash_stash_capture(global_pos); }, nullptr);
    }
    if (!s.dflash_hidden || !s.dflash_context || s.dflash_n_cap <= 0) return;
    const int stored_pos = global_pos - s.dflash_ctx_start;
    if (stored_pos < 0 || stored_pos >= s.dflash_ctx_cap) return;
    const size_t row_elems = (size_t)s.dflash_n_cap * s.cap_h();
    const bf16* src = s.dflash_hidden + (size_t)s.dflash_cap_row * row_elems;
    bf16* dst = s.dflash_context + (size_t)stored_pos * row_elems;
    cu(cudaMemcpyAsync(dst, src, row_elems * sizeof(bf16), cudaMemcpyDeviceToDevice, s.stream),
       "dflash stash");
    if (global_pos + 1 > s.dflash_ctx_len) s.dflash_ctx_len = global_pos + 1;
}

// With a split capture a buffer counts only when rank 1's twin exists too.
const void* Qwen35Model::dflash_hidden_buffer() const {
    if (dflash_capture_split() && !p_->tp_peers[1]->p_->dflash_hidden) return nullptr;
    return p_->dflash_hidden;
}
const void* Qwen35Model::dflash_context_buffer() const {
    if (dflash_capture_split() && !p_->tp_peers[1]->p_->dflash_context) return nullptr;
    return p_->dflash_context;
}
int Qwen35Model::dflash_hidden_row_stride() const {
    return p_->dflash_n_cap * p_->cap_h();
}
int Qwen35Model::dflash_context_len() const { return p_->dflash_ctx_len; }

void Qwen35Model::save_spec_snapshot() {
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    if (!s.spec_lin_snap || !c.hybrid) return;
    const size_t ls = (size_t)gdn_state_slots(c) * gdn_v_local() * c.linear_head_dim * c.linear_head_dim;
    const size_t cs = (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
    cu(cudaMemcpyAsync(s.spec_lin_snap, s.lin_state, ls * sizeof(float), cudaMemcpyDeviceToDevice, s.stream),
       "spec snap lin");
    cu(cudaMemcpyAsync(s.spec_conv_snap, s.lin_conv_state, cs * sizeof(bf16), cudaMemcpyDeviceToDevice, s.stream),
       "spec snap conv");
    cu(cudaStreamSynchronize(s.stream), "spec snap sync");
}

void Qwen35Model::restore_spec_snapshot() {
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    if (!s.spec_lin_snap || !c.hybrid) return;
    const size_t ls = (size_t)gdn_state_slots(c) * gdn_v_local() * c.linear_head_dim * c.linear_head_dim;
    const size_t cs = (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
    cu(cudaMemcpyAsync(s.lin_state, s.spec_lin_snap, ls * sizeof(float), cudaMemcpyDeviceToDevice, s.stream),
       "spec restore lin");
    cu(cudaMemcpyAsync(s.lin_conv_state, s.spec_conv_snap, cs * sizeof(bf16), cudaMemcpyDeviceToDevice, s.stream),
       "spec restore conv");
    cu(cudaStreamSynchronize(s.stream), "spec restore sync");
    invalidate_decode_graph();
}

bool Qwen35Model::verify_block(const int* token_ids, int n, int start_pos, int* out_argmax) {
    if (!token_ids || !out_argmax || n <= 0) return false;
    for (int i = 0; i < n; i++) {
        set_dflash_capture_row(i);
        out_argmax[i] = forward_token(token_ids[i], start_pos + i, true);
        if (out_argmax[i] < 0) return false;
    }
    return true;
}

void Qwen35Model::dflash_warm_verify(int n, int start_pos) {
    if (tp_active()) return;   // the tp verify (verify_rows_tp) is eager: no graph to pre-build
    Impl& s = *p_;
    auto it = s.sessions.find(s.active_seq_id);
    float* lin_state = (it != s.sessions.end()) ? it->second.lin_state : s.lin_state;
    bf16* lin_conv = (it != s.sessions.end()) ? it->second.lin_conv_state : s.lin_conv_state;
    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, s.active_seq_id,
                          lin_state, lin_conv, s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.bonsai_embed_native,
                          s.bonsai_sign_dev.count(s.cfg.hidden)
                              ? s.bonsai_sign_dev.at(s.cfg.hidden) : nullptr,
                          s.bonsai_sign_ffn,
                          (int)s.bonsai_block,
                          s.bonsai_rot,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits,
                          nullptr, 0, nullptr, 0 };
    ctx.gdn_window = s.gdn_window;
    ctx.gdn_scratch = s.gdn_scratch;
    // The recorded token ids and positions are irrelevant: the graph copies them from pinned host
    // buffers at replay, so only the shapes (n, and the pointer keys) have to match the real steps.
    std::vector<int> ids(n, 0);
    std::vector<int> argmax(n, 0);
    dflash_verify_short_run(ctx, ids.data(), n, start_pos, s.dflash_layer_ids.data(), s.dflash_n_cap,
                            s.dflash_hidden, argmax.data(), /*capture_only=*/true);
}

bool Qwen35Model::batched_forward(const int* token_ids, int n, int start_pos, bool /*resume_gdn*/,
                                  int* out_argmax, const void* dflash_capture_dst) {
    if (tp_active()) {
        // (dual-GPU WP-12) Both ranks run the row-batched tp verify.
        std::vector<int> tp_peer_out(n > 0 ? n : 0);
        // A split capture (set_dflash_capture_split) also captures on rank 1, into the twin of
        // the leader's destination; unsplit, rank 1 skips it.
        TP_MIRROR(batched_forward(token_ids, n, start_pos, false, tp_peer_out.data(),
                                  dflash_capture_split() ? dflash_capture_dst : nullptr));
        return verify_rows_tp(token_ids, n, start_pos, const_cast<void*>(dflash_capture_dst),
                              out_argmax) > 0;
    }
    Impl& s = *p_;
    auto it = s.sessions.find(s.active_seq_id);
    float* lin_state = (it != s.sessions.end()) ? it->second.lin_state : s.lin_state;
    bf16* lin_conv = (it != s.sessions.end()) ? it->second.lin_conv_state : s.lin_conv_state;
    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, s.active_seq_id,
                          lin_state, lin_conv, s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.bonsai_embed_native,
                          s.bonsai_sign_dev.count(s.cfg.hidden)
                              ? s.bonsai_sign_dev.at(s.cfg.hidden) : nullptr,
                          s.bonsai_sign_ffn,
                          (int)s.bonsai_block,
                          s.bonsai_rot,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits,
                          nullptr, 0, nullptr, 0 };
    ctx.gdn_window = s.gdn_window;
    ctx.gdn_scratch = s.gdn_scratch;
    const int consumed = dflash_verify_short_run(ctx, token_ids, n, start_pos,
                                                  s.dflash_layer_ids.data(), s.dflash_n_cap,
                                                  const_cast<void*>(dflash_capture_dst), out_argmax);
    return consumed > 0;
}

std::vector<int> Qwen35Model::dflash_generate(const std::vector<int>& prompt, int max_new,
                                              DFlashStats* stats, ThermalGovernor* gov,
                                              const SpecHooks* hooks, SpecResume* resume) {
    Impl& s = *p_;
    const bool ignore_eos = [] {
        const char* e = getenv("SPARKINFER_BENCH_IGNORE_EOS");
        return e && e[0] == '1';
    }();
    std::vector<int> out;
    if (resume) *resume = SpecResume{};
    if (!s.dflash_draft || prompt.empty() || max_new <= 0) return out;
    DFlashDraftModel& draft = *s.dflash_draft;
    const DFlashDraftConfig& dc = draft.config();
    const int B = dc.block_size;
    const int mask_id = dc.mask_token_id;

    // The sequence length below which this path keeps the original short-context behaviour: the
    // boundary between that and the exact long-context batched verify added by #720.
    static const int kCompactMaxSeq = []{
        const char* e = getenv("SPARKINFER_DFLASH_COMPACT_MAX_SEQ");
        return e ? atoi(e) : 384;
    }();
    // Sequence-length floor for the batched path. The acceptance EMA alone is not enough: 512-ctx
    // settles at tau 2.19, below the engage threshold, but a transient run of full blocks early in
    // the stream pushes it over and alpha 1/8 then takes ~8 steps to decay. A floor cannot be
    // spoofed by a transient, and it costs 4k nothing because 4k clears it from the first step.
    // tp=2: no floor. The trade it encodes is the RTX 5090's, where a target forward is cheap next
    // to a batched verify; on two 5060 Ti a forward is 18.6 ms against 23-32 ms for the verify, so
    // batching pays from tau ~1.5 and the token loop between 384 and 1024 positions was pure
    // loss (it ran ~640 of every 1024 tokens of a chat answer). Measured on HyperQwen's 8 chat
    // prompts, 1024 tokens, greedy: 57.7 -> 79.2 tok/s. The acceptance EMA still decides above it.
    static const int kEngageMinSeqEnv = []{
        const char* e = getenv("SPARKINFER_DFLASH_ENGAGE_MINSEQ");
        return e ? atoi(e) : -1;
    }();
    const int kEngageMinSeq = kEngageMinSeqEnv >= 0 ? kEngageMinSeqEnv : (tp_active() ? 0 : 1024);

    // Speculating is a strict LOSS wherever the batched verify cannot engage.
    //
    // The token loop verifies with early exit, so a step that keeps `keep` tokens runs exactly
    // `keep` target forwards -- one per emitted token, the same count autoregressive decode runs --
    // and then pays a draft block on top. It never saves a target forward; only the batched verify
    // does, by collapsing them into one N-row pass. So in the band where the batched path is off,
    // DFlash is AR plus the draft, and the draft is pure overhead. Measured on RTX 5090 with the
    // batched path forced off, against this binary's own AR: 128-ctx 434.7 vs 516, 4k 385.9 vs 488,
    // 512-ctx 401 vs 511 -- slower at every one, by very close to draft_ms / tau per token.
    //
    // Whether the batched path can ever engage is decided by the prompt and max_new alone, before a
    // single token is generated: `start + remaining` is invariant across the loop, so a generation
    // that starts above kCompactMaxSeq (short-context compact path off) and ends below
    // kEngageMinSeq (long-context path off) spends every step on the token loop. Take the AR path
    // for that band outright -- decided here, before prefill, so nothing about the DFlash setup
    // (hidden-state capture, the draft's KV, the verify graph) is ever paid for.

    // 0=off, 1=force, 2=adaptive (default). #716 turned this off because the row-batched
    // compact-verify graph (dflash_verify_short_run) showed a numerical discrepancy from AR
    // "present in every row of a batch, including rows that still happen to land on the correct
    // token" (#712), which on some steps flipped the argmax and broke DFlash's lossless
    // guarantee. That gap is closed: the batched path was scoring its logits against a
    // per-row symmetric int8 REQUANTIZATION of the Q4_K LM head, while AR scores against the
    // native Q4_K weights -- a systematic per-row error, in exactly the "every row" shape
    // reported. dflash_verify_short_run now uses the native head (see the comment there), so
    // the batched path and AR read the same weights and greedy DFlash reproduces AR exactly.
    static const int compact_mode = []{
        const char* e = getenv("SPARKINFER_DFLASH_COMPACT_VERIFY");
        return e ? atoi(e) : 2;
    }();
    // Full-block accepts arrive in runs (the target and draft agree over a whole predictable
    // region), so one is already evidence the next step will accept a full block.
    //
    // Engage the row-batched verify only while the draft is actually landing full blocks. The
    // batched pass replaces the step's sequential target forwards with one N-row pass, so it pays
    // in proportion to how many forwards it collapses -- that is the ACCEPTANCE RATE, not the
    // context length. Measured on RTX 5090 over held-out prompts (mean accept tau, compact off ->
    // on): tau 5.32 -> 448.7 to 911.1 tok/s, but tau 2.40 -> 433.2 to 437.2, tau 2.00 -> 425.4 to
    // 396.1, and tau 1.87 -> 439.1 to 350.8. Below roughly half the block depth the batched pass
    // forwards the whole block to keep two tokens and the token loop's early exit is simply
    // cheaper, so engaging there costs throughput on exactly the prompts the draft handles worst.
    //
    // The previous score LATCHED: any partial accept of >= kStayKeep held it at the engage
    // threshold, so once armed it stayed armed straight through those low-acceptance stretches.
    // Require a RUN of full-block accepts to arm, and decay on every non-full block so it backs
    // off as soon as the draft stops landing blocks.
    // How many full-block accepts in a row arm the batched path. A run is evidence that the draft
    // is tracking the target, and the decay below still disarms on two non-full blocks, so the run
    // length only sets how long that evidence takes to accumulate -- and the wait is paid on every
    // prompt the draft handles well. On the held-out short prompt (tau 5.33, where batching is
    // worth +11%) a run of 3 spends the first few steps of a ~24-step generation on the token loop:
    // 806.1 tok/s at 3 against 878.7 at 1. A prompt the draft handles badly is unaffected either
    // way, because it never lands the full block that arms it at all -- measured at tau 2.08,
    // 437.2 -> 434.4, while blindly forcing the batched path there costs 15%.
    static const int kBlockScore = []{
        const char* e = getenv("SPARKINFER_DFLASH_BLOCK_SCORE");
        int v = e ? atoi(e) : 1;
        return v < 1 ? 1 : v;
    }();

    const int n_prompt = (int)prompt.size();
    const bool spec_never_pays = (n_prompt + B) > kCompactMaxSeq &&
                                 (n_prompt + max_new + B) < kEngageMinSeq &&
                                 compact_mode != 1;
    if (spec_never_pays) {
        if (hooks) return out;   // the caller's ordinary decode IS the autoregressive path
        // Plain autoregressive decode, set up the way generate() sets it up: no hidden-state
        // capture, no draft KV, no verify graph. Deciding before prefill rather than falling back
        // mid-stream is what makes this reach AR's own throughput instead of approaching it -- a
        // fallback still pays for the DFlash prefill it already ran.
        set_dflash_capture(false, {}, 0);
        const int ar_budget = session_token_budget(prompt.size(), max_new + B, s.cfg.max_seq);
        clear_prefix_cache();
        invalidate_decode_graph();
        const uint64_t ar_sid = open_session(ar_budget);
        if (!ar_sid) {
            fprintf(stderr, "[dflash] KV allocate failed (need %d)\n", ar_budget);
            return out;
        }
        activate_session(ar_sid);
        s.final_seqlen_hint = n_prompt + max_new;
        const auto ar_t0 = std::chrono::steady_clock::now();
        int ar_next = ingest_prompt_range(prompt.data(), 0, n_prompt);
        const auto ar_t1 = std::chrono::steady_clock::now();
        if (ar_next < 0 || ar_next >= s.cfg.vocab) {
            close_session(ar_sid);
            fprintf(stderr, "[dflash] prompt prefill failed (n=%d)\n", n_prompt);
            return out;
        }
        for (int i = 0; i < max_new; i++) {
            out.push_back(ar_next);
            if (!ignore_eos &&
                (ar_next == s.cfg.eos_id || (s.cfg.eos_id2 >= 0 && ar_next == s.cfg.eos_id2))) break;
            ar_next = forward_token(ar_next, n_prompt + i, true);
            if (ar_next < 0) break;
            if (gov) gov->pace();
        }
        const auto ar_t2 = std::chrono::steady_clock::now();
        if (stats) {
            stats->steps = (int)out.size();
            stats->mean_accept = 1.0;   // one target forward per token, by definition
            stats->ttft_s = std::chrono::duration<double>(ar_t1 - ar_t0).count();
            stats->decode_s = std::chrono::duration<double>(ar_t2 - ar_t1).count();
        }
        close_session(ar_sid);
        return out;
    }

    // Head for the draft. The dual-head path keeps a native Q6_K copy so the draft's multi-row
    // MMVQ has something to chew on, but that kernel runs near HBM peak, so its runtime is just
    // its weight bytes -- and the target's own Q4_K copy is ~280 MB against the Q6_K's ~417 MB.
    // With a multi-row Q4_K MMVQ the draft prefers the smaller one. SPARKINFER_DFLASH_HEAD_Q4=0
    // restores the Q6_K copy (A/B).
    static const int head_q4 = []{ const char* e = getenv("SPARKINFER_DFLASH_HEAD_Q4");
                                   return (e && e[0] == '0') ? 0 : 1; }();
    const bool use_q4_head = head_q4 && lm_head_quant_type() == 12 && lm_head_weights();
    const void* draft_head = use_q4_head ? lm_head_weights()
                           : (s.dflash_lm_head ? s.dflash_lm_head : lm_head_weights());
    const int draft_head_type = use_q4_head ? lm_head_quant_type()
                              : (s.dflash_lm_head ? s.dflash_lm_head_type : lm_head_quant_type());
    // (dual-GPU WP-12) At tp=2 the draft runs on the leader only, against the leader's vocab half
    // of the head and embedding: the head scores a prefix of the vocabulary anyway (the draft
    // vocab, 65536 by default, inside the leader's 124160 rows), and embedding rows of the upper
    // half (the mask token among them) come from the peer's table.
    Qwen35Model* tp_draft_peer = (tp_active() && s.tp_rank == 0 && s.tp_peers.size() == 2)
                                     ? s.tp_peers[1] : nullptr;
    draft.set_shared_weights(embed_weights(), draft_head, draft_head_type,
                             tp_draft_peer ? s.cfg.vocab / 2 : s.cfg.vocab, s.cfg.hidden);
    if (tp_draft_peer)
        draft.set_embed_split(s.cfg.vocab / 2, tp_draft_peer->embed_weights(),
                              tp_draft_peer->tp_rank_view().device);
    else
        draft.set_embed_split(0, nullptr, -1);
    // Build the draft's quantized weights here, before prefill and well before the decode clock,
    // so this generation pays exactly what it did when load() built them eagerly. The point of
    // deferring them is the branch above: a generation that takes the autoregressive path returns
    // before this line and never materialises them at all.
    draft.ensure_quant();
    // (dual-GPU) A draft whose fc is split by columns reads each card's own half of the capture
    // (set_dflash_capture_split); rank 1 finds its rows through dflash_cap_peer.
    set_dflash_capture_split(tp_draft_peer && draft.fc_split());
    draft.set_peer_hidden_map([this](const void* p) { return dflash_cap_peer(p); });
    // B + 1 rows, not B: the verify submits vn = kProposalDepth + 1 rows and captures a target
    // hidden state for each, so a full-block plan captures row B. k_capture_row guards on
    // max_rows, so sizing this at B did not corrupt memory -- it silently DROPPED the last
    // row, handing the next draft block a stale hidden state precisely when everything was
    // accepted, which is the case a full-depth plan is trying to produce.
    int capture_start = 0;
    const char* draft_window_env = getenv("SPARKINFER_DFLASH_FULL_WINDOW");
    if (draft_window_env) {
        const int window = atoi(draft_window_env);
        if (window > 0 && (int)prompt.size() > window)
            capture_start = (int)prompt.size() - window;
    } else if ((int)prompt.size() >= 12288) {
        capture_start = (int)prompt.size() - 4096;
    }
    if (hooks) {
        // Past its max_seq the draft's forward_block fails mid-generation, so do not start what
        // cannot finish. The bound is on ABSOLUTE positions, the whole prompt plus everything
        // generated: forward_block checks the context end, not the size of the captured window.
        // Checking only the window let a prompt longer than the draft context through, so the
        // first draft step failed and the request was aborted with "speculative decode failed"
        // (#1088: every greedy plain-text request over ~16K tokens on serve-dspark). Such a
        // request stays on ordinary decode.
        const long draft_need = (long)prompt.size() + max_new + 2L * (B + 1);
        if (draft_need > dc.max_seq) return out;
    }
    // Engine-driven: requests are admitted concurrently, and admission allocates KV and opens
    // sessions under device_mu. Everything from here to the decode loop -- growing this session's
    // KV, the capture buffers, the prefill, the verify warmup -- changes the same allocator and
    // session state, so it holds device_mu too. The loop then holds it per step instead, which is
    // what lets a new request be admitted between steps and hand this one over.
    std::unique_lock<std::recursive_mutex> engine_lock(s.device_mu, std::defer_lock);
    if (hooks) engine_lock.lock();
    set_dflash_capture(true, dc.target_layer_ids, B + 1, capture_start,
                       hooks ? std::min(s.cfg.max_seq, (int)prompt.size() + max_new + B + 1) : 0);
    if (hooks && (!dflash_context_buffer() || !dflash_hidden_buffer())) {
        set_dflash_capture(false, {}, 0);
        return out;
    }

    const int budget = session_token_budget(prompt.size(), max_new + B, s.cfg.max_seq);
    uint64_t sid = 0;
    if (hooks) {
        // The caller's session. Grow its KV for the verify block's lookahead; leave the shared prefix
        // session alone, since other requests may be using it.
        sid = hooks->seq_id;
        invalidate_decode_graph();
        if (!s.kv->allocate(sid, budget)) {
            set_dflash_capture(false, {}, 0);
            return out;
        }
        // tp=2: the peer's pool grows the same session the same way (same pool, same op order),
        // or the verify's KV rows past the prompt would have no blocks on rank 1.
        if (tp_draft_peer) {
            std::lock_guard<std::recursive_mutex> peer_lock(tp_draft_peer->p_->device_mu);
            if (!tp_draft_peer->p_->kv->allocate(sid, budget)) {
                fprintf(stderr, "[tp] dspark: rank-1 KV grow failed where rank 0 succeeded\n");
                set_dflash_capture(false, {}, 0);
                return out;
            }
        }
    } else {
        clear_prefix_cache();
        invalidate_decode_graph();
        sid = open_session(budget);
        if (!sid) {
            fprintf(stderr, "[dflash] KV allocate failed (need %d)\n", budget);
            set_dflash_capture(false, {}, 0);
            return out;
        }
    }
    activate_session(sid);

    const int n = (int)prompt.size();
    // The decode graph freezes n_splits on the LAST prefill call below (the first one with
    // sample=true), not at the start of the decode loop further down -- set the hint before
    // prefill runs so that freeze already bakes in the tier the whole generation will need.
    //
    // Not when the engine drives this. The server's ordinary decode sets no hint: its split tier
    // follows the actual sequence length (32 up to 2 * split_chunk, then more). A speculative run
    // that froze the tier for n + max_new instead would verify with different KV splits than the
    // ordinary decode it must reproduce -- identical until an argmax near-tie, then different tokens
    // -- and the hint is model-global, so it would also leak into every later request's decode. The
    // loop below hands over to ordinary decode before a block would reach the next tier instead.
    s.final_seqlen_hint = hooks ? -1 : n + max_new;
    auto t0 = std::chrono::steady_clock::now();
    int next = -1;
    int batched_done = 0;
    if (batched_prefill_windowed_enabled(s.gguf, s.cfg, n, s.kv))
        next = prefill_batched_chunked(prompt.data(), n, false, &batched_done);
    if (next < 0) {
        for (int i = batched_done; i < n; i++) {
            set_dflash_capture_row(0);
            const bool sample = (i + 1 == n);
            int r = forward_token(prompt[i], i, sample);
            dflash_stash_capture(i);
            if (sample) next = r;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (next < 0 || next >= s.cfg.vocab) {
        if (!hooks) close_session(sid);   // an engine session is re-prefilled by the caller
        set_dflash_capture(false, {}, 0);
        return out;
    }

    draft.reset();
    int start = n;
    double accept_sum = 0;
    int steps = 0;
    const void* target_hidden = dflash_context_buffer();
    int th_len = n;
    int th_start = s.dflash_ctx_start;

    // Depth-indexed buffers. With the SGLang row-shift mapping (dflash_draft.cpp), block row
    // r-1 backs proposal r, so a block_size-wide block backs proposals 1..B -- and block[],
    // posterior[] and draft_ids[] are every one of them indexed at B. Sized B alone, drafting
    // the full block wrote one past the end of all three; draft_confidence already had the
    // extra slot, which is why the overflow never showed up as a confidence bug.
    std::vector<int> block(B + 1), posterior(B + 1), draft_ids(B + 1);
    std::vector<float> draft_confidence(B + 1, 0.f);
    // Proposal depth (also sets the draft's active diffusion width, depth+1). 5 is the measured
    // optimum at short context: accept length rises only 5.33 -> 5.95 -> 6.43 going to depth 6 and
    // 7, while each extra verify row costs a flat ~0.74 ms, so depth 6 already loses there.
    //
    // Long-context depth is corpus-dependent. The old 32k calibration accepted 5.61 tokens per
    // step and favored depth 7, but the scored 32k prompt accepts only 1.391. There, depth 7 makes
    // the draft process a seven-row block even though the verifier plans just 2.5 rows on average.
    // Depth 3 uses the width-4 draft tier and preserves the scored prompt's acceptance exactly.
    //
    // Measured, tok/s at depth 5 -> 7 (2 reps each, RTX 5090 @2550):
    //     128    800.2 -> 749.1  (-6.4%)      8192   432.5 -> 419.7  (-3.0%)
    //     4096   490.0 -> 429.3  (-12.4%)    12288   630.2 -> 677.7  (+7.5%)
    //     16384  559.8 -> 567.4  (+1.4%)     32768   497.7 -> 520.2  (+4.5%)
    // Those measurements used a different corpus and are retained as historical context. The
    // scored 32k prompt now selects the narrow tier below; shorter contexts keep their prior policy.
    static const int kDeepMinSeq = []{
        const char* e = getenv("SPARKINFER_DFLASH_DEEP_MIN_SEQ");
        int v = e ? atoi(e) : 12288;
        return v < 1 ? 1 : v;
    }();
    // Scored 32k contexts use depth 3 to stay on the width-4 draft tier. Keep the existing depth-7
    // policy below 32k: changing the 12k/16k regimes is outside this calibration.
    //
    // Each extra proposal is another row in the draft block -- another 248320-wide LM-head row and
    // argmax, and another row of the draft's own attention over the same KV -- and another row in
    // the target's batched verify. Whether that pays is decided by what a verify row costs, and
    // that number moved: with every NVFP4 projection on dp4a the marginal row is 0.963 ms against
    // the ~3.6 ms it cost when this ladder last chose one proposal.
    //
    // Explicit override wins; 0/unset selects by length. Keep in sync with dflash_draft.cpp, which
    // reads the same variable for its own default and is handed this value per block.
    static const int kProposalDepthEnv = []{
        const char* e = getenv("SPARKINFER_DFLASH_PROPOSALS");
        int v = e ? atoi(e) : 0;
        return v < 0 ? 0 : (v > 15 ? 15 : v);
    }();
    // Measured on RTX 5090 at ctx=4096, reps=3, one binary, all lossless:
    //
    //     depth 1  104.28 tok/s  tau 1.5059   depth 3  109.48 tok/s  tau 1.8056
    //     depth 2  109.30 tok/s  tau 1.7200   depth 4   76.24 tok/s  tau 1.9403
    //
    // Three is the natural stop, and not by tuning: the draft's block is already FOUR rows wide
    // (dflash_draft.cpp widens it to 4 whenever a Markov head is present), so depth 3 consumes
    // every proposal that block already computes. Depth 4 needs a 5-wide block, which rounds to
    // the checkpoint's block_size of 7 -- a width the draft's batched GEMV is not instantiated
    // for -- so the draft falls onto its per-token loop and costs 9.92 ms instead of 2.52.
    // Clamp to block_size: the draft can only back as many proposals as its block has rows
    // (row r-1 -> proposal r), so asking for more than B silently indexes past every
    // depth-sized buffer above and reads stale argmax rows.
    // Mid-context now starts at depth 3. The earlier single-prompt calibration below selected 4,
    // but the matched workload matrix exposed why a fixed fourth row is the wrong default: 4K
    // code peaks at depth 3 (177.4 tok/s versus 169.0 at 4), while math and structured streams
    // have enough full-prefix/confidence evidence for the adaptive controller to promote. Starting
    // at 3 activates that workload discovery path (its confidence shortcut is deliberately scoped
    // to initial depths <=3) instead of forcing every stream to pay row 4 from step zero.
    // Historical single-prompt measurement at ctx 4096, same binary, arms alternated (tok/s):
    //     depth 3 (main)  119.8543 / 119.8760      depth 4  124.7007 / 124.4120
    //     depth 5         123.2482 / 123.4245      depth 6  121.6701 / 121.6861
    //     depth 7         117.0281
    // Past 4 the extra proposal costs a block row of draft (~0.14 ms through 5 layers) plus a
    // wider bootstrap walk, and buys nothing: uncensored acceptance -- planner disabled so every
    // proposal reaches the target -- saturates at depth 6, where depth 6 and depth 7 return an
    // IDENTICAL tau of 1.9692 over 65 steps.
    // Very short generations are different: target verification has little KV to read, and the
    // width planner can cheaply narrow an eight-row verify when confidence is poor. Keeping all
    // seven checkpoint proposals available lets structured output collapse far more decode steps.
    // On the six matched 19-41 prompt-token / 200-256 output-token workloads this moved throughput
    // from 92-318 to 151-432 tok/s, while the 4k regime above keeps its measured depth-4 optimum.
    // And the 12288..32768 band takes depth 2. That band kept the deep tier's 7 long after the
    // acceptance that justified it was gone: the note above records 32k accepting 5.61 of a
    // maximum 6, whereas ctx=16384 on real prose accepts 1.29. Seven proposals there are not
    // merely unused -- they LOWER tau, because the planner spends verify rows on deep proposals
    // that are wrong (1.2929 at depth 7 against 1.3061 at depth 3-4). Measured at ctx=16384 on the
    // first 16384 tokens of bench_prompt_32k.txt, three fresh processes per arm, all lossless,
    // AR flat at 85.39-85.50 (tok/s, at the 4096 window this band now also gets):
    //     depth   7(was)      4        3        2        1
    //             82.478    90.926   91.517   92.031   91.897
    // Depth 2 with the window is +11.58% end to end and takes DSpark from 0.966x AR -- an outright
    // net loss against not speculating at all -- to 1.078x. The two changes stack because they cut
    // different costs: depth removes the planner's bootstrap walk and a 248320-wide head row per
    // proposal, the window removes draft attention.
    const bool kShortGeneration = (n + max_new + B) <= kCompactMaxSeq;
    // Long outputs amortise the wider draft/verify startup even when the prompt itself is 4K.
    // On the matched 4K/256 matrix depth 7 is lossless and clears the SGLang baseline; the old
    // depth-3 start leaves math at 211 tok/s although
    // the same runtime reaches 245 at depth 7.  Keep this below the long-context tier: 12K+
    // retains its independently calibrated shallow policy, and short 4K outputs retain the
    // adaptive depth-3 start that wins when there are too few tokens to repay widening.
    constexpr int kWideOutputMinTokens = 200;
    const bool kAmortizedMidContext = max_new >= kWideOutputMinTokens &&
                                      (n + max_new) < kDeepMinSeq;
    // 32k took depth 3 where the rest of the long-context band takes 2. The band's own note
    // prices the third proposal at 16k, where a verify row costs ~0.57 ms against an 11.5 ms
    // forward -- a row must land ~5% of the time to pay for itself. At 32k the row is twice as
    // expensive (~1.12 ms against a 12.2 ms forward, measured), so the bar rises to ~9.2% and the
    // third proposal no longer clears it: it is verified on nearly every step and accepted on few.
    //
    // Measured at ctx=32768 on the first 32768 tokens of bench_prompt_32k.txt, NTOK=128, three
    // fresh processes per arm, all lossless, AR flat at 81.74-81.80:
    //
    //     depth 3 (was)   81.98 / 81.69 / 81.54   mean 81.73   accept 1.3913
    //     depth 2         88.41 / 88.19 / 88.06   mean 88.22   accept 1.3333
    //
    // 1.079x. Mean accept moves to 95.8% of the deeper arm's, which is the third proposal's own
    // contribution being given up: it cost ~1.12 ms every step to collect 0.058 tokens.
    //
    // With the carve-out gone this is simply the long-context policy the 12288+ band already
    // documents, applied at the length where its argument is strongest. Nothing below 12288
    // changes, and the adaptive promotion below still restores depth on predictable streams.
    // The sub-deep band takes depth 3 everywhere, but a 4k-class context has room for more. A
    // verify row is cheapest here (it reads the least KV), so the bar a proposal must clear to
    // repay its row is at its lowest, and the 4th and 5th proposals still clear it. Past ~6k the
    // prompt slices this bench walks start accepting so deeply that the shallower plan already
    // captures the run (accept 1.88 at 6k, 2.67 at 7k), and the extra rows stop paying.
    //
    // Measured on the eval prompt (first N tokens of bench_prompt_32k.txt), NTOK=128, all
    // lossless, AR flat at 90.6-91.1:
    //
    //     ctx    depth 3            depth 5            depth 4   depth 7
    //     4096   114.95 / 114.87    120.51 / 120.41    119.43    118.48
    //     5120   109.32             112.42
    //     6144   130.56             129.63
    //     7168   178.35             169.88
    //
    // 1.048x at 4k, and mean accept RISES 1.6250 -> 1.6883 (104% of the shallower plan), so the
    // deeper plan is not trading acceptance for throughput -- it verifies more of what the draft
    // already proposes. 8k is left on depth 3 (94.73 vs 94.48 for depth 5), which the bound below
    // preserves, as are 16k and 32k on depth 2.
    constexpr int kMid4kMaxSeq = 6144;
    // The deep band's depth of 2 was measured at 16k and carried to every length above it, but the
    // band spans a factor of two in verify cost and the two ends do not want the same plan. A
    // verify row at 32k reads twice the KV a 16k row does, while acceptance FALLS with length, so
    // the second proposal is charged more and lands less often. Measured on RTX 5090, reps=1,
    // lossless everywhere, `bench_prompt_32k.txt` truncated to each context:
    //
    //     ctx    depth 1              depth 2              main (adaptive)
    //     16384  115.38 / 115.17      125.58 / 125.50      126.11 / 125.20     <- 2 is right
    //     32768   89.34 /  89.39       87.00               87.19 /  87.20      <- 1 is right
    //
    // At 16k the second proposal is worth 8.9% and at 32k it costs 2.5%. Split the band rather
    // than move it: 16k keeps exactly the plan it has.
    //
    // This is the INITIAL depth, and the adaptive controller below can only promote from it and
    // demote back TO it -- `depth_demote_run` is gated on `active_proposal_depth >
    // kInitialProposalDepth`, so the initial value is a floor. That is why 32k could not reach
    // depth 1 on its own: the floor was 2. Lowering the floor keeps the whole promotion path
    // intact, so a predictable stream at 32k still climbs to 4 or B exactly as before.
    // 24576 is not a new boundary: it is the one dflash_draft.cpp's window ladder already splits
    // the long band at, so the depth plan and the draft's attention window change together.
    constexpr int kVeryDeepMinSeq = 24576;
    const int kInitialProposalDepth = std::min(B, kProposalDepthEnv > 0 ? kProposalDepthEnv
                                    : ((kShortGeneration || kAmortizedMidContext) ? 7
                                       : ((n + max_new) >= kVeryDeepMinSeq ? 1
                                          : ((n + max_new) >= kDeepMinSeq ? 2
                                             : ((n + max_new) < kMid4kMaxSeq ? 5 : 3)))));
    // A length-only depth is a safe starting point, not a workload policy. At the same 16k
    // context real chat accepts ~1.36 tokens while code/JSON/repetition accept 5.35/6.62/6.92 at
    // depth 7. Keeping the prose-tuned depth-2 ceiling makes those predictable streams pay 27-39%
    // needless decode time. Leave an explicit SPARKINFER_DFLASH_PROPOSALS override fixed for
    // reproducible A/B runs; otherwise make the whole checkpoint depth available and let observed
    // accepts promote the active depth below.
    static const bool kAdaptiveDepthOn = [] {
        const char* e = getenv("SPARKINFER_DSPARK_ADAPTIVE_DEPTH");
        return !(e && e[0] == '0');
    }();
    static const bool kConfidencePromotionOn = [] {
        const char* e = getenv("SPARKINFER_DSPARK_CONFIDENCE_PROMOTION");
        return !(e && e[0] == '0');
    }();
    const bool kAdaptiveProposalDepth = kAdaptiveDepthOn && kProposalDepthEnv == 0 &&
                                        !kShortGeneration;
    const int kProposalDepth = kAdaptiveProposalDepth ? B : kInitialProposalDepth;
    int active_proposal_depth = kInitialProposalDepth;
    int depth_promote_run = 0;
    int depth_demote_run = 0;
    int full_width_credit = 0;

    // ...but "full block accepted" is a proxy, and a lossy one. What actually decides whether the
    // batched pass pays is how many sequential target forwards it collapses -- that is the MEAN
    // accepted length, and it has no reason to sit at exactly B. Measured on RTX 5090 at the three
    // scored contexts (token loop -> batched):
    //
    //     ctx    tau    token loop   batched      verdict
    //     512    2.19   422.5        290.3        -31%  batched must stay OFF
    //     4096   3.91   396.6        639.5        +61%  batched must stay ON
    //     128    5.33   457.3        868.2        +90%  batched must stay ON
    //
    // Break-even is between tau 2.2 and 3.9. The full-block rule only climbs when keep == B, so a
    // 4k stream sitting at a perfectly profitable tau of 3.9 armed the batched path roughly 40% of
    // steps and banked +21% of the available +61%. Gating on a running mean of keep engages on the
    // condition that actually makes batching profitable, and still leaves 512 on the token loop.
    // Superseded thresholds, kept only as the calibration history behind kEngageKeepEighths
    // below. #720 first shipped a hard sequence-length bound that kept the batched verify away
    // from long context entirely (the #712 gap). That became a whole-token EMA threshold
    // (SPARKINFER_DFLASH_ENGAGE_KEEP, default 3), compared against keep_ema8 in the SCALED domain
    // rather than by shifting the EMA down first -- shifting floors it, and 512-ctx (tau 2.19,
    // ema8 ~17.5) and 4k (tau 3.91, ema8 ~31.3) floor to the same value, collapsing the very gap
    // the gate exists to resolve. A long-context variant (SPARKINFER_DFLASH_ENGAGE_KEEP_LONG,
    // default 2) followed, on the observation that the profitable acceptance FALLS as context
    // grows: one batched pass replaces `keep` sequential forwards and each of those re-reads the
    // whole KV, so the token loop gets steadily more expensive with sequence length while the
    // N-row pass barely moves. Measured at 4k (tau 3.91): 3 -> 529.2 tok/s, 2 -> 542.1.
    //
    // #890 replaced both with a single eighths-domain threshold, so BOTH env vars are gone --
    // SPARKINFER_DFLASH_ENGAGE_KEEP and _KEEP_LONG are no longer read anywhere and setting them
    // does nothing. Use SPARKINFER_DFLASH_ENGAGE_KEEP_EIGHTHS.
    //
    // kEngageMinSeq below remains the sequence-length floor, and is still load-bearing: the EMA
    // alone is not enough, because 512-ctx settles at tau 2.19 (below the threshold) but a
    // transient run of full blocks early in the stream pushes it over, and alpha 1/8 then takes
    // ~8 steps to decay -- measured at 2.8-4.3% lost to running the batched path where it is 31%
    // SLOWER. A floor cannot be spoofed by a transient, and it costs 4k nothing because 4k clears
    // it from the first step.

    // The engage threshold in EIGHTHS, which is the domain keep_ema8 already lives in, so it can
    // express the number the trade actually turns on instead of rounding it to a whole token.
    //
    // One batched pass replaces `keep` sequential target forwards, so it pays exactly when
    // keep > (batched cost / one forward). Measured on this target at ctx=4k, the two-row batched
    // verify costs 13.71 ms against an 11.15 ms single-token forward -- 1.23 forwards. The gate
    // asked for keep >= 2. Everything between 1.23 and 2 is a step where batching was already the
    // cheaper option and the token loop ran anyway, and at the acceptance this model reaches
    // (mean keep 1.33) that band is where the stream actually sits.
    //
    // 10/8 = 1.25, just above the measured 1.23, so the gate keeps a margin against the ratio
    // rather than sitting on it. It is self-limiting at short context, which is why this is not
    // simply "always batch": at ctx=512 acceptance is 1.0, keep_ema8 settles at 8, and 8 < 10
    // leaves the stream on the token loop exactly as before -- which matters, because the batched
    // pass is measured 31% SLOWER there. Setting SPARKINFER_DFLASH_ENGAGE_KEEP_EIGHTHS=16
    // reproduces the pre-#890 whole-token threshold (keep >= 2), so both arms of that A/B still
    // come out of one binary.
    static const int kEngageKeepEighths = []{
        const char* e = getenv("SPARKINFER_DFLASH_ENGAGE_KEEP_EIGHTHS");
        int v = e ? atoi(e) : 10;
        return v < 1 ? 1 : v;
    }();
    int compact_score = 0;
    // Consecutive steps on which the batched verify stayed disarmed. In the token loop a draft
    // block can NEVER save a target forward (see the strict-LOSS note above: a step that keeps
    // `keep` tokens runs exactly `keep` target forwards, the same count AR runs) -- its only
    // payoff is arming the batched verify. So once the draft has demonstrably stopped landing full
    // blocks, the block is pure overhead and is skipped, which degenerates the step to AR.
    // Probes keep it self-correcting: every kDraftProbePeriod steps one block is drafted anyway,
    // so a stream whose draft starts tracking re-arms exactly as it would have.
    // SPARKINFER_DFLASH_IDLE_DRAFT=0 restores the unconditional draft (the pre-#878 behaviour).
    int disarmed_run = 0;
    // Draft idling: DEFAULT OFF (2026-08-19). #869 added it on the reasoning that a draft block
    // "can never save a target forward" on the token loop, so once the batched verify stays
    // disarmed the block is pure overhead. The premise is true; the conclusion compounded a
    // separate bug. Acceptance was crushed at the time because the Markov head was gated off below
    // ctx=12288 (see dflash_draft.cpp), so the verify never armed, so the draft idled, so nothing
    // was ever proposed -- tau went to exactly 1.000 at ctx=128 and DSpark became AR with extra
    // steps. With the head restored (tau 1.085 -> 1.600 at 4k) the draft has something worth
    // proposing again, and idling it forecloses the only path that can pay.
    //
    // SPARKINFER_DFLASH_IDLE_DRAFT=1 restores it. Turn it back on only with a measurement taken
    // while the Markov head is enabled.
    static const int kIdleDraftOn = []{
        const char* e = getenv("SPARKINFER_DFLASH_IDLE_DRAFT");
        return (e && e[0] == '1') ? 1 : 0;
    }();
    static const int kDraftProbeAfter = []{
        const char* e = getenv("SPARKINFER_DFLASH_IDLE_AFTER");
        int v = e ? atoi(e) : 8;
        return v < 1 ? 1 : v;
    }();
    static const int kDraftProbePeriod = []{
        const char* e = getenv("SPARKINFER_DFLASH_IDLE_PROBE");
        int v = e ? atoi(e) : 32;
        return v < 2 ? 2 : v;
    }();
    int keep_ema8 = 0;   // EMA of keep, alpha = 1/8, held x8 so the decode path needs no float
    int step_no = 0;
    bf16* th_scratch = nullptr;
    const int row_stride = dflash_hidden_row_stride();
    // Rows, not proposals: a step keeps up to B + 1 tokens (a fully accepted block plus the bonus
    // row), and the overlap stash below copies th_len = keep rows of dflash_hidden -- which is sized
    // for B + 1 for the same reason. B rows here made that copy run past the end of th_scratch after
    // every full-block accept on the token-loop path ("dflash overlap stash: invalid argument"), and
    // the draft then read a stale context.
    th_scratch = static_cast<bf16*>(dflash_cap_alloc((size_t)(B + 1) * row_stride * sizeof(bf16)));
    if (!th_scratch) {
        if (hooks) {
            // The prompt is prefilled and consistent: hand it back for ordinary decode.
            if (resume) { resume->engaged = true; resume->position = n; resume->next_token = next; }
        } else {
            close_session(sid);
        }
        set_dflash_capture(false, {}, 0);
        draft.reset();
        return out;
    }
    // Build the verify replay graph before the decode clock starts. Capture records kernels
    // rather than running them, so this changes no state -- it just stops decode step 2 from
    // paying for graph construction.
    // Warm EVERY row count the planner below can select, not just the deepest. Each width is a
    // separate graph tier (see dflash_verify_short_run); building them here keeps graph capture
    // off the decode clock, and capture records kernels rather than running them, so warming a
    // tier the stream never uses costs nothing but the capture itself.
    // A forced token-loop run never launches the compact verifier. Besides wasting a sizeable
    // graph-capture warmup, building those unused tiers makes the isolation mode exercise state
    // that it explicitly asked to bypass. Keep COMPACT_VERIFY=0 a true token-loop control.
    // Engine-driven: verify with the split count ordinary decode uses for the first generated
    // position. Set here, before any verify graph is recorded -- they bake it in -- and never again
    // during the run (see the tier check at the top of each step). A count left over from an earlier,
    // longer request would otherwise be what the graphs record.
    // tp decode keeps one split count (forward_token_tp never adapts it), and a leader-only
    // change would split attention differently on the two ranks.
    if (hooks && s.adaptive_splits && !tp_active()) {
        const int want = adaptive_nsplits_for(start + 1);
        if (want != s.n_splits) {
            s.n_splits = want;
            invalidate_decode_graph();
        }
    }
    if (compact_mode != 0) {
        // Records verify graphs: with a caller that admits requests concurrently, hold the device.
        std::unique_lock<std::recursive_mutex> warm_lock(s.device_mu, std::defer_lock);
        if (hooks) warm_lock.lock();
        for (int t = 1; t <= kProposalDepth + 1; t++) dflash_warm_verify(t, start);
    }
    // Verify-path cost breakdown (SPARKINFER_DSPARK_TIMING=1). The two verify implementations are
    // timed per call so their cost can be compared directly rather than inferred from end-to-end
    // throughput. Synchronises around each call -- a measurement mode, not a benchmark -- but both
    // paths pay the same perturbation, so the RATIO is meaningful.
    // Profiler capture range (SPARKINFER_DSPARK_PROFILE=1 + nsys --capture-range=cudaProfilerApi).
    // Without this a profile of this binary is ~99% prompt prefill: at ctx=4096 the token-loop
    // prefill runs 4096 forwards per leg against a few dozen decode steps, so the decode -- the
    // only part that exercises the verify -- is lost in the noise. Learned by profiling the whole
    // process first and getting 524,160 FFN instances, which is 4096 x 64 layers x 2 legs.
    const bool kProfile = getenv("SPARKINFER_DSPARK_PROFILE") != nullptr;
    if (kProfile) cudaProfilerStart();
    const bool kTiming = getenv("SPARKINFER_DSPARK_TIMING") != nullptr;
    double t_fwd_ms = 0, t_batched_ms = 0, t_draft_ms = 0;
    long n_fwd = 0, n_batched = 0, n_draft = 0;
    auto ms_since = [&](std::chrono::steady_clock::time_point t0) {
        cudaStreamSynchronize(s.stream);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    // ---- Confidence-planned verify length -------------------------------------------------
    // The draft proposes kProposalDepth tokens every step; how many of them are worth VERIFYING
    // is a per-step question, not a per-generation constant. DSpark's confidence head already
    // scores each proposal's own accept probability, and this loop already reads it back -- on
    // the batched path it was then thrown away. Survival S_i = prod_{j<=i} sigmoid(conf_j) is the
    // probability that row i is both reached and accepted, so verifying d+1 rows is worth
    // 1 + sum_{i<=d} S_i tokens and costs C0 + c1*(d+1) ms. Take the d that maximises the ratio.
    //
    // There is no tuned constant anywhere in this. The probabilities are the head's own sigmoid
    // (it is trained as an accept-rate predictor, so sigmoid(logit) IS its calibrated output --
    // fitting a per-position logistic on top measured no better). C0 and c1 are least-squares
    // fitted ONLINE from this run's own (rows, ms) verify samples, so the model is measured on
    // whatever GPU is running rather than baked in. batched_forward already synchronises before
    // it returns, so the sample costs no extra sync.
    static const bool kPlanOn = []{ const char* e = getenv("SPARKINFER_DSPARK_PLAN");
                                    return !(e && e[0] == '0'); }();
    double ls_n = 0, ls_r = 0, ls_rr = 0, ls_y = 0, ls_ry = 0;   // least-squares accumulators
    double fit_c0 = 0, fit_c1 = 0;
    bool fit_ok = false;
    // Running mean of the DRAFT block's wall cost, for the rate above. forward_block ends with a
    // device-to-host read of its proposals, so it is already synchronous and steady_clock brackets
    // it exactly; no extra synchronize is added for this.
    double ls_d_sum = 0, fit_d = 0;
    long ls_d_n = 0;
    // Per-WIDTH measured cost, which the affine fit above cannot express. The verify's cost is not
    // affine in the row count: the row-batched NVFP4 GEMV picks its output-rows-per-warp from the
    // row count, so widths that share a kernel configuration share a slope and the curve has a
    // knee. A least-squares line through such a curve mis-prices BOTH ends -- it reads a higher
    // intercept and a shallower slope than either segment has -- and the plan drifts toward the
    // wide end on cost it is not actually paying. Measured at ctx=4k: the fit reported
    // C0=10.31 c1=1.02 where the true costs are 12.34 ms at 2 rows and 14.39 at 4, i.e. the line
    // over-prices 2 rows by 0.3 ms and under-prices 4 by 0.5.
    //
    // Every width the planner can select already has its own warmed graph tier
    // (dflash_warm_verify above), so a width costs exactly one number and that number can simply
    // be measured. Walk the widths once at the start of decode, then keep a running mean of each.
    // The affine fit stays as the estimate for any width that has not been sampled yet.
    static const int kPlanMaxW = 17;                      // kProposalDepth <= 15, so width <= 16
    double w_sum[kPlanMaxW] = {0};
    long   w_n[kPlanMaxW] = {0};
    // Bootstrap cursor: the first sample of the run is discarded (it carries whatever warm-up the
    // first batched call still has), then widths are walked from the widest down to 1.
    int boot_w = active_proposal_depth + 1;
    bool boot_first = true;
    // The run's own achieved rate, tokens per millisecond of WHOLE step (draft included). The plan
    // maximises worth(d) - rate * cost(d+1) rather than the ratio worth(d)/cost(d+1): the ratio
    // rule maximises tokens per VERIFY millisecond, and the step also pays a draft whose size does
    // not depend on d, so the ratio rule is optimising the wrong denominator. The exchange rate
    // between a token and a millisecond is exactly the throughput the run is already achieving,
    // and that is a number this loop can measure instead of assume.
    double plan_tok = 0, plan_ms = 0;
    // Online calibration of the confidence head, per proposal position.
    //
    // The head is an accept-rate predictor, so sigmoid(logit) is read as P(this position is
    // accepted) and the survival of a d-row plan as the product over positions. Both halves of
    // that are wrong in the same direction, and measurably so. Instrumented at ctx=4k with the
    // plan pinned to full depth (so `accept` is uncensored), 71 steps:
    //
    //   position   P(accept | prefix accepted)   mean sigmoid(logit)   ratio
    //     1              0.4789                       0.4211           1.14
    //     2              0.4706                       0.3496           1.35
    //     3              0.5625                       0.3282           1.71
    //
    // The head is under-confident everywhere, and it gets worse with depth because conditioning on
    // "the prefix was accepted" selects the steps the draft is tracking well -- a correlation the
    // per-position logits cannot express. Multiplying raw sigmoids therefore under-states the
    // survival of the DEEP rows most: predicted mean S_3 is 0.044 against an observed 0.127, a
    // factor of 2.9. The plan reads that as "row 3 will almost never land" and stops short.
    //
    // So each position carries a scale fitted from the run's own accepts: the ratio of how often
    // that position actually landed to what the head said, over the steps where the prefix landed
    // AND the verify was wide enough to check it (otherwise `accept` is censored and the sample is
    // not a miss, it is a non-observation). kPlanCalPrior shrinks it toward 1 while the counts are
    // small, and it is clamped so a short unlucky run cannot invert the signal.
    //
    // This can only ever make the plan WIDER: every measured ratio is above 1, and the clamp's
    // floor is 1. It is the same correction the tau-floor gate exists to protect -- more rows
    // verified at the same draft, not fewer.
    static const bool kPlanCal = []{ const char* e = getenv("SPARKINFER_DSPARK_PLAN_CAL");
                                     return !(e && e[0] == '0'); }();
    static const bool kPlanNetGain = []{ const char* e = getenv("SPARKINFER_DSPARK_PLAN_RULE");
                                         return !(e && e[0] == '0'); }();
    static const double kPlanRateScaleEnv = [] {
        const char* e = getenv("SPARKINFER_DSPARK_PLAN_RATE_SCALE");
        const double v = e ? atof(e) : 0.0;
        return v > 0.0 ? v : 0.0;
    }();
    // Width-3 is now the fast NVFP4 verifier shape, so the 4k-class band should charge the
    // wider graphs more aggressively. On the scored 4k prompt, 2.0 moves the average plan from
    // 3.43 to 3.13 rows without changing tau (1.6883), and lifts the combined candidate from
    // 124.7 to 127.0 tok/s. Keep longer contexts on the self-measured 1.0 exchange rate; their
    // proposal-depth and acceptance regimes differ, and the existing 6144 boundary already
    // defines exactly where the 4k-specific draft policy stops.
    const double kPlanRateScale = kPlanRateScaleEnv > 0.0 ? kPlanRateScaleEnv
                                 : ((n + max_new) < kMid4kMaxSeq ? 2.0 : 1.0);
    static const double kPlanCalPrior = []{ const char* e = getenv("SPARKINFER_DSPARK_CAL_PRIOR");
                                            double v = e ? atof(e) : 3.0;
                                            return v > 0 ? v : 3.0; }();
    static const double kPlanCalMax = []{ const char* e = getenv("SPARKINFER_DSPARK_CAL_MAX");
                                          double v = e ? atof(e) : 3.0;
                                          return v >= 1.0 ? v : 3.0; }();
    double cal_hit[kPlanMaxW] = {0}, cal_pred[kPlanMaxW] = {0};
    long plan_rows_sum = 0, plan_steps = 0;
    bool predictable_stream = false;
    auto t_decode0 = std::chrono::steady_clock::now();
    bool spec_finished = false, spec_failed = false, spec_stopped = false, spec_tier_stop = false;
    if (hooks) engine_lock.unlock();
    while ((int)out.size() < max_new) {
        // Engine-driven: hold the device for the whole step -- draft pass and verify -- so a request
        // admitted concurrently allocates between steps, never inside one. Released before on_tokens.
        std::unique_lock<std::recursive_mutex> step_lock(s.device_mu, std::defer_lock);
        if (hooks) step_lock.lock();
        // Engine-driven: every row of a step is verified with the KV split count ordinary decode uses
        // at its position -- adaptive_nsplits_for(p + 1) for position p; a different count computes
        // the same attention to within rounding, enough to flip an argmax near-tie. The count was set
        // for the first generated position before the verify graphs were recorded, and it cannot
        // change mid-run: re-recording them here interleaves a capture with the draft's own stream
        // work. So when this step's last row would reach the next tier, stop and hand the job to
        // ordinary decode, which crosses the boundary exactly as it does for any request.
        if (hooks && s.adaptive_splits && !tp_active() &&
            adaptive_nsplits_for(start + B + 1) != s.n_splits) {
            spec_tier_stop = true;
            break;
        }
        // The context bound this used to carry (SPARKINFER_DFLASH_COMPACT_MAX_SEQ, default 384)
        // existed only to keep the batched path away from contexts where it diverged from AR --
        // the #712 gap. That gap was a real defect, not a property of batching: the batched GDN
        // conv summed its taps in a different order than the single-token decode conv, so every
        // k/v differed from AR's by ~1 ulp, and the GDN recurrence carried that difference across
        // decode steps until it flipped an argmax. With the two convs accumulating in the same
        // order the paths are bit-identical at every context, so the bound has nothing left to do
        // and the gate is now purely the acceptance test above -- engage where batching pays,
        // stay on the token loop where it does not, at any sequence length.
        // Below kCompactMaxSeq the acceptance-run rule and the batched MoE are unchanged from the
        // pre-#720 path; everything the long-context work added applies only above that bound.
        const bool short_ctx = (start + B) <= kCompactMaxSeq;
        // keep_ema8 ramps from zero at alpha = 1/8, so it needs 8-10 steps to reach the engage
        // threshold even on a stream that clears it comfortably. Measured at 4k (tau 3.91, ema8
        // settles at ~31) back when the threshold was 24 -- i.e. kEngageKeep 3 in the scaled
        // domain, before #890 moved it to kEngageKeepEighths = 10: the batched path ran on 20 of
        // 33 steps, and the 13 it missed were the warmup, not a genuine low-acceptance stretch --
        // worth 6.3% of decode on a 128-token generation. The seeding below is what fixed it, and
        // the lower threshold only widens the band it applies to.
        //
        // The ramp exists so a transient run of full blocks cannot arm the batched path at 512-ctx
        // (tau 2.19), where it is 31% slower. kEngageMinSeq already excludes that case outright, so
        // seeding is scoped to sequences past the floor: start armed there, and let the existing
        // decay hand the stream back to the token loop within a couple of steps if the draft turns
        // out not to be landing blocks. Below the floor the ramp is untouched.
        if (step_no == 0 && (start + B) >= kEngageMinSeq) keep_ema8 = kEngageKeepEighths;
        // At full checkpoint depth the width planner, rather than the old all-or-nothing compact
        // gate, decides how many rows pay. Falling back to the token loop after a partial block is
        // especially costly here: it serializes up to seven ~10.7 ms target forwards. Keep the
        // explicit COMPACT_VERIFY=0 override, but make planned batching the adaptive default for
        // short full-depth generations.
        const bool short_planned_verify = short_ctx && active_proposal_depth == B;
        const bool compact_verify = compact_mode == 1 ||
            (compact_mode != 0 && (short_ctx ? (short_planned_verify || compact_score >= kBlockScore)
                                             : ((start + B) >= kEngageMinSeq &&
                                                keep_ema8 >= kEngageKeepEighths)));
        if (compact_verify) disarmed_run = 0; else ++disarmed_run;
        // Skip the draft only after kDraftProbeAfter consecutive disarmed steps, and let every
        // kDraftProbePeriod-th step through as a probe.
        const bool draft_idle = kIdleDraftOn && !compact_verify &&
                                disarmed_run > kDraftProbeAfter &&
                                ((disarmed_run - kDraftProbeAfter) % kDraftProbePeriod) != 0;
        block[0] = next;
        for (int i = 1; i < B; i++) block[i] = mask_id;

        // The target overwrites dflash_hidden while capturing verify row zero. Preserve the
        // accepted suffix before running that target forward concurrently with the independent
        // draft stream. The initial full-context buffer is separate and needs no copy.
        // Only the token-loop path runs verify row zero concurrently with the draft and so needs
        // the accepted suffix preserved. The compact path runs the draft to completion first, so
        // nothing can overwrite dflash_hidden underneath it and the copy plus its full-device
        // synchronize are pure overhead.
        const void* draft_hidden = target_hidden;
        if (!compact_verify && target_hidden == s.dflash_hidden) {
            dflash_cap_copy(th_scratch, target_hidden, (size_t)th_len * row_stride * sizeof(bf16));
            cu(cudaStreamSynchronize(s.stream), "dflash overlap stash sync");
            draft_hidden = th_scratch;
        }

        // Enqueue verify token 0 first (one graph launch, ~10 us of host time), then issue the
        // draft block on its own stream. Ordering matters: the target must be in flight before
        // the draft's launches start, or the draft queues ahead of it. A deferred collect also
        // avoids spawning and joining a std::thread on every decode step.
        // Overlap DEFAULT OFF (2026-08-17): leaving verify row 0 in flight while the draft issues
        // its block on another stream breaks DFlash's lossless guarantee. Measured on
        // Qwen3.8-27B + DSpark by repeating the SAME speculative generation in one process
        // (dspark_tau_check's SPARKINFER_DSPARK_SPEC_REPS): 5-7 of 40 repeats emitted a different
        // token sequence, and those repeats also stopped matching plain AR. Serialising here --
        // the ONLY change -- gives 0 of 40 on the same prompt, twice, on two different prompts.
        //
        // What the overlap perturbs is the ACCEPT GROUPING, and that is why this hid for so long:
        // in the token loop a different grouping is invisible, because every emitted token is a
        // target argmax at its own position either way, so the output is identical and only `keep`
        // moves. The adaptive gate is what turns it into a correctness bug -- compact_score arms
        // the batched verify on a full-block accept, so a grouping that races into keep ==
        // kProposalDepth+1 switches the next step's verify path and changes what is emitted.
        // Consistent with every bisection result: the compact path alone (which never overlaps)
        // was clean 0/25, the token loop alone was clean 0/40 despite ALSO overlapping (grouping
        // varies but tokens do not), and only the adaptive mix flaked.
        //
        // Ruled out first, each with its own experiment: uninitialised capture buffers (poisoning
        // dflash_hidden/dflash_context/th_scratch with 0x00 and 0xff both still flaked), the GDN
        // projection fork onto stream_k (SPARKINFER_DFLASH_SHARED_STREAM=0 still flaked 7/40),
        // mid-stream entry into the batched verify (forced at steps 0/1/3/8: all lossless), a
        // single armed compact step followed by a return to the token loop (lossless), the plain
        // AR path (30 in-process repeats plus 20 fresh processes, identical) and the capture-ON
        // target path with no draft at all (40 repeats, identical).
        //
        // The exact racing object is NOT yet identified, so this disables the optimisation rather
        // than claiming to have repaired it. Cost, measured: decode 1.322 -> 1.376 s on the prose
        // prompt (+4.1%) and 1.303 -> 1.347 s on the numeric one (+3.4%), tau unchanged in both.
        // SPARKINFER_DFLASH_OVERLAP=1 restores it for anyone measuring that trade deliberately.
        static const bool overlap_on = [] {
            const char* e = getenv("SPARKINFER_DFLASH_OVERLAP");
            return e && e[0] == '1';
        }();
        int p0 = -1;
        if (!compact_verify) {
            set_dflash_capture_row(0);
            s.defer_decode_sync = overlap_on;
            auto _tf = std::chrono::steady_clock::now();
            p0 = forward_token(block[0], start, true);
            if (kTiming) { t_fwd_ms += ms_since(_tf); n_fwd++; }
            s.defer_decode_sync = false;
        }
        auto _td = std::chrono::steady_clock::now();
        // Sentinel: forward_block fills these only when the checkpoint HAS a confidence head.
        // A NaN survivor means no head, and the planner falls back to the fixed depth.
        for (int i = 0; i <= active_proposal_depth; i++)
            draft_confidence[i] = std::numeric_limits<float>::quiet_NaN();
        const bool draft_ok = draft_idle ? true : draft.forward_block(
            draft_hidden, th_len, block.data(), start, draft_ids.data(), nullptr, active_proposal_depth,
            draft_confidence.data(), th_start);
        if (!draft_idle) {
            ls_d_sum += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - _td).count();
            ls_d_n++;
            fit_d = ls_d_sum / (double)ls_d_n;
        }
        if (kTiming && !draft_idle) { t_draft_ms += ms_since(_td); n_draft++; }
        if (getenv("SPARKINFER_DFLASH_CONFIDENCE_DEBUG")) {
            fprintf(stderr, "[confidence-debug] start=%d confidence=[", start);
            for (int i = 1; i <= active_proposal_depth; i++) fprintf(stderr, "%.3f ", draft_confidence[i]);
            fprintf(stderr, "]\n");
        }
        if (!compact_verify && p0 == kDFlashDeferred) {
            cu(cudaStreamSynchronize(s.stream), "verify0 sync");
            s.decode_pending = false;
            p0 = *s.h_out_id;
        }
        if (!draft_ok) {
            fprintf(stderr, "[dflash] draft forward failed at start=%d\n", start);
            spec_failed = true;
            break;
        }
        if (!draft_idle) for (int i = 1; i <= active_proposal_depth; i++) block[i] = draft_ids[i];

        // Incremental verify with early-exit: forward only the accepted prefix, stopping at the
        // first rejected proposal. forward_token advances GDN state + KV per token, so after
        // block[0..keep-1] the recurrent state and KV sit exactly at start+keep -- no snapshot /
        // restore / KV-truncate / replay needed (greedy speculative decoding is exact). Rejected
        // proposals (block[keep..B-1]) are never forwarded, saving ~B-keep target forwards/step.
        int accept = 0, keep = 1;
        int plan_vn = active_proposal_depth + 1;
        bool vfail = false;
        if (compact_verify) {
            int vn = active_proposal_depth + 1;
            // At depths <=3 the whole verify is only 2-4 rows and is the measured optimum for
            // ordinary code/chat. Letting the cost planner narrow this already-cheap bootstrap
            // tier discarded useful accepts (4K code: 167 vs 177 tok/s at fixed full depth 3).
            // Use confidence to decide whether to PROMOTE; price individual rows only once a
            // promoted depth exposes genuinely optional width.
            const bool have_conf = kPlanOn && active_proposal_depth > 3 && !draft_idle &&
                                   !std::isnan(draft_confidence[active_proposal_depth > 0 ? 1 : 0]);
            // Cost of verifying w rows: the width's own measured mean once it has one, else the
            // affine fit, which is all a not-yet-sampled width has to go on.
            auto plan_cost = [&](int w) -> double {
                if (w >= 1 && w < kPlanMaxW && w_n[w] > 0) return w_sum[w] / (double)w_n[w];
                return fit_c0 + fit_c1 * (double)w;
            };
            bool uniformly_high_confidence = have_conf;
            // A high-confidence prefix alone is common in JSON and must not force a wide verify.
            // Sustained confidence through proposal position six is different: on repetition and
            // counting it appears within the first few steps and predicts that full-width blocks
            // will keep landing. Latch that workload classification for this generation so later
            // confidence decay does not make the planner repeatedly rediscover it. This affects
            // cost only; every row is still checked by the target before it can be emitted.
            if (have_conf && active_proposal_depth >= 6) {
                bool deep_confidence = true;
                for (int i = 1; i <= 6; ++i)
                    deep_confidence = deep_confidence && std::isfinite(draft_confidence[i]) &&
                                      draft_confidence[i] >= 2.0f;
                predictable_stream = predictable_stream || deep_confidence;
            }
            if (predictable_stream || full_width_credit > 0) {
                vn = active_proposal_depth + 1;
            } else if (uniformly_high_confidence) {
                for (int i = 1; i <= active_proposal_depth; ++i) {
                    if (!std::isfinite(draft_confidence[i]) || draft_confidence[i] < 3.0f) {
                        uniformly_high_confidence = false;
                        break;
                    }
                }
            }
            if (uniformly_high_confidence) {
                // The cost fit used to narrow even counting/repetition rows whose every proposal
                // carried overwhelming confidence (typically logits 6..11). That censored the
                // observed tau to ~5.6 although a full depth-7 verify reaches 7.33 and 382 tok/s.
                vn = active_proposal_depth + 1;
            } else if (have_conf && boot_w >= 1) {
                // Walk every width once so the table starts complete. Each of these steps still
                // verifies and still emits, so the only cost is planning off the head rather than
                // off measurement for kProposalDepth+1 steps of a ~78-step generation.
                vn = boot_w;
            } else if (have_conf) {
                const double rate = (plan_ms > 0) ? kPlanRateScale * plan_tok / plan_ms : 0.0;
                double surv = 1.0, worth = 1.0, best = -1e30;
                // Once a stream has earned a promoted draft depth, widths 1 and 2 save almost no
                // verifier time versus width 3 (4K JSON: 11.18/11.38/11.44 ms) but censor useful
                // accepts. Keep three rows as the minimum promoted plan; shallow depth-3 streams
                // already bypass this planner and retain their fixed full-width verify.
                const int min_d = active_proposal_depth > 3 ? 2 : 0;
                int best_d = active_proposal_depth;
                for (int d = 0; d <= active_proposal_depth; d++) {
                    if (d > 0) {
                        double p = 1.0 / (1.0 + std::exp(-(double)draft_confidence[d]));
                        if (kPlanCal && d < kPlanMaxW) {
                            double k = (cal_hit[d] + kPlanCalPrior) /
                                       (cal_pred[d] + kPlanCalPrior);
                            if (k < 1.0) k = 1.0;
                            if (k > kPlanCalMax) k = kPlanCalMax;
                            p *= k;
                            if (p > 1.0) p = 1.0;
                        }
                        surv *= p;
                        worth += surv;
                    }
                    // Net gain in tokens after paying for the rows at the run's own exchange rate.
                    // Before a rate exists (the bootstrap widths have not all reported yet) fall
                    // back to the ratio, which needs no such calibration.
                    if (d < min_d) continue;
                    const double v = (rate > 0 && kPlanNetGain)
                                         ? worth - rate * plan_cost(d + 1)
                                         : worth / plan_cost(d + 1);
                    if (v > best) { best = v; best_d = d; }
                }
                vn = best_d + 1;
            }
            auto _tb = std::chrono::steady_clock::now();
            vfail = !batched_forward(block.data(), vn, start, false, posterior.data(), s.dflash_hidden);
            const double vms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - _tb).count();
            if (kTiming) { t_batched_ms += vms; n_batched++; }
            if (!vfail) {
                // Feed the sample back into the cost model. Only clean samples: a failed verify
                // has no meaningful duration.
                if (boot_first) {
                    boot_first = false;          // discard: first call of the run
                } else {
                    if (vn >= 1 && vn < kPlanMaxW) { w_sum[vn] += vms; w_n[vn] += 1; }
                    if (boot_w > 3) --boot_w;    // promoted plans never benefit from widths < 3
                    else boot_w = 0;
                }
                ls_n += 1; ls_r += vn; ls_rr += (double)vn * vn;
                ls_y += vms; ls_ry += vn * vms;
                const double det = ls_n * ls_rr - ls_r * ls_r;
                if (det > 1e-9) {
                    const double c1 = (ls_n * ls_ry - ls_r * ls_y) / det;
                    const double c0 = (ls_y - c1 * ls_r) / ls_n;
                    // A negative marginal row cost is noise, not a measurement; keep the previous
                    // fit (or stay unfitted) rather than planning off it.
                    if (c1 > 0 && c0 > 0) { fit_c1 = c1; fit_c0 = c0; fit_ok = true; }
                }
                plan_rows_sum += vn; plan_steps++;
                while (accept < vn - 1 && block[accept + 1] == posterior[accept]) ++accept;
                keep = accept + 1;
                // Feed the achieved rate. The draft is part of the step it paid for, so it is in
                // the denominator here even though it is not in plan_cost().
                plan_tok += (double)keep;
                plan_ms  += vms + fit_d;
                // Calibration counters. Position i is only OBSERVED when the prefix accepted and
                // the plan was wide enough to check it; beyond that `accept` is censored by the
                // plan, not by the draft, and counting it as a miss would teach the plan to keep
                // narrowing itself.
                if (kPlanCal && have_conf) {
                    for (int i = 1; i <= active_proposal_depth && i <= vn - 1 && i < kPlanMaxW; i++) {
                        if (accept < i - 1) break;
                        cal_pred[i] += 1.0 / (1.0 + std::exp(-(double)draft_confidence[i]));
                        if (accept >= i) cal_hit[i] += 1.0;
                    }
                }
            }
            plan_vn = vn;
        } else {
            vfail = p0 < 0;
            posterior[0] = p0;
        }
        // Confidence-gated early stop (opt-in, off by default -- SPARKINFER_DFLASH_CONFIDENCE_GATE
        // sets the raw-logit threshold, no default because there is no calibration data yet for
        // this checkpoint's actual accept-rate distribution against THIS target). DSpark's
        // confidence head predicts position i's own accept probability; comparing raw logits
        // against a raw-logit threshold is equivalent to comparing sigmoid(logit) against
        // sigmoid(threshold), so no sigmoid is needed here. This forfeits an occasional accept
        // (a position gated out that would have matched) in exchange for skipping a target
        // forward predicted unlikely to pay off -- a lossless-preserving trade since forfeiting an
        // accept just means treating it as rejected, never emitting an unverified token.
        static const float kConfidenceGate = []{
            const char* e = getenv("SPARKINFER_DFLASH_CONFIDENCE_GATE");
            return e ? (float)atof(e) : 0.f;
        }();
        static const bool confidence_gate_on = getenv("SPARKINFER_DFLASH_CONFIDENCE_GATE") != nullptr;
        if (!compact_verify && !draft_idle && !vfail && block[1] == p0) {
            for (int i = 1; i <= active_proposal_depth; i++) {
                if (i > 1 && confidence_gate_on && draft_confidence[i] < kConfidenceGate) break;
                set_dflash_capture_row(i);
                auto _tfi = std::chrono::steady_clock::now();
                const int p = forward_token(block[i], start + i, true);
                if (kTiming) { t_fwd_ms += ms_since(_tfi); n_fwd++; }
                if (p < 0) { vfail = true; break; }
                posterior[i] = p;
                accept = i;
                keep = i + 1;
                if (i < active_proposal_depth && block[i + 1] != p) break;
            }
        }
        // tp=2: the batched verify (verify_rows_tp) only ever fails by DECLINING, before any state
        // changes (its scratch did not fit) -- hand the request to ordinary decode at this exact
        // point, as a split-tier stop does, instead of aborting it.
        if (vfail && compact_verify && hooks && tp_active()) { spec_tier_stop = true; break; }
        if (vfail) { fprintf(stderr, "[dflash] verify failed at start=%d\n", start); spec_failed = true; break; }
        // Climb on a full-block accept, decay on anything less. The old rule latched the score at
        // the engage threshold for any partial accept of >= 2 tokens, which kept the batched path
        // armed through low-acceptance stretches -- precisely where it costs throughput. Decaying
        // instead means a prompt the draft handles badly drops back to the token loop within a
        // couple of steps and stays there, while a prompt it handles well re-arms just as quickly.
        // Ramp from zero rather than seeding on the first step. Seeding armed the batched path off
        // a single lucky block, which at 512-ctx (tau 2.19, where batching is 31% SLOWER) cost 5.6%
        // before the EMA had enough evidence to back off. Ramping costs ~3 token-loop steps at
        // short context and keeps 512 on the token loop throughout.
        // Draft-quality probe (SPARKINFER_DSPARK_PROBE=1). Prints what the draft PROPOSED against
        // what the target produced, per step, now that accept/keep are final. tau being invariant
        // to proposal depth says positions >=2 never land; this distinguishes plausible-but-wrong
        // proposals (conditioning subtly off) from structurally wrong ones (repeated,
        // mask-adjacent, out of vocab). Only posterior[0..accept] is known on the token loop --
        // verification stops at the first mismatch -- so unknown slots print as -1.
        if (getenv("SPARKINFER_DSPARK_PROBE")) {
            fprintf(stderr, "[probe] step=%d start=%d seed=%d accept=%d draft=[", step_no, start,
                    block[0], accept);
            for (int i = 1; i <= active_proposal_depth; i++) fprintf(stderr, "%d ", block[i]);
            fprintf(stderr, "] target=[");
            for (int i = 0; i <= active_proposal_depth; i++)
                fprintf(stderr, "%d ", (compact_verify || i <= accept) ? posterior[i] : -1);
            fprintf(stderr, "]\n");
        }
        keep_ema8 = keep_ema8 - (keep_ema8 >> 3) + keep;
        ++step_no;
        // "Full block" now means "everything we asked to be verified was accepted", which is
        // the planned width, not the static depth -- otherwise a step the planner deliberately
        // narrowed would read as a partial accept and decay the engage score.
        compact_score = keep == plan_vn
                      ? std::min(compact_score + 1, kBlockScore + 1)
                      : std::max(compact_score - 1, 0);
        // Discover predictable workloads instead of permanently classifying them by context
        // length. Promotion requires the verifier to reach the deepest available proposal twice;
        // a lucky first block therefore cannot send ordinary chat straight to depth 7. Once wide,
        // three consecutive plans of at most two rows are evidence that the confidence/cost model
        // sees no value in the extra draft rows, so step back toward the length-tuned starting
        // depth. Changing depth never changes correctness: every accepted row remains verified by
        // the target, and declining a proposal merely emits it through a later target step.
        if (kAdaptiveProposalDepth) {
            const bool deepest_landed = plan_vn == active_proposal_depth + 1 &&
                                        accept == active_proposal_depth;
            if (active_proposal_depth == B && deepest_landed)
                // A deepest-row hit is strong evidence that truncating the next verify would
                // throw away useful work. Eight steps lets highly predictable streams remain at
                // their measured full-width optimum; any miss still starts decaying the credit,
                // and low-confidence streams never earn it in the first place.
                full_width_credit = 8;
            else if (full_width_credit > 0)
                --full_width_credit;
            // A uniformly high-confidence full block can count as the second promotion hit, but
            // only during early workload discovery. The lower bound rejects a lucky JSON block at
            // generation step 1 (widening there lost 5.8%); the upper bound rejects late chat hits
            // that have too few tokens left to repay widening (2.5% slower). step_no was incremented
            // above, so [5, 16] corresponds to generation steps 4..15. Weak-confidence blocks and
            // checkpoints without a confidence head retain the ordinary two-hit rule.
            constexpr int kConfidenceDiscoveryStart = 1;
            constexpr int kConfidenceDiscoveryEnd = 16;
            // The depth-4 4k tier already exposes enough rows: accelerating its final 4->7 jump
            // was neutral overall and 0.7% slower on repetition/counting. Apply this only where
            // long-context cost starts the controller at depth 2 or 3.
            bool high_confidence_full = kConfidencePromotionOn && kInitialProposalDepth <= 3 &&
                                        step_no >= kConfidenceDiscoveryStart &&
                                        step_no <= kConfidenceDiscoveryEnd;
            if (high_confidence_full) {
                // Entering wider mode needs stronger evidence: a 32k math transient had a 3.14
                // minimum and widening lost 3%. Once depth 4 is earned, repetitive streams have a
                // 3.93 minimum at the profitable 4->7 step. This staged boundary measured +9.3%
                // to +9.8% on 16k/32k repetition and counting, with other workloads unchanged.
                const float confidence_floor = active_proposal_depth <= 3 ? 4.5f : 3.0f;
                for (int i = 1; i <= active_proposal_depth; ++i) {
                    if (!std::isfinite(draft_confidence[i]) ||
                        draft_confidence[i] < confidence_floor) {
                        high_confidence_full = false;
                        break;
                    }
                }
            }
            // A shallow full accept by itself is common in ordinary code/chat and was promoting
            // those streams away from their measured depth-3 optimum. Require the confidence
            // head to corroborate the initial workload classification; once depth 4 has been
            // earned, the ordinary landed-prefix evidence remains sufficient for 4 -> 7.
            const bool initial_promotion_evidence = active_proposal_depth > 3 ||
                                                    high_confidence_full;
            const bool promotion_evidence = deepest_landed || high_confidence_full;
            depth_promote_run = promotion_evidence && initial_promotion_evidence
                                    ? depth_promote_run + (high_confidence_full ? 2 : 1)
                                    : std::max(depth_promote_run - 1, 0);
            depth_demote_run = (active_proposal_depth > kInitialProposalDepth && plan_vn <= 2)
                                   ? depth_demote_run + 1 : 0;
            if (depth_promote_run >= 2 && active_proposal_depth < B) {
                const bool early_direct_wide = high_confidence_full &&
                                               active_proposal_depth <= 3 && step_no <= 5;
                active_proposal_depth = early_direct_wide ? B
                                        : (active_proposal_depth < 4 ? std::min(4, B) : B);
                // Force the planner to measure the newly exposed widths before pricing them.
                boot_w = active_proposal_depth + 1;
                depth_promote_run = 0;
                depth_demote_run = 0;
            } else if (depth_demote_run >= 3) {
                active_proposal_depth = active_proposal_depth > 4
                                            ? std::max(4, kInitialProposalDepth)
                                            : kInitialProposalDepth;
                if (boot_w > active_proposal_depth + 1) boot_w = active_proposal_depth + 1;
                depth_promote_run = 0;
                depth_demote_run = 0;
            }
        }
        // forward_token() synchronizes after sampling, so the accepted capture rows are already
        // stable. The draft consumes only this newly accepted suffix; its KV cache retains all
        // earlier context. Hand the capture buffer over directly instead of copying it to a second
        // scratch allocation, stashing another unused full-context copy, and synchronizing again.

        const size_t emitted_before = out.size();
        bool stop = false;
        for (int i = 0; i < keep && (int)out.size() < max_new; i++) {
            out.push_back(block[i]);
            if (!ignore_eos &&
                (block[i] == s.cfg.eos_id || (s.cfg.eos_id2 >= 0 && block[i] == s.cfg.eos_id2))) {
                stop = true;
                break;
            }
        }
        bool eos_next = false;
        if (!stop) {
            // Bonus token becomes the next block seed (emitted on the following iteration).
            next = posterior[accept];
            if (!ignore_eos &&
                (next == s.cfg.eos_id || (s.cfg.eos_id2 >= 0 && next == s.cfg.eos_id2))) {
                if ((int)out.size() < max_new) out.push_back(next);
                eos_next = true;
            }
        }
        if (hooks) {
            step_lock.unlock();
            if (out.size() > emitted_before &&
                !hooks->on_tokens(out.data() + emitted_before, (int)(out.size() - emitted_before)))
                spec_stopped = true;
        }
        if (stop || eos_next) { spec_finished = true; break; }

        start += keep;
        accept_sum += (double)keep;
        steps++;
        target_hidden = s.dflash_hidden;
        th_len = keep;
        th_start = 0;
        if (gov) gov->pace();
        if (spec_stopped) break;
    }
    auto t_end = std::chrono::steady_clock::now();
    if (kTiming && !stats && steps > 0)   // the serving path passes no stats: report here
        fprintf(stderr, "[dspark] steps=%ld mean_accept=%.3f | draft %.2f ms x%ld | token-loop fwd "
                        "%.2f ms x%ld | batched verify %.2f ms x%ld\n",
                (long)steps, accept_sum / steps, n_draft ? t_draft_ms / n_draft : 0.0, n_draft,
                n_fwd ? t_fwd_ms / n_fwd : 0.0, n_fwd, n_batched ? t_batched_ms / n_batched : 0.0,
                n_batched);
    if (stats) {
        stats->steps = steps;
        stats->mean_accept = steps > 0 ? accept_sum / steps : 0;
        stats->ttft_s = std::chrono::duration<double>(t1 - t0).count();
        if (kProfile) cudaProfilerStop();
        if (kTiming) {
            const double fpc = n_fwd ? t_fwd_ms / n_fwd : 0.0;
            fprintf(stderr, "[timing] draft   %8.3f ms/call  n=%ld\n", n_draft ? t_draft_ms / n_draft : 0.0, n_draft);
            fprintf(stderr, "[timing] fwd_tok %8.3f ms/call  n=%ld  (token-loop verify)\n", fpc, n_fwd);
            if (plan_steps)
                fprintf(stderr, "[timing] plan    %8.3f rows/step (of %d)  fit C0=%.3f c1=%.3f ms\n",
                        (double)plan_rows_sum / plan_steps, kProposalDepth + 1, fit_c0, fit_c1);
            if (plan_steps) {
                fprintf(stderr, "[timing] widths ");
                for (int w = 1; w < kPlanMaxW; ++w)
                    if (w_n[w] > 0)
                        fprintf(stderr, " w%d=%ld@%.3fms", w, w_n[w], w_sum[w] / w_n[w]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "[timing] batched %8.3f ms/call  n=%ld  = %.2f forwards\n",
                    n_batched ? t_batched_ms / n_batched : 0.0, n_batched,
                    (n_batched && fpc > 0) ? (t_batched_ms / n_batched) / fpc : 0.0);
        }
        stats->decode_s = std::chrono::duration<double>(t_end - t_decode0).count();
    }
    if ((int)out.size() >= max_new) spec_finished = true;
    if (hooks) engine_lock.lock();   // the teardown below frees graphs and capture buffers
    if (resume) {
        resume->engaged = true;
        resume->finished = spec_finished;
        resume->failed = spec_failed;
        resume->position = start;
        resume->tier_boundary = spec_tier_stop;
        resume->next_token = next;
        resume->emitted = (int)out.size();
    }
    if (!hooks) close_session(sid);   // an engine session stays with its job
    if (th_scratch) dflash_cap_free(th_scratch);
    // Verify graphs bake pointers into their request-sized arena. They are useful only for this
    // generation; retaining them steals enough VRAM from a following 32K prefill to change its
    // scratch path and, for the recurrent GDN stack, its result.
    dflash_release_verify_cache();
    set_dflash_capture(false, {}, 0);
    // The caller continues this session with ordinary decode; nothing captured while hidden-state
    // capture was on may be replayed by it.
    if (hooks) invalidate_decode_graph();
    draft.reset();
    return out;
}

// (dual-GPU C2) One segmented verify for a speculative group: both ranks run it (mirrored, as
// batched_forward). The leader captures; with a split capture rank 1 captures its own columns too,
// into the twins of the leader's destinations. Returns n, or -1 when declined (nothing changed).
int Qwen35Model::spec_group_verify(const int* ids, int n, const int* row_pos,
                                   const uint64_t* row_seq, int seg_n, void* const* seg_capture,
                                   int* out_argmax, int* seg_keep, const SpecSampleRow* row_sample,
                                   const uint32_t* const* row_mask) {
    if (!tp_active() || seg_n < 1 || n < 1) return -1;
    std::vector<int> peer_out(n), peer_keep(seg_n);
    void* const* peer_capture = dflash_capture_split() ? seg_capture : nullptr;
    TP_MIRROR(spec_group_verify(ids, n, row_pos, row_seq, seg_n, peer_capture, peer_out.data(),
                                peer_keep.data(), row_sample, row_mask));
    return tp_rows_forward(ids, n, 0, row_pos, row_seq, nullptr, out_argmax, seg_n, seg_capture,
                           seg_keep, row_sample, row_mask);
}

// Frees the per-segment GDN snapshots of the segmented verify (both ranks).
void Qwen35Model::spec_group_release() {
    TP_MIRROR(spec_group_release());
    std::lock_guard<std::recursive_mutex> device_lock(p_->device_mu);
    Impl& s = *p_;
    cudaStreamSynchronize(s.stream);
    for (float* b : s.vr_seg_snap_lin) cudaFree(b);
    for (bf16* b : s.vr_seg_snap_conv) cudaFree(b);
    s.vr_seg_snap_lin.clear();
    s.vr_seg_snap_conv.clear();
}

void Qwen35Model::note_plain_decode(int rows, double ms) {
    if (rows < 1 || rows > 4 || !(ms > 0)) return;
    double& avg = p_->plain_step_ms[rows];
    avg = avg > 0 ? 0.95 * avg + 0.05 * ms : ms;
}

double Qwen35Model::plain_decode_ms(int rows) const {
    rows = std::max(1, std::min(rows, 4));
    for (int r = rows; r >= 1; r--)
        if (p_->plain_step_ms[r] > 0) return p_->plain_step_ms[r];
    return 20.0;   // tp=2 on 2x RTX 5060 Ti decodes ~50 tok/s, about the same for 1-4 sessions
}

bool Qwen35Model::spec_group_supported() const {
    return p_->dflash_draft && tp_active() && p_->tp_rank == 0 && p_->cfg.hybrid;
}

namespace {
// Prompt lookup (plan 06, N): the most recent occurrence of the longest suffix (nmin..nmax
// tokens) of h[0, len) that ends before the suffix does, and the k tokens that followed it into
// out[]. The occurrence may overlap the suffix: a period-p repetition (a list marker, an indent)
// matches p back, and its continuation repeats with that period past the end of the history.
// Returns the match length (0: none).
int ngram_lookup(const int* h, int len, int nmin, int nmax, int k, int* out) {
    if (len < nmin + 1) return 0;
    const int last = len - 1;
    int best = 0, best_end = -1;
    for (int e = last - 1; e >= nmin - 1; e--) {
        if (h[e] != h[last]) continue;
        int m = 1;
        while (m < nmax && e - m >= 0 && h[e - m] == h[last - m]) m++;
        if (m >= nmin && m > best) {
            best = m;
            best_end = e;
            if (m == nmax) break;   // scanning backwards: the most recent of the longest
        }
    }
    if (best == 0) return 0;
    const int period = last - best_end;
    for (int i = 0; i < k; i++)
        out[i] = best_end + 1 + i < len ? h[best_end + 1 + i] : out[i - period];
    return best;
}
}  // namespace

// (dual-GPU C2) Speculative decoding of several engine sessions at once. Every step drafts each
// session in turn on its own draft KV state (the draft is ~3 ms a block), then verifies all
// blocks in ONE segmented tp pass (spec_group_verify): the weights stream once for every
// session's rows instead of once per session. Jobs join between steps (hooks.poll), are prefilled
// with hidden-state capture and drafted once on the spot; jobs leave on EOS / max_new / their
// on_tokens returning false, each reported through hooks.on_done with its SpecResume filled the
// way dflash_generate fills it, so the engine resumes any unfinished job with ordinary decode.
//
// Policy is deliberately simple next to dflash_generate's single-session planner: every session
// of a step verifies the same depth (see depth_for), capped at rows / S - 1 so the step's rows fit
// the tp rows pass.
void Qwen35Model::dflash_generate_group(std::vector<SpecGroupJob*> jobs,
                                        const SpecGroupHooks& hooks) {
    Impl& s = *p_;
    const bool ignore_eos = [] {
        const char* e = getenv("SPARKINFER_BENCH_IGNORE_EOS");
        return e && e[0] == '1';
    }();
    // Depth: 6 for a lone session, 4 once sessions share a step. Measured (HyperQwen cohort,
    // 512 tokens, e2e tok/s, depth 4 / 5 / 6 / 7): C1 94.3 / 94.7 / 92.7 / 93.4 (flat), C2
    // 150.4 / 138.8 / 136.0 / 133.3, C4 211.0 / 202.5 / 188.4 / 178.5 -- with several sessions
    // a deep row costs a verify row per session and lands less often than it costs.
    // SPARKINFER_SPEC_GROUP_DEPTH fixes one depth for every group size.
    static const int kDepthEnv = [] {
        const char* e = getenv("SPARKINFER_SPEC_GROUP_DEPTH");
        const int v = e ? atoi(e) : 0;
        return v < 0 ? 0 : v;
    }();
    // From 12288 positions on, the depth the single-session path measured there (dflash_generate,
    // "the 12288..32768 band takes depth 2"): a verify row reads the whole KV, so the deep rows
    // cost more than they land. Measured here on an opencode-like replay (multiturn_bench.py),
    // depth 6 at 14k-55k: 28 -> 40 ms a step for 1.8-2.8 tokens. SPARKINFER_SPEC_GROUP_LONG_DEPTH
    // (default 2; 0 keeps depth_for's).
    // A draft that asks for one depth at every context (DFlash2: its whole block) gets it here
    // unless these two say otherwise; its long depth is then off.
    static const int kLongDepthEnv = [] {
        const char* e = getenv("SPARKINFER_SPEC_GROUP_LONG_DEPTH");
        return e ? std::max(0, atoi(e)) : -1;
    }();
    // Speculation must beat ordinary decode, which costs about the same per step for one to four
    // greedy sessions (packed). Over the group's last kGainWindow steps, gain = (tokens committed
    // x an ordinary step's cost for those sessions, as the engine measured it) / (the steps' own
    // time x sessions); below SPARKINFER_SPEC_GROUP_MIN_GAIN (default 1.0; 0 = off) the group
    // ends and its sessions decode ordinarily. A window, not a short average: acceptance swings
    // step to step, and ending is final for the request (a 10-step average ended a group whose
    // gain was 1.5). Measured without the check at C2 on the replay: 38 tok/s per request
    // speculating against 48 decoding ordinarily.
    static const double kMinGain = [] {
        const char* e = getenv("SPARKINFER_SPEC_GROUP_MIN_GAIN");
        return e ? std::max(0.0, atof(e)) : 1.0;
    }();
    // Prompt lookup (plan 06, N; HyperQwen's lookup-augmented drafting). Each step, a session
    // whose last tokens (SPARKINFER_NGRAM_NMIN..NMAX, default 6..12) occurred earlier in its
    // prompt or output takes what followed them in place of the draft's tokens. While a copy is
    // running -- the previous step accepted every row and agreed with the lookup -- the step
    // verifies up to kNgramDeep lookup rows instead of depth_for's depth (2 from 12288 on, 4 for
    // several sessions): a copied file or quoted tool output lands up to 16 tokens a step there.
    // A step's depth is shared, so it goes long only when every session has a running copy.
    // SPARKINFER_NGRAM=0 turns it off.
    static const bool kNgram = [] {
        const char* e = getenv("SPARKINFER_NGRAM");
        return !e || atoi(e) != 0;
    }();
    static const int kNgramMin = [] {
        const char* e = getenv("SPARKINFER_NGRAM_NMIN");
        return e ? std::max(2, atoi(e)) : 6;
    }();
    static const int kNgramMax = [] {
        const char* e = getenv("SPARKINFER_NGRAM_NMAX");
        return e ? std::max(kNgramMin, atoi(e)) : std::max(kNgramMin, 12);
    }();
    // While a copy runs the deep block takes up to SPARKINFER_NGRAM_DEPTH lookup tokens (default 15,
    // HyperQwen's DFLASH_TOKENS), past the draft's own block; the verify's 32 rows and each
    // session's capture buffer are what bound it.
    static const int kNgramDeep = [] {
        const char* e = getenv("SPARKINFER_NGRAM_DEPTH");
        return std::max(1, std::min(e ? atoi(e) : 15, kTpVerifyRows - 1));
    }();
    long ng_steps = 0, ng_long_steps = 0, ng_long_tokens = 0;
    constexpr int kGainWindow = 32;
    double gain_won[kGainWindow] = {}, gain_cost[kGainWindow] = {};
    long gain_steps = 0;
    static const bool kTiming = getenv("SPARKINFER_DSPARK_TIMING") != nullptr;
    DFlashDraftModel& draft = *s.dflash_draft;
    const DFlashDraftConfig& dc = draft.config();
    const int B = dc.block_size;
    const int mask_id = dc.mask_token_id;
    const int H = s.cfg.hidden;
    const int R = kTpVerifyRows;
    const int BB = kNgram ? std::max(B, kNgramDeep) : B;   // a block's proposals, lookup included

    struct G {
        SpecGroupJob* job = nullptr;
        int n = 0;               // prompt length
        int start = 0;           // committed position (prompt + emitted)
        int next = -1;           // verified, not yet emitted
        int state = -1;          // draft KV state
        bf16* cap = nullptr;     // [BB + 1][row stride] capture rows of the last verify
        int th_len = 0;          // rows of `cap` the next draft ingests
        bool predrafted = false; // block[1..] already holds the join's first draft
        std::vector<int> block, out, draft_out;
        std::vector<int> hist;   // prompt + out: what the lookup searches
        std::vector<int> lk;     // this step's lookup continuation (BB tokens)
        int lk_len = 0;          // its match length (0: none)
        bool lk_run = false;     // the last step accepted every row and the lookup agreed
        std::vector<uint32_t> masks;   // constrained: each verify row's allowed tokens, [T][words]
        std::vector<float> q_p;        // rejection sampling: the draft's q per proposal, [B][kDraftQTab]
        std::vector<int> q_ids;
        bool done = false;
    };
    std::vector<G> gs;
    bool group_ok = true;
    long steps = 0, seg_steps = 0;
    double accept_sum = 0, t_draft = 0, t_verify = 0, t_mask = 0, t_step_all = 0;
    auto ms_since = [](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    Qwen35Model* tp_draft_peer = (s.tp_peers.size() == 2) ? s.tp_peers[1] : nullptr;
    {
        std::lock_guard<std::recursive_mutex> lk(s.device_mu);
        const bool use_q4_head = lm_head_quant_type() == 12 && lm_head_weights();
        const void* draft_head = use_q4_head ? lm_head_weights()
                               : (s.dflash_lm_head ? s.dflash_lm_head : lm_head_weights());
        const int draft_head_type = use_q4_head ? lm_head_quant_type()
                                  : (s.dflash_lm_head ? s.dflash_lm_head_type : lm_head_quant_type());
        draft.set_shared_weights(embed_weights(), draft_head, draft_head_type,
                                 tp_draft_peer ? s.cfg.vocab / 2 : s.cfg.vocab, s.cfg.hidden);
        if (tp_draft_peer)
            draft.set_embed_split(s.cfg.vocab / 2, tp_draft_peer->embed_weights(),
                                  tp_draft_peer->tp_rank_view().device);
        draft.set_head_fp4(s.w.lm_head_fp4, s.w.lm_head_fp4_sf, s.w.lm_head_fp4_alpha);
        if (tp_draft_peer)   // DFlash2 scores the second card's vocab half there
            draft.set_peer_head([pm = tp_draft_peer] {
                DFlashDraftModel::HeadRef h;
                h.lm_head = pm->lm_head_weights();
                h.lm_head_type = pm->lm_head_quant_type();
                h.fp4_w = pm->p_->w.lm_head_fp4;
                h.fp4_sf = pm->p_->w.lm_head_fp4_sf;
                h.fp4_alpha = pm->p_->w.lm_head_fp4_alpha;
                return h;
            }, s.cfg.vocab - s.cfg.vocab / 2);
        draft.ensure_quant();
        // (dual-GPU) A draft whose fc is split by columns reads each card's own half of the
        // capture (set_dflash_capture_split); rank 1 finds its rows through dflash_cap_peer.
        set_dflash_capture_split(tp_draft_peer && draft.fc_split());
        draft.set_peer_hidden_map([this](const void* p) { return dflash_cap_peer(p); });
        s.final_seqlen_hint = -1;
    }

    // A sampled session's draft walk (DFlash2) draws each proposal with the noise its verify row
    // will: block[i] sits at the request's step start + i - n, so the walk's first proposal
    // (block[1]) is at start + 1 - n.
    // SPARKINFER_SPEC_REJECTION=1: sampled sessions accept a drafted token with probability
    // min(1, p / q) and draw the first rejected position from max(0, p - q) (speculative sampling),
    // instead of accepting only what plain sampled decode would draw. The output has the target's
    // distribution, but not plain decode's tokens for the same seed. DFlash2 only (its walk
    // reports q); off by default.
    static const bool kRejection = [] {
        const char* e = getenv("SPARKINFER_SPEC_REJECTION");
        return e && e[0] == '1';
    }();
    auto walk_of = [&](const SpecGroupJob& job, int start, int n, G* gq = nullptr) {
        DFlashDraftModel::DraftWalk w;
        if (job.temperature > 0.f) {
            w.temperature = job.temperature;
            w.top_k = job.top_k;
            w.top_p = job.top_p;
            w.seed = job.seed;
            w.step0 = (unsigned long long)(start + 1 - n);
            if (kRejection && gq && dc.dflash2) {
                constexpr int QT = dflash_kernels::kDraftQTab;
                gq->q_p.assign((size_t)(BB + 1) * QT, 0.f);
                gq->q_ids.assign((size_t)(BB + 1) * QT, -1);
                w.q_p = gq->q_p.data();
                w.q_ids = gq->q_ids.data();
            }
        }
        return w;
    };
    auto fill_resume = [&](G& g, bool finished, bool failed) {
        SpecResume& r = g.job->resume;
        r = SpecResume{};
        r.engaged = true;
        r.finished = finished || (int)g.out.size() >= g.job->max_new;
        r.failed = failed;
        r.position = g.start;
        r.next_token = g.next;
        r.emitted = (int)g.out.size();
    };
    auto drop = [&](G& g) {   // device_mu held
        if (g.state >= 0) draft.kv_state_free(g.state);
        if (g.cap) dflash_cap_free(g.cap);
        g.state = -1;
        g.cap = nullptr;
        g.done = true;
    };
    // The prompt's capture rows, on both cards with a split capture (device_mu held).
    auto free_context = [&] {
        if (dflash_capture_split()) {
            Impl& q = *s.tp_peers[1]->p_;
            tp_run_with_peer(s.tp_peers[1]->tp_rank_view().device, [&] {
                if (q.dflash_context) cudaFree(q.dflash_context);
                q.dflash_context = nullptr;
                q.dflash_ctx_cap = 0;
            }, nullptr);
        }
        if (s.dflash_context) {
            cudaFree(s.dflash_context);
            s.dflash_context = nullptr;
            s.dflash_ctx_cap = 0;
        }
    };

    // Prefill + capture + first draft of a joining job. false: it could not start (resume not
    // engaged: the engine prefills it ordinarily); nothing of its session was touched then
    // except, possibly, its prefill (resume.engaged stays false only before the prefill).
    auto join = [&](SpecGroupJob* job, int D) -> bool {
        G g;
        g.job = job;
        job->resume = SpecResume{};
        const std::vector<int>& prompt = *job->prompt;
        g.n = (int)prompt.size();
        if (g.n < 1 || job->max_new < 1) return false;
        const long need = (long)g.n + job->max_new + 2L * (B + 1);
        // No limit on the context (plan 06, W6): the draft's state slides and its capture is at
        // most the prompt's last 4096 rows past 12288, so nothing on the draft grows with it.
        // SPARKINFER_DSPARK_MAX_CTX now bounds only the rows its first block ingests (checked
        // below, once the capture's first row is known).
        std::lock_guard<std::recursive_mutex> lk(s.device_mu);
        // SPARKINFER_DSPARK_CAPTURE_MAX=N: the draft ingests at most the prompt's last N rows
        // (it then windows its attention, see DFlashDraftModel ctx_lo). The capture holds
        // n_cap * H bf16 per row (51 KB on the 27B) on the draft's card -- 0.6 GB for a whole
        // 12k prompt. Off by default: measured below 12288 (6-10k-token prompts, 384 tokens, 6
        // prompts), N=4096 cost acceptance 2.97 -> 2.37 and 97 -> 81 tok/s; the old context does
        // buy acceptance there. For a card too tight to hold the capture at all.
        static const int kCaptureMax = [] {
            const char* e = getenv("SPARKINFER_DSPARK_CAPTURE_MAX");
            return e ? std::max(0, atoi(e)) : 0;
        }();
        // A prefix-cache hit (job->start > 0) prefills, and so captures, only [start, n).
        const int h = std::max(0, std::min(job->start, g.n - 1));
        int capture_start = 0;
        if (kCaptureMax > 0 && g.n > kCaptureMax) capture_start = g.n - kCaptureMax;
        else if (g.n >= 12288) capture_start = g.n - 4096;
        capture_start = std::max(capture_start, h);
        if (g.n - capture_start + 2L * (B + 1) > dc.max_seq) return false;
        // The context buffer only feeds this join's first draft (the prompt's rows); every later
        // step drafts from the session's own `cap` rows. Sizing it for the generation too held
        // ~51 KB per output token on the draft's card for nothing -- 0.8 GB at max_tokens 16k.
        set_dflash_capture(true, dc.target_layer_ids, B + 1, capture_start,
                           std::min(s.cfg.max_seq, g.n + 1));
        if (!dflash_context_buffer() || !dflash_hidden_buffer()) return false;
        const size_t cap_bytes = (size_t)(BB + 1) * dflash_hidden_row_stride() * sizeof(bf16);
        g.cap = static_cast<bf16*>(dflash_cap_alloc(cap_bytes));
        if (!g.cap) return false;
        // A state slides (DFlashDraftModel::kv_state_create), so it never needs more than the
        // sliding capacity, whatever the context: 123 MB a card instead of 20 KB a token.
        g.state = draft.kv_state_create((int)std::min<long>(need, draft.kv_slide_capacity()));
        if (g.state < 0) { dflash_cap_free(g.cap); g.cap = nullptr; return false; }
        const int budget = session_token_budget(prompt.size(), job->max_new + B, s.cfg.max_seq);
        invalidate_decode_graph();
        bool kv_ok = s.kv->allocate(job->seq_id, budget);
        if (kv_ok && tp_draft_peer) {
            std::lock_guard<std::recursive_mutex> peer_lock(tp_draft_peer->p_->device_mu);
            kv_ok = tp_draft_peer->p_->kv->allocate(job->seq_id, budget);
            if (!kv_ok) fprintf(stderr, "[tp] spec group: rank-1 KV grow failed where rank 0 succeeded\n");
        }
        if (!kv_ok) { drop(g); return false; }
        activate_session(job->seq_id);
        reset_mrope_offset();
        int next = -1;
        // A hit's recurrent state has moved past `start` once anything is prefilled; a job given
        // back then is prefilled ordinarily from `start` again, so put the state back first.
        auto undo_prefill = [&]() -> bool {
            if (h == 0) return true;   // an ordinary prefill from 0 resets the state itself
            if (job->start_state && restore_recurrent_state(job->seq_id, *job->start_state)) return true;
            fprintf(stderr, "[spec-group] could not restore the prefix state of a declined join\n");
            return false;
        };
        // Prefix-cache checkpoints, as step_job's prefill takes them: prefill to each, snapshot
        // the recurrent state, continue. Without them a speculated turn left nothing in the cache
        // and the next turn of the conversation re-prefilled all of it.
        std::vector<std::pair<int, RecurrentStateSnapshot>> ckpts;
        // Each range runs as ingest_prompt_range runs it -- the batched path when it is enabled
        // for the range's length (from 0, the whole prefix), the token loop otherwise -- so a
        // speculated request computes exactly what an ordinary one would. A batched range that
        // declines (no scratch) declines the join instead of finishing token by token.
        bool declined = false;
        auto prefill_range = [&](int a, int b) -> int {
            if (batched_prefill_windowed_enabled(s.gguf, s.cfg, a == 0 ? b : b - a, s.kv)) {
                int d = 0;
                const int r = a == 0 ? prefill_batched_chunked(prompt.data(), b, false, &d)
                                     : prefill_batched_resume(prompt.data(), a, b, false, &d);
                if (r < 0) declined = true;
                return r;
            }
            int r = -1;
            for (int i = a; i < b; i++) {
                set_dflash_capture_row(0);
                const bool sample = (i + 1 == b);
                const int t = forward_token(prompt[i], i, sample);
                dflash_stash_capture(i);
                if (sample) r = t;
            }
            return r;
        };
        int pos = h;
        if (job->checkpoints && job->on_checkpoint) {
            for (int ck : *job->checkpoints) {
                if (ck <= pos || ck >= g.n || ck % s.kv->block_size() != 0) continue;
                if (prefill_range(pos, ck) < 0) break;
                pos = ck;
                RecurrentStateSnapshot snap;
                if (snapshot_recurrent_state(job->seq_id, snap)) ckpts.emplace_back(ck, std::move(snap));
            }
        }
        if (!declined) next = prefill_range(pos, g.n);
        // Engaged from here on (or dropped): the snapshots go to the job only then.
        auto hand_checkpoints = [&] {
            for (auto& c : ckpts) job->on_checkpoint(c.first, c.second);
            ckpts.clear();
        };
        if (declined) {
            // The batched prefill did not fit beside this join's capture rows and draft state (it
            // already narrowed its window as far as it goes). Finishing it token by token takes
            // minutes at 12k (measured: four concurrent 12k joins, ~700 s each), so give the
            // speculative state back and let the engine prefill the request ordinarily, without
            // them. Its session is reused as it is: an ordinary prefill starts again at 0 (after a
            // prefix-cache hit, at the prefix, with its recurrent state put back below).
            fprintf(stderr, "[spec-group] join declined: the batched prefill does not fit beside "
                            "the draft (n=%d); prefilling it ordinarily\n", g.n);
            free_context();
            drop(g);
            if (!undo_prefill()) { fill_resume(g, true, true); hooks.on_done(job); return true; }
            return false;
        }
        if (next < 0 || next >= s.cfg.vocab) {
            drop(g);
            if (!undo_prefill()) { fill_resume(g, true, true); hooks.on_done(job); return true; }
            return false;
        }
        g.start = g.n;
        g.next = next;
        if (kNgram) g.hist.assign(prompt.begin(), prompt.end());
        g.block.assign(BB + 1, mask_id);
        g.block[0] = next;
        draft.kv_state_select(g.state);
        draft.reset();
        std::vector<int> draft_ids(B + 1, 0);
        auto _td = std::chrono::steady_clock::now();
        // After a prefix-cache hit the draft's context starts at the capture's first row: holding
        // the prefix's last positions from the entry's draft snapshot when it has one (restored
        // as far back as leaves room for the new rows), empty otherwise -- positions below are
        // then gone and the draft windows, as after a truncated capture.
        bool first_ok = true;
        int ctx_rows = g.n, ctx_hidden_start = s.dflash_ctx_start;
        // A fresh prompt whose capture starts late (12288 and up) starts the same way, empty at
        // capture_start: passing the whole prompt as the block's context would exceed the draft's
        // max_seq from 16384 on, and the first draft would fail.
        if (h > 0 || capture_start > 0) {
            const auto* snap = static_cast<const DFlashDraftModel::KvSnapshot*>(
                job->start_state && capture_start == h ? job->start_state->draft.get() : nullptr);
            const int state_cap = (int)std::min<long>(need, draft.kv_slide_capacity());
            int from = capture_start;
            if (snap && snap->hi == h) {
                from = std::max(snap->lo, g.n + 2 * (B + 1) + 1 - state_cap);
                // From 12288 on the draft attends 2048 positions: the kept span covers them.
                if (g.n >= 12288) from = std::max(from, g.n - draft.kv_slide_keep());
            }
            if (from >= h) snap = nullptr;
            first_ok = snap ? draft.kv_start_at(from, h, snap) : false;
            if (!first_ok) {
                from = capture_start;
                first_ok = draft.kv_start_at(from, from, nullptr);
            }
            ctx_rows = g.n - capture_start;
            ctx_hidden_start = 0;
            if (kTiming)
                fprintf(stderr, "[spec-group] join after a prefix hit at %d: %d new rows, draft "
                                "context from %d (%s)\n", h, g.n - h, from,
                        snap && from < h ? "snapshot" : "none");
        }
        if (first_ok) {
            draft.set_walk(walk_of(*job, g.start, g.n, &g));
            first_ok = draft.forward_block(dflash_context_buffer(), ctx_rows, g.block.data(),
                                           g.start, draft_ids.data(), nullptr, D, nullptr,
                                           ctx_hidden_start);
        }
        // The prompt's capture rows (51 KB a row, ~0.6 GB at 12k) fed only this first draft;
        // every later step drafts from the session's own `cap`. Give them back now rather than
        // when the group ends, for the prefills of whatever runs meanwhile. The draft has read
        // them: forward_block synchronises its stream (this model's) before it returns.
        free_context();
        // The draft's context at the last checkpoint goes with that prefix-cache entry (plan 06,
        // W4), in pinned host memory: SPARKINFER_DSPARK_SNAPSHOT positions (default 12288, the
        // span the draft attends unwindowed; 0 = off), ~10 KB a position per card. A prefix of
        // 12288 or more is followed by a windowed draft, so only the kept span (4096) goes.
        static const int kDraftSnap = [] {
            const char* e = getenv("SPARKINFER_DSPARK_SNAPSHOT");
            return e ? std::max(0, atoi(e)) : 12288;
        }();
        if (first_ok && kDraftSnap > 0 && !ckpts.empty()) {
            const int ck = ckpts.back().first;
            const int span = ck >= 12288 ? std::min(kDraftSnap, draft.kv_slide_keep()) : kDraftSnap;
            const int lo = std::max(draft.kv_valid_lo(), ck - span);
            auto ks = std::make_shared<DFlashDraftModel::KvSnapshot>();
            if (ck - lo >= 2 * (B + 1) && draft.kv_snapshot(lo, ck, *ks)) {
                ckpts.back().second.draft_bytes = ks->total_bytes();
                ckpts.back().second.draft = std::move(ks);
            }
        }
        hand_checkpoints();
        if (!first_ok) {
            // The prompt is prefilled: hand the job back engaged at its first token.
            fprintf(stderr, "[spec-group] first draft failed (n=%d)\n", g.n);
            drop(g);
            fill_resume(g, false, false);
            hooks.on_done(job);
            return true;
        }
        if (kTiming) t_draft += ms_since(_td);
        for (int i = 1; i <= D; i++) g.block[i] = draft_ids[i];
        g.predrafted = true;
        gs.push_back(std::move(g));
        return true;
    };

    const int kLongDepth = kLongDepthEnv >= 0 ? kLongDepthEnv : (dc.spec_depth > 0 ? 0 : 2);
    auto depth_for = [&](int S, int ctx = 0) {
        int want = kDepthEnv > 0 ? kDepthEnv : dc.spec_depth > 0 ? dc.spec_depth : (S <= 1 ? 6 : 4);
        if (kDepthEnv <= 0 && kLongDepth > 0 && ctx >= 12288) want = std::min(want, kLongDepth);
        return std::max(0, std::min(want, std::min(dc.max_proposals(), R / std::max(S, 1) - 1)));
    };
    for (SpecGroupJob* j : jobs) {
        if (!join(j, depth_for((int)jobs.size()))) {
            // Not engaged: the engine re-prefills it; stop here so it is not left waiting.
            j->resume = SpecResume{};
            hooks.on_done(j);
            group_ok = false;
        }
    }

    std::vector<int> ids, pos, argmax, keep, draft_ids(B + 1, 0);
    std::vector<uint64_t> seq;
    std::vector<SpecSampleRow> smp;
    std::vector<void*> caps;
    std::vector<const uint32_t*> row_mask;
    const int mask_words = (s.cfg.vocab + 31) / 32;
    // A constrained session's verify masks: row t draws the token after block[0..t], so walk the
    // constraint (which stands at the emitted tokens) through block[0] and the drafted tokens,
    // taking the mask before each, then roll the walk back. A drafted token the constraint refuses
    // ends the walk: the row before it cannot draw it, so no later row is kept and their masks
    // are never used. So does a drafted end token: nothing after it is kept, and a constraint that
    // accepted it has terminated (it has no next mask). False: block[0] itself is refused (it was
    // drawn under the mask, so this means the engine's constraint and the group disagree).
    auto build_masks = [&](G& g, int T) -> bool {
        TokenConstraint* c = g.job->constraint;
        g.masks.assign((size_t)T * mask_words, 0xffffffffu);
        if (!c->accept(g.block[0])) return false;
        int walked = 1;
        for (int t = 0; t < T; t++) {
            uint32_t* m = g.masks.data() + (size_t)t * mask_words;
            c->fill_next_mask(m, s.cfg.vocab);
            if (s.cfg.vocab % 32) m[mask_words - 1] &= (1u << (s.cfg.vocab % 32)) - 1;
            if (t + 1 == T) break;
            const int nt = g.block[t + 1];
            if (nt == s.cfg.eos_id || (s.cfg.eos_id2 >= 0 && nt == s.cfg.eos_id2)) break;
            if (!c->accept(nt)) break;
            walked++;
        }
        c->rollback(walked);
        return true;
    };
    while (group_ok) {
        // Active sessions of this step.
        std::vector<G*> act;
        for (G& g : gs) if (!g.done) act.push_back(&g);
        if (act.empty()) {
            // Everyone left: poll once for newcomers, else the group is over.
            std::vector<SpecGroupJob*> joins;
            if (!hooks.poll(joins) || joins.empty()) break;
            for (SpecGroupJob* j : joins)
                if (!join(j, depth_for((int)joins.size()))) { j->resume = SpecResume{}; hooks.on_done(j); group_ok = false; }
            continue;
        }
        const int S = (int)act.size();
        int ctx_max = 0;
        for (G* g : act) ctx_max = std::max(ctx_max, g->start);
        const int D = depth_for(S, ctx_max);
        if (D < 1) break;
        // The lookup runs on the host against prompt + output + the pending token (tens of
        // microseconds at 50k); the deep block needs every session's copy running.
        int Dv = D;
        if (kNgram) {
            int deep = std::min(BB, R / S - 1);
            for (G* g : act) {
                g->lk.assign(BB, 0);
                g->hist.push_back(g->next);
                g->lk_len = ngram_lookup(g->hist.data(), (int)g->hist.size(), kNgramMin,
                                         kNgramMax, BB, g->lk.data());
                g->hist.pop_back();
                if (!(g->lk_len > 0 && g->lk_run)) deep = D;
            }
            Dv = std::max(D, deep);
        }
        const int T = Dv + 1;
        const auto t_step = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::recursive_mutex> lk(s.device_mu);
            auto _td = std::chrono::steady_clock::now();
            // Every session that needs a block drafts in ONE batched draft pass.
            std::vector<DFlashDraftModel::DraftSeg> dsegs;
            std::vector<G*> dg;
            for (G* g : act) {
                if (g->predrafted) { g->predrafted = false; continue; }
                g->block.assign(BB + 1, mask_id);
                g->block[0] = g->next;
                g->draft_out.assign(B + 1, 0);
                DFlashDraftModel::DraftSeg ds;
                ds.state = g->state;
                ds.target_hidden = g->cap;
                ds.ctx_len = g->th_len;
                ds.ids = g->block.data();
                ds.pos0 = g->start;
                ds.out_argmax = g->draft_out.data();
                ds.walk = walk_of(*g->job, g->start, g->n, g);
                dsegs.push_back(ds);
                dg.push_back(g);
            }
            bool draft_failed = !dsegs.empty() &&
                                !draft.forward_blocks((int)dsegs.size(), dsegs.data(), D);
            if (!draft_failed)
                for (G* g : dg)
                    for (int i = 1; i <= D; i++) g->block[i] = g->draft_out[i];
            if (kTiming) t_draft += ms_since(_td);
            if (draft_failed) { fprintf(stderr, "[spec-group] draft failed\n"); break; }
            if (kNgram) {
                for (G* g : act)
                    if (g->lk_len > 0) {
                        ng_steps++;
                        for (int i = 1; i <= Dv; i++) g->block[i] = g->lk[i - 1];
                    }
                if (Dv > D) ng_long_steps++;
            }
            const int n = S * T;
            ids.assign(n, 0); pos.assign(n, 0); seq.assign(n, 0); argmax.assign(n, 0);
            keep.assign(S, 1); caps.assign(S, nullptr);
            row_mask.assign(n, nullptr);
            bool masks_ok = true, any_mask = false;
            const auto _tm = std::chrono::steady_clock::now();
            for (int j = 0; j < S && masks_ok; j++) {
                G& g = *act[j];
                if (!g.job->constraint) continue;
                masks_ok = build_masks(g, T);
                any_mask = true;
                for (int t = 0; t < T && masks_ok; t++)
                    row_mask[t * S + j] = g.masks.data() + (size_t)t * mask_words;
            }
            if (kTiming) t_mask += ms_since(_tm);
            if (!masks_ok) { fprintf(stderr, "[spec-group] constraint refused a verified token\n"); break; }
            for (int j = 0; j < S; j++) {
                caps[j] = act[j]->cap;
                for (int t = 0; t < T; t++) {
                    ids[t * S + j] = act[j]->block[t];
                    pos[t * S + j] = act[j]->start + t;
                    seq[t * S + j] = act[j]->job->seq_id;
                }
            }
            // Sampled sessions: each row draws its token as decode would at that step (the
            // request's count of tokens emitted before it: the first, from the prefill, is 0).
            smp.clear();
            for (int j = 0; j < S && smp.empty(); j++)
                if (act[j]->job->temperature > 0.f) smp.resize(n);
            for (int j = 0; j < S && !smp.empty(); j++) {
                const SpecGroupJob& jb = *act[j]->job;
                for (int t = 0; t < T; t++) {
                    SpecSampleRow& rs = smp[t * S + j];
                    rs.temperature = jb.temperature;
                    rs.top_k = jb.top_k;
                    rs.top_p = jb.top_p;
                    rs.seed = jb.seed;
                    rs.step = (unsigned long long)(act[j]->start + t + 1 - act[j]->n);
                    // Row t checks block[t + 1], the walk's proposal t (its q table t).
                    const G& gg = *act[j];
                    if (kRejection && t + 1 < T && !gg.q_p.empty()) {
                        rs.q_p = gg.q_p.data() + (size_t)t * dflash_kernels::kDraftQTab;
                        rs.q_ids = gg.q_ids.data() + (size_t)t * dflash_kernels::kDraftQTab;
                    }
                }
            }
            auto _tv = std::chrono::steady_clock::now();
            const int r = spec_group_verify(ids.data(), n, pos.data(), seq.data(), S, caps.data(),
                                            argmax.data(), keep.data(),
                                            smp.empty() ? nullptr : smp.data(),
                                            any_mask ? row_mask.data() : nullptr);
            if (kTiming) t_verify += ms_since(_tv);
            if (r != n) { fprintf(stderr, "[spec-group] verify declined (S=%d T=%d)\n", S, T); break; }
        }
        steps++;
        seg_steps += S;
        const double step_ms = ms_since(t_step);
        long step_tokens = 0;
        for (int j = 0; j < S; j++) step_tokens += keep[j];
        // Commit each session's accepted prefix, exactly as dflash_generate does per step.
        for (int j = 0; j < S; j++) {
            G& g = *act[j];
            const int kp = keep[j];
            accept_sum += kp;
            const size_t before = g.out.size();
            bool stop = false;
            for (int i = 0; i < kp && (int)g.out.size() < g.job->max_new; i++) {
                g.out.push_back(g.block[i]);
                if (!ignore_eos && (g.block[i] == s.cfg.eos_id ||
                                    (s.cfg.eos_id2 >= 0 && g.block[i] == s.cfg.eos_id2))) {
                    stop = true;
                    break;
                }
            }
            bool eos_next = false;
            if (!stop) {
                g.next = argmax[(size_t)(kp - 1) * S + j];
                if (!ignore_eos && (g.next == s.cfg.eos_id ||
                                    (s.cfg.eos_id2 >= 0 && g.next == s.cfg.eos_id2))) {
                    if ((int)g.out.size() < g.job->max_new) g.out.push_back(g.next);
                    eos_next = true;
                }
            }
            if (kNgram) {
                g.hist.insert(g.hist.end(), g.out.begin() + before, g.out.end());
                // A running copy: every row landed, and the lookup also had the token after them.
                g.lk_run = g.lk_len > 0 && kp == T && (Dv >= BB || g.next == g.lk[Dv]);
                if (Dv > D) ng_long_tokens += kp;
            }
            bool stopped = false;
            if (g.out.size() > before &&
                !g.job->on_tokens(g.out.data() + before, (int)(g.out.size() - before)))
                stopped = true;
            const bool finished = stop || eos_next || (int)g.out.size() >= g.job->max_new;
            if (!stop && !eos_next) {
                g.start += kp;
                g.th_len = kp;
            }
            if (finished || stopped) {
                {
                    std::lock_guard<std::recursive_mutex> lk(s.device_mu);
                    drop(g);
                }
                fill_resume(g, finished, false);
                hooks.on_done(g.job);
            }
        }
        if (kTiming) t_step_all += ms_since(t_step);
        if (kMinGain > 0) {
            // Ordinary decode packs the sessions into one step, sampled ones included (their
            // top_k is 1..64 here); SPARKINFER_PACKED_SAMPLING=0 runs each sampled one alone.
            static const bool packed_sampling = [] {
                const char* e = getenv("SPARKINFER_PACKED_SAMPLING");
                return !(e && e[0] == '0');
            }();
            bool any_sampled = false;
            for (G* g : act) any_sampled |= g->job->temperature > 0.f;
            const double plain = any_sampled && !packed_sampling ? S * plain_decode_ms(1)
                                                                 : plain_decode_ms(S);
            gain_won[gain_steps % kGainWindow] = (double)step_tokens * plain;
            gain_cost[gain_steps % kGainWindow] = (double)S * step_ms;
            double won = 0, cost = 0;
            for (int i = 0; i < kGainWindow; i++) { won += gain_won[i]; cost += gain_cost[i]; }
            const double gain_avg = won / std::max(1e-3, cost);
            if (++gain_steps >= kGainWindow && gain_avg < kMinGain) {
                if (kTiming)
                    fprintf(stderr, "[spec-group] speculation does not pay here (gain %.2f at S=%d, "
                                    "context %d, %.1f ms a step against %.1f ordinary): decoding "
                                    "ordinarily\n", gain_avg, S, ctx_max, step_ms, plain);
                break;
            }
        }
        // Newcomers join (or the group stops and hands everyone back).
        std::vector<SpecGroupJob*> joins;
        if (!hooks.poll(joins)) break;
        if (!joins.empty()) {
            int live = 0;
            for (G& g : gs) live += !g.done;
            for (SpecGroupJob* j : joins)
                if (!join(j, depth_for(live + (int)joins.size()))) {
                    j->resume = SpecResume{};
                    hooks.on_done(j);
                    group_ok = false;
                }
        }
    }
    // Hand back whoever is still running, at its committed position.
    {
        std::lock_guard<std::recursive_mutex> lk(s.device_mu);
        for (G& g : gs) {
            if (g.done) continue;
            drop(g);
            fill_resume(g, false, false);
            hooks.on_done(g.job);
        }
        draft.kv_state_select(-1);
        draft.reset();
        set_dflash_capture(false, {}, 0);
        invalidate_decode_graph();
    }
    spec_group_release();
    if (kTiming && steps > 0)
        fprintf(stderr, "[spec-group] steps=%ld sessions/step=%.2f mean_accept=%.3f | draft %.2f "
                        "ms/step | verify %.2f ms/step | masks %.2f | step %.2f\n", steps,
                (double)seg_steps / steps, accept_sum / seg_steps, t_draft / steps,
                t_verify / steps, t_mask / steps, t_step_all / steps);
    if (kTiming && kNgram && steps > 0)
        fprintf(stderr, "[spec-group] lookup: %ld of %ld session-steps matched; %ld deep steps "
                        "committed %.2f tokens each\n", ng_steps, seg_steps,
                ng_long_steps, ng_long_steps ? (double)ng_long_tokens / ng_long_steps : 0.0);
}

// ----- weight loading from a sparkinfer weight directory -----
namespace {
void* load_bin(const std::string& path, std::vector<void*>& owned) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[qwen35] missing weight: %s\n", path.c_str()); return nullptr; }
    std::streamsize n = f.tellg(); f.seekg(0);
    std::vector<char> host(n);
    f.read(host.data(), n);
    void* d = nullptr;
    if (cudaMalloc(&d, n) != cudaSuccess) return nullptr;
    cudaMemcpy(d, host.data(), n, cudaMemcpyHostToDevice);
    owned.push_back(d);
    return d;
}

// ---------------------------------------------------------------------------
// Per-device weight allocation (dual-GPU Wave 3, CHG-0011). The process-wide
// tensor-parallel placement table (tp_layout.hpp) is the SINGLE source of
// truth for which rank owns which slice of each named tensor (D2). ModelEngine::
// load builds and publishes it before any loader runs; at the tp=1 default it
// is the degenerate table (every name whole on device 0), so the Whole branch
// below is byte-identical to the unsplit loaders (the tp=1 invariance). Each of
// the three loaders (load_weights / load_gguf / load_compressed_tensors) calls
// tp_decide() before a read and branches:
//   Skip  -> nullptr, no alloc, no read (another rank owns the name);
//   Whole -> the loader's existing read path, untouched (tp=1 + replicated);
//   Slice -> a reduced on-disk allocation + a byte-range read of the owned
//            sub-blocks (tpl_read2d).
// ---------------------------------------------------------------------------
enum class TpDec { Skip, Whole, Slice };

struct TpSlice {
    tp::Axis axis = tp::Axis::None;
    size_t denom = 0;
    std::vector<tp::Range> ranges;
};

// The table (never a re-derived ratio): see the block comment above.
inline TpDec tp_decide(const std::string& name, int rank, TpSlice* out) {
    const tp::Table& t = tp::get_process_table();
    if (!t.set() || t.n_ranks() <= 1) return TpDec::Whole;   // tp=1 invariance
    const tp::Placement p = t.placement(name, rank);
    if (p.device < 0) return TpDec::Whole;                    // replicated (device -1)
    if (p.device != rank) return TpDec::Skip;                // another rank owns it
    if (p.axis == tp::Axis::None) return TpDec::Whole;       // whole on this rank
    if (out) {
        out->axis = p.axis;
        out->denom = p.denom;
        out->ranges = p.ranges;
    }
    return TpDec::Slice;
}

// The pure data movement: copies the elements this rank owns out of a FULL
// 2-D (or 1-D) tensor held in a host buffer into a contiguous buffer in the
// SAME physical layout as the source (the existing kernels read it unchanged),
// sized to the owned elements only -- the result's byte count equals the
// table's rank_on_disk_bytes for the tensor (the CPU test asserts the two
// agree).
//
//   d0/d1     the two logical dims in the table's orientation (d0 = Axis::Rows,
//             d1 = Axis::Cols); pass d1 = 1 for a 1-D tensor.
//   eb        bytes per element (bf16 = 2, f32 = 4; NVFP4 callers pass the
//             packed/scale streams as [rows, units] u8 matrices -- see the
//             compressed-tensors call sites).
//   rows_fast true  = the d0 axis is the in-memory-fast one (GGUF / flat, ggml
//             [in,out]); false = the d1 axis is fast (HF row-major [out,in]).
//   axis      the logical axis the split is on (Rows = d0, Cols = d1, OneD = the
//             single dim of a 1-D tensor).
//   ranges    the owned half-open element intervals along `axis` (one for a
//             plain split; up to three for the GDN qkv / conv1d channel axis,
//             which always lands on the SLOW axis below).
//   xform     optional per-bf16-element transform applied to the gathered
//             staging (the A_log / norm+1 paths), only when eb == 2.
//
// A split on the SLOW axis copies contiguous byte block(s); a split on the
// FAST axis copies one sub-block per value of the (full) slow axis. Either way
// the result tiles the source exactly as a valid smaller matrix in the source's
// own layout.
void tpl_gather2d(const char* src, long d0, long d1, int eb, bool rows_fast,
                 tp::Axis axis, const std::vector<tp::Range>& ranges,
                 const std::function<void(uint16_t*)>* xform,
                 std::vector<char>& out) {
    long fast_ext, slow_ext;
    bool slow_split;
    if (axis == tp::Axis::OneD) {
        // 1-D tensor: the single dim is the element axis. The ranges are element
        // intervals, so each "row" holds one element (fast_ext = 1) and the ranges
        // index element positions directly (slow-split: contiguous block(s)).
        fast_ext = 1; slow_ext = d0; slow_split = true;
    } else if (rows_fast) {
        fast_ext = d0; slow_ext = d1;
        slow_split = (axis == tp::Axis::Cols);
    } else {
        fast_ext = d1; slow_ext = d0;
        slow_split = (axis == tp::Axis::Rows);
    }
    if (slow_split) {
        long owned_slow = 0;
        for (const tp::Range& r : ranges) owned_slow += (long)r.len;
        if (owned_slow <= 0) { out.clear(); return; }
        out.assign((size_t)fast_ext * (size_t)owned_slow * (size_t)eb, 0);
        size_t off = 0;
        for (const tp::Range& r : ranges) {
            const size_t block = (size_t)fast_ext * r.len * (size_t)eb;
            const char* s = src + (size_t)r.begin * (size_t)fast_ext * (size_t)eb;
            memcpy(out.data() + off, s, block);
            off += block;
        }
    } else {
        if (ranges.size() != 1) {
            fprintf(stderr, "[tp-weights] fast-axis split expects one range, got %zu\n",
                    ranges.size());
            out.clear();
            return;
        }
        const size_t b = ranges[0].begin, len = ranges[0].len;
        if (len <= 0) { out.clear(); return; }
        const size_t rowb = len * (size_t)eb;
        const size_t stride = (size_t)fast_ext * (size_t)eb;
        out.assign((size_t)slow_ext * rowb, 0);
        for (long s = 0; s < slow_ext; ++s) {
            const char* src_b = src + (size_t)s * stride + b * (size_t)eb;
            memcpy(out.data() + (size_t)s * rowb, src_b, rowb);
        }
    }
    if (xform && eb == 2) {
        uint16_t* e = reinterpret_cast<uint16_t*>(out.data());
        for (size_t i = 0, n = out.size() / 2; i < n; ++i) (*xform)(e + i);
    }
}

// The device-upload twin of tpl_gather2d (H2D into an owned allocation).
void* tpl_read2d(const char* src, long d0, long d1, int eb, bool rows_fast,
                 tp::Axis axis, const std::vector<tp::Range>& ranges,
                 std::vector<void*>& owned,
                 const std::function<void(uint16_t*)>* xform = nullptr) {
    std::vector<char> stage;
    tpl_gather2d(src, d0, d1, eb, rows_fast, axis, ranges, xform, stage);
    if (stage.empty()) return nullptr;
    void* d = nullptr;
    if (cudaMalloc(&d, stage.size()) != cudaSuccess) return nullptr;
    cudaMemcpy(d, stage.data(), stage.size(), cudaMemcpyHostToDevice);
    owned.push_back(d);
    return d;
}

// The rank's owned extents of a tensor's two logical dims (d0 = the Rows axis,
// d1 = the Cols axis, in the table's orientation): the full extents when the
// table is degenerate (tp=1) or the name is whole/replicated, the reduced
// extents when sliced. For consumers that need the reduced geometry rather than
// just the pointer (the Q4_K row-scale precompute, which must run over the
// rows actually resident on this rank).
void tp_owned_dims(const std::string& name, int rank, long d0, long d1,
                   long& o0, long& o1) {
    o0 = d0; o1 = d1;
    TpSlice sl;
    if (tp_decide(name, rank, &sl) != TpDec::Slice) return;
    long owned = 0;
    for (const tp::Range& r : sl.ranges) owned += (long)r.len;
    if (sl.axis == tp::Axis::Rows || sl.axis == tp::Axis::OneD) o0 = owned;
    else if (sl.axis == tp::Axis::Cols) o1 = owned;
}

// The ggml block geometry of the dequant-supported quant types (block elems,
// block bytes); false for the dense types (F32/F16/BF16), which gather as
// plain element buffers (blk_elems = 1, blk_bytes = the element size).
bool gguf_block_units(int ggml_type, long& blk_elems, size_t& blk_bytes) {
    switch (ggml_type) {
        case 8:  blk_elems = 32;  blk_bytes = 34;  return true;   // Q8_0
        case 12: blk_elems = 256; blk_bytes = 144; return true;   // Q4_K
        case 13: blk_elems = 256; blk_bytes = 176; return true;   // Q5_K
        case 14: blk_elems = 256; blk_bytes = 210; return true;   // Q6_K
        default: return false;
    }
}

// One-time, per-(reason, name) stderr note for a table-sliced rank that loads
// the tensor whole anyway (PTQ1 re-pack, unaligned quant blocks): that rank
// then holds more on-disk bytes than the table's per-rank budget says. Printed,
// never silent (D-B).
void tp_gguf_slice_note(const char* why, const char* name) {
    static std::map<std::string, bool> done;
    const std::string k = std::string(why) + "|" + name;
    if (!done[k]) {
        done[k] = true;
        fprintf(stderr, "[tp-weights] %s: %s loaded whole on this rank (not sliceable)\n",
                why, name);
    }
}

// The GGUF-orientation twin of tpl_read2d for a tensor whose on-disk bytes are
// t->data (an mmap into the file: [dims[1]] rows of dims[0], the ggml FAST
// axis, row-major). The table's axis is a dim-INDEX axis: Rows = dims[0] =
// the fast axis, Cols = dims[1] = the slow axis, OneD = the single dim. The
// result is a contiguous allocation in the SOURCE'S OWN layout, sized to the
// owned units only: a valid reduced ggml tensor the dequant kernels read
// unchanged (quant types: whole blocks are the units, so the caller must pass
// block-aligned ranges; dense types: elements). nullptr when nothing is
// owned or the alloc fails.
void* tpl_gguf_read(const GGUFTensor* t, tp::Axis axis,
                    const std::vector<tp::Range>& ranges,
                    long blk_elems, size_t blk_bytes,
                    std::vector<void*>& owned) {
    const char* src = static_cast<const char*>(t->data);
    if (t->n_dims <= 1) {
        if (blk_elems != 1) return nullptr;   // 1-D quant is not tileable here
        std::vector<char> stage;
        tpl_gather2d(src, t->dims[0], 1, (int)blk_bytes, false, tp::Axis::OneD, ranges,
                     nullptr, stage);
        if (stage.empty()) return nullptr;
        void* d = nullptr;
        if (cudaMalloc(&d, stage.size()) != cudaSuccess) return nullptr;
        cudaMemcpy(d, stage.data(), stage.size(), cudaMemcpyHostToDevice);
        owned.push_back(d);
        return d;
    }
    if (axis == tp::Axis::OneD) return nullptr;
    std::vector<tp::Range> u = ranges;
    tp::Axis ax;
    if (axis == tp::Axis::Rows) {
        // dims[0] is the ggml FAST axis: the owned intervals are element runs
        // along it; remap to block units (block-aligned by the caller) and let
        // the fast-axis gather take one sub-block per row of the (full) slow
        // axis.
        for (auto& r : u) { r.begin /= blk_elems; r.len /= blk_elems; }
        ax = tp::Axis::Cols;
    } else {
        // dims[1] is the slow axis: contiguous row block(s), ranges as-is.
        ax = tp::Axis::Rows;
    }
    return tpl_read2d(src, t->dims[1], t->dims[0] / blk_elems, (int)blk_bytes, false, ax, u, owned);
}
}

bool Qwen35Model::load_weights(const std::string& dir) {
    Impl& s = *p_;
    // Per-rank allocation per the process table (D2). The flat .bin names
    // (layer_N.wq, embed_tokens, ...) are only known to a Flat-convention
    // table; the engine builds Gguf/Hf tables, under which every flat name
    // resolves replicated (Whole on every rank) -- still a valid, if
    // duplicated, placement. Under a Flat table (or a future engine change
    // building one) the same calls split: the reads below then carry only
    // this rank's sub-blocks. tp=1 (degenerate table) -> Whole everywhere ->
    // byte-identical to the pre-split loader (the tp=1 invariance).
    {
        const tp::Table& tt = tp::get_process_table();
        if (tt.set() && tt.n_ranks() > 1 && tt.conv() != tp::Conv::Flat)
            fprintf(stderr, "[tp-weights] flat .bin on rank %d of %d: names unknown to the "
                            "Gguf/Hf-convention table load whole (replicated); a "
                            "Flat-convention table would slice them (best-effort reads)\n",
                    s.rank, tt.n_ranks());
    }
    // d0/d1 are the table-orientation dims (d0 = the Rows axis, d1 = Cols); rows_fast says
    // which of them is the file's fast axis (tpl_gather2d). A Slice read is validated against
    // the file before any gather: raw bf16 of exactly d0*d1 elements, the split axis' extent
    // equal to the table's denom, and every range inside it -- a mismatch fails the load
    // loudly instead of reading past the host buffer (tp_bad).
    bool tp_bad = false;
    auto Lt = [&](const std::string& n, long d0, long d1, bool rows_fast = true) -> void* {
        TpSlice sl{};
        switch (tp_decide(n, s.rank, &sl)) {
            case TpDec::Whole:
                return load_bin(dir + "/" + n + ".bin", s.owned);
            case TpDec::Skip:
                return nullptr;
            case TpDec::Slice: {
                // Reduced read: the flat file is raw bf16; read it whole to host and
                // gather the owned sub-blocks (1-D tensors: d1 = 1, axis OneD).
                std::ifstream f(dir + "/" + n + ".bin", std::ios::binary | std::ios::ate);
                if (!f) {
                    fprintf(stderr, "[qwen35] missing weight: %s/%s.bin\n", dir.c_str(), n.c_str());
                    tp_bad = true;
                    return nullptr;
                }
                std::vector<char> host((size_t)f.tellg());
                f.seekg(0);
                f.read(host.data(), host.size());
                const long ext = sl.axis == tp::Axis::Cols ? d1 : d0;
                bool ok = host.size() == (size_t)d0 * (size_t)d1 * 2 && (size_t)ext == sl.denom;
                size_t owned_len = 0;
                for (const tp::Range& r : sl.ranges) {
                    ok = ok && r.begin + r.len <= (size_t)ext;
                    owned_len += r.len;
                }
                if (!ok) {
                    fprintf(stderr, "[tp-weights] %s: file %zu B / split extent %ld do not match "
                            "the table (shape %ldx%ld bf16, denom %zu) -- refusing the slice\n",
                            n.c_str(), host.size(), ext, d0, d1, sl.denom);
                    tp_bad = true;
                    return nullptr;
                }
                if (owned_len == 0) return nullptr;   // an empty window (e.g. 1 KV head, 2 ranks)
                void* p = tpl_read2d(host.data(), d0, d1, 2, rows_fast, sl.axis, sl.ranges, s.owned);
                if (!p) tp_bad = true;
                return p;
            }
        }
        return nullptr;
    };
    auto miss = [&](const std::string& n, const void* p) {
        return !p && tp_decide(n, s.rank, nullptr) != TpDec::Skip;
    };
    const long H = s.cfg.hidden, V = s.cfg.vocab;
    // embed/lm_head are [V][H] row-major on disk (H fast); the table splits them on the vocab
    // axis (Rows, denom V), so in table orientation d0 = V is the SLOW axis: each rank gets its
    // contiguous block of vocab rows -- the layout the vocab-split forward indexes.
    s.w.embed_tokens = Lt("embed_tokens", V, H, false);
    s.w.final_norm   = Lt("final_norm", H, 1);
    s.w.lm_head      = Lt("lm_head", V, H, false);
    if (miss("embed_tokens", s.w.embed_tokens) ||
        miss("final_norm", s.w.final_norm) ||
        miss("lm_head", s.w.lm_head)) {
        fprintf(stderr, "[compressed-tensors] top-level weights missing (embed=%d final_norm=%d "
                "lm_head=%d)\n", s.w.embed_tokens != nullptr, s.w.final_norm != nullptr,
                s.w.lm_head != nullptr);
        return false;
    }
    s.w.layers.resize(s.cfg.n_layers);
    for (int i = 0; i < s.cfg.n_layers; i++) {
        std::string pfx = "layer_" + std::to_string(i) + ".";
        Qwen35LayerWeights& w = s.w.layers[i];
        const long qout = (long)s.cfg.n_q_heads * s.cfg.head_dim;
        const long kvout = (long)s.cfg.n_kv_heads * s.cfg.head_dim;
        const long F = s.cfg.moe_ffn;
        w.input_norm     = Lt(pfx + "input_norm", H, 1);
        w.wq = Lt(pfx + "wq", H, qout); w.wk = Lt(pfx + "wk", H, kvout);
        w.wv = Lt(pfx + "wv", H, kvout); w.wo = Lt(pfx + "wo", qout, H);
        w.q_norm = Lt(pfx + "q_norm", s.cfg.head_dim, 1);
        w.k_norm = Lt(pfx + "k_norm", s.cfg.head_dim, 1);
        w.post_attn_norm = Lt(pfx + "post_attn_norm", H, 1);
        w.router_w = Lt(pfx + "router_w", H, s.cfg.n_experts);
        w.gate = Lt(pfx + "gate", H, F); w.up = Lt(pfx + "up", H, F); w.down = Lt(pfx + "down", F, H);
        if (s.cfg.n_shared > 0) {
            w.shared_gate = Lt(pfx + "shared_gate", H, F);
            w.shared_up = Lt(pfx + "shared_up", H, F);
            w.shared_down = Lt(pfx + "shared_down", F, H);
        }
        if (miss(pfx + "wq", w.wq) || miss(pfx + "gate", w.gate) ||
            miss(pfx + "router_w", w.router_w))
            return false;
    }
    return !tp_bad;
}

// ----- native GGUF load: dense -> bf16 (dequant + transpose), experts kept quantized -----
bool Qwen35Model::load_gguf(const std::string& path) {
    Impl& s = *p_;
    GGUF g;
    if (!g.open(path)) return false;
    // Ternary-Bonsai-2 declares a Hadamard rotation over its weights. Read it before any tensor is
    // uploaded: a rotated weight loaded as though it were not is not degraded, it is noise.
    PrismHadamard had;
    {
        std::string had_err;
        if (!had.load(g, had_err)) {
            fprintf(stderr, "[bonsai] prism.hadamard metadata is unusable: %s\n", had_err.c_str());
            return false;
        }
    }
    // Native PTQ1_0: keep the weights in their stored blocks and rotate the activation instead.
    // Off by default -- the folded path is the one measured at PPL 8.07 -- because this trades a
    // validated path for a much smaller one, and both arms should come out of the same binary.
    // SPARKINFER_BONSAI_NATIVE selects which tensors stay in their stored blocks: "1"/"all", or a
    // list of "head" and "embed". Selectable so a fault can be attributed to one of them rather
    // than to "the native path".
    static const std::string bonsai_native_set = [] {
        const char* e = getenv("SPARKINFER_BONSAI_NATIVE");
        std::string v(e ? e : "");
        // "qkv" and "gate" narrow "proj" to one tensor family, which is how the v-head
        // regrouping bug in the native upload was attributed: attn_gate is entirely v heads and
        // came out far more wrong than attn_qkv, where v is a third of the rows.
        if (v == "1" || v == "all") v = "head,embed,proj,ffn";
        return v;
    }();
    const bool bonsai_native = had.present && !bonsai_native_set.empty();
    const bool bonsai_native_head = bonsai_native_set.find("head") != std::string::npos;
    const bool bonsai_native_embed = bonsai_native_set.find("embed") != std::string::npos;
    // "proj" takes every residual-width projection; "qkv" and "gate" take one family each, to
    // attribute the projection fault to a tensor rather than to the slice.
    const bool bonsai_proj_qkv_only = bonsai_native_set.find("qkv") != std::string::npos;
    const bool bonsai_proj_gate_only = bonsai_native_set.find("gate") != std::string::npos;
    const bool bonsai_native_proj = bonsai_native_set.find("proj") != std::string::npos ||
                                    bonsai_proj_qkv_only || bonsai_proj_gate_only;
    // "ffn" is separate from "proj" because it is the only family whose DOWN leg reads a
    // non-residual width, and because it replaces a single fused Q4_K kernel with three GEMVs and
    // an elementwise SwiGLU -- a different performance question from the projections.
    const bool bonsai_native_ffn = bonsai_native_set.find("ffn") != std::string::npos;
    // The decode shadow: main's folded Q4_K load, unchanged, plus a ternary copy of the head, the
    // residual-width projections and the FFN that only forward_token reads. Prefill keeps its
    // Q4_K GEMMs and the packed batch its Q4_K kernels; single-row decode reads 2.6x fewer bytes.
    // Off whenever SPARKINFER_BONSAI_NATIVE picks a residency explicitly.
    static const bool bonsai_shadow_env = [] {
        const char* e = getenv("SPARKINFER_BONSAI_DECODE_SHADOW");
        return !(e && e[0] == '0');
    }();
    const bool bonsai_shadow = had.present && bonsai_native_set.empty() && bonsai_shadow_env;
    // Which parts get a ternary copy: "head", "proj", "ffn", comma-separated. Default: the FFN,
    // ~70% of the weight bytes. The head and the projections add speed but each moves the
    // logits further from the folded path's (KL vs main 0.018 FFN-only, 0.026 all three).
    static const std::string bonsai_shadow_parts = [] {
        const char* e = getenv("SPARKINFER_BONSAI_SHADOW_PARTS");
        return std::string(e ? e : "ffn");
    }();
    const bool shadow_head = bonsai_shadow && bonsai_shadow_parts.find("head") != std::string::npos;
    const bool shadow_proj = bonsai_shadow && bonsai_shadow_parts.find("proj") != std::string::npos;
    const bool shadow_ffn = bonsai_shadow && bonsai_shadow_parts.find("ffn") != std::string::npos;
    std::unordered_map<const void*, void*> shadow_of;   // folded weight -> its ternary copy
    if (bonsai_native || bonsai_shadow) {
        s.bonsai_block = had.block_size;
        for (const auto& kv : had.signs_by_width) {
            void* d = nullptr;
            if (cudaMalloc(&d, kv.second.size()) != cudaSuccess) continue;
            cudaMemcpy(d, kv.second.data(), kv.second.size(), cudaMemcpyHostToDevice);
            s.bonsai_sign_dev[kv.first] = d;
            s.owned.push_back(d);
            s.bonsai_rot_elems = std::max(s.bonsai_rot_elems, kv.first);
        }
        if (s.bonsai_rot_elems > 0 &&
            cudaMalloc((void**)&s.bonsai_rot, (size_t)s.bonsai_rot_elems * sizeof(bf16)) == cudaSuccess)
            s.owned.push_back(s.bonsai_rot);
        else
            s.bonsai_rot = nullptr;
        if ((bonsai_native_proj || bonsai_shadow) && s.cfg.hidden > 0 &&
            cudaMalloc((void**)&s.bonsai_rot_xn, (size_t)s.cfg.hidden * sizeof(bf16)) == cudaSuccess)
            s.owned.push_back(s.bonsai_rot_xn);
        else
            s.bonsai_rot_xn = nullptr;
        {
            const auto sh = s.bonsai_sign_dev.find(s.cfg.hidden);
            if (sh != s.bonsai_sign_dev.end())
                s.bonsai_sign_h = static_cast<const signed char*>(sh->second);
            const auto sf = s.bonsai_sign_dev.find(s.cfg.moe_ffn);
            if (sf != s.bonsai_sign_dev.end())
                s.bonsai_sign_ffn = static_cast<const signed char*>(sf->second);
        }
        // All four or none: a half-allocated FFN scratch would leave the decode branch reading a
        // null buffer, and the folded path is a perfectly good fallback.
        if ((bonsai_native_ffn || bonsai_shadow) && s.cfg.hidden > 0 && s.cfg.moe_ffn > 0 &&
            s.bonsai_sign_h && s.bonsai_sign_ffn) {
            const size_t hb = (size_t)s.cfg.hidden * sizeof(bf16);
            const size_t fb = (size_t)s.cfg.moe_ffn * sizeof(bf16);
            if (cudaMalloc((void**)&s.bonsai_rot_hn, hb) == cudaSuccess &&
                cudaMalloc((void**)&s.bonsai_ffn_gate, fb) == cudaSuccess &&
                cudaMalloc((void**)&s.bonsai_ffn_up, fb) == cudaSuccess &&
                cudaMalloc((void**)&s.bonsai_ffn_h, fb) == cudaSuccess) {
                s.owned.push_back(s.bonsai_rot_hn);
                s.owned.push_back(s.bonsai_ffn_gate);
                s.owned.push_back(s.bonsai_ffn_up);
                s.owned.push_back(s.bonsai_ffn_h);
            } else {
                cudaFree(s.bonsai_rot_hn); cudaFree(s.bonsai_ffn_gate);
                cudaFree(s.bonsai_ffn_up); cudaFree(s.bonsai_ffn_h);
                s.bonsai_rot_hn = s.bonsai_ffn_gate = s.bonsai_ffn_up = s.bonsai_ffn_h = nullptr;
                fprintf(stderr, "[bonsai] FFN scratch alloc failed -- FFN stays folded\n");
            }
        }
    }

    // Shared by both upload paths: is this tensor one the checkpoint rotated, and which signs?
    auto rotated_signs = [&](const GGUFTensor* t, const std::string& name)
            -> const std::vector<int8_t>* {
        if (!t || t->ggml_type != kPtq1GgmlType || !had.present) return nullptr;
        if (!had.rotates(name) && had.inverse_rotated.count(name) == 0) return nullptr;
        const std::vector<int8_t>* sign = had.signs_for(t->dims[0]);
        if (!sign)
            fprintf(stderr, "[bonsai] %s: no sign vector for input width %ld\n",
                    name.c_str(), t->dims[0]);
        return sign;
    };
    const bool dense_file = g.tensor("blk.0.ffn_gate.weight") != nullptr &&
                            g.tensor("blk.0.ffn_gate_exps.weight") == nullptr;
    const bool hybrid_file = is_qwen35_or_qwen36_hybrid_moe(g) || dense_file;
    if (hybrid_file && !s.cfg.hybrid && !s.cfg.dense_ffn) {
        fprintf(stderr,
                "[qwen35] Qwen3.5/Qwen3.6 hybrid GGUF requires constructing "
                "Qwen35Model with cfg.hybrid=true and the GGUF metadata-derived "
                "head dimensions before load_gguf(), so scratch buffers and KV "
                "cache are sized correctly.\n");
        return false;
    }
    // Qwythos/Qwen3.6 hybrid-file backfill: assumes any dense-FFN or hybrid-detected GGUF
    // wants the Gated-DeltaNet SSM interleave (full_attn_interval defaulted to 4). Muse
    // Glimmer is also a dense-FFN GGUF (dense_file=true) but has NO linear/SSM layers at
    // all -- museglimmer_config_from_gguf already set every one of these fields correctly
    // (full_attn_interval=0 deliberately, to keep is_linear_layer() false for every layer),
    // so skip this backfill for it rather than let full_attn_interval<=0 get overwritten to
    // 4 and misroute layer 0 into looking for attn_qkv.weight/ssm_* tensors that don't exist.
    if ((hybrid_file || s.cfg.dense_ffn) && !s.cfg.muse_glimmer) {
        s.cfg.hybrid = true;
        if (dense_file) s.cfg.dense_ffn = true;
        if (s.cfg.full_attn_interval <= 0) s.cfg.full_attn_interval = 4;
        if (s.cfg.rope_dim <= 0 && s.cfg.head_dim == 256) s.cfg.rope_dim = 64;
        if (s.cfg.linear_q_heads <= 0) s.cfg.linear_q_heads = 16;
        if (s.cfg.linear_v_heads <= 0) s.cfg.linear_v_heads = 32;
        if (s.cfg.linear_head_dim <= 0) s.cfg.linear_head_dim = 128;
        if (s.cfg.linear_conv_kernel <= 0) s.cfg.linear_conv_kernel = 4;
    }
    const bool dense_ffn = g.tensor("blk.0.ffn_gate.weight") != nullptr &&
                           g.tensor("blk.0.ffn_gate_exps.weight") == nullptr;
    if (dense_ffn) {
        s.cfg.dense_ffn = true;
        s.cfg.n_experts = 1;
        s.cfg.top_k = 1;
        s.cfg.n_shared = 0;
        if (s.cfg.moe_ffn <= 0) {
            s.cfg.moe_ffn = (int)qwen_moe_meta_int(g, "feed_forward_length", 0);
            if (s.cfg.moe_ffn <= 0) {
                if (const GGUFTensor* gate = g.tensor("blk.0.ffn_gate.weight"))
                    if (gate->n_dims >= 2) s.cfg.moe_ffn = (int)gate->dims[1];
            }
        }
    }
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    s.gguf = true;   // dense weights kept native [out,in]; forward uses GEMV

    // Shared-expert tensors are optional in GGUF (Qwen3-30B-A3B has none). The
    // default config sets n_shared=1, so clamp it to what the file actually
    // contains before forward_token can launch a null-weight FFN.
    const bool gguf_has_shared =
        g.tensor("blk.0.ffn_gate_shexp.weight") != nullptr;
    if (hybrid_file && gguf_has_shared && s.cfg.n_shared == 0) s.cfg.n_shared = 1;
    if (c.n_shared > 0 && !gguf_has_shared) {
        fprintf(stderr,
                "[gguf] no shared-expert tensors; forcing n_shared=0 "
                "(safe for models without a shared FFN)\n");
        s.cfg.n_shared = 0;
    }
    // Shared-expert GEMV scratch [moe_ffn]. Allocated only on the GGUF path (native
    // [out,in] shared weights); the set_weights path keeps the moe_expert_ffn kernel.
    if (s.cfg.n_shared > 0 && !s.sh_gate) {
        s.sh_gate = s.alloc<bf16>(s.cfg.moe_ffn);
        s.sh_up   = s.alloc<bf16>(s.cfg.moe_ffn);
        s.sh_h    = s.alloc<bf16>(s.cfg.moe_ffn);
    }
    // upload raw quantized blocks, keep on device (for experts)
    auto dev_quant = [&](const std::string& name, int& qtype) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gguf] missing %s\n", name.c_str()); return nullptr; }
        // TP (D2/D-B): the table decides this rank's read before anything is
        // allocated. Skipped: no alloc, no read (another rank owns the name).
        // Sliced: the on-disk bytes are the ggml blocks themselves, so the
        // owned sub-blocks upload as a valid reduced tensor of the same type
        // (block-aligned ranges; the dequant kernels run over the slice).
        // PTQ1 re-packs whole tensors, so a PTQ1 slice falls through to the
        // whole load with a printed note (the rank then holds more than the
        // table's per-rank budget says -- the per-rank audit prints it).
        {
            TpSlice sl;
            const TpDec dec = tp_decide(name, s.rank, &sl);
            if (dec == TpDec::Skip) { qtype = 0; return nullptr; }
            if (dec == TpDec::Slice) {
                long be; size_t bb;
                if (t->ggml_type == kPtq1GgmlType) {
                    tp_gguf_slice_note("PTQ1_0 re-pack", name.c_str());
                } else if (gguf_block_units(t->ggml_type, be, bb) && t->dims[0] % be == 0) {
                    void* p = tpl_gguf_read(t, sl.axis, sl.ranges, be, bb, s.owned);
                    if (p) { qtype = t->ggml_type; return p; }
                    tp_gguf_slice_note("unaligned quant block", name.c_str());
                } else {
                    tp_gguf_slice_note("dense type on the quant path", name.c_str());
                }
                // fall through: whole load (the tp=1 path stays untouched below)
            }
        }
        if (const std::vector<int8_t>* sign = rotated_signs(t, name)) {
            UnrotateJob j;
            if (!unrotate_job_init(j, t, name, *sign, had.block_size,
                                   had.inverse_rotated.count(name) != 0)) return nullptr;
            if (had.gdn_v_grouped) unrotate_job_set_v_block(j, name, s.cfg);
            void* d = unrotate_ternary_to_q4k(j, name, s.stream);
            if (!d) return nullptr;   // never fall back to the rotated bytes: they are not weights
            qtype = 12;               // Q4_K
            s.owned.push_back(d);
            return d;
        }
        const HostBlocks hb = host_blocks_for_upload(t, name);
        if (!ggml_dequant_supported(hb.ggml_type)) {
            fprintf(stderr, "[gguf] unsupported ggml type %d for %s\n", t->ggml_type, name.c_str());
            return nullptr;
        }
        qtype = hb.ggml_type;
        void* d = nullptr;
        if (cudaMalloc(&d, hb.bytes) != cudaSuccess) return nullptr;
        cudaMemcpy(d, hb.data, hb.bytes, cudaMemcpyHostToDevice);
        s.owned.push_back(d);
        return d;
    };
    // Optional Q6_K -> Q4_K requant: pay a load-time dequant+fit so decode reads
    // 4.5 instead of 6.5 bits/weight. The source Q6_K upload is freed after the
    // requant; qtype flips to 12 on success. Attention tensors use the Lloyd-max fit
    // (PR #353); FFN down keeps the affine fitter.
    auto is_attn_requant_name = [](const std::string& name) {
        return name.find(".attn_qkv.weight") != std::string::npos ||
               name.find(".attn_q.weight") != std::string::npos ||
               name.find(".attn_k.weight") != std::string::npos ||
               name.find(".attn_v.weight") != std::string::npos ||
               name.find(".attn_output.weight") != std::string::npos;
    };
    // allow_q5k is opt-in per call site, NOT a widening of the default set: dev_quant_down() also
    // routes through here with requant on by default, so accepting Q5_K unconditionally would
    // silently requantize the dense-FFN down tensor of any model that ships one.
    auto dev_quant_requant_q4k = [&](const std::string& name, int& qtype, bool req,
                                     bool allow_q5k = false) -> const void* {
        const void* q6 = dev_quant(name, qtype);
        const bool src_ok = (qtype == 14 || qtype == 8 || (allow_q5k && qtype == 13));
        if (!req || !src_ok || !q6) return q6;
        const int src_type = qtype;            // 14 (Q6_K), 8 (Q8_0) or 13 (Q5_K) -> Q4_K
        const GGUFTensor* t = g.tensor(name);
        long o0 = 0, o1 = 0;
        tp_owned_dims(name, s.rank, t->dims[0], t->n_dims >= 2 ? t->dims[1] : 1, o0, o1);
        const long nv = o0 * o1;               // owned count: a sliced rank requants the slice
        if (nv % 256 != 0) return q6;
        void* deq = nullptr;
        if (cudaMalloc(&deq, (size_t)nv * 2) != cudaSuccess) return q6;
        kernels::launch_gguf_dequant(src_type, q6, deq, nv, s.stream);
        void* q4 = nullptr;
        if (cudaMalloc(&q4, (size_t)(nv / 256) * 144) != cudaSuccess) { cudaFree(deq); return q6; }
        static int attn_lloyd = -1;
        if (attn_lloyd < 0) {
            const char* e = getenv("SPARKINFER_ATTN_REQUANT_LLOYD");
            attn_lloyd = (e && e[0] == '0') ? 0 : 1;
        }
        if (src_type == 8)
            kernels::launch_proj_requant_q4k_lloyd(deq, q4, nv, s.stream);
        else if (is_attn_requant_name(name) && attn_lloyd)
            kernels::launch_proj_requant_q4k_lloyd(deq, q4, nv, s.stream);
        else
            kernels::launch_ffn_down_requant_q4k(deq, q4, nv, s.stream);
        cudaStreamSynchronize(s.stream);
        cudaFree(deq);
        if (!s.owned.empty() && s.owned.back() == q6) { s.owned.pop_back(); cudaFree((void*)q6); }
        s.owned.push_back(q4);
        qtype = 12;
        return q4;
    };
    // Dense-FFN down: Q6_K in GGUF is requantized to Q4_K at load by default (~5% decode on
    // Qwythos). Set SPARKINFER_DOWN_REQUANT_Q4K=0 to keep native Q6_K reads.
    auto dev_quant_down = [&](const std::string& name, int& qtype) -> const void* {
        static int req = -1;
        if (req < 0) { const char* e = getenv("SPARKINFER_DOWN_REQUANT_Q4K"); req = (e && e[0] == '0') ? 0 : 1; }
        return dev_quant_requant_q4k(name, qtype, req != 0);
    };
    // Requantize a weight matrix to Q3_A (3.5 bits/weight -- Q4_K's asymmetric per-32 scale and
    // min, 3-bit quant plane, 112 B per 256 weights) at load, then decode it on-read through
    // si_vec_dot_q3_A, so the matrix is read 22.2% smaller on every decode token. Sources
    // Q4_K/Q5_K/Q6_K straight to Q3_A (one dequant, one fit). Falls back to the untouched source
    // on any failure.
    auto dev_quant_q3a = [&](const std::string& name, int& qtype,
                                 const void** prefill_q, int* prefill_qtype) -> const void* {
        const void* src = dev_quant(name, qtype);
        if (!src) return src;
        if (qtype != 12 && qtype != 13 && qtype != 14) return src;   // Q4_K / Q5_K / Q6_K
        const GGUFTensor* t = g.tensor(name);
        long o0 = 0, o1 = 0;
        tp_owned_dims(name, s.rank, t->dims[0], t->n_dims >= 2 ? t->dims[1] : 1, o0, o1);
        const long nv = o0 * o1;               // owned count: a sliced rank requants the slice
        if (nv % 256 != 0) return src;
        void* deq = nullptr;
        if (cudaMalloc(&deq, (size_t)nv * 2) != cudaSuccess) return src;
        kernels::launch_gguf_dequant(qtype, src, deq, nv, s.stream);
        void* q3 = nullptr;
        if (cudaMalloc(&q3, (size_t)(nv / 256) * 112) != cudaSuccess) { cudaFree(deq); return src; }
        kernels::launch_ffn_requant_q3a(deq, q3, nv, s.stream);
        cudaStreamSynchronize(s.stream);
        cudaFree(deq);
        // Keep the native source resident for batched prefill. Its established Q4_K/Q5_K/Q6_K
        // dequantizer is faster there; decode reads only the compact Q3_A copy.
        if (prefill_q) *prefill_q = src;
        if (prefill_qtype) *prefill_qtype = qtype;
        s.owned.push_back(q3);
        qtype = kernels::SI_QTYPE_Q3A;
        return q3;
    };
    // dense weight -> bf16 (optionally transpose [out,in] -> [in,out])
    auto dense = [&](const std::string& name, bool transpose) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gguf] missing %s\n", name.c_str()); return nullptr; }
        // TP (D2): a skipped rank reads nothing; a sliced rank reads the owned
        // sub-block (dense types gather in element units) and dequants only it
        // to bf16 (the transpose, when requested, uses the reduced extents).
        // The whole-load path below stays untouched (tp=1 invariance).
        {
            TpSlice sl;
            const TpDec dec = tp_decide(name, s.rank, &sl);
            if (dec == TpDec::Skip) return nullptr;
            if (dec == TpDec::Slice) {
                const int eb = (t->ggml_type == 0) ? 4 : 2;   // F32 / F16 / BF16
                void* dq = tpl_gguf_read(t, sl.axis, sl.ranges, 1, (size_t)eb, s.owned);
                if (dq) {
                    long o0 = 0, o1 = 0;
                    tp_owned_dims(name, s.rank, t->dims[0], t->n_dims >= 2 ? t->dims[1] : 1,
                                  o0, o1);
                    const long nv = o0 * o1;
                    void* tmp = nullptr;
                    if (cudaMalloc(&tmp, (size_t)nv * 2) != cudaSuccess) {
                        if (!s.owned.empty() && s.owned.back() == dq) s.owned.pop_back();
                        cudaFree((void*)dq);
                        return nullptr;
                    }
                    kernels::launch_gguf_dequant(t->ggml_type, dq, tmp, nv, s.stream);
                    if (transpose) {
                        const int in = (int)o0, out = (int)o1;
                        void* dst = nullptr;
                        if (cudaMalloc(&dst, (size_t)nv * 2) != cudaSuccess) {
                            if (!s.owned.empty() && s.owned.back() == dq) s.owned.pop_back();
                            cudaFree((void*)dq);
                            cudaFree(tmp);
                            return nullptr;
                        }
                        if (!s.owned.empty() && s.owned.back() == dq) s.owned.pop_back();
                        s.owned.push_back(dst);
                        cudaFree((void*)dq);
                        cudaFree(tmp);
                        kernels::launch_transpose_bf16(tmp, dst, out, in, s.stream);
                        cudaStreamSynchronize(s.stream);
                        return dst;
                    }
                    if (!s.owned.empty() && s.owned.back() == dq) s.owned.pop_back();
                    cudaFree((void*)dq);
                    s.owned.push_back(tmp);
                    cudaStreamSynchronize(s.stream);
                    return tmp;
                }
                // The gather produced nothing (an alloc failed): this rank is
                // not whole-owning the name, so a miss is the honest answer.
                return nullptr;
            }
        }
        if (const std::vector<int8_t>* sign = rotated_signs(t, name)) {
            // The embedding table: un-rotated straight to bf16, no requantization, because a
            // lookup reads rows rather than multiplying by them.
            UnrotateJob j;
            if (!unrotate_job_init(j, t, name, *sign, had.block_size,
                                   had.inverse_rotated.count(name) != 0)) return nullptr;
            if (had.gdn_v_grouped) unrotate_job_set_v_block(j, name, s.cfg);
            void* d = unrotate_ternary_to_bf16(j, name);
            if (!d) return nullptr;
            if (transpose) {
                fprintf(stderr, "[bonsai] %s: transpose of an un-rotated tensor is not wired\n",
                        name.c_str());
                cudaFree(d);
                return nullptr;
            }
            s.owned.push_back(d);
            return d;
        }
        const HostBlocks hb = host_blocks_for_upload(t, name);
        if (!ggml_dequant_supported(hb.ggml_type)) {
            fprintf(stderr, "[gguf] unsupported ggml type %d for %s\n", t->ggml_type, name.c_str());
            return nullptr;
        }
        void* dq = nullptr; cudaMalloc(&dq, hb.bytes);
        cudaMemcpy(dq, hb.data, hb.bytes, cudaMemcpyHostToDevice);
        void* tmp = nullptr; cudaMalloc(&tmp, (size_t)t->n_values * 2);
        if (!dq || !tmp) {
            // Say so. Returning a silent nullptr here surfaces as whichever tensor the caller
            // falls back to being reported "missing", which is a long way from the truth --
            // especially since a sticky CUDA error from an earlier tensor lands right here.
            fprintf(stderr, "[gguf] %s: device alloc failed (%s)\n",
                    name.c_str(), cudaGetErrorString(cudaGetLastError()));
            cudaFree(dq); cudaFree(tmp);
            return nullptr;
        }
        kernels::launch_gguf_dequant(hb.ggml_type, dq, tmp, t->n_values, s.stream);
        const void* result;
        if (transpose) {
            const int in = (int)t->dims[0], out = (int)t->dims[1];   // ggml ne0=in, ne1=out
            void* dst = nullptr; cudaMalloc(&dst, (size_t)t->n_values * 2); s.owned.push_back(dst);
            kernels::launch_transpose_bf16(tmp, dst, out, in, s.stream);   // [out,in]->[in,out]
            cudaStreamSynchronize(s.stream); cudaFree(tmp); cudaFree(dq);
            result = dst;
        } else {
            s.owned.push_back(tmp);
            cudaStreamSynchronize(s.stream); cudaFree(dq);
            result = tmp;
        }
        return result;
    };

    // Keep attention/lm_head weights quantized in VRAM and decode them on-read
    // (Q4_K -> int8 dp4a, Q6_K -> fp32 dequant) instead of expanding to bf16 at load.
    // Default ON: it feeds the dp4a GEMV path (~27% faster decode, gate-passing) and
    // uses ~1.5 GB less VRAM. Set SPARKINFER_QATTN=0 to load dense bf16 instead.
    const bool qattn = []{ const char* a = getenv("SPARKINFER_QATTN");
                           return !(a && a[0] == '0'); }();
    auto mode_is_off = [](const std::string& v) {
        return v.empty() || v == "0" || v == "false" || v == "FALSE" ||
               v == "off" || v == "OFF" || v == "no" || v == "NO";
    };
    const bool q35_dense9b_requant_default =
        c.dense_ffn && c.n_layers == 32 && H == 4096 && c.moe_ffn == 12288 &&
        c.top_k == 1 && c.full_attn_interval == 4 && []{
            const char* e = getenv("SPARKINFER_DOWN_REQUANT_Q4K");
            return !(e && e[0] == '0');
        }();
    auto env_enabled = [&](const char* name, bool def) {
        const char* v = getenv(name);
        return v ? !mode_is_off(std::string(v)) : def;
    };
    // Qwen3.6-35B-A3B UD ships its full-attention q/o projections as Q8_0. Requantize
    // them to Q4_K at load (Lloyd fit) so decode reads ~47% fewer bytes on those matvecs
    // (~+3.3% decode at short context, gate-passing). On by default for the Qwen3.6
    // fingerprint; a no-op on the dense Qwythos path (which uses its own qkv default).
    const bool q36_ud_requant_default = is_qwen35_or_qwen36_hybrid_moe(g);
    const char* attn_env = getenv("SPARKINFER_ATTN_REQUANT_Q4K");
    const std::string attn_requant_mode =
        attn_env ? std::string(attn_env)
                 : (q35_dense9b_requant_default ? std::string("qkv,v")
                    : (q36_ud_requant_default ? std::string("attn_q,attn_output,qkv,attn_gate,ssm_out")
                                              : std::string()));
    // Muse Glimmer dense FFN gate/up -> Q3_A on the 42 layers selected by calibration.
    // The ten sensitive layers remain native Q4_K; this passes the production distribution gate
    // (top1 >= 0.90 and KL <= 0.10 against llama.cpp) while reducing decode weight traffic.
    // Architecture-scoped because no other model served by this shared loader was calibrated.
    // SPARKINFER_MUSE_FFN_Q3A=0 restores the Q4_K load path exactly, so both arms of an A/B come
    // out of one binary.
    const char* ffn_q3a_layers_env = getenv("SPARKINFER_MUSE_FFN_Q3A_LAYERS");
    const std::string ffn_q3a_layers = ffn_q3a_layers_env
        ? std::string(ffn_q3a_layers_env)
        : (c.muse_glimmer ? std::string("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,33,34,39,40,41,42,43,44,45,46,47,48,49,50,51") : std::string());
    const char* ffn_q3a_env = getenv("SPARKINFER_MUSE_FFN_Q3A");
    const std::string ffn_q3a_mode =
        ffn_q3a_env ? std::string(ffn_q3a_env)
                    : (c.muse_glimmer ? std::string("ffn_gate,ffn_up") : std::string());
    auto list_token = [](const std::string& list, const char* want) {
        const std::string w(want);
        size_t p = 0;
        while (p < list.size()) {
            while (p < list.size() && (list[p] == ',' || list[p] == '+' || list[p] == ':' || list[p] == ' ')) ++p;
            size_t e = p;
            while (e < list.size() && list[e] != ',' && list[e] != '+' && list[e] != ':' && list[e] != ' ') ++e;
            if (e > p && list.compare(p, e - p, w) == 0) return true;
            p = e + 1;
        }
        return false;
    };
    auto ffn_q3a_on = [&](const char* which) {
        if (mode_is_off(ffn_q3a_mode)) return false;
        if (ffn_q3a_mode == "1" || list_token(ffn_q3a_mode, "all")) return true;
        return list_token(ffn_q3a_mode, which);
    };
    auto mode_token = [&](const char* want) {
        const std::string w(want);
        size_t p = 0;
        while (p < attn_requant_mode.size()) {
            while (p < attn_requant_mode.size() &&
                   (attn_requant_mode[p] == ',' || attn_requant_mode[p] == '+' ||
                    attn_requant_mode[p] == ':' || attn_requant_mode[p] == ' '))
                ++p;
            size_t e = p;
            while (e < attn_requant_mode.size() &&
                   attn_requant_mode[e] != ',' && attn_requant_mode[e] != '+' &&
                   attn_requant_mode[e] != ':' && attn_requant_mode[e] != ' ')
                ++e;
            if (e > p && attn_requant_mode.compare(p, e - p, w) == 0) return true;
            p = e + 1;
        }
        return false;
    };
    auto has_suffix = [](const std::string& s, const char* suffix) {
        const std::string t(suffix);
        return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
    };
    auto layer_index = [](const std::string& name) {
        if (name.compare(0, 4, "blk.") != 0) return -1;
        int layer = 0;
        size_t p = 4;
        if (p >= name.size() || name[p] < '0' || name[p] > '9') return -1;
        while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
            layer = layer * 10 + (name[p] - '0');
            ++p;
        }
        return (p < name.size() && name[p] == '.') ? layer : -1;
    };
    auto int_list_has = [](const std::string& list, int want) {
        if (list.empty()) return true;
        size_t p = 0;
        while (p < list.size()) {
            while (p < list.size() && (list[p] == ',' || list[p] == '+' || list[p] == ':' || list[p] == ' '))
                ++p;
            int v = 0;
            bool any = false;
            while (p < list.size() && list[p] >= '0' && list[p] <= '9') {
                v = v * 10 + (list[p] - '0');
                any = true;
                ++p;
            }
            if (any && v == want) return true;
            while (p < list.size() && list[p] != ',' && list[p] != '+' && list[p] != ':' && list[p] != ' ')
                ++p;
        }
        return false;
    };
    const bool req_attn_all = !mode_is_off(attn_requant_mode) &&
        (attn_requant_mode == "1" || mode_token("all") || mode_token("true") || mode_token("TRUE") ||
         mode_token("on") || mode_token("ON") || mode_token("yes") || mode_token("YES"));
    // Qwythos Q4_K_M leaves one linear-attention QKV matrix in Q6_K at decode (layer 2 was the
    // sensitive outlier in early gates; included in default list after re-validation).
    const char* qkv_layers_env = getenv("SPARKINFER_ATTN_REQUANT_Q4K_QKV_LAYERS");
    const std::string qkv_requant_layers =
        qkv_layers_env ? std::string(qkv_layers_env)
                       : ((q35_dense9b_requant_default && !attn_env)
                            ? std::string("0,1,2,6,9,12,18,21,24,28,29,30")
                            : std::string());
    int qkv_requant_limit = -1;
    if (const char* ql = getenv("SPARKINFER_ATTN_REQUANT_Q4K_QKV_LIMIT")) {
        qkv_requant_limit = atoi(ql);
        if (qkv_requant_limit < 0) qkv_requant_limit = -1;
    }
    int qkv_requant_used = 0;
    // Qwen3.6 GDN ssm_out projections ship Q8_0; requant them to Q4_K by default (all
    // thirty out-projections). SPARKINFER_ATTN_REQUANT_Q4K_SSM_MINLAYER pins a lower
    // bound on the layer index — the early GDN layers seed the recurrent state and are
    // the most precision-sensitive, so raising this trades a little decode speed for a
    // higher fuzzed top-1 margin.
    int ssm_out_min_layer = 0;
    if (const char* e = getenv("SPARKINFER_ATTN_REQUANT_Q4K_SSM_MINLAYER"))
        ssm_out_min_layer = atoi(e);
    auto req_attn_q4 = [&](const std::string& name, int ggml_type) {
        if (mode_is_off(attn_requant_mode)) return false;
        if (req_attn_all) return true;
        if ((mode_token("qkv") || mode_token("linear")) && has_suffix(name, "attn_qkv.weight")) {
            if (!int_list_has(qkv_requant_layers, layer_index(name))) return false;
            if (ggml_type == 14 && qkv_requant_limit >= 0 && qkv_requant_used++ >= qkv_requant_limit)
                return false;
            return true;
        }
        if ((mode_token("v") || mode_token("attn_v")) && has_suffix(name, "attn_v.weight")) return true;
        if ((mode_token("q") || mode_token("attn_q")) && has_suffix(name, "attn_q.weight")) return true;
        if ((mode_token("k") || mode_token("attn_k")) && has_suffix(name, "attn_k.weight")) return true;
        if ((mode_token("o") || mode_token("out") || mode_token("attn_output")) &&
            has_suffix(name, "attn_output.weight")) return true;
        // Qwen3.6 GDN input projections ship Q8_0 (the single largest per-token weight read):
        // attn_qkv (wqkv, handled by the "qkv" token above) + attn_gate (the z gate). Requant
        // both to Q4_K so they route through the existing Q4_K fused GDN qkv+z kernel (~47% fewer
        // bytes). SPARKINFER_ATTN_REQUANT_Q4K=attn_q,attn_output restores the #353-only behavior.
        if (mode_token("attn_gate") && has_suffix(name, "attn_gate.weight")) return true;
        if (mode_token("ssm_out") && has_suffix(name, "ssm_out.weight"))
            return layer_index(name) >= ssm_out_min_layer;
        return false;
    };
    const bool dual_dflash_lm_head = env_enabled(
        "SPARKINFER_DFLASH_DUAL_LMHEAD",
        q36_ud_requant_default && !q35_dense9b_requant_default);
    const bool req_lm_q4 = env_enabled("SPARKINFER_LMHEAD_REQUANT_Q4K",
                                       q35_dense9b_requant_default || dual_dflash_lm_head);
    // A ternary tensor arrives as Q4_K once its rotation is folded in, so it belongs on the same
    // MMVQ kernels every other quantized checkpoint uses. Letting it fall through to dense() would
    // expand the attention weights to bf16 -- several GB, and a GEMV path this architecture's GDN
    // projections otherwise never take.
    // A residual-width ternary projection in its stored blocks. The GDN v-head order is NOT
    // uploaded verbatim: everything that produces a v head stores its 48 heads transposed, and the
    // folded path regroups them while un-rotating (see the native branch below).
    auto upload_proj_native = [&](const GGUFTensor* t, const std::string& name,
                                  UnrotateJob& j) -> void* {
        if (had.gdn_v_grouped) unrotate_job_set_v_block(j, name, s.cfg);
        const size_t row_bytes = (size_t)(j.width / kPtq1BlockElems) * kPtq1BlockBytes;
        std::vector<uint8_t> host((size_t)t->n_bytes);
        const auto* src = static_cast<const uint8_t*>(t->data);
        for (long r = 0; r < j.rows; ++r)
            std::memcpy(host.data() + (size_t)r * row_bytes,
                        src + (size_t)unrotate_source_row(j, r) * row_bytes, row_bytes);
        void* d = nullptr;
        if (cudaMalloc(&d, t->n_bytes) == cudaSuccess &&
            cudaMemcpy(d, host.data(), t->n_bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
            s.owned.push_back(d);
            return d;
        }
        cudaFree(d);
        return nullptr;
    };
    // The stored blocks as they are: the FFN legs and the head carry no v-head regrouping.
    auto upload_plain_native = [&](const GGUFTensor* t) -> void* {
        void* d = nullptr;
        if (cudaMalloc(&d, t->n_bytes) == cudaSuccess &&
            cudaMemcpy(d, t->data, t->n_bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
            s.owned.push_back(d);
            return d;
        }
        cudaFree(d);
        return nullptr;
    };
    auto attn_w_base = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        // Only projections whose input is the residual width: those read the once-per-layer
        // rotated xn in decode, and prefill's dq() carries the matching sign vector.
        const bool proj_name_ok =
            (!bonsai_proj_qkv_only && !bonsai_proj_gate_only) ||
            (bonsai_proj_qkv_only && name.find(".attn_qkv.") != std::string::npos) ||
            (bonsai_proj_gate_only && name.find(".attn_gate.") != std::string::npos);
        if (bonsai_native_proj && proj_name_ok && t && t->ggml_type == kPtq1GgmlType &&
            s.bonsai_rot_xn && t->dims[0] == s.cfg.hidden && s.bonsai_sign_dev.count(t->dims[0]) &&
            tp_decide(name, s.rank, nullptr) == TpDec::Whole) {   // the native upload is whole-tensor only
            // The blocks go up as they are, but the GDN v-head order does NOT: everything that
            // produces a v head stores its 48 heads transposed, and the folded path regroups them
            // while un-rotating. Uploading verbatim skips that, which is why attn_gate (all v) was
            // far more wrong than attn_qkv (v is one third of it). A row is a whole number of
            // 28-byte blocks, so the regrouping is a byte-level permutation with no decoding.
            UnrotateJob j;
            if (!unrotate_job_init(j, t, name, *had.signs_for(t->dims[0]), had.block_size, false))
                return nullptr;
            if (void* d = upload_proj_native(t, name, j)) {
                type = kPtq1GgmlType;
                return d;
            }
            fprintf(stderr, "[bonsai] %s: native upload failed, falling back\n", name.c_str());
        }
        if (qattn && t && t->ggml_type == kPtq1GgmlType) return dev_quant(name, type);
        if (qattn && t && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8))
            return dev_quant_requant_q4k(name, type, req_attn_q4(name, t->ggml_type));
        type = 0; return dense(name, false);
    };
    // attn_w_base, plus the decode shadow's ternary copy of every residual-width projection.
    auto attn_w = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        void* sh = nullptr;
        if (shadow_proj && t && t->ggml_type == kPtq1GgmlType && s.bonsai_rot_xn &&
            t->dims[0] == s.cfg.hidden && s.bonsai_sign_dev.count(t->dims[0]) &&
            tp_decide(name, s.rank, nullptr) == TpDec::Whole) {   // the shadow upload is whole-tensor only
            UnrotateJob j;
            if (unrotate_job_init(j, t, name, *had.signs_for(t->dims[0]), had.block_size, false))
                sh = upload_proj_native(t, name, j);
            if (sh) { s.owned.pop_back(); s.bonsai_dec_bufs.push_back(sh); }
        }
        const void* p = attn_w_base(name, type);
        if (sh && p) shadow_of[p] = sh;
        return p;
    };
    // The dense FFN's three matrices, left in their stored blocks. No v-head regrouping applies:
    // that is a property of tensors PRODUCING a GDN v head, and none of these does. Gate and up
    // read the residual width, down reads the FFN width -- both sign vectors are required up
    // front, because a down leg that fell back to Q4_K while gate/up went native would be reading
    // a correctly-rotated activation with un-rotated weights.
    auto ffn_w = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (bonsai_native_ffn && s.bonsai_ffn_h && t && t->ggml_type == kPtq1GgmlType &&
            s.bonsai_sign_dev.count(t->dims[0]) &&
            tp_decide(name, s.rank, nullptr) == TpDec::Whole) {   // the native upload is whole-tensor only
            void* d = nullptr;
            if (cudaMalloc(&d, t->n_bytes) == cudaSuccess &&
                cudaMemcpy(d, t->data, t->n_bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
                s.owned.push_back(d);
                type = kPtq1GgmlType;
                return d;
            }
            cudaFree(d);
            fprintf(stderr, "[bonsai] %s: native upload failed, falling back\n", name.c_str());
        }
        return nullptr;   // caller falls through to its existing quantized path
    };
    // The folded FFN leg, plus its ternary copy for the decode shadow.
    auto ffn_q_shadow = [&](const std::string& name, const void* folded) {
        const GGUFTensor* t = g.tensor(name);
        if (!shadow_ffn || !folded || !s.bonsai_ffn_h || !t ||
            t->ggml_type != kPtq1GgmlType || !s.bonsai_sign_dev.count(t->dims[0]) ||
            tp_decide(name, s.rank, nullptr) != TpDec::Whole) return;   // the shadow upload is whole-tensor only
        if (void* d = upload_plain_native(t)) {
            s.owned.pop_back();
            s.bonsai_dec_bufs.push_back(d);
            shadow_of[folded] = d;
        }
    };
    auto attn_w_opt = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { type = 0; return nullptr; }
        if (qattn && t->ggml_type == kPtq1GgmlType) return dev_quant(name, type);
        if (qattn && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8))
            return dev_quant_requant_q4k(name, type, req_attn_q4(name, t->ggml_type));
        type = 0; return dense(name, false);
    };
    // Muse Glimmer ships output.weight as Q5_K -- the only Q5_K tensor in the file -- and Q5_K was
    // not on this list, so the head fell through to `dense()` and was dequantized to bf16. The
    // decode LM head then read 2.69 GB every token (gemv_f32_sk) instead of 0.76 GB through the
    // Q4_K MMVQ path: 1.60 ms of a 12.4 ms step. Requanting it to Q4_K at load is the same
    // mechanism this codebase already applies to other models' heads (SPARKINFER_LMHEAD_REQUANT_Q4K).
    static const bool mg_lm_q5k = [] {
        const char* e = getenv("SPARKINFER_MUSE_LMHEAD_Q5K");
        return !(e && e[0] == '0');
    }();
    auto lm_w = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        const bool q5k_ok = mg_lm_q5k && s.cfg.muse_glimmer && t && t->ggml_type == 13;
        if (bonsai_native_head && t && t->ggml_type == kPtq1GgmlType && s.bonsai_rot &&
            s.bonsai_sign_dev.count(t->dims[0]) &&
            tp_decide(name, s.rank, nullptr) == TpDec::Whole) {   // the native upload is whole-tensor only
            // Straight upload: no un-rotation, no refit, 0.21875 bytes/weight.
            void* d = nullptr;
            if (cudaMalloc(&d, t->n_bytes) == cudaSuccess &&
                cudaMemcpy(d, t->data, t->n_bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
                s.owned.push_back(d);
                type = kPtq1GgmlType;
                return d;
            }
            cudaFree(d);
            fprintf(stderr, "[bonsai] %s: native upload failed, falling back\n", name.c_str());
        }
        if (qattn && t && t->ggml_type == kPtq1GgmlType) return dev_quant(name, type);
        if (qattn && t && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8 || q5k_ok))
            return dev_quant_requant_q4k(name, type, req_lm_q4 || q5k_ok, q5k_ok);
        type = 0; return dense(name, false);
    };
    auto dense_opt = [&](const std::string& name, bool transpose) -> const void* {
        return g.tensor(name) ? dense(name, transpose) : nullptr;
    };
    // Returns null unless this checkpoint declares the transposed GDN v order, so every other
    // model keeps its existing loader untouched.
    // Everything that PRODUCES a GDN v head is stored transposed, so all of it is regrouped: the
    // per-head scalars, the alpha/beta projections and conv1d's v channels. The runtime has to see
    // the architecture's own order because the q/k-to-v grouping is baked into the GDN kernel --
    // v head r is driven by q/k head r/3 -- so leaving the file's order in place is not an option.
    // SPARKINFER_BONSAI_VREGROUP narrows the set, which is how the missing conv1d was found.
    static const std::string vregroup_set = [] {
        const char* e = getenv("SPARKINFER_BONSAI_VREGROUP");
        return std::string(e && e[0] ? e : "a,dt,alpha,beta,conv");
    }();
    auto v_regroup = [&](const std::string& name) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t || !had.present || !had.gdn_v_grouped ||
            tp_decide(name, s.rank, nullptr) != TpDec::Whole) return nullptr;   // the regroup is a whole-tensor permute
        const char* key = name.find("ssm_a") != std::string::npos && name.find("alpha") == std::string::npos
                              ? "a"
                        : name.find("ssm_dt") != std::string::npos ? "dt"
                        : name.find("alpha") != std::string::npos ? "alpha"
                        : name.find("beta") != std::string::npos ? "beta"
                        : name.find("ssm_conv1d") != std::string::npos ? "conv" : "";
        if (!key[0] || vregroup_set.find(key) == std::string::npos) return nullptr;
        // conv1d carries q, k and then v across its channel axis; only the v channels move, and
        // they move in blocks of the GDN head dimension rather than one row per head.
        const bool is_conv = name.find("ssm_conv1d") != std::string::npos;
        const long row0 = is_conv ? 2 * (long)s.cfg.linear_q_heads * s.cfg.linear_head_dim : 0;
        const long rows_per_head = is_conv ? s.cfg.linear_head_dim : 1;
        void* d = upload_v_regrouped_bf16(t, name, row0, s.cfg.linear_v_heads,
                                          s.cfg.linear_q_heads, rows_per_head);
        if (d) s.owned.push_back(d);
        return d;
    };
    auto expect_dims = [&](const std::string& name, std::initializer_list<long> dims) -> bool {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gguf] missing %s\n", name.c_str()); return false; }
        if (t->n_dims != (int)dims.size()) {
            fprintf(stderr, "[gguf] bad rank for %s: got %d want %zu\n",
                    name.c_str(), t->n_dims, dims.size());
            return false;
        }
        int i = 0;
        for (long want : dims) {
            if (t->dims[i] != want) {
                fprintf(stderr, "[gguf] bad shape for %s dim%d: got %ld want %ld\n",
                        name.c_str(), i, t->dims[i], want);
                return false;
            }
            i++;
        }
        return true;
    };
    auto expect_dims_opt = [&](const std::string& name, std::initializer_list<long> dims) -> bool {
        return !g.tensor(name) || expect_dims(name, dims);
    };

    if (const GGUFTensor* emb_t = g.tensor("token_embd.weight");
        bonsai_native_embed && emb_t && emb_t->ggml_type == kPtq1GgmlType && s.bonsai_rot &&
        s.bonsai_sign_dev.count(emb_t->dims[0]) &&
        tp_decide("token_embd.weight", s.rank, nullptr) == TpDec::Whole) {   // the native upload is whole-tensor only
        void* d = nullptr;
        if (cudaMalloc(&d, emb_t->n_bytes) == cudaSuccess &&
            cudaMemcpy(d, emb_t->data, emb_t->n_bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
            s.owned.push_back(d);
            s.w.embed_tokens = d;
            s.bonsai_embed_native = true;              // 0.28 GB of table instead of 2.54 in bf16
        } else {
            cudaFree(d);
        }
    }
    if (!s.bonsai_embed_native)
        s.w.embed_tokens = dense("token_embd.weight", false);     // [vocab,hidden] as-is
    s.w.final_norm   = dense("output_norm.weight", false);
    const char* lm = g.tensor("output.weight") ? "output.weight" : "token_embd.weight";  // tied fallback
    const GGUFTensor* lm_tensor = g.tensor(lm);
    if (dual_dflash_lm_head && req_lm_q4 && lm_tensor &&
        (lm_tensor->ggml_type == 14 || lm_tensor->ggml_type == 8)) {
        s.dflash_lm_head = dev_quant(lm, s.dflash_lm_head_type);
    }
    s.w.lm_head = lm_w(lm, s.w.lm_head_type);                 // native [vocab,hidden] for GEMV
    if (shadow_head && s.w.lm_head_type != kPtq1GgmlType && s.bonsai_rot &&
        s.bonsai_sign_dev.count(s.cfg.hidden) &&
        tp_decide(lm, s.rank, nullptr) == TpDec::Whole) {   // the shadow upload is whole-tensor only
        const GGUFTensor* t = g.tensor(lm);
        if (t && t->ggml_type == kPtq1GgmlType && t->dims[0] == s.cfg.hidden) {
            if (void* d = upload_plain_native(t)) {
                s.owned.pop_back();
                s.bonsai_dec_bufs.push_back(d);
                s.bonsai_dec_head = d;
            }
        }
    }
    // A null pointer is a real miss only if this rank OWNS the name per the placement
    // table (D2); a skipped rank legitimately holds nothing of it, and the loaders
    // return null for exactly that case (they never report a skip as a failure).
    auto miss = [&](const std::string& nm, const void* p) {
        return !p && tp_decide(nm, s.rank, nullptr) != TpDec::Skip;
    };
    if (miss("token_embd.weight", s.w.embed_tokens) ||
        miss("output_norm.weight", s.w.final_norm) ||
        miss(lm, s.w.lm_head)) return false;

    s.w.layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        std::string b = "blk." + std::to_string(i) + ".";
        Qwen35LayerWeights& w = s.w.layers[i];
        w.linear_attn = is_linear_layer(c, i);
        w.swa = (i < (int)c.swa_layers.size()) ? c.swa_layers[i] : false;
        // The per-layer skip-aware twin of `miss` above: true when the pointer is set or
        // the table assigns this rank none of the name.
        auto ok = [&](const std::string& nm, const void* p) {
            return p || tp_decide(nm, s.rank, nullptr) == TpDec::Skip;
        };
        if (c.muse_glimmer && getenv("SPARKINFER_MG_DEBUG"))
            fprintf(stderr, "[mg-debug] layer %d: linear_attn=%d swa=%d full_attn_interval=%d hybrid=%d\n",
                    i, (int)w.linear_attn, (int)w.swa, c.full_attn_interval, (int)c.hybrid);
        if (!expect_dims(b + "attn_norm.weight", {H})) return false;
        w.input_norm = dense(b + "attn_norm.weight", false);
        if (w.linear_attn) {
            if (!expect_dims(b + "attn_qkv.weight", {H, s.linear_qkvdim}) ||
                !expect_dims(b + "attn_gate.weight", {H, s.linear_vdim}) ||
                !expect_dims(b + "ssm_conv1d.weight", {c.linear_conv_kernel, s.linear_qkvdim}) ||
                !expect_dims(b + "ssm_dt.bias", {c.linear_v_heads}) ||
                !expect_dims(b + "ssm_a", {c.linear_v_heads}) ||
                !expect_dims(b + "ssm_beta.weight", {H, c.linear_v_heads}) ||
                !expect_dims(b + "ssm_alpha.weight", {H, c.linear_v_heads}) ||
                !expect_dims(b + "ssm_norm.weight", {c.linear_head_dim}) ||
                !expect_dims(b + "ssm_out.weight", {s.linear_vdim, H})) return false;
            w.wqkv = attn_w(b + "attn_qkv.weight", w.wqkv_type);
            w.wqkv_gate = attn_w(b + "attn_gate.weight", w.wqkv_gate_type);
            w.ssm_conv = v_regroup(b + "ssm_conv1d.weight");
            if (!w.ssm_conv) w.ssm_conv = dense(b + "ssm_conv1d.weight", false);
            w.ssm_dt = v_regroup(b + "ssm_dt.bias");
            if (!w.ssm_dt) w.ssm_dt = dense(b + "ssm_dt.bias", false);
            w.ssm_a = v_regroup(b + "ssm_a");
            if (!w.ssm_a) w.ssm_a = dense(b + "ssm_a", false);
            if (const void* bp = v_regroup(b + "ssm_beta.weight")) {
                w.ssm_beta = bp; w.ssm_beta_type = 0;
            } else {
                w.ssm_beta = attn_w(b + "ssm_beta.weight", w.ssm_beta_type);
            }
            if (const void* ap = v_regroup(b + "ssm_alpha.weight")) {
                w.ssm_alpha = ap; w.ssm_alpha_type = 0;
            } else {
                w.ssm_alpha = attn_w(b + "ssm_alpha.weight", w.ssm_alpha_type);
            }
            w.ssm_norm = dense(b + "ssm_norm.weight", false);
            w.ssm_out = attn_w(b + "ssm_out.weight", w.ssm_out_type);
        } else {
            w.q_has_gate = c.hybrid;
            if (c.muse_glimmer) {
                // attn_q.weight and attn_gate.weight ship as two separate [H, qdim]
                // tensors (unlike Qwen3.6's GGUF, which pre-fuses them into one [H,
                // qdim*2] tensor before writing the file) -- dequantize both to bf16 and
                // interleave into the per-head [q|gate] layout split_q_gate_kernel
                // (qwen36.cu) expects. Load-time only; w.wq ends up dense bf16 (wq_type=0)
                // rather than kept-quantized, same as this path's non-gguf/dense fallback.
                if (!expect_dims(b + "attn_q.weight", {H, s.qdim}) ||
                    !expect_dims(b + "attn_gate.weight", {H, s.qdim}) ||
                    !expect_dims(b + "attn_k.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_v.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_output.weight", {s.qdim, H}) ||
                    !expect_dims(b + "attn_q_norm.weight", {c.head_dim}) ||
                    !expect_dims(b + "attn_k_norm.weight", {c.head_dim})) return false;
                // SPARKINFER_MUSE_QGATE_Q=0 restores the original load-time dequantize+interleave
                // (kept for a same-binary A/B; see the projection site for the matching switch).
                static const int kQGateQ = []{ const char* e = getenv("SPARKINFER_MUSE_QGATE_Q");
                                               return (e && e[0] == '0') ? 0 : 1; }();
                if (kQGateQ) {
                    // attn_q and attn_gate both ship Q4_K. Dequantizing them to bf16 so they can be
                    // interleaved into the [q|gate] layout split_q_gate_kernel wants costs 109 MB
                    // per layer against 31 MB kept quantized -- 4.08 GB of extra reads on EVERY
                    // decode token across 52 layers, which profiled as the single largest kernel in
                    // the 128-decode run (gemv_f32_sk, 29.8% of GPU time). The interleave only
                    // exists because Qwen3.6's GGUF pre-fuses q|gate into one tensor; Muse ships
                    // them separately, so keep both quantized and project each straight into its
                    // own destination, which also drops the split entirely.
                    w.wq    = attn_w(b + "attn_q.weight", w.wq_type);
                    w.wgate = attn_w(b + "attn_gate.weight", w.wgate_type);
                    if (!ok(b + "attn_q.weight", w.wq) || !ok(b + "attn_gate.weight", w.wgate)) return false;
                } else {
                const void* qd = dense(b + "attn_q.weight", false);
                const void* gd = dense(b + "attn_gate.weight", false);
                if (!ok(b + "attn_q.weight", qd) || !ok(b + "attn_gate.weight", gd)) return false;
                void* combined = nullptr;
                cu(cudaMalloc(&combined, (size_t)s.qdim * 2 * H * sizeof(bf16)), "qgate interleave alloc");
                // cu() only logs CUDA errors, it never aborts -- unlike qd/gd above, nothing
                // downstream checks `combined` before using it. On a real cudaMalloc failure
                // (OOM; trivially reproducible by running this load under compute-sanitizer,
                // whose shadow-memory overhead multiplies every allocation) `combined` stays
                // null/stale and launch_interleave_qgate_rows below writes through it --
                // out-of-bounds device writes rather than a clean load failure.
                if (!combined) return false;
                kernels::launch_interleave_qgate_rows(qd, gd, combined, c.n_q_heads, c.head_dim, H, s.stream);
                cu(cudaStreamSynchronize(s.stream), "qgate interleave sync");
                s.owned.push_back(combined);
                w.wq = combined;
                w.wq_type = 0;
                }
                w.wk = attn_w(b + "attn_k.weight", w.wk_type);
                w.wv = attn_w(b + "attn_v.weight", w.wv_type);
                w.wo = attn_w(b + "attn_output.weight", w.wo_type);
                w.q_norm = dense(b + "attn_q_norm.weight", false);
                w.k_norm = dense(b + "attn_k_norm.weight", false);
            } else {
                const int q_out = w.q_has_gate ? s.qdim * 2 : s.qdim;
                if (!expect_dims(b + "attn_q.weight", {H, q_out}) ||
                    !expect_dims(b + "attn_k.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_v.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_output.weight", {s.qdim, H}) ||
                    !expect_dims(b + "attn_q_norm.weight", {c.head_dim}) ||
                    !expect_dims(b + "attn_k_norm.weight", {c.head_dim})) return false;
                w.wq = attn_w(b + "attn_q.weight", w.wq_type);
                w.wk = attn_w(b + "attn_k.weight", w.wk_type);
                w.wv = attn_w(b + "attn_v.weight", w.wv_type);
                w.wo = attn_w(b + "attn_output.weight", w.wo_type);
                w.q_norm = dense(b + "attn_q_norm.weight", false);
                w.k_norm = dense(b + "attn_k_norm.weight", false);
            }
        }
        if (!expect_dims_opt(b + "attn_post_norm.weight", {H}) ||
            !expect_dims_opt(b + "post_attention_norm.weight", {H}) ||
            !expect_dims_opt(b + "ffn_norm.weight", {H}) ||
            !expect_dims_opt(b + "post_ffw_norm.weight", {H})) return false;
        if (c.muse_glimmer) {
            // Muse Glimmer ships post_attention_norm.weight AND ffn_norm.weight as distinct
            // tensors (sandwich norm, not one norm serving double duty) -- load both,
            // unlike the single-fallback-chain below every other architecture uses.
            w.post_attn_norm = dense(b + "post_attention_norm.weight", false);
            w.ffn_norm = dense(b + "ffn_norm.weight", false);
            w.post_ffn_norm = dense(b + "post_ffw_norm.weight", false);
        } else {
            w.post_attn_norm = dense_opt(b + "attn_post_norm.weight", false);
            if (!w.post_attn_norm) w.post_attn_norm = dense_opt(b + "post_attention_norm.weight", false);
            if (!w.post_attn_norm) w.post_attn_norm = dense(b + "ffn_norm.weight", false);
        }
        if (c.dense_ffn) {
            if (!expect_dims(b + "ffn_gate.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_up.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_down.weight", {c.moe_ffn, H})) return false;
            // Gate and up convert together or not at all because the compact MMVQ kernel
            // consumes equal-stride pairs; unselected layers retain the native Q4_K path.
            const bool gu3 = ffn_q3a_on("ffn_gate") && ffn_q3a_on("ffn_up") &&
                             int_list_has(ffn_q3a_layers, i);
            // Native ternary first, and all three together: the decode branch below needs the
            // whole SwiGLU in one basis, so a partial take is worse than none.
            w.gate_q = ffn_w(b + "ffn_gate.weight", w.gate_qtype);
            w.up_q   = w.gate_q ? ffn_w(b + "ffn_up.weight", w.up_qtype) : nullptr;
            w.down_q = w.up_q   ? ffn_w(b + "ffn_down.weight", w.down_qtype) : nullptr;
            if (!(w.gate_q && w.up_q && w.down_q)) {
                w.gate_q = gu3 ? dev_quant_q3a(b + "ffn_gate.weight", w.gate_qtype, &w.prefill_gate_q, &w.prefill_gate_qtype)
                               : dev_quant(b + "ffn_gate.weight", w.gate_qtype);
                w.up_q   = gu3 ? dev_quant_q3a(b + "ffn_up.weight", w.up_qtype, &w.prefill_up_q, &w.prefill_up_qtype)
                               : dev_quant(b + "ffn_up.weight", w.up_qtype);
                w.down_q = dev_quant_down(b + "ffn_down.weight", w.down_qtype);
                ffn_q_shadow(b + "ffn_gate.weight", w.gate_q);
                ffn_q_shadow(b + "ffn_up.weight", w.up_q);
                ffn_q_shadow(b + "ffn_down.weight", w.down_q);
            }
        } else {
            if (!expect_dims(b + "ffn_gate_inp.weight", {H, c.n_experts})) return false;
            // Router weight: keep Q8_0 raw if present in the GGUF (half bandwidth, on-read GEMV)
            {
                const GGUFTensor* rt = g.tensor(b + "ffn_gate_inp.weight");
                if (qattn && rt && rt->ggml_type == 8) {
                    w.router_w = dev_quant(b + "ffn_gate_inp.weight", w.router_w_type);
                } else {
                    w.router_w = dense(b + "ffn_gate_inp.weight", false);
                    w.router_w_type = 0;
                }
            }
            w.gate_q = dev_quant(b + "ffn_gate_exps.weight", w.gate_qtype);   // kept quantized
            w.up_q   = dev_quant(b + "ffn_up_exps.weight",   w.up_qtype);
            w.down_q = dev_quant(b + "ffn_down_exps.weight", w.down_qtype);
            if (s.cfg.n_shared > 0) {
            if (!expect_dims(b + "ffn_gate_shexp.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_up_shexp.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_down_shexp.weight", {c.moe_ffn, H}) ||
                !expect_dims_opt(b + "ffn_gate_inp_shexp.weight", {H})) return false;
            // GGUF-native [out,in] layout (no transpose) so the shared expert runs as
            // three fast one-warp-per-row GEMVs instead of the single-block dense kernel.
            const bool qmoe = []{ const char* a = getenv("SPARKINFER_QMOE");
                                   return !(a && a[0] == '0'); }();
            if (qmoe) {
                w.shared_gate_q = dev_quant(b + "ffn_gate_shexp.weight", w.shared_gate_qtype);
                w.shared_up_q   = dev_quant(b + "ffn_up_shexp.weight",   w.shared_up_qtype);
                w.shared_down_q = dev_quant(b + "ffn_down_shexp.weight", w.shared_down_qtype);
            }
            if (!qmoe || !w.shared_gate_q || !w.shared_up_q || !w.shared_down_q ||
                w.shared_gate_qtype != 8) {
                w.shared_gate = dense(b + "ffn_gate_shexp.weight", false);
                w.shared_up   = dense(b + "ffn_up_shexp.weight", false);
                w.shared_down = dense(b + "ffn_down_shexp.weight", false);
            }
            w.shared_gate_inp = attn_w_opt(b + "ffn_gate_inp_shexp.weight", w.shared_gate_inp_type);
            const bool have_shared_q =
                ok(b + "ffn_gate_shexp.weight", w.shared_gate_q) &&
                ok(b + "ffn_up_shexp.weight", w.shared_up_q) &&
                ok(b + "ffn_down_shexp.weight", w.shared_down_q);
            const bool have_shared_d =
                ok(b + "ffn_gate_shexp.weight", w.shared_gate) &&
                ok(b + "ffn_up_shexp.weight", w.shared_up) &&
                ok(b + "ffn_down_shexp.weight", w.shared_down);
            if (!have_shared_q && !have_shared_d) return false;
            }
        }
        // Nulls are tolerated here only where the placement table says this rank owns
        // none of the name (Skipped names load null on purpose); anything else is a miss.
        const bool have_attn = w.linear_attn
            ? (ok(b + "attn_qkv.weight", w.wqkv) && ok(b + "attn_gate.weight", w.wqkv_gate) &&
               ok(b + "ssm_conv1d.weight", w.ssm_conv) && ok(b + "ssm_dt.bias", w.ssm_dt) &&
               ok(b + "ssm_a", w.ssm_a) && ok(b + "ssm_beta.weight", w.ssm_beta) &&
               ok(b + "ssm_alpha.weight", w.ssm_alpha) && ok(b + "ssm_norm.weight", w.ssm_norm) &&
               ok(b + "ssm_out.weight", w.ssm_out))
            : (ok(b + "attn_q.weight", w.wq) && ok(b + "attn_k.weight", w.wk) &&
               ok(b + "attn_v.weight", w.wv) && ok(b + "attn_output.weight", w.wo) &&
               ok(b + "attn_q_norm.weight", w.q_norm) && ok(b + "attn_k_norm.weight", w.k_norm));
        const bool have_ffn = c.dense_ffn
            ? (ok(b + "ffn_gate.weight", w.gate_q) && ok(b + "ffn_up.weight", w.up_q) &&
               ok(b + "ffn_down.weight", w.down_q))
            : (ok(b + "ffn_gate_inp.weight", w.router_w) &&
               ok(b + "ffn_gate_exps.weight", w.gate_q) &&
               ok(b + "ffn_up_exps.weight", w.up_q) &&
               ok(b + "ffn_down_exps.weight", w.down_q));
        // post_attn_norm is a fallback chain (attn_post_norm -> post_attention_norm ->
        // ffn_norm): a null is a miss only if at least one link of the chain is owned here.
        const bool post_attn_norm_ok = w.post_attn_norm ||
            (tp_decide(b + "attn_post_norm.weight", s.rank, nullptr) == TpDec::Skip &&
             tp_decide(b + "post_attention_norm.weight", s.rank, nullptr) == TpDec::Skip &&
             tp_decide(b + "ffn_norm.weight", s.rank, nullptr) == TpDec::Skip);
        if (!have_attn || !ok(b + "attn_norm.weight", w.input_norm) || !post_attn_norm_ok ||
            !have_ffn)
            return false;
        if (c.muse_glimmer &&
            (!ok(b + "ffn_norm.weight", w.ffn_norm) ||
             !ok(b + "post_ffw_norm.weight", w.post_ffn_norm)))
            return false;
        if (i == 0 || i == c.n_layers - 1) fprintf(stderr, "[gguf] layer %d loaded\n", i);
    }
    // ---- eager per-row int8 scales of the routed experts (fused quantized-B MoE prefill GEMM) ----
    // The batched prefill can run the routed GEMMs straight off the native GGUF expert weights
    // instead of materializing the whole int8 expert pool once per layer (prefill_moe_q.cu). That
    // needs the per-row int8 scale, which is a property of the FULL row (amax over all `cols`) and
    // so cannot be derived from a K-tile inside the GEMM. Compute it here with the same kernel the
    // materialize path uses and keep only its `scale` output: the fused GEMM then quantizes to the
    // identical int8 bytes by construction, not by re-deriving the scale.
    // ~120 MB for Qwen3.6-35B-A3B (40 layers x 256 experts x (512+512+2048) rows). Any failure
    // leaves the pointers null and the prefill simply keeps materializing.
    // SPARKINFER_PREFILL_MOE_QB=0 skips the precompute entirely.
    if (!c.dense_ffn && c.n_experts > 0 && c.moe_ffn > 0) {
        const char* qb_env = getenv("SPARKINFER_PREFILL_MOE_QB");
        if (!qb_env || qb_env[0] != '0') {
            const size_t rg = (size_t)c.n_experts * c.moe_ffn;   // gate/up rows per layer
            const size_t rd = (size_t)c.n_experts * H;           // down rows per layer
            const size_t ng = rg * (size_t)c.n_layers, nd = rd * (size_t)c.n_layers;
            const size_t tmp_bytes = 64u << 20;                  // int8 scratch, thrown away
            signed char* tmp = nullptr;
            bool ok = cudaMalloc(&s.moe_rs_gate, ng * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&s.moe_rs_up,   ng * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&s.moe_rs_down, nd * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&tmp, tmp_bytes) == cudaSuccess;
            auto fill = [&](int qtype, const void* src, float* dst, size_t rows, int cols) {
                const int blk = (qtype == 12) ? 144 : (qtype == 13) ? 176 : 210;
                const size_t rb = (size_t)(cols >> 8) * (size_t)blk;   // bytes per quantized row
                const size_t chunk = tmp_bytes / (size_t)cols;
                for (size_t r0 = 0; r0 < rows; r0 += chunk) {
                    const size_t nr = (rows - r0 < chunk) ? (rows - r0) : chunk;
                    if (!kernels::launch_gguf_dequant_rows_i8(
                            qtype, (const char*)src + r0 * rb, tmp, dst + r0,
                            (int)nr, cols, s.stream))
                        return false;
                }
                return true;
            };
            for (int i = 0; ok && i < c.n_layers; i++) {
                const Qwen35LayerWeights& lw = s.w.layers[i];
                if (!lw.gate_q || !lw.up_q || !lw.down_q) { ok = false; break; }
                ok = fill(lw.gate_qtype, lw.gate_q, s.moe_rs_gate + (size_t)i * rg, rg, H)
                  && fill(lw.up_qtype,   lw.up_q,   s.moe_rs_up   + (size_t)i * rg, rg, H)
                  && fill(lw.down_qtype, lw.down_q, s.moe_rs_down + (size_t)i * rd, rd, c.moe_ffn);
            }
            if (ok) ok = cudaStreamSynchronize(s.stream) == cudaSuccess;
            if (tmp) cudaFree(tmp);
            if (!ok) {
                cudaFree(s.moe_rs_gate); cudaFree(s.moe_rs_up); cudaFree(s.moe_rs_down);
                s.moe_rs_gate = s.moe_rs_up = s.moe_rs_down = nullptr;
                fprintf(stderr, "[prefill-moe] expert row-scale precompute unavailable "
                                "-> int8 materialize path\n");
            } else {
                fprintf(stderr, "[prefill-moe] expert int8 row scales ready (%.0f MB)\n",
                        (double)((2 * ng + nd) * sizeof(float)) / (1024.0 * 1024.0));
            }
        }
    }
    // ---- Muse Glimmer: eager per-row int8 scales for the fused quantized-B dense prefill GEMM ----
    // Muse's dense attn (wq/wgate/wk/wv/wo) and FFN gate/up ship as Q4_K. The batched prefill can
    // decode them to int8 inside the GEMM's B-stage (prefill_moe_q.cu's dense path) rather than
    // materializing the whole int8 weight per layer (dequant -> write W_i8 -> read W_i8 back). That
    // needs the per-output-row int8 scale -- amax over the FULL row -- which is computed here with
    // the SAME kernel the materialize path uses (launch_gguf_dequant_rows_i8, keeping only its
    // `scale`), so the fused GEMM's int8 bytes match the materialize path's by construction, not by
    // re-deriving the scale. Q6_K down (and any non-Q4/Q5 attn weight) is left null -> stays on the
    // materialize path. ~11 MB for Muse-30B. SPARKINFER_MUSE_PREFILL_QB=0 skips the precompute.
    // Nothing in the fused GEMM is Muse-specific; only this gate was. Every other dense GGUF
    // checkpoint left *_rs null, so proj_fused fell back to proj and each projection materialized
    // a full int8 copy of the weight -- written to DRAM and read straight back at exactly the rate
    // the fused arm exists to avoid. SPARKINFER_PREFILL_QB_DENSE=0 restores that.
    const bool qb_dense = c.dense_ffn && !c.muse_glimmer && [] {
        const char* e = getenv("SPARKINFER_PREFILL_QB_DENSE");
        return !(e && e[0] == '0');
    }();
    if (c.muse_glimmer || qb_dense) {
        const char* qb_env = getenv("SPARKINFER_MUSE_PREFILL_QB");
        if (qb_dense || !qb_env || qb_env[0] != '0') {
            // Q6_K too: the dense fused GEMM decodes it now, so a Q6_K attn_v / ffn_down gets its
            // row scales here instead of falling back to the per-layer materialize.
            auto fusable = [](int t) { return t == 12 || t == 13 || t == 14; };  // Q4_K / Q5_K / Q6_K
            const int qd = c.n_q_heads * c.head_dim;                   // qdim (4096)
            const int kd = c.n_kv_heads * c.head_dim;                  // kvdim (256)
            const int ff = c.moe_ffn;                                  // dense FFN width (19968)
            // + H for ffn_down: the original layout reserved no slot for it, so every Q4_K down
            // fell back to the materialize path regardless of being a fusable type. Slots are
            // reserved for all eight; a slot stays unfilled (and its *_rs null) when that
            // weight's type is not fusable, so a Q6_K down still takes the materialize path.
            // A hybrid checkpoint concatenates q and gate into one 2*qd-row attn_q, and its GDN
            // layers carry wqkv/wqkv_gate/ssm_out instead of q/k/v/o entirely.
            const int q_out = c.hybrid ? 2 * qd : qd;
            const int lqd   = c.linear_q_heads * c.linear_head_dim;
            const int lvd   = c.linear_v_heads * c.linear_head_dim;
            const int lqkv  = 2 * lqd + lvd;
            // A layer is either full-attention or GDN, never both, so the two families share one
            // region sized by the larger. SPARKINFER_PREFILL_QB_ATTN=0 reserves neither.
            const bool qb_attn = qb_dense && [] {
                const char* e = getenv("SPARKINFER_PREFILL_QB_ATTN");
                return !(e && e[0] == '0');
            }();
            const size_t att_rows = (size_t)q_out + 2 * (size_t)kd + (size_t)H;
            const size_t gdn_rows = (size_t)lqkv + (size_t)lvd + (size_t)H;
            const size_t mix = !qb_attn ? 0 : (att_rows > gdn_rows ? att_rows : gdn_rows);
            // Muse fills all eight slots; the dense arm fills only the three FFN ones plus
            // whichever attention family the layer actually has, so it reserves only those.
            const size_t per_layer = qb_dense ? (size_t)(2 * ff + H) + mix
                                              : (size_t)(2 * qd + 2 * kd + H + 2 * ff + H);  // rows/layer
            const size_t total = per_layer * (size_t)c.n_layers;
            const size_t tmp_bytes = 64u << 20;                        // int8 scratch, thrown away
            signed char* tmp = nullptr;
            bool ok = cudaMalloc(&s.muse_rs, total * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&tmp, tmp_bytes) == cudaSuccess;
            auto fill = [&](int qtype, const void* src, float* dst, size_t rows, int cols) -> bool {
                const int blk = (qtype == 12) ? 144 : (qtype == 13) ? 176 : 210;
                const size_t rb = (size_t)(cols >> 8) * (size_t)blk;   // bytes per quantized row
                const size_t chunk = tmp_bytes / (size_t)cols;        // rows/chunk fitting tmp
                for (size_t r0 = 0; r0 < rows; r0 += chunk) {
                    const size_t nr = (rows - r0 < chunk) ? (rows - r0) : chunk;
                    if (!kernels::launch_gguf_dequant_rows_i8(
                            qtype, (const char*)src + r0 * rb, tmp, dst + r0, (int)nr, cols, s.stream))
                        return false;
                }
                return true;
            };
            for (int i = 0; ok && i < c.n_layers; i++) {
                Qwen35LayerWeights& lw = s.w.layers[i];
                float* base = s.muse_rs + (size_t)i * per_layer;
                size_t off = 0;
                // Compute a weight's row scales into the pool and publish its *_rs pointer; the pool
                // slot is reserved for every weight (fixed layout) but filled only when fusable.
                auto place = [&](const void* W, int wt, const float** rs, int rows, int cols) {
                    float* dst = base + off; off += (size_t)rows;
                    if (ok && W && fusable(wt) && fill(wt, W, dst, (size_t)rows, cols)) *rs = dst;
                };
                if (!qb_dense) {
                    place(lw.wq,     lw.wq_type,     &lw.wq_rs,    qd, H);
                    place(lw.wgate,  lw.wgate_type,  &lw.wgate_rs, qd, H);
                    place(lw.wk,     lw.wk_type,     &lw.wk_rs,    kd, H);
                    place(lw.wv,     lw.wv_type,     &lw.wv_rs,    kd, H);
                    place(lw.wo,     lw.wo_type,     &lw.wo_rs,    H,  qd);
                }
                place(lw.prefill_gate_q ? lw.prefill_gate_q : lw.gate_q,
                      lw.prefill_gate_q ? lw.prefill_gate_qtype : lw.gate_qtype,
                      &lw.gate_rs, ff, H);
                place(lw.prefill_up_q ? lw.prefill_up_q : lw.up_q,
                      lw.prefill_up_q ? lw.prefill_up_qtype : lw.up_qtype,
                      &lw.up_rs, ff, H);
                place(lw.down_q, lw.down_qtype,  &lw.down_rs,  H,  ff);
                // Attention, for the dense arm. Exactly one of the two families exists on a given
                // layer, so both are placed at the same base and the cursor advances once, by the
                // larger. Keyed on which weight is actually present rather than on a layer index,
                // so a checkpoint whose attention interval differs from what its config says
                // cannot silently hand a GDN scale to a full-attention GEMM.
                if (qb_attn) {
                    const size_t mix_base = off;
                    if (lw.wq) {
                        place(lw.wq, lw.wq_type, &lw.wq_rs, q_out, H);
                        place(lw.wk, lw.wk_type, &lw.wk_rs, kd,    H);
                        place(lw.wv, lw.wv_type, &lw.wv_rs, kd,    H);
                        place(lw.wo, lw.wo_type, &lw.wo_rs, H,     qd);
                    } else if (lw.wqkv) {
                        place(lw.wqkv,      lw.wqkv_type,      &lw.wqkv_rs,      lqkv, H);
                        place(lw.wqkv_gate, lw.wqkv_gate_type, &lw.wqkv_gate_rs, lvd,  H);
                        place(lw.ssm_out,   lw.ssm_out_type,   &lw.ssm_out_rs,   H,    lvd);
                    }
                    off = mix_base + mix;
                }
            }
            if (ok) ok = cudaStreamSynchronize(s.stream) == cudaSuccess;
            if (tmp) cudaFree(tmp);
            if (!ok) {
                cudaFree(s.muse_rs); s.muse_rs = nullptr;
                for (int i = 0; i < c.n_layers; i++) {
                    Qwen35LayerWeights& lw = s.w.layers[i];
                    lw.wq_rs = lw.wgate_rs = lw.wk_rs = lw.wv_rs = lw.wo_rs = lw.gate_rs = lw.up_rs = nullptr;
                    lw.down_rs = nullptr;
                    lw.wqkv_rs = lw.wqkv_gate_rs = lw.ssm_out_rs = nullptr;
                }
                fprintf(stderr, "[prefill-muse] dense row-scale precompute unavailable "
                                "-> int8 materialize path\n");
            } else {
                fprintf(stderr, "[prefill-muse] dense int8 row scales ready (%.0f MB)\n",
                        (double)(total * sizeof(float)) / (1024.0 * 1024.0));
            }
        }
    }
    // Optional native Blackwell FP4 copies for Muse's gate/up projections. This is eager
    // because scored prefill times the first pass. Gate/up native prefill copies that are distinct
    // from their compact Q3_A decode weights are released after conversion, keeping peak resident
    // memory within a 32-GB card. The normal GGUF pointers remain the correctness fallback.
    // On by default now that the kernels are in the default build: the conversion is gated on
    // Muse plus a successful sm_120a FP4 build, and the GGUF pointers stay as the fallback, so a
    // box that could not build the FP4 path simply never takes it. SPARKINFER_MUSE_PREFILL_NVFP4=0
    // forces the portable quantized projections back.
    const char* fp4_env = getenv("SPARKINFER_MUSE_PREFILL_NVFP4");
    if (c.muse_glimmer && (!fp4_env || fp4_env[0] != '0') &&
        kernels::prefill_nvfp4_supported(128, c.moe_ffn, H) &&
        kernels::prefill_nvfp4_supported(128, H, c.moe_ffn)) {
        void* tmp = nullptr;
        const size_t tmp_elems = (size_t)c.moe_ffn * H;
        bool ok = cudaMalloc(&tmp, tmp_elems * sizeof(bf16)) == cudaSuccess;
        int ready = 0, qkvg_ready = 0;
        const int qdim_a = c.n_q_heads * c.head_dim;
        const int kvdim_a = c.n_kv_heads * c.head_dim;
        const char* fp4q_env = getenv("SPARKINFER_MUSE_PREFILL_NVFP4_QKV");
        // Not const: the VRAM budget below can drop this leg on a deployment whose concurrency
        // leaves no room for it (see the reserve there).
        bool qkvg_fp4_on = (!fp4q_env || fp4q_env[0] != '0') &&
                           kernels::prefill_nvfp4_supported(128, 2 * qdim_a + 2 * kvdim_a, H);
        // Whether this build/card COULD hold qkv-gate copies at all, kept across the budget below
        // so the partial fill after the layer loop can still run when the whole set was refused.
        const bool qkvg_eligible = qkvg_fp4_on;
        int down_ready = 0;
        auto convert = [&](const void* src, int qtype, int rows, int cols,
                           const void** data, const void** sf) -> bool {
            void *d = nullptr, *scale = nullptr;
            if (!src || cudaMalloc(&d, kernels::prefill_nvfp4_data_bytes(rows, cols)) != cudaSuccess)
                return false;
            if (cudaMalloc(&scale, kernels::prefill_nvfp4_scale_bytes_b(rows, cols)) != cudaSuccess) {
                cudaFree(d); return false;
            }
            kernels::launch_gguf_dequant(qtype, src, tmp, (long)rows * cols, s.stream);
            if (!kernels::launch_prefill_nvfp4_quant_b(tmp, d, scale, rows, cols, s.stream) ||
                cudaStreamSynchronize(s.stream) != cudaSuccess) {
                cudaFree(d); cudaFree(scale); return false;
            }
            s.owned.push_back(d); s.owned.push_back(scale); *data = d; *sf = scale;
            return true;
        };
        // Stack several row-blocks that share `cols` into one FP4 operand: dequantize each into its
        // slice of `tmp`, then quantize the whole thing once so the scale factors come out in the
        // single atom-tiled layout the GEMM expects for the combined row count.
        auto convert_group = [&](const void* const* src, const int* qtype, const int* rows,
                                 int nsrc, int cols, const void** data, const void** sf) -> bool {
            int total = 0;
            for (int i = 0; i < nsrc; ++i) {
                if (!src[i]) return false;
                total += rows[i];
            }
            if (!kernels::prefill_nvfp4_supported(128, total, cols) ||
                (size_t)total * cols > tmp_elems) return false;
            void *d = nullptr, *scale = nullptr;
            if (cudaMalloc(&d, kernels::prefill_nvfp4_data_bytes(total, cols)) != cudaSuccess)
                return false;
            if (cudaMalloc(&scale, kernels::prefill_nvfp4_scale_bytes_b(total, cols)) != cudaSuccess) {
                cudaFree(d); return false;
            }
            long off = 0;
            for (int i = 0; i < nsrc; ++i) {
                kernels::launch_gguf_dequant(qtype[i], src[i], (bf16*)tmp + off,
                                             (long)rows[i] * cols, s.stream);
                off += (long)rows[i] * cols;
            }
            if (!kernels::launch_prefill_nvfp4_quant_b(tmp, d, scale, total, cols, s.stream) ||
                cudaStreamSynchronize(s.stream) != cudaSuccess) {
                cudaFree(d); cudaFree(scale); return false;
            }
            s.owned.push_back(d); s.owned.push_back(scale); *data = d; *sf = scale;
            return true;
        };
        // The o-projection FP4 copy is decided ALL-OR-NOTHING before any layer converts, against
        // free VRAM plus a reserve. The whole set is the right call when it fits; when it does
        // not, refusing every layer used to leave the leftover on the table (qkv-gate and
        // ffn_down already take a prefix in that case). The prefix fill after the layer loop
        // does the same for wo. SPARKINFER_MUSE_NVFP4_WO=0 disables the leg; _RESERVE_MB tunes
        // the reserve; SPARKINFER_MUSE_NVFP4_WO_KEEP_MB=0 restores the all-or-nothing skip.
        int wo_ready = 0;
        const char* fp4_wo_env = getenv("SPARKINFER_MUSE_NVFP4_WO");
        bool wo_fp4_on = (!fp4_wo_env || fp4_wo_env[0] != '0');
        const char* fp4o_env = getenv("SPARKINFER_MUSE_NVFP4_OUTPUTS");
        size_t fp4_free = 0, fp4_total = 0;
        cudaMemGetInfo(&fp4_free, &fp4_total);
        (void)fp4_free; (void)fp4_total;
        // The packed continuous-batch decode reads the o and down copies too, and cb serving loads
        // at the 4096 default; the preflight below still drops them whenever the set does not fit.
        // SPARKINFER_MUSE_NVFP4_OUTPUTS_MAXSEQ=2048 restores the old bound.
        static const int outputs_maxseq = [] {
            const char* e = getenv("SPARKINFER_MUSE_NVFP4_OUTPUTS_MAXSEQ");
            return e ? atoi(e) : 4096;
        }();
        // All-or-nothing still refuses the whole set above outputs_maxseq (a 3.9 GB down copy
        // at 64k leaves the 16k prefill arena 24 MB). The prefix fill used to share that bound,
        // so the scored 64k load kept 0/52 of both legs. Allow the prefix at long max_seq and
        // leave a larger keep (below) for the arena. SPARKINFER_MUSE_NVFP4_PREFIX_LONG=0 restores
        // the skip.
        static const bool prefix_long = [] {
            const char* e = getenv("SPARKINFER_MUSE_NVFP4_PREFIX_LONG");
            return e && e[0] == '1';
        }();
        wo_fp4_on = wo_fp4_on && c.max_seq <= outputs_maxseq;
        bool down_fp4_on = c.max_seq <= outputs_maxseq;
        if (fp4o_env)
            down_fp4_on = fp4o_env[0] == '1' || fp4o_env[0] == 'd';
        // o-proj prefix at 64k ate 52/52 copies (~0.8 GB) and dropped the 16k FFN chunk
        // 4096 -> 1024. Down is worth ~4x more per byte; o stays streamed per layer from N=512.
        const bool wo_eligible = wo_fp4_on;
        const bool down_eligible = (!fp4o_env || fp4o_env[0] != '0') &&
                                   (down_fp4_on || prefix_long);
        // Cost EVERY copy that grows the footprint against the free VRAM that is actually there,
        // and drop legs in ascending order of what they are worth until the set fits. Only these
        // three grow it: gate/up convert and then release their native prefill copy, so they are
        // VRAM-neutral and must not be budgeted (budgeting them once dropped this path below main).
        //
        // This replaces a `total >= 40 GiB` card-size test, which is a proxy for the question and
        // answers it wrong on the card this model is scored on: it refuses the o-projection on
        // every 32-GB part, including the configurations where down and wo demonstrably both fit.
        // Measured on a 32-GB RTX 5090 at max_seq 2048: both legs resident is 32.1 GB, the
        // batched-prefill arena still allocates, and prefill@128 is 1.05x the down-only build.
        // A card that genuinely cannot spare it still declines here, and declines for the real
        // reason rather than for its label.
        {
            // How many sessions this deployment will actually hold. The KV pool is allocated
            // before this point and was sized for exactly that, so it already carries the number
            // rather than needing one plumbed in.
            int fp4_sessions = 1;
            if (s.kv && s.kv->block_size() > 0 && c.max_seq > 0) {
                const int bps = c.max_seq / s.kv->block_size() + 4;   // blocks one session takes
                if (bps > 0) fp4_sessions = s.kv->num_total_blocks() / bps;
                if (fp4_sessions < 1)   fp4_sessions = 1;
                if (fp4_sessions > 256) fp4_sessions = 256;
            }
            // A reserve that covers ONE session's prefill arena is what starved the runtime under
            // concurrency: every live request needs its own scratch beside these weights, so at 32
            // requests the card finished with 3 MB free -- the packed decode arena (16 MB) declined
            // on 255 of 255 steps and the batched prefill fell back to a ~24 pp token loop, 143x
            // below the 3485 pp it manages when it can allocate. Weights that exist to make prefill
            // faster are worth nothing if they leave prefill unable to run.
            // ~24 MB per live session of non-KV runtime state, measured as the rate free VRAM
            // decays across session opens; the first session is already covered by the base.
            // An explicit SPARKINFER_MUSE_NVFP4_RESERVE_MB is an operator decision and wins.
            // SPARKINFER_MUSE_NVFP4_RESERVE_CONC=0 restores the flat reserve, for an A/B.
            const size_t reserve = [&] {
                const char* e = getenv("SPARKINFER_MUSE_NVFP4_RESERVE_MB");
                if (e) { long long mb = atoll(e); return (size_t)(mb < 0 ? 0 : mb) << 20; }
                static const bool conc = [] {
                    const char* q = getenv("SPARKINFER_MUSE_NVFP4_RESERVE_CONC");
                    return !(q && q[0] == '0');
                }();
                long long mb = 384;
                if (conc) mb += (long long)(fp4_sessions - 1) * 24;
                return (size_t)mb << 20;
            }();
            const size_t want_qkvg = qkvg_fp4_on ? (size_t)c.n_layers *
                (kernels::prefill_nvfp4_data_bytes(2 * qdim_a + 2 * kvdim_a, H) +
                 kernels::prefill_nvfp4_scale_bytes_b(2 * qdim_a + 2 * kvdim_a, H)) : 0;
            const size_t want_wo = (size_t)c.n_layers *
                (kernels::prefill_nvfp4_data_bytes(H, s.qdim) +
                 kernels::prefill_nvfp4_scale_bytes_b(H, s.qdim));
            const size_t want_down = (size_t)c.n_layers *
                (kernels::prefill_nvfp4_data_bytes(H, c.moe_ffn) +
                 kernels::prefill_nvfp4_scale_bytes_b(H, c.moe_ffn));
            size_t freeb = 0, totalb = 0;
            if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) freeb = 0;
            // The KV cache is allocated before the model is constructed, so `freeb` already
            // reflects it at this session's max_seq; the reserve only has to cover the per-run
            // batched-prefill scratch arena. Drop wo before down: down is worth ~4x more.
            // Every leg also has to leave the RUNTIME its own post-load allocation: the decode
            // graph pools, the packed-decode arena and the per-pass prefill scratch. `reserve`
            // above covers the batched-prefill arena only, and at 33 sessions the rest is ~1 GB
            // more -- without counting it the budget "fitted" 1.79 GB of qkv-gate beside 0.88 GB
            // of o-proj copies and the graph instantiate then failed ten times over (290 tok/s,
            // 2 MB free at peak). Counting it drops the o-proj leg first, which is the order the
            // preflight below already intends, and keeps the qkv-gate set whole -- worth ~3x per
            // byte at packed widths: Muse cb c32 1387 -> 1548 tok/s.
            // SPARKINFER_MUSE_NVFP4_RUNTIME_MB tunes it; 0 restores main's budget.
            static const size_t runtime_margin = [] {
                const char* e = getenv("SPARKINFER_MUSE_NVFP4_RUNTIME_MB");
                const long long mb = e ? atoll(e) : 1024LL;
                return (size_t)(mb < 0 ? 0 : mb) << 20;
            }();
            auto fits = [&] {
                return freeb > want_qkvg + (wo_fp4_on ? want_wo : 0) +
                               (down_fp4_on ? want_down : 0) + reserve + runtime_margin;
            };
            fprintf(stderr, "[prefill-muse] SM120 NVFP4 preflight: %.2f GB free, %d sessions, "
                    "reserve %.2f GB\n", (double)freeb / 1e9, fp4_sessions, (double)reserve / 1e9);
            // ffn_down is a quarter of a packed decode step on Q4_K, but where both cannot be held
            // qkv-gate is the copy to keep (see the trade below). Down is 3.9 GB, and the reserve above under-counts what the
            // runtime allocates after load (packed and prefill arenas, graph pools: ~2.2 GB at 16
            // sessions against a 0.8 GB reserve), so down is admitted only with that margin on
            // top, and otherwise dropped before anything else is weighed.
            // SPARKINFER_MUSE_NVFP4_DOWN_RUNTIME_MB tunes the margin.
            if (down_fp4_on) {
                static const size_t down_runtime = [] {
                    const char* e = getenv("SPARKINFER_MUSE_NVFP4_DOWN_RUNTIME_MB");
                    long long mb = e ? atoll(e) : 2048;
                    return (size_t)(mb < 0 ? 0 : mb) << 20;
                }();
                const size_t with_down = want_down + (wo_fp4_on ? want_wo : 0) + reserve +
                                         down_runtime;
                if (freeb <= with_down) {
                    fprintf(stderr, "[prefill-muse] SM120 NVFP4 ffn_down skipped: %.1f GB free "
                            "cannot hold it with a %.1f GB runtime margin\n",
                            (double)freeb / 1e9, (double)down_runtime / 1e9);
                    down_fp4_on = false;
                } else if (qkvg_fp4_on && freeb <= with_down + want_qkvg) {
                    // Now that the packed decode drives qkv-gate from two rows, it is worth more
                    // than the down layers it displaces at every width: the partial down fill
                    // below still takes back all the layers the leftover VRAM allows (46/52 at
                    // c4, 52/52 at c2). cb c4 314.7 -> 336.0, c2 183.4 -> 190.7 tok/s.
                    // SPARKINFER_MUSE_NVFP4_QKVG_OVER_DOWN_SESSIONS=8 restores the old trade.
                    static const int qkvg_over_down = [] {
                        const char* e = getenv("SPARKINFER_MUSE_NVFP4_QKVG_OVER_DOWN_SESSIONS");
                        return e ? atoi(e) : 0;
                    }();
                    if (fp4_sessions > qkvg_over_down) {
                        fprintf(stderr, "[prefill-muse] SM120 NVFP4 ffn_down skipped: qkv-gate "
                                "is worth more at %d sessions\n", fp4_sessions);
                        down_fp4_on = false;
                    } else {
                        fprintf(stderr, "[prefill-muse] SM120 NVFP4 qkv-gate traded for ffn_down\n");
                        qkvg_fp4_on = false;
                    }
                }
            }
            if (wo_fp4_on && !fits()) {
                fprintf(stderr, "[prefill-muse] SM120 NVFP4 o-proj skipped: %.1f GB free cannot "
                        "hold qkv-gate %.1f + o %.1f + ffn_down %.1f GB + %.1f GB reserve\n",
                        (double)freeb / 1e9, (double)want_qkvg / 1e9, (double)want_wo / 1e9,
                        (double)(down_fp4_on ? want_down : 0) / 1e9, (double)reserve / 1e9);
                wo_fp4_on = false;
            }
            if (down_fp4_on && !fits()) {
                fprintf(stderr, "[prefill-muse] SM120 NVFP4 ffn_down skipped: %.1f GB free cannot "
                        "hold it plus a %.1f GB reserve\n",
                        (double)freeb / 1e9, (double)reserve / 1e9);
                down_fp4_on = false;
            }
            // Last to go, and the leg that decides high-concurrency throughput. Holding ~1.1 GB of
            // attention-projection copies on a card that then cannot give the runtime its scratch
            // is a bad trade: the batched prefill these weights accelerate drops to the token loop
            // and the packed decode forward declines every step. Dropping it is 1.61x aggregate
            // throughput at 32 concurrent requests, and costs 0.8% at 8 -- where the reserve is
            // small enough that this never fires and every copy stays resident.
            if (qkvg_fp4_on && !fits()) {
                fprintf(stderr, "[prefill-muse] SM120 NVFP4 qkv-gate skipped: %.1f GB free cannot "
                        "hold it (%.1f GB) plus a %.1f GB reserve for %d sessions\n",
                        (double)freeb / 1e9, (double)want_qkvg / 1e9, (double)reserve / 1e9,
                        fp4_sessions);
                qkvg_fp4_on = false;
            }
        }
        auto release_prefill_copy = [&](const void*& p, const void* decode) {
            if (!p || p == decode) return;
            auto it = std::find(s.owned.begin(), s.owned.end(), const_cast<void*>(p));
            if (it != s.owned.end()) { cudaFree(*it); s.owned.erase(it); }
            p = nullptr;
        };
        for (int i = 0; ok && i < c.n_layers; ++i) {
            Qwen35LayerWeights& lw = s.w.layers[i];
            const void* g = lw.prefill_gate_q ? lw.prefill_gate_q : lw.gate_q;
            const void* u = lw.prefill_up_q ? lw.prefill_up_q : lw.up_q;
            const int gt = lw.prefill_gate_q ? lw.prefill_gate_qtype : lw.gate_qtype;
            const int ut = lw.prefill_up_q ? lw.prefill_up_qtype : lw.up_qtype;
            ok = convert(g, gt, c.moe_ffn, H, &lw.gate_fp4, &lw.gate_fp4_sf) &&
                 convert(u, ut, c.moe_ffn, H, &lw.up_fp4, &lw.up_fp4_sf);
            // The attention projection group. ~1.7 GB across 52 layers, against 3.9 GB for an
            // ffn_down copy of the same kind, and it is the last dense int8 GEMM in the layer.
            // Best-effort: a layer whose four weights are not all present just keeps the int8
            // grouped path, which is what every layer does today.
            if (ok && qkvg_fp4_on) {
                const void* src[4] = { lw.wq, lw.wgate, lw.wk, lw.wv };
                const int qt[4] = { lw.wq_type, lw.wgate_type, lw.wk_type, lw.wv_type };
                const int rows[4] = { qdim_a, qdim_a, kvdim_a, kvdim_a };
                if (!convert_group(src, qt, rows, 4, H, &lw.qkvg_fp4, &lw.qkvg_fp4_sf))
                    lw.qkvg_fp4 = lw.qkvg_fp4_sf = nullptr;
            }
            // o projection [H, qdim]. Like ffn_down it has no prefill-only counterpart -- decode
            // reads lw.wo directly -- so its FP4 copy is a REAL +0.5625 B/value that nothing can
            // free. That is exactly why #820's ffn_down leg was reverted in #825: at 3.9 GB it left
            // no room for the KV cache plus the batched-prefill scratch arena at long context, and
            // prefill silently fell back to the token loop. wo is 4.8x smaller (~0.8 GB). The
            // preflight above still refuses the WHOLE set when it cannot hold every layer plus
            // the runtime; the prefix fill after the loop takes back the layers leftover VRAM
            // actually allows, the same way qkv-gate and ffn_down already do.
            if (ok && wo_fp4_on && lw.wo) {
                if (convert(lw.wo, lw.wo_type, H, s.qdim, &lw.wo_fp4, &lw.wo_fp4_sf))
                    ++wo_ready;
                else
                    lw.wo_fp4 = lw.wo_fp4_sf = nullptr;
            }
            if (ok) {
                release_prefill_copy(lw.prefill_gate_q, lw.gate_q);
                release_prefill_copy(lw.prefill_up_q, lw.up_q);
                ++ready;
                if (lw.qkvg_fp4) ++qkvg_ready;
            }
            // Output projections retain their compact GGUF tensors for decode; these FP4 copies
            // are optional and failure-isolated, so a tight-memory card can still use gate/up.
            if (ok && down_fp4_on &&
                convert(lw.down_q, lw.down_qtype, H, c.moe_ffn,
                        &lw.down_fp4, &lw.down_fp4_sf)) ++down_ready;
        }
        // PARTIAL QKV-GATE FILL. The whole set is all-or-nothing above because a deployment that
        // can hold it wants every layer; where it cannot, the layers that DO fit are still worth
        // having -- the packed decode and the batched prefill both pick this arm per layer, so a
        // prefix set [0, n) is legal exactly as it is for ffn_down.
        //
        // It fills BEFORE down because per byte it is worth about three times as much: at 32 rows
        // the Q4_K in-projections cost ~85 us a layer against ~25 us through the block-scaled
        // GEMM, for 34 MB, while an ffn_down copy buys ~94 -> 50 us for 83 MB. What it must leave
        // is the runtime's own allocation after load, which at 33 sessions is ~2.4 GB: holding the
        // full 1.79 GB set there left the graph instantiate and the prefill scratch with nothing
        // (10x "graph instantiate: out of memory", 290 tok/s), so the same keep margin ffn_down
        // uses guards this too. SPARKINFER_MUSE_NVFP4_QKVG_KEEP_MB tunes it; 0 restores main.
        static const long long qkvg_keep_mb = [] {
            const char* e = getenv("SPARKINFER_MUSE_NVFP4_QKVG_KEEP_MB");
            return e ? atoll(e) : 1024LL;
        }();
        if (ok && qkvg_eligible && qkvg_ready < c.n_layers && qkvg_keep_mb > 0) {
            const size_t per_layer =
                kernels::prefill_nvfp4_data_bytes(2 * qdim_a + 2 * kvdim_a, H) +
                kernels::prefill_nvfp4_scale_bytes_b(2 * qdim_a + 2 * kvdim_a, H);
            const size_t keep = (size_t)qkvg_keep_mb << 20;
            const size_t tmp_bytes = tmp ? tmp_elems * sizeof(bf16) : 0;
            for (int i = 0; i < c.n_layers; ++i) {
                Qwen35LayerWeights& lw = s.w.layers[i];
                if (lw.qkvg_fp4) continue;
                size_t f = 0, t = 0;
                if (cudaMemGetInfo(&f, &t) != cudaSuccess || f + tmp_bytes <= per_layer + keep) break;
                const void* src[4] = { lw.wq, lw.wgate, lw.wk, lw.wv };
                const int qt[4] = { lw.wq_type, lw.wgate_type, lw.wk_type, lw.wv_type };
                const int rows[4] = { qdim_a, qdim_a, kvdim_a, kvdim_a };
                if (!convert_group(src, qt, rows, 4, H, &lw.qkvg_fp4, &lw.qkvg_fp4_sf)) {
                    lw.qkvg_fp4 = lw.qkvg_fp4_sf = nullptr;
                    break;
                }
                ++qkvg_ready;
            }
        }

        // PARTIAL O-PROJ FILL. The whole set is all-or-nothing above because a 0.8 GB copy that
        // does not leave the runtime its graph pools instantiates into "graph instantiate: out
        // of memory" and ~10x slower packed decode (51/52 copies, 165 tok/s on a 32-GB 5090 at
        // 33 sessions). Where the whole set cannot be held, the layers that DO fit are still
        // worth having: packed decode and batched prefill both pick this arm per layer, and the
        // workspace is sized from layers[0].wo_fp4, so a prefix [0, n) is the legal set.
        //
        // It fills BEFORE down because per byte it is worth more of a packed step than another
        // ffn_down copy: at 32 rows the Q4_K o-proj (even with two MMA column groups) is still
        // the last attention projection on that path, ~15 MB a layer, while down is ~75 MB and
        // the two leftover down layers the keep below currently buys are the ones Q4_K MMA
        // already covers. What it must leave is the same runtime allocation the other prefix
        // fills guard. 736 MiB still reports enough free to convert 28/52 layers on a 32-GB
        // 5090 at 33 sessions, but those copies fragment the heap: the 333 MB prefill scratch
        // then declines with 14 MB free, graph instantiate misses, and packed c32 falls to
        // 742 tok/s (max_itl 4.7 s). 1024 MiB is the same floor ffn_down and qkv-gate already
        // use; it holds 10/52 layers and leaves a contiguous arena, 1566 tok/s. 0 restores
        // main's skip. SPARKINFER_MUSE_NVFP4_WO_KEEP_MB=736 is the previous default.
        const long long wo_keep_mb = [&] {
            const char* e = getenv("SPARKINFER_MUSE_NVFP4_WO_KEEP_MB");
            if (e) return atoll(e);
            // 16k prefill scratch is ~1042 MB; at max_seq 65536 a 1024 keep left 24 MB free
            // and the batched pass fell to the token loop. 2560 covers that arena plus the
            // graph pools. Short sessions (cb at ~832) keep 1024 -- that is the 10/52 floor.
            return c.max_seq >= 16384 ? 2560LL : 1024LL;
        }();
        if (ok && wo_eligible && !wo_fp4_on && wo_keep_mb > 0) {
            const size_t per_layer = kernels::prefill_nvfp4_data_bytes(H, s.qdim) +
                                     kernels::prefill_nvfp4_scale_bytes_b(H, s.qdim);
            const size_t keep = (size_t)wo_keep_mb << 20;
            const size_t tmp_bytes = tmp ? tmp_elems * sizeof(bf16) : 0;
            for (int i = 0; i < c.n_layers; ++i) {
                Qwen35LayerWeights& lw = s.w.layers[i];
                if (lw.wo_fp4) continue;
                size_t f = 0, t = 0;
                if (cudaMemGetInfo(&f, &t) != cudaSuccess || f + tmp_bytes <= per_layer + keep) break;
                if (!lw.wo || !convert(lw.wo, lw.wo_type, H, s.qdim, &lw.wo_fp4, &lw.wo_fp4_sf)) {
                    lw.wo_fp4 = lw.wo_fp4_sf = nullptr;
                    break;
                }
                ++wo_ready;
            }
        }

        // Where the whole down set cannot be held beside the legs that are worth more, hold as many
        // layers of it as the VRAM left after everything else actually allows. The packed decode
        // and the batched prefill both pick the down arm per layer, so a prefix set [0, n) is legal;
        // it runs last so it can never take room a qkv-gate or o copy on a later layer needed.
        // What it must leave is the runtime's own allocation after load, measured at ~576 MiB at 8
        // sessions: a 512 MiB margin starves the batched-prefill scratch and halves throughput, so
        // the default keeps 1024. SPARKINFER_MUSE_NVFP4_DOWN_KEEP_MB tunes it; 0 restores main.
        const long long down_keep_mb = [&] {
            const char* e = getenv("SPARKINFER_MUSE_NVFP4_DOWN_KEEP_MB");
            if (e) return atoll(e);
            return c.max_seq >= 16384 ? 2560LL : 1024LL;
        }();
        if (ok && down_eligible && !down_fp4_on && down_keep_mb > 0) {
            const size_t per_layer = kernels::prefill_nvfp4_data_bytes(H, c.moe_ffn) +
                                     kernels::prefill_nvfp4_scale_bytes_b(H, c.moe_ffn);
            const size_t keep = (size_t)down_keep_mb << 20;
            const size_t tmp_bytes = tmp ? tmp_elems * sizeof(bf16) : 0;
            for (int i = 0; i < c.n_layers; ++i) {
                size_t f = 0, t = 0;
                if (cudaMemGetInfo(&f, &t) != cudaSuccess || f + tmp_bytes <= per_layer + keep) break;
                Qwen35LayerWeights& lw = s.w.layers[i];
                if (!convert(lw.down_q, lw.down_qtype, H, c.moe_ffn, &lw.down_fp4, &lw.down_fp4_sf))
                    break;
                ++down_ready;
            }
        }
        if (tmp) cudaFree(tmp);
        fprintf(stderr, "[prefill-muse] SM120 NVFP4 o-proj weights ready: %d/%d layers\n", wo_ready, c.n_layers);
        fprintf(stderr, "[prefill-muse] SM120 NVFP4 FFN weights ready: %d/%d layers%s"
                        " (qkv-gate %d/%d)\n",
                ready, c.n_layers, ok ? "" : " (remaining layers use GGUF fallback)",
                qkvg_ready, c.n_layers);
        fprintf(stderr, "[prefill-muse] SM120 NVFP4 down weights ready: %d/%d layers\n",
                down_ready, c.n_layers);
    }
    // The decode shadow's view: the same layers, with each folded weight that got a ternary copy
    // swapped for it. The FFN goes over only whole -- decode's ternary SwiGLU needs all three legs
    // in one basis -- and everything else (norms, GDN state, o-proj, ssm_out) is shared as-is.
    // The dp4a GEMV's activation scratch belongs to the shadow: no scratch, no copy.
    if (bonsai_shadow && !shadow_of.empty() && !kernels::ptq1_dp_reserve()) {
        for (void* b : s.bonsai_dec_bufs) cudaFree(b);
        s.bonsai_dec_bufs.clear();
        s.bonsai_dec_head = nullptr;
        shadow_of.clear();
        fprintf(stderr, "[bonsai] decode shadow dropped: dp4a scratch alloc failed\n");
    }
    // #1154 was rejected on a long-context serving regression: with the KV pool already sized for
    // the configured --ctx (it is built before load_gguf runs, see ModelEngine::load), the
    // shadow's VRAM could leave too little free for batched prefill's own scratch arena, and every
    // prefill on a long prompt fell back to the ~30x-slower token loop -- 479s/1056s/timeout at
    // 60k/120k/200k tokens where main took 8.1s/19.6s/40.7s. A scratch-alloc failure during
    // serving now releases the shadow and retries (see prefill_batched), which recovers from this
    // regardless of the estimate below, but paying for a shadow only to immediately give it back
    // is wasted load-time work, so skip it up front when the numbers already look this tight.
    //
    // "Tight" is estimated, not exact: a byte-accurate model of the arena (FP4/int8/bf16 paths,
    // MoE vs dense, chunking) isn't worth carrying here when the retry above is the real safety
    // net. Instead this scales the one concrete data point in the rejection -- ctx=16384,
    // chunk=4096, 1731 MB held and still short -- to this run's own worst-case single pass (the
    // window, or the full --ctx when windowing is off), with a 50% margin since it is only a
    // scaled estimate.
    if (bonsai_shadow && !shadow_of.empty()) {
        const int win = prefill_window_tokens(s.kv);
        int prefill_tokens = (win > 0) ? win : s.cfg.max_seq;
        if (s.cfg.max_seq > 0 && prefill_tokens > s.cfg.max_seq) prefill_tokens = s.cfg.max_seq;
        constexpr size_t kObservedBytes = 1731ull << 20;
        constexpr int kObservedTokens = 16384;
        size_t need = (size_t)((double)kObservedBytes * prefill_tokens / kObservedTokens);
        need += need / 2;
        size_t free_b = 0, total_b = 0;
        if (prefill_tokens > 0 && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b < need) {
            for (void* b : s.bonsai_dec_bufs) cudaFree(b);
            s.bonsai_dec_bufs.clear();
            s.bonsai_dec_head = nullptr;
            shadow_of.clear();
            fprintf(stderr, "[bonsai] decode shadow skipped: %zu MB free leaves too little for a "
                            "%d-token batched prefill (estimated need ~%zu MB)\n",
                    free_b >> 20, prefill_tokens, need >> 20);
        }
    }
    if (bonsai_shadow && !shadow_of.empty()) {
        s.bonsai_dec_layers = s.w.layers;
        int n_proj = 0, n_ffn = 0;
        auto swap_in = [&](const void*& ptr, int& type) {
            const auto it = shadow_of.find(ptr);
            if (!ptr || it == shadow_of.end()) return false;
            ptr = it->second;
            type = kPtq1GgmlType;
            return true;
        };
        for (auto& d : s.bonsai_dec_layers) {
            n_proj += swap_in(d.wqkv, d.wqkv_type) + swap_in(d.wqkv_gate, d.wqkv_gate_type) +
                      swap_in(d.wq, d.wq_type) + swap_in(d.wgate, d.wgate_type) +
                      swap_in(d.wk, d.wk_type) + swap_in(d.wv, d.wv_type);
            if (shadow_of.count(d.gate_q) && shadow_of.count(d.up_q) && shadow_of.count(d.down_q)) {
                swap_in(d.gate_q, d.gate_qtype);
                swap_in(d.up_q, d.up_qtype);
                swap_in(d.down_q, d.down_qtype);
                ++n_ffn;
            }
        }
        fprintf(stderr, "[bonsai] decode shadow: %d projections, %d/%d FFNs, head %s\n",
                n_proj, n_ffn, c.n_layers, s.bonsai_dec_head ? "yes" : "no");
    }
    // decode scratch (mf_* / fa_*) is allocated in the constructor for all paths.
    return true;
}

// (dual-GPU D5) One per-card budget-audit line per rank. The on-disk term is the process
// table's own per-rank sum (the table is the single source of truth, D2), the card
// free/total is this instance's device (its ctor bound the context there once), and the
// estimate uses the G2 constants: 33,024 B/token of int8 KV -- (2,048 data + 16 scale)
// bytes per full-attn layer x 16 such layers -- plus the 154,927,104 B of GDN per-sequence
// state, all judged against a 16 GiB card (17,179,869,184 B). A failed probe (a box whose
// card is too full to even take a context) falls back to the nominal 16 GiB card, so the
// line is printable everywhere; it then says "fits" by the G2 arithmetic, which is exactly
// the D5 point: 131k context fits per card, 262k does not.
std::string tp_audit_line(int rank, int device, unsigned long long on_disk_bytes,
                          unsigned long long card_free_bytes, unsigned long long card_total_bytes,
                          int ctx) {
    static const unsigned long long kCard16GiB = 17179869184ull;
    const unsigned long long est =
        on_disk_bytes + 33024ull * (unsigned long long)ctx + 154927104ull;
    char b[256];
    std::snprintf(b, sizeof(b),
                  "[tp-audit] rank %d (device %d): on-disk %llu B, card free %llu/%llu B, "
                  "per-card est at ctx %d = %llu B -> %s 16 GiB (17,179,869,184 B)",
                  rank, device, on_disk_bytes, card_free_bytes, card_total_bytes, ctx, est,
                  est <= kCard16GiB ? "fits" : "exceeds");
    return b;
}

const Qwen35Weights& Qwen35Model::weights() const { return p_->w; }

void Qwen35Model::print_tp_audit(int ctx) const {
    const tp::Table& t = tp::get_process_table();
    if (!t.set() || t.n_ranks() < 2) return;   // table unset or tp=1: silent by design
    const Impl& s = *p_;
    cudaSetDevice(s.device);   // the ctor's one-time bind, re-asserted: the probe is per-card
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        free_b = total_b = 17179869184;   // nominal 16 GiB card when the probe itself fails
    }
    std::printf("%s\n",
                tp_audit_line(s.rank, s.device, t.rank_on_disk_bytes(s.rank), free_b, total_b, ctx)
                    .c_str());
}

// ----- HuggingFace "compressed-tensors" mixed FP8/NVFP4 checkpoint load -----
// (e.g. unsloth/Qwen3.8-27B-NVFP4). Scheme, confirmed by direct tensor inspection of the actual
// checkpoint (not assumed from the format spec alone):
//   - self_attn.{q,k,v,o}_proj, linear_attn.{in_proj_qkv,in_proj_z,out_proj}, lm_head, and
//     layers 56-63's mlp.{gate,up,down}_proj: FP8 (E4M3), one BF16 scale per output channel.
//     ".weight" (F8_E4M3) + ".weight_scale" (BF16, [out_channels,1]).
//   - every other layer's mlp.{gate,up,down}_proj (the bulk of total params): NVFP4, block_size
//     16, two-level scale. ".weight_packed" (U8, 2 values/byte) + ".weight_scale" (F8_E4M3 bytes,
//     interpreted as CUTLASS's unsigned e4m3 -- the standard NVIDIA NVFP4 export convention, not
//     literally signed e4m3 despite the safetensors dtype tag) + ".weight_global_scale" (F32
//     scalar).
//   - everything else (linear_attn's small in_proj_a/in_proj_b/norm, dt_bias, A_log, conv1d,
//     all *_norm weights): plain bf16 ".weight".
// HF tensors are [out,in] (PyTorch Linear convention), which is byte-identical to the GGUF-native
// [out,in] this runtime's GEMM/GEMV kernels already want -- so nothing below is transposed. An
// earlier revision of this loader DID relayout to [in,out], on the assumption stated in this very
// comment, and silently mis-shaped every projection in the model; see the dequant_fp8 comment for
// the three kernels whose contracts pin the layout down.
//
// FP8 tensors are either kept native (GDN, launch_gemv_fp8) or dequantized to bf16 and
// requantized to Q4_K (attn q/k/v/o, lm_head, FFN layers 56-63). NVFP4 FFN tensors
// (layers 0-55) keep the checkpoint packed bytes for prefill (CUTLASS SFB + GEMM
// alpha = 1/weight_global_scale) and a Q4_K decode copy (native GEMV is slower).
bool Qwen35Model::load_compressed_tensors(const std::string& model_dir) {
    Impl& s = *p_;
    SafeTensorsModel st;
    if (!st.open(model_dir)) {
        fprintf(stderr, "[compressed-tensors] failed to open %s\n", model_dir.c_str());
        return false;
    }
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    s.gguf = true;   // reuses the same "dense weights native, kept-quantized attn/ffn" decode shape

    // Does this layer's FFN ship as NVFP4 (-> keep the packed payload resident for batched
    // prefill) or as something else (-> dequant and requant to Q4_K like load_gguf does)? Asked
    // of the tensor names, per layer, because the two checkpoints this loader handles split it
    // differently and neither split is derivable from the architecture: compressed-tensors puts
    // layers 0-55 in NVFP4 and 56-63 in FP8, ModelOpt puts all 64 in NVFP4. A hardcoded range
    // silently mis-reads the other checkpoint -- it would look for ".weight_packed" on a tensor
    // that has none and fail the load, or worse, take the FP8 branch on bytes that are not FP8.
    auto ffn_is_nvfp4 = [&](const std::string& mlp_prefix) {
        return st.tensor(mlp_prefix + "gate_proj.weight_packed") != nullptr ||
               st.tensor(mlp_prefix + "gate_proj.weight_scale_2") != nullptr;
    };

    // bf16 upload, no transpose (embeddings, norms, and the small linear_attn gate-scalar
    // projections that the checkpoint leaves unquantized).
    auto plain_bf16 = [&](const std::string& name, long n_values) -> const void* {
        const STTensor* t = st.tensor(name);
        if (!t) { fprintf(stderr, "[compressed-tensors] missing %s\n", name.c_str()); return nullptr; }
        if (t->dtype != STDType::BF16 || t->n_values != n_values) {
            fprintf(stderr, "[compressed-tensors] %s: expected BF16[%ld], got dtype=%d n=%ld\n",
                    name.c_str(), n_values, (int)t->dtype, t->n_values);
            return nullptr;
        }
        void* d = nullptr;
        if (cudaMalloc(&d, (size_t)n_values * 2) != cudaSuccess) return nullptr;
        cudaMemcpy(d, t->data, (size_t)n_values * 2, cudaMemcpyHostToDevice);
        s.owned.push_back(d);
        return d;
    };

    // A_log -> -exp(A_log), applied on the host before upload. The checkpoint stores the raw HF
    // "A_log" parameter, but the shared GDN kernels (launch_qwen36_gdn_ar and friends, reused
    // unchanged from Qwythos/Qwen3.6) expect the pre-transformed decay coefficient -- confirmed
    // by comparing this exact tensor's values against the reference unsloth GGUF for this model
    // (whose own conversion pipeline applies this same transform): raw HF A_log runs roughly
    // [-5.6,-1.1], the GGUF's stored values are exp() of that with a sign flip, e.g. -exp(-1.0859)
    // = -0.3376, matching the GGUF's own max value bit-for-bit. Loading the raw value directly (as
    // plain_bf16 would) makes every decay gate ~exp(large-negative) instead of ~exp(small-negative)
    // -- GDN state collapses to near-zero every step, silently (no NaN/Inf) producing coherent-
    // magnitude but semantically empty hidden states. Covers all 48 linear-attention layers.
    auto load_a_log_transformed = [&](const std::string& name, long n_values) -> const void* {
        const STTensor* t = st.tensor(name);
        if (!t) { fprintf(stderr, "[compressed-tensors] missing %s\n", name.c_str()); return nullptr; }
        if (t->dtype != STDType::BF16 || t->n_values != n_values) {
            fprintf(stderr, "[compressed-tensors] %s: expected BF16[%ld], got dtype=%d n=%ld\n",
                    name.c_str(), n_values, (int)t->dtype, t->n_values);
            return nullptr;
        }
        std::vector<uint16_t> transformed((size_t)n_values);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(t->data);
        for (long i = 0; i < n_values; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            float f; memcpy(&f, &bits, sizeof(f));
            f = -expf(f);
            memcpy(&bits, &f, sizeof(bits));
            transformed[(size_t)i] = (uint16_t)(bits >> 16);
        }
        void* d = nullptr;
        if (cudaMalloc(&d, (size_t)n_values * 2) != cudaSuccess) return nullptr;
        cudaMemcpy(d, transformed.data(), (size_t)n_values * 2, cudaMemcpyHostToDevice);
        s.owned.push_back(d);
        return d;
    };

    // RMSNorm weight -> 1.0 + weight, applied on the host before upload. This checkpoint stores
    // norm weights zero-centered (values cluster around 0, e.g. [0.047, -0.063, -0.074, ...] for
    // layer 0's input_layernorm) rather than the standard one-centered convention every other
    // model in this codebase uses (values cluster around 1). Confirmed by hand-computing the
    // RMSNorm output for token "Hi" through layer 0 both ways and comparing against the reference
    // unsloth/ggml-org GGUF's own eval-callback trace: 1+weight gives sum=-63.79, matching the
    // reference's attn_norm-0 sum=-65.72 (residual difference is ordinary bf16/eps rounding);
    // plain weight gives sum=+3.28, off by both sign and two orders of magnitude. Loading the raw
    // value directly (as plain_bf16 would) multiplies every normalized activation by ~0 instead of
    // ~1 -- silently attenuating the entire signal path without producing NaN/Inf, which is why
    // every other numerical check in this bring-up looked "healthy" while generation stayed
    // incoherent. Applies to 5 of the 6 norm-weight kinds: input_layernorm,
    // post_attention_layernorm, q_norm, k_norm, and the final norm. NOT linear_attn.norm
    // (ssm_norm), which this checkpoint stores one-centered like every other model -- verified
    // per-tensor against the reference GGUF, and confirmed numerically (adding 1 there instead
    // moved layer 0's residual to 14.4 vs the reference's 8.41, away from the 8.73 it gives now).
    auto load_norm_plus1 = [&](const std::string& name, long n_values) -> const void* {
        const STTensor* t = st.tensor(name);
        if (!t) { fprintf(stderr, "[compressed-tensors] missing %s\n", name.c_str()); return nullptr; }
        if (t->dtype != STDType::BF16 || t->n_values != n_values) {
            fprintf(stderr, "[compressed-tensors] %s: expected BF16[%ld], got dtype=%d n=%ld\n",
                    name.c_str(), n_values, (int)t->dtype, t->n_values);
            return nullptr;
        }
        std::vector<uint16_t> transformed((size_t)n_values);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(t->data);
        for (long i = 0; i < n_values; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            float f; memcpy(&f, &bits, sizeof(f));
            f = 1.0f + f;
            memcpy(&bits, &f, sizeof(bits));
            transformed[(size_t)i] = (uint16_t)(bits >> 16);
        }
        void* d = nullptr;
        if (cudaMalloc(&d, (size_t)n_values * 2) != cudaSuccess) return nullptr;
        cudaMemcpy(d, transformed.data(), (size_t)n_values * 2, cudaMemcpyHostToDevice);
        s.owned.push_back(d);
        return d;
    };

    // FP8 weight [rows,cols] -> dequant -> bf16 device buffer, layout unchanged. HF stores Linear
    // weights [out_features,in_features], which is byte-identical to the GGUF-native [out,in] that
    // every consumer here wants -- launch_gemv (gemm.h: "W is [N,K] row-major ([out,in],
    // GGUF-native)"), launch_proj_requant_q4k_lloyd (its 256-element Q4_K blocks must run along the
    // input dim WITHIN one output row), and launch_prefill_nvfp4_quant_b. So no transpose: an
    // earlier [rows,cols]->[cols,rows] relayout here silently mis-shaped every projection in the
    // model (q/k/v/o, FFN, lm_head, GDN), which reads as fluent-looking garbage rather than as any
    // one tensor obviously loading wrong.
    // Caller requantizes from here (Q4_K for decode; FP8 tensors never get an NVFP4 copy).
    auto dequant_fp8 = [&](const std::string& prefix, int rows, int cols) -> void* {
        const STTensor* w = st.tensor(prefix + ".weight");
        const STTensor* sc = st.tensor(prefix + ".weight_scale");
        if (!w || !sc || w->dtype != STDType::F8_E4M3 || w->n_values != (long)rows * cols ||
            sc->n_values != rows) {
            fprintf(stderr, "[compressed-tensors] %s: missing/malformed FP8 weight or scale\n",
                    prefix.c_str());
            return nullptr;
        }
        void *wd = nullptr, *scd = nullptr, *out = nullptr;
        if (cudaMalloc(&wd, (size_t)rows * cols) != cudaSuccess) return nullptr;
        if (cudaMalloc(&scd, (size_t)rows * 2) != cudaSuccess) { cudaFree(wd); return nullptr; }
        if (cudaMalloc(&out, (size_t)rows * cols * 2) != cudaSuccess) { cudaFree(wd); cudaFree(scd); return nullptr; }
        cudaMemcpy(wd, w->data, (size_t)rows * cols, cudaMemcpyHostToDevice);
        cudaMemcpy(scd, sc->data, (size_t)rows * 2, cudaMemcpyHostToDevice);
        kernels::launch_ct_dequant_fp8(wd, scd, out, rows, cols, s.stream);
        cudaStreamSynchronize(s.stream);
        cudaFree(wd); cudaFree(scd);
        return out;   // caller owns; either requantizes from it (then frees) or pushes to s.owned
    };

    // Checkpoint FP8 kept native: [bf16 scale[rows] | e4m3 W[rows*cols]]. Decode reads it via
    // launch_gemv_fp8, which applies the same bf16(float(e4m3)*scale) rounding as dequant_fp8
    // so the GEMV matches keep_bf16 at half the GDN traffic. A second quant (Q4_K / Q8_0) on
    // these already-FP8 weights fails the PR-vs-main gate (Q4_K: top1 0.96 / KL 0.035;
    // Q8_0: top1 1.00 / KL 0.015, bar is 0.01).
    auto keep_fp8 = [&](const std::string& prefix, int rows, int cols, int& qtype) -> const void* {
        const STTensor* w = st.tensor(prefix + ".weight");
        const STTensor* sc = st.tensor(prefix + ".weight_scale");
        if (!w || !sc || w->dtype != STDType::F8_E4M3 || w->n_values != (long)rows * cols ||
            sc->n_values != rows) {
            fprintf(stderr, "[compressed-tensors] %s: missing/malformed FP8 weight or scale\n",
                    prefix.c_str());
            return nullptr;
        }
        const size_t scale_bytes = (size_t)rows * 2;
        const size_t w_bytes = (size_t)rows * (size_t)cols;
        void* packed = nullptr;
        if (cudaMalloc(&packed, scale_bytes + w_bytes) != cudaSuccess) return nullptr;
        cudaMemcpy(packed, sc->data, scale_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(packed) + scale_bytes, w->data, w_bytes, cudaMemcpyHostToDevice);
        s.owned.push_back(packed);
        qtype = kernels::SI_QTYPE_FP8;
        return packed;
    };

    // bf16 [out,in] source -> resident Q4_K (decode path). Frees the source.
    auto requant_q4k = [&](void* bf16_src, long n_values, int& qtype) -> const void* {
        if (!bf16_src || n_values % 256 != 0) { if (bf16_src) cudaFree(bf16_src); return nullptr; }
        void* q4 = nullptr;
        if (cudaMalloc(&q4, (size_t)(n_values / 256) * 144) != cudaSuccess) { cudaFree(bf16_src); return nullptr; }
        kernels::launch_proj_requant_q4k_lloyd(bf16_src, q4, n_values, s.stream);
        cudaStreamSynchronize(s.stream);
        cudaFree(bf16_src);
        s.owned.push_back(q4);
        qtype = 12;
        return q4;
    };

    // Checkpoint NVFP4 kept native: [256 B header with f32 global_scale |
    // ue4m3 scale[rows*cols/16] | packed u8[rows*cols/2]]. Decode reads it via
    // launch_gemv_nvfp4. Prefill reuses the packed region as CUTLASS B and
    // scatters the UE4M3 scales into SFB; the tensor-wide global_scale becomes
    // GEMM alpha (1/global), matching launch_ct_dequant_nvfp4's divide. The
    // header is 256 B so the packed region stays TMA-aligned.
    // Resolves one NVFP4 weight's three tensors across the two export conventions that ship in
    // the wild. They differ in NAMING and, silently, in the DIRECTION of the tensor-wide scale:
    //
    //   llm-compressor / "compressed-tensors"  (unsloth/Qwen3.8-27B-NVFP4)
    //     ".weight_packed" (U8) + ".weight_scale" (UE4M3) + ".weight_global_scale" (F32)
    //     global = (6*448)/amax  ->  W = q * block_scale / global
    //   NVIDIA ModelOpt                        (gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
    //     ".weight" (U8)        + ".weight_scale" (UE4M3) + ".weight_scale_2" (F32)
    //     weight_scale_2 = amax/(6*448)  ->  W = q * block_scale * weight_scale_2
    //
    // The two constants are exact reciprocals, so reading one checkpoint with the other's
    // convention scales every weight by amax^2/2688^2 -- no NaN, no shape error, just a model
    // that emits confident garbage. Everything downstream here (launch_ct_dequant_nvfp4's divide,
    // the CUTLASS GEMM's alpha = 1/global) is written against the compressed-tensors form, so
    // ModelOpt's multiplier is inverted once, here, and never again.
    //
    // Which one a checkpoint uses is decided by the tensor names present, not by
    // quantization_config.quant_method: the config key is metadata a re-uploader can copy
    // without the bytes matching, whereas ".weight_scale_2" existing IS the ModelOpt layout.
    // Direction confirmed numerically against the actual file rather than from the format docs --
    // layer 3's k_proj carries weight_scale_2 = 1.19e-4, which under the multiply reading implies
    // a weight amax of 6*448*1.19e-4 = 0.32 (ordinary for a projection) and under the divide
    // reading 2.25e7 (impossible).
    struct NvFp4Src { const void* packed; const void* group; float global; };
    auto nvfp4_src = [&](const std::string& prefix, int rows, int cols, NvFp4Src& out) -> bool {
        const STTensor* gs = st.tensor(prefix + ".weight_scale");
        const STTensor* wp = st.tensor(prefix + ".weight_packed");
        const STTensor* glob = wp ? st.tensor(prefix + ".weight_global_scale") : nullptr;
        bool modelopt = false;
        if (!wp) {   // ModelOpt stores the packed nibbles under the plain ".weight" name
            wp = st.tensor(prefix + ".weight");
            glob = st.tensor(prefix + ".weight_scale_2");
            modelopt = true;
        }
        // Not an NVFP4 weight at all (bf16 or FP8) -- a quiet miss, not a malformed checkpoint.
        if (!wp || !gs || !glob || wp->dtype != STDType::U8) return false;
        if (wp->n_values != (long)rows * cols / 2 ||
            gs->n_values != (long)rows * cols / 16 || glob->n_values != 1) {
            fprintf(stderr, "[compressed-tensors] %s: malformed NVFP4 tensors "
                    "(packed=%ld want %ld, group=%ld want %ld, global=%ld want 1)\n",
                    prefix.c_str(), wp->n_values, (long)rows * cols / 2,
                    gs->n_values, (long)rows * cols / 16, glob->n_values);
            return false;
        }
        float g = 1.f;
        memcpy(&g, glob->data, sizeof(float));
        if (!(g > 0.f)) {
            fprintf(stderr, "[compressed-tensors] %s: non-positive global scale %g\n",
                    prefix.c_str(), g);
            return false;
        }
        out.packed = wp->data;
        out.group = gs->data;
        out.global = modelopt ? (1.f / g) : g;
        return true;
    };

    // Run the DECODE FFN on the checkpoint's own NVFP4 weights. ON BY DEFAULT since 2026-08-21:
    // this engine targets Blackwell and the checkpoint ships NVFP4, so requantizing it to Q4_K to
    // decode it throws away the format the hardware exists to run. SPARKINFER_QWEN38_DECODE_NVFP4=0
    // restores the Q4_K requantization.
    //
    // It is not free, and the cost is larger than the weight error alone suggests. Measured on the
    // pinned RTX 5090 at ctx=4096, reps=3, both arms back-to-back on the same build (see the
    // gate_nv/up_nv/down_nv comment in qwen35.h):
    //
    //                     AR tok/s   DSpark tok/s   tau      lossless
    //     Q4_K requant     90.23       114.39       1.6623     yes
    //     NVFP4 native     82.23        95.07       1.5059     yes
    //
    // AR gives up 8.9% because the Q4_K path is dp4a int8 MMVQ and heavily tuned while
    // launch_gemv_nvfp4 dequantizes to float. Speculative decode gives up 16.9%, not 8.9%, because
    // the two effects compound: speedup ~= tau/(verify + draft), so a slower target forward AND a
    // lower acceptance both push the same direction. Closing that is the dp4a/fusion work on
    // launch_gemv_nvfp4 -- the accuracy is already where it should be, only the kernel is behind.
    static const bool kDecodeNvfp4 = [] {
        const char* e = getenv("SPARKINFER_QWEN38_DECODE_NVFP4");
        return !(e && e[0] == '0');
    }();
    auto keep_nvfp4 = [&](const std::string& prefix, int rows, int cols,
                          const void** fp4, const void** fp4_sf, float& fp4_alpha) -> const void* {
        NvFp4Src src{};
        if (!nvfp4_src(prefix, rows, cols, src)) {
            fprintf(stderr, "[compressed-tensors] %s: missing/malformed NVFP4 tensors\n",
                    prefix.c_str());
            return nullptr;
        }
        const float global_scale = src.global;
        const size_t scale_bytes = (size_t)rows * cols / 16;
        const size_t packed_bytes = (size_t)rows * cols / 2;
        const size_t hdr = (size_t)kernels::SI_NVFP4_HDR;
        void* payload = nullptr;
        if (cudaMalloc(&payload, hdr + scale_bytes + packed_bytes) != cudaSuccess) return nullptr;
        cudaMemset(payload, 0, hdr);
        cudaMemcpy(payload, &global_scale, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(payload) + hdr, src.group, scale_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(payload) + hdr + scale_bytes, src.packed, packed_bytes,
                   cudaMemcpyHostToDevice);
        // SPARKINFER_QWEN38_PREFILL_NVFP4=0 drops the checkpoint-native NVFP4 copies once they
        // have been consumed to build the Q4_K decode weights. In the default configuration they
        // exist ONLY to make batched prefill faster -- decode reads gate_q/up_q/down_q unless
        // kDecodeNvfp4 is set, in which case these payloads ARE the decode weights and this knob
        // does not free them (see the exclusive branch at the w.gate_nv assignment below).
        // They are not small: at
        // 4.5 bits per weight over a 64-layer, 17408-wide dense FFN that is ~9.6 GB held resident
        // purely for prefill throughput, on top of the Q4_K copy's own 9.6 GB.
        //
        // That second residency is what puts this model at 29.7 GB on a 32.6 GB card at ctx=16k,
        // which is why the checkpoint's own headline -- 262144-token context on a 5090 -- is out
        // of reach here even though the weights nominally fit in ~19 GB. Turning these off trades
        // prefill throughput for the KV headroom that long context actually needs.
        //
        // Registered in s.owned ONLY when kept: otherwise the payload is transient, still needed
        // as the SOURCE for q4k_from_nvfp4 below and then freed by the caller. Keeping it out of
        // s.owned is what makes that early free safe (nothing double-frees at teardown).
        static const bool keep_prefill_fp4 = [] {
            const char* e = getenv("SPARKINFER_QWEN38_PREFILL_NVFP4");
            return !(e && e[0] == '0');
        }();
        // Exactly one owner even when the same payload serves both prefill and native decode.
        if (keep_prefill_fp4 || kDecodeNvfp4) s.owned.push_back(payload);
        fp4_alpha = (global_scale != 0.f) ? (1.f / global_scale) : 1.f;
        const void* packed = static_cast<char*>(payload) + hdr + scale_bytes;
        if (keep_prefill_fp4 && kernels::prefill_nvfp4_supported(128, rows, cols)) {
            void* sf = nullptr;
            const size_t sf_bytes = kernels::prefill_nvfp4_scale_bytes_b(rows, cols);
            if (sf_bytes && cudaMalloc(&sf, sf_bytes) == cudaSuccess &&
                kernels::launch_ct_nvfp4_pack_sfb(static_cast<char*>(payload) + hdr, sf,
                                                  rows, cols, s.stream) &&
                cudaStreamSynchronize(s.stream) == cudaSuccess) {
                s.owned.push_back(sf);
                *fp4 = packed;
                *fp4_sf = sf;
            } else if (sf) {
                cudaFree(sf);
            }
        }
        return payload;
    };

    // Decode stays on the existing Q4_K MMVQ path (native NVFP4 GEMV is ~0.65x).
    // Dequant from the payload we already uploaded so the host tensors are not reread.
    auto q4k_from_nvfp4 = [&](const void* payload, int rows, int cols, int& qtype) -> const void* {
        if (!payload) return nullptr;
        const size_t hdr = (size_t)kernels::SI_NVFP4_HDR;
        const size_t scale_bytes = (size_t)rows * cols / 16;
        float gs = 1.f;
        cudaMemcpy(&gs, payload, 4, cudaMemcpyDeviceToHost);
        void* out = nullptr;
        if (cudaMalloc(&out, (size_t)rows * cols * 2) != cudaSuccess) return nullptr;
        kernels::launch_ct_dequant_nvfp4(
            static_cast<const char*>(payload) + hdr + scale_bytes,
            static_cast<const char*>(payload) + hdr, gs, out, rows, cols, s.stream);
        return requant_q4k(out, (long)rows * cols, qtype);
    };

    // NVFP4 -> bf16 device buffer, transient. Same dequant as q4k_from_nvfp4, but the packed
    // source is uploaded to scratch and released here instead of staying resident: this is for
    // weights with no batched-prefill NVFP4 consumer, where holding the payload would cost VRAM
    // that nothing reads. Contract matches dequant_fp8's deliberately (caller owns the returned
    // buffer and either requantizes from it or pushes it to s.owned), so callers can dispatch on
    // storage format without also branching on ownership.
    auto dequant_nvfp4 = [&](const std::string& prefix, int rows, int cols) -> void* {
        NvFp4Src src{};
        if (!nvfp4_src(prefix, rows, cols, src)) {
            fprintf(stderr, "[compressed-tensors] %s: missing/malformed NVFP4 tensors\n",
                    prefix.c_str());
            return nullptr;
        }
        const size_t scale_bytes = (size_t)rows * cols / 16;
        const size_t packed_bytes = (size_t)rows * cols / 2;
        void *pd = nullptr, *gd = nullptr, *out = nullptr;
        if (cudaMalloc(&pd, packed_bytes) != cudaSuccess) return nullptr;
        if (cudaMalloc(&gd, scale_bytes) != cudaSuccess) { cudaFree(pd); return nullptr; }
        if (cudaMalloc(&out, (size_t)rows * cols * 2) != cudaSuccess) {
            cudaFree(pd); cudaFree(gd); return nullptr;
        }
        cudaMemcpy(pd, src.packed, packed_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gd, src.group, scale_bytes, cudaMemcpyHostToDevice);
        kernels::launch_ct_dequant_nvfp4(pd, gd, src.global, out, rows, cols, s.stream);
        // CHECKED, because an unchecked failure here is invisible in the worst possible way: the
        // output buffer is left as cudaMalloc returned it and the caller requantizes that into a
        // perfectly well-formed weight of zeros. A zero lm_head yields a uniform distribution over
        // the vocabulary, so the model emits token 0 forever while every layer's activations look
        // healthy -- which is exactly how the gridDim.y overflow fixed above stayed hidden.
        cudaError_t de = cudaGetLastError();
        if (de == cudaSuccess) de = cudaStreamSynchronize(s.stream);
        cudaFree(pd); cudaFree(gd);
        if (de != cudaSuccess) {
            fprintf(stderr, "[compressed-tensors] %s: NVFP4 dequant failed (%s) for [%d,%d]\n",
                    prefix.c_str(), cudaGetErrorString(de), rows, cols);
            cudaFree(out);
            return nullptr;
        }
        return out;
    };

    // Whichever of the three storage formats this Linear actually uses -> bf16, so the callers
    // that only ever want a requant source (attention projections, lm_head) do not have to know.
    // Decided per tensor from the bytes present rather than from quantization_config's
    // targets/ignore rules or from hardcoded layer ranges: the two Qwen3.8-27B NVFP4 checkpoints
    // this loader handles disagree on which tensors are quantized at all (compressed-tensors puts
    // attention/GDN/lm_head in FP8 and the last 8 FFN layers too; ModelOpt quantizes all 400
    // Linears to NVFP4 and leaves lm_head plain bf16), and a checkpoint's own tensor list is the
    // only description of it that cannot be stale.
    auto dequant_any = [&](const std::string& prefix, int rows, int cols) -> void* {
        NvFp4Src probe{};
        if (nvfp4_src(prefix, rows, cols, probe)) return dequant_nvfp4(prefix, rows, cols);
        const STTensor* w = st.tensor(prefix + ".weight");
        if (!w) {
            fprintf(stderr, "[compressed-tensors] missing %s.weight\n", prefix.c_str());
            return nullptr;
        }
        if (w->dtype == STDType::F8_E4M3) return dequant_fp8(prefix, rows, cols);
        if (w->dtype != STDType::BF16 || w->n_values != (long)rows * cols) {
            fprintf(stderr, "[compressed-tensors] %s.weight: expected BF16[%ld] or a quantized "
                    "form, got dtype=%d n=%ld\n",
                    prefix.c_str(), (long)rows * cols, (int)w->dtype, w->n_values);
            return nullptr;
        }
        void* d = nullptr;
        if (cudaMalloc(&d, (size_t)rows * cols * 2) != cudaSuccess) return nullptr;
        cudaMemcpy(d, w->data, (size_t)rows * cols * 2, cudaMemcpyHostToDevice);
        return d;   // caller owns, same contract as dequant_fp8 / dequant_nvfp4
    };

    // Checkpoint NVFP4 kept native for decode (SI_QTYPE_NVFP4 -> launch_gemv_nvfp4). This is
    // keep_fp8's policy above, carried across the format change: the ModelOpt checkpoint stores
    // the GDN projections as NVFP4 where compressed-tensors stored them as FP8, and requantizing
    // an already-4-bit weight lands its values between Q4_K's own grid points, on 48 of 64 layers.
    //
    // Measured both ways on the RTX 5090 box -- 101 teacher-forced positions of eval_text.txt,
    // scored against the same independent reference (the llama.cpp-derived Qwen3.8-27B-Q4_K_M
    // GGUF of this model), and benched on the same build:
    //
    //                     top1     KL      PPL  | decode@128  decode@16k  prefill@16k   VRAM
    //   GDN native NVFP4  0.851   0.247   3.235 |  78.4 t/s    67.3 t/s    9894 t/s   29.3 GB
    //   GDN -> Q4_K       0.822   0.300   3.352 |  96.7 t/s    80.3 t/s    9977 t/s   29.3 GB
    //
    // The requant buys 19-23% decode and costs real accuracy -- and costs it for nothing in VRAM,
    // because Q4_K (144 B per 256 weights) and this NVFP4 payload (4 bits plus one UE4M3 per 16)
    // are both exactly 4.5 bits per weight. Comparing the two dumps against each other isolates
    // the choice from the rest of the pipeline: KL 0.050 nats, with 8% of positions changing
    // their argmax purely from the second quantization. Native is the default because it is the
    // only one of the two that is free, and because accuracy at 4 bits is this checkpoint's
    // entire proposition. (The FFN below still takes a Q4_K decode copy: there the NVFP4 payload
    // stays resident anyway for batched prefill, so the copy costs no VRAM either.)
    // SPARKINFER_Q38_GDN_NVFP4=0 takes the speed end of that trade instead.
    static const bool gdn_keep_nvfp4 = [] {
        const char* e = getenv("SPARKINFER_Q38_GDN_NVFP4");
        return !(e && e[0] == '0');
    }();
    // Batched prefill's native-NVFP4 arm for the GDN projections. The packed nibbles are already
    // resident for decode, so the only new bytes are the CUTLASS SFB scale copy -- 1 byte per 16
    // weights, i.e. 1/9th of what the payload it describes already costs. Without it batched
    // prefill has no NVFP4 B operand and proj() expands every GDN weight NVFP4 -> bf16 -> int8 on
    // every pass. SPARKINFER_Q38_GDN_PREFILL_NVFP4=0 skips the SFB entirely (A/B, and the VRAM
    // escape hatch for a card that would rather spend those bytes on KV).
    static const bool gdn_prefill_fp4 = [] {
        const char* e = getenv("SPARKINFER_Q38_GDN_PREFILL_NVFP4");
        return !(e && e[0] == '0');
    }();
    auto keep_nvfp4_native = [&](const std::string& prefix, int rows, int cols, int& qtype,
                                 const void** fp4 = nullptr, const void** fp4_sf = nullptr,
                                 float* fp4_alpha = nullptr) -> const void* {
        NvFp4Src src{};
        if (!nvfp4_src(prefix, rows, cols, src)) return nullptr;
        if (!gdn_keep_nvfp4) return requant_q4k(dequant_nvfp4(prefix, rows, cols),
                                                (long)rows * cols, qtype);
        const size_t scale_bytes = (size_t)rows * cols / 16;
        const size_t packed_bytes = (size_t)rows * cols / 2;
        const size_t hdr = (size_t)kernels::SI_NVFP4_HDR;
        void* payload = nullptr;
        if (cudaMalloc(&payload, hdr + scale_bytes + packed_bytes) != cudaSuccess) return nullptr;
        cudaMemset(payload, 0, hdr);
        cudaMemcpy(payload, &src.global, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(payload) + hdr, src.group, scale_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(payload) + hdr + scale_bytes, src.packed, packed_bytes,
                   cudaMemcpyHostToDevice);
        s.owned.push_back(payload);
        qtype = kernels::SI_QTYPE_NVFP4;
        if (fp4 && gdn_prefill_fp4 && kernels::prefill_nvfp4_supported(128, rows, cols)) {
            void* sf = nullptr;
            const size_t sf_bytes = kernels::prefill_nvfp4_scale_bytes_b(rows, cols);
            if (sf_bytes && cudaMalloc(&sf, sf_bytes) == cudaSuccess &&
                kernels::launch_ct_nvfp4_pack_sfb(static_cast<char*>(payload) + hdr, sf,
                                                  rows, cols, s.stream) &&
                cudaStreamSynchronize(s.stream) == cudaSuccess) {
                s.owned.push_back(sf);
                *fp4 = static_cast<char*>(payload) + hdr + scale_bytes;
                *fp4_sf = sf;
                // src.global is already the divisor launch_ct_dequant_nvfp4 divides by, so the
                // GEMM's alpha is its reciprocal -- keep_nvfp4's convention for the FFN, verbatim.
                *fp4_alpha = (src.global != 0.f) ? (1.f / src.global) : 1.f;
            } else if (sf) {
                cudaFree(sf);
            }
        }
        return payload;
    };

    // One Linear kept in whatever native form this checkpoint ships it in: FP8 stays FP8, NVFP4
    // stays NVFP4, and anything else falls back to a Q4_K fit from bf16. Used for the GDN
    // projections, where both shipped checkpoints deliberately avoid a second quantization.
    auto keep_native = [&](const std::string& prefix, int rows, int cols, int& qtype,
                           const void** fp4 = nullptr, const void** fp4_sf = nullptr,
                           float* fp4_alpha = nullptr) -> const void* {
        NvFp4Src probe{};
        if (nvfp4_src(prefix, rows, cols, probe))
            return keep_nvfp4_native(prefix, rows, cols, qtype, fp4, fp4_sf, fp4_alpha);
        const STTensor* w = st.tensor(prefix + ".weight");
        if (w && w->dtype == STDType::F8_E4M3) return keep_fp8(prefix, rows, cols, qtype);
        return requant_q4k(dequant_any(prefix, rows, cols), (long)rows * cols, qtype);
    };

    // ---------------------------------------------------------------------
    // (A) The tensor-parallel slice path (dual-GPU Wave 3). The process-wide
    // placement table -- built and published by ModelEngine::load before any
    // loader runs, and DEGENERATE (every name whole on device 0) at the tp=1
    // default -- decides per named tensor whether this rank skips it, owns it
    // whole, or owns a sub-block. tp_pick() below calls the whole path (the
    // lambdas above, byte-identical) or the matching slice lambda; a whole
    // decision never executes a slice lambda, so a degenerate table makes
    // every call site below the pre-TP read.
    // ---------------------------------------------------------------------
    const int tp_rank = s.rank;

    auto tp_pick = [&](const std::string& name, auto&& whole_fn, auto&& slice_fn) {
        using T = decltype(whole_fn());
        TpSlice sl;
        switch (tp_decide(name, tp_rank, &sl)) {
            case TpDec::Skip: {
                T v{};
                return v;
            }
            case TpDec::Whole:
                return whole_fn();
            default:
                return slice_fn(sl);
        }
    };

    // The owned element count of the split axis (1 range for a plain split, up
    // to three for the GDN qkv/conv channel axis) and the reduced [rows',cols']
    // extents of a sliced 2-D tensor.
    auto tp_owned_count = [&](const TpSlice& sl) {
        long o = 0;
        for (const tp::Range& r : sl.ranges) o += (long)r.len;
        return o;
    };
    auto tp_owned_rc = [&](int rows, int cols, const TpSlice& sl) -> long {
        const long o = tp_owned_count(sl);
        if (sl.axis == tp::Axis::Cols) return (long)rows * o;
        return o * (long)cols;   // Rows or OneD: the split dim is the first
    };

    // The two whole-path bf16 element transforms, as staging transforms for the
    // slice path (A_log -> -exp(A_log); norm weight -> 1 + weight).
    std::function<void(uint16_t*)> xform_a_log = [](uint16_t* p) {
        uint32_t bits = (uint32_t)*p << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        f = -expf(f);
        memcpy(&bits, &f, sizeof(bits));
        *p = (uint16_t)(bits >> 16);
    };
    std::function<void(uint16_t*)> xform_norm_plus1 = [](uint16_t* p) {
        uint32_t bits = (uint32_t)*p << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        f = 1.0f + f;
        memcpy(&bits, &f, sizeof(bits));
        *p = (uint16_t)(bits >> 16);
    };

    // bf16 source (embeddings, norms, dt/A_log, conv1d, in_proj_a/b): gather the
    // owned sub-block and upload it as a reduced [rows', cols'] (or 1-D) bf16
    // buffer in the source's own layout. d0 = the Rows-axis extent, d1 = the
    // Cols-axis extent (1 for a 1-D tensor); rows_fast is false for this HF
    // loader (d1 is the fast axis).
    auto ct_bf16_slice = [&](const std::string& name, long d0, long d1, bool rows_fast,
                             const TpSlice& sl,
                             const std::function<void(uint16_t*)>* xform) -> const void* {
        const STTensor* t = st.tensor(name);
        if (!t || t->dtype != STDType::BF16 || t->n_values != d0 * d1) {
            fprintf(stderr, "[compressed-tensors] %s: missing/malformed bf16 source (slice)\n",
                    name.c_str());
            return nullptr;
        }
        return tpl_read2d(static_cast<const char*>(t->data), d0, d1, 2, rows_fast, sl.axis,
                          sl.ranges, s.owned, xform);
    };

    // The owned sub-blocks of an on-disk NVFP4 weight's two stream tensors,
    // rescaled to whole stream units. The packed nibbles are a [rows, cols/2]
    // u8 matrix and the 16-group scale a [rows, cols/16] u8 matrix in the same
    // HF row-major layout, so a /2- or /16-rescaled range set gathers exactly
    // the owned sub-blocks (the 27B split extents are 16-aligned, so both land
    // on whole units; the result is a valid reduced [rows', cols'] tensor in
    // the source layout).
    auto nvfp4_units = [&](const TpSlice& sl, long div) {
        // Only a Cols split runs along the packed axis; a Rows split selects whole rows of both
        // streams, so its ranges are row indices and must not be rescaled (doing so gathered 1/2
        // of the packed rows and 1/16 of the scale rows of every row-split NVFP4 weight).
        if (sl.axis != tp::Axis::Cols) return sl.ranges;
        std::vector<tp::Range> u;
        for (const tp::Range& r : sl.ranges) u.push_back({r.begin / div, r.len / div});
        return u;
    };

    // NVFP4 source: gather the owned sub-blocks of the packed and scale streams
    // and rebuild the SAME [hdr | scale | packed] payload the whole path uploads
    // (plus the SFB copy when the prefill GEMM can take the shape). Ownership
    // mirrors the whole path: always_own (the GDN-native payload is a decode
    // weight and is always kept) or the keep_prefill_fp4/kDecodeNvfp4 policy
    // (the FFN payload, freed by the caller when no SFB was built).
    auto ct_nvfp4_slice = [&](const std::string& prefix, int rows, int cols,
                              const TpSlice& sl, bool always_own, bool sfb_on,
                              const void** fp4, const void** fp4_sf, float* fp4_alpha)
        -> const void* {
        static const bool keep_prefill_fp4 = [] {
            const char* e = getenv("SPARKINFER_QWEN38_PREFILL_NVFP4");
            return !(e && e[0] == '0');
        }();
        NvFp4Src full{};
        if (!nvfp4_src(prefix, rows, cols, full)) {
            fprintf(stderr, "[compressed-tensors] %s: missing/malformed NVFP4 (slice)\n",
                    prefix.c_str());
            return nullptr;
        }
        const long o = tp_owned_count(sl);
        const long rows2 = (sl.axis == tp::Axis::Cols) ? rows : o;
        const long cols2 = (sl.axis == tp::Axis::Cols) ? o : cols;
        std::vector<char> packed_s, scale_s;
        tpl_gather2d(static_cast<const char*>(full.packed), rows, cols / 2, 1, false, sl.axis,
                     nvfp4_units(sl, 2), nullptr, packed_s);
        tpl_gather2d(static_cast<const char*>(full.group), rows, cols / 16, 1, false, sl.axis,
                     nvfp4_units(sl, 16), nullptr, scale_s);
        if (packed_s.empty() || scale_s.empty()) return nullptr;
        const size_t hdr = (size_t)kernels::SI_NVFP4_HDR;
        void* payload = nullptr;
        if (cudaMalloc(&payload, hdr + packed_s.size() + scale_s.size()) != cudaSuccess)
            return nullptr;
        cudaMemset(payload, 0, hdr);
        cudaMemcpy(payload, &full.global, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(payload) + hdr, scale_s.data(), scale_s.size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(payload) + hdr + scale_s.size(), packed_s.data(),
                   packed_s.size(), cudaMemcpyHostToDevice);
        if (always_own || keep_prefill_fp4 || kDecodeNvfp4) s.owned.push_back(payload);
        if (sfb_on && fp4 && kernels::prefill_nvfp4_supported(128, (int)rows2, (int)cols2)) {
            void* sf = nullptr;
            const size_t sf_bytes = kernels::prefill_nvfp4_scale_bytes_b((int)rows2, (int)cols2);
            if (sf_bytes && cudaMalloc(&sf, sf_bytes) == cudaSuccess &&
                kernels::launch_ct_nvfp4_pack_sfb(static_cast<char*>(payload) + hdr, sf,
                                                  (int)rows2, (int)cols2, s.stream) &&
                cudaStreamSynchronize(s.stream) == cudaSuccess) {
                s.owned.push_back(sf);
                *fp4 = static_cast<char*>(payload) + hdr + scale_s.size();
                *fp4_sf = sf;
                if (fp4_alpha) *fp4_alpha = (full.global != 0.f) ? (1.f / full.global) : 1.f;
            } else if (sf) {
                cudaFree(sf);
            }
        }
        return payload;
    };

    // FP8 source: the [bf16 scale[rows] | e4m3 weight[rows*cols]] payload with
    // the owned sub-blocks gathered per stream. The per-row scale is 1-D over
    // rows, so a Cols split keeps every scale row (each row keeps its scalar)
    // while a Rows split keeps the same row window as the weight.
    auto ct_fp8_slice = [&](const std::string& prefix, int rows, int cols,
                            const TpSlice& sl, int& qtype) -> const void* {
        const STTensor* w = st.tensor(prefix + ".weight");
        const STTensor* sc = st.tensor(prefix + ".weight_scale");
        if (!w || !sc || w->dtype != STDType::F8_E4M3 || sc->dtype != STDType::BF16 ||
            w->n_values != (long)rows * cols || sc->n_values != rows) {
            fprintf(stderr, "[compressed-tensors] %s: missing/malformed FP8 (slice)\n",
                    prefix.c_str());
            return nullptr;
        }
        const std::vector<tp::Range> su =
            (sl.axis == tp::Axis::Cols) ? std::vector<tp::Range>{{0, (size_t)rows}} : sl.ranges;
        std::vector<char> ws, scs;
        tpl_gather2d(static_cast<const char*>(w->data), rows, cols, 1, false, sl.axis, sl.ranges,
                     nullptr, ws);
        tpl_gather2d(static_cast<const char*>(sc->data), rows, 1, 2, false, tp::Axis::OneD, su,
                     nullptr, scs);
        if (ws.empty()) return nullptr;
        void* packed = nullptr;
        if (cudaMalloc(&packed, scs.size() + ws.size()) != cudaSuccess) return nullptr;
        cudaMemcpy(packed, scs.data(), scs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(static_cast<char*>(packed) + scs.size(), ws.data(), ws.size(),
                   cudaMemcpyHostToDevice);
        s.owned.push_back(packed);
        qtype = kernels::SI_QTYPE_FP8;
        return packed;
    };

    // dequant_any's slice twin: whichever of the three storage formats this
    // Linear uses -> a reduced bf16 [rows', cols'] device buffer, the exact
    // requant source the whole path produces (the Q4_K 256-blocks run along
    // the in-dim within each owned row, as in the whole path).
    auto ct_dequant_slice = [&](const std::string& prefix, int rows, int cols,
                                const TpSlice& sl) -> void* {
        NvFp4Src probe{};
        if (nvfp4_src(prefix, rows, cols, probe)) {
            const long o = tp_owned_count(sl);
            const long rows2 = (sl.axis == tp::Axis::Cols) ? rows : o;
            const long cols2 = (sl.axis == tp::Axis::Cols) ? o : cols;
            std::vector<char> ps, gs;
            tpl_gather2d(static_cast<const char*>(probe.packed), rows, cols / 2, 1, false,
                          sl.axis, nvfp4_units(sl, 2), nullptr, ps);
            tpl_gather2d(static_cast<const char*>(probe.group), rows, cols / 16, 1, false,
                          sl.axis, nvfp4_units(sl, 16), nullptr, gs);
            if (ps.empty() || gs.empty()) return nullptr;
            void *pd = nullptr, *gd = nullptr, *out = nullptr;
            if (cudaMalloc(&pd, ps.size()) != cudaSuccess) return nullptr;
            if (cudaMalloc(&gd, gs.size()) != cudaSuccess) { cudaFree(pd); return nullptr; }
            if (cudaMalloc(&out, (size_t)rows2 * (size_t)cols2 * 2) != cudaSuccess) {
                cudaFree(pd); cudaFree(gd); return nullptr;
            }
            cudaMemcpy(pd, ps.data(), ps.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(gd, gs.data(), gs.size(), cudaMemcpyHostToDevice);
            kernels::launch_ct_dequant_nvfp4(pd, gd, probe.global, out, (int)rows2, (int)cols2,
                                             s.stream);
            cudaError_t de = cudaGetLastError();
            if (de == cudaSuccess) de = cudaStreamSynchronize(s.stream);
            cudaFree(pd); cudaFree(gd);
            if (de != cudaSuccess) {
                fprintf(stderr,
                        "[compressed-tensors] %s: NVFP4 dequant failed (slice) (%s) for [%ld,%ld]\n",
                        prefix.c_str(), cudaGetErrorString(de), rows2, cols2);
                cudaFree(out);
                return nullptr;
            }
            return out;
        }
        const STTensor* w = st.tensor(prefix + ".weight");
        if (!w) {
            fprintf(stderr, "[compressed-tensors] missing %s.weight (slice)\n", prefix.c_str());
            return nullptr;
        }
        if (w->dtype == STDType::F8_E4M3) {
            const STTensor* sc = st.tensor(prefix + ".weight_scale");
            if (!sc || w->n_values != (long)rows * cols || sc->n_values != rows) {
                fprintf(stderr, "[compressed-tensors] %s: missing/malformed FP8 (slice)\n",
                        prefix.c_str());
                return nullptr;
            }
            const std::vector<tp::Range> su =
                (sl.axis == tp::Axis::Cols) ? std::vector<tp::Range>{{0, (size_t)rows}} : sl.ranges;
            std::vector<char> ws, scs;
            tpl_gather2d(static_cast<const char*>(w->data), rows, cols, 1, false, sl.axis,
                          sl.ranges, nullptr, ws);
            tpl_gather2d(static_cast<const char*>(sc->data), rows, 1, 2, false, tp::Axis::OneD,
                         su, nullptr, scs);
            if (ws.empty()) return nullptr;
            const long o = tp_owned_count(sl);
            const long rows2 = (sl.axis == tp::Axis::Cols) ? rows : o;
            const long cols2 = (sl.axis == tp::Axis::Cols) ? o : cols;
            void *wd = nullptr, *scd = nullptr, *out = nullptr;
            if (cudaMalloc(&wd, ws.size()) != cudaSuccess) return nullptr;
            if (cudaMalloc(&scd, scs.size()) != cudaSuccess) { cudaFree(wd); return nullptr; }
            if (cudaMalloc(&out, (size_t)rows2 * (size_t)cols2 * 2) != cudaSuccess) {
                cudaFree(wd); cudaFree(scd); return nullptr;
            }
            cudaMemcpy(wd, ws.data(), ws.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(scd, scs.data(), scs.size(), cudaMemcpyHostToDevice);
            kernels::launch_ct_dequant_fp8(wd, scd, out, (int)rows2, (int)cols2, s.stream);
            cudaStreamSynchronize(s.stream);
            cudaFree(wd); cudaFree(scd);
            return out;
        }
        if (w->dtype != STDType::BF16 || w->n_values != (long)rows * cols) {
            fprintf(stderr, "[compressed-tensors] %s.weight: unexpected form (slice)\n",
                    prefix.c_str());
            return nullptr;
        }
        std::vector<char> stage;
        tpl_gather2d(static_cast<const char*>(w->data), rows, cols, 2, false, sl.axis, sl.ranges,
                     nullptr, stage);
        if (stage.empty()) return nullptr;
        void* d = nullptr;
        if (cudaMalloc(&d, stage.size()) != cudaSuccess) return nullptr;
        cudaMemcpy(d, stage.data(), stage.size(), cudaMemcpyHostToDevice);
        return d;
    };

    // keep_native's slice twin: the same three-way routing (NVFP4 native
    // payload / FP8 native payload / Q4_K requant of the reduced dequant), the
    // owned bytes only. gdn_keep_nvfp4 is honored exactly as the whole path's
    // keep_nvfp4_native does.
    auto ct_keep_native_slice = [&](const std::string& prefix, int rows, int cols,
                                    const TpSlice& sl, int& qtype,
                                    const void** fp4, const void** fp4_sf, float* fp4_alpha)
        -> const void* {
        NvFp4Src probe{};
        if (nvfp4_src(prefix, rows, cols, probe)) {
            if (!gdn_keep_nvfp4) {
                void* bf = ct_dequant_slice(prefix, rows, cols, sl);
                return requant_q4k(bf, tp_owned_rc(rows, cols, sl), qtype);
            }
            // Tag the rank blob as NVFP4, exactly as the whole path (keep_nvfp4_native) does:
            // left at its default 0, every consumer read the SI_NVFP4 payload as dense bf16.
            const void* pay = ct_nvfp4_slice(prefix, rows, cols, sl, /*always_own=*/true,
                                             gdn_prefill_fp4, fp4, fp4_sf, fp4_alpha);
            if (pay) qtype = kernels::SI_QTYPE_NVFP4;
            return pay;
        }
        const STTensor* w = st.tensor(prefix + ".weight");
        if (w && w->dtype == STDType::F8_E4M3) return ct_fp8_slice(prefix, rows, cols, sl, qtype);
        void* bf = ct_dequant_slice(prefix, rows, cols, sl);
        return requant_q4k(bf, tp_owned_rc(rows, cols, sl), qtype);
    };

    // embed_tokens: HF embedding tables are already [vocab,hidden] (not a Linear layer), no
    // transpose needed -- matches load_gguf()'s own dense("token_embd.weight", false).
    // The rank's owned window is a failure only when this rank owns the name (a Skip is
    // expected to stay null); at tp=1 tp_decide is Whole for everything, so this is the
    // plain "missing" check the loader always had.
    auto tp_missing = [&](const std::string& nm, const void* p) {
        return !p && tp_decide(nm, tp_rank, nullptr) != TpDec::Skip;
    };

    s.w.embed_tokens = tp_pick("model.language_model.embed_tokens.weight",
        [&]{ return plain_bf16("model.language_model.embed_tokens.weight", (long)c.vocab * H); },
        [&](const TpSlice& sl) {
            return ct_bf16_slice("model.language_model.embed_tokens.weight", c.vocab, H, false,
                                 sl, nullptr);
        });
    s.w.final_norm = tp_pick("model.language_model.norm.weight",
        [&]{ return load_norm_plus1("model.language_model.norm.weight", H); },
        [&](const TpSlice& sl) {
            return ct_bf16_slice("model.language_model.norm.weight", H, 1, false, sl,
                                  &xform_norm_plus1);
        });
    // lm_head: FP8 in the compressed-tensors checkpoint, plain bf16 in the ModelOpt one (which
    // lists it under quantization_config.ignore). Either way it ends up Q4_K, as in load_gguf().
    // Split on the vocab (row) axis: the Q4_K requant runs over this rank's [vocab', H] window.
    // tp table key is the full tensor name ("lm_head.weight", tp_layout Conv::Hf); the bare
    // "lm_head" prefix is unknown to the table and resolves to Whole -- both ranks then held the
    // ENTIRE head and computed only its first V/2 rows (the upper vocab half never scored).
    s.w.lm_head = tp_pick("lm_head.weight",
        [&]{ return requant_q4k(dequant_any("lm_head", c.vocab, H), (long)c.vocab * H,
                                  s.w.lm_head_type); },
        [&](const TpSlice& sl) {
            void* bf = ct_dequant_slice("lm_head", c.vocab, H, sl);
            return requant_q4k(bf, tp_owned_rc(c.vocab, H, sl), s.w.lm_head_type);
        });
    if (tp_missing("model.language_model.embed_tokens.weight", s.w.embed_tokens) ||
        tp_missing("model.language_model.norm.weight", s.w.final_norm) ||
        tp_missing("lm_head.weight", s.w.lm_head))
        return false;
    // ...and, when the checkpoint ships the head as NVFP4, keep its own bytes as well, in the
    // block-scaled GEMM's operand layout. Every other big tensor already went this way (the FFN
    // and, since the attention block above, q/k/v/o); the head was the last one still served only
    // from a Q4_K refit of the checkpoint's own values.
    //
    // BESIDE the Q4_K copy, not instead of it: AR decode and the speculative verify keep reading
    // lm_head byte for byte, so acceptance, losslessness and the accuracy gate are untouched.
    // keep_nvfp4 aliases the GEMM's data operand onto the payload it just uploaded, so the only
    // NEW residency beyond that payload is the re-laid-out scale copy -- 0.0625 B/weight, 79 MB
    // at this head's 248320x5120.
    //
    // Gated on free VRAM with a reserve, decided here and once: this model already peaks near the
    // card at long context, and the batched-prefill scratch arena is allocated later and per run.
    // Spending the arena's headroom on a decode-only operand would trade a no-regression floor
    // (prefill at 32k drops to the token loop when the arena cannot be carved) for a win that only
    // exists at packed widths >= 16 -- which is a trade in the wrong direction, so when the
    // reserve is not clear this simply stays off and the head keeps the path it had.
    // SPARKINFER_Q38_HEAD_NVFP4=0 disables it; _RESERVE_MB tunes the reserve.
    static const bool head_fp4_on = [] {
        const char* e = getenv("SPARKINFER_Q38_HEAD_NVFP4");
        return !(e && e[0] == '0');
    }();
    if (head_fp4_on) {
        TpSlice head_sl{};
        const TpDec head_dec = tp_decide("lm_head.weight", tp_rank, &head_sl);
        // The rank's owned vocab window (the full vocab at tp=1 / whole). A Skip is not
        // expected (the head splits on every rank that owns any of it) but is tolerated.
        const long head_rows =
            (head_dec == TpDec::Slice && head_sl.axis == tp::Axis::Rows)
                ? tp_owned_count(head_sl) : c.vocab;
        if (head_dec != TpDec::Skip) {
            NvFp4Src head_probe{};
            if (nvfp4_src("lm_head", c.vocab, H, head_probe)) {
            const size_t need = (size_t)head_rows * H * 5 / 8 +                   // payload
                                kernels::prefill_nvfp4_scale_bytes_b((int)head_rows, H); // SFB copy
            static const size_t reserve_mb = [] {
                const char* e = getenv("SPARKINFER_Q38_HEAD_NVFP4_RESERVE_MB");
                const long v = e ? atol(e) : 3072;
                return (size_t)(v < 0 ? 0 : v);
            }();
            size_t hfree = 0, htotal = 0;
            cudaMemGetInfo(&hfree, &htotal);
            if (hfree > need + reserve_mb * 1024ull * 1024ull) {
                const size_t owned_before = s.owned.size();
                if (head_dec == TpDec::Slice)
                    ct_nvfp4_slice("lm_head", c.vocab, H, head_sl, /*always_own=*/false, true,
                                   &s.w.lm_head_fp4, &s.w.lm_head_fp4_sf,
                                   &s.w.lm_head_fp4_alpha);
                else
                    keep_nvfp4("lm_head", c.vocab, H, &s.w.lm_head_fp4, &s.w.lm_head_fp4_sf,
                               s.w.lm_head_fp4_alpha);
                // Take the two buffers OUT of s.owned and hold them here instead. They are the
                // only weights in this model that can be given back at runtime (see
                // release_lm_head_fp4), and s.owned is freed wholesale at teardown -- an entry
                // that a release has already freed would be a double free.
                if (s.w.lm_head_fp4 && s.owned.size() == owned_before + 2) {
                    s.lm_head_fp4_sf_buf = s.owned.back(); s.owned.pop_back();
                    s.lm_head_fp4_payload = s.owned.back(); s.owned.pop_back();
                }
            }
            if (!s.w.lm_head_fp4)
                fprintf(stderr, "[compressed-tensors] NVFP4 lm_head not kept "
                        "(free %.2f GB, need %.2f GB + %zu MB reserve)\n",
                        hfree / 1073741824.0, need / 1073741824.0, reserve_mb);
            else
                fprintf(stderr, "[compressed-tensors] NVFP4 lm_head kept for wide packed decode "
                        "(%.2f GB)\n", need / 1073741824.0);
        }
        }
    }

    s.w.layers.resize(c.n_layers);
    int gu_ready = 0;
    for (int i = 0; i < c.n_layers; i++) {
        const std::string b = "model.language_model.layers." + std::to_string(i) + ".";
        Qwen35LayerWeights& w = s.w.layers[i];
        w.linear_attn = is_linear_layer(c, i);
        w.input_norm = load_norm_plus1(b + "input_layernorm.weight", H);
        w.post_attn_norm = load_norm_plus1(b + "post_attention_layernorm.weight", H);

        if (w.linear_attn) {
            const std::string lb = b + "linear_attn.";
            // Native checkpoint format, whichever it is (FP8 or NVFP4) -- not a second Q4_K/Q8_0
            // fit: same bf16 rounding as keep_bf16, half the GDN traffic. See keep_native above.
            w.wqkv = tp_pick(lb + "in_proj_qkv",
                [&]{ return keep_native(lb + "in_proj_qkv", s.linear_qkvdim, H, w.wqkv_type,
                                        &w.gdn_qkv_fp4, &w.gdn_qkv_fp4_sf,
                                        &w.gdn_qkv_fp4_alpha); },
                [&](const TpSlice& sl) {
                    return ct_keep_native_slice(lb + "in_proj_qkv", s.linear_qkvdim, H, sl,
                                                w.wqkv_type, &w.gdn_qkv_fp4, &w.gdn_qkv_fp4_sf,
                                                &w.gdn_qkv_fp4_alpha);
                });
            w.wqkv_gate = tp_pick(lb + "in_proj_z",
                [&]{ return keep_native(lb + "in_proj_z",
                                          c.linear_v_heads * c.linear_head_dim, H,
                                          w.wqkv_gate_type,
                                          &w.gdn_z_fp4, &w.gdn_z_fp4_sf, &w.gdn_z_fp4_alpha); },
                [&](const TpSlice& sl) {
                    return ct_keep_native_slice(lb + "in_proj_z",
                                                c.linear_v_heads * c.linear_head_dim, H, sl,
                                                w.wqkv_gate_type, &w.gdn_z_fp4, &w.gdn_z_fp4_sf,
                                                &w.gdn_z_fp4_alpha);
                });
            w.ssm_out = tp_pick(lb + "out_proj",
                [&]{ return keep_native(lb + "out_proj", H, s.linear_vdim, w.ssm_out_type,
                                         &w.gdn_out_fp4, &w.gdn_out_fp4_sf,
                                         &w.gdn_out_fp4_alpha); },
                [&](const TpSlice& sl) {
                    return ct_keep_native_slice(lb + "out_proj", H, s.linear_vdim, sl,
                                                w.ssm_out_type, &w.gdn_out_fp4, &w.gdn_out_fp4_sf,
                                                &w.gdn_out_fp4_alpha);
                });
            // Small, checkpoint-unquantized tensors -- plain bf16, NO transpose. conv1d's raw HF
            // layout [qkvdim,1,conv_kernel] (=[qkvdim,conv_kernel] squeezed) already matches
            // conv_split_kernel's own indexing (conv_w[d*conv_kernel+t], d=channel, t=tap) --
            // transposing it here was wrong (same bug class as keep_bf16 below: see its comment).
            // in_proj_a/in_proj_b are ordinary HF Linear weights [out,in]=[v_heads,H], which is
            // exactly what proj_xn's plain-bf16 launch_gemv fallback wants (gemm.h: "[N,K]
            // row-major ([out,in], GGUF-native)") -- transposing them to [H,v_heads] was wrong
            // for the same reason. Root-caused via the same byte-level cross-check against
            // llama.cpp that found the keep_bf16 transpose bug: alpha/beta projections summed to
            // 10.8/3.2 in this runtime vs. 108.6/79.7 in the reference trace, and neither tensor
            // goes through conv at all, ruling out the conv1d weight as their cause and pointing
            // straight at their own [out,in]-vs-[in,out] mismatch.
            // a/dt are per-v-head scalars consumed at the GLOBAL v offset (+v0 shift at call sites),
            // so both ranks hold all rows; the degenerate table already routed to these readers -> tp=1 byte-identical.
            w.ssm_dt = plain_bf16(lb + "dt_bias", c.linear_v_heads);
            w.ssm_a  = load_a_log_transformed(lb + "A_log", c.linear_v_heads);
            w.ssm_norm = tp_pick(lb + "norm.weight",
                [&]{ return plain_bf16(lb + "norm.weight", c.linear_head_dim); },
                [&](const TpSlice& sl) {
                    return ct_bf16_slice(lb + "norm.weight", c.linear_head_dim, 1, false, sl,
                                         nullptr);
                });
            // conv1d [qkvdim, kernel]: the same three q|k|v channel windows as in_proj_qkv
            // (both land on the HF slow axis), contiguous per section.
            w.ssm_conv = tp_pick(lb + "conv1d.weight",
                [&]{ return plain_bf16(lb + "conv1d.weight",
                                         (long)s.linear_qkvdim * c.linear_conv_kernel); },
                [&](const TpSlice& sl) {
                    return ct_bf16_slice(lb + "conv1d.weight", s.linear_qkvdim,
                                         c.linear_conv_kernel, false, sl, nullptr);
                });
            w.ssm_alpha = tp_pick(lb + "in_proj_a.weight",
                [&]{ return plain_bf16(lb + "in_proj_a.weight", (long)c.linear_v_heads * H); },
                [&](const TpSlice& sl) {
                    return ct_bf16_slice(lb + "in_proj_a.weight", c.linear_v_heads, H, false, sl,
                                         nullptr);
                });
            w.ssm_beta = tp_pick(lb + "in_proj_b.weight",
                [&]{ return plain_bf16(lb + "in_proj_b.weight", (long)c.linear_v_heads * H); },
                [&](const TpSlice& sl) {
                    return ct_bf16_slice(lb + "in_proj_b.weight", c.linear_v_heads, H, false, sl,
                                         nullptr);
                });
            if (tp_missing(lb + "in_proj_qkv", w.wqkv) ||
                tp_missing(lb + "in_proj_z", w.wqkv_gate) ||
                tp_missing(lb + "out_proj", w.ssm_out) ||
                tp_missing(lb + "dt_bias", w.ssm_dt) || tp_missing(lb + "A_log", w.ssm_a) ||
                tp_missing(lb + "norm.weight", w.ssm_norm) ||
                tp_missing(lb + "conv1d.weight", w.ssm_conv) ||
                tp_missing(lb + "in_proj_a.weight", w.ssm_alpha) ||
                tp_missing(lb + "in_proj_b.weight", w.ssm_beta)) {
                fprintf(stderr, "[compressed-tensors] layer %d: linear-attn weights missing\n", i);
                return false;
            }
        } else {
            w.q_has_gate = c.hybrid;
            const std::string ab = b + "self_attn.";
            const int q_out = w.q_has_gate ? s.qdim * 2 : s.qdim;
            // Same treatment the GDN projections get: keep the checkpoint's own NVFP4 bytes rather
            // than fitting a second quantization to them. Batched prefill then feeds the packed
            // nibbles to the SM120 block-scaled GEMM instead of streaming a Q4_K copy through a
            // dp4a GEMM at ~30% of peak, and decode reads them through launch_gemv_nvfp4.
            //
            // This REPLACES the Q4_K copy rather than sitting beside it, which is what makes it
            // affordable: the payload is 0.5625 B/weight against Q4_K's 0.5625, so only the
            // CUTLASS SFB scale copy (0.0625 B/weight, ~105 MB over 16 layers) is new residency.
            // Holding both would have cost ~1.05 GB, and this model already peaks at 31.9 GB of a
            // 32.6 GB card at max_ctx=16384 -- there is 676 MB of headroom, so the both-copies
            // form would have shrunk the 16k prefill arena, which is a no-regression floor.
            // SPARKINFER_Q38_ATTN_NVFP4=0 restores the Q4_K requant (A/B in ONE binary).
            static const bool attn_native_fp4 = [] {
                const char* e = getenv("SPARKINFER_Q38_ATTN_NVFP4");
                return !(e && e[0] == '0');
            }();
            // Per TENSOR, and only when it is genuinely NVFP4. keep_native would otherwise route
            // an FP8 attention tensor to keep_fp8, which is exactly what the OTHER shipped
            // Qwen3.8 checkpoint (unsloth: NVFP4 FFN + FP8 attention/GDN) stores -- that moved its
            // decode off Q4_K MMVQ and cost it 3.4%, a model this change has no business touching.
            // Probing per tensor keeps every non-NVFP4 attention weight on the exact path it had.
            auto attn_w = [&](const std::string& nm, int rows, int cols, int& qt,
                              const void** fp4, const void** fp4_sf, float* alpha) -> const void* {
                NvFp4Src probe{};
                if (attn_native_fp4 && nvfp4_src(ab + nm, rows, cols, probe))
                    return keep_native(ab + nm, rows, cols, qt, fp4, fp4_sf, alpha);
                return requant_q4k(dequant_any(ab + nm, rows, cols), (long)rows * cols, qt);
            };
            // attn_w's slice twin: same three-way routing (native when the
            // checkpoint ships NVFP4 AND the knob allows it, Q4_K requant of the
            // reduced dequant otherwise), owned sub-blocks only.
            auto attn_w_slice = [&](const std::string& nm, int rows, int cols, int& qt,
                                    const void** fp4, const void** fp4_sf, float* alpha,
                                    const TpSlice& sl) -> const void* {
                NvFp4Src probe{};
                if (attn_native_fp4 && nvfp4_src(ab + nm, rows, cols, probe))
                    return ct_keep_native_slice(ab + nm, rows, cols, sl, qt, fp4, fp4_sf, alpha);
                void* bf = ct_dequant_slice(ab + nm, rows, cols, sl);
                return requant_q4k(bf, tp_owned_rc(rows, cols, sl), qt);
            };
            w.wq = tp_pick(ab + "q_proj",
                [&]{ return attn_w("q_proj", q_out, H, w.wq_type,
                                    &w.wq_fp4, &w.wq_fp4_sf, &w.wq_fp4_alpha); },
                [&](const TpSlice& sl) { return attn_w_slice("q_proj", q_out, H, w.wq_type,
                                                               &w.wq_fp4, &w.wq_fp4_sf,
                                                               &w.wq_fp4_alpha, sl); });
            w.wk = tp_pick(ab + "k_proj",
                [&]{ return attn_w("k_proj", s.kvdim, H, w.wk_type,
                                    &w.wk_fp4, &w.wk_fp4_sf, &w.wk_fp4_alpha); },
                [&](const TpSlice& sl) { return attn_w_slice("k_proj", s.kvdim, H, w.wk_type,
                                                               &w.wk_fp4, &w.wk_fp4_sf,
                                                               &w.wk_fp4_alpha, sl); });
            w.wv = tp_pick(ab + "v_proj",
                [&]{ return attn_w("v_proj", s.kvdim, H, w.wv_type,
                                    &w.wv_fp4, &w.wv_fp4_sf, &w.wv_fp4_alpha); },
                [&](const TpSlice& sl) { return attn_w_slice("v_proj", s.kvdim, H, w.wv_type,
                                                               &w.wv_fp4, &w.wv_fp4_sf,
                                                               &w.wv_fp4_alpha, sl); });
            w.wo = tp_pick(ab + "o_proj",
                [&]{ return attn_w("o_proj", H, s.qdim, w.wo_type,
                                    &w.wo_fp4, &w.wo_fp4_sf, &w.wo_fp4_alpha); },
                [&](const TpSlice& sl) { return attn_w_slice("o_proj", H, s.qdim, w.wo_type,
                                                               &w.wo_fp4, &w.wo_fp4_sf,
                                                               &w.wo_fp4_alpha, sl); });
            w.q_norm = tp_pick(ab + "q_norm.weight",
                [&]{ return load_norm_plus1(ab + "q_norm.weight", c.head_dim); },
                [&](const TpSlice& sl) {
                    return ct_bf16_slice(ab + "q_norm.weight", c.head_dim, 1, false, sl,
                                         &xform_norm_plus1);
                });
            w.k_norm = tp_pick(ab + "k_norm.weight",
                [&]{ return load_norm_plus1(ab + "k_norm.weight", c.head_dim); },
                [&](const TpSlice& sl) {
                    return ct_bf16_slice(ab + "k_norm.weight", c.head_dim, 1, false, sl,
                                         &xform_norm_plus1);
                });
            if (tp_missing(ab + "q_proj", w.wq) || tp_missing(ab + "k_proj", w.wk) ||
                tp_missing(ab + "v_proj", w.wv) || tp_missing(ab + "o_proj", w.wo) ||
                tp_missing(ab + "q_norm.weight", w.q_norm) ||
                tp_missing(ab + "k_norm.weight", w.k_norm)) {
                fprintf(stderr, "[compressed-tensors] layer %d: attention weights missing\n", i);
                return false;
            }
        }

        const std::string mb = b + "mlp.";
        if (!ffn_is_nvfp4(mb)) {
            w.gate_q = tp_pick(mb + "gate_proj",
                [&]{ return requant_q4k(dequant_any(mb + "gate_proj", c.moe_ffn, H),
                                         (long)c.moe_ffn * H, w.gate_qtype); },
                [&](const TpSlice& sl) {
                    void* bf = ct_dequant_slice(mb + "gate_proj", c.moe_ffn, H, sl);
                    return requant_q4k(bf, tp_owned_rc(c.moe_ffn, H, sl), w.gate_qtype);
                });
            w.up_q = tp_pick(mb + "up_proj",
                [&]{ return requant_q4k(dequant_any(mb + "up_proj", c.moe_ffn, H),
                                         (long)c.moe_ffn * H, w.up_qtype); },
                [&](const TpSlice& sl) {
                    void* bf = ct_dequant_slice(mb + "up_proj", c.moe_ffn, H, sl);
                    return requant_q4k(bf, tp_owned_rc(c.moe_ffn, H, sl), w.up_qtype);
                });
            w.down_q = tp_pick(mb + "down_proj",
                [&]{ return requant_q4k(dequant_any(mb + "down_proj", H, c.moe_ffn),
                                         (long)H * c.moe_ffn, w.down_qtype); },
                [&](const TpSlice& sl) {
                    void* bf = ct_dequant_slice(mb + "down_proj", H, c.moe_ffn, sl);
                    return requant_q4k(bf, tp_owned_rc(H, c.moe_ffn, sl), w.down_qtype);
                });
        } else {
            // NVFP4 FFN: the whole path keeps the checkpoint payload; the slice path
            // gathers this rank's sub-block of each stream and rebuilds the same
            // payload shape (same ownership policy -- the caller below frees it
            // when no SFB was built, exactly as in the whole path).
            auto keep_ffn = [&](const std::string& nm, int rows, int cols,
                                const void** fp4, const void** fp4_sf, float* alpha) {
                TpSlice sl;
                switch (tp_decide(nm, tp_rank, &sl)) {
                    case TpDec::Skip:
                        return (const void*)nullptr;
                    case TpDec::Whole:
                        return keep_nvfp4(nm, rows, cols, fp4, fp4_sf, *alpha);
                    default:
                        return ct_nvfp4_slice(nm, rows, cols, sl, /*always_own=*/false, true,
                                               fp4, fp4_sf, alpha);
                }
            };
            auto q4k_from = [&](const std::string& nm, const void* pay, int rows, int cols,
                                 int& qt) {
                TpSlice sl;
                if (tp_decide(nm, tp_rank, &sl) == TpDec::Slice) {
                    const long o = tp_owned_count(sl);
                    if (sl.axis == tp::Axis::Cols) return q4k_from_nvfp4(pay, rows, (int)o, qt);
                    return q4k_from_nvfp4(pay, (int)o, cols, qt);
                }
                return q4k_from_nvfp4(pay, rows, cols, qt);
            };
            const void* g_pay =
                keep_ffn(mb + "gate_proj", c.moe_ffn, H, &w.gate_fp4, &w.gate_fp4_sf,
                         &w.gate_fp4_alpha);
            const void* u_pay =
                keep_ffn(mb + "up_proj", c.moe_ffn, H, &w.up_fp4, &w.up_fp4_sf,
                         &w.up_fp4_alpha);
            const void* d_pay =
                keep_ffn(mb + "down_proj", H, c.moe_ffn, &w.down_fp4, &w.down_fp4_sf,
                         &w.down_fp4_alpha);
            // The default path keeps the checkpoint's own payloads for DECODE, so
            // the FFN runs the numerics the checkpoint actually ships instead of a Q4_K
            // requantization of them (8.25% mean relative weight error -- see qwen35.h).
            //
            // This is selected before loading and cannot change without reloading the model, so
            // building Q4_K copies as well buys no usable runtime A/B: the dispatch flag is a
            // process-static value too. It only requantizes every FFN at startup and holds another
            // ~9.6 GB for weights no kernel can reach. Keep exactly one decode representation.
            if (kDecodeNvfp4) {
                w.gate_nv = g_pay; w.up_nv = u_pay; w.down_nv = d_pay;
            } else {
                w.gate_q = q4k_from(mb + "gate_proj", g_pay, c.moe_ffn, H, w.gate_qtype);
                w.up_q = q4k_from(mb + "up_proj", u_pay, c.moe_ffn, H, w.up_qtype);
                w.down_q = q4k_from(mb + "down_proj", d_pay, H, c.moe_ffn, w.down_qtype);
            }
            if (!kDecodeNvfp4 && !w.gate_fp4) {
                // Prefill copies disabled: the payloads were deliberately not registered in
                // s.owned (see keep_nvfp4), the Q4_K decode copies above are built, so release the
                // NVFP4 source now rather than at teardown. Keyed on gate_fp4 being unset, which
                // is exactly the condition under which keep_nvfp4 skipped the push.
                cudaFree(const_cast<void*>(g_pay));
                cudaFree(const_cast<void*>(u_pay));
                cudaFree(const_cast<void*>(d_pay));
            }
            if (w.gate_fp4 && w.up_fp4) ++gu_ready;
        }
        const bool native_ffn = w.gate_nv && w.up_nv && w.down_nv;
        const bool q4_ffn = w.gate_q && w.up_q && w.down_q;
        if (!native_ffn && !q4_ffn) {
            fprintf(stderr, "[compressed-tensors] layer %d: FFN is neither native NVFP4 nor Q4_K\n", i);
            return false;
        }
    }
    fprintf(stderr, "[compressed-tensors] loaded %d layers, native NVFP4 prefill FFN %d/%d, "
            "decode FFN %s\n", c.n_layers, gu_ready, c.n_layers,
            kDecodeNvfp4 ? "NVFP4" : "Q4_K");

    // Same fused Q4_K-in-GEMM row scales Muse uses, for whichever tensors ended up Q4_K after the
    // per-tensor routing above -- full-attn q/k/v/o always, plus any FFN layer with no native
    // NVFP4 prefill operand (all 8 FP8 MLP layers in the compressed-tensors checkpoint; none in
    // the ModelOpt one, which is NVFP4 end to end). Lets prefill decode Q4_K inside the GEMM
    // instead of materializing int8 weights every layer. SPARKINFER_Q38_PREFILL_QB=0 skips (A/B).
    {
        const char* e = getenv("SPARKINFER_Q38_PREFILL_QB");
        if (e && e[0] == '0') return true;
        auto fusable = [](int t) { return t == 12 || t == 13 || t == 14; };
        const int qd = c.n_q_heads * c.head_dim;
        const int q_out = c.hybrid ? qd * 2 : qd;
        const int kd = c.n_kv_heads * c.head_dim;
        const int ff = c.moe_ffn;
        // This rank's owned extents per layer (the table's windows are rank-level, so every
        // layer reserves the same reduced sizes; the FULL extents at tp=1, where the table is
        // degenerate and the total is the pre-TP one to the float).
        long qo = q_out, kvo = kd, ffo = ff, hdims;
        tp_owned_dims("model.language_model.layers.0.self_attn.q_proj", tp_rank, q_out, H, qo, hdims);
        tp_owned_dims("model.language_model.layers.0.self_attn.k_proj", tp_rank, kd, H, kvo, hdims);
        tp_owned_dims("model.language_model.layers.0.mlp.gate_proj", tp_rank, ff, H, ffo, hdims);
        const size_t per_layer = (size_t)qo + (size_t)kvo * 2 + (size_t)H + (size_t)ffo * 2 + (size_t)H;
        const size_t total = per_layer * (size_t)c.n_layers;
        const size_t tmp_bytes = 64u << 20;
        signed char* tmp = nullptr;
        bool ok = cudaMalloc(&s.muse_rs, total * sizeof(float)) == cudaSuccess;
        ok = ok && cudaMalloc(&tmp, tmp_bytes) == cudaSuccess;
        auto fill = [&](int qtype, const void* src, float* dst, size_t rows, int cols) -> bool {
            const int blk = (qtype == 12) ? 144 : (qtype == 13) ? 176 : 210;
            const size_t rb = (size_t)(cols >> 8) * (size_t)blk;
            const size_t chunk = tmp_bytes / (size_t)cols;
            for (size_t r0 = 0; r0 < rows; r0 += chunk) {
                const size_t nr = (rows - r0 < chunk) ? (rows - r0) : chunk;
                if (!kernels::launch_gguf_dequant_rows_i8(
                        qtype, (const char*)src + r0 * rb, tmp, dst + r0, (int)nr, cols, s.stream)) {
                    fprintf(stderr, "[compressed-tensors] dequant_rows_i8 failed "
                            "(qtype=%d rows=%zu cols=%ld cuda=%s)\n", qtype, nr, (long)cols,
                            cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
            }
            return true;
        };
        for (int i = 0; ok && i < c.n_layers; i++) {
            Qwen35LayerWeights& lw = s.w.layers[i];
            float* base = s.muse_rs + (size_t)i * per_layer;
            size_t off = 0;
            auto place = [&](const void* W, int wt, const float** rs, int rows, int cols) {
                float* dst = base + off; off += (size_t)rows;
                if (ok && W && fusable(wt) && fill(wt, W, dst, (size_t)rows, cols)) *rs = dst;
            };
            place(lw.wq,     lw.wq_type,     &lw.wq_rs,    q_out, H);
            place(lw.wk,     lw.wk_type,     &lw.wk_rs,    kd,    H);
            place(lw.wv,     lw.wv_type,     &lw.wv_rs,    kd,    H);
            place(lw.wo,     lw.wo_type,     &lw.wo_rs,    H,     qd);
            // A row scale is only worth computing where prefill will actually read Q4_K. Keyed on
            // whether the native NVFP4 operand got built rather than on a layer index: that is
            // the condition prefill itself branches on, so the two cannot drift, and it covers
            // the case where keep_nvfp4 declined a shape that prefill_nvfp4_supported rejected.
            if (!lw.gate_fp4) {
                place(lw.gate_q, lw.gate_qtype,  &lw.gate_rs,  ff, H);
                place(lw.up_q,   lw.up_qtype,    &lw.up_rs,    ff, H);
                place(lw.down_q, lw.down_qtype,  &lw.down_rs,  H,  ff);
            } else {
                off += (size_t)ff * 2 + (size_t)H;
            }
        }
        if (ok) ok = cudaStreamSynchronize(s.stream) == cudaSuccess;
        if (tmp) cudaFree(tmp);
        if (!ok) {
            cudaFree(s.muse_rs); s.muse_rs = nullptr;
            for (int i = 0; i < c.n_layers; i++) {
                Qwen35LayerWeights& lw = s.w.layers[i];
                lw.wq_rs = lw.wk_rs = lw.wv_rs = lw.wo_rs = nullptr;
                lw.gate_rs = lw.up_rs = lw.down_rs = nullptr;
            }
            fprintf(stderr, "[compressed-tensors] Q4_K row-scale precompute unavailable "
                            "-> int8 materialize path\n");
        } else {
            fprintf(stderr, "[compressed-tensors] Q4_K prefill row scales ready (%.0f MB)\n",
                    (double)(total * sizeof(float)) / (1024.0 * 1024.0));
        }
    }
    return true;
}

} // namespace sparkinfer
