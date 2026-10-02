#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen_config.h"
#include "sparkinfer/moe/engine.h"

namespace sparkinfer {

// Largest packed continuous-batch decode row count. Mirrors the verify graph tiers in
// qwen35_prefill.cpp -- one captured graph per row count, so this bounds how many are kept.
inline constexpr int kQwen35MaxPackedRows = 32;

class ThermalGovernor;   // optional decode-time thermal pacing (thermal_governor.h)
class BridgeClient;      // optional external KV cache tier (lmcache_bridge_client.h)
class GpuLink;          // 2-rank tensor-parallel comm (gpu_link.h); tp_attach takes it non-owning

// Device (bf16) weight pointers for one layer.
struct Qwen35LayerWeights {
    bool linear_attn = false;
    bool q_has_gate = false;
    // Muse Glimmer: true = sliding-window attention (RoPE, windowed KV), false = global
    // attention (NoPE -- no RoPE at all, full KV). Mirrors Qwen35Config::swa_layers[i];
    // copied per-layer at weight-load time so the decode loop doesn't need the config.
    bool swa = false;
    const void* input_norm   = nullptr;  // [hidden]
    const void* wq = nullptr;            // [hidden, n_q_heads*head_dim]
    // Muse Glimmer only: attn_gate ships as its OWN [hidden, n_q_heads*head_dim] tensor rather
    // than pre-fused into wq the way Qwen3.6's GGUF writes it, so it is kept quantized and
    // projected separately instead of being dequantized and interleaved into wq at load.
    const void* wgate = nullptr;         // [hidden, n_q_heads*head_dim]
    const void* wk = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wv = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wo = nullptr;            // [n_q_heads*head_dim, hidden]
    const void* q_norm = nullptr;        // [head_dim]
    const void* k_norm = nullptr;        // [head_dim]
    const void* post_attn_norm = nullptr;// [hidden]
    // Muse Glimmer sandwich norm -- FOUR distinct norm weights per layer, vs the two
    // (input_norm, post_attn_norm double-duty as the FFN's pre-norm) every other
    // architecture here uses:
    //   input_norm       existing field -- pre-attention norm ("attn_norm" tensor)
    //   post_attn_norm   existing field, repurposed -- sandwich norm on the attention
    //                    output ("post_attention_norm" tensor), added to the residual via
    //                    launch_norm_then_add rather than being norm'd itself like every
    //                    other model's post_attn_norm usage
    //   ffn_norm         new -- genuine pre-FFN norm ("ffn_norm" tensor). Other
    //                    architectures reuse post_attn_norm for this (one norm serves as
    //                    both "post-attention" and "pre-FFN" since there's no sandwich
    //                    step between them); Muse Glimmer ships both as distinct tensors.
    //   post_ffn_norm    new -- sandwich norm on the FFN output ("post_ffw_norm" tensor),
    //                    added to the residual the same way post_attn_norm now is.
    const void* ffn_norm = nullptr;      // [hidden]
    const void* post_ffn_norm = nullptr; // [hidden]
    const void* router_w = nullptr;      // [hidden, n_experts]
    int router_w_type = 0;              // ggml_type when router_w is kept quantized (0 = bf16 dense)
    const void* gate = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* up   = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* down = nullptr;          // [n_experts, moe_ffn, hidden]
    const void* shared_gate = nullptr;   // bf16 dense fallback
    const void* shared_up   = nullptr;
    const void* shared_down = nullptr;
    const void* shared_gate_inp = nullptr;// [hidden] -> scalar shared-expert gate
    // Qwen3.6 UD: shared expert stored as Q8_0 (on-read GEMV, ~2x less weight BW vs bf16).
    const void* shared_gate_q = nullptr; const void* shared_up_q = nullptr; const void* shared_down_q = nullptr;
    int shared_gate_qtype = 0, shared_up_qtype = 0, shared_down_qtype = 0;

    // Qwen3.5/Qwen3.6 Gated DeltaNet tensors (linear-attention layers only).
    const void* wqkv = nullptr;           // [hidden, q+k+v]
    const void* wqkv_gate = nullptr;      // [hidden, value_dim]
    const void* ssm_conv = nullptr;       // [conv_kernel, q+k+v]
    const void* ssm_dt = nullptr;         // [value_heads]
    const void* ssm_a = nullptr;          // [value_heads]
    const void* ssm_beta = nullptr;       // [hidden, value_heads]
    const void* ssm_alpha = nullptr;      // [hidden, value_heads]
    const void* ssm_norm = nullptr;       // [linear_head_dim]
    const void* ssm_out = nullptr;        // [value_dim, hidden]

    // Experts kept quantized in VRAM (gguf-native [E,out,in] layout). Set by BOTH loaders: the
    // GGUF path keeps the file's own blocks, and load_compressed_tensors builds them by
    // requantizing the checkpoint's NVFP4 (see gate_nv below for what that costs).
    // When gate_q != nullptr decode reads these INSTEAD of the bf16 gate/up/down above, and reads
    // them in place: launch_moe_expert_ffn_q4k is a dp4a MMVQ that decodes each block in-register
    // during the dot product. There is no per-layer dequant-to-scratch step -- the only
    // launch_gguf_dequant calls on these tensors are at load time. *_qtype are ggml type ids.
    const void* gate_q = nullptr; const void* up_q = nullptr; const void* down_q = nullptr;
    int gate_qtype = 0, up_qtype = 0, down_qtype = 0;
    // Optional native gate/up copies retained for batched prefill when decode uses an
    // internal compact format. Decode continues to read gate_q/up_q.
    const void* prefill_gate_q = nullptr; const void* prefill_up_q = nullptr;
    int prefill_gate_qtype = 0, prefill_up_qtype = 0;
    // Optional SM120-native NVFP4 copies used only by Muse batched prefill. Decode and all
    // unsupported shapes continue to use the GGUF-native pointers above.
    const void* gate_fp4 = nullptr; const void* gate_fp4_sf = nullptr;
    const void* up_fp4 = nullptr;   const void* up_fp4_sf = nullptr;
    // 1/weight_global_scale for checkpoint-native NVFP4 B operands. Muse's
    // from-bf16 quant_b copies leave this at 1 (global already folded into UE4M3).
    float gate_fp4_alpha = 1.f, up_fp4_alpha = 1.f, down_fp4_alpha = 1.f;
    // q | attn_gate | k | v stacked into ONE [2*qdim + 2*kvdim, H] FP4 operand. They all project
    // the same `xn` and the batched prefill already runs them as a single grouped GEMM to keep the
    // grid full, so the FP4 form has to stay grouped too -- as four separate GEMMs at m=128 the
    // biggest is 32 CTAs of a 170-SM part. Row order matches the split below: q, gate, k, v.
    const void* qkvg_fp4 = nullptr; const void* qkvg_fp4_sf = nullptr;
    // o projection [H, qdim]. Unlike ffn_down (~3.9 GB, reverted in #825) this is ~0.8 GB, and it
    // is gated on a free-VRAM preflight so a card that cannot spare it keeps the int8 path.
    const void* wo_fp4 = nullptr;   const void* wo_fp4_sf = nullptr;
    // Short-context-only copy; decode retains down_q, so long-context configurations leave this
    // null to preserve KV and batched-prefill scratch headroom on 32-GB cards.
    const void* down_fp4 = nullptr; const void* down_fp4_sf = nullptr;
    // The checkpoint's OWN NVFP4 FFN payloads, in the single-blob layout launch_gemv_nvfp4 reads
    // ([SI_NVFP4_HDR | ue4m3 group scales | packed e2m1]). Distinct from the *_fp4/_fp4_sf pair
    // above, which is the split layout the batched-prefill GEMM wants.
    //
    // These ARE the decode weights as of 2026-08-21 (SPARKINFER_QWEN38_DECODE_NVFP4 defaults on;
    // =0 restores the old path). Historically decode ran gate_q/up_q/down_q instead, built by
    // dequantizing these payloads and REQUANTIZING to Q4_K, which measures 8.25% mean relative
    // weight error against the NVFP4 source (corr 0.9968) -- NVFP4's 8 magnitudes per 16-element
    // group with an e4m3 scale and Q4_K's 16 levels per 32 with a 6-bit scale/min do not nest, so
    // the conversion is pure loss on top of an already-lossy format. Quality cost of that
    // conversion, same corpus, same build, only this flag changed:
    //
    //     Q4_K decode   PPL 3.204
    //     NVFP4 decode  PPL 3.010     6.1% better
    //
    // CORRECTION (2026-08-21). This comment used to claim tau was "unchanged" across the two --
    // 1.6552 on Q4_K against 1.6333 on NVFP4 -- and concluded the requantized weights were not
    // holding the draft back. A proper A/B at ctx=4096 with reps=3, both arms back-to-back on one
    // build, says otherwise: tau 1.6623 on Q4_K against 1.5059 on NVFP4, a 9.4% drop. The earlier
    // pair was not measured at this context and should not have been quoted as a null result.
    // Acceptance DOES move with the target's numerics, and it moves against native NVFP4, which is
    // the opposite of the intuitive direction -- worth understanding rather than assuming away.
    // Both arms are byte-lossless vs AR over 3 repeats.
    const void* gate_nv = nullptr; const void* up_nv = nullptr; const void* down_nv = nullptr;
    // Full-attention q/k/v projections as the checkpoint ships them (ModelOpt NVFP4). Unlike the
    // GDN pointers below these do NOT alias a payload decode already holds: decode reads the Q4_K
    // wq/wk/wv/wo built from the same bytes, because native NVFP4 GEMV is ~0.65x at m=1. So the
    // packed nibbles are genuinely extra residency and are gated on a free-VRAM preflight.
    // q and attn_gate stay ONE wide [2*qdim, H] operand, exactly as wq already is, so the
    // split_q_gate that follows the projection is unchanged.
    const void* wq_fp4 = nullptr; const void* wq_fp4_sf = nullptr;
    const void* wk_fp4 = nullptr; const void* wk_fp4_sf = nullptr;
    const void* wv_fp4 = nullptr; const void* wv_fp4_sf = nullptr;
    // Muse builds wo_fp4 from bf16 and folds the global scale into UE4M3, leaving alpha 1; the
    // checkpoint-native path carries 1/weight_global_scale here instead (pack_sfb omits it).
    float wq_fp4_alpha = 1.f, wk_fp4_alpha = 1.f, wv_fp4_alpha = 1.f, wo_fp4_alpha = 1.f;
    // GDN in/out projections, for checkpoints that ship them as NVFP4 (ModelOpt). These point INTO
    // the payload keep_nvfp4_native already keeps resident for decode's launch_gemv_nvfp4, so only
    // the CUTLASS SFB scale copy (1 byte per 16 weights) is new VRAM -- the packed nibbles are
    // shared, not duplicated. Null on any checkpoint that stores the GDN projections in FP8.
    const void* gdn_qkv_fp4 = nullptr; const void* gdn_qkv_fp4_sf = nullptr;
    const void* gdn_z_fp4 = nullptr;   const void* gdn_z_fp4_sf = nullptr;
    const void* gdn_out_fp4 = nullptr; const void* gdn_out_fp4_sf = nullptr;
    float gdn_qkv_fp4_alpha = 1.f, gdn_z_fp4_alpha = 1.f, gdn_out_fp4_alpha = 1.f;
    // attention projections: 0 = bf16 dense (default); else ggml type id (12=Q4_K,
    // 14=Q6_K, 8=Q8_0) or SI_QTYPE_FP8 (108) / SI_QTYPE_NVFP4 (109) -> weights kept
    // quantized in VRAM, decoded on-read by launch_gemv_q / launch_gemv_fp8 /
    // launch_gemv_nvfp4.
    int wq_type = 0, wgate_type = 0, wk_type = 0, wv_type = 0, wo_type = 0;
    int wqkv_type = 0, wqkv_gate_type = 0, ssm_beta_type = 0, ssm_alpha_type = 0, ssm_out_type = 0;
    int shared_gate_inp_type = 0;

    // Muse Glimmer fused prefill: per-output-row int8 scales (amax/127) of the native Q4_K/Q5_K
    // attn + FFN gate/up weights, precomputed once at load. Let the batched prefill GEMM decode the
    // weight to int8 in-register (launch_prefill_gemm_qi8_dense) instead of materializing the whole
    // int8 weight first. Null => that weight stays on the materialize path (e.g. Q6_K down, or when
    // the precompute is disabled/unavailable). Owned by the model Impl, not freed per-layer.
    const float* wq_rs = nullptr; const float* wgate_rs = nullptr;
    const float* wk_rs = nullptr; const float* wv_rs = nullptr; const float* wo_rs = nullptr;
    const float* gate_rs = nullptr; const float* up_rs = nullptr;
    // ffn_down's per-row int8 scale. Left out of the original dense set on the assumption that
    // down is always Q6_K; in the shipped Muse Q4_K_M GGUF a large share of the down tensors are
    // Q4_K, i.e. fusable, and were falling back to the int8 materialize only because the scale
    // pool had no slot for them. Null when this layer's down really is unfusable (Q6_K).
    const float* down_rs = nullptr;
    // GDN-layer row scales, the same role wq_rs/wk_rs/... play on a full-attention layer. A hybrid
    // stack projects through wqkv/wqkv_gate/ssm_out on its linear-attention layers, and those had
    // no fused-B arm at all -- ~23% of a 27B hybrid's non-embedding parameters.
    const float* wqkv_rs = nullptr; const float* wqkv_gate_rs = nullptr;
    const float* ssm_out_rs = nullptr;
};

struct Qwen35Weights {
    const void* embed_tokens = nullptr;  // [vocab, hidden]
    const void* final_norm   = nullptr;  // [hidden]
    const void* lm_head      = nullptr;  // [hidden, vocab]  (pre-transposed)
    int lm_head_type = 0;                 // 0 = bf16; else ggml type -> on-read quantized GEMV
    // The checkpoint's OWN NVFP4 lm_head in the block-scaled GEMM's operand layout, kept beside
    // the Q4_K copy rather than replacing it: AR decode and the speculative verify keep reading
    // lm_head above byte for byte, and only a packed decode wide enough to want a GEMM reads
    // these. Built only when the checkpoint actually ships an NVFP4 head and VRAM allows.
    const void* lm_head_fp4 = nullptr; const void* lm_head_fp4_sf = nullptr;
    float lm_head_fp4_alpha = 1.f;
    std::vector<Qwen35LayerWeights> layers;
};

// Window over the GDN value-head state, for the per-device state split (dual-GPU). The
// recurrent lin_state ([slots][v_heads][HD][HD], dense) is split so each device keeps
// only its own [v_start, v_start + v_count) slice of the v-heads; conv state stays FULL on
// every device (it is not split per v-head at a cost/benefit worth paying -- activations are
// full-width per device, so splitting the conv rows buys nothing until the forward path is split).
// (0,0) = "all heads" = the tp=1 default: every derived size, pointer, and launch argument
// must evaluate byte-identical to the unsplit model under it.
struct GdnStateWindow {
    int v_start = 0;   // first local v-head index, in [0, linear_v_heads]
    int v_count = 0;   // local v-head count; 0 = all heads
};

// Validates a GdnStateWindow against the model config, returning the effective window.
// Degenerate cases (non-hybrid config, v_count <= 0) pass through as the all-heads window.
// Otherwise the window must be exactly expressible by the GDN kernels' qh mapping, else it is
// clamped to the all-heads window and *warned (out-param, may be null) is set so the caller
// prints the diagnostic once:
//   - OOB: v_start < 0 or v_start + v_count > linear_v_heads;
//   - block mode (gdn_qh_block): with g = v/q (v % q must be 0), g must divide BOTH
//     v_start and v_count (the kernel maps local v-head vh to q-head vh/(v/q));
//   - cyclic mode: only a LEADING window is expressible -- v_start == 0,
//     v_count <= linear_q_heads, and v_count % linear_q_heads == 0 (the kernel maps
//     vh to q-head vh % q, so a mid-range window would alias q-heads across the split).
inline GdnStateWindow gdn_window_normalize(const Qwen35Config& cfg, const GdnStateWindow& in, bool* warned)
{
    if (warned) *warned = false;
    GdnStateWindow w = in;
    const int v = cfg.linear_v_heads, q = cfg.linear_q_heads;
    if (!cfg.hybrid || v <= 0 || q <= 0 || w.v_count <= 0) return GdnStateWindow{};
    bool bad = w.v_start < 0 || w.v_start + w.v_count > v;
    if (!bad) {
        if (cfg.gdn_qh_block) {
            if (v % q == 0) {
                const int g = v / q;
                if (w.v_start % g != 0 || w.v_count % g != 0) bad = true;
            } else bad = true;
        } else {
            if (w.v_start != 0 || w.v_count > q || w.v_count % q != 0) bad = true;
        }
    }
    if (bad) {
        if (warned) *warned = true;
        return GdnStateWindow{};  // clamp to all heads
    }
    return w;
}

// (dual-GPU D5) Formats one line of the per-card budget audit for one rank of a tp>1 load.
// est = on_disk_bytes + 33,024 * ctx (G2 per-token int8 KV: (2,048 data + 16 scale) x 16
// full-attn layers) + 154,927,104 (G2 GDN per-sequence state); the verdict is against a
// 16 GiB card (17,179,869,184 B). Free (no CUDA dependency) so the CPU tests can check the
// exact line text; print_tp_audit feeds it the per-instance numbers.
std::string tp_audit_line(int rank, int device, unsigned long long on_disk_bytes,
                           unsigned long long card_free_bytes, unsigned long long card_total_bytes,
                           int ctx);

// Single-sequence (batch=1) greedy decoder for the Qwen family -- routed-MoE (Qwen3.6) and
// dense-FFN (Qwen3.8, Qwen3.5) alike, plus Muse Glimmer. Owns scratch buffers and
// drives embed -> N layers -> final norm -> LM head -> argmax per token.
class Qwen35Model {
public:
    // gdn_window: which slice of the GDN v-head recurrent state this model instance owns
    // (dual-GPU state split). Default (0,0) = all heads = the unsplit, byte-identical tp=1
    // behavior; see GdnStateWindow and gdn_window_normalize for the validity rules.
    //
    // rank/device (dual-GPU Wave 3, the per-device weight split): the rank indexes the process
    // tp-table (tp::get_process_table) and device is the real CUDA id this instance is built
    // on (the engine's eff[rank]). The ctor does a ONE-TIME cudaSetDevice(device) (and the dtor
    // mirrors it before its frees) so every buffer this instance allocates and frees lives on its
    // own card. Defaults (0,0) fire neither call, so the tp=1 default (rank 0 / device 0) is
    // byte-identical to the unsplit single-device model.
    Qwen35Model(const Qwen35Config& cfg, KVCacheManager* kv, moe::MoEEngine* engine,
               GdnStateWindow gdn_window = {}, int rank = 0, int device = 0);
    ~Qwen35Model();

    // (dual-GPU D5) Prints this instance's rank one per-card budget-audit line: the process
    // table's on-disk bytes for this rank, this card's free/total (a failed probe falls back
    // to the nominal 16 GiB card, so the line stays printable headless), and the per-card
    // estimate at ctx against a 16 GiB card. By design a no-op unless the process table is
    // set with n_ranks > 1 -- the tp=1 default never prints.
    void print_tp_audit(int ctx) const;
    // The loaded weight set (pointers into this instance's owned device buffers). Exposed for
    // the tp=1 byte-identity checksum in runtime/tests/tp_weights_cpu_test.cpp.
    const Qwen35Weights& weights() const;

    // (dual-GPU WP-9) Tensor-parallel attach for a tp=2 load. Called on the group-leader
    // instance (the rank-0 model held by the engine) with the process-wide GpuLink (non-owning;
    // the engine creates and destroys it) and the non-owning peer model pointers in rank order
    // (peers[r] is the model on rank r, so peers[tp_rank] is this instance). Every public op on
    // the leader is then mirrored onto every peer on a per-op worker thread; the peers run the
    // same code with their local (sliced) weights and contribute their column/row windows, and
    // one fused GpuLink all-reduce per transformer layer combines the per-rank partials.
    // No-op guard for the tp=1 case: tp_attach is never called there and every TP site below
    // is degenerate, keeping tp=1 byte-identical.
    void tp_attach(GpuLink* link, int my_rank, const std::vector<Qwen35Model*>& peers);
    bool tp_active() const;   // tp world > 1 and link attached
    int tp_rank() const;      // this instance's rank (0 when not attached)

    // Per-rank view of the tp=2 scratch, so the group leader can name the PEER's device,
    // stream and staging rows when it builds a GpuLink::RankRef pair for the all-reduce. Set
    // once by tp_attach (the values never change after that); all other fields are -1/null.
    struct TpRankView {
        int device = -1;
        cudaStream_t stream = nullptr;
        uint16_t* xrow = nullptr;   // bf16[hidden] op-entry embedding-exchange row
        uint16_t* ar   = nullptr;   // bf16[32][2*hidden] per-layer AR-A/B staging (row 0 in use)
        float* logits  = nullptr;   // f32[vocab] the decode logits row (epilogue all-reduce)
    };
    TpRankView tp_rank_view() const;

    void set_weights(const Qwen35Weights& w);

    // Serializes CUDA-graph capture against any OTHER thread issuing work on the legacy default
    // stream. Capture happens inside forward_token on the continuous-batch worker thread; the HTTP
    // thread concurrently runs submit-time device work (KVCacheManager::allocate's block-table
    // cudaMemcpy, open_session's cudaMalloc, reset_penalty_counts, set_logit_bias).
    //
    // The decode stream is created with cudaStreamCreate, i.e. a BLOCKING stream, so the legacy
    // stream implicitly synchronizes with it. Issuing anything on the legacy stream while that
    // stream is capturing is a hard CUDA error -- "operation would make the legacy stream depend
    // on a capturing blocking stream" -- and it poisons the capture, after which end-capture fails
    // and the process dies (std::length_error / corrupted size vs. prev_size / SIGSEGV, depending
    // on what reads the half-built graph first). Reported from a 16-concurrent burst against
    // /v1/chat/completions; never seen at <=12 because a submit has to land inside the capture
    // window, which is a narrow target until the request rate is high enough.
    //
    // cudaStreamCaptureModeThreadLocal does NOT protect against this: that mode governs the
    // capture-safety check for unsafe API calls, not the legacy stream's implicit-sync semantics,
    // which apply regardless of mode or calling thread.
    //
    // Lock ordering: ContinuousBatchEngine takes its own mu_ first, then this. The worker takes
    // only this. No cycle.
    // RECURSIVE: cache_prefix() holds it and then calls ingest_prompt_range() ->
    // forward_token(), which takes it again on the same thread. A plain mutex self-deadlocks
    // there. Recursion only helps same-thread re-entry, which is exactly this case; the worker
    // thread still blocks normally.
    std::recursive_mutex& device_mutex();

    // Stage an image for the NEXT prefill_batched call. emb is [n_img, hidden] float32 from
    // qwen_vision_forward; positions are the prompt indices carrying image_token_id, which the
    // caller has already checked number exactly n_img. Consumed by that one prefill and then
    // cleared -- it is per-request state, not model state, and leaving it set would splice a
    // stale image into the next prompt.
    bool set_pending_vision(const float* emb, const int* positions, int n_img, int hidden);

    // Supplies interleaved-MRoPE rotary positions for the NEXT prefill, plus the offset that every
    // subsequent decode step of this session must add to its rotary position.
    //
    //   positions:     [n_tokens * 3] host int32, laid out [t0,h0,w0, t1,h1,w1, ...]
    //   decode_offset: (rotary position after the prompt) - n_tokens. Zero or negative: a vision
    //                  span advances the rotary counter by max(h,w)/merge, not by its token count.
    //
    // Text-only callers never call this, and the prefill and decode paths then run exactly the
    // code they ran before MRoPE existed.
    bool set_pending_mrope(const int* positions, int n_tokens, int decode_offset);
    void clear_pending_mrope();
    // Clears the decode offset. Needed when a session is reused for a new, image-free prompt --
    // the offset outlives the positions by design, so it must be cleared explicitly.
    void reset_mrope_offset();
    void clear_pending_vision();

    // Load weights from a sparkinfer weight directory (see runtime/tools/convert_qwen35.py).
    // Returns false on failure. Allocates device buffers it owns.
    bool load_weights(const std::string& dir);

    // Load weights directly from a GGUF file (native). Dense tensors are
    // dequantized to bf16; expert tensors are kept quantized in VRAM and decoded on-read by the
    // MMVQ decode kernel, never materialized (Q4_K_M-sized resident footprint).
    bool load_gguf(const std::string& path);
    // HuggingFace "compressed-tensors" NVFP4 checkpoint -- both the uniform-NVFP4 ModelOpt build
    // the eval scores and the mixed NVFP4-FFN/FP8-attention unsloth build it guards
    // -- see runtime/src/models/qwen35.cpp's own comment on the function for the exact scheme.
    bool load_compressed_tensors(const std::string& model_dir);

    // Greedy generate: prompt token ids -> generated token ids (host). An optional ThermalGovernor
    // paces decode under thermal pressure (accuracy-preserving); nullptr = full speed, no overhead.
    // When a prefix cache is installed (cache_prefix), skips re-prefilling the matching prefix.
    // When SPARKINFER_DFLASH=1 and a draft is attached via set_dflash_draft(), uses DFlash
    // block-diffusion speculative decoding (greedy-equivalent to AR when correct).
    // Optional out_ttft_s/out_decode_s split prefill from decode wall-clock, same convention as
    // DFlashStats below -- without this, tok/s computed from total wall time collapses toward the
    // prefill rate at long context (32k prefill dwarfs a 128-token decode), not the decode rate.
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                              ThermalGovernor* gov = nullptr,
                              double* out_ttft_s = nullptr, double* out_decode_s = nullptr);

    // DFlash speculative generate (greedy). Requires set_dflash_draft(). Returns generated ids.
    // Optional stats: mean acceptance length τ and wall-clock seconds for decode (post-TTFT).
    struct DFlashStats {
        double mean_accept = 0;   // mean tokens committed per draft step (τ)
        double decode_s = 0;
        double ttft_s = 0;
        int    steps = 0;
    };
    //
    // Engine-driven use. A caller that owns the session (ContinuousBatchEngine) passes hooks:
    // dflash_generate then prefills and decodes on hooks->seq_id instead of opening a session of its
    // own, hands each step's committed tokens to on_tokens, and leaves the session open. on_tokens
    // returning false stops it between steps -- with the KV and the Gated-DeltaNet state exactly at
    // the committed position, so ordinary decode can take over from SpecResume::position with
    // SpecResume::next_token and produce what the speculative loop would have.
    struct SpecHooks {
        uint64_t seq_id = 0;
        std::function<bool(const int* tokens, int n)> on_tokens;
    };
    struct SpecResume {
        bool engaged = false;   // false: nothing ran -- speculation would not pay here, or could not start
        bool finished = false;  // EOS or max_new_tokens reached
        bool failed = false;    // a draft or verify pass failed; nothing past `position` is trustworthy
        int position = 0;       // prompt length + committed tokens: where decode resumes
        int next_token = -1;    // the verified token at `position`, not yet emitted nor ingested
        int emitted = 0;        // tokens handed to on_tokens
        bool tier_boundary = false;  // stopped where the next step would cross a KV split tier
    };
    std::vector<int> dflash_generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                                     DFlashStats* stats = nullptr,
                                     ThermalGovernor* gov = nullptr,
                                     const SpecHooks* hooks = nullptr,
                                     SpecResume* resume = nullptr);

    // Prefill `tokens` and retain KV + hybrid recurrent state for reuse on the next request
    // whose prompt starts with the same token sequence. Returns false on allocation failure.
    bool cache_prefix(const std::vector<int>& tokens);

    // Restore the Gated-DeltaNet recurrent state to its end-of-prefix snapshot taken by
    // cache_prefix(). Required before reusing a cached prefix: the KV blocks survive a
    // request, but generation advances lin_state/lin_conv_state past the prefix, and the
    // 48 hybrid layers would otherwise carry the PREVIOUS request's history. No-op (returns
    // false) when no prefix is installed.
    bool restore_prefix_state();

    // Logical KV blocks the installed prefix occupies, for callers that want to keep them
    // across requests via KVCacheManager::truncate_blocks(). 0 when no prefix is active.
    int prefix_block_count() const;

    // Drop the installed prefix cache and free its KV blocks.
    void clear_prefix_cache();

    // Soft-invalidate after session 0's KV was freed (generate / ContinuousBatchEngine).
    // Keeps prefix_tokens/len so the next cache_prefix() can re-warm; clears prefix_active
    // so callers do not skip re-warm while the KV is empty.
    void release_prefix_session();

    // Length of the currently cached prefix (0 if none).
    int prefix_cached_len() const;

    // Argmax seed after cache_prefix(); -1 if no active prefix cache.
    int prefix_seed_token() const;

    // True when `prompt` begins with the installed prefix token sequence (requires active cache).
    bool prompt_matches_prefix(const std::vector<int>& prompt) const;

    // Time-to-first-token: ingest `prompt` with prefill (no LM head on interior tokens when
    // not legacy), then one sampled forward. Reuses cache_prefix when the prompt starts with
    // the cached tokens (only the suffix is prefilled). Returns seconds.
    double bench_ttft(const std::vector<int>& prompt);

    // Run one token at `position`. When sample=false (prefill), runs embed→layers→final
    // norm without LM head/argmax and without CUDA-graph capture — teacher-forced ingestion.
    // When sample=true (decode / last prompt token), runs the full path and may capture/replay
    // the decode graph. Returns argmax next-token id when sample=true, else token_id.
    //
    // temperature <= 0 (the default) is plain greedy argmax, byte-identical to the pre-sampling
    // behavior. temperature > 0 draws via Gumbel-max (kernels::launch_temperature_sample) before
    // the argmax reduction; seed/sample_step select the draw -- sample_step must be a fresh,
    // request-scoped, 0-based decode-step counter (ContinuousBatchEngine::Job::decode_emitted is
    // the intended source) so the SAME captured decode graph can be replayed across separate
    // requests/sessions with different temperature/seed without invalidating reproducibility.
    //
    // top_k (<=0 or >=vocab disables) and top_p (<=0 or >=1.0 disables) truncate the candidate
    // set BEFORE the Gumbel draw above -- top_k first, then top_p narrows within it (see
    // kernels::launch_topk_topp_mask). Provably inert whenever temperature<=0: the greedy winner
    // (highest logit) is always in both the top_k and top_p surviving set by construction, so
    // masking never changes a greedy result -- neither param needs temperature>0 to be accepted,
    // and neither needs its own DFlash-incompatibility check (temperature>0 is already rejected
    // under DFlash independent of top_k/top_p).
    //
    // presence_penalty/frequency_penalty (OpenAI's [-2.0, 2.0] range; 0 disables both) subtract
    // count-weighted terms from the FULL vocab-sized logits row BEFORE top_k/top_p truncation
    // above -- see kernels::launch_presence_frequency_penalty. UNLIKE top_k/top_p, this has NO
    // inertness proof at temperature<=0: a nonzero penalty can change WHICH token has the highest
    // logit even under pure greedy argmax, since it isn't restricted to non-winning ranks the way
    // truncation is. DFlash (which requires exact greedy-argmax determinism against its draft
    // model) must therefore reject presence_penalty!=0 || frequency_penalty!=0 independently of
    // temperature -- see should_reject_dflash_penalty in chat_tools.hpp, mirroring
    // should_reject_dflash_temperature. Counts accumulate in the CURRENT session's per-session
    // penalty_counts buffer (see SessionBuffers, activate_session()) across every decode step of
    // the SAME request -- reset once per request at submit time
    // (Qwen35Model::reset_penalty_counts), NOT per forward_token() call.
    //
    // logit_bias (OpenAI's [-100, 100] per-token range) adds a fixed per-vocab-id bias to the same
    // FULL vocab-sized logits row, also before top_k/top_p truncation -- see
    // kernels::launch_logit_bias. Unlike every other sampling control here, it takes NO parameter
    // in this function: the bias is STATIC for the whole request (not refreshed every decode step
    // like temperature/top_k/top_p/the penalties), so it is set ONCE per request, at submit time,
    // directly into the CURRENT session's per-session logit_bias buffer (see SessionBuffers,
    // activate_session(), Qwen35Model::set_logit_bias) -- forward_token() just reads whatever is
    // currently in that buffer, every decode step, unconditionally (same graph-replay-safety
    // discipline as everything else on this path). Same "no inertness proof at temperature<=0,
    // needs its own DFlash check" story as presence/frequency penalty -- see
    // should_reject_dflash_logit_bias in chat_tools.hpp.
    int forward_token(int token_id, int position, bool sample = true, float temperature = 0.f,
                      unsigned long long seed = 0, unsigned long long sample_step = 0,
                      int top_k = 0, float top_p = 1.f,
                      float presence_penalty = 0.f, float frequency_penalty = 0.f);

    // Copy the most recent step's logits (vocab floats) to host. Valid after a
    // forward_token() call. Used for teacher-forced scoring (perplexity / KL).
    void copy_logits(float* host_logits) const;

    // Minimal, string-free (tokenization is a server-layer concern) -- the chosen token's own
    // logprob plus its top `top_alternatives.size()` alternatives by raw logit, sorted descending.
    struct TokenLogprob {
        int token_id = -1;
        float logprob = 0.f;
        std::vector<std::pair<int, float>> top_alternatives;
    };
    static constexpr int kMaxTopLogprobs = 20;   // OpenAI's own top_logprobs ceiling

    // Reads the RAW (post-softcap, pre-truncation, pre-temperature-noise) model distribution
    // behind the token the IMMEDIATELY PRECEDING forward_token() call produced: that token's own
    // logprob plus its top `top_n` (clamped to [0, kMaxTopLogprobs]) alternatives. "Raw" here means
    // this reports the model's true confidence, not an artifact of the caller's own
    // temperature/top_k/top_p choices -- matches real-world OpenAI/vLLM logprobs semantics.
    //
    // Valid under the same "forward_token() syncs its stream before returning" contract as
    // copy_logits() above -- NOT valid on the DFlash deferred-collect early return
    // (s.defer_decode_sync == true, forward_token() returns kDFlashDeferred without syncing);
    // unexercised in practice since ContinuousBatchEngine (the sole caller behind
    // /v1/chat/completions) never sets dflash_cap.
    //
    // Deliberately NOT baked into forward_token()'s own D2H tail (unlike h_out_id): this does its
    // own on-demand cudaMemcpy(s), so calling (or never calling) it costs nothing extra on the
    // decode hot path for logprobs=false requests -- the only always-on cost logprobs adds to
    // every decode step is the two small device-side kernel changes documented on
    // kernels::launch_topk_topp_mask/launch_extract_chosen_logit (both correctness-critical for
    // graph replay safety regardless of whether THIS request wants logprobs).
    TokenLogprob last_token_logprobs(int top_n = kMaxTopLogprobs) const;

    // Same distribution, different token: reports the logprob of `token_id` (any vocab entry)
    // under whatever distribution the preceding forward_token()/prefill produced, instead of the
    // argmax's. The primitive teacher-forced scoring is built on -- feed position i, ask for the
    // logprob of the token that ACTUALLY follows at i+1, regardless of what the model would have
    // picked. Same validity window and top_alternatives semantics as last_token_logprobs(); an
    // out-of-range token_id returns a default-constructed (token_id = -1) entry.
    TokenLogprob token_logprob_for(int token_id, int top_n = kMaxTopLogprobs) const;

    struct BenchDecodeResult {
        double decode_tps = 0;
        double prefill_pp = 0;
    };
    // Benchmark at a target KV depth: timed prefill, untimed warmup decode, timed decode.
    BenchDecodeResult bench_decode(int warmup, int n_tokens, int context_tokens = 0);

    const Qwen35Config& config() const;

    // The GDN v-head state window this instance owns, as normalized by gdn_window_normalize at
    // construction time (all heads when the model was constructed unsplit).
    const GdnStateWindow& gdn_state_window() const;
    // Number of v-heads this instance keeps per state slot: v_count when a window is set,
    // linear_v_heads otherwise. Every per-slot state size derives from this.
    int gdn_v_local() const;

    // Batched prompt prefill: process all `n` prompt tokens in one
    // pass, filling the paged KV cache and Gated-DeltaNet recurrent/conv state for positions
    // 0..n-1 so a subsequent decode is faithful to the forward_token loop. Returns the argmax at
    // the last prompt position (seed for the first decode step), or -1 if the batched path is
    // unsupported for this model/config. Implemented in qwen35_prefill.cpp.
    //
    // Written for the Qwen3.5 dense hybrid, but no longer restricted to it: the 256-expert MoE
    // (Qwen3.6-35B-A3B) and the dense-FFN Qwen3.8-27B both take this path, as does Muse Glimmer.
    // The eligibility checks at the top of prefill_batched_run (qwen35_prefill.cpp) are
    // authoritative -- do not infer the supported set from here.
    // want_seed_logprob additionally leaves the sampler-scratch distribution populated for the
    // seed token, so last_token_logprobs() is valid immediately after this returns -- see
    // ingest_prompt_range's identical parameter. Off by default: it costs a full-vocab sort, and
    // the seed's logprob is only wanted when the request asked for logprobs.
    // Hand back the NVFP4 LM-head operand (see Qwen35Weights::lm_head_fp4). Idempotent.
    void release_lm_head_fp4();
    int prefill_batched(const int* prompt_ids, int n, bool want_seed_logprob = false,
                        int pos0 = 0);
    // Prefill several FRESH sessions' prompts in ONE batched pass (Qwen35PrefillCtx::multi_n):
    // each session opened with nothing ingested yet, text only, no logit_bias. On success writes
    // each prompt's argmax seed -- the token ingest_prompt_range() would have returned for it -- to
    // seeds[i] and returns true. Returns false when the pack is not eligible or a stage declines;
    // the caller then ingests the prompts one at a time from position 0, which resets whatever
    // this pass had written.
    bool ingest_prompts_packed(const uint64_t* seq_ids, const int* const* prompts, const int* lens,
                               int n_prompts, int* seeds);
    // Same pass, ingested as position-windows so the scratch arena is bounded by the window
    // rather than by n. Returns the seed for the last token, or -1 if a window was refused --
    // in which case *out_done (when given) reports how many leading tokens ARE in the cache,
    // so the caller can finish from there instead of recomputing from zero.
    int prefill_batched_chunked(const int* prompt_ids, int n, bool want_seed_logprob = false,
                                int* out_done = nullptr);
    // prefill_batched_chunked() for a range that does NOT start at zero: prompt_ids[start, end) at
    // their absolute positions, continuing the KV and recurrent state already in place for
    // [0, start) -- a prefix restored from the prefix cache. Same single-pass/window split. Returns
    // the seed, or -1 with *out_done (when given) counting the tokens past `start` that did land.
    int prefill_batched_resume(const int* prompt_ids, int start, int end,
                               bool want_seed_logprob = false, int* out_done = nullptr);

    // Prefill prompt tokens [start, end) with the batched path when start==0 and eligible, else
    // the token loop. chunk_limit > 0 caps the token-loop path to at most chunk_limit tokens per
    // call (the batched path never chunks — it always covers the full range in one pass, so a
    // chunk_limit smaller than end-start forces the token-loop fallback); pass 0 for unlimited
    // (single call covers the whole range). out_pos, if non-null, receives the position reached
    // (== end once the whole range is consumed, < end if chunk_limit stopped it early). Returns
    // the argmax seed for decode once out_pos == end, or -1 while there is remaining work (or on
    // failure). The single funnel both cache_prefix()'s exclusive-session path and
    // ContinuousBatchEngine::step_job()'s continuous-batch path dispatch prefill through, so
    // batched-vs-token-loop routing and external KV cache lookup/store (when a bridge is
    // attached via set_lmcache_bridge()) only need to be implemented once.
    //
    // want_seed_logprob: leave the sampler scratch populated for the seed token so that
    // last_token_logprobs() is valid for it once this returns out_pos == end. The token-loop path
    // gets this for free (its final forward_token(sample=true) runs the whole sampler tail), but
    // the batched path's own LM-head tail stops at argmax, which is why the first token of every
    // response had no logprob entry. Off by default: a full-vocab sort is not worth paying when
    // the caller never asked for logprobs.
    //
    // allow_batched_resume: start > 0 continues KV and recurrent state the caller has put in place
    // (a prefix-cache hit), so the batched path may take [start, end) rather than the token loop.
    // Off by default: a start > 0 that is a token-loop continuation keeps its existing path.
    int ingest_prompt_range(const int* ids, int start, int end, int chunk_limit = 0,
                            int* out_pos = nullptr, bool want_seed_logprob = false,
                            bool allow_batched_resume = false);

    // Attaches an optional external KV cache tier (docs/lmcache_bridge_protocol.md). Null (the
    // default) leaves every lookup/store call site a no-op -- existing behavior is unchanged
    // unless a caller explicitly opts in. Does not take ownership; the caller (ModelEngine) is
    // responsible for the BridgeClient's lifetime, which must outlive this model.
    void set_lmcache_bridge(BridgeClient* bridge);

    // Per-request session lifecycle for continuous batching / serving.
    // open_session() allocates right-sized KV blocks (+ hybrid recurrent state when needed).
    // activate_session() binds forward_token / prefill to that seq_id. Returns 0 on OOM.
    // alloc_failed, when non-null, is set true only when a real device allocation failed (a
    // cudaMalloc for the hybrid recurrent-state buffers) -- distinct from the KV block pool
    // simply being full, which is a normal, transient "no capacity right now" and leaves
    // *alloc_failed untouched. Callers that care about this distinction (ContinuousBatchEngine,
    // to report 503 instead of 429 -- #779) should zero-init their bool before passing it in.
    // shared_prefix_blocks, when non-null and non-empty, become the session's first KV blocks
    // (KVCacheManager::allocate_with_prefix): a cached prefix is shared, not copied. The caller
    // restores the matching recurrent state and starts prefill after the shared blocks.
    uint64_t open_session(int num_tokens, bool* alloc_failed = nullptr,
                          const std::vector<int>* shared_prefix_blocks = nullptr);
    // store_tokens, when non-null and an LMCache bridge is attached, stores this session's KV
    // for [0, store_tokens->size()) to the bridge (chunk-aligned, see lmcache_maybe_store in
    // qwen35.cpp) before freeing it -- the "session close" eviction point. Most callers don't
    // have the original prompt at this call site and pass nullptr, which is a pure no-op.
    void close_session(uint64_t seq_id, const std::vector<int>* store_tokens = nullptr);
    void activate_session(uint64_t seq_id);

    // PREFIX-CACHE RECURRENT STATE. A cached prefix's KV can be shared block for block, but a
    // Gated-DeltaNet layer's state is one running value per sequence: it cannot be read back at an
    // earlier position, only copied at the position it is at. A snapshot is that copy -- lin_state
    // (fp32) then lin_conv_state (bf16) -- in pinned host memory, so a cached prefix costs host RAM
    // (~205 MB on Qwen3.8-27B) rather than VRAM. A model with no linear layers has nothing
    // recurrent to carry: its snapshot is empty and both calls succeed.
    struct RecurrentStateSnapshot {
        std::shared_ptr<void> host;   // pinned; state_bytes of lin_state, then conv_bytes
        size_t state_bytes = 0;
        size_t conv_bytes = 0;
        // (dual-GPU) Under tensor parallelism each rank holds only its own GDN window, so the
        // leader's snapshot carries rank 1's half here (taken/restored on rank 1's device).
        std::shared_ptr<RecurrentStateSnapshot> peer;
        size_t bytes() const { return state_bytes + conv_bytes + (peer ? peer->bytes() : 0); }
    };
    // Copy seq_id's recurrent state into `out`. False, leaving `out` untouched, when the session is
    // unknown, its state was compacted to bf16 by packed decode (a prefill-time snapshot never is),
    // or the pinned allocation fails.
    bool snapshot_recurrent_state(uint64_t seq_id, RecurrentStateSnapshot& out);
    // Overwrite seq_id's recurrent state with `snap`, in the fp32 form prefill resumes from. False
    // when the session is unknown or the snapshot was taken from a differently shaped model.
    bool restore_recurrent_state(uint64_t seq_id, const RecurrentStateSnapshot& snap);

    // PACKED CONTINUOUS-BATCH DECODE: advance `n` INDEPENDENT sequences by one token each in ONE
    // forward, instead of one full forward per sequence.
    //
    // This is what makes aggregate throughput scale with concurrency. Decode is bandwidth-bound on
    // weight reads, so N sequential forwards read every weight N times; packing the rows reads them
    // once and pays only the per-row cost, which for the dominant GEMVs is a few percent because
    // they already take R rows through a single weight read.
    //
    // tokens/positions/seq_ids are HOST arrays of n entries; out_sampled receives n token ids
    // (greedy argmax, matching what forward_token returns at temperature 0). Returns false --
    // having changed nothing -- when the shape is unsupported, so the caller can fall back to
    // stepping the jobs one at a time.
    //
    // Every sequence must have an open session and live KV. n is capped by the packed graph tiers.
    bool decode_packed(const int* tokens, const int* positions, const uint64_t* seq_ids, int n,
                       int* out_sampled);
    // Largest n decode_packed() accepts. Matches the packed graph tiers.
    static int max_packed_rows();
    uint64_t active_session() const;

    // Zeros seq_id's running presence/frequency-penalty count buffer. MUST be called once per
    // REQUEST that will use this seq_id, even for a freshly open_session()'d id (open_session
    // already zeros at allocation time, but this call is the single, unambiguous "this request's
    // counts start at zero" point -- see ContinuousBatchEngine::submit_locked's call sites).
    // CRITICALLY needed for seq_id == 0 (the shared prefix session): unlike lin_state/
    // lin_conv_state, which are DELIBERATELY persistent/shared across the many unrelated requests
    // that reuse session 0 via use_prefix_session (that sharing IS the prefix-cache
    // optimization), OpenAI's presence/frequency-penalty semantics are scoped to "the CURRENT
    // completion's generated tokens" -- a fresh request reusing session 0 must start with
    // all-zero counts, or one client's generation would incorrectly penalize a later, unrelated
    // client's request sharing the same prefix session. No-op if seq_id has no session entry.
    void reset_penalty_counts(uint64_t seq_id);

    // Sets seq_id's per-request logit_bias buffer: zeros it, then scatters the given sparse
    // (token_id, bias) pairs into it (each bias applied via kernels::launch_logit_bias every decode
    // step of this request, see forward_token's doc comment above). MUST be called once per REQUEST
    // that will use this seq_id, same "even for a freshly open_session()'d id" and "critically
    // needed for seq_id == 0" reasoning as reset_penalty_counts -- call it right alongside that
    // function at every call site (see ContinuousBatchEngine::submit_locked). An empty `bias` still
    // zeros the buffer (clears any prior request's bias when seq_id == 0 is reused) and returns.
    // token ids are NOT re-validated against vocab here (parse_request_controls already did that,
    // with the real vocab size in scope) -- the scatter kernel keeps a defensive bound check as a
    // backstop only. No-op if seq_id has no session entry.
    void set_logit_bias(uint64_t seq_id, const std::vector<std::pair<int, float>>& bias);
    // Constrained decoding: replace the session's whole logit_bias buffer with `bias` (cfg.vocab
    // floats: the request's own logit_bias plus a large negative value for every token the
    // constraint rules out), marking it set so the prefill seed applies it too. Called before every
    // sample of a constrained request; copies synchronously through a pinned staging buffer.
    void set_logit_bias_dense(uint64_t seq_id, const float* bias);

    // Token budget for KV allocation: prompt + decode headroom, capped at max_seq.
    static int session_token_budget(size_t prompt_len, int max_new, int max_seq);

    // Shared weights for DFlash draft (embed + lm_head come from target).
    const void* embed_weights() const;
    const void* lm_head_weights() const;
    int lm_head_quant_type() const;

    // Attach / detach a DFlash draft model (non-owning). nullptr clears.
    void set_dflash_draft(class DFlashDraftModel* draft);

    // DFlash: capture concat hidden states at target_layer_ids per forward step.
    // Disables CUDA-graph replay while enabled (capture needs eager layer outputs).
    // context_end > 0 sizes the context buffer for positions below it instead of max_seq -- at a
    // server context of 256K the unbounded buffer is tens of GB. On allocation failure the buffers
    // are left null; check dflash_context_buffer() / dflash_hidden_buffer(). Turning capture off
    // frees both.
    void set_dflash_capture(bool on, const std::vector<int>& target_layer_ids, int max_rows = 16,
                            int context_start = 0, int context_end = 0);
    void set_dflash_capture_row(int row);
    // Append the current capture-row into the growing context buffer at global_pos.
    void dflash_stash_capture(int global_pos);
    const void* dflash_hidden_buffer() const;   // scratch rows from last verify block
    const void* dflash_context_buffer() const;  // accumulated prefill/accept hiddens
    int dflash_hidden_row_stride() const;       // bf16 elems per row = n_capture * hidden
    int dflash_context_len() const;

    // Snapshot hybrid recurrent state for speculative rollback.
    void save_spec_snapshot();
    void restore_spec_snapshot();

    // Token-loop teacher-forced block verify with optional hidden capture.
    // Writes argmax at each position into out_argmax[0..n). Returns false on failure.
    bool verify_block(const int* token_ids, int n, int start_pos, int* out_argmax);

    // Build the DFlash verify replay graph without running it. Stream capture records kernels
    // instead of executing them, so this leaves model state untouched -- it exists purely to keep
    // ~4.9 ms of graph construction out of the decode loop, where it landed on decode step 2 and
    // made that token take twice as long as every other one.
    void dflash_warm_verify(int n, int start_pos);

    // (dual-GPU) Take the tp=2 DSpark verify scratch now on both ranks; false = it does not fit
    // (true when not tp). The DSpark loader calls it so an oversized --ctx fails at load.
    bool reserve_tp_verify();

    // Batched verify entry (may fall back to verify_block). Same contract as verify_block.
    bool batched_forward(const int* token_ids, int n, int start_pos, bool resume_gdn,
                         int* out_argmax, const void* dflash_capture_dst = nullptr);

private:
    void invalidate_decode_graph();
    // Frees every decode graph parked under a non-active session id (see the parking lot in
    // qwen35.cpp's Impl). Called by invalidate_decode_graph(), which is the "something global
    // changed" path, and as a size backstop.
    void drop_parked_decode_graphs();
    void dflash_maybe_capture_layer(int layer);
    // Depth-adaptive KV-split count for a given seqlen (32/128/160/256 tiers, GQA-8/hd256
    // occupancy correction). Shared by forward_token()'s normal per-token adaptation and
    // dflash_generate()'s one-time pre-capture initialization (see qwen35.cpp).
    int adaptive_nsplits_for(int seqlen) const;

    // (dual-GPU WP-9, tp>1 only; the tp=1 path never enters either of these.)
    // Tensor-parallel twin of forward_token: the group leader (rank 0) drives its own copy and
    // mirrors the call onto the other rank's model on its persistent worker thread; every rank runs
    // the same code against its own (sliced) weights on its own device/stream and combines the
    // per-rank partials through the GpuLink (the mirroring itself lives in forward_token, via
    // TpMirrorScope). Eager-only -- no CUDA-graph capture here. The
    // per-rank scratch (tp_xrow/tp_ar/..., see the Impl in qwen35.cpp) is
    // allocated by tp_attach on each rank's own device.
    int forward_token_tp(int token_id, int position, bool sample, float temperature,
                         unsigned long long seed, unsigned long long sample_step,
                         int top_k, float top_p,
                         float presence_penalty, float frequency_penalty);
    // Tensor-parallel twin of the packed continuous-batch decode entry: a per-row loop that runs
    // the single-row windowed path (forward_token_tp) once per row (design item 9). The rows-
    // kernels bit-identity invariant makes this correctness-preserving; the packed-kernel
    // throughput a split tp>1 rank cannot use is the accepted perf regression (M2 gate is
    // code-complete + in-regime unit test). The leader activates each row's session before its
    // call (the single-row path keys its per-rank state off the model's active session); the
    // peer replays the whole decode_packed call (mirrored in decode_packed), so it activates the
    // same sessions in the same order. Plain greedy only, exactly like decode_packed: out_sampled holds the argmax.
    bool decode_packed_tp(const int* tokens, const int* positions, const uint64_t* seq_ids,
                          int n, int* out_sampled);

    // (dual-GPU WP-9) tp>1 per-layer helpers: the real per-rank partial bodies (see the locked
    // design in wp-i2-brief.md). Each one computes this rank's partial into the tp_ar row --
    // GDN: windowed qkv GEMV + rank-local conv/recurrence/norm + K-windowed ssm_out; attn:
    // N-windowed q/k/v GEMVs + rank-local QK-norm/RoPE/KV-append/flash-decode + K-windowed
    // o_proj; FFN: N-windowed gate/up + K-windowed down -- which the layer loop's AR-A/AR-B
    // all-reduces combine across ranks. forward_token_tp calls them for every layer.
    void tp_attn_layer_tp(int l, uint16_t* x, const uint16_t* xn);
    void tp_gdn_layer_tp(int l, uint16_t* x, const uint16_t* xn);
    void tp_ffn_layer_tp(int l, const uint16_t* xn, uint16_t* out);
    // One 2-rank all-reduce (sum) of a bf16 row. Issued only by the group leader (rank 0):
    // a single GpuLink::allreduce posts the reduce on BOTH ranks' streams, so the peer must
    // not issue a second one (this helper is a no-op on the peer). `is_xrow` selects which
    // peer row to name (xrow for the op-entry exchange, ar for the per-layer AR-A/AR-B).
    void tp_allreduce_row(uint16_t* row, size_t elems, bool is_xrow);
    // (dual-GPU) Decode epilogue: in-place f32 sum all-reduce of the [vocab] logits row (each
    // rank holds its vocab half, zeros elsewhere), leaving the full row on both ranks.
    void tp_allreduce_logits();
    // (dual-GPU WP-12) DSpark batched verify at tp=2: n consecutive rows of the active sequence,
    // per-row argmax bit-identical to forward_token_tp; returns the accepted-prefix length after
    // committing exactly those rows' GDN state, or -1 when declined (nothing changed).
    int verify_rows_tp(const int* ids, int n, int start_pos, void* capture_dst, int* out_argmax);
    int tp_rows_forward(const int* ids, int n, int start_pos, const int* row_pos,
                        const uint64_t* row_seq, void* capture_dst, int* out_argmax);
    bool tp_verify_alloc();
    void reserve_tp_verify_local(bool* ok);
    // (dual-GPU) Rank-1 mirroring: the peer model to replay a state-changing public call on, or
    // null when this is not the attached group leader or the call is nested inside an already
    // mirrored one (only the outermost call mirrors; see TpMirrorScope in qwen35.cpp).
    Qwen35Model* tp_mirror_peer() const;

    struct Impl;
    Impl* p_;
};

// (dual-GPU S7a-1) Prefill-side tensor-parallel helper, process-scope (not a method: the caller
// is the prefill translation unit, which hands over its per-pass partial buffer). After the
// K-compact o_proj GEMM each rank's prefill pass calls this with its [N][H] bf16 partial; the
// rank on the link's device_a end (the leader) waits for the peer's registration, then posts the
// single GpuLink::allreduce that sums in place (in==out) on both ranks' streams; the peer waits
// for that post before enqueuing anything that consumes the sum. No attached link
// (a tp=1 process) makes this a no-op, exactly like tp_allreduce_row for a peer rank.
void tp_prefill_allreduce_bf16(void* in_out, size_t elems);
// (dual-GPU B1) The same all-reduce posted on a per-rank side stream (copy engines run while the
// compute stream continues); the covered rows are valid on the compute stream only after
// tp_prefill_allreduce_join(). Not tp: no-op.
// Returns a ticket (-1 when not tp): tp_prefill_allreduce_wait(ticket) orders the compute stream
// after that op (and every op posted before it); join waits for all of them.
int tp_prefill_allreduce_bf16_async(void* in_out, size_t elems);
void tp_prefill_allreduce_wait(int ticket);
void tp_prefill_allreduce_join();
// (dual-GPU) Both ranks of a mirrored pass pass their own value at the same point; both get the
// minimum (tp=1: v). For rank-local, memory-driven choices that must not diverge.
int tp_prefill_agree_min(int v);
// Bitwise-AND twin: each bit is an arm a rank can run; both get the set BOTH can run (tp=1: v).
int tp_prefill_agree_and(int v);
// Row-wise argmax over the vocab-split head: each rank passes its half's best value and local
// index per row (n <= 64); both get the global vocab index (ties to rank 0, the lower half).
void tp_exchange_argmax(const float* val, const int* idx, int n, int rows_per_rank, int* out);

// f32 twin of tp_prefill_allreduce_bf16 (same rendezvous): the prefill seed's zero-padded
// [vocab] logits row, summed in place so both ranks hold the full row.
void tp_prefill_allreduce_f32(float* in_out, size_t elems);

} // namespace sparkinfer
