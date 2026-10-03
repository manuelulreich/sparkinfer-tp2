#pragma once
// DFlash block-diffusion draft model for Qwen3.6-35B-A3B.
// Loads official z-lab BF16 safetensors; reuses target embed + lm_head.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace sparkinfer {

struct DFlashDraftConfig {
    int hidden = 2048;
    int intermediate = 6144;
    int n_layers = 6;
    int n_q_heads = 32;
    int n_kv_heads = 8;
    int head_dim = 128;
    int block_size = 16;
    int mask_token_id = 248077;
    int vocab = 248320;
    float rms_eps = 1e-6f;
    float rope_theta = 10000000.f;
    int sliding_window = 4096;
    int max_seq = 8192;
    std::vector<int> target_layer_ids = {1, 6, 11, 16, 22, 27, 32, 37};
    // Per-layer: true = sliding_attention (window), false = full_attention.
    std::vector<bool> sliding_layers;
    // false (default) = existing Qwen3.6 draft behavior: NeoX split-half RoPE pairing
    // (rotate h[i] with h[i+half]), via launch_rms_heads_rope. true = consecutive-pair
    // ("normal"/LLAMA_ROPE_TYPE_NORM) pairing, via launch_rms_heads_rope_normal -- set for the
    // Muse Glimmer draft (see museglimmer_dflash_config_from_gguf). This mirrors a fix already
    // made and validated on the Muse Glimmer TARGET model; on the draft it is an informed but
    // UNVERIFIED carry-over (no DFlash accuracy/SPEC_AGREE evaluation has run yet) -- if Muse
    // Glimmer draft proposals look wrong, check this flag first.
    bool rope_normal = false;
    // (dual-GPU) Build the quantized weight copies during load, layer by layer, and drop each
    // layer's dead bf16 copies straight away -- the load peak is then one layer's bf16 above the
    // steady state instead of every layer's. For a server that would call ensure_quant() at load
    // anyway (tp>1); the default keeps the deferred build (see Impl::pending_quant).
    bool eager_quant = false;
    // (dual-GPU) Tensor-parallel slice this instance loads: rank r of tp_size holds query heads,
    // KV heads and FFN columns [r/tp_size, (r+1)/tp_size) of every layer. After load(),
    // n_q_heads / n_kv_heads / intermediate hold this rank's LOCAL counts. Rank 0 also holds fc,
    // the head and the Markov/confidence heads; rank 1 only its half of the layers. 1 = whole.
    int tp_rank = 0;
    int tp_size = 1;

    // YaRN rotary scaling (RadixArk/Qwen3.8-27B-DSpark ships rope_type: "yarn"). factor <= 1
    // disables it and the draft uses plain theta^(-2i/d), so existing checkpoints are unaffected.
    // These cannot be folded into rope_theta: YaRN's NTK-by-parts ramp scales each frequency band
    // differently -- measured on this checkpoint, 36 of 64 bands are divided by `factor` and only
    // 15 are untouched -- and it additionally scales cos/sin magnitude by 0.1*ln(factor)+1.
    float yarn_factor = 0.f;          // "factor" (32.0 for DSpark); <= 1 => no YaRN
    int   yarn_orig_max_pos = 0;      // "original_max_position_embeddings" (8192)
    float yarn_beta_fast = 32.f;
    float yarn_beta_slow = 1.f;

    // DFlash2 (config.json "architectures": ["DFlash2DraftModel"]). Each layer wraps its attention
    // and its MLP in a grouped convolution along the block (conv_taps taps over the previous
    // positions of the same block, one coefficient per conv_group channels and row, from a
    // per-row kernel_projection), and the proposals come from a candidate selector instead of the
    // head's argmax: the head's top selector_top_k per row, then a walk over the rows that scores
    // each candidate given the previous pick (rank selector_rank codebooks). Block rows 1..depth
    // back proposals 1..depth (no row shift), so a block backs block_size - 1 proposals.
    bool dflash2 = false;
    int conv_taps = 2;
    int conv_group = 16;
    int selector_rank = 256;
    int selector_top_k = 16;
    float output_multiplier = 1.f;   // dflash_config "output_multiplier"
    float logit_softcap = 0.f;       // dflash_config "final_logit_softcapping" (<= 0: none)
    float embed_scale = 1.f;         // dflash_config "input_embedding_scale"
    int is_causal = -1;              // top-level "is_causal" (-1: absent); DFlash2 layers follow it
    // Proposal depth the checkpoint asks for at every context (0: the caller's own ladder).
    // DFlash2 drafts its whole trained block (block_size - 1), as HyperQwen runs it.
    int spec_depth = 0;

    // Most proposals one block can back.
    int max_proposals() const { return dflash2 ? block_size - 1 : block_size; }
};

class DFlashDraftModel {
public:
    explicit DFlashDraftModel(const DFlashDraftConfig& cfg);
    ~DFlashDraftModel();

    // Load model.safetensors (+ optional config.json) from a HF draft directory.
    bool load(const std::string& dir);

    // Load a GGUF-packed draft checkpoint (e.g. Muse Glimmer's dflash-kquant.gguf). Self-contained
    // like load(): opens the file, derives config from its metadata (see
    // museglimmer_dflash_config_from_gguf in runtime/examples/dflash_gguf_config.h), and uploads
    // dequantized weights. Does not touch/alter load()'s HF-safetensors path.
    bool load_gguf(const std::string& path);

    const DFlashDraftConfig& config() const;

    // Bind shared target embed / lm_head (non-owning device pointers).
    // (plan 07, B6) The target's NVFP4 lm_head copy (this rank's rows, row-major E2M1 + the
    // CUTLASS SFB scales + alpha), or null. With it the head over the draft vocabulary is one
    // tensor-core GEMM instead of the Q4_K multirow GEMV, whose cost grows with the rows (0.6 ms
    // at 6 rows, 2.2 ms at 16). The target clears it before it frees the copy.
    void set_head_fp4(const void* w, const void* sf, float alpha);

    // (dual-GPU) DFlash2 scores its candidates over the whole vocabulary: each card takes its
    // top-k of its own vocab half and the two are merged (tp_exchange_topk). The second card's
    // half of the target head, read at every block (the target may release its FP4 copy between
    // blocks): `rows` vocab rows, the last ones of the vocabulary. Ignored by other drafts and
    // without an attached peer.
    struct HeadRef {
        const void* lm_head = nullptr;
        int lm_head_type = 0;
        const void* fp4_w = nullptr;
        const void* fp4_sf = nullptr;
        float fp4_alpha = 1.f;
    };
    void set_peer_head(std::function<HeadRef()> src, int rows);

    // (dual-GPU) fc split by input columns: each rank holds the fc columns of its own half of
    // every captured layer, and the target captures that half on each card
    // (Qwen35Model::set_dflash_capture_split). True on a split draft unless
    // SPARKINFER_DFLASH_FC_SPLIT=0. `map` turns a rank-0 capture pointer into rank 1's twin; the
    // target sets it before the first forward.
    bool fc_split() const;
    void set_peer_hidden_map(std::function<const void*(const void*)> map);

    void set_shared_weights(const void* embed_bf16_or_null,
                            const void* lm_head,
                            int lm_head_type,
                            int vocab,
                            int hidden);

    // (dual-GPU tp=2) The shared embedding is vocab-split: set_shared_weights' table holds only
    // rows [0, local_rows) and the rest are rows [0, vocab - local_rows) of `hi_table` on device
    // `hi_device`. local_rows = 0 (the default) means the shared table is whole.
    void set_embed_split(int local_rows, const void* hi_table, int hi_device);

    // Reset draft KV length to 0.
    void reset();

    // Crop draft KV to the first `keep` tokens (speculative accept boundary).
    void crop(int keep);

    int seq_len() const;

    // (dual-GPU) Per-session draft KV caches, for drafting several sequences in turn. create
    // allocates one holding `capacity` positions (clamped to max_seq; -1 = out of memory);
    // select makes it the cache reset/crop/seq_len/forward_block act on (-1 = the built-in one);
    // free releases it (selecting the built-in cache if it was active).
    //
    // A state slides: when a block does not fit, forward_block keeps the newest
    // kv_slide_keep() positions, moves them to the front, and windows the attention (positions
    // below them are gone). So a state of kv_slide_capacity() positions serves any context.
    int kv_state_create(int capacity);
    // Positions a sliding state needs to behave exactly as an unbounded one for every context the
    // draft attends unwindowed (SPARKINFER_DSPARK_KV_CAP, default 12288 + 2 x block; DFlash2,
    // windowed everywhere: kv_slide_keep() + 4 x block).
    int kv_slide_capacity() const;
    // Positions kept when a state slides (SPARKINFER_DSPARK_KV_KEEP, default 4096).
    int kv_slide_keep() const;

    // (plan 06, W4/W5) Positions of the selected state on the host, kept with a prefix-cache entry
    // so a later request starting from that prefix restores the draft's context instead of
    // starting it empty.
    struct KvSnapshot {
        std::shared_ptr<void> host;   // pinned; per layer, K rows then V rows (this rank's heads)
        int lo = 0, hi = 0;           // positions [lo, hi)
        size_t bytes = 0;
        std::shared_ptr<KvSnapshot> peer;   // rank 1's heads (split draft)
        size_t total_bytes() const { return bytes + (peer ? peer->total_bytes() : 0); }
    };
    // Copy positions [lo, hi) of the selected state. false: a row is not held in every layer (see
    // kv_valid_lo), or no pinned memory.
    bool kv_snapshot(int lo, int hi, KvSnapshot& out);
    // Start the selected state at `seq_len`, holding positions [from, seq_len) from `snap` (which
    // must cover them and end at seq_len), or nothing with snap == nullptr (from == seq_len).
    // Positions below `from` are gone: the full-attention layer windows unless from == 0.
    // false (state unchanged): it does not fit the state's capacity.
    bool kv_start_at(int from, int seq_len, const KvSnapshot* snap);
    // Lowest position the selected state holds in every layer (rows below were never projected, or
    // slid out).
    int kv_valid_lo() const;
    bool kv_state_select(int id);
    void kv_state_free(int id);

    // One parallel block forward.
    //   target_hidden: [ctx_len, n_capture * hidden] bf16 (concat features before fc)
    //   noise_ids:     [block_size] token ids (mask-filled block; position 0 = seed)
    //   pos0:          absolute position of noise_ids[0]
    //   out_argmax:    [block_size] host argmax (only [1..] are draft proposals; [0] unused)
    // Returns false on failure.
    // Build the quantized weight copies now rather than on the first forward_block, so a caller
    // that primes the draft outside a timed region does not pay for it inside one. Idempotent.
    void ensure_quant();
    // After ensure_quant(): false when a quantized copy could not be allocated (out of memory).
    bool quant_ok() const;

    //   proposals:     how many rows after the seed to score (0 = the built-in default). The
    //                  verifier picks this by context length, so the draft has to be told rather
    //                  than deciding for itself, or the two disagree on how long a block is.
    //   out_confidence: optional, [1..proposals] host logits from DSpark's confidence head (raw
    //                  logit, not sigmoid'd -- sigmoid on the caller side if a probability is
    //                  needed). Left untouched (whatever the caller passed in) for checkpoints
    //                  without a confidence head, or when nullptr.
    bool forward_block(const void* target_hidden, int ctx_len,
                       const int* noise_ids, int pos0,
                       int* out_argmax, cudaStream_t stream = nullptr,
                       int proposals = 0, float* out_confidence = nullptr,
                       int target_hidden_start = 0);

    // (dual-GPU) One block for each of n sessions (each on its own kv_state), projections and
    // head batched across them; falls back to forward_block per session where it cannot.
    struct DraftSeg {
        int state = -1;
        const void* target_hidden = nullptr;
        int ctx_len = 0;
        int target_hidden_start = 0;
        const int* ids = nullptr;   // [block_size]
        int pos0 = 0;
        int* out_argmax = nullptr;  // [block_size + 1]; [1..proposals] written
    };
    bool forward_blocks(int n, const DraftSeg* seg, int proposals, cudaStream_t stream = nullptr);

    // (dual-GPU) Pair this rank-0 draft with `peer`, the rank-1 slice loaded on `peer_device`.
    // From here on every forward and KV-state call on this instance runs on both cards: the peer's
    // half on the peer device's tp worker thread, the two halves summed through the target's
    // GpuLink after each layer's o_proj and down_proj. `stream` / `peer_stream` are the target's
    // own streams on the two cards -- the draft runs on them, so its link ops share one stream
    // order per card with the target's. False (nothing attached) if the peer's scratch does not fit.
    bool tp_attach(DFlashDraftModel* peer, int peer_device, cudaStream_t stream,
                   cudaStream_t peer_stream);

    // Apply target lm_head to last forward's hidden states; writes device logits [block, vocab]
    // and host argmax. Called internally by forward_block; exposed for debugging.
    const float* last_logits() const;

private:
    struct Impl;
    Impl* p_;
    // This rank's share of forward_block / forward_blocks (the public calls run it on both ranks
    // when a peer is attached). mode: -1 = choose the batched path here, 0 = one by one,
    // 1 = batched (the leader's choice, so both ranks take the same path).
    bool forward_block_body(const void* target_hidden, int ctx_len, const int* noise_ids, int pos0,
                            int* out_argmax, cudaStream_t stream, int proposals,
                            float* out_confidence, int target_hidden_start);
    bool forward_blocks_body(int n, const DraftSeg* seg, int proposals, cudaStream_t stream,
                             int mode);
    int kv_state_create_local(int capacity);
    bool kv_snapshot_local(int lo, int hi, KvSnapshot& out);
    bool kv_start_at_local(int from, int seq_len, const KvSnapshot* snap);
    void kv_state_free_local(int id);
};

} // namespace sparkinfer
