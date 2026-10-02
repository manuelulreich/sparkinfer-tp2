#pragma once
// DFlash block-diffusion draft model for Qwen3.6-35B-A3B.
// Loads official z-lab BF16 safetensors; reuses target embed + lm_head.

#include <cstdint>
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

    // YaRN rotary scaling (RadixArk/Qwen3.8-27B-DSpark ships rope_type: "yarn"). factor <= 1
    // disables it and the draft uses plain theta^(-2i/d), so existing checkpoints are unaffected.
    // These cannot be folded into rope_theta: YaRN's NTK-by-parts ramp scales each frequency band
    // differently -- measured on this checkpoint, 36 of 64 bands are divided by `factor` and only
    // 15 are untouched -- and it additionally scales cos/sin magnitude by 0.1*ln(factor)+1.
    float yarn_factor = 0.f;          // "factor" (32.0 for DSpark); <= 1 => no YaRN
    int   yarn_orig_max_pos = 0;      // "original_max_position_embeddings" (8192)
    float yarn_beta_fast = 32.f;
    float yarn_beta_slow = 1.f;
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
    int kv_state_create(int capacity);
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

    // Apply target lm_head to last forward's hidden states; writes device logits [block, vocab]
    // and host argmax. Called internally by forward_block; exposed for debugging.
    const float* last_logits() const;

private:
    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
