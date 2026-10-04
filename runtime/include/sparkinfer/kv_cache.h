#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace sparkinfer {

enum class KVLayout {
    PAGED,       // PagedAttention-style block allocation
    CONTIGUOUS,  // flat contiguous (single sequence)
    COMPRESSED,  // quantized / compressed KV (future)
};

struct KVCacheConfig {
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int block_size = 16;        // tokens per page block
    KVLayout layout = KVLayout::PAGED;
    bool fp8_kv = false;        // FP8 KV cache compression
    bool int8_kv = false;       // int8 (Q8-style) KV cache; halves the long-context KV read
    // KV element format (see KVDtype below). -1 = derive from int8_kv (the old switch, unchanged).
    // KV_FP8 shares the int8 layout byte for byte (one byte per element + one fp16 scale per
    // (token, kv_head) head vector) with e4m3 codes instead of int8 ones. KV_NVFP4 packs each head
    // vector as head_dim/2 bytes of e2m1 nibbles followed by head_dim/16 e4m3 block scales
    // (9/16 byte per element), with the per-(token, kv_head) fp16 scale as the second-level scale.
    int kv_dtype = -1;
    // Hybrid stacks (Qwen3.5/3.6/3.8) give paged KV to the FULL-ATTENTION layers only -- the
    // Gated-DeltaNet layers carry a recurrent state instead and never touch these pools. Sizing
    // the pool for every layer therefore over-allocates by n_layers/n_attn_layers (4x on
    // Qwen3.8-27B: 16 attention layers of 64). layer_slot maps a layer index to its pool slot,
    // -1 for layers with no KV; empty = identity (every layer has a slot, the old behaviour).
    std::vector<int> layer_slot;

    // SLIDING-WINDOW SLOTS. A layer whose attention only ever reads the last `window_tokens`
    // tokens does not need a full-context slice: it needs the window, plus whatever one prefill
    // pass appends before that pass's attention runs (the pass writes every token of the pass
    // first, and its EARLIEST query still reads `window_tokens` back from itself). Those slots get
    // a private per-sequence RING of window_tokens + window_pass_tokens instead, which on Muse
    // Glimmer (39 of 52 layers windowed at 2048) is where most of the pool went.
    //
    // Capacity is deliberately unchanged: total_blocks/max_blocks_per_seq still come out of
    // pool_bytes as if every slot were full-context, so num_total_blocks() keeps meaning what
    // every caller already reads it as. The windowed slices simply allocate less of the pool, and
    // the difference shows up as free VRAM.
    //
    // Two things a ring cannot do, so both are refused rather than silently served:
    //   - CROSS-SEQUENCE PREFIX SHARING. A ring slot is private to one sequence; a shared prefix
    //     block would alias. prefix_sharing_supported() says so and allocate_with_prefix()
    //     refuses, so the engine recomputes instead of reading someone else's window.
    //   - A PREFILL PASS LONGER THAN window_pass_tokens. window_pass_limit() publishes the bound;
    //     a caller past it must chunk or decline. (Re-using ONE sequence's own session across
    //     requests is fine: its ring holds that sequence's last window, which is all a windowed
    //     layer ever reads.)
    // window_tokens = 0 (the default) keeps every slot full-context -- the old layout, byte for
    // byte. SPARKINFER_KV_SWA_CAP=0 also forces it off, for an A/B out of one binary.
    int window_tokens = 0;        // sliding window, in tokens (0 = no windowed slots)
    int window_pass_tokens = 0;   // longest prefill pass a windowed slot must survive
    int max_seq_tokens = 0;       // per-sequence context the pool was sized for (required to cap)
    std::vector<char> slot_windowed;   // per POOL SLOT: 1 = windowed ring, 0 = full context

    // TP KV-HEAD WINDOW (dual-gpu WP-7, the 2+2 split). This pool covers only a slice of the
    // model's KV heads: [kv_head_start, kv_head_start + kv_head_count). count 0 = all heads --
    // the tp=1 default, a single pool of the whole head set, byte-identical to the old layout.
    // The window mirrors the tp table's K/V convention (tp_layout.hpp head_window: contiguous
    // window per rank, last rank takes any remainder), so pool head j == model KV head
    // kv_head_start + j and lines up with this rank's k_proj/v_proj weights. num_kv_heads above
    // stays the MODEL's total head count; the window carves what this pool (this device) holds.
    int kv_head_start = 0;
    int kv_head_count = 0;
};

// Layer -> pool-slot map for the hybrid interval rule these models share: with
// full_attn_interval = k, layer L is a linear (Gated-DeltaNet) layer -- which carries a recurrent
// state and never touches paged KV -- unless (L+1) % k == 0. Returns an empty map for non-hybrid
// models, which KVCacheManager reads as the identity (every layer gets a slot).
enum KVDtype : int { KV_BF16 = 0, KV_INT8 = 1, KV_FP8 = 2, KV_NVFP4 = 3 };
// Bytes of pool storage for `elems` KV elements in format fmt (elems a multiple of 16).
inline size_t kv_dtype_bytes(int fmt, size_t elems) {
    switch (fmt) {
        case KV_INT8: case KV_FP8: return elems;
        case KV_NVFP4: return elems / 16 * 9;
        default: return elems * 2;
    }
}
inline const char* kv_dtype_name(int fmt) {
    switch (fmt) { case KV_INT8: return "int8"; case KV_FP8: return "fp8"; case KV_NVFP4: return "nvfp4"; default: return "bf16"; }
}

inline std::vector<int> hybrid_kv_layer_slots(int num_layers, bool hybrid, int full_attn_interval) {
    if (!hybrid || full_attn_interval <= 0 || num_layers <= 0) return {};
    std::vector<int> slot((size_t)num_layers, -1);
    int n = 0;
    for (int L = 0; L < num_layers; ++L)
        if (((L + 1) % full_attn_interval) == 0) slot[(size_t)L] = n++;
    if (n == 0 || n == num_layers) return {};   // nothing to compact
    return slot;
}
inline int kv_slot_count(const std::vector<int>& slot, int num_layers) {
    if (slot.empty()) return num_layers;
    int n = 0;
    for (int v : slot) if (v + 1 > n) n = v + 1;
    return n;
}

// Per-slot windowed flags from the model's own per-layer contract (Qwen35Config::swa_layers:
// true = slides over `sliding_window` tokens, false = full/global attention, which must keep a
// full-context slice). `layer_slot` is the same map KVCacheConfig carries (empty = identity).
// Returns an empty vector when nothing would be capped -- no windowed layer, or every one of them
// windowed, since a single group is the cheaper layout either way -- which KVCacheManager reads as
// "no windowed slots".
inline std::vector<char> swa_slot_flags(int num_layers, const std::vector<int>& layer_slot,
                                        const std::vector<bool>& swa_layers) {
    if (num_layers <= 0 || (int)swa_layers.size() != num_layers) return {};
    std::vector<char> win((size_t)kv_slot_count(layer_slot, num_layers), 0);
    int n = 0;
    for (int L = 0; L < num_layers; ++L) {
        const int s = layer_slot.empty() ? L : (L < (int)layer_slot.size() ? layer_slot[(size_t)L] : -1);
        if (s < 0 || s >= (int)win.size()) continue;
        if (swa_layers[(size_t)L]) { win[(size_t)s] = 1; ++n; }
    }
    if (n == 0 || n == (int)win.size()) return {};   // all or nothing to cap: keep one layout
    return win;
}

// GPU-side KV block pool.
// Manages a fixed-size pool of blocks and maps sequence positions
// to physical block indices via a per-sequence block table.
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& cfg, size_t pool_bytes);
    ~KVCacheManager();

    // Allocate physical blocks for a sequence (grows if already allocated).
    // Returns false if OOM. Idempotent when num_tokens fits existing allocation.
    bool allocate(uint64_t seq_id, int num_tokens);

    // Tokens already covered by the current block allocation for seq_id (0 if none).
    int allocated_tokens(uint64_t seq_id) const;

    // Free all blocks owned by a sequence
    void free(uint64_t seq_id);

    // Speculative decode: truncate a sequence to its first keep_blocks logical blocks
    // (returns false if seq_id is unknown). Freed physical blocks go back to the pool.
    bool truncate_blocks(uint64_t seq_id, int keep_blocks);

    // Number of logical KV blocks currently allocated for seq_id (0 if none).
    int num_blocks(uint64_t seq_id) const;

    // Returns device pointer to the block table for seq_id
    // Shape: [num_layers, max_blocks_per_seq]
    int* block_table(uint64_t seq_id) const;

    // Physical block ids owned by seq_id, in logical order (index 0 = the sequence's first
    // block). Empty if seq_id is unknown. Physical blocks are free-list allocated and not
    // guaranteed contiguous, so a caller walking (layer, physical_block) byte ranges for bulk
    // copy-out/copy-in (e.g. an external KV cache tier) needs this rather than assuming a
    // contiguous span. Returns a reference into internal state -- valid only until the next
    // allocate()/free()/truncate_blocks() call for this seq_id.
    const std::vector<int>& physical_block_ids(uint64_t seq_id) const;

    // PREFIX SHARING. Physical blocks are reference-counted, so one block can have several holders
    // at once: the sequence that wrote it, a prefix-cache entry that retained it, and any later
    // sequence that starts from that prefix. A block returns to the pool only when its last holder
    // lets go. Shared blocks are READ-ONLY by contract -- a sequence built on a prefix starts
    // writing at prefix.size() * block_size() -- so no copy-on-write is needed.
    //
    // Retain seq_id's first n_blocks physical blocks (+1 each) and return them in logical order, for
    // a holder that is not a sequence (it takes no block-table row). Empty, retaining nothing, when
    // seq_id is unknown or has fewer blocks.
    std::vector<int> retain_prefix_blocks(uint64_t seq_id, int n_blocks);
    // Drop one reference from each block; any that reach zero go back to the pool.
    void release_blocks(const std::vector<int>& physical_ids);
    // +1 on blocks that already have a holder (a second holder of a cached prefix's blocks).
    // False, retaining nothing, if any of them is free.
    bool retain_blocks(const std::vector<int>& physical_ids);
    // Take n free blocks for a holder that is not a sequence (the prefix cache restoring a prefix
    // from host memory): one reference each, in the order the pool hands them out, so a mirrored
    // manager with the same history returns the same ids. Empty if fewer than n are free or the
    // pool has windowed slices.
    std::vector<int> allocate_blocks(int n);
    // Elements of one block in one slot's sub-pool (block_size * kv heads here * head_dim).
    size_t block_elems() const;
    // allocate() for a seq_id that has no blocks yet, whose first logical blocks ARE `prefix`
    // (shared, +1 each); only the remainder of num_tokens comes from the pool. False, changing
    // nothing, when seq_id already has blocks, a prefix block is not live, prefix is longer than
    // num_tokens needs, or the pool cannot cover the rest.
    bool allocate_with_prefix(uint64_t seq_id, const std::vector<int>& prefix, int num_tokens);

    // Device pointers to K and V storage pools (base = the first slot).
    // Per-layer pointer = (bf16*)k_pool() + layer_base_elems(layer) -- NOT layer * layer_stride,
    // which is only the same thing while the pool is uncompacted.
    void* k_pool() const;
    void* v_pool() const;
    size_t layer_stride_elems() const;   // elements between consecutive slots' sub-pools
    int kv_slots() const;                // slots actually allocated (== num_layers uncompacted)
    // Element offset of `layer`'s slice. A layer with no KV lands on a trap slice (and warns once)
    // rather than aliasing a real layer's slot.
    size_t layer_base_elems(int layer) const;
    size_t scale_layer_base_elems(int layer) const;

    // int8 KV (Q8-style int8 + per-(token,kv_head) fp16 scale). When int8_kv(), k_pool/v_pool hold
    // int8 and k_scale_pool/v_scale_pool hold one __half scale per head vector.
    // Per-layer scale pointer = (__half*)k_scale_pool() + scale_layer_base_elems(layer).
    // int8_kv() stays true ONLY for the int8 format, so every int8-only path keeps its gate.
    // kv_dtype() is the format; quant_kv() is "has the scale pools" (int8, fp8, nvfp4);
    // kv_bytes(elems) converts a layer_base_elems()/element offset into a byte offset.
    bool int8_kv() const;
    int kv_dtype() const;
    bool quant_kv() const { return kv_dtype() != KV_BF16; }
    size_t kv_bytes(size_t elems) const { return kv_dtype_bytes(kv_dtype(), elems); }
    void* k_scale_pool() const;
    void* v_scale_pool() const;
    size_t scale_layer_stride_elems() const;

    // WINDOWED SLOTS (see KVCacheConfig::window_tokens). windowed() is false when this pool has
    // none, and then every accessor below behaves exactly as the full-context ones.
    bool windowed() const;
    // The ring table for seq_id: [max_blocks_per_seq] entries, logical block i -> the physical
    // block holding i's window slot, i.e. ring[i % ring_blocks]. A windowed layer passes THIS to
    // the same append/attention kernels the full layers drive with block_table(), so no kernel
    // learns about rings -- the repeated rows carry the wrap. Null for an unknown sequence, and
    // block_table() itself when this pool has no windowed slots.
    int* block_table_win(uint64_t seq_id) const;
    // Longest prefill pass a windowed slot can serve (window_pass_tokens), or INT_MAX uncapped.
    int window_pass_limit() const;
    // False when a ring is in play: cross-sequence prefix blocks would alias (see the header note).
    bool prefix_sharing_supported() const;

    int block_size() const;
    int max_blocks_per_seq() const;
    int num_free_blocks() const;
    int num_total_blocks() const;

    // TP KV-head window actually in force for this pool (see KVCacheConfig): count 0 means all
    // heads. An out-of-range window is clamped back to all heads (and warned once), never to a
    // smaller pool, so a mis-sized caller degrades to the safe layout instead of a silent one.
    int kv_head_start() const;
    int kv_head_count() const;
    // Per-physical-pool refcounts for a logical block: one id names ONE block in the K pool AND
    // one in the V pool (the shared logical numbering), and holders are counted per pool. -1 when
    // the id is out of range; a block is on the free list iff both pools show 0.
    int block_ref_k(int block) const;
    int block_ref_v(int block) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sparkinfer
