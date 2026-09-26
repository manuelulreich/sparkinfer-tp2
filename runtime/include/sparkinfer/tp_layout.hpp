// The per-tensor tensor-parallelism placement table (dual-GPU program, WP-6).
//
// WHAT THIS IS
//   A pure-C++ map from tensor name -> {which TP rank owns it, which axis it is
//   split on, and the element ranges that rank holds}, built from a
//   Qwen35Config and the number of effective devices. No CUDA, no runtime
//   linkage, no GPU: the whole module (and its CPU test,
//   runtime/tests/tp_layout_cpu_test.cpp) runs headless. ModelEngine::load()
//   builds one table per model load and publishes it process-wide
//   (set_process_table); the weight parsers (gguf.cpp / safetensors.cpp) and
//   the load-flow fit probes in model_engine.cpp consult it, and the
//   per-device allocation decisions in qwen35.cpp are the recorded handoff
//   consumers of the same placements.
//
// THE STABLE SPLIT CONVENTIONS (the single source of truth for all loaders)
//   * Rank order: rank r owns the r-th slice of n_ranks equal slices; the
//     FIRST effective device (eff[0]) gets the LOW half of every split
//     tensor, the second eff[1] the high half, and so on. Slice bounds are
//     n*r/R .. n*(r+1)/R, so n not divisible by R still partitions exactly
//     (last rank takes the remainder).
//   * Heads: q heads split into n_q/n_ranks contiguous head blocks (rank r:
//     heads [n_q*r/n_ranks, ...)); kv heads likewise over n_kv. The q->kv
//     pairing is GROUPED, never cyclic: kv head k serves q heads
//     [k*(n_q/n_kv), (k+1)*(n_q/n_kv)), so the low-half q rank always pairs
//     with the low-half kv rank and the two halves stay on the same device.
//   * GDN: the v heads split into n_v/n_ranks blocks (27B: v heads 0-23 ->
//     rank 0, 24-47 -> rank 1); the q and k heads travel with their v block
//     (gdn_qh_block convention: q/k head b belongs to v block b). GDN's
//     in_proj_qkv / conv1d are laid out [all q heads | all k heads | all v
//     heads], so one rank's slice is THREE column ranges, one per section.
//   * Axis convention: in-projections (q/k/v/gate, FFN gate/up) split on
//     their OUT-feature axis; out-projections (o, FFN down, GDN out) split
//     on their IN-feature axis (the head-dim concatenation); embedding and
//     lm_head split on the vocab (row) axis; expert pools split on the
//     EXPERT-index axis; 1-D per-v-head scalars (A_log, dt, beta, alpha)
//     split on the v-head axis. Norms, q/k_norm and the routers are
//     REPLICATED on every device.
//   * Per-convention axis orientation: GGUF (and flat .bin) store weights
//     ggml-order [in,out] with dims[0] fastest, so the out-feature axis is
//     dims[1] (Cols) and the in-feature axis is dims[0] (Rows); safetensors
//     stores row-major [out,in] with dims[0] slowest, so the same logical
//     split is Rows there. GDN conv1d is [taps, channels] in GGUF but
//     [channels, 1, taps] in HF, so its channel axis is Cols vs Rows.
//
// REFERENCE NUMBERS (27B = Qwen3.8-27B-NVFP4, the scored checkpoint; HF
// snapshot 5b7a687: 2 shards, 17,916,112,584 B total, mtp_num_hidden_layers
// 0)
//   vocab 248320, hidden 5120, 64 layers: 16 full-attn at (i+1)%4==0,
//   48 GDN. Full-attn n_q 24 / n_kv 4 (q:kv 6:1), head_dim 256, rope_dim
//   64 (partial_rotary 0.25, mrope sections [11,11,10], theta 1e7); the q
//   width 12288 = 2 x 6144 folds a sigmoid gate into every q head. GDN:
//   16 q / 16 k / 48 v heads, head_dim 128, conv kernel 4, qkvdim
//   10240 = 2048+2048+6144, gdn_qh_block. Dense SwiGLU FFN F 17408 on all
//   64 layers (no expert pools). 2,387 file tensors: 2,050
//   model.language_model.* (400 NVFP4 linears + 3 companions each = 1,200
//   companions, plus 450 bf16/f32) + 333 model.visual.* + 4 flat
//   lm_head.* (the 401st NVFP4 linear, stored flat in shard 2).
//   35B-A3B: 40 layers (10 full), n_experts 256 (rank split 128/128),
//   1 shared expert, per-expert F 768, GDN 16 q / 32 v heads (v block
//   size g_l = 2).
//
//   CLASS-AWARE per-tensor budget (replaces the flat 0.5625 B/weight
//   estimate, which mis-sizes NVFP4 by ~15%): an nvfp4 (r x c) weight
//   costs r*c/2 payload + r*((c+15)/16) scale bytes (the scales are
//   16-element blocks of the INNER dimension -- the file stores F8E4M3
//   shapes [rows, cols/16], NOT a single (r*c+15)/16 block stream) + 8
//   (two f32 companions: zero-point and dequant scalar); bf16 is
//   2.0 B/element, f32 scalars 4 B. For the 27B that partitions the file
//   exactly: S (split, language) 16,993,005,704 B, V (visual,
//   single-homed on rank 0) 921,460,192 B, R (replicated, per rank)
//   1,349,632 B, H (host f32) 297,056 B -- S+V+R+H is the file to the
//   byte. per-rank = S/2 + R = 8,497,852,484 B; rank_on_disk_bytes()
//   (the f32 companions kept whole per rank) = 8,497,854,088 B; the
//   rank-0 physical total = + V = 9,419,314,280 B; the per-card average
//   = 8,958,582,580 B = 8,543.57 MiB, i.e. +7.57 MiB over the G2 budget
//   table's 8,536 MiB/card (8,950,644,736 B). The CPU test recomputes all
//   of this from the table and prints the deltas.
//
// tp=1 INVARIANCE
//   Built with n_ranks <= 1 (or never set), EVERY name -- known or not --
//   resolves to {device 0, Axis::None, no ranges}: the whole tensor on
//   device 0, exactly what the unsplit loaders do today. No split, no
//   replication, no warnings, no changed allocation or ordering. A loader
//   consulting the table therefore cannot observe any difference at tp=1.
//
// OUT-OF-PATTERN NAMES (two-tier fallback)
//   At n_ranks == 1 every unknown name collapses to whole-on-device-0
//   (the tp=1 invariance holds for unknowns too). At n_ranks > 1 a name
//   starting "model.visual." is single-homed on rank 0 -- the vision tower
//   is not tensor-parallel and its ~921 MB sits on one card (the V budget
//   above) -- while every other unknown (MTP-era tensors, and the
//   HF-convention MoE expert pools, whose compressed-tensors names are
//   intentionally absent from the table) falls to the REPLICATED tier
//   (device -1, whole tensor on every device) so nothing is silently
//   dropped; known() reports false so the caller warns once per name,
//   which is the safe behaviour.

#pragma once

#include "models/qwen_config.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace sparkinfer {
// The weight-file types the loader bridges (end of file) take by const ref;
// their headers are intentionally NOT included here so this table stays
// includable from pure-C++ TUs (the CPU test among them) without pulling the
// loaders in.
class GGUF;
class SafeTensorsModel;

namespace tp {

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

enum class Axis {
    None,     // no split: whole tensor on the owning rank (or replicated)
    Rows,     // split along the tensor's first logical dimension
    Cols,    // split along its second logical dimension
    OneD,     // a 1-D tensor's only dimension
    Experts,  // a 3-D expert pool, split along the expert-index axis
};

// A half-open element interval along the split axis.
struct Range {
    size_t begin = 0;
    size_t len = 0;
};

// One rank's share of one tensor.
//
//   device: -1 = replicated -- the tensor is held WHOLE on every effective
//           device (a full copy each). Otherwise a rank index into the
//           effective device list (the loader maps rank r to eff[r]).
//   axis:   where the split lives (None => whole on the owning rank).
//   ranges: the element intervals this rank owns along `axis`; empty when
//           axis == None (whole tensor). A GDN qkv/conv tensor yields up to
//           three ranges per rank (the q | k | v sections each own one
//           sub-interval of the shared channel axis).
//   denom:  the full extent of the split axis in elements (0 when unsplittable).
struct Placement {
    int device = 0;
    Axis axis = Axis::None;
    std::vector<Range> ranges;
    size_t denom = 0;

    // Elements owned along the split axis; 0 when not split. Only meaningful
    // with axis != None; for a whole (axis == None) tensor the owner holds
    // everything and the caller uses full_elements() for the extent.
    size_t split_elements() const {
        size_t s = 0;
        for (const Range& r : ranges) s += r.len;
        return s;
    }
    // Fraction of the split axis this rank owns; 1.0 when not split.
    double fraction() const {
        return denom == 0 ? 1.0 : double(split_elements()) / double(denom);
    }
};

// Which naming convention the table speaks (see the axis-convention note in
// the file header for why the same logical split sits on different physical
// axes per convention).
enum class Conv { Gguf, Hf, Flat };

// The logical tensor classes a Qwen3.5/3.8 loader can meet.
enum class Kind {
    // top level
    Embd,      // token_embd / embed_tokens
    OutNorm,   // output_norm / final_norm / norm
    LmHead,    // output / lm_head
    // full attention
    Q,         // attn_q / q_proj / wq      ([q_out, in] out-feature axis split)
    AttnGate,  // muse's separate attn output gate (gate tensor)
    K,         // attn_k / k_proj / wk
    V,         // attn_v / v_proj / wv
    O,         // attn_output / o_proj / wo (in-feature axis = q heads)
    QNorm,     // attn_q_norm / q_norm
    KNorm,     // attn_k_norm / k_norm
    // block norms (1-D); slots: 0 input, 1 post-attn, 2 pre-ffn, 3 post-ffn
    Norm,
    // Gated DeltaNet
    GdnQKv,   // attn_qkv / in_proj_qkv   (q | k | v sections)
    GdnGate,  // attn_gate / in_proj_z    (v-head gate)
    GdnOut,   // ssm_out / out_proj
    GdnConv,  // ssm_conv1d / conv1d
    GdnA,     // ssm_a / A_log            (per v head)
    GdnDt,    // ssm_dt / dt_bias         (per v head)
    GdnAlpha, // ssm_alpha / in_proj_a
    GdnBeta,  // ssm_beta / in_proj_b
    GdnNorm,  // ssm_norm / linear_attn.norm
    // dense / shared FFN
    FfnGate,   // ffn_gate / gate_proj / gate
    FfnUp,     // ffn_up / up_proj / up
    FfnDown,   // ffn_down / down_proj / down
    ShGate,    // ffn_gate_shexp / shared_gate
    ShUp,      // ffn_up_shexp / shared_up
    ShDown,    // ffn_down_shexp / shared_down
    // MoE
    Router,    // ffn_gate_inp / router_w            (replicated)
    ShRouter,  // ffn_gate_inp_shexp                   (replicated)
    ExpGate,   // ffn_gate_exps  (expert pool, expert-axis split)
    ExpUp,     // ffn_up_exps
    ExpDown,   // ffn_down_exps
    Unknown,
};

// qwen35.cpp is_linear_layer() (L466-468) verbatim: a GDN ("linear") layer is
// any layer in a hybrid stack whose (layer+1) is NOT a multiple of the
// full-attention interval; non-hybrid stacks have no GDN layers at all.
inline bool is_gdn_layer(const Qwen35Config& c, int layer) {
    return c.hybrid && c.full_attn_interval > 0 &&
           ((layer + 1) % c.full_attn_interval) != 0;
}

// The full-attention q-projection OUT width: the 27B hybrid folds a sigmoid
// gate next to every q head ([q | gate] interleaved, 2*head_dim per head), so
// its q tensor is twice as wide as a classic or muse checkpoint's; classic
// (non-hybrid) and muse checkpoints carry the plain qdim (muse's gate is a
// separate tensor, see AttnGate).
inline int q_out_width(const Qwen35Config& c) {
    const int qdim = c.n_q_heads * c.head_dim;
    return (c.hybrid && !c.muse_glimmer) ? 2 * qdim : qdim;
}

// ---------------------------------------------------------------------------
// NVFP4 byte math -- pure mirrors of the two load-time fit gates.
// (kernels/csrc/cuda/fused/prefill_nvfp4_sm120.cu:706-719;
//  runtime/src/models/qwen35.cpp:7290-7320 and 8189-8222.)
// ---------------------------------------------------------------------------

// (r*c+1)/2: N fp4 values pack to N/2 bytes, low nibble first.
inline size_t nvfp4_data_bytes(size_t r, size_t c) { return (r * c + 1) / 2; }

// The SFB scale-factor layout is one ue4m3 byte per 16 values; at the 128-
// aligned shapes the GEMM accepts it equals the CUTLASS SFB tile size n*k/16
// exactly (lm_head 248320x5120 -> 79,462,400 B, the "79 MB" of the head gate).
// The +15 rounding only bites unaligned shapes, which nvfp4_supported() below
// already rejects.
inline size_t nvfp4_scale_bytes(size_t n, size_t k) { return (n * k + 15) / 16; }

inline size_t nvfp4_tensor_bytes(size_t r, size_t c) {
    return nvfp4_data_bytes(r, c) + nvfp4_scale_bytes(r, c);
}

// prefill_nvfp4_supported minus the driver query: SM 12.0, m%8, n,k %128.
inline bool nvfp4_supported(int cc_major, int cc_minor, size_t m, size_t n, size_t k) {
    return cc_major == 12 && cc_minor == 0 && m > 0 &&
           (m % 8) == 0 && (n % 128) == 0 && (k % 128) == 0;
}

// The compressed-tensors lm_head NVFP4 keep gate (qwen35.cpp:8192-8201), as a
// pure function of the rows this rank owns. NOTE: the code's payload term is
// 5/8 B/weight (0.625), slightly above payload+scale's 0.5625; mirrored
// verbatim, so a caller wanting the tighter 0.5625 figure subtracts
// 0.0625*rows*h from the payload part.
inline size_t nvfp4_head_keep_need(size_t rows, size_t h) {
    return rows * h * 5 / 8 + nvfp4_scale_bytes(rows, h);
}

// The three muse-preflight NVFP4-copy legs (qwen35.cpp:7290-7298), per layer,
// at the (possibly per-rank-reduced) head widths: qkvg = the q|gate (2q) + k
// + v (2kv) rows against H; wo = [H, qdim]; down = [H, ffn].
inline size_t nvfp4_muse_qkvg_layer(size_t qdim, size_t kvdim, size_t h) {
    const size_t r = 2 * qdim + 2 * kvdim;
    return nvfp4_data_bytes(r, h) + nvfp4_scale_bytes(r, h);
}
inline size_t nvfp4_muse_wo_layer(size_t qdim, size_t h) {
    return nvfp4_data_bytes(h, qdim) + nvfp4_scale_bytes(h, qdim);
}
inline size_t nvfp4_muse_down_layer(size_t h, size_t f) {
    return nvfp4_data_bytes(h, f) + nvfp4_scale_bytes(h, f);
}

// ---------------------------------------------------------------------------
// Class-aware on-disk byte model (header note: the flat 0.5625 B/weight
// average mis-sizes NVFP4 by ~15%, so the budget uses per-class math).
// A weight with r rows and c inner-dimension columns costs:
//   nvfp4:  nv4(r, c) = (r*c+1)/2 payload + r*((c+15)/16) scale bytes +
//           8 B companion pair (zero-point + dequant f32). The scales are
//           16-element blocks of the INNER dimension -- the 27B file stores
//           F8E4M3 [rows, cols/16], not one (r*c+15)/16 block stream -- and
//           every 27B shape is 16-aligned, so this equals
//           nvfp4_data_bytes + nvfp4_scale_bytes for them.
//   bf16:   2.0 B/element.   f32: 4.0 B/element (reserved; the 27B file has
//           no f32 tensor -- A_log and dt_bias are bf16, qwen35.cpp:7707,8256).
// Rows follow the HF file layout [out, in]; in the 27B file (the family this
// model is calibrated for; it is HF-only) every NVFP4 linear is nv4(out, in)
// and the 401 linears plus the bf16 remainder partition the file exactly.
// ---------------------------------------------------------------------------

// The on-disk storage class of a tensor kind.
enum class WClass { Nvfp4, Bf16, F32 };

// The on-disk storage class of a kind: the NVFP4 set is the 27B's 401
// quantized linears' logical classes; everything else is bf16 there.
// (F32 is reserved for future f32-stored tensors. Non-27B files -- the 35B
// GGUF is q4_k -- have no class here: the byte model is gated on cfg().qwen38
// and such models stay a placement-only cross-check.)
inline WClass class_for(Kind k) {
    switch (k) {
        case Kind::Q:
        case Kind::AttnGate:
        case Kind::K:
        case Kind::V:
        case Kind::O:
        case Kind::GdnQKv:
        case Kind::GdnGate:
        case Kind::GdnOut:
        case Kind::FfnGate:
        case Kind::FfnUp:
        case Kind::FfnDown:
        case Kind::ShGate:
        case Kind::ShUp:
        case Kind::ShDown:
        case Kind::LmHead:
            return WClass::Nvfp4;
        default:
            return WClass::Bf16;
    }
}

// The nvfp4 on-disk size above, in one line.
inline size_t nv4(size_t r, size_t c) {
    return (r * c + 1) / 2 + r * ((c + 15) / 16) + 8;
}

// The bytes this rank of one tensor holds on disk, by class. The unsplit
// tensor (denom == 0) sits whole on its rank (or on every rank, if
// replicated); a rank owning a num/denom share of the split axis takes that
// share of the payload, and the NVFP4 f32 companion pair (8 B) is NEVER
// split -- any rank owning any of the tensor keeps the whole pair, so the
// per-rank sum carries the (R - H)/2 companion rounding the budget note
// tracks. bf16/f32 take a linear share, llrounded.
inline size_t rank_on_disk_bytes(WClass cls, size_t on_disk, size_t num, size_t denom) {
    if (denom == 0) return on_disk;
    if (num == 0) return 0;
    if (cls == WClass::Nvfp4) return (on_disk - 8) * num / denom + 8;
    return (size_t)llround(double(on_disk) * double(num) / double(denom));
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

class Table {
public:
    // n_ranks <= 1: the degenerate single-device table (see tp=1 invariance
    // in the file header).
    static Table build(const Qwen35Config& cfg, int n_ranks, Conv conv = Conv::Gguf) {
        Table t;
        t.cfg_ = cfg;
        t.conv_ = conv;
        t.n_ranks_ = n_ranks < 1 ? 1 : n_ranks;
        t.build_index();
        return t;
    }

    // Unset (default-constructed) tables behave as the degenerate tp=1 table:
    // every name resolves whole on device 0, so a loader that consults the
    // table before it is set -- or at tp=1 -- is byte-identical to today.
    bool set() const { return n_ranks_ >= 1; }
    int n_ranks() const { return set() ? n_ranks_ : 1; }
    Conv conv() const { return conv_; }
    const Qwen35Config& cfg() const { return cfg_; }

    // Placement of `name` for rank `rank` (out-of-range ranks clamp to the
    // ends). The degenerate table -- unset, or built with one rank -- hands
    // EVERY name (known or not) the whole tensor on device 0, no split: the
    // tp=1 invariance. At n_ranks > 1 a known name gets the rank's slice; an
    // unknown name falls to the two-tier fallback (header note): the
    // "model.visual." prefix is single-homed whole on rank 0, everything else
    // (MTP-era tensors, HF MoE expert pools) is replicated (device -1; the
    // caller warns once per name via known()).
    Placement placement(const std::string& name, int rank) const {
        if (!set() || n_ranks_ == 1) {
            Placement p;
            p.device = 0;
            return p;
        }
        if (rank < 0) rank = 0;
        if (rank > n_ranks_ - 1) rank = n_ranks_ - 1;
        const auto it = index_.find(name);
        if (it == index_.end()) {
            Placement p;
            p.device = name.compare(0, 13, "model.visual.") == 0 ? 0 : -1;
            return p;
        }
        Placement p;
        p.device = rank;
        p.axis = axis_for(it->second);
        p.denom = denom_for(it->second);
        if (p.axis != Axis::None) p.ranges = ranges_for(it->second, rank);
        return p;
    }

    // True if the name matches a known pattern of this convention (unknown
    // names are placed replicated and callers warn once per name on !known).
    bool known(const std::string& name) const { return index_.count(name) > 0; }

    // The full (pre-split) element count of `name` under this convention;
    // 0 when unknown.
    size_t full_elements(const std::string& name) const {
        const auto it = index_.find(name);
        return it == index_.end() ? 0 : full_elements(it->second);
    }

    // The on-disk byte count of `name` under this convention (0 when
    // unknown): the class-aware model in the section above, applied to the
    // kind's file shape (rows = the file tensor's first stored dim).
    size_t on_disk_bytes(const std::string& name) const {
        const auto it = index_.find(name);
        return it == index_.end() ? 0 : on_disk_bytes(it->second);
    }

    // The sum of the on-disk size of every known tensor: for the 27B file
    // the S + R tensor bytes (16,994,355,336 B); the visual tower (V) and
    // the shard-header bytes (H) are not known tensors and stay out of the
    // table, tracked by the budget note instead.
    size_t total_on_disk_bytes() const {
        size_t s = 0;
        for (const auto& e : entries_) s += on_disk_bytes(e.second);
        return s;
    }

    // The on-disk bytes this rank holds across every known tensor: a split
    // tensor contributes the rank's class-aware share (the NVFP4 f32
    // companion pair stays whole on any rank that owns any part of it --
    // the (R - H)/2 companion rounding the budget note tracks), an unsplit
    // one its full size. This IS the per-card budget.
    size_t rank_on_disk_bytes(int rank) const {
        if (!set()) return 0;
        if (rank < 0) rank = 0;
        if (rank > n_ranks_ - 1) rank = n_ranks_ - 1;
        size_t s = 0;
        for (const auto& e : entries_) {
            const Placement p = placement(e.first, rank);
            s += p.denom == 0 ? on_disk_bytes(e.second) :
                 tp::rank_on_disk_bytes(class_for(e.second), on_disk_bytes(e.second),
                                         p.split_elements(), p.denom);
        }
        return s;
    }

    // The (name, kind) pairs this table expects, sorted by name: the expected
    // tensor inventory for the convention (what the report/print bridges and
    // the CPU test iterate over).
    const std::vector<std::pair<std::string, Kind>>& entries() const { return entries_; }

    // Total elements this rank owns across every known tensor (a replicated
    // tensor counts in full on each rank; that IS the per-device residency).
    size_t rank_elements(int rank) const {
        if (!set()) return 0;
        size_t s = 0;
        for (const auto& e : entries_) {
            const Placement p = placement(e.first, rank);
            s += full_elements(e.second) * p.fraction();
        }
        return s;
    }

private:
    Qwen35Config cfg_;
    Conv conv_ = Conv::Gguf;
    int n_ranks_ = 0;
    std::map<std::string, Kind> index_;   // name -> kind
    std::vector<std::pair<std::string, Kind>> entries_;

    void build_index() {
        std::map<std::string, Kind> m;
        auto add = [&](const std::string& name, Kind k) {
            if (name.empty()) return;
            m[name] = k;
        };
        // top level
        add(name_for(Kind::Embd, -1, 0), Kind::Embd);
        add(name_for(Kind::OutNorm, -1, 0), Kind::OutNorm);
        add(name_for(Kind::LmHead, -1, 0), Kind::LmHead);
        for (int i = 0; i < cfg_.n_layers; ++i) {
            const bool gdn = is_gdn_layer(cfg_, i);
            // the block norms (slots: 0 input, 1 post-attn, 2 pre-ffn, 3
            // post-ffn); the HF and flat conventions only carry the first two
            const int slots = conv_ == Conv::Gguf ? 4 : 2;
            for (int s = 0; s < slots; ++s) add(name_for(Kind::Norm, i, s), Kind::Norm);
            if (gdn) {
                add(name_for(Kind::GdnQKv, i, 0), Kind::GdnQKv);
                add(name_for(Kind::GdnGate, i, 0), Kind::GdnGate);
                add(name_for(Kind::GdnConv, i, 0), Kind::GdnConv);
                add(name_for(Kind::GdnOut, i, 0), Kind::GdnOut);
                add(name_for(Kind::GdnA, i, 0), Kind::GdnA);
                add(name_for(Kind::GdnDt, i, 0), Kind::GdnDt);
                add(name_for(Kind::GdnAlpha, i, 0), Kind::GdnAlpha);
                add(name_for(Kind::GdnBeta, i, 0), Kind::GdnBeta);
                add(name_for(Kind::GdnNorm, i, 0), Kind::GdnNorm);
            } else {
                add(name_for(Kind::Q, i, 0), Kind::Q);
                add(name_for(Kind::K, i, 0), Kind::K);
                add(name_for(Kind::V, i, 0), Kind::V);
                add(name_for(Kind::O, i, 0), Kind::O);
                add(name_for(Kind::QNorm, i, 0), Kind::QNorm);
                add(name_for(Kind::KNorm, i, 0), Kind::KNorm);
                if (cfg_.muse_glimmer) add(name_for(Kind::AttnGate, i, 0), Kind::AttnGate);
            }
            // the FFN: dense (27B and friends) vs routed MoE (35B-A3B and
            // friends). The HF convention's MoE expert names are unverified in
            // this repo, so they are intentionally absent from the table: any
            // HF MoE pool tensor falls into the replicated fallback (warned).
            const bool dense = cfg_.dense_ffn || cfg_.n_experts <= 1;
            if (dense) {
                add(name_for(Kind::FfnGate, i, 0), Kind::FfnGate);
                add(name_for(Kind::FfnUp, i, 0), Kind::FfnUp);
                add(name_for(Kind::FfnDown, i, 0), Kind::FfnDown);
            } else if (conv_ != Conv::Hf) {
                add(name_for(Kind::Router, i, 0), Kind::Router);
                add(name_for(Kind::ExpGate, i, 0), Kind::ExpGate);
                add(name_for(Kind::ExpUp, i, 0), Kind::ExpUp);
                add(name_for(Kind::ExpDown, i, 0), Kind::ExpDown);
            }
            if (cfg_.n_shared > 0 && conv_ != Conv::Hf) {
                add(name_for(Kind::ShGate, i, 0), Kind::ShGate);
                add(name_for(Kind::ShUp, i, 0), Kind::ShUp);
                add(name_for(Kind::ShDown, i, 0), Kind::ShDown);
                add(name_for(Kind::ShRouter, i, 0), Kind::ShRouter);
            }
        }
        index_.swap(m);
        entries_.assign(index_.begin(), index_.end());
    }

    // The tensor name a (kind, layer, slot) has under this convention; empty
    // when the kind does not exist for this convention/layer type (e.g. GDN
    // kinds at a full-attn layer, or GDN kinds under the flat convention, whose
    // GDN names are unverified in this repo and whose fallback is replicate).
    std::string name_for(Kind k, int layer, int slot) {
        auto layer_prefix = [this, layer]() -> std::string {
            if (layer < 0) return conv_ == Conv::Hf ? "model.language_model." : "";
            if (conv_ == Conv::Hf)
                return "model.language_model.layers." + std::to_string(layer) + ".";
            if (conv_ == Conv::Flat) return "layer_" + std::to_string(layer) + ".";
            return "blk." + std::to_string(layer) + ".";
        };
        const bool gdn_layer = layer >= 0 && is_gdn_layer(cfg_, layer);
        switch (k) {
            case Kind::Embd:
                return conv_ == Conv::Gguf ? "token_embd.weight" :
                conv_ == Conv::Hf ? "model.language_model.embed_tokens.weight" : "embed_tokens";
            case Kind::OutNorm:
                return conv_ == Conv::Gguf ? "output_norm.weight" :
                conv_ == Conv::Hf ? "model.language_model.norm.weight" : "final_norm";
            case Kind::LmHead:
                // The 27B/35B HF and flat files store the head FLAT ("lm_head.*"
                // in shard 2; the loaders look up the flat name --
                // qwen35.cpp:5957,8164,8191 -- never model.language_model.lm_head).
                return conv_ == Conv::Gguf ? "output.weight" :
                conv_ == Conv::Hf ? "lm_head.weight" : "lm_head";
            case Kind::Norm:
                if (conv_ == Conv::Gguf) {
                    static const char* n[4] = {"blk.%d.attn_norm", "blk.%d.attn_post_norm",
                                               "blk.%d.ffn_norm", "blk.%d.post_ffw_norm"};
                    char b[64]; std::snprintf(b, sizeof(b), n[slot & 3], layer); return b;
                }
                if (conv_ == Conv::Hf) {
                    static const char* n[2] = {"model.language_model.layers.%d.input_layernorm.weight",
                                              "model.language_model.layers.%d.post_attention_layernorm.weight"};
                    char b[96]; std::snprintf(b, sizeof(b), n[slot & 1], layer); return b;
                }
                static const char* n[2] = {"layer_%d.input_norm", "layer_%d.post_attn_norm"};
                char b[48]; std::snprintf(b, sizeof(b), n[slot & 1], layer); return b;
            case Kind::Q:
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_q.weight" :
                                     conv_ == Conv::Hf ? "self_attn.q_proj" : "wq");
            case Kind::AttnGate:
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_gate.weight" :
                                     conv_ == Conv::Hf ? "self_attn.gate_proj" : "wgate");
            case Kind::K:
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_k.weight" :
                                     conv_ == Conv::Hf ? "self_attn.k_proj" : "wk");
            case Kind::V:
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_v.weight" :
                                     conv_ == Conv::Hf ? "self_attn.v_proj" : "wv");
            case Kind::O:
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_output.weight" :
                                     conv_ == Conv::Hf ? "self_attn.o_proj" : "wo");
            case Kind::QNorm:
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_q_norm.weight" :
                                     conv_ == Conv::Hf ? "self_attn.q_norm.weight" : "q_norm");
            case Kind::KNorm:
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_k_norm.weight" :
                                     conv_ == Conv::Hf ? "self_attn.k_norm.weight" : "k_norm");
            case Kind::GdnQKv:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_qkv.weight" :
                                     conv_ == Conv::Hf ? "linear_attn.in_proj_qkv" : "wgqkv");
            case Kind::GdnGate:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "attn_gate.weight" :
                                     conv_ == Conv::Hf ? "linear_attn.in_proj_z" : "wgate");
            case Kind::GdnConv:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "ssm_conv1d.weight" :
                                     conv_ == Conv::Hf ? "linear_attn.conv1d.weight" : "ssm_conv1d");
            case Kind::GdnOut:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "ssm_out.weight" :
                                     conv_ == Conv::Hf ? "linear_attn.out_proj" : "ssm_out");
            case Kind::GdnA:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "ssm_a" :
                                     conv_ == Conv::Hf ? "linear_attn.A_log" : "ssm_a");
            case Kind::GdnDt:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "ssm_dt.bias" :
                                     conv_ == Conv::Hf ? "linear_attn.dt_bias" : "ssm_dt");
            case Kind::GdnAlpha:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "ssm_alpha.weight" :
                                     conv_ == Conv::Hf ? "linear_attn.in_proj_a.weight" : "ssm_alpha");
            case Kind::GdnBeta:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "ssm_beta.weight" :
                                     conv_ == Conv::Hf ? "linear_attn.in_proj_b.weight" : "ssm_beta");
            case Kind::GdnNorm:
                if (!gdn_layer) return {};
                return layer_prefix() + (conv_ == Conv::Gguf ? "ssm_norm.weight" :
                                     conv_ == Conv::Hf ? "linear_attn.norm.weight" : "ssm_norm");
            case Kind::FfnGate:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_gate.weight" :
                                     conv_ == Conv::Hf ? "mlp.gate_proj" : "gate");
            case Kind::FfnUp:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_up.weight" :
                                     conv_ == Conv::Hf ? "mlp.up_proj" : "up");
            case Kind::FfnDown:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_down.weight" :
                                     conv_ == Conv::Hf ? "mlp.down_proj" : "down");
            case Kind::Router:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_gate_inp.weight" : "router_w");
            case Kind::ShRouter:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_gate_inp_shexp.weight" : "shared_router_w");
            case Kind::ExpGate:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_gate_exps.weight" : "gate");
            case Kind::ExpUp:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_up_exps.weight" : "up");
            case Kind::ExpDown:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_down_exps.weight" : "down");
            case Kind::ShGate:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_gate_shexp.weight" : "shared_gate");
            case Kind::ShUp:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_up_shexp.weight" : "shared_up");
            case Kind::ShDown:
                return layer_prefix() + (conv_ == Conv::Gguf ? "ffn_down_shexp.weight" : "shared_down");
            default:
                return {};
        }
    }

    // The split axis a kind lives on under this convention (None = replicated
    // or whole on one rank). See the axis-convention note in the header.
    Axis axis_for(Kind k) const {
        switch (k) {
            case Kind::Embd:
            case Kind::LmHead:
                return Axis::Rows;
            case Kind::Q:
            case Kind::AttnGate:
            case Kind::K:
            case Kind::V:
            case Kind::GdnGate:
            case Kind::FfnGate:
            case Kind::FfnUp:
            case Kind::ShGate:
            case Kind::ShUp:
                // out-feature axis: dims[1] in GGUF/flat ([in,out]),
                // dims[0] in HF ([out,in])
                return conv_ == Conv::Hf ? Axis::Rows : Axis::Cols;
            case Kind::O:
            case Kind::GdnOut:
            case Kind::FfnDown:
            case Kind::ShDown:
                // in-feature (head-concat / ffn) axis: dims[0] in GGUF/flat,
                // dims[1] in HF
                return conv_ == Conv::Hf ? Axis::Cols : Axis::Rows;
            case Kind::GdnQKv:
            case Kind::GdnConv:
                // the shared q|k|v channel axis: dims[1] in GGUF ([taps,
                // channels] / [in, channels]), dims[0] in HF ([channels,H] /
                // [channels,1,taps])
                return conv_ == Conv::Hf ? Axis::Rows : Axis::Cols;
            case Kind::GdnA:
            case Kind::GdnDt:
                return Axis::OneD;
            case Kind::GdnAlpha:
            case Kind::GdnBeta:
                return conv_ == Conv::Hf ? Axis::Rows : Axis::Cols;
            case Kind::ExpGate:
            case Kind::ExpUp:
            case Kind::ExpDown:
                return Axis::Experts;
            default:
                return Axis::None;  // norms, q/k_norm, routers: replicated
        }
    }

    // The full extent of the split axis (in elements) for a kind; 0 when the
    // kind is not split.
    size_t denom_for(Kind k) const {
        const Qwen35Config& c = cfg_;
        const size_t V = c.vocab;
        const size_t qdim = size_t(c.n_q_heads) * c.head_dim;
        const size_t kvdim = size_t(c.n_kv_heads) * c.head_dim;
        const size_t lq = size_t(c.linear_q_heads) * c.linear_head_dim;
        const size_t lv = size_t(c.linear_v_heads) * c.linear_head_dim;
        const size_t F = c.moe_ffn;
        switch (k) {
            case Kind::Embd:
            case Kind::LmHead:
                return V;
            case Kind::Q:
                return size_t(q_out_width(c));
            case Kind::AttnGate:
                return qdim;
            case Kind::K:
            case Kind::V:
                return kvdim;
            case Kind::O:
                return qdim;
            case Kind::GdnQKv:
            case Kind::GdnConv:
                return 2 * lq + lv;
            case Kind::GdnGate:
            case Kind::GdnOut:
                return lv;
            case Kind::GdnA:
            case Kind::GdnDt:
                return c.linear_v_heads;
            case Kind::GdnAlpha:
            case Kind::GdnBeta:
                return c.linear_v_heads;
            case Kind::FfnGate:
            case Kind::FfnUp:
            case Kind::FfnDown:
            case Kind::ShGate:
            case Kind::ShUp:
            case Kind::ShDown:
                return F;
            case Kind::ExpGate:
            case Kind::ExpUp:
            case Kind::ExpDown:
                return c.n_experts;
            default:
                return 0;
        }
    }

    // The rank's element window on a head axis of `n` heads (the last rank
    // takes any remainder, so the windows tile n exactly).
    void head_window(size_t n, int r, size_t* h0, size_t* hlen) const {
        *h0 = n * size_t(r) / n_ranks_;
        *hlen = n * size_t(r + 1) / n_ranks_ - *h0;
    }
    void feature_window(size_t n, int r, size_t* b0, size_t* blen) const {
        *b0 = n * size_t(r) / n_ranks_;
        *blen = n * size_t(r + 1) / n_ranks_ - *b0;
    }

    // The rank's sub-interval(s) along the split axis for a kind.
    std::vector<Range> ranges_for(Kind k, int r) const {
        const Qwen35Config& c = cfg_;
        const int hd = c.head_dim, lhd = c.linear_head_dim;
        const size_t F = c.moe_ffn;
        size_t b0, blen;
        switch (k) {
            case Kind::Embd:
            case Kind::LmHead:
                feature_window(c.vocab, r, &b0, &blen);
                return {{b0, blen}};
            case Kind::Q: {
                const size_t w = size_t(q_out_width(c)) / c.n_q_heads;  // per head
                size_t h0, hl;
                head_window(c.n_q_heads, r, &h0, &hl);
                return {{h0 * w, hl * w}};
            }
            case Kind::AttnGate: {
                size_t h0, hl;
                head_window(c.n_q_heads, r, &h0, &hl);
                return {{h0 * hd, hl * hd}};
            }
            case Kind::K:
            case Kind::V: {
                size_t h0, hl;
                head_window(c.n_kv_heads, r, &h0, &hl);
                return {{h0 * hd, hl * hd}};
            }
            case Kind::O: {
                size_t h0, hl;
                head_window(c.n_q_heads, r, &h0, &hl);
                return {{h0 * hd, hl * hd}};
            }
            case Kind::GdnQKv:
            case Kind::GdnConv: {
                const size_t lq = c.linear_q_heads, lv = c.linear_v_heads;
                size_t q0, ql, v0, vl;
                head_window(lq, r, &q0, &ql);
                head_window(lv, r, &v0, &vl);
                // the [all q heads | all k heads | all v heads] layout: k heads
                // == q heads (1:1) in GDN, so the k section mirrors the q one.
                const size_t qb = 0, kb = lq * size_t(lhd), vb = 2 * lq * size_t(lhd);
                return {
                    {qb + q0 * size_t(lhd), ql * size_t(lhd)},
                    {kb + q0 * size_t(lhd), ql * size_t(lhd)},
                    {vb + v0 * size_t(lhd), vl * size_t(lhd)},
                };
            }
            case Kind::GdnGate:
            case Kind::GdnOut: {
                size_t v0, vl;
                head_window(c.linear_v_heads, r, &v0, &vl);
                return {{v0 * size_t(lhd), vl * size_t(lhd)}};
            }
            case Kind::GdnA:
            case Kind::GdnDt:
            case Kind::GdnAlpha:
            case Kind::GdnBeta: {
                size_t v0, vl;
                head_window(c.linear_v_heads, r, &v0, &vl);
                return {{v0, vl}};
            }
            case Kind::FfnGate:
            case Kind::FfnUp:
            case Kind::FfnDown:
            case Kind::ShGate:
            case Kind::ShUp:
            case Kind::ShDown:
                feature_window(F, r, &b0, &blen);
                return {{b0, blen}};
            case Kind::ExpGate:
            case Kind::ExpUp:
            case Kind::ExpDown:
                feature_window(c.n_experts, r, &b0, &blen);
                return {{b0, blen}};
            default:
                return {};
        }
    }

    // The full (pre-split) element count of a kind under this convention.
    size_t full_elements(Kind k) const {
        const Qwen35Config& c = cfg_;
        const size_t H = c.hidden, V = c.vocab;
        const size_t qdim = size_t(c.n_q_heads) * c.head_dim;
        const size_t kvdim = size_t(c.n_kv_heads) * c.head_dim;
        const size_t lq = size_t(c.linear_q_heads) * c.linear_head_dim;
        const size_t lv = size_t(c.linear_v_heads) * c.linear_head_dim;
        const size_t F = c.moe_ffn, E = c.n_experts;
        switch (k) {
            case Kind::Embd:
            case Kind::LmHead:
                return V * H;
            case Kind::OutNorm:
                return H;
            case Kind::Norm:
                return H;
            case Kind::QNorm:
            case Kind::KNorm:
                return c.head_dim;
            case Kind::GdnNorm:
                return c.linear_head_dim;
            case Kind::Q:
                return H * size_t(q_out_width(c));
            case Kind::AttnGate:
                return H * qdim;
            case Kind::K:
            case Kind::V:
                return H * kvdim;
            case Kind::O:
                return qdim * H;
            case Kind::GdnQKv:
                return H * (2 * lq + lv);
            case Kind::GdnGate:
                return H * lv;
            case Kind::GdnOut:
                return lv * H;
            case Kind::GdnConv:
                return size_t(c.linear_conv_kernel) * (2 * lq + lv);
            case Kind::GdnA:
            case Kind::GdnDt:
                return c.linear_v_heads;
            case Kind::GdnAlpha:
            case Kind::GdnBeta:
                return H * size_t(c.linear_v_heads);
            case Kind::FfnGate:
            case Kind::FfnUp:
                return H * F;
            case Kind::FfnDown:
                return F * H;
            case Kind::ShGate:
            case Kind::ShUp:
                return H * F;
            case Kind::ShDown:
                return F * H;
            case Kind::Router:
                return H * E;
            case Kind::ShRouter:
                return H;
            case Kind::ExpGate:
            case Kind::ExpUp:
                return E * H * F;
            case Kind::ExpDown:
                return E * F * H;
            default:
                return 0;
        }
    }

    // The on-disk byte count of a kind: the class-aware model in the section
    // above, applied to the kind's file shape. Rows follow the HF file
    // layout [out, in] -- the 27B NVFP4 family this model is calibrated for
    // is HF-only; non-qwen38 files (35B q4_k GGUF, ...) do NOT follow the
    // nvfp4/bf16 model, which is why the bridges gate the byte cross-check
    // on cfg().qwen38. The 27B file's non-quantized tensors (norms, A_log,
    // dt_bias, conv1d, in_proj_a/b, embed, visual) are all bf16
    // (qwen35.cpp:7707,8256,8259), hence Bf16 class for every non-NVFP4 kind.
    size_t on_disk_bytes(Kind k) const {
        const Qwen35Config& c = cfg_;
        const size_t H = c.hidden, V = c.vocab;
        const size_t qdim = size_t(c.n_q_heads) * c.head_dim;
        const size_t kvdim = size_t(c.n_kv_heads) * c.head_dim;
        const size_t lq = size_t(c.linear_q_heads) * c.linear_head_dim;
        const size_t lv = size_t(c.linear_v_heads) * c.linear_head_dim;
        const size_t F = c.moe_ffn, E = c.n_experts;
        switch (k) {
            case Kind::Embd:
                return 2 * V * H;
            case Kind::LmHead:
                return nv4(V, H);
            case Kind::OutNorm:
            case Kind::Norm:
                return 2 * H;
            case Kind::QNorm:
            case Kind::KNorm:
                return 2 * c.head_dim;
            case Kind::GdnNorm:
                return 2 * c.linear_head_dim;
            case Kind::Q:
                return nv4(size_t(q_out_width(c)), H);
            case Kind::AttnGate:
                return nv4(qdim, H);
            case Kind::K:
            case Kind::V:
                return nv4(kvdim, H);
            case Kind::O:
                return nv4(H, qdim);
            case Kind::GdnQKv:
                return nv4(2 * lq + lv, H);
            case Kind::GdnGate:
                return nv4(lv, H);
            case Kind::GdnOut:
                return nv4(H, lv);
            case Kind::GdnConv:
                return 2 * size_t(c.linear_conv_kernel) * (2 * lq + lv);
            case Kind::GdnA:
            case Kind::GdnDt:
                return 2 * c.linear_v_heads;
            case Kind::GdnAlpha:
            case Kind::GdnBeta:
                return 2 * H * size_t(c.linear_v_heads);
            case Kind::FfnGate:
            case Kind::FfnUp:
            case Kind::ShGate:
            case Kind::ShUp:
                return nv4(F, H);
            case Kind::FfnDown:
            case Kind::ShDown:
                return nv4(H, F);
            case Kind::Router:
                return 2 * H * E;
            case Kind::ShRouter:
                return 2 * H;
            case Kind::ExpGate:
            case Kind::ExpUp:
                return 2 * E * H * F;
            case Kind::ExpDown:
                return 2 * E * F * H;
            default:
                return 0;
        }
    }
};

// ---------------------------------------------------------------------------
// The expected-tensor inventory (pure; no file access)
// ---------------------------------------------------------------------------

struct InventorySummary {
    size_t on_disk_total = 0;               // all known tensors, pre-split, on-disk bytes
    size_t total_elements = 0;              // all known tensors, pre-split
    size_t replicated = 0;                 // kinds placed replicated at n_ranks>1
    size_t bytes_per_rank[8] = {0};        // on-disk, per rank (<=8): the per-card budget
    double elements_per_rank[8] = {0};
};

// Prints one line per expected tensor -- name, storage class, split axis, and
// each rank's ranges -- plus the per-rank on-disk totals in MiB, and returns
// the summary. `out` is typically std::cerr in the server flow or a
// stringstream in tests.
inline InventorySummary report(const Table& t, std::ostream& out) {
    InventorySummary s;
    const int R = t.n_ranks();
    auto cls_name = [](WClass c) -> const char* {
        return c == WClass::Nvfp4 ? "nvfp4" : c == WClass::Bf16 ? "bf16" : "f32";
    };
    for (const auto& e : t.entries()) {
        const Placement p0 = t.placement(e.first, 0);
        if (p0.device < 0 && R > 1) s.replicated++;
        s.total_elements += t.full_elements(e.first);
        const size_t ob = t.on_disk_bytes(e.first);
        s.on_disk_total += ob;
        for (int r = 0; r < R && r < 8; ++r) {
            const Placement p = t.placement(e.first, r);
            s.bytes_per_rank[r] += p.denom == 0 ? ob :
                rank_on_disk_bytes(class_for(e.second), ob, p.split_elements(), p.denom);
            s.elements_per_rank[r] += double(t.full_elements(e.first) * p.fraction());
        }
        if (R > 1) {
            out << "  " << e.first << "  (" << cls_name(class_for(e.second)) << ")  [";
            for (int r = 0; r < R; ++r) {
                const Placement p = t.placement(e.first, r);
                out << (r ? ", " : "") << (p.device < 0 ? "all" : ("dev" + std::to_string(p.device)));
                if (p.axis != Axis::None) {
                    out << " " << (p.axis == Axis::Rows ? "rows" : p.axis == Axis::Cols ? "cols" :
                                    p.axis == Axis::Experts ? "experts" : p.axis == Axis::OneD ? "1d" : "none")
                        << "(";
                    for (size_t i = 0; i < p.ranges.size(); ++i)
                        out << (i ? "," : "") << p.ranges[i].begin << ":" << p.ranges[i].len;
                    out << "/" << p.denom << ")";
                }
            }
            out << "]\n";
        }
    }
    for (int r = 0; r < R && r < 8; ++r)
        out << "  rank " << r << ": " << s.bytes_per_rank[r] / (1024.0 * 1024.0)
            << " MiB on disk (" << s.elements_per_rank[r] / 1.0e9 << "e9 elem)\n";
    return s;
}

// ---------------------------------------------------------------------------
// The process-wide store, set by ModelEngine::load() and consulted by the
// parser TUs. Unset/tp=1 degrades to whole-on-device-0 for every name (see the
// tp=1 invariance note in the header), so a premature or absent set is safe.
// ---------------------------------------------------------------------------

inline Table& process_table() {
    static Table t;
    return t;
}
inline void set_process_table(Table t) { process_table() = std::move(t); }
inline const Table& get_process_table() { return process_table(); }

// The one-line placement query the parsers use: whole-on-device-0 when the
// table is unset or degenerate (tp=1), the rank's slice when split, and
// replicated (device -1) for unknown names at n_ranks > 1 (callers warn via
// get_process_table().known(name)).
inline Placement placement_for(const std::string& name, int rank) {
    return get_process_table().placement(name, rank);
}

// ---------------------------------------------------------------------------
// Loader bridges -- defined in the weight-file TUs (gguf.cpp,
// safetensors.cpp), which are the only TUs that know the file layouts.
// ---------------------------------------------------------------------------

// Two-way cross-check against a loaded GGUF file: every table name vs the
// file (names the table expects but the file lacks) and every file name vs
// the table (names the file has but the table does not -- printed with the
// two-tier fallback placement they receive, examples capped). When
// cfg().qwen38 (the 27B NVFP4 family) it also cross-checks BYTES: each file
// tensor's n_bytes against the class model above, and the per-rank on-disk
// totals against the file's own per-rank split -- the per-card budget
// audit. Other models (q4_k GGUF, ...) stay a placement-only check: their
// file bytes do not follow the nvfp4/bf16 model.
void log_gguf_tp_inventory(const GGUF& g, std::ostream& out);

// One-way cross-check against a loaded model directory: SafeTensorsModel
// exposes only open(dir) + tensor(name) (its shard map is private), so this
// walks the table and queries the file; file-only names (model.visual.*,
// shard headers) cannot be listed and are noted as such instead. Same
// qwen38 byte gate as the GGUF bridge.
void log_safetensors_tp_inventory(const SafeTensorsModel& m, std::ostream& out);

}  // namespace tp
}  // namespace sparkinfer
