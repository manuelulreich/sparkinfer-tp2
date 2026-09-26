// The TP placement table (dual-gpu WP-6): per-rank split conventions, the
// class-aware on-disk byte model, the two-tier out-of-pattern fallback, and
// the tp=1 invariance -- pure C++ against runtime/include/sparkinfer/tp_layout.hpp.
// No CUDA, no runtime linkage, no GPU, so it pins the layout decisions on any
// box:
//   g++ -std=c++17 tp_layout_cpu_test.cpp -o tp_layout_cpu_test
//
// The 27B (Qwen3.8-27B NVFP4, HF layout) table is the scored target: it
// asserts the per-rank on-disk residency within 8 KB of the file-derived
// figure and prints the corrected per-card budget against the G2 table
// (the report's earlier "+0.757 MiB" line is superseded: per card we are
// +7.57 MiB over G2). The 35B-A3B (GGUF, q4_k) table is placement-only, as
// is the muse-glimmer profile and the classic MoE pair.

#include "../include/sparkinfer/tp_layout.hpp"

#include <cstdio>
#include <string>

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using sparkinfer::Qwen35Config;
using sparkinfer::tp::Axis;
using sparkinfer::tp::Conv;
using sparkinfer::tp::Kind;
using sparkinfer::tp::Placement;
using sparkinfer::tp::Range;
using sparkinfer::tp::Table;
using sparkinfer::tp::WClass;
using sparkinfer::tp::nv4;
using sparkinfer::tp::rank_on_disk_bytes;

// ---------------------------------------------------------------------------
// The four test checkpoints
// ---------------------------------------------------------------------------

// Qwen3.8-27B (the scored NVFP4 file, HF layout): 24 q / 4 kv heads of 256,
// 64 hybrid layers (full every 4th), GDN 16q/16k/48v x 128, dense F 17408.
static Qwen35Config cfg_27b() {
    Qwen35Config c;
    c.vocab = 248320;
    c.hidden = 5120;
    c.n_layers = 64;
    c.n_q_heads = 24;
    c.n_kv_heads = 4;
    c.head_dim = 256;
    c.n_experts = 1;
    c.n_shared = 0;
    c.moe_ffn = 17408;
    c.hybrid = true;
    c.full_attn_interval = 4;
    c.rope_dim = 64;
    c.linear_q_heads = 16;
    c.linear_v_heads = 48;
    c.linear_head_dim = 128;
    c.linear_conv_kernel = 4;
    c.qwen38 = true;
    c.gdn_qh_block = true;
    c.dense_ffn = true;
    return c;
}

// Qwen3.5/3.6 35B-A3B (GGUF, q4_k; placement-only cross-check): 48 q / 16 kv
// heads of 128, 40 hybrid layers, GDN 16q/16k/32v x 128, 256 experts + 1
// shared, F 768.
static Qwen35Config cfg_35b() {
    Qwen35Config c;
    c.n_q_heads = 48;
    c.n_kv_heads = 16;
    c.moe_ffn = 768;
    c.hybrid = true;
    c.linear_q_heads = 16;
    c.linear_v_heads = 32;
    c.linear_head_dim = 128;
    return c;  // the rest defaults: 40 layers, 256 experts, 1 shared, interval 4
}

// Muse Glimmer profile: dense (1 expert), the attention gate is a tensor, no
// GDN at all (non-hybrid), 1 shared expert.
static Qwen35Config cfg_muse() {
    Qwen35Config c;
    c.n_experts = 1;
    c.dense_ffn = true;
    c.muse_glimmer = true;
    return c;  // non-hybrid: every layer full attention
}

// Classic (non-hybrid) routed MoE, both conventions: 48 q / 16 kv, 256
// experts + 1 shared, F 768.
static Qwen35Config cfg_classic() {
    Qwen35Config c;
    c.n_q_heads = 48;
    c.n_kv_heads = 16;
    c.moe_ffn = 768;
    return c;  // n_experts 256, n_shared 1, hybrid false
}

static std::string hf(int i, const char* s) {
    return "model.language_model.layers." + std::to_string(i) + "." + s;
}
static std::string gg(int i, const char* s) {
    return "blk." + std::to_string(i) + "." + s;
}
static Placement P(const Table& t, const std::string& name, int rank) {
    return t.placement(name, rank);
}
static bool is_range(const Placement& p, size_t b, size_t l) {
    return p.ranges.size() == 1 && p.ranges[0].begin == b && p.ranges[0].len == l;
}
// The GDN [q | k | v] three-range layout (one sub-interval per section).
static bool is_gdn3(const Placement& p, size_t q0, size_t ql, size_t k0, size_t kl,
                    size_t v0, size_t vl) {
    return p.ranges.size() == 3 && p.ranges[0].begin == q0 && p.ranges[0].len == ql &&
           p.ranges[1].begin == k0 && p.ranges[1].len == kl && p.ranges[2].begin == v0 &&
           p.ranges[2].len == vl;
}

// ---------------------------------------------------------------------------
// The scored target: 27B Hf, two ranks
// ---------------------------------------------------------------------------

void test_27b_hf_r2() {
    const Table t = Table::build(cfg_27b(), 2, Conv::Hf);
    CHECK(t.set());
    CHECK(t.n_ranks() == 2);

    // 48 GDN layers x (2 norms + 9 GDN + 3 dense FFN) + 16 full layers x
    // (2 norms + 6 attn + 3 FFN) + embedding + final norm + flat head = 851.
    CHECK(t.entries().size() == 851);
    // The 400 language linears + the head: S + R of the file, exactly.
    CHECK(t.total_on_disk_bytes() == 16994355336ull);

    // The head is FLAT in the file: lm_head.weight, never model.language_model.lm_head.*
    CHECK(t.known("lm_head.weight"));
    CHECK(!t.known("model.language_model.lm_head.weight"));
    // The visual tower is not a known tensor (whole on rank 0, below).
    CHECK(!t.known("model.visual.blocks.0.attn.qkv.weight"));

    // --- a full-attention layer (3: (3+1) % 4 == 0)
    // Q carries the folded [q | gate] width: 24 heads of 512 -> 12288.
    CHECK(P(t, hf(3, "self_attn.q_proj"), 0).axis == Axis::Rows);
    CHECK(P(t, hf(3, "self_attn.q_proj"), 0).denom == 12288);
    CHECK(is_range(P(t, hf(3, "self_attn.q_proj"), 0), 0, 6144));
    CHECK(is_range(P(t, hf(3, "self_attn.q_proj"), 1), 6144, 6144));
    // O splits on the in-feature (head-concat) axis: 24 heads of 256.
    CHECK(P(t, hf(3, "self_attn.o_proj"), 0).denom == 6144);
    CHECK(is_range(P(t, hf(3, "self_attn.o_proj"), 0), 0, 3072));
    CHECK(is_range(P(t, hf(3, "self_attn.o_proj"), 1), 3072, 3072));
    // K/V: 4 kv heads of 256 -> 2 per rank of 512.
    CHECK(is_range(P(t, hf(3, "self_attn.k_proj"), 0), 0, 512));
    CHECK(is_range(P(t, hf(3, "self_attn.k_proj"), 1), 512, 512));
    CHECK(is_range(P(t, hf(3, "self_attn.v_proj"), 0), 0, 512));
    CHECK(P(t, hf(3, "self_attn.v_proj"), 1).denom == 1024);
    // q/k norms are per-head-dim vectors: replicated, unsplittable.
    CHECK(P(t, hf(3, "self_attn.q_norm.weight"), 0).axis == Axis::None);
    CHECK(P(t, hf(3, "self_attn.k_norm.weight"), 1).axis == Axis::None);
    // Block norms: whole on every rank (replicated residency).
    CHECK(P(t, hf(3, "input_layernorm.weight"), 0).axis == Axis::None);
    CHECK(P(t, hf(3, "post_attention_layernorm.weight"), 1).axis == Axis::None);
    // Dense SwiGLU FFN: F = 17408, 8704 per rank.
    CHECK(is_range(P(t, hf(3, "mlp.gate_proj"), 0), 0, 8704));
    CHECK(is_range(P(t, hf(3, "mlp.up_proj"), 1), 8704, 8704));
    CHECK(is_range(P(t, hf(3, "mlp.down_proj"), 1), 8704, 8704));
    // Embedding and head split on vocab rows: 248320 / 2 = 124160.
    CHECK(is_range(P(t, "model.language_model.embed_tokens.weight", 0), 0, 124160));
    CHECK(is_range(P(t, "model.language_model.embed_tokens.weight", 1), 124160, 124160));
    CHECK(is_range(P(t, "lm_head.weight", 0), 0, 124160));
    CHECK(is_range(P(t, "lm_head.weight", 1), 124160, 124160));
    CHECK(P(t, "model.language_model.norm.weight", 0).axis == Axis::None);

    // --- a GDN layer (0): in_proj_qkv is [q | k | v] over 16q/16k/48v x 128
    // (qkvdim 10240), and k heads mirror q (1:1), so one rank's slice is
    // three column ranges.
    CHECK(is_gdn3(P(t, hf(0, "linear_attn.in_proj_qkv"), 0), 0, 1024, 2048, 1024, 4096, 3072));
    CHECK(is_gdn3(P(t, hf(0, "linear_attn.in_proj_qkv"), 1), 1024, 1024, 3072, 1024, 7168, 3072));
    CHECK(P(t, hf(0, "linear_attn.in_proj_qkv"), 0).denom == 10240);
    // conv1d shares the channel axis (same three ranges).
    CHECK(is_gdn3(P(t, hf(0, "linear_attn.conv1d.weight"), 0), 0, 1024, 2048, 1024, 4096, 3072));
    // in_proj_z / out_proj: 48 v-heads of 128 -> 24 per rank of 3072.
    CHECK(is_range(P(t, hf(0, "linear_attn.in_proj_z"), 0), 0, 3072));
    CHECK(is_range(P(t, hf(0, "linear_attn.in_proj_z"), 1), 3072, 3072));
    CHECK(is_range(P(t, hf(0, "linear_attn.out_proj"), 0), 0, 3072));
    CHECK(is_range(P(t, hf(0, "linear_attn.out_proj"), 1), 3072, 3072));
    // 1-D per-v-head tensors split on the v-head axis: 48 -> 24 per rank.
    CHECK(is_range(P(t, hf(0, "linear_attn.A_log"), 0), 0, 24));
    CHECK(is_range(P(t, hf(0, "linear_attn.A_log"), 1), 24, 24));
    CHECK(is_range(P(t, hf(0, "linear_attn.dt_bias"), 0), 0, 24));
    CHECK(is_range(P(t, hf(0, "linear_attn.in_proj_a.weight"), 1), 24, 24));
    CHECK(is_range(P(t, hf(0, "linear_attn.in_proj_b.weight"), 1), 24, 24));
    CHECK(P(t, hf(0, "linear_attn.norm.weight"), 0).axis == Axis::None);

    // --- the per-rank on-disk budget (the scored figure)
    const size_t r0 = t.rank_on_disk_bytes(0);
    const size_t r1 = t.rank_on_disk_bytes(1);
    const size_t target = 8497852484ull;  // S/2 + R: the per-card target
    CHECK(r0 == r1);
    CHECK(r0 >= target && r0 - target <= 8192);  // the brief's 8 KB tolerance
    // Deterministic, so the exact figure too: the target plus the 401 NVFP4
    // companion pairs rounding +4 B each (the (R - H)/2 companion note).
    CHECK(r0 == 8497854088ull);

    // --- the corrected per-card budget, printed for the report
    {
        constexpr size_t kFile = 17916112584ull;  // the 27B file: S + V + R + H
        constexpr size_t kVisual = 921460192ull;   // the visual tower (not in the table)
        constexpr size_t kH = 297056ull;           // shard JSON headers + 16-B prefixes
        constexpr size_t kR = 1349632ull;          // norms: 12288 + 16384 + 1320960
        constexpr size_t kG2 = 8950644736ull;       // the G2 per-card budget line
        const auto mib = [](size_t b) { return b / 1048576.0; };
        const size_t avg = kFile / 2 + (kR - kH) / 2;  // the true per-card average
        std::printf(
            "  27B per-card budget context (R=2):\n"
            "    table per-rank on disk   = %11zu B = %8.2f MiB  (target %zu B = S/2 + R, off by %zd B = 401 companion pairs x 4 B)\n"
            "    rank-0 physical          = %11zu B = %8.2f MiB  (table rank 0 + the WHOLE visual tower)\n"
            "    per-card average         = %11zu B = %8.2f MiB  (file/2 = %.2f MiB; delta (R-H)/2 = %.3f MiB)\n"
            "    G2 per-card budget       = %11zu B = %8.1f MiB  -> per-card average is %+.2f MiB over G2\n",
            r0, mib(r0), target, (long)(r0 - target),
            r0 + kVisual, mib(r0 + kVisual),
            avg, mib(avg), mib(kFile / 2), mib((kR - kH) / 2),
            kG2, mib(kG2), mib(avg) - mib(kG2));
    }
}

// ---------------------------------------------------------------------------
// The 35B-A3B GGUF: placement-only cross-check (q4_k, not the nvfp4 model)
// ---------------------------------------------------------------------------

void test_35b_gguf_r2() {
    const Table t = Table::build(cfg_35b(), 2, Conv::Gguf);
    // 30 GDN layers x (4 norms + 9 GDN + 4 routed + 4 shared) +
    // 10 full layers x (4 norms + 6 attn + 8) + 3 = 813.
    CHECK(t.entries().size() == 813);

    // Full-attention layer 3: GGUF tensors are [in, out], so the in-projections
    // split on Cols; Q carries the folded [q | gate] width on this hybrid.
    CHECK(P(t, gg(3, "attn_q.weight"), 0).axis == Axis::Cols);
    CHECK(P(t, gg(3, "attn_q.weight"), 0).denom == 12288);  // 2 x 48 x 128
    CHECK(is_range(P(t, gg(3, "attn_q.weight"), 0), 0, 6144));
    CHECK(is_range(P(t, gg(3, "attn_q.weight"), 1), 6144, 6144));
    CHECK(is_range(P(t, gg(3, "attn_k.weight"), 0), 0, 1024));  // 8 of 16 kv heads x 128
    CHECK(is_range(P(t, gg(3, "attn_k.weight"), 1), 1024, 1024));
    CHECK(is_range(P(t, gg(3, "attn_v.weight"), 1), 1024, 1024));
    CHECK(P(t, gg(3, "attn_output.weight"), 0).axis == Axis::Rows);
    CHECK(is_range(P(t, gg(3, "attn_output.weight"), 0), 0, 3072));  // in-feature: qdim
    CHECK(P(t, gg(3, "attn_q_norm.weight"), 0).axis == Axis::None);
    CHECK(P(t, gg(3, "attn_post_norm.weight"), 1).axis == Axis::None);
    // No attention gate on a 35B full layer (that is muse-only) ...
    CHECK(!t.known(gg(3, "attn_gate.weight")));
    // ... but the GDN layer's gate reuses the SAME string
    CHECK(t.known(gg(0, "attn_gate.weight")));

    // GDN layer 0: 16q/16k/32v x 128 -> 8192 channels; the GDN gate is the
    // in_proj_z of 32 v-heads (4096), out_proj the in-feature of 4096.
    CHECK(is_gdn3(P(t, gg(0, "attn_qkv.weight"), 0), 0, 1024, 2048, 1024, 4096, 2048));
    CHECK(is_gdn3(P(t, gg(0, "attn_qkv.weight"), 1), 1024, 1024, 3072, 1024, 6144, 2048));
    CHECK(is_gdn3(P(t, gg(0, "ssm_conv1d.weight"), 0), 0, 1024, 2048, 1024, 4096, 2048));
    CHECK(is_range(P(t, gg(0, "attn_gate.weight"), 0), 0, 2048));
    CHECK(is_range(P(t, gg(0, "ssm_out.weight"), 0), 0, 2048));
    CHECK(P(t, gg(0, "ssm_a"), 0).axis == Axis::OneD);
    CHECK(is_range(P(t, gg(0, "ssm_a"), 0), 0, 16));
    CHECK(is_range(P(t, gg(0, "ssm_a"), 1), 16, 16));
    CHECK(is_range(P(t, gg(0, "ssm_dt.bias"), 0), 0, 16));
    CHECK(is_range(P(t, gg(0, "ssm_alpha.weight"), 1), 16, 16));
    CHECK(is_range(P(t, gg(0, "ssm_beta.weight"), 1), 16, 16));
    CHECK(P(t, gg(0, "ssm_norm.weight"), 0).axis == Axis::None);

    // The expert pools split on the expert axis: 256 -> 128 per rank.
    CHECK(P(t, gg(0, "ffn_gate_exps.weight"), 0).axis == Axis::Experts);
    CHECK(P(t, gg(0, "ffn_gate_exps.weight"), 0).denom == 256);
    CHECK(is_range(P(t, gg(0, "ffn_gate_exps.weight"), 0), 0, 128));
    CHECK(is_range(P(t, gg(0, "ffn_gate_exps.weight"), 1), 128, 128));
    CHECK(is_range(P(t, gg(0, "ffn_up_exps.weight"), 0), 0, 128));
    CHECK(is_range(P(t, gg(0, "ffn_down_exps.weight"), 1), 128, 128));
    // The routed router and the shared router: replicated.
    CHECK(P(t, gg(0, "ffn_gate_inp.weight"), 0).axis == Axis::None);
    CHECK(P(t, gg(0, "ffn_gate_inp_shexp.weight"), 1).axis == Axis::None);
    // The shared expert: F = 768, 384 per rank (gate/up on Cols, down on Rows).
    CHECK(is_range(P(t, gg(0, "ffn_gate_shexp.weight"), 0), 0, 384));
    CHECK(is_range(P(t, gg(0, "ffn_up_shexp.weight"), 1), 384, 384));
    CHECK(P(t, gg(0, "ffn_down_shexp.weight"), 0).axis == Axis::Rows);
    CHECK(is_range(P(t, gg(0, "ffn_down_shexp.weight"), 0), 0, 384));

    // The head is flat "output.weight" in the GGUF, split on vocab rows.
    CHECK(t.known("output.weight"));
    CHECK(is_range(P(t, "output.weight", 0), 0, 75968));  // 151936 / 2
    CHECK(is_range(P(t, "output.weight", 1), 75968, 75968));
    CHECK(t.known("token_embd.weight"));
    CHECK(is_range(P(t, "token_embd.weight", 1), 75968, 75968));
}

// ---------------------------------------------------------------------------
// Muse Glimmer: dense, the attention gate is a tensor, no GDN at all
// ---------------------------------------------------------------------------

void test_muse_gguf_r2() {
    const Table t = Table::build(cfg_muse(), 2, Conv::Gguf);
    // 40 full layers x (4 norms + 6 attn + 1 gate + 3 dense FFN + 4 shared) + 3
    CHECK(t.entries().size() == 723);

    // Every layer carries its attention gate; 16 q heads of 128 -> 1024 per rank.
    for (int i = 0; i < 40; i += 7) {
        CHECK(t.known(gg(i, "attn_gate.weight")));
        CHECK(P(t, gg(i, "attn_gate.weight"), i % 2).denom == 2048);  // qdim = 16 x 128
    }
    CHECK(is_range(P(t, gg(0, "attn_gate.weight"), 0), 0, 1024));
    CHECK(is_range(P(t, gg(0, "attn_gate.weight"), 1), 1024, 1024));

    // No GDN at all on a muse model, and no GDN layer names can exist.
    CHECK(!t.known(gg(0, "attn_qkv.weight")));
    CHECK(!t.known(gg(0, "ssm_conv1d.weight")));
    // Dense: no routed expert pools, no routed router (but the shared set is in).
    CHECK(!t.known(gg(0, "ffn_gate_exps.weight")));
    CHECK(!t.known(gg(0, "ffn_gate_inp.weight")));
    CHECK(t.known(gg(0, "ffn_gate_inp_shexp.weight")));
    CHECK(t.known(gg(0, "ffn_gate.weight")));

    // Q is the PLAIN qdim on a muse model (the gate is its own tensor).
    CHECK(P(t, gg(0, "attn_q.weight"), 0).denom == 2048);
    CHECK(is_range(P(t, gg(0, "attn_q.weight"), 0), 0, 1024));
    CHECK(is_range(P(t, gg(0, "attn_k.weight"), 0), 0, 128));  // 2 kv heads of 128
    CHECK(is_range(P(t, gg(0, "attn_k.weight"), 1), 128, 128));
}

// ---------------------------------------------------------------------------
// Classic (non-hybrid) routed MoE: the GGUF expert pools are in the table,
// the HF ones are intentionally absent (their file names are unverified in
// this repo) and fall to the replicated fallback
// ---------------------------------------------------------------------------

void test_classic_moe_r2() {
    {
        const Table t = Table::build(cfg_classic(), 2, Conv::Gguf);
        // 40 full layers x (4 + 6 + 4 routed + 4 shared) + 3
        CHECK(t.entries().size() == 723);
        CHECK(t.known(gg(0, "ffn_gate_exps.weight")));
        CHECK(P(t, gg(0, "ffn_down_exps.weight"), 0).axis == Axis::Experts);
        CHECK(is_range(P(t, gg(0, "ffn_down_exps.weight"), 0), 0, 128));
        CHECK(is_range(P(t, gg(0, "ffn_down_exps.weight"), 1), 128, 128));
        CHECK(!t.known(gg(0, "attn_gate.weight")));  // no muse gate on a classic model
        CHECK(t.known(gg(0, "ffn_gate_inp.weight")));  // the routed router is in
        // Classic non-hybrid: no GDN names, no folded gate in Q.
        CHECK(!t.known(gg(0, "attn_qkv.weight")));
        CHECK(P(t, gg(0, "attn_q.weight"), 0).denom == 6144);  // plain qdim: 48 x 128
    }
    {
        const Table t = Table::build(cfg_classic(), 2, Conv::Hf);
        // 40 full layers x (2 norms + 6 attn; NO MoE names, none shared) + 3
        CHECK(t.entries().size() == 323);
        // Any HF expert-pool tensor the file has is unknown here: replicated
        // (device -1) with a one-time caller warning, never silently split.
        const std::string pool = "model.language_model.layers.0.mlp.experts.gate_proj.weight";
        CHECK(!t.known(pool));
        CHECK(P(t, pool, 0).device == -1);
        // The two-tier fallback: a visual tensor is single-homed on rank 0 ...
        CHECK(P(t, "model.visual.blocks.0.attn.qkv.weight", 0).device == 0);
        CHECK(P(t, "model.visual.blocks.0.attn.qkv.weight", 0).axis == Axis::None);
        // ... everything else unknown is replicated on both ranks.
        CHECK(P(t, "some.unknown.tensor", 1).device == -1);
        CHECK(P(t, "mtp.layers.0.self_attn.q_proj.weight", 0).device == -1);
    }
}

// ---------------------------------------------------------------------------
// Pattern classes: every split name is (a) a single-range tiling or (b) the
// GDN three-range cover, all within [0, denom), ranks disjoint and covering
// ---------------------------------------------------------------------------

void test_pattern_classes() {
    const Table t = Table::build(cfg_27b(), 2, Conv::Hf);
    const Qwen35Config& c = t.cfg();
    const size_t lq = (size_t)c.linear_q_heads * c.linear_head_dim;  // 2048
    const size_t lv = (size_t)c.linear_v_heads * c.linear_head_dim;  // 6144
    size_t n_single = 0, n_gdn3 = 0, n_none = 0;
    for (const auto& e : t.entries()) {
        const Placement p0 = P(t, e.first, 0);
        const Placement p1 = P(t, e.first, 1);
        if (p0.axis == Axis::None) {
            ++n_none;
            CHECK(p0.ranges.empty() && p0.denom == 0);
            CHECK(p1.ranges.empty());
            continue;
        }
        CHECK(p0.denom != 0);
        CHECK(p1.denom == p0.denom);
        for (int r = 0; r < 2; ++r) {
            const Placement& p = r ? p1 : p0;
            for (const Range& rg : p.ranges)
                CHECK(rg.begin < p.denom && rg.begin + rg.len <= p.denom);
        }
        if (p0.ranges.size() == 1) {
            ++n_single;
            // rank 1 starts where rank 0 ends, and the two tile denom exactly
            CHECK(p0.ranges[0].begin == 0);
            CHECK(p0.ranges[0].begin + p0.ranges[0].len == p1.ranges[0].begin);
            CHECK(p1.ranges[0].begin + p1.ranges[0].len == p0.denom);
        } else if (p0.ranges.size() == 3) {
            ++n_gdn3;
            // each of the three [q | k | v] sections is tiled by the ranks
            for (int s = 0; s < 3; ++s) {
                const size_t base = s == 0 ? 0 : (s == 1 ? lq : 2 * lq);
                const size_t seclen = s < 2 ? lq : lv;
                CHECK(p0.ranges[s].begin + p0.ranges[s].len == p1.ranges[s].begin);
                CHECK(p1.ranges[s].begin + p1.ranges[s].len == base + seclen);
            }
        } else {
            CHECK(false);  // no other range count exists in this convention
        }
    }
    // 27B Hf: 16 full layers x 4 single-range attn names + 2 x 3-range GDN ...
    CHECK(n_single > 0);
    CHECK(n_gdn3 == 96);  // 48 GDN layers x 2 three-range kinds (qkv + conv)
    CHECK(n_none > 0);
}

// ---------------------------------------------------------------------------
// Rank symmetry: equal per-rank element counts, and per-name fractions that
// sum to 1 across the ranks for every split tensor
// ---------------------------------------------------------------------------

void test_rank_symmetry() {
    const Table tables[3] = {
        Table::build(cfg_27b(), 2, Conv::Hf),
        Table::build(cfg_35b(), 2, Conv::Gguf),
        Table::build(cfg_muse(), 2, Conv::Gguf),
    };
    for (const auto& t : tables) {
        CHECK(t.rank_elements(0) == t.rank_elements(1));
        for (const auto& e : t.entries()) {
            const Placement p0 = P(t, e.first, 0);
            const Placement p1 = P(t, e.first, 1);
            if (p0.denom == 0) continue;  // whole on its rank: fraction 1 each
            const double sum = p0.fraction() + p1.fraction();
            CHECK(sum > 0.9999 && sum < 1.0001);
        }
    }
}

// ---------------------------------------------------------------------------
// The tp=1 invariance: an unset table and a table built with one rank both
// hand EVERY name -- known or not -- the whole tensor on device 0
// ---------------------------------------------------------------------------

void test_tp1_degenerate() {
    {
        Table t;  // the process default before any load
        CHECK(!t.set());
        CHECK(t.n_ranks() == 1);
        for (const char* n : {"lm_head.weight", "model.visual.blocks.0.attn.qkv.weight",
                             "blk.0.attn_q.weight", "anything.at.all"}) {
            const Placement p = t.placement(n, 0);
            CHECK(p.device == 0 && p.axis == Axis::None && p.ranges.empty() && p.denom == 0);
        }
    }
    {
        const Table o = Table::build(cfg_27b(), 1, Conv::Hf);
        CHECK(o.set() && o.n_ranks() == 1);
        CHECK(o.known("lm_head.weight"));
        for (const char* n : {"lm_head.weight",  // known, but degenerate
                              "model.visual.blocks.0.attn.qkv.weight",  // two-tier: rank 0 whole
                              "mtp.layers.0.self_attn.q_proj.weight"}) {  // out-of-pattern:
            for (int r = -1; r < 3; ++r) {    // replicated, but clamped whole on rank 0
                const Placement p = o.placement(n, r);
                CHECK(p.device == 0 && p.axis == Axis::None && p.ranges.empty() && p.denom == 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The nvfp4 byte model and the companion-aware rank rule, unit checks
// ---------------------------------------------------------------------------

void test_nv4_and_rank_bytes() {
    // The 27B NVFP4 shapes, exact on-disk sizes (payload + 16-block scales +
    // the 8 B companion pair):
    CHECK(nv4(12288, 5120) == 35389448ull);    // Q (the [q | gate] width)
    CHECK(nv4(5120, 10240) == 29491208ull);   // GdnQKv / conv1d channels
    CHECK(nv4(5120, 6144) == 17694728ull);    // GdnGate / GdnOut
    CHECK(nv4(17408, 5120) == 50135048ull);   // one dense-FFN linear
    CHECK(nv4(248320, 5120) == 715161608ull); // lm_head
    CHECK(2ull * 248320 * 5120 == 2542796800ull);  // the bf16-equivalent head

    // The companion-aware rank rule: the 8 B pair is NEVER split -- a rank
    // owning any of the tensor keeps the whole pair.
    CHECK(rank_on_disk_bytes(WClass::Nvfp4, 100, 50, 100) == 54);  // 92 x 50/100 + 8
    CHECK(rank_on_disk_bytes(WClass::Nvfp4, 100, 0, 100) == 0);
    CHECK(rank_on_disk_bytes(WClass::Nvfp4, 100, 7, 0) == 100);    // denom 0: whole
    // bf16/f32 take a linear share, llrounded:
    CHECK(rank_on_disk_bytes(WClass::Bf16, 100, 54, 100) == 54);
    CHECK(rank_on_disk_bytes(WClass::Bf16, 100, 50, 100) == 50);
    CHECK(rank_on_disk_bytes(WClass::Bf16, 100, 0, 100) == 0);
    CHECK(rank_on_disk_bytes(WClass::Bf16, 100, 7, 0) == 100);
    CHECK(rank_on_disk_bytes(WClass::F32, 100, 33, 100) == 33);
}

int main() {
    test_27b_hf_r2();
    test_35b_gguf_r2();
    test_muse_gguf_r2();
    test_classic_moe_r2();
    test_pattern_classes();
    test_rank_symmetry();
    test_tp1_degenerate();
    test_nv4_and_rank_bytes();
    std::printf("tp_layout_cpu_test: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
