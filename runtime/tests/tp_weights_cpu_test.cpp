// (dual-GPU WP-6/CHG-0011, subagent H) Weight-loader tensor-parallel placement tests.
//
//   (a) The 27B Gguf placement table (tp=2): every tensor class' split axis / denominator /
//       ranges -- the 3-range GDN qkv/conv windows, Cols vs Rows vs OneD, per-rank whole norms,
//       the visual (device 0 on every rank) and unknown (replicated) fallbacks, and the dense
//       FFN Cols/Rows split.
//   (b) The 27B budget audit (D5): per-rank on-disk bytes, the rank-sum identity, the
//       G2-calibrated cross-reference, and the exact per-card audit lines at 131k (fits) and
//       262k (exceeds) context against a 16 GiB card.
//   (c) The tp=1 degenerate table: every name whole on device 0, axis None, no ranges -- the
//       loaders' byte-identical path.
//   (d) GPU one-shot (skip-guarded, exit 0 when skipped by design): a tiny hybrid model,
//       Flat-convention table, two Qwen35Model instances on two devices each loading their own
//       slice -- per-rank allocated bytes vs the table's own gather arithmetic, values
//       spot-checked against the file bytes the ranges select, print_tp_audit on both ranks,
//       and a third tp=1 instance reading back byte-identical to the files (FNV-1a).
//
// (a)-(c) and all the audit arithmetic are pure CPU and always run; on this box (an LLM holds
// the cards, no new context fits) only (d) SKIPs by design.

#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/tp_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace sparkinfer;
using namespace sparkinfer::tp;

static int failures = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            ++failures;                                                        \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
        }                                                                       \
    } while (0)

// The canonical Qwen3.8-27B shape (dense FFN, hybrid GDN, 4 KV heads) -- the exact config the
// dual-gpu/00-p0-probe budget table was calibrated against.
static Qwen35Config cfg27b() {
    Qwen35Config c;
    c.hybrid = true;
    c.qwen38 = true;
    c.n_layers = 64;
    c.full_attn_interval = 4;
    c.hidden = 5120;
    c.n_q_heads = 24;
    c.n_kv_heads = 4;
    c.head_dim = 256;
    c.dense_ffn = true;
    c.n_experts = 1;
    c.top_k = 1;
    c.n_shared = 0;
    c.moe_ffn = 17408;
    c.vocab = 248320;
    c.linear_q_heads = 16;
    c.linear_v_heads = 48;
    c.linear_head_dim = 128;
    c.linear_conv_kernel = 4;
    c.gdn_qh_block = true;
    return c;
}

static void check_place(const Table& t, const char* nm, int rank, int want_dev, int want_axis,
                        size_t want_denom, const std::vector<std::pair<size_t, size_t>>& want) {
    Placement p = t.placement(nm, rank);
    bool ok = p.device == want_dev && (int)p.axis == want_axis && p.denom == want_denom &&
              p.ranges.size() == want.size();
    if (ok)
        for (size_t i = 0; i < p.ranges.size(); ++i)
            if (p.ranges[i].begin != want[i].first || p.ranges[i].len != want[i].second) {
                ok = false;
                break;
            }
    if (!ok) {
        ++failures;
        std::printf("FAIL %s rank%d: got device=%d axis=%d denom=%zu ranges=[", nm, rank, p.device,
                    (int)p.axis, p.denom);
        for (const auto& r : p.ranges)
            std::printf("[%zu,%zu) ", r.begin, r.begin + r.len);
        std::printf("] want device=%d axis=%d denom=%zu\n", want_dev, want_axis, want_denom);
    }
}

// (a) tp=2, 27B Gguf convention: every placement the loaders will be driven by.
static void test_27b_placements() {
    const Table t = Table::build(cfg27b(), 2, Conv::Gguf);
    CHECK(t.n_ranks() == 2);

    // GDN layer 0: qkv/conv carry the 3-range (q|k|v) Cols windows; gate Cols; out Rows;
    // a/dt OneD; alpha/beta Cols; the per-rank norms stay whole on their own rank.
    check_place(t, "blk.0.attn_qkv.weight", 0, 0, 2, 10240, {{0, 1024}, {2048, 1024}, {4096, 3072}});
    check_place(t, "blk.0.attn_qkv.weight", 1, 1, 2, 10240, {{1024, 1024}, {3072, 1024}, {7168, 3072}});
    check_place(t, "blk.0.ssm_conv1d.weight", 0, 0, 2, 10240, {{0, 1024}, {2048, 1024}, {4096, 3072}});
    check_place(t, "blk.0.ssm_conv1d.weight", 1, 1, 2, 10240, {{1024, 1024}, {3072, 1024}, {7168, 3072}});
    check_place(t, "blk.0.attn_gate.weight", 0, 0, 2, 6144, {{0, 3072}});
    check_place(t, "blk.0.attn_gate.weight", 1, 1, 2, 6144, {{3072, 3072}});
    check_place(t, "blk.0.ssm_out.weight", 0, 0, 1, 6144, {{0, 3072}});
    check_place(t, "blk.0.ssm_out.weight", 1, 1, 1, 6144, {{3072, 3072}});
    check_place(t, "blk.0.ssm_a", 0, 0, 3, 48, {{0, 24}});
    check_place(t, "blk.0.ssm_a", 1, 1, 3, 48, {{24, 24}});
    check_place(t, "blk.0.ssm_dt.bias", 0, 0, 3, 48, {{0, 24}});
    check_place(t, "blk.0.ssm_dt.bias", 1, 1, 3, 48, {{24, 24}});
    check_place(t, "blk.0.ssm_alpha.weight", 0, 0, 2, 48, {{0, 24}});
    check_place(t, "blk.0.ssm_alpha.weight", 1, 1, 2, 48, {{24, 24}});
    check_place(t, "blk.0.ssm_beta.weight", 0, 0, 2, 48, {{0, 24}});
    check_place(t, "blk.0.ssm_beta.weight", 1, 1, 2, 48, {{24, 24}});
    check_place(t, "blk.0.ssm_norm.weight", 0, 0, 0, 0, {});
    check_place(t, "blk.0.ssm_norm.weight", 1, 1, 0, 0, {});
    check_place(t, "blk.0.attn_norm", 0, 0, 0, 0, {});
    check_place(t, "blk.0.attn_norm", 1, 1, 0, 0, {});
    // Dense-FFN 27B: no router/expert names exist in the table (dense branch) -> replicated.
    check_place(t, "blk.0.ffn_gate_inp.weight", 0, -1, 0, 0, {});
    check_place(t, "blk.0.ffn_gate_inp.weight", 1, -1, 0, 0, {});
    // Dense FFN: gate/up Cols over F, down Rows over F.
    check_place(t, "blk.0.ffn_gate.weight", 0, 0, 2, 17408, {{0, 8704}});
    check_place(t, "blk.0.ffn_gate.weight", 1, 1, 2, 17408, {{8704, 8704}});
    check_place(t, "blk.0.ffn_up.weight", 0, 0, 2, 17408, {{0, 8704}});
    check_place(t, "blk.0.ffn_up.weight", 1, 1, 2, 17408, {{8704, 8704}});
    check_place(t, "blk.0.ffn_down.weight", 0, 0, 1, 17408, {{0, 8704}});
    check_place(t, "blk.0.ffn_down.weight", 1, 1, 1, 17408, {{8704, 8704}});

    // Full-attention layer 3: Q Cols over 2*qdim (hybrid q_out_width), K/V Cols over kvdim,
    // O Rows over qdim; q/k norms whole on their own rank.
    check_place(t, "blk.3.attn_q.weight", 0, 0, 2, 12288, {{0, 6144}});
    check_place(t, "blk.3.attn_q.weight", 1, 1, 2, 12288, {{6144, 6144}});
    check_place(t, "blk.3.attn_k.weight", 0, 0, 2, 1024, {{0, 512}});
    check_place(t, "blk.3.attn_k.weight", 1, 1, 2, 1024, {{512, 512}});
    check_place(t, "blk.3.attn_v.weight", 0, 0, 2, 1024, {{0, 512}});
    check_place(t, "blk.3.attn_v.weight", 1, 1, 2, 1024, {{512, 512}});
    check_place(t, "blk.3.attn_output.weight", 0, 0, 1, 6144, {{0, 3072}});
    check_place(t, "blk.3.attn_output.weight", 1, 1, 1, 6144, {{3072, 3072}});
    check_place(t, "blk.3.attn_q_norm.weight", 0, 0, 0, 0, {});
    check_place(t, "blk.3.attn_q_norm.weight", 1, 1, 0, 0, {});
    check_place(t, "blk.3.attn_k_norm.weight", 0, 0, 0, 0, {});
    check_place(t, "blk.3.attn_k_norm.weight", 1, 1, 0, 0, {});
    check_place(t, "blk.3.attn_post_norm", 0, 0, 0, 0, {});
    check_place(t, "blk.3.attn_post_norm", 1, 1, 0, 0, {});
    check_place(t, "blk.3.ffn_norm", 0, 0, 0, 0, {});
    check_place(t, "blk.3.ffn_norm", 1, 1, 0, 0, {});
    check_place(t, "blk.3.post_ffw_norm", 0, 0, 0, 0, {});
    check_place(t, "blk.3.post_ffw_norm", 1, 1, 0, 0, {});

    // Top level: embed / lm_head Rows over the vocab, final norm per-rank whole, the vision
    // tower pinned to device 0 on EVERY rank, unknown names replicated.
    check_place(t, "token_embd.weight", 0, 0, 1, 248320, {{0, 124160}});
    check_place(t, "token_embd.weight", 1, 1, 1, 248320, {{124160, 124160}});
    check_place(t, "output.weight", 0, 0, 1, 248320, {{0, 124160}});
    check_place(t, "output.weight", 1, 1, 1, 248320, {{124160, 124160}});
    check_place(t, "output_norm.weight", 0, 0, 0, 0, {});
    check_place(t, "output_norm.weight", 1, 1, 0, 0, {});
    check_place(t, "model.visual.blocks.0.attn.qkv", 0, 0, 0, 0, {});
    check_place(t, "model.visual.blocks.0.attn.qkv", 1, 0, 0, 0, {});
    check_place(t, "some.unknown.tensor", 0, -1, 0, 0, {});
    check_place(t, "some.unknown.tensor", 1, -1, 0, 0, {});
}

// (b) The D5 per-card budget audit for 27B @ tp=2.
static void test_27b_audit() {
    const Table t = Table::build(cfg27b(), 2, Conv::Gguf);
    CHECK(t.total_on_disk_bytes() == 16995666056ull);
    CHECK(t.rank_on_disk_bytes(0) == 8499164808ull);
    CHECK(t.rank_on_disk_bytes(1) == 8499164808ull);

    // Rank-sum identity: rank0 + rank1 - total == (whole-on-every-rank bytes) + 3,208 B of
    // NVFP4 scale/companion overhead in the table's own byte model (probe-locked, 64 layers).
    unsigned long long wholeboth = 0;
    for (const auto& e : t.entries()) {
        Placement p0 = t.placement(e.first, 0);
        Placement p1 = t.placement(e.first, 1);
        if (p0.axis == Axis::None && p1.axis == Axis::None)
            wholeboth += (unsigned long long)t.on_disk_bytes(e.first);
    }
    CHECK(wholeboth == 2660352ull);
    const unsigned long long rsum = (unsigned long long)t.rank_on_disk_bytes(0) +
                                    (unsigned long long)t.rank_on_disk_bytes(1) -
                                    (unsigned long long)t.total_on_disk_bytes();
    CHECK(rsum - wholeboth == 3208ull);
    std::printf(
        "[tp-weights-cpu-test] 27B Gguf table (tp=2): total %llu B (G2 file-inspection "
        "calibrated 16,994,355,336; delta %llu B = the table's 4 hidden-norm slots/layer vs the "
        "27B file's 2), per-rank %llu B, r0+r1-total-wholeboth %llu\n",
        (unsigned long long)t.total_on_disk_bytes(),
        (unsigned long long)t.total_on_disk_bytes() - 16994355336ull,
        (unsigned long long)t.rank_on_disk_bytes(0), (unsigned long long)(rsum - wholeboth));

    // The exact audit line (D5), nominal 16 GiB card. The test computes est/verdict
    // independently; tp_audit_line must agree byte-for-byte.
    const unsigned long long od = 8499164808ull;
    const auto est = [&](unsigned long long ctx) {
        return od + 33024ull * ctx + 154927104ull;   // G2: 33,024 B/token KV + GDN per-seq state
    };
    const unsigned long long card = 17179869184ull;  // 16 GiB
    char exp[320];
    std::snprintf(exp, sizeof exp,
                  "[tp-audit] rank 0 (device 0): on-disk 8499164808 B, card free 17179869184/"
                  "17179869184 B, per-card est at ctx 131072 = %llu B -> fits 16 GiB "
                  "(17,179,869,184 B)",
                  est(131072));
    CHECK(tp_audit_line(0, 0, od, card, card, 131072) == std::string(exp));
    std::snprintf(exp, sizeof exp,
                  "[tp-audit] rank 1 (device 1): on-disk 8499164808 B, card free 17179869184/"
                  "17179869184 B, per-card est at ctx 262144 = %llu B -> exceeds 16 GiB "
                  "(17,179,869,184 B)",
                  est(262144));
    CHECK(tp_audit_line(1, 1, od, card, card, 262144) == std::string(exp));
    CHECK(est(131072) == 12982613640ull);
    CHECK(est(262144) == 17311135368ull);
    CHECK(est(262144) - card == 131266184ull);
    CHECK(est(131072) <= card && est(262144) > card);
    std::printf("%s\n", tp_audit_line(0, 0, od, card, card, 131072).c_str());
    std::printf("%s\n", tp_audit_line(1, 1, od, card, card, 262144).c_str());
}

// (c) tp=1: the degenerate table is the byte-identical path.
static void test_tp1_degenerate() {
    const Table t = Table::build(cfg27b(), 1, Conv::Gguf);
    CHECK(t.n_ranks() == 1);
    for (const auto& e : t.entries()) {
        Placement p = t.placement(e.first, 0);
        if (p.device != 0 || p.axis != Axis::None || !p.ranges.empty()) {
            ++failures;
            std::printf("FAIL tp=1 %s: device=%d axis=%d (want 0/None/empty)\n", e.first.c_str(),
                        p.device, (int)p.axis);
            break;
        }
    }
    Placement u = t.placement("some.unknown.tensor", 0);
    CHECK(u.device == 0 && u.axis == Axis::None && u.ranges.empty());
    Placement v = t.placement("model.visual.blocks.0.attn.qkv", 0);
    CHECK(v.device == 0 && v.axis == Axis::None && v.ranges.empty());
}

// ---------------------------------------------------------------------------
// (d) GPU one-shot: tiny hybrid model, Flat table, two instances on two cards.
// ---------------------------------------------------------------------------

static Qwen35Config cfgSmall() {
    Qwen35Config c;
    c.hybrid = true;
    c.n_layers = 2;              // layer 0 GDN, layer 1 full attention (interval 2)
    c.full_attn_interval = 2;
    c.hidden = 64;
    c.n_q_heads = 4;
    c.n_kv_heads = 1;
    c.head_dim = 16;
    c.linear_q_heads = 4;
    c.linear_v_heads = 8;
    c.linear_head_dim = 16;
    c.linear_conv_kernel = 4;
    c.dense_ffn = true;
    c.n_experts = 1;
    c.top_k = 1;
    c.n_shared = 0;
    c.moe_ffn = 128;
    c.vocab = 64;   // == hidden, on purpose: the flat file stores embed/lm as [hidden fast,
                    // vocab] while the table's Rows ranges address the fast axis, so keeping
                    // vocab == hidden (64) keeps every table range in-bounds of the file.
    c.gdn_qh_block = true;
    return c;
}

struct FlatFile {
    std::string name;
    size_t d0, d1;   // file layout: d0 (fast) x d1 (slow), raw bf16
};

static std::vector<FlatFile> flat_files(const Qwen35Config& c) {
    const size_t H = c.hidden, V = c.vocab, F = c.moe_ffn;
    const size_t qout = (size_t)c.n_q_heads * c.head_dim;
    const size_t kvout = (size_t)c.n_kv_heads * c.head_dim;
    std::vector<FlatFile> v;
    v.push_back({"embed_tokens", H, V});
    v.push_back({"final_norm", H, 1});
    v.push_back({"lm_head", H, V});
    for (int i = 0; i < c.n_layers; ++i) {
        char pfx[32];
        std::snprintf(pfx, sizeof pfx, "layer_%d.", i);
        v.push_back({std::string(pfx) + "input_norm", H, 1});
        v.push_back({std::string(pfx) + "wq", H, qout});
        v.push_back({std::string(pfx) + "wk", H, kvout});
        v.push_back({std::string(pfx) + "wv", H, kvout});
        v.push_back({std::string(pfx) + "wo", qout, H});
        v.push_back({std::string(pfx) + "q_norm", (size_t)c.head_dim, 1});
        v.push_back({std::string(pfx) + "k_norm", (size_t)c.head_dim, 1});
        v.push_back({std::string(pfx) + "post_attn_norm", H, 1});
        v.push_back({std::string(pfx) + "router_w", H, (size_t)c.n_experts});
        v.push_back({std::string(pfx) + "gate", H, F});
        v.push_back({std::string(pfx) + "up", H, F});
        v.push_back({std::string(pfx) + "down", F, H});
    }
    return v;
}

static void fill_bf16(std::vector<uint16_t>& v, uint32_t seed) {
    uint32_t x = seed | 1u;
    for (auto& e : v) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        float f = (float)(x % 100000) / 100000.0f - 0.5f;   // deterministic, valid bf16 bits
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof f);
        e = (uint16_t)(bits >> 16);
    }
}

static uint32_t fnv1a_step(uint32_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 16777619u;
    }
    return h;
}

static const uint8_t* dev_read(const void* src, size_t n) {
    static thread_local std::vector<uint8_t> buf;
    buf.resize(n);
    if (src == nullptr || cudaMemcpy(buf.data(), src, n, cudaMemcpyDeviceToHost) != cudaSuccess)
        return nullptr;
    return buf.data();
}

static void check_bytes(const char* what, const uint8_t* got, const uint8_t* want, size_t n) {
    if (std::memcmp(got, want, n) != 0) {
        ++failures;
        std::printf("FAIL %s: mismatch in %zu bytes\n", what, n);
    }
}

// What the table says this rank owns, in file bytes: replicated (device -1) or whole
// (axis None / no ranges) -> the whole file; Cols -> d0 x range-len; Rows -> d1 x range-len;
// assigned to another rank -> nothing.
static size_t expected_owned(const Table& t, const FlatFile& f, int rank) {
    const Placement p = t.placement(f.name, rank);
    const size_t file_bytes = f.d0 * f.d1 * 2;
    if (p.device < 0) return file_bytes;
    if (p.device != rank) return 0;
    if (p.axis == Axis::None || p.ranges.empty()) return file_bytes;
    size_t len = 0;
    for (const auto& r : p.ranges) len += r.len;
    if (p.axis == Axis::Cols) return f.d0 * len * 2;
    if (p.axis == Axis::Rows) return f.d1 * len * 2;
    return file_bytes;
}

static void test_gpu_oneshot() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) {
        std::printf("[tp-weights-cpu-test] GPU one-shot: SKIP (fewer than 2 CUDA devices)\n");
        return;
    }
    for (int d = 0; d < 2; ++d) {
        size_t free_b = 0, total_b = 0;
        if (cudaSetDevice(d) != cudaSuccess ||
            cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b < ((size_t)24 << 20)) {
            std::printf("[tp-weights-cpu-test] GPU one-shot: SKIP (device %d: no context or "
                        "<24 MiB free)\n",
                        d);
            cudaSetDevice(0);
            return;
        }
    }
    cudaSetDevice(0);

    const Qwen35Config cfg = cfgSmall();
    const std::vector<FlatFile> files = flat_files(cfg);

    // Deterministic weight directory under /tmp.
    char tmpl[] = "/tmp/tpw.XXXXXX";
    const char* dirp = mkdtemp(tmpl);
    if (!dirp) {
        std::printf("[tp-weights-cpu-test] GPU one-shot: SKIP (mkdtemp failed)\n");
        return;
    }
    const std::string dir(dirp);
    std::vector<std::vector<uint8_t>> fbytes(files.size());
    uint32_t ref_fnv = 2166136261u;
    for (size_t i = 0; i < files.size(); ++i) {
        std::vector<uint16_t> v(files[i].d0 * files[i].d1);
        fill_bf16(v, 0x9E37u + (uint32_t)i * 7919u);
        fbytes[i].assign((const uint8_t*)v.data(), (const uint8_t*)v.data() + v.size() * 2);
        ref_fnv = fnv1a_step(ref_fnv, fbytes[i].data(), fbytes[i].size());
        std::string path = dir + "/" + files[i].name + ".bin";
        std::FILE* fp = std::fopen(path.c_str(), "wb");
        if (!fp || std::fwrite(fbytes[i].data(), 1, fbytes[i].size(), fp) != fbytes[i].size()) {
            if (fp) std::fclose(fp);
            std::printf("[tp-weights-cpu-test] GPU one-shot: SKIP (cannot write %s)\n",
                        path.c_str());
            std::filesystem::remove_all(dir);
            return;
        }
        std::fclose(fp);
    }
    std::printf("[tp-weights-cpu-test] wrote %zu .bin files under %s, ref FNV-1a = %08X\n",
                files.size(), dir.c_str(), ref_fnv);

    // --- tp=2: rank 0 on device 0, rank 1 on device 1 ---
    const Table t2 = Table::build(cfg, 2, Conv::Flat);
    set_process_table(t2);
    {
        cudaSetDevice(0);
        auto m = std::make_unique<Qwen35Model>(cfg, nullptr, nullptr, GdnStateWindow{}, 0, 0);
        size_t fa = 0, ta = 0, fb = 0, tb = 0;
        CHECK(cudaMemGetInfo(&fa, &ta) == cudaSuccess);
        CHECK(m->load_weights(dir));
        CHECK(cudaMemGetInfo(&fb, &tb) == cudaSuccess);
        size_t delta = fa - fb;
        size_t expected = 0;
        for (const auto& f : files) expected += expected_owned(t2, f, 0);
        CHECK(delta == expected || (delta >= expected && delta - expected <= (1u << 20)));
        std::printf("[tp-weights-cpu-test] rank0 (dev 0): allocated %zu B (table-gather "
                    "expected %zu B)\n",
                    delta, expected);
        // Spot checks: Cols slices are file prefixes, Rows/OneD-style gather per row.
        const Qwen35Weights& w = m->weights();
        check_bytes("r0 layer_0.wq prefix", dev_read(w.layers[0].wq, 4096),
                    fbytes[4].data(), 4096);
        check_bytes("r0 layer_0.wk prefix", dev_read(w.layers[0].wk, 1024), fbytes[5].data(),
                    1024);
        check_bytes("r0 layer_0.gate prefix", dev_read(w.layers[0].gate, 8192), fbytes[12].data(),
                    8192);
        check_bytes("r0 final_norm whole", dev_read(w.final_norm, 128), fbytes[1].data(), 128);
        // embed/lm: per-row [0,32) element slice (d0 = d1 = 64, row stride 128 B).
        std::vector<uint8_t> expv(8192);
        for (int row = 0; row < 64; ++row)
            std::memcpy(&expv[row * 128], fbytes[0].data() + row * 128, 64);
        check_bytes("r0 embed_tokens rows[0,32)", dev_read(w.embed_tokens, 8192), expv.data(),
                    expv.size());
        for (int row = 0; row < 64; ++row)
            std::memcpy(&expv[row * 128], fbytes[2].data() + row * 128, 64);
        check_bytes("r0 lm_head rows[0,32)", dev_read(w.lm_head, 8192), expv.data(), expv.size());
        // down: Rows [0,64) over d0=128, 64 slow rows (row stride 256 B).
        for (int row = 0; row < 64; ++row)
            std::memcpy(&expv[row * 256], fbytes[14].data() + row * 256, 128);
        check_bytes("r0 layer_0.down rows[0,64)", dev_read(w.layers[0].down, 16384), expv.data(),
                    expv.size());
        m->print_tp_audit(131072);
    }   // m destroyed with the context on device 0
    {
        cudaSetDevice(1);
        auto m = std::make_unique<Qwen35Model>(cfg, nullptr, nullptr, GdnStateWindow{}, 1, 1);
        size_t fa = 0, ta = 0, fb = 0, tb = 0;
        CHECK(cudaMemGetInfo(&fa, &ta) == cudaSuccess);
        CHECK(m->load_weights(dir));
        CHECK(cudaMemGetInfo(&fb, &tb) == cudaSuccess);
        size_t delta = fa - fb;
        size_t expected = 0;
        for (const auto& f : files) expected += expected_owned(t2, f, 1);
        CHECK(delta == expected || (delta >= expected && delta - expected <= (1u << 20)));
        std::printf("[tp-weights-cpu-test] rank1 (dev 1): allocated %zu B (table-gather "
                    "expected %zu B)\n",
                    delta, expected);
        const Qwen35Weights& w = m->weights();
        // Cols [32,64) of wq -> file bytes [4096, 8192); wk [8,16) -> [1024, 2048);
        // gate [64,128) -> [8192, 16384).
        check_bytes("r1 layer_0.wq mid", dev_read(w.layers[0].wq, 4096),
                    fbytes[4].data() + 4096, 4096);
        check_bytes("r1 layer_0.wk mid", dev_read(w.layers[0].wk, 1024), fbytes[5].data() + 1024,
                    1024);
        check_bytes("r1 layer_0.gate mid", dev_read(w.layers[0].gate, 8192),
                    fbytes[12].data() + 8192, 8192);
        // Per-row [32,64) element slices (offset 64, length 64 in 128-B rows).
        std::vector<uint8_t> expv(8192);
        for (int row = 0; row < 64; ++row)
            std::memcpy(&expv[row * 128], fbytes[0].data() + row * 128 + 64, 64);
        check_bytes("r1 embed_tokens rows[32,64)", dev_read(w.embed_tokens, 8192), expv.data(),
                    expv.size());
        for (int row = 0; row < 64; ++row)
            std::memcpy(&expv[row * 128], fbytes[2].data() + row * 128 + 64, 64);
        check_bytes("r1 lm_head rows[32,64)", dev_read(w.lm_head, 8192), expv.data(), expv.size());
        for (int row = 0; row < 64; ++row)
            std::memcpy(&expv[row * 256], fbytes[14].data() + row * 256 + 128, 128);
        check_bytes("r1 layer_0.down rows[64,128)", dev_read(w.layers[0].down, 16384), expv.data(),
                    expv.size());
        m->print_tp_audit(131072);
    }   // m destroyed with the context on device 1
    set_process_table(Table());   // restore: unset

    // --- tp=1: the byte-identity read-back (FNV-1a over every loaded buffer vs the files) ---
    {
        const Table t1 = Table::build(cfg, 1, Conv::Flat);
        set_process_table(t1);
        cudaSetDevice(0);
        auto m = std::make_unique<Qwen35Model>(cfg, nullptr, nullptr, GdnStateWindow{}, 0, 0);
        CHECK(m->load_weights(dir));
        const Qwen35Weights& w = m->weights();
        uint32_t h = 2166136261u;
        h = fnv1a_step(h, w.embed_tokens, fbytes[0].size());
        h = fnv1a_step(h, w.final_norm, fbytes[1].size());
        h = fnv1a_step(h, w.lm_head, fbytes[2].size());
        for (int i = 0; i < cfg.n_layers; ++i) {
            const Qwen35LayerWeights& lw = w.layers[i];
            h = fnv1a_step(h, lw.input_norm, fbytes[3 + 12 * i].size());
            h = fnv1a_step(h, lw.wq, fbytes[4 + 12 * i].size());
            h = fnv1a_step(h, lw.wk, fbytes[5 + 12 * i].size());
            h = fnv1a_step(h, lw.wv, fbytes[6 + 12 * i].size());
            h = fnv1a_step(h, lw.wo, fbytes[7 + 12 * i].size());
            h = fnv1a_step(h, lw.q_norm, fbytes[8 + 12 * i].size());
            h = fnv1a_step(h, lw.k_norm, fbytes[9 + 12 * i].size());
            h = fnv1a_step(h, lw.post_attn_norm, fbytes[10 + 12 * i].size());
            h = fnv1a_step(h, lw.router_w, fbytes[11 + 12 * i].size());
            h = fnv1a_step(h, lw.gate, fbytes[12 + 12 * i].size());
            h = fnv1a_step(h, lw.up, fbytes[13 + 12 * i].size());
            h = fnv1a_step(h, lw.down, fbytes[14 + 12 * i].size());
        }
        std::printf("[tp-weights-cpu-test] tp=1 read-back FNV-1a = %08X (ref %08X)\n", h,
                    ref_fnv);
        CHECK(h == ref_fnv);
        set_process_table(Table());
    }
    std::filesystem::remove_all(dir);
}

int main() {
    std::printf("[tp-weights-cpu-test] start\n");
    test_27b_placements();
    test_27b_audit();
    test_tp1_degenerate();
    test_gpu_oneshot();
    std::printf("[tp-weights-cpu-test] %s (%d failures)\n", failures ? "FAIL" : "OK", failures);
    return failures;
}
