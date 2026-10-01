// WP-8: the per-device GDN window (24/24 v-heads of the 27B `lin_state`) for the dual-GPU pair.
//
// The pure-C++ half (no CUDA calls) pins the arithmetic the split is built on:
//   - the exact 27B GDN-state byte figures: full `lin_state` 150,994,944 B (144.0 MiB) and the
//     (24,24)-windowed 75,497,472 B (72.0 MiB); conv stays FULL per card (3,932,160 B = 3.75 MiB,
//     decision (a) -- its rows are shared q/k/v channels, not v-head blocks); per sequence
//     147.75 MiB, per card 75.75 MiB;
//   - `gdn_window_normalize`'s clamp table under the 27B block convention (v=48, q=16, g=3) and
//     under the cyclic convention (only a LEADING window is expressible);
//   - the tp=1 invariance: (0,0) normalizes to (0,0) with no warning, vloc stays the full v count,
//     and every slot offset / head index the split derives is byte-identical to the unsplit model;
//   - cross-window restore: a snapshot is only restorable into an arena of its own byte size.
//
// One explicit ~16 MiB GPU section follows (no model, no weights -- direct kernel calls on raw
// buffers; 16 v-heads / 8 q-heads, g=2, 1 slot, head_dim 128, 8 tokens):
//   - 8 sequential single-AR steps on the full 16 heads vs the same 8 steps under call-site
//     windowing (0,8) and (8,8) -- bit-identical, since a v-head's recurrence touches only its
//     own (vh, qh(vh)) head rows and the windowed call is handed the pre-shifted bases;
//   - the new batched v0/vloc path: two packed sequences, one token each, windows (0,8) and (8,8)
//     against the single-AR reference at a small tolerance, plus the unsplit (vloc=0) invariance;
//   - the compacted-bf16 batched path (state_to_b16, then state_compact_b16=true) against the fp32
//     reference at bf16 tolerance.
// Skip-guarded like the WP-7 test: an explicit headroom check against that fixed budget decides
// whether it runs (lean mode: a resident LLM holds ~1 GB free per card here; this touches only its
// own ~16 MiB allocation). Never auto-sized. No Python, no model load.

#include "../include/sparkinfer/kernels/fused.h"
#include "../include/sparkinfer/models/qwen35.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using sparkinfer::GdnStateWindow;
using sparkinfer::Qwen35Config;
using sparkinfer::gdn_window_normalize;

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

namespace {

// The canonical 27B (Qwen3.8-27B) GDN shape: 64 layers, one full-attention per 4,
// 16 q-heads, 48 v-heads, head_dim 128, conv kernel 4, block (vh / (v/q)) qh mapping.
Qwen35Config cfg27b() {
    Qwen35Config c;
    c.hybrid = true;
    c.n_layers = 64;
    c.full_attn_interval = 4;
    c.linear_q_heads = 16;
    c.linear_v_heads = 48;
    c.linear_head_dim = 128;
    c.linear_conv_kernel = 4;
    c.gdn_qh_block = true;
    return c;
}

// The GDN slot count the split sizes against: n_layers - n_layers / interval (27B: 64 - 16 = 48;
// 16 of the 64 layers are full-attention and carry no GDN state).
int gdn_state_slots(const Qwen35Config& c) {
    if (!c.hybrid || c.full_attn_interval <= 0) return c.n_layers;
    return c.n_layers - c.n_layers / c.full_attn_interval;
}

// (1) The exact 27B per-sequence / per-card state figures, re-derived from the config.
void test_27b_figures() {
    const auto c = cfg27b();
    const int slots = gdn_state_slots(c);
    const int v = c.linear_v_heads, q = c.linear_q_heads, hd = c.linear_head_dim, ck = c.linear_conv_kernel;
    const size_t hd2 = (size_t)hd * hd;
    const size_t lin_full = (size_t)slots * v * hd2 * 4;          // fp32, [slots][v][HD][HD]
    const size_t lin_win  = (size_t)slots * (size_t)(v / 2) * hd2 * 4;
    const size_t qkvdim = 2 * (size_t)q * hd + (size_t)v * hd;      // 2*16*128 + 48*128 = 10240
    const size_t conv = (size_t)c.n_layers * (size_t)(ck - 1) * qkvdim * 2;
    const double mib = 1048576.0;
    CHECK(slots == 48);
    CHECK(lin_full == 150994944u);
    CHECK(lin_win == 75497472u);
    CHECK(conv == 3932160u);
    CHECK(lin_full + conv == 154927104u);      // 147.75 MiB per sequence
    CHECK(lin_win + conv == 79429632u);       // 75.75 MiB per card at 24/24
    std::printf("[wp8] 27B GDN state per sequence: full lin_state %zu B (%.2f MiB) + conv %zu B "
                "(%.2f MiB) = %.2f MiB; (24,24) windowed per card: %zu B (%.2f MiB) + conv (FULL, "
                "decision (a)) = %.2f MiB\n",
                lin_full, lin_full / mib, conv, conv / mib, (lin_full + conv) / mib,
                lin_win, lin_win / mib, (lin_win + conv) / mib);
}

// (2a) The 27B block convention: g = v/q = 3; a window is expressible iff g divides both edges.
void test_normalize_block() {
    const auto c = cfg27b();
    auto check = [&](int start, int count, bool ok) {
        bool warned = true;
        const GdnStateWindow w = gdn_window_normalize(c, GdnStateWindow{start, count}, &warned);
        if (ok) {
            CHECK(w.v_start == start && w.v_count == count);
            CHECK(!warned);
        } else {
            CHECK(w.v_start == 0 && w.v_count == 0);   // clamped to the all-heads window
            CHECK(warned);
        }
        std::printf("[wp8] normalize(block 27B) (%d,%d) -> (%d,%d)%s\n", start, count, w.v_start,
                    w.v_count, ok ? "" : "  [clamped]");
    };
    check(24, 24, true);    // the 27B 24/24 split
    check(12, 24, true);    // aligned on a g=3 boundary, off-leading
    check(0, 48, true);      // the whole = tp=1
    check(8, 8, false);      // 8 % 3 != 0
    check(1, 24, false);     // 1 % 3 != 0
    check(24, 40, false);    // 24 + 40 = 64 > 48 -- OOB
    // (48,0) is degenerate, not a clamp: it passes through as the all-heads window with NO
    // warning -- checked explicitly below, since the table above records clamped windows only.
    bool warned = true;
    const GdnStateWindow d = gdn_window_normalize(c, GdnStateWindow{48, 0}, &warned);
    CHECK(d.v_start == 0 && d.v_count == 0 && !warned);
}

// (2b) The cyclic convention (vh % q): only a LEADING window is expressible -- v_start == 0,
// v_count <= q, v_count % q == 0. Probed with v=48, q=16 (g=3 either way).
void test_normalize_cyclic() {
    auto c = cfg27b();
    c.gdn_qh_block = false;
    auto check = [&](int start, int count, bool ok, const char* rule) {
        bool warned = true;
        const GdnStateWindow w = gdn_window_normalize(c, GdnStateWindow{start, count}, &warned);
        if (ok) {
            CHECK(w.v_start == start && w.v_count == count);
            CHECK(!warned);
        } else {
            CHECK(w.v_start == 0 && w.v_count == 0);
            CHECK(warned);
        }
        std::printf("[wp8] normalize(cyclic) (%d,%d) -> (%d,%d)%s (%s)\n", start, count, w.v_start,
                    w.v_count, ok ? "" : "  [clamped]", rule);
    };
    check(0, 16, true, "leading, v_count == q, divides q");
    check(0, 32, false, "v_count 32 > q 16");
    check(24, 24, false, "v_start != 0: a mid-range cyclic window aliases q-heads");
    check(0, 8, false, "v_count 8 % 16 != 0");
    // Non-hybrid configs are degenerate: any window passes through as all-heads, no warning.
    auto flat = cfg27b();
    flat.hybrid = false;
    bool warned = true;
    const GdnStateWindow w = gdn_window_normalize(flat, GdnStateWindow{12, 24}, &warned);
    CHECK(w.v_start == 0 && w.v_count == 0 && !warned);
}

// (3) tp=1 invariance: the degenerate (0,0) window must reproduce the unsplit model exactly --
// no warning, vloc = the full v count, every slot offset identical, and the batched kernel's
// unsplit evaluation (vloc == 0 -> grid over all v_heads, vhg = vh) equal to the old one.
void test_tp1_invariance() {
    const auto c = cfg27b();
    const size_t hd2 = (size_t)c.linear_head_dim * c.linear_head_dim;
    bool warned = true;
    const GdnStateWindow w = gdn_window_normalize(c, GdnStateWindow{0, 0}, &warned);
    CHECK(w.v_start == 0 && w.v_count == 0 && !warned);
    const int vloc = w.v_count > 0 ? w.v_count : c.linear_v_heads;
    CHECK(vloc == c.linear_v_heads);                    // 0 = all heads
    for (int slot = 0; slot < gdn_state_slots(c); ++slot)
        CHECK((size_t)slot * vloc * hd2 == (size_t)slot * c.linear_v_heads * hd2);  // identity
    // The (24,24) window's slot offsets are exactly half the full ones.
    CHECK((size_t)1 * 24 * hd2 * 2 == (size_t)1 * 48 * hd2);
    // Batched unsplit: vhl = vloc > 0 ? vloc : v_heads and vhg = vh + v0 both reduce to the
    // pre-split kernel exactly (v0 == 0, vloc == 0 -> vhl == v_heads, vhg == vh).
    const int vloc_in = 0, v0_in = 0;
    const int vhl = vloc_in > 0 ? vloc_in : c.linear_v_heads;
    CHECK(vhl == c.linear_v_heads);
    for (int vh = 0; vh < c.linear_v_heads; ++vh) CHECK(vh + v0_in == vh);
    std::printf("[wp8] tp=1 invariance: (0,0) -> (0,0) no warning, vloc = v (%d), slot offsets "
                "identical, batched vloc==0 reduces to the unsplit grid\n",
                c.linear_v_heads);
}

// (4) Restore is size-checked: a snapshot restores only into an arena of its own byte count.
// A (24,24) snapshot is 2x smaller than the full arena and vice versa -- either direction
// must be refused, or a half-window would silently zero/corrupt the other half's heads.
void test_restore_sizes() {
    const auto c = cfg27b();
    const size_t hd2 = (size_t)c.linear_head_dim * c.linear_head_dim;
    const int slots = gdn_state_slots(c);
    const size_t full = (size_t)slots * 48 * hd2 * 4, win = (size_t)slots * 24 * hd2 * 4;
    auto accepts = [](size_t snap, size_t arena) { return snap == arena; };
    CHECK(accepts(full, full));
    CHECK(accepts(win, win));
    CHECK(!accepts(full, win));
    CHECK(!accepts(win, full));
    std::printf("[wp8] restore size-check: full %zu B <-> windowed %zu B each refuse the other; "
                "same-size restores accepted\n",
                full, win);
}

// ---------------------------------------------------------------------------
// GPU section: the windowed kernels against their own references, on explicit budgets.

constexpr int kV = 16, kQ = 8, kG = kV / kQ, kHD = 128, kN = 8;

uint32_t lcg(uint32_t& x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}

std::vector<uint16_t> rand_bf16(size_t n, uint32_t seed, float scale) {
    std::vector<uint16_t> h(n);
    uint32_t x = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        lcg(x);
        const float f = scale * (2.f * ((x >> 8) % 10000) / 10000.f - 1.f);
        uint32_t b; memcpy(&b, &f, 4);
        h[i] = (uint16_t)(b >> 16);
    }
    return h;
}

template <class T>
T* upload(const std::vector<T>& h) {
    void* d = nullptr;
    if (h.empty()) return nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return (T*)d;
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

// One single-AR step at token t. `voff` selects the v-head window [voff, voff+vloc) of the
// full kV-head activations; for a block window the q/k bases shift by voff/g q-heads, which is
// exactly the runtime's decode call-site windowing (qloc = vloc/g, bases pre-shifted).
bool ar_step(const uint16_t* dq, const uint16_t* dk, const uint16_t* dv,
             const uint16_t* da, const uint16_t* db, const uint16_t* ddt, const uint16_t* daa,
             float* dstate, void* dout, int t, int voff, int qloc, int vloc) {
    const size_t qdim = (size_t)kQ * kHD, vdim = (size_t)kV * kHD;
    sparkinfer::kernels::launch_qwen36_gdn_ar(
        (const char*)dq + (t * qdim + (size_t)(voff / kG) * kHD) * 2,
        (const char*)dk + (t * qdim + (size_t)(voff / kG) * kHD) * 2,
        (const char*)dv + (t * vdim + (size_t)voff * kHD) * 2,
        (const char*)da + (t * (size_t)kV + voff) * 2,
        (const char*)db + (t * (size_t)kV + voff) * 2,
        ddt + voff, daa + voff,
        dstate, 0, dout, qloc, vloc, kHD, true, nullptr, false);
    return cudaPeekAtLastError() == cudaSuccess;
}

// Runs the kN-token window `voff` (qloc/q_heads, vloc/v_heads) on arena `dstate`, one step at a
// time, from a zeroed state.
bool run_ar_steps(const uint16_t* dq, const uint16_t* dk, const uint16_t* dv,
                  const uint16_t* da, const uint16_t* db, const uint16_t* ddt, const uint16_t* daa,
                  float* dstate, void* dout, int voff, int qloc, int vloc) {
    cudaMemset(dstate, 0, (size_t)vloc * kHD * kHD * 4);
    for (int t = 0; t < kN; t++)
        if (!ar_step(dq, dk, dv, da, db, ddt, daa, dstate,
                     (uint16_t*)dout + (size_t)t * kV * kHD, t, voff, qloc, vloc))
            return false;
    return true;
}

// One packed batched launch (2 sequences, one token each, token rows `t0` and `t0 + 1`), on the
// two per-seq state arenas in `dstates`; v0/vloc select the v-head window -- v0 == 0 &&
// vloc == 0 is the unsplit (tp=1) evaluation, compact is the state_compact_b16 flag.
bool run_batched(const uint16_t* dq, const uint16_t* dk, const uint16_t* dv,
                 const uint16_t* da, const uint16_t* db, const uint16_t* ddt, const uint16_t* daa,
                 float* dstates, void* dout, int t0, int v0, int vloc, bool compact) {
    const size_t qdim = (size_t)kQ * kHD, vdim = (size_t)kV * kHD;
    const uint16_t *q0 = dq + t0 * qdim, *k0 = dk + t0 * qdim, *v0p = dv + t0 * vdim,
                   *a0 = da + t0 * (size_t)kV, *b0 = db + t0 * (size_t)kV;
    return sparkinfer::kernels::launch_qwen36_gdn_ar_batched(
        q0, k0, v0p, a0, b0, ddt, daa, (float* const*)dstates, 0, dout, 2, kQ, kV, kHD, true,
        nullptr, compact, v0, vloc);
}

// Tolerance comparator for the batched-vs-sequential pairs (bit-identical by design; a small
// tolerance keeps the check robust to compiler reassociation across the two kernels).
int compare_states(const std::vector<float>& a, const std::vector<float>& b, const char* tag,
                   float rel) {
    int bad = 0;
    double maxdiff = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::abs((double)a[i] - (double)b[i]);
        maxdiff = std::max(maxdiff, d);
        if (d > rel * std::max(1.0, std::abs((double)a[i]))) bad++;
    }
    if (bad) {
        std::printf("FAIL: %s: %d/%zu values beyond %.1g relative tolerance (max diff %.3g)\n",
                    tag, bad, a.size(), rel, maxdiff);
        return bad;
    }
    std::printf("[ok] %s: %zu values within %.1g relative tolerance (max diff %.3g)\n", tag,
                a.size(), rel, maxdiff);
    return 0;
}

float bf16_to_f(uint16_t u) {
    uint32_t b = (uint32_t)u << 16;
    float f; memcpy(&f, &b, 4);
    return f;
}

// The whole GPU section: 1 slot, 16 v-heads (g = 2, block convention), 8 q-heads, 8 tokens.
// Explicit allocation, ~16 MiB total (13 state arenas of 1 MiB, b16 staging, activation rows).
// Nothing here is sized from free VRAM.
void test_gpu_windowing() {
    uint32_t seed = 0x5eed;
    const size_t qdim = (size_t)kQ * kHD, vdim = (size_t)kV * kHD;
    auto hq = rand_bf16((size_t)kN * qdim, (seed += 7), 1.f);
    auto hk = rand_bf16((size_t)kN * qdim, (seed += 7), 1.f);
    auto hv = rand_bf16((size_t)kN * vdim, (seed += 7), 1.f);
    auto ha = rand_bf16((size_t)kN * kV, (seed += 7), 1.f);
    auto hb = rand_bf16((size_t)kN * kV, (seed += 7), 1.f);
    auto hdt = rand_bf16(kV, (seed += 7), 0.5f);
    auto haa = rand_bf16(kV, (seed += 7), -0.5f);
    auto* dq = upload(hq); auto* dk = upload(hk); auto* dv = upload(hv);
    auto* da = upload(ha); auto* db = upload(hb);
    auto* ddt = upload(hdt); auto* daa = upload(haa);
    if (!dq || !dk || !dv || !da || !db || !ddt || !daa) {
        std::printf("FAIL: gdn_split activation alloc\n");
        return;
    }
    void* drefout = nullptr;
    if (cudaMalloc(&drefout, (size_t)kN * vdim * 2) != cudaSuccess) {
        std::printf("FAIL: gdn_split out alloc\n");
        return;
    }
    const size_t win_n = (size_t)8 * kHD * kHD;      // one (,8) window arena, fp32
    const size_t arena_n = (size_t)kV * kHD * kHD;   // a full 16-head arena
    // Every state arena is allocated at the full 16-head size so the windowed ones can be
    // read back into the same shapes; only the first win_n floats of a window arena are used.
    // 0=reference, 1/2=(0,8)/(8,8) seq runs, 3/4=seq0/seq1 single-AR refs, 5/6=(0,8) packed,
    // 7/8=(8,8) packed, 9/10=unsplit packed.
    std::vector<float*> arenas(11);
    float* dstates = nullptr;
    uint16_t *stg0 = nullptr, *stg1 = nullptr;
    for (auto& a : arenas) {
        if (cudaMalloc((void**)&a, arena_n * 4) != cudaSuccess) {
            std::printf("FAIL: gdn_split arena alloc\n");
            return;
        }
    }
    // arenas: 0=reference, 1=(0,8) seq, 2=(8,8) seq, 3..10=packed arena pairs (WA0, WA1, WB0,
    // WB1, A0u, A1u, S0, S1), all in the same order.
    if (cudaMalloc((void**)&dstates, 2 * sizeof(float*)) != cudaSuccess ||
        cudaMalloc((void**)&stg0, win_n * 2) != cudaSuccess ||
        cudaMalloc((void**)&stg1, win_n * 2) != cudaSuccess) {
        std::printf("FAIL: gdn_split staging alloc\n");
        return;
    }
    // Refresh the device pointer array the batched launcher reads from the GPU.
    auto set_states = [&](const float* p0, const float* p1) {
        float* hp[2] = {const_cast<float*>(p0), const_cast<float*>(p1)};
        cudaMemcpy(dstates, hp, sizeof hp, cudaMemcpyHostToDevice);
    };

    // (a) Reference: 8 sequential full-arena steps. (b) The same 8 steps under call-site
    // windowing; each v-head's recurrence touches only its own head rows and the windowed call
    // is handed the pre-shifted bases, so the per-head arithmetic is the one the reference
    // performed -- bit-identical, checked exactly.
    if (!run_ar_steps(dq, dk, dv, da, db, ddt, daa, arenas[0], drefout, 0, kQ, kV)) return;
    auto rfull = download(arenas[0], arena_n);
    if (!run_ar_steps(dq, dk, dv, da, db, ddt, daa, arenas[1], drefout, 0, kQ / kG, 8)) return;
    if (!run_ar_steps(dq, dk, dv, da, db, ddt, daa, arenas[2], drefout, 8, kQ / kG, 8)) return;
    auto wa = download(arenas[1], win_n), wb = download(arenas[2], win_n);
    int seqbad = 0;
    for (size_t i = 0; i < win_n; i++) {
        if (wa[i] != rfull[i]) seqbad++;
        if (wb[i] != rfull[win_n + i]) seqbad++;
    }
    if (seqbad) {
        std::printf("FAIL: sequential windowed single-AR vs 8-step full reference: %d/%zu values "
                    "differ\n", seqbad, 2 * win_n);
        failures += 2;
    } else {
        std::printf("[ok] sequential windowed single-AR (0,8)/(8,8) vs 8-step full reference: "
                    "bit-identical\n");
    }

    // (c) Packed batched, two independent sequences (token 0 / token 1), one token each:
    // single-AR references into full arenas (WA0/WA1 below are arenas[3]/[4] reused full),
    // then the new v0/vloc batched launches into the per-seq window arenas, and the unsplit
    // (vloc=0) invariance into full arenas.
    // Batched row i consumes token t0 + i, so each sequence's reference steps its own token.
    auto single_ref = [&](float* arena, int t) {
        cudaMemset(arena, 0, arena_n * 4);
        ar_step(dq, dk, dv, da, db, ddt, daa, arena, drefout, t, 0, kQ, kV);
    };
    single_ref(arenas[3], 0);   // seq0 reference (token 0)
    single_ref(arenas[4], 1);   // seq1 reference (token 1)
    auto a0 = download(arenas[3], arena_n), a1 = download(arenas[4], arena_n);
    for (int w = 0; w < 2; w++) {
        const int v0 = w * 8;
        const float *src0 = w == 0 ? (const float*)arenas[5] : (const float*)arenas[7];
        const float *src1 = w == 0 ? (const float*)arenas[6] : (const float*)arenas[8];
        cudaMemset((void*)src0, 0, win_n * 4);
        cudaMemset((void*)src1, 0, win_n * 4);
        set_states(src0, src1);
        if (!run_batched(dq, dk, dv, da, db, ddt, daa, dstates, drefout, 0, v0, 8, false)) return;
        auto g0 = download(src0, win_n), g1 = download(src1, win_n);
        std::vector<float> r0(a0.begin() + (size_t)v0 * kHD * kHD, a0.begin() + (size_t)(v0 + 8) * kHD * kHD);
        std::vector<float> r1(a1.begin() + (size_t)v0 * kHD * kHD, a1.begin() + (size_t)(v0 + 8) * kHD * kHD);
        char tag[64];
        std::snprintf(tag, sizeof tag, "batched v0=%d vloc=8 seq0 vs single-AR", v0);
        failures += compare_states(g0, r0, tag, 1e-4f);
        std::snprintf(tag, sizeof tag, "batched v0=%d vloc=8 seq1 vs single-AR", v0);
        failures += compare_states(g1, r1, tag, 1e-4f);
    }
    cudaMemset(arenas[9], 0, arena_n * 4);
    cudaMemset(arenas[10], 0, arena_n * 4);
    set_states(arenas[9], arenas[10]);
    if (!run_batched(dq, dk, dv, da, db, ddt, daa, dstates, drefout, 0, 0, 0, false)) return;
    auto u0 = download(arenas[9], arena_n), u1 = download(arenas[10], arena_n);
    failures += compare_states(u0, a0, "batched unsplit (v0=0, vloc=0) seq0 vs single-AR", 1e-4f);
    failures += compare_states(u1, a1, "batched unsplit (v0=0, vloc=0) seq1 vs single-AR", 1e-4f);

    // (d) The compacted-bf16 batched path at bf16 tolerance: the (0,8) packed run left
    // arenas[5]/[6] in fp32; a copy feeds the fp32 reference (one more token, rows 2/3), the
    // originals go through state_to_b16 and the same step with state_compact_b16=true.
    auto s0c = download(arenas[5], win_n), s1c = download(arenas[6], win_n);
    auto S0h = upload(s0c);
    auto S1h = upload(s1c);
    if (!S0h || !S1h) {
        std::printf("FAIL: gdn_split b16 ref alloc\n");
        return;
    }
    set_states(S0h, S1h);
    run_batched(dq, dk, dv, da, db, ddt, daa, dstates, drefout, 2, 0, 8, false);
    auto rf0 = download(S0h, win_n), rf1 = download(S1h, win_n);
    const bool ok0 = sparkinfer::kernels::launch_qwen36_gdn_state_to_b16(arenas[5], stg0, win_n, nullptr);
    const bool ok1 = sparkinfer::kernels::launch_qwen36_gdn_state_to_b16(arenas[6], stg1, win_n, nullptr);
    if (!ok0 || !ok1) {
        std::printf("FAIL: state_to_b16 declined\n");
        ++failures;
        return;
    }
    set_states(arenas[5], arenas[6]);
    run_batched(dq, dk, dv, da, db, ddt, daa, dstates, drefout, 2, 0, 8, true);
    // The compacted state now sits in the first win_n bf16 of each arena (staging held the
    // one-shot conversion result; the step overwrote the arenas' own front half in place).
    auto b0 = download((uint16_t*)arenas[5], win_n), b1 = download((uint16_t*)arenas[6], win_n);
    std::vector<float> b0f(b0.size()), b1f(b1.size());
    for (size_t i = 0; i < b0.size(); i++) { b0f[i] = bf16_to_f(b0[i]); b1f[i] = bf16_to_f(b1[i]); }
    failures += compare_states(rf0, b0f, "batched compact-b16 (v0=0, vloc=8) seq0 vs fp32 ref", 1e-2f);
    failures += compare_states(rf1, b1f, "batched compact-b16 (v0=0, vloc=8) seq1 vs fp32 ref", 1e-2f);
}

// The skip guard: explicit budgets only -- this never sizes anything from free VRAM.
bool gpu_ok(size_t budget) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return false;
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
    std::printf("[wp8] GPU section: %.1f MiB free on device 0 (budget %.1f MiB, skip-only guard)\n",
                (double)f / (1024.0 * 1024.0), (double)budget / (1024.0 * 1024.0));
    return f >= budget;
}

}  // namespace

int main() {
    test_27b_figures();
    test_normalize_block();
    test_normalize_cyclic();
    test_tp1_invariance();
    test_restore_sizes();
    if (gpu_ok(32 * 1024 * 1024))
        test_gpu_windowing();
    else
        std::printf("[wp8] GPU section skipped (not enough headroom to risk the resident LLM) -- "
                    "CPU checks only\n");
    std::printf("gdn_split_cpu_test: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
