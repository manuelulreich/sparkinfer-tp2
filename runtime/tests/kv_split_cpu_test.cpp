// WP-7: the KV-cache split (2+2 KV heads, per-device pools) for the dual-GPU pair.
//
// The pure-C++ half (no CUDA calls) pins the arithmetic the split is built on:
//   - the head-window tiling that the pools mirror from the tp table's K/V convention
//     (tp_layout.hpp head_window: contiguous window per rank, last rank the remainder):
//     27B 4 KV heads at tp=2 -> 2+2, 35B-A3B 16 kv heads -> 8+8, odd remainder 5 -> 2+3,
//     tp=1 -> one pool with ALL heads (the invariance);
//   - the G2 per-token KV figure for the canonical 27B (16 full-attn layers, 4 kv heads,
//     head_dim 256): 33,024 B/token int8-data + fp16-scale, re-derived here and asserted
//     against the probe's number;
//   - the 262k/131k context fit against G2's max KV remainder per card (6.74 GiB): the
//     262k-class pool (8.06 GiB) does not fit, 131,072 does (4.03 GiB total -- G2's 4.04 is
//     the same 4.03125 -- to 2.02 GiB/card at 2+2); the per-card figures are printed, as the
//     brief requires;
//   - the min-across-pools admission rule, in arithmetic form.
//
// One explicit small-budget GPU paging section follows (a few blocks, 2+2 heads, ~2 MiB
// of device memory at most): the real KVCacheManager on a 2+2 window -- paging, the
// logical->physical pair shared across the K and V pools, per-pool refcounts, prefix
// restore across both pools, and the min-across-pools admission with a second, smaller
// pool. It is SKIP-guarded, never auto-sized: an explicit headroom check against those
// fixed budgets decides whether it runs (lean mode: a resident LLM holds ~1 GB free per
// card here, and this touches only its own few-MB allocation). No Python, no model load.

#include "../include/sparkinfer/kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

using sparkinfer::KVCacheConfig;
using sparkinfer::KVCacheManager;

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

namespace {

// The tp table's K/V head convention (tp_layout.hpp head_window), verbatim arithmetic:
// rank r's window on n heads across tp ranks, last rank taking the remainder.
void head_window(int n, int r, int tp, int* h0, int* hl) {
    *h0 = n * r / tp;
    *hl = n * (r + 1) / tp - *h0;
}

// (a) The 2+2 (and 8+8, and the odd-remainder) tiling, and the tp=1 all-heads invariance.
void test_head_window() {
    int h0, hl;
    // 27B: 4 KV heads, tp=2 -> rank 0 [0,2), rank 1 [2,4) -- the 2+2 split.
    head_window(4, 0, 2, &h0, &hl);
    CHECK(h0 == 0 && hl == 2);
    head_window(4, 1, 2, &h0, &hl);
    CHECK(h0 == 2 && hl == 2);
    // 35B-A3B: 16 kv heads, tp=2 -> 8+8.
    head_window(16, 0, 2, &h0, &hl);
    CHECK(h0 == 0 && hl == 8);
    head_window(16, 1, 2, &h0, &hl);
    CHECK(h0 == 8 && hl == 8);
    // Odd head count: the windows still tile n exactly (2+3 for 5).
    head_window(5, 0, 2, &h0, &hl);
    CHECK(h0 == 0 && hl == 2);
    head_window(5, 1, 2, &h0, &hl);
    CHECK(h0 == 2 && hl == 3);
    // tp=1: one pool with ALL heads -- the degenerate (today's) layout.
    head_window(4, 0, 1, &h0, &hl);
    CHECK(h0 == 0 && hl == 4);
    head_window(16, 0, 1, &h0, &hl);
    CHECK(h0 == 0 && hl == 16);
    // Tiling in general: the windows partition [0, n) exactly.
    for (int n : {4, 16, 5, 1, 3})
        for (int r = 0; r < 2; ++r) {
            int a0, al, b0, bl;
            head_window(n, r, 2, &a0, &al);
            if (r + 1 < 2) head_window(n, r + 1, 2, &b0, &bl);
            CHECK(a0 >= 0 && a0 + al <= n);
            if (r + 1 < 2) {
                CHECK(b0 == a0 + al);           // no gap, no overlap
                CHECK(al + bl == n);           // exact tiling
            }
        }
}

// The canonical 27B per-token KV figure from G2, re-derived: 16 full-attn layers,
// 4 KV heads of head_dim 256, int8 data + one fp16 scale per (token, kv head) per pool.
size_t per_token_kv_bytes_27b() {
    const int n_slots = 16, n_kv = 4, hd = 256;
    const size_t data = (size_t)n_slots * 2 * n_kv * hd;        // int8: K and V, all layers
    const size_t scales = (size_t)n_slots * 2 * n_kv * 2;        // fp16 per (token, kv_head)
    return data + scales;
}

// (d) The G2 262k/131k pool arithmetic against the max KV remainder per card, with the
// per-card figures of the 2+2 split printed.
void test_g2_ctx_fit() {
    CHECK(per_token_kv_bytes_27b() == 33024);   // the G2 number, on the record
    const double gib = 1073741824.0;             // 2^30
    const double max_remainder_gib = 6.74;       // G2: max KV remainder per card on this pair
    const size_t per_token = per_token_kv_bytes_27b();
    for (int ctx : {262144, 131072}) {
        const double total_gib = (double)(ctx * per_token) / gib;
        const double per_card_gib = total_gib / 2.0;             // 2+2: each card its own pool
        std::printf("[wp7] %d ctx: %.2f GiB KV total (%.2f GiB/card at 2+2) vs %.2f GiB max "
                    "remainder/card -> %s\n",
                    ctx, total_gib, per_card_gib, max_remainder_gib,
                    (ctx == 262144) ? "DOES NOT FIT (G2: the 8.06 GiB pool is a single-card one)"
                                    : "FITS (the honest --ctx default for this pair)");
        if (ctx == 262144) {
            // G2's verdict: the 8.06 GiB pool must fit in 6.74 GiB of ONE card -- it does not.
            CHECK(total_gib > max_remainder_gib);
        } else {
            // 131,072: 4.04 GiB total; at 2+2 each card carries 2.02 GiB, well inside.
            CHECK(total_gib < 2 * max_remainder_gib);
            CHECK(per_card_gib < max_remainder_gib);
        }
    }
    // The per-card per-token figure the split implies: exactly half of 33,024.
    const int n_slots = 16, n_kv = 4, hd = 256;
    const size_t per_card_per_token =
        (size_t)n_slots * 2 * (n_kv / 2) * hd + (size_t)n_slots * 2 * (n_kv / 2) * 2;
    CHECK(per_card_per_token == 16512);
    CHECK(2 * per_card_per_token == per_token_kv_bytes_27b());
}

// (6)/(tp=1 invariance) the budget formula: tp=1 is the old whole-heads number byte for
// byte; two 2-head pools at tp=2 sum back to the one 4-head pool.
void test_pool_budget_invariance() {
    // The server's bf16-denominated budget: kvL * 2(K+V) * (16 * heads * head_dim) * 2B * blocks.
    auto budget = [](int kvL, int heads, int hd, size_t blocks) {
        return (size_t)kvL * 2 * (16 * (size_t)heads * (size_t)hd) * 2 * blocks;
    };
    const int kvL = 16, hd = 256;
    const size_t blocks = 262144 / 16 + 8;   // the 262k-class block count
    const size_t tp1 = budget(kvL, 4, hd, blocks);
    const size_t rank0 = budget(kvL, 4 / 2, hd, blocks);
    const size_t rank1 = budget(kvL, 4 / 2, hd, blocks);
    CHECK(tp1 == 4 * 16 * 256 * 2 * 2 * kvL * blocks);   // the old formula, verbatim: epb * 2(K+V) * 2B * kvL * blocks
    CHECK(rank0 + rank1 == tp1);                      // the 2+2 split sums to the whole
    // 35B-A3B (16 kv heads, tp=2): 8+8 likewise.
    CHECK(budget(16, 8, 128, blocks) * 2 == budget(16, 16, 128, blocks));
}

// (4) Admission capacity = the min across pools: with pools holding 8 and 4 free blocks,
// a 5-block request is refused (it fits the big pool alone) and a 3-block request passes.
void test_admission_min() {
    const int free_big = 8, free_small = 4;
    for (int need : {5, 4, 3, 2}) {
        const bool admit = std::min(free_big, free_small) >= need;
        const bool big_only = free_big >= need;
        std::printf("[wp7] admission: need=%d pools(%d,%d) -> %s\n", need, free_big, free_small,
                    admit ? "admit" : "refuse");
        if (need == 5) {
            CHECK(big_only && !admit);       // fits the big pool alone -- the min still refuses
        } else if (need == 3) {
            CHECK(admit);
        } else {
            CHECK(admit == big_only);
        }
    }
}

// The skip guard: explicit budgets only -- this never sizes anything from free VRAM.
bool gpu_ok(size_t need_plus_headroom) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return false;
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return false;
    std::printf("[wp7] GPU paging section: %.1f MiB free on device 0 (budget %.1f MiB, "
                "skip-only guard)\n",
                (double)f / (1024.0 * 1024.0), (double)need_plus_headroom / (1024.0 * 1024.0));
    return f >= need_plus_headroom;
}

KVCacheConfig mkcfg(int start, int count) {
    KVCacheConfig c;
    c.num_layers = 2;        // identity slot map -> 2 pool slots
    c.num_kv_heads = 4;      // the 27B kv-head count
    c.head_dim = 256;
    c.block_size = 16;
    c.kv_head_start = start;
    c.kv_head_count = count;   // 0 = all heads
    return c;
}

// (a)+(b)+(c)+(4) the real-pool half: one explicit few-MB run. A = rank-0's 2+2 window
// [0,2), B = rank-1's [2,4), C = a smaller pool of the same window (the min side of
// admission), D = an int8 pool (per-pool scale pool), E = the tp=1 all-heads degenerate
// pool. All on explicit budgets; every manager frees its own memory at scope end.
void test_gpu_paging() {
    const size_t big = 512 * 1024;    // 8 blocks of a 2+2 bf16 pool
    const size_t small = 256 * 1024;  // 4 blocks
    const size_t tiny = 64 * 1024;    // 2 blocks of an int8 1-slot pool
    KVCacheManager A(mkcfg(0, 2), big);
    KVCacheManager B(mkcfg(2, 2), big);
    KVCacheManager C(mkcfg(0, 2), small);
    KVCacheConfig dc = mkcfg(0, 2); dc.int8_kv = true; dc.num_layers = 1;
    KVCacheManager D(dc, tiny);
    KVCacheManager E(mkcfg(0, 0), big);   // count 0 = all heads = today's single pool

    // The pools: 2 slots x 2 pools x (16 tokens x 2 heads x 256) x 2B = 65,536 B/block.
    CHECK(A.num_total_blocks() == 8);
    CHECK(A.num_free_blocks() == 8);
    CHECK(B.num_total_blocks() == 8);
    CHECK(A.kv_head_start() == 0 && A.kv_head_count() == 2);
    CHECK(B.kv_head_start() == 2 && B.kv_head_count() == 2);
    CHECK(E.num_total_blocks() == 4);              // the tp=1 degenerate: 4 heads, 4 blocks
    CHECK(E.kv_head_count() == 0);                // 0 = all heads
    // A layer stride: 8 blocks x 16 x 2 heads x 256 = 65,536 elems -- the 2+2 pool's own size.
    CHECK(A.layer_stride_elems() == 8 * 16 * 2 * 256);
    CHECK(E.layer_stride_elems() == 4 * 16 * 4 * 256);   // = A's 65,536: half the blocks, double the block size

    // (a) Paging in the small explicit budget: grow, truncate, free -- exact block counts.
    CHECK(A.allocate(1, 96));                       // 6 blocks
    CHECK(A.num_blocks(1) == 6);
    CHECK(A.allocated_tokens(1) == 96);
    CHECK(A.num_free_blocks() == 2);
    CHECK(A.truncate_blocks(1, 3));                // 3 left
    CHECK(A.num_free_blocks() == 5);
    CHECK(A.allocate(900, 48));                   // a second sequence takes 3 blocks: free 2
    CHECK(A.num_free_blocks() == 2);
    CHECK(!A.allocate(1, 96));                    // would need 3 more, only 2 free
    CHECK(A.num_free_blocks() == 2);
    A.free(900);
    CHECK(A.num_free_blocks() == 5);

    // (b) Prefix restore across both pools: a retained 3-block prefix, then a second
    // sequence that starts from it -- the logical prefix resolves to the SAME physical
    // block in the K pool and in the V pool (the shared logical numbering), and each
    // pool's refcount says so (2 = the writer's ghost is gone, retained + reuser).
    const std::vector<int> ret = A.retain_prefix_blocks(1, 3);
    CHECK(ret.size() == 3);
    for (int b : ret)
        CHECK(A.block_ref_k(b) == 2 && A.block_ref_v(b) == 2);
    A.free(1);                                     // the writer's refs drop: retained stays
    CHECK(A.num_free_blocks() == 5);
    CHECK(A.allocate_with_prefix(2, ret, 64));     // 4 blocks: the 3 shared + 1 new
    CHECK(A.num_free_blocks() == 4);
    const std::vector<int>& seq2 = A.physical_block_ids(2);
    CHECK(seq2.size() == 4);
    for (size_t i = 0; i < 3; ++i)
        CHECK(seq2[i] == ret[i]);                  // logical i -> the retained physical, in both pools
    for (int b : ret)
        CHECK(A.block_ref_k(b) == 2 && A.block_ref_v(b) == 2);   // retained + reuser
    CHECK(A.block_ref_k(seq2[3]) == 1 && A.block_ref_v(seq2[3]) == 1);   // the fresh one
    A.release_blocks(ret);                         // the prefix holder lets go
    for (int b : ret)
        CHECK(A.block_ref_k(b) == 1 && A.block_ref_v(b) == 1);
    A.free(2);
    CHECK(A.num_free_blocks() == 8);
    for (int b : ret)
        CHECK(A.block_ref_k(b) == 0 && A.block_ref_v(b) == 0);

    // The rank-1 pool runs the same 2-block flow on its own window: the logical numbering
    // is shared (the same indices A and B walk), each pool resolving its own physical blocks.
    CHECK(B.allocate(11, 32));
    CHECK(B.num_free_blocks() == 6);
    const std::vector<int> retB = B.retain_prefix_blocks(11, 2);
    CHECK(retB.size() == 2);
    B.free(11);
    CHECK(B.allocate_with_prefix(12, retB, 48));   // 3 blocks: 2 shared + 1 new
    const std::vector<int>& seq12 = B.physical_block_ids(12);
    CHECK(seq12.size() == 3);
    for (size_t i = 0; i < 2; ++i)
        CHECK(seq12[i] == retB[i]);
    for (int b : retB)
        CHECK(B.block_ref_k(b) == 2 && B.block_ref_v(b) == 2);
    B.release_blocks(retB);
    B.free(12);
    CHECK(B.num_free_blocks() == 8);

    // (c) + (4) Admission = the min across pools: C holds only 4 blocks. A 5-block
    // request (80 tokens) fits A alone but not the pair -- the min refuses.
    CHECK(C.num_total_blocks() == 4);
    CHECK(!C.allocate(90, 80));                     // 5 needed > 4 in C
    CHECK(C.num_free_blocks() == 4);
    CHECK(A.allocate(91, 80));                     // A alone could take it: 5 <= 8
    CHECK(A.num_free_blocks() == 3);
    // Now a 3-block request (48 tokens): min(3, 4) >= 3 -- both pools admit it.
    CHECK(A.allocate(92, 48));
    CHECK(C.allocate(93, 48));
    CHECK(A.num_free_blocks() == 0);
    CHECK(C.num_free_blocks() == 1);
    A.free(91); A.free(92);
    C.free(90); C.free(93);
    CHECK(A.num_free_blocks() == 8);
    CHECK(C.num_free_blocks() == 4);

    // (int8) The per-pool scale pool: D is one slot, a 2+2 int8 pool. The scale stride is
    // layer_stride / head_dim: one fp16 scale per (token, kv head) -> 2 blocks x 16 x 2 = 64.
    CHECK(D.num_total_blocks() == 2);
    CHECK(D.scale_layer_stride_elems() == 2 * 16 * 2);
    CHECK(D.kv_head_count() == 2);
}

}  // namespace

int main() {
    test_head_window();
    test_g2_ctx_fit();
    test_pool_budget_invariance();
    test_admission_min();
    if (gpu_ok(2 * 512 * 1024 + 256 * 1024 + 64 * 1024 + 512 * 1024 + (size_t)2 * 1024 * 1024))
        test_gpu_paging();
    else
        std::printf("[wp7] GPU paging section skipped (not enough headroom to risk the "
                    "resident LLM) -- CPU checks only\n");
    std::printf("kv_split_cpu_test: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
