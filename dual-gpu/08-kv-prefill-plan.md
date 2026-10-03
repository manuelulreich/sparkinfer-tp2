# 8. KV formats and prefill: fp8/nvfp4 attention, nvfp4 decode, the link

Written 2026-10-03. Plan only; nothing here is implemented yet.

## Where we stand (2× RTX 5060 Ti, tp=2, `--ctx 131072`, no drafter)

Measured with a 128-token greedy decode after each prompt (tok/s):

| KV | decode @311 | decode @60k | decode @118k | prefill @60k | prefill @118k |
|---|---:|---:|---:|---:|---:|
| int8 (default) | 54.7 | 46.4 | 41.0 | **2935** | **2471** |
| fp8 | 54.9 | **48.3** | **44.0** | 2168 | 1569 |
| nvfp4 | 55.3 | 44.7 | 38.4 | 2169 | 1574 |

Largest `--ctx` that loads with the drafter: int8 229,376 (744 MiB left on card 0; 262,144 runs
out of memory loading the drafter), nvfp4 262,144 (2.1 GB left). Per card and token: int8 16,512 B,
nvfp4 9,344 B.

**Prefill profile, a 31k prompt** (nsys, rank 1's card; the window spans the whole pass):

| | int8 KV | nvfp4 KV |
|---|---:|---:|
| wall time of the pass | 9.99 s | 11.74 s |
| GPU busy | 5.07 s (51 %) | 6.83 s (58 %) |
| FP4 weight GEMMs (cutlass, block-scaled) | 2.08 s | 2.08 s |
| attention | 0.87 s (`pf_attn_mma_gqa_kernel`, int8 MMA) | 2.70 s (`pf_attn_mma_bf16_kernel`) |
| history plane dequant (`pf_kv_dequant_kernel`) | – | 7 ms |
| GDN (scan, prep, conv, norm) | 0.85 s | 0.85 s |
| idle, ending in `pf_add` (waiting for the peer's partial) | 4.74 s | similar |

- The weight GEMMs already run W4A4 on the FP4 tensor cores (plan 05, Part A): 7.55e14 FLOP per
  card in 2.08 s is **~363 TFLOPS**, close to the card's dense FP4 peak (36 SMs).
- **fp8/nvfp4 prefill is slow because the attention runs on bf16 tensor cores**, at about a third
  of the int8 kernel's effective rate (34 vs 106 TOPS on the same FLOPs). Building the bf16
  history plane itself is cheap (7 ms).
- **Half of the pass is the cross-card link.**
  - 129 all-reduces a pass, each N × 5120 bf16 per direction: 1.31 MB a token per direction.
  - The host is PCIe Gen3 (the GPUs are Gen5-capable; `pcie.link.gen.max` = 3, x8/x8); P2P
    measures 6.5–7.2 GB/s (00-p0-probe). That is ~190 µs a token, **~5.9 s of transfers for 31k**,
    against 5.1 s of compute.
  - Plan 05's chunked overlap is on; it hides ~1.2 s of the 5.9.
  - **Prefill ceiling from the link alone: ~5.3k tok/s, whatever the math does.**
- Decode is at the memory wall: ~8.1 GB of NVFP4 weights a card per token at 448 GB/s is ~18 ms;
  measured 18.3 ms at short context.

## Why nvfp4 KV does not give ~2×

- **Prefill.** The KV format only touches attention: 0.87 s of a 9.99 s pass at 31k (int8).
  Prefill attention is compute-bound (N² FLOPs), so smaller KV bytes buy nothing there; only a
  faster MMA does. The weights are already NVFP4 and already on the FP4 tensor cores. The rest is
  the link (4.7 s exposed) and GDN.
- **Decode.** The weights (~8.1 GB a card) are already 4-bit; the KV is 2.0 GB a card at 118k in
  int8, 1.1 GB in nvfp4. Even a kernel at full bandwidth saves 0.9 GB of ~10 GB a token: about
  +9 % over fp8 at 118k, nothing at short context. Today the nvfp4 kernel is compute-bound (e2m1 →
  e4m3 → f16 widening and the per-16 scale on every element), so it is slower than fp8.
- **FP4 attention MMA** (block-scaled `mxf4nvf4`, 2× the 8-bit MMA rate) needs *both* operands
  in FP4: Q and P quantized to 4 bits, and V re-laid with its scales along the token axis (the
  P·V reduction axis), not along the head dim as stored. That is SageAttention3-style attention
  with a measurable accuracy cost, and the gain is limited to the attention share (~9 % of the
  pass at 31k, ~25 % at 118k).

What nvfp4 KV does buy: the full 262k context with the drafter, ~0.9 GB a card at 131k, and
fewer KV bytes for long-context decode and verify once its kernel is bandwidth-bound.

## Part A: fp8/nvfp4 prefill attention on 8-bit tensor cores

Goal: fp8/nvfp4 prefill equal to int8 (2169 → ~2900 tok/s at 60k, 1570 → ~2450 at 118k).

- **A0. Microbench the MMA rates on this card** (minutes): `mma.sync m16n8k32` e4m3·e4m3 with f32
  and with f16 accumulate, against the int8 `m16n8k32` s32 the int8 kernel uses. Decides whether
  A1 is an fp8 kernel or an int8 requant.
- **A1. fp8 twin of the int8 prefill kernel** (`pf_attn_mma_gqa_kernel`, the 6:1 GQA tier).
  - fp8 KV has the int8 cache's layout (1 B an element + an fp16 scale per token and KV head), so
    the kernel reads K/V pages directly: no history plane at all (frees the arena's bf16 plane:
    ~240 MB a layer-pass at 118k).
  - Q quantized to e4m3 per row the way the int8 path quantizes it to int8; P' in e4m3 (in
    [0, 1] like the int8 kernel's P'/pd).
  - Fallback if A0 shows fp8 MMA with f32 accumulate below the int8 rate: requantize the history
    to int8 in the dequant kernel and run the existing int8 kernel (a second rounding; measure KL).
- **A2. nvfp4 into the same kernel.** `pf_kv_dequant_kernel<3>` writes an e4m3 (or int8) plane
  with the per-token-head scale instead of a bf16 plane: e2m1 × e4m3-scale rounded to 8 bits, one
  extra rounding of at most 2⁻⁴ relative. Half the plane bytes of today.
- **A3. Checks.** KL against bf16 KV at 1k/4k/16k/64k (int8 today: 0.012–0.018; nvfp4
  0.018–0.025), the 118k retrieval prompt, `tp2_gates.py`, the lossless DSpark check (ctx 32k,
  deterministic).

## Part B: nvfp4 decode at memory bandwidth

Goal: nvfp4 decode faster than fp8 at long context (118k: 38.4 → ≥ 47 tok/s), and the verify's
paired kernel too.

- **B0. Kernel-level baseline:** `fa_split_gqa_mma_f8_kernel` / `_pair_kernel` time a call at
  32k/118k for fp8 vs nvfp4, and achieved DRAM bandwidth (ncu, one call).
- **B1. QK: apply the scale on the accumulator, not the elements.** One `m16n8k16` k-step is
  exactly one 16-dim scale group, so: MMA on the raw e2m1 values (widened to f16), then one FMA
  `acc += s[token][group] · partial` per accumulator. That replaces 16 multiplies per element group
  with 2 per thread.
- **B2. PV: fold V's scale into P.** V's scales vary along the head dim (the n axis). For each
  16-dim group, scale the P fragment by s[token][group] (16× fewer multiplies than scaling V).
- **B3. Widening:** the hardware `cvt.rn.f16x2.e2m1x2` (sm_120a) instead of the byte-permute
  table plus e4m3 → f16, if A0-style microbench shows it is faster.
- **B4. Checks:** bit-for-bit is not expected (different rounding order); KL within the current
  nvfp4 range, and the in-server decode-vs-verify consistency check.

## Part C: the link (the actual ~2× lever for prefill on this machine)

- **C0. Re-profile the exposed link per all-reduce type** at 4k/31k: which of the 129 (embedding,
  GDN out, attention o, FFN down) are not overlapped, and how much compute each has to hide behind.
- **C1. Overlap closer to the ceiling** (lossless): post the GDN out / attention o all-reduce in
  chunks earlier (inside the projection GEMM's row loop), and start the next layer's input
  projections per chunk (plan 05 B6). Ceiling: max(link 5.9 s, compute 5.1 s) ≈ 6 s at 31k,
  ~5k tok/s instead of 3.1k.
- **C2. Optional lossy link format** (opt-in only, as plan 05 decided). Today every prefill
  all-reduce sends raw bf16 partials (`tp_prefill_allreduce_bf16`); nothing is compressed.
  - Codec as in b12x's `PCIeDmaAllReduce` wire modes: blocks of 128 values, e4m3 or int8 codes
    plus one fp32 scale a block, 132 B instead of 256 B (−48 %). At tp=2 the direct exchange stays
    (b12x's ring/RS+AG phases send the same bytes on two ranks), i.e. its quantize-once `a2a` form.
  - **Both ranks must add the same values**: each rank sums q(own) + q(peer), not own + q(peer),
    or the replicated residual stream diverges between the cards.
  - Decode all-reduces (10 KB rows, ~24 µs, latency-bound) stay bf16: fewer bytes do not help them.
  - Estimate at 31k: link 5.9 → 3.05 s; with today's overlap the pass goes ~10 → ~7 s (+40 %),
    with C1 it becomes compute-bound (~6k tok/s).
  - Gates: KL against the bf16 link at 4k/16k/64k for e4m3 and int8 (residual-stream partials
    carry outlier channels, so the per-block codec choice matters), retrieval at 118k, and the
    lossless DSpark check (prefill is shared by both servers, so it stays lossless if deterministic).
    Never the default.
- Hardware note: on a PCIe Gen4/Gen5 host the same cards get 2–4× the link, and prefill would be
  compute-bound without C2.

## Notes from other setups

- **co-l's single-5090 vLLM gist** (Qwen3.8-27B NVFP4, nvfp4 KV, 451K-token pool): the nvfp4
  KV there has one static global scale a layer (checkpoint k_scale/v_scale) under the per-16 e4m3
  scales, and attention reads it through FlashInfer's FA2 reader with bf16 queries, i.e. the same
  dequantize-then-bf16-MMA as our prefill today. No new kernel technique; our second level (an
  fp16 scale per token and head) is finer. Its practical point is capacity: with nvfp4 KV,
  several sessions' KV stays resident (3 × 75K), so agent sub-sessions resume without
  re-prefill. For us that is nvfp4 KV at `--ctx 262144` (the pool is shared across sessions);
  each session's GDN state (~148 MB) still counts separately.
- Its prefill (single 5090: 11.4k tok/s at 4K, 2.9k at 128K) is below our own tp=1 5090 numbers
  (14.4k at 4K); nothing to take for prefill.

## Not in this plan

- **Piecewise CUDA graphs** (vLLM's suggestion for its own engine): WP-11 measured launch gaps
  under 1 % of decode; the step waits on the all-reduce, not the host.
- **NVFP4 weights:** already native (prefill W4A4 on FP4 tensor cores, decode reads the packed
  bytes; the drafter since 5dca587).

## Order

| # | item | expected effect | effort |
|---|---|---|---|
| 1 | A0 microbench | decides A1's form | small |
| 2 | A1 fp8 prefill attention | fp8 prefill = int8 (+35 % @60k, +55 % @118k) | medium |
| 3 | A2 nvfp4 → 8-bit plane | nvfp4 prefill = int8 | small after A1 |
| 4 | C0 + C1 link overlap | +30–60 % prefill for every KV type | medium–large |
| 5 | B0–B3 nvfp4 decode | +15–20 % nvfp4 decode @118k | medium |
| 6 | C2 fp8 link (opt-in) | prefill compute-bound | small after C1 |
