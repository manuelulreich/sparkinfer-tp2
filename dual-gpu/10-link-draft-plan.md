# 10. nvfp4 + DFlash2: the decode step, the link, the drafter

Written 2026-10-03. Status blocks below record what is done.

Configuration: `startup.sh` (nvfp4 KV, DFlash2, `--ctx 131072`, int8 prefill wire), one opencode
session. Every lossless item keeps the speculative output equal to ordinary decode (checked as in
plan 09: `SPARKINFER_KV_DTYPE=int8 EXACT_LINK=1 SPARKINFER_DETERMINISTIC=1 CTX=32768`).

## Where the time goes (measured 2026-10-03, `SPARKINFER_DSPARK_TIMING`)

| | draft ms/step | verify ms/step (8 rows) | tokens/step | tok/s |
|---|---:|---:|---:|---:|
| opencode final answer, ~20k context, sampled | 4.64 | 26.66 | 3.26 | ~100 |
| 60k replay, sampled | 4.3–4.6 | 30.3–30.4 | 3.27–3.41 | 91–95 |
| HyperQwen, 60k, sampled (216 tokens only) | | | 3.74 | 113.9 |

Decode is most of an agent turn: the 42.5 s opencode turn spent ~26 s decoding its 2,568-token
answer, and each tool step's prefill was at most 1.7 s (the prefix cache resends only the delta).
So decode comes first here, prefill second.

From the profile in plan 07 (DSpark, int8 KV), extrapolated to 8 rows; to be re-measured (P0):

- weight GEMMs ~16 ms a step: the bandwidth floor (≈7.5 GB a card at 448 GB/s), no lever except
  more tokens per step;
- all-reduce ~3 ms: ~129 `glink_flag_allreduce_kernel` calls of ~22 µs, 80 KB each at 8 rows;
- attention: +3.7 ms from 20k to 60k with nvfp4 KV, so ~+9 ms at 120k;
- draft 4.6 ms, of which only ~2 ms is weight reads (5 layers at ~0.45 GB a card, plus the
  0.36 GB candidate head over each card's half vocabulary).

**About the 8-bit wire:** it is on the prefill all-reduces only (`tp_prefill_allreduce_bf16_async`).
Decode all-reduces are 10 KB a row; 80 KB at 8 rows is ~12 µs of link at ~6.5 GB/s, and the rest
of the ~22 µs is latency and waiting for the other card. Halving the bytes saves at most ~6 µs a
call, ~0.8 ms a step (D3). The larger decode lever is to stop waiting on the link at all (D1, D2).
In prefill the wire changed the bound: at 31k the link needs ~3.4 s instead of ~6.5 s against
~5 s of compute, so the remaining cost is link time that is still exposed, not link bandwidth (F1).

## P0. Profile first (needs both GPUs for ~5 minutes)

One nsys run of the 60k replay (`cap_si/req_007`, 256 tokens, sampled) and one at ~20k, with
`SPARKINFER_DSPARK_TIMING=1`:

- verify: share of GEMMs, all-reduce, attention, GDN, lm_head, norms and other small kernels, and
  host gaps (GPU idle) at 8 rows;
- all-reduce: link time vs waiting. Record `clock64()` when the push is fenced and when the peer
  flag arrives (a debug counter, removed afterwards). Waiting means the cards are skewed; then
  balancing work (rank 0 runs the selector, the merge and the sampling) is the fix, not the link;
- draft: kernels vs host gaps around its 2 host syncs (`d2_head` candidate merge, proposals).

Every estimate below is rescaled after P0, and items that turn out to be small are dropped.

**P0 status: done (2026-10-03).** nsys, 20k context, sampled, 8 rows; per card per step:

| | ms | |
|---|---:|---|
| weight GEMMs (CUTLASS NVFP4) | 15.6 | at the bandwidth floor |
| all-reduce (141 calls, 19.6 us avg) | 2.8 | of which ~1/3 in the draft |
| nvfp4 split attention | 2.3 | |
| verify head `si_mmvq_q4k_rows_exact` | 2.1 | the NVFP4 head had been released |
| draft head `si_mmvq_q4k_multirow` | 1.6 | same |
| draft linears (dp4a GEMVs) | 1.6 | ~65 % of bandwidth |
| GDN steps + conv | 1.5 | |
| `k_rows_topk` (2 calls) | 0.34 | one block a row |
| draft attention (5 calls) | 0.31 | |
| host gaps | ~2.5 | streaming (`decode_delta`), syncs |

Found and fixed first (not in the plan above): the NVFP4 head release, the top-k kernel, and
quadratic streaming. Step at 20k 31.3 -> 28.7 ms, 100 -> 114-117 tok/s. The grammar masks cost
0.17 ms a step; the host merge of the draft candidates 8 us (R1 is worth little).

**D3 status: tried, net loss, reverted (2026-10-04).** The 8-bit wire inside the flag kernel
(`SPARKINFER_TP_AR_WIRE_DECODE`): verify 25.1 -> 24.5 ms, but acceptance fell 3.2-3.3 -> 2.9-3.1
tokens a step (the lossy sums move the target away from what the drafter predicts), 112-116 ->
102-109 tok/s. The link moves only ~9 us of a ~20 us decode all-reduce; the two cards start it
within ~1 us of each other (no skew), so the rest is fence + flag latency and the reduce.

**L1 status: done differently (2026-10-04).** The verify's nvfp4 attention ran the single-row
kernel per row (8 KV reads per call). `fa_split_gqa_mma_nvfp4_rows_kernel` serves a session's rows
in one CTA (bit-identical, checked in place); 60k verify 28.8 -> 27.5 ms, 20k -0.3 ms. Still
latency-bound (125 us a call at 20k against ~26 us of KV bytes): a split's groups run one after
another, and the split partition is fixed by plain decode's (exactness), so 16 warps a CTA or
4-row groups measured no better.

**L2 status: fixed (2026-10-04).** The nvfp4/fp8 losslessness gap was the e4m3 prefill
attention reading 16 never-written P' columns (odd page x zero B: NaN for e4m3 garbage). Same
prompt, 5 requests, 5 different first-token distributions; now 1. nvfp4 gates pass.

## D. Decode all-reduce (lossless first)

- **D1. Fuse the receive with what follows** (lossless). After every all-reduce come the residual
  add, the RMSNorm and the FP4 quantize of the next GEMM's input, as separate kernels. Do them in the
  flag kernel after the sum: the reduce already reads both partials, and the norm of a 5120-wide row
  fits one block. This saves 2–3 launches and a memory pass per call, ×129. Same arithmetic, so it
  must stay bit-identical.
  Estimate: −0.5 to −1 ms a step (+2–3 %).
- **D2. Push while the GEMM runs** (lossless). The partial that goes over the link is the output of
  o_proj / out_proj / FFN down. Split those GEMMs into two N halves: the flag kernel pushes the first
  half while the second computes. The GEMMs are weight-bound, so splitting by N reads no extra
  weights. The receive side waits for both halves.
  Estimate: hides most of the ~12 µs of transfer per call, −1 to −1.5 ms (+3–5 %).
  Not tried in plan 07: B3 tried more blocks for the same op; this overlaps the op with compute.
- **D3. 8-bit wire for the decode all-reduce** (opt-in, lossy, follows `SPARKINFER_TP_AR_WIRE`).
  The same codec as prefill, inside the flag kernel: quantize in registers before the push,
  dequantize both sides before the sum (both ranks add deq(own) + deq(peer)). Per 128-value block
  within a row, so a row's sum does not depend on the row count: speculation stays exact against
  ordinary decode under the same wire.
  Estimate: −0.5 to −0.8 ms (+2 %). Check: teacher-forced KL against the exact link, as for prefill.
- **D4. CUDA graph for the verify step** (plan 07 C1). The flag all-reduce is a kernel and can be
  captured; the host rendezvous cannot and must move outside the graph. Worth it only if P0 shows
  host gaps (≥ 3 %).

## R. The drafter (tokens per second)

Today the draft is 15 % of a step (4.6 of 31 ms), and only ~2 ms of it is weight reads.

- **R1. Candidate merge on the device** (lossless). `d2_head` copies each card's top-16 to the host,
  synchronizes, merges on the host (`tp_exchange_topk`) and uploads the merged list again. Instead,
  rank 1 pushes its 7×16 candidates into rank 0 over P2P with a flag, as the all-reduce does, and a
  kernel merges them. That removes a host sync and two copies from every draft.
  Estimate: −0.2 to −0.5 ms.
- **R2. One graph per draft** (lossless). The draft has fixed shapes (one session, block 8, depth 7):
  capture the 5 layers + the head + the selector once per context bucket, so the host no longer
  launches ~100 kernels a card after each sync.
  Estimate: depends on P0's host gaps; up to −1.5 ms.
- **R3. Smaller candidate head** (changes proposals, output stays exact). The head reads 0.36 GB a
  card per draft (~0.8 ms) to find the top 16 of 248k tokens. Score only the most frequent tokens
  (e.g. 64k, from the target's own output statistics) and fall back to the full head only when
  needed. Plan 07 B6 measured a 32k vocabulary as a net loss for DSpark; DFlash2's codebook selector
  may tolerate it better. Gate it on acceptance (60k replay, greedy and sampled).
- **R4. Two paths per step** (research, large). The selector already has 16 candidates per position.
  Verifying a second path from the first likely divergence (e.g. 8 + 4 rows) would raise tokens per
  step. The cost is the GDN state: the second path starts from the recurrent state at the divergence
  point, which `gdn_ar_steps_kernel` would have to keep (~1.5 MB a GDN layer a card). Estimate the
  gain offline first: from the replay's logged proposals, how often the second-best candidate at the
  first rejected position was the target's token.
  Do this only if it is ≥ +10 % tokens per step.

## L. Long context with nvfp4 KV

- **L1. Plan 08 B1–B3** (nvfp4 decode attention at memory bandwidth): scale on the accumulator for
  QK, fold V's scale into P, hardware e2m1 widening. The verify's attention grows by ~0.09 ms per 1k
  of context (8 rows); the target is the fp8 kernel's cost per byte read.
  Estimate: −1.5 ms at 60k, −3 ms at 120k.
- **L2. nvfp4/fp8 KV drafting is not lossless** (open since plan 08). Find and fix it before any
  item here is measured for exactness with nvfp4 KV.

## F. Prefill link (with the int8 wire)

- **F1. Micro-batch interleave** (plan 08 C1 lever 1, lossless). Two half windows: A.mixer,
  B.mixer, A.FFN, B.FFN, so every all-reduce (FFN down above all, 22.5 ms a call with nothing behind
  it) hides behind the other half's compute. With the wire, the link (~3.4 s at 31k) is below compute
  (~5 s), so this should bring prefill close to compute-bound.
  Estimate: re-profile at 31k with the wire first; −1 to −1.5 s.
- **F2. Pipelined next-layer front for attention layers** (plan 08 C1 lever 2, bit-exact).
  It is done for the 48 GDN layers but not for the 16 attention layers.
- **F3. 4-bit wire** (lossy, experimental). nvfp4 codes with per-16 scales are 4.5 bits instead of
  8.25, roughly halving the bytes again. Only useful if F1 leaves the link exposed; gate it on KL
  (expected to be noticeably worse than int8's 0.05–0.09).

## Order

| # | item | expected | effort |
|---|---|---|---|
| 1 | P0 profile | decides the rest | small |
| 2 | R1 + R2 draft merge on device + graph | +4–7 % decode | medium |
| 3 | D1 fused receive + norm | +2–3 % | medium |
| 4 | D2 push during GEMM | +3–5 % | medium–large |
| 5 | L1 nvfp4 attention | +5–10 % from 60k | medium |
| 6 | D3 decode wire (opt-in) | +2 % | small |
| 7 | R3 frequent-token head | +2–3 % if acceptance holds | small–medium |
| 8 | F1 + F2 prefill | +15–25 % prefill | large |
| 9 | R4 two paths | +10–20 % if the offline estimate holds | large |

If 2–5 land, a step at 20k drops from ~31 ms to ~26–27 ms, i.e. ~100 → ~115–120 tok/s at the same
3.3 tokens a step. That would be at or above HyperQwen's 114 sampled at 60k.

Measurements stay short (a few minutes): the 60k replay (256 tokens), the ~20k opencode final answer,
`tp2_gates.py`, and the deterministic exactness check.
