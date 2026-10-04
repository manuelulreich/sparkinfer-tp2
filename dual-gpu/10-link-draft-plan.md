# 10. nvfp4 + DFlash2: the decode step, the link, the drafter

Written 2026-10-03. Status blocks below record what is done.

Configuration: `startup.sh` (nvfp4 KV, DFlash2, `--ctx 131072`, int8 prefill wire), one opencode
session. Every lossless item keeps the speculative output equal to ordinary decode (checked as in
plan 09: `SPARKINFER_KV_DTYPE=int8 EXACT_LINK=1 SPARKINFER_DETERMINISTIC=1 CTX=32768`).

## Status (2026-10-04)

`startup.sh` configuration, sampled (opencode's settings), exact mode unless noted:

| | before plan 10 | now |
|---|---:|---:|
| decode, opencode final answer (~20k context) | ~100 tok/s | 115-123 tok/s (rejection mode 122-124) |
| decode, 60k context | 91-95 tok/s | ~100-110 tok/s (rejection mode 105) |
| step at 20k: draft / verify | 4.6 / 26.7 ms | 3.1 / 24.1 ms |
| prefill, fresh 20k / 60k prompt | 5.9 / 17.1 s | 5.4 / 15.9 s |
| agent turn, ~2k-token delta over 19k | 0.8 s to first token | 0.7 s |
| opencode "Explain this repo to me." | 42.5 s | 33.7 s (HyperQwen 53 s) |
| tp2 gates with nvfp4 KV | 3 of 7 prompts not lossless | determinism, batching, lossless pass |

Done, in order of effect: the NVFP4 head kept at tp=2; linear-time streaming; the cluster top-k;
the e4m3 prefill attention's uninitialized P' (nvfp4/fp8 determinism and losslessness); the
nvfp4 verify attention per session; the verify all-reduces overlapping their GEMMs; opt-in
rejection sampling; the prefill link overlap (async last chunk, chunked projections, 4 chunks);
the wide prefill attention tier for small resumed passes; the pinned snapshot pool; DFlash2 conv
projections in NVFP4. Measured and dropped: the decode 8-bit wire, more KV splits, a restricted
draft head, 4-row verify attention groups.

Concurrency baseline (2026-10-04, after plan 10; `startup.sh`, HyperQwen's `prompts_real.jsonl`,
512 tokens, 8 requests a level, `bench/scripts/simple_bench.py --cohort`; sum of per-request decode
rates in tok/s):

| | C1 | C2 | C4 |
|---|---:|---:|---:|
| now, sampled (`--sampled`, opencode's settings) | 132.1 | 183.1 | 307.9 |
| now, greedy | 141.5 | 238.1 | 331.8 |
| HyperQwen, sampled | 110 | 191 | 306 |
| plan 07 (DSpark, int8 KV), sampled | 95.8 | 174 | 265 |

Per request C2 -> C4 drops 92 -> 77 tok/s (sampled): DFlash2 drafts one session after another, so
batched drafting is the C4 lever; sampled C2 trails greedy C2 (183 vs 238) on acceptance.

Left, each worth a few percent at most and each a kernel or scheduling project: the prefill F8
attention over a long history (~5x off its compute bound for small resumed passes); the verify
attention at long context (latency-bound key groups, 4.8 ms a step at 60k); the prefill
micro-batch interleave (~0.9 s of link waits a 20k prefill); fusing the GDN commit into the next
verify; DFlash2 batched drafting for concurrent sessions; first-request determinism (prefill
windows follow free memory).

## Memory audit (2026-10-04)

`SPARKINFER_MEM_LOG=1` (981c147) prints each card's used memory at the load milestones. Idle per
card at `CTX=131072`, before -> after: 13,159 -> 11,191 MiB (nvidia-smi, card 0), from
(1) the token embedding table in pinned host memory, 1,212 MiB (6bf4cfa, exact);
(2) the all-reduce scratch 512 -> 96 MiB with larger ops in pieces, 416 MiB (dcba169, exact);
(3) one lm_head: the Q4_K copy released, the NVFP4 head serving every row, ~340 MiB (71ca887;
gates pass, eval-corpus perplexity 4.468 -> 4.459). Left as it was: layers 7,856 MiB, DFlash2
draft 970/690 MiB, NVFP4 head 380 MiB, constructor (decode buffers) 240 MiB, CUDA context 204 MiB.

`SPARKINFER_KV_POOL_TOKENS` sizes the shared KV pool apart from `--ctx` (default: equal). Fresh
prompt, `startup.sh`, by pool size: 131k 3.7 / 13.8 s (20k / 60k), 262k 3.7 / 13.9 s, 327k
3.8 / 14.3 s. `CTX=262144` (pool = model max, 2.25 GiB per card) now costs no prefill speed
(it was 60k 20.8 s before the audit). HyperQwen's fp8 pool on the same cards: 210k tokens.
Note: more free memory can change the prefill's window choices (16k windows where 8k were
taken), which changes long-prompt output slightly -- the known "windows follow free memory".

**Fixed prefill reservation (plan 09 C1).** At `--tp 2` prefill now runs fixed 4096-token
windows out of scratch reserved at load (`SPARKINFER_PREFILL_RESERVE`): the warm-up takes
~1.9 GB per card (arena 1,257 MB, the rest the nvfp4 history plane for a full context, GDN
workspace and first-request decode/drafter buffers that a request took anyway). Measured
first: window 4096 / 8192 / adaptive = 20k 3.7 / 3.7 / 3.6 s, 60k 14.1 / 13.7 / 13.6 s, and the
adaptive one was really 8k (16k and the single pass declined every time at a 131k pool). With
the reservation: 3.6 / 13.8 s, at a 262k pool too. Greedy outputs identical with a 2.2 GB
ballast taken after load. The arena reuses buffers in allocation order, so the warm-up runs
every pass-shape class at its largest member (4096, 512 = largest split-K pass, 127 = largest
unaligned tail); remaining growth after it: 2 MB once. Exposed and fixed a startup race in
`batched_prefill_enabled()` (a rank could take the token loop alone).

**KV capacity with the reservation** (`CTX=262144`, nvfp4, DFlash2; free on card 0 after load):
pool 262,144 1.4 GiB; 327,680 674 MiB; 360,448 364 MiB; 393,216 does not load (its warm-up
declines the 4096 window, which now stops the load). 4 concurrent fresh prompts: 4 x 80k = 320k
tokens at a 327k pool ran clean (each 80k prefill ~20.7 s, served one after another); 4 x 89k at
a 360k pool completed, but the later streams' DFlash2 capture context and one session open ran
out of memory (those streams decoded without the drafter). Practical maximum: ~327k tokens
total. Decode cohort at `CTX=262144` (512 tok, prompts_real): greedy C1/C2/C4 130/231/292,
sampled 132/178/270; C4 A/B (131k 307, old memory layout 276) is within its run-to-run noise.

**Per-stream memory (2026-10-04).** Card 0 carried the DFlash2 selector codebooks alone
(243 MiB) and 36 MiB of unused batched-path scratch: codebooks now in mapped host memory (the
select kernel reads its ~113 rows per table over PCIe, draft 3.51 ms either way, bit-identical),
scratch skipped; cards within 4 MiB. Draft KV keep 4096 -> 2066 (window + two blocks) and the
join capture capped at the same span: deterministic acceptance unchanged at 2k/4k/10k, 21k
1.908-1.922 vs 1.892, 51k 1.688 vs 1.719 (rounding of the drafts). Peak over idle per card:
1 stream +168 MiB, 4 streams +522 (was +652 at 15k-token prompts, +718 at 7k). A load-time
reservation for 4 streams (item 1) would therefore take ~520 MiB per card (~57k pool tokens).
360k pool: 646 MiB free after load, 4 x 89k concurrent ran with no allocation failure.

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

**F status (2026-10-04).** Prefill profile, 20k prompt, `startup.sh` config: 4.99 s of kernels,
3.6 s busy per card; the link moved 15.7 GB in 2.4 s (6.5 GB/s). Done: the last FFN-down chunk async
(it was synchronous and exact, draining the queue first: 128 x ~3.3 ms waits), the out / o
projections chunked with early all-reduce posts, 4 FFN chunks. 20k 5.9 -> 5.4 s, 60k 17.1 -> 15.9 s.
Not the lever: memory (a smaller KV pool lets windows grow 4k -> 8k at the same speed), the prefix
cache checkpoints (0.1-0.3 s). Left: ~0.9 s of link waits a 20k prefill, structural -- each layer's
two all-reduces overlap only with the next chunk's row-wise work (F1, the micro-batch interleave).

**R3 status: tried, no gain.** The DFlash2 candidate head restricted to token ids < K (rank 0 only):
K = 65536 draft -0.38 ms but acceptance 3.70 -> 3.54 (greedy, 20k); K = 98304 -0.17 ms, 3.70 -> 3.68
and 3.30 -> 3.26 at 60k. Net zero or worse. The conv projections in NVFP4 (not in the plan) are a
small win: acceptance unchanged, draft -0.2 ms.

**D2 status: done (2026-10-04).** The verify's down, GDN-out and attention-o projections run in
two column halves with the first half's flag all-reduce on a high-priority side stream (small
async all-reduces now take the flag kernel there instead of the copy-engine pipeline). Verify
-0.66 ms at 20k, -0.71 ms at 60k. Not bit-identical to the single GEMM (the half-width GEMM's K
split differs); only the tensor-core verify takes it, which was not exact against plain decode
anyway. Also tried: more KV splits under tp (64 / 128): no gain -- the nvfp4 rows kernel is bound
by per-SM work, not by the split's serial key groups.

**Rejection sampling (opt-in, 2026-10-04).** `SPARKINFER_SPEC_REJECTION=1`: the DFlash2 walk
reports q, the verify accepts with min(1, p/q) and draws a rejection from max(0, p - q) (host,
counter RNG, identical on both ranks; an n-gram proposal is a point mass). Sampled acceptance
+6-10 %, decode +6-10 %. Exact mode (default) stays byte-identical to plain sampled decode; this
one keeps only the distribution.

**Open (found 2026-10-04): the first request after start computes differently.** Deterministic
mode, exact link, no prefix cache, the same greedy request: the first one after start gives one
output, every later one another. The prefill's window partition follows free memory (the scratch
arena is held after the first long prefill), and different window sizes take kernels that round
differently. Fix: a window size that does not depend on memory (reserve it at start, plan 09 C).

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
**F1 status: done as a pipelined GDN middle (2026-10-04, bit-identical).** Profile first (20k,
`startup.sh`, nsys): per 4096-row window, one 1024-row int8-wire all-reduce is 5.4 MB and ~0.85 ms
on the link, an FFN chunk ~0.9 ms of compute, so the FFN keeps the link just busy; but the GDN
middle (conv, scan, gated norm, ~2.3 ms) ran over the whole window with the link idle, and every
next-layer front chunk (~0.25 ms of compute) waited ~0.4-0.6 ms for its down all-reduce. Link per
layer ~6.8 ms against ~7.6 ms of compute, so a full two-half interleave could gain at most the
exposed waits. Instead of interleaving halves, the conv and scan (causal in the rows) run over
2048-row ranges as their fronts land, carrying the conv state and recurrence as consecutive windows
do, and each range posts its out-projection all-reduces before the next range's fronts. Then F2
(below) for the attention layers. Compute-stream gaps 0.87 -> ~0.4 s a 20k prompt.
Measured with `longdec.py` (random words, `/v1/completions`, fresh server), both off -> on:
20k 4.2 -> 3.8 s (4.7k -> 5.3k tok/s), 60k 15.2 -> 14.1 s (3.9k -> 4.2k tok/s). Of that, F1 ~+7 %,
F2 ~+2 % at 20k; 1024-row ranges measure the same as 2048. Left (~70 ms a 20k prompt): the first
FFN chunk after an attention layer waits for o-proj chunk 0 (the attention middle keeps the link
idle; chunking it is not bit-identical with the F8 history planes).

- **F2. Pipelined next-layer front for attention layers** (plan 08 C1 lever 2, bit-exact).
  It is done for the 48 GDN layers but not for the 16 attention layers.
  **Status: done (2026-10-04, bit-identical):** the tp wide arm's [q|gate] / k / v front runs chunk
  by chunk behind the down all-reduces (`SPARKINFER_TP_FRONT_PIPE_ATTN`), ~+2 % at 20k.
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
