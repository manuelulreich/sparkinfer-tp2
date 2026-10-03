# 7. Decode plan: catch up with HyperQwen at C1, C2 and C4

Prefill is ahead of HyperQwen (3,489 against ~1,600 tok/s). Decode is not. This plan lists what makes the difference and what to change, ordered by value per effort. Every item keeps decode lossless: speculative output equals ordinary decode (greedy byte for byte; sampled equal for the same seed).

## Where we stand (2026-10-02, measured)

HyperQwen's cohort test (`prompts_real.jsonl`, 512 tokens), 2× RTX 5060 Ti, sum of per-request decode rates in tok/s:

| | C1 | C2 | C4 |
|---|---:|---:|---:|
| HyperQwen, sampled (default temperature) | 110–115 | 191–196 | 306–321 |
| ours, sampled (`simple_bench.py --sampled`) | 80 | 159 | 229 |
| ours, greedy | 104 | 134–148 | 240 |

Gap when sampled: −30% at C1, −18% at C2, −27% at C4. opencode samples, so the sampled row is the one that counts.

**Our step, from an nsys profile** (cohort, greedy, with `SPARKINFER_DSPARK_TIMING`):

| | C1 (1 session, 7 rows) | C4 (~3.5 sessions × 5 rows) |
|---|---|---|
| step | 26 ms = verify 23.3 + draft 2.6 | 38.5 ms = verify 29.9 + draft 8.6 |
| tokens per session-step, greedy / sampled | 2.8–3.7 / 2.3 | 2.7 / 2.0–2.3 |
| GPU busy | 91–95% | 86–94% |
| weight GEMMs (`device_kernel`) | 62% | 46% |
| all-reduce (`glink_flag_allreduce_kernel`) | 12% (21.7 µs × ~139) | 18% (53 µs × ~139) |
| verify lm_head (`si_mmvq_q4k_multirow`) | 2% (0.6 ms) | 6% (2.2 ms) |
| GDN recurrent steps | 4% | 7% |
| draft GEMVs (`k_gemv_batched_fused3_q4_dp4a`) | 4% | 9% |

What this means:
- **The GPUs are busy.** Launch overhead and host gaps are a few percent, so CUDA graphs are not where the gap is.
- **The weight GEMMs are at the bandwidth limit.** About 16 ms a step reads ~7.5 GB per card at 448 GB/s, and the cost stays flat from 7 to 20 rows. The per-step weight cost is the same as HyperQwen's (its W4A16 is also ~4.25 bits a weight).
- **HyperQwen's step is not faster than ours.** 110 tok/s at its 3.3–3.6 tokens a step is ~31 ms a step, against our 26. **It wins on tokens per step**: 3.3–3.6 at every C and under sampling. We get 2.0–2.3 sampled, and use shallower blocks at C2/C4 and at long context.
- **Why our blocks are shallow:** each extra verify row costs ~0.5 ms at C4. That is all-reduce 0.33 ms (bytes grow with rows over PCIe Gen3 x8), lm_head 0.12 ms (the multirow kernel is linear in rows), and GDN 0.13 ms. At long context, attention adds a full KV read per row (below). So depth_for picks 4 at C2+ and 2 from 12k, where HyperQwen verifies 8 rows a session.
- **Long context (opencode's normal case):** every verify row re-reads the whole KV. The fold that shares one KV read across a session's rows (`fa_split_gqa_kernel<..., SEQ>`) is off for int8 KV (the default). The group verify's step-major row order (row `t*S+j`) also puts different sessions next to each other. Measured on the replay: depth 6 at 14k–55k costs 28 → 40 ms a step.

## A. Tokens per step under sampling (largest gap, opencode's case)

**A1 status: tried, no gain, reverted (2026-10-03).** Built as described (the draft's Markov chain drew each proposal with the verify's own top-k and Philox draw, keyed to the checking row's step; the noise was checked equal on both sides). Acceptance at temperature 1, top_k 20, top_p 0.95: story 1.72 -> 1.75, A* 2.92 -> 2.84 tokens a step, and the same within noise at draft temperature x0.25 / 0.5 / 2 / 4, at +0.7 ms a draft. The reason is the drafter: beyond its first choice its candidates do not match the target's (e.g. draft top 4: 557, 383, 11, 440; target: 557, 995, 66702, 5802), so a coupled draw lands on the target's draw no more often than the argmax does. HyperQwen's sampled acceptance comes from its drafter (DFlash2 with a trained candidate selector), not from the coupling alone.

**A1. Draw the draft with the verify's own Gumbel noise.** Today the drafter proposes its argmax. A sampled target draw lands on that token only with probability about p(argmax): 2.0–2.3 tokens a step, and the gain guard often hands the request back to plain decode. HyperQwen draws each draft position as `argmax(log q + g)`, using the same seeded noise g the target sampler will use at that position (its `_selector_walk_kernel`, "sampling keys a draw by the position before the sampled token"). The target draws `argmax(log p + g)` with the same g, so the two agree whenever p and q put their noisy maxima on the same token. That is far more often than p(argmax) when q ≈ p.
  - Ours: the draft head computes logits over its vocab (ids < 65536, card 0). Apply the request's temperature, top_k and top_p to the draft logits. Add the decode sampler's Philox noise for (seed, token id, step = start + i + 1 − n), the same key the verify row uses. Take the argmax. DSpark's Markov head conditions position i on the token chosen at i − 1, which is now the noised choice.
  - Lossless without further work: P's accept rule (draft == drawn token) is unchanged, and only which proposal gets made changes.
  - Cost: noise for 65,536 ids × 7 positions per session, a few µs. The top-p cumsum is per draft row.
  - Expected: sampled tokens per step near greedy's (2.0–2.3 → ~3). That is sampled C1 80 → ~105, and the gain guard stops ending sampled groups.
  - Check: acceptance per step, sampled against greedy, on the cohort and the multi-turn replay. Byte-identity at a fixed seed against ordinary sampled decode.

**A2 status: done (2026-10-03).** `decode_packed` takes a `SpecSampleRow` per row and the tp rows pass draws sampled rows with P's exact sampler; `tp_rows_forward` declines before touching state when a sampled row cannot take that draw. Deterministic check: two concurrent seeded sampled requests identical packed and unpacked. No drafter, sampled cohort C4: 54 -> 184 tok/s. With the drafter (512 tokens, sampled): C1 / C2 / C4 95.8 / 157 / 248 tok/s; the gain check compares against the packed step now.

**A2. Batched sampled plain decode.** `step_jobs_packed` declines when a session samples, so sampled sessions fall back one row at a time (S × 18 ms a step). This hurts twice: every handed-back sampled request at C2/C4 is slow, and the gain guard compares against that slow baseline.
  - Give the packed decode per-row sampling (the same kernel as P's per-row draw) so 2–4 sampled sessions share one weight read.
  - Expected: sampled ordinary decode at C4 ~4× faster (the same as greedy packed), and an honest guard.

**A3 status: done (2026-10-03).** The block and each session's capture buffer are sized for `SPARKINFER_NGRAM_DEPTH` (default 15) lookup tokens; a step whose copy runs verifies 16 rows. No draft-side change was needed: an ingest past 8 rows takes the per-session draft path, and the draft KV need is bounded by prompt + output, not by the chunk. Copy of a 150-line file: 267 -> 399 tok/s at 1.7k, 248 -> 354 behind 14k; lossless (deterministic, greedy and sampled); prose unchanged.

**A3. Prompt lookup past 8 rows** (plan 06, N, remaining). While a copy runs, verify up to 15 lookup tokens (HyperQwen's `DFLASH_TOKENS=15`, "381 tok/s while quoting").
  - Size `cap` and the draft's ingest for 16 rows. `forward_blocks` batches only ingests of ≤ 8 rows, so a longer one goes through `forward_block`, or the limit is raised. Leave 16 rows of draft KV slack.

## B. Make deep blocks cheap (lifts C2/C4 and long context)

**B1 status: done for int8 (2026-10-03).** `launch_flash_decode_split_pairs`: two rows of one session per CTA of the int8 6:1 tensor-core split (the second row's q-heads in the mma's unused M rows); rows whose split ranges start differently take two passes, the other row an exact no-op. Bit-identical (in-server check, 5000+ calls, one and two sessions). Smaller than estimated: verify at depth 6 behind 25.6k 30.3 -> 28.9 ms, 7 rows behind 14k 28.8 -> 27.6 ms; per verify row the attention is ~0.35 ms of the ~1.4 ms at 25.6k, so depth 2 stays the long-context default. The fp8/nvfp4 twin is done too (`fa_split_gqa_mma_f8_pair_kernel`, bit-identical over 3000+ calls each; 8 rows behind 14k: fp8 27.4 -> 26.5 ms, nvfp4 29.6 -> 27.7 ms). Not done: the remaining per-row cost (all-reduce, lm_head, GDN: B2-B5). Found on the way: a fresh prompt of 16k+ never speculated (first draft over the draft's max_seq) -- fixed, 49.6 -> 75 tok/s at 25.6k.

**B1. Fold verify attention across a session's rows, int8 KV included.**
  - Order the group verify session-major inside attention: gather a session's T rows next to each other, or index rows by (session, t).
  - Enable the SEQ fold for int8 KV. The int8 tile is dequantized into the same bf16 smem tile, so the fold's math is unchanged.
  - Also fold in `fa_split_gqa_mma_i8_kernel`, the int8 hd256 path, if that is what runs.
  - Expected at 30k: the step's attention cost becomes ~one row's instead of T rows'. That lets depth_for keep 6 above 12k instead of 2. On the replay, turns at 14k–55k drop from ~40 ms back toward ~28 ms at depth 6, and accept 2.8 instead of 1.8 tokens. This is the largest win for opencode's long conversations.

**B2 status: tried, no gain, reverted (2026-10-03).** A side stream issued `prefetch.global.L2::evict_last` over the first 4 / 8 / 16 MB of the next GEMM's weights (FFN gate during the attention all-reduce, the next layer's first projection during the FFN all-reduce). Cohort, 256 tokens, step-weighted verify: C1 23.14 -> 23.05 / 23.04 ms, C4 28.81 -> 28.82 / 27.91 (noise). Either the lines do not survive until the GEMM's CTAs reach them or the GEMM's read order does not start where the prefetch did; not pursued further.

**B2. Hide the all-reduce behind the next layer's weight stream.** The all-reduce waits on the link: 22 µs at 7 rows, 53 µs at 20 rows, ~139 of them a step. In that time the card can stream the next GEMM's weights into L2: 53 µs × 448 GB/s ≈ 24 MB, and the 5060 Ti has a 32 MB L2 (20 MB can be set aside as persisting). A layer's gate+up slice is ~50 MB per card, so the first ~40% of it (or all of a smaller projection) can be in L2 before the GEMM starts. Issue an L2 prefetch (`cp.async.bulk.prefetch.L2` / `prefetch.global.L2`) of the next weight tiles at the start of each all-reduce wait, and have the GEMM read those tiles first.
  - Numerics: unchanged (only memory traffic).
  - Expected: most of the 12% (C1) to 18% (C4) all-reduce time, and with it the main per-row cost.
  - First a microbenchmark: AR alone, AR + prefetch, then the following GEMM's time with and without the prefetched L2.

**B3 status: done in part (2026-10-03).** The flag all-reduce's push and reduce now move 16 bytes a thread (one element before, so the link carried 64-byte writes): verify at C4 28.7 -> 26.6 ms, at C1 23.1 -> 22.8 ms, the split draft at C4 6.0 -> 5.4 ms. Splitting the op over up to 16 blocks with a flag slot each measured no further gain (C1 -0.15 ms, C4 flat), so it stays one block. Not tried: reduce-scatter + all-gather, fusing the residual + RMSNorm into the receive.

**B3. Use the link better.** 130 KB of extra payload costs 31 µs (4.2 GB/s) on a Gen3 x8 link that sustains ~6.5.
  - Try more pushing blocks and 16-byte stores.
  - Try a split push: each card pushes half the rows and reduces locally, then pushes the result back (reduce-scatter + all-gather). This halves the bytes each card writes when both directions are used at once.
  - Fuse the residual add + RMSNorm into the receive side.

**B4. Tensor-core lm_head for verify rows.** `si_mmvq_q4k_multirow` costs 0.6 ms at 7 rows and 2.2 ms at 20 rows, linear in rows. An MMA GEMM over the q4 head is weight-bound (~0.6 ms flat).
  - Greedy keeps its argmax order; sampling needs the exact top-64 per card (P), which a GEMM tile + the existing top-k kernel provide.
  - Expected: −1.6 ms a step at C4.

**B5. GDN steps for many rows.** `gdn_ar_steps_kernel` runs T recurrent steps a session: 10 µs at C1, 27 µs at C4, ~95 launches a step.
  - Fuse `conv_split_l2norm_steps` into it.
  - Run sessions × heads across more CTAs.
  - For T ≥ 6, use the chunked (WY) form the prefill uses.
  - Expected: −1 to −2 ms a step at C4.

**B6 status: the head done (2026-10-03).** The profile's `si_mmvq_q4k_multirow` (0.6 ms at C1, 2.2 ms at C4) was the DRAFT head, not the verify's (the verify already uses the NVFP4 head copy). The batched draft now runs it as one NVFP4 tensor-core GEMM over the first 65536 rows of card 0's FP4 head (a whole-atom prefix of its data and N-outer SFB scales): draft at C4 7.6 -> 6.0 ms. The single-session draft keeps the GEMV (its block is narrower than the 8-row GEMM tile). A smaller draft vocabulary (32768) was measured as a net loss (acceptance 2.87 -> 2.73 for -0.4 ms). The drafter's own linears are still dp4a GEMVs.

**B6. Drafter GEMMs for many rows.** At C4 the drafter's dp4a GEMVs are 73 µs (38 at C1), because 4 sessions × 8 rows = 32 rows is past where a GEMV stays weight-bound. Switch the drafter's linears to an MMA GEMM at ≥ 16 rows.
  - Expected: draft 8.6 → ~4 ms at C4.

**B7. Depth per session (ragged verify).** Every session of a step verifies the same depth today. A copying session should go 15 deep while its neighbours stay at 4.
  - Lay rows out as per-session runs with their own lengths: the rows pass takes per-row positions and sequences already, and `seg_keep` becomes per-run.
  - Choose each session's depth from its own recent acceptance (the single-session planner's rule) and DSpark's confidence head.
  - Expected: deeper blocks where they land, shallow where they do not; this matters most with the lookup (A3).

**B8. Then retune depth_for and the gain guard** with B1–B7 in. HyperQwen's 8 rows a session at C4 is the target.

## C. Smaller items

- **C1. CUDA graphs for the verify step.** GPU busy is 91–95%, so ~3–5% at most. It needs a capturable all-reduce (the flag kernel itself is a kernel; the host rendezvous is not). Do it last.
- **C2. Overlap the draft with the verify tail.** The drafter runs after the verify and its exchange. Card 1 waits through the draft's card-0-only parts (head, Markov chain). Moving the Markov chain to the card that finishes first, or starting the next draft's ingest of the accepted rows while card 0 merges, saves ~1 ms a step.
- **C3. Bigger drafter window at long context.** Above 12288 the draft attends 2048 positions. Measure acceptance at 4096/8192 (memory: kept span, `SPARKINFER_DSPARK_KV_KEEP`).
- **C4. Seats and joins.** W2 (streamed ingestion) and W7 (chunked join) from plan 06, so a joining 30k prompt does not stall the group's decode.
- **C5. Memory on card 0.** The drafter, its session state and the verify scratch sit on card 0. At `--ctx 131072` that leaves 8–28 MB free and prefill windows shrink to 2176 (see the log in the answer to "is this expected"). Release the drafter's bf16 copies after quantizing, and move the 168 MB context-projection buffer to card 1 or shrink it (W2).

## Note: balancing VRAM between the cards

**Measured** at `--ctx 131072`, int8 KV, with the drafter:

| | card 0 | card 1 |
|---|---:|---:|
| idle | 14,317 MiB | 13,755 MiB |
| after a 25.6k-token request | 15,095 MiB | 14,457 MiB |

Card 0 carries 560–640 MiB more. The KV pool is sized by the tighter card, so card 1's spare is wasted, and card 0 is the one whose prefill scratch fails first.

**What sits only on card 0 (permanent):**

| item | size |
|---|---:|
| the draft's fc projector in bf16, kept for the first block's tensor-core GEMM | 262 MB |
| fc's Q4 copy | ~74 MB |
| Markov head w1 (bf16 [248320, 256]) | 127 MB |
| Markov head w2 (int8 + scales) | ~72 MB |
| draft logits / head scratch | ~20 MB |
| **total** | **~555 MB** (matches the measured gap) |

**Transient, during a join:** the capture rows, 51 KB a prompt row. That is up to 210 MB above 12k (the last 4096 rows) and the whole prompt below 12k (up to 630 MB at 12k). This is what pushes card 0 to 8 MB free during a long prefill.

**How vLLM avoids it:** its drafter is a tensor-parallel model like the target.
- fc is a column/row-parallel linear and the heads and embedding are vocab-parallel, so no rank holds an unsharded piece.
- The KV block count is the minimum of each rank's profiled free memory.

**Steps to make card 0 ≈ card 1:**
1. **Split fc along K across the cards (−131 MB bf16 and −37 MB Q4 on card 0, the same added on card 1).**
   - The target hidden states at the capture layers are identical on both cards after each all-reduce, so card 1 can capture its own half of fc's input columns.
   - Each card projects its half. The existing zero-padded all-reduce of `target_proj` becomes a real sum, with the same bytes on the link.
   - The capture rows split the same way, halving card 0's transient.
   - Gap after this step: ~560 → ~250 MB.

   **Step 1 status: done (2026-10-03).** Each card captures its own columns into its own half-width buffers; rank 1 finds its twins through a pointer map the leader keeps (`dflash_cap_peer`). Measured idle at `--ctx 32768`: card 0 12,669 → 12,505 MiB, card 1 12,107 → 12,273 MiB (gap 562 → 232). Greedy deterministic: same outputs, same acceptance (2.85 tokens a step), draft 2.42 → 2.23 ms. `SPARKINFER_DFLASH_FC_SPLIT=0` reverts.
2. **Drop fc's bf16 copy** by running the first block's projection as an NVFP4 tensor-core GEMM (as the verify and now the draft head do). That frees the 131 MB per card left by step 1. The draft's context projection changes numerically, but it only affects proposals, not output.

   **Step 2 status: done (2026-10-03), without FP4 activations.** All draft projections are NVFP4 payloads (o/gate/up/down as stored in the checkpoint, q/k/v/fc quantized at load); a new dp4a GEMV (`launch_gemv_nvfp4_q81`) takes the existing Q8_1 activations, and a prompt's context runs the bf16 tensor-core GEMM on slices dequantized on the fly. No bf16 fc, k/v or int4 copies remain: −238 MB on each card, draft time equal or lower (C1 2.23 → 2.19 ms; C4 deterministic 6.71 → 6.57 ms at equal acceptance). bf16 activations (W4A16) measured 4.35 ms a draft at C1 and were dropped. A W4A4 tensor-core path is left for later.
3. **Markov w1 to int8 with a per-row scale** (−63 MB on card 0). It is an embedding lookup of the previous token, read once per proposal row.
4. **Stream the capture into the draft (plan 06, W2)**, so a join holds a window of rows instead of up to 4096 (−170 MB transient).
5. What remains (~100–150 MB: w2, head scratch) can be matched by giving card 1 a symmetric buffer that is today only on card 0 (e.g. the verify's sampling scratch), or left as is.

After steps 1–3, card 0 carries ~120 MB more instead of ~560. Its prefill scratch then fails no earlier than card 1's.

## Order

| # | item | expected effect | effort |
|---|---|---|---|
| 1 | A1 Gumbel-coupled draft | sampled C1 80 → ~105, C4 229 → ~290 | small–medium |
| 2 | A2 packed sampled decode | sampled fallback ×S faster, honest guard | medium |
| 3 | B1 verify attention fold (int8, session-major) | depth 6 at long context; long turns ~1.5× | medium |
| 4 | B4 tensor-core verify lm_head | −1.6 ms/step at C4 | small–medium |
| 5 | B2 L2 prefetch during all-reduce | up to −12–18% of the step | medium, uncertain |
| 6 | B6 drafter MMA GEMMs | draft 8.6 → ~4 ms at C4 | medium |
| 7 | B7 ragged depth + A3 15-row lookup | copy turns ~1.5–2×, C2/C4 deeper | medium–large |
| 8 | B3, B5, B8 | a few ms each, then retune | medium |
| 9 | C1–C5 | small or situational | — |

1–4 should close the C1 and C2 gap. C4 needs the per-row items (B2, B4–B6) so that 8-row blocks pay off at four sessions.

## Gates for every item

- `tp2_gates.py --baseline` (lossless, decode, prefill, DSpark), `run_gpu_tests.sh`.
- The cohort at C1/C2/C4, greedy and `--sampled`. The multi-turn replay (short: `--turns 4`) for the long-context items.
- Sampled byte-identity at a fixed seed against ordinary sampled decode, at `--ctx 32768` with `SPARKINFER_DETERMINISTIC=1`.
