# 6. Speculation plan: split drafter, drafter memory by window, speculative sampling, n-gram drafting

Written 2026-10-02. Plan only; nothing here is implemented yet. Branch `dual-gpu-plan` at `fdc2dd4`.

**Goal:** opencode-style use gets speculative decoding on every turn, at default (sampled) settings, with
several sessions at once. Today the same use runs at plain decode speed (~52 tok/s).

**Order:**
1. **S: split the drafter across both cards.** Memory and draft time.
2. **W: drafter memory by window, speculation after prefix-cache hits, no 16k limit** (the "Phase 3" from before).
3. **P: speculative sampling.** Requests with temperature > 0.
4. **N: n-gram drafting.**

W comes before P on purpose (see "Why W before P" below).

## Where we stand

**Hardware and models:** 2× RTX 5060 Ti 16 GB at tp=2, Qwen3.8-27B NVFP4. The DSpark drafter:
- 5 layers, all full attention;
- hidden 5120, 40 query heads, 8 KV heads, head dim 128, FFN 10240;
- block 7;
- reads 5 target layers (4, 16, 28, 40, 52);
- has a Markov head (rank 256) and a confidence head;
- shares the target's embedding and lm_head.

**Measured (int8 KV, HyperQwen cohort prompts, 512 tokens, greedy):**

| | 1 request | 2 requests | 4 requests |
|---|---:|---:|---:|
| end-to-end tok/s | 94–98 | 134–148 | 201–208 |
| accepted tokens per session-step | 2.9 | 2.6 | 2.1–2.7 |
| draft ms per step | 4.9 | 4.8 | 9.1–9.5 |
| verify ms per step | 24.7 | 24.5 | 28.5–28.9 |

HyperQwen on the same cards, sampled at default temperature: 110–115 / 191–196 / 306–321.

**What stops speculation in opencode use (from the code):**
- **Greedy only.** `spec_eligible` (`inference_engine.cpp:329`) requires `temperature <= 0`.
  - A request without `temperature` gets the checkpoint's `generation_config.json`: temperature 1.0, top_k 20, top_p 0.95.
  - So it never speculates.
- **No speculation after a prefix-cache hit.** `spec_eligible` also requires `prefill_start == 0`.
- **Speculated turns put nothing into the prefix cache.**
  - The group join prefills the prompt itself (`dflash_generate_group`, `qwen35.cpp` `join`) and takes none of the request's `cache_checkpoints`.
  - So `finish_job_impl` has nothing to insert. A conversation that speculates on every turn re-prefills its whole history on every turn.
- **16k drafter limit.**
  - The drafter's per-session KV is a contiguous `[capacity, 8, 128]` per layer, sized prompt + max_tokens.
  - `dcfg.max_seq` is capped at 16384 (`SPARKINFER_DSPARK_MAX_CTX`), and `join` refuses beyond it.
- **Memory on card 0.** All of the following sit on card 0:
  - the drafter (~2 GB);
  - each session's draft KV (20 KB per token: 5 layers × K+V × 8 heads × 128 × bf16);
  - the join's capture buffer (51 KB per prompt row: 5 layers × 5120 × bf16; whole prompt below 12288, last 4096 rows above).
- **A long join stalls the group.** A joining 30k prompt prefills inside the group loop (~9 s), and the other sessions wait.

**What already exists to build on:**
- The drafter windows its attention: 2048 tokens once the context is ≥ 12288. It records `ctx_lo` for truncated captures.
- `forward_blocks` drafts all sessions of a step in one batched pass.
- The verify (`tp_rows_forward`) computes per-card head argmaxes and exchanges them on the host (`tp_exchange_argmax`). The accept rule is a few host lines (`seg_keep`, `qwen35.cpp` ~4410).
- Prefix-cache entries hold KV blocks plus a recurrent-state snapshot. The KV pool is mirrored on rank 1.
- The leader/peer mirroring (`TP_MIRROR`), `GpuLink` all-reduce, and the host rendezvous used by the prefill agreement.

## Gates for every step

Every step lands only with these passing:
- `dual-gpu/gates/tp2_gates.py --baseline`: lossless, decode, prefill and DSpark count, no perf regression.
- `run_gpu_tests.sh`.
- The cohort benchmark at C1/C2/C4 (`simple_bench.py --cohort prompts_real.jsonl --cohort-len 512`), arms alternated in one binary.
- The 131072 stress run (4 × 12k concurrent, 63k, 2 × 30k), int8 and nvfp4.

**New, needed for W, P and N:** a multi-turn replay (`bench/scripts/multiturn_bench.py`).
- It replays an opencode-like conversation: system prompt + tool schema, then alternating assistant output and tool results of 1–8k tokens, up to 60k tokens.
- 1, 2 and 4 conversations at a time.
- It reports per turn: TTFT, decode tok/s, whether speculation ran, cached tokens.
- It runs both greedy and with default sampling.
- Build it first (step W0), because it is the measurement the whole plan is judged by.

---

## S. Split the drafter across both cards

**Status: done (2026-10-02).** Deviations from the plan below, and measurements:
- The head and Markov chain stay on card 0. The draft scores only vocab ids below 65536 (`SPARKINFER_DFLASH_DRAFT_VOCAB`), all inside card 0's half of the lm_head, so S3's per-row argmax exchange is not needed.
- fc is not replicated. Card 0 projects the context and sends the 10 KB/row result through a zero-padded all-reduce, so card 1 needs no capture buffer (the S-risk about card-1 capture memory does not apply).
- The draft's built-in KV cache is allocated on first use at tp=2 (the group path drafts on per-session states).
- Memory at `--ctx 32768`: card 0 13341 → 12667 MiB, card 1 11489 → 12105 MiB. That is −674 MiB on card 0, not the −1.0 GB estimated: fc (bf16 + Q4), the Markov tables and the 168 MB context-projection buffer remain on card 0. W2 removes the buffer.
- Draft ms/step, split vs whole: C1 3.7 vs 4.5–4.8, C2 3.7 vs 4.7–4.8, C4 6.6–7.2 vs 8.0–9.0. End to end: C1 and C2 unchanged within noise, C4 +6% (221.6/211.9 vs 200.9/204.6).
- The 131072 stress run exposed a policy problem: with card 0 roomier, all four 12k requests joined the group, and the later joins' batched prefill no longer fit, so they fell to the token loop (~700 s each). A join whose batched prefill declines now leaves the group and is prefilled ordinarily. The capture rows are freed after the join's first draft. After that the stress run completes in 41–122 s per request.


**Why first:**
- It removes ~1 GB from card 0 and halves every per-session draft buffer, which W then shrinks further.
- It halves the draft's weight reads per card.
- Card 1 stops idling while the draft runs.
- Its KV layout (4 KV heads per card) is the layout W's sliding cache is built on, so W is written once.

### Layout (per card r ∈ {0, 1})

| tensor | split | per card |
|---|---|---|
| wq, q_norm | query heads [20r, 20r+20) | 20 heads |
| wk, wv, k_norm | KV heads [4r, 4r+4). GQA 5:1 keeps a card's query heads on its own KV heads | 4 heads |
| wo | K-split over the card's heads, then all-reduce | partial [rows, 5120] |
| gate, up | FFN columns [5120r, 5120r+5120) | 5120 |
| down | K-split, then all-reduce | partial |
| fc (5×5120 → 5120), hidden_norm, layer norms | replicated (fc only runs on 1–8 new context rows per step) | whole |
| draft KV cache | the card's 4 KV heads | 10 KB per token |
| lm_head | already vocab-split (the target's own) | half the vocab |
| Markov w2 (vocab × 256) | vocab rows with the lm_head | half |
| Markov w1, confidence head | replicated (small) | whole |
| embedding | as today: own half, plus the per-card cache of peer rows | |

The target's hidden states at the capture layers are already identical on both cards after each all-reduce. So each card captures its own copy, and no capture data crosses the link.

### Steps

**S1. Loader.**
- `DFlashDraftModel` takes a `tp_rank`/`tp_size`, and `load()` slices the safetensors per the table above. The quantized copies are built per card.
- `ModelEngine::load_draft` builds a rank-1 twin on card 1 next to the rank-0 drafter.
- The free-memory check runs on both cards.
- Check: per-card load memory. Expected: card 0 −1.0 GB, card 1 +1.0 GB.

**S2. Split forward.**
- `forward_block` and `forward_blocks` run on both ranks through the same mirroring the target uses: the leader posts the peer's call to the peer worker.
- Per layer: q/k/v and attention on the local heads, then `wo` partial → all-reduce → residual + norm → gate/up local → down partial → all-reduce.
- That is 10 all-reduces of `rows × 5120 × bf16` per draft pass: 70 KB at 1 session, 320 KB at 4 (~0.05 ms each over the link).

**S3. Head and Markov chain.**
- Per row, each card computes its vocab half's logits, plus its half of the Markov bias, then the argmax.
- The cards exchange (value, index) through `tp_exchange_argmax`; that row's token feeds the next row's Markov latent.
- That is one host exchange per block row for all sessions together: 7 per step, ~30 µs each.
- The confidence head is computed on both cards from the identical hidden state.

**S4. Group integration.**
- Draft KV states (`kv_state_create/select/free`) are mirrored like target sessions.
- `forward_blocks` segments are the same on both ranks.
- `SPARKINFER_DSPARK_SPLIT=0` keeps today's card-0 drafter for A/B.

### Correctness
- The split changes only how the draft's sums are ordered (K-split partials in bf16), so it can move acceptance slightly, never output.
- Emitted tokens stay the target's argmax: the lossless gate must stay at 1.
- Acceptance must stay within noise of today's (2.9 / 2.6 / 2.1–2.7 accepted per step).

### Expected
- Card 0: −1.0 GB at load, −10 KB per draft-KV token per session.
- Draft time per step: the batched draft is launch-bound today, so halving the bytes does not halve the time. Estimate: 9 → 5.5–6.5 ms at C4, 4.9 → 3.5–4 ms at C1.
- End-to-end: +3–5% at C1, +8–10% at C4.

### Risks
- The link adds 10 small all-reduces plus 7 host exchanges per draft pass. At C1 that overhead (~0.6–0.9 ms) eats part of the gain.
  - If C1 regresses, keep one card's drafter for single-session groups and split only at S ≥ 2 (both copies would then be needed). Decide on measurements.
- The capture buffer now exists on both cards. Today's prompt-length capture would cost card 1 up to 0.6 GB. W2 removes that buffer, so S lands with the 4096-row cap active on card 1 until W2.

---

## W. Drafter memory by window, speculation after prefix-cache hits, no 16k limit

**Status: W0, W1, W3–W6 done (2026-10-02); W2 and W7 deferred.** Deviations and measurements:
- W0 (8568720): the replay's first run exposed a prefill bug that cost more than anything here. After a prefix-cache hit, a continuation that could not get its batched scratch went to the token loop (102 s / 126 s TTFT at 22k / 28k cached on `--ctx 131072` with the drafter); it now halves and retries like a prefill from 0 (2.0 s / 3.4 s). Chat responses report `usage.speculative_tokens`.
- W1: as planned (C = 12288 + 2 blocks, K = 4096). Under `SPARKINFER_DETERMINISTIC=1`, requests that slide (mid-generation at 12299, and on the first block of a 14k prompt) take the same steps with the same acceptance as unbounded states. C = 8192 / 6144 slid an 11.6k prompt at once and cost 4.5% more steps for 41 MB a card, so C stays.
- W3: the join prefills each range exactly as `ingest_prompt_range` would (batched by the range's own length, else the token loop), so speculated and ordinary requests compute the same thing.
- W4: the snapshot is the draft's newest positions before the job's last checkpoint (≤ 12288; only the kept 4096 once the prefix is past 12288, where the draft windows), attached to that entry's recurrent-state snapshot and counted in its host bytes.
- W5: a join that declines after a hit's prefill started restores the entry's recurrent state before the job is prefilled ordinarily. With more than 4096 new rows past 12288, or no snapshot, the draft starts empty at the capture's first row.
- W6: the join checks only the rows its first block ingests against `SPARKINFER_DSPARK_MAX_CTX`.
- Added, not planned: from 12288 positions the group takes depth 2 (the single-session policy), and a group ends when, over its last 32 steps, it commits tokens slower than ordinary decode (measured by the engine per step size). Without that check, two concurrent long conversations speculated at 38 tok/s each against 48 plain.
- Lossless: at `--ctx 49152`, deterministic, with the prefix cache forced on, a conversation's turns (including a hit that restored a snapshot) match a server without the drafter request for request. At 131072 identical requests differ from each other even without the drafter (prefill windows depend on free memory), and two identical hits on one entry can differ (pre-existing; follow-up).
- Replay (`--ctx 131072`, greedy, one conversation to 60k): 13 of 13 turns speculated (before: 2 of 13), median decode 59.8 tok/s (48.5), end to end 42.9 tok/s (35.5), 112 s (129). Without the guard, four conversations: 35.6 against 28.8 tok/s end to end. Two and four conversations with the guard, and the 131072 stress runs, were not re-measured.

### Steps

**W0. Multi-turn replay benchmark** (see Gates). Record today's numbers greedy and sampled before changing anything.

**W1. Sliding draft KV.**
- Each draft KV state becomes a fixed-capacity buffer `C` with a base position.
- When it fills, the last `K` positions are moved to the front: one device copy of 5 layers × K × 10 KB per card, every `C − K` tokens.
- RoPE is applied when K/V are written, so moved rows stay valid. The attention kernels see a contiguous span, so no kernel learns about wrap-around.
- The defaults keep today's behaviour exactly: C = 12288 + 2 × block and K = 4096.
  - The drafter attends everything below 12288, as today.
  - Above it, its 2048 window always lies inside K.
- Memory per session: 12288 × 10 KB = 123 MB per card, for any context. Today it is 20 KB per token on card 0, 330 MB at 16k.
- Then measure smaller C (8192, 6144) on 6–12k prompts. The earlier 4096 capture cap cost acceptance there (2.97 → 2.37), so C only shrinks where the measurement allows.

**W2. Streamed context ingestion.**
- The drafter's context K/V for a prompt row depends only on that row's captured target states and its position, not on the draft block. So split the first block's context ingestion into `ingest_context(rows, n, pos0)`.
- Call it per prefill chunk, while the target prefills, on the rows that will stay inside the drafter's cache.
- The 51 KB per row capture then exists for one chunk (≤ 2048 rows, ~100 MB, freed after the prefill) instead of the whole prompt (0.6 GB at 12k).
- The first draft block then ingests nothing extra.
- Check: the first block's proposals are bit-identical to today's (same rows, same order).

**W3. Prefix checkpoints from speculated turns.**
- The group join takes the request's `cache_checkpoints` exactly like `step_job`'s prefill: prefill to each checkpoint, snapshot the recurrent state, continue.
- Speculated requests then leave prefix-cache entries.
- This alone fixes "every speculated turn re-prefills the whole conversation".

**W4. Draft context stored with the prefix-cache entry.**
- At each checkpoint, copy the drafter's last K positions (draft KV, 40 MB per card at K = 4096) to pinned host memory, owned by the prefix-cache entry, with the base position.
  - Host memory, not VRAM, so cached conversations cost no card memory.
  - Restoring is ~6 ms per card over PCIe.
- An entry without a draft snapshot behaves as today, so the prefix cache keeps working when the drafter is off.

**W5. Speculation after a prefix-cache hit.**
- `spec_eligible` drops `prefill_start == 0`.
- On a hit, the join:
  1. restores the target state (as ordinary admission does);
  2. restores the drafter's snapshot;
  3. prefills only the new tokens, with streamed ingestion (W2) into the drafter;
  4. drafts.
- With no snapshot (an entry made before W4, or evicted), the join ingests only the new rows and sets `ctx_lo`. The drafter works with a short window (the measured "truncated capture" path), and acceptance recovers as tokens are generated.

**W6. Remove the 16k limit.**
- With W1 nothing in the drafter grows with the context, so `join` checks the draft-state capacity instead of `prompt + max_tokens ≤ 16384`.
- `SPARKINFER_DSPARK_MAX_CTX` becomes the capacity C.
- Check at 32k, 64k and 120k prompts: lossless, and acceptance against the 2048-window measurements.

**W7. Chunked join** (prefill a joining session between group steps).
- A joining prompt prefills one chunk (~2048 tokens) per group step, so running sessions keep decoding.
- The session joins the speculative rows once its prefill completes.
- Matters as soon as opencode runs subagents (one 30k prefill blocks everyone for ~9 s today).

### Expected
- **Multi-turn, greedy:** speculation on every turn. The same tok/s as fresh prompts of the same length, about 95–100 at C1, against 52 today.
- **TTFT on turn N:** only the new tokens are prefilled (as with plain decode today).
- **Memory per speculating session:** about 123 MB per card plus ~100 MB of transient capture during its prefill. Today it is 330 MB + 600 MB on card 0 at 12–16k.

### Risks
- **Checkpoints from speculated turns must restore the same recurrent state** as from ordinary prefill. Check: the prefix-cache parity test, with the turn speculated vs not, then the same next turn compared token for token (greedy).
- **The draft snapshot must match the entry it belongs to:** same token prefix, same base position. Key it by the entry, never by session.

---

## P. Speculative sampling (temperature > 0)

**Rule.** The DSpark draft is greedy, so each proposal is a single token x (the draft distribution q is a point mass).
- With the target's processed distribution p (temperature, top-k, top-p, exactly as non-speculative sampling builds it), row t accepts x with probability p(x).
- On rejection, sample from p with x removed, renormalised.
- If every row accepts, the last row samples the bonus token from p.
- This reproduces the target's sampling distribution exactly; the emitted tokens differ from a non-speculative run only through the random numbers.

**Why it fits the existing verify.**
- The tp verify already reduces each row to a host-side decision, with per-card argmaxes exchanged.
- Sampling needs the per-row candidate set instead of one argmax: each card's top-M (value, index) of its vocab half, with M = 64.
- For top_k ≤ 64 (the checkpoint default is 20), merging the two halves' top-64 lists gives the exact global top-k. Temperature, top-p and the accept test then run on the host on ≤ 128 numbers per row.
- That costs one exchange of 64 × 8 B per row per card, instead of moving 1 MB logits rows.

### Steps

**P1. Per-row top-M on the verify head.**
- A kernel takes per-row top-64 over the card's vocab half (from `vr_lh`).
- `tp_exchange_topm` uses the same host rendezvous as `tp_exchange_argmax`.
- Greedy rows keep the argmax path untouched.

**P2. Accept rule.**
- `seg_keep` is computed per session from (u_t < p_t(x_t)), with the replacement or bonus token sampled from the merged candidates.
- The leader decides, and both ranks read the same result, so the GDN replay (`ms_pass`) and KV truncation stay symmetric.
- Random numbers come from the request's seed and absolute position (counter-based, so a resumed or re-verified row draws the same u).

**P3. Eligibility.**
- `spec_eligible` admits `temperature > 0` with `top_k ∈ [1, 64]` and any `top_p`.
- Penalties, logit bias, logprobs and constraints keep the per-token path for now.
- Sessions with different settings share a group (each row carries its session's parameters).
- `SPARKINFER_SPEC_SAMPLING=0` restores greedy-only.

**P4. Distribution test (new).**
- The target alone vs speculative, same prompts.
- 20k sampled tokens each at temperature 1.0 / top_k 20 / top_p 0.95, and at 0.6.
- Compare per-position token distributions (KS / χ² on the first-token distribution over many seeds, and n-gram statistics over long runs).
- Plus an exact unit test of the accept and residual rule on synthetic logits.

**P5. Measure** on the cohort and multi-turn benchmarks, with default sampling.
- Acceptance will be lower than greedy: per row it is p(x), not 1[x = argmax].
- Then tune the group depth (the measured 6 / 4 policy was chosen on greedy).
- Try a sampled draft (q not a point mass, with the matching min(1, p/q) rule) only if acceptance is poor.

### Expected
- At default sampling, roughly 85–95 tok/s at C1 (HyperQwen 110–115, sampled).
- C4 depends on the acceptance P5 measures.
- Greedy is unchanged.

### Later (not in this plan)
- top_k = 0 or > 64 needs the full-row path (1 MB per row over the link), or a fallback to ordinary decode.
- Presence and frequency penalties: per-row counts including the draft prefix.
- logprobs.
- The tp=1 path (`dflash_generate`) can take the same accept rule.

---

## N. n-gram drafting

Prompt-lookup drafting: when the last few tokens appeared earlier in the conversation, propose what followed them. Agent traffic repeats a lot: file contents read back, tool output quoted, code edits that keep most lines. HyperQwen runs it beside its DFlash drafter.

### Steps

**N1. Proposer.**
- A per-session host index over prompt + output: a hash of the last n tokens (n = 4, 3, 2), giving the most recent position.
- Updated incrementally per accepted token; a lookup is microseconds.
- Proposes up to D_ng tokens that followed the longest match.

**N2. Choice per session per step.**
- If the n-gram match is ≥ 3 tokens with a continuation of at least the DSpark depth, the session verifies the n-gram block and skips the drafter that step.
  - That block can be deeper: up to 12 rows when S ≤ 2, still within the 32-row verify.
- Otherwise it uses DSpark.
- The drafter still ingests the accepted rows' captures on its next DSpark step, so its context stays complete. The per-session capture buffer is sized for the deeper block.

**N3. Sampling.** n-gram proposals are single tokens, so P's accept rule applies unchanged.

**N4. Without the drafter.**
- n-gram-only speculation through the same group verify. Used when the drafter is not loaded or does not fit, and at tp=1.

**N5. Measure:**
- the multi-turn replay;
- a code-edit set (rewrite a 2–4k-token file with small changes);
- the cohort set, which should show no regression where n-grams rarely match.
- Report the share of steps that used n-gram and accepted tokens per step for each proposer.

### Expected
- Large gains on copy-heavy turns (file rewrites, quoted tool output). Several hundred tok/s per session on such spans is typical for prompt lookup.
- Neutral on prose.
- The choice rule must not take n-gram blocks that rarely land (checked by N5's per-proposer acceptance).

---

## Why W before P

P alone makes default opencode requests speculate, but only on turns without a prefix-cache hit. Since speculated turns don't checkpoint (fixed in W3), a conversation would then speculate every turn and re-prefill its whole history every turn: about 9 s TTFT at 30k. W gives cached prefills and speculation together. P can be developed alongside W (it touches only the verify head and the accept rule), but should be enabled by default only after W3.

## Summary of expected results (opencode-like multi-turn, per session)

| stage | greedy | default sampling |
|---|---|---|
| today | 52 tok/s after turn 1 | 52 tok/s |
| after S | same, ~1 GB more room on card 0, draft time −35–40% | same |
| after W | ~95–100 tok/s on every turn, any length | 52 tok/s |
| after P | ~95–100 tok/s | ~85–95 tok/s |
| after N | higher on copy-heavy turns | higher on copy-heavy turns |

These are estimates from today's measurements; each step reports the measured number before the next starts.
