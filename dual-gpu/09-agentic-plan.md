# 9. Agentic decode: speculate under tool grammars, DFlash2, prefill memory

Written 2026-10-03. Plan only; nothing here is implemented yet.

## What the user sees

opencode, this repository, the prompt "Explain this repo to me." on the same 2× RTX 5060 Ti:

| server | first turn | opencode tok/s |
|---|---:|---:|
| HyperQwen (vLLM 0.29, W4A16 target, fp8 KV, DFlash2 k=7) | 53 s | ~64 |
| sparkinfer-tp2 (`startup.sh`: nvfp4 KV, DSpark, `--ctx 262144`) | 1 min 40 s | ~35.5 |

The two sessions took different paths (ours read the 24k-token CHANGELOG and ended at ~74k
tokens of context, HyperQwen's at ~26k), so the 2× is not clean. Two measurements make it clean.

## Measurements

**M1. What our server did on an opencode turn.** opencode was run headless
(`opencode run`) through a logging proxy. The 7 requests (title + 6 agent steps, 8.4k → 60.7k
tokens of context) all decoded at 50–57 tok/s, which is plain decode. `/metrics`:
`sparkinfer_speculative_runs_total 1`, `sparkinfer_speculative_tokens_total 7`,
`sparkinfer_tool_calls_constrained_total 6`.

**Cause: DSpark never runs on an agent request.** opencode sends its 12 tool definitions with
every request. Every tool-calling request is decoded under the tool grammar (`TokenConstraint`,
server/src/tool_grammar.cpp), and `ContinuousBatchEngine::spec_eligible`
(runtime/src/inference_engine.cpp) refuses any request with `r.constraint`. That holds for the
reasoning and the final answer too, not just the tool-call JSON. vLLM applies no grammar under
`tool_choice: auto` (it parses the calls afterwards), so HyperQwen speculates on every token.

**M2. The same request on both servers.** We replayed the captured final step (60,682 tokens of
context, 1,500 generated tokens) on both servers. On ours it ran at `--ctx 131072` with
`SPARKINFER_DSPARK_TIMING=1`. To let DSpark run at all we used `tool_choice: "none"`, which keeps
the tools in the prompt but drops the grammar:

| server, setting | sampled (T=1, top_k 20) | greedy |
|---|---:|---:|
| ours, as opencode sends it (constrained, no speculation) | 49.5 tok/s | – |
| ours, DSpark, default depth (2 past 12288) | 50.8 tok/s; 1.53 tokens/step at 26.1 ms; the gain guard stops the group | 76.8 tok/s; 2.08 tokens/step at 24.7 ms |
| ours, DSpark, depth 6 (`LONG_DEPTH=0`, `MIN_GAIN=0`) | 61.6 tok/s; 1.97 tokens/step at 28.9 ms | 68.0 tok/s; 2.16 tokens/step at 28.7 ms |
| HyperQwen, DFlash2 k=7 | **113.9 tok/s**; 3.74 tokens/step | **102.2 tok/s**; 3.42 tokens/step, ~33 ms a step |
| plain decode, ours | ~50 tok/s, 20.0 ms a step | |

- **Our verify step is not the problem.** 3 rows cost 24.6 ms and 7 rows 28.7 ms, against
  20.0 ms for plain decode: ~1.1 ms per extra row. HyperQwen spends ~33 ms on 8 rows.
- **Our drafter is the problem at long context.** DSpark gets 1 extra token per step out of 6
  proposed (greedy 2.16, sampled 1.97). DFlash2 gets 2.4–2.7 out of 7.
  - The drafter's 2048-token attention window is not the cause: dflash_draft.cpp documents
    sweeps where acceptance is unchanged across windows.
  - DSpark's acceptance simply falls with context: tau 1.28 on prose at 32k.
- **Prefill is our advantage.** A cold 57–61k prompt took 15.7 s here (3,650 tok/s, 8-bit link,
  131k pool) against 46.7 s on HyperQwen (1,300 tok/s).
- **At `--ctx 262144` our prefill degrades.** Prefill windows fell to 2048–4096 tokens for lack
  of memory. The same 57k prompt took 23.4 s instead of 15.7 s, and a 14.5k-token resume ran at
  ~1,900 tok/s.

## Plan, in order

### A. Speculate under the tool grammar (the agent path gets DSpark at all)

Agent requests are 100 % constrained, so nothing else in this plan reaches opencode until this
lands.

- **A1. Grammar-aware verify.**
  - For a constrained session in a speculative group, walk the drafted path on the host. Use the
    grammar matcher's own state with rollback; xgrammar's `GrammarMatcher` keeps a rollback
    window (`max_rollback_tokens`). This still needs confirming against the vendored version.
  - Each verify row i gets the mask after drafted tokens 0..i-1.
  - The verify already reduces each row to its exact top-64 (the sampled path, plan 06 P), and
    the mask is applied to those candidates:
    - greedy: the first allowed candidate is the masked argmax;
    - sampled: draw over the allowed candidates with the same Gumbel noise.
  - When none of a row's 64 candidates is allowed, which can only happen inside the tool-call
    JSON, the step commits up to that row. The session hands that token to per-token
    constrained decode and rejoins the group afterwards.
  - After the commit, call `accept()` for the committed tokens and roll back the rest.
- **A2. Draft only where the grammar is free text.**
  - Inside the call JSON, DSpark's guesses mostly do not survive the mask. The arguments are short
    next to the reasoning and answer anyway.
  - So a constrained session's group step proposes depth 0 while the matcher is inside
    `<tool_call>…</tool_call>`.
  - A first version may simply hand back to per-token decode at `<tool_call>` and not rejoin. A
    turn ends with its calls, so that loses little.
- **A3. Accounting.**
  - `usage.completion_tokens_details.reasoning_tokens`: opencode shows 0 reasoning for us and
    its tok/s figures are not comparable with vLLM's.
  - A per-request `speculative_tokens` in streamed usage, so the next comparison can be read
    from the client.
- **Expected with DSpark.**
  - Sampled ~62 tok/s at 60k (M2, depth 6) instead of ~50.
  - Greedy ~77.
  - More below 12288, where DSpark is strong.
  - Interim default for sampled agent traffic: `SPARKINFER_SPEC_GROUP_LONG_DEPTH=0`. Depth 6 wins
    for sampled text (61.6 vs ~50) and loses for greedy (68.0 vs 76.8). Retune the depth ladder
    after B.
- **Check.**
  - Byte-identical output to constrained per-token decode under `SPARKINFER_DETERMINISTIC=1`,
    greedy, at ctx 32768.
  - The tool-call gates (`invalid_tool_output` stays 0).
  - The M1 replay shows `speculative_tokens` > 0 on every step.

**Status of A (2026-10-03): A1 and A3 done and verified on the GPUs; A2 not needed so far.**
- Exactness: int8 KV, `SPARKINFER_DETERMINISTIC=1`, `--ctx 32768`. Two captured tool turns,
  each greedy and with two sampled seeds, are byte-identical with and without speculation,
  and every token was speculated. tp2 gates all pass (lossless 7/7, DSpark 275 / 83.5 tok/s).
  With nvfp4 KV the outputs differ: that is plan 08's open nvfp4/fp8 losslessness issue.
- The first GPU run found a bug in the sampled verify's top-k (`k_rows_topk`). Masked rows tie
  ~124k values at -1e9, and the real candidates lost slots to them. The model then could not
  draw `<|im_end|>` or `</tool_call>`, and tool turns ran to max_tokens. Fixed.
- `build_masks` stops at a drafted end token: xgrammar aborts on a mask request after the
  stop token.
- opencode "Explain this repo", default `startup.sh` (`--ctx 131072`), 7 requests, all
  speculated:
  - 65.5 s, against 82 s without speculation;
  - tool steps at 7-24k context: 74-100 tok/s instead of ~50;
  - final answer at 26.5k context: 57 tok/s, 2,530 tokens, with DSpark at 1.74 tokens a step.
    The gain guard ended speculation after 1,105 tokens. That is B's job.
- `TokenConstraint::can_rollback/rollback`; GrammarConstraint keeps its UTF-8 state history and
  calls `GrammarMatcher::Rollback`. tool_grammar_test now rolls back random drafted walks and
  checks that the mask is unchanged.
- `SpecGroupJob::constraint`. The group loop's `build_masks` walks block[0] and the drafted
  tokens, takes each row's mask, then rolls back.
- `spec_group_verify` / `tp_rows_forward` take `row_mask`. Each rank adds -1e9 to its own vocab
  half of the masked rows (`launch_rows_mask_bits`, the engine's `kMasked`) before argmax or
  top-k. So greedy and sampled rows draw exactly what `apply_constraint_mask` + decode would.
- Engine:
  - `spec_eligible` admits constraints that can roll back;
  - `run_speculative` (the lone, non-group path) still refuses them;
  - `spec_emit` advances the constraint with each emitted token, as `step_job` does, but not for
    the end token.
- Usage: `completion_tokens_details.reasoning_tokens` (tokens before the first `</think>`) in
  streamed and non-streamed chat, and `speculative_tokens` in streamed chat.
- A2 is deliberately not done yet. The drafter proposes the model's own tool-call tokens, which
  the mask rarely changes, so first measure acceptance inside `<tool_call>` before adding a
  depth-0 rule.
- GPU checks, in order:
  1. M1 replay: every step reports `speculative_tokens` > 0 and `invalid_tool_output` stays 0.
  2. Deterministic greedy at ctx 32768, with a tool call: output identical to
     `SPARKINFER_SPEC_GROUP_MAX=0`.
  3. Cost of `build_masks` per step: `SPARKINFER_DSPARK_TIMING` draft/verify ms against the
     M2 numbers.

### B. Port DFlash2 (the drafter gap at long context)

The user deferred this earlier. M2 now makes it the main lever: 3.4–3.7 tokens/step against our
~2.0 at 60k context, with our verify costing about the same per row as vLLM's.

**What DFlash2 is.** HyperQwen's `patches/dflash2-backport.patch` (vLLM PR #52816) and the
checkpoint `incoai/Qwen3.8-27B-DFlash2` (HyperQwen ships a GPTQ W4A16 requant) define it as the
same block drafter family as DSpark:
- 5 Qwen3 layers, hidden 5120, block 8 (bonus + 7 masks);
- non-causal attention over the context KV built from 5 target layers through `fc`.

It differs from DSpark in:

| | DSpark (ours) | DFlash2 |
|---|---|---|
| target layers | 4, 16, 28, 40, 52 | 5, 19, 33, 47, 61 |
| heads / KV heads / FFN | 40 / 8 / 10240 | 32 / 8 / 17408 |
| attention | full (we window it at 2048 past 12288) | sliding 2048, trained that way; RoPE theta 1e7, no YaRN |
| per-layer extra | – | `attention_conv` and `mlp_conv`: grouped 2-tap convolution along the block positions, coefficients from a small `kernel_projection` (bf16) |
| token choice | lm_head argmax + Markov/confidence heads | `CandidateSelector`: top-16 per position from the target lm_head, then pairwise edge scores (`predecessor_codebook`, `successor_codebook`, vocab × 256, plus `hidden_projection`) and a path walk over the 7 × 16 lattice (greedy, or sampled with the request's top_k/top_p) |
| weights | NVFP4 (as stored) | bf16 original, 1.92B params; HyperQwen ships int4 GPTQ g128 |

- **B1. Weights.**
  - Take the bf16 original from Hugging Face and quantize the Linear layers to NVFP4 ourselves
    (W4A8 GEMV path `launch_gemv_nvfp4_q81`, as DSpark since plan 07). Keep the conv,
    selector and norms bf16.
  - Do not requantize HyperQwen's int4 g128: int4 levels 5 and 7 have no e2m1 code.
  - Per card about 0.55 GB of drafter weights split as DSpark is, plus 2 × 127 MB of selector
    codebooks. Check memory at `--ctx 131072`.
- **B2. Model.**
  - A `DFlash2DraftModel` beside `DFlashDraftModel`, reusing:
    - the context capture (other layer ids);
    - the sliding KV state (window 2048, now the trained one);
    - the tp=2 split;
    - the batched group path.
  - New kernels:
    - the grouped convolution (elementwise, 2 taps over block positions; fuse into the norm or
      the projection epilogue);
    - the candidate top-16 off our split lm_head (each card's top-16 then merge, as the verify
      top-64 does);
    - the edge scoring and the path walk. That is one small kernel per session: 7 × 16 × 16
      scores, rank-256 dot products.
- **B3. Verify.**
  - Sampled requests need the proposal distribution for rejection sampling. DFlash2 proposes
    from a top-16 lattice with explicit scores (`_cache_draft_logits`).
  - Our sampled verify (plan 06 P) emits exactly what ordinary sampled decode would, so it needs
    no proposal probabilities: tokens are accepted while the drafted token equals the verify
    draw. DFlash2 should drop into it unchanged.
  - Confirm that, and that the selector's sampled path (`draft_sample_method: probabilistic` in
    HyperQwen) beats a greedy path under our acceptance rule.
- **B4. Measure.**
  - M2 with DFlash2 at depth 7. Target ≥ 100 tok/s sampled at 60k, ≥ 3.3 tokens/step.
  - Then the short-context gates (`dspark_tps_*`). Keep DSpark selectable (`--draft-model` picks
    by `architectures`) until DFlash2 wins everywhere.
- **Estimate.**
  - ~8 rows at ~29 ms verify (M2 scaling) plus ~2–3 ms draft is ~31 ms a step.
  - At 3.4–3.7 tokens/step that is ~110–120 tok/s at 60k, on par with HyperQwen.

**Status of B (2026-10-03): B1-B3 done, B4 measured. A sub-agent ported it in a worktree; it
was merged and tested here.**
- `DFlashDraftModel` serves both drafters. `cfg.dflash2` comes from `config.json`.
- The bf16 checkpoint is quantized to NVFP4 once, on the CPU, into
  `sparkinfer-nvfp4.safetensors` (1.55 GB, ~30 s; `SPARKINFER_DFLASH2_NVFP4_CACHE`).
- Split at tp=2: 16/32 query heads, 4 KV heads, FFN 8704 per card.
- Selector: each card takes the top-16 of its vocab half, the halves merge on the host, then
  rank 0 runs the greedy walk.
- Depth 7 at every context (`spec_depth`).
- Memory at `--ctx 131072`: card 0 at 13.26 GB idle (DSpark: 12.97).
- Lossless: tp2 gates pass 7/7. Gate counting 303 tok/s, list prose 111.7 (DSpark 275 / 83.5).
- M2 replay (60.7k context, the opencode step with tools, constrained):

  | | greedy | sampled |
  |---|---:|---:|
  | DSpark | 76.8 tok/s, 2.08/step | 50.8 tok/s, 1.53/step (group stopped) |
  | DFlash2 | **100.0 tok/s, 3.56/step**, 30.3 ms verify + 4.3 ms draft | **76.6 tok/s, 2.73/step** |
  | HyperQwen | 102.2 tok/s, 3.42/step | 113.9 tok/s, 3.74/step |

**B5 (next): the sampled gap is the acceptance rule.**
- vLLM accepts with rejection sampling (Σ min(p, q) per token). We accept while the draft
  equals the verify row's own draw, which keeps output identical to ordinary sampled decode.
- With a greedy walk, that accepts a token with probability p(argmax).
- Couple the walk to the verify's noise instead:
  - in each step, the selector picks argmax(score / T + g), where g is the same Philox Gumbel
    noise the verify draws with for that position and candidate (seed, token id, step);
  - restrict it to the request's top_k / top_p the same way.
- When the selector's distribution is close to the target's, the two argmaxes coincide far
  more often than p(argmax). Output stays exactly ordinary sampled decode.
- DSpark's attempt (plan 07 A1) failed because its candidates do not match the target.
  DFlash2's selector scores are trained as a distribution over the top-16.

### C. Prefill memory: reserve it, then size the KV pool

- **C1. Reserve the prefill arena at startup.**
  - Reserve it for a chosen window (16384, fallback 8192) after weights and drafter, and size
    the KV pool from what remains.
  - Alternatively, refuse a `--ctx` that leaves no arena, as vLLM's profile run does.
  - Today the pool is sized from `--ctx` first and prefill allocates from whatever is left per
    request, so speed and output depend on free memory (M2: 23.4 s vs 15.7 s for the same 57k
    prompt).
- **C2. Shrink the arena.** Find what a 16384-token window holds; it does not fit in the ~2.8 GiB
  free at a 131k pool.
  - The startup log names one item: windowed GDN slots stage through the full-layout
    `gdn_scratch`.
  - The other items are not yet measured.
  - Goal: full-speed 16384 windows with a 262,144-token nvfp4 pool.
- **Until then: `startup.sh` defaults to `--ctx 131072`.** The opencode config's
  `limit.context` is 125,000.

### D. After A–C

- **Port the paired verify kernel to nvfp4 KV.** It is still open from plan 08; verify rows
  read the KV once per row today.
- **Investigate DSpark/DFlash2 losslessness with fp8/nvfp4 KV.** It is open from plan 08; int8
  is lossless.
- **Rerun M1 end to end with opencode** and compare with HyperQwen on the same captured
  trajectory, not a fresh session.

## Tools

The measurement scripts are in the session scratchpad and are not committed:
- the logging proxy (`proxy.py LISTEN TARGET OUTDIR`) and the replay (`replay.py URL KEY capdir
  ids [--notools] [--greedy]`);
- the captured requests.

Replaying a captured agent step gives both servers the same prompt. `--notools` sends
`tool_choice: "none"`, because dropping `tools` makes the history's calls invalid (400).
