# 11. Swift 1.5 (Qwen3.8-27B fine-tune) as an alternative target

Written 2026-10-05.

## Status (2026-10-06): Path A done, `PRESET=swift ./startup.sh`

- **Conversion.** Output in `~/models/Swift-1.5-Qwen3.8-27b-NVFP4-sparkinfer` (17.9 GB, 19 min on
  3 cores).
  - Inputs:
    - the NVFP4 export (manifest sha256 OK);
    - the BF16 originals of the 208 FP8 Linears, fetched by range from
      `ukisai/Swift-1.5-Qwen3.8-27b@b4c84d42`.
  - The tensor inventory (2,387 names, dtypes, shapes) is identical to the served Qwen3.8
    checkpoint.
- **Quantizer.**
  - The tensor-wide scale follows ModelOpt's `amax/(6*448)` and matches it exactly
    (`--selftest`).
  - ModelOpt does not use a plain `block_amax/6` block scale (only ~30 % of its block bytes
    match it). It picks lower-error scales: rel RMS 0.0847 for its bytes, 0.095 for the plain
    scale.
  - `quant_nvfp4` searches ±4 e4m3 codes per block: 0.0833 on all 208 tensors.
- **Memory.** Identical to Qwen3.8: 456k-token pool, 4 stream slots, 216 MiB free per card.
- **Gates** (`tp2_gates.py`, DFlash2): lossless 7/7, determinism and batching exact, no perf
  regression against `baseline_2x5060ti.json`.
- **Quality and speed against Qwen3.8 on the same build:**

  | | Swift | Qwen3.8 |
  |---|---|---|
  | perplexity (ppl.py) | 4.51 | 4.46 |
  | greedy C1 / C4 tok/s | 107 / ~315 | 105 / ~302 |
  | DFlash2 tokens/step: sampled C1, C4; greedy C4 | 2.61, 2.85; 3.08 | 2.39, 2.68; 2.75 |
  | opencode final step (cap_e2e3 007, 4 seeds): tokens / s / tok/s | 1,504 / 13.3 / 115 | 1,855 / 15.9 / 118 |
  | 12 easy tasks x2: correct, mean reasoning | 24/24, 172 | 24/24, 192 |
  | 6 hard tasks x2 (xhigh): correct, mean / median reasoning | 11/12, 2,407 / 849 | 12/12, 2,946 / 2,681 |
  | same at low | 12/12, 2,740 / 874 | 12/12, 3,573 / 1,273 |

  - The drafter, trained on base Qwen3.8, accepts as well or better on Swift.
  - One sampled C4 run ended its group halfway on the gain guard; a rerun did not. Sampled runs
    vary.
- **Template (S4)**, checked by a read-only comparison of Swift's `chat_template.jinja` with
  the server's renderer.
  - **Default effort, thinking off, user/system/tools framing:** identical.
  - **Effort text:** low and medium were wrong for both models (every effort got the xhigh
    sentence). Fixed in the server; see the CHANGELOG.
  - **Remaining differences.** All of them apply equally to the current model's own template:
    - tool JSON key order;
    - tool-call argument order (sorted);
    - Swift's upstream template emits an empty `<think>\n\n</think>` for a replayed assistant
      turn without reasoning, where the served template, and so the server, emits none.
- **Not done.**
  - **No absolute reference.** There is no KL against a Swift GGUF reference: it would need a
    llama.cpp build plus a 24–29 GB download.
  - **Quantization quality** is argued from the per-tensor error instead. It is the same
    recipe class as the served checkpoint, with a lower error than ModelOpt's own.
  - **Path B** was not needed.

Candidate: [`ukisai/Swift-1.5-Qwen3.8-27b-NVFP4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4)
(revision `25482027debd5485e8108897ba9fed8d3ba16595`, 21.9 GB).

## What the model is

- **What it is.** An RL + on-policy-distillation fine-tune of Qwen3.8-27B by UkisAI. The vendor
  reports 25–58 % fewer thinking tokens than the base at about the same scores:
  - LiveCodeBench v6: 76.8 → 81.7 %.
  - Terminal-Bench 2.1: 69.2 → 72.1 %, with mean reasoning tokens −16 % and the median unchanged.
- **Architecture.** Unchanged from what we serve (`Qwen3_5ForConditionalGeneration`):
  - 64 layers: 48 Gated-DeltaNet and 16 attention (one in four).
  - hidden 5120, 24 query heads, 4 KV heads, head_dim 256, FFN 17408, vocab 248320.
  - Same eos ids and the same sampling defaults (T 1.0, top_k 20, top_p 0.95).
  - It adds one bf16 MTP layer, which we would not use.
- **Tokenizer.** `tokenizer.json` is byte-identical to the current checkpoint (sha256 `0997f410…`).
- **Quantization (ModelOpt 0.47, `MIXED_PRECISION`).** This is the part that matters:

  | tensors | Swift | current (`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`) |
  |---|---|---|
  | FFN gate/up/down (192) | NVFP4 g16 | NVFP4 g16 |
  | GDN in_proj_qkv / in_proj_z / out_proj (144) | **FP8, per-tensor scale** | NVFP4 g16 |
  | attention q/k/v/o (64) | **FP8, per-tensor scale** | NVFP4 g16 |
  | lm_head | NVFP4 g16 | bf16 |
  | embed, norms, conv1d, in_proj_a/b, vision | bf16 | bf16 |

  FP8 here means an `F8_E4M3` weight, an F32 scalar `weight_scale` and an F32 scalar
  `input_scale`.
- **No KV scales.** `kv_cache_quant_algo` is null. Our nvfp4 KV computes its own block scales,
  so nothing is lost.
- **Vendor caveat.** The vendor states that end-to-end generation of this NVFP4 export was **not
  tested**. Their evaluations are BF16 and INT4.
- **License: Swift Open License 1.0.** Free for any use up to US$1M annual revenue; above that a
  commercial licence is required.

## Why it does not load today

- **The loader picks each weight's path from the tensors present** (`load_compressed_tensors`,
  runtime/src/models/qwen35.cpp).
  - An FP8 weight goes to `keep_fp8` / `dequant_fp8` / `ct_fp8_slice`.
  - All three expect the compressed-tensors layout: a **BF16 `weight_scale` with one value per
    row** (`sc->n_values != rows` is rejected).
  - ModelOpt's FP8 has one F32 scale per tensor, so the first GDN projection fails as
    "malformed".
- **Path B would also cost memory and speed.**
  - **Memory.** 144 GDN Linears stay FP8 (1 byte a weight instead of 0.5625): about +2.4 GB of
    weights, +1.2 GB a card.
  - **Prefill scratch.** `fp4_only` in prefill turns off, so the int8 scratch comes back (about
    +0.3–0.6 GB a card).
  - **KV pool.** Together, roughly 150–180k fewer pool tokens: about 280–300k instead of 456k.
  - **Decode.** It is bound by the weights each card reads per token, about 7.2 GB today at a
    15.6 ms GEMM floor. +1.2 GB is about +2.7 ms a step, i.e. **~12–15 % slower decode**
    (estimate).
  - **Attention.** The current policy requantizes non-NVFP4 attention weights to Q4_K. That is
    a second quantization; it failed the FP8 → Q4_K gate on GDN (KL 0.035).

## Plan

Recommended: **Path A**, convert offline to the exact layout of the current checkpoint.

- The loader, prefill (`fp4_only`), tp slicing, KV pool size (456k) and DFlash2 integration then
  stay as they are.
- No runtime code changes; only a conversion tool.
- **Path B** (native mixed FP8 support) is the fallback if Path A's quality check fails.

### S0. Fetch (~22 GB, ~10 min)

- Download the NVFP4 repo at the pinned revision into the HF cache. There are 511 GB free.
- Record the sha256 of each shard against `UPLOAD_MANIFEST.json`.

### S1. Conversion tool `dual-gpu/tools/swift_to_nvfp4.py` (CPU, streaming tensor by tensor)

- **Copied unchanged:** FFN NVFP4 (`.weight` U8, `.weight_scale` E4M3, `.weight_scale_2` F32,
  `.input_scale`), lm_head NVFP4, embed, norms, GDN small tensors (`A_log`, `dt_bias`,
  `conv1d`, `in_proj_a/b`, `norm`), vision.
- **Dropped:** `mtp.*`. The config gets `mtp_num_hidden_layers: 0`, as in the current checkpoint.
- **Requantized:** the 208 FP8 Linears to ModelOpt NVFP4:
  - global `weight_scale_2 = amax / (6·448)`;
  - per-16 `weight_scale = e4m3(block_amax / 6 / global)`;
  - codes RTN to e2m1, packed two per byte, low nibble first, as in the current checkpoint;
  - `input_scale` copied (the runtime never reads it).
- **Source for the 208 tensors:**
  - **Preferred: the BF16 originals** from `ukisai/Swift-1.5-Qwen3.8-27b` at `5ad04445…`. Only
    those tensors, fetched by HTTP range requests against the safetensors headers: about 7.2B
    parameters, ~14.4 GB, instead of 54 GB for all 18 shards.
  - **Fallback: dequantized FP8** (`w·scale`). FP8's error is small next to FP4's grid, but it is
    a double quantization; measure both in S2.
- **Layout self-test.** Dequantize a current-checkpoint NVFP4 tensor, run it through the
  quantizer, and require the original bytes back (idempotence). This catches nibble order,
  scale direction (`weight_scale_2` is ModelOpt's inverted form, see `nvfp4_src`) and block
  layout before a GPU is involved.
- **Output.**
  - `config.json`'s `quantization_config` is modelled on the current one (`quant_algo: NVFP4`,
    the same ignore list), plus `hf_quant_config.json`.
  - About 17 GB, written next to the HF cache.
- **Effort:** small–medium (a day). CPU-bound, so threads stay ≤ 4.

### S2. Load and quality check (needs the GPUs, ~15 min)

- Load with `MODEL=<converted> ./startup.sh` on a spare port.
  - The `stream slots` log line should report the same headroom as today; that confirms the
    layout matches.
  - `lm_head` arrives as NVFP4 instead of bf16, so check which head path the loader takes
    (`SPARKINFER_Q38_HEAD_NVFP4`). It must not requantize NVFP4 to Q4_K.
- Teacher-forced KL / top-1 on `eval_text.txt` (same procedure as the loader's GDN comment):
  - Reference: `ukisai/Swift-1.5-Qwen3.8-27B-GGUF` at Q8_0 or better.
  - Compare the BF16-sourced and FP8-sourced conversions.
  - **Bar:** KL against the reference no worse than the current checkpoint's KL against its
    own Q4_K_M/Q8 reference, plus about 0.02.
- Run `dual-gpu/gates/tp2_gates.py` with the model swapped (lossless spec, deterministic).

### S3. DFlash2 with Swift (speed only; the output stays lossless)

- **Why acceptance may drop.** The drafter (`~/models/Qwen3.8-27B-DFlash2`, target layers
  5/19/33/47/61) was trained on the **base** model's hidden states. Swift is RL-tuned from that
  base, so its hidden states and next-token choices have drifted.
- **The output stays exact.** Verify guarantees it, so only tokens per step can change.
- **Measure** tokens/step and tok/s on the same three short runs as plan 10:
  - the 20k opencode final answer,
  - the 60k replay (256 tokens),
  - the `c4.py` cohort.
- **Decide:**
  - Within about 10 % of the base: done.
  - Worse: retune `SPARKINFER_SPEC_GROUP_DEPTH` / `LONG_DEPTH`, since a lower acceptance favours
    shallower blocks.
  - Much worse (< 2.5 tokens/step at 20k): a Swift-specific drafter needs training, which is
    out of scope here.
- Swift's own MTP layer is a possible drafter, but we have no MTP draft path. It is not part of
  this plan.

### S4. Chat template

- **The server does not read `chat_template.jinja`.** It renders a built-in Qwen3.8 template
  (server/src/chat_tokenizer.cpp, chat_tools.cpp).
- Swift ships the upstream Qwen3.8 template:
  - `reasoning_effort` xhigh (default) / medium / low;
  - `preserve_thinking`;
  - XML tool calls.
- **Check** that the built-in rendering is token-identical to Swift's jinja for:
  - tools,
  - thinking on/off,
  - each effort level,
  - a multi-turn agent history.
- Render the jinja with Python and compare the token ids from the server's tokenizer.
- **Effort defaults.** Swift's numbers assume `xhigh` by default. Note what the server defaults
  to when a client sends no effort.

### S5. Serving

- **`startup.sh`.** A `MODEL` preset (e.g. `PRESET=swift`) that points at the converted
  checkpoint. The rest is unchanged: same CTX, the 456k pool (re-measured on the first load),
  4 stream slots, DFlash2.
- **opencode.** A second model entry under the `sparkinfer` provider (same base URL, `modelID`
  as served), so the two can be switched per session. Only one model is loaded at a time: the
  two do not fit side by side.

### S6. End-to-end comparison (the decision)

- Run the same opencode task on both models, one after the other.
- **Compare:** wall time, completion and reasoning tokens (from `usage`), tok/s, tokens/step,
  and whether the task succeeds.
- **Expected:** the thinking-token reduction outweighs any lost acceptance. Agent-task gains are
  the smallest the vendor reports (−16 % mean reasoning tokens), so this is where it has to
  show.

## Path B (fallback): native mixed NVFP4/FP8

Only if S2 shows the requantized GDN/attention losing measurable quality.

- **Loader.** Accept ModelOpt FP8 in `keep_fp8`, `dequant_fp8` and `ct_fp8_slice`: an F32
  scalar `weight_scale`, broadcast per row.
  - The packed form stores a **bf16** row scale, so a per-tensor F32 scale would be rounded
    (up to ~0.4 %, the same for the whole tensor).
  - Better: a per-tensor F32 variant of the `SI_QTYPE_FP8` payload header.
- **Attention.** Keep it native FP8 instead of the Q4_K requantization, which would be a second
  quantization.
- **Cost.** About −170k KV pool tokens and ~12–15 % slower decode (see above). Prefill also
  loses the `fp4_only` scratch saving.
- **Effort:** medium.

## Order and effort

| step | needs GPUs | effort |
|---|---|---|
| S0 fetch | no | small |
| S1 conversion + self-test | no | small–medium |
| S2 quality + gates | yes, ~15 min | small |
| S3 DFlash2 acceptance | yes, ~10 min | small |
| S4 template check | no (tokenizer only) | small |
| S5 startup/opencode | yes, one reload | small |
| S6 opencode comparison | yes, ~1 h of real use | small |
