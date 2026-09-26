# 0. Model identity — which Qwen3.8-27B this program optimizes

Recorded 2026-09-26 on explicit user instruction. **Every architecture fact,
weight, or budget in this program must be derived from the canonical model
below — never from the older revision in `~/Downloads/models/`.**

## Canonical (the optimization target)

- **Main model (27B):**
  `~/.cache/huggingface/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/5b7a687fc8211a5d631c8ca6a593dd37eb26ce33/`
  2 safetensors shards, **17,916,112,584 B on disk** (17.92 GB decimal; ÷2 =
  8,958,056,292 B = 8,545 MiB/card at tp=2). `config.json` `text_config`:
  64 layers (16 full-attn, interval 4) + 48 GDN; **24 q / 4 KV heads (6:1)**,
  head_dim 256, `partial_rotary_factor` 0.25 → rope_dim 64; GDN 16 q / 16 k /
  48 v heads × 128, conv kernel 4, qkvdim 10,240; hidden 5,120; intermediate
  17,408 (dense, one expert); vocab 248,320; **`mtp_num_hidden_layers: 0` —
  no MTP**.
- **DSpark drafter** (the WP-12 object — a *separate* model, not part of the
  27B):
  `~/.cache/huggingface/hub/models--gittensor-model-hub--Qwen3.8-27B-DSpark-NVFP4/snapshots/eba1ac5a66c74902eaa95a4000a7c5eda96d8e95/`
  `Qwen3DSparkModel`: 5 full-attention layers (40 q / 8 kv heads, head_dim 128),
  hidden 5,120, intermediate 10,240, yarn rope, `dflash_config`
  (markov_rank 256, target_layer_ids [4,16,28,40,52]); **1,399,670,058 B on
  disk** (1,335 MiB — inside G2's 931–1,397 MiB/card replicated budget).
  Both HF-cache trees are root-owned (read-only on this box) — never modify.

## Older revision — AVOID (not the optimization target)

- `~/Downloads/models/qwen38-nvfp4-0cc2795/` — an **older revision of the same
  27B**: identical `text_config` (24 q / 4 kv / 64 layers / same dims) except
  **`mtp_num_hidden_layers: 1` + `mtp_use_dedicated_embeddings`** (it still
  worked with MTP; the canonical revision does not — it uses the DSpark
  drafter above instead). 3 shards, 18,765,513,216 B on disk (~810 MiB more
  than the canonical: the MTP head + dedicated embeddings).
- Also in that directory, same caution: `qwen38-nvfp4-latest-tokenizer/`
  (tokenizer-only copy) and `dspark-nvfp4-vllm/` (a vLLM-format copy of the
  drafter; the canonical drafter is the HF-cache tree above).

## Verified consequences (checked against both `config.json` files, 2026-09-26)

- The **only** architectural difference between the two 27B revisions is the
  MTP block (present in the old one, absent in the canonical one). The
  in-tree model code (`qwen35.cpp` path; `qwen38_hf_config.h` parses the
  model's own `config.json` `text_config` — model-agnostic, no MTP fields) is
  correct for the canonical model as-is. If an MTP-era checkpoint is ever
  loaded, its MTP tensors fall into `tp_layout`'s out-of-pattern fallback
  (replicated + one warning per name) — safe behaviour, not a layout gap.
- **The G2 budget table (`00-p0-probe/budget_table.md`) was measured against
  the canonical model** (its 17.9 GB decimal total and its 4 KV heads match
  `5b7a…`, not `0cc2795`): it stands as the budget record.
- **The "8 KV heads / 3:1" wording that appeared in 01/03 was wrong**
  (corrected in those files 2026-09-26): the 27B is 24 q / 4 KV (6:1) and the
  GDN ratio is 16 q : 48 v (3 v per q/k head). At tp=2: full-attn KV
  **2+2 heads** per device (G2's 33,024 B/token KV arithmetic, built on
  4 kv-heads, is correct and stands); GDN v-heads 24+24, GDN q/k 8+8 per card.
