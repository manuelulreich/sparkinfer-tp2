# Qwen3.8-27B VRAM budget per card — 2× RTX 5060 Ti (16 GB) vs the marketed 262,144-token claim

WP-1 record (dual-gpu/00-p0-probe). Every number below is measured on this box or computed from
the source at `runtime/src/models/qwen35.cpp`, `runtime/include/sparkinfer/kv_cache.h`,
`server/src/model_engine.cpp` and `runtime/examples/qwen3_gguf_config.h` — not from the plan's
ballpark figures.

## 1. Measured card capacity

| fact | value | source |
|---|---|---|
| `nvidia-smi` total per card | 16,311 MiB | `precheck_nvidia_smi.csv` |
| free + used per card (at every measurement) | 15,890 MiB | same CSV |
| **driver-reserved (total − free − used)** | **421 MiB** | measured, not assumed |
| **usable at idle per card** | **≈ 15,890 MiB** | |
| topology | PHB (PCIe + host bridge), **no NVLink**; PCIe link **Gen3, currently 8x** (max 16x) | `nvml_topo.txt`, `nvidia-smi -q` |
| utilization at every measurement this session | 92–96% (the LLM is resident on both cards) | CSV |

## 2. Fixed per-card deductions

| item | MiB/card | how obtained |
|---|---|---|
| **weights, TP-sharded (17.9 GB NVFP4 ÷ 2)** | **8,536** | 8.95 GB (decimal) = 8,536 MiB; Qwen3.8-27B total from config (64 layers: 16 full-attn [n_q 24, n_kv 4, head_dim 256, rope 64] + 48 GDN [lin_q 16, lin_v 48, head 128, conv 4, qkvdim 10,240], hidden 5,120, dense FFN, MoE FFN 17,408, vocab 248,320) |
| **DSpark drafter (replicated on both cards, CHG-0030 default)** | **931 – 1,397** | plan WP-12: "2–3 GB total, ≈ 1–1.5 GB/card when replicated" |
| **runtime overhead (CUDA contexts, NCCL internals, workspace, activations)** | **307** (0.3 GiB, stated assumption) | |

## 3. GDN recurrent state per concurrent sequence (re-verified from the code)

Per sequence: `lin_state` = **48** `gdn_state_slots` (`qwen35_prefill.h:25`, 64/interval-4) × 48
v-heads × 128 × 128 × **4 B (fp32)** = 150,994,944 B = **144.0 MiB**, plus `lin_conv_state` =
**64 layers** (not 48 — code quirk, matches the shared allocation at `qwen35.cpp:930`) × 3 ×
10,240 × **2 B (bf16)** = 3,932,160 B = **3.75 MiB** → **147.75 MiB/sequence (fp32)**.
Bf16-compacted (`lin_state_b16`, used by packed decode/spec) halves the state: **75.75 MiB/sequence**.
(+1.9 MiB per sequence for the two vocab-sized penalty/logit-bias tensors, 2 × 248,320 × 4 B,
if you want to count them.)

The plan/manifest's "~205 MB/sequence" is the **pre-compaction 64-slot layout**
(64 × 48 × 128² × 4 + conv = 205,258,752 B); the shipped code allocates the 48-slot layout
above (code comment at `qwen35.cpp:841`: "~151 MB"), so this table uses the 48-slot number.
Prefix-cached snapshots live in **pinned host** memory, not VRAM.

"Chosen max batch" anchor: `dflash_max_rows = 16` (`qwen35.cpp:857`) — the code's
concurrent-decode graph cap. Rows below: batch 1 / 8 / 16 (+ 32 as a stretch).

Per-sequence convention: **147.75 MiB fp32** (144.0 + 3.75) / **75.75 MiB bf16** (72.0 + 3.75);
the +1.9 MiB penalty/logit-bias tensors are *not* counted in the table rows (add 1.9 × batch if
you want them).

| batch | GDN fp32, MiB | GDN bf16, MiB |
|---|---|---|
| 1 | 148 | 76 |
| 8 | 1,182 | 606 |
| 16 | 2,364 | 1,212 |
| 32 (stretch) | 4,728 | 2,424 |

## 4. KV cache: the only remaining budget

Only the **16 full-attn layers** get KV slots (`hybrid_kv_layer_slots`, interval 4 → 16 of 64).
Per token, over those 16 layers (K + V, 4 kv-heads × 256 head-dim each = 2,048 elements/layer):

| KV dtype | B/token | note |
|---|---|---|
| **int8** (the code's default for the Qwen3.8 hybrid at max_seq ≥ 4,096) | **33,024** | 2,048 data B + 16 scale B (one fp16 scale per (token, kv-head)) per layer |
| bf16 (forced only by `SPARKINFER_KV_INT8=0`; Muse Glimmer is the natural bf16 case) | **65,536** | 2,048 × 2 B per layer |

Pool size at a given context (blocks = max_seq/16 + 8): **262,144 ⇒ int8 8.06 GiB (8,661,270,528 B) / bf16 16.0 GiB**; **131,072 ⇒ int8 4.04 GiB / bf16 8.01 GiB**.

## 5. The budget table (KV remainder per card, and the context it buys)

`15,890 − 8,536 (weights/2) − 307 (overhead) − drafter − GDN(batch) = KV remainder`. The base
before drafter/GDN is `15,890 − 8,536 − 307 = 7,047 MiB`. (MiB shown; implied context =
remainder ÷ 33,024 B/token.)

| batch | drafter 1 GB | drafter 1.5 GB |
|---|---|---|
| **1** (GDN 148) | 5,968 MiB ≈ 5.8 GiB → **≈189k tok** | 5,502 MiB ≈ 5.4 GiB → **≈175k tok** |
| **8** (GDN 1,182) | 4,934 MiB ≈ 4.8 GiB → **≈157k tok** | 4,468 MiB ≈ 4.4 GiB → **≈142k tok** |
| **16** (GDN 2,364) | 3,752 MiB ≈ 3.7 GiB → **≈119k tok** | 3,286 MiB ≈ 3.2 GiB → **≈104k tok** |
| **32** (GDN 4,728) | 1,388 MiB ≈ 1.4 GiB → **≈44k tok** | 922 MiB ≈ 0.9 GiB → **≈29k tok** |

Bf16-KV variant of the same table (÷2 the context): batch 1 → ≈95k / ≈88k tok; batch 8 → ≈78k / ≈71k; batch 16 → ≈60k / ≈52k.

Cross-checks against the two advertised contexts (int8, the code default):

* **262,144 tokens need an 8.06 GiB KV pool.** The most a card can ever hand the KV pool here is
  the AR-only, no-drafter, batch-1 remainder: `15,890 − 8,536 − 307 − 148 = 6,899 MiB ≈ 6.74 GiB`
  — **below 8.06 GiB, at every batch, with or without the DSpark drafter.** The marketed 262k does
  **not** fit on 16 GB/card. (It does fit on the single 32 GB card the README describes — 8.06 GiB
  KV + 17.9 GB weights + drafter + GDN all fit in 32 GiB, which is why README lines 33–34/51/77
  advertise it and why `model_engine` prints "131072 fits a 32 GB card".)
* **131,072 tokens need a 4.04 GiB pool (≈4,137 MiB).** That fits **with the DSpark drafter at
  batch 1** (5,968 MiB remainder → ~1.8 GiB headroom; 5,502 MiB → ~1.3 GiB) **and at batch 8**
  (4,934 / 4,468 MiB → ~0.78 / ~0.32 GiB headroom). At batch 16 it does not fit (3,752 / 3,286 MiB
  < 4,137) — there, the context must drop to 65,536 (2.02 GiB pool ≈ 2,052 MiB; 3,752 MiB fits
  with ~1.7 GiB headroom).

## 6. Verdict

* **G2: 16 GB/card does not hold the marketed 262k.** 262,144 is only reachable on the single
  32 GB card (or a 2×32 pair); the honest default for this pair is **`--ctx 131,072`** — the
  highest advertised context that still fits with the DSpark drafter at the code's real batch
  anchors (1 and 8), matching the single-card DSpark mode and the `model_engine` fit message.
  AR-only (no drafter) at batch 1 could stretch further (≈219k int8) but that is not the
  marketed configuration, so it is not recommended as a default.
* The PCIe Gen3 ×8 link (measured; see R7) is the transport ceiling for the TP allreduce — expect
  staging-bandwidth-limited numbers, not Gen4.

## 7. Addendum from the probe runs (NCCL heap, measured)

If a serving process ever keeps a 2-rank NCCL communicator alive, NCCL 2.32.3 (the only version
that can init on this pair) commits **≈160 MiB per card** of symmetric-VA heap at init
("Symmetric VA size=16GB" + committed pages) and **holds it until process exit** — still allocated
after `ncclCommDestroy` ([1]→[6] delta in `p2p_probe_run2–4.txt`; fully reclaimed at exit, the
post-run nvidia-smi returns to 1109 MiB free). Subtract from the KV remainder above if NCCL is
ever run in-process. Caveat for whoever does: on this pair 2.32.3's SHM/direct allreduce returned
silent +inf in the 1024-float smoke (3/3, no NCCL error) — see the WP-1 report, R8.
