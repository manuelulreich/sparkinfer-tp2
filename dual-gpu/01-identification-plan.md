# Dual-GPU (tp=2, P2P) — identification plan

**Status:** active · **Created:** 2026-09-25 · **Companion record:** [`02-change-manifest.json`](02-change-manifest.json) (source of truth) / [`02-change-manifest.md`](02-change-manifest.md) (rendered view) · **Next step:** `03-implementation-plan.md`

---

## 1. Purpose and scope

This document answers two questions only:

1. **How do we identify ALL changes** that make sparkinfer run on two RTX 5060 Ti cards with tensor-parallel `tp=2` over P2P (no NCCL, no new dependencies — the 2.5 MB / no-Python-stack promise is a hard constraint)?
2. **How do we record those changes** so that the *next* step can turn them into a final implementation plan (`03-implementation-plan.md`) without re-deriving anything?

It is **not** the implementation plan. It defines no code, no APIs, and no patch order. The record it produces is the append-only change manifest (`02-change-manifest.json`), currently **2,100 items**: 34 hand-audited (CHG-0001…0034) + 2,066 mechanical scan candidates from `scan.py`.

**Target.** Qwen3.8-27B NVFP4 (17.9 GB) on 2 × RTX 5060 Ti (16 GB, sm_120, 48 SM, PCIe — **no NVLink**). The checkpoint does not fit one card, so `tp=2` is a memory necessity, not an optimization. Qwen3.6-35B-A3B (256 experts) is the second model the design must keep working for, at no cost beyond a decision item.

**Non-negotiable invariants** (each has a manifest item enforcing it):

- `tp=1` (the default, single-GPU path) stays **byte-identical**: same flags, same weights, same tokens/logits as today. Nothing in this program may loosen that for existing users.
- The **DSpark byte-lossless gate** (drafter vs autoregressive, 100 % token equality) is preserved as a property — re-proven *inside* `tp=2` (both paths run under the same split), which is the correct invariant.
- The **serving API surface** (`/v1/*`, `/metrics`, `/v1/info`) is preserved and only extended.

## 2. What we know about the hardware (and what we don't)

| Fact | Value | Source / status |
|---|---|---|
| GPUs | 2 × RTX 5060 Ti, sm_120 | known |
| VRAM | 16 GB each | known |
| SMs | 48 per card (vs 170 on the 5090 the kernels were tuned on) | known — drives R4 |
| Memory bandwidth | GDDR7 ≈ 448 GB/s per card | datasheet |
| Interconnect | PCIe; **P2P peer access UNVERIFIED** (consumer pair, no NVLink) | **open — R1, decided in P0** |
| Effective PCIe transfer | ≈ 25 GB/s (link + topology dependent) | estimate — P0 measures |
| Weights | 17.9 GB total NVFP4 → ≈ 8.95 GB per card at tp=2, before KV/drafter/state | known — drives R3 |
| GDN state | fp32 `lin_state` + bf16 conv state, per **value-head**; pinned-host prefix snapshots ≈ 205 MB/sequence on 27B | known |

**Open questions that are design inputs, not trivia:**

- **O1 (→ R1):** can these two cards do `cudaDeviceCanAccessPeer` at all? Consumer GeForce pairs without NVLink are *not guaranteed* P2P; it depends on BIOS/PCIe routing/ACS.
- **O2 (→ R7):** even where P2P exists, is the path a direct peer DMA or is it routed through the root complex (ACS/IOMMU), making it far slower than the link spec?
- **O3 (→ R3):** does 16 GB/card hold weights/2 + drafter + GDN state + the KV pool the operator wants? The 262k-context claim may have to be re-stated for 2×16 GB.

**Model facts used throughout (Qwen3.8-27B, per `Qwen35Config`):** 64 layers = 16 full-attention (GQA, **8 KV heads**) + 48 Gated-DeltaNet (48 value-heads per layer; q-heads = 3 per v-head, `gdn_qh_block`); `dense_ffn` (`n_experts=1`, `top_k=1`) — the FFN is one large SwiGLU, not an expert bank; 35B-A3B would carry 256 experts (→ 128/128 split). Exact hidden dim, FFN width, head dim, and full-attn q-head count are read from the config in P3 — they are deliberately *not* assumed here.

## 3. How changes are recorded (the record-keeping contract)

The record is the **manifest**, and these rules are what make it trustworthy at the end of the program:

1. **Single source of truth.** `02-change-manifest.json` is the record. `02-change-manifest.md` is a rendered view produced by `render.py` — never hand-edit the MD; edit the JSON, re-run `render.py`.
2. **Append-only.** Items are never deleted, renumbered, or re-used. If a site disappears from the source (refactored away), its item stays and a human closes it as `rejected` with a note saying why. This keeps the whole program auditable end to end.
3. **IDs.** `CHG-####`, monotonically increasing; the next free id is always `max + 1` (both `scan.py` and any manual edit follow this).
4. **Lifecycle.** `candidate → confirmed → planned → implemented → verified`, plus the terminal `rejected` ("audited, no change needed — note MUST say why"). Only `confirmed`-or-better items with a non-null `change` are raw material for the next step; only `planned/implemented/verified` items flow into the final plan's PR decomposition (`meta.consumed_by`).
5. **Two sources, one truth per site.** `source: manual` (hand-audited, higher confidence) vs `source: scan` (mechanical match from `scan.py`). A `(file, line)` is never double-filed: if a manual item owns a site, the scanner skips it. Manual wins on conflict.
6. **`change: null` means "not designed yet".** That is not a defect — it is the explicit division of labor: *this* step identifies and audits; P3 (below) designs; the *next* step sequences.
7. **Severity** (`blocker` = tp=2 cannot work at all without it · `major` = needed for a complete, correct, shippable tp=2 · `minor` = quality/ops/docs): assigned to all manual items; scan candidates get one when they are promoted in P2.
8. **Mechanical re-sweep is repeatable.** `python3 dual-gpu/scan.py` (from anywhere; it resolves the repo root as the parent of `dual-gpu/`) merges new sites as `candidate`s; `--dry-run` previews, `--report` summarizes. Re-run it after any code change to catch drift. A zero-file scan is a hard error, so the old CWD-path bug cannot fail silently again.
9. **Raw evidence is stored, not summarized.** P0 probe output, `nvidia-smi` dumps, and budget spreadsheets live in `dual-gpu/00-p0-probe/` as-is; manifest items point at them instead of paraphrasing.

## 4. The phases (P0 → P5)

Phases run in order except P1, which is independent and may overlap P0. Each phase has an **entry** precondition, a **work** list, and **exit criteria** that must hold before the next phase starts. All work is recorded into the manifest as it happens — a phase is done when its manifest state is, not when a memo is written.

### P0 — Hardware & P2P probe (the branch point)

*Entry:* plan accepted; a dual-5060-Ti box (or remote shell to one) is available.

*Work:*

- Write and run **`runtime/examples/p2p_probe.cpp`** (CHG-0006) — the first code of the whole program. It reports: device count/names/SM/VRAM; `cudaDeviceCanAccessPeer` **both directions**; timed `cudaMemcpyAsync` D2D both ways at several sizes (128 KB…1 GB); the same copy via pinned-host staging; `nvidia-smi topo -m` captured to file.
- Produce the **per-card budget table** (feeds CHG-0020): 16 GB − weights/2 − drafter − GDN state at max batch − overhead = KV remainder; state how much context that buys vs the single-32 GB card's 262k claim.
- Store all raw output in `dual-gpu/00-p0-probe/`.

*Exit:*

- R1 answered: `cudaDeviceCanAccessPeer` true/false each way → **the all-reduce transport is chosen** (P2P mapped-memory fast path vs pinned-host staging; the public API is the same in both cases — CHG-0005).
- R7 answered: measured D2D bandwidth (peer and staged) recorded, with the root-complex caveat noted.
- The R3 budget table is recorded in CHG-0020's notes, with a verdict on the default `--ctx`.

### P1 — Mechanical sweep triage

*Entry:* none (runs in parallel with P0; the scan has already been run once — see the manifest's `meta.last_scan`).

*Work:*

- Triage the **2,066 scan candidates** (200 C++ files, 84 tooling files) in batches, **by file, then by category**: `kernels/stream-graph` (531) and `runtime/memory-pool` (517) are the big masses and are overwhelmingly *call sites that take a slice pointer and need no code change* — those get batch-`rejected` with a shared one-line reason ("device-agnostic call site; receives per-device slice from the split map — verified in CHG-0029 audit").
- Anything that is *not* that gets kept as a live `candidate` for P2, or merged into the matching manual item (dedupe: one owner per site).

*Exit:*

- No scan candidate is left undecided; every one is `rejected` (with reason) or a live candidate tied to a P2 checklist line; `meta.last_scan` reflects the final run.
- The surviving candidate set is small enough to audit item-by-item (target: ≤ 100 live candidates before P2).

### P2 — Subsystem deep-dives (14 subsystems)

*Entry:* P0 done (the comm path is known — several subsystems' checklists branch on it).

*Work:* walk the 14 subsystem checklists in the manifest (`device-init`, `kv-cache`, `gdn-state`, `moe`, `weight-loading`, `decode-path`, `prefill-path`, `engine-serving`, `observability`, `server-config`, `tooling-ci`, `docs`, `kernels`, `runtime-tests`). For each: read the code; correct or confirm each owned item; add items the checklist missed; promote `candidate → confirmed` (with a code-level `current` description) or `→ rejected` (with reason); assign severity; flip the subsystem status to `audited`. The `kernels` audit (CHG-0029) produces a **per-kernel verdict list** (portable / retune-for-48-SM / rewrite) as sub-items.

*Exit:*

- No subsystem `pending-audit`; no item `candidate` without an explicit decision recorded.
- Every item has a severity; every `confirmed` item's `current` describes the code as it actually is (auditor can re-read the file and see the same thing).

### P3 — Tensor-ownership map + communication protocol design

*Entry:* P2 done. This is the design phase: it fills **every** null `change` field.

*Work:*

1. **Per-tensor ownership map** — Appendix A below is the seed (the proposed default for Qwen3.8-27B). Finalize it against the real `Qwen35Config` numbers: per layer, which tensor, on which device, split along which axis, per-device shape. This map is the single input all three weight loaders (CHG-0011) and every GEMM site consume.
2. **Communication protocol** — the layer-boundary all-reduce (one per layer, fused residual; see Appendix A) and the lm_head max-reduce: exact call sites, event ordering, buffer layout, and the graph representation. **Decision the phase must make:** (i) two per-session graphs (one per device), event-synced at each exchange — always valid; vs (ii) any single-graph formulation with cross-device memcpy nodes — only possible if peer access is enabled **at capture time** and only if a graph in one context may contain a memcpy node into the peer's memory (verify empirically; R5).
3. **Placement decisions** with recorded defaults: DSpark drafter (CHG-0030 — proposed: replicate on both), vision tower (CHG-0031 — proposed: run on card 0, broadcast embeddings), MoE (CHG-0010 — 27B dense FFN split; 35B-A3B experts 128/128), lmcache sidecar (CHG-0017 — proposed: defer to v2, reject tp=2+sidecar at load), thermal/failure policy (CHG-0019/0034 — proposed: max-temp pacing; any fatal GPU error downgrades the whole server).

*Exit:*

- Zero `change: null` among `confirmed` items.
- Appendix A is **arithmetically verified**: per-card byte budget ≤ 16 GB − reserve, with the numbers, and the head-count splits (8→4/4 KV, 48→24/24 v-heads) match the config exactly.

### P4 — Verification-matrix design

*Entry:* P3 done.

*Work:* for every `confirmed` item, specify the verification vehicle concretely: which test/script/eval gate, how to run it, pass criteria. Define the **tp=2 eval gates** (CHG-0028): re-baseline rather than loosen — DSpark lossless re-proven *inside* tp=2, token-match vs llama.cpp `--tensor-split 2,2`, no-regression tiers measured on the dual box with both cards clock-pinned (CHG-0023). State the **tp=1 byte-identical contract**: any PR must show the tp=1 path is numerically untouched (the existing eval-bot token-match vs llama.cpp on the 5090 stays as-is and must keep passing). New tests to design: P2P copy, 2-GPU all-reduce vs host reference (bf16), cross-context graph capture, per-device test-harness mode (CHG-0022: `SPARKINFER_TEST_DEVICE=n`).

*Exit:* every confirmed item's `verification` field is concrete and runnable; the tp=1 invariance contract is written down as the program's standing gate.

### P5 — Fold into the implementation plan (the next step)

*Entry:* all of P0–P4 done, i.e. the Definition of Done below is met.

*Work:* author `03-implementation-plan.md`: decompose the manifest into an **ordered PR sequence** respecting `depends_on` (the comm layer and layout map come first — everything else depends on them; graphs and DSpark come later), attach each PR its verification from P4 plus the standing tp=1 gate, and carry the one **governance ask** (CHG-0024: where does tp=2 get scored) to the org.

*Exit:* plan reviewed and accepted; manifest items move to `planned` as they are assigned to PRs; `render.py` re-run so the MD matches.

## 5. Risk register

| # | Risk | How detected | Mitigation / branch | Items |
|---|---|---|---|---|
| **R1** (top) | **No CUDA peer access at all** on the consumer pair (no NVLink; platform-dependent: BIOS, PCIe routing, ACS). Then the only path is pinned-host staging: every all-reduce doubles the bytes across PCIe (≈ 25 GB/s effective vs 448 GB/s GDDR7). | P0 probe: `cudaDeviceCanAccessPeer` both ways. | Design the all-reduce with **one API over two transports** (P2P mapped-memory fast path; pinned-staging fallback), decided in P0 before any layout work. If even staging fails, `tp=2` is rejected at startup with a clear diagnostic. | CHG-0027, 0006, 0005 |
| **R2** | **Numerics drift.** Split GEMM + bf16 all-reduce of partial sums rounds differently than one card's reference; logits move, token match vs the 5090 baseline can fall off. | Measured on the dual box in P4 (logit deltas, token-match rate, DSpark τ). | **Re-baseline the tp=2 gates; never loosen the tp=1 gates** (tp=1 stays bit-identical). The DSpark gate compares AR vs DSpark *under the same configuration*, so re-proving it inside tp=2 is the right invariant. | CHG-0028, 0015 |
| **R3** | **16 GB/card budget.** 8.95 GB weights/2 + drafter + GDN state + KV must fit; the 262k-context headline may not. | P0 budget table (per card, per component, at max batch). | Record the per-card table as a sizing fact; if KV is short, lower the default `--ctx` **with a message** rather than silently shrinking. | CHG-0020, 0007 |
| **R4** | **Kernel constants tuned for a 170-SM 5090.** `n_splits` tiers, occupancy targets, chunk/split-K sizes assume 5× the SMs we have (48). | CHG-0029 audit (per-kernel verdict) + microbenchmarks in P4. | Retune/parameterize the flagged kernels; the audit output lands in the manifest as sub-items, so the next step sees them. | CHG-0029 |
| **R5** | **Cross-context CUDA graph rules.** A context is one device; a single graph cannot span two contexts. A P2P memcpy node in a graph needs peer access **enabled at capture time**; and whether a graph on device A may contain a memcpy node into device B's memory at all is an empirical question. The existing legacy-stream capture-poisoning hazard is *per-context* — two contexts actually **weaken** it, but the invariants must be re-stated. | Empirical capture test in P4 (cross-context graph; P2P-mapped node; staging-memcpy node). | Default design: **two per-session graphs, event-synced** (always valid, staging or P2P). P3 may upgrade to graph-embedded P2P copies if the capture test passes and R1/R7 allow. | CHG-0013, 0005, 0012 |
| **R6** | **Eval-bot / CI governance.** The whole quality machinery (token-match vs llama.cpp, DSpark lossless, no-regression tiers, the `rtx5090-required` PR guard) is pinned to one 5090. tp=2 results currently have no home in it. | Human/org decision — this is the only risk that is *not* technical. | Three options on the table (CHG-0024): (a) new dual-5060-Ti eval target with its own label tier, (b) tp=2 as an arm on existing single-GPU boxes (two 5090s if the org can supply them), (c) tp=2 stays a release-level feature validated by a manual runbook until (a) exists. Decide before P5; the PR guard for tp code paths is defined whichever is chosen. | CHG-0024, 0028, 0023 |
| **R7** | **P2P exists but is slow.** Path routed through the root complex (ACS/IOMMU), or PCIe-generation-limited: effective bandwidth far below link spec, so per-layer all-reduce cost may matter on 48-SM cards even though the messages are tiny. | P0: measured D2D (peer and staged) vs GDDR7/local-link reference; `nvidia-smi topo -m` recorded. | Set performance expectations from **measured** P0 numbers, not datasheet numbers; the perf model in the next step uses the P0 table. The two-transport design (R1) already covers the cost case. | CHG-0006, 0027, 0005 |

## 6. Definition of done (handoff to the next step)

`03-implementation-plan.md` starts **only** when all of the following hold:

1. **Every manifest item is decided** — no `candidate` remains without an explicit verdict (`confirmed` or `rejected` with reason).
2. **Every `confirmed` item has a non-null `change`** (the design is recorded) and a severity.
3. **All 14 subsystems are `audited`** (no `pending-audit`).
4. **The P0 probe report is committed** in `dual-gpu/00-p0-probe/`; R1 and R7 are answered with measured numbers; the all-reduce transport is chosen.
5. **Appendix A is complete and arithmetically verified** — per-card byte budget ≤ 16 GB − reserve, head splits exact (4/4 KV, 24/24 v-heads), and it matches the real `Qwen35Config`.
6. **The verification matrix covers every confirmed item**, and the **tp=1 byte-identical contract** is stated as the standing gate for every PR.
7. **The tp=2 eval-gate question is decided or explicitly deferred** (CHG-0024/0028) — the next step must know whether its PR sequence is bot-scored, arm-scored, or runbook-validated.

## Appendix A — Proposed default TP=2 layout, Qwen3.8-27B

*Seed for P3 (to be finalized against `Qwen35Config`; hidden dim `H`, FFN width `F`, head dim `d_h`, full-attn q-head count `n_q` are read from the config, not assumed). Convention: **device A = KV-head group 0–3 / v-heads 0–23; device B = KV-head group 4–7 / v-heads 24–47.** Replicated tensors live on both cards and are never re-broadcast within a step.*

| Tensor (per occurrence) | Split axis | A holds | B holds | Notes |
|---|---|---|---|---|
| `embed_tokens` [V, H] | — (replicated) | all | all | shared with the drafter; input lookup only |
| Layer norms (RMSNorm [H]) | — (replicated) | all | all | elementwise over the full hidden state, which is identical on both devices |
| **Full-attn (16 layers)** | | | | |
| `q_proj` [H, n_q·d_h], `k_proj`/`v_proj` [H, 8·d_h] | **columns, by KV-head group** | cols of KV group 0–3 (and their q-heads) | cols of KV group 4–7 (and their q-heads) | q-heads travel with their KV head so a head never straddles cards |
| `o_proj` [n_q·d_h, H] | **rows, by q-head group** | rows for its q-heads → partial [·, H] | the other rows → partial [·, H] | summed at the layer-end all-reduce (below) |
| **GDN (48 layers)** | | | | |
| GDN q/k/v/gate projections | **columns, by v-head block** (`gdn_qh_block`: 3 q + 1 v per block) | v-heads 0–23 blocks | v-heads 24–47 blocks | output projection row-split like `o_proj` |
| `lin_state` (fp32), conv state (bf16) | **per v-head** | v-heads 0–23 | v-heads 24–47 | the clean split: no cross-head coupling in the GDN update — *verified in P2* (CHG-0009) |
| **Dense FFN (all 64 layers, `n_experts=1`)** | | | | |
| `gate`/`up` [H, F] | **columns** | F/2 | F/2 | SwiGLU pair stays in lockstep per device |
| `down` [F, H] | **rows** | F/2 rows → partial [·, H] | F/2 rows → partial [·, H] | fused into the layer-end all-reduce |
| `lm_head` [H, V] | **rows, by vocab half** | V/2 rows → logits [·, V/2] | V/2 rows → logits [·, V/2] | **not** a full all-reduce: greedy = max of the two local maxima; sampling = Gumbel-max per device, then one cross-device compare |
| **KV cache (the 16 full-attn layers only)** | **KV heads** | KV-heads 0–3 per {K,V}, pool + int8 scale pool + windowed ring + block table | KV-heads 4–7, same structure | one logical block = a pair of physical blocks (one per pool); prefix refcounts per pool; admission capacity = min over pools |
| **DSpark drafter (5 layers)** | — (replicated, default) | full copy | full copy | CHG-0030(b): it is small (5 layers); gated on the R3 budget table; shared embed/lm_head already replicated |
| **Vision tower** | — (single home, default) | full tower (card A) | — | embeddings [n_img, H] P2P-broadcast once per image batch (CHG-0031) |
| **MoE router / experts** | n/a on 27B (dense FFN above) | — | — | 35B-A3B only: experts 0–127 on A, 128–255 on B; router replicated or broadcast (CHG-0010) |

**Per-layer communication pattern.** Every layer *starts* with a full hidden state that is bit-identical on both devices (that is the invariant the layer-end reduction maintains). Inside a layer, everything is column-split (no communication) until the row-split output(s) (`o_proj` or GDN out-proj, plus `down`) produce **partial sums** of the same full vector. Those partials — together with the (identical-on-both-cards) residual — are combined in **one fused all-reduce per layer**: A sends its partial, B sends its partial, each adds the other's to its own, done (two async copies + one local add each, event-ordered; no host sync, no lock). So the per-token cost is **64 all-reduces of 2·(H·2 B) each** (bf16 partial) plus the tiny final max-reduce at `lm_head` — at measured ≈ 25 GB/s that is microseconds per token; R7 exists to confirm the "microseconds" with the real topology, not the spec sheet. The same pattern covers batched prefill (per row-group) and the DFlash verify pass, and is what the two-graph/event-sync scheme of R5 encodes.

**Why the residual can stay local.** The residual vector `x` is never split: it is identical on both devices, so only the *delta* (the row-split projection output) needs reducing. This is what keeps the all-reduce count at one per layer instead of two, and it is the property the next step's numerics analysis (R2) builds on.

## Appendix B — CUDA constraints that shape every decision

Recorded here so the next step doesn't re-derive them (and so a future auditor sees which assumptions are platform facts, not project choices):

- **One context = one device.** A process has one CUDA context per device it touches; a graph, a stream, and an event all belong to exactly one context/device. Hence *one* graph covering both devices is impossible; the design space is {two graphs per session, event-synced} or {graph on A + peer-mapped copy nodes, *if* peer access is enabled at capture time and the driver accepts cross-device memcpy nodes in a graph — open, P3/P4}.
- **`cudaSetDevice` is thread-local** — "the current device" is a per-thread fact, and the whole codebase's current behavior (one `setDevice` in `RuntimeImpl::initialize`) is what the CHG-0002 design removes in favor of stream-scoped launches from a single worker thread.
- **Streams and events are device-bound handles** — they cannot be used across contexts; the per-device stream sets (CHG-0012) are the natural unit of the split, and the documented legacy-stream capture-poisoning hazard in `qwen35.h` is per-context, so two contexts *weaken* that hazard rather than strengthen it (invariants re-stated in P2/P3).
- **`cudaMemcpyAsync` D2D between two peer devices** is legal (that is what P2P copy is), and is the atomic op of the all-reduce; when peers are unavailable the same call shape is replaced by device→pinned and pinned→device legs — the transport swap is deliberately invisible above the `gpu_link` API (CHG-0005).
- **No external communication library** (no NCCL): the 2-GPU all-reduce is two peer copies + a local add, which is all a 2-GPU ring ever needs; this keeps the binary inside its 2.5 MB / no-Python-stack envelope (CHG-0032).

---

*Provenance: this plan and the manifest were produced from a full survey of `runtime/`, `kernels/`, `moe/`, `server/`, `bench/`, `eval/`, `.github/`, and `docker/` (200 C++ files, 84 tooling files). The manifest is the record of that survey; this document is the method by which the survey becomes a buildable plan.*
