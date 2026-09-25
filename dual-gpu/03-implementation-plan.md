# Dual-GPU (tp=2, P2P) — implementation plan

**Status:** ready for review · **Created:** 2026-09-25 · **Inputs:** [`01-identification-plan.md`](01-identification-plan.md) (method), [`02-change-manifest.json`](02-change-manifest.json) (record — 3,115 items, source of truth) · **Output:** the PR sequence below; the manifest is the tracking device (items flip `candidate → confirmed → planned → implemented → verified` as their PRs land — this plan never keeps a second list)

---

## 0. Shape of the program at this point

The 01 plan defined phases P0–P5, with P5 (this document) nominally *after* P0–P4. We are writing it earlier, by the program's step structure; the safety mechanism is the **decision gates** (§2): every fact that could change the design — P2P availability, per-card budget, graph-capture rules, eval governance — is resolved by one specific early PR or one explicit org decision, and **no gate outcome changes the build order** (the all-reduce ships dual-transport, the layout is shape-driven, the graph scheme defaults to the always-valid one). The remaining identification work is absorbed, not skipped:

| 01 phase | lands as |
|---|---|
| P0 (P2P probe, budget table) | **PR-1** — the probe *is* code; its execution is a step on the 5060 Ti box, its output committed to `dual-gpu/00-p0-probe/` |
| P1 (candidate triage) + P2 (subsystem audits) | **Track A** — manifest bookkeeping, no code PR; exits when 01's DoD items 1–3 hold |
| P3 (tensor-ownership map + comm protocol) | **PR-5** (layout becomes code, verified against the real `Qwen35Config`) + **PR-2** (comm protocol) |
| P4 (verification matrix) | the **Verification** line of every PR, collected into the eval gates by **PR-14** |

One hardware fact that shapes scheduling: the dev box for this repo has **no GPUs**; the 5060 Ti pair (currently serving a live model) is a separate machine. So: PR-1 is *written and pushed* from here but *executed* there, and all functional testing follows the two-regime test strategy in §5.

## 1. Standing rules (every PR, no exceptions)

1. **tp=1 stays byte-identical.** Default configuration; existing users see no behavior, token, or performance change. Enforced per PR: (a) no tp=1-reachable edit of any existing reduction/sum path — new code lives on the tp>1 branch or in new files; (b) the org's standing eval-bot gates (token-match vs llama.cpp on the 5090, no-regression tiers, DSpark lossless) pass **unmodified**; (c) where a shared file must be touched, the diff is reviewed for no order-of-operations change on the tp=1 path.
2. **No new dependencies.** No NCCL, no new libraries; `gpu_link` is pure CUDA on the existing toolchain (the 2.5 MB / no-Python-stack promise, CHG-0032).
3. **The serving surface only grows.** No existing endpoint or flag changes meaning; additions are `--tp/--devices`, extra `/metrics` fields, extra `/v1/info` fields.
4. **Every PR declares** its manifest items, its verification, and its tp=1 check (the template in §3).
5. **Dual-GPU code paths are unreachable at tp=1 by construction** (branch at the engine level on tp size), so regression risk to existing users is confined to shared-file edits — which rule 1 polices.

## 2. Decision gates

The only open facts in the whole program. Each is resolved *inside* a PR that ships both branches of the design, which is what keeps the build order stable.

| Gate | Resolved by | Question | Outcome A | Outcome B | Order change |
|---|---|---|---|---|---|
| **G1 — P2P** | PR-1 (on the 5060 Ti box) | Does `cudaDeviceCanAccessPeer` hold, and at what measured bandwidth? (R1, R7) | default transport = P2P mapped-memory; the single-graph graph variant (G4) becomes admissible | transport = pinned-host staging, **same API, same all-reduce shape** | none |
| **G2 — budget** | PR-1 (per-card table) | Does 16 GB/card hold weights/2 + drafter + GDN state + KV at the marketed context? (R3) | default `--ctx` unchanged | default `--ctx` lowered **with a message** (CHG-0020) | none |
| **G3 — governance** | org decision, carried in PR-16 | Where is tp=2 scored? (R6) | (a) new dual-5060-Ti eval target with its own tier, or (b) tp=2 arm on a 5090 pair if the org can supply one | (c) tp=2 stays a release-level feature validated by a manual runbook until (a) exists | none — PR-16 lands in whichever form; (c) needs no org action |
| **G4 — graphs** | PR-10 (empirical capture test) | Can a graph in one context contain a memcpy node into the peer's memory (peer access enabled **at capture**)? (R5) | the single-graph variant is available for later perf work | **default: two per-session graphs, event-paired** — always valid, staging or P2P | none (default *is* the fallback) |

## 3. Milestones and PR sequence

```
M0  PR-1 probe+budget            [5060 Ti box]     ──► G1, G2
    Track A: P1 triage + P2 audits (manifest only)
M1  PR-2 gpu_link  ∥  PR-3 device model & config  ∥  PR-4 observability & policy
M2  PR-5 layout+loaders  ─►  (PR-6 KV ∥ PR-7 GDN state)  ─►  PR-8 decode  ─►  PR-9 prefill
M3  PR-10 graphs  ─►  PR-11 DSpark        PR-12 vision + lmcache boundary (needs PR-9)
M4  PR-13 kernel retune ∥ PR-14 tests/bench/eval gates ─► PR-15 docker+docs   PR-16 governance (G3)
```

Parallel symbols are independent; `─►` is a hard dependency. M2's headline: **the 27B loads and serves on 2×16 for the first time**.

---

### M0 — Ground truth

#### PR-1 — P2P probe + per-card budget table
**Items:** CHG-0006 · resolves CHG-0027 (R1), R7, and the numbers of CHG-0020 (R3) · **Depends on:** none · **Gates:** G1, G2
**Scope.** New `runtime/examples/p2p_probe.cpp` + CMake entry. On the target pair it prints: device count/names/SM/VRAM; `cudaDeviceCanAccessPeer` both directions; `nvidia-smi topo -m`; timed D2D copies (P2P path and pinned-staging path) at 1/16/64 MB both directions; and the per-card budget table (16 GB − weights/2 − drafter − GDN state at max batch − overhead = KV remainder → context size, vs the single-32-GB card's claim).
**Verification.** The probe *is* the verification: its raw output is committed verbatim to `dual-gpu/00-p0-probe/`, and the numbers are recorded into CHG-0027/0020/0005 notes. All buffers are explicit small allocations — it runs fine in the busy-box regime.
**tp=1 check.** New file + one CMake line; zero edits to existing code.

#### Track A — Candidate triage + subsystem audits (manifest only, no code)
**Items:** the 3,081 scan candidates + the 14 subsystem checklists · **Depends on:** none (runs in parallel with everything; needs the current tree, which it has since the switch)
**Work.** (1) P1 triage: the 3,081 candidates by file, then category — the bulk (kernels stream-graph/memory-pool call sites, ~2,900 of them) are batch-`rejected` with the shared reason "device-agnostic call site; receives the per-device slice from the layout map"; the rest are kept live or merged into manual items. (2) P2 audits: all 14 subsystems per their manifest checklists → items re-anchored to the current tree (the 23-commit move already shifted some line refs; 2 manual items were re-anchored at the switch, the audit finishes the rest), statuses flipped, severities assigned; the **`kernels` audit (CHG-0029) produces the per-kernel verdict list** (portable / retune-for-48-SM / rewrite) against the current tree — fresh counts here: **471 `__global__`, 292 `launch_*`** (the manifest's 494/239 predates the 23 commits; the audit re-baselines them, including the new int8-prefill and PTQ1 files).
**Exit** = 01's DoD items 1–3: no undecided `candidate`, no `pending-audit` subsystem, kernel verdict list recorded as sub-items.

---

### M1 — Communication substrate + configuration (all invisible at tp=1)

#### PR-2 — `gpu_link`: the 2-GPU communication primitive
**Items:** CHG-0005 (+ new tests under CHG-0022) · **Depends on:** PR-1 (G1 known) · **Gated by:** G1 (variant selection only)
**Scope.** New `runtime/src/gpu_link.cpp` + `include/sparkinfer/gpu_link.h`: (1) **all-reduce-2** — the 2-GPU ring: each device async-copies its partial to the peer (P2P mapped, or two legs via a pinned buffer when G1=B), each adds; event-ordered, no host synchronization anywhere; (2) **max-reduce** for the vocab-split `lm_head` (greedy = cross-device max of two local maxima; sampling = per-device Gumbel-max, one cross-device compare). One public API; the transport is chosen once at init from G1; buffers come from a small arena.
**Verification.** Unit tests vs a host reference (bf16, several sizes, both transports forced explicitly, not just the G1 choice); cross-context event-ordering test; bandwidth report (the measured numbers *become* the perf expectation, R7). Runs entirely in the busy-box regime.
**tp=1 check.** The module is never instantiated at tp=1.

#### PR-3 — Device model + configuration surface
**Items:** CHG-0001, CHG-0002, CHG-0003, CHG-0004 · **Depends on:** none (parallel with PR-2)
**Scope.** `RuntimeConfig` gains `devices[]` + `tp` (default: today's single device 0). `RuntimeImpl::initialize` (currently the one `cudaSetDevice`) queries **both** cards' properties into a per-device table and establishes no process-global "current device" — the worker runs stream-scoped (CHG-0002's recommendation becomes the only model). `ModelEngine::load`'s gate (line 219) becomes: count ≥ `tp`, requested ids exist, arch-mismatch warns. Server CLI gains `--tp N` and `--devices a,b` (+ `SPARKINFER_TP` / `SPARKINFER_DEVICES`); usage text updated.
**Verification.** Validation-matrix unit tests (tp=2 on 1-GPU box refused with the exact message; tp=1 with explicit device works; both ids resolved to names in the log). On the 2-GPU box: boots two contexts, one worker, zero setDevice calls in the hot path.
**tp=1 check.** Default config is textually the existing single-device path; the eval-bot 5090 gate is the arbiter.

#### PR-4 — Observability + failure policy
**Items:** CHG-0018, CHG-0019, CHG-0021, CHG-0034 · **Depends on:** PR-3
**Scope.** Per-card `GpuStats` on both (the NVML-by-PCI-bus-id path already works — this is plumbing); `/metrics` gains per-GPU vram/temp/power and per-pool free blocks; `/v1/info` reports tp size + both GPU names. The thermal governor paces on **max(card A, card B)** (sleep-only, so tokens are unchanged). `device_health`: the two-context failure policy — **any fatal error on either context downgrades the whole server** (decision (a), matching today's operator expectation; per-GPU isolation is documented as the rejected alternative, and why).
**Verification.** `/v1/info` shape test on the 2-card box; an injected-fault test (synthetic context-killing error on one card) showing the server degrades to the "restart required" state from *either* card.
**tp=1 check.** The stats arrays are simply of length 1.

**M1 working definition.** On the 2-GPU box: `--tp 2` boots two contexts and one worker; the all-reduce test passes on the real hardware (G1 answered, on the record); `/metrics` shows both cards. tp=1 is byte-identical (bot green).

---

### M2 — Tensor placement (the memory-necessity milestone)

#### PR-5 — Per-tensor ownership map + weight loaders
**Items:** CHG-0011 (finalizes 01-Appendix-A as code) · **Depends on:** PR-3
**Scope.** The layout becomes a table: tensor-name pattern → {device, axis, slice range}, generated from the real `Qwen35Config`: KV heads 8→4/4, GDN v-heads 48→24/24 (honoring `gdn_qh_block` grouping), full-attn q-heads travel with their KV group, dense FFN gate/up column-split (F/2) + down row-split (F/2 rows), `lm_head` row-split by vocab half, `embed` + norms replicated. **All three loaders** (GGUF, compressed-tensors, flat) consume the table; the opportunistic NVFP4-copy gates (o_fp4, down_fp4, …) become **per-device** decisions (a copy may fit on card A and not card B).
**Verification.** The moment of the program: **Qwen3.8-27B NVFP4 loads onto 2×16 GB** with per-card residency ≤ budget (this is R3 confirmed empirically — if G2 said the context must be lowered, that shows up here as the load succeeding at the lowered ctx); per-tensor placement unit test against a synthetic name set; budget table in the PR description matches the G2 table to the MB.
**tp=1 check.** Single-device placement is bit-identical to today's loaders (same allocation order — reviewed, not assumed).

#### PR-6 — KV cache split
**Items:** CHG-0007, CHG-0008 (+ pool half of CHG-0020) · **Depends on:** PR-3 (parallel with PR-7)
**Scope.** Two flat pools per {K, V} (4+4 KV heads, `head_dim` unchanged), each with its own int8 scale pool and windowed-ring tables; block tables per pool with **one shared logical block numbering**; prefix-sharing refcounts per physical pool (a logical prefix = a pair of blocks); admission capacity = min over pools; the auto-sizing budget (examples' 80%-of-free, the server's `max_seq` formula) takes the **per-device** free amount, never a single-card total.
**Verification.** Paging tests at small explicit budgets (a few blocks, 4+4 heads — the busy-box regime); prefix-restore consistency across both pools; the 262k-class pool arithmetic re-checked against the G2 table.
**tp=1 check.** One pool of 8 heads, as today.

#### PR-7 — GDN recurrent-state split
**Items:** CHG-0009, CHG-0016 · **Depends on:** PR-3, PR-5 (parallel with PR-6)
**Scope.** `open_session` allocates `lin_state`/conv-state per device: 24 v-heads each, respecting the q-head-block grouping; the pinned-host snapshot format (~205 MB/sequence on 27B) gains a per-device half; `save/restore_spec_snapshot` and the dflash spec snapshot follow; prefill conv-state zeroing per device.
**Verification.** **This PR stands on one audited fact: the GDN update has no cross-v-head coupling** (Track A establishes it; if the audit finds any coupling, this PR stops and escalates — the split would be invalid and the fallback is slot-level, a different design). The functional test: 24-head halves advanced on both cards vs the 48-head single-card reference for N steps (small batch/seq; busy-box regime).
**tp=1 check.** One allocation, as today.

#### PR-8 — Decode path: per-layer split + the per-layer all-reduce
**Items:** CHG-0012, CHG-0014 (+ decode half of CHG-0015) · **Depends on:** PR-2, PR-3, PR-5, PR-7 · **Gated by:** G1
**Scope.** The worker (one thread, per-device stream sets, no setDevice) runs each of the 64 layers split: full-attn q/k/v column-split by KV group, `o_proj` row-split; GDN projections block-split, out-proj row-split; FFN gate/up column + down row. The invariant that makes this cheap: **the hidden state is bit-identical on both devices at every layer boundary** — so only the layer's row-split deltas are reduced, in **one fused all-reduce per layer** (2·H bf16 bytes per partial; kilobytes, not megabytes) via `gpu_link`; the residual is added locally and never travels. `lm_head` → per-vocab-half logits → max-reduce. Packed decode follows the same pattern per row-group.
**Verification.** Split-vs-unsplit reference runs: one layer (busy box, ~150–450 MB peak — the tightest test in the program, run one layer at a time) and full model at small seq (quiet box); token-stream check per R2's honest framing — the *invariant* is split-vs-unsplit under identical numerics and the DSpark-internal gate (re-proven in PR-11), while absolute drift vs the tp=1 reference is *measured and recorded*, not assumed.
**tp=1 check.** The decode loop textually keeps its single-device path; all split code is inside the tp>1 branch.

#### PR-9 — Prefill path split
**Items:** prefill-path subsystem (audit-owned; the 23-commit int8-GEMM rework is included in its re-audit) · **Depends on:** PR-6, PR-7, PR-8 (id reuse)
**Scope.** Fused qkvg operand row-split by head group; the NVFP4/FP8/int8 GEMM families take per-device slice pointers (the audited kernels' new split-K/partial outputs recombine under the same numerics rule as decode); GDN chunked-prefill state continuation per device; the DFlash verify (batched_forward/verify_block/warm) runs on both; `ingest_prompts_packed` (multi-session) writes each session's state to both pools.
**Verification.** Chunked-prefill reference match (small chunks, busy box) + a 4k-context run on the quiet box — the spot where the reworked 64×64 int8 GEMM's split correctness matters most.
**tp=1 check.** Unchanged path.

**M2 working definition.** **The 27B NVFP4 serves on 2×16 GB** — AR, greedy and sampled, with split weights/KV/GDN state and one all-reduce per layer. Every number so far is functional, not performance.

---

### M3 — Graphs + DSpark + the remainder of the feature surface

#### PR-10 — CUDA graphs under two contexts
**Items:** CHG-0013 (R5) · **Depends on:** PR-8 · **Gates:** G4
**Scope.** Default scheme (always valid, staging or P2P): **two per-session decode graphs**, one per device, event-paired at each layer-end all-reduce point; the per-session parking lot (max 64) becomes per-device; prefill and DFlash-verify graphs captured per device. Capture stays under the recursive mutex — and the invariants in `qwen35.h` are **restated for two contexts**: the legacy-stream capture-poisoning hazard is per-context, so two contexts *weaken* the cross-thread hazard; the restated rules are written down, not implied. G4's capture test (a graph in context A containing a memcpy node into B's memory, peer access enabled at capture) decides whether the single-graph variant is even admissible — if it passes, that variant is *available* for later perf work, not required.
**Verification.** The capture test suite itself (both contexts; both peer-enabled and staging configurations; graph-vs-eager byte-match for a few tokens per configuration); the 16-concurrent-burst test re-proven with two contexts (submit-time work vs capture serialization).
**tp=1 check.** One graph, as today — the capture path is the same code with one device.

#### PR-11 — DSpark under tp=2
**Items:** CHG-0015, CHG-0030 · **Depends on:** PR-10, PR-5
**Scope.** The 5-layer drafter is **replicated on both cards** (default decision (b); it shares the already-replicated embed/lm_head, and its footprint is gated by the G2 budget table); the verify pass runs on both; spec snapshots use the per-device-half format from PR-7.
**Verification.** **The program's central numerics event:** inside tp=2, re-prove the **byte-lossless gate** (AR vs DSpark under the *same* split — the correct invariant, since both sides move together) and measure drift vs the tp=1 reference (logit deltas, token-match rate, τ). The numbers land in CHG-0028 and become the inputs to PR-14's tp=2 eval gates.
**tp=1 check.** DSpark's single-device path is untouched (replication code behind the tp>1 branch).

#### PR-12 — Vision tower + lmcache boundary
**Items:** CHG-0031, CHG-0017 · **Depends on:** PR-9
**Scope.** The vision tower (27B serves images and video, so this is in-scope, not exotic) runs **on card 0** and P2P/stage-broadcasts its `[n_img, H]` embedding block once per image batch into the prefill (default decision; the worker-thread/legacy-stream constraint in `Request::VisionImage` stays true per context). The lmcache sidecar (a forked process with its own CUDA setup): **tp=2 + sidecar is rejected at load with a clear message** (default decision (a) — defer the dual-GPU KV tier to v2); the sidecar's own single-GPU mode is unaffected.
**Verification.** An image request round-trips under tp=2 (small images; busy box fine); the rejection test for the sidecar combination.
**tp=1 check.** Vision runs exactly as today (on the single card); the rejection is a config check.

**M3 working definition.** Full model feature parity under tp=2: AR + DSpark (re-proven lossless) + vision; graphs on; the sidecar boundary enforced.

---

### M4 — Hardening, tooling, documentation, governance

#### PR-13 — Kernel retune for 48 SM
**Items:** CHG-0029 (the Track A verdict list becomes work here) · **Depends on:** PR-8/9 (the split paths exist to tune)
**Scope.** Apply the verdict list: `n_splits` tiers, occupancy targets, chunk/split-K sizes retuned or **parameterized by arch** for the 48-SM cards (flash-decode `adaptive_nsplits` first — it's the decode-critical one). The subtle bit: the retune must be **arch-keyed, not a constant swap** — 5090 users keep their 170-SM tuning, so the same binary serves both silicon shapes.
**Verification.** Kernel microbenchmarks before/after on the 5060 Ti pair (quiet box; perf is *not* a tp=1 gate — the 5090 bot gates stay untouched).
**tp=1 check.** 5090 arch path verified numerically unchanged (bot green).

#### PR-14 — Test harness + bench + the tp=2 eval gates
**Items:** CHG-0022, CHG-0023, CHG-0028 · **Depends on:** all of M1–M3
**Scope.** (1) The existing ~20 GPU tests run per device via `SPARKINFER_TEST_DEVICE=n` with **no code change**; the new P2P / all-reduce / capture tests from M1–M3 are formalized into the suite. (2) Bench tooling for the dual box: clock pinning for **both** cards (today `_common.sh` pins one and reads `head -1`), the llama.cpp apples-to-apples baseline becomes `--tensor-split 2,2` on the same box, and `bench.sh`/`evaluate.sh`/`accuracy.sh` gain a tp=2 target mode. (3) The **tp=2 eval gate set**, per G3: **re-baseline, never loosen** — token-match on the dual box, DSpark lossless *inside* tp=2 (PR-11's numbers), no-regression tiers — while the single-GPU 5090 bot gates keep running unmodified for tp=1.
**tp=1 check.** The standing bot gates are re-run and green *by this PR's own run*, not by assumption.

#### PR-15 — Docker + documentation
**Items:** CHG-0025, CHG-0026, CHG-0032, CHG-0033 · **Depends on:** M1–M3 (rides the last functional PRs)
**Scope.** Entrypoint passes `--tp/--devices` through; a docker smoke target runs tp=2 on a two-card box; the image otherwise unchanged (no new deps — the size/attestation CI check stays green). README: **2×16 GB as a first-class run mode** — the P2P prerequisite (G1's verdict, honestly stated including the R7 measured-bandwidth caveat: where the PCIe path is root-complex-routed, tp=2 is correct but its all-reduce cost is what the probe measured, not the datasheet), the G2 budget note, docker + from-source instructions. `server/README`: the new flag rows, `/v1/info` and `/metrics` shapes. miner-guide/CONTRIBUTING: how tp=2 PRs are benchmarked and what counts as regression (the G3 decision, whatever it is). CHANGELOG entry.
**tp=1 check.** N/A (docs + entrypoint pass-through).

#### PR-16 — Eval-bot governance landing (the one PR that waits on a human outside the repo)
**Items:** CHG-0024, CHG-0028 · **Depends on:** G3 · **Depends on (soft):** PR-14
**Scope.** Whichever of (a)/(b)/(c) the org decides: (a) wires the new dual-5060-Ti target into the bot with its label tier; (b) adds the tp=2 arm; (c) lands the manual release-level runbook. In all forms this PR also lands the **PR-guard** for tp code paths (what must run for a tp PR — the `rtx5090-required` workflow amended or explicitly exempted, per the decision).
**tp=1 check.** Existing single-GPU PR flow is unchanged in all three forms.

**M4 working definition.** Shippable: docker + docs + eval integration in place, perf numbers published (the R7 measurement becomes the README's honest performance note), and the program's risk register (R1–R7) has a closed entry for every row.

## 4. What this plan deliberately does not do

- No change to the llama.cpp baseline; no new dependencies; no org-policy change beyond the single carried ask (G3); tp=1 users see zero difference (§1 is the contract for that).
- **No gate outcome is pre-decided.** Where the 01 plan records a "default proposal" (drafter replicated, vision on card 0, lmcache deferred, two-graphs, whole-server failure, max-temp pacing), that proposal is the plan's default — and a PR changes it only when a *new measured fact* (PR-1's probe, PR-11's numerics) says so, with the manifest note updated at the same time.

## 5. Test strategy: the two regimes

The 5060 Ti pair is a **live serving box** (it currently runs a model for other work), so testing splits by regime — this is the 01 plan's functional-vs-integration split, executed in PR order:

- **Busy box (LLM resident, ~≤500 MB usable per card):** all *functional* tests in M1–M2. Rules: explicit small budgets only (never auto-sizing — the free-VRAM queries report a shared-tenant number); correctness assertions only (no timing gates); one heavy item resident at a time (tightest: a one-layer split-vs-unsplit at ~150–450 MB peak, one layer per run). The all-reduce *bandwidth* may be sampled here but is recorded as directional (R7) — the LLM's decode steals memory bandwidth.
- **Quiet box (the LLM off, or a dedicated pair):** integration at 4k/16k context, the full-model split-vs-unsplit runs, the DSpark re-proof, the performance tiers, and the final serve — **the M2/M3/M4 gates pass here.**

The practical consequence: M1–M2 can be developed and functionally verified *while the pair stays in production*, and only the milestone gates require the box to be quiet.

## 6. Risk → PR map

| Risk (01 plan) | Resolved by | Recorded in |
|---|---|---|
| R1 — no P2P at all | PR-1 (G1); PR-2 ships both transports regardless | CHG-0027 note + `00-p0-probe/` |
| R2 — numerics drift | PR-11 (measured) → PR-14 (gates set from the numbers) | CHG-0028 |
| R3 — 16 GB/card budget | PR-1 (G2, the table) → PR-5 (empirical load) | CHG-0020 |
| R4 — 170-SM kernel constants | Track A audit (verdict list) → PR-13 (arch-keyed retune) | CHG-0029 sub-items |
| R5 — cross-context graph rules | PR-10 (G4, the capture test) | CHG-0013 note |
| R6 — eval-bot governance | G3 (org) → PR-16 | CHG-0024 |
| R7 — P2P slow / root-complex-routed | PR-1 (measured) → PR-15 (README carries the honest perf note) | CHG-0027 / CHG-0005 |

---

*Provenance: this plan is the 01 plan's phase P5, written against the manifest as it stands after the repo switch to `manuelulreich/sparkinfer-tp2` (tree `20a4fbd`, 3,115 manifest items). Its living surface is the manifest — when PR-1 lands, CHG-0006 flips to `implemented`; when Track A finishes, the 14 subsystems read `audited`; the plan's text is the order and the invariants, the record is the state.*
