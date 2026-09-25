# Dual-GPU (tp=2) — implementation plan & execution brief

**Status:** ready for execution by a fresh agent session · **Created:** 2026-09-25 · **Base tree:** fork `manuelulreich/sparkinfer-tp2` @ `20a4fbd`, branch `dual-gpu-plan`
**Companions:** `01-identification-plan.md` (method, risks R1–R7) · `02-change-manifest.json` (the ledger — **source of truth**) · `02-change-manifest.md` (rendered) · `scan.py` / `render.py` (re-runnable)

---

## 0. Handoff status

*First thing a new session reads. Every session must append a dated line here and commit it before ending.*

- **2026-09-25 — session 1 (planning machine, no GPUs):** program identified (01), manifest built (02, 3,115 items), this brief written. **No dual-GPU code exists yet; the probe has not run; no gate (G1–G4) is resolved.**
- **Next session starts at WP-1** (or at the first WP whose exit criteria are not yet met, if a probe report is already in `dual-gpu/00-p0-probe/`).

---

## 1. Bootstrap — copy this into the new agentic session

> You are continuing the dual-GPU (tensor-parallel 2, P2P) program for **sparkinfer**, a native C++/CUDA LLM inference runtime (single 2.5 MB binary, no Python stack). This machine has **two NVIDIA RTX 5060 Ti** (16 GB each, sm_120, PCIe — no NVLink) and **is also serving a live LLM on a different engine right now** — so only a small slice of each card's VRAM is free for your work.
>
> **Setup:** `git clone -b dual-gpu-plan https://github.com/manuelulreich/sparkinfer-tp2.git` — or, into an existing sparkinfer checkout: `git remote add tp2 https://github.com/manuelulreich/sparkinfer-tp2.git && git fetch tp2 && git checkout -b dual-gpu-plan tp2/dual-gpu-plan`.
>
> **Read, in order:** (1) `dual-gpu/03-implementation-plan.md` — this file; (2) `dual-gpu/01-identification-plan.md` — method + risk register; (3) skim `dual-gpu/02-change-manifest.md` — the item ledger.
>
> **Hard rules:** (1) while the LLM is resident, all VRAM use is **explicit small allocations** — never auto-size from "free VRAM", and never touch, throttle, or kill the resident LLM; (2) **tp=1 behaviour stays byte-identical** at every commit; (3) **no new dependencies** (no NCCL, no libraries — the comm layer is hand-rolled CUDA); (4) the serving API surface only grows; (5) before ending the session: update `dual-gpu/03-implementation-plan.md` §0, update manifest item statuses, re-render (`python3 dual-gpu/scan.py && python3 dual-gpu/render.py`), and commit (`dual-gpu:` prefix). If the session has no GitHub credentials, leave commits local and list them in §0.
>
> **First actions:** run the pre-check below, then **WP-1** (the P2P probe) unless `dual-gpu/00-p0-probe/` already contains a report.
>
> **Pre-check (first minute of the session):**
> ```
> nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
> nvcc --version && cmake --version
> ```
> Two 5060 Ti with little free VRAM → **lean mode** (the default for this box today). Two 5060 Ti mostly empty (LLM off) → **full mode** available. No toolchain → stop and report: the box is a serving box, not necessarily a dev box; do not improvise a partial build.

**The two VRAM regimes, in one breath** (details in §5): lean mode = the LLM is resident, ~≤500 MB/card usable, everything is functional tests with explicit small budgets and correctness-only assertions; full mode = the LLM is off, the 27B loads (17.9 GB), long contexts and perf run. **The work order below is arranged so an agent in lean mode can complete the entire functional discovery of the program; only the scale integrations wait for full mode.**

---

## 2. The facts (what you are building against)

**The model.** Qwen3.8-27B, NVFP4, **17.9 GB — it does not fit one 16 GB card**, which is why tp=2 is a *memory* requirement, not a perf nicety. 64 layers: 16 full-attention (GQA, 8 KV heads) + 48 GDN (48 v-heads, 3 q-heads per v-head, grouped by `gdn_qh_block`); FFN is dense (one expert); `lm_head` spans the full vocab. (Qwen3.8-35B-A3B would split 256 experts 128/128 — out of scope for v1, but the layout table must not hard-against it.)

**The hardware.** 2× RTX 5060 Ti: sm_120, 16 GB GDDR7 (~448 GB/s/card), **48 SM** (this repo's kernels were tuned on a 170-SM 5090 — expect a retune, WP-14), PCIe Gen4 x16 per card (~25 GB/s effective), **no NVLink**. Whether P2P DMA works between this consumer pair at all is **unknown — WP-1 measures it** (consumer boards often say no; a "yes" routed through the root complex is slow — treat bandwidth as a *measured* number, never the spec-sheet one).

**CUDA rules that shape the design** (these are not optional style): one context = one device; streams/events are device-bound; `cudaSetDevice` is thread-local → the design is **one worker thread driving both devices through per-device stream sets, no `setDevice` in the hot path**; a graph cannot span two contexts → default is **two event-paired graphs per session**; a P2P memcpy node inside a graph needs peer access enabled *at capture* → that is a test (WP-11), and the single-graph variant is an optional upgrade, never a requirement; the legacy-stream capture-poisoning hazard is per-context, so two contexts *weaken* the cross-thread hazard — restate the invariants, don't imply them.

**Non-negotiable constraints** (violating any of these is a program failure, not a bug to discuss):
1. **tp=1 stays byte-identical** — no token, behavior, or perf change for existing users; the org's standing eval-bot gates (5090: token-match vs llama.cpp, no-regression tiers, DSpark lossless) pass unmodified. The on-box agent cannot run the 5090 bot (org-pinned hardware); it verifies tp=1 invariance by code-path argument + the in-repo GPU tests that fit the current regime; the bot is the human's/org's job and the final arbiter.
2. **No new dependencies** — no NCCL, no libraries; the 2.5 MB / no-Python-stack promise stands.
3. **The serving surface only grows** — no existing endpoint or flag changes meaning.
4. **DSpark stays byte-lossless** — re-proven *inside* tp=2 (WP-12), never assumed.

**Repo map** (as of `20a4fbd`; this branch's base):
- `runtime/` — C++ engine: `src/models/qwen35.cpp` (the 27B model: ctor streams ~line 894, graph decls ~554/632; one worker thread + recursive `device_mutex`), `src/model_engine.cpp` (**device-count gate at line 219**; KV pool sizing `kvL*2*epb*2*blocks`), `src/runtime.cpp` (the single `cudaSetDevice` in `RuntimeImpl::initialize`, line 15), `src/ternary_ptq1.cpp`, `src/prism_hadamard.cpp`, `examples/` (**WP-1's probe lands here**), `CMakeLists.txt` (superbuild hosting examples/tests).
- `kernels/csrc/` — all kernels (current tree: **471 `__global__`, 292 `launch_*`** — the manifest's 494/239 predates the 23-commit move; the audit re-baselines).
- `server/` — HTTP serving (`/metrics`, `/v1/*`, thermal-governor pacing, `docker/`).
- `dual-gpu/` — this program's record: 01 (method), 02 (manifest), 03 (this file), `scan.py` (repeatable mechanical sweep → candidate items), `render.py` (manifest → MD). After any source change touching audited areas: `python3 dual-gpu/scan.py && python3 dual-gpu/render.py`.
- The manifest is the **source of truth**: 34 hand-audited items (CHG-0001…0034, with severities and `change` design notes) + 3,081 scan candidates; lifecycle `candidate → confirmed → planned → implemented → verified` (terminal `rejected`). **Work packages below reference items by CHG number — flip an item's status when its WP lands, and write measured numbers into its `notes`.**

---

## 3. The plan in one paragraph

Build a two-GPU tensor-parallel path that is **unreachable at tp=1**: a small comm primitive (`gpu_link` — all-reduce + max-reduce, dual-transport: P2P-mapped or pinned-staging behind one API), a per-tensor ownership map that splits weights / KV / GDN state / `lm_head` across the two 16 GB cards, decode + prefill paths that compute both halves per layer from a single worker thread and fuse **one all-reduce per layer** (the hidden state is bit-identical at every layer boundary, so only the row-split deltas reduce — kilobytes, not megabytes), two event-paired CUDA graphs per session (the always-valid scheme), a replicated DSpark drafter with the lossless gate re-proven under tp=2, the vision tower on card 0, and a hard boundary that rejects the lmcache sidecar under tp=2. The work order below is arranged so the first code to land (the probe) is 100% new-file, and so a box with an LLM resident can still do *all* functional discovery within the ~500 MB/card window — only the scale integrations need the box to be quiet.

---

## 4. Work packages

Mode tags: **[L]** = runnable lean (LLM resident, ≤512 MB/card) · **[L-code]** = code lands in lean mode, full-size verification waits for full mode · **[F]** = full mode only (LLM off, or dedicated box). Order within a mode group is a hard sequence; (∥) = safe to run concurrently (file-ownership rules in §6).

### WP-1 — P2P probe + per-card budget table · **[L]** · *start here*
**Manifest:** CHG-0006 · resolves CHG-0027 (R1), R7, and the numbers half of CHG-0020 (R3).
**Do.** Write `runtime/examples/p2p_probe.cpp` + CMake entry (patterns exist in the current examples). It prints: both cards' name/SM/VRAM; `cudaDeviceCanAccessPeer` in both directions; timed device→device copies at 1/16/64 MB, both directions, for (a) the P2P path (if enabled) and (b) the pinned-host staging path (always available); and the **per-card budget table** — 16 GB − weights/2 (≈8.95 GB) − drafter (≈1–1.5 GB when replicated) − GDN state at max batch − runtime overhead = KV remainder → implied context size, side by side with the single-32 GB card's current claim. Buffers are a few MB total — trivially inside the window.
**Verify.** Run it on this box (during an inter-turn lull if you can catch one; record busy + idle — they will differ). Commit the raw output **verbatim** to `dual-gpu/00-p0-probe/`; write the answers into the manifest notes: **G1** (does P2P exist, and what's the default transport?) and **G2** (does 16 GB/card hold the marketed context, or which `--ctx` does?).
**Exit.** Probe report committed; G1 and G2 answered on the record.

### WP-2 — Manifest triage + subsystem audit · **[L]** · (∥ — record-keeping only, no code; may run on *any* machine, including the planning box, while WP-1 runs on the GPU box)
**Manifest:** the 3,081 scan candidates + the 14 subsystem checklists (01's DoD items 1–3).
**Do.** (1) P1 triage: group candidates by file, then by category; batch-`reject` the device-agnostic call sites (~2,900 of them: kernel call sites, stream/graph plumbing, memory-pool callers) with the shared reason *"device-agnostic call site; receives its per-device slice from the layout map"*; keep the rest live or fold into manual items. (2) P2 audit: walk the 14 subsystem checklists against the **current** tree (`20a4fbd` moved the prefill/GEMM areas: `prefill_gemm_i8.cu`, `batched_prefill.cu`, `qwen35.cpp` are the big ones) — re-anchor stale line numbers, assign severities to confirmed items. The **`kernels` audit (CHG-0029) must produce the per-kernel verdict list** (portable / retune-for-48-SM / rewrite) with a fresh `__global__`/`launch_*` count (471/292 baseline). (3) **The one fact that carries design weight:** does the GDN recurrence have **cross-v-head coupling**? WP-8 is only valid if it does not. If coupling is found, **stop and mark it** — the fallback is slot-level splitting, which is a different design and a human decision, not an agent decision.
**Exit.** No undecided `candidate`; all 14 subsystems read `audited`; kernel verdict list recorded; GDN-coupling question answered.

### WP-3 — `gpu_link`: the 2-GPU comm primitive · **[L]** · (∥ with WP-4)
**Manifest:** CHG-0005 + new tests under CHG-0022.
**Do.** New `runtime/src/gpu_link.cpp` + `include/sparkinfer/gpu_link.h` (follow existing include conventions): (1) **all-reduce-2** — each device async-copies its partial to the peer (P2P-mapped pointer, or two legs through a pinned buffer when G1 = staging), each adds; event-ordered; **no host synchronization anywhere**; (2) **max-reduce** for the vocab-split `lm_head` (greedy = cross-device max of the two local maxima; sampling = Gumbel-max compare on one partial). One public API; transport selected at init from G1; buffers from a small arena. **Implement both transports from day one** — the API is identical, so a G1 surprise costs nothing later.
**Verify (lean tests — explicit budgets, correctness only).** Two-GPU all-reduce vs a host reference (bf16, 16 B → 1 MB, both transports forced explicitly, not just the G1 choice); cross-context event-ordering test (A must not observe B's result before B's event); bandwidth recorded as **directional** (the LLM steals bandwidth; note busy + idle).
**tp=1 check.** Module is never instantiated at tp=1 (the test suite proves it unreachable at tp=1).

### WP-4 — Device model + configuration surface · **[L]** · (∥ with WP-3)
**Manifest:** CHG-0001, 0002, 0003, 0004.
**Do.** `RuntimeConfig` gains `devices[]` + `tp` (default = today's single device 0). `RuntimeImpl::initialize` (the one `cudaSetDevice`, `runtime.cpp:15`) queries **both** cards' properties into a per-device table and establishes **no** process-global "current device". `ModelEngine::load`'s gate (`model_engine.cpp:219`) becomes: count ≥ tp, requested ids exist, arch-mismatch warns. Server CLI gains `--tp N` and `--devices a,b` (+ `SPARKINFER_TP` / `SPARKINFER_DEVICES`); usage text updated.
**Verify.** Validation-matrix unit tests (tp=2 on a 1-GPU box refused with the exact message; explicit ids resolve to names in the log; tp=1 with an explicit device works). On the box: `--tp 2` boots two contexts and one worker, with a debug counter asserting **zero `cudaSetDevice` calls in the hot path**.
**tp=1 check.** Default config is textually the existing single-device path; the in-regime tp=1 suite is green.

### WP-5 — Observability + failure policy · **[L]**
**Manifest:** CHG-0018, 0019, 0021, 0034.
**Do.** Per-card `GpuStats` on both (the NVML-by-PCI-bus-id path already works — plumbing only); `/metrics` gains per-GPU vram/temp/power and per-pool free blocks; `/v1/info` reports tp size + both GPU names. The thermal governor paces on **max(card A, card B)** (sleep-only — token streams untouched). Failure policy: **any fatal error in either context downgrades the whole server** (decision (a) — matches today's operator expectation; per-GPU isolation is documented as the rejected alternative, with the reason).
**Verify.** `/v1/info` shape test on the 2-card box; injected-fault test (synthetic context-killing error on one card) → server degrades to the "restart required" state from *either* card.
**tp=1 check.** The stats arrays are simply length 1.

> **Lean-mode milestone M1.** On the 2-GPU box: `--tp 2` boots two contexts; the all-reduce passes on real hardware (G1 answered, on the record); `/metrics` shows both cards; in-regime tp=1 suite green.

### WP-6 — Per-tensor ownership map + weight loaders · **[L-code / F]**
**Manifest:** CHG-0011 (finalizes 01-Appendix-A as code).
**Do.** The layout becomes a table — tensor-name pattern → {device, axis, slice range} — generated from the actual `Qwen35Config`: KV heads 8→4/4; GDN v-heads 48→24/24 (respecting `gdn_qh_block` grouping; q-heads travel with their block); full-attn q-heads with their KV group; dense-FFN gate/up column-split (F/2) + down row-split (F/2 rows); `lm_head` row-split by vocab half; `embed` + norms replicated. **All three loaders** (GGUF, compressed-tensors, flat) consume the table; the opportunistic NVFP4-copy gates (`o_fp4`, `down_fp4`, …) become **per-device** decisions (a copy may fit on card A and not card B).
**Verify.** Lean: per-tensor placement unit tests against a synthetic name set (no model memory needed); budget arithmetic in the WP record matches WP-1's G2 table to the MB. **Full mode: the moment of the program — Qwen3.8-27B NVFP4 loads onto 2×16 GB** with per-card residency ≤ budget (R3 confirmed empirically; if G2 said the context must be lowered, that shows up here as a load that succeeds at the lowered default).
**tp=1 check.** Single-device placement is bit-identical to today's loaders (same allocation order — reviewed, not assumed).

### WP-7 — KV-cache split · **[L]** · (∥ with WP-8)
**Manifest:** CHG-0007, 0008 (+ pool half of CHG-0020).
**Do.** Two flat pools per {K, V} (4+4 KV heads, `head_dim` unchanged), each with its own int8 scale pool and windowed-ring tables; block tables per pool with **one shared logical block numbering**; prefix-share refcounts per physical pool (a logical prefix = a pair of blocks); admission capacity = **min across pools**; auto-sizing budgets (examples' 80%-of-free, server's `max_seq` formula) take the **per-device** free amount — never a single-card total.
**Verify (lean).** Paging tests at small explicit budgets (a few blocks, 4+4 heads — a few MB); prefix-restore consistency across both pools; the 262k-class pool arithmetic re-checked against the G2 table.
**tp=1 check.** One pool of 8 heads, as today.

### WP-8 — GDN recurrent-state split · **[L]** · (∥ with WP-7)
**Manifest:** CHG-0009, 0016. **Precondition: WP-2's audit answered no cross-v-head coupling** — if it found coupling you are not here; escalate.
**Do.** `open_session` allocates `lin_state`/conv state per device: 24 v-heads each; the pinned-host snapshot format (~205 MB/sequence on 27B total) gains a per-device half (~100 MB/card); `save/restore_spec_snapshot` and the dflash spec snapshot follow; prefill conv-state zeroing per device.
**Verify (lean — the tightest fit in the program).** 24-head halves advanced on both cards vs the 48-head single-card reference for N steps: budget **one sequence, ~100–200 MB peak per card**; if the measured free VRAM is less, shrink to a v-head subset (8/8) — the assertion is identical, the size smaller.
**tp=1 check.** One allocation, as today.

### WP-9 — Decode path: per-layer split + per-layer all-reduce · **[L-code / F]**
**Manifest:** CHG-0012, 0014 (+ decode half of CHG-0015). **Gated by:** G1 (transport variant; the code is the same either way).
**Do.** The worker (one thread, per-device stream sets, no `setDevice`) runs each of the 64 layers split: full-attn q/k/v column-split by KV group, `o_proj` row-split; GDN projections block-split, out-proj row-split; FFN gate/up column + down row. **The invariant that makes this cheap: the hidden state is bit-identical on both devices at every layer boundary** — so only the layer's row-split deltas reduce, in **one fused all-reduce per layer** (2·H bf16 per partial — kilobytes, not megabytes) via `gpu_link`; the residual is added locally and never travels. `lm_head` → per-vocab-half logits → max-reduce (greedy) / Gumbel compare (sampling). Packed decode: same pattern per row group.
**Verify.** Lean: **one-layer** split-vs-unsplit reference runs — full-layer variant ≈ 400–500 MB peak per card (unsplit 280 + split 140 + activations); if the window is tighter, sub-layer variants: FFN block only (~150–250 MB), then GDN block only — same assertions, smaller sizes; one heavy item resident at a time, freed between runs. Full mode: full-model decode vs unsplit reference. The token-stream check follows R2's honest framing: the **invariant** is split-vs-unsplit under identical numerics plus the DSpark-internal gate (re-proven in WP-12); absolute drift vs the tp=1 reference is *measured and recorded* (→ CHG-0028), not gated here.
**tp=1 check.** The decode loop keeps its textual single-device path; all split code sits inside the tp>1 branch.

### WP-10 — Prefill path split · **[L-code / F]**
**Manifest:** prefill-path subsystem (audit-owned; the 23-commit int8-GEMM rework is part of its re-audit).
**Do.** Fused qkvg operand row-split by head group; the NVFP4/FP8/int8 GEMM families take per-device slice pointers (split-K/partial outputs recombine under the same numerics rule as decode); GDN chunked-prefill state continuation per device; DFlash verify pass (`batched_forward` / `verify_block` / `warm`) runs on both; `ingest_prompts_packed` (multi-session) writes each session's state into both pools.
**Verify.** Lean: chunked-prefill reference match at small chunks (≤ window). Full mode: 4k-context run — where the reworked 64×64 int8 GEMM's split correctness matters most.
**tp=1 check.** Unchanged path.

> **Lean-mode milestone M2 (code-complete).** Everything above exists and is unit-tested in-regime; the only unverified pieces are the full-size integrations — **the 27B load (WP-6/F) and full-model decode/prefill (WP-9/10/F)** — which pass when the box goes full mode.

### WP-11 — CUDA graphs under two contexts · **[L-code / F]**
**Manifest:** CHG-0013 (R5). **Gates:** G4.
**Do.** Default scheme (always valid, staging or P2P): **two per-session decode graphs** (one per device), event-paired at each layer-end all-reduce point; per-session parking lot (max 64) becomes per-device; prefill + DFlash-verify graphs captured per device. Capture stays under the recursive mutex; **the invariants in `qwen35.h` are restated for two contexts** (the capture-poisoning hazard is per-context, so two contexts *weaken* the cross-thread hazard — write the rules down, don't imply them). The G4 test: a graph in context A containing a memcpy node into B's memory, peer access enabled **at capture** — if it passes, the single-graph variant is *available* for later perf work; it is never required.
**Verify.** The capture test suite (both contexts; peer-enabled and staging configurations; graph-vs-eager byte-match for a few tokens per configuration); the 16-concurrent-burst test re-proven with two contexts (submit-time work vs capture serialization).
**tp=1 check.** One graph, as today — same capture code with one device.

### WP-12 — DSpark under tp=2 · **[L-code / F]**
**Manifest:** CHG-0015, 0030.
**Do.** The 5-layer drafter is **replicated on both cards** (default decision; it shares the already-replicated embed/`lm_head`; its footprint is gated by the G2 table — 2–3 GB total, ≈1–1.5 GB/card when replicated); the verify pass runs on both; spec snapshots use the per-device-half format from WP-8.
**Verify (full mode only — the drafter does not fit the lean window).** **The program's central numerics event:** re-prove the **byte-lossless gate *inside* tp=2** (AR vs DSpark under the *same* split — the correct invariant, since both sides move together) and measure drift vs the tp=1 reference (logit deltas, token-match rate, τ). The numbers land in CHG-0028 and set WP-15's tp=2 eval gates.
**tp=1 check.** DSpark's single-device path is untouched (replication behind the tp>1 branch).

### WP-13 — Vision tower + lmcache boundary · **[L-code / F]**
**Manifest:** CHG-0031, 0017.
**Do.** The vision tower (27B serves images and video — this is in-scope, not exotic) runs **on card 0** and broadcasts its `[n_img, H]` embedding block (P2P or staged per G1) into the prefill once per image batch. The lmcache sidecar (forked process with its own CUDA setup): **tp=2 + sidecar is rejected at load with a clear message** (decision (a) — the dual-GPU KV tier is deferred to v2); the sidecar's own single-GPU mode is unaffected.
**Verify.** Image requests round-trip under tp=2 (small images — a lean functional test is fine); the rejection test.
**tp=1 check.** Vision runs exactly as today; the rejection is a config check.

### WP-14 — Kernel retune for 48 SM · **[F]**
**Manifest:** CHG-0029 (WP-2's verdict list becomes work here).
**Do.** Apply the verdict list: `n_splits` tiers, occupancy targets, chunk/split-K sizes retuned or **parameterized by arch** for the 48-SM cards (flash-decode `adaptive_nsplits` first — the decode-critical one). The subtle bit: the retune must be **arch-keyed, not a constant swap** — 5090 users keep their 170-SM tuning, so one binary serves both silicon shapes.
**Verify.** Kernel microbenchmarks before/after on the 5060 Ti pair (perf is *not* a tp=1 gate — the 5090 bot gates stay untouched).
**tp=1 check.** 5090 arch path verified numerically unchanged (bot green).

### WP-15 — Test harness + bench + tp=2 eval gates · **[F]**
**Manifest:** CHG-0022, 0023, 0028.
**Do.** (1) The existing ~20 GPU tests run per device via `SPARKINFER_TEST_DEVICE=n` with **no code change**; the new P2P/all-reduce/capture tests from WP-3/WP-11 are formalized into the suite. (2) Dual-box bench tooling: clock pinning for **both** cards (today `_common.sh` pins one and reads `head -1`); the llama.cpp apples-to-apples baseline becomes `--tensor-split 2,2` on the same box; `bench.sh`/`evaluate.sh`/`accuracy.sh` gain a tp=2 target mode. (3) The **tp=2 eval-gate set**, per G3: **re-baseline, never loosen** — token-match on the dual box, DSpark lossless *inside* tp=2 (WP-12's numbers), no-regression tiers — while the single-GPU 5090 bot gates keep running unmodified for tp=1.
**tp=1 check.** The standing bot gates are re-run and green by *this WP's own run*, not by assumption.

### WP-16 — Docker + documentation · **[L-code / F]**
**Manifest:** CHG-0025, 0026, 0032, 0033.
**Do.** Entrypoint passes `--tp`/`--devices` through; docker smoke target for two-card boxes; image otherwise unchanged (no new deps — the size/attestation CI check stays green). README: **2×16 GB as a first-class run mode** — the P2P prerequisite (G1's verdict, honestly, including the R7 measured-bandwidth caveat: where the path is root-complex-routed, tp=2 is *correct* and its all-reduce cost is what the probe *measured*, not the datasheet), the G2 budget note, docker + from-source instructions. `server/README`: new flag rows, `/v1/info` and `/metrics` shapes. miner-guide/CONTRIBUTING: how tp=2 PRs are benchmarked and what counts as a regression (the G3 decision, whatever it is). CHANGELOG entry.

### WP-17 — Eval-bot governance landing · **[F]**
**Manifest:** CHG-0024, 0028. **Gated by:** G3 (the one decision in this program that waits on a human outside the repo).
**Do.** Whichever of (a)/(b)/(c) is chosen: (a) wire the new dual-5060-Ti target into the bot with its label tier (org action); (b) add the tp=2 arm on a 5090 pair (org can supply); (c) land the manual release-level runbook — **(c) needs no org action, so it can land whenever the human decides, without blocking anything else**. In all forms this WP also lands the **PR-guard** for tp code paths (what must run for a tp PR — the `rtx5090-required` workflow amended or explicitly exempted, per the decision).
**tp=1 check.** Existing single-GPU PR flow unchanged in all three forms.

---

## 5. The two VRAM regimes (the single most important operational constraint)

This box **serves a live LLM** (another engine, same two cards). Free VRAM per card is roughly **500 MB** at a given moment and fluctuates with the serving load.

**Lean mode (default right now — everything that fits the window):**
- **Measure first, every session:** `nvidia-smi --query-gpu=memory.free` on both cards; record the minimum; budget every allocation *below* it, explicitly. **Never** auto-size from free VRAM (that number is the tenant's, not yours) and never from a busy card's "total − used".
- **One heavy item resident at a time.** Footprints: one 27B layer ≈ 280 MB NVFP4 (≈140 MB/card split); a one-layer split-vs-unsplit test ≈ 400–500 MB peak per card (sub-layer variants 150–250); a 24-v-head GDN state ≈ 100–200 MB/card per sequence; pinned snapshots ≈ 100 MB/card; KV test pools: a few MB; all-reduce test partials: 16 B–1 MB. **Anything that doesn't fit shrinks to a sub-variant (sub-layer, v-head subset, fewer blocks) — the assertion is identical, the size is smaller. Never shrink the assertion to fit the memory.**
- **Correctness assertions only.** No perf gates in lean mode: the LLM steals memory bandwidth and SM time, so any timing you take is contention-noisy — record it (busy + idle, when catchable) as *directional* and treat it as data, not a result.
- If measured free VRAM per card drops **below ~200 MB** (the serving load grew), even the sub-variants may not fit: record that in §0, keep working the light WPs (1, 2, 3, 4, 5, 6-table, 7), and **do not touch the resident LLM** — not throttle, not restart, not "borrow".

**Full mode (LLM off on this box, or a dedicated box):**
The full 27B load (WP-6/F), full-model decode/prefill vs unsplit (WP-9/10/F), 4k/16k-context integration, the DSpark re-proof (WP-12/F), performance tiers (WP-14/15), final serve. **The M2/F, M3, M4 milestone gates pass only here.** An agent must confirm the LLM is actually off (large `memory.free`) before starting any full-mode test — and if the human says the LLM must stay up, the agent stays lean-mode end-to-end and records exactly what is deferred.

**Program definition of done** (per 01): all WPs landed in the order above (lean-mode items may finish before full-mode ones — that is the point); G1–G4 all have recorded resolutions; the tp=1 contract held at every merge (in-regime suite here + the 5090 bot at each milestone, run by the human); the manifest's DoD 1–7 met; the `00-p0-probe/` report on the record; the README tells the bandwidth truth.

---

## 6. Parallelism and file ownership (for subagents)

One agent can do all of this serially; the WPs are partitioned so a **parent agent can safely dispatch subagents**. The rule that makes it work: **one writer per file at any time** — no subagent edits a file it does not own, and the parent integrates between waves.

| Subagent | Exclusive write ownership | WPs |
|---|---|---|
| A | `runtime/examples/p2p_probe.cpp`, CMake examples entry, `dual-gpu/00-p0-probe/` | WP-1 |
| B | `dual-gpu/` record files (02 manifest json/md) | WP-2 |
| C | `runtime/src/gpu_link.*`, `include/sparkinfer/gpu_link.h`, comm tests | WP-3 |
| D | `runtime/src/runtime.cpp`, `RuntimeConfig`, `runtime/src/model_engine.cpp`, server CLI/usage | WP-4, WP-5 |
| E | the three weight loaders, the `tp_layout` table | WP-6 |
| F | the KV-cache pool files | WP-7 |
| G | session/GDN state files (qwen35 session section, snapshot code) | WP-8 |
| H | `runtime/src/models/qwen35.cpp` forward path (decode, prefill, graphs, DSpark wiring) | WP-9, 10, 11, 12 — **one subagent, in sequence, over this file** |
| I | vision-tower files, lmcache boundary check, `docker/`, docs | WP-13, 16, 17 |

**Safe parallel sets (everything else is a hard serial dependency):**
- **Wave 0 (start immediately after bootstrap):** A ∥ B — the probe on the GPU box, the audit on any machine (they share no files; the audit needs zero VRAM, the probe a few MB — this is exactly why they are the start: they consume nothing the LLM needs and answer the two facts, G1/G2, everything else branches on).
- **Wave 1:** C ∥ D (new files vs config surface).
- **Wave 2:** E → (F ∥ G) — E first, because F/G's state tensors need the layout's naming.
- **Wave 3:** H alone (it is long; its only internal parallelism is sequential sub-variants — decode split, then prefill split — both inside H's single ownership of `qwen35.cpp`; **never** run two subagents against this file).
- **Wave 4:** I ∥ (WP-14/15 on the quiet box — those need the box, not a subagent).

**Integration duty (parent, after each subagent finishes, before the next wave):** (1) build; (2) run the tp=1 in-regime suite that fits current VRAM (the 5090 bot is unreachable from here — the human runs it at milestones); (3) re-render the manifest (`scan.py` + `render.py`); (4) commit with the `dual-gpu:` prefix and update §0; (5) *then* dispatch the next wave. Header drift is the only way this program can break itself: if two WPs ever both need a shared header, the parent resolves it in one sequential pass.

---

## 7. Risk → resolution map

| Risk | Resolved by | Recorded in |
|---|---|---|
| R1 — no P2P on this consumer pair at all | WP-1 (G1); WP-3 ships both transports regardless, so the order never changes | CHG-0027 note + `00-p0-probe/` |
| R2 — numerics drift (split GEMMs + bf16 reduce vs single card) | WP-12 measures it *inside* tp=2 (DSpark gate re-proven under the same split); WP-15 sets the gates from the numbers — re-baseline, never loosen | CHG-0028 |
| R3 — 16 GB/card budget doesn't hold at marketed context | WP-1 (G2, the table) → WP-6 (empirical load; lowered default `--ctx` *with a message* if needed) | CHG-0020 |
| R4 — kernels tuned for 170-SM 5090 on 48-SM cards | WP-2 audit (verdict list) → WP-14 (arch-keyed retune) | CHG-0029 sub-items |
| R5 — graphs can't span contexts; P2P nodes need peer access at capture | WP-11 (G4, the capture test; two-graphs default is the always-valid scheme) | CHG-0013 note |
| R6 — the eval bot only knows one 5090 | G3 (org) → WP-17 (form (c) needs no org action) | CHG-0024 |
| R7 — P2P "exists" but is root-complex-routed / slow | WP-1 measures it; WP-16's README carries the honest performance note | CHG-0027 / CHG-0005 |

---

## 8. What this plan deliberately does not do

- No change to the llama.cpp baseline; no new dependencies; no org-policy change beyond the single G3 ask; **no pull request is opened against the org repo at any point** — the human decides when that happens (WP-17 is the only governance-touching WP, and its form (c) needs no org action at all).
- **No gate outcome is pre-decided.** Where 01 records a "default proposal" (drafter replicated, vision on card 0, lmcache deferred, two-graphs, whole-server failure, max-temp pacing), that proposal is the plan's default — and a WP changes it only when a *new measured fact* (WP-1's probe, WP-12's numerics) says so, with the manifest note updated in the same commit.
- The tp=1 user sees zero difference. That is the contract; the tp=1 check in every WP is its enforcement.

---

*Provenance: P5 of `01-identification-plan.md`, written against fork `manuelulreich/sparkinfer-tp2` (branch `dual-gpu-plan`, base `20a4fbd`, manifest at 3,115 items). This file is the handoff brief: a new session on the 5060 Ti box starts at §1, works the WPs in wave order, and updates §0 before ending. The manifest is the state; this file is the order.*
