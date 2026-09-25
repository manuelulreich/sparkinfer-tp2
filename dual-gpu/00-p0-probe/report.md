# WP-1 verdict — P2P probe + NCCL smoke + per-card budget (2× RTX 5060 Ti)

Consolidated verdict of the probe runs in this directory. The raw outputs
(`p2p_probe_run1-4.txt`, `nccl_smoke_run*.log`, `probe_stderr_run*.log`,
`precheck/postcheck_nvidia_smi_*.csv`, `nvml_topo.txt`, `budget_table.md`)
are the verbatim record; this file interprets them and answers **G1, G2, R8**.
Probe: `runtime/examples/p2p_probe.cpp` (dlopen's NCCL at runtime — NCCL is
*never* a build dependency). CUDA 13.4, driver as of 2026-09-25.

## Run history

| run | time (CEST) | regime | NCCL build tested | NCCL outcome |
|---|---|---|---|---|
| 1 | 16:41–16:43 | busy (94–100% util, 975 MiB free/card) | **2.20.5+cu12.4** (pip wheel) | bootstrap OK, **`ncclCommInitRank` wedged → 120 s timeout** |
| 2 | 16:55 | mixed | **2.32.3+cu13.4** (nvidia cu13 wheel, fetched to `/tmp/opencode/nccl_new/`) | init OK (0.6 s); all-reduce **+inf, MISMATCH** |
| 3 | 16:58 | mixed | 2.32.3+cu13.4 | same as run 2 |
| 4 | 16:59 | caught a **0%-util inter-turn window** at the end | 2.32.3+cu13.4 | same as run 2 |

Final `nvidia-smi` (17:00, `postcheck_nvidia_smi_run1.csv` — written last,
mislabeled by the probe): 1109 MiB free/card, 100% util → the LLM was busy
again; the NCCL symmetric-VA heap was fully reclaimed on process exit.

## G1 — does P2P exist, and which transport will the comm layer use?

**P2P: YES, both directions** (`cudaDeviceCanAccessPeer` yes/yes). Topology is
**PHB** (PCIe + host bridge, `nvml_topo.txt`) — i.e. **root-complex-routed, no
NVLink** on this consumer pair (R1's "consumer boards often say no" answered:
they say *yes, slowly*).

Measured (D2D, 1/16/64 MB, both directions, 8 reps each):

| path | median range across runs | note |
|---|---|---|
| **P2P (D2D direct)** | **6.52 – 7.18 GB/s** | run4 (0%-util window) medians 7.14–7.18; run1 (busy) 5.75–7.17 |
| pinned-host staging (d2h → h2d, 2 serial legs) | 3.28 – 3.53 GB/s end-to-end | legs themselves 6.4–7.2 GB/s each |

**Plan-fact correction (R7-related):** the PCIe link is **Gen3, currently ×8**
(max ×16) per `nvidia-smi -q` — *not* the Gen4 ×16 the plan assumed. So the
P2P number is ≈ that (degraded) link's line rate (~7.9 GB/s raw), not a Gen4
ceiling. Consequences: the all-reduce per layer is bounded at ~7 GB/s of
payload; the two-graphs / event-pairing design (WP-11) is unaffected (it needs
ordering, not bandwidth). **The comm layer's actual transport on this box is
the hand-rolled P2P-mapped path** (see R8 below) — P2P measured ≈2× staging,
so P2P-mapped is the fast path and pinned-staging the runtime fallback.

## R8 — is NCCL usable on this pair? **No. Both builds tested are dead.**

1. **2.20.5+cu12.4** (the box's pip wheel, pre-sm_120): `ncclGetVersion` and
   bootstrap succeed, but `ncclCommInitRank` **never completes** — 120 s probe
   timeout (run 1). The cu12 prebuild does not init on this sm_120 /
   CUDA-13.4-drv pair.
2. **2.32.3+cu13.4** (the nvidia cu13 wheel, the correct-generation build):
   init **completes in 0.6 s**, 2 channels, transport **SHM/direct**
   (`Channel 0x : 0[0] -> 1[1] via SHM/direct`) — but the 1024-float
   all-reduce **silently returned +inf on both ranks (expected 3.0) in all
   three runs (2–4), with no NCCL error of any kind**. Silent data corruption
   is worse than a timeout: a serving runtime would serve wrong tokens.
3. Bookkeeping: a live 2-rank comm on 2.32.3 commits **≈160 MiB/card** of
   symmetric-VA heap at init, **held until process exit** (visible as the
   975→815 MiB free delta in run 4's [1]→[6]; reclaimed at exit).

**Decision (recorded for WP-3):** `gpu_link` ships the **hand-rolled 2-node
design** on this box (P2P-mapped fast path, pinned-staging fallback — the
CHG-0005 note), behind the one-API header. The NCCL code path remains in the
design for NVLink/server-class hardware, **gated by a re-run of this probe**.
The 2.5 MB promise is thereby restored (binary only, no libnccl on this pair).

## G2 — does 16 GB/card hold the marketed context?

`budget_table.md` (every number measured or read from source at
`20a4fbd`): 262,144 tokens need an **8.06 GiB** int8-KV pool; the most a card
can ever hand it here is **6.74 GiB** (AR-only, no drafter, batch 1) → **the
marketed 262k does not fit on 16 GB/card** (it fits only on the single 32 GB
card the README describes). **131,072 (4.04 GiB) does fit — with the DSpark
drafter — at the code's real batch anchors 1 and 8** → the honest default for
this pair is **`--ctx 131,072`** (matches the single-card DSpark mode and the
`model_engine` fit message). GDN state is 147.75 MiB/seq (fp32) /
75.75 MiB (bf16) under the shipped **48-slot** layout — the plan's "~205 MB"
was the pre-compaction 64-slot figure.

## Plan-fact corrections (measured vs assumed)

| fact | plan said | measured |
|---|---|---|
| SM count per card | 48 | **36** (SM 12.0) — WP-14 retune targets 36, arch-keyed as before |
| PCIe link | Gen4 ×16, ~25 GB/s | **Gen3, ×8 (max ×16)**, P2P ≈ line rate (~7.1 GB/s) |
| driver-reserved VRAM/card | not assumed | **421 MiB** (16,311 total − 15,890 free at idle) |
| GDN state/sequence | ~205 MB | **147.75 MiB fp32** (48-slot layout) |

## File inventory (this directory)

- `p2p_probe_run{1,2,3,4}.txt` — full verbatim probe stdout per run (identity/VRAM,
  `cudaDeviceCanAccessPeer`, timed D2D P2P + staging, NCCL smoke with its
  `NCCL_DEBUG=INFO` transcript, post-probe VRAM, wall time).
- `nccl_smoke_run1.log` — the 2.20.5 timeout notice; runs 2–4 are 0 bytes
  because their NCCL transcripts were captured inline in the run txts.
- `probe_stderr_run*.log` — all empty (the probe wrote nothing to stderr).
- `precheck_nvidia_smi.csv` (15:59) and `postcheck_nvidia_smi_run{1..4}.csv`
  — free VRAM + utilization before/after each run (the 17:00 "run1"-named
  file is the final snapshot, written after run 4).
- `nvml_topo.txt` — `nvidia-smi topo -m` (PHB/PHB).
- `budget_table.md` — the G2 budget table + NCCL-heap addendum.
