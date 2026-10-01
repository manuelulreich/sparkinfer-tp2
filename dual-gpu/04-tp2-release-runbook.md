# tp=2 release runbook — DRAFT for WP-17 form (c)

**Status: draft, not in force.** WP-17 waits on G3 — the choice between (a) a dual-RTX-5060-Ti
target in the eval bot, (b) a tp=2 arm on an RTX 5090 pair, and (c) this manual, release-level
runbook. Nothing here changes a workflow; landing it (and the PR guard at the end) is the
maintainers' decision.

## When

Before every release whose diff touches a tp code path (list under **PR guard**), and whenever the
dual box's driver or CUDA version changes (then re-record the baseline first, on the old commit).

## Box

Two identical 16 GB cards with peer-to-peer access, both idle (`nvidia-smi` shows no other process),
clocks pinned (`bench/scripts/_common.sh` → `pin_clocks`, which now pins every card). Record the
driver, CUDA and `/v1/info` `link` value in the release notes.

## Steps

1. Build: `cmake --build build -j` (server with `-DBUILD_SERVER=ON`).
2. Unit + device suite: `dual-gpu/gates/run_gpu_tests.sh` — must print `failures: 0` (each
   single-GPU test once per card, the 2-GPU link test, the tp CPU tests).
3. tp=2 gate set:
   `python3 dual-gpu/gates/tp2_gates.py --model $M --draft $D --baseline dual-gpu/gates/baseline_2x5060ti.json`
   — exit 0 required. Correctness is exact (DSpark lossless vs ordinary decode under the same
   split; determinism; batched == sequential); performance may not fall more than 5% below the
   recorded baseline on any tier.
4. Cross-machine numerics (tp=1 reference): on an RTX 5090 at the same commit,
   `score_gate.py capture` with `SPARKINFER_DETERMINISTIC=1`; on the dual box `score_gate.py score`
   + `compare`. Bars: top-1 ≥ 0.95, mean KL ≤ 0.02 (provisional until the first real comparison
   sets the tp noise floor; afterwards re-baseline, never loosen).
5. Container: `docker/smoke-tp2.sh <image>` and `docker/smoke-tp2.sh <image> serve-dspark`.
6. Record: the `tp2_gates` `result.json`, the score-gate table and the smoke output go into the
   release notes. A performance improvement is folded into the baseline with `--record` in the same
   release (on a quiet box), so the next release is held to it.

A failure blocks the release until fixed or explicitly waived by a maintainer with the reason on
record.

## PR guard (proposal)

A PR touching any of these must attach a `tp2_gates.py` result from the dual box (or be labelled
`tp-untested` and excluded from the next release until the runbook passes):

- `runtime/src/gpu_link.cpp`, `runtime/csrc/cuda/gpu_link_reduce.cu`, `runtime/include/sparkinfer/gpu_link.h`
- `runtime/src/models/qwen35.cpp` — `*_tp` functions, `TpMirrorScope`/`TP_MIRROR`, `tp_*` rendezvous/all-reduce helpers
- `runtime/src/models/qwen35_prefill.cpp` — `tp_active` branches, `dflash_verify_short_run`'s tp windowing
- `runtime/src/models/dflash_draft.cpp` — `set_embed_split`
- `runtime/src/inference_engine.cpp` — `kv_*` mirror wrappers; `runtime/src/prefix_cache.cpp` — `set_mirror`
- `server/src/model_engine.cpp` — tp load / draft placement / vision device; `server/include/tp_plan.hpp`

The single-GPU `rtx5090-required` workflow is unchanged under this form: tp=1 keeps its 5090 bot
gates; this runbook adds the tp=2 leg at release level only.
