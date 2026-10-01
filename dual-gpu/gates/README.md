# dual-gpu correctness gates (step 2)

`score_gate.py` is a differential, teacher-forced gate: it scores the **same** token streams on a
reference server and a candidate server (`/v1/score`) and compares the per-position
distributions (top-1 agreement, mean |Δ logprob| of the forced token, KL over the union of the
top-20 lists). Free-running text is not compared, because one near-tie fork makes two
otherwise-identical runs look unrelated.

## Cross-machine gate: RTX 5090 (tp=1) vs 2× 5060 Ti (tp=2)

The 27B does not fit one 16 GB card, so the tp=1 reference has to come from the 5090. Both
servers must serve the same checkpoint (`Qwen3.8-27B-NVFP4-RTX5090`, snapshot `5b7a687…`).

```bash
# on the 5090 box (start the server there with SPARKINFER_DETERMINISTIC=1)
python3 dual-gpu/gates/score_gate.py capture --url http://127.0.0.1:8091 --out ref_5090.json
# copy ref_5090.json to the dual-GPU box, start the tp=2 server, then:
python3 dual-gpu/gates/score_gate.py score --url http://127.0.0.1:8091 --ref ref_5090.json --out cand_tp2.json
python3 dual-gpu/gates/score_gate.py compare ref_5090.json cand_tp2.json   # exit 1 = FAIL
```

The default bars (top-1 ≥ 0.95, mean KL ≤ 0.02) are provisional until the first real 5090
comparison sets the expected tp noise floor (bf16 all-reduce rounding + per-rank activation
quantization grids).

## Self-checks on one server

* **Determinism:** start with `SPARKINFER_DETERMINISTIC=1`, `capture`, then `score` the capture
  again and `compare` — must be exact (KL 0).
* **Continuous batching:** `score --parallel 7` against the same capture — must match the
  sequential scores.

Recorded on the 2× 5060 Ti, full 27B at tp=2 (2026-10-01): both self-checks exact
(565 positions, KL 0.00000).

`SPARKINFER_DEBUG_N_LAYERS=N` (truncated stack) is useful for layer-level tp=1 vs tp=2 diffing,
but NOT for this gate: a truncated model's distributions are near-flat, so top-k KL magnifies
rounding noise.
