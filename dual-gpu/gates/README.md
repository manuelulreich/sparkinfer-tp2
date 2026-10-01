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

## tp=2 eval-gate set (WP-15)

`tp2_gates.py` starts the server itself (deterministic and default mode, with and without the
DSpark draft) and runs the gates that hold on the dual box. Correctness is exact, no tolerance:

* **lossless** — DSpark output byte-identical to plain decode under the same split
  (`SPARKINFER_DETERMINISTIC=1` both sides; 7 prompts incl. CJK, code, a long-context prompt),
  and the speculative path actually ran;
* **determinism** — a `score_gate.py` capture scored again: top-1 1.0, KL 0;
* **batching** — the same capture scored with 7 requests in flight: identical.

Performance is a no-regression tier against `baseline_2x5060ti.json` (`--tol`, default 5%):
greedy decode tok/s (server-reported, TTFT excluded), prefill tok/s at 3072 (128-aligned) and 3074
tokens, DSpark tok/s on two prompts that run to `max_tokens`. Re-baseline per box/driver with
`--record` on a quiet machine — never loosen.

```bash
M=~/.cache/huggingface/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/5b7a687fc8211a5d631c8ca6a593dd37eb26ce33
D=~/.cache/huggingface/hub/models--gittensor-model-hub--Qwen3.8-27B-DSpark-NVFP4/snapshots/eba1ac5a66c74902eaa95a4000a7c5eda96d8e95
python3 dual-gpu/gates/tp2_gates.py --model $M --draft $D --baseline dual-gpu/gates/baseline_2x5060ti.json
dual-gpu/gates/run_gpu_tests.sh     # every *_gpu_test on each card (CUDA_VISIBLE_DEVICES), + 2-GPU / tp tests
```

Recorded 2026-10-01 (2× RTX 5060 Ti 16 GB, PCIe Gen3 x8 P2P): all correctness gates pass;
decode 52.0 tok/s, prefill 2159 / 1935 tok/s (3072 / 3074 tokens), DSpark 186 / 76 tok/s.
