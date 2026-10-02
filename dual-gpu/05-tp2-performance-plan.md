# 5. tp=2 performance plan: FP4 prefill, all-reduce overlap, concurrency

Written 2026-10-02 after the first `bench/scripts/run_benchmarks.sh` ladder. Plan only; nothing
here is implemented yet.

## Where we stand (2× RTX 5060 Ti, tp=2, no vision, no drafter)

| context | prefill | decode | 5090 tp=1 (README) |
|---:|---:|---:|---|
| 1k | 1,887 tok/s | 53.9 tok/s | – |
| 4k | 2,222 | 53.2 | 14,364 / 93.6 |
| 16k | 2,259 | 51.4 | 13,794 / 90.2 |

- **Decode** is at the bandwidth ratio of the hardware (2 × 448 GB/s vs 1,792 GB/s). Little is left.
- **Prefill** is about 3× below what two 36-SM cards should do. Two causes:
  - no FP4 tensor-core GEMMs at tp=2;
  - un-overlapped all-reduces (~35% of the pass).
- **Concurrency does not scale.** At 1k context, C=1/2/4 give 54/54/52 tok/s aggregate, because `decode_packed_tp` loops over rows one at a time.

Evidence: `pprof2.sqlite` (a 3074-token pass) and `dec16k.sqlite` (16k prompt + 128-token decode), both in the session scratchpad.

## Part A: FP4 tensor cores in tp=2 prefill

**Facts:**
- The loader already builds per-rank NVFP4 operands for every split projection: `ct_nvfp4_slice`, `qwen35.cpp:10630-10680`.
  - Gathered nibbles, gathered group-16 scales, and the rank's own swizzled scale block.
  - `alpha` is tensor-wide, so it is identical on both ranks.
  - Every rank shape is 128-aligned.
- The prefill only uses those operands for attention. It does so **on one rank only**, and that is a bug (A0).
- GDN and FFN route through `proj_fused`: NVFP4 → int8 conversion every pass, then the int8 GEMM.

| projection (per rank) | N×K | today at tp=2 |
|---|---|---|
| GDN qkv / z | 5120×5120 / 3072×5120 | int8 (`prefill.cpp:2276-2277`) |
| GDN out (K-split) | 5120×3072 | int8 (2440) |
| attn q+gate / k / v / o | 6144 / 512 / 512 ×5120, 5120×3072 | FP4 on whichever rank had memory for staging |
| FFN gate, up / down (K-split) | 8704×5120 / 5120×8704 | int8 (3403-3405); FP4 arm blocked by `!tp_ffn_split` (3197) |

**Measured FP4 GEMM throughput** on one 5060 Ti (36 SMs): 345–388 TFLOPS at M ≥ 1024 for every rank shape.

**Estimate:** for 3072 rows, GEMM + conversion + activation quant per rank goes from about 909 ms to about 225 ms.
- At 3–4k: −0.6 s, from 2.2k to roughly 3.0–3.5k tok/s.
- At 16k: −1.7 s of 7.25 s.

### Steps

**A0. Agree the FP4 arms across ranks** (bug fix, do first).
- The problem: FP4 staging (`fp4_a/as/ws`, `fp4_down_*`, `fp4_gdn_*`, `fp4_attn_*`, `prefill.cpp:1443-1606`) is allocated *after* the cross-rank `a8.ok` agreement (1208). A null buffer silently disables an arm on one rank.
- Consequences:
  - The ranks run asymmetric numerics, chosen by free memory.
  - A drafter-loaded server and a plain server can pick different arms, which is a risk to the lossless gate.
- The fix: move those allocations before the agreement, or agree a per-arm bitmask with `tp_prefill_agree_min`. Log the chosen arm set once per pass.
- Verify: nsys shows equal cutlass GEMM counts on both devices; `tp2_gates.py` passes.

**A1. FFN FP4 arm inside `if (tp_ffn)` (3400).**
- Sequence: `quant_a(hn)` → gate and up GEMMs (N=8704) → `swiglu_quant_a` → down GEMM (K=8704, no fused residual) → the existing all-reduce.
- If any launch declines, fall back to `proj_fused`. `fn` is equal on both ranks, so the decision is symmetric.
- Check `prefill_nvfp4_workspace_bytes` at the rank shapes.

**A2. GDN FP4 arm in the `tp_gdn` branch (2270).**
- qkv/z: one `quant_a`, two GEMMs into `tp_qkv` / `tp_z`.
- out_proj (2436): `quant_a(tp_lnA)` → GEMM (K=3072) → all-reduce.
- Extend the `gdn_nvfp4` gate (1532) with the rank shapes.

**A3. Clean-up.**
- With every arm on FP4, skip the int8 `W_i8` staging at tp. That frees arena memory and allows larger FFN chunks.
- Fix the stale "DEFAULT OFF" comment at `prefill.cpp:1265`.
- Precompute the tp GDN padded conv weight once at load. Today `prefill.cpp:2309-2335` does a synchronous D2H, a host loop and an H2D for each of the 48 GDN layers on every pass.

**A4.** Re-record `baseline_2x5060ti.json`.

### Numerics
- tp=1 already uses FP4 for all of these projections (`q38_nvfp4` defaults on), so FP4 moves tp=2 *closer* to tp=1 than today's int8.
- Activation quant is per 16-block with no row-global factor, so N-split projections see the tp=1 activations bit for bit.
- K-split projections add one bf16 rounding per rank before the sum, the same structure as the int8 path today.
- DSpark stays lossless: the verify uses the decode GEMVs, and prefill only fills KV/state identically for both servers, provided A0 makes the arm choice independent of free memory.

### Per-step checks
- the prefill parity check at tp=2 (N = 32/128/512/2048/8192; top-1 ≥ 15/16, KL in the tp=1 range ~0.003);
- 4k/16k retrieval;
- `tp2_gates.py`;
- the prefill ladder;
- `SPARKINFER_DEBUG_PREFILL_DECLINE` for the decline path;
- the tp=1 vs tp=2 `score_gate` comparison once the 5090 capture exists.

## Part B: overlap the prefill all-reduces with compute

**Facts** (from `pprof2.sqlite`):
- There are 129 all-reduces per pass:
  - the embedding;
  - 48 GDN out;
  - 16 attention o;
  - 64 FFN down (one chunk at N=3074).
- Each is a full exchange of a 31.5 MB partial per direction, at 6.6–6.9 GB/s, about 4.7 ms. The two directions already run concurrently.
- The copies are `cudaMemcpyAsync` on the **compute stream** (`gpu_link.cpp:575-580`), so **0 of 258 copies overlap a kernel**, although both cards have 2 copy engines.
- Host rendezvous is not the bottleneck: about 20 µs from the producer kernel to the copy.
- Reduce-scatter + all-gather sends the same bytes on two ranks. Rejected.
- A lossy fp8/int8 link format is opt-in only, never the default.

**Approach:** split each all-reduce into 128-aligned row chunks on a separate copy stream. Everything between two sequence operations is row-wise:

> out-proj → all-reduce → residual + norm → gate/up/down → all-reduce → residual + next layer's input norm

So chunk *c* can go downstream as soon as its own all-reduce lands. The link needs about 9.4 ms per layer, against about 17.8 ms of compute per layer today. With K=4 chunks, exposed link time drops from about 600 ms to about 150 ms per 3k pass. After Part A the compute per layer is roughly the link time, so without this overlap the FP4 gain is capped by the link.

### Steps
- **B1. GpuLink asynchronous API.**
  - A copy stream per rank, plus `allreduce_async(…, ready_ev, done_ev, landing_off)`: wait on the producer, peer copy, exit fence, reduce, record `done`.
  - Add a 2-GPU test: the async result equals the sync result bit for bit, and wall time with a busy kernel is less than kernel time plus copy time.
- **B2. `tp_prefill_allreduce_bf16_async`.**
  - One rendezvous per group; the leader posts K chunks; per-chunk waits on the compute stream.
  - Behind `SPARKINFER_TP_AR_OVERLAP` until the gates pass; off for N < 1024.
- **B3. FFN chunk overlap.**
  - FC = round128(N/K), agreed; the down all-reduce runs asynchronously per chunk.
  - About −200 ms.
- **B4. Chunked out-proj all-reduce feeding the FFN loop.** The tp tail reuses Muse's `muse_tail_chunked` pattern. About −250 ms.
- **B5. Fuse reduce + residual + RMSNorm into one kernel.**
  - Keep today's rounding order. The sum a0+a1 vs a1+a0 is bit-identical, since IEEE addition is commutative.
  - About −70 ms per device.
- **B6. Optional:** chunk the next layer's input projections to hide the tail.

**Numerics:** the all-reduce is element-wise, so chunking it is bit-exact. Chunking the FFN changes the GEMM M dimension, which is already N-dependent today. K must be a pure function of N plus the agreed FC, so AR and DSpark servers prefill identically.

**Risks:**
- cross-device event deadlocks (keep GpuLink's snapshot-wait rule);
- the ranks queueing different chunk sequences (agree K);
- smaller-M GEMM efficiency (chunks ≥ 768 rows).

**Combined A + B estimate:** a 3k pass goes from about 1.76 s to about 0.75–0.9 s, roughly 3.5–4k tok/s.

## Part C: concurrency (what HyperQwen does)

HyperQwen on the same two cards (`bench_logs/default_tp2_bench.txt`, vLLM 0.29, `.env`: SPEC=dflash2, fp8 KV, MAX_LEN 140000):

| | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| decode, sum of per-request rates (sampled, default temperature) | 110–115 | 191–196 | 306–321 | 604 |
| end-to-end (greedy) | 117 | 181 | 314 | 308 |

Its prefill is about 1,600 tok/s (ours is about 2,250).

**What it is** (from its repo):
- **Target:** `Qwen3.8-27B-W4A16-AutoRound` (int4 g128). With `INT8_ACT=int8`, it runs Marlin W4A8 (int8 activations) on every linear.
- **Drafter:** DFlash2, 5 sliding-window layers, block 8, 7 speculative tokens, plus n-gram lookup drafting. vLLM shards it across both cards (inferred).
- **The 110 tok/s single-stream figure includes speculation:** 3.3–3.6 accepted tokens per step, so about 33 target steps/s.
  - Our plain decode is 54 tok/s, so our per-step target is faster.
  - The fair comparison is our DSpark (75–186 tok/s depending on the prompt) on the same prompts. Run `simple_bench`/HyperQwen's `prompts_real.jsonl` through our DSpark server (C1).
- **Why it scales:**
  - vLLM batches every running request's 8-token verify block into **one** target forward (continuous batching, async scheduling, full CUDA graphs captured up to 32 query tokens, `--max-num-seqs 4`).
  - The verify is weight-read bound, so 4×8 query rows cost little more than 8.
  - Acceptance per request stays at about 3.1–3.6 tokens per step, so aggregate throughput grows almost linearly to C4. It stops at C8 because of the 4-seat cap.
  - The all-reduce is NCCL (custom all-reduce disabled); no tp-specific communication patch.

**What that means for us:**
- **Today, tp=2 serves one stream at a time.**
  - DSpark is used only for a single active request.
  - Concurrent requests fall back to ordinary decode.
  - `decode_packed_tp` loops over rows (54 tok/s aggregate at any C).
- **The building blocks for batching exist:** `verify_rows_tp` already runs N rows through the row-batched decode kernels (NVFP4 rows-dp4a, the exact rows MMVQ), with one all-reduce per block for all rows.

### Steps, in order of value per effort
- **C0. Measure DSpark on HyperQwen's real prompts at C1** (an option in `simple_bench.py`: a prompt file plus a `--chat` mode). This makes the 110 vs ours comparison honest.
- **C1. Row-batched `decode_packed_tp`.**
  - Generalize `verify_rows_tp` from "n rows of one session" to "n rows of n sessions":
    - per-row block tables;
    - per-row GDN/conv state pointers;
    - per-row positions;
    - per-row attention via the existing packed kernels.
  - The weight read and the all-reduces are shared across rows.
  - Expected: plain decode aggregate scales roughly like the tp=1 packed path, since a 4-row GEMV costs about 1 row of weight reads. That gives about 54 → ~150–190 tok/s at C4.
  - Numerics: the rows kernels are bit-identical to single-row decode (the existing invariant), so batched == sequential still holds.
- **C2. Batched DSpark across requests.**
  - One verify forward for all active speculative requests: K requests × 8 rows = 32 rows through C1's multi-session rows path.
  - The drafter also runs per request on rank 0. Batch the drafter forward too, or run it on card 1, which today idles during drafting.
  - Admission follows HyperQwen's 4-seat model.
  - This is the change that matches HyperQwen's C2/C4 scaling. Expected aggregate is roughly C × (per-request DSpark rate) × (verify-cost dilution).
- **C3. KV capacity for concurrency + drafter.**
  - fp8/int8 KV is already int8 here.
  - The 49,152 ceiling with the drafter comes from the drafter's bf16 copies plus the verify scratch on card 0. Release the drafter's bf16 weights after quantization (a recorded follow-up), and consider placing the drafter on card 1.
  - HyperQwen holds 140k with the drafter at fp8 KV (5 GiB per GPU).

## Recommended order
1. **A0** (correctness: rank-symmetric arms), then **A1/A2/A3**: the largest single prefill gain, and it sets up B.
2. **B1–B5**: hides the link, which becomes the limit after A.
3. **C0**, then **C1**, then **C2**: concurrency, with C2 matching HyperQwen's scaling.
4. **C3** alongside C2 (memory for multiple speculative seats).

Each step lands with `run_gpu_tests.sh`, `tp2_gates.py --baseline` and the `run_benchmarks.sh` ladder, as in the earlier work packages.
