# CPU composition tests — pure C++, validate the wiring math (no GPU needed).
add_executable(decode_layer_cpu_test decode_layer_cpu_test.cpp)
set_target_properties(decode_layer_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME decode_layer_cpu_test COMMAND decode_layer_cpu_test)

add_executable(qwen35_cpu_test qwen35_cpu_test.cpp)
set_target_properties(qwen35_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen35_cpu_test COMMAND qwen35_cpu_test)

add_executable(qwen36_config_cpu_test qwen36_config_cpu_test.cpp ../src/gguf.cpp)
set_target_properties(qwen36_config_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen36_config_cpu_test COMMAND qwen36_config_cpu_test)

# Muse Glimmer DFlash draft GGUF config parsing. Links only gguf.cpp, same as
# qwen36_config_cpu_test above -- no kernels/moe/full runtime needed. dflash_draft.h (pulled in
# via ../examples/dflash_gguf_config.h for DFlashDraftConfig) needs <cuda_runtime.h> for
# cudaStream_t in DFlashDraftModel's method signatures even though this test never constructs
# one, so link CUDA::cudart for its headers -- lighter than sparkinfer_runtime, no device
# execution, still GPU-free in the sense that matters (no kernels, no MoE engine).
add_executable(dflash_gguf_config_cpu_test dflash_gguf_config_cpu_test.cpp ../src/gguf.cpp)
target_link_libraries(dflash_gguf_config_cpu_test PRIVATE CUDA::cudart)
set_target_properties(dflash_gguf_config_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME dflash_gguf_config_cpu_test COMMAND dflash_gguf_config_cpu_test)

# Thermal governor — pure tiering policy (no GPU); links the runtime for classify()/pace() symbols.
add_executable(thermal_governor_cpu_test thermal_governor_cpu_test.cpp)
target_link_libraries(thermal_governor_cpu_test PRIVATE sparkinfer_runtime)
set_target_properties(thermal_governor_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME thermal_governor_cpu_test COMMAND thermal_governor_cpu_test)

add_executable(scheduler_cpu_test scheduler_cpu_test.cpp)
target_link_libraries(scheduler_cpu_test PRIVATE sparkinfer_runtime)
set_target_properties(scheduler_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME scheduler_cpu_test COMMAND scheduler_cpu_test)

# BridgeClient (docs/lmcache_bridge_protocol.md) against a from-scratch fake AF_UNIX peer --
# no GPU, no real lmcache/Python, no shm KV staging (that's bridge/tests/test_bridge_e2e.py's
# job). Only needs lmcache_bridge_client.cpp, not the full sparkinfer_runtime -- it has zero CUDA
# dependencies (confirmed: compiles standalone with plain g++, no CUDA headers needed).
add_executable(lmcache_bridge_client_cpu_test lmcache_bridge_client_cpu_test.cpp
               ../src/lmcache_bridge_client.cpp)
target_link_libraries(lmcache_bridge_client_cpu_test PRIVATE Threads::Threads)
set_target_properties(lmcache_bridge_client_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME lmcache_bridge_client_cpu_test COMMAND lmcache_bridge_client_cpu_test)

# Packed-decode GDN kernels vs the unbatched path, bit for bit.
add_executable(gdn_batched_gpu_test gdn_batched_gpu_test.cpp)
target_link_libraries(gdn_batched_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME gdn_batched_gpu_test COMMAND gdn_batched_gpu_test)

add_executable(batch_engine_gpu_test batch_engine_gpu_test.cpp)
target_link_libraries(batch_engine_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME batch_engine_gpu_test COMMAND batch_engine_gpu_test)

add_executable(temperature_sample_gpu_test temperature_sample_gpu_test.cpp)
target_link_libraries(temperature_sample_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME temperature_sample_gpu_test COMMAND temperature_sample_gpu_test)

add_executable(topk_topp_sample_gpu_test topk_topp_sample_gpu_test.cpp)
target_link_libraries(topk_topp_sample_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME topk_topp_sample_gpu_test COMMAND topk_topp_sample_gpu_test)

add_executable(logprob_extract_gpu_test logprob_extract_gpu_test.cpp)
target_link_libraries(logprob_extract_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME logprob_extract_gpu_test COMMAND logprob_extract_gpu_test)

add_executable(penalty_gpu_test penalty_gpu_test.cpp)
target_link_libraries(penalty_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME penalty_gpu_test COMMAND penalty_gpu_test)

# Gated-DeltaNet state carried into a batched prefill scan that does not start at position 0: a pass
# split at a chunk boundary must reproduce the unsplit pass bit for bit, for both scan paths.
add_executable(gdn_prefill_carry_gpu_test gdn_prefill_carry_gpu_test.cpp)
target_link_libraries(gdn_prefill_carry_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME gdn_prefill_carry_gpu_test COMMAND gdn_prefill_carry_gpu_test)

add_executable(logit_bias_gpu_test logit_bias_gpu_test.cpp)
target_link_libraries(logit_bias_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME logit_bias_gpu_test COMMAND logit_bias_gpu_test)

# Prefix sharing in the paged KV pool (refcounted blocks) and the PrefixCache on top of it: every
# free-block count is exact, so a leaked or prematurely freed block fails a specific line.
add_executable(prefix_cache_gpu_test prefix_cache_gpu_test.cpp)
target_link_libraries(prefix_cache_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME prefix_cache_gpu_test COMMAND prefix_cache_gpu_test)

add_executable(kv_swa_ring_gpu_test kv_swa_ring_gpu_test.cpp)
target_link_libraries(kv_swa_ring_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME kv_swa_ring_gpu_test COMMAND kv_swa_ring_gpu_test)

# Real sidecar subprocess + real KV cache + a small synthetic model on real hardware -- the one
# place both halves of the LMCache bridge run together against the real model code path. Skips
# (not fails) unless SPARKINFER_LMCACHE_PYTHON/_BRIDGE_SCRIPT point at a real lmcache+torch venv
# (see bridge/README.md) -- most environments running the rest of this suite won't have that
# ~1.5GB, CPU-only-torch-is-enough dependency installed.
add_executable(lmcache_e2e_gpu_test lmcache_e2e_gpu_test.cpp)
target_link_libraries(lmcache_e2e_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME lmcache_e2e_gpu_test COMMAND lmcache_e2e_gpu_test)

# GPU integration tests — run the real device path; skip without a device.
add_executable(decode_runner_gpu_test decode_runner_gpu_test.cpp)
target_link_libraries(decode_runner_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME decode_runner_gpu_test COMMAND decode_runner_gpu_test)

add_executable(qwen35_gpu_test qwen35_gpu_test.cpp)
target_link_libraries(qwen35_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME qwen35_gpu_test COMMAND qwen35_gpu_test)

add_executable(dflash_gdn_checkpoint_gpu_test dflash_gdn_checkpoint_gpu_test.cpp)
target_link_libraries(dflash_gdn_checkpoint_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME dflash_gdn_checkpoint_gpu_test COMMAND dflash_gdn_checkpoint_gpu_test)

# Video preprocessing — pure C++, no GPU, no checkpoint. Anchors the video path to the image
# path by EQUIVALENCE: a 2-frame video of one still must be byte-identical to that still as an
# image, because the tower's Conv3d time extent is filled by repetition for images and by real
# frames for video. That single assertion transfers the image path's measured reference-exactness
# (PIL bicubic constant, the twice-rounded uint8 intermediate, the (C,T,P,P) layout) onto video
# without re-verifying any of it against transformers.
add_executable(qwen_vision_video_cpu_test qwen_vision_video_cpu_test.cpp
               ../src/models/qwen_vision_preprocess.cpp)
target_include_directories(qwen_vision_video_cpu_test PRIVATE ../include)
set_target_properties(qwen_vision_video_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen_vision_video_cpu_test COMMAND qwen_vision_video_cpu_test)

add_executable(ternary_ptq1_cpu_test ternary_ptq1_cpu_test.cpp ../src/ternary_ptq1.cpp)
set_target_properties(ternary_ptq1_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(ternary_ptq1_cpu_test PRIVATE ../include)
add_test(NAME ternary_ptq1_cpu_test COMMAND ternary_ptq1_cpu_test)

add_executable(prism_hadamard_cpu_test prism_hadamard_cpu_test.cpp ../src/prism_hadamard.cpp ../src/gguf.cpp)
set_target_properties(prism_hadamard_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(prism_hadamard_cpu_test PRIVATE ../include)
add_test(NAME prism_hadamard_cpu_test COMMAND prism_hadamard_cpu_test)

add_executable(gdn_v_regroup_cpu_test gdn_v_regroup_cpu_test.cpp)
set_target_properties(gdn_v_regroup_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(gdn_v_regroup_cpu_test PRIVATE ../include)
add_test(NAME gdn_v_regroup_cpu_test COMMAND gdn_v_regroup_cpu_test)

add_executable(hadamard_rotate_gpu_test hadamard_rotate_gpu_test.cpp ../src/prism_hadamard.cpp ../src/gguf.cpp)
set_target_properties(hadamard_rotate_gpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(hadamard_rotate_gpu_test PRIVATE ../include)
target_link_libraries(hadamard_rotate_gpu_test PRIVATE si_quant CUDA::cudart)
add_test(NAME hadamard_rotate_gpu_test COMMAND hadamard_rotate_gpu_test)

# Links the whole runtime rather than si_gemm alone: si_gemm's other translation units reach
# across to si_fused, so pulling it in on its own leaves those references undefined.
add_executable(gemv_ptq1_gpu_test gemv_ptq1_gpu_test.cpp)
set_target_properties(gemv_ptq1_gpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(gemv_ptq1_gpu_test PRIVATE ../include)
target_link_libraries(gemv_ptq1_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME gemv_ptq1_gpu_test COMMAND gemv_ptq1_gpu_test)

# WP-3: 2-rank comm (allreduce/maxreduce) on the 2-GPU pair. Skips (exit 0, repo precedent)
# unless >=2 GPUs and >=96 MiB free VRAM per card - an LLM is resident on this box (lean mode).
add_executable(gpu_link_2gpu_test gpu_link_2gpu_test.cpp)
target_link_libraries(gpu_link_2gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME gpu_link_2gpu_test COMMAND gpu_link_2gpu_test)

# WP-4: the TP gate's validation matrix (exact refusal strings, --tp/--devices parsing, arch
# match helper) -- pure C++ against server/include/tp_plan.hpp; no CUDA, no linkage, GPU-free.
add_executable(tp_config_cpu_test tp_config_cpu_test.cpp)
set_target_properties(tp_config_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME tp_config_cpu_test COMMAND tp_config_cpu_test)

# WP-5: per-card observability rendering (/metrics + /v1/info shape at tp=1 and tp=2) -- pure
# C++ against server/include/gpu_metrics.hpp; no CUDA, no linkage, GPU-free.
add_executable(gpu_metrics_cpu_test gpu_metrics_cpu_test.cpp)
set_target_properties(gpu_metrics_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME gpu_metrics_cpu_test COMMAND gpu_metrics_cpu_test)

# WP-5: the failure policy's bookkeeping (device_health.h: either-card / tp-link fatal -> whole
# engine unhealthy, first reason kept). CUDA headers + cudart symbols only; never calls the driver.
add_executable(device_health_cpu_test device_health_cpu_test.cpp)
target_include_directories(device_health_cpu_test PRIVATE ../include)
target_link_libraries(device_health_cpu_test PRIVATE CUDA::cudart Threads::Threads)
set_target_properties(device_health_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME device_health_cpu_test COMMAND device_health_cpu_test)

# WP-6: the per-tensor TP placement table -- split/disjoint/cover pattern classes, rank
# symmetry, the tp=1 degenerate invariance, the two-tier out-of-pattern fallback, and the 27B
# class-aware on-disk budget audit -- pure C++ against runtime/include/sparkinfer/tp_layout.hpp;
# no CUDA, no linkage, GPU-free.
add_executable(tp_layout_cpu_test tp_layout_cpu_test.cpp)
set_target_properties(tp_layout_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME tp_layout_cpu_test COMMAND tp_layout_cpu_test)

# WP-7: the KV-cache split (2+2 KV heads, per-device pools) -- the head-window tiling
# mirrored from tp_layout's K/V convention, the G2 33,024 B/token per-token figure, the
# 262k/131k per-card fit, the min-across-pools admission rule (pure C++), plus ONE explicit
# few-MB GPU paging run (a few blocks, 2+2 heads, skip-guarded, never auto-sized) -- the
# real KVCacheManager on its 2-head window, per-pool refcounts and prefix restore included.
add_executable(kv_split_cpu_test kv_split_cpu_test.cpp)
target_link_libraries(kv_split_cpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME kv_split_cpu_test COMMAND kv_split_cpu_test)

# WP-8: the GDN `lin_state` per-device window (24/24 v-heads on 27B) -- exact 27B byte figures
# (144.0 + 3.75 MiB per seq, 72.0 + 3.75 MiB per card, conv FULL per card), the window-normalizer
# clamp table (27B block + cyclic), the tp=1 invariance, and cross-window restore rejection
# (pure C++), plus ONE explicit ~16 MiB GPU run: sequential windowed single-AR vs the 8-step full
# reference (bit-identical), the new batched v0/vloc path incl. the unsplit invariance, and the
# compacted-bf16 batched path at bf16 tolerance (16v/8q, 1 slot, 8 tokens; skip-guarded, never
# auto-sized).
add_executable(gdn_split_cpu_test gdn_split_cpu_test.cpp)
target_link_libraries(gdn_split_cpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME gdn_split_cpu_test COMMAND gdn_split_cpu_test)

# WP-6 (CHG-0011, subagent H): the weight loaders consuming the process tp table -- the 27B
# Gguf per-tensor-class placements (3-range GDN qkv/conv, Cols/Rows/OneD, per-rank whole norms,
# visual/unknown fallbacks), the per-rank on-disk budget with the D5 per-card audit lines
# (131k fits, 262k exceeds, vs a 16 GiB card), the tp=1 degenerate invariance (pure C++, always
# run), plus ONE skip-guarded 2-device one-shot: a tiny hybrid Flat model loaded by two
# Qwen35Model instances on two cards -- per-rank allocated bytes vs the table's gather
# arithmetic, values spot-checked against the file bytes the ranges select, print_tp_audit on
# both, and a tp=1 FNV-1a byte-identity read-back (skips by design on boxes that cannot hold
# two contexts, e.g. with an LLM resident).
add_executable(tp_weights_cpu_test tp_weights_cpu_test.cpp)
set_target_properties(tp_weights_cpu_test PROPERTIES CXX_STANDARD 17)
target_link_libraries(tp_weights_cpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME tp_weights_cpu_test COMMAND tp_weights_cpu_test)
