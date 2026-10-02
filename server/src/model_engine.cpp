#include "model_engine.hpp"
#include "sparkinfer/kernels/deterministic.h"
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/device_health.h"

#include "sparkinfer/gguf.h"
#include "sparkinfer/gpu_link.h"
#include "sparkinfer/inference_engine.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/lmcache_bridge_client.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/qwen_vision.h"
#include "sparkinfer/models/qwen_vision_hf_config.h"
#include "sparkinfer/models/qwen_vision_preprocess.h"
#include "sparkinfer/safetensors.h"
#include "image_input.hpp"
#include "video_input.hpp"
#include "tp_plan.hpp"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/runtime.h"
#include "sparkinfer/tp_layout.hpp"

#include "../../runtime/examples/qwen3_gguf_config.h"
#include "../../runtime/examples/qwen38_hf_config.h"

#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

namespace sparkinfer_server {

namespace {

bool prompt_starts_with(const std::vector<int>& prompt, const std::vector<int>& prefix) {
    if (prefix.empty() || prompt.size() < prefix.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), prompt.begin());
}

int batch_tokens_per_step() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_BATCH_TOKENS");
        v = e ? std::max(1, atoi(e)) : 64;
    }
    return v;
}

// LMCache bridge sidecar lifecycle (docs/lmcache_bridge_protocol.md). Off by default -- gated on
// SPARKINFER_LMCACHE_ENABLE=1 -- so this feature ships with zero risk to existing behavior until
// explicitly turned on. All the actual cache logic lives behind Qwen35Model's BridgeClient*
// (null unless attached); this section only owns the sidecar subprocess and the socket path.

bool lmcache_enabled() {
    const char* e = getenv("SPARKINFER_LMCACHE_ENABLE");
    return e && e[0] == '1';
}

std::string lmcache_socket_path() {
    const char* e = getenv("SPARKINFER_LMCACHE_SOCKET");
    return e ? e : "/tmp/sparkinfer_lmcache.sock";
}

// fork+exec the sidecar (python3 bridge/lmcache_bridge.py --socket ... --num-layers ... ...),
// passing the KV layout as CLI args -- see bridge/lmcache_bridge.py's BridgeServer docstring for
// why this is CLI args and not deferred to the first HELLO (engine construction there is ~10s of
// first-time import + setup cost that must happen before any connection is accepted). Returns
// the child pid, or -1 on failure (missing SPARKINFER_LMCACHE_BRIDGE_SCRIPT, fork()/exec()
// failure) -- callers must treat -1 as "sidecar unavailable," never a fatal server error, matching
// the protocol doc's degradation invariant.
pid_t spawn_lmcache_sidecar(const std::string& socket_path, const sparkinfer::Qwen35Config& cfg,
                            const sparkinfer::KVCacheManager& kv, const std::string& model_name) {
    const char* script = getenv("SPARKINFER_LMCACHE_BRIDGE_SCRIPT");
    if (!script || !script[0]) {
        fprintf(stderr, "[sparkinfer-server] SPARKINFER_LMCACHE_ENABLE=1 but "
                        "SPARKINFER_LMCACHE_BRIDGE_SCRIPT is unset -- lmcache disabled\n");
        return -1;
    }
    const char* python_env = getenv("SPARKINFER_LMCACHE_PYTHON");
    const std::string python = python_env && python_env[0] ? python_env : "python3";
    const int elem_bytes = kv.int8_kv() ? 1 : 2;

    std::vector<std::string> args = {
        python,
        script,
        "--socket", socket_path,
        "--instance-id", "sparkinfer",
        "--num-layers", std::to_string(cfg.n_layers),
        "--num-kv-heads", std::to_string(cfg.n_kv_heads),
        "--head-dim", std::to_string(cfg.head_dim),
        "--block-size", std::to_string(kv.block_size()),
        "--elem-bytes", std::to_string(elem_bytes),
        "--model-name", model_name,
    };
    if (kv.int8_kv()) args.push_back("--int8-kv");

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[sparkinfer-server] lmcache sidecar fork() failed: %s\n", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // Child: stdout/stderr inherited from the parent (both point at the same stderr the
        // server itself logs to, per the doc -- no extra log plumbing needed to see sidecar
        // output). This spawn happens before sparkinfer_server ever binds its HTTP listener
        // (load() runs ahead of svr.listen() in main()), so there's no listening socket fd to
        // worry about leaking into the child; the CUDA context/device fds already open by this
        // point (weights are already loaded onto the GPU) are never touched by the child since
        // execvp runs immediately below with no CUDA calls in between, and NVIDIA's driver has
        // set O_CLOEXEC on its own device fds for years specifically to make this pattern safe.
        // execvp only returns on failure.
        execvp(python.c_str(), argv.data());
        fprintf(stderr, "[sparkinfer-server] lmcache sidecar exec failed: %s\n", strerror(errno));
        _exit(127);
    }
    fprintf(stderr, "[sparkinfer-server] lmcache sidecar spawned (pid=%d, socket=%s)\n",
            (int)pid, socket_path.c_str());
    return pid;
}

// SIGTERM + bounded wait (5s) before SIGKILL -- mirrors the HTTP listener's own graceful-
// shutdown grace period so a wedged sidecar never blocks server exit indefinitely.
void terminate_lmcache_sidecar(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 50; i++) {
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) return;
        usleep(100000);
    }
    fprintf(stderr, "[sparkinfer-server] lmcache sidecar did not exit within 5s, sending SIGKILL\n");
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}

}  // namespace

struct ModelEngine::Impl {
    // (tp=2, i2 item 11) The process-wide 2-rank GpuLink: created in load() when R==2 and
    // handed to both rank models non-owning via tp_attach. Declared FIRST so reverse member
    // destruction order makes it outlive every model that points into it -- belt and braces on
    // top of the explicit shutdown in ~Impl (a post-shutdown GpuLink dtor is a documented
    // no-op: the not-ready branch only re-runs the null-guarded best-effort release).
    std::unique_ptr<sparkinfer::GpuLink> tp_link;
    std::string path;
    sparkinfer::Qwen35Config cfg{};
    std::unique_ptr<sparkinfer::Runtime> rt;
    std::unique_ptr<sparkinfer::KVCacheManager> kv;
    std::unique_ptr<sparkinfer::moe::MoEEngine> engine;
    std::unique_ptr<sparkinfer::Qwen35Model> model;
    // Dual-GPU Wave 3 (per-device weight split): at tp>1, rank 0 is `model`/`kv` above (the
    // batch engine's serving instance, pool 0); these hold ranks 1..R-1, each a full
    // Qwen35Model on its own card holding that rank's slice of the weights + its own KV pool.
    // They are load-complete but their forward is numerically incomplete until WP-9/10 (the
    // per-layer all-reduce) -- see the warning printed in load(). `tp_kvs` is declared before
    // `tp_models` so the models (which hold pointers into the pools) are destroyed first.
    std::vector<std::unique_ptr<sparkinfer::KVCacheManager>> tp_kvs;
    std::vector<std::unique_ptr<sparkinfer::Qwen35Model>> tp_models;
    // Owned here, attached to model by pointer; declared before batch_engine so the engine, which
    // drives it, is destroyed first.
    std::unique_ptr<sparkinfer::DFlashDraftModel> draft;
    std::unique_ptr<sparkinfer::ContinuousBatchEngine> batch_engine;
    std::vector<int> prefix_tokens;
    bool ready = false;

    // Automatic prefix cache: see ModelEngine::set_prefix_cache_boundary_token.
    bool prefix_cache_on = false;
    int prefix_cache_boundary_token = -1;
    int prefix_cache_min_tokens = 1024;

    // LMCache bridge (docs/lmcache_bridge_protocol.md): the C++ socket client is owned here
    // (BridgeClient itself is declared in lmcache_bridge_client.h; Qwen35Model only holds a
    // non-owning pointer to it via set_lmcache_bridge()) alongside the sidecar subprocess this
    // ModelEngine spawned for it. Both null/-1 when the feature is disabled or unavailable.
    std::unique_ptr<sparkinfer::BridgeClient> lmcache_bridge;
    pid_t lmcache_sidecar_pid = -1;

    // Vision tower, when the checkpoint ships one. Owned here and handed to the batch engine by
    // pointer, so it must outlive it -- and must be freed while the CUDA context is still alive,
    // which is why reset_vision() is called from ~Impl's BODY rather than left to member order.
    sparkinfer::QwenVisionConfig vcfg{};
    sparkinfer::QwenVisionWeights vweights{};
    bool vision_ready = false;
    int vision_device = -1;   // the card the tower lives on (-1 = the loading thread's current one)

    void reset_vision() {
        if (!vision_ready) return;
        int prev = -1;
        if (vision_device >= 0) { cudaGetDevice(&prev); cudaSetDevice(vision_device); }
        free_qwen_vision_weights(vweights);
        if (prev >= 0) cudaSetDevice(prev);
        vweights = sparkinfer::QwenVisionWeights{};
        vision_ready = false;
    }

    // Explicit teardown order: destroy the BridgeClient first (joins its ping/store threads,
    // which may otherwise be mid-handshake against the sidecar) before killing the sidecar out
    // from under it -- tearing them down in the other order risks those threads observing a
    // closed socket mid-operation instead of a clean, already-stopped state.
    ~Impl() {
        reset_vision();
        lmcache_bridge.reset();
        terminate_lmcache_sidecar(lmcache_sidecar_pid);
        // (tp=2, i2 item 11) Teardown. Pin the original relative member order: batch_engine's
        // dtor can still call into model_/kv_ for unfinished jobs (inference_engine.cpp:120),
        // so it must run while every model is alive — exactly as declaration order guaranteed
        // pre-tp=2. The rank models are gone before the process-wide GpuLink is torn down;
        // the KV pools (tp_kvs) outlive the models that point into them, as before. tp=1 never
        // sets tp_link, so the shutdown is a null no-op and the tp=1 teardown is byte-identical.
        batch_engine.reset();
        draft.reset();
        tp_models.clear();
        tp_kvs.clear();
        model.reset();
        if (tp_link) tp_link->shutdown();
    }
};

ModelEngine::ModelEngine() : impl_(std::make_unique<Impl>()) {}
ModelEngine::~ModelEngine() = default;

bool ModelEngine::load(const std::string& gguf_path, int max_seq) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->ready = false;
    impl_->reset_vision();
    impl_->batch_engine.reset();
    impl_->draft.reset();
    impl_->model.reset();
    impl_->engine.reset();
    impl_->kv.reset();
    impl_->rt.reset();
    impl_->path.clear();
    // A reload (load() called again on an already-loaded engine) must not leak the previous
    // sidecar process or leave a BridgeClient pointing at a model that no longer exists.
    impl_->lmcache_bridge.reset();
    terminate_lmcache_sidecar(impl_->lmcache_sidecar_pid);
    impl_->lmcache_sidecar_pid = -1;

    // Device gate (dual-gpu WP-4): today's "no CUDA device" refusal generalized to the
    // tensor-parallel plan set by --tp/--devices (or the env): enough devices for the
    // requested tp, every requested id present. The tp=1 default validates {0} against the
    // count and produces exactly the legacy refusal; the arch check (warn-only, below) is
    // the third item of the plan's gate and needs CUDA, so it stays here, not in tp_plan.hpp.
    const TpPlan plan{g_tp, g_devices};
    const std::vector<int> eff = effective_devices(plan);

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess) ndev = 0;  // a failed query is "no device"
    {
        const TpValidation v = validate_tp_plan(plan.tp, eff, ndev);
        if (!v.ok) {
            fprintf(stderr, "%s", v.error.c_str());
            return false;
        }
    }
    // (dual-GPU WP-13) The LMCache sidecar stores and reloads one device's whole KV pool; at
    // tp>1 each rank holds only its share of the KV heads in its own pool, so a sidecar would
    // cache half the heads and splice them into both ranks. The dual-GPU KV tier is deferred
    // (v2): refuse the combination at load rather than serve wrong attention.
    // The split paths are written for exactly two ranks (vocab, head and FFN halves, a 2-node
    // link); a larger group would load and then compute with the wrong slices.
    if (plan.tp > 2) {
        fprintf(stderr, "[sparkinfer-server] --tp %d is not supported: tensor parallelism is implemented "
                        "for two cards (--tp 2)\n", plan.tp);
        return false;
    }
    if (plan.tp > 1 && lmcache_enabled()) {
        fprintf(stderr, "[sparkinfer-server] SPARKINFER_LMCACHE_ENABLE=1 is not supported with --tp %d: "
                        "the LMCache sidecar caches a single device's KV pool, and at tp>1 each card "
                        "holds only its share of the KV heads. Unset SPARKINFER_LMCACHE_ENABLE or run "
                        "with --tp 1.\n", plan.tp);
        return false;
    }
    if (plan.tp > 1 || !plan.devices.empty()) {
        // The resolved plan, so the log shows what the operator asked for (the default
        // tp=1/no-devices stays silent, as today).
        std::string ds;
        for (size_t i = 0; i < eff.size(); ++i) {
            if (i) ds += ",";
            ds += std::to_string(eff[i]);
        }
        fprintf(stderr, "[sparkinfer-server] tensor-parallel plan: tp=%d devices=%s (%s)\n",
                plan.tp, ds.c_str(), plan.devices.empty() ? "auto" : "explicit");
    }
    if (eff.size() > 1) {
        // The split is only safe across same-architecture cards; a mismatch across the
        // requested ones is a warning, not a failure -- the plan's gate is count/range, and
        // the layout work keys kernels on each card's cc, so a mixed pair degrades rather
        // than corrupting.
        struct Cc { int dev, major, minor; };
        std::vector<Cc> ccs;
        for (int dev : eff) {
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, dev) == cudaSuccess)
                ccs.push_back({dev, p.major, p.minor});
        }
        for (size_t i = 1; i < ccs.size(); ++i) {
            if (!same_cc(ccs[0].major, ccs[0].minor, ccs[i].major, ccs[i].minor))
                fprintf(stderr,
                        "[sparkinfer-server] WARN: tensor parallelism across different "
                        "architectures: device %d is cc %d.%d, device %d is cc %d.%d (continuing)\n",
                        ccs[0].dev, ccs[0].major, ccs[0].minor,
                        ccs[i].dev, ccs[i].major, ccs[i].minor);
        }
    }

    // Three-way dispatch on what `-m` actually points at: a .gguf file (every model shipped so
    // far), or a directory -- which is either a HuggingFace "compressed-tensors" mixed FP8/NVFP4
    // checkpoint (config.json has a quantization_config block, e.g. unsloth/Qwen3.8-27B-NVFP4) or
    // a plain (unquantized) safetensors checkpoint. Same "auto-detect from what's actually there"
    // pattern the GGUF path already uses (general.architecture sniffing) -- no new CLI flag.
    struct stat path_st{};
    const bool is_dir = stat(gguf_path.c_str(), &path_st) == 0 && S_ISDIR(path_st.st_mode);
    enum class LoadKind { Gguf, CompressedTensors, PlainSafetensors };
    LoadKind kind = LoadKind::Gguf;
    if (is_dir) {
        std::ifstream cf(gguf_path + "/config.json");
        if (!cf) {
            fprintf(stderr, "[sparkinfer-server] %s is a directory but has no config.json\n",
                    gguf_path.c_str());
            return false;
        }
        nlohmann::json root;
        try { cf >> root; } catch (const std::exception& e) {
            fprintf(stderr, "[sparkinfer-server] %s/config.json parse error: %s\n",
                    gguf_path.c_str(), e.what());
            return false;
        }
        kind = root.contains("quantization_config") ? LoadKind::CompressedTensors
                                                      : LoadKind::PlainSafetensors;
    }

    sparkinfer::GGUF g;
    impl_->cfg = sparkinfer::Qwen35Config{};
    if (kind == LoadKind::Gguf) {
        if (!g.open(gguf_path)) {
            fprintf(stderr, "[sparkinfer-server] cannot open %s\n", gguf_path.c_str());
            return false;
        }
        qwen3_config_from_gguf(g, impl_->cfg);
    } else {
        // Both directory kinds share the same base HF config.json shape (text_config block) --
        // quantization_config only changes how load() below reads the actual weight bytes, not
        // the architecture/hyperparameter population.
        std::string err;
        if (!qwen38_config_from_hf_json(gguf_path, impl_->cfg, err)) {
            fprintf(stderr, "[sparkinfer-server] %s: %s\n", gguf_path.c_str(), err.c_str());
            return false;
        }
    }
    if (max_seq > 0) impl_->cfg.max_seq = max_seq;
    else if (impl_->cfg.max_seq < 2048) impl_->cfg.max_seq = 2048;

    fprintf(stderr, "[sparkinfer-server] arch %s, layers=%d, experts=%d top-%d, max_seq=%d\n",
            qwen3_model_label(impl_->cfg), impl_->cfg.n_layers, impl_->cfg.n_experts,
            impl_->cfg.top_k, impl_->cfg.max_seq);

    // TP placement table (dual-gpu WP-6): built from the just-resolved config and
    // plan, published process-wide BEFORE any loader or fit estimate below runs, so
    // every consumer (the weight parsers' inventory cross-checks, the per-rank
    // fit numbers below, and later the per-device allocators in qwen35.cpp)
    // agrees on one name->(rank, axis, ranges) map. At the tp=1 default this is
    // the degenerate table -- every name whole on device 0, no split -- the tp=1
    // invariance, so an unset or single-device load behaves exactly as before.
    {
        const sparkinfer::tp::Conv conv =
            kind == LoadKind::Gguf ? sparkinfer::tp::Conv::Gguf : sparkinfer::tp::Conv::Hf;
        const sparkinfer::tp::Table t =
            sparkinfer::tp::Table::build(impl_->cfg, (int)eff.size(), conv);
        sparkinfer::tp::set_process_table(std::move(t));
        if (eff.size() > 1)
            fprintf(stderr, "[sparkinfer-server] tp placement table: %zu names, %d ranks (%s names)\n",
                    sparkinfer::tp::get_process_table().entries().size(), (int)eff.size(),
                    conv == sparkinfer::tp::Conv::Gguf ? "gguf" : "hf");
        if (kind == LoadKind::Gguf) {
            // g is the live, opened GGUF (above): table<->file both ways.
            sparkinfer::tp::log_gguf_tp_inventory(g, std::cerr);
        } else {
            // The directory kinds: open the shard set only to enumerate names for
            // the one-way cross-check (mmap of headers, no weight reads). A layout
            // the reader cannot open is a warning here -- the real load outcome is
            // decided by the dispatch below, never by the cross-check.
            sparkinfer::SafeTensorsModel m;
            if (m.open(gguf_path))
                sparkinfer::tp::log_safetensors_tp_inventory(m, std::cerr);
            else
                fprintf(stderr, "[sparkinfer-server] tp inventory: could not open safetensors "
                                "shards in %s (cross-check skipped)\n", gguf_path.c_str());
        }
    }

    // The runtime is created with the resolved plan: at the tp=1 default this is
    // value-identical to the legacy default config (device 0, single row), and
    // initialize() fills the per-device property table.
    sparkinfer::RuntimeConfig rcfg;
    rcfg.tp = plan.tp;
    rcfg.devices = eff;
    impl_->rt = sparkinfer::Runtime::create(rcfg);
    impl_->rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = impl_->cfg.n_layers;
    kvc.num_kv_heads = impl_->cfg.n_kv_heads;
    kvc.head_dim = impl_->cfg.head_dim;
    kvc.block_size = 16;
    { const char* e = getenv("SPARKINFER_KV_INT8");
      // Muse Glimmer: int8 KV cache is a confirmed correctness bug, not a precision tradeoff --
      // incoherent output from the very first decode token (#779), root-caused to its per-layer
      // sliding-window/NoPE alternation + sandwich-norm activations not matching what the int8
      // quantize/dequantize kernels were tuned against (Qwen3.6, same cfg.hybrid=true, is
      // unaffected). The CLI tools never caught this because their short eval prompts (<4096
      // tokens) always fell under the bf16 threshold below; the server activates int8 off its
      // configured max_seq (there's no per-request length at KV-pool-init time), and the default
      // max_seq (4096) satisfies ">=4096" unconditionally, so every default-config Muse Glimmer
      // server silently served garbage. Default to bf16 until the kernel bug itself is fixed;
      // SPARKINFER_KV_INT8=1 still force-enables it for anyone debugging that fix.
      kvc.int8_kv = e ? (e[0] != '0')
                      : (impl_->cfg.muse_glimmer ? false
                         : impl_->cfg.hybrid ? (impl_->cfg.max_seq >= 4096) : true); }
    // SPARKINFER_KV_DTYPE=bf16|int8|fp8|nvfp4 picks the KV element format explicitly and wins over
    // SPARKINFER_KV_INT8 (which keeps working: unset KV_DTYPE = the int8/bf16 choice above). fp8
    // (e4m3, per-head fp16 scale; int8's layout) and nvfp4 (e2m1 + e4m3 per-16 block scales) are
    // wired through the Qwen3.5/3.6/3.8 hybrid attention paths only (decode, tp=2 rows, batched
    // prefill); anything else falls back to the default with a warning.
    kvc.kv_dtype = kvc.int8_kv ? sparkinfer::KV_INT8 : sparkinfer::KV_BF16;
    if (const char* e = getenv("SPARKINFER_KV_DTYPE"); e && e[0]) {
        const std::string v(e);
        int want = -1;
        if (v == "bf16") want = sparkinfer::KV_BF16;
        else if (v == "int8") want = sparkinfer::KV_INT8;
        else if (v == "fp8" || v == "e4m3") want = sparkinfer::KV_FP8;
        else if (v == "nvfp4" || v == "fp4") want = sparkinfer::KV_NVFP4;
        else fprintf(stderr, "[sparkinfer-server] SPARKINFER_KV_DTYPE=%s not understood "
                             "(bf16|int8|fp8|nvfp4) -- keeping %s\n", e,
                     sparkinfer::kv_dtype_name(kvc.kv_dtype));
        const bool fp_ok = impl_->cfg.hybrid && !impl_->cfg.muse_glimmer &&
                           impl_->cfg.head_dim % 16 == 0;
        if ((want == sparkinfer::KV_FP8 || want == sparkinfer::KV_NVFP4) && !fp_ok) {
            fprintf(stderr, "[sparkinfer-server] SPARKINFER_KV_DTYPE=%s is only wired for the "
                            "Qwen3.5/3.6/3.8 hybrid models -- keeping %s\n", e,
                    sparkinfer::kv_dtype_name(kvc.kv_dtype));
            want = -1;
        }
        if (want >= 0) kvc.kv_dtype = want;
    }
    kvc.int8_kv = kvc.kv_dtype == sparkinfer::KV_INT8;
    if (kvc.kv_dtype >= sparkinfer::KV_FP8 && lmcache_enabled())
        fprintf(stderr, "[sparkinfer-server] warning: lmcache staging does not know the %s KV "
                        "layout\n", sparkinfer::kv_dtype_name(kvc.kv_dtype));
    // Only the full-attention layers get a pool slot. The Gated-DeltaNet layers of a hybrid model
    // carry a recurrent state and never read paged KV, so a slot for them is pure waste -- on
    // Qwen3.8-27B that is 16 slots of 64, i.e. the pool was 4x larger than the model can use.
    // Every example main (qwen3_gguf_bench, qwen3_gguf_prefill_check, ...) has always set this;
    // the server never did, so the process that actually serves traffic was the one paying for it.
    // Left empty for non-hybrid models, where hybrid_kv_layer_slots gives every layer a slot
    // anyway and the behaviour is unchanged.
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(impl_->cfg.n_layers, impl_->cfg.hybrid,
                                                       impl_->cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, impl_->cfg.n_layers);
    // PER-DEVICE KV BUDGET (dual-gpu WP-7, the 2+2 split): at tp>1 this manager is rank 0's
    // pool, which holds only rank 0's KV-head window -- n_kv_heads/tp by the tp table's
    // head_window convention (contiguous window, last rank the remainder) -- so the per-block
    // elements and the budget below are this card's share, never a single-card total. The
    // tp=1 default is all heads, i.e. the old formula byte-identically (and kvc keeps
    // kv_head_count = 0, the manager's "all heads" pool).
    const int kvh_rank = (plan.tp > 1 && impl_->cfg.n_kv_heads > 0)
        ? (std::max(1, impl_->cfg.n_kv_heads / plan.tp))
        : impl_->cfg.n_kv_heads;
    if (plan.tp > 1) { kvc.kv_head_start = 0; kvc.kv_head_count = kvh_rank; }
    const size_t epb = (size_t)16 * kvh_rank * impl_->cfg.head_dim;
    const size_t blocks = (size_t)impl_->cfg.max_seq / 16 + 8;
    // pool_bytes is a bf16-DENOMINATED BUDGET, not an allocation: KVCacheManager derives
    // total_blocks = pool_bytes / (n_slots * 2 * bf16_bytes_per_block) and then mallocs at the
    // real element width (int8 just mallocs less). So the `* 2` here is the bf16 element size and
    // is correct as written -- it must stay even when int8_kv is on, or capacity halves. What has
    // to match kvc.layer_slot is the SLOT COUNT: passing n_layers while the manager counts 16
    // slots would hand out 4x the blocks for the same memory rather than shrinking the pool.
    // At tp>1 the budget is this device's (per the rank's head window above); the other ranks'
    // pools in Wave 3 H size the same way from their own windows.
    impl_->kv = std::make_unique<sparkinfer::KVCacheManager>(
        kvc, (size_t)kvL * 2 * epb * 2 * blocks);

    // Reports the slot count actually used, and the resident bytes rather than the bf16 budget --
    // the old line multiplied by n_layers (all 64) and by 2 regardless of int8, so it overstated
    // a hybrid int8 pool by 8x and was the number anyone sizing a deployment would have read.
    fprintf(stderr, "[sparkinfer-server] kv_cache: dtype=%s int8=%d slots=%d/%d blocks=%zu resident=%.2f GiB\n",
            sparkinfer::kv_dtype_name(kvc.kv_dtype), kvc.int8_kv ? 1 : 0, kvL, impl_->cfg.n_layers,
            blocks,
            (double)kvL * 2.0 * (double)sparkinfer::kv_dtype_bytes(kvc.kv_dtype, epb) * blocks
                / (1024.0 * 1024.0 * 1024.0));
    if (plan.tp > 1)
        fprintf(stderr,
                " (per-device: this rank's pool holds %d of %d KV heads; admission is the min "
                "across pools)\n",
                kvh_rank, impl_->cfg.n_kv_heads);

    // Per-rank NVFP4 fit estimates (dual-gpu WP-6): a pure-arithmetic mirror of
    // the two real per-device gates in qwen35.cpp, printed per rank so a plan can
    // be eyeballed before the first token. Nothing here allocates or fails.
    //
    //   Gguf kind + muse_glimmer: the Muse prefill-fp4 preflight
    //     (qwen35.cpp:7143-7156,7187) -- per rank, the qkvg / wo / down operand
    //     shapes at that rank's q/kv head windows (the q width includes the
    //     folded gate, exactly as the real preflight's 2*qdim_a term), with the
    //     per-layer and whole-model byte totals and the hpp fit mirrors
    //     nvfp4_supported(12,0,128,n,k) (shape + cc 12.0 only; the sm_120a
    //     BUILD requirement and the VRAM budget stay in qwen35.cpp).
    //
    //   Hf kind + qwen38: the compressed-tensors lm_head keep
    //     (qwen35.cpp:8185-8221) -- per rank, the vocab rows that rank owns and
    //     the keep need at that window (plus the full-head need the current
    //     replicated keep pays), against the SPARKINFER_Q38_HEAD_NVFP4_RESERVE_MB
    //     reserve (3072 MB default). The actual keep/decline (free-VRAM
    //     dependent) still runs in the load path.
    if (eff.size() > 1) {
        const sparkinfer::Qwen35Config& fc = impl_->cfg;
        const int R = (int)eff.size();
        const size_t H = (size_t)fc.hidden, F = (size_t)fc.moe_ffn;
        if (kind == LoadKind::Gguf && fc.muse_glimmer) {
            for (int r = 0; r < R; ++r) {
                const size_t qh  = (size_t)fc.n_q_heads * (r + 1) / R - (size_t)fc.n_q_heads * r / R;
                const size_t kvh = (size_t)fc.n_kv_heads * (r + 1) / R - (size_t)fc.n_kv_heads * r / R;
                const size_t qd = qh * fc.head_dim, kd = kvh * fc.head_dim;
                cudaDeviceProp dp{};
                const bool have_cc = cudaGetDeviceProperties(&dp, eff[r]) == cudaSuccess;
                const size_t qkvg = sparkinfer::tp::nvfp4_muse_qkvg_layer(qd, kd, H);
                const size_t wo   = sparkinfer::tp::nvfp4_muse_wo_layer(qd, H);
                const size_t down = sparkinfer::tp::nvfp4_muse_down_layer(H, F);
                fprintf(stderr,
                        "[sparkinfer-server] tp fit rank%d (dev %d, cc %d.%d): per layer "
                        "qkvg %zu MiB (fit %d), wo %zu MiB (fit %d), down %zu MiB (fit %d); "
                        "model wants qkvg %zu MiB, wo %zu MiB, down %zu MiB over %d layers\n",
                        r, eff[r], have_cc ? dp.major : -1, have_cc ? dp.minor : -1,
                        qkvg / (1024 * 1024),
                        sparkinfer::tp::nvfp4_supported(12, 0, 128, 2 * qd + 2 * kd, H) ? 1 : 0,
                        wo / (1024 * 1024),
                        sparkinfer::tp::nvfp4_supported(12, 0, 128, H, qd) ? 1 : 0,
                        down / (1024 * 1024),
                        sparkinfer::tp::nvfp4_supported(12, 0, 128, H, F) ? 1 : 0,
                        (size_t)fc.n_layers * qkvg / (1024 * 1024),
                        (size_t)fc.n_layers * wo / (1024 * 1024),
                        (size_t)fc.n_layers * down / (1024 * 1024),
                        fc.n_layers);
            }
        } else if (kind != LoadKind::Gguf && fc.qwen38) {
            const size_t V = (size_t)fc.vocab;
            const size_t need_full = sparkinfer::tp::nvfp4_head_keep_need(V, H);
            const char* e = getenv("SPARKINFER_Q38_HEAD_NVFP4_RESERVE_MB");
            const long rv = e ? atol(e) : 3072;
            const size_t reserve_mb = (rv < 0) ? 0 : (size_t)rv;
            fprintf(stderr,
                    "[sparkinfer-server] tp head: the full (replicated) keep needs %zu MiB "
                    "+ %zu MiB reserve; split per rank it is:\n",
                    need_full / (1024 * 1024), reserve_mb);
            for (int r = 0; r < R; ++r) {
                const size_t rows = V * (size_t)(r + 1) / R - V * (size_t)r / R;
                fprintf(stderr,
                        "  rank%d (dev %d): vocab rows %zu, keep need %zu MiB\n",
                        r, eff[r], rows,
                        sparkinfer::tp::nvfp4_head_keep_need(rows, H) / (1024 * 1024));
            }
            fprintf(stderr,
                    "[sparkinfer-server] tp head: the actual keep/decline (free-VRAM dependent) "
                    "still runs in qwen35.cpp's compressed-tensors load (SPARKINFER_Q38_HEAD_NVFP4)\n");
        }
    }

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = impl_->cfg.n_experts;
    mc.top_k = impl_->cfg.top_k;
    mc.hidden_dim = impl_->cfg.hidden;
    mc.ffn_dim = impl_->cfg.moe_ffn;
    mc.num_layers = impl_->cfg.n_layers;
    impl_->engine = sparkinfer::moe::MoEEngine::create(mc);

    // Wave 3 (per-device weight split): rank 0's instance binds to eff[0] and owns rank 0's
    // GDN v-head window (the leading block, heads [0, v/R)). At the tp=1 default eff == {0} and
    // the window is the degenerate all-heads one, so this is byte-identical to the unsplit model.
    {
        const int R0 = (int)eff.size();
        const int vfull = impl_->cfg.linear_v_heads;
        sparkinfer::GdnStateWindow win0;
        if (plan.tp > 1 && vfull > 0)
            win0 = {0, vfull / R0};   // rank 0 = [0, vfull/R0)
        impl_->model = std::make_unique<sparkinfer::Qwen35Model>(
            impl_->cfg, impl_->kv.get(), impl_->engine.get(), win0, 0, eff[0]);
    }

    if (kind == LoadKind::Gguf) {
        fprintf(stderr, "[sparkinfer-server] loading GGUF ...\n");
        if (!impl_->model->load_gguf(gguf_path)) {
            fprintf(stderr, "[sparkinfer-server] load_gguf failed\n");
            return false;
        }
    } else if (kind == LoadKind::CompressedTensors) {
        fprintf(stderr, "[sparkinfer-server] loading compressed-tensors checkpoint ...\n");
        if (!impl_->model->load_compressed_tensors(gguf_path)) {
            fprintf(stderr, "[sparkinfer-server] load_compressed_tensors failed\n");
            return false;
        }
    } else {
        // Qwen35Model::load_weights() reads a directory of already-converted flat .bin files
        // (runtime/tools/convert_qwen35.py's own offline output format), not a safetensors
        // directory directly -- no C++ loader for plain (unquantized) safetensors exists yet, this
        // config-detection branch was written ahead of that loader rather than left undetectable.
        fprintf(stderr, "[sparkinfer-server] %s: plain safetensors directories are not yet "
                        "supported directly -- convert with runtime/tools/convert_qwen35.py first "
                        "and pass the .bin output directory instead\n", gguf_path.c_str());
        return false;
    }

    // (dual-GPU D5) Per-card budget audit for the rank-0 instance now that its load succeeded:
    // the table's on-disk bytes for this rank, this card's free/total, and the per-card estimate
    // at 128k context vs a 16 GiB card (fits). The tp=1 default (degenerate table) never prints.
    if (plan.tp > 1)
        impl_->model->print_tp_audit(131072);

    // Wave 3 (per-device weight split): build ranks 1..R-1, each on its own card with its own KV
    // pool (sized from the rank's contiguous head window, last rank the remainder) and its own GDN
    // v-head window, each loading its own slice of the weights from the same checkpoint. The batch
    // engine above keeps instance 0 / pool 0 as the serving path; these extra instances are
    // load-complete but their forward is numerically incomplete until the WP-9/10 per-layer
    // all-reduce (Wave 4 I). tp=1 (or a single effective device) never enters this block.
    if (plan.tp > 1 && eff.size() > 1) {
        const int R = (int)eff.size();
        const int nkv = impl_->cfg.n_kv_heads;
        const int vfull = impl_->cfg.linear_v_heads;
        for (int r = 1; r < R; ++r) {
            // One-time bind of this rank to its card before anything it allocates runs.
            if (cudaSetDevice(eff[r]) != cudaSuccess) {
                fprintf(stderr, "[sparkinfer-server] tp rank%d: cudaSetDevice(%d) failed\n",
                        r, eff[r]);
                return false;
            }
            // This rank's KV head window (contiguous; last rank takes the remainder).
            const int kv_start = nkv > 0 ? nkv * r / R : 0;
            int kv_count = nkv > 0 ? nkv * (r + 1) / R - kv_start : 0;
            kv_count = std::max(1, kv_count);
            sparkinfer::KVCacheConfig kc = kvc;   // layers/slots/int8 identical to rank 0
            kc.kv_head_start = kv_start;
            kc.kv_head_count = kv_count;
            const size_t epb_r = (size_t)16 * (size_t)kv_count * (size_t)impl_->cfg.head_dim;
            const size_t blocks_r = (size_t)impl_->cfg.max_seq / 16 + 8;
            impl_->tp_kvs.emplace_back(std::make_unique<sparkinfer::KVCacheManager>(
                kc, (size_t)kvL * 2 * epb_r * 2 * blocks_r));
            // This rank's GDN v-head state window: [vfull*r/R, vfull*(r+1)/R).
            sparkinfer::GdnStateWindow win;
            win.v_start = vfull > 0 ? vfull * r / R : 0;
            win.v_count = vfull > 0 ? vfull * (r + 1) / R - win.v_start : 0;
            auto mr = std::make_unique<sparkinfer::Qwen35Model>(
                impl_->cfg, impl_->tp_kvs.back().get(), impl_->engine.get(), win, r, eff[r]);
            const bool ok =
                (kind == LoadKind::Gguf) ? mr->load_gguf(gguf_path)
                : (kind == LoadKind::CompressedTensors) ? mr->load_compressed_tensors(gguf_path)
                                                        : false;
            if (!ok) {
                fprintf(stderr, "[sparkinfer-server] tp rank%d weight load failed\n", r);
                return false;
            }
            impl_->tp_models.emplace_back(std::move(mr));
            fprintf(stderr, "[sparkinfer-server] tp rank%d (dev %d): KV heads [%d,%d), GDN v [%d,%d) loaded\n",
                    r, eff[r], kv_start, kv_start + kv_count, win.v_start,
                    win.v_start + win.v_count);
            // (dual-GPU D5) This rank's per-card budget-audit line (same 128k-context estimate).
            impl_->tp_models.back()->print_tp_audit(131072);
        }
        // Restore the default device so any post-load host-side work (lmcache, vision) runs as
        // before; the per-instance binds already happened in each ctor.
        cudaSetDevice(eff[0]);
        if (R == 2) {
            // (i2 item 11) Wire the 27B tp=2 split-weight forward: one process-wide GpuLink.
            // Rank 0 (impl_->model, the batch engine's serving instance) is the leader: it
            // gets rank 1's model and mirrors every op onto it (one fused all-reduce per
            // layer). Rank 1 (impl_->tp_models[0]) is handed nothing -- it only runs on the
            // leader's threads and never issues a link op itself.
            impl_->tp_link = std::make_unique<sparkinfer::GpuLink>();
            // max_bytes 512 MiB: the 1 MiB default sits below the ~168 MB prefill
            // all-reduce at m=16k, so every large op would be refused by the byte budget.
            if (!impl_->tp_link->init(eff[0], eff[1], sparkinfer::GpuLink::Transport::Auto,
                512 * 1024 * 1024)) {
                fprintf(stderr,
                        "[sparkinfer-server] tp=2: GpuLink init failed (dev %d/%d)\n",
                        eff[0], eff[1]);
                return false;
            }
            impl_->model->tp_attach(impl_->tp_link.get(), 0, { impl_->tp_models[0].get() });
            impl_->tp_models[0]->tp_attach(impl_->tp_link.get(), 1, {});
            fprintf(stderr,
                    "[sparkinfer-server] tp=2: GpuLink attached (rank0 dev %d, rank1 dev %d); "
                    "split-weight forward active\n",
                    eff[0], eff[1]);
        } else {
            // R>2: an explicit --devices list is not tied to tp and not capped at 2, so a
            // >=3-rank load is reachable in principle on a >=3-device box. tp is capped at 2
            // by the model layer, so no >2 semantics are invented here -- a non-2-rank load
            // keeps the pending behavior (no link, no attach) instead of half-attaching.
            fprintf(stderr,
                    "[sparkinfer-server] tp>1: %d of %d instances loaded; split-weight forward is "
                    "incomplete until WP-9/10 (Wave 4 I)\n",
                    (int)impl_->tp_models.size() + 1, R);
        }
    }

    if (lmcache_enabled()) {
        const std::string socket_path = lmcache_socket_path();
        impl_->lmcache_sidecar_pid =
            spawn_lmcache_sidecar(socket_path, impl_->cfg, *impl_->kv, gguf_path);
        if (impl_->lmcache_sidecar_pid > 0) {
            sparkinfer::BridgeKVLayout layout;
            layout.num_layers = impl_->cfg.n_layers;
            layout.num_kv_heads = impl_->cfg.n_kv_heads;
            layout.head_dim = impl_->cfg.head_dim;
            layout.block_size = impl_->kv->block_size();
            layout.int8_kv = impl_->kv->int8_kv();
            layout.elem_bytes = impl_->kv->int8_kv() ? 1 : 2;
            layout.model_name = gguf_path;
            // Constructing BridgeClient does not block on the sidecar being ready yet (it
            // connects/handshakes lazily on first use, respecting its own timeout budget) --
            // safe to do immediately after fork() even though the sidecar's own ~10s cold-start
            // engine build is still running in the background.
            impl_->lmcache_bridge =
                std::make_unique<sparkinfer::BridgeClient>(socket_path, layout);
            impl_->model->set_lmcache_bridge(impl_->lmcache_bridge.get());
        }
        // lmcache_sidecar_pid <= 0 (spawn failure, e.g. missing SPARKINFER_LMCACHE_BRIDGE_SCRIPT)
        // leaves lmcache_bridge null -- the model's every lookup/store call site treats that as
        // "no cache tier," never a load failure. lmcache_enabled() being on with a broken sidecar
        // config is a misconfiguration worth the stderr line above, not a reason to refuse to serve.
    }

    sparkinfer::SchedulePolicy policy = sparkinfer::SchedulePolicy::CONTINUOUS_BATCHING;
    if (const char* p = getenv("SPARKINFER_SCHED_POLICY")) {
        // "chunked" / "chunked_prefill" → CHUNKED_PREFILL; "priority" → PRIORITY;
        // "continuous" (default) → CONTINUOUS_BATCHING. Match full keywords so
        // "continuous" is not misread as chunked (both start with 'c').
        if (strncmp(p, "chunk", 5) == 0) policy = sparkinfer::SchedulePolicy::CHUNKED_PREFILL;
        else if (p[0] == 'p' || p[0] == 'P') policy = sparkinfer::SchedulePolicy::PRIORITY;
        else policy = sparkinfer::SchedulePolicy::CONTINUOUS_BATCHING;
    }
    impl_->batch_engine = std::make_unique<sparkinfer::ContinuousBatchEngine>(
        impl_->model.get(), impl_->kv.get(), batch_tokens_per_step(), policy);
    // (dual-GPU WP-9b) The engine's own block ops (shared prefix session, prefix-cache retains,
    // truncation, frees) go to rank 1's identically sized pool too, keeping one block numbering.
    if (impl_->tp_link && !impl_->tp_kvs.empty())
        impl_->batch_engine->set_kv_mirror(impl_->tp_kvs[0].get());

    // Automatic prefix cache. Chat and agent clients resend the whole conversation every turn; this
    // is what stops the server recomputing it. Bounded three ways -- entries, pinned host memory
    // for recurrent-state snapshots, and half the KV pool -- and evicted least-recently-used,
    // including on demand when a new request cannot get KV blocks.
    {
        auto env_int = [](const char* name, long long dflt) {
            const char* e = getenv(name);
            return e ? atoll(e) : dflt;
        };
        const char* on = getenv("SPARKINFER_PREFIX_CACHE");
        const bool wanted = !(on && on[0] == '0');
        if (wanted && sparkinfer::deterministic_mode()) {
            fprintf(stderr, "[sparkinfer-server] prefix cache: off (SPARKINFER_DETERMINISTIC=1 -- a "
                            "request's output may not depend on what earlier requests cached)\n");
        } else if (wanted) {
            sparkinfer::PrefixCache::Limits lim;
            lim.max_entries = (size_t)std::max(1LL, env_int("SPARKINFER_PREFIX_CACHE_ENTRIES", 32));
            lim.max_host_bytes = (size_t)std::max(0LL, env_int("SPARKINFER_PREFIX_CACHE_HOST_MB", 8192)) << 20;
            lim.max_blocks = impl_->kv->num_total_blocks() / 2;
            impl_->prefix_cache_min_tokens =
                (int)std::max(1LL, env_int("SPARKINFER_PREFIX_CACHE_MIN_TOKENS", 1024));
            impl_->batch_engine->enable_prefix_cache(lim);
            impl_->prefix_cache_on = true;
            fprintf(stderr, "[sparkinfer-server] prefix cache: on (%zu entries, %zu MiB host, %d of %d "
                            "KV blocks, checkpoints from %d tokens)\n",
                    lim.max_entries, lim.max_host_bytes >> 20, lim.max_blocks,
                    impl_->kv->num_total_blocks(), impl_->prefix_cache_min_tokens);
        } else {
            fprintf(stderr, "[sparkinfer-server] prefix cache: off (SPARKINFER_PREFIX_CACHE=0)\n");
        }
    }

    // Vision tower. Absence is NOT an error -- a text-only checkpoint has no vision_config and
    // has_vision() stays false -- but a tower that is present and fails to load is reported here
    // rather than left to surface as a confusing per-request failure much later.
    // SPARKINFER_VISION=0 skips the tower: it sits on the first card (~1 GB), which at tp=2 with a
    // DSpark draft is the card that runs out of room first.
    const char* vision_env = getenv("SPARKINFER_VISION");
    const bool vision_off = vision_env && vision_env[0] == '0';
    if (vision_off) fprintf(stderr, "[sparkinfer-server] vision tower: off (SPARKINFER_VISION=0)\n");
    if (is_dir && !vision_off) {
        std::string verr;
        if (!qwen_vision_config_from_hf_json(gguf_path, impl_->vcfg, verr)) {
            fprintf(stderr, "[sparkinfer-server] vision config malformed: %s\n", verr.c_str());
        } else if (impl_->vcfg.present) {
            sparkinfer::SafeTensorsModel vst;
            if (!vst.open(gguf_path)) {
                fprintf(stderr, "[sparkinfer-server] vision: cannot open safetensors\n");
            } else if (![&] {
                           // (dual-GPU WP-13) At tp>1 the tower goes on the LAST card: card 0 also
                           // carries the DSpark draft, and the ranks' weights are otherwise
                           // balanced, so card 0 is the one that runs out of room first. Its
                           // output is host floats, so nothing crosses devices.
                           // SPARKINFER_VISION_DEVICE=n picks a card explicitly.
                           const char* vd = getenv("SPARKINFER_VISION_DEVICE");
                           impl_->vision_device = vd ? atoi(vd)
                                                     : (plan.tp > 1 && eff.size() > 1 ? eff.back() : -1);
                           int prev = -1;
                           if (impl_->vision_device >= 0) {
                               cudaGetDevice(&prev);
                               cudaSetDevice(impl_->vision_device);
                           }
                           const bool ok = load_qwen_vision_weights(vst, impl_->vcfg, impl_->vweights, verr);
                           if (prev >= 0) cudaSetDevice(prev);
                           return ok;
                       }()) {
                fprintf(stderr, "[sparkinfer-server] vision tower load failed: %s\n", verr.c_str());
            } else {
                impl_->vision_ready = true;
                impl_->batch_engine->set_vision(&impl_->vweights, &impl_->vcfg, impl_->vision_device);
                fprintf(stderr, "[sparkinfer-server] vision tower ready: %d blocks, out_hidden=%d%s\n",
                        impl_->vcfg.depth, impl_->vcfg.out_hidden,
                        impl_->vision_device >= 0
                            ? (" (device " + std::to_string(impl_->vision_device) + ")").c_str() : "");
            }
        }
    }

    impl_->path = gguf_path;
    impl_->ready = true;
    fprintf(stderr, "[sparkinfer-server] continuous batching enabled (policy=%d, batch=%d)\n",
            (int)policy, batch_tokens_per_step());
    fprintf(stderr, "[sparkinfer-server] model ready: %s\n", gguf_path.c_str());
    return true;
}

bool ModelEngine::loaded() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready;
}

std::string ModelEngine::model_path() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->path;
}

int ModelEngine::eos_id() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready ? impl_->cfg.eos_id : -1;
}

bool ModelEngine::device_healthy() const { return !sparkinfer::device_lost(); }

std::string ModelEngine::unhealthy_reason() const {
    const sparkinfer::DeviceLostInfo info = sparkinfer::device_lost_info();
    if (!info.lost) return std::string();
    return info.reason.empty() ? std::string("unrecoverable device error") : info.reason;
}

int ModelEngine::unhealthy_device() const {
    const sparkinfer::DeviceLostInfo info = sparkinfer::device_lost_info();
    return info.lost ? info.device : -1;
}

int ModelEngine::tp_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!impl_->ready || !impl_->rt) return 1;
    return 1 + (int)impl_->tp_models.size();
}

std::string ModelEngine::link_transport() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!impl_->tp_link || !impl_->tp_link->is_ready()) return std::string();
    return impl_->tp_link->transport() == sparkinfer::GpuLink::Transport::P2pMapped
               ? "p2p-mapped"
               : "pinned-staging";
}

std::vector<GpuRow> ModelEngine::gpu_rows() const {
    // Snapshot the rank -> device table and the KV pool counts under the lock, then sample the
    // cards OUTSIDE it: an NVML/cudaMemGetInfo query is milliseconds and must not hold up the
    // request path. query_gpu_stats saves/restores this (HTTP) thread's current device only, so
    // it cannot disturb the ranks' own threads.
    std::vector<GpuRow> rows;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!impl_->ready || !impl_->rt) return rows;
        const int n = impl_->rt->device_count();
        for (int i = 0; i < n; ++i) {
            const sparkinfer::GpuDeviceInfo d = impl_->rt->device_info((size_t)i);
            GpuRow r;
            r.rank = i;
            r.device = d.id;
            r.name = d.name;
            // Rank 0's pool is the serving pool; ranks 1.. hold their head-window pools.
            const sparkinfer::KVCacheManager* kv =
                (i == 0) ? impl_->kv.get()
                         : ((size_t)(i - 1) < impl_->tp_kvs.size() ? impl_->tp_kvs[i - 1].get()
                                                                   : nullptr);
            if (kv) {
                r.kv_free_blocks = kv->num_free_blocks();
                r.kv_total_blocks = kv->num_total_blocks();
            }
            rows.push_back(std::move(r));
        }
    }
    for (GpuRow& r : rows) {
        const sparkinfer::GpuStats s = sparkinfer::query_gpu_stats(r.device);
        r.valid = s.valid;
        r.temp_c = s.temp_c;
        r.power_w = s.power_w;
        r.sm_clock_mhz = s.sm_clock_mhz;
        r.util_pct = s.util_pct;
        r.vram_used_bytes = s.vram_used_bytes;
        r.vram_total_bytes = s.vram_total_bytes;
    }
    return rows;
}

bool ModelEngine::is_stop_token(int token_id) const {
    if (!impl_ || !impl_->ready || token_id < 0) return false;
    return token_id == impl_->cfg.eos_id ||
           (impl_->cfg.eos_id2 >= 0 && token_id == impl_->cfg.eos_id2);
}

int ModelEngine::vocab() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready ? impl_->cfg.vocab : 0;
}

int ModelEngine::max_seq() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready ? impl_->cfg.max_seq : 0;
}

bool ModelEngine::is_museglimmer() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready && impl_->cfg.muse_glimmer;
}

bool ModelEngine::has_vision() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->vision_ready;
}

bool ModelEngine::prepare_images(const std::vector<std::string>& urls, int image_token_id,
                                 std::vector<int>& prompt_ids, PreparedImages& out,
                                 std::string& err) const {
    out.src_images.clear();
    out.src_videos.clear();
    out.images.clear();
    out.positions.clear();
    if (urls.empty()) return true;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!impl_->vision_ready) {
            err = "this model has no vision tower; image input is not supported";
            return false;
        }
    }
    // vcfg is written only by load() and read-only afterwards, and the tower weights are not
    // touched here at all -- this does CPU preprocessing only, so it runs without the engine
    // mutex and without contending for the GPU.
    for (size_t i = 0; i < urls.size(); i++) {
        const std::string where = "image " + std::to_string(i);
        std::vector<unsigned char> bytes;
        if (!parse_image_url(urls[i], bytes, err)) { err = where + ": " + err; return false; }
        DecodedImage img;
        if (!decode_image(bytes.data(), bytes.size(), img, err)) { err = where + ": " + err; return false; }
        PreparedImages::Image pi;
        int gh = 0, gw = 0;
        auto pixels = std::make_shared<std::vector<float>>();
        if (!sparkinfer::qwen_vision_preprocess(img.rgb.data(), img.height, img.width, impl_->vcfg,
                                                *pixels, &gh, &gw, err)) {
            err = where + ": " + err;
            return false;
        }
        pi.pixels = std::move(pixels);
        pi.grid_h = gh;
        pi.grid_w = gw;
        out.src_images.push_back(std::move(pi));
    }
    // One placeholder per image goes in, the count each grid needs comes out. A mismatch here is
    // an error, never a best-effort expansion: guessing would hand the model a prompt whose image
    // span is quietly truncated or padded, which it will describe fluently either way.
    return reexpand_images(image_token_id, prompt_ids, out, err);
}

bool ModelEngine::prepare_vision(const std::vector<std::string>& image_urls,
                                 const std::vector<std::string>& video_urls,
                                 int image_token_id, int video_token_id,
                                 int vision_start_token_id, int vision_end_token_id,
                                 const VideoSampling& sampling,
                                 const std::function<std::vector<int>(const std::string&)>& tokenize,
                                 std::vector<int>& prompt_ids, PreparedImages& out,
                                 std::string& err) const {
    out.src_images.clear();
    out.src_videos.clear();
    out.images.clear();
    out.positions.clear();
    if (image_urls.empty() && video_urls.empty()) return true;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!impl_->vision_ready) {
            err = "this model has no vision tower; image and video input are not supported";
            return false;
        }
    }
    if (!video_urls.empty() && !tokenize) {
        err = "video input requires a tokenizer for its timestamp markers";
        return false;
    }

    // Everything below is CPU-only preprocessing -- no tower weights are touched -- so it runs
    // without the engine mutex and without contending for the GPU.
    for (size_t i = 0; i < image_urls.size(); i++) {
        const std::string where = "image " + std::to_string(i);
        std::vector<unsigned char> bytes;
        if (!parse_image_url(image_urls[i], bytes, err)) { err = where + ": " + err; return false; }
        DecodedImage img;
        if (!decode_image(bytes.data(), bytes.size(), img, err)) { err = where + ": " + err; return false; }
        PreparedImages::Image pi;
        int gh = 0, gw = 0;
        auto pixels = std::make_shared<std::vector<float>>();
        if (!sparkinfer::qwen_vision_preprocess(img.rgb.data(), img.height, img.width, impl_->vcfg,
                                                *pixels, &gh, &gw, err)) {
            err = where + ": " + err;
            return false;
        }
        pi.pixels = std::move(pixels);
        pi.grid_h = gh;
        pi.grid_w = gw;
        out.src_images.push_back(std::move(pi));
    }

    for (size_t i = 0; i < video_urls.size(); i++) {
        const std::string where = "video " + std::to_string(i);
        if (!video_decoder_available(nullptr)) {
            err = where + ": video input needs ffmpeg and ffprobe on PATH";
            return false;
        }
        std::vector<unsigned char> bytes;
        if (!parse_video_url(video_urls[i], bytes, err)) { err = where + ": " + err; return false; }
        DecodedVideo clip;
        const int max_frames = sampling.max_frames > 0 ? sampling.max_frames : kDefaultMaxVideoFrames;
        if (!decode_video(bytes.data(), bytes.size(), max_frames, sampling.fps, clip, err)) {
            err = where + ": " + err;
            return false;
        }

        std::vector<const unsigned char*> frame_ptrs;
        frame_ptrs.reserve(clip.frames.size());
        for (const auto& f : clip.frames) frame_ptrs.push_back(f.data());

        std::vector<float> pixels;
        int gt = 0, gh = 0, gw = 0;
        if (!sparkinfer::qwen_vision_preprocess_video(frame_ptrs.data(), (int)frame_ptrs.size(),
                                                      clip.height, clip.width, impl_->vcfg,
                                                      pixels, &gt, &gh, &gw, err)) {
            err = where + ": " + err;
            return false;
        }

        PreparedImages::Video pv;
        pv.tokens_per_frame = sparkinfer::qwen_vision_num_tokens(gh, gw, impl_->vcfg);
        if (pv.tokens_per_frame <= 0) {
            err = where + ": video grid does not divide into the spatial merge block";
            return false;
        }

        // Slice the clip's one contiguous buffer into per-group buffers. Each group is an
        // ordinary tower call, and Image owns its pixels, so the copy buys a uniform contract
        // with the image path rather than a second offset-aware code path in the tower.
        const size_t per_group = pixels.size() / (size_t)(gt > 0 ? gt : 1);
        pv.groups.reserve((size_t)gt);
        for (int g = 0; g < gt; g++) {
            PreparedImages::Image gi;
            auto buf = std::make_shared<std::vector<float>>(
                pixels.begin() + (size_t)g * per_group,
                pixels.begin() + (size_t)(g + 1) * per_group);
            gi.pixels = std::move(buf);
            gi.grid_h = gh;
            gi.grid_w = gw;
            pv.groups.push_back(std::move(gi));
        }

        const std::vector<float> ts = sparkinfer::qwen_vision_video_timestamps(
            clip.frame_indices, clip.fps, impl_->vcfg.temporal_patch);
        pv.timestamp_tokens.reserve(pv.groups.size());
        for (size_t g = 0; g < pv.groups.size(); g++) {
            // Reference format: one decimal, e.g. "<1.5 seconds>". snprintf rather than
            // std::to_string, which is locale-independent here but fixed at six decimals.
            char buf[64];
            const float t = g < ts.size() ? ts[g] : 0.0f;
            std::snprintf(buf, sizeof(buf), "<%.1f seconds>", (double)t);
            pv.timestamp_tokens.push_back(tokenize(buf));
        }
        out.src_videos.push_back(std::move(pv));
    }

    return reexpand_vision(image_token_id, video_token_id,
                           vision_start_token_id, vision_end_token_id, prompt_ids, out, err);
}

bool ModelEngine::reexpand_images(int image_token_id, std::vector<int>& prompt_ids,
                                  PreparedImages& io, std::string& err) const {
    return reexpand_vision(image_token_id, impl_->vcfg.video_token_id,
                           impl_->vcfg.vision_start_token_id, impl_->vcfg.vision_end_token_id,
                           prompt_ids, io, err);
}

bool ModelEngine::reexpand_vision(int image_token_id, int video_token_id,
                                  int vision_start_token_id, int vision_end_token_id,
                                  std::vector<int>& prompt_ids, PreparedImages& io,
                                  std::string& err) const {
    io.positions.clear();
    io.images.clear();
    if (io.src_images.empty() && io.src_videos.empty()) return true;

    // VIDEOS FIRST, then images. Each expansion scans for its own placeholder id and preserves
    // every other token, so the two are independent -- but doing videos first keeps the image
    // expansion working on a prompt whose video spans are already their final length, which is
    // what makes the single ordering walk below correct.
    if (!io.src_videos.empty()) {
        std::vector<sparkinfer::QwenVideoSpan> spans;
        spans.reserve(io.src_videos.size());
        for (const auto& v : io.src_videos) {
            sparkinfer::QwenVideoSpan sp;
            sp.tokens_per_frame = v.tokens_per_frame;
            sp.timestamp_tokens = v.timestamp_tokens;
            spans.push_back(std::move(sp));
        }
        std::vector<int> expanded;
        if (!sparkinfer::qwen_vision_expand_video_placeholders(
                prompt_ids, video_token_id, vision_start_token_id, vision_end_token_id,
                spans, expanded, err))
            return false;
        prompt_ids.swap(expanded);
    }
    if (!io.src_images.empty()) {
        std::vector<int> counts;
        counts.reserve(io.src_images.size());
        for (const auto& im : io.src_images)
            counts.push_back(sparkinfer::qwen_vision_num_tokens(im.grid_h, im.grid_w, impl_->vcfg));
        std::vector<int> expanded;
        if (!sparkinfer::qwen_vision_expand_placeholders(prompt_ids, image_token_id, counts,
                                                         expanded, err))
            return false;
        prompt_ids.swap(expanded);
    }

    // Walk the finished prompt once and emit tower units in the order their placeholders appear.
    // Each MAXIMAL run of one placeholder id is one unit: an image's run is its whole grid, and a
    // video group's run is bounded by the vision_end/timestamp tokens between groups. Ordering by
    // the walk -- rather than concatenating images-then-videos -- is what makes an interleaved
    // request correct, because the engine pairs images[i] with positions[i] by index alone.
    size_t next_image = 0, next_video = 0, next_group = 0;
    for (size_t i = 0; i < prompt_ids.size(); ) {
        const int id = prompt_ids[i];
        if (id != image_token_id && id != video_token_id) { i++; continue; }
        size_t j = i;
        while (j < prompt_ids.size() && prompt_ids[j] == id) {
            io.positions.push_back((int)j);
            j++;
        }
        if (id == image_token_id) {
            if (next_image >= io.src_images.size()) {
                err = "prompt carries more image spans than images were preprocessed";
                return false;
            }
            io.images.push_back(io.src_images[next_image++]);
        } else {
            if (next_video >= io.src_videos.size() ||
                next_group >= io.src_videos[next_video].groups.size()) {
                err = "prompt carries more video frame spans than frames were preprocessed";
                return false;
            }
            io.images.push_back(io.src_videos[next_video].groups[next_group++]);
            if (next_group == io.src_videos[next_video].groups.size()) {
                next_video++;
                next_group = 0;
            }
        }
        i = j;
    }
    if (next_image != io.src_images.size() || next_video != io.src_videos.size()) {
        err = "prompt carries fewer vision spans than were preprocessed";
        return false;
    }

    // MRoPE positions, computed from the SAME walk-ordered unit list so the spans cannot drift
    // out of step with the placeholder runs they describe.
    io.mrope_pos.clear();
    io.mrope_decode_offset = 0;
    // SPARKINFER_MROPE=0 falls back to the 1D positions used before MRoPE existed. Kept because
    // MRoPE is a BEHAVIOURAL change to already-shipped image handling, not a pure addition: an
    // operator who sees image answers move after an upgrade needs a way to attribute it, and an
    // A/B on one build is the only way to measure the effect on output at all.
    static const bool mrope_enabled = [] {
        const char* e = getenv("SPARKINFER_MROPE");
        return !(e && e[0] == '0');
    }();
    if (impl_->cfg.mrope() && mrope_enabled) {
        std::vector<sparkinfer::QwenVisionSpanGrid> grids;
        grids.reserve(io.images.size());
        // grid_t is 1 per unit: an image is one temporal group, and the reference splits a video's
        // grid into one t=1 row PER GROUP, which is exactly how the prompt spans them too.
        for (const auto& im : io.images) grids.push_back({1, im.grid_h, im.grid_w});
        if (!sparkinfer::qwen_vision_mrope_positions(prompt_ids, image_token_id, video_token_id,
                                                     grids, impl_->vcfg.spatial_merge, 0,
                                                     io.mrope_pos, err))
            return false;
        // The last token's rotary position + 1 is where decode resumes; the difference from the
        // prompt length is the constant every later decode step applies.
        if (!prompt_ids.empty()) {
            const int last_t = io.mrope_pos[(prompt_ids.size() - 1) * 3];
            io.mrope_decode_offset = (last_t + 1) - (int)prompt_ids.size();
        }
    }
    return true;
}

bool ModelEngine::is_qwen38() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->ready && impl_->cfg.qwen38;
}

void ModelEngine::set_prefix_tokens(const std::vector<int>& tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->prefix_tokens = tokens;
}

int ModelEngine::prefix_token_len() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)impl_->prefix_tokens.size();
}

CompletionResult ModelEngine::complete(const std::vector<int>& prompt_ids, int max_new_tokens) {
    return complete_streaming(prompt_ids, max_new_tokens, nullptr);
}

int ModelEngine::active_requests() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->num_active() : 0;
}

int ModelEngine::waiting_requests() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->num_waiting() : 0;
}

uint64_t ModelEngine::admission_waits() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->admission_waits() : 0;
}

uint64_t ModelEngine::admission_timeouts() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->admission_timeouts() : 0;
}

int ModelEngine::free_kv_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->num_free_kv_blocks() : 0;
}

int ModelEngine::max_queue_depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (impl_->ready && impl_->batch_engine) ? impl_->batch_engine->max_queue_depth() : 0;
}

bool ModelEngine::load_draft(const std::string& dir, std::string& err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!impl_->ready || !impl_->model || !impl_->batch_engine) {
        err = "load the target model before the draft";
        return false;
    }
    if (!impl_->cfg.qwen38) {
        err = "speculative decoding is supported for Qwen3.8-27B targets only";
        return false;
    }
    const char* e = getenv("SPARKINFER_DSPARK_MAX_CTX");
    sparkinfer::DFlashDraftConfig dcfg;
    dcfg.max_seq = std::min(impl_->cfg.max_seq, e ? std::max(1024, atoi(e)) : 16384);
    // At tp>1 the quantized copies are built right below anyway; building them per layer during
    // the load keeps its peak near the steady state on the draft's card (the bf16 copies of every
    // layer were all resident at once before, ~2.5 GB on DSpark, and that peak -- not the ~2 GB
    // the draft holds afterwards -- was what capped --ctx).
    dcfg.eager_quant = !impl_->tp_models.empty();
    auto draft = std::make_unique<sparkinfer::DFlashDraftModel>(dcfg);
    if (!draft->load(dir)) {
        // The usual cause is device memory, not the checkpoint: the target's KV pool is sized for
        // the whole --ctx before the draft loads, and at --ctx 262144 a 32 GB card has no room
        // left for it (#1086).
        err = "cannot load a DSpark draft from " + dir + " at --ctx " +
              std::to_string(impl_->cfg.max_seq) +
              " -- if the log above shows CUDA out-of-memory errors, lower --ctx (131072 fits a 32 GB card)";
        return false;
    }
    // At tp>1 the card holding the draft is the tight one, and the quantized copies (~1.2 GB) are
    // otherwise built lazily by the first speculative request -- where running out of memory
    // takes the request (and the tp group) down instead of failing here, cleanly, at load.
    if (!impl_->tp_models.empty()) {
        draft->ensure_quant();
        // Free-memory floor on every card after the draft is in: below it the server loads but
        // cannot open a session (each holds ~74 MB of GDN state per card on the 27B) or fit a
        // prefill window -- every request would fail with "device out of memory". Measured on
        // 2x 16 GB at tp=2: --ctx 49152 leaves enough, 65536 does not.
        const char* mf = getenv("SPARKINFER_DSPARK_MIN_FREE_MB");
        const long min_free_mb = mf ? std::max(0L, atol(mf)) : 256;
        auto tight_card = [&]() -> int {
            std::vector<int> devs{impl_->model->tp_rank_view().device};
            for (auto& m : impl_->tp_models) devs.push_back(m->tp_rank_view().device);
            int prev = -1;
            cudaGetDevice(&prev);
            int tight = -1;
            for (int d : devs) {
                size_t fb = 0, tb = 0;
                if (d >= 0 && cudaSetDevice(d) == cudaSuccess && cudaMemGetInfo(&fb, &tb) == cudaSuccess &&
                    (long)(fb >> 20) < min_free_mb) {
                    fprintf(stderr, "[sparkinfer-server] DSpark at tp>1: device %d has %zu MiB free after "
                                    "loading the draft (floor %ld MiB)\n", d, fb >> 20, min_free_mb);
                    tight = d;
                }
            }
            if (prev >= 0) cudaSetDevice(prev);
            return tight;
        };
        if (!draft->quant_ok() || !impl_->model->reserve_tp_verify() || tight_card() >= 0) {
            err = "cannot fit the DSpark draft (quantized weights + tp verify scratch + working headroom) next to the target at --ctx " +
                  std::to_string(impl_->cfg.max_seq) + " with --tp " +
                  std::to_string((int)impl_->tp_models.size() + 1) + " -- lower --ctx";
            return false;
        }
    }
    impl_->model->set_dflash_draft(draft.get());
    impl_->draft = std::move(draft);
    impl_->batch_engine->enable_speculative(true);
    fprintf(stderr, "[sparkinfer-server] speculative decoding: DSpark draft %s (block %d, draft context %d)\n",
            dir.c_str(), impl_->draft->config().block_size, impl_->draft->config().max_seq);
    return true;
}

ModelEngine::SpeculativeStats ModelEngine::speculative_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    SpeculativeStats out;
    if (!impl_->draft || !impl_->batch_engine) return out;
    const auto s = impl_->batch_engine->speculative_stats();
    out.enabled = true;
    out.runs = s.runs;
    out.tokens = s.tokens;
    out.handoffs = s.handoffs;
    out.tier_stops = s.tier_stops;
    return out;
}

bool ModelEngine::speculative() const {
    std::lock_guard<std::mutex> lock(mu_);
    return impl_->draft != nullptr;
}

void ModelEngine::set_prefix_cache_boundary_token(int token_id) {
    std::lock_guard<std::mutex> lock(mu_);
    impl_->prefix_cache_boundary_token = token_id;
}

ModelEngine::PrefixCacheStats ModelEngine::prefix_cache_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrefixCacheStats out;
    if (!impl_->ready || !impl_->batch_engine || !impl_->prefix_cache_on) return out;
    const sparkinfer::PrefixCache::Stats s = impl_->batch_engine->prefix_cache_stats();
    out.enabled = true;
    out.lookups = s.lookups;
    out.hits = s.hits;
    out.tokens_reused = s.tokens_reused;
    out.inserts = s.inserts;
    out.evictions = s.evictions;
    out.entries = s.entries;
    out.host_bytes = s.host_bytes;
    out.blocks = s.blocks;
    return out;
}

ModelEngine::LMCacheStats ModelEngine::lmcache_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    LMCacheStats stats;
    if (!impl_->lmcache_bridge) return stats;  // enabled=false, both counts 0
    stats.enabled = true;
    stats.lookup_hits = impl_->lmcache_bridge->lookup_hit_count();
    stats.lookup_misses = impl_->lmcache_bridge->lookup_miss_count();
    return stats;
}

CompletionResult ModelEngine::complete_streaming(const std::vector<int>& prompt_ids,
                                                 int max_new_tokens,
                                                 const std::function<bool(int)>& on_token,
                                                 float temperature, uint64_t seed,
                                                 int top_k, float top_p,
                                                 float presence_penalty, float frequency_penalty,
                                                 const std::vector<std::pair<int, float>>& logit_bias,
                                                 bool logprobs, int top_logprobs,
                                                 const std::function<void(const TokenLogprob&)>&
                                                     on_token_logprob,
                                                 const std::vector<int>& forced_tokens,
                                                 const PreparedImages* images,
                                                 std::shared_ptr<sparkinfer::TokenConstraint> constraint) {
    CompletionResult out;
    sparkinfer::ContinuousBatchEngine::Request req;
    req.prompt = prompt_ids;
    req.max_new_tokens = max_new_tokens;
    req.forced_tokens = forced_tokens;
    if (images && !images->mrope_pos.empty()) {
        // Carried whenever the checkpoint declares an mrope_section, independently of whether this
        // particular request has images: the positions describe every token, and a prompt with no
        // vision span still gets the (degenerate, all-axes-equal) values, which cost nothing and
        // keep one code path instead of two.
        req.mrope_pos = images->mrope_pos;
        req.mrope_decode_offset = images->mrope_decode_offset;
    }
    if (images && !images->positions.empty()) {
        req.vision_pos = images->positions;
        req.vision_images.reserve(images->images.size());
        for (const auto& im : images->images)
            req.vision_images.push_back({im.pixels, im.grid_h, im.grid_w});   // shares the buffer
    }
    req.temperature = temperature;
    req.seed = seed;
    req.top_k = top_k;
    req.top_p = top_p;
    req.presence_penalty = presence_penalty;
    req.frequency_penalty = frequency_penalty;
    req.logit_bias = logit_bias;
    req.constraint = std::move(constraint);
    req.logprobs = logprobs;
    req.top_logprobs = top_logprobs;

    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!impl_->ready || !impl_->model || !impl_->batch_engine) {
            out.error = "model not loaded";
            return out;
        }
        // A lost CUDA context (device_health.h) is permanent. Answer before any device work: the
        // prefix handling below can clear the prefix cache, which destroys graphs and frees KV
        // against a context that can no longer service them. /health is already 503.
        if (sparkinfer::device_lost()) {
            out.error = "cuda context lost (unrecoverable device error) -- restart required";
            // (WP-5) Name the first fatal event (which card, or the tp=2 link).
            const sparkinfer::DeviceLostInfo info = sparkinfer::device_lost_info();
            if (!info.reason.empty()) out.error += " [" + info.reason + "]";
            out.alloc_failed = true;   // 503, permanent until restart
            return out;
        }
        if (prompt_ids.empty()) {
            out.error = "empty prompt";
            return out;
        }
        if (max_new_tokens <= 0) {
            out.error = "max_new_tokens must be positive";
            return out;
        }
        if (!forced_tokens.empty()) {
            if ((int)forced_tokens.size() != max_new_tokens) {
                out.error = "forced_tokens size must equal max_new_tokens";
                return out;
            }
            for (int t : forced_tokens) {
                if (t < 0 || t >= impl_->cfg.vocab) {
                    out.error = "forced token id out of range: " + std::to_string(t);
                    return out;
                }
            }
        }
        if ((int)prompt_ids.size() + max_new_tokens > impl_->cfg.max_seq) {
            out.error = "prompt + max_tokens exceeds context limit (" +
                        std::to_string(impl_->cfg.max_seq) + ")";
            fprintf(stderr, "[sparkinfer-server] context overflow: prompt=%zu max_new=%d max_seq=%d\n",
                    prompt_ids.size(), max_new_tokens, impl_->cfg.max_seq);
            return out;
        }

        // Shared prefix KV (session 0) is only safe when no other request is in-flight.
        //
        // Never used for a teacher-forced score. Taking the prefix path prefills the shared
        // prefix on its own (batched prefill at M = prefix_len) and only the suffix per request,
        // versus one batched pass at M = prompt_len otherwise -- a different tile decomposition,
        // so a few ULP of difference in the KV the score is computed against. Which branch runs
        // depends on whether another request happened to be in flight (prefix_exclusive), so with
        // a prefix configured the same /v1/score call could return marginally different logprobs
        // depending on server load. A scoring endpoint exists to be compared against another copy
        // of itself; co-tenancy-dependent numerics defeat that, and the prefill it saves is not
        // worth it.
        const bool prefix_match = forced_tokens.empty() && !impl_->prefix_tokens.empty() &&
                                  prompt_starts_with(prompt_ids, impl_->prefix_tokens);
        const bool prefix_exclusive = impl_->batch_engine->num_active() == 0;
        if (prefix_match && prefix_exclusive) {
            if (impl_->model->prefix_cached_len() != (int)impl_->prefix_tokens.size()) {
                if (!impl_->model->cache_prefix(impl_->prefix_tokens)) {
                    out.error = "cache_prefix failed (KV alloc or batched prefill)";
                    fprintf(stderr, "[sparkinfer-server] %s\n", out.error.c_str());
                    return out;
                }
            }
            req.prefill_start = (int)impl_->prefix_tokens.size();
            req.use_prefix_session = true;
        } else {
            // clear_prefix_cache() frees whatever session is currently active on the shared
            // Qwen35Model (kv->free(active_seq_id)) -- but the continuous-batch worker thread
            // calls activate_session() for whichever job it's stepping right now, on its own
            // thread, independent of this mutex. Calling clear here without the same
            // prefix_exclusive guard the "use prefix" branch above already has meant an
            // unrelated new request (any request that doesn't match the prefix -- the common
            // case whenever no prefix is configured at all) could free the KV blocks out from
            // under an actively-decoding, completely unrelated job. Reproduced directly: under
            // concurrent load this corrupts the KV cache ("[kv] copy block table: an illegal
            // memory access was encountered", poisoning the CUDA context for the rest of the
            // process). Skipping the clear when non-exclusive just defers it -- cache_prefix()
            // already re-primes correctly the next time the prefix session is actually used.
            if (!prefix_match && prefix_exclusive) impl_->model->clear_prefix_cache();
            req.prefill_start = 0;
            req.use_prefix_session = false;
            // Automatic prefix cache. Never for a teacher-forced score (its numerics must not depend
            // on what another request cached) or for images/video (the cache keys on token ids,
            // and every image's placeholder tokens are the same ids).
            const bool has_vision = images && !images->positions.empty();
            if (impl_->prefix_cache_on && forced_tokens.empty() && !has_vision) {
                req.prefix_cache = true;
                if (impl_->prefix_cache_boundary_token >= 0) {
                    // Two checkpoints: the FIRST turn boundary past the minimum -- for a chat, the
                    // end of the system prompt, which other conversations share -- and the LAST,
                    // the start of the final assistant turn, where this conversation's next request
                    // stops matching. The engine skips any a cache hit already covers.
                    const int bs = impl_->kv->block_size();
                    int first = -1, last = -1;
                    for (int i = 0; i < (int)prompt_ids.size(); ++i) {
                        if (prompt_ids[i] != impl_->prefix_cache_boundary_token) continue;
                        const int ckpt = i / bs * bs;
                        if (ckpt < impl_->prefix_cache_min_tokens) continue;
                        if (first < 0) first = ckpt;
                        last = ckpt;
                    }
                    if (first >= 0) req.cache_checkpoints.push_back(first);
                    if (last > first) req.cache_checkpoints.push_back(last);
                }
            }
        }
    }

    // complete_streaming releases the engine mutex above so other HTTP workers can enqueue.
    // Errors must travel with this stack frame — a shared last_error_ slot would let one
    // request clear or observe another request's failure under concurrency.
    //
    // Pass nullptr straight through (not a lambda that internally no-ops) when the caller didn't
    // ask for logprobs -- ContinuousBatchEngine::step_job()'s cost gate is
    // `req.logprobs && on_token_logprob`, so an always-non-null glue lambda here would defeat the
    // "logprobs=false costs nothing extra" property this whole design depends on.
    std::function<void(const sparkinfer::Qwen35Model::TokenLogprob&)> glue_logprob;
    if (on_token_logprob) {
        glue_logprob = [&on_token_logprob](const sparkinfer::Qwen35Model::TokenLogprob& tl) {
            TokenLogprob mirrored;
            mirrored.token_id = tl.token_id;
            mirrored.logprob = tl.logprob;
            mirrored.top_alternatives = tl.top_alternatives;
            on_token_logprob(mirrored);
        };
    }
    auto result = impl_->batch_engine->complete_streaming(req, on_token, glue_logprob);

    std::lock_guard<std::mutex> lock(mu_);
    out.overloaded = result.overloaded;
    out.alloc_failed = result.alloc_failed;
    out.timed_out = result.timed_out;
    out.cancelled = result.cancelled;
    out.internal_error = result.internal_error;
    out.reached_token_limit = result.reached_token_limit;
    out.ttft_ms = result.ttft_ms;
    out.generation_ms = result.generation_ms;
    out.decode_tps = result.decode_tps;
    out.cached_tokens = result.cached_tokens;
    if (!result.error.empty()) {
        out.error = result.error;
        fprintf(stderr, "[sparkinfer-server] %s\n", out.error.c_str());
        return out;
    }
    if (result.cancelled) {
        out.tokens = std::move(result.tokens);
        return out;
    }

    // Only an error that kills the context fails a request the engine completed.
    // cudaGetLastError returns the last error ANY call raised, and an allocation failure another
    // path already handled -- a prefill scratch arena falling back to windows, another request's
    // session refused -- is not this request's: it turned 117 finished requests into
    // "cuda error after decode: out of memory" under concurrent long prompts (#1088).
    cudaError_t e = cudaGetLastError();
    if (sparkinfer::is_unrecoverable(e)) {
        out.error = std::string("cuda error after decode: ") + cudaGetErrorString(e);
        fprintf(stderr, "[sparkinfer-server] %s\n", out.error.c_str());
        return out;
    }
    if (result.tokens.empty() && max_new_tokens > 0)
        out.error = "generate returned no tokens (KV alloc failure?)";
    out.tokens = std::move(result.tokens);
    return out;
}

}  // namespace sparkinfer_server
