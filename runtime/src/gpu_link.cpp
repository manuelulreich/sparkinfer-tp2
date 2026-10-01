// WP-3: the 2-rank comm for tensor-parallel 2, on this 2x-RTX-5060-Ti pair. The contract —
// and the R8 verdict on why this is hand-rolled rather than NCCL — lives in
// include/sparkinfer/gpu_link.h (the only place NCCL may be mentioned in this module).
//
// Transports (resolved at init, logged to stdout):
//
//   p2p-mapped (the fast path here): each rank pulls the peer's buffer into its own out
//   (or into the module's private scratch when in==out) with a stream-scoped D2D copy —
//   the plain cudaMemcpyAsync(D2D) on the destination stream that WP-1's p2p_probe.cpp
//   proved on this pair, both directions — then runs its local reduce kernel. No host
//   memory on the path at all.
//
//   pinned-staging (the fallback): each rank copies its input to its own pinned host
//   buffer (D2H on its own stream) and records its own event. Both D2Hs are fully posted
//   before any wait, so no wait can ever target an unrecorded event. Then each rank waits
//   on the peer's event (cross-context — verified), H2D-copies the peer's pinned buffer
//   into its own out/scratch, and runs the same local reduce kernel. No host
//   synchronization anywhere in the op path.
//
// The kernel half is identical across transports: out_r = in_r op (peer data already
// landed on this rank); the scratch route exists for BOTH because the aliased in-place
// case (in==out) must never have the kernel read a source it is about to clobber.
//
// Scratch: each rank owns one max_bytes cudaMalloc scratch on its own device. It is a plain
// allocation, not a cudaMemPool one: a copy into a pool allocation from the other device's
// context is rejected (pool memory is not mapped by cudaDeviceEnablePeerAccess), which broke
// every in-place op. Shutdown frees it in that rank's context.
// cudaSetDevice is confined to init/shutdown (save/restore); the op path is 100%
// (device, stream)-pure: stream-scoped async copies, cross-context event waits, and
// stream-scoped kernel launches.
//
// Peer access is deliberately never disabled (contexts outlive the object; a re-init
// re-enables cleanly — cudaErrorPeerAccessAlreadyEnabled is treated as success).
//
// Driver-workaround notes (CUDA 13.4, this PHB pair) — all repro'd with /tmp probes, see the
// code sites:
//   1. The FIRST P2P D2D copy of ~>=512 KiB posted after peer access is enabled is delivered
//      with scattered holes in ~the 1 MiB size class (the destination's pre-copy contents show
//      through; the copy "completes" normally): the first such copy into a never-landed 1 MiB
//      destination failed 5/5 in the module shape and 3/3 in the harness shape, while every
//      later copy in the process (same or fresh buffers) was clean. Priming with raw copies to
//      OTHER buffers does not clear it, so the op path itself posts a one-time priming copy on
//      the real (src, dst) pair for each not-yet-primed destination (gated by size and a
//      small per-link cache: steady-state ops pay a single copy, exactly as before).
//   2. cudaDeviceEnablePeerAccess can queue cudaErrorPeerAccessAlreadyEnabled (704) even when
//      it returns success, and a later post (kernel launch, copy) can mis-attribute that stale
//      704 to itself ("reduce kernel on rank A: peer access is already enabled" failures seen
//      in (b)/(c)/(e) on the first real-test run). A cudaGetLastError() drain after every
//      enable in init keeps a stale 704 out of the op path.
//   3. The same first-landing window can also REJECT posts at post time: the process's
//      first ~1 MiB-class P2P post (and, in one observed run, the staging H2D post too) can
//      return cudaErrorInvalidValue, and the rejection LATCHES through the process's early
//      phase — it persisted across re-posts spaced 10 ms apart for the whole first test
//      subtest (12 observed runs: always the first subtest only, every 512 KiB-class copy and
//      every post-window landing clean). A rejected post is safe to re-post (it executes
//      nothing, the destination is untouched, the re-post is the same stream-ordered copy), so
//      post_copy_retry() below re-posts with a fixed exponential backoff (10 ms .. 640 ms;
//      8 posts, ~1.23 s worst-case stall, error path only). If the latch outlives that
//      horizon the op fails HONESTLY — it returns false, nothing executed, buffers untouched —
//      and the next op into the same buffers lands clean. The module never delivers corrupted
//      data and never stalls unboundedly.

#include "sparkinfer/gpu_link.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <cstdlib>

namespace sparkinfer {

namespace {

#define GLINK_LOG(...)          \
  do {                          \
    std::printf(__VA_ARGS__);   \
    std::fflush(stdout);        \
  } while (0)

// Save/restore the current device (gpu_stats.cpp's save/restore convention, RAII'd). The
// restore re-queries the current device and only restores when it is still the one we set —
// never clobbering a later caller's choice.
struct DeviceScope {
  int saved = -1;
  explicit DeviceScope(int dev) {
    cudaGetDevice(&saved);
    if (saved != dev) cudaSetDevice(dev);
  }
  ~DeviceScope() {
    int cur = -1;
    if (cudaGetDevice(&cur) == cudaSuccess && cur != saved) cudaSetDevice(saved);
  }
};

// Driver-workaround 3: an async-copy post can be REJECTED by the 13.4 first-landing window
// (cudaErrorInvalidValue, see the file-top note) — and a rejected post executes nothing, so
// re-posting the identical (dst, src, kind, stream) is safe and stream-ordered. Bounded retry
// with fixed exponential backoff (10 ms .. 640 ms; 8 attempts total, ~1.23 s worst-case stall,
// paid ONLY on the driver-error path): on this box the rejection latches through the process's
// early phase and outlives the horizon — in that case the op fails HONESTLY (returns false,
// nothing executed, the buffers are untouched) and the next op into the same buffers lands
// clean (every post-window landing measured across 12 test runs was bit-exact). Each re-post is
// logged; the final failure, if any, is logged by the caller's fail(). The steady path (no
// driver error) returns after one post, at zero cost.
constexpr int kPostRetries = 7;  // up to 8 total attempts
static const int kRepostDelayUs[kPostRetries] = {10'000, 20'000, 40'000, 80'000, 160'000,
                                                  320'000, 640'000};

cudaError_t post_copy_retry(const char* op, const char* what, void* dst, const void* src,
                             size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  cudaError_t e = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  for (int attempt = 2; e != cudaSuccess && attempt <= kPostRetries + 1; ++attempt) {
    GLINK_LOG("[gpu_link] %s: %s: re-posting (attempt %d/%d)\n", op, what, attempt,
              kPostRetries + 1);
    std::this_thread::sleep_for(std::chrono::microseconds(kRepostDelayUs[attempt - 2]));
    e = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  }
  return e;
}

// The four per-rank resources in a file-scope aggregate so the init/shutdown choreography
// below can live in free functions (GpuLink::Impl is a private nested type and its name is
// not addressable from a non-member function).
struct RankResources {
  cudaEvent_t event = nullptr;
  void* pinned = nullptr;
  void* scratch = nullptr;
  bool any() const { return event != nullptr || pinned != nullptr || scratch != nullptr; }
};

// Per-rank resource setup, in this rank's device context.
bool setup_rank_resources(int dev, RankResources& rs, GpuLink::Transport t, size_t max_bytes) {
  {
    DeviceScope scope(dev);
    // A trivial allocation establishes this device's primary context in this process; the
    // event and scratch below live in that context.
    void* tmp = nullptr;
    if (cudaMalloc(&tmp, 64) != cudaSuccess) {
      GLINK_LOG("[gpu_link] init: dev %d: context-establishing allocation failed: %s\n",
                dev, cudaGetErrorString(cudaGetLastError()));
      return false;
    }
    cudaFree(tmp);
  }
  {
    DeviceScope scope(dev);
    if (cudaEventCreate(&rs.event) != cudaSuccess) {
      GLINK_LOG("[gpu_link] init: dev %d: cudaEventCreate failed: %s\n", dev,
                cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  if (t == GpuLink::Transport::PinnedStaging) {
    // Pinned host staging buffer (host API: no device context involved).
    if (cudaHostAlloc(&rs.pinned, max_bytes, cudaHostAllocDefault) != cudaSuccess) {
      GLINK_LOG("[gpu_link] init: dev %d: cudaHostAlloc(%zu B) failed: %s\n", dev, max_bytes,
                cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  {
    DeviceScope scope(dev);
    // The scratch is a plain cudaMalloc, NOT a cudaMemPool allocation: pool memory is mapped
    // only into its own device's context (cudaDeviceEnablePeerAccess does not extend to it),
    // and a cudaMemcpyAsync is validated in the CALLING thread's current context - so a copy
    // into the non-current rank's pool scratch was rejected with cudaErrorInvalidValue on
    // every in-place op (both transports). cudaMalloc memory is UVA-addressable from any context.
    if (cudaMalloc(&rs.scratch, max_bytes) != cudaSuccess) {
      GLINK_LOG("[gpu_link] init: dev %d: cudaMalloc(%zu B) scratch failed: %s\n", dev, max_bytes,
                cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  return true;
}

// Tear one rank's resources down, in that rank's context. With `gate` set (a real
// shutdown), every free must succeed — a failed free is reported, not hidden.
bool release_rank_resources(int dev, const RankResources& rs, bool gate) {
  if (!rs.any()) return true;
  bool ok = true;
  DeviceScope scope(dev);
  if (rs.scratch) {
    // The caller has drained every stream it posted ops on (shutdown's documented contract).
    if (cudaFree(rs.scratch) != cudaSuccess) {
      GLINK_LOG("[gpu_link] shutdown: dev %d: cudaFree(scratch) failed: %s\n", dev,
                cudaGetErrorString(cudaGetLastError()));
      ok = false;
    }
  }
  if (rs.pinned) {
    if (cudaFreeHost(rs.pinned) != cudaSuccess) {
      GLINK_LOG("[gpu_link] shutdown: dev %d: cudaFreeHost(pinned) failed: %s\n", dev,
                cudaGetErrorString(cudaGetLastError()));
      ok = false;
    }
  }
  if (rs.event) {
    if (cudaEventDestroy(rs.event) != cudaSuccess) {
      GLINK_LOG("[gpu_link] shutdown: dev %d: cudaEventDestroy failed: %s\n", dev,
                cudaGetErrorString(cudaGetLastError()));
      ok = false;
    }
  }
  return ok;
}

}  // namespace

struct GpuLink::Impl {
  int dev_a = -1;
  int dev_b = -1;
  size_t max_bytes = 0;
  bool ready = false;
  GpuLink::Transport resolved = GpuLink::Transport::Auto;
  struct Rank {
    cudaEvent_t event = nullptr;
    void* pinned = nullptr;
      void* scratch = nullptr;
    // Flag all-reduce (P2P transport, small sums): the peer pushes its partial into `fbuf`
    // (two kFlagMaxBytes halves, alternating by op parity) and then stores the op's sequence
    // number into `fflag`. Both on this rank's device, written by the peer over P2P.
    void* fbuf = nullptr;
    unsigned* fflag = nullptr;
  };
  Rank ranks[2];  // [0] = dev_a, [1] = dev_b
  unsigned flag_seq = 0;   // flag-path ops posted so far (both ranks post every op, in order)
  bool flag_on = false;

  // Driver-workaround 1 state: destinations that already received a P2P priming copy, keyed by
  // (pointer, size class) so one priming covers one allocation at one magnitude class. Bounded
  // (a fresh link object per init starts empty; (d)'s 4 KiB ops never reach it, so the churn
  // test's RSS gate is unaffected).
  std::mutex prim_mu;
  std::unordered_set<uint64_t> primed;
};

// The 13.4 driver hole window: the first ~>=512 KiB P2P landing into a never-landed
// destination allocation is delivered with scattered holes (repro'd; see the file-top note).
// 64 KiB and 16 MiB first landings never misbehaved in any probe, so the gate is conservative
// on both edges: prime at >= 512 KiB, no upper cap (a stray extra 16 MiB priming copy is a
// one-time ~2 ms tax, never a data-integrity question).
constexpr size_t kPrimeGateBytes = 512u << 10;

// True the first time `dst` (a P2P D2D destination in this link) is seen at `bytes`' magnitude
// class — the caller then posts a one-time priming copy to it. Optimistic mark under lock: a
// concurrent same-dst op just skips its own priming (harmless — the copy is idempotent).
static bool glink_needs_prime(std::mutex& mu, std::unordered_set<uint64_t>& primed, void* dst,
                              size_t bytes) {
  if (bytes < kPrimeGateBytes) return false;
  // Size class in KiB, power of two: (512 KiB .. 1 MiB] -> 512, (1 MiB .. 2 MiB] -> 1024, ...
  uint64_t cls = 512;
  while (cls < (bytes >> 10)) cls <<= 1;
  const uint64_t key = (uint64_t)(std::uintptr_t)dst ^ cls * 0x9E3779B97F4A7C15ull;
  std::lock_guard<std::mutex> lk(mu);
  if (primed.count(key)) return false;
  primed.insert(key);
  if (primed.size() > 4096) primed.clear();
  return true;
}

GpuLink::GpuLink() : impl_(std::make_unique<Impl>()) {}

GpuLink::~GpuLink() {
  if (!impl_) return;
  if (impl_->ready) {
    // Full gated shutdown: a failed free is reported, not hidden.
    shutdown();
  } else {
    // Best-effort release of whatever a previously failed init left behind.
    for (int r = 0; r < 2; r++) {
      const int dev = (r == 0) ? impl_->dev_a : impl_->dev_b;
      const auto& rk = impl_->ranks[r];
      const RankResources rs{rk.event, rk.pinned, rk.scratch};
      if (dev >= 0) release_rank_resources(dev, rs, /*gate=*/false);
      impl_->ranks[r] = Impl::Rank{};
    }
  }
}

bool GpuLink::init(int dev_a, int dev_b, GpuLink::Transport transport, size_t max_bytes) {
  if (impl_ && impl_->ready) {
    GLINK_LOG("[gpu_link] init: already initialized (call shutdown() first) - refusing double-init\n");
    return false;
  }
  impl_.reset(new Impl());
  impl_->dev_a = dev_a;
  impl_->dev_b = dev_b;
  impl_->max_bytes = max_bytes;

  // Device ids: one per rank, distinct, in range.
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess) {
    GLINK_LOG("[gpu_link] init: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (dev_a < 0 || dev_a >= n_dev || dev_b < 0 || dev_b >= n_dev) {
    GLINK_LOG("[gpu_link] init: device ids (%d, %d) outside [0, %d)\n", dev_a, dev_b, n_dev);
    return false;
  }
  if (dev_a == dev_b) {
    GLINK_LOG("[gpu_link] init: the two ranks must live on distinct devices (got %d, %d)\n", dev_a, dev_b);
    return false;
  }
  if (max_bytes == 0) {
    GLINK_LOG("[gpu_link] init: max_bytes must be > 0\n");
    return false;
  }

  // Transport resolution. Auto probes with the side-effect-free CanAccessPeer, both ways,
  // before deciding; forced transports skip the probe.
  GpuLink::Transport resolved = transport;
  if (resolved == GpuLink::Transport::Auto) {
    int can_ab = 0, can_ba = 0;
    {
      DeviceScope s(dev_a);
      if (cudaDeviceCanAccessPeer(&can_ab, dev_a, dev_b) != cudaSuccess) {
        GLINK_LOG("[gpu_link] init: CanAccessPeer(%d->%d) errored: %s\n", dev_a, dev_b,
                  cudaGetErrorString(cudaGetLastError()));
        return false;
      }
    }
    {
      DeviceScope s(dev_b);
      if (cudaDeviceCanAccessPeer(&can_ba, dev_b, dev_a) != cudaSuccess) {
        GLINK_LOG("[gpu_link] init: CanAccessPeer(%d->%d) errored: %s\n", dev_b, dev_a,
                  cudaGetErrorString(cudaGetLastError()));
        return false;
      }
    }
    resolved = (can_ab != 0 && can_ba != 0) ? GpuLink::Transport::P2pMapped
                                              : GpuLink::Transport::PinnedStaging;
  }
  impl_->resolved = resolved;
  if (transport != GpuLink::Transport::Auto)
    GLINK_LOG("[gpu_link] init: requested transport %s (not auto)\n",
              transport == GpuLink::Transport::P2pMapped ? "p2p-mapped" : "pinned-staging");

  // Peer access, from each context to the other — only for the P2P transport (a forced
  // staging init never touches peer mapping, keeping that path cleanly isolated).
  // Deliberately never disabled (contexts outlive the object; re-init is idempotent).
  if (resolved == GpuLink::Transport::P2pMapped) {
    for (int r = 0; r < 2; r++) {
      const int dev = (r == 0) ? dev_a : dev_b;
      const int peer = (r == 0) ? dev_b : dev_a;
      cudaError_t e;
      {
        DeviceScope s(dev);
        e = cudaDeviceEnablePeerAccess(peer, 0);
        // Driver-workaround 2: 13.4 can queue cudaErrorPeerAccessAlreadyEnabled (704) even when
        // the call returns success, and the stale 704 then mis-attributes to the NEXT post in
        // this context (a kernel launch reports "peer access is already enabled" and the op
        // fails). Drain it here, in the same context, so no op-path post can pick it up.
        (void)cudaGetLastError();
      }
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        if (transport == GpuLink::Transport::P2pMapped)
          GLINK_LOG("[gpu_link] init: G1 regression: FORCED p2p-mapped unavailable - peer access "
                    "dev %d -> %d failed: %s (P2P was measured working at the WP-1 probe)\n",
                    dev, peer, cudaGetErrorString(e));
        else
          GLINK_LOG("[gpu_link] init: peer access dev %d -> %d failed after a CanAccessPeer=yes "
                    "probe: %s\n", dev, peer, cudaGetErrorString(e));
        return false;
      }
    }
  }

  // Per-rank resources, each built in its own rank's context (save/restore scopes).
  for (int r = 0; r < 2; r++) {
    const int dev = (r == 0) ? dev_a : dev_b;
    RankResources rs{};
    if (!setup_rank_resources(dev, rs, resolved, max_bytes)) {
      // Partial failure: tear down whatever exists (committed ranks from the impl plus the
      // uncommitted `rs` of the rank that just failed) and keep the not-ready impl so a
      // re-init starts from a clean slate.
      for (int q = 0; q < 2; q++) {
        const int qdev = (q == 0) ? dev_a : dev_b;
        const auto& rk = impl_->ranks[q];
        const RankResources qrs{rk.event, rk.pinned, rk.scratch};
        release_rank_resources(qdev, qrs, /*gate=*/false);
        impl_->ranks[q] = Impl::Rank{};
      }
      release_rank_resources(dev, rs, /*gate=*/false);
      return false;
    }
    // Commit this rank's resources into the impl.
    impl_->ranks[r].event = rs.event;
    impl_->ranks[r].pinned = rs.pinned;
    impl_->ranks[r].scratch = rs.scratch;
  }

  // Flag all-reduce buffers (P2P only; SPARKINFER_GLINK_FLAG=0 keeps every op on the copy path).
  if (resolved == GpuLink::Transport::P2pMapped) {
    const char* fe = getenv("SPARKINFER_GLINK_FLAG");
    bool ok = !(fe && fe[0] == '0');
    int prev = -1;
    cudaGetDevice(&prev);
    for (int r = 0; r < 2 && ok; r++) {
      if (cudaSetDevice(r == 0 ? dev_a : dev_b) != cudaSuccess ||
          cudaMalloc(&impl_->ranks[r].fbuf, 2 * detail::kFlagMaxBytes) != cudaSuccess ||
          cudaMalloc(&impl_->ranks[r].fflag, sizeof(unsigned)) != cudaSuccess ||
          cudaMemset(impl_->ranks[r].fflag, 0, sizeof(unsigned)) != cudaSuccess)
        ok = false;
    }
    if (ok && cudaDeviceSynchronize() != cudaSuccess) ok = false;
    if (prev >= 0) cudaSetDevice(prev);
    impl_->flag_on = ok;
    GLINK_LOG("[gpu_link] init: flag all-reduce %s (ops <= %zu bytes)\n", ok ? "on" : "off",
              (size_t)detail::kFlagMaxBytes);
  }

  impl_->ready = true;
  if (resolved == GpuLink::Transport::P2pMapped)
    GLINK_LOG("[gpu_link] init: transport=p2p-mapped (peer access enabled both ways), dev A=%d, dev B=%d, max_bytes=%zu\n",
              dev_a, dev_b, max_bytes);
  else
    GLINK_LOG("[gpu_link] init: transport=pinned-staging (peer access unavailable), dev A=%d, dev B=%d, max_bytes=%zu\n",
              dev_a, dev_b, max_bytes);
  return true;
}

bool GpuLink::shutdown() {
  if (!impl_) return true;  // never initialized: a no-op
  if (!impl_->ready) {
    // Not ready: best-effort release of whatever a failed init left (normally a no-op,
    // since init's failure path already released and zeroed the ranks).
    for (int r = 0; r < 2; r++) {
      const int dev = (r == 0) ? impl_->dev_a : impl_->dev_b;
      const auto& rk = impl_->ranks[r];
      const RankResources rs{rk.event, rk.pinned, rk.scratch};
      if (dev >= 0) release_rank_resources(dev, rs, /*gate=*/false);
      impl_->ranks[r] = Impl::Rank{};
    }
    impl_->ready = false;
    return true;
  }
  // Contract: the caller has drained every stream it posted ops on. This call may (and
  // must) block: it is where a failed free is reported.
  bool ok = true;
  {
    int prev = -1;
    cudaGetDevice(&prev);
    for (int r = 0; r < 2; r++) {
      auto& rk = impl_->ranks[r];
      if (!rk.fbuf && !rk.fflag) continue;
      cudaSetDevice(r == 0 ? impl_->dev_a : impl_->dev_b);
      if (rk.fbuf && cudaFree(rk.fbuf) != cudaSuccess) ok = false;
      if (rk.fflag && cudaFree(rk.fflag) != cudaSuccess) ok = false;
      rk.fbuf = nullptr; rk.fflag = nullptr;
    }
    if (prev >= 0) cudaSetDevice(prev);
    impl_->flag_on = false;
  }
  for (int r = 0; r < 2; r++) {
    const int dev = (r == 0) ? impl_->dev_a : impl_->dev_b;
    const auto& rk = impl_->ranks[r];
    const RankResources rs{rk.event, rk.pinned, rk.scratch};
    if (!release_rank_resources(dev, rs, /*gate=*/true)) ok = false;
    impl_->ranks[r] = Impl::Rank{};
  }
  impl_->ready = false;
  impl_->max_bytes = 0;
  if (!ok)
    GLINK_LOG("[gpu_link] shutdown: FAIL - a resource free failed (see lines above)\n");
  return ok;
}

bool GpuLink::is_ready() const { return impl_ && impl_->ready; }

GpuLink::Transport GpuLink::transport() const {
  return impl_ ? impl_->resolved : GpuLink::Transport::Auto;
}

int GpuLink::device_a() const { return impl_ ? impl_->dev_a : -1; }
int GpuLink::device_b() const { return impl_ ? impl_->dev_b : -1; }
size_t GpuLink::max_bytes() const { return impl_ ? impl_->max_bytes : 0; }

bool GpuLink::reduce_impl(const RankRef& a, const RankRef& b, size_t bytes, Dtype dtype, bool is_max) {
  const char* op = is_max ? "maxreduce" : "allreduce";
  auto fail = [&](const char* why, cudaError_t e) -> bool {
    if (e == cudaSuccess)
      GLINK_LOG("[gpu_link] %s: %s\n", op, why);
    else
      GLINK_LOG("[gpu_link] %s: %s: %s\n", op, why, cudaGetErrorString(e));
    return false;
  };

  if (!impl_ || !impl_->ready) return fail("called before a successful init", cudaSuccess);
  const Impl& im = *impl_;
  if (a.device != im.dev_a || b.device != im.dev_b)
    return fail("rank devices do not match the initialized (dev_a, dev_b) pair", cudaSuccess);
  if (a.stream == nullptr || b.stream == nullptr) return fail("null stream", cudaSuccess);
  if (a.in == nullptr || a.out == nullptr || b.in == nullptr || b.out == nullptr)
    return fail("null buffer pointer", cudaSuccess);
  const size_t esize = detail::glink_dtype_size(dtype);
  if (bytes == 0 || bytes > im.max_bytes) return fail("size 0 or beyond max_bytes", cudaSuccess);
  if (bytes % esize != 0) return fail("size not a multiple of the element size", cudaSuccess);
  const size_t n = bytes / esize;

  // Entry-time cross-context events: record each rank's event on its own stream NOW, before this
  // call posts any copy or kernel. The record point sits after every op the caller already
  // enqueued on that stream (the producers that wrote this rank's `in`), so each event snapshots
  // "this rank's producers are done up to here." In the P2P branch each side's peer-copy (which
  // reads the PEER's `in`) then waits on the PEER's entry event before reading, so a per-layer
  // GEMM still in flight on the peer cannot race the copy. Both records are posted before any
  // wait, so no wait can target an unrecorded event, and each stream waits only on the other's
  // entry snapshot (never on the other's in-flight copy) -> symmetric, deadlock-free. The
  // staging fallback records the same two events later (after its D2H); that later record
  // supersedes these, so the staging waits are unaffected.
  {
    cudaError_t re = cudaEventRecord(im.ranks[0].event, a.stream);
    if (re != cudaSuccess) return fail("entry event record on rank A's stream", re);
    re = cudaEventRecord(im.ranks[1].event, b.stream);
    if (re != cudaSuccess) return fail("entry event record on rank B's stream", re);
  }

  // The peer's data must land on this rank (out, or the module's scratch when in==out)
  // before this rank's reduce kernel reads it; the kernel half is identical across
  // transports, only the landing differs.
  void* dst_a = (a.in == a.out) ? im.ranks[0].scratch : const_cast<void*>(a.out);
  void* dst_b = (b.in == b.out) ? im.ranks[1].scratch : const_cast<void*>(b.out);

  // Flag all-reduce: the small per-layer sums. One kernel per rank pushes its partial into the
  // peer's landing buffer over P2P, publishes the op's sequence number in the peer's flag, waits
  // for the peer's flag, and sums -- no copy engine, no cross-device event, and no exit fence
  // (nothing ever reads this rank's `in` from the other side). Stream order alone sequences it
  // after this rank's producers. The landing halves alternate by op parity: a rank can only be
  // writing op k+2 into a half once the peer has finished op k+1, i.e. finished reading op k.
  if (im.resolved == GpuLink::Transport::P2pMapped && im.flag_on && !is_max &&
      bytes <= detail::kFlagMaxBytes && (dtype == Dtype::Float32 || dtype == Dtype::BFloat16)) {
    const unsigned seq = ++impl_->flag_seq;
    const size_t half = (seq & 1) ? detail::kFlagMaxBytes : 0;
    const auto& ra = im.ranks[0];
    const auto& rb = im.ranks[1];
    cudaError_t e = detail::launch_glink_flag_allreduce(
        a.in, a.out, static_cast<char*>(rb.fbuf) + half, static_cast<const char*>(ra.fbuf) + half,
        rb.fflag, ra.fflag, seq, n, dtype, a.stream);
    if (e != cudaSuccess) return fail("flag all-reduce on rank A", e);
    e = detail::launch_glink_flag_allreduce(
        b.in, b.out, static_cast<char*>(ra.fbuf) + half, static_cast<const char*>(rb.fbuf) + half,
        ra.fflag, rb.fflag, seq, n, dtype, b.stream);
    if (e != cudaSuccess) return fail("flag all-reduce on rank B", e);
    return true;
  }

  if (im.resolved == GpuLink::Transport::P2pMapped) {
    // Each rank pulls the peer's `in` buffer into its own out/scratch on its own stream — the
    // stream-scoped D2D (destination stream) WP-1's probe proved on this pair, both
    // directions — then runs its local reduce. No host memory, no setDevice.
    //
    // Cross-context ordering: before a rank's stream reads the PEER's `in`, it first waits on
    // the peer's entry-time event (the one recorded on the peer's stream at the top of
    // reduce_impl, after the peer's producers). That snapshots "peer's producers are done"
    // without waiting on the peer's in-flight copy, so the two waits are independent and cannot
    // deadlock (this is the same cross-context wait the staging fallback performs below).
    //
    // Driver-workaround 1: a destination that never received a P2P landing at this magnitude
    // first gets a one-time priming copy on the SAME (src, dst, stream) — the 13.4
    // first-landing hole window (file-top note). The priming content is irrelevant (the real
    // copy immediately overwrites the identical range) and both copies stream-order ahead of
    // the kernel, so the op stays (device, stream)-pure. Steady-state ops (already-primed
    // dst) pay exactly one copy.
    // Driver-workaround 3: every copy in this path goes through the bounded re-post, because
    // a first-landing burst can have the driver reject the next post at post time.
    {
      // Wait on the peer's entry-time event before this rank's stream first reads the peer's
      // `in` (the priming copy, when it fires, is also such a read, so the wait precedes it
      // too). A waits on B's entry event before A's copies read b.in; B waits on A's before B's
      // copies read a.in. Each stream waits only on the other's entry snapshot, never on the
      // other's in-flight copy -> no circular wait, deadlock-free.
      cudaError_t we = cudaStreamWaitEvent(a.stream, im.ranks[1].event, 0);
      if (we != cudaSuccess) return fail("P2P: rank A wait on rank B's entry event", we);
      we = cudaStreamWaitEvent(b.stream, im.ranks[0].event, 0);
      if (we != cudaSuccess) return fail("P2P: rank B wait on rank A's entry event", we);
    }
    if (glink_needs_prime(impl_->prim_mu, impl_->primed, dst_a, bytes)) {
      cudaError_t p =
          post_copy_retry(op, "P2P priming copy into rank A", dst_a, b.in, bytes,
                          cudaMemcpyDeviceToDevice, a.stream);
      if (p != cudaSuccess) return fail("P2P priming copy into rank A", p);
    }
    if (glink_needs_prime(impl_->prim_mu, impl_->primed, dst_b, bytes)) {
      cudaError_t p =
          post_copy_retry(op, "P2P priming copy into rank B", dst_b, a.in, bytes,
                          cudaMemcpyDeviceToDevice, b.stream);
      if (p != cudaSuccess) return fail("P2P priming copy into rank B", p);
    }
    cudaError_t e = post_copy_retry(op, "D2D copy into rank A", dst_a, b.in, bytes,
                                     cudaMemcpyDeviceToDevice, a.stream);
    if (e != cudaSuccess) return fail("D2D copy into rank A", e);
    e = post_copy_retry(op, "D2D copy into rank B", dst_b, a.in, bytes, cudaMemcpyDeviceToDevice,
                         b.stream);
    if (e != cudaSuccess) return fail("D2D copy into rank B", e);
    // Exit fence: neither rank may write its in/out (the in-place reduce below, or any later op
    // on its stream) until the PEER's copy has finished reading it. Without this a rank that runs
    // ahead overwrites its partial with the sum while the peer is still copying it, and the peer
    // computes b + (a+b) -- a silent, timing-dependent wrong sum. The entry events are re-recorded
    // as "my copy is done" (the waits above already snapshotted their entry records).
    e = cudaEventRecord(im.ranks[0].event, a.stream);
    if (e == cudaSuccess) e = cudaEventRecord(im.ranks[1].event, b.stream);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(a.stream, im.ranks[1].event, 0);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(b.stream, im.ranks[0].event, 0);
    if (e != cudaSuccess) return fail("P2P: exit fence", e);
    e = detail::launch_glink_reduce(a.out, a.in, dst_a, n, dtype, is_max, a.stream);
    if (e != cudaSuccess) return fail("reduce kernel on rank A", e);
    e = detail::launch_glink_reduce(b.out, b.in, dst_b, n, dtype, is_max, b.stream);
    if (e != cudaSuccess) return fail("reduce kernel on rank B", e);
    return true;
  }

  // Pinned-host staging (the fallback). (1) both ranks pull their inputs to their own
  // pinned buffers on their own streams and record an event — both D2Hs posted before any
  // wait, so no wait can target an unrecorded event; (2) each rank waits on the peer's
  // event (cross-context), H2D-copies the peer's pinned buffer into its own out/scratch,
  // and runs the same local reduce kernel.
  const auto& ra = im.ranks[0];
  const auto& rb = im.ranks[1];
  cudaError_t e;
  e = post_copy_retry(op, "D2H on rank A", ra.pinned, a.in, bytes, cudaMemcpyDeviceToHost,
                       a.stream);
  if (e == cudaSuccess) e = cudaEventRecord(ra.event, a.stream);
  if (e != cudaSuccess) return fail("D2H/event on rank A", e);
  e = post_copy_retry(op, "D2H on rank B", rb.pinned, b.in, bytes, cudaMemcpyDeviceToHost,
                       b.stream);
  if (e == cudaSuccess) e = cudaEventRecord(rb.event, b.stream);
  if (e != cudaSuccess) return fail("D2H/event on rank B", e);

  e = cudaStreamWaitEvent(a.stream, rb.event, 0);
  if (e == cudaSuccess)
    e = post_copy_retry(op, "H2D into rank A", dst_a, rb.pinned, bytes, cudaMemcpyHostToDevice,
                         a.stream);
  if (e == cudaSuccess) e = detail::launch_glink_reduce(a.out, a.in, dst_a, n, dtype, is_max, a.stream);
  if (e != cudaSuccess) return fail("peer staging on rank A", e);
  e = cudaStreamWaitEvent(b.stream, ra.event, 0);
  if (e == cudaSuccess)
    e = post_copy_retry(op, "H2D into rank B", dst_b, ra.pinned, bytes, cudaMemcpyHostToDevice,
                         b.stream);
  if (e == cudaSuccess) e = detail::launch_glink_reduce(b.out, b.in, dst_b, n, dtype, is_max, b.stream);
  if (e != cudaSuccess) return fail("peer staging on rank B", e);
  // Exit fence (staging): each rank's NEXT D2H overwrites its pinned buffer, which the peer's
  // H2D above may still be reading; order every later op on each stream after the peer's H2D.
  e = cudaEventRecord(ra.event, a.stream);
  if (e == cudaSuccess) e = cudaEventRecord(rb.event, b.stream);
  if (e == cudaSuccess) e = cudaStreamWaitEvent(a.stream, rb.event, 0);
  if (e == cudaSuccess) e = cudaStreamWaitEvent(b.stream, ra.event, 0);
  if (e != cudaSuccess) return fail("staging: exit fence", e);
  return true;
}

bool GpuLink::allreduce(const RankRef& a, const RankRef& b, size_t bytes, Dtype dtype) {
  return reduce_impl(a, b, bytes, dtype, /*is_max=*/false);
}

bool GpuLink::maxreduce(const RankRef& a, const RankRef& b, size_t bytes, Dtype dtype) {
  return reduce_impl(a, b, bytes, dtype, /*is_max=*/true);
}

}  // namespace sparkinfer
