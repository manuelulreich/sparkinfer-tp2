#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>

/*
 * WP-3: the 2-rank communication wrapper for tensor-parallel 2.
 *
 * R8 verdict (dual-gpu/00-p0-probe/report.md, on the record): NCCL is unusable on this
 * 2x-RTX-5060-Ti pair — 2.20.5+cu12.4 hangs in ncclCommInitRank on sm_120, and 2.32.3+cu13.4
 * inits in 0.6 s but its 2-rank all-reduce silently returns +inf (no error, 3/3 runs).
 * NCCL remains a *design* option for NVLink/server-class hardware, gated by a re-run of that
 * probe; this module is the shipping 2-rank path on this box.
 *
 * (device, stream) purity: the op path calls only stream-scoped async copies, cross-context
 * event waits, and stream-scoped kernel launches. cudaSetDevice is confined to init/shutdown
 * (save/restore), where it is unavoidable: peer access, event creation, scratch allocation
 * and pinned allocation are current-context-bound in the runtime API.
 */

namespace sparkinfer {

// One 2-rank comm object per process: created at init (outside any graph capture),
// destroyed at shutdown. Exactly two ops: allreduce (sum) and maxreduce (max — the
// vocab-split lm_head logits: greedy = cross-device max of the two local maxima).
class GpuLink {
public:
  enum class Transport {
    Auto,          // try P2P-mapped; fall back to pinned-host staging
    P2pMapped,     // peer-mapped D2D, the fast path (forced: init fails if unavailable)
    PinnedStaging, // host-pinned staging (always available; the fallback)
  };
  enum class Dtype {
    Float32,   // 4 bytes/element
    BFloat16,  // 2 bytes/element
    Float16,   // 2 bytes/element
  };

  // One rank's view of an op: the device its buffers and stream live on (must match the
  // device this rank was initialized on), and the input/output buffers on that device.
  // `in` may alias `out` (in-place); the in-place case routes the peer's data through a
  // private scratch buffer so the reduce kernel never reads a clobbered source.
  struct RankRef {
    int device;          // device this rank's buffers/stream belong to
    cudaStream_t stream; // stream to post on (must be a stream on `device`)
    const void* in;      // input on `device`
    void* out;           // output on `device` (may alias `in`)
  };

  GpuLink();
  ~GpuLink();
  GpuLink(const GpuLink&) = delete;
  GpuLink& operator=(const GpuLink&) = delete;

  // Establish the 2-rank comm. `transport` selects the data path: Auto tries the
  // P2P-mapped fast path (peer access in both directions) and falls back to pinned-host
  // staging; the choice is logged to stdout. Forced transports fail init (loudly) when
  // unavailable — a forced-P2pMapped failure on a pair that showed P2P at probe time is
  // the G1 regression signal. `max_bytes` bounds every op (bytes <= max_bytes and
  // bytes % element_size == 0). Returns false (and logs) on any driver failure.
  bool init(int dev_a, int dev_b, Transport transport = Transport::Auto,
            size_t max_bytes = 1u << 20);

  // Tear everything down: scratch, pinned staging, events (returns false if any free fails). Peer access
  // itself is deliberately left enabled for the life of the contexts (re-init is
  // idempotent). The caller must have drained all streams before calling.
  bool shutdown();

  bool is_ready() const;
  Transport transport() const;  // the resolved transport (meaningful after a successful init)
  int device_a() const;
  int device_b() const;
  size_t max_bytes() const;

  // Elementwise sum, both ranks, each on its own stream:
  //   out_r[i] = in_r[i] + in_peer[i]
  // No cudaSetDevice, no host sync, no host memory on the path.
  bool allreduce(const RankRef& a, const RankRef& b, size_t bytes, Dtype dtype);

  // Elementwise max, same shape (for the vocab-split lm_head logits):
  //   out_r[i] = max(in_r[i], in_peer[i])
  bool maxreduce(const RankRef& a, const RankRef& b, size_t bytes, Dtype dtype);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  bool reduce_impl(const RankRef& a, const RankRef& b, size_t bytes, Dtype dtype, bool is_max);
};

namespace detail {

// Element size in bytes for a GpuLink::Dtype (4 for Float32, 2 for the 16-bit types).
inline size_t glink_dtype_size(GpuLink::Dtype dtype) {
  return dtype == GpuLink::Dtype::Float32 ? 4 : 2;
}

// The elementwise sum/max kernel dispatcher behind both ops: dst[i] = a[i] op b[i],
// grid-stride, op chosen by is_max, math in float rounded back per dtype (ties in max
// take b). Returns cudaSuccess on a good launch; the caller treats a non-success as an
// op failure. n is the element count (not bytes).
cudaError_t launch_glink_reduce(void* dst, const void* a, const void* b, size_t n,
                                 GpuLink::Dtype dtype, bool is_max, cudaStream_t stream);

// Flag all-reduce (P2P, sum, f32/bf16, ops up to kFlagMaxBytes): one single-block kernel per
// rank. It stores this rank's `in` into the peer's landing buffer `peer_land` over P2P, fences,
// stores `seq` into the peer's flag, spins until its own flag reaches `seq` (the peer's data has
// landed in `my_land`), then writes out[i] = in[i] + my_land[i] -- the same float sum, rounded
// back per dtype, as the copy path's reduce kernel, so both transports give identical bits.
// A peer that never arrives traps the kernel after ~10 s instead of hanging the stream forever.
constexpr size_t kFlagMaxBytes = 256u << 10;
// Forces the flag kernels' module to load on the current device (lazy loading); false on failure.
bool preload_glink_flag_kernels();
cudaError_t launch_glink_flag_allreduce(const void* in, void* out, void* peer_land,
                                        const void* my_land, unsigned* peer_flag,
                                        const unsigned* my_flag, unsigned seq, size_t n,
                                        GpuLink::Dtype dtype, cudaStream_t stream);

}  // namespace detail

}  // namespace sparkinfer
