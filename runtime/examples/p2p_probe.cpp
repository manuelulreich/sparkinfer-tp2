// WP-1 (dual-gpu program, see dual-gpu/03-implementation-plan.md): P2P probe + NCCL smoke test
// for the 2x RTX 5060 Ti pair (16 GB each, sm_120, PCIe, no NVLink) that will carry Qwen3.8-27B
// at tp=2.
//
// Prints, in order:
//   [1] both cards' name / SM count (major.minor) / total + free VRAM
//   [2] cudaDeviceCanAccessPeer in both directions
//   [3] if P2P is available in either direction: cudaDeviceEnablePeerAccess per direction, then
//       timed device->device copies at 1/16/64 MB, one pass per available direction (GPU events,
//       several reps, GB/s). The LLM is resident on both cards, so treat the numbers as
//       *measured-with-contention*, not datasheet values (the plan's R7).
//   [4] the pinned-host staging path (always available, even without P2P):
//       device->pinned->device at the same sizes, both directions, each leg event-timed.
//   [5] NCCL smoke: dlopens the box's libnccl (the one allowed C runtime dependency -- WP-3's
//       gpu_link will dlopen/link the same library), prints ncclGetVersion, inits a 2-rank
//       communicator (one worker thread per card, cudaSetDevice first, uniqueId exchanged
//       in-process), runs one ncclAllReduce of 1024 float32 and verifies it equals the sum,
//       then destroys. NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=INIT,NET,SHM are set and NCCL's
//       stderr is redirected to a log file for the duration of the smoke, so the record shows
//       which transport NCCL itself selected (P2P vs SHM host-staging) and the bandwidth it
//       measured -- that is G1's answer, and the sm_120 outcome of a failed init is R8's answer.
//       A failed init here is a recorded RESULT (verbatim error), not a failure of this probe.
//   [6] per-card VRAM before/after the whole run, to show nothing leaks.
//
// Lean-mode VRAM rules (dual-gpu/03 §2 rule 1): every device allocation below is an explicit
// small buffer -- at most 64 MB src + 64 MB dst per card (D2D section, only when P2P is
// available), at most 64 MB src + 64 MB dst per card (staging section), 4 KB of NCCL smoke
// buffers, plus NCCL's own internal channel buffers (tens of MB at most) -- all well inside the
// ~512 MB/card window. Nothing is auto-sized from free VRAM, nothing here touches the resident
// LLM, and everything is freed on exit (the LLM is a separate process; a wedged smoke thread is
// detached and reaped by process exit, never by a signal to another process).
//
// Usage: p2p_probe [nccl_lib_path] [nccl_log_path] [--no-nccl]
//   nccl_lib_path  defaults to the cifarTest_cuda wheel copy (the only NCCL on this box)
//   nccl_log_path  where NCCL's NCCL_DEBUG stderr goes (the record file)
//   --no-nccl      run sections 1-4 and 6 only (skips the smoke entirely)
//
// Exit code: 0 when the probe ran to the end (including "NCCL init failed" -- that is the R8
// answer, a result, not a failure); 1 only when the probe's own harness failed (allocation,
// event or copy error, data mismatch, or the smoke watchdog tripped).

#include <cuda_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------- NCCL types, declared by hand
// The probe must NOT pull nccl.h into the build (WP-1 is new-file-only; the dependency lands in
// the build system in WP-3). The declarations below mirror nvidia/nccl/include/nccl.h from the
// 2.20.5 wheel (NCCL_UNIQUE_ID_BYTES 128, pass-by-value ncclUniqueId, ncclComm_t is a pointer to
// an opaque struct). ncclFloat == 0 and ncclSum == 0 in that header's enums.
struct NcclUniqueId { char internal[128]; };
typedef int NcclResult;
#define NCCL_OK 0
#define NCLC_FLOAT 0
#define NCLC_SUM 0
using NcclGetVersionFn = NcclResult (*)(int*);
using NcclGetUniqueIdFn = NcclResult (*)(NcclUniqueId*);
using NcclCommInitRankFn = NcclResult (*)(void**, int, NcclUniqueId, int);
using NcclAllReduceFn = NcclResult (*)(const void*, void*, size_t, int, int, void*, void*);
using NcclCommDestroyFn = NcclResult (*)(void*);
using NcclGetErrorStringFn = const char* (*)(NcclResult);

struct NcclFns {
  void* lib = nullptr;
  NcclGetVersionFn getVersion = nullptr;
  NcclGetUniqueIdFn getUniqueId = nullptr;
  NcclCommInitRankFn commInitRank = nullptr;
  NcclAllReduceFn allReduce = nullptr;
  NcclCommDestroyFn commDestroy = nullptr;
  NcclGetErrorStringFn getErrorString = nullptr;
  bool complete() const { return getVersion && getUniqueId && commInitRank && allReduce && commDestroy; }
};
static const char* nccl_err(const NcclFns& f, NcclResult r) {
  thread_local char buf[256];   // two worker threads call this concurrently
  if (f.getErrorString) {
    const char* s = f.getErrorString(r);
    if (s) { snprintf(buf, sizeof buf, "%s (ncclResult %d)", s, r); return buf; }
  }
  snprintf(buf, sizeof buf, "ncclResult %d", r);
  return buf;
}

// ------------------------------------------------------------------- small CUDA helpers
static bool ck(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  fprintf(stderr, "  [cuda] %s: %s\n", what, cudaGetErrorString(e));
  return false;
}
static void print_card(int i) {
  cudaSetDevice(i);
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, i) != cudaSuccess) {
    fprintf(stderr, "  card %d: cudaGetDeviceProperties failed\n", i);
    return;
  }
  size_t freeb = 0, totb = 0;
  cudaMemGetInfo(&freeb, &totb);
  printf("  card %d: %-30s SM %d.%d  %3d SMs   VRAM total %8.1f MiB  free %8.1f MiB\n",
         i, p.name, p.major, p.minor, p.multiProcessorCount, totb / 1048576.0, freeb / 1048576.0);
}
static void print_all_cards(const char* label) {
  int n = 0;
  cudaGetDeviceCount(&n);
  printf("=== %s ===\n", label);
  for (int i = 0; i < n; i++) print_card(i);
}

// ---------------------------------------------------------------- [2] peer access
struct PeerInfo { bool can01 = false, can10 = false, en01 = false, en10 = false; };
static PeerInfo probe_peer_access() {
  PeerInfo pi;
  int a = 0, b = 0;
  cudaDeviceCanAccessPeer(&a, 0, 1);
  cudaDeviceCanAccessPeer(&b, 1, 0);
  pi.can01 = a != 0;
  pi.can10 = b != 0;
  if (pi.can01) {
    cudaSetDevice(0);
    cudaError_t e = cudaDeviceEnablePeerAccess(1, 0);
    pi.en01 = (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled);
    if (!pi.en01) fprintf(stderr, "  cudaDeviceEnablePeerAccess(0->1): %s\n", cudaGetErrorString(e));
  }
  if (pi.can10) {
    cudaSetDevice(1);
    cudaError_t e = cudaDeviceEnablePeerAccess(0, 0);
    pi.en10 = (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled);
    if (!pi.en10) fprintf(stderr, "  cudaDeviceEnablePeerAccess(1->0): %s\n", cudaGetErrorString(e));
  }
  return pi;
}

// ---------------------------------------------------------------- [3] D2D copies (P2P)
// One 64 MB buffer per role per card (src/dst). The timed copy runs on the SOURCE device's
// stream (the copy is posted by the source, exactly as the CUDA p2pBandwidthLatencyTest sample
// drives it); each rep is bracketed by a start/end event pair on that stream, and both devices
// are synchronized after every rep so reps stay strictly serial. GB/s is bytes moved /
// event-measured seconds.
struct D2DSetup {
  void* buf[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};   // [card][0]=src, [1]=dst
  cudaStream_t stream[2] = {nullptr, nullptr};
  cudaEvent_t ev_start[2] = {nullptr, nullptr};
  cudaEvent_t ev_end[2] = {nullptr, nullptr};
  ~D2DSetup() {
    for (int d = 0; d < 2; d++) {
      cudaSetDevice(d);   // cudaFree / stream / event destroy all act on the *current* device
      if (buf[d][0]) cudaFree(buf[d][0]);
      if (buf[d][1]) cudaFree(buf[d][1]);
      if (stream[d]) cudaStreamDestroy(stream[d]);
      if (ev_start[d]) cudaEventDestroy(ev_start[d]);
      if (ev_end[d]) cudaEventDestroy(ev_end[d]);
    }
  }
};

// src card `s` (0/1) -> dst card `1-s`; `bytes` copied; `reps` timed reps.
// Returns true if all reps ran and the first one's data matched; prints the median GB/s line.
static bool d2d_direction(const D2DSetup& ds, int s, size_t bytes, int reps) {
  int d = 1 - s;
  cudaSetDevice(s);
  cudaStream_t stream = ds.stream[s];
  cudaEvent_t ev_start = ds.ev_start[s];
  cudaEvent_t ev_end = ds.ev_end[s];
  std::vector<double> gbs(reps, 0.0);
  for (int r = 0; r < reps; r++) {
    if (r == 0) {
      // Fill the src pattern once (deterministic); the head is verified after the first copy.
      std::vector<unsigned int> pat(bytes / 4);
      for (size_t i = 0; i < pat.size(); i++) pat[i] = 0xA5A5A5A5u + (unsigned int)i;
      if (!ck(cudaMemcpy(ds.buf[s][0], pat.data(), bytes, cudaMemcpyHostToDevice), "p2p fill src")) return false;
    }
    float ms = 0.0f;
    if (!ck(cudaEventRecord(ev_start, stream), "p2p ev start")) return false;
    cudaError_t e = cudaMemcpyAsync(ds.buf[d][1], ds.buf[s][0], bytes, cudaMemcpyDeviceToDevice, stream);
    if (!ck(e, "p2p d2d async")) return false;
    if (!ck(cudaEventRecord(ev_end, stream), "p2p ev end")) return false;
    if (!ck(cudaEventSynchronize(ev_end), "p2p ev sync")) return false;
    if (!ck(cudaEventElapsedTime(&ms, ev_start, ev_end), "p2p ev elapsed")) return false;
    // The completion edge seen from the destination can lag the source-side event (two engines
    // observing one DMA); sync both devices before the next rep so reps never overlap.
    cudaDeviceSynchronize();
    cudaSetDevice(d);
    cudaDeviceSynchronize();
    cudaSetDevice(s);
    gbs[r] = bytes / 1e9 / (ms / 1e3);
    if (r == 0) {
      // Verify the head of the dst against the src pattern.
      unsigned int hsrc[16] = {}, hdst[16] = {};
      cudaSetDevice(s);
      cudaMemcpy(hsrc, ds.buf[s][0], sizeof hsrc, cudaMemcpyDeviceToHost);
      cudaSetDevice(d);
      cudaMemcpy(hdst, ds.buf[d][1], sizeof hdst, cudaMemcpyDeviceToHost);
      cudaSetDevice(s);
      if (memcmp(hsrc, hdst, sizeof hsrc) != 0) {
        fprintf(stderr, "  d2d %d->%d: first copy MISMATCH (src 0x%08x vs dst 0x%08x)\n",
                s, d, hsrc[0], hdst[0]);
        return false;
      }
    }
  }
  std::vector<double> sorted = gbs;
  std::sort(sorted.begin(), sorted.end());
  double med = sorted[reps / 2];
  printf("  D2D (P2P) card%d->card%d %6zu B:  median %7.2f GB/s  (min %6.2f, max %6.2f, %d reps)\n",
         s, d, bytes, med, *std::min_element(gbs.begin(), gbs.end()),
         *std::max_element(gbs.begin(), gbs.end()), reps);
  return true;
}

// Runs every direction that is both CanAccessPeer AND EnablePeerAccess. The 128 MB/card buffers
// are allocated only if at least one direction is usable.
static bool run_d2d(const PeerInfo& pi) {
  printf("=== [3] device->device copies (P2P path, if available) ===\n");
  size_t sizes[] = {1u << 20, 16u << 20, 64u << 20};
  const int reps = 8;
  bool do01 = pi.can01 && pi.en01;
  bool do10 = pi.can10 && pi.en10;
  if (pi.can01 && !do01)
    printf("  card0->card1: CanAccessPeer=yes but EnablePeerAccess failed -> direction not usable\n");
  if (pi.can10 && !do10)
    printf("  card1->card0: CanAccessPeer=yes but EnablePeerAccess failed -> direction not usable\n");
  if (!do01 && !do10) {
    printf("  P2P NOT available on this pair (CanAccessPeer %s / %s) -- no D2D section; section [4] is the transport the comm layer would use.\n",
           pi.can01 ? "yes" : "no", pi.can10 ? "yes" : "no");
    return true;   // expected on a PCIe/PHB pair; the staging section below is the answer
  }
  bool ok = true;
  {
    D2DSetup ds;
    for (int c = 0; c < 2; c++) {
      cudaSetDevice(c);
      if (!ok) break;
      ok = ck(cudaMalloc(&ds.buf[c][0], 64u << 20), "p2p malloc src");
      if (ok) ok = ck(cudaMalloc(&ds.buf[c][1], 64u << 20), "p2p malloc dst");
      if (ok) ok = ck(cudaStreamCreateWithFlags(&ds.stream[c], cudaStreamNonBlocking), "p2p stream");
      if (ok) ok = ck(cudaEventCreate(&ds.ev_start[c]), "p2p ev start");
      if (ok) ok = ck(cudaEventCreate(&ds.ev_end[c]), "p2p ev end");
    }
    if (ok) {
      if (do01) { for (size_t bytes : sizes) { if (!d2d_direction(ds, 0, bytes, reps)) ok = false; } }
      if (ok && do10) { for (size_t bytes : sizes) { if (!d2d_direction(ds, 1, bytes, reps)) ok = false; } }
    }
  }
  if (!ok) fprintf(stderr, "  D2D section hit a harness error; see the [cuda] lines above\n");
  return ok;
}

// ---------------------------------------------------------------- [4] pinned-host staging
// device(src) -> pinned host -> device(dst). Always available regardless of P2P. Each leg is
// event-timed on its own device's stream; the reported total is leg1 + leg2 of one pass.
// `pinned` is a single 64 MiB host allocation owned by main (the largest copy size); the device
// buffers are allocated per pass at the exact copy size and freed before the pass returns.
static bool run_staging(int s, size_t bytes, int reps, void* pinned) {
  int d = 1 - s;
  void* src = nullptr;
  void* dst = nullptr;
  cudaSetDevice(s);
  bool ok = ck(cudaMalloc(&src, bytes), "staging malloc src");
  cudaSetDevice(d);
  if (ok) ok = ck(cudaMalloc(&dst, bytes), "staging malloc dst");
  cudaStream_t st_s = nullptr, st_d = nullptr;
  cudaEvent_t ev_s_start = nullptr, ev_s_end = nullptr, ev_d_start = nullptr, ev_d_end = nullptr;
  cudaSetDevice(s);
  if (ok) {
    ok = ck(cudaStreamCreateWithFlags(&st_s, cudaStreamNonBlocking), "staging stream src") &&
         ck(cudaEventCreate(&ev_s_start), "staging ev src start") &&
         ck(cudaEventCreate(&ev_s_end), "staging ev src end");
  }
  cudaSetDevice(d);
  if (ok) {
    ok = ck(cudaStreamCreateWithFlags(&st_d, cudaStreamNonBlocking), "staging stream dst") &&
         ck(cudaEventCreate(&ev_d_start), "staging ev dst start") &&
         ck(cudaEventCreate(&ev_d_end), "staging ev dst end");
  }

  std::vector<unsigned char> pat(bytes);
  for (size_t i = 0; i < bytes; i++) pat[i] = (unsigned char)(i * 7 + 3);
  std::vector<double> leg1(reps, 0.0), leg2(reps, 0.0);
  for (int r = 0; r < reps && ok; r++) {
    if (r == 0) {
      cudaSetDevice(s);
      ok = ck(cudaMemcpy(src, pat.data(), bytes, cudaMemcpyHostToDevice), "staging fill src");
      if (!ok) break;
    }
    float t1 = 0, t2 = 0;
    cudaSetDevice(s);
    if (!ck(cudaEventRecord(ev_s_start, st_s), "staging leg1 start")) { ok = false; break; }
    cudaError_t e = cudaMemcpyAsync(pinned, src, bytes, cudaMemcpyDeviceToHost, st_s);
    if (!ck(e, "staging d2h async")) { ok = false; break; }
    if (!ck(cudaEventRecord(ev_s_end, st_s), "staging leg1 end")) { ok = false; break; }
    if (!ck(cudaEventSynchronize(ev_s_end), "staging leg1 sync")) { ok = false; break; }
    if (!ck(cudaEventElapsedTime(&t1, ev_s_start, ev_s_end), "staging leg1 elapsed")) { ok = false; break; }
    cudaSetDevice(d);
    if (!ck(cudaEventRecord(ev_d_start, st_d), "staging leg2 start")) { ok = false; break; }
    e = cudaMemcpyAsync(dst, pinned, bytes, cudaMemcpyHostToDevice, st_d);
    if (!ck(e, "staging h2d async")) { ok = false; break; }
    if (!ck(cudaEventRecord(ev_d_end, st_d), "staging leg2 end")) { ok = false; break; }
    if (!ck(cudaEventSynchronize(ev_d_end), "staging leg2 sync")) { ok = false; break; }
    if (!ck(cudaEventElapsedTime(&t2, ev_d_start, ev_d_end), "staging leg2 elapsed")) { ok = false; break; }
    cudaSetDevice(s);
    cudaDeviceSynchronize();
    cudaSetDevice(d);
    cudaDeviceSynchronize();
    cudaSetDevice(s);
    leg1[r] = t1; leg2[r] = t2;
    if (r == 0) {
      // Verify the head of the final dst against the pattern (the whole pass's data check).
      unsigned char h[64] = {};
      cudaSetDevice(d);
      ok = (cudaMemcpy(h, dst, sizeof h, cudaMemcpyDeviceToHost) == cudaSuccess);
      cudaSetDevice(s);
      if (!ok) {
        fprintf(stderr, "  staging card%d->card%d: first pass dst unreadable\n", s, d);
        break;
      }
      if (memcmp(h, pat.data(), sizeof h) != 0) {
        fprintf(stderr, "  staging card%d->card%d: first copy MISMATCH\n", s, d);
        ok = false;
        break;
      }
    }
  }
  cudaSetDevice(s);
  if (src) cudaFree(src);
  if (st_s) cudaStreamDestroy(st_s);
  if (ev_s_start) cudaEventDestroy(ev_s_start);
  if (ev_s_end) cudaEventDestroy(ev_s_end);
  cudaSetDevice(d);
  if (dst) cudaFree(dst);
  if (st_d) cudaStreamDestroy(st_d);
  if (ev_d_start) cudaEventDestroy(ev_d_start);
  if (ev_d_end) cudaEventDestroy(ev_d_end);
  if (!ok) return false;
  std::sort(leg1.begin(), leg1.end());
  std::sort(leg2.begin(), leg2.end());
  double m1 = leg1[reps / 2], m2 = leg2[reps / 2];
  printf("  staging card%d->card%d %6zu B:  leg1 d2h %7.2f GB/s  leg2 h2d %7.2f GB/s  total %7.2f GB/s  (%d reps)\n",
         s, d, bytes, bytes / 1e9 / (m1 / 1e3), bytes / 1e9 / (m2 / 1e3),
         bytes / 1e9 / ((m1 + m2) / 1e3), reps);
  return true;
}

// ---------------------------------------------------------------- [5] NCCL smoke
// One worker thread per rank; the thread's FIRST act is cudaSetDevice(own card) (the CUDA rule
// that shapes the whole tp=2 design: contexts/streams are device-bound, setDevice is
// thread-local). Rank 0 generates the ncclUniqueId in-process and publishes it to rank 1's
// thread through an atomic pair of flags -- no network, no pipe. A rank whose peer's uniqueId
// is unusable or never published records the failure and returns WITHOUT entering the
// collective (a lone rank in ncclCommInitRank wedges in bootstrap until the watchdog). All
// NCCL calls happen inside a window in which fd 2 is redirected to the log file, so the
// NCCL_DEBUG=INFO record (which transport was selected, at what bandwidth) lands in the record
// file, not in the probe's stdout.
struct Smoke {
  NcclFns fns;
  NcclUniqueId uid{};
  std::atomic<bool> uid_valid{false};
  std::atomic<bool> uid_ready{false};
  std::atomic<int> init_ok{0};
  std::atomic<int> init_fail{0};
  std::atomic<int> allreduce_done{0};
  std::atomic<int> allreduce_bad{0};
  std::vector<float> res[2];
};

static void smoke_worker(Smoke* S, int rank) {
  const int N = 1024;
  // First thing: own the device. (One context per device; this thread drives only this one.)
  cudaSetDevice(rank);

  if (rank == 1) {
    // Wait for rank 0's uniqueId (published in-process). Bounded wait: a hang here is a bug in
    // this probe, not in NCCL, so fail loudly instead of spinning forever.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!S->uid_ready.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    if (!S->uid_ready.load(std::memory_order_acquire) || !S->uid_valid.load(std::memory_order_acquire)) {
      fprintf(stderr, "[nccl-smoke] rank1: peer uniqueId unusable or never published; not entering the collective\n");
      S->init_fail++;
      return;
    }
  } else {
    // Rank 0: generate and publish. ncclGetUniqueId is not a collective.
    NcclResult r = S->fns.getUniqueId(&S->uid);
    if (r != NCCL_OK) {
      fprintf(stderr, "[nccl-smoke] rank0: ncclGetUniqueId: %s\n", nccl_err(S->fns, r));
      S->uid_valid.store(false);
      S->uid_ready.store(true, std::memory_order_release);   // valid-flag first, ready second
      S->init_fail++;
      return;   // rank1 sees ready && !valid and stops without entering the collective
    }
    S->uid_valid.store(true);
    S->uid_ready.store(true, std::memory_order_release);
  }

  // Per-rank send/recv buffers on THIS device.
  std::vector<float> hsend(N, (float)(rank + 1));   // rank0: 1.0, rank1: 2.0 -> sum 3.0
  std::vector<float> hrecv(N, 0.f);
  void* dev = nullptr;
  cudaStream_t stream = nullptr;
  if (!ck(cudaMalloc(&dev, N * sizeof(float)), "nccl malloc dev")) { S->init_fail++; return; }
  if (!ck(cudaMemcpy(dev, hsend.data(), N * sizeof(float), cudaMemcpyHostToDevice), "nccl h2d")) {
    cudaFree(dev); S->init_fail++; return;
  }
  if (!ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "nccl stream")) {
    cudaFree(dev); S->init_fail++; return;
  }

  void* comm = nullptr;
  NcclResult r = S->fns.commInitRank(&comm, 2, S->uid, rank);
  if (r != NCCL_OK) {
    // R8's answer: the sm_120 prebuild cannot init on this pair. Verbatim, on the record.
    fprintf(stderr, "[nccl-smoke] rank%d: ncclCommInitRank FAILED: %s\n", rank, nccl_err(S->fns, r));
    S->init_fail++;
    cudaStreamDestroy(stream);
    cudaFree(dev);
    return;
  }
  S->init_ok++;
  // Rendezvous: allreduce is a 2-rank collective -- do not issue it until both sides are in.
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (S->init_ok.load() < 2 && S->init_fail.load() == 0 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  if (S->init_ok.load() < 2) {
    fprintf(stderr, "[nccl-smoke] rank%d: peer did not finish ncclCommInitRank in time\n", rank);
    S->allreduce_bad++;
    S->fns.commDestroy(comm);
    cudaStreamDestroy(stream);
    cudaFree(dev);
    return;
  }

  r = S->fns.allReduce(dev, dev, N, NCLC_FLOAT, NCLC_SUM, comm, stream);   // in-place: recv == sum of both sends
  if (!ck(cudaStreamSynchronize(stream), "nccl stream sync")) { S->allreduce_bad++; }
  if (r != NCCL_OK) {
    fprintf(stderr, "[nccl-smoke] rank%d: ncclAllReduce FAILED: %s\n", rank, nccl_err(S->fns, r));
    S->allreduce_bad++;
  }
  cudaMemcpy(hrecv.data(), dev, N * sizeof(float), cudaMemcpyDeviceToHost);
  S->res[rank].swap(hrecv);
  S->allreduce_done++;

  NcclResult rd = S->fns.commDestroy(comm);
  if (rd != NCCL_OK) fprintf(stderr, "[nccl-smoke] rank%d: ncclCommDestroy: %s\n", rank, nccl_err(S->fns, rd));
  cudaStreamDestroy(stream);
  cudaFree(dev);
}

// Returns 1 when the smoke produced a verdict (including "init failed" = the R8 answer),
// -1 when the probe's own harness failed (dlopen/symbols, watchdog timeout).
static int run_nccl_smoke(const char* libpath, const char* logpath) {
  printf("=== [5] NCCL smoke (dlopen %s) ===\n", libpath);
  printf("  NCCL_DEBUG=INFO  NCCL_DEBUG_SUBSYS=INIT,NET,SHM   (NCCL stderr -> %s)\n", logpath);
  setenv("NCCL_DEBUG", "INFO", 1);
  setenv("NCCL_DEBUG_SUBSYS", "INIT,NET,SHM", 1);

  int saved2 = dup(2);
  FILE* lf = fopen(logpath, "w");
  if (!lf) {
    fprintf(stderr, "  cannot open %s for the NCCL log; falling back to /dev/null\n", logpath);
    lf = fopen("/dev/null", "w");
    if (!lf) saved2 = -1;
  }
  if (saved2 >= 0 && lf && dup2(fileno(lf), 2) < 0) saved2 = -1;
  auto restore_stderr = [&]() {
    if (saved2 >= 0) { dup2(saved2, 2); close(saved2); }
    if (lf) fclose(lf);
  };

  Smoke S;
  S.fns.lib = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
  if (!S.fns.lib) {
    const char* e = dlerror();
    fprintf(stderr, "  dlopen FAILED: %s\n", e ? e : "(no message)");
    restore_stderr();
    return -1;   // the box has no loadable libnccl at that path -> WP-3's fallback question is live
  }
  auto sym = [&](const char* n, void* out) {
    void* p = dlsym(S.fns.lib, n);
    if (!p) fprintf(stderr, "  dlsym(%s) FAILED\n", n);
    *reinterpret_cast<void**>(out) = p;
    return p != nullptr;
  };
  if (!(sym("ncclGetVersion", (void*)&S.fns.getVersion) &&
        sym("ncclGetUniqueId", (void*)&S.fns.getUniqueId) &&
        sym("ncclCommInitRank", (void*)&S.fns.commInitRank) &&
        sym("ncclAllReduce", (void*)&S.fns.allReduce) &&
        sym("ncclCommDestroy", (void*)&S.fns.commDestroy) &&
        sym("ncclGetErrorString", (void*)&S.fns.getErrorString))) {
    fprintf(stderr, "  one or more NCCL symbols missing; aborting smoke\n");
    restore_stderr();
    return -1;
  }
  int ver = 0;
  S.fns.getVersion(&ver);
  printf("  ncclGetVersion() = %d  (major %d, minor %d, patch %d)\n", ver, ver / 10000, (ver / 100) % 100, ver % 100);

  std::thread w0([&] { smoke_worker(&S, 0); });
  std::thread w1([&] { smoke_worker(&S, 1); });
  // Watchdog: a healthy smoke takes seconds; a 120 s cap keeps a wedged NCCL init from hanging
  // the probe (process exit then reclaims every allocation the driver gave this process; the
  // LLM runs in a separate process and is never touched by that).
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (S.allreduce_done.load() + S.init_fail.load() + S.allreduce_bad.load() < 2) {
    if (std::chrono::steady_clock::now() > deadline) {
      fprintf(stderr, "  NCCL smoke TIMED OUT (120 s) -- init or allreduce wedged; see %s\n", logpath);
      restore_stderr();
      w0.detach(); w1.detach();
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  w0.join();
  w1.join();
  restore_stderr();

  if (S.init_fail.load() > 0) {
    printf("  RESULT: ncclCommInitRank FAILED (R8 verdict: this NCCL build cannot init on this pair) -- see %s for the verbatim error\n", logpath);
    return 1;   // a result, not a failure of this probe
  }
  if (S.allreduce_done.load() != 2) {
    printf("  RESULT: allreduce did not complete on both ranks -- see %s\n", logpath);
    return 1;
  }
  // Both ranks reported; verify the math: rank0 sent 1.0, rank1 sent 2.0 -> every element 3.0.
  bool sum_ok = true;
  for (int i = 0; i < 1024; i++) {
    if (S.res[0][i] != 3.0f || S.res[1][i] != 3.0f) { sum_ok = false; break; }
  }
  if (!sum_ok) {
    // Verbatim forensics: what did each rank actually receive? (own-send echo, mix, or garbage)
    auto diag = [&](int r) {
      float mn = 1e30f, mx = -1e30f;
      int n3 = 0;
      for (int i = 0; i < 1024; i++) {
        float v = S.res[r][i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (v == 3.0f) n3++;
      }
      printf("    rank%d recv[0..7] = %g %g %g %g %g %g %g %g   (min %g max %g, %d/1024 == 3.0f; rank%d's own send was %g)\n",
             r, S.res[r][0], S.res[r][1], S.res[r][2], S.res[r][3],
             S.res[r][4], S.res[r][5], S.res[r][6], S.res[r][7], mn, mx, n3, r, (float)(r + 1));
    };
    diag(0);
    diag(1);
  }
  printf("  RESULT: 2-rank ncclAllReduce of 1024 float32 completed; per-element sum check: %s\n",
         sum_ok ? "OK (all 1024 == 3.0f)" : "MISMATCH");
  printf("  (which transport NCCL selected, and its measured bandwidth: %s)\n", logpath);
  return sum_ok ? 1 : -1;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  std::string lib = "/home/manuel/cifarTest_cuda/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2";
  std::string log = "dual-gpu/00-p0-probe/nccl_smoke_run1.log";
  bool do_nccl = true;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--no-nccl") { do_nccl = false; continue; }
    if (a.size() > 2 && a[0] != '-' && i <= 2) {
      if (i == 1) lib = a; else log = a;
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  int ndev = 0;
  int rv = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess) {
    fprintf(stderr, "p2p_probe: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  cudaRuntimeGetVersion(&rv);
  time_t tt = time(nullptr);
  char tbuf[64] = "n/a";
  strftime(tbuf, sizeof tbuf, "%Y-%m-%d %H:%M:%S", localtime(&tt));
  printf("p2p_probe: %d GPU(s), CUDA runtime %d.%d, started %s (epoch %lld)\n",
         ndev, rv / 1000, (rv / 10) % 100, tbuf, (long long)tt);
  if (ndev < 2) {
    printf("  needs exactly 2 GPUs (saw %d); aborting\n", ndev);
    return 1;
  }

  print_all_cards("[1] per-card identity and VRAM (before the probe)");

  PeerInfo pi = probe_peer_access();
  printf("=== [2] cudaDeviceCanAccessPeer ===\n");
  printf("  card0 -> card1 : %s\n  card1 -> card0 : %s\n  (topology for this pair: PHB, PCIe + host bridge -- no NVLink; any P2P is root-complex-routed at best, see dual-gpu/01)\n",
         pi.can01 ? "yes" : "no", pi.can10 ? "yes" : "no");

  int rc = 0;
  if (!run_d2d(pi)) rc = 1;

  printf("=== [4] pinned-host staging (device -> pinned -> device; always available) ===\n");
  void* pinned = nullptr;
  if (cudaHostAlloc(&pinned, 64u << 20, cudaHostAllocDefault) != cudaSuccess) {
    fprintf(stderr, "  staging: cudaHostAlloc 64 MiB pinned failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    rc = 1;
  }
  if (pinned) {
    for (size_t bytes : {1u << 20, 16u << 20, 64u << 20})
      if (!run_staging(0, bytes, 8, pinned)) rc = 1;
    for (size_t bytes : {1u << 20, 16u << 20, 64u << 20})
      if (!run_staging(1, bytes, 8, pinned)) rc = 1;
    cudaFreeHost(pinned);   // the one host allocation, freed on the way out
  }

  if (do_nccl) {
    int nrc = run_nccl_smoke(lib.c_str(), log.c_str());
    if (nrc < 0) rc = 1;
  } else {
    printf("=== [5] NCCL smoke: SKIPPED (--no-nccl) ===\n");
  }

  print_all_cards("[6] per-card VRAM (after the probe; must match [1] within a few MB)");
  auto t1 = std::chrono::steady_clock::now();
  printf("probe wall time: %ld ms  (D2D/staging above ran with the LLM resident -- contention applies; see the plan's R7)\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  return rc;
}
