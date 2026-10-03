// WP-3 elementwise reduce kernels behind GpuLink's allreduce/maxreduce:
//
//   f32:  dst[i] = a[i] + b[i]            (sum)  |  dst[i] = max(a[i], b[i])   (max)
//   bf16: float(a[i]) op float(b[i]), rounded back to bf16 (RN)
//   f16:  float(a[i]) op float(b[i]), rounded back to f16 (RN)
//
// Grid-stride over 256-wide blocks; the grid is capped at 512 blocks (the 36-SM RTX 5060 Ti
// is saturated well below that) and never below 1.
//
// Max semantics: `out = (x > y) ? x : y` — a strict-greater rule, so every tie (including
// -0.0 vs +0.0, the only bit-distinct equal pair) resolves to b, the peer operand: a
// deterministic, transport-order-invariant choice. (NaN in either operand poisons the
// result, b winning a NaN-vs-NaN tie.)
//
// Portable CUDA — same arch policy as the rest of csrc/cuda (sm_89 .. sm_120).

#include <cuda_bf16.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
// The module header must be included at GLOBAL scope: a #include inside a namespace
// pastes its standard includes (<memory> -> <vector> -> ...) into that namespace,
// opening `namespace std` inside sparkinfer and shadowing ::std for everything
// included after (kv_ops.cu gets away with an in-namespace include only because
// kv_ops.h carries C headers alone).
#include "sparkinfer/gpu_link.h"
#include <climits>
#endif

namespace sparkinfer {

__global__ void glink_reduce_f32(float* __restrict__ dst, const float* __restrict__ a,
                                  const float* __restrict__ b, int n, int is_max) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float x = a[i];
    const float y = b[i];
    dst[i] = is_max ? (x > y ? x : y) : (x + y);
  }
}

__global__ void glink_reduce_bf16(__nv_bfloat16* __restrict__ dst,
                                   const __nv_bfloat16* __restrict__ a,
                                   const __nv_bfloat16* __restrict__ b, int n, int is_max) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float x = __bfloat162float(a[i]);
    const float y = __bfloat162float(b[i]);
    const float r = is_max ? (x > y ? x : y) : (x + y);
    dst[i] = __float2bfloat16(r);
  }
}

__global__ void glink_reduce_f16(__half* __restrict__ dst, const __half* __restrict__ a,
                                  const __half* __restrict__ b, int n, int is_max) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float x = __half2float(a[i]);
    const float y = __half2float(b[i]);
    const float r = is_max ? (x > y ? x : y) : (x + y);
    dst[i] = __float2half_rn(r);
  }
}


// Flag all-reduce (see gpu_link.h). One block; each thread pushes, then fences its own stores
// system-wide before the block barrier, so thread 0's flag store is ordered after every push.
// The landing buffer is read with ld.global.cg (L2, bypassing L1): the peer wrote it over P2P,
// which L1 does not observe.
// The push and the reduce go 16 bytes a thread when the size and every pointer allow: element-
// wide stores leave the peer link as warp-sized 64-byte writes, which PCIe carries at a fraction
// of its rate (measured: the extra 130 KB of a 20-row verify cost 31 us, 4.2 GB/s on a Gen3 x8
// link; vectorized, a C4 verify step is 2 ms shorter). Same values either way -- only the
// transaction size changes. More blocks (one flag slot each) measured no further gain.
template <typename T>
__global__ void glink_flag_allreduce_kernel(const T* __restrict__ in, T* out, T* peer_land,
                                            const T* my_land, unsigned* peer_flag,
                                            const unsigned* my_flag, unsigned seq, int n) {
  constexpr int PER = 16 / (int)sizeof(T);
  const bool vec = (n % PER) == 0 &&
                   ((reinterpret_cast<uintptr_t>(in) | reinterpret_cast<uintptr_t>(out) |
                     reinterpret_cast<uintptr_t>(peer_land) |
                     reinterpret_cast<uintptr_t>(my_land)) & 15) == 0;
  const int nv = n / PER;
  if (vec) {
    for (int i = threadIdx.x; i < nv; i += blockDim.x)
      reinterpret_cast<uint4*>(peer_land)[i] = reinterpret_cast<const uint4*>(in)[i];
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) peer_land[i] = in[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *(volatile unsigned*)peer_flag = seq;
    const long long t0 = clock64();
    while (*(const volatile unsigned*)my_flag < seq) {
      if (clock64() - t0 > 30000000000LL) __trap();   // ~10+ s: the peer rank is gone
    }
    __threadfence();
  }
  __syncthreads();
  if (vec) {
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
      const uint4 a = reinterpret_cast<const uint4*>(in)[i];
      const uint4 b = __ldcg(reinterpret_cast<const uint4*>(my_land) + i);
      uint4 r;
      const T* ta = reinterpret_cast<const T*>(&a);
      const T* tb = reinterpret_cast<const T*>(&b);
      T* tr = reinterpret_cast<T*>(&r);
      #pragma unroll
      for (int e = 0; e < PER; e++) {
        if constexpr (sizeof(T) == 4) tr[e] = ta[e] + tb[e];
        else tr[e] = __float2bfloat16(__bfloat162float(ta[e]) + __bfloat162float(tb[e]));
      }
      reinterpret_cast<uint4*>(out)[i] = r;
    }
    return;
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float x, y;
    if constexpr (sizeof(T) == 4) { x = in[i]; y = __ldcg(&my_land[i]); }
    else { x = __bfloat162float(in[i]); y = __bfloat162float(__ldcg(&my_land[i])); }
    const float r = x + y;
    if constexpr (sizeof(T) == 4) out[i] = r; else out[i] = __float2bfloat16(r);
  }
}

// Compressed wire (opt-in, prefill only): a bf16 partial as blocks of 128 values, one 8-bit code
// each (e4m3 or int8) followed by one fp32 scale per block -- [n codes][n/128 scales], 132 bytes
// per 128 values instead of 256. One warp per block, four values a lane.
template <int FMT>   // 1 e4m3 (scale amax/448), 2 int8 (scale amax/127)
__global__ void glink_wire_quant_bf16(const __nv_bfloat16* __restrict__ in,
                                      unsigned char* __restrict__ wire, int nblk) {
  const int lane = threadIdx.x & 31;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  float* scales = reinterpret_cast<float*>(wire + (size_t)nblk * 128);
  for (int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; blk < nblk; blk += warps) {
    const uint2 raw = *reinterpret_cast<const uint2*>(in + (size_t)blk * 128 + lane * 4);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
    float f[4], amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; j++) { f[j] = __bfloat162float(v[j]); amax = fmaxf(amax, fabsf(f[j])); }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / (FMT == 1 ? 448.0f : 127.0f);
    const float inv = (amax > 0.f) ? 1.0f / d : 0.f;
    unsigned w = 0u;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      unsigned c;
      if constexpr (FMT == 1)
        c = (unsigned)__nv_cvt_float_to_fp8(f[j] * inv, __NV_SATFINITE, __NV_E4M3);
      else
        c = (unsigned)(unsigned char)(signed char)__float2int_rn(f[j] * inv);
      w |= c << (8 * j);
    }
    *reinterpret_cast<unsigned*>(wire + (size_t)blk * 128 + lane * 4) = w;
    if (lane == 0) scales[blk] = d;
  }
}

template <int FMT>
__device__ __forceinline__ float glink_wire_deq(unsigned w, int j, float d) {
  const unsigned c = (w >> (8 * j)) & 0xffu;
  if constexpr (FMT == 1) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3);
    return __half2float(__half(h)) * d;
  } else {
    return (float)(signed char)c * d;
  }
}

// out = bf16(deq(own) + deq(peer)). Both ranks add the SAME two dequantized values (fp32 addition
// commutes), so the replicated sum stays bit-identical across the ranks.
template <int FMT>
__global__ void glink_wire_reduce_bf16(__nv_bfloat16* __restrict__ out,
                                       const unsigned char* __restrict__ own,
                                       const unsigned char* __restrict__ peer, int nblk) {
  const int lane = threadIdx.x & 31;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  const float* so = reinterpret_cast<const float*>(own + (size_t)nblk * 128);
  const float* sp = reinterpret_cast<const float*>(peer + (size_t)nblk * 128);
  for (int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; blk < nblk; blk += warps) {
    const unsigned wo = *reinterpret_cast<const unsigned*>(own + (size_t)blk * 128 + lane * 4);
    const unsigned wp = *reinterpret_cast<const unsigned*>(peer + (size_t)blk * 128 + lane * 4);
    const float d_o = so[blk], d_p = sp[blk];
    __align__(8) __nv_bfloat16 r[4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
      r[j] = __float2bfloat16(glink_wire_deq<FMT>(wo, j, d_o) + glink_wire_deq<FMT>(wp, j, d_p));
    *reinterpret_cast<uint2*>(out + (size_t)blk * 128 + lane * 4) = *reinterpret_cast<const uint2*>(r);
  }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY

namespace detail {

cudaError_t launch_glink_wire_quant(const void* in, void* wire, size_t n, int fmt,
                                    cudaStream_t stream) {
  if (n == 0 || n % 128 != 0 || n / 128 > (size_t)INT_MAX) return cudaErrorInvalidValue;
  const int nblk = (int)(n / 128);
  int blocks = (nblk + 7) / 8;
  if (blocks > 1024) blocks = 1024;
  (void)cudaGetLastError();
  if (fmt == 1)
    glink_wire_quant_bf16<1><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)in,
                                                         (unsigned char*)wire, nblk);
  else
    glink_wire_quant_bf16<2><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)in,
                                                         (unsigned char*)wire, nblk);
  return cudaGetLastError();
}

cudaError_t launch_glink_wire_reduce(void* out, const void* own, const void* peer, size_t n,
                                     int fmt, cudaStream_t stream) {
  if (n == 0 || n % 128 != 0 || n / 128 > (size_t)INT_MAX) return cudaErrorInvalidValue;
  const int nblk = (int)(n / 128);
  int blocks = (nblk + 7) / 8;
  if (blocks > 1024) blocks = 1024;
  (void)cudaGetLastError();
  if (fmt == 1)
    glink_wire_reduce_bf16<1><<<blocks, 256, 0, stream>>>(
        (__nv_bfloat16*)out, (const unsigned char*)own, (const unsigned char*)peer, nblk);
  else
    glink_wire_reduce_bf16<2><<<blocks, 256, 0, stream>>>(
        (__nv_bfloat16*)out, (const unsigned char*)own, (const unsigned char*)peer, nblk);
  return cudaGetLastError();
}

cudaError_t launch_glink_reduce(void* dst, const void* a, const void* b, size_t n,
                                 GpuLink::Dtype dtype, bool is_max, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;  // nothing to reduce
  if (n > (size_t)INT_MAX) return cudaErrorInvalidValue;  // the kernels index by int
  const int ni = (int)n;
  int blocks = (ni + 255) / 256;
  if (blocks > 512) blocks = 512;
  const int flag = is_max ? 1 : 0;
  dim3 grid(blocks);
  // Clear the thread's error slot first: cudaGetLastError below must report THIS launch, not an
  // earlier, already-handled failure on the thread (a declined cudaMalloc elsewhere), which would
  // otherwise be read as a failed collective and take the whole tp group down.
  (void)cudaGetLastError();
  if (dtype == GpuLink::Dtype::Float32)
    glink_reduce_f32<<<grid, 256, 0, stream>>>((float*)dst, (const float*)a, (const float*)b, ni, flag);
  else if (dtype == GpuLink::Dtype::BFloat16)
    glink_reduce_bf16<<<grid, 256, 0, stream>>>((__nv_bfloat16*)dst, (const __nv_bfloat16*)a,
                                                 (const __nv_bfloat16*)b, ni, flag);
  else
    glink_reduce_f16<<<grid, 256, 0, stream>>>((__half*)dst, (const __half*)a, (const __half*)b, ni, flag);
  // A stream-bound launch reports configuration failures (invalid dim3, unknown stream) here;
  // the caller treats any non-success as an op failure.
  return cudaGetLastError();
}

bool preload_glink_flag_kernels() {
  cudaFuncAttributes fa;
  return cudaFuncGetAttributes(&fa, glink_flag_allreduce_kernel<float>) == cudaSuccess &&
         cudaFuncGetAttributes(&fa, glink_flag_allreduce_kernel<__nv_bfloat16>) == cudaSuccess &&
         cudaFuncGetAttributes(&fa, glink_reduce_bf16) == cudaSuccess &&
         cudaFuncGetAttributes(&fa, glink_reduce_f32) == cudaSuccess;
}

cudaError_t launch_glink_flag_allreduce(const void* in, void* out, void* peer_land,
                                        const void* my_land, unsigned* peer_flag,
                                        const unsigned* my_flag, unsigned seq, size_t n,
                                        GpuLink::Dtype dtype, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  if (n > (size_t)INT_MAX) return cudaErrorInvalidValue;
  (void)cudaGetLastError();   // report this launch only (see launch_glink_reduce)
  if (dtype == GpuLink::Dtype::Float32)
    glink_flag_allreduce_kernel<float><<<1, 1024, 0, stream>>>(
        (const float*)in, (float*)out, (float*)peer_land, (const float*)my_land, peer_flag,
        my_flag, seq, (int)n);
  else if (dtype == GpuLink::Dtype::BFloat16)
    glink_flag_allreduce_kernel<__nv_bfloat16><<<1, 1024, 0, stream>>>(
        (const __nv_bfloat16*)in, (__nv_bfloat16*)out, (__nv_bfloat16*)peer_land,
        (const __nv_bfloat16*)my_land, peer_flag, my_flag, seq, (int)n);
  else
    return cudaErrorInvalidValue;
  return cudaGetLastError();
}

}  // namespace detail

#endif

}  // namespace sparkinfer
