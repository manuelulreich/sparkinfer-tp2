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
#include <cuda_fp16.h>
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
template <typename T>
__global__ void glink_flag_allreduce_kernel(const T* __restrict__ in, T* out, T* peer_land,
                                            const T* my_land, unsigned* peer_flag,
                                            const unsigned* my_flag, unsigned seq, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) peer_land[i] = in[i];
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
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float x, y;
    if constexpr (sizeof(T) == 4) { x = in[i]; y = __ldcg(&my_land[i]); }
    else { x = __bfloat162float(in[i]); y = __bfloat162float(__ldcg(&my_land[i])); }
    const float r = x + y;
    if constexpr (sizeof(T) == 4) out[i] = r; else out[i] = __float2bfloat16(r);
  }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY

namespace detail {

cudaError_t launch_glink_reduce(void* dst, const void* a, const void* b, size_t n,
                                 GpuLink::Dtype dtype, bool is_max, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;  // nothing to reduce
  if (n > (size_t)INT_MAX) return cudaErrorInvalidValue;  // the kernels index by int
  const int ni = (int)n;
  int blocks = (ni + 255) / 256;
  if (blocks > 512) blocks = 512;
  const int flag = is_max ? 1 : 0;
  dim3 grid(blocks);
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

cudaError_t launch_glink_flag_allreduce(const void* in, void* out, void* peer_land,
                                        const void* my_land, unsigned* peer_flag,
                                        const unsigned* my_flag, unsigned seq, size_t n,
                                        GpuLink::Dtype dtype, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  if (n > (size_t)INT_MAX) return cudaErrorInvalidValue;
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
