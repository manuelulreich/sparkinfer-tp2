#pragma once
// Quantized paged-KV element formats shared by the KV writers (rope.cu, batched_prefill.cu) and
// the readers (flash_decode_split.cu, the prefill dequant). The format code is KVDtype from
// kv_cache.h:
//
//   1 KV_INT8   one int8 per element; per-(token, kv_head) fp16 scale d = amax/127.
//   2 KV_FP8    one e4m3 byte per element; same fp16 scale pool with d = amax/448. Same bytes per
//               element and the same scale layout as int8 -- only the code changes.
//   3 KV_NVFP4  per (token, kv_head) head vector of HD elements: HD/2 bytes of e2m1 nibbles
//               (element 2i in the low nibble of byte i, 2i+1 in the high one) followed by HD/16
//               e4m3 block scales (one per 16 elements), i.e. HD*9/16 bytes per row. The fp16 scale
//               pool holds the second-level scale g = amax/(6*448), so element
//               x = e2m1 * e4m3(block) * g.
//
// Element offsets stay what every kernel already computes ((tok * n_kv + h) * HD + d); a format
// converts an element offset of a row start to bytes with kvq_bytes().
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace sparkinfer {
namespace kernels {

enum : int { KVQ_BF16 = 0, KVQ_INT8 = 1, KVQ_FP8 = 2, KVQ_NVFP4 = 3 };

__host__ __device__ __forceinline__ size_t kvq_bytes(int fmt, size_t elems) {
    return fmt == KVQ_NVFP4 ? elems / 16 * 9 : (fmt == KVQ_BF16 ? elems * 2 : elems);
}

__device__ __forceinline__ unsigned char kvq_f2e4m3(float x) {
    return (unsigned char)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float kvq_e4m3f(unsigned char b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(__half(h));
}
// e2m1 code of x (|x| <= 6 after scaling): values 0, .5, 1, 1.5, 2, 3, 4, 6, round to nearest.
__device__ __forceinline__ unsigned kvq_f2e2m1(float x) {
    const float a = fabsf(x);
    unsigned c;
    if      (a < 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a < 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a < 2.5f)  c = 4;
    else if (a < 3.5f)  c = 5;
    else if (a < 5.0f)  c = 6;
    else                c = 7;
    return c | ((x < 0.f && c) ? 8u : 0u);
}
__device__ __forceinline__ float kvq_e2m1f(unsigned nib) {
    // magnitudes for codes 0..7: 0 .5 1 1.5 2 3 4 6
    const unsigned m = nib & 7u;
    const float v = (m < 4u) ? 0.5f * (float)m : (float)(1u << (m - 2u)) * ((m & 1u) ? 1.5f : 1.0f);
    return (nib & 8u) ? -v : v;
}

// Store one element of a head vector. Called by EVERY thread of the row (thread t = element t,
// head_dim threads, whole warps; fmt is block-uniform) after the block has reduced the row's
// max-abs `amax`. pool is the layer's pool base (bytes), dst the row's ELEMENT offset, sidx the
// row's scale index. int8 is the exact arithmetic of the original writers.
__device__ __forceinline__ void kvq_store(int fmt, void* pool, __half* scale_pool, size_t dst,
                                          size_t sidx, int head_dim, int t, float val, float amax) {
    if (fmt == KVQ_FP8) {
        const __half dh = __float2half(amax / 448.0f);
        const float d = __half2float(dh);
        const unsigned char b = (amax == 0.f || d == 0.f) ? (unsigned char)0 : kvq_f2e4m3(val / d);
        reinterpret_cast<unsigned char*>(pool)[dst + t] = b;
        if (t == 0) scale_pool[sidx] = dh;
    } else if (fmt == KVQ_NVFP4) {
        const __half gh = __float2half(amax / (6.0f * 448.0f));
        const float g = __half2float(gh);
        float bam = fabsf(val);   // 16-element block max-abs: lanes [16j, 16j+16) of the warp
        #pragma unroll
        for (int m = 8; m > 0; m >>= 1) bam = fmaxf(bam, __shfl_xor_sync(0xffffffff, bam, m));
        const unsigned char sb = (g == 0.f) ? (unsigned char)0 : kvq_f2e4m3(bam / (6.0f * g));
        const float sbf = kvq_e4m3f(sb) * g;
        const unsigned code = (sbf == 0.f) ? 0u : kvq_f2e2m1(val / sbf);
        const unsigned other = __shfl_xor_sync(0xffffffff, code, 1);
        unsigned char* row = reinterpret_cast<unsigned char*>(pool) + dst / 16 * 9;
        if ((t & 1) == 0) row[t >> 1] = (unsigned char)(code | (other << 4));
        if ((t & 15) == 0) row[(head_dim >> 1) + (t >> 4)] = sb;
        if (t == 0) scale_pool[sidx] = gh;
    } else {
        const float d  = amax / 127.0f;
        const int   qi = (amax == 0.f) ? 0 : (int)roundf(val / d);
        reinterpret_cast<signed char*>(pool)[dst + t] = (signed char)qi;
        if (t == 0) scale_pool[sidx] = __float2half(d);
    }
}

// Dequantize 8 consecutive elements [e, e+8) of a row (e % 8 == 0) of a quantized pool. `elem` is
// the absolute element offset (row start + dim), head_dim the row length, s the row's fp16 scale.
template <int FMT>
__device__ __forceinline__ void kvq_load8(const void* __restrict__ pool, size_t elem, int head_dim,
                                          float s, float out[8]) {
    if constexpr (FMT == KVQ_INT8) {
        const int2 r = __ldg(reinterpret_cast<const int2*>(reinterpret_cast<const signed char*>(pool) + elem));
        const signed char* c = reinterpret_cast<const signed char*>(&r);
        #pragma unroll
        for (int j = 0; j < 8; j++) out[j] = (float)c[j] * s;
    } else if constexpr (FMT == KVQ_FP8) {
        const uint2 r = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const unsigned char*>(pool) + elem));
        const unsigned w[2] = {r.x, r.y};
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            #pragma unroll
            for (int p = 0; p < 2; p++) {
                const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
                    (__nv_fp8x2_storage_t)((w[h] >> (16 * p)) & 0xffffu), __NV_E4M3);
                const float2 f = __half22float2(__half2(hr));
                out[h * 4 + p * 2]     = f.x * s;
                out[h * 4 + p * 2 + 1] = f.y * s;
            }
        }
    } else {   // NVFP4
        const size_t d = elem % (size_t)head_dim;
        const unsigned char* row = reinterpret_cast<const unsigned char*>(pool) + (elem - d) / 16 * 9;
        const unsigned w = __ldg(reinterpret_cast<const unsigned*>(row + (d >> 1)));
        const float bs = kvq_e4m3f(__ldg(row + (head_dim >> 1) + (d >> 4))) * s;
        #pragma unroll
        for (int j = 0; j < 8; j++) out[j] = kvq_e2m1f((w >> (4 * j)) & 15u) * bs;
    }
}

// Scalar dequant of one element (slow paths only).
__device__ __forceinline__ float kvq_load1(int fmt, const void* __restrict__ pool, size_t elem,
                                           int head_dim, float s) {
    if (fmt == KVQ_FP8) return kvq_e4m3f(reinterpret_cast<const unsigned char*>(pool)[elem]) * s;
    if (fmt == KVQ_NVFP4) {
        const size_t d = elem % (size_t)head_dim;
        const unsigned char* row = reinterpret_cast<const unsigned char*>(pool) + (elem - d) / 16 * 9;
        const unsigned b = row[d >> 1];
        return kvq_e2m1f((d & 1) ? (b >> 4) : (b & 15u)) * kvq_e4m3f(row[(head_dim >> 1) + (d >> 4)]) * s;
    }
    return (float)reinterpret_cast<const signed char*>(pool)[elem] * s;
}

}  // namespace kernels
}  // namespace sparkinfer
