// Generic bf16 2D row gather/copy: dst[r*dst_pitch .. +width) = src[r*src_pitch .. +width)
// for r in [0, rows). One __nv_bfloat16 element per thread, grid-strided over rows*width.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

__global__ void gather_rows_kernel(__nv_bfloat16* __restrict__ dst,
                                   const __nv_bfloat16* __restrict__ src,
                                   size_t dst_pitch, size_t src_pitch, size_t width, size_t total) {
    for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < total;
         g += (size_t)blockDim.x * gridDim.x) {
        const size_t row = g / width;
        const size_t col = g % width;
        dst[dst_pitch * row + col] = src[src_pitch * row + col];
    }
}

void launch_gather_rows(void* dst_bf16, size_t dst_pitch, const void* src_bf16, size_t src_pitch,
                        size_t width, size_t rows, cudaStream_t stream) {
    if (rows == 0 || width == 0) return;
    const size_t total = rows * width;
    const int threads = 256;
    const size_t needed = (total + 255) / 256;
    const int grid = needed < 4096 ? (int)needed : 4096;
    gather_rows_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(dst_bf16),
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        dst_pitch, src_pitch, width, total);
}

}  // namespace kernels
}  // namespace sparkinfer
