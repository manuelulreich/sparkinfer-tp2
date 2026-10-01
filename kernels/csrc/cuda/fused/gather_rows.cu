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

// Vocab-row-split embedding (tp>1): this rank's table holds rows [v0, v0+vcount) of the full
// vocabulary. out[i] = table[ids[i] - v0] when ids[i] is in the window, else a zero row -- so a
// sum (all-reduce) over the ranks reconstructs the full embedding exactly.
__global__ void embedding_vocab_window_kernel(const int* __restrict__ ids,
                                              const __nv_bfloat16* __restrict__ table,
                                              __nv_bfloat16* __restrict__ out, int hidden,
                                              int v0, int vcount) {
    const int row = blockIdx.x;
    const int id = ids[row] - v0;
    const bool own = id >= 0 && id < vcount;
    __nv_bfloat16* o = out + (size_t)row * hidden;
    const __nv_bfloat16* t = own ? table + (size_t)id * hidden : nullptr;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        o[c] = own ? t[c] : __float2bfloat16(0.f);
}

void launch_embedding_vocab_window(const int* ids, const void* table, void* out, int n_tokens,
                                   int hidden, int v0, int vcount, cudaStream_t stream) {
    if (n_tokens <= 0) return;
    embedding_vocab_window_kernel<<<n_tokens, 256, 0, stream>>>(
        ids, reinterpret_cast<const __nv_bfloat16*>(table),
        reinterpret_cast<__nv_bfloat16*>(out), hidden, v0, vcount);
}

}  // namespace kernels
}  // namespace sparkinfer
