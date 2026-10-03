// The fp8 (e4m3) prefill attention, built for sm_120a on its own (see kernels/CMakeLists.txt,
// si_attn_f8): the block-scaled mma that runs e4m3 at the full int8 rate only assembles for the
// arch-specific target. prefill_attn_mma.cu holds the kernel; this translation unit compiles it
// with SPARKINFER_ATTN_F8_TU so that only launch_prefill_attn_mma_f8 is emitted here.
#define SPARKINFER_ATTN_F8_TU 1
#include "prefill_attn_mma.cu"
