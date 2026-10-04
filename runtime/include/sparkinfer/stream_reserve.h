#pragma once

// Device memory held back at load for the per-stream buffers of concurrent requests (stream
// slots): each session's recurrent state and sampling arrays, the drafter's per-session KV and
// capture rows. Those used to be cudaMalloc'd when a stream started, out of whatever the KV pool
// and the reserved prefill scratch had left -- so a pool sized close to the card's limit loaded
// fine and later streams lost their drafter (or failed) when they could not get memory. With a
// reserve, the first N streams' buffers come out of memory taken at load, and a deployment
// whose pool leaves no room for them fails to load instead.
//
// One reserve per device, a first-fit sub-allocator over a single allocation. sr_malloc tries
// the current device's reserve and falls back to cudaMalloc; sr_free returns either kind.
// Thread-safe.

#include <cstddef>
#include <cuda_runtime.h>

namespace sparkinfer {

// Take `bytes` on `device` as its reserve (replacing none; false if it does not fit).
bool stream_reserve_init(int device, size_t bytes);
// Release every device's reserve (blocks still handed out stay valid until freed).
void stream_reserve_release_all();
// The current device's reserve first, cudaMalloc when it has no room.
cudaError_t sr_malloc(void** p, size_t bytes);
// Free what sr_malloc returned (reserve or cudaMalloc). Null is a no-op.
void sr_free(void* p);
struct StreamReserveStats { size_t bytes = 0, used = 0, high_water = 0; unsigned long long fallbacks = 0; };
StreamReserveStats stream_reserve_stats(int device);

}  // namespace sparkinfer
