#pragma once

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace sparkinfer {

// Pinned host buffers for the prefix cache's snapshots (recurrent state, draft KV). Every agent
// turn takes and drops a few tens of MB of them, and cudaHostAlloc of that size costs 15-35 ms on
// the critical path between prefill passes (measured: ~50 ms a turn). A dropped buffer goes back
// here and the next request of a similar size (capacity within 1.5x) reuses it, and a background
// thread keeps a spare of each size pinned ahead of the next request (the prefix cache holds on to
// its snapshots, so dropped buffers alone rarely come back while a session grows). At most
// kPinnedPoolMaxIdle bytes are kept idle; beyond that a dropped buffer is freed.
namespace pinned_pool_detail {
constexpr size_t kPinnedPoolMaxIdle = size_t(2) << 30;
struct Pool {
    std::mutex mu;
    std::condition_variable cv;
    std::multimap<size_t, void*> idle;   // capacity -> buffer
    size_t idle_bytes = 0;
    std::deque<size_t> refill;           // sizes the background thread pins next
    bool worker = false;
};
inline Pool& pool() {
    static Pool* p = new Pool;   // never destroyed: buffers may be dropped during exit
    return *p;
}
// Keeps spares pinned ahead of use: every snapshot queues a refill of its size, so the next one of
// about that size (the next agent turn's) finds a buffer ready.
inline void pinned_pool_worker() {
    Pool& p = pool();
    std::unique_lock<std::mutex> lk(p.mu);
    for (;;) {
        p.cv.wait(lk, [&] { return !p.refill.empty(); });
        const size_t want = p.refill.front();
        p.refill.pop_front();
        // Two spares a size range (both ranks snapshot alike), 25 % over the last request (the
        // draft's KV snapshot grows a little from turn to turn), 2 MB granules.
        const size_t cap = ((want + want / 4) + (size_t(2) << 20) - 1) & ~((size_t(2) << 20) - 1);
        size_t have = 0;
        for (auto it = p.idle.lower_bound(want); it != p.idle.end() && it->first <= want + want / 2; ++it)
            ++have;
        if (p.idle_bytes + cap > kPinnedPoolMaxIdle || have >= 2) continue;
        lk.unlock();
        void* buf = nullptr;
        const bool ok = cudaHostAlloc(&buf, cap, cudaHostAllocDefault) == cudaSuccess && buf;
        if (!ok) cudaGetLastError();
        lk.lock();
        if (ok) { p.idle.emplace(cap, buf); p.idle_bytes += cap; }
    }
}
}  // namespace pinned_pool_detail

// A pinned buffer of at least `bytes`, or null. The deleter returns it to the pool.
inline std::shared_ptr<void> pinned_pool_get(size_t bytes) {
    using namespace pinned_pool_detail;
    Pool& p = pool();
    void* buf = nullptr;
    size_t cap = 0;
    {
        std::lock_guard<std::mutex> lk(p.mu);
        auto it = p.idle.lower_bound(bytes);
        if (it != p.idle.end() && it->first <= bytes + bytes / 2) {
            cap = it->first;
            buf = it->second;
            p.idle.erase(it);
            p.idle_bytes -= cap;
        }
    }
    if (!buf) {
        cap = bytes;
        if (cudaHostAlloc(&buf, cap, cudaHostAllocDefault) != cudaSuccess || !buf) {
            cudaGetLastError();
            return nullptr;
        }
    }
    {
        std::lock_guard<std::mutex> lk(p.mu);
        p.refill.push_back(cap);
        if (!p.worker) {
            p.worker = true;
            std::thread(pinned_pool_worker).detach();
        }
    }
    p.cv.notify_one();
    return std::shared_ptr<void>(buf, [cap](void* b) {
        Pool& q = pool();
        {
            std::lock_guard<std::mutex> lk(q.mu);
            if (q.idle_bytes + cap <= kPinnedPoolMaxIdle) {
                q.idle.emplace(cap, b);
                q.idle_bytes += cap;
                return;
            }
        }
        cudaFreeHost(b);
    });
}

}  // namespace sparkinfer
