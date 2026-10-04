#include "sparkinfer/stream_reserve.h"

#include <cstdio>
#include <iterator>
#include <map>
#include <mutex>

namespace sparkinfer {
namespace {

constexpr size_t kAlign = 256;

struct Reserve {
    char* base = nullptr;
    size_t size = 0;
    std::map<size_t, size_t> free_;   // offset -> length, coalesced
    std::map<size_t, size_t> used_;   // offset -> length
    size_t in_use = 0, high = 0;
    unsigned long long fallbacks = 0;
};

std::mutex g_mu;
std::map<int, Reserve> g_res;   // by device

bool in_reserve(const Reserve& r, const void* p) {
    const char* c = static_cast<const char*>(p);
    return r.base && c >= r.base && c < r.base + r.size;
}

}  // namespace

bool stream_reserve_init(int device, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_mu);
    Reserve& r = g_res[device];
    if (r.base) return r.size >= bytes;
    if (bytes == 0) return true;
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(device);
    void* p = nullptr;
    const bool ok = cudaMalloc(&p, bytes) == cudaSuccess;
    cudaSetDevice(cur);
    if (!ok) { cudaGetLastError(); return false; }
    r.base = static_cast<char*>(p);
    r.size = bytes;
    r.free_[0] = bytes;
    return true;
}

void stream_reserve_release_all() {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [dev, r] : g_res) {
        if (!r.base || !r.used_.empty()) continue;   // still lent out: leave it
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(dev);
        cudaFree(r.base);
        cudaSetDevice(cur);
        r = Reserve{};
    }
}

cudaError_t sr_malloc(void** p, size_t bytes) {
    *p = nullptr;
    if (bytes == 0) bytes = 1;
    const size_t need = (bytes + kAlign - 1) / kAlign * kAlign;
    int dev = 0;
    cudaGetDevice(&dev);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_res.find(dev);
        if (it != g_res.end() && it->second.base) {
            Reserve& r = it->second;
            for (auto f = r.free_.begin(); f != r.free_.end(); ++f) {
                if (f->second < need) continue;
                const size_t off = f->first, len = f->second;
                r.free_.erase(f);
                if (len > need) r.free_[off + need] = len - need;
                r.used_[off] = need;
                r.in_use += need;
                if (r.in_use > r.high) r.high = r.in_use;
                *p = r.base + off;
                return cudaSuccess;
            }
            r.fallbacks++;
        }
    }
    return cudaMalloc(p, bytes);
}

void sr_free(void* p) {
    if (!p) return;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [dev, r] : g_res) {
            if (!in_reserve(r, p)) continue;
            const size_t off = static_cast<char*>(p) - r.base;
            auto u = r.used_.find(off);
            if (u == r.used_.end()) {
                fprintf(stderr, "[stream-reserve] free of an address it did not hand out\n");
                return;
            }
            size_t o = off, len = u->second;
            r.in_use -= len;
            r.used_.erase(u);
            auto next = r.free_.lower_bound(o);
            if (next != r.free_.end() && next->first == o + len) { len += next->second; next = r.free_.erase(next); }
            if (next != r.free_.begin()) {
                auto prev = std::prev(next);
                if (prev->first + prev->second == o) { o = prev->first; len += prev->second; r.free_.erase(prev); }
            }
            r.free_[o] = len;
            return;
        }
    }
    cudaFree(p);
}

StreamReserveStats stream_reserve_stats(int device) {
    std::lock_guard<std::mutex> lk(g_mu);
    StreamReserveStats st;
    auto it = g_res.find(device);
    if (it == g_res.end()) return st;
    st.bytes = it->second.size;
    st.used = it->second.in_use;
    st.high_water = it->second.high;
    st.fallbacks = it->second.fallbacks;
    return st;
}

}  // namespace sparkinfer
