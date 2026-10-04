#include "sparkinfer/prefix_cache.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#include <cuda_runtime.h>

namespace sparkinfer {

// Pinned host copies of KV blocks, for both cards. Host blocks live in chunks of kChunk blocks
// that mirror the pool's layout -- per plane (K data, V data, K scales, V scales), [slot][block]
// -- so a run of consecutive pool blocks going to consecutive host blocks is ONE 2-D copy per
// plane, whatever the number of layers. Copies run on a stream per card, so a copy back is
// ordered after any copy out it depends on.
struct PrefixCache::HostTier {
    static constexpr int kChunk = 256;   // blocks per chunk (4096 tokens)
    struct Rank {
        KVCacheManager* kv = nullptr;
        int dev = 0;
        cudaStream_t st = nullptr;
        int slots = 0;
        size_t pb[4] = {0, 0, 0, 0};       // bytes of one block in one slot, per plane
        size_t spitch[4] = {0, 0, 0, 0};   // bytes between slots in the pool, per plane
        char* pool[4] = {nullptr, nullptr, nullptr, nullptr};
        size_t chunk_bytes = 0;
        std::vector<char*> chunks;
        char* plane(int chunk, int p) const {
            size_t off = 0;
            for (int q = 0; q < p; q++) off += (size_t)slots * kChunk * pb[q];
            return chunks[(size_t)chunk] + off;
        }
    };
    std::vector<Rank> ranks;
    size_t budget = 0;
    size_t chunk_bytes_all = 0;
    std::set<int> free_ids;   // lowest first, so a spill tends to get consecutive host blocks
    bool ok = false;
    // Pinning is slow (~13 ms per 38 MB chunk per card), and the cache runs under the device
    // mutex, where it would stall every running stream. A thread pins ahead: up to kSpare chunks
    // wait, ready to be adopted.
    static constexpr int kSpare = 8;
    std::mutex spare_mu;
    std::condition_variable spare_cv;
    std::vector<std::vector<char*>> spare;   // per spare chunk: one buffer per rank
    size_t pinned = 0;                       // adopted + spare, bytes over all ranks
    bool stop = false;
    std::thread pinner;

    HostTier(KVCacheManager* kv, KVCacheManager* peer, size_t budget_bytes) : budget(budget_bytes) {
        int cur = 0;
        cudaGetDevice(&cur);
        for (KVCacheManager* m : {kv, peer}) {
            if (!m) continue;
            Rank r;
            r.kv = m;
            cudaPointerAttributes at{};
            if (cudaPointerGetAttributes(&at, m->k_pool()) != cudaSuccess) { cudaGetLastError(); return; }
            r.dev = at.device;
            r.slots = m->kv_slots();
            const size_t epb = m->block_elems();
            r.pb[0] = r.pb[1] = m->kv_bytes(epb);
            r.spitch[0] = r.spitch[1] = m->kv_bytes(m->layer_stride_elems());
            r.pool[0] = static_cast<char*>(m->k_pool());
            r.pool[1] = static_cast<char*>(m->v_pool());
            if (m->k_scale_pool() && m->v_scale_pool()) {
                const size_t hd = m->layer_stride_elems() / m->scale_layer_stride_elems();
                r.pb[2] = r.pb[3] = epb / hd * sizeof(unsigned short);
                r.spitch[2] = r.spitch[3] = m->scale_layer_stride_elems() * sizeof(unsigned short);
                r.pool[2] = static_cast<char*>(m->k_scale_pool());
                r.pool[3] = static_cast<char*>(m->v_scale_pool());
            }
            for (int p = 0; p < 4; p++) r.chunk_bytes += (size_t)r.slots * kChunk * r.pb[p];
            cudaSetDevice(r.dev);
            if (cudaStreamCreateWithFlags(&r.st, cudaStreamNonBlocking) != cudaSuccess) {
                cudaGetLastError();
                cudaSetDevice(cur);
                return;
            }
            chunk_bytes_all += r.chunk_bytes;
            ranks.push_back(r);
        }
        cudaSetDevice(cur);
        ok = !ranks.empty() && chunk_bytes_all > 0 && budget >= chunk_bytes_all;
        if (ok) pinner = std::thread([this] { pin_ahead(); });
    }
    ~HostTier() {
        {
            std::lock_guard<std::mutex> l(spare_mu);
            stop = true;
        }
        spare_cv.notify_all();
        if (pinner.joinable()) pinner.join();
        sync();
        for (auto& c : spare) for (char* b : c) cudaFreeHost(b);
        for (Rank& r : ranks) {
            for (char* c : r.chunks) cudaFreeHost(c);
            if (r.st) cudaStreamDestroy(r.st);
        }
    }
    std::vector<char*> pin_chunk() {
        std::vector<char*> got;
        for (Rank& r : ranks) {
            void* p = nullptr;
            if (cudaHostAlloc(&p, r.chunk_bytes, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                for (char* g : got) cudaFreeHost(g);
                return {};
            }
            got.push_back(static_cast<char*>(p));
        }
        return got;
    }
    void pin_ahead() {
        std::unique_lock<std::mutex> l(spare_mu);
        while (!stop) {
            if ((int)spare.size() < kSpare && pinned + chunk_bytes_all <= budget) {
                pinned += chunk_bytes_all;   // claimed before pinning, so grow() cannot overshoot
                l.unlock();
                std::vector<char*> c = pin_chunk();
                l.lock();
                if (c.empty()) { pinned -= chunk_bytes_all; break; }   // out of pinnable memory
                spare.push_back(std::move(c));
                continue;
            }
            spare_cv.wait(l);
        }
    }
    size_t bytes() const { return ranks.empty() ? 0 : ranks[0].chunks.size() * chunk_bytes_all; }
    bool grow() {
        if (!ok) return false;
        std::vector<char*> got;
        {
            std::lock_guard<std::mutex> l(spare_mu);
            if (!spare.empty()) {
                got = std::move(spare.back());
                spare.pop_back();
            } else if (pinned + chunk_bytes_all <= budget) {
                pinned += chunk_bytes_all;
            } else {
                return false;
            }
        }
        spare_cv.notify_all();   // pin the next one
        if (got.empty()) {
            got = pin_chunk();   // nothing ready yet: pin here
            if (got.empty()) {
                std::lock_guard<std::mutex> l(spare_mu);
                pinned -= chunk_bytes_all;
                return false;
            }
        }
        const int c = (int)ranks[0].chunks.size();
        for (size_t i = 0; i < ranks.size(); i++) ranks[i].chunks.push_back(got[i]);
        for (int j = 0; j < kChunk; j++) free_ids.insert(c * kChunk + j);
        return true;
    }
    int alloc_id() {
        if (free_ids.empty() && !grow()) return -1;
        const int id = *free_ids.begin();
        free_ids.erase(free_ids.begin());
        return id;
    }
    void free_id(int id) { free_ids.insert(id); }

    // to_host: pool blocks gpu[i] -> host blocks host[i]; else the reverse. Queued, not waited for.
    bool copy(bool to_host, const std::vector<int>& gpu, const std::vector<int>& host) {
        int cur = 0;
        cudaGetDevice(&cur);
        bool good = true;
        for (Rank& r : ranks) {
            cudaSetDevice(r.dev);
            for (size_t i = 0; i < gpu.size();) {
                // A run: consecutive pool blocks to consecutive host blocks of one chunk.
                size_t n = 1;
                while (i + n < gpu.size() && gpu[i + n] == gpu[i] + (int)n &&
                       host[i + n] == host[i] + (int)n && host[i + n] / kChunk == host[i] / kChunk)
                    n++;
                const int c = host[i] / kChunk, j = host[i] % kChunk;
                for (int p = 0; p < 4 && good; p++) {
                    if (!r.pb[p]) continue;
                    char* h = r.plane(c, p) + (size_t)j * r.pb[p];
                    char* d = r.pool[p] + (size_t)gpu[i] * r.pb[p];
                    const size_t hp = (size_t)kChunk * r.pb[p], w = n * r.pb[p];
                    good = (to_host
                        ? cudaMemcpy2DAsync(h, hp, d, r.spitch[p], w, r.slots, cudaMemcpyDeviceToHost, r.st)
                        : cudaMemcpy2DAsync(d, r.spitch[p], h, hp, w, r.slots, cudaMemcpyHostToDevice, r.st))
                        == cudaSuccess;
                }
                i += n;
            }
        }
        cudaSetDevice(cur);
        if (!good) cudaGetLastError();
        return good;
    }
    bool sync() {
        bool good = true;
        for (Rank& r : ranks)
            if (r.st && cudaStreamSynchronize(r.st) != cudaSuccess) { cudaGetLastError(); good = false; }
        return good;
    }
};

namespace {
// Chained over blocks: key i covers every token of blocks 0..i, so equal keys mean equal KV.
// Two independent 64-bit lanes; a collision would need both.
void chain_keys(const std::vector<int>& tokens, size_t bs, std::vector<uint64_t>& a, std::vector<uint64_t>& b) {
    uint64_t ha = 0x243F6A8885A308D3ull, hb = 0x13198A2E03707344ull;
    for (size_t i = 0; i + bs <= tokens.size(); i += bs) {
        for (size_t t = i; t < i + bs; t++) {
            const uint64_t v = (uint64_t)(uint32_t)tokens[t];
            ha = (ha ^ v) * 0x100000001B3ull;
            ha ^= ha >> 29;
            hb = (hb + v + 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull;
            hb ^= hb >> 31;
        }
        a.push_back(ha);
        b.push_back(hb);
    }
}
}  // namespace

PrefixCache::PrefixCache(KVCacheManager* kv, const Limits& limits) : kv_(kv), limits_(limits) {}

void PrefixCache::set_mirror(KVCacheManager* peer) {
    std::lock_guard<std::mutex> lock(mu_);
    kv_peer_ = peer;
    // Start pinning now rather than at the first insert. A tier made for one card only is
    // remade: nothing has been copied yet when the engine sets the mirror.
    if (limits_.max_kv_host_bytes > 0 && kv_->prefix_sharing_supported() && entries_.empty()) {
        host_.reset();
        host_ = std::make_unique<HostTier>(kv_, kv_peer_, limits_.max_kv_host_bytes);
        if (!host_->ok) host_.reset();
    }
}

void PrefixCache::release(const std::vector<int>& blocks) {
    kv_->release_blocks(blocks);
    if (kv_peer_) kv_peer_->release_blocks(blocks);
}

PrefixCache::~PrefixCache() {
    std::lock_guard<std::mutex> lock(mu_);
    if (host_) host_->sync();
    for (const Entry& e : entries_) release(e.blocks);
    entries_.clear();
    host_.reset();
}

void PrefixCache::unhold_locked(const std::vector<int>& blocks) {
    for (int b : blocks) {
        auto it = held_.find(b);
        if (it != held_.end() && --it->second == 0) held_.erase(it);
    }
}

void PrefixCache::host_release_locked(Entry& e) {
    for (size_t i = 0; i < e.hblocks.size() && i < e.chain.size(); i++) {
        auto it = host_keys_.find(e.chain[i]);
        if (it == host_keys_.end()) continue;
        if (--it->second.second == 0) {
            host_->free_id(it->second.first);
            host_keys_.erase(it);
        }
    }
    e.hblocks.clear();
}

void PrefixCache::drop_locked(Entry* e) {
    if (!e->blocks.empty()) {
        if (host_) host_->sync();   // a copy out may still be reading these blocks
        release(e->blocks);
        unhold_locked(e->blocks);
    }
    if (host_) host_release_locked(*e);
    host_bytes_ -= e->state.bytes();
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
        if (&*it == e) { entries_.erase(it); break; }
    stats_.evictions++;
}

bool PrefixCache::gpu_release_locked(Entry* e) {
    if (e->blocks.empty() || e->hblocks.size() != e->blocks.size() || !host_) return false;
    if (!host_->sync()) return false;   // the host copy must be complete before the blocks go
    release(e->blocks);
    unhold_locked(e->blocks);
    e->blocks.clear();
    return true;
}

PrefixCache::Entry* PrefixCache::lru_locked(int which, const Entry* skip1, const Entry* skip2) {
    Entry* best = nullptr;
    for (Entry& e : entries_) {
        if (&e == skip1 || &e == skip2) continue;
        if ((which == 1 || which == 3) && e.blocks.empty()) continue;
        if (which == 2 && !e.blocks.empty()) continue;
        if (which == 3 && e.hblocks.size() != e.blocks.size()) continue;
        if (!best || e.last_used < best->last_used) best = &e;
    }
    return best;
}

bool PrefixCache::host_copy_locked(Entry& e) {
    if (limits_.max_kv_host_bytes == 0 || e.blocks.empty() || !kv_->prefix_sharing_supported()) return false;
    if (!host_) {
        host_ = std::make_unique<HostTier>(kv_, kv_peer_, limits_.max_kv_host_bytes);
        if (!host_->ok) {
            fprintf(stderr, "[prefix-cache] host KV tier unavailable (copy stream or pool layout)\n");
            limits_.max_kv_host_bytes = 0;
            host_.reset();
            return false;
        }
    }
    std::vector<int> gpu, host;
    e.hblocks.clear();
    for (size_t i = 0; i < e.blocks.size(); i++) {
        auto it = host_keys_.find(e.chain[i]);
        if (it != host_keys_.end()) {
            it->second.second++;
            e.hblocks.push_back(it->second.first);
            continue;
        }
        int id = host_->alloc_id();
        while (id < 0) {
            // Host memory full: drop the least recently used host-only entry, else take the host
            // copy of the least recently used entry still in the pool (it is copied again if it
            // is ever inserted again).
            if (Entry* o = lru_locked(2, &e)) {
                drop_locked(o);
            } else if (Entry* g = lru_locked(3, &e)) {
                if (g->hblocks.empty()) break;
                host_release_locked(*g);
            } else {
                break;
            }
            id = host_->alloc_id();
        }
        if (id < 0) {
            host_release_locked(e);   // what it took so far
            return false;
        }
        host_keys_[e.chain[i]] = {id, 1};
        e.hblocks.push_back(id);
        gpu.push_back(e.blocks[i]);
        host.push_back(id);
    }
    if (!gpu.empty() && !host_->copy(true, gpu, host)) {
        host_->sync();
        host_release_locked(e);
        return false;
    }
    return true;
}

bool PrefixCache::restore_locked(Entry* e) {
    const size_t n = e->hblocks.size();
    if (!host_ || n == 0 || n != e->chain.size() || !e->blocks.empty()) return false;
    // The pool may still hold the leading blocks under another entry with the same tokens.
    const Entry* donor = nullptr;
    size_t k = 0;
    for (const Entry& o : entries_) {
        if (&o == e || o.blocks.empty()) continue;
        size_t m = 0;
        const size_t lim = std::min(n, o.blocks.size());
        while (m < lim && o.chain[m] == e->chain[m]) m++;
        if (m > k) { k = m; donor = &o; }
    }
    const int need = (int)(n - k);
    // Make room: entries with a host copy leave the pool least recently used first (never the
    // donor). Nothing is dropped here, so the caller's candidates stay valid.
    while (kv_->num_free_blocks() < need) {
        Entry* o = lru_locked(3, e, donor);
        if (!o || !gpu_release_locked(o)) break;
    }
    if (kv_->num_free_blocks() < need) return false;
    std::vector<int> shared;
    if (k > 0) shared.assign(donor->blocks.begin(), donor->blocks.begin() + (std::ptrdiff_t)k);
    std::vector<int> fresh = kv_->allocate_blocks(need);
    if ((int)fresh.size() != need) { kv_->release_blocks(fresh); return false; }
    if (kv_peer_) {
        std::vector<int> pf = kv_peer_->allocate_blocks(need);
        if (pf != fresh) {
            fprintf(stderr, "[prefix-cache] rank-1 block ids differ on restore -- not restoring\n");
            kv_->release_blocks(fresh);
            kv_peer_->release_blocks(pf);
            return false;
        }
    }
    if (!shared.empty()) {
        const bool r0 = kv_->retain_blocks(shared);
        const bool r1 = !kv_peer_ || kv_peer_->retain_blocks(shared);
        if (!r0 || !r1) {
            if (r0) kv_->release_blocks(shared);
            if (r1 && kv_peer_) kv_peer_->release_blocks(shared);
            release(fresh);
            return false;
        }
    }
    std::vector<int> host(e->hblocks.begin() + (std::ptrdiff_t)k, e->hblocks.end());
    if (!fresh.empty() && (!host_->copy(false, fresh, host) || !host_->sync())) {
        release(fresh);
        release(shared);
        return false;
    }
    e->blocks = shared;
    e->blocks.insert(e->blocks.end(), fresh.begin(), fresh.end());
    for (int b : e->blocks) held_[b]++;
    stats_.restores++;
    stats_.restored_blocks += fresh.size();
    return true;
}

PrefixCache::Hit PrefixCache::lookup(const std::vector<int>& prompt) {
    std::lock_guard<std::mutex> lock(mu_);
    Hit hit;
    stats_.lookups++;
    // Longest first; a host-resident entry that cannot come back falls through to the next.
    std::vector<Entry*> cand;
    for (Entry& e : entries_) {
        if (e.tokens.size() >= prompt.size()) continue;
        if (!std::equal(e.tokens.begin(), e.tokens.end(), prompt.begin())) continue;
        cand.push_back(&e);
    }
    std::sort(cand.begin(), cand.end(),
              [](const Entry* a, const Entry* b) { return a->tokens.size() > b->tokens.size(); });
    Entry* best = nullptr;
    for (Entry* e : cand)
        if (!e->blocks.empty() || restore_locked(e)) { best = e; break; }
    if (!best) return hit;
    best->last_used = ++clock_;
    stats_.hits++;
    stats_.tokens_reused += best->tokens.size();
    hit.tokens = (int)best->tokens.size();
    hit.blocks = best->blocks;
    hit.state = best->state;   // shares the pinned buffer; nothing ever writes to a stored snapshot
    return hit;
}

void PrefixCache::insert(std::vector<int> tokens, std::vector<int> blocks,
                         Qwen35Model::RecurrentStateSnapshot state) {
    std::lock_guard<std::mutex> lock(mu_);
    const size_t bs = (size_t)kv_->block_size();
    const bool well_formed = !tokens.empty() && bs > 0 && tokens.size() % bs == 0 &&
                             blocks.size() * bs == tokens.size();
    bool duplicate = false;
    for (Entry& e : entries_) {
        if (e.tokens.size() == tokens.size() && e.tokens == tokens) {
            e.last_used = ++clock_;
            duplicate = true;
            break;
        }
    }
    if (!well_formed || duplicate || state.bytes() > limits_.max_host_bytes) {
        release(blocks);
        return;
    }
    Entry e;
    e.tokens = std::move(tokens);
    e.blocks = std::move(blocks);
    {
        std::vector<uint64_t> a, b;
        chain_keys(e.tokens, bs, a, b);
        e.chain.resize(a.size());
        for (size_t i = 0; i < a.size(); i++) e.chain[i] = Key{a[i], b[i]};
    }
    e.state = std::move(state);
    e.last_used = ++clock_;
    host_bytes_ += e.state.bytes();
    for (int b : e.blocks) held_[b]++;
    entries_.push_back(std::move(e));
    stats_.inserts++;
    if (limits_.max_kv_host_bytes > 0) host_copy_locked(entries_.back());
    enforce_limits_locked();
}

bool PrefixCache::evict_for(int need_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    bool evicted = false;
    while (kv_->num_free_blocks() < need_blocks) {
        Entry* e = lru_locked(1, nullptr);
        if (!e) break;
        if (!gpu_release_locked(e)) drop_locked(e);
        evicted = true;
    }
    return evicted;
}

PrefixCache::Stats PrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats s = stats_;
    s.entries = entries_.size();
    s.host_bytes = host_bytes_;
    s.blocks = (int)held_.size();
    for (const Entry& e : entries_) s.host_entries += e.blocks.empty() ? 1 : 0;
    s.kv_host_bytes = host_ ? host_->bytes() : 0;
    return s;
}

void PrefixCache::evict_one_locked() {
    if (Entry* e = lru_locked(0, nullptr)) drop_locked(e);
}

void PrefixCache::enforce_limits_locked() {
    while (!entries_.empty() &&
           (entries_.size() > limits_.max_entries || host_bytes_ > limits_.max_host_bytes))
        evict_one_locked();
    // The pool share: entries leave the pool least recently used first, keeping their host copy.
    while (limits_.max_blocks > 0 && (int)held_.size() > limits_.max_blocks) {
        Entry* e = lru_locked(1, nullptr);
        if (!e) break;
        if (!gpu_release_locked(e)) drop_locked(e);
    }
}

}  // namespace sparkinfer
