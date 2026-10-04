#pragma once

// Automatic prefix cache for ContinuousBatchEngine.
//
// Chat and agent clients resend the whole conversation on every turn, so each request's prompt is
// the previous request's prompt plus what happened since. Without a cache the server recomputes
// all of it. An entry here is a prefix that was already computed: its KV blocks, shared with any
// later sequence that starts from it (KVCacheManager's refcounted blocks, no copy), and a snapshot
// of the Gated-DeltaNet recurrent state at the end of it.
//
// The snapshot is why an entry is taken at ONE chosen position rather than at every block. KV can
// be cut at any block boundary after the fact, but a recurrent layer's state is a single running
// value -- it exists only at the position the sequence is at, and copying it costs ~205 MB of
// pinned host memory on Qwen3.8-27B. So the engine snapshots each request at a checkpoint the
// caller picks (the server picks the start of the final assistant turn, which is where the next
// turn's prompt stops matching this one), and the cache keeps the most recently used of those.
//
// Every method that touches blocks must run under the model's device mutex, the same lock every
// other KVCacheManager mutation already takes. The cache's own mutex only guards its bookkeeping,
// so stats() can be read from an HTTP thread.
//
// Host KV tier (Limits::max_kv_host_bytes > 0). An entry's KV blocks are also copied to pinned
// host memory when it is inserted -- asynchronously, on a copy stream per card, and only the
// blocks no other entry already copied (host blocks are keyed by a hash of every token up to the
// block's end, so the turns of one conversation share them). Taking an entry's blocks back for
// the pool then costs nothing: the entry stays, host-resident. A lookup that lands on a
// host-resident entry allocates blocks, copies the KV back on both cards (blocks a GPU-resident
// entry still holds for the same tokens are shared instead of copied) and serves the hit as
// usual. The copies are byte-exact, so a restored prefix is the prefix that was cached.

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

class PrefixCache {
public:
    struct Limits {
        size_t max_entries = 32;
        size_t max_host_bytes = size_t(8) << 30;   // pinned host memory for recurrent snapshots
        int max_blocks = 0;                        // distinct KV blocks entries may hold; 0 = no cap
        size_t max_kv_host_bytes = 0;              // pinned host memory for KV blocks; 0 = no host tier
    };

    struct Hit {
        int tokens = 0;                            // cached prefix length; 0 on a miss
        std::vector<int> blocks;                   // tokens / block_size physical blocks to share
        Qwen35Model::RecurrentStateSnapshot state;
    };

    struct Stats {
        uint64_t lookups = 0;
        uint64_t hits = 0;
        uint64_t tokens_reused = 0;
        uint64_t inserts = 0;
        uint64_t evictions = 0;
        size_t entries = 0;
        size_t host_bytes = 0;
        int blocks = 0;          // distinct KV blocks held: entries on one conversation share most
        size_t host_entries = 0;         // entries whose KV is in host memory only
        size_t kv_host_bytes = 0;        // pinned host memory holding KV blocks (both cards)
        uint64_t restores = 0;           // hits served by copying KV back from host memory
        uint64_t restored_blocks = 0;    // blocks copied back (blocks shared from the pool not counted)
    };

    PrefixCache(KVCacheManager* kv, const Limits& limits);
    ~PrefixCache();   // releases every block the entries hold
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    // The longest entry that is a PROPER prefix of `prompt` -- strictly shorter, so at least one
    // prompt token is still prefilled and produces the first generated token. Refreshes its LRU
    // position. The returned blocks are not retained by the hit; share them before anything can
    // evict (both happen under the device mutex).
    Hit lookup(const std::vector<int>& prompt);

    // Adopt `blocks` (already retained for this entry) and `state` as the entry for `tokens`, whose
    // length must be a whole number of blocks. A prefix already present, or a malformed entry, is
    // dropped and its blocks released. Evicts least-recently-used entries past the limits.
    void insert(std::vector<int> tokens, std::vector<int> blocks,
                Qwen35Model::RecurrentStateSnapshot state);

    // Evict least-recently-used entries until the pool has at least need_blocks free or the cache
    // is empty. True if anything was evicted. Blocks still shared with a live sequence do not
    // return to the pool until that sequence finishes, so this can empty the cache and still fall
    // short.
    bool evict_for(int need_blocks);

    Stats stats() const;

    // (dual-GPU) Tensor parallelism: rank 1's KV manager, which holds the same block numbering as
    // kv (every engine-side block op is mirrored onto it), so each block release is applied to it
    // too. Null (the default) at tp=1.
    void set_mirror(KVCacheManager* peer);

    struct HostTier;   // pinned host copies of KV blocks (prefix_cache.cpp)

private:
    struct Key { uint64_t a = 0, b = 0; bool operator==(const Key& o) const { return a == o.a && b == o.b; } };
    struct KeyHash { size_t operator()(const Key& k) const { return (size_t)(k.a ^ (k.b * 0x9E3779B97F4A7C15ull)); } };
    struct Entry {
        std::vector<int> tokens;
        std::vector<int> blocks;     // pool blocks; empty while the entry is host-resident
        std::vector<Key> chain;      // per block: hash of every token up to the block's end
        std::vector<int> hblocks;    // host blocks (all of them, or none)
        Qwen35Model::RecurrentStateSnapshot state;
        uint64_t last_used = 0;
    };

    void evict_one_locked();
    void enforce_limits_locked();
    void drop_locked(Entry* e);                  // remove the entry entirely
    bool gpu_release_locked(Entry* e);           // give its pool blocks back if a host copy exists
    void unhold_locked(const std::vector<int>& blocks);
    bool host_copy_locked(Entry& e);             // take host blocks for e and queue their copies
    void host_release_locked(Entry& e);
    bool restore_locked(Entry* e);               // bring a host-resident entry's KV back into the pool
    // Least recently used entry: which 0 any, 1 in the pool, 2 host-only, 3 in the pool with a
    // host copy. Null if none.
    Entry* lru_locked(int which, const Entry* skip1, const Entry* skip2 = nullptr);

    KVCacheManager* kv_;
    KVCacheManager* kv_peer_ = nullptr;
    void release(const std::vector<int>& blocks);   // kv_ and, under tp, kv_peer_
    Limits limits_;
    mutable std::mutex mu_;
    std::list<Entry> entries_;   // a list: entries are referred to by address across erasures
    uint64_t clock_ = 0;
    Stats stats_;
    size_t host_bytes_ = 0;
    // Entries holding each block. Entries along one conversation share most of their blocks, so a
    // cap on the sum would evict for memory that is not actually used twice.
    std::unordered_map<int, int> held_;
    std::unique_ptr<HostTier> host_;
    std::unordered_map<Key, std::pair<int, int>, KeyHash> host_keys_;   // key -> (host block, refs)
};

}  // namespace sparkinfer
