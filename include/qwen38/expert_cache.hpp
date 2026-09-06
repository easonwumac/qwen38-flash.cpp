#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace qwen38 {

// A single-inference-thread cache for expert values. The cache owns one
// shared_ptr reference; callers must retain the returned lease until all GPU
// work that uses the value has completed.
template <class Value>
class ExpertCache final {
public:
    using Handle = std::shared_ptr<const Value>;

    struct Stats {
        std::size_t hits{0};
        std::size_t misses{0};
        std::size_t evictions{0};
        std::size_t load_failures{0};
        std::size_t refusals{0};
        std::size_t resident_bytes{0};
        std::size_t peak_bytes{0};
        std::size_t entries{0};
        std::size_t hit_bytes{0};
        std::size_t miss_bytes{0};
        std::size_t history_entries{0};
        std::size_t history_hits{0};
    };

    explicit ExpertCache(const std::size_t budget_bytes, const std::uint64_t decay_period = 64,
                         const std::size_t history_limit = 0)
        : budget_bytes_(budget_bytes), decay_period_(decay_period), history_limit_(history_limit) {
        if (decay_period == 0) throw std::invalid_argument("expert-cache decay period is zero");
    }

    [[nodiscard]] Handle acquire(
        const std::string& key,
        const std::size_t bytes,
        const std::function<Handle()>& loader) {
        if (loading_) return refuse("reentrant expert-cache acquire");
        if (key.empty()) return refuse("expert-cache key is empty");
        if (bytes == 0) return refuse("expert-cache entry size is zero");

        begin_operation();
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
            if (found->second.bytes != bytes) {
                return refuse("expert-cache key has a different byte size");
            }
            ++stats_.hits;
            add_stat(stats_.hit_bytes, bytes);
            touch(found->second);
            return found->second.value;
        }

        ++stats_.misses;
        add_stat(stats_.miss_bytes, bytes);
        if (!loader) return refuse("expert-cache loader is empty");
        if (!make_room(bytes)) return refuse("expert-cache admission refused");

        pending_bytes_ += bytes;
        loading_ = true;
        try {
            Handle loaded = loader();
            if (!loaded) throw std::runtime_error("expert-cache loader returned null");

            if (resident_bytes_ > std::numeric_limits<std::size_t>::max() - bytes) {
                throw std::runtime_error("expert-cache resident byte counter overflow");
            }
            const auto history = history_.find(key);
            const auto frequency = history == history_.end() ? 1 :
                history->second.frequency + (history->second.frequency != std::numeric_limits<std::size_t>::max());
            Entry entry{loaded, bytes, frequency, next_tick()};
            auto inserted = entries_.emplace(key, std::move(entry));
            if (!inserted.second) {
                throw std::runtime_error("expert-cache insertion found an existing key");
            }
            if (history != history_.end()) {
                history_.erase(history);
                ++stats_.history_hits;
            }
            pending_bytes_ -= bytes;
            loading_ = false;
            // Everything after insertion is non-throwing: commit the reservation
            // once, so loader/insertion failures alone enter rollback above.
            resident_bytes_ += bytes;
            stats_.peak_bytes = std::max(stats_.peak_bytes, resident_bytes_);
            sync_stats();
            return std::move(loaded);
        } catch (...) {
            pending_bytes_ -= bytes;
            loading_ = false;
            ++stats_.load_failures;
            sync_stats();
            throw;
        }
    }

    // Returns false when pinned leases prevent the requested cap from being
    // reached. The requested budget is retained for a later trim/retry.
    bool set_budget(const std::size_t bytes) {
        reject_mutation_during_load();
        budget_bytes_ = bytes;
        return trim();
    }

    // Evicts only entries whose value has no caller lease.
    bool trim() {
        reject_mutation_during_load();
        while (resident_bytes_ > budget_bytes_) {
            if (!evict_one()) {
                sync_stats();
                return false;
            }
        }
        sync_stats();
        return true;
    }

    // Pinned entries remain resident and accounted for.
    void clear() {
        reject_mutation_during_load();
        while (evict_one()) {}
        history_.clear();
        sync_stats();
    }

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::size_t budget() const noexcept { return budget_bytes_; }

private:
    struct Entry {
        Handle value;
        std::size_t bytes{0};
        std::size_t frequency{1};
        std::uint64_t last_touch{0};
    };
    struct History { std::size_t frequency; std::uint64_t last_touch; };

    [[noreturn]] Handle refuse(const char* reason) {
        ++stats_.refusals;
        throw std::runtime_error(reason);
    }

    static void add_stat(std::size_t& destination, const std::size_t value) noexcept {
        if (destination > std::numeric_limits<std::size_t>::max() - value) {
            destination = std::numeric_limits<std::size_t>::max();
        } else {
            destination += value;
        }
    }

    void begin_operation() {
        if (operations_ == std::numeric_limits<std::uint64_t>::max()) operations_ = 0;
        ++operations_;
        if (operations_ % decay_period_ == 0) {
            for (auto& [unused, entry] : entries_) {
                static_cast<void>(unused);
                entry.frequency = std::max<std::size_t>(1, entry.frequency / 2 + entry.frequency % 2);
            }
            for (auto& [unused, entry] : history_) {
                static_cast<void>(unused);
                entry.frequency = std::max<std::size_t>(1, entry.frequency / 2 + entry.frequency % 2);
            }
        }
    }

    void touch(Entry& entry) {
        if (entry.frequency != std::numeric_limits<std::size_t>::max()) ++entry.frequency;
        entry.last_touch = next_tick();
    }

    std::uint64_t next_tick() noexcept {
        if (tick_ == std::numeric_limits<std::uint64_t>::max()) {
            for (auto& [unused, candidate] : entries_) {
                static_cast<void>(unused);
                candidate.last_touch /= 2;
            }
            tick_ /= 2;
            for (auto& [unused, history] : history_) {
                static_cast<void>(unused);
                history.last_touch /= 2;
            }
        }
        return ++tick_;
    }

    bool make_room(const std::size_t bytes) {
        if (bytes > budget_bytes_ || pending_bytes_ > budget_bytes_ - bytes) return false;
        const std::size_t reserved = pending_bytes_ + bytes;
        const std::size_t resident_limit = budget_bytes_ - reserved;
        while (resident_bytes_ > resident_limit) {
            if (!evict_one()) return false;
        }
        return true;
    }

    bool evict_one() {
        auto candidate = entries_.end();
        for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator) {
            if (iterator->second.value.use_count() != 1) continue;
            if (candidate == entries_.end() ||
                iterator->second.frequency < candidate->second.frequency ||
                (iterator->second.frequency == candidate->second.frequency &&
                 iterator->second.last_touch < candidate->second.last_touch)) {
                candidate = iterator;
            }
        }
        if (candidate == entries_.end()) return false;
        if (history_limit_ != 0) {
            if (history_.size() >= history_limit_) {
                auto oldest = history_.begin();
                for (auto it = history_.begin(); it != history_.end(); ++it)
                    if (it->second.last_touch < oldest->second.last_touch) oldest = it;
                history_.erase(oldest);
            }
            // Metadata only: never keep evicted weight ownership alive.
            history_.insert_or_assign(candidate->first,
                History{candidate->second.frequency, candidate->second.last_touch});
        }
        resident_bytes_ -= candidate->second.bytes;
        entries_.erase(candidate);
        ++stats_.evictions;
        sync_stats();
        return true;
    }

    void sync_stats() noexcept {
        stats_.resident_bytes = resident_bytes_;
        stats_.entries = entries_.size();
        stats_.history_entries = history_.size();
    }

    void reject_mutation_during_load() const {
        if (loading_) throw std::runtime_error("reentrant expert-cache mutation");
    }

    std::size_t budget_bytes_{0};
    std::uint64_t decay_period_;
    std::size_t history_limit_;
    std::size_t resident_bytes_{0};
    std::size_t pending_bytes_{0};
    std::uint64_t operations_{0};
    std::uint64_t tick_{0};
    bool loading_{false};
    std::unordered_map<std::string, Entry> entries_;
    std::unordered_map<std::string, History> history_;
    Stats stats_;
};

} // namespace qwen38
