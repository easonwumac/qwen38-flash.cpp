#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace qwen38 {

enum class HistoryDraftMode {
    disabled,
    adaptive,
    forced,
};

// Learned MTP is the stronger default drafter. History proposals are admitted
// only after a short run demonstrates that the learned head is locally weak,
// then removed again if their own measured acceptance is poor.
class HistoryDraftPolicy final {
public:
    explicit HistoryDraftPolicy(HistoryDraftMode mode = HistoryDraftMode::adaptive)
        : mode_(mode), active_(mode == HistoryDraftMode::forced) {}

    [[nodiscard]] bool enabled() const noexcept {
        return mode_ != HistoryDraftMode::disabled;
    }
    [[nodiscard]] bool should_try() const noexcept { return active_; }
    [[nodiscard]] bool exhausted() const noexcept { return rejected_; }
    void observe_learned(std::size_t proposed, std::size_t accepted);
    void observe_history(std::size_t proposed, std::size_t accepted);

    [[nodiscard]] std::size_t activations() const noexcept { return activations_; }
    [[nodiscard]] std::size_t deactivations() const noexcept { return deactivations_; }

private:
    HistoryDraftMode mode_;
    bool active_{false};
    bool rejected_{false};
    std::size_t learned_rounds_{0};
    std::size_t learned_proposed_{0};
    std::size_t learned_accepted_{0};
    std::size_t history_rounds_{0};
    std::size_t history_proposed_{0};
    std::size_t history_accepted_{0};
    std::size_t activations_{0};
    std::size_t deactivations_{0};
};

// A request-local suffix cache for exact speculative proposals. Entries are
// published only after their continuation is known, so a lookup never points
// at the suffix currently being queried.
class HistoryDraftCache final {
public:
    explicit HistoryDraftCache(std::size_t minimum_order = 2, std::size_t maximum_order = 5);

    void append(std::uint32_t token);
    void append(std::span<const std::uint32_t> tokens);

    [[nodiscard]] std::vector<std::uint32_t> propose(std::size_t maximum_tokens) const;
    [[nodiscard]] std::size_t size() const noexcept { return history_.size(); }

private:
    [[nodiscard]] std::uint64_t key(std::size_t start, std::size_t order) const noexcept;
    [[nodiscard]] bool suffix_matches(std::size_t start, std::size_t order) const noexcept;

    std::size_t minimum_order_;
    std::size_t maximum_order_;
    std::vector<std::uint32_t> history_;
    std::unordered_map<std::uint64_t, std::size_t> latest_;
};

} // namespace qwen38
