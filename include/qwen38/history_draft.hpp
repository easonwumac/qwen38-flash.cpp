#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace qwen38 {

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
