#include "qwen38/history_draft.hpp"
#include "test.hpp"

#include <array>

void run_history_draft_tests() {
    qwen38::HistoryDraftCache cache(2, 4);
    const std::array<std::uint32_t, 9> history{1, 2, 3, 4, 9, 1, 2, 3, 4};
    cache.append(history);
    QWEN38_CHECK(cache.propose(4) == std::vector<std::uint32_t>({9, 1, 2, 3}));

    cache.append(9);
    QWEN38_CHECK(cache.propose(3) == std::vector<std::uint32_t>({1, 2, 3}));

    qwen38::HistoryDraftCache miss;
    const std::array<std::uint32_t, 5> unique{10, 11, 12, 13, 14};
    miss.append(unique);
    QWEN38_CHECK(miss.propose(4).empty());
}
