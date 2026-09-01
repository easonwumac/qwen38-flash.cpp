#include "qwen38/history_draft.hpp"
#include "test.hpp"

#include <array>

void run_history_draft_tests() {
    qwen38::HistoryDraftPolicy adaptive;
    QWEN38_CHECK(!adaptive.should_try());
    for (int round = 0; round < 7; ++round) {
        adaptive.observe_learned(2, round == 0 ? 1 : 0);
        QWEN38_CHECK(!adaptive.should_try());
    }
    adaptive.observe_learned(2, 1);
    QWEN38_CHECK(adaptive.should_try());
    QWEN38_CHECK(adaptive.activations() == 1);
    for (int round = 0; round < 2; ++round) adaptive.observe_history(3, 1);
    QWEN38_CHECK(!adaptive.should_try());
    QWEN38_CHECK(adaptive.exhausted());
    QWEN38_CHECK(adaptive.deactivations() == 1);
    for (int round = 0; round < 8; ++round) adaptive.observe_learned(2, 0);
    QWEN38_CHECK(!adaptive.should_try());
    QWEN38_CHECK(adaptive.activations() == 1);

    qwen38::HistoryDraftPolicy strong_learned;
    for (int round = 0; round < 8; ++round) {
        strong_learned.observe_learned(2, round % 2 == 0 ? 2 : 1);
    }
    QWEN38_CHECK(!strong_learned.should_try());
    QWEN38_CHECK(strong_learned.activations() == 0);

    qwen38::HistoryDraftPolicy forced(qwen38::HistoryDraftMode::forced);
    QWEN38_CHECK(forced.enabled() && forced.should_try() && !forced.exhausted());
    for (int round = 0; round < 2; ++round) forced.observe_history(3, 0);
    QWEN38_CHECK(forced.should_try() && !forced.exhausted());

    qwen38::HistoryDraftPolicy disabled(qwen38::HistoryDraftMode::disabled);
    QWEN38_CHECK(!disabled.enabled() && !disabled.should_try() && !disabled.exhausted());

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
