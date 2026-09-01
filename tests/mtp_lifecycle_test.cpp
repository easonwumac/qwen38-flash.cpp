#include "qwen38/mtp_lifecycle.hpp"
#include "test.hpp"

#include <stdexcept>
#include <vector>

void run_mtp_lifecycle_tests() {
    using qwen38::MtpHiddenSource;
    using qwen38::MtpPromptLifecycle;
    using qwen38::MtpPromptPair;

    MtpPromptLifecycle lifecycle;
    const auto cold = lifecycle.plan_chunk(0, 3);
    QWEN38_CHECK(cold == std::vector<MtpPromptPair>({
        {MtpHiddenSource::chunk, 0, 1, 1},
        {MtpHiddenSource::chunk, 1, 2, 2},
    }));
    QWEN38_CHECK(lifecycle.has_carry());
    QWEN38_CHECK(lifecycle.next_position() == 3);

    const auto continuation = lifecycle.plan_chunk(3, 2);
    QWEN38_CHECK(continuation == std::vector<MtpPromptPair>({
        {MtpHiddenSource::carry, 0, 0, 3},
        {MtpHiddenSource::chunk, 0, 1, 4},
    }));

    bool rejected_gap = false;
    try {
        static_cast<void>(lifecycle.plan_chunk(7, 1));
    } catch (const std::runtime_error&) {
        rejected_gap = true;
    }
    QWEN38_CHECK(rejected_gap);

    const auto new_request = lifecycle.plan_chunk(0, 1);
    QWEN38_CHECK(new_request.empty());
    QWEN38_CHECK(lifecycle.next_position() == 1);

    const auto partial = qwen38::plan_mtp_commit(12, 3, 1);
    QWEN38_CHECK(partial.truncate_head_to == 12);
    QWEN38_CHECK(partial.target_rows_to_keep == 2);
    QWEN38_CHECK(partial.stash_rows == 2);
    QWEN38_CHECK(partial.committed_boundary == 14);

    bool rejected_accept = false;
    try {
        static_cast<void>(qwen38::plan_mtp_commit(0, 1, 2));
    } catch (const std::runtime_error&) {
        rejected_accept = true;
    }
    QWEN38_CHECK(rejected_accept);

    const std::vector<std::uint32_t> drafts{10, 20, 30};
    const auto zero = qwen38::decide_mtp_greedy(
        drafts, std::vector<std::uint32_t>{11, 20, 30, 40});
    QWEN38_CHECK(zero.accepted == 0 && zero.correction_row == 0 && zero.next_token == 11);
    const auto partial_decision = qwen38::decide_mtp_greedy(
        drafts, std::vector<std::uint32_t>{10, 20, 31, 40});
    QWEN38_CHECK(
        partial_decision.accepted == 2 && partial_decision.correction_row == 2 &&
        partial_decision.next_token == 31);
    const auto full = qwen38::decide_mtp_greedy(
        drafts, std::vector<std::uint32_t>{10, 20, 30, 40});
    QWEN38_CHECK(full.accepted == 3 && full.correction_row == 3 && full.next_token == 40);
}
