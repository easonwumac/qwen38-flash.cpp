#include "qwen38/qsa_checkpoint.hpp"
#include "test.hpp"

#include <array>

void run_qsa_checkpoint_tests() {
    // The original 2048-budget test dropped all 150 valid pools at token 600.
    const auto engaged = qwen38::plan_qsa_checkpoint(604, 151, 535, 600, 4, 64);
    QWEN38_CHECK(engaged.pooled_blocks == 150);
    QWEN38_CHECK(engaged.raw_start == 536);

    // Before pooling engages, even a short window must preserve every raw key.
    const auto not_engaged = qwen38::plan_qsa_checkpoint(505, 0, 0, 501, 4, 64);
    QWEN38_CHECK(not_engaged.pooled_blocks == 0);
    QWEN38_CHECK(not_engaged.raw_start == 0);
    const auto crossing = qwen38::plan_qsa_checkpoint(516, 129, 447, 513, 4, 64);
    QWEN38_CHECK(crossing.pooled_blocks == 128);
    QWEN38_CHECK(crossing.raw_start == 449);
    const auto untrimmed = qwen38::plan_qsa_checkpoint(604, 151, 0, 600, 4, 0);
    QWEN38_CHECK(untrimmed.pooled_blocks == 150);
    QWEN38_CHECK(untrimmed.raw_start == 0);

    for (const std::size_t budget : {512U, 2048U}) {
        for (const std::size_t window : {0U, 16U, 64U}) {
            for (std::size_t origin = budget - 32; origin < budget + 32; ++origin) {
                for (std::size_t rows = 1; rows <= 5; ++rows) {
                    const auto total = origin + rows;
                    const auto pools = total / 4 > budget / 4 ? total / 4 : 0;
                    const auto raw = window != 0 && pools != 0
                        ? total - (window + rows) : 0;
                    for (std::size_t accepted = 0; accepted <= rows; ++accepted) {
                        const auto tokens = origin + accepted;
                        const auto plan = qwen38::plan_qsa_checkpoint(
                            total, pools, raw, tokens, 4, window);
                        QWEN38_CHECK(plan.pooled_blocks <= tokens / 4);
                        QWEN38_CHECK(plan.pooled_blocks <= pools);
                        QWEN38_CHECK(plan.raw_start >= raw);
                        QWEN38_CHECK(plan.raw_start <= plan.pooled_blocks * 4);
                    }
                }
            }
        }
    }
    const std::array<std::array<std::size_t, 6>, 4> invalid{{
        {600, 150, 536, 600, 0, 64},
        {600, 150, 536, 601, 4, 64},
        {600, 151, 536, 600, 4, 64},
        {600, 0, 536, 600, 4, 64},
    }};
    for (const auto& args : invalid) {
        bool failed = false;
        try {
            static_cast<void>(qwen38::plan_qsa_checkpoint(
                args[0], args[1], args[2], args[3], args[4], args[5]));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        QWEN38_CHECK(failed);
    }
}
