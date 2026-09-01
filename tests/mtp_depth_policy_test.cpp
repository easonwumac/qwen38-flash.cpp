#include "qwen38/mtp_depth_policy.hpp"
#include "qwen38/mtp_profitability.hpp"

#include "test.hpp"

#include <cstdlib>

void run_mtp_depth_policy_tests() {
    qwen38::MtpDepthPolicy fixed_two(2, 10);
    for (int round = 0; round < 16; ++round) fixed_two.observe(2, 2);
    QWEN38_CHECK(fixed_two.depth() == 2);
    QWEN38_CHECK(fixed_two.promotions() == 0);

    qwen38::MtpDepthPolicy long_prompt(3, 2049);
    for (int round = 0; round < 16; ++round) long_prompt.observe(2, 2);
    QWEN38_CHECK(long_prompt.depth() == 2);
    QWEN38_CHECK(!long_prompt.probing());

    qwen38::MtpDepthPolicy high_acceptance(3, 32);
    for (int round = 0; round < 5; ++round) high_acceptance.observe(2, 2);
    for (int round = 0; round < 3; ++round) high_acceptance.observe(2, 0);
    QWEN38_CHECK(high_acceptance.depth() == 3);
    QWEN38_CHECK(high_acceptance.promotions() == 1);

    qwen38::MtpDepthPolicy low_acceptance(3, 32);
    for (int round = 0; round < 4; ++round) low_acceptance.observe(2, 2);
    for (int round = 0; round < 4; ++round) low_acceptance.observe(2, 0);
    QWEN38_CHECK(low_acceptance.depth() == 2);
    QWEN38_CHECK(low_acceptance.promotions() == 0);

    for (int round = 0; round < 11; ++round) high_acceptance.observe(3, round < 5 ? 2 : 1);
    QWEN38_CHECK(high_acceptance.depth() == 3);
    high_acceptance.observe(3, 0);
    QWEN38_CHECK(high_acceptance.depth() == 2);
    QWEN38_CHECK(high_acceptance.demotions() == 1);

    qwen38::MtpDepthPolicy exact_half(3, 32);
    for (int round = 0; round < 8; ++round) exact_half.observe(2, 2);
    for (int round = 0; round < 12; ++round) exact_half.observe(3, round < 6 ? 3 : 0);
    QWEN38_CHECK(exact_half.depth() == 3);
    QWEN38_CHECK(exact_half.demotions() == 0);

    setenv("QWEN38_MTP_EARLY_DEMOTION", "1", 1);
    qwen38::MtpDepthPolicy failed_promotion(3, 32);
    for (int round = 0; round < 8; ++round) failed_promotion.observe(2, 2);
    for (int round = 0; round < 3; ++round) failed_promotion.observe(3, 1);
    QWEN38_CHECK(failed_promotion.depth() == 3);
    failed_promotion.observe(3, 1);
    QWEN38_CHECK(failed_promotion.depth() == 2);
    QWEN38_CHECK(failed_promotion.demotions() == 1);

    qwen38::MtpDepthPolicy passed_probation(3, 32);
    for (int round = 0; round < 8; ++round) passed_probation.observe(2, 2);
    for (int round = 0; round < 4; ++round) passed_probation.observe(3, 2);
    QWEN38_CHECK(passed_probation.depth() == 3);
    unsetenv("QWEN38_MTP_EARLY_DEMOTION");

    qwen38::MtpDepthPolicy explicit_four(4, 100000);
    QWEN38_CHECK(explicit_four.depth() == 4);

    QWEN38_CHECK(qwen38::should_fallback_mtp(2, 2, 2, 0));
    QWEN38_CHECK(qwen38::should_fallback_mtp(2, 2, 11, 10));
    QWEN38_CHECK(!qwen38::should_fallback_mtp(1, 2, 2, 0));
    QWEN38_CHECK(!qwen38::should_fallback_mtp(2, 2, 28, 39));
    QWEN38_CHECK(!qwen38::should_fallback_mtp(2, 2, 11, 11));
    QWEN38_CHECK(!qwen38::cache_mtp_as_profitable(2, 0, 1));
    QWEN38_CHECK(!qwen38::cache_mtp_as_profitable(4, 3, 1));
    QWEN38_CHECK(qwen38::cache_mtp_as_profitable(47, 59, 1));
    QWEN38_CHECK(qwen38::cache_mtp_as_profitable(59, 70, 0));

    qwen38::MtpProfitabilityGuard early_loss;
    early_loss.observe(0);
    QWEN38_CHECK(!early_loss.should_fallback(2));
    early_loss.observe(0);
    QWEN38_CHECK(early_loss.should_fallback(2));

    qwen38::MtpProfitabilityGuard profitable_empty_streak;
    for (int round = 0; round < 6; ++round) profitable_empty_streak.observe(2);
    profitable_empty_streak.observe(0);
    profitable_empty_streak.observe(0);
    QWEN38_CHECK(!profitable_empty_streak.should_fallback(2));

    qwen38::MtpProfitabilityGuard exact_window;
    for (std::size_t round = 0; round < qwen38::MtpProfitabilityGuard::window_size; ++round) {
        exact_window.observe(1);
    }
    QWEN38_CHECK(!exact_window.should_fallback(2));

    qwen38::MtpProfitabilityGuard losing_window;
    for (std::size_t round = 0; round < qwen38::MtpProfitabilityGuard::window_size; ++round) {
        losing_window.observe(round < 15 ? 1 : 0);
    }
    QWEN38_CHECK(losing_window.should_fallback(2));
}
