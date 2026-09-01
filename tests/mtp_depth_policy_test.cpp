#include "qwen38/mtp_depth_policy.hpp"

#include "test.hpp"

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

    qwen38::MtpDepthPolicy explicit_four(4, 100000);
    QWEN38_CHECK(explicit_four.depth() == 4);
}
