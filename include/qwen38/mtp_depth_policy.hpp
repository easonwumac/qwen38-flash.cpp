#pragma once

#include <cstddef>

namespace qwen38 {

class MtpDepthPolicy final {
public:
    MtpDepthPolicy(std::size_t maximum_depth, std::size_t prompt_tokens);

    [[nodiscard]] std::size_t depth() const noexcept { return depth_; }
    [[nodiscard]] std::size_t promotions() const noexcept { return promotions_; }
    [[nodiscard]] std::size_t demotions() const noexcept { return demotions_; }
    [[nodiscard]] bool probing() const noexcept { return probing_; }

    void observe(std::size_t proposed, std::size_t accepted);

private:
    std::size_t maximum_depth_{0};
    std::size_t depth_{0};
    bool probing_{false};
    bool monitoring_{false};
    bool adaptive_four_{false};
    bool promotion_probation_complete_{false};
    std::size_t probe_rounds_{0};
    std::size_t probe_accepted_{0};
    std::size_t monitor_rounds_{0};
    std::size_t monitor_proposed_{0};
    std::size_t monitor_accepted_{0};
    std::size_t monitor_fourth_accepted_{0};
    std::size_t monitor_losing_windows_{0};
    std::size_t promotions_{0};
    std::size_t demotions_{0};
};

} // namespace qwen38
