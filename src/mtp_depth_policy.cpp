#include "qwen38/mtp_depth_policy.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace qwen38 {
namespace {

constexpr std::size_t short_prompt_limit = 2048;
constexpr std::size_t probe_round_limit = 8;
constexpr std::size_t probe_accept_threshold = 10;
constexpr std::size_t monitor_round_limit = 12;
constexpr std::size_t promotion_probation_round_limit = 4;

bool early_demotion_enabled() {
    const char* value = std::getenv("QWEN38_MTP_EARLY_DEMOTION");
    return value != nullptr && std::string_view(value) == "1";
}

bool demotion_enabled() {
    const char* value = std::getenv("QWEN38_MTP_DEMOTION");
    return value == nullptr || std::string_view(value) != "0";
}

} // namespace

MtpDepthPolicy::MtpDepthPolicy(
    const std::size_t maximum_depth,
    const std::size_t prompt_tokens)
    : maximum_depth_(maximum_depth), depth_(maximum_depth) {
    if (maximum_depth != 0 && (maximum_depth < 2 || maximum_depth > 4)) {
        throw std::runtime_error("MTP policy depth must be 0 or between 2 and 4");
    }
    if (maximum_depth == 3) {
        depth_ = 2;
        probing_ = prompt_tokens <= short_prompt_limit;
    }
}

void MtpDepthPolicy::observe(
    const std::size_t proposed,
    const std::size_t accepted) {
    if (accepted > proposed) {
        throw std::runtime_error("MTP accepted count exceeds proposals");
    }
    if (depth_ == 0 || proposed == 0) return;
    if (probing_) {
        ++probe_rounds_;
        probe_accepted_ += accepted;
        if (probe_rounds_ == probe_round_limit) {
            probing_ = false;
            if (probe_accepted_ >= probe_accept_threshold) {
                depth_ = maximum_depth_;
                monitoring_ = true;
                ++promotions_;
            }
        }
        return;
    }
    if (!monitoring_ || depth_ != 3 || !demotion_enabled()) return;
    ++monitor_rounds_;
    monitor_proposed_ += proposed;
    monitor_accepted_ += accepted;
    if (!promotion_probation_complete_ && early_demotion_enabled() &&
        monitor_rounds_ == promotion_probation_round_limit &&
        monitor_accepted_ * 2 < monitor_proposed_) {
        depth_ = 2;
        monitoring_ = false;
        ++demotions_;
        return;
    }
    if (monitor_rounds_ == promotion_probation_round_limit) {
        promotion_probation_complete_ = true;
    }
    if (monitor_rounds_ == monitor_round_limit) {
        if (monitor_accepted_ * 2 < monitor_proposed_) {
            ++monitor_losing_windows_;
            if (monitor_losing_windows_ == 2) {
                depth_ = 2;
                monitoring_ = false;
                ++demotions_;
            }
        } else {
            monitor_losing_windows_ = 0;
        }
        monitor_rounds_ = 0;
        monitor_proposed_ = 0;
        monitor_accepted_ = 0;
    }
}

} // namespace qwen38
