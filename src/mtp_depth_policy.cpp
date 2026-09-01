#include "qwen38/mtp_depth_policy.hpp"

#include <stdexcept>

namespace qwen38 {
namespace {

constexpr std::size_t short_prompt_limit = 2048;
constexpr std::size_t probe_round_limit = 8;
constexpr std::size_t probe_accept_threshold = 10;
constexpr std::size_t monitor_round_limit = 12;

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
    if (!monitoring_ || depth_ != 3) return;
    ++monitor_rounds_;
    monitor_proposed_ += proposed;
    monitor_accepted_ += accepted;
    if (monitor_rounds_ == monitor_round_limit) {
        if (monitor_accepted_ * 2 < monitor_proposed_) {
            depth_ = 2;
            monitoring_ = false;
            ++demotions_;
        } else {
            monitor_rounds_ = 0;
            monitor_proposed_ = 0;
            monitor_accepted_ = 0;
        }
    }
}

} // namespace qwen38
