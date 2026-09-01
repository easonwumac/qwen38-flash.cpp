#pragma once

#include <array>
#include <cstddef>

namespace qwen38 {

// A short zero-accept streak is evidence of a losing drafter only when the
// request has not already demonstrated at least one accepted token per round.
// That break-even proxy prevents random empty rounds from disabling a
// cumulatively profitable MTP stream while preserving the early-loss guard.
[[nodiscard]] inline bool should_fallback_mtp(
    const std::size_t zero_accept_streak,
    const std::size_t fallback_streak,
    const std::size_t rounds,
    const std::size_t accepted) noexcept {
    return zero_accept_streak >= fallback_streak && accepted < rounds;
}

class MtpProfitabilityGuard final {
public:
    static constexpr std::size_t window_size = 16;

    void observe(const std::size_t accepted) noexcept {
        zero_accept_streak_ = accepted == 0 ? zero_accept_streak_ + 1 : 0;
        ++rounds_;
        total_accepted_ += accepted;
        if (window_rows_ < window_size) {
            accepted_window_[window_rows_++] = accepted;
            window_accepted_ += accepted;
            return;
        }
        window_accepted_ -= accepted_window_[cursor_];
        accepted_window_[cursor_] = accepted;
        window_accepted_ += accepted;
        cursor_ = (cursor_ + 1) % window_size;
    }

    [[nodiscard]] bool should_fallback(const std::size_t fallback_streak) const noexcept {
        if (window_rows_ < window_size) {
            return should_fallback_mtp(
                zero_accept_streak_, fallback_streak, rounds_, total_accepted_);
        }
        return window_accepted_ < window_size;
    }

    [[nodiscard]] std::size_t rounds() const noexcept { return rounds_; }
    [[nodiscard]] std::size_t total_accepted() const noexcept { return total_accepted_; }
    [[nodiscard]] std::size_t window_accepted() const noexcept { return window_accepted_; }
    [[nodiscard]] std::size_t zero_accept_streak() const noexcept {
        return zero_accept_streak_;
    }

private:
    std::array<std::size_t, window_size> accepted_window_{};
    std::size_t rounds_{0};
    std::size_t total_accepted_{0};
    std::size_t window_rows_{0};
    std::size_t window_accepted_{0};
    std::size_t cursor_{0};
    std::size_t zero_accept_streak_{0};
};

} // namespace qwen38
