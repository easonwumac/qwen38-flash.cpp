#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qwen38 {

enum class MtpHiddenSource { carry, chunk };

struct MtpPromptPair {
    MtpHiddenSource hidden_source{MtpHiddenSource::chunk};
    std::size_t hidden_index{0};
    std::size_t token_index{0};
    std::size_t query_position{0};

    bool operator==(const MtpPromptPair&) const = default;
};

class MtpPromptLifecycle final {
public:
    [[nodiscard]] std::vector<MtpPromptPair> plan_chunk(
        std::size_t first_position,
        std::size_t token_count);
    void reset() noexcept;

    [[nodiscard]] bool has_carry() const noexcept { return has_carry_; }
    [[nodiscard]] std::size_t next_position() const noexcept { return next_position_; }

private:
    bool initialized_{false};
    bool has_carry_{false};
    std::size_t carry_position_{0};
    std::size_t next_position_{0};
};

struct MtpCommitPlan {
    std::size_t truncate_head_to{0};
    std::size_t target_rows_to_keep{0};
    std::size_t stash_rows{0};
    std::size_t committed_boundary{0};
};

[[nodiscard]] MtpCommitPlan plan_mtp_commit(
    std::size_t round_offset,
    std::size_t drafted,
    std::size_t accepted);

struct MtpGreedyDecision {
    std::size_t accepted{0};
    std::size_t correction_row{0};
    std::uint32_t next_token{0};
};

[[nodiscard]] MtpGreedyDecision decide_mtp_greedy(
    std::span<const std::uint32_t> drafts,
    std::span<const std::uint32_t> target_argmax_rows,
    std::span<const std::uint32_t> stop_tokens = {});

} // namespace qwen38
