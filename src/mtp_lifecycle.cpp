#include "qwen38/mtp_lifecycle.hpp"

#include <limits>
#include <stdexcept>

namespace qwen38 {

void MtpPromptLifecycle::reset() noexcept {
    initialized_ = false;
    has_carry_ = false;
    carry_position_ = 0;
    next_position_ = 0;
}

std::vector<MtpPromptPair> MtpPromptLifecycle::plan_chunk(
    const std::size_t first_position,
    const std::size_t token_count) {
    if (token_count == 0) throw std::runtime_error("MTP prompt chunk must not be empty");
    if (first_position == 0) reset();
    if (initialized_ && first_position != next_position_) {
        throw std::runtime_error("MTP prompt chunks must be position-contiguous");
    }
    if (token_count > std::numeric_limits<std::size_t>::max() - first_position) {
        throw std::runtime_error("MTP prompt position overflow");
    }

    std::vector<MtpPromptPair> pairs;
    pairs.reserve(token_count);
    if (has_carry_) {
        if (carry_position_ + 1 != first_position) {
            throw std::runtime_error("MTP hidden carry is not adjacent to the prompt chunk");
        }
        pairs.push_back({
            .hidden_source = MtpHiddenSource::carry,
            .hidden_index = 0,
            .token_index = 0,
            .query_position = first_position,
        });
    }
    for (std::size_t index = 0; index + 1 < token_count; ++index) {
        pairs.push_back({
            .hidden_source = MtpHiddenSource::chunk,
            .hidden_index = index,
            .token_index = index + 1,
            .query_position = first_position + index + 1,
        });
    }

    initialized_ = true;
    has_carry_ = true;
    carry_position_ = first_position + token_count - 1;
    next_position_ = first_position + token_count;
    return pairs;
}

MtpCommitPlan plan_mtp_commit(
    const std::size_t round_offset,
    const std::size_t drafted,
    const std::size_t accepted) {
    if (accepted > drafted) throw std::runtime_error("MTP accepted count exceeds drafted count");
    if (accepted == std::numeric_limits<std::size_t>::max() ||
        round_offset > std::numeric_limits<std::size_t>::max() - (accepted + 1)) {
        throw std::runtime_error("MTP committed boundary overflow");
    }
    const std::size_t committed = 1 + accepted;
    return {
        .truncate_head_to = round_offset,
        .target_rows_to_keep = committed,
        .stash_rows = committed,
        .committed_boundary = round_offset + committed,
    };
}

} // namespace qwen38
