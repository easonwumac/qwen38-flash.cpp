#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace qwen38 {

struct QsaCheckpointPlan {
    std::size_t pooled_blocks;
    std::size_t raw_start;
};

// Retire raw keys only after their complete block is present in this checkpoint.
// Selection budgets decide when attention uses pooling, not whether an already
// computed, causal pool may be retained during speculative rollback.
[[nodiscard]] inline QsaCheckpointPlan plan_qsa_checkpoint(
    const std::size_t complete_tokens,
    const std::size_t complete_pooled_blocks,
    const std::size_t complete_raw_start,
    const std::size_t checkpoint_tokens,
    const std::size_t ratio,
    const std::size_t window) {
    if (ratio == 0 || checkpoint_tokens > complete_tokens ||
        complete_raw_start > checkpoint_tokens ||
        complete_pooled_blocks > complete_tokens / ratio) {
        throw std::runtime_error("invalid QSA checkpoint geometry");
    }
    const std::size_t blocks = std::min(
        complete_pooled_blocks, checkpoint_tokens / ratio);
    const std::size_t frontier = blocks * ratio;
    if (complete_raw_start > frontier) {
        throw std::runtime_error("QSA checkpoint raw window misses the pooling frontier");
    }
    const std::size_t desired_start = window == 0 ? complete_raw_start
        : checkpoint_tokens > window ? checkpoint_tokens - window : 0;
    return {blocks, std::max(complete_raw_start, std::min(desired_start, frontier))};
}

} // namespace qwen38
