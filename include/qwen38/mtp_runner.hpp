#pragma once

#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qwen38 {

struct MtpRoundStep {
    std::vector<std::uint32_t> emitted_tokens;
    std::vector<std::uint32_t> draft_tokens;
    std::size_t accepted{0};
    std::uint32_t next_current_token{0};
    std::size_t next_query_position{0};
    MlxArray next_target_stream;
    double draft_ms{0.0};
    double verify_ms{0.0};
    double commit_ms{0.0};
};

// current_token has already been emitted but is not yet present in target_state.
// previous_target_stream is the committed target hidden immediately preceding it.
[[nodiscard]] MtpRoundStep run_greedy_mtp_round_reference(
    const QwenModel& target,
    const QwenMtpHead& head,
    std::uint32_t current_token,
    const MlxArray& previous_target_stream,
    std::size_t query_position,
    std::size_t draft_depth,
    ModelDecodeState& target_state,
    MtpDecodeState& head_state);

} // namespace qwen38
