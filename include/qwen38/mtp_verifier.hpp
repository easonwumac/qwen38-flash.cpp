#pragma once

#include "qwen38/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qwen38 {

// Correctness oracle for speculative verification. This deliberately executes
// one target decode per row; production MTP must replace the implementation
// with a layer-major S=2..4 pass while preserving these exact results.
struct MtpTargetVerifyRow {
    GreedyStep greedy;
    MlxArray pre_mixer_stream;
    ModelDecodeState state_after;
};

struct MtpTargetVerification {
    std::size_t draft_count{0};
    std::vector<MtpTargetVerifyRow> rows;
};

[[nodiscard]] MtpTargetVerification verify_mtp_target_serial_oracle(
    const QwenModel& model,
    std::uint32_t current_token,
    std::span<const std::uint32_t> drafts,
    const ModelDecodeState& origin);

[[nodiscard]] MtpTargetVerification verify_mtp_target_layer_major_reference(
    const QwenModel& model,
    std::uint32_t current_token,
    std::span<const std::uint32_t> drafts,
    const ModelDecodeState& origin);

// accepted=N keeps the checkpoint after current_token plus N accepted drafts.
// The correction/bonus token remains unconsumed and becomes the next round's
// current token.
void commit_mtp_target_verification(
    MtpTargetVerification&& verification,
    std::size_t accepted,
    ModelDecodeState& destination);

} // namespace qwen38
