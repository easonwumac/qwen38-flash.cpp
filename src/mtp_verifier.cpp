#include "qwen38/mtp_verifier.hpp"

#include <stdexcept>
#include <utility>

namespace qwen38 {
namespace {

constexpr std::size_t minimum_draft_count = 2;
constexpr std::size_t maximum_draft_count = 4;

GreedyStep greedy_from_logits(const MlxArray& logits) {
    MlxArray token = logits.argmax_all();
    MlxArray selected_logit = MlxArray::take(logits, token).astype(MLX_FLOAT32);
    selected_logit.eval();
    return {
        .token = token.item_uint32(),
        .logit = selected_logit.item_float32(),
    };
}

} // namespace

MtpTargetVerification verify_mtp_target_serial_oracle(
    const QwenModel& model,
    const std::uint32_t current_token,
    const std::span<const std::uint32_t> drafts,
    const ModelDecodeState& origin) {
    if (drafts.size() < minimum_draft_count || drafts.size() > maximum_draft_count) {
        throw std::runtime_error("MTP verifier requires draft depth between 2 and 4");
    }

    ModelDecodeState working = model.snapshot_state(origin);
    MtpTargetVerification verification{
        .draft_count = drafts.size(),
        .rows = {},
    };
    verification.rows.reserve(drafts.size() + 1);

    for (std::size_t row = 0; row <= drafts.size(); ++row) {
        const std::uint32_t input = row == 0 ? current_token : drafts[row - 1];
        TargetDecodeStep target = model.forward_decode_capture(input, working);
        GreedyStep greedy = greedy_from_logits(target.logits);
        verification.rows.push_back({
            .greedy = greedy,
            .pre_mixer_stream = std::move(target.pre_mixer_stream),
            .state_after = model.snapshot_state(working),
        });
    }
    return verification;
}

void commit_mtp_target_verification(
    MtpTargetVerification&& verification,
    const std::size_t accepted,
    ModelDecodeState& destination) {
    if (verification.rows.size() != verification.draft_count + 1) {
        throw std::runtime_error("MTP verification row count is invalid");
    }
    if (accepted > verification.draft_count) {
        throw std::runtime_error("MTP accepted count exceeds verified draft count");
    }
    destination = std::move(verification.rows[accepted].state_after);
}

} // namespace qwen38
