#include "qwen38/mtp_verifier.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace qwen38 {
namespace {

constexpr std::size_t minimum_draft_count = 2;
constexpr std::size_t maximum_draft_count = 24;

void validate_draft_count(const std::size_t count) {
    if (count < minimum_draft_count || count > maximum_draft_count) {
        throw std::runtime_error("speculative verifier requires 2 to 24 draft tokens");
    }
}

GreedyStep greedy_from_logits(const MlxArray& logits) {
    MlxArray token = logits.argmax_all();
    MlxArray selected_logit = MlxArray::take(logits, token).astype(MLX_FLOAT32);
    selected_logit.eval();
    return {
        .token = token.item_uint32(),
        .logit = selected_logit.item_float32(),
    };
}

std::vector<GreedyStep> greedy_batch_from_logits(
    const std::span<const MlxArray* const> logits) {
    std::vector<MlxArray> tokens;
    std::vector<MlxArray> selected_logits;
    tokens.reserve(logits.size());
    selected_logits.reserve(logits.size());
    for (const MlxArray* row : logits) {
        tokens.push_back(row->argmax_all());
        selected_logits.push_back(
            MlxArray::take(*row, tokens.back()).astype(MLX_FLOAT32));
    }
    std::vector<const MlxArray*> outputs;
    outputs.reserve(selected_logits.size());
    for (const MlxArray& selected : selected_logits) outputs.push_back(&selected);
    MlxArray::eval_all(outputs);

    std::vector<GreedyStep> result;
    result.reserve(logits.size());
    for (std::size_t row = 0; row < logits.size(); ++row) {
        result.push_back({
            .token = tokens[row].item_uint32(),
            .logit = selected_logits[row].item_float32(),
        });
    }
    return result;
}

} // namespace

MtpTargetVerification verify_mtp_target_serial_oracle(
    const QwenModel& model,
    const std::uint32_t current_token,
    const std::span<const std::uint32_t> drafts,
    const ModelDecodeState& origin) {
    validate_draft_count(drafts.size());

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
            .final_mixed = MlxArray{},
            .pre_mixer_stream = std::move(target.pre_mixer_stream),
            .state_after = model.snapshot_state(working),
        });
    }
    return verification;
}

MtpTargetVerification verify_mtp_target_layer_major_reference(
    const QwenModel& model,
    const std::uint32_t current_token,
    const std::span<const std::uint32_t> drafts,
    const ModelDecodeState& origin) {
    validate_draft_count(drafts.size());
    std::vector<std::uint32_t> inputs;
    inputs.reserve(drafts.size() + 1);
    inputs.push_back(current_token);
    inputs.insert(inputs.end(), drafts.begin(), drafts.end());

    std::vector<TargetVerifyStep> target =
        model.forward_verify_layer_major_reference(inputs, origin);
    MtpTargetVerification verification{
        .draft_count = drafts.size(),
        .rows = {},
    };
    verification.rows.reserve(target.size());
    const char* batch_argmax = std::getenv("QWEN38_BATCH_VERIFY_ARGMAX");
    const bool batch_argmax_enabled =
        batch_argmax == nullptr || std::string_view(batch_argmax) != "0";
    std::vector<GreedyStep> greedy_rows;
    if (batch_argmax_enabled) {
        std::vector<const MlxArray*> logits;
        logits.reserve(target.size());
        for (const TargetVerifyStep& row : target) logits.push_back(&row.logits);
        greedy_rows = greedy_batch_from_logits(logits);
    }
    for (std::size_t index = 0; index < target.size(); ++index) {
        TargetVerifyStep& row = target[index];
        verification.rows.push_back({
            .greedy = batch_argmax_enabled
                ? greedy_rows[index]
                : greedy_from_logits(row.logits),
            .final_mixed = std::move(row.final_mixed),
            .pre_mixer_stream = std::move(row.pre_mixer_stream),
            .state_after = std::move(row.state_after),
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
