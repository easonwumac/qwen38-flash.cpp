#include "qwen38/mtp_runner.hpp"

#include "qwen38/mtp_lifecycle.hpp"
#include "qwen38/mtp_verifier.hpp"

#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace qwen38 {
namespace {

std::uint32_t argmax_token(const MlxArray& logits) {
    MlxArray token = logits.argmax_all();
    token.eval();
    return token.item_uint32();
}

std::vector<std::uint32_t> draft_lazy_chain(
    const QwenMtpHead& head,
    const MlxArray& previous_target_stream,
    const std::uint32_t current_token,
    const std::size_t query_position,
    const std::size_t draft_depth,
    MtpDecodeState& head_state) {
    const std::vector<std::int32_t> first_value{static_cast<std::int32_t>(current_token)};
    const std::vector<int> scalar_shape{1};
    MlxArray token = MlxArray::from_int32(first_value, scalar_shape);
    MlxArray stream = previous_target_stream.share();
    std::vector<MlxArray> draft_arrays;
    draft_arrays.reserve(draft_depth);
    for (std::size_t index = 0; index < draft_depth; ++index) {
        MtpDecodeStep step = head.forward_decode_lazy_token(
            stream, token, query_position + index, head_state);
        draft_arrays.push_back(step.logits.argmax_all().reshape(scalar_shape));
        token = draft_arrays.back().share();
        stream = std::move(step.pre_mixer_stream);
    }
    std::vector<const MlxArray*> outputs;
    outputs.reserve(draft_arrays.size());
    for (const MlxArray& draft : draft_arrays) outputs.push_back(&draft);
    MlxArray::eval_all(outputs);
    std::vector<std::uint32_t> drafts;
    drafts.reserve(draft_arrays.size());
    for (const MlxArray& draft : draft_arrays) drafts.push_back(draft.item_uint32());
    return drafts;
}

MtpRoundStep finish_greedy_mtp_round(
    const QwenModel& target,
    const QwenMtpHead& head,
    const std::uint32_t current_token,
    const MlxArray& previous_target_stream,
    const std::size_t query_position,
    std::vector<std::uint32_t> drafts,
    const double draft_ms,
    MtpDecodeState head_origin,
    ModelDecodeState& target_state,
    MtpDecodeState& head_state,
    const std::span<const std::uint32_t> stop_tokens) {
    if (drafts.size() < 2 || drafts.size() > 4) {
        throw std::runtime_error("MTP proposal depth must be between 2 and 4");
    }

    if (query_position != target_state.token_count) {
        throw std::runtime_error("MTP query position must match target committed length");
    }
    if (query_position > std::numeric_limits<std::size_t>::max() - drafts.size()) {
        throw std::runtime_error("MTP query position overflow");
    }

    const auto verify_started = std::chrono::steady_clock::now();
    MtpTargetVerification verification = verify_mtp_target_layer_major_reference(
        target, current_token, drafts, target_state);
    const double verify_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - verify_started).count();
    std::vector<std::uint32_t> target_rows;
    target_rows.reserve(verification.rows.size());
    for (const MtpTargetVerifyRow& row : verification.rows) {
        target_rows.push_back(row.greedy.token);
    }
    const MtpGreedyDecision decision = decide_mtp_greedy(
        drafts, target_rows, stop_tokens);
    MlxArray next_target_stream =
        verification.rows[decision.correction_row].pre_mixer_stream.share();

    // Speculative head rows are never committed. Rebuild only the current row
    // and accepted draft rows from target-captured hidden streams.
    const auto commit_started = std::chrono::steady_clock::now();
    head_state = std::move(head_origin);
    const char* batch_commit = std::getenv("QWEN38_BATCH_MTP_COMMIT");
    const bool batch_commit_enabled =
        batch_commit == nullptr || std::string_view(batch_commit) != "0";
    if (decision.accepted != 0 && batch_commit_enabled) {
        std::vector<const MlxArray*> committed_streams{&previous_target_stream};
        std::vector<std::uint32_t> committed_tokens{current_token};
        committed_streams.reserve(decision.accepted + 1);
        committed_tokens.reserve(decision.accepted + 1);
        for (std::size_t index = 0; index < decision.accepted; ++index) {
            committed_streams.push_back(&verification.rows[index].pre_mixer_stream);
            committed_tokens.push_back(drafts[index]);
        }
        head.consume_committed_batch(
            committed_streams, committed_tokens, query_position, head_state);
    } else {
        head.consume_decode(
            previous_target_stream, current_token, query_position, head_state);
        for (std::size_t index = 0; index < decision.accepted; ++index) {
            head.consume_decode(
                verification.rows[index].pre_mixer_stream,
                drafts[index],
                query_position + index + 1,
                head_state);
        }
    }
    commit_mtp_target_verification(
        std::move(verification), decision.accepted, target_state);
    const std::size_t next_query_position = query_position + decision.accepted + 1;
    if (target_state.token_count != next_query_position) {
        throw std::runtime_error("MTP committed target length mismatch");
    }
    const double commit_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - commit_started).count();

    std::vector<std::uint32_t> emitted;
    emitted.reserve(decision.accepted + 1);
    for (std::size_t index = 0; index < decision.accepted; ++index) {
        emitted.push_back(drafts[index]);
    }
    emitted.push_back(decision.next_token);
    return {
        .emitted_tokens = std::move(emitted),
        .draft_tokens = std::move(drafts),
        .accepted = decision.accepted,
        .next_current_token = decision.next_token,
        .next_query_position = next_query_position,
        .next_target_stream = std::move(next_target_stream),
        .draft_ms = draft_ms,
        .verify_ms = verify_ms,
        .commit_ms = commit_ms,
    };
}

} // namespace

MtpRoundStep run_greedy_mtp_round_reference(
    const QwenModel& target,
    const QwenMtpHead& head,
    const std::uint32_t current_token,
    const MlxArray& previous_target_stream,
    const std::size_t query_position,
    const std::size_t draft_depth,
    ModelDecodeState& target_state,
    MtpDecodeState& head_state,
    const std::span<const std::uint32_t> stop_tokens) {
    if (draft_depth < 2 || draft_depth > 4) {
        throw std::runtime_error("MTP draft depth must be between 2 and 4");
    }
    MtpDecodeState head_origin = head.snapshot_state(head_state);
    const auto draft_started = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> drafts;
    const char* lazy_chain = std::getenv("QWEN38_LAZY_MTP_DRAFT_CHAIN");
    const bool lazy_chain_enabled =
        lazy_chain == nullptr || std::string_view(lazy_chain) != "0";
    if (lazy_chain_enabled) {
        drafts = draft_lazy_chain(
            head, previous_target_stream, current_token, query_position,
            draft_depth, head_state);
    } else {
        drafts.reserve(draft_depth);
        MtpDecodeStep draft = head.forward_decode(
            previous_target_stream, current_token, query_position, head_state);
        drafts.push_back(argmax_token(draft.logits));
        for (std::size_t index = 1; index < draft_depth; ++index) {
            draft = head.forward_decode(
                draft.pre_mixer_stream,
                drafts.back(),
                query_position + index,
                head_state);
            drafts.push_back(argmax_token(draft.logits));
        }
    }
    const double draft_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - draft_started).count();
    return finish_greedy_mtp_round(
        target, head, current_token, previous_target_stream, query_position,
        std::move(drafts), draft_ms, std::move(head_origin), target_state, head_state,
        stop_tokens);
}

MtpRoundStep run_greedy_external_draft_round_reference(
    const QwenModel& target,
    const QwenMtpHead& head,
    const std::uint32_t current_token,
    const MlxArray& previous_target_stream,
    const std::size_t query_position,
    std::vector<std::uint32_t> drafts,
    ModelDecodeState& target_state,
    MtpDecodeState& head_state,
    const std::span<const std::uint32_t> stop_tokens) {
    MtpDecodeState head_origin = head.snapshot_state(head_state);
    return finish_greedy_mtp_round(
        target, head, current_token, previous_target_stream, query_position,
        std::move(drafts), 0.0, std::move(head_origin), target_state, head_state,
        stop_tokens);
}

} // namespace qwen38
