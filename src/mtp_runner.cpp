#include "qwen38/mtp_runner.hpp"

#include "qwen38/mtp_lifecycle.hpp"
#include "qwen38/mtp_verifier.hpp"

#include <chrono>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace qwen38 {
namespace {

void append_head_qsa_state(
    std::vector<const MlxArray*>& outputs,
    const MtpDecodeState& state) {
    if (state.layer.full_attention.qsa_raw_keys.get().ctx != nullptr) {
        outputs.push_back(&state.layer.full_attention.qsa_raw_keys);
    }
    if (state.layer.full_attention.qsa_pooled_keys.get().ctx != nullptr) {
        outputs.push_back(&state.layer.full_attention.qsa_pooled_keys);
    }
}

std::uint32_t argmax_token(const MlxArray& logits, const MtpDecodeState& state) {
    MlxArray token = logits.argmax_all();
    std::vector<const MlxArray*> outputs{&token};
    append_head_qsa_state(outputs, state);
    MlxArray::eval_all(outputs);
    return token.item_uint32();
}

struct DraftChain {
    std::vector<std::uint32_t> primary;
    std::vector<std::uint32_t> secondary;
};

bool top2_oracle_enabled() {
    const char* value = std::getenv("QWEN38_MTP_TOP2_ORACLE");
    return value != nullptr && std::string_view(value) == "1";
}

bool top2_oracle_all_positions() {
    const char* value = std::getenv("QWEN38_MTP_TOP2_ORACLE");
    return value != nullptr && std::string_view(value) == "all";
}

DraftChain draft_lazy_chain(
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
    const bool collect_top2 = top2_oracle_enabled();
    const bool collect_all_top2 = top2_oracle_all_positions();
    std::vector<MlxArray> draft_arrays;
    std::vector<MlxArray> top2_arrays;
    std::vector<std::size_t> top2_positions;
    draft_arrays.reserve(draft_depth);
    if (collect_top2) top2_arrays.reserve(draft_depth);
    for (std::size_t index = 0; index < draft_depth; ++index) {
        MtpDecodeStep step = head.forward_decode_lazy_token(
            stream, token, query_position + index, head_state);
        draft_arrays.push_back(step.logits.argmax_all().reshape(scalar_shape));
        if (collect_top2 && (collect_all_top2 || index + 1 == draft_depth)) {
            const std::vector<int> logits_shape = step.logits.shape();
            if (logits_shape.empty() || logits_shape.back() < 2) {
                throw std::runtime_error("MTP top-2 oracle requires a vocabulary axis");
            }
            std::vector<int> start(logits_shape.size(), 0);
            std::vector<int> stop = logits_shape;
            std::vector<int> strides(logits_shape.size(), 1);
            start.back() = logits_shape.back() - 2;
            top2_arrays.push_back(
                step.logits.argpartition_axis(-2, -1)
                    .slice(start, stop, strides)
                    .reshape(std::vector<int>{2})
                    .astype(MLX_FLOAT32));
            top2_positions.push_back(index);
        }
        token = draft_arrays.back().share();
        stream = std::move(step.pre_mixer_stream);
    }
    std::vector<const MlxArray*> outputs;
    outputs.reserve(draft_arrays.size() + 2);
    for (const MlxArray& draft : draft_arrays) outputs.push_back(&draft);
    for (const MlxArray& top2 : top2_arrays) outputs.push_back(&top2);
    append_head_qsa_state(outputs, head_state);
    MlxArray::eval_all(outputs);
    DraftChain chain;
    chain.primary.reserve(draft_arrays.size());
    chain.secondary.assign(draft_arrays.size(), std::numeric_limits<std::uint32_t>::max());
    for (const MlxArray& draft : draft_arrays) chain.primary.push_back(draft.item_uint32());
    for (std::size_t index = 0; index < top2_arrays.size(); ++index) {
        const std::vector<float> candidates = top2_arrays[index].to_float32();
        if (candidates.size() != 2) {
            throw std::runtime_error("MTP top-2 oracle returned an invalid candidate set");
        }
        const std::uint32_t first = static_cast<std::uint32_t>(candidates[0]);
        const std::uint32_t second = static_cast<std::uint32_t>(candidates[1]);
        const std::size_t position = top2_positions[index];
        chain.secondary[position] = first == chain.primary[position] ? second : first;
    }
    return chain;
}

MtpRoundStep finish_greedy_mtp_round(
    const QwenModel& target,
    const QwenMtpHead& head,
    const std::uint32_t current_token,
    const MlxArray& previous_target_stream,
    const std::size_t query_position,
    std::vector<std::uint32_t> drafts,
    std::vector<std::uint32_t> secondary_drafts,
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
    std::array<std::size_t, 4> top2_rejected{};
    std::array<std::size_t, 4> top2_recovered{};
    if (decision.accepted < drafts.size() &&
        secondary_drafts.size() == drafts.size() &&
        secondary_drafts[decision.accepted] !=
            std::numeric_limits<std::uint32_t>::max()) {
        const std::size_t rejected_position = decision.accepted;
        top2_rejected[rejected_position] = 1;
        if (secondary_drafts[rejected_position] == target_rows[rejected_position]) {
            top2_recovered[rejected_position] = 1;
        }
    }
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
        .top2_rejected_by_position = top2_rejected,
        .top2_recovered_by_position = top2_recovered,
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
    std::vector<std::uint32_t> secondary_drafts;
    const char* lazy_chain = std::getenv("QWEN38_LAZY_MTP_DRAFT_CHAIN");
    const bool lazy_chain_enabled =
        lazy_chain == nullptr || std::string_view(lazy_chain) != "0";
    if (lazy_chain_enabled) {
        DraftChain chain = draft_lazy_chain(
            head, previous_target_stream, current_token, query_position,
            draft_depth, head_state);
        drafts = std::move(chain.primary);
        secondary_drafts = std::move(chain.secondary);
    } else {
        drafts.reserve(draft_depth);
        MtpDecodeStep draft = head.forward_decode(
            previous_target_stream, current_token, query_position, head_state);
        drafts.push_back(argmax_token(draft.logits, head_state));
        for (std::size_t index = 1; index < draft_depth; ++index) {
            draft = head.forward_decode(
                draft.pre_mixer_stream,
                drafts.back(),
                query_position + index,
                head_state);
            drafts.push_back(argmax_token(draft.logits, head_state));
        }
    }
    const double draft_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - draft_started).count();
    return finish_greedy_mtp_round(
        target, head, current_token, previous_target_stream, query_position,
        std::move(drafts), std::move(secondary_drafts), draft_ms,
        std::move(head_origin), target_state, head_state,
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
        std::move(drafts), {}, 0.0, std::move(head_origin), target_state, head_state,
        stop_tokens);
}

} // namespace qwen38
