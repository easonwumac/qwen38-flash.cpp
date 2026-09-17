#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

std::uint32_t argmax(const qwen38::MlxArray& logits) {
    const std::vector<float> values = logits.astype(MLX_FLOAT32).to_float32();
    return static_cast<std::uint32_t>(
        std::distance(values.begin(), std::ranges::max_element(values)));
}

void evaluate_steps(
    const std::vector<qwen38::MtpDecodeStep>& steps,
    const std::vector<qwen38::MtpDecodeState>& states) {
    std::vector<qwen38::MlxArray> tokens;
    std::vector<const qwen38::MlxArray*> outputs;
    tokens.reserve(steps.size());
    outputs.reserve(steps.size() * 3);
    for (const qwen38::MtpDecodeStep& step : steps) {
        tokens.push_back(step.logits.argmax_all());
        outputs.push_back(&tokens.back());
    }
    for (const qwen38::MtpDecodeState& state : states) {
        if (state.layer.full_attention.qsa_raw_keys.get().ctx != nullptr) {
            outputs.push_back(&state.layer.full_attention.qsa_raw_keys);
        }
        if (state.layer.full_attention.qsa_pooled_keys.get().ctx != nullptr) {
            outputs.push_back(&state.layer.full_attention.qsa_pooled_keys);
        }
    }
    qwen38::MlxArray::eval_all(outputs);
}

struct BranchPass {
    std::vector<qwen38::MtpDecodeStep> steps;
    std::vector<qwen38::MtpDecodeState> states;
    double milliseconds{0.0};
};

struct TargetBranchPass {
    std::vector<qwen38::TargetDecodeStep> steps;
    std::vector<qwen38::ModelDecodeState> states;
    double milliseconds{0.0};
};

BranchPass run_branch_pass(
    const qwen38::QwenMtpHead& mtp,
    const qwen38::MlxArray& target_stream,
    const std::span<const std::uint32_t> tokens,
    const qwen38::MtpDecodeState& origin,
    const std::size_t query_position,
    const bool batched) {
    BranchPass result;
    result.states.reserve(tokens.size());
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        result.states.push_back(mtp.snapshot_state(origin));
    }
    const auto started = std::chrono::steady_clock::now();
    if (batched) {
        std::vector<const qwen38::MlxArray*> streams(tokens.size(), &target_stream);
        std::vector<qwen38::MtpDecodeState*> states;
        states.reserve(result.states.size());
        for (qwen38::MtpDecodeState& state : result.states) states.push_back(&state);
        result.steps = mtp.forward_decode_multi(
            streams, tokens, query_position, states);
    } else {
        result.steps.reserve(tokens.size());
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            result.steps.push_back(
                mtp.forward_decode(
                    target_stream, tokens[row], query_position, result.states[row]));
        }
    }
    evaluate_steps(result.steps, result.states);
    result.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

TargetBranchPass run_target_branch_pass(
    const qwen38::QwenModel& target,
    const qwen38::ModelDecodeState& origin,
    const std::span<const std::uint32_t> tokens,
    const bool batched) {
    TargetBranchPass result;
    result.states.reserve(tokens.size());
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        result.states.push_back(target.snapshot_state(origin));
    }
    const auto started = std::chrono::steady_clock::now();
    if (batched) {
        std::vector<qwen38::ModelDecodeState*> states;
        states.reserve(result.states.size());
        for (qwen38::ModelDecodeState& state : result.states) states.push_back(&state);
        result.steps = target.forward_decode_capture_multi(tokens, states);
    } else {
        result.steps.reserve(tokens.size());
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            result.steps.push_back(target.forward_decode_capture(tokens[row], result.states[row]));
        }
        std::vector<qwen38::MlxArray> greedy;
        std::vector<const qwen38::MlxArray*> outputs;
        greedy.reserve(result.steps.size());
        for (const qwen38::TargetDecodeStep& step : result.steps) {
            greedy.push_back(step.logits.argmax_all());
            outputs.push_back(&greedy.back());
        }
        qwen38::MlxArray::eval_all(outputs);
    }
    result.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

double maximum_absolute_error(
    const qwen38::MlxArray& left,
    const qwen38::MlxArray& right) {
    const std::vector<float> a = left.astype(MLX_FLOAT32).to_float32();
    const std::vector<float> b = right.astype(MLX_FLOAT32).to_float32();
    if (a.size() != b.size()) throw std::runtime_error("MTP branch parity size mismatch");
    double maximum = 0.0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        maximum = std::max(maximum, static_cast<double>(std::abs(a[index] - b[index])));
    }
    return maximum;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model MTP smoke tests must run through devtools/memory_guard.py");
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel target(tensors);
        qwen38::QwenMtpHead mtp(tensors);
        auto target_state = target.make_state();
        auto mtp_state = mtp.make_state();

        qwen38::TargetDecodeStep target_step =
            target.forward_decode_capture(9419, target_state);
        const std::uint32_t target_token = argmax(target_step.logits);
        qwen38::MtpDecodeStep draft =
            mtp.forward_decode(target_step.pre_mixer_stream, target_token, 1, mtp_state);
        const std::uint32_t draft_token = argmax(draft.logits);
        const std::vector<float> top2 = draft.logits.top2_indices_all().to_float32();
        if (top2.size() != 2) throw std::runtime_error("MTP top-2 size mismatch");
        const std::array<std::uint32_t, 2> branch_tokens{
            static_cast<std::uint32_t>(top2[0]),
            static_cast<std::uint32_t>(top2[1])};
        const std::vector<int> stream_shape = draft.pre_mixer_stream.shape();
        if (stream_shape != std::vector<int>({1, 1, 10240})) {
            throw std::runtime_error("MTP output stream shape mismatch");
        }
        bool rejected_position_gap = false;
        try {
            static_cast<void>(
                mtp.forward_decode(draft.pre_mixer_stream, draft_token, 1, mtp_state));
        } catch (const std::runtime_error&) {
            rejected_position_gap = true;
        }
        if (!rejected_position_gap) {
            throw std::runtime_error("MTP position-gap guard did not engage");
        }

        qwen38::ModelDecodeState sibling_target_origin = target.snapshot_state(target_state);
        qwen38::TargetDecodeStep sibling_prefix =
            target.forward_decode_capture(target_token, sibling_target_origin);
        static_cast<void>(argmax(sibling_prefix.logits));
        static_cast<void>(run_branch_pass(
            mtp, draft.pre_mixer_stream, branch_tokens, mtp_state, 2, false));
        static_cast<void>(run_branch_pass(
            mtp, draft.pre_mixer_stream, branch_tokens, mtp_state, 2, true));
        BranchPass sequential = run_branch_pass(
            mtp, draft.pre_mixer_stream, branch_tokens, mtp_state, 2, false);
        BranchPass batched = run_branch_pass(
            mtp, draft.pre_mixer_stream, branch_tokens, mtp_state, 2, true);
        double branch_error = 0.0;
        for (std::size_t row = 0; row < branch_tokens.size(); ++row) {
            branch_error = std::max(branch_error, maximum_absolute_error(
                sequential.steps[row].logits, batched.steps[row].logits));
        }
        static_cast<void>(run_target_branch_pass(
            target, sibling_target_origin, branch_tokens, false));
        static_cast<void>(run_target_branch_pass(
            target, sibling_target_origin, branch_tokens, true));
        TargetBranchPass target_sequential = run_target_branch_pass(
            target, sibling_target_origin, branch_tokens, false);
        TargetBranchPass target_batched = run_target_branch_pass(
            target, sibling_target_origin, branch_tokens, true);
        double target_branch_error = 0.0;
        for (std::size_t row = 0; row < branch_tokens.size(); ++row) {
            target_branch_error = std::max(target_branch_error, maximum_absolute_error(
                target_sequential.steps[row].logits, target_batched.steps[row].logits));
        }
        const char* gdn_branch_batch = std::getenv("QWEN38_GDN_BRANCH_BATCH");
        if (gdn_branch_batch != nullptr && std::string_view(gdn_branch_batch) == "1" &&
            target_branch_error != 0.0) {
            throw std::runtime_error(
                "exact sibling batching changed target logits: max absolute error " +
                std::to_string(target_branch_error));
        }
        const double branch_sequential_ms = sequential.milliseconds;
        const double branch_batched_ms = batched.milliseconds;
        const double target_branch_sequential_ms = target_sequential.milliseconds;
        const double target_branch_batched_ms = target_batched.milliseconds;
        sequential = {};
        batched = {};
        target_sequential = {};
        target_batched = {};
        const std::array<std::uint32_t, 4> wide_branch_tokens{
            target_token, draft_token, 9419, 1};
        static_cast<void>(run_branch_pass(
            mtp, draft.pre_mixer_stream, wide_branch_tokens, mtp_state, 2, false));
        static_cast<void>(run_branch_pass(
            mtp, draft.pre_mixer_stream, wide_branch_tokens, mtp_state, 2, true));
        BranchPass wide_sequential = run_branch_pass(
            mtp, draft.pre_mixer_stream, wide_branch_tokens, mtp_state, 2, false);
        BranchPass wide_batched = run_branch_pass(
            mtp, draft.pre_mixer_stream, wide_branch_tokens, mtp_state, 2, true);
        double wide_branch_error = 0.0;
        for (std::size_t row = 0; row < wide_branch_tokens.size(); ++row) {
            wide_branch_error = std::max(wide_branch_error, maximum_absolute_error(
                wide_sequential.steps[row].logits, wide_batched.steps[row].logits));
        }
        static_cast<void>(run_target_branch_pass(
            target, sibling_target_origin, wide_branch_tokens, false));
        static_cast<void>(run_target_branch_pass(
            target, sibling_target_origin, wide_branch_tokens, true));
        TargetBranchPass wide_target_sequential = run_target_branch_pass(
            target, sibling_target_origin, wide_branch_tokens, false);
        TargetBranchPass wide_target_batched = run_target_branch_pass(
            target, sibling_target_origin, wide_branch_tokens, true);
        double wide_target_branch_error = 0.0;
        for (std::size_t row = 0; row < wide_branch_tokens.size(); ++row) {
            wide_target_branch_error = std::max(
                wide_target_branch_error, maximum_absolute_error(
                    wide_target_sequential.steps[row].logits,
                    wide_target_batched.steps[row].logits));
        }
        std::cout << "{\"target_token\":" << target_token
                  << ",\"draft_token\":" << draft_token
                  << ",\"mtp_rows\":" << mtp_state.row_count
                  << ",\"position_gap_rejected\":true"
                  << ",\"branch_sequential_ms\":" << branch_sequential_ms
                  << ",\"branch_batched_ms\":" << branch_batched_ms
                  << ",\"branch_speedup\":"
                  << branch_sequential_ms / branch_batched_ms
                  << ",\"branch_max_absolute\":" << branch_error
                  << ",\"target_branch_sequential_ms\":"
                  << target_branch_sequential_ms
                  << ",\"target_branch_batched_ms\":" << target_branch_batched_ms
                  << ",\"target_branch_speedup\":"
                  << target_branch_sequential_ms / target_branch_batched_ms
                  << ",\"target_branch_max_absolute\":" << target_branch_error
                  << ",\"wide_branch_speedup\":"
                  << wide_sequential.milliseconds / wide_batched.milliseconds
                  << ",\"wide_branch_max_absolute\":" << wide_branch_error
                  << ",\"wide_target_branch_speedup\":"
                  << wide_target_sequential.milliseconds /
                      wide_target_batched.milliseconds
                  << ",\"wide_target_branch_max_absolute\":"
                  << wide_target_branch_error
                  << ",\"stream_width\":" << stream_shape.back()
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-head-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
