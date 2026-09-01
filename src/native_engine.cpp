#include "qwen38/native_engine.hpp"
#include "qwen38/mtp_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>

namespace qwen38 {
namespace {

bool has_mtp_weights(const ModelManifest& manifest) {
    return manifest.weight_map().contains("language_model.mtp.fc_embedding.weight");
}

std::size_t resolved_mtp_depth(
    const ModelManifest& manifest,
    const NativeEngineOptions& options) {
    const bool available = has_mtp_weights(manifest);
    const std::size_t depth = options.mtp_depth.value_or(available ? 2 : 0);
    if (depth != 0 && (depth < 2 || depth > 4)) {
        throw std::runtime_error("MTP depth must be 0 or between 2 and 4");
    }
    if (depth != 0 && !available) {
        throw std::runtime_error("MTP was requested but the model has no MTP sidecar");
    }
    return depth;
}

ModelManifest prepare_manifest(
    const std::filesystem::path& model_directory,
    const NativeEngineOptions& options) {
    ModelManifest manifest = ModelManifest::load(model_directory);
    if (resolved_mtp_depth(manifest, options) != 0 &&
        std::getenv("QWEN38_RESIDENT_EXPERT_RANGE") == nullptr) {
        // Verified 64 GiB MTP-safe default. Explicit user policy always wins.
        if (setenv("QWEN38_RESIDENT_EXPERT_RANGE", "12:24", 0) != 0) {
            throw std::runtime_error("cannot set the MTP-safe resident expert range");
        }
    }
    return manifest;
}

std::uint32_t argmax_token(const MlxArray& logits) {
    MlxArray token = logits.argmax_all();
    token.eval();
    return token.item_uint32();
}

} // namespace

NativeEngine::NativeEngine(
    const std::filesystem::path& model_directory,
    NativeEngineOptions options)
    : options_(std::move(options)),
      tensors_(prepare_manifest(model_directory, options_)),
      tokenizer_(Tokenizer::load(model_directory)),
      model_(tensors_),
      mtp_depth_(resolved_mtp_depth(tensors_.manifest(), options_)) {
    if (options_.prefill_chunk_rows == 0 || options_.prefill_chunk_rows > 64) {
        throw std::runtime_error("prefill chunk rows must be between 1 and 64");
    }
    if (mtp_depth_ != 0) {
        if (options_.mtp_cache_limit_bytes == 0) {
            throw std::runtime_error("MTP cache limit must be positive");
        }
        if (options_.zero_accept_fallback_rounds == 0) {
            throw std::runtime_error("MTP fallback window must be positive");
        }
        static_cast<void>(MlxArray::set_cache_limit(options_.mtp_cache_limit_bytes));
        mtp_head_ = std::make_unique<QwenMtpHead>(tensors_);
    }
}

GenerationResult NativeEngine::complete(
    const std::string_view prompt,
    const std::size_t max_tokens) {
    if (max_tokens == 0 || max_tokens > 256) {
        throw std::runtime_error("max_tokens must be between 1 and 256");
    }
    std::scoped_lock lock(inference_mutex_);
    const std::vector<std::uint32_t> prompt_tokens = tokenizer_.encode(prompt);
    if (prompt_tokens.empty()) throw std::runtime_error("prompt produced no tokens");

    ModelDecodeState state = model_.make_state();
    MtpDecodeState mtp_state = mtp_head_ == nullptr
        ? MtpDecodeState{}
        : mtp_head_->make_state();
    std::optional<MlxArray> previous_target_stream;
    const auto prompt_started = std::chrono::steady_clock::now();
    const std::size_t prefill_rows = prompt_tokens.size() - 1;
    for (std::size_t offset = 0; offset < prefill_rows;
         offset += options_.prefill_chunk_rows) {
        const std::size_t count = std::min(
            options_.prefill_chunk_rows, prefill_rows - offset);
        std::vector<MlxArray> streams = model_.prefill_chunk(
            std::span<const std::uint32_t>(prompt_tokens.data() + offset, count), state);
        for (std::size_t row = 0; row < count; ++row) {
            const std::size_t index = offset + row;
            if (mtp_head_ != nullptr && previous_target_stream.has_value()) {
                mtp_head_->consume_decode(
                    *previous_target_stream, prompt_tokens[index], index, mtp_state);
            }
            previous_target_stream = std::move(streams[row]);
        }
    }
    const double prompt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prompt_started).count();

    GenerationResult result;
    result.prompt_tokens = prompt_tokens.size();
    result.prompt_ms = prompt_ms;
    result.tokens.reserve(max_tokens);
    std::uint32_t current = prompt_tokens.back();
    std::size_t zero_accept_streak = 0;
    bool mtp_profitable = mtp_head_ != nullptr;
    const auto generation_started = std::chrono::steady_clock::now();
    while (result.tokens.size() < max_tokens) {
        const std::size_t remaining = max_tokens - result.tokens.size();
        if (!mtp_profitable || !previous_target_stream.has_value() || remaining == 1) {
            TargetDecodeStep step = model_.forward_decode_capture(current, state);
            const std::uint32_t token = argmax_token(step.logits);
            previous_target_stream = std::move(step.pre_mixer_stream);
            result.tokens.push_back(token);
            current = token;
            if (token == tensors_.manifest().config().end_of_sequence_token) {
                result.finish_reason = "stop";
                break;
            }
            continue;
        }

        MtpRoundStep step = run_greedy_mtp_round_reference(
            model_, *mtp_head_, current, *previous_target_stream, state.token_count,
            mtp_depth_, state, mtp_state);
        ++result.mtp_rounds;
        result.mtp_proposed += step.draft_tokens.size();
        result.mtp_accepted += step.accepted;
        zero_accept_streak = step.accepted == 0 ? zero_accept_streak + 1 : 0;
        current = step.next_current_token;
        previous_target_stream = std::move(step.next_target_stream);
        for (const std::uint32_t token : step.emitted_tokens) {
            if (result.tokens.size() == max_tokens) break;
            result.tokens.push_back(token);
            if (token == tensors_.manifest().config().end_of_sequence_token) {
                result.finish_reason = "stop";
                break;
            }
        }
        if (options_.clear_cache_each_mtp_round) MlxArray::clear_cache();
        if (result.finish_reason == "stop") break;
        if (zero_accept_streak >= options_.zero_accept_fallback_rounds) {
            mtp_profitable = false;
            ++result.mtp_fallbacks;
        }
    }
    result.generation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - generation_started).count();
    result.text = tokenizer_.decode(result.tokens);
    return result;
}

void NativeEngine::clear_cache() {
    std::scoped_lock lock(inference_mutex_);
    // Request-owned decode state is released at request completion. This also
    // returns unused MLX allocator/cache blocks to the system on demand.
    MlxArray::clear_cache();
}

} // namespace qwen38
