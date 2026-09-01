#include "qwen38/native_engine.hpp"
#include "qwen38/history_draft.hpp"
#include "qwen38/mtp_depth_policy.hpp"
#include "qwen38/mtp_profitability.hpp"
#include "qwen38/mtp_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
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
    const std::size_t depth = options.mtp_depth.value_or(available ? 3 : 0);
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
        // Verified 64 GiB MTP-balanced default. Explicit user policy always wins.
        if (setenv("QWEN38_RESIDENT_EXPERT_RANGE", "12:28", 0) != 0) {
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

bool is_prefix(
    const std::span<const std::uint32_t> prefix,
    const std::span<const std::uint32_t> tokens) {
    return prefix.size() <= tokens.size() &&
        std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

bool use_long_history_depth_four(const std::size_t token_count) {
    if (token_count < 512) return false;
    const char* enabled = std::getenv("QWEN38_LONG_HISTORY_DEPTH4");
    if (enabled != nullptr && std::string_view(enabled) == "0") return false;
    const char* sdpa_decode = std::getenv("QWEN38_SDPA_DECODE");
    if (sdpa_decode != nullptr && std::string_view(sdpa_decode) != "1") return false;
    const char* batch_verify = std::getenv("QWEN38_BATCH_SDPA_VERIFY");
    return batch_verify == nullptr || std::string_view(batch_verify) != "0";
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
    if (options_.prefill_chunk_rows == 0 || options_.prefill_chunk_rows > 512) {
        throw std::runtime_error("prefill chunk rows must be between 1 and 512");
    }
    const char* metal_prefill = std::getenv("QWEN38_GDN_METAL_PREFILL");
    if (options_.prefill_chunk_rows > 64 &&
        (metal_prefill == nullptr || std::string_view(metal_prefill) != "1")) {
        throw std::runtime_error(
            "prefill chunks above 64 require QWEN38_GDN_METAL_PREFILL=1");
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
    std::optional<bool> cached_mtp_profitability;
    std::size_t prefill_offset = 0;
    const std::size_t prefill_rows = prompt_tokens.size() - 1;
    const std::size_t request_prefill_chunk =
        options_.prefill_chunk_rows > 256 && prefill_rows > 512
        ? (prefill_rows <= 6144 ? 384 : (prefill_rows <= 8192 ? 256 : 128))
        : options_.prefill_chunk_rows;
    if (prefix_cache_ != nullptr &&
        is_prefix(prefix_cache_->tokens,
            std::span<const std::uint32_t>(prompt_tokens.data(), prefill_rows))) {
        state = model_.snapshot_state(prefix_cache_->target_state);
        if (mtp_head_ != nullptr) {
            mtp_state = mtp_head_->snapshot_state(prefix_cache_->mtp_state);
        }
        if (prefix_cache_->previous_target_stream.has_value()) {
            previous_target_stream = prefix_cache_->previous_target_stream->share();
        }
        prefill_offset = prefix_cache_->tokens.size();
        if (prefill_offset == prefill_rows &&
            prefix_cache_->mtp_profitability_current_token == prompt_tokens.back()) {
            cached_mtp_profitability = prefix_cache_->mtp_profitable;
        }
    }
    const auto prompt_started = std::chrono::steady_clock::now();
    for (std::size_t offset = prefill_offset; offset < prefill_rows;
         offset += request_prefill_chunk) {
        const std::size_t count = std::min(
            request_prefill_chunk, prefill_rows - offset);
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
    result.cached_prompt_tokens = prefill_offset;
    result.prompt_ms = prompt_ms;

    if (options_.prefix_cache_max_tokens != 0 && prefill_rows != 0 &&
        prefill_rows <= options_.prefix_cache_max_tokens &&
        (prefix_cache_ == nullptr || prefill_offset != prefill_rows)) {
        prefix_cache_ = std::make_unique<PrefixCacheEntry>(PrefixCacheEntry{
            .tokens = std::vector<std::uint32_t>(
                prompt_tokens.begin(), prompt_tokens.begin() +
                    static_cast<std::ptrdiff_t>(prefill_rows)),
            .target_state = model_.snapshot_state(state),
            .mtp_state = mtp_head_ == nullptr
                ? MtpDecodeState{}
                : mtp_head_->snapshot_state(mtp_state),
            .previous_target_stream = previous_target_stream.has_value()
                ? std::optional<MlxArray>(previous_target_stream->share())
                : std::nullopt,
            .mtp_profitable = std::nullopt,
            .mtp_profitability_current_token = std::nullopt,
        });
    }
    result.tokens.reserve(max_tokens);
    std::uint32_t current = prompt_tokens.back();
    const char* history_draft_environment = std::getenv("QWEN38_HISTORY_DRAFT");
    const bool history_draft_enabled = history_draft_environment == nullptr ||
        std::string_view(history_draft_environment) != "0";
    HistoryDraftCache history_draft;
    if (history_draft_enabled) history_draft.append(prompt_tokens);
    MtpProfitabilityGuard profitability_guard;
    bool mtp_profitable = mtp_head_ != nullptr &&
        cached_mtp_profitability.value_or(true);
    result.mtp_profitability_cache_skip =
        cached_mtp_profitability.has_value() && !*cached_mtp_profitability;
    if (result.mtp_profitability_cache_skip) {
        const char* mtp_trace = std::getenv("QWEN38_MTP_TRACE");
        if (mtp_trace != nullptr && std::string_view(mtp_trace) == "1") {
            std::clog << "qwen38: prefix cache skipped previously losing MTP probe\n";
        }
    }
    MtpDepthPolicy depth_policy(mtp_depth_, prompt_tokens.size());
    result.mtp_final_depth = depth_policy.depth();
    const auto generation_started = std::chrono::steady_clock::now();
    while (result.tokens.size() < max_tokens) {
        const std::size_t remaining = max_tokens - result.tokens.size();
        if (!mtp_profitable || !previous_target_stream.has_value() || remaining == 1) {
            TargetDecodeStep step = model_.forward_decode_capture(current, state);
            const std::uint32_t token = argmax_token(step.logits);
            previous_target_stream = std::move(step.pre_mixer_stream);
            result.tokens.push_back(token);
            if (history_draft_enabled) history_draft.append(token);
            current = token;
            if (token == tensors_.manifest().config().end_of_sequence_token) {
                result.finish_reason = "stop";
                break;
            }
            continue;
        }

        std::vector<std::uint32_t> history_proposal;
        if (history_draft_enabled) {
            const bool try_depth_four = use_long_history_depth_four(state.token_count);
            const std::size_t history_depth = try_depth_four ? 4 : depth_policy.depth();
            history_proposal = history_draft.propose(
                std::min<std::size_t>(history_depth, remaining));
            if (history_proposal.size() < 2 && try_depth_four) {
                history_proposal = history_draft.propose(
                    std::min<std::size_t>(depth_policy.depth(), remaining));
            }
        }
        const bool used_history_draft = history_proposal.size() >= 2;
        MtpRoundStep step = used_history_draft
            ? run_greedy_external_draft_round_reference(
                  model_, *mtp_head_, current, *previous_target_stream,
                  state.token_count, std::move(history_proposal), state, mtp_state)
            : run_greedy_mtp_round_reference(
                  model_, *mtp_head_, current, *previous_target_stream, state.token_count,
                  depth_policy.depth(), state, mtp_state);
        ++result.mtp_rounds;
        result.mtp_proposed += step.draft_tokens.size();
        result.mtp_accepted += step.accepted;
        result.mtp_draft_ms += step.draft_ms;
        result.mtp_verify_ms += step.verify_ms;
        result.mtp_commit_ms += step.commit_ms;
        const char* mtp_trace = std::getenv("QWEN38_MTP_TRACE");
        if (mtp_trace != nullptr && std::string_view(mtp_trace) == "1") {
            std::clog << "qwen38: mtp round=" << result.mtp_rounds
                      << " depth=" << step.draft_tokens.size()
                      << " accepted=" << step.accepted
                      << " draft_ms=" << step.draft_ms
                      << " verify_ms=" << step.verify_ms
                      << " commit_ms=" << step.commit_ms
                      << " history=" << (used_history_draft ? 1 : 0) << '\n';
        }
        if (used_history_draft) {
            ++result.history_draft_rounds;
            result.history_draft_proposed += step.draft_tokens.size();
            result.history_draft_accepted += step.accepted;
        } else {
            depth_policy.observe(step.draft_tokens.size(), step.accepted);
            profitability_guard.observe(step.accepted);
        }
        result.mtp_final_depth = depth_policy.depth();
        result.mtp_promotions = depth_policy.promotions();
        result.mtp_demotions = depth_policy.demotions();
        current = step.next_current_token;
        previous_target_stream = std::move(step.next_target_stream);
        for (const std::uint32_t token : step.emitted_tokens) {
            if (result.tokens.size() == max_tokens) break;
            result.tokens.push_back(token);
            if (history_draft_enabled) history_draft.append(token);
            if (token == tensors_.manifest().config().end_of_sequence_token) {
                result.finish_reason = "stop";
                break;
            }
        }
        if (options_.clear_cache_each_mtp_round) MlxArray::clear_cache();
        if (result.finish_reason == "stop") break;
        const char* economic_fallback = std::getenv("QWEN38_ECONOMIC_MTP_FALLBACK");
        const bool economic_fallback_enabled = economic_fallback == nullptr ||
            std::string_view(economic_fallback) != "0";
        const bool should_fallback = !used_history_draft && (economic_fallback_enabled
            ? profitability_guard.should_fallback(options_.zero_accept_fallback_rounds)
            : profitability_guard.zero_accept_streak() >=
                  options_.zero_accept_fallback_rounds);
        if (should_fallback) {
            mtp_profitable = false;
            ++result.mtp_fallbacks;
        }
    }
    result.generation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - generation_started).count();
    if (prefix_cache_ != nullptr && prefill_rows == prefix_cache_->tokens.size() &&
        std::equal(prefix_cache_->tokens.begin(), prefix_cache_->tokens.end(),
            prompt_tokens.begin()) && result.mtp_rounds >= 2) {
        prefix_cache_->mtp_profitable = result.mtp_fallbacks == 0;
        prefix_cache_->mtp_profitability_current_token = prompt_tokens.back();
    }
    result.text = tokenizer_.decode(result.tokens);
    return result;
}

void NativeEngine::clear_cache() {
    std::scoped_lock lock(inference_mutex_);
    // Request-owned decode state is released at request completion. This also
    // returns unused MLX allocator/cache blocks to the system on demand.
    prefix_cache_.reset();
    MlxArray::clear_cache();
}

} // namespace qwen38
