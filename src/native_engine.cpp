#include "qwen38/native_engine.hpp"
#include "qwen38/history_draft.hpp"
#include "qwen38/mtp_depth_policy.hpp"
#include "qwen38/mtp_profitability.hpp"
#include "qwen38/mtp_runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <thread>
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

bool mtp_state_matches_target(
    const MtpDecodeState& mtp_state,
    const std::size_t target_token_count) {
    return mtp_state.position_base.has_value() &&
        *mtp_state.position_base <= target_token_count &&
        mtp_state.row_count == target_token_count - *mtp_state.position_base;
}

void consume_pending_mtp_rows(
    const QwenMtpHead& head,
    const std::vector<MlxArray>& streams,
    const std::vector<std::uint32_t>& tokens,
    MtpDecodeState& state) {
    if (streams.size() != tokens.size()) {
        throw std::runtime_error("cached MTP rows must have matching streams and tokens");
    }
    if (!tokens.empty() && !state.position_base.has_value()) {
        throw std::runtime_error("cached MTP rows require an initialized position base");
    }
    std::size_t offset = 0;
    while (offset < tokens.size()) {
        const std::size_t remaining = tokens.size() - offset;
        if (remaining == 1) {
            const std::size_t position = *state.position_base + state.row_count;
            head.consume_decode(streams[offset], tokens[offset], position, state);
            ++offset;
            continue;
        }
        const std::size_t count = std::min<std::size_t>(remaining, 5);
        std::vector<const MlxArray*> rows;
        rows.reserve(count);
        for (std::size_t row = 0; row < count; ++row) {
            rows.push_back(&streams[offset + row]);
        }
        const std::size_t position = *state.position_base + state.row_count;
        head.consume_committed_batch(
            rows,
            std::span<const std::uint32_t>(tokens.data() + offset, count),
            position,
            state);
        offset += count;
    }
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

std::size_t complete_utf8_prefix(const std::string_view text) {
    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto lead = static_cast<unsigned char>(text[offset]);
        std::size_t width = 1;
        if ((lead & 0x80U) == 0) {
            width = 1;
        } else if (lead >= 0xC2U && lead <= 0xDFU) {
            width = 2;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            width = 3;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            width = 4;
        }
        if (offset + width > text.size()) break;
        bool continuation = true;
        for (std::size_t index = 1; index < width; ++index) {
            const auto byte = static_cast<unsigned char>(text[offset + index]);
            continuation = continuation && (byte & 0xC0U) == 0x80U;
        }
        if (!continuation) width = 1;
        offset += width;
    }
    return offset;
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
    const std::vector<std::uint32_t> chat_end = tokenizer_.encode("<|im_end|>");
    if (chat_end.size() != 1) {
        throw std::runtime_error("chat end marker must encode to one token");
    }
    chat_end_token_ = chat_end.front();
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
    return complete_impl(prompt, max_tokens, nullptr);
}

GenerationResult NativeEngine::complete_stream(
    const std::string_view prompt,
    const std::size_t max_tokens,
    const TextDeltaCallback& on_delta) {
    return complete_impl(prompt, max_tokens, &on_delta);
}

GenerationResult NativeEngine::complete_impl(
    const std::string_view prompt,
    const std::size_t max_tokens,
    const TextDeltaCallback* on_delta) {
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
    const auto prompt_started = std::chrono::steady_clock::now();
    if (prefix_cache_ != nullptr &&
        is_prefix(prefix_cache_->tokens,
            std::span<const std::uint32_t>(prompt_tokens.data(), prefill_rows))) {
        state = model_.snapshot_state(prefix_cache_->target_state);
        if (mtp_head_ != nullptr) {
            mtp_state = mtp_head_->snapshot_state(prefix_cache_->mtp_state);
            consume_pending_mtp_rows(
                *mtp_head_,
                prefix_cache_->pending_mtp_streams,
                prefix_cache_->pending_mtp_tokens,
                mtp_state);
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
    std::size_t streamed_tokens = 0;
    std::string pending_stream_bytes;
    const auto emit_new_tokens = [&] {
        if (on_delta == nullptr || streamed_tokens == result.tokens.size()) return;
        pending_stream_bytes += tokenizer_.decode(std::span<const std::uint32_t>(
            result.tokens.data() + streamed_tokens,
            result.tokens.size() - streamed_tokens));
        streamed_tokens = result.tokens.size();
        const std::size_t complete = complete_utf8_prefix(pending_stream_bytes);
        if (complete != 0) {
            (*on_delta)(std::string_view(pending_stream_bytes).substr(0, complete));
            pending_stream_bytes.erase(0, complete);
        }
    };

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
            .pending_mtp_streams = {},
            .pending_mtp_tokens = {},
            .mtp_profitable = std::nullopt,
            .mtp_profitability_current_token = std::nullopt,
        });
    }
    result.tokens.reserve(max_tokens);
    const char* extend_cache = std::getenv("QWEN38_EXTEND_PREFIX_CACHE");
    const bool extend_cache_enabled =
        extend_cache != nullptr && std::string_view(extend_cache) == "1";
    bool mtp_cache_extendable = true;
    std::vector<MlxArray> pending_mtp_streams;
    std::vector<std::uint32_t> pending_mtp_tokens;
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
    bool stopped_on_terminator = false;
    const auto is_stop_token = [&](const std::uint32_t token) {
        return token == tensors_.manifest().config().end_of_sequence_token ||
            token == chat_end_token_;
    };
    const std::array<std::uint32_t, 2> stop_tokens{
        tensors_.manifest().config().end_of_sequence_token, chat_end_token_};
    const auto generation_started = std::chrono::steady_clock::now();
    while (result.tokens.size() < max_tokens) {
        const std::size_t remaining = max_tokens - result.tokens.size();
        if (!mtp_profitable || !previous_target_stream.has_value() ||
            remaining == 1 || (extend_cache_enabled && remaining == 2)) {
            if (mtp_head_ != nullptr) {
                if (extend_cache_enabled && mtp_profitable &&
                    previous_target_stream.has_value() &&
                    mtp_state_matches_target(mtp_state, state.token_count)) {
                    mtp_head_->consume_decode(
                        *previous_target_stream, current, state.token_count, mtp_state);
                } else if (extend_cache_enabled && previous_target_stream.has_value() &&
                    ((pending_mtp_streams.empty() &&
                         mtp_state_matches_target(mtp_state, state.token_count)) ||
                     (!pending_mtp_streams.empty() && mtp_state.position_base.has_value() &&
                         *mtp_state.position_base + mtp_state.row_count +
                             pending_mtp_streams.size() == state.token_count))) {
                    pending_mtp_streams.push_back(previous_target_stream->share());
                    pending_mtp_tokens.push_back(current);
                } else {
                    mtp_cache_extendable = false;
                }
            }
            TargetDecodeStep step = model_.forward_decode_capture(current, state);
            const std::uint32_t token = argmax_token(step.logits);
            previous_target_stream = std::move(step.pre_mixer_stream);
            current = token;
            if (is_stop_token(token)) {
                result.finish_reason = "stop";
                stopped_on_terminator = true;
                break;
            }
            result.tokens.push_back(token);
            emit_new_tokens();
            if (history_draft_enabled) history_draft.append(token);
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
                  state.token_count, std::move(history_proposal), state, mtp_state,
                  stop_tokens)
            : run_greedy_mtp_round_reference(
                  model_, *mtp_head_, current, *previous_target_stream, state.token_count,
                  depth_policy.depth(), state, mtp_state, stop_tokens);
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
            if (is_stop_token(token)) {
                result.finish_reason = "stop";
                stopped_on_terminator = true;
                break;
            }
            result.tokens.push_back(token);
            emit_new_tokens();
            if (history_draft_enabled) history_draft.append(token);
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
    const std::size_t complete_token_count = prompt_tokens.size() + result.tokens.size();
    const bool target_matches_output = stopped_on_terminator
        ? state.token_count == complete_token_count
        : (state.token_count < complete_token_count &&
              state.token_count + 1 == complete_token_count);
    const bool pending_mtp_matches_output = mtp_state.position_base.has_value() &&
        *mtp_state.position_base + mtp_state.row_count + pending_mtp_streams.size() ==
            state.token_count;
    const bool mtp_matches_output = mtp_head_ == nullptr || (mtp_cache_extendable &&
        (mtp_state_matches_target(mtp_state, state.token_count) ||
            pending_mtp_matches_output));
    if (extend_cache_enabled && options_.prefix_cache_max_tokens != 0 &&
        target_matches_output && mtp_matches_output && previous_target_stream.has_value() &&
        state.token_count <= options_.prefix_cache_max_tokens) {
        std::vector<std::uint32_t> consumed_tokens;
        consumed_tokens.reserve(state.token_count);
        consumed_tokens.insert(
            consumed_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
        consumed_tokens.insert(
            consumed_tokens.end(), result.tokens.begin(), result.tokens.end());
        consumed_tokens.resize(state.token_count);
        const std::optional<bool> extended_profitability = result.mtp_rounds >= 2
            ? std::optional<bool>(result.mtp_fallbacks == 0)
            : std::nullopt;
        prefix_cache_ = std::make_unique<PrefixCacheEntry>(PrefixCacheEntry{
            .tokens = std::move(consumed_tokens),
            .target_state = model_.snapshot_state(state),
            .mtp_state = mtp_head_ == nullptr
                ? MtpDecodeState{}
                : mtp_head_->snapshot_state(mtp_state),
            .previous_target_stream = previous_target_stream->share(),
            .pending_mtp_streams = std::move(pending_mtp_streams),
            .pending_mtp_tokens = std::move(pending_mtp_tokens),
            .mtp_profitable = extended_profitability,
            .mtp_profitability_current_token = extended_profitability.has_value()
                ? std::optional<std::uint32_t>(current)
                : std::nullopt,
        });
    }
    if (on_delta != nullptr && !pending_stream_bytes.empty()) {
        (*on_delta)(pending_stream_bytes);
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

struct NativeEngineExecutor::Impl final {
    using Task = std::function<void(NativeEngine&)>;

    ~Impl() { stop(); }

    void start(
        const std::filesystem::path& model_directory,
        NativeEngineOptions options) {
        auto started = std::make_shared<std::promise<void>>();
        std::future<void> ready = started->get_future();
        worker = std::thread([
            this, model_directory, options = std::move(options), started]() mutable {
            try {
                engine = std::make_unique<NativeEngine>(
                    model_directory, std::move(options));
                started->set_value();
            } catch (...) {
                started->set_exception(std::current_exception());
                return;
            }

            while (true) {
                Task task;
                {
                    std::unique_lock lock(mutex);
                    ready_for_work.wait(lock, [this] {
                        return stopping || !tasks.empty();
                    });
                    if (tasks.empty()) {
                        if (stopping) break;
                        continue;
                    }
                    task = std::move(tasks.front());
                    tasks.pop_front();
                }
                task(*engine);
            }
            engine.reset();
        });
        try {
            ready.get();
        } catch (...) {
            if (worker.joinable()) worker.join();
            throw;
        }
    }

    void enqueue(Task task) {
        {
            std::scoped_lock lock(mutex);
            if (stopping) throw std::runtime_error("inference executor is stopping");
            tasks.push_back(std::move(task));
        }
        ready_for_work.notify_one();
    }

    GenerationResult generate(
        std::string prompt,
        const std::size_t max_tokens,
        std::optional<TextDeltaCallback> on_delta) {
        auto result = std::make_shared<std::promise<GenerationResult>>();
        std::future<GenerationResult> future = result->get_future();
        enqueue([prompt = std::move(prompt), max_tokens,
                    on_delta = std::move(on_delta), result](NativeEngine& native) mutable {
            try {
                if (on_delta.has_value()) {
                    result->set_value(native.complete_stream(
                        prompt, max_tokens, *on_delta));
                } else {
                    result->set_value(native.complete(prompt, max_tokens));
                }
            } catch (...) {
                result->set_exception(std::current_exception());
            }
        });
        return future.get();
    }

    void clear() {
        auto result = std::make_shared<std::promise<void>>();
        std::future<void> future = result->get_future();
        enqueue([result](NativeEngine& native) {
            try {
                native.clear_cache();
                result->set_value();
            } catch (...) {
                result->set_exception(std::current_exception());
            }
        });
        future.get();
    }

    void stop() noexcept {
        {
            std::scoped_lock lock(mutex);
            stopping = true;
        }
        ready_for_work.notify_all();
        if (worker.joinable()) worker.join();
    }

    std::mutex mutex;
    std::condition_variable ready_for_work;
    std::deque<Task> tasks;
    std::thread worker;
    std::unique_ptr<NativeEngine> engine;
    bool stopping{false};
};

NativeEngineExecutor::NativeEngineExecutor(
    const std::filesystem::path& model_directory,
    NativeEngineOptions options)
    : impl_(std::make_unique<Impl>()) {
    impl_->start(model_directory, std::move(options));
}

NativeEngineExecutor::~NativeEngineExecutor() = default;

GenerationResult NativeEngineExecutor::complete(
    const std::string_view prompt,
    const std::size_t max_tokens) {
    return impl_->generate(std::string(prompt), max_tokens, std::nullopt);
}

GenerationResult NativeEngineExecutor::complete_stream(
    const std::string_view prompt,
    const std::size_t max_tokens,
    const TextDeltaCallback& on_delta) {
    return impl_->generate(std::string(prompt), max_tokens, on_delta);
}

void NativeEngineExecutor::clear_cache() {
    impl_->clear();
}

} // namespace qwen38
