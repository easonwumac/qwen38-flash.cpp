#include "qwen38/native_engine.hpp"
#include "qwen38/runtime_profile.hpp"
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
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <utility>
#include <pwd.h>
#include <unistd.h>

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

std::uint32_t argmax_token(const MlxArray& logits, const ModelDecodeState& state) {
    MlxArray token = logits.argmax_all();
    eval_with_decode_state(token, state);
    return token.item_uint32();
}

bool is_prefix(
    const std::span<const std::uint32_t> prefix,
    const std::span<const std::uint32_t> tokens) {
    return prefix.size() <= tokens.size() &&
        std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

MlxArray slice_sequence_rows(
    const MlxArray& batch,
    const std::size_t begin,
    const std::size_t end) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() != 3 || begin >= end || end > static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("invalid prefill stream slice");
    }
    return batch.slice(
        std::vector<int>{0, static_cast<int>(begin), 0},
        std::vector<int>{1, static_cast<int>(end), shape[2]},
        std::vector<int>{1, 1, 1});
}

bool batched_mtp_prefill_enabled() {
    const char* value = std::getenv("QWEN38_BATCH_MTP_PREFILL");
    return value == nullptr || std::string_view(value) != "0";
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

bool cache_completed_mtp_as_profitable(
    const std::size_t rounds,
    const std::size_t accepted,
    const std::size_t fallbacks) {
    const char* cumulative =
        std::getenv("QWEN38_MTP_CUMULATIVE_PROFITABILITY_CACHE");
    if (cumulative == nullptr || std::string_view(cumulative) != "1") {
        return fallbacks == 0;
    }
    return cache_mtp_as_profitable(rounds, accepted, fallbacks);
}

std::uint64_t fnv1a(const std::string_view text) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const char character : text) {
        const auto byte = static_cast<unsigned char>(character);
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string cache_compatibility_key(
    const ModelManifest& manifest,
    const std::size_t mtp_depth) {
    std::ostringstream source;
    std::error_code ignored;
    source << std::filesystem::weakly_canonical(manifest.directory(), ignored).string()
           << '|' << manifest.declared_weight_bytes()
           << '|' << manifest.config().layer_count
           << '|' << manifest.config().quantization_bits
           << '|' << mtp_depth;
    for (const char* file : {"config.json", "model.safetensors.index.json"}) {
        const std::filesystem::path path = manifest.directory() / file;
        std::uintmax_t bytes = std::filesystem::file_size(path, ignored);
        if (ignored) bytes = 0;
        ignored.clear();
        const auto modified = std::filesystem::last_write_time(path, ignored);
        const auto ticks = ignored ? 0LL : std::chrono::duration_cast<std::chrono::nanoseconds>(
            modified.time_since_epoch()).count();
        source << '|' << bytes << '|' << ticks;
        ignored.clear();
    }
    constexpr std::array<const char*, 10> state_environment{
        "QWEN38_COMPACT_QMETA",
        "QWEN38_FUSED_MOE",
        "QWEN38_DEVICE_ROUTER",
        "QWEN38_SELECTED_SOFTMAX_ROUTER",
        "QWEN38_GDN_METAL_PREFILL",
        "QWEN38_SDPA_PREFILL",
        "QWEN38_QSA_PREFILL",
        "QWEN38_HC_FUSED_INJECTION",
        "QWEN38_COMPILE_LAYER",
        "QWEN38_SDPA_DECODE",
    };
    for (const char* name : state_environment) {
        const char* value = std::getenv(name);
        source << '|' << name << '=' << (value == nullptr ? "" : value);
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << fnv1a(source.str());
    return result.str();
}

std::filesystem::path default_cache_directory(
    const ModelManifest& manifest,
    const std::size_t mtp_depth) {
    std::filesystem::path root;
    if (const char* configured = std::getenv("XDG_CACHE_HOME");
        configured != nullptr && *configured != '\0') {
        root = configured;
    } else if (const passwd* account = getpwuid(getuid());
               account != nullptr && account->pw_dir != nullptr) {
        root = std::filesystem::path(account->pw_dir) / "Library" / "Caches";
    } else {
        root = std::filesystem::temp_directory_path();
    }
    return root / "qwen38-flash.cpp" / "prefix" /
        cache_compatibility_key(manifest, mtp_depth);
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
    if (options_.max_generation_tokens == 0) {
        throw std::runtime_error("generation token limit must be positive");
    }
    if (options_.prefill_chunk_rows == 0 || options_.prefill_chunk_rows > 1024) {
        throw std::runtime_error("prefill chunk rows must be between 1 and 1024");
    }
    const char* metal_prefill = std::getenv("QWEN38_GDN_METAL_PREFILL");
    if (options_.prefill_chunk_rows > 64 &&
        (metal_prefill == nullptr || std::string_view(metal_prefill) != "1")) {
        throw std::runtime_error(
            "prefill chunks above 64 require QWEN38_GDN_METAL_PREFILL=1");
    }
    if (options_.allocator_cache_limit_bytes == 0) {
        throw std::runtime_error("allocator cache limit must be positive");
    }
    static_cast<void>(MlxArray::set_cache_limit(
        options_.allocator_cache_limit_bytes));
    if (mtp_depth_ != 0) {
        if (options_.zero_accept_fallback_rounds == 0) {
            throw std::runtime_error("MTP fallback window must be positive");
        }
        mtp_head_ = std::make_unique<QwenMtpHead>(tensors_);
    }
    if (options_.ssd_prefix_cache_max_bytes != 0) {
        const std::filesystem::path directory =
            options_.ssd_prefix_cache_directory.empty()
            ? default_cache_directory(tensors_.manifest(), mtp_depth_)
            : options_.ssd_prefix_cache_directory /
                cache_compatibility_key(tensors_.manifest(), mtp_depth_);
        ssd_prefix_cache_ = std::make_unique<PrefixCacheStore>(
            directory, options_.ssd_prefix_cache_max_bytes, model_.layer_count());
        std::clog << "qwen38-server: SSD prefix cache=" << directory
                  << " limit_bytes=" << options_.ssd_prefix_cache_max_bytes << '\n';
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
    if (max_tokens == 0 || max_tokens > options_.max_generation_tokens) {
        throw std::runtime_error(
            "max_tokens must be between 1 and the configured generation limit (" +
            std::to_string(options_.max_generation_tokens) + ")");
    }
    std::scoped_lock lock(inference_mutex_);
    begin_mtp_calibration_request();
    const std::vector<std::uint32_t> prompt_tokens = tokenizer_.encode(prompt);
    if (prompt_tokens.empty()) throw std::runtime_error("prompt produced no tokens");
    const std::size_t context_limit = tensors_.manifest().config().max_context_tokens;
    if (prompt_tokens.size() > context_limit ||
        max_tokens > context_limit - prompt_tokens.size()) {
        throw std::runtime_error(
            "prompt and max_tokens exceed the model context limit (" +
            std::to_string(context_limit) + " tokens)");
    }

    ModelDecodeState state = model_.make_state();
    MtpDecodeState mtp_state = mtp_head_ == nullptr
        ? MtpDecodeState{}
        : mtp_head_->make_state();
    std::optional<MlxArray> previous_target_stream;
    std::optional<bool> cached_mtp_profitability;
    bool cached_mtp_cumulative_keep = false;
    std::size_t prefill_offset = 0;
    const std::size_t prefill_rows = prompt_tokens.size() - 1;
    const std::size_t request_prefill_chunk = options_.adaptive_prefill_chunks
        ? qwen38::select_prefill_chunk_rows(options_.prefill_chunk_rows, prefill_rows)
        : options_.prefill_chunk_rows;
    model_.clear_prefill_qmeta_cache();
    model_.set_prefill_qmeta_cache_allowed(
        options_.qmeta_cache_max_prompt_tokens != 0 &&
        prefill_rows <= options_.qmeta_cache_max_prompt_tokens);
    const auto prompt_started = std::chrono::steady_clock::now();
    const std::span<const std::uint32_t> prefill_tokens(
        prompt_tokens.data(), prefill_rows);
    const char* profile_prefill = std::getenv("QWEN38_PROFILE_PREFILL");
    const bool profile_prefill_enabled =
        profile_prefill != nullptr && std::string_view(profile_prefill) == "1";
    std::vector<double> prefill_layer_ms;
    if (ssd_prefix_cache_ != nullptr &&
        (prefix_cache_ == nullptr || !is_prefix(prefix_cache_->tokens, prefill_tokens))) {
        // The SSD store is the fallback owner for nonmatching prefixes. Drop the
        // stale RAM snapshot before loading or running a full prefill so its
        // shared backing allocations do not overlap the replacement state.
        prefix_cache_.reset();
        std::optional<StoredPrefixState> stored =
            ssd_prefix_cache_->load_longest(prefill_tokens);
        if (stored.has_value()) {
            PersistedPrefixState& persistent = stored->state;
            prefix_cache_ = std::make_unique<PrefixCacheEntry>(PrefixCacheEntry{
                .tokens = std::move(stored->tokens),
                .target_state = std::move(persistent.target),
                .mtp_state = std::move(persistent.mtp),
                .previous_target_stream = std::move(persistent.previous_target_stream),
                .pending_mtp_streams = std::move(persistent.pending_mtp_streams),
                .pending_mtp_tokens = std::move(persistent.pending_mtp_tokens),
                .mtp_profitable = persistent.mtp_profitable,
                .mtp_profitability_current_token =
                    persistent.mtp_profitability_current_token,
                .mtp_cumulative_profitability_keep =
                    persistent.mtp_cumulative_profitability_keep,
            });
        }
    }
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
            cached_mtp_cumulative_keep =
                prefix_cache_->mtp_cumulative_profitability_keep;
        }
    }
    const auto consume_target_batch = [&](
        const MlxArray& stream_batch,
        const std::size_t offset,
        const std::size_t count) {
        if (mtp_head_ != nullptr && batched_mtp_prefill_enabled()) {
            const std::size_t first_token = previous_target_stream.has_value() ? offset : offset + 1;
            if (first_token < offset + count) {
                const std::size_t rows = offset + count - first_token;
                MlxArray target_rows = previous_target_stream.has_value()
                    ? (rows == 1
                        ? previous_target_stream->share()
                        : MlxArray::concatenate(
                              *previous_target_stream,
                              slice_sequence_rows(stream_batch, 0, rows - 1),
                              1))
                    : slice_sequence_rows(stream_batch, 0, rows);
                mtp_head_->consume_prefill_batch(
                    target_rows,
                    std::span<const std::uint32_t>(prompt_tokens.data() + first_token, rows),
                    first_token,
                    mtp_state);
            }
            previous_target_stream = slice_sequence_rows(stream_batch, count - 1, count);
        } else {
            for (std::size_t row = 0; row < count; ++row) {
                const std::size_t index = offset + row;
                if (mtp_head_ != nullptr && previous_target_stream.has_value()) {
                    mtp_head_->consume_decode(
                        *previous_target_stream, prompt_tokens[index], index, mtp_state);
                }
                previous_target_stream = slice_sequence_rows(stream_batch, row, row + 1);
            }
        }
    };
    for (std::size_t offset = prefill_offset; offset < prefill_rows;
         offset += request_prefill_chunk) {
        const std::size_t count = std::min(
            request_prefill_chunk, prefill_rows - offset);
        MlxArray stream_batch = model_.prefill_chunk_batch(
            std::span<const std::uint32_t>(prompt_tokens.data() + offset, count), state,
            profile_prefill_enabled ? &prefill_layer_ms : nullptr);
        for (std::size_t batch_offset = 0; batch_offset < count; batch_offset += 512) {
            const std::size_t batch_count = std::min<std::size_t>(512, count - batch_offset);
            MlxArray batch = batch_offset == 0 && batch_count == count
                ? stream_batch.share()
                : slice_sequence_rows(
                      stream_batch, batch_offset, batch_offset + batch_count);
            consume_target_batch(batch, offset + batch_offset, batch_count);
        }
    }
    model_.clear_prefill_qmeta_cache();
    const double prompt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prompt_started).count();
    if (!prefill_layer_ms.empty()) {
        double linear_ms = 0.0;
        double full_ms = 0.0;
        for (std::size_t layer = 0; layer < prefill_layer_ms.size(); ++layer) {
            (layer + 1) % 4 == 0
                ? full_ms += prefill_layer_ms[layer]
                : linear_ms += prefill_layer_ms[layer];
        }
        const auto slowest = std::ranges::max_element(prefill_layer_ms);
        std::clog << "qwen38-prefill-profile: tokens=" << prefill_rows
                  << " chunk=" << request_prefill_chunk
                  << " total_ms=" << prompt_ms
                  << " linear_layers_ms=" << linear_ms
                  << " full_layers_ms=" << full_ms
                  << " slowest_layer="
                  << std::distance(prefill_layer_ms.begin(), slowest)
                  << " slowest_layer_ms=" << *slowest << '\n';
    }

    GenerationResult result;
    result.prompt_tokens = prompt_tokens.size();
    result.cached_prompt_tokens = prefill_offset;
    result.prompt_ms = prompt_ms;
    bool prefix_cache_changed = false;
    std::vector<std::uint32_t> prompt_cache_tokens;
    std::optional<PersistedPrefixState> prompt_cache_state;
    std::size_t streamed_tokens = 0;
    std::string pending_stream_bytes;
    bool stream_connected = true;
    const auto emit_new_tokens = [&] {
        if (on_delta == nullptr || streamed_tokens == result.tokens.size()) return true;
        if (!stream_connected) {
            streamed_tokens = result.tokens.size();
            pending_stream_bytes.clear();
            return false;
        }
        pending_stream_bytes += tokenizer_.decode(std::span<const std::uint32_t>(
            result.tokens.data() + streamed_tokens,
            result.tokens.size() - streamed_tokens));
        streamed_tokens = result.tokens.size();
        const std::size_t complete = complete_utf8_prefix(pending_stream_bytes);
        if (complete != 0) {
            stream_connected = (*on_delta)(
                std::string_view(pending_stream_bytes).substr(0, complete));
            pending_stream_bytes.erase(0, complete);
        }
        return stream_connected;
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
            .mtp_cumulative_profitability_keep = false,
        });
        prefix_cache_changed = true;
        if (ssd_prefix_cache_ != nullptr) {
            prompt_cache_tokens = prefix_cache_->tokens;
            prompt_cache_state.emplace(snapshot_prefix_cache(*prefix_cache_));
        }
    }
    result.tokens.reserve(max_tokens);
    const char* extend_cache = std::getenv("QWEN38_EXTEND_PREFIX_CACHE");
    const bool extend_cache_enabled =
        ssd_prefix_cache_ != nullptr ||
        (extend_cache != nullptr && std::string_view(extend_cache) == "1");
    bool mtp_cache_extendable = true;
    std::vector<MlxArray> pending_mtp_streams;
    std::vector<std::uint32_t> pending_mtp_tokens;
    std::uint32_t current = prompt_tokens.back();
    const char* history_draft_environment = std::getenv("QWEN38_HISTORY_DRAFT");
    const HistoryDraftMode history_draft_mode = history_draft_environment == nullptr
        ? HistoryDraftMode::adaptive
        : (std::string_view(history_draft_environment) == "0"
                  ? HistoryDraftMode::disabled
                  : HistoryDraftMode::forced);
    HistoryDraftPolicy history_draft_policy(history_draft_mode);
    const bool history_draft_enabled = history_draft_policy.enabled();
    HistoryDraftCache history_draft;
    if (history_draft_enabled) history_draft.append(prompt_tokens);
    const char* context_copy_environment = std::getenv("QWEN38_CONTEXT_COPY");
    const bool context_copy_enabled = context_copy_environment != nullptr &&
        std::string_view(context_copy_environment) == "1" && mtp_head_ != nullptr;
    std::size_t context_copy_max_tokens = 16;
    if (const char* value = std::getenv("QWEN38_CONTEXT_COPY_MAX_TOKENS")) {
        try {
            std::size_t parsed = 0;
            const unsigned long long requested = std::stoull(value, &parsed);
            if (value[parsed] != '\0' || requested < 4 || requested > 24) {
                throw std::runtime_error("out of range");
            }
            context_copy_max_tokens = std::clamp<std::size_t>(
                requested, 4, 24);
        } catch (const std::exception&) {
            throw std::runtime_error(
                "QWEN38_CONTEXT_COPY_MAX_TOKENS must be an integer from 4 to 24");
        }
    }
    std::optional<ContextCopyCache> context_copy;
    if (context_copy_enabled) context_copy.emplace(prompt_tokens);
    std::size_t context_copy_seen = 0;
    std::size_t context_copy_perfect_rounds = 0;
    std::size_t context_copy_suspend_until = 0;
    std::size_t context_copy_backoff = 64;
    double context_copy_acceptance_ema = 0.5;
    MtpProfitabilityGuard profitability_guard;
    bool mtp_profitable = mtp_head_ != nullptr &&
        cached_mtp_profitability.value_or(true);
    result.mtp_profitability_cache_skip =
        cached_mtp_profitability.has_value() && !*cached_mtp_profitability;
    result.mtp_profitability_cache_keep =
        cached_mtp_profitability.value_or(false) && cached_mtp_cumulative_keep;
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
            const std::uint32_t token = argmax_token(step.logits, state);
            previous_target_stream = std::move(step.pre_mixer_stream);
            current = token;
            if (is_stop_token(token)) {
                result.finish_reason = "stop";
                stopped_on_terminator = true;
                break;
            }
            result.tokens.push_back(token);
            if (!emit_new_tokens()) {
                result.finish_reason = "cancelled";
                break;
            }
            if (history_draft_enabled) history_draft.append(token);
            continue;
        }

        std::vector<std::uint32_t> context_copy_proposal;
        std::size_t context_copy_match_extension = 0;
        if (context_copy.has_value() &&
            result.tokens.size() >= context_copy_suspend_until) {
            const std::size_t cap = context_copy_perfect_rounds >= 2 &&
                    context_copy_acceptance_ema >= 0.75
                ? context_copy_max_tokens
                : 4;
            ContextCopyProposal proposal = context_copy->propose_completion(
                result.tokens, std::min<std::size_t>(cap, remaining));
            context_copy_match_extension = proposal.match_extension;
            static constexpr std::array<std::size_t, 5> block_ladder{8, 12, 16, 24, 24};
            const std::size_t block = std::min(
                cap, block_ladder[std::min(
                    proposal.match_extension, block_ladder.size() - 1)]);
            if (proposal.tokens.size() > block) proposal.tokens.resize(block);
            // A bare six-token suffix is common enough to perturb otherwise
            // strong learned MTP. Require two additional tokens of exact
            // left context before spending a target verifier round on it.
            if (proposal.match_extension >= 2) {
                context_copy_proposal = std::move(proposal.tokens);
            }
        }
        const bool used_context_copy = context_copy_proposal.size() >= 2;
        std::vector<std::uint32_t> history_proposal;
        if (!used_context_copy && history_draft_policy.should_try()) {
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
        MtpRoundStep step = used_context_copy
            ? run_greedy_external_draft_round_reference(
                  model_, *mtp_head_, current, *previous_target_stream,
                  state.token_count, std::move(context_copy_proposal), state, mtp_state,
                  stop_tokens)
            : used_history_draft
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
        for (std::size_t position = 0;
             position < std::min(step.draft_tokens.size(), result.mtp_proposed_by_position.size());
             ++position) {
            ++result.mtp_proposed_by_position[position];
            if (position < step.accepted) ++result.mtp_accepted_by_position[position];
            result.mtp_top2_rejected_by_position[position] +=
                step.top2_rejected_by_position[position];
            result.mtp_top2_recovered_by_position[position] +=
                step.top2_recovered_by_position[position];
        }
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
                      << " history=" << (used_history_draft ? 1 : 0)
                      << " context_copy=" << (used_context_copy ? 1 : 0)
                      << " context_extension=" << context_copy_match_extension << '\n';
        }
        if (used_context_copy) {
            ++result.context_copy_rounds;
            result.context_copy_proposed += step.draft_tokens.size();
            result.context_copy_accepted += step.accepted;
            const double ratio = static_cast<double>(step.accepted) /
                static_cast<double>(step.draft_tokens.size());
            context_copy_acceptance_ema =
                0.7 * context_copy_acceptance_ema + 0.3 * ratio;
            ++context_copy_seen;
            if (step.accepted == step.draft_tokens.size()) {
                ++context_copy_perfect_rounds;
            } else {
                context_copy_perfect_rounds = 0;
            }
            if (ratio >= 0.5) context_copy_backoff = 64;
            if (context_copy_seen >= 3 && context_copy_acceptance_ema < 0.35) {
                context_copy_suspend_until = result.tokens.size() + context_copy_backoff;
                context_copy_backoff = std::min<std::size_t>(context_copy_backoff * 2, 4096);
                context_copy_acceptance_ema = 0.5;
                context_copy_seen = 0;
                context_copy_perfect_rounds = 0;
                ++result.context_copy_suspensions;
            }
        } else if (used_history_draft) {
            ++result.history_draft_rounds;
            result.history_draft_proposed += step.draft_tokens.size();
            result.history_draft_accepted += step.accepted;
            history_draft_policy.observe_history(step.draft_tokens.size(), step.accepted);
        } else {
            depth_policy.observe(step.draft_tokens.size(), step.accepted);
            profitability_guard.observe(step.accepted);
            history_draft_policy.observe_learned(step.draft_tokens.size(), step.accepted);
        }
        result.history_draft_activations = history_draft_policy.activations();
        result.history_draft_deactivations = history_draft_policy.deactivations();
        result.mtp_final_depth = depth_policy.depth();
        result.mtp_promotions = depth_policy.promotions();
        result.mtp_demotions = depth_policy.demotions();
        current = step.next_current_token;
        previous_target_stream = std::move(step.next_target_stream);
        bool cancel_after_committed_round = false;
        for (const std::uint32_t token : step.emitted_tokens) {
            if (result.tokens.size() == max_tokens) break;
            if (is_stop_token(token)) {
                result.finish_reason = "stop";
                stopped_on_terminator = true;
                break;
            }
            result.tokens.push_back(token);
            if (!emit_new_tokens()) cancel_after_committed_round = true;
            if (history_draft_enabled) history_draft.append(token);
        }
        // Long copy windows create substantially larger transient verifier
        // allocations than the learned four-token path. Return those buffers
        // after the committed state has been selected so repeated requests do
        // not ratchet the MLX allocator footprint upward.
        if (used_context_copy && step.draft_tokens.size() > 4) {
            MlxArray::clear_cache();
        }
        if (options_.clear_cache_each_mtp_round) MlxArray::clear_cache();
        if (result.finish_reason == "stop") break;
        if (cancel_after_committed_round) {
            result.finish_reason = "cancelled";
            break;
        }
        const char* economic_fallback = std::getenv("QWEN38_ECONOMIC_MTP_FALLBACK");
        const bool economic_fallback_enabled = economic_fallback == nullptr ||
            std::string_view(economic_fallback) != "0";
        const bool should_fallback = !used_history_draft && !used_context_copy &&
            (economic_fallback_enabled
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
        prefix_cache_->mtp_profitable = cache_completed_mtp_as_profitable(
            result.mtp_rounds, result.mtp_accepted, result.mtp_fallbacks);
        prefix_cache_->mtp_profitability_current_token = prompt_tokens.back();
        prefix_cache_->mtp_cumulative_profitability_keep =
            result.mtp_fallbacks != 0 && *prefix_cache_->mtp_profitable;
        prefix_cache_changed = true;
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
            ? std::optional<bool>(cache_completed_mtp_as_profitable(
                  result.mtp_rounds, result.mtp_accepted, result.mtp_fallbacks))
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
            .mtp_cumulative_profitability_keep =
                result.mtp_fallbacks != 0 && extended_profitability.value_or(false),
        });
        prefix_cache_changed = true;
    }
    if (on_delta != nullptr && stream_connected && !pending_stream_bytes.empty() &&
        !(*on_delta)(pending_stream_bytes)) {
        result.finish_reason = "cancelled";
    }
    result.text = tokenizer_.decode(result.tokens);
    if (prefix_cache_changed && ssd_prefix_cache_ != nullptr) {
        try {
            if (prompt_cache_state.has_value()) {
                ssd_prefix_cache_->save(prompt_cache_tokens, *prompt_cache_state);
            }
            if (prefix_cache_ != nullptr &&
                (!prompt_cache_state.has_value() ||
                    prefix_cache_->tokens != prompt_cache_tokens)) {
                persist_prefix_cache(*prefix_cache_);
            }
        } catch (const std::exception& error) {
            std::cerr << "SSD prefix cache write failed: " << error.what() << '\n';
        }
    }
    return result;
}

PersistedPrefixState NativeEngine::snapshot_prefix_cache(
    const PrefixCacheEntry& entry) const {
    PersistedPrefixState persistent(model_.layer_count());
    persistent.target = model_.snapshot_state(entry.target_state);
    persistent.mtp = mtp_head_ == nullptr
        ? MtpDecodeState{}
        : mtp_head_->snapshot_state(entry.mtp_state);
    if (entry.previous_target_stream.has_value()) {
        persistent.previous_target_stream =
            entry.previous_target_stream->share();
    }
    persistent.pending_mtp_streams.reserve(entry.pending_mtp_streams.size());
    for (const MlxArray& stream : entry.pending_mtp_streams) {
        persistent.pending_mtp_streams.push_back(stream.share());
    }
    persistent.pending_mtp_tokens = entry.pending_mtp_tokens;
    persistent.mtp_profitable = entry.mtp_profitable;
    persistent.mtp_profitability_current_token =
        entry.mtp_profitability_current_token;
    persistent.mtp_cumulative_profitability_keep =
        entry.mtp_cumulative_profitability_keep;
    return persistent;
}

void NativeEngine::persist_prefix_cache(const PrefixCacheEntry& entry) const {
    if (ssd_prefix_cache_ == nullptr) return;
    const PersistedPrefixState persistent = snapshot_prefix_cache(entry);
    ssd_prefix_cache_->save(entry.tokens, persistent);
}

void NativeEngine::clear_cache() {
    std::scoped_lock lock(inference_mutex_);
    // Request-owned decode state is released at request completion. This also
    // returns unused MLX allocator/cache blocks to the system on demand.
    prefix_cache_.reset();
    if (ssd_prefix_cache_ != nullptr) ssd_prefix_cache_->clear();
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
