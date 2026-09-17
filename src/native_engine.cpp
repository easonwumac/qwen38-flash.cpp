#include "qwen38/native_engine.hpp"
#include "qwen38/runtime_profile.hpp"
#include "qwen38/history_draft.hpp"
#include "qwen38/mtp_depth_policy.hpp"
#include "qwen38/mtp_profitability.hpp"
#include "qwen38/mtp_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <functional>
#include <iostream>
#include <iomanip>
#include <optional>
#include <random>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <utility>
#include <pwd.h>
#include <unistd.h>

namespace qwen38 {
namespace {

bool has_native_vqlab_mtp_weights(const ModelManifest& manifest) {
    return manifest.has_tensor("fc.weight") && manifest.has_tensor("norm_e.weight") &&
        manifest.has_tensor("norm_h.weight") &&
        manifest.has_tensor("block.self_attn.q_proj.weight") &&
        manifest.has_tensor("mixer.hc_norm.weight");
}

bool has_mtp_weights(const ModelManifest& manifest) {
    return manifest.weight_map().contains("language_model.mtp.fc_embedding.weight") ||
        has_native_vqlab_mtp_weights(manifest);
}

const char* external_mtp_model_directory() {
    const char* directory = std::getenv("QWEN38_MTP_MODEL_DIR");
    return directory != nullptr && *directory != '\0' ? directory : nullptr;
}

std::size_t resolved_mtp_depth(
    const ModelManifest& manifest,
    const NativeEngineOptions& options) {
    const bool available = has_mtp_weights(manifest) ||
        external_mtp_model_directory() != nullptr;
    const std::size_t depth = options.mtp_depth.value_or(
        external_mtp_model_directory() != nullptr || has_native_vqlab_mtp_weights(manifest)
            ? 4 : available ? 3 : 0);
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

std::uint32_t sample_token(
    const MlxArray& logits,
    const ModelDecodeState& state,
    const SamplingOptions& options,
    std::mt19937_64& random,
    const std::span<const std::uint32_t> generated_tokens) {
    if (options.temperature <= 0.0F) return argmax_token(logits, state);
    const std::vector<int> shape = logits.shape();
    if (shape.empty() || shape.back() <= 0) {
        throw std::runtime_error("sampling requires a vocabulary axis");
    }
    const int vocabulary = shape.back();
    // A modest frequency penalty can move a recent token out of the requested
    // top-k. Inspect a bounded wider frontier so replacements can enter without
    // transferring a full vocabulary to the CPU.
    const std::size_t frontier = options.frequency_penalty == 0.0F
        ? options.top_k
        : std::max<std::size_t>(options.top_k, 256);
    const int count = static_cast<int>(std::min<std::size_t>(
        frontier, static_cast<std::size_t>(vocabulary)));
    if (count <= 0) throw std::runtime_error("sampling requires top_k to be positive");

    std::vector<int> start(shape.size(), 0);
    std::vector<int> stop = shape;
    std::vector<int> strides(shape.size(), 1);
    start.back() = vocabulary - count;
    MlxArray indices = logits.argpartition_axis(-count, -1)
                           .slice(start, stop, strides)
                           .reshape(std::vector<int>{count});
    MlxArray values = MlxArray::take(logits, indices)
                          .reshape(std::vector<int>{count})
                          .astype(MLX_FLOAT32);
    MlxArray probabilities = MlxArray::take(logits.softmax_axis(-1), indices)
                                 .reshape(std::vector<int>{count})
                                 .astype(MLX_FLOAT32);
    eval_with_decode_state(values, state);
    std::vector<float> candidate_values = values.to_float32();
    std::vector<float> candidate_probabilities = probabilities.to_float32();
    const std::vector<float> candidate_indices = indices.astype(MLX_FLOAT32).to_float32();
    if (options.frequency_penalty != 0.0F && !generated_tokens.empty()) {
        constexpr std::size_t context_size = 64;
        const std::size_t begin = generated_tokens.size() > context_size
            ? generated_tokens.size() - context_size
            : 0;
        double normalization = 1.0;
        for (std::size_t candidate = 0; candidate < candidate_values.size(); ++candidate) {
            const std::uint32_t token = static_cast<std::uint32_t>(candidate_indices[candidate]);
            const std::size_t occurrences = static_cast<std::size_t>(std::count(
                generated_tokens.begin() + static_cast<std::ptrdiff_t>(begin),
                generated_tokens.end(), token));
            if (occurrences == 0) continue;
            const double penalty = static_cast<double>(options.frequency_penalty) *
                static_cast<double>(occurrences);
            const double multiplier = std::exp(-penalty);
            candidate_values[candidate] -= static_cast<float>(penalty);
            normalization += static_cast<double>(candidate_probabilities[candidate]) *
                (multiplier - 1.0);
            candidate_probabilities[candidate] *= static_cast<float>(multiplier);
        }
        if (normalization > 0.0) {
            for (float& probability : candidate_probabilities) {
                probability /= static_cast<float>(normalization);
            }
        }
    }
    std::vector<std::size_t> order(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
    std::ranges::sort(order, {}, [&](const std::size_t index) {
        return -candidate_values[index];
    });

    const float maximum = candidate_values[order.front()];
    std::vector<double> weights(order.size());
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        weights[rank] = std::exp(
            static_cast<double>(candidate_values[order[rank]] - maximum) /
            static_cast<double>(options.temperature));
    }
    // Match mlx-vlm's ordering: nucleus filtering uses full-vocabulary,
    // untempered probabilities, then top-k is applied, and temperature only
    // affects the final categorical draw. A descending token survives while
    // the probability mass strictly above it is below top_p.
    double higher_probability = 0.0;
    std::size_t retained_count = 0;
    const std::size_t top_k_count = std::min(options.top_k, weights.size());
    while (retained_count < top_k_count && higher_probability < options.top_p) {
        higher_probability += candidate_probabilities[order[retained_count]];
        ++retained_count;
    }
    double retained = 0.0;
    for (std::size_t rank = 0; rank < retained_count; ++rank) {
        retained += weights[rank];
    }
    std::uniform_real_distribution<double> draw(0.0, retained);
    const double target = draw(random);
    double cumulative = 0.0;
    for (std::size_t rank = 0; rank < retained_count; ++rank) {
        cumulative += weights[rank];
        if (target <= cumulative) {
            return static_cast<std::uint32_t>(candidate_indices[order[rank]]);
        }
    }
    return static_cast<std::uint32_t>(candidate_indices[order[retained_count - 1]]);
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
    if (const char* external_value = external_mtp_model_directory()) {
        const std::filesystem::path external(external_value);
        source << "|mtp="
               << std::filesystem::weakly_canonical(external, ignored).string();
        ignored.clear();
        for (const char* file : {"config.json", "model.safetensors.index.json"}) {
            const std::filesystem::path path = external / file;
            std::uintmax_t bytes = std::filesystem::file_size(path, ignored);
            if (ignored) bytes = 0;
            ignored.clear();
            const auto modified = std::filesystem::last_write_time(path, ignored);
            const auto ticks = ignored ? 0LL
                : std::chrono::duration_cast<std::chrono::nanoseconds>(
                      modified.time_since_epoch()).count();
            source << '|' << bytes << '|' << ticks;
            ignored.clear();
        }
    }
    constexpr std::array<const char*, 20> state_environment{
        "QWEN38_COMPACT_QMETA",
        "QWEN38_FUSED_MOE",
        "QWEN38_DEVICE_ROUTER",
        "QWEN38_SELECTED_SOFTMAX_ROUTER",
        "QWEN38_GDN_METAL_PREFILL",
        "QWEN38_SDPA_PREFILL",
        "QWEN38_QSA_PREFILL",
        "QWEN38_QSA_PACKED_PREFILL",
        "QWEN38_QSA_PACKED_MIN_TOKENS",
        "QWEN38_QSA_SHARED_ROWS",
        "QWEN38_KV_CACHE",
        "QWEN38_KV_Q8_MIN_TOKENS",
        "QWEN38_KV_Q8_FLUSH_TOKENS",
        "QWEN38_QSA_DECODE_BUDGET",
        "QWEN38_QSA_RAW_WINDOW",
        "QWEN38_HC_FUSED_INJECTION",
        "QWEN38_COMPILE_LAYER",
        "QWEN38_SDPA_DECODE",
        "QWEN38_NIWAKI_FULL_MAPS",
        "QWEN38_SKIP_NIWAKI_MAPS",
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
      mtp_depth_(resolved_mtp_depth(tensors_.manifest(), options_)),
      adaptive_mtp_depth_(
          !options_.mtp_depth.has_value() &&
          has_native_vqlab_mtp_weights(tensors_.manifest())) {
    const std::vector<std::uint32_t> chat_end = tokenizer_.encode("<|im_end|>");
    if (chat_end.size() != 1) {
        throw std::runtime_error("chat end marker must encode to one token");
    }
    chat_end_token_ = chat_end.front();
    if (options_.max_generation_tokens == 0) {
        throw std::runtime_error("generation token limit must be positive");
    }
    if (options_.prefill_chunk_rows == 0 || options_.prefill_chunk_rows > 2048) {
        throw std::runtime_error("prefill chunk rows must be between 1 and 2048");
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
    const char* persistent = std::getenv("QWEN38_PERSISTENT_METAL");
    const bool persistent_explicit = persistent != nullptr;
    const bool persistent_enabled = persistent == nullptr ||
        std::string_view(persistent) != "0";
    if (persistent_enabled) {
        MlxTensorStore* shared_weights = &tensors_;
        if (const char* share = std::getenv("QWEN38_PERSISTENT_SHARE_MLX_WEIGHTS");
            share != nullptr && std::string_view(share) == "0") {
            shared_weights = nullptr;
        }
        persistent_backend_ = PersistentMetalBackend::create(
            tensors_.manifest(), shared_weights);
        if (persistent_backend_ == nullptr && persistent_explicit &&
            std::string_view(persistent) == "1") {
            throw std::runtime_error("model is not eligible for persistent Metal");
        }
        if (persistent_backend_ != nullptr) {
            std::clog << "qwen38-server: persistent Metal decode enabled\n";
        }
    }
    if (mtp_depth_ != 0) {
        if (options_.zero_accept_fallback_rounds == 0) {
            throw std::runtime_error("MTP fallback window must be positive");
        }
        MlxTensorStore* mtp_tensors = &tensors_;
        if (!has_mtp_weights(tensors_.manifest())) {
            const char* external = external_mtp_model_directory();
            if (external == nullptr) {
                throw std::runtime_error("MTP was requested but no sidecar is available");
            }
            ModelManifest manifest = ModelManifest::load(external);
            const ModelConfig& target = tensors_.manifest().config();
            const ModelConfig& draft = manifest.config();
            if (!has_mtp_weights(manifest) || draft.hidden_size != target.hidden_size ||
                draft.hyper_connection_count != target.hyper_connection_count ||
                draft.vocabulary_size != target.vocabulary_size) {
                throw std::runtime_error("external MTP model is incompatible with the target");
            }
            mtp_tensors_ = std::make_unique<MlxTensorStore>(std::move(manifest));
            mtp_tensors = mtp_tensors_.get();
            std::clog << "qwen38-server: external MTP model=" << external << '\n';
        }
        mtp_head_ = std::make_unique<QwenMtpHead>(*mtp_tensors);
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
    const std::size_t max_tokens,
    const SamplingOptions& sampling) {
    return complete_impl(prompt, max_tokens, nullptr, sampling);
}

GenerationResult NativeEngine::complete_stream(
    const std::string_view prompt,
    const std::size_t max_tokens,
    const TextDeltaCallback& on_delta,
    const SamplingOptions& sampling) {
    return complete_impl(prompt, max_tokens, &on_delta, sampling);
}

void NativeEngine::complete_batch(std::vector<BatchRequest> requests,
                                  const BatchRefillCallback &refill) {
    if (requests.empty() || requests.size() > 4) {
        throw std::runtime_error("continuous decode batch must contain 1 to 4 requests");
    }
    if (requests.size() == 1 || std::ranges::any_of(requests, [](const BatchRequest &request) {
            return request.sampling.thinking_budget_tokens != 0;
        })) {
        for (BatchRequest &request : requests) {
            try {
                GenerationResult result =
                    request.on_delta.has_value()
                        ? complete_stream(request.prompt, request.max_tokens, *request.on_delta,
                                          request.sampling)
                        : complete(request.prompt, request.max_tokens, request.sampling);
                if (request.on_complete)
                    request.on_complete(std::move(result));
            } catch (...) {
                if (request.on_error)
                    request.on_error(std::current_exception());
            }
        }
        return;
    }

    struct Session {
        Session(BatchRequest value, const std::size_t layer_count)
            : request(std::move(value)), state(layer_count), random(request.sampling.seed) {}
        BatchRequest request;
        std::vector<std::uint32_t> prompt_tokens;
        ModelDecodeState state;
        GenerationResult result;
        std::mt19937_64 random;
        std::uint32_t current{0};
        std::string pending_stream_bytes;
        std::chrono::steady_clock::time_point generation_started;
        std::optional<ModelPrefillChunk> pending_prefill;
        std::size_t prefill_offset{0};
        std::size_t prefill_rows{0};
        std::size_t prefill_chunk_rows{0};
        std::size_t current_prefill_rows{0};
        double prefill_ms{0.0};
        double generation_ms{0.0};
        bool active{true};
        bool ready{false};
        bool prefill_started{false};
        std::size_t reserved_tokens{0};
    };

    std::vector<std::vector<std::uint32_t>> tokenized;
    tokenized.reserve(requests.size());
    std::size_t aggregate_capacity = 0;
    for (const BatchRequest &request : requests) {
        if (request.max_tokens == 0 || request.max_tokens > options_.max_generation_tokens) {
            throw std::runtime_error("batched max_tokens is outside the configured limit");
        }
        if (request.sampling.temperature < 0.0F || request.sampling.top_p <= 0.0F ||
            request.sampling.top_p > 1.0F || request.sampling.frequency_penalty < 0.0F ||
            request.sampling.frequency_penalty > 2.0F ||
            (request.sampling.temperature > 0.0F &&
             (request.sampling.top_k == 0 || request.sampling.top_k > 256))) {
            throw std::runtime_error("invalid sampling options in continuous decode batch");
        }
        tokenized.push_back(tokenizer_.encode(request.prompt));
        if (tokenized.back().empty())
            throw std::runtime_error("prompt produced no tokens");
        const std::size_t context_limit = tensors_.manifest().config().max_context_tokens;
        if (tokenized.back().size() > context_limit ||
            request.max_tokens > context_limit - tokenized.back().size()) {
            throw std::runtime_error("batched prompt and max_tokens exceed model context");
        }
        aggregate_capacity += tokenized.back().size() + request.max_tokens;
    }
    std::size_t aggregate_limit = 131072;
    if (const char *configured = std::getenv("QWEN38_BATCH_CONTEXT_TOKENS")) {
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(configured, &end, 10);
        if (end == configured || *end != '\0' || parsed < 4096) {
            throw std::runtime_error("QWEN38_BATCH_CONTEXT_TOKENS must be at least 4096");
        }
        aggregate_limit = static_cast<std::size_t>(parsed);
    }
    if (aggregate_capacity > aggregate_limit) {
        for (BatchRequest &request : requests) {
            try {
                GenerationResult result =
                    request.on_delta.has_value()
                        ? complete_stream(request.prompt, request.max_tokens, *request.on_delta,
                                          request.sampling)
                        : complete(request.prompt, request.max_tokens, request.sampling);
                if (request.on_complete)
                    request.on_complete(std::move(result));
            } catch (...) {
                if (request.on_error)
                    request.on_error(std::current_exception());
            }
        }
        return;
    }

    std::unique_lock lock(inference_mutex_);
    if (persistent_backend_ != nullptr)
        persistent_backend_->release_shared_weights();
    // A single retained prefix cannot safely own several advancing states.
    // Keep SSD contents, but release the RAM snapshot while the batch is live.
    prefix_cache_.reset();
    std::vector<Session> sessions;
    sessions.reserve(4);
    const auto initialize_session = [&](BatchRequest request,
                                        std::vector<std::uint32_t> prompt_tokens,
                                        const bool incremental_prefill) -> bool {
        Session session(std::move(request), model_.layer_count());
        try {
            session.prompt_tokens = std::move(prompt_tokens);
            session.reserved_tokens = session.prompt_tokens.size() + session.request.max_tokens;
            session.state = model_.make_state();
            session.current = session.prompt_tokens.back();
            session.result.prompt_tokens = session.prompt_tokens.size();
            session.prefill_rows = session.prompt_tokens.size() - 1;
            session.prefill_chunk_rows =
                options_.adaptive_prefill_chunks
                    ? select_prefill_chunk_rows(options_.prefill_chunk_rows,
                                                session.prefill_rows)
                    : options_.prefill_chunk_rows;
            if (session.prefill_rows == 0) {
                session.ready = true;
                session.generation_started = std::chrono::steady_clock::now();
            } else if (!incremental_prefill) {
                model_.clear_prefill_qmeta_cache();
                model_.set_prefill_qmeta_cache_allowed(
                    options_.qmeta_cache_max_prompt_tokens != 0 &&
                    session.prefill_rows <= options_.qmeta_cache_max_prompt_tokens);
                const auto started = std::chrono::steady_clock::now();
                for (std::size_t offset = 0; offset < session.prefill_rows;
                     offset += session.prefill_chunk_rows) {
                    const std::size_t count = std::min(
                        session.prefill_chunk_rows, session.prefill_rows - offset);
                    static_cast<void>(model_.prefill_chunk_batch(
                        std::span<const std::uint32_t>(
                            session.prompt_tokens.data() + offset, count),
                        session.state));
                }
                session.result.prompt_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - started)
                                               .count();
                session.generation_started = std::chrono::steady_clock::now();
                session.ready = true;
            }
            sessions.push_back(std::move(session));
            return true;
        } catch (...) {
            if (session.request.on_error)
                session.request.on_error(std::current_exception());
            return false;
        }
    };
    for (std::size_t slot = 0; slot < requests.size(); ++slot) {
        initialize_session(std::move(requests[slot]), std::move(tokenized[slot]), false);
    }
    model_.clear_prefill_qmeta_cache();

    const auto advance_pending_prefill = [&](Session &session) {
        if (session.ready)
            return;
        if (!session.prefill_started) {
            model_.clear_prefill_qmeta_cache();
            model_.set_prefill_qmeta_cache_allowed(
                options_.qmeta_cache_max_prompt_tokens != 0 &&
                session.prefill_rows <= options_.qmeta_cache_max_prompt_tokens);
            session.prefill_started = true;
        }
        if (!session.pending_prefill.has_value()) {
            session.current_prefill_rows = std::min(
                session.prefill_chunk_rows,
                session.prefill_rows - session.prefill_offset);
            const std::span<const std::uint32_t> tokens(
                session.prompt_tokens.data() + session.prefill_offset,
                session.current_prefill_rows);
            session.pending_prefill.emplace(
                model_.begin_prefill_chunk_batch(tokens, session.state));
        }
        const std::span<const std::uint32_t> tokens(
            session.prompt_tokens.data() + session.prefill_offset,
            session.current_prefill_rows);
        const auto started = std::chrono::steady_clock::now();
        const bool chunk_done = model_.advance_prefill_chunk_batch(
            *session.pending_prefill, tokens, session.state);
        session.prefill_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
        if (chunk_done) {
            session.prefill_offset += session.current_prefill_rows;
            session.current_prefill_rows = 0;
            session.pending_prefill.reset();
        }
        if (session.prefill_offset == session.prefill_rows) {
            session.result.prompt_ms = session.prefill_ms;
            session.generation_started = std::chrono::steady_clock::now();
            session.ready = true;
            model_.clear_prefill_qmeta_cache();
        }
    };

    std::vector<BatchRequest> deferred;
    try {
        while (!sessions.empty()) {
            std::vector<std::uint32_t> tokens;
            std::vector<ModelDecodeState *> states;
            std::vector<std::size_t> decode_rows;
            tokens.reserve(sessions.size());
            states.reserve(sessions.size());
            decode_rows.reserve(sessions.size());
            for (std::size_t slot = 0; slot < sessions.size(); ++slot) {
                if (!sessions[slot].ready)
                    continue;
                tokens.push_back(sessions[slot].current);
                states.push_back(&sessions[slot].state);
                decode_rows.push_back(slot);
            }
            std::vector<TargetDecodeStep> steps;
            if (!tokens.empty())
                steps = model_.forward_decode_capture_multi(tokens, states);
            std::vector<std::uint32_t> next_tokens(steps.size());
            std::vector<MlxArray> greedy_tokens;
            std::vector<std::size_t> greedy_rows;
            greedy_tokens.reserve(steps.size());
            greedy_rows.reserve(steps.size());
            for (std::size_t row = 0; row < steps.size(); ++row) {
                if (sessions[decode_rows[row]].request.sampling.temperature <= 0.0F) {
                    greedy_rows.push_back(row);
                    greedy_tokens.push_back(steps[row].logits.argmax_all());
                }
            }
            if (!greedy_tokens.empty()) {
                std::vector<const MlxArray *> token_arrays;
                token_arrays.reserve(greedy_tokens.size());
                for (const MlxArray &token : greedy_tokens)
                    token_arrays.push_back(&token);
                MlxArray::eval_all(token_arrays);
                for (std::size_t index = 0; index < greedy_tokens.size(); ++index) {
                    next_tokens[greedy_rows[index]] = greedy_tokens[index].item_uint32();
                }
            }
            for (std::size_t row = 0; row < steps.size(); ++row) {
                Session &session = sessions[decode_rows[row]];
                BatchRequest &request = session.request;
                const std::uint32_t next =
                    request.sampling.temperature <= 0.0F
                        ? next_tokens[row]
                        : sample_token(steps[row].logits, session.state, request.sampling,
                                       session.random, session.result.tokens);
                session.current = next;
                const bool stop = next == tensors_.manifest().config().end_of_sequence_token ||
                                  next == chat_end_token_;
                if (stop) {
                    session.result.finish_reason = "stop";
                    session.generation_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  session.generation_started)
                            .count();
                    session.active = false;
                    continue;
                }
                session.result.tokens.push_back(next);
                if (request.on_delta.has_value()) {
                    session.pending_stream_bytes += tokenizer_.decode(
                        std::span<const std::uint32_t>(&session.result.tokens.back(), 1));
                    const std::size_t complete = complete_utf8_prefix(session.pending_stream_bytes);
                    if (complete != 0 &&
                        !(*request.on_delta)(
                            std::string_view(session.pending_stream_bytes).substr(0, complete))) {
                        session.result.finish_reason = "cancelled";
                        session.generation_ms =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - session.generation_started)
                                .count();
                        session.active = false;
                    }
                    session.pending_stream_bytes.erase(0, complete);
                }
                if (session.active && session.result.tokens.size() == request.max_tokens) {
                    session.generation_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  session.generation_started)
                            .count();
                    session.active = false;
                }
            }

            std::size_t live_capacity = 0;
            for (Session &session : sessions) {
                if (session.active) {
                    live_capacity += session.reserved_tokens;
                    continue;
                }
                if (session.request.on_delta.has_value() &&
                    session.result.finish_reason != "cancelled" &&
                    !session.pending_stream_bytes.empty() &&
                    !(*session.request.on_delta)(session.pending_stream_bytes)) {
                    session.result.finish_reason = "cancelled";
                }
                session.result.generation_ms = session.generation_ms;
                session.result.text = tokenizer_.decode(session.result.tokens);
                if (session.request.on_complete) {
                    session.request.on_complete(std::move(session.result));
                }
            }
            std::erase_if(sessions, [](const Session &session) { return !session.active; });

            while (!sessions.empty() && sessions.size() < 4 && refill) {
                std::optional<BatchRequest> request = refill();
                if (!request.has_value())
                    break;
                try {
                    if (request->max_tokens == 0 ||
                        request->max_tokens > options_.max_generation_tokens) {
                        throw std::runtime_error(
                            "refill max_tokens is outside the configured limit");
                    }
                    if (request->sampling.temperature < 0.0F || request->sampling.top_p <= 0.0F ||
                        request->sampling.top_p > 1.0F ||
                        request->sampling.frequency_penalty < 0.0F ||
                        request->sampling.frequency_penalty > 2.0F ||
                        (request->sampling.temperature > 0.0F &&
                         (request->sampling.top_k == 0 || request->sampling.top_k > 256))) {
                        throw std::runtime_error("invalid sampling options in batch refill");
                    }
                    std::vector<std::uint32_t> prompt_tokens = tokenizer_.encode(request->prompt);
                    if (prompt_tokens.empty())
                        throw std::runtime_error("prompt produced no tokens");
                    const std::size_t context_limit =
                        tensors_.manifest().config().max_context_tokens;
                    if (prompt_tokens.size() > context_limit ||
                        request->max_tokens > context_limit - prompt_tokens.size()) {
                        throw std::runtime_error(
                            "refill prompt and max_tokens exceed model context");
                    }
                    const std::size_t capacity = prompt_tokens.size() + request->max_tokens;
                    if (request->sampling.thinking_budget_tokens != 0 ||
                        live_capacity + capacity > aggregate_limit) {
                        deferred.push_back(std::move(*request));
                        break;
                    }
                    if (initialize_session(
                            std::move(*request), std::move(prompt_tokens), true)) {
                        live_capacity += capacity;
                    }
                } catch (...) {
                    if (request->on_error)
                        request->on_error(std::current_exception());
                }
            }

            // Preserve the full prefill matrix shape and exact layer order,
            // but yield at its existing evaluation barriers so live decode
            // streams cannot be blocked by the entire prompt.
            const auto pending = std::ranges::find_if(
                sessions, [](const Session &session) {
                    return session.active && !session.ready;
                });
            if (pending != sessions.end()) {
                try {
                    advance_pending_prefill(*pending);
                } catch (...) {
                    if (pending->request.on_error)
                        pending->request.on_error(std::current_exception());
                    model_.clear_prefill_qmeta_cache();
                    sessions.erase(pending);
                }
            }
        }
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        for (Session &session : sessions) {
            if (session.request.on_error)
                session.request.on_error(error);
        }
        for (BatchRequest &request : deferred) {
            if (request.on_error)
                request.on_error(error);
        }
        MlxArray::clear_cache();
        return;
    }
    MlxArray::clear_cache();
    lock.unlock();
    for (BatchRequest &request : deferred) {
        try {
            GenerationResult result =
                request.on_delta.has_value()
                    ? complete_stream(request.prompt, request.max_tokens, *request.on_delta,
                                      request.sampling)
                    : complete(request.prompt, request.max_tokens, request.sampling);
            if (request.on_complete)
                request.on_complete(std::move(result));
        } catch (...) {
            if (request.on_error)
                request.on_error(std::current_exception());
        }
    }
}

GenerationResult NativeEngine::complete_impl(
    const std::string_view prompt,
    const std::size_t max_tokens,
    const TextDeltaCallback* on_delta,
    const SamplingOptions& sampling) {
    if (max_tokens == 0 || max_tokens > options_.max_generation_tokens) {
        throw std::runtime_error(
            "max_tokens must be between 1 and the configured generation limit (" +
            std::to_string(options_.max_generation_tokens) + ")");
    }
    if (sampling.temperature < 0.0F || sampling.top_p <= 0.0F ||
        sampling.top_p > 1.0F ||
        sampling.frequency_penalty < 0.0F || sampling.frequency_penalty > 2.0F ||
        (sampling.temperature > 0.0F &&
            (sampling.top_k == 0 || sampling.top_k > 256))) {
        throw std::runtime_error(
            "sampling requires temperature >= 0, top_p in (0, 1], top_k in 1..256, "
            "and frequency_penalty in [0, 2]");
    }
    const bool sampling_enabled = sampling.temperature > 0.0F;
    const bool serial_only = sampling_enabled || sampling.thinking_budget_tokens != 0;
    std::mt19937_64 sampling_random(sampling.seed);
    std::scoped_lock lock(inference_mutex_);
    if (persistent_backend_ != nullptr) {
        persistent_backend_->release_shared_weights();
    }
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
                .ssd_backed = true,
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
        if (ssd_prefix_cache_ != nullptr && prefix_cache_->ssd_backed &&
            prefill_offset < prefill_rows) {
            // The live snapshots above own everything needed for continuation.
            // Keeping this old checkpoint through extension pins the old KV
            // backing allocations. Its SSD copy remains available for retry;
            // no extra save/load or change to the compute path is necessary.
            // Exact hits retain their existing profitability/cache behavior.
            prefix_cache_.reset();
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
    const auto consume_stream_batch = [&](const MlxArray& stream_batch,
                                           const std::size_t offset,
                                           const std::size_t count) {
        for (std::size_t batch_offset = 0; batch_offset < count; batch_offset += 512) {
            const std::size_t batch_count = std::min<std::size_t>(512, count - batch_offset);
            MlxArray batch = batch_offset == 0 && batch_count == count
                ? stream_batch.share()
                : slice_sequence_rows(
                      stream_batch, batch_offset, batch_offset + batch_count);
            consume_target_batch(batch, offset + batch_offset, batch_count);
        }
    };
    const bool packed_vq_layer_major = mtp_head_ == nullptr &&
        prefill_rows - prefill_offset > request_prefill_chunk &&
        tensors_.manifest().vector_quantization_for(
            "language_model.model.layers.2.mlp.switch_mlp.gate_proj") != nullptr;
    if (packed_vq_layer_major) {
        // Bound retained hidden streams by rows rather than chunk count. Long
        // prompts automatically use smaller chunks but keep the same memory cap.
        constexpr std::size_t window_rows = 8192;
        for (std::size_t window_offset = prefill_offset; window_offset < prefill_rows;
             window_offset += window_rows) {
            const std::size_t count = std::min(window_rows, prefill_rows - window_offset);
            std::vector<MlxArray> outputs = model_.prefill_chunks_layer_major(
                std::span<const std::uint32_t>(prompt_tokens.data() + window_offset, count),
                request_prefill_chunk, state,
                profile_prefill_enabled ? &prefill_layer_ms : nullptr);
            std::size_t output_offset = window_offset;
            for (const MlxArray& output : outputs) {
                const std::size_t output_count =
                    std::min(request_prefill_chunk, prefill_rows - output_offset);
                consume_stream_batch(output, output_offset, output_count);
                output_offset += output_count;
            }
        }
    } else {
        for (std::size_t offset = prefill_offset; offset < prefill_rows;
             offset += request_prefill_chunk) {
            const std::size_t count = std::min(
                request_prefill_chunk, prefill_rows - offset);
            MlxArray stream_batch = model_.prefill_chunk_batch(
                std::span<const std::uint32_t>(prompt_tokens.data() + offset, count), state,
                profile_prefill_enabled ? &prefill_layer_ms : nullptr);
            consume_stream_batch(stream_batch, offset, count);
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
            const auto cache_write_started = std::chrono::steady_clock::now();
            prompt_cache_tokens = prefix_cache_->tokens;
            prompt_cache_state.emplace(snapshot_prefix_cache(*prefix_cache_));
            try {
                if (ssd_prefix_cache_->save(prompt_cache_tokens, *prompt_cache_state)) {
                    // The persistent copy now owns the prompt checkpoint. Drop
                    // both shared RAM owners before decode so advancing live
                    // state can release the old backing allocations.
                    prefix_cache_.reset();
                    prompt_cache_state.reset();
                }
            } catch (const std::exception& error) {
                // Preserve the RAM checkpoint and the existing end-of-request
                // retry path when persistence fails.
                std::cerr << "SSD prompt prefix cache write failed: "
                          << error.what() << '\n';
            }
            result.prompt_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cache_write_started).count();
        }
    }
    result.tokens.reserve(max_tokens);
    const char* extend_cache = std::getenv("QWEN38_EXTEND_PREFIX_CACHE");
    const bool extend_cache_enabled =
        persistent_backend_ == nullptr &&
        (ssd_prefix_cache_ != nullptr ||
         (extend_cache != nullptr && std::string_view(extend_cache) == "1"));
    bool mtp_cache_extendable = true;
    std::vector<MlxArray> pending_mtp_streams;
    std::vector<std::uint32_t> pending_mtp_tokens;
    std::uint32_t current = prompt_tokens.back();
    std::size_t persistent_anchor_remaining = 0;
    if (const char* configured = std::getenv("QWEN38_PERSISTENT_ANCHOR_TOKENS");
        persistent_backend_ != nullptr && configured != nullptr) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(configured, &end, 10);
        if (end == configured || *end != '\0' || parsed > 64) {
            throw std::runtime_error("QWEN38_PERSISTENT_ANCHOR_TOKENS must be 0..64");
        }
        persistent_anchor_remaining = std::min<std::size_t>(parsed, max_tokens);
    }
    bool persistent_reset = prefill_rows == 0 && persistent_anchor_remaining == 0;
    bool persistent_active = false;
    const char* shared_weights_environment =
        std::getenv("QWEN38_PERSISTENT_SHARE_MLX_WEIGHTS");
    const bool persistent_shared_weights = shared_weights_environment == nullptr ||
        std::string_view(shared_weights_environment) != "0";
    const auto activate_persistent = [&] {
        if (persistent_shared_weights) {
            // The first command on the independent direct queue pays Metal's
            // resource-registration handoff. Consume it before the imported
            // state becomes live, then overwrite all mutated decode state.
            persistent_backend_->prepare_shared_weights();
            static_cast<void>(persistent_backend_->greedy_decode(0, true));
        }
        persistent_backend_->import_state(state);
        const std::size_t imported_token_count = state.token_count;
        state = model_.make_state();
        state.token_count = imported_token_count;
        previous_target_stream.reset();
        MlxArray::clear_cache();
        persistent_active = true;
        persistent_reset = false;
    };
    if (!serial_only && persistent_backend_ != nullptr && mtp_head_ == nullptr &&
        persistent_anchor_remaining == 0) {
        const auto prepare_started = std::chrono::steady_clock::now();
        if (!persistent_reset) {
            activate_persistent();
        } else if (persistent_shared_weights) {
            persistent_backend_->prepare_shared_weights();
            static_cast<void>(persistent_backend_->greedy_decode(0, true));
            persistent_active = true;
        }
        result.prompt_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prepare_started).count();
    }
    const char* history_draft_environment = std::getenv("QWEN38_HISTORY_DRAFT");
    const HistoryDraftMode history_draft_mode = history_draft_environment == nullptr
        ? HistoryDraftMode::adaptive
        : (std::string_view(history_draft_environment) == "0"
                  ? HistoryDraftMode::disabled
                  : HistoryDraftMode::forced);
    HistoryDraftPolicy history_draft_policy(history_draft_mode);
    const bool history_draft_enabled = !serial_only && history_draft_policy.enabled();
    HistoryDraftCache history_draft;
    if (history_draft_enabled) history_draft.append(prompt_tokens);
    const char* context_copy_environment = std::getenv("QWEN38_CONTEXT_COPY");
    const bool context_copy_enabled = !serial_only && context_copy_environment != nullptr &&
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
    bool mtp_profitable = !serial_only && mtp_head_ != nullptr &&
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
    MtpDepthPolicy depth_policy(
        mtp_depth_, prompt_tokens.size(), adaptive_mtp_depth_);
    result.mtp_final_depth = serial_only ? 0 : depth_policy.depth();
    bool stopped_on_terminator = false;
    const auto is_stop_token = [&](const std::uint32_t token) {
        return token == tensors_.manifest().config().end_of_sequence_token ||
            token == chat_end_token_;
    };
    const std::array<std::uint32_t, 2> stop_tokens{
        tensors_.manifest().config().end_of_sequence_token, chat_end_token_};
    std::size_t persistent_min_tokens = persistent_backend_ == nullptr
        ? 0 : std::min<std::size_t>(8, max_tokens);
    if (const char* configured = std::getenv("QWEN38_PERSISTENT_MIN_TOKENS");
        persistent_backend_ != nullptr && configured != nullptr) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(configured, &end, 10);
        if (end == configured || *end != '\0' || parsed > 64) {
            throw std::runtime_error("QWEN38_PERSISTENT_MIN_TOKENS must be 0..64");
        }
        persistent_min_tokens = std::min<std::size_t>(parsed, max_tokens);
    }
    const char* profile_serial_decode = std::getenv("QWEN38_PROFILE_SERIAL_DECODE");
    if (profile_serial_decode != nullptr &&
        std::string_view(profile_serial_decode) == "1") {
        ModelDecodeState profiled_state = model_.snapshot_state(state);
        std::vector<double> checksums;
        std::vector<double> layer_ms;
        const auto profile_started = std::chrono::steady_clock::now();
        static_cast<void>(model_.trace_decode(
            current, profiled_state, checksums, layer_ms)
            .astype(MLX_FLOAT32).to_float32());
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - profile_started).count();
        double linear_ms = 0.0;
        double full_ms = 0.0;
        for (std::size_t layer = 0; layer < layer_ms.size(); ++layer) {
            (layer + 1) % 4 == 0 ? full_ms += layer_ms[layer]
                                 : linear_ms += layer_ms[layer];
        }
        std::clog << "qwen38-serial-decode-profile: context=" << state.token_count
                  << " total_ms=" << total_ms
                  << " linear_layers_ms=" << linear_ms
                  << " full_layers_ms=" << full_ms
                  << " layer_ms=[";
        for (std::size_t layer = 0; layer < layer_ms.size(); ++layer) {
            if (layer != 0) std::clog << ',';
            std::clog << layer_ms[layer];
        }
        std::clog << "]\n";
    }
    const auto generation_started = std::chrono::steady_clock::now();
    result.thinking_budget_tokens = sampling.thinking_budget_tokens;
    const std::vector<std::uint32_t> thinking_close_tokens =
        sampling.thinking_budget_tokens == 0
        ? std::vector<std::uint32_t>{}
        : tokenizer_.encode("</think>");
    if (sampling.thinking_budget_tokens != 0 && thinking_close_tokens.size() != 1) {
        throw std::runtime_error("Qwen thinking close marker must encode to one token");
    }
    const std::vector<std::uint32_t> forced_thinking_suffix =
        sampling.thinking_budget_tokens == 0
        ? std::vector<std::uint32_t>{}
        : tokenizer_.encode(
              "\n\nConsidering the limited time by the user, I have to give the solution "
              "based on the thinking directly now.\n</think>\n\n");
    std::deque<std::uint32_t> forced_tokens;
    bool thinking_closed = false;
    bool thinking_intervention_attempted = false;
    const auto begin_thinking_intervention = [&] {
        if (thinking_intervention_attempted || thinking_closed ||
            forced_thinking_suffix.empty()) return;
        thinking_intervention_attempted = true;
        // Keep at least one model-generated final-answer token available.
        if (result.tokens.size() + forced_thinking_suffix.size() >= max_tokens) return;
        forced_tokens.insert(
            forced_tokens.end(), forced_thinking_suffix.begin(), forced_thinking_suffix.end());
        result.thinking_budget_forced = true;
    };
    std::optional<std::size_t> pending_top2_recovery_position;
    while (result.tokens.size() < max_tokens) {
        if (!thinking_closed && forced_tokens.empty() &&
            sampling.thinking_budget_tokens != 0 &&
            result.tokens.size() >= sampling.thinking_budget_tokens) {
            begin_thinking_intervention();
        }
        const std::size_t remaining = max_tokens - result.tokens.size();
        if (!mtp_profitable || !previous_target_stream.has_value() ||
            remaining == 1 || (extend_cache_enabled && remaining == 2)) {
            pending_top2_recovery_position.reset();
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
            if (persistent_active && persistent_anchor_remaining == 0) {
                const PersistentMetalBackend::GreedyResult step =
                    persistent_backend_->greedy_decode(current, persistent_reset);
                if (const char* trace = std::getenv("QWEN38_PERSISTENT_TRACE");
                    trace != nullptr && std::string_view(trace) == "1") {
                    std::clog << "qwen38-persistent-decode: context=" << state.token_count
                              << " gpu_ms=" << step.gpu_ms
                              << " wall_ms=" << step.wall_ms << '\n';
                }
                persistent_reset = false;
                current = is_stop_token(step.token) &&
                        result.tokens.size() < persistent_min_tokens
                    ? step.alternative_token : step.token;
                ++state.token_count;
            } else {
                TargetDecodeStep step = model_.forward_decode_capture(current, state);
                std::uint32_t token = forced_tokens.empty()
                    ? sample_token(
                          step.logits, state, sampling, sampling_random, result.tokens)
                    : forced_tokens.front();
                previous_target_stream = std::move(step.pre_mixer_stream);
                if (forced_tokens.empty() && !thinking_closed && is_stop_token(token)) {
                    begin_thinking_intervention();
                }
                if (!forced_tokens.empty()) {
                    token = forced_tokens.front();
                    forced_tokens.pop_front();
                }
                current = token;
                if (persistent_anchor_remaining != 0 &&
                    --persistent_anchor_remaining == 0 && !is_stop_token(current)) {
                    activate_persistent();
                    persistent_reset = false;
                }
            }
            if (is_stop_token(current)) {
                result.finish_reason = "stop";
                stopped_on_terminator = true;
                break;
            }
            result.tokens.push_back(current);
            if (!thinking_closed && !thinking_close_tokens.empty() &&
                current == thinking_close_tokens.front()) {
                thinking_closed = true;
            }
            if (!emit_new_tokens()) {
                result.finish_reason = "cancelled";
                break;
            }
            if (history_draft_enabled) history_draft.append(current);
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
        const bool learned_mtp_round = !used_context_copy && !used_history_draft;
        if (pending_top2_recovery_position.has_value()) {
            if (learned_mtp_round && step.accepted != 0) {
                ++result.mtp_top2_descendant_recovered_by_position[
                    *pending_top2_recovery_position];
                result.mtp_top2_descendant_accepted_by_position[
                    *pending_top2_recovery_position] += step.accepted;
            }
            pending_top2_recovery_position.reset();
        }
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
            if (learned_mtp_round && step.top2_recovered_by_position[position] != 0) {
                pending_top2_recovery_position = position;
            }
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
            if (!serial_only && persistent_backend_ != nullptr && !persistent_active) {
                activate_persistent();
            }
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
                if (prefix_cache_ != nullptr && prefix_cache_->tokens == prompt_cache_tokens)
                    prefix_cache_->ssd_backed = false;
                const bool saved = ssd_prefix_cache_->save(prompt_cache_tokens, *prompt_cache_state);
                if (prefix_cache_ != nullptr && prefix_cache_->tokens == prompt_cache_tokens)
                    prefix_cache_->ssd_backed = saved;
            }
            if (prefix_cache_ != nullptr &&
                (!prompt_cache_state.has_value() ||
                    prefix_cache_->tokens != prompt_cache_tokens)) {
                prefix_cache_->ssd_backed = false;
                prefix_cache_->ssd_backed = persist_prefix_cache(*prefix_cache_);
            }
        } catch (const std::exception& error) {
            std::cerr << "SSD prefix cache write failed: " << error.what() << '\n';
        }
    }
    // SSD is the between-request cache tier. Only discard a RAM checkpoint
    // after confirmed persistence; capacity refusals/write failures keep the
    // existing RAM fallback. Live request state owns its own shared handles.
    if (ssd_prefix_cache_ != nullptr && prefix_cache_ != nullptr && prefix_cache_->ssd_backed)
        prefix_cache_.reset();
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

bool NativeEngine::persist_prefix_cache(const PrefixCacheEntry& entry) const {
    if (ssd_prefix_cache_ == nullptr) return false;
    const PersistedPrefixState persistent = snapshot_prefix_cache(entry);
    return ssd_prefix_cache_->save(entry.tokens, persistent);
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
    struct Task {
        enum class Kind { generate, clear } kind{Kind::generate};
        NativeEngine::BatchRequest request;
        std::shared_ptr<std::promise<GenerationResult>> generation_result;
        std::shared_ptr<std::promise<void>> clear_result;
        std::shared_ptr<std::atomic_bool> resolved;
    };

    ~Impl() {
        stop();
    }

    void start(const std::filesystem::path &model_directory, NativeEngineOptions options) {
        auto started = std::make_shared<std::promise<void>>();
        std::future<void> ready = started->get_future();
        worker = std::thread([this, model_directory, options = std::move(options),
                              started]() mutable {
            try {
                engine = std::make_unique<NativeEngine>(model_directory, std::move(options));
                started->set_value();
            } catch (...) {
                started->set_exception(std::current_exception());
                return;
            }

            while (true) {
                std::vector<Task> batch;
                {
                    std::unique_lock lock(mutex);
                    ready_for_work.wait(lock, [this] { return stopping || !tasks.empty(); });
                    if (tasks.empty()) {
                        if (stopping)
                            break;
                        continue;
                    }
                    batch.push_back(std::move(tasks.front()));
                    tasks.pop_front();
                    if (batch.front().kind == Task::Kind::generate &&
                        batch.front().request.sampling.thinking_budget_tokens == 0) {
                        const auto deadline =
                            std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
                        while (batch.size() < 4) {
                            if (tasks.empty()) {
                                ready_for_work.wait_until(
                                    lock, deadline, [this] { return stopping || !tasks.empty(); });
                            }
                            if (tasks.empty() || tasks.front().kind != Task::Kind::generate ||
                                tasks.front().request.sampling.thinking_budget_tokens != 0)
                                break;
                            batch.push_back(std::move(tasks.front()));
                            tasks.pop_front();
                        }
                    }
                }
                if (batch.front().kind == Task::Kind::clear) {
                    try {
                        engine->clear_cache();
                        batch.front().clear_result->set_value();
                    } catch (...) {
                        batch.front().clear_result->set_exception(std::current_exception());
                    }
                    continue;
                }
                try {
                    if (batch.size() == 1) {
                        NativeEngine::BatchRequest &request = batch.front().request;
                        GenerationResult value =
                            request.on_delta.has_value()
                                ? engine->complete_stream(request.prompt, request.max_tokens,
                                                          *request.on_delta, request.sampling)
                                : engine->complete(request.prompt, request.max_tokens,
                                                   request.sampling);
                        batch.front().generation_result->set_value(std::move(value));
                    } else {
                        const auto bind_completion = [](Task &task) {
                            NativeEngine::BatchRequest request = std::move(task.request);
                            const auto result = task.generation_result;
                            const auto resolved = task.resolved;
                            request.on_complete = [result,
                                                   resolved](GenerationResult value) mutable {
                                if (!resolved->exchange(true))
                                    result->set_value(std::move(value));
                            };
                            request.on_error = [result, resolved](const std::exception_ptr error) {
                                if (!resolved->exchange(true))
                                    result->set_exception(error);
                            };
                            return request;
                        };
                        std::vector<NativeEngine::BatchRequest> requests;
                        requests.reserve(batch.size());
                        for (Task &task : batch) {
                            requests.push_back(bind_completion(task));
                        }
                        const NativeEngine::BatchRefillCallback refill = [this, &bind_completion] {
                            std::unique_lock lock(mutex);
                            const auto deadline =
                                std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
                            if (tasks.empty()) {
                                ready_for_work.wait_until(
                                    lock, deadline, [this] { return stopping || !tasks.empty(); });
                            }
                            if (tasks.empty() || tasks.front().kind != Task::Kind::generate ||
                                tasks.front().request.sampling.thinking_budget_tokens != 0) {
                                return std::optional<NativeEngine::BatchRequest>{};
                            }
                            Task task = std::move(tasks.front());
                            tasks.pop_front();
                            return std::optional<NativeEngine::BatchRequest>(bind_completion(task));
                        };
                        engine->complete_batch(std::move(requests), refill);
                    }
                } catch (...) {
                    const std::exception_ptr error = std::current_exception();
                    for (Task &task : batch) {
                        if (task.generation_result != nullptr && task.resolved != nullptr &&
                            !task.resolved->exchange(true)) {
                            task.generation_result->set_exception(error);
                        }
                    }
                }
            }
            engine.reset();
        });
        try {
            ready.get();
        } catch (...) {
            if (worker.joinable())
                worker.join();
            throw;
        }
    }

    void enqueue(Task task) {
        {
            std::scoped_lock lock(mutex);
            if (stopping)
                throw std::runtime_error("inference executor is stopping");
            tasks.push_back(std::move(task));
        }
        ready_for_work.notify_one();
    }

    GenerationResult generate(std::string prompt, const std::size_t max_tokens,
                              std::optional<TextDeltaCallback> on_delta, SamplingOptions sampling) {
        auto result = std::make_shared<std::promise<GenerationResult>>();
        std::future<GenerationResult> future = result->get_future();
        enqueue(Task{
            .kind = Task::Kind::generate,
            .request =
                NativeEngine::BatchRequest{
                    .prompt = std::move(prompt),
                    .max_tokens = max_tokens,
                    .on_delta = std::move(on_delta),
                    .sampling = sampling,
                },
            .generation_result = result,
            .resolved = std::make_shared<std::atomic_bool>(false),
        });
        return future.get();
    }

    void clear() {
        auto result = std::make_shared<std::promise<void>>();
        std::future<void> future = result->get_future();
        enqueue(Task{
            .kind = Task::Kind::clear,
            .clear_result = result,
        });
        future.get();
    }

    void stop() noexcept {
        {
            std::scoped_lock lock(mutex);
            stopping = true;
        }
        ready_for_work.notify_all();
        if (worker.joinable())
            worker.join();
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
    const std::size_t max_tokens,
    const SamplingOptions& sampling) {
    return impl_->generate(std::string(prompt), max_tokens, std::nullopt, sampling);
}

GenerationResult NativeEngineExecutor::complete_stream(
    const std::string_view prompt,
    const std::size_t max_tokens,
    const TextDeltaCallback& on_delta,
    const SamplingOptions& sampling) {
    return impl_->generate(std::string(prompt), max_tokens, on_delta, sampling);
}

void NativeEngineExecutor::clear_cache() {
    impl_->clear();
}

} // namespace qwen38
