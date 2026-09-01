#pragma once

#include "qwen38/inference.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"
#include "qwen38/prefix_cache_store.hpp"
#include "qwen38/tokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace qwen38 {

struct NativeEngineOptions {
    // Bound request duration and incremental decode-state growth without
    // imposing an API-level 256-token product limitation. The model context
    // window remains the final per-request limit.
    std::size_t max_generation_tokens{4096};
    // Layer-major prompt ingestion. 64 matched the retained oMLX production
    // path while keeping the temporary batch bounded on 64 GiB machines.
    std::size_t prefill_chunk_rows{64};
    // Fixed chunks are useful for throughput tuning within an external memory
    // guard. Adaptive sizing remains the conservative default.
    bool adaptive_prefill_chunks{true};
    // Retain one exact complete-state prefix. Zero disables caching.
    std::size_t prefix_cache_max_tokens{8192};
    // Decoded compact-qmeta materially accelerates short prefill but competes
    // with growing attention state. Disable it automatically above this prompt
    // size on 64 GiB systems. Zero disables it for every request.
    std::size_t qmeta_cache_max_prompt_tokens{32768};
    // Zero disables persistent prefix caching. The on-disk store mirrors the
    // exact complete-state RAM prefix and evicts least-recently-used entries.
    std::uint64_t ssd_prefix_cache_max_bytes{0};
    std::filesystem::path ssd_prefix_cache_directory;
    // nullopt selects the verified adaptive depth-3 policy when an MTP sidecar
    // is present: start at 2, promote only on strong short-prompt acceptance,
    // and demote if verification becomes unprofitable.
    std::optional<std::size_t> mtp_depth;
    std::size_t allocator_cache_limit_bytes{256ULL * 1024ULL * 1024ULL};
    std::size_t zero_accept_fallback_rounds{2};
    // The allocator cache already has a strict byte cap. Per-round clearing
    // forces a GPU synchronization and is reserved for diagnostic fallback.
    bool clear_cache_each_mtp_round{false};
};

class NativeEngine final : public InferenceEngine {
public:
    explicit NativeEngine(
        const std::filesystem::path& model_directory,
        NativeEngineOptions options = {});

    [[nodiscard]] GenerationResult complete(
        std::string_view prompt,
        std::size_t max_tokens) override;
    [[nodiscard]] GenerationResult complete_stream(
        std::string_view prompt,
        std::size_t max_tokens,
        const TextDeltaCallback& on_delta) override;
    void clear_cache() override;

private:
    struct PrefixCacheEntry {
        std::vector<std::uint32_t> tokens;
        ModelDecodeState target_state;
        MtpDecodeState mtp_state;
        std::optional<MlxArray> previous_target_stream;
        std::vector<MlxArray> pending_mtp_streams;
        std::vector<std::uint32_t> pending_mtp_tokens;
        std::optional<bool> mtp_profitable;
        std::optional<std::uint32_t> mtp_profitability_current_token;
        bool mtp_cumulative_profitability_keep{false};
    };

    NativeEngineOptions options_;
    MlxTensorStore tensors_;
    Tokenizer tokenizer_;
    QwenModel model_;
    std::unique_ptr<QwenMtpHead> mtp_head_;
    std::unique_ptr<PrefixCacheEntry> prefix_cache_;
    std::unique_ptr<PrefixCacheStore> ssd_prefix_cache_;
    std::size_t mtp_depth_{0};
    std::uint32_t chat_end_token_{0};
    std::mutex inference_mutex_;

    [[nodiscard]] GenerationResult complete_impl(
        std::string_view prompt,
        std::size_t max_tokens,
        const TextDeltaCallback* on_delta);
    [[nodiscard]] PersistedPrefixState snapshot_prefix_cache(
        const PrefixCacheEntry& entry) const;
    void persist_prefix_cache(const PrefixCacheEntry& entry) const;
};

// Owns the native engine on one dedicated thread. Recent MLX releases bind
// default CPU/GPU streams to their creating thread, so model construction,
// evaluation, cache mutation, and destruction must all remain on that thread.
class NativeEngineExecutor final : public InferenceEngine {
public:
    explicit NativeEngineExecutor(
        const std::filesystem::path& model_directory,
        NativeEngineOptions options = {});
    ~NativeEngineExecutor() override;

    NativeEngineExecutor(const NativeEngineExecutor&) = delete;
    NativeEngineExecutor& operator=(const NativeEngineExecutor&) = delete;

    [[nodiscard]] GenerationResult complete(
        std::string_view prompt,
        std::size_t max_tokens) override;
    [[nodiscard]] GenerationResult complete_stream(
        std::string_view prompt,
        std::size_t max_tokens,
        const TextDeltaCallback& on_delta) override;
    void clear_cache() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen38
