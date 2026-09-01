#pragma once

#include "qwen38/inference.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"
#include "qwen38/tokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace qwen38 {

struct NativeEngineOptions {
    // Layer-major prompt ingestion. 64 matched the retained oMLX production
    // path while keeping the temporary batch bounded on 64 GiB machines.
    std::size_t prefill_chunk_rows{64};
    // Retain one exact complete-state prefix. Zero disables caching.
    std::size_t prefix_cache_max_tokens{8192};
    // nullopt selects the verified adaptive depth-3 policy when an MTP sidecar
    // is present: start at 2, promote only on strong short-prompt acceptance,
    // and demote if verification becomes unprofitable.
    std::optional<std::size_t> mtp_depth;
    std::size_t mtp_cache_limit_bytes{256ULL * 1024ULL * 1024ULL};
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
    };

    NativeEngineOptions options_;
    MlxTensorStore tensors_;
    Tokenizer tokenizer_;
    QwenModel model_;
    std::unique_ptr<QwenMtpHead> mtp_head_;
    std::unique_ptr<PrefixCacheEntry> prefix_cache_;
    std::size_t mtp_depth_{0};
    std::uint32_t chat_end_token_{0};
    std::mutex inference_mutex_;
};

} // namespace qwen38
