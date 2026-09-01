#pragma once

#include "qwen38/inference.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"
#include "qwen38/tokenizer.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>

namespace qwen38 {

struct NativeEngineOptions {
    // Layer-major prompt ingestion. 64 matched the retained oMLX production
    // path while keeping the temporary batch bounded on 64 GiB machines.
    std::size_t prefill_chunk_rows{64};
    // nullopt selects depth 2 when an MTP sidecar is present, otherwise serial.
    std::optional<std::size_t> mtp_depth;
    std::size_t mtp_cache_limit_bytes{256ULL * 1024ULL * 1024ULL};
    std::size_t zero_accept_fallback_rounds{2};
    bool clear_cache_each_mtp_round{true};
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
    NativeEngineOptions options_;
    MlxTensorStore tensors_;
    Tokenizer tokenizer_;
    QwenModel model_;
    std::unique_ptr<QwenMtpHead> mtp_head_;
    std::size_t mtp_depth_{0};
    std::mutex inference_mutex_;
};

} // namespace qwen38
