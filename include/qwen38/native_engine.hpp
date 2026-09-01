#pragma once

#include "qwen38/inference.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model.hpp"
#include "qwen38/tokenizer.hpp"

#include <filesystem>
#include <mutex>

namespace qwen38 {

class NativeEngine final : public InferenceEngine {
public:
    explicit NativeEngine(const std::filesystem::path& model_directory);

    [[nodiscard]] GenerationResult complete(
        std::string_view prompt,
        std::size_t max_tokens) override;
    void clear_cache() override;

private:
    MlxTensorStore tensors_;
    Tokenizer tokenizer_;
    QwenModel model_;
    std::mutex inference_mutex_;
};

} // namespace qwen38
