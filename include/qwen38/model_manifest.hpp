#pragma once

#include "qwen38/safetensors.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qwen38 {

struct ModelConfig {
    std::string architecture;
    std::string model_type;
    std::string text_model_type;
    std::size_t hidden_size{0};
    std::size_t layer_count{0};
    std::size_t expert_count{0};
    std::size_t experts_per_token{0};
    std::size_t vocabulary_size{0};
    std::size_t max_context_tokens{0};
    std::size_t quantization_bits{0};
    std::size_t quantization_group_size{0};
    std::size_t mtp_layer_count{0};
    std::size_t attention_head_count{0};
    std::size_t key_value_head_count{0};
    std::size_t head_dimension{0};
    std::size_t hyper_connection_count{0};
    std::size_t hyper_connection_low_rank{0};
    std::size_t moe_intermediate_size{0};
    std::size_t shared_expert_intermediate_size{0};
    bool normalize_topk_probability{true};
    std::size_t linear_convolution_kernel_size{0};
    std::size_t linear_key_head_dimension{0};
    std::size_t linear_value_head_dimension{0};
    std::size_t linear_key_head_count{0};
    std::size_t linear_value_head_count{0};
    std::string output_gate_type;
    double rope_theta{0.0};
    double partial_rotary_factor{0.0};
    std::size_t ngram_size{0};
    std::size_t heads_per_ngram{0};
    std::size_t ngram_vocabulary_base{0};
    std::size_t ngram_vocabulary_divisor{0};
    std::size_t ple_embedding_dimension{0};
    std::size_t ple_convolution_kernel_size{0};
    std::uint64_t ngram_seed{1234};
    std::uint32_t end_of_sequence_token{0};
    double rms_norm_epsilon{0.0};
    std::vector<std::string> layer_types;
};

class ModelManifest final {
public:
    [[nodiscard]] static ModelManifest load(const std::filesystem::path& model_directory);

    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }
    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::unordered_map<std::string, std::string>& weight_map() const noexcept {
        return weight_map_;
    }
    [[nodiscard]] std::uint64_t declared_weight_bytes() const noexcept { return declared_weight_bytes_; }

private:
    std::filesystem::path directory_;
    ModelConfig config_;
    std::unordered_map<std::string, std::string> weight_map_;
    std::uint64_t declared_weight_bytes_{0};
};

class TensorStore final {
public:
    explicit TensorStore(ModelManifest manifest) : manifest_(std::move(manifest)) {}

    [[nodiscard]] TensorView tensor(std::string_view name);
    [[nodiscard]] std::size_t open_shard_count() const;
    [[nodiscard]] std::size_t mapped_virtual_bytes() const;
    [[nodiscard]] const ModelManifest& manifest() const noexcept { return manifest_; }

private:
    ModelManifest manifest_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<SafetensorsFile>> shards_;
};

} // namespace qwen38
