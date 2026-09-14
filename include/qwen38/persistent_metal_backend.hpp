#pragma once

#include "qwen38/model_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace qwen38 {

class PersistentMetalBackend final {
public:
    struct Inventory {
        std::size_t pipeline_count{0};
        std::size_t shard_count{0};
        std::size_t mapped_weight_bytes{0};
    };

    [[nodiscard]] static bool supports(const ModelConfig& config) noexcept;
    [[nodiscard]] static std::unique_ptr<PersistentMetalBackend> create(
        const ModelManifest& manifest);

    ~PersistentMetalBackend();
    PersistentMetalBackend(PersistentMetalBackend&&) noexcept;
    PersistentMetalBackend& operator=(PersistentMetalBackend&&) noexcept;

    PersistentMetalBackend(const PersistentMetalBackend&) = delete;
    PersistentMetalBackend& operator=(const PersistentMetalBackend&) = delete;

    [[nodiscard]] const Inventory& inventory() const noexcept;
    [[nodiscard]] std::vector<std::uint16_t> decode_gdn_layer(
        std::size_t layer,
        std::span<const std::uint16_t> stream,
        bool reset_state = false,
        double* gpu_ms = nullptr);
    [[nodiscard]] std::vector<std::uint16_t> decode_attention_layer(
        std::size_t layer,
        std::span<const std::uint16_t> stream,
        bool reset_state = false,
        double* gpu_ms = nullptr);
    [[nodiscard]] std::vector<std::uint16_t> decode_trunk(
        std::uint32_t token,
        std::span<const std::uint16_t> initial_stream,
        bool reset_state = false,
        double* gpu_ms = nullptr);
    [[nodiscard]] std::vector<std::uint16_t> decode_ple(
        std::uint32_t token,
        std::span<const std::uint16_t> stream,
        bool reset_state = false,
        double* gpu_ms = nullptr);

private:
    class Impl;
    explicit PersistentMetalBackend(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen38
