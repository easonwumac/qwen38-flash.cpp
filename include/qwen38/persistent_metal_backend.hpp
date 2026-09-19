#pragma once

#include "qwen38/model_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace qwen38 {

struct ModelDecodeState;
class MlxTensorStore;
class MlxArray;

class PersistentMetalBackend final {
public:
    struct GreedyResult {
        std::uint32_t token{0};
        std::uint32_t alternative_token{0};
        float logit{0.0F};
        double gpu_ms{0.0};
        double wall_ms{0.0};
    };
    struct Inventory {
        std::size_t pipeline_count{0};
        std::size_t shard_count{0};
        std::size_t mapped_weight_bytes{0};
    };

    [[nodiscard]] static bool supports(const ModelConfig& config) noexcept;
    [[nodiscard]] static std::unique_ptr<PersistentMetalBackend> create(
        const ModelManifest& manifest, MlxTensorStore* shared_weights = nullptr);

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
    [[nodiscard]] std::vector<std::uint16_t> embed(
        std::uint32_t token, double* gpu_ms = nullptr);
    [[nodiscard]] GreedyResult greedy_head(std::span<const std::uint16_t> stream);
    [[nodiscard]] GreedyResult greedy_decode(std::uint32_t token, bool reset_state = false);
    void prepare_shared_weights();
    void release_shared_weights();
    void import_state(const ModelDecodeState& state);
    // Owning snapshots: later native steps never mutate an exported MLX array.
    [[nodiscard]] ModelDecodeState export_state() const;
    [[nodiscard]] MlxArray export_stream() const;
    [[nodiscard]] bool can_decode() const noexcept;

private:
    class Impl;
    explicit PersistentMetalBackend(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen38
