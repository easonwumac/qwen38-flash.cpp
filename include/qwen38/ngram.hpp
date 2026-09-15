#pragma once

#include "qwen38/model_manifest.hpp"
#include "qwen38/safetensors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace qwen38 {

struct NgramState {
    std::array<std::uint32_t, 2> previous{};
    std::size_t segment_length{0};
    bool initialized{false};
};

class NgramHash final {
public:
    explicit NgramHash(const ModelConfig& config);

    [[nodiscard]] std::array<std::int64_t, 16> row_ids(
        std::uint32_t token,
        NgramState& state) const;
    [[nodiscard]] std::uint64_t total_rows() const noexcept { return total_rows_; }

private:
    std::uint32_t eos_;
    std::array<std::int64_t, 3> multipliers_{};
    std::array<std::int64_t, 16> vocabulary_{};
    std::array<std::int64_t, 16> offsets_{};
    std::uint64_t total_rows_{0};
};

class NgramTable final {
public:
    NgramTable(
        const std::filesystem::path& model_directory,
        std::uint64_t rows,
        bool prefer_aos = true);
    ~NgramTable();

    NgramTable(const NgramTable&) = delete;
    NgramTable& operator=(const NgramTable&) = delete;

    [[nodiscard]] std::vector<float> gather(
        std::span<const std::int64_t> row_ids) const;
    [[nodiscard]] bool uses_aos() const noexcept {
        return aos_fd_ >= 0 || bf16_aos_fd_ >= 0;
    }
    [[nodiscard]] bool uses_q8_aos() const noexcept {
        return aos_fd_ >= 0 && bits_ == 8;
    }
    [[nodiscard]] bool uses_bf16_aos() const noexcept { return bf16_aos_fd_ >= 0; }
    [[nodiscard]] bool uses_paired_shards() const noexcept { return paired_store_ != nullptr; }
    [[nodiscard]] std::size_t row_dimension() const noexcept { return dimension_; }

private:
    struct PairedShard {
        std::uint64_t logical_begin{0};
        std::uint64_t logical_end{0};
        TensorView weight;
        TensorView scales;
        TensorView biases;
    };

    void initialize_paired(ModelManifest manifest);
    void decode_row(std::span<const std::byte> packed, std::span<float> output) const;
    void decode_paired_row(
        const PairedShard& shard,
        std::uint64_t logical_row,
        std::span<float> output) const;

    std::uint64_t rows_;
    std::size_t dimension_{160};
    std::size_t packed_word_count_{20};
    std::size_t scale_count_{5};
    std::size_t group_size_{32};
    std::size_t bits_{4};
    std::size_t pair_{1};
    int aos_fd_{-1};
    int bf16_aos_fd_{-1};
    std::unique_ptr<SafetensorsFile> fallback_;
    TensorView fallback_weight_;
    TensorView fallback_scales_;
    TensorView fallback_biases_;
    std::unique_ptr<TensorStore> paired_store_;
    std::vector<PairedShard> paired_shards_;
};

} // namespace qwen38
