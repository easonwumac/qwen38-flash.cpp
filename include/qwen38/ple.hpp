#pragma once

#include "qwen38/mlx_backend.hpp"
#include "qwen38/ngram.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace qwen38 {

struct PleState {
    NgramState ngram;
    MlxArray convolution;
    bool convolution_initialized{false};
};

class Ple final {
public:
    Ple(
        MlxTensorStore& tensors,
        std::string_view prefix,
        const ModelConfig& config);

    [[nodiscard]] MlxArray forward_decode(
        const MlxArray& stream,
        std::uint32_t token,
        PleState& state) const;

private:
    struct QuantizedProjection {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
    };

    [[nodiscard]] static QuantizedProjection load_projection(
        MlxTensorStore& tensors,
        std::string_view name);
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const QuantizedProjection& projection) const;
    [[nodiscard]] MlxArray grouped_norm(
        const MlxArray& input,
        const MlxArray& weight) const;

    std::size_t hidden_size_;
    std::size_t stream_count_;
    std::size_t convolution_state_length_;
    int bits_;
    int group_size_;
    float epsilon_;
    NgramHash hash_;
    NgramTable table_;
    QuantizedProjection key_projection_;
    QuantizedProjection value_projection_;
    MlxArray norm_key_;
    MlxArray norm_query_;
    MlxArray norm_convolution_;
    MlxArray convolution_weight_;
};

} // namespace qwen38
