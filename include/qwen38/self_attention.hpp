#pragma once

#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace qwen38 {

struct SelfAttentionState {
    MlxArray keys;
    MlxArray values;
    std::size_t token_count{0};
    std::size_t position_base{0};
};

class SelfAttention final {
public:
    SelfAttention(
        MlxTensorStore& tensors,
        std::string_view prefix,
        const ModelConfig& config);

    [[nodiscard]] MlxArray forward_decode(
        const MlxArray& input,
        SelfAttentionState& state) const;
    [[nodiscard]] MlxArray forward_verify(
        const MlxArray& input,
        const SelfAttentionState& origin,
        std::vector<SelfAttentionState>& checkpoints) const;
    [[nodiscard]] MlxArray forward_prefill(
        const MlxArray& input,
        SelfAttentionState& state) const;

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
    [[nodiscard]] MlxArray apply_rope(const MlxArray& input, std::size_t position) const;
    [[nodiscard]] MlxArray apply_rope_prefill(
        const MlxArray& input,
        std::size_t position) const;

    std::size_t attention_heads_;
    std::size_t key_value_heads_;
    std::size_t head_dimension_;
    std::size_t rotary_dimension_;
    int bits_;
    int group_size_;
    float epsilon_;
    double rope_theta_;
    QuantizedProjection query_projection_;
    QuantizedProjection key_projection_;
    QuantizedProjection value_projection_;
    QuantizedProjection output_projection_;
    MlxArray query_norm_weight_;
    MlxArray key_norm_weight_;
};

} // namespace qwen38
