#pragma once

#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace qwen38 {

struct SelfAttentionState {
    MlxArray keys;
    MlxArray values;
    MlxArray qsa_raw_keys;
    MlxArray qsa_pooled_keys;
    std::size_t qsa_pooled_count{0};
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
    struct QsaSelection {
        MlxArray dense_mask;
        MlxArray packed_indices;
        MlxArray packed_mask;
    };

    struct QuantizedProjection {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
        int bits{0};
    };

    [[nodiscard]] static QuantizedProjection load_projection(
        MlxTensorStore& tensors,
        std::string_view name,
        std::size_t group_size);
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const QuantizedProjection& projection) const;
    [[nodiscard]] MlxArray apply_rope(const MlxArray& input, std::size_t position) const;
    [[nodiscard]] MlxArray apply_rope_prefill(
        const MlxArray& input,
        std::size_t position) const;
    [[nodiscard]] MlxArray apply_rope_rows(
        const MlxArray& input,
        std::size_t position,
        std::size_t step,
        std::size_t vector_dimension) const;
    [[nodiscard]] QsaSelection update_qsa_and_build_mask(
        const MlxArray& input,
        SelfAttentionState& state,
        bool packed = false) const;
    [[nodiscard]] MlxArray packed_qsa_attention(
        const MlxArray& query,
        const MlxArray& keys,
        const MlxArray& values,
        const QsaSelection& selection) const;
    void copy_qsa_checkpoint(
        const SelfAttentionState& complete,
        std::size_t token_count,
        SelfAttentionState& checkpoint) const;

    std::size_t attention_heads_;
    std::size_t key_value_heads_;
    std::size_t head_dimension_;
    std::size_t rotary_dimension_;
    std::size_t indexer_head_count_;
    std::size_t indexer_head_dimension_;
    std::size_t indexer_budget_;
    std::size_t indexer_compress_ratio_;
    int group_size_;
    float epsilon_;
    double rope_theta_;
    QuantizedProjection query_projection_;
    QuantizedProjection key_projection_;
    QuantizedProjection value_projection_;
    QuantizedProjection output_projection_;
    QuantizedProjection indexer_projection_;
    MlxArray query_norm_weight_;
    MlxArray key_norm_weight_;
    MlxArray indexer_query_norm_weight_;
    MlxArray indexer_key_norm_weight_;
};

} // namespace qwen38
