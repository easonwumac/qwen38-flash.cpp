#pragma once

#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <optional>
#include <vector>
#include <string>
#include <string_view>

namespace qwen38 {

class GatedDeltaNetProjectionHook {
public:
    virtual ~GatedDeltaNetProjectionHook() = default;
    // Non-owning: the hook must outlive every GatedDeltaNet that receives it.
    [[nodiscard]] virtual std::optional<MlxArray> project_prefill(
        const MlxArray& input,
        std::string_view name,
        const MlxArray& weight,
        const MlxArray& scales,
        const MlxArray& biases,
        int group_size,
        int bits) = 0;
};

struct GatedDeltaNetState {
    MlxArray convolution;
    MlxArray recurrent;
    MlxArray rollback_key;
    MlxArray rollback_value;
    MlxArray rollback_decay;
    MlxArray rollback_beta;
    std::size_t rollback_rows{0};
    bool initialized{false};
};

class GatedDeltaNet final {
public:
    GatedDeltaNet(
        MlxTensorStore& tensors,
        std::string_view prefix,
        const ModelConfig& config,
        GatedDeltaNetProjectionHook* projection_hook = nullptr);

    // Correct first-token path with zero convolution and recurrent state.
    // Stateful decode and chunked prefill are added on top of this fixture.
    [[nodiscard]] MlxArray forward_first(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_decode(
        const MlxArray& input,
        GatedDeltaNetState& state) const;
    [[nodiscard]] MlxArray forward_verify(
        const MlxArray& input,
        const GatedDeltaNetState& origin,
        std::vector<GatedDeltaNetState>& checkpoints) const;
    [[nodiscard]] MlxArray forward_prefill(
        const MlxArray& input,
        GatedDeltaNetState& state) const;
    void materialize_rollback(GatedDeltaNetState& state) const;

private:
    struct QuantizedProjection {
        std::string name;
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
    [[nodiscard]] MlxArray project_prefill(
        const MlxArray& input,
        const QuantizedProjection& projection) const;

    std::size_t key_head_count_;
    std::size_t value_head_count_;
    std::size_t key_head_dimension_;
    std::size_t value_head_dimension_;
    std::size_t convolution_kernel_size_;
    int bits_;
    int group_size_;
    float epsilon_;
    std::string output_gate_type_;
    GatedDeltaNetProjectionHook* projection_hook_;
    QuantizedProjection qkv_projection_;
    QuantizedProjection z_projection_;
    QuantizedProjection beta_projection_;
    QuantizedProjection decay_projection_;
    QuantizedProjection output_projection_;
    MlxArray convolution_weight_;
    MlxArray decay_log_;
    MlxArray decay_bias_;
    MlxArray norm_weight_;
};

} // namespace qwen38
