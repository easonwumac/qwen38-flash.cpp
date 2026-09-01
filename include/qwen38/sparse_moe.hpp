#pragma once

#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace qwen38 {

struct RouterSelection {
    std::vector<std::size_t> experts;
    std::vector<float> weights;
};

class SparseMoe final {
public:
    SparseMoe(
        MlxTensorStore& tensors,
        std::string_view prefix,
        std::size_t expert_count,
        std::size_t experts_per_token,
        std::size_t quantization_bits,
        std::size_t quantization_group_size,
        bool normalize_topk_probability);

    [[nodiscard]] RouterSelection route_decode(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_decode(const MlxArray& input) const;

private:
    struct QuantizedProjection {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
    };

    [[nodiscard]] static QuantizedProjection load_projection(
        MlxTensorStore& tensors,
        std::string_view name);
    static void make_resident(QuantizedProjection& projection);
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const QuantizedProjection& projection) const;
    [[nodiscard]] MlxArray project_expert(
        const MlxArray& input,
        const QuantizedProjection& projection,
        std::size_t expert) const;

    std::size_t expert_count_;
    std::size_t experts_per_token_;
    int bits_;
    int group_size_;
    bool normalize_topk_probability_;
    MlxArray router_weight_;
    QuantizedProjection expert_gate_;
    QuantizedProjection expert_up_;
    QuantizedProjection expert_down_;
    QuantizedProjection shared_gate_;
    QuantizedProjection shared_up_;
    QuantizedProjection shared_down_;
    MlxArray shared_router_weight_;
    std::shared_ptr<MlxMetalKernel> fused_gate_up_;
    std::shared_ptr<MlxMetalKernel> fused_down_;
};

} // namespace qwen38
