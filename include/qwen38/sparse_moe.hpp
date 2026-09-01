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

struct MoePrefillTimings {
    double routing_ms{0.0};
    double gate_up_ms{0.0};
    double gate_qmm_ms{0.0};
    double up_qmm_ms{0.0};
    double swiglu_ms{0.0};
    double down_reduce_ms{0.0};
    double down_qmm_ms{0.0};
    double route_reduce_ms{0.0};
    double shared_expert_ms{0.0};
    double merge_ms{0.0};
};

struct MoeVerifyTimings {
    double routing_ms{0.0};
    double gate_up_ms{0.0};
    double down_ms{0.0};
    double shared_expert_ms{0.0};
    double merge_ms{0.0};
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
    [[nodiscard]] MlxArray forward_verify(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_verify_profiled(
        const MlxArray& input,
        MoeVerifyTimings& timings) const;
    [[nodiscard]] MlxArray forward_prefill(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_prefill_profiled(
        const MlxArray& input,
        MoePrefillTimings& timings) const;
    [[nodiscard]] bool clear_prefill_qmeta_cache() const;

private:
    struct CompactQmeta {
        MlxArray tags;
        MlxArray dictionary;
        int bits{0};
        int groups{0};
        int row_bytes{0};

        [[nodiscard]] bool present() const noexcept { return bits != 0; }
    };

    struct DecodedQmeta {
        MlxArray scales;
        MlxArray biases;
    };

    struct QuantizedProjection {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
        CompactQmeta qmeta;
        mutable DecodedQmeta cached_qmeta;
        mutable bool qmeta_cached{false};
        int bits{0};
    };

    [[nodiscard]] static QuantizedProjection load_projection(
        MlxTensorStore& tensors,
        std::string_view name,
        std::size_t group_size);
    static void make_resident(QuantizedProjection& projection);
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const QuantizedProjection& projection) const;
    [[nodiscard]] MlxArray project_expert(
        const MlxArray& input,
        const QuantizedProjection& projection,
        std::size_t expert) const;
    [[nodiscard]] MlxArray forward_experts_decode(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_compact_routed(
        const MlxArray& input,
        const MlxArray& experts,
        const MlxArray& weights) const;
    [[nodiscard]] static DecodedQmeta decode_qmeta(const QuantizedProjection& projection);
    [[nodiscard]] MlxArray forward_shared(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_verify_impl(
        const MlxArray& input,
        MoeVerifyTimings* timings) const;
    [[nodiscard]] MlxArray forward_prefill_impl(
        const MlxArray& input,
        MoePrefillTimings* timings) const;

    std::size_t expert_count_;
    std::size_t experts_per_token_;
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
    bool fused_q8_exact_{false};
    bool compact_qmeta_{false};
};

} // namespace qwen38
