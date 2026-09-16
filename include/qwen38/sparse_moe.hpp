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

struct MoeDecodeTimings {
    double routing_ms{0.0};
    double gate_up_ms{0.0};
    double down_reduce_ms{0.0};
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
    [[nodiscard]] MlxArray forward_decode_profiled(
        const MlxArray& input,
        MoeDecodeTimings& timings) const;
    [[nodiscard]] std::vector<MlxArray> forward_decode_multi(
        const std::vector<MlxArray>& inputs) const;
    [[nodiscard]] MlxArray forward_verify(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_verify_profiled(
        const MlxArray& input,
        MoeVerifyTimings& timings) const;
    [[nodiscard]] MlxArray forward_prefill(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_prefill_profiled(
        const MlxArray& input,
        MoePrefillTimings& timings) const;
    void set_prefill_qmeta_cache_allowed(bool allowed) const noexcept;
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
        MlxArray codebook;
        MlxArray codebook_u8;
        MlxArray codebook_u8_scales;
        MlxArray codebook_u8_biases;
        CompactQmeta qmeta;
        mutable DecodedQmeta cached_qmeta;
        mutable bool qmeta_cached{false};
        int bits{0};
        int group_size{0};
        int input_dimension{0};
        int output_dimension{0};
        int vector_dimension{0};
        int packed_bits{0};
        bool vector_quantized{false};
        bool codebook_u8_ready{false};
    };

    struct LinearProjection {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
        int bits{0};
        int group_size{0};
        bool quantized{false};
    };

    [[nodiscard]] static QuantizedProjection load_projection(
        MlxTensorStore& tensors,
        std::string_view name);
    [[nodiscard]] static LinearProjection load_linear(
        MlxTensorStore& tensors,
        std::string_view name);
    static void make_resident(QuantizedProjection& projection);
    static void prepare_u8_codebook(QuantizedProjection& projection);
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const QuantizedProjection& projection) const;
    [[nodiscard]] static MlxArray project_linear(
        const MlxArray& input,
        const LinearProjection& projection);
    [[nodiscard]] MlxArray project_expert(
        const MlxArray& input,
        const QuantizedProjection& projection,
        std::size_t expert) const;
    [[nodiscard]] static MlxArray decode_vector_quantized_expert(
        const QuantizedProjection& projection,
        const MlxArray& expert);
    [[nodiscard]] MlxArray forward_experts_decode(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_experts_decode_profiled(
        const MlxArray& input,
        MoeDecodeTimings* timings) const;
    [[nodiscard]] MlxArray forward_paged(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_paged_grouped(const MlxArray& input) const;
    [[nodiscard]] MlxArray forward_paged_packed(const MlxArray& input) const;
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

    std::size_t layer_index_;
    std::size_t expert_count_;
    std::size_t experts_per_token_;
    int group_size_;
    bool normalize_topk_probability_;
    MlxTensorStore* paged_store_;
    std::string prefix_;
    bool has_routed_{true};
    LinearProjection router_;
    QuantizedProjection expert_gate_;
    QuantizedProjection expert_up_;
    QuantizedProjection expert_down_;
    QuantizedProjection shared_gate_;
    QuantizedProjection shared_up_;
    QuantizedProjection shared_down_;
    LinearProjection shared_router_;
    std::shared_ptr<MlxMetalKernel> fused_gate_up_;
    std::shared_ptr<MlxMetalKernel> fused_down_;
    bool fused_q8_exact_{false};
    bool fused_vq_{false};
    bool compact_qmeta_{false};
    mutable bool prefill_qmeta_cache_allowed_{true};
};

} // namespace qwen38
