#pragma once

#include "qwen38/gated_delta_net.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/ple.hpp"
#include "qwen38/self_attention.hpp"
#include "qwen38/sparse_moe.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace qwen38 {

struct DecoderLayerState {
    GatedDeltaNetState linear_attention;
    SelfAttentionState full_attention;
    PleState ple;
};

[[nodiscard]] DecoderLayerState snapshot_decoder_layer_state(
    const DecoderLayerState& state);

struct DecoderLayerTrace {
    MlxArray attention_mixed;
    MlxArray attention_injection;
    MlxArray attention_output;
    MlxArray post_attention_stream;
    MlxArray mlp_mixed;
    MlxArray mlp_injection;
    MlxArray mlp_output;
    MlxArray post_mlp_stream;
};

class DecoderLayer final {
public:
    DecoderLayer(MlxTensorStore& tensors, std::size_t layer_index, const ModelConfig& config);
    DecoderLayer(
        MlxTensorStore& tensors,
        std::string prefix,
        std::size_t layer_index,
        const ModelConfig& config);
    ~DecoderLayer();

    [[nodiscard]] MlxArray forward_decode(
        const MlxArray& stream,
        std::uint32_t token,
        DecoderLayerState& state,
        DecoderLayerTrace* trace = nullptr) const;

    [[nodiscard]] std::size_t layer_index() const noexcept { return layer_index_; }
    [[nodiscard]] bool uses_full_attention() const noexcept {
        return full_attention_ != nullptr;
    }
    [[nodiscard]] bool uses_ple() const noexcept { return ple_ != nullptr; }

private:
    [[nodiscard]] MlxArray forward_decode_graph(
        const MlxArray& stream,
        std::uint32_t token,
        DecoderLayerState& state,
        DecoderLayerTrace* trace) const;
    void ensure_compiled() const;
    [[nodiscard]] MlxArray apply_compiled(
        const MlxArray& stream,
        DecoderLayerState& state) const;
    static int compile_callback(
        mlx_vector_array* outputs,
        mlx_vector_array inputs,
        void* payload);

    std::size_t layer_index_;
    HyperConnection attention_hyper_connection_;
    HyperConnection mlp_hyper_connection_;
    std::unique_ptr<GatedDeltaNet> linear_attention_;
    std::unique_ptr<SelfAttention> full_attention_;
    std::unique_ptr<Ple> ple_;
    SparseMoe mlp_;
    mutable std::once_flag compile_once_;
    mutable mlx_closure compiled_{};
};

} // namespace qwen38
