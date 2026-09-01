#pragma once

#include "qwen38/gated_delta_net.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/ple.hpp"
#include "qwen38/self_attention.hpp"
#include "qwen38/sparse_moe.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace qwen38 {

struct DecoderLayerState {
    GatedDeltaNetState linear_attention;
    SelfAttentionState full_attention;
    PleState ple;
};

class DecoderLayer final {
public:
    DecoderLayer(MlxTensorStore& tensors, std::size_t layer_index, const ModelConfig& config);

    [[nodiscard]] MlxArray forward_decode(
        const MlxArray& stream,
        std::uint32_t token,
        DecoderLayerState& state) const;

    [[nodiscard]] std::size_t layer_index() const noexcept { return layer_index_; }
    [[nodiscard]] bool uses_full_attention() const noexcept {
        return full_attention_ != nullptr;
    }
    [[nodiscard]] bool uses_ple() const noexcept { return ple_ != nullptr; }

private:
    std::size_t layer_index_;
    HyperConnection attention_hyper_connection_;
    HyperConnection mlp_hyper_connection_;
    std::unique_ptr<GatedDeltaNet> linear_attention_;
    std::unique_ptr<SelfAttention> full_attention_;
    std::unique_ptr<Ple> ple_;
    SparseMoe mlp_;
};

} // namespace qwen38
