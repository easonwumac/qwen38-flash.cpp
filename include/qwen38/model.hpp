#pragma once

#include "qwen38/decoder_layer.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace qwen38 {

struct ModelDecodeState {
    explicit ModelDecodeState(std::size_t layer_count) : layers(layer_count) {}
    std::vector<DecoderLayerState> layers;
    std::size_t token_count{0};
};

struct GreedyStep {
    std::uint32_t token{0};
    float logit{0.0F};
};

// The Qwen3.8 MTP companion consumes the four-stream residual before the
// target's final mixer. Keeping it beside the logits avoids a second trunk
// forward and prevents callers from accidentally feeding the collapsed hidden.
struct TargetDecodeStep {
    MlxArray logits;
    MlxArray pre_mixer_stream;
};

class QwenModel final {
public:
    explicit QwenModel(MlxTensorStore& tensors);

    [[nodiscard]] ModelDecodeState make_state() const;
    [[nodiscard]] MlxArray forward_decode(std::uint32_t token, ModelDecodeState& state) const;
    [[nodiscard]] TargetDecodeStep forward_decode_capture(
        std::uint32_t token,
        ModelDecodeState& state) const;
    void consume_decode(std::uint32_t token, ModelDecodeState& state) const;
    [[nodiscard]] MlxArray trace_decode(
        std::uint32_t token,
        ModelDecodeState& state,
        std::vector<double>& layer_checksums,
        std::vector<double>& layer_ms) const;
    [[nodiscard]] GreedyStep greedy_decode(std::uint32_t token, ModelDecodeState& state) const;
    [[nodiscard]] std::size_t layer_count() const noexcept { return layers_.size(); }

private:
    struct HiddenDecodeStep {
        MlxArray mixed;
        MlxArray pre_mixer_stream;
    };

    struct QuantizedTensor {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
    };

    [[nodiscard]] static QuantizedTensor load_quantized(
        MlxTensorStore& tensors,
        const char* prefix);
    [[nodiscard]] MlxArray embed(std::uint32_t token) const;
    [[nodiscard]] MlxArray forward_decode_impl(
        std::uint32_t token,
        ModelDecodeState& state,
        std::vector<double>* layer_checksums,
        std::vector<double>* layer_ms) const;
    [[nodiscard]] HiddenDecodeStep forward_hidden_decode_impl(
        std::uint32_t token,
        ModelDecodeState& state,
        std::vector<double>* layer_checksums,
        std::vector<double>* layer_ms) const;

    std::size_t hidden_size_;
    std::size_t stream_count_;
    std::size_t vocabulary_size_;
    int bits_;
    int group_size_;
    QuantizedTensor embedding_;
    QuantizedTensor language_head_;
    std::vector<std::unique_ptr<DecoderLayer>> layers_;
    HyperConnection final_mixer_;
};

} // namespace qwen38
