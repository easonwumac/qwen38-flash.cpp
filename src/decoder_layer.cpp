#include "qwen38/decoder_layer.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {

DecoderLayer::DecoderLayer(
    MlxTensorStore& tensors,
    const std::size_t layer_index,
    const ModelConfig& config)
    : layer_index_(layer_index),
      attention_hyper_connection_(
          tensors,
          "language_model.model.layers." + std::to_string(layer_index) +
              ".attn_hyper_connection",
          config.hidden_size,
          config.hyper_connection_count,
          config.quantization_bits,
          config.quantization_group_size,
          static_cast<float>(config.rms_norm_epsilon),
          true),
      mlp_hyper_connection_(
          tensors,
          "language_model.model.layers." + std::to_string(layer_index) +
              ".mlp_hyper_connection",
          config.hidden_size,
          config.hyper_connection_count,
          config.quantization_bits,
          config.quantization_group_size,
          static_cast<float>(config.rms_norm_epsilon),
          true),
      mlp_(
          tensors,
          "language_model.model.layers." + std::to_string(layer_index) + ".mlp",
          config.expert_count,
          config.experts_per_token,
          config.quantization_bits,
          config.quantization_group_size,
          config.normalize_topk_probability) {
    if (layer_index >= config.layer_count || config.layer_types.size() != config.layer_count) {
        throw std::runtime_error("decoder layer index is out of range");
    }
    const std::string prefix = "language_model.model.layers." + std::to_string(layer_index);
    if (config.layer_types[layer_index] == "linear_attention") {
        linear_attention_ = std::make_unique<GatedDeltaNet>(
            tensors, prefix + ".linear_attn", config);
    } else if (config.layer_types[layer_index] == "full_attention") {
        full_attention_ = std::make_unique<SelfAttention>(
            tensors, prefix + ".self_attn", config);
    } else {
        throw std::runtime_error("unsupported decoder attention type");
    }
    if (std::ranges::find(config.ple_layer_ids, layer_index + 1) !=
        config.ple_layer_ids.end()) {
        ple_ = std::make_unique<Ple>(tensors, prefix + ".ple", config);
    }
}

MlxArray DecoderLayer::forward_decode(
    const MlxArray& input_stream,
    const std::uint32_t token,
    DecoderLayerState& state) const {
    MlxArray stream;
    if (ple_ != nullptr) {
        stream = MlxArray::add(
            input_stream, ple_->forward_decode(input_stream, token, state.ple));
    } else {
        const std::vector<int> zero_shape{1};
        // MLX arrays are move-only. Adding a broadcast zero preserves the graph
        // value while giving this layer ownership of its working stream.
        stream = MlxArray::add(
            input_stream, MlxArray::zeros(zero_shape, input_stream.dtype()));
    }

    HyperConnectionRead attention = attention_hyper_connection_.read(stream);
    MlxArray attention_output = linear_attention_ != nullptr
        ? linear_attention_->forward_decode(attention.mixed, state.linear_attention)
        : full_attention_->forward_decode(attention.mixed, state.full_attention);
    stream = attention_hyper_connection_.write(
        stream, attention_output, attention.injection);

    HyperConnectionRead mlp = mlp_hyper_connection_.read(stream);
    MlxArray mlp_output = mlp_.forward_decode(mlp.mixed);
    return mlp_hyper_connection_.write(stream, mlp_output, mlp.injection);
}

} // namespace qwen38
