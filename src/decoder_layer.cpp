#include "qwen38/decoder_layer.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {

DecoderLayerState snapshot_decoder_layer_state(const DecoderLayerState& state) {
    DecoderLayerState snapshot;
    snapshot.linear_attention.initialized = state.linear_attention.initialized;
    if (state.linear_attention.initialized) {
        snapshot.linear_attention.convolution =
            state.linear_attention.convolution.share();
        snapshot.linear_attention.recurrent = state.linear_attention.recurrent.share();
    }
    snapshot.full_attention.token_count = state.full_attention.token_count;
    snapshot.full_attention.position_base = state.full_attention.position_base;
    if (state.full_attention.token_count != 0) {
        snapshot.full_attention.keys = state.full_attention.keys.share();
        snapshot.full_attention.values = state.full_attention.values.share();
    }
    snapshot.ple.ngram = state.ple.ngram;
    snapshot.ple.convolution_initialized = state.ple.convolution_initialized;
    if (state.ple.convolution_initialized) {
        snapshot.ple.convolution = state.ple.convolution.share();
    }
    return snapshot;
}

DecoderLayer::DecoderLayer(
    MlxTensorStore& tensors,
    const std::size_t layer_index,
    const ModelConfig& config)
    : DecoderLayer(
          tensors,
          "language_model.model.layers." + std::to_string(layer_index),
          layer_index,
          config) {}

DecoderLayer::DecoderLayer(
    MlxTensorStore& tensors,
    std::string prefix,
    const std::size_t layer_index,
    const ModelConfig& config)
    : layer_index_(layer_index),
      attention_hyper_connection_(
          tensors,
          prefix + ".attn_hyper_connection",
          config.hidden_size,
          config.hyper_connection_count,
          config.quantization_bits,
          config.quantization_group_size,
          static_cast<float>(config.rms_norm_epsilon),
          true),
      mlp_hyper_connection_(
          tensors,
          prefix + ".mlp_hyper_connection",
          config.hidden_size,
          config.hyper_connection_count,
          config.quantization_bits,
          config.quantization_group_size,
          static_cast<float>(config.rms_norm_epsilon),
          true),
      mlp_(
          tensors,
          prefix + ".mlp",
          config.expert_count,
          config.experts_per_token,
          config.quantization_bits,
          config.quantization_group_size,
          config.normalize_topk_probability) {
    if (layer_index >= config.layer_count || config.layer_types.size() != config.layer_count) {
        throw std::runtime_error("decoder layer index is out of range");
    }
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

DecoderLayer::~DecoderLayer() {
    if (compiled_.ctx != nullptr) static_cast<void>(mlx_closure_free(compiled_));
}

MlxArray DecoderLayer::forward_decode(
    const MlxArray& input_stream,
    const std::uint32_t token,
    DecoderLayerState& state,
    DecoderLayerTrace* trace) const {
    const char* compile = std::getenv("QWEN38_COMPILE_LAYER");
    if (linear_attention_ != nullptr && ple_ == nullptr && state.linear_attention.initialized &&
        trace == nullptr && compile != nullptr && std::string_view(compile) == "1") {
        ensure_compiled();
        return apply_compiled(input_stream, state);
    }
    return forward_decode_graph(input_stream, token, state, trace);
}

MlxArray DecoderLayer::forward_decode_graph(
    const MlxArray& input_stream,
    const std::uint32_t token,
    DecoderLayerState& state,
    DecoderLayerTrace* trace) const {
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
    if (trace != nullptr) {
        const std::vector<int> zero_shape{1};
        trace->attention_mixed = MlxArray::add(
            attention.mixed, MlxArray::zeros(zero_shape, attention.mixed.dtype()));
        trace->attention_injection = MlxArray::add(
            attention.injection, MlxArray::zeros(zero_shape, attention.injection.dtype()));
        trace->attention_output = MlxArray::add(
            attention_output, MlxArray::zeros(zero_shape, attention_output.dtype()));
    }
    stream = attention_hyper_connection_.write(
        stream, attention_output, attention.injection);
    if (trace != nullptr) {
        const std::vector<int> zero_shape{1};
        trace->post_attention_stream = MlxArray::add(
            stream, MlxArray::zeros(zero_shape, stream.dtype()));
    }

    HyperConnectionRead mlp = mlp_hyper_connection_.read(stream);
    MlxArray mlp_output = mlp_.forward_decode(mlp.mixed);
    if (trace != nullptr) {
        const std::vector<int> zero_shape{1};
        trace->mlp_mixed = MlxArray::add(
            mlp.mixed, MlxArray::zeros(zero_shape, mlp.mixed.dtype()));
        trace->mlp_injection = MlxArray::add(
            mlp.injection, MlxArray::zeros(zero_shape, mlp.injection.dtype()));
        trace->mlp_output = MlxArray::add(
            mlp_output, MlxArray::zeros(zero_shape, mlp_output.dtype()));
    }
    MlxArray output = mlp_hyper_connection_.write(stream, mlp_output, mlp.injection);
    if (trace != nullptr) {
        const std::vector<int> zero_shape{1};
        trace->post_mlp_stream = MlxArray::add(
            output, MlxArray::zeros(zero_shape, output.dtype()));
    }
    return output;
}

int DecoderLayer::compile_callback(
    mlx_vector_array* outputs,
    const mlx_vector_array inputs,
    void* payload) {
    if (mlx_vector_array_size(inputs) != 3) return 1;
    mlx_array raw_stream = mlx_array_new();
    mlx_array raw_convolution = mlx_array_new();
    mlx_array raw_recurrent = mlx_array_new();
    if (mlx_vector_array_get(&raw_stream, inputs, 0) != 0 ||
        mlx_vector_array_get(&raw_convolution, inputs, 1) != 0 ||
        mlx_vector_array_get(&raw_recurrent, inputs, 2) != 0) {
        mlx_array_free(raw_stream);
        mlx_array_free(raw_convolution);
        mlx_array_free(raw_recurrent);
        return 1;
    }
    MlxArray stream(raw_stream);
    DecoderLayerState state;
    state.linear_attention = {
        .convolution = MlxArray(raw_convolution),
        .recurrent = MlxArray(raw_recurrent),
        .initialized = true,
    };
    const auto* self = static_cast<const DecoderLayer*>(payload);
    MlxArray output = self->forward_decode_graph(stream, 0, state, nullptr);
    const mlx_array values[]{
        output.get(),
        state.linear_attention.convolution.get(),
        state.linear_attention.recurrent.get(),
    };
    return mlx_vector_array_set_data(outputs, values, 3);
}

void DecoderLayer::ensure_compiled() const {
    std::call_once(compile_once_, [this]() {
        mlx_closure function = mlx_closure_new_func_payload(
            &DecoderLayer::compile_callback,
            const_cast<DecoderLayer*>(this),
            nullptr);
        compiled_ = mlx_closure_new();
        const int status = mlx_compile(&compiled_, function, false);
        static_cast<void>(mlx_closure_free(function));
        if (status != 0) throw std::runtime_error("MLX failed to compile decoder layer closure");
    });
}

MlxArray DecoderLayer::apply_compiled(
    const MlxArray& stream,
    DecoderLayerState& state) const {
    const mlx_array values[]{
        stream.get(),
        state.linear_attention.convolution.get(),
        state.linear_attention.recurrent.get(),
    };
    mlx_vector_array inputs = mlx_vector_array_new_data(values, 3);
    mlx_vector_array outputs = mlx_vector_array_new();
    const int status = mlx_closure_apply(&outputs, compiled_, inputs);
    static_cast<void>(mlx_vector_array_free(inputs));
    if (status != 0 || mlx_vector_array_size(outputs) != 3) {
        static_cast<void>(mlx_vector_array_free(outputs));
        throw std::runtime_error("compiled decoder layer invocation failed");
    }
    mlx_array raw_output = mlx_array_new();
    mlx_array raw_convolution = mlx_array_new();
    mlx_array raw_recurrent = mlx_array_new();
    const int get_status =
        mlx_vector_array_get(&raw_output, outputs, 0) |
        mlx_vector_array_get(&raw_convolution, outputs, 1) |
        mlx_vector_array_get(&raw_recurrent, outputs, 2);
    static_cast<void>(mlx_vector_array_free(outputs));
    if (get_status != 0) {
        mlx_array_free(raw_output);
        mlx_array_free(raw_convolution);
        mlx_array_free(raw_recurrent);
        throw std::runtime_error("compiled decoder layer returned invalid outputs");
    }
    state.linear_attention.convolution = MlxArray(raw_convolution);
    state.linear_attention.recurrent = MlxArray(raw_recurrent);
    return MlxArray(raw_output);
}

} // namespace qwen38
