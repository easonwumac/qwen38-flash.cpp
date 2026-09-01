#include "qwen38/decoder_layer.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {

MlxArray concatenate_rows(const std::vector<MlxArray>& rows) {
    if (rows.empty()) throw std::runtime_error("cannot concatenate an empty verifier batch");
    MlxArray result = rows.front().share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, rows[row], 1);
    }
    return result;
}

MlxArray slice_row(const MlxArray& batch, const std::size_t row) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() < 2 || row >= static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("verifier batch row is out of range");
    }
    std::vector<int> start(shape.size(), 0);
    std::vector<int> stop = shape;
    std::vector<int> strides(shape.size(), 1);
    start[1] = static_cast<int>(row);
    stop[1] = static_cast<int>(row + 1);
    return batch.slice(start, stop, strides);
}

} // namespace

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
    snapshot.full_attention.qsa_pooled_count =
        state.full_attention.qsa_pooled_count;
    if (state.full_attention.token_count != 0) {
        snapshot.full_attention.keys = state.full_attention.keys.share();
        snapshot.full_attention.values = state.full_attention.values.share();
        snapshot.full_attention.qsa_raw_keys =
            state.full_attention.qsa_raw_keys.share();
        if (state.full_attention.qsa_pooled_count != 0) {
            snapshot.full_attention.qsa_pooled_keys =
                state.full_attention.qsa_pooled_keys.share();
        }
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

void DecoderLayer::set_prefill_qmeta_cache_allowed(const bool allowed) const noexcept {
    mlp_.set_prefill_qmeta_cache_allowed(allowed);
}

bool DecoderLayer::clear_prefill_qmeta_cache() const {
    return mlp_.clear_prefill_qmeta_cache();
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

std::vector<MlxArray> DecoderLayer::forward_verify_dense_batched(
    std::vector<MlxArray> streams,
    const std::span<const std::uint32_t> tokens,
    const DecoderLayerState& origin,
    std::vector<DecoderLayerState>& checkpoints) const {
    if (streams.empty() || streams.size() != tokens.size() || streams.size() > 64) {
        throw std::runtime_error("decoder batch must contain 1 to 64 matching rows");
    }
    checkpoints.clear();
    checkpoints.resize(streams.size());

    if (ple_ != nullptr) {
        MlxArray stream_batch = concatenate_rows(streams);
        std::vector<PleState> ple_checkpoints;
        MlxArray ple_output = ple_->forward_verify(
            stream_batch, tokens, origin.ple, ple_checkpoints);
        stream_batch = MlxArray::add(stream_batch, ple_output);
        for (std::size_t row = 0; row < streams.size(); ++row) {
            streams[row] = slice_row(stream_batch, row);
            checkpoints[row].ple = std::move(ple_checkpoints[row]);
        }
    }

    MlxArray attention_streams = concatenate_rows(streams);
    HyperConnectionRead attention = attention_hyper_connection_.read(attention_streams);
    std::vector<MlxArray> attention_outputs;
    attention_outputs.reserve(streams.size());
    if (linear_attention_ != nullptr) {
        std::vector<GatedDeltaNetState> linear_checkpoints;
        MlxArray outputs = linear_attention_->forward_verify(
            attention.mixed, origin.linear_attention, linear_checkpoints);
        for (std::size_t row = 0; row < streams.size(); ++row) {
            attention_outputs.push_back(slice_row(outputs, row));
            checkpoints[row].linear_attention = std::move(linear_checkpoints[row]);
        }
    } else {
        std::vector<SelfAttentionState> attention_checkpoints;
        MlxArray outputs = full_attention_->forward_verify(
            attention.mixed, origin.full_attention, attention_checkpoints);
        for (std::size_t row = 0; row < streams.size(); ++row) {
            attention_outputs.push_back(slice_row(outputs, row));
            checkpoints[row].full_attention = std::move(attention_checkpoints[row]);
        }
    }
    MlxArray post_attention = attention_hyper_connection_.write(
        attention_streams, concatenate_rows(attention_outputs), attention.injection);

    HyperConnectionRead mlp = mlp_hyper_connection_.read(post_attention);
    MlxArray mlp_outputs = mlp_.forward_verify(mlp.mixed);
    MlxArray output = mlp_hyper_connection_.write(
        post_attention, mlp_outputs, mlp.injection);
    std::vector<MlxArray> result;
    result.reserve(streams.size());
    for (std::size_t row = 0; row < streams.size(); ++row) {
        result.push_back(slice_row(output, row));
    }
    return result;
}

MlxArray DecoderLayer::forward_prefill(
    MlxArray stream_batch,
    const std::span<const std::uint32_t> tokens,
    DecoderLayerState& state) const {
    const std::vector<int> shape = stream_batch.shape();
    if (tokens.empty() || tokens.size() > 512 || shape.size() != 3 || shape[0] != 1 ||
        shape[1] != static_cast<int>(tokens.size())) {
        throw std::runtime_error("decoder prefill requires a matching [1,S,streams] batch");
    }
    if (tokens.size() == 1) {
        return forward_decode(stream_batch, tokens.front(), state);
    }

    if (ple_ != nullptr) {
        std::vector<PleState> checkpoints;
        MlxArray ple_output = ple_->forward_verify(
            stream_batch, tokens, state.ple, checkpoints);
        stream_batch = MlxArray::add(stream_batch, ple_output);
        state.ple = std::move(checkpoints.back());
    }

    HyperConnectionRead attention = attention_hyper_connection_.read(stream_batch);
    MlxArray attention_output;
    if (linear_attention_ != nullptr) {
        attention_output = linear_attention_->forward_prefill(
            attention.mixed, state.linear_attention);
    } else {
        attention_output = full_attention_->forward_prefill(
            attention.mixed, state.full_attention);
    }
    MlxArray post_attention = attention_hyper_connection_.write(
        stream_batch, attention_output, attention.injection);
    HyperConnectionRead mlp = mlp_hyper_connection_.read(post_attention);
    MlxArray mlp_output = mlp_.forward_prefill(mlp.mixed);
    return mlp_hyper_connection_.write(
        post_attention, mlp_output, mlp.injection);
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
