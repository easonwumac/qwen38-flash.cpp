#include "qwen38/model.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid model dimension: ") + name);
    }
    return static_cast<int>(value);
}

} // namespace

QwenModel::QuantizedTensor QwenModel::load_quantized(
    MlxTensorStore& tensors,
    const char* prefix) {
    const std::string base(prefix);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
    };
}

QwenModel::QwenModel(MlxTensorStore& tensors)
    : hidden_size_(tensors.manifest().config().hidden_size),
      stream_count_(tensors.manifest().config().hyper_connection_count),
      vocabulary_size_(tensors.manifest().config().vocabulary_size),
      bits_(dimension(tensors.manifest().config().quantization_bits, "quantization bits")),
      group_size_(dimension(
          tensors.manifest().config().quantization_group_size, "quantization group size")),
      embedding_(load_quantized(tensors, "language_model.model.embed_tokens")),
      language_head_(load_quantized(tensors, "language_model.lm_head")),
      final_mixer_(
          tensors,
          "language_model.model.hyper_connection_mixer",
          tensors.manifest().config().hidden_size,
          tensors.manifest().config().hyper_connection_count,
          tensors.manifest().config().quantization_bits,
          tensors.manifest().config().quantization_group_size,
          static_cast<float>(tensors.manifest().config().rms_norm_epsilon),
          false) {
    const ModelConfig& config = tensors.manifest().config();
    layers_.reserve(config.layer_count);
    for (std::size_t index = 0; index < config.layer_count; ++index) {
        layers_.push_back(std::make_unique<DecoderLayer>(tensors, index, config));
    }
}

ModelDecodeState QwenModel::make_state() const {
    return ModelDecodeState(layers_.size());
}

ModelDecodeState snapshot_decode_state(const ModelDecodeState& state) {
    ModelDecodeState snapshot(state.layers.size());
    snapshot.token_count = state.token_count;
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        snapshot.layers[index] = snapshot_decoder_layer_state(state.layers[index]);
    }
    return snapshot;
}

ModelDecodeState QwenModel::snapshot_state(const ModelDecodeState& state) const {
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    return snapshot_decode_state(state);
}

MlxArray QwenModel::embed(const std::uint32_t token) const {
    if (token >= vocabulary_size_) throw std::runtime_error("token id is out of range");
    const std::vector<std::int32_t> token_values{static_cast<std::int32_t>(token)};
    const std::vector<int> token_shape{1};
    const MlxArray ids = MlxArray::from_int32(token_values, token_shape);
    MlxArray value = MlxArray::dequantize(
        MlxArray::take_axis(embedding_.weight, ids, 0),
        MlxArray::take_axis(embedding_.scales, ids, 0),
        MlxArray::take_axis(embedding_.biases, ids, 0),
        group_size_,
        bits_);
    const std::vector<int> shape{1, 1, dimension(hidden_size_, "hidden size")};
    return value.reshape(shape);
}

MlxArray QwenModel::forward_decode(const std::uint32_t token, ModelDecodeState& state) const {
    return std::move(forward_decode_capture(token, state).logits);
}

TargetDecodeStep QwenModel::forward_decode_capture(
    const std::uint32_t token,
    ModelDecodeState& state) const {
    HiddenDecodeStep hidden = forward_hidden_decode_impl(token, state, nullptr, nullptr);
    MlxArray logits = MlxArray::quantized_matmul(
        hidden.mixed,
        language_head_.weight,
        language_head_.scales,
        language_head_.biases,
        group_size_,
        bits_);
    return {
        .logits = std::move(logits),
        .pre_mixer_stream = std::move(hidden.pre_mixer_stream),
    };
}

MlxArray QwenModel::consume_decode_capture(
    const std::uint32_t token,
    ModelDecodeState& state) const {
    HiddenDecodeStep hidden =
        forward_hidden_decode_impl(token, state, nullptr, nullptr);
    hidden.pre_mixer_stream.eval();
    return std::move(hidden.pre_mixer_stream);
}

std::vector<MlxArray> QwenModel::prefill_chunk(
    const std::span<const std::uint32_t> tokens,
    ModelDecodeState& state) const {
    constexpr std::size_t max_prefill_rows = 64;
    if (tokens.empty() || tokens.size() > max_prefill_rows) {
        throw std::runtime_error("prefill chunk must contain 1 to 64 tokens");
    }
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    if (state.token_count > std::numeric_limits<std::size_t>::max() - tokens.size()) {
        throw std::runtime_error("prefill token count overflow");
    }

    std::vector<MlxArray> streams;
    streams.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
        streams.push_back(HyperConnection::initialize_stream(embed(token), stream_count_));
    }
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        std::vector<DecoderLayerState> checkpoints;
        streams = layers_[layer]->forward_verify_dense_batched(
            std::move(streams), tokens, state.layers[layer], checkpoints);
        if (checkpoints.size() != tokens.size()) {
            throw std::runtime_error("prefill layer checkpoint count mismatch");
        }
        state.layers[layer] = std::move(checkpoints.back());
        std::vector<const MlxArray*> layer_outputs;
        layer_outputs.reserve(streams.size());
        for (const MlxArray& stream : streams) layer_outputs.push_back(&stream);
        MlxArray::eval_all(layer_outputs);
    }
    state.token_count += tokens.size();
    return streams;
}

std::vector<TargetVerifyStep> QwenModel::forward_verify_layer_major_reference(
    const std::span<const std::uint32_t> tokens,
    const ModelDecodeState& origin) const {
    if (tokens.empty() || tokens.size() > 5) {
        throw std::runtime_error("target verify row count must be between 1 and 5");
    }
    if (origin.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    if (origin.token_count > std::numeric_limits<std::size_t>::max() - tokens.size()) {
        throw std::runtime_error("target verify token count overflow");
    }

    std::vector<MlxArray> streams;
    streams.reserve(tokens.size());
    std::vector<ModelDecodeState> checkpoints;
    checkpoints.reserve(tokens.size());
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        streams.push_back(HyperConnection::initialize_stream(embed(tokens[row]), stream_count_));
        checkpoints.emplace_back(layers_.size());
        checkpoints.back().token_count = origin.token_count + row + 1;
    }

    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        std::vector<DecoderLayerState> layer_checkpoints;
        streams = layers_[layer]->forward_verify_dense_batched(
            std::move(streams), tokens, origin.layers[layer], layer_checkpoints);
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            checkpoints[row].layers[layer] = std::move(layer_checkpoints[row]);
        }
        // One graph barrier per layer retains bounded memory without the S
        // extra synchronization points of evaluating each row separately.
        std::vector<const MlxArray*> layer_outputs;
        layer_outputs.reserve(streams.size());
        for (const MlxArray& stream : streams) layer_outputs.push_back(&stream);
        MlxArray::eval_all(layer_outputs);
    }

    std::vector<TargetVerifyStep> result;
    result.reserve(tokens.size());
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        HyperConnectionRead final = final_mixer_.read(streams[row]);
        MlxArray logits = MlxArray::quantized_matmul(
            final.mixed,
            language_head_.weight,
            language_head_.scales,
            language_head_.biases,
            group_size_,
            bits_);
        result.push_back({
            .logits = std::move(logits),
            .pre_mixer_stream = std::move(streams[row]),
            .state_after = std::move(checkpoints[row]),
        });
    }
    return result;
}

MlxArray QwenModel::trace_decode(
    const std::uint32_t token,
    ModelDecodeState& state,
    std::vector<double>& layer_checksums,
    std::vector<double>& layer_ms) const {
    layer_checksums.clear();
    layer_checksums.reserve(layers_.size());
    layer_ms.clear();
    layer_ms.reserve(layers_.size());
    return forward_decode_impl(token, state, &layer_checksums, &layer_ms);
}

MlxArray QwenModel::forward_decode_impl(
    const std::uint32_t token,
    ModelDecodeState& state,
    std::vector<double>* layer_checksums,
    std::vector<double>* layer_ms) const {
    HiddenDecodeStep hidden =
        forward_hidden_decode_impl(token, state, layer_checksums, layer_ms);
    return MlxArray::quantized_matmul(
        hidden.mixed,
        language_head_.weight,
        language_head_.scales,
        language_head_.biases,
        group_size_,
        bits_);
}

QwenModel::HiddenDecodeStep QwenModel::forward_hidden_decode_impl(
    const std::uint32_t token,
    ModelDecodeState& state,
    std::vector<double>* layer_checksums,
    std::vector<double>* layer_ms) const {
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    MlxArray stream = HyperConnection::initialize_stream(embed(token), stream_count_);
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const auto started = std::chrono::steady_clock::now();
        stream = layers_[index]->forward_decode(stream, token, state.layers[index]);
        if (layer_checksums != nullptr) {
            const std::vector<float> values = stream.astype(MLX_FLOAT32).to_float32();
            layer_checksums->push_back(
                std::accumulate(values.begin(), values.end(), 0.0));
        }
        if (layer_ms != nullptr) {
            layer_ms->push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
    }
    HyperConnectionRead final = final_mixer_.read(stream);
    ++state.token_count;
    return {
        .mixed = std::move(final.mixed),
        .pre_mixer_stream = std::move(stream),
    };
}

void QwenModel::consume_decode(
    const std::uint32_t token,
    ModelDecodeState& state) const {
    HiddenDecodeStep hidden = forward_hidden_decode_impl(token, state, nullptr, nullptr);
    static_cast<void>(hidden.mixed.astype(MLX_FLOAT32).to_float32());
}

GreedyStep QwenModel::greedy_decode(const std::uint32_t token, ModelDecodeState& state) const {
    MlxArray logits = forward_decode(token, state);
    if (logits.size() != vocabulary_size_) throw std::runtime_error("language head width mismatch");
    MlxArray token_array = logits.argmax_all();
    MlxArray selected_logit = MlxArray::take(logits, token_array).astype(MLX_FLOAT32);
    selected_logit.eval();
    return {
        .token = token_array.item_uint32(),
        .logit = selected_logit.item_float32(),
    };
}

} // namespace qwen38
