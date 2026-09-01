#include "qwen38/model.hpp"

#include <algorithm>
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

ModelState QwenModel::make_state() const {
    return ModelState(layers_.size());
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

MlxArray QwenModel::forward_decode(const std::uint32_t token, ModelState& state) const {
    return forward_decode_impl(token, state, nullptr);
}

MlxArray QwenModel::trace_decode(
    const std::uint32_t token,
    ModelState& state,
    std::vector<double>& layer_checksums) const {
    layer_checksums.clear();
    layer_checksums.reserve(layers_.size());
    return forward_decode_impl(token, state, &layer_checksums);
}

MlxArray QwenModel::forward_decode_impl(
    const std::uint32_t token,
    ModelState& state,
    std::vector<double>* layer_checksums) const {
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    MlxArray stream = HyperConnection::initialize_stream(embed(token), stream_count_);
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        stream = layers_[index]->forward_decode(stream, token, state.layers[index]);
        if (layer_checksums != nullptr) {
            const std::vector<float> values = stream.astype(MLX_FLOAT32).to_float32();
            layer_checksums->push_back(
                std::accumulate(values.begin(), values.end(), 0.0));
        }
    }
    HyperConnectionRead final = final_mixer_.read(stream);
    MlxArray logits = MlxArray::quantized_matmul(
        final.mixed,
        language_head_.weight,
        language_head_.scales,
        language_head_.biases,
        group_size_,
        bits_);
    ++state.token_count;
    return logits;
}

GreedyStep QwenModel::greedy_decode(const std::uint32_t token, ModelState& state) const {
    const std::vector<float> logits =
        forward_decode(token, state).astype(MLX_FLOAT32).to_float32();
    if (logits.size() != vocabulary_size_) {
        throw std::runtime_error("language head output size mismatch");
    }
    const auto best = std::ranges::max_element(logits);
    return {
        .token = static_cast<std::uint32_t>(std::distance(logits.begin(), best)),
        .logit = *best,
    };
}

} // namespace qwen38
