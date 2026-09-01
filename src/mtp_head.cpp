#include "qwen38/mtp_head.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid MTP dimension: ") + name);
    }
    return static_cast<int>(value);
}

MlxArray offset_norm(MlxArray raw, const std::size_t width) {
    const std::vector<float> values(width, 1.0F);
    const std::vector<int> shape{dimension(width, "norm width")};
    return MlxArray::add(raw, MlxArray::from_float32(values, shape).astype(raw.dtype()));
}

ModelConfig mtp_layer_config(const ModelConfig& target) {
    ModelConfig result = target;
    result.layer_count = 1;
    result.layer_types = {"full_attention"};
    result.ple_layer_ids.clear();
    result.quantization_bits = 8;
    return result;
}

} // namespace

QwenMtpHead::QuantizedProjection QwenMtpHead::load_projection(
    MlxTensorStore& tensors,
    const char* prefix,
    const int bits) {
    const std::string base(prefix);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
        .bits = bits,
    };
}

QwenMtpHead::QwenMtpHead(MlxTensorStore& tensors)
    : hidden_size_(tensors.manifest().config().hidden_size),
      stream_count_(tensors.manifest().config().hyper_connection_count),
      vocabulary_size_(tensors.manifest().config().vocabulary_size),
      group_size_(dimension(
          tensors.manifest().config().quantization_group_size,
          "quantization group size")),
      epsilon_(static_cast<float>(tensors.manifest().config().rms_norm_epsilon)),
      embedding_(load_projection(
          tensors,
          "language_model.model.embed_tokens",
          dimension(tensors.manifest().config().quantization_bits, "target bits"))),
      language_head_(load_projection(
          tensors,
          "language_model.lm_head",
          dimension(tensors.manifest().config().quantization_bits, "target bits"))),
      fc_embedding_(load_projection(tensors, "language_model.mtp.fc_embedding", 8)),
      fc_hidden_(load_projection(tensors, "language_model.mtp.fc_hidden", 8)),
      embedding_norm_(offset_norm(
          tensors.tensor("language_model.mtp.pre_fc_norm_embedding.weight"), hidden_size_)),
      hidden_norm_(offset_norm(
          tensors.tensor("language_model.mtp.pre_fc_norm_hidden.weight"),
          hidden_size_ * stream_count_)),
      layer_(
          tensors,
          "language_model.mtp.layers.0",
          0,
          mtp_layer_config(tensors.manifest().config())),
      final_mixer_(
          tensors,
          "language_model.mtp.hyper_connection_mixer",
          hidden_size_,
          stream_count_,
          8,
          tensors.manifest().config().quantization_group_size,
          epsilon_,
          false) {
    if (tensors.manifest().config().mtp_layer_count != 1) {
        throw std::runtime_error("Qwen3.8 MTP head must contain exactly one layer");
    }
}

MlxArray QwenMtpHead::project(
    const MlxArray& input,
    const QuantizedProjection& projection) const {
    return MlxArray::quantized_matmul(
        input,
        projection.weight,
        projection.scales,
        projection.biases,
        group_size_,
        projection.bits);
}

MlxArray QwenMtpHead::embed(const std::uint32_t token) const {
    if (token >= vocabulary_size_) throw std::runtime_error("MTP token id is out of range");
    const std::vector<std::int32_t> values{static_cast<std::int32_t>(token)};
    const std::vector<int> id_shape{1};
    const MlxArray ids = MlxArray::from_int32(values, id_shape);
    MlxArray value = MlxArray::dequantize(
        MlxArray::take_axis(embedding_.weight, ids, 0),
        MlxArray::take_axis(embedding_.scales, ids, 0),
        MlxArray::take_axis(embedding_.biases, ids, 0),
        group_size_,
        embedding_.bits);
    const std::vector<int> shape{1, 1, dimension(hidden_size_, "hidden size")};
    return value.reshape(shape);
}

MtpDecodeStep QwenMtpHead::forward_decode(
    const MlxArray& target_pre_mixer_stream,
    const std::uint32_t next_token,
    const std::size_t query_position,
    MtpDecodeState& state) const {
    const std::vector<int> expected{
        1, 1, dimension(hidden_size_ * stream_count_, "stream width")};
    if (target_pre_mixer_stream.shape() != expected) {
        throw std::runtime_error("MTP target stream must have shape [1,1,10240]");
    }
    if (!state.position_base.has_value()) state.position_base = query_position;
    if (query_position != *state.position_base + state.row_count) {
        throw std::runtime_error("MTP query positions must be contiguous");
    }

    MlxArray normalized_embedding = embed(next_token).rms_norm(embedding_norm_, epsilon_);
    MlxArray embedding_projection = project(normalized_embedding, fc_embedding_);
    MlxArray normalized_hidden =
        target_pre_mixer_stream.rms_norm(hidden_norm_, epsilon_);
    const std::vector<int> grouped{
        1, 1, dimension(stream_count_, "stream count"), dimension(hidden_size_, "hidden size")};
    MlxArray hidden_projection = project(normalized_hidden.reshape(grouped), fc_hidden_);
    const std::vector<int> embedding_grouped{1, 1, 1, dimension(hidden_size_, "hidden size")};
    MlxArray combined = MlxArray::add(
        hidden_projection, embedding_projection.reshape(embedding_grouped));
    MlxArray stream = combined.reshape(expected);
    stream = layer_.forward_decode(stream, next_token, state.layer);
    ++state.row_count;
    HyperConnectionRead final = final_mixer_.read(stream);
    MlxArray logits = project(final.mixed, language_head_);
    return {
        .logits = std::move(logits),
        .pre_mixer_stream = std::move(stream),
    };
}

} // namespace qwen38
