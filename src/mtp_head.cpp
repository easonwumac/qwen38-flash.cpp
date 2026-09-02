#include "qwen38/mtp_head.hpp"

#include "qwen38/quantization_geometry.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid MTP dimension: ") + name);
    }
    return static_cast<int>(value);
}

MlxArray concatenate_sequence(const std::span<const MlxArray* const> rows) {
    if (rows.empty()) throw std::runtime_error("cannot concatenate an empty MTP batch");
    MlxArray result = rows.front()->share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, *rows[row], 1);
    }
    return result;
}

MlxArray slice_sequence_row(const MlxArray& batch, const std::size_t row) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() != 3 || row >= static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("MTP batch row is out of range");
    }
    return batch.slice(
        std::vector<int>{0, static_cast<int>(row), 0},
        std::vector<int>{1, static_cast<int>(row + 1), shape[2]},
        std::vector<int>{1, 1, 1});
}

MlxArray offset_norm(MlxArray raw, const std::size_t width) {
    const std::vector<float> values(width, 1.0F);
    const std::vector<int> shape{dimension(width, "norm width")};
    return MlxArray::add(raw, MlxArray::from_float32(values, shape).astype(raw.dtype()));
}

int infer_projection_bits(
    MlxTensorStore& tensors,
    const char* prefix,
    const std::size_t group_size) {
    const std::string base(prefix);
    const std::vector<int> weight_shape = tensors.tensor(base + ".weight").shape();
    const std::vector<int> scale_shape = tensors.tensor(base + ".scales").shape();
    return infer_affine_quantization_bits(
        weight_shape, scale_shape, group_size, "MTP");
}

ModelConfig mtp_layer_config(const ModelConfig& target, const int bits) {
    ModelConfig result = target;
    result.layer_count = 1;
    result.layer_types = {"full_attention"};
    result.ple_layer_ids.clear();
    result.quantization_bits = static_cast<std::size_t>(bits);
    // The retained ReleaseFast sidecar stores pre-FC/HC norms as deltas, but
    // its attention q/k norms are already effective weights (P192 fixture).
    result.attention_norm_has_offset = false;
    return result;
}

} // namespace

MtpDecodeState snapshot_mtp_decode_state(const MtpDecodeState& state) {
    return {
        .layer = snapshot_decoder_layer_state(state.layer),
        .row_count = state.row_count,
        .position_base = state.position_base,
    };
}

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
      mtp_bits_(infer_projection_bits(
          tensors,
          "language_model.mtp.fc_embedding",
          tensors.manifest().config().quantization_group_size)),
      epsilon_(static_cast<float>(tensors.manifest().config().rms_norm_epsilon)),
      embedding_(load_projection(
          tensors,
          "language_model.model.embed_tokens",
          dimension(tensors.manifest().config().quantization_bits, "target bits"))),
      language_head_(load_projection(
          tensors,
          "language_model.lm_head",
          dimension(tensors.manifest().config().quantization_bits, "target bits"))),
      fc_embedding_(load_projection(
          tensors, "language_model.mtp.fc_embedding", mtp_bits_)),
      fc_hidden_(load_projection(
          tensors,
          "language_model.mtp.fc_hidden",
          infer_projection_bits(
              tensors,
              "language_model.mtp.fc_hidden",
              tensors.manifest().config().quantization_group_size))),
      embedding_norm_(offset_norm(
          tensors.tensor("language_model.mtp.pre_fc_norm_embedding.weight"), hidden_size_)),
      hidden_norm_(offset_norm(
          tensors.tensor("language_model.mtp.pre_fc_norm_hidden.weight"),
          hidden_size_ * stream_count_)),
      layer_(
          tensors,
          "language_model.mtp.layers.0",
          0,
          mtp_layer_config(tensors.manifest().config(), mtp_bits_)),
      final_mixer_(
          tensors,
          "language_model.mtp.hyper_connection_mixer",
          hidden_size_,
          stream_count_,
          static_cast<std::size_t>(mtp_bits_),
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
    return embed(ids);
}

MlxArray QwenMtpHead::embed(const MlxArray& ids) const {
    if (ids.shape() != std::vector<int>{1}) {
        throw std::runtime_error("MTP lazy token id must have shape [1]");
    }
    MlxArray value = MlxArray::dequantize(
        MlxArray::take_axis(embedding_.weight, ids, 0),
        MlxArray::take_axis(embedding_.scales, ids, 0),
        MlxArray::take_axis(embedding_.biases, ids, 0),
        group_size_,
        embedding_.bits);
    const std::vector<int> shape{1, 1, dimension(hidden_size_, "hidden size")};
    return value.reshape(shape);
}

MlxArray QwenMtpHead::forward_stream(
    const MlxArray& target_pre_mixer_stream,
    const std::uint32_t next_token,
    const std::size_t query_position,
    MtpDecodeState& state,
    MtpTrace* trace) const {
    if (next_token >= vocabulary_size_) {
        throw std::runtime_error("MTP token id is out of range");
    }
    const std::vector<std::int32_t> values{static_cast<std::int32_t>(next_token)};
    const std::vector<int> shape{1};
    const MlxArray token = MlxArray::from_int32(values, shape);
    return forward_stream_lazy_token(
        target_pre_mixer_stream, token, query_position, state, trace);
}

MlxArray QwenMtpHead::forward_stream_lazy_token(
    const MlxArray& target_pre_mixer_stream,
    const MlxArray& next_token,
    const std::size_t query_position,
    MtpDecodeState& state,
    MtpTrace* trace) const {
    const std::vector<int> expected{
        1, 1, dimension(hidden_size_ * stream_count_, "stream width")};
    if (target_pre_mixer_stream.shape() != expected) {
        throw std::runtime_error("MTP target stream must have shape [1,1,10240]");
    }
    if (!state.position_base.has_value()) state.position_base = query_position;
    if (query_position != *state.position_base + state.row_count) {
        throw std::runtime_error("MTP query positions must be contiguous");
    }
    if (state.row_count == 0) {
        state.layer.full_attention.position_base = query_position;
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
    if (trace != nullptr) {
        const std::vector<int> zero_shape{1};
        trace->combined_stream = MlxArray::add(
            stream, MlxArray::zeros(zero_shape, stream.dtype()));
    }
    stream = layer_.forward_decode(
        stream, 0, state.layer, trace == nullptr ? nullptr : &trace->layer);
    ++state.row_count;
    return stream;
}

void QwenMtpHead::consume_decode(
    const MlxArray& target_pre_mixer_stream,
    const std::uint32_t next_token,
    const std::size_t query_position,
    MtpDecodeState& state) const {
    MlxArray stream = forward_stream(
        target_pre_mixer_stream, next_token, query_position, state, nullptr);
    std::vector<const MlxArray*> outputs{&stream};
    if (state.layer.full_attention.qsa_raw_keys.get().ctx != nullptr) {
        outputs.push_back(&state.layer.full_attention.qsa_raw_keys);
    }
    if (state.layer.full_attention.qsa_pooled_keys.get().ctx != nullptr) {
        outputs.push_back(&state.layer.full_attention.qsa_pooled_keys);
    }
    MlxArray::eval_all(outputs);
}

void QwenMtpHead::consume_committed_batch(
    const std::span<const MlxArray* const> target_pre_mixer_streams,
    const std::span<const std::uint32_t> tokens,
    const std::size_t query_position,
    MtpDecodeState& state) const {
    if (tokens.size() < 2 || tokens.size() > 5 ||
        target_pre_mixer_streams.size() != tokens.size()) {
        throw std::runtime_error("MTP committed batch requires 2 to 5 matching rows");
    }
    if (!state.position_base.has_value()) state.position_base = query_position;
    if (query_position != *state.position_base + state.row_count) {
        throw std::runtime_error("MTP committed batch positions must be contiguous");
    }
    if (state.row_count == 0) {
        state.layer.full_attention.position_base = query_position;
    }

    std::vector<MlxArray> embeddings;
    embeddings.reserve(tokens.size());
    std::vector<const MlxArray*> embedding_rows;
    embedding_rows.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
        embeddings.push_back(embed(token));
        embedding_rows.push_back(&embeddings.back());
    }
    MlxArray embedding_batch = concatenate_sequence(embedding_rows);
    MlxArray normalized_embedding = embedding_batch.rms_norm(embedding_norm_, epsilon_);
    MlxArray embedding_projection = project(normalized_embedding, fc_embedding_);

    MlxArray hidden_batch = concatenate_sequence(target_pre_mixer_streams);
    MlxArray normalized_hidden = hidden_batch.rms_norm(hidden_norm_, epsilon_);
    const int rows = static_cast<int>(tokens.size());
    const std::vector<int> grouped{
        1, rows, dimension(stream_count_, "stream count"),
        dimension(hidden_size_, "hidden size")};
    MlxArray hidden_projection = project(normalized_hidden.reshape(grouped), fc_hidden_);
    MlxArray combined = MlxArray::add(
        hidden_projection,
        embedding_projection.reshape(
            std::vector<int>{1, rows, 1, dimension(hidden_size_, "hidden size")}));
    MlxArray stream_batch = combined.reshape(
        std::vector<int>{1, rows, dimension(hidden_size_ * stream_count_, "stream width")});

    std::vector<MlxArray> streams;
    streams.reserve(tokens.size());
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        streams.push_back(slice_sequence_row(stream_batch, row));
    }
    std::vector<DecoderLayerState> checkpoints;
    streams = layer_.forward_verify_dense_batched(
        std::move(streams), tokens, state.layer, checkpoints);
    const char* defer_eval = std::getenv("QWEN38_DEFER_MTP_COMMIT_EVAL");
    const bool defer_eval_enabled =
        defer_eval == nullptr || std::string_view(defer_eval) != "0";
    if (!defer_eval_enabled) {
        std::vector<const MlxArray*> outputs;
        outputs.reserve(streams.size());
        for (const MlxArray& stream : streams) outputs.push_back(&stream);
        MlxArray::eval_all(outputs);
    }
    state.layer = std::move(checkpoints.back());
    state.row_count += tokens.size();
}

MtpDecodeStep QwenMtpHead::forward_decode(
    const MlxArray& target_pre_mixer_stream,
    const std::uint32_t next_token,
    const std::size_t query_position,
    MtpDecodeState& state,
    MtpTrace* trace) const {
    MlxArray stream = forward_stream(
        target_pre_mixer_stream, next_token, query_position, state, trace);
    HyperConnectionRead final = final_mixer_.read(stream);
    if (trace != nullptr) {
        const std::vector<int> zero_shape{1};
        trace->final_mixed = MlxArray::add(
            final.mixed, MlxArray::zeros(zero_shape, final.mixed.dtype()));
    }
    MlxArray logits = project(final.mixed, language_head_);
    return {
        .logits = std::move(logits),
        .pre_mixer_stream = std::move(stream),
    };
}

MtpDecodeStep QwenMtpHead::forward_decode_lazy_token(
    const MlxArray& target_pre_mixer_stream,
    const MlxArray& next_token,
    const std::size_t query_position,
    MtpDecodeState& state) const {
    MlxArray stream = forward_stream_lazy_token(
        target_pre_mixer_stream, next_token, query_position, state, nullptr);
    HyperConnectionRead final = final_mixer_.read(stream);
    MlxArray logits = project(final.mixed, language_head_);
    return {
        .logits = std::move(logits),
        .pre_mixer_stream = std::move(stream),
    };
}

} // namespace qwen38
