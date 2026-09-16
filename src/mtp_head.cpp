#include "qwen38/mtp_head.hpp"

#include "qwen38/quantization_geometry.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
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

bool has_native_vqlab_head(const MlxTensorStore& tensors) {
    const ModelManifest& manifest = tensors.manifest();
    return manifest.has_tensor("fc.weight") && manifest.has_tensor("norm_e.weight") &&
        manifest.has_tensor("norm_h.weight") &&
        manifest.has_tensor("block.self_attn.q_proj.weight") &&
        manifest.has_tensor("mixer.hc_norm.weight");
}

ModelConfig mtp_layer_config(
    const ModelConfig& target,
    const int bits,
    const std::size_t group_size,
    const bool native_vqlab) {
    ModelConfig result = target;
    result.layer_count = 1;
    result.layer_types = {"full_attention"};
    result.ple_layer_ids.clear();
    result.quantization_bits = static_cast<std::size_t>(bits);
    result.quantization_group_size = group_size;
    // The retained ReleaseFast sidecar stores pre-FC/HC norms as deltas, but
    // its attention q/k norms are already effective weights (P192 fixture).
    result.attention_norm_has_offset = native_vqlab;
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
    const char* prefix) {
    const std::string base(prefix);
    const QuantizationSpec quantization = tensors.manifest().quantization_for(base);
    const int group_size = dimension(quantization.group_size, "projection group size");
    const int bits = infer_projection_bits(tensors, prefix, quantization.group_size);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
        .bits = bits,
        .group_size = group_size,
    };
}

QwenMtpHead::QwenMtpHead(MlxTensorStore& tensors)
    : native_vqlab_(has_native_vqlab_head(tensors)),
      hidden_size_(tensors.manifest().config().hidden_size),
      stream_count_(tensors.manifest().config().hyper_connection_count),
      vocabulary_size_(tensors.manifest().config().vocabulary_size),
      mtp_bits_(native_vqlab_ ? 6 : infer_projection_bits(
          tensors, "language_model.mtp.fc_embedding",
          tensors.manifest().quantization_for(
              "language_model.mtp.fc_embedding").group_size)),
      epsilon_(static_cast<float>(tensors.manifest().config().rms_norm_epsilon)),
      embedding_(load_projection(tensors, "language_model.model.embed_tokens")),
      language_head_(load_projection(tensors, "language_model.lm_head")),
      fc_embedding_(native_vqlab_ ? QuantizedProjection{} :
          load_projection(tensors, "language_model.mtp.fc_embedding")),
      fc_hidden_(native_vqlab_ ? QuantizedProjection{} :
          load_projection(tensors, "language_model.mtp.fc_hidden")),
      fused_fc_(native_vqlab_ ? tensors.tensor("fc.weight") : MlxArray{}),
      embedding_norm_(offset_norm(
          tensors.tensor(native_vqlab_ ? "norm_e.weight" :
              "language_model.mtp.pre_fc_norm_embedding.weight"), hidden_size_)),
      hidden_norm_(offset_norm(
          tensors.tensor(native_vqlab_ ? "norm_h.weight" :
              "language_model.mtp.pre_fc_norm_hidden.weight"),
          hidden_size_ * stream_count_)),
      layer_(
          tensors,
          native_vqlab_ ? "block" : "language_model.mtp.layers.0",
          0,
          mtp_layer_config(
              tensors.manifest().config(), mtp_bits_,
              native_vqlab_ ? 32 :
                  tensors.manifest().config().quantization_group_size,
              native_vqlab_)),
      final_mixer_(
          tensors,
          native_vqlab_ ? "mixer" : "language_model.mtp.hyper_connection_mixer",
          hidden_size_,
          stream_count_,
          static_cast<std::size_t>(mtp_bits_),
          native_vqlab_ ? 32 : tensors.manifest().config().quantization_group_size,
          epsilon_,
          false) {
    if (tensors.manifest().config().mtp_layer_count != 1) {
        throw std::runtime_error("Qwen3.8 MTP head must contain exactly one layer");
    }
    if (const char* path = std::getenv("QWEN38_MTP_HEAD_ADAPTER");
        path != nullptr && path[0] != '\0') {
        head_adapter_store_ = std::make_unique<MlxSafetensors>(path);
        head_adapter_b_ = head_adapter_store_->tensor("lora_b");
        const std::vector<int> b_shape = head_adapter_b_.shape();
        if (b_shape.size() != 2 ||
            b_shape[0] != dimension(vocabulary_size_, "adapter vocabulary")) {
            throw std::runtime_error("MTP head adapter lora_b shape mismatch");
        }
        const int rank = b_shape[1];
        for (std::size_t depth = 1; depth <= 4; ++depth) {
            MlxArray a = head_adapter_store_->tensor(
                "lora_a.depth" + std::to_string(depth));
            if (a.shape() != std::vector<int>{rank, dimension(hidden_size_, "adapter hidden")}) {
                throw std::runtime_error("MTP head adapter lora_a shape mismatch");
            }
            head_adapter_a_.push_back(std::move(a));
        }
        std::clog << "qwen38: MTP head adapter enabled rank=" << rank << '\n';
    }
    if (const char* path = std::getenv("QWEN38_MTP_HIDDEN_ADAPTER");
        path != nullptr && path[0] != '\0') {
        hidden_adapter_store_ = std::make_unique<MlxSafetensors>(path);
        hidden_adapter_b_ = hidden_adapter_store_->tensor("hidden_lora_b");
        const std::vector<int> b_shape = hidden_adapter_b_.shape();
        if (b_shape.size() != 2 ||
            b_shape[0] != dimension(hidden_size_, "hidden adapter output")) {
            throw std::runtime_error("MTP hidden adapter lora_b shape mismatch");
        }
        const int rank = b_shape[1];
        for (std::size_t depth = 1; depth <= 4; ++depth) {
            MlxArray a = hidden_adapter_store_->tensor(
                "hidden_lora_a.depth" + std::to_string(depth));
            if (a.shape() != std::vector<int>{rank, dimension(hidden_size_, "hidden adapter input")}) {
                throw std::runtime_error("MTP hidden adapter lora_a shape mismatch");
            }
            hidden_adapter_a_.push_back(std::move(a));
        }
        std::clog << "qwen38: MTP hidden adapter enabled rank=" << rank << '\n';
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
        projection.group_size,
        projection.bits);
}

MlxArray QwenMtpHead::combine_inputs(
    const MlxArray& target_pre_mixer_streams,
    const MlxArray& next_tokens) const {
    const std::vector<int> target_shape = target_pre_mixer_streams.shape();
    const std::vector<int> token_shape = next_tokens.shape();
    if (target_shape.size() != 3 || target_shape[0] != 1 || target_shape[1] < 1 ||
        target_shape[2] != dimension(hidden_size_ * stream_count_, "stream width") ||
        token_shape != std::vector<int>{target_shape[1]}) {
        throw std::runtime_error("MTP target streams and tokens have incompatible shapes");
    }
    const int rows = target_shape[1];
    MlxArray normalized_embedding = embed(next_tokens).rms_norm(embedding_norm_, epsilon_);
    const std::vector<int> grouped{
        1, rows, dimension(stream_count_, "stream count"),
        dimension(hidden_size_, "hidden size")};
    if (native_vqlab_) {
        MlxArray grouped_hidden = target_pre_mixer_streams.reshape(grouped);
        MlxArray mean_square = MlxArray::multiply(
            grouped_hidden, grouped_hidden).mean_axis(-1, true);
        const std::vector<float> epsilon_data{epsilon_};
        const std::vector<int> scalar_shape{};
        MlxArray denominator = MlxArray::add(
            mean_square, MlxArray::from_float32(epsilon_data, scalar_shape))
            .square_root();
        MlxArray normalized_hidden = MlxArray::multiply(
            MlxArray::divide(grouped_hidden, denominator),
            hidden_norm_.reshape(std::vector<int>{
                1, 1, dimension(stream_count_, "stream count"),
                dimension(hidden_size_, "hidden size")}));
        MlxArray repeated_embedding = normalized_embedding.reshape(
            std::vector<int>{1, rows, 1, dimension(hidden_size_, "hidden size")})
            .broadcast_to(grouped);
        MlxArray joined = MlxArray::concatenate(
            repeated_embedding, normalized_hidden, -1);
        return MlxArray::matmul(joined, fused_fc_.transpose()).reshape(target_shape);
    }
    MlxArray embedding_projection = project(normalized_embedding, fc_embedding_);
    MlxArray normalized_hidden = target_pre_mixer_streams.rms_norm(hidden_norm_, epsilon_);
    MlxArray hidden_projection = project(normalized_hidden.reshape(grouped), fc_hidden_);
    return MlxArray::add(
        hidden_projection,
        embedding_projection.reshape(
            std::vector<int>{1, rows, 1, dimension(hidden_size_, "hidden size")}))
        .reshape(target_shape);
}

MlxArray QwenMtpHead::embed(const std::uint32_t token) const {
    if (token >= vocabulary_size_) throw std::runtime_error("MTP token id is out of range");
    const std::vector<std::int32_t> values{static_cast<std::int32_t>(token)};
    const std::vector<int> id_shape{1};
    const MlxArray ids = MlxArray::from_int32(values, id_shape);
    return embed(ids);
}

MlxArray QwenMtpHead::embed(const MlxArray& ids) const {
    const std::vector<int> id_shape = ids.shape();
    if (id_shape.size() != 1 || id_shape[0] < 1 || id_shape[0] > 1024) {
        throw std::runtime_error("MTP token ids must have shape [S], S=1..1024");
    }
    MlxArray value = MlxArray::dequantize(
        MlxArray::take_axis(embedding_.weight, ids, 0),
        MlxArray::take_axis(embedding_.scales, ids, 0),
        MlxArray::take_axis(embedding_.biases, ids, 0),
        embedding_.group_size,
        embedding_.bits);
    const std::vector<int> shape{
        1, id_shape[0], dimension(hidden_size_, "hidden size")};
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

    MlxArray stream = combine_inputs(target_pre_mixer_stream, next_token);
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
    if (tokens.size() < 2 || tokens.size() > 25 ||
        target_pre_mixer_streams.size() != tokens.size()) {
        throw std::runtime_error("MTP committed batch requires 2 to 25 matching rows");
    }
    if (!state.position_base.has_value()) state.position_base = query_position;
    if (query_position != *state.position_base + state.row_count) {
        throw std::runtime_error("MTP committed batch positions must be contiguous");
    }
    if (state.row_count == 0) {
        state.layer.full_attention.position_base = query_position;
    }

    std::vector<std::int32_t> token_values;
    token_values.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
        if (token >= vocabulary_size_) {
            throw std::runtime_error("MTP token id is out of range");
        }
        token_values.push_back(static_cast<std::int32_t>(token));
    }
    MlxArray ids = MlxArray::from_int32(
        token_values, std::vector<int>{dimension(tokens.size(), "committed rows")});
    MlxArray hidden_batch = concatenate_sequence(target_pre_mixer_streams);
    MlxArray stream_batch = combine_inputs(hidden_batch, ids);

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

void QwenMtpHead::consume_prefill_batch(
    const MlxArray& target_pre_mixer_streams,
    const std::span<const std::uint32_t> tokens,
    const std::size_t query_position,
    MtpDecodeState& state) const {
    const std::vector<int> expected{
        1, dimension(tokens.size(), "prefill rows"),
        dimension(hidden_size_ * stream_count_, "stream width")};
    if (tokens.empty() || tokens.size() > 512 ||
        target_pre_mixer_streams.shape() != expected) {
        throw std::runtime_error(
            "MTP prefill batch requires 1 to 512 matching target rows");
    }
    if (!state.position_base.has_value()) state.position_base = query_position;
    if (query_position != *state.position_base + state.row_count) {
        throw std::runtime_error("MTP prefill positions must be contiguous");
    }
    if (state.row_count == 0) {
        state.layer.full_attention.position_base = query_position;
    }

    std::vector<std::int32_t> token_values;
    token_values.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
        if (token >= vocabulary_size_) {
            throw std::runtime_error("MTP token id is out of range");
        }
        token_values.push_back(static_cast<std::int32_t>(token));
    }
    MlxArray ids = MlxArray::from_int32(
        token_values, std::vector<int>{dimension(tokens.size(), "prefill rows")});
    MlxArray combined = combine_inputs(target_pre_mixer_streams, ids);
    MlxArray output = layer_.forward_prefill(
        std::move(combined), tokens, state.layer);
    state.row_count += tokens.size();

    std::vector<const MlxArray*> outputs{&output};
    if (state.layer.full_attention.qsa_raw_keys.get().ctx != nullptr) {
        outputs.push_back(&state.layer.full_attention.qsa_raw_keys);
    }
    if (state.layer.full_attention.qsa_pooled_keys.get().ctx != nullptr) {
        outputs.push_back(&state.layer.full_attention.qsa_pooled_keys);
    }
    MlxArray::eval_all(outputs);
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
    MtpDecodeState& state,
    const std::size_t adapter_depth,
    MlxArray* final_mixed_trace) const {
    MlxArray stream = forward_stream_lazy_token(
        target_pre_mixer_stream, next_token, query_position, state, nullptr);
    HyperConnectionRead final = final_mixer_.read(stream);
    if (final_mixed_trace != nullptr) {
        *final_mixed_trace = final.mixed.share();
    }
    MlxArray head_input = final.mixed.share();
    if (adapter_depth != 0 && !hidden_adapter_a_.empty()) {
        const std::size_t index = std::min(adapter_depth, hidden_adapter_a_.size()) - 1;
        MlxArray projected = MlxArray::matmul(
            head_input, hidden_adapter_a_[index].transpose());
        head_input = MlxArray::add(
            head_input, MlxArray::matmul(projected, hidden_adapter_b_.transpose()));
    }
    MlxArray logits = project(head_input, language_head_);
    if (adapter_depth != 0 && !head_adapter_a_.empty()) {
        const std::size_t index = std::min(adapter_depth, head_adapter_a_.size()) - 1;
        MlxArray projected = MlxArray::matmul(
            final.mixed, head_adapter_a_[index].transpose());
        logits = MlxArray::add(
            logits, MlxArray::matmul(projected, head_adapter_b_.transpose()));
    }
    return {
        .logits = std::move(logits),
        .pre_mixer_stream = std::move(stream),
    };
}

} // namespace qwen38
