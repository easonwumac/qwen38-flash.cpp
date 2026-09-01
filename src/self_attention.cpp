#include "qwen38/self_attention.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid attention dimension: ") + name);
    }
    return static_cast<int>(value);
}

MlxArray offset_norm_weight(MlxArray raw, const std::size_t width) {
    const std::vector<float> ones_data(width, 1.0F);
    const std::vector<int> shape{dimension(width, "norm width")};
    MlxArray ones = MlxArray::from_float32(ones_data, shape).astype(raw.dtype());
    return MlxArray::add(raw, ones);
}

MlxArray effective_norm_weight(
    MlxArray raw,
    const std::size_t width,
    const bool has_offset) {
    if (!has_offset) return raw;
    return offset_norm_weight(std::move(raw), width);
}

MlxArray scalar(const float value, const mlx_dtype dtype) {
    const std::vector<float> values{value};
    const std::vector<int> shape{};
    return MlxArray::from_float32(values, shape).astype(dtype);
}

MlxArray slice_sequence_row(const MlxArray& batch, const std::size_t row) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() < 2 || row >= static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("attention verifier row is out of range");
    }
    std::vector<int> start(shape.size(), 0);
    std::vector<int> stop = shape;
    std::vector<int> strides(shape.size(), 1);
    start[1] = static_cast<int>(row);
    stop[1] = static_cast<int>(row + 1);
    return batch.slice(start, stop, strides);
}

MlxArray concatenate_sequence_rows(const std::vector<MlxArray>& rows) {
    if (rows.empty()) throw std::runtime_error("attention verifier has no outputs");
    MlxArray result = rows.front().share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, rows[row], 1);
    }
    return result;
}

} // namespace

SelfAttention::SelfAttention(
    MlxTensorStore& tensors,
    const std::string_view prefix,
    const ModelConfig& config)
    : attention_heads_(config.attention_head_count),
      key_value_heads_(config.key_value_head_count),
      head_dimension_(config.head_dimension),
      rotary_dimension_(static_cast<std::size_t>(
          static_cast<double>(config.head_dimension) * config.partial_rotary_factor)),
      bits_(dimension(config.quantization_bits, "quantization bits")),
      group_size_(dimension(config.quantization_group_size, "quantization group size")),
      epsilon_(static_cast<float>(config.rms_norm_epsilon)),
      rope_theta_(config.rope_theta),
      query_projection_(load_projection(tensors, std::string(prefix) + ".q_proj")),
      key_projection_(load_projection(tensors, std::string(prefix) + ".k_proj")),
      value_projection_(load_projection(tensors, std::string(prefix) + ".v_proj")),
      output_projection_(load_projection(tensors, std::string(prefix) + ".o_proj")),
      query_norm_weight_(effective_norm_weight(
          tensors.tensor(std::string(prefix) + ".q_norm.weight"),
          config.head_dimension,
          config.attention_norm_has_offset)),
      key_norm_weight_(effective_norm_weight(
          tensors.tensor(std::string(prefix) + ".k_norm.weight"),
          config.head_dimension,
          config.attention_norm_has_offset)) {
    if (attention_heads_ % key_value_heads_ != 0 || rotary_dimension_ == 0 ||
        rotary_dimension_ % 2 != 0 || rotary_dimension_ > head_dimension_) {
        throw std::runtime_error("unsupported attention head/rotary configuration");
    }
}

SelfAttention::QuantizedProjection SelfAttention::load_projection(
    MlxTensorStore& tensors,
    const std::string_view name) {
    const std::string base(name);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
    };
}

MlxArray SelfAttention::project(
    const MlxArray& input,
    const QuantizedProjection& projection) const {
    return MlxArray::quantized_matmul(
        input, projection.weight, projection.scales, projection.biases, group_size_, bits_);
}

MlxArray SelfAttention::apply_rope(const MlxArray& input, const std::size_t position) const {
    const auto shape = input.shape();
    if (shape.size() != 4 || shape[2] != 1 ||
        shape[3] != dimension(head_dimension_, "head dimension")) {
        throw std::runtime_error("RoPE input shape mismatch");
    }
    const int rotary = dimension(rotary_dimension_, "rotary dimension");
    const int half = rotary / 2;
    const std::size_t rotary_size = static_cast<std::size_t>(rotary);
    const std::size_t half_size = static_cast<std::size_t>(half);
    std::vector<float> cosine(rotary_size);
    std::vector<float> sine(rotary_size);
    for (std::size_t index = 0; index < half_size; ++index) {
        const double frequency = std::pow(
            rope_theta_, -2.0 * static_cast<double>(index) / static_cast<double>(rotary));
        const double angle = static_cast<double>(position) * frequency;
        cosine[index] = cosine[index + half_size] = static_cast<float>(std::cos(angle));
        sine[index] = sine[index + half_size] = static_cast<float>(std::sin(angle));
    }
    const std::vector<int> angle_shape{1, 1, 1, rotary};
    MlxArray cos_array = MlxArray::from_float32(cosine, angle_shape).astype(input.dtype());
    MlxArray sin_array = MlxArray::from_float32(sine, angle_shape).astype(input.dtype());
    const std::vector<int> strides{1, 1, 1, 1};
    const std::vector<int> rope_start{0, 0, 0, 0};
    const std::vector<int> rope_stop{shape[0], shape[1], shape[2], rotary};
    MlxArray rope = input.slice(rope_start, rope_stop, strides);
    const std::vector<int> first_stop{shape[0], shape[1], shape[2], half};
    const std::vector<int> second_start{0, 0, 0, half};
    MlxArray first = rope.slice(rope_start, first_stop, strides);
    MlxArray second = rope.slice(second_start, rope_stop, strides);
    MlxArray rotated = MlxArray::concatenate(second.negative(), first, 3);
    MlxArray transformed = MlxArray::add(
        MlxArray::multiply(rope, cos_array), MlxArray::multiply(rotated, sin_array));
    if (rotary_dimension_ == head_dimension_) return transformed;
    const std::vector<int> rest_start{0, 0, 0, rotary};
    const std::vector<int> rest_stop{shape[0], shape[1], shape[2], shape[3]};
    return MlxArray::concatenate(
        transformed, input.slice(rest_start, rest_stop, strides), 3);
}

MlxArray SelfAttention::forward_decode(
    const MlxArray& input,
    SelfAttentionState& state) const {
    const auto input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] != 1) {
        throw std::runtime_error("attention decode requires shape [1,1,hidden]");
    }
    const int heads = dimension(attention_heads_, "attention heads");
    const int kv_heads = dimension(key_value_heads_, "key/value heads");
    const int head_dimension = dimension(head_dimension_, "head dimension");
    const std::vector<int> query_shape{1, 1, heads, 2 * head_dimension};
    MlxArray query_gate = project(input, query_projection_).reshape(query_shape);
    const std::vector<int> strides{1, 1, 1, 1};
    const std::vector<int> query_start{0, 0, 0, 0};
    const std::vector<int> query_stop{1, 1, heads, head_dimension};
    const std::vector<int> gate_start{0, 0, 0, head_dimension};
    const std::vector<int> gate_stop{1, 1, heads, 2 * head_dimension};
    MlxArray query = query_gate.slice(query_start, query_stop, strides);
    MlxArray gate = query_gate.slice(gate_start, gate_stop, strides);
    query = query.rms_norm(query_norm_weight_, epsilon_).swapaxes(1, 2);
    const std::vector<int> key_value_shape{1, 1, kv_heads, head_dimension};
    MlxArray key = project(input, key_projection_).reshape(key_value_shape);
    key = key.rms_norm(key_norm_weight_, epsilon_).swapaxes(1, 2);
    MlxArray value = project(input, value_projection_).reshape(key_value_shape).swapaxes(1, 2);
    const std::size_t position = state.position_base + state.token_count;
    query = apply_rope(query, position);
    key = apply_rope(key, position);
    if (state.token_count == 0) {
        state.keys = std::move(key);
        state.values = std::move(value);
    } else {
        state.keys = MlxArray::concatenate(state.keys, key, 2);
        state.values = MlxArray::concatenate(state.values, value, 2);
    }
    ++state.token_count;
    const int repetitions = heads / kv_heads;
    MlxArray repeated_keys = state.keys.repeat_axis(repetitions, 1);
    MlxArray repeated_values = state.values.repeat_axis(repetitions, 1);
    MlxArray scores = MlxArray::matmul(query, repeated_keys.swapaxes(2, 3));
    MlxArray scale = scalar(1.0F / std::sqrt(static_cast<float>(head_dimension_)), scores.dtype());
    scores = MlxArray::multiply(scores, scale);
    MlxArray probabilities = scores.astype(MLX_FLOAT32).softmax_axis(-1).astype(query.dtype());
    MlxArray attended = MlxArray::matmul(probabilities, repeated_values).swapaxes(1, 2);
    const std::vector<int> flat_shape{1, 1, heads * head_dimension};
    MlxArray gated = MlxArray::multiply(
        attended.reshape(flat_shape), gate.reshape(flat_shape).sigmoid());
    return project(gated, output_projection_);
}

MlxArray SelfAttention::forward_verify(
    const MlxArray& input,
    const SelfAttentionState& origin,
    std::vector<SelfAttentionState>& checkpoints) const {
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] < 1 ||
        input_shape[1] > 64) {
        throw std::runtime_error("attention batch requires shape [1,S,hidden], S=1..64");
    }
    const std::size_t rows = static_cast<std::size_t>(input_shape[1]);
    const int heads = dimension(attention_heads_, "attention heads");
    const int kv_heads = dimension(key_value_heads_, "key/value heads");
    const int head_dimension = dimension(head_dimension_, "head dimension");
    MlxArray query_gate_batch = project(input, query_projection_).reshape(
        std::vector<int>{1, input_shape[1], heads, 2 * head_dimension});
    MlxArray key_batch = project(input, key_projection_).reshape(
        std::vector<int>{1, input_shape[1], kv_heads, head_dimension});
    MlxArray value_batch = project(input, value_projection_).reshape(
        std::vector<int>{1, input_shape[1], kv_heads, head_dimension});

    SelfAttentionState working;
    working.token_count = origin.token_count;
    working.position_base = origin.position_base;
    if (origin.token_count != 0) {
        working.keys = origin.keys.share();
        working.values = origin.values.share();
    }
    checkpoints.clear();
    checkpoints.resize(rows);
    std::vector<MlxArray> gated_rows;
    gated_rows.reserve(rows);
    const std::vector<int> strides{1, 1, 1, 1};
    for (std::size_t row = 0; row < rows; ++row) {
        MlxArray query_gate = slice_sequence_row(query_gate_batch, row);
        MlxArray query = query_gate.slice(
            std::vector<int>{0, 0, 0, 0},
            std::vector<int>{1, 1, heads, head_dimension},
            strides);
        MlxArray gate = query_gate.slice(
            std::vector<int>{0, 0, 0, head_dimension},
            std::vector<int>{1, 1, heads, 2 * head_dimension},
            strides);
        query = query.rms_norm(query_norm_weight_, epsilon_).swapaxes(1, 2);
        MlxArray key = slice_sequence_row(key_batch, row)
            .rms_norm(key_norm_weight_, epsilon_).swapaxes(1, 2);
        MlxArray value = slice_sequence_row(value_batch, row).swapaxes(1, 2);
        const std::size_t position = working.position_base + working.token_count;
        query = apply_rope(query, position);
        key = apply_rope(key, position);
        if (working.token_count == 0) {
            working.keys = std::move(key);
            working.values = std::move(value);
        } else {
            working.keys = MlxArray::concatenate(working.keys, key, 2);
            working.values = MlxArray::concatenate(working.values, value, 2);
        }
        ++working.token_count;
        checkpoints[row].keys = working.keys.share();
        checkpoints[row].values = working.values.share();
        checkpoints[row].token_count = working.token_count;
        checkpoints[row].position_base = working.position_base;

        const int repetitions = heads / kv_heads;
        MlxArray repeated_keys = working.keys.repeat_axis(repetitions, 1);
        MlxArray repeated_values = working.values.repeat_axis(repetitions, 1);
        MlxArray scores = MlxArray::matmul(query, repeated_keys.swapaxes(2, 3));
        scores = MlxArray::multiply(
            scores,
            scalar(1.0F / std::sqrt(static_cast<float>(head_dimension_)), scores.dtype()));
        MlxArray probabilities =
            scores.astype(MLX_FLOAT32).softmax_axis(-1).astype(query.dtype());
        MlxArray attended =
            MlxArray::matmul(probabilities, repeated_values).swapaxes(1, 2);
        const std::vector<int> flat_shape{1, 1, heads * head_dimension};
        gated_rows.push_back(MlxArray::multiply(
            attended.reshape(flat_shape), gate.reshape(flat_shape).sigmoid()));
    }
    return project(concatenate_sequence_rows(gated_rows), output_projection_);
}

} // namespace qwen38
