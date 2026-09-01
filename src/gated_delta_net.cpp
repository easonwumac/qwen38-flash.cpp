#include "qwen38/gated_delta_net.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid GatedDeltaNet dimension: ") + name);
    }
    return static_cast<int>(value);
}

MlxArray scalar(const float value, const mlx_dtype dtype) {
    const std::vector<float> values{value};
    const std::vector<int> shape{};
    return MlxArray::from_float32(values, shape).astype(dtype);
}

} // namespace

GatedDeltaNet::GatedDeltaNet(
    MlxTensorStore& tensors,
    const std::string_view prefix,
    const ModelConfig& config)
    : key_head_count_(config.linear_key_head_count),
      value_head_count_(config.linear_value_head_count),
      key_head_dimension_(config.linear_key_head_dimension),
      value_head_dimension_(config.linear_value_head_dimension),
      convolution_kernel_size_(config.linear_convolution_kernel_size),
      bits_(dimension(config.quantization_bits, "quantization_bits")),
      group_size_(dimension(config.quantization_group_size, "quantization_group_size")),
      epsilon_(static_cast<float>(config.rms_norm_epsilon)),
      output_gate_type_(config.output_gate_type),
      qkv_projection_(load_projection(tensors, std::string(prefix) + ".in_proj_qkv")),
      z_projection_(load_projection(tensors, std::string(prefix) + ".in_proj_z")),
      beta_projection_(load_projection(tensors, std::string(prefix) + ".in_proj_b")),
      decay_projection_(load_projection(tensors, std::string(prefix) + ".in_proj_a")),
      output_projection_(load_projection(tensors, std::string(prefix) + ".out_proj")),
      convolution_weight_(tensors.tensor(std::string(prefix) + ".conv1d.weight")),
      decay_log_(tensors.tensor(std::string(prefix) + ".A_log")),
      decay_bias_(tensors.tensor(std::string(prefix) + ".dt_bias")),
      norm_weight_(tensors.tensor(std::string(prefix) + ".norm.weight")) {
    static_cast<void>(dimension(key_head_count_, "key_head_count"));
    static_cast<void>(dimension(value_head_count_, "value_head_count"));
    static_cast<void>(dimension(key_head_dimension_, "key_head_dimension"));
    static_cast<void>(dimension(value_head_dimension_, "value_head_dimension"));
    if (value_head_count_ % key_head_count_ != 0) {
        throw std::runtime_error("value heads must be divisible by key heads");
    }
    if (convolution_kernel_size_ != 4) {
        throw std::runtime_error("the retained Qwen3.8 checkpoint requires convolution width 4");
    }
}

GatedDeltaNet::QuantizedProjection GatedDeltaNet::load_projection(
    MlxTensorStore& tensors,
    const std::string_view name) {
    const std::string base(name);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
    };
}

MlxArray GatedDeltaNet::project(
    const MlxArray& input,
    const QuantizedProjection& projection) const {
    return MlxArray::quantized_matmul(
        input, projection.weight, projection.scales, projection.biases, group_size_, bits_);
}

MlxArray GatedDeltaNet::forward_first(const MlxArray& input) const {
    GatedDeltaNetState state;
    return forward_decode(input, state);
}

MlxArray GatedDeltaNet::forward_decode(
    const MlxArray& input,
    GatedDeltaNetState& state) const {
    const auto input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] != 1) {
        throw std::runtime_error("first-token GatedDeltaNet requires shape [1,1,hidden]");
    }
    const int key_heads = dimension(key_head_count_, "key_head_count");
    const int value_heads = dimension(value_head_count_, "value_head_count");
    const int key_dimension = dimension(key_head_dimension_, "key_head_dimension");
    const int value_dimension = dimension(value_head_dimension_, "value_head_dimension");
    const int key_width = key_heads * key_dimension;
    const int value_width = value_heads * value_dimension;
    const int convolution_width = 2 * key_width + value_width;

    MlxArray qkv = project(input, qkv_projection_);
    if (!state.initialized) {
        const std::vector<int> convolution_state_shape{1, 3, convolution_width};
        const std::vector<int> recurrent_state_shape{
            1, value_heads, value_dimension, key_dimension};
        state.convolution = MlxArray::zeros(convolution_state_shape, qkv.dtype());
        state.recurrent = MlxArray::zeros(recurrent_state_shape, qkv.dtype());
        state.initialized = true;
    }
    MlxArray convolution_input = MlxArray::concatenate(state.convolution, qkv, 1);
    const std::vector<int> state_start{0, 1, 0};
    const std::vector<int> state_stop{1, 4, convolution_width};
    const std::vector<int> state_strides{1, 1, 1};
    state.convolution = convolution_input.slice(state_start, state_stop, state_strides);
    MlxArray convolved = MlxArray::conv1d(
        convolution_input,
        convolution_weight_,
        1,
        0,
        1,
        convolution_width).silu();

    const std::vector<int> strides{1, 1, 1};
    const std::vector<int> query_start{0, 0, 0};
    const std::vector<int> query_stop{1, 1, key_width};
    const std::vector<int> key_start{0, 0, key_width};
    const std::vector<int> key_stop{1, 1, 2 * key_width};
    const std::vector<int> value_start{0, 0, 2 * key_width};
    const std::vector<int> value_stop{1, 1, convolution_width};
    MlxArray query = convolved.slice(query_start, query_stop, strides);
    MlxArray key = convolved.slice(key_start, key_stop, strides);
    MlxArray value = convolved.slice(value_start, value_stop, strides);
    const std::vector<int> qk_shape{1, 1, key_heads, key_dimension};
    const std::vector<int> value_shape{1, 1, value_heads, value_dimension};
    query = query.reshape(qk_shape);
    key = key.reshape(qk_shape);
    value = value.reshape(value_shape);
    const std::vector<float> ones_data(key_head_dimension_, 1.0F);
    const std::vector<int> norm_shape{key_dimension};
    MlxArray ones = MlxArray::from_float32(ones_data, norm_shape).astype(query.dtype());
    query = query.rms_norm(ones, 1.0e-6F);
    key = key.rms_norm(ones, 1.0e-6F);
    MlxArray query_scale = scalar(1.0F / static_cast<float>(key_head_dimension_), query.dtype());
    MlxArray key_scale = scalar(
        1.0F / std::sqrt(static_cast<float>(key_head_dimension_)), key.dtype());
    query = MlxArray::multiply(query, query_scale);
    key = MlxArray::multiply(key, key_scale);
    const int repetition = value_heads / key_heads;
    query = query.repeat_axis(repetition, 2);
    key = key.repeat_axis(repetition, 2);

    MlxArray beta = project(input, beta_projection_).sigmoid();
    const std::vector<int> beta_shape{1, value_heads, 1};
    beta = beta.reshape(beta_shape);
    MlxArray decay_input = MlxArray::add(project(input, decay_projection_), decay_bias_);
    MlxArray softplus = decay_input.astype(MLX_FLOAT32).exp().log1p();
    MlxArray decay_rate = decay_log_.astype(MLX_FLOAT32).exp();
    MlxArray decay = MlxArray::multiply(decay_rate, softplus).negative().exp().astype(qkv.dtype());
    const std::vector<int> decay_shape{1, value_heads, 1, 1};
    state.recurrent = MlxArray::multiply(state.recurrent, decay.reshape(decay_shape));

    const std::vector<int> qk_recurrent_shape{1, value_heads, 1, key_dimension};
    const std::vector<int> value_recurrent_shape{1, value_heads, value_dimension};
    MlxArray query_recurrent = query.reshape(qk_recurrent_shape);
    MlxArray key_recurrent = key.reshape(qk_recurrent_shape);
    MlxArray value_recurrent = value.reshape(value_recurrent_shape);
    MlxArray recalled = MlxArray::multiply(
        state.recurrent, key_recurrent).sum_axis(3);
    MlxArray delta = MlxArray::multiply(
        MlxArray::subtract(value_recurrent, recalled), beta);
    const std::vector<int> delta_shape{1, value_heads, value_dimension, 1};
    MlxArray update = MlxArray::multiply(delta.reshape(delta_shape), key_recurrent);
    state.recurrent = MlxArray::add(state.recurrent, update);
    MlxArray recurrent_output = MlxArray::multiply(
        state.recurrent, query_recurrent).sum_axis(3).reshape(value_shape);

    MlxArray normalized = recurrent_output.rms_norm(norm_weight_, epsilon_);
    MlxArray z = project(input, z_projection_).reshape(value_shape);
    MlxArray gate = output_gate_type_ == "sigmoid" ? z.sigmoid() : z.silu();
    MlxArray gated = MlxArray::multiply(normalized, gate);
    const std::vector<int> flat_shape{1, 1, value_width};
    return project(gated.reshape(flat_shape), output_projection_);
}

} // namespace qwen38
