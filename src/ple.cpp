#include "qwen38/ple.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid PLE dimension: ") + name);
    }
    return static_cast<int>(value);
}

MlxArray scalar(const float value, const mlx_dtype dtype) {
    const std::vector<float> values{value};
    const std::vector<int> shape{};
    return MlxArray::from_float32(values, shape).astype(dtype);
}

MlxArray offset_norm_weight(
    MlxArray raw,
    const std::size_t stream_count,
    const std::size_t hidden_size) {
    const std::vector<int> shape{
        dimension(stream_count, "stream count"), dimension(hidden_size, "hidden size")};
    raw = raw.reshape(shape);
    const std::vector<float> ones_data(stream_count * hidden_size, 1.0F);
    MlxArray ones = MlxArray::from_float32(ones_data, shape).astype(raw.dtype());
    return MlxArray::add(raw, ones);
}

} // namespace

Ple::Ple(
    MlxTensorStore& tensors,
    const std::string_view prefix,
    const ModelConfig& config)
    : hidden_size_(config.hidden_size),
      stream_count_(config.hyper_connection_count),
      convolution_state_length_(
          (config.ple_convolution_kernel_size - 1) * config.ngram_size),
      bits_(dimension(config.quantization_bits, "quantization bits")),
      group_size_(dimension(config.quantization_group_size, "quantization group size")),
      epsilon_(static_cast<float>(config.rms_norm_epsilon)),
      hash_(config),
      table_(tensors.manifest().directory(), hash_.total_rows()),
      key_projection_(load_projection(tensors, std::string(prefix) + ".key_proj")),
      value_projection_(load_projection(tensors, std::string(prefix) + ".value_proj")),
      norm_key_(offset_norm_weight(
          tensors.tensor(std::string(prefix) + ".norm_key.weight"),
          config.hyper_connection_count,
          config.hidden_size)),
      norm_query_(offset_norm_weight(
          tensors.tensor(std::string(prefix) + ".norm_query.weight"),
          config.hyper_connection_count,
          config.hidden_size)),
      norm_convolution_(offset_norm_weight(
          tensors.tensor(std::string(prefix) + ".norm_conv.weight"),
          config.hyper_connection_count,
          config.hidden_size)),
      // The converted checkpoint retains the PyTorch depthwise-convolution
      // layout [channels, 1, kernel]. MLX conv1d consumes
      // [output_channels, kernel, input_channels/groups].
      convolution_weight_(tensors.tensor(std::string(prefix) + ".conv1d.weight").swapaxes(1, 2)) {
    if (convolution_state_length_ != 9) {
        throw std::runtime_error("the retained PLE requires a 9-token convolution state");
    }
}

Ple::QuantizedProjection Ple::load_projection(
    MlxTensorStore& tensors,
    const std::string_view name) {
    const std::string base(name);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
    };
}

MlxArray Ple::project(
    const MlxArray& input,
    const QuantizedProjection& projection) const {
    return MlxArray::quantized_matmul(
        input, projection.weight, projection.scales, projection.biases, group_size_, bits_);
}

MlxArray Ple::grouped_norm(const MlxArray& input, const MlxArray& weight) const {
    const auto shape = input.shape();
    const int streams = dimension(stream_count_, "stream count");
    const int hidden = dimension(hidden_size_, "hidden size");
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1 ||
        shape[2] != streams * hidden) {
        throw std::runtime_error("PLE grouped norm shape mismatch");
    }
    const std::vector<int> grouped_shape{1, 1, streams, hidden};
    const std::vector<float> ones_data(hidden_size_, 1.0F);
    const std::vector<int> ones_shape{hidden};
    MlxArray ones = MlxArray::from_float32(ones_data, ones_shape).astype(input.dtype());
    return MlxArray::multiply(
        input.reshape(grouped_shape).rms_norm(ones, epsilon_), weight);
}

MlxArray Ple::forward_decode(
    const MlxArray& stream,
    const std::uint32_t token,
    PleState& state) const {
    const auto row_ids = hash_.row_ids(token, state.ngram);
    const std::vector<float> host_embedding = table_.gather(row_ids);
    const std::vector<int> embedding_shape{1, 1, dimension(hidden_size_, "hidden size")};
    MlxArray embedding = MlxArray::from_float32(
        host_embedding, embedding_shape).astype(stream.dtype());
    const int streams = dimension(stream_count_, "stream count");
    const int hidden = dimension(hidden_size_, "hidden size");
    const std::vector<int> flat_shape{1, 1, streams * hidden};
    MlxArray key = grouped_norm(project(embedding, key_projection_), norm_key_);
    MlxArray query = grouped_norm(stream, norm_query_);
    MlxArray gate = MlxArray::multiply(key, query).sum_axis(3);
    MlxArray scale = scalar(1.0F / std::sqrt(static_cast<float>(hidden_size_)), gate.dtype());
    gate = MlxArray::multiply(gate, scale);
    MlxArray floor = scalar(1.0e-6F, gate.dtype());
    MlxArray signed_root = MlxArray::multiply(
        MlxArray::maximum(gate.absolute(), floor).square_root(), gate.sign());
    const std::vector<int> gate_shape{1, 1, streams, 1};
    MlxArray value = project(embedding, value_projection_);
    const std::vector<int> value_shape{1, 1, 1, hidden};
    MlxArray gated = MlxArray::multiply(
        signed_root.sigmoid().reshape(gate_shape), value.reshape(value_shape)).reshape(flat_shape);
    MlxArray normalized = grouped_norm(gated, norm_convolution_).reshape(flat_shape);
    if (!state.convolution_initialized) {
        const std::vector<int> state_shape{
            1, dimension(convolution_state_length_, "convolution state"), streams * hidden};
        state.convolution = MlxArray::zeros(state_shape, stream.dtype());
        state.convolution_initialized = true;
    }
    MlxArray convolution_input = MlxArray::concatenate(state.convolution, normalized, 1);
    const std::vector<int> state_start{0, 1, 0};
    const std::vector<int> state_stop{1, 10, streams * hidden};
    const std::vector<int> state_strides{1, 1, 1};
    state.convolution = convolution_input.slice(state_start, state_stop, state_strides);
    MlxArray convolution = MlxArray::conv1d(
        convolution_input,
        convolution_weight_,
        1,
        0,
        3,
        streams * hidden).silu();
    return MlxArray::add(gated, convolution);
}

} // namespace qwen38
