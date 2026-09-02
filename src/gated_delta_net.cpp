#include "qwen38/gated_delta_net.hpp"

#include "gdn_metal_kernels.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
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

MlxArray slice_sequence_row(const MlxArray& batch, const std::size_t row) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() < 2 || row >= static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("GatedDeltaNet verifier row is out of range");
    }
    std::vector<int> start(shape.size(), 0);
    std::vector<int> stop = shape;
    std::vector<int> strides(shape.size(), 1);
    start[1] = static_cast<int>(row);
    stop[1] = static_cast<int>(row + 1);
    return batch.slice(start, stop, strides);
}

MlxArray concatenate_sequence_rows(const std::vector<MlxArray>& rows) {
    if (rows.empty()) throw std::runtime_error("GatedDeltaNet verifier has no outputs");
    MlxArray result = rows.front().share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, rows[row], 1);
    }
    return result;
}

std::shared_ptr<MlxMetalKernel> prefill_recurrence_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"q", "k", "v", "g", "beta", "state_in", "T"};
        const char* outputs[]{"y", "state_out"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_gated_delta_prefill",
            inputs,
            outputs,
            gdn_metal::recurrence,
            gdn_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> verify_recurrence_bf16_sum_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"q", "k", "v", "g", "beta", "state_in"};
        const char* outputs[]{"y", "state_rows"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_gdn_verify_recurrence_bf16_sum",
            inputs,
            outputs,
            gdn_metal::verify_recurrence_bf16_sum,
            gdn_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> rollback_recurrence_bf16_sum_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"k", "v", "g", "beta", "state_in"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_gdn_rollback_recurrence_bf16_sum",
            inputs,
            "state_out",
            gdn_metal::rollback_recurrence_bf16_sum,
            gdn_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> prework_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{
            "qkv",
            "conv_state",
            "conv_weight",
            "q_scale",
            "k_scale",
            "beta_in",
            "decay_in",
            "decay_log",
            "decay_bias",
        };
        const char* outputs[]{"q_out", "k_out", "v_out", "conv_out", "decay_out", "beta_out"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_gdn_prework",
            inputs,
            outputs,
            gdn_metal::prework,
            gdn_metal::prework_header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> norm_gate_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"y", "z", "norm_weight", "epsilon"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_gdn_norm_gate", inputs, "output", gdn_metal::norm_gate);
    }();
    return kernel;
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

void GatedDeltaNet::materialize_rollback(GatedDeltaNetState& state) const {
    if (state.rollback_rows == 0) return;
    if (!state.initialized || state.rollback_rows > 64) {
        throw std::runtime_error("invalid deferred GatedDeltaNet rollback state");
    }
    const int value_heads = dimension(value_head_count_, "value_head_count");
    const int key_dimension = dimension(key_head_dimension_, "key_head_dimension");
    const int value_dimension = dimension(value_head_dimension_, "value_head_dimension");
    const std::array<const MlxArray*, 5> inputs{
        &state.rollback_key,
        &state.rollback_value,
        &state.rollback_decay,
        &state.rollback_beta,
        &state.recurrent,
    };
    const std::array<MlxMetalOutputSpec, 1> outputs{{
        {.shape = {1, value_heads, value_dimension, key_dimension},
         .dtype = state.recurrent.dtype()},
    }};
    const std::array<int, 3> grid{32, value_dimension, value_heads};
    const std::array<int, 3> threadgroup{32, 4, 1};
    const std::array<MlxMetalDtypeTemplate, 2> dtype_templates{{
        {.name = "InT", .value = state.rollback_key.dtype()},
        {.name = "StT", .value = state.recurrent.dtype()},
    }};
    const std::array<MlxMetalIntTemplate, 4> int_templates{{
        {.name = "DK", .value = key_dimension},
        {.name = "DV", .value = value_dimension},
        {.name = "HV", .value = value_heads},
        {.name = "ROWS", .value = static_cast<int>(state.rollback_rows)},
    }};
    std::vector<MlxArray> rollback = rollback_recurrence_bf16_sum_kernel()->apply(
        inputs, outputs, grid, threadgroup, dtype_templates, int_templates);
    state.recurrent = std::move(rollback[0]);
    state.rollback_key = {};
    state.rollback_value = {};
    state.rollback_decay = {};
    state.rollback_beta = {};
    state.rollback_rows = 0;
}

MlxArray GatedDeltaNet::forward_decode(
    const MlxArray& input,
    GatedDeltaNetState& state) const {
    materialize_rollback(state);
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
    const std::vector<int> qk_shape{1, 1, key_heads, key_dimension};
    const std::vector<int> value_shape{1, 1, value_heads, value_dimension};
    MlxArray query;
    MlxArray key;
    MlxArray value;
    MlxArray beta;
    MlxArray decay;
    const char* prework_flag = std::getenv("QWEN38_GDN_PREWORK");
    const bool fused_prework =
        prework_flag != nullptr && std::string_view(prework_flag) == "1" &&
        key_dimension == 128 && value_dimension == 128 && qkv.dtype() == MLX_BFLOAT16 &&
        state.convolution.dtype() == MLX_BFLOAT16 &&
        convolution_weight_.dtype() == MLX_BFLOAT16 && decay_bias_.dtype() == MLX_BFLOAT16;
    if (fused_prework) {
        MlxArray beta_input = project(input, beta_projection_);
        MlxArray decay_input = project(input, decay_projection_);
        MlxArray query_scale = scalar(
            1.0F / static_cast<float>(key_head_dimension_), qkv.dtype());
        MlxArray key_scale = scalar(
            1.0F / std::sqrt(static_cast<float>(key_head_dimension_)), qkv.dtype());
        const std::array<const MlxArray*, 9> inputs{
            &qkv,
            &state.convolution,
            &convolution_weight_,
            &query_scale,
            &key_scale,
            &beta_input,
            &decay_input,
            &decay_log_,
            &decay_bias_,
        };
        const std::array<MlxMetalOutputSpec, 6> outputs{{
            {.shape = qk_shape, .dtype = qkv.dtype()},
            {.shape = qk_shape, .dtype = qkv.dtype()},
            {.shape = value_shape, .dtype = qkv.dtype()},
            {.shape = {1, 3, convolution_width}, .dtype = qkv.dtype()},
            {.shape = {1, 1, value_heads}, .dtype = qkv.dtype()},
            {.shape = {1, 1, value_heads}, .dtype = qkv.dtype()},
        }};
        const std::array<int, 3> grid{32, 1, 2 * key_heads + value_heads};
        const std::array<int, 3> threadgroup{32, 1, 1};
        const std::array<MlxMetalDtypeTemplate, 1> dtype_templates{{
            {.name = "T", .value = qkv.dtype()},
        }};
        const std::array<MlxMetalIntTemplate, 5> int_templates{{
            {.name = "HK", .value = key_heads},
            {.name = "HV", .value = value_heads},
            {.name = "DK", .value = key_dimension},
            {.name = "DV", .value = value_dimension},
            {.name = "C", .value = convolution_width},
        }};
        std::vector<MlxArray> fused = prework_kernel()->apply(
            inputs, outputs, grid, threadgroup, dtype_templates, int_templates);
        query = std::move(fused[0]);
        key = std::move(fused[1]);
        value = std::move(fused[2]);
        state.convolution = std::move(fused[3]);
        decay = std::move(fused[4]);
        beta = std::move(fused[5]);
    } else {
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
        query = convolved.slice(
            std::vector<int>{0, 0, 0},
            std::vector<int>{1, 1, key_width},
            strides).reshape(qk_shape);
        key = convolved.slice(
            std::vector<int>{0, 0, key_width},
            std::vector<int>{1, 1, 2 * key_width},
            strides).reshape(qk_shape);
        value = convolved.slice(
            std::vector<int>{0, 0, 2 * key_width},
            std::vector<int>{1, 1, convolution_width},
            strides).reshape(value_shape);
        const std::vector<float> ones_data(key_head_dimension_, 1.0F);
        const std::vector<int> norm_shape{key_dimension};
        MlxArray ones = MlxArray::from_float32(ones_data, norm_shape).astype(query.dtype());
        query = MlxArray::multiply(
            query.rms_norm(ones, 1.0e-6F),
            scalar(1.0F / static_cast<float>(key_head_dimension_), query.dtype()));
        key = MlxArray::multiply(
            key.rms_norm(ones, 1.0e-6F),
            scalar(1.0F / std::sqrt(static_cast<float>(key_head_dimension_)), key.dtype()));
        beta = project(input, beta_projection_).sigmoid();
        MlxArray decay_input = MlxArray::add(project(input, decay_projection_), decay_bias_);
        MlxArray softplus = decay_input.astype(MLX_FLOAT32).exp().log1p();
        MlxArray decay_rate = decay_log_.astype(MLX_FLOAT32).exp();
        decay = MlxArray::multiply(
            decay_rate, softplus).negative().exp().astype(qkv.dtype());
    }
    const int repetition = value_heads / key_heads;
    query = query.repeat_axis(repetition, 2);
    key = key.repeat_axis(repetition, 2);

    const std::vector<int> beta_shape{1, value_heads, 1};
    beta = beta.reshape(beta_shape);
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

    MlxArray z = project(input, z_projection_).reshape(value_shape);
    const std::vector<int> flat_shape{1, 1, value_width};
    const char* fused_gate = std::getenv("QWEN38_GDN_NORM_GATE");
    if (fused_gate != nullptr && std::string_view(fused_gate) == "1" &&
        value_dimension == 128 && recurrent_output.dtype() == MLX_BFLOAT16 &&
        z.dtype() == MLX_BFLOAT16 && norm_weight_.dtype() == MLX_BFLOAT16) {
        const std::array<float, 1> epsilon_value{epsilon_};
        const std::array<int, 1> epsilon_shape{1};
        MlxArray epsilon = MlxArray::from_float32(epsilon_value, epsilon_shape);
        const std::array<const MlxArray*, 4> inputs{
            &recurrent_output, &z, &norm_weight_, &epsilon};
        const std::array<MlxMetalOutputSpec, 1> outputs{{
            {.shape = flat_shape, .dtype = recurrent_output.dtype()},
        }};
        const std::array<int, 3> grid{32, 1, value_heads};
        const std::array<int, 3> threadgroup{32, 1, 1};
        const std::array<MlxMetalDtypeTemplate, 1> dtype_templates{{
            {.name = "T", .value = recurrent_output.dtype()},
        }};
        const std::array<MlxMetalIntTemplate, 3> int_templates{{
            {.name = "HV", .value = value_heads},
            {.name = "DV", .value = value_dimension},
            {.name = "SWISH", .value = output_gate_type_ == "sigmoid" ? 0 : 1},
        }};
        std::vector<MlxArray> fused = norm_gate_kernel()->apply(
            inputs, outputs, grid, threadgroup, dtype_templates, int_templates);
        return project(fused[0], output_projection_);
    }
    MlxArray normalized = recurrent_output.rms_norm(norm_weight_, epsilon_);
    MlxArray gate = output_gate_type_ == "sigmoid" ? z.sigmoid() : z.silu();
    return project(MlxArray::multiply(normalized, gate).reshape(flat_shape), output_projection_);
}

MlxArray GatedDeltaNet::forward_verify(
    const MlxArray& input,
    const GatedDeltaNetState& origin,
    std::vector<GatedDeltaNetState>& checkpoints) const {
    if (origin.rollback_rows != 0) {
        throw std::runtime_error("GatedDeltaNet verifier received deferred rollback state");
    }
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] < 1 ||
        input_shape[1] > 64) {
        throw std::runtime_error("GatedDeltaNet batch requires shape [1,S,hidden], S=1..64");
    }
    const std::size_t rows = static_cast<std::size_t>(input_shape[1]);
    const int key_heads = dimension(key_head_count_, "key_head_count");
    const int value_heads = dimension(value_head_count_, "value_head_count");
    const int key_dimension = dimension(key_head_dimension_, "key_head_dimension");
    const int value_dimension = dimension(value_head_dimension_, "value_head_dimension");
    const int key_width = key_heads * key_dimension;
    const int value_width = value_heads * value_dimension;
    const int convolution_width = 2 * key_width + value_width;

    MlxArray qkv = project(input, qkv_projection_);
    MlxArray convolution_state;
    MlxArray recurrent;
    if (origin.initialized) {
        convolution_state = origin.convolution.share();
        recurrent = origin.recurrent.share();
    } else {
        convolution_state = MlxArray::zeros(
            std::vector<int>{1, 3, convolution_width}, qkv.dtype());
        recurrent = MlxArray::zeros(
            std::vector<int>{1, value_heads, value_dimension, key_dimension}, qkv.dtype());
    }
    MlxArray convolution_input = MlxArray::concatenate(convolution_state, qkv, 1);
    MlxArray convolved = MlxArray::conv1d(
        convolution_input, convolution_weight_, 1, 0, 1, convolution_width).silu();

    const std::vector<int> strides{1, 1, 1};
    MlxArray query = convolved.slice(
        std::vector<int>{0, 0, 0},
        std::vector<int>{1, input_shape[1], key_width},
        strides).reshape(std::vector<int>{1, input_shape[1], key_heads, key_dimension});
    MlxArray key = convolved.slice(
        std::vector<int>{0, 0, key_width},
        std::vector<int>{1, input_shape[1], 2 * key_width},
        strides).reshape(std::vector<int>{1, input_shape[1], key_heads, key_dimension});
    MlxArray value = convolved.slice(
        std::vector<int>{0, 0, 2 * key_width},
        std::vector<int>{1, input_shape[1], convolution_width},
        strides).reshape(std::vector<int>{1, input_shape[1], value_heads, value_dimension});
    const std::vector<float> ones_data(key_head_dimension_, 1.0F);
    MlxArray ones = MlxArray::from_float32(
        ones_data, std::vector<int>{key_dimension}).astype(query.dtype());
    query = query.rms_norm(ones, 1.0e-6F);
    key = key.rms_norm(ones, 1.0e-6F);
    query = MlxArray::multiply(
        query, scalar(1.0F / static_cast<float>(key_head_dimension_), query.dtype()));
    key = MlxArray::multiply(
        key,
        scalar(1.0F / std::sqrt(static_cast<float>(key_head_dimension_)), key.dtype()));
    const int repetition = value_heads / key_heads;
    query = query.repeat_axis(repetition, 2);
    key = key.repeat_axis(repetition, 2);

    MlxArray beta = project(input, beta_projection_).sigmoid();
    MlxArray decay_input = MlxArray::add(project(input, decay_projection_), decay_bias_);
    MlxArray softplus = decay_input.astype(MLX_FLOAT32).exp().log1p();
    MlxArray decay_rate = decay_log_.astype(MLX_FLOAT32).exp();
    MlxArray decay = MlxArray::multiply(decay_rate, softplus).negative().exp().astype(qkv.dtype());
    MlxArray z = project(input, z_projection_).reshape(
        std::vector<int>{1, input_shape[1], value_heads, value_dimension});

    const char* metal_verify = std::getenv("QWEN38_GDN_METAL_VERIFY_BF16_SUM");
    if (metal_verify != nullptr && std::string_view(metal_verify) == "1") {
        const char* compact_rollback_env = std::getenv("QWEN38_COMPACT_GDN_ROLLBACK");
        const bool compact_rollback = compact_rollback_env != nullptr &&
            std::string_view(compact_rollback_env) == "1";
        const std::array<const MlxArray*, 6> inputs{
            &query, &key, &value, &decay, &beta, &recurrent};
        const std::array<MlxMetalOutputSpec, 2> outputs{{
            {.shape = {1, input_shape[1], value_heads, value_dimension},
             .dtype = qkv.dtype()},
            {.shape = {compact_rollback ? 1 : input_shape[1], 1, value_heads,
                       value_dimension, key_dimension},
             .dtype = recurrent.dtype()},
        }};
        const std::array<int, 3> grid{32, value_dimension, value_heads};
        const std::array<int, 3> threadgroup{32, 4, 1};
        const std::array<MlxMetalDtypeTemplate, 3> dtype_templates{{
            {.name = "InT", .value = query.dtype()},
            {.name = "StT", .value = recurrent.dtype()},
            {.name = "OutT", .value = qkv.dtype()},
        }};
        const std::array<MlxMetalIntTemplate, 5> int_templates{{
            {.name = "DK", .value = key_dimension},
            {.name = "DV", .value = value_dimension},
            {.name = "HV", .value = value_heads},
            {.name = "ROWS", .value = input_shape[1]},
            {.name = "STORE_ROWS", .value = compact_rollback ? 0 : 1},
        }};
        std::vector<MlxArray> recurrence = verify_recurrence_bf16_sum_kernel()->apply(
            inputs, outputs, grid, threadgroup, dtype_templates, int_templates);
        checkpoints.clear();
        checkpoints.resize(rows);
        const std::vector<int> state_strides{1, 1, 1, 1, 1};
        for (std::size_t row = 0; row < rows; ++row) {
            const int begin = static_cast<int>(row + 1);
            checkpoints[row].convolution = convolution_input.slice(
                std::vector<int>{0, begin, 0},
                std::vector<int>{1, begin + 3, convolution_width},
                strides);
            if (compact_rollback && row + 1 != rows) {
                checkpoints[row].recurrent = recurrent.share();
                checkpoints[row].rollback_key = key.share();
                checkpoints[row].rollback_value = value.share();
                checkpoints[row].rollback_decay = decay.share();
                checkpoints[row].rollback_beta = beta.share();
                checkpoints[row].rollback_rows = row + 1;
            } else {
                const int state_row = compact_rollback ? 0 : static_cast<int>(row);
                checkpoints[row].recurrent = recurrence[1].slice(
                    std::vector<int>{state_row, 0, 0, 0, 0},
                    std::vector<int>{state_row + 1, 1, value_heads,
                                     value_dimension, key_dimension},
                    state_strides).reshape(
                        std::vector<int>{1, value_heads, value_dimension, key_dimension});
            }
            checkpoints[row].initialized = true;
        }
        MlxArray normalized = recurrence[0].rms_norm(norm_weight_, epsilon_);
        MlxArray gate = output_gate_type_ == "sigmoid" ? z.sigmoid() : z.silu();
        return project(
            MlxArray::multiply(normalized, gate).reshape(
                std::vector<int>{1, input_shape[1], value_width}),
            output_projection_);
    }

    checkpoints.clear();
    checkpoints.resize(rows);
    std::vector<MlxArray> gated_rows;
    gated_rows.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        MlxArray query_row = slice_sequence_row(query, row).reshape(
            std::vector<int>{1, value_heads, 1, key_dimension});
        MlxArray key_row = slice_sequence_row(key, row).reshape(
            std::vector<int>{1, value_heads, 1, key_dimension});
        MlxArray value_row = slice_sequence_row(value, row).reshape(
            std::vector<int>{1, value_heads, value_dimension});
        MlxArray beta_row = slice_sequence_row(beta, row).reshape(
            std::vector<int>{1, value_heads, 1});
        MlxArray decay_row = slice_sequence_row(decay, row).reshape(
            std::vector<int>{1, value_heads, 1, 1});
        recurrent = MlxArray::multiply(recurrent, decay_row);
        MlxArray recalled = MlxArray::multiply(recurrent, key_row).sum_axis(3);
        MlxArray delta = MlxArray::multiply(
            MlxArray::subtract(value_row, recalled), beta_row);
        MlxArray update = MlxArray::multiply(
            delta.reshape(std::vector<int>{1, value_heads, value_dimension, 1}), key_row);
        recurrent = MlxArray::add(recurrent, update);
        MlxArray recurrent_output = MlxArray::multiply(recurrent, query_row).sum_axis(3).reshape(
            std::vector<int>{1, 1, value_heads, value_dimension});
        MlxArray normalized = recurrent_output.rms_norm(norm_weight_, epsilon_);
        MlxArray z_row = slice_sequence_row(z, row).reshape(
            std::vector<int>{1, 1, value_heads, value_dimension});
        MlxArray gate = output_gate_type_ == "sigmoid" ? z_row.sigmoid() : z_row.silu();
        gated_rows.push_back(MlxArray::multiply(normalized, gate).reshape(
            std::vector<int>{1, 1, value_width}));

        const int begin = static_cast<int>(row + 1);
        checkpoints[row].convolution = convolution_input.slice(
            std::vector<int>{0, begin, 0},
            std::vector<int>{1, begin + 3, convolution_width},
            strides);
        checkpoints[row].recurrent = recurrent.share();
        checkpoints[row].initialized = true;
    }
    return project(concatenate_sequence_rows(gated_rows), output_projection_);
}

MlxArray GatedDeltaNet::forward_prefill(
    const MlxArray& input,
    GatedDeltaNetState& state) const {
    materialize_rollback(state);
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] < 2 ||
        input_shape[1] > 1024) {
        throw std::runtime_error("GatedDeltaNet prefill requires shape [1,S,hidden], S=2..1024");
    }
    const char* enabled = std::getenv("QWEN38_GDN_METAL_PREFILL");
    if (enabled == nullptr || std::string_view(enabled) != "1") {
        std::vector<GatedDeltaNetState> checkpoints;
        MlxArray output = forward_verify(input, state, checkpoints);
        state = std::move(checkpoints.back());
        return output;
    }

    const int rows = input_shape[1];
    const int key_heads = dimension(key_head_count_, "key_head_count");
    const int value_heads = dimension(value_head_count_, "value_head_count");
    const int key_dimension = dimension(key_head_dimension_, "key_head_dimension");
    const int value_dimension = dimension(value_head_dimension_, "value_head_dimension");
    if (key_dimension % 32 != 0 || value_heads % key_heads != 0) {
        throw std::runtime_error("unsupported GatedDeltaNet Metal prefill geometry");
    }
    const int key_width = key_heads * key_dimension;
    const int value_width = value_heads * value_dimension;
    const int convolution_width = 2 * key_width + value_width;

    MlxArray qkv = project(input, qkv_projection_);
    MlxArray convolution_state = state.initialized
        ? state.convolution.share()
        : MlxArray::zeros(std::vector<int>{1, 3, convolution_width}, qkv.dtype());
    MlxArray recurrent = state.initialized
        ? state.recurrent.share()
        : MlxArray::zeros(
              std::vector<int>{1, value_heads, value_dimension, key_dimension},
              qkv.dtype());
    MlxArray convolution_input = MlxArray::concatenate(convolution_state, qkv, 1);
    MlxArray convolved = MlxArray::conv1d(
        convolution_input, convolution_weight_, 1, 0, 1, convolution_width).silu();

    const std::vector<int> strides{1, 1, 1};
    MlxArray query = convolved.slice(
        std::vector<int>{0, 0, 0},
        std::vector<int>{1, rows, key_width},
        strides).reshape(std::vector<int>{1, rows, key_heads, key_dimension});
    MlxArray key = convolved.slice(
        std::vector<int>{0, 0, key_width},
        std::vector<int>{1, rows, 2 * key_width},
        strides).reshape(std::vector<int>{1, rows, key_heads, key_dimension});
    MlxArray value = convolved.slice(
        std::vector<int>{0, 0, 2 * key_width},
        std::vector<int>{1, rows, convolution_width},
        strides).reshape(std::vector<int>{1, rows, value_heads, value_dimension});
    const std::vector<float> ones_data(key_head_dimension_, 1.0F);
    MlxArray ones = MlxArray::from_float32(
        ones_data, std::vector<int>{key_dimension}).astype(query.dtype());
    query = MlxArray::multiply(
        query.rms_norm(ones, 1.0e-6F),
        scalar(1.0F / static_cast<float>(key_head_dimension_), query.dtype()));
    key = MlxArray::multiply(
        key.rms_norm(ones, 1.0e-6F),
        scalar(1.0F / std::sqrt(static_cast<float>(key_head_dimension_)), key.dtype()));

    MlxArray beta = project(input, beta_projection_).sigmoid();
    MlxArray decay_input = MlxArray::add(project(input, decay_projection_), decay_bias_);
    MlxArray softplus = decay_input.astype(MLX_FLOAT32).exp().log1p();
    MlxArray decay_rate = decay_log_.astype(MLX_FLOAT32).exp();
    MlxArray decay = MlxArray::multiply(
        decay_rate, softplus).negative().exp().astype(qkv.dtype());
    const std::array<std::int32_t, 1> row_value{rows};
    const std::array<int, 0> scalar_shape{};
    MlxArray row_count = MlxArray::from_int32(row_value, scalar_shape);

    const std::array<const MlxArray*, 7> inputs{
        &query, &key, &value, &decay, &beta, &recurrent, &row_count};
    const std::array<MlxMetalOutputSpec, 2> outputs{{
        {.shape = {1, rows, value_heads, value_dimension}, .dtype = qkv.dtype()},
        {.shape = {1, value_heads, value_dimension, key_dimension},
         .dtype = recurrent.dtype()},
    }};
    const std::array<int, 3> grid{32, value_dimension, value_heads};
    const std::array<int, 3> threadgroup{32, 4, 1};
    const std::array<MlxMetalDtypeTemplate, 3> dtype_templates{{
        {.name = "InT", .value = query.dtype()},
        {.name = "StT", .value = recurrent.dtype()},
        {.name = "OutT", .value = qkv.dtype()},
    }};
    const std::array<MlxMetalIntTemplate, 4> int_templates{{
        {.name = "Dk", .value = key_dimension},
        {.name = "Dv", .value = value_dimension},
        {.name = "Hk", .value = key_heads},
        {.name = "Hv", .value = value_heads},
    }};
    std::vector<MlxArray> recurrence = prefill_recurrence_kernel()->apply(
        inputs, outputs, grid, threadgroup, dtype_templates, int_templates);

    state.convolution = convolution_input.slice(
        std::vector<int>{0, rows, 0},
        std::vector<int>{1, rows + 3, convolution_width},
        strides);
    state.recurrent = std::move(recurrence[1]);
    state.initialized = true;

    MlxArray normalized = recurrence[0].rms_norm(norm_weight_, epsilon_);
    MlxArray z = project(input, z_projection_).reshape(
        std::vector<int>{1, rows, value_heads, value_dimension});
    MlxArray gate = output_gate_type_ == "sigmoid" ? z.sigmoid() : z.silu();
    MlxArray gated = MlxArray::multiply(normalized, gate);
    return project(
        gated.reshape(std::vector<int>{1, rows, value_width}),
        output_projection_);
}

} // namespace qwen38
