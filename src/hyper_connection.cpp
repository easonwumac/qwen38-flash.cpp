#include "qwen38/hyper_connection.hpp"

#include "qwen38/quantization_geometry.hpp"

#include "hc_metal_kernels.hpp"
#include "gdn_metal_kernels.hpp"

#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {

int checked_dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid MLX dimension: ") + name);
    }
    return static_cast<int>(value);
}

MlxArray scalar_like(const float value, const mlx_dtype dtype) {
    const std::vector<float> data{value};
    const std::vector<int> shape{};
    return MlxArray::from_float32(data, shape).astype(dtype);
}

std::shared_ptr<MlxMetalKernel> hc_normalize_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"x", "norm_weight", "eps"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_read_normalize", inputs, "xn", hc_metal::normalize);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_normalize_injection_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"x", "norm_weight", "injection_weight", "eps"};
        const char* outputs[]{"xn", "ipart"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_read_normalize_injection",
            inputs,
            outputs,
            hc_metal::normalize_injection);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_down_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"xn", "weight", "scales", "biases"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_read_down", inputs, "activation", hc_metal::down);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_down_injection_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"xn", "weight", "scales", "biases", "ipart"};
        const char* outputs[]{"activation", "inject"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_read_down_injection",
            inputs,
            outputs,
            hc_metal::down_injection);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_up_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"xn", "activation", "weight", "scales", "biases"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_read_up", inputs, "mixed", hc_metal::up);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_q8_branch2_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"x", "weight", "scales", "biases"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_q8_branch2", inputs, "y",
            gdn_metal::q8_branch2, gdn_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_q8_branch2_general_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"x", "weight", "scales", "biases"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_q8_branch2_general", inputs, "y",
            gdn_metal::q8_branch2_general, gdn_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_q6_branch2_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"x", "weight", "scales", "biases"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_q6_branch2", inputs, "y",
            gdn_metal::q6_branch2, gdn_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> hc_q6_branch2_general_kernel() {
    static const auto kernel = [] {
        const char* inputs[]{"x", "weight", "scales", "biases"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_hc_q6_branch2_general", inputs, "y",
            gdn_metal::q6_branch2_general, gdn_metal::header);
    }();
    return kernel;
}

} // namespace

HyperConnection::HyperConnection(
    MlxTensorStore& tensors,
    const std::string_view prefix,
    const std::size_t hidden_size,
    const std::size_t stream_count,
    const std::size_t quantization_bits,
    const std::size_t quantization_group_size,
    const float rms_norm_epsilon,
    const bool with_injection)
    : hidden_size_(hidden_size),
      stream_count_(stream_count),
      bits_(checked_dimension(
          tensors.manifest().quantization_for(
              std::string(prefix) + ".input_mix_weight_down").bits,
          "quantization_bits")),
      group_size_(checked_dimension(
          tensors.manifest().quantization_for(
              std::string(prefix) + ".input_mix_weight_down").group_size,
          "quantization_group_size")),
      rms_norm_epsilon_(rms_norm_epsilon),
      with_injection_(with_injection),
      norm_weight_(),
      down_(load_projection(tensors, std::string(prefix) + ".input_mix_weight_down")),
      up_(load_projection(tensors, std::string(prefix) + ".input_mix_weight_up")),
      injection_(with_injection
              ? load_projection(tensors, std::string(prefix) + ".block_inject_weight")
              : Projection{}) {
    static_cast<void>(quantization_bits);
    static_cast<void>(quantization_group_size);
    static_cast<void>(checked_dimension(hidden_size_, "hidden_size"));
    static_cast<void>(checked_dimension(stream_count_, "stream_count"));
    if (!(rms_norm_epsilon_ > 0.0F)) throw std::runtime_error("RMS norm epsilon must be positive");
    MlxArray raw_norm = tensors.tensor(std::string(prefix) + ".hc_norm.weight");
    const std::vector<int> norm_shape{
        checked_dimension(stream_count_, "stream_count"),
        checked_dimension(hidden_size_, "hidden_size")};
    raw_norm = raw_norm.reshape(norm_shape);
    const std::vector<float> ones_data(hidden_size_ * stream_count_, 1.0F);
    MlxArray ones = MlxArray::from_float32(ones_data, norm_shape).astype(raw_norm.dtype());
    norm_weight_ = MlxArray::add(raw_norm, ones);
    const char* fused_read = std::getenv("QWEN38_HC_FUSED");
    const char* fused_injection = std::getenv("QWEN38_HC_FUSED_INJECTION");
    const char* dense_injection = std::getenv("QWEN38_HC_DENSE_INJECTION");
    const bool fused_injection_requested = fused_read != nullptr &&
        std::string_view(fused_read) == "1" && fused_injection != nullptr &&
        std::string_view(fused_injection) == "1";
    const bool dense_injection_requested = dense_injection != nullptr &&
        std::string_view(dense_injection) == "1";
    if (with_injection_ && injection_.quantized &&
        (fused_injection_requested || dense_injection_requested)) {
        injection_dense_ = MlxArray::multiply(
            MlxArray::dequantize(
                injection_.weight,
                injection_.scales,
                injection_.biases,
                group_size_,
                injection_.bits).transpose(),
            scalar_like(1.0F / static_cast<float>(stream_count_), raw_norm.dtype()));
        injection_dense_.eval();
        fused_injection_ready_ = true;
    }
}

HyperConnection::Projection HyperConnection::load_projection(
    MlxTensorStore& tensors,
    const std::string_view name) const {
    const std::string base(name);
    const bool quantized = tensors.manifest().has_tensor(base + ".scales");
    if (!quantized) {
        return {
            .weight = tensors.tensor(base + ".weight"),
            .scales = MlxArray{},
            .biases = MlxArray{},
            .quantized = false,
            .bits = 0,
        };
    }
    MlxArray weight = tensors.tensor(base + ".weight");
    MlxArray scales = tensors.tensor(base + ".scales");
    const int bits = infer_affine_quantization_bits(
        weight.shape(), scales.shape(), static_cast<std::size_t>(group_size_),
        "hyper-connection");
    return {
        .weight = std::move(weight),
        .scales = std::move(scales),
        .biases = tensors.tensor(base + ".biases"),
        .quantized = true,
        .bits = bits,
    };
}

MlxArray HyperConnection::project(
    const MlxArray& input,
    const Projection& projection) const {
    if (!projection.quantized) {
        return MlxArray::matmul(input, projection.weight.transpose());
    }
    return MlxArray::quantized_matmul(
        input, projection.weight, projection.scales, projection.biases,
        group_size_, projection.bits);
}

MlxArray HyperConnection::project_branch2(
    const MlxArray& input,
    const Projection& projection) const {
    const std::vector<int> input_shape = input.shape();
    const std::vector<int> weight_shape = projection.weight.shape();
    const std::vector<int> scale_shape = projection.scales.shape();
    const int input_width = scale_shape.size() == 2
        ? scale_shape[1] * group_size_ : 0;
    const int output_width = weight_shape.size() == 2 ? weight_shape[0] : 0;
    const bool supported = (projection.bits == 8 &&
                            (group_size_ == 32 || group_size_ == 64)) ||
        (projection.bits == 6 && group_size_ == 32);
    const bool aligned = input_width % 4 == 0;
    if (!projection.quantized || !supported ||
        input_shape != std::vector<int>({1, 2, input_width}) ||
        !aligned || output_width < 1) {
        throw std::runtime_error(
            "HC sibling projection requires supported affine [1,2,K]: bits=" +
            std::to_string(projection.bits) + " group=" + std::to_string(group_size_) +
            " input=" + std::to_string(input_width) +
            " output=" + std::to_string(output_width));
    }
    const std::array<const MlxArray*, 4> inputs{
        &input, &projection.weight, &projection.scales, &projection.biases};
    const std::array<MlxMetalOutputSpec, 1> outputs{{
        {.shape = {1, 2, output_width}, .dtype = input.dtype()},
    }};
    const std::array<MlxMetalDtypeTemplate, 1> dtype_templates{{
        {.name = "T", .value = input.dtype()},
    }};
    const std::array<MlxMetalIntTemplate, 3> int_templates{{
        {.name = "IN", .value = input_width},
        {.name = "OUT", .value = output_width},
        {.name = "GROUP", .value = group_size_},
    }};
    const std::shared_ptr<MlxMetalKernel>& kernel = projection.bits == 8
        ? (input_width % 256 == 0
              ? hc_q8_branch2_kernel() : hc_q8_branch2_general_kernel())
        : (input_width % 256 == 0
              ? hc_q6_branch2_kernel() : hc_q6_branch2_general_kernel());
    return std::move(kernel->apply(
        inputs, outputs,
        std::array<int, 3>{((output_width + 7) / 8) * 64, 1, 1},
        std::array<int, 3>{64, 1, 1}, dtype_templates, int_templates).front());
}

MlxArray HyperConnection::initialize_stream(
    const MlxArray& embedding,
    const std::size_t stream_count) {
    const auto dimensions = embedding.shape();
    if (dimensions.size() != 3) {
        throw std::runtime_error("hyper-connection embedding must have rank 3");
    }
    const std::vector<int> repetitions{1, 1, checked_dimension(stream_count, "stream_count")};
    return embedding.tile(repetitions);
}

HyperConnectionRead HyperConnection::read(const MlxArray& stream) const {
    const auto dimensions = stream.shape();
    if (dimensions.size() != 3 ||
        dimensions[2] != checked_dimension(hidden_size_ * stream_count_, "stream_width")) {
        throw std::runtime_error("invalid hyper-connection stream shape");
    }
    const int batch = dimensions[0];
    const int sequence = dimensions[1];
    const int streams = checked_dimension(stream_count_, "stream_count");
    const int hidden = checked_dimension(hidden_size_, "hidden_size");
    const std::vector<int> grouped_shape{batch, sequence, streams, hidden};
    const std::vector<int> flat_shape{batch, sequence, streams * hidden};

    const char* fused_flag = std::getenv("QWEN38_HC_FUSED");
    const bool fused_enabled = fused_flag != nullptr && std::string_view(fused_flag) == "1";
    const int fused_bits = down_.quantized ? down_.bits : bits_;
    const int values_per_word = 32 / fused_bits;
    const std::vector<int> down_shape = down_.weight.shape();
    const std::vector<int> up_shape = up_.weight.shape();
    const bool fused_supported =
        fused_enabled && batch == 1 && sequence == 1 &&
        (stream.dtype() == MLX_BFLOAT16 || stream.dtype() == MLX_FLOAT16) &&
        down_.quantized && up_.quantized && down_.bits == up_.bits &&
        fused_bits > 0 && 32 % fused_bits == 0 &&
        hidden % 256 == 0 && group_size_ % values_per_word == 0 &&
        down_shape.size() == 2 && up_shape.size() == 2 &&
        down_shape[1] * values_per_word == streams * hidden &&
        up_shape[0] == streams * hidden &&
        up_shape[1] * values_per_word == down_shape[0] &&
        down_shape[1] % 256 == 0 && down_shape[0] % values_per_word == 0;
    if (fused_supported) {
        const int rank = down_shape[0];
        const bool fused_injection = with_injection_ && fused_injection_ready_;
        const std::array<float, 1> epsilon_value{rms_norm_epsilon_};
        const std::array<int, 1> epsilon_shape{1};
        MlxArray epsilon = MlxArray::from_float32(epsilon_value, epsilon_shape);
        const std::array<MlxMetalDtypeTemplate, 1> dtype_templates{{
            {.name = "T", .value = stream.dtype()},
        }};
        const std::array<MlxMetalIntTemplate, 5> int_templates{{
            {.name = "HC", .value = streams},
            {.name = "H", .value = hidden},
            {.name = "R", .value = rank},
            {.name = "GS", .value = group_size_},
            {.name = "BITS", .value = fused_bits},
        }};

        const std::array<int, 3> normalize_grid{256 * streams, 1, 1};
        const std::array<int, 3> normalize_group{256, 1, 1};
        std::vector<MlxArray> normalized_outputs;
        if (fused_injection) {
            const std::array<const MlxArray*, 4> normalize_inputs{
                &stream, &norm_weight_, &injection_dense_, &epsilon};
            const std::array<MlxMetalOutputSpec, 2> normalize_output{{
                {.shape = {streams * hidden}, .dtype = stream.dtype()},
                {.shape = {streams * streams}, .dtype = MLX_FLOAT32},
            }};
            normalized_outputs = hc_normalize_injection_kernel()->apply(
                normalize_inputs,
                normalize_output,
                normalize_grid,
                normalize_group,
                dtype_templates,
                int_templates);
        } else {
            const std::array<const MlxArray*, 3> normalize_inputs{
                &stream, &norm_weight_, &epsilon};
            const std::array<MlxMetalOutputSpec, 1> normalize_output{{
                {.shape = {streams * hidden}, .dtype = stream.dtype()},
            }};
            normalized_outputs = hc_normalize_kernel()->apply(
                normalize_inputs,
                normalize_output,
                normalize_grid,
                normalize_group,
                dtype_templates,
                int_templates);
        }
        MlxArray normalized = std::move(normalized_outputs[0]);

        const std::array<int, 3> down_grid{
            256, fused_injection ? rank + streams : rank, 1};
        const std::array<int, 3> down_group{256, 1, 1};
        std::vector<MlxArray> down_outputs;
        if (fused_injection) {
            const std::array<const MlxArray*, 5> down_inputs{
                &normalized, &down_.weight, &down_.scales, &down_.biases,
                &normalized_outputs[1]};
            const std::array<MlxMetalOutputSpec, 2> down_output{{
                {.shape = {rank}, .dtype = stream.dtype()},
                {.shape = {streams}, .dtype = stream.dtype()},
            }};
            down_outputs = hc_down_injection_kernel()->apply(
                down_inputs,
                down_output,
                down_grid,
                down_group,
                dtype_templates,
                int_templates);
        } else {
            const std::array<const MlxArray*, 4> down_inputs{
                &normalized, &down_.weight, &down_.scales, &down_.biases};
            const std::array<MlxMetalOutputSpec, 1> down_output{{
                {.shape = {rank}, .dtype = stream.dtype()},
            }};
            down_outputs = hc_down_kernel()->apply(
                down_inputs,
                down_output,
                down_grid,
                down_group,
                dtype_templates,
                int_templates);
        }
        MlxArray activation = std::move(down_outputs[0]);

        const std::array<int, 3> up_grid{32, hidden, 1};
        const std::array<int, 3> up_group{32, 8, 1};
        const std::array<const MlxArray*, 5> up_inputs{
            &normalized, &activation, &up_.weight, &up_.scales, &up_.biases};
        const std::array<MlxMetalOutputSpec, 1> up_output{{
            {.shape = {hidden}, .dtype = stream.dtype()},
        }};
        std::vector<MlxArray> up_outputs = hc_up_kernel()->apply(
            up_inputs, up_output, up_grid, up_group, dtype_templates, int_templates);
        MlxArray mixed = std::move(up_outputs[0]).reshape(
            std::vector<int>{1, 1, hidden});

        if (!with_injection_) {
            return {.mixed = std::move(mixed), .injection = MlxArray{}, .has_injection = false};
        }
        if (fused_injection) {
            return {
                .mixed = std::move(mixed),
                .injection = std::move(down_outputs[1]).reshape(
                    std::vector<int>{1, 1, streams, 1}),
                .has_injection = true,
            };
        }
        MlxArray normalized_flat = normalized.reshape(flat_shape);
        MlxArray raw_injection = project(normalized_flat, injection_);
        MlxArray inv_streams = scalar_like(
            1.0F / static_cast<float>(stream_count_), raw_injection.dtype());
        MlxArray injection_gate = MlxArray::multiply(raw_injection, inv_streams).sigmoid();
        MlxArray doubled = MlxArray::multiply(
            injection_gate, scalar_like(2.0F, injection_gate.dtype()));
        return {
            .mixed = std::move(mixed),
            .injection = doubled.reshape(std::vector<int>{1, 1, streams, 1}),
            .has_injection = true,
        };
    }

    MlxArray grouped = stream.reshape(grouped_shape);
    const std::vector<float> ones_data(hidden_size_, 1.0F);
    const std::vector<int> ones_shape{hidden};
    MlxArray ones = MlxArray::from_float32(ones_data, ones_shape).astype(stream.dtype());
    MlxArray normalized = grouped.rms_norm(ones, rms_norm_epsilon_);
    MlxArray normalized_weighted = MlxArray::multiply(normalized, norm_weight_);
    MlxArray flat = normalized_weighted.reshape(flat_shape);

    MlxArray down = project(flat, down_);
    MlxArray inv_streams = scalar_like(1.0F / static_cast<float>(stream_count_), down.dtype());
    MlxArray scaled_down = MlxArray::multiply(down, inv_streams);
    MlxArray activated = scaled_down.silu();
    MlxArray up = project(activated, up_);
    MlxArray mix_gate = up.reshape(grouped_shape).sigmoid();
    MlxArray mixed_product = MlxArray::multiply(mix_gate, normalized_weighted);
    MlxArray mixed = mixed_product.mean_axis(2);

    if (!with_injection_) {
        return {.mixed = std::move(mixed), .injection = MlxArray{}, .has_injection = false};
    }
    MlxArray scaled_injection = fused_injection_ready_
        ? MlxArray::matmul(flat, injection_dense_)
        : MlxArray::multiply(project(flat, injection_), inv_streams);
    MlxArray injection_gate = scaled_injection.sigmoid();
    MlxArray two = scalar_like(2.0F, injection_gate.dtype());
    MlxArray doubled = MlxArray::multiply(injection_gate, two);
    const std::vector<int> injection_shape{batch, sequence, streams, 1};
    return {
        .mixed = std::move(mixed),
        .injection = doubled.reshape(injection_shape),
        .has_injection = true,
    };
}

std::vector<HyperConnectionRead> HyperConnection::read_branch2(
    const std::span<const MlxArray> input_streams) const {
    if (input_streams.size() != 2 || fused_injection_ready_) {
        throw std::runtime_error("exact HC sibling read requires two quantized rows");
    }
    const int streams = checked_dimension(stream_count_, "stream_count");
    const int hidden = checked_dimension(hidden_size_, "hidden_size");
    const std::vector<int> grouped_shape{1, 1, streams, hidden};
    const std::vector<int> flat_shape{1, 1, streams * hidden};
    const std::vector<float> ones_data(hidden_size_, 1.0F);
    const std::vector<int> ones_shape{hidden};
    std::vector<MlxArray> normalized;
    std::vector<MlxArray> flat;
    normalized.reserve(2);
    flat.reserve(2);
    for (const MlxArray& stream : input_streams) {
        if (stream.shape() != flat_shape) {
            throw std::runtime_error("exact HC sibling rows require [1,1,width]");
        }
        MlxArray ones = MlxArray::from_float32(ones_data, ones_shape).astype(stream.dtype());
        MlxArray weighted = MlxArray::multiply(
            stream.reshape(grouped_shape).rms_norm(ones, rms_norm_epsilon_),
            norm_weight_);
        flat.push_back(weighted.reshape(flat_shape));
        normalized.push_back(std::move(weighted));
    }
    MlxArray flat_batch = MlxArray::concatenate(flat[0], flat[1], 1);
    MlxArray down_batch = project_branch2(flat_batch, down_);
    MlxArray injection_batch;
    if (with_injection_) injection_batch = project_branch2(flat_batch, injection_);

    const std::vector<int> strides3{1, 1, 1};
    std::vector<MlxArray> activated_rows;
    activated_rows.reserve(2);
    for (int row = 0; row < 2; ++row) {
        MlxArray down = down_batch.slice(
            std::vector<int>{0, row, 0},
            std::vector<int>{1, row + 1, down_batch.shape()[2]}, strides3);
        activated_rows.push_back(MlxArray::multiply(
            down, scalar_like(
                1.0F / static_cast<float>(stream_count_), down.dtype())).silu());
    }
    MlxArray activated_batch = MlxArray::concatenate(
        activated_rows[0], activated_rows[1], 1);
    MlxArray up_batch = project_branch2(activated_batch, up_);
    std::vector<HyperConnectionRead> result;
    result.reserve(2);
    for (int row = 0; row < 2; ++row) {
        const std::vector<int> start{0, row, 0};
        MlxArray up = up_batch.slice(
            start, std::vector<int>{1, row + 1, up_batch.shape()[2]}, strides3);
        MlxArray mix_gate = up.reshape(grouped_shape).sigmoid();
        MlxArray mixed = MlxArray::multiply(
            mix_gate, normalized[static_cast<std::size_t>(row)]).mean_axis(2);
        if (!with_injection_) {
            result.push_back({
                .mixed = std::move(mixed),
                .injection = MlxArray{},
                .has_injection = false,
            });
            continue;
        }
        MlxArray raw_injection = injection_batch.slice(
            start,
            std::vector<int>{1, row + 1, injection_batch.shape()[2]},
            strides3);
        MlxArray injection_gate = MlxArray::multiply(
            raw_injection,
            scalar_like(
                1.0F / static_cast<float>(stream_count_), raw_injection.dtype())).sigmoid();
        result.push_back({
            .mixed = std::move(mixed),
            .injection = MlxArray::multiply(
                injection_gate, scalar_like(2.0F, injection_gate.dtype())).reshape(
                    std::vector<int>{1, 1, streams, 1}),
            .has_injection = true,
        });
    }
    return result;
}

MlxArray HyperConnection::write(
    const MlxArray& stream,
    const MlxArray& block_output,
    const MlxArray& injection) const {
    const auto dimensions = stream.shape();
    if (dimensions.size() != 3) throw std::runtime_error("hyper-connection stream must have rank 3");
    const int batch = dimensions[0];
    const int sequence = dimensions[1];
    const int streams = checked_dimension(stream_count_, "stream_count");
    const int hidden = checked_dimension(hidden_size_, "hidden_size");
    const std::vector<int> grouped_shape{batch, sequence, streams, hidden};
    const std::vector<int> output_shape{batch, sequence, 1, hidden};
    const std::vector<int> flat_shape{batch, sequence, streams * hidden};
    MlxArray grouped = stream.reshape(grouped_shape);
    MlxArray output = block_output.reshape(output_shape);
    MlxArray update = MlxArray::multiply(output, injection);
    return MlxArray::add(grouped, update).reshape(flat_shape);
}

} // namespace qwen38
