#include "w8a8_projection.hpp"

#include "w8a8_gdn_probe_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38::devtools {
namespace {

int checked_dimension(const int value, const char* name) {
    if (value <= 0) throw std::runtime_error(std::string("invalid W8A8 ") + name);
    return value;
}

std::shared_ptr<MlxMetalKernel> quantizer() {
    static const auto kernel = [] {
        const char* inputs[]{"x"};
        const char* outputs[]{"aq", "scale"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_dev_w8a8_quantize", inputs, outputs,
            w8a8_probe_metal::quantize, w8a8_probe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> mpp_projection() {
    static const auto kernel = [] {
        const char* inputs[]{"aq", "w", "scale", "wscale"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_dev_w8a8_mpp", inputs, "y",
            w8a8_probe_metal::mpp_fused, w8a8_probe_metal::mpp_header);
    }();
    return kernel;
}

} // namespace

W8A8Projection::W8A8Projection(
    const MlxArray& q4_weight,
    const MlxArray& q4_scales,
    const MlxArray& q4_biases,
    const int group_size,
    const int bits) {
    checked_dimension(group_size, "group size");
    checked_dimension(bits, "bit width");
    MlxArray dequantized = MlxArray::dequantize(
        q4_weight, q4_scales, q4_biases, group_size, bits, MLX_FLOAT32);
    const std::vector<int> shape = dequantized.shape();
    if (shape.size() != 2) throw std::runtime_error("W8A8 weight must have rank two");
    n_ = checked_dimension(shape[0], "output width");
    k_ = checked_dimension(shape[1], "input width");
    if (n_ % 128 != 0 || k_ % 16 != 0)
        throw std::runtime_error("W8A8 MPP requires N divisible by 128 and K divisible by 16");

    const std::vector<float> source = dequantized.astype(MLX_FLOAT32).to_float32();
    std::vector<std::int32_t> values(source.size());
    std::vector<float> scales(static_cast<std::size_t>(n_));
    const std::size_t k_size = static_cast<std::size_t>(k_);
    for (int row = 0; row < n_; ++row) {
        float absmax = 0.0F;
        for (int column = 0; column < k_; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * k_size +
                                      static_cast<std::size_t>(column);
            absmax = std::max(absmax, std::abs(source[index]));
        }
        const float scale = absmax == 0.0F ? 1.0F : absmax / 127.0F;
        scales[static_cast<std::size_t>(row)] = scale;
        for (int column = 0; column < k_; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * k_size +
                                      static_cast<std::size_t>(column);
            values[index] = static_cast<std::int32_t>(std::clamp(
                std::round(source[index] / scale), -128.0F, 127.0F));
        }
    }
    const std::array<int, 2> weight_shape{n_, k_};
    const std::array<int, 1> scale_shape{n_};
    weight_ = MlxArray::from_int32(values, weight_shape).astype(MLX_INT8);
    scales_ = MlxArray::from_float32(scales, scale_shape);
    weight_.eval();
    scales_.eval();
    sidecar_bytes_ = static_cast<std::size_t>(n_) * k_size +
                     static_cast<std::size_t>(n_) * sizeof(float);
}

MlxArray W8A8Projection::apply(const MlxArray& input) const {
    const std::vector<int> input_shape = input.shape();
    if (input_shape.empty() || input_shape.back() != k_)
        throw std::runtime_error("W8A8 projection input shape mismatch");
    const std::size_t elements = input.size();
    const std::size_t k_size = static_cast<std::size_t>(k_);
    if (elements % k_size != 0 || elements / k_size >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("invalid flattened W8A8 row count");
    const int rows = static_cast<int>(elements / k_size);
    if (rows != 32 && rows != 128 && rows != 512)
        throw std::runtime_error("W8A8 MPP rows must be 32, 128, or 512");

    const std::array<const MlxArray*, 1> quantize_inputs{&input};
    const std::array<MlxMetalOutputSpec, 2> quantize_outputs{{
        {.shape = {rows, k_}, .dtype = MLX_INT8},
        {.shape = {rows}, .dtype = MLX_FLOAT32},
    }};
    const std::array<int, 3> quantize_grid{rows * 256, 1, 1};
    const std::array<int, 3> quantize_group{256, 1, 1};
    const std::array<MlxMetalIntTemplate, 1> quantize_args{{{"K", k_}}};
    std::vector<MlxArray> quantized = quantizer()->apply(
        quantize_inputs, quantize_outputs, quantize_grid, quantize_group, {}, quantize_args);

    const int block_m = rows == 32 ? 32 : 128;
    const int groups_m = rows / block_m;
    const int threads = rows == 32 ? 128 : 512;
    const int tiles_n = n_ / 128;
    const std::array<const MlxArray*, 4> inputs{
        &quantized[0], &weight_, &quantized[1], &scales_};
    const std::array<MlxMetalOutputSpec, 1> outputs{{
        {.shape = {rows, n_}, .dtype = MLX_FLOAT16},
    }};
    const std::array<int, 3> grid{tiles_n * threads, groups_m, 1};
    const std::array<int, 3> group{threads, 1, 1};
    const std::array<MlxMetalIntTemplate, 8> args{{
        {"M", rows}, {"N", n_}, {"K", k_}, {"BM", block_m},
        {"WM", rows == 32 ? 1 : 4}, {"SWIZZLE", 0},
        {"TILES_M", groups_m}, {"TILES_N", tiles_n},
    }};
    MlxArray output = mpp_projection()->apply(inputs, outputs, grid, group, {}, args)[0]
                          .astype(MLX_BFLOAT16);
    std::vector<int> output_shape = input_shape;
    output_shape.back() = n_;
    return output.reshape(output_shape);
}

} // namespace qwen38::devtools
