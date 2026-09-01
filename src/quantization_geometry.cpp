#include "qwen38/quantization_geometry.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace qwen38 {

int infer_affine_quantization_bits(
    const std::span<const int> packed_weight_shape,
    const std::span<const int> scale_shape,
    const std::size_t group_size,
    const std::string_view projection_kind) {
    const std::string label(projection_kind);
    if (packed_weight_shape.empty() || packed_weight_shape.size() != scale_shape.size() ||
        packed_weight_shape.back() <= 0 || scale_shape.back() <= 0 || group_size == 0) {
        throw std::runtime_error("cannot infer " + label + " quantization geometry");
    }
    const std::size_t packed = static_cast<std::size_t>(packed_weight_shape.back());
    const std::size_t groups = static_cast<std::size_t>(scale_shape.back());
    if (groups > std::numeric_limits<std::size_t>::max() / group_size) {
        throw std::runtime_error(label + " quantization geometry overflows");
    }
    const std::size_t width = groups * group_size;
    if (width == 0 || packed > std::numeric_limits<std::size_t>::max() / 32 ||
        packed * 32 % width != 0) {
        throw std::runtime_error("non-integral " + label + " quantization bits");
    }
    const std::size_t bits = packed * 32 / width;
    if (bits != 4 && bits != 8) {
        throw std::runtime_error(label + " projections must use affine Q4 or Q8");
    }
    return static_cast<int>(bits);
}

} // namespace qwen38
