#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace qwen38 {

[[nodiscard]] int infer_affine_quantization_bits(
    std::span<const int> packed_weight_shape,
    std::span<const int> scale_shape,
    std::size_t group_size,
    std::string_view projection_kind);

} // namespace qwen38
