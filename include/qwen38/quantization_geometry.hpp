#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace qwen38 {

[[nodiscard]] int infer_affine_quantization_bits(
    std::span<const int> packed_weight_shape,
    std::span<const int> scale_shape,
    std::size_t group_size,
    std::string_view projection_kind);

[[nodiscard]] std::size_t packed_vq_word_count(
    std::size_t code_count,
    std::size_t bits_per_code);

[[nodiscard]] std::uint32_t unpack_vq_code(
    std::span<const std::uint32_t> words,
    std::size_t code_index,
    std::size_t bits_per_code);

} // namespace qwen38
