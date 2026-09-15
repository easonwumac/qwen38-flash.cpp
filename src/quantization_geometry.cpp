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
    if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 && bits != 8) {
        throw std::runtime_error(
            label + " projections must use affine Q2, Q3, Q4, Q5, Q6, or Q8");
    }
    return static_cast<int>(bits);
}

std::size_t packed_vq_word_count(
    const std::size_t code_count,
    const std::size_t bits_per_code) {
    if (bits_per_code == 0 || bits_per_code > 31) {
        throw std::runtime_error("packed VQ codes must use 1 to 31 bits");
    }
    const std::size_t blocks = code_count / 32 + (code_count % 32 != 0 ? 1 : 0);
    if (blocks > std::numeric_limits<std::size_t>::max() / bits_per_code) {
        throw std::runtime_error("packed VQ word count overflows");
    }
    return blocks * bits_per_code;
}

std::uint32_t unpack_vq_code(
    const std::span<const std::uint32_t> words,
    const std::size_t code_index,
    const std::size_t bits_per_code) {
    const std::size_t required = packed_vq_word_count(code_index + 1, bits_per_code);
    if (words.size() < required) throw std::out_of_range("packed VQ code is out of range");

    const std::size_t in_block = code_index % 32;
    const std::size_t bit_offset = in_block * bits_per_code;
    const std::size_t word = (code_index / 32) * bits_per_code + bit_offset / 32;
    const std::size_t shift = bit_offset % 32;
    std::uint64_t value = static_cast<std::uint64_t>(words[word]) >> shift;
    if (shift + bits_per_code > 32) {
        value |= static_cast<std::uint64_t>(words[word + 1]) << (32 - shift);
    }
    return static_cast<std::uint32_t>(
        value & ((std::uint64_t{1} << bits_per_code) - 1));
}

} // namespace qwen38
