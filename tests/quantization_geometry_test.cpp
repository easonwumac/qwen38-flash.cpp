#include "qwen38/quantization_geometry.hpp"

#include "test.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

void run_quantization_geometry_tests() {
    const std::array<int, 2> q4_weight{2560, 320};
    const std::array<int, 2> q3_weight{2560, 240};
    const std::array<int, 2> q8_weight{2560, 640};
    const std::array<int, 2> scales{2560, 40};
    QWEN38_CHECK(qwen38::infer_affine_quantization_bits(
        q4_weight, scales, 64, "test") == 4);
    QWEN38_CHECK(qwen38::infer_affine_quantization_bits(
        q3_weight, scales, 64, "test") == 3);
    QWEN38_CHECK(qwen38::infer_affine_quantization_bits(
        q8_weight, scales, 64, "test") == 8);

    bool rejected_q7 = false;
    try {
        const std::array<int, 2> q7_weight{2560, 560};
        static_cast<void>(qwen38::infer_affine_quantization_bits(
            q7_weight, scales, 64, "test"));
    } catch (const std::runtime_error&) {
        rejected_q7 = true;
    }
    QWEN38_CHECK(rejected_q7);

    bool rejected_mismatch = false;
    try {
        const std::array<int, 1> bad_scales{40};
        static_cast<void>(qwen38::infer_affine_quantization_bits(
            q4_weight, bad_scales, 64, "test"));
    } catch (const std::runtime_error&) {
        rejected_mismatch = true;
    }
    QWEN38_CHECK(rejected_mismatch);

    constexpr std::size_t bits = 14;
    std::vector<std::uint32_t> expected(67);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<std::uint32_t>((index * 997 + 17) & 0x3FFF);
    }
    std::vector<std::uint32_t> packed(
        qwen38::packed_vq_word_count(expected.size(), bits), 0);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::size_t in_block = index % 32;
        const std::size_t bit_offset = in_block * bits;
        const std::size_t word = (index / 32) * bits + bit_offset / 32;
        const std::size_t shift = bit_offset % 32;
        const std::uint64_t value = static_cast<std::uint64_t>(expected[index]) << shift;
        packed[word] |= static_cast<std::uint32_t>(value);
        if (shift + bits > 32) packed[word + 1] |= static_cast<std::uint32_t>(value >> 32);
    }
    QWEN38_CHECK(packed.size() == 42);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        QWEN38_CHECK(qwen38::unpack_vq_code(packed, index, bits) == expected[index]);
    }

    bool rejected_short = false;
    try {
        static_cast<void>(qwen38::unpack_vq_code(
            std::span<const std::uint32_t>(packed.data(), 13), 31, bits));
    } catch (const std::out_of_range&) {
        rejected_short = true;
    }
    QWEN38_CHECK(rejected_short);
}
