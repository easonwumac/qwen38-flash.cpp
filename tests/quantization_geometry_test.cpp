#include "qwen38/quantization_geometry.hpp"

#include "test.hpp"

#include <array>
#include <stdexcept>

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
}
