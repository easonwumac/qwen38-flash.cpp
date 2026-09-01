#include "qwen38/mlx_backend.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const std::array<int, 2> shape{2, 2};
    const std::array<float, 4> left_values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> right_values{5.0F, 6.0F, 7.0F, 8.0F};
    const auto left = qwen38::MlxArray::from_float32(left_values, shape);
    const auto right = qwen38::MlxArray::from_float32(right_values, shape);
    const auto sum = qwen38::MlxArray::add(left, right).to_float32();
    const auto product = qwen38::MlxArray::matmul(left, right).to_float32();
    const auto elementwise = qwen38::MlxArray::multiply(left, right).to_float32();
    if (sum != std::vector<float>({6.0F, 8.0F, 10.0F, 12.0F})) {
        std::cerr << "MLX add mismatch\n";
        return 1;
    }
    if (product != std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F})) {
        std::cerr << "MLX matmul mismatch\n";
        return 1;
    }
    if (elementwise != std::vector<float>({5.0F, 12.0F, 21.0F, 32.0F})) {
        std::cerr << "MLX multiply mismatch\n";
        return 1;
    }
    const std::array<int, 3> expanded_shape{1, 2, 2};
    const std::array<int, 3> repetitions{2, 1, 1};
    const auto expanded = left.reshape(expanded_shape).tile(repetitions);
    if (expanded.shape() != std::vector<int>({2, 2, 2}) ||
        expanded.mean_axis(0).to_float32() != std::vector<float>({1, 2, 3, 4})) {
        std::cerr << "MLX reshape/tile/mean mismatch\n";
        return 1;
    }
    const auto transposed = left.transpose().to_float32();
    if (transposed != std::vector<float>({1, 3, 2, 4})) {
        std::cerr << "MLX transpose mismatch\n";
        return 1;
    }
    if (left.sum_axis(0).to_float32() != std::vector<float>({4, 6})) {
        std::cerr << "MLX sum mismatch\n";
        return 1;
    }
    const std::array<int, 2> slice_start{1, 0};
    const std::array<int, 2> slice_stop{3, 2};
    const std::array<int, 2> slice_strides{1, 1};
    if (left.repeat_axis(2, 0).slice(slice_start, slice_stop, slice_strides).to_float32() !=
        std::vector<float>({1, 2, 3, 4})) {
        std::cerr << "MLX repeat/slice mismatch\n";
        return 1;
    }
    const auto sigmoid = left.sigmoid().to_float32();
    if (std::abs(sigmoid.front() - 0.7310586F) > 1.0e-5F) {
        std::cerr << "MLX sigmoid mismatch\n";
        return 1;
    }
    return 0;
}
