#include "qwen38/mlx_backend.hpp"

#include <array>
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
    if (sum != std::vector<float>({6.0F, 8.0F, 10.0F, 12.0F})) {
        std::cerr << "MLX add mismatch\n";
        return 1;
    }
    if (product != std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F})) {
        std::cerr << "MLX matmul mismatch\n";
        return 1;
    }
    return 0;
}
