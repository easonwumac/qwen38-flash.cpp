#include "qwen38/mlx_backend.hpp"

#include <array>
#include <cstdlib>
#include <iostream>

int main() {
    const std::array<float, 4> left_values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> right_values{5.0F, 6.0F, 7.0F, 8.0F};
    const std::array<int, 2> shape{2, 2};
    const auto left = qwen38::MlxArray::from_float32(left_values, shape);
    const auto right = qwen38::MlxArray::from_float32(right_values, shape);
    const auto result = qwen38::MlxArray::matmul(left, right);
    const auto values = result.to_float32();
    std::cout << qwen38::mlx_backend_description() << '\n';
    std::cout << values[0] << ' ' << values[1] << ' ' << values[2] << ' ' << values[3] << '\n';
    return values == std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F}) ? EXIT_SUCCESS
                                                                            : EXIT_FAILURE;
}
