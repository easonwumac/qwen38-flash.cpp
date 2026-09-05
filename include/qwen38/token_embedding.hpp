#pragma once

#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace qwen38 {

// Gathers and dequantizes one token batch as [1,S,H]. Token validation happens
// before the uint32-to-int32 conversion required by the MLX take-axis wrapper.
[[nodiscard]] MlxArray embed_token_batch(
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    std::span<const std::uint32_t> tokens,
    std::size_t vocabulary_size,
    std::size_t hidden_size,
    int group_size,
    int bits);

} // namespace qwen38
