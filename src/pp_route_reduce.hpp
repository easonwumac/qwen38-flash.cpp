#pragma once

#include "qwen38/mlx_backend.hpp"
#include <array>
#include <stdexcept>

namespace qwen38 {

// MLX Sigmoid formula (Apple Inc., MIT), with both BF16 multiplication
// boundaries retained. PP-only candidate: no sigmoid/activated global arrays.
inline MlxArray pp_swiglu(const MlxArray& gate, const MlxArray& up) {
    if (gate.dtype() != MLX_BFLOAT16 || up.dtype() != MLX_BFLOAT16 ||
        gate.shape() != up.shape() || gate.size() == 0 || gate.size() > 16777216)
        throw std::runtime_error("invalid PP SwiGLU geometry");
    static const MlxMetalKernel kernel("qwen38_pp_swiglu",
        std::array<const char*, 2>{"gate", "up"}, "output", R"metal(
        const uint i = thread_position_in_grid.x;
        if (i >= N) return;
        const T x = gate[i];
        const auto y = 1 / (1 + metal::exp(metal::abs(x)));
        const T sigmoid = T(x < 0 ? y : 1 - y);
        const T activated = T(float(x) * float(sigmoid));
        output[i] = T(float(activated) * float(up[i]));
        )metal", "#include <metal_stdlib>\nusing namespace metal;\n");
    const std::array<const MlxArray*, 2> inputs{&gate, &up};
    const std::array<MlxMetalOutputSpec, 1> outputs{{{.shape = gate.shape(), .dtype = MLX_BFLOAT16}}};
    const std::array<MlxMetalDtypeTemplate, 1> types{{{.name = "T", .value = MLX_BFLOAT16}}};
    const std::array<MlxMetalIntTemplate, 1> ints{{{.name = "N", .value = static_cast<int>(gate.size())}}};
    return std::move(kernel.apply(inputs, outputs,
        std::array<int, 3>{static_cast<int>(gate.size()), 1, 1},
        std::array<int, 3>{256, 1, 1}, types, ints).front());
}

// Developer candidate: gather + BF16 product + BF16 route reduction without
// materializing the reordered and weighted [rows, topk, hidden] buffers.
// Reduction ordering follows Apple MLX col_reduce_small (MIT, Apple Inc.),
// with eight partial accumulators for the fixed top-10 / hidden-2560 path.
inline MlxArray pp_route_reduce(const MlxArray& down, const MlxArray& weights,
                                const MlxArray& inverse, int rows, int topk, int hidden) {
    if (rows <= 0 || rows > 1024 || topk != 10 || hidden != 2560 || down.dtype() != MLX_BFLOAT16 ||
        weights.dtype() != MLX_BFLOAT16 ||
        (inverse.dtype() != MLX_INT32 && inverse.dtype() != MLX_UINT32) ||
        down.shape() != std::vector<int>{rows * topk, hidden} ||
        weights.size() != static_cast<std::size_t>(rows * topk) ||
        inverse.size() != weights.size())
        throw std::runtime_error("invalid PP route reduction geometry");
    static const MlxMetalKernel kernel(
        "qwen38_pp_route_reduce", std::array<const char*, 3>{"down", "weights", "inverse"},
        "output", R"metal(
        const uint i = thread_position_in_grid.x;
        if (i >= ROWS * HIDDEN) return;
        const uint row = i / HIDDEN;
        const uint col = i % HIDDEN;
        T partial[8];
        for (uint j = 0; j < 8; ++j) partial[j] = T(0);
        for (uint k = 0; k < TOPK; ++k) {
            const uint slot = row * TOPK + k;
            const T product = T(float(down[size_t(inverse[slot]) * HIDDEN + col]) *
                                float(weights[slot]));
            partial[k % 8] = T(float(product) + float(partial[k % 8]));
        }
        T sum = partial[0];
        for (uint j = 1; j < 8; ++j) sum = T(float(partial[j]) + float(sum));
        output[i] = sum;
        )metal", "#include <metal_stdlib>\nusing namespace metal;\n");
    const std::array<const MlxArray*, 3> inputs{&down, &weights, &inverse};
    const std::array<MlxMetalOutputSpec, 1> outputs{{
        {.shape = {1, rows, hidden}, .dtype = MLX_BFLOAT16}}};
    const std::array<MlxMetalDtypeTemplate, 1> types{{{.name = "T", .value = MLX_BFLOAT16}}};
    const std::array<MlxMetalIntTemplate, 3> ints{{
        {.name = "ROWS", .value = rows}, {.name = "TOPK", .value = topk},
        {.name = "HIDDEN", .value = hidden}}};
    return std::move(kernel.apply(inputs, outputs, std::array<int, 3>{rows * hidden, 1, 1},
                                  std::array<int, 3>{256, 1, 1}, types, ints).front());
}
} // namespace qwen38
