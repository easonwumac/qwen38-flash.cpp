#pragma once
#include "qwen38/mlx_backend.hpp"
#include <array>
#include <algorithm>
#include <stdexcept>

namespace qwen38 {
inline MlxArray qsa_tiled_scores(const MlxArray& queries, const MlxArray& pooled_transposed,
                                 int tile = 256, bool asynchronous = true) {
    const auto shape = queries.shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 4 || shape[3] != 128 ||
        shape[2] < 1 || shape[2] > 1024 || (tile != 128 && tile != 256))
        throw std::runtime_error("invalid tiled QSA query geometry");
    auto pooled = pooled_transposed.astype(MLX_FLOAT32);
    const auto zero = MlxArray::from_float32(std::array<float, 1>{0}, std::array<int, 1>{1});
    std::vector<MlxArray> parts;
    for (int offset = 0; offset < shape[2]; offset += tile) {
        const int end = std::min(offset + tile, shape[2]);
        auto q = queries.slice(std::array<int, 4>{0, 0, offset, 0},
            std::array<int, 4>{1, 4, end, 128}, std::array<int, 4>{1, 1, 1, 1});
        auto product = MlxArray::matmul(q.astype(MLX_FLOAT32), pooled);
        auto reduced = MlxArray::maximum(product, zero).sum_axis(1);
        if (asynchronous) {
            const mlx_array value = reduced.get();
            mlx_vector_array outputs = mlx_vector_array_new_data(&value, 1);
            const int status = mlx_async_eval(outputs);
            static_cast<void>(mlx_vector_array_free(outputs));
            if (status != 0) throw std::runtime_error("async QSA tile evaluation failed");
        } else {
            reduced.eval();
        }
        parts.push_back(std::move(reduced));
    }
    return MlxArray::concatenate_many(parts, 1);
}
// Fuse the temporary FP32 ReLU bank into the four-head index-score sum.
// Preserve the head order of the pinned MLX small-column reduction (Apple, MIT).
inline MlxArray qsa_score_reduce(const MlxArray& scores) {
    const auto shape = scores.shape();
    if (scores.dtype() != MLX_FLOAT32 || shape.size() != 4 || shape[0] != 1 ||
        shape[1] != 4 || shape[2] < 1 || shape[2] > 1024 ||
        shape[3] < 1 || shape[3] > 65536)
        throw std::runtime_error("invalid QSA score reduction geometry");
    static const MlxMetalKernel kernel("qwen38_qsa_score_reduce",
        std::array<const char*, 1>{"scores"}, "output", R"metal(
        #pragma clang fp reassociate(off)
        const uint i = thread_position_in_grid.x;
        if (i >= N) return;
        float sum = metal::max(scores[i], 0.0f);
        for (uint h = 1; h < 4; ++h) {
            const float value = metal::max(scores[h * N + i], 0.0f);
            sum = value + sum;
        }
        output[i] = sum;
        )metal", "#include <metal_stdlib>\nusing namespace metal;\n");
    const std::array<const MlxArray*, 1> inputs{&scores};
    const std::array<MlxMetalOutputSpec, 1> outputs{{
        {.shape = {1, shape[2], shape[3]}, .dtype = MLX_FLOAT32}}};
    const std::array<MlxMetalIntTemplate, 1> ints{{{.name = "N", .value = shape[2] * shape[3]}}};
    return std::move(kernel.apply(inputs, outputs,
        std::array<int, 3>{shape[2] * shape[3], 1, 1},
        std::array<int, 3>{256, 1, 1}, {}, ints).front());
}
} // namespace qwen38
