#pragma once

#include "qwen38/mlx_backend.hpp"

#include <cstddef>

namespace qwen38::devtools {

// Developer-only W8A8 projection used to evaluate MPP prefill feasibility.
// The sidecar is derived from the supplied affine-Q4 tuple.
class W8A8Projection final {
public:
    W8A8Projection(
        const MlxArray& q4_weight,
        const MlxArray& q4_scales,
        const MlxArray& q4_biases,
        int group_size,
        int bits);

    [[nodiscard]] MlxArray apply(const MlxArray& input) const;
    [[nodiscard]] std::size_t sidecar_bytes() const noexcept { return sidecar_bytes_; }
    [[nodiscard]] int input_width() const noexcept { return k_; }
    [[nodiscard]] int output_width() const noexcept { return n_; }

private:
    MlxArray weight_;
    MlxArray scales_;
    int n_ = 0;
    int k_ = 0;
    std::size_t sidecar_bytes_ = 0;
};

} // namespace qwen38::devtools
