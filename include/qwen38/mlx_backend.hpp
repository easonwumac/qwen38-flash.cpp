#pragma once

#include <mlx/c/mlx.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace qwen38 {

class MlxArray final {
public:
    MlxArray() noexcept;
    explicit MlxArray(mlx_array value) noexcept;
    ~MlxArray();

    MlxArray(const MlxArray&) = delete;
    MlxArray& operator=(const MlxArray&) = delete;
    MlxArray(MlxArray&& other) noexcept;
    MlxArray& operator=(MlxArray&& other) noexcept;

    [[nodiscard]] static MlxArray from_float32(
        std::span<const float> values,
        std::span<const int> shape);
    [[nodiscard]] static MlxArray add(const MlxArray& left, const MlxArray& right);
    [[nodiscard]] static MlxArray matmul(const MlxArray& left, const MlxArray& right);

    void eval() const;
    [[nodiscard]] std::vector<float> to_float32() const;
    [[nodiscard]] std::vector<int> shape() const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] mlx_dtype dtype() const noexcept;
    [[nodiscard]] mlx_array get() const noexcept { return value_; }

private:
    mlx_array value_{};
};

[[nodiscard]] std::string mlx_backend_description();

} // namespace qwen38
