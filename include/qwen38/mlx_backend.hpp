#pragma once

#include <mlx/c/mlx.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "qwen38/model_manifest.hpp"

namespace qwen38 {

class MlxMetalKernel;

class MlxArray final {
public:
    MlxArray() noexcept;
    explicit MlxArray(mlx_array value) noexcept;
    ~MlxArray();

    MlxArray(const MlxArray&) = delete;
    MlxArray& operator=(const MlxArray&) = delete;
    MlxArray(MlxArray&& other) noexcept;
    MlxArray& operator=(MlxArray&& other) noexcept;

    [[nodiscard]] MlxArray share() const;

    [[nodiscard]] static MlxArray from_float32(
        std::span<const float> values,
        std::span<const int> shape);
    [[nodiscard]] static MlxArray from_int32(
        std::span<const std::int32_t> values,
        std::span<const int> shape);
    [[nodiscard]] static MlxArray add(const MlxArray& left, const MlxArray& right);
    [[nodiscard]] static MlxArray subtract(const MlxArray& left, const MlxArray& right);
    [[nodiscard]] static MlxArray multiply(const MlxArray& left, const MlxArray& right);
    [[nodiscard]] static MlxArray divide(const MlxArray& left, const MlxArray& right);
    [[nodiscard]] static MlxArray concatenate(
        const MlxArray& left,
        const MlxArray& right,
        int axis);
    [[nodiscard]] static MlxArray matmul(const MlxArray& left, const MlxArray& right);
    [[nodiscard]] MlxArray reshape(std::span<const int> shape) const;
    [[nodiscard]] MlxArray transpose() const;
    [[nodiscard]] MlxArray swapaxes(int axis1, int axis2) const;
    [[nodiscard]] MlxArray tile(std::span<const int> repetitions) const;
    [[nodiscard]] MlxArray repeat_axis(int repeats, int axis) const;
    [[nodiscard]] MlxArray slice(
        std::span<const int> start,
        std::span<const int> stop,
        std::span<const int> strides) const;
    [[nodiscard]] MlxArray sigmoid() const;
    [[nodiscard]] MlxArray silu() const;
    [[nodiscard]] MlxArray exp() const;
    [[nodiscard]] MlxArray log1p() const;
    [[nodiscard]] MlxArray negative() const;
    [[nodiscard]] MlxArray absolute() const;
    [[nodiscard]] MlxArray square_root() const;
    [[nodiscard]] MlxArray sign() const;
    [[nodiscard]] static MlxArray maximum(const MlxArray& left, const MlxArray& right);
    [[nodiscard]] MlxArray softmax_axis(int axis) const;
    [[nodiscard]] MlxArray mean_axis(int axis, bool keep_dimensions = false) const;
    [[nodiscard]] MlxArray sum_axis(int axis, bool keep_dimensions = false) const;
    [[nodiscard]] MlxArray rms_norm(const MlxArray& weight, float epsilon) const;
    [[nodiscard]] static MlxArray zeros(std::span<const int> shape, mlx_dtype dtype);
    [[nodiscard]] static MlxArray conv1d(
        const MlxArray& input,
        const MlxArray& weight,
        int stride,
        int padding,
        int dilation,
        int groups);
    [[nodiscard]] MlxArray astype(mlx_dtype dtype) const;
    void lock_pages();
    [[nodiscard]] static MlxArray quantized_matmul(
        const MlxArray& input,
        const MlxArray& weight,
        const MlxArray& scales,
        const MlxArray& biases,
        int group_size,
        int bits,
        bool transpose = true);
    [[nodiscard]] static MlxArray take_axis(
        const MlxArray& input,
        const MlxArray& indices,
        int axis);
    [[nodiscard]] static MlxArray take(
        const MlxArray& input,
        const MlxArray& indices);
    [[nodiscard]] static MlxArray take_along_axis(
        const MlxArray& input,
        const MlxArray& indices,
        int axis);
    [[nodiscard]] MlxArray argpartition_axis(int kth, int axis) const;
    [[nodiscard]] MlxArray argmax_all() const;
    [[nodiscard]] static MlxArray dequantize(
        const MlxArray& weight,
        const MlxArray& scales,
        const MlxArray& biases,
        int group_size,
        int bits,
        mlx_dtype output_dtype = MLX_BFLOAT16);

    void eval() const;
    static void eval_all(std::span<const MlxArray* const> arrays);
    static void clear_cache();
    [[nodiscard]] static std::size_t set_cache_limit(std::size_t bytes);
    [[nodiscard]] std::vector<float> to_float32() const;
    [[nodiscard]] std::uint32_t item_uint32() const;
    [[nodiscard]] float item_float32() const;
    [[nodiscard]] std::vector<int> shape() const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] mlx_dtype dtype() const noexcept;
    [[nodiscard]] mlx_array get() const noexcept { return value_; }

private:
    friend class MlxSafetensors;
    friend class MlxMetalKernel;
    mlx_array value_{};
    void* locked_address_{nullptr};
    std::size_t locked_bytes_{0};
};

class MlxMetalKernel final {
public:
    MlxMetalKernel(
        std::string_view name,
        std::span<const char* const> input_names,
        std::string_view output_name,
        std::string_view source,
        std::string_view header = {});
    ~MlxMetalKernel();

    MlxMetalKernel(const MlxMetalKernel&) = delete;
    MlxMetalKernel& operator=(const MlxMetalKernel&) = delete;
    MlxMetalKernel(MlxMetalKernel&&) = delete;
    MlxMetalKernel& operator=(MlxMetalKernel&&) = delete;

    [[nodiscard]] MlxArray apply(
        std::span<const MlxArray* const> inputs,
        std::span<const int> output_shape,
        mlx_dtype output_dtype,
        std::span<const int, 3> grid,
        std::span<const int, 3> threadgroup) const;

private:
    mlx_fast_metal_kernel kernel_{};
};

class MlxSafetensors final {
public:
    explicit MlxSafetensors(const std::filesystem::path& path);
    ~MlxSafetensors();

    MlxSafetensors(const MlxSafetensors&) = delete;
    MlxSafetensors& operator=(const MlxSafetensors&) = delete;
    MlxSafetensors(MlxSafetensors&&) = delete;
    MlxSafetensors& operator=(MlxSafetensors&&) = delete;

    [[nodiscard]] MlxArray tensor(std::string_view name) const;

private:
    mlx_map_string_to_array tensors_{};
    mlx_map_string_to_string metadata_{};
};

class MlxTensorStore final {
public:
    explicit MlxTensorStore(ModelManifest manifest) : manifest_(std::move(manifest)) {}

    [[nodiscard]] MlxArray tensor(std::string_view name);
    [[nodiscard]] std::size_t open_shard_count() const;
    [[nodiscard]] const ModelManifest& manifest() const noexcept { return manifest_; }

private:
    ModelManifest manifest_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<MlxSafetensors>> shards_;
};

[[nodiscard]] std::string mlx_backend_description();

} // namespace qwen38
