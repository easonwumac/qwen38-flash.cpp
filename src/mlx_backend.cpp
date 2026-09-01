#include "qwen38/mlx_backend.hpp"

#include <stdexcept>
#include <utility>

namespace qwen38 {
namespace {

void check(const int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string("MLX operation failed: ") + operation);
    }
}

mlx_stream gpu_stream() {
    mlx_stream stream = mlx_default_gpu_stream_new();
    if (stream.ctx == nullptr) {
        throw std::runtime_error("MLX did not provide a default GPU stream");
    }
    return stream;
}

} // namespace

MlxArray::MlxArray() noexcept : value_(mlx_array_new()) {}

MlxArray::MlxArray(const mlx_array value) noexcept : value_(value) {}

MlxArray::~MlxArray() {
    if (value_.ctx != nullptr) {
        static_cast<void>(mlx_array_free(value_));
    }
}

MlxArray::MlxArray(MlxArray&& other) noexcept : value_(std::exchange(other.value_, mlx_array{})) {}

MlxArray& MlxArray::operator=(MlxArray&& other) noexcept {
    if (this != &other) {
        if (value_.ctx != nullptr) {
            static_cast<void>(mlx_array_free(value_));
        }
        value_ = std::exchange(other.value_, mlx_array{});
    }
    return *this;
}

MlxArray MlxArray::from_float32(
    const std::span<const float> values,
    const std::span<const int> shape) {
    std::size_t expected = 1;
    for (const int dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("negative MLX array dimension");
        }
        expected *= static_cast<std::size_t>(dimension);
    }
    if (expected != values.size()) {
        throw std::runtime_error("MLX array shape does not match input data");
    }
    return MlxArray(mlx_array_new_data(
        values.data(), shape.data(), static_cast<int>(shape.size()), MLX_FLOAT32));
}

MlxArray MlxArray::add(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    mlx_stream stream = gpu_stream();
    check(mlx_add(&result.value_, left.value_, right.value_, stream), "add");
    static_cast<void>(mlx_stream_free(stream));
    return result;
}

MlxArray MlxArray::matmul(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    mlx_stream stream = gpu_stream();
    check(mlx_matmul(&result.value_, left.value_, right.value_, stream), "matmul");
    static_cast<void>(mlx_stream_free(stream));
    return result;
}

void MlxArray::eval() const {
    check(mlx_array_eval(value_), "array_eval");
}

std::vector<float> MlxArray::to_float32() const {
    if (dtype() != MLX_FLOAT32) {
        throw std::runtime_error("MLX array is not float32");
    }
    eval();
    const float* data = mlx_array_data_float32(value_);
    if (data == nullptr && size() != 0) {
        throw std::runtime_error("MLX returned null array data");
    }
    return {data, data + size()};
}

std::vector<int> MlxArray::shape() const {
    const std::size_t dimensions = mlx_array_ndim(value_);
    const int* data = mlx_array_shape(value_);
    return {data, data + dimensions};
}

std::size_t MlxArray::size() const noexcept {
    return mlx_array_size(value_);
}

mlx_dtype MlxArray::dtype() const noexcept {
    return mlx_array_dtype(value_);
}

std::string mlx_backend_description() {
    mlx_device device = mlx_device_new_type(MLX_GPU, 0);
    mlx_string description = mlx_string_new();
    check(mlx_device_tostring(&description, device), "device_tostring");
    const char* text = mlx_string_data(description);
    std::string result = text == nullptr ? "MLX GPU" : text;
    static_cast<void>(mlx_string_free(description));
    static_cast<void>(mlx_device_free(device));
    return result;
}

} // namespace qwen38
