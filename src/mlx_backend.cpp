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

class Stream final {
public:
    Stream() : value_(mlx_default_gpu_stream_new()) {
        if (value_.ctx == nullptr) {
            throw std::runtime_error("MLX did not provide a default GPU stream");
        }
    }
    ~Stream() { static_cast<void>(mlx_stream_free(value_)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    [[nodiscard]] mlx_stream get() const noexcept { return value_; }

private:
    mlx_stream value_{};
};

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

MlxArray MlxArray::from_int32(
    const std::span<const std::int32_t> values,
    const std::span<const int> shape) {
    std::size_t expected = 1;
    for (const int dimension : shape) {
        if (dimension < 0) throw std::runtime_error("negative MLX array dimension");
        expected *= static_cast<std::size_t>(dimension);
    }
    if (expected != values.size()) {
        throw std::runtime_error("MLX array shape does not match input data");
    }
    return MlxArray(mlx_array_new_data(
        values.data(), shape.data(), static_cast<int>(shape.size()), MLX_INT32));
}

MlxArray MlxArray::add(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_add(&result.value_, left.value_, right.value_, stream.get()), "add");
    return result;
}

MlxArray MlxArray::multiply(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_multiply(&result.value_, left.value_, right.value_, stream.get()), "multiply");
    return result;
}

MlxArray MlxArray::matmul(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_matmul(&result.value_, left.value_, right.value_, stream.get()), "matmul");
    return result;
}

MlxArray MlxArray::reshape(const std::span<const int> shape) const {
    MlxArray result;
    const Stream stream;
    check(mlx_reshape(&result.value_, value_, shape.data(), shape.size(), stream.get()), "reshape");
    return result;
}

MlxArray MlxArray::tile(const std::span<const int> repetitions) const {
    MlxArray result;
    const Stream stream;
    check(mlx_tile(
              &result.value_, value_, repetitions.data(), repetitions.size(), stream.get()),
        "tile");
    return result;
}

MlxArray MlxArray::sigmoid() const {
    MlxArray result;
    const Stream stream;
    check(mlx_sigmoid(&result.value_, value_, stream.get()), "sigmoid");
    return result;
}

MlxArray MlxArray::silu() const {
    MlxArray gate = sigmoid();
    return multiply(*this, gate);
}

MlxArray MlxArray::mean_axis(const int axis, const bool keep_dimensions) const {
    MlxArray result;
    const Stream stream;
    check(mlx_mean_axis(
              &result.value_, value_, axis, keep_dimensions, stream.get()),
        "mean_axis");
    return result;
}

MlxArray MlxArray::rms_norm(const MlxArray& weight, const float epsilon) const {
    MlxArray result;
    const Stream stream;
    check(mlx_fast_rms_norm(
              &result.value_, value_, weight.value_, epsilon, stream.get()),
        "fast_rms_norm");
    return result;
}

MlxArray MlxArray::astype(const mlx_dtype dtype) const {
    MlxArray result;
    const Stream stream;
    check(mlx_astype(&result.value_, value_, dtype, stream.get()), "astype");
    return result;
}

MlxArray MlxArray::quantized_matmul(
    const MlxArray& input,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    const int group_size,
    const int bits,
    const bool transpose) {
    if (group_size <= 0 || bits <= 0) {
        throw std::runtime_error("invalid quantization parameters");
    }
    MlxArray result;
    const Stream stream;
    check(mlx_quantized_matmul(
              &result.value_,
              input.value_,
              weight.value_,
              scales.value_,
              biases.value_,
              transpose,
              mlx_optional_int{.value = group_size, .has_value = true},
              mlx_optional_int{.value = bits, .has_value = true},
              "affine",
              stream.get()),
        "quantized_matmul");
    return result;
}

MlxArray MlxArray::take_axis(
    const MlxArray& input,
    const MlxArray& indices,
    const int axis) {
    MlxArray result;
    const Stream stream;
    check(mlx_take_axis(&result.value_, input.value_, indices.value_, axis, stream.get()), "take_axis");
    return result;
}

MlxArray MlxArray::dequantize(
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    const int group_size,
    const int bits,
    const mlx_dtype output_dtype) {
    if (group_size <= 0 || bits <= 0) {
        throw std::runtime_error("invalid quantization parameters");
    }
    MlxArray result;
    const Stream stream;
    check(mlx_dequantize(
              &result.value_,
              weight.value_,
              scales.value_,
              biases.value_,
              mlx_optional_int{.value = group_size, .has_value = true},
              mlx_optional_int{.value = bits, .has_value = true},
              "affine",
              mlx_array{},
              mlx_optional_dtype{.value = output_dtype, .has_value = true},
              stream.get()),
        "dequantize");
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

MlxSafetensors::MlxSafetensors(const std::filesystem::path& path)
    : tensors_(mlx_map_string_to_array_new()), metadata_(mlx_map_string_to_string_new()) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    if (stream.ctx == nullptr) {
        throw std::runtime_error("MLX did not provide a default CPU stream");
    }
    const int status = mlx_load_safetensors(&tensors_, &metadata_, path.c_str(), stream);
    static_cast<void>(mlx_stream_free(stream));
    if (status != 0) {
        static_cast<void>(mlx_map_string_to_array_free(tensors_));
        static_cast<void>(mlx_map_string_to_string_free(metadata_));
        tensors_ = {};
        metadata_ = {};
        throw std::runtime_error("MLX could not load safetensors shard: " + path.string());
    }
}

MlxSafetensors::~MlxSafetensors() {
    if (tensors_.ctx != nullptr) static_cast<void>(mlx_map_string_to_array_free(tensors_));
    if (metadata_.ctx != nullptr) static_cast<void>(mlx_map_string_to_string_free(metadata_));
}

MlxArray MlxSafetensors::tensor(const std::string_view name) const {
    MlxArray result;
    const std::string key(name);
    const int status = mlx_map_string_to_array_get(&result.value_, tensors_, key.c_str());
    if (status == 2) throw std::out_of_range("tensor not found in MLX shard: " + key);
    check(status, "map_string_to_array_get");
    return result;
}

MlxArray MlxTensorStore::tensor(const std::string_view name) {
    std::scoped_lock lock(mutex_);
    const auto mapping = manifest_.weight_map().find(std::string(name));
    if (mapping == manifest_.weight_map().end()) {
        throw std::out_of_range("tensor is not present in model index: " + std::string(name));
    }
    auto shard = shards_.find(mapping->second);
    if (shard == shards_.end()) {
        auto file = std::make_unique<MlxSafetensors>(manifest_.directory() / mapping->second);
        shard = shards_.emplace(mapping->second, std::move(file)).first;
    }
    return shard->second->tensor(name);
}

std::size_t MlxTensorStore::open_shard_count() const {
    std::scoped_lock lock(mutex_);
    return shards_.size();
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
