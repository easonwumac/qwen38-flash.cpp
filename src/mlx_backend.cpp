#include "qwen38/mlx_backend.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <utility>

namespace qwen38 {
namespace {

void check(const int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string("MLX operation failed: ") + operation);
    }
}

class SharedStream final {
public:
    SharedStream() : value_(mlx_default_gpu_stream_new()) {
        if (value_.ctx == nullptr) {
            throw std::runtime_error("MLX did not provide a default GPU stream");
        }
    }
    ~SharedStream() { static_cast<void>(mlx_stream_free(value_)); }
    SharedStream(const SharedStream&) = delete;
    SharedStream& operator=(const SharedStream&) = delete;
    [[nodiscard]] mlx_stream get() const noexcept { return value_; }

private:
    mlx_stream value_{};
};

[[nodiscard]] mlx_stream default_gpu_stream() {
    static const SharedStream stream;
    return stream.get();
}

class Stream final {
public:
    [[nodiscard]] mlx_stream get() const { return default_gpu_stream(); }
};

} // namespace

MlxArray::MlxArray() noexcept : value_(mlx_array_new()) {}

MlxArray::MlxArray(const mlx_array value) noexcept : value_(value) {}

MlxArray::~MlxArray() {
    if (locked_address_ != nullptr) {
        static_cast<void>(munlock(locked_address_, locked_bytes_));
    }
    if (value_.ctx != nullptr) {
        static_cast<void>(mlx_array_free(value_));
    }
}

MlxArray::MlxArray(MlxArray&& other) noexcept
    : value_(std::exchange(other.value_, mlx_array{})),
      locked_address_(std::exchange(other.locked_address_, nullptr)),
      locked_bytes_(std::exchange(other.locked_bytes_, 0)) {}

MlxArray& MlxArray::operator=(MlxArray&& other) noexcept {
    if (this != &other) {
        if (locked_address_ != nullptr) {
            static_cast<void>(munlock(locked_address_, locked_bytes_));
        }
        if (value_.ctx != nullptr) {
            static_cast<void>(mlx_array_free(value_));
        }
        value_ = std::exchange(other.value_, mlx_array{});
        locked_address_ = std::exchange(other.locked_address_, nullptr);
        locked_bytes_ = std::exchange(other.locked_bytes_, 0);
    }
    return *this;
}

MlxMetalKernel::MlxMetalKernel(
    const std::string_view name,
    const std::span<const char* const> input_names,
    const std::string_view output_name,
    const std::string_view source,
    const std::string_view header) {
    std::vector<const char*> mutable_input_names(input_names.begin(), input_names.end());
    const char* output_names[]{output_name.data()};
    mlx_vector_string inputs = mlx_vector_string_new_data(
        mutable_input_names.data(), mutable_input_names.size());
    mlx_vector_string outputs = mlx_vector_string_new_data(output_names, 1);
    kernel_ = mlx_fast_metal_kernel_new(
        std::string(name).c_str(),
        inputs,
        outputs,
        std::string(source).c_str(),
        std::string(header).c_str(),
        true,
        false);
    static_cast<void>(mlx_vector_string_free(inputs));
    static_cast<void>(mlx_vector_string_free(outputs));
    if (kernel_.ctx == nullptr) throw std::runtime_error("MLX could not create Metal kernel");
}

MlxMetalKernel::~MlxMetalKernel() {
    if (kernel_.ctx != nullptr) mlx_fast_metal_kernel_free(kernel_);
}

MlxArray MlxMetalKernel::apply(
    const std::span<const MlxArray* const> inputs,
    const std::span<const int> output_shape,
    const mlx_dtype output_dtype,
    const std::span<const int, 3> grid,
    const std::span<const int, 3> threadgroup) const {
    std::vector<mlx_array> raw_inputs;
    raw_inputs.reserve(inputs.size());
    for (const MlxArray* input : inputs) {
        if (input == nullptr) throw std::runtime_error("null Metal kernel input");
        raw_inputs.push_back(input->value_);
    }
    mlx_vector_array input_vector = mlx_vector_array_new_data(raw_inputs.data(), raw_inputs.size());
    mlx_vector_array output_vector = mlx_vector_array_new();
    mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
    if (input_vector.ctx == nullptr || output_vector.ctx == nullptr || config.ctx == nullptr) {
        throw std::runtime_error("MLX could not create Metal kernel arguments");
    }
    check(mlx_fast_metal_kernel_config_add_output_arg(
              config, output_shape.data(), output_shape.size(), output_dtype),
        "metal_kernel output");
    check(mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]),
        "metal_kernel grid");
    check(mlx_fast_metal_kernel_config_set_thread_group(
              config, threadgroup[0], threadgroup[1], threadgroup[2]),
        "metal_kernel threadgroup");
    check(mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", output_dtype),
        "metal_kernel dtype");
    const Stream stream;
    const int status = mlx_fast_metal_kernel_apply(
        &output_vector, kernel_, input_vector, config, stream.get());
    MlxArray result;
    if (status == 0) check(mlx_vector_array_get(&result.value_, output_vector, 0), "metal_kernel get");
    static_cast<void>(mlx_vector_array_free(input_vector));
    static_cast<void>(mlx_vector_array_free(output_vector));
    mlx_fast_metal_kernel_config_free(config);
    check(status, "metal_kernel apply");
    return result;
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

MlxArray MlxArray::subtract(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_subtract(&result.value_, left.value_, right.value_, stream.get()), "subtract");
    return result;
}

MlxArray MlxArray::multiply(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_multiply(&result.value_, left.value_, right.value_, stream.get()), "multiply");
    return result;
}

MlxArray MlxArray::divide(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_divide(&result.value_, left.value_, right.value_, stream.get()), "divide");
    return result;
}

MlxArray MlxArray::concatenate(
    const MlxArray& left,
    const MlxArray& right,
    const int axis) {
    const mlx_array values[]{left.value_, right.value_};
    const mlx_vector_array vector = mlx_vector_array_new_data(values, 2);
    if (vector.ctx == nullptr) throw std::runtime_error("MLX could not create concatenate input");
    MlxArray result;
    const Stream stream;
    const int status = mlx_concatenate_axis(&result.value_, vector, axis, stream.get());
    static_cast<void>(mlx_vector_array_free(vector));
    check(status, "concatenate_axis");
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

MlxArray MlxArray::transpose() const {
    MlxArray result;
    const Stream stream;
    check(mlx_transpose(&result.value_, value_, stream.get()), "transpose");
    return result;
}

MlxArray MlxArray::swapaxes(const int axis1, const int axis2) const {
    MlxArray result;
    const Stream stream;
    check(mlx_swapaxes(&result.value_, value_, axis1, axis2, stream.get()), "swapaxes");
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

MlxArray MlxArray::repeat_axis(const int repeats, const int axis) const {
    if (repeats <= 0) throw std::runtime_error("repeat count must be positive");
    MlxArray result;
    const Stream stream;
    check(mlx_repeat_axis(&result.value_, value_, repeats, axis, stream.get()), "repeat_axis");
    return result;
}

MlxArray MlxArray::slice(
    const std::span<const int> start,
    const std::span<const int> stop,
    const std::span<const int> strides) const {
    if (start.size() != stop.size() || start.size() != strides.size()) {
        throw std::runtime_error("slice vectors must have equal lengths");
    }
    MlxArray result;
    const Stream stream;
    check(mlx_slice(
              &result.value_,
              value_,
              start.data(),
              start.size(),
              stop.data(),
              stop.size(),
              strides.data(),
              strides.size(),
              stream.get()),
        "slice");
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

MlxArray MlxArray::exp() const {
    MlxArray result;
    const Stream stream;
    check(mlx_exp(&result.value_, value_, stream.get()), "exp");
    return result;
}

MlxArray MlxArray::log1p() const {
    MlxArray result;
    const Stream stream;
    check(mlx_log1p(&result.value_, value_, stream.get()), "log1p");
    return result;
}

MlxArray MlxArray::negative() const {
    MlxArray result;
    const Stream stream;
    check(mlx_negative(&result.value_, value_, stream.get()), "negative");
    return result;
}

MlxArray MlxArray::absolute() const {
    MlxArray result;
    const Stream stream;
    check(mlx_abs(&result.value_, value_, stream.get()), "abs");
    return result;
}

MlxArray MlxArray::square_root() const {
    MlxArray result;
    const Stream stream;
    check(mlx_sqrt(&result.value_, value_, stream.get()), "sqrt");
    return result;
}

MlxArray MlxArray::sign() const {
    MlxArray result;
    const Stream stream;
    check(mlx_sign(&result.value_, value_, stream.get()), "sign");
    return result;
}

MlxArray MlxArray::maximum(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_maximum(&result.value_, left.value_, right.value_, stream.get()), "maximum");
    return result;
}

MlxArray MlxArray::softmax_axis(const int axis) const {
    MlxArray result;
    const Stream stream;
    check(mlx_softmax_axis(&result.value_, value_, axis, true, stream.get()), "softmax_axis");
    return result;
}

MlxArray MlxArray::mean_axis(const int axis, const bool keep_dimensions) const {
    MlxArray result;
    const Stream stream;
    check(mlx_mean_axis(
              &result.value_, value_, axis, keep_dimensions, stream.get()),
        "mean_axis");
    return result;
}

MlxArray MlxArray::sum_axis(const int axis, const bool keep_dimensions) const {
    MlxArray result;
    const Stream stream;
    check(mlx_sum_axis(
              &result.value_, value_, axis, keep_dimensions, stream.get()),
        "sum_axis");
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

MlxArray MlxArray::zeros(const std::span<const int> shape, const mlx_dtype dtype) {
    MlxArray result;
    const Stream stream;
    check(mlx_zeros(&result.value_, shape.data(), shape.size(), dtype, stream.get()), "zeros");
    return result;
}

MlxArray MlxArray::conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {
    MlxArray result;
    const Stream stream;
    check(mlx_conv1d(
              &result.value_,
              input.value_,
              weight.value_,
              stride,
              padding,
              dilation,
              groups,
              stream.get()),
        "conv1d");
    return result;
}

MlxArray MlxArray::astype(const mlx_dtype dtype) const {
    MlxArray result;
    const Stream stream;
    check(mlx_astype(&result.value_, value_, dtype, stream.get()), "astype");
    return result;
}

void MlxArray::lock_pages() {
    if (locked_address_ != nullptr || size() == 0) return;
    eval();
    const auto* data = mlx_array_data_uint8(value_);
    if (data == nullptr) throw std::runtime_error("MLX returned null array data for mlock");
    const std::size_t bytes = mlx_array_nbytes(value_);
    if (mlock(data, bytes) != 0) {
        throw std::runtime_error(std::string("mlock failed: ") + std::strerror(errno));
    }
    locked_address_ = const_cast<std::uint8_t*>(data);
    locked_bytes_ = bytes;
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

MlxArray MlxArray::take(const MlxArray& input, const MlxArray& indices) {
    MlxArray result;
    const Stream stream;
    check(mlx_take(&result.value_, input.value_, indices.value_, stream.get()), "take");
    return result;
}

MlxArray MlxArray::take_along_axis(
    const MlxArray& input,
    const MlxArray& indices,
    const int axis) {
    MlxArray result;
    const Stream stream;
    check(
        mlx_take_along_axis(&result.value_, input.value_, indices.value_, axis, stream.get()),
        "take_along_axis");
    return result;
}

MlxArray MlxArray::argpartition_axis(const int kth, const int axis) const {
    MlxArray result;
    const Stream stream;
    check(mlx_argpartition_axis(&result.value_, value_, kth, axis, stream.get()),
        "argpartition_axis");
    return result;
}

MlxArray MlxArray::argmax_all() const {
    MlxArray result;
    const Stream stream;
    check(mlx_argmax(&result.value_, value_, false, stream.get()), "argmax");
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
    MlxArray contiguous;
    const Stream stream;
    check(mlx_contiguous(&contiguous.value_, value_, false, stream.get()), "contiguous");
    contiguous.eval();
    const float* data = mlx_array_data_float32(contiguous.value_);
    if (data == nullptr && contiguous.size() != 0) {
        throw std::runtime_error("MLX returned null array data");
    }
    return {data, data + contiguous.size()};
}

std::uint32_t MlxArray::item_uint32() const {
    eval();
    std::uint32_t result = 0;
    check(mlx_array_item_uint32(&result, value_), "item_uint32");
    return result;
}

float MlxArray::item_float32() const {
    eval();
    float result = 0.0F;
    check(mlx_array_item_float32(&result, value_), "item_float32");
    return result;
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
