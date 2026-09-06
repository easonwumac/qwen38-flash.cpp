#include "qwen38/mlx_backend.hpp"

#include <cerrno>
#include <chrono>
#include <fstream>
#include <future>
#include <array>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <utility>

extern "C" int mlx_fast_scaled_dot_product_attention(
    mlx_array* result,
    mlx_array queries,
    mlx_array keys,
    mlx_array values,
    float scale,
    const char* mask_mode,
    mlx_array mask,
    mlx_array sinks,
    bool force_fused,
    mlx_stream stream);

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

MlxArray MlxArray::share() const {
    MlxArray result;
    check(mlx_array_set(&result.value_, value_), "share array");
    return result;
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

MlxMetalKernel::MlxMetalKernel(
    const std::string_view name,
    const std::span<const char* const> input_names,
    const std::span<const char* const> output_names,
    const std::string_view source,
    const std::string_view header) {
    std::vector<const char*> mutable_input_names(input_names.begin(), input_names.end());
    mlx_vector_string inputs = mlx_vector_string_new_data(
        mutable_input_names.data(), mutable_input_names.size());
    std::vector<const char*> mutable_output_names(output_names.begin(), output_names.end());
    mlx_vector_string outputs = mlx_vector_string_new_data(
        mutable_output_names.data(), mutable_output_names.size());
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
    for (const auto& [key, config] : configs_) {
        static_cast<void>(key);
        if (config.ctx != nullptr) mlx_fast_metal_kernel_config_free(config);
    }
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

std::vector<MlxArray> MlxMetalKernel::apply(
    const std::span<const MlxArray* const> inputs,
    const std::span<const MlxMetalOutputSpec> outputs,
    const std::span<const int, 3> grid,
    const std::span<const int, 3> threadgroup,
    const std::span<const MlxMetalDtypeTemplate> dtype_templates,
    const std::span<const MlxMetalIntTemplate> int_templates) const {
    if (outputs.empty()) throw std::runtime_error("Metal kernel requires an output");

    std::string config_key;
    const auto append_int = [&config_key](const int value) {
        config_key.append(std::to_string(value));
        config_key.push_back(',');
    };
    for (const MlxMetalOutputSpec& output : outputs) {
        append_int(static_cast<int>(output.dtype));
        append_int(static_cast<int>(output.shape.size()));
        for (const int dimension : output.shape) append_int(dimension);
    }
    for (const int dimension : grid) append_int(dimension);
    for (const int dimension : threadgroup) append_int(dimension);
    for (const MlxMetalDtypeTemplate& argument : dtype_templates) {
        config_key.append(argument.name);
        config_key.push_back('=');
        append_int(static_cast<int>(argument.value));
    }
    for (const MlxMetalIntTemplate& argument : int_templates) {
        config_key.append(argument.name);
        config_key.push_back('=');
        append_int(argument.value);
    }

    std::scoped_lock config_lock(config_mutex_);
    std::vector<mlx_array> raw_inputs;
    raw_inputs.reserve(inputs.size());
    for (const MlxArray* input : inputs) {
        if (input == nullptr) throw std::runtime_error("null Metal kernel input");
        raw_inputs.push_back(input->value_);
    }
    mlx_vector_array input_vector = mlx_vector_array_new_data(raw_inputs.data(), raw_inputs.size());
    mlx_vector_array output_vector = mlx_vector_array_new();
    auto cached = configs_.find(config_key);
    mlx_fast_metal_kernel_config config = cached == configs_.end()
        ? mlx_fast_metal_kernel_config_new()
        : cached->second;
    const bool new_config = cached == configs_.end();
    bool config_retained = !new_config;
    const auto cleanup = [&] {
        if (input_vector.ctx != nullptr) mlx_vector_array_free(input_vector);
        if (output_vector.ctx != nullptr) mlx_vector_array_free(output_vector);
        if (!config_retained && config.ctx != nullptr) {
            mlx_fast_metal_kernel_config_free(config);
        }
    };
    if (input_vector.ctx == nullptr || output_vector.ctx == nullptr || config.ctx == nullptr) {
        cleanup();
        throw std::runtime_error("MLX could not create Metal kernel arguments");
    }
    try {
        if (new_config) {
            for (const MlxMetalOutputSpec& output : outputs) {
                check(mlx_fast_metal_kernel_config_add_output_arg(
                          config, output.shape.data(), output.shape.size(), output.dtype),
                    "metal_kernel output");
            }
            check(mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]),
                "metal_kernel grid");
            check(mlx_fast_metal_kernel_config_set_thread_group(
                      config, threadgroup[0], threadgroup[1], threadgroup[2]),
                "metal_kernel threadgroup");
            for (const MlxMetalDtypeTemplate& argument : dtype_templates) {
                check(mlx_fast_metal_kernel_config_add_template_arg_dtype(
                          config, argument.name.c_str(), argument.value),
                    "metal_kernel dtype template");
            }
            for (const MlxMetalIntTemplate& argument : int_templates) {
                check(mlx_fast_metal_kernel_config_add_template_arg_int(
                          config, argument.name.c_str(), argument.value),
                    "metal_kernel integer template");
            }
            configs_.emplace(config_key, config);
            config_retained = true;
        }
        const Stream stream;
        check(mlx_fast_metal_kernel_apply(
                  &output_vector, kernel_, input_vector, config, stream.get()),
            "metal_kernel apply");
        if (mlx_vector_array_size(output_vector) != outputs.size()) {
            throw std::runtime_error("Metal kernel returned an unexpected output count");
        }
        std::vector<MlxArray> result(outputs.size());
        for (std::size_t index = 0; index < result.size(); ++index) {
            check(mlx_vector_array_get(&result[index].value_, output_vector, index),
                "metal_kernel get");
        }
        static_cast<void>(mlx_vector_array_free(input_vector));
        static_cast<void>(mlx_vector_array_free(output_vector));
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
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

MlxArray MlxArray::concatenate_many(const std::span<const MlxArray> arrays, const int axis) {
    if (arrays.empty()) throw std::invalid_argument("empty concatenate inputs");
    std::vector<mlx_array> values;
    values.reserve(arrays.size());
    for (const auto& array : arrays) values.push_back(array.get());
    const mlx_vector_array vector = mlx_vector_array_new_data(values.data(), values.size());
    if (vector.ctx == nullptr) throw std::runtime_error("MLX concatenate vector failed");
    MlxArray result;
    const int status = mlx_concatenate_axis(&result.value_, vector, axis, default_gpu_stream());
    static_cast<void>(mlx_vector_array_free(vector));
    check(status, "concatenate_many");
    return result;
}

MlxArray MlxArray::matmul(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_matmul(&result.value_, left.value_, right.value_, stream.get()), "matmul");
    return result;
}

MlxArray MlxArray::scaled_dot_product_attention(
    const MlxArray& queries,
    const MlxArray& keys,
    const MlxArray& values,
    const float scale,
    const bool causal) {
    MlxArray result;
    MlxArray mask;
    MlxArray sinks;
    const Stream stream;
    check(
        mlx_fast_scaled_dot_product_attention(
            &result.value_,
            queries.value_,
            keys.value_,
            values.value_,
            scale,
            causal ? "causal" : "",
            mask.value_,
            sinks.value_,
            false,
            stream.get()),
        "scaled dot product attention");
    return result;
}

MlxArray MlxArray::scaled_dot_product_attention(
    const MlxArray& queries,
    const MlxArray& keys,
    const MlxArray& values,
    const float scale,
    const MlxArray& mask) {
    MlxArray result;
    MlxArray sinks;
    const Stream stream;
    check(
        mlx_fast_scaled_dot_product_attention(
            &result.value_,
            queries.value_,
            keys.value_,
            values.value_,
            scale,
            "array",
            mask.value_,
            sinks.value_,
            false,
            stream.get()),
        "masked scaled dot product attention");
    return result;
}

MlxArray MlxArray::arange(
    const double start,
    const double stop,
    const double step,
    const mlx_dtype dtype) {
    MlxArray result;
    const Stream stream;
    check(mlx_arange(&result.value_, start, stop, step, dtype, stream.get()), "arange");
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

MlxArray MlxArray::expand_dims(const int axis) const {
    MlxArray result;
    const Stream stream;
    check(mlx_expand_dims(&result.value_, value_, axis, stream.get()), "expand_dims");
    return result;
}

MlxArray MlxArray::broadcast_to(const std::span<const int> shape) const {
    MlxArray result;
    const Stream stream;
    check(
        mlx_broadcast_to(&result.value_, value_, shape.data(), shape.size(), stream.get()),
        "broadcast_to");
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

MlxArray MlxArray::less_equal(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_less_equal(&result.value_, left.value_, right.value_, stream.get()),
        "less_equal");
    return result;
}

MlxArray MlxArray::greater_equal(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_greater_equal(&result.value_, left.value_, right.value_, stream.get()),
        "greater_equal");
    return result;
}

MlxArray MlxArray::logical_and(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_logical_and(&result.value_, left.value_, right.value_, stream.get()),
        "logical_and");
    return result;
}

MlxArray MlxArray::logical_or(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_logical_or(&result.value_, left.value_, right.value_, stream.get()),
        "logical_or");
    return result;
}

MlxArray MlxArray::where(
    const MlxArray& condition,
    const MlxArray& when_true,
    const MlxArray& when_false) {
    MlxArray result;
    const Stream stream;
    check(mlx_where(
              &result.value_, condition.value_, when_true.value_, when_false.value_,
              stream.get()),
        "where");
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

MlxArray MlxArray::put_along_axis(
    const MlxArray& input,
    const MlxArray& indices,
    const MlxArray& values,
    const int axis) {
    MlxArray result;
    const Stream stream;
    check(mlx_put_along_axis(
              &result.value_, input.value_, indices.value_, values.value_, axis,
              stream.get()),
        "put_along_axis");
    return result;
}

MlxArray MlxArray::argpartition_axis(const int kth, const int axis) const {
    MlxArray result;
    const Stream stream;
    check(mlx_argpartition_axis(&result.value_, value_, kth, axis, stream.get()),
        "argpartition_axis");
    return result;
}

MlxArray MlxArray::argsort_axis(const int axis) const {
    MlxArray result;
    const Stream stream;
    check(mlx_argsort_axis(&result.value_, value_, axis, stream.get()), "argsort");
    return result;
}

MlxArray MlxArray::floor_divide(const MlxArray& left, const MlxArray& right) {
    MlxArray result;
    const Stream stream;
    check(mlx_floor_divide(&result.value_, left.value_, right.value_, stream.get()),
        "floor_divide");
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

MlxArray MlxArray::gather_quantized_matmul(
    const MlxArray& input,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    const MlxArray& lhs_indices,
    const MlxArray& rhs_indices,
    const int group_size,
    const int bits,
    const bool sorted_indices,
    const bool transpose) {
    if (group_size <= 0 || bits <= 0) {
        throw std::runtime_error("invalid gathered quantization parameters");
    }
    MlxArray result;
    const Stream stream;
    check(mlx_gather_qmm(
              &result.value_,
              input.value_,
              weight.value_,
              scales.value_,
              biases.value_,
              lhs_indices.value_,
              rhs_indices.value_,
              transpose,
              mlx_optional_int{.value = group_size, .has_value = true},
              mlx_optional_int{.value = bits, .has_value = true},
              "affine",
              sorted_indices,
              stream.get()),
        "gather_qmm");
    return result;
}

void MlxArray::eval() const {
    check(mlx_array_eval(value_), "array_eval");
}

void MlxArray::eval_all(const std::span<const MlxArray* const> arrays) {
    if (arrays.empty()) return;
    std::vector<mlx_array> values;
    values.reserve(arrays.size());
    for (const MlxArray* array : arrays) {
        if (array == nullptr) throw std::runtime_error("cannot evaluate a null MLX array");
        values.push_back(array->value_);
    }
    mlx_vector_array outputs = mlx_vector_array_new_data(values.data(), values.size());
    const int status = mlx_eval(outputs);
    static_cast<void>(mlx_vector_array_free(outputs));
    check(status, "eval_all");
}

void MlxArray::clear_cache() {
    check(mlx_clear_cache(), "clear_cache");
}

std::size_t MlxArray::set_cache_limit(const std::size_t bytes) {
    std::size_t previous = 0;
    check(mlx_set_cache_limit(&previous, bytes), "set_cache_limit");
    return previous;
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

std::optional<std::string> MlxSafetensors::metadata(const std::string_view name) const {
    const char* value = nullptr;
    const std::string key(name);
    const int status = mlx_map_string_to_string_get(&value, metadata_, key.c_str());
    if (status == 2) return std::nullopt;
    check(status, "map_string_to_string_get");
    return value == nullptr ? std::optional<std::string>{} : std::string(value);
}

void MlxSafetensors::save(
    const std::filesystem::path& path,
    const std::span<const NamedArray> arrays,
    const std::span<const std::pair<std::string, std::string>> metadata) {
    mlx_map_string_to_array tensor_map = mlx_map_string_to_array_new();
    mlx_map_string_to_string metadata_map = mlx_map_string_to_string_new();
    if (tensor_map.ctx == nullptr || metadata_map.ctx == nullptr) {
        if (tensor_map.ctx != nullptr) mlx_map_string_to_array_free(tensor_map);
        if (metadata_map.ctx != nullptr) mlx_map_string_to_string_free(metadata_map);
        throw std::runtime_error("MLX could not allocate safetensors maps");
    }
    const auto cleanup = [&] {
        mlx_map_string_to_array_free(tensor_map);
        mlx_map_string_to_string_free(metadata_map);
    };
    try {
        for (const NamedArray& named : arrays) {
            if (named.array == nullptr) {
                throw std::runtime_error("cannot save a null MLX array");
            }
            check(
                mlx_map_string_to_array_insert(
                    tensor_map, named.name.c_str(), named.array->get()),
                "map_string_to_array_insert");
        }
        for (const auto& [key, value] : metadata) {
            check(
                mlx_map_string_to_string_insert(
                    metadata_map, key.c_str(), value.c_str()),
                "map_string_to_string_insert");
        }
        check(
            mlx_save_safetensors(path.c_str(), tensor_map, metadata_map),
            "save_safetensors");
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

MlxArray MlxTensorStore::tensor(const std::string_view name) {
    std::scoped_lock lock(mutex_);
    if (paged_) return read_tensor(std::string(name), std::nullopt);
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

TensorView MlxTensorStore::disk_view(const std::string& name) {
    const auto mapping = manifest_.weight_map().find(name);
    if (mapping == manifest_.weight_map().end()) throw std::runtime_error("missing tensor: " + name);
    auto& catalog = catalogs_[mapping->second];
    if (!catalog) catalog = std::make_unique<SafetensorsFile>(manifest_.directory() / mapping->second);
    return catalog->tensor(name);
}

MlxArray MlxTensorStore::read_tensor(const std::string& name, const std::optional<std::size_t> row,
                                    const std::span<const std::byte> prefetched) {
    const TensorView view = disk_view(name);
    const auto& shard = manifest_.weight_map().at(name);
    const auto mapping = catalogs_.at(shard)->mapped_view();
    std::size_t offset = static_cast<std::size_t>(view.bytes.data() - mapping.data());
    std::size_t bytes = view.bytes.size();
    std::vector<int> shape;
    for (std::size_t i = row ? 1 : 0; i < view.shape.size(); ++i) {
        if (view.shape[i] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("paged tensor dimension exceeds int");
        shape.push_back(static_cast<int>(view.shape[i]));
    }
    if (row) {
        if (view.shape.size() != 3 || *row >= view.shape[0] || bytes % view.shape[0] != 0)
            throw std::runtime_error("invalid paged expert row");
        bytes /= view.shape[0];
        offset += *row * bytes;
    }
    // Bound staging independently of allocator/process limits. No complete
    // expert tensor is loaded through this path.
    if (bytes == 0 || bytes > 1024ULL * 1024ULL * 1024ULL)
        throw std::runtime_error("paged tensor exceeds staging bound");
    mlx_dtype dtype;
    if (view.dtype == "U32") dtype = MLX_UINT32;
    else if (view.dtype == "BF16") dtype = MLX_BFLOAT16;
    else if (view.dtype == "F32") dtype = MLX_FLOAT32;
    else if (view.dtype == "F16") dtype = MLX_FLOAT16;
    else throw std::runtime_error("unsupported paged tensor dtype");
    std::vector<std::byte> staging;
    if (prefetched.empty()) {
        staging.resize(bytes);
        std::ifstream file(manifest_.directory() / shard, std::ios::binary);
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(reinterpret_cast<char*>(staging.data()), static_cast<std::streamsize>(bytes));
        if (!file) throw std::runtime_error("paged tensor read failed");
    } else if (prefetched.size() != bytes) throw std::runtime_error("prefetched byte size mismatch");
    const auto* data = prefetched.empty() ? staging.data() : prefetched.data();
    MlxArray result(mlx_array_new_data(data, shape.data(), static_cast<int>(shape.size()), dtype));
    if (result.get().ctx == nullptr) throw std::runtime_error("paged tensor allocation failed");
    if (mlx_array_nbytes(result.get()) != bytes)
        throw std::runtime_error("paged tensor byte geometry mismatch");
    // The pinned MLX-C new_data constructor synchronously copies host data.
    // Keep the original per-field fences as an A/B reference, but batch mode
    // only fences once after the complete expert bundle has been constructed.
    if (!batch_experts_ || !row) {
        result.eval();
        check(mlx_synchronize(default_gpu_stream()), "paged tensor completion");
    }
    return result;
}

struct MlxTensorStore::FixedLayer {
    struct Field {
        std::shared_ptr<void> storage;
        std::filesystem::path path;
        std::size_t offset{}, bytes{};
    };
    std::array<Field,9> disk;
    ExpertArrays arrays;
    std::vector<ExpertLease> views;
    std::vector<int> ids;
    std::vector<std::uint64_t> touches;
    std::uint64_t clock{};
    bool failed{false};
    void fill(std::size_t slot, std::size_t expert) {
        check(mlx_synchronize(default_gpu_stream()), "fixed slot write fence");
        // On partial I/O failure this layer becomes unusable; abort the probe.
        failed = true;
        for (auto& f : disk) {
            std::ifstream file(f.path,std::ios::binary);
            file.seekg(static_cast<std::streamoff>(f.offset+expert*f.bytes));
            file.read(static_cast<char*>(f.storage.get())+slot*f.bytes,
                static_cast<std::streamsize>(f.bytes));
            if (!file) throw std::runtime_error("fixed slot read failed");
        }
        ids[slot] = static_cast<int>(expert); // publish only after all nine reads
        failed = false;
    }
};

void MlxTensorStore::enable_fixed_slots(std::size_t hot_count) {
    const char* strict=std::getenv("MLX_STRICT_MEMORY_LIMIT");
    if (!std::getenv("QWEN38_MEMORY_GUARD") || !strict || std::string_view(strict)!="1")
        throw std::runtime_error("fixed slots require guarded strict research runtime");
    if ((hot_count != 224 && hot_count != 256 && hot_count != 288) || !catalogs_.empty() || !shards_.empty() || fixed_slots())
        throw std::runtime_error("fixed slots must be configured before model load");
    fixed_hot_ = hot_count;
    paged_ = true;
    batch_experts_ = true;
}

MlxTensorStore::FixedLayer& MlxTensorStore::fixed_layer(const std::string& prefix) {
    if (auto it = fixed_layers_.find(prefix); it != fixed_layers_.end()) return *it->second;
    auto layer = std::make_shared<FixedLayer>();
    const std::size_t capacity = fixed_hot_ + (fixed_hot_ < 288 ? 8 : 0);
    layer->ids.assign(capacity,-1); layer->touches.resize(capacity);
    std::size_t index = 0, charged = 0;
    for (const char* projection : {"gate_proj","up_proj","down_proj"}) {
        for (const char* field : {"weight","scales","biases"}) {
            const auto name = prefix+".switch_mlp."+projection+"."+field;
            const auto view = disk_view(name);
            const std::size_t rows = index < 6 ? 640 : 2560;
            const std::size_t cols = (index < 6 ? 2560 : 640)/(index%3 == 0 ? 8 : 64);
            if (view.shape.size()!=3 || view.shape[0]!=288 || view.shape[1]!=rows ||
                view.shape[2]!=cols || view.dtype != (index%3==0 ? "U32" : "BF16"))
                throw std::runtime_error("fixed slots require Q4/group64 BF16 experts");
            auto& f = layer->disk[index];
            const auto& shard = manifest_.weight_map().at(name);
            f.path = manifest_.directory()/shard;
            f.offset = static_cast<std::size_t>(view.bytes.data()-catalogs_.at(shard)->mapped_view().data());
            f.bytes = view.bytes.size()/288;
            const auto bytes = capacity*f.bytes;
            const auto padded = (bytes+16383)/16384*16384;
            f.storage = std::shared_ptr<void>(std::aligned_alloc(16384,padded),std::free);
            if (!f.storage) throw std::bad_alloc();
            std::memset(f.storage.get(),0,padded);
            // Managed-array ownership outlives the layer if a view is leased.
            struct Owner { std::shared_ptr<void> bytes; std::shared_ptr<bool> released; };
            auto released = std::make_shared<bool>(false);
            auto* owner = new Owner{f.storage,released};
            const std::vector<int> shape{static_cast<int>(capacity),static_cast<int>(rows),static_cast<int>(cols)};
            auto raw = mlx_array_new_data_managed_payload(f.storage.get(),shape.data(),3,
                index%3==0 ? MLX_UINT32 : MLX_BFLOAT16,owner,[](void* p) {
                    auto* o=static_cast<Owner*>(p); *o->released=true; delete o;
                });
            if (!raw.ctx) {
                if (!*released) delete owner;
                throw std::runtime_error("fixed pool managed import refused");
            }
            layer->arrays.emplace_back(raw); charged += padded; ++index;
        }
    }
    for (std::size_t slot=0; slot<capacity; ++slot) {
        auto views=std::make_shared<ExpertArrays>();
        for (auto& a : layer->arrays) {
            auto end=a.shape(); end[0]=static_cast<int>(slot+1);
            views->push_back(a.slice(std::vector<int>{static_cast<int>(slot),0,0},end,
                std::vector<int>{1,1,1}).reshape(std::vector<int>{end[1],end[2]}));
            views->back().eval();
        }
        layer->views.push_back(std::move(views));
    }
    for (std::size_t expert=0; expert<fixed_hot_; ++expert) layer->fill(expert,expert);
    fixed_stats_.resident_bytes += charged;
    fixed_stats_.peak_bytes = fixed_stats_.resident_bytes;
    return *fixed_layers_.emplace(prefix,std::move(layer)).first->second;
}

void MlxTensorStore::preload_fixed_slots() {
    if (!fixed_slots()) throw std::runtime_error("fixed slots disabled");
    for (int layer=0; layer<48; ++layer)
        fixed_layer("language_model.model.layers."+std::to_string(layer)+".mlp");
}

bool MlxTensorStore::fixed_batch_fits(std::span<const std::size_t> ids) const {
    // Apply the same overflow arithmetic to the all-resident control.
    return !fixed_slots() || std::count_if(ids.begin(),ids.end(),[&](auto id) {
        return id>=std::min<std::size_t>(fixed_hot_,256); })<=8;
}

MlxTensorStore::ExpertLease MlxTensorStore::fixed_expert(const std::string& prefix,std::size_t id) {
    if (id>=288) throw std::runtime_error("fixed expert outside range");
    auto& layer=fixed_layer(prefix);
    if (layer.failed) throw std::runtime_error("fixed layer invalid after I/O error");
    auto found=std::find(layer.ids.begin(),layer.ids.end(),static_cast<int>(id));
    std::size_t slot=static_cast<std::size_t>(found-layer.ids.begin());
    if (found==layer.ids.end()) {
        slot=layer.ids.size();
        for (std::size_t s=fixed_hot_; s<layer.ids.size(); ++s) {
            if (layer.views[s].use_count()!=1) continue;
            if (slot==layer.ids.size() || layer.touches[s]<layer.touches[slot]) slot=s;
        }
        if (slot==layer.ids.size()) throw std::runtime_error("all fixed cold slots leased");
        const auto start=std::chrono::steady_clock::now();
        if (layer.ids[slot]>=0) ++fixed_stats_.evictions;
        layer.fill(slot,id);
        expert_load_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        ++fixed_stats_.misses;
    } else ++fixed_stats_.hits;
    layer.touches[slot]=++layer.clock;
    return layer.views[slot];
}

const MlxTensorStore::ExpertArrays& MlxTensorStore::fixed_fields(std::string_view prefix) const {
    return fixed_layers_.at(std::string(prefix))->arrays;
}
std::vector<std::int32_t> MlxTensorStore::fixed_ids(std::string_view prefix,std::span<const std::size_t> ids) const {
    const auto& layer=*fixed_layers_.at(std::string(prefix));
    std::vector<std::int32_t> result;
    for (auto id:ids) {
        auto it=std::find(layer.ids.begin(),layer.ids.end(),static_cast<int>(id));
        if (it==layer.ids.end()) throw std::runtime_error("unleased fixed expert");
        result.push_back(static_cast<std::int32_t>(it-layer.ids.begin()));
    }
    return result;
}

MlxTensorStore::ExpertLease MlxTensorStore::expert(const std::string_view prefix, const std::size_t id) {
    std::scoped_lock lock(mutex_);
    if (!paged_) throw std::runtime_error("expert paging is disabled");
    if (fixed_slots()) {
        auto lease=fixed_expert(std::string(prefix),id);
        if (trace_enabled_) {
            std::size_t bytes=0;
            for (const auto& f:fixed_layers_.at(std::string(prefix))->disk) bytes+=f.bytes;
            trace_expert(ExpertTraceKind::access,std::string(prefix)+"/"+std::to_string(id),bytes);
        }
        return lease;
    }
    std::vector<std::string> names;
    std::size_t bytes = 0;
    for (const auto* projection : {"gate_proj", "up_proj", "down_proj"}) {
        for (const auto* field : {"weight", "scales", "biases"}) {
            names.push_back(std::string(prefix) + ".switch_mlp." + projection + "." + field);
            const auto view = disk_view(names.back());
            if (view.shape.size() != 3 || id >= view.shape[0] || view.bytes.size() % view.shape[0] != 0)
                throw std::runtime_error("invalid paged expert geometry");
            const std::size_t raw = view.bytes.size() / view.shape[0];
            // This experimental runtime disables the MLX free-block cache.
            constexpr std::size_t page = 16384;
            if (raw > 16 * 1024 * 1024) throw std::runtime_error("expert field too large");
            bytes += (raw + page - 1) / page * page;
        }
    }
    const auto key = std::string(prefix) + "/" + std::to_string(id);
    auto lease = experts_.acquire(key, bytes, [&] {
        const auto started = std::chrono::steady_clock::now();
        // Retire prior asynchronous work before taking a global allocator
        // snapshot; otherwise old graph releases can make the delta negative.
        check(mlx_synchronize(default_gpu_stream()), "paged pre-load accounting fence");
        std::size_t before = 0;
        check(mlx_get_active_memory(&before), "paged memory before load");
        auto arrays = std::make_shared<ExpertArrays>();
        arrays->reserve(names.size());
        if (!parallel_reads_) {
            for (const auto& name : names) arrays->push_back(read_tensor(name, id));
        } else {
            struct Request { std::filesystem::path path; std::size_t offset, bytes; };
            std::array<Request, 9> requests;
            for (std::size_t i = 0; i < names.size(); ++i) {
                const auto view = disk_view(names[i]);
                const auto& shard = manifest_.weight_map().at(names[i]);
                const auto extent = catalogs_.at(shard)->mapped_view();
                const auto row_bytes = view.bytes.size() / view.shape[0];
                requests[i] = {manifest_.directory() / shard,
                    static_cast<std::size_t>(view.bytes.data() - extent.data()) + id * row_bytes, row_bytes};
            }
            // At most three CPU readers, one expert at a time; no MLX calls on
            // worker threads. Futures join even when a read/copy throws.
            using Buffers = std::array<std::vector<std::byte>, 3>;
            std::array<std::future<Buffers>, 3> reads;
            for (std::size_t p = 0; p < reads.size(); ++p) {
                reads[p] = std::async(std::launch::async, [&requests, p] {
                    Buffers buffers;
                    for (std::size_t f = 0; f < buffers.size(); ++f) {
                        const auto& request = requests[p * 3 + f];
                        buffers[f].resize(request.bytes);
                        std::ifstream file(request.path, std::ios::binary);
                        file.seekg(static_cast<std::streamoff>(request.offset));
                        file.read(reinterpret_cast<char*>(buffers[f].data()),
                            static_cast<std::streamsize>(request.bytes));
                        if (!file) throw std::runtime_error("parallel expert read failed");
                    }
                    return buffers;
                });
            }
            for (std::size_t p = 0; p < reads.size(); ++p) {
                const auto buffers = reads[p].get();
                for (std::size_t f = 0; f < buffers.size(); ++f)
                    arrays->push_back(read_tensor(names[p * 3 + f], id, buffers[f]));
            }
        }
        check(mlx_synchronize(default_gpu_stream()), "paged expert bundle completion");
        std::size_t after = 0;
        check(mlx_get_active_memory(&after), "paged memory after load");
        if (after < before || after - before > bytes)
            throw std::runtime_error("paged expert accounting mismatch before=" + std::to_string(before) +
                " after=" + std::to_string(after) + " charge=" + std::to_string(bytes));
        expert_load_ms_ += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return arrays;
    });
    if (trace_enabled_) trace_expert(ExpertTraceKind::access, key, bytes);
    return lease;
}

void MlxTensorStore::trace_expert(const ExpertTraceKind kind, std::string key, const std::size_t bytes) {
    if (!trace_enabled_) return;
    if (expert_trace_.size() >= 200000) throw std::runtime_error("expert trace reached bounded event limit");
    expert_trace_.push_back({kind, std::move(key), bytes});
}

void MlxTensorStore::finish_expert_batch() {
    trace_expert(ExpertTraceKind::boundary, {}, 0);
}

bool MlxTensorStore::set_expert_budget(const std::size_t bytes) {
    std::scoped_lock lock(mutex_);
    if (!paged_) throw std::runtime_error("expert paging is disabled");
    if (fixed_slots()) throw std::runtime_error("fixed slot budget is immutable");
    check(mlx_synchronize(default_gpu_stream()), "paged resize fence");
    const bool result = experts_.set_budget(bytes);
    trace_expert(ExpertTraceKind::budget, {}, bytes);
    return result;
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
