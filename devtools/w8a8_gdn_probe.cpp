#include "qwen38/mlx_backend.hpp"
#include "w8a8_gdn_probe_kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using qwen38::MlxArray;
using Clock = std::chrono::steady_clock;
constexpr int q4_group_size = 64;
constexpr int q4_bits = 4;

struct Options {
    int rows = 512;
    int k = 2560;
    int n = 6144;
    int warmups = 3;
    int iterations = 10;
    std::filesystem::path shard;
    std::string tensor;
};

int positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<int>(value);
}

Options options_from(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](const char* name) -> const char* {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[i];
        };
        if (arg == "--rows") options.rows = positive(value("--rows"), "rows");
        else if (arg == "--k") options.k = positive(value("--k"), "K");
        else if (arg == "--n") options.n = positive(value("--n"), "N");
        else if (arg == "--warmups") options.warmups = positive(value("--warmups"), "warmups");
        else if (arg == "--iterations") options.iterations = positive(value("--iterations"), "iterations");
        else if (arg == "--shard") options.shard = value("--shard");
        else if (arg == "--tensor") options.tensor = value("--tensor");
        else if (arg == "--help") {
            std::cout << "usage: w8a8-gdn-probe [--rows 32|128|512] [--k K --n N] "
                         "[--warmups W --iterations I] [--shard FILE --tensor BASE]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + std::string(arg));
    }
    if (options.shard.empty() != options.tensor.empty())
        throw std::runtime_error("--shard and --tensor must be supplied together");
    if (options.rows != 32 && options.rows != 128 && options.rows != 512)
        throw std::runtime_error("--rows must be 32, 128, or 512");
    if (options.n % 128 != 0 || options.k % 16 != 0)
        throw std::runtime_error("MPP requires N divisible by 128 and K divisible by 16");
    return options;
}

struct Q4Tuple { MlxArray packed, scales, biases; };

class VectorGuard final {
public:
    VectorGuard() : value_(mlx_vector_array_new()) {
        if (value_.ctx == nullptr) throw std::runtime_error("MLX could not allocate vector");
    }
    ~VectorGuard() { static_cast<void>(mlx_vector_array_free(value_)); }
    [[nodiscard]] mlx_vector_array get() const noexcept { return value_; }
    [[nodiscard]] mlx_vector_array* address() noexcept { return &value_; }
private:
    mlx_vector_array value_{};
};

class StreamGuard final {
public:
    StreamGuard() : value_(mlx_default_gpu_stream_new()) {
        if (value_.ctx == nullptr) throw std::runtime_error("MLX did not provide a GPU stream");
    }
    ~StreamGuard() { static_cast<void>(mlx_stream_free(value_)); }
    [[nodiscard]] mlx_stream get() const noexcept { return value_; }
private:
    mlx_stream value_{};
};

MlxArray vector_element(const mlx_vector_array values, const std::size_t index) {
    mlx_array value = mlx_array_new();
    if (mlx_vector_array_get(&value, values, index) != 0) {
        static_cast<void>(mlx_array_free(value));
        throw std::runtime_error("MLX could not read quantization output");
    }
    return MlxArray(value);
}

Q4Tuple quantize_affine_q4(const MlxArray& source) {
    VectorGuard outputs;
    StreamGuard stream;
    const int status = mlx_quantize(
        outputs.address(), source.get(),
        mlx_optional_int{.value = q4_group_size, .has_value = true},
        mlx_optional_int{.value = q4_bits, .has_value = true}, "affine", mlx_array{},
        stream.get());
    if (status != 0 || mlx_vector_array_size(outputs.get()) != 3)
        throw std::runtime_error("MLX affine-Q4 quantization failed");
    return {vector_element(outputs.get(), 0), vector_element(outputs.get(), 1),
            vector_element(outputs.get(), 2)};
}

struct W8Weight {
    std::vector<std::int32_t> values;
    std::vector<float> scales;
    std::vector<float> q4_dequantized;
    int n = 0;
    int k = 0;
};

W8Weight quantize_w8(std::vector<float> source, const int n, const int k) {
    const std::size_t k_size = static_cast<std::size_t>(k);
    if (source.size() != static_cast<std::size_t>(n) * k_size)
        throw std::runtime_error("weight shape and data size disagree");
    W8Weight result{std::vector<std::int32_t>(source.size()),
                    std::vector<float>(static_cast<std::size_t>(n)),
                    std::move(source), n, k};
    for (int row = 0; row < n; ++row) {
        float absmax = 0.0F;
        for (int column = 0; column < k; ++column)
            absmax = std::max(absmax, std::abs(result.q4_dequantized[
                static_cast<std::size_t>(row) * k_size + static_cast<std::size_t>(column)]));
        const float scale = absmax == 0.0F ? 1.0F : absmax / 127.0F;
        result.scales[static_cast<std::size_t>(row)] = scale;
        for (int column = 0; column < k; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * k_size +
                                      static_cast<std::size_t>(column);
            result.values[index] = static_cast<std::int32_t>(std::clamp(
                std::round(result.q4_dequantized[index] / scale), -128.0F, 127.0F));
        }
    }
    return result;
}

Q4Tuple synthetic_q4(const int n, const int k) {
    std::vector<float> values(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int>((i * 17 + i / 7) % 251) - 125) / 512.0F;
    const std::array<int, 2> shape{n, k};
    MlxArray source = MlxArray::from_float32(values, shape).astype(MLX_BFLOAT16);
    return quantize_affine_q4(source);
}

Q4Tuple load_q4(const Options& options) {
    qwen38::MlxSafetensors file(options.shard);
    return {file.tensor(options.tensor + ".weight"), file.tensor(options.tensor + ".scales"),
            file.tensor(options.tensor + ".biases")};
}

W8Weight make_w8(const Q4Tuple& q4) {
    MlxArray dequantized = MlxArray::dequantize(
        q4.packed, q4.scales, q4.biases, q4_group_size, q4_bits, MLX_FLOAT32);
    const std::vector<int> shape = dequantized.shape();
    if (shape.size() != 2) throw std::runtime_error("dequantized Q4 weight must be rank 2");
    return quantize_w8(dequantized.astype(MLX_FLOAT32).to_float32(), shape[0], shape[1]);
}

std::vector<float> make_input(const int rows, const int k) {
    std::mt19937 generator(38);
    std::normal_distribution<float> distribution(0.0F, 0.35F);
    std::vector<float> values(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k));
    std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
    return values;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("cosine shape mismatch");
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    return dot / std::sqrt(aa * bb);
}

struct Quality {
    double cosine = 0.0;
    double max_abs = 0.0;
    double rmse = 0.0;
    bool finite = true;
};

Quality compare(const std::vector<float>& actual, const std::vector<float>& reference) {
    if (actual.size() != reference.size()) throw std::runtime_error("quality shape mismatch");
    Quality result;
    double squared_error = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        result.finite = result.finite && std::isfinite(actual[i]) && std::isfinite(reference[i]);
        const double error = static_cast<double>(actual[i]) - reference[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        squared_error += error * error;
    }
    result.cosine = cosine(actual, reference);
    result.rmse = std::sqrt(squared_error / static_cast<double>(actual.size()));
    return result;
}

double weight_cosine(const W8Weight& weight) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (int row = 0; row < weight.n; ++row) {
        const double scale = weight.scales[static_cast<std::size_t>(row)];
        for (int column = 0; column < weight.k; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(weight.k) +
                                      static_cast<std::size_t>(column);
            const double source = weight.q4_dequantized[index];
            const double w8 = static_cast<double>(weight.values[index]) * scale;
            dot += source * w8; aa += source * source; bb += w8 * w8;
        }
    }
    return dot / std::sqrt(aa * bb);
}

std::int32_t expected_dot(const int row, const int column, const int k) {
    const auto product = [&](const int index) {
        return ((row * k + index) % 7 - 3) * ((column * k + index) % 5 - 2);
    };
    std::int32_t period_sum = 0;
    for (int index = 0; index < 35; ++index) period_sum += product(index);
    std::int32_t result = period_sum * (k / 35);
    for (int index = (k / 35) * 35; index < k; ++index) result += product(index);
    return result;
}

float prove_exact(const int rows, const int n, const int k) {
    std::vector<std::int32_t> a(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(k));
    std::vector<std::int32_t> b(
        static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int>(i % 7) - 3;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int>(i % 5) - 2;
    const std::array<int, 2> a_shape{rows, k}, b_shape{n, k};
    MlxArray aq = MlxArray::from_int32(a, a_shape).astype(MLX_INT8);
    MlxArray w = MlxArray::from_int32(b, b_shape).astype(MLX_INT8);
    const char* names[]{"aq", "w"};
    qwen38::MlxMetalKernel kernel("qwen38_probe_mpp_int32", names, "y",
        qwen38::w8a8_probe_metal::mpp_int32, qwen38::w8a8_probe_metal::mpp_header);
    const std::array<const MlxArray*, 2> inputs{&aq, &w};
    const std::array<qwen38::MlxMetalOutputSpec, 1> outputs{{
        {.shape = {rows, n}, .dtype = MLX_INT32}}};
    const int block_m = rows == 32 ? 32 : 128;
    const int groups_m = rows / block_m;
    const int threads = rows == 32 ? 128 : 512;
    const int tiles_n = n / 128;
    const std::array<int, 3> grid{tiles_n * threads, groups_m, 1}, group{threads, 1, 1};
    const std::array<qwen38::MlxMetalIntTemplate, 8> args{{
        {"M", rows}, {"N", n}, {"K", k}, {"BM", block_m},
        {"WM", rows == 32 ? 1 : 4}, {"SWIZZLE", 0},
        {"TILES_M", groups_m}, {"TILES_N", tiles_n}}};
    const std::vector<float> actual = kernel.apply(inputs, outputs, grid, group, {}, args)[0]
                                          .astype(MLX_FLOAT32).to_float32();
    float max_abs = 0.0F;
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < n; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(n) +
                                      static_cast<std::size_t>(column);
            max_abs = std::max(max_abs, std::abs(
                actual[index] - static_cast<float>(expected_dot(row, column, k))));
        }
    return max_abs;
}

struct Stats { double mean, median, minimum, maximum; };

Stats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) / 2.0 : samples[middle];
    return {std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
            median, samples.front(), samples.back()};
}

template <typename Function> double measure(Function&& function) {
    const auto start = Clock::now();
    MlxArray output = function();
    output.eval();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace

int main(int argc, char** argv) try {
    const Options options = options_from(argc, argv);
    const float proof32 = prove_exact(32, options.n, options.k);
    const float proof128 = prove_exact(128, options.n, options.k);
    if (proof32 != 0.0F || proof128 != 0.0F)
        throw std::runtime_error("MPP INT8xINT8 integer proof failed");

    Q4Tuple q4 = options.shard.empty() ? synthetic_q4(options.n, options.k) : load_q4(options);
    W8Weight weight = make_w8(q4);
    if (weight.n != options.n || weight.k != options.k)
        throw std::runtime_error("Q4 tuple shape does not match --n/--k");
    const int rows = options.rows, n = weight.n, k = weight.k;
    const std::array<int, 2> input_shape{rows, k}, weight_shape{n, k};
    const std::array<int, 1> scale_shape{n};
    MlxArray input = MlxArray::from_float32(make_input(rows, k), input_shape).astype(MLX_BFLOAT16);
    MlxArray w8 = MlxArray::from_int32(weight.values, weight_shape).astype(MLX_INT8);
    MlxArray wscale = MlxArray::from_float32(weight.scales, scale_shape);

    const char* qin_names[]{"x"};
    const char* qout_names[]{"aq", "scale"};
    qwen38::MlxMetalKernel quantizer("qwen38_probe_w8a8_quantize", qin_names, qout_names,
        qwen38::w8a8_probe_metal::quantize, qwen38::w8a8_probe_metal::header);
    const std::array<const MlxArray*, 1> qi{&input};
    const std::array<qwen38::MlxMetalOutputSpec, 2> qo{{
        {.shape = {rows, k}, .dtype = MLX_INT8}, {.shape = {rows}, .dtype = MLX_FLOAT32}}};
    const std::array<int, 3> qgrid{rows * 256, 1, 1}, qgroup{256, 1, 1};
    const std::array<qwen38::MlxMetalIntTemplate, 1> qargs{{{"K", k}}};

    const char* min_names[]{"aq", "w", "scale", "wscale"};
    qwen38::MlxMetalKernel mpp("qwen38_probe_w8a8_mpp_fused", min_names, "y",
        qwen38::w8a8_probe_metal::mpp_fused, qwen38::w8a8_probe_metal::mpp_header);
    const int block_m = rows == 32 ? 32 : 128;
    const int groups_m = rows / block_m;
    const int threads = rows == 32 ? 128 : 512;
    const int tiles_n = n / 128;
    const std::array<qwen38::MlxMetalOutputSpec, 1> mo{{
        {.shape = {rows, n}, .dtype = MLX_FLOAT16}}};
    const std::array<int, 3> mgrid{tiles_n * threads, groups_m, 1}, mgroup{threads, 1, 1};
    const std::array<qwen38::MlxMetalIntTemplate, 8> margs{{
        {"M", rows}, {"N", n}, {"K", k}, {"BM", block_m},
        {"WM", rows == 32 ? 1 : 4}, {"SWIZZLE", 0},
        {"TILES_M", groups_m}, {"TILES_N", tiles_n}}};

    const auto run_mpp = [&]() {
        auto quantized = quantizer.apply(qi, qo, qgrid, qgroup, {}, qargs);
        const std::array<const MlxArray*, 4> inputs{&quantized[0], &w8, &quantized[1], &wscale};
        return mpp.apply(inputs, mo, mgrid, mgroup, {}, margs)[0].astype(MLX_BFLOAT16);
    };
    const auto run_q4 = [&]() {
        return MlxArray::quantized_matmul(input, q4.packed, q4.scales, q4.biases,
                                          q4_group_size, q4_bits);
    };
    MlxArray mpp_output = run_mpp();
    MlxArray q4_output = run_q4();
    const std::array<const MlxArray*, 2> outputs{&mpp_output, &q4_output};
    MlxArray::eval_all(outputs);
    const Quality quality = compare(mpp_output.astype(MLX_FLOAT32).to_float32(),
                                    q4_output.astype(MLX_FLOAT32).to_float32());
    if (!quality.finite || quality.cosine < 0.99)
        throw std::runtime_error("MPP affine output quality check failed");
    const double source_weight_cosine = weight_cosine(weight);

    for (int i = 0; i < options.warmups; ++i) {
        if (i % 2 == 0) { run_mpp().eval(); run_q4().eval(); }
        else { run_q4().eval(); run_mpp().eval(); }
    }
    std::vector<double> mpp_samples, q4_samples;
    for (int i = 0; i < options.iterations; ++i) {
        if (i % 2 == 0) {
            mpp_samples.push_back(measure(run_mpp)); q4_samples.push_back(measure(run_q4));
        } else {
            q4_samples.push_back(measure(run_q4)); mpp_samples.push_back(measure(run_mpp));
        }
    }
    const Stats mpp_stats = summarize(std::move(mpp_samples));
    const Stats q4_stats = summarize(std::move(q4_samples));
    const double sidecar_mib = (static_cast<double>(n) * k + static_cast<double>(n) * sizeof(float)) /
                               (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(6)
              << "{\"mode\":\"" << (options.shard.empty() ? "synthetic-affine-q4" : "real-affine-q4")
              << "\",\"activation_source\":\"synthetic-normal-seed-38-bf16\""
              << ",\"rows\":" << rows << ",\"n\":" << n << ",\"k\":" << k
              << ",\"warmups\":" << options.warmups << ",\"iterations\":" << options.iterations
              << ",\"integer_proof\":{\"m32_max_abs\":" << proof32
              << ",\"m128_max_abs\":" << proof128 << "}"
              << ",\"mpp_output_dtype\":\"bfloat16\""
              << ",\"quality\":{\"finite\":" << (quality.finite ? "true" : "false")
              << ",\"mpp_vs_q4_cosine\":" << quality.cosine
              << ",\"max_abs\":" << quality.max_abs << ",\"rmse\":" << quality.rmse
              << ",\"w8_vs_dequantized_q4_weight_cosine\":" << source_weight_cosine << "}"
              << ",\"timing_ms\":{\"mpp_fused_mean\":" << mpp_stats.mean
              << ",\"mpp_fused_median\":" << mpp_stats.median
              << ",\"mpp_fused_min\":" << mpp_stats.minimum << ",\"mpp_fused_max\":" << mpp_stats.maximum
              << ",\"q4_bf16_mean\":" << q4_stats.mean << ",\"q4_bf16_median\":" << q4_stats.median
              << ",\"q4_bf16_min\":" << q4_stats.minimum << ",\"q4_bf16_max\":" << q4_stats.maximum
              << ",\"mean_speedup\":" << q4_stats.mean / mpp_stats.mean << "}"
              << ",\"w8_sidecar_mib\":" << sidecar_mib << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "w8a8-gdn-probe: " << error.what() << '\n';
    return 1;
}
