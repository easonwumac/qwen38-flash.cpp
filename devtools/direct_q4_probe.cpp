#include "qwen38/mlx_backend.hpp"
#include "direct_q4_kernels.hpp"
#include "w8a8_gdn_probe_kernels.hpp"

#include <algorithm>
#include <array>
#include <bit>
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
constexpr int group_size = 64;

struct Options {
    int warmups = 3;
    int iterations = 15;
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

Options parse_options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* name) {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[i];
        };
        if (arg == "--shard") result.shard = next("--shard");
        else if (arg == "--tensor") result.tensor = next("--tensor");
        else if (arg == "--warmups") result.warmups = positive(next("--warmups"), "warmups");
        else if (arg == "--iterations") result.iterations = positive(next("--iterations"), "iterations");
        else if (arg == "--help") {
            std::cout << "usage: qwen38-direct-q4-probe --shard FILE --tensor BASE "
                         "[--warmups 3 --iterations 15]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + std::string(arg));
    }
    if (result.shard.empty() || result.tensor.empty())
        throw std::runtime_error("--shard and --tensor are required");
    if (result.iterations < 15) throw std::runtime_error("--iterations must be at least 15");
    return result;
}

struct Q4 { MlxArray packed, scales, biases; };

Q4 load_q4(const Options& options) {
    qwen38::MlxSafetensors file(options.shard);
    return {file.tensor(options.tensor + ".weight"), file.tensor(options.tensor + ".scales"),
            file.tensor(options.tensor + ".biases")};
}

std::vector<float> make_input(int rows, int k) {
    std::mt19937 generator(38);
    std::normal_distribution<float> distribution(0.0F, 0.35F);
    std::vector<float> values(static_cast<std::size_t>(rows) * k);
    std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
    return values;
}

MlxArray from_u32(const std::vector<std::uint32_t>& values, std::span<const int> shape) {
    std::vector<std::int32_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](std::uint32_t v) { return std::bit_cast<std::int32_t>(v); });
    return MlxArray::from_int32(bits, shape).astype(MLX_UINT32);
}

struct Quality { bool finite = true; double cosine = 0, max_abs = 0, rmse = 0; };

Quality compare(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("quality shape mismatch");
    Quality result;
    double dot = 0, aa = 0, bb = 0, square_error = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result.finite = result.finite && std::isfinite(a[i]) && std::isfinite(b[i]);
        const double error = static_cast<double>(a[i]) - b[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        square_error += error * error;
        dot += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    result.cosine = aa == 0 && bb == 0 ? 1.0 : dot / std::sqrt(aa * bb);
    result.rmse = std::sqrt(square_error / a.size());
    return result;
}

class DirectQ4 final {
public:
    DirectQ4(int rows, int n, int k)
        : rows_(rows), n_(n), k_(k), groups_(k / group_size),
          source_(std::string(qwen38::w8a8_probe_metal::mpp_header) +
                  std::string(qwen38::direct_q4_metal::header)),
          quantizer_("qwen38_direct_q4_quantize", quant_input_names_, quant_output_names_,
                     qwen38::direct_q4_metal::quantize),
          mpp_("qwen38_direct_q4_affine", mpp_input_names_, "y",
               qwen38::direct_q4_metal::fused, source_) {
        if (k % group_size != 0 || n % 128 != 0 || (rows != 32 && rows % 128 != 0))
            throw std::runtime_error("unsupported direct Q4 geometry");
    }

    MlxArray apply(const MlxArray& x, const Q4& q4) const {
        const std::array<const MlxArray*, 1> qi{&x};
        const std::array<qwen38::MlxMetalOutputSpec, 3> qo{{
            {.shape = {rows_, k_}, .dtype = MLX_INT8},
            {.shape = {rows_, groups_}, .dtype = MLX_FLOAT32},
            {.shape = {rows_, groups_}, .dtype = MLX_FLOAT32}}};
        const std::array<int, 3> qgrid{rows_ * groups_ * 64, 1, 1}, qgroup{64, 1, 1};
        const std::array<qwen38::MlxMetalIntTemplate, 2> qargs{{{"K", k_}, {"G", groups_}}};
        auto q = quantizer_.apply(qi, qo, qgrid, qgroup, {}, qargs);
        const std::array<const MlxArray*, 6> mi{&q[0], &q4.packed, &q[1], &q[2],
                                                &q4.scales, &q4.biases};
        const std::array<qwen38::MlxMetalOutputSpec, 1> mo{{
            {.shape = {rows_, n_}, .dtype = MLX_FLOAT16}}};
        const int bm = rows_ == 32 ? 32 : 128;
        const int threads = rows_ == 32 ? 128 : 512;
        const int tiles_m = rows_ / bm, tiles_n = n_ / 128;
        const std::array<int, 3> grid{tiles_n * threads, tiles_m, 1}, group{threads, 1, 1};
        const std::array<qwen38::MlxMetalIntTemplate, 7> args{{
            {"M", rows_}, {"N", n_}, {"K", k_}, {"BM", bm},
            {"WM", rows_ == 32 ? 1 : 4}, {"TILES_M", tiles_m}, {"TILES_N", tiles_n}}};
        return mpp_.apply(mi, mo, grid, group, {}, args)[0].astype(MLX_BFLOAT16);
    }

private:
    int rows_, n_, k_, groups_;
    std::string source_;
    inline static constexpr const char* quant_input_names_[]{"x"};
    inline static constexpr const char* quant_output_names_[]{"aq", "scale", "xsum"};
    inline static constexpr const char* mpp_input_names_[]{
        "aq", "packed", "scale", "xsum", "wscale", "wbias"};
    qwen38::MlxMetalKernel quantizer_;
    qwen38::MlxMetalKernel mpp_;
};

float integer_proof(int rows, int n, int k) {
    std::vector<std::int32_t> a(static_cast<std::size_t>(rows) * k);
    for (int m = 0; m < rows; ++m) for (int x = 0; x < k; ++x)
        a[static_cast<std::size_t>(m) * k + x] = (m + x) % 7 - 3;
    std::vector<std::uint32_t> packed(static_cast<std::size_t>(n) * k / 8, 0);
    for (int output = 0; output < n; ++output) for (int x = 0; x < k; ++x) {
        const std::size_t linear = static_cast<std::size_t>(output) * k + x;
        packed[linear / 8] |= static_cast<std::uint32_t>((output + x) % 16) << ((linear & 7) * 4);
    }
    const std::array<int, 2> ashape{rows, k}, pshape{n, k / 8}, wshape{n, k};
    MlxArray aq = MlxArray::from_int32(a, ashape).astype(MLX_INT8);
    MlxArray words = from_u32(packed, pshape);
    const char* unpack_names[]{"packed"};
    qwen38::MlxMetalKernel unpack("qwen38_direct_q4_unpack_proof", unpack_names, "w",
                                  qwen38::direct_q4_metal::unpack);
    const std::array<const MlxArray*, 1> ui{&words};
    const std::array<qwen38::MlxMetalOutputSpec, 1> uo{{
        {.shape = {n, k}, .dtype = MLX_INT8}}};
    const int count = n * k;
    const std::array<int, 3> ugrid{((count + 255) / 256) * 256, 1, 1}, ugroup{256, 1, 1};
    const std::array<qwen38::MlxMetalIntTemplate, 1> uargs{{{"COUNT", count}}};
    auto unpacked = unpack.apply(ui, uo, ugrid, ugroup, {}, uargs);
    MlxArray w = std::move(unpacked[0]);
    const char* names[]{"aq", "w"};
    qwen38::MlxMetalKernel mpp("qwen38_direct_q4_mpp_proof", names, "y",
                               qwen38::w8a8_probe_metal::mpp_int32,
                               qwen38::w8a8_probe_metal::mpp_header);
    const std::array<const MlxArray*, 2> inputs{&aq, &w};
    const std::array<qwen38::MlxMetalOutputSpec, 1> outputs{{
        {.shape = {rows, n}, .dtype = MLX_INT32}}};
    const int bm = rows == 32 ? 32 : 128, threads = rows == 32 ? 128 : 512;
    const std::array<int, 3> grid{(n / 128) * threads, rows / bm, 1}, group{threads, 1, 1};
    const std::array<qwen38::MlxMetalIntTemplate, 8> args{{
        {"M", rows}, {"N", n}, {"K", k}, {"BM", bm}, {"WM", rows == 32 ? 1 : 4},
        {"SWIZZLE", 0}, {"TILES_M", rows / bm}, {"TILES_N", n / 128}}};
    const auto actual = mpp.apply(inputs, outputs, grid, group, {}, args)[0]
                            .astype(MLX_FLOAT32).to_float32();
    float max_abs = 0;
    for (int m = 0; m < rows; ++m) for (int output = 0; output < n; ++output) {
        std::int32_t expected = 0;
        for (int x = 0; x < k; ++x) expected += ((m + x) % 7 - 3) * ((output + x) % 16);
        max_abs = std::max(max_abs, std::abs(actual[static_cast<std::size_t>(m) * n + output] - expected));
    }
    return max_abs;
}

struct Semantic { double zero_max_abs, bias_direct_max_abs, bias_q4_max_abs; };

Semantic semantic_checks() {
    constexpr int rows = 32, n = 128, k = 64;
    const std::array<int, 2> xshape{rows, k}, pshape{n, k / 8}, gshape{n, 1};
    std::vector<float> zeros(static_cast<std::size_t>(rows) * k, 0);
    std::vector<float> signed_x(zeros.size());
    for (std::size_t i = 0; i < signed_x.size(); ++i)
        signed_x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0F;
    std::vector<std::uint32_t> words(static_cast<std::size_t>(n) * k / 8, 0);
    std::vector<float> scales(n, 0.125F), biases(n);
    for (int output = 0; output < n; ++output)
        biases[output] = static_cast<float>(output % 9 - 4) / 32.0F;
    Q4 q4{from_u32(words, pshape), MlxArray::from_float32(scales, gshape).astype(MLX_BFLOAT16),
          MlxArray::from_float32(biases, gshape).astype(MLX_BFLOAT16)};
    MlxArray zero = MlxArray::from_float32(zeros, xshape).astype(MLX_BFLOAT16);
    MlxArray x = MlxArray::from_float32(signed_x, xshape).astype(MLX_BFLOAT16);
    DirectQ4 direct(rows, n, k);
    const auto zero_out = direct.apply(zero, q4).astype(MLX_FLOAT32).to_float32();
    const auto direct_out = direct.apply(x, q4).astype(MLX_FLOAT32).to_float32();
    const auto q4_out = MlxArray::quantized_matmul(x, q4.packed, q4.scales, q4.biases, 64, 4)
                            .astype(MLX_FLOAT32).to_float32();
    double zero_max = 0, direct_max = 0, q4_max = 0;
    for (int m = 0; m < rows; ++m) {
        float sum = 0;
        for (int column = 0; column < k; ++column) sum += signed_x[static_cast<std::size_t>(m) * k + column];
        for (int output = 0; output < n; ++output) {
            const std::size_t index = static_cast<std::size_t>(m) * n + output;
            const float expected = sum * biases[output];
            zero_max = std::max(zero_max, std::abs(static_cast<double>(zero_out[index])));
            direct_max = std::max(direct_max, std::abs(static_cast<double>(direct_out[index] - expected)));
            q4_max = std::max(q4_max, std::abs(static_cast<double>(q4_out[index] - expected)));
        }
    }
    if (zero_max != 0 || direct_max > 0.0625 || q4_max > 0.0625)
        throw std::runtime_error("affine semantic edge check failed");
    return {zero_max, direct_max, q4_max};
}

struct Stats { double mean, median, minimum, maximum; };
Stats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return {std::accumulate(values.begin(), values.end(), 0.0) / values.size(), values[middle],
            values.front(), values.back()};
}
template <class F> double measure(F&& function) {
    const auto start = Clock::now();
    MlxArray output = function(); output.eval();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void print_samples(const std::vector<double>& samples) {
    std::cout << '[';
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << samples[i];
    }
    std::cout << ']';
}
} // namespace

int main(int argc, char** argv) try {
    constexpr int rows = 512, n = 6144, k = 2560;
    const Options options = parse_options(argc, argv);
    const float proof32 = integer_proof(32, n, k);
    const float proof128 = integer_proof(128, n, k);
    if (proof32 != 0 || proof128 != 0) throw std::runtime_error("unsigned-Q4 MPP integer proof failed");
    const Semantic semantic = semantic_checks();
    Q4 q4 = load_q4(options);
    if (q4.packed.shape() != std::vector<int>({n, k / 8}) ||
        q4.scales.shape() != std::vector<int>({n, k / 64}) ||
        q4.biases.shape() != std::vector<int>({n, k / 64}))
        throw std::runtime_error("expected affine-Q4 tuple [6144,2560], group64");
    const std::array<int, 2> xshape{rows, k};
    MlxArray x = MlxArray::from_float32(make_input(rows, k), xshape).astype(MLX_BFLOAT16);
    DirectQ4 direct(rows, n, k);
    const auto run_direct = [&] { return direct.apply(x, q4); };
    const auto run_q4 = [&] { return MlxArray::quantized_matmul(x, q4.packed, q4.scales,
                                                                q4.biases, 64, 4); };
    MlxArray direct_out = run_direct(), q4_out = run_q4();
    const std::array<const MlxArray*, 2> initial{&direct_out, &q4_out};
    MlxArray::eval_all(initial);
    const Quality quality = compare(direct_out.astype(MLX_FLOAT32).to_float32(),
                                    q4_out.astype(MLX_FLOAT32).to_float32());
    if (!quality.finite || quality.cosine < 0.99) throw std::runtime_error("real output quality gate failed");
    for (int i = 0; i < options.warmups; ++i) {
        if (i % 2 == 0) { run_direct().eval(); run_q4().eval(); }
        else { run_q4().eval(); run_direct().eval(); }
    }
    std::vector<double> direct_samples, q4_samples;
    for (int i = 0; i < options.iterations; ++i) {
        if (i % 2 == 0) {
            direct_samples.push_back(measure(run_direct)); q4_samples.push_back(measure(run_q4));
        } else {
            q4_samples.push_back(measure(run_q4)); direct_samples.push_back(measure(run_direct));
        }
    }
    const Stats ds = summarize(direct_samples), qs = summarize(q4_samples);
    const std::size_t scratch = static_cast<std::size_t>(rows) * k +
                                2 * static_cast<std::size_t>(rows) * (k / 64) * sizeof(float);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"mode\":\"real-layer0-in-proj-z-direct-affine-q4\",\"rows\":" << rows
              << ",\"n\":" << n << ",\"k\":" << k
              << ",\"activation_source\":\"synthetic-normal-seed-38-bf16\""
              << ",\"q4_layout\":\"unsigned-0-to-15-low-to-high-nibbles\""
              << ",\"warmups\":" << options.warmups << ",\"iterations\":" << options.iterations
              << ",\"integer_proof\":{\"m32_max_abs\":" << proof32
              << ",\"m128_max_abs\":" << proof128 << "}"
              << ",\"semantic\":{\"zero_activation_max_abs\":" << semantic.zero_max_abs
              << ",\"zero_codes_bias_direct_vs_expected_max_abs\":" << semantic.bias_direct_max_abs
              << ",\"zero_codes_bias_q4_vs_expected_max_abs\":" << semantic.bias_q4_max_abs << "}"
              << ",\"quality\":{\"finite\":" << (quality.finite ? "true" : "false")
              << ",\"cosine\":" << quality.cosine << ",\"max_abs\":" << quality.max_abs
              << ",\"rmse\":" << quality.rmse << "}"
              << ",\"timing_ms\":{\"direct_mean\":" << ds.mean << ",\"direct_median\":" << ds.median
              << ",\"direct_min\":" << ds.minimum << ",\"direct_max\":" << ds.maximum
              << ",\"q4_mean\":" << qs.mean << ",\"q4_median\":" << qs.median
              << ",\"q4_min\":" << qs.minimum << ",\"q4_max\":" << qs.maximum
              << ",\"mean_speedup\":" << qs.mean / ds.mean
              << ",\"median_speedup\":" << qs.median / ds.median << "}"
              << ",\"direct_samples_ms\":"; print_samples(direct_samples);
    std::cout << ",\"q4_samples_ms\":"; print_samples(q4_samples);
    std::cout << ",\"scratch_bytes\":" << scratch << ",\"scratch_mib\":"
              << static_cast<double>(scratch) / (1024.0 * 1024.0)
              << ",\"persistent_w8_bytes\":0,\"output_dtype\":\"bfloat16\"}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "qwen38-direct-q4-probe: " << error.what() << '\n';
    return 1;
}
