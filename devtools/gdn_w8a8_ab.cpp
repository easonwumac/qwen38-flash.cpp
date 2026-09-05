#include "qwen38/gated_delta_net.hpp"
#include "qwen38/hyper_connection.hpp"

#include "w8a8_projection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
using qwen38::GatedDeltaNet;
using qwen38::GatedDeltaNetState;
using qwen38::MlxArray;
using Clock = std::chrono::steady_clock;

constexpr int rows = 512;
constexpr int warmups = 3;
constexpr int iterations = 15;

class ProjectionBank final : public qwen38::GatedDeltaNetProjectionHook {
public:
    explicit ProjectionBank(std::unordered_set<std::string> enabled)
        : enabled_(std::move(enabled)) {}

    std::optional<MlxArray> project_prefill(
        const MlxArray& input,
        const std::string_view name,
        const MlxArray& weight,
        const MlxArray& scales,
        const MlxArray& biases,
        const int group_size,
        const int bits) override {
        const std::string key(name);
        if (!enabled_.contains(key.substr(key.rfind('.') + 1))) return std::nullopt;
        auto found = projections_.find(key);
        if (found == projections_.end()) {
            auto projection = std::make_unique<qwen38::devtools::W8A8Projection>(
                weight, scales, biases, group_size, bits);
            sidecar_bytes_ += projection->sidecar_bytes();
            found = projections_.emplace(key, std::move(projection)).first;
        }
        return found->second->apply(input);
    }

    [[nodiscard]] std::size_t sidecar_bytes() const noexcept { return sidecar_bytes_; }

private:
    std::unordered_set<std::string> enabled_;
    std::unordered_map<std::string, std::unique_ptr<qwen38::devtools::W8A8Projection>> projections_;
    std::size_t sidecar_bytes_ = 0;
};

MlxArray mixed_input(qwen38::MlxTensorStore& tensors, const bool second_cohort) {
    const auto& config = tensors.manifest().config();
    if (config.vocabulary_size <= 256 || config.vocabulary_size >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("unsupported vocabulary size for GDN A/B inputs");
    std::vector<std::int32_t> ids_data(rows);
    const std::uint64_t cohort_offset = second_cohort ? 104729U : 0U;
    for (int row = 0; row < rows; ++row) {
        std::uint64_t token = (9419U + static_cast<std::uint64_t>(row) * 7919U +
                               cohort_offset) % config.vocabulary_size;
        if (token < 100U) token += 100U;
        ids_data[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(token);
    }
    const std::array<int, 1> ids_shape{rows};
    MlxArray ids = MlxArray::from_int32(ids_data, ids_shape);
    MlxArray embedding = MlxArray::dequantize(
        MlxArray::take_axis(tensors.tensor("language_model.model.embed_tokens.weight"), ids, 0),
        MlxArray::take_axis(tensors.tensor("language_model.model.embed_tokens.scales"), ids, 0),
        MlxArray::take_axis(tensors.tensor("language_model.model.embed_tokens.biases"), ids, 0),
        64, 4);
    const std::array<int, 3> embedding_shape{1, rows, static_cast<int>(config.hidden_size)};
    MlxArray stream = qwen38::HyperConnection::initialize_stream(
        embedding.reshape(embedding_shape), config.hyper_connection_count);
    qwen38::HyperConnection mixer(
        tensors, "language_model.model.layers.0.attn_hyper_connection",
        config.hidden_size, config.hyper_connection_count, config.quantization_bits,
        config.quantization_group_size, static_cast<float>(config.rms_norm_epsilon), true);
    return mixer.read(stream).mixed;
}

struct Execution { MlxArray first, second, convolution, recurrent; };

Execution execute(GatedDeltaNet& layer, const MlxArray& first, const MlxArray& second) {
    GatedDeltaNetState state;
    MlxArray first_output = layer.forward_prefill(first, state);
    MlxArray second_output = layer.forward_prefill(second, state);
    const std::array<const MlxArray*, 4> values{
        &first_output, &second_output, &state.convolution, &state.recurrent};
    MlxArray::eval_all(values);
    return {std::move(first_output), std::move(second_output),
            std::move(state.convolution), std::move(state.recurrent)};
}

struct Quality { double cosine, max_abs, rmse; bool finite; };

Quality compare(const MlxArray& candidate, const MlxArray& reference) {
    const std::vector<float> a = candidate.astype(MLX_FLOAT32).to_float32();
    const std::vector<float> b = reference.astype(MLX_FLOAT32).to_float32();
    if (a.size() != b.size()) throw std::runtime_error("A/B shape mismatch");
    double dot = 0.0, aa = 0.0, bb = 0.0, square = 0.0, max_abs = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
        finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]);
        dot += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
        const double error = static_cast<double>(a[i]) - b[i];
        max_abs = std::max(max_abs, std::abs(error));
        square += error * error;
    }
    return {dot / std::sqrt(aa * bb), max_abs,
            std::sqrt(square / static_cast<double>(a.size())), finite};
}

double timed(GatedDeltaNet& layer, const MlxArray& first, const MlxArray& second) {
    const auto start = Clock::now();
    static_cast<void>(execute(layer, first, second));
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double mean(const std::vector<double>& samples) {
    return std::accumulate(samples.begin(), samples.end(), 0.0) /
           static_cast<double>(samples.size());
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void print_samples(const std::vector<double>& samples) {
    std::cout << '[';
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << samples[i];
    }
    std::cout << ']';
}

void print_quality(const char* name, const Quality& quality) {
    std::cout << '"' << name << "\":{\"finite\":" << (quality.finite ? "true" : "false")
              << ",\"cosine\":" << quality.cosine << ",\"max_abs\":" << quality.max_abs
              << ",\"rmse\":" << quality.rmse << '}';
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("usage: qwen38-gdn-w8a8-ab MODEL_DIRECTORY");
    setenv("QWEN38_GDN_METAL_PREFILL", "1", 1);
    qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
    MlxArray first_input = mixed_input(tensors, false);
    MlxArray second_input = mixed_input(tensors, true);
    const auto& config = tensors.manifest().config();
    const std::string prefix = "language_model.model.layers.0.linear_attn";
    GatedDeltaNet reference(tensors, prefix, config);
    Execution expected = execute(reference, first_input, second_input);
    ProjectionBank noop_bank({});
    GatedDeltaNet noop(tensors, prefix, config, &noop_bank);
    Execution noop_result = execute(noop, first_input, second_input);
    const Quality noop_first = compare(noop_result.first, expected.first);
    const Quality noop_second = compare(noop_result.second, expected.second);
    const Quality noop_convolution = compare(noop_result.convolution, expected.convolution);
    const Quality noop_recurrent = compare(noop_result.recurrent, expected.recurrent);
    if (noop_first.max_abs != 0.0 || noop_second.max_abs != 0.0 ||
        noop_convolution.max_abs != 0.0 || noop_recurrent.max_abs != 0.0)
        throw std::runtime_error("no-op projection hook changed default GDN behavior");

    struct Arm { const char* name; std::unordered_set<std::string> banks; };
    const std::array<Arm, 3> arms{{
        {"z", {"in_proj_z"}},
        {"z_out", {"in_proj_z", "out_proj"}},
        {"qkv_z_out", {"in_proj_qkv", "in_proj_z", "out_proj"}},
    }};
    std::cout << std::fixed << std::setprecision(6)
              << "{\"rows\":512,\"chunks_per_sample\":2,\"warmups\":" << warmups
              << ",\"iterations\":" << iterations
              << ",\"input\":\"real-embedding-and-layer0-attention-hc;two-diverse-deterministic-token-id-cohorts;not-natural-text\""
              << ",\"noop_hook_exact\":true"
              << ",\"arms\":[";
    bool first_arm = true;
    for (const Arm& arm : arms) {
        ProjectionBank bank(arm.banks);
        GatedDeltaNet candidate(tensors, prefix, config, &bank);
        static_cast<void>(execute(candidate, first_input, second_input));
        Execution actual = execute(candidate, first_input, second_input);
        const Quality first_quality = compare(actual.first, expected.first);
        const Quality second_quality = compare(actual.second, expected.second);
        const Quality convolution_quality = compare(actual.convolution, expected.convolution);
        const Quality recurrent_quality = compare(actual.recurrent, expected.recurrent);
        if (std::string_view(arm.name) != "qkv_z_out" &&
            (convolution_quality.max_abs != 0.0 || recurrent_quality.max_abs != 0.0))
            throw std::runtime_error("selective z/out hook changed recurrent state");
        for (int i = 0; i < warmups; ++i) {
            if (i % 2 == 0) {
                static_cast<void>(execute(candidate, first_input, second_input));
                static_cast<void>(execute(reference, first_input, second_input));
            } else {
                static_cast<void>(execute(reference, first_input, second_input));
                static_cast<void>(execute(candidate, first_input, second_input));
            }
        }
        std::vector<double> reference_samples, candidate_samples;
        for (int i = 0; i < iterations; ++i) {
            if (i % 2 == 0) {
                candidate_samples.push_back(timed(candidate, first_input, second_input));
                reference_samples.push_back(timed(reference, first_input, second_input));
            } else {
                reference_samples.push_back(timed(reference, first_input, second_input));
                candidate_samples.push_back(timed(candidate, first_input, second_input));
            }
        }
        if (!first_arm) std::cout << ',';
        first_arm = false;
        const double reference_mean = mean(reference_samples);
        const double candidate_mean = mean(candidate_samples);
        std::cout << "{\"name\":\"" << arm.name << "\",\"sidecar_mib\":"
                  << static_cast<double>(bank.sidecar_bytes()) / (1024.0 * 1024.0)
                  << ",\"timing_scope\":\"sum-of-two-consecutive-m512-chunks-including-final-states\""
                  << ",\"reference_mean_ms\":" << reference_mean
                  << ",\"reference_median_ms\":" << median(reference_samples)
                  << ",\"candidate_mean_ms\":" << candidate_mean
                  << ",\"candidate_median_ms\":" << median(candidate_samples)
                  << ",\"speedup\":" << reference_mean / candidate_mean << ",\"quality\":{";
        print_quality("first_output", first_quality); std::cout << ',';
        print_quality("second_output", second_quality); std::cout << ',';
        print_quality("convolution", convolution_quality); std::cout << ',';
        print_quality("recurrent", recurrent_quality);
        std::cout << "},\"reference_samples_ms\":"; print_samples(reference_samples);
        std::cout << ",\"candidate_samples_ms\":"; print_samples(candidate_samples);
        std::cout << '}';
    }
    std::cout << "],\"open_shards\":" << tensors.open_shard_count() << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "qwen38-gdn-w8a8-ab: " << error.what() << '\n';
    return 1;
}
