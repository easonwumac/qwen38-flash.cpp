#include "qwen38/decoder_layer.hpp"

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
using qwen38::DecoderLayer;
using qwen38::DecoderLayerState;
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
        const MlxArray& input, const std::string_view name, const MlxArray& weight,
        const MlxArray& scales, const MlxArray& biases, const int group_size,
        const int bits) override {
        const std::string key(name);
        if (!enabled_.contains(key.substr(key.rfind('.') + 1))) return std::nullopt;
        auto found = projections_.find(key);
        if (found == projections_.end()) {
            auto value = std::make_unique<qwen38::devtools::W8A8Projection>(
                weight, scales, biases, group_size, bits);
            sidecar_bytes_ += value->sidecar_bytes();
            found = projections_.emplace(key, std::move(value)).first;
        }
        return found->second->apply(input);
    }
    [[nodiscard]] std::size_t sidecar_bytes() const noexcept { return sidecar_bytes_; }
private:
    std::unordered_set<std::string> enabled_;
    std::unordered_map<std::string, std::unique_ptr<qwen38::devtools::W8A8Projection>> projections_;
    std::size_t sidecar_bytes_ = 0;
};

struct Input { MlxArray stream; std::vector<std::uint32_t> tokens; };

Input make_input(qwen38::MlxTensorStore& tensors, const bool second) {
    const auto& config = tensors.manifest().config();
    if (config.vocabulary_size <= 256 || config.vocabulary_size >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("unsupported vocabulary size");
    std::vector<std::int32_t> ids_data(rows);
    std::vector<std::uint32_t> tokens(rows);
    const std::uint64_t offset = second ? 104729U : 0U;
    for (int row = 0; row < rows; ++row) {
        std::uint64_t token = (9419U + static_cast<std::uint64_t>(row) * 7919U + offset) %
                              config.vocabulary_size;
        if (token < 100U) token += 100U;
        ids_data[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(token);
        tokens[static_cast<std::size_t>(row)] = static_cast<std::uint32_t>(token);
    }
    const std::array<int, 1> ids_shape{rows};
    MlxArray ids = MlxArray::from_int32(ids_data, ids_shape);
    MlxArray embedding = MlxArray::dequantize(
        MlxArray::take_axis(tensors.tensor("language_model.model.embed_tokens.weight"), ids, 0),
        MlxArray::take_axis(tensors.tensor("language_model.model.embed_tokens.scales"), ids, 0),
        MlxArray::take_axis(tensors.tensor("language_model.model.embed_tokens.biases"), ids, 0),
        static_cast<int>(config.quantization_group_size),
        static_cast<int>(config.quantization_bits));
    const std::array<int, 3> shape{1, rows, static_cast<int>(config.hidden_size)};
    return {qwen38::HyperConnection::initialize_stream(
                embedding.reshape(shape), config.hyper_connection_count), std::move(tokens)};
}

struct Execution { MlxArray first, second, convolution, recurrent; };

Execution execute(DecoderLayer& layer, const Input& first, const Input& second) {
    DecoderLayerState state;
    MlxArray first_output = layer.forward_prefill(first.stream.share(), first.tokens, state);
    MlxArray second_output = layer.forward_prefill(second.stream.share(), second.tokens, state);
    const std::array<const MlxArray*, 4> values{
        &first_output, &second_output, &state.linear_attention.convolution,
        &state.linear_attention.recurrent};
    MlxArray::eval_all(values);
    return {std::move(first_output), std::move(second_output),
            std::move(state.linear_attention.convolution),
            std::move(state.linear_attention.recurrent)};
}

struct Quality { double cosine, max_abs, rmse; bool finite; };
Quality compare(const MlxArray& candidate, const MlxArray& reference) {
    const auto a = candidate.astype(MLX_FLOAT32).to_float32();
    const auto b = reference.astype(MLX_FLOAT32).to_float32();
    if (a.size() != b.size()) throw std::runtime_error("decoder A/B shape mismatch");
    double dot = 0.0, aa = 0.0, bb = 0.0, error2 = 0.0, max_abs = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
        finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]);
        dot += static_cast<double>(a[i]) * b[i]; aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
        const double error = static_cast<double>(a[i]) - b[i];
        error2 += error * error; max_abs = std::max(max_abs, std::abs(error));
    }
    return {dot / std::sqrt(aa * bb), max_abs,
            std::sqrt(error2 / static_cast<double>(a.size())), finite};
}

double timed(DecoderLayer& layer, const Input& first, const Input& second) {
    const auto start = Clock::now();
    static_cast<void>(execute(layer, first, second));
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end()); return values[values.size() / 2];
}
void print_quality(const char* name, const Quality& q) {
    std::cout << '"' << name << "\":{\"finite\":" << (q.finite ? "true" : "false")
              << ",\"cosine\":" << q.cosine << ",\"max_abs\":" << q.max_abs
              << ",\"rmse\":" << q.rmse << '}';
}
void print_samples(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("usage: qwen38-decoder-w8a8-ab MODEL_DIRECTORY");
    setenv("QWEN38_GDN_METAL_PREFILL", "1", 1);
    setenv("QWEN38_GROUPED_PREFILL", "1", 1);
    unsetenv("QWEN38_PROFILE_MOE_PREFILL");
    qwen38::ModelManifest manifest = qwen38::ModelManifest::load(argv[1]);
    const bool compact_qmeta = manifest.has_tensor(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.qmeta13_tags");
    if (compact_qmeta) {
        setenv("QWEN38_COMPACT_QMETA", "lossless13", 1);
        setenv("QWEN38_QMETA_PREFILL_CACHE", "1", 1);
        setenv("QWEN38_QMETA_PREFILL_DEFER_REDUCE", "1", 1);
        setenv("QWEN38_QMETA_PREFILL_DEFER_TEMPORARY", "1", 1);
    } else {
        unsetenv("QWEN38_COMPACT_QMETA"); unsetenv("QWEN38_QMETA_PREFILL_CACHE");
        unsetenv("QWEN38_QMETA_PREFILL_DEFER_REDUCE");
        unsetenv("QWEN38_QMETA_PREFILL_DEFER_TEMPORARY");
    }
    qwen38::MlxTensorStore tensors(std::move(manifest));
    Input first = make_input(tensors, false), second = make_input(tensors, true);
    const auto& config = tensors.manifest().config();
    DecoderLayer reference(tensors, 0, config);
    Execution expected = execute(reference, first, second);
    ProjectionBank noop_bank({});
    DecoderLayer noop(tensors, 0, config, &noop_bank);
    Execution noop_value = execute(noop, first, second);
    if (compare(noop_value.first, expected.first).max_abs != 0.0 ||
        compare(noop_value.second, expected.second).max_abs != 0.0 ||
        compare(noop_value.convolution, expected.convolution).max_abs != 0.0 ||
        compare(noop_value.recurrent, expected.recurrent).max_abs != 0.0)
        throw std::runtime_error("empty decoder hook changed default behavior");

    struct Arm { const char* name; std::unordered_set<std::string> banks; };
    const std::array<Arm, 2> arms{{
        {"z_out", {"in_proj_z", "out_proj"}},
        {"qkv_z_out", {"in_proj_qkv", "in_proj_z", "out_proj"}},
    }};
    std::cout << std::fixed << std::setprecision(6)
              << "{\"rows\":512,\"chunks_per_sample\":2,\"warmups\":3,\"iterations\":15"
              << ",\"profile\":{\"label\":\"exact-q4-base-top10-full-metadata\""
              << ",\"top_k\":" << config.experts_per_token
              << ",\"grouped_prefill\":true,\"gdn_metal_prefill\":true"
              << ",\"compact_qmeta_lossless13\":" << (compact_qmeta ? "true" : "false") << '}'
              << ",\"input\":\"real-embedding-streams;two-diverse-deterministic-token-id-cohorts;not-natural-text\""
              << ",\"noop_hook_exact\":true,\"arms\":[";
    for (std::size_t arm_index = 0; arm_index < arms.size(); ++arm_index) {
        const Arm& arm = arms[arm_index];
        ProjectionBank bank(arm.banks);
        DecoderLayer candidate(tensors, 0, config, &bank);
        static_cast<void>(execute(candidate, first, second));
        Execution actual = execute(candidate, first, second);
        const Quality first_q = compare(actual.first, expected.first);
        const Quality second_q = compare(actual.second, expected.second);
        const Quality conv_q = compare(actual.convolution, expected.convolution);
        const Quality recurrent_q = compare(actual.recurrent, expected.recurrent);
        if (std::string_view(arm.name) == "z_out" &&
            (conv_q.max_abs != 0.0 || recurrent_q.max_abs != 0.0))
            throw std::runtime_error("z+out changed decoder GDN state");
        for (int i = 0; i < warmups; ++i) {
            if (i % 2 == 0) { static_cast<void>(execute(candidate, first, second));
                static_cast<void>(execute(reference, first, second)); }
            else { static_cast<void>(execute(reference, first, second));
                static_cast<void>(execute(candidate, first, second)); }
        }
        std::vector<double> ref_samples, candidate_samples;
        for (int i = 0; i < iterations; ++i) {
            if (i % 2 == 0) { candidate_samples.push_back(timed(candidate, first, second));
                ref_samples.push_back(timed(reference, first, second)); }
            else { ref_samples.push_back(timed(reference, first, second));
                candidate_samples.push_back(timed(candidate, first, second)); }
        }
        if (arm_index != 0) std::cout << ',';
        const double ref_mean = mean(ref_samples), candidate_mean = mean(candidate_samples);
        std::cout << "{\"name\":\"" << arm.name << "\",\"sidecar_mib\":"
                  << static_cast<double>(bank.sidecar_bytes()) / (1024.0 * 1024.0)
                  << ",\"timing_scope\":\"complete-layer-sum-of-two-m512-chunks-including-gdn-states\""
                  << ",\"reference_mean_ms\":" << ref_mean
                  << ",\"reference_median_ms\":" << median(ref_samples)
                  << ",\"candidate_mean_ms\":" << candidate_mean
                  << ",\"candidate_median_ms\":" << median(candidate_samples)
                  << ",\"mean_speedup\":" << ref_mean / candidate_mean << ",\"quality\":{";
        print_quality("first_output", first_q); std::cout << ',';
        print_quality("second_output", second_q); std::cout << ',';
        print_quality("convolution", conv_q); std::cout << ',';
        print_quality("recurrent", recurrent_q);
        std::cout << "},\"reference_samples_ms\":"; print_samples(ref_samples);
        std::cout << ",\"candidate_samples_ms\":"; print_samples(candidate_samples);
        std::cout << '}';
    }
    std::cout << "],\"open_shards\":" << tensors.open_shard_count() << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "qwen38-decoder-w8a8-ab: " << error.what() << '\n'; return 1;
}
