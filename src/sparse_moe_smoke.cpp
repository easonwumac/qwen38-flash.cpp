#include "qwen38/hyper_connection.hpp"
#include "qwen38/runtime_profile.hpp"
#include "qwen38/sparse_moe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

qwen38::MlxArray make_input(qwen38::MlxTensorStore& tensors, const std::size_t rows) {
    const auto& config = tensors.manifest().config();
    std::vector<std::int32_t> token_values;
    token_values.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        token_values.push_back(static_cast<std::int32_t>(
            (9419 + row * 7919) % config.vocabulary_size));
    }
    const std::vector<int> token_shape{static_cast<int>(rows)};
    const auto ids = qwen38::MlxArray::from_int32(token_values, token_shape);
    const auto weight = tensors.tensor("language_model.model.embed_tokens.weight");
    const auto scales = tensors.tensor("language_model.model.embed_tokens.scales");
    const auto biases = tensors.tensor("language_model.model.embed_tokens.biases");
    auto embedding = qwen38::MlxArray::dequantize(
        qwen38::MlxArray::take_axis(weight, ids, 0),
        qwen38::MlxArray::take_axis(scales, ids, 0),
        qwen38::MlxArray::take_axis(biases, ids, 0),
        static_cast<int>(config.quantization_group_size),
        static_cast<int>(config.quantization_bits));
    const std::vector<int> embedding_shape{
        1, static_cast<int>(rows), static_cast<int>(config.hidden_size)};
    auto stream = qwen38::HyperConnection::initialize_stream(
        embedding.reshape(embedding_shape), config.hyper_connection_count);
    qwen38::HyperConnection mixer(
        tensors,
        "language_model.model.layers.0.mlp_hyper_connection",
        config.hidden_size,
        config.hyper_connection_count,
        config.quantization_bits,
        config.quantization_group_size,
        static_cast<float>(config.rms_norm_epsilon),
        true);
    return mixer.read(stream).mixed;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL_DIRECTORY [PREFILL_ROWS [components]]\n";
        return EXIT_FAILURE;
    }
    try {
        const std::size_t rows = argc >= 3 ? std::stoul(argv[2]) : 1;
        const bool profile_components = argc == 4 && std::string_view(argv[3]) == "components";
        if (argc == 4 && !profile_components) {
            throw std::runtime_error("the only supported profiling mode is 'components'");
        }
        if (profile_components && rows == 1) {
            throw std::runtime_error("component profiling requires PREFILL_ROWS greater than 1");
        }
        if (rows == 0 || rows > 512) {
            throw std::runtime_error("PREFILL_ROWS must be between 1 and 512");
        }
        if (rows > 1 && std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "MoE prefill tests must run through devtools/memory_guard.py");
        }
        qwen38::apply_runtime_profile("speed");
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        const auto& config = tensors.manifest().config();
        auto input = make_input(tensors, rows);
        qwen38::SparseMoe moe(
            tensors,
            "language_model.model.layers.0.mlp",
            config.expert_count,
            config.experts_per_token,
            config.quantization_bits,
            config.quantization_group_size,
            config.normalize_topk_probability);
        const auto selection = rows == 1
            ? moe.route_decode(input)
            : qwen38::RouterSelection{};
        std::vector<double> timings;
        std::vector<qwen38::MoePrefillTimings> component_timings;
        std::vector<float> values;
        const int iterations = profile_components ? 21 : 6;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            qwen38::MoePrefillTimings components;
            auto output = rows == 1
                ? moe.forward_decode(input)
                : profile_components
                    ? moe.forward_prefill_profiled(input, components)
                    : moe.forward_prefill(input);
            values = output.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
            if (profile_components) component_timings.push_back(components);
        }
        if (values.size() != rows * config.hidden_size ||
            !std::all_of(values.begin(), values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("MoE output is invalid");
        }
        const double checksum = std::accumulate(values.begin(), values.end(), 0.0);
        std::vector<double> warm(timings.begin() + 1, timings.end());
        std::sort(warm.begin(), warm.end());
        const auto component_median = [&component_timings](
                                          const auto qwen38::MoePrefillTimings::* field) {
            std::vector<double> values;
            values.reserve(component_timings.size() - 1);
            for (std::size_t index = 1; index < component_timings.size(); ++index) {
                values.push_back(component_timings[index].*field);
            }
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        };
        std::cout << "{\"rows\":" << rows << ",\"experts\":[";
        for (std::size_t index = 0; index < selection.experts.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << selection.experts[index];
        }
        std::cout << "],\"weights\":[";
        for (std::size_t index = 0; index < selection.weights.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << selection.weights[index];
        }
        std::cout << std::setprecision(10)
                  << "],\"checksum\":" << checksum << ",\"cold_ms\":" << timings.front()
                  << ",\"warm_median_ms\":" << warm[warm.size() / 2]
                  << ",\"prefill_tps\":"
                  << 1000.0 * static_cast<double>(rows) / warm[warm.size() / 2]
                  << ",\"open_shards\":" << tensors.open_shard_count();
        if (profile_components) {
            std::cout << ",\"components_ms\":{"
                      << "\"routing\":"
                      << component_median(&qwen38::MoePrefillTimings::routing_ms)
                      << ",\"gate_up\":"
                      << component_median(&qwen38::MoePrefillTimings::gate_up_ms)
                      << ",\"down_reduce\":"
                      << component_median(&qwen38::MoePrefillTimings::down_reduce_ms)
                      << ",\"shared_expert\":"
                      << component_median(&qwen38::MoePrefillTimings::shared_expert_ms)
                      << ",\"merge\":"
                      << component_median(&qwen38::MoePrefillTimings::merge_ms)
                      << '}';
        }
        std::cout << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-sparse-moe-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
