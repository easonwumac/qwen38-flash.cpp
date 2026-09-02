#include "qwen38/hyper_connection.hpp"
#include "qwen38/runtime_profile.hpp"
#include "qwen38/sparse_moe.hpp"

#include <algorithm>
#include <bit>
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

qwen38::MlxArray make_input(
    qwen38::MlxTensorStore& tensors,
    const std::size_t rows,
    const std::size_t layer) {
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
        "language_model.model.layers." + std::to_string(layer) + ".mlp_hyper_connection",
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
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL_DIRECTORY [ROWS [components|verify-components|layer=N [LAYER]]]\n";
        return EXIT_FAILURE;
    }
    try {
        const std::size_t rows = argc >= 3 ? std::stoul(argv[2]) : 1;
        const std::string_view mode = argc >= 4 ? std::string_view(argv[3]) : std::string_view{};
        const bool profile_components = mode == "components";
        const bool profile_verify = mode == "verify-components";
        const bool layer_mode = mode.starts_with("layer=");
        const bool explicit_profile_layer = argc == 5;
        const std::size_t layer = explicit_profile_layer
            ? std::stoul(argv[4])
            : layer_mode ? std::stoul(std::string(mode.substr(6))) : 0;
        if (argc >= 4 && !profile_components && !profile_verify && !layer_mode) {
            throw std::runtime_error(
                "mode must be 'components', 'verify-components', or 'layer=N'");
        }
        if (explicit_profile_layer && !profile_components && !profile_verify) {
            throw std::runtime_error("explicit LAYER requires a component profiling mode");
        }
        if ((profile_components || profile_verify) && rows == 1) {
            throw std::runtime_error("component profiling requires ROWS greater than 1");
        }
        if (profile_verify && rows > 5) {
            throw std::runtime_error("verify component profiling requires ROWS <= 5");
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
        if (layer >= config.layer_count) {
            throw std::runtime_error("layer index is out of range");
        }
        auto input = make_input(tensors, rows, layer);
        const std::string prefix =
            "language_model.model.layers." + std::to_string(layer) + ".mlp";
        qwen38::SparseMoe moe(
            tensors,
            prefix,
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
        std::vector<qwen38::MoeVerifyTimings> verify_timings;
        std::vector<float> values;
        const int iterations = profile_components || profile_verify ? 21 : 6;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            qwen38::MoePrefillTimings components;
            qwen38::MoeVerifyTimings verify_components;
            auto output = rows == 1
                ? moe.forward_decode(input)
                : profile_verify
                    ? moe.forward_verify_profiled(input, verify_components)
                    : profile_components
                        ? moe.forward_prefill_profiled(input, components)
                        : moe.forward_prefill(input);
            values = output.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
            if (profile_components) component_timings.push_back(components);
            if (profile_verify) verify_timings.push_back(verify_components);
        }
        if (values.size() != rows * config.hidden_size ||
            !std::all_of(values.begin(), values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("MoE output is invalid");
        }
        const double checksum = std::accumulate(values.begin(), values.end(), 0.0);
        std::uint64_t bit_hash = 1469598103934665603ULL;
        for (const float value : values) {
            bit_hash ^= std::bit_cast<std::uint32_t>(value);
            bit_hash *= 1099511628211ULL;
        }
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
        const auto verify_median = [&verify_timings](
                                       const auto qwen38::MoeVerifyTimings::* field) {
            std::vector<double> values;
            values.reserve(verify_timings.size() - 1);
            for (std::size_t index = 1; index < verify_timings.size(); ++index) {
                values.push_back(verify_timings[index].*field);
            }
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        };
        std::cout << "{\"rows\":" << rows << ",\"layer\":" << layer
                  << ",\"experts\":[";
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
                  << "],\"checksum\":" << checksum
                  << ",\"bit_hash\":" << bit_hash
                  << ",\"cold_ms\":" << timings.front()
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
                      << ",\"gate_qmm\":"
                      << component_median(&qwen38::MoePrefillTimings::gate_qmm_ms)
                      << ",\"up_qmm\":"
                      << component_median(&qwen38::MoePrefillTimings::up_qmm_ms)
                      << ",\"swiglu\":"
                      << component_median(&qwen38::MoePrefillTimings::swiglu_ms)
                      << ",\"down_reduce\":"
                      << component_median(&qwen38::MoePrefillTimings::down_reduce_ms)
                      << ",\"down_qmm\":"
                      << component_median(&qwen38::MoePrefillTimings::down_qmm_ms)
                      << ",\"route_reduce\":"
                      << component_median(&qwen38::MoePrefillTimings::route_reduce_ms)
                      << ",\"shared_expert\":"
                      << component_median(&qwen38::MoePrefillTimings::shared_expert_ms)
                      << ",\"merge\":"
                      << component_median(&qwen38::MoePrefillTimings::merge_ms)
                      << '}';
        } else if (profile_verify) {
            std::cout << ",\"verify_components_ms\":{\"routing\":"
                      << verify_median(&qwen38::MoeVerifyTimings::routing_ms)
                      << ",\"gate_up\":"
                      << verify_median(&qwen38::MoeVerifyTimings::gate_up_ms)
                      << ",\"down\":"
                      << verify_median(&qwen38::MoeVerifyTimings::down_ms)
                      << ",\"shared_expert\":"
                      << verify_median(&qwen38::MoeVerifyTimings::shared_expert_ms)
                      << ",\"merge\":"
                      << verify_median(&qwen38::MoeVerifyTimings::merge_ms)
                      << '}';
        }
        std::cout << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-sparse-moe-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
