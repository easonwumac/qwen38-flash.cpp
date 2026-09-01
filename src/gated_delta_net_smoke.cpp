#include "qwen38/gated_delta_net.hpp"
#include "qwen38/hyper_connection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

qwen38::MlxArray make_input(
    qwen38::MlxTensorStore& tensors,
    const std::int32_t token) {
    const std::vector<std::int32_t> token_values{token};
    const std::vector<int> token_shape{1};
    const auto ids = qwen38::MlxArray::from_int32(token_values, token_shape);
    const auto weight = tensors.tensor("language_model.model.embed_tokens.weight");
    const auto scales = tensors.tensor("language_model.model.embed_tokens.scales");
    const auto biases = tensors.tensor("language_model.model.embed_tokens.biases");
    const auto& config = tensors.manifest().config();
    auto embedding = qwen38::MlxArray::dequantize(
        qwen38::MlxArray::take_axis(weight, ids, 0),
        qwen38::MlxArray::take_axis(scales, ids, 0),
        qwen38::MlxArray::take_axis(biases, ids, 0),
        static_cast<int>(config.quantization_group_size),
        static_cast<int>(config.quantization_bits));
    const std::vector<int> embedding_shape{1, 1, static_cast<int>(config.hidden_size)};
    auto stream = qwen38::HyperConnection::initialize_stream(
        embedding.reshape(embedding_shape), config.hyper_connection_count);
    qwen38::HyperConnection mixer(
        tensors,
        "language_model.model.layers.0.attn_hyper_connection",
        config.hidden_size,
        config.hyper_connection_count,
        config.quantization_bits,
        config.quantization_group_size,
        static_cast<float>(config.rms_norm_epsilon),
        true);
    return mixer.read(stream).mixed;
}

float maximum_absolute_error(
    const std::vector<float>& reference,
    const std::vector<float>& candidate) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("GatedDeltaNet comparison size mismatch");
    }
    float result = 0.0F;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        result = std::max(result, std::abs(reference[index] - candidate[index]));
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        auto first_input = make_input(tensors, 9419);
        auto second_input = make_input(tensors, 11);
        qwen38::GatedDeltaNet layer(
            tensors,
            "language_model.model.layers.0.linear_attn",
            tensors.manifest().config());
        if (std::getenv("QWEN38_GDN_COMPARE") != nullptr) {
            unsetenv("QWEN38_GDN_PREWORK");
            unsetenv("QWEN38_GDN_NORM_GATE");
            qwen38::GatedDeltaNetState reference_state;
            auto reference_first = layer.forward_decode(first_input, reference_state)
                .astype(MLX_FLOAT32).to_float32();
            auto reference_second = layer.forward_decode(second_input, reference_state)
                .astype(MLX_FLOAT32).to_float32();
            auto reference_convolution = reference_state.convolution.astype(
                MLX_FLOAT32).to_float32();
            auto reference_recurrent = reference_state.recurrent.astype(
                MLX_FLOAT32).to_float32();

            setenv("QWEN38_GDN_PREWORK", "1", 1);
            setenv("QWEN38_GDN_NORM_GATE", "1", 1);
            qwen38::GatedDeltaNetState candidate_state;
            auto candidate_first = layer.forward_decode(first_input, candidate_state)
                .astype(MLX_FLOAT32).to_float32();
            auto candidate_second = layer.forward_decode(second_input, candidate_state)
                .astype(MLX_FLOAT32).to_float32();
            auto candidate_convolution = candidate_state.convolution.astype(
                MLX_FLOAT32).to_float32();
            auto candidate_recurrent = candidate_state.recurrent.astype(
                MLX_FLOAT32).to_float32();
            std::cout << "{\"first_max_abs\":"
                      << maximum_absolute_error(reference_first, candidate_first)
                      << ",\"second_max_abs\":"
                      << maximum_absolute_error(reference_second, candidate_second)
                      << ",\"convolution_max_abs\":"
                      << maximum_absolute_error(reference_convolution, candidate_convolution)
                      << ",\"recurrent_max_abs\":"
                      << maximum_absolute_error(reference_recurrent, candidate_recurrent)
                      << "}\n";
            return EXIT_SUCCESS;
        }
        std::vector<double> timings;
        std::vector<float> first_values;
        std::vector<float> second_values;
        std::vector<float> recurrent_values;
        for (int iteration = 0; iteration < 6; ++iteration) {
            qwen38::GatedDeltaNetState state;
            auto first_output = layer.forward_decode(first_input, state);
            first_values = first_output.astype(MLX_FLOAT32).to_float32();
            const auto started = std::chrono::steady_clock::now();
            auto second_output = layer.forward_decode(second_input, state);
            second_values = second_output.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
            recurrent_values = state.recurrent.astype(MLX_FLOAT32).to_float32();
        }
        const auto& config = tensors.manifest().config();
        if (first_values.size() != config.hidden_size || second_values.size() != config.hidden_size ||
            !std::all_of(second_values.begin(), second_values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("GatedDeltaNet output is invalid");
        }
        const double first_checksum = std::accumulate(first_values.begin(), first_values.end(), 0.0);
        const double second_checksum = std::accumulate(second_values.begin(), second_values.end(), 0.0);
        const double state_checksum = std::accumulate(
            recurrent_values.begin(), recurrent_values.end(), 0.0);
        std::vector<double> warm(timings.begin() + 1, timings.end());
        std::sort(warm.begin(), warm.end());
        std::cout << "{\"first_checksum\":" << first_checksum
                  << ",\"second_checksum\":" << second_checksum
                  << ",\"state_checksum\":" << state_checksum
                  << ",\"second_cold_ms\":" << timings.front()
                  << ",\"warm_median_ms\":" << warm[warm.size() / 2]
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-gated-delta-net-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
