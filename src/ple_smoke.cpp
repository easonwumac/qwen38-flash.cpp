#include "qwen38/hyper_connection.hpp"
#include "qwen38/ple.hpp"

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

qwen38::MlxArray make_stream(qwen38::MlxTensorStore& tensors, const std::int32_t token) {
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
    const std::vector<int> shape{1, 1, static_cast<int>(config.hidden_size)};
    return qwen38::HyperConnection::initialize_stream(
        embedding.reshape(shape), config.hyper_connection_count);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        auto first_stream = make_stream(tensors, 9419);
        auto second_stream = make_stream(tensors, 11);
        qwen38::Ple ple(
            tensors,
            "language_model.model.layers.1.ple",
            tensors.manifest().config());
        std::vector<double> timings;
        std::vector<float> first_values;
        std::vector<float> second_values;
        for (int iteration = 0; iteration < 6; ++iteration) {
            qwen38::PleState state;
            auto first = ple.forward_decode(first_stream, 9419, state);
            first_values = first.astype(MLX_FLOAT32).to_float32();
            const auto started = std::chrono::steady_clock::now();
            auto second = ple.forward_decode(second_stream, 11, state);
            second_values = second.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
        const std::size_t width = tensors.manifest().config().hidden_size *
            tensors.manifest().config().hyper_connection_count;
        if (first_values.size() != width || second_values.size() != width ||
            !std::all_of(second_values.begin(), second_values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("PLE output is invalid");
        }
        const double first_checksum = std::accumulate(first_values.begin(), first_values.end(), 0.0);
        const double second_checksum = std::accumulate(second_values.begin(), second_values.end(), 0.0);
        const double first_l1 = std::accumulate(
            first_values.begin(), first_values.end(), 0.0,
            [](const double sum, const float value) { return sum + std::abs(value); });
        const double second_l1 = std::accumulate(
            second_values.begin(), second_values.end(), 0.0,
            [](const double sum, const float value) { return sum + std::abs(value); });
        std::vector<double> warm(timings.begin() + 1, timings.end());
        std::sort(warm.begin(), warm.end());
        std::cout << "{\"first_checksum\":" << first_checksum
                  << ",\"second_checksum\":" << second_checksum
                  << ",\"first_l1\":" << first_l1
                  << ",\"second_l1\":" << second_l1
                  << ",\"second_cold_ms\":" << timings.front()
                  << ",\"warm_median_ms\":" << warm[warm.size() / 2]
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-ple-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
