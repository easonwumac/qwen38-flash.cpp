#include "qwen38/hyper_connection.hpp"
#include "qwen38/sparse_moe.hpp"

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

qwen38::MlxArray make_input(qwen38::MlxTensorStore& tensors) {
    const std::vector<std::int32_t> token_values{9419};
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        const auto& config = tensors.manifest().config();
        auto input = make_input(tensors);
        qwen38::SparseMoe moe(
            tensors,
            "language_model.model.layers.0.mlp",
            config.expert_count,
            config.experts_per_token,
            config.quantization_bits,
            config.quantization_group_size,
            config.normalize_topk_probability);
        const auto selection = moe.route_decode(input);
        std::vector<double> timings;
        std::vector<float> values;
        for (int iteration = 0; iteration < 6; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            auto output = moe.forward_decode(input);
            values = output.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
        if (values.size() != config.hidden_size ||
            !std::all_of(values.begin(), values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("MoE output is invalid");
        }
        const double checksum = std::accumulate(values.begin(), values.end(), 0.0);
        std::vector<double> warm(timings.begin() + 1, timings.end());
        std::sort(warm.begin(), warm.end());
        std::cout << "{\"experts\":[";
        for (std::size_t index = 0; index < selection.experts.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << selection.experts[index];
        }
        std::cout << "],\"weights\":[";
        for (std::size_t index = 0; index < selection.weights.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << selection.weights[index];
        }
        std::cout << "],\"checksum\":" << checksum << ",\"cold_ms\":" << timings.front()
                  << ",\"warm_median_ms\":" << warm[warm.size() / 2]
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-sparse-moe-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
