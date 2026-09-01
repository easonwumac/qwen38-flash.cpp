#include "qwen38/mlx_backend.hpp"
#include "qwen38/model_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        qwen38::TensorStore store(qwen38::ModelManifest::load(argv[1]));
        constexpr std::string_view weight_name = "language_model.model.embed_tokens.weight";
        const std::string& shard_name = store.manifest().weight_map().at(std::string(weight_name));
        qwen38::MlxSafetensors shard(store.manifest().directory() / shard_name);
        const auto weight = shard.tensor(weight_name);
        const auto scales = shard.tensor("language_model.model.embed_tokens.scales");
        const auto biases = shard.tensor("language_model.model.embed_tokens.biases");
        const std::vector<std::int32_t> token_values{9419, 11, 1814, 0};
        const std::vector<int> token_shape{static_cast<int>(token_values.size())};
        const auto token_ids = qwen38::MlxArray::from_int32(token_values, token_shape);

        std::vector<double> timings;
        std::vector<float> values;
        for (int iteration = 0; iteration < 6; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            const auto selected_weight = qwen38::MlxArray::take_axis(weight, token_ids, 0);
            const auto selected_scales = qwen38::MlxArray::take_axis(scales, token_ids, 0);
            const auto selected_biases = qwen38::MlxArray::take_axis(biases, token_ids, 0);
            const auto embedding = qwen38::MlxArray::dequantize(
                selected_weight,
                selected_scales,
                selected_biases,
                static_cast<int>(store.manifest().config().quantization_group_size),
                static_cast<int>(store.manifest().config().quantization_bits));
            values = embedding.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
        const std::size_t expected = token_values.size() * store.manifest().config().hidden_size;
        if (values.size() != expected ||
            !std::all_of(values.begin(), values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            std::cerr << "invalid embedding result\n";
            return EXIT_FAILURE;
        }
        const double checksum = std::accumulate(values.begin(), values.end(), 0.0);
        double squared = 0.0;
        for (const float value : values) squared += static_cast<double>(value) * value;
        std::vector<double> warm(timings.begin() + 1, timings.end());
        std::sort(warm.begin(), warm.end());
        std::cout << "{\"shape\":[" << token_values.size() << ','
                  << store.manifest().config().hidden_size << "],\"checksum\":" << checksum
                  << ",\"l2\":" << std::sqrt(squared)
                  << ",\"cold_ms\":" << timings.front()
                  << ",\"warm_median_ms\":" << warm[warm.size() / 2] << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-embedding-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
