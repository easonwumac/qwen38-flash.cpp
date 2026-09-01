#include "qwen38/mlx_backend.hpp"
#include "qwen38/model_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        qwen38::TensorStore store(qwen38::ModelManifest::load(argv[1]));
        constexpr std::string_view weight_name = "language_model.lm_head.weight";
        const std::string& shard_name = store.manifest().weight_map().at(std::string(weight_name));
        qwen38::MlxSafetensors shard(store.manifest().directory() / shard_name);
        const auto weight = shard.tensor(weight_name);
        const auto scales = shard.tensor("language_model.lm_head.scales");
        const auto biases = shard.tensor("language_model.lm_head.biases");
        const std::size_t hidden_size = store.manifest().config().hidden_size;
        std::vector<float> input_values(hidden_size);
        for (std::size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = std::sin(static_cast<float>(i) * 0.01F) * 0.01F;
        }
        const std::vector<int> input_shape{1, static_cast<int>(hidden_size)};
        const auto input_float = qwen38::MlxArray::from_float32(input_values, input_shape);
        const auto input = input_float.astype(MLX_BFLOAT16);

        const auto started = std::chrono::steady_clock::now();
        const auto logits = qwen38::MlxArray::quantized_matmul(
            input,
            weight,
            scales,
            biases,
            static_cast<int>(store.manifest().config().quantization_group_size),
            static_cast<int>(store.manifest().config().quantization_bits));
        const std::vector<float> values = logits.astype(MLX_FLOAT32).to_float32();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        const std::size_t nonfinite = static_cast<std::size_t>(std::count_if(
            values.begin(), values.end(), [](const float value) { return !std::isfinite(value); }));
        const bool finite = nonfinite == 0;
        if (values.size() != store.manifest().config().vocabulary_size || !finite) {
            std::cerr << "invalid lm_head result: size=" << values.size()
                      << " expected=" << store.manifest().config().vocabulary_size
                      << " nonfinite=" << nonfinite << " first="
                      << (values.empty() ? 0.0F : values.front()) << '\n';
            return EXIT_FAILURE;
        }
        const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
        if (*minimum == *maximum) {
            std::cerr << "lm_head result is unexpectedly constant\n";
            return EXIT_FAILURE;
        }
        const auto argmax = static_cast<std::size_t>(
            std::distance(values.begin(), std::max_element(values.begin(), values.end())));
        const double checksum = std::accumulate(values.begin(), values.end(), 0.0);
        std::cout << "{\"device\":\"" << qwen38::mlx_backend_description()
                  << "\",\"logits\":" << values.size()
                  << ",\"argmax\":" << argmax
                  << ",\"checksum\":" << checksum
                  << ",\"elapsed_ms\":" << elapsed_ms << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-qmm-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
