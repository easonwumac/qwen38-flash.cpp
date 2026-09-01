#include "qwen38/hyper_connection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

qwen38::MlxArray embedding(
    qwen38::MlxTensorStore& tensors,
    const std::vector<std::int32_t>& token_values) {
    const std::vector<int> token_shape{static_cast<int>(token_values.size())};
    const auto token_ids = qwen38::MlxArray::from_int32(token_values, token_shape);
    const auto weight = tensors.tensor("language_model.model.embed_tokens.weight");
    const auto scales = tensors.tensor("language_model.model.embed_tokens.scales");
    const auto biases = tensors.tensor("language_model.model.embed_tokens.biases");
    const auto selected_weight = qwen38::MlxArray::take_axis(weight, token_ids, 0);
    const auto selected_scales = qwen38::MlxArray::take_axis(scales, token_ids, 0);
    const auto selected_biases = qwen38::MlxArray::take_axis(biases, token_ids, 0);
    const auto& config = tensors.manifest().config();
    auto result = qwen38::MlxArray::dequantize(
        selected_weight,
        selected_scales,
        selected_biases,
        static_cast<int>(config.quantization_group_size),
        static_cast<int>(config.quantization_bits));
    const std::vector<int> shape{
        1, static_cast<int>(token_values.size()), static_cast<int>(config.hidden_size)};
    return result.reshape(shape);
}

double checked_checksum(const qwen38::MlxArray& value, const std::vector<int>& expected_shape) {
    if (value.shape() != expected_shape) throw std::runtime_error("unexpected hyper-connection shape");
    const std::vector<float> values = value.astype(MLX_FLOAT32).to_float32();
    if (!std::all_of(values.begin(), values.end(), [](const float item) {
            return std::isfinite(item);
        })) {
        throw std::runtime_error("hyper-connection produced non-finite values");
    }
    return std::accumulate(values.begin(), values.end(), 0.0);
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
        const std::vector<std::int32_t> token_values{9419, 11, 1814, 0};
        auto stream = qwen38::HyperConnection::initialize_stream(
            embedding(tensors, token_values), config.hyper_connection_count);
        qwen38::HyperConnection layer(
            tensors,
            "language_model.model.layers.0.attn_hyper_connection",
            config.hidden_size,
            config.hyper_connection_count,
            config.quantization_bits,
            config.quantization_group_size,
            static_cast<float>(config.rms_norm_epsilon),
            true);
        auto read = layer.read(stream);
        auto written = layer.write(stream, read.mixed, read.injection);
        qwen38::HyperConnection final_mixer(
            tensors,
            "language_model.model.hyper_connection_mixer",
            config.hidden_size,
            config.hyper_connection_count,
            config.quantization_bits,
            config.quantization_group_size,
            static_cast<float>(config.rms_norm_epsilon),
            false);
        auto final = final_mixer.read(written);
        const int sequence = static_cast<int>(token_values.size());
        const int hidden = static_cast<int>(config.hidden_size);
        const int stream_width = hidden * static_cast<int>(config.hyper_connection_count);
        const double mixed_checksum = checked_checksum(read.mixed, {1, sequence, hidden});
        const double injection_checksum = checked_checksum(
            read.injection,
            {1, sequence, static_cast<int>(config.hyper_connection_count), 1});
        const double stream_checksum = checked_checksum(written, {1, sequence, stream_width});
        const double final_checksum = checked_checksum(final.mixed, {1, sequence, hidden});
        std::cout << "{\"mixed_checksum\":" << mixed_checksum
                  << ",\"injection_checksum\":" << injection_checksum
                  << ",\"stream_checksum\":" << stream_checksum
                  << ",\"final_checksum\":" << final_checksum
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-hyper-connection-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
