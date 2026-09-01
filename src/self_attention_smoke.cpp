#include "qwen38/hyper_connection.hpp"
#include "qwen38/self_attention.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

qwen38::MlxArray make_input(qwen38::MlxTensorStore& tensors, const std::int32_t token) {
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
        "language_model.model.layers.3.attn_hyper_connection",
        config.hidden_size,
        config.hyper_connection_count,
        config.quantization_bits,
        config.quantization_group_size,
        static_cast<float>(config.rms_norm_epsilon),
        true);
    return mixer.read(stream).mixed;
}

qwen38::SelfAttentionState snapshot(const qwen38::SelfAttentionState& state) {
    qwen38::SelfAttentionState result;
    result.token_count = state.token_count;
    result.position_base = state.position_base;
    result.qsa_pooled_count = state.qsa_pooled_count;
    if (state.token_count != 0) {
        result.keys = state.keys.share();
        result.values = state.values.share();
        result.qsa_raw_keys = state.qsa_raw_keys.share();
    }
    if (state.qsa_pooled_count != 0) {
        result.qsa_pooled_keys = state.qsa_pooled_keys.share();
    }
    return result;
}

double cosine(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size() || left.empty()) {
        throw std::runtime_error("QSA parity output shape mismatch");
    }
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        dot += static_cast<double>(left[index]) * right[index];
        left_norm += static_cast<double>(left[index]) * left[index];
        right_norm += static_cast<double>(right[index]) * right[index];
    }
    return dot / std::sqrt(left_norm * right_norm);
}

int qsa_smoke(
    qwen38::MlxTensorStore& tensors,
    qwen38::SelfAttention& layer,
    const qwen38::MlxArray& input) {
    const auto& config = tensors.manifest().config();
    const std::size_t budget = config.indexer_budget;
    const std::size_t ratio = config.indexer_compress_ratio;
    if (budget % 512 != 0 || ratio == 0) {
        throw std::runtime_error("QSA smoke expects a 512-aligned selection budget");
    }
    static_cast<void>(::setenv("QWEN38_SDPA_PREFILL", "1", 1));
    qwen38::SelfAttentionState origin;
    for (std::size_t offset = 0; offset < budget; offset += 512) {
        const std::array<int, 3> repetitions{1, 512, 1};
        auto chunk = input.tile(repetitions);
        auto output = layer.forward_prefill(chunk, origin);
        output.eval();
    }
    if (origin.token_count != budget ||
        origin.qsa_raw_keys.shape() != std::vector<int>({
            1, static_cast<int>(budget), static_cast<int>(config.indexer_head_dimension)})) {
        throw std::runtime_error("QSA pre-engagement state is incomplete");
    }

    qwen38::SelfAttentionState batched_origin = snapshot(origin);
    qwen38::SelfAttentionState serial = snapshot(origin);
    const std::array<int, 3> verify_repetitions{1, static_cast<int>(ratio), 1};
    auto verify_input = input.tile(verify_repetitions);
    std::vector<qwen38::SelfAttentionState> checkpoints;
    auto batched = layer.forward_verify(verify_input, batched_origin, checkpoints);

    std::vector<qwen38::MlxArray> serial_rows;
    serial_rows.reserve(ratio);
    for (std::size_t row = 0; row < ratio; ++row) {
        serial_rows.push_back(layer.forward_decode(input, serial));
    }
    qwen38::MlxArray serial_output = serial_rows.front().share();
    for (std::size_t row = 1; row < serial_rows.size(); ++row) {
        serial_output = qwen38::MlxArray::concatenate(serial_output, serial_rows[row], 1);
    }
    const auto batch_values = batched.astype(MLX_FLOAT32).to_float32();
    const auto serial_values = serial_output.astype(MLX_FLOAT32).to_float32();
    const double output_cosine = cosine(batch_values, serial_values);
    const std::size_t engaged_tokens = budget + ratio;
    const std::size_t expected_blocks = engaged_tokens / ratio;
    if (checkpoints.size() != ratio || serial.token_count != engaged_tokens ||
        checkpoints.back().token_count != engaged_tokens ||
        checkpoints.back().qsa_pooled_count != expected_blocks ||
        checkpoints.back().qsa_raw_keys.shape() != std::vector<int>({
            1, static_cast<int>(engaged_tokens),
            static_cast<int>(config.indexer_head_dimension)}) ||
        output_cosine < 0.999) {
        throw std::runtime_error("QSA batched/serial parity or cache contract failed");
    }
    for (std::size_t row = 0; row < ratio; ++row) {
        const std::size_t row_tokens = budget + row + 1;
        const std::size_t row_blocks = row_tokens / ratio;
        const std::size_t expected_row_pooled =
            row_blocks > budget / ratio ? row_blocks : 0;
        if (checkpoints[row].token_count != budget + row + 1 ||
            checkpoints[row].qsa_raw_keys.shape()[1] !=
                static_cast<int>(budget + row + 1) ||
            checkpoints[row].qsa_pooled_count != expected_row_pooled) {
            throw std::runtime_error("QSA verifier checkpoint is misaligned");
        }
    }
    std::cout << "{\"qsa_engaged_tokens\":" << engaged_tokens
              << ",\"raw_rows\":" << checkpoints.back().qsa_raw_keys.shape()[1]
              << ",\"pooled_blocks\":" << checkpoints.back().qsa_pooled_count
              << ",\"batch_serial_cosine\":" << output_cosine << "}\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY [--qsa]\n";
        return EXIT_FAILURE;
    }
    try {
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        auto first_input = make_input(tensors, 9419);
        auto second_input = make_input(tensors, 11);
        qwen38::SelfAttention layer(
            tensors,
            "language_model.model.layers.3.self_attn",
            tensors.manifest().config());
        if (argc == 3) {
            if (std::string_view(argv[2]) != "--qsa") {
                throw std::runtime_error("unknown self-attention smoke mode");
            }
            return qsa_smoke(tensors, layer, first_input);
        }
        std::vector<double> timings;
        std::vector<float> first_values;
        std::vector<float> second_values;
        for (int iteration = 0; iteration < 6; ++iteration) {
            qwen38::SelfAttentionState state;
            auto first = layer.forward_decode(first_input, state);
            first_values = first.astype(MLX_FLOAT32).to_float32();
            const auto started = std::chrono::steady_clock::now();
            auto second = layer.forward_decode(second_input, state);
            second_values = second.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
        const auto& config = tensors.manifest().config();
        if (first_values.size() != config.hidden_size || second_values.size() != config.hidden_size ||
            !std::all_of(second_values.begin(), second_values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("self-attention output is invalid");
        }
        const double first_checksum = std::accumulate(first_values.begin(), first_values.end(), 0.0);
        const double second_checksum = std::accumulate(second_values.begin(), second_values.end(), 0.0);
        std::vector<double> warm(timings.begin() + 1, timings.end());
        std::sort(warm.begin(), warm.end());
        std::cout << "{\"first_checksum\":" << first_checksum
                  << ",\"second_checksum\":" << second_checksum
                  << ",\"second_cold_ms\":" << timings.front()
                  << ",\"warm_median_ms\":" << warm[warm.size() / 2]
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-self-attention-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
