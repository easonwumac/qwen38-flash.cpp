#include "qwen38/decoder_layer.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model_manifest.hpp"
#include "qwen38/persistent_metal_backend.hpp"

#include <bit>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::uint16_t bf16(const float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float from_bf16(const std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

void check_layer(qwen38::PersistentMetalBackend& backend,
                 qwen38::MlxTensorStore& tensors, std::size_t layer_index,
                 const std::vector<float>& input_f32,
                 const std::vector<std::uint16_t>& input_bf16) {
    qwen38::DecoderLayer layer(tensors, layer_index, tensors.manifest().config());
    qwen38::DecoderLayerState state;
    qwen38::MlxArray input = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    const std::vector<float> expected =
        layer.forward_decode(input, 9419, state).astype(MLX_FLOAT32).to_float32();
    double gpu_ms = 0.0;
    const std::vector<std::uint16_t> actual =
        backend.decode_gdn_layer(layer_index, input_bf16, true, &gpu_ms);
    for (int warmup = 0; warmup < 5; ++warmup) {
        static_cast<void>(backend.decode_gdn_layer(layer_index, input_bf16, true));
    }
    std::vector<double> samples;
    samples.reserve(31);
    for (int sample = 0; sample < 31; ++sample) {
        double measured = 0.0;
        static_cast<void>(backend.decode_gdn_layer(layer_index, input_bf16, true, &measured));
        samples.push_back(measured);
    }
    std::ranges::sort(samples);
    gpu_ms = samples[samples.size() / 2];
    double dot = 0.0, aa = 0.0, bb = 0.0, squared_error = 0.0, max_abs = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double lhs = from_bf16(actual[index]);
        const double rhs = expected[index];
        const double delta = lhs - rhs;
        dot += lhs * rhs;
        aa += lhs * lhs;
        bb += rhs * rhs;
        squared_error += delta * delta;
        max_abs = std::max(max_abs, std::abs(delta));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    const double rmse = std::sqrt(squared_error / static_cast<double>(actual.size()));
    std::cout << "layer " << layer_index << " gpu_ms " << gpu_ms
              << " cosine " << cosine << " rmse " << rmse
              << " max_abs " << max_abs << '\n';
    if (cosine < 0.99998 || rmse > 0.002 || max_abs > 0.012) {
        throw std::runtime_error("persistent GDN layer parity failed");
    }
}

void check_attention_layer(qwen38::PersistentMetalBackend& backend,
                           qwen38::MlxTensorStore& tensors, std::size_t layer_index,
                           const std::vector<float>& input_f32,
                           const std::vector<std::uint16_t>& input_bf16) {
    qwen38::DecoderLayer layer(tensors, layer_index, tensors.manifest().config());
    qwen38::DecoderLayerState state;
    qwen38::MlxArray input = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    const std::vector<float> expected =
        layer.forward_decode(input, 9419, state).astype(MLX_FLOAT32).to_float32();
    double gpu_ms = 0.0;
    const std::vector<std::uint16_t> actual =
        backend.decode_attention_layer(layer_index, input_bf16, true, &gpu_ms);
    for (int warmup = 0; warmup < 5; ++warmup) {
        static_cast<void>(backend.decode_attention_layer(layer_index, input_bf16, true));
    }
    std::vector<double> samples;
    samples.reserve(31);
    for (int sample = 0; sample < 31; ++sample) {
        double measured = 0.0;
        static_cast<void>(backend.decode_attention_layer(
            layer_index, input_bf16, true, &measured));
        samples.push_back(measured);
    }
    std::ranges::sort(samples);
    gpu_ms = samples[samples.size() / 2];
    double dot = 0.0, aa = 0.0, bb = 0.0, squared_error = 0.0, max_abs = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double lhs = from_bf16(actual[index]);
        const double rhs = expected[index];
        const double delta = lhs - rhs;
        dot += lhs * rhs;
        aa += lhs * lhs;
        bb += rhs * rhs;
        squared_error += delta * delta;
        max_abs = std::max(max_abs, std::abs(delta));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    const double rmse = std::sqrt(squared_error / static_cast<double>(actual.size()));
    std::cout << "layer " << layer_index << " gpu_ms " << gpu_ms
              << " cosine " << cosine << " rmse " << rmse
              << " max_abs " << max_abs << '\n';
    if (cosine < 0.99994 || rmse > 0.004 || max_abs > 0.025) {
        throw std::runtime_error("persistent attention layer parity failed");
    }

    qwen38::DecoderLayer trajectory_layer(
        tensors, layer_index, tensors.manifest().config());
    qwen38::DecoderLayerState trajectory_state;
    qwen38::MlxArray oracle_stream = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    std::vector<std::uint16_t> direct_stream = input_bf16;
    double minimum_cosine = 1.0, maximum_rmse = 0.0, maximum_abs = 0.0;
    for (int step = 0; step < 4; ++step) {
        qwen38::MlxArray oracle_output = trajectory_layer.forward_decode(
            oracle_stream, 9419, trajectory_state);
        const std::vector<float> oracle_values =
            oracle_output.astype(MLX_FLOAT32).to_float32();
        direct_stream = backend.decode_attention_layer(
            layer_index, direct_stream, step == 0);
        double step_dot = 0.0, step_aa = 0.0, step_bb = 0.0, step_squared = 0.0;
        for (std::size_t index = 0; index < direct_stream.size(); ++index) {
            const double lhs = from_bf16(direct_stream[index]);
            const double rhs = oracle_values[index];
            const double delta = lhs - rhs;
            step_dot += lhs * rhs;
            step_aa += lhs * lhs;
            step_bb += rhs * rhs;
            step_squared += delta * delta;
            maximum_abs = std::max(maximum_abs, std::abs(delta));
        }
        minimum_cosine = std::min(
            minimum_cosine, step_dot / std::sqrt(step_aa * step_bb));
        maximum_rmse = std::max(maximum_rmse,
            std::sqrt(step_squared / static_cast<double>(direct_stream.size())));
        std::cout << "layer " << layer_index << " trajectory_step " << step + 1
                  << " cosine " << step_dot / std::sqrt(step_aa * step_bb)
                  << " rmse "
                  << std::sqrt(step_squared / static_cast<double>(direct_stream.size()))
                  << '\n';
        oracle_stream = std::move(oracle_output);
    }
    std::cout << "layer " << layer_index << " trajectory_min_cosine "
              << minimum_cosine << " trajectory_max_rmse " << maximum_rmse
              << " trajectory_max_abs " << maximum_abs << '\n';
    if (minimum_cosine < 0.99990 || maximum_rmse > 0.015 || maximum_abs > 0.20) {
        throw std::runtime_error("persistent attention trajectory parity failed");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: qwen38-persistent-metal-backend-smoke MODEL");
        }
        const qwen38::ModelManifest manifest = qwen38::ModelManifest::load(argv[1]);
        auto backend = qwen38::PersistentMetalBackend::create(manifest);
        if (!backend) throw std::runtime_error("model is not eligible for persistent Metal");
        const auto& inventory = backend->inventory();
        std::cout << "pipelines " << inventory.pipeline_count << '\n'
                  << "shards " << inventory.shard_count << '\n'
                  << "mapped_weight_bytes " << inventory.mapped_weight_bytes << '\n';
        std::vector<float> input_f32(10240);
        for (std::size_t index = 0; index < input_f32.size(); ++index) {
            input_f32[index] = static_cast<float>(
                static_cast<int>((index * 29 + 7) % 521) - 260) / 512.0F;
        }
        std::vector<std::uint16_t> input_bf16(input_f32.size());
        for (std::size_t index = 0; index < input_f32.size(); ++index) {
            input_bf16[index] = bf16(input_f32[index]);
        }
        qwen38::MlxTensorStore tensors(manifest);
        check_layer(*backend, tensors, 0, input_f32, input_bf16);
        check_layer(*backend, tensors, 10, input_f32, input_bf16);
        check_attention_layer(*backend, tensors, 3, input_f32, input_bf16);
        check_attention_layer(*backend, tensors, 11, input_f32, input_bf16);
        return inventory.pipeline_count == 29 && inventory.shard_count != 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
