#include "qwen38/decoder_layer.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model_manifest.hpp"
#include "qwen38/model.hpp"
#include "qwen38/persistent_metal_backend.hpp"
#include "qwen38/ple.hpp"

#include <bit>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
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
    if (!std::isfinite(cosine) || !std::isfinite(rmse) ||
        cosine < 0.99998 || rmse > 0.002 || max_abs > 0.012) {
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
    if (!std::isfinite(cosine) || !std::isfinite(rmse) ||
        cosine < 0.99994 || rmse > 0.004 || max_abs > 0.025) {
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
    if (!std::isfinite(minimum_cosine) || !std::isfinite(maximum_rmse) ||
        minimum_cosine < 0.99990 || maximum_rmse > 0.015 || maximum_abs > 0.20) {
        throw std::runtime_error("persistent attention trajectory parity failed");
    }
}

void check_trunk(qwen38::PersistentMetalBackend& backend,
                 qwen38::MlxTensorStore& tensors,
                 const std::vector<float>& input_f32,
                 const std::vector<std::uint16_t>& input_bf16) {
    double gpu_ms = 0.0;
    const std::vector<std::uint16_t> actual =
        backend.decode_trunk(9419, input_bf16, true, &gpu_ms);
    std::vector<std::unique_ptr<qwen38::DecoderLayer>> layers;
    std::vector<qwen38::DecoderLayerState> states(48);
    layers.reserve(48);
    for (std::size_t index = 0; index < 48; ++index) {
        layers.push_back(std::make_unique<qwen38::DecoderLayer>(
            tensors, index, tensors.manifest().config()));
    }
    qwen38::MlxArray oracle = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    for (std::size_t index = 0; index < 48; ++index) {
        oracle = layers[index]->forward_decode(oracle, 9419, states[index]);
    }
    const std::vector<float> expected = oracle.astype(MLX_FLOAT32).to_float32();
    for (int warmup = 0; warmup < 3; ++warmup) {
        static_cast<void>(backend.decode_trunk(9419, input_bf16, true));
    }
    std::vector<double> trunk_samples;
    std::vector<double> trunk_wall_samples;
    trunk_samples.reserve(11);
    for (int sample = 0; sample < 11; ++sample) {
        double measured = 0.0;
        const auto started = std::chrono::steady_clock::now();
        static_cast<void>(backend.decode_trunk(9419, input_bf16, true, &measured));
        trunk_wall_samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        trunk_samples.push_back(measured);
    }
    std::ranges::sort(trunk_samples);
    std::ranges::sort(trunk_wall_samples);
    gpu_ms = trunk_samples[trunk_samples.size() / 2];
    double dot = 0.0, aa = 0.0, bb = 0.0, squared = 0.0, max_abs = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double lhs = from_bf16(actual[index]);
        const double rhs = expected[index];
        const double delta = lhs - rhs;
        dot += lhs * rhs;
        aa += lhs * lhs;
        bb += rhs * rhs;
        squared += delta * delta;
        max_abs = std::max(max_abs, std::abs(delta));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    const double rmse = std::sqrt(squared / static_cast<double>(actual.size()));
    std::cout << "trunk_gpu_ms " << gpu_ms << " trunk_wall_ms "
              << trunk_wall_samples[trunk_wall_samples.size() / 2]
              << " trunk_cosine " << cosine
              << " trunk_rmse " << rmse << " trunk_max_abs " << max_abs << '\n';
    if (cosine < 0.99 || !std::isfinite(rmse)) {
        throw std::runtime_error("persistent trunk parity failed");
    }
}

void trace_trunk_layers(qwen38::PersistentMetalBackend& backend,
                        qwen38::MlxTensorStore& tensors,
                        const std::vector<float>& input_f32,
                        const std::vector<std::uint16_t>& input_bf16) {
    std::vector<std::unique_ptr<qwen38::DecoderLayer>> layers;
    std::vector<qwen38::DecoderLayerState> states(48);
    layers.reserve(48);
    for (std::size_t index = 0; index < 48; ++index) {
        layers.push_back(std::make_unique<qwen38::DecoderLayer>(
            tensors, index, tensors.manifest().config()));
    }
    qwen38::MlxArray oracle = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    std::vector<std::uint16_t> direct = input_bf16;
    for (std::size_t index = 0; index < 48; ++index) {
        if (index == 1) direct = backend.decode_ple(9419, direct, true);
        direct = index % 4 == 3
            ? backend.decode_attention_layer(index, direct, true)
            : backend.decode_gdn_layer(index, direct, true);
        oracle = layers[index]->forward_decode(oracle, 9419, states[index]);
        const std::vector<float> expected = oracle.astype(MLX_FLOAT32).to_float32();
        double dot = 0.0, aa = 0.0, bb = 0.0;
        for (std::size_t component = 0; component < direct.size(); ++component) {
            const double lhs = from_bf16(direct[component]);
            const double rhs = expected[component];
            dot += lhs * rhs;
            aa += lhs * lhs;
            bb += rhs * rhs;
        }
        const double cosine = dot / std::sqrt(aa * bb);
        std::cout << "trunk_layer " << index << " cosine " << cosine << '\n';
    }
}

void check_ple(qwen38::PersistentMetalBackend& backend,
               qwen38::MlxTensorStore& tensors,
               const std::vector<float>& input_f32,
               const std::vector<std::uint16_t>& input_bf16) {
    qwen38::Ple ple(tensors, "language_model.model.layers.1.ple",
                    tensors.manifest().config());
    qwen38::PleState state;
    qwen38::MlxArray input = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    qwen38::MlxArray ple_output = ple.forward_decode(input, 9419, state);
    const std::vector<float> expected = qwen38::MlxArray::add(input, ple_output)
        .astype(MLX_FLOAT32).to_float32();
    double gpu_ms = 0.0;
    const std::vector<std::uint16_t> actual =
        backend.decode_ple(9419, input_bf16, true, &gpu_ms);
    double dot = 0.0, aa = 0.0, bb = 0.0, squared = 0.0, max_abs = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double lhs = from_bf16(actual[index]);
        const double rhs = expected[index];
        const double delta = lhs - rhs;
        dot += lhs * rhs;
        aa += lhs * lhs;
        bb += rhs * rhs;
        squared += delta * delta;
        max_abs = std::max(max_abs, std::abs(delta));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    const double rmse = std::sqrt(squared / static_cast<double>(actual.size()));
    std::cout << "ple_gpu_ms " << gpu_ms << " ple_cosine " << cosine
              << " ple_rmse " << rmse << " ple_max_abs " << max_abs << '\n';
    if (cosine < 0.999 || rmse > 0.05) throw std::runtime_error("persistent PLE parity failed");
}

void check_head(qwen38::PersistentMetalBackend& backend,
                qwen38::MlxTensorStore& tensors,
                const std::vector<float>& input_f32,
                const std::vector<std::uint16_t>& input_bf16) {
    const qwen38::ModelConfig& config = tensors.manifest().config();
    qwen38::HyperConnection mixer(
        tensors, "language_model.model.hyper_connection_mixer",
        config.hidden_size, config.hyper_connection_count,
        config.quantization_bits, config.quantization_group_size,
        static_cast<float>(config.rms_norm_epsilon), false);
    qwen38::MlxArray input = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    qwen38::MlxArray mixed = mixer.read(input).mixed;
    const qwen38::QuantizationSpec quantization =
        tensors.manifest().quantization_for("language_model.lm_head");
    qwen38::MlxArray logits = qwen38::MlxArray::quantized_matmul(
        mixed,
        tensors.tensor("language_model.lm_head.weight"),
        tensors.tensor("language_model.lm_head.scales"),
        tensors.tensor("language_model.lm_head.biases"),
        static_cast<int>(quantization.group_size),
        static_cast<int>(quantization.bits));
    const std::uint32_t expected = logits.argmax_all().item_uint32();
    const auto actual = backend.greedy_head(input_bf16);
    std::cout << "head_token " << actual.token << " mlx_token " << expected
              << " head_logit " << actual.logit << '\n';
    if (actual.token != expected || !std::isfinite(actual.logit)) {
        throw std::runtime_error("persistent head parity failed");
    }
}

void check_greedy(qwen38::PersistentMetalBackend& backend,
                  qwen38::MlxTensorStore& tensors) {
    qwen38::QwenModel model(tensors);
    qwen38::ModelDecodeState head_state = model.make_state();
    qwen38::TargetDecodeStep captured = model.forward_decode_capture(9419, head_state);
    const std::vector<float> oracle_stream_f32 =
        captured.pre_mixer_stream.astype(MLX_FLOAT32).to_float32();
    std::vector<std::uint16_t> oracle_stream;
    oracle_stream.reserve(oracle_stream_f32.size());
    for (const float value : oracle_stream_f32) oracle_stream.push_back(bf16(value));
    const auto direct_oracle_head = backend.greedy_head(oracle_stream);
    qwen38::MlxArray oracle_token_array = captured.logits.argmax_all();
    qwen38::MlxArray oracle_selected =
        qwen38::MlxArray::take(captured.logits, oracle_token_array).astype(MLX_FLOAT32);
    std::cout << "oracle_stream_head_token " << direct_oracle_head.token
              << " mlx_token " << oracle_token_array.item_uint32()
              << " direct_logit " << direct_oracle_head.logit
              << " mlx_logit " << oracle_selected.item_float32() << '\n';

    qwen38::ModelDecodeState state = model.make_state();
    const qwen38::GreedyStep expected = model.greedy_decode(9419, state);
    const auto actual = backend.greedy_decode(9419, true);
    std::vector<double> gpu_samples, wall_samples;
    for (int warmup = 0; warmup < 3; ++warmup) {
        static_cast<void>(backend.greedy_decode(9419, true));
    }
    for (int sample = 0; sample < 11; ++sample) {
        const auto measured = backend.greedy_decode(9419, true);
        gpu_samples.push_back(measured.gpu_ms);
        wall_samples.push_back(measured.wall_ms);
    }
    std::ranges::sort(gpu_samples);
    std::ranges::sort(wall_samples);
    std::cout << "greedy_token " << actual.token << " mlx_token " << expected.token
              << " greedy_logit " << actual.logit << " mlx_logit " << expected.logit
              << " greedy_gpu_ms " << gpu_samples[gpu_samples.size() / 2]
              << " greedy_wall_ms " << wall_samples[wall_samples.size() / 2] << '\n';
    if (actual.token != expected.token) {
        throw std::runtime_error("persistent greedy token parity failed");
    }
    std::uint32_t oracle_token = expected.token;
    std::size_t matching_steps = 1;
    for (std::size_t step = 1; step < 16; ++step) {
        const qwen38::GreedyStep oracle_step = model.greedy_decode(oracle_token, state);
        const auto direct_step = backend.greedy_decode(oracle_token, false);
        oracle_token = oracle_step.token;
        std::cout << "greedy_step " << (step + 1) << " direct " << direct_step.token
                  << " mlx " << oracle_token << " direct_logit " << direct_step.logit
                  << " mlx_logit " << oracle_step.logit << '\n';
        if (direct_step.token == oracle_token) ++matching_steps;
    }
    std::cout << "teacher_forced_greedy_agreement " << matching_steps << " of 16\n";
    if (matching_steps < 4) {
        throw std::runtime_error("persistent teacher-forced greedy agreement regressed");
    }
}

void check_state_import(qwen38::PersistentMetalBackend& backend,
                        qwen38::MlxTensorStore& tensors) {
    qwen38::QwenModel model(tensors);
    qwen38::ModelDecodeState state = model.make_state();
    const std::array<std::uint32_t, 3> prefix{9419, 11, 353};
    static_cast<void>(model.prefill_chunk(prefix, state));
    backend.import_state(state);
    const qwen38::GreedyStep expected = model.greedy_decode(2688, state);
    const auto actual = backend.greedy_decode(2688, false);
    std::cout << "imported_state_token " << actual.token << " mlx_token "
              << expected.token << " imported_state_logit " << actual.logit
              << " mlx_logit " << expected.logit << '\n';
    if (actual.token != expected.token) {
        throw std::runtime_error("persistent imported-state token parity failed");
    }
}

void check_q8_state_import(qwen38::PersistentMetalBackend& backend,
                           qwen38::MlxTensorStore& tensors,
                           const std::vector<float>& input_f32,
                           const std::vector<std::uint16_t>& input_bf16) {
    qwen38::QwenModel model(tensors);
    qwen38::ModelDecodeState state = model.make_state();
    const std::vector<std::uint32_t> prefix(2050, 9419);
    constexpr std::size_t chunk = 64;
    for (std::size_t offset = 0; offset < prefix.size(); offset += chunk) {
        const std::size_t count = std::min(chunk, prefix.size() - offset);
        static_cast<void>(model.prefill_chunk(
            std::span<const std::uint32_t>(prefix.data() + offset, count), state));
    }
    std::cout << "q8_prefill_token_count " << state.layers[3].full_attention.token_count
              << " q8_enabled " << state.layers[3].full_attention.kv_q8
              << " q8_cold " << state.layers[3].full_attention.kv_q8_cold_tokens << '\n';
    if (!state.layers[3].full_attention.kv_q8) {
        throw std::runtime_error("Q8 import smoke did not create Q8 KV state");
    }
    backend.import_state(state);
    qwen38::DecoderLayer layer(tensors, 3, tensors.manifest().config());
    qwen38::DecoderLayerState layer_state =
        qwen38::snapshot_decoder_layer_state(state.layers[3]);
    qwen38::MlxArray input = qwen38::MlxArray::from_float32(
        input_f32, std::vector<int>{1, 1, 10240}).astype(MLX_BFLOAT16);
    const std::vector<float> expected =
        layer.forward_decode(input, 9419, layer_state).astype(MLX_FLOAT32).to_float32();
    const std::vector<std::uint16_t> actual =
        backend.decode_attention_layer(3, input_bf16, false);
    double dot = 0.0, aa = 0.0, bb = 0.0, squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double lhs = from_bf16(actual[index]);
        const double rhs = expected[index];
        const double delta = lhs - rhs;
        dot += lhs * rhs;
        aa += lhs * lhs;
        bb += rhs * rhs;
        squared += delta * delta;
    }
    const double cosine = dot / std::sqrt(aa * bb);
    const double rmse = std::sqrt(squared / actual.size());
    std::cout << "q8_import_tokens 2050 q8_import_cosine " << cosine
              << " q8_import_rmse " << rmse << '\n';
    if (cosine < 0.999 || rmse > 0.05) {
        throw std::runtime_error("persistent Q8 imported-state parity failed");
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
                  << "mapped_weight_bytes " << inventory.mapped_weight_bytes << std::endl;
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
        std::size_t first_layer = 0;
        if (const char* requested = std::getenv("QWEN38_PERSISTENT_SMOKE_LAYER");
            requested != nullptr && *requested != '\0') {
            first_layer = static_cast<std::size_t>(std::strtoul(requested, nullptr, 10));
        }
        check_layer(*backend, tensors, first_layer, input_f32, input_bf16);
        if (const char* quick = std::getenv("QWEN38_PERSISTENT_SMOKE_QUICK");
            quick != nullptr && std::string_view(quick) == "1") {
            check_ple(*backend, tensors, input_f32, input_bf16);
            check_attention_layer(*backend, tensors, 3, input_f32, input_bf16);
            check_head(*backend, tensors, input_f32, input_bf16);
            return inventory.pipeline_count == 48 && inventory.shard_count != 0 ? 0 : 1;
        }
        check_layer(*backend, tensors, 10, input_f32, input_bf16);
        check_attention_layer(*backend, tensors, 3, input_f32, input_bf16);
        check_attention_layer(*backend, tensors, 11, input_f32, input_bf16);
        check_ple(*backend, tensors, input_f32, input_bf16);
        trace_trunk_layers(*backend, tensors, input_f32, input_bf16);
        check_trunk(*backend, tensors, input_f32, input_bf16);
        check_greedy(*backend, tensors);
        check_state_import(*backend, tensors);
        if (const char* q8_import = std::getenv("QWEN38_TEST_Q8_IMPORT");
            q8_import != nullptr && std::string_view(q8_import) == "1") {
            check_q8_state_import(*backend, tensors, input_f32, input_bf16);
        }
        return inventory.pipeline_count == 48 && inventory.shard_count != 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
