#include "qwen38/self_attention.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/token_embedding.hpp"
#include "qwen38/runtime_profile.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace qwen38;
struct Packed { MlxArray weights, scales, biases; };
Packed pack_q8(const MlxArray& value) {
    auto vector = mlx_vector_array_new();
    const auto stream = mlx_default_gpu_stream_new();
    const int status = mlx_quantize(
        &vector, value.get(), {.value = 64, .has_value = true},
        {.value = 8, .has_value = true}, "affine", mlx_array{}, stream);
    mlx_stream_free(stream);
    if (status != 0) {
        mlx_vector_array_free(vector);
        throw std::runtime_error("Q8 packing failed");
    }
    std::array<mlx_array, 3> raw{
        mlx_array_new(), mlx_array_new(), mlx_array_new()};
    const int extract = mlx_vector_array_get(&raw[0], vector, 0) |
        mlx_vector_array_get(&raw[1], vector, 1) |
        mlx_vector_array_get(&raw[2], vector, 2);
    mlx_vector_array_free(vector);
    if (extract != 0) throw std::runtime_error("Q8 tuple extraction failed");
    Packed packed{MlxArray(raw[0]), MlxArray(raw[1]), MlxArray(raw[2])};
    const std::array<const MlxArray*, 3> outputs{
        &packed.weights, &packed.scales, &packed.biases};
    MlxArray::eval_all(outputs);
    return packed;
}
MlxArray history(std::vector<int> shape, int seed) {
    std::size_t count = 1;
    for (int n : shape) count *= static_cast<std::size_t>(n);
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = static_cast<float>(static_cast<int>((i * 13 + seed) % 257) - 128) / 128.F;
    auto value = MlxArray::from_float32(values, shape).astype(MLX_BFLOAT16);
    value.eval(); return value;
}
int main(int argc, char** argv) try {
    if (argc != 4 || std::getenv("QWEN38_MEMORY_GUARD") == nullptr)
        throw std::runtime_error(
            "usage: guarded qwen38-qsa-peak-probe MODEL CONTEXT MODE(0..4|decode)");
    const int context = std::stoi(argv[2]);
    const bool decode_probe = std::string_view(argv[3]) == "decode";
    if ((context != 32768 && context != 65536 && context != 131072) ||
        (!decode_probe && (std::string_view(argv[3]).size() != 1 ||
         argv[3][0] < '0' || argv[3][0] > '4')))
        throw std::runtime_error("invalid bounded probe configuration");
    const char* raw_toggle = std::getenv("QWEN38_QSA_BENCH_TOGGLE");
    const std::string toggle = raw_toggle == nullptr ? "" : raw_toggle;
    if (!toggle.empty() && (!toggle.starts_with("QWEN38_") ||
            toggle.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos))
        throw std::runtime_error("invalid QSA benchmark switch");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 3ULL * 1024 * 1024 * 1024) != 0)
        throw std::runtime_error("allocation limit failed");
    static_cast<void>(MlxArray::set_cache_limit(64ULL * 1024 * 1024));
    apply_runtime_profile("speed");
    setenv("QWEN38_QSA_PACKED_PREFILL", "1", 1);
    setenv("QWEN38_QSA_PACKED_MIN_TOKENS", "0", 1);
    setenv("QWEN38_QSA_SCORE_REDUCE", decode_probe ? "0" : argv[3], 1);
    if (decode_probe) {
        if (std::getenv("QWEN38_QSA_DECODE_BUDGET") == nullptr)
            setenv("QWEN38_QSA_DECODE_BUDGET", "512", 1);
        // Phase barriers are opt-in: ordinary A/B must measure the lazy graph.
    }
    unsetenv("QWEN38_QSA_TILED_SCORES");
    if (!decode_probe && argv[3][0] >= '2') {
        const char* modes[]{"128", "256", "async"};
        setenv("QWEN38_QSA_TILED_SCORES", modes[argv[3][0] - '2'], 1);
    }
    MlxTensorStore tensors(ModelManifest::load(argv[1]));
    const auto& c = tensors.manifest().config();
    SelfAttention layer(tensors, "language_model.model.layers.3.self_attn", c);
    std::vector<std::uint32_t> tokens(decode_probe ? 1 : 512);
    for (std::size_t i = 0; i < tokens.size(); ++i)
        tokens[i] = static_cast<std::uint32_t>((9419 + 7919 * i) % c.vocabulary_size);
    const auto embedding_quantization = tensors.manifest().quantization_for(
        "language_model.model.embed_tokens");
    const auto mixer_quantization = tensors.manifest().quantization_for(
        "language_model.model.layers.3.attn_hyper_connection.input_mix_weight_down");
    auto embedding = embed_token_batch(tensors.tensor("language_model.model.embed_tokens.weight"),
        tensors.tensor("language_model.model.embed_tokens.scales"),
        tensors.tensor("language_model.model.embed_tokens.biases"), tokens,
        c.vocabulary_size, c.hidden_size,
        static_cast<int>(embedding_quantization.group_size),
        static_cast<int>(embedding_quantization.bits));
    HyperConnection mixer(tensors, "language_model.model.layers.3.attn_hyper_connection",
        c.hidden_size, c.hyper_connection_count, mixer_quantization.bits,
        mixer_quantization.group_size, static_cast<float>(c.rms_norm_epsilon), true);
    auto input = mixer.read(HyperConnection::initialize_stream(embedding, c.hyper_connection_count)).mixed;
    input.eval();
    SelfAttentionState origin;
    origin.keys = history({1, 2, context, 256}, 1);
    origin.values = history({1, 2, context, 256}, 7);
    origin.qsa_raw_keys = history({1, context, 128}, 13);
    origin.qsa_pooled_keys = history({1, context / 4, 128}, 19);
    origin.token_count = context; origin.qsa_pooled_count = context / 4;
    if (decode_probe && std::getenv("QWEN38_QSA_RAW_WINDOW") != nullptr) {
        origin.qsa_raw_keys = history({1, 64, 128}, 13);
        origin.qsa_raw_start = static_cast<std::size_t>(context - 64);
    }
    if (decode_probe) {
        Packed key = pack_q8(origin.keys);
        Packed value = pack_q8(origin.values);
        origin.key_weights = std::move(key.weights);
        origin.key_scales = std::move(key.scales);
        origin.key_biases = std::move(key.biases);
        origin.value_weights = std::move(value.weights);
        origin.value_scales = std::move(value.scales);
        origin.value_biases = std::move(value.biases);
        const std::vector<int> strides{1, 1, 1, 1};
        origin.keys = origin.keys.slice(
            std::vector<int>{0, 0, 0, 0}, std::vector<int>{1, 2, 0, 256}, strides);
        origin.values = origin.values.slice(
            std::vector<int>{0, 0, 0, 0}, std::vector<int>{1, 2, 0, 256}, strides);
        origin.kv_q8 = true;
        origin.kv_q8_cold_tokens = static_cast<std::size_t>(context);
    }
    std::vector<double> times;
    std::vector<std::size_t> peaks;
    std::uint64_t hash = 1469598103934665603ULL;
    std::vector<double> control_times, candidate_times;
    std::vector<float> reference_values;
    const int warmups = toggle.empty() ? 3 : 40;
    const int iterations = toggle.empty() ? 14 : 160;
    for (int i = 0; i < iterations; ++i) {
        const bool candidate = i % 4 == 1 || i % 4 == 2;
        if (!toggle.empty()) setenv(toggle.c_str(), candidate ? "1" : "0", 1);
        SelfAttentionState state;
        state.keys = origin.keys.share(); state.values = origin.values.share();
        state.qsa_raw_keys = origin.qsa_raw_keys.share();
        state.qsa_raw_start = origin.qsa_raw_start;
        state.qsa_pooled_keys = origin.qsa_pooled_keys.share();
        state.token_count = origin.token_count; state.qsa_pooled_count = origin.qsa_pooled_count;
        state.kv_q8 = origin.kv_q8;
        state.kv_q8_cold_tokens = origin.kv_q8_cold_tokens;
        if (decode_probe) {
            state.key_weights = origin.key_weights.share();
            state.key_scales = origin.key_scales.share();
            state.key_biases = origin.key_biases.share();
            state.value_weights = origin.value_weights.share();
            state.value_scales = origin.value_scales.share();
            state.value_biases = origin.value_biases.share();
        }
        if (mlx_reset_peak_memory() != 0) throw std::runtime_error("peak reset failed");
        const auto start = std::chrono::steady_clock::now();
        auto output = decode_probe
            ? layer.forward_decode(input,  state)
            : layer.forward_prefill(input, state);
        const std::array<const MlxArray*, 5> evaluated{
            &output, &state.keys, &state.values, &state.qsa_raw_keys, &state.qsa_pooled_keys};
        MlxArray::eval_all(evaluated);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::size_t peak{};
        if (mlx_get_peak_memory(&peak) != 0) throw std::runtime_error("peak query failed");
        if (i >= warmups) {
            times.push_back(ms); peaks.push_back(peak);
            (candidate ? candidate_times : control_times).push_back(ms);
        }
        std::vector<float> observed;
        if (!toggle.empty() || i == iterations - 1) for (const auto* value : evaluated)
            for (float x : value->astype(MLX_FLOAT32).to_float32()) {
                if (!std::isfinite(x)) throw std::runtime_error("nonfinite state/output");
                observed.push_back(x);
                if (i == iterations - 1) {
                    hash ^= std::bit_cast<std::uint32_t>(x); hash *= 1099511628211ULL;
                }
            }
        if (!toggle.empty()) {
            if (i == 0) reference_values = observed;
            if (observed.size() != reference_values.size() || !std::equal(
                    observed.begin(), observed.end(), reference_values.begin(),
                    [](float a, float b) {
                        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
                    })) throw std::runtime_error("interleaved QSA A/B changed output/state bits");
        }
    }
    std::cout << std::setprecision(10)
              << "{\"synthetic_history\":true,\"context\":" << context
              << ",\"rows\":" << (decode_probe ? 1 : 512)
              << ",\"phase_barriers\":"
              << (std::getenv("QWEN38_PROFILE_QSA_DECODE") != nullptr ? "true" : "false")
              << ",\"mode\":";
    if (decode_probe) std::cout << "\"decode\"";
    else std::cout << argv[3];
    std::cout << ",\"samples_ms\":[";
    for (std::size_t i = 0; i < times.size(); ++i) { if (i) std::cout << ','; std::cout << times[i]; }
    std::sort(times.begin(), times.end()); std::sort(peaks.begin(), peaks.end());
    const auto median = [](const auto& sorted) {
        const std::size_t middle = sorted.size() / 2;
        return (static_cast<double>(sorted[middle]) + sorted[(sorted.size() - 1) / 2]) / 2;
    };
    std::cout << (toggle.empty() ? "],\"median_ms\":" : "],\"combined_sample_median_ms\":")
              << median(times)
              << ",\"median_peak_mlx_bytes\":" << median(peaks)
              << ",\"output_state_hash\":\"" << hash << "\"";
    if (!toggle.empty()) {
        const auto print_samples = [](const std::vector<double>& samples) {
            for (std::size_t i = 0; i < samples.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << samples[i];
            }
        };
        std::cout << ",\"ab\":{\"switch\":\"" << toggle
                  << "\",\"warmup_samples_per_arm\":" << warmups / 2
                  << ",\"exact_bits\":true,\"control_ms\":[";
        print_samples(control_times);
        std::cout << "],\"candidate_ms\":[";
        print_samples(candidate_times);
        std::cout << "]}";
    }
    std::cout << "}\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
