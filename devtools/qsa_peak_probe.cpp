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
#include <iostream>
#include <stdexcept>

using namespace qwen38;
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
        throw std::runtime_error("usage: guarded qwen38-qsa-peak-probe MODEL CONTEXT MODE(0..4)");
    const int context = std::stoi(argv[2]);
    if ((context != 32768 && context != 65536 && context != 131072) ||
        (std::string_view(argv[3]).size() != 1 || argv[3][0] < '0' || argv[3][0] > '4'))
        throw std::runtime_error("invalid bounded probe configuration");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 3ULL * 1024 * 1024 * 1024) != 0)
        throw std::runtime_error("allocation limit failed");
    static_cast<void>(MlxArray::set_cache_limit(64ULL * 1024 * 1024));
    apply_runtime_profile("speed");
    setenv("QWEN38_QSA_PACKED_PREFILL", "1", 1);
    setenv("QWEN38_QSA_PACKED_MIN_TOKENS", "0", 1);
    setenv("QWEN38_QSA_SCORE_REDUCE", argv[3], 1);
    unsetenv("QWEN38_QSA_TILED_SCORES");
    if (argv[3][0] >= '2') {
        const char* modes[]{"128", "256", "async"};
        setenv("QWEN38_QSA_TILED_SCORES", modes[argv[3][0] - '2'], 1);
    }
    MlxTensorStore tensors(ModelManifest::load(argv[1]));
    const auto& c = tensors.manifest().config();
    SelfAttention layer(tensors, "language_model.model.layers.3.self_attn", c);
    std::vector<std::uint32_t> tokens(512);
    for (std::size_t i = 0; i < tokens.size(); ++i)
        tokens[i] = static_cast<std::uint32_t>((9419 + 7919 * i) % c.vocabulary_size);
    auto embedding = embed_token_batch(tensors.tensor("language_model.model.embed_tokens.weight"),
        tensors.tensor("language_model.model.embed_tokens.scales"),
        tensors.tensor("language_model.model.embed_tokens.biases"), tokens,
        c.vocabulary_size, c.hidden_size, static_cast<int>(c.quantization_group_size),
        static_cast<int>(c.quantization_bits));
    HyperConnection mixer(tensors, "language_model.model.layers.3.attn_hyper_connection",
        c.hidden_size, c.hyper_connection_count, c.quantization_bits,
        c.quantization_group_size, static_cast<float>(c.rms_norm_epsilon), true);
    auto input = mixer.read(HyperConnection::initialize_stream(embedding, c.hyper_connection_count)).mixed;
    input.eval();
    SelfAttentionState origin;
    origin.keys = history({1, 2, context, 256}, 1);
    origin.values = history({1, 2, context, 256}, 7);
    origin.qsa_raw_keys = history({1, context, 128}, 13);
    origin.qsa_pooled_keys = history({1, context / 4, 128}, 19);
    origin.token_count = context; origin.qsa_pooled_count = context / 4;
    std::vector<double> times;
    std::vector<std::size_t> peaks;
    std::uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < 14; ++i) {
        SelfAttentionState state;
        state.keys = origin.keys.share(); state.values = origin.values.share();
        state.qsa_raw_keys = origin.qsa_raw_keys.share();
        state.qsa_pooled_keys = origin.qsa_pooled_keys.share();
        state.token_count = origin.token_count; state.qsa_pooled_count = origin.qsa_pooled_count;
        if (mlx_reset_peak_memory() != 0) throw std::runtime_error("peak reset failed");
        const auto start = std::chrono::steady_clock::now();
        auto output = layer.forward_prefill(input, state);
        const std::array<const MlxArray*, 5> evaluated{
            &output, &state.keys, &state.values, &state.qsa_raw_keys, &state.qsa_pooled_keys};
        MlxArray::eval_all(evaluated);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::size_t peak{};
        if (mlx_get_peak_memory(&peak) != 0) throw std::runtime_error("peak query failed");
        if (i >= 3) { times.push_back(ms); peaks.push_back(peak); }
        if (i == 13) for (const auto* value : evaluated)
            for (float x : value->astype(MLX_FLOAT32).to_float32()) {
                if (!std::isfinite(x)) throw std::runtime_error("nonfinite state/output");
                hash ^= std::bit_cast<std::uint32_t>(x); hash *= 1099511628211ULL;
            }
    }
    std::cout << "{\"synthetic_history\":true,\"context\":" << context
              << ",\"rows\":512,\"mode\":" << argv[3] << ",\"samples_ms\":[";
    for (std::size_t i = 0; i < times.size(); ++i) { if (i) std::cout << ','; std::cout << times[i]; }
    std::sort(times.begin(), times.end()); std::sort(peaks.begin(), peaks.end());
    std::cout << "],\"median_ms\":" << times[5] << ",\"median_peak_mlx_bytes\":" << peaks[5]
              << ",\"output_state_hash\":\"" << hash << "\"}\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
