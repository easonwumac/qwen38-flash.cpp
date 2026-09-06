#include "qwen38/self_attention.hpp"
#include "qwen38/token_embedding.hpp"
#include "qwen38/runtime_profile.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace qwen38;
using Clock = std::chrono::steady_clock;
struct Packed { MlxArray w, s, b; };
Packed pack8(const MlxArray& value) {
    auto vector = mlx_vector_array_new();
    const auto stream = mlx_default_gpu_stream_new();
    const int status = mlx_quantize(&vector, value.get(), {.value=64, .has_value=true}, {.value=8, .has_value=true},
        "affine", mlx_array{}, stream);
    mlx_stream_free(stream);
    if (status) { mlx_vector_array_free(vector); throw std::runtime_error("quantize failed"); }
    std::array<mlx_array, 3> raw{mlx_array_new(), mlx_array_new(), mlx_array_new()};
    const int extract = mlx_vector_array_get(&raw[0], vector, 0) |
        mlx_vector_array_get(&raw[1], vector, 1) | mlx_vector_array_get(&raw[2], vector, 2);
    mlx_vector_array_free(vector);
    Packed result{MlxArray(raw[0]), MlxArray(raw[1]), MlxArray(raw[2])};
    if (extract) throw std::runtime_error("quantize tuple failed");
    const std::array<const MlxArray*, 3> outputs{&result.w, &result.s, &result.b};
    MlxArray::eval_all(outputs);
    return result;
}
std::size_t bytes(const Packed& p) {
    return mlx_array_nbytes(p.w.get()) + mlx_array_nbytes(p.s.get()) + mlx_array_nbytes(p.b.get());
}
int main(int argc, char** argv) try {
    if ((argc != 2 && argc != 3) || !std::getenv("QWEN38_MEMORY_GUARD"))
        throw std::runtime_error("usage: guarded kv8-layer-probe MODEL [8192|65536]");
    const std::size_t context = argc == 3 ? std::stoull(argv[2]) : 8192;
    if (context != 8192 && context != 65536) throw std::runtime_error("unsupported context");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 3ULL * 1024 * 1024 * 1024)) throw std::runtime_error("cap failed");
    static_cast<void>(MlxArray::set_cache_limit(64ULL * 1024 * 1024));
    apply_runtime_profile("speed");
    MlxTensorStore tensors(ModelManifest::load(argv[1]));
    const auto& c = tensors.manifest().config();
    SelfAttention attention(tensors, "language_model.model.layers.3.self_attn", c);
    const auto embedding = [&](std::size_t offset, std::size_t rows) {
        std::vector<std::uint32_t> tokens(rows);
        for (std::size_t i = 0; i < rows; ++i) tokens[i] = (9419 + 7919 * (i + offset)) % c.vocabulary_size;
        return embed_token_batch(tensors.tensor("language_model.model.embed_tokens.weight"),
            tensors.tensor("language_model.model.embed_tokens.scales"),
            tensors.tensor("language_model.model.embed_tokens.biases"), tokens,
            c.vocabulary_size, c.hidden_size, c.quantization_group_size, c.quantization_bits);
    };
    SelfAttentionState origin;
    // Real projection weights, but isolated-layer embedding inputs: these are
    // not layer-3 activations produced by the complete upstream model.
    for (std::size_t offset = 0; offset < context; offset += 512) {
        auto out = attention.forward_prefill(embedding(offset, 512), origin);
        const std::array<const MlxArray*, 3> outputs{&out, &origin.keys, &origin.values};
        MlxArray::eval_all(outputs);
    }
    const auto packing_started = Clock::now();
    auto k = pack8(origin.keys), v = pack8(origin.values);
    const double pack_ms = std::chrono::duration<double, std::milli>(Clock::now() - packing_started).count();
    auto input = embedding(context, 16); input.eval();
    std::vector<float> reference;
    std::cout << std::unitbuf << "{\"context\":" << context << ",\"bf16_kv_bytes\":" <<
        mlx_array_nbytes(origin.keys.get()) + mlx_array_nbytes(origin.values.get())
        << ",\"q8_kv_bytes\":" << bytes(k) + bytes(v) << ",\"pack_ms\":" << pack_ms << "}\n";
    for (int repeat = 0; repeat < 5; ++repeat) {
        for (int mode = 0; mode < 2; ++mode) {
            const auto started = Clock::now();
            SelfAttentionState state;
            state.token_count = origin.token_count;
            state.qsa_pooled_count = origin.qsa_pooled_count;
            state.qsa_raw_keys = origin.qsa_raw_keys.share();
            state.qsa_pooled_keys = origin.qsa_pooled_keys.share();
            state.keys = mode ? MlxArray::dequantize(k.w, k.s, k.b, 64, 8) : origin.keys.share();
            state.values = mode ? MlxArray::dequantize(v.w, v.s, v.b, 64, 8) : origin.values.share();
            auto out = attention.forward_prefill(input, state);
            out.eval();
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            auto actual = out.astype(MLX_FLOAT32).to_float32();
            if (repeat == 0 && mode == 0) reference = actual;
            double error2=0, norm2=0, max_error=0;
            for (std::size_t i=0; i<actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite attention output");
                const double difference = double(actual[i]) - reference[i];
                error2 += difference*difference; norm2 += double(reference[i])*reference[i];
                max_error = std::max(max_error, std::abs(difference));
            }
            std::cout << "{\"repeat\":" << repeat << ",\"q8\":" << mode
                << ",\"ms\":" << ms << ",\"max_abs_error\":" << max_error
                << ",\"relative_l2\":" << std::sqrt(error2 / (norm2 + 1e-30)) << "}\n";
        }
    }
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
