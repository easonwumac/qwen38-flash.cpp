#include "qwen38/decoder_layer.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/runtime_profile.hpp"
#include "qwen38/token_embedding.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace qwen38;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) try {
    if (argc != 2 || std::getenv("QWEN38_MEMORY_GUARD") == nullptr)
        throw std::runtime_error("usage: guarded qwen38-pp-reduce-layer-ab MODEL");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 3ULL * 1024 * 1024 * 1024) != 0)
        throw std::runtime_error("cannot set allocation limit");
    static_cast<void>(MlxArray::set_cache_limit(64ULL * 1024 * 1024));
    apply_runtime_profile("speed");
    setenv("QWEN38_COMPACT_QMETA", "lossless16", 1);
    setenv("QWEN38_QMETA_PREFILL_CACHE", "0", 1);
    setenv("QWEN38_QMETA_PREFILL_DEFER_TEMPORARY", "1", 1);
    MlxTensorStore tensors(ModelManifest::load(argv[1]));
    const auto& config = tensors.manifest().config();
    DecoderLayer layer(tensors, 0, config);
    const auto w = tensors.tensor("language_model.model.embed_tokens.weight");
    const auto s = tensors.tensor("language_model.model.embed_tokens.scales");
    const auto b = tensors.tensor("language_model.model.embed_tokens.biases");
    for (const int rows : {128, 512}) for (const bool diverse : {false, true}) {
        std::vector<std::uint32_t> tokens(rows), next(rows);
        for (int i = 0; i < rows; ++i) {
            tokens[i] = diverse ? (9419U + 7919U * i) % config.vocabulary_size : 9419U;
            next[i] = diverse ? (104729U + 7919U * i) % config.vocabulary_size : 9419U;
        }
        const auto embed = [&](const auto& ids) {
            return HyperConnection::initialize_stream(embed_token_batch(w, s, b, ids,
                config.vocabulary_size, config.hidden_size,
                static_cast<int>(config.quantization_group_size),
                static_cast<int>(config.quantization_bits)), config.hyper_connection_count);
        };
        auto first_input = embed(tokens), next_input = embed(next);
        const std::array<const MlxArray*, 2> prepared{&first_input, &next_input};
        MlxArray::eval_all(prepared);
        const auto run = [&](bool fused) {
            setenv("QWEN38_PP_ROUTE_REDUCE", fused ? "1" : "0", 1);
            DecoderLayerState state;
            auto first = layer.forward_prefill(first_input.share(), tokens, state);
            auto second = layer.forward_prefill(next_input.share(), next, state);
            const std::array<const MlxArray*, 4> outputs{
                &first, &second, &state.linear_attention.convolution, &state.linear_attention.recurrent};
            MlxArray::eval_all(outputs);
            return std::array<MlxArray, 4>{std::move(first), std::move(second),
                std::move(state.linear_attention.convolution), std::move(state.linear_attention.recurrent)};
        };
        {
            const auto reference = run(false), candidate = run(true);
            for (std::size_t i = 0; i < reference.size(); ++i)
                if (reference[i].astype(MLX_FLOAT32).to_float32() !=
                    candidate[i].astype(MLX_FLOAT32).to_float32())
                    throw std::runtime_error("complete layer output/state parity failed");
        }
        std::array<std::vector<double>, 2> samples;
        for (int repeat = -3; repeat < 21; ++repeat) {
            for (int order = 0; order < 2; ++order) {
                const int mode = (repeat + 4 + order) % 2;
                const auto start = Clock::now();
                static_cast<void>(run(mode != 0));
                const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                if (repeat >= 0) samples[mode].push_back(elapsed);
            }
        }
        std::cout << "{\"rows\":" << rows << ",\"diverse\":" << diverse
                  << ",\"parity\":true,\"samples_ms\":[";
        for (int mode = 0; mode < 2; ++mode) {
            if (mode) std::cout << ',';
            std::cout << '[';
            for (std::size_t i = 0; i < samples[mode].size(); ++i) {
                if (i) std::cout << ',';
                std::cout << samples[mode][i];
            }
            std::cout << ']';
        }
        std::sort(samples[0].begin(), samples[0].end());
        std::sort(samples[1].begin(), samples[1].end());
        std::cout << "],\"control_ms\":" << samples[0][10]
                  << ",\"candidate_ms\":" << samples[1][10] << "}\n";
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
