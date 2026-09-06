#include "qwen38/model.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/runtime_profile.hpp"
#include "qwen38/token_embedding.hpp"
#include <libproc.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace qwen38;
using Clock = std::chrono::steady_clock;

std::uint64_t footprint() {
    rusage_info_v4 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
        reinterpret_cast<rusage_info_t*>(&info)) != 0) throw std::runtime_error("footprint unavailable");
    return info.ri_phys_footprint;
}

int main(int argc, char** argv) try {
    if ((argc != 3 && argc != 4) || std::getenv("QWEN38_MEMORY_GUARD") == nullptr ||
        (std::string_view(argv[2]) != "0" && std::string_view(argv[2]) != "1"))
        throw std::runtime_error("usage: guarded qwen38-pp-reduce-group-probe MODEL 0|1 [barrier=1|2|4|8]");
    const int barrier = argc == 4 ? std::stoi(argv[3]) : 8;
    if (barrier != 1 && barrier != 2 && barrier != 4 && barrier != 8)
        throw std::runtime_error("invalid barrier stride");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 10ULL * 1024 * 1024 * 1024) != 0)
        throw std::runtime_error("allocation limit failed");
    static_cast<void>(MlxArray::set_cache_limit(64ULL * 1024 * 1024));
    apply_runtime_profile("speed");
    setenv("QWEN38_COMPACT_QMETA", "lossless16", 1);
    setenv("QWEN38_QMETA_PREFILL_CACHE", "0", 1);
    setenv("QWEN38_QMETA_PREFILL_DEFER_TEMPORARY", "1", 1);
    setenv("QWEN38_PP_ROUTE_REDUCE", argv[2], 1);
    MlxTensorStore tensors(ModelManifest::load(argv[1]));
    const auto& config = tensors.manifest().config();
    constexpr std::size_t count = 8, rows = 512, chunks = 4;
    std::vector<std::unique_ptr<DecoderLayer>> layers;
    for (std::size_t i = 0; i < count; ++i)
        layers.push_back(std::make_unique<DecoderLayer>(tensors, i, config));
    auto w = tensors.tensor("language_model.model.embed_tokens.weight");
    auto s = tensors.tensor("language_model.model.embed_tokens.scales");
    auto b = tensors.tensor("language_model.model.embed_tokens.biases");
    std::vector<double> times;
    std::vector<std::size_t> peaks;
    std::vector<std::uint64_t> physical;
    std::uint64_t hash = 1469598103934665603ULL;
    std::cout << std::unitbuf;
    for (int repeat = 0; repeat < 8; ++repeat) {
        ModelDecodeState state(count);
        MlxArray output;
        if (mlx_reset_peak_memory() != 0) throw std::runtime_error("reset peak failed");
        const auto start = Clock::now();
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            std::vector<std::uint32_t> tokens(rows);
            for (std::size_t i = 0; i < rows; ++i)
                tokens[i] = static_cast<std::uint32_t>((9419 + 7919 * (chunk * rows + i)) % config.vocabulary_size);
            output = HyperConnection::initialize_stream(embed_token_batch(w, s, b, tokens,
                config.vocabulary_size, config.hidden_size,
                static_cast<int>(config.quantization_group_size),
                static_cast<int>(config.quantization_bits)), config.hyper_connection_count);
            for (std::size_t i = 0; i < count; ++i) {
                output = layers[i]->forward_prefill(std::move(output), tokens, state.layers[i]);
                if ((i + 1) % static_cast<std::size_t>(barrier) == 0)
                    eval_with_decode_state(output, state);
            }
            state.token_count += rows;
        }
        for (std::size_t i = 0; i < count; ++i) layers[i]->materialize_speculative_state(state.layers[i]);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        std::size_t peak{}, active{};
        if (mlx_get_peak_memory(&peak) != 0 || mlx_get_active_memory(&active) != 0)
            throw std::runtime_error("memory query failed");
        const auto phys = footprint();
        std::cout << "{\"repeat\":" << repeat << ",\"ms\":" << ms
                  << ",\"peak_mlx_bytes\":" << peak << ",\"active_mlx_bytes\":" << active
                  << ",\"physical_bytes\":" << phys << "}\n";
        if (repeat >= 2) { times.push_back(ms); peaks.push_back(peak); physical.push_back(phys); }
        if (repeat == 7) {
            const auto append = [&](const MlxArray& value) {
                if (!value.get().ctx) return;
                for (float element : value.astype(MLX_FLOAT32).to_float32()) {
                    if (!std::isfinite(element)) throw std::runtime_error("nonfinite output/state");
                    hash ^= std::bit_cast<std::uint32_t>(element);
                    hash *= 1099511628211ULL;
                }
            };
            append(output);
            for (const auto& layer : state.layers) {
                append(layer.linear_attention.convolution); append(layer.linear_attention.recurrent);
                append(layer.full_attention.keys); append(layer.full_attention.values);
                append(layer.full_attention.qsa_raw_keys); append(layer.full_attention.qsa_pooled_keys);
                append(layer.ple.convolution);
            }
        }
    }
    std::sort(times.begin(), times.end()); std::sort(peaks.begin(), peaks.end());
    std::sort(physical.begin(), physical.end());
    std::cout << "{\"fused\":" << argv[2] << ",\"barrier\":" << barrier << ",\"layers\":8,\"tokens\":2048"
              << ",\"median_ms\":" << (times[2] + times[3]) / 2
              << ",\"median_peak_mlx_bytes\":" << (peaks[2] + peaks[3]) / 2
              << ",\"median_physical_bytes\":" << (physical[2] + physical[3]) / 2
              << ",\"output_state_hash\":\"" << hash << "\"}\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
