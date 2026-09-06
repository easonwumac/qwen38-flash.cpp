#include "qwen38/model.hpp"
#include "qwen38/sparse_moe.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <numeric>
#include <mach/mach.h>

namespace {
constexpr std::size_t gib = 1024ULL * 1024ULL * 1024ULL;
using Clock = std::chrono::steady_clock;
void check(int code) { if (code != 0) throw std::runtime_error("MLX budget operation failed"); }
void usage(const char* phase) {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        throw std::runtime_error("cannot observe process footprint");
    std::cout << "phase=" << phase << " footprint_gib=" << double(info.phys_footprint) / gib << std::endl;
    if (info.phys_footprint > 22 * gib) throw std::runtime_error("probe stops early at 22 GiB safety threshold");
}
}

int main(int argc, char** argv) {
    try {
        const bool layer_only = argc == 3 && std::string_view(argv[2]) == "--layer-parity";
        const bool extended = argc == 3 && std::string_view(argv[2]) == "--extended";
        if ((argc != 2 && !layer_only && !extended) || std::getenv("QWEN38_MEMORY_GUARD") == nullptr ||
            std::getenv("MLX_STRICT_MEMORY_LIMIT") == nullptr ||
            std::string_view(std::getenv("MLX_STRICT_MEMORY_LIMIT")) != "1")
            throw std::runtime_error("usage: guarded private-strict probe MODEL_DIRECTORY");
        // Research-only: leave 4 GiB outside the MLX cap for host staging and
        // framework memory. The external guard independently limits the process.
        mlx_set_error_handler([](const char* message, void*) { std::cerr << message << '\n'; }, nullptr, nullptr);
        std::size_t old = 0;
        check(mlx_set_memory_limit(&old, (layer_only ? 3 : 20) * gib));
        check(mlx_set_cache_limit(&old, 0));
        check(mlx_clear_cache());
        if (layer_only) {
          for (const int layer : {0, 47}) {
            qwen38::MlxTensorStore paged(qwen38::ModelManifest::load(argv[1]), 8 * 1024 * 1024);
            qwen38::MlxTensorStore reference(qwen38::ModelManifest::load(argv[1]));
            const auto& config = paged.manifest().config();
            const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".mlp";
            qwen38::SparseMoe a(paged, prefix, 288, 10, 4, 64, true);
            qwen38::SparseMoe b(reference, prefix, 288, 10, 4, 64, true);
            for (const int rows : {1, 4}) {
                std::vector<float> values(static_cast<std::size_t>(rows) * config.hidden_size);
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = std::sin(static_cast<float>(i) * 0.031F);
                auto input = qwen38::MlxArray::from_float32(values,
                    std::vector<int>{1, rows, static_cast<int>(config.hidden_size)}).astype(MLX_BFLOAT16);
                auto actual = a.forward_prefill(input).astype(MLX_FLOAT32).to_float32();
                auto expected = b.forward_prefill(input).astype(MLX_FLOAT32).to_float32();
                if (actual.size() != expected.size()) throw std::runtime_error("parity shape mismatch");
                for (std::size_t i = 0; i < actual.size(); ++i)
                    if (!std::isfinite(actual[i]) || actual[i] != expected[i])
                        throw std::runtime_error("paged real-router/shared-expert parity mismatch");
                std::cout << "layer_parity layer=" << layer << " rows=" << rows << " exact=true evictions="
                          << paged.expert_stats().evictions << std::endl;
            }
            usage("layer_parity");
          }
            return 0;
        }
        const auto load_started = Clock::now();
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]), 16 * gib);
        qwen38::QwenModel model(tensors);
        usage("loaded");
        std::cout << "load_seconds=" << std::chrono::duration<double>(Clock::now() - load_started).count() << std::endl;
        auto state = model.make_state();
        std::uint32_t token = 9419;
        std::vector<double> seconds;
        for (int i = 0; i < (extended ? 64 : 8); ++i) {
            const auto start = Clock::now();
            const auto result = model.greedy_decode(token, state);
            token = result.token;
            seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
            const auto& stats = tensors.expert_stats();
            std::cout << "step=" << i << " token=" << token
                << " seconds=" << seconds.back()
                << " hits=" << stats.hits << " misses=" << stats.misses
                << " evictions=" << stats.evictions << " expert_gib=" << double(stats.resident_bytes) / gib
                << " load_ms=" << tensors.expert_load_ms() << std::endl;
            usage("decode");
        }
        const double steady = std::accumulate(seconds.begin() + 1, seconds.end(), 0.0);
        std::vector<double> sorted(seconds.begin() + 1, seconds.end());
        std::sort(sorted.begin(), sorted.end());
        std::cout << "decode_tps_excluding_first=" << double(seconds.size() - 1) / steady
                  << " step_median_seconds=" << sorted[sorted.size() / 2]
                  << " step_p95_seconds=" << sorted[(sorted.size() - 1) * 95 / 100] << std::endl;
        const auto start = Clock::now();
        const std::vector<std::uint32_t> prompt{9419, 11, 358, 1440, 264, 1937, 13, 198};
        auto prefill_state = model.make_state();
        auto output = model.prefill_chunk_batch(prompt, prefill_state);
        output.eval();
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << "pp_tokens=" << prompt.size() << " pp_seconds=" << elapsed
                  << " pp_tps=" << double(prompt.size()) / elapsed << std::endl;
        usage("prefill");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "paged probe: " << e.what() << std::endl;
        return 1;
    }
}
