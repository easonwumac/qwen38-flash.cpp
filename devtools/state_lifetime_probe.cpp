#include "qwen38/mtp_verifier.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace qwen38;

int main(int argc, char** argv) try {
    if (argc != 2 || !std::getenv("QWEN38_MEMORY_GUARD"))
        throw std::runtime_error("usage: guarded state-lifetime-probe retain|release|concat|reserve|blocks");
    const std::string_view mode(argv[1]);
    if (mode != "retain" && mode != "release" && mode != "concat" && mode != "reserve" && mode != "blocks")
        throw std::runtime_error("invalid mode");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 1024ULL * 1024 * 1024))
        throw std::runtime_error("memory cap failed");
    static_cast<void>(MlxArray::set_cache_limit(16ULL * 1024 * 1024));
    if (mode == "retain" || mode == "release") {
        for (std::size_t accepted = 0; accepted <= 4; ++accepted) {
            MtpTargetVerification verification{.draft_count = 4, .rows = {}};
            for (std::size_t i = 0; i <= 4; ++i) {
                ModelDecodeState state(1);
                state.token_count = 10 + i;
                state.layers[0].full_attention.keys = MlxArray::zeros(
                    std::array<int, 1>{8 * 1024 * 1024}, MLX_FLOAT32);
                state.layers[0].full_attention.keys.eval();
                verification.rows.push_back({.greedy = {}, .final_mixed = {},
                    .pre_mixer_stream = {}, .state_after = std::move(state)});
            }
            std::size_t before{}, after{};
            mlx_get_active_memory(&before);
            ModelDecodeState destination(0);
            if (mode == "release")
                commit_mtp_target_verification(std::move(verification), accepted, destination);
            else destination = std::move(verification.rows[accepted].state_after);
            destination.layers[0].full_attention.keys.eval();
            const auto fence = mlx_default_gpu_stream_new();
            const int fence_status = mlx_synchronize(fence);
            mlx_stream_free(fence);
            if (fence_status) throw std::runtime_error("completion fence failed");
            mlx_get_active_memory(&after);
            if (destination.token_count != 10 + accepted) throw std::runtime_error("bad commit");
            std::cout << "{\"mode\":\"" << mode << "\",\"accepted\":" << accepted
                << ",\"before_bytes\":" << before << ",\"after_bytes\":" << after << "}\n";
        }
        return 0;
    }
    // One synthetic K/V-like history. Measures append only, not attention.
    // Keep an origin snapshot, as speculative verification/prefix reuse does.
    const auto stream = mlx_default_gpu_stream_new();
    for (int repeat = 0; repeat < 7; ++repeat) {
        constexpr int start_rows = 32768, chunks = 32, chunk = 512, width = 256;
        auto state = MlxArray::zeros(std::array<int, 2>{
            mode == "reserve" ? start_rows + chunks * chunk : start_rows, width}, MLX_BFLOAT16);
        state.eval();
        auto origin = state.share();
        std::vector<MlxArray> blocks;
        blocks.reserve(chunks);
        mlx_reset_peak_memory();
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < chunks; ++i) {
            auto update = MlxArray::from_float32(std::vector<float>(chunk * width, 1.F),
                std::array<int, 2>{chunk, width}).astype(MLX_BFLOAT16);
            update.eval();
            if (mode == "blocks") {
                // Independent output buffer per chunk, not 32 aliases of one
                // update. A future attention kernel must consume these blocks
                // directly; concatenating them would restore the copy cost.
                blocks.push_back(std::move(update));
            } else if (mode == "concat") state = MlxArray::concatenate(state, update, 0);
            else {
                mlx_array output = mlx_array_new();
                const std::array<int, 2> begin{start_rows + i * chunk, 0};
                const std::array<int, 2> end{start_rows + (i + 1) * chunk, width};
                const std::array<int, 2> stride{1, 1};
                const int status = mlx_slice_update(&output, state.get(), update.get(),
                    begin.data(), 2, end.data(), 2, stride.data(), 2, stream);
                MlxArray next(output);
                if (status) throw std::runtime_error("slice update failed");
                state = std::move(next);
            }
            state.eval();
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::size_t peak{}; mlx_get_peak_memory(&peak);
        const auto values = state.astype(MLX_FLOAT32).to_float32();
        for (std::size_t i = 0; i < values.size(); ++i)
            if (values[i] != (i < start_rows * width ? 0.F : 1.F))
                throw std::runtime_error("append parity failed");
        if (mode == "blocks") {
            if (blocks.size() != chunks) throw std::runtime_error("missing blocks");
            for (const auto& block : blocks)
                for (float value : block.astype(MLX_FLOAT32).to_float32())
                    if (value != 1.F) throw std::runtime_error("block parity failed");
        }
        for (float value : origin.astype(MLX_FLOAT32).to_float32())
            if (value != 0.F) throw std::runtime_error("origin was mutated");
        std::cout << "{\"mode\":\"" << mode << "\",\"repeat\":" << repeat
            << ",\"ms\":" << ms << ",\"peak_bytes\":" << peak << ",\"parity\":true}\n";
    }
    mlx_stream_free(stream);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
