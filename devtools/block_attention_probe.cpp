#include "qwen38/mlx_backend.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace qwen38;

int main() try {
    if (!std::getenv("QWEN38_MEMORY_GUARD")) throw std::runtime_error("run through memory guard");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 1024ULL * 1024 * 1024)) throw std::runtime_error("cap failed");
    static_cast<void>(MlxArray::set_cache_limit(16ULL * 1024 * 1024));
    constexpr int rows = 16, dim = 128, block = 4096, count = 16;
    auto make = [](int n, int offset) {
        std::vector<float> data(n * dim);
        for (int i = 0; i < n * dim; ++i)
            data[i] = static_cast<float>((i * 13 + offset * 17) % 251 - 125) / 128.F;
        return MlxArray::from_float32(data, std::array<int, 2>{n, dim});
    };
    auto query = make(rows, 9);
    auto scale = MlxArray::from_float32(std::array<float, 1>{1.F / std::sqrt(float(dim))},
        std::array<int, 1>{1});
    std::vector<MlxArray> keys, values;
    for (int i = 0; i < count; ++i) {
        keys.push_back(make(block, i)); values.push_back(make(block, i + 97));
        keys.back().eval(); values.back().eval();
    }
    auto dense_k = keys.front().share(), dense_v = values.front().share();
    for (int i = 1; i < count; ++i) {
        dense_k = MlxArray::concatenate(dense_k, keys[i], 0);
        dense_v = MlxArray::concatenate(dense_v, values[i], 0);
    }
    dense_k.eval(); dense_v.eval(); query.eval();
    auto dense = [&] {
        auto scores = MlxArray::multiply(MlxArray::matmul(query, dense_k.transpose()), scale);
        return MlxArray::matmul(scores.softmax_axis(-1), dense_v);
    };
    auto segmented = [&] {
        // No concatenated K/V input. Scores remain contiguous for a globally
        // normalized softmax; per-block probabilities then weight each V bank.
        // This is a correctness prototype, not the production QSA kernel.
        auto scores = MlxArray::matmul(query, keys[0].transpose());
        for (int i = 1; i < count; ++i)
            scores = MlxArray::concatenate(scores, MlxArray::matmul(query, keys[i].transpose()), 1);
        auto probabilities = MlxArray::multiply(scores, scale).softmax_axis(-1);
        MlxArray output;
        for (int i = 0; i < count; ++i) {
            auto p = probabilities.slice(std::array<int, 2>{0, i * block},
                std::array<int, 2>{rows, (i + 1) * block}, std::array<int, 2>{1, 1});
            auto partial = MlxArray::matmul(p, values[i]);
            output = i == 0 ? std::move(partial) : MlxArray::add(output, partial);
        }
        return output;
    };
    auto reference = dense().to_float32();
    std::cout << std::unitbuf;
    for (int repeat = 0; repeat < 8; ++repeat) {
        for (int slot = 0; slot < 2; ++slot) {
            const bool blocks = (repeat + slot) % 2 != 0;
            const auto start = std::chrono::steady_clock::now();
            auto output = blocks ? segmented() : dense();
            output.eval();
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            float max_error = 0;
            auto actual = output.to_float32();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite output");
                max_error = std::max(max_error, std::abs(actual[i] - reference[i]));
            }
            if (max_error > 1e-5F) throw std::runtime_error("attention tolerance exceeded");
            std::cout << "{\"repeat\":" << repeat << ",\"blocks\":" << blocks
                << ",\"ms\":" << ms << ",\"max_abs_error\":" << max_error << "}\n";
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
