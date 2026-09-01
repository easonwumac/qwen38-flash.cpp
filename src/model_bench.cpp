#include "qwen38/model.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

std::size_t parse_steps(const char* value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 2 || parsed > 64 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("STEPS must be between 2 and 64");
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY [STEPS]\n";
        return EXIT_FAILURE;
    }
    try {
        const std::size_t steps = argc == 3 ? parse_steps(argv[2]) : 8;
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel model(tensors);
        auto state = model.make_state();
        std::uint32_t token = 9419;
        std::vector<std::uint32_t> tokens;
        std::vector<double> timings;
        tokens.reserve(steps);
        timings.reserve(steps);
        for (std::size_t index = 0; index < steps; ++index) {
            const auto started = std::chrono::steady_clock::now();
            const qwen38::GreedyStep result = model.greedy_decode(token, state);
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
            token = result.token;
            tokens.push_back(token);
        }
        const double sustained_ms =
            std::accumulate(timings.begin() + 1, timings.end(), 0.0);
        std::cout << "{\"tokens\":[";
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << tokens[index];
        }
        std::cout << "],\"step_ms\":[";
        for (std::size_t index = 0; index < timings.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << timings[index];
        }
        std::cout << "],\"sustained_tps\":"
                  << (1000.0 * static_cast<double>(steps - 1) / sustained_ms)
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-model-bench: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
