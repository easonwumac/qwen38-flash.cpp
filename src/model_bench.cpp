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
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model benchmarks must run through devtools/memory_guard.py");
        }
        const std::size_t steps = argc == 3 ? parse_steps(argv[2]) : 8;
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel model(tensors);
        auto state = model.make_state();
        std::uint32_t token = 9419;
        std::vector<std::uint32_t> tokens;
        std::vector<double> timings;
        const char* fixed_input_environment = std::getenv("QWEN38_FIXED_INPUT");
        const bool fixed_input = fixed_input_environment != nullptr &&
            std::string_view(fixed_input_environment) == "1";
        tokens.reserve(steps);
        timings.reserve(steps);
        for (std::size_t index = 0; index < steps; ++index) {
            const auto started = std::chrono::steady_clock::now();
            const qwen38::GreedyStep result = model.greedy_decode(token, state);
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
            token = fixed_input ? 9419 : result.token;
            tokens.push_back(token);
        }
        std::vector<double> profile_layer_checksums;
        std::vector<double> profile_layer_ms;
        double profile_total_ms = 0.0;
        if (const char* profile = std::getenv("QWEN38_PROFILE_DECODE");
            profile != nullptr && std::string_view(profile) == "1") {
            const auto profile_started = std::chrono::steady_clock::now();
            static_cast<void>(model.trace_decode(
                token, state, profile_layer_checksums, profile_layer_ms)
                .astype(MLX_FLOAT32).to_float32());
            profile_total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - profile_started).count();
        }
        const double sustained_ms =
            std::accumulate(timings.begin() + 1, timings.end(), 0.0);
        const double steady_ms = timings.size() > 2
            ? std::accumulate(timings.begin() + 2, timings.end(), 0.0)
            : sustained_ms;
        const std::size_t steady_steps = timings.size() > 2 ? timings.size() - 2 : steps - 1;
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
                  << ",\"steady_tps\":"
                  << (1000.0 * static_cast<double>(steady_steps) / steady_ms)
                  << ",\"open_shards\":" << tensors.open_shard_count();
        if (!profile_layer_ms.empty()) {
            std::cout << ",\"profile_total_ms\":" << profile_total_ms
                      << ",\"profile_layer_ms\":[";
            for (std::size_t index = 0; index < profile_layer_ms.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << profile_layer_ms[index];
            }
            std::cout << ']';
        }
        std::cout << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-model-bench: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
