#include "qwen38/model.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY [--trace]\n";
        return EXIT_FAILURE;
    }
    try {
        const auto load_started = std::chrono::steady_clock::now();
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel model(tensors);
        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - load_started).count();
        auto state = model.make_state();
        const auto first_started = std::chrono::steady_clock::now();
        qwen38::GreedyStep first;
        std::vector<double> trace;
        std::vector<double> trace_ms;
        if (argc == 3 && std::string(argv[2]) == "--trace") {
            const std::vector<float> logits =
                model.trace_decode(9419, state, trace, trace_ms).astype(MLX_FLOAT32).to_float32();
            const auto best = std::ranges::max_element(logits);
            first.token = static_cast<std::uint32_t>(std::distance(logits.begin(), best));
            first.logit = *best;
        } else {
            first = model.greedy_decode(9419, state);
        }
        const double first_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - first_started).count();
        const auto second_started = std::chrono::steady_clock::now();
        qwen38::GreedyStep second;
        std::vector<double> second_trace;
        std::vector<double> second_trace_ms;
        if (argc == 3 && std::string(argv[2]) == "--trace") {
            const std::vector<float> logits = model.trace_decode(
                first.token, state, second_trace, second_trace_ms).astype(MLX_FLOAT32).to_float32();
            const auto best = std::ranges::max_element(logits);
            second.token = static_cast<std::uint32_t>(std::distance(logits.begin(), best));
            second.logit = *best;
        } else {
            second = model.greedy_decode(first.token, state);
        }
        const double second_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - second_started).count();
        std::cout << "{\"layers\":" << model.layer_count()
                  << ",\"first_token\":" << first.token
                  << ",\"first_logit\":" << first.logit
                  << ",\"second_token\":" << second.token
                  << ",\"second_logit\":" << second.logit
                  << ",\"load_ms\":" << load_ms
                  << ",\"first_ms\":" << first_ms
                  << ",\"second_ms\":" << second_ms
                  << ",\"second_tps\":" << (1000.0 / second_ms)
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        if (!trace.empty()) {
            std::cout << "{\"layer_checksums\":[";
            for (std::size_t index = 0; index < trace.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << trace[index];
            }
            std::cout << "]}\n";
            std::cout << "{\"first_layer_ms\":[";
            for (std::size_t index = 0; index < trace_ms.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << trace_ms[index];
            }
            std::cout << "],\"second_layer_ms\":[";
            for (std::size_t index = 0; index < second_trace_ms.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << second_trace_ms[index];
            }
            std::cout << "]}\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-model-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
