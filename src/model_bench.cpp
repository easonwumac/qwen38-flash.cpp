#include "qwen38/model.hpp"
#include "qwen38/runtime_profile.hpp"

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

std::size_t parse_batch_width(const char* value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 4) {
        throw std::runtime_error("BATCH must be between 1 and 4");
    }
    return static_cast<std::size_t>(parsed);
}

std::uint32_t greedy_token(const qwen38::MlxArray& logits) {
    return logits.argmax_all().item_uint32();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY [STEPS] [BATCH]\n";
        return EXIT_FAILURE;
    }
    try {
        if (const char* profile = std::getenv("QWEN38_BENCH_PROFILE")) {
            qwen38::apply_runtime_profile(profile);
        }
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model benchmarks must run through devtools/memory_guard.py");
        }
        const std::size_t steps = argc >= 3 ? parse_steps(argv[2]) : 8;
        const std::size_t batch_width = argc == 4 ? parse_batch_width(argv[3]) : 1;
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel model(tensors);
        if (batch_width > 1) {
            std::vector<qwen38::ModelDecodeState> serial_states;
            std::vector<qwen38::ModelDecodeState> batched_states;
            std::vector<std::uint32_t> serial_tokens;
            std::vector<std::uint32_t> batched_tokens;
            serial_states.reserve(batch_width);
            batched_states.reserve(batch_width);
            serial_tokens.reserve(batch_width);
            batched_tokens.reserve(batch_width);
            for (std::size_t row = 0; row < batch_width; ++row) {
                serial_states.push_back(model.make_state());
                batched_states.push_back(model.make_state());
                const std::uint32_t initial = static_cast<std::uint32_t>(9419 + row);
                serial_tokens.push_back(model.greedy_decode(
                    initial, serial_states.back()).token);
                batched_tokens.push_back(model.greedy_decode(
                    initial, batched_states.back()).token);
            }
            std::vector<double> serial_ms;
            std::vector<double> batched_ms;
            bool token_parity = true;
            std::size_t first_mismatch_step = steps;
            std::size_t first_mismatch_row = batch_width;
            for (std::size_t step = 0; step < steps; ++step) {
                const auto serial_started = std::chrono::steady_clock::now();
                for (std::size_t row = 0; row < batch_width; ++row) {
                    serial_tokens[row] = model.greedy_decode(
                        serial_tokens[row], serial_states[row]).token;
                }
                serial_ms.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - serial_started).count());

                std::vector<qwen38::ModelDecodeState*> state_ptrs;
                state_ptrs.reserve(batch_width);
                for (qwen38::ModelDecodeState& state : batched_states) {
                    state_ptrs.push_back(&state);
                }
                const auto batched_started = std::chrono::steady_clock::now();
                std::vector<qwen38::TargetDecodeStep> outputs =
                    model.forward_decode_capture_multi(batched_tokens, state_ptrs);
                for (std::size_t row = 0; row < batch_width; ++row) {
                    batched_tokens[row] = greedy_token(outputs[row].logits);
                    if (batched_tokens[row] != serial_tokens[row] && token_parity) {
                        first_mismatch_step = step;
                        first_mismatch_row = row;
                        token_parity = false;
                    }
                }
                batched_ms.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - batched_started).count());
            }
            const std::size_t warmup = std::min<std::size_t>(2, steps - 1);
            const double serial_total = std::accumulate(
                serial_ms.begin() + static_cast<std::ptrdiff_t>(warmup),
                serial_ms.end(), 0.0);
            const double batched_total = std::accumulate(
                batched_ms.begin() + static_cast<std::ptrdiff_t>(warmup),
                batched_ms.end(), 0.0);
            const std::size_t measured_steps = steps - warmup;
            const double measured_tokens =
                static_cast<double>(measured_steps * batch_width);
            std::cout << "{\"batch_width\":" << batch_width
                      << ",\"steps\":" << steps
                      << ",\"token_parity\":" << (token_parity ? "true" : "false")
                      << ",\"first_mismatch_step\":" << first_mismatch_step
                      << ",\"first_mismatch_row\":" << first_mismatch_row
                      << ",\"serial_aggregate_tps\":"
                      << 1000.0 * measured_tokens / serial_total
                      << ",\"batched_aggregate_tps\":"
                      << 1000.0 * measured_tokens / batched_total
                      << ",\"speedup\":" << serial_total / batched_total
                      << ",\"serial_final_tokens\":[";
            for (std::size_t row = 0; row < batch_width; ++row) {
                if (row != 0) std::cout << ',';
                std::cout << serial_tokens[row];
            }
            std::cout << "],\"batched_final_tokens\":[";
            for (std::size_t row = 0; row < batch_width; ++row) {
                if (row != 0) std::cout << ',';
                std::cout << batched_tokens[row];
            }
            std::cout << "]}\n";
            return token_parity ? EXIT_SUCCESS : EXIT_FAILURE;
        }
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
