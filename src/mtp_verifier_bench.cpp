#include "qwen38/model.hpp"
#include "qwen38/mtp_verifier.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

struct Sample {
    double milliseconds{0.0};
    std::vector<std::uint32_t> tokens;
};

template <typename Verify>
Sample measure(Verify&& verify) {
    const auto started = std::chrono::steady_clock::now();
    qwen38::MtpTargetVerification result = verify();
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    Sample sample{.milliseconds = elapsed, .tokens = {}};
    sample.tokens.reserve(result.rows.size());
    for (const qwen38::MtpTargetVerifyRow& row : result.rows) {
        sample.tokens.push_back(row.greedy.token);
    }
    return sample;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model verifier benchmarks must run through devtools/memory_guard.py");
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel model(tensors);
        const bool seeded_origin = std::getenv("QWEN38_VERIFY_SEEDED_ORIGIN") != nullptr;
        std::size_t depth = 2;
        if (const char* raw_depth = std::getenv("QWEN38_VERIFY_DEPTH")) {
            std::size_t parsed = 0;
            depth = std::stoul(raw_depth, &parsed);
            if (raw_depth[parsed] != '\0' || depth < 2 || depth > 4) {
                throw std::runtime_error("QWEN38_VERIFY_DEPTH must be 2, 3, or 4");
            }
        }
        const std::vector<std::uint32_t> token_fixture = seeded_origin
            ? std::vector<std::uint32_t>{271, 40, 11, 9419}
            : std::vector<std::uint32_t>{11, 271, 40, 9419};
        const std::vector<std::uint32_t> drafts(
            token_fixture.begin(), token_fixture.begin() + static_cast<std::ptrdiff_t>(depth));
        qwen38::ModelDecodeState origin = model.make_state();
        const std::uint32_t current = seeded_origin ? 11 : 9419;
        if (seeded_origin) model.consume_decode(9419, origin);
        const auto serial = [&] {
            return qwen38::verify_mtp_target_serial_oracle(model, current, drafts, origin);
        };
        const auto layer_major = [&] {
            return qwen38::verify_mtp_target_layer_major_reference(model, current, drafts, origin);
        };

        static_cast<void>(measure(serial));
        static_cast<void>(setenv("QWEN38_BATCH_VERIFY_HEAD", "0", 1));
        static_cast<void>(measure(layer_major));
        const Sample serial_a = measure(serial);
        const Sample control_a = measure(layer_major);
        static_cast<void>(setenv("QWEN38_BATCH_VERIFY_HEAD", "1", 1));
        static_cast<void>(measure(layer_major));
        const Sample candidate_a = measure(layer_major);
        const Sample candidate_b = measure(layer_major);
        static_cast<void>(setenv("QWEN38_BATCH_VERIFY_HEAD", "0", 1));
        const Sample control_b = measure(layer_major);
        const Sample serial_b = measure(serial);
        if (serial_a.tokens != control_a.tokens || serial_a.tokens != control_b.tokens ||
            serial_a.tokens != candidate_a.tokens || serial_a.tokens != candidate_b.tokens ||
            serial_a.tokens != serial_b.tokens) {
            throw std::runtime_error("interleaved verifier benchmark lost token parity");
        }
        const double serial_ms = (serial_a.milliseconds + serial_b.milliseconds) / 2.0;
        const double control_ms = (control_a.milliseconds + control_b.milliseconds) / 2.0;
        const double candidate_ms =
            (candidate_a.milliseconds + candidate_b.milliseconds) / 2.0;
        static_cast<void>(setenv("QWEN38_BATCH_VERIFY_HEAD", "1", 1));
        std::vector<double> layer_ms;
        double head_ms = 0.0;
        // Production groups 16 layers behind each barrier. That is the right
        // latency path, but attributing the whole synchronization to layers
        // 15/31/47 makes the per-layer profile misleading. Profile a separate
        // stride-1 pass after all authoritative latency samples instead.
        static_cast<void>(setenv("QWEN38_VERIFY_BARRIER_STRIDE", "1", 1));
        std::vector<std::uint32_t> profile_tokens{current};
        profile_tokens.insert(profile_tokens.end(), drafts.begin(), drafts.end());
        static_cast<void>(model.forward_verify_layer_major_reference(
            profile_tokens, origin, &layer_ms, &head_ms));
        double linear_ms = 0.0;
        double full_ms = 0.0;
        for (std::size_t layer = 0; layer < layer_ms.size(); ++layer) {
            (layer + 1) % 4 == 0 ? full_ms += layer_ms[layer] : linear_ms += layer_ms[layer];
        }
        const auto slowest = std::max_element(layer_ms.begin(), layer_ms.end());
        std::cout << "{\"depth\":" << depth << ",\"rows\":" << depth + 1
                  << ",\"serial_ms\":" << serial_ms
                  << ",\"separate_head_ms\":" << control_ms
                  << ",\"batched_head_ms\":" << candidate_ms
                  << ",\"head_speedup\":" << control_ms / candidate_ms
                  << ",\"serial_speedup\":" << serial_ms / candidate_ms
                  << ",\"profile\":{\"barrier_stride\":1,\"linear_layers_ms\":" << linear_ms
                  << ",\"full_layers_ms\":" << full_ms
                  << ",\"head_ms\":" << head_ms
                  << ",\"slowest_layer\":"
                  << std::distance(layer_ms.begin(), slowest)
                  << ",\"slowest_layer_ms\":" << *slowest
                  << ",\"layers_ms\":[";
        for (std::size_t layer = 0; layer < layer_ms.size(); ++layer) {
            if (layer != 0) std::cout << ',';
            std::cout << layer_ms[layer];
        }
        std::cout << "]}"
                  << ",\"parity\":true}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-verifier-bench: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
