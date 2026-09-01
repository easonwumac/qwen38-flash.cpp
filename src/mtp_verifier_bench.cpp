#include "qwen38/model.hpp"
#include "qwen38/mtp_verifier.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
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
        const std::vector<std::uint32_t> drafts{11, 271};
        const qwen38::ModelDecodeState origin = model.make_state();
        const auto serial = [&] {
            return qwen38::verify_mtp_target_serial_oracle(model, 9419, drafts, origin);
        };
        const auto layer_major = [&] {
            return qwen38::verify_mtp_target_layer_major_reference(
                model, 9419, drafts, origin);
        };

        static_cast<void>(measure(serial));
        static_cast<void>(measure(layer_major));
        const Sample serial_a = measure(serial);
        const Sample layer_a = measure(layer_major);
        const Sample layer_b = measure(layer_major);
        const Sample serial_b = measure(serial);
        if (serial_a.tokens != layer_a.tokens || serial_a.tokens != layer_b.tokens ||
            serial_a.tokens != serial_b.tokens) {
            throw std::runtime_error("interleaved verifier benchmark lost token parity");
        }
        const double serial_ms = (serial_a.milliseconds + serial_b.milliseconds) / 2.0;
        const double layer_ms = (layer_a.milliseconds + layer_b.milliseconds) / 2.0;
        std::cout << "{\"depth\":2,\"rows\":3,\"serial_ms\":" << serial_ms
                  << ",\"layer_major_ms\":" << layer_ms
                  << ",\"speedup\":" << serial_ms / layer_ms
                  << ",\"parity\":true}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-verifier-bench: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
