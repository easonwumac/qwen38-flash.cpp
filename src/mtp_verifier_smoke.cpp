#include "qwen38/model.hpp"
#include "qwen38/mtp_verifier.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model verifier smoke tests must run through devtools/memory_guard.py");
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel model(tensors);
        const std::vector<std::uint32_t> drafts{11, 271};
        const qwen38::ModelDecodeState origin = model.make_state();

        qwen38::MtpTargetVerification serial =
            qwen38::verify_mtp_target_serial_oracle(model, 9419, drafts, origin);
        qwen38::MtpTargetVerification layer_major =
            qwen38::verify_mtp_target_layer_major_reference(model, 9419, drafts, origin);
        if (serial.rows.size() != layer_major.rows.size()) {
            throw std::runtime_error("verifier row-count mismatch");
        }
        std::vector<std::uint32_t> tokens;
        tokens.reserve(serial.rows.size());
        for (std::size_t row = 0; row < serial.rows.size(); ++row) {
            if (serial.rows[row].greedy.token != layer_major.rows[row].greedy.token ||
                serial.rows[row].state_after.token_count !=
                    layer_major.rows[row].state_after.token_count) {
                throw std::runtime_error("layer-major verifier diverged from serial oracle");
            }
            tokens.push_back(serial.rows[row].greedy.token);
        }

        const std::uint32_t correction = tokens[1];
        qwen38::ModelDecodeState serial_committed = model.make_state();
        qwen38::ModelDecodeState layer_major_committed = model.make_state();
        qwen38::commit_mtp_target_verification(
            std::move(serial), 1, serial_committed);
        qwen38::commit_mtp_target_verification(
            std::move(layer_major), 1, layer_major_committed);
        const qwen38::GreedyStep serial_continuation =
            model.greedy_decode(correction, serial_committed);
        const qwen38::GreedyStep layer_major_continuation =
            model.greedy_decode(correction, layer_major_committed);
        if (serial_continuation.token != layer_major_continuation.token) {
            throw std::runtime_error("verifier rollback continuation mismatch");
        }

        std::cout << "{\"rows\":" << tokens.size() << ",\"tokens\":[";
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << tokens[index];
        }
        std::cout << "],\"partial_accept_continuation\":"
                  << serial_continuation.token << ",\"parity\":true}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-verifier-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
