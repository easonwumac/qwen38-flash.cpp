#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"
#include "qwen38/mtp_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::uint32_t argmax_token(const qwen38::MlxArray& logits) {
    qwen38::MlxArray token = logits.argmax_all();
    token.eval();
    return token.item_uint32();
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
                "full-model MTP round tests must run through devtools/memory_guard.py");
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        constexpr std::size_t cache_limit = 256ULL * 1024ULL * 1024ULL;
        static_cast<void>(qwen38::MlxArray::set_cache_limit(cache_limit));
        qwen38::QwenModel target(tensors);
        qwen38::QwenMtpHead head(tensors);
        qwen38::ModelDecodeState target_state = target.make_state();
        qwen38::MtpDecodeState head_state = head.make_state();

        qwen38::TargetDecodeStep bootstrap =
            target.forward_decode_capture(9419, target_state);
        std::uint32_t current = argmax_token(bootstrap.logits);
        qwen38::MlxArray previous_stream = std::move(bootstrap.pre_mixer_stream);
        std::size_t query_position = target_state.token_count;
        qwen38::ModelDecodeState serial_state = target.snapshot_state(target_state);

        constexpr std::size_t token_goal = 4;
        std::vector<std::uint32_t> serial_tokens;
        serial_tokens.reserve(token_goal);
        std::uint32_t serial_current = current;
        const auto serial_started = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < token_goal; ++index) {
            qwen38::GreedyStep step = target.greedy_decode(serial_current, serial_state);
            serial_tokens.push_back(step.token);
            serial_current = step.token;
        }
        const double serial_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - serial_started).count();
        qwen38::MlxArray::clear_cache();

        std::vector<std::uint32_t> mtp_tokens;
        std::size_t accepted = 0;
        std::size_t rounds = 0;
        const auto mtp_started = std::chrono::steady_clock::now();
        while (mtp_tokens.size() < token_goal && rounds < token_goal) {
            qwen38::MtpRoundStep step = qwen38::run_greedy_mtp_round_reference(
                target,
                head,
                current,
                previous_stream,
                query_position,
                2,
                target_state,
                head_state);
            mtp_tokens.insert(
                mtp_tokens.end(), step.emitted_tokens.begin(), step.emitted_tokens.end());
            accepted += step.accepted;
            ++rounds;
            current = step.next_current_token;
            query_position = step.next_query_position;
            previous_stream = std::move(step.next_target_stream);
            qwen38::MlxArray::clear_cache();
        }
        const double mtp_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - mtp_started).count();
        if (mtp_tokens.size() < token_goal ||
            !std::equal(serial_tokens.begin(), serial_tokens.end(), mtp_tokens.begin())) {
            throw std::runtime_error("multi-round MTP output diverged from serial greedy decode");
        }
        std::cout << "{\"tokens_checked\":" << token_goal << ",\"rounds\":" << rounds
                  << ",\"accepted\":" << accepted
                  << ",\"serial_ms\":" << serial_ms
                  << ",\"serial_tps\":"
                  << 1000.0 * static_cast<double>(token_goal) / serial_ms
                  << ",\"mtp_ms\":" << mtp_ms
                  << ",\"mtp_emitted\":" << mtp_tokens.size()
                  << ",\"mtp_tps\":"
                  << 1000.0 * static_cast<double>(mtp_tokens.size()) / mtp_ms
                  << ",\"tokens\":[";
        for (std::size_t index = 0; index < token_goal; ++index) {
            if (index != 0) std::cout << ',';
            std::cout << mtp_tokens[index];
        }
        std::cout << "],\"serial_parity\":true}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-round-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
