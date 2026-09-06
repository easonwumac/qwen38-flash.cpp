#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"
#include "qwen38/mtp_runner.hpp"
#include "qwen38/runtime_profile.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <libproc.h>
#include <unistd.h>

namespace {

void report_memory(const char* phase) {
    std::size_t active{}, cache{}, peak{};
    rusage_info_v4 usage{};
    if (mlx_get_active_memory(&active) || mlx_get_cache_memory(&cache) ||
        mlx_get_peak_memory(&peak) || proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
            reinterpret_cast<rusage_info_t*>(&usage)))
        throw std::runtime_error("memory accounting failed");
    std::cerr << "[memory] " << phase << " active=" << active << " cache=" << cache
              << " peak=" << peak << " footprint=" << usage.ri_phys_footprint << '\n';
}

std::uint32_t argmax_token(const qwen38::MlxArray& logits) {
    qwen38::MlxArray token = logits.argmax_all();
    token.eval();
    return token.item_uint32();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY [PROFILE]\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model MTP round tests must run through devtools/memory_guard.py");
        }
        // Allocation-time cap complements the polling process guard. Enforced
        // strictly only when using the private strict MLX runtime.
        std::size_t previous_limit{};
        if (mlx_set_memory_limit(&previous_limit, 40ULL * 1024 * 1024 * 1024) != 0)
            throw std::runtime_error("could not set full-model allocation cap");
        if (argc == 3) qwen38::apply_runtime_profile(argv[2]);
        std::cerr << "[phase] loading target and MTP head\n";
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        constexpr std::size_t cache_limit = 256ULL * 1024ULL * 1024ULL;
        static_cast<void>(qwen38::MlxArray::set_cache_limit(cache_limit));
        qwen38::QwenModel target(tensors);
        qwen38::QwenMtpHead head(tensors);
        report_memory("constructed");
        qwen38::ModelDecodeState target_state = target.make_state();
        qwen38::MtpDecodeState head_state = head.make_state();

        std::cerr << "[phase] bootstrap decode\n";
        qwen38::TargetDecodeStep bootstrap =
            target.forward_decode_capture(9419, target_state);
        std::uint32_t current = argmax_token(bootstrap.logits);
        qwen38::MlxArray previous_stream = std::move(bootstrap.pre_mixer_stream);
        std::size_t query_position = target_state.token_count;
        report_memory("bootstrap_before_release");
        bootstrap.logits = {};
        report_memory("bootstrap_after_release");
        qwen38::ModelDecodeState serial_state = target.snapshot_state(target_state);

        constexpr std::size_t token_goal = 4;
        std::vector<std::uint32_t> serial_tokens;
        serial_tokens.reserve(token_goal);
        std::uint32_t serial_current = current;
        std::cerr << "[phase] serial greedy oracle\n";
        const auto serial_started = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < token_goal; ++index) {
            qwen38::GreedyStep step = target.greedy_decode(serial_current, serial_state);
            serial_tokens.push_back(step.token);
            serial_current = step.token;
        }
        const double serial_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - serial_started).count();
        report_memory("serial_before_release");
        serial_state = qwen38::ModelDecodeState(0);
        report_memory("serial_after_release");
        qwen38::MlxArray::clear_cache();
        report_memory("serial_after_cache_clear");

        std::vector<std::uint32_t> mtp_tokens;
        std::size_t accepted = 0;
        std::size_t rounds = 0;
        std::cerr << "[phase] MTP rounds\n";
        const auto mtp_started = std::chrono::steady_clock::now();
        while (mtp_tokens.size() < token_goal && rounds < token_goal) {
            report_memory("mtp_before_round");
            qwen38::MtpRoundStep step = qwen38::run_greedy_mtp_round_reference(
                target,
                head,
                current,
                previous_stream,
                query_position,
                2,
                target_state,
                head_state);
            report_memory("mtp_after_round");
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
