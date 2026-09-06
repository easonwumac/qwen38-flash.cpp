#include "qwen38/native_engine.hpp"
#include "qwen38/chat_template.hpp"
#include "qwen38/runtime_profile.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <libproc.h>
#include <unistd.h>

int main(int argc, char** argv) try {
    if ((argc != 3 && argc != 4) || !std::getenv("QWEN38_MEMORY_GUARD"))
        throw std::runtime_error("usage: guarded mtp-cache-probe MODEL SSD_DIRECTORY_OR_DASH [8192|32768]");
    const int long_context = argc == 4 ? std::stoi(argv[3]) : 0;
    if (argc == 4 && long_context != 8192 && long_context != 32768)
        throw std::runtime_error("unsupported context probe size");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 40ULL * 1024 * 1024 * 1024))
        throw std::runtime_error("allocation cap failed");
    qwen38::apply_runtime_profile("speed");
    qwen38::NativeEngineOptions options;
    options.max_generation_tokens = 64;
    options.mtp_depth = 4;
    options.prefill_chunk_rows = 512;
    options.qmeta_cache_max_prompt_tokens = 0;
    const bool ssd = std::string_view(argv[2]) != "-";
    options.prefix_cache_max_tokens = ssd ? 65536 : 0;
    if (ssd) {
        options.ssd_prefix_cache_directory = argv[2];
        options.ssd_prefix_cache_max_bytes = (long_context ? 4ULL : 1ULL) * 1024 * 1024 * 1024;
    }
    qwen38::NativeEngine engine(argv[1], options);
    const std::array<const char*, 4> prompts{
        "Write a Python function that merges two sorted integer lists without using sort().",
        "Return JSON with keys name, version, features, and enabled for a local inference server.",
        "Explain why memory bandwidth limits token generation on a local language model.",
        "Write a short scene about an astronomer discovering a silent blue comet."};
    qwen38::ChatTemplateOptions chat;
    chat.enable_thinking = false;
    std::string long_prompt;
    if (long_context) {
        auto tokenizer = qwen38::Tokenizer::load(argv[1]);
        std::string body = "Read these numbered records.\n";
        for (int line = 0;; ++line) {
            body += "Record " + std::to_string(line) + ": the archive stores ordinary weather observations.\n";
            if (line % 8 != 0) continue;
            auto candidate = qwen38::render_chat_prompt({{.role=qwen38::ChatRole::user,
                .content=body + "\nNow write a short Python function that adds two integers.",
                .reasoning_content={}, .tool_calls={}}}, chat);
            if (tokenizer.encode(candidate).size() >= static_cast<std::size_t>(long_context)) {
                long_prompt = std::move(candidate); break;
            }
        }
    }
    std::cout << std::unitbuf;
    for (std::size_t i = 0; i < (long_context ? 1 : prompts.size()); ++i) {
        auto prompt = long_context ? long_prompt : qwen38::render_chat_prompt({{.role=qwen38::ChatRole::user,
            .content=prompts[i], .reasoning_content={}, .tool_calls={}}}, chat);
        std::vector<std::uint32_t> expected;
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            double first_delta_ms = -1;
            const auto result = engine.complete_stream(prompt, long_context ? 16 : 64,
                [&](std::string_view) {
                    if (first_delta_ms < 0) first_delta_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
                    return true;
                });
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            rusage_info_v4 usage{};
            if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                reinterpret_cast<rusage_info_t*>(&usage))) throw std::runtime_error("footprint query failed");
            std::size_t active{}, cache{};
            if (mlx_get_active_memory(&active) || mlx_get_cache_memory(&cache))
                throw std::runtime_error("allocator query failed");
            if (repeat == 0) expected = result.tokens;
            std::cout << "{\"case\":" << i << ",\"repeat\":" << repeat
                << ",\"cached\":" << result.cached_prompt_tokens
                << ",\"prompt_tokens\":" << result.prompt_tokens
                << ",\"prompt_ms\":" << result.prompt_ms
                << ",\"generation_ms\":" << result.generation_ms
                << ",\"first_delta_ms\":" << first_delta_ms
                << ",\"wall_ms\":" << wall_ms
                << ",\"idle_footprint_bytes\":" << usage.ri_phys_footprint
                << ",\"idle_active_bytes\":" << active
                << ",\"idle_cache_bytes\":" << cache
                << ",\"accepted\":" << result.mtp_accepted
                << ",\"proposed\":" << result.mtp_proposed
                << ",\"rounds\":" << result.mtp_rounds
                << ",\"fallbacks\":" << result.mtp_fallbacks
                << ",\"repeat_equal\":" << (result.tokens == expected ? "true" : "false")
                << ",\"tokens\":[";
            for (std::size_t j = 0; j < result.tokens.size(); ++j) {
                if (j) std::cout << ',';
                std::cout << result.tokens[j];
            }
            std::cout << "]}\n";
            if (long_context && repeat == 1 &&
                (result.tokens != expected || (ssd && result.cached_prompt_tokens == 0)))
                throw std::runtime_error("long-context cache reuse/parity check failed");
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
