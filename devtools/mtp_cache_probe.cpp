#include "qwen38/native_engine.hpp"
#include "qwen38/chat_template.hpp"
#include "qwen38/runtime_profile.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) try {
    if (argc != 3 || !std::getenv("QWEN38_MEMORY_GUARD"))
        throw std::runtime_error("usage: guarded mtp-cache-probe MODEL SSD_DIRECTORY_OR_DASH");
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
    options.prefix_cache_max_tokens = ssd ? 8192 : 0;
    if (ssd) {
        options.ssd_prefix_cache_directory = argv[2];
        options.ssd_prefix_cache_max_bytes = 1024ULL * 1024 * 1024;
    }
    qwen38::NativeEngine engine(argv[1], options);
    const std::array<const char*, 4> prompts{
        "Write a Python function that merges two sorted integer lists without using sort().",
        "Return JSON with keys name, version, features, and enabled for a local inference server.",
        "Explain why memory bandwidth limits token generation on a local language model.",
        "Write a short scene about an astronomer discovering a silent blue comet."};
    qwen38::ChatTemplateOptions chat;
    chat.enable_thinking = false;
    std::cout << std::unitbuf;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        auto prompt = qwen38::render_chat_prompt({{.role=qwen38::ChatRole::user,
            .content=prompts[i], .reasoning_content={}, .tool_calls={}}}, chat);
        std::vector<std::uint32_t> expected;
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto result = engine.complete(prompt, 64);
            if (repeat == 0) expected = result.tokens;
            std::cout << "{\"case\":" << i << ",\"repeat\":" << repeat
                << ",\"cached\":" << result.cached_prompt_tokens
                << ",\"prompt_tokens\":" << result.prompt_tokens
                << ",\"prompt_ms\":" << result.prompt_ms
                << ",\"generation_ms\":" << result.generation_ms
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
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
