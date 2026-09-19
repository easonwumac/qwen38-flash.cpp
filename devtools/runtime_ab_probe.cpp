#include "qwen38/chat_template.hpp"
#include "qwen38/native_engine.hpp"
#include "qwen38/runtime_profile.hpp"

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4 || std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error("Usage under memory_guard: runtime-ab-probe MODEL QWEN38_SWITCH [TOKENS]");
        }
        const std::string toggle = argv[2];
        if (!toggle.starts_with("QWEN38_") || toggle.find_first_not_of(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos) {
            throw std::runtime_error("invalid runtime switch name");
        }
        const std::size_t tokens = argc == 4 ? std::stoul(argv[3]) : 256;
        if (tokens < 1 || tokens > 4096) throw std::runtime_error("TOKENS must be 1..4096");
        qwen38::apply_automatic_runtime_config();
        qwen38::NativeEngineOptions options;
        options.max_generation_tokens = tokens;
        options.prefix_cache_max_tokens = 0;
        options.prefill_chunk_rows = 2048;
        static_cast<void>(::setenv(toggle.c_str(), "0", 1));
        qwen38::NativeEngine engine(argv[1], options);
        const std::array<const char*, 4> names{"code", "json", "explain", "creative"};
        const std::array<const char*, 4> prompts{
            "Write a Python function that merges two sorted integer lists without using sort().",
            "Return JSON with keys name, version, features, and enabled for a local inference server.",
            "Explain why memory bandwidth limits token generation on a local language model.",
            "Write a short scene about an astronomer discovering a silent blue comet."};
        std::array<std::vector<std::uint32_t>, 4> control;
        constexpr std::array<int, 6> order{0, 1, 0, 1, 1, 0};
        std::cout << std::setprecision(10);
        for (std::size_t pass = 0; pass < order.size(); ++pass) {
            const bool candidate = order[pass] != 0;
            static_cast<void>(::setenv(toggle.c_str(), candidate ? "1" : "0", 1));
            for (std::size_t c = 0; c < prompts.size(); ++c) {
                qwen38::ChatTemplateOptions chat;
                chat.enable_thinking = false;
                const auto prompt = qwen38::render_chat_prompt({
                    {qwen38::ChatRole::user, prompts[c], std::nullopt}}, chat);
                const auto result = engine.complete(prompt, tokens);
                // Always compare against the first control, including later
                // control passes: state leakage must not silently reset it.
                if (pass == 0) control[c] = result.tokens;
                std::uint64_t hash = 1469598103934665603ULL;
                for (auto token : result.tokens) { hash ^= token; hash *= 1099511628211ULL; }
                std::cout << "{\"switch\":\"" << toggle << "\",\"pass\":" << pass
                          << ",\"warmup\":" << (pass < 2 ? "true" : "false")
                          << ",\"candidate\":" << (candidate ? "true" : "false")
                          << ",\"case\":\"" << names[c] << "\",\"prompt_tokens\":" << result.prompt_tokens
                          << ",\"tokens\":" << result.tokens.size()
                          << ",\"prompt_ms\":" << result.prompt_ms
                          << ",\"generation_ms\":" << result.generation_ms
                          << ",\"tps\":" << result.tokens.size() * 1000.0 / result.generation_ms
                          << ",\"token_hash\":\"" << hash << "\",\"control_token_parity\":"
                          << (control[c] == result.tokens ? "true" : "false")
                          << ",\"rounds\":" << result.mtp_rounds << ",\"accepted\":" << result.mtp_accepted
                          << ",\"proposed\":" << result.mtp_proposed << ",\"depth\":" << result.mtp_final_depth
                          << ",\"demotions\":" << result.mtp_demotions << ",\"promotions\":" << result.mtp_promotions
                          << ",\"fallbacks\":" << result.mtp_fallbacks
                          << ",\"draft_ms\":" << result.mtp_draft_ms << ",\"verify_ms\":" << result.mtp_verify_ms
                          << ",\"commit_ms\":" << result.mtp_commit_ms << "}\n" << std::flush;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "runtime-ab-probe: " << error.what() << '\n';
        return 1;
    }
}
