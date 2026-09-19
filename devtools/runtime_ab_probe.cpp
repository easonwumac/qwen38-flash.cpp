#include "qwen38/chat_template.hpp"
#include "qwen38/native_engine.hpp"
#include "qwen38/runtime_profile.hpp"

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 5 || std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error("Usage under memory_guard: runtime-ab-probe MODEL QWEN38_SWITCH [TOKENS [LINES]]; LINES enables target-only numbered-notes benchmark");
        }
        const std::string toggle = argv[2];
        if (!toggle.starts_with("QWEN38_") || toggle.find_first_not_of(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos) {
            throw std::runtime_error("invalid runtime switch name");
        }
        const std::size_t tokens = argc >= 4 ? std::stoul(argv[3]) : 256;
        if (tokens < 1 || tokens > 4096) throw std::runtime_error("TOKENS must be 1..4096");
        const std::size_t lines = argc == 5 ? std::stoul(argv[4]) : 0;
        if (argc == 5 && (lines < 1 || lines > 8192))
            throw std::runtime_error("LINES must be 1..8192");
        qwen38::apply_automatic_runtime_config();
        // Match main.cpp's production KV defaults. The core configuration alone
        // does not set these; leaving them absent silently benchmarks BF16 KV.
        for (const auto& [name, value] : std::array<std::pair<const char*, const char*>, 3>{{
                 {"QWEN38_KV_CACHE", "q8"},
                 {"QWEN38_KV_Q8_MIN_TOKENS", "8192"},
                 {"QWEN38_KV_Q8_FLUSH_TOKENS", "2048"}}}) {
            if (::setenv(name, value, 0) != 0)
                throw std::runtime_error("cannot configure benchmark KV defaults");
        }
        qwen38::NativeEngineOptions options;
        options.max_generation_tokens = tokens;
        options.prefix_cache_max_tokens = 0;
        options.prefill_chunk_rows = 2048;
        if (lines != 0) options.mtp_depth = 0;
        static_cast<void>(::setenv(toggle.c_str(), "0", 1));
        qwen38::NativeEngine engine(argv[1], options);
        std::vector<std::string> names{"code", "json", "explain", "creative"};
        std::vector<std::string> prompts{
            "Write a Python function that merges two sorted integer lists without using sort().",
            "Return JSON with keys name, version, features, and enabled for a local inference server.",
            "Explain why memory bandwidth limits token generation on a local language model.",
            "Write a short scene about an astronomer discovering a silent blue comet."};
        if (lines != 0) {
            std::ostringstream notes;
            notes << "Summarize the following numbered notes in one concise sentence.\n";
            for (std::size_t i = 0; i < lines; ++i) {
                if (i) notes << '\n';
                notes << std::setfill('0') << std::setw(5) << i
                      << ": alpha beta gamma delta epsilon zeta eta theta";
            }
            names = {"numbered_notes"};
            prompts = {notes.str()};
        }
        std::vector<std::vector<std::uint32_t>> control(prompts.size());
        std::vector<std::string> rendered_prompts;
        std::vector<std::uint64_t> prompt_hashes;
        const auto tokenizer = qwen38::Tokenizer::load(argv[1]);
        for (const auto& content : prompts) {
            qwen38::ChatTemplateOptions chat;
            chat.enable_thinking = false;
            rendered_prompts.push_back(qwen38::render_chat_prompt({
                {qwen38::ChatRole::user, content, std::nullopt}}, chat));
            std::uint64_t hash = 1469598103934665603ULL;
            for (auto token : tokenizer.encode(rendered_prompts.back())) {
                hash ^= token; hash *= 1099511628211ULL;
            }
            prompt_hashes.push_back(hash);
        }
        const std::vector<int> order = lines != 0
            ? std::vector<int>{0, 1, 0, 1, 1, 0, 0, 1}
            : std::vector<int>{0, 1, 0, 1, 1, 0};
        std::cout << std::setprecision(10);
        for (std::size_t pass = 0; pass < order.size(); ++pass) {
            const bool candidate = order[pass] != 0;
            static_cast<void>(::setenv(toggle.c_str(), candidate ? "1" : "0", 1));
            for (std::size_t c = 0; c < prompts.size(); ++c) {
                std::clog << "runtime-ab-probe: pass=" << pass << " candidate=" << candidate
                          << " case=" << names[c] << '\n';
                const auto result = engine.complete(rendered_prompts[c], tokens);
                // Always compare against the first control, including later
                // control passes: state leakage must not silently reset it.
                if (pass == 0) control[c] = result.tokens;
                std::uint64_t hash = 1469598103934665603ULL;
                for (auto token : result.tokens) { hash ^= token; hash *= 1099511628211ULL; }
                std::cout << "{\"switch\":\"" << toggle << "\",\"pass\":" << pass
                          << ",\"warmup\":" << (pass < 2 ? "true" : "false")
                          << ",\"candidate\":" << (candidate ? "true" : "false")
                          << ",\"case\":\"" << names[c] << "\",\"prompt_tokens\":" << result.prompt_tokens
                          << ",\"prompt_token_hash\":\"" << prompt_hashes[c] << "\""
                          << ",\"kv_cache\":\"" << std::getenv("QWEN38_KV_CACHE") << "\""
                          << ",\"kv_q8_min_tokens\":\"" << std::getenv("QWEN38_KV_Q8_MIN_TOKENS") << "\""
                          << ",\"kv_q8_flush_tokens\":\"" << std::getenv("QWEN38_KV_Q8_FLUSH_TOKENS") << "\""
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
