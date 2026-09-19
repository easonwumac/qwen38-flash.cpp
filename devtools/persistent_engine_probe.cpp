#include "qwen38/chat_template.hpp"
#include "qwen38/native_engine.hpp"
#include "qwen38/runtime_profile.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// In-process serving lifecycle checks, not model-quality or throughput claims.
int main(int argc, char** argv) try {
    using namespace qwen38;
    if (argc != 2 || std::getenv("QWEN38_MEMORY_GUARD") == nullptr)
        throw std::runtime_error("usage under memory_guard: persistent-engine-probe MODEL");
    ::setenv("QWEN38_PERSISTENT_VQ", "1", 1);
    apply_automatic_runtime_config();
    NativeEngineOptions options;
    options.mtp_depth = 0;
    options.max_generation_tokens = 64;
    options.prefix_cache_max_tokens = 4096;
    options.prefill_chunk_rows = 2048;
    // RAM cache only: clear_cache never removes user-owned disk entries here.
    NativeEngine engine(argv[1], options);
    const auto tokenizer = Tokenizer::load(argv[1]);
    ChatTemplateOptions chat;
    chat.enable_thinking = false;
    const auto prompt = render_chat_prompt({
        {ChatRole::user, "Write a Python function that merges two sorted integer lists without using sort().",
         std::nullopt, {}}}, chat);
    const auto expected = engine.complete(prompt, 64);
    if (expected.tokens.size() != 64)
        throw std::runtime_error("fixture stopped before the continuation frontier");
    engine.clear_cache();
    const auto first = engine.complete(prompt, 32);
    if (first.tokens.size() != 32 || !std::equal(
            first.tokens.begin(), first.tokens.end(), expected.tokens.begin()))
        throw std::runtime_error("repeat prefix token mismatch");
    auto prefix_ids = tokenizer.encode(prompt);
    prefix_ids.insert(prefix_ids.end(), first.tokens.begin(), first.tokens.end());
    const auto resumed_prompt = prompt + tokenizer.decode(first.tokens);
    if (tokenizer.encode(resumed_prompt) != prefix_ids)
        throw std::runtime_error("fixture text retokenized at resume boundary");
    const auto resumed = engine.complete(resumed_prompt, 32);
    if (resumed.cached_prompt_tokens + 1 != prefix_ids.size() ||
        resumed.tokens != std::vector<std::uint32_t>(expected.tokens.begin() + 32, expected.tokens.end()))
        throw std::runtime_error("generated-prefix native resume mismatch");
    std::cout << "generated_prefix_cached=" << resumed.cached_prompt_tokens
              << " uninterrupted_64_equals_32_plus_32=true\n" << std::flush;
    engine.clear_cache();
    const auto cancelled = engine.complete_stream(prompt, 64,
        [](std::string_view) { return false; });
    if (cancelled.finish_reason != "cancelled") throw std::runtime_error("cancellation failed");
    engine.clear_cache();
    if (engine.complete(prompt, 64).tokens != expected.tokens)
        throw std::runtime_error("post-cancellation request mismatch");
    bool threw = false;
    engine.clear_cache();
    try {
        static_cast<void>(engine.complete_stream(prompt, 64, [](std::string_view) -> bool {
            throw std::runtime_error("injected callback failure");
        }));
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()) != "injected callback failure") throw;
        threw = true;
    }
    if (!threw) throw std::runtime_error("callback failure was not propagated");
    engine.clear_cache();
    if (engine.complete(prompt, 64).tokens != expected.tokens)
        throw std::runtime_error("post-exception request mismatch");
    std::cout << "cancel_recovery=true callback_exception_recovery=true\n";
    engine.clear_cache();
    const auto short_prompt = render_chat_prompt({
        {ChatRole::user, "Reply with only the word yes.", std::nullopt, {}}}, chat);
    const auto short_answer = engine.complete(short_prompt, 64);
    if (short_answer.finish_reason != "stop" || short_answer.tokens.size() >= 8)
        throw std::runtime_error("native early EOS was suppressed");
    std::cout << "early_eos_respected=true completion_tokens=" << short_answer.tokens.size() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
