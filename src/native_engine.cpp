#include "qwen38/native_engine.hpp"

#include <chrono>
#include <stdexcept>

namespace qwen38 {

NativeEngine::NativeEngine(const std::filesystem::path& model_directory)
    : tensors_(ModelManifest::load(model_directory)),
      tokenizer_(Tokenizer::load(model_directory)),
      model_(tensors_) {}

GenerationResult NativeEngine::complete(
    const std::string_view prompt,
    const std::size_t max_tokens) {
    if (max_tokens == 0 || max_tokens > 256) {
        throw std::runtime_error("max_tokens must be between 1 and 256");
    }
    std::scoped_lock lock(inference_mutex_);
    const std::vector<std::uint32_t> prompt_tokens = tokenizer_.encode(prompt);
    if (prompt_tokens.empty()) throw std::runtime_error("prompt produced no tokens");

    ModelDecodeState state = model_.make_state();
    const auto prompt_started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index + 1 < prompt_tokens.size(); ++index) {
        model_.consume_decode(prompt_tokens[index], state);
    }
    const double prompt_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prompt_started).count();

    GenerationResult result;
    result.prompt_tokens = prompt_tokens.size();
    result.prompt_ms = prompt_ms;
    result.tokens.reserve(max_tokens);
    std::uint32_t current = prompt_tokens.back();
    const auto generation_started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < max_tokens; ++index) {
        const GreedyStep step = model_.greedy_decode(current, state);
        result.tokens.push_back(step.token);
        current = step.token;
        if (step.token == tensors_.manifest().config().end_of_sequence_token) {
            result.finish_reason = "stop";
            break;
        }
    }
    result.generation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - generation_started).count();
    result.text = tokenizer_.decode(result.tokens);
    return result;
}

void NativeEngine::clear_cache() {
    // Requests own their decode state today, so there is no cross-request KV
    // state to evict. The method remains a stable maintenance API boundary.
}

} // namespace qwen38
