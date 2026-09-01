#include "qwen38/chat_template.hpp"
#include "test.hpp"

#include <stdexcept>
#include <string>
#include <vector>

void run_chat_template_tests() {
    constexpr auto instruction =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
        "and clarity in the final answer.";

    QWEN38_CHECK(qwen38::render_chat_prompt({{qwen38::ChatRole::user, "Hello", {}}}) ==
        std::string("<|im_start|>system\n") + instruction +
        "<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n");

    QWEN38_CHECK(qwen38::render_chat_prompt({
        {qwen38::ChatRole::system, "Be concise.", {}},
        {qwen38::ChatRole::user, "你好", {}},
    }) == std::string("<|im_start|>system\n") + instruction +
        "\n\nBe concise.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n");

    QWEN38_CHECK(qwen38::render_chat_prompt({
        {qwen38::ChatRole::user, "Question", {}},
        {qwen38::ChatRole::assistant, "Answer", "Reason"},
        {qwen38::ChatRole::user, "Next", {}},
    }) == std::string("<|im_start|>system\n") + instruction +
        "<|im_end|>\n<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
        "<think>\nReason\n</think>\n\nAnswer<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n");

    qwen38::ChatTemplateOptions disabled;
    disabled.enable_thinking = false;
    QWEN38_CHECK(qwen38::render_chat_prompt({{qwen38::ChatRole::user, "Hi", {}}}, disabled) ==
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

    qwen38::ChatTemplateOptions medium;
    medium.reasoning_effort = qwen38::ReasoningEffort::medium;
    QWEN38_CHECK(qwen38::render_chat_prompt({{qwen38::ChatRole::user, "Hi", {}}}, medium) ==
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");

    bool rejected = false;
    try {
        static_cast<void>(qwen38::render_chat_prompt({
            {qwen38::ChatRole::user, "Hi", {}},
            {qwen38::ChatRole::system, "late", {}},
        }));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    QWEN38_CHECK(rejected);
}
