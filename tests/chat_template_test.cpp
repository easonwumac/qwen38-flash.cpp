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

    qwen38::ChatTemplateOptions tools;
    tools.reasoning_effort = qwen38::ReasoningEffort::medium;
    tools.tools_json = {
        R"({"function":{"description":"Read a file","name":"read","parameters":{"properties":{"path":{"type":"string"}},"required":["path"],"type":"object"}},"type":"function"})",
    };
    const std::string tool_prompt = qwen38::render_chat_prompt({
        {qwen38::ChatRole::system, "Use the workspace.", {}, {}},
        {qwen38::ChatRole::user, "Inspect it", {}, {}},
        {qwen38::ChatRole::assistant, "", {}, {{"read", {{"path", "/tmp/a"}}}}},
        {qwen38::ChatRole::tool, "first result", {}, {}},
        {qwen38::ChatRole::tool, "second result", {}, {}},
    }, tools);
    QWEN38_CHECK(tool_prompt.starts_with(
        "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n"
        "<tools>\n{\"function\":"));
    QWEN38_CHECK(tool_prompt.find(
        "</IMPORTANT>\n\nUse the workspace.<|im_end|>\n") != std::string::npos);
    QWEN38_CHECK(tool_prompt.find(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n"
        "<function=read>\n<parameter=path>\n/tmp/a\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>\n") != std::string::npos);
    QWEN38_CHECK(tool_prompt.find(
        "<|im_start|>user\n<tool_response>\nfirst result\n</tool_response>\n"
        "<tool_response>\nsecond result\n</tool_response><|im_end|>\n") != std::string::npos);
    QWEN38_CHECK(tool_prompt.ends_with("<|im_start|>assistant\n<think>\n"));

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
