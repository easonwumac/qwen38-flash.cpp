#include "qwen38/chat_template.hpp"

#include <stdexcept>

namespace qwen38 {
namespace {

constexpr std::string_view xhigh_instruction =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
constexpr std::string_view low_instruction =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";

std::string trim_ascii(const std::string_view input) {
    std::size_t begin = 0;
    std::size_t end = input.size();
    while (begin < end && (input[begin] == ' ' || input[begin] == '\t' || input[begin] == '\r' ||
                           input[begin] == '\n')) {
        ++begin;
    }
    while (end > begin && (input[end - 1] == ' ' || input[end - 1] == '\t' ||
                           input[end - 1] == '\r' || input[end - 1] == '\n')) {
        --end;
    }
    return std::string(input.substr(begin, end - begin));
}

std::string reasoning_instruction(const ChatTemplateOptions& options) {
    if (!options.enable_thinking) return {};
    switch (options.reasoning_effort) {
    case ReasoningEffort::xhigh: return std::string(xhigh_instruction);
    case ReasoningEffort::medium: return {};
    case ReasoningEffort::low: return std::string(low_instruction);
    }
    throw std::runtime_error("unsupported reasoning effort");
}

bool is_tool_response(const ChatMessage& message) {
    if (message.role != ChatRole::user) return false;
    const std::string content = trim_ascii(message.content);
    return content.starts_with("<tool_response>") && content.ends_with("</tool_response>");
}

} // namespace

std::string render_chat_prompt(
    const std::vector<ChatMessage>& messages,
    const ChatTemplateOptions& options) {
    if (messages.empty()) throw std::runtime_error("no chat messages provided");
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role == ChatRole::system && i != 0) {
            throw std::runtime_error("system message must be first");
        }
        if (messages[i].role == ChatRole::tool) {
            throw std::runtime_error("tool messages are not implemented in this milestone");
        }
    }

    std::size_t last_query = messages.size();
    for (std::size_t i = messages.size(); i > 0; --i) {
        if (messages[i - 1].role == ChatRole::user && !is_tool_response(messages[i - 1])) {
            last_query = i - 1;
            break;
        }
    }
    if (last_query == messages.size()) throw std::runtime_error("no user query found in messages");

    const std::string instruction = reasoning_instruction(options);
    const bool has_system = messages.front().role == ChatRole::system;
    const std::string system_content = has_system ? trim_ascii(messages.front().content) : std::string{};
    std::string output;
    if (has_system) {
        if (!system_content.empty() || !instruction.empty()) {
            output += "<|im_start|>system\n";
            if (!instruction.empty()) {
                output += instruction;
                if (!system_content.empty()) output += "\n\n";
            }
            output += system_content;
            output += "<|im_end|>\n";
        }
    } else if (!instruction.empty()) {
        output += "<|im_start|>system\n" + instruction + "<|im_end|>\n";
    }

    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (message.role == ChatRole::system) continue;
        const std::string content = trim_ascii(message.content);
        if (message.role == ChatRole::user) {
            output += "<|im_start|>user\n" + content + "<|im_end|>\n";
            continue;
        }
        if (message.role == ChatRole::assistant) {
            output += "<|im_start|>assistant\n";
            if (options.preserve_thinking || i > last_query) {
                const std::string reasoning = message.reasoning_content.has_value()
                    ? trim_ascii(*message.reasoning_content)
                    : std::string{};
                output += "<think>\n" + reasoning + "\n</think>\n\n";
            }
            output += content + "<|im_end|>\n";
        }
    }

    if (options.add_generation_prompt) {
        output += "<|im_start|>assistant\n";
        output += options.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    }
    return output;
}

ChatRole parse_chat_role(const std::string_view role) {
    if (role == "system") return ChatRole::system;
    if (role == "user") return ChatRole::user;
    if (role == "assistant") return ChatRole::assistant;
    if (role == "tool") return ChatRole::tool;
    throw std::runtime_error("unsupported chat role: " + std::string(role));
}

} // namespace qwen38
