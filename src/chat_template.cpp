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
constexpr std::string_view tool_instructions =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view tool_suffix =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n</IMPORTANT>";

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

void require_xml_name(const std::string_view value, const std::string_view kind) {
    if (value.empty() || value.find_first_of("<>\r\n") != std::string_view::npos) {
        throw std::runtime_error("invalid " + std::string(kind) + " name");
    }
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
    if (!options.tools_json.empty()) {
        output += "<|im_start|>system\n";
        if (!instruction.empty()) output += instruction + "\n\n";
        output += tool_instructions;
        for (const std::string& tool : options.tools_json) output += "\n" + tool;
        output += tool_suffix;
        if (!system_content.empty()) output += "\n\n" + system_content;
        output += "<|im_end|>\n";
    } else if (has_system) {
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
            output += content;
            for (std::size_t call_index = 0; call_index < message.tool_calls.size();
                 ++call_index) {
                const ChatToolCall& call = message.tool_calls[call_index];
                require_xml_name(call.name, "tool");
                if (call_index == 0 && !content.empty()) output += "\n\n";
                else if (call_index != 0) output += '\n';
                output += "<tool_call>\n<function=" + call.name + ">\n";
                for (const ChatToolArgument& argument : call.arguments) {
                    require_xml_name(argument.name, "tool parameter");
                    output += "<parameter=" + argument.name + ">\n" +
                        argument.rendered_value + "\n</parameter>\n";
                }
                output += "</function>\n</tool_call>";
            }
            output += "<|im_end|>\n";
            continue;
        }
        if (message.role == ChatRole::tool) {
            if (i == 0 || messages[i - 1].role != ChatRole::tool) {
                output += "<|im_start|>user";
            }
            output += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i + 1 == messages.size() || messages[i + 1].role != ChatRole::tool) {
                output += "<|im_end|>\n";
            }
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
