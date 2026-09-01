#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qwen38 {

enum class ChatRole { system, user, assistant, tool };

struct ChatMessage {
    ChatRole role;
    std::string content;
    std::optional<std::string> reasoning_content;
};

enum class ReasoningEffort { xhigh, medium, low };

struct ChatTemplateOptions {
    bool add_generation_prompt{true};
    bool enable_thinking{true};
    bool preserve_thinking{true};
    ReasoningEffort reasoning_effort{ReasoningEffort::xhigh};
};

[[nodiscard]] std::string render_chat_prompt(
    const std::vector<ChatMessage>& messages,
    const ChatTemplateOptions& options = {});
[[nodiscard]] ChatRole parse_chat_role(std::string_view role);

} // namespace qwen38
