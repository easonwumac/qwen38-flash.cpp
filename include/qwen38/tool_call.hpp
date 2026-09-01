#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qwen38 {

struct ParsedToolCall {
    std::string name;
    std::string arguments_json;
};

struct ParsedToolOutput {
    std::string content;
    std::vector<ParsedToolCall> calls;
};

// Parses the canonical XML emitted by the Qwen3.8 Flash Next chat template.
// Any malformed wrapper is returned unchanged as visible content.
[[nodiscard]] ParsedToolOutput parse_qwen_tool_output(
    std::string_view output,
    std::span<const std::string> tool_schemas = {});

} // namespace qwen38
