#include "qwen38/tool_call.hpp"

#include "qwen38/json.hpp"

#include <cctype>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace qwen38 {
namespace {

constexpr std::string_view call_open = "<tool_call>";
constexpr std::string_view call_close = "</tool_call>";
constexpr std::string_view function_open = "<function=";
constexpr std::string_view function_close = "</function>";
constexpr std::string_view parameter_open = "<parameter=";
constexpr std::string_view parameter_close = "</parameter>";

void skip_space(const std::string_view input, std::size_t& cursor) {
    while (cursor < input.size() &&
        std::isspace(static_cast<unsigned char>(input[cursor]))) {
        ++cursor;
    }
}

std::string trim(const std::string_view input) {
    std::size_t begin = 0;
    std::size_t end = input.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(input[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) --end;
    return std::string(input.substr(begin, end - begin));
}

bool consume(const std::string_view input, std::size_t& cursor, const std::string_view value) {
    if (input.substr(cursor, value.size()) != value) return false;
    cursor += value.size();
    return true;
}

std::optional<std::string> tagged_name(
    const std::string_view input,
    std::size_t& cursor,
    const std::string_view opening) {
    if (!consume(input, cursor, opening)) return std::nullopt;
    const std::size_t close = input.find('>', cursor);
    if (close == std::string_view::npos || close == cursor) return std::nullopt;
    const std::string_view name = input.substr(cursor, close - cursor);
    if (name.find_first_of("<>\r\n") != std::string_view::npos) return std::nullopt;
    cursor = close + 1;
    return std::string(name);
}

bool schema_requires_string(
    const std::span<const std::string> schemas,
    const std::string_view tool_name,
    const std::string_view parameter_name) {
    for (const std::string& source : schemas) {
        try {
            const Json schema = Json::parse(source);
            const Json* function = schema.find("function");
            if (function == nullptr) function = &schema;
            if (function->at("name").as_string() != tool_name) continue;
            const Json* parameters = function->find("parameters");
            const Json* properties = parameters == nullptr
                ? nullptr
                : parameters->find("properties");
            const Json* property = properties == nullptr
                ? nullptr
                : properties->find(parameter_name);
            const Json* type = property == nullptr ? nullptr : property->find("type");
            if (type == nullptr) return false;
            if (type->is_string()) return type->as_string() == "string";
            if (type->is_array()) {
                for (const Json& candidate : type->as_array()) {
                    if (candidate.is_string() && candidate.as_string() == "string") return true;
                }
            }
            return false;
        } catch (const std::exception&) {
            continue;
        }
    }
    return false;
}

Json inferred_argument(const std::string& value, const bool force_string) {
    if (force_string) return Json(value);
    try {
        return Json::parse(value);
    } catch (const std::runtime_error&) {
        return Json(value);
    }
}

std::optional<ParsedToolCall> parse_call(
    const std::string_view input,
    std::size_t& cursor,
    const std::span<const std::string> tool_schemas) {
    if (!consume(input, cursor, call_open)) return std::nullopt;
    skip_space(input, cursor);
    std::optional<std::string> name = tagged_name(input, cursor, function_open);
    if (!name.has_value()) return std::nullopt;
    Json::Object arguments;
    while (true) {
        skip_space(input, cursor);
        if (consume(input, cursor, function_close)) break;
        std::optional<std::string> parameter = tagged_name(input, cursor, parameter_open);
        if (!parameter.has_value()) return std::nullopt;
        if (cursor < input.size() && input[cursor] == '\r') ++cursor;
        if (cursor < input.size() && input[cursor] == '\n') ++cursor;
        const std::size_t close = input.find(parameter_close, cursor);
        if (close == std::string_view::npos) return std::nullopt;
        std::string value = trim(input.substr(cursor, close - cursor));
        const bool force_string = schema_requires_string(
            tool_schemas, *name, *parameter);
        if (!arguments.emplace(*parameter, inferred_argument(value, force_string)).second) {
            return std::nullopt;
        }
        cursor = close + parameter_close.size();
    }
    skip_space(input, cursor);
    if (!consume(input, cursor, call_close)) return std::nullopt;
    return ParsedToolCall{
        .name = std::move(*name),
        .arguments_json = Json(std::move(arguments)).dump(),
    };
}

} // namespace

ParsedToolOutput parse_qwen_tool_output(
    const std::string_view output,
    const std::span<const std::string> tool_schemas) {
    const std::size_t first_call = output.find(call_open);
    if (first_call == std::string_view::npos) return {.content = trim(output), .calls = {}};

    ParsedToolOutput parsed{.content = trim(output.substr(0, first_call))};
    std::size_t cursor = first_call;
    while (cursor < output.size()) {
        skip_space(output, cursor);
        if (cursor == output.size()) break;
        std::optional<ParsedToolCall> call = parse_call(output, cursor, tool_schemas);
        if (!call.has_value()) return {.content = trim(output), .calls = {}};
        parsed.calls.push_back(std::move(*call));
    }
    if (parsed.calls.empty()) return {.content = trim(output), .calls = {}};
    return parsed;
}

} // namespace qwen38
