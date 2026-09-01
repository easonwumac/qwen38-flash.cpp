#include "qwen38/api.hpp"
#include "qwen38/chat_template.hpp"
#include "qwen38/json.hpp"
#include "qwen38/tool_call.hpp"

#include <algorithm>
#include <cstddef>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace qwen38 {
namespace {

HttpResponse json_error(const int status, std::string_view code, std::string_view message) {
    return {
        .status = status,
        .content_type = "application/json",
        .body = "{\"error\":{\"code\":\"" + json_escape(code) +
            "\",\"message\":\"" + json_escape(message) + "\",\"type\":\"server_error\"}}",
    };
}

std::size_t max_tokens(const Json& body) {
    const Json* value = body.find("max_completion_tokens");
    if (value == nullptr) value = body.find("max_tokens");
    if (value == nullptr) return 16;
    const std::int64_t parsed = value->as_integer();
    if (parsed < 1) {
        throw std::runtime_error("max_tokens must be positive");
    }
    return static_cast<std::size_t>(parsed);
}

bool streaming_enabled(const Json& body) {
    const Json* stream = body.find("stream");
    return stream != nullptr && stream->as_boolean();
}

bool stream_usage_enabled(const Json& body) {
    const Json* options = body.find("stream_options");
    if (options == nullptr) return false;
    const Json* include = options->find("include_usage");
    return include != nullptr && include->as_boolean();
}

ChatTemplateOptions chat_template_options(const Json& body) {
    ChatTemplateOptions options;
    const Json* enable_thinking = body.find("enable_thinking");
    const Json* thinking = body.find("thinking");
    if (enable_thinking != nullptr && thinking != nullptr &&
        enable_thinking->as_boolean() != thinking->as_boolean()) {
        throw std::runtime_error("thinking and enable_thinking must agree");
    }
    if (enable_thinking != nullptr) {
        options.enable_thinking = enable_thinking->as_boolean();
    } else if (thinking != nullptr) {
        // mlx-serve and the established local benchmark surface use the
        // shorter spelling. Keep it as a first-class compatibility alias so
        // thinking-off requests actually receive the tokenizer template's
        // closed empty <think> block.
        options.enable_thinking = thinking->as_boolean();
    }
    if (const Json* value = body.find("reasoning_effort"); value != nullptr) {
        const std::string& effort = value->as_string();
        if (effort == "xhigh") options.reasoning_effort = ReasoningEffort::xhigh;
        else if (effort == "medium") options.reasoning_effort = ReasoningEffort::medium;
        else if (effort == "low") options.reasoning_effort = ReasoningEffort::low;
        else throw std::runtime_error("reasoning_effort must be xhigh, medium, or low");
    }
    if (const Json* tools = body.find("tools"); tools != nullptr) {
        for (const Json& tool : tools->as_array()) {
            if (!tool.is_object()) throw std::runtime_error("tools entries must be objects");
            options.tools_json.push_back(tool.dump());
        }
    }
    return options;
}

std::string message_content(const Json& message) {
    const Json* content = message.find("content");
    if (content == nullptr || content->is_null()) return {};
    return content->as_string();
}

std::vector<ChatToolCall> message_tool_calls(const Json& message) {
    const Json* calls = message.find("tool_calls");
    if (calls == nullptr || calls->is_null()) return {};
    std::vector<ChatToolCall> result;
    for (const Json& item : calls->as_array()) {
        const Json* function = item.find("function");
        if (function == nullptr) function = &item;
        ChatToolCall call{.name = function->at("name").as_string()};
        if (const Json* arguments = function->find("arguments"); arguments != nullptr &&
            !arguments->is_null()) {
            if (arguments->is_string() && arguments->as_string().empty()) {
                result.push_back(std::move(call));
                continue;
            }
            const Json parsed = arguments->is_string()
                ? Json::parse(arguments->as_string())
                : *arguments;
            for (const auto& [name, value] : parsed.as_object()) {
                call.arguments.push_back({
                    .name = name,
                    .rendered_value = value.is_string() ? value.as_string() : value.dump(),
                });
            }
        }
        result.push_back(std::move(call));
    }
    return result;
}

struct ChatOutput {
    std::string content;
    std::optional<std::string> reasoning_content;
    std::vector<ParsedToolCall> tool_calls;
};

std::string trim_chat_segment(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

ChatOutput split_chat_output(
    const std::string& text,
    const bool enable_thinking,
    const std::vector<std::string>& tool_schemas) {
    ChatOutput output{.content = text};
    if (enable_thinking) {
        constexpr std::string_view close = "</think>";
        const std::size_t separator = text.find(close);
        if (separator != std::string::npos) {
            output.content = trim_chat_segment(
                std::string_view(text).substr(separator + close.size()));
            output.reasoning_content = trim_chat_segment(
                std::string_view(text).substr(0, separator));
        }
    }
    if (!tool_schemas.empty()) {
        ParsedToolOutput parsed = parse_qwen_tool_output(output.content, tool_schemas);
        output.content = std::move(parsed.content);
        output.tool_calls = std::move(parsed.calls);
    }
    return output;
}

void append_tool_calls_json(
    std::ostringstream& out,
    const std::vector<ParsedToolCall>& calls,
    const bool streaming) {
    out << "\"tool_calls\":[";
    for (std::size_t index = 0; index < calls.size(); ++index) {
        if (index != 0) out << ',';
        out << '{';
        if (streaming) out << "\"index\":" << index << ',';
        out << "\"id\":\"call_qwen38_" << index
            << "\",\"type\":\"function\",\"function\":{\"name\":\""
            << json_escape(calls[index].name) << "\",\"arguments\":\""
            << json_escape(calls[index].arguments_json) << "\"}}";
    }
    out << ']';
}

std::string completion_json(
    const RuntimeSnapshot& snapshot,
    const GenerationResult& result,
    const bool chat,
    const bool enable_thinking,
    const std::vector<std::string>& tool_schemas) {
    std::ostringstream out;
    std::string finish_reason = result.finish_reason;
    out << "{\"id\":\"qwen38-native\",\"object\":\""
        << (chat ? "chat.completion" : "text_completion")
        << "\",\"model\":\"" << json_escape(snapshot.model_id) << "\",\"choices\":[{";
    if (chat) {
        const ChatOutput output = split_chat_output(result.text, enable_thinking, tool_schemas);
        out << "\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
            << json_escape(output.content) << '"';
        if (output.reasoning_content.has_value()) {
            out << ",\"reasoning_content\":\""
                << json_escape(*output.reasoning_content) << '"';
        }
        if (!output.tool_calls.empty()) {
            out << ',';
            append_tool_calls_json(out, output.tool_calls, false);
            finish_reason = "tool_calls";
        }
        out << '}';
    } else {
        out << "\"index\":0,\"text\":\"" << json_escape(result.text) << '"';
    }
    out << ",\"finish_reason\":\"" << json_escape(finish_reason)
        << "\"}],\"usage\":{\"prompt_tokens\":" << result.prompt_tokens
        << ",\"completion_tokens\":" << result.tokens.size()
        << ",\"total_tokens\":" << result.prompt_tokens + result.tokens.size()
        << ",\"prompt_tokens_details\":{\"cached_tokens\":"
        << result.cached_prompt_tokens << "}"
        << "},\"performance\":{\"prompt_ms\":" << result.prompt_ms
        << ",\"cached_prompt_tokens\":" << result.cached_prompt_tokens
        << ",\"generation_ms\":" << result.generation_ms
        << ",\"generation_tps\":"
        << (result.generation_ms > 0.0
                ? 1000.0 * static_cast<double>(result.tokens.size()) / result.generation_ms
                : 0.0)
        << ",\"mtp\":{\"rounds\":" << result.mtp_rounds
        << ",\"proposed\":" << result.mtp_proposed
        << ",\"accepted\":" << result.mtp_accepted
        << ",\"fallbacks\":" << result.mtp_fallbacks
        << ",\"depth\":" << result.mtp_final_depth
        << ",\"promotions\":" << result.mtp_promotions
        << ",\"demotions\":" << result.mtp_demotions
        << ",\"profitability_cache_skip\":"
        << (result.mtp_profitability_cache_skip ? "true" : "false")
        << ",\"profitability_cache_keep\":"
        << (result.mtp_profitability_cache_keep ? "true" : "false")
        << ",\"draft_ms\":" << result.mtp_draft_ms
        << ",\"verify_ms\":" << result.mtp_verify_ms
        << ",\"commit_ms\":" << result.mtp_commit_ms << "}"
        << ",\"history_draft\":{\"rounds\":" << result.history_draft_rounds
        << ",\"proposed\":" << result.history_draft_proposed
        << ",\"accepted\":" << result.history_draft_accepted
        << ",\"activations\":" << result.history_draft_activations
        << ",\"deactivations\":" << result.history_draft_deactivations << "}"
        << "}}";
    return out.str();
}

std::string stream_delta_json(
    const RuntimeSnapshot& snapshot,
    const bool chat,
    const std::string_view field,
    const std::string_view delta) {
    std::ostringstream out;
    out << "{\"id\":\"qwen38-native\",\"object\":\""
        << (chat ? "chat.completion.chunk" : "text_completion")
        << "\",\"model\":\"" << json_escape(snapshot.model_id)
        << "\",\"choices\":[{\"index\":0,";
    if (chat) {
        out << "\"delta\":{\"" << field << "\":\""
            << json_escape(delta) << "\"}";
    } else {
        out << "\"text\":\"" << json_escape(delta) << "\"";
    }
    out << ",\"finish_reason\":null}]}";
    return out.str();
}

std::string stream_finish_json(
    const RuntimeSnapshot& snapshot,
    const GenerationResult& result,
    const bool chat,
    const bool include_usage) {
    std::ostringstream out;
    out << "{\"id\":\"qwen38-native\",\"object\":\""
        << (chat ? "chat.completion.chunk" : "text_completion")
        << "\",\"model\":\"" << json_escape(snapshot.model_id)
        << "\",\"choices\":[{\"index\":0,";
    if (chat) out << "\"delta\":{}";
    else out << "\"text\":\"\"";
    out << ",\"finish_reason\":\"" << json_escape(result.finish_reason) << "\"}]";
    if (include_usage) {
        out << ",\"usage\":{\"prompt_tokens\":" << result.prompt_tokens
            << ",\"completion_tokens\":" << result.tokens.size()
            << ",\"total_tokens\":" << result.prompt_tokens + result.tokens.size()
            << ",\"prompt_tokens_details\":{\"cached_tokens\":"
            << result.cached_prompt_tokens << "}}";
    }
    out << '}';
    return out.str();
}

std::string stream_tool_calls_json(
    const RuntimeSnapshot& snapshot,
    const std::vector<ParsedToolCall>& calls) {
    std::ostringstream out;
    out << "{\"id\":\"qwen38-native\",\"object\":\"chat.completion.chunk\",\"model\":\""
        << json_escape(snapshot.model_id)
        << "\",\"choices\":[{\"index\":0,\"delta\":{";
    append_tool_calls_json(out, calls, true);
    out << "},\"finish_reason\":null}]}";
    return out.str();
}

std::string sse_data(const std::string_view json) {
    return "data: " + std::string(json) + "\n\n";
}

std::string status_json(const RuntimeSnapshot& snapshot) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "{\"object\":\"qwen38.runtime_status\",\"state\":\""
        << to_string(snapshot.model_state) << "\",\"model\":\""
        << json_escape(snapshot.model_id) << "\",\"model_path\":\""
        << json_escape(snapshot.model_path) << "\",\"last_error\":\""
        << json_escape(snapshot.last_error) << "\",\"uptime_seconds\":"
        << snapshot.uptime_seconds << ",\"requests\":{\"total\":"
        << snapshot.requests_total << ",\"active\":" << snapshot.requests_active
        << ",\"cancelled\":" << snapshot.requests_cancelled
        << "},\"tokens\":{\"prompt\":" << snapshot.prompt_tokens_total
        << ",\"generated\":" << snapshot.generated_tokens_total << "}}";
    return out.str();
}

} // namespace

HttpResponse Api::handle(const HttpRequest& request) const {
    const auto snapshot = runtime_.snapshot();

    if (request.method == "GET" && request.target == "/healthz") {
        return {.body = "{\"status\":\"ok\"}"};
    }
    if (request.method == "GET" && request.target == "/readyz") {
        if (runtime_.ready()) {
            return {.body = "{\"status\":\"ready\"}"};
        }
        return json_error(503, "model_not_ready", "The model is not ready");
    }
    if (request.method == "GET" && request.target == "/v1/status") {
        return {.body = status_json(snapshot)};
    }
    if (request.method == "GET" && request.target == "/v1/models") {
        if (snapshot.model_id.empty()) {
            return {.body = "{\"object\":\"list\",\"data\":[]}"};
        }
        return {.body = "{\"object\":\"list\",\"data\":[{\"id\":\"" +
            json_escape(snapshot.model_id) +
            "\",\"object\":\"model\",\"owned_by\":\"qwen38-flash.cpp\"}]}"};
    }
    if (request.method == "GET" && request.target == "/metrics") {
        std::ostringstream out;
        out << "# TYPE qwen38_requests_total counter\nqwen38_requests_total "
            << snapshot.requests_total
            << "\n# TYPE qwen38_requests_active gauge\nqwen38_requests_active "
            << snapshot.requests_active
            << "\n# TYPE qwen38_requests_cancelled_total counter\n"
               "qwen38_requests_cancelled_total "
            << snapshot.requests_cancelled
            << "\n# TYPE qwen38_prompt_tokens_total counter\nqwen38_prompt_tokens_total "
            << snapshot.prompt_tokens_total
            << "\n# TYPE qwen38_generated_tokens_total counter\nqwen38_generated_tokens_total "
            << snapshot.generated_tokens_total << '\n';
        return {.content_type = "text/plain; version=0.0.4", .body = out.str()};
    }
    if (request.method == "POST" &&
        (request.target == "/v1/chat/completions" || request.target == "/v1/completions")) {
        if (!runtime_.ready() || engine_ == nullptr) {
            return json_error(503, "model_not_ready", "Inference backend is not loaded");
        }
        try {
            const Json body = Json::parse(request.body);
            const bool stream = streaming_enabled(body);
            const bool include_stream_usage = stream && stream_usage_enabled(body);
            const bool chat = request.target == "/v1/chat/completions";
            ChatTemplateOptions template_options;
            std::string prompt;
            if (chat) {
                template_options = chat_template_options(body);
                std::vector<ChatMessage> messages;
                for (const Json& item : body.at("messages").as_array()) {
                    const Json* reasoning = item.find("reasoning_content");
                    messages.push_back({
                        .role = parse_chat_role(item.at("role").as_string()),
                        .content = message_content(item),
                        .reasoning_content = reasoning == nullptr
                            ? std::nullopt
                            : std::optional<std::string>(reasoning->as_string()),
                        .tool_calls = message_tool_calls(item),
                    });
                }
                if (messages.empty()) throw std::runtime_error("messages must not be empty");
                prompt = render_chat_prompt(messages, template_options);
            } else {
                prompt = body.at("prompt").as_string();
            }
            const std::size_t requested_tokens = max_tokens(body);
            if (stream) {
                return {
                    .status = 200,
                    .content_type = "text/event-stream; charset=utf-8",
                    .body = {},
                    .body_stream = [this, prompt = std::move(prompt), requested_tokens,
                                       chat, template_options, include_stream_usage,
                                       snapshot](const auto& sink) {
                        bool connected = true;
                        const auto emit_json = [&](const std::string_view json) {
                            if (connected) connected = sink(sse_data(json));
                            return connected;
                        };
                        runtime_.request_started();
                        if (chat) {
                            emit_json(stream_delta_json(
                                snapshot, true, "role", "assistant"));
                        }
                        if (!connected) {
                            runtime_.request_finished(0, 0, true);
                            return;
                        }
                        std::string thinking_carry;
                        bool in_reasoning = chat && template_options.enable_thinking;
                        const bool parse_tools = chat && !template_options.tools_json.empty();
                        std::string tool_carry;
                        bool capturing_tool_calls = false;
                        std::size_t captured_tool_deltas = 0;
                        const auto emit_text = [&](const std::string_view field,
                                                   const std::string_view text) {
                            if (!text.empty()) {
                                emit_json(stream_delta_json(snapshot, chat, field, text));
                            }
                            return connected;
                        };
                        const auto emit_content = [&](const std::string_view text) {
                            if (!parse_tools) return emit_text("content", text);
                            tool_carry.append(text);
                            constexpr std::string_view marker = "<tool_call>";
                            if (capturing_tool_calls) {
                                ++captured_tool_deltas;
                                if (connected && captured_tool_deltas % 16 == 0) {
                                    connected = sink(": qwen38 tool-call\n\n");
                                }
                                return connected;
                            }
                            if (const std::size_t start = tool_carry.find(marker);
                                start != std::string::npos) {
                                emit_text("content",
                                    std::string_view(tool_carry).substr(0, start));
                                tool_carry.erase(0, start);
                                capturing_tool_calls = true;
                                return connected;
                            }
                            if (tool_carry.size() >= marker.size()) {
                                const std::size_t ready =
                                    tool_carry.size() - (marker.size() - 1);
                                emit_text("content",
                                    std::string_view(tool_carry).substr(0, ready));
                                tool_carry.erase(0, ready);
                            }
                            return connected;
                        };
                        const auto on_delta = [&](const std::string_view raw_delta) {
                            if (!connected) return false;
                            if (!chat) {
                                emit_text("text", raw_delta);
                                return connected;
                            }
                            if (!in_reasoning) {
                                emit_content(raw_delta);
                                return connected;
                            }
                            constexpr std::string_view close = "</think>";
                            constexpr std::string_view tool_marker = "<tool_call>";
                            thinking_carry.append(raw_delta);
                            if (const std::size_t separator = thinking_carry.find(close);
                                separator != std::string::npos) {
                                emit_text("reasoning_content",
                                    std::string_view(thinking_carry).substr(0, separator));
                                std::string_view content(thinking_carry);
                                content.remove_prefix(separator + close.size());
                                while (!content.empty() && std::isspace(
                                    static_cast<unsigned char>(content.front()))) {
                                    content.remove_prefix(1);
                                }
                                emit_content(content);
                                thinking_carry.clear();
                                in_reasoning = false;
                            } else if (parse_tools &&
                                thinking_carry.find(tool_marker) != std::string::npos) {
                                const std::size_t tool_separator =
                                    thinking_carry.find(tool_marker);
                                emit_text("reasoning_content",
                                    std::string_view(thinking_carry).substr(0, tool_separator));
                                std::string calls = thinking_carry.substr(tool_separator);
                                thinking_carry.clear();
                                in_reasoning = false;
                                emit_content(calls);
                            } else {
                                const std::size_t holdback = parse_tools
                                    ? std::max(close.size(), tool_marker.size()) - 1
                                    : close.size() - 1;
                                if (thinking_carry.size() <= holdback) return connected;
                                const std::size_t ready =
                                    thinking_carry.size() - holdback;
                                emit_text("reasoning_content",
                                    std::string_view(thinking_carry).substr(0, ready));
                                thinking_carry.erase(0, ready);
                            }
                            return connected;
                        };

                        try {
                            GenerationResult result = engine_->complete_stream(
                                prompt, requested_tokens, on_delta);
                            if (!thinking_carry.empty()) {
                                if (in_reasoning) emit_text("reasoning_content", thinking_carry);
                                else emit_content(thinking_carry);
                            }
                            if (parse_tools && !tool_carry.empty()) {
                                if (capturing_tool_calls) {
                                    ParsedToolOutput parsed =
                                        parse_qwen_tool_output(
                                            tool_carry, template_options.tools_json);
                                    if (parsed.calls.empty()) {
                                        emit_text("content", parsed.content);
                                    } else {
                                        emit_text("content", parsed.content);
                                        emit_json(stream_tool_calls_json(snapshot, parsed.calls));
                                        result.finish_reason = "tool_calls";
                                    }
                                } else {
                                    emit_text("content", tool_carry);
                                }
                            }
                            runtime_.request_finished(
                                result.prompt_tokens, result.tokens.size(),
                                result.finish_reason == "cancelled");
                            emit_json(stream_finish_json(
                                snapshot, result, chat, include_stream_usage));
                        } catch (const std::exception& error) {
                            runtime_.request_finished(0, 0);
                            emit_json("{\"error\":{\"code\":\"generation_error\","
                                "\"message\":\"" + json_escape(error.what()) +
                                "\",\"type\":\"server_error\"}}");
                        }
                        if (connected) static_cast<void>(sink("data: [DONE]\n\n"));
                    },
                };
            }
            runtime_.request_started();
            try {
                GenerationResult result = engine_->complete(prompt, requested_tokens);
                runtime_.request_finished(result.prompt_tokens, result.tokens.size());
                return {.body = completion_json(
                    snapshot, result, chat, template_options.enable_thinking,
                    template_options.tools_json)};
            } catch (...) {
                runtime_.request_finished(0, 0);
                throw;
            }
        } catch (const std::exception& error) {
            return json_error(400, "invalid_request", error.what());
        }
    }
    if (request.method == "POST" && request.target == "/admin/cache/clear") {
        if (!runtime_.ready()) {
            return json_error(503, "model_not_ready", "No loaded model cache to clear");
        }
        if (engine_ == nullptr) {
            return json_error(503, "model_not_ready", "Inference backend is not loaded");
        }
        engine_->clear_cache();
        return {.body = "{\"status\":\"cleared\"}"};
    }
    return json_error(404, "not_found", "Route not found");
}

std::string json_escape(const std::string_view input) {
    std::ostringstream out;
    for (const char raw : input) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
        case '\"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(c) << std::dec;
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

} // namespace qwen38
