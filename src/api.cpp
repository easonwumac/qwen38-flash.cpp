#include "qwen38/api.hpp"
#include "qwen38/chat_template.hpp"
#include "qwen38/json.hpp"

#include <cstddef>
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
    if (parsed < 1 || parsed > 256) {
        throw std::runtime_error("max_tokens must be between 1 and 256");
    }
    return static_cast<std::size_t>(parsed);
}

void reject_streaming(const Json& body) {
    if (const Json* stream = body.find("stream"); stream != nullptr && stream->as_boolean()) {
        throw std::runtime_error("streaming responses are not implemented yet");
    }
}

ChatTemplateOptions chat_template_options(const Json& body) {
    ChatTemplateOptions options;
    if (const Json* value = body.find("enable_thinking"); value != nullptr) {
        options.enable_thinking = value->as_boolean();
    }
    if (const Json* value = body.find("reasoning_effort"); value != nullptr) {
        const std::string& effort = value->as_string();
        if (effort == "xhigh") options.reasoning_effort = ReasoningEffort::xhigh;
        else if (effort == "medium") options.reasoning_effort = ReasoningEffort::medium;
        else if (effort == "low") options.reasoning_effort = ReasoningEffort::low;
        else throw std::runtime_error("reasoning_effort must be xhigh, medium, or low");
    }
    return options;
}

std::string completion_json(
    const RuntimeSnapshot& snapshot,
    const GenerationResult& result,
    const bool chat) {
    std::ostringstream out;
    out << "{\"id\":\"qwen38-native\",\"object\":\""
        << (chat ? "chat.completion" : "text_completion")
        << "\",\"model\":\"" << json_escape(snapshot.model_id) << "\",\"choices\":[{";
    if (chat) {
        out << "\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
            << json_escape(result.text) << "\"}";
    } else {
        out << "\"index\":0,\"text\":\"" << json_escape(result.text) << '"';
    }
    out << ",\"finish_reason\":\"" << json_escape(result.finish_reason)
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
        << ",\"fallbacks\":" << result.mtp_fallbacks << "}"
        << "}}";
    return out.str();
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
            reject_streaming(body);
            const bool chat = request.target == "/v1/chat/completions";
            std::string prompt;
            if (chat) {
                std::vector<ChatMessage> messages;
                for (const Json& item : body.at("messages").as_array()) {
                    messages.push_back({
                        .role = parse_chat_role(item.at("role").as_string()),
                        .content = item.at("content").as_string(),
                        .reasoning_content = std::nullopt,
                    });
                }
                if (messages.empty()) throw std::runtime_error("messages must not be empty");
                prompt = render_chat_prompt(messages, chat_template_options(body));
            } else {
                prompt = body.at("prompt").as_string();
            }
            runtime_.request_started();
            try {
                GenerationResult result = engine_->complete(prompt, max_tokens(body));
                runtime_.request_finished(result.prompt_tokens, result.tokens.size());
                return {.body = completion_json(snapshot, result, chat)};
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
