#include "qwen38/api.hpp"

#include <iomanip>
#include <sstream>

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
        return json_error(503, "model_not_ready", "Inference backend is not loaded");
    }
    if (request.method == "POST" && request.target == "/admin/cache/clear") {
        if (!runtime_.ready()) {
            return json_error(503, "model_not_ready", "No loaded model cache to clear");
        }
        return json_error(501, "not_implemented", "Cache management is not implemented yet");
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
