#pragma once

#include "qwen38/runtime.hpp"
#include "qwen38/inference.hpp"

#include <string>
#include <string_view>
#include <unordered_map>

namespace qwen38 {

struct HttpRequest {
    std::string method;
    std::string target;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status{200};
    std::string content_type{"application/json"};
    std::string body;
};

class Api final {
public:
    explicit Api(RuntimeState& runtime, InferenceEngine* engine = nullptr)
        : runtime_(runtime), engine_(engine) {}

    [[nodiscard]] HttpResponse handle(const HttpRequest& request) const;

private:
    RuntimeState& runtime_;
    InferenceEngine* engine_;
};

[[nodiscard]] std::string json_escape(std::string_view input);

} // namespace qwen38
