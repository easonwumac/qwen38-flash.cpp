#include "qwen38/api.hpp"
#include "test.hpp"

#include <string>
#include <string_view>

namespace {

qwen38::HttpRequest get(std::string target) {
    return {.method = "GET", .target = std::move(target)};
}

class FakeEngine final : public qwen38::InferenceEngine {
public:
    qwen38::GenerationResult complete(std::string_view prompt, std::size_t) override {
        last_prompt = std::string(prompt);
        return {
            .text = "answer",
            .tokens = {1, 2},
            .prompt_tokens = 3,
            .prompt_ms = 4.0,
            .generation_ms = 5.0,
            .finish_reason = "stop",
        };
    }
    void clear_cache() override { cleared = true; }
    std::string last_prompt;
    bool cleared{false};
};

qwen38::HttpRequest post(std::string target, std::string body) {
    return {.method = "POST", .target = std::move(target), .body = std::move(body)};
}

} // namespace

void run_api_tests() {
    qwen38::RuntimeState runtime;
    const qwen38::Api api(runtime);

    QWEN38_CHECK(api.handle(get("/healthz")).status == 200);
    QWEN38_CHECK(api.handle(get("/readyz")).status == 503);
    QWEN38_CHECK(api.handle(get("/missing")).status == 404);
    QWEN38_CHECK(api.handle(get("/v1/models")).body.find("\"data\":[]") != std::string::npos);
    QWEN38_CHECK(api.handle(get("/metrics")).content_type == "text/plain; version=0.0.4");

    runtime.begin_loading("model-path");
    runtime.mark_ready("model-id");
    QWEN38_CHECK(api.handle(get("/readyz")).status == 200);
    QWEN38_CHECK(api.handle(get("/v1/models")).body.find("model-id") != std::string::npos);

    FakeEngine engine;
    const qwen38::Api inference_api(runtime, &engine);
    const auto completion = inference_api.handle(post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":2})"));
    QWEN38_CHECK(completion.status == 200);
    QWEN38_CHECK(completion.body.find("answer") != std::string::npos);
    QWEN38_CHECK(engine.last_prompt == "hello");
    const auto chat = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":2})"));
    QWEN38_CHECK(chat.status == 200);
    QWEN38_CHECK(engine.last_prompt.find("<|im_start|>user") != std::string::npos);
    QWEN38_CHECK(inference_api.handle(post("/admin/cache/clear", "{}")).status == 200);
    QWEN38_CHECK(engine.cleared);
    QWEN38_CHECK(runtime.snapshot().generated_tokens_total == 4);
    QWEN38_CHECK(inference_api.handle(post(
        "/v1/completions", R"({"prompt":"x","stream":true})")).status == 400);

    QWEN38_CHECK(qwen38::json_escape("a\n\"b") == "a\\n\\\"b");
}
