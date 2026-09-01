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
            .text = response_text,
            .tokens = {1, 2},
            .prompt_tokens = 3,
            .cached_prompt_tokens = 2,
            .prompt_ms = 4.0,
            .generation_ms = 5.0,
            .finish_reason = "stop",
        };
    }
    void clear_cache() override { cleared = true; }
    std::string last_prompt;
    std::string response_text{"answer"};
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
    QWEN38_CHECK(completion.body.find("\"cached_tokens\":2") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"mtp\":{\"rounds\":0") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"depth\":0") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"promotions\":0") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"demotions\":0") != std::string::npos);
    QWEN38_CHECK(
        completion.body.find("\"profitability_cache_skip\":false") != std::string::npos);
    QWEN38_CHECK(
        completion.body.find("\"profitability_cache_keep\":false") != std::string::npos);
    QWEN38_CHECK(engine.last_prompt == "hello");
    const auto chat = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":2})"));
    QWEN38_CHECK(chat.status == 200);
    QWEN38_CHECK(engine.last_prompt.find("<|im_start|>user") != std::string::npos);
    engine.response_text = "brief reasoning</think>\n\nfinal answer";
    const auto reasoning_chat = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"assistant","content":"prior","reasoning_content":"prior thought"},{"role":"user","content":"next"}]})"));
    QWEN38_CHECK(reasoning_chat.status == 200);
    QWEN38_CHECK(reasoning_chat.body.find("\"content\":\"final answer\"") != std::string::npos);
    QWEN38_CHECK(reasoning_chat.body.find(
        "\"reasoning_content\":\"brief reasoning\"") != std::string::npos);
    QWEN38_CHECK(engine.last_prompt.find(
        "<think>\nprior thought\n</think>\n\nprior") != std::string::npos);
    engine.response_text = "answer";
    const auto no_think = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"enable_thinking":false})"));
    QWEN38_CHECK(no_think.status == 200);
    QWEN38_CHECK(engine.last_prompt.ends_with("<think>\n\n</think>\n\n"));
    QWEN38_CHECK(inference_api.handle(post("/admin/cache/clear", "{}")).status == 200);
    QWEN38_CHECK(engine.cleared);

    const auto stream = inference_api.handle(post(
        "/v1/completions",
        R"({"prompt":"x","stream":true,"stream_options":{"include_usage":true}})"));
    QWEN38_CHECK(stream.status == 200);
    QWEN38_CHECK(stream.content_type == "text/event-stream; charset=utf-8");
    QWEN38_CHECK(static_cast<bool>(stream.body_stream));
    std::string stream_wire;
    stream.body_stream([&](const std::string_view fragment) {
        stream_wire.append(fragment);
        return true;
    });
    QWEN38_CHECK(stream_wire.find("\"text\":\"answer\"") != std::string::npos);
    QWEN38_CHECK(stream_wire.find("\"completion_tokens\":2") != std::string::npos);
    QWEN38_CHECK(stream_wire.ends_with("data: [DONE]\n\n"));

    engine.response_text = "brief reasoning</think>\n\nfinal answer";
    const auto chat_stream = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"stream":true})"));
    std::string chat_stream_wire;
    chat_stream.body_stream([&](const std::string_view fragment) {
        chat_stream_wire.append(fragment);
        return true;
    });
    QWEN38_CHECK(chat_stream_wire.find("\"role\":\"assistant\"") != std::string::npos);
    QWEN38_CHECK(chat_stream_wire.find(
        "\"reasoning_content\":\"brief reasoning\"") != std::string::npos);
    QWEN38_CHECK(chat_stream_wire.find("\"content\":\"final answer\"") !=
        std::string::npos);
    QWEN38_CHECK(chat_stream_wire.ends_with("data: [DONE]\n\n"));
    QWEN38_CHECK(runtime.snapshot().generated_tokens_total == 12);

    QWEN38_CHECK(qwen38::json_escape("a\n\"b") == "a\\n\\\"b");
}
