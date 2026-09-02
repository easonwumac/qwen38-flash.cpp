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
    qwen38::GenerationResult complete(
        std::string_view prompt,
        const std::size_t max_tokens) override {
        last_prompt = std::string(prompt);
        last_max_tokens = max_tokens;
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
    qwen38::GenerationResult complete_stream(
        std::string_view prompt,
        const std::size_t max_tokens,
        const qwen38::TextDeltaCallback& on_delta) override {
        ++stream_calls;
        qwen38::GenerationResult result = complete(prompt, max_tokens);
        if (!on_delta(result.text)) {
            stream_cancelled = true;
            result.finish_reason = "cancelled";
        }
        return result;
    }
    void clear_cache() override { cleared = true; }
    std::string last_prompt;
    std::size_t last_max_tokens{0};
    std::string response_text{"answer"};
    bool cleared{false};
    std::size_t stream_calls{0};
    bool stream_cancelled{false};
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
    QWEN38_CHECK(
        completion.body.find("\"proposed_by_position\":[0,0,0,0]") != std::string::npos);
    QWEN38_CHECK(
        completion.body.find("\"accepted_by_position\":[0,0,0,0]") != std::string::npos);
    QWEN38_CHECK(
        completion.body.find("\"top2_rejected_by_position\":[0,0,0,0]") !=
        std::string::npos);
    QWEN38_CHECK(
        completion.body.find("\"top2_recovered_by_position\":[0,0,0,0]") !=
        std::string::npos);
    QWEN38_CHECK(completion.body.find("\"depth\":0") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"promotions\":0") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"demotions\":0") != std::string::npos);
    QWEN38_CHECK(
        completion.body.find("\"profitability_cache_skip\":false") != std::string::npos);
    QWEN38_CHECK(
        completion.body.find("\"profitability_cache_keep\":false") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"activations\":0") != std::string::npos);
    QWEN38_CHECK(completion.body.find("\"deactivations\":0") != std::string::npos);
    QWEN38_CHECK(engine.last_prompt == "hello");
    const auto long_completion = inference_api.handle(post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":512})"));
    QWEN38_CHECK(long_completion.status == 200);
    QWEN38_CHECK(engine.last_max_tokens == 512);
    const auto invalid_token_limit = inference_api.handle(post(
        "/v1/completions", R"({"prompt":"hello","max_tokens":0})"));
    QWEN38_CHECK(invalid_token_limit.status == 400);
    const auto chat = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"max_completion_tokens":2})"));
    QWEN38_CHECK(chat.status == 200);
    QWEN38_CHECK(engine.last_prompt.find("<|im_start|>user") != std::string::npos);
    const auto tool_followup = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"tools":[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"messages":[{"role":"user","content":"read it"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"path\":\"/tmp/a\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"done"}]})"));
    QWEN38_CHECK(tool_followup.status == 200);
    QWEN38_CHECK(engine.last_prompt.find("# Tools\n\nYou have access") !=
        std::string::npos);
    QWEN38_CHECK(engine.last_prompt.find(
        "<function=read>\n<parameter=path>\n/tmp/a\n</parameter>") !=
        std::string::npos);
    QWEN38_CHECK(engine.last_prompt.find(
        "<|im_start|>user\n<tool_response>\ndone\n</tool_response><|im_end|>") !=
        std::string::npos);
    engine.response_text =
        "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/b\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto tool_call_response = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"read b"}]})"));
    QWEN38_CHECK(tool_call_response.status == 200);
    QWEN38_CHECK(tool_call_response.body.find("\"finish_reason\":\"tool_calls\"") !=
        std::string::npos);
    QWEN38_CHECK(tool_call_response.body.find("\"content\":\"\"") !=
        std::string::npos);
    QWEN38_CHECK(tool_call_response.body.find(
        "\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/b\\\"}\"") !=
        std::string::npos);
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
    const auto no_think_compat = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"thinking":false})"));
    QWEN38_CHECK(no_think_compat.status == 200);
    QWEN38_CHECK(engine.last_prompt.ends_with("<think>\n\n</think>\n\n"));
    const auto conflicting_thinking = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"thinking":false,"enable_thinking":true})"));
    QWEN38_CHECK(conflicting_thinking.status == 400);
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

    engine.response_text =
        "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/c\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto tool_stream = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"stream":true,"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"read c"}]})"));
    std::string tool_stream_wire;
    tool_stream.body_stream([&](const std::string_view fragment) {
        tool_stream_wire.append(fragment);
        return true;
    });
    QWEN38_CHECK(tool_stream_wire.find("\"tool_calls\":[{\"index\":0") !=
        std::string::npos);
    QWEN38_CHECK(tool_stream_wire.find(
        "\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/c\\\"}\"") !=
        std::string::npos);
    QWEN38_CHECK(tool_stream_wire.find("\"finish_reason\":\"tool_calls\"") !=
        std::string::npos);
    QWEN38_CHECK(tool_stream_wire.find("<tool_call>") == std::string::npos);
    QWEN38_CHECK(tool_stream_wire.ends_with("data: [DONE]\n\n"));

    engine.response_text = "cancel me";
    const auto disconnected_stream = inference_api.handle(post(
        "/v1/completions", R"({"prompt":"x","stream":true})"));
    disconnected_stream.body_stream([](const std::string_view) { return false; });
    QWEN38_CHECK(engine.stream_cancelled);
    const std::size_t stream_calls_before_early_disconnect = engine.stream_calls;
    const auto early_disconnected_chat = inference_api.handle(post(
        "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":"hello"}],"stream":true})"));
    early_disconnected_chat.body_stream(
        [](const std::string_view) { return false; });
    QWEN38_CHECK(engine.stream_calls == stream_calls_before_early_disconnect);
    QWEN38_CHECK(runtime.snapshot().requests_cancelled == 2);
    QWEN38_CHECK(runtime.snapshot().generated_tokens_total == 24);
    QWEN38_CHECK(inference_api.handle(get("/v1/status")).body.find(
        "\"cancelled\":2") != std::string::npos);
    QWEN38_CHECK(inference_api.handle(get("/metrics")).body.find(
        "qwen38_requests_cancelled_total 2") != std::string::npos);

    QWEN38_CHECK(qwen38::json_escape("a\n\"b") == "a\\n\\\"b");
}
