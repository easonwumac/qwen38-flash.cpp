#include "qwen38/api.hpp"
#include "test.hpp"

#include <string>

namespace {

qwen38::HttpRequest get(std::string target) {
    return {.method = "GET", .target = std::move(target)};
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

    QWEN38_CHECK(qwen38::json_escape("a\n\"b") == "a\\n\\\"b");
}
