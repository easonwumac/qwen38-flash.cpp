#include "qwen38/runtime.hpp"
#include "test.hpp"

void run_runtime_tests() {
    qwen38::RuntimeState runtime;
    QWEN38_CHECK(!runtime.ready());
    QWEN38_CHECK(runtime.snapshot().model_state == qwen38::ModelState::unloaded);

    runtime.begin_loading("/tmp/model");
    QWEN38_CHECK(runtime.snapshot().model_state == qwen38::ModelState::loading);
    QWEN38_CHECK(runtime.snapshot().model_path == "/tmp/model");

    runtime.mark_ready("qwen-test");
    QWEN38_CHECK(runtime.ready());
    QWEN38_CHECK(runtime.snapshot().model_id == "qwen-test");

    runtime.request_started();
    QWEN38_CHECK(runtime.snapshot().requests_active == 1);
    runtime.request_finished(7, 3);
    const auto snapshot = runtime.snapshot();
    QWEN38_CHECK(snapshot.requests_total == 1);
    QWEN38_CHECK(snapshot.requests_active == 0);
    QWEN38_CHECK(snapshot.requests_cancelled == 0);
    QWEN38_CHECK(snapshot.prompt_tokens_total == 7);
    QWEN38_CHECK(snapshot.generated_tokens_total == 3);

    runtime.request_started();
    runtime.request_finished(2, 1, true);
    QWEN38_CHECK(runtime.snapshot().requests_cancelled == 1);
}
