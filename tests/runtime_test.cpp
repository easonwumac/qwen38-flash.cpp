#include "qwen38/runtime.hpp"
#include "qwen38/runtime_profile.hpp"
#include "test.hpp"

void run_runtime_tests() {
    QWEN38_CHECK(qwen38::select_prefill_chunk_rows(512, 512) == 512);
    QWEN38_CHECK(qwen38::select_prefill_chunk_rows(512, 513) == 384);
    QWEN38_CHECK(qwen38::select_prefill_chunk_rows(512, 6144) == 384);
    QWEN38_CHECK(qwen38::select_prefill_chunk_rows(512, 6145) == 256);
    QWEN38_CHECK(qwen38::select_prefill_chunk_rows(512, 8193) == 128);
    QWEN38_CHECK(qwen38::select_prefill_chunk_rows(64, 262144) == 64);

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
