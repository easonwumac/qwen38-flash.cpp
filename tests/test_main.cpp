#include "test.hpp"

int main() {
    run_json_tests();
    run_chat_template_tests();
    run_safetensors_tests();
    run_model_manifest_tests();
    run_mtp_lifecycle_tests();
    run_ngram_tests();
    run_runtime_tests();
    run_api_tests();
    if (qwen38::test::failures == 0) {
        std::cout << "all tests passed\n";
    }
    return qwen38::test::failures == 0 ? 0 : 1;
}
