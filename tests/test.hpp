#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qwen38::test {

inline int failures = 0;

inline void check(const bool condition, std::string_view expression, std::string_view file, int line) {
    if (!condition) {
        ++failures;
        std::cerr << file << ':' << line << ": check failed: " << expression << '\n';
    }
}

} // namespace qwen38::test

#define QWEN38_CHECK(expression) \
    ::qwen38::test::check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

void run_api_tests();
void run_chat_template_tests();
void run_history_draft_tests();
void run_json_tests();
void run_model_manifest_tests();
void run_mtp_depth_policy_tests();
void run_mtp_lifecycle_tests();
void run_ngram_tests();
void run_runtime_tests();
void run_safetensors_tests();
