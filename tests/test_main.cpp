#include "test.hpp"

int main() {
    run_runtime_tests();
    run_api_tests();
    if (qwen38::test::failures == 0) {
        std::cout << "all tests passed\n";
    }
    return qwen38::test::failures == 0 ? 0 : 1;
}
