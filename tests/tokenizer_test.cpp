#include "qwen38/tokenizer.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check(
    const qwen38::Tokenizer& tokenizer,
    const std::string& text,
    const std::vector<std::uint32_t>& expected,
    const std::string& expected_decode = {}) {
    const auto actual = tokenizer.encode(text);
    if (actual != expected) {
        std::cerr << "token mismatch for: " << text << "\nactual:";
        for (const auto id : actual) std::cerr << ' ' << id;
        std::cerr << "\nexpected:";
        for (const auto id : expected) std::cerr << ' ' << id;
        std::cerr << '\n';
        return false;
    }
    const std::string decoded = tokenizer.decode(actual);
    const std::string& wanted = expected_decode.empty() ? text : expected_decode;
    if (decoded != wanted) {
        std::cerr << "decode mismatch\nactual: " << decoded << "\nexpected: " << wanted << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    const char* model_path = std::getenv("QWEN38_TEST_MODEL");
    if (model_path == nullptr || *model_path == '\0') {
        std::cout << "QWEN38_TEST_MODEL is unset; skipping real-tokenizer parity\n";
        return 77;
    }
    const qwen38::Tokenizer tokenizer = qwen38::Tokenizer::load(model_path);
    bool passed = true;
    passed &= check(tokenizer, "Hello, world!", {9419, 11, 1814, 0});
    passed &= check(tokenizer, "你好，世界！", {109266, 3709, 96748, 6115});
    passed &= check(
        tokenizer,
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
        {727, 73111, 1393, 1590, 198, 262, 460, 307, 413, 307, 361, 220, 17,
         745, 73111, 1393, 12, 16, 7030, 72758, 38044, 1393, 12, 17, 8});
    passed &= check(
        tokenizer,
        "<|im_start|>user\nExplain MLX.<|im_end|>\n<|im_start|>assistant\n",
        {248045, 846, 198, 814, 20139, 19014, 55, 13, 248046, 198, 248045, 74455, 198});
    passed &= check(
        tokenizer,
        "é café 🚀\n\n",
        {933, 50203, 10838, 248, 222, 271},
        "é café 🚀\n\n");
    passed &= check(tokenizer, "  hello", {220, 23066});
    passed &= check(tokenizer, "hello  world", {14556, 220, 1814});
    passed &= check(tokenizer, "hello   ", {14556, 262});
    passed &= check(tokenizer, "I'm we're DON'T", {40, 2688, 567, 2224, 42803, 16813});
    passed &= check(tokenizer, "12345 67", {16, 17, 18, 19, 20, 220, 21, 22});
    passed &= check(tokenizer, "a\r\n\r\n b", {64, 845, 292});
    passed &= check(tokenizer, "foo\tbar", {7724, 87586});
    passed &= check(tokenizer, "🙂🙂 !!!\n", {169171, 169171, 31780, 198});
    passed &= check(tokenizer, "中文 English 한글 日本語", {99986, 6163, 209758, 220, 247359});
    passed &= check(tokenizer, "A__b... c", {32, 548, 65, 1076, 272});
    passed &= check(tokenizer, " \n \n  x", {30858, 220, 830});
    passed &= check(tokenizer, "x y", {87, 3966, 88});
    return passed ? 0 : 1;
}
