#include "qwen38/json.hpp"
#include "test.hpp"

#include <stdexcept>
#include <string>

void run_json_tests() {
    const qwen38::Json value = qwen38::Json::parse(
        R"({"name":"Qwen\nFlash","integer":42,"float":1.25,"ok":true,"none":null,"array":[1,"\u4e2d\u6587","\ud83d\ude80"]})");
    QWEN38_CHECK(value.at("name").as_string() == "Qwen\nFlash");
    QWEN38_CHECK(value.at("integer").as_integer() == 42);
    QWEN38_CHECK(value.at("float").as_number() == 1.25);
    QWEN38_CHECK(value.at("ok").as_boolean());
    QWEN38_CHECK(value.at("none").is_null());
    QWEN38_CHECK(value.at("array").as_array().at(1).as_string() == "中文");
    QWEN38_CHECK(value.at("array").as_array().at(2).as_string() == "🚀");
    QWEN38_CHECK(value.find("missing") == nullptr);

    bool rejected_duplicate = false;
    try {
        static_cast<void>(qwen38::Json::parse(R"({"x":1,"x":2})"));
    } catch (const std::runtime_error&) {
        rejected_duplicate = true;
    }
    QWEN38_CHECK(rejected_duplicate);
}
