#include "qwen38/tool_call.hpp"
#include "test.hpp"

void run_tool_call_tests() {
    const auto parsed = qwen38::parse_qwen_tool_output(
        "I will inspect it.\n\n<tool_call>\n<function=read>\n"
        "<parameter=path>\n/tmp/a\n</parameter>\n"
        "<parameter=lines>\n20\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=notify>\n<parameter=urgent>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>");
    QWEN38_CHECK(parsed.content == "I will inspect it.");
    QWEN38_CHECK(parsed.calls.size() == 2);
    QWEN38_CHECK(parsed.calls[0].name == "read");
    QWEN38_CHECK(parsed.calls[0].arguments_json == R"({"lines":20,"path":"/tmp/a"})");
    QWEN38_CHECK(parsed.calls[1].arguments_json == R"({"urgent":true})");

    const auto object_argument = qwen38::parse_qwen_tool_output(
        "<tool_call><function=write><parameter=data>\n{\"x\":1}\n"
        "</parameter></function></tool_call>");
    QWEN38_CHECK(object_argument.calls.size() == 1);
    QWEN38_CHECK(object_argument.calls[0].arguments_json == R"({"data":{"x":1}})");

    const std::vector<std::string> string_schema{
        R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"id":{"type":"string"}}}}})",
    };
    const auto typed = qwen38::parse_qwen_tool_output(
        "<tool_call><function=lookup><parameter=id>123</parameter>"
        "</function></tool_call>", string_schema);
    QWEN38_CHECK(typed.calls.size() == 1);
    QWEN38_CHECK(typed.calls[0].arguments_json == R"({"id":"123"})");

    const auto no_arguments = qwen38::parse_qwen_tool_output(
        "<tool_call><function=refresh></function></tool_call>");
    QWEN38_CHECK(no_arguments.calls.size() == 1);
    QWEN38_CHECK(no_arguments.calls[0].arguments_json == "{}");

    const std::string malformed =
        "visible <tool_call><function=read><parameter=path>missing close";
    const auto fallback = qwen38::parse_qwen_tool_output(malformed);
    QWEN38_CHECK(fallback.calls.empty());
    QWEN38_CHECK(fallback.content == malformed);
}
