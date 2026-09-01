#include "qwen38/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qwen38 {
namespace {

class Parser final {
public:
    explicit Parser(const std::string_view source) : source_(source) {}

    Json parse_document() {
        skip_space();
        Json result = parse_value();
        skip_space();
        if (cursor_ != source_.size()) {
            fail("unexpected trailing input");
        }
        return result;
    }

private:
    [[noreturn]] void fail(const std::string_view message) const {
        throw std::runtime_error(
            "JSON parse error at byte " + std::to_string(cursor_) + ": " + std::string(message));
    }

    void skip_space() noexcept {
        while (cursor_ < source_.size()) {
            const char c = source_[cursor_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            ++cursor_;
        }
    }

    bool consume(const char expected) noexcept {
        if (cursor_ < source_.size() && source_[cursor_] == expected) {
            ++cursor_;
            return true;
        }
        return false;
    }

    void expect(const char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    void expect_literal(const std::string_view literal) {
        if (source_.substr(cursor_, literal.size()) != literal) {
            fail("invalid literal");
        }
        cursor_ += literal.size();
    }

    Json parse_value() {
        if (cursor_ >= source_.size()) {
            fail("expected a value");
        }
        switch (source_[cursor_]) {
        case 'n': expect_literal("null"); return Json(nullptr);
        case 't': expect_literal("true"); return Json(true);
        case 'f': expect_literal("false"); return Json(false);
        case '"': return Json(parse_string());
        case '[': return Json(parse_array());
        case '{': return Json(parse_object());
        default:
            if (source_[cursor_] == '-' || (source_[cursor_] >= '0' && source_[cursor_] <= '9')) {
                return parse_number();
            }
            fail("unexpected token");
        }
    }

    static void append_utf8(std::string& output, const std::uint32_t codepoint) {
        if (codepoint <= 0x7FU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0x10FFFFU) {
            output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else {
            throw std::runtime_error("JSON contains an invalid Unicode code point");
        }
    }

    std::uint32_t parse_hex4() {
        if (source_.size() - cursor_ < 4) {
            fail("incomplete Unicode escape");
        }
        std::uint32_t result = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = source_[cursor_++];
            result <<= 4U;
            if (c >= '0' && c <= '9') result |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') result |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') result |= static_cast<std::uint32_t>(c - 'A' + 10);
            else fail("invalid Unicode escape");
        }
        return result;
    }

    std::string parse_string() {
        expect('"');
        std::string output;
        while (cursor_ < source_.size()) {
            const unsigned char c = static_cast<unsigned char>(source_[cursor_++]);
            if (c == '"') {
                return output;
            }
            if (c < 0x20U) {
                fail("unescaped control character");
            }
            if (c != '\\') {
                output.push_back(static_cast<char>(c));
                continue;
            }
            if (cursor_ >= source_.size()) {
                fail("incomplete escape");
            }
            const char escaped = source_[cursor_++];
            switch (escaped) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u': {
                std::uint32_t codepoint = parse_hex4();
                if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                    if (!consume('\\') || !consume('u')) {
                        fail("high surrogate without low surrogate");
                    }
                    const std::uint32_t low = parse_hex4();
                    if (low < 0xDC00U || low > 0xDFFFU) {
                        fail("invalid low surrogate");
                    }
                    codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
                } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
                    fail("unexpected low surrogate");
                }
                append_utf8(output, codepoint);
                break;
            }
            default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    Json::Array parse_array() {
        expect('[');
        skip_space();
        Json::Array result;
        if (consume(']')) return result;
        while (true) {
            skip_space();
            result.push_back(parse_value());
            skip_space();
            if (consume(']')) return result;
            expect(',');
        }
    }

    Json::Object parse_object() {
        expect('{');
        skip_space();
        Json::Object result;
        if (consume('}')) return result;
        while (true) {
            skip_space();
            if (cursor_ >= source_.size() || source_[cursor_] != '"') {
                fail("expected an object key");
            }
            std::string key = parse_string();
            skip_space();
            expect(':');
            skip_space();
            auto [iterator, inserted] = result.emplace(std::move(key), parse_value());
            if (!inserted) {
                fail("duplicate object key");
            }
            skip_space();
            if (consume('}')) return result;
            expect(',');
        }
    }

    Json parse_number() {
        const std::size_t start = cursor_;
        if (consume('-') && cursor_ >= source_.size()) fail("incomplete number");
        if (consume('0')) {
            if (cursor_ < source_.size() && source_[cursor_] >= '0' && source_[cursor_] <= '9') {
                fail("leading zero in number");
            }
        } else {
            if (cursor_ >= source_.size() || source_[cursor_] < '1' || source_[cursor_] > '9') {
                fail("invalid number");
            }
            while (cursor_ < source_.size() && source_[cursor_] >= '0' && source_[cursor_] <= '9') ++cursor_;
        }
        bool floating = false;
        if (consume('.')) {
            floating = true;
            if (cursor_ >= source_.size() || source_[cursor_] < '0' || source_[cursor_] > '9') {
                fail("fraction requires digits");
            }
            while (cursor_ < source_.size() && source_[cursor_] >= '0' && source_[cursor_] <= '9') ++cursor_;
        }
        if (cursor_ < source_.size() && (source_[cursor_] == 'e' || source_[cursor_] == 'E')) {
            floating = true;
            ++cursor_;
            if (cursor_ < source_.size() && (source_[cursor_] == '+' || source_[cursor_] == '-')) ++cursor_;
            if (cursor_ >= source_.size() || source_[cursor_] < '0' || source_[cursor_] > '9') {
                fail("exponent requires digits");
            }
            while (cursor_ < source_.size() && source_[cursor_] >= '0' && source_[cursor_] <= '9') ++cursor_;
        }
        const std::string_view token = source_.substr(start, cursor_ - start);
        if (!floating) {
            std::int64_t integer = 0;
            const auto parsed = std::from_chars(token.data(), token.data() + token.size(), integer);
            if (parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size()) {
                return Json(integer);
            }
        }
        std::string owned(token);
        char* end = nullptr;
        const double number = std::strtod(owned.c_str(), &end);
        if (end != owned.c_str() + owned.size() || !std::isfinite(number)) {
            fail("number is out of range");
        }
        return Json(number);
    }

    std::string_view source_;
    std::size_t cursor_{0};
};

template <typename T>
const T& get_or_throw(const Json::Value& value, const std::string_view expected) {
    const auto* result = std::get_if<T>(&value);
    if (result == nullptr) {
        throw std::runtime_error("JSON value is not " + std::string(expected));
    }
    return *result;
}

std::string quote_json_string(const std::string_view input) {
    std::string output{"\""};
    constexpr char hexadecimal[] = "0123456789abcdef";
    for (const char raw : input) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (c < 0x20U) {
                output += "\\u00";
                output += hexadecimal[c >> 4U];
                output += hexadecimal[c & 0x0fU];
            } else {
                output += static_cast<char>(c);
            }
        }
    }
    output += '"';
    return output;
}

} // namespace

bool Json::is_null() const noexcept { return std::holds_alternative<std::nullptr_t>(value_); }
bool Json::is_object() const noexcept { return std::holds_alternative<Object>(value_); }
bool Json::is_array() const noexcept { return std::holds_alternative<Array>(value_); }
bool Json::is_string() const noexcept { return std::holds_alternative<std::string>(value_); }
const Json::Object& Json::as_object() const { return get_or_throw<Object>(value_, "an object"); }
const Json::Array& Json::as_array() const { return get_or_throw<Array>(value_, "an array"); }
const std::string& Json::as_string() const { return get_or_throw<std::string>(value_, "a string"); }
std::int64_t Json::as_integer() const { return get_or_throw<std::int64_t>(value_, "an integer"); }
bool Json::as_boolean() const { return get_or_throw<bool>(value_, "a boolean"); }

double Json::as_number() const {
    if (const auto* integer = std::get_if<std::int64_t>(&value_)) return static_cast<double>(*integer);
    return get_or_throw<double>(value_, "a number");
}

const Json& Json::at(const std::string_view key) const {
    const Json* result = find(key);
    if (result == nullptr) throw std::out_of_range("missing JSON key: " + std::string(key));
    return *result;
}

const Json* Json::find(const std::string_view key) const noexcept {
    const auto* object = std::get_if<Object>(&value_);
    if (object == nullptr) return nullptr;
    const auto iterator = object->find(std::string(key));
    return iterator == object->end() ? nullptr : &iterator->second;
}

std::string Json::dump() const {
    if (is_null()) return "null";
    if (const auto* value = std::get_if<bool>(&value_)) return *value ? "true" : "false";
    if (const auto* value = std::get_if<std::int64_t>(&value_)) {
        return std::to_string(*value);
    }
    if (const auto* value = std::get_if<double>(&value_)) {
        char buffer[64];
        const auto result = std::to_chars(
            buffer, buffer + sizeof(buffer), *value, std::chars_format::general,
            std::numeric_limits<double>::max_digits10);
        if (result.ec != std::errc{}) throw std::runtime_error("cannot serialize JSON number");
        return std::string(buffer, result.ptr);
    }
    if (const auto* value = std::get_if<std::string>(&value_)) {
        return quote_json_string(*value);
    }
    if (const auto* value = std::get_if<Array>(&value_)) {
        std::string output{"["};
        for (std::size_t index = 0; index < value->size(); ++index) {
            if (index != 0) output += ',';
            output += (*value)[index].dump();
        }
        output += ']';
        return output;
    }
    const auto& object = std::get<Object>(value_);
    std::vector<std::string_view> keys;
    keys.reserve(object.size());
    for (const auto& [key, _] : object) keys.push_back(key);
    std::sort(keys.begin(), keys.end());
    std::string output{"{"};
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (index != 0) output += ',';
        output += quote_json_string(keys[index]);
        output += ':';
        output += object.at(std::string(keys[index])).dump();
    }
    output += '}';
    return output;
}

Json Json::parse(const std::string_view source) { return Parser(source).parse_document(); }

} // namespace qwen38
