#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace qwen38 {

class Json final {
public:
    using Array = std::vector<Json>;
    using Object = std::unordered_map<std::string, Json>;
    using Value = std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object>;

    explicit Json(Value value) : value_(std::move(value)) {}

    [[nodiscard]] bool is_null() const noexcept;
    [[nodiscard]] bool is_object() const noexcept;
    [[nodiscard]] bool is_array() const noexcept;
    [[nodiscard]] bool is_string() const noexcept;
    [[nodiscard]] const Object& as_object() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] std::int64_t as_integer() const;
    [[nodiscard]] double as_number() const;
    [[nodiscard]] bool as_boolean() const;
    [[nodiscard]] const Json& at(std::string_view key) const;
    [[nodiscard]] const Json* find(std::string_view key) const noexcept;
    [[nodiscard]] std::string dump() const;

    [[nodiscard]] static Json parse(std::string_view source);

private:
    Value value_;
};

} // namespace qwen38
