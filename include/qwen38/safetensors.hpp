#pragma once

#include "qwen38/mapped_file.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qwen38 {

struct TensorMetadata {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t begin{0};
    std::size_t end{0};
};

struct TensorView {
    std::string_view dtype;
    std::span<const std::size_t> shape;
    std::span<const std::byte> bytes;
};

class SafetensorsFile final {
public:
    explicit SafetensorsFile(const std::filesystem::path& path);

    [[nodiscard]] const std::unordered_map<std::string, TensorMetadata>& tensors() const noexcept {
        return tensors_;
    }
    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] TensorView tensor(std::string_view name) const;
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return file_.size(); }

private:
    MappedFile file_;
    std::size_t data_offset_{0};
    std::unordered_map<std::string, TensorMetadata> tensors_;
};

} // namespace qwen38
