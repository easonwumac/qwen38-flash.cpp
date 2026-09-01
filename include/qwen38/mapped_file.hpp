#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace qwen38 {

class MappedFile final {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    void reset() noexcept;

    void* data_{nullptr};
    std::size_t size_{0};
};

} // namespace qwen38
