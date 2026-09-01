#include "qwen38/safetensors.hpp"
#include "test.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace {

std::filesystem::path unique_path(const std::string& suffix) {
    return std::filesystem::temp_directory_path() /
        ("qwen38-test-" + std::to_string(::getpid()) + '-' + suffix);
}

void write_safetensors(const std::filesystem::path& path) {
    const std::string header =
        R"({"__metadata__":null,"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t size = header.size();
    for (unsigned int i = 0; i < 8; ++i) {
        output.put(static_cast<char>((size >> (i * 8U)) & 0xFFU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::array<char, 4> data{1, 2, 3, 4};
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
}

} // namespace

void run_safetensors_tests() {
    const auto path = unique_path("tensor.safetensors");
    write_safetensors(path);
    {
        const qwen38::SafetensorsFile file(path);
        QWEN38_CHECK(file.contains("weight"));
        QWEN38_CHECK(!file.contains("missing"));
        const auto tensor = file.tensor("weight");
        QWEN38_CHECK(tensor.dtype == "U8");
        QWEN38_CHECK(tensor.shape.size() == 1);
        QWEN38_CHECK(tensor.shape[0] == 4);
        QWEN38_CHECK(tensor.bytes.size() == 4);
        QWEN38_CHECK(std::to_integer<unsigned char>(tensor.bytes[2]) == 3);
    }
    std::filesystem::remove(path);
}
