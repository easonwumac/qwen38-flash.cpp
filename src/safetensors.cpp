#include "qwen38/safetensors.hpp"

#include "qwen38/json.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace qwen38 {
namespace {

std::uint64_t read_u64_le(const std::span<const std::byte> bytes) {
    if (bytes.size() < sizeof(std::uint64_t)) {
        throw std::runtime_error("safetensors file is shorter than its length prefix");
    }
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
        result |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[i])) << (i * 8U);
    }
    return result;
}

std::size_t checked_size(const std::int64_t value, const std::string_view field) {
    if (value < 0 || static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("invalid safetensors " + std::string(field));
    }
    return static_cast<std::size_t>(value);
}

} // namespace

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : file_(path) {
    const auto bytes = file_.bytes();
    const std::uint64_t header_size_u64 = read_u64_le(bytes);
    if (header_size_u64 > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("safetensors header is too large: " + path.string());
    }
    const auto header_size = static_cast<std::size_t>(header_size_u64);
    if (header_size > bytes.size() - sizeof(std::uint64_t)) {
        throw std::runtime_error("safetensors header exceeds file size: " + path.string());
    }
    data_offset_ = sizeof(std::uint64_t) + header_size;
    const auto* header_data = reinterpret_cast<const char*>(bytes.data() + sizeof(std::uint64_t));
    const Json header = Json::parse(std::string_view(header_data, header_size));

    for (const auto& [name, description] : header.as_object()) {
        if (name == "__metadata__") continue;
        TensorMetadata metadata;
        metadata.dtype = description.at("dtype").as_string();
        for (const Json& dimension : description.at("shape").as_array()) {
            metadata.shape.push_back(checked_size(dimension.as_integer(), "shape"));
        }
        const auto& offsets = description.at("data_offsets").as_array();
        if (offsets.size() != 2) {
            throw std::runtime_error("tensor data_offsets must contain two values: " + name);
        }
        metadata.begin = checked_size(offsets[0].as_integer(), "begin offset");
        metadata.end = checked_size(offsets[1].as_integer(), "end offset");
        if (metadata.begin > metadata.end || metadata.end > bytes.size() - data_offset_) {
            throw std::runtime_error("tensor data range exceeds shard: " + name);
        }
        tensors_.emplace(name, std::move(metadata));
    }
}

bool SafetensorsFile::contains(const std::string_view name) const {
    return tensors_.contains(std::string(name));
}

TensorView SafetensorsFile::tensor(const std::string_view name) const {
    const auto iterator = tensors_.find(std::string(name));
    if (iterator == tensors_.end()) {
        throw std::out_of_range("tensor not found: " + std::string(name));
    }
    const TensorMetadata& metadata = iterator->second;
    return {
        .dtype = metadata.dtype,
        .shape = metadata.shape,
        .bytes = file_.bytes().subspan(data_offset_ + metadata.begin, metadata.end - metadata.begin),
    };
}

} // namespace qwen38
