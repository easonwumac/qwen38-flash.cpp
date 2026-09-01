#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qwen38 {

class Tokenizer final {
public:
    [[nodiscard]] static Tokenizer load(const std::filesystem::path& model_directory);

    [[nodiscard]] std::vector<std::uint32_t> encode(
        std::string_view text,
        bool recognize_special_tokens = true) const;
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> token_ids) const;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return id_to_token_.size(); }

private:
    [[nodiscard]] std::vector<std::uint32_t> encode_ordinary(std::string_view text) const;
    [[nodiscard]] std::vector<std::string> bpe(std::string_view piece) const;

    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::uint32_t> merge_rank_;
    std::unordered_map<std::string, std::uint32_t> special_to_id_;
    std::unordered_set<std::uint32_t> special_ids_;
    std::array<std::string, 256> byte_to_symbol_;
    std::unordered_map<std::uint32_t, std::uint8_t> symbol_to_byte_;
    mutable std::unique_ptr<std::mutex> cache_mutex_{std::make_unique<std::mutex>()};
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
};

} // namespace qwen38
