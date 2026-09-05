#pragma once

#include "qwen38/decode_state_io.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace qwen38 {

struct StoredPrefixState {
    std::vector<std::uint32_t> tokens;
    PersistedPrefixState state;
};

class PrefixCacheStore final {
public:
    PrefixCacheStore(
        std::filesystem::path directory,
        std::uint64_t max_bytes,
        std::size_t model_layer_count);

    [[nodiscard]] std::optional<StoredPrefixState> load_longest(
        std::span<const std::uint32_t> prompt_tokens);
    // Returns false when the completed entry is immediately refused by the
    // configured capacity limit. This is not an fsync durability guarantee.
    bool save(
        std::span<const std::uint32_t> tokens,
        const PersistedPrefixState& state);
    void clear();

    [[nodiscard]] const std::filesystem::path& directory() const noexcept {
        return directory_;
    }
    [[nodiscard]] std::uint64_t max_bytes() const noexcept { return max_bytes_; }

private:
    std::filesystem::path directory_;
    std::uint64_t max_bytes_;
    std::size_t model_layer_count_;

    void evict_to_limit();
};

} // namespace qwen38
