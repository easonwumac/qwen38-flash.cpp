#pragma once

#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace qwen38 {

struct PersistedPrefixState {
    explicit PersistedPrefixState(std::size_t layer_count) : target(layer_count) {}

    ModelDecodeState target;
    MtpDecodeState mtp;
    std::optional<MlxArray> previous_target_stream;
    std::vector<MlxArray> pending_mtp_streams;
    std::vector<std::uint32_t> pending_mtp_tokens;
    std::optional<bool> mtp_profitable;
    std::optional<std::uint32_t> mtp_profitability_current_token;
    bool mtp_cumulative_profitability_keep{false};
};

void save_prefix_state(
    const std::filesystem::path& path,
    const PersistedPrefixState& state);

[[nodiscard]] PersistedPrefixState load_prefix_state(
    const std::filesystem::path& path,
    std::size_t expected_layer_count);

} // namespace qwen38
