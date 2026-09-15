#pragma once

#include <cstddef>
#include <string_view>

namespace qwen38 {

struct AutomaticRuntimeConfig {
    std::size_t allocator_cache_mib = 256;
    std::string_view resident_expert_range;
};

[[nodiscard]] AutomaticRuntimeConfig automatic_runtime_config();
void apply_automatic_runtime_config();
// Developer-only compatibility for reproducing historical benchmark fixtures.
// The production server does not expose profile selection.
void apply_runtime_profile(std::string_view profile);
[[nodiscard]] std::size_t select_prefill_chunk_rows(
    std::size_t configured_rows,
    std::size_t prompt_rows);

} // namespace qwen38
