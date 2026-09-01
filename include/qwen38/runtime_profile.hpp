#pragma once

#include <cstddef>
#include <string_view>

namespace qwen38 {

void apply_runtime_profile(std::string_view profile);
[[nodiscard]] std::size_t select_prefill_chunk_rows(
    std::size_t configured_rows,
    std::size_t prompt_rows);

} // namespace qwen38
