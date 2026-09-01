#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qwen38 {

struct GenerationResult {
    std::string text;
    std::vector<std::uint32_t> tokens;
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    double prompt_ms{0.0};
    double generation_ms{0.0};
    std::string finish_reason{"length"};
    std::size_t mtp_rounds{0};
    std::size_t mtp_proposed{0};
    std::size_t mtp_accepted{0};
    std::size_t mtp_fallbacks{0};
    std::size_t mtp_final_depth{0};
    std::size_t mtp_promotions{0};
    std::size_t mtp_demotions{0};
};

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    [[nodiscard]] virtual GenerationResult complete(
        std::string_view prompt,
        std::size_t max_tokens) = 0;
    virtual void clear_cache() = 0;
};

} // namespace qwen38
