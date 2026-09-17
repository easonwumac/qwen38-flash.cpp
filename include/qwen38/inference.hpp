#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace qwen38 {

// Returning false requests cancellation at the next committed-token boundary.
using TextDeltaCallback = std::function<bool(std::string_view)>;

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
    std::array<std::size_t, 4> mtp_proposed_by_position{};
    std::array<std::size_t, 4> mtp_accepted_by_position{};
    std::array<std::size_t, 4> mtp_top2_rejected_by_position{};
    std::array<std::size_t, 4> mtp_top2_recovered_by_position{};
    // A recovered second choice is useful to a tree only when the learned
    // head's first proposal from that corrected prefix also matches target.
    std::array<std::size_t, 4> mtp_top2_descendant_recovered_by_position{};
    std::array<std::size_t, 4> mtp_top2_descendant_accepted_by_position{};
    std::size_t mtp_fallbacks{0};
    std::size_t mtp_final_depth{0};
    std::size_t mtp_promotions{0};
    std::size_t mtp_demotions{0};
    bool mtp_profitability_cache_skip{false};
    bool mtp_profitability_cache_keep{false};
    double mtp_draft_ms{0.0};
    double mtp_verify_ms{0.0};
    double mtp_commit_ms{0.0};
    std::size_t history_draft_rounds{0};
    std::size_t history_draft_proposed{0};
    std::size_t history_draft_accepted{0};
    std::size_t history_draft_activations{0};
    std::size_t history_draft_deactivations{0};
    std::size_t context_copy_rounds{0};
    std::size_t context_copy_proposed{0};
    std::size_t context_copy_accepted{0};
    std::size_t context_copy_suspensions{0};
    std::size_t thinking_budget_tokens{0};
    bool thinking_budget_forced{false};
};

struct SamplingOptions {
    float temperature{0.0F};
    float top_p{1.0F};
    std::size_t top_k{0};
    std::uint64_t seed{0};
    float frequency_penalty{0.0F};
    // Zero disables lifecycle intervention. A positive value caps generated
    // reasoning before the engine forces Qwen's early-stop phrase and
    // </think>, then continues the final answer in the same decode state.
    std::size_t thinking_budget_tokens{0};
};

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    [[nodiscard]] virtual GenerationResult complete(
        std::string_view prompt,
        std::size_t max_tokens,
        const SamplingOptions& sampling = {}) = 0;
    [[nodiscard]] virtual GenerationResult complete_stream(
        std::string_view prompt,
        std::size_t max_tokens,
        const TextDeltaCallback& on_delta,
        const SamplingOptions& sampling = {}) {
        GenerationResult result = complete(prompt, max_tokens, sampling);
        if (!result.text.empty() && !on_delta(result.text)) {
            result.finish_reason = "cancelled";
        }
        return result;
    }
    virtual void clear_cache() = 0;
};

} // namespace qwen38
