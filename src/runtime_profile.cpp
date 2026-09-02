#include "qwen38/runtime_profile.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace qwen38 {
namespace {

void set_environment_default(const char* name, const char* value) {
    if (setenv(name, value, 0) != 0) {
        throw std::runtime_error(std::string("cannot set runtime profile option ") + name);
    }
}

} // namespace

RuntimeProfileConfig runtime_profile_config(const std::string_view profile) {
    if (profile == "safe") return {};
    if (profile == "speed") return {.optimized = true, .resident_expert_range = "12:28"};
    if (profile == "latency") return {.optimized = true, .resident_expert_range = "12:34"};
    if (profile == "long-context") {
        return {.optimized = true, .resident_expert_range = ""};
    }
    if (profile == "memory") {
        return {
            .optimized = true,
            .memory_efficient = true,
            .resident_expert_range = "",
        };
    }
    throw std::runtime_error("invalid profile: " + std::string(profile));
}

void apply_runtime_profile(const std::string_view profile) {
    const RuntimeProfileConfig config = runtime_profile_config(profile);
    if (!config.optimized) return;
    const std::pair<const char*, const char*> settings[]{
        {"QWEN38_RESIDENT_EXPERT_RANGE", config.resident_expert_range.data()},
        {"QWEN38_FUSED_MOE", "1"},
        {"QWEN38_DEVICE_ROUTER", "1"},
        {"QWEN38_COMPILE_LAYER", "1"},
        {"QWEN38_HC_FUSED", "1"},
        {"QWEN38_HC_FUSED_INJECTION", "1"},
        {"QWEN38_GDN_NORM_GATE", "1"},
        {"QWEN38_GDN_PREWORK", "1"},
        {"QWEN38_GDN_METAL_VERIFY_BF16_SUM", "1"},
        {"QWEN38_BATCH_KV_VERIFY", "1"},
        {"QWEN38_SDPA_PREFILL", "1"},
        {"QWEN38_GDN_METAL_PREFILL", "1"},
        {"QWEN38_GROUPED_PREFILL", "1"},
        {"QWEN38_PREFILL_BARRIER_STRIDE", "8"},
        {"QWEN38_SELECTED_SOFTMAX_ROUTER", "1"},
        {"QWEN38_MTP_EARLY_DEMOTION", "1"},
        {"QWEN38_MTP_DEMOTION", "1"},
        {"QWEN38_Q8_EXACT_MOE", "1"},
        {"QWEN38_MTP_CUMULATIVE_PROFITABILITY_CACHE", "1"},
        {"QWEN38_EXTEND_PREFIX_CACHE", "1"},
    };
    for (const auto& [name, value] : settings) set_environment_default(name, value);
    if (config.memory_efficient) {
        set_environment_default("QWEN38_COMPACT_QMETA", "lossless13");
        set_environment_default("QWEN38_QMETA_PREFILL_CACHE", "0");
        set_environment_default("QWEN38_QMETA_PREFILL_DEFER_TEMPORARY", "1");
        set_environment_default("QWEN38_QSA_PACKED_PREFILL", "1");
    }
}

std::size_t select_prefill_chunk_rows(
    const std::size_t configured_rows,
    const std::size_t prompt_rows) {
    if (configured_rows == 0 || configured_rows > 1024) {
        throw std::runtime_error("prefill chunk rows must be between 1 and 1024");
    }
    if (configured_rows <= 256 || prompt_rows <= 32768) return configured_rows;
    if (configured_rows > 512) return 512;
    return 128;
}

} // namespace qwen38
