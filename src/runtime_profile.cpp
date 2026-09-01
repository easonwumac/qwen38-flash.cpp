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

void apply_runtime_profile(const std::string_view profile) {
    if (profile == "safe") return;
    if (profile != "speed" && profile != "latency") {
        throw std::runtime_error("invalid profile: " + std::string(profile));
    }
    const char* resident_range = profile == "latency" ? "12:34" : "12:28";
    const std::pair<const char*, const char*> settings[]{
        {"QWEN38_RESIDENT_EXPERT_RANGE", resident_range},
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
}

} // namespace qwen38
