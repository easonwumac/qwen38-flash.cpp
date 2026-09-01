#pragma once

// Adapted from oMLX's gated-delta recurrence kernel (Copyright 2026 oMLX
// contributors), Apache-2.0. oMLX in turn follows the MLX-LM GatedDeltaNet
// implementation. This fixed contract serves Qwen3.8 Flash prefill only.

#include <string_view>

namespace qwen38::gdn_metal {

inline constexpr std::string_view header = R"metal(
#include <metal_stdlib>
using namespace metal;
)metal";

inline constexpr std::string_view recurrence = R"metal(
    const int n = thread_position_in_grid.z;
    const int b_idx = n / Hv;
    const int hv_idx = n % Hv;
    const int hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    const device InT* q_row = q + (b_idx * T * Hk + hk_idx) * Dk;
    const device InT* k_row = k + (b_idx * T * Hk + hk_idx) * Dk;
    const device InT* v_row = v + (b_idx * T * Hv + hv_idx) * Dv;
    device OutT* y_row = y + (b_idx * T * Hv + hv_idx) * Dv;

    const int dk_idx = thread_position_in_threadgroup.x;
    const int dv_idx = thread_position_in_grid.y;
    const device StT* input_state =
        state_in + (n * Dv + dv_idx) * Dk;
    device StT* output_state =
        state_out + (n * Dv + dv_idx) * Dk;
    const device InT* gate_row = g + b_idx * T * Hv;
    const device InT* beta_row = beta + b_idx * T * Hv;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
        const int state_index = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(input_state[state_index]);
    }

    for (int token = 0; token < T; ++token) {
        float recalled = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            const int state_index = n_per_t * dk_idx + i;
            state[i] *= gate_row[hv_idx];
            recalled += state[i] * static_cast<float>(k_row[state_index]);
        }
        recalled = simd_sum(recalled);
        const float delta =
            (static_cast<float>(v_row[dv_idx]) - recalled) *
            static_cast<float>(beta_row[hv_idx]);

        float output = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            const int state_index = n_per_t * dk_idx + i;
            state[i] += static_cast<float>(k_row[state_index]) * delta;
            output += state[i] * static_cast<float>(q_row[state_index]);
        }
        output = simd_sum(output);
        if (thread_index_in_simdgroup == 0) {
            y_row[dv_idx] = static_cast<OutT>(output);
        }

        q_row += Hk * Dk;
        k_row += Hk * Dk;
        v_row += Hv * Dv;
        y_row += Hv * Dv;
        gate_row += Hv;
        beta_row += Hv;
    }

    for (int i = 0; i < n_per_t; ++i) {
        const int state_index = n_per_t * dk_idx + i;
        output_state[state_index] = static_cast<StT>(state[i]);
    }
)metal";

} // namespace qwen38::gdn_metal
