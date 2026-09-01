#pragma once

// Adapted from oMLX's gated-delta recurrence kernel (Copyright 2026 oMLX
// contributors), Apache-2.0. oMLX in turn follows the MLX-LM GatedDeltaNet
// implementation. These fixed contracts serve Qwen3.8 Flash prefill and
// exact low-row MTP verification.

#include <string_view>

namespace qwen38::gdn_metal {

inline constexpr std::string_view header = R"metal(
#include <metal_stdlib>
using namespace metal;
)metal";

inline constexpr std::string_view prework_header = R"metal(
inline float qwen38_log1p(float x) {
    float plus_one = 1.0f + x;
    if (plus_one == metal::numeric_limits<float>::max()) {
        return metal::numeric_limits<float>::max();
    }
    if (plus_one == 1.0f) return x;
    return x * (metal::log(plus_one) / (plus_one - 1.0f));
}
)metal";

// Decode-width packed prework adapted from mlx-serve's MIT-licensed kernel.
inline constexpr std::string_view prework = R"metal(
    uint lane = thread_position_in_threadgroup.x;
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint key_head_base = uint(HK);
    constexpr uint value_head_base = 2 * uint(HK);
    bool is_query = logical_head < q_heads;
    bool is_key = logical_head >= key_head_base && logical_head < value_head_base;
    uint head = is_query ? logical_head
        : (is_key ? logical_head - key_head_base : logical_head - value_head_base);
    uint channel_base = is_query ? head * uint(DK)
        : (is_key ? uint(HK) * uint(DK) + head * uint(DK)
                  : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float square_sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float accumulated = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            T value = tap < 3
                ? conv_state[tap * uint(C) + channel]
                : qkv[channel];
            accumulated += float(value) * float(conv_weight[channel * 4 + tap]);
        }
        T convolved = T(accumulated);
        T small = T(1) / (T(1) + metal::exp(metal::abs(convolved)));
        T value = convolved * ((convolved < T(0)) ? small : T(1) - small);
        activated[i] = value;
        square_sum += float(value) * float(value);
        conv_out[(2 * uint(C)) + channel] = qkv[channel];
    }
    if (is_query || is_key) {
        square_sum = simd_sum(square_sum);
        float inverse_rms = metal::precise::rsqrt(square_sum / float(DK) + 1e-6f);
        T scale = is_query ? q_scale : k_scale;
        uint output_base = head * uint(DK) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            T rms = T(float(activated[i]) * inverse_rms);
            T value = scale * rms;
            if (is_query) q_out[output_base + i] = value;
            else k_out[output_base + i] = value;
        }
    } else {
        uint output_base = head * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) v_out[output_base + i] = activated[i];
        if (lane == 0) {
            T beta_value = beta_in[head];
            T small = T(1) / (T(1) + metal::exp(metal::abs(beta_value)));
            beta_out[head] = (beta_value < T(0)) ? small : T(1) - small;
            T biased = T(float(decay_in[head]) + float(decay_bias[head]));
            float softplus = qwen38_log1p(metal::precise::exp(float(biased)));
            float rate = metal::precise::exp(float(decay_log[head]));
            decay_out[head] = T(metal::precise::exp(-(rate * softplus)));
        }
    }
    if (logical_head < uint(2 * HK + HV)) {
        for (uint i = 0; i < 4; ++i) {
            uint channel = channel_base + lane * 4 + i;
            conv_out[channel] = conv_state[uint(C) + channel];
            conv_out[uint(C) + channel] = conv_state[2 * uint(C) + channel];
        }
    }
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

inline constexpr std::string_view verify_recurrence_bf16_sum = R"metal(
    const int hv_idx = thread_position_in_grid.z;
    const int lane = thread_position_in_threadgroup.x;
    const int dv_idx = thread_position_in_grid.y;
    constexpr int ITEMS = DK / 32;
    const device InT* q_row = q + hv_idx * DK;
    const device InT* k_row = k + hv_idx * DK;
    const device InT* v_row = v + hv_idx * DV;
    const device InT* gate_row = g + hv_idx;
    const device InT* beta_row = beta + hv_idx;
    const device StT* input_state = state_in + (hv_idx * DV + dv_idx) * DK;
    StT state[ITEMS];
    for (int i = 0; i < ITEMS; ++i) state[i] = input_state[lane * ITEMS + i];
    for (int token = 0; token < ROWS; ++token) {
        InT recalled = InT(0);
        for (int i = 0; i < ITEMS; ++i) {
            const int index = lane * ITEMS + i;
            state[i] = StT(float(state[i]) * float(gate_row[0]));
            InT product = InT(float(state[i]) * float(k_row[index]));
            recalled = InT(float(recalled) + float(product));
        }
        recalled = simd_sum(recalled);
        InT residual = InT(float(v_row[dv_idx]) - float(recalled));
        InT delta = InT(float(residual) * float(beta_row[0]));
        InT output = InT(0);
        for (int i = 0; i < ITEMS; ++i) {
            const int index = lane * ITEMS + i;
            StT update = StT(float(k_row[index]) * float(delta));
            state[i] = StT(float(state[i]) + float(update));
            InT product = InT(float(state[i]) * float(q_row[index]));
            output = InT(float(output) + float(product));
            state_rows[(((token * HV + hv_idx) * DV + dv_idx) * DK) + index] = state[i];
        }
        output = simd_sum(output);
        if (thread_index_in_simdgroup == 0) {
            y[(token * HV + hv_idx) * DV + dv_idx] = OutT(output);
        }
        q_row += HV * DK;
        k_row += HV * DK;
        v_row += HV * DV;
        gate_row += HV;
        beta_row += HV;
    }
)metal";

// Adapted from mlx-serve's MIT-licensed fused GDN norm/gate epilogue.
inline constexpr std::string_view norm_gate = R"metal(
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint head = threadgroup_position_in_grid.z;
    uint base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    float values[4];
    float square_sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        values[i] = float(y[base + i]);
        square_sum += values[i] * values[i];
    }
    square_sum = simd_sum(square_sum);
    float inverse_rms = metal::precise::rsqrt(square_sum / float(DV) + epsilon[0]);
    uint zbase = row * uint(HV * DV) + head * uint(DV) + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        T normalized = norm_weight[lane * 4 + i] * T(values[i] * inverse_rms);
        T z_value = z[zbase + i];
        T small = T(1) / (T(1) + metal::exp(metal::abs(z_value)));
        T sigmoid = (z_value < T(0)) ? small : T(1) - small;
        output[base + i] = SWISH ? (z_value * sigmoid) * normalized : normalized * sigmoid;
    }
)metal";

} // namespace qwen38::gdn_metal
