#pragma once

#include <string_view>

namespace qwen38::hc_metal {

// Adapted from mlx-serve's MIT-licensed qwen4_exp fused HC read kernels.
// This read-only variant leaves injection projection in the MLX graph so the
// first port can be parity- and performance-gated independently.
inline constexpr std::string_view normalize = R"metal(
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint h = threadgroup_position_in_grid.x;
    threadgroup float partial[8];
    const int base = int(h) * H;
    const int per_thread = H / 256;
    float values[per_thread];
    float square_sum = 0.0f;
    for (int i = 0; i < per_thread; ++i) {
        int k = base + int(tid) + 256 * i;
        values[i] = float(x[k]);
        square_sum += values[i] * values[i];
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0) partial[sg] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int group = 0; group < 8; ++group) total += partial[group];
    float inverse_rms = rsqrt(total / float(H) + eps[0]);
    for (int i = 0; i < per_thread; ++i) {
        int k = base + int(tid) + 256 * i;
        T normalized = T(values[i] * inverse_rms);
        xn[k] = T(float(normalized) * float(norm_weight[k]));
    }
)metal";

inline constexpr std::string_view normalize_injection = R"metal(
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint h = threadgroup_position_in_grid.x;
    threadgroup float partial[8];
    threadgroup float injection_partial[8 * HC];
    const int base = int(h) * H;
    const int per_thread = H / 256;
    float values[per_thread];
    float square_sum = 0.0f;
    for (int i = 0; i < per_thread; ++i) {
        int k = base + int(tid) + 256 * i;
        values[i] = float(x[k]);
        square_sum += values[i] * values[i];
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0) partial[sg] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int group = 0; group < 8; ++group) total += partial[group];
    float inverse_rms = rsqrt(total / float(H) + eps[0]);
    float injection_accum[HC];
    for (int c = 0; c < HC; ++c) injection_accum[c] = 0.0f;
    for (int i = 0; i < per_thread; ++i) {
        int k = base + int(tid) + 256 * i;
        T normalized = T(float(T(values[i] * inverse_rms)) * float(norm_weight[k]));
        xn[k] = normalized;
        for (int c = 0; c < HC; ++c) {
            injection_accum[c] +=
                float(normalized) * float(injection_weight[size_t(k) * HC + c]);
        }
    }
    for (int c = 0; c < HC; ++c) {
        float value = simd_sum(injection_accum[c]);
        if (lane == 0) injection_partial[sg * HC + c] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(HC)) {
        float value = 0.0f;
        for (int group = 0; group < 8; ++group) {
            value += injection_partial[group * HC + tid];
        }
        ipart[h * HC + tid] = value;
    }
)metal";

inline constexpr std::string_view down = R"metal(
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.y;
    threadgroup float partial[8];
    const int K = HC * H;
    const int values_per_word = 32 / BITS;
    const int packed_k = K / values_per_word;
    const int groups_k = K / GS;
    const int slice = packed_k / 8;
    const int iterations = slice / 32;
    uint mask = (1u << BITS) - 1u;
    size_t weight_base = size_t(row) * size_t(packed_k);
    size_t group_base = size_t(row) * size_t(groups_k);
    int packed_start = int(sg) * slice + int(lane);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        int packed_index = packed_start + 32 * i;
        uint32_t packed = weight[weight_base + size_t(packed_index)];
        int k_base = packed_index * values_per_word;
        int group = k_base / GS;
        float scale = float(scales[group_base + size_t(group)]);
        float bias = float(biases[group_base + size_t(group)]);
        for (int ki = 0; ki < values_per_word; ki += 4) {
            int k = k_base + ki;
            uint32_t q = packed >> (ki * BITS);
            a0 += float(xn[k + 0]) * (float((q >> (0 * BITS)) & mask) * scale + bias);
            a1 += float(xn[k + 1]) * (float((q >> (1 * BITS)) & mask) * scale + bias);
            a2 += float(xn[k + 2]) * (float((q >> (2 * BITS)) & mask) * scale + bias);
            a3 += float(xn[k + 3]) * (float((q >> (3 * BITS)) & mask) * scale + bias);
        }
    }
    float accumulated = simd_sum((a0 + a1) + (a2 + a3));
    if (lane == 0) partial[sg] = accumulated;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (int group = 0; group < 8; ++group) total += partial[group];
        T value = T(total / float(HC));
        T sigmoid = T(1.0f / (1.0f + metal::exp(-float(value))));
        activation[row] = value * sigmoid;
    }
)metal";

inline constexpr std::string_view down_injection = R"metal(
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.y;
    threadgroup float partial[8];
    const int K = HC * H;
    const int values_per_word = 32 / BITS;
    const int packed_k = K / values_per_word;
    const int groups_k = K / GS;
    if (row < uint(R)) {
        const int slice = packed_k / 8;
        const int iterations = slice / 32;
        uint mask = (1u << BITS) - 1u;
        size_t weight_base = size_t(row) * size_t(packed_k);
        size_t group_base = size_t(row) * size_t(groups_k);
        int packed_start = int(sg) * slice + int(lane);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int i = 0; i < iterations; ++i) {
            int packed_index = packed_start + 32 * i;
            uint32_t packed = weight[weight_base + size_t(packed_index)];
            int k_base = packed_index * values_per_word;
            int group = k_base / GS;
            float scale = float(scales[group_base + size_t(group)]);
            float bias = float(biases[group_base + size_t(group)]);
            for (int ki = 0; ki < values_per_word; ki += 4) {
                int k = k_base + ki;
                uint32_t q = packed >> (ki * BITS);
                a0 += float(xn[k + 0]) *
                    (float((q >> (0 * BITS)) & mask) * scale + bias);
                a1 += float(xn[k + 1]) *
                    (float((q >> (1 * BITS)) & mask) * scale + bias);
                a2 += float(xn[k + 2]) *
                    (float((q >> (2 * BITS)) & mask) * scale + bias);
                a3 += float(xn[k + 3]) *
                    (float((q >> (3 * BITS)) & mask) * scale + bias);
            }
        }
        float accumulated = simd_sum((a0 + a1) + (a2 + a3));
        if (lane == 0) partial[sg] = accumulated;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (int group = 0; group < 8; ++group) total += partial[group];
            T value = T(total / float(HC));
            T sigmoid = T(1.0f / (1.0f + metal::exp(-float(value))));
            activation[row] = value * sigmoid;
        }
    } else if (tid == 0) {
        int c = int(row) - R;
        float total = 0.0f;
        for (int h = 0; h < HC; ++h) total += ipart[h * HC + c];
        T value = T(total);
        T sigmoid = T(1.0f / (1.0f + metal::exp(-float(value))));
        inject[c] = sigmoid * T(2.0f);
    }
)metal";

inline constexpr std::string_view up = R"metal(
    uint lane = thread_index_in_simdgroup;
    uint column = thread_position_in_grid.y;
    const int values_per_word = 32 / BITS;
    const int packed_r = R / values_per_word;
    const int groups_r = R / GS;
    const int iterations = (packed_r + 31) / 32;
    uint mask = (1u << BITS) - 1u;
    float stream_sum = 0.0f;
    for (int h = 0; h < HC; ++h) {
        size_t row = size_t(h) * size_t(H) + size_t(column);
        size_t weight_base = row * size_t(packed_r);
        size_t group_base = row * size_t(groups_r);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int i = 0; i < iterations; ++i) {
            int packed_index = int(lane) + 32 * i;
            if (packed_index < packed_r) {
                uint32_t packed = weight[weight_base + size_t(packed_index)];
                int r_base = packed_index * values_per_word;
                int group = r_base / GS;
                float scale = float(scales[group_base + size_t(group)]);
                float bias = float(biases[group_base + size_t(group)]);
                for (int ri = 0; ri < values_per_word; ri += 4) {
                    int r = r_base + ri;
                    uint32_t q = packed >> (ri * BITS);
                    a0 += float(activation[r + 0]) * (float((q >> (0 * BITS)) & mask) * scale + bias);
                    a1 += float(activation[r + 1]) * (float((q >> (1 * BITS)) & mask) * scale + bias);
                    a2 += float(activation[r + 2]) * (float((q >> (2 * BITS)) & mask) * scale + bias);
                    a3 += float(activation[r + 3]) * (float((q >> (3 * BITS)) & mask) * scale + bias);
                }
            }
        }
        float accumulated = simd_sum((a0 + a1) + (a2 + a3));
        T value = T(accumulated);
        T gate = T(1.0f / (1.0f + metal::exp(-float(value))));
        stream_sum += float(T(float(gate) * float(xn[row])));
    }
    if (lane == 0) mixed[column] = T(float(T(stream_sum)) / float(HC));
)metal";

} // namespace qwen38::hc_metal
