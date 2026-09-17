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

// Exact two-row counterpart of down_injection. Each quantized weight is read
// once while both branch accumulators retain the single-row association and
// reduction order.
inline constexpr std::string_view down_injection_branch2 = R"metal(
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.y;
    threadgroup float partial0[8];
    threadgroup float partial1[8];
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
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
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
                float w0 = float((q >> (0 * BITS)) & mask) * scale + bias;
                float w1 = float((q >> (1 * BITS)) & mask) * scale + bias;
                float w2 = float((q >> (2 * BITS)) & mask) * scale + bias;
                float w3 = float((q >> (3 * BITS)) & mask) * scale + bias;
                a0 += float(xn0[k + 0]) * w0;
                a1 += float(xn0[k + 1]) * w1;
                a2 += float(xn0[k + 2]) * w2;
                a3 += float(xn0[k + 3]) * w3;
                b0 += float(xn1[k + 0]) * w0;
                b1 += float(xn1[k + 1]) * w1;
                b2 += float(xn1[k + 2]) * w2;
                b3 += float(xn1[k + 3]) * w3;
            }
        }
        float accumulated0 = simd_sum((a0 + a1) + (a2 + a3));
        float accumulated1 = simd_sum((b0 + b1) + (b2 + b3));
        if (lane == 0) {
            partial0[sg] = accumulated0;
            partial1[sg] = accumulated1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total0 = 0.0f, total1 = 0.0f;
            for (int group = 0; group < 8; ++group) {
                total0 += partial0[group];
                total1 += partial1[group];
            }
            T value0 = T(total0 / float(HC));
            T value1 = T(total1 / float(HC));
            T sigmoid0 = T(1.0f / (1.0f + metal::exp(-float(value0))));
            T sigmoid1 = T(1.0f / (1.0f + metal::exp(-float(value1))));
            activation[row] = value0 * sigmoid0;
            activation[R + row] = value1 * sigmoid1;
        }
    } else if (tid == 0) {
        int c = int(row) - R;
        float total0 = 0.0f, total1 = 0.0f;
        for (int h = 0; h < HC; ++h) {
            total0 += ipart0[h * HC + c];
            total1 += ipart1[h * HC + c];
        }
        T value0 = T(total0), value1 = T(total1);
        T sigmoid0 = T(1.0f / (1.0f + metal::exp(-float(value0))));
        T sigmoid1 = T(1.0f / (1.0f + metal::exp(-float(value1))));
        inject[c] = sigmoid0 * T(2.0f);
        inject[HC + c] = sigmoid1 * T(2.0f);
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

inline constexpr std::string_view up_branch2 = R"metal(
    uint lane = thread_index_in_simdgroup;
    uint column = thread_position_in_grid.y;
    const int values_per_word = 32 / BITS;
    const int packed_r = R / values_per_word;
    const int groups_r = R / GS;
    const int iterations = (packed_r + 31) / 32;
    uint mask = (1u << BITS) - 1u;
    float stream_sum0 = 0.0f, stream_sum1 = 0.0f;
    for (int h = 0; h < HC; ++h) {
        size_t row = size_t(h) * size_t(H) + size_t(column);
        size_t weight_base = row * size_t(packed_r);
        size_t group_base = row * size_t(groups_r);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
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
                    float w0 = float((q >> (0 * BITS)) & mask) * scale + bias;
                    float w1 = float((q >> (1 * BITS)) & mask) * scale + bias;
                    float w2 = float((q >> (2 * BITS)) & mask) * scale + bias;
                    float w3 = float((q >> (3 * BITS)) & mask) * scale + bias;
                    a0 += float(activation[r + 0]) * w0;
                    a1 += float(activation[r + 1]) * w1;
                    a2 += float(activation[r + 2]) * w2;
                    a3 += float(activation[r + 3]) * w3;
                    b0 += float(activation[R + r + 0]) * w0;
                    b1 += float(activation[R + r + 1]) * w1;
                    b2 += float(activation[R + r + 2]) * w2;
                    b3 += float(activation[R + r + 3]) * w3;
                }
            }
        }
        T value0 = T(simd_sum((a0 + a1) + (a2 + a3)));
        T value1 = T(simd_sum((b0 + b1) + (b2 + b3)));
        T gate0 = T(1.0f / (1.0f + metal::exp(-float(value0))));
        T gate1 = T(1.0f / (1.0f + metal::exp(-float(value1))));
        stream_sum0 += float(T(float(gate0) * float(xn0[row])));
        stream_sum1 += float(T(float(gate1) * float(xn1[row])));
    }
    if (lane == 0) {
        mixed[column] = T(float(T(stream_sum0)) / float(HC));
        mixed[H + column] = T(float(T(stream_sum1)) / float(HC));
    }
)metal";

} // namespace qwen38::hc_metal
