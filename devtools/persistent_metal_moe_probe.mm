#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38/hyper_connection.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model_manifest.hpp"
#include "qwen38/sparse_moe.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using qwen38::MlxArray;
constexpr int hidden_size = 2560;
constexpr int expert_width = 448;
constexpr int expert_count = 512;
constexpr int top_k = 10;
constexpr int group_size = 64;

constexpr std::string_view metal_source = R"metal(
#include <metal_stdlib>
using namespace metal;

inline float q3_dot8(const device uchar* packed, const device bfloat* x) {
    const uint bits = uint(packed[0]) | (uint(packed[1]) << 8) |
        (uint(packed[2]) << 16);
    float value = 0.0f;
    for (uint i = 0; i < 8; ++i) value += float((bits >> (3 * i)) & 7) * float(x[i]);
    return value;
}

inline float q4_dot8(const device uchar* packed, const device bfloat* x) {
    const uint bits = uint(packed[0]) | (uint(packed[1]) << 8) |
        (uint(packed[2]) << 16) | (uint(packed[3]) << 24);
    float value = 0.0f;
    for (uint i = 0; i < 8; ++i) value += float((bits >> (4 * i)) & 15) * float(x[i]);
    return value;
}

inline float q8_dot4(const device uchar* packed, const device bfloat* x) {
    float value = 0.0f;
    for (uint i = 0; i < 4; ++i) value += float(packed[i]) * float(x[i]);
    return value;
}

kernel void q3_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* gate_weight [[buffer(1)]],
    const device bfloat* gate_scale [[buffer(2)]],
    const device bfloat* gate_bias [[buffer(3)]],
    const device uchar* up_weight [[buffer(4)]],
    const device bfloat* up_scale [[buffer(5)]],
    const device bfloat* up_bias [[buffer(6)]],
    const device uint* experts [[buffer(7)]],
    device bfloat* output [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 448, k = 2560, row_bytes = k / 8 * 3, groups = k / 64;
    const uint linear = group * 4 + simd;
    if (linear >= 10 * rows) return;
    const uint slot = linear / rows, row = linear % rows, expert = experts[slot];
    const ulong matrix_row = ulong(expert) * rows + row;
    const device uchar* gw = gate_weight + matrix_row * row_bytes;
    const device uchar* uw = up_weight + matrix_row * row_bytes;
    const device bfloat* gs = gate_scale + matrix_row * groups;
    const device bfloat* gb = gate_bias + matrix_row * groups;
    const device bfloat* us = up_scale + matrix_row * groups;
    const device bfloat* ub = up_bias + matrix_row * groups;
    float gate = 0.0f, up = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
        const uint quant = base / 8 * 3, affine = base / 64;
        gate += float(gs[affine]) * q3_dot8(gw + quant, x + base) + float(gb[affine]) * sum;
        up += float(us[affine]) * q3_dot8(uw + quant, x + base) + float(ub[affine]) * sum;
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (lane == 0) {
        const float rounded_gate = float(bfloat(gate));
        const float rounded_up = float(bfloat(up));
        output[slot * rows + row] = bfloat(
            float(bfloat(rounded_gate / (1.0f + metal::exp(-rounded_gate)))) * rounded_up);
    }
}

kernel void q3_down_reduce(
    const device bfloat* input [[buffer(0)]],
    const device uchar* weight [[buffer(1)]],
    const device bfloat* scale [[buffer(2)]],
    const device bfloat* bias [[buffer(3)]],
    const device uint* experts [[buffer(4)]],
    const device bfloat* route_weights [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 448, row_bytes = k / 8 * 3, groups = k / 64;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    bfloat total = bfloat(0.0f);
    for (uint slot = 0; slot < 10; ++slot) {
        const ulong matrix_row = ulong(experts[slot]) * rows + row;
        const device uchar* wr = weight + matrix_row * row_bytes;
        const device bfloat* sr = scale + matrix_row * groups;
        const device bfloat* br = bias + matrix_row * groups;
        const device bfloat* xv = input + slot * k;
        float dot = 0.0f;
        for (uint base = lane * 8; base < k; base += 256) {
            float sum = 0.0f;
            for (uint i = 0; i < 8; ++i) sum += float(xv[base + i]);
            const uint quant = base / 8 * 3, affine = base / 64;
            dot += float(sr[affine]) * q3_dot8(wr + quant, xv + base) +
                float(br[affine]) * sum;
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            const bfloat weighted = bfloat(float(route_weights[slot]) * float(bfloat(dot)));
            total = bfloat(float(total) + float(weighted));
        }
    }
    if (lane == 0) output[row] = total;
}

kernel void q4_shared_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* gate_weight [[buffer(1)]],
    const device bfloat* gate_scale [[buffer(2)]],
    const device bfloat* gate_bias [[buffer(3)]],
    const device uchar* up_weight [[buffer(4)]],
    const device bfloat* up_scale [[buffer(5)]],
    const device bfloat* up_bias [[buffer(6)]],
    device bfloat* output [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 640, k = 2560, row_bytes = k / 2, groups = k / 32;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    const device uchar* gw = gate_weight + row * row_bytes;
    const device uchar* uw = up_weight + row * row_bytes;
    const device bfloat* gs = gate_scale + row * groups;
    const device bfloat* gb = gate_bias + row * groups;
    const device bfloat* us = up_scale + row * groups;
    const device bfloat* ub = up_bias + row * groups;
    float gate = 0.0f, up = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
        const uint quant = base / 2, affine = base / 32;
        gate += float(gs[affine]) * q4_dot8(gw + quant, x + base) + float(gb[affine]) * sum;
        up += float(us[affine]) * q4_dot8(uw + quant, x + base) + float(ub[affine]) * sum;
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (lane == 0) {
        const float rounded_gate = float(bfloat(gate));
        const float rounded_up = float(bfloat(up));
        output[row] = bfloat(
            float(bfloat(rounded_gate / (1.0f + metal::exp(-rounded_gate)))) * rounded_up);
    }
}

kernel void q8_shared_router(
    const device bfloat* x [[buffer(0)]],
    const device uchar* weight [[buffer(1)]],
    const device bfloat* scale [[buffer(2)]],
    const device bfloat* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 2560, groups = k / 64;
    float dot = 0.0f;
    for (uint base = lane * 4; base < k; base += 128) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) sum += float(x[base + i]);
        dot += float(scale[base / 64]) * q8_dot4(weight + base, x + base) +
            float(bias[base / 64]) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) {
        const float rounded = float(bfloat(dot));
        output[0] = bfloat(1.0f / (1.0f + metal::exp(-rounded)));
    }
}

kernel void q4_shared_down_merge(
    const device bfloat* input [[buffer(0)]],
    const device uchar* weight [[buffer(1)]],
    const device bfloat* scale [[buffer(2)]],
    const device bfloat* bias [[buffer(3)]],
    const device bfloat* router [[buffer(4)]],
    const device bfloat* routed [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 640, row_bytes = k / 2, groups = k / 32;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    const device uchar* wr = weight + row * row_bytes;
    const device bfloat* sr = scale + row * groups;
    const device bfloat* br = bias + row * groups;
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(input[base + i]);
        dot += float(sr[base / 32]) * q4_dot8(wr + base / 2, input + base) +
            float(br[base / 32]) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) {
        const bfloat shared = bfloat(float(bfloat(dot)) * float(router[0]));
        output[row] = bfloat(float(routed[row]) + float(shared));
    }
}

kernel void fused_all_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* rgw [[buffer(1)]], const device bfloat* rgs [[buffer(2)]],
    const device bfloat* rgb [[buffer(3)]], const device uchar* ruw [[buffer(4)]],
    const device bfloat* rus [[buffer(5)]], const device bfloat* rub [[buffer(6)]],
    const device uint* experts [[buffer(7)]], device bfloat* routed [[buffer(8)]],
    const device uchar* sgw [[buffer(9)]], const device bfloat* sgs [[buffer(10)]],
    const device bfloat* sgb [[buffer(11)]], const device uchar* suw [[buffer(12)]],
    const device bfloat* sus [[buffer(13)]], const device bfloat* sub [[buffer(14)]],
    device bfloat* shared [[buffer(15)]], const device uchar* srw [[buffer(16)]],
    const device bfloat* srs [[buffer(17)]], const device bfloat* srb [[buffer(18)]],
    device bfloat* router [[buffer(19)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint linear = group * 4 + simd;
    if (linear < 4480) {
        constexpr uint rows = 448, k = 2560, row_bytes = 960, groups = 40;
        const uint slot = linear / rows, row = linear % rows, expert = experts[slot];
        const ulong matrix_row = ulong(expert) * rows + row;
        float gate = 0.0f, up = 0.0f;
        for (uint base = lane * 8; base < k; base += 256) {
            float sum = 0.0f;
            for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
            const uint quant = base / 8 * 3, affine = base / 64;
            gate += float(rgs[matrix_row * groups + affine]) *
                    q3_dot8(rgw + matrix_row * row_bytes + quant, x + base) +
                float(rgb[matrix_row * groups + affine]) * sum;
            up += float(rus[matrix_row * groups + affine]) *
                    q3_dot8(ruw + matrix_row * row_bytes + quant, x + base) +
                float(rub[matrix_row * groups + affine]) * sum;
        }
        gate = simd_sum(gate); up = simd_sum(up);
        if (lane == 0) {
            const float g = float(bfloat(gate)), u = float(bfloat(up));
            routed[linear] = bfloat(float(bfloat(g / (1.0f + metal::exp(-g)))) * u);
        }
    } else if (linear < 5120) {
        constexpr uint k = 2560, row_bytes = 1280, groups = 80;
        const uint row = linear - 4480;
        float gate = 0.0f, up = 0.0f;
        for (uint base = lane * 8; base < k; base += 256) {
            float sum = 0.0f;
            for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
            const uint quant = base / 2, affine = base / 32;
            gate += float(sgs[row * groups + affine]) *
                    q4_dot8(sgw + row * row_bytes + quant, x + base) +
                float(sgb[row * groups + affine]) * sum;
            up += float(sus[row * groups + affine]) *
                    q4_dot8(suw + row * row_bytes + quant, x + base) +
                float(sub[row * groups + affine]) * sum;
        }
        gate = simd_sum(gate); up = simd_sum(up);
        if (lane == 0) {
            const float g = float(bfloat(gate)), u = float(bfloat(up));
            shared[row] = bfloat(float(bfloat(g / (1.0f + metal::exp(-g)))) * u);
        }
    } else if (linear == 5120) {
        float dot = 0.0f;
        for (uint base = lane * 4; base < 2560; base += 128) {
            float sum = 0.0f;
            for (uint i = 0; i < 4; ++i) sum += float(x[base + i]);
            dot += float(srs[base / 64]) * q8_dot4(srw + base, x + base) +
                float(srb[base / 64]) * sum;
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            const float rounded = float(bfloat(dot));
            router[0] = bfloat(1.0f / (1.0f + metal::exp(-rounded)));
        }
    }
}

kernel void fused_all_down(
    const device bfloat* routed_hidden [[buffer(0)]],
    const device uchar* rw [[buffer(1)]], const device bfloat* rs [[buffer(2)]],
    const device bfloat* rb [[buffer(3)]], const device uint* experts [[buffer(4)]],
    const device bfloat* route_weights [[buffer(5)]],
    const device bfloat* shared_hidden [[buffer(6)]],
    const device uchar* sw [[buffer(7)]], const device bfloat* ss [[buffer(8)]],
    const device bfloat* sb [[buffer(9)]], const device bfloat* router [[buffer(10)]],
    device bfloat* output [[buffer(11)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    bfloat routed_total = bfloat(0.0f);
    for (uint slot = 0; slot < 10; ++slot) {
        constexpr uint k = 448, row_bytes = 168, groups = 7;
        const ulong matrix_row = ulong(experts[slot]) * rows + row;
        const device bfloat* x = routed_hidden + slot * k;
        float dot = 0.0f;
        for (uint base = lane * 8; base < k; base += 256) {
            float sum = 0.0f;
            for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
            dot += float(rs[matrix_row * groups + base / 64]) *
                    q3_dot8(rw + matrix_row * row_bytes + base / 8 * 3, x + base) +
                float(rb[matrix_row * groups + base / 64]) * sum;
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            const bfloat weighted = bfloat(float(route_weights[slot]) * float(bfloat(dot)));
            routed_total = bfloat(float(routed_total) + float(weighted));
        }
    }
    float shared_dot = 0.0f;
    for (uint base = lane * 8; base < 640; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(shared_hidden[base + i]);
        shared_dot += float(ss[row * 20 + base / 32]) *
                q4_dot8(sw + row * 320 + base / 2, shared_hidden + base) +
            float(sb[row * 20 + base / 32]) * sum;
    }
    shared_dot = simd_sum(shared_dot);
    if (lane == 0) {
        const bfloat shared = bfloat(float(bfloat(shared_dot)) * float(router[0]));
        output[row] = bfloat(float(routed_total) + float(shared));
    }
}

kernel void q8_router_logits(
    const device bfloat* x [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device bfloat* scale [[buffer(2)]], const device bfloat* bias [[buffer(3)]],
    device float* logits [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= 512) return;
    float dot = 0.0f;
    for (uint base = lane * 4; base < 2560; base += 128) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) sum += float(x[base + i]);
        dot += float(scale[row * 40 + base / 64]) *
                q8_dot4(weight + row * 2560 + base, x + base) +
            float(bias[row * 40 + base / 64]) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) logits[row] = float(bfloat(dot));
}

kernel void select_top10(
    device float* logits [[buffer(0)]], device uint* experts [[buffer(1)]],
    device bfloat* weights [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float selected[10];
    for (uint slot = 0; slot < 10; ++slot) {
        float best = -INFINITY;
        uint best_id = 0;
        for (uint expert = 0; expert < 512; ++expert) {
            const float value = logits[expert];
            if (value > best) { best = value; best_id = expert; }
        }
        experts[slot] = best_id;
        selected[slot] = best;
        logits[best_id] = -INFINITY;
    }
    float denominator = 0.0f;
    for (uint slot = 0; slot < 10; ++slot) denominator += metal::exp(selected[slot] - selected[0]);
    for (uint slot = 0; slot < 10; ++slot)
        weights[slot] = bfloat(metal::exp(selected[slot] - selected[0]) / denominator);
}

kernel void hc_normalize(
    const device bfloat* stream [[buffer(0)]], const device bfloat* norm [[buffer(1)]],
    device bfloat* normalized [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint hc [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    float values[10];
    float square_sum = 0.0f;
    const uint base = hc * 2560;
    for (uint i = 0; i < 10; ++i) {
        const uint index = base + tid + i * 256;
        values[i] = float(stream[index]);
        square_sum += values[i] * values[i];
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0) partial[simd] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float inverse_rms = rsqrt(total / 2560.0f + 1.0e-6f);
    for (uint i = 0; i < 10; ++i) {
        const uint index = base + tid + i * 256;
        const bfloat value = bfloat(values[i] * inverse_rms);
        const bfloat weight = bfloat(float(norm[index]) + 1.0f);
        normalized[index] = bfloat(float(value) * float(weight));
    }
}

kernel void hc_down_injection(
    const device bfloat* x [[buffer(0)]],
    const device uchar* down_weight [[buffer(1)]],
    const device bfloat* down_scale [[buffer(2)]],
    const device bfloat* down_bias [[buffer(3)]],
    const device uchar* injection_weight [[buffer(4)]],
    const device bfloat* injection_scale [[buffer(5)]],
    const device bfloat* injection_bias [[buffer(6)]],
    device bfloat* activation [[buffer(7)]], device bfloat* injection [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    const bool is_down = row < 320;
    const uint local_row = is_down ? row : row - 320;
    const device uchar* weight = is_down ? down_weight : injection_weight;
    const device bfloat* scale = is_down ? down_scale : injection_scale;
    const device bfloat* bias = is_down ? down_bias : injection_bias;
    float dot = 0.0f;
    for (uint base = tid * 8; base < 10240; base += 2048) {
        const device uchar* bytes = weight + local_row * 5120 + base / 2;
        const float s = float(scale[local_row * 320 + base / 32]);
        const float b = float(bias[local_row * 320 + base / 32]);
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
        dot += s * q4_dot8(bytes, x + base) + b * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) partial[simd] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += partial[i];
        const bfloat value = bfloat(total / 4.0f);
        if (is_down) {
            const bfloat gate = bfloat(1.0f / (1.0f + metal::exp(-float(value))));
            activation[row] = value * gate;
        } else {
            injection[local_row] =
                bfloat(2.0f * (1.0f / (1.0f + metal::exp(-float(value)))));
        }
    }
}

kernel void hc_up_mix(
    const device bfloat* normalized [[buffer(0)]], const device bfloat* activation [[buffer(1)]],
    const device uchar* weight [[buffer(2)]], const device bfloat* scale [[buffer(3)]],
    const device bfloat* bias [[buffer(4)]], device bfloat* mixed [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint column = group * 8 + simd;
    if (column >= 2560) return;
    float stream_sum = 0.0f;
    for (uint hc = 0; hc < 4; ++hc) {
        const uint row = hc * 2560 + column;
        float dot = 0.0f;
        for (uint base = lane * 8; base < 320; base += 256) {
            const device uchar* bytes = weight + row * 160 + base / 2;
            const float s = float(scale[row * 10 + base / 32]);
            const float b = float(bias[row * 10 + base / 32]);
            float sum = 0.0f;
            for (uint i = 0; i < 8; ++i) sum += float(activation[base + i]);
            dot += s * q4_dot8(bytes, activation + base) + b * sum;
        }
        dot = simd_sum(dot);
        const bfloat gate = bfloat(1.0f / (1.0f + metal::exp(-float(bfloat(dot)))));
        stream_sum += float(bfloat(float(gate) * float(normalized[row])));
    }
    if (lane == 0) mixed[column] = bfloat(float(bfloat(stream_sum)) / 4.0f);
}

kernel void healing_left(
    const device bfloat* input [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    device bfloat* hidden [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint column = group * 4 + simd;
    if (column >= 64) return;
    float dot = 0.0f;
    for (uint row = lane; row < 2560; row += 32)
        dot += float(input[row]) * float(weight[row * 64 + column]);
    dot = simd_sum(dot);
    if (lane == 0) hidden[column] = bfloat(dot);
}

kernel void healing_right_write(
    const device bfloat* block [[buffer(0)]], const device bfloat* hidden [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], const device bfloat* stream [[buffer(3)]],
    const device bfloat* injection [[buffer(4)]], device bfloat* output [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint column = group * 4 + simd;
    if (column >= 2560) return;
    float dot = 0.0f;
    for (uint row = lane; row < 64; row += 32)
        dot += float(hidden[row]) * float(weight[row * 2560 + column]);
    dot = simd_sum(dot);
    if (lane == 0) {
        const bfloat healed = bfloat(float(block[column]) + float(bfloat(dot)));
        for (uint hc = 0; hc < 4; ++hc) {
            const bfloat update = bfloat(float(healed) * float(injection[hc]));
            output[hc * 2560 + column] =
                bfloat(float(stream[hc * 2560 + column]) + float(update));
        }
    }
}
)metal";

std::uint16_t bf16(const float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(const std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil)
        throw std::runtime_error("Metal function is missing");
    NSError *error = nil;
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
    if (result == nil)
        throw std::runtime_error(error.localizedDescription.UTF8String);
    return result;
}

struct Shard {
    qwen38::SafetensorsFile file;
    id<MTLBuffer> buffer;

    Shard(id<MTLDevice> device, const std::filesystem::path &path) : file(path) {
        const auto mapped = file.mapped_view();
        const std::size_t page = static_cast<std::size_t>(::getpagesize());
        const std::size_t rounded = (mapped.size() + page - 1) / page * page;
        buffer = [device newBufferWithBytesNoCopy:const_cast<std::byte *>(mapped.data())
                                           length:rounded
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
        if (buffer == nil)
            throw std::runtime_error("Metal rejected the mmap-backed shard");
    }

    [[nodiscard]] NSUInteger offset(std::string_view name) const {
        const auto mapped = file.mapped_view();
        const auto tensor = file.tensor(name);
        return static_cast<NSUInteger>(tensor.bytes.data() - mapped.data());
    }
};

struct ProjectionNames {
    std::string weight, scales, biases;
};

ProjectionNames projection(std::string prefix) {
    return {prefix + ".weight", prefix + ".scales", prefix + ".biases"};
}

void bind_projection(id<MTLComputeCommandEncoder> encoder, const Shard &shard,
                     const ProjectionNames &names, NSUInteger first) {
    [encoder setBuffer:shard.buffer offset:shard.offset(names.weight) atIndex:first];
    [encoder setBuffer:shard.buffer offset:shard.offset(names.scales) atIndex:first + 1];
    [encoder setBuffer:shard.buffer offset:shard.offset(names.biases) atIndex:first + 2];
}

void bind_tensor(id<MTLComputeCommandEncoder> encoder, const Shard &shard, const std::string &name,
                 const NSUInteger index) {
    [encoder setBuffer:shard.buffer offset:shard.offset(name) atIndex:index];
}

double run_hc_read(id<MTLCommandQueue> queue, id<MTLComputePipelineState> normalize_state,
                   id<MTLComputePipelineState> down_state, id<MTLComputePipelineState> up_state,
                   id<MTLBuffer> stream, const Shard &shard, const std::string &norm,
                   const ProjectionNames &down, const ProjectionNames &up,
                   const ProjectionNames &injection, id<MTLBuffer> normalized,
                   id<MTLBuffer> activation, id<MTLBuffer> injection_output, id<MTLBuffer> mixed) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> normalize_encoder = [command computeCommandEncoder];
    [normalize_encoder setComputePipelineState:normalize_state];
    [normalize_encoder setBuffer:stream offset:0 atIndex:0];
    bind_tensor(normalize_encoder, shard, norm, 1);
    [normalize_encoder setBuffer:normalized offset:0 atIndex:2];
    [normalize_encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [normalize_encoder endEncoding];
    id<MTLComputeCommandEncoder> down_encoder = [command computeCommandEncoder];
    [down_encoder setComputePipelineState:down_state];
    [down_encoder setBuffer:normalized offset:0 atIndex:0];
    bind_projection(down_encoder, shard, down, 1);
    bind_projection(down_encoder, shard, injection, 4);
    [down_encoder setBuffer:activation offset:0 atIndex:7];
    [down_encoder setBuffer:injection_output offset:0 atIndex:8];
    [down_encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [down_encoder endEncoding];
    id<MTLComputeCommandEncoder> up_encoder = [command computeCommandEncoder];
    [up_encoder setComputePipelineState:up_state];
    [up_encoder setBuffer:normalized offset:0 atIndex:0];
    [up_encoder setBuffer:activation offset:0 atIndex:1];
    bind_projection(up_encoder, shard, up, 2);
    [up_encoder setBuffer:mixed offset:0 atIndex:5];
    [up_encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [up_encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

void encode_gate(id<MTLCommandBuffer> command, id<MTLComputePipelineState> state,
                 id<MTLBuffer> input, const Shard &shard, const ProjectionNames &gate,
                 const ProjectionNames &up, id<MTLBuffer> experts, id<MTLBuffer> hidden) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(encoder, shard, gate, 1);
    bind_projection(encoder, shard, up, 4);
    [encoder setBuffer:experts offset:0 atIndex:7];
    [encoder setBuffer:hidden offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake((top_k * expert_width + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
}

void encode_down(id<MTLCommandBuffer> command, id<MTLComputePipelineState> state,
                 id<MTLBuffer> hidden, const Shard &shard, const ProjectionNames &down,
                 id<MTLBuffer> experts, id<MTLBuffer> weights, id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:hidden offset:0 atIndex:0];
    bind_projection(encoder, shard, down, 1);
    [encoder setBuffer:experts offset:0 atIndex:4];
    [encoder setBuffer:weights offset:0 atIndex:5];
    [encoder setBuffer:output offset:0 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((hidden_size + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
}

void encode_shared_gate(id<MTLCommandBuffer> command, id<MTLComputePipelineState> state,
                        id<MTLBuffer> input, const Shard &shard, const ProjectionNames &gate,
                        const ProjectionNames &up, id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(encoder, shard, gate, 1);
    bind_projection(encoder, shard, up, 4);
    [encoder setBuffer:output offset:0 atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(160, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
}

void encode_shared_router(id<MTLCommandBuffer> command, id<MTLComputePipelineState> state,
                          id<MTLBuffer> input, const Shard &shard, const ProjectionNames &router,
                          id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(encoder, shard, router, 1);
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
}

void encode_shared_down(id<MTLCommandBuffer> command, id<MTLComputePipelineState> state,
                        id<MTLBuffer> hidden, const Shard &shard, const ProjectionNames &down,
                        id<MTLBuffer> router, id<MTLBuffer> routed, id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:hidden offset:0 atIndex:0];
    bind_projection(encoder, shard, down, 1);
    [encoder setBuffer:router offset:0 atIndex:4];
    [encoder setBuffer:routed offset:0 atIndex:5];
    [encoder setBuffer:output offset:0 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
}

double run_full_joined(id<MTLCommandQueue> queue, id<MTLComputePipelineState> routed_gate_state,
                       id<MTLComputePipelineState> routed_down_state,
                       id<MTLComputePipelineState> shared_gate_state,
                       id<MTLComputePipelineState> shared_router_state,
                       id<MTLComputePipelineState> shared_down_state, id<MTLBuffer> input,
                       const Shard &expert_shard, const Shard &router_shard,
                       const ProjectionNames &routed_gate, const ProjectionNames &routed_up,
                       const ProjectionNames &routed_down, const ProjectionNames &shared_gate,
                       const ProjectionNames &shared_up, const ProjectionNames &shared_down,
                       const ProjectionNames &shared_router, id<MTLBuffer> experts,
                       id<MTLBuffer> weights, id<MTLBuffer> routed_hidden,
                       id<MTLBuffer> routed_output, id<MTLBuffer> shared_hidden,
                       id<MTLBuffer> router_output, id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    encode_gate(command, routed_gate_state, input, expert_shard, routed_gate, routed_up, experts,
                routed_hidden);
    encode_shared_gate(command, shared_gate_state, input, expert_shard, shared_gate, shared_up,
                       shared_hidden);
    encode_shared_router(command, shared_router_state, input, router_shard, shared_router,
                         router_output);
    encode_down(command, routed_down_state, routed_hidden, expert_shard, routed_down, experts,
                weights, routed_output);
    encode_shared_down(command, shared_down_state, shared_hidden, expert_shard, shared_down,
                       router_output, routed_output, output);
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_fused_full(id<MTLCommandQueue> queue, id<MTLComputePipelineState> gate_state,
                      id<MTLComputePipelineState> down_state, id<MTLBuffer> input,
                      const Shard &expert_shard, const Shard &router_shard,
                      const ProjectionNames &routed_gate, const ProjectionNames &routed_up,
                      const ProjectionNames &routed_down, const ProjectionNames &shared_gate,
                      const ProjectionNames &shared_up, const ProjectionNames &shared_down,
                      const ProjectionNames &shared_router, id<MTLBuffer> experts,
                      id<MTLBuffer> weights, id<MTLBuffer> routed_hidden,
                      id<MTLBuffer> shared_hidden, id<MTLBuffer> router_output,
                      id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> gate_encoder = [command computeCommandEncoder];
    [gate_encoder setComputePipelineState:gate_state];
    [gate_encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(gate_encoder, expert_shard, routed_gate, 1);
    bind_projection(gate_encoder, expert_shard, routed_up, 4);
    [gate_encoder setBuffer:experts offset:0 atIndex:7];
    [gate_encoder setBuffer:routed_hidden offset:0 atIndex:8];
    bind_projection(gate_encoder, expert_shard, shared_gate, 9);
    bind_projection(gate_encoder, expert_shard, shared_up, 12);
    [gate_encoder setBuffer:shared_hidden offset:0 atIndex:15];
    bind_projection(gate_encoder, router_shard, shared_router, 16);
    [gate_encoder setBuffer:router_output offset:0 atIndex:19];
    [gate_encoder dispatchThreadgroups:MTLSizeMake(1281, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [gate_encoder endEncoding];

    id<MTLComputeCommandEncoder> down_encoder = [command computeCommandEncoder];
    [down_encoder setComputePipelineState:down_state];
    [down_encoder setBuffer:routed_hidden offset:0 atIndex:0];
    bind_projection(down_encoder, expert_shard, routed_down, 1);
    [down_encoder setBuffer:experts offset:0 atIndex:4];
    [down_encoder setBuffer:weights offset:0 atIndex:5];
    [down_encoder setBuffer:shared_hidden offset:0 atIndex:6];
    bind_projection(down_encoder, expert_shard, shared_down, 7);
    [down_encoder setBuffer:router_output offset:0 atIndex:10];
    [down_encoder setBuffer:output offset:0 atIndex:11];
    [down_encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [down_encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_device_routed_full(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> router_state,
    id<MTLComputePipelineState> select_state, id<MTLComputePipelineState> gate_state,
    id<MTLComputePipelineState> down_state, id<MTLBuffer> input, const Shard &expert_shard,
    const Shard &router_shard, const ProjectionNames &device_router,
    const ProjectionNames &routed_gate, const ProjectionNames &routed_up,
    const ProjectionNames &routed_down, const ProjectionNames &shared_gate,
    const ProjectionNames &shared_up, const ProjectionNames &shared_down,
    const ProjectionNames &shared_router, id<MTLBuffer> logits, id<MTLBuffer> experts,
    id<MTLBuffer> weights, id<MTLBuffer> routed_hidden, id<MTLBuffer> shared_hidden,
    id<MTLBuffer> router_output, id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> router_encoder = [command computeCommandEncoder];
    [router_encoder setComputePipelineState:router_state];
    [router_encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(router_encoder, router_shard, device_router, 1);
    [router_encoder setBuffer:logits offset:0 atIndex:4];
    [router_encoder dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [router_encoder endEncoding];
    id<MTLComputeCommandEncoder> select_encoder = [command computeCommandEncoder];
    [select_encoder setComputePipelineState:select_state];
    [select_encoder setBuffer:logits offset:0 atIndex:0];
    [select_encoder setBuffer:experts offset:0 atIndex:1];
    [select_encoder setBuffer:weights offset:0 atIndex:2];
    [select_encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [select_encoder endEncoding];
    id<MTLComputeCommandEncoder> gate_encoder = [command computeCommandEncoder];
    [gate_encoder setComputePipelineState:gate_state];
    [gate_encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(gate_encoder, expert_shard, routed_gate, 1);
    bind_projection(gate_encoder, expert_shard, routed_up, 4);
    [gate_encoder setBuffer:experts offset:0 atIndex:7];
    [gate_encoder setBuffer:routed_hidden offset:0 atIndex:8];
    bind_projection(gate_encoder, expert_shard, shared_gate, 9);
    bind_projection(gate_encoder, expert_shard, shared_up, 12);
    [gate_encoder setBuffer:shared_hidden offset:0 atIndex:15];
    bind_projection(gate_encoder, router_shard, shared_router, 16);
    [gate_encoder setBuffer:router_output offset:0 atIndex:19];
    [gate_encoder dispatchThreadgroups:MTLSizeMake(1281, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [gate_encoder endEncoding];
    id<MTLComputeCommandEncoder> down_encoder = [command computeCommandEncoder];
    [down_encoder setComputePipelineState:down_state];
    [down_encoder setBuffer:routed_hidden offset:0 atIndex:0];
    bind_projection(down_encoder, expert_shard, routed_down, 1);
    [down_encoder setBuffer:experts offset:0 atIndex:4];
    [down_encoder setBuffer:weights offset:0 atIndex:5];
    [down_encoder setBuffer:shared_hidden offset:0 atIndex:6];
    bind_projection(down_encoder, expert_shard, shared_down, 7);
    [down_encoder setBuffer:router_output offset:0 atIndex:10];
    [down_encoder setBuffer:output offset:0 atIndex:11];
    [down_encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [down_encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_hc_device_routed_full(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> normalize_state,
    id<MTLComputePipelineState> hc_down_state, id<MTLComputePipelineState> hc_up_state,
    id<MTLComputePipelineState> router_state, id<MTLComputePipelineState> select_state,
    id<MTLComputePipelineState> gate_state, id<MTLComputePipelineState> moe_down_state,
    id<MTLComputePipelineState> healing_left_state, id<MTLComputePipelineState> healing_right_state,
    id<MTLBuffer> stream, const Shard &expert_shard, const Shard &router_shard,
    const Shard &healing_shard, const std::string &norm, const ProjectionNames &hc_down,
    const ProjectionNames &hc_up, const ProjectionNames &injection,
    const ProjectionNames &device_router, const ProjectionNames &routed_gate,
    const ProjectionNames &routed_up, const ProjectionNames &routed_down,
    const ProjectionNames &shared_gate, const ProjectionNames &shared_up,
    const ProjectionNames &shared_down, const ProjectionNames &shared_router,
    const std::string &healing_left, const std::string &healing_right, id<MTLBuffer> normalized,
    id<MTLBuffer> activation, id<MTLBuffer> injection_output, id<MTLBuffer> mixed,
    id<MTLBuffer> logits, id<MTLBuffer> experts, id<MTLBuffer> weights, id<MTLBuffer> routed_hidden,
    id<MTLBuffer> shared_hidden, id<MTLBuffer> router_output, id<MTLBuffer> block_output,
    id<MTLBuffer> healing_hidden, id<MTLBuffer> output_stream) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:normalize_state];
    [encoder setBuffer:stream offset:0 atIndex:0];
    bind_tensor(encoder, router_shard, norm, 1);
    [encoder setBuffer:normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:hc_down_state];
    [encoder setBuffer:normalized offset:0 atIndex:0];
    bind_projection(encoder, router_shard, hc_down, 1);
    bind_projection(encoder, router_shard, injection, 4);
    [encoder setBuffer:activation offset:0 atIndex:7];
    [encoder setBuffer:injection_output offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:hc_up_state];
    [encoder setBuffer:normalized offset:0 atIndex:0];
    [encoder setBuffer:activation offset:0 atIndex:1];
    bind_projection(encoder, router_shard, hc_up, 2);
    [encoder setBuffer:mixed offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:router_state];
    [encoder setBuffer:mixed offset:0 atIndex:0];
    bind_projection(encoder, router_shard, device_router, 1);
    [encoder setBuffer:logits offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(128, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:select_state];
    [encoder setBuffer:logits offset:0 atIndex:0];
    [encoder setBuffer:experts offset:0 atIndex:1];
    [encoder setBuffer:weights offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:gate_state];
    [encoder setBuffer:mixed offset:0 atIndex:0];
    bind_projection(encoder, expert_shard, routed_gate, 1);
    bind_projection(encoder, expert_shard, routed_up, 4);
    [encoder setBuffer:experts offset:0 atIndex:7];
    [encoder setBuffer:routed_hidden offset:0 atIndex:8];
    bind_projection(encoder, expert_shard, shared_gate, 9);
    bind_projection(encoder, expert_shard, shared_up, 12);
    [encoder setBuffer:shared_hidden offset:0 atIndex:15];
    bind_projection(encoder, router_shard, shared_router, 16);
    [encoder setBuffer:router_output offset:0 atIndex:19];
    [encoder dispatchThreadgroups:MTLSizeMake(1281, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:moe_down_state];
    [encoder setBuffer:routed_hidden offset:0 atIndex:0];
    bind_projection(encoder, expert_shard, routed_down, 1);
    [encoder setBuffer:experts offset:0 atIndex:4];
    [encoder setBuffer:weights offset:0 atIndex:5];
    [encoder setBuffer:shared_hidden offset:0 atIndex:6];
    bind_projection(encoder, expert_shard, shared_down, 7);
    [encoder setBuffer:router_output offset:0 atIndex:10];
    [encoder setBuffer:block_output offset:0 atIndex:11];
    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:healing_left_state];
    [encoder setBuffer:block_output offset:0 atIndex:0];
    bind_tensor(encoder, healing_shard, healing_left, 1);
    [encoder setBuffer:healing_hidden offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:healing_right_state];
    [encoder setBuffer:block_output offset:0 atIndex:0];
    [encoder setBuffer:healing_hidden offset:0 atIndex:1];
    bind_tensor(encoder, healing_shard, healing_right, 2);
    [encoder setBuffer:stream offset:0 atIndex:3];
    [encoder setBuffer:injection_output offset:0 atIndex:4];
    [encoder setBuffer:output_stream offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_joined(id<MTLCommandQueue> queue, id<MTLComputePipelineState> gate_state,
                  id<MTLComputePipelineState> down_state, id<MTLBuffer> input, const Shard &shard,
                  const ProjectionNames &gate, const ProjectionNames &up,
                  const ProjectionNames &down, id<MTLBuffer> experts, id<MTLBuffer> weights,
                  id<MTLBuffer> hidden, id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    encode_gate(command, gate_state, input, shard, gate, up, experts, hidden);
    encode_down(command, down_state, hidden, shard, down, experts, weights, output);
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_split(id<MTLCommandQueue> queue, id<MTLComputePipelineState> gate_state,
                 id<MTLComputePipelineState> down_state, id<MTLBuffer> input, const Shard &shard,
                 const ProjectionNames &gate, const ProjectionNames &up,
                 const ProjectionNames &down, id<MTLBuffer> experts, id<MTLBuffer> weights,
                 id<MTLBuffer> hidden, id<MTLBuffer> output) {
    id<MTLCommandBuffer> first = [queue commandBuffer];
    encode_gate(first, gate_state, input, shard, gate, up, experts, hidden);
    [first commit];
    [first waitUntilCompleted];
    id<MTLCommandBuffer> second = [queue commandBuffer];
    encode_down(second, down_state, hidden, shard, down, experts, weights, output);
    [second commit];
    [second waitUntilCompleted];
    if (first.status != MTLCommandBufferStatusCompleted ||
        second.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("split Metal command failed");
    return (first.GPUEndTime - first.GPUStartTime + second.GPUEndTime - second.GPUStartTime) *
           1000.0;
}

void evict_device_cache(id<MTLCommandQueue> queue, id<MTLBuffer> buffer, const std::uint8_t value) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    [encoder fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:value];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("cache eviction command failed");
}

double median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}

struct OracleResult {
    std::vector<float> output;
    double median_ms{0.0};
};

OracleResult mlx_oracle(const qwen38::ModelManifest &manifest, const ProjectionNames &gate,
                        const ProjectionNames &up, const ProjectionNames &down,
                        const ProjectionNames &shared_gate, const ProjectionNames &shared_up,
                        const ProjectionNames &shared_down, const ProjectionNames &shared_router,
                        std::span<const float> input, std::span<const std::uint32_t> experts,
                        std::span<const float> route_weights) {
    qwen38::MlxTensorStore store(manifest);
    const auto x =
        MlxArray::from_float32(input, std::array<int, 3>{1, 1, hidden_size}).astype(MLX_BFLOAT16);
    const auto load = [&](const ProjectionNames &names, const std::uint32_t expert) {
        const auto index = MlxArray::from_int32(
            std::array<std::int32_t, 1>{static_cast<std::int32_t>(expert)}, std::span<const int>{});
        return std::array<MlxArray, 3>{
            MlxArray::take_axis(store.tensor(names.weight), index, 0),
            MlxArray::take_axis(store.tensor(names.scales), index, 0),
            MlxArray::take_axis(store.tensor(names.biases), index, 0),
        };
    };
    std::vector<std::array<MlxArray, 3>> gates, ups, downs;
    for (const std::uint32_t expert : experts) {
        gates.push_back(load(gate, expert));
        ups.push_back(load(up, expert));
        downs.push_back(load(down, expert));
    }
    const auto load_full = [&](const ProjectionNames &names) {
        return std::array<MlxArray, 3>{store.tensor(names.weight), store.tensor(names.scales),
                                       store.tensor(names.biases)};
    };
    const auto shared_g = load_full(shared_gate);
    const auto shared_u = load_full(shared_up);
    const auto shared_d = load_full(shared_down);
    const auto shared_r = load_full(shared_router);
    const auto evaluate = [&] {
        MlxArray sum;
        for (int slot = 0; slot < top_k; ++slot) {
            auto g = MlxArray::quantized_matmul(x, gates[slot][0], gates[slot][1], gates[slot][2],
                                                group_size, 3)
                         .silu();
            auto u = MlxArray::quantized_matmul(x, ups[slot][0], ups[slot][1], ups[slot][2],
                                                group_size, 3);
            auto h = MlxArray::multiply(g, u);
            auto y = MlxArray::quantized_matmul(h, downs[slot][0], downs[slot][1], downs[slot][2],
                                                group_size, 3);
            const auto weight =
                MlxArray::from_float32(std::span<const float>(&route_weights[slot], 1),
                                       std::span<const int>{})
                    .astype(MLX_BFLOAT16);
            auto weighted = MlxArray::multiply(y, weight);
            sum = slot == 0 ? std::move(weighted) : MlxArray::add(sum, weighted);
        }
        auto sg =
            MlxArray::quantized_matmul(x, shared_g[0], shared_g[1], shared_g[2], 32, 4).silu();
        auto su = MlxArray::quantized_matmul(x, shared_u[0], shared_u[1], shared_u[2], 32, 4);
        auto sh = MlxArray::multiply(sg, su);
        auto sy = MlxArray::quantized_matmul(sh, shared_d[0], shared_d[1], shared_d[2], 32, 4);
        auto sr =
            MlxArray::quantized_matmul(x, shared_r[0], shared_r[1], shared_r[2], 64, 8).sigmoid();
        return MlxArray::add(sum, MlxArray::multiply(sy, sr)).astype(MLX_FLOAT32).to_float32();
    };
    for (int i = 0; i < 3; ++i)
        static_cast<void>(evaluate());
    std::vector<double> samples;
    std::vector<float> output;
    for (int i = 0; i < 15; ++i) {
        const auto start = std::chrono::steady_clock::now();
        output = evaluate();
        samples.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
    }
    return {std::move(output), median(std::move(samples))};
}

struct HcOracleResult {
    std::vector<float> mixed;
    std::vector<float> injection;
    double median_ms{0.0};
};

HcOracleResult mlx_hc_oracle(const qwen38::ModelManifest &manifest, const std::string &prefix,
                             std::span<const float> stream_values) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    qwen38::MlxTensorStore store(manifest);
    qwen38::HyperConnection hc(store, prefix, 2560, 4, 4, 32, 1.0e-6F, true);
    const auto stream =
        MlxArray::from_float32(stream_values, std::array<int, 3>{1, 1, 10240}).astype(MLX_BFLOAT16);
    const auto evaluate = [&] {
        auto read = hc.read(stream);
        std::array<const MlxArray *, 2> outputs{&read.mixed, &read.injection};
        MlxArray::eval_all(outputs);
        return std::pair{read.mixed.astype(MLX_FLOAT32).to_float32(),
                         read.injection.astype(MLX_FLOAT32).to_float32()};
    };
    for (int i = 0; i < 3; ++i)
        static_cast<void>(evaluate());
    std::vector<double> samples;
    std::pair<std::vector<float>, std::vector<float>> values;
    for (int i = 0; i < 15; ++i) {
        const auto started = std::chrono::steady_clock::now();
        values = evaluate();
        samples.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count());
    }
    return {std::move(values.first), std::move(values.second), median(std::move(samples))};
}

OracleResult mlx_mlp_half_oracle(const qwen38::ModelManifest &manifest,
                                 const std::string &layer_prefix, const std::string &hc_prefix,
                                 std::span<const float> stream_values) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    qwen38::MlxTensorStore store(manifest);
    qwen38::HyperConnection hc(store, hc_prefix, 2560, 4, 4, 32, 1.0e-6F, true);
    qwen38::SparseMoe moe(store, layer_prefix, manifest.config().expert_count,
                          manifest.config().experts_per_token, manifest.config().quantization_bits,
                          manifest.config().quantization_group_size,
                          manifest.config().normalize_topk_probability);
    const auto left = store.tensor(layer_prefix + ".T_delta_left");
    const auto right = store.tensor(layer_prefix + ".T_delta_right");
    const auto stream =
        MlxArray::from_float32(stream_values, std::array<int, 3>{1, 1, 10240}).astype(MLX_BFLOAT16);
    const auto evaluate = [&] {
        auto read = hc.read(stream);
        auto block = moe.forward_decode(read.mixed).astype(MLX_BFLOAT16);
        auto correction = MlxArray::matmul(MlxArray::matmul(block, left), right);
        auto healed = MlxArray::add(block, correction);
        auto output = hc.write(stream, healed, read.injection);
        output.eval();
        return output.astype(MLX_FLOAT32).to_float32();
    };
    for (int i = 0; i < 3; ++i)
        static_cast<void>(evaluate());
    std::vector<double> samples;
    std::vector<float> output;
    for (int i = 0; i < 15; ++i) {
        const auto start = std::chrono::steady_clock::now();
        output = evaluate();
        samples.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
    }
    return {std::move(output), median(std::move(samples))};
}

} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 2)
                throw std::runtime_error("usage: qwen38-persistent-metal-moe-probe MODEL");
            const auto manifest = qwen38::ModelManifest::load(argv[1]);
            const std::string layer_prefix = "language_model.model.layers.3.mlp";
            const std::string prefix = layer_prefix + ".switch_mlp";
            const auto gate = projection(prefix + ".gate_proj");
            const auto up = projection(prefix + ".up_proj");
            const auto down = projection(prefix + ".down_proj");
            const auto shared_gate = projection(layer_prefix + ".shared_expert.gate_proj");
            const auto shared_up = projection(layer_prefix + ".shared_expert.up_proj");
            const auto shared_down = projection(layer_prefix + ".shared_expert.down_proj");
            const auto shared_router = projection(layer_prefix + ".shared_expert_gate");
            const auto device_router = projection(layer_prefix + ".gate");
            const std::string hc_prefix = "language_model.model.layers.3.mlp_hyper_connection";
            const std::string hc_norm = hc_prefix + ".hc_norm.weight";
            const auto hc_down = projection(hc_prefix + ".input_mix_weight_down");
            const auto hc_up = projection(hc_prefix + ".input_mix_weight_up");
            const auto hc_injection = projection(hc_prefix + ".block_inject_weight");
            const std::string healing_left = layer_prefix + ".T_delta_left";
            const std::string healing_right = layer_prefix + ".T_delta_right";
            const std::string shard_name = manifest.weight_map().at(gate.weight);
            for (const std::string &name :
                 {gate.scales, gate.biases, up.weight, up.scales, up.biases, down.weight,
                  down.scales, down.biases, shared_gate.weight, shared_gate.scales,
                  shared_gate.biases, shared_up.weight, shared_up.scales, shared_up.biases,
                  shared_down.weight, shared_down.scales, shared_down.biases}) {
                if (manifest.weight_map().at(name) != shard_name)
                    throw std::runtime_error("layer expert tensors do not share one shard");
            }
            const std::string router_shard_name = manifest.weight_map().at(shared_router.weight);
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil)
                throw std::runtime_error("Metal device is unavailable");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_2;
            NSError *error = nil;
            id<MTLLibrary> library =
                [device newLibraryWithSource:[NSString stringWithUTF8String:metal_source.data()]
                                     options:options
                                       error:&error];
            if (library == nil)
                throw std::runtime_error(error.localizedDescription.UTF8String);
            const auto gate_state = pipeline(device, library, @"q3_gate_up");
            const auto down_state = pipeline(device, library, @"q3_down_reduce");
            const auto shared_gate_state = pipeline(device, library, @"q4_shared_gate_up");
            const auto shared_router_state = pipeline(device, library, @"q8_shared_router");
            const auto shared_down_state = pipeline(device, library, @"q4_shared_down_merge");
            const auto fused_gate_state = pipeline(device, library, @"fused_all_gate_up");
            const auto fused_down_state = pipeline(device, library, @"fused_all_down");
            const auto router_state = pipeline(device, library, @"q8_router_logits");
            const auto select_state = pipeline(device, library, @"select_top10");
            const auto hc_normalize_state = pipeline(device, library, @"hc_normalize");
            const auto hc_down_state = pipeline(device, library, @"hc_down_injection");
            const auto hc_up_state = pipeline(device, library, @"hc_up_mix");
            const auto healing_left_state = pipeline(device, library, @"healing_left");
            const auto healing_right_state = pipeline(device, library, @"healing_right_write");
            Shard shard(device, manifest.directory() / shard_name);
            Shard router_shard(device, manifest.directory() / router_shard_name);
            Shard healing_shard(device,
                                manifest.directory() / manifest.weight_map().at(healing_left));

            std::vector<float> input_f32(hidden_size);
            for (int i = 0; i < hidden_size; ++i)
                input_f32[i] = static_cast<float>((i * 17 + 11) % 257 - 128) / 256.0F;
            std::vector<std::uint16_t> input_bf16(hidden_size);
            std::ranges::transform(input_f32, input_bf16.begin(), bf16);
            std::vector<float> stream_f32(10240);
            for (int i = 0; i < 10240; ++i)
                stream_f32[i] = static_cast<float>((i * 29 + 7) % 521 - 260) / 512.0F;
            std::vector<std::uint16_t> stream_bf16(10240);
            std::ranges::transform(stream_f32, stream_bf16.begin(), bf16);
            const std::array<std::uint32_t, top_k> expert_ids{0,   287, 31, 129, 7,
                                                              256, 63,  17, 201, 95};
            std::array<float, top_k> route_weights{};
            for (int i = 0; i < top_k; ++i)
                route_weights[i] = static_cast<float>(i + 1);
            const float sum = std::accumulate(route_weights.begin(), route_weights.end(), 0.0F);
            for (float &value : route_weights)
                value /= sum;
            std::array<std::uint16_t, top_k> route_weights_bf16{};
            std::ranges::transform(route_weights, route_weights_bf16.begin(), bf16);

            id<MTLBuffer> input =
                [device newBufferWithBytes:input_bf16.data()
                                    length:input_bf16.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_stream =
                [device newBufferWithBytes:stream_bf16.data()
                                    length:stream_bf16.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_normalized = [device newBufferWithLength:10240 * sizeof(std::uint16_t)
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_activation = [device newBufferWithLength:320 * sizeof(std::uint16_t)
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_injection_output =
                [device newBufferWithLength:4 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_mixed = [device newBufferWithLength:2560 * sizeof(std::uint16_t)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> experts =
                [device newBufferWithBytes:expert_ids.data()
                                    length:expert_ids.size() * sizeof(std::uint32_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> weights =
                [device newBufferWithBytes:route_weights_bf16.data()
                                    length:route_weights_bf16.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> hidden =
                [device newBufferWithLength:top_k * expert_width * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> output = [device newBufferWithLength:hidden_size * sizeof(std::uint16_t)
                                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> shared_hidden = [device newBufferWithLength:640 * sizeof(std::uint16_t)
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> shared_router_output =
                [device newBufferWithLength:sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> full_output =
                [device newBufferWithLength:hidden_size * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> healing_hidden =
                [device newBufferWithLength:64 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_output_stream =
                [device newBufferWithLength:10240 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> logits = [device newBufferWithLength:512 * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> cache_evict = [device newBufferWithLength:64ULL * 1024 * 1024
                                                            options:MTLResourceStorageModePrivate];
            if (input == nil || experts == nil || weights == nil || hidden == nil ||
                output == nil || shared_hidden == nil || shared_router_output == nil ||
                full_output == nil || logits == nil || cache_evict == nil || hc_stream == nil ||
                hc_normalized == nil || hc_activation == nil || hc_injection_output == nil ||
                hc_mixed == nil || healing_hidden == nil || hc_output_stream == nil)
                throw std::runtime_error("Metal scratch allocation failed");

            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_hc_read(queue, hc_normalize_state, hc_down_state, hc_up_state,
                                              hc_stream, router_shard, hc_norm, hc_down, hc_up,
                                              hc_injection, hc_normalized, hc_activation,
                                              hc_injection_output, hc_mixed));
            std::vector<double> hc_gpu, hc_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 197));
                const auto started = std::chrono::steady_clock::now();
                hc_gpu.push_back(run_hc_read(queue, hc_normalize_state, hc_down_state, hc_up_state,
                                             hc_stream, router_shard, hc_norm, hc_down, hc_up,
                                             hc_injection, hc_normalized, hc_activation,
                                             hc_injection_output, hc_mixed));
                hc_wall.push_back(std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - started)
                                      .count());
            }

            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_joined(queue, gate_state, down_state, input, shard, gate, up,
                                             down, experts, weights, hidden, output));
            std::vector<double> joined, split, joined_wall, split_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i));
                auto started = std::chrono::steady_clock::now();
                joined.push_back(run_joined(queue, gate_state, down_state, input, shard, gate, up,
                                            down, experts, weights, hidden, output));
                joined_wall.push_back(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - started)
                                          .count());
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 31));
                started = std::chrono::steady_clock::now();
                split.push_back(run_split(queue, gate_state, down_state, input, shard, gate, up,
                                          down, experts, weights, hidden, output));
                split_wall.push_back(std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_full_joined(
                    queue, gate_state, down_state, shared_gate_state, shared_router_state,
                    shared_down_state, input, shard, router_shard, gate, up, down, shared_gate,
                    shared_up, shared_down, shared_router, experts, weights, hidden, output,
                    shared_hidden, shared_router_output, full_output));
            std::vector<double> full_gpu, full_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 77));
                const auto started = std::chrono::steady_clock::now();
                full_gpu.push_back(run_full_joined(
                    queue, gate_state, down_state, shared_gate_state, shared_router_state,
                    shared_down_state, input, shard, router_shard, gate, up, down, shared_gate,
                    shared_up, shared_down, shared_router, experts, weights, hidden, output,
                    shared_hidden, shared_router_output, full_output));
                full_wall.push_back(std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - started)
                                        .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_fused_full(
                    queue, fused_gate_state, fused_down_state, input, shard, router_shard, gate, up,
                    down, shared_gate, shared_up, shared_down, shared_router, experts, weights,
                    hidden, shared_hidden, shared_router_output, full_output));
            std::vector<double> fused_gpu, fused_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 117));
                const auto started = std::chrono::steady_clock::now();
                fused_gpu.push_back(run_fused_full(
                    queue, fused_gate_state, fused_down_state, input, shard, router_shard, gate, up,
                    down, shared_gate, shared_up, shared_down, shared_router, experts, weights,
                    hidden, shared_hidden, shared_router_output, full_output));
                fused_wall.push_back(std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_device_routed_full(
                    queue, router_state, select_state, fused_gate_state, fused_down_state, input,
                    shard, router_shard, device_router, gate, up, down, shared_gate, shared_up,
                    shared_down, shared_router, logits, experts, weights, hidden, shared_hidden,
                    shared_router_output, full_output));
            std::vector<double> device_routed_gpu, device_routed_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 157));
                const auto started = std::chrono::steady_clock::now();
                device_routed_gpu.push_back(run_device_routed_full(
                    queue, router_state, select_state, fused_gate_state, fused_down_state, input,
                    shard, router_shard, device_router, gate, up, down, shared_gate, shared_up,
                    shared_down, shared_router, logits, experts, weights, hidden, shared_hidden,
                    shared_router_output, full_output));
                device_routed_wall.push_back(std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_hc_device_routed_full(
                    queue, hc_normalize_state, hc_down_state, hc_up_state, router_state,
                    select_state, fused_gate_state, fused_down_state, healing_left_state,
                    healing_right_state, hc_stream, shard, router_shard, healing_shard, hc_norm,
                    hc_down, hc_up, hc_injection, device_router, gate, up, down, shared_gate,
                    shared_up, shared_down, shared_router, healing_left, healing_right,
                    hc_normalized, hc_activation, hc_injection_output, hc_mixed, logits, experts,
                    weights, hidden, shared_hidden, shared_router_output, full_output,
                    healing_hidden, hc_output_stream));
            std::vector<double> hc_moe_gpu, hc_moe_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 223));
                const auto started = std::chrono::steady_clock::now();
                hc_moe_gpu.push_back(run_hc_device_routed_full(
                    queue, hc_normalize_state, hc_down_state, hc_up_state, router_state,
                    select_state, fused_gate_state, fused_down_state, healing_left_state,
                    healing_right_state, hc_stream, shard, router_shard, healing_shard, hc_norm,
                    hc_down, hc_up, hc_injection, device_router, gate, up, down, shared_gate,
                    shared_up, shared_down, shared_router, healing_left, healing_right,
                    hc_normalized, hc_activation, hc_injection_output, hc_mixed, logits, experts,
                    weights, hidden, shared_hidden, shared_router_output, full_output,
                    healing_hidden, hc_output_stream));
                hc_moe_wall.push_back(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - started)
                                          .count());
            }
            std::array<std::uint32_t, top_k> selected_experts{};
            std::copy_n(static_cast<const std::uint32_t *>(experts.contents), top_k,
                        selected_experts.begin());
            std::array<float, top_k> selected_weights{};
            const auto *selected_bf16 = static_cast<const std::uint16_t *>(weights.contents);
            std::ranges::transform(selected_bf16, selected_bf16 + top_k, selected_weights.begin(),
                                   from_bf16);
            std::vector<float> hc_direct(hidden_size);
            const auto *hc_mixed_bits = static_cast<const std::uint16_t *>(hc_mixed.contents);
            std::ranges::transform(hc_mixed_bits, hc_mixed_bits + hidden_size, hc_direct.begin(),
                                   from_bf16);
            const auto *result = static_cast<const std::uint16_t *>(full_output.contents);
            std::uint64_t hash = 1469598103934665603ULL;
            std::vector<float> direct(hidden_size);
            for (int i = 0; i < hidden_size; ++i) {
                hash ^= result[i];
                hash *= 1099511628211ULL;
                direct[i] = std::bit_cast<float>(static_cast<std::uint32_t>(result[i]) << 16);
            }
            bool router_ids_match = false;
            double router_weight_max_abs = 0.0;
            {
                qwen38::MlxTensorStore routing_store(manifest);
                qwen38::SparseMoe routing_moe(
                    routing_store, layer_prefix, manifest.config().expert_count,
                    manifest.config().experts_per_token, manifest.config().quantization_bits,
                    manifest.config().quantization_group_size,
                    manifest.config().normalize_topk_probability);
                const auto routing_input =
                    MlxArray::from_float32(hc_direct, std::array<int, 3>{1, 1, hidden_size})
                        .astype(MLX_BFLOAT16);
                const auto expected = routing_moe.route_decode(routing_input);
                router_ids_match = std::equal(expected.experts.begin(), expected.experts.end(),
                                              selected_experts.begin());
                for (int slot = 0; slot < top_k; ++slot) {
                    const float rounded = from_bf16(bf16(expected.weights[slot]));
                    router_weight_max_abs =
                        std::max(router_weight_max_abs,
                                 std::abs(static_cast<double>(rounded - selected_weights[slot])));
                }
            }
            const OracleResult oracle =
                mlx_oracle(manifest, gate, up, down, shared_gate, shared_up, shared_down,
                           shared_router, hc_direct, selected_experts, selected_weights);
            double dot = 0.0, aa = 0.0, bb = 0.0, squared_error = 0.0, max_abs = 0.0;
            for (int i = 0; i < hidden_size; ++i) {
                const double a = direct[i], b = oracle.output[i], delta = a - b;
                if (!std::isfinite(a) || !std::isfinite(b))
                    throw std::runtime_error("non-finite MoE output");
                dot += a * b;
                aa += a * a;
                bb += b * b;
                squared_error += delta * delta;
                max_abs = std::max(max_abs, std::abs(delta));
            }
            const HcOracleResult hc_oracle = mlx_hc_oracle(manifest, hc_prefix, stream_f32);
            double hc_dot = 0.0, hc_aa = 0.0, hc_bb = 0.0, hc_squared_error = 0.0;
            double hc_max_abs = 0.0, hc_injection_max_abs = 0.0;
            for (int i = 0; i < hidden_size; ++i) {
                const double a = from_bf16(hc_mixed_bits[i]);
                const double b = hc_oracle.mixed[static_cast<std::size_t>(i)];
                const double delta = a - b;
                hc_dot += a * b;
                hc_aa += a * a;
                hc_bb += b * b;
                hc_squared_error += delta * delta;
                hc_max_abs = std::max(hc_max_abs, std::abs(delta));
            }
            const auto *hc_injection_bits =
                static_cast<const std::uint16_t *>(hc_injection_output.contents);
            for (int i = 0; i < 4; ++i) {
                hc_injection_max_abs = std::max(
                    hc_injection_max_abs, std::abs(static_cast<double>(
                                              from_bf16(hc_injection_bits[i]) -
                                              hc_oracle.injection[static_cast<std::size_t>(i)])));
            }
            const OracleResult mlp_oracle =
                mlx_mlp_half_oracle(manifest, layer_prefix, hc_prefix, stream_f32);
            const auto *mlp_direct_bits =
                static_cast<const std::uint16_t *>(hc_output_stream.contents);
            double mlp_dot = 0.0, mlp_aa = 0.0, mlp_bb = 0.0, mlp_squared_error = 0.0;
            double mlp_max_abs = 0.0;
            for (int i = 0; i < 10240; ++i) {
                const double a = from_bf16(mlp_direct_bits[i]);
                const double b = mlp_oracle.output[static_cast<std::size_t>(i)];
                const double delta = a - b;
                mlp_dot += a * b;
                mlp_aa += a * a;
                mlp_bb += b * b;
                mlp_squared_error += delta * delta;
                mlp_max_abs = std::max(mlp_max_abs, std::abs(delta));
            }
            std::cout << "{\"device\":\"" << device.name.UTF8String
                      << "\",\"layer\":3,\"experts\":10,\"mmap_backed\":true"
                      << ",\"joined_gpu_median_ms\":" << median(joined)
                      << ",\"split_gpu_median_ms\":" << median(split)
                      << ",\"joined_wall_median_ms\":" << median(joined_wall)
                      << ",\"split_wall_median_ms\":" << median(split_wall)
                      << ",\"full_gpu_median_ms\":" << median(full_gpu)
                      << ",\"full_wall_median_ms\":" << median(full_wall)
                      << ",\"fused_gpu_median_ms\":" << median(fused_gpu)
                      << ",\"fused_wall_median_ms\":" << median(fused_wall)
                      << ",\"device_routed_gpu_median_ms\":" << median(device_routed_gpu)
                      << ",\"device_routed_wall_median_ms\":" << median(device_routed_wall)
                      << ",\"hc_moe_gpu_median_ms\":" << median(hc_moe_gpu)
                      << ",\"hc_moe_wall_median_ms\":" << median(hc_moe_wall)
                      << ",\"router_ids_match\":" << (router_ids_match ? "true" : "false")
                      << ",\"router_weight_max_abs\":" << router_weight_max_abs
                      << ",\"hc_gpu_median_ms\":" << median(hc_gpu)
                      << ",\"hc_wall_median_ms\":" << median(hc_wall)
                      << ",\"mlx_hc_oracle_median_ms\":" << hc_oracle.median_ms
                      << ",\"hc_cosine\":" << hc_dot / std::sqrt(hc_aa * hc_bb)
                      << ",\"hc_rmse\":" << std::sqrt(hc_squared_error / hidden_size)
                      << ",\"hc_max_abs\":" << hc_max_abs
                      << ",\"hc_injection_max_abs\":" << hc_injection_max_abs
                      << ",\"mlx_mlp_oracle_median_ms\":" << mlp_oracle.median_ms
                      << ",\"mlp_cosine\":" << mlp_dot / std::sqrt(mlp_aa * mlp_bb)
                      << ",\"mlp_rmse\":" << std::sqrt(mlp_squared_error / 10240.0)
                      << ",\"mlp_max_abs\":" << mlp_max_abs
                      << ",\"mlx_full_oracle_median_ms\":" << oracle.median_ms
                      << ",\"cosine\":" << dot / std::sqrt(aa * bb)
                      << ",\"rmse\":" << std::sqrt(squared_error / hidden_size)
                      << ",\"max_abs\":" << max_abs << ",\"output_hash\":\"" << hash << "\"}\n";
            return 0;
        } catch (const std::exception &exception) {
            std::cerr << "qwen38-persistent-metal-moe-probe: " << exception.what() << '\n';
            return 1;
        }
    }
}
