#pragma once

#include <string_view>

namespace qwen38::persistent_metal {

inline constexpr std::string_view metal_source_prefix = R"metal(
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

inline float q8_dot8(const device uchar* packed, const device bfloat* x) {
    float value = 0.0f;
    for (uint i = 0; i < 8; ++i) value += float(packed[i]) * float(x[i]);
    return value;
}

inline bfloat load_bf16_unaligned(const device uchar* bytes, ulong index) {
    const ulong offset = index * 2;
    const ushort bits = ushort(ushort(bytes[offset]) |
                               ushort(ushort(bytes[offset + 1]) << 8));
    return as_type<bfloat>(bits);
}

inline half load_f16_unaligned(const device uchar* bytes, ulong index) {
    const ulong offset = index * 2;
    const ushort bits = ushort(ushort(bytes[offset]) |
                               ushort(ushort(bytes[offset + 1]) << 8));
    return as_type<half>(bits);
}

inline uint load_u32_unaligned(const device uchar* bytes, ulong index) {
    const ulong offset = index * 4;
    return uint(bytes[offset]) | (uint(bytes[offset + 1]) << 8) |
        (uint(bytes[offset + 2]) << 16) | (uint(bytes[offset + 3]) << 24);
}

inline uint vq14_code(const device uchar* row, uint subvector) {
    const uint within = subvector & 31u;
    const uint bit_offset = within * 14u;
    const uint word = (subvector >> 5) * 14u + (bit_offset >> 5);
    const uint shift = bit_offset & 31u;
    ulong packed = ulong(load_u32_unaligned(row, word) >> shift);
    if (shift + 14u > 32u)
        packed |= ulong(load_u32_unaligned(row, word + 1u)) << (32u - shift);
    return uint(packed & 0x3fffu);
}

inline float vq_d8_group_dot(const device uchar* codes, const device uchar* codebook,
                             const device bfloat* x, uint group) {
    float value = 0.0f;
    for (uint local = 0; local < 8; ++local) {
        const uint code = vq14_code(codes, group * 8u + local);
        const uint xb = group * 64u + local * 8u;
        const uint cb = code * 8u;
        for (uint element = 0; element < 8; ++element)
            value += float(x[xb + element]) *
                float(load_f16_unaligned(codebook, cb + element));
    }
    return value;
}

// A d8/group-64 quant group is exactly eight packed 14-bit codes (112 bits).
// Cache the four covering words once instead of reloading overlapping words
// for each code.  Code extraction and accumulation order are unchanged.
inline float vq_d8_group_dot_cached(const device uchar* codes,
                                    const device uchar* codebook,
                                    const device bfloat* x, uint group) {
    const uint first_sub = group * 8u;
    const uint block_bit = (first_sub & 31u) * 14u;
    const uint first_word = (first_sub >> 5) * 14u + (block_bit >> 5);
    const uint cached_shift = block_bit & 31u;
    uint packed_words[4];
    for (uint word = 0; word < 4u; ++word)
        packed_words[word] = load_u32_unaligned(codes, first_word + word);
    float value = 0.0f;
    for (uint local = 0; local < 8u; ++local) {
        const uint bit = cached_shift + local * 14u;
        const uint word = bit >> 5;
        const uint shift = bit & 31u;
        ulong packed = ulong(packed_words[word] >> shift);
        if (shift + 14u > 32u)
            packed |= ulong(packed_words[word + 1u]) << (32u - shift);
        const uint code = uint(packed & 0x3fffu);
        const uint xb = group * 64u + local * 8u;
        const uint cb = code * 8u;
        for (uint element = 0; element < 8u; ++element)
            value += float(x[xb + element]) *
                float(load_f16_unaligned(codebook, cb + element));
    }
    return value;
}

inline float vq_d8_group_dot_s8(const device uchar* codes, const device char* codebook,
                                const device uchar* codebook_scales,
                                const device bfloat* x, uint group) {
    const uint first_sub = group * 8u;
    const uint block_bit = (first_sub & 31u) * 14u;
    const uint first_word = (first_sub >> 5) * 14u + (block_bit >> 5);
    const uint cached_shift = block_bit & 31u;
    uint packed_words[4];
    for (uint word = 0; word < 4u; ++word)
        packed_words[word] = load_u32_unaligned(codes, first_word + word);
    float value = 0.0f;
    for (uint local = 0; local < 8u; ++local) {
        const uint bit = cached_shift + local * 14u;
        const uint word = bit >> 5;
        const uint shift = bit & 31u;
        ulong packed = ulong(packed_words[word] >> shift);
        if (shift + 14u > 32u)
            packed |= ulong(packed_words[word + 1u]) << (32u - shift);
        const uint code = uint(packed & 0x3fffu);
        const uint xb = group * 64u + local * 8u;
        const uint cb = code * 8u;
        for (uint element = 0; element < 8u; ++element)
            value += float(x[xb + element]) * float(codebook[cb + element]) *
                float(load_f16_unaligned(codebook_scales, element));
    }
    return value;
}

inline float vq_d8_group_dot_u8(
    const device uchar* codes, const device uchar* codebook,
    const device uchar* codebook_scales, const device uchar* codebook_biases,
    const device bfloat* x, uint group) {
    const uint first_sub = group * 8u;
    const uint block_bit = (first_sub & 31u) * 14u;
    const uint first_word = (first_sub >> 5) * 14u + (block_bit >> 5);
    const uint cached_shift = block_bit & 31u;
    uint packed_words[4];
    for (uint word = 0; word < 4u; ++word)
        packed_words[word] = load_u32_unaligned(codes, first_word + word);
    float value = 0.0f;
    for (uint local = 0; local < 8u; ++local) {
        const uint bit = cached_shift + local * 14u;
        const uint word = bit >> 5;
        const uint shift = bit & 31u;
        ulong packed = ulong(packed_words[word] >> shift);
        if (shift + 14u > 32u)
            packed |= ulong(packed_words[word + 1u]) << (32u - shift);
        const uint code = uint(packed & 0x3fffu);
        const uint xb = group * 64u + local * 8u;
        const uint cb = code * 8u;
        for (uint element = 0; element < 8u; ++element) {
            const float weight = float(codebook[cb + element]) *
                float(load_f16_unaligned(codebook_scales, element)) +
                float(load_f16_unaligned(codebook_biases, element));
            value += float(x[xb + element]) * weight;
        }
    }
    return value;
}

inline float vq_d2_group_dot(const device uchar* codes, const device uchar* codebook,
                             const device bfloat* x, uint group) {
    float value = 0.0f;
    for (uint local = 0; local < 32; ++local) {
        const uint code = uint(codes[group * 32u + local]);
        const uint xb = group * 64u + local * 2u;
        value += float(x[xb]) * float(load_f16_unaligned(codebook, code * 2u));
        value += float(x[xb + 1u]) *
            float(load_f16_unaligned(codebook, code * 2u + 1u));
    }
    return value;
}

kernel void q3_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* gate_weight [[buffer(1)]],
    const device uchar* gate_scale [[buffer(2)]],
    const device uchar* gate_bias [[buffer(3)]],
    const device uchar* up_weight [[buffer(4)]],
    const device uchar* up_scale [[buffer(5)]],
    const device uchar* up_bias [[buffer(6)]],
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
    float gate = 0.0f, up = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
        const uint quant = base / 8 * 3, affine = base / 64;
        gate += float(load_bf16_unaligned(gate_scale, matrix_row * groups + affine)) *
                q3_dot8(gw + quant, x + base) +
            float(load_bf16_unaligned(gate_bias, matrix_row * groups + affine)) * sum;
        up += float(load_bf16_unaligned(up_scale, matrix_row * groups + affine)) *
                q3_dot8(uw + quant, x + base) +
            float(load_bf16_unaligned(up_bias, matrix_row * groups + affine)) * sum;
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
    const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]],
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
        const device bfloat* xv = input + slot * k;
        float dot = 0.0f;
        for (uint base = lane * 8; base < k; base += 256) {
            float sum = 0.0f;
            for (uint i = 0; i < 8; ++i) sum += float(xv[base + i]);
            const uint quant = base / 8 * 3, affine = base / 64;
            dot += float(load_bf16_unaligned(scale, matrix_row * groups + affine)) *
                    q3_dot8(wr + quant, xv + base) +
                float(load_bf16_unaligned(bias, matrix_row * groups + affine)) * sum;
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            const bfloat weighted = bfloat(float(route_weights[slot]) * float(bfloat(dot)));
            total = bfloat(float(total) + float(weighted));
        }
    }
    if (lane == 0) output[row] = total;
}

kernel void vq_d8_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* gate_codes [[buffer(1)]],
    const device uchar* gate_codebook [[buffer(2)]],
    const device uchar* gate_scales [[buffer(3)]],
    const device uchar* up_codes [[buffer(4)]],
    const device uchar* up_codebook [[buffer(5)]],
    const device uchar* up_scales [[buffer(6)]],
    const device uint* experts [[buffer(7)]],
    device bfloat* output [[buffer(8)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 640, groups = 40, words = 140;
    const uint linear = group_id * 4u + simd;
    if (linear >= 10u * rows) return;
    const uint slot = linear / rows, row = linear % rows;
    const ulong matrix_row = ulong(experts[slot]) * rows + row;
    const device uchar* gate_row = gate_codes + matrix_row * words * 4u;
    const device uchar* up_row = up_codes + matrix_row * words * 4u;
    float gate = 0.0f, up = 0.0f;
    for (uint quant_group = lane; quant_group < groups; quant_group += 32u) {
        gate += float(load_f16_unaligned(
                    gate_scales, matrix_row * groups + quant_group)) *
            vq_d8_group_dot(gate_row, gate_codebook, x, quant_group);
        up += float(load_f16_unaligned(
                    up_scales, matrix_row * groups + quant_group)) *
            vq_d8_group_dot(up_row, up_codebook, x, quant_group);
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (lane == 0) {
        const float rounded_gate = float(bfloat(gate));
        const float rounded_up = float(bfloat(up));
        output[linear] = bfloat(
            float(bfloat(rounded_gate / (1.0f + metal::exp(-rounded_gate)))) * rounded_up);
    }
}

kernel void vq_d2_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* gate_codes [[buffer(1)]],
    const device uchar* gate_codebook [[buffer(2)]],
    const device uchar* gate_scales [[buffer(3)]],
    const device uchar* up_codes [[buffer(4)]],
    const device uchar* up_codebook [[buffer(5)]],
    const device uchar* up_scales [[buffer(6)]],
    const device uint* experts [[buffer(7)]],
    device bfloat* output [[buffer(8)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 640, groups = 40, codes_per_row = 1280;
    const uint linear = group_id * 4u + simd;
    if (linear >= 10u * rows) return;
    const uint slot = linear / rows, row = linear % rows;
    const ulong matrix_row = ulong(experts[slot]) * rows + row;
    const device uchar* gate_row = gate_codes + matrix_row * codes_per_row;
    const device uchar* up_row = up_codes + matrix_row * codes_per_row;
    float gate = 0.0f, up = 0.0f;
    for (uint quant_group = lane; quant_group < groups; quant_group += 32u) {
        gate += float(load_f16_unaligned(
                    gate_scales, matrix_row * groups + quant_group)) *
            vq_d2_group_dot(gate_row, gate_codebook, x, quant_group);
        up += float(load_f16_unaligned(
                    up_scales, matrix_row * groups + quant_group)) *
            vq_d2_group_dot(up_row, up_codebook, x, quant_group);
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (lane == 0) {
        const float rounded_gate = float(bfloat(gate));
        const float rounded_up = float(bfloat(up));
        output[linear] = bfloat(
            float(bfloat(rounded_gate / (1.0f + metal::exp(-rounded_gate)))) * rounded_up);
    }
}

kernel void vq_d8_down_reduce(
    const device bfloat* input [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device uchar* codebook [[buffer(2)]],
    const device uchar* scales [[buffer(3)]],
    const device uint* experts [[buffer(4)]],
    const device bfloat* route_weights [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 640, groups = 10, words = 42;
    const uint row = group_id * 4u + simd;
    if (row >= rows) return;
    bfloat total = bfloat(0.0f);
    for (uint slot = 0; slot < 10; ++slot) {
        const ulong matrix_row = ulong(experts[slot]) * rows + row;
        const device uchar* code_row = codes + matrix_row * words * 4u;
        const device bfloat* x = input + slot * k;
        float dot = 0.0f;
        if (lane < groups) {
            dot = float(load_f16_unaligned(scales, matrix_row * groups + lane)) *
                vq_d8_group_dot(code_row, codebook, x, lane);
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            const bfloat weighted = bfloat(float(route_weights[slot]) * float(bfloat(dot)));
            total = bfloat(float(total) + float(weighted));
        }
    }
    if (lane == 0) output[row] = total;
}

kernel void vq_d2_down_reduce(
    const device bfloat* input [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device uchar* codebook [[buffer(2)]],
    const device uchar* scales [[buffer(3)]],
    const device uint* experts [[buffer(4)]],
    const device bfloat* route_weights [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 640, groups = 10, codes_per_row = 320;
    const uint row = group_id * 4u + simd;
    if (row >= rows) return;
    bfloat total = bfloat(0.0f);
    for (uint slot = 0; slot < 10; ++slot) {
        const ulong matrix_row = ulong(experts[slot]) * rows + row;
        const device uchar* code_row = codes + matrix_row * codes_per_row;
        const device bfloat* x = input + slot * k;
        float dot = 0.0f;
        if (lane < groups) {
            dot = float(load_f16_unaligned(scales, matrix_row * groups + lane)) *
                vq_d2_group_dot(code_row, codebook, x, lane);
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            const bfloat weighted = bfloat(float(route_weights[slot]) * float(bfloat(dot)));
            total = bfloat(float(total) + float(weighted));
        }
    }
    if (lane == 0) output[row] = total;
}

kernel void vq_d8_all_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* rgc [[buffer(1)]], const device char* rgcb [[buffer(2)]],
    const device uchar* rgcbs [[buffer(3)]], const device uchar* rgs [[buffer(4)]],
    const device uchar* ruc [[buffer(5)]], const device uchar* rucb [[buffer(6)]],
    const device uchar* rucbs [[buffer(7)]], const device uchar* rucbb [[buffer(8)]],
    const device uchar* rus [[buffer(9)]], const device uint* experts [[buffer(10)]],
    device bfloat* routed [[buffer(11)]], const device uchar* sgw [[buffer(12)]],
    const device uchar* sgs [[buffer(13)]], const device uchar* sgb [[buffer(14)]],
    const device uchar* suw [[buffer(15)]], const device uchar* sus [[buffer(16)]],
    const device uchar* sub [[buffer(17)]], device bfloat* shared [[buffer(18)]],
    const device uchar* srw [[buffer(19)]], const device uchar* srs [[buffer(20)]],
    const device uchar* srb [[buffer(21)]], device bfloat* router [[buffer(22)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint routed_rows = 640, hidden = 2560, hidden_groups = 40;
    const uint linear = group_id * 4u + simd;
    if (linear < 10u * routed_rows) {
        constexpr uint words = 140;
        const uint slot = linear / routed_rows, row = linear % routed_rows;
        const ulong matrix_row = ulong(experts[slot]) * routed_rows + row;
        const device uchar* gate_row = rgc + matrix_row * words * 4u;
        const device uchar* up_row = ruc + matrix_row * words * 4u;
        float gate = 0.0f, up = 0.0f;
        for (uint quant_group = lane; quant_group < hidden_groups; quant_group += 32u) {
            gate += float(load_f16_unaligned(rgs, matrix_row * hidden_groups + quant_group)) *
                vq_d8_group_dot_s8(gate_row, rgcb, rgcbs, x, quant_group);
            up += float(load_f16_unaligned(rus, matrix_row * hidden_groups + quant_group)) *
                vq_d8_group_dot_u8(up_row, rucb, rucbs, rucbb, x, quant_group);
        }
        gate = simd_sum(gate); up = simd_sum(up);
        if (lane == 0) {
            const float g = float(bfloat(gate)), u = float(bfloat(up));
            routed[linear] = bfloat(float(bfloat(g / (1.0f + metal::exp(-g)))) * u);
        }
    } else if (linear < 10u * routed_rows + routed_rows) {
        const uint row = linear - 10u * routed_rows;
        float gate = 0.0f, up = 0.0f;
        for (uint base = lane * 8u; base < hidden; base += 256u) {
            float sum = 0.0f;
            for (uint i = 0; i < 8u; ++i) sum += float(x[base + i]);
            const uint affine = base / 64u;
            gate += float(load_bf16_unaligned(sgs, row * hidden_groups + affine)) *
                    q8_dot8(sgw + row * hidden + base, x + base) +
                float(load_bf16_unaligned(sgb, row * hidden_groups + affine)) * sum;
            up += float(load_bf16_unaligned(sus, row * hidden_groups + affine)) *
                    q8_dot8(suw + row * hidden + base, x + base) +
                float(load_bf16_unaligned(sub, row * hidden_groups + affine)) * sum;
        }
        gate = simd_sum(gate); up = simd_sum(up);
        if (lane == 0) {
            const float g = float(bfloat(gate)), u = float(bfloat(up));
            shared[row] = bfloat(float(bfloat(g / (1.0f + metal::exp(-g)))) * u);
        }
    } else if (linear == 10u * routed_rows + routed_rows) {
        float dot_value = 0.0f;
        for (uint base = lane * 4u; base < hidden; base += 128u) {
            float sum = 0.0f;
            for (uint i = 0; i < 4u; ++i) sum += float(x[base + i]);
            dot_value += float(load_bf16_unaligned(srs, base / 64u)) *
                    q8_dot4(srw + base, x + base) +
                float(load_bf16_unaligned(srb, base / 64u)) * sum;
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0) router[0] = bfloat(1.0f / (1.0f + metal::exp(-float(bfloat(dot_value)))));
    }
}

kernel void vq_d2_all_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* rgc [[buffer(1)]], const device uchar* rgcb [[buffer(2)]],
    const device uchar* rgs [[buffer(3)]], const device uchar* ruc [[buffer(4)]],
    const device uchar* rucb [[buffer(5)]], const device uchar* rus [[buffer(6)]],
    const device uint* experts [[buffer(7)]], device bfloat* routed [[buffer(8)]],
    const device uchar* sgw [[buffer(9)]], const device uchar* sgs [[buffer(10)]],
    const device uchar* sgb [[buffer(11)]], const device uchar* suw [[buffer(12)]],
    const device uchar* sus [[buffer(13)]], const device uchar* sub [[buffer(14)]],
    device bfloat* shared [[buffer(15)]], const device uchar* srw [[buffer(16)]],
    const device uchar* srs [[buffer(17)]], const device uchar* srb [[buffer(18)]],
    device bfloat* router [[buffer(19)]], uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint routed_rows = 640, hidden = 2560, hidden_groups = 40;
    const uint linear = group_id * 4u + simd;
    if (linear < 10u * routed_rows) {
        constexpr uint codes_per_row = 1280;
        const uint slot = linear / routed_rows, row = linear % routed_rows;
        const ulong matrix_row = ulong(experts[slot]) * routed_rows + row;
        const device uchar* gate_row = rgc + matrix_row * codes_per_row;
        const device uchar* up_row = ruc + matrix_row * codes_per_row;
        float gate = 0.0f, up = 0.0f;
        for (uint quant_group = lane; quant_group < hidden_groups; quant_group += 32u) {
            gate += float(load_f16_unaligned(rgs, matrix_row * hidden_groups + quant_group)) *
                vq_d2_group_dot(gate_row, rgcb, x, quant_group);
            up += float(load_f16_unaligned(rus, matrix_row * hidden_groups + quant_group)) *
                vq_d2_group_dot(up_row, rucb, x, quant_group);
        }
        gate = simd_sum(gate); up = simd_sum(up);
        if (lane == 0) {
            const float g = float(bfloat(gate)), u = float(bfloat(up));
            routed[linear] = bfloat(float(bfloat(g / (1.0f + metal::exp(-g)))) * u);
        }
    } else if (linear < 10u * routed_rows + routed_rows) {
        const uint row = linear - 10u * routed_rows;
        float gate = 0.0f, up = 0.0f;
        for (uint base = lane * 8u; base < hidden; base += 256u) {
            float sum = 0.0f;
            for (uint i = 0; i < 8u; ++i) sum += float(x[base + i]);
            const uint affine = base / 64u;
            gate += float(load_bf16_unaligned(sgs, row * hidden_groups + affine)) *
                    q8_dot8(sgw + row * hidden + base, x + base) +
                float(load_bf16_unaligned(sgb, row * hidden_groups + affine)) * sum;
            up += float(load_bf16_unaligned(sus, row * hidden_groups + affine)) *
                    q8_dot8(suw + row * hidden + base, x + base) +
                float(load_bf16_unaligned(sub, row * hidden_groups + affine)) * sum;
        }
        gate = simd_sum(gate); up = simd_sum(up);
        if (lane == 0) {
            const float g = float(bfloat(gate)), u = float(bfloat(up));
            shared[row] = bfloat(float(bfloat(g / (1.0f + metal::exp(-g)))) * u);
        }
    } else if (linear == 10u * routed_rows + routed_rows) {
        float dot_value = 0.0f;
        for (uint base = lane * 4u; base < hidden; base += 128u) {
            float sum = 0.0f;
            for (uint i = 0; i < 4u; ++i) sum += float(x[base + i]);
            dot_value += float(load_bf16_unaligned(srs, base / 64u)) *
                    q8_dot4(srw + base, x + base) +
                float(load_bf16_unaligned(srb, base / 64u)) * sum;
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0) router[0] = bfloat(1.0f / (1.0f + metal::exp(-float(bfloat(dot_value)))));
    }
}

kernel void vq_d8_all_down(
    const device bfloat* routed_hidden [[buffer(0)]],
    const device uchar* codes [[buffer(1)]], const device uchar* codebook [[buffer(2)]],
    const device uchar* codebook_scales [[buffer(3)]],
    const device uchar* codebook_biases [[buffer(4)]],
    const device uchar* scales [[buffer(5)]], const device uint* experts [[buffer(6)]],
    const device bfloat* route_weights [[buffer(7)]],
    const device bfloat* shared_hidden [[buffer(8)]],
    const device uchar* sw [[buffer(9)]], const device uchar* ss [[buffer(10)]],
    const device uchar* sb [[buffer(11)]], const device bfloat* router [[buffer(12)]],
    device bfloat* output [[buffer(13)]], uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 640, groups = 10, words = 42;
    const uint row = group_id * 4u + simd;
    if (row >= rows) return;
    bfloat routed_total = bfloat(0.0f);
    constexpr uint slots_per_wave = 3u;
    for (uint slot_base = 0; slot_base < 10u; slot_base += slots_per_wave) {
        const uint local_slot = lane / groups;
        const uint quant_group = lane - local_slot * groups;
        const uint slot = slot_base + local_slot;
        float scaled_group = 0.0f;
        if (local_slot < slots_per_wave && slot < 10u) {
            const ulong matrix_row = ulong(experts[slot]) * rows + row;
            const device uchar* code_row = codes + matrix_row * words * 4u;
            const device bfloat* x = routed_hidden + slot * k;
            scaled_group =
                float(load_f16_unaligned(scales, matrix_row * groups + quant_group)) *
                vq_d8_group_dot_u8(
                    code_row, codebook, codebook_scales, codebook_biases, x, quant_group);
        }
        for (uint wave_slot = 0; wave_slot < slots_per_wave; ++wave_slot) {
            const float ordered = lane < groups
                ? simd_shuffle(scaled_group, ushort(wave_slot * groups + lane))
                : 0.0f;
            const float value = simd_sum(ordered);
            const uint reduced_slot = slot_base + wave_slot;
            if (lane == 0 && reduced_slot < 10u) {
                routed_total = bfloat(float(routed_total) +
                    float(bfloat(float(route_weights[reduced_slot]) *
                        float(bfloat(value)))));
            }
        }
    }
    float shared_dot = 0.0f;
    for (uint base = lane * 8u; base < k; base += 256u) {
        float sum = 0.0f;
        for (uint i = 0; i < 8u; ++i) sum += float(shared_hidden[base + i]);
        shared_dot += float(load_bf16_unaligned(ss, row * groups + base / 64u)) *
                q8_dot8(sw + row * k + base, shared_hidden + base) +
            float(load_bf16_unaligned(sb, row * groups + base / 64u)) * sum;
    }
    shared_dot = simd_sum(shared_dot);
    if (lane == 0) output[row] = bfloat(float(routed_total) +
        float(bfloat(float(bfloat(shared_dot)) * float(router[0]))));
}

kernel void vq_d2_all_down(
    const device bfloat* routed_hidden [[buffer(0)]],
    const device uchar* codes [[buffer(1)]], const device uchar* codebook [[buffer(2)]],
    const device uchar* scales [[buffer(3)]], const device uint* experts [[buffer(4)]],
    const device bfloat* route_weights [[buffer(5)]],
    const device bfloat* shared_hidden [[buffer(6)]],
    const device uchar* sw [[buffer(7)]], const device uchar* ss [[buffer(8)]],
    const device uchar* sb [[buffer(9)]], const device bfloat* router [[buffer(10)]],
    device bfloat* output [[buffer(11)]], uint group_id [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 640, groups = 10, codes_per_row = 320;
    const uint row = group_id * 4u + simd;
    if (row >= rows) return;
    bfloat routed_total = bfloat(0.0f);
    for (uint slot = 0; slot < 10u; ++slot) {
        const ulong matrix_row = ulong(experts[slot]) * rows + row;
        const device uchar* code_row = codes + matrix_row * codes_per_row;
        const device bfloat* x = routed_hidden + slot * k;
        float value = 0.0f;
        if (lane < groups) {
            value = float(load_f16_unaligned(scales, matrix_row * groups + lane)) *
                vq_d2_group_dot(code_row, codebook, x, lane);
        }
        value = simd_sum(value);
        if (lane == 0) routed_total = bfloat(float(routed_total) +
            float(bfloat(float(route_weights[slot]) * float(bfloat(value)))));
    }
    float shared_dot = 0.0f;
    for (uint base = lane * 8u; base < k; base += 256u) {
        float sum = 0.0f;
        for (uint i = 0; i < 8u; ++i) sum += float(shared_hidden[base + i]);
        shared_dot += float(load_bf16_unaligned(ss, row * groups + base / 64u)) *
                q8_dot8(sw + row * k + base, shared_hidden + base) +
            float(load_bf16_unaligned(sb, row * groups + base / 64u)) * sum;
    }
    shared_dot = simd_sum(shared_dot);
    if (lane == 0) output[row] = bfloat(float(routed_total) +
        float(bfloat(float(bfloat(shared_dot)) * float(router[0]))));
}

kernel void q4_shared_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* gate_weight [[buffer(1)]],
    const device uchar* gate_scale [[buffer(2)]],
    const device uchar* gate_bias [[buffer(3)]],
    const device uchar* up_weight [[buffer(4)]],
    const device uchar* up_scale [[buffer(5)]],
    const device uchar* up_bias [[buffer(6)]],
    device bfloat* output [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 640, k = 2560, row_bytes = k / 2, groups = k / 32;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    const device uchar* gw = gate_weight + row * row_bytes;
    const device uchar* uw = up_weight + row * row_bytes;
    float gate = 0.0f, up = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
        const uint quant = base / 2, affine = base / 32;
        gate += float(load_bf16_unaligned(gate_scale, row * groups + affine)) *
                q4_dot8(gw + quant, x + base) +
            float(load_bf16_unaligned(gate_bias, row * groups + affine)) * sum;
        up += float(load_bf16_unaligned(up_scale, row * groups + affine)) *
                q4_dot8(uw + quant, x + base) +
            float(load_bf16_unaligned(up_bias, row * groups + affine)) * sum;
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

kernel void q8_shared_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* gate_weight [[buffer(1)]],
    const device uchar* gate_scale [[buffer(2)]],
    const device uchar* gate_bias [[buffer(3)]],
    const device uchar* up_weight [[buffer(4)]],
    const device uchar* up_scale [[buffer(5)]],
    const device uchar* up_bias [[buffer(6)]],
    device bfloat* output [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 640, k = 2560, groups = k / 64;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    const device uchar* gw = gate_weight + row * k;
    const device uchar* uw = up_weight + row * k;
    float gate = 0.0f, up = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
        const uint affine = base / 64;
        gate += float(load_bf16_unaligned(gate_scale, row * groups + affine)) *
                q8_dot8(gw + base, x + base) +
            float(load_bf16_unaligned(gate_bias, row * groups + affine)) * sum;
        up += float(load_bf16_unaligned(up_scale, row * groups + affine)) *
                q8_dot8(uw + base, x + base) +
            float(load_bf16_unaligned(up_bias, row * groups + affine)) * sum;
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
    const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 2560, groups = k / 64;
    float dot = 0.0f;
    for (uint base = lane * 4; base < k; base += 128) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) sum += float(x[base + i]);
        dot += float(load_bf16_unaligned(scale, base / 64)) *
                q8_dot4(weight + base, x + base) +
            float(load_bf16_unaligned(bias, base / 64)) * sum;
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
    const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]],
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
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(input[base + i]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 32)) *
                q4_dot8(wr + base / 2, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 32)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) {
        const bfloat shared = bfloat(float(bfloat(dot)) * float(router[0]));
        output[row] = bfloat(float(routed[row]) + float(shared));
    }
}

kernel void q8_shared_down_merge(
    const device bfloat* input [[buffer(0)]],
    const device uchar* weight [[buffer(1)]],
    const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]],
    const device bfloat* router [[buffer(4)]],
    const device bfloat* routed [[buffer(5)]],
    device bfloat* output [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 640, groups = k / 64;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    const device uchar* wr = weight + row * k;
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(input[base + i]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                q8_dot8(wr + base, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) {
        const bfloat shared = bfloat(float(bfloat(dot)) * float(router[0]));
        output[row] = bfloat(float(routed[row]) + float(shared));
    }
}

kernel void fused_all_gate_up(
    const device bfloat* x [[buffer(0)]],
    const device uchar* rgw [[buffer(1)]], const device uchar* rgs [[buffer(2)]],
    const device uchar* rgb [[buffer(3)]], const device uchar* ruw [[buffer(4)]],
    const device uchar* rus [[buffer(5)]], const device uchar* rub [[buffer(6)]],
    const device uint* experts [[buffer(7)]], device bfloat* routed [[buffer(8)]],
    const device uchar* sgw [[buffer(9)]], const device uchar* sgs [[buffer(10)]],
    const device uchar* sgb [[buffer(11)]], const device uchar* suw [[buffer(12)]],
    const device uchar* sus [[buffer(13)]], const device uchar* sub [[buffer(14)]],
    device bfloat* shared [[buffer(15)]], const device uchar* srw [[buffer(16)]],
    const device uchar* srs [[buffer(17)]], const device uchar* srb [[buffer(18)]],
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
            gate += float(load_bf16_unaligned(rgs, matrix_row * groups + affine)) *
                    q3_dot8(rgw + matrix_row * row_bytes + quant, x + base) +
                float(load_bf16_unaligned(rgb, matrix_row * groups + affine)) * sum;
            up += float(load_bf16_unaligned(rus, matrix_row * groups + affine)) *
                    q3_dot8(ruw + matrix_row * row_bytes + quant, x + base) +
                float(load_bf16_unaligned(rub, matrix_row * groups + affine)) * sum;
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
            gate += float(load_bf16_unaligned(sgs, row * groups + affine)) *
                    q4_dot8(sgw + row * row_bytes + quant, x + base) +
                float(load_bf16_unaligned(sgb, row * groups + affine)) * sum;
            up += float(load_bf16_unaligned(sus, row * groups + affine)) *
                    q4_dot8(suw + row * row_bytes + quant, x + base) +
                float(load_bf16_unaligned(sub, row * groups + affine)) * sum;
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
            dot += float(load_bf16_unaligned(srs, base / 64)) *
                    q8_dot4(srw + base, x + base) +
                float(load_bf16_unaligned(srb, base / 64)) * sum;
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
    const device uchar* rw [[buffer(1)]], const device uchar* rs [[buffer(2)]],
    const device uchar* rb [[buffer(3)]], const device uint* experts [[buffer(4)]],
    const device bfloat* route_weights [[buffer(5)]],
    const device bfloat* shared_hidden [[buffer(6)]],
    const device uchar* sw [[buffer(7)]], const device uchar* ss [[buffer(8)]],
    const device uchar* sb [[buffer(9)]], const device bfloat* router [[buffer(10)]],
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
            dot += float(load_bf16_unaligned(rs, matrix_row * groups + base / 64)) *
                    q3_dot8(rw + matrix_row * row_bytes + base / 8 * 3, x + base) +
                float(load_bf16_unaligned(rb, matrix_row * groups + base / 64)) * sum;
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
        shared_dot += float(load_bf16_unaligned(ss, row * 20 + base / 32)) *
                q4_dot8(sw + row * 320 + base / 2, shared_hidden + base) +
            float(load_bf16_unaligned(sb, row * 20 + base / 32)) * sum;
    }
    shared_dot = simd_sum(shared_dot);
    if (lane == 0) {
        const bfloat shared = bfloat(float(bfloat(shared_dot)) * float(router[0]));
        output[row] = bfloat(float(routed_total) + float(shared));
    }
}

kernel void q8_router_logits(
    const device bfloat* x [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device uchar* scale [[buffer(2)]], const device uchar* bias [[buffer(3)]],
    device float* logits [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= 512) return;
    float dot = 0.0f;
    for (uint base = lane * 4; base < 2560; base += 128) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) sum += float(x[base + i]);
        dot += float(load_bf16_unaligned(scale, row * 40 + base / 64)) *
                q8_dot4(weight + row * 2560 + base, x + base) +
            float(load_bf16_unaligned(bias, row * 40 + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) logits[row] = float(bfloat(dot));
}

kernel void bf16_router_logits(
    const device bfloat* x [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    device float* logits [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4u + simd;
    if (row >= 512u) return;
    float dot = 0.0f;
    for (uint column = lane; column < 2560u; column += 32u)
        dot += float(load_bf16_unaligned(weight, ulong(row) * 2560u + column)) *
            float(x[column]);
    dot = simd_sum(dot);
    if (lane == 0) logits[row] = dot;
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

kernel void qsa_score_blocks(
    const device bfloat* query [[buffer(0)]], const device bfloat* pooled [[buffer(1)]],
    device float* scores [[buffer(2)]], constant uint& block_count [[buffer(3)]],
    constant uint& padded_count [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint block = group * 4 + simd;
    if (block >= padded_count) return;
    if (block >= block_count) {
        if (lane == 0) scores[block] = -INFINITY;
        return;
    }
    float score = 0.0f;
    for (uint head = 0; head < 4; ++head) {
        float dot = 0.0f;
        for (uint component = lane; component < 128; component += 32)
            dot += float(query[head * 128 + component]) *
                float(pooled[block * 128 + component]);
        dot = simd_sum(dot);
        if (lane == 0) score += max(dot, 0.0f);
    }
    if (lane == 0) scores[block] = score - float(block) * 1.0e-7f;
}

kernel void qsa_append_decode_state(
    const device bfloat* projected [[buffer(0)]],
    const device bfloat* normalized_key [[buffer(1)]],
    device bfloat* hot_key [[buffer(2)]], device bfloat* hot_value [[buffer(3)]],
    device bfloat* pending_raw [[buffer(4)]], device bfloat* pooled [[buffer(5)]],
    const device uchar* index_key_norm [[buffer(6)]],
    const device bfloat* pool_rope_cos [[buffer(7)]],
    const device bfloat* pool_rope_sin [[buffer(8)]],
    constant uint& hot_index [[buffer(9)]], constant uint& hot_capacity [[buffer(10)]],
    constant uint& pending_index [[buffer(11)]], constant uint& pooled_index [[buffer(12)]],
    constant uint& complete_block [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]]) {
    for (uint head = 0; head < 2; ++head) {
        const uint source = head * 256 + lane * 8;
        const uint target = (head * hot_capacity + hot_index) * 256 + lane * 8;
        for (uint item = 0; item < 8; ++item) {
            hot_key[target + item] = normalized_key[source + item];
            hot_value[target + item] = projected[13440 + source + item];
        }
    }
    for (uint item = 0; item < 4; ++item)
        pending_raw[pending_index * 128 + lane + item * 32] =
            projected[512 + lane + item * 32];
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (complete_block == 0) return;

    float values[4];
    float square_sum = 0.0f;
    for (uint item = 0; item < 4; ++item) {
        const uint component = lane + item * 32;
        float mean = 0.0f;
        for (uint row = 0; row < 4; ++row)
            mean += float(pending_raw[row * 128 + component]);
        values[item] = float(bfloat(mean * 0.25f));
        square_sum += values[item] * values[item];
    }
    square_sum = simd_sum(square_sum);
    const float inverse_rms = rsqrt(square_sum / 128.0f + 1.0e-6f);
    bfloat normalized[4];
    for (uint item = 0; item < 4; ++item) {
        const uint component = lane + item * 32;
        normalized[item] = bfloat(values[item] * inverse_rms *
                                  (float(load_bf16_unaligned(index_key_norm, component)) + 1.0f));
    }
    const bfloat first = normalized[0], second = normalized[1];
    normalized[0] = bfloat(float(first) * float(pool_rope_cos[lane]) -
                           float(second) * float(pool_rope_sin[lane]));
    normalized[1] = bfloat(float(second) * float(pool_rope_cos[lane + 32]) +
                           float(first) * float(pool_rope_sin[lane + 32]));
    for (uint item = 0; item < 4; ++item)
        pooled[pooled_index * 128 + lane + item * 32] = normalized[item];
}

inline void qsa_bitonic256(threadgroup float* values, threadgroup uint* ids, uint tid) {
    for (uint width = 2; width <= 256; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            const uint left = (tid / stride) * (stride * 2) + tid % stride;
            const uint right = left + stride;
            const bool ascending = (left & width) == 0;
            const bool swap = ascending ? values[left] > values[right]
                                        : values[left] < values[right];
            if (swap) {
                const float score = values[left];
                const uint id = ids[left];
                values[left] = values[right]; ids[left] = ids[right];
                values[right] = score; ids[right] = id;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void qsa_top128_first(
    const device float* scores [[buffer(0)]], device float* output_scores [[buffer(1)]],
    device uint* output_ids [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint ids[256];
    const uint source = group * 256;
    values[tid] = scores[source + tid]; ids[tid] = source + tid;
    values[tid + 128] = scores[source + tid + 128]; ids[tid + 128] = source + tid + 128;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qsa_bitonic256(values, ids, tid);
    output_scores[group * 128 + tid] = values[128 + tid];
    output_ids[group * 128 + tid] = ids[128 + tid];
}

kernel void qsa_top128_merge(
    const device float* input_scores [[buffer(0)]], const device uint* input_ids [[buffer(1)]],
    device float* output_scores [[buffer(2)]], device uint* output_ids [[buffer(3)]],
    constant uint& input_groups [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (group * 2 + 1 >= input_groups) {
        output_scores[group * 128 + tid] = input_scores[group * 256 + tid];
        output_ids[group * 128 + tid] = input_ids[group * 256 + tid];
        return;
    }
    threadgroup float values[256];
    threadgroup uint ids[256];
    const uint source = group * 256;
    values[tid] = input_scores[source + tid]; ids[tid] = input_ids[source + tid];
    values[tid + 128] = input_scores[source + tid + 128];
    ids[tid + 128] = input_ids[source + tid + 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qsa_bitonic256(values, ids, tid);
    output_scores[group * 128 + tid] = values[128 + tid];
    output_ids[group * 128 + tid] = ids[128 + tid];
}

kernel void qsa_attention_q8_blocks(
    const device bfloat* query [[buffer(0)]], const device uint* key_weight [[buffer(1)]],
    const device bfloat* key_scale [[buffer(2)]], const device bfloat* key_bias [[buffer(3)]],
    const device uint* value_weight [[buffer(4)]], const device bfloat* value_scale [[buffer(5)]],
    const device bfloat* value_bias [[buffer(6)]], const device uint* selected [[buffer(7)]],
    device bfloat* output [[buffer(8)]], constant uint& cold_count [[buffer(9)]],
    const device bfloat* hot_key [[buffer(10)]], const device bfloat* hot_value [[buffer(11)]],
    constant uint& hot_count [[buffer(12)]], constant uint& hot_capacity [[buffer(13)]],
    constant uint& selected_count [[buffer(14)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint kv_head [[threadgroup_position_in_grid]]) {
    constexpr uint tile = 16, dimension = 256, packed_dimension = 64;
    constexpr uint query_heads_per_kv = 12;
    const bool active = simd < query_heads_per_kv;
    const uint query_head = kv_head * query_heads_per_kv + simd;
    float q[8];
    float accumulator[8];
    for (uint component = 0; component < 8; ++component) {
        q[component] = active ? float(query[query_head * dimension + lane * 8 + component]) : 0.0f;
        accumulator[component] = 0.0f;
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    threadgroup bfloat shared_key[tile * dimension];
    threadgroup bfloat shared_value[tile * dimension];
    for (uint tile_start = 0; tile_start < selected_count; tile_start += tile) {
        const uint tile_count = min(tile, selected_count - tile_start);
        for (uint offset = tid; offset < 2 * tile * packed_dimension; offset += 384) {
            const bool is_value = offset >= tile * packed_dimension;
            const uint local = is_value ? offset - tile * packed_dimension : offset;
            const uint slot = local / packed_dimension;
            const uint packed_channel = local % packed_dimension;
            const uint selected_slot = tile_start + slot;
            if (slot >= tile_count) continue;
            const uint token = selected[selected_slot / 4] * 4 + selected_slot % 4;
            const uint shared_base = slot * dimension + packed_channel * 4;
            if (token < cold_count) {
                const ulong vector = ulong(kv_head) * cold_count + token;
                const ulong word_index = vector * packed_dimension + packed_channel;
                const uint packed = is_value ? value_weight[word_index] : key_weight[word_index];
                const ulong affine_index = vector * 4 + packed_channel / 16;
                const float scale =
                    float(is_value ? value_scale[affine_index] : key_scale[affine_index]);
                const float bias =
                    float(is_value ? value_bias[affine_index] : key_bias[affine_index]);
                for (uint component = 0; component < 4; ++component) {
                    const bfloat value =
                        bfloat(float((packed >> (component * 8)) & 255) * scale + bias);
                    if (is_value) shared_value[shared_base + component] = value;
                    else shared_key[shared_base + component] = value;
                }
            } else {
                const uint hot = token - cold_count;
                const ulong base = (ulong(kv_head) * hot_capacity + hot) * dimension +
                    packed_channel * 4;
                for (uint component = 0; component < 4; ++component) {
                    const bfloat value = is_value ? hot_value[base + component]
                                                  : hot_key[base + component];
                    if (is_value) shared_value[shared_base + component] = value;
                    else shared_key[shared_base + component] = value;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slot = 0; slot < tile_count; ++slot) {
            float partial = 0.0f;
            const uint base = slot * dimension + lane * 8;
            for (uint component = 0; component < 8; ++component)
                partial += q[component] * float(shared_key[base + component]);
            const float score = simd_sum(partial) * 0.0625f;
            if (active) {
                const float next_max = max(running_max, score);
                const float alpha = isfinite(running_max) ? metal::exp(running_max - next_max) : 0.0f;
                const float probability = metal::exp(score - next_max);
                running_sum = running_sum * alpha + probability;
                for (uint component = 0; component < 8; ++component)
                    accumulator[component] = accumulator[component] * alpha +
                        probability * float(shared_value[base + component]);
                running_max = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint total_count = cold_count + hot_count;
    const uint tail_start = (total_count / 4) * 4;
    for (uint token = tail_start; active && token < total_count; ++token) {
        const uint hot = token - cold_count;
        const ulong base = (ulong(kv_head) * hot_capacity + hot) * dimension + lane * 8;
        float partial = 0.0f;
        for (uint component = 0; component < 8; ++component)
            partial += q[component] * float(hot_key[base + component]);
        const float score = simd_sum(partial) * 0.0625f;
        const float next_max = max(running_max, score);
        const float alpha = metal::exp(running_max - next_max);
        const float probability = metal::exp(score - next_max);
        running_sum = running_sum * alpha + probability;
        for (uint component = 0; component < 8; ++component)
            accumulator[component] = accumulator[component] * alpha +
                probability * float(hot_value[base + component]);
        running_max = next_max;
    }
    if (active) {
        const float inverse_sum = 1.0f / running_sum;
        for (uint component = 0; component < 8; ++component)
            output[query_head * dimension + lane * 8 + component] =
                bfloat(accumulator[component] * inverse_sum);
    }
}

)metal";

inline constexpr std::string_view metal_source_suffix = R"metal(
kernel void attention_qkv_index(
    const device bfloat* input [[buffer(0)]],
    const device uchar* iw [[buffer(1)]], const device uchar* is [[buffer(2)]],
    const device uchar* ib [[buffer(3)]], const device uchar* qw [[buffer(4)]],
    const device uchar* qs [[buffer(5)]], const device uchar* qb [[buffer(6)]],
    const device uchar* kw [[buffer(7)]], const device uchar* ks [[buffer(8)]],
    const device uchar* kb [[buffer(9)]], const device uchar* vw [[buffer(10)]],
    const device uchar* vs [[buffer(11)]], const device uchar* vb [[buffer(12)]],
    device bfloat* output [[buffer(13)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint linear = group * 4 + simd;
    if (linear >= 13952) return;
    const device uchar* weight;
    const device uchar* scale;
    const device uchar* bias;
    uint row;
    if (linear < 640) {
        row = linear; weight = iw; scale = is; bias = ib;
    } else if (linear < 12928) {
        row = linear - 640; weight = qw; scale = qs; bias = qb;
    } else if (linear < 13440) {
        row = linear - 12928; weight = kw; scale = ks; bias = kb;
    } else {
        row = linear - 13440; weight = vw; scale = vs; bias = vb;
    }
    float dot = 0.0f;
    for (uint base = lane * 8; base < 2560; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * 80 + base / 32)) *
                q4_dot8(weight + row * 1280 + base / 2, input + base) +
            float(load_bf16_unaligned(bias, row * 80 + base / 32)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[linear] = bfloat(dot);
}

kernel void attention_qkv_index_q8(
    const device bfloat* input [[buffer(0)]],
    const device uchar* iw [[buffer(1)]], const device uchar* is [[buffer(2)]],
    const device uchar* ib [[buffer(3)]], const device uchar* qw [[buffer(4)]],
    const device uchar* qs [[buffer(5)]], const device uchar* qb [[buffer(6)]],
    const device uchar* kw [[buffer(7)]], const device uchar* ks [[buffer(8)]],
    const device uchar* kb [[buffer(9)]], const device uchar* vw [[buffer(10)]],
    const device uchar* vs [[buffer(11)]], const device uchar* vb [[buffer(12)]],
    device bfloat* output [[buffer(13)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 2560, groups = k / 64;
    const uint linear = group * 4 + simd;
    if (linear >= 13952) return;
    const device uchar* weight;
    const device uchar* scale;
    const device uchar* bias;
    uint row;
    if (linear < 640) {
        row = linear; weight = iw; scale = is; bias = ib;
    } else if (linear < 12928) {
        row = linear - 640; weight = qw; scale = qs; bias = qb;
    } else if (linear < 13440) {
        row = linear - 12928; weight = kw; scale = ks; bias = kb;
    } else {
        row = linear - 13440; weight = vw; scale = vs; bias = vb;
    }
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                q8_dot8(weight + row * k + base, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[linear] = bfloat(dot);
}

kernel void attention_normalize_rope(
    const device bfloat* projected [[buffer(0)]], const device uchar* query_norm [[buffer(1)]],
    const device uchar* key_norm [[buffer(2)]], const device uchar* index_norm [[buffer(3)]],
    const device bfloat* rope_cos [[buffer(4)]], const device bfloat* rope_sin [[buffer(5)]],
    device bfloat* selector_query [[buffer(6)]], device bfloat* attention_query [[buffer(7)]],
    device bfloat* attention_key [[buffer(8)]], uint lane [[thread_index_in_simdgroup]],
    uint head [[threadgroup_position_in_grid]]) {
    const bool is_index = head < 4;
    const bool is_query = head >= 4 && head < 28;
    const uint local_head = is_index ? head : (is_query ? head - 4 : head - 28);
    const uint dimension = is_index ? 128 : 256;
    const uint source_base = is_index ? local_head * 128
        : (is_query ? 640 + local_head * 512 : 12928 + local_head * 256);
    const device uchar* norm = is_index ? index_norm : (is_query ? query_norm : key_norm);
    float values[8];
    float square_sum = 0.0f;
    const uint items = dimension / 32;
    for (uint item = 0; item < items; ++item) {
        values[item] = float(projected[source_base + lane + item * 32]);
        square_sum += values[item] * values[item];
    }
    square_sum = simd_sum(square_sum);
    const float inverse_rms = rsqrt(square_sum / float(dimension) + 1.0e-6f);
    bfloat normalized[8];
    for (uint item = 0; item < items; ++item) {
        const uint component = lane + item * 32;
        const bfloat weight = bfloat(float(load_bf16_unaligned(norm, component)) + 1.0f);
        normalized[item] = bfloat(values[item] * inverse_rms * float(weight));
    }
    const bfloat first = normalized[0], second = normalized[1];
    normalized[0] = bfloat(float(first) * float(rope_cos[lane]) -
                           float(second) * float(rope_sin[lane]));
    normalized[1] = bfloat(float(second) * float(rope_cos[lane + 32]) +
                           float(first) * float(rope_sin[lane + 32]));
    device bfloat* output = is_index ? selector_query : (is_query ? attention_query : attention_key);
    const uint output_base = local_head * dimension;
    for (uint item = 0; item < items; ++item)
        output[output_base + lane + item * 32] = normalized[item];
}

kernel void attention_apply_gate(
    const device bfloat* attended [[buffer(0)]], const device bfloat* projected [[buffer(1)]],
    device bfloat* gated [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= 6144) return;
    const uint head = index / 256;
    const uint component = index % 256;
    const bfloat raw = projected[640 + head * 512 + 256 + component];
    const bfloat gate = bfloat(1.0f / (1.0f + metal::exp(-float(raw))));
    gated[index] = bfloat(float(attended[index]) * float(gate));
}

kernel void attention_output_projection(
    const device bfloat* input [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device uchar* scale [[buffer(2)]], const device uchar* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= 2560) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < 6144; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * 192 + base / 32)) *
                q4_dot8(weight + row * 3072 + base / 2, input + base) +
            float(load_bf16_unaligned(bias, row * 192 + base / 32)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[row] = bfloat(dot);
}

kernel void q8_output_projection_6144(
    const device bfloat* input [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device uchar* scale [[buffer(2)]], const device uchar* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows = 2560, k = 6144, groups = k / 64;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                q8_dot8(weight + row * k + base, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[row] = bfloat(dot);
}

kernel void q4_input_projection(
    const device bfloat* input [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device uchar* scale [[buffer(2)]], const device uchar* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]], constant uint& rows [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < 2560; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * 80 + base / 32)) *
                q4_dot8(weight + row * 1280 + base / 2, input + base) +
            float(load_bf16_unaligned(bias, row * 80 + base / 32)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[row] = bfloat(dot);
}

kernel void q8_input_projection(
    const device bfloat* input [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device uchar* scale [[buffer(2)]], const device uchar* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]], constant uint& rows [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 2560, groups = k / 64;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                q8_dot8(weight + row * k + base, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[row] = bfloat(dot);
}

kernel void q8_gdn_input_projections(
    const device bfloat* input [[buffer(0)]],
    const device uchar* qkv_weight [[buffer(1)]],
    const device uchar* qkv_scale [[buffer(2)]],
    const device uchar* qkv_bias [[buffer(3)]],
    const device uchar* z_weight [[buffer(4)]],
    const device uchar* z_scale [[buffer(5)]],
    const device uchar* z_bias [[buffer(6)]],
    const device uchar* b_weight [[buffer(7)]],
    const device uchar* b_scale [[buffer(8)]],
    const device uchar* b_bias [[buffer(9)]],
    const device uchar* a_weight [[buffer(10)]],
    const device uchar* a_scale [[buffer(11)]],
    const device uchar* a_bias [[buffer(12)]],
    device bfloat* output [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 2560, groups = k / 64;
    const device uchar* weight;
    const device uchar* scale;
    const device uchar* bias;
    uint local_group;
    uint output_base;
    if (group < 2560) {
        weight = qkv_weight; scale = qkv_scale; bias = qkv_bias;
        local_group = group; output_base = 0;
    } else if (group < 4096) {
        weight = z_weight; scale = z_scale; bias = z_bias;
        local_group = group - 2560; output_base = 10240;
    } else if (group < 4108) {
        weight = b_weight; scale = b_scale; bias = b_bias;
        local_group = group - 4096; output_base = 16384;
    } else {
        weight = a_weight; scale = a_scale; bias = a_bias;
        local_group = group - 4108; output_base = 16432;
    }
    const uint row = local_group * 4 + simd;
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                q8_dot8(weight + row * k + base, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[output_base + row] = bfloat(dot);
}

kernel void gdn_prework(
    const device bfloat* projected [[buffer(0)]], device bfloat* convolution [[buffer(1)]],
    const device uchar* convolution_weight [[buffer(2)]],
    const device uchar* decay_log [[buffer(3)]], const device uchar* decay_bias [[buffer(4)]],
    device bfloat* query [[buffer(5)]], device bfloat* key [[buffer(6)]],
    device bfloat* value [[buffer(7)]], device bfloat* decay [[buffer(8)]],
    device bfloat* beta [[buffer(9)]], uint lane [[thread_index_in_simdgroup]],
    uint logical_head [[threadgroup_position_in_grid]]) {
    constexpr uint hk = 16, hv = 48, dimension = 128, width = 10240;
    const bool is_query = logical_head < hk;
    const bool is_key = logical_head >= hk && logical_head < 2 * hk;
    const uint head = is_query ? logical_head : (is_key ? logical_head - hk
                                                       : logical_head - 2 * hk);
    const uint channel_base = is_query ? head * dimension
        : (is_key ? hk * dimension + head * dimension
                  : 2 * hk * dimension + head * dimension);
    float square_sum = 0.0f;
    bfloat activated[4];
    for (uint item = 0; item < 4; ++item) {
        const uint channel = channel_base + lane * 4 + item;
        float sum = 0.0f;
        for (uint tap = 0; tap < 3; ++tap)
            sum += float(convolution[tap * width + channel]) *
                   float(load_bf16_unaligned(convolution_weight, channel * 4 + tap));
        sum += float(projected[channel]) *
               float(load_bf16_unaligned(convolution_weight, channel * 4 + 3));
        const bfloat convolved = bfloat(sum);
        const bfloat small = bfloat(1.0f) /
            bfloat(1.0f + metal::exp(metal::abs(float(convolved))));
        activated[item] = bfloat(float(convolved) *
            float(convolved < bfloat(0.0f) ? small : bfloat(1.0f) - small));
        square_sum += float(activated[item]) * float(activated[item]);
        convolution[channel] = convolution[width + channel];
        convolution[width + channel] = convolution[2 * width + channel];
        convolution[2 * width + channel] = projected[channel];
    }
    if (is_query || is_key) {
        square_sum = simd_sum(square_sum);
        const float inverse_rms = rsqrt(square_sum / 128.0f + 1.0e-6f);
        const float scale = is_query ? (1.0f / 128.0f) : (1.0f / sqrt(128.0f));
        device bfloat* destination = is_query ? query : key;
        for (uint item = 0; item < 4; ++item)
            destination[head * dimension + lane * 4 + item] =
                bfloat(scale * float(bfloat(float(activated[item]) * inverse_rms)));
    } else {
        for (uint item = 0; item < 4; ++item)
            value[head * dimension + lane * 4 + item] = activated[item];
        if (lane == 0) {
            const bfloat beta_input = projected[16384 + head];
            beta[head] = bfloat(1.0f / (1.0f + metal::exp(-float(beta_input))));
            const bfloat biased = bfloat(float(projected[16432 + head]) +
                                         float(load_bf16_unaligned(decay_bias, head)));
            const float softplus = metal::log(1.0f + metal::exp(float(biased)));
            decay[head] = bfloat(metal::exp(
                -metal::exp(float(load_bf16_unaligned(decay_log, head))) * softplus));
        }
    }
}

kernel void gdn_recurrence(
    const device bfloat* query [[buffer(0)]], const device bfloat* key [[buffer(1)]],
    const device bfloat* value [[buffer(2)]], const device bfloat* decay [[buffer(3)]],
    const device bfloat* beta [[buffer(4)]], device bfloat* recurrent [[buffer(5)]],
    device bfloat* output [[buffer(6)]], uint lane [[thread_index_in_simdgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr uint dimension = 128, value_dimension = 128;
    const uint dv = group.x, hv = group.y, hk = hv / 3;
    device bfloat* state = recurrent + (hv * value_dimension + dv) * dimension;
    bfloat local[4];
    bfloat recalled = bfloat(0.0f);
    for (uint item = 0; item < 4; ++item) {
        const uint component = lane * 4 + item;
        local[item] = bfloat(float(state[component]) * float(decay[hv]));
        const bfloat product = bfloat(float(local[item]) *
                                      float(key[hk * dimension + component]));
        recalled = bfloat(float(recalled) + float(product));
    }
    recalled = bfloat(simd_sum(float(recalled)));
    const bfloat residual = bfloat(float(value[hv * value_dimension + dv]) - float(recalled));
    const bfloat delta = bfloat(float(residual) * float(beta[hv]));
    bfloat result = bfloat(0.0f);
    for (uint item = 0; item < 4; ++item) {
        const uint component = lane * 4 + item;
        const bfloat update = bfloat(float(key[hk * dimension + component]) * float(delta));
        local[item] = bfloat(float(local[item]) + float(update));
        state[component] = local[item];
        const bfloat product = bfloat(float(local[item]) *
                                      float(query[hk * dimension + component]));
        result = bfloat(float(result) + float(product));
    }
    result = bfloat(simd_sum(float(result)));
    if (lane == 0) output[hv * value_dimension + dv] = result;
}

kernel void gdn_norm_gate(
    const device bfloat* recurrent_output [[buffer(0)]],
    const device bfloat* projected [[buffer(1)]], const device uchar* norm [[buffer(2)]],
    device bfloat* output [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],
    uint head [[threadgroup_position_in_grid]]) {
    float values[4];
    float square_sum = 0.0f;
    for (uint item = 0; item < 4; ++item) {
        const uint component = lane * 4 + item;
        values[item] = float(recurrent_output[head * 128 + component]);
        square_sum += values[item] * values[item];
    }
    square_sum = simd_sum(square_sum);
    const float inverse_rms = rsqrt(square_sum / 128.0f + 1.0e-6f);
    for (uint item = 0; item < 4; ++item) {
        const uint component = lane * 4 + item;
        const bfloat normalized = bfloat(
            values[item] * inverse_rms * float(load_bf16_unaligned(norm, component)));
        const bfloat raw_gate = projected[10240 + head * 128 + component];
        const bfloat gate = bfloat(1.0f / (1.0f + metal::exp(-float(raw_gate))));
        output[head * 128 + component] = bfloat(float(normalized) * float(gate));
    }
}

kernel void gdn_output_projection(
    const device bfloat* input [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    const device uchar* scale [[buffer(2)]], const device uchar* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= 2560) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < 6144; base += 256) {
        float sum = 0.0f;
        for (uint component = 0; component < 8; ++component)
            sum += float(input[base + component]);
        dot += float(load_bf16_unaligned(scale, row * 192 + base / 32)) *
                q4_dot8(weight + row * 3072 + base / 2, input + base) +
            float(load_bf16_unaligned(bias, row * 192 + base / 32)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) output[row] = bfloat(dot);
}

kernel void hc_write(
    const device bfloat* stream [[buffer(0)]], const device bfloat* block [[buffer(1)]],
    const device bfloat* injection [[buffer(2)]], device bfloat* output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= 10240) return;
    const uint hc = index / 2560;
    const uint column = index % 2560;
    const bfloat update = bfloat(float(block[column]) * float(injection[hc]));
    output[index] = bfloat(float(stream[index]) + float(update));
}

kernel void hc_normalize(
    const device bfloat* stream [[buffer(0)]], const device uchar* norm [[buffer(1)]],
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
        const bfloat weight = bfloat(float(load_bf16_unaligned(norm, index)) + 1.0f);
        normalized[index] = bfloat(float(value) * float(weight));
    }
}

kernel void hc_down_injection(
    const device bfloat* x [[buffer(0)]],
    const device uchar* down_weight [[buffer(1)]],
    const device uchar* down_scale [[buffer(2)]],
    const device uchar* down_bias [[buffer(3)]],
    const device uchar* injection_weight [[buffer(4)]],
    const device uchar* injection_scale [[buffer(5)]],
    const device uchar* injection_bias [[buffer(6)]],
    device bfloat* activation [[buffer(7)]], device bfloat* injection [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    const bool is_down = row < 320;
    const uint local_row = is_down ? row : row - 320;
    const device uchar* weight = is_down ? down_weight : injection_weight;
    const device uchar* scale = is_down ? down_scale : injection_scale;
    const device uchar* bias = is_down ? down_bias : injection_bias;
    float dot = 0.0f;
    for (uint base = tid * 8; base < 10240; base += 2048) {
        const device uchar* bytes = weight + local_row * 5120 + base / 2;
        const float s = float(load_bf16_unaligned(scale, local_row * 320 + base / 32));
        const float b = float(load_bf16_unaligned(bias, local_row * 320 + base / 32));
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

kernel void hc_down_injection_q8(
    const device bfloat* x [[buffer(0)]],
    const device uchar* down_weight [[buffer(1)]],
    const device uchar* down_scale [[buffer(2)]],
    const device uchar* down_bias [[buffer(3)]],
    const device uchar* injection_weight [[buffer(4)]],
    const device uchar* injection_scale [[buffer(5)]],
    const device uchar* injection_bias [[buffer(6)]],
    device bfloat* activation [[buffer(7)]], device bfloat* injection [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    constexpr uint k = 10240, groups = k / 64;
    threadgroup float partial[8];
    const bool is_down = row < 320;
    const uint local_row = is_down ? row : row - 320;
    const device uchar* weight = is_down ? down_weight : injection_weight;
    const device uchar* scale = is_down ? down_scale : injection_scale;
    const device uchar* bias = is_down ? down_bias : injection_bias;
    float dot = 0.0f;
    for (uint base = tid * 8; base < k; base += 2048) {
        const device uchar* bytes = weight + local_row * k + base;
        const float s = float(load_bf16_unaligned(scale, local_row * groups + base / 64));
        const float b = float(load_bf16_unaligned(bias, local_row * groups + base / 64));
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += float(x[base + i]);
        dot += s * q8_dot8(bytes, x + base) + b * sum;
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
    const device uchar* weight [[buffer(2)]], const device uchar* scale [[buffer(3)]],
    const device uchar* bias [[buffer(4)]], device bfloat* mixed [[buffer(5)]],
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
            const float s = float(load_bf16_unaligned(scale, row * 10 + base / 32));
            const float b = float(load_bf16_unaligned(bias, row * 10 + base / 32));
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

kernel void hc_up_mix_q8(
    const device bfloat* normalized [[buffer(0)]], const device bfloat* activation [[buffer(1)]],
    const device uchar* weight [[buffer(2)]], const device uchar* scale [[buffer(3)]],
    const device uchar* bias [[buffer(4)]], device bfloat* mixed [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 320, groups = k / 64;
    const uint column = group * 8 + simd;
    if (column >= 2560) return;
    float stream_sum = 0.0f;
    for (uint hc = 0; hc < 4; ++hc) {
        const uint row = hc * 2560 + column;
        float dot = 0.0f;
        for (uint base = lane * 8; base < k; base += 256) {
            float sum = 0.0f;
            for (uint i = 0; i < 8; ++i) sum += float(activation[base + i]);
            dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                    q8_dot8(weight + row * k + base, activation + base) +
                float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
        }
        dot = simd_sum(dot);
        const bfloat gate = bfloat(1.0f / (1.0f + metal::exp(-float(bfloat(dot)))));
        stream_sum += float(bfloat(float(gate) * float(normalized[row])));
    }
    if (lane == 0) mixed[column] = bfloat(float(bfloat(stream_sum)) / 4.0f);
}

kernel void healing_left(
    const device bfloat* input [[buffer(0)]], const device uchar* weight [[buffer(1)]],
    device bfloat* hidden [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint column = group * 4 + simd;
    if (column >= 64) return;
    float dot = 0.0f;
    for (uint row = lane; row < 2560; row += 32)
        dot += float(input[row]) * float(load_bf16_unaligned(weight, row * 64 + column));
    dot = simd_sum(dot);
    if (lane == 0) hidden[column] = bfloat(dot);
}

kernel void healing_right_write(
    const device bfloat* block [[buffer(0)]], const device bfloat* hidden [[buffer(1)]],
    const device uchar* weight [[buffer(2)]], const device bfloat* stream [[buffer(3)]],
    const device bfloat* injection [[buffer(4)]], device bfloat* output [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint column = group * 4 + simd;
    if (column >= 2560) return;
    float dot = 0.0f;
    for (uint row = lane; row < 64; row += 32)
        dot += float(hidden[row]) * float(load_bf16_unaligned(weight, row * 2560 + column));
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

kernel void ple_fused_update(
    const device bfloat* projected [[buffer(0)]], const device bfloat* stream [[buffer(1)]],
    const device uchar* norm_key [[buffer(2)]], const device uchar* norm_query [[buffer(3)]],
    const device uchar* norm_conv [[buffer(4)]], const device uchar* conv_weight [[buffer(5)]],
    device bfloat* conv_state [[buffer(6)]], device bfloat* output [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint hc [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[8];
    threadgroup float inverse_key, inverse_query, gate_value, inverse_conv;
    threadgroup bfloat normalized[2560];
    float key_square = 0.0f, query_square = 0.0f;
    const uint stream_base = hc * 2560;
    for (uint column = tid; column < 2560; column += 256) {
        const float key = float(projected[stream_base + column]);
        const float query = float(stream[stream_base + column]);
        key_square += key * key;
        query_square += query * query;
    }
    key_square = simd_sum(key_square);
    query_square = simd_sum(query_square);
    if (lane == 0) reduction[simd] = key_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += reduction[i];
        inverse_key = rsqrt(sum / 2560.0f + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) reduction[simd] = query_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += reduction[i];
        inverse_query = rsqrt(sum / 2560.0f + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint column = tid; column < 2560; column += 256) {
        const float key = float(projected[stream_base + column]) * inverse_key *
            (float(load_bf16_unaligned(norm_key, stream_base + column)) + 1.0f);
        const float query = float(stream[stream_base + column]) * inverse_query *
            (float(load_bf16_unaligned(norm_query, stream_base + column)) + 1.0f);
        dot += key * query;
    }
    dot = simd_sum(dot);
    if (lane == 0) reduction[simd] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += reduction[i];
        sum /= sqrt(2560.0f);
        const float signed_root = copysign(sqrt(max(abs(sum), 1.0e-6f)), sum);
        gate_value = 1.0f / (1.0f + metal::exp(-signed_root));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gated_square = 0.0f;
    for (uint column = tid; column < 2560; column += 256) {
        const bfloat gated = bfloat(gate_value * float(projected[10240 + column]));
        output[stream_base + column] = gated;
        gated_square += float(gated) * float(gated);
    }
    gated_square = simd_sum(gated_square);
    if (lane == 0) reduction[simd] = gated_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += reduction[i];
        inverse_conv = rsqrt(sum / 2560.0f + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < 2560; column += 256) {
        const uint component = stream_base + column;
        normalized[column] = bfloat(float(output[component]) * inverse_conv *
            (float(load_bf16_unaligned(norm_conv, component)) + 1.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < 2560; column += 256) {
        const uint component = stream_base + column;
        float conv = 0.0f;
        conv += float(conv_state[component]) *
            float(load_bf16_unaligned(conv_weight, component * 4));
        conv += float(conv_state[3 * 10240 + component]) *
            float(load_bf16_unaligned(conv_weight, component * 4 + 1));
        conv += float(conv_state[6 * 10240 + component]) *
            float(load_bf16_unaligned(conv_weight, component * 4 + 2));
        conv += float(normalized[column]) *
            float(load_bf16_unaligned(conv_weight, component * 4 + 3));
        const float silu = conv / (1.0f + metal::exp(-conv));
        output[component] = bfloat(float(output[component]) + silu + float(stream[component]));
        for (uint row = 0; row < 8; ++row)
            conv_state[row * 10240 + component] = conv_state[(row + 1) * 10240 + component];
        conv_state[8 * 10240 + component] = normalized[column];
    }
}

kernel void embedding_q4_stream(
    const device uchar* weight [[buffer(0)]], const device uchar* scale [[buffer(1)]],
    const device uchar* bias [[buffer(2)]], device bfloat* stream [[buffer(3)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 2560) return;
    const uint packed = uint(weight[column / 2]);
    const uint quantized = (packed >> ((column & 1) * 4)) & 15;
    const uint group = column / 32;
    const bfloat value = bfloat(float(quantized) *
        float(load_bf16_unaligned(scale, group)) +
        float(load_bf16_unaligned(bias, group)));
    for (uint hc = 0; hc < 4; ++hc) stream[hc * 2560 + column] = value;
}

kernel void embedding_q8_stream(
    const device uchar* weight [[buffer(0)]], const device uchar* scale [[buffer(1)]],
    const device uchar* bias [[buffer(2)]], device bfloat* stream [[buffer(3)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= 2560) return;
    const uint group = column / 64;
    const bfloat value = bfloat(float(weight[column]) *
        float(load_bf16_unaligned(scale, group)) +
        float(load_bf16_unaligned(bias, group)));
    for (uint hc = 0; hc < 4; ++hc) stream[hc * 2560 + column] = value;
}

kernel void hc_down_only(
    const device bfloat* input [[buffer(0)]],
    const device uchar* weight [[buffer(1)]], const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]], device bfloat* activation [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= 320) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < 10240; base += 256) {
        float sum = 0.0f;
        for (uint item = 0; item < 8; ++item) sum += float(input[base + item]);
        dot += float(load_bf16_unaligned(scale, row * 320 + base / 32)) *
                q4_dot8(weight + row * 5120 + base / 2, input + base) +
            float(load_bf16_unaligned(bias, row * 320 + base / 32)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) {
        const bfloat value = bfloat(dot / 4.0f);
        const bfloat gate = bfloat(1.0f / (1.0f + metal::exp(-float(value))));
        activation[row] = value * gate;
    }
}

kernel void hc_down_only_q8(
    const device bfloat* input [[buffer(0)]],
    const device uchar* weight [[buffer(1)]], const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]], device bfloat* activation [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 10240, groups = k / 64;
    const uint row = group * 4 + simd;
    if (row >= 320) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint item = 0; item < 8; ++item) sum += float(input[base + item]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                q8_dot8(weight + row * k + base, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) {
        const bfloat value = bfloat(dot / 4.0f);
        const bfloat gate = bfloat(1.0f / (1.0f + metal::exp(-float(value))));
        activation[row] = value * gate;
    }
}

kernel void lm_head_q4(
    const device bfloat* input [[buffer(0)]],
    const device uchar* weight [[buffer(1)]], const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]], device float* logits [[buffer(4)]],
    constant uint& rows [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < 2560; base += 256) {
        float sum = 0.0f;
        for (uint item = 0; item < 8; ++item) sum += float(input[base + item]);
        dot += float(load_bf16_unaligned(scale, row * 80 + base / 32)) *
                q4_dot8(weight + row * 1280 + base / 2, input + base) +
            float(load_bf16_unaligned(bias, row * 80 + base / 32)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) logits[row] = dot;
}

kernel void lm_head_q8(
    const device bfloat* input [[buffer(0)]],
    const device uchar* weight [[buffer(1)]], const device uchar* scale [[buffer(2)]],
    const device uchar* bias [[buffer(3)]], device float* logits [[buffer(4)]],
    constant uint& rows [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint k = 2560, groups = k / 64;
    const uint row = group * 4 + simd;
    if (row >= rows) return;
    float dot = 0.0f;
    for (uint base = lane * 8; base < k; base += 256) {
        float sum = 0.0f;
        for (uint item = 0; item < 8; ++item) sum += float(input[base + item]);
        dot += float(load_bf16_unaligned(scale, row * groups + base / 64)) *
                q8_dot8(weight + row * k + base, input + base) +
            float(load_bf16_unaligned(bias, row * groups + base / 64)) * sum;
    }
    dot = simd_sum(dot);
    if (lane == 0) logits[row] = dot;
}

kernel void head_top2_reduce(
    const device float* logits [[buffer(0)]], device float* candidate_values [[buffer(1)]],
    device uint* candidate_ids [[buffer(2)]], constant uint& rows [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float best_values[256];
    threadgroup float second_values[256];
    threadgroup uint best_ids[256];
    threadgroup uint second_ids[256];
    float best_value = -INFINITY, second_value = -INFINITY;
    uint best_id = 0xffffffffu, second_id = 0xffffffffu;
    const uint begin = group * 1024u + tid * 4u;
    for (uint offset = 0; offset < 4u; ++offset) {
        const uint id = begin + offset;
        if (id >= rows) continue;
        const float value = logits[id];
        if (value > best_value || (value == best_value && id < best_id)) {
            second_value = best_value; second_id = best_id;
            best_value = value; best_id = id;
        } else if (id != best_id &&
                   (value > second_value || (value == second_value && id < second_id))) {
            second_value = value; second_id = id;
        }
    }
    best_values[tid] = best_value;
    second_values[tid] = second_value;
    best_ids[tid] = best_id;
    second_ids[tid] = second_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) {
            float bv = best_values[tid], sv = second_values[tid];
            uint bi = best_ids[tid], si = second_ids[tid];
            const uint other = tid + stride;
            const float values[2] = {best_values[other], second_values[other]};
            const uint ids[2] = {best_ids[other], second_ids[other]};
            for (uint item = 0; item < 2u; ++item) {
                const float value = values[item];
                const uint id = ids[item];
                if (id == 0xffffffffu) continue;
                if (value > bv || (value == bv && id < bi)) {
                    sv = bv; si = bi;
                    bv = value; bi = id;
                } else if (id != bi &&
                           (value > sv || (value == sv && id < si))) {
                    sv = value; si = id;
                }
            }
            best_values[tid] = bv; second_values[tid] = sv;
            best_ids[tid] = bi; second_ids[tid] = si;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const uint output = group * 2u;
        candidate_values[output] = best_values[0];
        candidate_values[output + 1u] = second_values[0];
        candidate_ids[output] = best_ids[0];
        candidate_ids[output + 1u] = second_ids[0];
    }
}
)metal";

} // namespace qwen38::persistent_metal
