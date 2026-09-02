#pragma once

// Adapted from MTPLX's moe_glu_decode.py (Copyright 2026 MTPLX), Apache-2.0.
// Modified for separate gate/up packs and fixed Qwen3.8 Flash dimensions.

#include <string_view>

namespace qwen38::moe_metal {

inline constexpr std::string_view header = R"metal(
#include <metal_stdlib>
using namespace metal;
)metal";

inline constexpr std::string_view gate_up = R"metal(
    constexpr int K = 2560;
    constexpr int GS = 64;
    constexpr int NG = K / GS;
    constexpr int WPG = GS / 8;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gsid = threadgroup_position_in_grid.x * 32 + sg;
    if (gsid >= 6400) return;
    const int slot = gsid / 640;
    const int row = gsid % 640;
    const uint expert = experts[slot];
    const device uint32_t* gwr = gw + ((size_t)expert * 640 + row) * (K / 8);
    const device uint32_t* uwr = uw + ((size_t)expert * 640 + row) * (K / 8);
    const device T* gsr = gs + ((size_t)expert * 640 + row) * NG;
    const device T* gbr = gb + ((size_t)expert * 640 + row) * NG;
    const device T* usr = us + ((size_t)expert * 640 + row) * NG;
    const device T* ubr = ub + ((size_t)expert * 640 + row) * NG;
    float ag = 0.0f, au = 0.0f;
    for (int g = lane; g < NG; g += 32) {
        const device T* xg = x + g * GS;
        float qg = 0.0f, qu = 0.0f, xs = 0.0f;
        for (int wi = 0; wi < WPG; ++wi) {
            uint32_t wg = gwr[g * WPG + wi];
            uint32_t wu = uwr[g * WPG + wi];
            for (int nib = 0; nib < 8; ++nib) {
                float xv = (float)xg[wi * 8 + nib];
                xs += xv;
                qg += (float)((wg >> (4 * nib)) & 15) * xv;
                qu += (float)((wu >> (4 * nib)) & 15) * xv;
            }
        }
        ag += (float)gsr[g] * qg + (float)gbr[g] * xs;
        au += (float)usr[g] * qu + (float)ubr[g] * xs;
    }
    ag = simd_sum(ag); au = simd_sum(au);
    if (lane == 0) {
        float gv = (float)((T)ag), uv = (float)((T)au);
        h[(size_t)slot * 640 + row] =
            (T)((float)((T)(gv / (1.0f + metal::exp(-gv)))) * uv);
    }
)metal";

// The 16-bit mask unpacking pattern is adapted from Apple MLX's qmv kernel
// (Copyright Apple Inc., MIT license) for this fixed selected-expert layout.
inline constexpr std::string_view down = R"metal(
    constexpr int K = 640;
    constexpr int GS = 64;
    constexpr int ROWS = 4;
    constexpr int VALUES = 8;
    constexpr int BLOCK = VALUES * 32;
    constexpr int NG = K / GS;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = threadgroup_position_in_grid.x * 8 + sg * ROWS;
    if (row0 >= 2560) return;
    float acc[ROWS] = {0.0f};
    for (int slot = 0; slot < 10; ++slot) {
        const uint expert = experts[slot];
        const int packed_row = K / 8;
        const size_t base = (size_t)expert * 2560 + row0;
        const device uint8_t* wr =
            (const device uint8_t*)(dw + base * packed_row) + lane * 4;
        const device T* sr = ds + base * NG + lane / 8;
        const device T* br = db + base * NG + lane / 8;
        const device T* xv = h + slot * K + lane * VALUES;
        float dot[ROWS] = {0.0f};
        for (int k0 = 0; k0 < K; k0 += BLOCK) {
            float xt[VALUES] = {0.0f};
            float sum = 0.0f;
            if (k0 + lane * VALUES < K) {
                for (int i = 0; i < VALUES; i += 4) {
                    float x0 = (float)xv[i], x1 = (float)xv[i + 1];
                    float x2 = (float)xv[i + 2], x3 = (float)xv[i + 3];
                    sum += x0 + x1 + x2 + x3;
                    xt[i] = x0;
                    xt[i + 1] = x1 / 16.0f;
                    xt[i + 2] = x2 / 256.0f;
                    xt[i + 3] = x3 / 4096.0f;
                }
            }
            for (int r = 0; r < ROWS; ++r) {
                const device uint16_t* words =
                    (const device uint16_t*)(wr + r * packed_row * 4);
                float q = 0.0f;
                for (int i = 0; i < VALUES / 4; ++i) {
                    uint16_t value = words[i];
                    q += xt[4 * i] * (value & 0x000f) +
                        xt[4 * i + 1] * (value & 0x00f0) +
                        xt[4 * i + 2] * (value & 0x0f00) +
                        xt[4 * i + 3] * (value & 0xf000);
                }
                dot[r] += (float)sr[r * NG] * q + (float)br[r * NG] * sum;
            }
            wr += BLOCK / 2;
            sr += BLOCK / GS;
            br += BLOCK / GS;
            xv += BLOCK;
        }
        for (int r = 0; r < ROWS; ++r) {
            dot[r] = simd_sum(dot[r]);
            acc[r] += rw[slot] * (float)((T)dot[r]);
        }
    }
    if (lane == 0) {
        for (int r = 0; r < ROWS; ++r) y[row0 + r] = (T)acc[r];
    }
)metal";

inline constexpr std::string_view gate_up_verify = R"metal(
    constexpr int K = 2560;
    constexpr int GS = 64;
    constexpr int NG = K / GS;
    constexpr int WPG = GS / 8;
    constexpr int SLOTS = 10;
    constexpr int EXPERT_ROWS = 640;
    constexpr int GROUPS_PER_BATCH = SLOTS * EXPERT_ROWS;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gsid = threadgroup_position_in_grid.x * 32 + sg;
    const uint batch = gsid / GROUPS_PER_BATCH;
    const uint local = gsid % GROUPS_PER_BATCH;
    const int slot = local / EXPERT_ROWS;
    const int row = local % EXPERT_ROWS;
    const uint expert = experts[batch * SLOTS + slot];
    const device T* xb = x + (size_t)batch * K;
    const device uint32_t* gwr = gw + ((size_t)expert * EXPERT_ROWS + row) * (K / 8);
    const device uint32_t* uwr = uw + ((size_t)expert * EXPERT_ROWS + row) * (K / 8);
    const device T* gsr = gs + ((size_t)expert * EXPERT_ROWS + row) * NG;
    const device T* gbr = gb + ((size_t)expert * EXPERT_ROWS + row) * NG;
    const device T* usr = us + ((size_t)expert * EXPERT_ROWS + row) * NG;
    const device T* ubr = ub + ((size_t)expert * EXPERT_ROWS + row) * NG;
    float ag = 0.0f, au = 0.0f;
    for (int g = lane; g < NG; g += 32) {
        const device T* xg = xb + g * GS;
        float qg = 0.0f, qu = 0.0f, xs = 0.0f;
        for (int wi = 0; wi < WPG; ++wi) {
            uint32_t wg = gwr[g * WPG + wi];
            uint32_t wu = uwr[g * WPG + wi];
            for (int nib = 0; nib < 8; ++nib) {
                float xv = (float)xg[wi * 8 + nib];
                xs += xv;
                qg += (float)((wg >> (4 * nib)) & 15) * xv;
                qu += (float)((wu >> (4 * nib)) & 15) * xv;
            }
        }
        ag += (float)gsr[g] * qg + (float)gbr[g] * xs;
        au += (float)usr[g] * qu + (float)ubr[g] * xs;
    }
    ag = simd_sum(ag); au = simd_sum(au);
    if (lane == 0) {
        float gv = (float)((T)ag), uv = (float)((T)au);
        h[((size_t)batch * SLOTS + slot) * EXPERT_ROWS + row] =
            (T)((float)((T)(gv / (1.0f + metal::exp(-gv)))) * uv);
    }
)metal";

inline constexpr std::string_view down_verify = R"metal(
    constexpr int K = 640;
    constexpr int GS = 64;
    constexpr int ROWS = 4;
    constexpr int VALUES = 8;
    constexpr int BLOCK = VALUES * 32;
    constexpr int NG = K / GS;
    constexpr int SLOTS = 10;
    constexpr int OUTPUT_ROWS = 2560;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint global_row = threadgroup_position_in_grid.x * 8 + sg * ROWS;
    const uint batch = global_row / OUTPUT_ROWS;
    const uint row0 = global_row % OUTPUT_ROWS;
    float acc[ROWS] = {0.0f};
    for (int slot = 0; slot < SLOTS; ++slot) {
        const uint expert = experts[batch * SLOTS + slot];
        const int packed_row = K / 8;
        const size_t base = (size_t)expert * OUTPUT_ROWS + row0;
        const device uint8_t* wr =
            (const device uint8_t*)(dw + base * packed_row) + lane * 4;
        const device T* sr = ds + base * NG + lane / 8;
        const device T* br = db + base * NG + lane / 8;
        const device T* xv = h + ((size_t)batch * SLOTS + slot) * K + lane * VALUES;
        float dot[ROWS] = {0.0f};
        for (int k0 = 0; k0 < K; k0 += BLOCK) {
            float xt[VALUES] = {0.0f};
            float sum = 0.0f;
            if (k0 + lane * VALUES < K) {
                for (int i = 0; i < VALUES; i += 4) {
                    float x0 = (float)xv[i], x1 = (float)xv[i + 1];
                    float x2 = (float)xv[i + 2], x3 = (float)xv[i + 3];
                    sum += x0 + x1 + x2 + x3;
                    xt[i] = x0;
                    xt[i + 1] = x1 / 16.0f;
                    xt[i + 2] = x2 / 256.0f;
                    xt[i + 3] = x3 / 4096.0f;
                }
            }
            for (int r = 0; r < ROWS; ++r) {
                const device uint16_t* words =
                    (const device uint16_t*)(wr + r * packed_row * 4);
                float q = 0.0f;
                for (int i = 0; i < VALUES / 4; ++i) {
                    uint16_t value = words[i];
                    q += xt[4 * i] * (value & 0x000f) +
                        xt[4 * i + 1] * (value & 0x00f0) +
                        xt[4 * i + 2] * (value & 0x0f00) +
                        xt[4 * i + 3] * (value & 0xf000);
                }
                dot[r] += (float)sr[r * NG] * q + (float)br[r * NG] * sum;
            }
            wr += BLOCK / 2;
            sr += BLOCK / GS;
            br += BLOCK / GS;
            xv += BLOCK;
        }
        for (int r = 0; r < ROWS; ++r) {
            dot[r] = simd_sum(dot[r]);
            acc[r] += rw[batch * SLOTS + slot] * (float)((T)dot[r]);
        }
    }
    if (lane == 0) {
        for (int r = 0; r < ROWS; ++r) {
            y[(size_t)batch * OUTPUT_ROWS + row0 + r] = (T)acc[r];
        }
    }
)metal";

// Compact qmeta kernels are adapted from mlx-serve (Copyright 2026 David
// Dalcu), MIT licensed. The Q4 codes stay unchanged; only the BF16 affine
// scale/bias pair is decoded from a bank-local packed dictionary index.
inline constexpr std::string_view qmeta_gate_up = R"metal(
    constexpr int K = 2560;
    constexpr int GS = 64;
    constexpr int NG = K / GS;
    constexpr int WPG = GS / 8;
    constexpr int SLOTS = 10;
    constexpr int EXPERT_ROWS = 640;
    constexpr int GROUPS_PER_BATCH = SLOTS * EXPERT_ROWS;
    constexpr int GATE_ROW_BYTES = (NG * QGBITS + 7) / 8;
    constexpr int UP_ROW_BYTES = (NG * QUBITS + 7) / 8;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint group_id = threadgroup_position_in_grid.x * 32 + sg;
    const uint batch = group_id / GROUPS_PER_BATCH;
    const uint local = group_id % GROUPS_PER_BATCH;
    const int slot = local / EXPERT_ROWS;
    const int row_index = local % EXPERT_ROWS;
    const uint expert = experts[batch * SLOTS + slot];
    const device T* input_row = x + (size_t)batch * K;
    const size_t row = (size_t)expert * EXPERT_ROWS + row_index;
    const device uint32_t* gate_row = gate_weight + row * (K / 8);
    const device uint32_t* up_row = up_weight + row * (K / 8);
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int group = (int)lane; group < NG; group += 32) {
        uint gate_tag;
        if (QGBITS == 16 && ALIGNED16 != 0) {
            gate_tag = uint(((const device ushort*)gate_tags)[row * NG + group]);
        } else {
            const int gate_bit = group * QGBITS;
            const size_t gate_pos = row * GATE_ROW_BYTES + (gate_bit >> 3);
            const uint gate_shift = gate_bit & 7;
            uint gate_window = uint(gate_tags[gate_pos]) |
                (uint(gate_tags[gate_pos + 1]) << 8);
            if (gate_shift + QGBITS > 16) {
                gate_window |= uint(gate_tags[gate_pos + 2]) << 16;
            }
            gate_tag = (gate_window >> gate_shift) & ((1u << QGBITS) - 1u);
        }
        const uint gate_pair = gate_dictionary[gate_tag];
        const float gate_scale = float(as_type<T>(ushort(gate_pair & 65535u)));
        const float gate_bias = float(as_type<T>(ushort(gate_pair >> 16)));
        uint up_tag;
        if (QUBITS == 16 && ALIGNED16 != 0) {
            up_tag = uint(((const device ushort*)up_tags)[row * NG + group]);
        } else {
            const int up_bit = group * QUBITS;
            const size_t up_pos = row * UP_ROW_BYTES + (up_bit >> 3);
            const uint up_shift = up_bit & 7;
            uint up_window = uint(up_tags[up_pos]) | (uint(up_tags[up_pos + 1]) << 8);
            if (up_shift + QUBITS > 16) up_window |= uint(up_tags[up_pos + 2]) << 16;
            up_tag = (up_window >> up_shift) & ((1u << QUBITS) - 1u);
        }
        const uint up_pair = up_dictionary[up_tag];
        const float up_scale = float(as_type<T>(ushort(up_pair & 65535u)));
        const float up_bias = float(as_type<T>(ushort(up_pair >> 16)));
        const device T* input_group = input_row + group * GS;
        float gate_dot = 0.0f;
        float up_dot = 0.0f;
        float input_sum = 0.0f;
        for (int word = 0; word < WPG; ++word) {
            const uint packed_gate = gate_row[group * WPG + word];
            const uint packed_up = up_row[group * WPG + word];
            for (int nibble = 0; nibble < 8; ++nibble) {
                const float value = float(input_group[word * 8 + nibble]);
                input_sum += value;
                gate_dot += float((packed_gate >> (4 * nibble)) & 15u) * value;
                up_dot += float((packed_up >> (4 * nibble)) & 15u) * value;
            }
        }
        gate_acc += gate_scale * gate_dot + gate_bias * input_sum;
        up_acc += up_scale * up_dot + up_bias * input_sum;
    }
    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);
    if (lane == 0) {
        const float gate = float(T(gate_acc));
        const float up = float(T(up_acc));
        hidden[((size_t)batch * SLOTS + slot) * EXPERT_ROWS + row_index] =
            T(float(T(gate / (1.0f + metal::exp(-gate)))) * up);
    }
)metal";

inline constexpr std::string_view qmeta_down_reduce = R"metal(
    constexpr int K = 640;
    constexpr int GS = 64;
    constexpr int ROWS_PER_SIMD = 4;
    constexpr int VALUES = 8;
    constexpr int BLOCK = VALUES * 32;
    constexpr int GROUPS = K / 64;
    constexpr int ROW_BYTES = (GROUPS * QBITS + 7) / 8;
    constexpr int SLOTS = 10;
    constexpr int OUTPUT_ROWS = 2560;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint global_row = threadgroup_position_in_grid.x * 8 + simd * ROWS_PER_SIMD;
    const uint batch = global_row / OUTPUT_ROWS;
    const uint row0 = global_row % OUTPUT_ROWS;
    float accumulated[ROWS_PER_SIMD] = {0.0f};
    for (int slot = 0; slot < SLOTS; ++slot) {
        const uint expert = experts[batch * SLOTS + slot];
        const size_t base = (size_t)expert * OUTPUT_ROWS + row0;
        const device uint8_t* weight_row =
            (const device uint8_t*)(weight + base * (K / 8)) + lane * 4;
        const device T* values = x + ((size_t)batch * SLOTS + slot) * K + lane * VALUES;
        float dot[ROWS_PER_SIMD] = {0.0f};
        for (int k0 = 0; k0 < K; k0 += BLOCK) {
            float inputs[VALUES] = {0.0f};
            float input_sum = 0.0f;
            if (k0 + lane * VALUES < K) {
                for (int i = 0; i < VALUES; i += 4) {
                    const float x0 = float(values[i]);
                    const float x1 = float(values[i + 1]);
                    const float x2 = float(values[i + 2]);
                    const float x3 = float(values[i + 3]);
                    input_sum += x0 + x1 + x2 + x3;
                    inputs[i] = x0;
                    inputs[i + 1] = x1 / 16.0f;
                    inputs[i + 2] = x2 / 256.0f;
                    inputs[i + 3] = x3 / 4096.0f;
                }
            }
            const int group = k0 / GS + lane / 8;
            if (k0 + lane * VALUES < K) {
                for (int row_offset = 0; row_offset < ROWS_PER_SIMD; ++row_offset) {
                    const size_t row = base + row_offset;
                    uint tag;
                    if (QBITS == 16 && ALIGNED16 != 0) {
                        tag = uint(((const device ushort*)tags)[row * GROUPS + group]);
                    } else {
                        const int bit = group * QBITS;
                        const size_t tag_pos = row * ROW_BYTES + (bit >> 3);
                        const uint shift = bit & 7;
                        uint window = uint(tags[tag_pos]) | (uint(tags[tag_pos + 1]) << 8);
                        if (shift + QBITS > 16) {
                            window |= uint(tags[tag_pos + 2]) << 16;
                        }
                        tag = (window >> shift) & ((1u << QBITS) - 1u);
                    }
                    const uint pair = dictionary[tag];
                    const float scale = float(as_type<T>(ushort(pair & 65535u)));
                    const float bias = float(as_type<T>(ushort(pair >> 16)));
                    const device uint16_t* words =
                        (const device uint16_t*)(weight_row + row_offset * (K / 8) * 4);
                    float quantized = 0.0f;
                    for (int i = 0; i < VALUES / 4; ++i) {
                        const uint16_t packed = words[i];
                        quantized += inputs[4 * i] * (packed & 0x000f) +
                            inputs[4 * i + 1] * (packed & 0x00f0) +
                            inputs[4 * i + 2] * (packed & 0x0f00) +
                            inputs[4 * i + 3] * (packed & 0xf000);
                    }
                    dot[row_offset] += scale * quantized + bias * input_sum;
                }
            }
            weight_row += BLOCK / 2;
            values += BLOCK;
        }
        for (int row_offset = 0; row_offset < ROWS_PER_SIMD; ++row_offset) {
            dot[row_offset] = simd_sum(dot[row_offset]);
            accumulated[row_offset] += route_weights[batch * SLOTS + slot] *
                float(T(dot[row_offset]));
        }
    }
    if (lane == 0) {
        for (int row_offset = 0; row_offset < ROWS_PER_SIMD; ++row_offset) {
            output[(size_t)batch * OUTPUT_ROWS + row0 + row_offset] =
                T(accumulated[row_offset]);
        }
    }
)metal";

inline constexpr std::string_view qmeta_decode = R"metal(
    const uint index = thread_position_in_grid.x;
    const uint group = index % uint(GROUPS);
    const uint row = index / uint(GROUPS);
    const uint bit = group * uint(QBITS);
    const size_t tag_pos = (size_t)row * uint(ROW_BYTES) + (bit >> 3);
    const uint shift = bit & 7u;
    uint window = uint(tags[tag_pos]) | (uint(tags[tag_pos + 1]) << 8);
    if (shift + uint(QBITS) > 16u) window |= uint(tags[tag_pos + 2]) << 16;
    const uint tag = (window >> shift) & ((1u << uint(QBITS)) - 1u);
    const uint pair = dictionary[tag];
    scales[index] = as_type<T>(ushort(pair & 65535u));
    biases[index] = as_type<T>(ushort(pair >> 16));
)metal";

// Portions of the Q8 kernels below are derived from mlx-serve,
// Copyright (c) 2026 David Dalcu, under the MIT license. They reproduce its gather-QMV
// accumulation and BF16 SwiGLU rounding order for the fixed Qwen3.8 MTP
// geometry. Keeping this as a separate lane avoids changing the established
// Q4 trunk trajectory while making the retained Q8 drafter numerically
// comparable to the historical 65+ tok/s implementation.
inline constexpr std::string_view gate_up_q8_exact = R"metal(
    constexpr int K = 2560;
    constexpr int N = 640;
    constexpr int GS = 64;
    constexpr int VPW = 4;
    constexpr int K_PACKED = K / VPW;
    constexpr int K_GROUPS = K / GS;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gsid = threadgroup_position_in_grid.x * 32 + sg;
    if (gsid >= 10 * N) return;
    const uint slot = gsid / N;
    const uint row = gsid % N;
    const uint expert = experts[slot];
    const size_t wbase = ((size_t)expert * N + row) * K_PACKED;
    const size_t gbase = ((size_t)expert * N + row) * K_GROUPS;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    for (int pack = int(lane); pack < K_PACKED; pack += 32) {
        uint32_t qg = gw[wbase + (size_t)pack];
        uint32_t qu = uw[wbase + (size_t)pack];
        const int k = pack * VPW;
        const int gi = k / GS;
        const float sgate = float(gs[gbase + (size_t)gi]);
        const float bgate = float(gb[gbase + (size_t)gi]);
        const float sup = float(us[gbase + (size_t)gi]);
        const float bup = float(ub[gbase + (size_t)gi]);
        const float x0 = float(x[k + 0]);
        const float x1 = float(x[k + 1]);
        const float x2 = float(x[k + 2]);
        const float x3 = float(x[k + 3]);
        g0 += x0 * (float((qg >> 0) & 255u) * sgate + bgate);
        g1 += x1 * (float((qg >> 8) & 255u) * sgate + bgate);
        g2 += x2 * (float((qg >> 16) & 255u) * sgate + bgate);
        g3 += x3 * (float((qg >> 24) & 255u) * sgate + bgate);
        u0 += x0 * (float((qu >> 0) & 255u) * sup + bup);
        u1 += x1 * (float((qu >> 8) & 255u) * sup + bup);
        u2 += x2 * (float((qu >> 16) & 255u) * sup + bup);
        u3 += x3 * (float((qu >> 24) & 255u) * sup + bup);
    }
    const float gate_acc = simd_sum((g0 + g1) + (g2 + g3));
    const float up_acc = simd_sum((u0 + u1) + (u2 + u3));
    if (lane == 0) {
        const T gt = T(gate_acc);
        const T ut = T(up_acc);
        const T sig = sigtab[as_type<ushort>(gt)];
        h[(size_t)slot * N + row] = (gt * sig) * ut;
    }
)metal";

inline constexpr std::string_view gate_up_verify_q8_exact = R"metal(
    constexpr int K = 2560;
    constexpr int N = 640;
    constexpr int GS = 64;
    constexpr int VPW = 4;
    constexpr int K_PACKED = K / VPW;
    constexpr int K_GROUPS = K / GS;
    constexpr int SLOTS = 10;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gsid = threadgroup_position_in_grid.x * 32 + sg;
    const uint batch = gsid / (SLOTS * N);
    const uint local = gsid % (SLOTS * N);
    const uint slot = local / N;
    const uint row = local % N;
    const uint expert = experts[batch * SLOTS + slot];
    const device T* xb = x + (size_t)batch * K;
    const size_t wbase = ((size_t)expert * N + row) * K_PACKED;
    const size_t gbase = ((size_t)expert * N + row) * K_GROUPS;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    for (int pack = int(lane); pack < K_PACKED; pack += 32) {
        uint32_t qg = gw[wbase + (size_t)pack];
        uint32_t qu = uw[wbase + (size_t)pack];
        const int k = pack * VPW;
        const int gi = k / GS;
        const float sgate = float(gs[gbase + (size_t)gi]);
        const float bgate = float(gb[gbase + (size_t)gi]);
        const float sup = float(us[gbase + (size_t)gi]);
        const float bup = float(ub[gbase + (size_t)gi]);
        const float x0 = float(xb[k + 0]);
        const float x1 = float(xb[k + 1]);
        const float x2 = float(xb[k + 2]);
        const float x3 = float(xb[k + 3]);
        g0 += x0 * (float((qg >> 0) & 255u) * sgate + bgate);
        g1 += x1 * (float((qg >> 8) & 255u) * sgate + bgate);
        g2 += x2 * (float((qg >> 16) & 255u) * sgate + bgate);
        g3 += x3 * (float((qg >> 24) & 255u) * sgate + bgate);
        u0 += x0 * (float((qu >> 0) & 255u) * sup + bup);
        u1 += x1 * (float((qu >> 8) & 255u) * sup + bup);
        u2 += x2 * (float((qu >> 16) & 255u) * sup + bup);
        u3 += x3 * (float((qu >> 24) & 255u) * sup + bup);
    }
    const float gate_acc = simd_sum((g0 + g1) + (g2 + g3));
    const float up_acc = simd_sum((u0 + u1) + (u2 + u3));
    if (lane == 0) {
        const T gt = T(gate_acc);
        const T ut = T(up_acc);
        const T sig = sigtab[as_type<ushort>(gt)];
        h[((size_t)batch * SLOTS + slot) * N + row] = (gt * sig) * ut;
    }
)metal";

inline constexpr std::string_view down_q8_exact = R"metal(
    constexpr int K = 640;
    constexpr int N = 2560;
    constexpr int GS = 64;
    constexpr int VPW = 4;
    constexpr int K_PACKED = K / VPW;
    constexpr int K_GROUPS = K / GS;
    constexpr int TOPK = 10;
    constexpr int ROWS = 4;
    const uint lane = thread_index_in_simdgroup;
    const uint slot = simdgroup_index_in_threadgroup;
    const uint tile = threadgroup_position_in_grid.x;
    const uint expert = experts[slot];
    threadgroup T slot_values[TOPK * ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        const uint n = tile * ROWS + r;
        const size_t wbase = ((size_t)expert * N + n) * K_PACKED;
        const size_t gbase = ((size_t)expert * N + n) * K_GROUPS;
        const size_t xbase = (size_t)slot * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int pack = int(lane); pack < K_PACKED; pack += 32) {
            const uint32_t q = dw[wbase + (size_t)pack];
            const int k = pack * VPW;
            const int gi = k / GS;
            const float scale = float(ds[gbase + (size_t)gi]);
            const float bias = float(db[gbase + (size_t)gi]);
            a0 += float(h[xbase + k + 0]) * (float((q >> 0) & 255u) * scale + bias);
            a1 += float(h[xbase + k + 1]) * (float((q >> 8) & 255u) * scale + bias);
            a2 += float(h[xbase + k + 2]) * (float((q >> 16) & 255u) * scale + bias);
            a3 += float(h[xbase + k + 3]) * (float((q >> 24) & 255u) * scale + bias);
        }
        const float acc = simd_sum((a0 + a1) + (a2 + a3));
        if (lane == 0) slot_values[slot * ROWS + r] = T(acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < ROWS) {
        const T p0 = slot_values[0 * ROWS + lane] * rw[0];
        const T p8 = slot_values[8 * ROWS + lane] * rw[8];
        const T p1 = slot_values[1 * ROWS + lane] * rw[1];
        const T p9 = slot_values[9 * ROWS + lane] * rw[9];
        T total = (p8 + p0) + (p9 + p1);
        for (uint s = 2; s < 8; ++s) total = slot_values[s * ROWS + lane] * rw[s] + total;
        y[(size_t)tile * ROWS + lane] = total;
    }
)metal";

inline constexpr std::string_view down_verify_q8_exact = R"metal(
    constexpr int K = 640;
    constexpr int N = 2560;
    constexpr int GS = 64;
    constexpr int VPW = 4;
    constexpr int K_PACKED = K / VPW;
    constexpr int K_GROUPS = K / GS;
    constexpr int TOPK = 10;
    constexpr int ROWS = 4;
    constexpr int TILES = N / ROWS;
    const uint lane = thread_index_in_simdgroup;
    const uint slot = simdgroup_index_in_threadgroup;
    const uint group = threadgroup_position_in_grid.x;
    const uint batch = group / TILES;
    const uint tile = group % TILES;
    const uint expert = experts[batch * TOPK + slot];
    threadgroup T slot_values[TOPK * ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        const uint n = tile * ROWS + r;
        const size_t wbase = ((size_t)expert * N + n) * K_PACKED;
        const size_t gbase = ((size_t)expert * N + n) * K_GROUPS;
        const size_t xbase = ((size_t)batch * TOPK + slot) * K;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int pack = int(lane); pack < K_PACKED; pack += 32) {
            const uint32_t q = dw[wbase + (size_t)pack];
            const int k = pack * VPW;
            const int gi = k / GS;
            const float scale = float(ds[gbase + (size_t)gi]);
            const float bias = float(db[gbase + (size_t)gi]);
            a0 += float(h[xbase + k + 0]) * (float((q >> 0) & 255u) * scale + bias);
            a1 += float(h[xbase + k + 1]) * (float((q >> 8) & 255u) * scale + bias);
            a2 += float(h[xbase + k + 2]) * (float((q >> 16) & 255u) * scale + bias);
            a3 += float(h[xbase + k + 3]) * (float((q >> 24) & 255u) * scale + bias);
        }
        const float acc = simd_sum((a0 + a1) + (a2 + a3));
        if (lane == 0) slot_values[slot * ROWS + r] = T(acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < ROWS) {
        const device T* scores = rw + (size_t)batch * TOPK;
        const T p0 = slot_values[0 * ROWS + lane] * scores[0];
        const T p8 = slot_values[8 * ROWS + lane] * scores[8];
        const T p1 = slot_values[1 * ROWS + lane] * scores[1];
        const T p9 = slot_values[9 * ROWS + lane] * scores[9];
        T total = (p8 + p0) + (p9 + p1);
        for (uint s = 2; s < 8; ++s) total = slot_values[s * ROWS + lane] * scores[s] + total;
        y[(size_t)batch * N + (size_t)tile * ROWS + lane] = total;
    }
)metal";

} // namespace qwen38::moe_metal
