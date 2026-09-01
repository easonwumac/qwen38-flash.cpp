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
