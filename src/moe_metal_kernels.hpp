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

} // namespace qwen38::moe_metal
