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

inline constexpr std::string_view down = R"metal(
    constexpr int K = 640;
    constexpr int GS = 64;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint d = threadgroup_position_in_grid.x * 32 + sg;
    if (d >= 2560) return;
    float acc = 0.0f;
    for (int slot = 0; slot < 10; ++slot) {
        const uint expert = experts[slot];
        const device uint32_t* wr = dw + ((size_t)expert * 2560 + d) * (K / 8);
        const device T* sr = ds + ((size_t)expert * 2560 + d) * (K / GS);
        const device T* br = db + ((size_t)expert * 2560 + d) * (K / GS);
        const device T* hv = h + slot * K;
        float dot = 0.0f;
        for (int k = lane; k < K; k += 32) {
            uint32_t word = wr[k / 8];
            float q = (float)((word >> (4 * (k % 8))) & 15);
            dot = metal::fma(
                (float)sr[k / GS] * q + (float)br[k / GS], (float)hv[k], dot);
        }
        dot = simd_sum(dot);
        acc += rw[slot] * (float)((T)dot);
    }
    if (lane == 0) y[d] = (T)acc;
)metal";

} // namespace qwen38::moe_metal
