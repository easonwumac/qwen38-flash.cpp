#pragma once

// The MPP matmul structure below is adapted from Cider's
// cider/kernels/pergroup_int8_gemm.metal at commit
// 4d91fcee9439f7aea17ae6e965271d9536c604a0.
// Copyright (c) 2025 Cider Contributors
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <string_view>

namespace qwen38::direct_q4_metal {

inline constexpr std::string_view quantize = R"metal(
    const uint block = threadgroup_position_in_grid.x;
    const uint row = block / G;
    const uint group = block - row * G;
    const uint lid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float maxima[2];
    threadgroup float sums[2];
    threadgroup float inverse_scale;
    const uint index = (size_t)row * K + group * 64 + lid;
    const float value = float(x[index]);
    float maximum = simd_max(abs(value));
    float sum = simd_sum(value);
    if (lane == 0) { maxima[sg] = maximum; sums[sg] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        maximum = simd_max(lane < 2 ? maxima[lane] : 0.0f);
        sum = simd_sum(lane < 2 ? sums[lane] : 0.0f);
        if (lane == 0) {
            const float s = maximum == 0.0f ? 1.0f : maximum / 127.0f;
            scale[(size_t)row * G + group] = s;
            xsum[(size_t)row * G + group] = sum;
            inverse_scale = 1.0f / s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    aq[index] = int8_t(clamp(round(value * inverse_scale), -128.0f, 127.0f));
)metal";

inline constexpr std::string_view unpack = R"metal(
    const uint index = thread_position_in_grid.x;
    if (index < COUNT) {
        const uint word = packed[index >> 3];
        w[index] = int8_t((word >> ((index & 7) * 4)) & 15);
    }
)metal";

inline constexpr std::string_view header = R"metal(
template <int BM, int WM>
void direct_q4_affine_impl(
    const device int8_t *A, const device uint *B, device half *C,
    uint M, uint N, uint K, const device float *scale_a,
    const device float *xsum, const device bfloat16_t *scale_w,
    const device bfloat16_t *bias_w, uint tiles_m, uint tiles_n,
    uint2 tgid, uint sgid, uint lid) {
  constexpr int WN = 4, SM = BM / WM, SN = 32, TM = SM / 16, TN = 2, TK = 2;
  const uint tile_n = tgid.x, tile_m = tgid.y;
  if (tile_n >= tiles_n || tile_m >= tiles_m) return;
  const short2 sc = nax_get_coord(ushort(lid));
  const uint m_base = tile_m * BM + (sgid / WN) * SM;
  const uint n_base = tile_n * 128 + (sgid % WN) * SN;
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto cb = op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto cc = op.get_destination_cooperative_tensor<decltype(ca), decltype(cb), int32_t>();
  float acc[TM * TN][kElemsPerFrag];
  for (int f = 0; f < TM * TN; ++f) for (int i = 0; i < kElemsPerFrag; ++i) acc[f][i] = 0.0f;
  const uint groups = K / 64;
  for (uint g = 0; g < groups; ++g) {
    int c[TM * TN][kElemsPerFrag];
    for (int f = 0; f < TM * TN; ++f) for (int i = 0; i < kElemsPerFrag; ++i) c[f][i] = 0;
    for (int kk0 = 0; kk0 < 64; kk0 += 32) {
      int8_t af[TM][TK][kElemsPerFrag];
      int8_t bf[TN][TK][kElemsPerFrag];
      for (short mm = 0; mm < TM; ++mm) for (short kk = 0; kk < TK; ++kk)
        nax_frag_load(af[mm][kk], A + (size_t)m_base * K + g * 64 + kk0,
                      int(K), sc, short(mm * 16), short(kk * 16));
      for (short nn = 0; nn < TN; ++nn) for (short kk = 0; kk < TK; ++kk) {
        for (short i = 0; i < 2; ++i) for (short j = 0; j < kElemCols; ++j) {
          const uint ni = n_base + sc.y + nn * 16 + i * kElemRowsJump;
          const uint ki = g * 64 + kk0 + sc.x + kk * 16 + j;
          const uint word = B[(size_t)ni * (K / 8) + ki / 8];
          bf[nn][kk][i * kElemCols + j] = int8_t((word >> ((ki & 7) * 4)) & 15);
        }
      }
      for (short mm = 0; mm < TM; ++mm) for (short nn = 0; nn < TN; nn += 2)
        for (short kk = 0; kk < TK; ++kk) {
          for (short i = 0; i < kElemsPerFrag; ++i) { ca[i] = af[mm][kk][i]; cb[i] = bf[nn][kk][i]; cb[kElemsPerFrag+i] = bf[nn+1][kk][i]; }
          const short c0 = mm * TN + nn, c1 = c0 + 1;
          for (short i = 0; i < kElemsPerFrag; ++i) { cc[i] = c[c0][i]; cc[kElemsPerFrag+i] = c[c1][i]; }
          op.run(ca, cb, cc);
          for (short i = 0; i < kElemsPerFrag; ++i) { c[c0][i] = cc[i]; c[c1][i] = cc[kElemsPerFrag+i]; }
        }
    }
    for (short mm = 0; mm < TM; ++mm) for (short nn = 0; nn < TN; ++nn)
      for (short i = 0; i < 2; ++i) for (short j = 0; j < kElemCols; ++j) {
        const uint mi = m_base + sc.y + mm * 16 + i * kElemRowsJump;
        const uint ni = n_base + sc.x + nn * 16 + j;
        const uint index = i * kElemCols + j;
        acc[mm * TN + nn][index] += float(c[mm * TN + nn][index]) *
            scale_a[(size_t)mi * groups + g] * float(scale_w[(size_t)ni * groups + g]) +
            xsum[(size_t)mi * groups + g] * float(bias_w[(size_t)ni * groups + g]);
      }
  }
  device half *D = C + (size_t)m_base * N + n_base;
  for (short mm = 0; mm < TM; ++mm) for (short nn = 0; nn < TN; ++nn)
    for (short i = 0; i < 2; ++i) for (short j = 0; j < kElemCols; ++j) {
      const uint mi = m_base + sc.y + mm * 16 + i * kElemRowsJump;
      const uint ni = n_base + sc.x + nn * 16 + j;
      if (mi < M && ni < N) D[(sc.y + mm*16 + i*kElemRowsJump)*N + sc.x + nn*16 + j] = half(acc[mm*TN+nn][i*kElemCols+j]);
    }
}
)metal";

inline constexpr std::string_view fused = R"metal(
    direct_q4_affine_impl<BM, WM>(aq, packed, y, M, N, K, scale, xsum,
        wscale, wbias, TILES_M, TILES_N, threadgroup_position_in_grid.xy,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
)metal";

} // namespace qwen38::direct_q4_metal
