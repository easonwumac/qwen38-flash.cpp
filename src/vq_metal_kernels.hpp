#pragma once

#include <string_view>

namespace qwen38::vq_metal {

inline constexpr std::string_view header = R"metal(
#include <metal_stdlib>
using namespace metal;
)metal";

// Segmented-tile VQ GEMM for Flash-Next d2/K256 direct-U8, d4/K256 packed-8,
// and d8/K16384 packed-14 projections. Each route writes its FP16 result in sorted
// order; a compact follow-up kernel applies route weights and restores rows.
// The design is adapted from VQLab's checkpoint-bundled VQ runtime, distributed
// under the Qwen Community License 1.0; see NOTICE and
// licenses/qwen-community-1.0.txt. It decodes one 32-output x 64-input weight
// tile and reuses it for 32 routed rows before advancing to the next group.
inline constexpr std::string_view gemmseg_d8 = R"metal(
    const uint lane = thread_position_in_threadgroup.x;
    const uint simdgroup = thread_position_in_threadgroup.y;
    const uint tid = simdgroup * 32 + lane;
    const uint output_tile = threadgroup_position_in_grid.x;
    const uint route_tile = thread_position_in_grid.y / (OTILE / 8);

    const int expert = tmeta[route_tile * 3];
    const int route_start = tmeta[route_tile * 3 + 1];
    const int route_count = tmeta[route_tile * 3 + 2];
    const int output_start = (int)output_tile * OTILE;
    const int output_row = output_start + (int)lane;
    const int subvectors = IN / D;
    const int words_per_row = BITS == 0 ? 0 : ((subvectors + 31) / 32) * BITS;

    threadgroup half weight_tile[64][OTILE];
    threadgroup half input_tile[RTILE][64];
    threadgroup float output_tile_buffer[RTILE][OTILE];

    const int weight_row = (int)tid / 4;
    constexpr int codes_per_lane = 16 / D;
    const int code_begin = ((int)tid % 4) * codes_per_lane;
    const device uint* packed_row = (const device uint*)codes
        + ((size_t)expert * OUT + (size_t)(output_start + weight_row))
            * words_per_row;
    const device uchar* direct_row = (const device uchar*)codes
        + ((size_t)expert * OUT + (size_t)(output_start + weight_row))
            * subvectors;
    const device half* scale_row = scales
        + ((size_t)expert * OUT + (size_t)(output_start + weight_row)) * NGRP;
    const device half4* codebook4 = (const device half4*)codebook;
    const int input_row = (int)tid / 4;

    simdgroup_float8x8 accum0 = simdgroup_float8x8(0);
    simdgroup_float8x8 accum1 = simdgroup_float8x8(0);
    simdgroup_float8x8 accum2 = simdgroup_float8x8(0);
    simdgroup_float8x8 accum3 = simdgroup_float8x8(0);

    for (int group = 0; group < NGRP; ++group) {
        const int subvector_begin = group * (64 / D);
        const float scale = (float)scale_row[group];
        for (int local = code_begin; local < code_begin + codes_per_lane; ++local) {
            const int subvector = subvector_begin + local;
            uint code;
            if constexpr (BITS == 0) {
                code = (uint)direct_row[subvector];
            } else {
                const uint within = (uint)subvector & 31u;
                const uint bit_offset = within * BITS;
                const uint word = ((uint)subvector >> 5) * BITS + (bit_offset >> 5);
                const uint shift = bit_offset & 31u;
                ulong packed = (ulong)(packed_row[word] >> shift);
                if (shift + BITS > 32u)
                    packed |= (ulong)packed_row[word + 1] << (32u - shift);
                code = (uint)(packed & ((1u << BITS) - 1u));
            }
            const int column = local * D;
            if constexpr (D == 8) {
                const half4 low = codebook4[2u * code];
                const half4 high = codebook4[2u * code + 1u];
                weight_tile[column][weight_row] = (half)(scale * (float)low.x);
                weight_tile[column + 1][weight_row] = (half)(scale * (float)low.y);
                weight_tile[column + 2][weight_row] = (half)(scale * (float)low.z);
                weight_tile[column + 3][weight_row] = (half)(scale * (float)low.w);
                weight_tile[column + 4][weight_row] = (half)(scale * (float)high.x);
                weight_tile[column + 5][weight_row] = (half)(scale * (float)high.y);
                weight_tile[column + 6][weight_row] = (half)(scale * (float)high.z);
                weight_tile[column + 7][weight_row] = (half)(scale * (float)high.w);
            } else if constexpr (D == 4) {
                const half4 value = codebook4[code];
                weight_tile[column][weight_row] = (half)(scale * (float)value.x);
                weight_tile[column + 1][weight_row] = (half)(scale * (float)value.y);
                weight_tile[column + 2][weight_row] = (half)(scale * (float)value.z);
                weight_tile[column + 3][weight_row] = (half)(scale * (float)value.w);
            } else {
                const device half2* codebook2 = (const device half2*)codebook;
                const half2 value = codebook2[code];
                weight_tile[column][weight_row] = (half)(scale * (float)value.x);
                weight_tile[column + 1][weight_row] = (half)(scale * (float)value.y);
            }
        }

        if (input_row < RTILE) {
            const device half* source = nullptr;
            if (input_row < route_count)
                source = xsrc + (size_t)srcrows[route_start + input_row] * IN
                    + (size_t)group * 64;
            for (int local = code_begin; local < code_begin + codes_per_lane; ++local) {
                const int column = local * D;
                for (int element = 0; element < D; ++element)
                    input_tile[input_row][column + element] = input_row < route_count
                        ? source[column + element] : (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k8 = 0; k8 < 8; ++k8) {
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(weight_matrix,
                &weight_tile[k8 * 8][(int)simdgroup * 8], OTILE);
            simdgroup_half8x8 input_matrix;
            simdgroup_load(input_matrix, &input_tile[0][k8 * 8], 64);
            simdgroup_multiply_accumulate(accum0, input_matrix, weight_matrix, accum0);
            if constexpr (RTILE >= 16) {
                simdgroup_load(input_matrix, &input_tile[8][k8 * 8], 64);
                simdgroup_multiply_accumulate(accum1, input_matrix, weight_matrix, accum1);
            }
            if constexpr (RTILE >= 24) {
                simdgroup_load(input_matrix, &input_tile[16][k8 * 8], 64);
                simdgroup_multiply_accumulate(accum2, input_matrix, weight_matrix, accum2);
            }
            if constexpr (RTILE >= 32) {
                simdgroup_load(input_matrix, &input_tile[24][k8 * 8], 64);
                simdgroup_multiply_accumulate(accum3, input_matrix, weight_matrix, accum3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accum0, &output_tile_buffer[0][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 16)
        simdgroup_store(accum1, &output_tile_buffer[8][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 24)
        simdgroup_store(accum2, &output_tile_buffer[16][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 32)
        simdgroup_store(accum3, &output_tile_buffer[24][(int)simdgroup * 8], OTILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint threadgroup_threads =
        threads_per_threadgroup.x * threads_per_threadgroup.y;
    for (uint linear = tid; linear < (uint)(route_count * OTILE);
         linear += threadgroup_threads) {
        const int route = (int)linear / OTILE;
        const int column = (int)linear - route * OTILE;
        const int output_column = output_start + column;
        if (output_column < OUT) {
            const int sorted_route = route_start + route;
            y[(size_t)sorted_route * OUT + output_column] =
                (half)output_tile_buffer[route][column];
        }
    }
)metal";

// Gate/up twin of gemmseg_d8. Both VQ geometries share the route plan and input
// tile. The final FP16 SwiGLU follows MLX's stable sigmoid and rounding order
// inside the same dispatch, avoiding two route-sized global intermediates.
inline constexpr std::string_view gemmseg_gate_up_d8 = R"metal(
    const uint lane = thread_position_in_threadgroup.x;
    const uint simdgroup = thread_position_in_threadgroup.y;
    const uint tid = simdgroup * 32 + lane;
    const uint output_tile = threadgroup_position_in_grid.x;
    const uint route_tile = thread_position_in_grid.y / (OTILE / 8);
    const int expert = tmeta[route_tile * 3];
    const int route_start = tmeta[route_tile * 3 + 1];
    const int route_count = tmeta[route_tile * 3 + 2];
    const int output_start = (int)output_tile * OTILE;
    const int subvectors = IN / D;
    const int words_per_row = BITS == 0 ? 0 : ((subvectors + 31) / 32) * BITS;
    const int weight_row = (int)tid / 4;
    constexpr int codes_per_lane = 16 / D;
    const int code_begin = ((int)tid % 4) * codes_per_lane;
    const int input_row = (int)tid / 4;

    threadgroup half weight_tile[64][OTILE];
    threadgroup half input_tile[RTILE][64];
    threadgroup float output_buffer[RTILE][OTILE];

    const size_t row_offset =
        ((size_t)expert * OUT + (size_t)(output_start + weight_row));
    const device uint* gate_row = (const device uint*)gate_codes
        + row_offset * words_per_row;
    const device uint* up_row = (const device uint*)up_codes
        + row_offset * words_per_row;
    const device uchar* gate_direct_row =
        (const device uchar*)gate_codes + row_offset * subvectors;
    const device uchar* up_direct_row =
        (const device uchar*)up_codes + row_offset * subvectors;
    const device half* gate_scale_row = gate_scales + row_offset * NGRP;
    const device half* up_scale_row = up_scales + row_offset * NGRP;
    const device half4* gate_codebook4 = (const device half4*)gate_codebook;
    const device half4* up_codebook4 = (const device half4*)up_codebook;

    simdgroup_float8x8 gate0 = simdgroup_float8x8(0);
    simdgroup_float8x8 gate1 = simdgroup_float8x8(0);
    simdgroup_float8x8 gate2 = simdgroup_float8x8(0);
    simdgroup_float8x8 gate3 = simdgroup_float8x8(0);
    simdgroup_float8x8 up0 = simdgroup_float8x8(0);
    simdgroup_float8x8 up1 = simdgroup_float8x8(0);
    simdgroup_float8x8 up2 = simdgroup_float8x8(0);
    simdgroup_float8x8 up3 = simdgroup_float8x8(0);

    for (int group = 0; group < NGRP; ++group) {
        const int subvector_begin = group * (64 / D);
        if (input_row < RTILE) {
            const device half* source = nullptr;
            if (input_row < route_count)
                source = xsrc + (size_t)srcrows[route_start + input_row] * IN
                    + (size_t)group * 64;
            for (int local = code_begin; local < code_begin + codes_per_lane; ++local) {
                const int column = local * D;
                for (int element = 0; element < D; ++element)
                    input_tile[input_row][column + element] = input_row < route_count
                        ? source[column + element] : (half)0;
            }
        }

        const float gate_scale = (float)gate_scale_row[group];
        for (int local = code_begin; local < code_begin + codes_per_lane; ++local) {
            const int subvector = subvector_begin + local;
            uint code;
            if constexpr (BITS == 0) {
                code = (uint)gate_direct_row[subvector];
            } else {
                const uint within = (uint)subvector & 31u;
                const uint bit_offset = within * BITS;
                const uint word = ((uint)subvector >> 5) * BITS + (bit_offset >> 5);
                const uint shift = bit_offset & 31u;
                ulong packed = (ulong)(gate_row[word] >> shift);
                if (shift + BITS > 32u)
                    packed |= (ulong)gate_row[word + 1] << (32u - shift);
                code = (uint)(packed & ((1u << BITS) - 1u));
            }
            const int column = local * D;
            if constexpr (D == 8) {
                const half4 low = gate_codebook4[2u * code];
                const half4 high = gate_codebook4[2u * code + 1u];
                weight_tile[column][weight_row] = (half)(gate_scale * (float)low.x);
                weight_tile[column + 1][weight_row] = (half)(gate_scale * (float)low.y);
                weight_tile[column + 2][weight_row] = (half)(gate_scale * (float)low.z);
                weight_tile[column + 3][weight_row] = (half)(gate_scale * (float)low.w);
                weight_tile[column + 4][weight_row] = (half)(gate_scale * (float)high.x);
                weight_tile[column + 5][weight_row] = (half)(gate_scale * (float)high.y);
                weight_tile[column + 6][weight_row] = (half)(gate_scale * (float)high.z);
                weight_tile[column + 7][weight_row] = (half)(gate_scale * (float)high.w);
            } else if constexpr (D == 4) {
                const half4 value = gate_codebook4[code];
                weight_tile[column][weight_row] = (half)(gate_scale * (float)value.x);
                weight_tile[column + 1][weight_row] = (half)(gate_scale * (float)value.y);
                weight_tile[column + 2][weight_row] = (half)(gate_scale * (float)value.z);
                weight_tile[column + 3][weight_row] = (half)(gate_scale * (float)value.w);
            } else {
                const device half2* codebook2 = (const device half2*)gate_codebook;
                const half2 value = codebook2[code];
                weight_tile[column][weight_row] = (half)(gate_scale * (float)value.x);
                weight_tile[column + 1][weight_row] = (half)(gate_scale * (float)value.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k8 = 0; k8 < 8; ++k8) {
            simdgroup_half8x8 weight_matrix;
            simdgroup_half8x8 input_matrix;
            simdgroup_load(weight_matrix,
                &weight_tile[k8 * 8][(int)simdgroup * 8], OTILE);
            simdgroup_load(input_matrix, &input_tile[0][k8 * 8], 64);
            simdgroup_multiply_accumulate(gate0, input_matrix, weight_matrix, gate0);
            if constexpr (RTILE >= 16) {
                simdgroup_load(input_matrix, &input_tile[8][k8 * 8], 64);
                simdgroup_multiply_accumulate(gate1, input_matrix, weight_matrix, gate1);
            }
            if constexpr (RTILE >= 24) {
                simdgroup_load(input_matrix, &input_tile[16][k8 * 8], 64);
                simdgroup_multiply_accumulate(gate2, input_matrix, weight_matrix, gate2);
            }
            if constexpr (RTILE >= 32) {
                simdgroup_load(input_matrix, &input_tile[24][k8 * 8], 64);
                simdgroup_multiply_accumulate(gate3, input_matrix, weight_matrix, gate3);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float up_scale = (float)up_scale_row[group];
        for (int local = code_begin; local < code_begin + codes_per_lane; ++local) {
            const int subvector = subvector_begin + local;
            uint code;
            if constexpr (BITS == 0) {
                code = (uint)up_direct_row[subvector];
            } else {
                const uint within = (uint)subvector & 31u;
                const uint bit_offset = within * BITS;
                const uint word = ((uint)subvector >> 5) * BITS + (bit_offset >> 5);
                const uint shift = bit_offset & 31u;
                ulong packed = (ulong)(up_row[word] >> shift);
                if (shift + BITS > 32u)
                    packed |= (ulong)up_row[word + 1] << (32u - shift);
                code = (uint)(packed & ((1u << BITS) - 1u));
            }
            const int column = local * D;
            if constexpr (D == 8) {
                const half4 low = up_codebook4[2u * code];
                const half4 high = up_codebook4[2u * code + 1u];
                weight_tile[column][weight_row] = (half)(up_scale * (float)low.x);
                weight_tile[column + 1][weight_row] = (half)(up_scale * (float)low.y);
                weight_tile[column + 2][weight_row] = (half)(up_scale * (float)low.z);
                weight_tile[column + 3][weight_row] = (half)(up_scale * (float)low.w);
                weight_tile[column + 4][weight_row] = (half)(up_scale * (float)high.x);
                weight_tile[column + 5][weight_row] = (half)(up_scale * (float)high.y);
                weight_tile[column + 6][weight_row] = (half)(up_scale * (float)high.z);
                weight_tile[column + 7][weight_row] = (half)(up_scale * (float)high.w);
            } else if constexpr (D == 4) {
                const half4 value = up_codebook4[code];
                weight_tile[column][weight_row] = (half)(up_scale * (float)value.x);
                weight_tile[column + 1][weight_row] = (half)(up_scale * (float)value.y);
                weight_tile[column + 2][weight_row] = (half)(up_scale * (float)value.z);
                weight_tile[column + 3][weight_row] = (half)(up_scale * (float)value.w);
            } else {
                const device half2* codebook2 = (const device half2*)up_codebook;
                const half2 value = codebook2[code];
                weight_tile[column][weight_row] = (half)(up_scale * (float)value.x);
                weight_tile[column + 1][weight_row] = (half)(up_scale * (float)value.y);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int k8 = 0; k8 < 8; ++k8) {
            simdgroup_half8x8 weight_matrix;
            simdgroup_half8x8 input_matrix;
            simdgroup_load(weight_matrix,
                &weight_tile[k8 * 8][(int)simdgroup * 8], OTILE);
            simdgroup_load(input_matrix, &input_tile[0][k8 * 8], 64);
            simdgroup_multiply_accumulate(up0, input_matrix, weight_matrix, up0);
            if constexpr (RTILE >= 16) {
                simdgroup_load(input_matrix, &input_tile[8][k8 * 8], 64);
                simdgroup_multiply_accumulate(up1, input_matrix, weight_matrix, up1);
            }
            if constexpr (RTILE >= 24) {
                simdgroup_load(input_matrix, &input_tile[16][k8 * 8], 64);
                simdgroup_multiply_accumulate(up2, input_matrix, weight_matrix, up2);
            }
            if constexpr (RTILE >= 32) {
                simdgroup_load(input_matrix, &input_tile[24][k8 * 8], 64);
                simdgroup_multiply_accumulate(up3, input_matrix, weight_matrix, up3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(gate0, &output_buffer[0][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 16)
        simdgroup_store(gate1, &output_buffer[8][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 24)
        simdgroup_store(gate2, &output_buffer[16][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 32)
        simdgroup_store(gate3, &output_buffer[24][(int)simdgroup * 8], OTILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint threadgroup_threads =
        threads_per_threadgroup.x * threads_per_threadgroup.y;
    threadgroup half* gate_buffer = &input_tile[0][0];
    for (uint linear = tid; linear < (uint)(route_count * OTILE);
         linear += threadgroup_threads)
        gate_buffer[linear] = (half)output_buffer[linear / OTILE][linear % OTILE];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(up0, &output_buffer[0][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 16)
        simdgroup_store(up1, &output_buffer[8][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 24)
        simdgroup_store(up2, &output_buffer[16][(int)simdgroup * 8], OTILE);
    if constexpr (RTILE >= 32)
        simdgroup_store(up3, &output_buffer[24][(int)simdgroup * 8], OTILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = tid; linear < (uint)(route_count * OTILE);
         linear += threadgroup_threads) {
        const int route = (int)linear / OTILE;
        const int column = (int)linear - route * OTILE;
        const int output_column = output_start + column;
        if (output_column < OUT) {
            const half gate = gate_buffer[linear];
            const half up = (half)output_buffer[route][column];
            const auto y = 1 / (1 + metal::exp(metal::abs(gate)));
            const half sigmoid = half(gate < 0 ? y : 1 - y);
            const half activated = half(float(gate) * float(sigmoid));
            h[(size_t)(route_start + route) * OUT + output_column] =
                half(float(activated) * float(up));
        }
    }
)metal";

// Correctness-first decoder for one selected expert. Packed rows contain
// independent 32-code blocks, each occupying BITS uint32 words.
inline constexpr std::string_view decode_expert = R"metal(
    const uint linear = thread_position_in_grid.x;
    if (linear >= (uint)(OUT * IN)) return;
    const uint row = linear / IN;
    const uint column = linear - row * IN;
    const uint expert_id = (uint)expert;
    const uint subvector = column / D;
    uint code = 0;
    if constexpr (BITS == 0) {
        const uint nsub = IN / D;
        code = (uint)codes[((size_t)expert_id * OUT + row) * nsub + subvector];
    } else {
        const uint nsub = IN / D;
        const uint words_per_row = ((nsub + 31) / 32) * BITS;
        const device uint* packed = (const device uint*)codes;
        const device uint* code_row =
            packed + ((size_t)expert_id * OUT + row) * words_per_row;
        const uint within = subvector & 31;
        const uint bit_offset = within * BITS;
        const uint word = (subvector >> 5) * BITS + (bit_offset >> 5);
        const uint shift = bit_offset & 31;
        ulong value = (ulong)(code_row[word] >> shift);
        if (shift + BITS > 32) {
            value |= (ulong)code_row[word + 1] << (32 - shift);
        }
        code = (uint)(value & ((1ul << BITS) - 1ul));
    }
    const uint groups = IN / GROUP;
    const float scale = (float)scales[
        ((size_t)expert_id * OUT + row) * groups + column / GROUP];
    w[linear] = (T)(scale * (float)codebook[(size_t)code * D + column % D]);
)metal";

// Direct selected-expert matrix-vector product. Each SIMD group owns one
// output row and accumulates codebook dots in float before the final cast.
inline constexpr std::string_view project_expert = R"metal(
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
    if (row >= (uint)OUT) return;
    const uint expert_id = (uint)expert;
    const uint groups = IN / GROUP;
    const uint subvectors_per_group = GROUP / D;
    const uint nsub = IN / D;
    float acc = 0.0f;
    for (uint group = lane; group < groups; group += 32) {
        float group_acc = 0.0f;
        for (uint local = 0; local < subvectors_per_group; ++local) {
            const uint subvector = group * subvectors_per_group + local;
            uint code = 0;
            if constexpr (BITS == 0) {
                code = (uint)codes[
                    ((size_t)expert_id * OUT + row) * nsub + subvector];
            } else {
                const uint words_per_row = ((nsub + 31) / 32) * BITS;
                const device uint* packed = (const device uint*)codes;
                const device uint* code_row =
                    packed + ((size_t)expert_id * OUT + row) * words_per_row;
                const uint within = subvector & 31;
                const uint bit_offset = within * BITS;
                const uint word = (subvector >> 5) * BITS + (bit_offset >> 5);
                const uint shift = bit_offset & 31;
                ulong value = (ulong)(code_row[word] >> shift);
                if (shift + BITS > 32) {
                    value |= (ulong)code_row[word + 1] << (32 - shift);
                }
                code = (uint)(value & ((1ul << BITS) - 1ul));
            }
            const uint input_base = subvector * D;
            const uint code_base = code * D;
            for (uint element = 0; element < D; ++element) {
                group_acc += (float)x[input_base + element] *
                    (float)codebook[code_base + element];
            }
        }
        const float scale = (float)scales[
            ((size_t)expert_id * OUT + row) * groups + group];
        acc += scale * group_acc;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = (T)acc;
)metal";

inline constexpr std::string_view gate_up = R"metal(
    const uint lane = thread_index_in_simdgroup;
    const uint item = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
    if (item >= (uint)(BATCH * SLOTS * OUT)) return;
    const uint batch = item / (SLOTS * OUT);
    const uint local_item = item - batch * SLOTS * OUT;
    const uint slot = local_item / OUT;
    const uint row = local_item - slot * OUT;
    const uint expert = (uint)experts[batch * SLOTS + slot];
    const device T* input_row = x + (size_t)batch * IN;
    const uint groups = IN / GROUP;
    const uint spg = GROUP / D;
    const uint nsub = IN / D;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint group = lane; group < groups; group += 32) {
        float gate_group = 0.0f;
        float up_group = 0.0f;
        uint gate_words[4] = {0, 0, 0, 0};
        uint up_words[4] = {0, 0, 0, 0};
        uint cached_shift = 0;
        // Eight 14-bit d8 codes occupy exactly 112 bits.  Cache their four
        // covering words once per quant group instead of reloading overlapping
        // words for every code.
        if constexpr (BITS == 14 && D == 8 && GROUP == 64) {
            const uint wpr = ((nsub + 31) / 32) * BITS;
            const uint first_sub = group * spg;
            const uint block_bit = (first_sub & 31) * BITS;
            const uint first_word = (first_sub >> 5) * BITS + (block_bit >> 5);
            cached_shift = block_bit & 31;
            const device uint* gate_row = (const device uint*)gate_codes +
                ((size_t)expert * OUT + row) * wpr + first_word;
            const device uint* up_row = (const device uint*)up_codes +
                ((size_t)expert * OUT + row) * wpr + first_word;
            for (uint word = 0; word < 4; ++word) {
                gate_words[word] = gate_row[word];
                up_words[word] = up_row[word];
            }
        }
        for (uint local = 0; local < spg; ++local) {
            const uint sub = group * spg + local;
            uint gate_code = 0;
            uint up_code = 0;
            if constexpr (BITS == 0) {
                const size_t offset = ((size_t)expert * OUT + row) * nsub + sub;
                gate_code = (uint)gate_codes[offset];
                up_code = (uint)up_codes[offset];
            } else if constexpr (BITS == 14 && D == 8 && GROUP == 64) {
                const uint bit = cached_shift + local * BITS;
                const uint word = bit >> 5;
                const uint shift = bit & 31;
                ulong gv = (ulong)(gate_words[word] >> shift);
                ulong uv = (ulong)(up_words[word] >> shift);
                if (shift + BITS > 32) {
                    gv |= (ulong)gate_words[word + 1] << (32 - shift);
                    uv |= (ulong)up_words[word + 1] << (32 - shift);
                }
                gate_code = (uint)(gv & 0x3ffful);
                up_code = (uint)(uv & 0x3ffful);
            } else {
                const uint wpr = ((nsub + 31) / 32) * BITS;
                const uint within = sub & 31;
                const uint bit_offset = within * BITS;
                const uint word = (sub >> 5) * BITS + (bit_offset >> 5);
                const uint shift = bit_offset & 31;
                const device uint* gate_row = (const device uint*)gate_codes +
                    ((size_t)expert * OUT + row) * wpr;
                const device uint* up_row = (const device uint*)up_codes +
                    ((size_t)expert * OUT + row) * wpr;
                ulong gv = (ulong)(gate_row[word] >> shift);
                ulong uv = (ulong)(up_row[word] >> shift);
                if (shift + BITS > 32) {
                    gv |= (ulong)gate_row[word + 1] << (32 - shift);
                    uv |= (ulong)up_row[word + 1] << (32 - shift);
                }
                const ulong mask = (1ul << BITS) - 1ul;
                gate_code = (uint)(gv & mask);
                up_code = (uint)(uv & mask);
            }
            const uint xb = sub * D;
            const uint gb = gate_code * D;
            const uint ub = up_code * D;
            for (uint element = 0; element < D; ++element) {
                const float value = (float)input_row[xb + element];
                float gate_weight;
                float up_weight;
                if constexpr (GATE_CBQ != 0) {
                    gate_weight = (float)gate_codebook[gb + element] *
                        (float)gate_cb_scales[element];
                    if constexpr (GATE_CBQ == 1)
                        gate_weight += (float)gate_cb_biases[element];
                } else {
                    gate_weight = (float)gate_codebook[gb + element];
                }
                if constexpr (UP_CBQ != 0) {
                    up_weight = (float)up_codebook[ub + element] *
                        (float)up_cb_scales[element];
                    if constexpr (UP_CBQ == 1)
                        up_weight += (float)up_cb_biases[element];
                } else {
                    up_weight = (float)up_codebook[ub + element];
                }
                gate_group += value * gate_weight;
                up_group += value * up_weight;
            }
        }
        const size_t scale_offset = ((size_t)expert * OUT + row) * groups + group;
        gate_acc += (float)gate_scales[scale_offset] * gate_group;
        up_acc += (float)up_scales[scale_offset] * up_group;
    }
    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);
    if (lane == 0) {
        const float gate_value = (float)((T)gate_acc);
        const float up_value = (float)((T)up_acc);
        h[((size_t)batch * SLOTS + slot) * OUT + row] =
            (T)((float)((T)(gate_value / (1.0f + metal::exp(-gate_value)))) * up_value);
    }
)metal";

inline constexpr std::string_view down_reduce = R"metal(
    const uint lane = thread_index_in_simdgroup;
    const uint item = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
    if (item >= (uint)(BATCH * OUT)) return;
    const uint batch = item / OUT;
    const uint row = item - batch * OUT;
    const uint groups = IN / GROUP;
    const uint spg = GROUP / D;
    const uint nsub = IN / D;
    // The packed d8/d4 down projection has ten independent routed experts but
    // only ten quant groups per expert. Spread three slots across the SIMD
    // group so 30 lanes perform useful codebook work, then shuffle each
    // slot's groups back to lanes 0..9 before the original simd_sum. This
    // preserves the reduction and route-weight order exactly.
#ifndef QWEN38_VQ_TEST_SCALAR_DOWN
    if constexpr (((BITS == 14 && D == 8) || (BITS == 8 && D == 4)) &&
                  GROUP == 64 && IN == 640 && SLOTS == 10) {
        float routed = 0.0f;
        constexpr uint slots_per_wave = 3;
        for (uint slot_base = 0; slot_base < SLOTS; slot_base += slots_per_wave) {
            const uint local_slot = lane / groups;
            const uint group = lane - local_slot * groups;
            const uint slot = slot_base + local_slot;
            float scaled_group = 0.0f;
            if (local_slot < slots_per_wave && slot < SLOTS) {
                const uint expert = (uint)experts[batch * SLOTS + slot];
                const uint wpr = ((nsub + 31) / 32) * BITS;
                const uint first_sub = group * spg;
                const uint block_bit = (first_sub & 31) * BITS;
                const uint first_word =
                    (first_sub >> 5) * BITS + (block_bit >> 5);
                const uint cached_shift = block_bit & 31;
                const device uint* code_row = (const device uint*)codes +
                    ((size_t)expert * OUT + row) * wpr + first_word;
                uint packed_words[4];
                for (uint word = 0; word < 4; ++word)
                    packed_words[word] = code_row[word];
                float group_acc = 0.0f;
                for (uint local = 0; local < spg; ++local) {
                    const uint bit = cached_shift + local * BITS;
                    const uint word = bit >> 5;
                    const uint shift = bit & 31;
                    ulong value = (ulong)(packed_words[word] >> shift);
                    if (shift + BITS > 32)
                        value |= (ulong)packed_words[word + 1] << (32 - shift);
                    const uint code = (uint)(value & ((1ul << BITS) - 1ul));
                    const uint xb = (batch * SLOTS + slot) * IN +
                        (group * spg + local) * D;
                    const uint cb = code * D;
                    for (uint element = 0; element < D; ++element) {
                        float codebook_value;
                        if constexpr (CBQ != 0) {
                            codebook_value = (float)codebook[cb + element] *
                                (float)cb_scales[element];
                            if constexpr (CBQ == 1)
                                codebook_value += (float)cb_biases[element];
                        } else {
                            codebook_value = (float)codebook[cb + element];
                        }
                        group_acc += (float)x[xb + element] * codebook_value;
                    }
                }
                scaled_group = (float)scales[
                    ((size_t)expert * OUT + row) * groups + group] * group_acc;
            }
            for (uint wave_slot = 0; wave_slot < slots_per_wave; ++wave_slot) {
                const float ordered = lane < groups
                    ? simd_shuffle(scaled_group, (ushort)(wave_slot * groups + lane))
                    : 0.0f;
                const float slot_acc = simd_sum(ordered);
                const uint reduced_slot = slot_base + wave_slot;
                if (lane == 0 && reduced_slot < SLOTS) {
                    routed += route_weights[batch * SLOTS + reduced_slot] *
                        (float)((T)slot_acc);
                }
            }
        }
        if (lane == 0) {
            const size_t output_index = (size_t)batch * OUT + row;
            const T routed_value = (T)routed;
#ifdef QWEN38_VQ_ADD_SHARED
            y[output_index] = (T)((float)routed_value + (float)shared[output_index]);
#else
            y[output_index] = routed_value;
#endif
        }
        return;
    }
#endif
    float routed = 0.0f;
    for (uint slot = 0; slot < SLOTS; ++slot) {
        const uint expert = (uint)experts[batch * SLOTS + slot];
        float acc = 0.0f;
        for (uint group = lane; group < groups; group += 32) {
            float group_acc = 0.0f;
            uint packed_words[4] = {0, 0, 0, 0};
            uint cached_shift = 0;
            // See gate_up: one group is eight 14-bit d8 codes (four words).
            if constexpr (BITS == 14 && D == 8 && GROUP == 64) {
                const uint wpr = ((nsub + 31) / 32) * BITS;
                const uint first_sub = group * spg;
                const uint block_bit = (first_sub & 31) * BITS;
                const uint first_word = (first_sub >> 5) * BITS + (block_bit >> 5);
                cached_shift = block_bit & 31;
                const device uint* code_row = (const device uint*)codes +
                    ((size_t)expert * OUT + row) * wpr + first_word;
                for (uint word = 0; word < 4; ++word) {
                    packed_words[word] = code_row[word];
                }
            }
            for (uint local = 0; local < spg; ++local) {
                const uint sub = group * spg + local;
                uint code = 0;
                if constexpr (BITS == 0) {
                    code = (uint)codes[((size_t)expert * OUT + row) * nsub + sub];
                } else if constexpr (BITS == 14 && D == 8 && GROUP == 64) {
                    const uint bit = cached_shift + local * BITS;
                    const uint word = bit >> 5;
                    const uint shift = bit & 31;
                    ulong value = (ulong)(packed_words[word] >> shift);
                    if (shift + BITS > 32) {
                        value |= (ulong)packed_words[word + 1] << (32 - shift);
                    }
                    code = (uint)(value & 0x3ffful);
                } else {
                    const uint wpr = ((nsub + 31) / 32) * BITS;
                    const uint within = sub & 31;
                    const uint bit_offset = within * BITS;
                    const uint word = (sub >> 5) * BITS + (bit_offset >> 5);
                    const uint shift = bit_offset & 31;
                    const device uint* code_row = (const device uint*)codes +
                        ((size_t)expert * OUT + row) * wpr;
                    ulong value = (ulong)(code_row[word] >> shift);
                    if (shift + BITS > 32) {
                        value |= (ulong)code_row[word + 1] << (32 - shift);
                    }
                    code = (uint)(value & ((1ul << BITS) - 1ul));
                }
                const uint xb = (batch * SLOTS + slot) * IN + sub * D;
                const uint cb = code * D;
                for (uint element = 0; element < D; ++element) {
                    float codebook_value;
                    if constexpr (CBQ != 0) {
                        codebook_value = (float)codebook[cb + element] *
                            (float)cb_scales[element];
                        if constexpr (CBQ == 1)
                            codebook_value += (float)cb_biases[element];
                    } else {
                        codebook_value = (float)codebook[cb + element];
                    }
                    group_acc += (float)x[xb + element] * codebook_value;
                }
            }
            acc += (float)scales[
                ((size_t)expert * OUT + row) * groups + group] * group_acc;
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            routed += route_weights[batch * SLOTS + slot] * (float)((T)acc);
        }
    }
    if (lane == 0) {
        const size_t output_index = (size_t)batch * OUT + row;
        const T routed_value = (T)routed;
#ifdef QWEN38_VQ_ADD_SHARED
        y[output_index] = (T)((float)routed_value + (float)shared[output_index]);
#else
        y[output_index] = routed_value;
#endif
    }
)metal";

} // namespace qwen38::vq_metal
