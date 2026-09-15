#pragma once

#include <string_view>

namespace qwen38::vq_metal {

inline constexpr std::string_view header = R"metal(
#include <metal_stdlib>
using namespace metal;
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
    if (item >= (uint)(SLOTS * OUT)) return;
    const uint slot = item / OUT;
    const uint row = item - slot * OUT;
    const uint expert = (uint)experts[slot];
    const uint groups = IN / GROUP;
    const uint spg = GROUP / D;
    const uint nsub = IN / D;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint group = lane; group < groups; group += 32) {
        float gate_group = 0.0f;
        float up_group = 0.0f;
        for (uint local = 0; local < spg; ++local) {
            const uint sub = group * spg + local;
            uint gate_code = 0;
            uint up_code = 0;
            if constexpr (BITS == 0) {
                const size_t offset = ((size_t)expert * OUT + row) * nsub + sub;
                gate_code = (uint)gate_codes[offset];
                up_code = (uint)up_codes[offset];
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
                const float value = (float)x[xb + element];
                gate_group += value * (float)gate_codebook[gb + element];
                up_group += value * (float)up_codebook[ub + element];
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
        h[(size_t)slot * OUT + row] =
            (T)((float)((T)(gate_value / (1.0f + metal::exp(-gate_value)))) * up_value);
    }
)metal";

inline constexpr std::string_view down_reduce = R"metal(
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
    if (row >= (uint)OUT) return;
    const uint groups = IN / GROUP;
    const uint spg = GROUP / D;
    const uint nsub = IN / D;
    float routed = 0.0f;
    for (uint slot = 0; slot < SLOTS; ++slot) {
        const uint expert = (uint)experts[slot];
        float acc = 0.0f;
        for (uint group = lane; group < groups; group += 32) {
            float group_acc = 0.0f;
            for (uint local = 0; local < spg; ++local) {
                const uint sub = group * spg + local;
                uint code = 0;
                if constexpr (BITS == 0) {
                    code = (uint)codes[((size_t)expert * OUT + row) * nsub + sub];
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
                const uint xb = slot * IN + sub * D;
                const uint cb = code * D;
                for (uint element = 0; element < D; ++element) {
                    group_acc += (float)x[xb + element] * (float)codebook[cb + element];
                }
            }
            acc += (float)scales[
                ((size_t)expert * OUT + row) * groups + group] * group_acc;
        }
        acc = simd_sum(acc);
        if (lane == 0) routed += route_weights[slot] * (float)((T)acc);
    }
    if (lane == 0) y[row] = (T)routed;
)metal";

} // namespace qwen38::vq_metal
