#pragma once

// The vector-load and online-softmax strategy is informed by the Apache-2.0
// m4-prefill-engine research artifact by Mohamed Hossam Mohamed. This kernel is
// a new Qwen3.8-specific implementation: it consumes QSA-selected token indices,
// shares each KV tile across the 16 query heads of one grouped-query KV head,
// and never materializes per-query packed K/V tensors.

#include <string_view>

namespace qwen38::qsa_metal {

inline constexpr std::string_view header = R"metal(
#include <metal_stdlib>
using namespace metal;
)metal";

inline constexpr std::string_view packed_attention = R"metal(
    constexpr uint TILE = uint(TILE_SIZE);
    constexpr uint DIMS_PER_LANE = D / 32;
    constexpr uint QUERY_HEADS_PER_KV = HQ / HK;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint kv_head = threadgroup_position_in_grid.z;
    const bool active = simd < QUERY_HEADS_PER_KV;
    const uint query_head = kv_head * QUERY_HEADS_PER_KV + simd;
    const uint query_base = (query_head * uint(R) + row) * uint(D);
    float q[DIMS_PER_LANE];
    float acc[DIMS_PER_LANE];
    for (uint component = 0; component < DIMS_PER_LANE; ++component) {
        q[component] = active
            ? float(query[query_base + lane * DIMS_PER_LANE + component]) : 0.0f;
        acc[component] = 0.0f;
    }
    float running_max = -metal::numeric_limits<float>::infinity();
    float running_sum = 0.0f;

    threadgroup T shared_k[TILE * D];
    threadgroup T shared_v[TILE * D];
    for (uint tile_start = 0; tile_start < uint(S); tile_start += TILE) {
        constexpr uint TILE_VALUES = TILE * D;
        for (uint offset = tid; offset < 2 * TILE_VALUES; offset += uint(TG)) {
            const bool load_value = offset >= TILE_VALUES;
            const uint local = load_value ? offset - TILE_VALUES : offset;
            const uint slot = local / uint(D);
            const uint channel = local % uint(D);
            const uint selected_slot = tile_start + slot;
            T loaded = T(0);
            if (selected_slot < uint(S) && valid[row * uint(S) + selected_slot]) {
                const uint token = uint(indices[row * uint(S) + selected_slot]);
                const uint source = (kv_head * uint(TOTAL) + token) * uint(D) + channel;
                loaded = load_value ? values[source] : keys[source];
            }
            if (load_value) shared_v[local] = loaded;
            else shared_k[local] = loaded;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_count = min(TILE, uint(S) - tile_start);
        for (uint slot = 0; slot < tile_count; ++slot) {
            const uint selected_slot = tile_start + slot;
            if (!active || !valid[row * uint(S) + selected_slot]) continue;
            float partial = 0.0f;
            const uint shared_base = slot * uint(D) + lane * DIMS_PER_LANE;
            for (uint component = 0; component < DIMS_PER_LANE; ++component) {
                partial += q[component] * float(shared_k[shared_base + component]);
            }
            const float score = simd_sum(partial) * float(scale);
            const float next_max = max(running_max, score);
            const float alpha = isfinite(running_max)
                ? metal::exp(running_max - next_max) : 0.0f;
            const float probability = metal::exp(score - next_max);
            running_sum = running_sum * alpha + probability;
            for (uint component = 0; component < DIMS_PER_LANE; ++component) {
                acc[component] = acc[component] * alpha +
                    probability * float(shared_v[shared_base + component]);
            }
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        const uint output_base = (row * uint(HQ) + query_head) * uint(D);
        for (uint component = 0; component < DIMS_PER_LANE; ++component) {
            output[output_base + lane * DIMS_PER_LANE + component] =
                T(acc[component] * inverse_sum);
        }
    }
)metal";

// Affine Q8 K/V variant. Q8 words pack four channels along the last axis;
// scale and bias are stored once per 64-channel group. Only QSA-selected rows
// are decoded into threadgroup memory, so no full BF16 cache is reconstructed.
inline constexpr std::string_view packed_attention_q8 = R"metal(
    constexpr uint TILE = uint(TILE_SIZE);
    constexpr uint DIMS_PER_LANE = D / 32;
    constexpr uint QUERY_HEADS_PER_KV = HQ / HK;
    constexpr uint PACKED_D = D / 4;
    constexpr uint GROUPS = D / 64;
    const uint total_count = uint(total);
    const uint cold_count = uint(cold);
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint kv_head = threadgroup_position_in_grid.z;
    const bool active = simd < QUERY_HEADS_PER_KV;
    const uint query_head = kv_head * QUERY_HEADS_PER_KV + simd;
    const uint query_base = (query_head * uint(R) + row) * uint(D);
    float q[DIMS_PER_LANE];
    float acc[DIMS_PER_LANE];
    for (uint component = 0; component < DIMS_PER_LANE; ++component) {
        q[component] = active
            ? float(query[query_base + lane * DIMS_PER_LANE + component]) : 0.0f;
        acc[component] = 0.0f;
    }
    float running_max = -metal::numeric_limits<float>::infinity();
    float running_sum = 0.0f;

    threadgroup T shared_k[TILE * D];
    threadgroup T shared_v[TILE * D];
    for (uint tile_start = 0; tile_start < uint(S); tile_start += TILE) {
        constexpr uint TILE_WORDS = TILE * PACKED_D;
        for (uint offset = tid; offset < 2 * TILE_WORDS; offset += uint(TG)) {
            const bool load_value = offset >= TILE_WORDS;
            const uint local = load_value ? offset - TILE_WORDS : offset;
            const uint slot = local / PACKED_D;
            const uint packed_channel = local % PACKED_D;
            const uint selected_slot = tile_start + slot;
            const bool selected_valid = selected_slot < uint(S) &&
                valid[row * uint(S) + selected_slot];
            uint quantized = 0;
            float s = 0.0f;
            float b = 0.0f;
            uint token = 0;
            bool cold = false;
            if (selected_valid) {
                token = uint(indices[row * uint(S) + selected_slot]);
                cold = token < cold_count;
            }
            if (cold) {
                const size_t vector = size_t(kv_head) * size_t(cold_count) + token;
                const size_t word_index = vector * PACKED_D + packed_channel;
                quantized = load_value ? vw[word_index] : kw[word_index];
                const size_t group_index = vector * GROUPS + packed_channel / 16;
                s = float(load_value ? vs[group_index] : ks[group_index]);
                b = float(load_value ? vb[group_index] : kb[group_index]);
            }
            const uint shared_base = slot * uint(D) + packed_channel * 4;
            for (uint component = 0; component < 4; ++component) {
                T loaded = T(0);
                if (cold) {
                    loaded = T(float((quantized >> (component * 8)) & 255u) * s + b);
                } else if (selected_valid) {
                    const size_t hot_token = size_t(token - cold_count);
                    const size_t hot_vector = size_t(kv_head) *
                        size_t(total_count - cold_count) + hot_token;
                    const size_t hot_index = hot_vector * size_t(D) +
                        packed_channel * 4 + component;
                    loaded = load_value ? hot_values[hot_index] : hot_keys[hot_index];
                }
                if (load_value) shared_v[shared_base + component] = loaded;
                else shared_k[shared_base + component] = loaded;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_count = min(TILE, uint(S) - tile_start);
        for (uint slot = 0; slot < tile_count; ++slot) {
            const uint selected_slot = tile_start + slot;
            if (!active || !valid[row * uint(S) + selected_slot]) continue;
            float partial = 0.0f;
            const uint shared_base = slot * uint(D) + lane * DIMS_PER_LANE;
            for (uint component = 0; component < DIMS_PER_LANE; ++component) {
                partial += q[component] * float(shared_k[shared_base + component]);
            }
            const float score = simd_sum(partial) * float(scale);
            const float next_max = max(running_max, score);
            const float alpha = isfinite(running_max)
                ? metal::exp(running_max - next_max) : 0.0f;
            const float probability = metal::exp(score - next_max);
            running_sum = running_sum * alpha + probability;
            for (uint component = 0; component < DIMS_PER_LANE; ++component) {
                acc[component] = acc[component] * alpha +
                    probability * float(shared_v[shared_base + component]);
            }
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        const uint output_base = (row * uint(HQ) + query_head) * uint(D);
        for (uint component = 0; component < DIMS_PER_LANE; ++component) {
            output[output_base + lane * DIMS_PER_LANE + component] =
                T(acc[component] * inverse_sum);
        }
    }
)metal";

inline constexpr std::string_view sample_queries = R"metal(
    const uint index = thread_position_in_grid.x;
    if (index >= uint(H * GROUPS * D)) return;
    const uint channel = index % uint(D);
    const uint group = (index / uint(D)) % uint(GROUPS);
    const uint head = index / (uint(D) * uint(GROUPS));
    const uint source = (head * uint(ROWS) + group * uint(STRIDE)) * uint(D) + channel;
    output[index] = query[source];
)metal";

inline constexpr std::string_view expand_block_indices = R"metal(
    const uint index = thread_position_in_grid.x;
    if (index >= uint(ROWS * TOPK)) return;
    const uint row = index / uint(TOPK);
    const uint slot = index % uint(TOPK);
    output[index] = block_indices[(row / uint(STRIDE)) * uint(TOPK) + slot];
)metal";

} // namespace qwen38::qsa_metal
