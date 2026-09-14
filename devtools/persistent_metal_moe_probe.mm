#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38/mlx_backend.hpp"
#include "qwen38/model_manifest.hpp"

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
)metal";

std::uint16_t bf16(const float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
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
            Shard shard(device, manifest.directory() / shard_name);
            Shard router_shard(device, manifest.directory() / router_shard_name);

            std::vector<float> input_f32(hidden_size);
            for (int i = 0; i < hidden_size; ++i)
                input_f32[i] = static_cast<float>((i * 17 + 11) % 257 - 128) / 256.0F;
            std::vector<std::uint16_t> input_bf16(hidden_size);
            std::ranges::transform(input_f32, input_bf16.begin(), bf16);
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
            id<MTLBuffer> cache_evict = [device newBufferWithLength:64ULL * 1024 * 1024
                                                            options:MTLResourceStorageModePrivate];
            if (input == nil || experts == nil || weights == nil || hidden == nil ||
                output == nil || shared_hidden == nil || shared_router_output == nil ||
                full_output == nil || cache_evict == nil)
                throw std::runtime_error("Metal scratch allocation failed");

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
            const auto *result = static_cast<const std::uint16_t *>(full_output.contents);
            std::uint64_t hash = 1469598103934665603ULL;
            std::vector<float> direct(hidden_size);
            for (int i = 0; i < hidden_size; ++i) {
                hash ^= result[i];
                hash *= 1099511628211ULL;
                direct[i] = std::bit_cast<float>(static_cast<std::uint32_t>(result[i]) << 16);
            }
            const OracleResult oracle =
                mlx_oracle(manifest, gate, up, down, shared_gate, shared_up, shared_down,
                           shared_router, input_f32, expert_ids, route_weights);
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
