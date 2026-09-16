#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38/decoder_layer.hpp"
#include "qwen38/gated_delta_net.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model_manifest.hpp"
#include "qwen38/sparse_moe.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../src/persistent_metal_kernels.hpp"

namespace {

using qwen38::MlxArray;
constexpr int hidden_size = 2560;
constexpr int expert_width = 448;
constexpr int expert_count = 512;
constexpr int top_k = 10;
constexpr int group_size = 64;

using qwen38::persistent_metal::metal_source_prefix;
using qwen38::persistent_metal::metal_source_suffix;

std::uint16_t bf16(const float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(const std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
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

void bind_tensor(id<MTLComputeCommandEncoder> encoder, const Shard &shard, const std::string &name,
                 const NSUInteger index) {
    [encoder setBuffer:shard.buffer offset:shard.offset(name) atIndex:index];
}

double run_hc_read(id<MTLCommandQueue> queue, id<MTLComputePipelineState> normalize_state,
                   id<MTLComputePipelineState> down_state, id<MTLComputePipelineState> up_state,
                   id<MTLBuffer> stream, const Shard &shard, const std::string &norm,
                   const ProjectionNames &down, const ProjectionNames &up,
                   const ProjectionNames &injection, id<MTLBuffer> normalized,
                   id<MTLBuffer> activation, id<MTLBuffer> injection_output, id<MTLBuffer> mixed) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> normalize_encoder = [command computeCommandEncoder];
    [normalize_encoder setComputePipelineState:normalize_state];
    [normalize_encoder setBuffer:stream offset:0 atIndex:0];
    bind_tensor(normalize_encoder, shard, norm, 1);
    [normalize_encoder setBuffer:normalized offset:0 atIndex:2];
    [normalize_encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [normalize_encoder endEncoding];
    id<MTLComputeCommandEncoder> down_encoder = [command computeCommandEncoder];
    [down_encoder setComputePipelineState:down_state];
    [down_encoder setBuffer:normalized offset:0 atIndex:0];
    bind_projection(down_encoder, shard, down, 1);
    bind_projection(down_encoder, shard, injection, 4);
    [down_encoder setBuffer:activation offset:0 atIndex:7];
    [down_encoder setBuffer:injection_output offset:0 atIndex:8];
    [down_encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [down_encoder endEncoding];
    id<MTLComputeCommandEncoder> up_encoder = [command computeCommandEncoder];
    [up_encoder setComputePipelineState:up_state];
    [up_encoder setBuffer:normalized offset:0 atIndex:0];
    [up_encoder setBuffer:activation offset:0 atIndex:1];
    bind_projection(up_encoder, shard, up, 2);
    [up_encoder setBuffer:mixed offset:0 atIndex:5];
    [up_encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [up_encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_qsa_selector(id<MTLCommandQueue> queue, id<MTLComputePipelineState> score_state,
                        id<MTLComputePipelineState> first_state,
                        id<MTLComputePipelineState> merge_state, id<MTLBuffer> query,
                        id<MTLBuffer> pooled, id<MTLBuffer> scores, id<MTLBuffer> temp_scores_a,
                        id<MTLBuffer> temp_ids_a, id<MTLBuffer> temp_scores_b,
                        id<MTLBuffer> temp_ids_b, id<MTLBuffer> selected,
                        const std::uint32_t block_count) {
    const std::uint32_t padded_count = (block_count + 255) / 256 * 256;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:score_state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:pooled offset:0 atIndex:1];
    [encoder setBuffer:scores offset:0 atIndex:2];
    [encoder setBytes:&block_count length:sizeof(block_count) atIndex:3];
    [encoder setBytes:&padded_count length:sizeof(padded_count) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(padded_count / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:first_state];
    [encoder setBuffer:scores offset:0 atIndex:0];
    [encoder setBuffer:temp_scores_a offset:0 atIndex:1];
    [encoder setBuffer:temp_ids_a offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(padded_count / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    std::uint32_t input_groups = padded_count / 256;
    bool source_a = true;
    while (input_groups > 1) {
        const std::uint32_t output_groups = (input_groups + 1) / 2;
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:merge_state];
        [encoder setBuffer:(source_a ? temp_scores_a : temp_scores_b) offset:0 atIndex:0];
        [encoder setBuffer:(source_a ? temp_ids_a : temp_ids_b) offset:0 atIndex:1];
        [encoder setBuffer:(source_a ? temp_scores_b : temp_scores_a) offset:0 atIndex:2];
        [encoder setBuffer:(output_groups == 1 ? selected : (source_a ? temp_ids_b : temp_ids_a))
                    offset:0
                   atIndex:3];
        [encoder setBytes:&input_groups length:sizeof(input_groups) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(output_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        input_groups = output_groups;
        source_a = !source_a;
    }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_qsa_attention(id<MTLCommandQueue> queue, id<MTLComputePipelineState> attention_state,
                         id<MTLBuffer> query, id<MTLBuffer> key_weight, id<MTLBuffer> key_scale,
                         id<MTLBuffer> key_bias, id<MTLBuffer> value_weight,
                         id<MTLBuffer> value_scale, id<MTLBuffer> value_bias,
                         id<MTLBuffer> selected, id<MTLBuffer> output,
                         const std::uint32_t token_count) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:attention_state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key_weight offset:0 atIndex:1];
    [encoder setBuffer:key_scale offset:0 atIndex:2];
    [encoder setBuffer:key_bias offset:0 atIndex:3];
    [encoder setBuffer:value_weight offset:0 atIndex:4];
    [encoder setBuffer:value_scale offset:0 atIndex:5];
    [encoder setBuffer:value_bias offset:0 atIndex:6];
    [encoder setBuffer:selected offset:0 atIndex:7];
    [encoder setBuffer:output offset:0 atIndex:8];
    [encoder setBytes:&token_count length:sizeof(token_count) atIndex:9];
    [encoder setBuffer:query offset:0 atIndex:10];
    [encoder setBuffer:query offset:0 atIndex:11];
    const std::uint32_t hot_count = 0, hot_capacity = 1;
    const std::uint32_t selected_count = 512;
    [encoder setBytes:&hot_count length:sizeof(hot_count) atIndex:12];
    [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:13];
    [encoder setBytes:&selected_count length:sizeof(selected_count) atIndex:14];
    [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(384, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_qsa_selected_attention(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> score_state,
    id<MTLComputePipelineState> first_state, id<MTLComputePipelineState> merge_state,
    id<MTLComputePipelineState> attention_state, id<MTLBuffer> selector_query, id<MTLBuffer> pooled,
    id<MTLBuffer> scores, id<MTLBuffer> temp_scores_a, id<MTLBuffer> temp_ids_a,
    id<MTLBuffer> temp_scores_b, id<MTLBuffer> temp_ids_b, id<MTLBuffer> selected,
    id<MTLBuffer> attention_query, id<MTLBuffer> key_weight, id<MTLBuffer> key_scale,
    id<MTLBuffer> key_bias, id<MTLBuffer> value_weight, id<MTLBuffer> value_scale,
    id<MTLBuffer> value_bias, id<MTLBuffer> output, const std::uint32_t block_count,
    const std::uint32_t token_count) {
    const std::uint32_t padded_count = (block_count + 255) / 256 * 256;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:score_state];
    [encoder setBuffer:selector_query offset:0 atIndex:0];
    [encoder setBuffer:pooled offset:0 atIndex:1];
    [encoder setBuffer:scores offset:0 atIndex:2];
    [encoder setBytes:&block_count length:sizeof(block_count) atIndex:3];
    [encoder setBytes:&padded_count length:sizeof(padded_count) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(padded_count / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:first_state];
    [encoder setBuffer:scores offset:0 atIndex:0];
    [encoder setBuffer:temp_scores_a offset:0 atIndex:1];
    [encoder setBuffer:temp_ids_a offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(padded_count / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    std::uint32_t input_groups = padded_count / 256;
    bool source_a = true;
    while (input_groups > 1) {
        const std::uint32_t output_groups = (input_groups + 1) / 2;
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:merge_state];
        [encoder setBuffer:(source_a ? temp_scores_a : temp_scores_b) offset:0 atIndex:0];
        [encoder setBuffer:(source_a ? temp_ids_a : temp_ids_b) offset:0 atIndex:1];
        [encoder setBuffer:(source_a ? temp_scores_b : temp_scores_a) offset:0 atIndex:2];
        [encoder setBuffer:(output_groups == 1 ? selected : (source_a ? temp_ids_b : temp_ids_a))
                    offset:0
                   atIndex:3];
        [encoder setBytes:&input_groups length:sizeof(input_groups) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(output_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        input_groups = output_groups;
        source_a = !source_a;
    }
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:attention_state];
    [encoder setBuffer:attention_query offset:0 atIndex:0];
    [encoder setBuffer:key_weight offset:0 atIndex:1];
    [encoder setBuffer:key_scale offset:0 atIndex:2];
    [encoder setBuffer:key_bias offset:0 atIndex:3];
    [encoder setBuffer:value_weight offset:0 atIndex:4];
    [encoder setBuffer:value_scale offset:0 atIndex:5];
    [encoder setBuffer:value_bias offset:0 atIndex:6];
    [encoder setBuffer:selected offset:0 atIndex:7];
    [encoder setBuffer:output offset:0 atIndex:8];
    [encoder setBytes:&token_count length:sizeof(token_count) atIndex:9];
    [encoder setBuffer:attention_query offset:0 atIndex:10];
    [encoder setBuffer:attention_query offset:0 atIndex:11];
    const std::uint32_t hot_count = 0, hot_capacity = 1;
    const std::uint32_t selected_count = 512;
    [encoder setBytes:&hot_count length:sizeof(hot_count) atIndex:12];
    [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:13];
    [encoder setBytes:&selected_count length:sizeof(selected_count) atIndex:14];
    [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(384, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_attention_projections(id<MTLCommandQueue> queue, id<MTLComputePipelineState> state,
                                 id<MTLBuffer> input, const Shard &shard,
                                 const ProjectionNames &index, const ProjectionNames &query,
                                 const ProjectionNames &key, const ProjectionNames &value,
                                 id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(encoder, shard, index, 1);
    bind_projection(encoder, shard, query, 4);
    bind_projection(encoder, shard, key, 7);
    bind_projection(encoder, shard, value, 10);
    [encoder setBuffer:output offset:0 atIndex:13];
    [encoder dispatchThreadgroups:MTLSizeMake((13952 + 3) / 4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_attention_normalize(id<MTLCommandQueue> queue, id<MTLComputePipelineState> state,
                               id<MTLBuffer> projected, const Shard &shard,
                               const std::string &query_norm, const std::string &key_norm,
                               const std::string &index_norm, id<MTLBuffer> rope_cos,
                               id<MTLBuffer> rope_sin, id<MTLBuffer> selector_query,
                               id<MTLBuffer> attention_query, id<MTLBuffer> attention_key) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:projected offset:0 atIndex:0];
    bind_tensor(encoder, shard, query_norm, 1);
    bind_tensor(encoder, shard, key_norm, 2);
    bind_tensor(encoder, shard, index_norm, 3);
    [encoder setBuffer:rope_cos offset:0 atIndex:4];
    [encoder setBuffer:rope_sin offset:0 atIndex:5];
    [encoder setBuffer:selector_query offset:0 atIndex:6];
    [encoder setBuffer:attention_query offset:0 atIndex:7];
    [encoder setBuffer:attention_key offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(30, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

double run_attention_output(id<MTLCommandQueue> queue, id<MTLComputePipelineState> gate_state,
                            id<MTLComputePipelineState> projection_state, id<MTLBuffer> attended,
                            id<MTLBuffer> projected, const Shard &shard,
                            const ProjectionNames &projection, id<MTLBuffer> gated,
                            id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:gate_state];
    [encoder setBuffer:attended offset:0 atIndex:0];
    [encoder setBuffer:projected offset:0 atIndex:1];
    [encoder setBuffer:gated offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(6144, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:projection_state];
    [encoder setBuffer:gated offset:0 atIndex:0];
    bind_projection(encoder, shard, projection, 1);
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
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

double run_device_routed_full(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> router_state,
    id<MTLComputePipelineState> select_state, id<MTLComputePipelineState> gate_state,
    id<MTLComputePipelineState> down_state, id<MTLBuffer> input, const Shard &expert_shard,
    const Shard &router_shard, const ProjectionNames &device_router,
    const ProjectionNames &routed_gate, const ProjectionNames &routed_up,
    const ProjectionNames &routed_down, const ProjectionNames &shared_gate,
    const ProjectionNames &shared_up, const ProjectionNames &shared_down,
    const ProjectionNames &shared_router, id<MTLBuffer> logits, id<MTLBuffer> experts,
    id<MTLBuffer> weights, id<MTLBuffer> routed_hidden, id<MTLBuffer> shared_hidden,
    id<MTLBuffer> router_output, id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> router_encoder = [command computeCommandEncoder];
    [router_encoder setComputePipelineState:router_state];
    [router_encoder setBuffer:input offset:0 atIndex:0];
    bind_projection(router_encoder, router_shard, device_router, 1);
    [router_encoder setBuffer:logits offset:0 atIndex:4];
    [router_encoder dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [router_encoder endEncoding];
    id<MTLComputeCommandEncoder> select_encoder = [command computeCommandEncoder];
    [select_encoder setComputePipelineState:select_state];
    [select_encoder setBuffer:logits offset:0 atIndex:0];
    [select_encoder setBuffer:experts offset:0 atIndex:1];
    [select_encoder setBuffer:weights offset:0 atIndex:2];
    [select_encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [select_encoder endEncoding];
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

double run_hc_device_routed_full(
    id<MTLCommandQueue> queue, id<MTLComputePipelineState> normalize_state,
    id<MTLComputePipelineState> hc_down_state, id<MTLComputePipelineState> hc_up_state,
    id<MTLComputePipelineState> router_state, id<MTLComputePipelineState> select_state,
    id<MTLComputePipelineState> gate_state, id<MTLComputePipelineState> moe_down_state,
    id<MTLComputePipelineState> healing_left_state, id<MTLComputePipelineState> healing_right_state,
    id<MTLBuffer> stream, const Shard &expert_shard, const Shard &router_shard,
    const Shard &healing_shard, const std::string &norm, const ProjectionNames &hc_down,
    const ProjectionNames &hc_up, const ProjectionNames &injection,
    const ProjectionNames &device_router, const ProjectionNames &routed_gate,
    const ProjectionNames &routed_up, const ProjectionNames &routed_down,
    const ProjectionNames &shared_gate, const ProjectionNames &shared_up,
    const ProjectionNames &shared_down, const ProjectionNames &shared_router,
    const std::string &healing_left, const std::string &healing_right, id<MTLBuffer> normalized,
    id<MTLBuffer> activation, id<MTLBuffer> injection_output, id<MTLBuffer> mixed,
    id<MTLBuffer> logits, id<MTLBuffer> experts, id<MTLBuffer> weights, id<MTLBuffer> routed_hidden,
    id<MTLBuffer> shared_hidden, id<MTLBuffer> router_output, id<MTLBuffer> block_output,
    id<MTLBuffer> healing_hidden, id<MTLBuffer> output_stream) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:normalize_state];
    [encoder setBuffer:stream offset:0 atIndex:0];
    bind_tensor(encoder, router_shard, norm, 1);
    [encoder setBuffer:normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:hc_down_state];
    [encoder setBuffer:normalized offset:0 atIndex:0];
    bind_projection(encoder, router_shard, hc_down, 1);
    bind_projection(encoder, router_shard, injection, 4);
    [encoder setBuffer:activation offset:0 atIndex:7];
    [encoder setBuffer:injection_output offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:hc_up_state];
    [encoder setBuffer:normalized offset:0 atIndex:0];
    [encoder setBuffer:activation offset:0 atIndex:1];
    bind_projection(encoder, router_shard, hc_up, 2);
    [encoder setBuffer:mixed offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:router_state];
    [encoder setBuffer:mixed offset:0 atIndex:0];
    bind_projection(encoder, router_shard, device_router, 1);
    [encoder setBuffer:logits offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(128, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:select_state];
    [encoder setBuffer:logits offset:0 atIndex:0];
    [encoder setBuffer:experts offset:0 atIndex:1];
    [encoder setBuffer:weights offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:gate_state];
    [encoder setBuffer:mixed offset:0 atIndex:0];
    bind_projection(encoder, expert_shard, routed_gate, 1);
    bind_projection(encoder, expert_shard, routed_up, 4);
    [encoder setBuffer:experts offset:0 atIndex:7];
    [encoder setBuffer:routed_hidden offset:0 atIndex:8];
    bind_projection(encoder, expert_shard, shared_gate, 9);
    bind_projection(encoder, expert_shard, shared_up, 12);
    [encoder setBuffer:shared_hidden offset:0 atIndex:15];
    bind_projection(encoder, router_shard, shared_router, 16);
    [encoder setBuffer:router_output offset:0 atIndex:19];
    [encoder dispatchThreadgroups:MTLSizeMake(1281, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:moe_down_state];
    [encoder setBuffer:routed_hidden offset:0 atIndex:0];
    bind_projection(encoder, expert_shard, routed_down, 1);
    [encoder setBuffer:experts offset:0 atIndex:4];
    [encoder setBuffer:weights offset:0 atIndex:5];
    [encoder setBuffer:shared_hidden offset:0 atIndex:6];
    bind_projection(encoder, expert_shard, shared_down, 7);
    [encoder setBuffer:router_output offset:0 atIndex:10];
    [encoder setBuffer:block_output offset:0 atIndex:11];
    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:healing_left_state];
    [encoder setBuffer:block_output offset:0 atIndex:0];
    bind_tensor(encoder, healing_shard, healing_left, 1);
    [encoder setBuffer:healing_hidden offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:healing_right_state];
    [encoder setBuffer:block_output offset:0 atIndex:0];
    [encoder setBuffer:healing_hidden offset:0 atIndex:1];
    bind_tensor(encoder, healing_shard, healing_right, 2);
    [encoder setBuffer:stream offset:0 atIndex:3];
    [encoder setBuffer:injection_output offset:0 atIndex:4];
    [encoder setBuffer:output_stream offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
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

struct HcOracleResult {
    std::vector<float> mixed;
    std::vector<float> injection;
    double median_ms{0.0};
};

HcOracleResult mlx_hc_oracle(const qwen38::ModelManifest &manifest, const std::string &prefix,
                             std::span<const float> stream_values) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    qwen38::MlxTensorStore store(manifest);
    qwen38::HyperConnection hc(store, prefix, 2560, 4, 4, 32, 1.0e-6F, true);
    const auto stream =
        MlxArray::from_float32(stream_values, std::array<int, 3>{1, 1, 10240}).astype(MLX_BFLOAT16);
    const auto evaluate = [&] {
        auto read = hc.read(stream);
        std::array<const MlxArray *, 2> outputs{&read.mixed, &read.injection};
        MlxArray::eval_all(outputs);
        return std::pair{read.mixed.astype(MLX_FLOAT32).to_float32(),
                         read.injection.astype(MLX_FLOAT32).to_float32()};
    };
    for (int i = 0; i < 3; ++i)
        static_cast<void>(evaluate());
    std::vector<double> samples;
    std::pair<std::vector<float>, std::vector<float>> values;
    for (int i = 0; i < 15; ++i) {
        const auto started = std::chrono::steady_clock::now();
        values = evaluate();
        samples.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count());
    }
    return {std::move(values.first), std::move(values.second), median(std::move(samples))};
}

OracleResult mlx_mlp_half_oracle(const qwen38::ModelManifest &manifest,
                                 const std::string &layer_prefix, const std::string &hc_prefix,
                                 std::span<const float> stream_values) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    qwen38::MlxTensorStore store(manifest);
    qwen38::HyperConnection hc(store, hc_prefix, 2560, 4, 4, 32, 1.0e-6F, true);
    qwen38::SparseMoe moe(store, layer_prefix, manifest.config().expert_count,
                          manifest.config().experts_per_token, manifest.config().quantization_bits,
                          manifest.config().quantization_group_size,
                          manifest.config().normalize_topk_probability);
    const auto left = store.tensor(layer_prefix + ".T_delta_left");
    const auto right = store.tensor(layer_prefix + ".T_delta_right");
    const auto stream =
        MlxArray::from_float32(stream_values, std::array<int, 3>{1, 1, 10240}).astype(MLX_BFLOAT16);
    const auto evaluate = [&] {
        auto read = hc.read(stream);
        auto block = moe.forward_decode(read.mixed).astype(MLX_BFLOAT16);
        auto correction = MlxArray::matmul(MlxArray::matmul(block, left), right);
        auto healed = MlxArray::add(block, correction);
        auto output = hc.write(stream, healed, read.injection);
        output.eval();
        return output.astype(MLX_FLOAT32).to_float32();
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

std::vector<float>
mlx_attention_projection_oracle(const qwen38::ModelManifest &manifest, const ProjectionNames &index,
                                const ProjectionNames &query, const ProjectionNames &key,
                                const ProjectionNames &value, std::span<const float> input_values) {
    qwen38::MlxTensorStore store(manifest);
    const auto input =
        MlxArray::from_float32(input_values, std::array<int, 3>{1, 1, 2560}).astype(MLX_BFLOAT16);
    const auto project = [&](const ProjectionNames &names) {
        return MlxArray::quantized_matmul(input, store.tensor(names.weight),
                                          store.tensor(names.scales), store.tensor(names.biases),
                                          32, 4);
    };
    std::array<MlxArray, 4> outputs{project(index), project(query), project(key), project(value)};
    const std::array<const MlxArray *, 4> evaluated{&outputs[0], &outputs[1], &outputs[2],
                                                    &outputs[3]};
    MlxArray::eval_all(evaluated);
    std::vector<float> result;
    result.reserve(13952);
    for (const MlxArray &output : outputs) {
        auto values = output.astype(MLX_FLOAT32).to_float32();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

OracleResult mlx_full_layer_oracle(const qwen38::ModelManifest &manifest,
                                   std::span<const std::uint16_t> stream_values,
                                   std::span<const std::uint16_t> pooled_values,
                                   const std::uint32_t *key_weight, const std::uint16_t *key_scale,
                                   const std::uint16_t *key_bias, const std::uint32_t *value_weight,
                                   const std::uint16_t *value_scale,
                                   const std::uint16_t *value_bias,
                                   const std::uint32_t token_count,
                                   const std::size_t layer_index = 3) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    setenv("QWEN38_QSA_DECODE_BUDGET", "512", 1);
    setenv("QWEN38_QSA_RAW_WINDOW", "64", 1);
    const auto raw = [](const void *data, const std::span<const int> shape, const mlx_dtype dtype) {
        return MlxArray(
            mlx_array_new_data(data, shape.data(), static_cast<int>(shape.size()), dtype));
    };
    qwen38::MlxTensorStore store(manifest);
    qwen38::DecoderLayer layer(store, layer_index, manifest.config());
    const std::array<int, 3> stream_shape{1, 1, 10240};
    MlxArray stream = raw(stream_values.data(), stream_shape, MLX_BFLOAT16);
    qwen38::DecoderLayerState state;
    auto &attention = state.full_attention;
    const std::array<int, 4> weight_shape{1, 2, static_cast<int>(token_count), 64};
    const std::array<int, 4> affine_shape{1, 2, static_cast<int>(token_count), 4};
    attention.key_weights = raw(key_weight, weight_shape, MLX_UINT32);
    attention.key_scales = raw(key_scale, affine_shape, MLX_BFLOAT16);
    attention.key_biases = raw(key_bias, affine_shape, MLX_BFLOAT16);
    attention.value_weights = raw(value_weight, weight_shape, MLX_UINT32);
    attention.value_scales = raw(value_scale, affine_shape, MLX_BFLOAT16);
    attention.value_biases = raw(value_bias, affine_shape, MLX_BFLOAT16);
    attention.keys = MlxArray::zeros(std::array<int, 4>{1, 2, 0, 256}, MLX_BFLOAT16);
    attention.values = MlxArray::zeros(std::array<int, 4>{1, 2, 0, 256}, MLX_BFLOAT16);
    attention.qsa_raw_keys = MlxArray::zeros(std::array<int, 3>{1, 64, 128}, MLX_BFLOAT16);
    attention.qsa_raw_start = token_count - 64;
    attention.qsa_pooled_keys =
        raw(pooled_values.data(), std::array<int, 3>{1, static_cast<int>(token_count / 4), 128},
            MLX_BFLOAT16);
    attention.qsa_pooled_count = token_count / 4;
    attention.token_count = token_count;
    attention.kv_q8 = true;
    attention.kv_q8_cold_tokens = token_count;
    const auto started = std::chrono::steady_clock::now();
    MlxArray output = layer.forward_decode(stream, 9419, state);
    output.eval();
    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    return {output.astype(MLX_FLOAT32).to_float32(), elapsed};
}

std::vector<std::vector<float>> mlx_full_layer_trajectory_oracle(
    const qwen38::ModelManifest &manifest, std::span<const std::uint16_t> stream_values,
    std::span<const std::uint16_t> pooled_values, const std::uint32_t *key_weight,
    const std::uint16_t *key_scale, const std::uint16_t *key_bias,
    const std::uint32_t *value_weight, const std::uint16_t *value_scale,
    const std::uint16_t *value_bias, const std::uint32_t token_count,
    const std::uint32_t steps) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    setenv("QWEN38_QSA_DECODE_BUDGET", "512", 1);
    setenv("QWEN38_QSA_RAW_WINDOW", "64", 1);
    const auto raw = [](const void *data, const std::span<const int> shape, const mlx_dtype dtype) {
        return MlxArray(
            mlx_array_new_data(data, shape.data(), static_cast<int>(shape.size()), dtype));
    };
    qwen38::MlxTensorStore store(manifest);
    qwen38::DecoderLayer layer(store, 3, manifest.config());
    MlxArray stream = raw(stream_values.data(), std::array<int, 3>{1, 1, 10240}, MLX_BFLOAT16);
    qwen38::DecoderLayerState state;
    auto &attention = state.full_attention;
    attention.key_weights = raw(
        key_weight, std::array<int, 4>{1, 2, static_cast<int>(token_count), 64}, MLX_UINT32);
    attention.key_scales = raw(
        key_scale, std::array<int, 4>{1, 2, static_cast<int>(token_count), 4}, MLX_BFLOAT16);
    attention.key_biases = raw(
        key_bias, std::array<int, 4>{1, 2, static_cast<int>(token_count), 4}, MLX_BFLOAT16);
    attention.value_weights = raw(
        value_weight, std::array<int, 4>{1, 2, static_cast<int>(token_count), 64}, MLX_UINT32);
    attention.value_scales = raw(
        value_scale, std::array<int, 4>{1, 2, static_cast<int>(token_count), 4}, MLX_BFLOAT16);
    attention.value_biases = raw(
        value_bias, std::array<int, 4>{1, 2, static_cast<int>(token_count), 4}, MLX_BFLOAT16);
    attention.keys = MlxArray::zeros(std::array<int, 4>{1, 2, 0, 256}, MLX_BFLOAT16);
    attention.values = MlxArray::zeros(std::array<int, 4>{1, 2, 0, 256}, MLX_BFLOAT16);
    attention.qsa_raw_keys = MlxArray::zeros(std::array<int, 3>{1, 64, 128}, MLX_BFLOAT16);
    attention.qsa_raw_start = token_count - 64;
    attention.qsa_pooled_keys = raw(
        pooled_values.data(), std::array<int, 3>{1, static_cast<int>(token_count / 4), 128},
        MLX_BFLOAT16);
    attention.qsa_pooled_count = token_count / 4;
    attention.token_count = token_count;
    attention.kv_q8 = true;
    attention.kv_q8_cold_tokens = token_count;
    std::vector<std::vector<float>> outputs;
    outputs.reserve(steps);
    for (std::uint32_t step = 0; step < steps; ++step) {
        MlxArray output = layer.forward_decode(stream, 9419, state);
        output.eval();
        outputs.push_back(output.astype(MLX_FLOAT32).to_float32());
        stream = output.share();
    }
    return outputs;
}

struct GdnOracleResult {
    std::vector<float> output;
    std::vector<float> convolution;
    std::vector<float> recurrent;
};

GdnOracleResult mlx_gdn_oracle(const qwen38::ModelManifest &manifest,
                               const std::string &prefix,
                               std::span<const std::uint16_t> input_values) {
    setenv("QWEN38_GDN_PREWORK", "1", 1);
    setenv("QWEN38_GDN_NORM_GATE", "1", 1);
    qwen38::MlxTensorStore store(manifest);
    qwen38::GatedDeltaNet gdn(store, prefix, manifest.config());
    const auto input = MlxArray(
        mlx_array_new_data(input_values.data(), std::array<int, 3>{1, 1, 2560}.data(), 3,
                           MLX_BFLOAT16));
    qwen38::GatedDeltaNetState state;
    MlxArray output = gdn.forward_decode(input, state);
    const std::array<const MlxArray *, 3> evaluated{&output, &state.convolution, &state.recurrent};
    MlxArray::eval_all(evaluated);
    return {output.astype(MLX_FLOAT32).to_float32(),
            state.convolution.astype(MLX_FLOAT32).to_float32(),
            state.recurrent.astype(MLX_FLOAT32).to_float32()};
}

struct GdnTrajectoryResult {
    std::vector<std::vector<float>> outputs;
    std::vector<float> convolution;
    std::vector<float> recurrent;
};

GdnTrajectoryResult mlx_gdn_trajectory_oracle(
    const qwen38::ModelManifest &manifest, const std::string &prefix,
    std::span<const std::uint16_t> input_values, const std::uint32_t steps) {
    setenv("QWEN38_GDN_PREWORK", "1", 1);
    setenv("QWEN38_GDN_NORM_GATE", "1", 1);
    qwen38::MlxTensorStore store(manifest);
    qwen38::GatedDeltaNet gdn(store, prefix, manifest.config());
    const std::array<int, 3> shape{1, 1, 2560};
    MlxArray input(mlx_array_new_data(input_values.data(), shape.data(), 3, MLX_BFLOAT16));
    qwen38::GatedDeltaNetState state;
    GdnTrajectoryResult result;
    result.outputs.reserve(steps);
    for (std::uint32_t step = 0; step < steps; ++step) {
        MlxArray output = gdn.forward_decode(input, state);
        output.eval();
        result.outputs.push_back(output.astype(MLX_FLOAT32).to_float32());
        input = output.share();
    }
    const std::array<const MlxArray *, 2> evaluated{&state.convolution, &state.recurrent};
    MlxArray::eval_all(evaluated);
    result.convolution = state.convolution.astype(MLX_FLOAT32).to_float32();
    result.recurrent = state.recurrent.astype(MLX_FLOAT32).to_float32();
    return result;
}

OracleResult mlx_gdn_layer_oracle(const qwen38::ModelManifest &manifest,
                                  std::span<const std::uint16_t> stream_values,
                                  const std::size_t layer_index = 0) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    setenv("QWEN38_GDN_PREWORK", "1", 1);
    setenv("QWEN38_GDN_NORM_GATE", "1", 1);
    qwen38::MlxTensorStore store(manifest);
    qwen38::DecoderLayer layer(store, layer_index, manifest.config());
    const std::array<int, 3> shape{1, 1, 10240};
    const MlxArray stream(
        mlx_array_new_data(stream_values.data(), shape.data(), 3, MLX_BFLOAT16));
    qwen38::DecoderLayerState state;
    const auto started = std::chrono::steady_clock::now();
    MlxArray output = layer.forward_decode(stream, 9419, state);
    output.eval();
    return {output.astype(MLX_FLOAT32).to_float32(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count()};
}

std::vector<std::vector<float>> mlx_gdn_layer_trajectory_oracle(
    const qwen38::ModelManifest &manifest, std::span<const std::uint16_t> stream_values,
    const std::uint32_t steps) {
    setenv("QWEN38_HC_FUSED", "1", 1);
    unsetenv("QWEN38_HC_FUSED_INJECTION");
    setenv("QWEN38_GDN_PREWORK", "1", 1);
    setenv("QWEN38_GDN_NORM_GATE", "1", 1);
    qwen38::MlxTensorStore store(manifest);
    qwen38::DecoderLayer layer(store, 0, manifest.config());
    const std::array<int, 3> shape{1, 1, 10240};
    MlxArray stream(mlx_array_new_data(stream_values.data(), shape.data(), 3, MLX_BFLOAT16));
    qwen38::DecoderLayerState state;
    std::vector<std::vector<float>> outputs;
    outputs.reserve(steps);
    for (std::uint32_t step = 0; step < steps; ++step) {
        MlxArray output = layer.forward_decode(stream, 9419, state);
        output.eval();
        outputs.push_back(output.astype(MLX_FLOAT32).to_float32());
        stream = output.share();
    }
    return outputs;
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
            const auto device_router = projection(layer_prefix + ".gate");
            const std::string hc_prefix = "language_model.model.layers.3.mlp_hyper_connection";
            const std::string hc_norm = hc_prefix + ".hc_norm.weight";
            const auto hc_down = projection(hc_prefix + ".input_mix_weight_down");
            const auto hc_up = projection(hc_prefix + ".input_mix_weight_up");
            const auto hc_injection = projection(hc_prefix + ".block_inject_weight");
            const std::string attention_prefix = "language_model.model.layers.3.self_attn";
            const std::string attention_hc_prefix =
                "language_model.model.layers.3.attn_hyper_connection";
            const std::string attention_hc_norm = attention_hc_prefix + ".hc_norm.weight";
            const auto attention_hc_down =
                projection(attention_hc_prefix + ".input_mix_weight_down");
            const auto attention_hc_up = projection(attention_hc_prefix + ".input_mix_weight_up");
            const auto attention_hc_injection =
                projection(attention_hc_prefix + ".block_inject_weight");
            const auto attention_index = projection(attention_prefix + ".indexer.index_qk_proj");
            const auto attention_query = projection(attention_prefix + ".q_proj");
            const auto attention_key = projection(attention_prefix + ".k_proj");
            const auto attention_value = projection(attention_prefix + ".v_proj");
            const auto attention_output_projection = projection(attention_prefix + ".o_proj");
            const std::string attention_query_norm = attention_prefix + ".q_norm.weight";
            const std::string attention_key_norm = attention_prefix + ".k_norm.weight";
            const std::string attention_index_norm =
                attention_prefix + ".indexer.q_layernorm.weight";
            const std::string gdn_prefix = "language_model.model.layers.0.linear_attn";
            const auto gdn_qkv = projection(gdn_prefix + ".in_proj_qkv");
            const auto gdn_z = projection(gdn_prefix + ".in_proj_z");
            const auto gdn_beta = projection(gdn_prefix + ".in_proj_b");
            const auto gdn_decay = projection(gdn_prefix + ".in_proj_a");
            const auto gdn_output = projection(gdn_prefix + ".out_proj");
            const std::string gdn_convolution = gdn_prefix + ".conv1d.weight";
            const std::string gdn_decay_log = gdn_prefix + ".A_log";
            const std::string gdn_decay_bias = gdn_prefix + ".dt_bias";
            const std::string gdn_norm = gdn_prefix + ".norm.weight";
            const std::string gdn_mlp_prefix = "language_model.model.layers.0.mlp";
            const std::string gdn_switch_prefix = gdn_mlp_prefix + ".switch_mlp";
            const auto gdn_mlp_gate = projection(gdn_switch_prefix + ".gate_proj");
            const auto gdn_mlp_up = projection(gdn_switch_prefix + ".up_proj");
            const auto gdn_mlp_down = projection(gdn_switch_prefix + ".down_proj");
            const auto gdn_shared_gate = projection(gdn_mlp_prefix + ".shared_expert.gate_proj");
            const auto gdn_shared_up = projection(gdn_mlp_prefix + ".shared_expert.up_proj");
            const auto gdn_shared_down = projection(gdn_mlp_prefix + ".shared_expert.down_proj");
            const auto gdn_shared_router = projection(gdn_mlp_prefix + ".shared_expert_gate");
            const auto gdn_router = projection(gdn_mlp_prefix + ".gate");
            const std::string gdn_mlp_hc_prefix =
                "language_model.model.layers.0.mlp_hyper_connection";
            const std::string gdn_mlp_hc_norm = gdn_mlp_hc_prefix + ".hc_norm.weight";
            const auto gdn_mlp_hc_down =
                projection(gdn_mlp_hc_prefix + ".input_mix_weight_down");
            const auto gdn_mlp_hc_up =
                projection(gdn_mlp_hc_prefix + ".input_mix_weight_up");
            const auto gdn_mlp_hc_injection =
                projection(gdn_mlp_hc_prefix + ".block_inject_weight");
            const std::string gdn_attn_hc_prefix =
                "language_model.model.layers.0.attn_hyper_connection";
            const std::string gdn_attn_hc_norm = gdn_attn_hc_prefix + ".hc_norm.weight";
            const auto gdn_attn_hc_down =
                projection(gdn_attn_hc_prefix + ".input_mix_weight_down");
            const auto gdn_attn_hc_up =
                projection(gdn_attn_hc_prefix + ".input_mix_weight_up");
            const auto gdn_attn_hc_injection =
                projection(gdn_attn_hc_prefix + ".block_inject_weight");
            const std::string gdn_healing_left = gdn_mlp_prefix + ".T_delta_left";
            const std::string gdn_healing_right = gdn_mlp_prefix + ".T_delta_right";
            const std::string shared_prefix = "language_model.model.layers.10";
            const std::string shared_gdn_prefix = shared_prefix + ".linear_attn";
            const auto shared_gdn_qkv = projection(shared_gdn_prefix + ".in_proj_qkv");
            const auto shared_gdn_z = projection(shared_gdn_prefix + ".in_proj_z");
            const auto shared_gdn_beta = projection(shared_gdn_prefix + ".in_proj_b");
            const auto shared_gdn_decay = projection(shared_gdn_prefix + ".in_proj_a");
            const auto shared_gdn_output = projection(shared_gdn_prefix + ".out_proj");
            const std::string shared_gdn_convolution = shared_gdn_prefix + ".conv1d.weight";
            const std::string shared_gdn_decay_log = shared_gdn_prefix + ".A_log";
            const std::string shared_gdn_decay_bias = shared_gdn_prefix + ".dt_bias";
            const std::string shared_gdn_norm = shared_gdn_prefix + ".norm.weight";
            const std::string shared_mlp_prefix = shared_prefix + ".mlp";
            const auto shared_only_gate =
                projection(shared_mlp_prefix + ".shared_expert.gate_proj");
            const auto shared_only_up = projection(shared_mlp_prefix + ".shared_expert.up_proj");
            const auto shared_only_down =
                projection(shared_mlp_prefix + ".shared_expert.down_proj");
            const auto shared_only_router = projection(shared_mlp_prefix + ".shared_expert_gate");
            const std::string shared_mlp_hc_prefix = shared_prefix + ".mlp_hyper_connection";
            const std::string shared_mlp_hc_norm = shared_mlp_hc_prefix + ".hc_norm.weight";
            const auto shared_mlp_hc_down =
                projection(shared_mlp_hc_prefix + ".input_mix_weight_down");
            const auto shared_mlp_hc_up =
                projection(shared_mlp_hc_prefix + ".input_mix_weight_up");
            const auto shared_mlp_hc_injection =
                projection(shared_mlp_hc_prefix + ".block_inject_weight");
            const std::string shared_attn_hc_prefix = shared_prefix + ".attn_hyper_connection";
            const std::string shared_attn_hc_norm = shared_attn_hc_prefix + ".hc_norm.weight";
            const auto shared_attn_hc_down =
                projection(shared_attn_hc_prefix + ".input_mix_weight_down");
            const auto shared_attn_hc_up =
                projection(shared_attn_hc_prefix + ".input_mix_weight_up");
            const auto shared_attn_hc_injection =
                projection(shared_attn_hc_prefix + ".block_inject_weight");
            const std::string shared_healing_left = shared_mlp_prefix + ".T_delta_left";
            const std::string shared_healing_right = shared_mlp_prefix + ".T_delta_right";
            const std::string shared_attention_prefix = "language_model.model.layers.11";
            const std::string shared_attention_block = shared_attention_prefix + ".self_attn";
            const auto shared_attention_index =
                projection(shared_attention_block + ".indexer.index_qk_proj");
            const auto shared_attention_query = projection(shared_attention_block + ".q_proj");
            const auto shared_attention_key = projection(shared_attention_block + ".k_proj");
            const auto shared_attention_value = projection(shared_attention_block + ".v_proj");
            const auto shared_attention_output = projection(shared_attention_block + ".o_proj");
            const std::string shared_attention_query_norm = shared_attention_block + ".q_norm.weight";
            const std::string shared_attention_key_norm = shared_attention_block + ".k_norm.weight";
            const std::string shared_attention_index_norm =
                shared_attention_block + ".indexer.q_layernorm.weight";
            const std::string shared_attention_attn_hc =
                shared_attention_prefix + ".attn_hyper_connection";
            const std::string shared_attention_attn_hc_norm =
                shared_attention_attn_hc + ".hc_norm.weight";
            const auto shared_attention_attn_hc_down =
                projection(shared_attention_attn_hc + ".input_mix_weight_down");
            const auto shared_attention_attn_hc_up =
                projection(shared_attention_attn_hc + ".input_mix_weight_up");
            const auto shared_attention_attn_hc_injection =
                projection(shared_attention_attn_hc + ".block_inject_weight");
            const std::string shared_attention_mlp = shared_attention_prefix + ".mlp";
            const auto shared_attention_mlp_gate =
                projection(shared_attention_mlp + ".shared_expert.gate_proj");
            const auto shared_attention_mlp_up =
                projection(shared_attention_mlp + ".shared_expert.up_proj");
            const auto shared_attention_mlp_down =
                projection(shared_attention_mlp + ".shared_expert.down_proj");
            const auto shared_attention_mlp_router =
                projection(shared_attention_mlp + ".shared_expert_gate");
            const std::string shared_attention_mlp_hc =
                shared_attention_prefix + ".mlp_hyper_connection";
            const std::string shared_attention_mlp_hc_norm =
                shared_attention_mlp_hc + ".hc_norm.weight";
            const auto shared_attention_mlp_hc_down =
                projection(shared_attention_mlp_hc + ".input_mix_weight_down");
            const auto shared_attention_mlp_hc_up =
                projection(shared_attention_mlp_hc + ".input_mix_weight_up");
            const auto shared_attention_mlp_hc_injection =
                projection(shared_attention_mlp_hc + ".block_inject_weight");
            const std::string shared_attention_healing_left =
                shared_attention_mlp + ".T_delta_left";
            const std::string shared_attention_healing_right =
                shared_attention_mlp + ".T_delta_right";
            const std::string healing_left = layer_prefix + ".T_delta_left";
            const std::string healing_right = layer_prefix + ".T_delta_right";
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
            std::string metal_source;
            metal_source.reserve(metal_source_prefix.size() + metal_source_suffix.size());
            metal_source.append(metal_source_prefix);
            metal_source.append(metal_source_suffix);
            id<MTLLibrary> library =
                [device newLibraryWithSource:[NSString stringWithUTF8String:metal_source.c_str()]
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
            const auto router_state = pipeline(device, library, @"q8_router_logits");
            const auto select_state = pipeline(device, library, @"select_top10");
            const auto qsa_score_state = pipeline(device, library, @"qsa_score_blocks");
            const auto qsa_first_state = pipeline(device, library, @"qsa_top128_first");
            const auto qsa_merge_state = pipeline(device, library, @"qsa_top128_merge");
            const auto qsa_attention_state = pipeline(device, library, @"qsa_attention_q8_blocks");
            const auto qsa_append_state = pipeline(device, library, @"qsa_append_decode_state");
            const auto attention_projection_state =
                pipeline(device, library, @"attention_qkv_index");
            const auto attention_normalize_state =
                pipeline(device, library, @"attention_normalize_rope");
            const auto attention_gate_state = pipeline(device, library, @"attention_apply_gate");
            const auto attention_output_state =
                pipeline(device, library, @"attention_output_projection");
            const auto q4_input_state = pipeline(device, library, @"q4_input_projection");
            const auto gdn_prework_state = pipeline(device, library, @"gdn_prework");
            const auto gdn_recurrence_state = pipeline(device, library, @"gdn_recurrence");
            const auto gdn_norm_gate_state = pipeline(device, library, @"gdn_norm_gate");
            const auto gdn_output_state = pipeline(device, library, @"gdn_output_projection");
            const auto hc_write_state = pipeline(device, library, @"hc_write");
            const auto hc_normalize_state = pipeline(device, library, @"hc_normalize");
            const auto hc_down_state = pipeline(device, library, @"hc_down_injection");
            const auto hc_up_state = pipeline(device, library, @"hc_up_mix");
            const auto healing_left_state = pipeline(device, library, @"healing_left");
            const auto healing_right_state = pipeline(device, library, @"healing_right_write");
            Shard shard(device, manifest.directory() / shard_name);
            Shard router_shard(device, manifest.directory() / router_shard_name);
            Shard healing_shard(device,
                                manifest.directory() / manifest.weight_map().at(healing_left));
            Shard gdn_shard(device,
                            manifest.directory() / manifest.weight_map().at(gdn_qkv.weight));
            Shard gdn_moe_shard(
                device, manifest.directory() / manifest.weight_map().at(gdn_mlp_gate.weight));
            Shard gdn_healing_shard(
                device, manifest.directory() / manifest.weight_map().at(gdn_healing_left));
            Shard shared_layer_shard(
                device, manifest.directory() / manifest.weight_map().at(shared_gdn_qkv.weight));
            Shard shared_mlp_shard(
                device, manifest.directory() / manifest.weight_map().at(shared_only_gate.weight));
            Shard shared_attention_shard(
                device, manifest.directory() / manifest.weight_map().at(shared_attention_query.weight));
            Shard shared_attention_aux_shard(
                device, manifest.directory() /
                    manifest.weight_map().at(shared_attention_attn_hc_norm));
            Shard shared_attention_mlp_shard(
                device, manifest.directory() /
                    manifest.weight_map().at(shared_attention_mlp_gate.weight));

            std::vector<float> input_f32(hidden_size);
            for (int i = 0; i < hidden_size; ++i)
                input_f32[i] = static_cast<float>((i * 17 + 11) % 257 - 128) / 256.0F;
            std::vector<std::uint16_t> input_bf16(hidden_size);
            std::ranges::transform(input_f32, input_bf16.begin(), bf16);
            std::vector<float> stream_f32(10240);
            for (int i = 0; i < 10240; ++i)
                stream_f32[i] = static_cast<float>((i * 29 + 7) % 521 - 260) / 512.0F;
            std::vector<std::uint16_t> stream_bf16(10240);
            std::ranges::transform(stream_f32, stream_bf16.begin(), bf16);
            constexpr std::uint32_t qsa_blocks = 32768;
            std::vector<std::uint16_t> qsa_query_values(4 * 128);
            std::vector<std::uint16_t> qsa_pooled_values(static_cast<std::size_t>(qsa_blocks) *
                                                         128);
            std::vector<std::uint16_t> qsa_attention_query_values(24 * 256);
            for (std::size_t i = 0; i < qsa_query_values.size(); ++i)
                qsa_query_values[i] =
                    bf16(static_cast<float>(static_cast<int>((i * 37 + 5) % 257) - 128) / 256.0F);
            for (std::size_t i = 0; i < qsa_pooled_values.size(); ++i)
                qsa_pooled_values[i] =
                    bf16(static_cast<float>(static_cast<int>((i * 43 + 17) % 263) - 131) / 256.0F);
            for (std::size_t i = 0; i < qsa_attention_query_values.size(); ++i)
                qsa_attention_query_values[i] =
                    bf16(static_cast<float>(static_cast<int>((i * 19 + 3) % 127) - 63) / 512.0F);
            std::array<std::uint16_t, 64> rope_cos_values{}, rope_sin_values{};
            for (std::size_t i = 0; i < 32; ++i) {
                const double frequency = std::pow(10000000.0, -2.0 * static_cast<double>(i) / 64.0);
                const double angle = 131072.0 * frequency;
                rope_cos_values[i] = rope_cos_values[i + 32] =
                    bf16(static_cast<float>(std::cos(angle)));
                rope_sin_values[i] = rope_sin_values[i + 32] =
                    bf16(static_cast<float>(std::sin(angle)));
            }
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
            id<MTLBuffer> hc_stream =
                [device newBufferWithBytes:stream_bf16.data()
                                    length:stream_bf16.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_normalized = [device newBufferWithLength:10240 * sizeof(std::uint16_t)
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_activation = [device newBufferWithLength:320 * sizeof(std::uint16_t)
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_injection_output =
                [device newBufferWithLength:4 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_mixed = [device newBufferWithLength:2560 * sizeof(std::uint16_t)
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
            id<MTLBuffer> healing_hidden =
                [device newBufferWithLength:64 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> hc_output_stream =
                [device newBufferWithLength:10240 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_query =
                [device newBufferWithBytes:qsa_query_values.data()
                                    length:qsa_query_values.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_pooled =
                [device newBufferWithLength:(qsa_pooled_values.size() + 128) *
                                            sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            std::memcpy(qsa_pooled.contents, qsa_pooled_values.data(),
                        qsa_pooled_values.size() * sizeof(std::uint16_t));
            id<MTLBuffer> qsa_scores = [device newBufferWithLength:65536 * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_selected = [device newBufferWithLength:128 * sizeof(std::uint32_t)
                                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_temp_scores_a =
                [device newBufferWithLength:32768 * sizeof(float)
                                    options:MTLResourceStorageModePrivate];
            id<MTLBuffer> qsa_temp_scores_b =
                [device newBufferWithLength:32768 * sizeof(float)
                                    options:MTLResourceStorageModePrivate];
            id<MTLBuffer> qsa_temp_ids_a =
                [device newBufferWithLength:32768 * sizeof(std::uint32_t)
                                    options:MTLResourceStorageModePrivate];
            id<MTLBuffer> qsa_temp_ids_b =
                [device newBufferWithLength:32768 * sizeof(std::uint32_t)
                                    options:MTLResourceStorageModePrivate];
            constexpr std::uint32_t qsa_tokens = qsa_blocks * 4;
            const NSUInteger qsa_word_count = static_cast<NSUInteger>(2) * qsa_tokens * 64;
            const NSUInteger qsa_affine_count = static_cast<NSUInteger>(2) * qsa_tokens * 4;
            id<MTLBuffer> qsa_attention_query =
                [device newBufferWithBytes:qsa_attention_query_values.data()
                                    length:qsa_attention_query_values.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_key_weight =
                [device newBufferWithLength:qsa_word_count * sizeof(std::uint32_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_value_weight =
                [device newBufferWithLength:qsa_word_count * sizeof(std::uint32_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_key_scale =
                [device newBufferWithLength:qsa_affine_count * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_key_bias =
                [device newBufferWithLength:qsa_affine_count * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_value_scale =
                [device newBufferWithLength:qsa_affine_count * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_value_bias =
                [device newBufferWithLength:qsa_affine_count * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_attention_output =
                [device newBufferWithLength:24 * 256 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_projection_output =
                [device newBufferWithLength:13952 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> rope_cos =
                [device newBufferWithBytes:rope_cos_values.data()
                                    length:rope_cos_values.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> rope_sin =
                [device newBufferWithBytes:rope_sin_values.data()
                                    length:rope_sin_values.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> pool_rope_cos =
                [device newBufferWithBytes:rope_cos_values.data()
                                    length:rope_cos_values.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> pool_rope_sin =
                [device newBufferWithBytes:rope_sin_values.data()
                                    length:rope_sin_values.size() * sizeof(std::uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_selector_query =
                [device newBufferWithLength:4 * 128 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_query_normalized =
                [device newBufferWithLength:24 * 256 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_key_normalized =
                [device newBufferWithLength:2 * 256 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            constexpr std::uint32_t hot_capacity = 2048;
            id<MTLBuffer> attention_hot_keys =
                [device newBufferWithLength:2ULL * hot_capacity * 256 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_hot_values =
                [device newBufferWithLength:2ULL * hot_capacity * 256 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> qsa_pending_raw =
                [device newBufferWithLength:4 * 128 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_gated =
                [device newBufferWithLength:24 * 256 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_block_output =
                [device newBufferWithLength:2560 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_output_stream =
                [device newBufferWithLength:10240 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_projected =
                [device newBufferWithLength:16480 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_convolution_state =
                [device newBufferWithLength:3 * 10240 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_query_buffer =
                [device newBufferWithLength:2048 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_key_buffer =
                [device newBufferWithLength:2048 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_value_buffer =
                [device newBufferWithLength:6144 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_decay_buffer =
                [device newBufferWithLength:48 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_beta_buffer =
                [device newBufferWithLength:48 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_recurrent_state_buffer =
                [device newBufferWithLength:48ULL * 128 * 128 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_recurrent_output =
                [device newBufferWithLength:6144 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_gated_output =
                [device newBufferWithLength:6144 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> gdn_block_output =
                [device newBufferWithLength:2560 * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> logits = [device newBufferWithLength:512 * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> cache_evict = [device newBufferWithLength:64ULL * 1024 * 1024
                                                            options:MTLResourceStorageModePrivate];
            if (input == nil || experts == nil || weights == nil || hidden == nil ||
                output == nil || shared_hidden == nil || shared_router_output == nil ||
                full_output == nil || logits == nil || cache_evict == nil || hc_stream == nil ||
                hc_normalized == nil || hc_activation == nil || hc_injection_output == nil ||
                hc_mixed == nil || healing_hidden == nil || hc_output_stream == nil ||
                qsa_query == nil || qsa_pooled == nil || qsa_scores == nil || qsa_selected == nil ||
                qsa_temp_scores_a == nil || qsa_temp_scores_b == nil || qsa_temp_ids_a == nil ||
                qsa_temp_ids_b == nil || qsa_attention_query == nil || qsa_key_weight == nil ||
                qsa_value_weight == nil || qsa_key_scale == nil || qsa_key_bias == nil ||
                qsa_value_scale == nil || qsa_value_bias == nil || qsa_attention_output == nil ||
                attention_projection_output == nil || rope_cos == nil || rope_sin == nil ||
                pool_rope_cos == nil || pool_rope_sin == nil ||
                attention_selector_query == nil || attention_query_normalized == nil ||
                attention_key_normalized == nil || attention_hot_keys == nil ||
                attention_hot_values == nil || qsa_pending_raw == nil || attention_gated == nil ||
                attention_block_output == nil || attention_output_stream == nil ||
                gdn_projected == nil || gdn_convolution_state == nil ||
                gdn_query_buffer == nil || gdn_key_buffer == nil || gdn_value_buffer == nil ||
                gdn_decay_buffer == nil || gdn_beta_buffer == nil ||
                gdn_recurrent_state_buffer == nil || gdn_recurrent_output == nil ||
                gdn_gated_output == nil || gdn_block_output == nil)
                throw std::runtime_error("Metal scratch allocation failed");

            auto *qkw = static_cast<std::uint32_t *>(qsa_key_weight.contents);
            auto *qvw = static_cast<std::uint32_t *>(qsa_value_weight.contents);
            for (NSUInteger i = 0; i < qsa_word_count; ++i) {
                qkw[i] = static_cast<std::uint32_t>(i * 2654435761U + 17U);
                qvw[i] = static_cast<std::uint32_t>(i * 2246822519U + 31U);
            }
            auto *qks = static_cast<std::uint16_t *>(qsa_key_scale.contents);
            auto *qkb = static_cast<std::uint16_t *>(qsa_key_bias.contents);
            auto *qvs = static_cast<std::uint16_t *>(qsa_value_scale.contents);
            auto *qvb = static_cast<std::uint16_t *>(qsa_value_bias.contents);
            for (NSUInteger i = 0; i < qsa_affine_count; ++i) {
                qks[i] = bf16(0.0020F + static_cast<float>(i % 5) * 0.00005F);
                qkb[i] = bf16(-0.255F);
                qvs[i] = bf16(0.0018F + static_cast<float>(i % 7) * 0.00004F);
                qvb[i] = bf16(-0.230F);
            }

            const auto run_gdn = [&](const bool reset_state) {
                if (reset_state) {
                    std::memset(gdn_convolution_state.contents, 0,
                                3 * 10240 * sizeof(std::uint16_t));
                    std::memset(gdn_recurrent_state_buffer.contents, 0,
                                48ULL * 128 * 128 * sizeof(std::uint16_t));
                }
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = nil;
                const auto encode_input_projection = [&](const ProjectionNames &projection,
                                                         const std::uint32_t rows,
                                                         const NSUInteger output_offset) {
                    id<MTLComputeCommandEncoder> projection_encoder =
                        [command computeCommandEncoder];
                    [projection_encoder setComputePipelineState:q4_input_state];
                    [projection_encoder setBuffer:input offset:0 atIndex:0];
                    bind_projection(projection_encoder, gdn_shard, projection, 1);
                    [projection_encoder setBuffer:gdn_projected
                                           offset:output_offset * sizeof(std::uint16_t)
                                          atIndex:4];
                    [projection_encoder setBytes:&rows length:sizeof(rows) atIndex:5];
                    [projection_encoder dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, 1, 1)
                                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [projection_encoder endEncoding];
                };
                encode_input_projection(gdn_qkv, 10240, 0);
                encode_input_projection(gdn_z, 6144, 10240);
                encode_input_projection(gdn_beta, 48, 16384);
                encode_input_projection(gdn_decay, 48, 16432);

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_prework_state];
                [encoder setBuffer:gdn_projected offset:0 atIndex:0];
                [encoder setBuffer:gdn_convolution_state offset:0 atIndex:1];
                bind_tensor(encoder, gdn_shard, gdn_convolution, 2);
                bind_tensor(encoder, gdn_shard, gdn_decay_log, 3);
                bind_tensor(encoder, gdn_shard, gdn_decay_bias, 4);
                [encoder setBuffer:gdn_query_buffer offset:0 atIndex:5];
                [encoder setBuffer:gdn_key_buffer offset:0 atIndex:6];
                [encoder setBuffer:gdn_value_buffer offset:0 atIndex:7];
                [encoder setBuffer:gdn_decay_buffer offset:0 atIndex:8];
                [encoder setBuffer:gdn_beta_buffer offset:0 atIndex:9];
                [encoder dispatchThreadgroups:MTLSizeMake(80, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_recurrence_state];
                [encoder setBuffer:gdn_query_buffer offset:0 atIndex:0];
                [encoder setBuffer:gdn_key_buffer offset:0 atIndex:1];
                [encoder setBuffer:gdn_value_buffer offset:0 atIndex:2];
                [encoder setBuffer:gdn_decay_buffer offset:0 atIndex:3];
                [encoder setBuffer:gdn_beta_buffer offset:0 atIndex:4];
                [encoder setBuffer:gdn_recurrent_state_buffer offset:0 atIndex:5];
                [encoder setBuffer:gdn_recurrent_output offset:0 atIndex:6];
                [encoder dispatchThreadgroups:MTLSizeMake(128, 48, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_norm_gate_state];
                [encoder setBuffer:gdn_recurrent_output offset:0 atIndex:0];
                [encoder setBuffer:gdn_projected offset:0 atIndex:1];
                bind_tensor(encoder, gdn_shard, gdn_norm, 2);
                [encoder setBuffer:gdn_gated_output offset:0 atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(48, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_output_state];
                [encoder setBuffer:gdn_gated_output offset:0 atIndex:0];
                bind_projection(encoder, gdn_shard, gdn_output, 1);
                [encoder setBuffer:gdn_block_output offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error(command.error.localizedDescription.UTF8String);
                return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            };

            static_cast<void>(run_gdn(true));
            const auto gdn_projection_oracle = mlx_attention_projection_oracle(
                manifest, gdn_qkv, gdn_z, gdn_beta, gdn_decay, input_f32);
            const GdnOracleResult gdn_oracle = mlx_gdn_oracle(manifest, gdn_prefix, input_bf16);
            const auto *gdn_projected_bits =
                static_cast<const std::uint16_t *>(gdn_projected.contents);
            const auto *gdn_direct_bits =
                static_cast<const std::uint16_t *>(gdn_block_output.contents);
            const auto *gdn_conv_bits =
                static_cast<const std::uint16_t *>(gdn_convolution_state.contents);
            const auto *gdn_recurrent_bits =
                static_cast<const std::uint16_t *>(gdn_recurrent_state_buffer.contents);
            double gdn_dot = 0.0, gdn_aa = 0.0, gdn_bb = 0.0, gdn_squared_error = 0.0;
            double gdn_max_abs = 0.0, gdn_conv_max_abs = 0.0, gdn_recurrent_max_abs = 0.0;
            double gdn_projection_max_abs = 0.0;
            std::size_t gdn_direct_nonfinite = 0, gdn_oracle_nonfinite = 0;
            std::array<std::size_t, 4> gdn_projection_nonfinite{};
            for (std::size_t component = 0; component < 16480; ++component) {
                const double actual = from_bf16(gdn_projected_bits[component]);
                const double expected = gdn_projection_oracle[component];
                const std::size_t section = component < 10240 ? 0
                    : (component < 16384 ? 1 : (component < 16432 ? 2 : 3));
                gdn_projection_nonfinite[section] += !std::isfinite(actual);
                gdn_projection_max_abs =
                    std::max(gdn_projection_max_abs, std::abs(actual - expected));
            }
            for (std::size_t component = 0; component < 2560; ++component) {
                const double actual = from_bf16(gdn_direct_bits[component]);
                const double expected = gdn_oracle.output[component];
                gdn_direct_nonfinite += !std::isfinite(actual);
                gdn_oracle_nonfinite += !std::isfinite(expected);
                const double delta = actual - expected;
                gdn_dot += actual * expected;
                gdn_aa += actual * actual;
                gdn_bb += expected * expected;
                gdn_squared_error += delta * delta;
                gdn_max_abs = std::max(gdn_max_abs, std::abs(delta));
            }
            for (std::size_t component = 0; component < gdn_oracle.convolution.size(); ++component)
                gdn_conv_max_abs = std::max(
                    gdn_conv_max_abs,
                    std::abs(static_cast<double>(from_bf16(gdn_conv_bits[component]) -
                                                 gdn_oracle.convolution[component])));
            for (std::size_t component = 0; component < gdn_oracle.recurrent.size(); ++component)
                gdn_recurrent_max_abs = std::max(
                    gdn_recurrent_max_abs,
                    std::abs(static_cast<double>(from_bf16(gdn_recurrent_bits[component]) -
                                                 gdn_oracle.recurrent[component])));
            if (gdn_direct_nonfinite != 0 || gdn_oracle_nonfinite != 0 ||
                std::ranges::any_of(gdn_projection_nonfinite,
                                    [](const std::size_t count) { return count != 0; }))
                throw std::runtime_error("non-finite persistent GDN result");
            for (int warmup = 0; warmup < 5; ++warmup) static_cast<void>(run_gdn(true));
            std::vector<double> gdn_gpu;
            for (int iteration = 0; iteration < 31; ++iteration) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(iteration + 181));
                gdn_gpu.push_back(run_gdn(true));
            }

            const auto encode_hc_read = [&](id<MTLCommandBuffer> command,
                                            id<MTLBuffer> stream, const Shard &weight_shard,
                                            const std::string &norm,
                                            const ProjectionNames &down_projection,
                                            const ProjectionNames &up_projection,
                                            const ProjectionNames &injection_projection) {
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_normalize_state];
                [encoder setBuffer:stream offset:0 atIndex:0];
                bind_tensor(encoder, weight_shard, norm, 1);
                [encoder setBuffer:hc_normalized offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_down_state];
                [encoder setBuffer:hc_normalized offset:0 atIndex:0];
                bind_projection(encoder, weight_shard, down_projection, 1);
                bind_projection(encoder, weight_shard, injection_projection, 4);
                [encoder setBuffer:hc_activation offset:0 atIndex:7];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:8];
                [encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_up_state];
                [encoder setBuffer:hc_normalized offset:0 atIndex:0];
                [encoder setBuffer:hc_activation offset:0 atIndex:1];
                bind_projection(encoder, weight_shard, up_projection, 2);
                [encoder setBuffer:hc_mixed offset:0 atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            const auto encode_gdn_block = [&] (
                id<MTLCommandBuffer> command, id<MTLBuffer> block_input,
                const Shard &weight_shard, const ProjectionNames &qkv_projection,
                const ProjectionNames &z_projection, const ProjectionNames &beta_projection,
                const ProjectionNames &decay_projection,
                const ProjectionNames &output_projection, const std::string &convolution_weight,
                const std::string &decay_log, const std::string &decay_bias,
                const std::string &norm_weight) {
                const auto encode_projection = [&](const ProjectionNames &projection,
                                                   const std::uint32_t rows,
                                                   const NSUInteger output_offset) {
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:q4_input_state];
                    [encoder setBuffer:block_input offset:0 atIndex:0];
                    bind_projection(encoder, weight_shard, projection, 1);
                    [encoder setBuffer:gdn_projected
                                offset:output_offset * sizeof(std::uint16_t)
                               atIndex:4];
                    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
                    [encoder dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];
                };
                encode_projection(qkv_projection, 10240, 0);
                encode_projection(z_projection, 6144, 10240);
                encode_projection(beta_projection, 48, 16384);
                encode_projection(decay_projection, 48, 16432);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_prework_state];
                [encoder setBuffer:gdn_projected offset:0 atIndex:0];
                [encoder setBuffer:gdn_convolution_state offset:0 atIndex:1];
                bind_tensor(encoder, weight_shard, convolution_weight, 2);
                bind_tensor(encoder, weight_shard, decay_log, 3);
                bind_tensor(encoder, weight_shard, decay_bias, 4);
                [encoder setBuffer:gdn_query_buffer offset:0 atIndex:5];
                [encoder setBuffer:gdn_key_buffer offset:0 atIndex:6];
                [encoder setBuffer:gdn_value_buffer offset:0 atIndex:7];
                [encoder setBuffer:gdn_decay_buffer offset:0 atIndex:8];
                [encoder setBuffer:gdn_beta_buffer offset:0 atIndex:9];
                [encoder dispatchThreadgroups:MTLSizeMake(80, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_recurrence_state];
                [encoder setBuffer:gdn_query_buffer offset:0 atIndex:0];
                [encoder setBuffer:gdn_key_buffer offset:0 atIndex:1];
                [encoder setBuffer:gdn_value_buffer offset:0 atIndex:2];
                [encoder setBuffer:gdn_decay_buffer offset:0 atIndex:3];
                [encoder setBuffer:gdn_beta_buffer offset:0 atIndex:4];
                [encoder setBuffer:gdn_recurrent_state_buffer offset:0 atIndex:5];
                [encoder setBuffer:gdn_recurrent_output offset:0 atIndex:6];
                [encoder dispatchThreadgroups:MTLSizeMake(128, 48, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_norm_gate_state];
                [encoder setBuffer:gdn_recurrent_output offset:0 atIndex:0];
                [encoder setBuffer:gdn_projected offset:0 atIndex:1];
                bind_tensor(encoder, weight_shard, norm_weight, 2);
                [encoder setBuffer:gdn_gated_output offset:0 atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(48, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:gdn_output_state];
                [encoder setBuffer:gdn_gated_output offset:0 atIndex:0];
                bind_projection(encoder, weight_shard, output_projection, 1);
                [encoder setBuffer:gdn_block_output offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
            };
            const auto encode_gdn_mlp = [&](id<MTLCommandBuffer> command,
                                            id<MTLBuffer> stream) {
                encode_hc_read(command, stream, gdn_shard, gdn_mlp_hc_norm,
                               gdn_mlp_hc_down, gdn_mlp_hc_up, gdn_mlp_hc_injection);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:router_state];
                [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                bind_projection(encoder, gdn_shard, gdn_router, 1);
                [encoder setBuffer:logits offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:select_state];
                [encoder setBuffer:logits offset:0 atIndex:0];
                [encoder setBuffer:experts offset:0 atIndex:1];
                [encoder setBuffer:weights offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:fused_gate_state];
                [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                bind_projection(encoder, gdn_moe_shard, gdn_mlp_gate, 1);
                bind_projection(encoder, gdn_moe_shard, gdn_mlp_up, 4);
                [encoder setBuffer:experts offset:0 atIndex:7];
                [encoder setBuffer:hidden offset:0 atIndex:8];
                bind_projection(encoder, gdn_moe_shard, gdn_shared_gate, 9);
                bind_projection(encoder, gdn_moe_shard, gdn_shared_up, 12);
                [encoder setBuffer:shared_hidden offset:0 atIndex:15];
                bind_projection(encoder, gdn_shard, gdn_shared_router, 16);
                [encoder setBuffer:shared_router_output offset:0 atIndex:19];
                [encoder dispatchThreadgroups:MTLSizeMake(1281, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:fused_down_state];
                [encoder setBuffer:hidden offset:0 atIndex:0];
                bind_projection(encoder, gdn_moe_shard, gdn_mlp_down, 1);
                [encoder setBuffer:experts offset:0 atIndex:4];
                [encoder setBuffer:weights offset:0 atIndex:5];
                [encoder setBuffer:shared_hidden offset:0 atIndex:6];
                bind_projection(encoder, gdn_moe_shard, gdn_shared_down, 7);
                [encoder setBuffer:shared_router_output offset:0 atIndex:10];
                [encoder setBuffer:full_output offset:0 atIndex:11];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:healing_left_state];
                [encoder setBuffer:full_output offset:0 atIndex:0];
                bind_tensor(encoder, gdn_healing_shard, gdn_healing_left, 1);
                [encoder setBuffer:healing_hidden offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:healing_right_state];
                [encoder setBuffer:full_output offset:0 atIndex:0];
                [encoder setBuffer:healing_hidden offset:0 atIndex:1];
                bind_tensor(encoder, gdn_healing_shard, gdn_healing_right, 2);
                [encoder setBuffer:stream offset:0 atIndex:3];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:4];
                [encoder setBuffer:hc_output_stream offset:0 atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
            };
            const auto run_gdn_layer = [&](const bool reset_state = true) {
                if (reset_state) {
                    std::memset(gdn_convolution_state.contents, 0,
                                3 * 10240 * sizeof(std::uint16_t));
                    std::memset(gdn_recurrent_state_buffer.contents, 0,
                                48ULL * 128 * 128 * sizeof(std::uint16_t));
                }
                id<MTLCommandBuffer> command = [queue commandBuffer];
                encode_hc_read(command, hc_stream, gdn_shard, gdn_attn_hc_norm,
                               gdn_attn_hc_down, gdn_attn_hc_up, gdn_attn_hc_injection);
                encode_gdn_block(command, hc_mixed, gdn_shard, gdn_qkv, gdn_z, gdn_beta,
                                 gdn_decay, gdn_output, gdn_convolution, gdn_decay_log,
                                 gdn_decay_bias, gdn_norm);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_write_state];
                [encoder setBuffer:hc_stream offset:0 atIndex:0];
                [encoder setBuffer:gdn_block_output offset:0 atIndex:1];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:2];
                [encoder setBuffer:attention_output_stream offset:0 atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(10240, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                encode_gdn_mlp(command, attention_output_stream);
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error(command.error.localizedDescription.UTF8String);
                return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            };
            const double gdn_layer_first_gpu = run_gdn_layer();
            const OracleResult gdn_layer_oracle = mlx_gdn_layer_oracle(manifest, stream_bf16);
            const auto *gdn_layer_bits =
                static_cast<const std::uint16_t *>(hc_output_stream.contents);
            double gdn_layer_dot = 0.0, gdn_layer_aa = 0.0, gdn_layer_bb = 0.0;
            double gdn_layer_squared_error = 0.0, gdn_layer_max_abs = 0.0;
            for (std::size_t component = 0; component < 10240; ++component) {
                const double actual = from_bf16(gdn_layer_bits[component]);
                const double expected = gdn_layer_oracle.output[component];
                const double delta = actual - expected;
                gdn_layer_dot += actual * expected;
                gdn_layer_aa += actual * actual;
                gdn_layer_bb += expected * expected;
                gdn_layer_squared_error += delta * delta;
                gdn_layer_max_abs = std::max(gdn_layer_max_abs, std::abs(delta));
            }
            for (int warmup = 0; warmup < 5; ++warmup) static_cast<void>(run_gdn_layer());
            std::vector<double> gdn_layer_gpu;
            for (int iteration = 0; iteration < 31; ++iteration) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(iteration + 213));
                gdn_layer_gpu.push_back(run_gdn_layer());
            }
            const auto encode_shared_only_mlp = [&] (
                id<MTLCommandBuffer> command, id<MTLBuffer> stream,
                const Shard &aux_shard, const Shard &mlp_shard,
                const ProjectionNames &shared_gate_projection,
                const ProjectionNames &shared_up_projection,
                const ProjectionNames &shared_down_projection,
                const ProjectionNames &shared_router_projection,
                const std::string &hc_norm, const ProjectionNames &hc_down,
                const ProjectionNames &hc_up, const ProjectionNames &hc_injection,
                const std::string &healing_left, const std::string &healing_right) {
                encode_hc_read(command, stream, aux_shard, hc_norm, hc_down, hc_up, hc_injection);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:shared_gate_state];
                [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                bind_projection(encoder, mlp_shard, shared_gate_projection, 1);
                bind_projection(encoder, mlp_shard, shared_up_projection, 4);
                [encoder setBuffer:shared_hidden offset:0 atIndex:7];
                [encoder dispatchThreadgroups:MTLSizeMake(160, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:shared_router_state];
                [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                bind_projection(encoder, aux_shard, shared_router_projection, 1);
                [encoder setBuffer:shared_router_output offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:shared_down_state];
                [encoder setBuffer:shared_hidden offset:0 atIndex:0];
                bind_projection(encoder, mlp_shard, shared_down_projection, 1);
                [encoder setBuffer:shared_router_output offset:0 atIndex:4];
                [encoder setBuffer:full_output offset:0 atIndex:5];
                [encoder setBuffer:full_output offset:0 atIndex:6];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:healing_left_state];
                [encoder setBuffer:full_output offset:0 atIndex:0];
                bind_tensor(encoder, gdn_healing_shard, healing_left, 1);
                [encoder setBuffer:healing_hidden offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:healing_right_state];
                [encoder setBuffer:full_output offset:0 atIndex:0];
                [encoder setBuffer:healing_hidden offset:0 atIndex:1];
                bind_tensor(encoder, gdn_healing_shard, healing_right, 2);
                [encoder setBuffer:stream offset:0 atIndex:3];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:4];
                [encoder setBuffer:hc_output_stream offset:0 atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
            };
            const auto run_shared_gdn_layer = [&] {
                std::memset(gdn_convolution_state.contents, 0,
                            3 * 10240 * sizeof(std::uint16_t));
                std::memset(gdn_recurrent_state_buffer.contents, 0,
                            48ULL * 128 * 128 * sizeof(std::uint16_t));
                std::memset(full_output.contents, 0, 2560 * sizeof(std::uint16_t));
                id<MTLCommandBuffer> command = [queue commandBuffer];
                encode_hc_read(command, hc_stream, shared_layer_shard, shared_attn_hc_norm,
                               shared_attn_hc_down, shared_attn_hc_up,
                               shared_attn_hc_injection);
                encode_gdn_block(command, hc_mixed, shared_layer_shard, shared_gdn_qkv,
                                 shared_gdn_z, shared_gdn_beta, shared_gdn_decay,
                                 shared_gdn_output, shared_gdn_convolution,
                                 shared_gdn_decay_log, shared_gdn_decay_bias, shared_gdn_norm);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_write_state];
                [encoder setBuffer:hc_stream offset:0 atIndex:0];
                [encoder setBuffer:gdn_block_output offset:0 atIndex:1];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:2];
                [encoder setBuffer:attention_output_stream offset:0 atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(10240, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                encode_shared_only_mlp(
                    command, attention_output_stream, shared_layer_shard, shared_mlp_shard,
                    shared_only_gate, shared_only_up, shared_only_down, shared_only_router,
                    shared_mlp_hc_norm, shared_mlp_hc_down, shared_mlp_hc_up,
                    shared_mlp_hc_injection, shared_healing_left, shared_healing_right);
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error(command.error.localizedDescription.UTF8String);
                return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            };
            const double shared_gdn_layer_first_gpu = run_shared_gdn_layer();
            const OracleResult shared_gdn_layer_oracle =
                mlx_gdn_layer_oracle(manifest, stream_bf16, 10);
            const auto *shared_gdn_layer_bits =
                static_cast<const std::uint16_t *>(hc_output_stream.contents);
            double shared_gdn_layer_dot = 0.0, shared_gdn_layer_aa = 0.0;
            double shared_gdn_layer_bb = 0.0, shared_gdn_layer_squared_error = 0.0;
            double shared_gdn_layer_max_abs = 0.0;
            for (std::size_t component = 0; component < 10240; ++component) {
                const double actual = from_bf16(shared_gdn_layer_bits[component]);
                const double expected = shared_gdn_layer_oracle.output[component];
                const double delta = actual - expected;
                shared_gdn_layer_dot += actual * expected;
                shared_gdn_layer_aa += actual * actual;
                shared_gdn_layer_bb += expected * expected;
                shared_gdn_layer_squared_error += delta * delta;
                shared_gdn_layer_max_abs =
                    std::max(shared_gdn_layer_max_abs, std::abs(delta));
            }
            for (int warmup = 0; warmup < 5; ++warmup)
                static_cast<void>(run_shared_gdn_layer());
            std::vector<double> shared_gdn_layer_gpu;
            for (int iteration = 0; iteration < 31; ++iteration) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(iteration + 229));
                shared_gdn_layer_gpu.push_back(run_shared_gdn_layer());
            }
            const auto run_shared_attention_layer = [&] {
                std::memset(full_output.contents, 0, 2560 * sizeof(std::uint16_t));
                id<MTLCommandBuffer> command = [queue commandBuffer];
                encode_hc_read(command, hc_stream, shared_attention_aux_shard,
                               shared_attention_attn_hc_norm,
                               shared_attention_attn_hc_down, shared_attention_attn_hc_up,
                               shared_attention_attn_hc_injection);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_projection_state];
                [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                bind_projection(encoder, shared_attention_shard, shared_attention_index, 1);
                bind_projection(encoder, shared_attention_shard, shared_attention_query, 4);
                bind_projection(encoder, shared_attention_shard, shared_attention_key, 7);
                bind_projection(encoder, shared_attention_shard, shared_attention_value, 10);
                [encoder setBuffer:attention_projection_output offset:0 atIndex:13];
                [encoder dispatchThreadgroups:MTLSizeMake((13952 + 3) / 4, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_normalize_state];
                [encoder setBuffer:attention_projection_output offset:0 atIndex:0];
                bind_tensor(encoder, shared_attention_shard, shared_attention_query_norm, 1);
                bind_tensor(encoder, shared_attention_shard, shared_attention_key_norm, 2);
                bind_tensor(encoder, shared_attention_shard, shared_attention_index_norm, 3);
                [encoder setBuffer:rope_cos offset:0 atIndex:4];
                [encoder setBuffer:rope_sin offset:0 atIndex:5];
                [encoder setBuffer:attention_selector_query offset:0 atIndex:6];
                [encoder setBuffer:attention_query_normalized offset:0 atIndex:7];
                [encoder setBuffer:attention_key_normalized offset:0 atIndex:8];
                [encoder dispatchThreadgroups:MTLSizeMake(30, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_append_state];
                [encoder setBuffer:attention_projection_output offset:0 atIndex:0];
                [encoder setBuffer:attention_key_normalized offset:0 atIndex:1];
                [encoder setBuffer:attention_hot_keys offset:0 atIndex:2];
                [encoder setBuffer:attention_hot_values offset:0 atIndex:3];
                [encoder setBuffer:qsa_pending_raw offset:0 atIndex:4];
                [encoder setBuffer:qsa_pooled offset:0 atIndex:5];
                bind_tensor(encoder, shared_attention_shard, shared_attention_index_norm, 6);
                [encoder setBuffer:pool_rope_cos offset:0 atIndex:7];
                [encoder setBuffer:pool_rope_sin offset:0 atIndex:8];
                const std::uint32_t zero = 0;
                [encoder setBytes:&zero length:sizeof(zero) atIndex:9];
                [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:10];
                [encoder setBytes:&zero length:sizeof(zero) atIndex:11];
                [encoder setBytes:&qsa_blocks length:sizeof(qsa_blocks) atIndex:12];
                [encoder setBytes:&zero length:sizeof(zero) atIndex:13];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_score_state];
                [encoder setBuffer:attention_selector_query offset:0 atIndex:0];
                [encoder setBuffer:qsa_pooled offset:0 atIndex:1];
                [encoder setBuffer:qsa_scores offset:0 atIndex:2];
                [encoder setBytes:&qsa_blocks length:sizeof(qsa_blocks) atIndex:3];
                [encoder setBytes:&qsa_blocks length:sizeof(qsa_blocks) atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(qsa_blocks / 4, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_first_state];
                [encoder setBuffer:qsa_scores offset:0 atIndex:0];
                [encoder setBuffer:qsa_temp_scores_a offset:0 atIndex:1];
                [encoder setBuffer:qsa_temp_ids_a offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(qsa_blocks / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                std::uint32_t selection_groups = qsa_blocks / 256;
                bool selection_source_a = true;
                while (selection_groups > 1) {
                    const std::uint32_t output_groups = (selection_groups + 1) / 2;
                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:qsa_merge_state];
                    [encoder setBuffer:(selection_source_a ? qsa_temp_scores_a : qsa_temp_scores_b)
                                offset:0 atIndex:0];
                    [encoder setBuffer:(selection_source_a ? qsa_temp_ids_a : qsa_temp_ids_b)
                                offset:0 atIndex:1];
                    [encoder setBuffer:(selection_source_a ? qsa_temp_scores_b : qsa_temp_scores_a)
                                offset:0 atIndex:2];
                    [encoder setBuffer:(output_groups == 1 ? qsa_selected
                                                            : (selection_source_a ? qsa_temp_ids_b
                                                                                  : qsa_temp_ids_a))
                                offset:0 atIndex:3];
                    [encoder setBytes:&selection_groups length:sizeof(selection_groups) atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(output_groups, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];
                    selection_groups = output_groups;
                    selection_source_a = !selection_source_a;
                }
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_attention_state];
                [encoder setBuffer:attention_query_normalized offset:0 atIndex:0];
                [encoder setBuffer:qsa_key_weight offset:0 atIndex:1];
                [encoder setBuffer:qsa_key_scale offset:0 atIndex:2];
                [encoder setBuffer:qsa_key_bias offset:0 atIndex:3];
                [encoder setBuffer:qsa_value_weight offset:0 atIndex:4];
                [encoder setBuffer:qsa_value_scale offset:0 atIndex:5];
                [encoder setBuffer:qsa_value_bias offset:0 atIndex:6];
                [encoder setBuffer:qsa_selected offset:0 atIndex:7];
                [encoder setBuffer:qsa_attention_output offset:0 atIndex:8];
                [encoder setBytes:&qsa_tokens length:sizeof(qsa_tokens) atIndex:9];
                [encoder setBuffer:attention_hot_keys offset:0 atIndex:10];
                [encoder setBuffer:attention_hot_values offset:0 atIndex:11];
                const std::uint32_t one = 1;
                [encoder setBytes:&one length:sizeof(one) atIndex:12];
                [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:13];
                const std::uint32_t selected_count = 512;
                [encoder setBytes:&selected_count length:sizeof(selected_count) atIndex:14];
                [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(384, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_gate_state];
                [encoder setBuffer:qsa_attention_output offset:0 atIndex:0];
                [encoder setBuffer:attention_projection_output offset:0 atIndex:1];
                [encoder setBuffer:attention_gated offset:0 atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(6144, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_output_state];
                [encoder setBuffer:attention_gated offset:0 atIndex:0];
                bind_projection(encoder, shared_attention_shard, shared_attention_output, 1);
                [encoder setBuffer:attention_block_output offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_write_state];
                [encoder setBuffer:hc_stream offset:0 atIndex:0];
                [encoder setBuffer:attention_block_output offset:0 atIndex:1];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:2];
                [encoder setBuffer:attention_output_stream offset:0 atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(10240, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                encode_shared_only_mlp(
                    command, attention_output_stream, shared_attention_aux_shard,
                    shared_attention_mlp_shard, shared_attention_mlp_gate,
                    shared_attention_mlp_up, shared_attention_mlp_down,
                    shared_attention_mlp_router, shared_attention_mlp_hc_norm,
                    shared_attention_mlp_hc_down, shared_attention_mlp_hc_up,
                    shared_attention_mlp_hc_injection, shared_attention_healing_left,
                    shared_attention_healing_right);
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error(command.error.localizedDescription.UTF8String);
                return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            };
            const double shared_attention_first_gpu = run_shared_attention_layer();
            const OracleResult shared_attention_oracle =
                mlx_full_layer_oracle(manifest, stream_bf16, qsa_pooled_values, qkw, qks, qkb,
                                      qvw, qvs, qvb, qsa_tokens, 11);
            const auto *shared_attention_bits =
                static_cast<const std::uint16_t *>(hc_output_stream.contents);
            double shared_attention_dot = 0.0, shared_attention_aa = 0.0;
            double shared_attention_bb = 0.0, shared_attention_squared_error = 0.0;
            double shared_attention_max_abs = 0.0;
            for (std::size_t component = 0; component < 10240; ++component) {
                const double actual = from_bf16(shared_attention_bits[component]);
                const double expected = shared_attention_oracle.output[component];
                const double delta = actual - expected;
                shared_attention_dot += actual * expected;
                shared_attention_aa += actual * actual;
                shared_attention_bb += expected * expected;
                shared_attention_squared_error += delta * delta;
                shared_attention_max_abs =
                    std::max(shared_attention_max_abs, std::abs(delta));
            }
            for (int warmup = 0; warmup < 5; ++warmup)
                static_cast<void>(run_shared_attention_layer());
            std::vector<double> shared_attention_gpu;
            for (int iteration = 0; iteration < 31; ++iteration) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(iteration + 241));
                shared_attention_gpu.push_back(run_shared_attention_layer());
            }

            const auto run_attention_half = [&](const bool include_mlp,
                                                const std::uint32_t state_hot_index = 0,
                                                const std::uint32_t state_pending_index = 0,
                                                const std::uint32_t state_block_count = qsa_blocks,
                                                const std::uint32_t state_hot_count = 1,
                                                const std::uint32_t state_complete_block = 0) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_normalize_state];
                [encoder setBuffer:hc_stream offset:0 atIndex:0];
                bind_tensor(encoder, router_shard, attention_hc_norm, 1);
                [encoder setBuffer:hc_normalized offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_down_state];
                [encoder setBuffer:hc_normalized offset:0 atIndex:0];
                bind_projection(encoder, router_shard, attention_hc_down, 1);
                bind_projection(encoder, router_shard, attention_hc_injection, 4);
                [encoder setBuffer:hc_activation offset:0 atIndex:7];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:8];
                [encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_up_state];
                [encoder setBuffer:hc_normalized offset:0 atIndex:0];
                [encoder setBuffer:hc_activation offset:0 atIndex:1];
                bind_projection(encoder, router_shard, attention_hc_up, 2);
                [encoder setBuffer:hc_mixed offset:0 atIndex:5];
                [encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_projection_state];
                [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                bind_projection(encoder, router_shard, attention_index, 1);
                bind_projection(encoder, router_shard, attention_query, 4);
                bind_projection(encoder, router_shard, attention_key, 7);
                bind_projection(encoder, router_shard, attention_value, 10);
                [encoder setBuffer:attention_projection_output offset:0 atIndex:13];
                [encoder dispatchThreadgroups:MTLSizeMake((13952 + 3) / 4, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_normalize_state];
                [encoder setBuffer:attention_projection_output offset:0 atIndex:0];
                bind_tensor(encoder, router_shard, attention_query_norm, 1);
                bind_tensor(encoder, router_shard, attention_key_norm, 2);
                bind_tensor(encoder, router_shard, attention_index_norm, 3);
                [encoder setBuffer:rope_cos offset:0 atIndex:4];
                [encoder setBuffer:rope_sin offset:0 atIndex:5];
                [encoder setBuffer:attention_selector_query offset:0 atIndex:6];
                [encoder setBuffer:attention_query_normalized offset:0 atIndex:7];
                [encoder setBuffer:attention_key_normalized offset:0 atIndex:8];
                [encoder dispatchThreadgroups:MTLSizeMake(30, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_append_state];
                [encoder setBuffer:attention_projection_output offset:0 atIndex:0];
                [encoder setBuffer:attention_key_normalized offset:0 atIndex:1];
                [encoder setBuffer:attention_hot_keys offset:0 atIndex:2];
                [encoder setBuffer:attention_hot_values offset:0 atIndex:3];
                [encoder setBuffer:qsa_pending_raw offset:0 atIndex:4];
                [encoder setBuffer:qsa_pooled offset:0 atIndex:5];
                bind_tensor(encoder, router_shard, attention_index_norm, 6);
                [encoder setBuffer:pool_rope_cos offset:0 atIndex:7];
                [encoder setBuffer:pool_rope_sin offset:0 atIndex:8];
                [encoder setBytes:&state_hot_index length:sizeof(state_hot_index) atIndex:9];
                [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:10];
                [encoder setBytes:&state_pending_index length:sizeof(state_pending_index) atIndex:11];
                [encoder setBytes:&qsa_blocks length:sizeof(qsa_blocks) atIndex:12];
                [encoder setBytes:&state_complete_block length:sizeof(state_complete_block)
                           atIndex:13];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_score_state];
                [encoder setBuffer:attention_selector_query offset:0 atIndex:0];
                [encoder setBuffer:qsa_pooled offset:0 atIndex:1];
                [encoder setBuffer:qsa_scores offset:0 atIndex:2];
                const std::uint32_t state_padded_count =
                    (state_block_count + 255) / 256 * 256;
                [encoder setBytes:&state_block_count length:sizeof(state_block_count) atIndex:3];
                [encoder setBytes:&state_padded_count length:sizeof(state_padded_count) atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(state_padded_count / 4, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_first_state];
                [encoder setBuffer:qsa_scores offset:0 atIndex:0];
                [encoder setBuffer:qsa_temp_scores_a offset:0 atIndex:1];
                [encoder setBuffer:qsa_temp_ids_a offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(state_padded_count / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];
                std::uint32_t selection_groups = state_padded_count / 256;
                bool selection_source_a = true;
                while (selection_groups > 1) {
                    const std::uint32_t output_groups = (selection_groups + 1) / 2;
                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:qsa_merge_state];
                    [encoder setBuffer:(selection_source_a ? qsa_temp_scores_a : qsa_temp_scores_b)
                                offset:0
                               atIndex:0];
                    [encoder setBuffer:(selection_source_a ? qsa_temp_ids_a : qsa_temp_ids_b)
                                offset:0
                               atIndex:1];
                    [encoder setBuffer:(selection_source_a ? qsa_temp_scores_b : qsa_temp_scores_a)
                                offset:0
                               atIndex:2];
                    [encoder setBuffer:(output_groups == 1 ? qsa_selected
                                                            : (selection_source_a ? qsa_temp_ids_b
                                                                                  : qsa_temp_ids_a))
                                offset:0
                               atIndex:3];
                    [encoder setBytes:&selection_groups length:sizeof(selection_groups) atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(output_groups, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];
                    selection_groups = output_groups;
                    selection_source_a = !selection_source_a;
                }

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_attention_state];
                [encoder setBuffer:attention_query_normalized offset:0 atIndex:0];
                [encoder setBuffer:qsa_key_weight offset:0 atIndex:1];
                [encoder setBuffer:qsa_key_scale offset:0 atIndex:2];
                [encoder setBuffer:qsa_key_bias offset:0 atIndex:3];
                [encoder setBuffer:qsa_value_weight offset:0 atIndex:4];
                [encoder setBuffer:qsa_value_scale offset:0 atIndex:5];
                [encoder setBuffer:qsa_value_bias offset:0 atIndex:6];
                [encoder setBuffer:qsa_selected offset:0 atIndex:7];
                [encoder setBuffer:qsa_attention_output offset:0 atIndex:8];
                [encoder setBytes:&qsa_tokens length:sizeof(qsa_tokens) atIndex:9];
                [encoder setBuffer:attention_hot_keys offset:0 atIndex:10];
                [encoder setBuffer:attention_hot_values offset:0 atIndex:11];
                [encoder setBytes:&state_hot_count length:sizeof(state_hot_count) atIndex:12];
                [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:13];
                const std::uint32_t selected_count = 512;
                [encoder setBytes:&selected_count length:sizeof(selected_count) atIndex:14];
                [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(384, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_gate_state];
                [encoder setBuffer:qsa_attention_output offset:0 atIndex:0];
                [encoder setBuffer:attention_projection_output offset:0 atIndex:1];
                [encoder setBuffer:attention_gated offset:0 atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(6144, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:attention_output_state];
                [encoder setBuffer:attention_gated offset:0 atIndex:0];
                bind_projection(encoder, router_shard, attention_output_projection, 1);
                [encoder setBuffer:attention_block_output offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [encoder endEncoding];

                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:hc_write_state];
                [encoder setBuffer:hc_stream offset:0 atIndex:0];
                [encoder setBuffer:attention_block_output offset:0 atIndex:1];
                [encoder setBuffer:hc_injection_output offset:0 atIndex:2];
                [encoder setBuffer:attention_output_stream offset:0 atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(10240, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];

                if (include_mlp) {
                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:hc_normalize_state];
                    [encoder setBuffer:attention_output_stream offset:0 atIndex:0];
                    bind_tensor(encoder, router_shard, hc_norm, 1);
                    [encoder setBuffer:hc_normalized offset:0 atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:hc_down_state];
                    [encoder setBuffer:hc_normalized offset:0 atIndex:0];
                    bind_projection(encoder, router_shard, hc_down, 1);
                    bind_projection(encoder, router_shard, hc_injection, 4);
                    [encoder setBuffer:hc_activation offset:0 atIndex:7];
                    [encoder setBuffer:hc_injection_output offset:0 atIndex:8];
                    [encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:hc_up_state];
                    [encoder setBuffer:hc_normalized offset:0 atIndex:0];
                    [encoder setBuffer:hc_activation offset:0 atIndex:1];
                    bind_projection(encoder, router_shard, hc_up, 2);
                    [encoder setBuffer:hc_mixed offset:0 atIndex:5];
                    [encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:router_state];
                    [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                    bind_projection(encoder, router_shard, device_router, 1);
                    [encoder setBuffer:logits offset:0 atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:select_state];
                    [encoder setBuffer:logits offset:0 atIndex:0];
                    [encoder setBuffer:experts offset:0 atIndex:1];
                    [encoder setBuffer:weights offset:0 atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:fused_gate_state];
                    [encoder setBuffer:hc_mixed offset:0 atIndex:0];
                    bind_projection(encoder, shard, gate, 1);
                    bind_projection(encoder, shard, up, 4);
                    [encoder setBuffer:experts offset:0 atIndex:7];
                    [encoder setBuffer:hidden offset:0 atIndex:8];
                    bind_projection(encoder, shard, shared_gate, 9);
                    bind_projection(encoder, shard, shared_up, 12);
                    [encoder setBuffer:shared_hidden offset:0 atIndex:15];
                    bind_projection(encoder, router_shard, shared_router, 16);
                    [encoder setBuffer:shared_router_output offset:0 atIndex:19];
                    [encoder dispatchThreadgroups:MTLSizeMake(1281, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:fused_down_state];
                    [encoder setBuffer:hidden offset:0 atIndex:0];
                    bind_projection(encoder, shard, down, 1);
                    [encoder setBuffer:experts offset:0 atIndex:4];
                    [encoder setBuffer:weights offset:0 atIndex:5];
                    [encoder setBuffer:shared_hidden offset:0 atIndex:6];
                    bind_projection(encoder, shard, shared_down, 7);
                    [encoder setBuffer:shared_router_output offset:0 atIndex:10];
                    [encoder setBuffer:full_output offset:0 atIndex:11];
                    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:healing_left_state];
                    [encoder setBuffer:full_output offset:0 atIndex:0];
                    bind_tensor(encoder, healing_shard, healing_left, 1);
                    [encoder setBuffer:healing_hidden offset:0 atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];

                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:healing_right_state];
                    [encoder setBuffer:full_output offset:0 atIndex:0];
                    [encoder setBuffer:healing_hidden offset:0 atIndex:1];
                    bind_tensor(encoder, healing_shard, healing_right, 2);
                    [encoder setBuffer:attention_output_stream offset:0 atIndex:3];
                    [encoder setBuffer:hc_injection_output offset:0 atIndex:4];
                    [encoder setBuffer:hc_output_stream offset:0 atIndex:5];
                    [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [encoder endEncoding];
                }
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error(command.error.localizedDescription.UTF8String);
                return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            };

            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_attention_half(false));
            std::vector<double> attention_half_gpu, attention_half_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 101));
                const auto started = std::chrono::steady_clock::now();
                attention_half_gpu.push_back(run_attention_half(false));
                attention_half_wall.push_back(std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() - started)
                                                  .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_attention_half(true));
            std::vector<double> persistent_layer_gpu, persistent_layer_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 137));
                const auto started = std::chrono::steady_clock::now();
                persistent_layer_gpu.push_back(run_attention_half(true));
                persistent_layer_wall.push_back(std::chrono::duration<double, std::milli>(
                                                    std::chrono::steady_clock::now() - started)
                                                    .count());
            }
            std::uint64_t persistent_layer_hash = 1469598103934665603ULL;
            const auto *persistent_layer_bits =
                static_cast<const std::uint16_t *>(hc_output_stream.contents);
            std::vector<float> persistent_layer_output(10240);
            for (int i = 0; i < 10240; ++i) {
                if (!std::isfinite(from_bf16(persistent_layer_bits[i])))
                    throw std::runtime_error("non-finite persistent layer output");
                persistent_layer_hash ^= persistent_layer_bits[i];
                persistent_layer_hash *= 1099511628211ULL;
                persistent_layer_output[static_cast<std::size_t>(i)] =
                    from_bf16(persistent_layer_bits[i]);
            }

            std::vector<double> state_append_gpu;
            for (std::uint32_t step = 0; step < 4; ++step) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:qsa_append_state];
                [encoder setBuffer:attention_projection_output offset:0 atIndex:0];
                [encoder setBuffer:attention_key_normalized offset:0 atIndex:1];
                [encoder setBuffer:attention_hot_keys offset:0 atIndex:2];
                [encoder setBuffer:attention_hot_values offset:0 atIndex:3];
                [encoder setBuffer:qsa_pending_raw offset:0 atIndex:4];
                [encoder setBuffer:qsa_pooled offset:0 atIndex:5];
                bind_tensor(encoder, router_shard, attention_index_norm, 6);
                [encoder setBuffer:rope_cos offset:0 atIndex:7];
                [encoder setBuffer:rope_sin offset:0 atIndex:8];
                [encoder setBytes:&step length:sizeof(step) atIndex:9];
                [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:10];
                [encoder setBytes:&step length:sizeof(step) atIndex:11];
                [encoder setBytes:&qsa_blocks length:sizeof(qsa_blocks) atIndex:12];
                const std::uint32_t complete_block = step == 3 ? 1 : 0;
                [encoder setBytes:&complete_block length:sizeof(complete_block) atIndex:13];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error(command.error.localizedDescription.UTF8String);
                state_append_gpu.push_back((command.GPUEndTime - command.GPUStartTime) * 1000.0);
            }
            const auto *projected_bits =
                static_cast<const std::uint16_t *>(attention_projection_output.contents);
            const auto *normalized_key_bits =
                static_cast<const std::uint16_t *>(attention_key_normalized.contents);
            const auto *hot_key_bits =
                static_cast<const std::uint16_t *>(attention_hot_keys.contents);
            const auto *hot_value_bits =
                static_cast<const std::uint16_t *>(attention_hot_values.contents);
            const auto *pending_bits =
                static_cast<const std::uint16_t *>(qsa_pending_raw.contents);
            double state_hot_max_abs = 0.0, state_pending_max_abs = 0.0;
            for (std::uint32_t step = 0; step < 4; ++step) {
                for (std::uint32_t head = 0; head < 2; ++head) {
                    for (std::uint32_t component = 0; component < 256; ++component) {
                        const std::size_t target =
                            (static_cast<std::size_t>(head) * hot_capacity + step) * 256 + component;
                        const std::size_t source = head * 256 + component;
                        state_hot_max_abs = std::max(
                            state_hot_max_abs,
                            std::abs(static_cast<double>(from_bf16(hot_key_bits[target]) -
                                                         from_bf16(normalized_key_bits[source]))));
                        state_hot_max_abs = std::max(
                            state_hot_max_abs,
                            std::abs(static_cast<double>(from_bf16(hot_value_bits[target]) -
                                                         from_bf16(projected_bits[13440 + source]))));
                    }
                }
                for (std::uint32_t component = 0; component < 128; ++component)
                    state_pending_max_abs = std::max(
                        state_pending_max_abs,
                        std::abs(static_cast<double>(from_bf16(pending_bits[step * 128 + component]) -
                                                     from_bf16(projected_bits[512 + component]))));
            }
            const auto index_norm_tensor = router_shard.file.tensor(attention_index_norm);
            const auto *index_norm_bits =
                reinterpret_cast<const std::uint16_t *>(index_norm_tensor.bytes.data());
            float state_square_sum = 0.0F;
            for (std::uint32_t component = 0; component < 128; ++component) {
                const float value = from_bf16(projected_bits[512 + component]);
                state_square_sum += value * value;
            }
            const float state_inverse_rms = std::sqrt(128.0F / (state_square_sum + 128.0e-6F));
            std::array<float, 128> expected_pool{};
            for (std::uint32_t component = 0; component < 128; ++component)
                expected_pool[component] = from_bf16(bf16(
                    from_bf16(projected_bits[512 + component]) * state_inverse_rms *
                    (from_bf16(index_norm_bits[component]) + 1.0F)));
            for (std::uint32_t component = 0; component < 32; ++component) {
                const float first = expected_pool[component];
                const float second = expected_pool[component + 32];
                expected_pool[component] = from_bf16(bf16(
                    first * from_bf16(rope_cos_values[component]) -
                    second * from_bf16(rope_sin_values[component])));
                expected_pool[component + 32] = from_bf16(bf16(
                    second * from_bf16(rope_cos_values[component + 32]) +
                    first * from_bf16(rope_sin_values[component + 32])));
            }
            const auto *pooled_bits = static_cast<const std::uint16_t *>(qsa_pooled.contents);
            double state_pool_max_abs = 0.0;
            for (std::uint32_t component = 0; component < 128; ++component)
                state_pool_max_abs = std::max(
                    state_pool_max_abs,
                    std::abs(static_cast<double>(
                        from_bf16(pooled_bits[static_cast<std::size_t>(qsa_blocks) * 128 + component]) -
                        expected_pool[component])));

            static_cast<void>(run_qsa_selector(
                queue, qsa_score_state, qsa_first_state, qsa_merge_state, qsa_query, qsa_pooled,
                qsa_scores, qsa_temp_scores_a, qsa_temp_ids_a, qsa_temp_scores_b,
                qsa_temp_ids_b, qsa_selected, qsa_blocks + 1));
            const auto *padded_selected =
                static_cast<const std::uint32_t *>(qsa_selected.contents);
            bool qsa_padded_selection_valid = true;
            for (std::uint32_t slot = 0; slot < 128; ++slot)
                qsa_padded_selection_valid &= padded_selected[slot] < qsa_blocks + 1;
            std::vector<std::pair<float, std::uint32_t>> padded_cpu_scores;
            padded_cpu_scores.reserve(qsa_blocks + 1);
            for (std::uint32_t block = 0; block < qsa_blocks + 1; ++block) {
                float score = 0.0F;
                for (std::uint32_t head = 0; head < 4; ++head) {
                    float dot = 0.0F;
                    for (std::uint32_t component = 0; component < 128; ++component)
                        dot += from_bf16(qsa_query_values[head * 128 + component]) *
                               from_bf16(pooled_bits[static_cast<std::size_t>(block) * 128 +
                                                     component]);
                    score += std::max(dot, 0.0F);
                }
                padded_cpu_scores.emplace_back(score - static_cast<float>(block) * 1.0e-7F,
                                               block);
            }
            std::ranges::sort(padded_cpu_scores, std::greater{},
                              &std::pair<float, std::uint32_t>::first);
            std::vector<std::uint32_t> expected_padded(128), actual_padded(128);
            for (std::size_t slot = 0; slot < 128; ++slot) {
                expected_padded[slot] = padded_cpu_scores[slot].second;
                actual_padded[slot] = padded_selected[slot];
            }
            std::ranges::sort(expected_padded);
            std::ranges::sort(actual_padded);
            const bool qsa_padded_selected_match = expected_padded == actual_padded;

            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_qsa_selector(
                    queue, qsa_score_state, qsa_first_state, qsa_merge_state, qsa_query, qsa_pooled,
                    qsa_scores, qsa_temp_scores_a, qsa_temp_ids_a, qsa_temp_scores_b,
                    qsa_temp_ids_b, qsa_selected, qsa_blocks));
            std::vector<double> qsa_selector_gpu, qsa_selector_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 11));
                const auto started = std::chrono::steady_clock::now();
                qsa_selector_gpu.push_back(run_qsa_selector(
                    queue, qsa_score_state, qsa_first_state, qsa_merge_state, qsa_query, qsa_pooled,
                    qsa_scores, qsa_temp_scores_a, qsa_temp_ids_a, qsa_temp_scores_b,
                    qsa_temp_ids_b, qsa_selected, qsa_blocks));
                qsa_selector_wall.push_back(std::chrono::duration<double, std::milli>(
                                                std::chrono::steady_clock::now() - started)
                                                .count());
            }
            std::vector<std::pair<float, std::uint32_t>> cpu_qsa_scores;
            cpu_qsa_scores.reserve(qsa_blocks);
            for (std::uint32_t block = 0; block < qsa_blocks; ++block) {
                float score = 0.0F;
                for (std::uint32_t head = 0; head < 4; ++head) {
                    float dot = 0.0F;
                    for (std::uint32_t component = 0; component < 128; ++component) {
                        dot += from_bf16(qsa_query_values[head * 128 + component]) *
                               from_bf16(qsa_pooled_values[static_cast<std::size_t>(block) * 128 +
                                                           component]);
                    }
                    score += std::max(dot, 0.0F);
                }
                cpu_qsa_scores.emplace_back(score - static_cast<float>(block) * 1.0e-7F, block);
            }
            std::ranges::sort(cpu_qsa_scores, std::greater{},
                              &std::pair<float, std::uint32_t>::first);
            std::vector<std::uint32_t> expected_qsa(128);
            for (std::size_t i = 0; i < expected_qsa.size(); ++i)
                expected_qsa[i] = cpu_qsa_scores[i].second;
            std::ranges::sort(expected_qsa);
            const auto *device_qsa = static_cast<const std::uint32_t *>(qsa_selected.contents);
            std::vector<std::uint32_t> actual_qsa(device_qsa, device_qsa + 128);
            std::ranges::sort(actual_qsa);
            const bool qsa_selected_match = expected_qsa == actual_qsa;
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_qsa_attention(
                    queue, qsa_attention_state, qsa_attention_query, qsa_key_weight, qsa_key_scale,
                    qsa_key_bias, qsa_value_weight, qsa_value_scale, qsa_value_bias, qsa_selected,
                    qsa_attention_output, qsa_tokens));
            std::vector<double> qsa_attention_gpu, qsa_attention_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 43));
                const auto started = std::chrono::steady_clock::now();
                qsa_attention_gpu.push_back(run_qsa_attention(
                    queue, qsa_attention_state, qsa_attention_query, qsa_key_weight, qsa_key_scale,
                    qsa_key_bias, qsa_value_weight, qsa_value_scale, qsa_value_bias, qsa_selected,
                    qsa_attention_output, qsa_tokens));
                qsa_attention_wall.push_back(std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_qsa_selected_attention(
                    queue, qsa_score_state, qsa_first_state, qsa_merge_state, qsa_attention_state,
                    qsa_query, qsa_pooled, qsa_scores, qsa_temp_scores_a, qsa_temp_ids_a,
                    qsa_temp_scores_b, qsa_temp_ids_b, qsa_selected, qsa_attention_query,
                    qsa_key_weight, qsa_key_scale, qsa_key_bias, qsa_value_weight, qsa_value_scale,
                    qsa_value_bias, qsa_attention_output, qsa_blocks, qsa_tokens));
            std::vector<double> qsa_pipeline_gpu, qsa_pipeline_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 89));
                const auto started = std::chrono::steady_clock::now();
                qsa_pipeline_gpu.push_back(run_qsa_selected_attention(
                    queue, qsa_score_state, qsa_first_state, qsa_merge_state, qsa_attention_state,
                    qsa_query, qsa_pooled, qsa_scores, qsa_temp_scores_a, qsa_temp_ids_a,
                    qsa_temp_scores_b, qsa_temp_ids_b, qsa_selected, qsa_attention_query,
                    qsa_key_weight, qsa_key_scale, qsa_key_bias, qsa_value_weight, qsa_value_scale,
                    qsa_value_bias, qsa_attention_output, qsa_blocks, qsa_tokens));
                qsa_pipeline_wall.push_back(std::chrono::duration<double, std::milli>(
                                                std::chrono::steady_clock::now() - started)
                                                .count());
            }
            std::uint64_t qsa_attention_hash = 1469598103934665603ULL;
            for (const auto *bits =
                     static_cast<const std::uint16_t *>(qsa_attention_output.contents);
                 bits !=
                 static_cast<const std::uint16_t *>(qsa_attention_output.contents) + 24 * 256;
                 ++bits) {
                if (!std::isfinite(from_bf16(*bits)))
                    throw std::runtime_error("non-finite QSA attention output");
                qsa_attention_hash ^= *bits;
                qsa_attention_hash *= 1099511628211ULL;
            }
            std::array<float, 256> qsa_attention_reference{};
            float attention_running_max = -std::numeric_limits<float>::infinity();
            float attention_running_sum = 0.0F;
            for (std::size_t slot = 0; slot < 512; ++slot) {
                const std::uint32_t token = device_qsa[slot / 4] * 4 + slot % 4;
                float score = 0.0F;
                std::array<float, 256> value{};
                for (std::size_t component = 0; component < 256; ++component) {
                    const std::size_t packed_channel = component / 4;
                    const std::size_t word_index =
                        static_cast<std::size_t>(token) * 64 + packed_channel;
                    const std::size_t affine_index =
                        static_cast<std::size_t>(token) * 4 + component / 64;
                    const float key = from_bf16(
                        bf16(static_cast<float>((qkw[word_index] >> ((component % 4) * 8)) & 255U) *
                                 from_bf16(qks[affine_index]) +
                             from_bf16(qkb[affine_index])));
                    value[component] = from_bf16(
                        bf16(static_cast<float>((qvw[word_index] >> ((component % 4) * 8)) & 255U) *
                                 from_bf16(qvs[affine_index]) +
                             from_bf16(qvb[affine_index])));
                    score += from_bf16(qsa_attention_query_values[component]) * key;
                }
                score *= 0.0625F;
                const float next_max = std::max(attention_running_max, score);
                const float alpha = std::isfinite(attention_running_max)
                                        ? std::exp(attention_running_max - next_max)
                                        : 0.0F;
                const float probability = std::exp(score - next_max);
                attention_running_sum = attention_running_sum * alpha + probability;
                for (std::size_t component = 0; component < 256; ++component) {
                    qsa_attention_reference[component] =
                        qsa_attention_reference[component] * alpha + probability * value[component];
                }
                attention_running_max = next_max;
            }
            const auto *qsa_attention_bits =
                static_cast<const std::uint16_t *>(qsa_attention_output.contents);
            double qsa_attention_dot = 0.0, qsa_attention_aa = 0.0, qsa_attention_bb = 0.0;
            double qsa_attention_max_abs = 0.0;
            for (std::size_t component = 0; component < 256; ++component) {
                const double actual = from_bf16(qsa_attention_bits[component]);
                const double expected = qsa_attention_reference[component] / attention_running_sum;
                qsa_attention_dot += actual * expected;
                qsa_attention_aa += actual * actual;
                qsa_attention_bb += expected * expected;
                qsa_attention_max_abs =
                    std::max(qsa_attention_max_abs, std::abs(actual - expected));
            }

            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_hc_read(queue, hc_normalize_state, hc_down_state, hc_up_state,
                                              hc_stream, router_shard, hc_norm, hc_down, hc_up,
                                              hc_injection, hc_normalized, hc_activation,
                                              hc_injection_output, hc_mixed));
            std::vector<double> hc_gpu, hc_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 197));
                const auto started = std::chrono::steady_clock::now();
                hc_gpu.push_back(run_hc_read(queue, hc_normalize_state, hc_down_state, hc_up_state,
                                             hc_stream, router_shard, hc_norm, hc_down, hc_up,
                                             hc_injection, hc_normalized, hc_activation,
                                             hc_injection_output, hc_mixed));
                hc_wall.push_back(std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - started)
                                      .count());
            }

            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_attention_projections(
                    queue, attention_projection_state, hc_mixed, router_shard, attention_index,
                    attention_query, attention_key, attention_value, attention_projection_output));
            std::vector<double> attention_projection_gpu, attention_projection_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 211));
                const auto started = std::chrono::steady_clock::now();
                attention_projection_gpu.push_back(run_attention_projections(
                    queue, attention_projection_state, hc_mixed, router_shard, attention_index,
                    attention_query, attention_key, attention_value, attention_projection_output));
                attention_projection_wall.push_back(std::chrono::duration<double, std::milli>(
                                                        std::chrono::steady_clock::now() - started)
                                                        .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_attention_normalize(
                    queue, attention_normalize_state, attention_projection_output, router_shard,
                    attention_query_norm, attention_key_norm, attention_index_norm, rope_cos,
                    rope_sin, attention_selector_query, attention_query_normalized,
                    attention_key_normalized));
            std::vector<double> attention_normalize_gpu, attention_normalize_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 239));
                const auto started = std::chrono::steady_clock::now();
                attention_normalize_gpu.push_back(run_attention_normalize(
                    queue, attention_normalize_state, attention_projection_output, router_shard,
                    attention_query_norm, attention_key_norm, attention_index_norm, rope_cos,
                    rope_sin, attention_selector_query, attention_query_normalized,
                    attention_key_normalized));
                attention_normalize_wall.push_back(std::chrono::duration<double, std::milli>(
                                                       std::chrono::steady_clock::now() - started)
                                                       .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_attention_output(
                    queue, attention_gate_state, attention_output_state, qsa_attention_output,
                    attention_projection_output, router_shard, attention_output_projection,
                    attention_gated, attention_block_output));
            std::vector<double> attention_output_gpu, attention_output_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 17));
                const auto started = std::chrono::steady_clock::now();
                attention_output_gpu.push_back(run_attention_output(
                    queue, attention_gate_state, attention_output_state, qsa_attention_output,
                    attention_projection_output, router_shard, attention_output_projection,
                    attention_gated, attention_block_output));
                attention_output_wall.push_back(std::chrono::duration<double, std::milli>(
                                                    std::chrono::steady_clock::now() - started)
                                                    .count());
            }

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
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_device_routed_full(
                    queue, router_state, select_state, fused_gate_state, fused_down_state, input,
                    shard, router_shard, device_router, gate, up, down, shared_gate, shared_up,
                    shared_down, shared_router, logits, experts, weights, hidden, shared_hidden,
                    shared_router_output, full_output));
            std::vector<double> device_routed_gpu, device_routed_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 157));
                const auto started = std::chrono::steady_clock::now();
                device_routed_gpu.push_back(run_device_routed_full(
                    queue, router_state, select_state, fused_gate_state, fused_down_state, input,
                    shard, router_shard, device_router, gate, up, down, shared_gate, shared_up,
                    shared_down, shared_router, logits, experts, weights, hidden, shared_hidden,
                    shared_router_output, full_output));
                device_routed_wall.push_back(std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count());
            }
            for (int i = 0; i < 5; ++i)
                static_cast<void>(run_hc_device_routed_full(
                    queue, hc_normalize_state, hc_down_state, hc_up_state, router_state,
                    select_state, fused_gate_state, fused_down_state, healing_left_state,
                    healing_right_state, hc_stream, shard, router_shard, healing_shard, hc_norm,
                    hc_down, hc_up, hc_injection, device_router, gate, up, down, shared_gate,
                    shared_up, shared_down, shared_router, healing_left, healing_right,
                    hc_normalized, hc_activation, hc_injection_output, hc_mixed, logits, experts,
                    weights, hidden, shared_hidden, shared_router_output, full_output,
                    healing_hidden, hc_output_stream));
            std::vector<double> hc_moe_gpu, hc_moe_wall;
            for (int i = 0; i < 31; ++i) {
                evict_device_cache(queue, cache_evict, static_cast<std::uint8_t>(i + 223));
                const auto started = std::chrono::steady_clock::now();
                hc_moe_gpu.push_back(run_hc_device_routed_full(
                    queue, hc_normalize_state, hc_down_state, hc_up_state, router_state,
                    select_state, fused_gate_state, fused_down_state, healing_left_state,
                    healing_right_state, hc_stream, shard, router_shard, healing_shard, hc_norm,
                    hc_down, hc_up, hc_injection, device_router, gate, up, down, shared_gate,
                    shared_up, shared_down, shared_router, healing_left, healing_right,
                    hc_normalized, hc_activation, hc_injection_output, hc_mixed, logits, experts,
                    weights, hidden, shared_hidden, shared_router_output, full_output,
                    healing_hidden, hc_output_stream));
                hc_moe_wall.push_back(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - started)
                                          .count());
            }
            std::array<std::uint32_t, top_k> selected_experts{};
            std::copy_n(static_cast<const std::uint32_t *>(experts.contents), top_k,
                        selected_experts.begin());
            std::array<float, top_k> selected_weights{};
            const auto *selected_bf16 = static_cast<const std::uint16_t *>(weights.contents);
            std::ranges::transform(selected_bf16, selected_bf16 + top_k, selected_weights.begin(),
                                   from_bf16);
            std::vector<float> hc_direct(hidden_size);
            const auto *hc_mixed_bits = static_cast<const std::uint16_t *>(hc_mixed.contents);
            std::ranges::transform(hc_mixed_bits, hc_mixed_bits + hidden_size, hc_direct.begin(),
                                   from_bf16);
            const std::vector<float> attention_projection_oracle =
                mlx_attention_projection_oracle(manifest, attention_index, attention_query,
                                                attention_key, attention_value, hc_direct);
            const auto *attention_projection_bits =
                static_cast<const std::uint16_t *>(attention_projection_output.contents);
            double attention_projection_dot = 0.0, attention_projection_aa = 0.0;
            double attention_projection_bb = 0.0, attention_projection_squared_error = 0.0;
            double attention_projection_max_abs = 0.0;
            for (std::size_t i = 0; i < attention_projection_oracle.size(); ++i) {
                const double actual = from_bf16(attention_projection_bits[i]);
                const double expected = attention_projection_oracle[i];
                const double delta = actual - expected;
                attention_projection_dot += actual * expected;
                attention_projection_aa += actual * actual;
                attention_projection_bb += expected * expected;
                attention_projection_squared_error += delta * delta;
                attention_projection_max_abs =
                    std::max(attention_projection_max_abs, std::abs(delta));
            }
            const auto *result = static_cast<const std::uint16_t *>(full_output.contents);
            std::uint64_t hash = 1469598103934665603ULL;
            std::vector<float> direct(hidden_size);
            for (int i = 0; i < hidden_size; ++i) {
                hash ^= result[i];
                hash *= 1099511628211ULL;
                direct[i] = std::bit_cast<float>(static_cast<std::uint32_t>(result[i]) << 16);
            }
            bool router_ids_match = false;
            double router_weight_max_abs = 0.0;
            {
                qwen38::MlxTensorStore routing_store(manifest);
                qwen38::SparseMoe routing_moe(
                    routing_store, layer_prefix, manifest.config().expert_count,
                    manifest.config().experts_per_token, manifest.config().quantization_bits,
                    manifest.config().quantization_group_size,
                    manifest.config().normalize_topk_probability);
                const auto routing_input =
                    MlxArray::from_float32(hc_direct, std::array<int, 3>{1, 1, hidden_size})
                        .astype(MLX_BFLOAT16);
                const auto expected = routing_moe.route_decode(routing_input);
                router_ids_match = std::equal(expected.experts.begin(), expected.experts.end(),
                                              selected_experts.begin());
                for (int slot = 0; slot < top_k; ++slot) {
                    const float rounded = from_bf16(bf16(expected.weights[slot]));
                    router_weight_max_abs =
                        std::max(router_weight_max_abs,
                                 std::abs(static_cast<double>(rounded - selected_weights[slot])));
                }
            }
            const OracleResult oracle =
                mlx_oracle(manifest, gate, up, down, shared_gate, shared_up, shared_down,
                           shared_router, hc_direct, selected_experts, selected_weights);
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
            const HcOracleResult hc_oracle = mlx_hc_oracle(manifest, hc_prefix, stream_f32);
            double hc_dot = 0.0, hc_aa = 0.0, hc_bb = 0.0, hc_squared_error = 0.0;
            double hc_max_abs = 0.0, hc_injection_max_abs = 0.0;
            for (int i = 0; i < hidden_size; ++i) {
                const double a = from_bf16(hc_mixed_bits[i]);
                const double b = hc_oracle.mixed[static_cast<std::size_t>(i)];
                const double delta = a - b;
                hc_dot += a * b;
                hc_aa += a * a;
                hc_bb += b * b;
                hc_squared_error += delta * delta;
                hc_max_abs = std::max(hc_max_abs, std::abs(delta));
            }
            const auto *hc_injection_bits =
                static_cast<const std::uint16_t *>(hc_injection_output.contents);
            for (int i = 0; i < 4; ++i) {
                hc_injection_max_abs = std::max(
                    hc_injection_max_abs, std::abs(static_cast<double>(
                                              from_bf16(hc_injection_bits[i]) -
                                              hc_oracle.injection[static_cast<std::size_t>(i)])));
            }
            const OracleResult mlp_oracle =
                mlx_mlp_half_oracle(manifest, layer_prefix, hc_prefix, stream_f32);
            const auto *mlp_direct_bits =
                static_cast<const std::uint16_t *>(hc_output_stream.contents);
            double mlp_dot = 0.0, mlp_aa = 0.0, mlp_bb = 0.0, mlp_squared_error = 0.0;
            double mlp_max_abs = 0.0;
            for (int i = 0; i < 10240; ++i) {
                const double a = from_bf16(mlp_direct_bits[i]);
                const double b = mlp_oracle.output[static_cast<std::size_t>(i)];
                const double delta = a - b;
                mlp_dot += a * b;
                mlp_aa += a * a;
                mlp_bb += b * b;
                mlp_squared_error += delta * delta;
                mlp_max_abs = std::max(mlp_max_abs, std::abs(delta));
            }
            const OracleResult layer_oracle = mlx_full_layer_oracle(
                manifest, stream_bf16, qsa_pooled_values, qkw, qks, qkb, qvw, qvs, qvb, qsa_tokens);
            double layer_dot = 0.0, layer_aa = 0.0, layer_bb = 0.0;
            double layer_squared_error = 0.0, layer_max_abs = 0.0;
            for (int i = 0; i < 10240; ++i) {
                const double actual = persistent_layer_output[static_cast<std::size_t>(i)];
                const double expected = layer_oracle.output[static_cast<std::size_t>(i)];
                const double delta = actual - expected;
                layer_dot += actual * expected;
                layer_aa += actual * actual;
                layer_bb += expected * expected;
                layer_squared_error += delta * delta;
                layer_max_abs = std::max(layer_max_abs, std::abs(delta));
            }
            const auto gdn_layer_trajectory_oracle =
                mlx_gdn_layer_trajectory_oracle(manifest, stream_bf16, 4);
            std::memcpy(hc_stream.contents, stream_bf16.data(),
                        stream_bf16.size() * sizeof(std::uint16_t));
            double gdn_layer_trajectory_min_cosine = 1.0;
            double gdn_layer_trajectory_max_rmse = 0.0, gdn_layer_trajectory_max_abs = 0.0;
            for (std::uint32_t step = 0; step < 4; ++step) {
                static_cast<void>(run_gdn_layer(step == 0));
                const auto *actual_bits =
                    static_cast<const std::uint16_t *>(hc_output_stream.contents);
                double dot = 0.0, aa = 0.0, bb = 0.0, squared_error = 0.0;
                for (std::size_t component = 0; component < 10240; ++component) {
                    const double actual = from_bf16(actual_bits[component]);
                    const double expected = gdn_layer_trajectory_oracle[step][component];
                    const double delta = actual - expected;
                    dot += actual * expected;
                    aa += actual * actual;
                    bb += expected * expected;
                    squared_error += delta * delta;
                    gdn_layer_trajectory_max_abs =
                        std::max(gdn_layer_trajectory_max_abs, std::abs(delta));
                }
                gdn_layer_trajectory_min_cosine =
                    std::min(gdn_layer_trajectory_min_cosine, dot / std::sqrt(aa * bb));
                gdn_layer_trajectory_max_rmse = std::max(
                    gdn_layer_trajectory_max_rmse, std::sqrt(squared_error / 10240.0));
                std::memcpy(hc_stream.contents, actual_bits, 10240 * sizeof(std::uint16_t));
            }
            const GdnTrajectoryResult gdn_trajectory_oracle =
                mlx_gdn_trajectory_oracle(manifest, gdn_prefix, input_bf16, 4);
            std::memcpy(input.contents, input_bf16.data(),
                        input_bf16.size() * sizeof(std::uint16_t));
            double gdn_trajectory_min_cosine = 1.0, gdn_trajectory_max_rmse = 0.0;
            double gdn_trajectory_max_abs = 0.0;
            for (std::uint32_t step = 0; step < 4; ++step) {
                static_cast<void>(run_gdn(step == 0));
                const auto *actual_bits =
                    static_cast<const std::uint16_t *>(gdn_block_output.contents);
                double dot = 0.0, aa = 0.0, bb = 0.0, squared_error = 0.0;
                for (std::size_t component = 0; component < 2560; ++component) {
                    const double actual = from_bf16(actual_bits[component]);
                    const double expected = gdn_trajectory_oracle.outputs[step][component];
                    const double delta = actual - expected;
                    dot += actual * expected;
                    aa += actual * actual;
                    bb += expected * expected;
                    squared_error += delta * delta;
                    gdn_trajectory_max_abs =
                        std::max(gdn_trajectory_max_abs, std::abs(delta));
                }
                gdn_trajectory_min_cosine =
                    std::min(gdn_trajectory_min_cosine, dot / std::sqrt(aa * bb));
                gdn_trajectory_max_rmse = std::max(
                    gdn_trajectory_max_rmse, std::sqrt(squared_error / 2560.0));
                std::memcpy(input.contents, actual_bits, 2560 * sizeof(std::uint16_t));
            }
            const auto *gdn_final_conv =
                static_cast<const std::uint16_t *>(gdn_convolution_state.contents);
            const auto *gdn_final_recurrent =
                static_cast<const std::uint16_t *>(gdn_recurrent_state_buffer.contents);
            double gdn_trajectory_conv_max_abs = 0.0, gdn_trajectory_recurrent_max_abs = 0.0;
            for (std::size_t component = 0;
                 component < gdn_trajectory_oracle.convolution.size(); ++component)
                gdn_trajectory_conv_max_abs = std::max(
                    gdn_trajectory_conv_max_abs,
                    std::abs(static_cast<double>(from_bf16(gdn_final_conv[component]) -
                                                 gdn_trajectory_oracle.convolution[component])));
            for (std::size_t component = 0;
                 component < gdn_trajectory_oracle.recurrent.size(); ++component)
                gdn_trajectory_recurrent_max_abs = std::max(
                    gdn_trajectory_recurrent_max_abs,
                    std::abs(static_cast<double>(from_bf16(gdn_final_recurrent[component]) -
                                                 gdn_trajectory_oracle.recurrent[component])));
            const auto trajectory_oracle = mlx_full_layer_trajectory_oracle(
                manifest, stream_bf16, qsa_pooled_values, qkw, qks, qkb, qvw, qvs, qvb,
                qsa_tokens, 4);
            std::memcpy(hc_stream.contents, stream_bf16.data(),
                        stream_bf16.size() * sizeof(std::uint16_t));
            std::vector<double> trajectory_gpu;
            double trajectory_min_cosine = 1.0, trajectory_max_rmse = 0.0;
            double trajectory_max_abs = 0.0;
            for (std::uint32_t step = 0; step < 4; ++step) {
                auto *cos_bits = static_cast<std::uint16_t *>(rope_cos.contents);
                auto *sin_bits = static_cast<std::uint16_t *>(rope_sin.contents);
                for (std::size_t component = 0; component < 32; ++component) {
                    const double frequency =
                        std::pow(10000000.0, -2.0 * static_cast<double>(component) / 64.0);
                    const double angle = static_cast<double>(qsa_tokens + step) * frequency;
                    cos_bits[component] = cos_bits[component + 32] =
                        bf16(static_cast<float>(std::cos(angle)));
                    sin_bits[component] = sin_bits[component + 32] =
                        bf16(static_cast<float>(std::sin(angle)));
                }
                const std::uint32_t complete = step == 3 ? 1 : 0;
                trajectory_gpu.push_back(run_attention_half(
                    true, step, step, qsa_blocks + complete, step + 1, complete));
                const auto *actual_bits =
                    static_cast<const std::uint16_t *>(hc_output_stream.contents);
                double dot = 0.0, aa = 0.0, bb = 0.0, squared_error = 0.0;
                for (std::size_t component = 0; component < 10240; ++component) {
                    const double actual = from_bf16(actual_bits[component]);
                    const double expected = trajectory_oracle[step][component];
                    const double delta = actual - expected;
                    dot += actual * expected;
                    aa += actual * actual;
                    bb += expected * expected;
                    squared_error += delta * delta;
                    trajectory_max_abs = std::max(trajectory_max_abs, std::abs(delta));
                }
                trajectory_min_cosine =
                    std::min(trajectory_min_cosine, dot / std::sqrt(aa * bb));
                trajectory_max_rmse =
                    std::max(trajectory_max_rmse, std::sqrt(squared_error / 10240.0));
                std::memcpy(hc_stream.contents, actual_bits, 10240 * sizeof(std::uint16_t));
            }
            std::cout
                << "{\"device\":\"" << device.name.UTF8String
                << "\",\"layer\":3,\"gdn_layer\":0,\"experts\":10,\"mmap_backed\":true"
                << ",\"joined_gpu_median_ms\":" << median(joined)
                << ",\"gdn_gpu_median_ms\":" << median(gdn_gpu)
                << ",\"gdn_layer_first_gpu_ms\":" << gdn_layer_first_gpu
                << ",\"gdn_layer_gpu_median_ms\":" << median(gdn_layer_gpu)
                << ",\"gdn_layer_oracle_ms\":" << gdn_layer_oracle.median_ms
                << ",\"gdn_layer_cosine\":"
                << gdn_layer_dot / std::sqrt(gdn_layer_aa * gdn_layer_bb)
                << ",\"gdn_layer_rmse\":" << std::sqrt(gdn_layer_squared_error / 10240.0)
                << ",\"gdn_layer_max_abs\":" << gdn_layer_max_abs
                << ",\"gdn_layer_trajectory_tokens\":4"
                << ",\"gdn_layer_trajectory_min_cosine\":"
                << gdn_layer_trajectory_min_cosine
                << ",\"gdn_layer_trajectory_max_rmse\":"
                << gdn_layer_trajectory_max_rmse
                << ",\"gdn_layer_trajectory_max_abs\":"
                << gdn_layer_trajectory_max_abs
                << ",\"shared_gdn_layer_first_gpu_ms\":" << shared_gdn_layer_first_gpu
                << ",\"shared_gdn_layer_gpu_median_ms\":" << median(shared_gdn_layer_gpu)
                << ",\"shared_gdn_layer_oracle_ms\":" << shared_gdn_layer_oracle.median_ms
                << ",\"shared_gdn_layer_cosine\":"
                << shared_gdn_layer_dot /
                       std::sqrt(shared_gdn_layer_aa * shared_gdn_layer_bb)
                << ",\"shared_gdn_layer_rmse\":"
                << std::sqrt(shared_gdn_layer_squared_error / 10240.0)
                << ",\"shared_gdn_layer_max_abs\":" << shared_gdn_layer_max_abs
                << ",\"shared_attention_first_gpu_ms\":" << shared_attention_first_gpu
                << ",\"shared_attention_gpu_median_ms\":" << median(shared_attention_gpu)
                << ",\"shared_attention_oracle_ms\":" << shared_attention_oracle.median_ms
                << ",\"shared_attention_cosine\":"
                << shared_attention_dot /
                       std::sqrt(shared_attention_aa * shared_attention_bb)
                << ",\"shared_attention_rmse\":"
                << std::sqrt(shared_attention_squared_error / 10240.0)
                << ",\"shared_attention_max_abs\":" << shared_attention_max_abs
                << ",\"gdn_projection_max_abs\":" << gdn_projection_max_abs
                << ",\"gdn_projection_nonfinite\":[" << gdn_projection_nonfinite[0] << ','
                << gdn_projection_nonfinite[1] << ',' << gdn_projection_nonfinite[2] << ','
                << gdn_projection_nonfinite[3] << ']'
                << ",\"gdn_direct_nonfinite\":" << gdn_direct_nonfinite
                << ",\"gdn_oracle_nonfinite\":" << gdn_oracle_nonfinite
                << ",\"gdn_cosine\":" << gdn_dot / std::sqrt(gdn_aa * gdn_bb)
                << ",\"gdn_rmse\":" << std::sqrt(gdn_squared_error / 2560.0)
                << ",\"gdn_max_abs\":" << gdn_max_abs
                << ",\"gdn_convolution_max_abs\":" << gdn_conv_max_abs
                << ",\"gdn_recurrent_max_abs\":" << gdn_recurrent_max_abs
                << ",\"gdn_trajectory_tokens\":4"
                << ",\"gdn_trajectory_min_cosine\":" << gdn_trajectory_min_cosine
                << ",\"gdn_trajectory_max_rmse\":" << gdn_trajectory_max_rmse
                << ",\"gdn_trajectory_max_abs\":" << gdn_trajectory_max_abs
                << ",\"gdn_trajectory_convolution_max_abs\":"
                << gdn_trajectory_conv_max_abs
                << ",\"gdn_trajectory_recurrent_max_abs\":"
                << gdn_trajectory_recurrent_max_abs
                << ",\"qsa_selector_gpu_median_ms\":" << median(qsa_selector_gpu)
                << ",\"qsa_selector_wall_median_ms\":" << median(qsa_selector_wall)
                << ",\"qsa_selected_match\":" << (qsa_selected_match ? "true" : "false")
                << ",\"qsa_attention_gpu_median_ms\":" << median(qsa_attention_gpu)
                << ",\"qsa_attention_wall_median_ms\":" << median(qsa_attention_wall)
                << ",\"qsa_attention_hash\":\"" << qsa_attention_hash << "\""
                << ",\"qsa_attention_cosine\":"
                << qsa_attention_dot / std::sqrt(qsa_attention_aa * qsa_attention_bb)
                << ",\"qsa_attention_max_abs\":" << qsa_attention_max_abs
                << ",\"qsa_pipeline_gpu_median_ms\":" << median(qsa_pipeline_gpu)
                << ",\"qsa_pipeline_wall_median_ms\":" << median(qsa_pipeline_wall)
                << ",\"attention_projection_gpu_median_ms\":" << median(attention_projection_gpu)
                << ",\"attention_projection_wall_median_ms\":" << median(attention_projection_wall)
                << ",\"attention_projection_cosine\":"
                << attention_projection_dot /
                       std::sqrt(attention_projection_aa * attention_projection_bb)
                << ",\"attention_projection_rmse\":"
                << std::sqrt(attention_projection_squared_error / 13952.0)
                << ",\"attention_projection_max_abs\":" << attention_projection_max_abs
                << ",\"attention_normalize_gpu_median_ms\":" << median(attention_normalize_gpu)
                << ",\"attention_normalize_wall_median_ms\":" << median(attention_normalize_wall)
                << ",\"attention_output_gpu_median_ms\":" << median(attention_output_gpu)
                << ",\"attention_output_wall_median_ms\":" << median(attention_output_wall)
                << ",\"attention_half_gpu_median_ms\":" << median(attention_half_gpu)
                << ",\"attention_half_wall_median_ms\":" << median(attention_half_wall)
                << ",\"persistent_layer_gpu_median_ms\":" << median(persistent_layer_gpu)
                << ",\"persistent_layer_wall_median_ms\":" << median(persistent_layer_wall)
                << ",\"persistent_layer_hash\":\"" << persistent_layer_hash << "\""
                << ",\"state_append_gpu_median_ms\":" << median(state_append_gpu)
                << ",\"state_hot_max_abs\":" << state_hot_max_abs
                << ",\"state_pending_max_abs\":" << state_pending_max_abs
                << ",\"state_pool_max_abs\":" << state_pool_max_abs
                << ",\"qsa_padded_selection_valid\":"
                << (qsa_padded_selection_valid ? "true" : "false")
                << ",\"qsa_padded_selected_match\":"
                << (qsa_padded_selected_match ? "true" : "false")
                << ",\"trajectory_tokens\":4"
                << ",\"trajectory_gpu_median_ms\":" << median(trajectory_gpu)
                << ",\"trajectory_min_cosine\":" << trajectory_min_cosine
                << ",\"trajectory_max_rmse\":" << trajectory_max_rmse
                << ",\"trajectory_max_abs\":" << trajectory_max_abs
                << ",\"mlx_layer_oracle_ms\":" << layer_oracle.median_ms
                << ",\"persistent_layer_cosine\":" << layer_dot / std::sqrt(layer_aa * layer_bb)
                << ",\"persistent_layer_rmse\":" << std::sqrt(layer_squared_error / 10240.0)
                << ",\"persistent_layer_max_abs\":" << layer_max_abs
                << ",\"split_gpu_median_ms\":" << median(split)
                << ",\"joined_wall_median_ms\":" << median(joined_wall)
                << ",\"split_wall_median_ms\":" << median(split_wall)
                << ",\"full_gpu_median_ms\":" << median(full_gpu)
                << ",\"full_wall_median_ms\":" << median(full_wall)
                << ",\"fused_gpu_median_ms\":" << median(fused_gpu)
                << ",\"fused_wall_median_ms\":" << median(fused_wall)
                << ",\"device_routed_gpu_median_ms\":" << median(device_routed_gpu)
                << ",\"device_routed_wall_median_ms\":" << median(device_routed_wall)
                << ",\"hc_moe_gpu_median_ms\":" << median(hc_moe_gpu)
                << ",\"hc_moe_wall_median_ms\":" << median(hc_moe_wall)
                << ",\"router_ids_match\":" << (router_ids_match ? "true" : "false")
                << ",\"router_weight_max_abs\":" << router_weight_max_abs
                << ",\"hc_gpu_median_ms\":" << median(hc_gpu)
                << ",\"hc_wall_median_ms\":" << median(hc_wall)
                << ",\"mlx_hc_oracle_median_ms\":" << hc_oracle.median_ms
                << ",\"hc_cosine\":" << hc_dot / std::sqrt(hc_aa * hc_bb)
                << ",\"hc_rmse\":" << std::sqrt(hc_squared_error / hidden_size)
                << ",\"hc_max_abs\":" << hc_max_abs
                << ",\"hc_injection_max_abs\":" << hc_injection_max_abs
                << ",\"mlx_mlp_oracle_median_ms\":" << mlp_oracle.median_ms
                << ",\"mlp_cosine\":" << mlp_dot / std::sqrt(mlp_aa * mlp_bb)
                << ",\"mlp_rmse\":" << std::sqrt(mlp_squared_error / 10240.0)
                << ",\"mlp_max_abs\":" << mlp_max_abs
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
