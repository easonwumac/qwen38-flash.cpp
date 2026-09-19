#pragma once

#import <Metal/Metal.h>
#include <cstdint>
#include <stdexcept>

namespace qwen38::persistent_metal {
// Publish the complete top-128 selection, including the single-group case
// (129..256 candidate blocks), which does not execute a merge pass.
inline void encode_qsa_top128(
    id<MTLCommandBuffer> command,
    id<MTLComputePipelineState> first_pipeline, id<MTLComputePipelineState> merge_pipeline,
    id<MTLBuffer> scores, id<MTLBuffer> scores_a, id<MTLBuffer> scores_b,
    id<MTLBuffer> ids_a, id<MTLBuffer> ids_b, id<MTLBuffer> selected,
    std::uint32_t padded_count) {
    if (padded_count < 256 || padded_count > 65536 || padded_count % 256 != 0)
        throw std::invalid_argument("QSA selector padding must be 256..65536 in groups of 256");
    std::uint32_t groups = padded_count / 256;
    if (scores.length < padded_count * sizeof(float) ||
        scores_a.length < groups * 128 * sizeof(float) ||
        scores_b.length < groups * 128 * sizeof(float) ||
        ids_a.length < groups * 128 * sizeof(std::uint32_t) ||
        ids_b.length < groups * 128 * sizeof(std::uint32_t) ||
        selected.length < 128 * sizeof(std::uint32_t))
        throw std::invalid_argument("QSA selector buffer is too small");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:first_pipeline];
    [encoder setBuffer:scores offset:0 atIndex:0];
    [encoder setBuffer:scores_a offset:0 atIndex:1];
    [encoder setBuffer:(groups == 1 ? selected : ids_a) offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    bool source_a = true;
    while (groups > 1) {
        const std::uint32_t output_groups = (groups + 1) / 2;
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:merge_pipeline];
        [encoder setBuffer:(source_a ? scores_a : scores_b) offset:0 atIndex:0];
        [encoder setBuffer:(source_a ? ids_a : ids_b) offset:0 atIndex:1];
        [encoder setBuffer:(source_a ? scores_b : scores_a) offset:0 atIndex:2];
        [encoder setBuffer:(output_groups == 1 ? selected : (source_a ? ids_b : ids_a))
                    offset:0 atIndex:3];
        [encoder setBytes:&groups length:sizeof(groups) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(output_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        groups = output_groups;
        source_a = !source_a;
    }
}
} // namespace qwen38::persistent_metal
