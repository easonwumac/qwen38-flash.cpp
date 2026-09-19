#include "../src/persistent_metal_kernels.hpp"
#include "../src/persistent_qsa_selector.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

int main() {
    @autoreleasepool {
        try {
            const auto device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                std::cout << "Metal device unavailable; skipping GPU selector regression\n";
                return 77;
            }
            const auto queue = [device newCommandQueue];
            std::string source(qwen38::persistent_metal::metal_source_prefix);
            source += qwen38::persistent_metal::metal_source_suffix;
            NSError* error = nil;
            const auto library = [device newLibraryWithSource:
                [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
            if (library == nil) throw std::runtime_error(error.localizedDescription.UTF8String);
            const auto pipeline = [&](NSString* name) {
                auto result = [device newComputePipelineStateWithFunction:
                    [library newFunctionWithName:name] error:&error];
                if (result == nil) throw std::runtime_error(error.localizedDescription.UTF8String);
                return result;
            };
            const auto first = pipeline(@"qsa_top128_first"), merge = pipeline(@"qsa_top128_merge");
            const auto buffer = [&](std::size_t bytes) {
                auto result = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (result == nil) throw std::bad_alloc();
                return result;
            };
            const auto scores = buffer(65536 * sizeof(float));
            const auto scores_a = buffer(32768 * sizeof(float)), scores_b = buffer(32768 * sizeof(float));
            const auto ids_a = buffer(32768 * sizeof(std::uint32_t)), ids_b = buffer(32768 * sizeof(std::uint32_t));
            const auto selected = buffer(128 * sizeof(std::uint32_t));
            for (std::uint32_t count : {129U, 255U, 256U, 257U, 511U, 512U, 513U, 769U, 65536U}) {
                const auto padded = (count + 255) / 256 * 256;
                auto* input = static_cast<float*>(scores.contents);
                std::fill(input, input + padded, -std::numeric_limits<float>::infinity());
                for (std::uint32_t index = 0; index < count; ++index)
                    input[index] = static_cast<float>((index * 107 + 43) % 131071);
                std::vector<std::uint32_t> expected(count);
                std::iota(expected.begin(), expected.end(), 0);
                std::partial_sort(expected.begin(), expected.begin() + 128, expected.end(),
                    [&](auto left, auto right) { return input[left] > input[right]; });
                expected.resize(128);
                std::sort(expected.begin(), expected.end());
                for (int repeat = 0; repeat < 2; ++repeat) {
                    std::memset(selected.contents, 0xff, selected.length);
                    const auto command = [queue commandBuffer];
                    qwen38::persistent_metal::encode_qsa_top128(command, first, merge,
                        scores, scores_a, scores_b, ids_a, ids_b, selected, padded);
                    [command commit];
                    [command waitUntilCompleted];
                    if (command.status != MTLCommandBufferStatusCompleted)
                        throw std::runtime_error(command.error.localizedDescription.UTF8String);
                    const auto* ids = static_cast<const std::uint32_t*>(selected.contents);
                    std::vector<std::uint32_t> actual(ids, ids + 128);
                    std::sort(actual.begin(), actual.end());
                    if (actual != expected)
                        throw std::runtime_error("QSA top-128 mismatch at " + std::to_string(count));
                }
                std::cout << "qsa candidates=" << count << " cpu_top128_exact=true\n";
            }
            for (std::uint32_t invalid : {0U, 128U, 257U, 65792U}) {
                bool rejected = false;
                try {
                    qwen38::persistent_metal::encode_qsa_top128([queue commandBuffer], first, merge,
                        scores, scores_a, scores_b, ids_a, ids_b, selected, invalid);
                } catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected) throw std::runtime_error("invalid padding accepted");
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
