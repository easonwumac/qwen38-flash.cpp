#include "qwen38/persistent_metal_backend.hpp"

#include "persistent_metal_kernels.hpp"
#include "qwen38/safetensors.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qwen38 {
namespace {

constexpr std::array<const char*, 29> pipeline_names{
    "q3_gate_up", "q3_down_reduce", "q4_shared_gate_up", "q8_shared_router",
    "q4_shared_down_merge", "fused_all_gate_up", "fused_all_down",
    "q8_router_logits", "select_top10", "qsa_score_blocks",
    "qsa_append_decode_state", "qsa_top128_first", "qsa_top128_merge",
    "qsa_attention_q8_blocks", "attention_qkv_index", "attention_normalize_rope",
    "attention_apply_gate", "attention_output_projection", "q4_input_projection",
    "gdn_prework", "gdn_recurrence", "gdn_norm_gate", "gdn_output_projection",
    "hc_write", "hc_normalize", "hc_down_injection", "hc_up_mix", "healing_left",
    "healing_right_write",
};

bool has_expected_layer_split(const ModelConfig& config) noexcept {
    if (config.layer_types.size() != 48 || config.shared_only_layers.size() != 24) return false;
    for (std::size_t layer = 0; layer < config.layer_types.size(); ++layer) {
        const char* expected = layer % 4 == 3 ? "full_attention" : "linear_attention";
        if (config.layer_types[layer] != expected) return false;
    }
    for (std::size_t index = 0; index < config.shared_only_layers.size(); ++index) {
        if (config.shared_only_layers[index] != index + 10) return false;
    }
    return true;
}

} // namespace

class PersistentMetalBackend::Impl final {
public:
    struct Shard {
        explicit Shard(id<MTLDevice> device, const std::filesystem::path& path)
            : file(path) {
            const auto mapped = file.mapped_view();
            buffer = [device newBufferWithBytesNoCopy:const_cast<std::byte*>(mapped.data())
                                             length:mapped.size()
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
            if (buffer == nil) throw std::runtime_error("Metal rejected mmap-backed weights");
        }
        SafetensorsFile file;
        id<MTLBuffer> buffer{nil};
    };

    struct GdnState {
        id<MTLBuffer> convolution{nil};
        id<MTLBuffer> recurrent{nil};
    };

    explicit Impl(const ModelManifest& manifest) {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (device_ == nil) throw std::runtime_error("Metal device is unavailable");

            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:
                persistent_metal::metal_source.data()];
            library_ = [device_ newLibraryWithSource:source options:nil error:&error];
            if (library_ == nil) {
                throw std::runtime_error(error == nil ? "Metal library compilation failed" :
                    error.localizedDescription.UTF8String);
            }
            pipelines_.reserve(pipeline_names.size());
            for (const char* name : pipeline_names) {
                NSString* function_name = [NSString stringWithUTF8String:name];
                id<MTLFunction> function = [library_ newFunctionWithName:function_name];
                if (function == nil) {
                    throw std::runtime_error(std::string("missing persistent Metal kernel: ") + name);
                }
                id<MTLComputePipelineState> pipeline =
                    [device_ newComputePipelineStateWithFunction:function error:&error];
                if (pipeline == nil) {
                    throw std::runtime_error(error == nil ?
                        std::string("failed to compile persistent Metal kernel: ") + name :
                        error.localizedDescription.UTF8String);
                }
                pipelines_.emplace(name, pipeline);
            }

            std::unordered_set<std::string> shard_names;
            shard_names.reserve(manifest.weight_map().size());
            for (const auto& [tensor, shard] : manifest.weight_map()) {
                static_cast<void>(tensor);
                shard_names.insert(shard);
            }
            shards_.reserve(shard_names.size());
            for (const std::string& name : shard_names) {
                auto file = std::make_unique<Shard>(device_, manifest.directory() / name);
                inventory_.mapped_weight_bytes += file->file.mapped_bytes();
                shards_.emplace(name, std::move(file));
            }
            inventory_.pipeline_count = pipelines_.size();
            inventory_.shard_count = shards_.size();
            weight_map_ = manifest.weight_map();
            queue_ = [device_ newCommandQueue];
            if (queue_ == nil) throw std::runtime_error("cannot create persistent Metal queue");
            allocate_scratch();
            for (std::size_t layer = 0; layer < 48; ++layer) {
                if (layer % 4 == 3) continue;
                gdn_states_[layer].convolution = make_buffer(3 * 10240 * sizeof(std::uint16_t));
                gdn_states_[layer].recurrent =
                    make_buffer(48ULL * 128 * 128 * sizeof(std::uint16_t));
            }
        }
    }

    id<MTLBuffer> make_buffer(NSUInteger bytes) {
        id<MTLBuffer> value = [device_ newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (value == nil) throw std::bad_alloc();
        std::memset(value.contents, 0, bytes);
        return value;
    }

    void allocate_scratch() {
        stream_ = make_buffer(10240 * sizeof(std::uint16_t));
        normalized_ = make_buffer(10240 * sizeof(std::uint16_t));
        activation_ = make_buffer(320 * sizeof(std::uint16_t));
        injection_ = make_buffer(4 * sizeof(std::uint16_t));
        mixed_ = make_buffer(2560 * sizeof(std::uint16_t));
        projected_ = make_buffer(16480 * sizeof(std::uint16_t));
        query_ = make_buffer(2048 * sizeof(std::uint16_t));
        key_ = make_buffer(2048 * sizeof(std::uint16_t));
        value_ = make_buffer(6144 * sizeof(std::uint16_t));
        decay_ = make_buffer(48 * sizeof(std::uint16_t));
        beta_ = make_buffer(48 * sizeof(std::uint16_t));
        recurrent_output_ = make_buffer(6144 * sizeof(std::uint16_t));
        gated_ = make_buffer(6144 * sizeof(std::uint16_t));
        block_output_ = make_buffer(2560 * sizeof(std::uint16_t));
        post_attention_ = make_buffer(10240 * sizeof(std::uint16_t));
        expert_ids_ = make_buffer(10 * sizeof(std::uint32_t));
        route_weights_ = make_buffer(10 * sizeof(std::uint16_t));
        routed_hidden_ = make_buffer(10 * 448 * sizeof(std::uint16_t));
        shared_hidden_ = make_buffer(640 * sizeof(std::uint16_t));
        shared_scale_ = make_buffer(sizeof(std::uint16_t));
        moe_output_ = make_buffer(2560 * sizeof(std::uint16_t));
        healing_hidden_ = make_buffer(64 * sizeof(std::uint16_t));
        output_stream_ = make_buffer(10240 * sizeof(std::uint16_t));
        router_logits_ = make_buffer(512 * sizeof(float));
    }

    id<MTLComputePipelineState> pipeline(const char* name) const {
        return pipelines_.at(name);
    }

    void bind(id<MTLComputeCommandEncoder> encoder, const std::string& tensor,
              NSUInteger index) const {
        const std::string& shard_name = weight_map_.at(tensor);
        const Shard& shard = *shards_.at(shard_name);
        const auto mapped = shard.file.mapped_view();
        const auto view = shard.file.tensor(tensor);
        const auto offset = static_cast<NSUInteger>(view.bytes.data() - mapped.data());
        [encoder setBuffer:shard.buffer offset:offset atIndex:index];
    }

    void bind_projection(id<MTLComputeCommandEncoder> encoder, const std::string& base,
                         NSUInteger first) const {
        bind(encoder, base + ".weight", first);
        bind(encoder, base + ".scales", first + 1);
        bind(encoder, base + ".biases", first + 2);
    }

    void encode_hc_read(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                        const std::string& base) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("hc_normalize")];
        [encoder setBuffer:input offset:0 atIndex:0];
        bind(encoder, base + ".hc_norm.weight", 1);
        [encoder setBuffer:normalized_ offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("hc_down_injection")];
        [encoder setBuffer:normalized_ offset:0 atIndex:0];
        bind_projection(encoder, base + ".input_mix_weight_down", 1);
        bind_projection(encoder, base + ".block_inject_weight", 4);
        [encoder setBuffer:activation_ offset:0 atIndex:7];
        [encoder setBuffer:injection_ offset:0 atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(324, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("hc_up_mix")];
        [encoder setBuffer:normalized_ offset:0 atIndex:0];
        [encoder setBuffer:activation_ offset:0 atIndex:1];
        bind_projection(encoder, base + ".input_mix_weight_up", 2);
        [encoder setBuffer:mixed_ offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(320, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }

    void encode_gdn(id<MTLCommandBuffer> command, const std::string& base,
                    GdnState& state) {
        const auto project = [&](const char* suffix, std::uint32_t rows,
                                 NSUInteger output_offset) {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("q4_input_projection")];
            [encoder setBuffer:mixed_ offset:0 atIndex:0];
            bind_projection(encoder, base + suffix, 1);
            [encoder setBuffer:projected_ offset:output_offset * sizeof(std::uint16_t) atIndex:4];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        };
        project(".in_proj_qkv", 10240, 0);
        project(".in_proj_z", 6144, 10240);
        project(".in_proj_b", 48, 16384);
        project(".in_proj_a", 48, 16432);

        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("gdn_prework")];
        [encoder setBuffer:projected_ offset:0 atIndex:0];
        [encoder setBuffer:state.convolution offset:0 atIndex:1];
        bind(encoder, base + ".conv1d.weight", 2);
        bind(encoder, base + ".A_log", 3);
        bind(encoder, base + ".dt_bias", 4);
        [encoder setBuffer:query_ offset:0 atIndex:5];
        [encoder setBuffer:key_ offset:0 atIndex:6];
        [encoder setBuffer:value_ offset:0 atIndex:7];
        [encoder setBuffer:decay_ offset:0 atIndex:8];
        [encoder setBuffer:beta_ offset:0 atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(80, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("gdn_recurrence")];
        [encoder setBuffer:query_ offset:0 atIndex:0];
        [encoder setBuffer:key_ offset:0 atIndex:1];
        [encoder setBuffer:value_ offset:0 atIndex:2];
        [encoder setBuffer:decay_ offset:0 atIndex:3];
        [encoder setBuffer:beta_ offset:0 atIndex:4];
        [encoder setBuffer:state.recurrent offset:0 atIndex:5];
        [encoder setBuffer:recurrent_output_ offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(128, 48, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("gdn_norm_gate")];
        [encoder setBuffer:recurrent_output_ offset:0 atIndex:0];
        [encoder setBuffer:projected_ offset:0 atIndex:1];
        bind(encoder, base + ".norm.weight", 2);
        [encoder setBuffer:gated_ offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(48, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("gdn_output_projection")];
        [encoder setBuffer:gated_ offset:0 atIndex:0];
        bind_projection(encoder, base + ".out_proj", 1);
        [encoder setBuffer:block_output_ offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
    }

    void encode_hc_write(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                         id<MTLBuffer> block, id<MTLBuffer> output) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("hc_write")];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:block offset:0 atIndex:1];
        [encoder setBuffer:injection_ offset:0 atIndex:2];
        [encoder setBuffer:output offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(10240, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }

    void encode_healing_write(id<MTLCommandBuffer> command, const std::string& mlp,
                              id<MTLBuffer> source_stream) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("healing_left")];
        [encoder setBuffer:moe_output_ offset:0 atIndex:0];
        bind(encoder, mlp + ".T_delta_left", 1);
        [encoder setBuffer:healing_hidden_ offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("healing_right_write")];
        [encoder setBuffer:moe_output_ offset:0 atIndex:0];
        [encoder setBuffer:healing_hidden_ offset:0 atIndex:1];
        bind(encoder, mlp + ".T_delta_right", 2);
        [encoder setBuffer:source_stream offset:0 atIndex:3];
        [encoder setBuffer:injection_ offset:0 atIndex:4];
        [encoder setBuffer:output_stream_ offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
    }

    void encode_mlp(id<MTLCommandBuffer> command, const std::string& layer,
                    bool shared_only) {
        const std::string mlp = layer + ".mlp";
        encode_hc_read(command, post_attention_, layer + ".mlp_hyper_connection");
        id<MTLComputeCommandEncoder> encoder;
        if (shared_only) {
            std::memset(moe_output_.contents, 0, 2560 * sizeof(std::uint16_t));
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("q4_shared_gate_up")];
            [encoder setBuffer:mixed_ offset:0 atIndex:0];
            bind_projection(encoder, mlp + ".shared_expert.gate_proj", 1);
            bind_projection(encoder, mlp + ".shared_expert.up_proj", 4);
            [encoder setBuffer:shared_hidden_ offset:0 atIndex:7];
            [encoder dispatchThreadgroups:MTLSizeMake(160, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("q8_shared_router")];
            [encoder setBuffer:mixed_ offset:0 atIndex:0];
            bind_projection(encoder, mlp + ".shared_expert_gate", 1);
            [encoder setBuffer:shared_scale_ offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("q4_shared_down_merge")];
            [encoder setBuffer:shared_hidden_ offset:0 atIndex:0];
            bind_projection(encoder, mlp + ".shared_expert.down_proj", 1);
            [encoder setBuffer:shared_scale_ offset:0 atIndex:4];
            [encoder setBuffer:moe_output_ offset:0 atIndex:5];
            [encoder setBuffer:moe_output_ offset:0 atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        } else {
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("q8_router_logits")];
            [encoder setBuffer:mixed_ offset:0 atIndex:0];
            bind_projection(encoder, mlp + ".gate", 1);
            [encoder setBuffer:router_logits_ offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(128, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("select_top10")];
            [encoder setBuffer:router_logits_ offset:0 atIndex:0];
            [encoder setBuffer:expert_ids_ offset:0 atIndex:1];
            [encoder setBuffer:route_weights_ offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("fused_all_gate_up")];
            [encoder setBuffer:mixed_ offset:0 atIndex:0];
            bind_projection(encoder, mlp + ".switch_mlp.gate_proj", 1);
            bind_projection(encoder, mlp + ".switch_mlp.up_proj", 4);
            [encoder setBuffer:expert_ids_ offset:0 atIndex:7];
            [encoder setBuffer:routed_hidden_ offset:0 atIndex:8];
            bind_projection(encoder, mlp + ".shared_expert.gate_proj", 9);
            bind_projection(encoder, mlp + ".shared_expert.up_proj", 12);
            [encoder setBuffer:shared_hidden_ offset:0 atIndex:15];
            bind_projection(encoder, mlp + ".shared_expert_gate", 16);
            [encoder setBuffer:shared_scale_ offset:0 atIndex:19];
            [encoder dispatchThreadgroups:MTLSizeMake(1281, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("fused_all_down")];
            [encoder setBuffer:routed_hidden_ offset:0 atIndex:0];
            bind_projection(encoder, mlp + ".switch_mlp.down_proj", 1);
            [encoder setBuffer:expert_ids_ offset:0 atIndex:4];
            [encoder setBuffer:route_weights_ offset:0 atIndex:5];
            [encoder setBuffer:shared_hidden_ offset:0 atIndex:6];
            bind_projection(encoder, mlp + ".shared_expert.down_proj", 7);
            [encoder setBuffer:shared_scale_ offset:0 atIndex:10];
            [encoder setBuffer:moe_output_ offset:0 atIndex:11];
            [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        }
        encode_healing_write(command, mlp, post_attention_);
    }

    std::vector<std::uint16_t> decode_gdn_layer(std::size_t index,
                                                std::span<const std::uint16_t> stream,
                                                bool reset, double* gpu_ms) {
        if (index >= 48 || index % 4 == 3) throw std::out_of_range("not a GDN layer");
        if (stream.size() != 10240) throw std::runtime_error("persistent stream width mismatch");
        GdnState& state = gdn_states_[index];
        if (reset) {
            std::memset(state.convolution.contents, 0, 3 * 10240 * sizeof(std::uint16_t));
            std::memset(state.recurrent.contents, 0,
                        48ULL * 128 * 128 * sizeof(std::uint16_t));
        }
        std::memcpy(stream_.contents, stream.data(), stream.size_bytes());
        const std::string layer = "language_model.model.layers." + std::to_string(index);
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        encode_hc_read(command, stream_, layer + ".attn_hyper_connection");
        encode_gdn(command, layer + ".linear_attn", state);
        encode_hc_write(command, stream_, block_output_, post_attention_);
        encode_mlp(command, layer, index >= 10 && index <= 33);
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
        }
        if (gpu_ms != nullptr) *gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        const auto* begin = static_cast<const std::uint16_t*>(output_stream_.contents);
        return {begin, begin + 10240};
    }

    Inventory inventory_;
    id<MTLDevice> device_{nil};
    id<MTLLibrary> library_{nil};
    id<MTLCommandQueue> queue_{nil};
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::unordered_map<std::string, std::unique_ptr<Shard>> shards_;
    std::unordered_map<std::string, std::string> weight_map_;
    std::array<GdnState, 48> gdn_states_;
    id<MTLBuffer> stream_{nil}, normalized_{nil}, activation_{nil}, injection_{nil}, mixed_{nil};
    id<MTLBuffer> projected_{nil}, query_{nil}, key_{nil}, value_{nil}, decay_{nil}, beta_{nil};
    id<MTLBuffer> recurrent_output_{nil}, gated_{nil}, block_output_{nil};
    id<MTLBuffer> post_attention_{nil}, expert_ids_{nil}, route_weights_{nil};
    id<MTLBuffer> routed_hidden_{nil}, shared_hidden_{nil}, shared_scale_{nil};
    id<MTLBuffer> moe_output_{nil}, healing_hidden_{nil}, output_stream_{nil}, router_logits_{nil};
};

bool PersistentMetalBackend::supports(const ModelConfig& config) noexcept {
    return config.hidden_size == 2560 && config.layer_count == 48 &&
        config.expert_count == 512 && config.experts_per_token == 10 &&
        config.hyper_connection_count == 4 && config.moe_intermediate_size == 448 &&
        config.shared_expert_intermediate_size == 640 && config.attention_head_count == 24 &&
        config.key_value_head_count == 2 && config.head_dimension == 256 &&
        config.indexer_head_count == 4 && config.indexer_key_value_head_count == 1 &&
        config.indexer_head_dimension == 128 && config.indexer_compress_ratio == 4 &&
        config.linear_convolution_kernel_size == 4 && config.linear_key_head_dimension == 128 &&
        config.linear_value_head_dimension == 128 && config.linear_key_head_count == 16 &&
        config.linear_value_head_count == 48 && config.niwaki_maps_unfolded &&
        has_expected_layer_split(config);
}

std::unique_ptr<PersistentMetalBackend> PersistentMetalBackend::create(
    const ModelManifest& manifest) {
    if (!supports(manifest.config())) return nullptr;
    return std::unique_ptr<PersistentMetalBackend>(
        new PersistentMetalBackend(std::make_unique<Impl>(manifest)));
}

PersistentMetalBackend::PersistentMetalBackend(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

PersistentMetalBackend::~PersistentMetalBackend() = default;
PersistentMetalBackend::PersistentMetalBackend(PersistentMetalBackend&&) noexcept = default;
PersistentMetalBackend& PersistentMetalBackend::operator=(PersistentMetalBackend&&) noexcept = default;

const PersistentMetalBackend::Inventory& PersistentMetalBackend::inventory() const noexcept {
    return impl_->inventory_;
}

std::vector<std::uint16_t> PersistentMetalBackend::decode_gdn_layer(
    const std::size_t layer, const std::span<const std::uint16_t> stream,
    const bool reset_state, double* gpu_ms) {
    return impl_->decode_gdn_layer(layer, stream, reset_state, gpu_ms);
}

} // namespace qwen38
