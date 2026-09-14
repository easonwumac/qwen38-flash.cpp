#include "qwen38/persistent_metal_backend.hpp"

#include "persistent_metal_kernels.hpp"
#include "qwen38/model.hpp"
#include "qwen38/ngram.hpp"
#include "qwen38/safetensors.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qwen38 {
namespace {

constexpr std::array<const char*, 33> pipeline_names{
    "q3_gate_up", "q3_down_reduce", "q4_shared_gate_up", "q8_shared_router",
    "q4_shared_down_merge", "fused_all_gate_up", "fused_all_down",
    "q8_router_logits", "select_top10", "qsa_score_blocks",
    "qsa_append_decode_state", "qsa_top128_first", "qsa_top128_merge",
    "qsa_attention_q8_blocks", "attention_qkv_index", "attention_normalize_rope",
    "attention_apply_gate", "attention_output_projection", "q4_input_projection",
    "gdn_prework", "gdn_recurrence", "gdn_norm_gate", "gdn_output_projection",
    "hc_write", "hc_normalize", "hc_down_injection", "hc_up_mix", "healing_left",
    "healing_right_write", "ple_fused_update", "embedding_q4_stream",
    "hc_down_only", "lm_head_q4",
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

    struct SharedTensor {
        MlxArray array;
        const void* data{nullptr};
        id<MTLBuffer> buffer{nil};
    };

    struct GdnState {
        id<MTLBuffer> convolution{nil};
        id<MTLBuffer> recurrent{nil};
    };

    struct AttentionState {
        id<MTLBuffer> hot_keys{nil};
        id<MTLBuffer> hot_values{nil};
        id<MTLBuffer> key_weights{nil};
        id<MTLBuffer> key_scales{nil};
        id<MTLBuffer> key_biases{nil};
        id<MTLBuffer> value_weights{nil};
        id<MTLBuffer> value_scales{nil};
        id<MTLBuffer> value_biases{nil};
        id<MTLBuffer> pending{nil};
        id<MTLBuffer> pooled{nil};
        std::uint32_t token_count{0};
        std::uint32_t position_base{0};
        std::uint32_t cold_count{0};
    };

    explicit Impl(const ModelManifest& manifest, MlxTensorStore* shared_weights) {
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
            if (shared_weights != nullptr) {
                shared_tensors_.reserve(manifest.weight_map().size());
                for (const auto& [tensor, shard] : manifest.weight_map()) {
                    static_cast<void>(shard);
                    MlxArray array = shared_weights->tensor(tensor);
                    const void* data = array.data_bytes();
                    shared_tensors_.emplace(
                        tensor, SharedTensor{std::move(array), data, nil});
                }
            }
            inventory_.pipeline_count = pipelines_.size();
            inventory_.shard_count = shards_.size();
            weight_map_ = manifest.weight_map();
            config_ = manifest.config();
            queue_ = [device_ newCommandQueue];
            if (queue_ == nil) throw std::runtime_error("cannot create persistent Metal queue");
            allocate_scratch();
            ple_hash_ = std::make_unique<NgramHash>(config_);
            const char* external_ngram = std::getenv("QWEN38_NGRAM_TABLE_DIR");
            const std::filesystem::path ngram_directory =
                external_ngram != nullptr && *external_ngram != '\0'
                ? std::filesystem::path(external_ngram) : manifest.directory();
            ple_table_ = std::make_unique<NgramTable>(
                ngram_directory, ple_hash_->total_rows());
            for (std::size_t layer = 0; layer < 48; ++layer) {
                if (layer % 4 == 3) {
                    attention_states_[layer].hot_keys =
                        make_buffer(2ULL * hot_capacity * 256 * sizeof(std::uint16_t));
                    attention_states_[layer].hot_values =
                        make_buffer(2ULL * hot_capacity * 256 * sizeof(std::uint16_t));
                    attention_states_[layer].pending = make_buffer(4 * 128 * sizeof(std::uint16_t));
                    attention_states_[layer].pooled = make_buffer(
                        65536ULL * 128 * sizeof(std::uint16_t), false);
                } else {
                    gdn_states_[layer].convolution =
                        make_buffer(3 * 10240 * sizeof(std::uint16_t));
                    gdn_states_[layer].recurrent =
                        make_buffer(48ULL * 128 * 128 * sizeof(std::uint16_t));
                }
            }
        }
    }

    id<MTLBuffer> make_buffer(NSUInteger bytes, bool clear = true) {
        id<MTLBuffer> value = [device_ newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (value == nil) throw std::bad_alloc();
        if (clear) std::memset(value.contents, 0, bytes);
        return value;
    }

    void prepare_shared_weights() {
        for (auto& [name, shared] : shared_tensors_) {
            static_cast<void>(name);
            if (shared.buffer != nil) continue;
            shared.buffer = [device_ newBufferWithBytesNoCopy:const_cast<void*>(shared.data)
                                                       length:shared.array.byte_size()
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
            if (shared.buffer == nil) {
                throw std::runtime_error("Metal rejected an MLX tensor allocation");
            }
        }
    }

    void release_shared_weights() {
        for (auto& [name, shared] : shared_tensors_) {
            static_cast<void>(name);
            shared.buffer = nil;
        }
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
        zero_output_ = make_buffer(2560 * sizeof(std::uint16_t));
        healing_hidden_ = make_buffer(64 * sizeof(std::uint16_t));
        output_stream_ = make_buffer(10240 * sizeof(std::uint16_t));
        router_logits_ = make_buffer(512 * sizeof(float));
        attention_projected_ = make_buffer(13952 * sizeof(std::uint16_t));
        selector_query_ = make_buffer(4 * 128 * sizeof(std::uint16_t));
        attention_query_ = make_buffer(24 * 256 * sizeof(std::uint16_t));
        attention_key_ = make_buffer(2 * 256 * sizeof(std::uint16_t));
        attention_values_ = make_buffer(24 * 256 * sizeof(std::uint16_t));
        attention_gated_ = make_buffer(24 * 256 * sizeof(std::uint16_t));
        rope_cos_ = make_buffer(64 * sizeof(std::uint16_t));
        rope_sin_ = make_buffer(64 * sizeof(std::uint16_t));
        pool_rope_cos_ = make_buffer(64 * sizeof(std::uint16_t));
        pool_rope_sin_ = make_buffer(64 * sizeof(std::uint16_t));
        qsa_scores_ = make_buffer(65536 * sizeof(float), false);
        selected_ = make_buffer(128 * sizeof(std::uint32_t));
        temp_scores_a_ = make_buffer(32768 * sizeof(float), false);
        temp_scores_b_ = make_buffer(32768 * sizeof(float), false);
        temp_ids_a_ = make_buffer(32768 * sizeof(std::uint32_t), false);
        temp_ids_b_ = make_buffer(32768 * sizeof(std::uint32_t), false);
        q8_dummy_ = make_buffer(16);
        ple_embedding_ = make_buffer(2560 * sizeof(std::uint16_t));
        ple_projected_ = make_buffer(12800 * sizeof(std::uint16_t));
        ple_stream_ = make_buffer(10240 * sizeof(std::uint16_t));
        ple_convolution_ = make_buffer(9ULL * 10240 * sizeof(std::uint16_t));
        head_logits_ = make_buffer(config_.vocabulary_size * sizeof(float), false);
    }

    id<MTLComputePipelineState> pipeline(const char* name) const {
        return pipelines_.at(name);
    }

    void bind(id<MTLComputeCommandEncoder> encoder, const std::string& tensor,
              NSUInteger index) const {
        if (!shared_tensors_.empty()) {
            [encoder setBuffer:shared_tensors_.at(tensor).buffer offset:0 atIndex:index];
            return;
        }
        const std::string& shard_name = weight_map_.at(tensor);
        const Shard& shard = *shards_.at(shard_name);
        const auto mapped = shard.file.mapped_view();
        const auto view = shard.file.tensor(tensor);
        const auto offset = static_cast<NSUInteger>(view.bytes.data() - mapped.data());
        [encoder setBuffer:shard.buffer offset:offset atIndex:index];
    }

    void bind_row(id<MTLComputeCommandEncoder> encoder, const std::string& tensor,
                  std::size_t row, NSUInteger index) const {
        if (!shared_tensors_.empty()) {
            const SharedTensor& shared = shared_tensors_.at(tensor);
            const std::vector<int> shape = shared.array.shape();
            if (shape.empty() || row >= static_cast<std::size_t>(shape.front()) ||
                shared.array.byte_size() % static_cast<std::size_t>(shape.front()) != 0) {
                throw std::runtime_error("invalid shared persistent row tensor geometry");
            }
            const std::size_t row_bytes =
                shared.array.byte_size() / static_cast<std::size_t>(shape.front());
            [encoder setBuffer:shared.buffer offset:row * row_bytes atIndex:index];
            return;
        }
        const std::string& shard_name = weight_map_.at(tensor);
        const Shard& shard = *shards_.at(shard_name);
        const auto mapped = shard.file.mapped_view();
        const auto view = shard.file.tensor(tensor);
        if (view.shape.empty() || row >= view.shape.front() ||
            view.bytes.size() % view.shape.front() != 0) {
            throw std::runtime_error("invalid persistent row tensor geometry");
        }
        const std::size_t row_bytes = view.bytes.size() / view.shape.front();
        const std::size_t tensor_offset = static_cast<std::size_t>(
            view.bytes.data() - mapped.data());
        const auto offset = static_cast<NSUInteger>(tensor_offset + row * row_bytes);
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
            [encoder setBuffer:zero_output_ offset:0 atIndex:5];
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

    static std::uint16_t bf16(float value) {
        std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        return static_cast<std::uint16_t>(bits >> 16U);
    }

    void update_rope(id<MTLBuffer> cosine, id<MTLBuffer> sine, std::uint32_t position) {
        auto* cos_values = static_cast<std::uint16_t*>(cosine.contents);
        auto* sin_values = static_cast<std::uint16_t*>(sine.contents);
        for (std::size_t index = 0; index < 32; ++index) {
            const double frequency = std::pow(10000000.0, -2.0 * static_cast<double>(index) / 64.0);
            const double angle = static_cast<double>(position) * frequency;
            cos_values[index] = cos_values[index + 32] = bf16(static_cast<float>(std::cos(angle)));
            sin_values[index] = sin_values[index + 32] = bf16(static_cast<float>(std::sin(angle)));
        }
    }

    void encode_qsa_selection(id<MTLCommandBuffer> command, AttentionState& state,
                              std::uint32_t block_count, std::uint32_t& selected_count) {
        if (block_count <= 128) {
            auto* ids = static_cast<std::uint32_t*>(selected_.contents);
            for (std::uint32_t index = 0; index < block_count; ++index) ids[index] = index;
            selected_count = block_count * 4;
            return;
        }
        selected_count = 512;
        const std::uint32_t padded = (block_count + 255) / 256 * 256;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("qsa_score_blocks")];
        [encoder setBuffer:selector_query_ offset:0 atIndex:0];
        [encoder setBuffer:state.pooled offset:0 atIndex:1];
        [encoder setBuffer:qsa_scores_ offset:0 atIndex:2];
        [encoder setBytes:&block_count length:sizeof(block_count) atIndex:3];
        [encoder setBytes:&padded length:sizeof(padded) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(padded / 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("qsa_top128_first")];
        [encoder setBuffer:qsa_scores_ offset:0 atIndex:0];
        [encoder setBuffer:temp_scores_a_ offset:0 atIndex:1];
        [encoder setBuffer:temp_ids_a_ offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(padded / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        std::uint32_t groups = padded / 256;
        bool source_a = true;
        while (groups > 1) {
            const std::uint32_t output_groups = (groups + 1) / 2;
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("qsa_top128_merge")];
            [encoder setBuffer:(source_a ? temp_scores_a_ : temp_scores_b_) offset:0 atIndex:0];
            [encoder setBuffer:(source_a ? temp_ids_a_ : temp_ids_b_) offset:0 atIndex:1];
            [encoder setBuffer:(source_a ? temp_scores_b_ : temp_scores_a_) offset:0 atIndex:2];
            [encoder setBuffer:(output_groups == 1 ? selected_ :
                (source_a ? temp_ids_b_ : temp_ids_a_)) offset:0 atIndex:3];
            [encoder setBytes:&groups length:sizeof(groups) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(output_groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            groups = output_groups;
            source_a = !source_a;
        }
    }

    void encode_attention(id<MTLCommandBuffer> command, const std::string& base,
                          AttentionState& state) {
        update_rope(rope_cos_, rope_sin_, state.position_base + state.token_count);
        const std::uint32_t pending_index = state.token_count % 4;
        const std::uint32_t old_blocks = state.token_count / 4;
        const std::uint32_t resulting_tokens = state.token_count + 1;
        const std::uint32_t block_count = resulting_tokens / 4;
        const std::uint32_t complete_block = block_count > old_blocks ? 1 : 0;
        update_rope(pool_rope_cos_, pool_rope_sin_,
                    state.position_base + old_blocks * 4);

        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("attention_qkv_index")];
        [encoder setBuffer:mixed_ offset:0 atIndex:0];
        bind_projection(encoder, base + ".indexer.index_qk_proj", 1);
        bind_projection(encoder, base + ".q_proj", 4);
        bind_projection(encoder, base + ".k_proj", 7);
        bind_projection(encoder, base + ".v_proj", 10);
        [encoder setBuffer:attention_projected_ offset:0 atIndex:13];
        [encoder dispatchThreadgroups:MTLSizeMake((13952 + 3) / 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("attention_normalize_rope")];
        [encoder setBuffer:attention_projected_ offset:0 atIndex:0];
        bind(encoder, base + ".q_norm.weight", 1);
        bind(encoder, base + ".k_norm.weight", 2);
        bind(encoder, base + ".indexer.q_layernorm.weight", 3);
        [encoder setBuffer:rope_cos_ offset:0 atIndex:4];
        [encoder setBuffer:rope_sin_ offset:0 atIndex:5];
        [encoder setBuffer:selector_query_ offset:0 atIndex:6];
        [encoder setBuffer:attention_query_ offset:0 atIndex:7];
        [encoder setBuffer:attention_key_ offset:0 atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(30, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("qsa_append_decode_state")];
        [encoder setBuffer:attention_projected_ offset:0 atIndex:0];
        [encoder setBuffer:attention_key_ offset:0 atIndex:1];
        [encoder setBuffer:state.hot_keys offset:0 atIndex:2];
        [encoder setBuffer:state.hot_values offset:0 atIndex:3];
        [encoder setBuffer:state.pending offset:0 atIndex:4];
        [encoder setBuffer:state.pooled offset:0 atIndex:5];
        bind(encoder, base + ".indexer.k_layernorm.weight", 6);
        [encoder setBuffer:pool_rope_cos_ offset:0 atIndex:7];
        [encoder setBuffer:pool_rope_sin_ offset:0 atIndex:8];
        const std::uint32_t hot_index = state.token_count - state.cold_count;
        [encoder setBytes:&hot_index length:sizeof(hot_index) atIndex:9];
        [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:10];
        [encoder setBytes:&pending_index length:sizeof(pending_index) atIndex:11];
        [encoder setBytes:&old_blocks length:sizeof(old_blocks) atIndex:12];
        [encoder setBytes:&complete_block length:sizeof(complete_block) atIndex:13];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];

        std::uint32_t selected_count = 0;
        encode_qsa_selection(command, state, block_count, selected_count);
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("qsa_attention_q8_blocks")];
        [encoder setBuffer:attention_query_ offset:0 atIndex:0];
        [encoder setBuffer:(state.cold_count == 0 ? q8_dummy_ : state.key_weights)
                    offset:0 atIndex:1];
        [encoder setBuffer:(state.cold_count == 0 ? q8_dummy_ : state.key_scales)
                    offset:0 atIndex:2];
        [encoder setBuffer:(state.cold_count == 0 ? q8_dummy_ : state.key_biases)
                    offset:0 atIndex:3];
        [encoder setBuffer:(state.cold_count == 0 ? q8_dummy_ : state.value_weights)
                    offset:0 atIndex:4];
        [encoder setBuffer:(state.cold_count == 0 ? q8_dummy_ : state.value_scales)
                    offset:0 atIndex:5];
        [encoder setBuffer:(state.cold_count == 0 ? q8_dummy_ : state.value_biases)
                    offset:0 atIndex:6];
        [encoder setBuffer:selected_ offset:0 atIndex:7];
        [encoder setBuffer:attention_values_ offset:0 atIndex:8];
        [encoder setBytes:&state.cold_count length:sizeof(state.cold_count) atIndex:9];
        [encoder setBuffer:state.hot_keys offset:0 atIndex:10];
        [encoder setBuffer:state.hot_values offset:0 atIndex:11];
        const std::uint32_t hot_count = resulting_tokens - state.cold_count;
        [encoder setBytes:&hot_count length:sizeof(hot_count) atIndex:12];
        [encoder setBytes:&hot_capacity length:sizeof(hot_capacity) atIndex:13];
        [encoder setBytes:&selected_count length:sizeof(selected_count) atIndex:14];
        [encoder dispatchThreadgroups:MTLSizeMake(2, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(384, 1, 1)];
        [encoder endEncoding];

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("attention_apply_gate")];
        [encoder setBuffer:attention_values_ offset:0 atIndex:0];
        [encoder setBuffer:attention_projected_ offset:0 atIndex:1];
        [encoder setBuffer:attention_gated_ offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(6144, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("attention_output_projection")];
        [encoder setBuffer:attention_gated_ offset:0 atIndex:0];
        bind_projection(encoder, base + ".o_proj", 1);
        [encoder setBuffer:block_output_ offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(640, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        state.token_count = resulting_tokens;
    }

    std::vector<std::uint16_t> decode_attention_layer(
        std::size_t index, std::span<const std::uint16_t> stream, bool reset,
        double* gpu_ms) {
        if (index >= 48 || index % 4 != 3) throw std::out_of_range("not an attention layer");
        if (stream.size() != 10240) throw std::runtime_error("persistent stream width mismatch");
        AttentionState& state = attention_states_[index];
        if (reset) {
            state.token_count = 0;
            std::memset(state.hot_keys.contents, 0,
                        2ULL * hot_capacity * 256 * sizeof(std::uint16_t));
            std::memset(state.hot_values.contents, 0,
                        2ULL * hot_capacity * 256 * sizeof(std::uint16_t));
            std::memset(state.pending.contents, 0, 4 * 128 * sizeof(std::uint16_t));
        }
        if (state.token_count - state.cold_count >= hot_capacity) {
            throw std::runtime_error("persistent attention hot slab requires Q8 flush");
        }
        std::memcpy(stream_.contents, stream.data(), stream.size_bytes());
        const std::string layer = "language_model.model.layers." + std::to_string(index);
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        encode_hc_read(command, stream_, layer + ".attn_hyper_connection");
        encode_attention(command, layer + ".self_attn", state);
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

    void encode_ple(id<MTLCommandBuffer> command, std::uint32_t token,
                    id<MTLBuffer> input) {
        const auto rows = ple_hash_->row_ids(token, ple_ngram_state_);
        const std::vector<float> gathered = ple_table_->gather(rows);
        if (gathered.size() != 2560) throw std::runtime_error("PLE gather width mismatch");
        auto* embedding = static_cast<std::uint16_t*>(ple_embedding_.contents);
        for (std::size_t index = 0; index < gathered.size(); ++index) {
            embedding[index] = bf16(gathered[index]);
        }
        const std::string base = "language_model.model.layers.1.ple";
        const auto project = [&](const char* suffix, std::uint32_t rows_count,
                                 NSUInteger output_offset) {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline("q4_input_projection")];
            [encoder setBuffer:ple_embedding_ offset:0 atIndex:0];
            bind_projection(encoder, base + suffix, 1);
            [encoder setBuffer:ple_projected_
                        offset:output_offset * sizeof(std::uint16_t) atIndex:4];
            [encoder setBytes:&rows_count length:sizeof(rows_count) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake((rows_count + 3) / 4, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        };
        project(".key_proj", 10240, 0);
        project(".value_proj", 2560, 10240);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("ple_fused_update")];
        [encoder setBuffer:ple_projected_ offset:0 atIndex:0];
        [encoder setBuffer:input offset:0 atIndex:1];
        bind(encoder, base + ".norm_key.weight", 2);
        bind(encoder, base + ".norm_query.weight", 3);
        bind(encoder, base + ".norm_conv.weight", 4);
        bind(encoder, base + ".conv1d.weight", 5);
        [encoder setBuffer:ple_convolution_ offset:0 atIndex:6];
        [encoder setBuffer:ple_stream_ offset:0 atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }

    std::vector<std::uint16_t> decode_ple(
        std::uint32_t token, std::span<const std::uint16_t> stream,
        bool reset, double* gpu_ms) {
        if (stream.size() != 10240) throw std::runtime_error("PLE stream width mismatch");
        if (reset) {
            std::memset(ple_convolution_.contents, 0, 9ULL * 10240 * sizeof(std::uint16_t));
            ple_ngram_state_ = {};
        }
        std::memcpy(stream_.contents, stream.data(), stream.size_bytes());
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        encode_ple(command, token, stream_);
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
        }
        if (gpu_ms != nullptr) *gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        const auto* begin = static_cast<const std::uint16_t*>(ple_stream_.contents);
        return {begin, begin + 10240};
    }

    void reset_all_state() {
        for (std::size_t layer = 0; layer < 48; ++layer) {
            if (layer % 4 == 3) {
                attention_states_[layer].token_count = 0;
                attention_states_[layer].position_base = 0;
                attention_states_[layer].cold_count = 0;
                std::memset(attention_states_[layer].hot_keys.contents, 0,
                            2ULL * hot_capacity * 256 * sizeof(std::uint16_t));
                std::memset(attention_states_[layer].hot_values.contents, 0,
                            2ULL * hot_capacity * 256 * sizeof(std::uint16_t));
                std::memset(attention_states_[layer].pending.contents, 0,
                            4 * 128 * sizeof(std::uint16_t));
            } else {
                std::memset(gdn_states_[layer].convolution.contents, 0,
                            3 * 10240 * sizeof(std::uint16_t));
                std::memset(gdn_states_[layer].recurrent.contents, 0,
                            48ULL * 128 * 128 * sizeof(std::uint16_t));
            }
        }
        std::memset(ple_convolution_.contents, 0, 9ULL * 10240 * sizeof(std::uint16_t));
        ple_ngram_state_ = {};
    }

    static void copy_array(const MlxArray& source, id<MTLBuffer> target,
                           std::size_t offset = 0) {
        const std::vector<std::uint8_t> bytes = source.to_bytes();
        if (offset > target.length || bytes.size() > target.length - offset) {
            throw std::runtime_error("persistent state buffer is too small");
        }
        if (!bytes.empty()) {
            std::memcpy(static_cast<std::uint8_t*>(target.contents) + offset,
                        bytes.data(), bytes.size());
        }
    }

    void copy_or_replace(const MlxArray& source, id<MTLBuffer> __strong& target) {
        const std::vector<std::uint8_t> bytes = source.to_bytes();
        if (bytes.empty()) throw std::runtime_error("cannot import an empty state array");
        if (target == nil || target.length != bytes.size()) {
            target = make_buffer(bytes.size(), false);
        }
        std::memcpy(target.contents, bytes.data(), bytes.size());
    }

    void import_state(const ModelDecodeState& source) {
        if (source.layers.size() != 48) {
            throw std::runtime_error("persistent state layer count mismatch");
        }
        for (std::size_t layer = 0; layer < 48; ++layer) {
            const DecoderLayerState& input = source.layers[layer];
            if (layer % 4 != 3) {
                if (!input.linear_attention.initialized) {
                    throw std::runtime_error("cannot import uninitialized GDN state");
                }
                copy_array(input.linear_attention.convolution,
                           gdn_states_[layer].convolution);
                copy_array(input.linear_attention.recurrent,
                           gdn_states_[layer].recurrent);
                continue;
            }
            const SelfAttentionState& attention = input.full_attention;
            if (attention.token_count > std::numeric_limits<std::uint32_t>::max() ||
                attention.position_base > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("persistent attention state is too large");
            }
            const std::size_t cold_tokens = attention.kv_q8
                ? attention.kv_q8_cold_tokens : 0;
            if (cold_tokens > attention.token_count ||
                attention.token_count - cold_tokens > hot_capacity) {
                throw std::runtime_error("persistent attention hot state is too large");
            }
            AttentionState& output = attention_states_[layer];
            output.token_count = static_cast<std::uint32_t>(attention.token_count);
            output.position_base = static_cast<std::uint32_t>(attention.position_base);
            output.cold_count = static_cast<std::uint32_t>(cold_tokens);
            if (attention.kv_q8) {
                copy_or_replace(attention.key_weights, output.key_weights);
                copy_or_replace(attention.key_scales, output.key_scales);
                copy_or_replace(attention.key_biases, output.key_biases);
                copy_or_replace(attention.value_weights, output.value_weights);
                copy_or_replace(attention.value_scales, output.value_scales);
                copy_or_replace(attention.value_biases, output.value_biases);
            }
            std::memset(output.hot_keys.contents, 0, output.hot_keys.length);
            std::memset(output.hot_values.contents, 0, output.hot_values.length);
            const std::size_t hot_tokens = attention.token_count - cold_tokens;
            if (hot_tokens != 0) {
                const std::vector<std::uint8_t> keys = attention.keys.to_bytes();
                const std::vector<std::uint8_t> values = attention.values.to_bytes();
                const std::size_t head_bytes =
                    hot_tokens * 256 * sizeof(std::uint16_t);
                if (keys.size() != 2 * head_bytes || values.size() != 2 * head_bytes) {
                    throw std::runtime_error("persistent BF16 KV geometry mismatch");
                }
                const std::size_t target_head_bytes =
                    hot_capacity * 256 * sizeof(std::uint16_t);
                for (std::size_t head = 0; head < 2; ++head) {
                    std::memcpy(static_cast<std::uint8_t*>(output.hot_keys.contents) +
                                    head * target_head_bytes,
                                keys.data() + head * head_bytes, head_bytes);
                    std::memcpy(static_cast<std::uint8_t*>(output.hot_values.contents) +
                                    head * target_head_bytes,
                                values.data() + head * head_bytes, head_bytes);
                }
            }
            std::memset(output.pending.contents, 0, output.pending.length);
            const std::size_t pending_rows = attention.token_count % 4;
            if (pending_rows != 0) {
                const std::vector<std::uint8_t> raw = attention.qsa_raw_keys.to_bytes();
                const std::size_t row_bytes = 128 * sizeof(std::uint16_t);
                if (raw.size() < pending_rows * row_bytes) {
                    throw std::runtime_error("persistent QSA raw state is truncated");
                }
                std::memcpy(output.pending.contents,
                            raw.data() + raw.size() - pending_rows * row_bytes,
                            pending_rows * row_bytes);
            }
            if (attention.qsa_pooled_count != 0) {
                copy_array(attention.qsa_pooled_keys, output.pooled);
            }
        }
        const PleState& ple = source.layers[1].ple;
        if (!ple.convolution_initialized) {
            throw std::runtime_error("cannot import uninitialized PLE state");
        }
        copy_array(ple.convolution, ple_convolution_);
        ple_ngram_state_ = ple.ngram;
    }

    std::vector<std::uint16_t> decode_trunk(
        std::uint32_t token, std::span<const std::uint16_t> initial_stream,
        bool reset, double* gpu_ms) {
        if (initial_stream.size() != 10240) {
            throw std::runtime_error("persistent stream width mismatch");
        }
        if (reset) reset_all_state();
        std::memcpy(stream_.contents, initial_stream.data(), initial_stream.size_bytes());
        std::size_t group_size = 3;
        if (const char* configured = std::getenv("QWEN38_PERSISTENT_LAYER_GROUP")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(configured, &end, 10);
            if (end == configured || *end != '\0' || parsed < 1 || parsed > 48) {
                throw std::runtime_error("persistent layer group must be 1..48");
            }
            group_size = static_cast<std::size_t>(parsed);
        }
        std::vector<id<MTLCommandBuffer>> commands;
        commands.reserve((48 + group_size - 1) / group_size);
        id<MTLCommandBuffer> command = nil;
        for (std::size_t index = 0; index < 48; ++index) {
            if (index % group_size == 0) command = [queue_ commandBuffer];
            id<MTLBuffer> layer_input = index == 0 ? stream_ : output_stream_;
            if (index == 1) {
                encode_ple(command, token, layer_input);
                layer_input = ple_stream_;
            }
            const std::string layer =
                "language_model.model.layers." + std::to_string(index);
            encode_hc_read(command, layer_input, layer + ".attn_hyper_connection");
            if (index % 4 == 3) {
                if (attention_states_[index].token_count -
                        attention_states_[index].cold_count >= hot_capacity) {
                    throw std::runtime_error("persistent attention hot slab requires Q8 flush");
                }
                encode_attention(command, layer + ".self_attn", attention_states_[index]);
            } else {
                encode_gdn(command, layer + ".linear_attn", gdn_states_[index]);
            }
            encode_hc_write(command, layer_input, block_output_, post_attention_);
            encode_mlp(command, layer, index >= 10 && index <= 33);
            if ((index + 1) % group_size == 0 || index + 1 == 48) {
                [command commit];
                commands.push_back(command);
            }
        }
        [commands.back() waitUntilCompleted];
        double total_gpu_ms = 0.0;
        for (id<MTLCommandBuffer> completed : commands) {
            if (completed.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error(completed.error.localizedDescription.UTF8String);
            }
            total_gpu_ms += (completed.GPUEndTime - completed.GPUStartTime) * 1000.0;
        }
        if (gpu_ms != nullptr) *gpu_ms = total_gpu_ms;
        const auto* begin = static_cast<const std::uint16_t*>(output_stream_.contents);
        return {begin, begin + 10240};
    }

    std::vector<std::uint16_t> embed(std::uint32_t token, double* gpu_ms) {
        if (token >= config_.vocabulary_size) throw std::out_of_range("token id out of range");
        const std::string base = "language_model.model.embed_tokens";
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("embedding_q4_stream")];
        bind_row(encoder, base + ".weight", token, 0);
        bind_row(encoder, base + ".scales", token, 1);
        bind_row(encoder, base + ".biases", token, 2);
        [encoder setBuffer:stream_ offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(2560, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
        }
        if (gpu_ms != nullptr) *gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        const auto* begin = static_cast<const std::uint16_t*>(stream_.contents);
        return {begin, begin + 10240};
    }

    GreedyResult head(std::span<const std::uint16_t> stream) {
        if (stream.size() != 10240) throw std::runtime_error("head stream width mismatch");
        std::memcpy(stream_.contents, stream.data(), stream.size_bytes());
        const auto started = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        encode_hc_read_final(command, stream_);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("lm_head_q4")];
        [encoder setBuffer:mixed_ offset:0 atIndex:0];
        bind_projection(encoder, "language_model.lm_head", 1);
        [encoder setBuffer:head_logits_ offset:0 atIndex:4];
        const std::uint32_t rows = static_cast<std::uint32_t>(config_.vocabulary_size);
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
        }
        const auto* logits = static_cast<const float*>(head_logits_.contents);
        std::uint32_t best = 0;
        std::uint32_t second = 1;
        if (logits[second] > logits[best]) std::swap(best, second);
        for (std::uint32_t index = 2; index < rows; ++index) {
            if (logits[index] > logits[best]) {
                second = best;
                best = index;
            } else if (logits[index] > logits[second]) {
                second = index;
            }
        }
        return {
            .token = best,
            .alternative_token = second,
            .logit = logits[best],
            .gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0,
            .wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count(),
        };
    }

    void encode_hc_read_final(id<MTLCommandBuffer> command, id<MTLBuffer> input) {
        const std::string base = "language_model.model.hyper_connection_mixer";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("hc_normalize")];
        [encoder setBuffer:input offset:0 atIndex:0];
        bind(encoder, base + ".hc_norm.weight", 1);
        [encoder setBuffer:normalized_ offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline("hc_down_only")];
        [encoder setBuffer:normalized_ offset:0 atIndex:0];
        bind_projection(encoder, base + ".input_mix_weight_down", 1);
        [encoder setBuffer:activation_ offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(80, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
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

    GreedyResult greedy_decode(std::uint32_t token, bool reset) {
        const auto started = std::chrono::steady_clock::now();
        double embedding_gpu = 0.0, trunk_gpu = 0.0;
        std::vector<std::uint16_t> initial = embed(token, &embedding_gpu);
        std::vector<std::uint16_t> trunk = decode_trunk(token, initial, reset, &trunk_gpu);
        GreedyResult result = head(trunk);
        result.gpu_ms += embedding_gpu + trunk_gpu;
        result.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }

    Inventory inventory_;
    id<MTLDevice> device_{nil};
    id<MTLLibrary> library_{nil};
    id<MTLCommandQueue> queue_{nil};
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::unordered_map<std::string, std::unique_ptr<Shard>> shards_;
    std::unordered_map<std::string, SharedTensor> shared_tensors_;
    std::unordered_map<std::string, std::string> weight_map_;
    ModelConfig config_;
    std::unique_ptr<NgramHash> ple_hash_;
    std::unique_ptr<NgramTable> ple_table_;
    NgramState ple_ngram_state_;
    std::array<GdnState, 48> gdn_states_;
    std::array<AttentionState, 48> attention_states_;
    static constexpr std::uint32_t hot_capacity = 8192;
    id<MTLBuffer> stream_{nil}, normalized_{nil}, activation_{nil}, injection_{nil}, mixed_{nil};
    id<MTLBuffer> projected_{nil}, query_{nil}, key_{nil}, value_{nil}, decay_{nil}, beta_{nil};
    id<MTLBuffer> recurrent_output_{nil}, gated_{nil}, block_output_{nil};
    id<MTLBuffer> post_attention_{nil}, expert_ids_{nil}, route_weights_{nil};
    id<MTLBuffer> routed_hidden_{nil}, shared_hidden_{nil}, shared_scale_{nil};
    id<MTLBuffer> moe_output_{nil}, zero_output_{nil}, healing_hidden_{nil};
    id<MTLBuffer> output_stream_{nil}, router_logits_{nil};
    id<MTLBuffer> attention_projected_{nil}, selector_query_{nil}, attention_query_{nil};
    id<MTLBuffer> attention_key_{nil}, attention_values_{nil}, attention_gated_{nil};
    id<MTLBuffer> rope_cos_{nil}, rope_sin_{nil}, pool_rope_cos_{nil}, pool_rope_sin_{nil};
    id<MTLBuffer> qsa_scores_{nil}, selected_{nil}, temp_scores_a_{nil}, temp_scores_b_{nil};
    id<MTLBuffer> temp_ids_a_{nil}, temp_ids_b_{nil}, q8_dummy_{nil};
    id<MTLBuffer> ple_embedding_{nil}, ple_projected_{nil}, ple_stream_{nil};
    id<MTLBuffer> ple_convolution_{nil};
    id<MTLBuffer> head_logits_{nil};
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
    const ModelManifest& manifest, MlxTensorStore* shared_weights) {
    if (!supports(manifest.config())) return nullptr;
    return std::unique_ptr<PersistentMetalBackend>(
        new PersistentMetalBackend(std::make_unique<Impl>(manifest, shared_weights)));
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

std::vector<std::uint16_t> PersistentMetalBackend::decode_attention_layer(
    const std::size_t layer, const std::span<const std::uint16_t> stream,
    const bool reset_state, double* gpu_ms) {
    return impl_->decode_attention_layer(layer, stream, reset_state, gpu_ms);
}

std::vector<std::uint16_t> PersistentMetalBackend::decode_trunk(
    const std::uint32_t token, const std::span<const std::uint16_t> initial_stream,
    const bool reset_state, double* gpu_ms) {
    return impl_->decode_trunk(token, initial_stream, reset_state, gpu_ms);
}

std::vector<std::uint16_t> PersistentMetalBackend::decode_ple(
    const std::uint32_t token, const std::span<const std::uint16_t> stream,
    const bool reset_state, double* gpu_ms) {
    return impl_->decode_ple(token, stream, reset_state, gpu_ms);
}

PersistentMetalBackend::GreedyResult PersistentMetalBackend::greedy_decode(
    const std::uint32_t token, const bool reset_state) {
    return impl_->greedy_decode(token, reset_state);
}

std::vector<std::uint16_t> PersistentMetalBackend::embed(
    const std::uint32_t token, double* gpu_ms) {
    return impl_->embed(token, gpu_ms);
}

PersistentMetalBackend::GreedyResult PersistentMetalBackend::greedy_head(
    const std::span<const std::uint16_t> stream) {
    return impl_->head(stream);
}

void PersistentMetalBackend::import_state(const ModelDecodeState& state) {
    impl_->import_state(state);
}

void PersistentMetalBackend::prepare_shared_weights() {
    impl_->prepare_shared_weights();
}

void PersistentMetalBackend::release_shared_weights() {
    impl_->release_shared_weights();
}

} // namespace qwen38
