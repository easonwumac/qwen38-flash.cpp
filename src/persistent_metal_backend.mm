#include "qwen38/persistent_metal_backend.hpp"

#include "persistent_metal_kernels.hpp"
#include "qwen38/safetensors.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
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
                pipelines_.push_back(pipeline);
            }

            std::unordered_set<std::string> shard_names;
            shard_names.reserve(manifest.weight_map().size());
            for (const auto& [tensor, shard] : manifest.weight_map()) {
                static_cast<void>(tensor);
                shard_names.insert(shard);
            }
            shards_.reserve(shard_names.size());
            for (const std::string& name : shard_names) {
                auto file = std::make_unique<SafetensorsFile>(manifest.directory() / name);
                inventory_.mapped_weight_bytes += file->mapped_bytes();
                shards_.emplace(name, std::move(file));
            }
            inventory_.pipeline_count = pipelines_.size();
            inventory_.shard_count = shards_.size();
        }
    }

    Inventory inventory_;
    id<MTLDevice> device_{nil};
    id<MTLLibrary> library_{nil};
    std::vector<id<MTLComputePipelineState>> pipelines_;
    std::unordered_map<std::string, std::unique_ptr<SafetensorsFile>> shards_;
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

} // namespace qwen38
