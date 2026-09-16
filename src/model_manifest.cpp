#include "qwen38/model_manifest.hpp"

#include "qwen38/json.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace qwen38 {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) throw std::runtime_error("cannot read " + path.string());
    return contents.str();
}

std::size_t size_value(const Json& value, const std::string_view name) {
    const std::int64_t integer = value.as_integer();
    if (integer < 0 || static_cast<std::uint64_t>(integer) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("invalid model config value for " + std::string(name));
    }
    return static_cast<std::size_t>(integer);
}

std::size_t optional_size_value(
    const Json& object,
    const std::string_view key,
    const std::string_view name) {
    const Json* value = object.find(key);
    return value == nullptr || value->is_null() ? 0 : size_value(*value, name);
}

} // namespace

ModelManifest ModelManifest::load(const std::filesystem::path& model_directory) {
    if (!std::filesystem::is_directory(model_directory)) {
        throw std::runtime_error("model path is not a directory: " + model_directory.string());
    }

    ModelManifest result;
    result.directory_ = std::filesystem::canonical(model_directory);
    const Json root = Json::parse(read_text(result.directory_ / "config.json"));
    const auto& architectures = root.at("architectures").as_array();
    if (architectures.size() != 1) {
        throw std::runtime_error("expected exactly one model architecture");
    }
    result.config_.architecture = architectures.front().as_string();
    result.config_.model_type = root.at("model_type").as_string();
    if (result.config_.architecture != "Qwen4ExpForConditionalGeneration" ||
        result.config_.model_type != "qwen4_exp") {
        throw std::runtime_error(
            "unsupported architecture: " + result.config_.architecture + " / " +
            result.config_.model_type);
    }

    const Json& text = root.at("text_config");
    result.config_.text_model_type = text.at("model_type").as_string();
    result.config_.hidden_size = size_value(text.at("hidden_size"), "hidden_size");
    result.config_.layer_count = size_value(text.at("num_hidden_layers"), "num_hidden_layers");
    result.config_.expert_count = size_value(text.at("num_experts"), "num_experts");
    result.config_.experts_per_token = size_value(text.at("num_experts_per_tok"), "num_experts_per_tok");
    result.config_.vocabulary_size = size_value(text.at("vocab_size"), "vocab_size");
    result.config_.max_context_tokens = size_value(text.at("max_position_embeddings"), "max_position_embeddings");
    result.config_.mtp_layer_count = size_value(text.at("mtp_num_hidden_layers"), "mtp_num_hidden_layers");
    result.config_.attention_head_count = size_value(
        text.at("num_attention_heads"), "num_attention_heads");
    result.config_.key_value_head_count = size_value(
        text.at("num_key_value_heads"), "num_key_value_heads");
    result.config_.head_dimension = size_value(text.at("head_dim"), "head_dim");
    result.config_.indexer_head_count = size_value(
        text.at("indexer_n_heads"), "indexer_n_heads");
    result.config_.indexer_key_value_head_count = size_value(
        text.at("indexer_kv_heads"), "indexer_kv_heads");
    result.config_.indexer_head_dimension = size_value(
        text.at("indexer_head_dim"), "indexer_head_dim");
    result.config_.indexer_budget = size_value(
        text.at("indexer_budget"), "indexer_budget");
    result.config_.indexer_compress_ratio = size_value(
        text.at("indexer_compress_ratio"), "indexer_compress_ratio");
    if (result.config_.indexer_head_count == 0 ||
        result.config_.indexer_key_value_head_count != 1 ||
        result.config_.indexer_head_dimension == 0 ||
        result.config_.indexer_compress_ratio == 0 ||
        result.config_.indexer_budget == 0 ||
        result.config_.indexer_budget % result.config_.indexer_compress_ratio != 0) {
        throw std::runtime_error("unsupported Qwen3.8 QSA geometry");
    }
    result.config_.hyper_connection_count = size_value(text.at("hc_count"), "hc_count");
    result.config_.hyper_connection_low_rank = size_value(text.at("hc_lowrank"), "hc_lowrank");
    result.config_.moe_intermediate_size = size_value(
        text.at("moe_intermediate_size"), "moe_intermediate_size");
    result.config_.shared_expert_intermediate_size = size_value(
        text.at("shared_expert_intermediate_size"), "shared_expert_intermediate_size");
    if (const Json* normalize = text.find("norm_topk_prob")) {
        result.config_.normalize_topk_probability = normalize->as_boolean();
    }
    result.config_.linear_convolution_kernel_size = size_value(
        text.at("linear_conv_kernel_dim"), "linear_conv_kernel_dim");
    result.config_.linear_key_head_dimension = size_value(
        text.at("linear_key_head_dim"), "linear_key_head_dim");
    result.config_.linear_value_head_dimension = size_value(
        text.at("linear_value_head_dim"), "linear_value_head_dim");
    result.config_.linear_key_head_count = size_value(
        text.at("linear_num_key_heads"), "linear_num_key_heads");
    result.config_.linear_value_head_count = size_value(
        text.at("linear_num_value_heads"), "linear_num_value_heads");
    result.config_.output_gate_type = text.at("output_gate_type").as_string();
    if (result.config_.output_gate_type != "sigmoid" &&
        result.config_.output_gate_type != "silu") {
        throw std::runtime_error("unsupported output_gate_type");
    }
    result.config_.partial_rotary_factor = text.at("partial_rotary_factor").as_number();
    result.config_.rope_theta = text.at("rope_parameters").at("rope_theta").as_number();
    if (!(result.config_.partial_rotary_factor > 0.0 &&
          result.config_.partial_rotary_factor <= 1.0) ||
        !(result.config_.rope_theta > 0.0)) {
        throw std::runtime_error("invalid rotary configuration");
    }
    result.config_.ngram_size = size_value(text.at("ngram_size"), "ngram_size");
    result.config_.heads_per_ngram = size_value(text.at("heads_per_ngram"), "heads_per_ngram");
    result.config_.ngram_vocabulary_base = size_value(
        text.at("ngram_vocab_size_base"), "ngram_vocab_size_base");
    result.config_.ngram_vocabulary_divisor = size_value(
        text.at("make_ngram_vocab_size_divisible_by"),
        "make_ngram_vocab_size_divisible_by");
    result.config_.ple_embedding_dimension = size_value(text.at("ple_embed_dim"), "ple_embed_dim");
    result.config_.ple_convolution_kernel_size = size_value(
        text.at("ple_conv_kernel_size"), "ple_conv_kernel_size");
    if (const Json* pair = text.find("niwaki_ple_pair")) {
        result.config_.niwaki_ple_pair = size_value(*pair, "niwaki_ple_pair");
        const Json& quantization = text.at("niwaki_ple_quant");
        result.config_.niwaki_ple_bits = size_value(
            quantization.at("bits"), "niwaki_ple_quant.bits");
        result.config_.niwaki_ple_group_size = size_value(
            quantization.at("group_size"), "niwaki_ple_quant.group_size");
        const std::size_t physical_dimension =
            result.config_.niwaki_ple_pair * result.config_.ple_embedding_dimension /
            (result.config_.heads_per_ngram * 2);
        if (result.config_.niwaki_ple_pair < 2 ||
            result.config_.niwaki_ple_bits < 2 || result.config_.niwaki_ple_bits > 8 ||
            result.config_.niwaki_ple_group_size == 0 ||
            physical_dimension % result.config_.niwaki_ple_group_size != 0 ||
            (physical_dimension * result.config_.niwaki_ple_bits) % 32 != 0) {
            throw std::runtime_error("unsupported Niwaki paired PLE geometry");
        }
    }
    if (const Json* seed = text.find("seed")) {
        result.config_.ngram_seed = size_value(*seed, "seed");
    }
    result.config_.end_of_sequence_token = static_cast<std::uint32_t>(
        size_value(text.at("eos_token_id"), "eos_token_id"));
    if (result.config_.ngram_size != 3 || result.config_.heads_per_ngram != 8 ||
        result.config_.ple_embedding_dimension != result.config_.hidden_size) {
        throw std::runtime_error("unsupported Qwen3.8 PLE geometry");
    }
    result.config_.rms_norm_epsilon = text.at("rms_norm_eps").as_number();
    if (!(result.config_.rms_norm_epsilon > 0.0)) {
        throw std::runtime_error("rms_norm_eps must be positive");
    }
    for (const Json& layer_type : text.at("layer_types").as_array()) {
        const std::string type = layer_type.as_string();
        if (type != "linear_attention" && type != "full_attention") {
            throw std::runtime_error("unsupported layer type: " + type);
        }
        result.config_.layer_types.push_back(type);
    }
    if (result.config_.layer_types.size() != result.config_.layer_count) {
        throw std::runtime_error("layer_types count does not match num_hidden_layers");
    }
    for (const Json& layer_id : text.at("ple_layer_ids").as_array()) {
        const std::size_t one_based = size_value(layer_id, "ple_layer_ids");
        if (one_based == 0 || one_based > result.config_.layer_count) {
            throw std::runtime_error("PLE layer id is out of range");
        }
        result.config_.ple_layer_ids.push_back(one_based);
    }

    const Json* quantization = root.find("quantization");
    if (quantization == nullptr) quantization = root.find("quantization_config");
    if (quantization == nullptr) throw std::runtime_error("model is missing quantization metadata");
    result.config_.quantization_bits = size_value(quantization->at("bits"), "quantization.bits");
    result.config_.quantization_group_size = size_value(
        quantization->at("group_size"), "quantization.group_size");
    for (const auto& [module, value] : quantization->as_object()) {
        if (!value.is_object()) continue;
        const Json* bits = value.find("bits");
        const Json* group_size = value.find("group_size");
        if (bits == nullptr || group_size == nullptr) continue;
        const QuantizationSpec spec{
            .bits = size_value(*bits, "quantization override bits"),
            .group_size = size_value(*group_size, "quantization override group_size"),
        };
        if (spec.bits < 2 || spec.bits > 8 || spec.group_size == 0) {
            throw std::runtime_error("invalid quantization override for " + module);
        }
        result.quantization_overrides_.emplace(module, spec);
    }

    if (const Json* modules = root.find("vq_modules")) {
        for (const auto& [module, value] : modules->as_object()) {
            const VectorQuantizationSpec spec{
                .expert_count = size_value(value.at("experts"), "vq_modules.experts"),
                .output_dimension = size_value(value.at("out"), "vq_modules.out"),
                .input_dimension = size_value(value.at("in"), "vq_modules.in"),
                .codebook_size = size_value(value.at("k"), "vq_modules.k"),
                .vector_dimension = size_value(value.at("dim"), "vq_modules.dim"),
                .group_size = size_value(value.at("group"), "vq_modules.group"),
                .packed_bits = optional_size_value(value, "pack_bits", "vq_modules.pack_bits"),
            };
            if (spec.expert_count != result.config_.expert_count ||
                spec.output_dimension == 0 || spec.input_dimension == 0 ||
                spec.codebook_size < 2 || spec.vector_dimension == 0 ||
                spec.group_size == 0 || spec.input_dimension % spec.vector_dimension != 0 ||
                spec.input_dimension % spec.group_size != 0 ||
                (spec.packed_bits != 0 && spec.packed_bits > 31)) {
                throw std::runtime_error("unsupported vector quantization geometry for " + module);
            }
            result.vector_quantization_.emplace(module, spec);
        }
    }

    if (const Json* ple = root.find("vq_ple")) {
        const Json& geometry = ple->at("geometry");
        VectorQuantizedPleSpec spec{
            .codebook_size = size_value(geometry.at("k"), "vq_ple.geometry.k"),
            .vector_dimension = size_value(geometry.at("dim"), "vq_ple.geometry.dim"),
            .group_size = size_value(geometry.at("group"), "vq_ple.geometry.group"),
            .row_bytes = size_value(geometry.at("row_bytes"), "vq_ple.geometry.row_bytes"),
        };
        for (const Json& key : ple->at("keys").as_array()) {
            spec.keys.push_back(key.as_string());
        }
        if (spec.codebook_size < 2 || spec.vector_dimension == 0 ||
            spec.group_size == 0 || spec.row_bytes == 0 || spec.keys.empty()) {
            throw std::runtime_error("unsupported vector-quantized PLE geometry");
        }
        result.vector_quantized_ple_ = std::move(spec);
    }

    if (const Json* niwaki = root.find("niwaki")) {
        if (const Json* maps = niwaki->find("maps_unfolded")) {
            result.config_.niwaki_maps_unfolded = maps->as_boolean();
        }
        if (const Json* layers = niwaki->find("shared_only_layers")) {
            std::unordered_set<std::size_t> seen;
            for (const Json& layer : layers->as_array()) {
                const std::size_t index = size_value(layer, "niwaki.shared_only_layers");
                if (index >= result.config_.layer_count || !seen.insert(index).second) {
                    throw std::runtime_error("invalid Niwaki shared-only layer index");
                }
                result.config_.shared_only_layers.push_back(index);
            }
        }
    }

    const Json index = Json::parse(read_text(result.directory_ / "model.safetensors.index.json"));
    const std::int64_t declared_weight_bytes = index.at("metadata").at("total_size").as_integer();
    if (declared_weight_bytes < 0) {
        throw std::runtime_error("model index declares a negative total size");
    }
    result.declared_weight_bytes_ = static_cast<std::uint64_t>(declared_weight_bytes);
    std::unordered_set<std::string> shards;
    for (const auto& [tensor_name, shard_value] : index.at("weight_map").as_object()) {
        const std::string shard = shard_value.as_string();
        if (tensor_name.empty() || shard.empty() || shard.find('/') != std::string::npos ||
            shard.find('\\') != std::string::npos) {
            throw std::runtime_error("invalid model shard mapping for " + tensor_name);
        }
        result.weight_map_.emplace(tensor_name, shard);
        shards.insert(shard);
    }
    if (result.weight_map_.empty()) throw std::runtime_error("model weight map is empty");
    for (const std::string& shard : shards) {
        if (!std::filesystem::is_regular_file(result.directory_ / shard)) {
            throw std::runtime_error("model shard is missing: " + shard);
        }
    }

    // Optional compact affine-metadata sidecar used by the Qwen3.8 routed-MoE
    // kernels. It deliberately stays outside the upstream weight index so the
    // original checkpoint remains loadable by ordinary MLX tooling. Discover
    // its tensor names from the safetensors header instead of baking all 288
    // entries into another JSON manifest.
    constexpr std::array<std::string_view, 3> qmeta_sidecars{
        "model-qmeta-lossless16.safetensors",
        "model-qmeta-lossless13.safetensors",
        "model-qmeta-joint9.safetensors",
    };
    for (const std::string_view qmeta_sidecar : qmeta_sidecars) {
        const std::filesystem::path qmeta_path = result.directory_ / qmeta_sidecar;
        if (std::filesystem::is_regular_file(qmeta_path)) {
            const SafetensorsFile qmeta(qmeta_path);
            for (const auto& [tensor_name, metadata] : qmeta.tensors()) {
                static_cast<void>(metadata);
                if (tensor_name.empty() || !result.weight_map_.emplace(
                        tensor_name, std::string(qmeta_sidecar)).second) {
                    throw std::runtime_error(
                        "compact qmeta sidecar duplicates a model tensor: " + tensor_name);
                }
            }
        }
    }

    // VQLab publishes the model-native Qwen4-Exp MTP head as a single optional
    // file beside the target shards.  It is intentionally absent from the
    // upstream index, so discover it from its safetensors header.  All affine
    // tensors in this sidecar use Q6/group-32; recording exact module
    // overrides keeps the target checkpoint's selective Q8 metadata intact.
    constexpr std::string_view mtp_sidecar = "mtp-head-q6.safetensors";
    const std::filesystem::path mtp_path = result.directory_ / mtp_sidecar;
    if (std::filesystem::is_regular_file(mtp_path)) {
        const SafetensorsFile mtp(mtp_path);
        constexpr std::array<std::string_view, 7> required{
            "norm_e.weight",
            "norm_h.weight",
            "fc.weight",
            "block.self_attn.q_proj.weight",
            "block.mlp.switch_mlp.gate_proj.weight",
            "block.attn_hyper_connection.hc_norm.weight",
            "mixer.hc_norm.weight",
        };
        for (const std::string_view name : required) {
            if (!mtp.contains(name)) {
                throw std::runtime_error(
                    "native MTP sidecar is missing tensor: " + std::string(name));
            }
        }
        for (const auto& [tensor_name, metadata] : mtp.tensors()) {
            static_cast<void>(metadata);
            if (tensor_name.empty() || !result.weight_map_.emplace(
                    tensor_name, std::string(mtp_sidecar)).second) {
                throw std::runtime_error(
                    "native MTP sidecar duplicates a model tensor: " + tensor_name);
            }
            constexpr std::string_view scales_suffix = ".scales";
            if (tensor_name.ends_with(scales_suffix)) {
                result.quantization_overrides_.insert_or_assign(
                    tensor_name.substr(0, tensor_name.size() - scales_suffix.size()),
                    QuantizationSpec{.bits = 6, .group_size = 32});
            }
        }
        const std::uintmax_t sidecar_bytes = std::filesystem::file_size(mtp_path);
        if (sidecar_bytes > std::numeric_limits<std::uint64_t>::max() -
                result.declared_weight_bytes_) {
            throw std::runtime_error("native MTP sidecar size overflows model size");
        }
        result.declared_weight_bytes_ += static_cast<std::uint64_t>(sidecar_bytes);
    }
    return result;
}

QuantizationSpec ModelManifest::quantization_for(const std::string_view module) const {
    const auto found = quantization_overrides_.find(std::string(module));
    if (found != quantization_overrides_.end()) return found->second;
    constexpr std::string_view wrapper = "language_model.";
    if (module.starts_with(wrapper)) {
        const auto flat = quantization_overrides_.find(std::string(module.substr(wrapper.size())));
        if (flat != quantization_overrides_.end()) return flat->second;
    }
    return {
        .bits = config_.quantization_bits,
        .group_size = config_.quantization_group_size,
    };
}

const VectorQuantizationSpec* ModelManifest::vector_quantization_for(
    const std::string_view module) const {
    auto found = vector_quantization_.find(std::string(module));
    constexpr std::string_view wrapper = "language_model.";
    if (found == vector_quantization_.end() && module.starts_with(wrapper)) {
        found = vector_quantization_.find(std::string(module.substr(wrapper.size())));
    }
    return found == vector_quantization_.end() ? nullptr : &found->second;
}

std::string ModelManifest::resolve_tensor_name(const std::string_view name) const {
    if (weight_map_.contains(std::string(name))) return std::string(name);
    constexpr std::string_view wrapper = "language_model.";
    if (name.starts_with(wrapper)) {
        std::string flat(name.substr(wrapper.size()));
        if (weight_map_.contains(flat)) return flat;
    }
    return std::string(name);
}

bool ModelManifest::has_tensor(const std::string_view name) const {
    return weight_map_.contains(resolve_tensor_name(name));
}

TensorView TensorStore::tensor(const std::string_view name) {
    std::scoped_lock lock(mutex_);
    const std::string resolved = manifest_.resolve_tensor_name(name);
    const auto mapping = manifest_.weight_map().find(resolved);
    if (mapping == manifest_.weight_map().end()) {
        throw std::out_of_range("tensor is not present in model index: " + std::string(name));
    }
    auto shard = shards_.find(mapping->second);
    if (shard == shards_.end()) {
        auto file = std::make_unique<SafetensorsFile>(manifest_.directory() / mapping->second);
        shard = shards_.emplace(mapping->second, std::move(file)).first;
    }
    return shard->second->tensor(resolved);
}

std::size_t TensorStore::open_shard_count() const {
    std::scoped_lock lock(mutex_);
    return shards_.size();
}

std::size_t TensorStore::mapped_virtual_bytes() const {
    std::scoped_lock lock(mutex_);
    std::size_t result = 0;
    for (const auto& [name, shard] : shards_) {
        static_cast<void>(name);
        result += shard->mapped_bytes();
    }
    return result;
}

} // namespace qwen38
