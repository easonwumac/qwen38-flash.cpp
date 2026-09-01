#include "qwen38/model_manifest.hpp"

#include "qwen38/json.hpp"

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
    result.config_.hyper_connection_count = size_value(text.at("hc_count"), "hc_count");
    result.config_.hyper_connection_low_rank = size_value(text.at("hc_lowrank"), "hc_lowrank");
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

    const Json* quantization = root.find("quantization");
    if (quantization == nullptr) quantization = root.find("quantization_config");
    if (quantization == nullptr) throw std::runtime_error("model is missing quantization metadata");
    result.config_.quantization_bits = size_value(quantization->at("bits"), "quantization.bits");
    result.config_.quantization_group_size = size_value(
        quantization->at("group_size"), "quantization.group_size");

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
    return result;
}

TensorView TensorStore::tensor(const std::string_view name) {
    std::scoped_lock lock(mutex_);
    const auto mapping = manifest_.weight_map().find(std::string(name));
    if (mapping == manifest_.weight_map().end()) {
        throw std::out_of_range("tensor is not present in model index: " + std::string(name));
    }
    auto shard = shards_.find(mapping->second);
    if (shard == shards_.end()) {
        auto file = std::make_unique<SafetensorsFile>(manifest_.directory() / mapping->second);
        shard = shards_.emplace(mapping->second, std::move(file)).first;
    }
    return shard->second->tensor(name);
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
