#include "qwen38/model_manifest.hpp"
#include "test.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

void write_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
}

void write_shard(const std::filesystem::path& path) {
    const std::string header =
        R"({"tensor":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t size = header.size();
    for (unsigned int i = 0; i < 8; ++i) {
        output.put(static_cast<char>((size >> (i * 8U)) & 0xFFU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::array<char, 2> bytes{7, 9};
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_qmeta_sidecar(
    const std::filesystem::path& path,
    const int bits,
    const char second_tag) {
    const std::string prefix = "layer.qmeta" + std::to_string(bits);
    const std::string header =
        "{\"" + prefix + "_tags\":{\"dtype\":\"U8\",\"shape\":[2],"
        "\"data_offsets\":[0,2]},\"" + prefix +
        "_dict\":{\"dtype\":\"U32\",\"shape\":[1],\"data_offsets\":[2,6]}}";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t size = header.size();
    for (unsigned int i = 0; i < 8; ++i) {
        output.put(static_cast<char>((size >> (i * 8U)) & 0xFFU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::array<char, 6> bytes{3, second_tag, 7, 0, 0, 0};
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

} // namespace

void run_model_manifest_tests() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("qwen38-model-test-" + std::to_string(::getpid()));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    write_file(directory / "config.json", R"({
      "architectures":["Qwen4ExpForConditionalGeneration"],
      "model_type":"qwen4_exp",
      "quantization":{"bits":4,"group_size":64},
      "text_config":{
        "model_type":"qwen4_exp_text","hidden_size":2560,
        "num_hidden_layers":48,"num_experts":288,"num_experts_per_tok":10,
        "vocab_size":248320,"max_position_embeddings":262144,
        "mtp_num_hidden_layers":1,"num_attention_heads":24,
        "num_key_value_heads":2,"head_dim":256,"hc_count":4,
        "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,
        "indexer_budget":2048,"indexer_compress_ratio":4,
        "hc_lowrank":320,"rms_norm_eps":0.000001,
        "moe_intermediate_size":640,"shared_expert_intermediate_size":640,
        "linear_conv_kernel_dim":4,"linear_key_head_dim":128,
        "linear_value_head_dim":128,"linear_num_key_heads":16,
        "linear_num_value_heads":48,"output_gate_type":"sigmoid",
        "partial_rotary_factor":0.25,"rope_parameters":{"rope_theta":10000000},
        "ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,
        "make_ngram_vocab_size_divisible_by":128,"ple_embed_dim":2560,
        "ple_conv_kernel_size":4,"ple_layer_ids":[2],"eos_token_id":248044,
        "layer_types":["linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention","linear_attention","linear_attention","linear_attention",
          "full_attention"]
      }
    })");
    write_file(directory / "model.safetensors.index.json", R"({
      "metadata":{"total_size":2},
      "weight_map":{"tensor":"model-00001-of-00001.safetensors"}
    })");
    write_shard(directory / "model-00001-of-00001.safetensors");
    write_qmeta_sidecar(directory / "model-qmeta-joint9.safetensors", 9, 5);
    write_qmeta_sidecar(directory / "model-qmeta-lossless16.safetensors", 16, 6);

    qwen38::ModelManifest manifest = qwen38::ModelManifest::load(directory);
    QWEN38_CHECK(manifest.config().hidden_size == 2560);
    QWEN38_CHECK(manifest.config().layer_count == 48);
    QWEN38_CHECK(manifest.config().expert_count == 288);
    QWEN38_CHECK(manifest.config().max_context_tokens == 262144);
    QWEN38_CHECK(manifest.config().quantization_bits == 4);
    QWEN38_CHECK(manifest.config().hyper_connection_count == 4);
    QWEN38_CHECK(manifest.config().indexer_head_count == 4);
    QWEN38_CHECK(manifest.config().indexer_key_value_head_count == 1);
    QWEN38_CHECK(manifest.config().indexer_head_dimension == 128);
    QWEN38_CHECK(manifest.config().indexer_budget == 2048);
    QWEN38_CHECK(manifest.config().indexer_compress_ratio == 4);
    QWEN38_CHECK(manifest.config().hyper_connection_low_rank == 320);
    QWEN38_CHECK(manifest.config().moe_intermediate_size == 640);
    QWEN38_CHECK(manifest.config().normalize_topk_probability);
    QWEN38_CHECK(manifest.config().linear_value_head_count == 48);
    QWEN38_CHECK(manifest.config().output_gate_type == "sigmoid");
    QWEN38_CHECK(manifest.config().rope_theta == 10000000.0);
    QWEN38_CHECK(manifest.config().ngram_seed == 1234);
    QWEN38_CHECK(manifest.config().end_of_sequence_token == 248044);
    QWEN38_CHECK(manifest.config().layer_types.at(3) == "full_attention");
    QWEN38_CHECK(manifest.config().ple_layer_ids == std::vector<std::size_t>({2}));
    QWEN38_CHECK(manifest.declared_weight_bytes() == 2);
    QWEN38_CHECK(manifest.has_tensor("tensor"));
    QWEN38_CHECK(manifest.has_tensor("layer.qmeta9_tags"));
    QWEN38_CHECK(manifest.has_tensor("layer.qmeta9_dict"));
    QWEN38_CHECK(manifest.has_tensor("layer.qmeta16_tags"));
    QWEN38_CHECK(manifest.has_tensor("layer.qmeta16_dict"));
    QWEN38_CHECK(!manifest.has_tensor("missing"));

    qwen38::TensorStore store(std::move(manifest));
    QWEN38_CHECK(store.open_shard_count() == 0);
    const auto tensor = store.tensor("tensor");
    QWEN38_CHECK(store.open_shard_count() == 1);
    QWEN38_CHECK(tensor.bytes.size() == 2);
    QWEN38_CHECK(std::to_integer<unsigned char>(tensor.bytes[1]) == 9);
    const auto qmeta_tags = store.tensor("layer.qmeta9_tags");
    QWEN38_CHECK(store.open_shard_count() == 2);
    QWEN38_CHECK(qmeta_tags.bytes.size() == 2);
    QWEN38_CHECK(std::to_integer<unsigned char>(qmeta_tags.bytes[1]) == 5);
    const auto qmeta16_tags = store.tensor("layer.qmeta16_tags");
    QWEN38_CHECK(store.open_shard_count() == 3);
    QWEN38_CHECK(qmeta16_tags.bytes.size() == 2);
    QWEN38_CHECK(std::to_integer<unsigned char>(qmeta16_tags.bytes[1]) == 6);
    QWEN38_CHECK(store.mapped_virtual_bytes() > 2);

    std::filesystem::remove_all(directory);
}
