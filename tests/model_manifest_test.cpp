#include "qwen38/model_manifest.hpp"
#include "test.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

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
        "hc_lowrank":320,"rms_norm_eps":0.000001,
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

    qwen38::ModelManifest manifest = qwen38::ModelManifest::load(directory);
    QWEN38_CHECK(manifest.config().hidden_size == 2560);
    QWEN38_CHECK(manifest.config().layer_count == 48);
    QWEN38_CHECK(manifest.config().expert_count == 288);
    QWEN38_CHECK(manifest.config().max_context_tokens == 262144);
    QWEN38_CHECK(manifest.config().quantization_bits == 4);
    QWEN38_CHECK(manifest.config().hyper_connection_count == 4);
    QWEN38_CHECK(manifest.config().hyper_connection_low_rank == 320);
    QWEN38_CHECK(manifest.config().layer_types.at(3) == "full_attention");
    QWEN38_CHECK(manifest.declared_weight_bytes() == 2);

    qwen38::TensorStore store(std::move(manifest));
    QWEN38_CHECK(store.open_shard_count() == 0);
    const auto tensor = store.tensor("tensor");
    QWEN38_CHECK(store.open_shard_count() == 1);
    QWEN38_CHECK(tensor.bytes.size() == 2);
    QWEN38_CHECK(std::to_integer<unsigned char>(tensor.bytes[1]) == 9);
    QWEN38_CHECK(store.mapped_virtual_bytes() > 2);

    std::filesystem::remove_all(directory);
}
