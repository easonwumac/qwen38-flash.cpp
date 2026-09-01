#include "qwen38/api.hpp"
#include "qwen38/model_manifest.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY [--tensor NAME]\n";
        return EXIT_FAILURE;
    }
    std::optional<std::string> tensor_name;
    if (argc == 4) {
        if (std::string(argv[2]) != "--tensor") {
            std::cerr << "expected --tensor before tensor name\n";
            return EXIT_FAILURE;
        }
        tensor_name = argv[3];
    }
    try {
        qwen38::TensorStore store(qwen38::ModelManifest::load(argv[1]));
        const auto& manifest = store.manifest();
        const auto& config = manifest.config();
        std::cout << "{\"architecture\":\"" << qwen38::json_escape(config.architecture)
                  << "\",\"model_type\":\"" << qwen38::json_escape(config.model_type)
                  << "\",\"hidden_size\":" << config.hidden_size
                  << ",\"layers\":" << config.layer_count
                  << ",\"experts\":" << config.expert_count
                  << ",\"experts_per_token\":" << config.experts_per_token
                  << ",\"vocabulary_size\":" << config.vocabulary_size
                  << ",\"max_context_tokens\":" << config.max_context_tokens
                  << ",\"quantization_bits\":" << config.quantization_bits
                  << ",\"quantization_group_size\":" << config.quantization_group_size
                  << ",\"mtp_layers\":" << config.mtp_layer_count
                  << ",\"indexed_tensors\":" << manifest.weight_map().size()
                  << ",\"declared_weight_bytes\":" << manifest.declared_weight_bytes();
        if (tensor_name.has_value()) {
            const qwen38::TensorView tensor = store.tensor(*tensor_name);
            std::cout << ",\"tensor\":{\"name\":\"" << qwen38::json_escape(*tensor_name)
                      << "\",\"dtype\":\"" << qwen38::json_escape(tensor.dtype)
                      << "\",\"shape\":[";
            for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << tensor.shape[i];
            }
            std::cout << "],\"bytes\":" << tensor.bytes.size() << '}';
        }
        std::cout << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-inspect: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
