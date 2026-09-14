#include "qwen38/model_manifest.hpp"
#include "qwen38/persistent_metal_backend.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: qwen38-persistent-metal-backend-smoke MODEL");
        }
        const qwen38::ModelManifest manifest = qwen38::ModelManifest::load(argv[1]);
        auto backend = qwen38::PersistentMetalBackend::create(manifest);
        if (!backend) throw std::runtime_error("model is not eligible for persistent Metal");
        const auto& inventory = backend->inventory();
        std::cout << "pipelines " << inventory.pipeline_count << '\n'
                  << "shards " << inventory.shard_count << '\n'
                  << "mapped_weight_bytes " << inventory.mapped_weight_bytes << '\n';
        return inventory.pipeline_count == 29 && inventory.shard_count != 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
