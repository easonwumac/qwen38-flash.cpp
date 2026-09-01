#include "qwen38/chat_template.hpp"
#include "qwen38/model.hpp"
#include "qwen38/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open prompt file: " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " MODEL PROMPT.txt [CHUNK_ROWS [MAX_TOKENS]]\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "long-context tests must run through devtools/memory_guard.py");
        }
        const std::filesystem::path model_path = argv[1];
        const std::size_t chunk_rows = argc >= 4 ? std::stoul(argv[3]) : 64;
        if (chunk_rows == 0 || chunk_rows > 512) {
            throw std::runtime_error("CHUNK_ROWS must be between 1 and 512");
        }

        qwen38::ChatTemplateOptions options;
        options.enable_thinking = false;
        const std::string rendered = qwen38::render_chat_prompt(
            {
                {qwen38::ChatRole::user, read_file(argv[2]), std::nullopt},
            },
            options);
        const qwen38::Tokenizer tokenizer = qwen38::Tokenizer::load(model_path);
        std::vector<std::uint32_t> tokens = tokenizer.encode(rendered);
        if (argc == 5) {
            const std::size_t max_tokens = std::stoul(argv[4]);
            if (max_tokens < 2) {
                throw std::runtime_error("MAX_TOKENS must be at least 2");
            }
            if (tokens.size() > max_tokens)
                tokens.resize(max_tokens);
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(model_path));
        const qwen38::ModelConfig &config = tensors.manifest().config();

        qwen38::QwenModel model(tensors);
        qwen38::ModelDecodeState state = model.make_state();
        std::vector<double> layer_ms;
        const auto started = std::chrono::steady_clock::now();
        const std::size_t prefill_tokens = tokens.size() - 1;
        for (std::size_t offset = 0; offset < prefill_tokens; offset += chunk_rows) {
            const std::size_t count = std::min(chunk_rows, prefill_tokens - offset);
            static_cast<void>(model.prefill_chunk(
                std::span<const std::uint32_t>(tokens.data() + offset, count), state, &layer_ms));
        }
        const auto prefill_done = std::chrono::steady_clock::now();
        const qwen38::GreedyStep next = model.greedy_decode(tokens.back(), state);
        const auto finished = std::chrono::steady_clock::now();

        std::size_t full_attention_layers = 0;
        std::size_t qsa_layers = 0;
        std::size_t pooled_blocks = 0;
        const std::size_t selected_block_limit =
            config.indexer_budget / config.indexer_compress_ratio;
        for (std::size_t layer = 0; layer < state.layers.size(); ++layer) {
            const qwen38::SelfAttentionState &attention = state.layers[layer].full_attention;
            if (attention.token_count == 0)
                continue;
            ++full_attention_layers;
            if (full_attention_layers == 1) {
                pooled_blocks = attention.qsa_pooled_count;
            } else if (attention.qsa_pooled_count != pooled_blocks) {
                throw std::runtime_error("QSA pooled counts differ between layers");
            }
            if (attention.qsa_pooled_count > selected_block_limit)
                ++qsa_layers;
            const std::vector<int> raw_shape = attention.qsa_raw_keys.shape();
            const std::size_t complete_blocks =
                state.token_count / config.indexer_compress_ratio;
            const std::size_t expected_blocks =
                complete_blocks > selected_block_limit ? complete_blocks : 0;
            if (attention.token_count != state.token_count ||
                raw_shape != std::vector<int>({1, static_cast<int>(state.token_count),
                                               static_cast<int>(config.indexer_head_dimension)}) ||
                attention.qsa_pooled_count != expected_blocks) {
                throw std::runtime_error("QSA cache mismatch at full-attention layer " +
                                         std::to_string(layer));
            }
            if (expected_blocks != 0 &&
                attention.qsa_pooled_keys.shape() !=
                    std::vector<int>({1, static_cast<int>(expected_blocks),
                                      static_cast<int>(config.indexer_head_dimension)})) {
                throw std::runtime_error("QSA pooled cache mismatch at full-attention layer " +
                                         std::to_string(layer));
            }
        }
        if (full_attention_layers == 0) {
            throw std::runtime_error("no full-attention layers were exercised");
        }
        const bool expect_qsa =
            state.token_count / config.indexer_compress_ratio > selected_block_limit;
        if (expect_qsa && qsa_layers != full_attention_layers) {
            throw std::runtime_error("QSA did not engage on every full-attention layer");
        }

        double linear_layer_ms = 0.0;
        double full_layer_ms = 0.0;
        for (std::size_t layer = 0; layer < layer_ms.size(); ++layer) {
            (layer + 1) % 4 == 0 ? full_layer_ms += layer_ms[layer]
                                 : linear_layer_ms += layer_ms[layer];
        }

        const double prefill_ms =
            std::chrono::duration<double, std::milli>(prefill_done - started).count();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(finished - prefill_done).count();
        std::cout << "{\"prompt_tokens\":" << tokens.size() << ",\"chunk_rows\":" << chunk_rows
                  << ",\"full_attention_layers\":" << full_attention_layers
                  << ",\"qsa_layers\":" << qsa_layers
                  << ",\"qsa_engaged\":" << (qsa_layers != 0 ? "true" : "false")
                  << ",\"pooled_blocks\":" << pooled_blocks
                  << ",\"prefill_ms\":" << prefill_ms << ",\"prefill_tps\":"
                  << 1000.0 * static_cast<double>(prefill_tokens) / prefill_ms
                  << ",\"linear_layer_ms\":" << linear_layer_ms
                  << ",\"full_layer_ms\":" << full_layer_ms << ",\"decode_ms\":" << decode_ms
                  << ",\"next_token\":" << next.token << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "qwen38-long-context-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
