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

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open prompt file: " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::uint32_t top1(const qwen38::MlxArray& logits) {
    const std::vector<float> values = logits.astype(MLX_FLOAT32).to_float32();
    return static_cast<std::uint32_t>(
        std::distance(values.begin(), std::ranges::max_element(values)));
}

struct Result {
    std::uint32_t token_id;
    double milliseconds;
};

Result serial_prefill(
    qwen38::QwenModel& model,
    const std::span<const std::uint32_t> tokens) {
    qwen38::ModelDecodeState state = model.make_state();
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        static_cast<void>(model.consume_decode_capture(tokens[index], state));
    }
    const std::uint32_t predicted = top1(model.forward_decode(tokens.back(), state));
    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return {predicted, milliseconds};
}

Result batched_prefill(
    qwen38::QwenModel& model,
    const std::span<const std::uint32_t> tokens,
    const std::size_t chunk_rows,
    const std::size_t serial_tail) {
    qwen38::ModelDecodeState state = model.make_state();
    const std::size_t prefill_rows = tokens.size() - 1;
    const std::size_t batched_rows = prefill_rows - serial_tail;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t offset = 0; offset < batched_rows; offset += chunk_rows) {
        const std::size_t count = std::min(chunk_rows, batched_rows - offset);
        static_cast<void>(model.prefill_chunk(tokens.subspan(offset, count), state));
    }
    for (std::size_t index = batched_rows; index < prefill_rows; ++index) {
        static_cast<void>(model.consume_decode_capture(tokens[index], state));
    }
    const std::uint32_t predicted = top1(model.forward_decode(tokens.back(), state));
    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return {predicted, milliseconds};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL USER_PROMPT.txt CHUNK_ROWS [SERIAL_TAIL [EXPECTED_TOKEN_ID]]\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model prefill tests must run through devtools/memory_guard.py");
        }
        const std::filesystem::path model_path = argv[1];
        const std::size_t chunk_rows = std::stoul(argv[3]);
        if (chunk_rows == 0 || chunk_rows > 64) {
            throw std::runtime_error("CHUNK_ROWS must be between 1 and 64");
        }
        const std::size_t serial_tail = argc >= 5 ? std::stoul(argv[4]) : 0;
        const std::uint32_t expected = argc == 6
            ? static_cast<std::uint32_t>(std::stoul(argv[5]))
            : 40U;

        const std::string user_prompt = read_file(argv[2]);
        qwen38::ChatTemplateOptions template_options;
        template_options.enable_thinking = false;
        const std::string rendered = qwen38::render_chat_prompt({
            {qwen38::ChatRole::user, user_prompt, std::nullopt},
        }, template_options);
        const qwen38::Tokenizer tokenizer = qwen38::Tokenizer::load(model_path);
        const std::vector<std::uint32_t> tokens = tokenizer.encode(rendered);
        if (tokens.size() < 2 || serial_tail >= tokens.size()) {
            throw std::runtime_error("rendered prompt is too short for requested serial tail");
        }

        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(model_path));
        qwen38::QwenModel model(tensors);
        {
            qwen38::ModelDecodeState warmup = model.make_state();
            static_cast<void>(model.consume_decode_capture(tokens[0], warmup));
            static_cast<void>(model.consume_decode_capture(tokens[1], warmup));
        }
        qwen38::MlxArray::clear_cache();
        const Result serial = serial_prefill(model, tokens);
        qwen38::MlxArray::clear_cache();
        const Result batched = batched_prefill(model, tokens, chunk_rows, serial_tail);

        const std::array<std::uint32_t, 1> serial_id{serial.token_id};
        const std::array<std::uint32_t, 1> batched_id{batched.token_id};
        std::cout << "{\"prompt_tokens\":" << tokens.size()
                  << ",\"chunk_rows\":" << chunk_rows
                  << ",\"serial_tail\":" << serial_tail
                  << ",\"serial_ms\":" << serial.milliseconds
                  << ",\"batched_ms\":" << batched.milliseconds
                  << ",\"speedup\":" << serial.milliseconds / batched.milliseconds
                  << ",\"serial_token_id\":" << serial.token_id
                  << ",\"serial_text\":\"" << tokenizer.decode(serial_id) << "\""
                  << ",\"batched_token_id\":" << batched.token_id
                  << ",\"batched_text\":\"" << tokenizer.decode(batched_id) << "\""
                  << ",\"reference_token_id\":" << expected
                  << ",\"serial_reference_match\":"
                  << (serial.token_id == expected ? "true" : "false")
                  << ",\"batched_reference_match\":"
                  << (batched.token_id == expected ? "true" : "false") << "}\n";
        return batched.token_id == expected ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-prefill-prompt-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
