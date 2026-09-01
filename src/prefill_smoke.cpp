#include "qwen38/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

double cosine(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size() || left.empty()) {
        throw std::runtime_error("prefill comparison shape mismatch");
    }
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        dot += static_cast<double>(left[index]) * right[index];
        left_norm += static_cast<double>(left[index]) * left[index];
        right_norm += static_cast<double>(right[index]) * right[index];
    }
    return dot / std::sqrt(left_norm * right_norm);
}

std::uint32_t top1(const std::vector<float>& logits) {
    return static_cast<std::uint32_t>(
        std::distance(logits.begin(), std::ranges::max_element(logits)));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL HISTORY.safetensors ROWS [CHUNK_ROWS [SERIAL_TAIL]]\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model prefill tests must run through devtools/memory_guard.py");
        }
        const std::size_t rows = std::stoul(argv[3]);
        if (rows < 2 || rows > 64) {
            throw std::runtime_error("ROWS must be between 2 and 64");
        }
        const std::size_t chunk_rows = argc >= 5 ? std::stoul(argv[4]) : rows;
        if (chunk_rows == 0 || chunk_rows > 64) {
            throw std::runtime_error("CHUNK_ROWS must be between 1 and 64");
        }
        const std::size_t serial_tail = argc == 6 ? std::stoul(argv[5]) : 0;
        if (serial_tail > rows) {
            throw std::runtime_error("SERIAL_TAIL cannot exceed ROWS");
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel model(tensors);
        qwen38::MlxSafetensors history(argv[2]);
        qwen38::MlxArray token_array = history.tensor("token_ids");
        const std::vector<int> token_shape = token_array.shape();
        if (token_shape.size() != 2 || token_shape[0] != 1 ||
            static_cast<std::size_t>(token_shape[1]) <= rows) {
            throw std::runtime_error("prefill history does not contain enough tokens");
        }
        const std::vector<int> strides{1, 1};
        std::vector<std::uint32_t> tokens;
        tokens.reserve(rows + 1);
        for (std::size_t row = 0; row <= rows; ++row) {
            tokens.push_back(token_array.slice(
                std::vector<int>{0, static_cast<int>(row)},
                std::vector<int>{1, static_cast<int>(row + 1)},
                strides).item_uint32());
        }

        // Load/compile every target layer before the order-balanced timing.
        {
            qwen38::ModelDecodeState state = model.make_state();
            for (std::size_t row = 0; row < 2; ++row) {
                static_cast<void>(model.consume_decode_capture(tokens[row], state));
            }
        }
        qwen38::MlxArray::clear_cache();

        std::vector<float> serial_logits;
        std::vector<float> serial_stream;
        double serial_ms = 0.0;
        {
            qwen38::ModelDecodeState state = model.make_state();
            qwen38::MlxArray last_stream;
            const auto started = std::chrono::steady_clock::now();
            for (std::size_t row = 0; row < rows; ++row) {
                last_stream = model.consume_decode_capture(tokens[row], state);
            }
            serial_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            serial_stream = last_stream.astype(MLX_FLOAT32).to_float32();
            serial_logits = model.forward_decode(tokens[rows], state)
                                .astype(MLX_FLOAT32).to_float32();
        }
        qwen38::MlxArray::clear_cache();

        std::vector<float> chunk_logits;
        std::vector<float> chunk_stream;
        double chunk_ms = 0.0;
        {
            qwen38::ModelDecodeState state = model.make_state();
            const auto started = std::chrono::steady_clock::now();
            std::vector<qwen38::MlxArray> streams;
            qwen38::MlxArray last_stream;
            const std::size_t batched_rows = rows - serial_tail;
            for (std::size_t offset = 0; offset < batched_rows; offset += chunk_rows) {
                const std::size_t count = std::min(chunk_rows, batched_rows - offset);
                streams = model.prefill_chunk(
                    std::span<const std::uint32_t>(tokens.data() + offset, count), state);
                last_stream = streams.back().share();
            }
            for (std::size_t row = batched_rows; row < rows; ++row) {
                last_stream = model.consume_decode_capture(tokens[row], state);
            }
            chunk_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            chunk_stream = last_stream.astype(MLX_FLOAT32).to_float32();
            chunk_logits = model.forward_decode(tokens[rows], state)
                               .astype(MLX_FLOAT32).to_float32();
        }
        qwen38::MlxArray::clear_cache();

        std::vector<float> serial_logits_after;
        std::vector<float> serial_stream_after;
        double serial_ms_after = 0.0;
        {
            qwen38::ModelDecodeState state = model.make_state();
            qwen38::MlxArray last_stream;
            const auto started = std::chrono::steady_clock::now();
            for (std::size_t row = 0; row < rows; ++row) {
                last_stream = model.consume_decode_capture(tokens[row], state);
            }
            serial_ms_after = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            serial_stream_after = last_stream.astype(MLX_FLOAT32).to_float32();
            serial_logits_after = model.forward_decode(tokens[rows], state)
                                      .astype(MLX_FLOAT32).to_float32();
        }

        const std::uint32_t serial_top1 = top1(serial_logits);
        const std::uint32_t serial_top1_after = top1(serial_logits_after);
        const std::uint32_t chunk_top1 = top1(chunk_logits);
        const double stream_cosine = cosine(serial_stream, chunk_stream);
        const double logits_cosine = cosine(serial_logits, chunk_logits);
        const double serial_stream_cosine = cosine(serial_stream, serial_stream_after);
        const double serial_logits_cosine = cosine(serial_logits, serial_logits_after);
        const double serial_control_ms = (serial_ms + serial_ms_after) / 2.0;
        std::cout << "{\"rows\":" << rows
                  << ",\"chunk_rows\":" << chunk_rows
                  << ",\"serial_tail\":" << serial_tail
                  << ",\"serial_ms_before\":" << serial_ms
                  << ",\"serial_ms_after\":" << serial_ms_after
                  << ",\"serial_control_ms\":" << serial_control_ms
                  << ",\"serial_pp_tps\":"
                  << 1000.0 * static_cast<double>(rows) / serial_control_ms
                  << ",\"chunk_ms\":" << chunk_ms
                  << ",\"chunk_pp_tps\":"
                  << 1000.0 * static_cast<double>(rows) / chunk_ms
                  << ",\"speedup\":" << serial_control_ms / chunk_ms
                  << ",\"stream_cosine\":" << stream_cosine
                  << ",\"logits_cosine\":" << logits_cosine
                  << ",\"serial_stream_cosine\":" << serial_stream_cosine
                  << ",\"serial_logits_cosine\":" << serial_logits_cosine
                  << ",\"serial_top1\":" << serial_top1
                  << ",\"serial_top1_after\":" << serial_top1_after
                  << ",\"chunk_top1\":" << chunk_top1
                  << ",\"top1_match\":"
                  << (serial_top1 == chunk_top1 ? "true" : "false") << "}\n";
        return serial_top1 == serial_top1_after && serial_top1 == chunk_top1 &&
                stream_cosine > 0.995 && logits_cosine > 0.995 &&
                serial_stream_cosine > 0.9999 && serial_logits_cosine > 0.9999
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-prefill-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
