#include "qwen38/mtp_head.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct Similarity {
    double cosine{0.0};
    double max_absolute{0.0};
};

Similarity compare(const qwen38::MlxArray& actual_array, const qwen38::MlxArray& expected_array) {
    const std::vector<float> actual = actual_array.astype(MLX_FLOAT32).to_float32();
    const std::vector<float> expected = expected_array.astype(MLX_FLOAT32).to_float32();
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error("MTP fixture tap shape mismatch");
    }
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    double max_absolute = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        dot += static_cast<double>(actual[index]) * expected[index];
        actual_norm += static_cast<double>(actual[index]) * actual[index];
        expected_norm += static_cast<double>(expected[index]) * expected[index];
        max_absolute = std::max(
            max_absolute,
            std::abs(static_cast<double>(actual[index]) - expected[index]));
    }
    return {
        .cosine = dot / std::sqrt(actual_norm * expected_norm),
        .max_absolute = max_absolute,
    };
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL_DIRECTORY FIXTURE.safetensors [HISTORY.safetensors]\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "real-weight MTP fixture tests must run through devtools/memory_guard.py");
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::MlxSafetensors fixture(argv[2]);
        qwen38::QwenMtpHead mtp(tensors);
        auto state = mtp.make_state();
        std::size_t primed_rows = 0;
        if (argc == 4) {
            qwen38::MlxSafetensors history(argv[3]);
            qwen38::MlxArray history_stream = history.tensor("stream_prev");
            qwen38::MlxArray history_tokens = history.tensor("token_ids");
            qwen38::MlxArray history_positions = history.tensor("query_positions");
            const std::vector<int> shape = history_stream.shape();
            if (shape.size() != 3 || shape[0] != 1 || shape[2] != 10240 || shape[1] < 1) {
                throw std::runtime_error("MTP history fixture shape mismatch");
            }
            const std::vector<int> stream_strides{1, 1, 1};
            const std::vector<int> row_strides{1, 1};
            for (int row = 0; row + 1 < shape[1]; ++row) {
                const std::vector<int> stream_start{0, row, 0};
                const std::vector<int> stream_stop{1, row + 1, 10240};
                const std::vector<int> row_start{0, row};
                const std::vector<int> row_stop{1, row + 1};
                qwen38::MlxArray row_stream = history_stream.slice(
                    stream_start, stream_stop, stream_strides);
                const std::uint32_t row_token = history_tokens.slice(
                    row_start, row_stop, row_strides).item_uint32();
                const std::size_t row_position = static_cast<std::size_t>(
                    history_positions.slice(row_start, row_stop, row_strides)
                        .astype(MLX_FLOAT32).item_float32());
                mtp.consume_decode(row_stream, row_token, row_position, state);
                ++primed_rows;
            }
        }
        qwen38::MlxArray stream = fixture.tensor("stream_prev");
        const std::uint32_t token = fixture.tensor("mtp_input_token_id").item_uint32();
        const std::size_t position = static_cast<std::size_t>(
            fixture.tensor("query_position").astype(MLX_FLOAT32).item_float32());
        const std::uint32_t expected_top1 = fixture.tensor("mtp_head_top1").item_uint32();
        qwen38::MtpTrace trace;
        qwen38::MtpDecodeStep output =
            mtp.forward_decode(stream, token, position, state, &trace);

        const std::vector<float> actual = output.logits.astype(MLX_FLOAT32).to_float32();
        const std::vector<float> expected =
            fixture.tensor("logits").astype(MLX_FLOAT32).to_float32();
        if (actual.size() != expected.size() || actual.empty()) {
            throw std::runtime_error("MTP fixture logits shape mismatch");
        }
        const auto best = std::ranges::max_element(actual);
        const std::uint32_t actual_top1 = static_cast<std::uint32_t>(
            std::distance(actual.begin(), best));
        const Similarity logits = compare(output.logits, fixture.tensor("logits"));
        const Similarity combined = compare(trace.combined_stream, fixture.tensor("combined_stream"));
        const Similarity attention_mixed = compare(trace.layer.attention_mixed, fixture.tensor("attn_hc_mixed"));
        const Similarity attention_output = compare(trace.layer.attention_output, fixture.tensor("attn_out"));
        const Similarity post_attention = compare(trace.layer.post_attention_stream, fixture.tensor("post_attn_stream"));
        const Similarity mlp_mixed = compare(trace.layer.mlp_mixed, fixture.tensor("mlp_hc_mixed"));
        const Similarity mlp_output = compare(trace.layer.mlp_output, fixture.tensor("mlp_out"));
        const Similarity post_mlp = compare(trace.layer.post_mlp_stream, fixture.tensor("post_mlp_stream"));
        const Similarity final_mixed = compare(trace.final_mixed, fixture.tensor("final_mixed"));
        std::cout << "{\"token\":" << token
                  << ",\"position\":" << position
                  << ",\"primed_rows\":" << primed_rows
                  << ",\"actual_top1\":" << actual_top1
                  << ",\"expected_top1\":" << expected_top1
                  << ",\"top1_match\":" << (actual_top1 == expected_top1 ? "true" : "false")
                  << ",\"cosine\":" << logits.cosine
                  << ",\"max_absolute\":" << logits.max_absolute
                  << ",\"taps\":{"
                  << "\"combined\":" << combined.cosine
                  << ",\"attention_mixed\":" << attention_mixed.cosine
                  << ",\"attention_output\":" << attention_output.cosine
                  << ",\"post_attention\":" << post_attention.cosine
                  << ",\"mlp_mixed\":" << mlp_mixed.cosine
                  << ",\"mlp_output\":" << mlp_output.cosine
                  << ",\"post_mlp\":" << post_mlp.cosine
                  << ",\"final_mixed\":" << final_mixed.cosine << "}}\n";
        return actual_top1 == expected_top1 && logits.cosine > 0.995
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-fixture-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
