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

struct LogitSummary {
    std::uint32_t top1{0};
    float margin{0.0F};
};

LogitSummary summarize(const std::vector<float>& logits) {
    if (logits.size() < 2) throw std::runtime_error("MTP logits are too short");
    float first = -std::numeric_limits<float>::infinity();
    float second = first;
    std::uint32_t token = 0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (logits[index] > first) {
            second = first;
            first = logits[index];
            token = static_cast<std::uint32_t>(index);
        } else if (logits[index] > second) {
            second = logits[index];
        }
    }
    return {.top1 = token, .margin = first - second};
}

double cosine(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size() || left.empty()) {
        throw std::runtime_error("MTP comparison logits shape mismatch");
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL_A MODEL_B HISTORY.safetensors\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "real-weight MTP comparisons must run through devtools/memory_guard.py");
        }
        qwen38::MlxTensorStore tensors_a(qwen38::ModelManifest::load(argv[1]));
        qwen38::MlxTensorStore tensors_b(qwen38::ModelManifest::load(argv[2]));
        qwen38::QwenMtpHead head_a(tensors_a);
        qwen38::QwenMtpHead head_b(tensors_b);
        qwen38::MtpDecodeState state_a = head_a.make_state();
        qwen38::MtpDecodeState state_b = head_b.make_state();
        qwen38::MlxSafetensors history(argv[3]);
        qwen38::MlxArray streams = history.tensor("stream_prev");
        qwen38::MlxArray tokens = history.tensor("token_ids");
        qwen38::MlxArray positions = history.tensor("query_positions");
        const std::vector<int> shape = streams.shape();
        if (shape.size() != 3 || shape[0] != 1 || shape[2] != 10240 || shape[1] < 2) {
            throw std::runtime_error("MTP comparison history shape mismatch");
        }

        const std::vector<int> stream_strides{1, 1, 1};
        const std::vector<int> row_strides{1, 1};
        std::size_t agreements = 0;
        std::size_t target_matches_a = 0;
        std::size_t target_matches_b = 0;
        double cosine_sum = 0.0;
        double cosine_min = 1.0;
        double margin_sum_a = 0.0;
        double margin_sum_b = 0.0;
        std::vector<std::size_t> disagreements;
        for (int row = 0; row + 1 < shape[1]; ++row) {
            const std::vector<int> stream_start{0, row, 0};
            const std::vector<int> stream_stop{1, row + 1, 10240};
            const std::vector<int> row_start{0, row};
            const std::vector<int> row_stop{1, row + 1};
            const std::vector<int> next_start{0, row + 1};
            const std::vector<int> next_stop{1, row + 2};
            qwen38::MlxArray stream = streams.slice(
                stream_start, stream_stop, stream_strides);
            const std::uint32_t token = tokens.slice(
                row_start, row_stop, row_strides).item_uint32();
            const std::uint32_t target_next = tokens.slice(
                next_start, next_stop, row_strides).item_uint32();
            const std::size_t position = static_cast<std::size_t>(
                positions.slice(row_start, row_stop, row_strides)
                    .astype(MLX_FLOAT32).item_float32());
            qwen38::MtpDecodeStep step_a =
                head_a.forward_decode(stream, token, position, state_a);
            qwen38::MtpDecodeStep step_b =
                head_b.forward_decode(stream, token, position, state_b);
            const std::vector<float> logits_a =
                step_a.logits.astype(MLX_FLOAT32).to_float32();
            const std::vector<float> logits_b =
                step_b.logits.astype(MLX_FLOAT32).to_float32();
            const LogitSummary summary_a = summarize(logits_a);
            const LogitSummary summary_b = summarize(logits_b);
            if (summary_a.top1 == summary_b.top1) {
                ++agreements;
            } else if (disagreements.size() < 16) {
                disagreements.push_back(static_cast<std::size_t>(row));
            }
            target_matches_a += summary_a.top1 == target_next;
            target_matches_b += summary_b.top1 == target_next;
            const double row_cosine = cosine(logits_a, logits_b);
            cosine_sum += row_cosine;
            cosine_min = std::min(cosine_min, row_cosine);
            margin_sum_a += summary_a.margin;
            margin_sum_b += summary_b.margin;
            qwen38::MlxArray::clear_cache();
        }
        const std::size_t rows = static_cast<std::size_t>(shape[1] - 1);
        std::cout << "{\"rows\":" << rows
                  << ",\"top1_agreements\":" << agreements
                  << ",\"top1_agreement_rate\":"
                  << static_cast<double>(agreements) / static_cast<double>(rows)
                  << ",\"target_matches_a\":" << target_matches_a
                  << ",\"target_matches_b\":" << target_matches_b
                  << ",\"target_match_rate_a\":"
                  << static_cast<double>(target_matches_a) / static_cast<double>(rows)
                  << ",\"target_match_rate_b\":"
                  << static_cast<double>(target_matches_b) / static_cast<double>(rows)
                  << ",\"mean_cosine\":" << cosine_sum / static_cast<double>(rows)
                  << ",\"min_cosine\":" << cosine_min
                  << ",\"mean_margin_a\":" << margin_sum_a / static_cast<double>(rows)
                  << ",\"mean_margin_b\":" << margin_sum_b / static_cast<double>(rows)
                  << ",\"first_disagreements\":[";
        for (std::size_t index = 0; index < disagreements.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << disagreements[index];
        }
        std::cout << "]}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-head-compare: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
