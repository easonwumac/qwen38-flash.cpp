#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace {

std::uint32_t argmax(const qwen38::MlxArray& logits) {
    const std::vector<float> values = logits.astype(MLX_FLOAT32).to_float32();
    return static_cast<std::uint32_t>(
        std::distance(values.begin(), std::ranges::max_element(values)));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
            throw std::runtime_error(
                "full-model MTP smoke tests must run through devtools/memory_guard.py");
        }
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        qwen38::QwenModel target(tensors);
        qwen38::QwenMtpHead mtp(tensors);
        auto target_state = target.make_state();
        auto mtp_state = mtp.make_state();

        qwen38::TargetDecodeStep target_step =
            target.forward_decode_capture(9419, target_state);
        const std::uint32_t target_token = argmax(target_step.logits);
        qwen38::MtpDecodeStep draft =
            mtp.forward_decode(target_step.pre_mixer_stream, target_token, 1, mtp_state);
        const std::uint32_t draft_token = argmax(draft.logits);
        const std::vector<int> stream_shape = draft.pre_mixer_stream.shape();
        if (stream_shape != std::vector<int>({1, 1, 10240})) {
            throw std::runtime_error("MTP output stream shape mismatch");
        }
        bool rejected_position_gap = false;
        try {
            static_cast<void>(
                mtp.forward_decode(draft.pre_mixer_stream, draft_token, 1, mtp_state));
        } catch (const std::runtime_error&) {
            rejected_position_gap = true;
        }
        if (!rejected_position_gap) {
            throw std::runtime_error("MTP position-gap guard did not engage");
        }
        std::cout << "{\"target_token\":" << target_token
                  << ",\"draft_token\":" << draft_token
                  << ",\"mtp_rows\":" << mtp_state.row_count
                  << ",\"position_gap_rejected\":true"
                  << ",\"stream_width\":" << stream_shape.back()
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-mtp-head-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
