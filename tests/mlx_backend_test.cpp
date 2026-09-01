#include "qwen38/mlx_backend.hpp"
#include "qwen38/model.hpp"
#include "qwen38/mtp_verifier.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    const std::array<int, 2> shape{2, 2};
    const std::array<float, 4> left_values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> right_values{5.0F, 6.0F, 7.0F, 8.0F};
    const auto left = qwen38::MlxArray::from_float32(left_values, shape);
    const auto right = qwen38::MlxArray::from_float32(right_values, shape);
    const auto sum = qwen38::MlxArray::add(left, right).to_float32();
    const auto product = qwen38::MlxArray::matmul(left, right).to_float32();
    const auto elementwise = qwen38::MlxArray::multiply(left, right).to_float32();
    if (sum != std::vector<float>({6.0F, 8.0F, 10.0F, 12.0F})) {
        std::cerr << "MLX add mismatch\n";
        return 1;
    }
    if (product != std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F})) {
        std::cerr << "MLX matmul mismatch\n";
        return 1;
    }
    if (elementwise != std::vector<float>({5.0F, 12.0F, 21.0F, 32.0F})) {
        std::cerr << "MLX multiply mismatch\n";
        return 1;
    }
    const std::array<int, 3> expanded_shape{1, 2, 2};
    const std::array<int, 3> repetitions{2, 1, 1};
    const auto expanded = left.reshape(expanded_shape).tile(repetitions);
    if (expanded.shape() != std::vector<int>({2, 2, 2}) ||
        expanded.mean_axis(0).to_float32() != std::vector<float>({1, 2, 3, 4})) {
        std::cerr << "MLX reshape/tile/mean mismatch\n";
        return 1;
    }
    const auto transposed = left.transpose().to_float32();
    if (transposed != std::vector<float>({1, 3, 2, 4})) {
        std::cerr << "MLX transpose mismatch\n";
        return 1;
    }
    if (left.sum_axis(0).to_float32() != std::vector<float>({4, 6})) {
        std::cerr << "MLX sum mismatch\n";
        return 1;
    }
    const std::array<int, 2> slice_start{1, 0};
    const std::array<int, 2> slice_stop{3, 2};
    const std::array<int, 2> slice_strides{1, 1};
    if (left.repeat_axis(2, 0).slice(slice_start, slice_stop, slice_strides).to_float32() !=
        std::vector<float>({1, 2, 3, 4})) {
        std::cerr << "MLX repeat/slice mismatch\n";
        return 1;
    }
    const std::array<int, 2> zero_shape{1, 2};
    const auto zero = qwen38::MlxArray::zeros(zero_shape, MLX_FLOAT32);
    const auto joined = qwen38::MlxArray::concatenate(zero, left, 0);
    if (joined.shape() != std::vector<int>({3, 2}) ||
        qwen38::MlxArray::subtract(left, right).to_float32() !=
            std::vector<float>({-4, -4, -4, -4})) {
        std::cerr << "MLX zeros/concatenate/subtract mismatch\n";
        return 1;
    }
    const auto sigmoid = left.sigmoid().to_float32();
    if (std::abs(sigmoid.front() - 0.7310586F) > 1.0e-5F) {
        std::cerr << "MLX sigmoid mismatch\n";
        return 1;
    }
    const auto swapped = left.swapaxes(0, 1).to_float32();
    const auto softmax = left.softmax_axis(-1).to_float32();
    if (swapped != std::vector<float>({1, 3, 2, 4}) ||
        std::abs(softmax[0] + softmax[1] - 1.0F) > 1.0e-5F ||
        std::abs(softmax[2] + softmax[3] - 1.0F) > 1.0e-5F) {
        std::cerr << "MLX swapaxes/softmax mismatch\n";
        return 1;
    }
    const std::array<float, 4> signed_values{-4.0F, -1.0F, 0.0F, 9.0F};
    const auto signed_array = qwen38::MlxArray::from_float32(signed_values, shape);
    if (signed_array.absolute().to_float32() != std::vector<float>({4, 1, 0, 9}) ||
        signed_array.sign().to_float32() != std::vector<float>({-1, -1, 0, 1}) ||
        signed_array.absolute().square_root().to_float32() !=
            std::vector<float>({2, 1, 0, 3}) ||
        qwen38::MlxArray::maximum(signed_array, zero).to_float32() !=
            std::vector<float>({0, 0, 0, 9})) {
        std::cerr << "MLX abs/sign/sqrt/maximum mismatch\n";
        return 1;
    }
    qwen38::MlxArray shared;
    {
        const auto owner = qwen38::MlxArray::from_float32(left_values, shape);
        shared = owner.share();
    }
    if (shared.to_float32() != std::vector<float>({1, 2, 3, 4})) {
        std::cerr << "MLX shared reference mismatch\n";
        return 1;
    }
    qwen38::ModelDecodeState state(1);
    state.token_count = 7;
    state.layers[0].linear_attention.initialized = true;
    state.layers[0].linear_attention.convolution =
        qwen38::MlxArray::from_float32(left_values, shape);
    state.layers[0].linear_attention.recurrent =
        qwen38::MlxArray::from_float32(right_values, shape);
    state.layers[0].ple.ngram.previous = {11, 22};
    state.layers[0].ple.ngram.initialized = true;
    state.layers[0].ple.convolution_initialized = true;
    state.layers[0].ple.convolution = state.layers[0].linear_attention.convolution.share();
    auto snapshot = qwen38::snapshot_decode_state(state);
    state = qwen38::ModelDecodeState(1);
    if (snapshot.token_count != 7 ||
        snapshot.layers[0].linear_attention.convolution.to_float32() !=
            std::vector<float>({1, 2, 3, 4}) ||
        snapshot.layers[0].linear_attention.recurrent.to_float32() !=
            std::vector<float>({5, 6, 7, 8}) ||
        snapshot.layers[0].ple.ngram.previous != std::array<std::uint32_t, 2>({11, 22}) ||
        snapshot.layers[0].ple.convolution.to_float32() !=
            std::vector<float>({1, 2, 3, 4})) {
        std::cerr << "model decode-state snapshot mismatch\n";
        return 1;
    }

    qwen38::MtpTargetVerification verification{.draft_count = 2, .rows = {}};
    for (std::size_t index = 0; index < 3; ++index) {
        qwen38::ModelDecodeState checkpoint(0);
        checkpoint.token_count = 10 + index;
        verification.rows.push_back({
            .greedy = {.token = static_cast<std::uint32_t>(100 + index), .logit = 0.0F},
            .pre_mixer_stream = {},
            .state_after = std::move(checkpoint),
        });
    }
    qwen38::ModelDecodeState committed(0);
    qwen38::commit_mtp_target_verification(std::move(verification), 1, committed);
    if (committed.token_count != 11) {
        std::cerr << "MTP verifier committed the wrong checkpoint\n";
        return 1;
    }

    bool rejected_invalid_commit = false;
    try {
        qwen38::MtpTargetVerification invalid{.draft_count = 2, .rows = {}};
        qwen38::commit_mtp_target_verification(std::move(invalid), 0, committed);
    } catch (const std::runtime_error&) {
        rejected_invalid_commit = true;
    }
    if (!rejected_invalid_commit) {
        std::cerr << "MTP verifier accepted an invalid row timeline\n";
        return 1;
    }
    return 0;
}
