#include "qwen38/decode_state_io.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/model.hpp"
#include "qwen38/mtp_head.hpp"
#include "qwen38/mtp_verifier.hpp"
#include "qwen38/prefix_cache_store.hpp"

#include "../src/gdn_metal_kernels.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
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
    const std::array<const qwen38::MlxArray*, 2> eval_arrays{&left, &right};
    qwen38::MlxArray::eval_all(eval_arrays);
    if (product != std::vector<float>({19.0F, 22.0F, 43.0F, 50.0F})) {
        std::cerr << "MLX matmul mismatch\n";
        return 1;
    }
    if (elementwise != std::vector<float>({5.0F, 12.0F, 21.0F, 32.0F})) {
        std::cerr << "MLX multiply mismatch\n";
        return 1;
    }
    const std::filesystem::path safetensors_path =
        std::filesystem::temp_directory_path() /
        ("qwen38-mlx-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
    const std::array<qwen38::MlxSafetensors::NamedArray, 1> saved_arrays{{
        {.name = "left", .array = &left},
    }};
    const std::array<std::pair<std::string, std::string>, 1> saved_metadata{{
        {"purpose", "roundtrip"},
    }};
    qwen38::MlxSafetensors::save(safetensors_path, saved_arrays, saved_metadata);
    {
        qwen38::MlxSafetensors saved(safetensors_path);
        if (saved.tensor("left").to_float32() != std::vector<float>({1, 2, 3, 4}) ||
            saved.metadata("purpose") != std::optional<std::string>("roundtrip") ||
            saved.metadata("missing").has_value()) {
            std::cerr << "MLX safetensors roundtrip mismatch\n";
            std::filesystem::remove(safetensors_path);
            return 1;
        }
    }
    std::filesystem::remove(safetensors_path);
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
    const std::array<int, 4> attention_shape{1, 1, 2, 1};
    const std::array<float, 2> zero_attention_values{0.0F, 0.0F};
    const std::array<float, 2> value_attention_values{2.0F, 6.0F};
    const auto attention_queries = qwen38::MlxArray::from_float32(
        zero_attention_values, attention_shape);
    const auto attention_keys = qwen38::MlxArray::from_float32(
        zero_attention_values, attention_shape);
    const auto attention_values = qwen38::MlxArray::from_float32(
        value_attention_values, attention_shape);
    const auto causal_attention = qwen38::MlxArray::scaled_dot_product_attention(
        attention_queries, attention_keys, attention_values, 1.0F, true).to_float32();
    if (causal_attention.size() != 2 ||
        std::abs(causal_attention[0] - 2.0F) > 1.0e-5F ||
        std::abs(causal_attention[1] - 4.0F) > 1.0e-5F) {
        std::cerr << "MLX causal SDPA mismatch\n";
        return 1;
    }
    const std::array<int, 4> suffix_query_shape{1, 1, 1, 1};
    const std::array<int, 4> suffix_key_shape{1, 1, 3, 1};
    const std::array<float, 1> suffix_query_values{0.0F};
    const std::array<float, 3> suffix_key_values{0.0F, 0.0F, 0.0F};
    const std::array<float, 3> suffix_value_values{2.0F, 4.0F, 6.0F};
    const auto suffix_query = qwen38::MlxArray::from_float32(
        suffix_query_values, suffix_query_shape);
    const auto suffix_keys = qwen38::MlxArray::from_float32(
        suffix_key_values, suffix_key_shape);
    const auto suffix_values = qwen38::MlxArray::from_float32(
        suffix_value_values, suffix_key_shape);
    const auto suffix_attention = qwen38::MlxArray::scaled_dot_product_attention(
        suffix_query, suffix_keys, suffix_values, 1.0F, true).to_float32();
    if (suffix_attention.size() != 1 ||
        std::abs(suffix_attention[0] - 4.0F) > 1.0e-5F) {
        std::cerr << "MLX causal SDPA prefix offset mismatch\n";
        return 1;
    }
    const std::array<std::int32_t, 3> mask_values{1, 0, 1};
    const std::array<int, 4> mask_shape{1, 1, 1, 3};
    const auto attention_mask = qwen38::MlxArray::from_int32(
        mask_values, mask_shape).astype(MLX_BOOL);
    const auto masked_attention = qwen38::MlxArray::scaled_dot_product_attention(
        suffix_query, suffix_keys, suffix_values, 1.0F, attention_mask).to_float32();
    if (masked_attention.size() != 1 ||
        std::abs(masked_attention[0] - 4.0F) > 1.0e-5F) {
        std::cerr << "MLX masked SDPA mismatch\n";
        return 1;
    }
    const auto range = qwen38::MlxArray::arange(0.0, 3.0, 1.0, MLX_INT32);
    const std::array<int, 2> range_row_shape{1, 3};
    const std::array<int, 2> range_grid_shape{2, 3};
    const auto range_grid = range.reshape(range_row_shape).broadcast_to(range_grid_shape);
    const std::array<std::int32_t, 1> two_value{2};
    const std::array<int, 0> scalar_shape{};
    const auto two = qwen38::MlxArray::from_int32(two_value, scalar_shape);
    const auto low = qwen38::MlxArray::less_equal(range_grid, two);
    const auto high = qwen38::MlxArray::greater_equal(range_grid, two);
    const auto edge = qwen38::MlxArray::logical_and(low, high);
    const auto union_mask = qwen38::MlxArray::logical_or(edge, high);
    const auto chosen = qwen38::MlxArray::where(
        union_mask,
        range_grid,
        qwen38::MlxArray::zeros(scalar_shape, MLX_INT32));
    if (chosen.astype(MLX_FLOAT32).to_float32() !=
        std::vector<float>({0, 0, 2, 0, 0, 2})) {
        std::cerr << "MLX QSA selection primitives mismatch\n";
        return 1;
    }
    const std::array<std::int32_t, 2> scatter_indices_values{0, 2};
    const std::array<int, 2> scatter_indices_shape{1, 2};
    const auto scatter_indices = qwen38::MlxArray::from_int32(
        scatter_indices_values, scatter_indices_shape);
    const std::array<std::int32_t, 1> true_value{1};
    const auto true_scalar = qwen38::MlxArray::from_int32(
        true_value, scalar_shape).astype(MLX_BOOL);
    const auto scattered = qwen38::MlxArray::put_along_axis(
        qwen38::MlxArray::zeros(range_row_shape, MLX_BOOL),
        scatter_indices,
        true_scalar,
        -1).astype(MLX_FLOAT32).to_float32();
    if (scattered != std::vector<float>({1, 0, 1})) {
        std::cerr << "MLX put_along_axis mismatch\n";
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
    {
        constexpr std::string_view source = R"metal(
            const uint i = thread_position_in_grid.x;
            plus[i] = static_cast<T>((float)x[i] + N);
            times[i] = static_cast<T>((float)x[i] * N);
        )metal";
        const char* inputs[]{"x"};
        const char* outputs[]{"plus", "times"};
        const qwen38::MlxMetalKernel kernel(
            "qwen38_multi_output_test", inputs, outputs, source);
        const std::array<const qwen38::MlxArray*, 1> kernel_inputs{&left};
        const std::array<qwen38::MlxMetalOutputSpec, 2> output_specs{{
            {.shape = {2, 2}, .dtype = MLX_FLOAT32},
            {.shape = {2, 2}, .dtype = MLX_FLOAT32},
        }};
        const std::array<int, 3> grid{4, 1, 1};
        const std::array<int, 3> threadgroup{4, 1, 1};
        const std::array<qwen38::MlxMetalDtypeTemplate, 1> dtype_templates{{
            {.name = "T", .value = MLX_FLOAT32},
        }};
        const std::array<qwen38::MlxMetalIntTemplate, 1> int_templates{{
            {.name = "N", .value = 3},
        }};
        const auto kernel_outputs = kernel.apply(
            kernel_inputs,
            output_specs,
            grid,
            threadgroup,
            dtype_templates,
            int_templates);
        if (kernel_outputs[0].to_float32() != std::vector<float>({4, 5, 6, 7}) ||
            kernel_outputs[1].to_float32() != std::vector<float>({3, 6, 9, 12})) {
            std::cerr << "MLX multi-output Metal kernel mismatch\n";
            return 1;
        }
        auto cached_outputs = kernel.apply(
            kernel_inputs,
            output_specs,
            grid,
            threadgroup,
            dtype_templates,
            int_templates);
        if (cached_outputs[0].to_float32() != std::vector<float>({4, 5, 6, 7}) ||
            cached_outputs[1].to_float32() != std::vector<float>({3, 6, 9, 12})) {
            std::cerr << "cached MLX Metal configuration mismatch\n";
            return 1;
        }
    }
    {
        constexpr int heads = 2;
        constexpr int width = 128;
        std::vector<float> y_values(heads * width);
        std::vector<float> z_values(heads * width);
        std::vector<float> norm_values(width);
        for (std::size_t index = 0; index < y_values.size(); ++index) {
            y_values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 8.0F;
            z_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 7.0F;
        }
        for (std::size_t index = 0; index < norm_values.size(); ++index) {
            norm_values[index] = 0.75F + static_cast<float>(index % 9) / 32.0F;
        }
        const std::array<int, 4> y_shape{1, 1, heads, width};
        const std::array<int, 1> norm_shape{width};
        auto y = qwen38::MlxArray::from_float32(y_values, y_shape).astype(MLX_BFLOAT16);
        auto z = qwen38::MlxArray::from_float32(z_values, y_shape).astype(MLX_BFLOAT16);
        auto norm = qwen38::MlxArray::from_float32(norm_values, norm_shape).astype(MLX_BFLOAT16);
        constexpr float epsilon_value = 1.0e-6F;
        const std::array<float, 1> epsilon_values{epsilon_value};
        const std::array<int, 1> epsilon_shape{1};
        auto epsilon = qwen38::MlxArray::from_float32(epsilon_values, epsilon_shape);
        const char* inputs[]{"y", "z", "norm_weight", "epsilon"};
        const qwen38::MlxMetalKernel kernel(
            "qwen38_gdn_norm_gate_test", inputs, "output", qwen38::gdn_metal::norm_gate);
        const std::array<const qwen38::MlxArray*, 4> kernel_inputs{&y, &z, &norm, &epsilon};
        const std::array<qwen38::MlxMetalOutputSpec, 1> output_specs{{
            {.shape = {1, 1, heads * width}, .dtype = MLX_BFLOAT16},
        }};
        const std::array<int, 3> grid{32, 1, heads};
        const std::array<int, 3> threadgroup{32, 1, 1};
        const std::array<qwen38::MlxMetalDtypeTemplate, 1> dtype_templates{{
            {.name = "T", .value = MLX_BFLOAT16},
        }};
        const std::array<qwen38::MlxMetalIntTemplate, 3> int_templates{{
            {.name = "HV", .value = heads},
            {.name = "DV", .value = width},
            {.name = "SWISH", .value = 1},
        }};
        auto fused = kernel.apply(
            kernel_inputs,
            output_specs,
            grid,
            threadgroup,
            dtype_templates,
            int_templates);
        const std::array<int, 3> flat_shape{1, 1, heads * width};
        const auto reference = qwen38::MlxArray::multiply(
            y.rms_norm(norm, epsilon_value), z.silu()).reshape(
                flat_shape).astype(MLX_FLOAT32).to_float32();
        const auto candidate = fused[0].astype(MLX_FLOAT32).to_float32();
        float maximum_error = 0.0F;
        for (std::size_t index = 0; index < reference.size(); ++index) {
            maximum_error = std::max(maximum_error, std::abs(reference[index] - candidate[index]));
        }
        if (maximum_error > 0.015625F) {
            std::cerr << "GDN norm/gate Metal kernel mismatch: " << maximum_error << '\n';
            return 1;
        }
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
    qwen38::PersistedPrefixState persisted(1);
    persisted.target = qwen38::snapshot_decode_state(snapshot);
    persisted.target.layers[0].full_attention.token_count = 2;
    persisted.target.layers[0].full_attention.position_base = 5;
    persisted.target.layers[0].full_attention.keys = left.share();
    persisted.target.layers[0].full_attention.values = right.share();
    persisted.target.layers[0].full_attention.qsa_raw_keys = left.share();
    persisted.target.layers[0].full_attention.qsa_pooled_keys = right.share();
    persisted.target.layers[0].full_attention.qsa_pooled_count = 1;
    persisted.previous_target_stream = right.share();
    persisted.pending_mtp_streams.push_back(left.share());
    persisted.pending_mtp_tokens.push_back(99);
    persisted.mtp.row_count = 3;
    persisted.mtp.position_base = 7;
    persisted.mtp_profitable = true;
    persisted.mtp_profitability_current_token = 123;
    persisted.mtp_cumulative_profitability_keep = true;
    const std::filesystem::path state_path =
        std::filesystem::temp_directory_path() /
        ("qwen38-state-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
    qwen38::save_prefix_state(state_path, persisted);
    qwen38::PersistedPrefixState restored = qwen38::load_prefix_state(state_path, 1);
    std::filesystem::remove(state_path);
    if (restored.target.token_count != 7 || restored.target.layers.size() != 1 ||
        restored.target.layers[0].full_attention.position_base != 5 ||
        restored.target.layers[0].full_attention.qsa_pooled_count != 1 ||
        restored.target.layers[0].linear_attention.recurrent.to_float32() !=
            std::vector<float>({5, 6, 7, 8}) ||
        restored.target.layers[0].full_attention.keys.to_float32() !=
            std::vector<float>({1, 2, 3, 4}) ||
        !restored.previous_target_stream.has_value() ||
        restored.previous_target_stream->to_float32() !=
            std::vector<float>({5, 6, 7, 8}) ||
        restored.pending_mtp_tokens != std::vector<std::uint32_t>({99}) ||
        restored.pending_mtp_streams[0].to_float32() !=
            std::vector<float>({1, 2, 3, 4}) ||
        restored.mtp.row_count != 3 || restored.mtp.position_base != 7 ||
        restored.mtp_profitable != true ||
        restored.mtp_profitability_current_token != 123 ||
        !restored.mtp_cumulative_profitability_keep) {
        std::cerr << "persisted prefix-state roundtrip mismatch\n";
        return 1;
    }
    const std::filesystem::path cache_path =
        std::filesystem::temp_directory_path() /
        ("qwen38-prefix-cache-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        qwen38::PrefixCacheStore store(cache_path, 16ULL * 1024ULL * 1024ULL, 1);
        const std::array<std::uint32_t, 2> short_tokens{1, 2};
        persisted.target.token_count = short_tokens.size();
        store.save(short_tokens, persisted);
        const std::array<std::uint32_t, 3> long_tokens{1, 2, 3};
        persisted.target.token_count = long_tokens.size();
        store.save(long_tokens, persisted);
        const std::array<std::uint32_t, 4> prompt_tokens{1, 2, 3, 4};
        std::optional<qwen38::StoredPrefixState> cache_hit =
            store.load_longest(prompt_tokens);
        if (!cache_hit.has_value() ||
            cache_hit->tokens != std::vector<std::uint32_t>({1, 2, 3}) ||
            cache_hit->state.target.token_count != 3 ||
            cache_hit->state.target.layers[0].linear_attention.recurrent.to_float32() !=
                std::vector<float>({5, 6, 7, 8})) {
            std::cerr << "SSD prefix-cache longest-prefix mismatch\n";
            std::filesystem::remove_all(cache_path);
            return 1;
        }
        store.clear();
        if (store.load_longest(prompt_tokens).has_value()) {
            std::cerr << "SSD prefix-cache clear mismatch\n";
            std::filesystem::remove_all(cache_path);
            return 1;
        }
    }
    std::filesystem::remove_all(cache_path);

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

    qwen38::MtpDecodeState mtp_state;
    mtp_state.row_count = 19;
    mtp_state.position_base = 41;
    mtp_state.layer.linear_attention.initialized = true;
    mtp_state.layer.linear_attention.convolution =
        qwen38::MlxArray::from_float32(left_values, shape);
    mtp_state.layer.linear_attention.recurrent =
        qwen38::MlxArray::from_float32(right_values, shape);
    mtp_state.layer.full_attention.token_count = 2;
    mtp_state.layer.full_attention.keys =
        qwen38::MlxArray::from_float32(left_values, shape);
    mtp_state.layer.full_attention.values =
        qwen38::MlxArray::from_float32(right_values, shape);
    mtp_state.layer.full_attention.qsa_raw_keys =
        qwen38::MlxArray::from_float32(left_values, shape);
    mtp_state.layer.full_attention.qsa_pooled_keys =
        qwen38::MlxArray::from_float32(right_values, shape);
    mtp_state.layer.full_attention.qsa_pooled_count = 1;
    auto mtp_snapshot = qwen38::snapshot_mtp_decode_state(mtp_state);
    mtp_state = {};
    if (mtp_snapshot.row_count != 19 || mtp_snapshot.position_base != 41 ||
        mtp_snapshot.layer.linear_attention.convolution.to_float32() !=
            std::vector<float>({1, 2, 3, 4}) ||
        mtp_snapshot.layer.linear_attention.recurrent.to_float32() !=
            std::vector<float>({5, 6, 7, 8}) ||
        mtp_snapshot.layer.full_attention.qsa_raw_keys.to_float32() !=
            std::vector<float>({1, 2, 3, 4}) ||
        mtp_snapshot.layer.full_attention.qsa_pooled_keys.to_float32() !=
            std::vector<float>({5, 6, 7, 8}) ||
        mtp_snapshot.layer.full_attention.qsa_pooled_count != 1) {
        std::cerr << "MTP decode-state snapshot mismatch\n";
        return 1;
    }
    return 0;
}
