#include "qwen38/hyper_connection.hpp"
#include "qwen38/self_attention.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

qwen38::MlxArray make_input(qwen38::MlxTensorStore& tensors, const std::int32_t token) {
    const std::vector<std::int32_t> token_values{token};
    const std::vector<int> token_shape{1};
    const auto ids = qwen38::MlxArray::from_int32(token_values, token_shape);
    const auto weight = tensors.tensor("language_model.model.embed_tokens.weight");
    const auto scales = tensors.tensor("language_model.model.embed_tokens.scales");
    const auto biases = tensors.tensor("language_model.model.embed_tokens.biases");
    const auto& config = tensors.manifest().config();
    const auto embedding_quantization = tensors.manifest().quantization_for(
        "language_model.model.embed_tokens");
    const auto mixer_quantization = tensors.manifest().quantization_for(
        "language_model.model.layers.3.attn_hyper_connection.input_mix_weight_down");
    auto embedding = qwen38::MlxArray::dequantize(
        qwen38::MlxArray::take_axis(weight, ids, 0),
        qwen38::MlxArray::take_axis(scales, ids, 0),
        qwen38::MlxArray::take_axis(biases, ids, 0),
        static_cast<int>(embedding_quantization.group_size),
        static_cast<int>(embedding_quantization.bits));
    const std::vector<int> embedding_shape{1, 1, static_cast<int>(config.hidden_size)};
    auto stream = qwen38::HyperConnection::initialize_stream(
        embedding.reshape(embedding_shape), config.hyper_connection_count);
    qwen38::HyperConnection mixer(
        tensors,
        "language_model.model.layers.3.attn_hyper_connection",
        config.hidden_size,
        config.hyper_connection_count,
        mixer_quantization.bits,
        mixer_quantization.group_size,
        static_cast<float>(config.rms_norm_epsilon),
        true);
    return mixer.read(stream).mixed;
}

qwen38::SelfAttentionState snapshot(const qwen38::SelfAttentionState& state) {
    qwen38::SelfAttentionState result;
    result.token_count = state.token_count;
    result.position_base = state.position_base;
    result.qsa_pooled_count = state.qsa_pooled_count;
    result.qsa_raw_start = state.qsa_raw_start;
    result.kv_q8 = state.kv_q8;
    result.kv_q8_cold_tokens = state.kv_q8_cold_tokens;
    if (state.kv_q8) {
        result.key_weights = state.key_weights.share();
        result.key_scales = state.key_scales.share();
        result.key_biases = state.key_biases.share();
        result.value_weights = state.value_weights.share();
        result.value_scales = state.value_scales.share();
        result.value_biases = state.value_biases.share();
    }
    if (state.token_count != 0) {
        if (state.keys.get().ctx != nullptr) result.keys = state.keys.share();
        if (state.values.get().ctx != nullptr) result.values = state.values.share();
        result.qsa_raw_keys = state.qsa_raw_keys.share();
    }
    if (state.qsa_pooled_count != 0) {
        result.qsa_pooled_keys = state.qsa_pooled_keys.share();
    }
    return result;
}

double cosine(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size() || left.empty()) {
        throw std::runtime_error("QSA parity output shape mismatch");
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

void check_equal(const qwen38::MlxArray& left, const qwen38::MlxArray& right,
                 const char* what) {
    const auto a = left.astype(MLX_FLOAT32).to_float32();
    const auto b = right.astype(MLX_FLOAT32).to_float32();
    if (left.shape() != right.shape() || a.size() != b.size() || !std::equal(
            a.begin(), a.end(), b.begin(), [](const float x, const float y) {
                return std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y);
            })) {
        throw std::runtime_error(std::string("QSA rollback parity failed: ") + what);
    }
}

void check_qsa_state(const qwen38::SelfAttentionState& bounded,
                     const qwen38::SelfAttentionState& reference,
                     const std::size_t ratio) {
    if (bounded.token_count != reference.token_count ||
        bounded.qsa_pooled_count != reference.qsa_pooled_count ||
        bounded.qsa_raw_start < reference.qsa_raw_start ||
        bounded.qsa_raw_start > bounded.qsa_pooled_count * ratio) {
        throw std::runtime_error("QSA rollback metadata parity failed");
    }
    if (bounded.qsa_pooled_count != 0) {
        check_equal(bounded.qsa_pooled_keys, reference.qsa_pooled_keys, "pooled keys");
    }
    const auto shape = reference.qsa_raw_keys.shape();
    const auto suffix = reference.qsa_raw_keys.slice(
        std::vector<int>{0, static_cast<int>(bounded.qsa_raw_start - reference.qsa_raw_start), 0},
        shape, std::vector<int>{1, 1, 1});
    check_equal(bounded.qsa_raw_keys, suffix, "raw suffix");
}

int qsa_rollback_smoke(qwen38::MlxTensorStore& tensors, qwen38::SelfAttention& layer,
                       const qwen38::MlxArray& input, const qwen38::MlxArray& next) {
    if (std::getenv("QWEN38_MEMORY_GUARD") == nullptr) {
        throw std::runtime_error("QSA rollback smoke requires memory_guard.py");
    }
    static_cast<void>(qwen38::MlxArray::set_cache_limit(64ULL * 1024ULL * 1024ULL));
    static_cast<void>(::setenv("QWEN38_SDPA_PREFILL", "1", 1));
    static_cast<void>(::setenv("QWEN38_SDPA_DECODE", "1", 1));
    static_cast<void>(::setenv("QWEN38_BATCH_SDPA_VERIFY", "1", 1));
    static_cast<void>(::setenv("QWEN38_BATCH_Q8_VERIFY", "1", 1));
    const auto ratio = tensors.manifest().config().indexer_compress_ratio;
    const auto configured_budget = tensors.manifest().config().indexer_budget;
    const std::array<int, 3> repetitions{1, 5, 1};
    auto batch = input.tile(repetitions);
    std::size_t cases = 0;
    std::size_t continuations = 0;
    for (const bool q8 : {false, true}) {
        static_cast<void>(::setenv("QWEN38_KV_CACHE", q8 ? "q8" : "bf16", 1));
        static_cast<void>(::setenv("QWEN38_KV_Q8_MIN_TOKENS", "2049", 1));
        static_cast<void>(::setenv("QWEN38_QSA_PACKED_MIN_TOKENS", "0", 1));
        for (const auto budget : {std::size_t{512}, configured_budget}) {
            static_cast<void>(::setenv("QWEN38_QSA_DECODE_BUDGET",
                                      std::to_string(budget).c_str(), 1));
            const std::array<std::size_t, 4> contexts = q8
                ? std::array<std::size_t, 4>{2051, 2052, 2060, 2112}
                : std::array<std::size_t, 4>{budget - 8, budget - 1, budget + 1, budget + 32};
            for (const auto tokens : contexts) {
                static_cast<void>(::setenv("QWEN38_QSA_RAW_WINDOW", "0", 1));
                qwen38::SelfAttentionState origin;
                for (std::size_t offset = 0; offset < tokens;) {
                    const auto rows = std::min(std::size_t{256}, tokens - offset);
                    auto prompt = input.tile(std::array<int, 3>{1, static_cast<int>(rows), 1});
                    layer.forward_prefill(prompt, origin).eval();
                    offset += rows;
                }
                // For post-frontier contexts also exercise an already pooled origin.
                layer.forward_decode(next, origin).eval();
                if (q8 && !origin.kv_q8) throw std::runtime_error("Q8 smoke not engaged");
                auto reference_origin = snapshot(origin);
                auto bounded_origin = snapshot(origin);
                std::vector<qwen38::SelfAttentionState> reference, bounded;
                auto expected = layer.forward_verify(batch, reference_origin, reference);
                expected.eval();
                static_cast<void>(::setenv("QWEN38_QSA_RAW_WINDOW", "64", 1));
                auto actual = layer.forward_verify(batch, bounded_origin, bounded);
                check_equal(actual, expected, "verifier output");
                if (reference.size() != 5 || bounded.size() != 5) {
                    throw std::runtime_error("QSA rollback checkpoint count mismatch");
                }
                // Zero accepted drafts still commits the current-token row. Include
                // the untouched origin as well as every partial/full accepted prefix.
                for (std::size_t accepted = 0; accepted <= 5; ++accepted) {
                    auto control = snapshot(accepted == 0 ? reference_origin : reference[accepted - 1]);
                    auto candidate = snapshot(accepted == 0 ? bounded_origin : bounded[accepted - 1]);
                    check_qsa_state(candidate, control, ratio);
                    for (int step = 0; step < 12; ++step) {
                        const auto& token_input = step % 2 == 0 ? next : input;
                        static_cast<void>(::setenv("QWEN38_QSA_RAW_WINDOW", "0", 1));
                        auto control_output = layer.forward_decode(token_input, control);
                        control_output.eval();
                        static_cast<void>(::setenv("QWEN38_QSA_RAW_WINDOW", "64", 1));
                        auto candidate_output = layer.forward_decode(token_input, candidate);
                        check_equal(candidate_output, control_output, "continuation output");
                        check_qsa_state(candidate, control, ratio);
                        ++continuations;
                    }
                }
                ++cases;
                qwen38::MlxArray::clear_cache();
            }
        }
    }
    std::cout << "{\"qsa_rollback_cases\":" << cases
              << ",\"continuation_steps\":" << continuations
              << ",\"exact_output_and_qsa_state\":true}\n";
    return EXIT_SUCCESS;
}

int qsa_smoke(
    qwen38::MlxTensorStore& tensors,
    qwen38::SelfAttention& layer,
    const qwen38::MlxArray& input) {
    static_cast<void>(qwen38::MlxArray::set_cache_limit(256ULL * 1024ULL * 1024ULL));
    static_cast<void>(::setenv("QWEN38_QSA_PACKED_MIN_TOKENS", "0", 1));
    const auto& config = tensors.manifest().config();
    const std::size_t budget = config.indexer_budget;
    const std::size_t ratio = config.indexer_compress_ratio;
    if (budget % 512 != 0 || ratio == 0) {
        throw std::runtime_error("QSA smoke expects a 512-aligned selection budget");
    }
    static_cast<void>(::setenv("QWEN38_SDPA_PREFILL", "1", 1));
    qwen38::SelfAttentionState origin;
    for (std::size_t offset = 0; offset < budget; offset += 512) {
        const std::array<int, 3> repetitions{1, 512, 1};
        auto chunk = input.tile(repetitions);
        auto output = layer.forward_prefill(chunk, origin);
        output.eval();
    }
    if (origin.token_count != budget ||
        origin.qsa_raw_keys.shape() != std::vector<int>({
            1, static_cast<int>(budget), static_cast<int>(config.indexer_head_dimension)})) {
        throw std::runtime_error("QSA pre-engagement state is incomplete");
    }

    qwen38::SelfAttentionState dense_prefill_state = snapshot(origin);
    qwen38::SelfAttentionState packed_prefill_state = snapshot(origin);
    const std::array<int, 3> packed_repetitions{1, 512, 1};
    auto packed_input = input.tile(packed_repetitions);
    static_cast<void>(::unsetenv("QWEN38_QSA_PACKED_PREFILL"));
    auto dense_prefill = layer.forward_prefill(packed_input, dense_prefill_state);
    const auto dense_prefill_values = dense_prefill.astype(MLX_FLOAT32).to_float32();
    static_cast<void>(::setenv("QWEN38_QSA_PACKED_PREFILL", "1", 1));
    auto packed_prefill = layer.forward_prefill(packed_input, packed_prefill_state);
    const auto packed_prefill_values = packed_prefill.astype(MLX_FLOAT32).to_float32();
    static_cast<void>(::unsetenv("QWEN38_QSA_PACKED_PREFILL"));
    const double packed_prefill_cosine = cosine(
        dense_prefill_values, packed_prefill_values);
    if (packed_prefill_state.token_count != budget + 512 ||
        packed_prefill_cosine < 0.999) {
        throw std::runtime_error("packed QSA prefill diverged from dense attention");
    }

    qwen38::SelfAttentionState batched_origin = snapshot(origin);
    qwen38::SelfAttentionState serial = snapshot(origin);
    const std::array<int, 3> verify_repetitions{1, static_cast<int>(ratio), 1};
    auto verify_input = input.tile(verify_repetitions);
    std::vector<qwen38::SelfAttentionState> checkpoints;
    const auto verify_started = std::chrono::steady_clock::now();
    auto batched = layer.forward_verify(verify_input, batched_origin, checkpoints);
    const auto batch_values = batched.astype(MLX_FLOAT32).to_float32();
    const double verify_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - verify_started).count();

    std::vector<qwen38::MlxArray> serial_rows;
    serial_rows.reserve(ratio);
    for (std::size_t row = 0; row < ratio; ++row) {
        serial_rows.push_back(layer.forward_decode(input, serial));
    }
    qwen38::MlxArray serial_output = serial_rows.front().share();
    for (std::size_t row = 1; row < serial_rows.size(); ++row) {
        serial_output = qwen38::MlxArray::concatenate(serial_output, serial_rows[row], 1);
    }
    const auto serial_values = serial_output.astype(MLX_FLOAT32).to_float32();
    const double output_cosine = cosine(batch_values, serial_values);
    double output_max_abs = 0.0;
    std::size_t output_differences = 0;
    for (std::size_t index = 0; index < batch_values.size(); ++index) {
        output_max_abs = std::max(
            output_max_abs,
            std::abs(static_cast<double>(batch_values[index]) - serial_values[index]));
        output_differences += batch_values[index] != serial_values[index] ? 1 : 0;
    }
    const std::size_t engaged_tokens = budget + ratio;
    const std::size_t expected_blocks = engaged_tokens / ratio;
    if (checkpoints.size() != ratio || serial.token_count != engaged_tokens ||
        checkpoints.back().token_count != engaged_tokens ||
        checkpoints.back().qsa_pooled_count != expected_blocks ||
        checkpoints.back().qsa_raw_keys.shape() != std::vector<int>({
            1, static_cast<int>(engaged_tokens - checkpoints.back().qsa_raw_start),
            static_cast<int>(config.indexer_head_dimension)}) ||
        output_cosine < 0.999) {
        throw std::runtime_error("QSA batched/serial parity or cache contract failed");
    }
    for (std::size_t row = 0; row < ratio; ++row) {
        const std::size_t row_tokens = budget + row + 1;
        const std::size_t row_blocks = row_tokens / ratio;
        const std::size_t expected_row_pooled = std::min(
            row_blocks, checkpoints.back().qsa_pooled_count);
        if (checkpoints[row].token_count != budget + row + 1 ||
            checkpoints[row].qsa_raw_keys.shape()[1] !=
                static_cast<int>(row_tokens - checkpoints[row].qsa_raw_start) ||
            checkpoints[row].qsa_pooled_count != expected_row_pooled) {
            throw std::runtime_error("QSA verifier checkpoint is misaligned");
        }
    }

    std::size_t scale_context = budget;
    if (const char* configured = std::getenv("QWEN38_QSA_SMOKE_CONTEXT_TOKENS");
        configured != nullptr) {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(configured, &end, 10);
        if (end == configured || *end != '\0' || parsed < budget || parsed % 512 != 0) {
            throw std::runtime_error(
                "QWEN38_QSA_SMOKE_CONTEXT_TOKENS must be a 512-aligned value at least budget");
        }
        scale_context = static_cast<std::size_t>(parsed);
    }
    qwen38::SelfAttentionState scale_origin = snapshot(origin);
    static_cast<void>(::unsetenv("QWEN38_QSA_PACKED_PREFILL"));
    for (std::size_t offset = budget; offset < scale_context; offset += 512) {
        auto extension = layer.forward_prefill(packed_input, scale_origin);
        extension.eval();
        qwen38::MlxArray::clear_cache();
    }
    std::size_t scale_rows = 512;
    if (const char* configured = std::getenv("QWEN38_QSA_SMOKE_ROWS");
        configured != nullptr) {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(configured, &end, 10);
        if (end == configured || *end != '\0' || parsed == 0 || parsed > 512) {
            throw std::runtime_error("QWEN38_QSA_SMOKE_ROWS must be between 1 and 512");
        }
        scale_rows = static_cast<std::size_t>(parsed);
    }
    const std::array<int, 3> scale_repetitions{
        1, static_cast<int>(scale_rows), 1};
    auto scale_input = input.tile(scale_repetitions);
    static_cast<void>(::unsetenv("QWEN38_QSA_PACKED_PREFILL"));
    const auto measure_prefill = [&](const bool packed) {
        qwen38::SelfAttentionState measured_state = snapshot(scale_origin);
        if (packed) {
            static_cast<void>(::setenv("QWEN38_QSA_PACKED_PREFILL", "1", 1));
        } else {
            static_cast<void>(::unsetenv("QWEN38_QSA_PACKED_PREFILL"));
        }
        const auto started = std::chrono::steady_clock::now();
        auto output = layer.forward_prefill(scale_input, measured_state);
        auto values = output.astype(MLX_FLOAT32).to_float32();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return std::pair{elapsed_ms, std::move(values)};
    };
    const auto dense_scale = measure_prefill(false);
    const auto packed_scale = measure_prefill(true);
    const auto reverse_dense_scale = measure_prefill(false);
    static_cast<void>(::unsetenv("QWEN38_QSA_PACKED_PREFILL"));
    const double scale_cosine = cosine(
        reverse_dense_scale.second, packed_scale.second);
    if (scale_cosine < 0.999) {
        throw std::runtime_error("scaled packed QSA prefill diverged from dense attention");
    }
    std::cout << "{\"qsa_engaged_tokens\":" << engaged_tokens
              << ",\"raw_rows\":" << checkpoints.back().qsa_raw_keys.shape()[1]
              << ",\"pooled_blocks\":" << checkpoints.back().qsa_pooled_count
              << ",\"batch_serial_cosine\":" << output_cosine
              << ",\"batch_serial_max_abs\":" << output_max_abs
              << ",\"batch_serial_differences\":" << output_differences
              << ",\"verify_ms\":" << verify_ms
              << ",\"packed_prefill_cosine\":" << packed_prefill_cosine
              << ",\"scale_context\":" << scale_context
              << ",\"scale_rows\":" << scale_rows
              << ",\"dense_ms\":" << dense_scale.first
              << ",\"packed_ms\":" << packed_scale.first
              << ",\"reverse_dense_ms\":" << reverse_dense_scale.first
              << ",\"scale_cosine\":" << scale_cosine << "}\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY [--qsa|--qsa-rollback]\n";
        return EXIT_FAILURE;
    }
    try {
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]));
        auto first_input = make_input(tensors, 9419);
        auto second_input = make_input(tensors, 11);
        qwen38::SelfAttention layer(
            tensors,
            "language_model.model.layers.3.self_attn",
            tensors.manifest().config());
        if (argc == 3) {
            if (std::string_view(argv[2]) == "--qsa-rollback") {
                return qsa_rollback_smoke(tensors, layer, first_input, second_input);
            }
            if (std::string_view(argv[2]) != "--qsa") {
                throw std::runtime_error("unknown self-attention smoke mode");
            }
            return qsa_smoke(tensors, layer, first_input);
        }
        std::vector<double> timings;
        std::vector<float> first_values;
        std::vector<float> second_values;
        for (int iteration = 0; iteration < 6; ++iteration) {
            qwen38::SelfAttentionState state;
            auto first = layer.forward_decode(first_input, state);
            first_values = first.astype(MLX_FLOAT32).to_float32();
            const auto started = std::chrono::steady_clock::now();
            auto second = layer.forward_decode(second_input, state);
            second_values = second.astype(MLX_FLOAT32).to_float32();
            timings.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
        const auto& config = tensors.manifest().config();
        if (first_values.size() != config.hidden_size || second_values.size() != config.hidden_size ||
            !std::all_of(second_values.begin(), second_values.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("self-attention output is invalid");
        }
        const double first_checksum = std::accumulate(first_values.begin(), first_values.end(), 0.0);
        const double second_checksum = std::accumulate(second_values.begin(), second_values.end(), 0.0);
        std::vector<double> warm(timings.begin() + 1, timings.end());
        std::sort(warm.begin(), warm.end());
        std::cout << "{\"first_checksum\":" << first_checksum
                  << ",\"second_checksum\":" << second_checksum
                  << ",\"second_cold_ms\":" << timings.front()
                  << ",\"warm_median_ms\":" << warm[warm.size() / 2]
                  << ",\"open_shards\":" << tensors.open_shard_count() << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-self-attention-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
