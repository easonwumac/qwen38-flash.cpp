#include "qwen38/decoder_layer.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/mlx_backend.hpp"
#include "qwen38/token_embedding.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using qwen38::DecoderLayer;
using qwen38::DecoderLayerState;
using qwen38::MlxArray;
using Clock = std::chrono::steady_clock;
constexpr int warmups = 3;
constexpr int iterations = 10;

struct Embedding final {
    MlxArray weight;
    MlxArray scales;
    MlxArray biases;
    int group_size;
    int bits;
    int hidden;
    int streams;
    std::size_t vocabulary;
};

std::vector<std::uint32_t> make_tokens(
    const int rows, const std::size_t vocabulary, const bool diverse) {
    if (vocabulary <= 256 || vocabulary >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("unsupported vocabulary size");
    }
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        std::uint64_t token = diverse
            ? (9419U + static_cast<std::uint64_t>(row) * 7919U) % vocabulary
            : 9419U % vocabulary;
        if (token < 100U) token += 100U;
        tokens[static_cast<std::size_t>(row)] = static_cast<std::uint32_t>(token);
    }
    return tokens;
}

MlxArray embed_ids(const Embedding& embedding, const std::vector<std::int32_t>& ids_data) {
    MlxArray ids = MlxArray::from_int32(
        ids_data, std::array<int, 1>{static_cast<int>(ids_data.size())});
    return MlxArray::dequantize(
        MlxArray::take_axis(embedding.weight, ids, 0),
        MlxArray::take_axis(embedding.scales, ids, 0),
        MlxArray::take_axis(embedding.biases, ids, 0),
        embedding.group_size,
        embedding.bits);
}

MlxArray build_token_loop(
    const Embedding& embedding, const std::span<const std::uint32_t> tokens) {
    std::vector<MlxArray> rows;
    rows.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
        const std::vector<std::int32_t> id{static_cast<std::int32_t>(token)};
        MlxArray value = embed_ids(embedding, id).reshape(
            std::array<int, 3>{1, 1, embedding.hidden});
        rows.push_back(qwen38::HyperConnection::initialize_stream(value, embedding.streams));
    }
    MlxArray result = rows.front().share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, rows[row], 1);
    }
    return result;
}

MlxArray build_batch(
    const Embedding& embedding, const std::span<const std::uint32_t> tokens) {
    return qwen38::HyperConnection::initialize_stream(
        qwen38::embed_token_batch(
            embedding.weight,
            embedding.scales,
            embedding.biases,
            tokens,
            embedding.vocabulary,
            static_cast<std::size_t>(embedding.hidden),
            embedding.group_size,
            embedding.bits),
        embedding.streams);
}

struct Execution final {
    MlxArray output;
    MlxArray convolution;
    MlxArray recurrent;
};

Execution execute(
    DecoderLayer& layer,
    const Embedding& embedding,
    const std::vector<std::uint32_t>& tokens,
    const bool batched) {
    MlxArray input = batched
        ? build_batch(embedding, tokens)
        : build_token_loop(embedding, tokens);
    DecoderLayerState state;
    MlxArray output = layer.forward_prefill(std::move(input), tokens, state);
    const std::array<const MlxArray*, 3> values{
        &output, &state.linear_attention.convolution, &state.linear_attention.recurrent};
    MlxArray::eval_all(values);
    return {std::move(output), std::move(state.linear_attention.convolution),
            std::move(state.linear_attention.recurrent)};
}

double max_abs(const MlxArray& candidate, const MlxArray& reference) {
    if (candidate.shape() != reference.shape()) return std::numeric_limits<double>::infinity();
    const auto a = candidate.astype(MLX_FLOAT32).to_float32();
    const auto b = reference.astype(MLX_FLOAT32).to_float32();
    double result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        result = std::max(result, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    return result;
}

double timed(
    DecoderLayer& layer,
    const Embedding& embedding,
    const std::vector<std::uint32_t>& tokens,
    const bool batched) {
    const auto start = Clock::now();
    static_cast<void>(execute(layer, embedding, tokens, batched));
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Stats final { double mean, median, minimum, maximum; };
Stats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    const double median = values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
    return {
        std::accumulate(values.begin(), values.end(), 0.0) / values.size(),
        median, values.front(), values.back()};
}

void print_samples(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

struct LogicalBytes final {
    std::size_t leaf_streams;
    std::size_t concat_outputs;
    std::size_t batch_packed_gather;
    std::size_t batch_qmeta_gathers;
    std::size_t batch_embedding;
    std::size_t batch_stream;
};

LogicalBytes logical_bytes(const int rows, const int hidden, const int streams) {
    const std::size_t count = static_cast<std::size_t>(rows);
    const std::size_t stream_row = static_cast<std::size_t>(hidden) * streams * sizeof(std::uint16_t);
    return {
        .leaf_streams = count * stream_row,
        .concat_outputs = stream_row * (count * (count + 1) / 2 - 1),
        .batch_packed_gather = count * static_cast<std::size_t>(hidden / 8) * sizeof(std::uint32_t),
        .batch_qmeta_gathers = 2 * count * static_cast<std::size_t>(hidden / 64) * sizeof(std::uint16_t),
        .batch_embedding = count * static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
        .batch_stream = count * stream_row,
    };
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("usage: qwen38-prefill-embedding-ab MODEL_DIRECTORY");
    // This must precede MlxTensorStore construction and every array allocation.
    static_cast<void>(MlxArray::set_cache_limit(256ULL * 1024ULL * 1024ULL));
    setenv("QWEN38_GDN_METAL_PREFILL", "1", 1);
    setenv("QWEN38_GROUPED_PREFILL", "1", 1);
    unsetenv("QWEN38_COMPACT_QMETA");
    unsetenv("QWEN38_QMETA_PREFILL_CACHE");
    unsetenv("QWEN38_QMETA_PREFILL_DEFER_REDUCE");
    unsetenv("QWEN38_QMETA_PREFILL_DEFER_TEMPORARY");
    unsetenv("QWEN38_TARGET_TOPK");
    unsetenv("QWEN38_PROFILE_MOE_PREFILL");

    qwen38::ModelManifest manifest = qwen38::ModelManifest::load(argv[1]);
    qwen38::MlxTensorStore tensors(std::move(manifest));
    const auto& config = tensors.manifest().config();
    Embedding embedding{
        .weight = tensors.tensor("language_model.model.embed_tokens.weight"),
        .scales = tensors.tensor("language_model.model.embed_tokens.scales"),
        .biases = tensors.tensor("language_model.model.embed_tokens.biases"),
        .group_size = static_cast<int>(config.quantization_group_size),
        .bits = static_cast<int>(config.quantization_bits),
        .hidden = static_cast<int>(config.hidden_size),
        .streams = static_cast<int>(config.hyper_connection_count),
        .vocabulary = config.vocabulary_size,
    };
    DecoderLayer layer(tensors, 0, config);
    const std::array<int, 2> widths{128, 512};
    bool first_case = true;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"mode\":\"layer0-prefill-embedding-construction-ab\""
              << ",\"profile\":\"exact-q4-base-top10-full-metadata\""
              << ",\"allocator_cache_mib\":256,\"warmups\":3,\"iterations\":10"
              << ",\"timing_scope\":\"input-graph-construction-through-complete-layer0-output-and-gdn-state-eval\""
              << ",\"cases\":[";
    for (const int rows : widths) for (const bool diverse : {false, true}) {
        const auto tokens = make_tokens(rows, config.vocabulary_size, diverse);
        Execution reference = execute(layer, embedding, tokens, false);
        Execution candidate = execute(layer, embedding, tokens, true);
        const double output_error = max_abs(candidate.output, reference.output);
        const double convolution_error = max_abs(candidate.convolution, reference.convolution);
        const double recurrent_error = max_abs(candidate.recurrent, reference.recurrent);
        if (output_error != 0 || convolution_error != 0 || recurrent_error != 0) {
            throw std::runtime_error("batched embedding changed layer0 output or state");
        }
        for (int sample = 0; sample < warmups; ++sample) {
            if (sample % 2 == 0) {
                static_cast<void>(execute(layer, embedding, tokens, true));
                static_cast<void>(execute(layer, embedding, tokens, false));
            } else {
                static_cast<void>(execute(layer, embedding, tokens, false));
                static_cast<void>(execute(layer, embedding, tokens, true));
            }
        }
        std::vector<double> token_samples, batch_samples;
        for (int sample = 0; sample < iterations; ++sample) {
            if (sample % 2 == 0) {
                batch_samples.push_back(timed(layer, embedding, tokens, true));
                token_samples.push_back(timed(layer, embedding, tokens, false));
            } else {
                token_samples.push_back(timed(layer, embedding, tokens, false));
                batch_samples.push_back(timed(layer, embedding, tokens, true));
            }
        }
        const Stats ts = summarize(token_samples), bs = summarize(batch_samples);
        const LogicalBytes bytes = logical_bytes(rows, embedding.hidden, embedding.streams);
        if (!first_case) std::cout << ',';
        first_case = false;
        std::cout << "{\"rows\":" << rows << ",\"token_cohort\":\""
                  << (diverse ? "diverse-deterministic-ids" : "repeated-id-9419") << "\""
                  << ",\"exact\":{\"output_max_abs\":" << output_error
                  << ",\"convolution_max_abs\":" << convolution_error
                  << ",\"recurrent_max_abs\":" << recurrent_error << "}"
                  << ",\"logical_bytes_not_simultaneous_peak\":{\"token_leaf_streams\":"
                  << bytes.leaf_streams << ",\"token_concat_outputs\":" << bytes.concat_outputs
                  << ",\"batch_packed_gather\":" << bytes.batch_packed_gather
                  << ",\"batch_qmeta_gathers\":" << bytes.batch_qmeta_gathers
                  << ",\"batch_embedding\":" << bytes.batch_embedding
                  << ",\"batch_stream\":" << bytes.batch_stream << "}"
                  << ",\"timing_ms\":{\"token_loop_mean\":" << ts.mean
                  << ",\"token_loop_median\":" << ts.median
                  << ",\"token_loop_min\":" << ts.minimum << ",\"token_loop_max\":" << ts.maximum
                  << ",\"batch_mean\":" << bs.mean << ",\"batch_median\":" << bs.median
                  << ",\"batch_min\":" << bs.minimum << ",\"batch_max\":" << bs.maximum
                  << ",\"mean_speedup\":" << ts.mean / bs.mean
                  << ",\"median_speedup\":" << ts.median / bs.median << "}"
                  << ",\"token_loop_samples_ms\":"; print_samples(token_samples);
        std::cout << ",\"batch_samples_ms\":"; print_samples(batch_samples);
        std::cout << '}';
    }
    std::cout << "],\"open_shards\":" << tensors.open_shard_count() << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "qwen38-prefill-embedding-ab: " << error.what() << '\n';
    return 1;
}
