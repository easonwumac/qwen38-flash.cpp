#include "qwen38/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid model dimension: ") + name);
    }
    return static_cast<int>(value);
}

std::size_t verify_barrier_stride() {
    const char* raw = std::getenv("QWEN38_VERIFY_BARRIER_STRIDE");
    if (raw == nullptr) return 16;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 1 || parsed > 48) {
        throw std::runtime_error("verify barrier stride must be between 1 and 48");
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t prefill_barrier_stride() {
    const char* raw = std::getenv("QWEN38_PREFILL_BARRIER_STRIDE");
    if (raw == nullptr) return 1;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 1 || parsed > 48) {
        throw std::runtime_error("prefill barrier stride must be between 1 and 48");
    }
    return static_cast<std::size_t>(parsed);
}

MlxArray concatenate_sequence_rows(const std::vector<MlxArray>& rows) {
    if (rows.empty()) throw std::runtime_error("cannot concatenate an empty model batch");
    MlxArray result = rows.front().share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, rows[row], 1);
    }
    return result;
}

MlxArray slice_sequence_row(const MlxArray& batch, const std::size_t row) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() != 3 || row >= static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("model batch row is out of range");
    }
    const std::vector<int> start{0, static_cast<int>(row), 0};
    const std::vector<int> stop{shape[0], static_cast<int>(row + 1), shape[2]};
    const std::vector<int> strides{1, 1, 1};
    return batch.slice(start, stop, strides);
}

} // namespace

QwenModel::QuantizedTensor QwenModel::load_quantized(
    MlxTensorStore& tensors,
    const char* prefix) {
    const std::string base(prefix);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
    };
}

QwenModel::QwenModel(MlxTensorStore& tensors)
    : hidden_size_(tensors.manifest().config().hidden_size),
      stream_count_(tensors.manifest().config().hyper_connection_count),
      vocabulary_size_(tensors.manifest().config().vocabulary_size),
      bits_(dimension(tensors.manifest().config().quantization_bits, "quantization bits")),
      group_size_(dimension(
          tensors.manifest().config().quantization_group_size, "quantization group size")),
      embedding_(load_quantized(tensors, "language_model.model.embed_tokens")),
      language_head_(load_quantized(tensors, "language_model.lm_head")),
      final_mixer_(
          tensors,
          "language_model.model.hyper_connection_mixer",
          tensors.manifest().config().hidden_size,
          tensors.manifest().config().hyper_connection_count,
          tensors.manifest().config().quantization_bits,
          tensors.manifest().config().quantization_group_size,
          static_cast<float>(tensors.manifest().config().rms_norm_epsilon),
          false) {
    const ModelConfig& config = tensors.manifest().config();
    layers_.reserve(config.layer_count);
    for (std::size_t index = 0; index < config.layer_count; ++index) {
        layers_.push_back(std::make_unique<DecoderLayer>(tensors, index, config));
    }
}

ModelDecodeState QwenModel::make_state() const {
    return ModelDecodeState(layers_.size());
}

ModelDecodeState snapshot_decode_state(const ModelDecodeState& state) {
    ModelDecodeState snapshot(state.layers.size());
    snapshot.token_count = state.token_count;
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        snapshot.layers[index] = snapshot_decoder_layer_state(state.layers[index]);
    }
    return snapshot;
}

ModelDecodeState QwenModel::snapshot_state(const ModelDecodeState& state) const {
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    return snapshot_decode_state(state);
}

MlxArray QwenModel::embed(const std::uint32_t token) const {
    if (token >= vocabulary_size_) throw std::runtime_error("token id is out of range");
    const std::vector<std::int32_t> token_values{static_cast<std::int32_t>(token)};
    const std::vector<int> token_shape{1};
    const MlxArray ids = MlxArray::from_int32(token_values, token_shape);
    MlxArray value = MlxArray::dequantize(
        MlxArray::take_axis(embedding_.weight, ids, 0),
        MlxArray::take_axis(embedding_.scales, ids, 0),
        MlxArray::take_axis(embedding_.biases, ids, 0),
        group_size_,
        bits_);
    const std::vector<int> shape{1, 1, dimension(hidden_size_, "hidden size")};
    return value.reshape(shape);
}

MlxArray QwenModel::forward_decode(const std::uint32_t token, ModelDecodeState& state) const {
    return std::move(forward_decode_capture(token, state).logits);
}

TargetDecodeStep QwenModel::forward_decode_capture(
    const std::uint32_t token,
    ModelDecodeState& state) const {
    HiddenDecodeStep hidden = forward_hidden_decode_impl(token, state, nullptr, nullptr);
    MlxArray logits = MlxArray::quantized_matmul(
        hidden.mixed,
        language_head_.weight,
        language_head_.scales,
        language_head_.biases,
        group_size_,
        bits_);
    return {
        .logits = std::move(logits),
        .pre_mixer_stream = std::move(hidden.pre_mixer_stream),
    };
}

MlxArray QwenModel::consume_decode_capture(
    const std::uint32_t token,
    ModelDecodeState& state) const {
    HiddenDecodeStep hidden =
        forward_hidden_decode_impl(token, state, nullptr, nullptr);
    hidden.pre_mixer_stream.eval();
    return std::move(hidden.pre_mixer_stream);
}

std::vector<MlxArray> QwenModel::prefill_chunk(
    const std::span<const std::uint32_t> tokens,
    ModelDecodeState& state,
    std::vector<double>* layer_ms) const {
    constexpr std::size_t max_prefill_rows = 256;
    if (tokens.empty() || tokens.size() > max_prefill_rows) {
        throw std::runtime_error("prefill chunk must contain 1 to 256 tokens");
    }
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    if (state.token_count > std::numeric_limits<std::size_t>::max() - tokens.size()) {
        throw std::runtime_error("prefill token count overflow");
    }
    if (layer_ms != nullptr) {
        if (layer_ms->empty()) {
            layer_ms->resize(layers_.size(), 0.0);
        } else if (layer_ms->size() != layers_.size()) {
            throw std::runtime_error("prefill profile layer count mismatch");
        }
    }

    std::vector<MlxArray> streams;
    streams.reserve(tokens.size());
    for (const std::uint32_t token : tokens) {
        streams.push_back(HyperConnection::initialize_stream(embed(token), stream_count_));
    }
    MlxArray stream_batch = concatenate_sequence_rows(streams);
    streams.clear();
    const std::size_t barrier_stride = prefill_barrier_stride();
    auto barrier_started = std::chrono::steady_clock::now();
    std::size_t barrier_begin = 0;
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        stream_batch = layers_[layer]->forward_prefill(
            std::move(stream_batch), tokens, state.layers[layer]);
        const bool barrier = (layer + 1) % barrier_stride == 0 ||
            layer + 1 == layers_.size();
        if (barrier) {
            stream_batch.eval();
            if (layer_ms != nullptr) {
                const auto now = std::chrono::steady_clock::now();
                const double window_ms = std::chrono::duration<double, std::milli>(
                    now - barrier_started).count();
                const double per_layer_ms = window_ms /
                    static_cast<double>(layer + 1 - barrier_begin);
                for (std::size_t profiled = barrier_begin; profiled <= layer; ++profiled) {
                    (*layer_ms)[profiled] += per_layer_ms;
                }
                barrier_started = now;
                barrier_begin = layer + 1;
            }
        }
    }
    state.token_count += tokens.size();
    streams.reserve(tokens.size());
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        streams.push_back(slice_sequence_row(stream_batch, row));
    }
    return streams;
}

std::vector<TargetVerifyStep> QwenModel::forward_verify_layer_major_reference(
    const std::span<const std::uint32_t> tokens,
    const ModelDecodeState& origin,
    std::vector<double>* layer_ms,
    double* head_ms) const {
    if (tokens.empty() || tokens.size() > 5) {
        throw std::runtime_error("target verify row count must be between 1 and 5");
    }
    if (origin.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    if (origin.token_count > std::numeric_limits<std::size_t>::max() - tokens.size()) {
        throw std::runtime_error("target verify token count overflow");
    }
    if (layer_ms != nullptr) {
        layer_ms->clear();
        layer_ms->reserve(layers_.size());
    }
    if (head_ms != nullptr) *head_ms = 0.0;

    std::vector<MlxArray> streams;
    streams.reserve(tokens.size());
    std::vector<ModelDecodeState> checkpoints;
    checkpoints.reserve(tokens.size());
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        streams.push_back(HyperConnection::initialize_stream(embed(tokens[row]), stream_count_));
        checkpoints.emplace_back(layers_.size());
        checkpoints.back().token_count = origin.token_count + row + 1;
    }

    const std::size_t barrier_stride = verify_barrier_stride();

    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        std::chrono::steady_clock::time_point layer_started;
        if (layer_ms != nullptr) layer_started = std::chrono::steady_clock::now();
        std::vector<DecoderLayerState> layer_checkpoints;
        streams = layers_[layer]->forward_verify_dense_batched(
            std::move(streams), tokens, origin.layers[layer], layer_checkpoints);
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            checkpoints[row].layers[layer] = std::move(layer_checkpoints[row]);
        }
        // Adjacent layers share one lazy graph to remove synchronization
        // overhead. The default 16-layer group is memory- and parity-gated;
        // an explicit stride keeps a clean diagnostic rollback.
        if ((layer + 1) % barrier_stride == 0 || layer + 1 == layers_.size()) {
            std::vector<const MlxArray*> layer_outputs;
            layer_outputs.reserve(streams.size());
            for (const MlxArray& stream : streams) layer_outputs.push_back(&stream);
            MlxArray::eval_all(layer_outputs);
        }
        if (layer_ms != nullptr) {
            layer_ms->push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - layer_started).count());
        }
    }

    std::vector<TargetVerifyStep> result;
    result.reserve(tokens.size());
    const char* batch_head = std::getenv("QWEN38_BATCH_VERIFY_HEAD");
    if (batch_head != nullptr && std::string_view(batch_head) == "0") {
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            HyperConnectionRead final = final_mixer_.read(streams[row]);
            MlxArray logits = MlxArray::quantized_matmul(
                final.mixed,
                language_head_.weight,
                language_head_.scales,
                language_head_.biases,
                group_size_,
                bits_);
            result.push_back({
                .logits = std::move(logits),
                .pre_mixer_stream = std::move(streams[row]),
                .state_after = std::move(checkpoints[row]),
            });
        }
        return result;
    }

    std::chrono::steady_clock::time_point head_started;
    if (head_ms != nullptr) head_started = std::chrono::steady_clock::now();
    MlxArray stream_batch = concatenate_sequence_rows(streams);
    HyperConnectionRead final = final_mixer_.read(stream_batch);
    MlxArray logits_batch = MlxArray::quantized_matmul(
        final.mixed,
        language_head_.weight,
        language_head_.scales,
        language_head_.biases,
        group_size_,
        bits_);
    logits_batch.eval();
    if (head_ms != nullptr) {
        *head_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - head_started).count();
    }
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        result.push_back({
            .logits = slice_sequence_row(logits_batch, row),
            .pre_mixer_stream = std::move(streams[row]),
            .state_after = std::move(checkpoints[row]),
        });
    }
    return result;
}

MlxArray QwenModel::trace_decode(
    const std::uint32_t token,
    ModelDecodeState& state,
    std::vector<double>& layer_checksums,
    std::vector<double>& layer_ms) const {
    layer_checksums.clear();
    layer_checksums.reserve(layers_.size());
    layer_ms.clear();
    layer_ms.reserve(layers_.size());
    return forward_decode_impl(token, state, &layer_checksums, &layer_ms);
}

MlxArray QwenModel::forward_decode_impl(
    const std::uint32_t token,
    ModelDecodeState& state,
    std::vector<double>* layer_checksums,
    std::vector<double>* layer_ms) const {
    HiddenDecodeStep hidden =
        forward_hidden_decode_impl(token, state, layer_checksums, layer_ms);
    return MlxArray::quantized_matmul(
        hidden.mixed,
        language_head_.weight,
        language_head_.scales,
        language_head_.biases,
        group_size_,
        bits_);
}

QwenModel::HiddenDecodeStep QwenModel::forward_hidden_decode_impl(
    const std::uint32_t token,
    ModelDecodeState& state,
    std::vector<double>* layer_checksums,
    std::vector<double>* layer_ms) const {
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("model state layer count mismatch");
    }
    MlxArray stream = HyperConnection::initialize_stream(embed(token), stream_count_);
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const auto started = std::chrono::steady_clock::now();
        stream = layers_[index]->forward_decode(stream, token, state.layers[index]);
        if (layer_checksums != nullptr) {
            const std::vector<float> values = stream.astype(MLX_FLOAT32).to_float32();
            layer_checksums->push_back(
                std::accumulate(values.begin(), values.end(), 0.0));
        }
        if (layer_ms != nullptr) {
            layer_ms->push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
    }
    HyperConnectionRead final = final_mixer_.read(stream);
    ++state.token_count;
    return {
        .mixed = std::move(final.mixed),
        .pre_mixer_stream = std::move(stream),
    };
}

void QwenModel::consume_decode(
    const std::uint32_t token,
    ModelDecodeState& state) const {
    HiddenDecodeStep hidden = forward_hidden_decode_impl(token, state, nullptr, nullptr);
    static_cast<void>(hidden.mixed.astype(MLX_FLOAT32).to_float32());
}

GreedyStep QwenModel::greedy_decode(const std::uint32_t token, ModelDecodeState& state) const {
    MlxArray logits = forward_decode(token, state);
    if (logits.size() != vocabulary_size_) throw std::runtime_error("language head width mismatch");
    MlxArray token_array = logits.argmax_all();
    MlxArray selected_logit = MlxArray::take(logits, token_array).astype(MLX_FLOAT32);
    selected_logit.eval();
    return {
        .token = token_array.item_uint32(),
        .logit = selected_logit.item_float32(),
    };
}

} // namespace qwen38
