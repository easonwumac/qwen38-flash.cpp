#pragma once

#include "qwen38/decoder_layer.hpp"
#include "qwen38/hyper_connection.hpp"
#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace qwen38 {

struct ModelDecodeState {
    explicit ModelDecodeState(std::size_t layer_count) : layers(layer_count) {}
    std::vector<DecoderLayerState> layers;
    std::size_t token_count{0};
};

[[nodiscard]] ModelDecodeState snapshot_decode_state(const ModelDecodeState& state);
void eval_with_decode_state(const MlxArray& output, const ModelDecodeState& state);

struct GreedyStep {
    std::uint32_t token{0};
    float logit{0.0F};
};

// The Qwen3.8 MTP companion consumes the four-stream residual before the
// target's final mixer. Keeping it beside the logits avoids a second trunk
// forward and prevents callers from accidentally feeding the collapsed hidden.
struct TargetDecodeStep {
    MlxArray logits;
    MlxArray pre_mixer_stream;
};

struct TargetVerifyStep {
    MlxArray logits;
    MlxArray final_mixed;
    MlxArray pre_mixer_stream;
    ModelDecodeState state_after;
};

// Move-only state for pausing a full-width prefill chunk at the same layer
// barriers used by the uninterrupted path. The row shape and arithmetic stay
// unchanged; only independent work may run between barrier groups.
struct ModelPrefillChunk {
    MlxArray stream_batch;
    std::size_t next_layer{0};
    std::size_t row_count{0};
};

class QwenModel final {
public:
    explicit QwenModel(MlxTensorStore& tensors);

    [[nodiscard]] ModelDecodeState make_state() const;
    [[nodiscard]] ModelDecodeState snapshot_state(const ModelDecodeState& state) const;
    [[nodiscard]] MlxArray forward_decode(std::uint32_t token, ModelDecodeState& state) const;
    [[nodiscard]] TargetDecodeStep forward_decode_capture(
        std::uint32_t token,
        ModelDecodeState& state) const;
    [[nodiscard]] std::vector<TargetDecodeStep> forward_decode_capture_multi(
        std::span<const std::uint32_t> tokens,
        std::span<ModelDecodeState* const> states) const;
    [[nodiscard]] MlxArray consume_decode_capture(
        std::uint32_t token,
        ModelDecodeState& state) const;
    [[nodiscard]] std::vector<MlxArray> prefill_chunk(
        std::span<const std::uint32_t> tokens,
        ModelDecodeState& state,
        std::vector<double>* layer_ms = nullptr) const;
    [[nodiscard]] MlxArray prefill_chunk_batch(
        std::span<const std::uint32_t> tokens,
        ModelDecodeState& state,
        std::vector<double>* layer_ms = nullptr) const;
    // Process several consecutive prompt chunks at each graph barrier before
    // advancing to the next layers. Token order within every layer is unchanged.
    [[nodiscard]] std::vector<MlxArray> prefill_chunks_layer_major(
        std::span<const std::uint32_t> tokens,
        std::size_t chunk_rows,
        ModelDecodeState& state,
        std::vector<double>* layer_ms = nullptr) const;
    [[nodiscard]] ModelPrefillChunk begin_prefill_chunk_batch(
        std::span<const std::uint32_t> tokens,
        const ModelDecodeState& state) const;
    // Returns true after the final layer and commits row_count to state.
    [[nodiscard]] bool advance_prefill_chunk_batch(
        ModelPrefillChunk& chunk,
        std::span<const std::uint32_t> tokens,
        ModelDecodeState& state,
        std::vector<double>* layer_ms = nullptr) const;
    void set_prefill_qmeta_cache_allowed(bool allowed) const noexcept;
    void clear_prefill_qmeta_cache() const;
    [[nodiscard]] std::vector<TargetVerifyStep> forward_verify_layer_major_reference(
        std::span<const std::uint32_t> tokens,
        const ModelDecodeState& origin,
        std::vector<double>* layer_ms = nullptr,
        double* head_ms = nullptr) const;
    void materialize_speculative_state(ModelDecodeState& state) const;
    void consume_decode(std::uint32_t token, ModelDecodeState& state) const;
    [[nodiscard]] MlxArray trace_decode(
        std::uint32_t token,
        ModelDecodeState& state,
        std::vector<double>& layer_checksums,
        std::vector<double>& layer_ms) const;
    [[nodiscard]] GreedyStep greedy_decode(std::uint32_t token, ModelDecodeState& state) const;
    [[nodiscard]] std::size_t layer_count() const noexcept { return layers_.size(); }

private:
    struct HiddenDecodeStep {
        MlxArray mixed;
        MlxArray pre_mixer_stream;
    };

    struct QuantizedTensor {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
    };

    [[nodiscard]] static QuantizedTensor load_quantized(
        MlxTensorStore& tensors,
        const char* prefix);
    [[nodiscard]] MlxArray embed(std::uint32_t token) const;
    [[nodiscard]] MlxArray forward_decode_impl(
        std::uint32_t token,
        ModelDecodeState& state,
        std::vector<double>* layer_checksums,
        std::vector<double>* layer_ms) const;
    [[nodiscard]] HiddenDecodeStep forward_hidden_decode_impl(
        std::uint32_t token,
        ModelDecodeState& state,
        std::vector<double>* layer_checksums,
        std::vector<double>* layer_ms) const;

    std::size_t hidden_size_;
    std::size_t stream_count_;
    std::size_t vocabulary_size_;
    int embedding_bits_;
    int embedding_group_size_;
    int head_bits_;
    int head_group_size_;
    QuantizedTensor embedding_;
    QuantizedTensor language_head_;
    std::vector<std::unique_ptr<DecoderLayer>> layers_;
    HyperConnection final_mixer_;
};

} // namespace qwen38
