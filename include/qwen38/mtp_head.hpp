#pragma once

#include "qwen38/decoder_layer.hpp"
#include "qwen38/hyper_connection.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace qwen38 {

struct MtpDecodeState {
    DecoderLayerState layer;
    std::size_t row_count{0};
    std::optional<std::size_t> position_base;
};

[[nodiscard]] MtpDecodeState snapshot_mtp_decode_state(const MtpDecodeState& state);

struct MtpDecodeStep {
    MlxArray logits;
    MlxArray pre_mixer_stream;
};

struct MtpTrace {
    MlxArray combined_stream;
    DecoderLayerTrace layer;
    MlxArray final_mixed;
};

class QwenMtpHead final {
public:
    explicit QwenMtpHead(MlxTensorStore& tensors);

    [[nodiscard]] MtpDecodeState make_state() const { return {}; }
    [[nodiscard]] MtpDecodeState snapshot_state(const MtpDecodeState& state) const {
        return snapshot_mtp_decode_state(state);
    }
    [[nodiscard]] MtpDecodeStep forward_decode(
        const MlxArray& target_pre_mixer_stream,
        std::uint32_t next_token,
        std::size_t query_position,
        MtpDecodeState& state,
        MtpTrace* trace = nullptr) const;
    [[nodiscard]] MtpDecodeStep forward_decode_lazy_token(
        const MlxArray& target_pre_mixer_stream,
        const MlxArray& next_token,
        std::size_t query_position,
        MtpDecodeState& state,
        std::size_t adapter_depth = 0,
        MlxArray* final_mixed_trace = nullptr) const;
    void consume_decode(
        const MlxArray& target_pre_mixer_stream,
        std::uint32_t next_token,
        std::size_t query_position,
        MtpDecodeState& state) const;
    void consume_committed_batch(
        std::span<const MlxArray* const> target_pre_mixer_streams,
        std::span<const std::uint32_t> tokens,
        std::size_t query_position,
        MtpDecodeState& state) const;

private:
    struct QuantizedProjection {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
        int bits{0};
    };

    [[nodiscard]] static QuantizedProjection load_projection(
        MlxTensorStore& tensors,
        const char* prefix,
        int bits);
    [[nodiscard]] MlxArray embed(std::uint32_t token) const;
    [[nodiscard]] MlxArray embed(const MlxArray& token) const;
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const QuantizedProjection& projection) const;
    [[nodiscard]] MlxArray forward_stream(
        const MlxArray& target_pre_mixer_stream,
        std::uint32_t next_token,
        std::size_t query_position,
        MtpDecodeState& state,
        MtpTrace* trace) const;
    [[nodiscard]] MlxArray forward_stream_lazy_token(
        const MlxArray& target_pre_mixer_stream,
        const MlxArray& next_token,
        std::size_t query_position,
        MtpDecodeState& state,
        MtpTrace* trace) const;

    std::size_t hidden_size_;
    std::size_t stream_count_;
    std::size_t vocabulary_size_;
    int group_size_;
    int mtp_bits_;
    float epsilon_;
    QuantizedProjection embedding_;
    QuantizedProjection language_head_;
    QuantizedProjection fc_embedding_;
    QuantizedProjection fc_hidden_;
    MlxArray embedding_norm_;
    MlxArray hidden_norm_;
    DecoderLayer layer_;
    HyperConnection final_mixer_;
    std::unique_ptr<MlxSafetensors> head_adapter_store_;
    std::vector<MlxArray> head_adapter_a_;
    MlxArray head_adapter_b_;
    std::unique_ptr<MlxSafetensors> hidden_adapter_store_;
    std::vector<MlxArray> hidden_adapter_a_;
    MlxArray hidden_adapter_b_;
};

} // namespace qwen38
