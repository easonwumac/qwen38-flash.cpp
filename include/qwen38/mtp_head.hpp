#pragma once

#include "qwen38/decoder_layer.hpp"
#include "qwen38/hyper_connection.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace qwen38 {

struct MtpDecodeState {
    DecoderLayerState layer;
    std::size_t row_count{0};
    std::optional<std::size_t> position_base;
};

struct MtpDecodeStep {
    MlxArray logits;
    MlxArray pre_mixer_stream;
};

class QwenMtpHead final {
public:
    explicit QwenMtpHead(MlxTensorStore& tensors);

    [[nodiscard]] MtpDecodeState make_state() const { return {}; }
    [[nodiscard]] MtpDecodeStep forward_decode(
        const MlxArray& target_pre_mixer_stream,
        std::uint32_t next_token,
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
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const QuantizedProjection& projection) const;

    std::size_t hidden_size_;
    std::size_t stream_count_;
    std::size_t vocabulary_size_;
    int group_size_;
    float epsilon_;
    QuantizedProjection embedding_;
    QuantizedProjection language_head_;
    QuantizedProjection fc_embedding_;
    QuantizedProjection fc_hidden_;
    MlxArray embedding_norm_;
    MlxArray hidden_norm_;
    DecoderLayer layer_;
    HyperConnection final_mixer_;
};

} // namespace qwen38
