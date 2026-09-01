#pragma once

#include "qwen38/mlx_backend.hpp"

#include <cstddef>
#include <string_view>

namespace qwen38 {

struct HyperConnectionRead {
    MlxArray mixed;
    MlxArray injection;
    bool has_injection{false};
};

class HyperConnection final {
public:
    HyperConnection(
        MlxTensorStore& tensors,
        std::string_view prefix,
        std::size_t hidden_size,
        std::size_t stream_count,
        std::size_t quantization_bits,
        std::size_t quantization_group_size,
        float rms_norm_epsilon,
        bool with_injection);

    [[nodiscard]] static MlxArray initialize_stream(
        const MlxArray& embedding,
        std::size_t stream_count);
    [[nodiscard]] HyperConnectionRead read(const MlxArray& stream) const;
    [[nodiscard]] MlxArray write(
        const MlxArray& stream,
        const MlxArray& block_output,
        const MlxArray& injection) const;

private:
    struct Projection {
        MlxArray weight;
        MlxArray scales;
        MlxArray biases;
        bool quantized{false};
        int bits{0};
    };

    [[nodiscard]] Projection load_projection(
        MlxTensorStore& tensors,
        std::string_view name) const;
    [[nodiscard]] MlxArray project(
        const MlxArray& input,
        const Projection& projection) const;

    std::size_t hidden_size_;
    std::size_t stream_count_;
    int bits_;
    int group_size_;
    float rms_norm_epsilon_;
    bool with_injection_;
    MlxArray norm_weight_;
    Projection down_;
    Projection up_;
    Projection injection_;
    MlxArray injection_dense_;
    bool fused_injection_ready_{false};
};

} // namespace qwen38
