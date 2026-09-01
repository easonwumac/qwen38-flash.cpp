#include "qwen38/sparse_moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace qwen38 {
namespace {

int checked_int(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid MoE parameter: ") + name);
    }
    return static_cast<int>(value);
}

} // namespace

SparseMoe::SparseMoe(
    MlxTensorStore& tensors,
    const std::string_view prefix,
    const std::size_t expert_count,
    const std::size_t experts_per_token,
    const std::size_t quantization_bits,
    const std::size_t quantization_group_size,
    const bool normalize_topk_probability)
    : expert_count_(expert_count),
      experts_per_token_(experts_per_token),
      bits_(checked_int(quantization_bits, "quantization_bits")),
      group_size_(checked_int(quantization_group_size, "quantization_group_size")),
      normalize_topk_probability_(normalize_topk_probability),
      router_weight_(tensors.tensor(std::string(prefix) + ".gate.weight")),
      expert_gate_(load_projection(tensors, std::string(prefix) + ".switch_mlp.gate_proj")),
      expert_up_(load_projection(tensors, std::string(prefix) + ".switch_mlp.up_proj")),
      expert_down_(load_projection(tensors, std::string(prefix) + ".switch_mlp.down_proj")),
      shared_gate_(load_projection(tensors, std::string(prefix) + ".shared_expert.gate_proj")),
      shared_up_(load_projection(tensors, std::string(prefix) + ".shared_expert.up_proj")),
      shared_down_(load_projection(tensors, std::string(prefix) + ".shared_expert.down_proj")),
      shared_router_weight_(tensors.tensor(std::string(prefix) + ".shared_expert_gate.weight")) {
    static_cast<void>(checked_int(expert_count_, "expert_count"));
    if (experts_per_token_ == 0 || experts_per_token_ > expert_count_) {
        throw std::runtime_error("invalid experts_per_token");
    }
}

SparseMoe::QuantizedProjection SparseMoe::load_projection(
    MlxTensorStore& tensors,
    const std::string_view name) {
    const std::string base(name);
    return {
        .weight = tensors.tensor(base + ".weight"),
        .scales = tensors.tensor(base + ".scales"),
        .biases = tensors.tensor(base + ".biases"),
    };
}

MlxArray SparseMoe::project(
    const MlxArray& input,
    const QuantizedProjection& projection) const {
    return MlxArray::quantized_matmul(
        input, projection.weight, projection.scales, projection.biases, group_size_, bits_);
}

MlxArray SparseMoe::project_expert(
    const MlxArray& input,
    const QuantizedProjection& projection,
    const std::size_t expert) const {
    if (expert >= expert_count_) throw std::runtime_error("expert index is out of range");
    const std::vector<std::int32_t> index_values{static_cast<std::int32_t>(expert)};
    const std::vector<int> index_shape{};
    const MlxArray index = MlxArray::from_int32(index_values, index_shape);
    const MlxArray weight = MlxArray::take_axis(projection.weight, index, 0);
    const MlxArray scales = MlxArray::take_axis(projection.scales, index, 0);
    const MlxArray biases = MlxArray::take_axis(projection.biases, index, 0);
    return MlxArray::quantized_matmul(
        input, weight, scales, biases, group_size_, bits_);
}

RouterSelection SparseMoe::route_decode(const MlxArray& input) const {
    const auto shape = input.shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1) {
        throw std::runtime_error("decode router requires input shape [1,1,hidden]");
    }
    MlxArray logits = MlxArray::matmul(input, router_weight_.transpose());
    const std::vector<float> values = logits.astype(MLX_FLOAT32).to_float32();
    if (values.size() != expert_count_) throw std::runtime_error("router width mismatch");
    const float maximum = *std::max_element(values.begin(), values.end());
    std::vector<float> probabilities(values.size());
    double denominator = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        probabilities[index] = std::exp(values[index] - maximum);
        denominator += probabilities[index];
    }
    for (float& probability : probabilities) {
        probability = static_cast<float>(static_cast<double>(probability) / denominator);
    }
    std::vector<std::size_t> order(expert_count_);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(),
        order.begin() + static_cast<std::ptrdiff_t>(experts_per_token_),
        order.end(),
        [&probabilities](const std::size_t left, const std::size_t right) {
            if (probabilities[left] != probabilities[right]) {
                return probabilities[left] > probabilities[right];
            }
            return left < right;
        });
    order.resize(experts_per_token_);
    std::vector<float> weights;
    weights.reserve(experts_per_token_);
    float selected_sum = 0.0F;
    for (const std::size_t expert : order) {
        weights.push_back(probabilities[expert]);
        selected_sum += probabilities[expert];
    }
    if (normalize_topk_probability_) {
        for (float& weight : weights) weight /= selected_sum;
    }
    return {.experts = std::move(order), .weights = std::move(weights)};
}

MlxArray SparseMoe::forward_decode(const MlxArray& input) const {
    const RouterSelection selection = route_decode(input);
    MlxArray expert_sum;
    bool has_expert = false;
    for (std::size_t index = 0; index < selection.experts.size(); ++index) {
        const std::size_t expert = selection.experts[index];
        MlxArray gate = project_expert(input, expert_gate_, expert).silu();
        MlxArray up = project_expert(input, expert_up_, expert);
        MlxArray hidden = MlxArray::multiply(gate, up);
        MlxArray output = project_expert(hidden, expert_down_, expert);
        const std::vector<float> weight_value{selection.weights[index]};
        const std::vector<int> scalar_shape{};
        MlxArray weight = MlxArray::from_float32(weight_value, scalar_shape).astype(output.dtype());
        MlxArray weighted = MlxArray::multiply(output, weight);
        if (!has_expert) {
            expert_sum = std::move(weighted);
            has_expert = true;
        } else {
            expert_sum = MlxArray::add(expert_sum, weighted);
        }
    }

    MlxArray shared_gate = project(input, shared_gate_).silu();
    MlxArray shared_up = project(input, shared_up_);
    MlxArray shared_hidden = MlxArray::multiply(shared_gate, shared_up);
    MlxArray shared_output = project(shared_hidden, shared_down_);
    MlxArray shared_router = MlxArray::matmul(
        input, shared_router_weight_.transpose()).sigmoid();
    return MlxArray::add(expert_sum, MlxArray::multiply(shared_output, shared_router));
}

} // namespace qwen38
