#include "qwen38/sparse_moe.hpp"

#include "moe_metal_kernels.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
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

std::shared_ptr<MlxMetalKernel> fused_gate_up_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"x", "gw", "gs", "gb", "uw", "us", "ub", "experts"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_gate_up_q4", inputs, "h", moe_metal::gate_up, moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> fused_down_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"h", "dw", "ds", "db", "experts", "rw"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_down_q4_qdot", inputs, "y", moe_metal::down, moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> fused_verify_gate_up_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"x", "gw", "gs", "gb", "uw", "us", "ub", "experts"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_verify_gate_up_q4",
            inputs,
            "h",
            moe_metal::gate_up_verify,
            moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> fused_verify_down_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"h", "dw", "ds", "db", "experts", "rw"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_verify_down_q4_qdot",
            inputs,
            "y",
            moe_metal::down_verify,
            moe_metal::header);
    }();
    return kernel;
}

std::size_t layer_index(const std::string_view prefix) {
    constexpr std::string_view marker = ".layers.";
    const std::size_t begin = prefix.find(marker);
    if (begin == std::string_view::npos) throw std::runtime_error("MoE layer prefix is invalid");
    const std::size_t digits = begin + marker.size();
    const std::size_t end = prefix.find('.', digits);
    std::size_t result = 0;
    const auto parsed = std::from_chars(
        prefix.data() + digits,
        prefix.data() + (end == std::string_view::npos ? prefix.size() : end),
        result);
    if (parsed.ec != std::errc{}) throw std::runtime_error("MoE layer index is invalid");
    return result;
}

bool resident_layer_enabled(const std::size_t layer) {
    const char* raw = std::getenv("QWEN38_RESIDENT_EXPERT_RANGE");
    if (raw == nullptr || *raw == '\0') return false;
    const std::string_view value(raw);
    const std::size_t separator = value.find(':');
    if (separator == std::string_view::npos) {
        throw std::runtime_error("QWEN38_RESIDENT_EXPERT_RANGE must be BEGIN:END");
    }
    std::size_t begin = 0;
    std::size_t end = 0;
    const auto first = std::from_chars(value.data(), value.data() + separator, begin);
    const auto second = std::from_chars(
        value.data() + separator + 1, value.data() + value.size(), end);
    if (first.ec != std::errc{} || second.ec != std::errc{} || begin >= end || end > 48 ||
        end - begin > 22) {
        throw std::runtime_error(
            "QWEN38_RESIDENT_EXPERT_RANGE must be within 0:48 and span at most 22 layers");
    }
    return layer >= begin && layer < end;
}

MlxArray slice_sequence_row(const MlxArray& batch, const std::size_t row) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() != 3 || row >= static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("MoE verifier row is out of range");
    }
    return batch.slice(
        std::vector<int>{0, static_cast<int>(row), 0},
        std::vector<int>{1, static_cast<int>(row + 1), shape[2]},
        std::vector<int>{1, 1, 1});
}

MlxArray concatenate_sequence_rows(const std::vector<MlxArray>& rows) {
    if (rows.empty()) throw std::runtime_error("MoE verifier has no routed outputs");
    MlxArray result = rows.front().share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, rows[row], 1);
    }
    return result;
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
    if (resident_layer_enabled(layer_index(prefix))) {
        make_resident(expert_gate_);
        make_resident(expert_up_);
        make_resident(expert_down_);
    }
    const char* fused = std::getenv("QWEN38_FUSED_MOE");
    if (fused != nullptr && std::string_view(fused) == "1" &&
        expert_count_ >= experts_per_token_ && experts_per_token_ == 10 && bits_ == 4 &&
        group_size_ == 64) {
        fused_gate_up_ = fused_gate_up_kernel();
        fused_down_ = fused_down_kernel();
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

void SparseMoe::make_resident(QuantizedProjection& projection) {
    projection.weight.lock_pages();
    projection.scales.lock_pages();
    projection.biases.lock_pages();
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

MlxArray SparseMoe::forward_experts_decode(const MlxArray& input) const {
    MlxArray expert_sum;
    if (fused_gate_up_ && fused_down_) {
        MlxArray experts;
        MlxArray weights;
        const char* device_router = std::getenv("QWEN38_DEVICE_ROUTER");
        if (device_router != nullptr && std::string_view(device_router) == "1") {
            MlxArray gates =
                MlxArray::matmul(input, router_weight_.transpose()).softmax_axis(-1);
            MlxArray partition =
                gates.argpartition_axis(-static_cast<int>(experts_per_token_), -1);
            const std::vector<int> start{
                0, 0, static_cast<int>(expert_count_ - experts_per_token_)};
            const std::vector<int> stop{1, 1, static_cast<int>(expert_count_)};
            const std::vector<int> strides{1, 1, 1};
            experts = partition.slice(start, stop, strides).reshape(
                std::vector<int>{static_cast<int>(experts_per_token_)});
            weights = MlxArray::take_along_axis(gates, partition.slice(start, stop, strides), -1)
                          .reshape(std::vector<int>{static_cast<int>(experts_per_token_)});
            if (normalize_topk_probability_) {
                weights = MlxArray::divide(weights, weights.sum_axis(-1, true));
            }
            weights = weights.astype(MLX_FLOAT32);
        } else {
            const RouterSelection selection = route_decode(input);
            std::vector<std::int32_t> expert_values;
            expert_values.reserve(selection.experts.size());
            for (const std::size_t expert : selection.experts) {
                expert_values.push_back(static_cast<std::int32_t>(expert));
            }
            const std::vector<int> selected_shape{static_cast<int>(experts_per_token_)};
            experts = MlxArray::from_int32(expert_values, selected_shape);
            weights = MlxArray::from_float32(selection.weights, selected_shape);
        }
        const MlxArray* gate_inputs[]{
            &input,
            &expert_gate_.weight, &expert_gate_.scales, &expert_gate_.biases,
            &expert_up_.weight, &expert_up_.scales, &expert_up_.biases,
            &experts};
        const std::vector<int> hidden_shape{10, 640};
        const std::array<int, 3> gate_grid{200 * 1024, 1, 1};
        const std::array<int, 3> threadgroup{1024, 1, 1};
        MlxArray hidden = fused_gate_up_->apply(
            gate_inputs, hidden_shape, input.dtype(), gate_grid, threadgroup);
        const MlxArray* down_inputs[]{
            &hidden,
            &expert_down_.weight, &expert_down_.scales, &expert_down_.biases,
            &experts, &weights};
        const std::vector<int> output_shape{1, 1, 2560};
        const std::array<int, 3> down_grid{320 * 64, 1, 1};
        const std::array<int, 3> down_threadgroup{64, 1, 1};
        expert_sum = fused_down_->apply(
            down_inputs, output_shape, input.dtype(), down_grid, down_threadgroup);
    } else {
        const RouterSelection selection = route_decode(input);
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
    }

    return expert_sum;
}

MlxArray SparseMoe::forward_shared(const MlxArray& input) const {
    MlxArray shared_gate = project(input, shared_gate_).silu();
    MlxArray shared_up = project(input, shared_up_);
    MlxArray shared_hidden = MlxArray::multiply(shared_gate, shared_up);
    MlxArray shared_output = project(shared_hidden, shared_down_);
    MlxArray shared_router = MlxArray::matmul(
        input, shared_router_weight_.transpose()).sigmoid();
    return MlxArray::multiply(shared_output, shared_router);
}

MlxArray SparseMoe::forward_decode(const MlxArray& input) const {
    return MlxArray::add(forward_experts_decode(input), forward_shared(input));
}

MlxArray SparseMoe::forward_verify(const MlxArray& input) const {
    const std::vector<int> shape = input.shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > 64) {
        throw std::runtime_error("MoE batch requires shape [1,S,hidden], S=1..64");
    }
    const char* device_router = std::getenv("QWEN38_DEVICE_ROUTER");
    if (fused_gate_up_ && fused_down_ && device_router != nullptr &&
        std::string_view(device_router) == "1") {
        const int rows = shape[1];
        MlxArray gates = MlxArray::matmul(input, router_weight_.transpose()).softmax_axis(-1);
        MlxArray partition = gates.argpartition_axis(-static_cast<int>(experts_per_token_), -1);
        const std::vector<int> start{
            0, 0, static_cast<int>(expert_count_ - experts_per_token_)};
        const std::vector<int> stop{1, rows, static_cast<int>(expert_count_)};
        const std::vector<int> strides{1, 1, 1};
        MlxArray selected = partition.slice(start, stop, strides);
        MlxArray experts = selected.reshape(
            std::vector<int>{rows, static_cast<int>(experts_per_token_)});
        MlxArray weights = MlxArray::take_along_axis(gates, selected, -1).reshape(
            std::vector<int>{rows, static_cast<int>(experts_per_token_)});
        if (normalize_topk_probability_) {
            weights = MlxArray::divide(weights, weights.sum_axis(-1, true));
        }
        weights = weights.astype(MLX_FLOAT32);
        const MlxArray* gate_inputs[]{
            &input,
            &expert_gate_.weight, &expert_gate_.scales, &expert_gate_.biases,
            &expert_up_.weight, &expert_up_.scales, &expert_up_.biases,
            &experts};
        const std::vector<int> hidden_shape{rows, 10, 640};
        const std::array<int, 3> gate_grid{
            static_cast<int>(rows * 200 * 1024), 1, 1};
        const std::array<int, 3> gate_threadgroup{1024, 1, 1};
        MlxArray hidden = fused_verify_gate_up_kernel()->apply(
            gate_inputs, hidden_shape, input.dtype(), gate_grid, gate_threadgroup);
        const MlxArray* down_inputs[]{
            &hidden,
            &expert_down_.weight, &expert_down_.scales, &expert_down_.biases,
            &experts, &weights};
        const std::vector<int> output_shape{1, rows, 2560};
        const std::array<int, 3> down_grid{
            static_cast<int>(rows * 320 * 64), 1, 1};
        const std::array<int, 3> down_threadgroup{64, 1, 1};
        MlxArray routed = fused_verify_down_kernel()->apply(
            down_inputs, output_shape, input.dtype(), down_grid, down_threadgroup);
        return MlxArray::add(routed, forward_shared(input));
    }

    std::vector<MlxArray> routed;
    routed.reserve(static_cast<std::size_t>(shape[1]));
    for (int row = 0; row < shape[1]; ++row) {
        routed.push_back(forward_experts_decode(
            slice_sequence_row(input, static_cast<std::size_t>(row))));
    }
    return MlxArray::add(concatenate_sequence_rows(routed), forward_shared(input));
}

} // namespace qwen38
