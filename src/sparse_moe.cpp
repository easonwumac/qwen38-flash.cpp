#include "qwen38/sparse_moe.hpp"

#include "qwen38/quantization_geometry.hpp"

#include "moe_metal_kernels.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
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

std::shared_ptr<MlxMetalKernel> fused_q8_gate_up_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"x", "gw", "gs", "gb", "uw", "us", "ub", "experts", "sigtab"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_gate_up_q8_exact", inputs, "h",
            moe_metal::gate_up_q8_exact, moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> fused_q8_down_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"h", "dw", "ds", "db", "experts", "rw"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_down_q8_exact", inputs, "y",
            moe_metal::down_q8_exact, moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> fused_q8_verify_gate_up_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"x", "gw", "gs", "gb", "uw", "us", "ub", "experts", "sigtab"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_verify_gate_up_q8_exact", inputs, "h",
            moe_metal::gate_up_verify_q8_exact, moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> fused_q8_verify_down_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"h", "dw", "ds", "db", "experts", "rw"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_moe_verify_down_q8_exact", inputs, "y",
            moe_metal::down_verify_q8_exact, moe_metal::header);
    }();
    return kernel;
}

const MlxArray& bf16_sigmoid_table() {
    static const MlxArray table = [] {
        constexpr std::size_t count = 1U << 16;
        std::vector<float> patterns(count);
        for (std::size_t index = 0; index < count; ++index) {
            patterns[index] = std::bit_cast<float>(static_cast<std::uint32_t>(index) << 16);
        }
        const std::vector<int> shape{static_cast<int>(count)};
        MlxArray values = MlxArray::from_float32(patterns, shape).astype(MLX_BFLOAT16);
        MlxArray result = values.sigmoid();
        result.eval();
        return result;
    }();
    return table;
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

bool selected_softmax_router_enabled(const bool normalize_topk_probability) {
    const char* selected_softmax = std::getenv("QWEN38_SELECTED_SOFTMAX_ROUTER");
    return normalize_topk_probability && selected_softmax != nullptr &&
        std::string_view(selected_softmax) == "1";
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
      group_size_(checked_int(quantization_group_size, "quantization_group_size")),
      normalize_topk_probability_(normalize_topk_probability),
      router_weight_(tensors.tensor(std::string(prefix) + ".gate.weight")),
      expert_gate_(load_projection(
          tensors, std::string(prefix) + ".switch_mlp.gate_proj", quantization_group_size)),
      expert_up_(load_projection(
          tensors, std::string(prefix) + ".switch_mlp.up_proj", quantization_group_size)),
      expert_down_(load_projection(
          tensors, std::string(prefix) + ".switch_mlp.down_proj", quantization_group_size)),
      shared_gate_(load_projection(
          tensors, std::string(prefix) + ".shared_expert.gate_proj", quantization_group_size)),
      shared_up_(load_projection(
          tensors, std::string(prefix) + ".shared_expert.up_proj", quantization_group_size)),
      shared_down_(load_projection(
          tensors, std::string(prefix) + ".shared_expert.down_proj", quantization_group_size)),
      shared_router_weight_(tensors.tensor(std::string(prefix) + ".shared_expert_gate.weight")) {
    static_cast<void>(checked_int(quantization_bits, "quantization_bits"));
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
    const char* q8_exact = std::getenv("QWEN38_Q8_EXACT_MOE");
    if (fused != nullptr && std::string_view(fused) == "1" &&
        expert_count_ >= experts_per_token_ && experts_per_token_ == 10 &&
        expert_gate_.bits == expert_up_.bits && expert_gate_.bits == expert_down_.bits &&
        (expert_gate_.bits == 4 || expert_gate_.bits == 8) &&
        group_size_ == 64) {
        if (expert_gate_.bits == 8) {
            fused_q8_exact_ = q8_exact == nullptr || std::string_view(q8_exact) != "0";
            if (fused_q8_exact_) {
                fused_gate_up_ = fused_q8_gate_up_kernel();
                fused_down_ = fused_q8_down_kernel();
            }
        } else {
            fused_gate_up_ = fused_gate_up_kernel();
            fused_down_ = fused_down_kernel();
        }
    }
}

SparseMoe::QuantizedProjection SparseMoe::load_projection(
    MlxTensorStore& tensors,
    const std::string_view name,
    const std::size_t group_size) {
    const std::string base(name);
    MlxArray weight = tensors.tensor(base + ".weight");
    MlxArray scales = tensors.tensor(base + ".scales");
    const int bits = infer_affine_quantization_bits(
        weight.shape(), scales.shape(), group_size, "MoE");
    return {
        .weight = std::move(weight),
        .scales = std::move(scales),
        .biases = tensors.tensor(base + ".biases"),
        .bits = bits,
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
        input, projection.weight, projection.scales, projection.biases,
        group_size_, projection.bits);
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
        input, weight, scales, biases, group_size_, projection.bits);
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
            MlxArray logits = MlxArray::matmul(input, router_weight_.transpose());
            const bool use_selected_softmax =
                selected_softmax_router_enabled(normalize_topk_probability_);
            MlxArray gates = use_selected_softmax ? logits.share() : logits.softmax_axis(-1);
            MlxArray partition =
                gates.argpartition_axis(-static_cast<int>(experts_per_token_), -1);
            const std::vector<int> start{
                0, 0, static_cast<int>(expert_count_ - experts_per_token_)};
            const std::vector<int> stop{1, 1, static_cast<int>(expert_count_)};
            const std::vector<int> strides{1, 1, 1};
            experts = partition.slice(start, stop, strides).reshape(
                std::vector<int>{static_cast<int>(experts_per_token_)});
            weights = MlxArray::take_along_axis(
                gates, partition.slice(start, stop, strides), -1);
            if (use_selected_softmax) {
                weights = weights.softmax_axis(-1);
            } else if (normalize_topk_probability_) {
                weights = MlxArray::divide(weights, weights.sum_axis(-1, true));
            }
            weights = weights.reshape(
                std::vector<int>{static_cast<int>(experts_per_token_)});
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
        const MlxArray& sigmoid_table = bf16_sigmoid_table();
        const MlxArray* q4_gate_inputs[]{
            &input, &expert_gate_.weight, &expert_gate_.scales, &expert_gate_.biases,
            &expert_up_.weight, &expert_up_.scales, &expert_up_.biases, &experts};
        const MlxArray* q8_gate_inputs[]{
            &input, &expert_gate_.weight, &expert_gate_.scales, &expert_gate_.biases,
            &expert_up_.weight, &expert_up_.scales, &expert_up_.biases, &experts,
            &sigmoid_table};
        const std::vector<int> hidden_shape{10, 640};
        const std::array<int, 3> gate_grid{200 * 1024, 1, 1};
        const std::array<int, 3> threadgroup{1024, 1, 1};
        MlxArray hidden = fused_gate_up_->apply(
            fused_q8_exact_
                ? std::span<const MlxArray* const>(q8_gate_inputs)
                : std::span<const MlxArray* const>(q4_gate_inputs),
            hidden_shape, input.dtype(), gate_grid, threadgroup);
        MlxArray down_weights = fused_q8_exact_ ? weights.astype(input.dtype()) : weights.share();
        const MlxArray* down_inputs[]{
            &hidden,
            &expert_down_.weight, &expert_down_.scales, &expert_down_.biases,
            &experts, &down_weights};
        const std::vector<int> output_shape{1, 1, 2560};
        const std::array<int, 3> down_grid{
            fused_q8_exact_ ? 640 * 320 : 320 * 64, 1, 1};
        const std::array<int, 3> down_threadgroup{
            fused_q8_exact_ ? 320 : 64, 1, 1};
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
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > 512) {
        throw std::runtime_error("MoE batch requires shape [1,S,hidden], S=1..512");
    }
    const char* device_router = std::getenv("QWEN38_DEVICE_ROUTER");
    if (fused_gate_up_ && fused_down_ && device_router != nullptr &&
        std::string_view(device_router) == "1") {
        const int rows = shape[1];
        MlxArray logits = MlxArray::matmul(input, router_weight_.transpose());
        const bool use_selected_softmax =
            selected_softmax_router_enabled(normalize_topk_probability_);
        MlxArray gates = use_selected_softmax ? logits.share() : logits.softmax_axis(-1);
        MlxArray partition = gates.argpartition_axis(-static_cast<int>(experts_per_token_), -1);
        const std::vector<int> start{
            0, 0, static_cast<int>(expert_count_ - experts_per_token_)};
        const std::vector<int> stop{1, rows, static_cast<int>(expert_count_)};
        const std::vector<int> strides{1, 1, 1};
        MlxArray selected = partition.slice(start, stop, strides);
        MlxArray experts = selected.reshape(
            std::vector<int>{rows, static_cast<int>(experts_per_token_)});
        MlxArray weights = MlxArray::take_along_axis(gates, selected, -1);
        if (use_selected_softmax) {
            weights = weights.softmax_axis(-1);
        } else if (normalize_topk_probability_) {
            weights = MlxArray::divide(weights, weights.sum_axis(-1, true));
        }
        weights = weights.reshape(
            std::vector<int>{rows, static_cast<int>(experts_per_token_)});
        weights = weights.astype(MLX_FLOAT32);
        const MlxArray& sigmoid_table = bf16_sigmoid_table();
        const MlxArray* q4_gate_inputs[]{
            &input, &expert_gate_.weight, &expert_gate_.scales, &expert_gate_.biases,
            &expert_up_.weight, &expert_up_.scales, &expert_up_.biases, &experts};
        const MlxArray* q8_gate_inputs[]{
            &input, &expert_gate_.weight, &expert_gate_.scales, &expert_gate_.biases,
            &expert_up_.weight, &expert_up_.scales, &expert_up_.biases, &experts,
            &sigmoid_table};
        const std::vector<int> hidden_shape{rows, 10, 640};
        const std::array<int, 3> gate_grid{
            static_cast<int>(rows * 200 * 1024), 1, 1};
        const std::array<int, 3> gate_threadgroup{1024, 1, 1};
        MlxArray hidden = (fused_q8_exact_
            ? fused_q8_verify_gate_up_kernel()
            : fused_verify_gate_up_kernel())->apply(
                fused_q8_exact_
                    ? std::span<const MlxArray* const>(q8_gate_inputs)
                    : std::span<const MlxArray* const>(q4_gate_inputs),
                hidden_shape, input.dtype(), gate_grid, gate_threadgroup);
        MlxArray down_weights = fused_q8_exact_ ? weights.astype(input.dtype()) : weights.share();
        const MlxArray* down_inputs[]{
            &hidden,
            &expert_down_.weight, &expert_down_.scales, &expert_down_.biases,
            &experts, &down_weights};
        const std::vector<int> output_shape{1, rows, 2560};
        const std::array<int, 3> down_grid{
            static_cast<int>(rows * (fused_q8_exact_ ? 640 * 320 : 320 * 64)),
            1, 1};
        const std::array<int, 3> down_threadgroup{
            fused_q8_exact_ ? 320 : 64, 1, 1};
        MlxArray routed = (fused_q8_exact_
            ? fused_q8_verify_down_kernel()
            : fused_verify_down_kernel())->apply(
            down_inputs, output_shape, input.dtype(), down_grid, down_threadgroup);
        return MlxArray::add(routed, forward_shared(input));
    }

    const char* grouped = std::getenv("QWEN38_GROUPED_PREFILL");
    if ((!fused_gate_up_ || !fused_down_) && shape[1] > 1 &&
        device_router != nullptr && std::string_view(device_router) == "1" &&
        grouped != nullptr && std::string_view(grouped) == "1") {
        return forward_prefill(input);
    }

    std::vector<MlxArray> routed;
    routed.reserve(static_cast<std::size_t>(shape[1]));
    for (int row = 0; row < shape[1]; ++row) {
        routed.push_back(forward_experts_decode(
            slice_sequence_row(input, static_cast<std::size_t>(row))));
    }
    return MlxArray::add(concatenate_sequence_rows(routed), forward_shared(input));
}

MlxArray SparseMoe::forward_prefill(const MlxArray& input) const {
    return forward_prefill_impl(input, nullptr);
}

MlxArray SparseMoe::forward_prefill_profiled(
    const MlxArray& input,
    MoePrefillTimings& timings) const {
    timings = {};
    return forward_prefill_impl(input, &timings);
}

MlxArray SparseMoe::forward_prefill_impl(
    const MlxArray& input,
    MoePrefillTimings* timings) const {
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ms = [](const Clock::time_point started) {
        return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    };
    const std::vector<int> shape = input.shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > 512) {
        throw std::runtime_error("MoE prefill requires shape [1,S,hidden], S=1..512");
    }
    const char* grouped = std::getenv("QWEN38_GROUPED_PREFILL");
    const bool mixed_quantization = !fused_gate_up_ || !fused_down_;
    if ((!mixed_quantization && shape[1] < 16) || grouped == nullptr ||
        std::string_view(grouped) != "1") {
        return forward_verify(input);
    }

    const int rows = shape[1];
    const int hidden_size = shape[2];
    const int top_k = checked_int(experts_per_token_, "experts_per_token");
    const int slots = rows * top_k;
    const auto routing_started = Clock::now();
    MlxArray gates = MlxArray::matmul(input, router_weight_.transpose()).softmax_axis(-1);
    MlxArray partition = gates.argpartition_axis(-top_k, -1);
    const std::vector<int> start{0, 0, static_cast<int>(expert_count_) - top_k};
    const std::vector<int> stop{1, rows, static_cast<int>(expert_count_)};
    const std::vector<int> strides{1, 1, 1};
    MlxArray selected = partition.slice(start, stop, strides);
    MlxArray weights = MlxArray::take_along_axis(gates, selected, -1);
    if (normalize_topk_probability_) {
        weights = MlxArray::divide(weights, weights.sum_axis(-1, true));
    }
    weights = weights.astype(input.dtype());

    MlxArray flat_experts = selected.reshape(std::vector<int>{slots});
    MlxArray order = flat_experts.argsort_axis(0);
    MlxArray inverse_order = order.argsort_axis(0);
    MlxArray sorted_experts = MlxArray::take_axis(flat_experts, order, 0);
    const std::array<std::int32_t, 1> divisor_value{top_k};
    const std::array<int, 0> scalar_shape{};
    MlxArray divisor = MlxArray::from_int32(divisor_value, scalar_shape);
    MlxArray source_rows = MlxArray::floor_divide(order, divisor);
    MlxArray flat_input = input.reshape(std::vector<int>{rows, hidden_size});
    MlxArray gathered_input = MlxArray::take_axis(flat_input, source_rows, 0).reshape(
        std::vector<int>{slots, 1, hidden_size});
    MlxArray no_indices;
    if (timings != nullptr) {
        const std::array<const MlxArray*, 4> routing_outputs{
            &weights, &sorted_experts, &inverse_order, &gathered_input};
        MlxArray::eval_all(routing_outputs);
        timings->routing_ms = elapsed_ms(routing_started);
    }

    const auto gate_up_started = Clock::now();
    MlxArray gate = MlxArray::gather_quantized_matmul(
        gathered_input,
        expert_gate_.weight,
        expert_gate_.scales,
        expert_gate_.biases,
        no_indices,
        sorted_experts,
        group_size_,
        expert_gate_.bits,
        true).silu();
    MlxArray up = MlxArray::gather_quantized_matmul(
        gathered_input,
        expert_up_.weight,
        expert_up_.scales,
        expert_up_.biases,
        no_indices,
        sorted_experts,
        group_size_,
        expert_up_.bits,
        true);
    MlxArray expert_hidden = MlxArray::multiply(gate, up);
    if (timings != nullptr) {
        expert_hidden.eval();
        timings->gate_up_ms = elapsed_ms(gate_up_started);
    }
    const std::vector<int> expert_shape = expert_hidden.shape();
    if (expert_shape.size() != 3 || expert_shape[0] != slots || expert_shape[1] != 1) {
        throw std::runtime_error("grouped MoE gate/up returned an invalid shape");
    }
    const auto down_started = Clock::now();
    MlxArray down = MlxArray::gather_quantized_matmul(
        expert_hidden,
        expert_down_.weight,
        expert_down_.scales,
        expert_down_.biases,
        no_indices,
        sorted_experts,
        group_size_,
        expert_down_.bits,
        true).reshape(std::vector<int>{slots, hidden_size});
    MlxArray unsorted = MlxArray::take_axis(down, inverse_order, 0).reshape(
        std::vector<int>{1, rows, top_k, hidden_size});
    MlxArray weighted = MlxArray::multiply(
        unsorted, weights.reshape(std::vector<int>{1, rows, top_k, 1}));
    MlxArray routed = weighted.sum_axis(2);
    if (timings != nullptr) {
        routed.eval();
        timings->down_reduce_ms = elapsed_ms(down_started);
    }
    const auto shared_started = Clock::now();
    MlxArray shared = forward_shared(input);
    if (timings != nullptr) {
        shared.eval();
        timings->shared_expert_ms = elapsed_ms(shared_started);
    }
    const auto merge_started = Clock::now();
    MlxArray output = MlxArray::add(routed, shared);
    if (timings != nullptr) {
        output.eval();
        timings->merge_ms = elapsed_ms(merge_started);
    }
    return output;
}

} // namespace qwen38
