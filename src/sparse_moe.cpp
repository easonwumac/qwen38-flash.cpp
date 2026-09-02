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
#include <mutex>
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

std::size_t effective_experts_per_token(
    const std::string_view prefix,
    const std::size_t configured) {
    const char* value = std::getenv("QWEN38_TARGET_TOPK");
    if (value == nullptr || value[0] == '\0' ||
        !prefix.starts_with("language_model.model.layers.")) {
        return configured;
    }
    std::size_t parsed_value = 0;
    const std::string_view text(value);
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), parsed_value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        parsed_value < 6 || parsed_value > configured) {
        throw std::runtime_error("QWEN38_TARGET_TOPK must be between 6 and configured top-k");
    }
    return parsed_value;
}

int compact_qmeta_requested_bits() {
    const char* value = std::getenv("QWEN38_COMPACT_QMETA");
    if (value == nullptr || std::string_view(value) == "0") return 0;
    const std::string_view mode(value);
    if (mode == "1" || mode == "16" || mode == "lossless16") return 16;
    if (mode == "13" || mode == "lossless13") return 13;
    if (mode == "9" || mode == "lossy9") return 9;
    throw std::runtime_error(
        "QWEN38_COMPACT_QMETA must be 0, lossless13, lossless16, or lossy9");
}

bool qmeta_prefill_cache_enabled(const std::size_t layer) {
    const char* value = std::getenv("QWEN38_QMETA_PREFILL_CACHE");
    if (value == nullptr || std::string_view(value) != "1") return false;
    const char* layers_value = std::getenv("QWEN38_QMETA_PREFILL_CACHE_LAYERS");
    if (layers_value == nullptr) return true;
    std::size_t layers = 0;
    const std::string_view text(layers_value);
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), layers);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || layers > 48) {
        throw std::runtime_error("QWEN38_QMETA_PREFILL_CACHE_LAYERS must be between 0 and 48");
    }
    return layer < layers;
}

bool qmeta_prefill_defer_reduce_enabled() {
    const char* value = std::getenv("QWEN38_QMETA_PREFILL_DEFER_REDUCE");
    return value == nullptr || std::string_view(value) == "1";
}

bool qmeta_prefill_defer_temporary_enabled() {
    const char* value = std::getenv("QWEN38_QMETA_PREFILL_DEFER_TEMPORARY");
    return value != nullptr && std::string_view(value) == "1";
}

bool qmeta_aligned16_enabled() {
    const char* value = std::getenv("QWEN38_QMETA_ALIGNED16");
    return value == nullptr || std::string_view(value) != "0";
}

std::shared_ptr<MlxMetalKernel> qmeta_gate_up_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{
            "x", "gate_weight", "gate_tags", "gate_dictionary",
            "up_weight", "up_tags", "up_dictionary", "experts"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_compact_qmeta_gate_up", inputs, "hidden",
            moe_metal::qmeta_gate_up, moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> qmeta_down_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{
            "x", "weight", "tags", "dictionary", "experts", "route_weights"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_compact_qmeta_down_reduce", inputs, "output",
            moe_metal::qmeta_down_reduce, moe_metal::header);
    }();
    return kernel;
}

std::shared_ptr<MlxMetalKernel> qmeta_decode_kernel() {
    static const std::shared_ptr<MlxMetalKernel> kernel = [] {
        const char* inputs[]{"tags", "dictionary"};
        const char* outputs[]{"scales", "biases"};
        return std::make_shared<MlxMetalKernel>(
            "qwen38_compact_qmeta_decode", inputs, outputs,
            moe_metal::qmeta_decode, moe_metal::header);
    }();
    return kernel;
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
    : layer_index_(layer_index(prefix)),
      expert_count_(expert_count),
      experts_per_token_(effective_experts_per_token(prefix, experts_per_token)),
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
    compact_qmeta_ =
        expert_gate_.qmeta.present() && expert_up_.qmeta.present() &&
        expert_down_.qmeta.present();
    if (experts_per_token_ != experts_per_token && !compact_qmeta_) {
        throw std::runtime_error("QWEN38_TARGET_TOPK currently requires compact qmeta");
    }
    if (compact_qmeta_requested_bits() != 0 &&
        std::string_view(prefix).starts_with("language_model.model.layers.") &&
        !compact_qmeta_) {
        throw std::runtime_error(
            "QWEN38_COMPACT_QMETA requires a complete sidecar for the requested mode");
    }
    if (resident_layer_enabled(layer_index_)) {
        make_resident(expert_gate_);
        make_resident(expert_up_);
        make_resident(expert_down_);
    }
    const char* fused = std::getenv("QWEN38_FUSED_MOE");
    const char* q8_exact = std::getenv("QWEN38_Q8_EXACT_MOE");
    if (fused != nullptr && std::string_view(fused) == "1" &&
        expert_count_ >= experts_per_token_ &&
        ((compact_qmeta_ && experts_per_token_ >= 6 && experts_per_token_ <= 10) ||
         experts_per_token_ == 10) &&
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
    CompactQmeta qmeta;
    const int requested_qmeta_bits = compact_qmeta_requested_bits();
    const std::string suffix = ".qmeta" + std::to_string(requested_qmeta_bits);
    const std::string tags_name = base + suffix + "_tags";
    const std::string dictionary_name = base + suffix + "_dict";
    const bool has_tags = tensors.manifest().has_tensor(tags_name);
    const bool has_dictionary = tensors.manifest().has_tensor(dictionary_name);
    if (has_tags != has_dictionary) {
        throw std::runtime_error("incomplete compact qmeta pair for " + base);
    }
    if (requested_qmeta_bits != 0 && has_tags) {
        qmeta.tags = tensors.tensor(tags_name);
        qmeta.dictionary = tensors.tensor(dictionary_name);
        const std::vector<int> weight_shape = weight.shape();
        const std::vector<int> scale_shape = scales.shape();
        const std::vector<int> tag_shape = qmeta.tags.shape();
        const std::vector<int> dictionary_shape = qmeta.dictionary.shape();
        if (bits != 4 || weight_shape.size() != 3 || scale_shape.size() != 3 ||
            tag_shape.size() != 3 || dictionary_shape.size() != 1 ||
            dictionary_shape[0] <= 0 ||
            dictionary_shape[0] > (1 << requested_qmeta_bits) ||
            qmeta.tags.dtype() != MLX_UINT8 || qmeta.dictionary.dtype() != MLX_UINT32 ||
            tag_shape[0] != scale_shape[0] || tag_shape[1] != scale_shape[1]) {
            throw std::runtime_error("invalid compact qmeta geometry for " + base);
        }
        qmeta.bits = requested_qmeta_bits;
        qmeta.groups = scale_shape[2];
        qmeta.row_bytes = (qmeta.groups * qmeta.bits + 7) / 8;
        if (tag_shape[2] != qmeta.row_bytes) {
            throw std::runtime_error("compact qmeta row width mismatch for " + base);
        }
    }
    return {
        .weight = std::move(weight),
        .scales = std::move(scales),
        .biases = tensors.tensor(base + ".biases"),
        .qmeta = std::move(qmeta),
        .bits = bits,
    };
}

void SparseMoe::make_resident(QuantizedProjection& projection) {
    projection.weight.lock_pages();
    if (projection.qmeta.present()) {
        projection.qmeta.tags.lock_pages();
        projection.qmeta.dictionary.lock_pages();
    } else {
        projection.scales.lock_pages();
        projection.biases.lock_pages();
    }
}

SparseMoe::DecodedQmeta SparseMoe::decode_qmeta(
    const QuantizedProjection& projection) {
    if (!projection.qmeta.present()) {
        throw std::runtime_error("compact qmeta decode requested without metadata");
    }
    const std::vector<int> weight_shape = projection.weight.shape();
    if (weight_shape.size() != 3) {
        throw std::runtime_error("compact qmeta projection must be rank three");
    }
    const std::vector<int> output_shape{
        weight_shape[0], weight_shape[1], projection.qmeta.groups};
    const std::array<MlxMetalOutputSpec, 2> outputs{{
        {.shape = output_shape, .dtype = MLX_BFLOAT16},
        {.shape = output_shape, .dtype = MLX_BFLOAT16},
    }};
    const std::array<int, 3> grid{
        weight_shape[0] * weight_shape[1] * projection.qmeta.groups, 1, 1};
    const std::array<int, 3> threadgroup{256, 1, 1};
    const std::array<MlxMetalDtypeTemplate, 1> dtype_templates{{
        {.name = "T", .value = MLX_BFLOAT16},
    }};
    const std::array<MlxMetalIntTemplate, 3> int_templates{{
        {.name = "QBITS", .value = projection.qmeta.bits},
        {.name = "GROUPS", .value = projection.qmeta.groups},
        {.name = "ROW_BYTES", .value = projection.qmeta.row_bytes},
    }};
    const MlxArray* inputs[]{&projection.qmeta.tags, &projection.qmeta.dictionary};
    std::vector<MlxArray> decoded = qmeta_decode_kernel()->apply(
        inputs, outputs, grid, threadgroup, dtype_templates, int_templates);
    return {
        .scales = std::move(decoded[0]),
        .biases = std::move(decoded[1]),
    };
}

bool SparseMoe::clear_prefill_qmeta_cache() const {
    bool cleared = false;
    const auto clear_projection = [&cleared](const QuantizedProjection& projection) {
        if (!projection.qmeta_cached) return;
        projection.cached_qmeta = {};
        projection.qmeta_cached = false;
        cleared = true;
    };
    clear_projection(expert_gate_);
    clear_projection(expert_up_);
    clear_projection(expert_down_);
    return cleared;
}

void SparseMoe::set_prefill_qmeta_cache_allowed(const bool allowed) const noexcept {
    prefill_qmeta_cache_allowed_ = allowed;
}

MlxArray SparseMoe::forward_compact_routed(
    const MlxArray& input,
    const MlxArray& experts,
    const MlxArray& weights) const {
    if (!compact_qmeta_) {
        throw std::runtime_error("compact routed MoE requested without qmeta");
    }
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 ||
        input_shape[2] != 2560 || input_shape[1] < 1 || input_shape[1] > 8 ||
        experts_per_token_ < 6 || experts_per_token_ > 10) {
        throw std::runtime_error(
            "compact routed MoE requires [1,S,2560], S=1..8, top-k=6..10");
    }
    const int rows = input_shape[1];
    const int topk = checked_int(experts_per_token_, "compact top-k");
    const int slots = rows * topk;
    MlxArray flat_experts = experts.reshape(std::vector<int>{slots}).astype(MLX_UINT32);
    MlxArray flat_weights = weights.reshape(std::vector<int>{slots});

    const MlxArray* gate_inputs[]{
        &input,
        &expert_gate_.weight, &expert_gate_.qmeta.tags, &expert_gate_.qmeta.dictionary,
        &expert_up_.weight, &expert_up_.qmeta.tags, &expert_up_.qmeta.dictionary,
        &flat_experts};
    const std::array<MlxMetalOutputSpec, 1> gate_outputs{{
        {.shape = {slots, 640}, .dtype = input.dtype()},
    }};
    const std::array<int, 3> gate_grid{rows * topk * 20 * 1024, 1, 1};
    const std::array<int, 3> gate_threadgroup{1024, 1, 1};
    const std::array<MlxMetalDtypeTemplate, 1> dtype_templates{{
        {.name = "T", .value = input.dtype()},
    }};
    const std::array<MlxMetalIntTemplate, 4> gate_int_templates{{
        {.name = "QGBITS", .value = expert_gate_.qmeta.bits},
        {.name = "QUBITS", .value = expert_up_.qmeta.bits},
        {.name = "ALIGNED16", .value = qmeta_aligned16_enabled() ? 1 : 0},
        {.name = "SLOTS", .value = topk},
    }};
    std::vector<MlxArray> gate_result = qmeta_gate_up_kernel()->apply(
        gate_inputs,
        gate_outputs,
        gate_grid,
        gate_threadgroup,
        dtype_templates,
        gate_int_templates);
    MlxArray hidden = std::move(gate_result[0]);

    const MlxArray* down_inputs[]{
        &hidden, &expert_down_.weight, &expert_down_.qmeta.tags,
        &expert_down_.qmeta.dictionary, &flat_experts, &flat_weights};
    const std::array<MlxMetalOutputSpec, 1> down_outputs{{
        {.shape = {rows, 2560}, .dtype = input.dtype()},
    }};
    const std::array<int, 3> down_grid{rows * 320 * 64, 1, 1};
    const std::array<int, 3> down_threadgroup{64, 1, 1};
    const std::array<MlxMetalIntTemplate, 3> down_int_templates{{
        {.name = "QBITS", .value = expert_down_.qmeta.bits},
        {.name = "ALIGNED16", .value = qmeta_aligned16_enabled() ? 1 : 0},
        {.name = "SLOTS", .value = topk},
    }};
    std::vector<MlxArray> down_result = qmeta_down_kernel()->apply(
        down_inputs,
        down_outputs,
        down_grid,
        down_threadgroup,
        dtype_templates,
        down_int_templates);
    static std::once_flag announced;
    const int qmeta_bits = expert_down_.qmeta.bits;
    std::call_once(announced, [rows, qmeta_bits, topk] {
        std::cerr << "[qmeta] compact " << qmeta_bits
                  << "-bit routed MoE engaged: rows=" << rows
                  << " topk=" << topk << '\n';
    });
    return std::move(down_result[0]).reshape(std::vector<int>{1, rows, 2560});
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
        if (compact_qmeta_) {
            expert_sum = forward_compact_routed(input, experts, weights);
        } else {
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
            MlxArray down_weights =
                fused_q8_exact_ ? weights.astype(input.dtype()) : weights.share();
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
        }
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
    const char* device_router = std::getenv("QWEN38_DEVICE_ROUTER");
    const char* grouped = std::getenv("QWEN38_GROUPED_PREFILL");
    if (!compact_qmeta_ && (!fused_gate_up_ || !fused_down_) &&
        device_router != nullptr &&
        std::string_view(device_router) == "1" && grouped != nullptr &&
        std::string_view(grouped) == "1") {
        return forward_verify(input);
    }
    return MlxArray::add(forward_experts_decode(input), forward_shared(input));
}

MlxArray SparseMoe::forward_verify(const MlxArray& input) const {
    return forward_verify_impl(input, nullptr);
}

MlxArray SparseMoe::forward_verify_profiled(
    const MlxArray& input,
    MoeVerifyTimings& timings) const {
    timings = {};
    return forward_verify_impl(input, &timings);
}

MlxArray SparseMoe::forward_verify_impl(
    const MlxArray& input,
    MoeVerifyTimings* timings) const {
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ms = [](const Clock::time_point started) {
        return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    };
    const std::vector<int> shape = input.shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > 1024) {
        throw std::runtime_error("MoE batch requires shape [1,S,hidden], S=1..1024");
    }
    const char* device_router = std::getenv("QWEN38_DEVICE_ROUTER");
    const char* verify_grouped = std::getenv("QWEN38_GROUPED_PREFILL");
    if (timings == nullptr && !compact_qmeta_ &&
        (!fused_gate_up_ || !fused_down_) &&
        device_router != nullptr && std::string_view(device_router) == "1" &&
        verify_grouped != nullptr && std::string_view(verify_grouped) == "1") {
        return forward_prefill_impl(input, nullptr);
    }
    if (fused_gate_up_ && fused_down_ && device_router != nullptr &&
        std::string_view(device_router) == "1") {
        const int rows = shape[1];
        const auto routing_started = Clock::now();
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
        const char* profile_overlap = std::getenv("QWEN38_PROFILE_VERIFY_EXPERT_OVERLAP");
        const std::string_view overlap_mode = profile_overlap == nullptr
            ? std::string_view{}
            : std::string_view(profile_overlap);
        if (overlap_mode == "all" || (layer_index_ == 0 && overlap_mode == "1")) {
            const std::vector<float> routed_ids =
                experts.astype(MLX_FLOAT32).to_float32();
            const std::vector<float> routed_weights = weights.to_float32();
            std::vector<bool> seen(expert_count_, false);
            std::size_t unique = 0;
            for (const float value : routed_ids) {
                const std::size_t expert = static_cast<std::size_t>(value);
                if (expert < seen.size() && !seen[expert]) {
                    seen[expert] = true;
                    ++unique;
                }
            }
            double bottom_one_sum = 0.0;
            double bottom_two_sum = 0.0;
            float bottom_two_max = 0.0F;
            for (int row = 0; row < rows; ++row) {
                const auto begin = static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(row) * experts_per_token_);
                const auto end = begin + static_cast<std::ptrdiff_t>(experts_per_token_);
                std::vector<float> row_weights(
                    routed_weights.begin() + begin,
                    routed_weights.begin() + end);
                std::sort(row_weights.begin(), row_weights.end());
                const float bottom_one = row_weights.front();
                const float bottom_two = bottom_one + row_weights[1];
                bottom_one_sum += bottom_one;
                bottom_two_sum += bottom_two;
                bottom_two_max = std::max(bottom_two_max, bottom_two);
            }
            std::clog << "qwen38-verify-expert-overlap: layer=" << layer_index_
                      << " rows=" << rows
                      << " selected=" << routed_ids.size()
                      << " unique=" << unique
                      << " bottom1_mean=" << bottom_one_sum / rows
                      << " bottom2_mean=" << bottom_two_sum / rows
                      << " bottom2_max=" << bottom_two_max << '\n';
        }
        if (timings != nullptr) {
            const std::array<const MlxArray*, 2> routing_outputs{&experts, &weights};
            MlxArray::eval_all(routing_outputs);
            timings->routing_ms = elapsed_ms(routing_started);
        }
        if (compact_qmeta_ && rows <= 8) {
            const auto compact_started = Clock::now();
            MlxArray routed = forward_compact_routed(input, experts, weights);
            if (timings != nullptr) {
                routed.eval();
                // Compact gate/up and down are one lazy chain. Attribute the
                // combined routed cost here instead of inserting a diagnostic
                // barrier that would alter production scheduling.
                timings->down_ms = elapsed_ms(compact_started);
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
        const auto gate_up_started = Clock::now();
        MlxArray hidden = (fused_q8_exact_
            ? fused_q8_verify_gate_up_kernel()
            : fused_verify_gate_up_kernel())->apply(
                fused_q8_exact_
                    ? std::span<const MlxArray* const>(q8_gate_inputs)
                    : std::span<const MlxArray* const>(q4_gate_inputs),
                hidden_shape, input.dtype(), gate_grid, gate_threadgroup);
        if (timings != nullptr) {
            hidden.eval();
            timings->gate_up_ms = elapsed_ms(gate_up_started);
        }
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
        const auto down_started = Clock::now();
        MlxArray routed = (fused_q8_exact_
            ? fused_q8_verify_down_kernel()
            : fused_verify_down_kernel())->apply(
            down_inputs, output_shape, input.dtype(), down_grid, down_threadgroup);
        if (timings != nullptr) {
            routed.eval();
            timings->down_ms = elapsed_ms(down_started);
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
    const char* profile = std::getenv("QWEN38_PROFILE_MOE_PREFILL");
    if (layer_index_ == 0 && profile != nullptr && std::string_view(profile) == "1") {
        MoePrefillTimings timings;
        MlxArray output = forward_prefill_impl(input, &timings);
        std::clog << "qwen38-moe-prefill-profile: rows=" << input.shape()[1]
                  << " routing_ms=" << timings.routing_ms
                  << " gate_qmm_ms=" << timings.gate_qmm_ms
                  << " up_qmm_ms=" << timings.up_qmm_ms
                  << " swiglu_ms=" << timings.swiglu_ms
                  << " down_qmm_ms=" << timings.down_qmm_ms
                  << " route_reduce_ms=" << timings.route_reduce_ms
                  << " shared_ms=" << timings.shared_expert_ms
                  << " merge_ms=" << timings.merge_ms << '\n';
        return output;
    }
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
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > 1024) {
        throw std::runtime_error("MoE prefill requires shape [1,S,hidden], S=1..1024");
    }
    const char* grouped = std::getenv("QWEN38_GROUPED_PREFILL");
    const bool mixed_quantization = !fused_gate_up_ || !fused_down_;
    if (compact_qmeta_ && shape[1] <= 8) {
        return forward_verify(input);
    }
    if (!compact_qmeta_ &&
        ((!mixed_quantization && shape[1] < 16) || grouped == nullptr ||
         std::string_view(grouped) != "1")) {
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

    DecodedQmeta decoded_gate;
    DecodedQmeta decoded_up;
    DecodedQmeta decoded_down;
    const MlxArray* gate_scales = &expert_gate_.scales;
    const MlxArray* gate_biases = &expert_gate_.biases;
    const MlxArray* up_scales = &expert_up_.scales;
    const MlxArray* up_biases = &expert_up_.biases;
    const MlxArray* down_scales = &expert_down_.scales;
    const MlxArray* down_biases = &expert_down_.biases;
    const bool cache_qmeta = compact_qmeta_ && prefill_qmeta_cache_allowed_ &&
        qmeta_prefill_cache_enabled(layer_index_);
    if (compact_qmeta_) {
        if (cache_qmeta) {
            const auto ensure_cached = [](const QuantizedProjection& projection) {
                if (projection.qmeta_cached) return;
                projection.cached_qmeta = decode_qmeta(projection);
                projection.qmeta_cached = true;
            };
            ensure_cached(expert_gate_);
            ensure_cached(expert_up_);
            ensure_cached(expert_down_);
            gate_scales = &expert_gate_.cached_qmeta.scales;
            gate_biases = &expert_gate_.cached_qmeta.biases;
            up_scales = &expert_up_.cached_qmeta.scales;
            up_biases = &expert_up_.cached_qmeta.biases;
            down_scales = &expert_down_.cached_qmeta.scales;
            down_biases = &expert_down_.cached_qmeta.biases;
        } else {
            decoded_gate = decode_qmeta(expert_gate_);
            decoded_up = decode_qmeta(expert_up_);
            decoded_down = decode_qmeta(expert_down_);
            gate_scales = &decoded_gate.scales;
            gate_biases = &decoded_gate.biases;
            up_scales = &decoded_up.scales;
            up_biases = &decoded_up.biases;
            down_scales = &decoded_down.scales;
            down_biases = &decoded_down.biases;
        }
    }

    const auto gate_up_started = Clock::now();
    const auto gate_started = Clock::now();
    MlxArray raw_gate = MlxArray::gather_quantized_matmul(
        gathered_input,
        expert_gate_.weight,
        *gate_scales,
        *gate_biases,
        no_indices,
        sorted_experts,
        group_size_,
        expert_gate_.bits,
        true);
    if (timings != nullptr) {
        raw_gate.eval();
        timings->gate_qmm_ms = elapsed_ms(gate_started);
    }
    const auto up_started = Clock::now();
    MlxArray up = MlxArray::gather_quantized_matmul(
        gathered_input,
        expert_up_.weight,
        *up_scales,
        *up_biases,
        no_indices,
        sorted_experts,
        group_size_,
        expert_up_.bits,
        true);
    if (timings != nullptr) {
        up.eval();
        timings->up_qmm_ms = elapsed_ms(up_started);
    }
    const auto swiglu_started = Clock::now();
    MlxArray gate = raw_gate.silu();
    MlxArray expert_hidden = MlxArray::multiply(gate, up);
    if (timings != nullptr) {
        expert_hidden.eval();
        timings->swiglu_ms = elapsed_ms(swiglu_started);
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
        *down_scales,
        *down_biases,
        no_indices,
        sorted_experts,
        group_size_,
        expert_down_.bits,
        true).reshape(std::vector<int>{slots, hidden_size});
    if (timings != nullptr) {
        down.eval();
        timings->down_qmm_ms = elapsed_ms(down_started);
    }
    const auto route_reduce_started = Clock::now();
    MlxArray unsorted = MlxArray::take_axis(down, inverse_order, 0).reshape(
        std::vector<int>{1, rows, top_k, hidden_size});
    MlxArray weighted = MlxArray::multiply(
        unsorted, weights.reshape(std::vector<int>{1, rows, top_k, 1}));
    MlxArray routed = weighted.sum_axis(2);
    const bool defer_cached_qmeta_reduce =
        cache_qmeta && qmeta_prefill_defer_reduce_enabled();
    const bool defer_temporary_qmeta_reduce =
        compact_qmeta_ && !cache_qmeta && qmeta_prefill_defer_temporary_enabled();
    if (defer_cached_qmeta_reduce) {
        static std::once_flag announced;
        std::call_once(announced, [] {
            std::cerr << "[qmeta] cached prefill reduction uses grouped layer barriers\n";
        });
    }
    if (defer_temporary_qmeta_reduce) {
        static std::once_flag announced;
        std::call_once(announced, [] {
            std::cerr << "[qmeta] temporary prefill metadata uses grouped layer barriers\n";
        });
    }
    if (compact_qmeta_ && !defer_cached_qmeta_reduce && !defer_temporary_qmeta_reduce) {
        // Diagnostic rollback: drain each layer immediately. Production
        // memory mode lets the lazy graph retain temporary decoded inputs only
        // until the model's bounded layer-group barrier.
        routed.eval();
    }
    if (timings != nullptr) {
        routed.eval();
        timings->route_reduce_ms = elapsed_ms(route_reduce_started);
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
