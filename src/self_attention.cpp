#include "qwen38/self_attention.hpp"

#include "qwen38/quantization_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen38 {
namespace {

int dimension(const std::size_t value, const char* name) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid attention dimension: ") + name);
    }
    return static_cast<int>(value);
}

int coordinate(const std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("invalid attention coordinate: ") + name);
    }
    return static_cast<int>(value);
}

MlxArray offset_norm_weight(MlxArray raw, const std::size_t width) {
    const std::vector<float> ones_data(width, 1.0F);
    const std::vector<int> shape{dimension(width, "norm width")};
    MlxArray ones = MlxArray::from_float32(ones_data, shape).astype(raw.dtype());
    return MlxArray::add(raw, ones);
}

MlxArray effective_norm_weight(
    MlxArray raw,
    const std::size_t width,
    const bool has_offset) {
    if (!has_offset) return raw;
    return offset_norm_weight(std::move(raw), width);
}

MlxArray scalar(const float value, const mlx_dtype dtype) {
    const std::vector<float> values{value};
    const std::vector<int> shape{};
    return MlxArray::from_float32(values, shape).astype(dtype);
}

MlxArray integer_scalar(const std::int32_t value) {
    const std::vector<std::int32_t> values{value};
    const std::vector<int> shape{};
    return MlxArray::from_int32(values, shape);
}

bool use_sdpa_decode(const std::size_t token_count) {
    const char* configured = std::getenv("QWEN38_SDPA_DECODE");
    const bool enabled = configured == nullptr
        ? token_count >= 512
        : std::string_view(configured) == "1";
    if (enabled) {
        static std::once_flag announced;
        std::call_once(announced, [token_count] {
            std::clog << "qwen38: grouped-query SDPA decode engaged at "
                      << token_count << " cached attention tokens\n";
        });
    }
    return enabled;
}

MlxArray slice_sequence_row(const MlxArray& batch, const std::size_t row) {
    const std::vector<int> shape = batch.shape();
    if (shape.size() < 2 || row >= static_cast<std::size_t>(shape[1])) {
        throw std::runtime_error("attention verifier row is out of range");
    }
    std::vector<int> start(shape.size(), 0);
    std::vector<int> stop = shape;
    std::vector<int> strides(shape.size(), 1);
    start[1] = static_cast<int>(row);
    stop[1] = static_cast<int>(row + 1);
    return batch.slice(start, stop, strides);
}

MlxArray concatenate_sequence_rows(const std::vector<MlxArray>& rows) {
    if (rows.empty()) throw std::runtime_error("attention verifier has no outputs");
    MlxArray result = rows.front().share();
    for (std::size_t row = 1; row < rows.size(); ++row) {
        result = MlxArray::concatenate(result, rows[row], 1);
    }
    return result;
}

struct PrefillRopeTableCache {
    std::mutex mutex;
    std::size_t position{0};
    int rows{0};
    int rotary{0};
    double theta{0.0};
    std::size_t step{1};
    mlx_dtype dtype{MLX_FLOAT32};
    bool valid{false};
    MlxArray cosine;
    MlxArray sine;
};

std::pair<MlxArray, MlxArray> prefill_rope_tables(
    const std::size_t position,
    const std::size_t step,
    const int rows,
    const int rotary,
    const double theta,
    const mlx_dtype dtype) {
    static PrefillRopeTableCache cache;
    std::scoped_lock lock(cache.mutex);
    if (!cache.valid || cache.position != position || cache.rows != rows ||
        cache.rotary != rotary || cache.theta != theta || cache.step != step ||
        cache.dtype != dtype) {
        const int half = rotary / 2;
        std::vector<float> cosine(static_cast<std::size_t>(rows * rotary));
        std::vector<float> sine(static_cast<std::size_t>(rows * rotary));
        for (int row = 0; row < rows; ++row) {
            for (int index = 0; index < half; ++index) {
                const double frequency = std::pow(
                    theta, -2.0 * static_cast<double>(index) /
                        static_cast<double>(rotary));
                const double angle = static_cast<double>(
                    position + static_cast<std::size_t>(row) * step) * frequency;
                const float c = static_cast<float>(std::cos(angle));
                const float s = static_cast<float>(std::sin(angle));
                const std::size_t base = static_cast<std::size_t>(row * rotary);
                cosine[base + static_cast<std::size_t>(index)] = c;
                cosine[base + static_cast<std::size_t>(index + half)] = c;
                sine[base + static_cast<std::size_t>(index)] = s;
                sine[base + static_cast<std::size_t>(index + half)] = s;
            }
        }
        const std::vector<int> shape{1, 1, rows, rotary};
        cache.cosine = MlxArray::from_float32(cosine, shape).astype(dtype);
        cache.sine = MlxArray::from_float32(sine, shape).astype(dtype);
        cache.position = position;
        cache.rows = rows;
        cache.rotary = rotary;
        cache.theta = theta;
        cache.step = step;
        cache.dtype = dtype;
        cache.valid = true;
    }
    return {cache.cosine.share(), cache.sine.share()};
}

} // namespace

SelfAttention::SelfAttention(
    MlxTensorStore& tensors,
    const std::string_view prefix,
    const ModelConfig& config)
    : attention_heads_(config.attention_head_count),
      key_value_heads_(config.key_value_head_count),
      head_dimension_(config.head_dimension),
      rotary_dimension_(static_cast<std::size_t>(
          static_cast<double>(config.head_dimension) * config.partial_rotary_factor)),
      indexer_head_count_(config.indexer_head_count),
      indexer_head_dimension_(config.indexer_head_dimension),
      indexer_budget_(config.indexer_budget),
      indexer_compress_ratio_(config.indexer_compress_ratio),
      group_size_(dimension(config.quantization_group_size, "quantization group size")),
      epsilon_(static_cast<float>(config.rms_norm_epsilon)),
      rope_theta_(config.rope_theta),
      query_projection_(load_projection(
          tensors, std::string(prefix) + ".q_proj", config.quantization_group_size)),
      key_projection_(load_projection(
          tensors, std::string(prefix) + ".k_proj", config.quantization_group_size)),
      value_projection_(load_projection(
          tensors, std::string(prefix) + ".v_proj", config.quantization_group_size)),
      output_projection_(load_projection(
          tensors, std::string(prefix) + ".o_proj", config.quantization_group_size)),
      indexer_projection_(load_projection(
          tensors,
          std::string(prefix) + ".indexer.index_qk_proj",
          config.quantization_group_size)),
      query_norm_weight_(effective_norm_weight(
          tensors.tensor(std::string(prefix) + ".q_norm.weight"),
          config.head_dimension,
          config.attention_norm_has_offset)),
      key_norm_weight_(effective_norm_weight(
          tensors.tensor(std::string(prefix) + ".k_norm.weight"),
          config.head_dimension,
          config.attention_norm_has_offset)),
      indexer_query_norm_weight_(effective_norm_weight(
          tensors.tensor(std::string(prefix) + ".indexer.q_layernorm.weight"),
          config.indexer_head_dimension,
          config.indexer_norm_has_offset)),
      indexer_key_norm_weight_(effective_norm_weight(
          tensors.tensor(std::string(prefix) + ".indexer.k_layernorm.weight"),
          config.indexer_head_dimension,
          config.indexer_norm_has_offset)) {
    static_cast<void>(dimension(config.quantization_bits, "quantization bits"));
    if (attention_heads_ % key_value_heads_ != 0 || rotary_dimension_ == 0 ||
        rotary_dimension_ % 2 != 0 || rotary_dimension_ > head_dimension_ ||
        rotary_dimension_ > indexer_head_dimension_ || indexer_head_count_ == 0 ||
        indexer_budget_ % indexer_compress_ratio_ != 0) {
        throw std::runtime_error("unsupported attention head/rotary configuration");
    }
}

SelfAttention::QuantizedProjection SelfAttention::load_projection(
    MlxTensorStore& tensors,
    const std::string_view name,
    const std::size_t group_size) {
    const std::string base(name);
    MlxArray weight = tensors.tensor(base + ".weight");
    MlxArray scales = tensors.tensor(base + ".scales");
    const int bits = infer_affine_quantization_bits(
        weight.shape(), scales.shape(), group_size, "attention");
    return {
        .weight = std::move(weight),
        .scales = std::move(scales),
        .biases = tensors.tensor(base + ".biases"),
        .bits = bits,
    };
}

MlxArray SelfAttention::project(
    const MlxArray& input,
    const QuantizedProjection& projection) const {
    return MlxArray::quantized_matmul(
        input, projection.weight, projection.scales, projection.biases,
        group_size_, projection.bits);
}

MlxArray SelfAttention::apply_rope(const MlxArray& input, const std::size_t position) const {
    const auto shape = input.shape();
    if (shape.size() != 4 || shape[2] != 1 ||
        shape[3] != dimension(head_dimension_, "head dimension")) {
        throw std::runtime_error("RoPE input shape mismatch");
    }
    const int rotary = dimension(rotary_dimension_, "rotary dimension");
    const int half = rotary / 2;
    const std::size_t rotary_size = static_cast<std::size_t>(rotary);
    const std::size_t half_size = static_cast<std::size_t>(half);
    std::vector<float> cosine(rotary_size);
    std::vector<float> sine(rotary_size);
    for (std::size_t index = 0; index < half_size; ++index) {
        const double frequency = std::pow(
            rope_theta_, -2.0 * static_cast<double>(index) / static_cast<double>(rotary));
        const double angle = static_cast<double>(position) * frequency;
        cosine[index] = cosine[index + half_size] = static_cast<float>(std::cos(angle));
        sine[index] = sine[index + half_size] = static_cast<float>(std::sin(angle));
    }
    const std::vector<int> angle_shape{1, 1, 1, rotary};
    MlxArray cos_array = MlxArray::from_float32(cosine, angle_shape).astype(input.dtype());
    MlxArray sin_array = MlxArray::from_float32(sine, angle_shape).astype(input.dtype());
    const std::vector<int> strides{1, 1, 1, 1};
    const std::vector<int> rope_start{0, 0, 0, 0};
    const std::vector<int> rope_stop{shape[0], shape[1], shape[2], rotary};
    MlxArray rope = input.slice(rope_start, rope_stop, strides);
    const std::vector<int> first_stop{shape[0], shape[1], shape[2], half};
    const std::vector<int> second_start{0, 0, 0, half};
    MlxArray first = rope.slice(rope_start, first_stop, strides);
    MlxArray second = rope.slice(second_start, rope_stop, strides);
    MlxArray rotated = MlxArray::concatenate(second.negative(), first, 3);
    MlxArray transformed = MlxArray::add(
        MlxArray::multiply(rope, cos_array), MlxArray::multiply(rotated, sin_array));
    if (rotary_dimension_ == head_dimension_) return transformed;
    const std::vector<int> rest_start{0, 0, 0, rotary};
    const std::vector<int> rest_stop{shape[0], shape[1], shape[2], shape[3]};
    return MlxArray::concatenate(
        transformed, input.slice(rest_start, rest_stop, strides), 3);
}

MlxArray SelfAttention::apply_rope_prefill(
    const MlxArray& input,
    const std::size_t position) const {
    return apply_rope_rows(input, position, 1, head_dimension_);
}

MlxArray SelfAttention::apply_rope_rows(
    const MlxArray& input,
    const std::size_t position,
    const std::size_t step,
    const std::size_t vector_dimension) const {
    const auto shape = input.shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[2] < 1 ||
        shape[3] != dimension(vector_dimension, "RoPE vector dimension") || step == 0) {
        throw std::runtime_error("prefill RoPE input shape mismatch");
    }
    const int rows = shape[2];
    const int rotary = dimension(rotary_dimension_, "rotary dimension");
    const int half = rotary / 2;
    auto [cos_array, sin_array] = prefill_rope_tables(
        position, step, rows, rotary, rope_theta_, input.dtype());
    const std::vector<int> strides{1, 1, 1, 1};
    const std::vector<int> rope_start{0, 0, 0, 0};
    const std::vector<int> rope_stop{shape[0], shape[1], rows, rotary};
    MlxArray rope = input.slice(rope_start, rope_stop, strides);
    const std::vector<int> first_stop{shape[0], shape[1], rows, half};
    const std::vector<int> second_start{0, 0, 0, half};
    MlxArray first = rope.slice(rope_start, first_stop, strides);
    MlxArray second = rope.slice(second_start, rope_stop, strides);
    MlxArray rotated = MlxArray::concatenate(second.negative(), first, 3);
    MlxArray transformed = MlxArray::add(
        MlxArray::multiply(rope, cos_array), MlxArray::multiply(rotated, sin_array));
    if (rotary_dimension_ == vector_dimension) return transformed;
    const std::vector<int> rest_start{0, 0, 0, rotary};
    const std::vector<int> rest_stop{shape[0], shape[1], rows, shape[3]};
    return MlxArray::concatenate(
        transformed, input.slice(rest_start, rest_stop, strides), 3);
}

MlxArray SelfAttention::update_qsa_and_build_mask(
    const MlxArray& input,
    SelfAttentionState& state) const {
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] < 1) {
        throw std::runtime_error("QSA requires a single [1,S,hidden] sequence");
    }
    const int rows = input_shape[1];
    const int index_heads = dimension(indexer_head_count_, "indexer heads");
    const int index_dimension = dimension(indexer_head_dimension_, "indexer dimension");
    const int ratio = dimension(indexer_compress_ratio_, "indexer ratio");
    const int budget = dimension(indexer_budget_, "indexer budget");
    const int block_topk = budget / ratio;
    const std::vector<int> strides3{1, 1, 1};

    MlxArray qk = project(input, indexer_projection_);
    MlxArray raw_keys = qk.slice(
        std::vector<int>{0, 0, index_heads * index_dimension},
        std::vector<int>{1, rows, (index_heads + 1) * index_dimension},
        strides3);
    if (state.token_count == 0) {
        state.qsa_raw_keys = raw_keys.share();
    } else {
        const std::vector<int> raw_shape = state.qsa_raw_keys.shape();
        if (raw_shape != std::vector<int>({1, dimension(state.token_count, "QSA history"),
                                          index_dimension})) {
            throw std::runtime_error("QSA raw-key state is not aligned with attention KV");
        }
        state.qsa_raw_keys = MlxArray::concatenate(state.qsa_raw_keys, raw_keys, 1);
    }

    const std::size_t total = state.token_count + static_cast<std::size_t>(rows);
    const std::size_t block_count = total / indexer_compress_ratio_;
    if (block_count <= static_cast<std::size_t>(block_topk)) return {};

    if (state.qsa_pooled_count > block_count) {
        throw std::runtime_error("QSA pooled state extends beyond the KV frontier");
    }
    if (state.qsa_pooled_count < block_count) {
        const std::size_t first_block = state.qsa_pooled_count;
        const std::size_t new_blocks = block_count - first_block;
        MlxArray fresh = state.qsa_raw_keys.slice(
            std::vector<int>{0, coordinate(first_block * indexer_compress_ratio_,
                                           "QSA pooled start"), 0},
            std::vector<int>{1, dimension(block_count * indexer_compress_ratio_,
                                          "QSA pooled stop"), index_dimension},
            strides3);
        fresh = fresh.reshape(std::vector<int>{
            1, dimension(new_blocks, "QSA new blocks"), ratio, index_dimension});
        MlxArray pooled = fresh.astype(MLX_FLOAT32)
                              .mean_axis(2)
                              .astype(raw_keys.dtype())
                              .rms_norm(indexer_key_norm_weight_, epsilon_)
                              .reshape(std::vector<int>{
                                  1, 1, dimension(new_blocks, "QSA new blocks"),
                                  index_dimension});
        pooled = apply_rope_rows(
            pooled,
            state.position_base + first_block * indexer_compress_ratio_,
            indexer_compress_ratio_,
            indexer_head_dimension_)
                     .reshape(std::vector<int>{
                         1, dimension(new_blocks, "QSA new blocks"), index_dimension});
        state.qsa_pooled_keys = first_block == 0
            ? pooled.share()
            : MlxArray::concatenate(state.qsa_pooled_keys, pooled, 1);
        state.qsa_pooled_count = block_count;
    }

    MlxArray queries = qk.slice(
        std::vector<int>{0, 0, 0},
        std::vector<int>{1, rows, index_heads * index_dimension},
        strides3)
                           .reshape(std::vector<int>{1, rows, index_heads, index_dimension})
                           .rms_norm(indexer_query_norm_weight_, epsilon_)
                           .swapaxes(1, 2);
    queries = apply_rope_rows(
        queries,
        state.position_base + state.token_count,
        1,
        indexer_head_dimension_);
    MlxArray pooled_transposed = state.qsa_pooled_keys
                                     .reshape(std::vector<int>{
                                         1, 1, dimension(block_count, "QSA blocks"),
                                         index_dimension})
                                     .swapaxes(2, 3);
    MlxArray scores = MlxArray::matmul(
        queries.astype(MLX_FLOAT32), pooled_transposed.astype(MLX_FLOAT32));
    scores = MlxArray::maximum(scores, scalar(0.0F, MLX_FLOAT32)).sum_axis(1);

    MlxArray positions = MlxArray::arange(
        static_cast<double>(state.token_count),
        static_cast<double>(total),
        1.0,
        MLX_INT32).reshape(std::vector<int>{rows, 1});
    MlxArray block_ends = MlxArray::arange(
        static_cast<double>(ratio - 1),
        static_cast<double>(block_count * indexer_compress_ratio_),
        static_cast<double>(ratio),
        MLX_INT32).reshape(std::vector<int>{1, dimension(block_count, "QSA blocks")});
    MlxArray visible = MlxArray::less_equal(block_ends, positions).expand_dims(0);

    MlxArray block_indices = MlxArray::arange(
        0.0, static_cast<double>(block_count), 1.0, MLX_FLOAT32);
    MlxArray tie_bias = MlxArray::multiply(
        block_indices, scalar(1.0e-7F, MLX_FLOAT32));
    MlxArray biased_scores = MlxArray::subtract(scores, tie_bias);
    MlxArray masked_scores = MlxArray::where(
        visible,
        biased_scores,
        scalar(-std::numeric_limits<float>::infinity(), MLX_FLOAT32));
    const int first_top = dimension(block_count, "QSA blocks") - block_topk;
    MlxArray partition = masked_scores.argpartition_axis(first_top, -1);
    MlxArray top_indices = partition.slice(
        std::vector<int>{0, 0, first_top},
        std::vector<int>{1, rows, dimension(block_count, "QSA blocks")},
        strides3);
    MlxArray selected = MlxArray::put_along_axis(
        MlxArray::zeros(
            std::vector<int>{1, rows, dimension(block_count, "QSA blocks")},
            MLX_BOOL),
        top_indices,
        integer_scalar(1).astype(MLX_BOOL),
        -1);
    selected = MlxArray::logical_and(selected, visible);

    MlxArray selected_tokens = selected.expand_dims(-1)
                                   .broadcast_to(std::vector<int>{
                                       1, rows, dimension(block_count, "QSA blocks"), ratio})
                                   .reshape(std::vector<int>{
                                       1, rows,
                                       dimension(block_count * indexer_compress_ratio_,
                                                 "QSA selected tokens")});
    const std::size_t pad = total - block_count * indexer_compress_ratio_;
    if (pad != 0) {
        selected_tokens = MlxArray::concatenate(
            selected_tokens,
            MlxArray::zeros(
                std::vector<int>{1, rows, dimension(pad, "QSA tail padding")}, MLX_BOOL),
            2);
    }

    MlxArray key_indices = MlxArray::arange(
        0.0, static_cast<double>(total), 1.0, MLX_INT32)
                               .reshape(std::vector<int>{1, dimension(total, "QSA KV")});
    MlxArray completed_for_query = MlxArray::floor_divide(
        MlxArray::add(positions, integer_scalar(1)), integer_scalar(ratio));
    MlxArray tail_start = MlxArray::multiply(completed_for_query, integer_scalar(ratio));
    MlxArray tail = MlxArray::greater_equal(key_indices, tail_start);
    MlxArray causal = MlxArray::less_equal(key_indices, positions);
    MlxArray mask = MlxArray::logical_and(
        MlxArray::logical_or(selected_tokens, tail), causal);

    static std::once_flag announced;
    std::call_once(announced, [total, block_count, block_topk] {
        std::clog << "qwen38: QSA engaged at " << total << " tokens ("
                  << block_count << " blocks, top " << block_topk << ")\n";
    });
    return mask.reshape(std::vector<int>{1, 1, rows, dimension(total, "QSA KV")});
}

void SelfAttention::copy_qsa_checkpoint(
    const SelfAttentionState& complete,
    const std::size_t token_count,
    SelfAttentionState& checkpoint) const {
    if (token_count == 0) return;
    const int index_dimension = dimension(indexer_head_dimension_, "indexer dimension");
    const std::vector<int> strides{1, 1, 1};
    checkpoint.qsa_raw_keys = complete.qsa_raw_keys.slice(
        std::vector<int>{0, 0, 0},
        std::vector<int>{1, dimension(token_count, "QSA checkpoint"), index_dimension},
        strides);
    const std::size_t complete_blocks = token_count / indexer_compress_ratio_;
    const std::size_t selection_limit = indexer_budget_ / indexer_compress_ratio_;
    const std::size_t blocks = complete_blocks > selection_limit
        ? std::min(complete_blocks, complete.qsa_pooled_count)
        : 0;
    if (blocks != 0) {
        checkpoint.qsa_pooled_keys = complete.qsa_pooled_keys.slice(
            std::vector<int>{0, 0, 0},
            std::vector<int>{1, dimension(blocks, "QSA checkpoint blocks"),
                             index_dimension},
            strides);
        checkpoint.qsa_pooled_count = blocks;
    }
}

MlxArray SelfAttention::forward_decode(
    const MlxArray& input,
    SelfAttentionState& state) const {
    const auto input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] != 1) {
        throw std::runtime_error("attention decode requires shape [1,1,hidden]");
    }
    MlxArray qsa_mask = update_qsa_and_build_mask(input, state);
    const int heads = dimension(attention_heads_, "attention heads");
    const int kv_heads = dimension(key_value_heads_, "key/value heads");
    const int head_dimension = dimension(head_dimension_, "head dimension");
    const std::vector<int> query_shape{1, 1, heads, 2 * head_dimension};
    MlxArray query_gate = project(input, query_projection_).reshape(query_shape);
    const std::vector<int> strides{1, 1, 1, 1};
    const std::vector<int> query_start{0, 0, 0, 0};
    const std::vector<int> query_stop{1, 1, heads, head_dimension};
    const std::vector<int> gate_start{0, 0, 0, head_dimension};
    const std::vector<int> gate_stop{1, 1, heads, 2 * head_dimension};
    MlxArray query = query_gate.slice(query_start, query_stop, strides);
    MlxArray gate = query_gate.slice(gate_start, gate_stop, strides);
    query = query.rms_norm(query_norm_weight_, epsilon_).swapaxes(1, 2);
    const std::vector<int> key_value_shape{1, 1, kv_heads, head_dimension};
    MlxArray key = project(input, key_projection_).reshape(key_value_shape);
    key = key.rms_norm(key_norm_weight_, epsilon_).swapaxes(1, 2);
    MlxArray value = project(input, value_projection_).reshape(key_value_shape).swapaxes(1, 2);
    const std::size_t position = state.position_base + state.token_count;
    query = apply_rope(query, position);
    key = apply_rope(key, position);
    if (state.token_count == 0) {
        state.keys = std::move(key);
        state.values = std::move(value);
    } else {
        state.keys = MlxArray::concatenate(state.keys, key, 2);
        state.values = MlxArray::concatenate(state.values, value, 2);
    }
    ++state.token_count;
    MlxArray attended;
    if (qsa_mask.get().ctx != nullptr) {
        attended = MlxArray::scaled_dot_product_attention(
            query,
            state.keys,
            state.values,
            1.0F / std::sqrt(static_cast<float>(head_dimension_)),
            qsa_mask).swapaxes(1, 2);
    } else if (use_sdpa_decode(state.token_count)) {
        attended = MlxArray::scaled_dot_product_attention(
            query,
            state.keys,
            state.values,
            1.0F / std::sqrt(static_cast<float>(head_dimension_)),
            false).swapaxes(1, 2);
    } else {
        const int repetitions = heads / kv_heads;
        MlxArray repeated_keys = state.keys.repeat_axis(repetitions, 1);
        MlxArray repeated_values = state.values.repeat_axis(repetitions, 1);
        MlxArray scores = MlxArray::matmul(query, repeated_keys.swapaxes(2, 3));
        MlxArray scale = scalar(
            1.0F / std::sqrt(static_cast<float>(head_dimension_)), scores.dtype());
        scores = MlxArray::multiply(scores, scale);
        MlxArray probabilities = scores.astype(MLX_FLOAT32).softmax_axis(-1).astype(query.dtype());
        attended = MlxArray::matmul(probabilities, repeated_values).swapaxes(1, 2);
    }
    const std::vector<int> flat_shape{1, 1, heads * head_dimension};
    MlxArray gated = MlxArray::multiply(
        attended.reshape(flat_shape), gate.reshape(flat_shape).sigmoid());
    return project(gated, output_projection_);
}

MlxArray SelfAttention::forward_verify(
    const MlxArray& input,
    const SelfAttentionState& origin,
    std::vector<SelfAttentionState>& checkpoints) const {
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] < 1 ||
        input_shape[1] > 512) {
        throw std::runtime_error("attention batch requires shape [1,S,hidden], S=1..512");
    }
    const std::size_t rows = static_cast<std::size_t>(input_shape[1]);
    const int heads = dimension(attention_heads_, "attention heads");
    const int kv_heads = dimension(key_value_heads_, "key/value heads");
    const int head_dimension = dimension(head_dimension_, "head dimension");
    MlxArray query_gate_batch = project(input, query_projection_).reshape(
        std::vector<int>{1, input_shape[1], heads, 2 * head_dimension});
    MlxArray key_batch = project(input, key_projection_).reshape(
        std::vector<int>{1, input_shape[1], kv_heads, head_dimension});
    MlxArray value_batch = project(input, value_projection_).reshape(
        std::vector<int>{1, input_shape[1], kv_heads, head_dimension});

    const char* batch_sdpa_verify = std::getenv("QWEN38_BATCH_SDPA_VERIFY");
    const bool batch_sdpa_enabled = batch_sdpa_verify == nullptr ||
        std::string_view(batch_sdpa_verify) != "0";
    if (rows > 1 && batch_sdpa_enabled && use_sdpa_decode(origin.token_count + 1)) {
        static std::once_flag announced_batch_verify;
        std::call_once(announced_batch_verify, [rows, origin_tokens = origin.token_count] {
            std::clog << "qwen38: batched grouped-query SDPA verifier engaged with "
                      << rows << " rows at " << origin_tokens
                      << " cached attention tokens\n";
        });
        const std::vector<int> strides{1, 1, 1, 1};
        SelfAttentionState complete;
        complete.token_count = origin.token_count;
        complete.position_base = origin.position_base;
        complete.qsa_pooled_count = origin.qsa_pooled_count;
        if (origin.token_count != 0) {
            complete.qsa_raw_keys = origin.qsa_raw_keys.share();
            if (origin.qsa_pooled_count != 0) {
                complete.qsa_pooled_keys = origin.qsa_pooled_keys.share();
            }
        }
        MlxArray qsa_mask = update_qsa_and_build_mask(input, complete);
        MlxArray query = query_gate_batch.slice(
            std::vector<int>{0, 0, 0, 0},
            std::vector<int>{1, input_shape[1], heads, head_dimension},
            strides).rms_norm(query_norm_weight_, epsilon_).swapaxes(1, 2);
        MlxArray gate = query_gate_batch.slice(
            std::vector<int>{0, 0, 0, head_dimension},
            std::vector<int>{1, input_shape[1], heads, 2 * head_dimension},
            strides);
        MlxArray keys = key_batch.rms_norm(
            key_norm_weight_, epsilon_).swapaxes(1, 2);
        MlxArray values = value_batch.swapaxes(1, 2);
        const std::size_t position = origin.position_base + origin.token_count;
        query = apply_rope_prefill(query, position);
        keys = apply_rope_prefill(keys, position);

        MlxArray complete_keys = origin.token_count == 0
            ? keys.share()
            : MlxArray::concatenate(origin.keys, keys, 2);
        MlxArray complete_values = origin.token_count == 0
            ? values.share()
            : MlxArray::concatenate(origin.values, values, 2);
        complete.keys = complete_keys.share();
        complete.values = complete_values.share();
        complete.token_count = origin.token_count + rows;
        checkpoints.clear();
        checkpoints.resize(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            const int stop = dimension(
                origin.token_count + row + 1, "attention checkpoint length");
            checkpoints[row].keys = complete_keys.slice(
                std::vector<int>{0, 0, 0, 0},
                std::vector<int>{1, kv_heads, stop, head_dimension},
                strides);
            checkpoints[row].values = complete_values.slice(
                std::vector<int>{0, 0, 0, 0},
                std::vector<int>{1, kv_heads, stop, head_dimension},
                strides);
            checkpoints[row].token_count = origin.token_count + row + 1;
            checkpoints[row].position_base = origin.position_base;
            copy_qsa_checkpoint(
                complete, checkpoints[row].token_count, checkpoints[row]);
        }
        MlxArray attended = qsa_mask.get().ctx == nullptr
            ? MlxArray::scaled_dot_product_attention(
                  query,
                  complete_keys,
                  complete_values,
                  1.0F / std::sqrt(static_cast<float>(head_dimension_)),
                  true).swapaxes(1, 2)
            : MlxArray::scaled_dot_product_attention(
                  query,
                  complete_keys,
                  complete_values,
                  1.0F / std::sqrt(static_cast<float>(head_dimension_)),
                  qsa_mask).swapaxes(1, 2);
        const std::vector<int> flat_shape{1, input_shape[1], heads * head_dimension};
        MlxArray gated = MlxArray::multiply(
            attended.reshape(flat_shape), gate.reshape(flat_shape).sigmoid());
        return project(gated, output_projection_);
    }

    SelfAttentionState working;
    working.token_count = origin.token_count;
    working.position_base = origin.position_base;
    working.qsa_pooled_count = origin.qsa_pooled_count;
    if (origin.token_count != 0) {
        working.keys = origin.keys.share();
        working.values = origin.values.share();
        working.qsa_raw_keys = origin.qsa_raw_keys.share();
        if (origin.qsa_pooled_count != 0) {
            working.qsa_pooled_keys = origin.qsa_pooled_keys.share();
        }
    }
    checkpoints.clear();
    checkpoints.resize(rows);
    std::vector<MlxArray> gated_rows;
    gated_rows.reserve(rows);
    const std::vector<int> strides{1, 1, 1, 1};
    const char* batch_kv_verify = std::getenv("QWEN38_BATCH_KV_VERIFY");
    const bool batch_kv_enabled = rows > 1 && batch_kv_verify != nullptr &&
        std::string_view(batch_kv_verify) == "1";
    MlxArray complete_keys;
    MlxArray complete_values;
    if (batch_kv_enabled) {
        MlxArray new_keys = key_batch.rms_norm(
            key_norm_weight_, epsilon_).swapaxes(1, 2);
        MlxArray new_values = value_batch.swapaxes(1, 2);
        new_keys = apply_rope_prefill(
            new_keys, origin.position_base + origin.token_count);
        complete_keys = origin.token_count == 0
            ? new_keys.share()
            : MlxArray::concatenate(origin.keys, new_keys, 2);
        complete_values = origin.token_count == 0
            ? new_values.share()
            : MlxArray::concatenate(origin.values, new_values, 2);
    }
    for (std::size_t row = 0; row < rows; ++row) {
        MlxArray qsa_mask = update_qsa_and_build_mask(
            slice_sequence_row(input, row), working);
        MlxArray query_gate = slice_sequence_row(query_gate_batch, row);
        MlxArray query = query_gate.slice(
            std::vector<int>{0, 0, 0, 0},
            std::vector<int>{1, 1, heads, head_dimension},
            strides);
        MlxArray gate = query_gate.slice(
            std::vector<int>{0, 0, 0, head_dimension},
            std::vector<int>{1, 1, heads, 2 * head_dimension},
            strides);
        query = query.rms_norm(query_norm_weight_, epsilon_).swapaxes(1, 2);
        const std::size_t position = working.position_base + working.token_count;
        query = apply_rope(query, position);
        if (batch_kv_enabled) {
            ++working.token_count;
            const int stop = dimension(working.token_count, "attention checkpoint length");
            working.keys = complete_keys.slice(
                std::vector<int>{0, 0, 0, 0},
                std::vector<int>{1, kv_heads, stop, head_dimension},
                strides);
            working.values = complete_values.slice(
                std::vector<int>{0, 0, 0, 0},
                std::vector<int>{1, kv_heads, stop, head_dimension},
                strides);
        } else {
            MlxArray key = slice_sequence_row(key_batch, row)
                .rms_norm(key_norm_weight_, epsilon_).swapaxes(1, 2);
            MlxArray value = slice_sequence_row(value_batch, row).swapaxes(1, 2);
            key = apply_rope(key, position);
            if (working.token_count == 0) {
                working.keys = std::move(key);
                working.values = std::move(value);
            } else {
                working.keys = MlxArray::concatenate(working.keys, key, 2);
                working.values = MlxArray::concatenate(working.values, value, 2);
            }
            ++working.token_count;
        }
        checkpoints[row].keys = working.keys.share();
        checkpoints[row].values = working.values.share();
        checkpoints[row].token_count = working.token_count;
        checkpoints[row].position_base = working.position_base;
        checkpoints[row].qsa_raw_keys = working.qsa_raw_keys.share();
        checkpoints[row].qsa_pooled_count = working.qsa_pooled_count;
        if (working.qsa_pooled_count != 0) {
            checkpoints[row].qsa_pooled_keys = working.qsa_pooled_keys.share();
        }

        MlxArray attended;
        if (qsa_mask.get().ctx != nullptr) {
            attended = MlxArray::scaled_dot_product_attention(
                query,
                working.keys,
                working.values,
                1.0F / std::sqrt(static_cast<float>(head_dimension_)),
                qsa_mask).swapaxes(1, 2);
        } else if (use_sdpa_decode(working.token_count)) {
            attended = MlxArray::scaled_dot_product_attention(
                query,
                working.keys,
                working.values,
                1.0F / std::sqrt(static_cast<float>(head_dimension_)),
                false).swapaxes(1, 2);
        } else {
            const int repetitions = heads / kv_heads;
            MlxArray repeated_keys = working.keys.repeat_axis(repetitions, 1);
            MlxArray repeated_values = working.values.repeat_axis(repetitions, 1);
            MlxArray scores = MlxArray::matmul(query, repeated_keys.swapaxes(2, 3));
            scores = MlxArray::multiply(
                scores,
                scalar(1.0F / std::sqrt(static_cast<float>(head_dimension_)), scores.dtype()));
            MlxArray probabilities =
                scores.astype(MLX_FLOAT32).softmax_axis(-1).astype(query.dtype());
            attended = MlxArray::matmul(
                probabilities, repeated_values).swapaxes(1, 2);
        }
        const std::vector<int> flat_shape{1, 1, heads * head_dimension};
        gated_rows.push_back(MlxArray::multiply(
            attended.reshape(flat_shape), gate.reshape(flat_shape).sigmoid()));
    }
    return project(concatenate_sequence_rows(gated_rows), output_projection_);
}

MlxArray SelfAttention::forward_prefill(
    const MlxArray& input,
    SelfAttentionState& state) const {
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 3 || input_shape[0] != 1 || input_shape[1] < 1 ||
        input_shape[1] > 512) {
        throw std::runtime_error("attention prefill requires shape [1,S,hidden], S=1..512");
    }
    const char* enabled = std::getenv("QWEN38_SDPA_PREFILL");
    if (enabled == nullptr || std::string_view(enabled) != "1" ||
        input_shape[1] < 16) {
        std::vector<SelfAttentionState> checkpoints;
        MlxArray output = forward_verify(input, state, checkpoints);
        state = std::move(checkpoints.back());
        return output;
    }

    const int rows = input_shape[1];
    const int heads = dimension(attention_heads_, "attention heads");
    const int kv_heads = dimension(key_value_heads_, "key/value heads");
    const int head_dimension = dimension(head_dimension_, "head dimension");
    MlxArray qsa_mask = update_qsa_and_build_mask(input, state);
    MlxArray query_gate = project(input, query_projection_).reshape(
        std::vector<int>{1, rows, heads, 2 * head_dimension});
    const std::vector<int> strides{1, 1, 1, 1};
    MlxArray query = query_gate.slice(
        std::vector<int>{0, 0, 0, 0},
        std::vector<int>{1, rows, heads, head_dimension},
        strides).rms_norm(query_norm_weight_, epsilon_).swapaxes(1, 2);
    MlxArray gate = query_gate.slice(
        std::vector<int>{0, 0, 0, head_dimension},
        std::vector<int>{1, rows, heads, 2 * head_dimension},
        strides);
    MlxArray key = project(input, key_projection_).reshape(
        std::vector<int>{1, rows, kv_heads, head_dimension})
        .rms_norm(key_norm_weight_, epsilon_).swapaxes(1, 2);
    MlxArray value = project(input, value_projection_).reshape(
        std::vector<int>{1, rows, kv_heads, head_dimension}).swapaxes(1, 2);
    const std::size_t position = state.position_base + state.token_count;
    query = apply_rope_prefill(query, position);
    key = apply_rope_prefill(key, position);
    if (state.token_count == 0) {
        state.keys = key.share();
        state.values = value.share();
    } else {
        state.keys = MlxArray::concatenate(state.keys, key, 2);
        state.values = MlxArray::concatenate(state.values, value, 2);
    }
    state.token_count += static_cast<std::size_t>(rows);

    MlxArray attended = qsa_mask.get().ctx == nullptr
        ? MlxArray::scaled_dot_product_attention(
              query,
              state.keys,
              state.values,
              1.0F / std::sqrt(static_cast<float>(head_dimension_)),
              true).swapaxes(1, 2)
        : MlxArray::scaled_dot_product_attention(
              query,
              state.keys,
              state.values,
              1.0F / std::sqrt(static_cast<float>(head_dimension_)),
              qsa_mask).swapaxes(1, 2);
    const std::vector<int> flat_shape{1, rows, heads * head_dimension};
    MlxArray gated = MlxArray::multiply(
        attended.reshape(flat_shape), gate.reshape(flat_shape).sigmoid());
    return project(gated, output_projection_);
}

} // namespace qwen38
