#include "qwen38/runtime_profile.hpp"
#include "qwen38/self_attention.hpp"
#include "qwen38/token_embedding.hpp"
#include "qsa_metal_kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace qwen38;
using Clock = std::chrono::steady_clock;

namespace {
struct Packed { MlxArray weight, scale, bias; };

Packed pack(const MlxArray& value, const int bits, const int group_size = 64) {
    auto tuple = mlx_vector_array_new();
    const auto stream = mlx_default_gpu_stream_new();
    const int status = mlx_quantize(
        &tuple, value.get(), {.value = group_size, .has_value = true},
        {.value = bits, .has_value = true}, "affine", mlx_array{}, stream);
    mlx_stream_free(stream);
    if (status != 0) {
        mlx_vector_array_free(tuple);
        throw std::runtime_error("KV quantization failed");
    }
    std::array<mlx_array, 3> raw{mlx_array_new(), mlx_array_new(), mlx_array_new()};
    const int extract = mlx_vector_array_get(&raw[0], tuple, 0) |
        mlx_vector_array_get(&raw[1], tuple, 1) |
        mlx_vector_array_get(&raw[2], tuple, 2);
    mlx_vector_array_free(tuple);
    if (extract != 0) throw std::runtime_error("KV quantization tuple failed");
    Packed result{MlxArray(raw[0]), MlxArray(raw[1]), MlxArray(raw[2])};
    const std::array<const MlxArray*, 3> outputs{&result.weight, &result.scale, &result.bias};
    MlxArray::eval_all(outputs);
    return result;
}

Packed pack_q4_group16(const MlxArray& value) {
    static const char* inputs[]{"x"};
    static const char* outputs[]{"weight", "scale", "bias"};
    static MlxMetalKernel kernel(
        "qwen38_kv4_probe_pack_g16", inputs, outputs, R"metal(
            const uint group = thread_position_in_grid.x;
            if (group >= uint(GROUPS)) return;
            const uint base = group * 16u;
            float low = INFINITY;
            float high = -INFINITY;
            for (uint channel = 0; channel < 16u; ++channel) {
                const float value = float(x[base + channel]);
                low = min(low, value);
                high = max(high, value);
            }
            const float step = high > low ? (high - low) / 15.0f : 1.0f;
            uint first = 0u;
            uint second = 0u;
            for (uint channel = 0; channel < 16u; ++channel) {
                const uint quantized = uint(clamp(round((float(x[base + channel]) - low) /
                    step), 0.0f, 15.0f));
                if (channel < 8u) first |= quantized << (channel * 4u);
                else second |= quantized << ((channel - 8u) * 4u);
            }
            weight[group * 2u] = first;
            weight[group * 2u + 1u] = second;
            scale[group] = T(step);
            bias[group] = T(low);
        )metal");
    const std::vector<int> shape = value.shape();
    if (shape.empty() || shape.back() % 16 != 0)
        throw std::runtime_error("group-16 Q4 input shape mismatch");
    std::vector<int> weight_shape = shape;
    std::vector<int> scale_shape = shape;
    weight_shape.back() /= 8;
    scale_shape.back() /= 16;
    const int groups = static_cast<int>(value.size() / 16);
    const std::array<const MlxArray*, 1> arguments{&value};
    const std::array<MlxMetalOutputSpec, 3> specs{{
        {.shape = weight_shape, .dtype = MLX_UINT32},
        {.shape = scale_shape, .dtype = value.dtype()},
        {.shape = scale_shape, .dtype = value.dtype()},
    }};
    const std::array<int, 3> grid{((groups + 255) / 256) * 256, 1, 1};
    const std::array<int, 3> threadgroup{256, 1, 1};
    const std::array<MlxMetalDtypeTemplate, 1> types{{{"T", value.dtype()}}};
    const std::array<MlxMetalIntTemplate, 1> constants{{{"GROUPS", groups}}};
    std::vector<MlxArray> result = kernel.apply(
        arguments, specs, grid, threadgroup, types, constants);
    Packed packed{std::move(result[0]), std::move(result[1]), std::move(result[2])};
    const std::array<const MlxArray*, 3> ready{&packed.weight, &packed.scale, &packed.bias};
    MlxArray::eval_all(ready);
    return packed;
}

std::size_t bytes(const Packed& packed) {
    return packed.weight.byte_size() + packed.scale.byte_size() + packed.bias.byte_size();
}

struct Quality { double cosine{}, relative_l2{}, max_abs{}; };
Quality compare(const std::vector<float>& reference, const std::vector<float>& actual) {
    if (reference.size() != actual.size()) throw std::runtime_error("quality size mismatch");
    double dot = 0, rr = 0, aa = 0, error = 0, maximum = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i]))
            throw std::runtime_error("non-finite attention output");
        const double difference = actual[i] - reference[i];
        dot += static_cast<double>(reference[i]) * actual[i];
        rr += static_cast<double>(reference[i]) * reference[i];
        aa += static_cast<double>(actual[i]) * actual[i];
        error += difference * difference;
        maximum = std::max(maximum, std::abs(difference));
    }
    return {dot / std::sqrt(rr * aa), std::sqrt(error / rr), maximum};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

class AttentionKernels final {
public:
    AttentionKernels()
        : bf16_("qwen38_kv4_probe_bf16", bf16_names_, "output",
                qsa_metal::packed_attention, qsa_metal::header),
          q8_("qwen38_kv4_probe_q8", q8_names_, "output",
              qsa_metal::packed_attention_q8, qsa_metal::header),
          q4_("qwen38_kv4_probe_q4", q4_names_, "output",
              qsa_metal::packed_attention_q4, qsa_metal::header) {}

    MlxArray bf16(const MlxArray& query, const MlxArray& keys, const MlxArray& values,
                  const MlxArray& indices, const MlxArray& valid, const MlxArray& scale,
                  const MlxArray& total, int heads, int kv_heads, int selected, int dimension) {
        const std::array<const MlxArray*, 7> inputs{
            &query, &keys, &values, &indices, &valid, &scale, &total};
        return apply(bf16_, inputs, query, heads, kv_heads, selected, dimension);
    }

    MlxArray q8(const MlxArray& query, const Packed& keys, const Packed& values,
                const MlxArray& indices, const MlxArray& valid, const MlxArray& scale,
                const MlxArray& total, int heads, int kv_heads, int selected, int dimension) {
        const std::array<int, 4> empty_shape{1, kv_heads, 0, dimension};
        MlxArray empty = MlxArray::zeros(empty_shape, query.dtype());
        const std::array<const MlxArray*, 14> inputs{
            &query, &keys.weight, &keys.scale, &keys.bias,
            &values.weight, &values.scale, &values.bias, &empty, &empty,
            &indices, &valid, &scale, &total, &total};
        return apply(q8_, inputs, query, heads, kv_heads, selected, dimension);
    }

    MlxArray q4(const MlxArray& query, const Packed& keys, const Packed& values,
                const MlxArray& indices, const MlxArray& valid, const MlxArray& scale,
                const MlxArray& total, int heads, int kv_heads, int selected, int dimension,
                int quantization_group) {
        const std::array<const MlxArray*, 11> inputs{
            &query, &keys.weight, &keys.scale, &keys.bias,
            &values.weight, &values.scale, &values.bias,
            &indices, &valid, &scale, &total};
        constexpr int threads = 512;
        const std::array<int, 3> grid{threads, 1, kv_heads};
        const std::array<int, 3> group{threads, 1, 1};
        const std::array<MlxMetalDtypeTemplate, 1> types{{{"T", query.dtype()}}};
        const std::array<MlxMetalIntTemplate, 8> args{{
            {"R", 1}, {"S", selected}, {"HQ", heads}, {"HK", kv_heads},
            {"D", dimension}, {"TG", threads}, {"TILE_SIZE", 32},
            {"QGROUP", quantization_group}}};
        const std::array<MlxMetalOutputSpec, 1> outputs{{
            {.shape = {1, 1, heads, dimension}, .dtype = query.dtype()}}};
        std::vector<MlxArray> result = q4_.apply(inputs, outputs, grid, group, types, args);
        return std::move(result.front());
    }

private:
    template <std::size_t N>
    static MlxArray apply(MlxMetalKernel& kernel,
                          const std::array<const MlxArray*, N>& inputs,
                          const MlxArray& query, int heads, int kv_heads,
                          int selected, int dimension) {
        constexpr int threads = 512;
        const std::array<int, 3> grid{threads, 1, kv_heads};
        const std::array<int, 3> group{threads, 1, 1};
        const std::array<MlxMetalDtypeTemplate, 1> types{{{"T", query.dtype()}}};
        const std::array<MlxMetalIntTemplate, 7> args{{
            {"R", 1}, {"S", selected}, {"HQ", heads}, {"HK", kv_heads},
            {"D", dimension}, {"TG", threads}, {"TILE_SIZE", 32}}};
        const std::array<MlxMetalOutputSpec, 1> outputs{{
            {.shape = {1, 1, heads, dimension}, .dtype = query.dtype()}}};
        std::vector<MlxArray> result = kernel.apply(inputs, outputs, grid, group, types, args);
        return std::move(result.front());
    }

    inline static constexpr const char* bf16_names_[]{
        "query", "keys", "values", "indices", "valid", "scale", "total"};
    inline static constexpr const char* q8_names_[]{
        "query", "kw", "ks", "kb", "vw", "vs", "vb", "hot_keys", "hot_values",
        "indices", "valid", "scale", "total", "cold"};
    inline static constexpr const char* q4_names_[]{
        "query", "kw", "ks", "kb", "vw", "vs", "vb",
        "indices", "valid", "scale", "total"};
    MlxMetalKernel bf16_, q8_, q4_;
};
} // namespace

int main(int argc, char** argv) try {
    if ((argc != 2 && argc != 3) || std::getenv("QWEN38_MEMORY_GUARD") == nullptr)
        throw std::runtime_error("usage: guarded kv4-attention-probe MODEL [8192|65536]");
    const std::size_t context = argc == 3 ? std::stoull(argv[2]) : 8192;
    if (context != 8192 && context != 65536) throw std::runtime_error("unsupported context");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 3ULL * 1024 * 1024 * 1024) != 0)
        throw std::runtime_error("MLX memory cap failed");
    static_cast<void>(MlxArray::set_cache_limit(64ULL * 1024 * 1024));
    apply_runtime_profile("speed");

    MlxTensorStore tensors(ModelManifest::load(argv[1]));
    const auto& config = tensors.manifest().config();
    const QuantizationSpec embedding_quantization =
        tensors.manifest().quantization_for("language_model.model.embed_tokens");
    SelfAttention attention(tensors, "language_model.model.layers.3.self_attn", config);
    const auto embedding = [&](std::size_t offset, std::size_t rows) {
        std::vector<std::uint32_t> tokens(rows);
        for (std::size_t i = 0; i < rows; ++i)
            tokens[i] = static_cast<std::uint32_t>((9419 + 7919 * (i + offset)) %
                                                   config.vocabulary_size);
        return embed_token_batch(
            tensors.tensor("language_model.model.embed_tokens.weight"),
            tensors.tensor("language_model.model.embed_tokens.scales"),
            tensors.tensor("language_model.model.embed_tokens.biases"), tokens,
            config.vocabulary_size, config.hidden_size,
            embedding_quantization.group_size, embedding_quantization.bits);
    };
    SelfAttentionState state;
    for (std::size_t offset = 0; offset < context; offset += 512) {
        MlxArray output = attention.forward_prefill(embedding(offset, 512), state);
        const std::array<const MlxArray*, 3> ready{&output, &state.keys, &state.values};
        MlxArray::eval_all(ready);
    }
    const auto shape = state.keys.shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[2] != static_cast<int>(context))
        throw std::runtime_error("unexpected KV shape");
    const int kv_heads = shape[1], dimension = shape[3];
    const int heads = static_cast<int>(config.attention_head_count);
    constexpr int selected = 2048;
    std::vector<std::int32_t> selected_tokens(selected);
    std::vector<std::int32_t> valid_values(selected, 1);
    for (int i = 0; i < selected; ++i)
        selected_tokens[i] = static_cast<std::int32_t>(
            (static_cast<std::size_t>(i) * context) / selected);
    const std::array<int, 3> selection_shape{1, 1, selected};
    MlxArray indices = MlxArray::from_int32(selected_tokens, selection_shape);
    MlxArray valid = MlxArray::from_int32(valid_values, selection_shape).astype(MLX_BOOL);
    const std::array<int, 0> scalar_shape{};
    MlxArray scale = MlxArray::from_float32(
        std::array<float, 1>{1.0F / std::sqrt(static_cast<float>(dimension))}, scalar_shape);
    MlxArray total = MlxArray::from_int32(
        std::array<std::int32_t, 1>{static_cast<std::int32_t>(context)}, scalar_shape);
    MlxArray query = state.keys.slice(
        std::array<int, 4>{0, 0, static_cast<int>(context) - 1, 0},
        std::array<int, 4>{1, kv_heads, static_cast<int>(context), dimension},
        std::array<int, 4>{1, 1, 1, 1}).repeat_axis(heads / kv_heads, 1);

    const auto packing_started = Clock::now();
    Packed k8 = pack(state.keys, 8), v8 = pack(state.values, 8);
    const double q8_pack_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - packing_started).count();
    const auto q4_started = Clock::now();
    Packed k4 = pack(state.keys, 4), v4 = pack(state.values, 4);
    const double q4_pack_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - q4_started).count();

    Packed k4g32 = pack(state.keys, 4, 32), v4g32 = pack(state.values, 4, 32);
    Packed k4g16 = pack_q4_group16(state.keys), v4g16 = pack_q4_group16(state.values);
    MlxArray decoded_k4g32 = MlxArray::dequantize(
        k4g32.weight, k4g32.scale, k4g32.bias, 32, 4);
    MlxArray decoded_v4g32 = MlxArray::dequantize(
        v4g32.weight, v4g32.scale, v4g32.bias, 32, 4);
    AttentionKernels kernels;
    auto invoke = [&](int mode) {
        if (mode == 0)
            return kernels.bf16(query, state.keys, state.values, indices, valid, scale,
                                total, heads, kv_heads, selected, dimension);
        if (mode == 1)
            return kernels.q8(query, k8, v8, indices, valid, scale, total,
                              heads, kv_heads, selected, dimension);
        if (mode == 2)
            return kernels.q4(query, k4, v4, indices, valid, scale, total,
                              heads, kv_heads, selected, dimension, 64);
        if (mode == 3)
            return kernels.q4(query, k4g32, v4g32, indices, valid, scale, total,
                              heads, kv_heads, selected, dimension, 32);
        return kernels.q4(query, k4g16, v4g16, indices, valid, scale, total,
                          heads, kv_heads, selected, dimension, 16);
    };
    std::vector<float> reference;
    std::array<Quality, 5> quality{};
    std::array<std::vector<double>, 5> samples;
    for (int repeat = 0; repeat < 18; ++repeat) {
        for (int mode = 0; mode < 5; ++mode) {
            const auto started = Clock::now();
            MlxArray output = invoke(mode);
            output.eval();
            const double elapsed = std::chrono::duration<double, std::milli>(
                Clock::now() - started).count();
            const std::vector<float> values = output.astype(MLX_FLOAT32).to_float32();
            if (repeat == 0 && mode == 0) reference = values;
            if (repeat == 0) quality[mode] = compare(reference, values);
            if (repeat >= 3) samples[mode].push_back(elapsed);
        }
    }
    const std::vector<float> k4_only = kernels.bf16(
        query, decoded_k4g32, state.values, indices, valid, scale, total,
        heads, kv_heads, selected, dimension).astype(MLX_FLOAT32).to_float32();
    const std::vector<float> v4_only = kernels.bf16(
        query, state.keys, decoded_v4g32, indices, valid, scale, total,
        heads, kv_heads, selected, dimension).astype(MLX_FLOAT32).to_float32();
    const std::vector<float> materialized_q4 = kernels.bf16(
        query, decoded_k4g32, decoded_v4g32, indices, valid, scale, total,
        heads, kv_heads, selected, dimension).astype(MLX_FLOAT32).to_float32();
    const std::vector<float> direct_q4 = kernels.q4(
        query, k4g32, v4g32, indices, valid, scale, total,
        heads, kv_heads, selected, dimension, 32).astype(MLX_FLOAT32).to_float32();
    const Quality k4_only_quality = compare(reference, k4_only);
    const Quality v4_only_quality = compare(reference, v4_only);
    const Quality direct_parity = compare(materialized_q4, direct_q4);
    const std::size_t bf16_bytes = state.keys.byte_size() + state.values.byte_size();
    std::cout << "{\"context\":" << context << ",\"selected\":" << selected
              << ",\"bf16_bytes\":" << bf16_bytes
              << ",\"q8_bytes\":" << bytes(k8) + bytes(v8)
              << ",\"q4_bytes\":" << bytes(k4) + bytes(v4)
              << ",\"q4g32_bytes\":" << bytes(k4g32) + bytes(v4g32)
              << ",\"q4g16_bytes\":" << bytes(k4g16) + bytes(v4g16)
              << ",\"q8_pack_ms\":" << q8_pack_ms
              << ",\"q4_pack_ms\":" << q4_pack_ms << "}\n";
    constexpr std::array<const char*, 5> names{
        "bf16", "q8", "q4g64", "q4g32", "q4g16"};
    for (int mode = 0; mode < 5; ++mode)
        std::cout << "{\"mode\":\"" << names[mode] << "\",\"median_ms\":"
                  << median(samples[mode]) << ",\"cosine\":" << quality[mode].cosine
                  << ",\"relative_l2\":" << quality[mode].relative_l2
                  << ",\"max_abs\":" << quality[mode].max_abs << "}\n";
    const auto print_quality = [](const char* name, const Quality& value) {
        std::cout << "{\"mode\":\"" << name << "\",\"quality_only\":true"
                  << ",\"cosine\":" << value.cosine
                  << ",\"relative_l2\":" << value.relative_l2
                  << ",\"max_abs\":" << value.max_abs << "}\n";
    };
    print_quality("q4g32_keys_bf16_values", k4_only_quality);
    print_quality("bf16_keys_q4g32_values", v4_only_quality);
    print_quality("q4g32_direct_vs_materialized", direct_parity);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
