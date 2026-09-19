#include "qwen38/model.hpp"
#include "qwen38/persistent_metal_backend.hpp"
#include "qwen38/runtime_profile.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
using namespace qwen38;

MlxArray pattern(const std::vector<int>& shape, int salt, mlx_dtype dtype = MLX_BFLOAT16) {
    const auto count = std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                                      std::multiplies<>{});
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = dtype == MLX_UINT32
            ? static_cast<float>((i * 29 + static_cast<std::size_t>(salt)) % 65521)
            : static_cast<float>((i + static_cast<std::size_t>(salt)) % 61) / 64.0F;
    return MlxArray::from_float32(values, shape).astype(dtype);
}

void equal(const MlxArray& a, const MlxArray& b, const char* name) {
    if (a.shape() != b.shape() || a.dtype() != b.dtype() || a.to_bytes() != b.to_bytes())
        throw std::runtime_error(std::string("state mismatch: ") + name);
}

ModelDecodeState fixture(int tokens, int cold) {
    ModelDecodeState state(48);
    state.token_count = static_cast<std::size_t>(tokens);
    for (int l = 0; l < 48; ++l) {
        auto& layer = state.layers[static_cast<std::size_t>(l)];
        if (l % 4 != 3) {
            layer.linear_attention.initialized = true;
            layer.linear_attention.convolution = pattern({1, 3, 10240}, l);
            layer.linear_attention.recurrent = pattern({1, 48, 128, 128}, l + 2);
            continue;
        }
        auto& a = layer.full_attention;
        a.token_count = state.token_count;
        a.position_base = 13;
        a.qsa_raw_start = static_cast<std::size_t>(tokens - tokens % 4);
        a.qsa_raw_keys = pattern({1, tokens % 4, 128}, l + 1);
        a.qsa_pooled_count = static_cast<std::size_t>(tokens / 4);
        if (tokens >= 4) a.qsa_pooled_keys = pattern({1, tokens / 4, 128}, l + 2);
        if (tokens > cold) {
            a.keys = pattern({1, 2, tokens - cold, 256}, l + 3);
            a.values = pattern({1, 2, tokens - cold, 256}, l + 4);
        }
        if (cold) {
            a.kv_q8 = true;
            a.kv_q8_cold_tokens = static_cast<std::size_t>(cold);
            a.key_weights = pattern({1, 2, cold, 64}, l + 5, MLX_UINT32);
            a.value_weights = pattern({1, 2, cold, 64}, l + 6, MLX_UINT32);
            a.key_scales = pattern({1, 2, cold, 4}, l + 7);
            a.key_biases = pattern({1, 2, cold, 4}, l + 8);
            a.value_scales = pattern({1, 2, cold, 4}, l + 9);
            a.value_biases = pattern({1, 2, cold, 4}, l + 10);
        }
    }
    state.layers[1].ple.convolution_initialized = true;
    state.layers[1].ple.convolution = pattern({1, 9, 10240}, 7);
    state.layers[1].ple.ngram.initialized = true;
    state.layers[1].ple.ngram.previous = {9419, 11};
    return state;
}

void equal_states(const ModelDecodeState& a, const ModelDecodeState& b) {
    if (a.token_count != b.token_count) throw std::runtime_error("token count mismatch");
    for (std::size_t l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            equal(a.layers[l].linear_attention.convolution,
                  b.layers[l].linear_attention.convolution, "GDN convolution");
            equal(a.layers[l].linear_attention.recurrent,
                  b.layers[l].linear_attention.recurrent, "GDN recurrence");
            continue;
        }
        const auto& x = a.layers[l].full_attention;
        const auto& y = b.layers[l].full_attention;
        if (x.token_count != y.token_count || x.position_base != y.position_base || x.qsa_raw_start != y.qsa_raw_start ||
            x.qsa_pooled_count != y.qsa_pooled_count || x.kv_q8 != y.kv_q8 ||
            x.kv_q8_cold_tokens != y.kv_q8_cold_tokens)
            throw std::runtime_error("attention metadata mismatch");
        equal(x.qsa_raw_keys, y.qsa_raw_keys, "QSA pending");
        if (x.qsa_pooled_count) equal(x.qsa_pooled_keys, y.qsa_pooled_keys, "QSA pools");
        if (x.token_count > x.kv_q8_cold_tokens) {
            equal(x.keys, y.keys, "hot K"); equal(x.values, y.values, "hot V");
        }
        if (x.kv_q8) {
            equal(x.key_weights, y.key_weights, "Q8 K");
            equal(x.value_weights, y.value_weights, "Q8 V");
            equal(x.key_scales, y.key_scales, "Q8 K scales");
            equal(x.key_biases, y.key_biases, "Q8 K biases");
            equal(x.value_scales, y.value_scales, "Q8 V scales");
            equal(x.value_biases, y.value_biases, "Q8 V biases");
        }
    }
    equal(a.layers[1].ple.convolution, b.layers[1].ple.convolution, "PLE convolution");
    if (a.layers[1].ple.ngram.previous != b.layers[1].ple.ngram.previous)
        throw std::runtime_error("PLE ngram mismatch");
}

void state_io(PersistentMetalBackend& backend, MlxTensorStore& tensors) {
    for (const auto [tokens, cold] : std::array<std::pair<int, int>, 8>{{
             {3, 0}, {4, 0}, {7, 0}, {511, 0}, {513, 0}, {8192, 0},
             {8193, 2048}, {2048, 2048}}}) {
        auto a = fixture(tokens, cold);
        backend.import_state(a);
        auto b = backend.export_state();
        equal_states(a, b);
        if (backend.can_decode() != (tokens - cold < 8192))
            throw std::runtime_error("capacity preflight mismatch");
        if (!backend.can_decode()) {
            bool refused = false;
            try { static_cast<void>(backend.greedy_decode(9419)); }
            catch (const std::runtime_error&) { refused = true; }
            if (!refused) throw std::runtime_error("full slab was not rejected before dispatch");
            equal_states(b, backend.export_state());
        }
        // An exported checkpoint must remain immutable across re-import.
        backend.import_state(fixture(3, 0));
        equal_states(a, b);
        std::cout << "state_roundtrip tokens=" << tokens << " cold=" << cold
                  << " exact=true\n" << std::flush;
    }
    auto invalid = fixture(7, 0);
    invalid.layers[3].full_attention.qsa_pooled_count = 0;
    bool rejected = false;
    try { backend.import_state(invalid); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("missing pools were not rejected");
    auto before_invalid = backend.export_state();
    invalid = fixture(7, 0);
    invalid.layers[47].full_attention.values = pattern({1, 2, 6, 256}, 2);
    rejected = false;
    try { backend.import_state(invalid); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("bad final-layer geometry was not rejected");
    equal_states(before_invalid, backend.export_state());
    SelfAttention attention(tensors, "language_model.model.layers.3.self_attn",
                            tensors.manifest().config());
    for (int n : {4, 7, 511, 513}) {
        SelfAttentionState state;
        state.token_count = static_cast<std::size_t>(n);
        state.qsa_raw_keys = pattern({1, n, 128}, 3);
        attention.materialize_qsa_pools(state);
        if (state.qsa_pooled_count != static_cast<std::size_t>(n / 4))
            throw std::runtime_error("deferred pools missing");
        auto before = state.qsa_pooled_keys.share();
        attention.materialize_qsa_pools(state);
        equal(before, state.qsa_pooled_keys, "idempotent pool materialization");
    }
    std::cout << "deferred_pool_frontiers=4 idempotent=true malformed_import_rejected=true\n";
}

void pipeline(PersistentMetalBackend& backend) {
    std::vector<PersistentMetalBackend::GreedyResult> control;
    std::optional<ModelDecodeState> reference;
    int pass = 0;
    for (int arm : {0, 1, 0, 1, 1, 0, 0, 1}) {
        std::vector<double> walls, gpu;
        for (int step = 0; step < 64; ++step) {
            const auto started = std::chrono::steady_clock::now();
            PersistentMetalBackend::GreedyResult result;
            if (arm) result = backend.greedy_decode(9419, step == 0);
            else {
                double embed_gpu = 0, trunk_gpu = 0;
                auto embedded = backend.embed(9419, &embed_gpu);
                auto trunk = backend.decode_trunk(9419, embedded, step == 0, &trunk_gpu);
                result = backend.greedy_head(trunk);
                result.gpu_ms += embed_gpu + trunk_gpu;
                result.wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
            }
            if (control.size() < 64) control.push_back(result);
            const auto& expected = control[static_cast<std::size_t>(step)];
            if (result.token != expected.token || result.alternative_token != expected.alternative_token ||
                result.logit != expected.logit) throw std::runtime_error("pipeline token/logit mismatch");
            if (step >= 2) { walls.push_back(result.wall_ms); gpu.push_back(result.gpu_ms); }
        }
        auto state = backend.export_state();
        if (reference) equal_states(*reference, state);
        else reference.emplace(std::move(state));
        const auto array = [](const std::vector<double>& values) {
            std::cout << '[';
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << values[i];
            }
            std::cout << ']';
        };
        std::cout << "{\"arm\":" << arm << ",\"warmup\":"
                  << (pass++ < 2 ? "true" : "false") << ",\"wall_ms\":"; array(walls);
        std::cout << ",\"gpu_ms\":"; array(gpu);
        std::cout << ",\"token_logit_state_exact\":true}\n" << std::flush;
    }
    const auto checkpoint = backend.export_state();
    const auto first = backend.greedy_decode(9419);
    const auto continued = backend.export_state();
    backend.import_state(checkpoint);
    const auto replay = backend.greedy_decode(9419);
    if (first.token != replay.token || first.alternative_token != replay.alternative_token ||
        first.logit != replay.logit) throw std::runtime_error("resume token/logit mismatch");
    equal_states(continued, backend.export_state());
    std::cout << "native_resume_token_logit_state_exact=true\n";
}

void parity(const ModelManifest& manifest) {
    apply_automatic_runtime_config();
    MlxTensorStore tensors(manifest);
    QwenModel model(tensors);
    auto backend = PersistentMetalBackend::create(manifest, &tensors);
    backend->prepare_shared_weights();
    auto state = model.make_state();
    const std::vector<std::uint32_t> prompt(127, 9419);
    static_cast<void>(model.prefill_chunk_batch(prompt, state));
    model.prepare_persistent_state(state);
    backend->import_state(state);
    auto roundtrip = backend->export_state();
    // Normalize raw history to the pending tail for the exact IO comparison.
    for (std::size_t l = 3; l < 48; l += 4) {
        auto& a = state.layers[l].full_attention;
        const int pending = static_cast<int>(a.token_count % 4);
        const int stored = a.qsa_raw_keys.shape()[1];
        a.qsa_raw_keys = a.qsa_raw_keys.slice(std::vector<int>{0, stored - pending, 0},
            std::vector<int>{1, stored, 128}, std::vector<int>{1, 1, 1});
        a.qsa_raw_start = a.token_count - static_cast<std::size_t>(pending);
    }
    equal_states(state, roundtrip);
    std::cout << "real_prefill_state_roundtrip=true tokens=127\n" << std::flush;
    std::uint32_t token = 9419;
    std::size_t matches = 0;
    for (int step = 0; step < 16; ++step) {
        const auto expected = model.greedy_decode(token, state);
        const auto actual = backend->greedy_decode(token);
        matches += expected.token == actual.token;
        std::cout << "parity step=" << step << " mlx=" << expected.token
                  << " native=" << actual.token << " mlx_logit=" << expected.logit
                  << " native_logit=" << actual.logit << '\n' << std::flush;
        token = expected.token;
    }
    std::cout << "teacher_forced_matches=" << matches << "/16\n";
    if (matches != 16) throw std::runtime_error("native backend is not production-parity qualified");
}

void boundary(const ModelManifest& manifest) {
    apply_automatic_runtime_config();
    ::setenv("QWEN38_KV_CACHE", "q8", 1);
    ::setenv("QWEN38_KV_Q8_MIN_TOKENS", "8192", 1);
    ::setenv("QWEN38_KV_Q8_FLUSH_TOKENS", "2048", 1);
    MlxTensorStore tensors(manifest);
    QwenModel model(tensors);
    auto backend = PersistentMetalBackend::create(manifest, &tensors);
    auto state = model.make_state();
    // This is a state-boundary fixture, not a PP benchmark. Bound the retained
    // single-call graph while both backend owners exist; never raise the guard.
    for (int offset = 0; offset < 8191; offset += 512) {
        const std::vector<std::uint32_t> tokens(static_cast<std::size_t>(std::min(512, 8191 - offset)), 9419);
        static_cast<void>(model.prefill_chunk_batch(tokens, state));
    }
    model.prepare_persistent_state(state);
    backend->prepare_shared_weights();
    backend->import_state(state);
    state = model.make_state();
    static_cast<void>(backend->greedy_decode(9419));
    if (backend->can_decode()) throw std::runtime_error("hot slab did not fill at 8192");
    state = backend->export_state();
    backend->release_shared_weights();
    static_cast<void>(model.forward_decode_capture(9419, state));
    if (state.token_count != 8193) throw std::runtime_error("handoff token count mismatch");
    for (std::size_t l = 3; l < 48; l += 4) {
        const auto& a = state.layers[l].full_attention;
        // The first Q8 transition packs the entire BF16 history; the 2,048
        // setting controls subsequent hot-slab flushes, not this transition.
        if (!a.kv_q8 || a.kv_q8_cold_tokens != 8192)
            throw std::runtime_error("MLX handoff did not flush Q8 slab");
    }
    model.prepare_persistent_state(state);
    backend->prepare_shared_weights();
    backend->import_state(state);
    if (!backend->can_decode()) throw std::runtime_error("flushed native state has no capacity");
    static_cast<void>(backend->greedy_decode(9419));
    auto exported = backend->export_state();
    if (exported.token_count != 8194) throw std::runtime_error("reimport continuation failed");
    std::cout << "real_prefill_tokens=8191 native_full=8192 mlx_flushed=8193 cold=8192 native_resumed=8194\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::getenv("QWEN38_MEMORY_GUARD") == nullptr)
            throw std::runtime_error("usage under memory_guard: persistent-state-probe MODEL state|pipeline|parity|boundary");
        const std::string mode(argv[2]);
        if (mode != "state" && mode != "pipeline" && mode != "parity" && mode != "boundary")
            throw std::runtime_error("invalid mode");
        ::setenv("QWEN38_PERSISTENT_VQ", "1", 1);
        static_cast<void>(MlxArray::set_cache_limit(64 * 1024 * 1024));
        const auto manifest = ModelManifest::load(argv[1]);
        if (mode == "parity") { parity(manifest); return 0; }
        if (mode == "boundary") { boundary(manifest); return 0; }
        auto backend = PersistentMetalBackend::create(manifest);
        if (!backend) throw std::runtime_error("unsupported backend");
        std::cout << std::setprecision(10);
        if (mode == "state") {
            MlxTensorStore tensors(manifest);
            state_io(*backend, tensors);
        } else pipeline(*backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
