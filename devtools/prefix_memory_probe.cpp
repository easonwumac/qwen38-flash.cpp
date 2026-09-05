#include "qwen38/decode_state_io.hpp"

#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kLayers = 48;
constexpr std::size_t kPrompt = 32768;
constexpr std::size_t kLive = 65536;
constexpr std::size_t kSharedSteps = 16;

qwen38::MlxArray filled(const std::vector<int>& shape) {
    std::size_t count = 1;
    for (const int extent : shape) count *= static_cast<std::size_t>(extent);
    return qwen38::MlxArray::arange(1.0, static_cast<double>(count) + 1.0, 1.0,
                                   MLX_BFLOAT16).reshape(shape);
}

void set_full(qwen38::SelfAttentionState& state, const std::size_t tokens) {
    const int rows = static_cast<int>(tokens);
    state.keys = filled({1, 2, rows, 256});
    state.values = filled({1, 2, rows, 256});
    state.qsa_raw_keys = filled({1, rows, 128});
    state.qsa_pooled_keys = filled({1, rows / 4, 128});
    state.token_count = tokens;
    state.qsa_pooled_count = tokens / 4;
}

void set_gdn(qwen38::GatedDeltaNetState& state) {
    state.convolution = filled({1, 3, 10240});
    state.recurrent = filled({1, 48, 128, 128});
    state.initialized = true;
}

qwen38::PersistedPrefixState make_state(const std::size_t tokens) {
    qwen38::PersistedPrefixState state(kLayers);
    state.target.token_count = tokens;
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        if ((layer + 1) % 4 == 0) set_full(state.target.layers[layer].full_attention, tokens);
        else set_gdn(state.target.layers[layer].linear_attention);
    }
    state.target.layers[2].ple.convolution = filled({1, 9, 10240});
    state.target.layers[2].ple.convolution_initialized = true;
    set_full(state.mtp.layer.full_attention, tokens);
    state.mtp.row_count = tokens;
    state.mtp.position_base = 0;
    state.previous_target_stream = filled({1, 1, 10240});
    return state;
}

qwen38::PersistedPrefixState snapshot_state(
    const qwen38::PersistedPrefixState& source) {
    qwen38::PersistedPrefixState result(source.target.layers.size());
    result.target = qwen38::snapshot_decode_state(source.target);
    result.mtp = qwen38::snapshot_mtp_decode_state(source.mtp);
    if (source.previous_target_stream)
        result.previous_target_stream = source.previous_target_stream->share();
    result.pending_mtp_tokens = source.pending_mtp_tokens;
    result.mtp_profitable = source.mtp_profitable;
    result.mtp_profitability_current_token =
        source.mtp_profitability_current_token;
    result.mtp_cumulative_profitability_keep =
        source.mtp_cumulative_profitability_keep;
    return result;
}

std::vector<const qwen38::MlxArray*> arrays(
    const qwen38::PersistedPrefixState& state);

void append_full(qwen38::SelfAttentionState& state, const std::size_t rows) {
    const int count = static_cast<int>(rows);
    state.keys = qwen38::MlxArray::concatenate(
        state.keys, filled({1, 2, count, 256}), 2);
    state.values = qwen38::MlxArray::concatenate(
        state.values, filled({1, 2, count, 256}), 2);
    state.qsa_raw_keys = qwen38::MlxArray::concatenate(
        state.qsa_raw_keys, filled({1, count, 128}), 1);
    const std::size_t old_blocks = state.token_count / 4;
    const std::size_t new_blocks = (state.token_count + rows) / 4;
    if (new_blocks != old_blocks) {
        state.qsa_pooled_keys = qwen38::MlxArray::concatenate(
            state.qsa_pooled_keys,
            filled({1, static_cast<int>(new_blocks - old_blocks), 128}), 1);
    }
    state.token_count += rows;
    state.qsa_pooled_count = new_blocks;
    if (new_blocks != old_blocks) {
        const std::array<const qwen38::MlxArray*, 4> changed{
            &state.keys, &state.values, &state.qsa_raw_keys, &state.qsa_pooled_keys};
        qwen38::MlxArray::eval_all(changed);
    } else {
        const std::array<const qwen38::MlxArray*, 3> changed{
            &state.keys, &state.values, &state.qsa_raw_keys};
        qwen38::MlxArray::eval_all(changed);
    }
}

void check_advanced_state(const qwen38::PersistedPrefixState& state) {
    const std::size_t expected_tokens = kPrompt + kSharedSteps;
    const auto bf16 = [](const float value) {
        const std::array<float, 1> data{value};
        const std::array<int, 1> shape{1};
        return qwen38::MlxArray::from_float32(data, shape).astype(MLX_BFLOAT16)
            .astype(MLX_FLOAT32).item_float32();
    };
    const auto sample = [](const qwen38::MlxArray& value, const int index) {
        const std::vector<int> flat{static_cast<int>(value.size())};
        const std::vector<int> start{index}, stop{index + 1}, stride{1};
        return value.reshape(flat).slice(start, stop, stride)
            .astype(MLX_FLOAT32).item_float32();
    };
    const auto check_array = [&](const qwen38::MlxArray& value,
                                 const std::vector<int>& shape,
                                 const float tail) {
        if (value.shape() != shape || value.dtype() != MLX_BFLOAT16 ||
            sample(value, 7) != bf16(8.0F) ||
            sample(value, static_cast<int>(value.size()) - 1) != bf16(tail))
            throw std::runtime_error("advanced array value/metadata mismatch");
    };
    if (state.target.token_count != expected_tokens ||
        state.mtp.row_count != expected_tokens || state.target.layers.size() != kLayers)
        throw std::runtime_error("advanced timeline metadata mismatch");
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        const auto& current = state.target.layers[layer];
        if ((layer + 1) % 4 == 0) {
            const auto& full = current.full_attention;
            if (full.token_count != expected_tokens || full.position_base != 0 ||
                full.qsa_pooled_count != expected_tokens / 4)
                throw std::runtime_error("advanced full-attention metadata mismatch");
            check_array(full.keys, {1, 2, static_cast<int>(expected_tokens), 256}, 512);
            check_array(full.values, {1, 2, static_cast<int>(expected_tokens), 256}, 512);
            check_array(full.qsa_raw_keys,
                        {1, static_cast<int>(expected_tokens), 128}, 128);
            check_array(full.qsa_pooled_keys,
                        {1, static_cast<int>(expected_tokens / 4), 128}, 128);
        } else {
            const auto& gdn = current.linear_attention;
            if (!gdn.initialized) throw std::runtime_error("advanced GDN metadata mismatch");
            check_array(gdn.convolution, {1, 3, 10240}, 30720);
            check_array(gdn.recurrent, {1, 48, 128, 128}, 786432);
        }
    }
    check_array(state.target.layers[2].ple.convolution, {1, 9, 10240}, 92160);
    const auto& mtp = state.mtp.layer.full_attention;
    if (mtp.token_count != expected_tokens || mtp.position_base != 0 ||
        mtp.qsa_pooled_count != expected_tokens / 4)
        throw std::runtime_error("advanced MTP metadata mismatch");
    check_array(mtp.keys, {1, 2, static_cast<int>(expected_tokens), 256}, 512);
    check_array(mtp.values, {1, 2, static_cast<int>(expected_tokens), 256}, 512);
    check_array(mtp.qsa_raw_keys, {1, static_cast<int>(expected_tokens), 128}, 128);
    check_array(mtp.qsa_pooled_keys,
                {1, static_cast<int>(expected_tokens / 4), 128}, 128);
    if (!state.previous_target_stream)
        throw std::runtime_error("previous target stream missing");
    check_array(*state.previous_target_stream, {1, 1, 10240}, 10240);
}

void advance_state(qwen38::PersistedPrefixState& state, const std::size_t rows) {
    for (std::size_t layer = 0; layer < state.target.layers.size(); ++layer) {
        auto& current = state.target.layers[layer];
        if ((layer + 1) % 4 == 0) append_full(current.full_attention, rows);
        else {
            set_gdn(current.linear_attention);
            const std::array<const qwen38::MlxArray*, 2> changed{
                &current.linear_attention.convolution,
                &current.linear_attention.recurrent};
            qwen38::MlxArray::eval_all(changed);
        }
    }
    state.target.token_count += rows;
    state.target.layers[2].ple.convolution = filled({1, 9, 10240});
    state.target.layers[2].ple.convolution.eval();
    append_full(state.mtp.layer.full_attention, rows);
    state.mtp.row_count += rows;
}

void collect(const qwen38::DecoderLayerState& layer,
             std::vector<const qwen38::MlxArray*>& arrays) {
    const auto& g = layer.linear_attention;
    if (g.initialized) {
        arrays.push_back(&g.convolution);
        arrays.push_back(&g.recurrent);
    }
    const auto& a = layer.full_attention;
    if (a.token_count != 0) {
        arrays.insert(arrays.end(), {&a.keys, &a.values, &a.qsa_raw_keys,
                                     &a.qsa_pooled_keys});
    }
    if (layer.ple.convolution_initialized) arrays.push_back(&layer.ple.convolution);
}

std::vector<const qwen38::MlxArray*> arrays(const qwen38::PersistedPrefixState& state) {
    std::vector<const qwen38::MlxArray*> result;
    for (const auto& layer : state.target.layers) collect(layer, result);
    collect(state.mtp.layer, result);
    if (state.previous_target_stream) result.push_back(&*state.previous_target_stream);
    return result;
}

std::size_t logical_bytes(const qwen38::PersistedPrefixState& state) {
    std::size_t total = 0;
    for (const auto* array : arrays(state)) total += array->size() * 2;
    return total;
}

std::uint64_t footprint() {
    rusage_info_v4 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                        reinterpret_cast<rusage_info_t*>(&info)) != 0) {
        throw std::runtime_error("proc_pid_rusage failed");
    }
    return info.ri_phys_footprint;
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        std::string pattern = (std::filesystem::temp_directory_path() /
                               "qwen38-prefix-probe.XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* made = mkdtemp(buffer.data());
        if (!made) throw std::runtime_error("mkdtemp failed");
        path = std::filesystem::canonical(made);
    }
    ~TempDir() {
        std::error_code error;
        const auto parent = std::filesystem::canonical(path.parent_path(), error);
        if (!error && parent == std::filesystem::canonical(
                std::filesystem::temp_directory_path(), error) &&
            path.filename().string().starts_with("qwen38-prefix-probe.")) {
            std::filesystem::remove_all(path, error);
        }
    }
};

void check_parity(const qwen38::PersistedPrefixState& left,
                  const qwen38::PersistedPrefixState& right) {
    const auto a = arrays(left);
    const auto b = arrays(right);
    if (left.target.token_count != right.target.token_count || a.size() != b.size())
        throw std::runtime_error("state metadata parity failed");
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i]->shape() != b[i]->shape() || a[i]->dtype() != b[i]->dtype() ||
            a[i]->size() != b[i]->size())
            throw std::runtime_error("array metadata parity failed");
        const int size = static_cast<int>(a[i]->size());
        const auto sample = [size](const qwen38::MlxArray& value, const int index) {
            const std::vector<int> flat{size};
            const std::vector<int> start{index};
            const std::vector<int> stop{index + 1};
            const std::vector<int> stride{1};
            return value.reshape(flat).slice(start, stop, stride)
                .astype(MLX_FLOAT32).item_float32();
        };
        const float a_first = sample(*a[i], 0);
        const float b_first = sample(*b[i], 0);
        const float a_last = sample(*a[i], size - 1);
        const float b_last = sample(*b[i], size - 1);
        if (a_first != b_first || a_last != b_last)
            throw std::runtime_error("array sampled-value parity failed at " +
                                     std::to_string(i) + ": " +
                                     std::to_string(a_first) + "/" +
                                     std::to_string(b_first) + ", " +
                                     std::to_string(a_last) + "/" +
                                     std::to_string(b_last));
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    const bool shared = mode == "shared-baseline" || mode == "shared-release";
    if (argc != 2 || (mode != "baseline" && mode != "release" && !shared)) {
        std::cerr << "usage: qwen38-prefix-memory-probe "
                     "baseline|release|shared-baseline|shared-release\n";
        return 2;
    }
    const bool release = mode == "release" || mode == "shared-release";
    static_cast<void>(qwen38::MlxArray::set_cache_limit(256ULL * 1024ULL * 1024ULL));
    std::cout << std::unitbuf;
    TempDir temporary;
    auto prompt = make_state(kPrompt);
    qwen38::MlxArray::eval_all(arrays(prompt));
    const auto prompt_footprint = footprint();
    std::cout << "sample=prompt footprint=" << prompt_footprint << '\n';
    double save_ms = 0.0;
    double load_ms = 0.0;
    if (release) {
        const auto file = temporary.path / "prefix.safetensors";
        auto started = std::chrono::steady_clock::now();
        qwen38::save_prefix_state(file, prompt);
        save_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        if (std::filesystem::file_size(file) > 3ULL * 1024 * 1024 * 1024)
            throw std::runtime_error("probe state exceeds 3 GiB disk bound");
        started = std::chrono::steady_clock::now();
        auto loaded = qwen38::load_prefix_state(file, kLayers);
        qwen38::MlxArray::eval_all(arrays(loaded));
        load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        check_parity(prompt, loaded);
        std::cout << "sample=saved footprint=" << footprint() << '\n';
        loaded = qwen38::PersistedPrefixState(0);
        if (!shared) {
            prompt = qwen38::PersistedPrefixState(0);
            qwen38::MlxArray::clear_cache();
        }
    }
    if (shared) {
        auto prefix_owner = snapshot_state(prompt);
        auto prompt_owner = snapshot_state(prompt);
        auto live = snapshot_state(prompt);
        std::cout << "sample=shared footprint=" << footprint() << '\n';
        if (release) {
            prefix_owner = qwen38::PersistedPrefixState(0);
            prompt_owner = qwen38::PersistedPrefixState(0);
            prompt = qwen38::PersistedPrefixState(0);
            std::cout << "sample=released footprint=" << footprint() << '\n';
        }
        for (std::size_t step = 0; step < kSharedSteps; ++step) {
            const auto step_started = std::chrono::steady_clock::now();
            advance_state(live, 1);
            std::cout << "sample=step" << (step + 1)
                      << " footprint=" << footprint()
                      << " step_ms=" << std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - step_started).count()
                      << '\n';
        }
        check_advanced_state(live);
        std::cout << "mode=" << mode
                  << " live_logical_bytes=" << logical_bytes(live)
                  << " final_footprint=" << footprint()
                  << " save_ms=" << save_ms << " load_ms=" << load_ms
                  << " parity=pass\n";
        return 0;
    }
    auto live = make_state(kLive);
    qwen38::MlxArray::eval_all(arrays(live));
    const auto peak = footprint();
    std::cout << "mode=" << argv[1]
              << " prompt_logical_bytes=" << (release ? 0 : logical_bytes(prompt))
              << " live_logical_bytes=" << logical_bytes(live)
              << " prompt_footprint=" << prompt_footprint
              << " final_footprint=" << peak
              << " save_ms=" << save_ms << " load_ms=" << load_ms
              << " parity=pass\n";
}
