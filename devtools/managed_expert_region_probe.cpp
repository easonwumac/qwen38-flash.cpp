#include "qwen38/safetensors.hpp"

#include <mlx/c/mlx.h>
#include <libproc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t mib = 1024ULL * 1024ULL;
constexpr std::size_t strict_limit = 512ULL * mib;
constexpr std::size_t cache_limit = 16ULL * mib;
constexpr std::size_t failure_bytes = 1ULL * mib;

struct CleanupState {
    std::atomic<std::size_t> callbacks{0};
    std::atomic<std::size_t> manual{0};
};

struct Payload {
    std::shared_ptr<const qwen38::SafetensorsFile> owner;
    std::shared_ptr<CleanupState> state;
};

void release_payload(void* opaque) noexcept {
    auto* payload = static_cast<Payload*>(opaque);
    payload->state->callbacks.fetch_add(1, std::memory_order_relaxed);
    delete payload;
}

struct Imported {
    mlx_array value{};
    std::string label;
    std::size_t bytes{0};
    mlx_dtype type{MLX_UINT8};
    std::vector<int> shape;
    std::vector<std::byte> oracle;
    bool alias{false};
    std::shared_ptr<CleanupState> cleanup;

    Imported() = default;
    Imported(const Imported&) = delete;
    Imported& operator=(const Imported&) = delete;
    Imported(Imported&& other) noexcept
        : value(std::exchange(other.value, mlx_array{})),
          label(std::move(other.label)),
          bytes(other.bytes),
          type(other.type),
          shape(std::move(other.shape)),
          oracle(std::move(other.oracle)),
          alias(other.alias),
          cleanup(std::move(other.cleanup)) {}
    Imported& operator=(Imported&& other) noexcept {
        if (this != &other) {
            if (value.ctx != nullptr) static_cast<void>(mlx_array_free(value));
            value = std::exchange(other.value, mlx_array{});
            label = std::move(other.label);
            bytes = other.bytes;
            type = other.type;
            shape = std::move(other.shape);
            oracle = std::move(other.oracle);
            alias = other.alias;
            cleanup = std::move(other.cleanup);
        }
        return *this;
    }
    ~Imported() {
        if (value.ctx != nullptr) static_cast<void>(mlx_array_free(value));
    }
};

struct Projection {
    std::string name;
    std::shared_ptr<const qwen38::SafetensorsFile> owner;
    qwen38::TensorView weight;
    qwen38::TensorView scales;
    qwen38::TensorView biases;
    mlx_dtype weight_dtype{MLX_UINT32};
    mlx_dtype scales_dtype{MLX_BFLOAT16};
    mlx_dtype biases_dtype{MLX_BFLOAT16};
};

void check(const int status, const char* operation) {
    if (status != 0) throw std::runtime_error(std::string("MLX failed: ") + operation);
}

std::size_t memory(int (*getter)(std::size_t*)) {
    std::size_t value = 0;
    check(getter(&value), "memory query");
    return value;
}

mlx_stream gpu_stream() {
    mlx_stream stream = mlx_default_gpu_stream_new();
    if (stream.ctx == nullptr) throw std::runtime_error("MLX did not provide a GPU stream");
    return stream;
}

void synchronize_gpu() {
    mlx_stream stream = gpu_stream();
    const int status = mlx_synchronize(stream);
    static_cast<void>(mlx_stream_free(stream));
    check(status, "GPU synchronize");
}

rusage_info_v4 process_usage() {
    rusage_info_v4 usage{};
    if (proc_pid_rusage(
            getpid(), RUSAGE_INFO_V4,
            reinterpret_cast<rusage_info_t*>(&usage)) != 0) {
        throw std::runtime_error("could not measure process memory");
    }
    return usage;
}

void print_memory(const std::string_view stage) {
    synchronize_gpu();
    std::cout << stage
              << " active_mib=" << memory(mlx_get_active_memory) / mib
              << " cache_mib=" << memory(mlx_get_cache_memory) / mib
              << " peak_mib=" << memory(mlx_get_peak_memory) / mib;
    const auto usage = process_usage();
    std::cout << " footprint_mib=" << usage.ri_phys_footprint / mib
              << " lifetime_peak_footprint_mib="
              << usage.ri_lifetime_max_phys_footprint / mib
              << " rss_mib=" << usage.ri_resident_size / mib;
    std::cout << '\n';
}

mlx_dtype dtype(std::string_view value) {
    if (value == "U8") return MLX_UINT8;
    if (value == "U16") return MLX_UINT16;
    if (value == "U32") return MLX_UINT32;
    if (value == "BF16") return MLX_BFLOAT16;
    if (value == "F16") return MLX_FLOAT16;
    if (value == "F32") return MLX_FLOAT32;
    throw std::runtime_error("unsupported probe dtype: " + std::string(value));
}

std::size_t elements(std::span<const std::size_t> shape) {
    std::size_t result = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0 || result > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("tensor shape overflows");
        }
        result *= dimension;
    }
    return result;
}

void validate_tensor(const qwen38::TensorView& view, const char* label) {
    const std::size_t item_size = mlx_dtype_size(dtype(view.dtype));
    const std::size_t count = elements(view.shape);
    if (view.shape.size() != 3 || view.shape[0] == 0 || view.shape[1] == 0 ||
        view.shape[2] == 0 || item_size == 0 ||
        count > std::numeric_limits<std::size_t>::max() / item_size ||
        count * item_size != view.bytes.size()) {
        throw std::runtime_error(std::string("invalid layer-0 tensor geometry: ") + label);
    }
}

struct Window {
    const std::byte* begin{nullptr};
    std::size_t bytes{0};
    std::size_t delta{0};
};

Window page_window(const qwen38::TensorView& view, const std::size_t expert) {
    validate_tensor(view, "page-window tensor");
    const std::size_t expert_bytes = view.bytes.size() / view.shape[0];
    if (expert >= view.shape[0]) {
        throw std::runtime_error("expert offset is out of range");
    }
    const std::size_t offset = expert * expert_bytes;
    const auto* view_begin = view.bytes.data();
    const auto* view_end = view_begin + view.bytes.size();
    const auto* source_begin = view_begin + offset;
    const auto* source_end = source_begin + expert_bytes;
    const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    if (page == 0 || (page & (page - 1)) != 0) {
        throw std::runtime_error("unsupported non-power-of-two page size");
    }
    const auto source_address = reinterpret_cast<std::uintptr_t>(source_begin);
    const auto end_address = reinterpret_cast<std::uintptr_t>(source_end);
    const auto aligned_begin = source_address & ~(static_cast<std::uintptr_t>(page) - 1U);
    if (end_address > std::numeric_limits<std::uintptr_t>::max() - page + 1U) {
        throw std::runtime_error("page-window address overflow");
    }
    const auto aligned_end =
        (end_address + static_cast<std::uintptr_t>(page) - 1U) &
        ~(static_cast<std::uintptr_t>(page) - 1U);
    const auto* begin = reinterpret_cast<const std::byte*>(aligned_begin);
    const auto* end = reinterpret_cast<const std::byte*>(aligned_end);
    if (begin < view_begin || end > view_end || begin > source_begin ||
        end < source_end || aligned_begin % page != 0 || aligned_end % page != 0) {
        throw std::runtime_error("expert page window exceeds tensor view; refusing out-of-file bytes");
    }
    return {
        .begin = begin,
        .bytes = static_cast<std::size_t>(end - begin),
        .delta = static_cast<std::size_t>(source_begin - begin),
    };
}

Imported managed_import(
    const std::string& label,
    const std::byte* source,
    const std::size_t bytes,
    const std::vector<int>& shape,
    const mlx_dtype type,
    const std::shared_ptr<const qwen38::SafetensorsFile>& owner,
    const std::shared_ptr<CleanupState>& cleanup,
    std::vector<std::shared_ptr<CleanupState>>& cleanup_states) {
    if (source == nullptr || bytes == 0 || shape.empty() || shape.size() > 3) {
        throw std::runtime_error("invalid managed import request: " + label);
    }
    cleanup_states.push_back(cleanup);
    auto* payload = new Payload{owner, cleanup};
    mlx_array value = mlx_array_new_data_managed_payload(
        const_cast<std::byte*>(source), shape.data(), static_cast<int>(shape.size()), type,
        payload, release_payload);
    if (value.ctx == nullptr) {
        if (cleanup->callbacks.load(std::memory_order_acquire) == 0) {
            delete payload;
            cleanup->manual.fetch_add(1, std::memory_order_relaxed);
        }
        throw std::runtime_error("managed import refused: " + label);
    }
    if (mlx_array_nbytes(value) != bytes) {
        static_cast<void>(mlx_array_free(value));
        throw std::runtime_error("managed import byte count mismatch: " + label);
    }
    Imported result;
    result.value = value;
    result.label = label;
    result.bytes = bytes;
    result.type = type;
    result.shape = shape;
    result.oracle.assign(source, source + bytes);
    result.cleanup = cleanup;
    return result;
}

const void* array_data(const mlx_array value, const mlx_dtype type) {
    switch (type) {
        case MLX_UINT8: return mlx_array_data_uint8(value);
        case MLX_UINT16: return mlx_array_data_uint16(value);
        case MLX_UINT32: return mlx_array_data_uint32(value);
        case MLX_BFLOAT16: return mlx_array_data_bfloat16(value);
        case MLX_FLOAT16: return mlx_array_data_float16(value);
        case MLX_FLOAT32: return mlx_array_data_float32(value);
        default: throw std::runtime_error("unsupported alias dtype");
    }
}

void verify_gpu_bytes(
    const mlx_array value,
    const std::byte* source,
    const std::size_t bytes,
    const std::vector<int>& shape,
    const mlx_dtype type,
    const std::string_view label) {
    mlx_array reference = mlx_array_new_data(
        source, shape.data(), static_cast<int>(shape.size()), type);
    if (reference.ctx == nullptr) throw std::runtime_error("reference array failed: " + std::string(label));
    mlx_array equal{};
    mlx_stream stream = gpu_stream();
    const int equal_status = mlx_array_equal(&equal, value, reference, false, stream);
    if (equal_status != 0) {
        if (equal.ctx != nullptr) static_cast<void>(mlx_array_free(equal));
        static_cast<void>(mlx_array_free(reference));
        static_cast<void>(mlx_stream_free(stream));
        throw std::runtime_error("GPU byte comparison setup failed: " + std::string(label));
    }
    const int eval_status = mlx_array_eval(equal);
    const int synchronize_status = eval_status == 0 ? mlx_synchronize(stream) : 0;
    bool matches = false;
    const int item_status = eval_status == 0 && synchronize_status == 0
        ? mlx_array_item_bool(&matches, equal) : -1;
    static_cast<void>(mlx_array_free(equal));
    const int free_reference_status = mlx_array_free(reference);
    static_cast<void>(mlx_stream_free(stream));
    if (eval_status != 0 || synchronize_status != 0 || item_status != 0 ||
        free_reference_status != 0 || !matches) {
        throw std::runtime_error("GPU byte comparison failed: " + std::string(label));
    }
    const void* actual = array_data(value, type);
    if (actual == nullptr && bytes != 0) throw std::runtime_error("array data unavailable: " + std::string(label));
}

int mlx_dimension(const std::size_t value, const std::string_view label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MLX dimension overflows int: " + std::string(label));
    }
    return static_cast<int>(value);
}

void check_cleanup(
    const std::vector<std::shared_ptr<CleanupState>>& states,
    const std::size_t expected_successes,
    const std::shared_ptr<CleanupState>& failure_state) {
    std::size_t success_callbacks = 0;
    for (const auto& state : states) {
        if (state == failure_state) continue;
        if (state->callbacks.load(std::memory_order_acquire) != 1 ||
            state->manual.load(std::memory_order_acquire) != 0) {
            throw std::runtime_error("successful payload was not cleaned exactly once");
        }
        ++success_callbacks;
    }
    if (success_callbacks != expected_successes ||
        failure_state->callbacks.load(std::memory_order_acquire) != 0 ||
        failure_state->manual.load(std::memory_order_acquire) != 1) {
        throw std::runtime_error("failed payload cleanup was not exactly once");
    }
}

Projection load_projection(
    const std::shared_ptr<const qwen38::SafetensorsFile>& owner,
    const std::string_view base) {
    Projection result;
    result.name = std::string(base);
    result.owner = owner;
    result.weight = owner->tensor(std::string(base) + ".weight");
    result.scales = owner->tensor(std::string(base) + ".scales");
    result.biases = owner->tensor(std::string(base) + ".biases");
    result.weight_dtype = dtype(result.weight.dtype);
    result.scales_dtype = dtype(result.scales.dtype);
    result.biases_dtype = dtype(result.biases.dtype);
    for (const auto* view : {&result.weight, &result.scales, &result.biases}) {
        validate_tensor(*view, result.name.c_str());
        if (view->shape[0] != result.weight.shape[0]) {
            throw std::runtime_error("projection expert counts disagree: " + result.name);
        }
    }
    return result;
}

std::shared_ptr<CleanupState> run_failure_probe(
    const std::shared_ptr<const qwen38::SafetensorsFile>& owner,
    std::vector<std::shared_ptr<CleanupState>>& cleanup_states,
    const std::size_t restore_limit) {
    const std::size_t active = memory(mlx_get_active_memory);
    std::size_t previous = 0;
    check(mlx_set_memory_limit(&previous, active), "set refusal limit");
    void* backing = std::aligned_alloc(16384, failure_bytes);
    if (backing == nullptr) throw std::runtime_error("failure backing allocation failed");
    std::memset(backing, 0xA5, failure_bytes);
    auto failure_state = std::make_shared<CleanupState>();
    const std::array<int, 1> shape{static_cast<int>(failure_bytes)};
    auto* payload = new Payload{owner, failure_state};
    cleanup_states.push_back(failure_state);
    mlx_array refused = mlx_array_new_data_managed_payload(
        static_cast<std::byte*>(backing), shape.data(), 1, MLX_UINT8,
        payload, release_payload);
    if (refused.ctx != nullptr) {
        static_cast<void>(mlx_array_free(refused));
        std::free(backing);
        check(mlx_set_memory_limit(&previous, restore_limit), "restore refusal limit");
        throw std::runtime_error("strict managed refusal unexpectedly succeeded");
    }
    if (failure_state->callbacks.load(std::memory_order_acquire) == 0) {
        delete payload;
        failure_state->manual.fetch_add(1, std::memory_order_relaxed);
    }
    std::free(backing);
    check(mlx_set_memory_limit(&previous, restore_limit), "restore memory limit");
    std::cout << "refused_import bytes=" << failure_bytes
              << " callbacks=" << failure_state->callbacks.load()
              << " manual=" << failure_state->manual.load() << '\n';
    return failure_state;
}

} // namespace

int run_probe(int argc, char** argv) {
    mlx_set_error_handler([](const char*, void*) {}, nullptr, nullptr);
    if (argc > 1 && std::string_view(argv[1]) == "--help") {
        std::cout << "usage: qwen38-managed-expert-region-probe MODEL_DIRECTORY [--rounds 1..10]\n";
        return 0;
    }
    if (argc != 2 || std::getenv("MLX_STRICT_MEMORY_LIMIT") == nullptr ||
        std::string_view(std::getenv("MLX_STRICT_MEMORY_LIMIT")) != "1") {
        std::cerr << "usage: MLX_STRICT_MEMORY_LIMIT=1 qwen38-managed-expert-region-probe MODEL_DIRECTORY\n";
        return 2;
    }
    std::size_t previous = 0;
    if (mlx_set_cache_limit(&previous, cache_limit) != 0 ||
        mlx_set_memory_limit(&previous, strict_limit) != 0 ||
        mlx_reset_peak_memory() != 0) {
        std::cerr << "could not configure strict MLX allocator\n";
        return 3;
    }

    try {
        const std::filesystem::path model_directory = std::filesystem::canonical(argv[1]);
        auto gate_catalog = std::make_shared<qwen38::SafetensorsFile>(
            model_directory / "model-00002-of-00131.safetensors");
        auto down_catalog = std::make_shared<qwen38::SafetensorsFile>(
            model_directory / "model-00003-of-00131.safetensors");
        const std::weak_ptr<const qwen38::SafetensorsFile> gate_weak = gate_catalog;
        const std::weak_ptr<const qwen38::SafetensorsFile> down_weak = down_catalog;
        constexpr std::array<std::string_view, 2> gate_names{
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
            "language_model.model.layers.0.mlp.switch_mlp.up_proj",
        };
        constexpr std::string_view down_name =
            "language_model.model.layers.0.mlp.switch_mlp.down_proj";
        std::vector<Projection> projections;
        projections.reserve(3);
        for (const std::string_view name : gate_names) {
            projections.push_back(load_projection(gate_catalog, name));
        }
        projections.push_back(load_projection(down_catalog, down_name));
        const std::size_t expert_count = projections.front().weight.shape[0];
        for (const Projection& projection : projections) {
            if (projection.weight.shape[0] != expert_count) {
                throw std::runtime_error("layer-0 projection expert count mismatch");
            }
        }
        std::cout << "geometry experts=" << expert_count
                  << " gate_weight_bytes_per_expert="
                  << projections.front().weight.bytes.size() / expert_count
                  << " down_weight_bytes_per_expert="
                  << projections.back().weight.bytes.size() / expert_count
                  << "\n";

        std::vector<Imported> imports;
        std::vector<std::shared_ptr<CleanupState>> cleanup_states;
        std::size_t alias_count = 0;
        std::size_t copy_count = 0;
        for (const std::size_t count : {std::size_t{1}, std::size_t{8}, std::size_t{10}}) {
            const std::size_t first_expert = 1;
            if (count + first_expert > expert_count) throw std::runtime_error("fixture lacks requested experts");
            for (const Projection& projection : projections) {
                for (std::size_t expert = first_expert; expert < first_expert + count; ++expert) {
                    const std::string prefix = projection.name + ".expert" + std::to_string(expert) + ".U" + std::to_string(count);
                    for (const auto [suffix, view] : {
                             std::pair<std::string_view, const qwen38::TensorView&>{"weight", projection.weight},
                             {"scales", projection.scales},
                             {"biases", projection.biases}}) {
                        const Window window = page_window(view, expert);
                        const auto window_cleanup = std::make_shared<CleanupState>();
                        Imported imported_window = managed_import(
                            prefix + "." + std::string(suffix) + ".window", window.begin, window.bytes,
                            {mlx_dimension(window.bytes, prefix + "." + std::string(suffix))},
                            MLX_UINT8, projection.owner, window_cleanup,
                            cleanup_states);
                        imported_window.alias = array_data(imported_window.value, MLX_UINT8) == window.begin;
                        verify_gpu_bytes(
                            imported_window.value, imported_window.oracle.data(), window.bytes,
                            {mlx_dimension(window.bytes, prefix + "." + std::string(suffix))}, MLX_UINT8,
                            prefix + "." + std::string(suffix) + ".window");
                        alias_count += imported_window.alias ? 1U : 0U;
                        copy_count += imported_window.alias ? 0U : 1U;
                        imports.push_back(std::move(imported_window));

                        const Imported& window_import = imports.back();
                        const std::size_t expert_bytes = view.bytes.size() / view.shape[0];
                        const std::array<int, 1> slice_start{
                            mlx_dimension(window.delta, prefix + ".slice_start")};
                        const std::array<int, 1> slice_stop{
                            mlx_dimension(window.delta + expert_bytes, prefix + ".slice_stop")};
                        const std::array<int, 1> slice_stride{1};
                        mlx_array exact_window{};
                        mlx_stream slice_stream = gpu_stream();
                        check(mlx_slice(
                                  &exact_window, window_import.value,
                                  slice_start.data(), 1, slice_stop.data(), 1,
                                  slice_stride.data(), 1, slice_stream),
                            "slice page window");
                        verify_gpu_bytes(
                            exact_window, window_import.oracle.data() + window.delta,
                            expert_bytes, {mlx_dimension(expert_bytes, prefix + ".slice")}, MLX_UINT8,
                            prefix + "." + std::string(suffix) + ".window.exact");
                        static_cast<void>(mlx_array_free(exact_window));
                        static_cast<void>(mlx_stream_free(slice_stream));
                    }
                }
            }
            print_memory(std::string("U") + std::to_string(count) + "_loaded");
        }
        std::cout << "imports=" << imports.size() << " alias=" << alias_count
                  << " copy=" << copy_count << '\n';
        if (process_usage().ri_lifetime_max_phys_footprint > 1024ULL * mib) {
            throw std::runtime_error("probe exceeded 1 GiB lifetime footprint guard");
        }

        for (Projection& projection : projections) projection.owner.reset();
        gate_catalog.reset();
        down_catalog.reset();
        for (const Imported& imported : imports) {
            // Enqueue a fresh GPU comparison against the copied oracle after
            // dropping catalog owners. Each managed payload must retain the
            // mapping independently.
            verify_gpu_bytes(
                imported.value, imported.oracle.data(), imported.bytes, imported.shape, imported.type,
                imported.label + ".survival");
        }
        print_memory("catalog_dropped");
        const auto failure_state = run_failure_probe({}, cleanup_states, strict_limit);
        const std::size_t expected_successes = imports.size();
        imports.clear();
        synchronize_gpu();
        if (!gate_weak.expired() || !down_weak.expired()) {
            throw std::runtime_error("safetensors mapping owner survived final array release");
        }
        check_cleanup(cleanup_states, expected_successes, failure_state);
        print_memory("arrays_released");
        check(mlx_clear_cache(), "clear allocator cache");
        print_memory("done");
        if (memory(mlx_get_active_memory) != 0 || memory(mlx_get_cache_memory) != 0) {
            throw std::runtime_error("strict allocator did not return to zero");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 7;
    }
}

int main(int argc, char** argv) {
    int rounds = 1;
    if (argc == 4 && std::string_view(argv[2]) == "--rounds") {
        const std::string_view raw(argv[3]);
        const auto parsed = std::from_chars(raw.data(), raw.data() + raw.size(), rounds);
        if (parsed.ec != std::errc{} || parsed.ptr != raw.data() + raw.size() ||
            rounds < 1 || rounds > 10) {
            std::cerr << "rounds must be between 1 and 10\n";
            return 2;
        }
        argc = 2;
    }
    for (int round = 0; round < rounds; ++round) {
        std::cout << "round=" << round + 1 << '/' << rounds << '\n';
        const int status = run_probe(argc, argv);
        if (status != 0) return status;
    }
    return 0;
}
