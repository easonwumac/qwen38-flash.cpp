#include <mlx/c/mlx.h>
#include <libproc.h>

#include <array>
#include <barrier>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <thread>
#include <unistd.h>

namespace {

constexpr std::size_t mib = 1024ULL * 1024ULL;

std::size_t memory(int (*getter)(std::size_t*)) {
    std::size_t value = 0;
    if (getter(&value) != 0) throw std::runtime_error("MLX memory query failed");
    return value;
}

void synchronize_gpu() {
    mlx_stream stream = mlx_default_gpu_stream_new();
    if (stream.ctx == nullptr || mlx_synchronize(stream) != 0) {
        if (stream.ctx != nullptr) mlx_stream_free(stream);
        throw std::runtime_error("MLX GPU synchronization failed");
    }
    mlx_stream_free(stream);
}

mlx_array materialize(const std::size_t bytes, const bool require_active_delta = true) {
    const std::size_t committed_before =
        memory(mlx_get_active_memory) + memory(mlx_get_cache_memory);
    const std::array<int, 1> shape{static_cast<int>(bytes / sizeof(float))};
    mlx_array value{};
    mlx_stream stream = mlx_default_gpu_stream_new();
    if (mlx_ones(&value, shape.data(), shape.size(), MLX_FLOAT32, stream) != 0) {
        mlx_stream_free(stream);
        throw std::runtime_error("mlx_ones failed");
    }
    mlx_vector_array values = mlx_vector_array_new_data(&value, 1);
    const int status = mlx_eval(values);
    const int synchronize_status = status == 0 ? mlx_synchronize(stream) : 0;
    mlx_vector_array_free(values);
    mlx_stream_free(stream);
    if (status != 0 || synchronize_status != 0) {
        mlx_array_free(value);
        throw std::runtime_error("mlx_eval allocation failed");
    }
    const std::size_t committed_after =
        memory(mlx_get_active_memory) + memory(mlx_get_cache_memory);
    if (require_active_delta && committed_after < committed_before + bytes) {
        mlx_array_free(value);
        throw std::runtime_error("materialized allocation was smaller than requested");
    }
    return value;
}

void print_memory(const std::string_view stage) {
    synchronize_gpu();
    rusage_info_v4 usage{};
    const int footprint_status = proc_pid_rusage(
        getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&usage));
    std::cout << stage << " active_mib=" << memory(mlx_get_active_memory) / mib
              << " cache_mib=" << memory(mlx_get_cache_memory) / mib
              << " peak_mib=" << memory(mlx_get_peak_memory) / mib;
    if (footprint_status == 0) {
        std::cout << " footprint_mib=" << usage.ri_phys_footprint / mib;
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    mlx_set_error_handler([](const char*, void*) {}, nullptr, nullptr);
    if (argc != 2 || (std::string_view(argv[1]) != "baseline" &&
                     std::string_view(argv[1]) != "strict")) {
        std::cerr << "usage: qwen38-strict-allocator-probe baseline|strict\n";
        return 1;
    }
    const bool expect_strict = std::string_view(argv[1]) == "strict";
    std::size_t previous = 0;
    if (mlx_set_cache_limit(&previous, 256 * mib) != 0 ||
        mlx_set_memory_limit(&previous, 512 * mib) != 0 ||
        mlx_reset_peak_memory() != 0) {
        std::cerr << "could not configure MLX allocator\n";
        return 2;
    }

    try {
        mlx_array first = materialize(384 * mib);
        print_memory("accepted_384");
        bool rejected_256 = false;
        mlx_array second{};
        try {
            second = materialize(256 * mib);
        } catch (const std::exception&) {
            rejected_256 = true;
        }
        print_memory(rejected_256 ? "rejected_256" : "accepted_256");
        if (second.ctx != nullptr) mlx_array_free(second);

        if (expect_strict != rejected_256) {
            mlx_array_free(first);
            std::cerr << "allocation result did not match mode\n";
            return 3;
        }

        if (expect_strict) {
            constexpr std::size_t import_bytes = 160 * mib;
            void* imported_data = std::aligned_alloc(16384, import_bytes);
            if (imported_data == nullptr) throw std::runtime_error("host import allocation failed");
            std::memset(imported_data, 1, import_bytes);
            const std::array<int, 1> import_shape{
                static_cast<int>(import_bytes / sizeof(float))};
            mlx_array imported = mlx_array_new_data_managed(
                imported_data, import_shape.data(), import_shape.size(), MLX_FLOAT32,
                std::free);
            if (imported.ctx != nullptr) {
                mlx_array_free(imported);
                mlx_array_free(first);
                std::cerr << "managed import bypassed strict limit\n";
                return 8;
            }
            std::free(imported_data);
            print_memory("rejected_160_import");

            std::barrier rendezvous(2);
            std::array<bool, 2> accepted{};
            std::array<std::thread, 2> workers;
            for (std::size_t i = 0; i < workers.size(); ++i) {
                workers[i] = std::thread([&, i] {
                    mlx_array value{};
                    try {
                        value = materialize(96 * mib);
                        accepted[i] = true;
                    } catch (const std::exception&) {
                    }
                    rendezvous.arrive_and_wait();
                    if (value.ctx != nullptr) mlx_array_free(value);
                });
            }
            for (auto& worker : workers) worker.join();
            if (static_cast<int>(accepted[0]) + static_cast<int>(accepted[1]) != 1) {
                mlx_array_free(first);
                std::cerr << "concurrent reservation admitted " << accepted[0] + accepted[1]
                          << " allocations\n";
                return 4;
            }
            print_memory("concurrent_one_of_two");

            mlx_array recovery = materialize(64 * mib, false);
            mlx_array_free(recovery);
            mlx_array reuse = materialize(64 * mib, false);
            mlx_array_free(reuse);
            print_memory("recovery_and_reuse");

            if (mlx_set_memory_limit(&previous, 320 * mib) != 0) {
                mlx_array_free(first);
                return 5;
            }
            bool shrink_rejected = false;
            try {
                mlx_array extra = materialize(1 * mib);
                mlx_array_free(extra);
            } catch (const std::exception&) {
                shrink_rejected = true;
            }
            if (!shrink_rejected) {
                mlx_array_free(first);
                std::cerr << "shrink-below-active allocation was accepted\n";
                return 6;
            }
            print_memory("shrink_rejected");
        }
        mlx_array_free(first);
        synchronize_gpu();
        mlx_clear_cache();
        synchronize_gpu();
        print_memory("done");
        if (memory(mlx_get_active_memory) != 0 || memory(mlx_get_cache_memory) != 0) {
            std::cerr << "allocator did not return to zero\n";
            return 9;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 7;
    }
    return 0;
}
