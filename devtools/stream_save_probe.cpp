#include "qwen38/mlx_backend.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace qwen38;

int main(int argc, char** argv) try {
    if ((argc != 3 && argc != 4) || !std::getenv("QWEN38_MEMORY_GUARD"))
        throw std::runtime_error("usage: guarded stream-save-probe batch|stream|group4|group12 NEW_OUTPUT_FILE");
    const int group_size = std::string_view(argv[1]) == "group12" ? 12 :
        (std::string_view(argv[1]) == "group4" ? 4 : 1);
    const bool stream_mode = std::string_view(argv[1]) == "stream" || group_size != 1;
    const bool warm = argc == 4 && std::string_view(argv[3]) == "warm";
    if (argc == 4 && !warm) throw std::runtime_error("invalid warmup option");
    if (!stream_mode && std::string_view(argv[1]) != "batch") throw std::runtime_error("invalid mode");
    if (std::filesystem::exists(argv[2])) throw std::runtime_error("refusing existing output");
    std::size_t old{};
    if (mlx_set_memory_limit(&old, 1024ULL * 1024 * 1024)) throw std::runtime_error("cap failed");
    static_cast<void>(MlxArray::set_cache_limit(16ULL * 1024 * 1024));
    constexpr int count = 12, rows = 8192, width = 128, capacity = rows + 128;
    constexpr std::size_t bytes = 2ULL * rows * width * sizeof(float);
    std::vector<MlxArray> banks;
    for (int bank = 0; bank < count; ++bank) {
        std::vector<float> data(2 * capacity * width);
        for (std::size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<float>((i + bank * 19) % 127) / 128.F;
        auto backing = MlxArray::from_float32(data, std::array<int, 3>{2, capacity, width});
        backing.eval();
        banks.push_back(backing.slice(std::array<int, 3>{0, 0, 0},
            std::array<int, 3>{2, rows, width}, std::array<int, 3>{1, 1, 1}));
        banks.back().eval();
    }
    std::vector<MlxSafetensors::NamedArray> named;
    for (int bank = 0; bank < count; ++bank)
        named.push_back({"bank" + std::to_string(bank), &banks[bank]});
    const std::array<std::pair<std::string, std::string>, 1> metadata{{{"probe", "strided-f32"}}};
    if (warm) {
        const auto stream = mlx_default_gpu_stream_new();
        mlx_array raw = mlx_array_new();
        const int status = mlx_contiguous(&raw, banks.front().get(), false, stream);
        MlxArray contiguous(raw);
        if (status) throw std::runtime_error("warmup failed");
        contiguous.eval();
        mlx_stream_free(stream);
    }
    if (mlx_reset_peak_memory()) throw std::runtime_error("peak reset failed");
    const auto started = std::chrono::steady_clock::now();
    if (!stream_mode) MlxSafetensors::save(argv[2], named, metadata);
    else {
        // Narrow format prototype: fixed F32 fixtures only. This is not the
        // production serializer and does not implement atomic checkpoint save.
        std::string header = "{\"__metadata__\":{\"probe\":\"strided-f32\"}";
        for (int bank = 0; bank < count; ++bank)
            header += ",\"bank" + std::to_string(bank) + "\":{\"dtype\":\"F32\",\"shape\":[2,8192,128],\"data_offsets\":[" +
                std::to_string(bank * bytes) + "," + std::to_string((bank + 1) * bytes) + "]}";
        header += "}";
        while (header.size() % 8) header += ' ';
        std::ofstream file(argv[2], std::ios::binary);
        const std::uint64_t size = header.size();
        // Apple Silicon is little-endian, as required by safetensors.
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        const auto stream = mlx_default_gpu_stream_new();
        for (int start = 0; start < count; start += group_size) {
            std::vector<MlxArray> group;
            std::vector<const MlxArray*> outputs;
            group.reserve(group_size);
            for (int j = 0; j < group_size; ++j) {
                mlx_array raw = mlx_array_new();
                const int status = mlx_contiguous(&raw, banks[start + j].get(), false, stream);
                group.emplace_back(raw);
                if (status) throw std::runtime_error("contiguous failed");
                outputs.push_back(&group.back());
            }
            MlxArray::eval_all(outputs);
            for (const auto& contiguous : group) {
                const auto* data = mlx_array_data_float32(contiguous.get());
                if (!data) throw std::runtime_error("missing tensor data");
                file.write(reinterpret_cast<const char*>(data), bytes);
                if (!file) throw std::runtime_error("write failed");
            }
        }
        mlx_stream_free(stream);
        file.close();
        if (!file) throw std::runtime_error("close failed");
    }
    const auto ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::size_t peak{}, active{};
    if (mlx_get_peak_memory(&peak) || mlx_get_active_memory(&active))
        throw std::runtime_error("memory query failed");
    MlxSafetensors loaded(argv[2]);
    if (loaded.metadata("probe") != "strided-f32") throw std::runtime_error("metadata mismatch");
    for (int bank = 0; bank < count; ++bank) {
        const auto actual = loaded.tensor("bank" + std::to_string(bank));
        if (actual.shape() != banks[bank].shape() || actual.dtype() != MLX_FLOAT32 ||
            actual.to_float32() != banks[bank].to_float32()) throw std::runtime_error("roundtrip mismatch");
    }
    std::cout << "{\"mode\":\"" << argv[1] << "\",\"save_ms\":" << ms
        << ",\"warm\":" << (warm ? "true" : "false")
        << ",\"peak_mlx_bytes\":" << peak << ",\"active_after_save_bytes\":" << active
        << ",\"data_bytes\":" << count * bytes << ",\"roundtrip\":true}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
