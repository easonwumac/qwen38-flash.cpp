#include "qwen38/mlx_backend.hpp"
#include "../src/moe_metal_kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <mach/mach.h>

// Isolated synchronous experiment. No arrays/graphs may escape Pool's lifetime.
// Writes occur only after GPU completion; every invocation creates a fresh graph.
// Do not transplant host mutation into asynchronous serving without slot leases.
namespace {
using namespace qwen38;
constexpr std::size_t gib = 1ULL << 30;
void check(int code) { if (code) throw std::runtime_error("MLX operation failed"); }
void fence() {
    auto stream = mlx_default_gpu_stream_new();
    const int code = mlx_synchronize(stream);
    static_cast<void>(mlx_stream_free(stream));
    check(code);
}
struct Field {
    std::shared_ptr<void> storage;
    MlxArray array; // destroyed before storage, including constructor failures
    std::filesystem::path path;
    std::size_t offset{}, row_bytes{};
    std::vector<int> shape;
    mlx_dtype dtype{};
};
struct Pool {
    std::array<Field, 9> fields;
    int capacity;
    explicit Pool(const std::filesystem::path& model, int layer, int slots = 16) : capacity(slots) {
        if (slots < 1 || slots > 288) throw std::runtime_error("invalid slot capacity");
        const auto manifest = ModelManifest::load(model);
        std::size_t f = 0;
        for (const auto* projection : {"gate_proj", "up_proj", "down_proj"}) {
            for (const auto* name : {"weight", "scales", "biases"}) {
                auto& field = fields[f++];
                const auto key = "language_model.model.layers." + std::to_string(layer) +
                    ".mlp.switch_mlp." + projection + "." + name;
                field.path = model / manifest.weight_map().at(key);
                const SafetensorsFile file(field.path);
                const auto view = file.tensor(key);
                if (view.shape.size() != 3 || view.shape[0] != 288 ||
                    view.bytes.size() % 288 != 0)
                    throw std::runtime_error("unsupported expert geometry");
                field.row_bytes = view.bytes.size() / 288;
                if (field.row_bytes > 4 * 1024 * 1024)
                    throw std::runtime_error("field exceeds probe bound");
                field.dtype = view.dtype == "U32" ? MLX_UINT32 : MLX_BFLOAT16;
                if (view.dtype != "U32" && view.dtype != "BF16")
                    throw std::runtime_error("unsupported expert dtype");
                field.shape = {slots, static_cast<int>(view.shape[1]), static_cast<int>(view.shape[2])};
                field.offset = static_cast<std::size_t>(view.bytes.data() - file.mapped_view().data());
                const std::size_t bytes = static_cast<std::size_t>(slots) * field.row_bytes;
                const std::size_t padded = (bytes + 16383) / 16384 * 16384;
                field.storage = std::shared_ptr<void>(std::aligned_alloc(16384, padded), std::free);
                if (!field.storage) throw std::bad_alloc();
                std::memset(field.storage.get(), 0, padded);
                field.array = MlxArray(mlx_array_new_data_managed(field.storage.get(),
                    field.shape.data(), 3, field.dtype, [](void*) {}));
                if (!field.array.get().ctx || mlx_array_nbytes(field.array.get()) != bytes)
                    throw std::runtime_error("managed pool import failed");
            }
        }
        // Kernels have fixed Q4/group64 geometry; reject other model layouts.
        for (std::size_t f = 0; f < fields.size(); ++f) {
            const int rows = f < 6 ? 640 : 2560;
            const int cols = (f < 6 ? 2560 : 640) / (f % 3 == 0 ? 8 : 64);
            if (fields[f].shape != std::vector<int>{slots, rows, cols} ||
                fields[f].dtype != (f % 3 == 0 ? MLX_UINT32 : MLX_BFLOAT16))
                throw std::runtime_error("pool requires Q4/group64 BF16");
        }
    }
    void load(int slot, int expert) {
        if (slot < 0 || slot >= capacity || expert < 0 || expert >= 288)
            throw std::runtime_error("pool index outside bounds");
        fence();
        for (auto& f : fields) {
            std::ifstream file(f.path, std::ios::binary);
            file.seekg(static_cast<std::streamoff>(f.offset + expert * f.row_bytes));
            file.read(static_cast<char*>(f.storage.get()) + slot * f.row_bytes,
                static_cast<std::streamsize>(f.row_bytes));
            if (!file) throw std::runtime_error("slot SSD read failed");
        }
        // A failed load aborts the probe; production needs transactional publish.
    }
};
double footprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count) != KERN_SUCCESS)
        throw std::runtime_error("cannot observe footprint");
    const double value = double(info.phys_footprint)/gib;
    if (value > 35.8) throw std::runtime_error("layout probe early footprint stop");
    return value;
}
void memory_layout(const std::filesystem::path& model, bool baseline) {
    // This tests physical expert residency only, not routing, KV or inference.
    // Numeric IDs stand in for equal-sized hot/cold sets, not a REAP ID mapping.
    std::vector<std::unique_ptr<Pool>> hot, dynamic;
    const int hot_count = baseline ? 288 : 256;
    std::size_t hot_bytes = 0, cold_bytes = 0;
    for (int layer = 0; layer < 48; ++layer) {
        auto pool = std::make_unique<Pool>(model,layer,hot_count);
        for (int expert = 0; expert < hot_count; ++expert) pool->load(expert,expert);
        for (auto& f : pool->fields) hot_bytes += mlx_array_nbytes(f.array.get());
        hot.push_back(std::move(pool));
        std::cout << "hot_layer=" << layer << " footprint_gib=" << footprint() << std::endl;
    }
    std::cout << "hot_payload_gib=" << double(hot_bytes)/gib
              << " hot_footprint_gib=" << footprint() << std::endl;
    // 8 slots per layer = 384 experts, approximately 1 GiB. Filled with real
    // cold weights, not untouched virtual reservation. No cache policy claim.
    for (int layer = 0; layer < (baseline ? 0 : 48); ++layer) {
        auto pool = std::make_unique<Pool>(model,layer,8);
        for (int slot = 0; slot < 8; ++slot) pool->load(slot,256+slot);
        for (auto& f : pool->fields) cold_bytes += mlx_array_nbytes(f.array.get());
        dynamic.push_back(std::move(pool));
        footprint();
    }
    fence();
    std::cout << "dynamic_payload_gib=" << double(cold_bytes)/gib
              << " combined_footprint_gib=" << footprint()
              << " unpaged_payload_gib=" << double(hot_bytes)*288/hot_count/gib
              << " net_payload_saved_gib=" << (double(hot_bytes)*(288-hot_count)/hot_count-cold_bytes)/gib << std::endl;
    dynamic.clear(); hot.clear(); fence();
    std::size_t active{}; check(mlx_get_active_memory(&active));
    std::cout << "released_footprint_gib=" << footprint() << " released_mlx_bytes=" << active << std::endl;
    if (active != 0) throw std::runtime_error("slot layout did not release managed arrays");
}
struct Kernels {
    MlxMetalKernel gate, down;
    Kernels() : gate("slot_pool_gate", std::array<const char*,8>{"x","gw","gs","gb","uw","us","ub","experts"},
                    "h", moe_metal::gate_up, moe_metal::header),
        down("slot_pool_down", std::array<const char*,6>{"h","dw","ds","db","experts","rw"},
                    "y", moe_metal::down, moe_metal::header) {}
    MlxArray run(const std::vector<MlxArray>& f, const MlxArray& x,
                 const MlxArray& ids, const MlxArray& weights) {
        const MlxArray* g[]{&x,&f[0],&f[1],&f[2],&f[3],&f[4],&f[5],&ids};
        auto h = gate.apply(g, std::array<int,2>{10,640}, MLX_BFLOAT16,
            std::array<int,3>{200*1024,1,1}, std::array<int,3>{1024,1,1});
        const MlxArray* d[]{&h,&f[6],&f[7],&f[8],&ids,&weights};
        return down.apply(d, std::array<int,3>{1,1,2560}, MLX_BFLOAT16,
            std::array<int,3>{320*64,1,1}, std::array<int,3>{64,1,1});
    }
};
MlxArray reference(const std::vector<MlxArray>& f, const MlxArray& x,
                   const std::vector<std::int32_t>& ids, const std::vector<float>& weights) {
    MlxArray sum;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto project = [&](const MlxArray& input, std::size_t p) {
            std::vector<MlxArray> parts;
            for (std::size_t j = p; j < p + 3; ++j) {
                auto end = f[j].shape(); end[0] = ids[i] + 1;
                parts.push_back(f[j].slice(std::vector<int>{ids[i],0,0}, end,
                    std::vector<int>{1,1,1}).reshape(std::vector<int>{end[1],end[2]}));
            }
            return MlxArray::quantized_matmul(input, parts[0],parts[1],parts[2],64,4);
        };
        auto h = MlxArray::multiply(project(x,0).silu(),project(x,3));
        auto w = MlxArray::from_float32(std::vector<float>{weights[i]}, std::vector<int>{}).astype(MLX_BFLOAT16);
        auto y = MlxArray::multiply(project(h,6),w);
        sum = i == 0 ? std::move(y) : MlxArray::add(sum,y);
    }
    return sum;
}
template<class F> void bench(const char* name, F run) {
    std::vector<double> ms;
    for (int n = 0; n < 55; ++n) {
        const auto start = std::chrono::steady_clock::now();
        { auto out = run(); out.eval(); fence(); }
        if (n >= 5) ms.push_back(std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count());
    }
    std::sort(ms.begin(),ms.end());
    std::cout << name << " median_ms=" << ms[25] << " p95_ms=" << ms[46] << std::endl;
}
}
int main(int argc, char** argv) {
    try {
        const bool baseline = argc == 3 && std::string_view(argv[2]) == "--memory-baseline";
        const bool layout = baseline || (argc == 3 && std::string_view(argv[2]) == "--memory-layout");
        if ((argc != 2 && !layout) || !std::getenv("QWEN38_MEMORY_GUARD") ||
            !std::getenv("MLX_STRICT_MEMORY_LIMIT") ||
            std::string_view(std::getenv("MLX_STRICT_MEMORY_LIMIT")) != "1")
            throw std::runtime_error("requires guarded strict allocator and MODEL path");
        mlx_set_error_handler([](const char* msg, void*) { std::cerr << msg << '\n'; },nullptr,nullptr);
        std::size_t old{}; check(mlx_set_memory_limit(&old,layout ? 36*gib : gib)); check(mlx_set_cache_limit(&old,0));
        if (layout) { memory_layout(argv[1],baseline); return 0; }
        Kernels kernels;
        for (int layer : {0,47}) {
            Pool pool(argv[1],layer);
            for (int i = 0; i < 16; ++i) pool.load(i,i*7);
            std::vector<MlxArray> fields;
            for (auto& f : pool.fields) fields.push_back(f.array.share());
            std::vector<float> input(2560), weights(10);
            for (int i = 0; i < 2560; ++i) input[i] = std::sin(float(i)*0.13f)*0.2f;
            for (int i = 0; i < 10; ++i) weights[i] = float(i+1)/55;
            auto x = MlxArray::from_float32(input,std::vector<int>{1,1,2560}).astype(MLX_BFLOAT16);
            const std::vector<std::int32_t> slots{9,3,15,0,8,1,12,5,10,7};
            auto ids = MlxArray::from_int32(slots,std::vector<int>{10});
            auto rw = MlxArray::from_float32(weights,std::vector<int>{10});
            std::vector<float> previous, initial;
            for (int generation = 0; generation < 3; ++generation) {
                if (generation) pool.load(3,generation == 1 ? 201 : 21);
                auto actual = kernels.run(fields,x,ids,rw).astype(MLX_FLOAT32).to_float32(); fence();
                std::vector<MlxArray> copied;
                for (auto& f : pool.fields) copied.emplace_back(mlx_array_new_data(
                    f.storage.get(),f.shape.data(),3,f.dtype));
                auto expected = kernels.run(copied,x,ids,rw).astype(MLX_FLOAT32).to_float32(); fence();
                if (actual != expected) throw std::runtime_error("slot visibility parity failed");
                if (!previous.empty() && actual == previous) throw std::runtime_error("replacement did not change output");
                if (generation == 0) initial = actual;
                if (generation == 2 && actual != initial) throw std::runtime_error("slot restoration changed output");
                previous = actual;
                double max_error = 0, squared = 0;
                auto oracle = reference(fields,x,slots,weights).astype(MLX_FLOAT32).to_float32(); fence();
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i]) || !std::isfinite(oracle[i]))
                        throw std::runtime_error("non-finite output");
                    const double e = double(actual[i])-oracle[i];
                    max_error = std::max(max_error,std::abs(e)); squared += e*e;
                }
                std::cout << "layer=" << layer << " generation=" << generation
                    << " slot_vs_copied_exact=true qmm_max_abs=" << max_error
                    << " qmm_rmse=" << std::sqrt(squared/actual.size()) << std::endl;
            }
            pool.load(6,202); // Unselected slot must not affect this route.
            auto unaffected = kernels.run(fields,x,ids,rw).astype(MLX_FLOAT32).to_float32(); fence();
            if (unaffected != initial) throw std::runtime_error("unselected slot affected output");
            std::cout << "restoration_and_unselected_slot_exact=true" << std::endl;
            bench("fused_pool",[&] { return kernels.run(fields,x,ids,rw); });
            bench("unfused_pool",[&] { return reference(fields,x,slots,weights); });
            bench("unfused_pool_repeat",[&] { return reference(fields,x,slots,weights); });
            bench("fused_pool_repeat",[&] { return kernels.run(fields,x,ids,rw); });
            std::size_t peak{}; check(mlx_get_peak_memory(&peak));
            std::cout << "mlx_peak_mib=" << double(peak)/(1024*1024) << std::endl;
            fence();
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
