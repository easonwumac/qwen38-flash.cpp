#include "qwen38/model.hpp"
#include "qwen38/sparse_moe.hpp"
#include "qwen38/memory_budget.hpp"
#ifdef QWEN38_PROBE_TOKENIZER
#include "qwen38/tokenizer.hpp"
#endif

#include <chrono>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <numeric>
#include <fstream>
#include <mach/mach.h>

namespace {
constexpr std::size_t gib = 1024ULL * 1024ULL * 1024ULL;
using Clock = std::chrono::steady_clock;
std::size_t process_ceiling = 24 * gib;
std::size_t elastic_steady = 24 * gib;
void check(int code) { if (code != 0) throw std::runtime_error("MLX budget operation failed"); }
std::size_t usage(const char* phase) {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        throw std::runtime_error("cannot observe process footprint");
    std::cout << "phase=" << phase << " footprint_gib=" << double(info.phys_footprint) / gib << std::endl;
    if (info.phys_footprint > process_ceiling - 2 * gib)
        throw std::runtime_error("probe reached early process safety threshold");
    return static_cast<std::size_t>(info.phys_footprint);
}

void replay_policies(const qwen38::MlxTensorStore& tensors, const std::size_t initial_budget) {
    // Fixed-cap counterfactuals reuse routing, not measured elapsed time. They
    // cannot predict storage throughput, workspace pressure, or MTP acceptance.
    for (const std::size_t fixed_budget : {std::size_t{0}, 16 * gib, 24 * gib}) {
    for (const std::uint64_t period : {64ULL, 4096ULL, 8192ULL, 16384ULL}) {
        qwen38::ExpertCache<int> cache(fixed_budget == 0 ? initial_budget : fixed_budget,
            period, period == 64 ? 0 : 16384);
        std::vector<qwen38::ExpertCache<int>::Handle> leases;
        for (const auto& event : tensors.expert_trace()) {
            using Kind = qwen38::MlxTensorStore::ExpertTraceKind;
            if (event.kind == Kind::boundary) leases.clear();
            else if (event.kind == Kind::budget) {
                if (fixed_budget == 0 && !cache.set_budget(event.bytes))
                    throw std::runtime_error("trace budget blocked");
            } else {
                leases.push_back(cache.acquire(event.key, event.bytes,
                    [] { return std::make_shared<const int>(0); }));
            }
        }
        const auto& stats = cache.stats();
        if (fixed_budget == 0 && period == 4096 && (stats.misses != tensors.expert_stats().misses ||
                              stats.evictions != tensors.expert_stats().evictions))
            throw std::runtime_error("trace replay differs from live cache");
        std::cout << "policy_replay fixed_expert_gib=" << fixed_budget / gib
                  << " decay=" << period << " hits=" << stats.hits
                  << " misses=" << stats.misses << " evictions=" << stats.evictions
                  << " history_hits=" << stats.history_hits << std::endl;
    }
    }
}
}

int main(int argc, char** argv) {
    try {
        bool layer_only = false, extended = false, elastic = false, serial = false, long_run = false;
        bool steady_only = false;
        bool legacy_frequency = false;
        bool pp64 = false;
        bool rowwise_pp = false, prompt_suite = false;
        bool unpacked_decode = true;
        bool trace_policy = false;
        bool parallel_reads = false;
        bool fixed_slots=false, fixed_baseline=false;
        std::string study_path;
        for (int i = 2; i < argc; ++i) {
            const std::string_view flag(argv[i]);
            if (flag == "--layer-parity") layer_only = true;
            else if (flag == "--extended") extended = true;
            else if (flag == "--elastic") elastic = true;
            else if (flag == "--elastic-28-36") { elastic = true; elastic_steady = 28 * gib; }
            else if (flag == "--serial") serial = true;
            else if (flag == "--long") long_run = true;
            else if (flag == "--steady-only") steady_only = true;
            else if (flag == "--legacy-frequency") legacy_frequency = true;
            else if (flag == "--pp64") pp64 = true;
            else if (flag == "--rowwise-pp") rowwise_pp = true;
            else if (flag == "--prompt-suite") prompt_suite = true;
            else if (flag == "--unpacked-decode") unpacked_decode = true;
            else if (flag == "--packed-decode") unpacked_decode = false;
            else if (flag == "--trace-policy") trace_policy = true;
            else if (flag == "--parallel-reads") parallel_reads = true;
            else if (flag == "--fixed-slots") fixed_slots=true;
            else if (flag == "--fixed-baseline") { fixed_slots=true; fixed_baseline=true; }
            else if (flag == "--routing-study" && i+1<argc) {
                study_path=argv[++i]; fixed_slots=true; extended=true; prompt_suite=true;
            }
            else throw std::runtime_error("unknown probe flag");
        }
        if (argc < 2 || std::getenv("QWEN38_MEMORY_GUARD") == nullptr ||
            std::getenv("MLX_STRICT_MEMORY_LIMIT") == nullptr ||
            std::string_view(std::getenv("MLX_STRICT_MEMORY_LIMIT")) != "1")
            throw std::runtime_error("usage: guarded private-strict probe MODEL_DIRECTORY");
        if (steady_only && !elastic) throw std::runtime_error("--steady-only requires --elastic");
        if (trace_policy && legacy_frequency) throw std::runtime_error("trace validation requires default frequency");
        if (fixed_slots && (trace_policy || elastic || serial || legacy_frequency || parallel_reads || !unpacked_decode))
            throw std::runtime_error("fixed slots require an isolated probe configuration");
        if (!study_path.empty() && (layer_only || long_run || fixed_baseline))
            throw std::runtime_error("routing study requires short fixed candidate");
        std::ofstream study;
        if (!study_path.empty()) {
            if (std::filesystem::exists(study_path)) throw std::runtime_error("study output already exists");
            study.open(study_path);
            if (!study) throw std::runtime_error("cannot create study trace");
            study << "phase\tkind\tkey\tbytes\n";
        }
        // Research-only: leave 4 GiB outside the MLX cap for host staging and
        // framework memory. The external guard independently limits the process.
        mlx_set_error_handler([](const char* message, void*) { std::cerr << message << '\n'; }, nullptr, nullptr);
        std::size_t old = 0;
        process_ceiling = elastic ? elastic_steady + 8 * gib : 24 * gib;
        if (fixed_slots) process_ceiling=42*gib;
        if (!study_path.empty()) process_ceiling=36*gib;
        check(mlx_set_memory_limit(&old, layer_only ? 3 * gib : process_ceiling - (fixed_slots ? 2 : 4) * gib));
        check(mlx_set_cache_limit(&old, 0));
        check(mlx_clear_cache());
        if (layer_only) {
          for (const int layer : {0, 47}) {
            qwen38::MlxTensorStore paged(qwen38::ModelManifest::load(argv[1]),
                (serial ? 8 : 64) * 1024 * 1024, !serial, 4096, !rowwise_pp, !unpacked_decode, parallel_reads);
            qwen38::MlxTensorStore reference(qwen38::ModelManifest::load(argv[1]));
            if (fixed_slots) { paged.enable_fixed_slots(256); reference.enable_fixed_slots(288); }
            const auto& config = paged.manifest().config();
            const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".mlp";
            if (fixed_slots) {
                std::vector<qwen38::MlxTensorStore::ExpertLease> held;
                for (std::size_t id=256; id<264; ++id) held.push_back(paged.expert(prefix,id));
                bool refused=false;
                try { static_cast<void>(paged.expert(prefix,264)); }
                catch (const std::runtime_error& e) {
                    if (std::string_view(e.what())!="all fixed cold slots leased") throw;
                    refused=true;
                }
                if (!refused) throw std::runtime_error("fixed slot pin protection failed");
                held.clear();
                static_cast<void>(paged.expert(prefix,264));
                std::cout << "fixed_pinned_refusal_and_reuse=true" << std::endl;
            }
            qwen38::SparseMoe a(paged, prefix, 288, 10, 4, 64, true);
            qwen38::SparseMoe b(reference, prefix, 288, 10, 4, 64, true);
            for (const int rows : {1, 4, 32, 64}) {
                std::vector<float> values(static_cast<std::size_t>(rows) * config.hidden_size);
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = std::sin(static_cast<float>(i) * 0.031F);
                auto input = qwen38::MlxArray::from_float32(values,
                    std::vector<int>{1, rows, static_cast<int>(config.hidden_size)}).astype(MLX_BFLOAT16);
                auto actual = a.forward_prefill(input).astype(MLX_FLOAT32).to_float32();
                auto expected = b.forward_prefill(input).astype(MLX_FLOAT32).to_float32();
                if (actual.size() != expected.size()) throw std::runtime_error("parity shape mismatch");
                for (std::size_t i = 0; i < actual.size(); ++i)
                    if (!std::isfinite(actual[i]) || actual[i] != expected[i])
                        throw std::runtime_error("paged real-router/shared-expert parity mismatch");
                std::cout << "layer_parity layer=" << layer << " rows=" << rows << " exact=true evictions="
                          << paged.expert_stats().evictions << std::endl;
            }
            usage("layer_parity");
          }
            return 0;
        }
        const auto load_started = Clock::now();
        // 4 GiB measured conservative allowance for this SHORT probe's base and
        // states. Do not reuse this estimate for arbitrary context lengths.
        const auto plan = qwen38::plan_expert_budget(
            {elastic ? elastic_steady : 20 * gib, process_ceiling}, {.non_expert_bytes = 4 * gib,
             .minimum_expert_bytes = 64 * 1024 * 1024});
        const std::size_t steady_budget = plan.steady_expert_bytes;
        // Independently leave 4 GiB in the MLX pool for non-expert allocations.
        const std::size_t burst_budget = std::min(plan.burst_expert_bytes, process_ceiling - 8 * gib);
        if (!plan.feasible) throw std::runtime_error("infeasible probe memory plan");
        qwen38::MlxTensorStore tensors(qwen38::ModelManifest::load(argv[1]), steady_budget, !serial,
            legacy_frequency ? 64 : 4096, !rowwise_pp, !unpacked_decode, parallel_reads);
        const std::size_t hot_count=fixed_baseline ? 288 : study_path.empty() ? 256 : 224;
        if (fixed_slots) tensors.enable_fixed_slots(hot_count);
        if (fixed_slots) std::cout << "fixed_hot_per_layer=" << hot_count
            << " fixed_cold_slots_per_layer=" << (fixed_baseline ? 0 : 8)
            << " process_ceiling_gib=" << double(process_ceiling)/gib << std::endl;
        else std::cout << "expert_steady_gib=" << double(steady_budget) / gib
                  << " expert_burst_limit_gib=" << double(burst_budget) / gib
                  << " process_ceiling_gib=" << double(process_ceiling) / gib
                  << " batch_experts=" << !serial << std::endl;
        qwen38::QwenModel model(tensors);
        if (fixed_slots) tensors.preload_fixed_slots();
        usage("loaded");
        std::cout << "load_seconds=" << std::chrono::duration<double>(Clock::now() - load_started).count() << std::endl;
        auto state = model.make_state();
        if (trace_policy || !study_path.empty()) tensors.start_expert_trace();
        std::size_t exported=0;
        const auto export_phase=[&](const std::string& phase) {
            if (study_path.empty()) return;
            const auto& events=tensors.expert_trace();
            for (; exported<events.size(); ++exported) {
                const auto& e=events[exported];
                study << phase << '\t' << static_cast<int>(e.kind) << '\t' << e.key << '\t' << e.bytes << '\n';
            }
            study.flush();
            if (!study) throw std::runtime_error("study trace write failed");
        };
        std::uint32_t token = 9419;
        std::vector<double> seconds;
        std::uint64_t token_hash = 14695981039346656037ULL;
        bool borrowed = false;
        std::size_t previous_evictions = 0, previous_misses = 0;
        for (int i = 0; i < (long_run ? 256 : extended ? 64 : 8); ++i) {
            const auto start = Clock::now();
            const auto result = model.greedy_decode(token, state);
            token = result.token;
            token_hash = (token_hash ^ token) * 1099511628211ULL;
            seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
            const auto& stats = tensors.expert_stats();
            std::cout << "step=" << i << " token=" << token
                << " seconds=" << seconds.back()
                << " hits=" << stats.hits << " misses=" << stats.misses
                << " evictions=" << stats.evictions << " expert_gib=" << double(stats.resident_bytes) / gib
                << " load_ms=" << tensors.expert_load_ms() << std::endl;
            usage("decode");
            if (elastic && !steady_only && !borrowed && stats.evictions > previous_evictions &&
                stats.misses - previous_misses > 48) {
                if (!tensors.set_expert_budget(burst_budget)) throw std::runtime_error("burst resize failed");
                borrowed = true;
                std::cout << "expert_burst_gib=" << double(burst_budget) / gib << std::endl;
            }
            previous_evictions = stats.evictions;
            previous_misses = stats.misses;
        }
        const double steady = std::accumulate(seconds.begin() + 1, seconds.end(), 0.0);
        std::vector<double> sorted(seconds.begin() + 1, seconds.end());
        std::sort(sorted.begin(), sorted.end());
        std::cout << "decode_tps_excluding_first=" << double(seconds.size() - 1) / steady
                  << " step_median_seconds=" << sorted[sorted.size() / 2]
                  << " step_p95_seconds=" << sorted[(sorted.size() - 1) * 95 / 100] << std::endl;
        std::cout << "token_hash=" << token_hash << std::endl;
        export_phase("warmup_decode");
        if (trace_policy) {
            tensors.stop_expert_trace();
            replay_policies(tensors, steady_budget);
        }
        // Return borrowed expert capacity before building the PP workspace.
        if (!fixed_slots && !tensors.set_expert_budget(steady_budget)) throw std::runtime_error("PP reserve shrink blocked");
        usage("pp_reserve");
        std::vector<std::uint32_t> prompt{9419, 11, 358, 1440, 264, 1937, 13, 198};
        if (pp64) {
            const auto seed = prompt;
            for (int i = 1; i < 8; ++i) prompt.insert(prompt.end(), seed.begin(), seed.end());
        }
        std::vector<std::vector<std::uint32_t>> prompts{prompt};
        if (prompt_suite) {
#ifdef QWEN38_PROBE_TOKENIZER
            const auto tokenizer = qwen38::Tokenizer::load(argv[1]);
            prompts.clear();
            for (const auto text : {
                "Explain how an operating system decides which file pages to keep in memory. Compare recent access with frequent access, describe how a large sequential scan can displace useful data, and propose a bounded cache policy that adapts when the workload changes. State how you would measure latency and memory pressure without confusing a warm cache with faster storage.",
                "Review this Python function for correctness and explain edge cases: def average(xs): return sum(xs) / len(xs). Discuss empty input, iterators, missing values, floating point precision, and invalid types. Then propose a small test suite and an implementation whose behavior is explicit rather than silently ignoring errors. Keep the explanation suitable for a new programmer.",
                "請整理以下會議紀錄：工程團隊希望降低推論伺服器的記憶體用量，產品團隊要求回應速度不能明顯退步。測試需涵蓋短對話、程式碼與長文件，並分開記錄首次載入、快取命中及連續使用。請列出待確認的問題、驗收條件與下一次會議需要的數據。"})
                prompts.push_back(tokenizer.encode(text));
#else
            throw std::runtime_error("prompt suite requires tokenizer build");
#endif
        }
        for (std::size_t p = 0; p < prompts.size(); ++p) {
        prompt = prompts[p];
        if (prompt.empty() || prompt.size() > 1024) throw std::runtime_error("probe prompt outside 1..1024");
        const auto misses_before = tensors.expert_stats().misses;
        const auto loads_before = tensors.expert_load_ms();
        const auto start = Clock::now();
        auto prefill_state = model.make_state();
        std::uint32_t continuation=0;
        std::uint64_t pp_hash = 14695981039346656037ULL;
        std::cout << "prompt_start=" << p << " tokens=" << prompt.size() << std::endl;
        for (std::size_t offset = 0; offset < prompt.size(); offset += 64) {
            const auto chunk = std::span<const std::uint32_t>(prompt).subspan(offset,
                std::min<std::size_t>(64, prompt.size() - offset));
            auto output = model.prefill_chunk_batch(chunk, prefill_state);
            const auto logits = output.astype(MLX_FLOAT32).to_float32();
            if (!study_path.empty()) {
                const auto vocab=static_cast<std::size_t>(output.shape().back());
                auto begin=logits.end()-static_cast<std::ptrdiff_t>(vocab);
                continuation=static_cast<std::uint32_t>(std::max_element(begin,logits.end())-begin);
            }
            for (const float value : logits) {
                if (!std::isfinite(value)) throw std::runtime_error("non-finite prefill logits");
                pp_hash = (pp_hash ^ std::bit_cast<std::uint32_t>(value)) * 1099511628211ULL;
            }
            usage("pp_chunk");
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << "prompt=" << p << " pp_tokens=" << prompt.size() << " pp_seconds=" << elapsed
                  << " pp_tps=" << double(prompt.size()) / elapsed << std::endl;
        std::cout << "pp_hash=" << pp_hash << " misses=" << tensors.expert_stats().misses - misses_before
                  << " load_ms=" << tensors.expert_load_ms() - loads_before << std::endl;
        usage("prefill");
        export_phase("prompt"+std::to_string(p)+"_pp");
        if (!study_path.empty()) {
            const auto decode_start=Clock::now();
            const auto before=tensors.expert_stats().misses;
            for (int n=0; n<64; ++n) {
                continuation=model.greedy_decode(continuation,prefill_state).token;
                usage("study_decode");
            }
            std::cout << "study_prompt=" << p << " decode_steps=64 seconds="
                << std::chrono::duration<double>(Clock::now()-decode_start).count()
                << " misses=" << tensors.expert_stats().misses-before << std::endl;
            export_phase("prompt"+std::to_string(p)+"_decode");
        }
        }
        if (!fixed_slots && !tensors.set_expert_budget(steady_budget)) throw std::runtime_error("steady shrink blocked");
        std::cout << (fixed_slots ? "fixed_expert_gib=" : "expert_shrunk_gib=")
            << double(tensors.expert_stats().resident_bytes) / gib << std::endl;
        const auto restored = usage(fixed_slots ? "fixed_final" : "steady_restored");
        if (elastic && restored > elastic_steady)
            throw std::runtime_error("process failed to return below steady target");
        if (!study_path.empty()) std::cout << "study_live_misses=" << tensors.expert_stats().misses
            << " evictions=" << tensors.expert_stats().evictions << " events=" << exported << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "paged probe: " << e.what() << std::endl;
        return 1;
    }
}
