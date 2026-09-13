#include "qwen38/api.hpp"
#include "qwen38/http_server.hpp"
#include "qwen38/model_manifest.hpp"
#include "qwen38/runtime.hpp"
#include "qwen38/runtime_profile.hpp"
#ifdef QWEN38_HAS_INFERENCE
#include "qwen38/native_engine.hpp"
#endif

#include <charconv>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

qwen38::HttpServer* active_server = nullptr;

void handle_signal(const int) {
    if (active_server != nullptr) {
        active_server->stop();
    }
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [--host IPv4] [--port PORT] [--model PATH]"
        << " [--profile safe|speed|turbo|latency|long-context|memory]"
        << " [--prefill-chunk 1..1024] [--prefill-chunk-fixed]"
        << " [--prefix-cache-tokens N]"
        << " [--qmeta-cache-max-prompt-tokens N] [--qmeta-cache-layers 0..48]"
        << " [--qsa-packed-min-tokens N] [--qsa-shared-rows 1|2|4|8]"
        << " [--ssd-prefix-cache-gib N] [--ssd-prefix-cache-dir PATH]"
        << " [--allocator-cache-mib N]"
        << " [--max-generation-tokens N]"
        << " [--kv-cache bf16|q8] [--kv-q8-min-tokens N] [--kv-q8-flush-tokens N]"
        << " [--mtp-depth auto|off|2|3|4]\n"
        << "\n"
        << "qwen38-flash.cpp native inference server.\n";
}

std::optional<std::size_t> parse_mtp_depth(const std::string& value) {
    if (value == "auto") return std::nullopt;
    if (value == "off" || value == "0") return 0;
    if (value == "2") return 2;
    if (value == "3") return 3;
    if (value == "4") return 4;
    throw std::runtime_error("invalid MTP depth: " + value);
}

std::string mtp_depth_name(const std::optional<std::size_t> depth) {
    if (!depth.has_value()) return "auto";
    if (*depth == 0) return "off";
    return std::to_string(*depth);
}

std::uint16_t parse_port(const std::string& value) {
    unsigned int parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed == 0 || parsed > 65535) {
        throw std::runtime_error("invalid port: " + value);
    }
    return static_cast<std::uint16_t>(parsed);
}

std::size_t parse_prefill_chunk(const std::string& value) {
    std::size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed == 0 || parsed > 1024) {
        throw std::runtime_error("invalid prefill chunk: " + value);
    }
    return parsed;
}

std::size_t parse_size(const std::string& value, const char* name) {
    std::size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return parsed;
}

std::string environment_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        qwen38::ServerConfig config;
        std::string profile = "safe";
        // Stable serial execution is the server default. Loading and executing a
        // companion drafter is opt-in because its memory and throughput benefit
        // depend on both the checkpoint and the request trajectory.
        std::optional<std::size_t> mtp_depth = 0;
        bool mtp_depth_explicit = false;
        std::size_t prefill_chunk_rows = 64;
        bool prefill_chunk_explicit = false;
        bool adaptive_prefill_chunks = true;
        std::size_t prefix_cache_max_tokens = 8192;
        bool prefix_cache_explicit = false;
        std::size_t qmeta_cache_max_prompt_tokens = 32768;
        bool qmeta_cache_limit_explicit = false;
        std::optional<std::size_t> qmeta_cache_layers;
        std::optional<std::size_t> qsa_packed_min_tokens;
        std::optional<std::size_t> qsa_shared_rows;
        std::uint64_t ssd_prefix_cache_max_bytes = 0;
        std::filesystem::path ssd_prefix_cache_directory;
        std::size_t allocator_cache_limit_bytes = 256ULL * 1024ULL * 1024ULL;
        bool allocator_cache_explicit = false;
        std::size_t max_generation_tokens = 4096;
        std::string kv_cache = "bf16";
        std::size_t kv_q8_min_tokens = 65536;
        std::size_t kv_q8_flush_tokens = 8192;
        std::optional<std::string> model_path;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
            if ((argument == "--host" || argument == "--port" || argument == "--model" ||
                 argument == "--mtp-depth" || argument == "--prefill-chunk" ||
                 argument == "--prefix-cache-tokens" ||
                 argument == "--qmeta-cache-max-prompt-tokens" ||
                 argument == "--qmeta-cache-layers" ||
                 argument == "--qsa-packed-min-tokens" ||
                 argument == "--qsa-shared-rows" ||
                 argument == "--ssd-prefix-cache-gib" ||
                 argument == "--ssd-prefix-cache-dir" ||
                 argument == "--allocator-cache-mib" ||
                 argument == "--max-generation-tokens" ||
                 argument == "--kv-cache" || argument == "--kv-q8-min-tokens" ||
                 argument == "--kv-q8-flush-tokens" ||
                 argument == "--profile") &&
                i + 1 >= argc) {
                throw std::runtime_error("missing value for " + argument);
            }
            if (argument == "--host") {
                config.host = argv[++i];
            } else if (argument == "--port") {
                config.port = parse_port(argv[++i]);
            } else if (argument == "--model") {
                model_path = argv[++i];
            } else if (argument == "--profile") {
                profile = argv[++i];
            } else if (argument == "--mtp-depth") {
                mtp_depth = parse_mtp_depth(argv[++i]);
                mtp_depth_explicit = true;
            } else if (argument == "--prefill-chunk") {
                prefill_chunk_rows = parse_prefill_chunk(argv[++i]);
                prefill_chunk_explicit = true;
            } else if (argument == "--prefill-chunk-fixed") {
                adaptive_prefill_chunks = false;
            } else if (argument == "--prefix-cache-tokens") {
                prefix_cache_max_tokens = parse_size(argv[++i], "prefix cache token limit");
                prefix_cache_explicit = true;
            } else if (argument == "--qmeta-cache-max-prompt-tokens") {
                qmeta_cache_max_prompt_tokens =
                    parse_size(argv[++i], "qmeta cache prompt token limit");
                qmeta_cache_limit_explicit = true;
            } else if (argument == "--qmeta-cache-layers") {
                qmeta_cache_layers = parse_size(argv[++i], "qmeta cache layer count");
                if (*qmeta_cache_layers > 48) {
                    throw std::runtime_error("qmeta cache layer count must be between 0 and 48");
                }
            } else if (argument == "--qsa-packed-min-tokens") {
                qsa_packed_min_tokens =
                    parse_size(argv[++i], "packed QSA activation threshold");
            } else if (argument == "--qsa-shared-rows") {
                qsa_shared_rows = parse_size(argv[++i], "QSA shared row count");
                if (*qsa_shared_rows != 1 && *qsa_shared_rows != 2 &&
                    *qsa_shared_rows != 4 && *qsa_shared_rows != 8) {
                    throw std::runtime_error("QSA shared row count must be 1, 2, 4, or 8");
                }
            } else if (argument == "--ssd-prefix-cache-gib") {
                const std::size_t gib = parse_size(argv[++i], "SSD prefix cache size");
                constexpr std::uint64_t bytes_per_gib = 1024ULL * 1024ULL * 1024ULL;
                if (gib > std::numeric_limits<std::uint64_t>::max() / bytes_per_gib) {
                    throw std::runtime_error("SSD prefix cache size is too large");
                }
                ssd_prefix_cache_max_bytes = gib * bytes_per_gib;
            } else if (argument == "--ssd-prefix-cache-dir") {
                ssd_prefix_cache_directory = argv[++i];
            } else if (argument == "--allocator-cache-mib") {
                const std::size_t mib = parse_size(argv[++i], "allocator cache size");
                constexpr std::size_t bytes_per_mib = 1024ULL * 1024ULL;
                if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / bytes_per_mib) {
                    throw std::runtime_error("allocator cache size must be positive and bounded");
                }
                allocator_cache_limit_bytes = mib * bytes_per_mib;
                allocator_cache_explicit = true;
            } else if (argument == "--max-generation-tokens") {
                max_generation_tokens = parse_size(argv[++i], "generation token limit");
                if (max_generation_tokens == 0) {
                    throw std::runtime_error("generation token limit must be positive");
                }
            } else if (argument == "--kv-cache") {
                kv_cache = argv[++i];
                if (kv_cache != "bf16" && kv_cache != "q8") {
                    throw std::runtime_error("KV cache must be bf16 or q8");
                }
            } else if (argument == "--kv-q8-min-tokens") {
                kv_q8_min_tokens = parse_size(argv[++i], "Q8 KV activation threshold");
                if (kv_q8_min_tokens < 2049) {
                    throw std::runtime_error("Q8 KV activation threshold must exceed 2048");
                }
            } else if (argument == "--kv-q8-flush-tokens") {
                kv_q8_flush_tokens = parse_size(argv[++i], "Q8 KV flush interval");
                if (kv_q8_flush_tokens < 512) {
                    throw std::runtime_error("Q8 KV flush interval must be at least 512");
                }
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }
        const qwen38::RuntimeProfileConfig profile_config =
            qwen38::runtime_profile_config(profile);
        if (!allocator_cache_explicit) {
            allocator_cache_limit_bytes =
                profile_config.allocator_cache_mib * 1024ULL * 1024ULL;
        }
        qwen38::apply_runtime_profile(profile);
        if (qmeta_cache_layers.has_value()) {
            const std::string layers = std::to_string(*qmeta_cache_layers);
            const char* enabled = *qmeta_cache_layers == 0 ? "0" : "1";
            if (setenv("QWEN38_QMETA_PREFILL_CACHE", enabled, 1) != 0 ||
                setenv("QWEN38_QMETA_PREFILL_CACHE_LAYERS", layers.c_str(), 1) != 0) {
                throw std::runtime_error("cannot configure qmeta prefill cache");
            }
        }
        if (qsa_packed_min_tokens.has_value()) {
            const std::string threshold = std::to_string(*qsa_packed_min_tokens);
            if (setenv("QWEN38_QSA_PACKED_MIN_TOKENS", threshold.c_str(), 1) != 0) {
                throw std::runtime_error("cannot configure packed QSA activation threshold");
            }
        }
        if (qsa_shared_rows.has_value()) {
            const std::string rows = std::to_string(*qsa_shared_rows);
            if (setenv("QWEN38_QSA_SHARED_ROWS", rows.c_str(), 1) != 0) {
                throw std::runtime_error("cannot configure QSA row sharing");
            }
        }
        const std::string kv_q8_min_tokens_text = std::to_string(kv_q8_min_tokens);
        const std::string kv_q8_flush_tokens_text = std::to_string(kv_q8_flush_tokens);
        if (setenv("QWEN38_KV_CACHE", kv_cache.c_str(), 1) != 0 ||
            setenv("QWEN38_KV_Q8_MIN_TOKENS",
                kv_q8_min_tokens_text.c_str(), 1) != 0 ||
            setenv("QWEN38_KV_Q8_FLUSH_TOKENS",
                kv_q8_flush_tokens_text.c_str(), 1) != 0) {
            throw std::runtime_error("cannot configure KV cache mode");
        }
        if (profile_config.optimized && !prefill_chunk_explicit) {
            // Normal interactive profiles have enough headroom for a wider
            // trunk batch. Memory-bearing and MTP-heavy profiles keep the
            // smaller working set.
            prefill_chunk_rows =
                profile == "speed" || profile == "latency" ? 1024 : 512;
        }
        if (profile_config.memory_efficient) {
            if (!mtp_depth_explicit) mtp_depth = 0;
            if (!prefix_cache_explicit) prefix_cache_max_tokens = 0;
            if (!qmeta_cache_limit_explicit) qmeta_cache_max_prompt_tokens = 0;
        }

        qwen38::RuntimeState runtime;
        std::unique_ptr<qwen38::InferenceEngine> engine;
        if (model_path.has_value()) {
            runtime.begin_loading(*model_path);
            try {
#ifdef QWEN38_HAS_INFERENCE
                qwen38::NativeEngineOptions engine_options;
                engine_options.max_generation_tokens = max_generation_tokens;
                engine_options.mtp_depth = mtp_depth;
                engine_options.prefill_chunk_rows = prefill_chunk_rows;
                engine_options.adaptive_prefill_chunks = adaptive_prefill_chunks;
                engine_options.prefix_cache_max_tokens = prefix_cache_max_tokens;
                engine_options.qmeta_cache_max_prompt_tokens =
                    qmeta_cache_max_prompt_tokens;
                engine_options.ssd_prefix_cache_max_bytes =
                    ssd_prefix_cache_max_bytes;
                engine_options.ssd_prefix_cache_directory =
                    ssd_prefix_cache_directory;
                engine_options.allocator_cache_limit_bytes = allocator_cache_limit_bytes;
                engine = std::make_unique<qwen38::NativeEngineExecutor>(
                    *model_path, engine_options);
                std::clog << "qwen38-server: profile=" << profile
                          << " mtp_depth=" << mtp_depth_name(mtp_depth)
                          << " prefill_chunk=" << prefill_chunk_rows
                          << " adaptive_prefill_chunks="
                          << (adaptive_prefill_chunks ? "true" : "false")
                          << " qmeta_cache_max_prompt_tokens="
                          << qmeta_cache_max_prompt_tokens
                          << " qmeta_cache_layers="
                          << environment_or("QWEN38_QMETA_PREFILL_CACHE_LAYERS", "all")
                          << " qsa_packed_min_tokens="
                          << environment_or("QWEN38_QSA_PACKED_MIN_TOKENS", "65536")
                          << " qsa_shared_rows="
                          << environment_or("QWEN38_QSA_SHARED_ROWS", "1")
                          << " ssd_prefix_cache_gib="
                          << ssd_prefix_cache_max_bytes / (1024ULL * 1024ULL * 1024ULL)
                          << " allocator_cache_mib="
                          << allocator_cache_limit_bytes / (1024ULL * 1024ULL)
                          << " kv_cache=" << kv_cache
                          << " kv_q8_min_tokens=" << kv_q8_min_tokens
                          << " kv_q8_flush_tokens=" << kv_q8_flush_tokens
                          << " max_generation_tokens=" << max_generation_tokens << '\n';
                runtime.mark_ready(std::filesystem::path(*model_path).filename().string());
#else
                static_cast<void>(qwen38::ModelManifest::load(*model_path));
                runtime.mark_failed("server was built without MLX/tokenizer inference support");
#endif
            } catch (const std::exception& error) {
                runtime.mark_failed(error.what());
                std::cerr << "model load failed: " << error.what() << '\n';
            }
        }
        const qwen38::Api api(runtime, engine.get());
        qwen38::HttpServer server(config, api);
        active_server = &server;
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        server.run();
        active_server = nullptr;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-server: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
