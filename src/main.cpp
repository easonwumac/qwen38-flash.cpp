#include "qwen38/api.hpp"
#include "qwen38/http_server.hpp"
#include "qwen38/model_manifest.hpp"
#include "qwen38/runtime.hpp"
#ifdef QWEN38_HAS_INFERENCE
#include "qwen38/native_engine.hpp"
#endif

#include <charconv>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

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
        << " [--profile safe|speed|latency]"
        << " [--prefill-chunk 1..512] [--prefix-cache-tokens N]"
        << " [--max-generation-tokens N]"
        << " [--mtp-depth auto|off|2|3|4]\n"
        << "\n"
        << "qwen38-flash.cpp native inference server.\n";
}

void set_environment_default(const char* name, const char* value) {
    if (setenv(name, value, 0) != 0) {
        throw std::runtime_error(std::string("cannot set speed profile option ") + name);
    }
}

void apply_profile(const std::string& profile) {
    if (profile == "safe") return;
    if (profile != "speed" && profile != "latency") {
        throw std::runtime_error("invalid profile: " + profile);
    }
    const char* resident_range = profile == "latency" ? "12:34" : "12:28";
    const std::pair<const char*, const char*> settings[]{
        {"QWEN38_RESIDENT_EXPERT_RANGE", resident_range},
        {"QWEN38_FUSED_MOE", "1"},
        {"QWEN38_DEVICE_ROUTER", "1"},
        {"QWEN38_COMPILE_LAYER", "1"},
        {"QWEN38_HC_FUSED", "1"},
        {"QWEN38_HC_FUSED_INJECTION", "1"},
        {"QWEN38_GDN_NORM_GATE", "1"},
        {"QWEN38_GDN_PREWORK", "1"},
        {"QWEN38_GDN_METAL_VERIFY_BF16_SUM", "1"},
        {"QWEN38_BATCH_KV_VERIFY", "1"},
        {"QWEN38_SDPA_PREFILL", "1"},
        {"QWEN38_GDN_METAL_PREFILL", "1"},
        {"QWEN38_GROUPED_PREFILL", "1"},
        {"QWEN38_PREFILL_BARRIER_STRIDE", "8"},
        {"QWEN38_SELECTED_SOFTMAX_ROUTER", "1"},
        {"QWEN38_MTP_EARLY_DEMOTION", "1"},
        {"QWEN38_MTP_DEMOTION", "1"},
        {"QWEN38_Q8_EXACT_MOE", "1"},
        {"QWEN38_MTP_CUMULATIVE_PROFITABILITY_CACHE", "1"},
        {"QWEN38_EXTEND_PREFIX_CACHE", "1"},
    };
    for (const auto& [name, value] : settings) set_environment_default(name, value);
}

std::optional<std::size_t> parse_mtp_depth(const std::string& value) {
    if (value == "auto") return std::nullopt;
    if (value == "off" || value == "0") return 0;
    if (value == "2") return 2;
    if (value == "3") return 3;
    if (value == "4") return 4;
    throw std::runtime_error("invalid MTP depth: " + value);
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
        parsed == 0 || parsed > 512) {
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

} // namespace

int main(int argc, char** argv) {
    try {
        qwen38::ServerConfig config;
        std::string profile = "safe";
        std::optional<std::size_t> mtp_depth;
        std::size_t prefill_chunk_rows = 64;
        bool prefill_chunk_explicit = false;
        std::size_t prefix_cache_max_tokens = 8192;
        std::size_t max_generation_tokens = 4096;
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
                 argument == "--max-generation-tokens" || argument == "--profile") &&
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
            } else if (argument == "--prefill-chunk") {
                prefill_chunk_rows = parse_prefill_chunk(argv[++i]);
                prefill_chunk_explicit = true;
            } else if (argument == "--prefix-cache-tokens") {
                prefix_cache_max_tokens = parse_size(argv[++i], "prefix cache token limit");
            } else if (argument == "--max-generation-tokens") {
                max_generation_tokens = parse_size(argv[++i], "generation token limit");
                if (max_generation_tokens == 0) {
                    throw std::runtime_error("generation token limit must be positive");
                }
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }
        apply_profile(profile);
        if ((profile == "speed" || profile == "latency") &&
            !prefill_chunk_explicit) {
            prefill_chunk_rows = 512;
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
                engine_options.prefix_cache_max_tokens = prefix_cache_max_tokens;
                engine = std::make_unique<qwen38::NativeEngineExecutor>(
                    *model_path, engine_options);
                std::clog << "qwen38-server: profile=" << profile
                          << " prefill_chunk=" << prefill_chunk_rows
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
