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
        << " [--host IPv4] [--port PORT] [--model PATH] [--mtp-depth auto|off|2|3|4]\n"
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

std::uint16_t parse_port(const std::string& value) {
    unsigned int parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed == 0 || parsed > 65535) {
        throw std::runtime_error("invalid port: " + value);
    }
    return static_cast<std::uint16_t>(parsed);
}

} // namespace

int main(int argc, char** argv) {
    try {
        qwen38::ServerConfig config;
        std::optional<std::size_t> mtp_depth;
        std::optional<std::string> model_path;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
            if ((argument == "--host" || argument == "--port" || argument == "--model" ||
                 argument == "--mtp-depth") &&
                i + 1 >= argc) {
                throw std::runtime_error("missing value for " + argument);
            }
            if (argument == "--host") {
                config.host = argv[++i];
            } else if (argument == "--port") {
                config.port = parse_port(argv[++i]);
            } else if (argument == "--model") {
                model_path = argv[++i];
            } else if (argument == "--mtp-depth") {
                mtp_depth = parse_mtp_depth(argv[++i]);
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }

        qwen38::RuntimeState runtime;
        std::unique_ptr<qwen38::InferenceEngine> engine;
        if (model_path.has_value()) {
            runtime.begin_loading(*model_path);
            try {
#ifdef QWEN38_HAS_INFERENCE
                qwen38::NativeEngineOptions engine_options;
                engine_options.mtp_depth = mtp_depth;
                engine = std::make_unique<qwen38::NativeEngine>(
                    *model_path, engine_options);
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
