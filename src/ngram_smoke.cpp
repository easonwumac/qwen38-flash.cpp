#include "qwen38/model_manifest.hpp"
#include "qwen38/ngram.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " MODEL_DIRECTORY\n";
        return EXIT_FAILURE;
    }
    try {
        const qwen38::ModelManifest manifest = qwen38::ModelManifest::load(argv[1]);
        qwen38::NgramHash hash(manifest.config());
        qwen38::NgramTable aos(manifest.directory(), hash.total_rows(), true);
        qwen38::NgramTable fallback(manifest.directory(), hash.total_rows(), false);
        if (!aos.uses_aos() || fallback.uses_aos()) {
            throw std::runtime_error("n-gram table layout selection failed");
        }
        qwen38::NgramState state;
        const auto first_rows = hash.row_ids(9419, state);
        const auto started = std::chrono::steady_clock::now();
        const auto first = aos.gather(first_rows);
        const double first_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        const auto reference = fallback.gather(first_rows);
        if (first != reference || first.size() != manifest.config().ple_embedding_dimension ||
            !std::all_of(first.begin(), first.end(), [](const float value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error("AoS and safetensors n-gram rows disagree");
        }
        const auto second_rows = hash.row_ids(11, state);
        const auto second_started = std::chrono::steady_clock::now();
        const auto second = aos.gather(second_rows);
        const double second_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - second_started).count();
        const double first_checksum = std::accumulate(first.begin(), first.end(), 0.0);
        const double second_checksum = std::accumulate(second.begin(), second.end(), 0.0);
        std::cout << "{\"first_row\":" << first_rows.front()
                  << ",\"last_row\":" << first_rows.back()
                  << ",\"first_checksum\":" << first_checksum
                  << ",\"second_checksum\":" << second_checksum
                  << ",\"first_ms\":" << first_ms
                  << ",\"second_ms\":" << second_ms << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen38-ngram-smoke: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
