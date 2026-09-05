#include "qwen38/prefix_cache_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace qwen38 {
namespace {

constexpr std::array<char, 8> token_magic{'Q', '3', '8', 'P', 'F', 'X', '1', '\0'};
constexpr std::size_t token_header_bytes = token_magic.size() + sizeof(std::uint64_t);
std::atomic<std::uint64_t> temporary_counter{0};

std::uint64_t fnv1a_tokens(const std::span<const std::uint32_t> tokens) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const std::uint32_t token : tokens) {
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(token >> shift);
            hash *= 1099511628211ULL;
        }
    }
    hash ^= tokens.size();
    hash *= 1099511628211ULL;
    return hash;
}

std::string cache_key(const std::span<const std::uint32_t> tokens) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << fnv1a_tokens(tokens)
           << '-' << std::dec << tokens.size();
    return output.str();
}

std::filesystem::path temporary_path(
    const std::filesystem::path& destination,
    const char* suffix) {
    return destination.string() + ".tmp-" +
        std::to_string(temporary_counter.fetch_add(1, std::memory_order_relaxed)) + suffix;
}

void write_tokens(
    const std::filesystem::path& path,
    const std::span<const std::uint32_t> tokens) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create prefix token file: " + path.string());
    output.write(token_magic.data(), static_cast<std::streamsize>(token_magic.size()));
    const std::uint64_t count = tokens.size();
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(
        reinterpret_cast<const char*>(tokens.data()),
        static_cast<std::streamsize>(tokens.size_bytes()));
    output.flush();
    if (!output) throw std::runtime_error("cannot write prefix token file: " + path.string());
}

std::optional<std::vector<std::uint32_t>> read_tokens(
    const std::filesystem::path& path,
    const std::size_t maximum_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    std::array<char, 8> magic{};
    std::uint64_t count = 0;
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!input || magic != token_magic || count > maximum_count ||
        count > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    const std::uintmax_t expected = token_header_bytes + count * sizeof(std::uint32_t);
    std::error_code error;
    if (std::filesystem::file_size(path, error) != expected || error) return std::nullopt;
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(count));
    input.read(
        reinterpret_cast<char*>(tokens.data()),
        static_cast<std::streamsize>(tokens.size() * sizeof(std::uint32_t)));
    if (!input) return std::nullopt;
    return tokens;
}

bool recognized_cache_file(const std::filesystem::path& path) {
    return path.extension() == ".tokens" || path.extension() == ".safetensors" ||
        path.string().find(".tmp-") != std::string::npos;
}

} // namespace

PrefixCacheStore::PrefixCacheStore(
    std::filesystem::path directory,
    const std::uint64_t max_bytes,
    const std::size_t model_layer_count)
    : directory_(std::move(directory)),
      max_bytes_(max_bytes),
      model_layer_count_(model_layer_count) {
    if (directory_.empty()) throw std::runtime_error("prefix cache directory is empty");
    if (max_bytes_ == 0) throw std::runtime_error("prefix cache size must be positive");
    if (model_layer_count_ == 0) throw std::runtime_error("model layer count must be positive");
    std::filesystem::create_directories(directory_);
}

std::optional<StoredPrefixState> PrefixCacheStore::load_longest(
    const std::span<const std::uint32_t> prompt_tokens) {
    std::vector<std::uint32_t> best_tokens;
    std::filesystem::path best_state_path;
    std::filesystem::path best_token_path;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(directory_)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".tokens") continue;
        const std::filesystem::path state_path =
            entry.path().parent_path() / (entry.path().stem().string() + ".safetensors");
        if (!std::filesystem::is_regular_file(state_path)) continue;
        std::optional<std::vector<std::uint32_t>> candidate =
            read_tokens(entry.path(), prompt_tokens.size());
        if (!candidate.has_value() || candidate->size() <= best_tokens.size() ||
            !std::equal(candidate->begin(), candidate->end(), prompt_tokens.begin())) {
            continue;
        }
        best_tokens = std::move(*candidate);
        best_state_path = state_path;
        best_token_path = entry.path();
    }
    if (best_tokens.empty()) return std::nullopt;
    try {
        PersistedPrefixState state =
            load_prefix_state(best_state_path, model_layer_count_);
        if (state.target.token_count != best_tokens.size()) return std::nullopt;
        const auto now = std::filesystem::file_time_type::clock::now();
        std::error_code ignored;
        std::filesystem::last_write_time(best_token_path, now, ignored);
        std::filesystem::last_write_time(best_state_path, now, ignored);
        return StoredPrefixState{.tokens = std::move(best_tokens), .state = std::move(state)};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

bool PrefixCacheStore::save(
    const std::span<const std::uint32_t> tokens,
    const PersistedPrefixState& state) {
    if (tokens.empty()) return false;
    if (state.target.token_count != tokens.size() ||
        state.target.layers.size() != model_layer_count_) {
        throw std::runtime_error("prefix token/state timeline mismatch");
    }
    const std::string key = cache_key(tokens);
    const std::filesystem::path token_path = directory_ / (key + ".tokens");
    const std::filesystem::path state_path = directory_ / (key + ".safetensors");
    const std::filesystem::path temporary_tokens = temporary_path(token_path, ".tokens");
    const std::filesystem::path temporary_state = temporary_path(state_path, ".safetensors");
    try {
        save_prefix_state(temporary_state, state);
        write_tokens(temporary_tokens, tokens);
        std::filesystem::rename(temporary_state, state_path);
        std::filesystem::rename(temporary_tokens, token_path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_tokens, ignored);
        std::filesystem::remove(temporary_state, ignored);
        throw;
    }
    evict_to_limit();
    std::error_code token_error;
    std::error_code state_error;
    return std::filesystem::is_regular_file(token_path, token_error) && !token_error &&
        std::filesystem::is_regular_file(state_path, state_error) && !state_error;
}

void PrefixCacheStore::clear() {
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(directory_)) {
        if (!entry.is_regular_file() || !recognized_cache_file(entry.path())) continue;
        std::error_code ignored;
        std::filesystem::remove(entry.path(), ignored);
    }
}

void PrefixCacheStore::evict_to_limit() {
    struct Entry {
        std::filesystem::path token_path;
        std::filesystem::path state_path;
        std::filesystem::file_time_type modified;
        std::uintmax_t bytes{0};
    };
    std::vector<Entry> entries;
    std::uintmax_t total = 0;
    for (const std::filesystem::directory_entry& token_entry :
         std::filesystem::directory_iterator(directory_)) {
        if (!token_entry.is_regular_file() || token_entry.path().extension() != ".tokens") {
            continue;
        }
        const std::filesystem::path state_path = token_entry.path().parent_path() /
            (token_entry.path().stem().string() + ".safetensors");
        std::error_code error;
        if (!std::filesystem::is_regular_file(state_path, error) || error) continue;
        const std::uintmax_t token_bytes = token_entry.file_size(error);
        if (error) continue;
        const std::uintmax_t state_bytes = std::filesystem::file_size(state_path, error);
        if (error) continue;
        const std::uintmax_t bytes = token_bytes + state_bytes;
        entries.push_back({
            .token_path = token_entry.path(),
            .state_path = state_path,
            .modified = token_entry.last_write_time(),
            .bytes = bytes,
        });
        total += bytes;
    }
    std::ranges::sort(entries, {}, &Entry::modified);
    for (const Entry& entry : entries) {
        if (total <= max_bytes_) break;
        std::error_code ignored;
        std::filesystem::remove(entry.token_path, ignored);
        std::filesystem::remove(entry.state_path, ignored);
        total -= entry.bytes;
    }
}

} // namespace qwen38
